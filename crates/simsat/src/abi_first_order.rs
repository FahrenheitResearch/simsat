//! End-to-end spectral FIRST-ORDER reference, not the finished ABI operator.
//!
//! Actual model-grid satellite rays cross the existing unscaled 3-D extinction
//! field. Dry-air Rayleigh scattering and measured spectral Lambertian land feed
//! the exact first-order segment integrator, then NOAA FM4 SRFs and TSIS HSRS.
//! Gray conservative cloud extinction/dual-HG phase is an explicit interim
//! assumption. This does not silently apply display opacity or illumination.
//! Multiple scattering, gas absorption, aerosols, native atmospheric profiles,
//! spectral cloud particles, diffuse ocean/surface illumination, terrain casting,
//! fractional-cloud overlap, finite solar disk and instrument PSF remain missing.
//!
//! CPU scene reference. Its first-order transport and Rayleigh/dual-HG math use
//! the existing independently tested CPU/WGSL counterparts; there is no GPU
//! scene implementation or GPU-preview fallback for this product.
use crate::{
    atmosphere::CameraGeometry,
    bricks::VolumeBrick,
    camera::SurfaceRaster,
    clouds::{self, DecodedVolume, OccupancyMip},
    frame::GridGeoref,
    optics,
    spectral_molecular::{DryAirRayleigh, dry_air_number_density_m3},
    spectral_surface::SpectralSurface,
    spectral_transport::{SingleScatterSegment, SolarDepthEndpoints, integrate_single_scatter},
    visible_sensor::AbiReflectiveBand,
    visible_solar::AbiSolarResponse,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    f64::consts::PI,
    sync::atomic::{AtomicUsize, Ordering},
};

pub(crate) const R: f64 = optics::EARTH_RADIUS_M;
pub(crate) const TOP: f64 = R + crate::atmosphere::ATMOSPHERE_HEIGHT_M;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FirstOrderConfig {
    /// Explicit reference atmosphere, NOT claimed to be model or observed air.
    pub dry_pressure_pa: f64,
    pub reference_temperature_k: f64,
    pub dry_scale_height_m: f64,
    pub co2_ppm: f64,
    pub spectral_step_um: f64,
    /// Numerical ray steps must be at least 1 m, preventing sub-ULP progress
    /// at geostationary distances. Sub-metre cloud detail is not resolved here.
    pub view_step_m: f64,
    pub sun_step_m: f64,
    pub air_step_m: f64,
    /// Simpson intervals for spherical molecular solar columns; must be even.
    pub air_column_intervals: usize,
    /// 1 computes every model-grid sample. Larger values leave NaN holes and
    /// preserve the original grid; they NEVER masquerade as native full images.
    pub sample_stride: usize,
}
impl FirstOrderConfig {
    pub fn validate(&self) -> Result<(), String> {
        dry_air_number_density_m3(self.dry_pressure_pa, self.reference_temperature_k)
            .map_err(|e| e.to_string())?;
        DryAirRayleigh::new(0.5, self.co2_ppm).map_err(|e| e.to_string())?;
        if !self.dry_scale_height_m.is_finite()
            || self.dry_scale_height_m <= 0.0
            || !self.spectral_step_um.is_finite()
            || !(0.001..=0.02).contains(&self.spectral_step_um)
            || [self.view_step_m, self.sun_step_m, self.air_step_m]
                .iter()
                .any(|v| !v.is_finite() || *v < 1.0)
            || self.air_column_intervals < 8
            || self.air_column_intervals > 4096
            || !self.air_column_intervals.is_multiple_of(2)
            || self.sample_stride == 0
        {
            return Err("invalid explicit first-order numerical/atmosphere configuration".into());
        }
        Ok(())
    }
}
#[derive(Debug)]
pub struct FirstOrderFrame {
    pub nx: usize,
    pub ny: usize,
    /// North-first interleaved C01/C02/C03 reflectance factors pi*L/E.
    pub reflectance: Vec<f32>,
    pub scattered_reflectance: Vec<f32>,
    pub surface_reflectance: Vec<f32>,
    /// 0 unsampled; bit 0 sampled, bit 1 view ray outside cloud domain,
    /// bit 2 solar ray outside cloud domain. Exterior cloud is explicitly zero.
    pub support_flags: Vec<u8>,
    pub glint_mask: Vec<u8>,
    pub wavelength_um: Vec<f64>,
    pub active_wavelengths: usize,
}

#[derive(Clone)]
struct SpectralPlan {
    wavelength: Vec<f64>,
    weights: Vec<[f64; 3]>,
    molecular: Vec<DryAirRayleigh>,
}
impl SpectralPlan {
    fn new(cfg: &FirstOrderConfig) -> Result<Self, String> {
        let intervals = (0.6 / cfg.spectral_step_um).ceil() as usize;
        let wavelength: Vec<_> = (0..=intervals)
            .map(|i| 0.4 + 0.6 * i as f64 / intervals as f64)
            .collect();
        let mut weights = vec![[0.0; 3]; wavelength.len()];
        for (b, band) in AbiReflectiveBand::ALL.into_iter().enumerate() {
            let response = AbiSolarResponse::for_band(band);
            for node in response.nodes() {
                let x = node.wavelength_um;
                if x < wavelength[0] || x > *wavelength.last().unwrap() {
                    return Err("spectral transfer grid does not cover official SRF".into());
                }
                let hi = wavelength
                    .partition_point(|v| *v < x)
                    .clamp(1, wavelength.len() - 1);
                let f = (x - wavelength[hi - 1]) / (wavelength[hi] - wavelength[hi - 1]);
                let w = PI * node.solar_response_weight_w_m2
                    / response.solar_response_integral_1au_w_m2();
                weights[hi - 1][b] += w * (1.0 - f);
                weights[hi][b] += w * f;
            }
        }
        let molecular = wavelength
            .iter()
            .map(|&w| DryAirRayleigh::new(w, cfg.co2_ppm).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            wavelength,
            weights,
            molecular,
        })
    }
}

#[derive(Clone, Copy)]
struct SunDepth {
    gray: f64,
    dry: f64,
    exterior: bool,
}
struct Segment {
    gray: f64,
    dry: f64,
    phase_gray: f64,
    near: Option<SunDepth>,
    far: Option<SunDepth>,
}
struct PreparedRay {
    segments: Vec<Segment>,
    surface_sun: Option<SunDepth>,
    mu0: f64,
    cosine: f64,
    water_brf: f64,
    flags: u8,
    glint_core: bool,
}

pub(crate) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub(crate) fn add(p: [f64; 3], v: [f64; 3], t: f64) -> [f64; 3] {
    [p[0] + v[0] * t, p[1] + v[1] * t, p[2] + v[2] * t]
}
fn unit(p: [f64; 3]) -> [f64; 3] {
    let r = dot(p, p).sqrt();
    p.map(|v| v / r)
}
pub(crate) fn sphere(p: [f64; 3], v: [f64; 3], radius: f64) -> Option<(f64, f64)> {
    let b = dot(p, v);
    let d = b * b - dot(p, p) + radius * radius;
    (d >= 0.0).then(|| (-b - d.sqrt(), -b + d.sqrt()))
}
pub(crate) fn dry_density(p: [f64; 3], n0: f64, h: f64) -> f64 {
    let z = dot(p, p).sqrt() - R;
    if !(0.0..=TOP - R).contains(&z) {
        0.0
    } else {
        n0 * (-z / h).exp()
    }
}
pub(crate) fn dry_column(p: [f64; 3], v: [f64; 3], n0: f64, cfg: &FirstOrderConfig) -> Option<f64> {
    // A point Sun is blocked by solid Earth; no hidden twilight fill.
    if sphere(p, v, R)
        .is_some_and(|(a, b)| a > 1.0e-5 || (a >= -1.0e-5 && b > 1.0e-5 && dot(p, v) < 0.0))
    {
        return None;
    }
    let (_, exit) = sphere(p, v, TOP)?;
    if exit <= 0.0 {
        return Some(0.0);
    }
    let count = cfg.air_column_intervals;
    let ds = exit / count as f64;
    let mut sum = 0.0;
    for i in 0..=count {
        let weight = if i == 0 || i == count {
            1.0
        } else if i % 2 == 0 {
            2.0
        } else {
            4.0
        };
        sum += weight * dry_density(add(p, v, i as f64 * ds), n0, cfg.dry_scale_height_m);
    }
    Some(sum * ds / 3.0)
}
pub(crate) fn outside(fi: f64, fj: f64, vol: &DecodedVolume) -> bool {
    fi < 0.0 || fj < 0.0 || fi > (vol.nx - 1) as f64 || fj > (vol.ny - 1) as f64
}
pub(crate) fn cloud_sun_depth(
    vol: &DecodedVolume,
    mip: &OccupancyMip,
    georef: &GridGeoref,
    p: [f64; 3],
    sun: [f64; 3],
    step: f64,
) -> (f64, bool) {
    if vol.nz < 2 {
        return (0.0, false);
    }
    let Some((a, b)) = clouds::ray_shell_segment(p, sun, vol.r_bottom(), vol.r_top() - vol.dz_m)
    else {
        return (0.0, false);
    };
    let fine = step.min(vol.voxel_pitch_m());
    let coarse = vol.voxel_pitch_m() * (mip.factor as f64 * 0.5).clamp(1.0, 8.0);
    let (mut t, mut tau, mut exterior) = (a, 0.0, false);
    while t < b {
        let point = add(p, sun, t);
        let (fi, fj, fk, _) = clouds::ecef_to_brick(point, georef, vol.z_min_m, vol.dz_m);
        exterior |= outside(fi, fj, vol);
        let occupied = mip.maxext_at(fi, fj, fk) > 0.0;
        let ds = (if occupied { fine } else { coarse }).min(b - t);
        if ds <= 0.0 {
            break;
        }
        if occupied {
            let (i, j, k, _) =
                clouds::ecef_to_brick(add(p, sun, t + ds * 0.5), georef, vol.z_min_m, vol.dz_m);
            tau += vol.sample(i, j, k).total_ext() * ds;
        }
        t += ds;
    }
    (tau, exterior)
}
struct Scene<'a> {
    vol: &'a DecodedVolume,
    mip: &'a OccupancyMip,
    georef: &'a GridGeoref,
    sun: [f64; 3],
    n0: f64,
    cfg: &'a FirstOrderConfig,
}
impl Scene<'_> {
    fn sun_depth(&self, p: [f64; 3]) -> Option<SunDepth> {
        let dry = dry_column(p, self.sun, self.n0, self.cfg)?;
        let (gray, exterior) = cloud_sun_depth(
            self.vol,
            self.mip,
            self.georef,
            p,
            self.sun,
            self.cfg.sun_step_m,
        );
        Some(SunDepth {
            gray,
            dry,
            exterior,
        })
    }
    fn prepare_ray(
        &self,
        cam: [f64; 3],
        view: [f64; 3],
        height: f64,
        wind: f64,
    ) -> Result<PreparedRay, String> {
        let (enter, _) = sphere(cam, view, TOP).ok_or("view misses atmosphere")?;
        let (end, _) =
            sphere(cam, view, R + height.max(0.0)).ok_or("view misses target-height surface")?;
        if end <= enter {
            return Err("invalid atmosphere/surface ray ordering".into());
        }
        let surface = add(cam, view, end);
        let up = unit(surface);
        let mu0 = dot(up, self.sun).max(0.0);
        // The existing kernel is pi*BRDF. Multiplying by mu0 below gives pi*L/E,
        // matching the independent hemispheric Fresnel integration in optics.rs.
        let water_brf = optics::cox_munk_glint_reflectance(
            self.sun,
            view.map(|v| -v),
            up,
            optics::cox_munk_mean_square_slope(wind),
        );
        let facet = unit([
            self.sun[0] - view[0],
            self.sun[1] - view[1],
            self.sun[2] - view[2],
        ]);
        let cb = dot(facet, up);
        let glint_core = mu0 > 0.0
            && cb > 0.0
            && (1.0 - cb * cb).max(0.0) / (cb * cb) <= optics::cox_munk_mean_square_slope(wind);
        let cosine = dot(view, self.sun).clamp(-1.0, 1.0);
        let phase_l = clouds::phase_liquid(cosine);
        let phase_i = clouds::phase_ice(cosine);
        let (mut t, mut flags) = (enter.max(0.0), 1u8);
        let mut near = self.sun_depth(add(cam, view, t));
        let mut segments = Vec::new();
        while t < end {
            let r = dot(add(cam, view, t), add(cam, view, t)).sqrt();
            // Resolve the cloud shell at the supplied fine step, and split at
            // its top. Molecular air above it uses its separately reported step.
            let step = if r > self.vol.r_top() + self.cfg.air_step_m {
                self.cfg.air_step_m
            } else {
                self.cfg.view_step_m.min(self.vol.voxel_pitch_m())
            };
            let mut ds = step.min(end - t);
            let mut far = self.sun_depth(add(cam, view, t + ds));
            // Split point-Sun Earth-shadow crossings instead of illuminating an
            // entire thick segment from one lit endpoint. Residual span <=1 m.
            while near.is_some() != far.is_some() && ds > 1.0 {
                ds *= 0.5;
                far = self.sun_depth(add(cam, view, t + ds));
            }
            if ds <= 0.0 {
                break;
            }
            let mid = add(cam, view, t + ds * 0.5);
            let (i, j, k, rm) =
                clouds::ecef_to_brick(mid, self.georef, self.vol.z_min_m, self.vol.dz_m);
            if rm <= self.vol.r_top() && rm >= self.vol.r_bottom() && outside(i, j, self.vol) {
                flags |= 2;
            }
            let cloud = self.vol.sample(i, j, k);
            let gray = cloud.total_ext() * ds;
            let phase_gray =
                (cloud.ext_liquid * phase_l + (cloud.ext_ice + cloud.ext_precip) * phase_i) * ds;
            let dry = ds / 6.0
                * (dry_density(add(cam, view, t), self.n0, self.cfg.dry_scale_height_m)
                    + 4.0 * dry_density(mid, self.n0, self.cfg.dry_scale_height_m)
                    + dry_density(add(cam, view, t + ds), self.n0, self.cfg.dry_scale_height_m));
            if near.is_some_and(|s| s.exterior) || far.is_some_and(|s| s.exterior) {
                flags |= 4;
            }
            let (lit_near, lit_far) = if near.is_some() != far.is_some() {
                let m = self.sun_depth(mid);
                (near.or(m), far.or(m))
            } else {
                (near, far)
            };
            segments.push(Segment {
                gray,
                dry,
                phase_gray,
                near: lit_near,
                far: lit_far,
            });
            near = far;
            t += ds;
        }
        Ok(PreparedRay {
            segments,
            surface_sun: near,
            mu0,
            cosine,
            water_brf,
            flags,
            glint_core,
        })
    }
}
fn evaluate(
    ray: &PreparedRay,
    molecular: DryAirRayleigh,
    surface_brf: f64,
) -> Result<(f64, f64), String> {
    let sigma = molecular.cross_section_m2();
    let phase = molecular.phase_sr1(ray.cosine).map_err(|e| e.to_string())?;
    let segments = ray
        .segments
        .iter()
        .map(|s| {
            let tau = s.gray + s.dry * sigma;
            let beta_phase = s.phase_gray + s.dry * sigma * phase;
            let solar = match (s.near, s.far) {
                (Some(a), Some(b)) => Some(
                    SolarDepthEndpoints::new(a.gray + a.dry * sigma, b.gray + b.dry * sigma)
                        .map_err(|e| e.to_string())?,
                ),
                _ => None,
            };
            SingleScatterSegment::new(
                tau,
                tau,
                if tau > 0.0 { beta_phase / tau } else { 0.0 },
                solar,
            )
            .map_err(|e| e.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let result = integrate_single_scatter(segments, None).map_err(|e| e.to_string())?;
    let surface = ray.surface_sun.map_or(0.0, |s| {
        surface_brf * ray.mu0 / PI * (-s.gray - s.dry * sigma).exp() * result.view_transmittance
    });
    if !surface.is_finite() || surface < 0.0 {
        return Err("invalid first-order boundary result".into());
    }
    Ok((result.scattered_normalized_radiance_sr1, surface))
}

/// Returns genuine response-integrated TOA first-order reflectance factors.
/// Spectral linear interpolation applies to computed L_lambda/E_lambda, not to
/// the three legacy RGB channels. Convergence of this grid must be measured.
#[allow(clippy::too_many_arguments)]
pub fn render(
    brick: &VolumeBrick,
    georef: &GridGeoref,
    raster: &SurfaceRaster,
    camera: &CameraGeometry,
    sun: [f64; 3],
    surface: &SpectralSurface,
    horizontal_pitch_m: f64,
    cfg: &FirstOrderConfig,
) -> Result<FirstOrderFrame, String> {
    cfg.validate()?;
    let plan = SpectralPlan::new(cfg)?;
    // Validate coordinate identity to the model and the complete land spectra
    // using the established spectral-source contract, including model LANDMASK.
    surface.raster_rgba(brick.nx, brick.ny, &brick.landmask, raster, |la, lo| {
        georef.forward(la, lo)
    })?;
    let mut source_cells = vec![usize::MAX; brick.nx * brick.ny];
    for (cell, &[lat, lon]) in surface.coordinates().iter().enumerate() {
        let (i, j) = georef.forward(lat, lon);
        source_cells[j.round() as usize * brick.nx + i.round() as usize] = cell;
    }
    let vol = DecodedVolume::from_brick(brick, horizontal_pitch_m);
    let mip = OccupancyMip::build(&vol, clouds::OCCUPANCY_MIP_FACTOR);
    let scene = Scene {
        vol: &vol,
        mip: &mip,
        georef,
        sun,
        n0: dry_air_number_density_m3(cfg.dry_pressure_pa, cfg.reference_temperature_k)
            .map_err(|e| e.to_string())?,
        cfg,
    };
    let count = raster.nx * raster.ny;
    let completed = AtomicUsize::new(0);
    type PixelTerms = ([f32; 3], [f32; 3], u8);
    let rows = (0..raster.ny)
        .into_par_iter()
        .map(|y| -> Result<Vec<PixelTerms>, String> {
            let mut row = Vec::with_capacity(raster.nx);
            for x in 0..raster.nx {
                if !x.is_multiple_of(cfg.sample_stride) || !y.is_multiple_of(cfg.sample_stride) {
                    row.push(([f32::NAN; 3], [f32::NAN; 3], 0));
                    continue;
                }
                let p = y * raster.nx + x;
                let i = (raster.grid_i[p] as f64)
                    .round()
                    .clamp(0.0, (brick.nx - 1) as f64) as usize;
                let j = (raster.grid_j[p] as f64)
                    .round()
                    .clamp(0.0, (brick.ny - 1) as f64) as usize;
                let cell = j * brick.nx + i;
                let source = source_cells[cell];
                if source == usize::MAX {
                    return Err("unassigned spectral land coordinate".into());
                }
                let (sx, sy) = raster.model_scan_angle(x, y);
                let view = camera.view_dir(sx, sy);
                let wind = (brick.u10[cell] as f64).hypot(brick.v10[cell] as f64);
                let ray = scene.prepare_ray(camera.camera, view, brick.hgt[cell] as f64, wind)?;
                let (mut scatter, mut ground) = ([0.0; 3], [0.0; 3]);
                for (node, &w) in plan.weights.iter().enumerate() {
                    if w.iter().all(|v| *v == 0.0) {
                        continue;
                    }
                    let a = if surface.is_land(source) {
                        surface
                            .sample(source, plan.wavelength[node])
                            .ok_or("incomplete spectral boundary")?
                    } else {
                        ray.water_brf
                    };
                    let (s, g) = evaluate(&ray, plan.molecular[node], a)?;
                    for b in 0..3 {
                        scatter[b] += w[b] * s;
                        ground[b] += w[b] * g;
                    }
                }
                row.push((
                    scatter.map(|v| v as f32),
                    ground.map(|v| v as f32),
                    ray.flags
                        | if !surface.is_land(source) && ray.glint_core {
                            8
                        } else {
                            0
                        },
                ));
            }
            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if done.is_multiple_of(32) || done == raster.ny {
                crate::log_line!("ABI first-order: {done}/{} rows complete", raster.ny);
            }
            Ok(row)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut frame = FirstOrderFrame {
        nx: raster.nx,
        ny: raster.ny,
        reflectance: Vec::with_capacity(count * 3),
        scattered_reflectance: Vec::with_capacity(count * 3),
        surface_reflectance: Vec::with_capacity(count * 3),
        support_flags: Vec::with_capacity(count),
        glint_mask: Vec::with_capacity(count),
        wavelength_um: plan.wavelength,
        active_wavelengths: plan
            .weights
            .iter()
            .filter(|w| w.iter().any(|v| *v > 0.0))
            .count(),
    };
    for (s, g, flags) in rows.into_iter().flatten() {
        frame.reflectance.extend((0..3).map(|i| s[i] + g[i]));
        frame.scattered_reflectance.extend(s);
        frame.surface_reflectance.extend(g);
        frame.support_flags.push(flags & 7);
        frame.glint_mask.push(u8::from(flags & 8 != 0));
    }
    Ok(frame)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cfg() -> FirstOrderConfig {
        FirstOrderConfig {
            dry_pressure_pa: 101325.0,
            reference_temperature_k: 288.15,
            dry_scale_height_m: 8000.0,
            co2_ppm: 360.0,
            spectral_step_um: 0.01,
            view_step_m: 125.0,
            sun_step_m: 125.0,
            air_step_m: 1000.0,
            air_column_intervals: 128,
            sample_stride: 1,
        }
    }
    #[test]
    fn satellite_ray_through_known_3d_slab_matches_closed_form_and_night_shadow() {
        // Independently known 10-km slab, zenith Sun and nadir satellite. This
        // exercises actual sphere intersections, georeferencing, occupancy,
        // volume samples and both ray paths, rather than supplying tau directly.
        let (nx, ny, nz) = (5, 5, 41);
        let n = nx * ny * nz;
        let beta = 1e-4_f32;
        let vol = DecodedVolume {
            nx,
            ny,
            nz,
            z_min_m: 0.0,
            dz_m: 250.0,
            horiz_pitch_m: 100000.0,
            ext_liquid: vec![beta; n],
            ext_ice: vec![0.0; n],
            ext_precip: vec![0.0; n],
            tau_up: vec![0.0; n],
            ext_snow: vec![0; n],
            ext_snow_quant: crate::bricks::LogQuant {
                vmin: 0.0,
                vmax: 0.0,
            },
            science_ext_snow: Vec::new(),
            cloud_fraction: vec![255; n],
            has_cloud_fraction: false,
        };
        let mip = OccupancyMip::build(&vol, clouds::OCCUPANCY_MIP_FACTOR);
        let georef = GridGeoref::new(
            crate::frame::MapProjection::LatLon {
                central_meridian_deg: 0.0,
            },
            2.0,
            2.0,
            0.0,
            0.0,
            1.0,
            1.0,
        );
        let c = cfg();
        let mut scene = Scene {
            vol: &vol,
            mip: &mip,
            georef: &georef,
            sun: [1.0, 0.0, 0.0],
            n0: 0.0,
            cfg: &c,
        };
        let camera = [42164000.0, 0.0, 0.0];
        let view = [-1.0, 0.0, 0.0];
        let ray = scene.prepare_ray(camera, view, 0.0, 5.0).unwrap();
        let tau = beta as f64 * 10000.0;
        let view_tau: f64 = ray.segments.iter().map(|s| s.gray).sum();
        assert!((view_tau - tau).abs() < 1e-8);
        assert!((ray.surface_sun.unwrap().gray - tau).abs() < 1e-8);
        assert_eq!(ray.flags, 1);
        let optic = DryAirRayleigh::new(0.64, 360.0).unwrap();
        let (scatter, ground) = evaluate(&ray, optic, 0.2).unwrap();
        let expected = clouds::phase_liquid(-1.0) * 0.5 * -(-2.0 * tau).exp_m1();
        assert!((scatter - expected).abs() < 1e-10, "{scatter} {expected}");
        assert!((ground - 0.2 / PI * (-2.0 * tau).exp()).abs() < 1e-10);
        scene.sun = [-1.0, 0.0, 0.0];
        let night = scene.prepare_ray(camera, view, 0.0, 5.0).unwrap();
        assert_eq!(evaluate(&night, optic, 0.2).unwrap(), (0.0, 0.0));
    }
    #[test]
    fn invalid_numerics_and_unrecognized_configuration_are_rejected() {
        let c = cfg();
        c.validate().unwrap();
        for bad in [0.0, -1.0, f64::EPSILON, f64::NAN, f64::INFINITY] {
            for field in 0..3 {
                let mut q = c.clone();
                match field {
                    0 => q.view_step_m = bad,
                    1 => q.sun_step_m = bad,
                    _ => q.air_step_m = bad,
                }
                assert!(q.validate().is_err());
            }
        }
        let mut json = serde_json::to_value(c).unwrap();
        json.as_object_mut()
            .unwrap()
            .insert("exposure".into(), 1.5.into());
        assert!(serde_json::from_value::<FirstOrderConfig>(json).is_err());
    }
    #[test]
    fn curved_molecular_transfer_converges_against_full_response_quadrature() {
        // Analytic plane-parallel single-scatter air + attenuated direct land;
        // evaluates every official SRF/HSRS node for the independent integral.
        for (mu0, muv) in [(0.1, 0.3), (0.4, 0.8), (1.0, 1.0)] {
            let transfer = |w| {
                let optic = DryAirRayleigh::new(w, 360.0).unwrap();
                let tau = 2.5e25 * 8000.0 * optic.cross_section_m2();
                let path = tau * (1.0 / mu0 + 1.0 / muv);
                let phase = optic.phase_sr1(-0.45).unwrap();
                phase * mu0 / (mu0 + muv) * -(-path).exp_m1() + 0.2 * mu0 / PI * (-path).exp()
            };
            for (b, band) in AbiReflectiveBand::ALL.into_iter().enumerate() {
                let expected = AbiSolarResponse::for_band(band)
                    .reflectance_factor_from_normalized_radiance(transfer)
                    .unwrap();
                let mut errors = Vec::new();
                for step in [0.01, 0.005, 0.0025] {
                    let mut c = cfg();
                    c.spectral_step_um = step;
                    let p = SpectralPlan::new(&c).unwrap();
                    let got: f64 = p
                        .weights
                        .iter()
                        .zip(&p.wavelength)
                        .map(|(w, l)| w[b] * transfer(*l))
                        .sum();
                    errors.push((got - expected).abs());
                }
                assert!(errors[1] < errors[0], "{errors:?}");
                assert!(errors[2] < errors[1], "{errors:?}");
                assert!(errors[2] < 5e-5, "{errors:?}");
            }
        }
    }
    #[test]
    fn official_response_preserves_constant_and_affine_transfer() {
        let c = cfg();
        let p = SpectralPlan::new(&c).unwrap();
        for (b, band) in AbiReflectiveBand::ALL.into_iter().enumerate() {
            let w: f64 = p.weights.iter().map(|v| v[b]).sum();
            assert!((w - PI).abs() < 1e-12);
            let got: f64 = p
                .weights
                .iter()
                .zip(&p.wavelength)
                .map(|(w, l)| w[b] * (0.1 + 0.2 * l))
                .sum();
            let expected = AbiSolarResponse::for_band(band)
                .reflectance_factor_from_normalized_radiance(|l| 0.1 + 0.2 * l)
                .unwrap();
            assert!((got - expected).abs() < 1e-12);
        }
    }
    #[test]
    fn spherical_air_column_matches_vertical_analytic_and_night_shadow() {
        let c = cfg();
        let n = 2.5e25;
        for height in [0.1, 1000.0, 20000.0, 70000.0] {
            let got = dry_column([0.0, 0.0, R + height], [0.0, 0.0, 1.0], n, &c).unwrap();
            let expected = n
                * c.dry_scale_height_m
                * ((-height / c.dry_scale_height_m).exp()
                    - (-(TOP - R) / c.dry_scale_height_m).exp());
            assert!(
                (got / expected - 1.0).abs() < 2e-6,
                "{height} {got} {expected}"
            );
        }
        assert!(dry_column([0.0, 0.0, R + 1000.0], [0.0, 0.0, -1.0], n, &c).is_none());
    }
    #[test]
    fn mixed_cloud_molecular_slab_matches_analytic_first_order_and_absorbed_boundary() {
        let optic = DryAirRayleigh::new(0.64, 360.0).unwrap();
        for mu0 in [0.05, 0.4, 1.0] {
            for muv in [0.2, 0.8, 1.0] {
                let sigma = optic.cross_section_m2();
                let molecular_tau = 0.1;
                let cloud_tau = 0.7;
                let molecular_column = molecular_tau / sigma;
                let cloud_phase = 0.03;
                let cosine = -0.45;
                let mut ray = PreparedRay {
                    segments: vec![Segment {
                        gray: cloud_tau / muv,
                        dry: molecular_column / muv,
                        phase_gray: cloud_phase * cloud_tau / muv,
                        near: Some(SunDepth {
                            gray: 0.0,
                            dry: 0.0,
                            exterior: false,
                        }),
                        far: Some(SunDepth {
                            gray: cloud_tau / mu0,
                            dry: molecular_column / mu0,
                            exterior: false,
                        }),
                    }],
                    surface_sun: Some(SunDepth {
                        gray: cloud_tau / mu0,
                        dry: molecular_column / mu0,
                        exterior: false,
                    }),
                    mu0,
                    cosine,
                    water_brf: 0.0,
                    flags: 1,
                    glint_core: false,
                };
                let (s, g) = evaluate(&ray, optic, 0.2).unwrap();
                let tau = cloud_tau + molecular_tau;
                let phase = (cloud_tau * cloud_phase
                    + molecular_tau * optic.phase_sr1(cosine).unwrap())
                    / tau;
                let total_path = tau * (1.0 / mu0 + 1.0 / muv);
                let expected = phase * mu0 / (mu0 + muv) * (1.0 - (-total_path).exp());
                assert!((s - expected).abs() < 1e-14);
                assert!((g - 0.2 * mu0 / PI * (-total_path).exp()).abs() < 1e-14);
                ray.segments[0].near = None;
                ray.segments[0].far = None;
                ray.surface_sun = None;
                assert_eq!(evaluate(&ray, optic, 0.2).unwrap(), (0.0, 0.0));
            }
        }
    }

    #[test]
    fn boundary_uses_one_solar_cosine_and_no_display_exposure() {
        let ray = PreparedRay {
            segments: Vec::new(),
            surface_sun: Some(SunDepth {
                gray: 0.0,
                dry: 0.0,
                exterior: false,
            }),
            mu0: 0.25,
            cosine: -0.5,
            water_brf: 0.0,
            flags: 1,
            glint_core: false,
        };
        let (s, g) = evaluate(&ray, DryAirRayleigh::new(0.64, 360.0).unwrap(), 0.2).unwrap();
        assert_eq!(s, 0.0);
        assert!((PI * g - 0.05).abs() < 1e-15);
    }
}
