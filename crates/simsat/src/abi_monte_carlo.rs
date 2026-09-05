//! All-order scalar Monte Carlo ABI reference through the actual 3D model volume.
//! This adds repeated scattering and diffuse surface reflection to the explicit
//! gray-cloud/reference-air experiment. It is not yet the full sensor operator:
//! native profiles, gases/aerosols, spectral particles, terrain casting, finite
//! Sun, polarization, fractional overlap and instrument footprint remain absent.
use crate::{
    abi_first_order::{self, FirstOrderConfig},
    atmosphere::CameraGeometry,
    bricks::VolumeBrick,
    camera::SurfaceRaster,
    clouds::{self, DecodedVolume, OccupancyMip},
    frame::GridGeoref,
    optics,
    spectral_molecular::{DryAirRayleigh, dry_air_number_density_m3},
    spectral_path::{
        self, Event, EventWithSupport, Material, Moments, PathConfig, PhaseFunction, Random,
    },
    spectral_surface::SpectralSurface,
    visible_sensor::AbiReflectiveBand,
    visible_solar::AbiSolarResponse,
};
use abi_first_order::{
    R, TOP, add, cloud_sun_depth, dot, dry_column, dry_density, outside, sphere,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    f64::consts::PI,
    sync::atomic::{AtomicUsize, Ordering},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MonteCarloConfig {
    pub dry_pressure_pa: f64,
    pub reference_temperature_k: f64,
    pub dry_scale_height_m: f64,
    pub co2_ppm: f64,
    pub view_step_m: f64,
    pub sun_step_m: f64,
    pub air_step_m: f64,
    pub air_column_intervals: usize,
    pub collision_step_optical_depth: f64,
    pub sample_stride: usize,
    pub samples_per_band: usize,
    pub seed: u32,
    pub path: PathConfig,
}
impl MonteCarloConfig {
    fn quadrature(&self) -> FirstOrderConfig {
        FirstOrderConfig {
            dry_pressure_pa: self.dry_pressure_pa,
            reference_temperature_k: self.reference_temperature_k,
            dry_scale_height_m: self.dry_scale_height_m,
            co2_ppm: self.co2_ppm,
            view_step_m: self.view_step_m,
            sun_step_m: self.sun_step_m,
            air_step_m: self.air_step_m,
            air_column_intervals: self.air_column_intervals,
            sample_stride: self.sample_stride,
            spectral_step_um: 0.005,
        } // only geometric air/Sun helpers consume this adapter
    }
    pub fn validate(&self) -> Result<(), String> {
        self.quadrature().validate()?;
        self.path.validate()?;
        if self.samples_per_band < 2 || !(0.01..=1.0).contains(&self.collision_step_optical_depth) {
            return Err("at least two spectral paths per band and collision-step optical depth within 0.01..=1 required".into());
        }
        Ok(())
    }
}
pub struct MonteCarloFrame {
    pub nx: usize,
    pub ny: usize,
    pub reflectance: Vec<f32>,
    pub first_order_reflectance: Vec<f32>,
    pub higher_order_reflectance: Vec<f32>,
    pub standard_error: Vec<f32>,
    pub first_order_standard_error: Vec<f32>,
    pub higher_order_standard_error: Vec<f32>,
    pub path_exterior_fraction: Vec<f32>,
    pub sun_exterior_fraction: Vec<f32>,
    pub surface_exterior_fraction: Vec<f32>,
    pub mean_events: Vec<f32>,
    pub support_flags: Vec<u8>,
    pub glint_mask: Vec<u8>,
}
struct SolarSampler {
    wavelength: Vec<f64>,
    cdf: Vec<f64>,
}
impl SolarSampler {
    fn new(band: AbiReflectiveBand) -> Self {
        let response = AbiSolarResponse::for_band(band);
        let mut sum = 0.0;
        let mut cdf = Vec::new();
        let mut wavelength = Vec::new();
        for node in response.nodes() {
            if node.solar_response_weight_w_m2 > 0.0 {
                sum += node.solar_response_weight_w_m2;
                cdf.push(sum);
                wavelength.push(node.wavelength_um);
            }
        }
        for value in &mut cdf {
            *value /= sum;
        }
        *cdf.last_mut().unwrap() = 1.0;
        Self { wavelength, cdf }
    }
    fn sample(&self, u: f64) -> f64 {
        self.wavelength[self.cdf.partition_point(|c| *c < u).min(self.cdf.len() - 1)]
    }
}
struct World<'a> {
    vol: &'a DecodedVolume,
    mip: &'a OccupancyMip,
    georef: &'a GridGeoref,
    brick: &'a VolumeBrick,
    surface: &'a SpectralSurface,
    source_cells: &'a [usize],
    sun: [f64; 3],
    n0: f64,
    cfg: &'a MonteCarloConfig,
    quad: FirstOrderConfig,
}
struct SpectralScene<'a> {
    world: &'a World<'a>,
    molecular: DryAirRayleigh,
    ground_radius: f64,
}
impl SpectralScene<'_> {
    fn material(&self, p: [f64; 3]) -> Result<(Material, u8), String> {
        let w = self.world;
        let (i, j, _, _) = clouds::ecef_to_brick(p, w.georef, w.vol.z_min_m, w.vol.dz_m);
        if outside(i, j, w.vol) {
            // Unmeasured exterior surface is an explicit absorbing boundary. It
            // must never inherit a nearest-edge land color or an invented band.
            return Ok((Material::Lambertian { albedo: 0.0 }, 8));
        }
        let cell = j.round() as usize * w.brick.nx + i.round() as usize;
        let source = w.source_cells[cell];
        if source == usize::MAX {
            return Err("unassigned spectral surface cell".into());
        }
        if w.surface.is_land(source) {
            Ok((
                Material::Lambertian {
                    albedo: w
                        .surface
                        .sample(source, self.molecular.wavelength_um())
                        .ok_or("missing spectral land value")?,
                },
                0,
            ))
        } else {
            let wind = (w.brick.u10[cell] as f64).hypot(w.brick.v10[cell] as f64);
            Ok((
                Material::CoxMunk {
                    mean_square_slope: optics::cox_munk_mean_square_slope(wind),
                },
                0,
            ))
        }
    }
}
impl spectral_path::Scene for SpectralScene<'_> {
    fn sun_direction(&self) -> [f64; 3] {
        self.world.sun
    }
    fn sun_transmittance(&self, p: [f64; 3]) -> Result<(f64, u8), String> {
        let w = self.world;
        let Some(air) = dry_column(p, w.sun, w.n0, &w.quad) else {
            return Ok((0.0, 0));
        };
        let (cloud, exterior) = cloud_sun_depth(w.vol, w.mip, w.georef, p, w.sun, w.cfg.sun_step_m);
        Ok((
            (-cloud - air * self.molecular.cross_section_m2()).exp(),
            if exterior { 4 } else { 0 },
        ))
    }
    fn next_event(
        &self,
        p: [f64; 3],
        d: [f64; 3],
        mut tau: f64,
    ) -> Result<EventWithSupport, String> {
        let w = self.world;
        let Some((entry, exit)) = sphere(p, d, TOP) else {
            return Ok(EventWithSupport {
                event: Event::Escape,
                flags: 0,
            });
        };
        if exit <= 0.0 {
            return Ok(EventWithSupport {
                event: Event::Escape,
                flags: 0,
            });
        }
        let mut t = entry.max(0.0);
        let ground = sphere(p, d, self.ground_radius).and_then(|(a, _)| {
            if a >= t + 1e-5 {
                Some(a)
            } else if a >= -1e-5 && dot(p, d) < 0.0 {
                Some(t)
            } else {
                None
            }
        });
        let end = ground.unwrap_or(exit).min(exit);
        let mut flags = 0;
        let sigma = self.molecular.cross_section_m2();
        let mut steps = 0usize;
        while t < end {
            steps += 1;
            if steps > 2000000 {
                return Err(
                    "Monte Carlo segment safety limit reached; no partial image saved".into(),
                );
            }
            let point = add(p, d, t);
            let (i, j, k, r) = clouds::ecef_to_brick(point, w.georef, w.vol.z_min_m, w.vol.dz_m);
            let inside_shell = r >= w.vol.r_bottom() && r <= w.vol.r_top();
            if inside_shell && outside(i, j, w.vol) {
                flags |= 2;
            }
            let start_cloud = w.vol.sample(i, j, k);
            let start_beta = start_cloud.total_ext()
                + dry_density(point, w.n0, w.cfg.dry_scale_height_m) * sigma;
            let requested = if r > w.vol.r_top() + w.cfg.air_step_m {
                w.cfg.air_step_m
            } else if w.mip.maxext_at(i, j, k) <= 0.0 {
                w.cfg.air_step_m.min(w.vol.voxel_pitch_m() * 4.0)
            } else {
                w.cfg.view_step_m.min(w.vol.voxel_pitch_m())
            };
            let optical_step = if start_beta > 0.0 {
                (w.cfg.collision_step_optical_depth / start_beta).max(1e-4)
            } else {
                requested
            };
            let mut ds = requested.min(optical_step).min(end - t);
            let (cloud, air, beta) = loop {
                if ds <= 0.0 || t + ds == t {
                    return Err("non-progressing spectral free path".into());
                }
                let mid = add(p, d, t + 0.5 * ds);
                let (i, j, k, rm) = clouds::ecef_to_brick(mid, w.georef, w.vol.z_min_m, w.vol.dz_m);
                if rm >= w.vol.r_bottom() && rm <= w.vol.r_top() && outside(i, j, w.vol) {
                    flags |= 2;
                }
                let cloud = w.vol.sample(i, j, k);
                let air = dry_density(mid, w.n0, w.cfg.dry_scale_height_m) * sigma;
                let beta = air + cloud.total_ext();
                // Also resolve entry into denser cloud, where the starting-point
                // extinction alone would not constrain the segment adequately.
                if beta * ds <= w.cfg.collision_step_optical_depth || ds <= 1e-4 {
                    break (cloud, air, beta);
                }
                ds *= 0.5;
            };
            if beta > 0.0 && tau < beta * ds {
                let phase = PhaseFunction::model_mixture(
                    air,
                    cloud.ext_liquid,
                    cloud.ext_ice + cloud.ext_precip,
                    self.molecular.phase_gamma(),
                )?;
                return Ok(EventWithSupport {
                    event: Event::Scatter {
                        point: add(p, d, t + tau / beta),
                        phase,
                        single_scatter_albedo: 1.0,
                    },
                    flags,
                });
            }
            tau -= beta * ds;
            t += ds;
        }
        if ground.is_some_and(|g| g <= exit) {
            let point = add(p, d, end);
            let (material, extra) = self.material(point)?;
            Ok(EventWithSupport {
                event: Event::Surface {
                    point,
                    normal: spectral_path::unit(point),
                    material,
                },
                flags: flags | extra,
            })
        } else {
            Ok(EventWithSupport {
                event: Event::Escape,
                flags,
            })
        }
    }
}
struct Pixel {
    total: [f32; 3],
    first: [f32; 3],
    higher: [f32; 3],
    stderr: [f32; 3],
    first_stderr: [f32; 3],
    higher_stderr: [f32; 3],
    path_exterior: [f32; 3],
    sun_exterior: [f32; 3],
    surface_exterior: [f32; 3],
    events: [f32; 3],
    flags: u8,
    glint: u8,
}
impl Pixel {
    fn missing() -> Self {
        Self {
            total: [f32::NAN; 3],
            first: [f32::NAN; 3],
            higher: [f32::NAN; 3],
            stderr: [f32::NAN; 3],
            first_stderr: [f32::NAN; 3],
            higher_stderr: [f32::NAN; 3],
            path_exterior: [f32::NAN; 3],
            sun_exterior: [f32::NAN; 3],
            surface_exterior: [f32::NAN; 3],
            events: [f32::NAN; 3],
            flags: 0,
            glint: 0,
        }
    }
}
#[allow(clippy::too_many_arguments)]
pub fn render(
    brick: &VolumeBrick,
    georef: &GridGeoref,
    raster: &SurfaceRaster,
    camera: &CameraGeometry,
    sun: [f64; 3],
    surface: &SpectralSurface,
    horizontal_pitch_m: f64,
    cfg: &MonteCarloConfig,
) -> Result<MonteCarloFrame, String> {
    cfg.validate()?;
    surface.raster_rgba(brick.nx, brick.ny, &brick.landmask, raster, |la, lo| {
        georef.forward(la, lo)
    })?;
    let mut source_cells = vec![usize::MAX; brick.nx * brick.ny];
    for (cell, &[la, lo]) in surface.coordinates().iter().enumerate() {
        let (i, j) = georef.forward(la, lo);
        source_cells[j.round() as usize * brick.nx + i.round() as usize] = cell;
    }
    let vol = DecodedVolume::from_brick(brick, horizontal_pitch_m);
    let mip = OccupancyMip::build(&vol, clouds::OCCUPANCY_MIP_FACTOR);
    let world = World {
        vol: &vol,
        mip: &mip,
        georef,
        brick,
        surface,
        source_cells: &source_cells,
        sun,
        n0: dry_air_number_density_m3(cfg.dry_pressure_pa, cfg.reference_temperature_k)
            .map_err(|e| e.to_string())?,
        cfg,
        quad: cfg.quadrature(),
    };
    let solar = AbiReflectiveBand::ALL.map(SolarSampler::new);
    let completed = AtomicUsize::new(0);
    let rows = (0..raster.ny)
        .into_par_iter()
        .map(|y| -> Result<Vec<Pixel>, String> {
            let mut row = Vec::with_capacity(raster.nx);
            for x in 0..raster.nx {
                if !x.is_multiple_of(cfg.sample_stride) || !y.is_multiple_of(cfg.sample_stride) {
                    row.push(Pixel::missing());
                    continue;
                }
                let pixel = y * raster.nx + x;
                let i = (raster.grid_i[pixel] as f64)
                    .round()
                    .clamp(0.0, (brick.nx - 1) as f64) as usize;
                let j = (raster.grid_j[pixel] as f64)
                    .round()
                    .clamp(0.0, (brick.ny - 1) as f64) as usize;
                let cell = j * brick.nx + i;
                let (sx, sy) = raster.model_scan_angle(x, y);
                let direction = camera.view_dir(sx, sy);
                let radius = R + (brick.hgt[cell] as f64).max(0.0);
                let mut result = Pixel::missing();
                result.flags = 1;
                for (b, response) in solar.iter().enumerate() {
                    let (mut total, mut first, mut higher) =
                        (Moments::default(), Moments::default(), Moments::default());
                    let mut events = 0usize;
                    let mut support_counts = [0usize; 3];
                    for sample in 0..cfg.samples_per_band {
                        // Each path receives a fresh deterministic stream. Common seeds
                        // across bands improve color correlation without sharing physics.
                        let mut rng = Random::new(
                            cfg.seed
                                .wrapping_add((pixel as u32).wrapping_mul(2654435761))
                                .wrapping_add(sample as u32),
                        );
                        let lambda = response.sample(rng.uniform());
                        let scene = SpectralScene {
                            world: &world,
                            molecular: DryAirRayleigh::new(lambda, cfg.co2_ppm)
                                .map_err(|e| e.to_string())?,
                            ground_radius: radius,
                        };
                        let path = spectral_path::trace(
                            &scene,
                            camera.camera,
                            direction,
                            cfg.path,
                            &mut rng,
                        )
                        .map_err(|error| {
                            format!("pixel ({x},{y}), C0{}, path {sample}: {error}", b + 1)
                        })?;
                        total.push(PI * path.total());
                        first.push(PI * path.first_order_l_over_e);
                        higher.push(PI * path.higher_order_l_over_e);
                        events += path.events;
                        result.flags |= path.flags;
                        for (count, mask) in support_counts.iter_mut().zip([2u8, 4, 8]) {
                            *count += usize::from(path.flags & mask != 0);
                        }
                    }
                    result.total[b] = total.mean as f32;
                    result.first[b] = first.mean as f32;
                    result.higher[b] = higher.mean as f32;
                    result.stderr[b] = total.standard_error().unwrap() as f32;
                    result.first_stderr[b] = first.standard_error().unwrap() as f32;
                    result.higher_stderr[b] = higher.standard_error().unwrap() as f32;
                    result.path_exterior[b] =
                        (support_counts[0] as f64 / cfg.samples_per_band as f64) as f32;
                    result.sun_exterior[b] =
                        (support_counts[1] as f64 / cfg.samples_per_band as f64) as f32;
                    result.surface_exterior[b] =
                        (support_counts[2] as f64 / cfg.samples_per_band as f64) as f32;
                    result.events[b] = (events as f64 / cfg.samples_per_band as f64) as f32;
                }
                if brick.landmask[cell] < 0.5
                    && let Some((g, _)) = sphere(camera.camera, direction, radius)
                {
                    let up = spectral_path::unit(add(camera.camera, direction, g));
                    let facet = spectral_path::unit(std::array::from_fn(|k| sun[k] - direction[k]));
                    let ch = dot(facet, up);
                    let wind = (brick.u10[cell] as f64).hypot(brick.v10[cell] as f64);
                    if dot(up, sun) > 0.0
                        && ch > 0.0
                        && (1.0 - ch * ch).max(0.0) / (ch * ch)
                            <= optics::cox_munk_mean_square_slope(wind)
                    {
                        result.glint = 1;
                    }
                }
                row.push(result);
            }
            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if done.is_multiple_of(16) || done == raster.ny {
                crate::log_line!("ABI Monte Carlo: {done}/{} rows complete", raster.ny);
            }
            Ok(row)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let count = raster.nx * raster.ny;
    let mut frame = MonteCarloFrame {
        nx: raster.nx,
        ny: raster.ny,
        reflectance: Vec::with_capacity(count * 3),
        first_order_reflectance: Vec::with_capacity(count * 3),
        higher_order_reflectance: Vec::with_capacity(count * 3),
        standard_error: Vec::with_capacity(count * 3),
        first_order_standard_error: Vec::with_capacity(count * 3),
        higher_order_standard_error: Vec::with_capacity(count * 3),
        path_exterior_fraction: Vec::with_capacity(count * 3),
        sun_exterior_fraction: Vec::with_capacity(count * 3),
        surface_exterior_fraction: Vec::with_capacity(count * 3),
        mean_events: Vec::with_capacity(count * 3),
        support_flags: Vec::with_capacity(count),
        glint_mask: Vec::with_capacity(count),
    };
    for p in rows.into_iter().flatten() {
        frame.reflectance.extend(p.total);
        frame.first_order_reflectance.extend(p.first);
        frame.higher_order_reflectance.extend(p.higher);
        frame.standard_error.extend(p.stderr);
        frame.first_order_standard_error.extend(p.first_stderr);
        frame.higher_order_standard_error.extend(p.higher_stderr);
        frame.path_exterior_fraction.extend(p.path_exterior);
        frame.sun_exterior_fraction.extend(p.sun_exterior);
        frame.surface_exterior_fraction.extend(p.surface_exterior);
        frame.mean_events.extend(p.events);
        frame.support_flags.push(p.flags);
        frame.glint_mask.push(p.glint);
    }
    Ok(frame)
}
