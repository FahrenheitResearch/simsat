//! Synthetic infrared (ABI band 13, 10.3 um) radiative-transfer pass — CPU
//! reference (design doc section 7, M6).
//!
//! This is the SHIPPING IR path and its tested reference (matching the M4/M5
//! decision that the CPU cloud march is the shipping path). It marches each IR
//! output pixel's actual slant ray — geostationary geometry, so the limb slant is
//! honest — TOP-DOWN through the brick, accumulating gray-body THERMAL emission,
//! and converts the accumulated 10.3 um radiance back to a true-Kelvin BRIGHTNESS
//! TEMPERATURE. It is purely thermal: NO sun/exposure/multi-scatter dependence, so
//! it works day AND night — the whole point for overnight convection (Enderlin).
//! A WGSL twin is OPTIONAL and deferred to a future M6-GPU pass (alongside the
//! deferred cloud GPU pass); the correctness-critical ECEF + projection march
//! cannot be validated on the headless nodes, so shipping the fully unit-tested
//! CPU reference is the honest choice (design section 9).
//!
//! Radiative transfer (design section 7). Per voxel, absorption is the gray
//! 10.3 um absorption of each hydrometeor class recovered from the brick's stored
//! VISIBLE extinction (`optics::ir_absorption_from_ext`) plus a weak water-vapor
//! continuum from the qvapor column (`optics::ir_wv_continuum_absorption`); the
//! voxel emits `B(T_voxel, 10.3 um)` (`optics::planck_radiance`) weighted by that
//! absorption. The march runs FRONT-TO-BACK (space -> down) with a running
//! transmittance; after the volume the surface term `emissivity * B(TSK)` is added
//! through the remaining transmittance. The accumulated band radiance is inverted
//! to a brightness temperature (`optics::inverse_planck`). An optically thick
//! anvil (tau_ir >> 1) drives the transmittance to ~0 within its cold top, so its
//! BT equals its cloud-top temperature (the M6 proof); a clear column keeps the
//! transmittance near 1, so its BT equals TSK (minus the weak WV depression); a
//! thin cloud lands between the two, monotone in optical depth (all unit-tested).
//!
//! Geometry mirrors `clouds.rs`: distances in metres, ECEF radii from the earth
//! centre, brick vertical axis MSL height `z(k) = z_min + k*dz`, ground sphere at
//! `R_GROUND_M`. The per-step ECEF -> lat/lon/h -> projection-forward -> `(i,j)`
//! transform and the shell intersection are reused from `clouds.rs`.

use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;

use crate::atmosphere::{CameraGeometry, R_GROUND_M};
use crate::bricks::{VolumeBrick, decode_temperature_kelvin};
use crate::camera::ScanGrid;
use crate::clouds::{OccupancyMip, ecef_to_brick, ray_shell_segment};
use crate::frame::GridGeoref;
use crate::optics::{
    HydrometeorClass, IR_BAND13_WAVELENGTH_M, IR_SURFACE_EMISSIVITY,
    IR_WV_CONTINUUM_MASS_ABS_M2_KG, inverse_planck, ir_absorption_from_ext, planck_radiance,
    wv_absorption,
};

/// ABI band number for the 10.3 um clean longwave window (the design's band 13).
pub const IR_BAND: u8 = 13;

/// Process-wide sticky diagnostic: set when any [`IrVolume::from_brick`] engaged the
/// TSK FALLBACK (a missing / all-zero skin-temperature plane, substituted by the
/// lowest-level air temperature). An `IrVolume` field would be the natural home,
/// but the struct is literal-constructed outside this module (WS1 does not own
/// those files, and a field addition breaks their exhaustive literals), so the
/// sticky flag + a stderr warning carry the diagnostic to the CLI harness instead
/// (a documented deviation; `render_ir` prints it in `IRSUMMARY`).
static TSK_FALLBACK_ENGAGED: AtomicBool = AtomicBool::new(false);

/// Whether any brick decoded in this process engaged the TSK fallback (see
/// [`IrVolume::from_brick`]). Sticky for the life of the process.
pub fn tsk_fallback_engaged() -> bool {
    TSK_FALLBACK_ENGAGED.load(Ordering::Relaxed)
}

// ── vec3 helpers (local; the clouds.rs ones are private) ──────────────────────

#[inline]
fn madd3(a: [f64; 3], b: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s]
}

// ── decoded IR volume ─────────────────────────────────────────────────────────

/// A brick decoded for the IR emission march: the three VISIBLE extinction classes
/// (from which the 10.3 um absorption is recovered per class), the absolute
/// temperature field (Kelvin), the water-vapor mixing ratio (for the continuum),
/// and the 2-D skin-temperature plane (the surface emission source). Index
/// `(k*ny + j)*nx + i` for the 3-D fields, `j*nx + i` for `tsk`.
#[derive(Debug, Clone)]
pub struct IrVolume {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub z_min_m: f64,
    pub dz_m: f64,
    /// Horizontal cell size (m) — drives the march step pitch (min of dx/dy).
    pub horiz_pitch_m: f64,
    /// Cloud-liquid visible extinction (m^-1).
    pub ext_liquid: Vec<f32>,
    /// Ice (QICE+QSNOW) visible extinction (m^-1).
    pub ext_ice: Vec<f32>,
    /// Precip (QRAIN+QGRAUP) visible extinction (m^-1).
    pub ext_precip: Vec<f32>,
    /// Absolute temperature (K), decoded from the f16-Celsius Texture B.
    pub temperature_k: Vec<f32>,
    /// Water-vapor mixing ratio (kg kg^-1).
    pub qvapor: Vec<f32>,
    /// Skin temperature (K), 2-D `ny*nx`.
    pub tsk: Vec<f32>,
}

/// One trilinearly-sampled IR voxel.
#[derive(Debug, Clone, Copy, Default)]
pub struct IrSample {
    pub ext_liquid: f64,
    pub ext_ice: f64,
    pub ext_precip: f64,
    pub temperature_k: f64,
    pub qvapor: f64,
}

impl IrVolume {
    /// Decode a brick's IR inputs. The extinction channels decode via their per-
    /// volume `LogQuant` scales (same as the cloud path); temperature via the
    /// f16-Celsius decoder (`K = C + 273.15`); qvapor via its own log scale; TSK is
    /// carried through as the Kelvin 2-D plane.
    ///
    /// SUB-TERRAIN VAPOR CLIP (WS1 march-physics pass). The ingest fills brick
    /// levels below the lowest WRF model level with the CLAMPED surface vapor
    /// (`Extrap::ClampEdge`), so the raw decoded qvapor carries FICTITIOUS vapor
    /// below elevated terrain — over a ~1500 m surface the 6.2-7.3 um WV bands
    /// accumulated ~tau 3 of absorption/emission from air that does not exist
    /// (band 13's weak continuum shifted < 1 K). Each voxel's vapor is scaled by
    /// its layer's above-terrain fraction (layer `k` spans `[z_k, z_k + dz)`, the
    /// SAME per-layer clip `derived::precipitable_water_field` applies), so the
    /// trilinearly-sampled march integrates a terrain-clipped moisture field.
    /// `hgt <= z_min` leaves the column bit-identical (sea-level / flat test
    /// bricks). Cloud extinction needs no clip — the ingest fills it `Extrap::Zero`
    /// below the column.
    ///
    /// TSK FALLBACK (WS1 march-physics pass). A brick whose TSK plane is missing or
    /// ALL-ZERO (the ingest zero-fills a TSK-less wrfout) would emit `B(0 K) = 0`
    /// from the ground — a deep-space-cold surface. When no valid TSK value exists,
    /// the per-column LOWEST-LEVEL AIR TEMPERATURE is substituted (the closest
    /// physical stand-in the brick carries), a stderr warning is printed, and the
    /// process-wide [`tsk_fallback_engaged`] diagnostic is set.
    pub fn from_brick(brick: &VolumeBrick, horiz_pitch_m: f64) -> Self {
        let ql = brick.quant.get("ext_liquid");
        let qi = brick.quant.get("ext_ice");
        let qp = brick.quant.get("ext_precip");
        let qv = brick.quant.get("qvapor");
        let (nx, ny, nz) = (brick.nx, brick.ny, brick.nz);
        let temperature_k = decode_temperature_kelvin(&brick.temperature_f16);

        let mut qvapor: Vec<f32> = brick.qvapor.iter().map(|&c| qv.decode(c)).collect();
        if brick.hgt.len() == nx * ny {
            for j in 0..ny {
                for i in 0..nx {
                    let surface = brick.hgt[j * nx + i] as f64;
                    if surface <= brick.z_min_m {
                        continue;
                    }
                    for k in 0..nz {
                        let zb = brick.z_min_m + k as f64 * brick.dz_m;
                        if zb >= surface {
                            break;
                        }
                        // The fraction of layer [zb, zb+dz) above the terrain.
                        let frac = ((zb + brick.dz_m - surface) / brick.dz_m).clamp(0.0, 1.0);
                        qvapor[(k * ny + j) * nx + i] *= frac as f32;
                    }
                }
            }
        }

        let mut tsk = brick.tsk.clone();
        let has_valid_tsk = tsk.len() == nx * ny && tsk.iter().any(|v| v.is_finite() && *v > 0.0);
        if !has_valid_tsk {
            // Lowest-level (k = 0) air temperature, per column.
            tsk = temperature_k[..nx * ny].to_vec();
            TSK_FALLBACK_ENGAGED.store(true, Ordering::Relaxed);
            eprintln!(
                "simsat ir: TSK plane missing or all-zero — substituting the lowest-level \
                 air temperature for the surface emission (tsk_fallback)"
            );
        }

        Self {
            nx,
            ny,
            nz,
            z_min_m: brick.z_min_m,
            dz_m: brick.dz_m,
            horiz_pitch_m,
            ext_liquid: brick.ext_liquid.iter().map(|&c| ql.decode(c)).collect(),
            ext_ice: brick.ext_ice.iter().map(|&c| qi.decode(c)).collect(),
            ext_precip: brick.ext_precip.iter().map(|&c| qp.decode(c)).collect(),
            temperature_k,
            qvapor,
            tsk,
        }
    }

    #[inline]
    fn cell(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.ny + j) * self.nx + i
    }

    /// Trilinearly sample at fractional grid coords, EDGE-CLAMPING each axis to
    /// `[0, n-1]`. Edge-clamp (rather than zero-outside like the cloud march) keeps
    /// the temperature field physical everywhere along the slant ray — a 0 K sample
    /// would emit no radiance and corrupt the BT. The domain edge is normally clear
    /// (near-zero extinction), so clamping the extinction there is benign; a limb
    /// ray that grazes just outside the horizontal domain reads the domain edge (a
    /// documented approximation, tiny for the near-nadir geostationary-over-domain
    /// geometry).
    pub fn sample(&self, fi: f64, fj: f64, fk: f64) -> IrSample {
        if !(fi.is_finite() && fj.is_finite() && fk.is_finite()) {
            return IrSample::default();
        }
        let fi = fi.clamp(0.0, (self.nx - 1) as f64);
        let fj = fj.clamp(0.0, (self.ny - 1) as f64);
        let fk = fk.clamp(0.0, (self.nz - 1) as f64);
        let i0 = fi.floor() as usize;
        let j0 = fj.floor() as usize;
        let k0 = fk.floor() as usize;
        let i1 = (i0 + 1).min(self.nx - 1);
        let j1 = (j0 + 1).min(self.ny - 1);
        let k1 = (k0 + 1).min(self.nz - 1);
        let ti = fi - i0 as f64;
        let tj = fj - j0 as f64;
        let tk = fk - k0 as f64;
        let trilerp = |ch: &[f32]| -> f64 {
            let g = |i: usize, j: usize, k: usize| ch[self.cell(i, j, k)] as f64;
            let c00 = g(i0, j0, k0) * (1.0 - ti) + g(i1, j0, k0) * ti;
            let c10 = g(i0, j1, k0) * (1.0 - ti) + g(i1, j1, k0) * ti;
            let c01 = g(i0, j0, k1) * (1.0 - ti) + g(i1, j0, k1) * ti;
            let c11 = g(i0, j1, k1) * (1.0 - ti) + g(i1, j1, k1) * ti;
            let c0 = c00 * (1.0 - tj) + c10 * tj;
            let c1 = c01 * (1.0 - tj) + c11 * tj;
            c0 * (1.0 - tk) + c1 * tk
        };
        IrSample {
            ext_liquid: trilerp(&self.ext_liquid),
            ext_ice: trilerp(&self.ext_ice),
            ext_precip: trilerp(&self.ext_precip),
            temperature_k: trilerp(&self.temperature_k),
            qvapor: trilerp(&self.qvapor),
        }
    }

    /// Bilinearly sample the 2-D skin-temperature plane (K) at fractional `(fi, fj)`,
    /// edge-clamped. Returns `NaN` if the coords are non-finite.
    pub fn sample_tsk(&self, fi: f64, fj: f64) -> f64 {
        if !(fi.is_finite() && fj.is_finite()) {
            return f64::NAN;
        }
        let fi = fi.clamp(0.0, (self.nx - 1) as f64);
        let fj = fj.clamp(0.0, (self.ny - 1) as f64);
        let i0 = fi.floor() as usize;
        let j0 = fj.floor() as usize;
        let i1 = (i0 + 1).min(self.nx - 1);
        let j1 = (j0 + 1).min(self.ny - 1);
        let ti = fi - i0 as f64;
        let tj = fj - j0 as f64;
        let g = |i: usize, j: usize| self.tsk[j * self.nx + i] as f64;
        let a = g(i0, j0) * (1.0 - ti) + g(i1, j0) * ti;
        let b = g(i0, j1) * (1.0 - ti) + g(i1, j1) * ti;
        a * (1.0 - tj) + b * tj
    }

    /// The 10.3 um cloud absorption coefficient (m^-1) of a sample: the sum over the
    /// three classes of `optics::ir_absorption_from_ext` (liquid, ice, precip). Ice
    /// carries QICE+QSNOW (ice mass absorption); precip carries QRAIN+QGRAUP (precip
    /// mass absorption).
    #[inline]
    pub fn cloud_ir_absorption(&self, s: &IrSample) -> f64 {
        ir_absorption_from_ext(HydrometeorClass::CloudLiquid, s.ext_liquid)
            + ir_absorption_from_ext(HydrometeorClass::Ice, s.ext_ice)
            + ir_absorption_from_ext(HydrometeorClass::Rain, s.ext_precip)
    }

    /// Top-of-brick ECEF radius (m).
    #[inline]
    pub fn r_top(&self) -> f64 {
        R_GROUND_M + self.z_min_m + self.nz as f64 * self.dz_m
    }

    /// Bottom-of-brick ECEF radius (m).
    #[inline]
    pub fn r_bottom(&self) -> f64 {
        R_GROUND_M + self.z_min_m
    }

    /// The march step pitch (m): the finest of dz and the horizontal cell.
    #[inline]
    pub fn voxel_pitch_m(&self) -> f64 {
        self.dz_m.min(self.horiz_pitch_m).max(1.0)
    }
}

// ── the IR march ──────────────────────────────────────────────────────────────

/// IR march tuning (design section 7).
#[derive(Debug, Clone, Copy)]
pub struct IrConfig {
    /// ABI band number recorded in the output (13).
    pub band: u8,
    /// Band centre wavelength (m) for the Planck / inverse-Planck.
    pub wavelength_m: f64,
    /// Longwave surface emissivity for the `epsilon * B(TSK)` surface term.
    pub surface_emissivity: f64,
    /// Coarse-step multiplier of the voxel pitch through cloud-free space (the mip
    /// says the block is empty). The weak WV continuum is still integrated on the
    /// coarse step; the mip is conservative, so no cloud is stepped over.
    pub coarse_mult: f64,
    /// Fine-step multiplier of the voxel pitch inside cloud.
    pub fine_mult: f64,
    /// Hard primary-step cap.
    pub max_steps: usize,
    /// Include the water-vapor absorption term (design section 7). On by default; the
    /// analytic clear-sky test turns it off to check `BT == TSK` exactly.
    pub wv_continuum: bool,
    /// Water-vapor mass-absorption coefficient (m^2 kg^-1) applied per voxel to
    /// `rho_air(z) * qvapor`. For band 13 this is the WEAK window continuum
    /// ([`IR_WV_CONTINUUM_MASS_ABS_M2_KG`]); the WV bands (8/9/10) set a strong per-band
    /// value (`wv::WvBand::ir_config`) so water vapor becomes the dominant emitter and
    /// the weighting function moves up into the troposphere (design section 7, WV addendum).
    pub wv_mass_abs_m2_kg: f64,
    /// Stop the march once the transmittance drops below this (the cloud is opaque).
    pub transmittance_floor: f64,
}

impl IrConfig {
    /// Defaults for a band-13 IR frame at a given voxel pitch.
    pub fn band13() -> Self {
        Self {
            band: IR_BAND,
            wavelength_m: IR_BAND13_WAVELENGTH_M,
            surface_emissivity: IR_SURFACE_EMISSIVITY,
            coarse_mult: 4.0,
            fine_mult: 0.5,
            max_steps: 768,
            wv_continuum: true,
            wv_mass_abs_m2_kg: IR_WV_CONTINUUM_MASS_ABS_M2_KG,
            transmittance_floor: 1.0e-4,
        }
    }
}

impl Default for IrConfig {
    fn default() -> Self {
        Self::band13()
    }
}

/// The scene resources one IR march reads.
pub struct IrScene<'a> {
    pub vol: &'a IrVolume,
    /// Occupancy mip for coarse empty-space skipping (built from the same brick's
    /// extinction; conservative, so coarse steps never skip cloud).
    pub mip: &'a OccupancyMip,
    pub georef: &'a GridGeoref,
    pub cfg: IrConfig,
}

/// The result of one IR view-ray march.
#[derive(Debug, Clone, Copy)]
pub struct IrMarch {
    /// Accumulated upwelling 10.3 um band radiance at the satellite (W m^-3 sr^-1).
    pub radiance: f64,
    /// Transmittance from space down to the surface (1 = clear, ~0 = opaque cloud).
    pub surface_transmittance: f64,
    /// Whether a surface (skin-temperature) term was added (the ray hit the ground).
    pub hit_surface: bool,
}

impl IrMarch {
    /// The brightness temperature (K) of the accumulated radiance (inverse Planck).
    #[inline]
    pub fn brightness_temperature(&self, wavelength_m: f64) -> f64 {
        inverse_planck(self.radiance, wavelength_m)
    }
}

/// March one view ray from the camera down through the brick and add the surface
/// emission, returning the accumulated band radiance. `None` if the ray misses the
/// brick shell entirely (deep space / off-earth — no IR data). Front-to-back
/// (space -> down) with a running transmittance. Twin of the eventual WGSL
/// `march_ir` (deferred).
pub fn march_ir(scene: &IrScene, cam: [f64; 3], view: [f64; 3]) -> Option<IrMarch> {
    let vol = scene.vol;
    let cfg = scene.cfg;
    let (t_enter, t_exit) = ray_shell_segment(cam, view, vol.r_bottom(), vol.r_top())?;
    let seg = t_exit - t_enter;
    if seg <= 0.0 {
        return None;
    }
    let pitch = vol.voxel_pitch_m();
    let coarse = cfg.coarse_mult * pitch;
    let fine = cfg.fine_mult * pitch;

    let mut trans = 1.0f64;
    let mut radiance = 0.0f64;
    let mut t = t_enter;
    let mut steps = 0usize;

    while t < t_exit && steps < cfg.max_steps && trans > cfg.transmittance_floor {
        let p = madd3(cam, view, t);
        let (fi, fj, fk, _r) = ecef_to_brick(p, scene.georef, vol.z_min_m, vol.dz_m);
        let occ = scene.mip.maxext_at(fi, fj, fk);
        let mut ds = if occ > 0.0 { fine } else { coarse };
        if t + ds > t_exit {
            ds = t_exit - t;
        }
        if ds <= 0.0 {
            break;
        }
        // Integrate absorption + emission over [t, t+ds] at the segment midpoint.
        let pm = madd3(cam, view, t + 0.5 * ds);
        let (mi, mj, mk, rm) = ecef_to_brick(pm, scene.georef, vol.z_min_m, vol.dz_m);
        let s = vol.sample(mi, mj, mk);
        let height = (rm - R_GROUND_M).max(0.0);
        let mut beta = vol.cloud_ir_absorption(&s);
        if cfg.wv_continuum {
            beta += wv_absorption(s.qvapor, height, cfg.wv_mass_abs_m2_kg);
        }
        if beta > 0.0 {
            let step_t = (-beta * ds).exp();
            let b = planck_radiance(s.temperature_k, cfg.wavelength_m);
            radiance += trans * b * (1.0 - step_t);
            trans *= step_t;
        }
        t += ds;
        steps += 1;
    }

    // Surface term: the ground hit is the inner-shell exit (z_min = 0 -> r_bottom =
    // R_GROUND). Add `emissivity * B(TSK)` through the remaining transmittance when
    // the ray actually reaches the ground.
    let mut hit_surface = false;
    let p_ground = madd3(cam, view, t_exit);
    let (gi, gj, _gk, rg) = ecef_to_brick(p_ground, scene.georef, vol.z_min_m, vol.dz_m);
    if (rg - vol.r_bottom()).abs() < vol.dz_m {
        let tsk = vol.sample_tsk(gi, gj);
        if tsk.is_finite() && tsk > 0.0 {
            radiance += trans * cfg.surface_emissivity * planck_radiance(tsk, cfg.wavelength_m);
            hit_surface = true;
        }
    }

    Some(IrMarch {
        radiance,
        surface_transmittance: trans,
        hit_surface,
    })
}

/// The brightness temperature (K) of one view ray, or `None` if the ray misses the
/// brick shell.
pub fn march_ir_bt(scene: &IrScene, cam: [f64; 3], view: [f64; 3]) -> Option<f64> {
    march_ir(scene, cam, view).map(|m| m.brightness_temperature(scene.cfg.wavelength_m))
}

/// Render a full IR brightness-temperature plane (Kelvin) for a scan raster: for
/// each IN-DOMAIN pixel (`grid_i` finite) march its slant ray and store its BT;
/// off-earth / out-of-domain pixels are `NaN` (no IR data). Row-major, row 0 =
/// north, `scan.nx * scan.ny`. Rows in parallel (rayon) on the below-normal worker.
/// This is the M6 STUDIO render path (the tested CPU reference; a GPU twin is
/// deferred). The BT plane is what the enhancement (`ir_enhance`) colours and what
/// the store writer (`store_out::write_ir_frame`) writes as a single-band Kelvin
/// `surface2d` for live re-enhancement.
pub fn render_ir_bt_frame(
    scene: &IrScene,
    cam: &CameraGeometry,
    scan: &ScanGrid,
    grid_i: &[f32],
) -> Vec<f32> {
    let (nx, ny) = (scan.nx, scan.ny);
    let rows: Vec<Vec<f32>> = (0..ny)
        .into_par_iter()
        .map(|py| {
            let mut row = vec![f32::NAN; nx];
            for (px, out) in row.iter_mut().enumerate() {
                let idx = py * nx + px;
                if !grid_i[idx].is_finite() {
                    continue; // off-earth or outside the WRF domain -> no IR data
                }
                let (sx, sy) = scan.scan_angle(px, py);
                let view = cam.view_dir(sx, sy);
                if let Some(bt) = march_ir_bt(scene, cam.camera, view) {
                    *out = bt as f32;
                }
            }
            row
        })
        .collect();
    rows.into_iter().flatten().collect()
}

/// Coverage / range summary of a BT plane, for the QA harness + the env-gated
/// fixture assertion (design section 9): the count of finite (in-domain) pixels and
/// the coldest / warmest / median BT (K). The coldest is the deep-anvil cloud-top
/// BT; the warmest is the warm ground.
#[derive(Debug, Clone, Copy)]
pub struct IrFrameStats {
    pub finite: usize,
    pub min_bt: f64,
    pub max_bt: f64,
    pub median_bt: f64,
    pub all_finite_in_domain: bool,
}

/// Summarise a BT plane's in-domain pixels (finite entries).
pub fn ir_frame_stats(bt: &[f32]) -> IrFrameStats {
    let mut vals: Vec<f64> = bt
        .iter()
        .filter(|v| v.is_finite())
        .map(|&v| v as f64)
        .collect();
    if vals.is_empty() {
        return IrFrameStats {
            finite: 0,
            min_bt: 0.0,
            max_bt: 0.0,
            median_bt: 0.0,
            all_finite_in_domain: true,
        };
    }
    vals.sort_by(f64::total_cmp);
    let min_bt = vals[0];
    let max_bt = vals[vals.len() - 1];
    let median_bt = vals[vals.len() / 2];
    // "all finite in domain" = every non-NaN entry is a real positive BT (no Inf,
    // no zero/negative). NaN entries are the out-of-domain mask, not failures.
    let all_finite_in_domain = vals.iter().all(|&v| v.is_finite() && v > 0.0);
    IrFrameStats {
        finite: vals.len(),
        min_bt,
        max_bt,
        median_bt,
        all_finite_in_domain,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clouds::{DecodedVolume, OCCUPANCY_MIP_FACTOR, brick_to_ecef};
    use crate::frame::MapProjection;

    fn test_georef(nx: usize, ny: usize, dx: f64) -> GridGeoref {
        let proj = MapProjection::lambert(30.0, 60.0, -100.0);
        GridGeoref::new(
            proj,
            (nx - 1) as f64 / 2.0,
            (ny - 1) as f64 / 2.0,
            45.0,
            -100.0,
            dx,
            dx,
        )
    }

    /// Build a synthetic IR volume with caller-filled per-voxel `(ext_liquid,
    /// ext_ice, ext_precip, temperature_k, qvapor)` and a uniform TSK.
    #[allow(clippy::too_many_arguments)]
    fn build_ir_volume(
        nx: usize,
        ny: usize,
        nz: usize,
        dz: f64,
        horiz: f64,
        tsk_k: f32,
        fill: impl Fn(usize, usize, usize) -> (f64, f64, f64, f64, f64),
    ) -> IrVolume {
        let n = nx * ny * nz;
        let mut ext_liquid = vec![0.0f32; n];
        let mut ext_ice = vec![0.0f32; n];
        let mut ext_precip = vec![0.0f32; n];
        let mut temperature_k = vec![0.0f32; n];
        let mut qvapor = vec![0.0f32; n];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let (l, ic, p, t, q) = fill(i, j, k);
                    let c = (k * ny + j) * nx + i;
                    ext_liquid[c] = l as f32;
                    ext_ice[c] = ic as f32;
                    ext_precip[c] = p as f32;
                    temperature_k[c] = t as f32;
                    qvapor[c] = q as f32;
                }
            }
        }
        IrVolume {
            nx,
            ny,
            nz,
            z_min_m: 0.0,
            dz_m: dz,
            horiz_pitch_m: horiz,
            ext_liquid,
            ext_ice,
            ext_precip,
            temperature_k,
            qvapor,
            tsk: vec![tsk_k; nx * ny],
        }
    }

    /// A DecodedVolume with the SAME extinction as an IrVolume (for the occupancy
    /// mip). The IR march only needs the mip for step sizing.
    fn mip_for(vol: &IrVolume) -> OccupancyMip {
        let dv = DecodedVolume {
            nx: vol.nx,
            ny: vol.ny,
            nz: vol.nz,
            z_min_m: vol.z_min_m,
            dz_m: vol.dz_m,
            horiz_pitch_m: vol.horiz_pitch_m,
            ext_liquid: vol.ext_liquid.clone(),
            ext_ice: vol.ext_ice.clone(),
            ext_precip: vol.ext_precip.clone(),
            tau_up: vec![0.0; vol.ext_liquid.len()],
        };
        OccupancyMip::build(&dv, OCCUPANCY_MIP_FACTOR)
    }

    /// The nadir view ray from a geostationary camera to the domain centre ground.
    fn nadir_ray(georef: &GridGeoref, cam: &CameraGeometry, gi: f64, gj: f64, dz: f64) -> [f64; 3] {
        let target = brick_to_ecef(georef, gi, gj, 0.0, 0.0, dz).unwrap();
        let d = [
            target[0] - cam.camera[0],
            target[1] - cam.camera[1],
            target[2] - cam.camera[2],
        ];
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        [d[0] / len, d[1] / len, d[2] / len]
    }

    #[test]
    fn clear_sky_bt_equals_skin_temperature() {
        // No cloud, no WV continuum, emissivity 1: the BT is exactly TSK (design
        // section 7 test 2). A warm ground plus a cool troposphere emits nothing in
        // the window (no absorber), so all the radiance is the surface term.
        let (nx, ny, nz) = (24, 24, 40);
        let dz = 250.0;
        let tsk = 298.0f32;
        let vol = build_ir_volume(nx, ny, nz, dz, 3000.0, tsk, |_, _, k| {
            let t = 288.0 - 6.5 * (k as f64 * dz / 1000.0); // lapse-rate air temp
            (0.0, 0.0, 0.0, t, 0.0)
        });
        let mip = mip_for(&vol);
        let georef = test_georef(nx, ny, 3000.0);
        let mut cfg = IrConfig::band13();
        cfg.wv_continuum = false;
        cfg.surface_emissivity = 1.0;
        let scene = IrScene {
            vol: &vol,
            mip: &mip,
            georef: &georef,
            cfg,
        };
        let cam = CameraGeometry::from_sub_lon(-100.0);
        let view = nadir_ray(
            &georef,
            &cam,
            (nx - 1) as f64 / 2.0,
            (ny - 1) as f64 / 2.0,
            dz,
        );
        let m = march_ir(&scene, cam.camera, view).expect("ray hits the brick");
        assert!(m.hit_surface, "the nadir ray must hit the ground");
        let bt = m.brightness_temperature(cfg.wavelength_m);
        assert!(
            (bt - tsk as f64).abs() < 0.5,
            "clear-sky BT {bt} != TSK {tsk}"
        );
        // With the weak WV continuum ON and a moist column, the BT drops a little
        // below TSK but stays close (a few K), never above.
        let mut cfg2 = IrConfig::band13();
        cfg2.surface_emissivity = 1.0;
        let vol2 = build_ir_volume(nx, ny, nz, dz, 3000.0, tsk, |_, _, k| {
            let z = k as f64 * dz;
            let t = 288.0 - 6.5 * (z / 1000.0);
            (0.0, 0.0, 0.0, t, 0.012 * (-z / 2500.0).exp())
        });
        let scene2 = IrScene {
            vol: &vol2,
            mip: &mip,
            georef: &georef,
            cfg: cfg2,
        };
        let bt2 = march_ir_bt(&scene2, cam.camera, view).unwrap();
        assert!(
            bt2 < tsk as f64 && bt2 > tsk as f64 - 5.0,
            "moist clear-sky BT {bt2} not a few K below TSK {tsk}"
        );
    }

    #[test]
    fn thick_anvil_bt_equals_cloud_top_temperature() {
        // A deep, optically thick ice anvil: the BT must equal the anvil's COLD TOP
        // temperature, not the warm ground (design section 7 test 1 / the section-10
        // proof standard). Anvil occupies the top half of the column at 215 K; warm
        // ground at 300 K.
        let (nx, ny, nz) = (24, 24, 60);
        let dz = 250.0;
        let cloud_top_t = 215.0f64;
        let vol = build_ir_volume(nx, ny, nz, dz, 3000.0, 300.0, |_, _, k| {
            if k >= nz / 2 {
                // thick ice anvil (visible ext 3e-2 m^-1 over ~7.5 km -> huge tau)
                (0.0, 3.0e-2, 0.0, cloud_top_t, 0.0)
            } else {
                (0.0, 0.0, 0.0, 280.0, 0.0)
            }
        });
        let mip = mip_for(&vol);
        let georef = test_georef(nx, ny, 3000.0);
        let mut cfg = IrConfig::band13();
        cfg.wv_continuum = false;
        let scene = IrScene {
            vol: &vol,
            mip: &mip,
            georef: &georef,
            cfg,
        };
        let cam = CameraGeometry::from_sub_lon(-100.0);
        let view = nadir_ray(
            &georef,
            &cam,
            (nx - 1) as f64 / 2.0,
            (ny - 1) as f64 / 2.0,
            dz,
        );
        let m = march_ir(&scene, cam.camera, view).unwrap();
        let bt = m.brightness_temperature(cfg.wavelength_m);
        assert!(
            (bt - cloud_top_t).abs() < 1.0,
            "thick-anvil BT {bt} != cloud-top T {cloud_top_t}"
        );
        // The anvil is IR-opaque: the surface transmittance is essentially zero, so
        // the warm ground never shows through.
        assert!(
            m.surface_transmittance < 1.0e-3,
            "anvil not opaque: surface_transmittance {}",
            m.surface_transmittance
        );
    }

    #[test]
    fn cloud_bt_is_between_and_monotone_in_optical_depth() {
        // A uniform cold cloud DECK over a warm ground: the BT must lie strictly
        // between the cloud-top T and TSK, and DECREASE monotonically as the cloud
        // optical depth increases (design section 7 test 3). The deck is several voxels
        // thick at a UNIFORM temperature so the emission-weighted temperature is the
        // deck temperature at every optical depth (a thin 1-2 voxel cloud blends its
        // sharp temperature edge trilinearly, and at high opacity the ray would sample
        // the warm cloud-top EDGE — not the physics under test; the fully-opaque
        // BT = cloud-top-T limit is the separate anvil test above).
        let (nx, ny, nz) = (20, 20, 40);
        let dz = 250.0;
        let tsk = 300.0f64;
        let cloud_t = 230.0f64;
        let georef = test_georef(nx, ny, 3000.0);
        let cam = CameraGeometry::from_sub_lon(-100.0);
        let view = nadir_ray(
            &georef,
            &cam,
            (nx - 1) as f64 / 2.0,
            (ny - 1) as f64 / 2.0,
            dz,
        );
        let mut prev = f64::INFINITY;
        let mut saw_between = false;
        // Stay in the semi-transparent-to-moderately-thick regime (max tau ~ 3.7 over
        // the 8-voxel deck, above the transmittance floor) so the sweep is cleanly
        // monotone.
        for &ext in &[1.0e-5, 5.0e-5, 1.0e-4, 3.0e-4, 1.0e-3] {
            // A uniform 8-voxel ice deck (k = nz-12..nz-4) at cloud_t. The temperature
            // is cloud_t for the whole cloud AND its immediate neighbours (k >= nz-14),
            // so the emitting cloud is surrounded by cloud_t: the trilinearly-sampled
            // emission temperature is cloud_t everywhere in the cloud regardless of where
            // the fine-step midpoints fall (no sharp temperature edge to alias).
            let vol = build_ir_volume(nx, ny, nz, dz, 3000.0, tsk as f32, |_, _, k| {
                let t = if k >= nz - 14 { cloud_t } else { 280.0 };
                let ice = if (nz - 12..nz - 4).contains(&k) {
                    ext
                } else {
                    0.0
                };
                (0.0, ice, 0.0, t, 0.0)
            });
            let mip = mip_for(&vol);
            let mut cfg = IrConfig::band13();
            cfg.wv_continuum = false;
            let scene = IrScene {
                vol: &vol,
                mip: &mip,
                georef: &georef,
                cfg,
            };
            let bt = march_ir_bt(&scene, cam.camera, view).unwrap();
            assert!(
                bt > cloud_t - 0.5 && bt < tsk + 0.5,
                "cloud BT {bt} not between {cloud_t} and {tsk}"
            );
            assert!(
                bt <= prev + 1.0e-6,
                "BT not monotone in optical depth at ext {ext}"
            );
            if bt > cloud_t + 5.0 && bt < tsk - 5.0 {
                saw_between = true;
            }
            prev = bt;
        }
        assert!(
            saw_between,
            "no genuinely-intermediate BT across the optical-depth sweep"
        );
    }

    #[test]
    fn surface_emissivity_below_one_lowers_the_clear_bt_slightly() {
        // A grey surface (emissivity 0.99) reads a hair colder than a black one.
        let (nx, ny, nz) = (16, 16, 30);
        let dz = 250.0;
        let tsk = 295.0f32;
        let vol = build_ir_volume(nx, ny, nz, dz, 3000.0, tsk, |_, _, _| {
            (0.0, 0.0, 0.0, 280.0, 0.0)
        });
        let mip = mip_for(&vol);
        let georef = test_georef(nx, ny, 3000.0);
        let cam = CameraGeometry::from_sub_lon(-100.0);
        let view = nadir_ray(
            &georef,
            &cam,
            (nx - 1) as f64 / 2.0,
            (ny - 1) as f64 / 2.0,
            dz,
        );
        let make = |eps: f64| {
            let mut cfg = IrConfig::band13();
            cfg.wv_continuum = false;
            cfg.surface_emissivity = eps;
            let scene = IrScene {
                vol: &vol,
                mip: &mip,
                georef: &georef,
                cfg,
            };
            march_ir_bt(&scene, cam.camera, view).unwrap()
        };
        let black = make(1.0);
        let grey = make(IR_SURFACE_EMISSIVITY);
        assert!((black - tsk as f64).abs() < 0.5);
        assert!(
            grey < black && grey > tsk as f64 - 3.0,
            "grey {grey} black {black}"
        );
        // Sanity: the emitted surface radiance scales with emissivity.
        let b = planck_radiance(tsk as f64, IR_BAND13_WAVELENGTH_M);
        assert!(b > 0.0);
    }

    #[test]
    fn off_earth_ray_has_no_ir_data() {
        // A ray that misses the brick shell (pointed away from the domain) returns
        // None (space -> the frame masks it NaN).
        let (nx, ny, nz) = (16, 16, 30);
        let vol = build_ir_volume(nx, ny, nz, 250.0, 3000.0, 295.0, |_, _, _| {
            (0.0, 0.0, 0.0, 280.0, 0.0)
        });
        let mip = mip_for(&vol);
        let georef = test_georef(nx, ny, 3000.0);
        let cam = CameraGeometry::from_sub_lon(-100.0);
        let scene = IrScene {
            vol: &vol,
            mip: &mip,
            georef: &georef,
            cfg: IrConfig::band13(),
        };
        // A view straight up/away from the earth from the satellite: misses the shell.
        let away = {
            let c = cam.camera;
            let len = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
            [c[0] / len, c[1] / len, c[2] / len] // pointing away from earth centre
        };
        assert!(march_ir(&scene, cam.camera, away).is_none());
    }

    #[test]
    fn frame_stats_report_cold_top_and_warm_ground() {
        // A frame with an anvil in one corner and clear warm ground elsewhere: the
        // min BT is the cold top, the max is the warm ground, all finite in domain.
        let bt = vec![
            215.0f32,
            f32::NAN,
            300.0,
            295.0,
            f32::NAN,
            230.0,
            298.0,
            290.0,
        ];
        let s = ir_frame_stats(&bt);
        assert_eq!(s.finite, 6);
        assert!((s.min_bt - 215.0).abs() < 1e-6);
        assert!((s.max_bt - 300.0).abs() < 1e-6);
        assert!(s.all_finite_in_domain);
        // An empty plane is handled without panicking.
        let empty = ir_frame_stats(&[f32::NAN, f32::NAN]);
        assert_eq!(empty.finite, 0);
    }

    // ── water-vapor bands (ABI 8/9/10) — the WV march is the SAME march with a WV
    // `IrConfig` from `wv::WvBand` (design section 7 / the WV addendum). These reuse the
    // IR test scaffolding above; the only difference is `scene.cfg`. ─────────────────

    use crate::wv::WvBand;

    /// The centre nadir BT for a given brick-fill + `IrConfig` (the shared scaffolding).
    fn nadir_bt(
        nx: usize,
        ny: usize,
        nz: usize,
        dz: f64,
        tsk: f32,
        cfg: IrConfig,
        fill: impl Fn(usize, usize, usize) -> (f64, f64, f64, f64, f64),
    ) -> IrMarch {
        let vol = build_ir_volume(nx, ny, nz, dz, 3000.0, tsk, fill);
        let mip = mip_for(&vol);
        let georef = test_georef(nx, ny, 3000.0);
        let cam = CameraGeometry::from_sub_lon(-100.0);
        let view = nadir_ray(
            &georef,
            &cam,
            (nx - 1) as f64 / 2.0,
            (ny - 1) as f64 / 2.0,
            dz,
        );
        let scene = IrScene {
            vol: &vol,
            mip: &mip,
            georef: &georef,
            cfg,
        };
        march_ir(&scene, cam.camera, view).expect("ray hits the brick")
    }

    /// A moist / dry lapse-rate column (no cloud): temperature `288 - 6.5 z[km]`, vapor
    /// `q0 exp(-z/2500)` — the standard exponential moisture profile the weighting
    /// function integrates against.
    fn column_fill(q0: f64) -> impl Fn(usize, usize, usize) -> (f64, f64, f64, f64, f64) {
        let dz = 250.0;
        move |_, _, k| {
            let z = k as f64 * dz;
            let t = 288.0 - 6.5 * (z / 1000.0);
            (0.0, 0.0, 0.0, t, q0 * (-z / 2500.0).exp())
        }
    }

    #[test]
    fn wv_moist_upper_column_is_colder_than_a_dry_one_in_62() {
        // A MOIST column emits from the cold UPPER troposphere in 6.2 um (high weighting
        // function) -> a COLD BT; a DRY column lets 6.2 see deeper/warmer -> a WARMER BT.
        // This is the headline WV physics: 6.2 tracks upper-level moisture.
        let (nx, ny, nz, dz) = (16, 16, 60, 250.0);
        let cfg = WvBand::Upper.ir_config();
        let moist = nadir_bt(nx, ny, nz, dz, 300.0, cfg, column_fill(0.014))
            .brightness_temperature(cfg.wavelength_m);
        let dry = nadir_bt(nx, ny, nz, dz, 300.0, cfg, column_fill(0.002))
            .brightness_temperature(cfg.wavelength_m);
        assert!(
            moist < dry - 5.0,
            "moist 6.2 BT {moist} should be well below dry {dry}"
        );
        // Both are in the classic (cold) WV range, never at the warm skin temperature.
        assert!(moist < 250.0, "moist 6.2 BT {moist} not cold-troposphere");
        assert!(
            (200.0..300.0).contains(&moist) && (200.0..300.0).contains(&dry),
            "WV BTs {moist}/{dry} out of a plausible range"
        );
    }

    #[test]
    fn wv_62_is_colder_than_73_for_the_same_moist_column() {
        // For one moist column, the more strongly absorbing 6.2 um band emits from
        // HIGHER (colder) than 7.3 um (weighting-function ordering): BT_6.2 < BT_6.9 <
        // BT_7.3. This is the upper/mid/lower-level moisture separation.
        let (nx, ny, nz, dz) = (16, 16, 60, 250.0);
        let bt = |band: WvBand| {
            let cfg = band.ir_config();
            nadir_bt(nx, ny, nz, dz, 300.0, cfg, column_fill(0.013))
                .brightness_temperature(cfg.wavelength_m)
        };
        let (b62, b69, b73) = (bt(WvBand::Upper), bt(WvBand::Mid), bt(WvBand::Low));
        assert!(b62 < b69, "6.2 BT {b62} !< 6.9 BT {b69}");
        assert!(b69 < b73, "6.9 BT {b69} !< 7.3 BT {b73}");
        // All three are colder than the surface (WV never sees the warm ground here).
        assert!(b73 < 295.0, "7.3 BT {b73} should still be sub-surface");
    }

    #[test]
    fn wv_thick_cloud_top_dominates_like_the_window() {
        // Cloud ice is opaque in the WV bands too (the same cloud IR absorption): a thick
        // anvil top over a moist column drives the transmittance to ~0 within its cold
        // top, so the WV BT equals the cloud-top temperature — cloud opacity dominates the
        // vapor emission, exactly as band 13.
        let (nx, ny, nz, dz) = (16, 16, 60, 250.0);
        let cloud_top_t = 218.0f64;
        let cfg = WvBand::Low.ir_config(); // the WARMEST WV band; still reads the cold top
        let fill = move |_: usize, _: usize, k: usize| {
            let z = k as f64 * dz;
            let t = 288.0 - 6.5 * (z / 1000.0);
            let q = 0.012 * (-z / 2500.0).exp();
            if k >= nz / 2 {
                (0.0, 3.0e-2, 0.0, cloud_top_t, q) // thick ice anvil, cold top
            } else {
                (0.0, 0.0, 0.0, t, q)
            }
        };
        let m = nadir_bt(nx, ny, nz, dz, 300.0, cfg, fill);
        let bt = m.brightness_temperature(cfg.wavelength_m);
        assert!(
            (bt - cloud_top_t).abs() < 2.0,
            "WV cloud-top BT {bt} != anvil top {cloud_top_t}"
        );
        assert!(
            m.surface_transmittance < 1.0e-3,
            "anvil not opaque in WV: transmittance {}",
            m.surface_transmittance
        );
    }

    // ── WS1 march-physics: the sub-terrain vapor clip + the TSK fallback (both
    // live in IrVolume::from_brick, so these tests build real BRICKS) ──────────

    use crate::bricks::{ChannelQuant, LogQuant, encode_log_channel, encode_temperature_celsius};
    use std::collections::BTreeMap;

    /// Build a synthetic brick with caller-supplied per-cell `(qvapor, kelvin)`, a
    /// uniform terrain height and TSK plane (mirrors derived.rs's test builder).
    fn build_brick(
        nx: usize,
        ny: usize,
        nz: usize,
        dz: f64,
        hgt_m: f32,
        tsk_k: f32,
        fill: impl Fn(usize, usize, usize) -> (f32, f32),
    ) -> VolumeBrick {
        let n3 = nx * ny * nz;
        let n2 = nx * ny;
        let mut qvapor = vec![0.0f32; n3];
        let mut kelvin = vec![0.0f32; n3];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let (q, t) = fill(i, j, k);
                    let c = (k * ny + j) * nx + i;
                    qvapor[c] = q;
                    kelvin[c] = t;
                }
            }
        }
        let (qv, qvapor) = encode_log_channel(&qvapor);
        let zero = LogQuant {
            vmin: 0.0,
            vmax: 0.0,
        };
        let mut map = BTreeMap::new();
        map.insert("ext_liquid".to_string(), zero);
        map.insert("ext_ice".to_string(), zero);
        map.insert("ext_precip".to_string(), zero);
        map.insert("tau_up".to_string(), zero);
        map.insert("qvapor".to_string(), qv);
        VolumeBrick {
            nx,
            ny,
            nz,
            z_min_m: 0.0,
            dz_m: dz,
            time_iso: None,
            quant: ChannelQuant(map),
            ext_liquid: vec![0u8; n3],
            ext_ice: vec![0u8; n3],
            ext_precip: vec![0u8; n3],
            tau_up: vec![0u8; n3],
            qvapor,
            temperature_f16: encode_temperature_celsius(&kelvin),
            hgt: vec![hgt_m; n2],
            landmask: vec![1.0f32; n2],
            tsk: vec![tsk_k; n2],
            u10: vec![0.0f32; n2],
            v10: vec![0.0f32; n2],
            snowh: None,
            ivgtyp: None,
        }
    }

    /// The centre nadir IR march of a brick through `from_brick` (the WS1 decode
    /// path under test), with the caller's `IrConfig`.
    fn nadir_bt_of_brick(brick: &VolumeBrick, cfg: IrConfig) -> IrMarch {
        let vol = IrVolume::from_brick(brick, 3000.0);
        let mip = mip_for(&vol);
        let georef = test_georef(brick.nx, brick.ny, 3000.0);
        let cam = CameraGeometry::from_sub_lon(-100.0);
        let view = nadir_ray(
            &georef,
            &cam,
            (brick.nx - 1) as f64 / 2.0,
            (brick.ny - 1) as f64 / 2.0,
            brick.dz_m,
        );
        let scene = IrScene {
            vol: &vol,
            mip: &mip,
            georef: &georef,
            cfg,
        };
        march_ir(&scene, cam.camera, view).expect("ray hits the brick")
    }

    #[test]
    fn wv_march_clips_sub_terrain_vapor() {
        // A realistic EXPONENTIAL moisture column over 1000 m terrain, with the
        // ingest's CLAMPED sub-terrain fill: the march must see the SAME water
        // vapor as the same encoded brick with its sub-terrain codes explicitly
        // zeroed (identical quant grid) — the clamped sub-terrain vapor is
        // fictitious air. Band 10 (7.3 um, the deepest-seeing WV band) read ~0.6 K
        // cold from it in this scene before the WS1 clip (measured on the
        // fail-before probe at ec80e88); a NON-uniform profile matters — a very
        // moist uniform column is WV-opaque far above the terrain and hides the bug.
        let (nx, ny, nz, dz) = (16usize, 16usize, 60usize, 250.0f64);
        let q0 = 0.006f64;
        let temp = |k: usize| (288.0 - 6.5 * (k as f64 * dz / 1000.0)) as f32;
        let q_at = |z: f64| (q0 * (-z / 2500.0).exp()) as f32;
        let surface = 1000.0f64; // exactly 4 layers: the clip zeroes k = 0..3 fully
        let elevated = build_brick(nx, ny, nz, dz, surface as f32, 295.0, |_, _, k| {
            let z = k as f64 * dz;
            // The ingest clamps sub-terrain levels to the lowest-above-terrain value.
            let q = if z < surface { q_at(surface) } else { q_at(z) };
            (q, temp(k))
        });
        let mut reference = elevated.clone();
        reference.hgt.iter_mut().for_each(|h| *h = 0.0);
        for k in 0..4 {
            for c in 0..nx * ny {
                reference.qvapor[k * nx * ny + c] = 0; // code 0 decodes to exactly 0
            }
        }
        let cfg = WvBand::Low.ir_config();
        let bt_elev = nadir_bt_of_brick(&elevated, cfg).brightness_temperature(cfg.wavelength_m);
        let bt_ref = nadir_bt_of_brick(&reference, cfg).brightness_temperature(cfg.wavelength_m);
        assert!(
            (bt_elev - bt_ref).abs() < 0.1,
            "terrain-clipped BT {bt_elev} != zeroed-sub-terrain reference {bt_ref}"
        );
    }

    #[test]
    fn sea_level_brick_decodes_bit_identical_qvapor() {
        // hgt = 0 (<= z_min) must leave the decoded qvapor untouched — the clip is
        // a no-op for a sea-level column (the anchor for every existing IR result).
        let (nx, ny, nz, dz) = (8usize, 8usize, 30usize, 250.0f64);
        let brick = build_brick(nx, ny, nz, dz, 0.0, 295.0, |_, _, k| {
            (0.012 * (-(k as f32) * 0.05).exp(), 280.0)
        });
        let vol = IrVolume::from_brick(&brick, 3000.0);
        let qv = brick.quant.get("qvapor");
        let raw: Vec<f32> = brick.qvapor.iter().map(|&c| qv.decode(c)).collect();
        assert_eq!(vol.qvapor, raw, "hgt=0 decode must be bit-identical");
        // And the valid TSK plane is carried through verbatim.
        assert_eq!(vol.tsk, brick.tsk);
    }

    #[test]
    fn tsk_missing_falls_back_to_lowest_level_air_temperature() {
        // An ALL-ZERO TSK plane (the ingest zero-fills a TSK-less wrfout) used to
        // emit B(0 K) = 0 from the ground -> a deep-space-cold clear column. The
        // WS1 fallback substitutes the per-column lowest-level air temperature.
        let (nx, ny, nz, dz) = (12usize, 12usize, 30usize, 250.0f64);
        let t0 = 285.0f32;
        let brick = build_brick(nx, ny, nz, dz, 0.0, 0.0, |_, _, k| {
            (0.0, t0 - 6.5 * (k as f32 * dz as f32 / 1000.0))
        });
        let vol = IrVolume::from_brick(&brick, 3000.0);
        // The substituted plane is the k = 0 air temperature (f16 quantization slop).
        for j in 0..ny {
            for i in 0..nx {
                let want = vol.temperature_k[j * nx + i];
                let got = vol.tsk[j * nx + i];
                assert!(
                    (got - want).abs() < 0.1,
                    "fallback TSK {got} != lowest-level air temp {want}"
                );
            }
        }
        assert!(
            tsk_fallback_engaged(),
            "the process-wide TSK-fallback diagnostic must be set"
        );
        // A clear-column BT now reads within a few K of the lowest-level air temp
        // (not deep-space cold).
        let mut cfg = IrConfig::band13();
        cfg.wv_continuum = false;
        cfg.surface_emissivity = 1.0;
        let bt = nadir_bt_of_brick(&brick, cfg).brightness_temperature(cfg.wavelength_m);
        assert!(
            (bt - t0 as f64).abs() < 3.0,
            "clear-column BT {bt} should be near the surface air temp {t0}"
        );
    }
}
