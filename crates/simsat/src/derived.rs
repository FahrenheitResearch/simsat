//! Derived scalar-field map products (the last of the product suite).
//!
//! Three TOP-DOWN, map-registered SCALAR fields computed directly from the brick, for a
//! downstream meteorologist end-user to plot with their own colormaps. Each
//! is a per-column `(i, j)` VERTICAL integral / march through the brick — they are
//! intrinsically view-independent column quantities (a column of water vapor, a column
//! optical depth, the cloud-top level a satellite sees), NOT a slant-ray render — so the
//! natural, physically-correct, and testable computation is a per-column native 2-D field
//! ([`compute_field`]) that is then resampled onto the output raster ([`resample_field`])
//! via the raster's per-pixel fractional WRF indices. The RAW physical `f32` field is the
//! primary deliverable; a basic studio colormap ([`colorize`]) is provided for the in-app
//! display, but the meteorologist plots the raw array with his own colormaps/cartopy.
//!
//! The three fields (design section 2 — the brick channels + the affine vertical axis
//! `z(k) = z_min_m + k*dz_m`, MSL):
//!
//! 1. PRECIPITABLE WATER (mm): the vertically-integrated water-vapor column,
//!    `PW = sum_k rho_air(z_k) * qvapor(i,j,k) * dz`. Density is the standard-atmosphere
//!    exponential [`optics::standard_air_density_kg_m3`] (the brick carries no pressure) —
//!    the SAME density kernel `ir.rs`'s water-vapor continuum uses, so PW is consistent with
//!    the WV bands. Units: `kg m^-2 == mm`. `qvapor` is a mixing ratio; `rho_air * mixing`
//!    slightly overestimates vs `rho_dry * r` (a documented ~<2% simplification, matching
//!    the optics convention). Typical values 5-60 mm.
//! 2. CLOUD-TOP TEMPERATURE (K): march DOWN from the brick top accumulating VISIBLE cloud
//!    optical depth (`total extinction = ext_liquid + ext_ice + ext_precip`, the SAME
//!    extinction the cloud march uses); the effective cloud top is the level where the
//!    cumulative optical depth crosses [`CLOUD_TOP_TAU`] (~1, the level a satellite sees).
//!    Report the temperature (Kelvin) at that level. A clear / optically-thin column (column
//!    OD below the threshold) has NO cloud top -> `NaN` (the cleanest no-cloud sentinel for a
//!    plotter). This is close to the IR brightness temperature for a thick cloud but is the
//!    actual physical cloud-top temperature at the visible tau~1 surface.
//! 3. CLOUD OPTICAL DEPTH (dimensionless): the total-column visible optical depth,
//!    `COD = sum_k total_ext(i,j,k) * dz`, integrated DIRECTLY over the column. (The brick's
//!    `tau_up` channel at the surface level `k = 0` is the equivalent PRECOMPUTED full-column
//!    value, but it is log-quantized u8 — lossy under accumulation — so the direct integral
//!    of the decoded extinction is used instead, matching the PW integration style and giving
//!    a clean analytic test.) Clear -> 0; a thick storm core -> tens+.

use crate::bricks::{StorageProfile, VolumeBrick, decode_log2_f16, decode_temperature_kelvin};
use crate::ir_enhance::{IrEnhancement, bt_to_rgba};
use crate::optics::{HydrometeorClass, RHO_W, standard_air_density_kg_m3};

/// The cumulative VISIBLE optical depth at which the effective cloud top is placed (the
/// level a satellite sees; design section 2). Marching down from the brick top, the first
/// level where the accumulated optical depth crosses this value is the cloud top.
pub const CLOUD_TOP_TAU: f64 = 1.0;

/// Display range top (mm) for the precipitable-water colormap (the raw array is unclamped;
/// this only normalises the studio moisture ramp). A moist tropical column reaches ~60-70 mm.
pub const PW_DISPLAY_MAX_MM: f32 = 70.0;

/// Display range top (dimensionless) for the cloud-optical-depth colormap (the raw array is
/// unclamped; this only normalises the studio ramp). Thick convective cores exceed this and
/// saturate to the dark end.
pub const COD_DISPLAY_MAX: f32 = 80.0;

/// Which derived scalar field to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivedField {
    /// Precipitable water — the vertically-integrated water-vapor column (mm).
    PrecipitableWater,
    /// Cloud-top temperature — the temperature at the visible tau~1 level (K; `NaN` = clear).
    CloudTopTemp,
    /// Cloud optical depth — the total-column visible optical depth (dimensionless).
    CloudOpticalDepth,
}

impl DerivedField {
    /// All fields in UI order.
    pub const ALL: [DerivedField; 3] = [
        Self::PrecipitableWater,
        Self::CloudTopTemp,
        Self::CloudOpticalDepth,
    ];

    /// Human-readable label for the picker / status line.
    pub fn label(self) -> &'static str {
        match self {
            Self::PrecipitableWater => "Precipitable Water",
            Self::CloudTopTemp => "Cloud-Top Temp",
            Self::CloudOpticalDepth => "Cloud Optical Depth",
        }
    }

    /// The physical units of the raw field.
    pub fn units(self) -> &'static str {
        match self {
            Self::PrecipitableWater => "mm",
            Self::CloudTopTemp => "K",
            Self::CloudOpticalDepth => "", // dimensionless
        }
    }

    /// Stable slug (the CLI / binding token).
    pub fn slug(self) -> &'static str {
        match self {
            Self::PrecipitableWater => "pw",
            Self::CloudTopTemp => "ctt",
            Self::CloudOpticalDepth => "cod",
        }
    }

    /// Parse a slug (accepts a few friendly aliases); `None` for an unknown token.
    pub fn parse(value: &str) -> Option<Self> {
        match value
            .trim()
            .to_ascii_lowercase()
            .replace(['-', '_', ' '], "")
            .as_str()
        {
            "pw" | "pwat" | "precipitablewater" | "tpw" => Some(Self::PrecipitableWater),
            "ctt" | "cloudtoptemp" | "cloudtoptemperature" | "ctop" => Some(Self::CloudTopTemp),
            "cod" | "tau" | "opticaldepth" | "cloudopticaldepth" => Some(Self::CloudOpticalDepth),
            _ => None,
        }
    }
}

// ── per-column native field computation ─────────────────────────────────────────

#[inline]
fn cell3(nx: usize, ny: usize, i: usize, j: usize, k: usize) -> usize {
    (k * ny + j) * nx + i
}

/// Compute the requested derived field as a NATIVE per-column 2-D field (`nx*ny`, grid
/// orientation `j*nx + i`, `j = 0` south / `i = 0` west), directly from the decoded brick.
/// This is the RAW physical field before it is resampled onto an output raster.
pub fn compute_field(brick: &VolumeBrick, field: DerivedField) -> Vec<f32> {
    match field {
        DerivedField::PrecipitableWater => precipitable_water_field(brick),
        DerivedField::CloudTopTemp => cloud_top_temp_field(brick),
        DerivedField::CloudOpticalDepth => cloud_optical_depth_field(brick),
    }
}

/// Precipitable water (mm) per column: `PW = integral_{surface}^{top} rho_air(z) * qvapor(z) dz`,
/// with the standard-atmosphere density (the brick has no pressure) at the MSL level height
/// `z_k = z_min_m + k*dz_m`. `kg m^-2 == mm`.
///
/// The integration starts at the TERRAIN SURFACE (`brick.hgt`), not at MSL `z = 0`: the ingest
/// fills brick levels below the lowest WRF model level (≈ the terrain) with the CLAMPED surface
/// vapor (`Extrap::ClampEdge`), so integrating the full MSL column would add spurious vapor below
/// elevated terrain (e.g. ~8 mm below a 440 m surface). Each layer's contribution is clipped to
/// its portion above the surface; for a sea-level column (`hgt = 0`) this reduces EXACTLY to the
/// plain `sum_k rho_air(z_k) * qvapor(k) * dz`.
pub fn precipitable_water_field(brick: &VolumeBrick) -> Vec<f32> {
    let (nx, ny, nz) = (brick.nx, brick.ny, brick.nz);
    let qv = brick.quant.get("qvapor");
    let dz = brick.dz_m;
    let z0 = brick.z_min_m;
    // Precompute the per-level (left-edge) density rho_air(z_k) — the same for every column.
    let rho: Vec<f64> = (0..nz)
        .map(|k| standard_air_density_kg_m3(z0 + k as f64 * dz))
        .collect();
    let mut out = vec![0.0f32; nx * ny];
    for j in 0..ny {
        for i in 0..nx {
            let surface = brick.hgt[j * nx + i] as f64;
            let mut pw = 0.0f64;
            for (k, &r) in rho.iter().enumerate() {
                let zb = z0 + k as f64 * dz; // this layer is [zb, zb+dz)
                // The layer thickness ABOVE the terrain surface (dz for a fully-above layer,
                // partial for the layer straddling the surface, 0 below it).
                let thickness = (zb + dz - zb.max(surface)).max(0.0);
                if thickness <= 0.0 {
                    continue;
                }
                let q = qv.decode(brick.qvapor[cell3(nx, ny, i, j, k)]) as f64;
                if q > 0.0 {
                    pw += r * q * thickness;
                }
            }
            out[j * nx + i] = pw as f32;
        }
    }
    out
}

/// Cloud-top temperature (K) per column: march DOWN from the brick top accumulating the
/// visible optical depth of each layer (`total_ext * dz`); the first level where the
/// cumulative optical depth crosses [`CLOUD_TOP_TAU`] is the effective cloud top — report the
/// temperature there. A column whose full-column optical depth never reaches the threshold
/// has no cloud top -> `NaN`.
pub fn cloud_top_temp_field(brick: &VolumeBrick) -> Vec<f32> {
    let (nx, ny, nz) = (brick.nx, brick.ny, brick.nz);
    let ql = brick.quant.get("ext_liquid");
    let qi = brick.quant.get("ext_ice");
    let qp = brick.quant.get("ext_precip");
    let dz = brick.dz_m;
    let temp_k = decode_temperature_kelvin(&brick.temperature_f16);
    let science = (brick.storage_profile == StorageProfile::ScienceCloudF16)
        .then_some(brick.science_cloud_f16.as_ref())
        .flatten();
    let mut out = vec![f32::NAN; nx * ny];
    for j in 0..ny {
        for i in 0..nx {
            let mut cum = 0.0f64;
            // Top (k = nz-1) down to the surface (k = 0).
            for k in (0..nz).rev() {
                let c = cell3(nx, ny, i, j, k);
                let ext = science.map_or_else(
                    || {
                        ql.decode(brick.ext_liquid[c]) as f64
                            + qi.decode(brick.ext_ice[c]) as f64
                            + qp.decode(brick.ext_precip[c]) as f64
                    },
                    |payload| {
                        decode_log2_f16(payload.ext_liquid[c]) as f64
                            + decode_log2_f16(payload.ext_ice[c]) as f64
                            + decode_log2_f16(payload.ext_precip[c]) as f64
                    },
                );
                let layer_od = ext.max(0.0) * dz;
                if cum + layer_od >= CLOUD_TOP_TAU {
                    out[j * nx + i] = temp_k[c];
                    break;
                }
                cum += layer_od;
            }
        }
    }
    out
}

/// Cloud optical depth (dimensionless) per column: `COD = sum_k total_ext(i,j,k) * dz`, the
/// full-column visible optical depth integrated directly over the decoded extinction. Clear
/// -> 0.
pub fn cloud_optical_depth_field(brick: &VolumeBrick) -> Vec<f32> {
    let (nx, ny, nz) = (brick.nx, brick.ny, brick.nz);
    let ql = brick.quant.get("ext_liquid");
    let qi = brick.quant.get("ext_ice");
    let qp = brick.quant.get("ext_precip");
    let dz = brick.dz_m;
    let science = (brick.storage_profile == StorageProfile::ScienceCloudF16)
        .then_some(brick.science_cloud_f16.as_ref())
        .flatten();
    let mut out = vec![0.0f32; nx * ny];
    for j in 0..ny {
        for i in 0..nx {
            let mut tau = 0.0f64;
            for k in 0..nz {
                let c = cell3(nx, ny, i, j, k);
                let ext = science.map_or_else(
                    || {
                        ql.decode(brick.ext_liquid[c]) as f64
                            + qi.decode(brick.ext_ice[c]) as f64
                            + qp.decode(brick.ext_precip[c]) as f64
                    },
                    |payload| {
                        decode_log2_f16(payload.ext_liquid[c]) as f64
                            + decode_log2_f16(payload.ext_ice[c]) as f64
                            + decode_log2_f16(payload.ext_precip[c]) as f64
                    },
                );
                tau += ext.max(0.0) * dz;
            }
            out[j * nx + i] = tau as f32;
        }
    }
    out
}

// ── synthetic condensate cloud mask (validator regime support) ───────────────────

/// Default condensate mixing-ratio threshold (kg kg^-1) above which a column is
/// "cloudy" for the synthetic cloud mask. 1e-6 kg kg^-1 is the customary minimum
/// condensate floor of model cloud diagnostics (the order used by UPP / WRF cloud-
/// fraction and cloud-top routines to separate cloud from numerical trace
/// condensate; e.g. `qc + qi > 1e-6` in NCEP UPP `CALCLDFRA`-style tests). It is a
/// DEFINITION for a categorical mask, not a radiative quantity: a column at exactly
/// this floor is optically negligible at 10.3 um (see [`condensate_mixing_ratio_kg_kg`]
/// for the inverse optics that recover kg kg^-1 from the brick).
pub const CONDENSATE_CLOUD_MASK_THRESHOLD_KG_KG: f64 = 1.0e-6;

/// Cloud-mask code: the column carries no condensate above the threshold.
pub const CLOUD_MASK_CLEAR: u8 = 0;
/// Cloud-mask code: the column carries condensate above the threshold.
pub const CLOUD_MASK_CLOUDY: u8 = 1;
/// Cloud-mask code: no data (an output pixel outside the model domain).
pub const CLOUD_MASK_NO_DATA: u8 = 255;

/// Hydrometeor mixing ratio (kg kg^-1) of one class recovered from the brick's stored
/// VISIBLE geometric-optics extinction (m^-1) at MSL height `z_m`. This is the EXACT
/// algebraic inverse of the ingest's forward conversion
/// [`crate::optics::extinction_coefficient`], `beta = (3/2) rho_air q / (rho_w r_e)`:
///
/// `q = beta * rho_w * r_e / ((3/2) * rho_air(z))`
///
/// with the class's rho_w-normalized effective radius `r_e`
/// ([`HydrometeorClass::effective_radius_m`]: liquid 10 um, ice 40 um, snow 150 um,
/// rain/graupel 1 mm — the SAME table the ingest used to build the brick) and
/// [`RHO_W`] = 1000 kg m^-3. The brick carries no pressure, so `rho_air` is the
/// standard-atmosphere exponential [`standard_air_density_kg_m3`] — the same density
/// kernel `precipitable_water_field` and the IR water-vapor continuum use — whereas
/// the ingest used the WRF ideal-gas density (`optics::air_density`). The two differ
/// by up to ~10 % in a real column, so a recovered mixing ratio is the ingest's to
/// that tolerance (a documented approximation; it moves the mask threshold, not the
/// physics).
#[inline]
pub fn condensate_mixing_ratio_kg_kg(ext_vis_per_m: f64, class: HydrometeorClass, z_m: f64) -> f64 {
    if ext_vis_per_m <= 0.0 {
        return 0.0;
    }
    let rho_air = standard_air_density_kg_m3(z_m);
    ext_vis_per_m * RHO_W * class.effective_radius_m() / (1.5 * rho_air)
}

/// Per-column synthetic CLOUD MASK (`nx*ny` u8, grid orientation `j*nx + i`, `j = 0`
/// south): [`CLOUD_MASK_CLOUDY`] where ANY voxel of the column carries total condensate
/// mixing ratio `q_c + q_i + q_s + q_(r+g)` above `threshold_kg_kg`, else
/// [`CLOUD_MASK_CLEAR`]. Each class is recovered from its own extinction channel by
/// [`condensate_mixing_ratio_kg_kg`]: `ext_liquid` -> cloud liquid, `ext_ice` -> cloud
/// ice, the snow-only auxiliary `ext_snow` -> snow, and the remainder
/// `ext_precip - ext_snow` -> rain/graupel (both at the shared 1 mm radius, so the
/// mixed channel inverts without ambiguity). A brick whose `ext_snow` scale is zero (a
/// pre-SSB-v6 or programmatic brick) inverts all of `ext_precip` at the rain radius —
/// which OVERSTATES the mass of any snow in that channel (1 mm vs 150 um, 6.7x), i.e.
/// errs toward "cloudy" for the mixed channel; documented, and moot for SSB v6 bricks.
///
/// Floor honesty: the compact brick log-quantizes each channel over five decades below
/// its per-volume peak (`bricks::QUANT_DYNAMIC_RANGE`), so condensate below
/// `peak_ext / 1e5` decodes as zero. For a real convective brick (liquid peak ~0.1-0.3
/// m^-1) that floor is ~1e-8 kg kg^-1 — two decades under the default threshold — so
/// the quantization does not move the mask; a brick with an extreme peak could raise
/// the effective floor and is not detected here. Below-terrain levels are zero by
/// construction (the ingest extrapolates extinction with `Extrap::Zero`).
pub fn condensate_cloud_mask_field(brick: &VolumeBrick, threshold_kg_kg: f64) -> Vec<u8> {
    let (nx, ny, nz) = (brick.nx, brick.ny, brick.nz);
    let ql = brick.quant.get("ext_liquid");
    let qi = brick.quant.get("ext_ice");
    let qs = brick.quant.get("ext_snow");
    let qp = brick.quant.get("ext_precip");
    let dz = brick.dz_m;
    let z0 = brick.z_min_m;
    let science = (brick.storage_profile == StorageProfile::ScienceCloudF16)
        .then_some(brick.science_cloud_f16.as_ref())
        .flatten();
    let mut out = vec![CLOUD_MASK_CLEAR; nx * ny];
    for j in 0..ny {
        for i in 0..nx {
            let mut cloudy = false;
            for k in 0..nz {
                let c = cell3(nx, ny, i, j, k);
                let (liquid, ice, snow, precip) = science.map_or_else(
                    || {
                        (
                            ql.decode(brick.ext_liquid[c]) as f64,
                            qi.decode(brick.ext_ice[c]) as f64,
                            qs.decode(brick.ext_snow[c]) as f64,
                            qp.decode(brick.ext_precip[c]) as f64,
                        )
                    },
                    |payload| {
                        (
                            decode_log2_f16(payload.ext_liquid[c]) as f64,
                            decode_log2_f16(payload.ext_ice[c]) as f64,
                            decode_log2_f16(payload.ext_snow[c]) as f64,
                            decode_log2_f16(payload.ext_precip[c]) as f64,
                        )
                    },
                );
                if liquid <= 0.0 && ice <= 0.0 && precip <= 0.0 {
                    continue;
                }
                let z = z0 + k as f64 * dz;
                let q = condensate_mixing_ratio_kg_kg(liquid, HydrometeorClass::CloudLiquid, z)
                    + condensate_mixing_ratio_kg_kg(ice, HydrometeorClass::Ice, z)
                    + condensate_mixing_ratio_kg_kg(snow, HydrometeorClass::Snow, z)
                    + condensate_mixing_ratio_kg_kg(
                        (precip - snow).max(0.0),
                        HydrometeorClass::Rain,
                        z,
                    );
                if q > threshold_kg_kg {
                    cloudy = true;
                    break;
                }
            }
            if cloudy {
                out[j * nx + i] = CLOUD_MASK_CLOUDY;
            }
        }
    }
    out
}

/// Resample a native per-column u8 mask (`nx*ny`, grid orientation) onto an output raster
/// (`out_nx*out_ny`, row 0 = north) exactly like [`resample_field`] (nearest via the
/// raster's fractional WRF indices; exact `grid_i == px`, `grid_j == ny-1-py` for the
/// top-down native map), with [`CLOUD_MASK_NO_DATA`] where the grid index is non-finite.
pub fn resample_mask(
    mask: &[u8],
    nx: usize,
    ny: usize,
    grid_i: &[f32],
    grid_j: &[f32],
    out_nx: usize,
    out_ny: usize,
) -> Vec<u8> {
    let mut out = vec![CLOUD_MASK_NO_DATA; out_nx * out_ny];
    for idx in 0..out_nx * out_ny {
        let (gi, gj) = (grid_i[idx], grid_j[idx]);
        if !gi.is_finite() || !gj.is_finite() {
            continue;
        }
        let i = (gi.round() as i64).clamp(0, nx as i64 - 1) as usize;
        let j = (gj.round() as i64).clamp(0, ny as i64 - 1) as usize;
        out[idx] = mask[j * nx + i];
    }
    out
}

// ── resample the native column field onto an output raster ──────────────────────

/// Resample a native per-column field (`nx*ny`, grid orientation) onto an output raster of
/// `out_nx*out_ny` (row 0 = north), NEAREST-neighbor via the raster's per-pixel fractional
/// WRF indices `grid_i`/`grid_j` (`NaN` off-domain). For the TOP-DOWN native map raster the
/// mapping is exact (`grid_i == px`, `grid_j == ny-1-py`); for the from-space geostationary
/// raster the indices are fractional and nearest is a documented approximation (these are
/// map-registered scalar fields — the top-down native view is the primary product). A
/// no-data output pixel (grid index non-finite) stays `NaN`, EXCEPT a `NaN` input value (a
/// clear cloud-top column) is carried through as `NaN` on a valid grid index.
pub fn resample_field(
    field: &[f32],
    nx: usize,
    ny: usize,
    grid_i: &[f32],
    grid_j: &[f32],
    out_nx: usize,
    out_ny: usize,
) -> Vec<f32> {
    let mut out = vec![f32::NAN; out_nx * out_ny];
    for idx in 0..out_nx * out_ny {
        let (gi, gj) = (grid_i[idx], grid_j[idx]);
        if !gi.is_finite() || !gj.is_finite() {
            continue; // off-domain / off-earth -> no data
        }
        let i = (gi.round() as i64).clamp(0, nx as i64 - 1) as usize;
        let j = (gj.round() as i64).clamp(0, ny as i64 - 1) as usize;
        out[idx] = field[j * nx + i];
    }
    out
}

// ── basic studio colormaps (raw array is the primary deliverable) ───────────────

#[inline]
fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t)
        .round()
        .clamp(0.0, 255.0) as u8
}

/// Linear multi-stop ramp: `t` in `[0, 1]`, `stops` ascending by position with a stop at
/// 0.0 and 1.0. Returns the interpolated `[r, g, b]`.
fn ramp(t: f32, stops: &[(f32, [u8; 3])]) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    for w in stops.windows(2) {
        let (p0, c0) = w[0];
        let (p1, c1) = w[1];
        if t <= p1 {
            let f = if p1 > p0 { (t - p0) / (p1 - p0) } else { 0.0 };
            return [
                lerp_u8(c0[0], c1[0], f),
                lerp_u8(c0[1], c1[1], f),
                lerp_u8(c0[2], c1[2], f),
            ];
        }
    }
    stops[stops.len() - 1].1
}

/// The precipitable-water moisture ramp: dry (brown) -> tan -> green -> blue -> moist
/// (purple), over `0..`[`PW_DISPLAY_MAX_MM`]. `NaN` (off-domain) -> black.
const PW_RAMP: &[(f32, [u8; 3])] = &[
    (0.0, [120, 74, 30]),   // dry brown
    (0.2, [200, 170, 110]), // tan
    (0.4, [90, 170, 90]),   // green
    (0.6, [60, 160, 200]),  // cyan-blue
    (0.8, [40, 70, 200]),   // blue
    (1.0, [150, 40, 190]),  // moist purple
];

/// The cloud-optical-depth ramp: thin (near-white) -> blue -> thick (dark navy), over
/// `0..`[`COD_DISPLAY_MAX`]. `NaN` (off-domain) -> black; a clear column (0) is near-white.
const COD_RAMP: &[(f32, [u8; 3])] = &[
    (0.0, [245, 245, 245]),  // clear / thin: near white
    (0.25, [150, 200, 230]), // light blue
    (0.5, [70, 130, 200]),   // blue
    (0.75, [30, 60, 130]),   // dark blue
    (1.0, [10, 15, 40]),     // thick core: near black navy
];

/// Colour one field value to `[r, g, b]` (no-data / `NaN` -> black `[0, 0, 0]`).
pub fn value_color(value: f32, field: DerivedField) -> [u8; 3] {
    if !value.is_finite() {
        return [0, 0, 0];
    }
    match field {
        DerivedField::PrecipitableWater => ramp(value / PW_DISPLAY_MAX_MM, PW_RAMP),
        DerivedField::CloudOpticalDepth => ramp(value / COD_DISPLAY_MAX, COD_RAMP),
        // Cloud-top temperature is a real Kelvin field -> reuse the IR "rainbow" thermal
        // enhancement (cold tops light up), exactly the IR-style ramp the brief asks for.
        DerivedField::CloudTopTemp => {
            let c = bt_to_rgba(value, 13, IrEnhancement::Rainbow);
            [c[0], c[1], c[2]]
        }
    }
}

/// Colour a full field plane (`out`) to row-major RGB8 (`len*3`): the basic studio map. The
/// raw `f32` array remains the primary deliverable — this is for the in-app display + the QA
/// PNG only. No-data (`NaN`) pixels are black.
pub fn colorize(values: &[f32], field: DerivedField) -> Vec<u8> {
    let mut out = vec![0u8; values.len() * 3];
    for (i, &v) in values.iter().enumerate() {
        out[i * 3..i * 3 + 3].copy_from_slice(&value_color(v, field));
    }
    out
}

// ── field summary (for the CLI / studio status range) ───────────────────────────

/// Min / max / median of a field's finite (in-domain) values + the finite count. `NaN` (a
/// clear cloud-top column / off-domain pixel) is excluded — those are no-data, not failures.
#[derive(Debug, Clone, Copy)]
pub struct FieldStats {
    pub finite: usize,
    pub min: f64,
    pub max: f64,
    pub median: f64,
}

/// Summarise a field's finite values.
pub fn field_stats(values: &[f32]) -> FieldStats {
    let mut vals: Vec<f64> = values
        .iter()
        .filter(|v| v.is_finite())
        .map(|&v| v as f64)
        .collect();
    if vals.is_empty() {
        return FieldStats {
            finite: 0,
            min: 0.0,
            max: 0.0,
            median: 0.0,
        };
    }
    vals.sort_by(f64::total_cmp);
    FieldStats {
        finite: vals.len(),
        min: vals[0],
        max: vals[vals.len() - 1],
        median: vals[vals.len() / 2],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bricks::{ChannelQuant, LogQuant, encode_log_channel, encode_temperature_celsius};
    use std::collections::BTreeMap;

    /// Build a synthetic brick with caller-supplied per-cell `(ext_liquid, ext_ice,
    /// ext_precip, qvapor, kelvin)` and a uniform TSK, on the affine axis `z = k*dz`.
    fn build_brick(
        nx: usize,
        ny: usize,
        nz: usize,
        dz: f64,
        fill: impl Fn(usize, usize, usize) -> (f32, f32, f32, f32, f32),
    ) -> VolumeBrick {
        let n3 = nx * ny * nz;
        let n2 = nx * ny;
        let mut ext_liquid = vec![0.0f32; n3];
        let mut ext_ice = vec![0.0f32; n3];
        let mut ext_precip = vec![0.0f32; n3];
        let mut qvapor = vec![0.0f32; n3];
        let mut kelvin = vec![0.0f32; n3];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let (l, ic, p, q, t) = fill(i, j, k);
                    let c = (k * ny + j) * nx + i;
                    ext_liquid[c] = l;
                    ext_ice[c] = ic;
                    ext_precip[c] = p;
                    qvapor[c] = q;
                    kelvin[c] = t;
                }
            }
        }
        let (ql, ext_liquid) = encode_log_channel(&ext_liquid);
        let (qi, ext_ice) = encode_log_channel(&ext_ice);
        let (qp, ext_precip) = encode_log_channel(&ext_precip);
        let (qv, qvapor) = encode_log_channel(&qvapor);
        let mut map = BTreeMap::new();
        map.insert("ext_liquid".to_string(), ql);
        map.insert("ext_ice".to_string(), qi);
        map.insert(
            "ext_snow".to_string(),
            LogQuant {
                vmin: 0.0,
                vmax: 0.0,
            },
        );
        map.insert("ext_precip".to_string(), qp);
        map.insert(
            "tau_up".to_string(),
            LogQuant {
                vmin: 0.0,
                vmax: 0.0,
            },
        );
        map.insert("qvapor".to_string(), qv);
        VolumeBrick {
            storage_profile: crate::bricks::StorageProfile::CompactU8,
            science_cloud_f16: None,
            nx,
            ny,
            nz,
            z_min_m: 0.0,
            dz_m: dz,
            time_iso: None,
            quant: ChannelQuant(map),
            ext_liquid,
            ext_ice,
            ext_snow: vec![0u8; n3],
            ext_precip,
            tau_up: vec![0u8; n3],
            qvapor,
            cloud_fraction: vec![255u8; n3],
            has_cloud_fraction: false,
            temperature_f16: encode_temperature_celsius(&kelvin),
            hgt: vec![0.0f32; n2],
            landmask: vec![1.0f32; n2],
            tsk: vec![290.0f32; n2],
            u10: vec![0.0f32; n2],
            v10: vec![0.0f32; n2],
            snowh: None,
            ivgtyp: None,
        }
    }

    #[test]
    fn precipitable_water_matches_the_analytic_column_integral() {
        // A UNIFORM moisture channel: the log-quant of a single-valued channel is exact (decoded
        // q == q), so the field equals the independent hand integral `sum_k rho(z_k)*q*dz` to
        // f32 slop. (A uniform q through the whole depth is not a realistic sounding — the
        // realistic typical-mm range is checked in the exponential-column test below; here the
        // point is the analytic exactness of the integrator.)
        let (nx, ny, nz, dz) = (8usize, 6usize, 60usize, 250.0f64);
        let q = 0.004f64;
        let brick = build_brick(nx, ny, nz, dz, |_, _, _| (0.0, 0.0, 0.0, q as f32, 290.0));
        let f = precipitable_water_field(&brick);
        assert_eq!(f.len(), nx * ny);
        let reference: f64 = (0..nz)
            .map(|k| standard_air_density_kg_m3(k as f64 * dz) * q * dz)
            .sum();
        assert!(reference > 0.0);
        for &v in &f {
            assert!(
                (v as f64 - reference).abs() < 0.05,
                "PW {v} != reference {reference}"
            );
        }
    }

    #[test]
    fn precipitable_water_is_realistic_and_moist_exceeds_dry() {
        // A realistic EXPONENTIAL moisture column `q(z) = q0 exp(-z/2500)`: the PW lands in the
        // typical 5-60 mm range (moisture concentrated in the low troposphere), and a moister
        // column reads a larger PW than a drier one (the brief's sanity check).
        let (nx, ny, nz, dz) = (6usize, 6usize, 60usize, 250.0f64);
        let col = |q0: f64| {
            build_brick(nx, ny, nz, dz, move |_, _, k| {
                let z = k as f64 * dz;
                (0.0, 0.0, 0.0, (q0 * (-z / 2500.0).exp()) as f32, 290.0)
            })
        };
        let (m, d) = (
            precipitable_water_field(&col(0.014))[0] as f64,
            precipitable_water_field(&col(0.004))[0] as f64,
        );
        assert!(
            (5.0..60.0).contains(&m),
            "moist PW {m} not in the typical 5-60 mm range"
        );
        assert!(m > d + 3.0, "moist PW {m} not > dry PW {d}");
    }

    #[test]
    fn precipitable_water_integrates_from_the_terrain_surface() {
        // The brick fills sub-terrain levels with the CLAMPED surface vapor, so PW must
        // integrate from the terrain (`hgt`), not MSL z=0: a column over elevated terrain reads
        // LESS PW than the same column at sea level (the sub-terrain layer excluded). The excluded
        // amount is the surface-density vapor in that layer (~11-12 mm for a 1000 m rise at
        // 10 g/kg), NOT the full column.
        let (nx, ny, nz, dz) = (4usize, 4usize, 60usize, 250.0f64);
        let q = 0.010f64;
        let mut brick = build_brick(nx, ny, nz, dz, |_, _, _| (0.0, 0.0, 0.0, q as f32, 290.0));
        let sea_pw = precipitable_water_field(&brick)[0] as f64; // hgt = 0 (sea level)
        brick.hgt.iter_mut().for_each(|h| *h = 1000.0); // raise the terrain to 1000 m MSL
        let elev_pw = precipitable_water_field(&brick)[0] as f64;
        assert!(
            elev_pw < sea_pw,
            "elevated PW {elev_pw} not < sea-level PW {sea_pw}"
        );
        let excluded = sea_pw - elev_pw;
        assert!(
            (8.0..14.0).contains(&excluded),
            "excluded sub-terrain PW {excluded} not ~ the 1000 m surface layer"
        );
    }

    #[test]
    fn cloud_top_temp_reads_the_cold_top_and_clear_is_nan() {
        // A thick ice cloud in the TOP quarter of the column at a cold temperature: the
        // cloud-top field reads that cold top (the tau~1 level is inside the cold cloud),
        // NOT the warm lapse-rate air below. A CLEAR column reads NaN (no cloud top).
        let (nx, ny, nz, dz) = (6usize, 4usize, 40usize, 250.0f64);
        let cold_top = 215.0f32;
        let top_start = nz * 3 / 4;
        // Left half (i < nx/2) cloudy; right half clear -> both cases in one brick.
        let brick = build_brick(nx, ny, nz, dz, |i, _, k| {
            let t = (288.0 - 6.5 * (k as f64 * dz / 1000.0)) as f32;
            if i < nx / 2 && k >= top_start {
                // 0.05 1/m over ~2.5 km -> tau >> 1; the top level is cold_top.
                (0.0, 0.05, 0.0, 0.0, cold_top)
            } else {
                (0.0, 0.0, 0.0, 0.0, t)
            }
        });
        let f = cloud_top_temp_field(&brick);
        let cloudy = f[0]; // i=0, j=0 (cloudy half)
        let clear = f[nx - 1]; // i=nx-1, j=0 (clear half)
        assert!(
            (cloudy as f64 - cold_top as f64).abs() < 1.0,
            "cloudy cloud-top T {cloudy} != cold top {cold_top}"
        );
        assert!(
            clear.is_nan(),
            "clear column cloud-top T should be NaN, got {clear}"
        );
    }

    #[test]
    fn cloud_top_is_higher_and_colder_for_deeper_cloud() {
        // A cloud whose top is HIGHER (colder on the lapse profile) yields a COLDER cloud-top
        // temperature than one whose top is lower.
        let (nx, ny, nz, dz) = (4usize, 4usize, 40usize, 250.0f64);
        let temp = |k: usize| (288.0 - 6.5 * (k as f64 * dz / 1000.0)) as f32;
        // High cloud: opaque deck near the top.
        let high = build_brick(nx, ny, nz, dz, |_, _, k| {
            if k >= nz - 6 {
                (0.0, 0.05, 0.0, 0.0, temp(k))
            } else {
                (0.0, 0.0, 0.0, 0.0, temp(k))
            }
        });
        // Low cloud: opaque deck lower down.
        let low = build_brick(nx, ny, nz, dz, |_, _, k| {
            if (nz / 3..nz / 3 + 6).contains(&k) {
                (0.0, 0.05, 0.0, 0.0, temp(k))
            } else {
                (0.0, 0.0, 0.0, 0.0, temp(k))
            }
        });
        let th = cloud_top_temp_field(&high)[0];
        let tl = cloud_top_temp_field(&low)[0];
        assert!(th.is_finite() && tl.is_finite());
        assert!(
            th < tl - 5.0,
            "high cloud top {th} not colder than low cloud top {tl}"
        );
    }

    #[test]
    fn cloud_optical_depth_is_zero_clear_and_grows_with_cloud() {
        // Clear -> 0; a uniform extinction column -> ext*nz*dz (analytic); a thicker cloud
        // -> a larger, monotone optical depth.
        let (nx, ny, nz, dz) = (5usize, 5usize, 40usize, 250.0f64);
        let clear = build_brick(nx, ny, nz, dz, |_, _, _| (0.0, 0.0, 0.0, 0.0, 280.0));
        assert!(cloud_optical_depth_field(&clear).iter().all(|&v| v == 0.0));

        let ext = 1.0e-3f32;
        let uniform = build_brick(nx, ny, nz, dz, |_, _, _| (ext, 0.0, 0.0, 0.0, 280.0));
        let cod = cloud_optical_depth_field(&uniform);
        let expect = ext as f64 * nz as f64 * dz;
        assert!(
            expect > 5.0,
            "uniform COD {expect} too small to be a thick column"
        );
        assert!(
            (cod[0] as f64 - expect).abs() < 0.05,
            "COD {} != {expect}",
            cod[0]
        );

        // Monotone with cloud amount: double the extinction -> double the optical depth.
        let thicker = build_brick(nx, ny, nz, dz, |_, _, _| (2.0 * ext, 0.0, 0.0, 0.0, 280.0));
        let cod2 = cloud_optical_depth_field(&thicker);
        assert!(
            cod2[0] > cod[0] * 1.9,
            "thicker COD {} not ~2x {}",
            cod2[0],
            cod[0]
        );
    }

    #[test]
    fn condensate_mixing_ratio_inverts_the_ingest_extinction_kernel() {
        // Round trip through the ingest's forward kernel at the SAME density: q -> beta ->
        // q must be exact (to f64 slop) for every class, including the 1 mm rain/graupel
        // pair and 150 um snow.
        use crate::optics::extinction_coefficient;
        for class in [
            HydrometeorClass::CloudLiquid,
            HydrometeorClass::Ice,
            HydrometeorClass::Snow,
            HydrometeorClass::Rain,
            HydrometeorClass::Graupel,
        ] {
            for z in [0.0, 3000.0, 9000.0] {
                let q = 2.5e-4;
                let beta = extinction_coefficient(
                    standard_air_density_kg_m3(z),
                    q,
                    class.effective_radius_m(),
                );
                let back = condensate_mixing_ratio_kg_kg(beta, class, z);
                assert!(
                    (back - q).abs() < 1e-12,
                    "{class:?} at z={z}: {back} != {q}"
                );
            }
        }
        assert_eq!(
            condensate_mixing_ratio_kg_kg(0.0, HydrometeorClass::Ice, 0.0),
            0.0
        );
        assert_eq!(
            condensate_mixing_ratio_kg_kg(-1.0, HydrometeorClass::Ice, 0.0),
            0.0
        );
    }

    #[test]
    fn condensate_cloud_mask_flags_only_columns_above_the_threshold() {
        // Three columns: clear (0); a THIN liquid layer whose recovered mixing ratio sits a
        // decade UNDER the 1e-6 kg/kg floor; a THICK liquid layer a decade OVER it. Only
        // the thick column is cloudy. The extinctions are set from the forward kernel at
        // the exact mixing ratios so the test is about the threshold, not the optics.
        use crate::optics::extinction_coefficient;
        let (nx, ny, nz, dz) = (3usize, 2usize, 20usize, 250.0f64);
        let r_e = HydrometeorClass::CloudLiquid.effective_radius_m();
        let k_cloud = 8usize;
        let z = k_cloud as f64 * dz;
        let thin = extinction_coefficient(standard_air_density_kg_m3(z), 1.0e-7, r_e) as f32;
        let thick = extinction_coefficient(standard_air_density_kg_m3(z), 1.0e-5, r_e) as f32;
        let brick = build_brick(nx, ny, nz, dz, move |i, _, k| {
            let l = if k == k_cloud {
                match i {
                    1 => thin,
                    2 => thick,
                    _ => 0.0,
                }
            } else {
                0.0
            };
            (l, 0.0, 0.0, 0.0, 280.0)
        });
        let mask = condensate_cloud_mask_field(&brick, CONDENSATE_CLOUD_MASK_THRESHOLD_KG_KG);
        assert_eq!(mask.len(), nx * ny);
        for j in 0..ny {
            assert_eq!(mask[j * nx], CLOUD_MASK_CLEAR, "clear column j={j}");
            assert_eq!(mask[j * nx + 1], CLOUD_MASK_CLEAR, "thin column j={j}");
            assert_eq!(mask[j * nx + 2], CLOUD_MASK_CLOUDY, "thick column j={j}");
        }
        // Lowering the threshold below the thin column's 1e-7 kg/kg flags it too; the
        // clear column never flags.
        let low = condensate_cloud_mask_field(&brick, 1.0e-8);
        assert_eq!(low[1], CLOUD_MASK_CLOUDY);
        assert_eq!(low[0], CLOUD_MASK_CLEAR);

        // Precip-only column (rain radius 1 mm): the SAME extinction as the thick liquid
        // column carries 100x the mass (r_e ratio 1e-3 / 1e-5), so it is cloudy as well —
        // and an ice column at 40 um carries 4x, cloudy too.
        let precip_brick = build_brick(nx, ny, nz, dz, move |i, _, k| {
            if k == k_cloud && i == 0 {
                (0.0, 0.0, thick, 0.0, 280.0)
            } else if k == k_cloud && i == 1 {
                (0.0, thick, 0.0, 0.0, 280.0)
            } else {
                (0.0, 0.0, 0.0, 0.0, 280.0)
            }
        });
        let pm = condensate_cloud_mask_field(&precip_brick, CONDENSATE_CLOUD_MASK_THRESHOLD_KG_KG);
        assert_eq!(pm[0], CLOUD_MASK_CLOUDY);
        assert_eq!(pm[1], CLOUD_MASK_CLOUDY);
        assert_eq!(pm[2], CLOUD_MASK_CLEAR);
    }

    #[test]
    fn resample_mask_maps_native_grid_and_marks_off_domain() {
        // A 2x2 native mask; three output pixels: exact (1,0), rounded (0.4,1.4)->(0,1), and
        // an off-domain NaN pixel -> NO_DATA.
        let mask = [
            CLOUD_MASK_CLEAR,
            CLOUD_MASK_CLOUDY,
            CLOUD_MASK_CLOUDY,
            CLOUD_MASK_CLEAR,
        ];
        let gi = [1.0f32, 0.4, f32::NAN];
        let gj = [0.0f32, 1.4, 0.0];
        let out = resample_mask(&mask, 2, 2, &gi, &gj, 3, 1);
        assert_eq!(
            out,
            vec![CLOUD_MASK_CLOUDY, CLOUD_MASK_CLOUDY, CLOUD_MASK_NO_DATA]
        );
    }

    #[test]
    fn resample_maps_native_grid_and_masks_off_domain() {
        // A tiny known field; a raster with two in-domain pixels (grid indices) and one
        // off-domain (NaN) pixel. Nearest sampling picks the rounded cell; NaN grid -> NaN.
        let (nx, ny) = (3usize, 2usize);
        // field[j*nx+i] = 10*i + j
        let field: Vec<f32> = (0..nx * ny)
            .map(|idx| {
                let (i, j) = (idx % nx, idx / nx);
                (10 * i + j) as f32
            })
            .collect();
        let grid_i = vec![0.0f32, 2.4f32, f32::NAN];
        let grid_j = vec![0.0f32, 0.6f32, 0.0f32];
        let out = resample_field(&field, nx, ny, &grid_i, &grid_j, 3, 1);
        assert_eq!(out[0], 0.0); // (i=0,j=0) -> 0
        assert_eq!(out[1], 21.0); // (round 2.4=2, round 0.6=1) -> 10*2+1
        assert!(out[2].is_nan(), "off-domain grid -> NaN");
    }

    #[test]
    fn resample_carries_nan_cloud_top_through_a_valid_grid_index() {
        // A clear cloud-top column (NaN value) on a VALID grid index must stay NaN (no-cloud
        // sentinel preserved), not read as a neighbour.
        let field = vec![f32::NAN, 220.0f32];
        let out = resample_field(&field, 2, 1, &[0.0f32], &[0.0f32], 1, 1);
        assert!(out[0].is_nan());
    }

    #[test]
    fn colorize_blacks_out_nan_and_colours_finite() {
        let field = DerivedField::PrecipitableWater;
        let values = vec![f32::NAN, 0.0f32, 35.0f32];
        let rgb = colorize(&values, field);
        assert_eq!(rgb.len(), values.len() * 3);
        assert_eq!(&rgb[0..3], &[0, 0, 0], "NaN -> black");
        // A mid-range moisture value is a non-black colour.
        assert!(rgb[6] as u16 + rgb[7] as u16 + rgb[8] as u16 > 0);
        // Cloud-top temp: a cold value is coloured, NaN is black.
        let ctt = colorize(&[f32::NAN, 210.0f32], DerivedField::CloudTopTemp);
        assert_eq!(&ctt[0..3], &[0, 0, 0]);
        assert!(ctt[3] as u16 + ctt[4] as u16 + ctt[5] as u16 > 0);
    }

    #[test]
    fn field_stats_summarise_finite_values_only() {
        let s = field_stats(&[f32::NAN, 5.0, 1.0, 9.0, f32::NAN]);
        assert_eq!(s.finite, 3);
        assert!((s.min - 1.0).abs() < 1e-9);
        assert!((s.max - 9.0).abs() < 1e-9);
        assert!((s.median - 5.0).abs() < 1e-9);
        let empty = field_stats(&[f32::NAN, f32::NAN]);
        assert_eq!(empty.finite, 0);
    }

    #[test]
    fn derived_field_slug_round_trips() {
        for f in DerivedField::ALL {
            assert_eq!(DerivedField::parse(f.slug()), Some(f));
        }
        assert_eq!(
            DerivedField::parse("PWAT"),
            Some(DerivedField::PrecipitableWater)
        );
        assert_eq!(
            DerivedField::parse("optical-depth"),
            Some(DerivedField::CloudOpticalDepth)
        );
        assert_eq!(DerivedField::parse("nope"), None);
    }
}
