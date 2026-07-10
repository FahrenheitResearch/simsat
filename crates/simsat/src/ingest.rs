//! Streaming wrfout -> `.ssb` brick ingest (design doc section 2).
//!
//! Hard memory discipline (docs/bowecho-precedents.md sections 2, 5): NEVER
//! `getvar` a 3-D field — wrf-core's `WrfFile` memoizes every 3-D f64 intermediate
//! (measured 8.87 GB peak). We read raw variables one at a time via
//! `WrfFile::read_var`, do the cheap arithmetic ourselves in f32, fold each field
//! into the (brick-resolution) accumulation buffers, and drop it before the next
//! read. netcrust is used only as the metadata/`IVGTYP` fallback, mirroring
//! `local_import.rs`'s split. The ingest worker lowers its thread priority
//! (`platform::lower_ingest_thread_priority`) and the peak RSS is logged and, in
//! the env-gated fixture test, asserted < 2.5 GB (the design contract).
//!
//! Derived fields: `p = P + PB`; `T = (theta' + 300)*(p/p0)^kappa`;
//! `z = (PH + PHB)/g0` with a VERTICAL destagger (`bottom_top_stag` -> mass
//! levels; no such helper exists in BowEcho, digest section 8, so it is written
//! and tested here). Extinction per class from `optics.rs`. `tau_up` is the
//! cumulative optical depth integrated from the brick top downward.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use wrf_core::WrfFile;

use crate::bricks::{
    self, ChannelQuant, ManifestAnchor, ManifestProjection, ManifestTimestep, RunManifest,
    VolumeBrick,
};
use crate::frame::{FrameError, GridGeoref, WrfProjectionParams, wrf_center_anchor};
use crate::optics::{self, HydrometeorClass};
use crate::platform;

/// Default uniform vertical spacing of the brick axis (m).
pub const DEFAULT_DZ_M: f64 = 250.0;
/// Default brick base height (m MSL): sea level.
pub const DEFAULT_Z_MIN_M: f64 = 0.0;
/// Default number of vertical slices (0..19750 m ~= 20 km at 250 m spacing).
pub const DEFAULT_NZ_BRICK: usize = 80;

/// Above this 3-D cell count the best-effort `IVGTYP` netcrust fallback is
/// skipped (a netcrust reopen is a ~57 s metadata pass on a 2 GB file; `IVGTYP`
/// is a best-effort, later-milestone field). Matches the digest's large-grid
/// threshold (`LARGE_WRF_WARN_CELLS_3D`).
pub const NETCRUST_FALLBACK_MAX_CELLS: usize = 10_000_000;

/// Vertical extrapolation policy outside a native WRF column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extrap {
    /// Hold the nearest column-edge value (used for temperature, near-surface qvapor).
    ClampEdge,
    /// Extrapolate as zero (used for extinction above/below the column).
    Zero,
}

/// Configuration for a single-timestep ingest.
#[derive(Debug, Clone)]
pub struct IngestConfig {
    /// Cache root (explicit param in M0; the studio app wires settings later).
    pub cache_dir: PathBuf,
    /// Run identifier; `None` derives it from the wrfout file stem.
    pub run_id: Option<String>,
    /// Which time index in the file (default 0).
    pub timestep: usize,
    pub dz_m: f64,
    pub z_min_m: f64,
    pub nz_brick: usize,
}

impl IngestConfig {
    /// Sensible defaults for a given cache dir.
    pub fn new(cache_dir: PathBuf) -> Self {
        Self {
            cache_dir,
            run_id: None,
            timestep: 0,
            dz_m: DEFAULT_DZ_M,
            z_min_m: DEFAULT_Z_MIN_M,
            nz_brick: DEFAULT_NZ_BRICK,
        }
    }
}

/// What an ingest produced.
#[derive(Debug, Clone)]
pub struct IngestReport {
    pub run_id: String,
    pub brick_path: PathBuf,
    pub manifest_path: PathBuf,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub hhmm: u16,
    pub wall: Duration,
    pub peak_rss_bytes: Option<u64>,
    pub ssb_bytes: u64,
}

/// Geometry + projection read cheaply for the ratchet and the manifest.
#[derive(Debug, Clone)]
pub struct GridGeometry {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub nz_stag: usize,
    /// Number of timesteps in the source file (the selected timestep was validated
    /// against it — see [`read_grid_geometry`]).
    pub nt: usize,
    pub params: WrfProjectionParams,
    /// Stored latitude/longitude planes AT the selected timestep (a moving nest
    /// re-centres between outputs, so these are per-timestep coordinates).
    pub xlat: Vec<f32>,
    pub xlong: Vec<f32>,
    pub time_iso: Option<String>,
    pub hhmm: u16,
}

impl GridGeometry {
    /// Build a center-anchored georeference from this geometry.
    pub fn georef(&self) -> Result<GridGeoref, FrameError> {
        GridGeoref::from_wrf_center(&self.params, self.nx, self.ny, &self.xlat, &self.xlong)
    }

    /// The persistable per-timestep georef anchor (the exact values [`Self::georef`]
    /// anchors with), for the run manifest. `None` only for a degenerate grid, which
    /// [`read_grid_geometry`] already refuses.
    pub fn manifest_anchor(&self) -> Option<ManifestAnchor> {
        wrf_center_anchor(&self.params, self.nx, self.ny, &self.xlat, &self.xlong)
            .ok()
            .map(
                |(ref_i, ref_j, ref_lat_deg, ref_lon_deg, dx, dy)| ManifestAnchor {
                    ref_i,
                    ref_j,
                    ref_lat_deg,
                    ref_lon_deg,
                    dx,
                    dy,
                },
            )
    }
}

/// Ingest errors.
#[derive(Debug)]
pub enum IngestError {
    Wrf(String),
    Frame(FrameError),
    Brick(bricks::BrickError),
    MissingVar(String),
    Shape(String),
}

impl std::fmt::Display for IngestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Wrf(s) => write!(f, "wrf read error: {s}"),
            Self::Frame(e) => write!(f, "projection error: {e}"),
            Self::Brick(e) => write!(f, "brick error: {e}"),
            Self::MissingVar(s) => write!(f, "required variable missing: {s}"),
            Self::Shape(s) => write!(f, "unexpected shape: {s}"),
        }
    }
}
impl std::error::Error for IngestError {}
impl From<FrameError> for IngestError {
    fn from(e: FrameError) -> Self {
        Self::Frame(e)
    }
}
impl From<bricks::BrickError> for IngestError {
    fn from(e: bricks::BrickError) -> Self {
        Self::Brick(e)
    }
}

fn nonempty_env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// A SimSat Studio data-dir default for the brick cache (no settings system in M0).
pub fn default_cache_dir() -> PathBuf {
    if let Some(local) = nonempty_env("LOCALAPPDATA") {
        return PathBuf::from(local).join("SimSatStudio").join("cache");
    }
    if let Some(xdg) = nonempty_env("XDG_DATA_HOME") {
        return PathBuf::from(xdg).join("simsat-studio").join("cache");
    }
    if let Some(home) = nonempty_env("HOME") {
        return PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("simsat-studio")
            .join("cache");
    }
    std::env::temp_dir().join("simsat-studio").join("cache")
}

fn run_id_from_path(path: &Path) -> String {
    let stem = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("run")
        .to_string();
    sanitize_token(&stem)
}

/// The default run identifier ingest derives from a wrfout path (sanitized file
/// name). Exposed so the studio can predict the brick/run path deterministically.
pub fn default_run_id(path: &Path) -> String {
    run_id_from_path(path)
}

fn sanitize_token(value: &str) -> String {
    let mut out: String = value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    while out.contains("__") {
        out = out.replace("__", "_");
    }
    let trimmed = out.trim_matches('_').to_string();
    if trimmed.is_empty() {
        "run".to_string()
    } else {
        trimmed
    }
}

fn parse_wrf_time(t: &str) -> (u16, String) {
    if let Some(us) = t.find('_') {
        let (date, rest) = t.split_at(us);
        let time = &rest[1..];
        let hh: u16 = time.get(0..2).and_then(|s| s.parse().ok()).unwrap_or(0);
        let mm: u16 = time.get(3..5).and_then(|s| s.parse().ok()).unwrap_or(0);
        (hh * 100 + mm, format!("{date}T{time}Z"))
    } else {
        (0, t.to_string())
    }
}

fn wrf_projection_params(wrf: &WrfFile) -> WrfProjectionParams {
    // A missing MAP_PROJ is a malformed / non-WRF file being navigated on a GUESS.
    // The Lambert default is kept (it has always been the fallback and CONUS-class
    // domains are Lambert), but the guess is now LOGGED instead of silent.
    let map_proj = match wrf.global_attr_i32("MAP_PROJ") {
        Ok(v) => v,
        Err(_) => {
            crate::log_line!(
                "simsat ingest: WARNING — global attribute MAP_PROJ is missing; \
                 assuming Lambert conformal (1). Georeferencing may be wrong if the \
                 file is not a Lambert-projected wrfout."
            );
            1
        }
    };
    let attr = |name: &str, default: f64| wrf.global_attr_f64(name).unwrap_or(default);
    WrfProjectionParams {
        map_proj,
        truelat1_deg: attr("TRUELAT1", 30.0),
        truelat2_deg: wrf
            .global_attr_f64("TRUELAT2")
            .or_else(|_| wrf.global_attr_f64("TRUELAT1"))
            .unwrap_or(60.0),
        stand_lon_deg: wrf
            .global_attr_f64("STAND_LON")
            .or_else(|_| wrf.global_attr_f64("CEN_LON"))
            .unwrap_or(0.0),
        cen_lat_deg: attr("CEN_LAT", 0.0),
        cen_lon_deg: attr("CEN_LON", 0.0),
        dx_m: wrf.global_attr_f64("DX").unwrap_or(wrf.dx),
        dy_m: wrf.global_attr_f64("DY").unwrap_or(wrf.dy),
    }
}

/// Minimal variable-read seam over `WrfFile`, so the field readers can be unit
/// tested for correct timestep threading (M1-review MAJOR-1) without a real
/// multi-time wrfout — the canonical fixtures (and the owner's 193 MB candidate)
/// are all single-time. `WrfFile` is the production impl; a test fake varies its
/// data by `(name, t)` so a hardcoded `t = 0` is detectable.
trait VarReader {
    fn has_var(&self, name: &str) -> bool;
    fn read_var_t(&self, name: &str, t: usize) -> Result<Vec<f64>, String>;
}

impl VarReader for WrfFile {
    fn has_var(&self, name: &str) -> bool {
        WrfFile::has_var(self, name)
    }
    fn read_var_t(&self, name: &str, t: usize) -> Result<Vec<f64>, String> {
        WrfFile::read_var(self, name, t).map_err(|e| e.to_string())
    }
}

/// The metadata seam over `WrfFile` for [`read_geometry`] (dims, time count/labels,
/// projection attributes), extending the [`VarReader`] field seam so the geometry read
/// — the timestep validation, the malformed-dims guards, and the PER-TIMESTEP
/// `XLAT`/`XLONG` anchoring (moving nests) — is unit-testable against a fake
/// multi-time file. `WrfFile` is the production impl.
trait GeomReader: VarReader {
    /// `(nx, ny, nz, nz_stag)` from the file headers.
    fn dims(&self) -> (usize, usize, usize, usize);
    /// Number of timesteps in the file.
    fn time_count(&self) -> usize;
    /// The raw `Times` labels (empty when the variable is absent/unreadable).
    fn time_labels(&self) -> Vec<String>;
    /// The projection globals.
    fn projection_params(&self) -> WrfProjectionParams;
}

impl GeomReader for WrfFile {
    fn dims(&self) -> (usize, usize, usize, usize) {
        (self.nx, self.ny, self.nz, self.nz_stag)
    }
    fn time_count(&self) -> usize {
        self.nt
    }
    fn time_labels(&self) -> Vec<String> {
        self.times().unwrap_or_default()
    }
    fn projection_params(&self) -> WrfProjectionParams {
        wrf_projection_params(self)
    }
}

fn read_2d_required<R: VarReader>(
    wrf: &R,
    name: &str,
    nx: usize,
    ny: usize,
    t: usize,
) -> Result<Vec<f32>, IngestError> {
    read_2d_opt(wrf, name, nx, ny, t)?.ok_or_else(|| IngestError::MissingVar(name.to_string()))
}

fn read_2d_opt<R: VarReader>(
    wrf: &R,
    name: &str,
    nx: usize,
    ny: usize,
    t: usize,
) -> Result<Option<Vec<f32>>, IngestError> {
    if !wrf.has_var(name) {
        return Ok(None);
    }
    let values = wrf
        .read_var_t(name, t)
        .map_err(|e| IngestError::Wrf(format!("{name}: {e}")))?;
    if values.len() != nx * ny {
        return Err(IngestError::Shape(format!(
            "{name}: expected {} (2-D), got {}",
            nx * ny,
            values.len()
        )));
    }
    Ok(Some(values.into_iter().map(|v| v as f32).collect()))
}

fn read_3d_required<R: VarReader>(
    wrf: &R,
    name: &str,
    nz: usize,
    ny: usize,
    nx: usize,
    t: usize,
) -> Result<Vec<f32>, IngestError> {
    read_3d_opt(wrf, name, nz, ny, nx, t)?.ok_or_else(|| IngestError::MissingVar(name.to_string()))
}

/// Read a 3-D field (nz*ny*nx) at time index `t` as f32, or `None` if absent.
fn read_3d_opt<R: VarReader>(
    wrf: &R,
    name: &str,
    nz: usize,
    ny: usize,
    nx: usize,
    t: usize,
) -> Result<Option<Vec<f32>>, IngestError> {
    if !wrf.has_var(name) {
        return Ok(None);
    }
    let values = wrf
        .read_var_t(name, t)
        .map_err(|e| IngestError::Wrf(format!("{name}: {e}")))?;
    let expected = nz * ny * nx;
    if values.len() != expected {
        return Err(IngestError::Shape(format!(
            "{name}: expected {expected}, got {}",
            values.len()
        )));
    }
    Ok(Some(values.into_iter().map(|v| v as f32).collect()))
}

/// Best-effort `IVGTYP`: wrf-core fast path, then (when `allow_netcrust`) the
/// netcrust fallback (the int dataset trips wrf-core's "no layout message"),
/// then give up.
fn read_ivgtyp_best_effort(
    wrf: &WrfFile,
    path: &Path,
    nx: usize,
    ny: usize,
    t: usize,
    allow_netcrust: bool,
) -> Option<Vec<f32>> {
    if wrf.has_var("IVGTYP")
        && let Ok(v) = wrf.read_var("IVGTYP", t)
        && v.len() == nx * ny
    {
        return Some(v.into_iter().map(|x| x as f32).collect());
    }
    if !allow_netcrust {
        return None;
    }
    // netcrust fallback (documented in docs/bowecho-precedents.md section 5).
    match netcrust::open(path) {
        Ok(file) => match file.read_f64_first_record_or_all("IVGTYP") {
            Ok(v) if v.len() == nx * ny => Some(v.into_iter().map(|x| x as f32).collect()),
            _ => None,
        },
        Err(_) => None,
    }
}

/// Vertical destagger `[nz_stag, ny, nx]` geopotential-height -> `[nz, ny, nx]`
/// mass levels by averaging adjacent z faces (`nz = nz_stag - 1`).
pub fn destagger_vertical(stag: &[f64], nz_stag: usize, ny: usize, nx: usize) -> Vec<f32> {
    let nz = nz_stag - 1;
    let mut out = vec![0f32; nz * ny * nx];
    for k in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let lo = (k * ny + y) * nx + x;
                let hi = ((k + 1) * ny + y) * nx + x;
                out[(k * ny + y) * nx + x] = (0.5 * (stag[lo] + stag[hi])) as f32;
            }
        }
    }
    out
}

/// Horizontal destagger `[nz, ny, nx+1]` (`west_east_stag`) -> `[nz, ny, nx]`.
/// Ported pattern from `local_import.rs::destagger_x`. Not used by the M0 brick
/// (no staggered 3-D winds are read yet); provided and tested for later winds.
pub fn destagger_x(src: &[f64], nz: usize, ny: usize, nx: usize) -> Vec<f64> {
    let nxs = nx + 1;
    let mut out = vec![0f64; nz * ny * nx];
    for k in 0..nz {
        for y in 0..ny {
            let base_s = (k * ny + y) * nxs;
            let base_d = (k * ny + y) * nx;
            for x in 0..nx {
                out[base_d + x] = 0.5 * (src[base_s + x] + src[base_s + x + 1]);
            }
        }
    }
    out
}

/// Horizontal destagger `[nz, ny+1, nx]` (`south_north_stag`) -> `[nz, ny, nx]`.
/// Ported pattern from `local_import.rs::destagger_y`.
pub fn destagger_y(src: &[f64], nz: usize, ny: usize, nx: usize) -> Vec<f64> {
    let mut out = vec![0f64; nz * ny * nx];
    for k in 0..nz {
        for y in 0..ny {
            let base_lo = (k * (ny + 1) + y) * nx;
            let base_hi = (k * (ny + 1) + y + 1) * nx;
            let base_d = (k * ny + y) * nx;
            for x in 0..nx {
                out[base_d + x] = 0.5 * (src[base_lo + x] + src[base_hi + x]);
            }
        }
    }
    out
}

/// Linear-resample one native vertical profile onto the affine brick axis.
#[allow(clippy::too_many_arguments)]
pub fn resample_column(
    z_native: &[f64],
    f_native: &[f64],
    z_min: f64,
    dz: f64,
    nz_brick: usize,
    below: Extrap,
    above: Extrap,
    out: &mut Vec<f64>,
) {
    out.clear();
    let n = z_native.len();
    if n == 0 {
        out.resize(nz_brick, 0.0);
        return;
    }
    let below_value = |z: f64| match below {
        Extrap::ClampEdge => f_native[0],
        Extrap::Zero if z == z_native[0] => f_native[0],
        Extrap::Zero => 0.0,
    };
    let above_value = |z: f64| match above {
        Extrap::ClampEdge => f_native[n - 1],
        Extrap::Zero if z == z_native[n - 1] => f_native[n - 1],
        Extrap::Zero => 0.0,
    };
    let mut k = 0usize;
    for m in 0..nz_brick {
        let zb = z_min + m as f64 * dz;
        // Strict inequalities: a brick level exactly at a column edge takes the
        // edge value (interpolation), only strictly-outside levels extrapolate.
        if zb < z_native[0] || (n == 1 && zb <= z_native[0]) {
            out.push(below_value(zb));
        } else if zb > z_native[n - 1] {
            out.push(above_value(zb));
        } else {
            while k + 1 < n && z_native[k + 1] < zb {
                k += 1;
            }
            let (z0, z1) = (z_native[k], z_native[k + 1]);
            let t = if z1 > z0 { (zb - z0) / (z1 - z0) } else { 0.0 };
            out.push(f_native[k] + t * (f_native[k + 1] - f_native[k]));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn resample_volume(
    native: &[f32],
    z: &[f32],
    nx: usize,
    ny: usize,
    nz_native: usize,
    z_min: f64,
    dz: f64,
    nz_brick: usize,
    below: Extrap,
    above: Extrap,
) -> Vec<f32> {
    let mut out = vec![0f32; nx * ny * nz_brick];
    let mut zc = vec![0f64; nz_native];
    let mut fc = vec![0f64; nz_native];
    let mut col: Vec<f64> = Vec::with_capacity(nz_brick);
    for y in 0..ny {
        for x in 0..nx {
            for (k, (zc_k, fc_k)) in zc.iter_mut().zip(fc.iter_mut()).enumerate() {
                let idx = (k * ny + y) * nx + x;
                *zc_k = z[idx] as f64;
                *fc_k = native[idx] as f64;
            }
            resample_column(&zc, &fc, z_min, dz, nz_brick, below, above, &mut col);
            for (m, &value) in col.iter().enumerate() {
                out[(m * ny + y) * nx + x] = value as f32;
            }
        }
    }
    out
}

/// Cumulative native integral `F(z) = integral[z_native[0], z] f dz` for the
/// piecewise-linear profile `(z_native, f_native)`, honoring the `below`/`above`
/// extrapolation policy outside the native span. `cum[k]` must be the prefilled
/// trapezoidal cumulative at native nodes (`cum[0] = 0`). `hint` is a forward-moving
/// segment cursor (queries arrive monotonically increasing). Below the column,
/// `Zero` contributes nothing and `ClampEdge` extends `f_native[0]`; symmetrically
/// above. This is the integral primitive the conserving resample differences.
fn native_cumulative_at(
    z_native: &[f64],
    f_native: &[f64],
    cum: &[f64],
    z: f64,
    below: Extrap,
    above: Extrap,
    hint: &mut usize,
) -> f64 {
    let n = z_native.len();
    if z <= z_native[0] {
        return match below {
            Extrap::Zero => 0.0,
            Extrap::ClampEdge => f_native[0] * (z - z_native[0]),
        };
    }
    if z >= z_native[n - 1] {
        return cum[n - 1]
            + match above {
                Extrap::Zero => 0.0,
                Extrap::ClampEdge => f_native[n - 1] * (z - z_native[n - 1]),
            };
    }
    // Locate the segment [z_native[k], z_native[k+1]] containing z; advance the
    // forward hint, with a bounded backward step for safety if the hint is stale.
    let mut k = (*hint).min(n - 2);
    while k < n - 2 && z_native[k + 1] <= z {
        k += 1;
    }
    while k > 0 && z_native[k] > z {
        k -= 1;
    }
    *hint = k;
    let seg = z_native[k + 1] - z_native[k];
    let fz = if seg > 0.0 {
        f_native[k] + (z - z_native[k]) / seg * (f_native[k + 1] - f_native[k])
    } else {
        f_native[k]
    };
    cum[k] + 0.5 * (f_native[k] + fz) * (z - z_native[k])
}

/// Integral-conserving vertical resample of one native column onto the affine brick
/// axis (M0-review MAJOR-1). Brick level `m` represents the cell centered on its
/// node `z(m) = z_min + m*dz`, i.e. `[z(m) - dz/2, z(m) + dz/2)`, and its value is
/// the AVERAGE of the native piecewise-linear profile over that cell:
/// `value_m = (1/dz) integral[cell_m] f dz`. Hence `sum_m value_m * dz` equals the
/// native column integral of `f` (to interpolation accuracy) for any stack of brick
/// cells — so column optical depth is conserved and a thin native layer lying
/// between two brick nodes is smeared into the covering cells instead of dropped
/// (the failure the point-sampling [`resample_column`] had). Used for the
/// extinction channels and qvapor; temperature stays point-sampled (intensive).
///
/// `cum` is scratch (`>= z_native.len()` capacity); it is cleared and refilled.
#[allow(clippy::too_many_arguments)]
pub fn resample_column_conservative(
    z_native: &[f64],
    f_native: &[f64],
    z_min: f64,
    dz: f64,
    nz_brick: usize,
    below: Extrap,
    above: Extrap,
    cum: &mut Vec<f64>,
    out: &mut Vec<f64>,
) {
    out.clear();
    let n = z_native.len();
    if n == 0 {
        out.resize(nz_brick, 0.0);
        return;
    }
    if n == 1 {
        // Single native level: no column to integrate; fall back to a policy-aware
        // point value (not reached for real WRF, where nz is many tens).
        for m in 0..nz_brick {
            let zc = z_min + m as f64 * dz;
            let v = if zc < z_native[0] {
                match below {
                    Extrap::ClampEdge => f_native[0],
                    Extrap::Zero => 0.0,
                }
            } else if zc > z_native[0] {
                match above {
                    Extrap::ClampEdge => f_native[0],
                    Extrap::Zero => 0.0,
                }
            } else {
                f_native[0]
            };
            out.push(v);
        }
        return;
    }
    // Prefill the trapezoidal cumulative at native nodes.
    cum.clear();
    cum.resize(n, 0.0);
    for k in 1..n {
        cum[k] =
            cum[k - 1] + 0.5 * (f_native[k - 1] + f_native[k]) * (z_native[k] - z_native[k - 1]);
    }
    // Each brick cell integral is F(z_top) - F(z_bottom), divided by dz for the mean.
    let mut hint = 0usize;
    for m in 0..nz_brick {
        let zc = z_min + m as f64 * dz;
        let za = zc - 0.5 * dz;
        let zb = zc + 0.5 * dz;
        let fa = native_cumulative_at(z_native, f_native, cum, za, below, above, &mut hint);
        let fb = native_cumulative_at(z_native, f_native, cum, zb, below, above, &mut hint);
        out.push((fb - fa) / dz);
    }
}

#[allow(clippy::too_many_arguments)]
fn resample_volume_conservative(
    native: &[f32],
    z: &[f32],
    nx: usize,
    ny: usize,
    nz_native: usize,
    z_min: f64,
    dz: f64,
    nz_brick: usize,
    below: Extrap,
    above: Extrap,
) -> Vec<f32> {
    let mut out = vec![0f32; nx * ny * nz_brick];
    let mut zc = vec![0f64; nz_native];
    let mut fc = vec![0f64; nz_native];
    let mut cum: Vec<f64> = Vec::with_capacity(nz_native);
    let mut col: Vec<f64> = Vec::with_capacity(nz_brick);
    for y in 0..ny {
        for x in 0..nx {
            for (k, (zc_k, fc_k)) in zc.iter_mut().zip(fc.iter_mut()).enumerate() {
                let idx = (k * ny + y) * nx + x;
                *zc_k = z[idx] as f64;
                *fc_k = native[idx] as f64;
            }
            resample_column_conservative(
                &zc, &fc, z_min, dz, nz_brick, below, above, &mut cum, &mut col,
            );
            for (m, &value) in col.iter().enumerate() {
                out[(m * ny + y) * nx + x] = value as f32;
            }
        }
    }
    out
}

/// Cumulative optical depth from the brick top down to each level (trapezoidal).
/// `beta` is total extinction (m^-1) per level, index 0 = base, last = top.
/// Returns `tau[m]` = optical depth from the top-of-brick down to level `m`.
///
/// Fed the integral-conserving [`resample_volume_conservative`] extinction (via
/// `beta_total`), this now conserves the column: `tau[base]` recovers the native
/// column optical depth instead of the point-sampled underestimate (M0-review
/// MAJOR-1 — tau_up must be consistent with the conserved profile).
pub fn integrate_tau_up_column(beta: &[f64], dz: f64) -> Vec<f64> {
    let n = beta.len();
    let mut tau = vec![0f64; n];
    if n == 0 {
        return tau;
    }
    for m in (0..n - 1).rev() {
        tau[m] = tau[m + 1] + 0.5 * (beta[m] + beta[m + 1]) * dz;
    }
    tau
}

fn tau_up_volume(beta_total: &[f32], nx: usize, ny: usize, nz: usize, dz: f64) -> Vec<f32> {
    let mut out = vec![0f32; nx * ny * nz];
    let mut col = vec![0f64; nz];
    for y in 0..ny {
        for x in 0..nx {
            for (m, col_m) in col.iter_mut().enumerate() {
                *col_m = beta_total[(m * ny + y) * nx + x] as f64;
            }
            let tau = integrate_tau_up_column(&col, dz);
            for (m, &value) in tau.iter().enumerate() {
                out[(m * ny + y) * nx + x] = value as f32;
            }
        }
    }
    out
}

fn beta_from_q(q: &[f32], rho: &[f32], effective_radius_m: f64) -> Vec<f32> {
    q.iter()
        .zip(rho.iter())
        .map(|(&qi, &ri)| {
            optics::extinction_coefficient(ri as f64, qi as f64, effective_radius_m) as f32
        })
        .collect()
}

/// Add a second species' extinction into an existing beta buffer at its OWN
/// effective radius. Extinctions add linearly, so a shared brick channel can carry
/// several species as long as each converts at its own optics before the sum (the
/// SSB v3 snow-optics fix: QSNOW joins ext_precip at the snow aggregate beta).
fn add_beta_from_q(beta: &mut [f32], q: &[f32], rho: &[f32], effective_radius_m: f64) {
    for ((b, &qi), &ri) in beta.iter_mut().zip(q.iter()).zip(rho.iter()) {
        *b += optics::extinction_coefficient(ri as f64, qi as f64, effective_radius_m) as f32;
    }
}

/// Add a brick-resolution extinction channel into the running `beta_total`
/// (for `tau_up`) and quantize it to a log u8 channel.
fn accumulate_and_encode(beta_total: &mut [f32], ext: &[f32]) -> (bricks::LogQuant, Vec<u8>) {
    for (bt, e) in beta_total.iter_mut().zip(ext.iter()) {
        *bt += *e;
    }
    bricks::encode_log_channel(ext)
}

fn read_geometry<R: GeomReader>(wrf: &R, timestep: usize) -> Result<GridGeometry, IngestError> {
    let (nx, ny, nz, nz_stag) = wrf.dims();
    if nx < 2 || ny < 2 || nz < 1 {
        return Err(IngestError::Shape(format!(
            "degenerate grid {nx}x{ny}x{nz}"
        )));
    }
    // Malformed-geometry guard: the vertical destagger computes `nz_stag - 1` and the
    // ingest assumes staggered fields carry exactly one more level than mass fields;
    // any other relationship (a corrupt or non-WRF file) would underflow or misindex
    // downstream instead of erroring here.
    if nz_stag != nz + 1 {
        return Err(IngestError::Shape(format!(
            "staggered vertical dimension {nz_stag} != nz+1 ({}) — corrupt or non-WRF file",
            nz + 1
        )));
    }
    // Timestep validation (the wrfout path; the cached path validates against the
    // manifest in api.rs): a clean, actionable error instead of whatever the
    // downstream field reads would produce for an out-of-range time index.
    let nt = wrf.time_count().max(1);
    if timestep >= nt {
        return Err(IngestError::Shape(format!(
            "timestep {timestep} is out of range: the file has {nt} timestep(s) (valid 0..={})",
            nt - 1
        )));
    }
    // XLAT/XLONG are read AT THE SELECTED TIMESTEP. For a static domain every
    // timestep stores identical planes, so this is byte-identical to the historic
    // always-t-0 read; but a MOVING NEST re-centres the domain between outputs, and
    // anchoring frame t on the t=0 coordinates would georeference the frame at the
    // nest's OLD position (silently wrong by the full nest displacement).
    let xlat = read_2d_required(wrf, "XLAT", nx, ny, timestep)
        .or_else(|_| read_2d_required(wrf, "XLAT_M", nx, ny, timestep))?;
    let xlong = read_2d_required(wrf, "XLONG", nx, ny, timestep)
        .or_else(|_| read_2d_required(wrf, "XLONG_M", nx, ny, timestep))?;
    // Moving-nest detection (log-only): compare the domain-centre coordinate at the
    // selected timestep against t = 0. Two cheap extra 2-D reads, only for t > 0.
    if timestep > 0 {
        let c = ((ny - 1) / 2) * nx + (nx - 1) / 2;
        let lat0 = read_2d_opt(wrf, "XLAT", nx, ny, 0)
            .ok()
            .flatten()
            .or_else(|| read_2d_opt(wrf, "XLAT_M", nx, ny, 0).ok().flatten());
        let lon0 = read_2d_opt(wrf, "XLONG", nx, ny, 0)
            .ok()
            .flatten()
            .or_else(|| read_2d_opt(wrf, "XLONG_M", nx, ny, 0).ok().flatten());
        if let (Some(lat0), Some(lon0)) = (lat0, lon0) {
            let (dlat, dlon) = (
                (xlat[c] - lat0[c]).abs() as f64,
                (xlong[c] - lon0[c]).abs() as f64,
            );
            if dlat > 1.0e-6 || dlon > 1.0e-6 {
                crate::log_line!(
                    "simsat ingest: moving nest detected — domain centre moved \
                     ({:.4}, {:.4}) -> ({:.4}, {:.4}) between t0 and t{timestep}; \
                     the georef is anchored per-timestep",
                    lat0[c],
                    lon0[c],
                    xlat[c],
                    xlong[c]
                );
            }
        }
    }
    let params = wrf.projection_params();
    let (hhmm, time_iso) = wrf
        .time_labels()
        .get(timestep)
        .map(|t| {
            let (hhmm, iso) = parse_wrf_time(t);
            (hhmm, Some(iso))
        })
        .unwrap_or((0, None));
    Ok(GridGeometry {
        nx,
        ny,
        nz,
        nz_stag,
        nt,
        params,
        xlat,
        xlong,
        time_iso,
        hhmm,
    })
}

/// Read just the grid geometry + projection (for the ratchet and quick probes).
pub fn read_grid_geometry(path: &Path, timestep: usize) -> Result<GridGeometry, IngestError> {
    let wrf = WrfFile::open(path).map_err(|e| IngestError::Wrf(e.to_string()))?;
    read_geometry(&wrf, timestep)
}

/// The source-file identity recorded per ingested timestep — `(byte length, mtime as
/// unix seconds)` — so a later cache hit can detect a re-run WRF writing over the same
/// path ([`crate::bricks::cache_entry_is_fresh`]). `None` components when the metadata
/// is unavailable (then the cache entry can never be judged fresh — the safe side).
pub fn source_identity(path: &Path) -> (Option<u64>, Option<i64>) {
    match std::fs::metadata(path) {
        Ok(m) => {
            let mtime = m.modified().ok().and_then(|t| {
                t.duration_since(std::time::UNIX_EPOCH)
                    .ok()
                    .map(|d| d.as_secs() as i64)
            });
            (Some(m.len()), mtime)
        }
        Err(_) => (None, None),
    }
}

/// A cheap wrfout probe: dims + timestep labels + projection attributes, read
/// WITHOUT decoding any field (dims come from HDF5 headers). Safe on the UI
/// thread right after a file dialog; feeds the studio's size-gate and pickers.
#[derive(Debug, Clone)]
pub struct WrfProbe {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub nt: usize,
    pub times: Vec<String>,
    pub params: WrfProjectionParams,
    pub file_bytes: u64,
}

/// Probe a wrfout file's dims/times/projection cheaply (no field decode).
pub fn probe_wrf(path: &Path) -> Result<WrfProbe, IngestError> {
    let wrf = WrfFile::open(path).map_err(|e| IngestError::Wrf(e.to_string()))?;
    let times = wrf.times().unwrap_or_default();
    let params = wrf_projection_params(&wrf);
    let file_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    Ok(WrfProbe {
        nx: wrf.nx,
        ny: wrf.ny,
        nz: wrf.nz,
        nt: wrf.nt,
        times,
        params,
        file_bytes,
    })
}

/// Ingest one wrfout timestep into an `.ssb` brick + `run.json`. Streaming, f32,
/// one 3-D field resident at a time; logs wall time and peak RSS.
pub fn ingest_timestep(path: &Path, config: &IngestConfig) -> Result<IngestReport, IngestError> {
    platform::lower_ingest_thread_priority();
    let start = Instant::now();

    let wrf = WrfFile::open(path).map_err(|e| IngestError::Wrf(e.to_string()))?;
    let geom = read_geometry(&wrf, config.timestep)?;
    let (nx, ny, nz, nz_stag) = (geom.nx, geom.ny, geom.nz, geom.nz_stag);
    let (z_min, dz, nz_brick) = (config.z_min_m, config.dz_m, config.nz_brick);
    // The selected time index. Every atmospheric/surface field read below uses `t`
    // (M1-review MAJOR-1: field reads previously hardcoded time 0, so any
    // `config.timestep > 0` cached time-0 data mislabeled with the later time).
    let t = config.timestep;

    // Geometry: z = (PH+PHB)/g0, vertical destagger to mass levels. One staggered
    // field resident at a time (read PH, add PHB, drop both).
    let ph = read_3d_stag(&wrf, "PH", nz_stag, ny, nx, t)?;
    let phb = read_3d_stag(&wrf, "PHB", nz_stag, ny, nx, t)?;
    let mut geopot = ph;
    for (g, b) in geopot.iter_mut().zip(phb.iter()) {
        *g += *b;
    }
    drop(phb);
    for g in geopot.iter_mut() {
        *g /= optics::G0;
    }
    let z = destagger_vertical(&geopot, nz_stag, ny, nx); // native mass heights (f32)
    drop(geopot);

    // Full pressure p = P + PB (f32 Pa).
    let ncell = nz * ny * nx;
    let mut p = read_3d_required(&wrf, "P", nz, ny, nx, t)?;
    let pb = read_3d_required(&wrf, "PB", nz, ny, nx, t)?;
    for (pi, bi) in p.iter_mut().zip(pb.iter()) {
        *pi += *bi;
    }
    drop(pb);

    // Temperature (K) and air density (kg/m^3) at native levels.
    let theta = read_3d_required(&wrf, "T", nz, ny, nx, t)?;
    let mut t_kelvin = vec![0f32; ncell];
    let mut rho = vec![0f32; ncell];
    for (((tk_out, rho_out), &th), &pp) in t_kelvin
        .iter_mut()
        .zip(rho.iter_mut())
        .zip(theta.iter())
        .zip(p.iter())
    {
        let tk = optics::temperature_from_theta(th as f64, pp as f64);
        *tk_out = tk as f32;
        *rho_out = optics::air_density(pp as f64, tk) as f32;
    }
    drop(theta);
    drop(p);

    // Peak-RSS discipline: resample temperature first and drop the native field,
    // then build each extinction channel and quantize it to u8 immediately (adding
    // it into `beta_total` for tau_up) so no five-channel-wide f32 buffer set is
    // ever resident at once. One native 3-D field is read at a time and dropped.
    let temp_k_f32 = resample_volume(
        &t_kelvin,
        &z,
        nx,
        ny,
        nz,
        z_min,
        dz,
        nz_brick,
        Extrap::ClampEdge,
        Extrap::ClampEdge,
    );
    let temperature_f16 = bricks::encode_temperature_celsius(&temp_k_f32);
    drop(temp_k_f32);
    drop(t_kelvin);

    let mut beta_total = vec![0f32; nx * ny * nz_brick];

    // ext_liquid = QCLOUD.
    let (ql, ext_liquid) = {
        let beta = match read_3d_opt(&wrf, "QCLOUD", nz, ny, nx, t)? {
            Some(q) => beta_from_q(&q, &rho, HydrometeorClass::CloudLiquid.effective_radius_m()),
            None => vec![0f32; ncell],
        };
        let ext = resample_volume_conservative(
            &beta,
            &z,
            nx,
            ny,
            nz,
            z_min,
            dz,
            nz_brick,
            Extrap::Zero,
            Extrap::Zero,
        );
        accumulate_and_encode(&mut beta_total, &ext)
    };
    // ext_ice = QICE only (small pristine ice). SSB v3 snow-optics fix: QSNOW no
    // longer shares cloud-ice optics here — sharing them inflated snow's visible
    // extinction 3.75x wherever QSNOW dominates (anvil plates, stratiform shields,
    // deep decks), the "clouds too thick" defect. Snow now enters ext_precip below
    // at its own aggregate beta (see `optics::HydrometeorClass::Snow`).
    let (qi, ext_ice) = {
        let beta = match read_3d_opt(&wrf, "QICE", nz, ny, nx, t)? {
            Some(q) => beta_from_q(&q, &rho, HydrometeorClass::Ice.effective_radius_m()),
            None => vec![0f32; ncell],
        };
        let ext = resample_volume_conservative(
            &beta,
            &z,
            nx,
            ny,
            nz,
            z_min,
            dz,
            nz_brick,
            Extrap::Zero,
            Extrap::Zero,
        );
        accumulate_and_encode(&mut beta_total, &ext)
    };
    // ext_precip = QRAIN + QGRAUP (rain optics) + QSNOW (snow aggregate optics).
    // The channel is the LARGE-PARTICLE class: extinctions add linearly, so each
    // species converts at its OWN beta before the sum, and the single
    // per-unit-extinction IR recovery `ir.rs` applies to this channel (ratio
    // ~0.467) stays exact for all three because Q_abs/Q_ext is size-independent
    // in the geometric regime (see the optics.rs IR table). The visible march is
    // untouched: it consumes total extinction, and ext_ice + ext_precip already
    // share the ice phase lobe.
    let (qp, ext_precip) = {
        let mut beta = match read_3d_opt(&wrf, "QRAIN", nz, ny, nx, t)? {
            Some(q) => beta_from_q(&q, &rho, HydrometeorClass::Rain.effective_radius_m()),
            None => vec![0f32; ncell],
        };
        if let Some(qgraup) = read_3d_opt(&wrf, "QGRAUP", nz, ny, nx, t)? {
            add_beta_from_q(
                &mut beta,
                &qgraup,
                &rho,
                HydrometeorClass::Graupel.effective_radius_m(),
            );
        }
        if let Some(qsnow) = read_3d_opt(&wrf, "QSNOW", nz, ny, nx, t)? {
            add_beta_from_q(
                &mut beta,
                &qsnow,
                &rho,
                HydrometeorClass::Snow.effective_radius_m(),
            );
        }
        let ext = resample_volume_conservative(
            &beta,
            &z,
            nx,
            ny,
            nz,
            z_min,
            dz,
            nz_brick,
            Extrap::Zero,
            Extrap::Zero,
        );
        accumulate_and_encode(&mut beta_total, &ext)
    };
    drop(rho);

    // QVAPOR channel (owner decision 6): its own log-quantized channel. Resampled
    // with the same integral-conserving scheme as extinction (M0-review MAJOR-1):
    // qvapor is a non-negative concentration whose vertical INTEGRAL (the
    // precipitable-water path) is the physically meaningful quantity for the later
    // 6.2 um water-vapor IR band, so conserving that column beats point-sampling and
    // avoids dropping a thin moist layer between brick nodes. (Temperature, an
    // intensive quantity, stays point-sampled/linear below.)
    let (qv, qvapor) = {
        let qvraw =
            read_3d_opt(&wrf, "QVAPOR", nz, ny, nx, t)?.unwrap_or_else(|| vec![0f32; ncell]);
        let qvb = resample_volume_conservative(
            &qvraw,
            &z,
            nx,
            ny,
            nz,
            z_min,
            dz,
            nz_brick,
            Extrap::ClampEdge,
            Extrap::Zero,
        );
        bricks::encode_log_channel(&qvb)
    };
    drop(z);

    // tau_up from the accumulated total brick extinction.
    let (qt, tau_up) = {
        let tau_f32 = tau_up_volume(&beta_total, nx, ny, nz_brick, dz);
        bricks::encode_log_channel(&tau_f32)
    };
    drop(beta_total);

    // 2-D planes (HGT/LANDMASK/IVGTYP are time-invariant but read at `t` for a
    // fully self-consistent timestep-`t` payload; TSK/U10/V10/SNOWH vary in time).
    let hgt = read_2d_required(&wrf, "HGT", nx, ny, t)?;
    let landmask = read_2d_opt(&wrf, "LANDMASK", nx, ny, t)?.unwrap_or_else(|| vec![0f32; nx * ny]);
    let tsk = read_2d_opt(&wrf, "TSK", nx, ny, t)?.unwrap_or_else(|| vec![0f32; nx * ny]);
    let u10 = read_2d_opt(&wrf, "U10", nx, ny, t)?.unwrap_or_else(|| vec![0f32; nx * ny]);
    let v10 = read_2d_opt(&wrf, "V10", nx, ny, t)?.unwrap_or_else(|| vec![0f32; nx * ny]);
    let snowh = read_2d_opt(&wrf, "SNOWH", nx, ny, t)?;
    // IVGTYP best-effort. The netcrust fallback (for the int-dataset "no layout
    // message" case) does a full ~57 s metadata reopen on a 2 GB file, so it is
    // gated to modest grids to protect the ingest wall budget on large domains.
    let allow_netcrust = ncell <= NETCRUST_FALLBACK_MAX_CELLS;
    let ivgtyp = read_ivgtyp_best_effort(&wrf, path, nx, ny, t, allow_netcrust);

    let mut quant_map = std::collections::BTreeMap::new();
    quant_map.insert("ext_liquid".to_string(), ql);
    quant_map.insert("ext_ice".to_string(), qi);
    quant_map.insert("ext_precip".to_string(), qp);
    quant_map.insert("tau_up".to_string(), qt);
    quant_map.insert("qvapor".to_string(), qv);
    let quant = ChannelQuant(quant_map);

    let brick = VolumeBrick {
        nx,
        ny,
        nz: nz_brick,
        z_min_m: z_min,
        dz_m: dz,
        time_iso: geom.time_iso.clone(),
        quant: quant.clone(),
        ext_liquid,
        ext_ice,
        ext_precip,
        tau_up,
        qvapor,
        temperature_f16,
        hgt,
        landmask,
        tsk,
        u10,
        v10,
        snowh,
        ivgtyp,
    };

    // Write brick + manifest.
    let run_id = config
        .run_id
        .clone()
        .unwrap_or_else(|| run_id_from_path(path));
    let dir = bricks::run_dir(&config.cache_dir, &run_id);
    // Key the brick file + manifest entry on the full datetime (M0-review MINOR-2):
    // `t{YYYYMMDD_HHMM}.ssb`, so two timesteps at the same wall-clock HHMM on
    // different days of a >24 h run no longer collide.
    let stamp = bricks::time_stamp(geom.time_iso.as_deref(), geom.hhmm);
    let brick_file = bricks::brick_file_name(&stamp);
    let brick_path = dir.join(&brick_file);
    let ssb_bytes = bricks::write_ssb(&brick_path, &brick)?;

    let manifest_path = RunManifest::path(&config.cache_dir, &run_id);
    let planes_2d = brick.planes_2d_names();
    let projection = manifest_projection(&geom.params);
    let mut manifest = RunManifest::load_or_new(
        &manifest_path,
        &run_id,
        nx,
        ny,
        nz_brick,
        z_min,
        dz,
        planes_2d,
        projection,
    )?;
    // Source identity (staleness gate) + the per-timestep georef anchor (moving
    // nests; bit-identical cached-path reconstruction) ride on the timestep entry.
    let (source_bytes, source_mtime_unix) = source_identity(path);
    let anchor = geom.manifest_anchor();
    manifest.register_timestep(ManifestTimestep {
        key: stamp,
        hhmm: geom.hhmm,
        file: brick_file,
        time_iso: geom.time_iso,
        quant,
        ssb_bytes,
        source_bytes,
        source_mtime_unix,
        anchor,
    });
    manifest.save(&manifest_path)?;

    let wall = start.elapsed();
    let peak_rss_bytes = platform::peak_rss_bytes();
    crate::log_line!(
        "simsat ingest: run={run_id} dims={nx}x{ny}x{nz_brick} wall={:.2}s peak_rss={} ssb_bytes={ssb_bytes}",
        wall.as_secs_f64(),
        peak_rss_bytes
            .map(|b| format!("{:.1}MB", b as f64 / (1024.0 * 1024.0)))
            .unwrap_or_else(|| "n/a".to_string()),
    );

    Ok(IngestReport {
        run_id,
        brick_path,
        manifest_path,
        nx,
        ny,
        nz: nz_brick,
        hhmm: geom.hhmm,
        wall,
        peak_rss_bytes,
        ssb_bytes,
    })
}

/// Read a staggered 3-D field at time index `t` as f64 (kept in f64 for the
/// geopotential math).
fn read_3d_stag<R: VarReader>(
    wrf: &R,
    name: &str,
    nz_stag: usize,
    ny: usize,
    nx: usize,
    t: usize,
) -> Result<Vec<f64>, IngestError> {
    if !wrf.has_var(name) {
        return Err(IngestError::MissingVar(name.to_string()));
    }
    let values = wrf
        .read_var_t(name, t)
        .map_err(|e| IngestError::Wrf(format!("{name}: {e}")))?;
    let expected = nz_stag * ny * nx;
    if values.len() != expected {
        return Err(IngestError::Shape(format!(
            "{name}: expected {expected} (staggered), got {}",
            values.len()
        )));
    }
    Ok(values)
}

fn manifest_projection(p: &WrfProjectionParams) -> ManifestProjection {
    ManifestProjection {
        map_proj: p.map_proj,
        truelat1_deg: p.truelat1_deg,
        truelat2_deg: p.truelat2_deg,
        stand_lon_deg: p.stand_lon_deg,
        cen_lat_deg: p.cen_lat_deg,
        cen_lon_deg: p.cen_lon_deg,
        dx_m: p.dx_m,
        dy_m: p.dy_m,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertical_destagger_averages_faces() {
        // nz_stag=3, ny=1, nx=2. Faces at 0,100,300 (col A) and 0,200,600 (col B).
        let stag = vec![0.0, 0.0, 100.0, 200.0, 300.0, 600.0];
        let mass = destagger_vertical(&stag, 3, 1, 2);
        // mass level 0: 0.5*(face0+face1); level 1: 0.5*(face1+face2)
        assert_eq!(mass.len(), 4);
        assert!((mass[0] - 50.0).abs() < 1e-4); // col A lvl0 = 0.5*(0+100)
        assert!((mass[1] - 100.0).abs() < 1e-4); // col B lvl0 = 0.5*(0+200)
        assert!((mass[2] - 200.0).abs() < 1e-4); // col A lvl1 = 0.5*(100+300)
        assert!((mass[3] - 400.0).abs() < 1e-4); // col B lvl1 = 0.5*(200+600)
    }

    #[test]
    fn horizontal_destaggers_average_adjacent_faces() {
        // x-destagger: nz=1, ny=1, nx=2, staggered nx+1=3 faces [10,20,40].
        let sx = vec![10.0, 20.0, 40.0];
        let dx = destagger_x(&sx, 1, 1, 2);
        assert_eq!(dx, vec![15.0, 30.0]);
        // y-destagger: nz=1, ny=2, nx=1, staggered ny+1=3 faces column [10;20;40].
        let sy = vec![10.0, 20.0, 40.0];
        let dy = destagger_y(&sy, 1, 2, 1);
        assert_eq!(dy, vec![15.0, 30.0]);
    }

    #[test]
    fn resample_column_interpolates_and_extrapolates() {
        // Native heights 0, 1000, 2000 with values 10, 20, 30.
        let z = vec![0.0, 1000.0, 2000.0];
        let f = vec![10.0, 20.0, 30.0];
        let mut out = Vec::new();
        // Brick: z_min=0, dz=500, 6 levels: 0,500,1000,1500,2000,2500.
        resample_column(&z, &f, 0.0, 500.0, 6, Extrap::Zero, Extrap::Zero, &mut out);
        assert!((out[0] - 10.0).abs() < 1e-6); // z=0 edge
        assert!((out[1] - 15.0).abs() < 1e-6); // z=500 midpoint
        assert!((out[2] - 20.0).abs() < 1e-6); // z=1000
        assert!((out[3] - 25.0).abs() < 1e-6); // z=1500
        assert!((out[4] - 30.0).abs() < 1e-6); // z=2000 top edge
        assert!((out[5] - 0.0).abs() < 1e-6); // z=2500 above -> Zero
        // ClampEdge above holds the top value instead.
        resample_column(
            &z,
            &f,
            0.0,
            500.0,
            6,
            Extrap::Zero,
            Extrap::ClampEdge,
            &mut out,
        );
        assert!((out[5] - 30.0).abs() < 1e-6);
    }

    #[test]
    fn tau_up_on_analytic_slab() {
        // Uniform beta = 0.02 /m over 10 levels at dz=250 m.
        let beta = vec![0.02f64; 10];
        let dz = 250.0;
        let tau = integrate_tau_up_column(&beta, dz);
        // Top level tau=0; each step down adds 0.02*250 = 5 units.
        assert!((tau[9] - 0.0).abs() < 1e-9);
        assert!((tau[8] - 5.0).abs() < 1e-9);
        assert!((tau[0] - 45.0).abs() < 1e-9); // 9 steps of 5
        // A localized slab: beta nonzero only on levels 4..=6.
        let mut b2 = vec![0f64; 10];
        b2[4] = 0.1;
        b2[5] = 0.1;
        b2[6] = 0.1;
        let t2 = integrate_tau_up_column(&b2, dz);
        // Above the slab (levels 7..9) tau stays 0 except the trapezoid into 6.
        assert!((t2[9]).abs() < 1e-9);
        assert!((t2[7] - 0.5 * (b2[7] + b2[8]) * dz).abs() < 1e-9); // = 0
        // Below the slab tau is the full slab integral and stays constant.
        assert!(t2[0] > 0.0);
        assert!((t2[0] - t2[3]).abs() < 1e-9); // no extinction below level 4
    }

    #[test]
    fn beta_from_q_matches_optics_kernel() {
        let q = vec![1.0e-3f32, 0.0];
        let rho = vec![1.0f32, 1.0];
        let beta = beta_from_q(&q, &rho, HydrometeorClass::CloudLiquid.effective_radius_m());
        assert!((beta[0] - 0.15).abs() < 1e-6);
        assert_eq!(beta[1], 0.0);
    }

    /// SSB v3 snow-optics fix, the synthetic two-layer proof: through the real
    /// ingest kernels (`beta_from_q`/`add_beta_from_q` + `integrate_tau_up_column`),
    /// an equal-mass SNOW column now carries exactly r_snow/r_ice = 3.75x LESS
    /// visible optical depth than a cloud-ice column, while the cloud-ice column
    /// itself is byte-unchanged from the M0 constant (beta = 0.0375 m^-1 at
    /// 1 g/kg, rho_air = 1) — the convective core (QICE/QCLOUD-dominated) look is
    /// preserved and only snow-dominated regions thin.
    #[test]
    fn snow_layer_tau_drops_vs_ice_optics_and_ice_is_unchanged() {
        let nz = 10;
        let dz = 250.0;
        let q = vec![1.0e-3f32; nz]; // 1 g/kg through the whole column
        let rho = vec![1.0f32; nz];
        let ice_beta = beta_from_q(&q, &rho, HydrometeorClass::Ice.effective_radius_m());
        // The snow column enters through the SAME accumulation path the ingest
        // uses for ext_precip (rain absent -> zero base, snow added on top).
        let mut snow_beta = vec![0.0f32; nz];
        add_beta_from_q(
            &mut snow_beta,
            &q,
            &rho,
            HydrometeorClass::Snow.effective_radius_m(),
        );
        // Cloud-ice extinction is UNCHANGED from M0: 1.5*1*1e-3/(1000*40e-6).
        assert!(
            (ice_beta[0] - 0.0375).abs() < 1e-7,
            "ice beta {}",
            ice_beta[0]
        );
        // Snow extinction is its own aggregate beta: 1.5*1*1e-3/(1000*150e-6).
        assert!(
            (snow_beta[0] - 0.01).abs() < 1e-7,
            "snow beta {}",
            snow_beta[0]
        );
        let to_f64 = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
        let ice_tau = integrate_tau_up_column(&to_f64(&ice_beta), dz);
        let snow_tau = integrate_tau_up_column(&to_f64(&snow_beta), dz);
        // Surface-level column optical depths: 9 trapezoid steps of beta*dz
        // (tolerances allow the f32 channel quantization of beta).
        assert!((ice_tau[0] - 84.375).abs() < 1e-4, "ice tau {}", ice_tau[0]);
        assert!(
            (snow_tau[0] - 22.5).abs() < 1e-4,
            "snow tau {}",
            snow_tau[0]
        );
        // The fix factor: the equal-mass snow layer is exactly 3.75x thinner than
        // it was under the old shared-ice-optics treatment (which produced ice_tau).
        assert!(((ice_tau[0] / snow_tau[0]) - 3.75).abs() < 1e-5);
    }

    #[test]
    fn sanitize_and_time_parse() {
        assert_eq!(
            run_id_from_path(Path::new("/a/wrfout_d01_2018-10-10_12_00_00")),
            "wrfout_d01_2018_10_10_12_00_00"
        );
        let (hhmm, iso) = parse_wrf_time("2018-10-10_12:15:00");
        assert_eq!(hhmm, 1215);
        assert_eq!(iso, "2018-10-10T12:15:00Z");
    }

    #[test]
    fn default_cache_dir_is_nonempty() {
        assert!(!default_cache_dir().as_os_str().is_empty());
    }

    // ── M1-review MAJOR-1: the field reads must thread the selected timestep ──

    /// A `VarReader` whose returned data encodes the requested timestep in every
    /// element, so a read that hardcoded `t = 0` is detectable (the exact bug).
    struct FakeReader {
        nx: usize,
        ny: usize,
        nz: usize,
        nz_stag: usize,
    }
    impl VarReader for FakeReader {
        fn has_var(&self, _name: &str) -> bool {
            true
        }
        fn read_var_t(&self, name: &str, t: usize) -> Result<Vec<f64>, String> {
            let len = match name {
                "PH" | "PHB" => self.nz_stag * self.ny * self.nx,
                "HGT" | "LANDMASK" | "TSK" | "U10" | "V10" | "SNOWH" => self.ny * self.nx,
                _ => self.nz * self.ny * self.nx,
            };
            Ok(vec![t as f64; len])
        }
    }

    #[test]
    fn field_reads_thread_the_selected_timestep() {
        let r = FakeReader {
            nx: 3,
            ny: 2,
            nz: 4,
            nz_stag: 5,
        };
        // 3-D at t=2 must return all 2.0; a hardcoded-0 read would return all 0.0.
        let v2 = read_3d_opt(&r, "QCLOUD", 4, 2, 3, 2).unwrap().unwrap();
        assert_eq!(v2.len(), 4 * 2 * 3);
        assert!(
            v2.iter().all(|&x| x == 2.0),
            "3-D read must reach read_var(t=2)"
        );
        let v0 = read_3d_opt(&r, "QCLOUD", 4, 2, 3, 0).unwrap().unwrap();
        assert!(v0.iter().all(|&x| x == 0.0), "t=0 is distinct from t=2");
        // 2-D at t=3.
        let p = read_2d_opt(&r, "TSK", 3, 2, 3).unwrap().unwrap();
        assert!(
            p.iter().all(|&x| x == 3.0),
            "2-D read must reach read_var(t=3)"
        );
        // Staggered at t=4.
        let s = read_3d_stag(&r, "PH", 5, 2, 3, 4).unwrap();
        assert!(
            s.iter().all(|&x| x == 4.0),
            "staggered read must reach read_var(t=4)"
        );
        // The `_required` wrappers forward `t` too.
        let rq = read_3d_required(&r, "P", 4, 2, 3, 1).unwrap();
        assert!(rq.iter().all(|&x| x == 1.0));
        let r2 = read_2d_required(&r, "HGT", 3, 2, 5).unwrap();
        assert!(r2.iter().all(|&x| x == 5.0));
    }

    // ── WS3: geometry validation + moving-nest per-timestep anchoring ─────────

    /// A fake multi-time file whose XLAT/XLONG SHIFT with the timestep (a moving
    /// nest): lat/lon = base + 0.5*t + a small per-cell gradient. `nt = 3`.
    struct MovingNestFake {
        nx: usize,
        ny: usize,
        nz: usize,
        nz_stag: usize,
        nt: usize,
    }
    impl VarReader for MovingNestFake {
        fn has_var(&self, name: &str) -> bool {
            matches!(name, "XLAT" | "XLONG")
        }
        fn read_var_t(&self, name: &str, t: usize) -> Result<Vec<f64>, String> {
            let (nx, ny) = (self.nx, self.ny);
            let mut out = vec![0f64; nx * ny];
            for j in 0..ny {
                for i in 0..nx {
                    out[j * nx + i] = match name {
                        "XLAT" => 39.0 + 0.5 * t as f64 + 0.01 * j as f64,
                        "XLONG" => -97.5 + 0.5 * t as f64 + 0.01 * i as f64,
                        other => return Err(format!("unexpected var {other}")),
                    };
                }
            }
            Ok(out)
        }
    }
    impl GeomReader for MovingNestFake {
        fn dims(&self) -> (usize, usize, usize, usize) {
            (self.nx, self.ny, self.nz, self.nz_stag)
        }
        fn time_count(&self) -> usize {
            self.nt
        }
        fn time_labels(&self) -> Vec<String> {
            (0..self.nt)
                .map(|t| format!("2025-06-21_{:02}:15:00", 2 + t))
                .collect()
        }
        fn projection_params(&self) -> WrfProjectionParams {
            WrfProjectionParams {
                map_proj: 1,
                truelat1_deg: 30.0,
                truelat2_deg: 60.0,
                stand_lon_deg: -97.5,
                cen_lat_deg: 39.0,
                cen_lon_deg: -97.5,
                dx_m: 3000.0,
                dy_m: 3000.0,
            }
        }
    }

    fn nest(nx: usize, ny: usize, nz: usize, nz_stag: usize, nt: usize) -> MovingNestFake {
        MovingNestFake {
            nx,
            ny,
            nz,
            nz_stag,
            nt,
        }
    }

    /// WS3 item 1: a timestep at/past the file's time count is a CLEAN Shape error
    /// (the wrfout path previously fell through to whatever the field reads did),
    /// and the malformed `nz_stag != nz+1` relationship is refused (it previously
    /// reached a `nz_stag - 1` underflow / misindexed resample downstream).
    #[test]
    fn read_geometry_validates_timestep_and_staggered_dims() {
        // In-range timesteps pass and carry nt.
        let g = read_geometry(&nest(4, 3, 5, 6, 3), 2).expect("valid timestep");
        assert_eq!(g.nt, 3);
        // Out of range: clean, actionable error.
        let err = read_geometry(&nest(4, 3, 5, 6, 3), 3).unwrap_err();
        let msg = err.to_string();
        assert!(
            matches!(err, IngestError::Shape(_)) && msg.contains("out of range"),
            "unexpected: {msg}"
        );
        assert!(
            msg.contains("3 timestep(s)") && msg.contains("0..=2"),
            "message should name the valid range: {msg}"
        );
        // Malformed staggered dim: refused (both directions).
        for bad_stag in [5usize, 8] {
            let err = read_geometry(&nest(4, 3, 5, bad_stag, 3), 0).unwrap_err();
            assert!(
                err.to_string().contains("nz+1"),
                "nz_stag {bad_stag} should be refused: {err}"
            );
        }
    }

    /// WS3 item 3 (must FAIL against the old always-t-0 read): the geometry's
    /// XLAT/XLONG are read at the SELECTED timestep, so a moving nest's later
    /// timestep anchors at the nest's CURRENT position, and the persisted manifest
    /// anchor follows it.
    #[test]
    fn read_geometry_anchors_xlat_xlong_at_the_selected_timestep() {
        let fake = nest(5, 5, 4, 5, 3);
        let g0 = read_geometry(&fake, 0).unwrap();
        let g2 = read_geometry(&fake, 2).unwrap();
        let c = (5 / 2) * 5 + 5 / 2; // centre cell (ci = cj = 2)
        // t=0 centre: 39.0 + 0.01*2 = 39.02; t=2 centre: 40.0 + 0.01*2 = 40.02.
        assert!(
            (g0.xlat[c] - 39.02).abs() < 1e-4,
            "t0 centre lat {}",
            g0.xlat[c]
        );
        assert!(
            (g2.xlat[c] - 40.02).abs() < 1e-4,
            "t2 must read the MOVED nest coordinates, got centre lat {}",
            g2.xlat[c]
        );
        assert!((g2.xlong[c] - (-96.48)).abs() < 1e-4, "{}", g2.xlong[c]);
        // The persisted anchor follows the per-timestep coordinates.
        let a0 = g0.manifest_anchor().expect("anchor t0");
        let a2 = g2.manifest_anchor().expect("anchor t2");
        assert_eq!((a0.ref_i, a0.ref_j), (2.0, 2.0));
        assert!((a2.ref_lat_deg - a0.ref_lat_deg - 1.0).abs() < 1e-4);
        assert!((a2.ref_lon_deg - a0.ref_lon_deg - 1.0).abs() < 1e-4);
        // And the time label is the selected timestep's.
        assert_eq!(g2.time_iso.as_deref(), Some("2025-06-21T04:15:00Z"));
        assert_eq!(g2.hhmm, 415);
    }

    // ── M0-review MAJOR-1: the vertical resample must conserve column integrals ──

    /// Native trapezoidal column integral of a profile, for the conservation checks.
    fn native_column_integral(z: &[f64], f: &[f64]) -> f64 {
        (1..z.len())
            .map(|k| 0.5 * (f[k - 1] + f[k]) * (z[k] - z[k - 1]))
            .sum()
    }

    #[test]
    fn conservative_resample_preserves_a_thin_layer_between_nodes() {
        // The exact M0-review failure case: a cloud confined to a single native
        // level at 150 m, between brick nodes 0 and 250.
        let z = vec![50.0, 150.0, 250.0];
        let f = vec![0.0, 1.0, 0.0];
        let (z_min, dz, nz) = (0.0, 250.0, 4);

        // Point-sampling (temperature path) DROPS it: node 0 is below the column
        // (Zero), node 250 lands exactly on the f=0 top edge.
        let mut point = Vec::new();
        resample_column(
            &z,
            &f,
            z_min,
            dz,
            nz,
            Extrap::Zero,
            Extrap::Zero,
            &mut point,
        );
        assert!(
            point.iter().all(|&v| v.abs() < 1e-12),
            "point-sample drops the thin layer: {point:?}"
        );

        // Conservative resample keeps the mass and conserves the column integral.
        let mut cum = Vec::new();
        let mut cons = Vec::new();
        resample_column_conservative(
            &z,
            &f,
            z_min,
            dz,
            nz,
            Extrap::Zero,
            Extrap::Zero,
            &mut cum,
            &mut cons,
        );
        assert!(cons[0] > 0.0 && cons[1] > 0.0, "layer survives: {cons:?}");
        let col_brick: f64 = cons.iter().map(|v| v * dz).sum();
        let col_native = native_column_integral(&z, &f); // = 100
        assert!(
            (col_brick - col_native).abs() < 1e-9,
            "column OD not conserved: brick {col_brick} vs native {col_native}"
        );
    }

    #[test]
    fn conservative_resample_conserves_random_column_integrals() {
        // Deterministic LCG so the test is reproducible without a rand dependency.
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / (1u64 << 31) as f64 // in [0, 1)
        };
        let (z_min, dz, nz) = (0.0, 250.0, 40); // brick covers 0..9750 m (+/- half-cell)
        for _ in 0..12 {
            // 20 increasing native levels in ~[100, 5000] m, random non-negative f.
            let mut z = Vec::new();
            let mut zc = 100.0 + next() * 100.0;
            for _ in 0..20 {
                z.push(zc);
                zc += 100.0 + next() * 300.0;
            }
            let f: Vec<f64> = (0..20).map(|_| next() * 2.0).collect();
            let mut cum = Vec::new();
            let mut out = Vec::new();
            resample_column_conservative(
                &z,
                &f,
                z_min,
                dz,
                nz,
                Extrap::Zero,
                Extrap::Zero,
                &mut cum,
                &mut out,
            );
            let col_brick: f64 = out.iter().map(|v| v * dz).sum();
            let col_native = native_column_integral(&z, &f);
            // Exact for piecewise-linear (telescoping cumulative), modulo float.
            assert!(
                (col_brick - col_native).abs() <= 1e-6 * col_native.max(1.0),
                "column not conserved: brick {col_brick} vs native {col_native}"
            );
        }
    }

    #[test]
    fn tau_up_recovers_the_column_from_the_conserved_profile() {
        // A resolved slab (beta=0.02 /m over 500..3500 m) resampled conservatively,
        // then tau_up integrated from it, must recover the column optical depth at
        // the base to within a cell (trapezoidal-of-cell-averages half-cell error).
        let z: Vec<f64> = (0..40).map(|k| 250.0 + k as f64 * 100.0).collect();
        let f: Vec<f64> = z
            .iter()
            .map(|&zz| {
                if (500.0..=3500.0).contains(&zz) {
                    0.02
                } else {
                    0.0
                }
            })
            .collect();
        let (z_min, dz, nz) = (0.0, 250.0, 80);
        let mut cum = Vec::new();
        let mut beta = Vec::new();
        resample_column_conservative(
            &z,
            &f,
            z_min,
            dz,
            nz,
            Extrap::Zero,
            Extrap::Zero,
            &mut cum,
            &mut beta,
        );
        let col = native_column_integral(&z, &f); // ~ 0.02 * 3000 = 60
        let tau = integrate_tau_up_column(&beta, dz);
        // tau is monotonic non-increasing with height, ~0 at top, and the base
        // recovers the column to within one cell's optical depth.
        assert!(tau[nz - 1].abs() < 1e-9, "top tau ~ 0");
        assert!(
            (tau[0] - col).abs() < 0.02 * dz,
            "tau base {} should recover column {col} within a cell",
            tau[0]
        );
        for m in 1..nz {
            assert!(tau[m - 1] >= tau[m] - 1e-12, "tau_up monotonic downward");
        }
    }
}
