//! Sat-store frame writer (design doc section 7, M1 slice).
//!
//! Writes a rendered visible frame as the three baked `rgb_r`/`rgb_g`/`rgb_b`
//! planes BowEcho's composite path renders verbatim, under a `simsat` model dir,
//! on the per-pixel lat/lon mesh from the scan-angle grid (this mesh is what makes
//! BowEcho's map layer work). The writing machinery is the pinned `rw_store` crate
//! (`HourWriter`/`write_grid`/`RwsRunManifest`/`RunLock`); the run-naming,
//! bit-identical-grid reuse, and composite-plane layout are PORTED (structure)
//! from BowEcho's composite writer in `crates/app_ui/src/sat_worker.rs`
//! (`write_himawari_grid_frame` sibling / `COMPOSITE_R/G/B_VAR`).
//!
//! Store layout: `{store_root}/simsat/{sector}_rgb_{satslug}_{YYYYMMDD}[_k]/`
//! with `grid.rwg` + `t{HHMM}.rws` + `run.json`. Owner points BowEcho's sat store
//! dir at `{store_root}` to see `simsat` in the Satellite window.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rustwx_core::{GridShape, LatLonGrid};
use rw_sat::store::frame_file_name;
use rw_store::format::RwsWriterInfo;
use rw_store::grid::{GridFile, write_grid};
use rw_store::lock::RunLock;
use rw_store::reader::HourReader;
use rw_store::run::{RwsHourEntry, RwsRunManifest};
use rw_store::writer::HourWriter;

use crate::camera::{PERSPECTIVE_HEIGHT_M, SatellitePreset};
use crate::gpu::RenderedFrame;
use crate::optics::EARTH_RADIUS_M;

/// The model dir every SimSat run lives under.
pub const SIMSAT_MODEL: &str = "simsat";

/// The three composite plane variable names BowEcho detects a composite by.
pub const COMPOSITE_R_VAR: &str = "rgb_r";
pub const COMPOSITE_G_VAR: &str = "rgb_g";
pub const COMPOSITE_B_VAR: &str = "rgb_b";

const WRITER_BUILD: &str = concat!("simsat_studio ", env!("CARGO_PKG_VERSION"));

/// A rendered visible frame ready to write: three `[0,255]` planes (`NaN` =
/// transparent / off-earth / night-black is kept opaque, only space is NaN) plus
/// the per-pixel lat/lon mesh, all row-major `ny*nx`, row 0 = north.
#[derive(Debug, Clone)]
pub struct VisibleFrame {
    pub nx: usize,
    pub ny: usize,
    pub rgb_r: Vec<f32>,
    pub rgb_g: Vec<f32>,
    pub rgb_b: Vec<f32>,
    pub lat: Vec<f32>,
    pub lon: Vec<f32>,
    /// Domain token (sector) for the run name (sanitized on write).
    pub sector: String,
    pub satellite: SatellitePreset,
    /// Band label recorded in the selector (metadata only; 2 = ABI red-vis class).
    pub band: u8,
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub hhmm: u16,
}

impl VisibleFrame {
    /// Assemble planes + mesh from a GPU-rendered frame and its surface raster.
    /// `rendered.rgba` alpha is the on-earth mask: `a < 128` -> off-earth -> the
    /// plane value is `NaN` (transparent space); otherwise the sRGB byte in
    /// `[0,255]`. The mesh comes from the raster (row 0 = north, same order).
    #[allow(clippy::too_many_arguments)]
    pub fn from_rendered(
        rendered: &RenderedFrame,
        lat: Vec<f32>,
        lon: Vec<f32>,
        sector: String,
        satellite: SatellitePreset,
        year: i32,
        month: u32,
        day: u32,
        hhmm: u16,
    ) -> Self {
        let n = (rendered.width * rendered.height) as usize;
        let mut rgb_r = vec![f32::NAN; n];
        let mut rgb_g = vec![f32::NAN; n];
        let mut rgb_b = vec![f32::NAN; n];
        for i in 0..n {
            let a = rendered.rgba[i * 4 + 3];
            if a >= 128 {
                rgb_r[i] = rendered.rgba[i * 4] as f32;
                rgb_g[i] = rendered.rgba[i * 4 + 1] as f32;
                rgb_b[i] = rendered.rgba[i * 4 + 2] as f32;
            }
        }
        Self {
            nx: rendered.width as usize,
            ny: rendered.height as usize,
            rgb_r,
            rgb_g,
            rgb_b,
            lat,
            lon,
            sector,
            satellite,
            band: 2,
            year,
            month,
            day,
            hhmm,
        }
    }

    fn day_token(&self) -> String {
        format!("{:04}{:02}{:02}", self.year, self.month, self.day)
    }

    fn scan_start_rfc3339(&self) -> String {
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:00Z",
            self.year,
            self.month,
            self.day,
            self.hhmm / 100,
            self.hhmm % 100
        )
    }

    fn selector(&self) -> serde_json::Value {
        // Spherical M1 projection (a = b = R): recorded for metadata; the map
        // layer uses the explicit grid.rwg mesh, not this. sweep_angle_axis "y"
        // matches the CGMS sweep=y math the camera actually uses.
        let projection = serde_json::json!({
            "perspective_point_height_m": PERSPECTIVE_HEIGHT_M,
            "semi_major_axis_m": EARTH_RADIUS_M,
            "semi_minor_axis_m": EARTH_RADIUS_M,
            "longitude_of_projection_origin_deg": self.satellite.sub_lon_deg(),
            "sweep_angle_axis": "y",
        });
        serde_json::json!({
            "satellite": {
                "provider": "simsat",
                "instrument": "synthetic-visible",
                "satellite": self.satellite.slug(),
                "model": SIMSAT_MODEL,
                "product": "surface_visible",
                "sector": sanitize_store_token(&self.sector),
                "band": self.band,
                "layer": "rgb_simsat",
                "source_variable": "wrf_surface",
                "scan_start_utc": self.scan_start_rfc3339(),
                "scan_end_utc": self.scan_start_rfc3339(),
                "projection": projection,
            }
        })
    }
}

/// What a store write produced.
#[derive(Debug, Clone)]
pub struct WrittenVisibleFrame {
    pub model: String,
    pub run: String,
    pub run_dir: PathBuf,
    pub frame_path: PathBuf,
    pub grid_path: PathBuf,
    pub created_run: bool,
    pub bytes: u64,
    pub hhmm: u16,
}

/// ascii-alnum lowercased token; others -> `_`, collapsed, trimmed; empty ->
/// `"unknown"`. Mirrors `sat_worker::sanitize_store_token`.
pub fn sanitize_store_token(value: &str) -> String {
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
        "unknown".to_string()
    } else {
        trimmed
    }
}

fn coords_bit_identical(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// A resolved run directory ready for a frame write: the run name, its dir, the
/// `grid.rwg` content hash, whether the run was freshly created, and the held run
/// lock (kept alive for the duration of the frame write). Shared by the visible and
/// IR writers so the bit-identical-grid reuse + `{run_base}[_k]` naming is one
/// implementation.
struct ResolvedRun {
    run_name: String,
    run_dir: PathBuf,
    grid_hash: String,
    created_run: bool,
    _lock: RunLock,
}

/// Resolve (reuse or create) the run dir for a `{model}/{run_base}` frame under
/// `store_root` on the given grid: scan the model dir for a run whose `grid.rwg` is
/// bit-identical (reuse it — same domain, later timestep) else take the first free
/// `{run_base}[_k]` name (a moved/rescaled grid forks a fresh run). Creates the dir,
/// acquires the 60 s frame lock, and writes `grid.rwg` when the run is new. Ported
/// from `sat_worker::write_himawari_grid_frame` (the grid-reuse rule).
fn resolve_run(
    store_root: &Path,
    model: &str,
    run_base: &str,
    grid: &LatLonGrid,
    nx: usize,
    ny: usize,
) -> Result<ResolvedRun, String> {
    let model_dir = store_root.join(model);
    let mut candidates: Vec<String> = Vec::new();
    if model_dir.is_dir() {
        for entry in std::fs::read_dir(&model_dir).map_err(|e| e.to_string())? {
            let name = entry
                .map_err(|e| e.to_string())?
                .file_name()
                .to_string_lossy()
                .to_string();
            if name == run_base || name.starts_with(&format!("{run_base}_")) {
                candidates.push(name);
            }
        }
    }
    candidates.sort();
    let mut resolved: Option<(String, String)> = None;
    for name in &candidates {
        let grid_path = model_dir.join(name).join("grid.rwg");
        if !grid_path.is_file() {
            continue;
        }
        let existing = GridFile::open(&grid_path).map_err(|e| e.to_string())?;
        if existing.nx == nx
            && existing.ny == ny
            && coords_bit_identical(&existing.lat, &grid.lat_deg)
            && coords_bit_identical(&existing.lon, &grid.lon_deg)
        {
            resolved = Some((name.clone(), existing.hash));
            break;
        }
    }
    let created_run = resolved.is_none();
    let (run_name, existing_grid_hash) = match resolved {
        Some((name, hash)) => (name, Some(hash)),
        None => {
            let mut suffix = 1usize;
            loop {
                let name = if suffix == 1 {
                    run_base.to_string()
                } else {
                    format!("{run_base}_{suffix}")
                };
                if !candidates.contains(&name) {
                    break (name, None);
                }
                suffix += 1;
            }
        }
    };

    let run_dir = model_dir.join(&run_name);
    std::fs::create_dir_all(&run_dir).map_err(|e| e.to_string())?;
    let lock = RunLock::acquire(&run_dir, Duration::from_secs(60)).map_err(|e| e.to_string())?;
    let grid_path = run_dir.join("grid.rwg");
    let grid_hash = match existing_grid_hash {
        Some(hash) => hash,
        None => write_grid(&grid_path, grid, None).map_err(|e| e.to_string())?,
    };
    Ok(ResolvedRun {
        run_name,
        run_dir,
        grid_hash,
        created_run,
        _lock: lock,
    })
}

/// Write a visible frame under `store_root`. Reuses an existing run whose
/// `grid.rwg` is bit-identical (a moved/rescaled grid opens a fresh run dir).
pub fn write_visible_frame(
    store_root: &Path,
    frame: &VisibleFrame,
) -> Result<WrittenVisibleFrame, String> {
    let n = frame.nx * frame.ny;
    for (name, plane) in [
        ("rgb_r", &frame.rgb_r),
        ("rgb_g", &frame.rgb_g),
        ("rgb_b", &frame.rgb_b),
    ] {
        if plane.len() != n {
            return Err(format!("{name}: expected {n} values, got {}", plane.len()));
        }
    }
    if frame.lat.len() != n || frame.lon.len() != n {
        return Err(format!(
            "mesh length mismatch: lat {} lon {} vs {n}",
            frame.lat.len(),
            frame.lon.len()
        ));
    }

    let model = SIMSAT_MODEL.to_string();
    let sector = sanitize_store_token(&frame.sector);
    let day = frame.day_token();
    let run_base = format!("{sector}_rgb_{}_{day}", frame.satellite.slug());

    let shape = GridShape::new(frame.nx, frame.ny).map_err(|e| e.to_string())?;
    let grid =
        LatLonGrid::new(shape, frame.lat.clone(), frame.lon.clone()).map_err(|e| e.to_string())?;

    // Bit-identical-grid reuse + run naming (shared with the IR writer).
    let resolved = resolve_run(store_root, &model, &run_base, &grid, frame.nx, frame.ny)?;
    let run_name = resolved.run_name.clone();
    let run_dir = resolved.run_dir.clone();
    let created_run = resolved.created_run;
    let grid_path = run_dir.join("grid.rwg");
    let grid_hash = resolved.grid_hash.clone();

    let started = Instant::now();
    let selector = frame.selector();
    let mut writer = HourWriter::new(
        &model,
        &run_name,
        frame.hhmm,
        frame.nx,
        frame.ny,
        &grid_hash,
        WRITER_BUILD,
    );
    writer
        .add_surface2d(COMPOSITE_R_VAR, "rgb8", selector, &frame.rgb_r)
        .map_err(|e| e.to_string())?;
    writer
        .add_surface2d(
            COMPOSITE_G_VAR,
            "rgb8",
            serde_json::Value::Null,
            &frame.rgb_g,
        )
        .map_err(|e| e.to_string())?;
    writer
        .add_surface2d(
            COMPOSITE_B_VAR,
            "rgb8",
            serde_json::Value::Null,
            &frame.rgb_b,
        )
        .map_err(|e| e.to_string())?;
    let file_name = frame_file_name(frame.hhmm);
    let frame_path = run_dir.join(&file_name);
    writer.finish(&frame_path).map_err(|e| e.to_string())?;
    let encode_ms = started.elapsed().as_millis() as u64;
    let bytes = std::fs::metadata(&frame_path)
        .map_err(|e| e.to_string())?
        .len();
    let written_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let manifest_path = run_dir.join("run.json");
    let writer_info = RwsWriterInfo {
        name: "simsat".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        build: WRITER_BUILD.to_string(),
    };
    let mut manifest = RwsRunManifest::load_or_new(
        &manifest_path,
        &model,
        &run_name,
        &grid_hash,
        frame.nx,
        frame.ny,
        writer_info,
    )
    .map_err(|e| e.to_string())?;
    manifest.register_hour(
        frame.hhmm,
        RwsHourEntry {
            file: file_name,
            written_unix,
            encode_ms,
            variables: vec![
                COMPOSITE_R_VAR.to_string(),
                COMPOSITE_G_VAR.to_string(),
                COMPOSITE_B_VAR.to_string(),
            ],
        },
    );
    manifest.save(&manifest_path).map_err(|e| e.to_string())?;

    Ok(WrittenVisibleFrame {
        model,
        run: run_name,
        run_dir,
        frame_path,
        grid_path,
        created_run,
        bytes,
        hhmm: frame.hhmm,
    })
}

// ── IR (band 13) single-band Kelvin frame (design section 7, M6) ──────────────

/// A rendered IR frame ready to write: one true-Kelvin brightness-temperature
/// plane (`NaN` = off-earth / out-of-domain, the store's transparent marker) plus
/// the per-pixel lat/lon mesh, all row-major `ny*nx`, row 0 = north. Written as a
/// SINGLE-BAND `surface2d` variable so BowEcho (or the studio) re-enhances it live
/// with `ir_enhancement_anchors(band, ...)`.
#[derive(Debug, Clone)]
pub struct IrFrame {
    pub nx: usize,
    pub ny: usize,
    /// Brightness temperature (K); `NaN` off-earth / out-of-domain.
    pub bt: Vec<f32>,
    pub lat: Vec<f32>,
    pub lon: Vec<f32>,
    /// Domain token (sector) for the run name (sanitized on write).
    pub sector: String,
    pub satellite: SatellitePreset,
    /// ABI band number: 13 for the 10.3 um clean-window IR, or 8/9/10 for the
    /// 6.2/6.9/7.3 um water-vapor bands (same single-band Kelvin store contract).
    pub band: u8,
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub hhmm: u16,
}

impl IrFrame {
    /// Assemble a band-13 (10.3 um clean-window) IR frame from a BT plane (Kelvin,
    /// `NaN` off-earth) + mesh. Convenience wrapper over [`IrFrame::new_band`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nx: usize,
        ny: usize,
        bt: Vec<f32>,
        lat: Vec<f32>,
        lon: Vec<f32>,
        sector: String,
        satellite: SatellitePreset,
        year: i32,
        month: u32,
        day: u32,
        hhmm: u16,
    ) -> Self {
        Self::new_band(
            nx, ny, bt, lat, lon, sector, satellite, 13, year, month, day, hhmm,
        )
    }

    /// Assemble an IR frame for an explicit ABI band from a BT plane (Kelvin, `NaN`
    /// off-earth) + mesh. The water-vapor bands (8/9/10) reuse this with the same
    /// single-band Kelvin store contract as band 13 — only the `band` selector +
    /// `ahi_bt_c{band:02}` variable / `_c{band:02}_` run key differ, so BowEcho
    /// re-enhances each band through its own `band_anchors(band)` (the WV moisture
    /// palette for 8/9/10). Synthetic WV medians (~230-260 K) stay below the 320 K
    /// legacy-stretch threshold, so they classify as true Kelvin.
    #[allow(clippy::too_many_arguments)]
    pub fn new_band(
        nx: usize,
        ny: usize,
        bt: Vec<f32>,
        lat: Vec<f32>,
        lon: Vec<f32>,
        sector: String,
        satellite: SatellitePreset,
        band: u8,
        year: i32,
        month: u32,
        day: u32,
        hhmm: u16,
    ) -> Self {
        Self {
            nx,
            ny,
            bt,
            lat,
            lon,
            sector,
            satellite,
            band,
            year,
            month,
            day,
            hhmm,
        }
    }

    /// The single-band store variable name for this band: `ahi_bt_c{band:02}`
    /// (e.g. `ahi_bt_c13`). The `ahi_bt_` prefix + band-8..16 range is what makes
    /// BowEcho's legacy-stretch heuristic classify it as true Kelvin.
    pub fn variable_name(&self) -> String {
        format!("ahi_bt_c{:02}", self.band)
    }

    fn day_token(&self) -> String {
        format!("{:04}{:02}{:02}", self.year, self.month, self.day)
    }

    fn scan_start_rfc3339(&self) -> String {
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:00Z",
            self.year,
            self.month,
            self.day,
            self.hhmm / 100,
            self.hhmm % 100
        )
    }

    /// The band-13 satellite selector JSON — the `himawari_selector` single-band
    /// shape (design section 7 / digest section 4). `sweep_angle_axis: "y"` matches
    /// the camera's CGMS sweep=y math; `band` is what keys the enhancement.
    fn selector(&self) -> serde_json::Value {
        let projection = serde_json::json!({
            "perspective_point_height_m": PERSPECTIVE_HEIGHT_M,
            "semi_major_axis_m": EARTH_RADIUS_M,
            "semi_minor_axis_m": EARTH_RADIUS_M,
            "longitude_of_projection_origin_deg": self.satellite.sub_lon_deg(),
            "sweep_angle_axis": "y",
        });
        serde_json::json!({
            "satellite": {
                "provider": "simsat",
                "instrument": "synthetic-ir",
                "satellite": self.satellite.slug(),
                "model": SIMSAT_MODEL,
                "product": "synthetic_ir",
                "sector": sanitize_store_token(&self.sector),
                "band": self.band,
                "layer": "ir_simsat",
                "source_variable": "wrf_ir_bt",
                "scan_start_utc": self.scan_start_rfc3339(),
                "scan_end_utc": self.scan_start_rfc3339(),
                "projection": projection,
            }
        })
    }
}

/// Write an IR frame under `store_root` as a SINGLE-BAND true-Kelvin `surface2d`
/// (design section 7). Run name `{sector}_c{band:02}_{day}` (NO `_rgb_` token) under
/// the `simsat` model dir, one `ahi_bt_c{band:02}` variable (units `K`) carrying the
/// Kelvin BT, the band-13 selector, and the bit-identical `grid.rwg` reuse rule. The
/// `_c13_` single-band path (not `_rgb_`) is what lets BowEcho re-enhance it live:
/// BowEcho's `_rgb_`-only display filter is scoped to GOES runs, and `simsat` is its
/// own model — the single-band `_c13_` run is the Himawari-style path BowEcho shows
/// and recolours (digest section 8). Synthetic BT medians (~290 K) are below the
/// 320 K legacy threshold, so it classifies as true Kelvin, not a percentile stretch.
pub fn write_ir_frame(store_root: &Path, frame: &IrFrame) -> Result<WrittenVisibleFrame, String> {
    let n = frame.nx * frame.ny;
    if frame.bt.len() != n {
        return Err(format!("bt: expected {n} values, got {}", frame.bt.len()));
    }
    if frame.lat.len() != n || frame.lon.len() != n {
        return Err(format!(
            "mesh length mismatch: lat {} lon {} vs {n}",
            frame.lat.len(),
            frame.lon.len()
        ));
    }

    let model = SIMSAT_MODEL.to_string();
    let sector = sanitize_store_token(&frame.sector);
    let day = frame.day_token();
    // Single-band run key `{sector}_c{band:02}_{day}` (no `_rgb_`).
    let run_base = format!("{sector}_c{:02}_{day}", frame.band);

    let shape = GridShape::new(frame.nx, frame.ny).map_err(|e| e.to_string())?;
    let grid =
        LatLonGrid::new(shape, frame.lat.clone(), frame.lon.clone()).map_err(|e| e.to_string())?;

    let resolved = resolve_run(store_root, &model, &run_base, &grid, frame.nx, frame.ny)?;
    let run_name = resolved.run_name.clone();
    let run_dir = resolved.run_dir.clone();
    let created_run = resolved.created_run;
    let grid_hash = resolved.grid_hash.clone();

    let started = Instant::now();
    let variable = frame.variable_name();
    let selector = frame.selector();
    let mut writer = HourWriter::new(
        &model,
        &run_name,
        frame.hhmm,
        frame.nx,
        frame.ny,
        &grid_hash,
        WRITER_BUILD,
    );
    // One single-band Kelvin surface2d (units "K"): the true-Kelvin BT plane.
    writer
        .add_surface2d(&variable, "K", selector, &frame.bt)
        .map_err(|e| e.to_string())?;
    let file_name = frame_file_name(frame.hhmm);
    let frame_path = run_dir.join(&file_name);
    writer.finish(&frame_path).map_err(|e| e.to_string())?;
    let encode_ms = started.elapsed().as_millis() as u64;
    let bytes = std::fs::metadata(&frame_path)
        .map_err(|e| e.to_string())?
        .len();
    let written_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let manifest_path = run_dir.join("run.json");
    let writer_info = RwsWriterInfo {
        name: "simsat".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        build: WRITER_BUILD.to_string(),
    };
    let mut manifest = RwsRunManifest::load_or_new(
        &manifest_path,
        &model,
        &run_name,
        &grid_hash,
        frame.nx,
        frame.ny,
        writer_info,
    )
    .map_err(|e| e.to_string())?;
    manifest.register_hour(
        frame.hhmm,
        RwsHourEntry {
            file: file_name,
            written_unix,
            encode_ms,
            variables: vec![variable.clone()],
        },
    );
    manifest.save(&manifest_path).map_err(|e| e.to_string())?;

    let grid_path = run_dir.join("grid.rwg");
    Ok(WrittenVisibleFrame {
        model,
        run: run_name,
        run_dir,
        frame_path,
        grid_path,
        created_run,
        bytes,
        hhmm: frame.hhmm,
    })
}

// ── Store-run readback (the GIF/animation export half) ────────────────────────

/// One frame of a completed store run: its valid time (`hhmm`) and `t{HHMM}.rws`
/// path.
#[derive(Debug, Clone)]
pub struct RunFrameRef {
    pub hhmm: u16,
    pub path: PathBuf,
}

/// A completed store run indexed for readback: the grid dims (from `grid.rwg`)
/// plus every frame file, sorted by valid time.
#[derive(Debug, Clone)]
pub struct RunFrameIndex {
    pub nx: usize,
    pub ny: usize,
    pub frames: Vec<RunFrameRef>,
}

/// Parse a store frame file name back to its `hhmm`. Accepts EXACTLY the names
/// [`frame_file_name`] produces (round-trip checked), so stray files in a run dir
/// are never misread as frames.
fn parse_frame_file_name(name: &str) -> Option<u16> {
    let digits = name.strip_prefix('t')?.strip_suffix(".rws")?;
    let hhmm: u16 = digits.parse().ok()?;
    (frame_file_name(hhmm) == name).then_some(hhmm)
}

/// Index a completed store RUN directory (the folder holding `grid.rwg` +
/// `t{HHMM}.rws` + `run.json`) for frame readback: dims from `grid.rwg`, frames
/// discovered by their canonical file names and sorted by valid time. Works for
/// ANY completed run — including runs longer than the studio's in-memory
/// `frame_cap` (the store is the full-loop persistence).
pub fn list_run_frames(run_dir: &Path) -> Result<RunFrameIndex, String> {
    let grid_path = run_dir.join("grid.rwg");
    let grid = GridFile::open(&grid_path)
        .map_err(|e| format!("{}: {e} (not a store run dir?)", grid_path.display()))?;
    let mut frames = Vec::new();
    for entry in std::fs::read_dir(run_dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name().to_string_lossy().to_string();
        if let Some(hhmm) = parse_frame_file_name(&name) {
            frames.push(RunFrameRef {
                hhmm,
                path: entry.path(),
            });
        }
    }
    frames.sort_by_key(|f| f.hhmm);
    Ok(RunFrameIndex {
        nx: grid.nx,
        ny: grid.ny,
        frames,
    })
}

/// Read one VISIBLE store frame back as interleaved RGB8 (`nx * ny * 3`, row 0 =
/// north). The three `rgb_r/g/b` planes hold exact `[0,255]` byte values (the
/// write path is lossless for 2-D planes), so the conversion is exact; `NaN`
/// (off-earth / space) pixels become black `(0,0,0)`, matching the studio
/// display. `nx`/`ny` come from the run's [`RunFrameIndex`].
pub fn read_visible_frame_rgb(frame_path: &Path, nx: usize, ny: usize) -> Result<Vec<u8>, String> {
    let reader =
        HourReader::open(frame_path).map_err(|e| format!("{}: {e}", frame_path.display()))?;
    let n = nx * ny;
    let mut rgb = vec![0u8; n * 3];
    for (c, var) in [COMPOSITE_R_VAR, COMPOSITE_G_VAR, COMPOSITE_B_VAR]
        .iter()
        .enumerate()
    {
        let vals = reader
            .read_full_2d(var)
            .map_err(|e| format!("{}: {var}: {e}", frame_path.display()))?;
        if vals.len() != n {
            return Err(format!(
                "{}: {var}: {} values, expected {n}",
                frame_path.display(),
                vals.len()
            ));
        }
        for (k, &v) in vals.iter().enumerate() {
            if v.is_finite() {
                rgb[k * 3 + c] = v.clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(rgb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rw_store::reader::HourReader;
    use std::sync::atomic::{AtomicU64, Ordering};

    static UNIQ: AtomicU64 = AtomicU64::new(0);

    fn temp_root() -> PathBuf {
        let n = UNIQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("simsat-store-{}-{}", std::process::id(), n));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn synthetic_frame(nx: usize, ny: usize) -> VisibleFrame {
        let n = nx * ny;
        let mut lat = vec![0f32; n];
        let mut lon = vec![0f32; n];
        let mut r = vec![0f32; n];
        let g = vec![128f32; n];
        let b = vec![64f32; n];
        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                lat[idx] = 45.0 - j as f32 * 0.1; // row 0 = north
                lon[idx] = -100.0 + i as f32 * 0.1;
                r[idx] = (i * 10 % 255) as f32;
            }
        }
        VisibleFrame {
            nx,
            ny,
            rgb_r: r,
            rgb_g: g,
            rgb_b: b,
            lat,
            lon,
            sector: "enderlin d03".to_string(),
            satellite: SatellitePreset::GoesEast,
            band: 2,
            year: 2025,
            month: 6,
            day: 21,
            hhmm: 215,
        }
    }

    #[test]
    fn write_and_read_back_a_composite_frame() {
        let root = temp_root();
        let frame = synthetic_frame(12, 8);
        let written = write_visible_frame(&root, &frame).expect("write");
        assert!(written.created_run);
        assert_eq!(written.model, "simsat");
        assert_eq!(written.run, "enderlin_d03_rgb_goese_20250621");
        assert!(written.frame_path.is_file());
        assert!(written.grid_path.is_file());
        assert!(written.run_dir.join("run.json").is_file());

        // Grid readback: dims + the exact per-pixel mesh.
        let grid = GridFile::open(&written.grid_path).expect("grid");
        assert_eq!((grid.nx, grid.ny), (12, 8));
        assert!(coords_bit_identical(&grid.lat, &frame.lat));
        assert!(coords_bit_identical(&grid.lon, &frame.lon));
        // Row 0 north -> lat descending with row -> Some(true).
        assert_eq!(grid.lat_descending(), Some(true));

        // Frame readback: the three planes + the selector shape on rgb_r. Compare
        // the VALUES (not just the length): the synthetic frame's r/g/b are all
        // distinct (r = i*10%255, g = 128, b = 64), so a swapped rgb plane, a
        // wrong-ordered mesh, or a dropped pixel is caught here. 2-D tiles are
        // lossless (zstd raw f32), so the round trip is bit-exact.
        let reader = HourReader::open(&written.frame_path).expect("frame");
        for (name, expected) in [
            (COMPOSITE_R_VAR, &frame.rgb_r),
            (COMPOSITE_G_VAR, &frame.rgb_g),
            (COMPOSITE_B_VAR, &frame.rgb_b),
        ] {
            let vals = reader
                .read_full_2d(name)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(vals.len(), 12 * 8);
            for (k, (&got, &want)) in vals.iter().zip(expected.iter()).enumerate() {
                assert_eq!(got, want, "{name}[{k}] readback value mismatch");
            }
        }
        let r_var = reader.variable(COMPOSITE_R_VAR).expect("rgb_r var");
        let sat = r_var
            .selector
            .get("satellite")
            .expect("satellite selector object");
        assert_eq!(sat.get("model").and_then(|v| v.as_str()), Some("simsat"));
        assert_eq!(sat.get("band").and_then(|v| v.as_u64()), Some(2));
        assert_eq!(
            sat.get("projection")
                .and_then(|p| p.get("sweep_angle_axis"))
                .and_then(|v| v.as_str()),
            Some("y")
        );
        // G/B carry a null selector.
        assert!(reader.variable(COMPOSITE_G_VAR).unwrap().selector.is_null());

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn identical_grid_reuses_the_run_a_moved_grid_forks() {
        let root = temp_root();
        let frame = synthetic_frame(10, 6);
        let first = write_visible_frame(&root, &frame).unwrap();
        assert!(first.created_run);
        // Same grid, next timestep -> reuse the run (no new dir).
        let mut frame2 = frame.clone();
        frame2.hhmm = 230;
        let second = write_visible_frame(&root, &frame2).unwrap();
        assert!(!second.created_run);
        assert_eq!(second.run, first.run);
        // A moved grid (shifted mesh) forks a fresh run dir.
        let mut moved = frame.clone();
        for v in &mut moved.lat {
            *v += 5.0;
        }
        let third = write_visible_frame(&root, &moved).unwrap();
        assert!(third.created_run);
        assert_ne!(third.run, first.run);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn sanitize_store_token_rules() {
        assert_eq!(sanitize_store_token("Enderlin d03!!"), "enderlin_d03");
        assert_eq!(sanitize_store_token("__--__"), "unknown");
        assert_eq!(sanitize_store_token("GOES-East"), "goes_east");
    }

    #[test]
    fn a_transposed_mesh_is_detected_by_the_readback() {
        // The correct frame is north-up (lat descends down rows). A mesh transposed
        // (i<->j swapped) must NOT read back as north-up and must differ from the
        // correct mesh — so the ordering assertion that guards a correct frame would
        // FAIL for a transposed one (M1-review MINOR-2: catch a wrong-ordered mesh).
        let root = temp_root();
        let n = 8;
        let frame = synthetic_frame(n, n);
        let good = write_visible_frame(&root, &frame).unwrap();
        let grid = GridFile::open(&good.grid_path).unwrap();
        assert!(coords_bit_identical(&grid.lat, &frame.lat));
        assert_eq!(
            grid.lat_descending(),
            Some(true),
            "correct frame is north-up"
        );

        let mut bad = frame.clone();
        for j in 0..n {
            for i in 0..n {
                bad.lat[j * n + i] = frame.lat[i * n + j];
                bad.lon[j * n + i] = frame.lon[i * n + j];
            }
        }
        let bad_written = write_visible_frame(&root, &bad).unwrap();
        assert!(bad_written.created_run, "a different mesh forks a new run");
        let bad_grid = GridFile::open(&bad_written.grid_path).unwrap();
        assert_ne!(
            bad_grid.lat_descending(),
            Some(true),
            "a transposed mesh must not read back as north-up"
        );
        assert!(
            !coords_bit_identical(&bad_grid.lat, &frame.lat),
            "a transposed mesh must differ from the correct mesh"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn from_rendered_masks_space_and_orders_rgb_bytes() {
        // Two pixels: a space pixel (alpha < 128 -> NaN on every plane) and a lit
        // earth pixel (alpha >= 128 -> the exact sRGB bytes in r, g, b order). A
        // swapped byte mapping or an inverted earth mask would be caught here
        // (M1-review MINOR-2: VisibleFrame::from_rendered was untested).
        let rendered = crate::gpu::RenderedFrame {
            width: 2,
            height: 1,
            rgba: vec![
                // pixel 0: space (alpha 0) — bytes present but must mask to NaN.
                5, 6, 7, 0, // pixel 1: lit earth (alpha 255) — distinct r/g/b bytes.
                10, 20, 30, 255,
            ],
        };
        let f = VisibleFrame::from_rendered(
            &rendered,
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            "sec".to_string(),
            SatellitePreset::GoesEast,
            2025,
            6,
            21,
            215,
        );
        assert_eq!((f.nx, f.ny), (2, 1));
        // Space pixel -> NaN on every plane (off-earth is transparent).
        assert!(f.rgb_r[0].is_nan() && f.rgb_g[0].is_nan() && f.rgb_b[0].is_nan());
        // Earth pixel -> exact bytes, r/g/b in order (a swap fails here).
        assert_eq!(f.rgb_r[1], 10.0);
        assert_eq!(f.rgb_g[1], 20.0);
        assert_eq!(f.rgb_b[1], 30.0);
        assert_eq!(f.band, 2);
        // Mesh carried through verbatim.
        assert_eq!(f.lat, vec![1.0, 2.0]);
        assert_eq!(f.lon, vec![3.0, 4.0]);
    }

    fn synthetic_ir_frame(nx: usize, ny: usize) -> IrFrame {
        let n = nx * ny;
        let mut lat = vec![0f32; n];
        let mut lon = vec![0f32; n];
        let mut bt = vec![f32::NAN; n];
        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                lat[idx] = 45.0 - j as f32 * 0.1; // row 0 = north
                lon[idx] = -100.0 + i as f32 * 0.1;
                // A cold anvil in one corner over warm ground elsewhere; a couple of
                // NaN (off-earth) pixels. Median stays ~290 K (true Kelvin).
                bt[idx] = if i == 0 && j == 0 {
                    f32::NAN
                } else if i < 2 && j < 2 {
                    212.0
                } else {
                    295.0 + (i as f32 * 0.1)
                };
            }
        }
        IrFrame::new(
            nx,
            ny,
            bt,
            lat,
            lon,
            "enderlin d03".to_string(),
            SatellitePreset::GoesEast,
            2025,
            6,
            21,
            215,
        )
    }

    #[test]
    fn write_and_read_back_a_band13_kelvin_frame() {
        let root = temp_root();
        let frame = synthetic_ir_frame(10, 8);
        let written = write_ir_frame(&root, &frame).expect("write IR");
        assert!(written.created_run);
        assert_eq!(written.model, "simsat");
        // Single-band run key `{sector}_c13_{day}` — NO `_rgb_` token (the Himawari-
        // style path BowEcho re-enhances; digest section 8).
        assert_eq!(written.run, "enderlin_d03_c13_20250621");
        assert!(
            !written.run.contains("_rgb_"),
            "IR run must not be an _rgb_ family"
        );
        assert!(written.frame_path.is_file());
        assert!(written.grid_path.is_file());

        let reader = HourReader::open(&written.frame_path).expect("frame");
        // ONE variable: `ahi_bt_c13`, units K, band-13 selector, sweep=y.
        let var = reader.variable("ahi_bt_c13").expect("ahi_bt_c13 var");
        assert_eq!(var.units, "K");
        let sat = var
            .selector
            .get("satellite")
            .expect("satellite selector object");
        assert_eq!(sat.get("band").and_then(|v| v.as_u64()), Some(13));
        assert_eq!(sat.get("model").and_then(|v| v.as_str()), Some("simsat"));
        assert_eq!(
            sat.get("product").and_then(|v| v.as_str()),
            Some("synthetic_ir")
        );
        assert_eq!(
            sat.get("projection")
                .and_then(|p| p.get("sweep_angle_axis"))
                .and_then(|v| v.as_str()),
            Some("y")
        );

        // The values come back as Kelvin (bit-exact for finite entries; NaN preserved
        // as the off-earth marker). This asserts the store holds true Kelvin — not a
        // baked colour — so `ir_enhancement_anchors(13, ...)` applies downstream.
        let vals = reader.read_full_2d("ahi_bt_c13").expect("read bt");
        assert_eq!(vals.len(), 10 * 8);
        for (k, (&got, &want)) in vals.iter().zip(frame.bt.iter()).enumerate() {
            if want.is_nan() {
                assert!(got.is_nan(), "IR[{k}] should stay NaN off-earth");
            } else {
                assert_eq!(got, want, "IR[{k}] Kelvin readback mismatch");
            }
        }
        // The finite median is well below the 320 K legacy-stretch threshold, so the
        // frame classifies as true Kelvin (not the percentile pseudo-BT path).
        let mut finite: Vec<f32> = vals.iter().copied().filter(|v| v.is_finite()).collect();
        finite.sort_by(f32::total_cmp);
        let median = finite[finite.len() / 2];
        assert!(
            median < 320.0,
            "median BT {median} would trip the legacy stretch"
        );
        assert!(median > 250.0, "median BT {median} implausibly cold");

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn write_and_read_back_a_wv_band8_kelvin_frame() {
        // The 6.2 um water-vapor band writes the SAME single-band Kelvin contract as band
        // 13, only with band 8: run key `{sector}_c08_{day}`, variable `ahi_bt_c08`, band
        // 8 selector. BowEcho then re-enhances it through `band_anchors(8)` (the WV
        // moisture palette). This mirrors the band-13 write with a WV-cold median.
        let root = temp_root();
        let base = synthetic_ir_frame(10, 8);
        // Rebuild at band 8 with a WV-cold plane (medians ~230 K).
        let n = base.nx * base.ny;
        let mut bt = vec![f32::NAN; n];
        for (j, chunk) in bt.chunks_mut(base.nx).enumerate() {
            for (i, v) in chunk.iter_mut().enumerate() {
                *v = if i == 0 && j == 0 {
                    f32::NAN
                } else if i < 2 && j < 2 {
                    205.0 // cold moist upper-level
                } else {
                    235.0 + i as f32 * 0.1
                };
            }
        }
        let frame = IrFrame::new_band(
            base.nx,
            base.ny,
            bt,
            base.lat.clone(),
            base.lon.clone(),
            "enderlin d03".to_string(),
            SatellitePreset::GoesEast,
            8,
            2025,
            6,
            21,
            215,
        );
        assert_eq!(frame.band, 8);
        assert_eq!(frame.variable_name(), "ahi_bt_c08");
        let written = write_ir_frame(&root, &frame).expect("write WV");
        assert!(written.created_run);
        assert_eq!(written.run, "enderlin_d03_c08_20250621");
        assert!(
            !written.run.contains("_rgb_") && written.run.contains("_c08_"),
            "WV run must be a single-band _c08_ run"
        );
        let reader = HourReader::open(&written.frame_path).expect("frame");
        let var = reader.variable("ahi_bt_c08").expect("ahi_bt_c08 var");
        assert_eq!(var.units, "K");
        let sat = var.selector.get("satellite").expect("satellite selector");
        assert_eq!(sat.get("band").and_then(|v| v.as_u64()), Some(8));
        assert_eq!(sat.get("model").and_then(|v| v.as_str()), Some("simsat"));
        // The finite median is cold WV but still below the 320 K legacy-stretch threshold.
        let vals = reader.read_full_2d("ahi_bt_c08").expect("read bt");
        let mut finite: Vec<f32> = vals.iter().copied().filter(|v| v.is_finite()).collect();
        finite.sort_by(f32::total_cmp);
        let median = finite[finite.len() / 2];
        // Below the 320 K legacy-stretch threshold (true Kelvin) and cold-WV (~235 K here).
        assert!(
            median < 320.0,
            "WV median {median} would trip the legacy stretch"
        );
        assert!(
            (200.0..260.0).contains(&median),
            "WV median {median} implausible"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn ir_and_visible_frames_reuse_the_grid_but_are_distinct_runs() {
        // On the SAME grid the IR run (`_c13_`) and the visible run (`_rgb_`) are
        // separate run dirs (different run_base) but each reuses its own grid.rwg.
        let root = temp_root();
        let vis = synthetic_frame(10, 8);
        let ir = synthetic_ir_frame(10, 8);
        // Give the IR mesh the SAME lat/lon as the visible frame so the grids match.
        let mut ir = ir;
        ir.lat = vis.lat.clone();
        ir.lon = vis.lon.clone();
        let w_vis = write_visible_frame(&root, &vis).unwrap();
        let w_ir = write_ir_frame(&root, &ir).unwrap();
        assert_ne!(w_vis.run, w_ir.run, "IR and visible are distinct runs");
        assert!(w_vis.run.contains("_rgb_"));
        assert!(w_ir.run.contains("_c13_") && !w_ir.run.contains("_rgb_"));
        // Writing the IR frame again at a later time reuses the run (same grid).
        let mut ir2 = ir.clone();
        ir2.hhmm = 230;
        let w_ir2 = write_ir_frame(&root, &ir2).unwrap();
        assert!(!w_ir2.created_run);
        assert_eq!(w_ir2.run, w_ir.run);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_multi_frame_run_holds_every_frame_retrievable() {
        // The M7 loop-render path writes each rendered timestep into ONE run (the
        // bit-identical-grid reuse). Prove that N frames at distinct times land as N
        // distinct `t{HHMM}.rws` files in a SINGLE run dir and that each reads back its
        // OWN distinct pixels — i.e. a proper multi-frame run BowEcho's player loops.
        let root = temp_root();
        let base = synthetic_frame(10, 8);
        // The Enderlin cadence: 01:30, 02:00, 02:30, 05:00, 05:30, 06:00.
        let hhmms = [130u16, 200, 230, 500, 530, 600];
        let mut run_name: Option<String> = None;
        for (k, &hhmm) in hhmms.iter().enumerate() {
            let mut f = base.clone();
            f.hhmm = hhmm;
            // A distinct constant red plane per frame, so a cross-frame mixup shows up.
            for r in &mut f.rgb_r {
                *r = (k * 20) as f32;
            }
            let w = write_visible_frame(&root, &f).unwrap();
            match &run_name {
                None => {
                    assert!(w.created_run, "the first frame creates the run");
                    run_name = Some(w.run.clone());
                }
                Some(name) => {
                    assert!(!w.created_run, "later frames reuse the one run");
                    assert_eq!(&w.run, name, "all frames share one run");
                }
            }
        }
        let run_name = run_name.unwrap();
        let run_dir = root.join(SIMSAT_MODEL).join(&run_name);
        assert!(run_dir.join("run.json").is_file(), "run manifest present");
        assert!(run_dir.join("grid.rwg").is_file(), "one shared grid");

        // Every frame file exists and reads back its OWN red value (no aliasing across
        // the N frames of the run).
        for (k, &hhmm) in hhmms.iter().enumerate() {
            let path = run_dir.join(frame_file_name(hhmm));
            assert!(path.is_file(), "frame {hhmm} missing from the run");
            let reader = HourReader::open(&path).expect("open frame");
            let r = reader.read_full_2d(COMPOSITE_R_VAR).expect("read rgb_r");
            assert_eq!(r.len(), 10 * 8);
            assert!(
                r.iter().all(|&v| v == (k * 20) as f32),
                "frame {hhmm} read back the wrong plane (cross-frame aliasing)"
            );
        }
        // Exactly N frame files landed in the run dir (no extra/missing frames). The
        // per-frame file name is `frame_file_name(hhmm)` (a `t{HHMM}.rws`); count the
        // files matching that exact set so the check does not depend on the extension.
        let expected: std::collections::HashSet<String> =
            hhmms.iter().map(|&h| frame_file_name(h)).collect();
        let frame_files = std::fs::read_dir(&run_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| expected.contains(&e.file_name().to_string_lossy().to_string()))
            .count();
        assert_eq!(frame_files, hhmms.len(), "one frame file per timestep");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn list_run_frames_and_rgb8_readback_round_trip() {
        // The GIF-export readback half: write three frames OUT of time order (plus a
        // stray non-frame file), then (1) list_run_frames returns dims + the frames
        // sorted by hhmm with the stray file ignored, and (2) read_visible_frame_rgb
        // returns the exact interleaved bytes with NaN (space) pixels black.
        let root = temp_root();
        let base = synthetic_frame(10, 6);
        let mut run_dir = PathBuf::new();
        for (hhmm, red) in [(300u16, 180f32), (145, 60.0), (215, 120.0)] {
            let mut f = base.clone();
            f.hhmm = hhmm;
            for r in &mut f.rgb_r {
                *r = red;
            }
            // Pixel 0 is space (NaN on every plane) — must read back black.
            f.rgb_r[0] = f32::NAN;
            f.rgb_g[0] = f32::NAN;
            f.rgb_b[0] = f32::NAN;
            run_dir = write_visible_frame(&root, &f).unwrap().run_dir;
        }
        // A stray file that must NOT be indexed as a frame.
        std::fs::write(run_dir.join("t9999.tmp"), b"stray").unwrap();

        let index = list_run_frames(&run_dir).expect("index");
        assert_eq!((index.nx, index.ny), (10, 6));
        let hhmms: Vec<u16> = index.frames.iter().map(|f| f.hhmm).collect();
        assert_eq!(hhmms, vec![145, 215, 300], "frames sorted by valid time");

        for (frame, want_red) in index.frames.iter().zip([60u8, 120, 180]) {
            let rgb = read_visible_frame_rgb(&frame.path, index.nx, index.ny).expect("rgb");
            assert_eq!(rgb.len(), 10 * 6 * 3);
            // Pixel 0 (NaN space) is black on all three channels.
            assert_eq!(&rgb[0..3], &[0, 0, 0], "space pixel blacked out");
            // Pixel 1 carries the frame's exact bytes (r = the per-frame red,
            // g = 128, b = 64 from the synthetic frame).
            assert_eq!(
                &rgb[3..6],
                &[want_red, 128, 64],
                "t{:04} exact RGB8 readback",
                frame.hhmm
            );
        }

        // parse_frame_file_name accepts only canonical names.
        assert_eq!(parse_frame_file_name("t0145.rws"), Some(145));
        assert_eq!(parse_frame_file_name("t145.rws"), None);
        assert_eq!(parse_frame_file_name("t9999.tmp"), None);
        assert_eq!(parse_frame_file_name("grid.rwg"), None);
        std::fs::remove_dir_all(&root).ok();
    }
}
