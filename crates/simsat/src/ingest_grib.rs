//! Streaming NOAA HRRR / RRFS GRIB2 -> `.ssb` brick ingest (parallel to [`crate::ingest`]).
//!
//! This module writes the same SSB v5 u8 channels + f16-Celsius temperature +
//! 2-D planes + `run.json` manifest that the wrfout path writes, so an operational HRRR
//! brick is indistinguishable from a WRF brick to every downstream consumer
//! (visible/IR/WV/GeoColor/Sandwich/derived, geo + top-down, studio, Python).
//!
//! Streaming discipline (the GRIB analog of "never getvar 3-D in bulk"):
//! `grib_core::grib2::Grib2File::open` reads the WHOLE file (a RRFS natlev file
//! is 9.3 GB), so this module never calls it on a file. GRIB2 messages are
//! self-contained; we SEEK-INDEX the message boundaries from each message's
//! 16-byte Section-0 header, then parse + decode ONE message slice at a time via
//! `Grib2File::from_bytes` and fold each decoded level directly into the native
//! accumulation buffers (one native 3-D f32 field resident at a time). The
//! brick-resolution channels are produced by a TWO-PASS per-column resample
//! (pass 1: quantization stats; pass 2: encode straight to u8), so no
//! brick-resolution f32 channel buffer is ever materialized, and the running
//! total extinction for `tau_up` is held as f16 bits. Probe evidence + the
//! field inventory for both models: `notes/grib-probe/probe_{hrrr,rrfs}.log` and
//! `notes/grib-ingest-notes.md`.
//!
//! Physics mapping (per-species, the SAME `optics.rs` constants as WRF):
//! CLMR -> ext_liquid; CIMIXR (HRRR, 0/1/82) or ICMR (RRFS, 0/1/23) -> ext_ice;
//! RWMR + GRLE + SNMR -> total ext_precip (each at its OWN beta, the SSB v3 rule).
//! The v4+ snow-only auxiliary subset is conservatively unavailable for the initial
//! GRIB path (`ext_snow = 0`). HRRR's native `cc` field (0/6/32) supplies trusted
//! fractional coverage when it is a complete 1..N hybrid-level volume on the same
//! vertical grid as the hydrometeors. Missing, partial, malformed, or differently
//! scaled coverage retains the exact legacy fallback (`cloud_fraction = 255`,
//! provenance false);
//! SPFH -> the qvapor channel as a MIXING RATIO (`w = q / (1 - q)`); TMP is
//! sensible temperature directly (no theta conversion); per-level geopotential
//! height (HGT @ hybrid) is the native z coordinate directly (no destagger);
//! `rho = p / (R_d T)` from PRES + TMP per level. 2-D: TMP@surface -> TSK,
//! HGT@surface -> HGT, LAND -> LANDMASK, SNOD -> SNOWH, UGRD/VGRD@10 m ->
//! U10/V10, VGTYP -> IVGTYP (best-effort).
//!
//! Georeference: the HRRR Lambert grid (template 3.30) maps onto the EXISTING
//! `frame.rs` projection with one documented conversion — GRIB declares a sphere
//! of R_g = 6,371,229 m (shape-of-earth 6) while `frame.rs` is locked to WRF's
//! R_o = 6,370,000 m (owner decision 5). Lambert plane coordinates scale
//! LINEARLY with the sphere radius, so scaling the grid spacing by `R_o / R_g`
//! reproduces every grid point's lat/lon EXACTLY on our sphere
//! (spherical-to-spherical, no datum shift); the fixture test ratchets it.
//!
//! Valid time comes from the GRIB Section-1 reference time + the Section-4
//! forecast horizon — a real header time (`time_is_fallback` semantics hold).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::Instant;

use chrono::{Datelike, NaiveDateTime, Timelike};
use grib_core::grib2::{Grib2File, GridDefinition, grid_latlon, unpack_message};

use crate::bricks::{
    self, CELSIUS_OFFSET_K, ChannelQuant, LogQuant, ManifestProjection, ManifestTimestep,
    QUANT_DYNAMIC_RANGE, RunManifest, VolumeBrick, f16_bits_to_f32, f32_to_f16_bits,
};
use crate::frame::{
    FrameError, MAP_PROJ_ROTATED_LATLON, ROTATED_LATLON_M_PER_DEG, WrfProjectionParams,
};
use crate::ingest::{
    Extrap, GridGeometry, IngestConfig, IngestReport, integrate_tau_up_column, resample_column,
    resample_column_conservative, resample_column_fraction_max_overlap, source_identity,
};
use crate::optics::{self, HydrometeorClass};
use crate::platform;

// ── errors ─────────────────────────────────────────────────────────────────────

/// GRIB-ingest errors (this module's own enum so [`crate::ingest::IngestError`]
/// stays untouched — parallel-agent file ownership).
#[derive(Debug)]
pub enum GribIngestError {
    Io(std::io::Error),
    /// grib-core parse/unpack failure, or a malformed message stream.
    Grib(String),
    Frame(FrameError),
    Brick(bricks::BrickError),
    /// A field the brick requires is absent from the file.
    MissingField(String),
    /// Unexpected geometry / level structure / unsupported grid.
    Shape(String),
}

impl std::fmt::Display for GribIngestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "grib i/o error: {e}"),
            Self::Grib(s) => write!(f, "grib decode error: {s}"),
            Self::Frame(e) => write!(f, "projection error: {e}"),
            Self::Brick(e) => write!(f, "brick error: {e}"),
            Self::MissingField(s) => write!(f, "required field missing: {s}"),
            Self::Shape(s) => write!(f, "unexpected shape: {s}"),
        }
    }
}
impl std::error::Error for GribIngestError {}
impl From<std::io::Error> for GribIngestError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<FrameError> for GribIngestError {
    fn from(e: FrameError) -> Self {
        Self::Frame(e)
    }
}
impl From<bricks::BrickError> for GribIngestError {
    fn from(e: bricks::BrickError) -> Self {
        Self::Brick(e)
    }
}

// ── input classification (pure helpers; the api.rs / studio seams call these) ──

/// Whether a path LOOKS like a GRIB2 file by extension (`.grib2`, `.grb2`,
/// `.grib`, `.grb`, case-insensitive).
pub fn is_grib_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            e.eq_ignore_ascii_case("grib2")
                || e.eq_ignore_ascii_case("grb2")
                || e.eq_ignore_ascii_case("grib")
                || e.eq_ignore_ascii_case("grb")
        })
        .unwrap_or(false)
}

/// Whether the file STARTS with the `GRIB` magic (a cheap 4-byte sniff; false on
/// any i/o error). NCEP operational files carry the magic at byte 0.
pub fn sniff_grib_magic(path: &Path) -> bool {
    let mut buf = [0u8; 4];
    match File::open(path) {
        Ok(mut f) => f.read_exact(&mut buf).is_ok() && &buf == b"GRIB",
        Err(_) => false,
    }
}

/// The input-classification predicate: extension OR magic. (The api.rs
/// `resolve_source` seam and the studio open flow are owned by parallel agents;
/// the exact one-line diffs that call this live in `notes/grib-ingest-notes.md`.)
pub fn is_grib_input(path: &Path) -> bool {
    is_grib_path(path) || sniff_grib_magic(path)
}

// ── message index (seek-based; the whole file is never resident) ───────────────

/// One GRIB2 message's byte range within the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MessageLocation {
    pub offset: u64,
    pub length: u64,
}

/// Seek-index every GRIB2 message: read each message's 16-byte Section-0 header
/// (`GRIB` magic + total length at bytes 8..16, big-endian) and hop to the next.
/// Stops cleanly at trailing garbage / a truncated tail (the caller decides
/// whether zero messages is an error). `total_len` is the stream length.
pub fn index_grib_messages<R: Read + Seek>(
    reader: &mut R,
    total_len: u64,
) -> std::io::Result<Vec<MessageLocation>> {
    let mut out = Vec::new();
    let mut pos = 0u64;
    let mut hdr = [0u8; 16];
    while pos + 16 <= total_len {
        reader.seek(SeekFrom::Start(pos))?;
        reader.read_exact(&mut hdr)?;
        if &hdr[0..4] != b"GRIB" {
            break;
        }
        let length = u64::from_be_bytes(hdr[8..16].try_into().expect("8-byte slice"));
        if length < 16 || pos + length > total_len {
            break;
        }
        out.push(MessageLocation {
            offset: pos,
            length,
        });
        pos += length;
    }
    Ok(out)
}

// ── field codes (GRIB2 discipline / parameter category / parameter number) ─────

/// A GRIB2 parameter identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FieldCode {
    pub discipline: u8,
    pub category: u8,
    pub number: u8,
}

const fn fc(discipline: u8, category: u8, number: u8) -> FieldCode {
    FieldCode {
        discipline,
        category,
        number,
    }
}

/// Sensible temperature (K).
pub const CODE_TMP: FieldCode = fc(0, 0, 0);
/// Specific humidity (kg/kg moist air).
pub const CODE_SPFH: FieldCode = fc(0, 1, 0);
/// Pressure (Pa).
pub const CODE_PRES: FieldCode = fc(0, 3, 0);
/// Geopotential height (m MSL).
pub const CODE_HGT: FieldCode = fc(0, 3, 5);
/// Cloud (liquid) water mixing ratio.
pub const CODE_CLMR: FieldCode = fc(0, 1, 22);
/// Cloud-ice mixing ratio, NCEP local code — what HRRR writes (CIMIXR).
pub const CODE_CIMIXR: FieldCode = fc(0, 1, 82);
/// Ice water mixing ratio, WMO code — what RRFS writes (ICMR).
pub const CODE_ICMR: FieldCode = fc(0, 1, 23);
/// Rain mixing ratio.
pub const CODE_RWMR: FieldCode = fc(0, 1, 24);
/// Snow mixing ratio.
pub const CODE_SNMR: FieldCode = fc(0, 1, 25);
/// Graupel mixing ratio.
pub const CODE_GRLE: FieldCode = fc(0, 1, 32);
/// Fraction of cloud cover (`cc`, numeric fraction 0..1).
pub const CODE_CLOUD_FRACTION: FieldCode = fc(0, 6, 32);
/// Snow depth (m).
pub const CODE_SNOD: FieldCode = fc(0, 1, 11);
/// U wind component (m/s).
pub const CODE_UGRD: FieldCode = fc(0, 2, 2);
/// V wind component (m/s).
pub const CODE_VGRD: FieldCode = fc(0, 2, 3);
/// Land cover (0 = sea, 1 = land).
pub const CODE_LAND: FieldCode = fc(2, 0, 0);
/// Vegetation type (NCEP local).
pub const CODE_VGTYP: FieldCode = fc(2, 0, 198);

/// GRIB2 fixed-surface types (code table 4.5).
pub const LEVEL_SURFACE: u8 = 1;
pub const LEVEL_HEIGHT_AGL: u8 = 103;
pub const LEVEL_HYBRID: u8 = 105;

/// The one scan mode this ingest accepts: +i (west->east), +j (south->north),
/// i-consecutive — 0x40, which is what every probed NCEP HRRR/RRFS product uses
/// and EXACTLY the wrfout row order, so decoded planes are already in brick
/// layout. Anything else is refused with the value in the message (rather than
/// silently mis-orienting the grid).
pub const REQUIRED_SCAN_MODE: u8 = 0x40;

/// GRIB2 shape-of-earth (code table 3.2) -> sphere radius in metres. Only the
/// spherical codes appear in NCEP regional output; an ellipsoidal shape would
/// need real datum work and is refused.
pub fn earth_radius_m(shape_of_earth: u8) -> Option<f64> {
    match shape_of_earth {
        0 => Some(6_367_470.0),
        6 => Some(6_371_229.0),
        _ => None,
    }
}

// ── catalog (one metadata pass; no field decode) ────────────────────────────────

/// One field instance in the file: which message (and which field within a
/// multi-field envelope), what parameter, at what level.
#[derive(Debug, Clone, Copy)]
struct CatalogEntry {
    loc: usize,
    sub: usize,
    code: FieldCode,
    level_type: u8,
    level_value: f64,
}

/// The parsed skeleton of a GRIB2 file: message locations + per-field entries +
/// the (shared) grid definition + the reference/forecast time.
struct GribCatalog {
    locs: Vec<MessageLocation>,
    entries: Vec<CatalogEntry>,
    grid: GridDefinition,
    reference_time: NaiveDateTime,
    forecast_time: u32,
    time_range_unit: u8,
}

fn build_catalog(path: &Path) -> Result<GribCatalog, GribIngestError> {
    let mut file = File::open(path)?;
    let total_len = file.metadata()?.len();
    let locs = index_grib_messages(&mut file, total_len)?;
    if locs.is_empty() {
        return Err(GribIngestError::Grib(format!(
            "no GRIB2 messages found in {}",
            path.display()
        )));
    }
    let mut entries = Vec::with_capacity(locs.len());
    let mut grid: Option<GridDefinition> = None;
    let mut reference_time = None;
    let mut forecast_time = 0u32;
    let mut time_range_unit = 1u8;
    let mut buf: Vec<u8> = Vec::new();
    for (li, loc) in locs.iter().enumerate() {
        read_message_bytes(&mut file, *loc, &mut buf)?;
        let parsed = Grib2File::from_bytes(&buf)
            .map_err(|e| GribIngestError::Grib(format!("message #{li}: {e}")))?;
        for (si, msg) in parsed.messages.iter().enumerate() {
            let g = &msg.grid;
            match &grid {
                None => {
                    if g.scan_mode != REQUIRED_SCAN_MODE {
                        return Err(GribIngestError::Shape(format!(
                            "unsupported scan mode 0x{:02x} (this ingest requires 0x40: \
                             +i, +j from the south-west corner — the NCEP standard)",
                            g.scan_mode
                        )));
                    }
                    grid = Some(g.clone());
                    reference_time = Some(msg.reference_time);
                    forecast_time = msg.product.forecast_time;
                    time_range_unit = msg.product.time_range_unit;
                }
                Some(first) => {
                    if g.nx != first.nx || g.ny != first.ny || g.scan_mode != first.scan_mode {
                        return Err(GribIngestError::Shape(format!(
                            "message #{li}.{si} grid {}x{} scan 0x{:02x} differs from the \
                             file grid {}x{} scan 0x{:02x}",
                            g.nx, g.ny, g.scan_mode, first.nx, first.ny, first.scan_mode
                        )));
                    }
                }
            }
            entries.push(CatalogEntry {
                loc: li,
                sub: si,
                code: fc(
                    msg.discipline,
                    msg.product.parameter_category,
                    msg.product.parameter_number,
                ),
                level_type: msg.product.level_type,
                level_value: msg.product.level_value,
            });
        }
    }
    Ok(GribCatalog {
        locs,
        entries,
        grid: grid.expect("locs is non-empty"),
        reference_time: reference_time.expect("locs is non-empty"),
        forecast_time,
        time_range_unit,
    })
}

fn read_message_bytes(
    file: &mut File,
    loc: MessageLocation,
    buf: &mut Vec<u8>,
) -> std::io::Result<()> {
    buf.clear();
    buf.resize(loc.length as usize, 0);
    file.seek(SeekFrom::Start(loc.offset))?;
    file.read_exact(buf)
}

/// Decode one catalog entry to f64 values in brick row order (scan 0x40 raw ==
/// south-to-north row-major, verified at catalog build). Bitmapped-out cells are
/// NaN (grib-core's contract); the caller applies the field's NaN policy.
fn decode_entry(
    file: &mut File,
    catalog: &GribCatalog,
    entry: &CatalogEntry,
    buf: &mut Vec<u8>,
) -> Result<Vec<f64>, GribIngestError> {
    read_message_bytes(file, catalog.locs[entry.loc], buf)?;
    let parsed = Grib2File::from_bytes(buf)
        .map_err(|e| GribIngestError::Grib(format!("message #{}: {e}", entry.loc)))?;
    let msg = parsed.messages.get(entry.sub).ok_or_else(|| {
        GribIngestError::Grib(format!(
            "message #{} lost sub-field {} on re-parse",
            entry.loc, entry.sub
        ))
    })?;
    let values = unpack_message(msg)
        .map_err(|e| GribIngestError::Grib(format!("message #{} unpack: {e}", entry.loc)))?;
    let expected = (catalog.grid.nx as usize) * (catalog.grid.ny as usize);
    if values.len() != expected {
        return Err(GribIngestError::Shape(format!(
            "message #{} decoded {} values, expected {expected}",
            entry.loc,
            values.len()
        )));
    }
    Ok(values)
}

// ── level structure ────────────────────────────────────────────────────────────

/// Hybrid-level entries of `code`, keyed by integer level (GRIB level 1 = lowest).
fn hybrid_level_entries(catalog: &GribCatalog, code: FieldCode) -> BTreeMap<u32, CatalogEntry> {
    let mut map = BTreeMap::new();
    for e in &catalog.entries {
        if e.code == code && e.level_type == LEVEL_HYBRID {
            let lvl = e.level_value.round() as u32;
            map.entry(lvl).or_insert(*e);
        }
    }
    map
}

/// Validate that a field's hybrid levels are EXACTLY 1..=n contiguous and return
/// `n`. A gap means a truncated/partial file — better refused than silently
/// ingested with a hole in the column.
pub fn validate_complete_levels(levels: &[u32], field: &str) -> Result<usize, GribIngestError> {
    if levels.is_empty() {
        return Err(GribIngestError::MissingField(format!(
            "{field} has no hybrid-level messages"
        )));
    }
    let n = levels.len();
    for (i, &lvl) in levels.iter().enumerate() {
        let want = (i + 1) as u32;
        if lvl != want {
            return Err(GribIngestError::Shape(format!(
                "{field} hybrid levels are not contiguous: expected level {want}, found {lvl} \
                 (truncated or partial file?)"
            )));
        }
    }
    Ok(n)
}

fn require_hybrid_volume_entries(
    catalog: &GribCatalog,
    code: FieldCode,
    field: &str,
    nz: usize,
) -> Result<Vec<CatalogEntry>, GribIngestError> {
    let map = hybrid_level_entries(catalog, code);
    let levels: Vec<u32> = map.keys().copied().collect();
    let n = validate_complete_levels(&levels, field)?;
    if n != nz {
        return Err(GribIngestError::Shape(format!(
            "{field} has {n} hybrid levels, expected {nz}"
        )));
    }
    Ok(map.into_values().collect())
}

// ── value policies ─────────────────────────────────────────────────────────────

/// Select HRRR's native cloud-fraction messages without weakening the required
/// field rules used by the rest of the ingest. Coverage is optional, so every
/// incompatibility is reported to the caller as a reason to use the all-255
/// fallback rather than as a fatal ingest error.
///
/// The returned entries are explicitly sorted level 1 -> N (bottom -> top in
/// HRRR native output), independent of message order in the GRIB envelope.
fn optional_cloud_fraction_entries(
    entries: &[CatalogEntry],
    nz: usize,
) -> Result<Option<Vec<CatalogEntry>>, String> {
    let candidates: Vec<CatalogEntry> = entries
        .iter()
        .filter(|e| e.code == CODE_CLOUD_FRACTION && e.level_type == LEVEL_HYBRID)
        .copied()
        .collect();
    if candidates.is_empty() {
        return Ok(None);
    }

    let mut by_level = BTreeMap::new();
    for entry in candidates {
        let rounded = entry.level_value.round();
        if !entry.level_value.is_finite()
            || rounded < 1.0
            || rounded > u32::MAX as f64
            || (entry.level_value - rounded).abs() > 1.0e-6
        {
            return Err(format!(
                "cc (0/6/32) has invalid hybrid level {}",
                entry.level_value
            ));
        }
        let level = rounded as u32;
        if by_level.insert(level, entry).is_some() {
            return Err(format!("cc (0/6/32) has duplicate hybrid level {level}"));
        }
    }

    let levels: Vec<u32> = by_level.keys().copied().collect();
    validate_complete_levels(&levels, "cc (cloud fraction)").map_err(|e| e.to_string())?;
    if by_level.len() != nz {
        return Err(format!(
            "cc (0/6/32) has {} hybrid levels, expected {nz} on the hydrometeor grid",
            by_level.len()
        ));
    }
    Ok(Some(by_level.into_values().collect()))
}

/// GRIB code 0/6/32 is defined as a numeric 0..1 fraction. Permit only tiny
/// packing roundoff outside that interval, then clamp it. A percent-like value
/// (for example 50) is not guessed or divided by 100: without units metadata
/// proving that contract, treating it as native coverage would silently make a
/// malformed field look trustworthy.
const CLOUD_FRACTION_RANGE_TOLERANCE: f32 = 1.0e-4;

fn normalize_native_cloud_fraction(values: &mut [f32]) -> Result<(), String> {
    for (index, value) in values.iter_mut().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "cc (0/6/32) contains a non-finite value at native cell {index}"
            ));
        }
        if *value < -CLOUD_FRACTION_RANGE_TOLERANCE || *value > 1.0 + CLOUD_FRACTION_RANGE_TOLERANCE
        {
            return Err(format!(
                "cc (0/6/32) value {} at native cell {index} violates its 0..1 fraction contract; refusing to guess percent scaling",
                *value
            ));
        }
        *value = value.clamp(0.0, 1.0);
    }
    Ok(())
}

/// Specific humidity (kg/kg moist air) -> water-vapor MIXING RATIO (kg/kg dry
/// air), the brick's qvapor convention (WRF QVAPOR): `w = q / (1 - q)`.
/// Non-finite / non-positive input -> 0 (clear air).
pub fn spfh_to_mixing_ratio(q: f64) -> f64 {
    if !q.is_finite() || q <= 0.0 {
        return 0.0;
    }
    let q = q.min(0.5); // physically absurd ceiling guard; keeps the division tame
    q / (1.0 - q)
}

/// In-place NaN fill for a CONTINUOUS native volume (temperature/pressure/height):
/// each NaN cell takes the nearest finite value in its own column (below first,
/// then above). Returns the number of filled cells; an entirely-NaN column is an
/// error (the RRFS rotated grid masks cells outside the model domain — a proper
/// crop stays inside valid data, so a full-NaN column means a bad crop).
pub fn fill_column_nan(
    values: &mut [f32],
    nx: usize,
    ny: usize,
    nz: usize,
) -> Result<u64, GribIngestError> {
    let plane = nx * ny;
    let mut filled = 0u64;
    for ci in 0..plane {
        // Fast path: any NaN in this column?
        let mut has_nan = false;
        let mut has_finite = false;
        for k in 0..nz {
            let v = values[k * plane + ci];
            if v.is_nan() {
                has_nan = true;
            } else {
                has_finite = true;
            }
        }
        if !has_nan {
            continue;
        }
        if !has_finite {
            return Err(GribIngestError::Shape(format!(
                "column {ci} is entirely NaN (masked outside the model domain?); \
                 crop the ingest to the valid region"
            )));
        }
        // Forward fill from below, then backward fill from above.
        let mut last = f32::NAN;
        for k in 0..nz {
            let idx = k * plane + ci;
            if values[idx].is_nan() {
                if !last.is_nan() {
                    values[idx] = last;
                    filled += 1;
                }
            } else {
                last = values[idx];
            }
        }
        let mut last = f32::NAN;
        for k in (0..nz).rev() {
            let idx = k * plane + ci;
            if values[idx].is_nan() {
                if !last.is_nan() {
                    values[idx] = last;
                    filled += 1;
                }
            } else {
                last = values[idx];
            }
        }
    }
    Ok(filled)
}

// ── grid -> projection ─────────────────────────────────────────────────────────

fn normalize_lon(lon_deg: f64) -> f64 {
    let mut lon = lon_deg % 360.0;
    if lon > 180.0 {
        lon -= 360.0;
    } else if lon <= -180.0 {
        lon += 360.0;
    }
    lon
}

/// Route a GRIB grid definition to the matching projection-params builder:
/// template 3.30 Lambert (HRRR) or template 3.1 rotated lat-lon (RRFS NA).
pub fn params_from_grid(
    grid: &GridDefinition,
    cen_lat_deg: f64,
    cen_lon_deg: f64,
) -> Result<WrfProjectionParams, GribIngestError> {
    match grid.template {
        30 => lambert_params_from_grid(grid, cen_lat_deg, cen_lon_deg),
        1 => rotated_params_from_grid(grid, cen_lat_deg, cen_lon_deg),
        other => Err(GribIngestError::Shape(format!(
            "grid template 3.{other} is not supported (Lambert 3.30 and rotated \
             lat-lon 3.1 are)"
        ))),
    }
}

/// Build the rotated lat-lon projection params from a GRIB template-3.1 grid
/// (the RRFS NA native grid): `map_proj` = [`MAP_PROJ_ROTATED_LATLON`] with the
/// rotated NORTH pole riding in the reused `truelat1`/`truelat2` fields (the
/// frame.rs convention) and the plane spacing in METRES (rotated-degree
/// increments x [`ROTATED_LATLON_M_PER_DEG`]). The probed RRFS files store
/// `dx = dy = 0` in the grid definition — the increments derive from the corner
/// span exactly as grib-core's own grid math takes them. Angular spacing is
/// radius-independent, so no earth-radius rescale applies here.
pub fn rotated_params_from_grid(
    grid: &GridDefinition,
    cen_lat_deg: f64,
    cen_lon_deg: f64,
) -> Result<WrfProjectionParams, GribIngestError> {
    if grid.template != 1 {
        return Err(GribIngestError::Shape(format!(
            "rotated_params_from_grid on template 3.{}",
            grid.template
        )));
    }
    if grid.rotation_angle != 0.0 {
        return Err(GribIngestError::Shape(format!(
            "rotated lat-lon grid with a nonzero rotation angle ({}) is not \
             supported (RRFS uses 0)",
            grid.rotation_angle
        )));
    }
    if grid.nx < 2 || grid.ny < 2 {
        return Err(GribIngestError::Shape(format!(
            "degenerate rotated grid {}x{}",
            grid.nx, grid.ny
        )));
    }
    // The rotated NORTH pole is the antipode of the GRIB south pole.
    let pole_lat = -grid.south_pole_lat;
    let pole_lon = normalize_lon(grid.south_pole_lon - 180.0);
    // Rotated-degree increments from the corner span (lon2 unwrapped past 360,
    // exactly as grib-core's `latlon_grid` computes the step).
    let dlat = (grid.lat2 - grid.lat1) / ((grid.ny - 1) as f64);
    let lon2 = if grid.lon2 < grid.lon1 {
        grid.lon2 + 360.0
    } else {
        grid.lon2
    };
    let dlon = (lon2 - grid.lon1) / ((grid.nx - 1) as f64);
    if dlat <= 0.0 || dlon <= 0.0 || !dlat.is_finite() || !dlon.is_finite() {
        return Err(GribIngestError::Shape(format!(
            "rotated grid increments must be positive (dlat={dlat}, dlon={dlon})"
        )));
    }
    Ok(WrfProjectionParams {
        map_proj: MAP_PROJ_ROTATED_LATLON,
        truelat1_deg: pole_lat,
        truelat2_deg: pole_lon,
        stand_lon_deg: 0.0,
        cen_lat_deg,
        cen_lon_deg,
        dx_m: dlon * ROTATED_LATLON_M_PER_DEG,
        dy_m: dlat * ROTATED_LATLON_M_PER_DEG,
    })
}

/// Build WRF-style Lambert projection params from a GRIB template-3.30 grid.
///
/// The one conversion: GRIB grid spacing is metres on ITS sphere (`R_g`, from
/// shape-of-earth); `frame.rs` projects on WRF's `R_o = 6.37e6`. Lambert plane
/// coordinates scale linearly with the radius, so `dx_ours = dx_grib * R_o/R_g`
/// makes our projection reproduce every GRIB grid point exactly (module doc).
/// `cen_lat`/`cen_lon` are the grid-centre coordinates (the caller reads them
/// off the synthesized lat/lon planes).
pub fn lambert_params_from_grid(
    grid: &GridDefinition,
    cen_lat_deg: f64,
    cen_lon_deg: f64,
) -> Result<WrfProjectionParams, GribIngestError> {
    if grid.template != 30 {
        return Err(GribIngestError::Shape(format!(
            "lambert_params_from_grid on template 3.{}",
            grid.template
        )));
    }
    let r_grib = earth_radius_m(grid.shape_of_earth).ok_or_else(|| {
        GribIngestError::Shape(format!(
            "unsupported shape-of-earth {} (only spherical codes 0 and 6 appear in \
             NCEP regional output)",
            grid.shape_of_earth
        ))
    })?;
    let scale = optics::EARTH_RADIUS_M / r_grib;
    Ok(WrfProjectionParams {
        map_proj: 1,
        truelat1_deg: grid.latin1,
        truelat2_deg: grid.latin2,
        stand_lon_deg: normalize_lon(grid.lov),
        cen_lat_deg,
        cen_lon_deg,
        dx_m: grid.dx * scale,
        dy_m: grid.dy * scale,
    })
}

/// Synthesize the per-cell lat/lon planes (the wrfout `XLAT`/`XLONG` analog) from
/// the GRIB grid definition via grib-core's own grid math, longitudes normalized
/// to the WRF +/-180 convention. Row order matches the decoded data (scan 0x40).
fn latlon_planes(grid: &GridDefinition) -> Result<(Vec<f32>, Vec<f32>), GribIngestError> {
    let (lats, lons) = grid_latlon(grid);
    let expected = (grid.nx as usize) * (grid.ny as usize);
    if lats.len() != expected || lons.len() != expected {
        return Err(GribIngestError::Shape(format!(
            "grid template 3.{} produced no coordinates (unsupported template?)",
            grid.template
        )));
    }
    let xlat: Vec<f32> = lats.into_iter().map(|v| v as f32).collect();
    let xlong: Vec<f32> = lons.into_iter().map(|v| normalize_lon(v) as f32).collect();
    Ok((xlat, xlong))
}

// ── time / run id ──────────────────────────────────────────────────────────────

/// Valid time = reference time + forecast horizon. Returns the wrfout-style ISO
/// string (`YYYY-MM-DDTHH:MM:SSZ`, what `bricks::time_stamp` keys on) + `HHMM`.
pub fn valid_time(
    reference: NaiveDateTime,
    forecast_time: u32,
    time_range_unit: u8,
) -> Result<(String, u16), GribIngestError> {
    let delta = match time_range_unit {
        0 => chrono::Duration::minutes(forecast_time as i64),
        1 => chrono::Duration::hours(forecast_time as i64),
        2 => chrono::Duration::days(forecast_time as i64),
        u => {
            return Err(GribIngestError::Shape(format!(
                "unsupported forecast time-range unit {u} (0=min, 1=hr, 2=day supported)"
            )));
        }
    };
    let valid = reference + delta;
    let iso = valid.format("%Y-%m-%dT%H:%M:%SZ").to_string();
    let hhmm = (valid.hour() * 100 + valid.minute()) as u16;
    Ok((iso, hhmm))
}

/// Local copy of the wrfout run-id sanitizer (private in `ingest.rs`, which a
/// parallel agent owns): ascii-lowercase alphanumerics, everything else `_`,
/// runs collapsed, edges trimmed.
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

/// Default run id for a GRIB input: the sanitized file name + the CYCLE
/// (reference) date. HRRR/RRFS file names carry no date (it lives in the bucket
/// directory name), so `hrrr.t20z.wrfnatf00.grib2` downloaded on two days would
/// collide without it.
pub fn default_grib_run_id(path: &Path, reference: NaiveDateTime) -> String {
    let stem = path.file_name().and_then(|s| s.to_str()).unwrap_or("run");
    format!(
        "{}_{:04}{:02}{:02}",
        sanitize_token(stem),
        reference.year(),
        reference.month(),
        reference.day()
    )
}

// ── two-pass channel encoding (no brick-resolution f32 buffer) ─────────────────

/// Iterate every column of a native volume, resampling onto the brick axis with
/// the integral-conserving kernel, and hand each column to `sink(base_ci, col)`.
#[allow(clippy::too_many_arguments)]
fn for_each_resampled_column<F: FnMut(usize, &[f64])>(
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
    mut sink: F,
) {
    let plane = nx * ny;
    let mut zc = vec![0f64; nz_native];
    let mut fcol = vec![0f64; nz_native];
    let mut cum: Vec<f64> = Vec::with_capacity(nz_native);
    let mut col: Vec<f64> = Vec::with_capacity(nz_brick);
    for ci in 0..plane {
        for (k, (zc_k, fc_k)) in zc.iter_mut().zip(fcol.iter_mut()).enumerate() {
            let idx = k * plane + ci;
            *zc_k = z[idx] as f64;
            *fc_k = native[idx] as f64;
        }
        resample_column_conservative(
            &zc, &fcol, z_min, dz, nz_brick, below, above, &mut cum, &mut col,
        );
        sink(ci, &col);
    }
}

/// Two-pass conservative resample + log-quant encode of one native field into a
/// brick u8 channel, WITHOUT materializing the brick-resolution f32 volume
/// (pass 1 streams the quantization stats exactly as `LogQuant::from_values`
/// would compute them; pass 2 encodes straight to u8). When `beta_total_f16` is
/// given, pass 2 also accumulates the channel's resampled extinction into the
/// running f16 total (for `tau_up`).
#[allow(clippy::too_many_arguments)]
pub fn resample_encode_channel(
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
    mut beta_total_f16: Option<&mut [u16]>,
) -> (LogQuant, Vec<u8>) {
    let plane = nx * ny;
    // Pass 1: the exact `LogQuant::from_values` statistic over the values pass 2
    // will encode (each column value cast to f32 first, as the stored volume is).
    let mut vmax = 0.0f64;
    for_each_resampled_column(
        native,
        z,
        nx,
        ny,
        nz_native,
        z_min,
        dz,
        nz_brick,
        below,
        above,
        |_ci, col| {
            for &v in col {
                let vf = v as f32;
                if vf.is_finite() && vf > 0.0 {
                    vmax = vmax.max(vf as f64);
                }
            }
        },
    );
    let quant = if vmax <= 0.0 {
        LogQuant {
            vmin: 0.0,
            vmax: 0.0,
        }
    } else {
        LogQuant {
            vmin: vmax / QUANT_DYNAMIC_RANGE,
            vmax,
        }
    };
    // Pass 2: encode + (optionally) accumulate the running total extinction.
    let mut codes = vec![0u8; plane * nz_brick];
    for_each_resampled_column(
        native,
        z,
        nx,
        ny,
        nz_native,
        z_min,
        dz,
        nz_brick,
        below,
        above,
        |ci, col| {
            for (m, &v) in col.iter().enumerate() {
                let vf = v as f32;
                let idx = m * plane + ci;
                codes[idx] = quant.encode(vf);
                if let Some(bt) = beta_total_f16.as_deref_mut()
                    && vf.is_finite()
                    && vf > 0.0
                {
                    bt[idx] = f32_to_f16_bits(f16_bits_to_f32(bt[idx]) + vf);
                }
            }
        },
    );
    (quant, codes)
}

/// Point-sample (linear) temperature resample straight to the brick's f16-Celsius
/// bits — the value chain matches the wrfout path bit-for-bit (`resample_volume`
/// to f32, then `encode_temperature_celsius`), without the f32 volume.
#[allow(clippy::too_many_arguments)]
fn resample_temperature_f16(
    t_kelvin: &[f32],
    z: &[f32],
    nx: usize,
    ny: usize,
    nz_native: usize,
    z_min: f64,
    dz: f64,
    nz_brick: usize,
) -> Vec<u16> {
    let plane = nx * ny;
    let mut out = vec![0u16; plane * nz_brick];
    let mut zc = vec![0f64; nz_native];
    let mut fcol = vec![0f64; nz_native];
    let mut col: Vec<f64> = Vec::with_capacity(nz_brick);
    for ci in 0..plane {
        for (k, (zc_k, fc_k)) in zc.iter_mut().zip(fcol.iter_mut()).enumerate() {
            let idx = k * plane + ci;
            *zc_k = z[idx] as f64;
            *fc_k = t_kelvin[idx] as f64;
        }
        resample_column(
            &zc,
            &fcol,
            z_min,
            dz,
            nz_brick,
            Extrap::ClampEdge,
            Extrap::ClampEdge,
            &mut col,
        );
        for (m, &v) in col.iter().enumerate() {
            let vf = v as f32;
            out[m * plane + ci] = f32_to_f16_bits((vf as f64 - CELSIUS_OFFSET_K) as f32);
        }
    }
    out
}

/// Two-pass `tau_up` from the accumulated f16 total extinction (top-down
/// cumulative optical depth per column, then log-quant encode).
fn encode_tau_from_beta_total(
    beta_total_f16: &[u16],
    nx: usize,
    ny: usize,
    nz_brick: usize,
    dz: f64,
) -> (LogQuant, Vec<u8>) {
    let plane = nx * ny;
    let mut col = vec![0f64; nz_brick];
    let mut vmax = 0.0f64;
    for ci in 0..plane {
        for (m, c) in col.iter_mut().enumerate() {
            *c = f16_bits_to_f32(beta_total_f16[m * plane + ci]) as f64;
        }
        let tau = integrate_tau_up_column(&col, dz);
        for &v in &tau {
            let vf = v as f32;
            if vf.is_finite() && vf > 0.0 {
                vmax = vmax.max(vf as f64);
            }
        }
    }
    let quant = if vmax <= 0.0 {
        LogQuant {
            vmin: 0.0,
            vmax: 0.0,
        }
    } else {
        LogQuant {
            vmin: vmax / QUANT_DYNAMIC_RANGE,
            vmax,
        }
    };
    let mut codes = vec![0u8; plane * nz_brick];
    for ci in 0..plane {
        for (m, c) in col.iter_mut().enumerate() {
            *c = f16_bits_to_f32(beta_total_f16[m * plane + ci]) as f64;
        }
        let tau = integrate_tau_up_column(&col, dz);
        for (m, &v) in tau.iter().enumerate() {
            codes[m * plane + ci] = quant.encode(v as f32);
        }
    }
    (quant, codes)
}

// ── volume / plane readers ─────────────────────────────────────────────────────

/// Read a complete hybrid-level volume as f32 in brick layout `(k*ny+y)*nx+x`
/// (GRIB level 1 -> k = 0), cropped to `rect`. NaN (bitmap) cells are preserved
/// for the caller's policy.
#[allow(clippy::too_many_arguments)]
fn read_hybrid_volume(
    file: &mut File,
    catalog: &GribCatalog,
    code: FieldCode,
    field: &str,
    nz: usize,
    rect: &CropRect,
    buf: &mut Vec<u8>,
) -> Result<Vec<f32>, GribIngestError> {
    let entries = require_hybrid_volume_entries(catalog, code, field, nz)?;
    let full_nx = catalog.grid.nx as usize;
    let plane = rect.nx() * rect.ny();
    let mut out = vec![f32::NAN; plane * nz];
    for (k, entry) in entries.iter().enumerate() {
        let values = crop_values(decode_entry(file, catalog, entry, buf)?, full_nx, rect);
        let dst = &mut out[k * plane..(k + 1) * plane];
        for (d, &v) in dst.iter_mut().zip(values.iter()) {
            *d = v as f32;
        }
    }
    Ok(out)
}

/// Decode the optional native cloud-fraction volume. Unlike required fields,
/// any catalog/decode/value problem rejects only this auxiliary source and lets
/// the caller retain the exact full-cell fallback.
#[allow(clippy::too_many_arguments)]
fn read_optional_cloud_fraction_volume(
    file: &mut File,
    catalog: &GribCatalog,
    nz: usize,
    rect: &CropRect,
    buf: &mut Vec<u8>,
) -> Result<Option<Vec<f32>>, String> {
    let Some(entries) = optional_cloud_fraction_entries(&catalog.entries, nz)? else {
        return Ok(None);
    };
    let full_nx = catalog.grid.nx as usize;
    let plane = rect.nx() * rect.ny();
    let mut out = vec![0.0f32; plane * nz];
    for (k, entry) in entries.iter().enumerate() {
        let values = decode_entry(file, catalog, entry, buf).map_err(|e| e.to_string())?;
        let values = crop_values(values, full_nx, rect);
        if values.len() != plane {
            return Err(format!(
                "cc (0/6/32) hybrid level {} decoded {} cropped cells, expected {plane}",
                k + 1,
                values.len()
            ));
        }
        let dst = &mut out[k * plane..(k + 1) * plane];
        for (d, &value) in dst.iter_mut().zip(values.iter()) {
            *d = value as f32;
        }
    }
    normalize_native_cloud_fraction(&mut out)?;
    Ok(Some(out))
}

#[inline]
fn encode_cloud_fraction_value(value: f64) -> u8 {
    // Match bricks::encode_cloud_fraction exactly, including the f32 rounding
    // chain and preservation of every tiny positive fraction as code 1.
    let f = value as f32;
    if !f.is_finite() || f <= 0.0 {
        0
    } else {
        ((f.min(1.0) * 255.0).round() as u8).max(1)
    }
}

fn cloud_fraction_channel_or_fallback(candidate: Option<Vec<u8>>, cells: usize) -> (Vec<u8>, bool) {
    match candidate {
        Some(codes) if codes.len() == cells => (codes, true),
        _ => (vec![255u8; cells], false),
    }
}

/// Maximum-overlap resample of the native intensive coverage profile directly
/// into SSB v5's linear-u8 channel. This is the same vertical closure as the WRF
/// ingest and avoids materializing a brick-resolution f32 volume.
#[allow(clippy::too_many_arguments)]
fn resample_encode_cloud_fraction(
    native: &[f32],
    z: &[f32],
    nx: usize,
    ny: usize,
    nz_native: usize,
    z_min: f64,
    dz: f64,
    nz_brick: usize,
) -> Result<Vec<u8>, String> {
    let plane = nx
        .checked_mul(ny)
        .ok_or_else(|| "cloud-fraction plane dimensions overflow".to_string())?;
    let native_cells = plane
        .checked_mul(nz_native)
        .ok_or_else(|| "cloud-fraction native dimensions overflow".to_string())?;
    if native.len() != native_cells || z.len() != native_cells {
        return Err(format!(
            "cc (0/6/32) volume shape is incompatible: fraction={} height={} expected={native_cells}",
            native.len(),
            z.len()
        ));
    }
    if !z_min.is_finite() || !dz.is_finite() || dz <= 0.0 {
        return Err(format!(
            "cloud-fraction brick axis is invalid (z_min={z_min}, dz={dz})"
        ));
    }

    let mut out = vec![0u8; plane * nz_brick];
    let mut zc = vec![0f64; nz_native];
    let mut fc = vec![0f64; nz_native];
    let mut col = Vec::with_capacity(nz_brick);
    for ci in 0..plane {
        for k in 0..nz_native {
            let idx = k * plane + ci;
            zc[k] = z[idx] as f64;
            fc[k] = native[idx] as f64;
        }
        if zc.iter().any(|v| !v.is_finite()) || zc.windows(2).any(|pair| pair[1] <= pair[0]) {
            return Err(format!(
                "cc (0/6/32) column {ci} is incompatible with a finite, bottom-to-top height grid"
            ));
        }
        resample_column_fraction_max_overlap(
            &zc,
            &fc,
            z_min,
            dz,
            nz_brick,
            Extrap::Zero,
            Extrap::Zero,
            &mut col,
        );
        for (m, &value) in col.iter().enumerate() {
            out[m * plane + ci] = encode_cloud_fraction_value(value);
        }
    }
    Ok(out)
}

/// Fold one hydrometeor species into the native extinction buffer at its OWN
/// optics (extinctions add linearly; the SSB v3 per-species rule). Missing
/// species contribute nothing (like the wrfout `read_3d_opt` -> zeros path);
/// a PRESENT species must have complete levels. Returns whether it was present.
#[allow(clippy::too_many_arguments)]
fn add_species_beta(
    file: &mut File,
    catalog: &GribCatalog,
    code: FieldCode,
    field: &str,
    class: HydrometeorClass,
    rho: &[f32],
    beta: &mut [f32],
    nz: usize,
    rect: &CropRect,
    buf: &mut Vec<u8>,
) -> Result<bool, GribIngestError> {
    let map = hybrid_level_entries(catalog, code);
    if map.is_empty() {
        return Ok(false);
    }
    let entries = require_hybrid_volume_entries(catalog, code, field, nz)?;
    let full_nx = catalog.grid.nx as usize;
    let plane = rect.nx() * rect.ny();
    let radius = class.effective_radius_m();
    for (k, entry) in entries.iter().enumerate() {
        let values = crop_values(decode_entry(file, catalog, entry, buf)?, full_nx, rect);
        let base = k * plane;
        for (i, &q) in values.iter().enumerate() {
            if q.is_finite() && q > 0.0 {
                beta[base + i] +=
                    optics::extinction_coefficient(rho[base + i] as f64, q, radius) as f32;
            }
        }
    }
    Ok(true)
}

/// Read a single-level plane as f32 (brick row order), cropped to `rect`.
/// `level_value = None` matches any level value at that level type; `Some(v)`
/// matches within 0.5 (grib-core's `find` tolerance). Returns `None` when absent.
fn read_plane(
    file: &mut File,
    catalog: &GribCatalog,
    code: FieldCode,
    level_type: u8,
    level_value: Option<f64>,
    rect: &CropRect,
    buf: &mut Vec<u8>,
) -> Result<Option<Vec<f32>>, GribIngestError> {
    let entry = catalog.entries.iter().find(|e| {
        e.code == code
            && e.level_type == level_type
            && level_value.is_none_or(|lv| (e.level_value - lv).abs() < 0.5)
    });
    let Some(entry) = entry else {
        return Ok(None);
    };
    let full_nx = catalog.grid.nx as usize;
    let values = crop_values(decode_entry(file, catalog, entry, buf)?, full_nx, rect);
    Ok(Some(values.into_iter().map(|v| v as f32).collect()))
}

/// NaN / negative -> 0 for non-negative surface quantities (snow depth, land
/// mask; RRFS bitmaps mask water/off-domain cells).
fn sanitize_nonneg_plane(values: &mut [f32]) {
    for v in values.iter_mut() {
        if !v.is_finite() || *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// NaN -> 0 (off-domain masked cells; sign preserved — winds are signed).
fn sanitize_nan_plane(values: &mut [f32]) {
    for v in values.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
}

// ── ingest-time crop (oversize grids: the RRFS NA rotated grid) ────────────────

/// The largest grid axis this ingest accepts WITHOUT a crop. Mirrors the render
/// side's `camera::MAX_AXIS` (4096): a full RRFS NA grid (4881x2961, 14.45M
/// columns) exceeds both the raster cap and the brick memory budget, so it must
/// be cropped at ingest.
pub const GRIB_MAX_INGEST_AXIS: usize = 4096;

/// A TRUE-geographic crop box (degrees, lon in +/-180). The ingest keeps every
/// grid cell whose coordinates fall inside it — the INDEX HULL, so the kept
/// sub-grid stays rectangular in the grid's own (rotated) coordinates.
/// Antimeridian-straddling boxes are not supported (`lon_min < lon_max`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GribCrop {
    pub lat_min: f64,
    pub lat_max: f64,
    pub lon_min: f64,
    pub lon_max: f64,
}

/// The default CONUS crop: the HRRR-coverage box. On the RRFS NA rotated grid
/// this selects ~1800x1300 cells (measured on the staged fixture) — the same
/// order as the full HRRR grid.
pub const CONUS_CROP: GribCrop = GribCrop {
    lat_min: 21.0,
    lat_max: 50.5,
    lon_min: -125.0,
    lon_max: -66.0,
};

/// Parse a crop argument: `conus` or `lat_min,lat_max,lon_min,lon_max`.
pub fn parse_crop(s: &str) -> Result<GribCrop, GribIngestError> {
    let t = s.trim();
    if t.eq_ignore_ascii_case("conus") {
        return Ok(CONUS_CROP);
    }
    let parts: Vec<&str> = t.split(',').map(str::trim).collect();
    if parts.len() != 4 {
        return Err(GribIngestError::Shape(format!(
            "crop must be `conus` or `lat_min,lat_max,lon_min,lon_max`, got `{s}`"
        )));
    }
    let mut v = [0f64; 4];
    for (dst, p) in v.iter_mut().zip(parts.iter()) {
        *dst = p
            .parse::<f64>()
            .map_err(|_| GribIngestError::Shape(format!("crop component `{p}` is not a number")))?;
    }
    let crop = GribCrop {
        lat_min: v[0],
        lat_max: v[1],
        lon_min: v[2],
        lon_max: v[3],
    };
    let finite = v.iter().all(|x| x.is_finite());
    if !finite || crop.lat_min >= crop.lat_max || crop.lon_min >= crop.lon_max {
        return Err(GribIngestError::Shape(format!(
            "crop box is empty or inverted (lat {}..{}, lon {}..{}); antimeridian-\
             straddling boxes are not supported",
            crop.lat_min, crop.lat_max, crop.lon_min, crop.lon_max
        )));
    }
    Ok(crop)
}

/// GRIB-ingest options beyond [`IngestConfig`] (which is shared with the wrfout
/// path and stays untouched).
#[derive(Debug, Clone, Copy, Default)]
pub struct GribIngestOptions {
    /// Keep only the sub-grid whose cells fall inside this true-geographic box.
    /// REQUIRED for grids over [`GRIB_MAX_INGEST_AXIS`] (the RRFS NA grid).
    pub crop: Option<GribCrop>,
}

/// An inclusive index rectangle within the native grid (the kept sub-grid).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropRect {
    pub i0: usize,
    pub i1: usize,
    pub j0: usize,
    pub j1: usize,
}

impl CropRect {
    pub fn full(nx: usize, ny: usize) -> Self {
        Self {
            i0: 0,
            i1: nx - 1,
            j0: 0,
            j1: ny - 1,
        }
    }
    pub fn nx(&self) -> usize {
        self.i1 - self.i0 + 1
    }
    pub fn ny(&self) -> usize {
        self.j1 - self.j0 + 1
    }
    fn is_full(&self, full_nx: usize, full_ny: usize) -> bool {
        self.i0 == 0 && self.j0 == 0 && self.i1 == full_nx - 1 && self.j1 == full_ny - 1
    }
}

/// The no-crop admission gate: a grid over the axis cap REQUIRES a crop, with
/// the remedy named in the error (the approved full-NA refusal).
pub fn require_crop_for_oversize(
    nx: usize,
    ny: usize,
    has_crop: bool,
) -> Result<(), GribIngestError> {
    if !has_crop && nx.max(ny) > GRIB_MAX_INGEST_AXIS {
        return Err(GribIngestError::Shape(format!(
            "grid {nx}x{ny} exceeds the {GRIB_MAX_INGEST_AXIS}-cell ingest axis cap \
             (the full RRFS NA domain); pass a crop — `crop=conus` or \
             `crop=lat_min,lat_max,lon_min,lon_max` — to ingest the region you need"
        )));
    }
    Ok(())
}

/// The ingest peak-RSS contract (the design's < 2.5 GB, hard-asserted by the
/// fixture tests).
pub const GRIB_PEAK_RSS_BUDGET_BYTES: u64 = 2_500_000_000;

/// Estimate the ingest's peak-RSS-driving bytes for a kept sub-grid, CALIBRATED
/// against the measured HRRR fixture ingest (1799x1059x50 native -> 80-level
/// brick: estimate 2.37 GB, measured 2.34 GB). Two candidate peaks:
/// the FOLD phase (three native f32 volumes: z + rho + the per-channel beta,
/// plus the f16 running total, the finished u8 channels, the f16 temperature)
/// and the WRITE phase. SSB v5's writer streams directly into zlib, so WRITE no
/// longer duplicates the complete raw payload; the two v4+ fallback channels are
/// allocated only after native fold buffers are dropped. The fixed term covers
/// the one decoded full-grid f64 plane in flight plus process overhead.
pub fn estimated_ingest_peak_bytes(
    columns: usize,
    nz_native: usize,
    nz_brick: usize,
    full_plane_points: usize,
) -> u64 {
    let c = columns as u64;
    let nzn = nz_native as u64;
    let nzb = nz_brick as u64;
    let fixed = 8 * full_plane_points as u64 + 150_000_000;
    let fold = c * (3 * nzn * 4 + 7 * nzb) + fixed;
    let write = c * (9 * nzb) + fixed + 100_000_000;
    fold.max(write)
}

/// The peak-RSS admission gate: refuse an ingest whose ESTIMATE busts the
/// 2.5 GB contract, naming the remedy. Discovered on the real RRFS grid: the
/// rotated NA grid is tilted ~17-20 deg relative to CONUS, so the full-CONUS
/// box hulls to ~2376x1564 cells (~3.7M columns) — the brick write phase ALONE
/// would exceed the contract, so no streaming trick admits it. A regional crop
/// (e.g. `25,49.5,-110,-78`) fits comfortably.
pub fn require_ingest_fits_budget(
    columns: usize,
    nz_native: usize,
    nz_brick: usize,
    full_plane_points: usize,
) -> Result<(), GribIngestError> {
    let est = estimated_ingest_peak_bytes(columns, nz_native, nz_brick, full_plane_points);
    if est > GRIB_PEAK_RSS_BUDGET_BYTES {
        return Err(GribIngestError::Shape(format!(
            "the selected sub-grid ({columns} columns x {nz_native} native levels) has an \
             estimated ingest peak of {:.2} GB, over the {:.1} GB contract; shrink the \
             crop box (e.g. `crop=25,49.5,-110,-78` for the central/eastern CONUS on the \
             RRFS NA grid — its rotated grid is tilted relative to CONUS, so a full-CONUS \
             box needs ~2x the cells the same region needs on the HRRR Lambert grid)",
            est as f64 / 1.0e9,
            GRIB_PEAK_RSS_BUDGET_BYTES as f64 / 1.0e9
        )));
    }
    Ok(())
}

/// Find the index hull of every cell inside the crop box, scanning the full
/// lat/lon planes (works for ANY grid template — the hull is rectangular in the
/// grid's own coordinates by construction).
pub fn crop_rect_from_planes(
    xlat: &[f32],
    xlong: &[f32],
    nx: usize,
    ny: usize,
    crop: &GribCrop,
) -> Result<CropRect, GribIngestError> {
    let (mut i0, mut i1, mut j0, mut j1) = (usize::MAX, 0usize, usize::MAX, 0usize);
    for j in 0..ny {
        for i in 0..nx {
            let idx = j * nx + i;
            let (la, lo) = (xlat[idx] as f64, xlong[idx] as f64);
            if la >= crop.lat_min && la <= crop.lat_max && lo >= crop.lon_min && lo <= crop.lon_max
            {
                i0 = i0.min(i);
                i1 = i1.max(i);
                j0 = j0.min(j);
                j1 = j1.max(j);
            }
        }
    }
    if i0 == usize::MAX {
        return Err(GribIngestError::Shape(format!(
            "crop box (lat {}..{}, lon {}..{}) selects no grid cells",
            crop.lat_min, crop.lat_max, crop.lon_min, crop.lon_max
        )));
    }
    let rect = CropRect { i0, i1, j0, j1 };
    if rect.nx() < 2 || rect.ny() < 2 {
        return Err(GribIngestError::Shape(format!(
            "crop selects a degenerate {}x{} sub-grid",
            rect.nx(),
            rect.ny()
        )));
    }
    if rect.nx().max(rect.ny()) > GRIB_MAX_INGEST_AXIS {
        return Err(GribIngestError::Shape(format!(
            "crop still selects {}x{} cells, over the {GRIB_MAX_INGEST_AXIS}-cell \
             axis cap; shrink the box",
            rect.nx(),
            rect.ny()
        )));
    }
    Ok(rect)
}

/// Slice a decoded full-grid plane down to the crop rect (no-op for the full rect).
fn crop_values(values: Vec<f64>, full_nx: usize, rect: &CropRect) -> Vec<f64> {
    let full_ny = values.len() / full_nx;
    if rect.is_full(full_nx, full_ny) {
        return values;
    }
    let mut out = Vec::with_capacity(rect.nx() * rect.ny());
    for j in rect.j0..=rect.j1 {
        let base = j * full_nx;
        out.extend_from_slice(&values[base + rect.i0..=base + rect.i1]);
    }
    out
}

/// Slice an f32 plane (the synthesized lat/lon planes) down to the crop rect.
fn crop_plane_f32(values: &[f32], full_nx: usize, rect: &CropRect) -> Vec<f32> {
    let full_ny = values.len() / full_nx;
    if rect.is_full(full_nx, full_ny) {
        return values.to_vec();
    }
    let mut out = Vec::with_capacity(rect.nx() * rect.ny());
    for j in rect.j0..=rect.j1 {
        let base = j * full_nx;
        out.extend_from_slice(&values[base + rect.i0..=base + rect.i1]);
    }
    out
}

// ── geometry probe ─────────────────────────────────────────────────────────────

/// A cheap GRIB probe (one metadata pass over the message headers, NO field
/// decode): dims, valid/reference time, message count, and the run id an ingest
/// would produce. The GRIB analog of [`crate::ingest::WrfProbe`] — the seam the
/// (deferred) api.rs `resolve_source` diff and the studio open flow consume
/// (cache-path prediction + size gate + pickers).
#[derive(Debug, Clone)]
pub struct GribProbe {
    pub nx: usize,
    pub ny: usize,
    /// Native hybrid-level count (the brick still resamples to `nz_brick`).
    pub nz: usize,
    /// Valid time (reference + forecast horizon), wrfout-style ISO.
    pub time_iso: String,
    pub hhmm: u16,
    /// Cycle (reference) time, ISO.
    pub reference_iso: String,
    /// What [`ingest_grib_timestep`] will use when `config.run_id` is `None`.
    pub default_run_id: String,
    pub messages: usize,
    pub file_bytes: u64,
}

/// Probe a GRIB input's dims/time/run-id cheaply (metadata pass, no decode).
/// Reports the FILE's own full-grid dims (no crop, no oversize refusal — the
/// probe describes the input; the ingest admission gate is separate).
pub fn probe_grib(path: &Path) -> Result<GribProbe, GribIngestError> {
    let catalog = build_catalog(path)?;
    let tmp_levels: Vec<u32> = hybrid_level_entries(&catalog, CODE_TMP)
        .keys()
        .copied()
        .collect();
    let nz = validate_complete_levels(&tmp_levels, "TMP (temperature)")?;
    let (time_iso, hhmm) = valid_time(
        catalog.reference_time,
        catalog.forecast_time,
        catalog.time_range_unit,
    )?;
    Ok(GribProbe {
        nx: catalog.grid.nx as usize,
        ny: catalog.grid.ny as usize,
        nz,
        time_iso,
        hhmm,
        reference_iso: catalog
            .reference_time
            .format("%Y-%m-%dT%H:%M:%SZ")
            .to_string(),
        default_run_id: default_grib_run_id(path, catalog.reference_time),
        messages: catalog.locs.len(),
        file_bytes: std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
    })
}

/// Read the grid geometry + projection + valid time of a GRIB input WITHOUT
/// decoding any field (one metadata pass over the message headers). The GRIB
/// analog of [`crate::ingest::read_grid_geometry`]; the fixture ratchet and the
/// (deferred) api/studio seams consume it. Oversize grids (RRFS NA) are refused
/// here exactly as at ingest — use [`read_grib_geometry_with`] to crop.
pub fn read_grib_geometry(path: &Path) -> Result<GridGeometry, GribIngestError> {
    read_grib_geometry_with(path, &GribIngestOptions::default())
}

/// [`read_grib_geometry`] with options: the returned geometry describes the
/// CROPPED sub-grid (dims, planes, centre anchor) when a crop is given.
pub fn read_grib_geometry_with(
    path: &Path,
    options: &GribIngestOptions,
) -> Result<GridGeometry, GribIngestError> {
    let catalog = build_catalog(path)?;
    Ok(grib_geometry(&catalog, options)?.0)
}

/// The geometry of the (possibly cropped) sub-grid + the crop rect the field
/// readers must apply. Enforces the oversize-without-crop refusal.
fn grib_geometry(
    catalog: &GribCatalog,
    options: &GribIngestOptions,
) -> Result<(GridGeometry, CropRect), GribIngestError> {
    let full_nx = catalog.grid.nx as usize;
    let full_ny = catalog.grid.ny as usize;
    if full_nx < 2 || full_ny < 2 {
        return Err(GribIngestError::Shape(format!(
            "degenerate grid {full_nx}x{full_ny}"
        )));
    }
    require_crop_for_oversize(full_nx, full_ny, options.crop.is_some())?;
    let tmp_levels: Vec<u32> = hybrid_level_entries(catalog, CODE_TMP)
        .keys()
        .copied()
        .collect();
    let nz = validate_complete_levels(&tmp_levels, "TMP (temperature)")?;
    // Full planes first (grib-core's own grid math is the coordinate truth),
    // then the crop hull + the cropped planes.
    let (xlat_full, xlong_full) = latlon_planes(&catalog.grid)?;
    let rect = match &options.crop {
        None => CropRect::full(full_nx, full_ny),
        Some(crop) => crop_rect_from_planes(&xlat_full, &xlong_full, full_nx, full_ny, crop)?,
    };
    let (xlat, xlong) = (
        crop_plane_f32(&xlat_full, full_nx, &rect),
        crop_plane_f32(&xlong_full, full_nx, &rect),
    );
    drop(xlat_full);
    drop(xlong_full);
    let (nx, ny) = (rect.nx(), rect.ny());
    let c = ((ny - 1) / 2) * nx + (nx - 1) / 2;
    let params = params_from_grid(&catalog.grid, xlat[c] as f64, xlong[c] as f64)?;
    let (time_iso, hhmm) = valid_time(
        catalog.reference_time,
        catalog.forecast_time,
        catalog.time_range_unit,
    )?;
    if !rect.is_full(full_nx, full_ny) {
        crate::log_line!(
            "simsat grib ingest: crop keeps [{}..={}, {}..={}] of {full_nx}x{full_ny} \
             -> {nx}x{ny} cells",
            rect.i0,
            rect.i1,
            rect.j0,
            rect.j1
        );
    }
    Ok((
        GridGeometry {
            nx,
            ny,
            nz,
            nz_stag: nz + 1,
            nt: 1,
            params,
            xlat,
            xlong,
            time_iso: Some(time_iso),
            hhmm,
        },
        rect,
    ))
}

// ── the ingest ─────────────────────────────────────────────────────────────────

/// Ingest one HRRR/RRFS native-level GRIB2 file into an `.ssb` brick +
/// `run.json`, streaming (one native 3-D field resident at a time; two-pass
/// channel encode — module doc). A GRIB file carries ONE valid time, so
/// `config.timestep` must be 0. Reuses [`IngestConfig`] / [`IngestReport`].
/// Oversize grids (the RRFS NA rotated grid) are refused — use
/// [`ingest_grib_timestep_with`] and a crop.
pub fn ingest_grib_timestep(
    path: &Path,
    config: &IngestConfig,
) -> Result<IngestReport, GribIngestError> {
    ingest_grib_timestep_with(path, config, &GribIngestOptions::default())
}

/// [`ingest_grib_timestep`] with [`GribIngestOptions`] (the RRFS crop).
pub fn ingest_grib_timestep_with(
    path: &Path,
    config: &IngestConfig,
    options: &GribIngestOptions,
) -> Result<IngestReport, GribIngestError> {
    platform::lower_ingest_thread_priority();
    let start = Instant::now();

    if config.timestep != 0 {
        return Err(GribIngestError::Shape(format!(
            "timestep {} is out of range: a GRIB file carries a single valid time \
             (use timestep 0; forecast hours are separate files)",
            config.timestep
        )));
    }

    let catalog = build_catalog(path)?;
    let (geom, rect) = grib_geometry(&catalog, options)?;
    let (nx, ny, nz) = (geom.nx, geom.ny, geom.nz);
    let plane = nx * ny;
    let (z_min, dz, nz_brick) = (config.z_min_m, config.dz_m, config.nz_brick);
    // Peak-RSS admission (the < 2.5 GB contract): refuse before allocating.
    let full_plane_points = (catalog.grid.nx as usize) * (catalog.grid.ny as usize);
    require_ingest_fits_budget(plane, nz, nz_brick, full_plane_points)?;
    let mut file = File::open(path)?;
    let mut buf: Vec<u8> = Vec::new();

    // Native z coordinate: per-level geopotential height (m MSL), directly.
    let mut z = read_hybrid_volume(
        &mut file,
        &catalog,
        CODE_HGT,
        "HGT (height)",
        nz,
        &rect,
        &mut buf,
    )?;
    let z_filled = fill_column_nan(&mut z, nx, ny, nz)?;

    // rho = p / (R_d T) per native cell; temperature resampled to f16 while the
    // native TMP volume is resident. rho is computed INTO the pressure buffer.
    let mut t_kelvin = read_hybrid_volume(
        &mut file,
        &catalog,
        CODE_TMP,
        "TMP (temperature)",
        nz,
        &rect,
        &mut buf,
    )?;
    let t_filled = fill_column_nan(&mut t_kelvin, nx, ny, nz)?;
    let mut rho = read_hybrid_volume(
        &mut file,
        &catalog,
        CODE_PRES,
        "PRES (pressure)",
        nz,
        &rect,
        &mut buf,
    )?;
    let p_filled = fill_column_nan(&mut rho, nx, ny, nz)?;
    for (r, &t) in rho.iter_mut().zip(t_kelvin.iter()) {
        *r = optics::air_density(*r as f64, t as f64) as f32;
    }
    if z_filled + t_filled + p_filled > 0 {
        crate::log_line!(
            "simsat grib ingest: filled {} masked cells from column neighbors (z {}, T {}, p {})",
            z_filled + t_filled + p_filled,
            z_filled,
            t_filled,
            p_filled
        );
    }
    let temperature_f16 = resample_temperature_f16(&t_kelvin, &z, nx, ny, nz, z_min, dz, nz_brick);
    drop(t_kelvin);

    // The running total extinction for tau_up, as f16 bits (halves the resident
    // footprint on the 1799x1059 HRRR grid; the f16 relative step ~5e-4 is far
    // below the u8 log-quant step the renderer reads).
    let mut beta_total_f16 = vec![0u16; plane * nz_brick];
    let mut beta = vec![0f32; plane * nz];

    // ext_liquid = CLMR at cloud-droplet optics.
    let (ql, ext_liquid) = {
        add_species_beta(
            &mut file,
            &catalog,
            CODE_CLMR,
            "CLMR (cloud water)",
            HydrometeorClass::CloudLiquid,
            &rho,
            &mut beta,
            nz,
            &rect,
            &mut buf,
        )?;
        resample_encode_channel(
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
            Some(&mut beta_total_f16),
        )
    };

    // ext_ice = cloud ice at small-ice optics. HRRR writes the NCEP-local CIMIXR
    // (0/1/82); RRFS writes the WMO ICMR (0/1/23). Prefer CIMIXR when present.
    let (qi, ext_ice) = {
        beta.iter_mut().for_each(|b| *b = 0.0);
        let had_cimixr = add_species_beta(
            &mut file,
            &catalog,
            CODE_CIMIXR,
            "CIMIXR (cloud ice)",
            HydrometeorClass::Ice,
            &rho,
            &mut beta,
            nz,
            &rect,
            &mut buf,
        )?;
        if !had_cimixr {
            add_species_beta(
                &mut file,
                &catalog,
                CODE_ICMR,
                "ICMR (cloud ice)",
                HydrometeorClass::Ice,
                &rho,
                &mut beta,
                nz,
                &rect,
                &mut buf,
            )?;
        }
        resample_encode_channel(
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
            Some(&mut beta_total_f16),
        )
    };

    // ext_precip = RWMR + GRLE (rain/graupel optics) + SNMR (snow aggregate
    // optics) — each species at its OWN beta before the sum (SSB v3).
    let (qp, ext_precip) = {
        beta.iter_mut().for_each(|b| *b = 0.0);
        add_species_beta(
            &mut file,
            &catalog,
            CODE_RWMR,
            "RWMR (rain)",
            HydrometeorClass::Rain,
            &rho,
            &mut beta,
            nz,
            &rect,
            &mut buf,
        )?;
        add_species_beta(
            &mut file,
            &catalog,
            CODE_GRLE,
            "GRLE (graupel)",
            HydrometeorClass::Graupel,
            &rho,
            &mut beta,
            nz,
            &rect,
            &mut buf,
        )?;
        add_species_beta(
            &mut file,
            &catalog,
            CODE_SNMR,
            "SNMR (snow)",
            HydrometeorClass::Snow,
            &rho,
            &mut beta,
            nz,
            &rect,
            &mut buf,
        )?;
        resample_encode_channel(
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
            Some(&mut beta_total_f16),
        )
    };
    drop(rho);

    // qvapor channel: SPFH -> mixing ratio, integral-conserving (the PW column
    // is the physically meaningful quantity — same rationale as the wrfout path).
    let (qv, qvapor) = {
        beta.iter_mut().for_each(|b| *b = 0.0);
        let entries =
            require_hybrid_volume_entries(&catalog, CODE_SPFH, "SPFH (specific humidity)", nz)?;
        let full_nx = catalog.grid.nx as usize;
        for (k, entry) in entries.iter().enumerate() {
            let values = crop_values(
                decode_entry(&mut file, &catalog, entry, &mut buf)?,
                full_nx,
                &rect,
            );
            let base = k * plane;
            for (i, &q) in values.iter().enumerate() {
                beta[base + i] = spfh_to_mixing_ratio(q) as f32;
            }
        }
        resample_encode_channel(
            &beta,
            &z,
            nx,
            ny,
            nz,
            z_min,
            dz,
            nz_brick,
            Extrap::ClampEdge,
            Extrap::Zero,
            None,
        )
    };
    drop(beta);

    // Optional native grid-cell cloud coverage. HRRR supplies `cc` (0/6/32)
    // on the same 1..N hybrid levels as the hydrometeors. It is deliberately
    // fail-closed: a missing/partial/malformed/incompatible field leaves the
    // exact historical all-255 channel + false provenance, while the rest of
    // the GRIB ingest continues unchanged.
    let cloud_fraction_candidate = match read_optional_cloud_fraction_volume(
        &mut file, &catalog, nz, &rect, &mut buf,
    ) {
        Ok(Some(native)) => {
            match resample_encode_cloud_fraction(&native, &z, nx, ny, nz, z_min, dz, nz_brick) {
                Ok(codes) => {
                    crate::log_line!(
                        "simsat grib ingest: native cloud fraction ON (cc 0/6/32, {nz} hybrid levels)"
                    );
                    Some(codes)
                }
                Err(reason) => {
                    crate::log_line!(
                        "simsat grib ingest: native cloud fraction rejected ({reason}); using full-cell fallback"
                    );
                    None
                }
            }
        }
        Ok(None) => {
            crate::log_line!(
                "simsat grib ingest: native cloud fraction unavailable (cc 0/6/32 missing); using full-cell fallback"
            );
            None
        }
        Err(reason) => {
            crate::log_line!(
                "simsat grib ingest: native cloud fraction rejected ({reason}); using full-cell fallback"
            );
            None
        }
    };
    drop(z);

    // tau_up from the accumulated total extinction.
    let (qt, tau_up) = encode_tau_from_beta_total(&beta_total_f16, nx, ny, nz_brick, dz);
    drop(beta_total_f16);

    // SNMR remains in total ext_precip exactly as before; the snow-only
    // auxiliary subset is still conservatively unavailable.
    let qs = LogQuant {
        vmin: 0.0,
        vmax: 0.0,
    };
    let ext_snow = vec![0u8; plane * nz_brick];
    let (cloud_fraction, has_cloud_fraction) =
        cloud_fraction_channel_or_fallback(cloud_fraction_candidate, plane * nz_brick);

    // 2-D planes.
    let mut hgt = read_plane(
        &mut file,
        &catalog,
        CODE_HGT,
        LEVEL_SURFACE,
        None,
        &rect,
        &mut buf,
    )?
    .ok_or_else(|| GribIngestError::MissingField("HGT at surface (terrain)".to_string()))?;
    sanitize_nan_plane(&mut hgt);
    let mut landmask = read_plane(
        &mut file,
        &catalog,
        CODE_LAND,
        LEVEL_SURFACE,
        None,
        &rect,
        &mut buf,
    )?
    .unwrap_or_else(|| vec![0f32; plane]);
    sanitize_nonneg_plane(&mut landmask);
    let mut tsk = read_plane(
        &mut file,
        &catalog,
        CODE_TMP,
        LEVEL_SURFACE,
        None,
        &rect,
        &mut buf,
    )?
    .unwrap_or_else(|| vec![0f32; plane]);
    // NaN skin temperature -> 0: the render-side WS1 TSK fallback substitutes the
    // lowest-level air temperature for zero cells.
    sanitize_nan_plane(&mut tsk);
    let mut u10 = read_plane(
        &mut file,
        &catalog,
        CODE_UGRD,
        LEVEL_HEIGHT_AGL,
        Some(10.0),
        &rect,
        &mut buf,
    )?
    .unwrap_or_else(|| vec![0f32; plane]);
    sanitize_nan_plane(&mut u10);
    let mut v10 = read_plane(
        &mut file,
        &catalog,
        CODE_VGRD,
        LEVEL_HEIGHT_AGL,
        Some(10.0),
        &rect,
        &mut buf,
    )?
    .unwrap_or_else(|| vec![0f32; plane]);
    sanitize_nan_plane(&mut v10);
    let snowh = read_plane(
        &mut file,
        &catalog,
        CODE_SNOD,
        LEVEL_SURFACE,
        None,
        &rect,
        &mut buf,
    )?
    .map(|mut v| {
        sanitize_nonneg_plane(&mut v);
        v
    });
    let ivgtyp = read_plane(
        &mut file,
        &catalog,
        CODE_VGTYP,
        LEVEL_SURFACE,
        None,
        &rect,
        &mut buf,
    )?
    .map(|mut v| {
        sanitize_nan_plane(&mut v);
        v
    });

    let mut quant_map = BTreeMap::new();
    quant_map.insert("ext_liquid".to_string(), ql);
    quant_map.insert("ext_ice".to_string(), qi);
    quant_map.insert("ext_snow".to_string(), qs);
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
        ext_snow,
        ext_precip,
        tau_up,
        qvapor,
        cloud_fraction,
        has_cloud_fraction,
        temperature_f16,
        hgt,
        landmask,
        tsk,
        u10,
        v10,
        snowh,
        ivgtyp,
    };

    // Brick + manifest, exactly the wrfout tail (same cache layout, same schema).
    let run_id = config
        .run_id
        .clone()
        .unwrap_or_else(|| default_grib_run_id(path, catalog.reference_time));
    let dir = bricks::run_dir(&config.cache_dir, &run_id);
    let stamp = bricks::time_stamp(geom.time_iso.as_deref(), geom.hhmm);
    let brick_file = bricks::brick_file_name(&stamp);
    let brick_path = dir.join(&brick_file);
    let ssb_bytes = bricks::write_ssb(&brick_path, &brick)?;

    let manifest_path = RunManifest::path(&config.cache_dir, &run_id);
    let planes_2d = brick.planes_2d_names();
    let p = &geom.params;
    let projection = ManifestProjection {
        map_proj: p.map_proj,
        truelat1_deg: p.truelat1_deg,
        truelat2_deg: p.truelat2_deg,
        stand_lon_deg: p.stand_lon_deg,
        cen_lat_deg: p.cen_lat_deg,
        cen_lon_deg: p.cen_lon_deg,
        dx_m: p.dx_m,
        dy_m: p.dy_m,
    };
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
    let (source_bytes, source_mtime_unix) = source_identity(path);
    let anchor = geom.manifest_anchor();
    manifest.register_timestep(ManifestTimestep {
        key: stamp,
        hhmm: geom.hhmm,
        file: brick_file,
        time_iso: geom.time_iso,
        quant,
        has_cloud_fraction,
        ssb_bytes,
        source_bytes,
        source_mtime_unix,
        anchor,
    });
    manifest.save(&manifest_path)?;

    let wall = start.elapsed();
    let peak_rss_bytes = platform::peak_rss_bytes();
    crate::log_line!(
        "simsat grib ingest: run={run_id} dims={nx}x{ny}x{nz_brick} wall={:.2}s peak_rss={} ssb_bytes={ssb_bytes}",
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

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    // ── classification ─────────────────────────────────────────────────────────

    #[test]
    fn grib_extension_classifies() {
        assert!(is_grib_path(Path::new("hrrr.t20z.wrfnatf00.grib2")));
        assert!(is_grib_path(Path::new("UPPER.GRB2")));
        assert!(is_grib_path(Path::new("era.grib")));
        assert!(is_grib_path(Path::new("x.grb")));
        assert!(!is_grib_path(Path::new("run.json")));
        assert!(!is_grib_path(Path::new("wrfout_d03_2025-06-21_02_15_00")));
        assert!(!is_grib_path(Path::new("frame.png")));
    }

    #[test]
    fn grib_magic_sniff_classifies() {
        let dir = std::env::temp_dir().join(format!("simsat-grib-sniff-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let yes = dir.join("noext");
        std::fs::write(&yes, b"GRIB\x00\x00\x00\x02rest").unwrap();
        let no = dir.join("not_grib");
        std::fs::write(&no, b"CDF\x01netcdfish").unwrap();
        assert!(sniff_grib_magic(&yes));
        assert!(!sniff_grib_magic(&no));
        assert!(!sniff_grib_magic(&dir.join("missing")));
        assert!(is_grib_input(&yes));
        assert!(!is_grib_input(&no));
        std::fs::remove_dir_all(&dir).ok();
    }

    // ── message index ──────────────────────────────────────────────────────────

    fn synthetic_message(total_len: usize) -> Vec<u8> {
        // Section 0: "GRIB" + reserved(2) + discipline + edition + u64 BE length.
        let mut m = Vec::with_capacity(total_len);
        m.extend_from_slice(b"GRIB");
        m.extend_from_slice(&[0, 0, 0, 2]);
        m.extend_from_slice(&(total_len as u64).to_be_bytes());
        m.resize(total_len, 0xAA);
        m
    }

    #[test]
    fn message_index_walks_synthetic_stream() {
        let mut data = Vec::new();
        data.extend_from_slice(&synthetic_message(64));
        data.extend_from_slice(&synthetic_message(120));
        data.extend_from_slice(b"JUNKJUNKJUNKJUNKJUNK"); // trailing garbage stops cleanly
        let mut cur = std::io::Cursor::new(&data);
        let len = data.len() as u64;
        let locs = index_grib_messages(&mut cur, len).unwrap();
        assert_eq!(
            locs,
            vec![
                MessageLocation {
                    offset: 0,
                    length: 64
                },
                MessageLocation {
                    offset: 64,
                    length: 120
                },
            ]
        );
        // A message claiming to run past EOF is not indexed.
        let mut truncated = synthetic_message(64);
        truncated.extend_from_slice(b"GRIB\x00\x00\x00\x02");
        truncated.extend_from_slice(&(400u64).to_be_bytes());
        truncated.resize(truncated.len() + 8, 0);
        let tlen = truncated.len() as u64;
        let mut cur = std::io::Cursor::new(&truncated);
        let locs = index_grib_messages(&mut cur, tlen).unwrap();
        assert_eq!(locs.len(), 1);
    }

    // ── level structure ────────────────────────────────────────────────────────

    #[test]
    fn hybrid_levels_must_be_contiguous_from_one() {
        assert_eq!(validate_complete_levels(&[1, 2, 3, 4], "TMP").unwrap(), 4);
        assert!(validate_complete_levels(&[], "TMP").is_err());
        let gap = validate_complete_levels(&[1, 2, 4, 5], "TMP").unwrap_err();
        assert!(gap.to_string().contains("expected level 3"));
        let no_first = validate_complete_levels(&[2, 3], "TMP").unwrap_err();
        assert!(no_first.to_string().contains("expected level 1"));
    }

    // ── value policies ─────────────────────────────────────────────────────────

    fn cloud_fraction_entry(level: f64, loc: usize) -> CatalogEntry {
        CatalogEntry {
            loc,
            sub: 0,
            code: CODE_CLOUD_FRACTION,
            level_type: LEVEL_HYBRID,
            level_value: level,
        }
    }

    #[test]
    fn native_cloud_fraction_identity_and_level_order_are_guarded() {
        assert_eq!(CODE_CLOUD_FRACTION, fc(0, 6, 32));

        // Message order is irrelevant: native level 1 must occupy k=0, level 3
        // k=2, matching HGT/hydrometeor bottom-to-top storage.
        let entries = vec![
            cloud_fraction_entry(3.0, 30),
            CatalogEntry {
                loc: 99,
                sub: 0,
                code: CODE_TMP,
                level_type: LEVEL_HYBRID,
                level_value: 1.0,
            },
            cloud_fraction_entry(1.0, 10),
            cloud_fraction_entry(2.0, 20),
        ];
        let ordered = optional_cloud_fraction_entries(&entries, 3)
            .unwrap()
            .expect("complete native cc volume");
        assert_eq!(
            ordered.iter().map(|e| e.loc).collect::<Vec<_>>(),
            vec![10, 20, 30]
        );

        assert!(optional_cloud_fraction_entries(&[], 3).unwrap().is_none());
        let gap = optional_cloud_fraction_entries(
            &[cloud_fraction_entry(1.0, 1), cloud_fraction_entry(3.0, 3)],
            3,
        )
        .unwrap_err();
        assert!(gap.contains("expected level 2"), "{gap}");
        let short = optional_cloud_fraction_entries(
            &[cloud_fraction_entry(1.0, 1), cloud_fraction_entry(2.0, 2)],
            3,
        )
        .unwrap_err();
        assert!(short.contains("expected 3"), "{short}");
        let duplicate = optional_cloud_fraction_entries(
            &[cloud_fraction_entry(1.0, 1), cloud_fraction_entry(1.0, 2)],
            1,
        )
        .unwrap_err();
        assert!(duplicate.contains("duplicate"), "{duplicate}");
    }

    #[test]
    fn native_cloud_fraction_normalization_rejects_bad_scale_and_nonfinite() {
        let mut valid = vec![-5.0e-5, 0.0, 0.25, 1.0, 1.0 + 5.0e-5];
        normalize_native_cloud_fraction(&mut valid).unwrap();
        assert_eq!(valid, vec![0.0, 0.0, 0.25, 1.0, 1.0]);

        let mut nonfinite = vec![0.0, f32::NAN, 1.0];
        assert!(
            normalize_native_cloud_fraction(&mut nonfinite)
                .unwrap_err()
                .contains("non-finite")
        );
        let mut percent_like = vec![0.0, 50.0, 100.0];
        let err = normalize_native_cloud_fraction(&mut percent_like).unwrap_err();
        assert!(err.contains("refusing to guess percent scaling"), "{err}");
    }

    #[test]
    fn native_cloud_fraction_resamples_bottom_to_top_and_preserves_provenance() {
        let z = vec![0.0f32, 1000.0, 2000.0];
        let native = vec![0.1f32, 0.5, 0.9];
        let codes = resample_encode_cloud_fraction(&native, &z, 1, 1, 3, 0.0, 1000.0, 3)
            .expect("compatible cloud fraction");
        // Maximum overlap over the three brick layers yields 0.3, 0.7, 0.9.
        assert_eq!(codes, bricks::encode_cloud_fraction(&[0.3f32, 0.7, 0.9]));
        assert!(codes.windows(2).all(|pair| pair[1] > pair[0]));

        let (channel, has_native) = cloud_fraction_channel_or_fallback(Some(codes.clone()), 3);
        assert!(has_native);
        assert_eq!(channel, codes);

        // Missing or malformed candidates retain the exact historical channel.
        let (missing, has_native) = cloud_fraction_channel_or_fallback(None, 4);
        assert!(!has_native);
        assert_eq!(missing, vec![255; 4]);
        let (wrong_shape, has_native) = cloud_fraction_channel_or_fallback(Some(vec![0, 128]), 4);
        assert!(!has_native);
        assert_eq!(wrong_shape, vec![255; 4]);
    }

    #[test]
    fn spfh_converts_to_mixing_ratio() {
        // w = q/(1-q): 0.02 -> 0.020408...
        assert!((spfh_to_mixing_ratio(0.02) - 0.02 / 0.98).abs() < 1e-12);
        assert_eq!(spfh_to_mixing_ratio(0.0), 0.0);
        assert_eq!(spfh_to_mixing_ratio(-1.0e-6), 0.0);
        assert_eq!(spfh_to_mixing_ratio(f64::NAN), 0.0);
        // The mixing ratio exceeds specific humidity (denominator < 1).
        assert!(spfh_to_mixing_ratio(0.02) > 0.02);
    }

    #[test]
    fn column_nan_fill_takes_nearest_finite_neighbor() {
        // One column (nx=ny=1), nz=5: NaN at bottom, middle, top.
        let mut v = vec![f32::NAN, 280.0, f32::NAN, 270.0, f32::NAN];
        let filled = fill_column_nan(&mut v, 1, 1, 5).unwrap();
        assert_eq!(filled, 3);
        // The interior NaN forward-fills from below (280); the bottom NaN
        // backward-fills; the top NaN forward-fills from 270.
        assert_eq!(v, vec![280.0, 280.0, 280.0, 270.0, 270.0]);
        let mut all_nan = vec![f32::NAN; 3];
        assert!(fill_column_nan(&mut all_nan, 1, 1, 3).is_err());
        let mut clean = vec![1.0f32, 2.0, 3.0];
        assert_eq!(fill_column_nan(&mut clean, 1, 1, 3).unwrap(), 0);
        assert_eq!(clean, vec![1.0, 2.0, 3.0]);
    }

    // ── time / run id ──────────────────────────────────────────────────────────

    fn dt(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> NaiveDateTime {
        NaiveDate::from_ymd_opt(y, mo, d)
            .unwrap()
            .and_hms_opt(h, mi, 0)
            .unwrap()
    }

    #[test]
    fn valid_time_adds_forecast_horizon() {
        let (iso, hhmm) = valid_time(dt(2026, 7, 9, 20, 0), 0, 1).unwrap();
        assert_eq!(iso, "2026-07-09T20:00:00Z");
        assert_eq!(hhmm, 2000);
        // f06 crosses midnight into the next day.
        let (iso, hhmm) = valid_time(dt(2026, 7, 9, 20, 0), 6, 1).unwrap();
        assert_eq!(iso, "2026-07-10T02:00:00Z");
        assert_eq!(hhmm, 200);
        // Sub-hourly output uses minutes.
        let (iso, hhmm) = valid_time(dt(2026, 7, 9, 20, 0), 45, 0).unwrap();
        assert_eq!(iso, "2026-07-09T20:45:00Z");
        assert_eq!(hhmm, 2045);
        assert!(valid_time(dt(2026, 7, 9, 20, 0), 1, 7).is_err());
    }

    #[test]
    fn run_id_embeds_cycle_date() {
        // HRRR file names carry no date — the cycle (reference) date is appended
        // so two days' downloads of the same file name never collide.
        let id = default_grib_run_id(
            Path::new("hrrr.t20z.wrfnatf00.grib2"),
            dt(2026, 7, 9, 20, 0),
        );
        assert_eq!(id, "hrrr_t20z_wrfnatf00_grib2_20260709");
        let id = default_grib_run_id(
            Path::new("rrfs.t20z.natlev.3km.f000.na.grib2"),
            dt(2026, 7, 9, 20, 0),
        );
        assert_eq!(id, "rrfs_t20z_natlev_3km_f000_na_grib2_20260709");
    }

    // ── projection ─────────────────────────────────────────────────────────────

    fn hrrr_like_grid(nx: u32, ny: u32) -> GridDefinition {
        GridDefinition {
            template: 30,
            nx,
            ny,
            lat1: 21.138123,
            lon1: 237.280472,
            dx: 3000.0,
            dy: 3000.0,
            latin1: 38.5,
            latin2: 38.5,
            lov: 262.5,
            scan_mode: 0x40,
            shape_of_earth: 6,
            num_data_points: nx * ny,
            ..GridDefinition::default()
        }
    }

    #[test]
    fn lambert_params_rescale_dx_to_wrf_sphere() {
        let grid = hrrr_like_grid(7, 5);
        let p = lambert_params_from_grid(&grid, 38.5, -97.5).unwrap();
        assert_eq!(p.map_proj, 1);
        assert_eq!(p.truelat1_deg, 38.5);
        assert_eq!(p.truelat2_deg, 38.5);
        // lov 262.5 East normalizes to the WRF -97.5 convention.
        assert!((p.stand_lon_deg - (-97.5)).abs() < 1e-9);
        // dx scaled onto WRF's sphere: 3000 * 6370000 / 6371229 = 2999.4213...
        let expected = 3000.0 * optics::EARTH_RADIUS_M / 6_371_229.0;
        assert!((p.dx_m - expected).abs() < 1e-9);
        assert!((p.dx_m - 2999.4213).abs() < 1e-3);
        // An ellipsoidal shape-of-earth is refused, not guessed.
        let mut bad = hrrr_like_grid(7, 5);
        bad.shape_of_earth = 4;
        assert!(lambert_params_from_grid(&bad, 0.0, 0.0).is_err());
        // A non-Lambert template is refused here (rotated lat-lon is the RRFS stage).
        let mut rot = hrrr_like_grid(7, 5);
        rot.template = 1;
        assert!(lambert_params_from_grid(&rot, 0.0, 0.0).is_err());
    }

    #[test]
    fn grib_lambert_georef_reproduces_grid_latlon() {
        // The dx-rescale exactness proof: grib-core's own Lambert grid math
        // (sphere R = 6,371,229) generates the truth lat/lon planes; our
        // WrfProjectionParams (R = 6,370,000 + rescaled dx) anchored at the grid
        // centre must project every point back onto its own (i, j).
        let nx = 7usize;
        let ny = 5usize;
        let grid = hrrr_like_grid(nx as u32, ny as u32);
        let (xlat, xlong) = latlon_planes(&grid).unwrap();
        let c = ((ny - 1) / 2) * nx + (nx - 1) / 2;
        let params = lambert_params_from_grid(&grid, xlat[c] as f64, xlong[c] as f64).unwrap();
        let geom = GridGeometry {
            nx,
            ny,
            nz: 2,
            nz_stag: 3,
            nt: 1,
            params,
            xlat: xlat.clone(),
            xlong: xlong.clone(),
            time_iso: None,
            hhmm: 0,
        };
        let georef = geom.georef().unwrap();
        let mut worst = 0.0f64;
        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                let (fi, fj) = georef.forward(xlat[idx] as f64, xlong[idx] as f64);
                worst = worst.max((fi - i as f64).abs()).max((fj - j as f64).abs());
            }
        }
        // f32-plane rounding dominates; the projections agree to ~1e-4 cells.
        assert!(worst < 1e-2, "worst georef error {worst} cells");
    }

    #[test]
    fn latlon_planes_normalize_to_wrf_longitudes() {
        let grid = hrrr_like_grid(7, 5);
        let (_, xlong) = latlon_planes(&grid).unwrap();
        // GRIB longitudes are 0..360 East (237.28); WRF planes are +/-180.
        assert!(xlong.iter().all(|&l| (-180.0..=180.0).contains(&l)));
        assert!((xlong[0] - (237.280472 - 360.0) as f32).abs() < 1e-4);
    }

    // ── two-pass channel encoding ──────────────────────────────────────────────

    /// A tiny native volume with varying z so the resample is non-trivial.
    fn small_native() -> (Vec<f32>, Vec<f32>, usize, usize, usize) {
        let (nx, ny, nz) = (3usize, 2usize, 4usize);
        let plane = nx * ny;
        let mut z = vec![0f32; plane * nz];
        let mut f = vec![0f32; plane * nz];
        for ci in 0..plane {
            for k in 0..nz {
                let idx = k * plane + ci;
                // Terrain-following-ish: columns start at different heights.
                z[idx] = 100.0 + 60.0 * ci as f32 + 400.0 * k as f32;
                // A layer of "extinction" peaking at k = 2, zero at k = 0.
                f[idx] = match k {
                    0 => 0.0,
                    1 => 1.0e-4 * (1.0 + ci as f32),
                    2 => 8.0e-4 * (1.0 + 0.5 * ci as f32),
                    _ => 2.0e-5,
                };
            }
        }
        (f, z, nx, ny, nz)
    }

    #[test]
    fn two_pass_encode_matches_encode_log_channel() {
        // The streamed two-pass quant/encode must be BYTE-IDENTICAL to the
        // reference path (materialize the resampled f32 volume, then
        // bricks::encode_log_channel) — same LogQuant, same codes.
        let (f, z, nx, ny, nz) = small_native();
        let plane = nx * ny;
        let (z_min, dz, nz_brick) = (0.0, 250.0, 8usize);

        let (quant, codes) = resample_encode_channel(
            &f,
            &z,
            nx,
            ny,
            nz,
            z_min,
            dz,
            nz_brick,
            Extrap::Zero,
            Extrap::Zero,
            None,
        );

        // Reference: full f32 volume via the same public column kernel.
        let mut reference = vec![0f32; plane * nz_brick];
        for_each_resampled_column(
            &f,
            &z,
            nx,
            ny,
            nz,
            z_min,
            dz,
            nz_brick,
            Extrap::Zero,
            Extrap::Zero,
            |ci, col| {
                for (m, &v) in col.iter().enumerate() {
                    reference[m * plane + ci] = v as f32;
                }
            },
        );
        let (ref_quant, ref_codes) = bricks::encode_log_channel(&reference);
        assert_eq!(quant, ref_quant);
        assert_eq!(codes, ref_codes);
        assert!(
            quant.vmax > 0.0,
            "the synthetic layer must quantize non-zero"
        );
        assert!(codes.iter().any(|&c| c > 0));
    }

    #[test]
    fn beta_total_f16_accumulation_error_is_negligible() {
        // Three per-channel contributions accumulated through f16 bits stay
        // within ~2e-3 relative of the f64 sum (far below the u8 log step).
        let contributions = [3.1e-4f32, 8.7e-5, 1.9e-4];
        let mut bits = 0u16;
        let mut exact = 0.0f64;
        for &c in &contributions {
            bits = f32_to_f16_bits(f16_bits_to_f32(bits) + c);
            exact += c as f64;
        }
        let got = f16_bits_to_f32(bits) as f64;
        assert!(
            (got - exact).abs() / exact < 2.0e-3,
            "f16 drift {got} vs {exact}"
        );
    }

    #[test]
    fn tau_encoding_matches_reference_integration() {
        // encode_tau_from_beta_total == integrate each f16-decoded column with the
        // public kernel, then encode_log_channel — byte-identical.
        let (f, z, nx, ny, nz) = small_native();
        let plane = nx * ny;
        let (z_min, dz, nz_brick) = (0.0, 250.0, 8usize);
        let mut bt = vec![0u16; plane * nz_brick];
        let (_q, _codes) = resample_encode_channel(
            &f,
            &z,
            nx,
            ny,
            nz,
            z_min,
            dz,
            nz_brick,
            Extrap::Zero,
            Extrap::Zero,
            Some(&mut bt),
        );
        let (quant, codes) = encode_tau_from_beta_total(&bt, nx, ny, nz_brick, dz);

        let mut tau_f32 = vec![0f32; plane * nz_brick];
        for ci in 0..plane {
            let col: Vec<f64> = (0..nz_brick)
                .map(|m| f16_bits_to_f32(bt[m * plane + ci]) as f64)
                .collect();
            let tau = integrate_tau_up_column(&col, dz);
            for (m, &v) in tau.iter().enumerate() {
                tau_f32[m * plane + ci] = v as f32;
            }
        }
        let (ref_quant, ref_codes) = bricks::encode_log_channel(&tau_f32);
        assert_eq!(quant, ref_quant);
        assert_eq!(codes, ref_codes);
        assert!(quant.vmax > 0.0);
    }

    // ── RRFS stage: crop machinery ─────────────────────────────────────────────

    #[test]
    fn parse_crop_accepts_conus_and_bbox() {
        assert_eq!(parse_crop("conus").unwrap(), CONUS_CROP);
        assert_eq!(parse_crop("CONUS").unwrap(), CONUS_CROP);
        let c = parse_crop("25, 49.5, -110, -78").unwrap();
        assert_eq!(
            c,
            GribCrop {
                lat_min: 25.0,
                lat_max: 49.5,
                lon_min: -110.0,
                lon_max: -78.0
            }
        );
        assert!(parse_crop("nonsense").is_err());
        assert!(parse_crop("1,2,3").is_err());
        assert!(parse_crop("50,25,-110,-78").is_err()); // inverted lat
        assert!(parse_crop("25,50,170,-170").is_err()); // antimeridian straddle
    }

    #[test]
    fn crop_rect_from_planes_selects_the_bbox_hull() {
        // A 6x5 plain lat/lon ramp: lat rows 20..24, lon cols -100..-95.
        let (nx, ny) = (6usize, 5usize);
        let mut xlat = vec![0f32; nx * ny];
        let mut xlong = vec![0f32; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                xlat[j * nx + i] = 20.0 + j as f32;
                xlong[j * nx + i] = -100.0 + i as f32;
            }
        }
        let crop = GribCrop {
            lat_min: 20.5,
            lat_max: 23.2,
            lon_min: -98.4,
            lon_max: -96.0,
        };
        let rect = crop_rect_from_planes(&xlat, &xlong, nx, ny, &crop).unwrap();
        // Cells inside: lat 21..23 (j 1..=3), lon -98..-96 (i 2..=4).
        assert_eq!(
            rect,
            CropRect {
                i0: 2,
                i1: 4,
                j0: 1,
                j1: 3
            }
        );
        // A box outside the grid selects nothing.
        let off = GribCrop {
            lat_min: 60.0,
            lat_max: 70.0,
            lon_min: 0.0,
            lon_max: 10.0,
        };
        assert!(crop_rect_from_planes(&xlat, &xlong, nx, ny, &off).is_err());
        // Slicing a decoded plane to the rect keeps exactly the hull cells.
        let plane: Vec<f64> = (0..nx * ny).map(|v| v as f64).collect();
        let cropped = crop_values(plane, nx, &rect);
        assert_eq!(cropped.len(), 9);
        assert_eq!(cropped[0], (nx + 2) as f64); // (i=2, j=1)
        assert_eq!(cropped[8], (3 * nx + 4) as f64); // (i=4, j=3)
    }

    #[test]
    fn oversize_grid_without_crop_is_refused() {
        // The RRFS NA dims without a crop -> the remedy names the crop option.
        let err = require_crop_for_oversize(4881, 2961, false).unwrap_err();
        assert!(err.to_string().contains("crop=conus"), "{err}");
        assert!(require_crop_for_oversize(4881, 2961, true).is_ok());
        assert!(require_crop_for_oversize(1799, 1059, false).is_ok());
        // The peak-RSS admission: the measured-HRRR-calibrated estimate admits the
        // HRRR grid, refuses the full-CONUS hull on the tilted RRFS grid.
        assert!(require_ingest_fits_budget(1799 * 1059, 50, 80, 1799 * 1059).is_ok());
        let err = require_ingest_fits_budget(2376 * 1564, 65, 80, 4881 * 2961).unwrap_err();
        assert!(err.to_string().contains("estimated ingest peak"), "{err}");
    }

    // ── RRFS stage: rotated lat-lon projection ─────────────────────────────────

    /// The RRFS NA rotated grid definition (from the probe), scaled down.
    fn rrfs_like_grid(nx: u32, ny: u32, dinc: f64) -> GridDefinition {
        // Corner span chosen so the rotated-lon coordinates CROSS 360 (the real
        // grid runs 299 -> 421): lon1 near 360 - span/2.
        let lon_span = dinc * ((nx - 1) as f64);
        let lat_span = dinc * ((ny - 1) as f64);
        GridDefinition {
            template: 1,
            nx,
            ny,
            lat1: -lat_span / 2.0,
            lat2: lat_span / 2.0,
            lon1: 360.0 - lon_span / 2.0,
            lon2: lon_span / 2.0,
            south_pole_lat: -35.0,
            south_pole_lon: 247.0,
            rotation_angle: 0.0,
            scan_mode: 0x40,
            shape_of_earth: 6,
            num_data_points: nx * ny,
            ..GridDefinition::default()
        }
    }

    #[test]
    fn rotated_params_read_the_pole_and_increments() {
        let grid = rrfs_like_grid(9, 7, 0.5);
        let p = rotated_params_from_grid(&grid, 55.0, -113.0).unwrap();
        assert_eq!(p.map_proj, MAP_PROJ_ROTATED_LATLON);
        // North pole = the antipode of the GRIB south pole (-35, 247).
        assert!((p.truelat1_deg - 35.0).abs() < 1e-12);
        assert!((p.truelat2_deg - 67.0).abs() < 1e-12);
        // Increments from the corner span (the file stores dx = dy = 0),
        // converted to plane metres.
        assert!((p.dx_m - 0.5 * ROTATED_LATLON_M_PER_DEG).abs() < 1e-6);
        assert!((p.dy_m - 0.5 * ROTATED_LATLON_M_PER_DEG).abs() < 1e-6);
        // A nonzero rotation angle is refused, not guessed.
        let mut spun = rrfs_like_grid(9, 7, 0.5);
        spun.rotation_angle = 10.0;
        assert!(rotated_params_from_grid(&spun, 0.0, 0.0).is_err());
    }

    #[test]
    fn rotated_math_matches_grib_core_reference() {
        // Our frame.rs unproject must reproduce grib-core's own
        // rotated_to_geographic for the RRFS pole, point for point.
        let proj = crate::frame::MapProjection::rotated_latlon(35.0, 67.0);
        for &(rlat, rlon) in &[
            (0.0f64, 0.0f64),
            (-34.0, -20.0),
            (10.5, 45.0),
            (-2.25, 60.9),
            (36.9, -60.9),
        ] {
            let (glat, glon) =
                grib_core::grib2::rotated_to_geographic(rlat, rlon, -35.0, 247.0, 0.0);
            let (mlat, mlon) = proj
                .unproject(
                    rlon * ROTATED_LATLON_M_PER_DEG,
                    rlat * ROTATED_LATLON_M_PER_DEG,
                )
                .unwrap();
            let dlon = (mlon - glon + 540.0).rem_euclid(360.0) - 180.0;
            assert!(
                (mlat - glat).abs() < 1e-9 && dlon.abs() < 1e-9,
                "mismatch at rotated ({rlat}, {rlon}): grib-core ({glat}, {glon}) vs ours ({mlat}, {mlon})"
            );
        }
    }

    #[test]
    fn rotated_georef_reproduces_grid_latlon() {
        // End-to-end: grib-core's template-3.1 grid math generates the truth
        // planes (crossing rotated-lon 360 like the real RRFS grid); our params +
        // centre-anchored georef must project every point back onto its (i, j).
        let (nx, ny) = (9usize, 7usize);
        let grid = rrfs_like_grid(nx as u32, ny as u32, 0.5);
        let (xlat, xlong) = latlon_planes(&grid).unwrap();
        let c = ((ny - 1) / 2) * nx + (nx - 1) / 2;
        let params = rotated_params_from_grid(&grid, xlat[c] as f64, xlong[c] as f64).unwrap();
        let geom = GridGeometry {
            nx,
            ny,
            nz: 2,
            nz_stag: 3,
            nt: 1,
            params,
            xlat: xlat.clone(),
            xlong: xlong.clone(),
            time_iso: None,
            hhmm: 0,
        };
        let georef = geom.georef().unwrap();
        let mut worst = 0.0f64;
        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                let (fi, fj) = georef.forward(xlat[idx] as f64, xlong[idx] as f64);
                worst = worst.max((fi - i as f64).abs()).max((fj - j as f64).abs());
            }
        }
        assert!(worst < 1e-2, "worst rotated georef error {worst} cells");
    }

    #[test]
    fn manifest_projection_round_trips_the_rotated_pole() {
        // The rotated pole rides in the EXISTING truelat fields, so the manifest
        // schema is untouched and a cached RRFS run rebuilds the projection
        // bit-identically (the wrfout cached-path guarantee).
        let p = ManifestProjection {
            map_proj: MAP_PROJ_ROTATED_LATLON,
            truelat1_deg: 35.0,
            truelat2_deg: 67.0,
            stand_lon_deg: 0.0,
            cen_lat_deg: 38.7,
            cen_lon_deg: -94.2,
            dx_m: 0.025 * ROTATED_LATLON_M_PER_DEG,
            dy_m: 0.025 * ROTATED_LATLON_M_PER_DEG,
        };
        let json = serde_json::to_string(&p).unwrap();
        let back: ManifestProjection = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
        let params = WrfProjectionParams {
            map_proj: back.map_proj,
            truelat1_deg: back.truelat1_deg,
            truelat2_deg: back.truelat2_deg,
            stand_lon_deg: back.stand_lon_deg,
            cen_lat_deg: back.cen_lat_deg,
            cen_lon_deg: back.cen_lon_deg,
            dx_m: back.dx_m,
            dy_m: back.dy_m,
        };
        let direct = crate::frame::MapProjection::rotated_latlon(35.0, 67.0);
        let rebuilt = crate::frame::MapProjection::from_wrf(&params).unwrap();
        assert_eq!(direct, rebuilt);
    }
}
