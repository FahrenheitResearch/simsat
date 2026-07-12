//! Multi-timestep loop pipeline logic (design doc section 2 "Animation" + section 8
//! "prerender-then-play"). The M7 loop-rendering slice.
//!
//! This module holds the PURE, headless-testable logic behind the studio's batch
//! render + playback flow, kept out of `main.rs` (which owns the egui/GPU plumbing):
//!
//! 1. Sequence discovery + chronological ordering. A "sequence" is multiple wrfout
//!    files (each single-time in the owner's data) OR a single multi-time wrfout; both
//!    reduce to an ordered list of `(file, time-index, valid-time)` timesteps. The
//!    valid time is parsed from the wrfout filename (the `wrfout_d03_2020-01-05_01:30:00`
//!    convention) or a `Times` variable string — one parser handles both.
//! 2. Playback frame-index math: advance-with-loop-wrap, scrub-clamp, and the
//!    fps -> frames-to-advance timing. Pure functions so `cargo test` on the headless
//!    nodes covers the wrap/clamp/timing behavior with no egui or GPU.
//! 3. The WS4 studio-UX pure logic: the scene-cache key types + single-slot hit
//!    decision ([`CacheSlot`]), the [`LogBuffer`] ring with sticky-error semantics,
//!    the PNG-export filename builder, the drag-and-drop classifier, and the global
//!    rayon pool sizing. All engine-free plain data, node-tested.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use simsat::camera::{MAX_AXIS, PerspectiveCamera};

/// A calendar valid-time parsed from a wrfout filename or a `Times` string. Field
/// order (year..second) makes the derived `Ord` chronological, so a `Vec<SeqItem>`
/// sorts into time order directly.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct ValidTime {
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
}

impl ValidTime {
    /// HHMM (the sat-store frame key convention, `hour*100 + minute`).
    pub fn hhmm(&self) -> u16 {
        (self.hour * 100 + self.minute) as u16
    }

    /// RFC3339 UTC string (`2020-01-05T01:30:00Z`) for the solar/store path.
    pub fn iso_utc(&self) -> String {
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
    }

    /// A short human label for the picker/timeline (`2020-01-05 01:30 UTC`).
    pub fn label(&self) -> String {
        format!(
            "{:04}-{:02}-{:02} {:02}:{:02} UTC",
            self.year, self.month, self.day, self.hour, self.minute
        )
    }
}

fn is_datetime_sep(c: u8) -> bool {
    // Between the date and the time: wrfout filenames use `_`, ISO uses `T`, some
    // tools use a space.
    c == b'_' || c == b'T' || c == b' '
}

fn is_time_sep(c: u8) -> bool {
    // Between HH/MM/SS: real (Linux) wrfout uses `:`; a Windows-safe copy may use `_`
    // or `-`.
    c == b':' || c == b'_' || c == b'-'
}

fn parse_uint(b: &[u8], start: usize, len: usize) -> Option<u32> {
    let mut v: u32 = 0;
    for &c in b.get(start..start + len)? {
        if !c.is_ascii_digit() {
            return None;
        }
        v = v * 10 + (c - b'0') as u32;
    }
    Some(v)
}

/// Parse the FIRST `YYYY-MM-DD<sep>HH<sep>MM<sep>SS` datetime embedded anywhere in
/// `s`, returning its `ValidTime`. Handles a wrfout filename
/// (`wrfout_d03_2020-01-05_01:30:00`), a `Times` variable string
/// (`2020-01-05_01:30:00`), and an ISO string (`2020-01-05T01:30:00Z`) with one
/// parser — the date separators are fixed `-`, the datetime and intra-time separators
/// accept the several conventions above.
pub fn parse_valid_time(s: &str) -> Option<ValidTime> {
    let b = s.as_bytes();
    let n = b.len();
    // The pattern is 19 characters: yyyy-mm-dd?hh?mm?ss.
    if n < 19 {
        return None;
    }
    for i in 0..=(n - 19) {
        let d = |off: usize| b[i + off].is_ascii_digit();
        let ok = d(0)
            && d(1)
            && d(2)
            && d(3)
            && b[i + 4] == b'-'
            && d(5)
            && d(6)
            && b[i + 7] == b'-'
            && d(8)
            && d(9)
            && is_datetime_sep(b[i + 10])
            && d(11)
            && d(12)
            && is_time_sep(b[i + 13])
            && d(14)
            && d(15)
            && is_time_sep(b[i + 16])
            && d(17)
            && d(18);
        if !ok {
            continue;
        }
        let year = parse_uint(b, i, 4)?;
        let month = parse_uint(b, i + 5, 2)?;
        let day = parse_uint(b, i + 8, 2)?;
        let hour = parse_uint(b, i + 11, 2)?;
        let minute = parse_uint(b, i + 14, 2)?;
        let second = parse_uint(b, i + 17, 2)?;
        // Reject an obviously-invalid calendar field (keeps a random digit run from
        // parsing as a time); ranges are generous, not a full calendar validation.
        if !(1..=12).contains(&month)
            || !(1..=31).contains(&day)
            || hour > 23
            || minute > 59
            || second > 59
        {
            continue;
        }
        return Some(ValidTime {
            year: year as i32,
            month,
            day,
            hour,
            minute,
            second,
        });
    }
    None
}

/// One discovered input file's identity + the times it holds. `times` are `Times`
/// variable strings from a cheap probe (empty if the probe returned none); `name` is
/// the file name for the filename-time fallback + labels.
#[derive(Clone, Debug)]
pub struct FileTimes {
    pub name: String,
    pub times: Vec<String>,
}

/// One ordered timestep of a sequence: which input file (`file_index`), which time
/// index within it (`ts_index`, the ingest `timestep` parameter), and its valid time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeqItem {
    pub file_index: usize,
    pub ts_index: usize,
    pub valid: ValidTime,
    pub label: String,
}

/// Build the chronologically-ordered sequence from a set of input files. Each file
/// contributes one timestep per `Times` entry (a multi-time wrfout expands to N), or
/// one timestep at its filename time when the probe gave no `Times`. A time that
/// parses from neither the `Times` string nor the filename is skipped (a non-wrfout
/// file discovered in a directory). Ties break by `(file_index, ts_index)` so the
/// order is deterministic.
pub fn build_sequence(files: &[FileTimes]) -> Vec<SeqItem> {
    let mut items: Vec<SeqItem> = Vec::new();
    for (fi, f) in files.iter().enumerate() {
        let fname_time = parse_valid_time(&f.name);
        if f.times.is_empty() {
            if let Some(v) = fname_time {
                items.push(SeqItem {
                    file_index: fi,
                    ts_index: 0,
                    valid: v,
                    label: v.label(),
                });
            }
        } else {
            for (ti, t) in f.times.iter().enumerate() {
                // Prefer the per-time `Times` string; fall back to the filename time
                // (a single-time file whose `Times` was unreadable but whose name has
                // the valid time).
                if let Some(v) = parse_valid_time(t).or(fname_time) {
                    items.push(SeqItem {
                        file_index: fi,
                        ts_index: ti,
                        valid: v,
                        label: v.label(),
                    });
                }
            }
        }
    }
    items.sort_by(|a, b| {
        a.valid
            .cmp(&b.valid)
            .then(a.file_index.cmp(&b.file_index))
            .then(a.ts_index.cmp(&b.ts_index))
    });
    items
}

// ── playback frame-index math (pure; unit-tested) ─────────────────────────────

/// Clamp a scrub target (which may be produced from a signed drag) to a valid frame
/// index in `[0, n-1]`; an empty timeline clamps to 0.
pub fn clamp_scrub(target: i64, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    target.clamp(0, n as i64 - 1) as usize
}

/// Advance the play head `current` by `steps` frames over an `n`-frame timeline.
/// Returns `(next_index, stopped)`. Looping wraps modulo `n` and never stops; a
/// non-looping timeline clamps at the last frame and reports `stopped = true` once it
/// reaches (or would pass) the end so the caller can pause.
pub fn advance_index(current: usize, steps: u32, n: usize, looping: bool) -> (usize, bool) {
    if n == 0 {
        return (0, true);
    }
    if steps == 0 {
        return (current.min(n - 1), false);
    }
    if looping {
        ((current + steps as usize) % n, false)
    } else {
        let target = current + steps as usize;
        if target >= n - 1 {
            (n - 1, true)
        } else {
            (target, false)
        }
    }
}

/// Convert wall-clock elapsed `dt` seconds at `fps` into a whole number of frames to
/// advance plus the leftover accumulator (carried to the next tick so the average rate
/// is exactly `fps`). A non-positive `fps` advances nothing. The per-tick step is
/// capped so a long UI stall (e.g. after a heavy render) cannot advance hundreds of
/// frames at once — playback resumes smoothly instead of jumping.
pub fn fps_frame_step(accumulator: f32, dt: f32, fps: f32) -> (u32, f32) {
    if fps <= 0.0 || !dt.is_finite() {
        return (0, 0.0);
    }
    let interval = 1.0 / fps;
    let mut acc = accumulator + dt.max(0.0);
    let mut frames = 0u32;
    const MAX_STEP: u32 = 8;
    while acc >= interval && frames < MAX_STEP {
        acc -= interval;
        frames += 1;
    }
    // If the cap clipped a huge backlog, drop the residual so we do not keep sprinting.
    if acc >= interval {
        acc = 0.0;
    }
    (frames, acc)
}

// ── scene-cache keys + the single-slot hit decision (WS4 item 1) ──────────────
//
// The studio's prepare worker caches the expensive, timestep-INDEPENDENT scene
// resources (Blue Marble crop, atmosphere LUTs + SH table, horizon map, output
// raster + geo LUT) across the frames of a sequence render. Each resource kind
// gets ONE slot: a `(key, value)` pair where a lookup HITS only when the new key
// is EQUAL to the stored one, and any mismatch REPLACES the entry (so memory
// stays bounded at one live entry per kind, and a stale artifact can never leak
// into a frame — the key carries every parameter the resource depends on;
// see the key structs below). The keys are plain data (float parameters are
// carried as raw bits so NaN/-0.0 never make "equal" ambiguous); `main.rs`
// constructs them from the engine types.

/// A single-entry cache: HIT only on key equality, any other key replaces.
pub struct CacheSlot<K: PartialEq, V> {
    entry: Option<(K, Arc<V>)>,
}

impl<K: PartialEq, V> Default for CacheSlot<K, V> {
    fn default() -> Self {
        Self { entry: None }
    }
}

impl<K: PartialEq, V> CacheSlot<K, V> {
    /// Whether a lookup with `key` would hit (the pure hit decision).
    pub fn matches(&self, key: &K) -> bool {
        self.entry.as_ref().is_some_and(|(k, _)| k == key)
    }

    /// The cached value if `key` hits.
    pub fn get(&self, key: &K) -> Option<Arc<V>> {
        if !self.matches(key) {
            return None;
        }
        self.entry.as_ref().map(|(_, v)| v.clone())
    }

    /// Store `value` under `key` (replacing any previous entry) and return it.
    pub fn put(&mut self, key: K, value: V) -> Arc<V> {
        let v = Arc::new(value);
        self.entry = Some((key, v.clone()));
        v
    }

    /// Get the cached value for `key`, or build + store it. Returns `(value, hit)`.
    pub fn get_or_insert_with(&mut self, key: K, build: impl FnOnce() -> V) -> (Arc<V>, bool) {
        if let Some(v) = self.get(&key) {
            return (v, true);
        }
        (self.put(key, build()), false)
    }

    /// Fallible [`Self::get_or_insert_with`]: a build error caches nothing.
    pub fn get_or_try_insert_with<E>(
        &mut self,
        key: K,
        build: impl FnOnce() -> Result<V, E>,
    ) -> Result<(Arc<V>, bool), E> {
        if let Some(v) = self.get(&key) {
            return Ok((v, true));
        }
        Ok((self.put(key, build()?), false))
    }
}

/// Key for the season-blended Blue Marble crop (wraps `asset_pack::load_season_ground`).
/// Carries the resolved month blend (the pure function of date/override), the manual
/// override + download policy (they change WHICH asset is resolved), and the crop
/// request itself (domain bbox bits + max dimension). Any change re-crops.
#[derive(Debug, Clone, PartialEq)]
pub struct BmCacheKey {
    pub month_a: u32,
    pub month_b: u32,
    /// `weight_b` blend weight as raw f32 bits.
    pub weight_b_bits: u32,
    pub month_override: Option<u32>,
    pub allow_download: bool,
    /// Domain bbox `(lat_min, lat_max, lon_min, lon_max)` as raw f32 bits.
    pub bbox_bits: [u32; 4],
    pub max_dim: u32,
}

/// Key for the atmosphere LUT set (`AtmosphereLuts` + `SkyShTable`): every
/// `AtmosphereParams` field as raw f64 bits + the SH table entry count. `pw_ratio`
/// comes from the BRICK (per timestep), so two timesteps hit only when their
/// domain-mean precipitable water is bit-identical — a stale LUT can never
/// silently recolour a frame.
#[derive(Debug, Clone, PartialEq)]
pub struct AtmoLutKey {
    pub aod_bits: u64,
    pub pw_ratio_bits: u64,
    pub swelling_bits: u64,
    pub ground_albedo_bits: u64,
    pub sh_entries: usize,
}

/// Key for the M3 horizon map (terrain-only, sun-independent): the run + grid
/// identity. HGT is static across a run's timesteps, so a sequence builds it once.
#[derive(Debug, Clone, PartialEq)]
pub struct HorizonCacheKey {
    pub run_id: String,
    pub nx: usize,
    pub ny: usize,
    /// Cell size (m) as raw f64 bits.
    pub dx_bits: u64,
    pub dy_bits: u64,
}

/// Key for the output raster (generic over the engine's `GridGeoref`, kept out of
/// this engine-free module): the georef + domain dims + every camera parameter the
/// raster depends on. NOTE: the brief named (georef, resolution, margin, view); the
/// SATELLITE preset is added because the geostationary raster is built from the
/// satellite's camera — switching GOES-E -> GOES-W must miss.
#[derive(Debug, Clone, PartialEq)]
pub struct RasterCacheKey<G: PartialEq> {
    pub georef: G,
    pub nx: usize,
    pub ny: usize,
    /// `ResolutionMode` ordinal (0 = Native, 1 = ABI 1 km, 2 = ABI 2 km).
    pub resolution: u8,
    /// Zoom-out margin fraction as raw f64 bits.
    pub margin_bits: u64,
    /// `ViewMode` ordinal (0 = Geostationary, 1 = Top-down map).
    pub view: u8,
    /// `SatellitePreset` ordinal (0 = GOES-E, 1 = GOES-W, 2 = Himawari).
    pub sat: u8,
    /// `GeoNavigation` ordinal (0 = model sphere, 1 = GOES-R ABI ellipsoid).
    pub navigation: u8,
}

/// Key for the per-pixel GEO lookup texture: the raster it was built over plus the
/// Blue Marble crop's geographic bounds (the geo LUT bakes the BM UV mapping), or
/// `None` for the flat-albedo / no-ground case. The LIGHT half is rebuilt per
/// timestep (solar geometry changes every frame) and is not cached.
#[derive(Debug, Clone, PartialEq)]
pub struct GeoLutKey<G: PartialEq> {
    pub raster: RasterCacheKey<G>,
    /// BM crop `(lon_min, lon_max, lat_min, lat_max)` as raw f32 bits, if present.
    pub bm_bounds_bits: Option<[u32; 4]>,
}

// ── error + progress surfacing (WS4 item 3) ───────────────────────────────────

/// Log severity for the studio's in-app log ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Info,
    Error,
}

/// One log line.
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub level: LogLevel,
    pub message: String,
}

/// A bounded in-app log: a VecDeque ring (oldest evicted) + STICKY last-error
/// semantics — `last_error` is set by every `error` push and cleared ONLY by an
/// explicit dismiss or a subsequent successful render (`note_render_success`),
/// never by a later info/status line. That keeps a failure visible in the banner
/// until the owner acts or the app demonstrably recovers.
pub struct LogBuffer {
    entries: VecDeque<LogEntry>,
    cap: usize,
    last_error: Option<String>,
}

impl LogBuffer {
    pub fn new(cap: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            cap: cap.max(1),
            last_error: None,
        }
    }

    fn push(&mut self, level: LogLevel, message: String) {
        if self.entries.len() >= self.cap {
            self.entries.pop_front();
        }
        self.entries.push_back(LogEntry { level, message });
    }

    /// Append an info/progress line (does NOT touch the sticky error).
    pub fn info(&mut self, message: impl Into<String>) {
        self.push(LogLevel::Info, message.into());
    }

    /// Append an error line and make it the sticky banner error.
    pub fn error(&mut self, message: impl Into<String>) {
        let message = message.into();
        self.last_error = Some(message.clone());
        self.push(LogLevel::Error, message);
    }

    /// The sticky error shown in the banner, if any.
    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    /// Explicitly dismiss the sticky error (the banner's Dismiss button).
    pub fn dismiss_error(&mut self) {
        self.last_error = None;
    }

    /// A render completed successfully — the one non-explicit way the sticky
    /// error clears (the app demonstrably recovered).
    pub fn note_render_success(&mut self) {
        self.last_error = None;
    }

    /// Oldest-to-newest iteration for the log view.
    pub fn entries(&self) -> impl Iterator<Item = &LogEntry> {
        self.entries.iter()
    }
}

// ── global rayon pool sizing (WS4 item 6) ─────────────────────────────────────

/// Worker-thread count for the global rayon pool: all cores MINUS ONE spare (the
/// machine-stability discipline — the owner's box has hard-crashed under all-core
/// load; one core stays free for the UI/desktop), floored at 1.
pub fn pool_threads_leaving_spare(available: usize) -> usize {
    available.saturating_sub(1).max(1)
}

// ── PNG-export filename (WS4 item 4) ──────────────────────────────────────────

/// Sanitize one filename token: lowercase, `[a-z0-9._-]` kept, every other run of
/// characters collapsed to a single `-`, trimmed; empty falls back to `frame`.
fn sanitize_token(token: &str) -> String {
    let mut out = String::with_capacity(token.len());
    let mut last_dash = false;
    for c in token.chars() {
        let c = c.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
            out.push(c);
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    let trimmed = out.trim_matches('-').to_string();
    if trimmed.is_empty() {
        "frame".to_string()
    } else {
        trimmed
    }
}

/// Default file name for the "Save PNG..." export:
/// `{scene}_{product}_{view}_{yyyymmdd}_{hhmm}[{_whatif-sun}].png`, every free-text
/// token sanitized. The what-if suffix marks a fake-sun frame in the file name
/// itself so a non-physical render can never masquerade as the real view.
#[allow(clippy::too_many_arguments)]
pub fn build_export_filename(
    scene: &str,
    product: &str,
    view: &str,
    year: i32,
    month: u32,
    day: u32,
    hhmm: u16,
    sun_override: bool,
) -> String {
    format!(
        "{}_{}_{}_{:04}{:02}{:02}_{:04}{}.png",
        sanitize_token(scene),
        sanitize_token(product),
        sanitize_token(view),
        year,
        month,
        day,
        hhmm,
        if sun_override { "_whatif-sun" } else { "" },
    )
}

/// RGBA -> RGB for the PNG export, with SPACE-BLACK conversion: a fully
/// transparent pixel (alpha 0 — off-earth space in the rendered frame) exports as
/// pure black instead of leaking whatever RGB the renderer left behind.
pub fn rgba_to_rgb_space_black(rgba: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgba.len() / 4 * 3);
    for px in rgba.chunks_exact(4) {
        if px[3] == 0 {
            out.extend_from_slice(&[0, 0, 0]);
        } else {
            out.extend_from_slice(&[px[0], px[1], px[2]]);
        }
    }
    out
}

// ── drag-and-drop classification (WS4 item 5) ─────────────────────────────────

/// What a set of dropped paths opens as.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DropOpen {
    /// A single non-JSON file: open as a wrfout.
    Wrfout(PathBuf),
    /// A single `.json` file: open as a cached `run.json`.
    CachedRun(PathBuf),
    /// A directory (expanded by the sequence opener) or a multi-file drop.
    Sequence(Vec<PathBuf>),
}

/// Classify a drag-and-drop: a directory -> a sequence (the opener expands it),
/// a single `.json` -> a cached run, a single other file -> a wrfout, multiple
/// paths -> a sequence. `is_dir` is injected for testability (no filesystem in
/// tests). An empty (path-less) drop returns `None`. Busy-rejection is the CALL
/// site's job (drops are ignored while a render is in flight).
pub fn classify_dropped(paths: &[PathBuf], is_dir: &dyn Fn(&Path) -> bool) -> Option<DropOpen> {
    match paths {
        [] => None,
        [p] => {
            if is_dir(p) {
                Some(DropOpen::Sequence(vec![p.clone()]))
            } else if p
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("json"))
            {
                Some(DropOpen::CachedRun(p.clone()))
            } else {
                Some(DropOpen::Wrfout(p.clone()))
            }
        }
        many => Some(DropOpen::Sequence(many.to_vec())),
    }
}

// ── perspective orbit camera (studio Perspective (3-D) view) ──────────────────

/// The studio's orbit parameterization of the engine's free perspective camera:
/// instead of raw eye coordinates, the owner drags an orbit AROUND THE DOMAIN
/// CENTRE — azimuth (compass direction the camera sits FROM the centre), tilt
/// (elevation above the horizontal, seen from the centre), slant range from the
/// centre, horizontal FOV, and the output dims. Pure data; the mapping to a
/// [`PerspectiveCamera`] is [`orbit_to_camera`], node-tested below.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitParams {
    /// Compass bearing FROM the domain centre TO the camera position (deg;
    /// 0 = the camera sits north of the centre, 180 = south of it looking north).
    pub az_deg: f64,
    /// Camera elevation above the horizontal as seen from the centre (deg;
    /// clamped [`ORBIT_TILT_MIN_DEG`]..[`ORBIT_TILT_MAX_DEG`] — near-90 is
    /// overhead, small values are a low oblique).
    pub tilt_deg: f64,
    /// Slant range from the domain centre to the eye (km); clamped at render to
    /// [`orbit_range_bounds_m`] of the domain diagonal.
    pub range_km: f64,
    /// Horizontal field of view (deg); clamped 15..120 (the engine allows 1..160).
    pub fov_deg: f64,
    /// Output image dims (px); each clamped 2..=`MAX_AXIS`.
    pub width: usize,
    pub height: usize,
}

pub const ORBIT_TILT_MIN_DEG: f64 = 5.0;
pub const ORBIT_TILT_MAX_DEG: f64 = 85.0;
pub const ORBIT_FOV_MIN_DEG: f64 = 15.0;
pub const ORBIT_FOV_MAX_DEG: f64 = 120.0;

/// The engine's spherical-earth radius convention (owner decision 5; matches
/// `camera.rs`/`frame.rs` R = 6.37e6).
const EARTH_RADIUS_M: f64 = 6.37e6;

/// Sane orbit-range clamps for a domain of diagonal `domain_diag_m`: 0.3x..5x the
/// diagonal (a tiny-domain floor keeps the bounds positive and ordered).
pub fn orbit_range_bounds_m(domain_diag_m: f64) -> (f64, f64) {
    let d = if domain_diag_m.is_finite() {
        domain_diag_m.max(1_000.0)
    } else {
        1_000.0
    };
    (0.3 * d, 5.0 * d)
}

/// Map an orbit around the domain centre to the engine's free perspective camera.
///
/// Geometry (spherical earth, R = 6.37e6): the slant range splits into an eye
/// altitude `range*sin(tilt)` and a ground offset `range*cos(tilt)`; the ground
/// offset is walked as a GREAT-CIRCLE arc from the centre along the azimuth
/// bearing (standard destination-point formula), so the eye sits `range*sin(tilt)`
/// above the sphere at that point. The look-at is the domain centre at ground
/// level (h = 0). At tilt near 90 the eye is (nearly) overhead; azimuth 180 puts
/// the camera south of the centre looking north. All parameters are clamped here
/// (range to the domain-derived bounds, tilt/fov/dims to the documented ranges),
/// so the returned camera always passes the engine's `validate()`.
pub fn orbit_to_camera(
    o: &OrbitParams,
    center_lat_deg: f64,
    center_lon_deg: f64,
    domain_diag_m: f64,
) -> PerspectiveCamera {
    let (lo, hi) = orbit_range_bounds_m(domain_diag_m);
    let range_m = if (o.range_km * 1000.0).is_finite() {
        (o.range_km * 1000.0).clamp(lo, hi)
    } else {
        hi.min(lo.max(300_000.0))
    };
    let tilt = o
        .tilt_deg
        .clamp(ORBIT_TILT_MIN_DEG, ORBIT_TILT_MAX_DEG)
        .to_radians();
    let az = o.az_deg.rem_euclid(360.0).to_radians();
    let fov_deg = o.fov_deg.clamp(ORBIT_FOV_MIN_DEG, ORBIT_FOV_MAX_DEG);
    let alt_m = range_m * tilt.sin();
    let ground_m = range_m * tilt.cos();
    // Great-circle destination point: from the centre, bearing `az`, arc `ground_m`.
    let (lat1, lon1) = (center_lat_deg.to_radians(), center_lon_deg.to_radians());
    let ang = ground_m / EARTH_RADIUS_M;
    let lat2 = (lat1.sin() * ang.cos() + lat1.cos() * ang.sin() * az.cos()).asin();
    let lon2 =
        lon1 + (az.sin() * ang.sin() * lat1.cos()).atan2(ang.cos() - lat1.sin() * lat2.sin());
    PerspectiveCamera {
        eye_lat_deg: lat2.to_degrees(),
        eye_lon_deg: (lon2.to_degrees() + 540.0).rem_euclid(360.0) - 180.0,
        eye_alt_m: alt_m,
        look_lat_deg: center_lat_deg,
        look_lon_deg: center_lon_deg,
        look_alt_m: 0.0,
        fov_deg,
        width: o.width.clamp(2, MAX_AXIS),
        height: o.height.clamp(2, MAX_AXIS),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_wrfout_filename_convention() {
        let v = parse_valid_time("wrfout_d03_2020-01-05_01:30:00").unwrap();
        assert_eq!(
            v,
            ValidTime {
                year: 2020,
                month: 1,
                day: 5,
                hour: 1,
                minute: 30,
                second: 0
            }
        );
        assert_eq!(v.hhmm(), 130);
        assert_eq!(v.iso_utc(), "2020-01-05T01:30:00Z");
    }

    #[test]
    fn parses_times_var_iso_and_windows_safe_separators() {
        // A `Times` variable string.
        assert_eq!(
            parse_valid_time("2018-10-10_12:15:00").unwrap().hhmm(),
            1215
        );
        // An ISO string with the `T` datetime separator + trailing `Z`.
        assert_eq!(
            parse_valid_time("2018-10-10T12:15:00Z").unwrap().hhmm(),
            1215
        );
        // A Windows-safe copy using `_` between the time fields.
        assert_eq!(
            parse_valid_time("wrfout_d01_2021-07-04_18_45_00")
                .unwrap()
                .hhmm(),
            1845
        );
        // No datetime present.
        assert!(parse_valid_time("not_a_wrfout_file.txt").is_none());
        // A near-miss (out-of-range month) is rejected.
        assert!(parse_valid_time("2020-13-05_01:30:00").is_none());
    }

    #[test]
    fn build_sequence_orders_the_enderlin_naming() {
        // The Enderlin folder: single-time files across 01:30..02:30 and 05:00..06:00,
        // supplied OUT of order (as a directory listing might be) — must come back
        // strictly chronological.
        let names = [
            "wrfout_d03_2020-01-05_05:30:00",
            "wrfout_d03_2020-01-05_01:30:00",
            "wrfout_d03_2020-01-05_06:00:00",
            "wrfout_d03_2020-01-05_02:00:00",
            "wrfout_d03_2020-01-05_05:00:00",
            "wrfout_d03_2020-01-05_02:30:00",
        ];
        let files: Vec<FileTimes> = names
            .iter()
            .map(|n| FileTimes {
                name: n.to_string(),
                times: vec![],
            })
            .collect();
        let seq = build_sequence(&files);
        let hhmm: Vec<u16> = seq.iter().map(|s| s.valid.hhmm()).collect();
        assert_eq!(hhmm, vec![130, 200, 230, 500, 530, 600]);
        // Each item still points back at its own file (index into the input list).
        assert_eq!(seq[0].file_index, 1); // the 01:30 file was input #1
        assert_eq!(seq[5].file_index, 2); // the 06:00 file was input #2
        assert!(seq.iter().all(|s| s.ts_index == 0));
    }

    #[test]
    fn build_sequence_expands_a_multi_time_wrfout() {
        // A single file with three times (the multi-time wrfout case) expands to three
        // timesteps, one per time index, in order.
        let files = vec![FileTimes {
            name: "wrfout_d01_2020-06-21_00:00:00".to_string(),
            times: vec![
                "2020-06-21_00:00:00".to_string(),
                "2020-06-21_01:00:00".to_string(),
                "2020-06-21_02:00:00".to_string(),
            ],
        }];
        let seq = build_sequence(&files);
        assert_eq!(seq.len(), 3);
        assert_eq!(
            seq.iter().map(|s| s.ts_index).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(
            seq.iter().map(|s| s.valid.hhmm()).collect::<Vec<_>>(),
            vec![0, 100, 200]
        );
        assert!(seq.iter().all(|s| s.file_index == 0));
        // A non-wrfout entry mixed in is dropped.
        let mixed = vec![
            FileTimes {
                name: "README.md".to_string(),
                times: vec![],
            },
            FileTimes {
                name: "wrfout_d01_2020-06-21_03:00:00".to_string(),
                times: vec![],
            },
        ];
        let seq2 = build_sequence(&mixed);
        assert_eq!(seq2.len(), 1);
        assert_eq!(seq2[0].file_index, 1);
    }

    #[test]
    fn scrub_clamps_into_range() {
        assert_eq!(clamp_scrub(-5, 10), 0);
        assert_eq!(clamp_scrub(3, 10), 3);
        assert_eq!(clamp_scrub(99, 10), 9);
        assert_eq!(clamp_scrub(0, 0), 0); // empty timeline
    }

    #[test]
    fn advance_wraps_when_looping_and_stops_at_end_otherwise() {
        // Looping wraps modulo n and never stops.
        assert_eq!(advance_index(8, 1, 10, true), (9, false));
        assert_eq!(advance_index(9, 1, 10, true), (0, false));
        assert_eq!(advance_index(9, 3, 10, true), (2, false));
        // Non-looping clamps at the last frame and reports stopped.
        assert_eq!(advance_index(8, 1, 10, false), (9, true));
        assert_eq!(advance_index(9, 1, 10, false), (9, true));
        assert_eq!(advance_index(3, 2, 10, false), (5, false));
        // Zero steps holds position; empty timeline is a no-op stop.
        assert_eq!(advance_index(4, 0, 10, false), (4, false));
        assert_eq!(advance_index(0, 1, 0, true), (0, true));
    }

    #[test]
    fn fps_timing_accumulates_to_the_exact_rate() {
        // At 10 fps the interval is 0.1 s. Three 0.1 s ticks advance one frame each.
        let mut acc = 0.0;
        let mut total = 0u32;
        for _ in 0..3 {
            let (f, a) = fps_frame_step(acc, 0.1, 10.0);
            total += f;
            acc = a;
        }
        assert_eq!(total, 3);

        // A single tick shorter than the interval advances nothing but banks the time,
        // and two such ticks then release exactly one frame.
        let (f0, a0) = fps_frame_step(0.0, 0.06, 10.0);
        assert_eq!(f0, 0);
        let (f1, a1) = fps_frame_step(a0, 0.06, 10.0);
        assert_eq!(f1, 1);
        assert!(a1 < 0.1, "leftover carried, not a whole interval");

        // A long stall is capped (no hundreds-of-frames jump) and the residual dropped.
        let (f, a) = fps_frame_step(0.0, 100.0, 10.0);
        assert!(f <= 8, "per-tick step is capped, got {f}");
        assert_eq!(a, 0.0, "residual backlog dropped");

        // Non-positive fps advances nothing.
        assert_eq!(fps_frame_step(0.5, 0.1, 0.0), (0, 0.0));
    }

    // ── WS4: scene-cache slots + keys ─────────────────────────────────────────

    #[test]
    fn cache_slot_hits_only_on_equal_key() {
        let mut slot: CacheSlot<u32, String> = CacheSlot::default();
        assert!(!slot.matches(&1));
        let (v1, hit1) = slot.get_or_insert_with(1, || "one".to_string());
        assert!(!hit1, "first build is a miss");
        // Same key: HIT, and it is the SAME cached allocation (freshly-built equality
        // is what the key guarantees; identity proves nothing was rebuilt).
        let (v1b, hit2) = slot.get_or_insert_with(1, || unreachable!("must not rebuild"));
        assert!(hit2);
        assert!(Arc::ptr_eq(&v1, &v1b));
        assert_eq!(*v1b, "one");
        // A different key MISSES and replaces the single slot entry.
        let (v2, hit3) = slot.get_or_insert_with(2, || "two".to_string());
        assert!(!hit3);
        assert_eq!(*v2, "two");
        assert!(!slot.matches(&1), "single slot: old entry evicted");
        // The fallible path: an error caches nothing.
        let mut fslot: CacheSlot<u32, u32> = CacheSlot::default();
        let err: Result<_, String> = fslot.get_or_try_insert_with(7, || Err("boom".to_string()));
        assert!(err.is_err());
        assert!(!fslot.matches(&7), "failed build must not be cached");
        let ok = fslot
            .get_or_try_insert_with(7, || Ok::<_, String>(42))
            .unwrap();
        assert_eq!((*ok.0, ok.1), (42, false));
    }

    /// The explicit stale-artifact guard from the brief: a changed `pw_ratio` (or any
    /// other atmosphere parameter) MUST miss — the LUTs recolour the whole frame.
    #[test]
    fn atmo_key_pw_ratio_change_misses() {
        let key = |pw: f64, aod: f64| AtmoLutKey {
            aod_bits: aod.to_bits(),
            pw_ratio_bits: pw.to_bits(),
            swelling_bits: 1.0f64.to_bits(),
            ground_albedo_bits: 0.3f64.to_bits(),
            sh_entries: 48,
        };
        let mut slot: CacheSlot<AtmoLutKey, u32> = CacheSlot::default();
        slot.put(key(1.0, 0.05), 1);
        assert!(slot.matches(&key(1.0, 0.05)));
        // pw_ratio moved by one brick's moisture -> MISS.
        assert!(!slot.matches(&key(1.0000001, 0.05)));
        // AOD slider moved -> MISS.
        assert!(!slot.matches(&key(1.0, 0.06)));
    }

    /// The stale-ground guard: a changed month override (or blend weight / bbox /
    /// download policy) MUST re-crop the Blue Marble.
    #[test]
    fn bm_key_override_or_blend_change_misses() {
        let key = |ov: Option<u32>, w: f32, dl: bool| BmCacheKey {
            month_a: 6,
            month_b: 7,
            weight_b_bits: w.to_bits(),
            month_override: ov,
            allow_download: dl,
            bbox_bits: [
                30.0f32.to_bits(),
                45.0f32.to_bits(),
                (-110.0f32).to_bits(),
                (-90.0f32).to_bits(),
            ],
            max_dim: 4096,
        };
        let mut slot: CacheSlot<BmCacheKey, u32> = CacheSlot::default();
        slot.put(key(None, 0.18, true), 1);
        assert!(slot.matches(&key(None, 0.18, true)));
        assert!(!slot.matches(&key(Some(12), 0.18, true)), "override misses");
        assert!(!slot.matches(&key(None, 0.19, true)), "weight misses");
        assert!(!slot.matches(&key(None, 0.18, false)), "download misses");
        let mut moved = key(None, 0.18, true);
        moved.bbox_bits[0] = 31.0f32.to_bits();
        assert!(!slot.matches(&moved), "bbox misses");
    }

    /// The stale-raster guard: margin / resolution / view / satellite / navigation each miss.
    #[test]
    fn raster_key_margin_resolution_view_sat_change_misses() {
        let key = |res: u8, margin: f64, view: u8, sat: u8, navigation: u8| RasterCacheKey {
            georef: 77u32, // stand-in for the engine GridGeoref (generic)
            nx: 800,
            ny: 800,
            resolution: res,
            margin_bits: margin.to_bits(),
            view,
            sat,
            navigation,
        };
        let mut slot: CacheSlot<RasterCacheKey<u32>, u32> = CacheSlot::default();
        slot.put(key(0, 0.0, 0, 0, 0), 1);
        assert!(slot.matches(&key(0, 0.0, 0, 0, 0)));
        assert!(!slot.matches(&key(1, 0.0, 0, 0, 0)), "resolution misses");
        assert!(!slot.matches(&key(0, 0.3, 0, 0, 0)), "margin misses");
        assert!(!slot.matches(&key(0, 0.0, 1, 0, 0)), "view misses");
        assert!(!slot.matches(&key(0, 0.0, 0, 1, 0)), "satellite misses");
        assert!(!slot.matches(&key(0, 0.0, 0, 0, 1)), "navigation misses");
        // The geo LUT key additionally misses when the BM crop bounds change.
        let geo = |bm: Option<[u32; 4]>| GeoLutKey {
            raster: key(0, 0.0, 0, 0, 0),
            bm_bounds_bits: bm,
        };
        let mut gslot: CacheSlot<GeoLutKey<u32>, u32> = CacheSlot::default();
        gslot.put(geo(Some([1, 2, 3, 4])), 1);
        assert!(gslot.matches(&geo(Some([1, 2, 3, 4]))));
        assert!(!gslot.matches(&geo(Some([1, 2, 3, 5]))));
        assert!(!gslot.matches(&geo(None)));
    }

    // ── WS4: log buffer ───────────────────────────────────────────────────────

    #[test]
    fn log_ring_caps_and_preserves_order() {
        let mut log = LogBuffer::new(3);
        assert_eq!(log.entries().count(), 0);
        log.info("a");
        log.info("b");
        log.error("c");
        log.info("d"); // evicts "a"
        assert_eq!(log.entries().count(), 3);
        let msgs: Vec<&str> = log.entries().map(|e| e.message.as_str()).collect();
        assert_eq!(msgs, vec!["b", "c", "d"], "oldest evicted, order kept");
        let levels: Vec<LogLevel> = log.entries().map(|e| e.level).collect();
        assert_eq!(
            levels,
            vec![LogLevel::Info, LogLevel::Error, LogLevel::Info]
        );
    }

    #[test]
    fn log_last_error_is_sticky_until_dismiss_or_success() {
        let mut log = LogBuffer::new(10);
        assert_eq!(log.last_error(), None);
        log.error("render failed: boom");
        log.info("Preparing render...");
        log.info("some later status");
        // Later INFO lines must NOT clear the sticky error.
        assert_eq!(log.last_error(), Some("render failed: boom"));
        // A newer error replaces it.
        log.error("store write failed");
        assert_eq!(log.last_error(), Some("store write failed"));
        // Explicit dismiss clears.
        log.dismiss_error();
        assert_eq!(log.last_error(), None);
        // ... and a subsequent successful render clears too.
        log.error("again");
        log.note_render_success();
        assert_eq!(log.last_error(), None);
        // The ring still holds every line (dismiss touches only the banner state).
        assert_eq!(log.entries().count(), 5);
    }

    // ── WS4: pool sizing, export filename, drop classification ───────────────

    #[test]
    fn pool_threads_leaves_a_spare_core() {
        assert_eq!(pool_threads_leaving_spare(0), 1);
        assert_eq!(pool_threads_leaving_spare(1), 1);
        assert_eq!(pool_threads_leaving_spare(2), 1);
        assert_eq!(pool_threads_leaving_spare(8), 7);
        assert_eq!(pool_threads_leaving_spare(24), 23);
    }

    #[test]
    fn export_filename_is_sanitized_and_complete() {
        // Free-text tokens are sanitized (spaces/parens -> '-'), the date/time is
        // zero-padded, and the what-if suffix marks a fake-sun frame.
        assert_eq!(
            build_export_filename(
                "Enderlin d03 (run)",
                "IR band 13",
                "geo",
                2025,
                6,
                21,
                215,
                false
            ),
            "enderlin-d03-run_ir-band-13_geo_20250621_0215.png"
        );
        assert_eq!(
            build_export_filename("storm", "visible", "topdown", 2020, 1, 5, 130, true),
            "storm_visible_topdown_20200105_0130_whatif-sun.png"
        );
        // Empty/garbage tokens never produce an empty segment.
        assert_eq!(
            build_export_filename("", "??", "geo", 2020, 12, 31, 2359, false),
            "frame_frame_geo_20201231_2359.png"
        );
    }

    #[test]
    fn rgba_to_rgb_converts_space_to_black() {
        // A space pixel (alpha 0) exports black; opaque pixels keep their RGB.
        let rgba = [10u8, 20, 30, 0, 200, 100, 50, 255, 7, 8, 9, 128];
        assert_eq!(
            rgba_to_rgb_space_black(&rgba),
            vec![0, 0, 0, 200, 100, 50, 7, 8, 9]
        );
        assert!(rgba_to_rgb_space_black(&[]).is_empty());
    }

    #[test]
    fn classify_dropped_covers_the_matrix() {
        let dir = |p: &Path| p.to_string_lossy().ends_with("folder");
        let pb = |s: &str| PathBuf::from(s);
        // Path-less / empty drop: ignored.
        assert_eq!(classify_dropped(&[], &dir), None);
        // A directory -> a sequence (the opener expands it).
        assert_eq!(
            classify_dropped(&[pb("C:/runs/enderlin_folder")], &dir),
            Some(DropOpen::Sequence(vec![pb("C:/runs/enderlin_folder")]))
        );
        // A single .json (any case) -> a cached run.
        assert_eq!(
            classify_dropped(&[pb("C:/cache/run.JSON")], &dir),
            Some(DropOpen::CachedRun(pb("C:/cache/run.JSON")))
        );
        // A single other file -> a wrfout.
        assert_eq!(
            classify_dropped(&[pb("C:/runs/wrfout_d03_2020-01-05_01:30:00")], &dir),
            Some(DropOpen::Wrfout(pb(
                "C:/runs/wrfout_d03_2020-01-05_01:30:00"
            )))
        );
        // Multiple files -> a sequence of exactly those files.
        let many = vec![pb("a"), pb("b"), pb("c")];
        assert_eq!(
            classify_dropped(&many, &dir),
            Some(DropOpen::Sequence(many.clone()))
        );
    }

    // ── perspective orbit camera ─────────────────────────────────────────────

    fn orbit(az: f64, tilt: f64, range_km: f64) -> OrbitParams {
        OrbitParams {
            az_deg: az,
            tilt_deg: tilt,
            range_km,
            fov_deg: 45.0,
            width: 1280,
            height: 720,
        }
    }

    /// Straight-line ECEF distance between the eye and the look-at point.
    fn eye_look_distance_m(cam: &PerspectiveCamera) -> f64 {
        let e = simsat::camera::geodetic_to_ecef(cam.eye_lat_deg, cam.eye_lon_deg, cam.eye_alt_m);
        let l =
            simsat::camera::geodetic_to_ecef(cam.look_lat_deg, cam.look_lon_deg, cam.look_alt_m);
        ((e[0] - l[0]).powi(2) + (e[1] - l[1]).powi(2) + (e[2] - l[2]).powi(2)).sqrt()
    }

    #[test]
    fn orbit_near_vertical_tilt_puts_the_eye_overhead() {
        // Tilt 90 clamps to ORBIT_TILT_MAX_DEG (85): the eye sits (nearly) over the
        // centre at altitude ~range*sin(85), and the camera validates + has a basis.
        let cam = orbit_to_camera(&orbit(0.0, 90.0, 300.0), 44.5, -100.2, 300_000.0);
        let expect_alt = 300_000.0 * ORBIT_TILT_MAX_DEG.to_radians().sin();
        assert!(
            (cam.eye_alt_m - expect_alt).abs() < 1.0,
            "alt {}",
            cam.eye_alt_m
        );
        assert!(
            (cam.eye_lat_deg - 44.5).abs() < 0.3,
            "lat {}",
            cam.eye_lat_deg
        );
        assert!(
            (cam.eye_lon_deg - -100.2).abs() < 0.3,
            "lon {}",
            cam.eye_lon_deg
        );
        assert_eq!(
            (cam.look_lat_deg, cam.look_lon_deg, cam.look_alt_m),
            (44.5, -100.2, 0.0)
        );
        assert!(cam.validate().is_ok());
        assert!(cam.basis().is_ok());
        // The straight eye->centre distance is the slant range (within curvature 1%).
        let d = eye_look_distance_m(&cam);
        assert!((d - 300_000.0).abs() / 300_000.0 < 0.01, "distance {d}");
    }

    #[test]
    fn orbit_azimuth_180_places_the_camera_south_of_the_centre() {
        // Azimuth 180 = the camera sits SOUTH of the centre (looking north): the eye
        // latitude drops by ~ground_offset/R and the longitude is unchanged; the eye
        // altitude is range*sin(tilt).
        let cam = orbit_to_camera(&orbit(180.0, 30.0, 300.0), 44.5, -100.2, 300_000.0);
        let ground_deg = (300_000.0 * 30.0_f64.to_radians().cos() / 6.37e6).to_degrees();
        assert!(
            (cam.eye_lat_deg - (44.5 - ground_deg)).abs() < 0.01,
            "lat {} vs {}",
            cam.eye_lat_deg,
            44.5 - ground_deg
        );
        assert!(
            (cam.eye_lon_deg - -100.2).abs() < 1e-6,
            "lon {}",
            cam.eye_lon_deg
        );
        let expect_alt = 300_000.0 * 30.0_f64.to_radians().sin();
        assert!(
            (cam.eye_alt_m - expect_alt).abs() < 1.0,
            "alt {}",
            cam.eye_alt_m
        );
        assert!(cam.validate().is_ok());
    }

    #[test]
    fn orbit_range_and_dims_clamp_to_the_domain_bounds() {
        // Bounds are 0.3x..5x the domain diagonal; an out-of-range request clamps and
        // the eye->centre distance tracks the clamped range. Dims clamp to 2..=MAX_AXIS.
        let diag = 100_000.0;
        let (lo, hi) = orbit_range_bounds_m(diag);
        assert_eq!((lo, hi), (30_000.0, 500_000.0));
        let near = orbit_to_camera(&orbit(90.0, 45.0, 5.0), 44.5, -100.2, diag);
        let d_near = eye_look_distance_m(&near);
        assert!((d_near - lo).abs() / lo < 0.01, "near distance {d_near}");
        let mut far_orbit = orbit(90.0, 45.0, 10_000.0);
        far_orbit.width = 100_000;
        far_orbit.height = 0;
        let far = orbit_to_camera(&far_orbit, 44.5, -100.2, diag);
        let d_far = eye_look_distance_m(&far);
        // The flat range->(alt, ground-arc) split deviates from the straight chord as
        // the arc grows (earth curvature): ~1.4% at a 500 km range / 45 deg tilt.
        assert!((d_far - hi).abs() / hi < 0.02, "far distance {d_far}");
        assert_eq!((far.width, far.height), (MAX_AXIS, 2));
        assert!(far.validate().is_ok());
    }
}
