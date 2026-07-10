//! Seasonal Blue Marble asset pack (design doc section 5, M7 slice).
//!
//! The 12-month NASA Blue Marble Next Generation "w/ Topography" global 2 km
//! composites (`world.topo.2004MM.3x21600x10800.jpg`, Visible Earth collection 1484,
//! PUBLIC DOMAIN) are the seasonal ground: a render lazily fetches only the one or two
//! months the day-of-year lerp needs, verifies each against a pinned SHA-256, and
//! caches it under the existing `{cache}/bluemarble/` dir. The renderer never hard-
//! fails offline — a ~6.4 MB set of 8 km composites (downscaled to 3600x1800) is
//! vendored IN the binary (`include_bytes!`) and materialized to the cache on demand.
//!
//! HOSTING (owner decision 4): the 12 JPEGs + a `manifest.json` (month -> sha256 ->
//! asset-url map) are uploaded as GitHub release assets (`bluemarble-2km-v1` on the
//! public `FahrenheitResearch/simsat` repo). The downloader tries the GitHub asset URL
//! FIRST and then falls back to the authoritative NASA `eoimages.gsfc.nasa.gov` URL —
//! both gated by the SAME SHA-256 — so a missing/unreachable mirror degrades cleanly
//! to NASA. The pinned manifest ships embedded in the binary (`bluemarble_manifest.json`).
//!
//! The streaming-download + SHA-256 gate is PORTED (with attribution) from BowEcho's
//! `crates/app_ui/src/self_update.rs` (`download_update_asset`, `parse_sha256_asset`,
//! `self_update_http_client`) — the real downloadable-asset mechanism in that app.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;

use crate::bluemarble::{self, BlueMarbleCrop, MonthBlend};

/// The pinned pack manifest, embedded so the app knows every month's SHA-256 + URLs
/// without a network round-trip (verification never depends on fetching the manifest).
pub const EMBEDDED_MANIFEST_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/bluemarble_manifest.json"
));

/// One month's asset entry in the manifest.
#[derive(Debug, Clone, Deserialize)]
pub struct MonthAsset {
    /// Calendar month `1..=12`.
    pub month: u32,
    /// Bare file name (`world.topo.2004MM.3x21600x10800.jpg`).
    pub filename: String,
    /// GitHub release-asset URL (tried first; 404s while the repo is private).
    pub asset_url: String,
    /// NASA `eoimages` URL (public fallback; authoritative source).
    pub nasa_url: String,
    /// Lowercase hex SHA-256 of the exact JPEG bytes (the download gate).
    pub sha256: String,
    /// File size in bytes (a cheap cached-file integrity proxy).
    pub bytes: u64,
}

/// The parsed pack manifest (month -> sha256 -> asset-url map + hosting metadata).
#[derive(Debug, Clone, Deserialize)]
pub struct PackManifest {
    /// The GitHub release tag hosting the assets (`bluemarble-2km-v1`).
    pub release_tag: String,
    /// Base URL for the release assets (informational).
    pub asset_base_url: String,
    /// Human-readable resolution label (`2km`).
    #[serde(default)]
    pub resolution_label: String,
    /// The 12 monthly assets.
    pub months: Vec<MonthAsset>,
}

impl PackManifest {
    /// The manifest entry for `month` in `1..=12`, if present.
    pub fn month(&self, month: u32) -> Option<&MonthAsset> {
        self.months.iter().find(|m| m.month == month)
    }
}

/// Parse the embedded pinned manifest. Panics only on a corrupt vendored file (a build
/// error, not a runtime condition) — the JSON is validated at commit time.
pub fn embedded_manifest() -> PackManifest {
    serde_json::from_str(EMBEDDED_MANIFEST_JSON)
        .expect("embedded bluemarble_manifest.json must be valid PackManifest JSON")
}

/// The vendored 8 km emergency-fallback JPEG bytes for `month` in `1..=12` (the NASA
/// 8 km monthly composite downscaled to 3600x1800; ~500 KB each, ~6.4 MB total). `None`
/// outside `1..=12`. These guarantee an offline render never hard-fails.
pub fn fallback_8k_bytes(month: u32) -> Option<&'static [u8]> {
    macro_rules! fb {
        ($mm:literal) => {
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/bluemarble_8k/world.topo.2004",
                $mm,
                ".3x3600x1800.jpg"
            ))
        };
    }
    const FALLBACK: [&[u8]; 12] = [
        fb!("01"),
        fb!("02"),
        fb!("03"),
        fb!("04"),
        fb!("05"),
        fb!("06"),
        fb!("07"),
        fb!("08"),
        fb!("09"),
        fb!("10"),
        fb!("11"),
        fb!("12"),
    ];
    if (1..=12).contains(&month) {
        Some(FALLBACK[(month - 1) as usize])
    } else {
        None
    }
}

// ── SHA-256 gate (ported from self_update.rs) ────────────────────────────────────

/// Lowercase hex SHA-256 of `bytes`.
pub fn sha256_of_bytes(bytes: &[u8]) -> String {
    use sha2::Digest as _;
    format!("{:x}", sha2::Sha256::digest(bytes))
}

/// Stream a file through SHA-256 without loading it whole (the assets are ~24 MB).
/// PORTED from the hashing loop in `self_update.rs::download_update_asset`.
pub fn sha256_of_file(path: &Path) -> Result<String, String> {
    use sha2::Digest as _;
    use std::io::Read as _;
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut hasher = sha2::Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Verify an on-disk asset against an expected lowercase-hex SHA-256. `Ok(())` iff the
/// file hashes to `expected` (case-insensitive); an explicit mismatch error otherwise.
pub fn verify_file(path: &Path, expected: &str) -> Result<(), String> {
    let actual = sha256_of_file(path)?;
    if actual.eq_ignore_ascii_case(expected) {
        Ok(())
    } else {
        Err(format!(
            "SHA-256 mismatch for {}: manifest {expected}, file hashed to {actual}",
            path.display()
        ))
    }
}

/// Parse a `.sha256` sidecar into the lowercase hex digest — accepts `sha256sum`
/// binary mode (`<hex> *<name>`), text mode (`<hex>  <name>`), and a bare digest.
/// PORTED VERBATIM from `self_update.rs::parse_sha256_asset`.
pub fn parse_sha256(contents: &str) -> Option<String> {
    let token = contents.split_whitespace().next()?;
    (token.len() == 64 && token.bytes().all(|b| b.is_ascii_hexdigit()))
        .then(|| token.to_ascii_lowercase())
}

// ── streaming download (ported from self_update.rs) ──────────────────────────────

/// One-shot download HTTP client: generous total timeout for a ~24 MB asset on a slow
/// link, rustls TLS like every other outbound call. PORTED from
/// `self_update.rs::self_update_http_client`.
pub fn http_client() -> Result<reqwest::blocking::Client, String> {
    reqwest::blocking::Client::builder()
        .user_agent(concat!(
            "simsat/",
            env!("CARGO_PKG_VERSION"),
            " (bluemarble)"
        ))
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(15 * 60))
        .build()
        .map_err(|e| format!("could not build HTTP client: {e}"))
}

/// Stream `url` to `destination`, hashing the exact bytes written; returns the
/// lowercase hex SHA-256. `progress` fires `(received, total)` (total from
/// Content-Length when present). PORTED from `self_update.rs::download_update_asset`.
fn download_stream(
    client: &reqwest::blocking::Client,
    url: &str,
    destination: &Path,
    mut progress: impl FnMut(u64, Option<u64>),
) -> Result<String, String> {
    use sha2::Digest as _;
    use std::io::{Read as _, Write as _};
    let mut response = client
        .get(url)
        .send()
        .and_then(|r| r.error_for_status())
        .map_err(|e| format!("download failed: {e}"))?;
    let total = response.content_length();
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("could not create {}: {e}", parent.display()))?;
    }
    let mut file = std::fs::File::create(destination)
        .map_err(|e| format!("could not create {}: {e}", destination.display()))?;
    let mut hasher = sha2::Sha256::new();
    let mut received = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = response
            .read(&mut buffer)
            .map_err(|e| format!("download interrupted: {e}"))?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read])
            .map_err(|e| format!("could not write {}: {e}", destination.display()))?;
        hasher.update(&buffer[..read]);
        received += read as u64;
        progress(received, total);
    }
    file.flush()
        .map_err(|e| format!("could not finish writing {}: {e}", destination.display()))?;
    Ok(format!("{:x}", hasher.finalize()))
}

/// Download the first of `urls` that both fetches AND matches `expected_sha256`,
/// atomically installing it at `dest`. Each attempt streams to a `.part` sibling,
/// hashes, and (on match) renames into place; a mismatch or fetch error deletes the
/// partial and tries the next URL. `status` receives coarse human-readable progress.
/// Mirrors the self_update gate: an unverified file is NEVER left in place.
pub fn download_verified(
    client: &reqwest::blocking::Client,
    urls: &[&str],
    dest: &Path,
    expected_sha256: &str,
    status: &mut dyn FnMut(String),
) -> Result<(), String> {
    let part = dest.with_extension("part");
    let mut last_err = String::from("no download URLs provided");
    for url in urls {
        let mut last_step = u64::MAX;
        let result = download_stream(client, url, &part, |received, total| {
            let step = match total {
                Some(t) if t > 0 => received * 50 / t, // ~2% granularity
                _ => received / (4 * 1024 * 1024),     // per 4 MiB when size unknown
            };
            if step != last_step {
                last_step = step;
                match total {
                    Some(t) if t > 0 => status(format!(
                        "  downloading {:.0}% ({:.1}/{:.1} MB)",
                        received as f64 / t as f64 * 100.0,
                        received as f64 / 1.0e6,
                        t as f64 / 1.0e6,
                    )),
                    _ => status(format!("  downloading {:.1} MB", received as f64 / 1.0e6)),
                }
            }
        });
        match result {
            Ok(actual) if actual.eq_ignore_ascii_case(expected_sha256) => {
                std::fs::rename(&part, dest).map_err(|e| {
                    let _ = std::fs::remove_file(&part);
                    format!("could not install {}: {e}", dest.display())
                })?;
                return Ok(());
            }
            Ok(actual) => {
                let _ = std::fs::remove_file(&part);
                last_err = format!(
                    "SHA-256 mismatch from {url}: expected {expected_sha256}, got {actual}"
                );
            }
            Err(e) => {
                let _ = std::fs::remove_file(&part);
                last_err = format!("{url}: {e}");
            }
        }
    }
    Err(last_err)
}

// ── month resolution + seasonal load ─────────────────────────────────────────────

/// Where a resolved month's pixels came from (for status + QA honesty).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonthSource {
    /// A cached 2 km composite already on disk (the expected size).
    TwoKmCached,
    /// A 2 km composite just downloaded + SHA-256-verified.
    TwoKmDownloaded,
    /// The vendored 8 km emergency fallback (offline / download failed).
    EightKmFallback,
}

impl MonthSource {
    pub fn label(self) -> &'static str {
        match self {
            MonthSource::TwoKmCached => "2km cached",
            MonthSource::TwoKmDownloaded => "2km downloaded",
            MonthSource::EightKmFallback => "8km fallback",
        }
    }
    pub fn is_fallback(self) -> bool {
        matches!(self, MonthSource::EightKmFallback)
    }
}

/// A resolved month on the local filesystem.
#[derive(Debug, Clone)]
pub struct ResolvedMonth {
    pub path: PathBuf,
    pub source: MonthSource,
}

fn file_len(path: &Path) -> Option<u64> {
    std::fs::metadata(path).ok().map(|m| m.len())
}

/// Materialize the vendored 8 km fallback for `month` to the cache (once) and return
/// its path. Always succeeds when disk is writable — the offline never-fail guarantee.
pub fn ensure_fallback_8k(cache_dir: &Path, month: u32) -> Result<PathBuf, String> {
    let path = bluemarble::month_fallback_path(cache_dir, month);
    if path.is_file() {
        return Ok(path);
    }
    let bytes =
        fallback_8k_bytes(month).ok_or_else(|| format!("no 8 km fallback for month {month}"))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("could not create {}: {e}", parent.display()))?;
    }
    std::fs::write(&path, bytes).map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(path)
}

/// Resolve `month` to a local file. Order: (1) cached 2 km of the expected size ->
/// `TwoKmCached`; (2) if `allow_download` + a client + a manifest entry, download the
/// 2 km asset SHA-256-gated (GitHub URL then NASA URL) -> `TwoKmDownloaded`; (3) the
/// vendored 8 km fallback -> `EightKmFallback`. Only a hard disk failure errors.
pub fn resolve_month(
    cache_dir: &Path,
    manifest: &PackManifest,
    month: u32,
    allow_download: bool,
    client: Option<&reqwest::blocking::Client>,
    status: &mut dyn FnMut(String),
) -> Result<ResolvedMonth, String> {
    let asset = manifest.month(month);
    let cached = bluemarble::month_texture_path(cache_dir, month);

    // 1. Cached 2 km present with the expected size (cheap integrity proxy; the full
    //    SHA-256 gate ran at download time). If we have no manifest entry (e.g. the M1
    //    June dev asset) a present file is accepted as-is.
    match asset {
        Some(a) if file_len(&cached) == Some(a.bytes) => {
            return Ok(ResolvedMonth {
                path: cached,
                source: MonthSource::TwoKmCached,
            });
        }
        None if cached.is_file() => {
            return Ok(ResolvedMonth {
                path: cached,
                source: MonthSource::TwoKmCached,
            });
        }
        _ => {}
    }

    // 2. Download the 2 km asset (GitHub release URL first, NASA URL fallback).
    if allow_download && let (Some(a), Some(client)) = (asset, client) {
        status(format!(
            "Downloading Blue Marble {} 2 km ({:.1} MB)...",
            bluemarble::month_abbr(month),
            a.bytes as f64 / 1.0e6,
        ));
        let urls = [a.asset_url.as_str(), a.nasa_url.as_str()];
        match download_verified(client, &urls, &cached, &a.sha256, status) {
            Ok(()) => {
                return Ok(ResolvedMonth {
                    path: cached,
                    source: MonthSource::TwoKmDownloaded,
                });
            }
            Err(e) => status(format!(
                "Blue Marble {} download failed ({e}); using 8 km fallback.",
                bluemarble::month_abbr(month)
            )),
        }
    }

    // 3. Vendored 8 km fallback (always available offline).
    let path = ensure_fallback_8k(cache_dir, month)?;
    if allow_download {
        status(format!(
            "Blue Marble {} unavailable; using vendored 8 km fallback.",
            bluemarble::month_abbr(month)
        ));
    }
    Ok(ResolvedMonth {
        path,
        source: MonthSource::EightKmFallback,
    })
}

/// The resolved seasonal ground for one render.
pub struct SeasonGround {
    /// The season-blended domain crop (blend baked in; `sample_bilinear` unchanged).
    pub crop: BlueMarbleCrop,
    /// The chosen month blend (for the status line).
    pub blend: MonthBlend,
    pub source_a: MonthSource,
    /// `None` when the blend is a single month (`source_a` covers it).
    pub source_b: Option<MonthSource>,
}

impl SeasonGround {
    /// Whether ANY contributing month came from the 8 km fallback.
    pub fn used_fallback(&self) -> bool {
        self.source_a.is_fallback() || self.source_b.is_some_and(MonthSource::is_fallback)
    }
    /// A status line, e.g. `"Blue Marble: Dec/Jan blend (65% Jan) [2km cached]"`.
    pub fn status_line(&self) -> String {
        let src = if self.used_fallback() {
            " [8km fallback]"
        } else {
            ""
        };
        format!("Blue Marble: {}{}", self.blend.label(), src)
    }
}

/// Resolve + load + blend the seasonal ground for `(month, day)` (or a manual
/// `month_override` for what-if), cropped to the domain bbox. Fetches only the 1-2
/// months the blend needs (lazy per-month), each SHA-256-gated, with the 8 km fallback
/// on any miss. `status` receives human-readable progress lines.
#[allow(clippy::too_many_arguments)]
pub fn load_season_ground(
    cache_dir: &Path,
    manifest: &PackManifest,
    month: u32,
    day: u32,
    month_override: Option<u32>,
    allow_download: bool,
    lat_min: f32,
    lat_max: f32,
    lon_min: f32,
    lon_max: f32,
    margin_deg: f32,
    max_dim: u32,
    status: &mut dyn FnMut(String),
) -> Result<SeasonGround, String> {
    let blend = match month_override {
        Some(m) => MonthBlend::single(m),
        None => bluemarble::month_blend(month, day),
    };
    let client = if allow_download {
        http_client().ok()
    } else {
        None
    };

    let ra = resolve_month(
        cache_dir,
        manifest,
        blend.month_a,
        allow_download,
        client.as_ref(),
        status,
    )?;
    let (path_b, source_b) = if blend.is_single() {
        (None, None)
    } else {
        let rb = resolve_month(
            cache_dir,
            manifest,
            blend.month_b,
            allow_download,
            client.as_ref(),
            status,
        )?;
        (Some(rb.path), Some(rb.source))
    };

    let crop = bluemarble::load_season_crop(
        Some(&ra.path),
        path_b.as_deref(),
        blend.weight_b,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        margin_deg,
        max_dim,
    )
    .map_err(|e| e.to_string())?;

    Ok(SeasonGround {
        crop,
        blend,
        source_a: ra.source,
        source_b,
    })
}

/// Eagerly download the full-year 2 km pack (all 12 months), SHA-256-gated. Returns the
/// count of months now available at 2 km (cached or freshly downloaded). Any month that
/// cannot be fetched is left on its 8 km fallback and NOT counted. `status` reports each.
pub fn download_full_year(
    cache_dir: &Path,
    manifest: &PackManifest,
    status: &mut dyn FnMut(String),
) -> Result<usize, String> {
    let client = http_client()?;
    let mut have = 0usize;
    for month in 1..=12 {
        let r = resolve_month(cache_dir, manifest, month, true, Some(&client), status)?;
        if !r.source.is_fallback() {
            have += 1;
        }
    }
    status(format!(
        "Full-year pack: {have}/12 months available at 2 km."
    ));
    Ok(have)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_manifest_parses_all_12_months() {
        let m = embedded_manifest();
        assert_eq!(m.release_tag, "bluemarble-2km-v1");
        assert_eq!(m.months.len(), 12);
        // June's pinned SHA-256 is the known-good M1 value (validates the whole set).
        let june = m.month(6).expect("June entry");
        assert_eq!(
            june.sha256,
            "6aebde5c1e11864198a0d104e75250bf4feb68f170026aecef4cd36ec0768aff"
        );
        assert_eq!(june.filename, bluemarble::month_file_2km(6));
        // Every month 1..=12 present exactly once, with a 64-hex sha + a NASA URL.
        for month in 1..=12 {
            let a = m.month(month).expect("month present");
            assert_eq!(a.sha256.len(), 64);
            assert!(a.nasa_url.starts_with("https://eoimages.gsfc.nasa.gov/"));
            assert!(a.asset_url.contains("bluemarble-2km-v1"));
        }
    }

    #[test]
    fn fallback_8k_bytes_present_for_all_months() {
        for month in 1..=12 {
            let b = fallback_8k_bytes(month).expect("fallback bytes");
            // A real JPEG SOI marker + non-trivial size.
            assert!(b.len() > 10_000, "month {month} fallback too small");
            assert_eq!(&b[..2], &[0xFF, 0xD8], "month {month} not a JPEG");
        }
        assert!(fallback_8k_bytes(0).is_none());
        assert!(fallback_8k_bytes(13).is_none());
    }

    #[test]
    fn parse_sha256_accepts_ci_formats() {
        let hex = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        assert_eq!(
            parse_sha256(&format!("{hex} *f.jpg\n")).as_deref(),
            Some(hex)
        );
        assert_eq!(
            parse_sha256(&format!("{hex}  f.jpg\n")).as_deref(),
            Some(hex)
        );
        assert_eq!(
            parse_sha256(&hex.to_ascii_uppercase()).as_deref(),
            Some(hex)
        );
        assert_eq!(parse_sha256(""), None);
        assert_eq!(parse_sha256("not-a-hash *f"), None);
        assert_eq!(parse_sha256(&hex[..63]), None);
    }

    /// The SHA-256 gate accepts a good asset and rejects a corrupted one (brief D).
    #[test]
    fn verify_file_accepts_good_rejects_corrupted() {
        let dir = std::env::temp_dir().join(format!("simsat-verify-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("asset.bin");
        let bytes = b"blue marble monthly composite bytes";
        std::fs::write(&path, bytes).unwrap();
        let good = sha256_of_bytes(bytes);
        // Good asset: the streamed file hash matches the in-memory hash and verifies.
        assert_eq!(sha256_of_file(&path).unwrap(), good);
        assert!(verify_file(&path, &good).is_ok());
        // Corrupted asset: flip one byte -> the same expected hash now REJECTS it.
        std::fs::write(&path, b"blue marble monthly composite bytez").unwrap();
        let err = verify_file(&path, &good).unwrap_err();
        assert!(err.contains("SHA-256 mismatch"), "unexpected: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The 8 km fallback path loads when a 2 km month is absent AND downloads are off
    /// (offline) — the never-hard-fail contract (brief D). `resolve_month` returns the
    /// materialized fallback, and `load_crop` decodes it to a real domain crop.
    #[test]
    fn resolves_and_loads_8k_fallback_when_2km_absent_offline() {
        let manifest = embedded_manifest();
        let dir = std::env::temp_dir().join(format!("simsat-fallback-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut noop = |_: String| {};
        let resolved = resolve_month(
            &dir, &manifest, 12, /*allow_download=*/ false, None, &mut noop,
        )
        .unwrap();
        assert_eq!(resolved.source, MonthSource::EightKmFallback);
        assert!(resolved.path.is_file());
        // The materialized fallback is a decodable equirectangular JPEG: a small domain
        // crop over the US Rockies loads to a non-empty RGBA tile.
        let crop = bluemarble::load_crop(&resolved.path, 30.0, 45.0, -115.0, -100.0, 1.0, 4096)
            .expect("8 km fallback must decode + crop");
        assert!(crop.width >= 1 && crop.height >= 1);
        assert_eq!(crop.rgba.len(), (crop.width * crop.height * 4) as usize);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The end-to-end seasonal load, fully offline, blends the two bracketing 8 km
    /// fallbacks into one domain crop (Dec/Jan across the year boundary).
    #[test]
    fn season_ground_blends_offline_fallbacks() {
        let manifest = embedded_manifest();
        let dir = std::env::temp_dir().join(format!("simsat-season-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut noop = |_: String| {};
        let g = load_season_ground(
            &dir, &manifest, 1, 5, None, /*allow_download=*/ false, 30.0, 45.0, -115.0,
            -100.0, 1.0, 4096, &mut noop,
        )
        .expect("offline season load");
        assert_eq!((g.blend.month_a, g.blend.month_b), (12, 1));
        assert!(g.used_fallback());
        assert!(g.crop.width >= 1 && g.crop.height >= 1);
        assert!(g.status_line().contains("Dec/Jan"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
