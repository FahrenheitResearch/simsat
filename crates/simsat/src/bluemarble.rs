//! Blue Marble Next Generation ground texture (design doc section 5, M1 slice).
//!
//! M1 ships ONE dev month (June / 200406, matching the Enderlin run) of the NASA
//! Blue Marble Next Generation "w/ Topography" global composite (Visible Earth,
//! public domain). The 12-month pack + GitHub release-asset hosting is M7 — NOT
//! built here. The global 2 km equirectangular JPEG (21600x10800) is fetched out
//! of band into the app cache dir (never committed; see notes/m1-notes.md for the
//! exact URL + sha256). At render time the studio decodes it once, crops to the
//! domain's lat/lon bounding box (+ a margin), resamples to `<= 4096^2`, and drops
//! the full decode. If the file is absent the studio degrades to a flat albedo and
//! names the expected path (it never crashes).
//!
//! Antimeridian: M1 does NOT handle a crop that wraps +/-180 (WRF domains rarely
//! cross it); such a domain would clamp at the edge. Documented, deferred.

use std::path::{Path, PathBuf};

/// The June (200406) 2 km Blue Marble NGB "w/ Topography" world JPEG file name.
/// Kept as the back-compatible single-month dev asset name (== `month_file_2km(6)`).
pub const BLUE_MARBLE_FILE: &str = "world.topo.200406.3x21600x10800.jpg";

/// Canonical download URL (recorded here + in notes/m1-notes.md).
pub const BLUE_MARBLE_URL: &str = "https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74368/world.topo.200406.3x21600x10800.jpg";

/// The expected on-disk path for the single-month dev texture under a cache dir
/// (M1 back-compat; the June asset). The seasonal pack uses `month_texture_path`.
pub fn texture_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join("bluemarble").join(BLUE_MARBLE_FILE)
}

/// The subdirectory under a cache dir where all Blue Marble months (2 km + the
/// materialized 8 km fallback) live.
pub fn bluemarble_dir(cache_dir: &Path) -> PathBuf {
    cache_dir.join("bluemarble")
}

/// Three-letter English month abbreviation for `month` in `1..=12` (wraps).
pub fn month_abbr(month: u32) -> &'static str {
    const ABBR: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    ABBR[((month.max(1) - 1) % 12) as usize]
}

/// 2 km monthly composite file name for `month` in `1..=12`
/// (`world.topo.2004MM.3x21600x10800.jpg` — the NASA BMNG w/ Topography set).
pub fn month_file_2km(month: u32) -> String {
    format!("world.topo.2004{:02}.3x21600x10800.jpg", month.clamp(1, 12))
}

/// 8 km emergency-fallback file name for `month` in `1..=12`. This is the NASA
/// 8 km (5400x2700) monthly composite downscaled to 3600x1800 and vendored in the
/// repo (see `asset_pack::fallback_8k_bytes`); the renderer never hard-fails offline.
pub fn month_file_8km(month: u32) -> String {
    format!("world.topo.2004{:02}.3x3600x1800.jpg", month.clamp(1, 12))
}

/// On-disk path of the cached 2 km composite for `month` under a cache dir.
pub fn month_texture_path(cache_dir: &Path, month: u32) -> PathBuf {
    bluemarble_dir(cache_dir).join(month_file_2km(month))
}

/// On-disk path of the materialized 8 km fallback for `month` under a cache dir
/// (written from the embedded bytes on first use, in an `8k/` subdir so it never
/// collides with a real 2 km download).
pub fn month_fallback_path(cache_dir: &Path, month: u32) -> PathBuf {
    bluemarble_dir(cache_dir)
        .join("8k")
        .join(month_file_8km(month))
}

/// Days per calendar month (non-leap). A seasonal GROUND blend is insensitive to the
/// one-day leap shift (the composites are monthly), so a fixed table keeps the
/// day-of-year -> month math pure and trivially testable.
const DAYS_IN_MONTH: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

/// The two bracketing monthly composites and the blend weight toward the second,
/// chosen by day-of-year with mid-month anchors. `weight_b == 0` (or `month_a ==
/// month_b`) means a single month (the date sits on that month's anchor).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MonthBlend {
    /// The earlier bracketing month (`1..=12`).
    pub month_a: u32,
    /// The later bracketing month (`1..=12`, wraps Dec->Jan).
    pub month_b: u32,
    /// Blend weight toward `month_b` in `[0,1]` (`0` = pure `month_a`).
    pub weight_b: f32,
}

impl MonthBlend {
    /// A pure single month (no seasonal blend) — used by the manual month override.
    pub fn single(month: u32) -> Self {
        let m = month.clamp(1, 12);
        MonthBlend {
            month_a: m,
            month_b: m,
            weight_b: 0.0,
        }
    }

    /// Whether this blend degenerates to a single month (on-anchor or override).
    pub fn is_single(&self) -> bool {
        self.month_a == self.month_b || self.weight_b <= 1.0e-4
    }

    /// A short status label, e.g. `"Dec/Jan blend (65% Jan)"` or `"Jun"`.
    pub fn label(&self) -> String {
        if self.is_single() {
            month_abbr(self.month_a).to_string()
        } else {
            format!(
                "{}/{} blend ({:.0}% {})",
                month_abbr(self.month_a),
                month_abbr(self.month_b),
                self.weight_b * 100.0,
                month_abbr(self.month_b),
            )
        }
    }
}

/// Choose the two bracketing monthly composites + blend weight for a `(month, day)`
/// date using MID-MONTH anchors: each composite represents the middle of its month.
///
/// The date maps to a continuous month coordinate `c = (month-1) + (day-0.5)/days`
/// in `[0,12)`; shifting by `0.5` puts the mid-month anchors on integer indices, so
/// `floor` gives the earlier month and the fractional part is the weight toward the
/// later one. The wrap makes early-January blend December->January and late-December
/// blend December->January across the year boundary. On a month's anchor day the
/// weight is `0` (a single month).
///
/// Examples: Jun 21 -> Jun/Jul, ~18% Jul; Jan 5 -> Dec/Jan, ~65% Jan.
pub fn month_blend(month: u32, day: u32) -> MonthBlend {
    let m = month.clamp(1, 12);
    let dim = DAYS_IN_MONTH[(m - 1) as usize] as f32;
    let day = day.clamp(1, 31) as f32;
    let c = (m as f32 - 1.0) + (day - 0.5) / dim; // [0, 12)
    let shifted = c - 0.5; // anchors now at integer month indices
    let lower = shifted.floor();
    let frac = (shifted - lower).clamp(0.0, 1.0);
    let wrap = |k: i32| -> u32 { k.rem_euclid(12) as u32 + 1 };
    MonthBlend {
        month_a: wrap(lower as i32),
        month_b: wrap(lower as i32 + 1),
        weight_b: frac,
    }
}

/// A cropped, resampled Blue Marble tile ready to upload as an Rgba8 texture. The
/// crop's exact geographic bounds are returned so the shader maps a pixel's
/// lat/lon into `[0,1]` UV precisely.
#[derive(Debug, Clone)]
pub struct BlueMarbleCrop {
    pub width: u32,
    pub height: u32,
    /// Row-major RGBA8 (`width * height * 4` bytes), row 0 = north.
    pub rgba: Vec<u8>,
    pub lat_min: f32,
    pub lat_max: f32,
    pub lon_min: f32,
    pub lon_max: f32,
}

impl BlueMarbleCrop {
    /// Bilinearly sample the crop at UV (`u` across west->east, `v` down north->south,
    /// both in `[0,1]`), returning the sRGB texel in `[0,1]` (the surface pass converts
    /// it to linear).
    ///
    /// Nearest-texel sampling made the 2 km ground read as hard blocks under the
    /// finite scan raster (owner-reported "pixelated mountains", visible directly and
    /// through thin cirrus). Bilinear lerps the 4 surrounding texels using the
    /// texel-centre convention: the continuous texel coordinate is `u*width - 0.5`
    /// (a texel centre sits at `(x+0.5)/width`), so sampling exactly at a texel centre
    /// returns that texel unchanged, and the geometric midpoint of a 2x2 tile returns
    /// the mean of the four texels. Neighbour indices clamp to the edge texels
    /// (clamp-to-edge addressing — the crop is a bounded tile, so no wrap), matching
    /// the prior nearest-texel clamp.
    pub fn sample_bilinear(&self, u: f32, v: f32) -> [f32; 3] {
        let w = self.width.max(1) as i64;
        let h = self.height.max(1) as i64;
        let stride = self.width.max(1) as usize;
        // Continuous texel coordinate (texel centres at (i+0.5)/dim).
        let fx = u.clamp(0.0, 1.0) * self.width as f32 - 0.5;
        let fy = v.clamp(0.0, 1.0) * self.height as f32 - 0.5;
        let x0 = fx.floor();
        let y0 = fy.floor();
        let tx = fx - x0; // fractional part in [0,1)
        let ty = fy - y0;
        let clampi = |i: i64, n: i64| i.clamp(0, n - 1);
        let x0i = clampi(x0 as i64, w);
        let x1i = clampi(x0 as i64 + 1, w);
        let y0i = clampi(y0 as i64, h);
        let y1i = clampi(y0 as i64 + 1, h);
        let texel = |xi: i64, yi: i64, c: usize| -> f32 {
            let o = (yi as usize * stride + xi as usize) * 4 + c;
            self.rgba[o] as f32 / 255.0
        };
        let mut out = [0.0f32; 3];
        for (c, o) in out.iter_mut().enumerate() {
            let top = texel(x0i, y0i, c) * (1.0 - tx) + texel(x1i, y0i, c) * tx;
            let bot = texel(x0i, y1i, c) * (1.0 - tx) + texel(x1i, y1i, c) * tx;
            *o = top * (1.0 - ty) + bot * ty;
        }
        out
    }
}

/// Errors loading the Blue Marble texture.
#[derive(Debug)]
pub enum BlueMarbleError {
    /// The texture file is not present at `path` (studio should degrade to flat).
    NotFound(PathBuf),
    /// The image failed to decode / had unexpected dimensions.
    Decode(String),
}

impl std::fmt::Display for BlueMarbleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(p) => write!(f, "Blue Marble texture not found: {}", p.display()),
            Self::Decode(s) => write!(f, "Blue Marble decode error: {s}"),
        }
    }
}

impl std::error::Error for BlueMarbleError {}

/// Integer source-pixel crop window plus the exact geographic bounds it covers,
/// for an equirectangular global image of `src_w x src_h`. Pure geometry (no IO),
/// so it is unit-tested without the 21 MB asset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CropBounds {
    pub x0: u32,
    pub y0: u32,
    pub width: u32,
    pub height: u32,
    pub lat_min: f32,
    pub lat_max: f32,
    pub lon_min: f32,
    pub lon_max: f32,
}

/// Compute the source crop window for a lat/lon bbox (+ margin) over an
/// equirectangular image. Longitude maps `-180..180 -> 0..src_w`, latitude maps
/// `90..-90 -> 0..src_h` (row 0 = north).
pub fn crop_bounds(
    src_w: u32,
    src_h: u32,
    lat_min: f32,
    lat_max: f32,
    lon_min: f32,
    lon_max: f32,
    margin_deg: f32,
) -> CropBounds {
    let (sw, sh) = (src_w as f64, src_h as f64);
    let lat_lo = (lat_min - margin_deg).clamp(-90.0, 90.0) as f64;
    let lat_hi = (lat_max + margin_deg).clamp(-90.0, 90.0) as f64;
    let lon_lo = (lon_min - margin_deg).clamp(-180.0, 180.0) as f64;
    let lon_hi = (lon_max + margin_deg).clamp(-180.0, 180.0) as f64;

    let x0 = ((lon_lo + 180.0) / 360.0 * sw).floor().clamp(0.0, sw - 1.0) as u32;
    let x1 = ((lon_hi + 180.0) / 360.0 * sw).ceil().clamp(1.0, sw) as u32;
    let y0 = ((90.0 - lat_hi) / 180.0 * sh).floor().clamp(0.0, sh - 1.0) as u32;
    let y1 = ((90.0 - lat_lo) / 180.0 * sh).ceil().clamp(1.0, sh) as u32;
    let x1 = x1.max(x0 + 1);
    let y1 = y1.max(y0 + 1);

    // Exact geographic bounds of the integer crop window.
    let clon_min = (x0 as f64 / sw) * 360.0 - 180.0;
    let clon_max = (x1 as f64 / sw) * 360.0 - 180.0;
    let clat_max = 90.0 - (y0 as f64 / sh) * 180.0;
    let clat_min = 90.0 - (y1 as f64 / sh) * 180.0;
    CropBounds {
        x0,
        y0,
        width: x1 - x0,
        height: y1 - y0,
        lat_min: clat_min as f32,
        lat_max: clat_max as f32,
        lon_min: clon_min as f32,
        lon_max: clon_max as f32,
    }
}

/// Largest power-preserving output dimensions `<= max_dim` for a crop, never
/// upsampling beyond the source crop (2 km is 2 km; upsampling adds no detail).
fn output_dims(crop_w: u32, crop_h: u32, max_dim: u32) -> (u32, u32) {
    let w = crop_w.min(max_dim).max(1);
    let h = crop_h.min(max_dim).max(1);
    (w, h)
}

/// Load + crop + resample the Blue Marble texture at `path` to a domain bbox.
/// `max_dim` caps each output axis (design section 5: `<= 4096`).
pub fn load_crop(
    path: &Path,
    lat_min: f32,
    lat_max: f32,
    lon_min: f32,
    lon_max: f32,
    margin_deg: f32,
    max_dim: u32,
) -> Result<BlueMarbleCrop, BlueMarbleError> {
    if !path.is_file() {
        return Err(BlueMarbleError::NotFound(path.to_path_buf()));
    }
    // The 21600x10800 global JPEG decodes to ~667 MB as RGB8, which EXCEEDS the
    // `image` crate's DEFAULT 512 MiB allocation limit — so the plain `image::open`
    // (which uses `Limits::default()`) fails with a `LimitError` and the studio then
    // silently fell back to a flat albedo on a perfectly good asset (owner-reported).
    // Decode through an explicit reader with the allocation limit lifted; the asset
    // is a fixed-size NASA texture fetched from a pinned URL, so removing the guard
    // here is safe and its footprint is bounded and known.
    let mut reader =
        image::ImageReader::open(path).map_err(|e| BlueMarbleError::Decode(e.to_string()))?;
    reader.no_limits();
    let img = reader
        .decode()
        .map_err(|e| BlueMarbleError::Decode(e.to_string()))?;
    let rgb = img.into_rgb8();
    let (src_w, src_h) = (rgb.width(), rgb.height());
    if src_w < 2 || src_h < 2 {
        return Err(BlueMarbleError::Decode(format!(
            "unexpected dimensions {src_w}x{src_h}"
        )));
    }
    let bounds = crop_bounds(src_w, src_h, lat_min, lat_max, lon_min, lon_max, margin_deg);
    let cropped =
        image::imageops::crop_imm(&rgb, bounds.x0, bounds.y0, bounds.width, bounds.height)
            .to_image();
    drop(rgb); // release the full-globe decode as early as possible

    let (out_w, out_h) = output_dims(bounds.width, bounds.height, max_dim);
    let resized = if out_w == bounds.width && out_h == bounds.height {
        cropped
    } else {
        image::imageops::resize(
            &cropped,
            out_w,
            out_h,
            image::imageops::FilterType::Triangle,
        )
    };

    let mut rgba = Vec::with_capacity((out_w * out_h * 4) as usize);
    for px in resized.pixels() {
        let [r, g, b] = px.0;
        rgba.extend_from_slice(&[r, g, b, 255]);
    }
    Ok(BlueMarbleCrop {
        width: out_w,
        height: out_h,
        rgba,
        lat_min: bounds.lat_min,
        lat_max: bounds.lat_max,
        lon_min: bounds.lon_min,
        lon_max: bounds.lon_max,
    })
}

/// Resample a crop to exactly `out_w x out_h` (triangle filter), preserving its
/// geographic bounds. Used to bring a fallback (8 km, lower-res) month onto the
/// dimensions of its higher-res blend partner before per-texel blending.
pub fn resample_crop(src: &BlueMarbleCrop, out_w: u32, out_h: u32) -> BlueMarbleCrop {
    let out_w = out_w.max(1);
    let out_h = out_h.max(1);
    if src.width == out_w && src.height == out_h {
        return src.clone();
    }
    let img = image::RgbaImage::from_raw(src.width, src.height, src.rgba.clone())
        .expect("crop rgba length matches width*height*4");
    let resized =
        image::imageops::resize(&img, out_w, out_h, image::imageops::FilterType::Triangle);
    BlueMarbleCrop {
        width: out_w,
        height: out_h,
        rgba: resized.into_raw(),
        lat_min: src.lat_min,
        lat_max: src.lat_max,
        lon_min: src.lon_min,
        lon_max: src.lon_max,
    }
}

/// Blend two domain crops into one season crop: `out = a*(1-w) + b*w` per texel,
/// with `w = weight_b`. `b` is resampled to `a`'s dimensions first if they differ
/// (the 2 km + 8 km-fallback mixed case), so the result always carries `a`'s grid +
/// geographic bounds. The blend is baked into the returned crop's `rgba`, so
/// `sample_bilinear(u, v)` samples the season-blended ground behind its UNCHANGED
/// signature (M3's surface shading keeps working verbatim).
pub fn blend_crops(a: &BlueMarbleCrop, b: &BlueMarbleCrop, weight_b: f32) -> BlueMarbleCrop {
    let w = weight_b.clamp(0.0, 1.0);
    // Degenerate weights avoid the (possibly resampling) blend entirely.
    if w <= 0.0 {
        return a.clone();
    }
    let matched;
    let b = if a.width == b.width && a.height == b.height {
        b
    } else {
        matched = resample_crop(b, a.width, a.height);
        &matched
    };
    let mut rgba = vec![0u8; a.rgba.len()];
    for (out, (pa, pb)) in rgba
        .chunks_exact_mut(4)
        .zip(a.rgba.chunks_exact(4).zip(b.rgba.chunks_exact(4)))
    {
        for (c, o) in out.iter_mut().enumerate() {
            let v = pa[c] as f32 * (1.0 - w) + pb[c] as f32 * w;
            *o = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    BlueMarbleCrop {
        width: a.width,
        height: a.height,
        rgba,
        lat_min: a.lat_min,
        lat_max: a.lat_max,
        lon_min: a.lon_min,
        lon_max: a.lon_max,
    }
}

/// Load the two bracketing months (each `path_*` may be a 2 km download OR an 8 km
/// fallback file) to the SAME domain bbox and blend them per `weight_b` into one
/// season crop.
///
/// - Both present -> `blend_crops(a, b, weight_b)`.
/// - Exactly one present -> that single month (no blend; the season degrades to the
///   month that is available — e.g. the partner failed to download offline).
/// - Neither present -> `NotFound(path_a)` (the caller renders flat albedo).
///
/// The domain crop args are identical for both months, so two 2 km months produce
/// identical dimensions and blend directly; a mixed 2 km/8 km pair is dimension-
/// matched inside `blend_crops`.
#[allow(clippy::too_many_arguments)]
pub fn load_season_crop(
    path_a: Option<&Path>,
    path_b: Option<&Path>,
    weight_b: f32,
    lat_min: f32,
    lat_max: f32,
    lon_min: f32,
    lon_max: f32,
    margin_deg: f32,
    max_dim: u32,
) -> Result<BlueMarbleCrop, BlueMarbleError> {
    let load = |p: Option<&Path>| -> Option<BlueMarbleCrop> {
        let p = p?;
        load_crop(p, lat_min, lat_max, lon_min, lon_max, margin_deg, max_dim).ok()
    };
    let a = load(path_a);
    let b = load(path_b);
    match (a, b) {
        (Some(a), Some(b)) => Ok(blend_crops(&a, &b, weight_b)),
        (Some(a), None) => Ok(a),
        (None, Some(b)) => Ok(b),
        (None, None) => Err(BlueMarbleError::NotFound(
            path_a.or(path_b).map(Path::to_path_buf).unwrap_or_default(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_path_is_under_bluemarble_subdir() {
        let p = texture_path(Path::new("/cache"));
        assert!(p.ends_with(format!("bluemarble/{BLUE_MARBLE_FILE}")));
    }

    /// A 2x2 tile with distinct red values; the geometric midpoint UV (0.5,0.5) lands
    /// exactly between all four texel centres, so bilinear returns their mean — the
    /// interpolated-between-neighbours behaviour the nearest sampler lacked.
    #[test]
    fn bilinear_midpoint_averages_the_four_texels() {
        let crop = BlueMarbleCrop {
            width: 2,
            height: 2,
            // row 0 (north): (0,0)=10, (1,0)=20 ; row 1 (south): (0,1)=30, (1,1)=50
            rgba: vec![
                10, 0, 0, 255, 20, 0, 0, 255, // north row
                30, 0, 0, 255, 50, 0, 0, 255, // south row
            ],
            lat_min: -1.0,
            lat_max: 1.0,
            lon_min: -1.0,
            lon_max: 1.0,
        };
        let px = crop.sample_bilinear(0.5, 0.5);
        let expected = (10.0 + 20.0 + 30.0 + 50.0) / 4.0 / 255.0;
        assert!(
            (px[0] - expected).abs() < 1e-6,
            "midpoint red {} != mean {}",
            px[0],
            expected
        );
        // A purely horizontal midpoint on the north row averages just that row pair.
        let mid_top = crop.sample_bilinear(0.5, 0.25);
        assert!(
            (mid_top[0] - (10.0 + 20.0) / 2.0 / 255.0).abs() < 1e-6,
            "north-row midpoint red {} != 15/255",
            mid_top[0]
        );
    }

    /// Sampling exactly at a texel centre (`(x+0.5)/dim`) must return that texel
    /// unchanged — bilinear degenerates to the exact texel with zero fractional weight.
    #[test]
    fn bilinear_at_texel_centre_returns_exact_texel() {
        let crop = BlueMarbleCrop {
            width: 2,
            height: 2,
            rgba: vec![
                10, 0, 0, 255, 20, 0, 0, 255, // north row
                30, 0, 0, 255, 50, 0, 0, 255, // south row
            ],
            lat_min: -1.0,
            lat_max: 1.0,
            lon_min: -1.0,
            lon_max: 1.0,
        };
        // Texel (1,0) centre is UV (0.75, 0.25); texel (0,1) centre is UV (0.25, 0.75).
        assert!((crop.sample_bilinear(0.75, 0.25)[0] - 20.0 / 255.0).abs() < 1e-6);
        assert!((crop.sample_bilinear(0.25, 0.75)[0] - 30.0 / 255.0).abs() < 1e-6);
        // Corner texel (0,0) centre UV (0.25,0.25); out-of-range UV clamps to the edge.
        assert!((crop.sample_bilinear(0.25, 0.25)[0] - 10.0 / 255.0).abs() < 1e-6);
        assert!((crop.sample_bilinear(1.0, 1.0)[0] - 50.0 / 255.0).abs() < 1e-6);
    }

    /// Day-of-year -> (month_a, month_b, weight) with mid-month anchors.
    #[test]
    fn month_blend_selects_bracketing_months() {
        // Jun 21 sits just past mid-June -> June/July, a small weight toward July.
        let jun = month_blend(6, 21);
        assert_eq!((jun.month_a, jun.month_b), (6, 7));
        assert!(
            (jun.weight_b - 0.183).abs() < 0.02,
            "Jun 21 weight_b {} != ~0.18",
            jun.weight_b
        );
        assert!(!jun.is_single());

        // Jan 5 is before mid-January -> the previous month wraps to December, with a
        // majority weight toward January (across the year boundary).
        let jan = month_blend(1, 5);
        assert_eq!((jan.month_a, jan.month_b), (12, 1));
        assert!(
            (jan.weight_b - 0.645).abs() < 0.02,
            "Jan 5 weight_b {} != ~0.65",
            jan.weight_b
        );
        assert_eq!(jan.label(), "Dec/Jan blend (65% Jan)");

        // Late December wraps forward to January too (Dec 28 -> Dec/Jan).
        let dec = month_blend(12, 28);
        assert_eq!((dec.month_a, dec.month_b), (12, 1));

        // A mid-month anchor (June's 30-day midpoint, day ~15.5) is a single month:
        // day 16 lands essentially on the anchor with a near-zero weight.
        let mid = month_blend(6, 16);
        assert_eq!(mid.month_a, 6);
        assert!(
            mid.weight_b < 0.02,
            "mid-month weight {} not ~0",
            mid.weight_b
        );
    }

    /// A single-month override / on-anchor date degrades cleanly to one month.
    #[test]
    fn month_blend_single_is_pure() {
        let s = MonthBlend::single(8);
        assert!(s.is_single());
        assert_eq!(s.label(), "Aug");
        assert_eq!(s.weight_b, 0.0);
    }

    /// The per-texel season blend equals the weighted average of the two months at a
    /// sample point (the interpolation the seasonal ground is built on). Two 2x2 crops
    /// with distinct constant colours; the blend sampled at a texel centre is the exact
    /// weighted mean of the two months' texels.
    #[test]
    fn blend_crops_is_weighted_average() {
        let mk = |r: u8, g: u8, b: u8| BlueMarbleCrop {
            width: 2,
            height: 2,
            rgba: {
                let mut v = Vec::new();
                for _ in 0..4 {
                    v.extend_from_slice(&[r, g, b, 255]);
                }
                v
            },
            lat_min: -1.0,
            lat_max: 1.0,
            lon_min: -1.0,
            lon_max: 1.0,
        };
        // Month A = winter-brown (100,80,60), Month B = summer-green (40,120,50).
        let a = mk(100, 80, 60);
        let b = mk(40, 120, 50);
        let w = 0.25f32;
        let blended = blend_crops(&a, &b, w);
        // Sample at a texel centre (uniform crops -> any UV gives the constant colour).
        let px = blended.sample_bilinear(0.75, 0.25);
        let expect = |ca: f32, cb: f32| (ca * (1.0 - w) + cb * w).round() / 255.0;
        assert!((px[0] - expect(100.0, 40.0)).abs() < 1.0 / 255.0);
        assert!((px[1] - expect(80.0, 120.0)).abs() < 1.0 / 255.0);
        assert!((px[2] - expect(60.0, 50.0)).abs() < 1.0 / 255.0);
        // weight 0 returns month A unchanged; weight 1 returns month B.
        assert_eq!(blend_crops(&a, &b, 0.0).rgba, a.rgba);
        assert_eq!(blend_crops(&a, &b, 1.0).rgba[..4], b.rgba[..4]);
    }

    /// A mixed-resolution blend (a 2x2 partner against a 4x4 partner, the 2 km/8 km-
    /// fallback case) resamples the second onto the first's grid, so the blend still
    /// carries the first crop's dimensions and produces a finite in-range texel.
    #[test]
    fn blend_crops_matches_dimensions() {
        let a = BlueMarbleCrop {
            width: 2,
            height: 2,
            rgba: vec![
                200, 0, 0, 255, 200, 0, 0, 255, 200, 0, 0, 255, 200, 0, 0, 255,
            ],
            lat_min: -1.0,
            lat_max: 1.0,
            lon_min: -1.0,
            lon_max: 1.0,
        };
        let b = BlueMarbleCrop {
            width: 4,
            height: 4,
            rgba: [0u8, 200, 0, 255].repeat(16),
            lat_min: -1.0,
            lat_max: 1.0,
            lon_min: -1.0,
            lon_max: 1.0,
        };
        let blended = blend_crops(&a, &b, 0.5);
        assert_eq!((blended.width, blended.height), (2, 2));
        let px = blended.sample_bilinear(0.5, 0.5);
        // Red ~100/255 (avg of 200 and 0), green ~100/255.
        assert!((px[0] - 100.0 / 255.0).abs() < 2.0 / 255.0);
        assert!((px[1] - 100.0 / 255.0).abs() < 2.0 / 255.0);
    }

    #[test]
    fn month_file_names_follow_the_nasa_pattern() {
        assert_eq!(month_file_2km(6), BLUE_MARBLE_FILE);
        assert_eq!(month_file_2km(1), "world.topo.200401.3x21600x10800.jpg");
        assert_eq!(month_file_2km(12), "world.topo.200412.3x21600x10800.jpg");
        assert_eq!(month_file_8km(6), "world.topo.200406.3x3600x1800.jpg");
        assert_eq!(month_abbr(1), "Jan");
        assert_eq!(month_abbr(12), "Dec");
    }

    #[test]
    fn crop_bounds_center_of_globe() {
        // 360x180 image = 1 deg per pixel. Bbox around the equator/prime meridian.
        let b = crop_bounds(360, 180, -10.0, 10.0, -20.0, 20.0, 0.0);
        // lon -20..20 -> x 160..200; lat -10..10 -> y 80..100.
        assert_eq!(b.x0, 160);
        assert_eq!(b.width, 40);
        assert_eq!(b.y0, 80);
        assert_eq!(b.height, 20);
        assert!((b.lon_min + 20.0).abs() < 1e-4);
        assert!((b.lon_max - 20.0).abs() < 1e-4);
        assert!((b.lat_max - 10.0).abs() < 1e-4);
        assert!((b.lat_min + 10.0).abs() < 1e-4);
    }

    #[test]
    fn crop_bounds_applies_margin_and_clamps() {
        let b = crop_bounds(360, 180, 80.0, 89.0, -179.0, 179.0, 5.0);
        // Latitude clamps at the pole; longitude clamps at the edges.
        assert!(b.lat_max <= 90.0);
        assert_eq!(b.x0, 0);
        assert_eq!(b.x0 + b.width, 360);
    }

    #[test]
    fn crop_load_from_a_synthetic_image() {
        // Write a tiny 8x4 equirect PNG, then crop the eastern half.
        let dir = std::env::temp_dir().join(format!("simsat-bm-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.png");
        let mut img = image::RgbImage::new(8, 4);
        for (x, _y, px) in img.enumerate_pixels_mut() {
            *px = image::Rgb([(x * 30) as u8, 0, 0]);
        }
        img.save(&path).unwrap();
        let crop = load_crop(&path, -10.0, 10.0, 90.0, 170.0, 0.0, 4096).unwrap();
        assert!(crop.width >= 1 && crop.height >= 1);
        assert_eq!(crop.rgba.len(), (crop.width * crop.height * 4) as usize);
        // Eastern longitudes -> higher red channel than the western edge.
        assert!(crop.lon_min > 0.0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_file_reports_not_found() {
        let err = load_crop(
            Path::new("/no/such/bluemarble.jpg"),
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            4096,
        );
        assert!(matches!(err, Err(BlueMarbleError::NotFound(_))));
    }

    /// The real 21600x10800 Blue Marble decodes to ~667 MB (RGB8), which exceeds the
    /// `image` crate's DEFAULT 512 MiB allocation limit — the owner-reported cause of
    /// the silent flat-albedo fallback. This reproduces the failure on a synthetic
    /// image over that limit, then confirms OUR loader (which lifts the limit)
    /// decodes it. Solid color + fast PNG keeps the file tiny and the test quick
    /// while the DECODE still reserves the full ~588 MB.
    #[test]
    fn decodes_past_the_default_512_mib_alloc_limit() {
        use image::ImageEncoder;
        let (w, h) = (14000u32, 14000u32); // 14000^2 * 3 = 588 MB > 512 MiB
        let dir = std::env::temp_dir().join(format!("simsat-bmbig-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("big.png");
        {
            let buf = vec![32u8; (w as usize) * (h as usize) * 3];
            let file = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
            image::codecs::png::PngEncoder::new_with_quality(
                file,
                image::codecs::png::CompressionType::Fast,
                image::codecs::png::FilterType::NoFilter,
            )
            .write_image(&buf, w, h, image::ExtendedColorType::Rgb8)
            .unwrap();
        }
        // The plain default-limits decode fails (the exact owner-reported cause).
        assert!(
            image::open(&path).is_err(),
            "default-limits image::open must hit the 512 MiB alloc limit"
        );
        // OUR loader lifts the limit and decodes; a small bbox crops to a small tile.
        let crop = load_crop(&path, -10.0, 10.0, -20.0, 20.0, 0.0, 256)
            .expect("load_crop must decode past the default limit");
        assert!(crop.width >= 1 && crop.height >= 1);
        assert_eq!(crop.rgba.len(), (crop.width * crop.height * 4) as usize);
        std::fs::remove_dir_all(&dir).ok();
    }
}
