//! Web-map LAYER export (the broadcast-visuals / Mapbox-class integration).
//!
//! Turns the top-down cloud-layer render ([`crate::topdown::render_cloud_layer_frame`])
//! into something a third-party web map engine can composite over its OWN basemap:
//! a north-up, EPSG:3857-aligned (Web Mercator) image pair — the premultiplied-alpha
//! cloud RGBA plus a single-channel ground cloud-shadow MULTIPLY layer — with the four
//! corner lon/lats a Mapbox GL `ImageSource` needs.
//!
//! PIPELINE. The native cloud layer is rendered on the domain's own Lambert map raster
//! (uniform in WRF grid index, row 0 = north — the same raster every top-down product
//! uses, so the layer registers with them). A web map cannot place a Lambert-grid image
//! directly, so [`reproject_cloud_layer`] resamples it onto a [`MercatorGrid`]: a
//! north-up output raster whose rows/columns are EXACT constant-y / constant-x lines of
//! EPSG:3857. Per output pixel: inverse-Mercator -> (lat, lon) -> the WRF projection
//! forward ([`crate::frame::GridGeoref::forward`]) -> fractional native-raster pixel ->
//! bilinear sample. Out-of-coverage pixels are honest no-data: cloud alpha 0,
//! shadow 1.0 (neutral white — no shadow).
//!
//! ALPHA MODEL (the load-bearing decision). The engine computes and RESAMPLES the cloud
//! layer in PREMULTIPLIED alpha: the color channels hold the tonemapped cloud-only
//! radiance — the additive term of the shipped composite `L = L_bg*T + L_cloud` — and
//! alpha = `1 - T_cloud`. Premultiplied is the correct space for filtering (a bilinear
//! tap of straight-alpha data bleeds the arbitrary color of transparent texels into the
//! cloud edge) and matches the physical over-composite (`src + dst*(1-a)`). DELIVERY to
//! a PNG or a host that composites with STRAIGHT alpha (`src*a + dst*(1-a)`) goes
//! through [`unpremultiply_rgba`]: PNG stores straight alpha by spec, and browsers /
//! Mapbox GL premultiply at texture upload — shipping premultiplied bytes in a PNG
//! would double-multiply. The un-premultiply clamps color to 1.0 where the additive
//! cloud light exceeds its own alpha (an optically thin but bright wisp), losing that
//! excess in straight mode — a documented approximation; opaque cloud (alpha ~1) is
//! exact.
//!
//! DATUM NOTE. Our lat/lon live on the WRF sphere (`R = 6.37e6`, owner decision 5);
//! the Mercator math here is the STANDARD EPSG:3857 spherical formulas
//! (`R = 6_378_137`), so the output grid is a true Web Mercator raster and the corner
//! coordinates are ordinary WGS84-style lon/lat numbers. Treating WRF-sphere lat/lon
//! as web-map lat/lon is the same datum approximation every WRF field plotted on a web
//! map makes; the standard here is physical plausibility, not sub-pixel registration
//! against real imagery.
//!
//! TILING (the (b) delivery path) is a documented FOLLOW-UP, not built this wave:
//! because the [`MercatorGrid`] rows/cols are exact EPSG:3857 lines, an XYZ tile
//! pyramid is a straightforward slice/resample of this grid (or a per-tile
//! `MercatorGrid` render) — no new projection math is needed.

use crate::frame::GridGeoref;
use crate::optics::EARTH_RADIUS_M;
use crate::topdown::CloudLayerFrame;

/// The EPSG:3857 (Web Mercator) sphere radius (m). NOT the WRF sphere — see the
/// module's datum note.
pub const WEB_MERCATOR_RADIUS_M: f64 = 6_378_137.0;

/// The Web Mercator latitude clamp (deg): the latitude whose Mercator y equals the
/// projection's square-world half-extent (`atan(sinh(pi))`). Standard EPSG:3857 limit.
pub const MERCATOR_MAX_LAT_DEG: f64 = 85.051_128_779_806_59;

/// Web Mercator FORWARD: geodetic `(lat, lon)` (deg) -> EPSG:3857 `(x, y)` (m).
/// Latitude is clamped to the standard `+/-` [`MERCATOR_MAX_LAT_DEG`]; longitude is
/// used as given (no wrap — a WRF domain never spans a full revolution, and the
/// antimeridian coarse-crop caveat of the camera module applies here identically).
pub fn mercator_forward(lat_deg: f64, lon_deg: f64) -> (f64, f64) {
    let lat = lat_deg.clamp(-MERCATOR_MAX_LAT_DEG, MERCATOR_MAX_LAT_DEG);
    let x = WEB_MERCATOR_RADIUS_M * lon_deg.to_radians();
    let t = (std::f64::consts::FRAC_PI_4 + lat.to_radians() * 0.5).tan();
    let y = WEB_MERCATOR_RADIUS_M * t.ln();
    (x, y)
}

/// Web Mercator INVERSE: EPSG:3857 `(x, y)` (m) -> geodetic `(lat, lon)` (deg).
/// Exact inverse of [`mercator_forward`] within the latitude clamp.
pub fn mercator_inverse(x: f64, y: f64) -> (f64, f64) {
    let lon = (x / WEB_MERCATOR_RADIUS_M).to_degrees();
    let lat_rad = 2.0 * (y / WEB_MERCATOR_RADIUS_M).exp().atan() - std::f64::consts::FRAC_PI_2;
    (lat_rad.to_degrees(), lon)
}

/// A north-up, EPSG:3857-aligned output raster: `nx * ny` pixel CENTRES spanning
/// `[x_min, x_max] x [y_min, y_max]` (Mercator metres), row 0 = north (`y_max`),
/// column 0 = west (`x_min`). Every row is a constant-`y` line and every column a
/// constant-`x` line of EPSG:3857 — the alignment a web-map image source needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MercatorGrid {
    pub nx: usize,
    pub ny: usize,
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
}

impl MercatorGrid {
    /// The EPSG:3857 `(x, y)` of pixel CENTRE `(px, py)`; row 0 = north.
    pub fn pixel_xy(&self, px: usize, py: usize) -> (f64, f64) {
        let fx = if self.nx > 1 {
            px as f64 / (self.nx - 1) as f64
        } else {
            0.5
        };
        let fy = if self.ny > 1 {
            py as f64 / (self.ny - 1) as f64
        } else {
            0.5
        };
        (
            self.x_min + fx * (self.x_max - self.x_min),
            self.y_max - fy * (self.y_max - self.y_min),
        )
    }

    /// The geodetic `(lat, lon)` (deg) of pixel CENTRE `(px, py)`.
    pub fn pixel_lonlat(&self, px: usize, py: usize) -> (f64, f64) {
        let (x, y) = self.pixel_xy(px, py);
        mercator_inverse(x, y)
    }

    /// The FOUR corner `[lon, lat]` pairs of the image in the Mapbox GL `ImageSource`
    /// `coordinates` order: top-left (NW), top-right (NE), bottom-right (SE),
    /// bottom-left (SW). These are the OUTER pixel-centre corners of the grid (the
    /// image spans exactly the grid extent).
    pub fn corners_lonlat(&self) -> [[f64; 2]; 4] {
        let (lat_n, lon_w) = mercator_inverse(self.x_min, self.y_max);
        let (lat_s, lon_e) = mercator_inverse(self.x_max, self.y_min);
        [
            [lon_w, lat_n], // NW (top-left)
            [lon_e, lat_n], // NE (top-right)
            [lon_e, lat_s], // SE (bottom-right)
            [lon_w, lat_s], // SW (bottom-left)
        ]
    }

    /// The `imshow`-style extent `(x_min, x_max, y_min, y_max)` in EPSG:3857 metres
    /// (row 0 = north aligns to `y_max`, i.e. `origin='upper'`).
    pub fn extent_3857(&self) -> [f64; 4] {
        [self.x_min, self.x_max, self.y_min, self.y_max]
    }
}

/// Build the Web-Mercator output grid covering a geodetic bounding box at ~one output
/// pixel per `ground_pitch_m` of native WRF ground resolution. The pixel pitch in
/// Mercator metres is the ground pitch scaled by `sec(centre_lat)` (Mercator stretches
/// with latitude) and by the sphere-radius ratio (3857 metres are on the 6378 km
/// sphere; WRF ground metres on the 6370 km one). Each axis clamps to
/// `[2, max_axis]` — the honest coarsening exception at the cap (the caller logs it).
/// `None` for a degenerate/non-finite bbox.
pub fn mercator_grid_for_bbox(
    lat_min: f64,
    lat_max: f64,
    lon_min: f64,
    lon_max: f64,
    ground_pitch_m: f64,
    max_axis: usize,
) -> Option<MercatorGrid> {
    if !(lat_min.is_finite() && lat_max.is_finite() && lon_min.is_finite() && lon_max.is_finite())
        || lat_max <= lat_min
        || lon_max <= lon_min
        || !(ground_pitch_m.is_finite() && ground_pitch_m > 0.0)
    {
        return None;
    }
    let (x_min, y_min) = mercator_forward(lat_min, lon_min);
    let (x_max, y_max) = mercator_forward(lat_max, lon_max);
    if !(x_max > x_min && y_max > y_min) {
        return None;
    }
    let centre_lat = 0.5 * (lat_min + lat_max);
    let sec = 1.0 / centre_lat.to_radians().cos().max(1.0e-6);
    let pitch = ground_pitch_m * sec * (WEB_MERCATOR_RADIUS_M / EARTH_RADIUS_M);
    let cap = max_axis.max(2);
    let nx = (((x_max - x_min) / pitch).ceil() as usize + 1).clamp(2, cap);
    let ny = (((y_max - y_min) / pitch).ceil() as usize + 1).clamp(2, cap);
    Some(MercatorGrid {
        nx,
        ny,
        x_min,
        x_max,
        y_min,
        y_max,
    })
}

/// The native map raster's SAMPLE-CENTRE grid-index span for a domain of
/// `domain_nx x domain_ny` cells with a zoom-out `margin_frac`:
/// `(i_lo, i_hi, j_lo, j_hi)` — exactly the range [`crate::camera::build_map_raster`]
/// samples (`[-m*(n-1), (n-1)*(1+m)]` per axis). Shared here so the Mercator resample
/// maps a fractional WRF grid index back onto native-raster pixel coordinates with the
/// same convention the raster was built with.
pub fn map_sample_index_span(
    domain_nx: usize,
    domain_ny: usize,
    margin_frac: f64,
) -> (f64, f64, f64, f64) {
    let m = margin_frac.max(0.0);
    let di = (domain_nx.max(2) - 1) as f64;
    let dj = (domain_ny.max(2) - 1) as f64;
    (-m * di, di + m * di, -m * dj, dj + m * dj)
}

/// Resample a NATIVE (Lambert-map-raster) cloud layer onto a Web-Mercator grid:
/// bilinear in PREMULTIPLIED-alpha space for the cloud RGBA (correct alpha handling —
/// see the module note) and bilinear for the shadow channel. Output pixels whose
/// inverse-projected WRF grid index falls outside the native raster's coverage are
/// no-data: cloud `[0, 0, 0, 0]` (fully transparent) and shadow `1.0` (neutral white).
/// Returns `(rgba_premultiplied, shadow)` with `grid.nx * grid.ny` pixels each,
/// row 0 = north.
pub fn reproject_cloud_layer(
    native: &CloudLayerFrame,
    georef: &GridGeoref,
    domain_nx: usize,
    domain_ny: usize,
    margin_frac: f64,
    grid: &MercatorGrid,
) -> (Vec<u8>, Vec<f32>) {
    let n = grid.nx * grid.ny;
    let mut rgba = vec![0u8; n * 4];
    let mut shadow = vec![1.0f32; n];
    if native.nx == 0 || native.ny == 0 {
        return (rgba, shadow);
    }
    let (i_lo, i_hi, j_lo, j_hi) = map_sample_index_span(domain_nx, domain_ny, margin_frac);
    let (i_span, j_span) = (i_hi - i_lo, j_hi - j_lo);
    if i_span <= 0.0 || j_span <= 0.0 {
        return (rgba, shadow);
    }
    for py in 0..grid.ny {
        for px in 0..grid.nx {
            let (lat, lon) = grid.pixel_lonlat(px, py);
            let (fi, fj) = georef.forward(lat, lon);
            if !fi.is_finite() || !fj.is_finite() {
                continue;
            }
            // Fractional native-raster pixel coords (row 0 = north = max j).
            let nx_f = (fi - i_lo) / i_span * (native.nx - 1) as f64;
            let ny_f = (j_hi - fj) / j_span * (native.ny - 1) as f64;
            if !(0.0..=(native.nx - 1) as f64).contains(&nx_f)
                || !(0.0..=(native.ny - 1) as f64).contains(&ny_f)
            {
                continue; // outside the native coverage -> transparent / unshadowed
            }
            let x0 = nx_f.floor() as usize;
            let y0 = ny_f.floor() as usize;
            let x1 = (x0 + 1).min(native.nx - 1);
            let y1 = (y0 + 1).min(native.ny - 1);
            let tx = nx_f - x0 as f64;
            let ty = ny_f - y0 as f64;
            let o = (py * grid.nx + px) * 4;
            let src = &native.rgba_premul;
            for c in 0..4 {
                let s = |xx: usize, yy: usize| src[(yy * native.nx + xx) * 4 + c] as f64;
                let a = s(x0, y0) * (1.0 - tx) + s(x1, y0) * tx;
                let b = s(x0, y1) * (1.0 - tx) + s(x1, y1) * tx;
                rgba[o + c] = (a * (1.0 - ty) + b * ty).round().clamp(0.0, 255.0) as u8;
            }
            let sh = |xx: usize, yy: usize| native.shadow[yy * native.nx + xx] as f64;
            let a = sh(x0, y0) * (1.0 - tx) + sh(x1, y0) * tx;
            let b = sh(x0, y1) * (1.0 - tx) + sh(x1, y1) * tx;
            shadow[py * grid.nx + px] = (a * (1.0 - ty) + b * ty).clamp(0.0, 1.0) as f32;
        }
    }
    (rgba, shadow)
}

/// Convert a PREMULTIPLIED-alpha RGBA buffer to STRAIGHT alpha for PNG / straight-
/// compositing hosts (see the module's alpha-model note): `rgb_straight =
/// clamp(rgb_premul / alpha)`, alpha unchanged; a fully transparent pixel becomes
/// `[0, 0, 0, 0]`. The clamp discards the additive light of a thin-but-bright wisp
/// whose premultiplied color exceeds its alpha (documented approximation; exact for
/// opaque cloud).
pub fn unpremultiply_rgba(rgba_premul: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; rgba_premul.len()];
    for (dst, src) in out.chunks_exact_mut(4).zip(rgba_premul.chunks_exact(4)) {
        let a = src[3];
        if a == 0 {
            continue;
        }
        let af = a as f64 / 255.0;
        for c in 0..3 {
            dst[c] = ((src[c] as f64 / 255.0 / af) * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
        dst[3] = a;
    }
    out
}

/// Quantize the shadow multiply layer to 8-bit grayscale (255 = no shadow / neutral
/// white; toward 0 = fully shadowed) — the PNG delivery of the multiply layer.
pub fn shadow_to_gray(shadow: &[f32]) -> Vec<u8> {
    shadow
        .iter()
        .map(|&s| (s.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect()
}

/// The in-process SYNTHETIC COMPOSITE PROOF: composite the two delivered layers over
/// an arbitrary basemap EXACTLY the way a web-map host would — first the shadow layer
/// as a MULTIPLY blend (`base * shadow`), then the STRAIGHT-alpha cloud layer as a
/// source-over (`cloud*a + base*(1-a)`). `base_rgb` is `n*3`, `cloud_rgba_straight`
/// `n*4`, `shadow` `n` — all on the same grid. Returns the composited `n*3` RGB.
/// This is the registration/appearance proof that needs no Mapbox: if the output
/// looks right over a synthetic basemap, the delivered layers are right.
pub fn composite_over_basemap(
    base_rgb: &[u8],
    cloud_rgba_straight: &[u8],
    shadow: &[f32],
    n: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; n * 3];
    for i in 0..n {
        let s = shadow.get(i).copied().unwrap_or(1.0).clamp(0.0, 1.0) as f64;
        let a = cloud_rgba_straight[i * 4 + 3] as f64 / 255.0;
        for c in 0..3 {
            let base = base_rgb[i * 3 + c] as f64 * s; // multiply blend (the shadow layer)
            let cloud = cloud_rgba_straight[i * 4 + c] as f64;
            out[i * 3 + c] = (cloud * a + base * (1.0 - a)).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// A flat checkerboard basemap (`n = nx*ny` pixels, `nx*ny*3` RGB) in two muted
/// earth tones — the stand-in basemap for the synthetic composite proof (the checker
/// makes the layer registration and the shadow multiply visually obvious).
pub fn checker_basemap(nx: usize, ny: usize, cell_px: usize) -> Vec<u8> {
    let cell = cell_px.max(1);
    let a = [96u8, 124, 88]; // muted green
    let b = [168u8, 158, 128]; // muted tan
    let mut out = vec![0u8; nx * ny * 3];
    for y in 0..ny {
        for x in 0..nx {
            let src = if ((x / cell) + (y / cell)).is_multiple_of(2) {
                a
            } else {
                b
            };
            out[(y * nx + x) * 3..(y * nx + x) * 3 + 3].copy_from_slice(&src);
        }
    }
    out
}

/// The JSON sidecar for a delivered cloud-layer pair: the Mapbox `ImageSource` corner
/// coordinates, the EPSG:3857 extent, dims, alpha mode, and the layer semantics. The
/// `corners_lonlat` order is the Mapbox `coordinates` order (TL, TR, BR, BL).
pub fn cloud_layer_sidecar_json(
    grid: &MercatorGrid,
    cloud_png: &str,
    shadow_png: &str,
    valid_time_iso: &str,
    sun_elev_deg: f64,
    granulation: bool,
) -> String {
    let corners = grid.corners_lonlat();
    let extent = grid.extent_3857();
    serde_json::json!({
        "product": "cloud-layer",
        "crs": "EPSG:3857",
        "proj4": "+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +no_defs",
        "datum_note": "lat/lon computed on the WRF sphere R=6370000 (see web_layer.rs datum note)",
        "width": grid.nx,
        "height": grid.ny,
        "extent_3857": { "x_min": extent[0], "x_max": extent[1], "y_min": extent[2], "y_max": extent[3] },
        "corners_lonlat": corners,
        "cloud_image": { "file": cloud_png, "alpha": "straight", "semantics": "tonemapped cloud radiance; composite source-over" },
        "shadow_image": { "file": shadow_png, "semantics": "ground cloud-shadow; composite as MULTIPLY (255 = no shadow)" },
        "valid_time": valid_time_iso,
        "sun_elev_deg": sun_elev_deg,
        "granulation": granulation,
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn mercator_round_trips_and_is_monotone() {
        for &(lat, lon) in &[
            (0.0, 0.0),
            (45.0, -100.0),
            (-45.0, 140.7),
            (80.0, -179.0),
            (-80.0, 179.0),
            (33.5, -87.2), // the 1974 Super Outbreak neighbourhood
        ] {
            let (x, y) = mercator_forward(lat, lon);
            let (blat, blon) = mercator_inverse(x, y);
            assert!((blat - lat).abs() < 1.0e-9, "lat {lat} -> {blat}");
            assert!((blon - lon).abs() < 1.0e-9, "lon {lon} -> {blon}");
        }
        // y strictly increases with latitude; x with longitude.
        let mut prev_y = f64::NEG_INFINITY;
        for lat in [-85.0, -60.0, -30.0, 0.0, 30.0, 60.0, 85.0] {
            let (_, y) = mercator_forward(lat, 0.0);
            assert!(y > prev_y, "mercator y not monotone at {lat}");
            prev_y = y;
        }
        // Latitude clamps at the standard Web Mercator limit (square world).
        let (_, y_cap) = mercator_forward(89.9, 0.0);
        let (_, y_max) = mercator_forward(MERCATOR_MAX_LAT_DEG, 0.0);
        assert!((y_cap - y_max).abs() < 1.0e-6);
        assert!(
            (y_max - std::f64::consts::PI * WEB_MERCATOR_RADIUS_M).abs() < 1.0,
            "square-world half-extent: {y_max}"
        );
    }

    #[test]
    fn mercator_grid_covers_bbox_north_up_with_ordered_corners() {
        let g = mercator_grid_for_bbox(43.0, 47.0, -103.0, -97.0, 3000.0, 4096).unwrap();
        assert!(g.nx >= 2 && g.ny >= 2 && g.nx <= 4096 && g.ny <= 4096);
        // Pixel (0,0) is the NW corner: max lat, min lon; the far corner is SE.
        let (lat_nw, lon_nw) = g.pixel_lonlat(0, 0);
        let (lat_se, lon_se) = g.pixel_lonlat(g.nx - 1, g.ny - 1);
        assert!((lat_nw - 47.0).abs() < 1.0e-9 && (lon_nw - -103.0).abs() < 1.0e-9);
        assert!((lat_se - 43.0).abs() < 1.0e-9 && (lon_se - -97.0).abs() < 1.0e-9);
        // Corners in the Mapbox order TL, TR, BR, BL; west < east, north > south.
        let c = g.corners_lonlat();
        assert!(c[0][0] < c[1][0] && c[3][0] < c[2][0], "lon order: {c:?}");
        assert!(c[0][1] > c[3][1] && c[1][1] > c[2][1], "lat order: {c:?}");
        assert_eq!(c[0][1], c[1][1], "top corners share the north latitude");
        assert_eq!(c[0][0], c[3][0], "left corners share the west longitude");
        // A tiny cap engages (the honest coarsening exception).
        let capped = mercator_grid_for_bbox(43.0, 47.0, -103.0, -97.0, 30.0, 64).unwrap();
        assert_eq!((capped.nx, capped.ny), (64, 64));
    }

    #[test]
    fn reproject_registers_a_known_pixel_and_preserves_alpha_zero() {
        // A 33x33 native layer over a 33x33-cell domain (native one-pixel-per-cell,
        // margin 0) with ONE opaque red pixel at native (px, py) = (8, 4). Reprojected
        // to Mercator, the output pixel nearest that native sample's lat/lon must be
        // red with high alpha, and pixels far away must stay EXACTLY [0,0,0,0] with
        // shadow 1.0 (alpha-zero + shadow-neutral preservation).
        let (nx, ny) = (33usize, 33usize);
        let georef = test_georef(nx, ny, 3000.0);
        let mut native = CloudLayerFrame {
            nx,
            ny,
            rgba_premul: vec![0u8; nx * ny * 4],
            shadow: vec![1.0f32; nx * ny],
        };
        let (tpx, tpy) = (8usize, 4usize);
        let o = (tpy * nx + tpx) * 4;
        native.rgba_premul[o..o + 4].copy_from_slice(&[255, 0, 0, 255]);
        native.shadow[tpy * nx + tpx] = 0.25;

        // The native sample's geodetic position: row 0 = north (max j), margin 0 ->
        // fi = px, fj = (ny-1) - py.
        let (lat_t, lon_t) = georef
            .inverse(tpx as f64, (ny - 1 - tpy) as f64)
            .expect("inverse");

        // Output grid over the domain's lat/lon bbox at ~native pitch.
        let mut bbox = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for &(i, j) in &[
            (0.0, 0.0),
            ((nx - 1) as f64, 0.0),
            (0.0, (ny - 1) as f64),
            ((nx - 1) as f64, (ny - 1) as f64),
        ] {
            let (la, lo) = georef.inverse(i, j).unwrap();
            bbox = (
                bbox.0.min(la),
                bbox.1.max(la),
                bbox.2.min(lo),
                bbox.3.max(lo),
            );
        }
        let grid = mercator_grid_for_bbox(bbox.0, bbox.1, bbox.2, bbox.3, 3000.0, 4096).unwrap();
        let (rgba, shadow) = reproject_cloud_layer(&native, &georef, nx, ny, 0.0, &grid);

        // The output pixel nearest the target lat/lon (where registration SAYS the
        // red blob must land).
        let (xt, yt) = mercator_forward(lat_t, lon_t);
        let fx = (xt - grid.x_min) / (grid.x_max - grid.x_min);
        let fy = (grid.y_max - yt) / (grid.y_max - grid.y_min);
        let opx = ((fx * (grid.nx - 1) as f64).round() as usize).min(grid.nx - 1);
        let opy = ((fy * (grid.ny - 1) as f64).round() as usize).min(grid.ny - 1);
        // REGISTRATION: the alpha argmax of the whole reprojected image must land
        // within 2 output pixels of the target position (bilinear spread + the
        // ceil-rounded output pitch allow ~1), red-dominant, with real alpha.
        let (mut best, mut bx, mut by) = (0u8, 0usize, 0usize);
        for py in 0..grid.ny {
            for px in 0..grid.nx {
                let a = rgba[(py * grid.nx + px) * 4 + 3];
                if a > best {
                    (best, bx, by) = (a, px, py);
                }
            }
        }
        // The bilinear weight of a single native texel at the argmax output sample is
        // at least ~0.16-0.25 (grid-phase dependent), so the amplitude bound is loose;
        // the REGISTRATION assertion is the distance bound below.
        assert!(best > 30, "reprojected cloud pixel lost: max alpha {best}");
        assert!(
            bx.abs_diff(opx) <= 2 && by.abs_diff(opy) <= 2,
            "registration off: argmax ({bx},{by}) vs expected ({opx},{opy})"
        );
        let bo = (by * grid.nx + bx) * 4;
        assert!(
            rgba[bo] > rgba[bo + 1] && rgba[bo] > rgba[bo + 2],
            "target pixel not red-dominant: {:?}",
            &rgba[bo..bo + 4]
        );
        assert!(
            shadow[by * grid.nx + bx] < 0.95,
            "shadow not carried: {}",
            shadow[by * grid.nx + bx]
        );

        // Everything farther than 3 output pixels from the target is untouched
        // no-data / clear: exact [0,0,0,0] + shadow 1.0.
        for py in 0..grid.ny {
            for px in 0..grid.nx {
                if px.abs_diff(opx) <= 3 && py.abs_diff(opy) <= 3 {
                    continue;
                }
                let o = (py * grid.nx + px) * 4;
                assert_eq!(
                    &rgba[o..o + 4],
                    &[0, 0, 0, 0],
                    "alpha-zero not preserved at ({px},{py})"
                );
                assert_eq!(
                    shadow[py * grid.nx + px],
                    1.0,
                    "shadow not neutral at ({px},{py})"
                );
            }
        }
    }

    #[test]
    fn unpremultiply_is_exact_at_full_alpha_and_clamps() {
        // Full alpha: straight == premultiplied.
        let full = unpremultiply_rgba(&[200, 100, 50, 255]);
        assert_eq!(full, vec![200, 100, 50, 255]);
        // Half alpha: color doubles (128/255 alpha -> 100 -> ~199).
        let half = unpremultiply_rgba(&[100, 50, 25, 128]);
        assert_eq!(half[3], 128);
        assert!((half[0] as i32 - 199).abs() <= 1, "{half:?}");
        // Thin bright wisp: premultiplied color > alpha clamps to 255 (documented).
        let wisp = unpremultiply_rgba(&[60, 60, 60, 20]);
        assert_eq!(&wisp[0..3], &[255, 255, 255]);
        // Zero alpha: exactly zero out.
        assert_eq!(unpremultiply_rgba(&[7, 8, 9, 0]), vec![0, 0, 0, 0]);
    }

    #[test]
    fn composite_over_basemap_matches_the_host_model() {
        // One pixel each: (a) transparent + no shadow -> the basemap verbatim;
        // (b) transparent + half shadow -> the basemap halved (multiply);
        // (c) opaque white cloud -> white regardless of basemap/shadow.
        let base = [100u8, 150, 200, 100, 150, 200, 100, 150, 200];
        let cloud = [
            0u8, 0, 0, 0, // (a)
            0, 0, 0, 0, // (b)
            255, 255, 255, 255, // (c)
        ];
        let shadow = [1.0f32, 0.5, 0.5];
        let out = composite_over_basemap(&base, &cloud, &shadow, 3);
        assert_eq!(&out[0..3], &[100, 150, 200]);
        assert_eq!(&out[3..6], &[50, 75, 100]);
        assert_eq!(&out[6..9], &[255, 255, 255]);
    }

    #[test]
    fn sidecar_json_carries_corners_and_alpha_mode() {
        let grid = mercator_grid_for_bbox(43.0, 47.0, -103.0, -97.0, 3000.0, 512).unwrap();
        let s = cloud_layer_sidecar_json(
            &grid,
            "clouds.png",
            "clouds_shadow.png",
            "1974-04-03T17:20:00Z",
            41.3,
            false,
        );
        let v: serde_json::Value = serde_json::from_str(&s).expect("valid JSON");
        assert_eq!(v["crs"], "EPSG:3857");
        assert_eq!(v["cloud_image"]["alpha"], "straight");
        let corners = v["corners_lonlat"].as_array().unwrap();
        assert_eq!(corners.len(), 4);
        // TL lon < TR lon; TL lat > BL lat.
        let tl = corners[0].as_array().unwrap();
        let tr = corners[1].as_array().unwrap();
        let bl = corners[3].as_array().unwrap();
        assert!(tl[0].as_f64().unwrap() < tr[0].as_f64().unwrap());
        assert!(tl[1].as_f64().unwrap() > bl[1].as_f64().unwrap());
        assert_eq!(v["width"].as_u64().unwrap() as usize, grid.nx);
    }
}
