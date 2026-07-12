//! Geostationary fixed-grid camera (design doc section 1, M1 slice).
//!
//! The output raster *is* an ABI/AHI-style scan-angle grid: each output pixel
//! maps to scan angles `(x, y)` via the CGMS normalized geostationary mapping,
//! and each pixel's lat/lon comes from the inverse of that mapping. For M1's
//! surface-only render we precompute, per output pixel, the geodetic lat/lon and
//! the fractional WRF grid index `(i, j)` on the CPU and upload them as lookup
//! textures (design section 1 explicitly allows this for M1; the per-step ECEF
//! ray march arrives with volumetrics in M4).
//!
//! The default remains owner decision 5 (spherical earth, WRF `R = 6_370_000 m`):
//! scan navigation and model physics share the WRF sphere.  v0.1.7 adds an
//! explicit [`GeoNavigation::GoesRAbiFixedGrid`] alternative: official GOES-R ABI
//! sweep-x/ellipsoid navigation places sensor raster pixels, then their geodetic
//! coordinates are mapped back to the unchanged spherical WRF projection and ray
//! camera.  This separates real-ABI registration from model-volume geometry and
//! does not claim exact ABI radiometry.
//!
//! ---
//! ATTRIBUTION. The scan-angle math below is PORTED, with the changes noted, from
//! BowEcho (the sibling rusty-weather app; rev pinned in Cargo.toml):
//!   - [`lat_lon_to_scan_angles`] from
//!     `crates/app_ui/src/sat_window.rs::ahi_lat_lon_to_scan_angles` (the CGMS
//!     normalized geostationary forward incl. the GOES-R PUG section 5.1.2.8.1
//!     visibility condition).
//!   - [`scan_angles_to_lat_lon`] from
//!     `crates/app_ui/src/sat_worker.rs::ahi_scan_angles_to_lat_lon` (the exact
//!     inverse). We port this rather than call `rw_sat::geostationary` because the
//!     pinned `rw_sat` sweep=y inverse has a live transpose bug (docs/
//!     bowecho-precedents.md section 8); the round-trip tripwire test guards it.
//!   - [`scan_angle_rect`] + the 9-sample-point pattern from
//!     `sat_window.rs::window_scan_angle_rect` / `SatNativeWindow::sample_points`.
//!   - sub-lon presets from `sat_window.rs` (GOES-East -75.2, GOES-West -137.0,
//!     Himawari 140.7).
//!
//! The historical path uses one CGMS sweep-y pair for all three presets.  The
//! opt-in ABI path is GOES-East/West only and uses sweep-x; the distinction is
//! material for registration (several ABI pixels over CONUS), not cosmetic.

use crate::atmosphere::ATMOSPHERE_HEIGHT_M;
use crate::frame::GridGeoref;
use crate::optics::EARTH_RADIUS_M as R_EARTH;

/// Geostationary orbit radius from earth center (m). The nominal geosynchronous
/// radius; with the spherical `R_EARTH` this gives a perspective height of
/// ~35_794_000 m (cf. the ellipsoid AHI nominal 35_785_863 m).
pub const GEO_ORBIT_RADIUS_M: f64 = 42_164_000.0;

/// Perspective-point height above the (spherical) surface used by the M1 camera.
pub const PERSPECTIVE_HEIGHT_M: f64 = GEO_ORBIT_RADIUS_M - R_EARTH;

/// GOES-R ABI fixed-grid ellipsoid metadata published in the GOES-R Product
/// Definition and User's Guide and carried by the `goes_imager_projection`
/// variable in ABI L1b/L2 files.  These constants describe navigation only;
/// SimSat's WRF projection and volume/ray physics deliberately remain on the
/// existing 6,370 km model sphere.
pub const GOES_R_ABI_PERSPECTIVE_HEIGHT_M: f64 = 35_786_023.0;
pub const GOES_R_ABI_SEMI_MAJOR_AXIS_M: f64 = 6_378_137.0;
pub const GOES_R_ABI_SEMI_MINOR_AXIS_M: f64 = 6_356_752.314_14;
/// Nominal longitude in current GOES-East (GOES-19) ABI metadata.
pub const GOES_R_EAST_SUB_LON_DEG: f64 = -75.0;
/// Nominal longitude in current GOES-West (GOES-18) ABI metadata.
pub const GOES_R_WEST_SUB_LON_DEG: f64 = -137.0;

/// GOES-East sub-satellite longitude (deg). `sat_window.rs:224`.
pub const GOES_EAST_SUB_LON_DEG: f64 = -75.2;
/// GOES-West sub-satellite longitude (deg). `sat_window.rs:225`.
pub const GOES_WEST_SUB_LON_DEG: f64 = -137.0;
/// Himawari-8/9 sub-satellite longitude (deg). `sat_window.rs:218`.
pub const HIMAWARI_SUB_LON_DEG: f64 = 140.7;

/// ABI visible (1 km) class output pixel pitch (radians): 28 urad. This was the
/// M1 fixed default; the owner native-resolution fix makes [`ResolutionMode::Native`]
/// the studio default so the render samples the WRF grid at its OWN resolution
/// (one output pixel per cell) instead of this coarser fixed 1 km scan pitch, which
/// undersamples any finer (250-500 m) WRF grid. Retained for the
/// [`ResolutionMode::Abi1km`] option.
pub const VISIBLE_PITCH_RAD: f64 = 28.0e-6;
/// ABI IR (2 km) class output pixel pitch (radians): 56 urad. Retained for the
/// [`ResolutionMode::Abi2km`] option.
pub const IR_PITCH_RAD: f64 = 56.0e-6;
/// Number of samples on one axis of the ABI 2 km full-disk fixed grid.
///
/// The even sample count is important: the sub-satellite point is the corner shared
/// by four pixels, never a pixel centre.  The two centres nearest zero are therefore
/// at `-28` and `+28` microradians.
pub const ABI_2KM_FULL_DISK_AXIS: usize = 5424;
/// Smallest signed half-pitch lattice index in the ABI 2 km full disk.
pub const ABI_2KM_LATTICE_INDEX_MIN: i32 = -(ABI_2KM_FULL_DISK_AXIS as i32 / 2);
/// Largest signed half-pitch lattice index in the ABI 2 km full disk.
pub const ABI_2KM_LATTICE_INDEX_MAX: i32 = ABI_2KM_FULL_DISK_AXIS as i32 / 2 - 1;
/// Hard per-axis raster cap (design section 1 / player 4096^2 cap).
pub const MAX_AXIS: usize = 4096;

/// The three v1 satellites (owner decision 3), selectable in the studio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SatellitePreset {
    GoesEast,
    GoesWest,
    Himawari,
}

/// Geostationary sensor-raster navigation contract.
///
/// `ModelSphere` is the unchanged, model-consistent default.  `GoesRAbiFixedGrid`
/// uses the official ABI ellipsoid and sweep-x equations to place raster pixel
/// centres, then converts each resulting geodetic point back to a ray on SimSat's
/// existing spherical model geometry.  The latter improves registration against
/// real ABI imagery without pretending the WRF volume itself uses an ellipsoid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GeoNavigation {
    #[default]
    ModelSphere,
    GoesRAbiFixedGrid,
}

impl GeoNavigation {
    pub const ALL: [Self; 2] = [Self::ModelSphere, Self::GoesRAbiFixedGrid];

    pub fn label(self) -> &'static str {
        match self {
            Self::ModelSphere => "Model sphere (default)",
            Self::GoesRAbiFixedGrid => "GOES-R ABI ellipsoid (exact navigation)",
        }
    }

    pub fn slug(self) -> &'static str {
        match self {
            Self::ModelSphere => "model-sphere",
            Self::GoesRAbiFixedGrid => "goes-r-abi",
        }
    }
}

/// Numeric geometry recorded with a geostationary render.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoNavigationGeometry {
    pub navigation: GeoNavigation,
    pub perspective_point_height_m: f64,
    pub semi_major_axis_m: f64,
    pub semi_minor_axis_m: f64,
    pub sub_lon_deg: f64,
    /// Longitude of the existing spherical physics camera.  GOES-19 products
    /// deliberately distinguish their -75.0 fixed-grid origin from the nominal
    /// platform subpoint near -75.2.
    pub model_sub_lon_deg: f64,
    /// CF/ABI sweep-axis token (`"x"` for ABI, `"y"` for the historical
    /// model-sphere camera).
    pub sweep_angle_axis: &'static str,
}

impl SatellitePreset {
    /// All presets in UI order.
    pub const ALL: [SatellitePreset; 3] = [
        SatellitePreset::GoesEast,
        SatellitePreset::GoesWest,
        SatellitePreset::Himawari,
    ];

    /// Sub-satellite longitude (deg).
    pub fn sub_lon_deg(self) -> f64 {
        match self {
            Self::GoesEast => GOES_EAST_SUB_LON_DEG,
            Self::GoesWest => GOES_WEST_SUB_LON_DEG,
            Self::Himawari => HIMAWARI_SUB_LON_DEG,
        }
    }

    /// Human-readable label for the picker.
    pub fn label(self) -> &'static str {
        match self {
            Self::GoesEast => "GOES-East (-75.2)",
            Self::GoesWest => "GOES-West (-137.0)",
            Self::Himawari => "Himawari (140.7)",
        }
    }

    /// Short store-token slug (ascii-alnum) for the run name.
    pub fn slug(self) -> &'static str {
        match self {
            Self::GoesEast => "goese",
            Self::GoesWest => "goesw",
            Self::Himawari => "himawari",
        }
    }
}

/// The output VIEW MODE: the physically-authentic from-space geostationary product
/// (the existing M1..M6 scan-angle raster) or the top-down map-registered product
/// added for the WRF-Runner integration.
///
/// Design-doc section 1 rejected a bespoke top-down camera FOR THE STANDALONE
/// PRODUCT (redundant with BowEcho's map layer). The owner later chose to ship a
/// top-down product for the WRF-Runner integration (that suite is entirely
/// top-down Lambert map plots on the same spherical R = 6.37e6 the WRF georeference
/// uses, so a top-down simulated-visible/IR registers pixel-for-pixel with its other
/// field plots), so it is added here as an integration/output mode. It is a SYNTHETIC
/// near-nadir view (each output pixel is a straight-down ray at that map location),
/// NOT a specific satellite's oblique view — [`ViewMode::Geostationary`] remains the
/// physically-authentic from-space product.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewMode {
    /// From-space geostationary scan-angle raster (the existing product).
    Geostationary,
    /// Top-down, north-up, map-registered near-nadir raster over the WRF domain's own
    /// Lambert extent (the new integration product; see [`build_map_raster`]).
    TopDownMap,
}

impl ViewMode {
    /// Both modes in UI order (Geostationary first — the default).
    pub const ALL: [ViewMode; 2] = [ViewMode::Geostationary, ViewMode::TopDownMap];

    /// Human-readable label for the picker.
    pub fn label(self) -> &'static str {
        match self {
            Self::Geostationary => "Geostationary (from space)",
            Self::TopDownMap => "Top-down map",
        }
    }

    /// Short slug (ascii) for logs / CLI.
    pub fn slug(self) -> &'static str {
        match self {
            Self::Geostationary => "geo",
            Self::TopDownMap => "topdown",
        }
    }
}

/// The output-raster resolution policy (owner native-resolution fix).
///
/// The M1 camera always built the scan-angle raster at a FIXED ABI-class pixel
/// pitch (28 urad = "1 km"), which UNDERSAMPLES a finer WRF grid: a 250-500 m
/// domain has more grid cells per axis than the fixed-pitch raster has pixels, so
/// the render threw away the owner's native resolution and the studio then
/// magnified the small frame with hard pixels. `Native` (the studio default) instead
/// sizes the raster to the WRF grid — approximately one output pixel per WRF cell
/// across the domain, so full native detail is preserved and never oversampled
/// beyond the data. The fixed ABI pitches remain selectable as options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionMode {
    /// One output pixel per WRF grid cell (the studio default — full native res).
    Native,
    /// Fixed ABI visible 1 km class ([`VISIBLE_PITCH_RAD`], 28 urad).
    Abi1km,
    /// Fixed ABI IR 2 km class ([`IR_PITCH_RAD`], 56 urad).
    Abi2km,
}

impl ResolutionMode {
    /// All modes in UI order (Native first — the default).
    pub const ALL: [ResolutionMode; 3] = [
        ResolutionMode::Native,
        ResolutionMode::Abi1km,
        ResolutionMode::Abi2km,
    ];

    /// Short label for the picker + status bar. The explicit source-grid wording keeps
    /// `Native` from being mistaken for the highest available output resolution.
    pub fn label(self) -> &'static str {
        match self {
            Self::Native => "Model native (one pixel per source grid cell)",
            Self::Abi1km => "ABI 1 km (28 urad)",
            Self::Abi2km => "ABI 2 km (56 urad)",
        }
    }

    /// The fixed scan-angle pitch (rad/pixel) for the ABI modes; `None` for
    /// `Native`, whose pitch is derived per-domain from the WRF cell count.
    pub fn fixed_pitch_rad(self) -> Option<f64> {
        match self {
            Self::Native => None,
            Self::Abi1km => Some(VISIBLE_PITCH_RAD),
            Self::Abi2km => Some(IR_PITCH_RAD),
        }
    }

    /// Build the [`ScanGrid`] for this mode over a domain's scan-angle `rect`.
    /// `Native` targets one output pixel per WRF cell (`native_nx`/`native_ny`); the
    /// ABI modes use their fixed pitch. Both clamp each axis to `<= max_axis`.
    pub fn scan_grid(
        self,
        rect: ScanAngleRect,
        native_nx: usize,
        native_ny: usize,
        max_axis: usize,
    ) -> ScanGrid {
        match self.fixed_pitch_rad() {
            Some(pitch) => ScanGrid::from_rect(rect, pitch, max_axis),
            None => ScanGrid::native(rect, native_nx, native_ny, max_axis),
        }
    }
}

/// CGMS normalized geostationary FORWARD: geodetic `(lat, lon)` -> scan angles
/// `(x, y)` in radians, or `None` when the point faces away from the satellite.
///
/// PORTED from `sat_window.rs::ahi_lat_lon_to_scan_angles` (sweep=y convention:
/// `x = atan(-sy/sx)`, `y = atan(sz / hypot(sx, sy))`), including the GOES-R PUG
/// section 5.1.2.8.1 visibility gate. Unchanged except for naming.
pub fn lat_lon_to_scan_angles(
    perspective_point_height_m: f64,
    semi_major_axis_m: f64,
    semi_minor_axis_m: f64,
    lon0_deg: f64,
    lat_deg: f64,
    lon_deg: f64,
) -> Option<(f64, f64)> {
    let h = perspective_point_height_m + semi_major_axis_m;
    let a = semi_major_axis_m;
    let b = semi_minor_axis_m;
    if !(h.is_finite() && lon0_deg.is_finite() && lat_deg.is_finite() && lon_deg.is_finite())
        || h <= 0.0
        || a <= 0.0
        || b <= 0.0
    {
        return None;
    }

    let lat = lat_deg.to_radians();
    let lon_delta = (lon_deg - lon0_deg).to_radians();
    let pol_by_eq = (b * b) / (a * a);
    let geocentric_lat = (pol_by_eq * lat.tan()).atan();
    let radius = b / (1.0 - (1.0 - pol_by_eq) * geocentric_lat.cos().powi(2)).sqrt();

    // Satellite-relative components (x toward the earth center, y east, z north).
    let sx = h - radius * geocentric_lat.cos() * lon_delta.cos();
    let sy = -radius * geocentric_lat.cos() * lon_delta.sin();
    let sz = radius * geocentric_lat.sin();

    // GOES-R PUG visibility condition: the point must face the satellite.
    if h * (h - sx) < sy * sy + sz * sz / pol_by_eq {
        return None;
    }

    let x = (-sy / sx).atan();
    let y = (sz / sx.hypot(sy)).atan();
    (x.is_finite() && y.is_finite()).then_some((x, y))
}

/// CGMS normalized geostationary INVERSE: scan angles `(x, y)` (radians) ->
/// geodetic `(lat, lon)` (deg), or `None` when the ray looks past the limb.
///
/// PORTED from `sat_worker.rs::ahi_scan_angles_to_lat_lon` (the exact inverse of
/// the forward above: `Vy = tan(x)`, `Vz = tan(y) * hypot(1, Vy)`, then the
/// ellipsoid-intersection quadratic). We port this rather than lean on the pinned
/// `rw_sat` sweep=y inverse, which has a live transpose bug (digest section 8).
/// Unchanged except for naming and returning f64.
pub fn scan_angles_to_lat_lon(
    perspective_point_height_m: f64,
    semi_major_axis_m: f64,
    semi_minor_axis_m: f64,
    lon0_deg: f64,
    x_rad: f64,
    y_rad: f64,
) -> Option<(f64, f64)> {
    let h = perspective_point_height_m + semi_major_axis_m;
    let a = semi_major_axis_m;
    let b = semi_minor_axis_m;
    if !h.is_finite() || !lon0_deg.is_finite() || !x_rad.is_finite() || !y_rad.is_finite() {
        return None;
    }
    if h <= 0.0 || a <= 0.0 || b <= 0.0 {
        return None;
    }

    let v_y = x_rad.tan();
    let v_z = y_rad.tan() * 1.0_f64.hypot(v_y);
    let eq_to_pol = (a * a) / (b * b);

    let a_var = 1.0 + v_y * v_y + eq_to_pol * v_z * v_z;
    let b_var = -2.0 * h;
    let c_var = h * h - a * a;
    let discriminant = b_var * b_var - 4.0 * a_var * c_var;
    if discriminant < 0.0 {
        return None; // looking past the limb
    }

    let r_s = (-b_var - discriminant.sqrt()) / (2.0 * a_var);
    if !r_s.is_finite() || r_s <= 0.0 {
        return None;
    }

    let s_x = r_s;
    let s_y = -r_s * v_y;
    let s_z = r_s * v_z;

    let latitude = (eq_to_pol * (s_z / (h - s_x).hypot(s_y))).atan();
    let longitude = lon0_deg.to_radians() - (s_y / (h - s_x)).atan();
    let lat_deg = latitude.to_degrees();
    let mut lon_deg = (longitude.to_degrees() + 180.0).rem_euclid(360.0) - 180.0;
    if lon_deg == -180.0 {
        lon_deg = 180.0;
    }
    if !lat_deg.is_finite() || !lon_deg.is_finite() {
        return None;
    }
    Some((lat_deg, lon_deg))
}

/// Official GOES-R ABI sweep-x fixed-grid forward navigation.
///
/// This is the scalar Rust twin of `scripts/fetch-goes-abi-reference.py`'s
/// independently pyproj-checked NumPy reference and the equations documented by
/// NOAA/NESDIS/STAR.  It intentionally lives beside, rather than replaces, the
/// historical sweep-y model-sphere transform above.
pub fn goes_r_lat_lon_to_scan_angles(
    perspective_point_height_m: f64,
    semi_major_axis_m: f64,
    semi_minor_axis_m: f64,
    lon0_deg: f64,
    lat_deg: f64,
    lon_deg: f64,
) -> Option<(f64, f64)> {
    let h = perspective_point_height_m + semi_major_axis_m;
    let a = semi_major_axis_m;
    let b = semi_minor_axis_m;
    if !(h.is_finite()
        && a.is_finite()
        && b.is_finite()
        && lon0_deg.is_finite()
        && lat_deg.is_finite()
        && lon_deg.is_finite())
        || h <= 0.0
        || a <= 0.0
        || b <= 0.0
    {
        return None;
    }

    let lat = lat_deg.to_radians();
    let lon_delta = (lon_deg - lon0_deg).to_radians();
    let b2_over_a2 = (b * b) / (a * a);
    let geocentric_lat = (b2_over_a2 * lat.tan()).atan();
    let eccentricity_sq = (a * a - b * b) / (a * a);
    let radius = b / (1.0 - eccentricity_sq * geocentric_lat.cos().powi(2)).sqrt();
    let earth_x = radius * geocentric_lat.cos() * lon_delta.cos();
    let earth_y = radius * geocentric_lat.cos() * lon_delta.sin();
    let earth_z = radius * geocentric_lat.sin();

    // GOES-R PUG visibility condition for an ellipsoidal limb.
    if h * earth_x < earth_x * earth_x + earth_y * earth_y + (a * a / (b * b)) * earth_z * earth_z {
        return None;
    }

    let ray_x = h - earth_x;
    let ray_y = -earth_y;
    let ray_z = earth_z;
    let ray_length = (ray_x * ray_x + ray_y * ray_y + ray_z * ray_z).sqrt();
    if !ray_length.is_finite() || ray_length <= 0.0 {
        return None;
    }
    // ABI sweep=x: E/W is the angle about the full ray norm; N/S is the
    // elevation about the satellite-to-earth-centre axis.
    let x = (-ray_y / ray_length).clamp(-1.0, 1.0).asin();
    let y = ray_z.atan2(ray_x);
    (x.is_finite() && y.is_finite()).then_some((x, y))
}

/// Official GOES-R ABI sweep-x fixed-grid inverse navigation.
///
/// Returns geodetic latitude/longitude on the supplied ellipsoid.  A negative
/// intersection discriminant is the exact off-disk/limb test.
pub fn goes_r_scan_angles_to_lat_lon(
    perspective_point_height_m: f64,
    semi_major_axis_m: f64,
    semi_minor_axis_m: f64,
    lon0_deg: f64,
    x_rad: f64,
    y_rad: f64,
) -> Option<(f64, f64)> {
    let h = perspective_point_height_m + semi_major_axis_m;
    let a = semi_major_axis_m;
    let b = semi_minor_axis_m;
    if !(h.is_finite()
        && a.is_finite()
        && b.is_finite()
        && lon0_deg.is_finite()
        && x_rad.is_finite()
        && y_rad.is_finite())
        || h <= 0.0
        || a <= 0.0
        || b <= 0.0
    {
        return None;
    }

    let (sin_x, cos_x) = x_rad.sin_cos();
    let (sin_y, cos_y) = y_rad.sin_cos();
    let a2_over_b2 = (a * a) / (b * b);
    let qa = sin_x * sin_x + cos_x * cos_x * (cos_y * cos_y + a2_over_b2 * sin_y * sin_y);
    let qb = -2.0 * h * cos_x * cos_y;
    let qc = h * h - a * a;
    let mut discriminant = qb * qb - 4.0 * qa * qc;
    if !discriminant.is_finite() {
        return None;
    }
    if discriminant < 0.0 {
        let roundoff = 64.0 * f64::EPSILON * (qb * qb + (4.0 * qa * qc).abs());
        if discriminant >= -roundoff {
            discriminant = 0.0;
        } else {
            return None;
        }
    }
    let r_s = (-qb - discriminant.sqrt()) / (2.0 * qa);
    if !r_s.is_finite() || r_s <= 0.0 {
        return None;
    }
    let s_x = r_s * cos_x * cos_y;
    let s_y = -r_s * sin_x;
    let s_z = r_s * cos_x * sin_y;
    let latitude = (a2_over_b2 * s_z / (h - s_x).hypot(s_y)).atan();
    let longitude = lon0_deg.to_radians() - s_y.atan2(h - s_x);
    let lat_deg = latitude.to_degrees();
    let mut lon_deg = (longitude.to_degrees() + 180.0).rem_euclid(360.0) - 180.0;
    if lon_deg == -180.0 {
        lon_deg = 180.0;
    }
    (lat_deg.is_finite() && lon_deg.is_finite()).then_some((lat_deg, lon_deg))
}

/// A geostationary camera at a sub-lon, on the spherical earth (owner decision 5).
/// Thin wrapper binding the ported forward/inverse to the M1 spherical constants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoCamera {
    pub sub_lon_deg: f64,
    pub model_sub_lon_deg: f64,
    pub navigation: GeoNavigation,
    pub perspective_point_height_m: f64,
    pub semi_major_axis_m: f64,
    pub semi_minor_axis_m: f64,
}

impl GeoCamera {
    /// Camera for a preset.
    pub fn new(preset: SatellitePreset) -> Self {
        Self {
            sub_lon_deg: preset.sub_lon_deg(),
            model_sub_lon_deg: preset.sub_lon_deg(),
            navigation: GeoNavigation::ModelSphere,
            perspective_point_height_m: PERSPECTIVE_HEIGHT_M,
            semi_major_axis_m: R_EARTH,
            semi_minor_axis_m: R_EARTH,
        }
    }

    /// Build a camera for an explicit navigation contract.  The ABI ellipsoid is
    /// meaningful only for the two GOES presets; Himawari retains its separate
    /// historical sweep-y geometry.
    pub fn for_navigation(
        preset: SatellitePreset,
        navigation: GeoNavigation,
    ) -> Result<Self, &'static str> {
        match navigation {
            GeoNavigation::ModelSphere => Ok(Self::new(preset)),
            GeoNavigation::GoesRAbiFixedGrid => {
                let sub_lon_deg = match preset {
                    SatellitePreset::GoesEast => GOES_R_EAST_SUB_LON_DEG,
                    SatellitePreset::GoesWest => GOES_R_WEST_SUB_LON_DEG,
                    SatellitePreset::Himawari => {
                        return Err("GOES-R ABI ellipsoid navigation is unavailable for Himawari");
                    }
                };
                Ok(Self {
                    sub_lon_deg,
                    model_sub_lon_deg: preset.sub_lon_deg(),
                    navigation,
                    perspective_point_height_m: GOES_R_ABI_PERSPECTIVE_HEIGHT_M,
                    semi_major_axis_m: GOES_R_ABI_SEMI_MAJOR_AXIS_M,
                    semi_minor_axis_m: GOES_R_ABI_SEMI_MINOR_AXIS_M,
                })
            }
        }
    }

    pub fn geometry(&self) -> GeoNavigationGeometry {
        GeoNavigationGeometry {
            navigation: self.navigation,
            perspective_point_height_m: self.perspective_point_height_m,
            semi_major_axis_m: self.semi_major_axis_m,
            semi_minor_axis_m: self.semi_minor_axis_m,
            sub_lon_deg: self.sub_lon_deg,
            model_sub_lon_deg: self.model_sub_lon_deg,
            sweep_angle_axis: match self.navigation {
                GeoNavigation::ModelSphere => "y",
                GeoNavigation::GoesRAbiFixedGrid => "x",
            },
        }
    }

    /// The existing spherical physics camera placed at this sensor's subpoint.
    pub fn model_sphere(&self) -> Self {
        Self {
            sub_lon_deg: self.model_sub_lon_deg,
            model_sub_lon_deg: self.model_sub_lon_deg,
            navigation: GeoNavigation::ModelSphere,
            perspective_point_height_m: PERSPECTIVE_HEIGHT_M,
            semi_major_axis_m: R_EARTH,
            semi_minor_axis_m: R_EARTH,
        }
    }

    /// Forward: geodetic `(lat, lon)` -> scan angles `(x, y)` (rad), spherical.
    pub fn forward(&self, lat_deg: f64, lon_deg: f64) -> Option<(f64, f64)> {
        match self.navigation {
            GeoNavigation::ModelSphere => lat_lon_to_scan_angles(
                self.perspective_point_height_m,
                self.semi_major_axis_m,
                self.semi_minor_axis_m,
                self.sub_lon_deg,
                lat_deg,
                lon_deg,
            ),
            GeoNavigation::GoesRAbiFixedGrid => goes_r_lat_lon_to_scan_angles(
                self.perspective_point_height_m,
                self.semi_major_axis_m,
                self.semi_minor_axis_m,
                self.sub_lon_deg,
                lat_deg,
                lon_deg,
            ),
        }
    }

    /// Inverse: scan angles `(x, y)` (rad) -> geodetic `(lat, lon)`, spherical.
    pub fn inverse(&self, x_rad: f64, y_rad: f64) -> Option<(f64, f64)> {
        match self.navigation {
            GeoNavigation::ModelSphere => scan_angles_to_lat_lon(
                self.perspective_point_height_m,
                self.semi_major_axis_m,
                self.semi_minor_axis_m,
                self.sub_lon_deg,
                x_rad,
                y_rad,
            ),
            GeoNavigation::GoesRAbiFixedGrid => goes_r_scan_angles_to_lat_lon(
                self.perspective_point_height_m,
                self.semi_major_axis_m,
                self.semi_minor_axis_m,
                self.sub_lon_deg,
                x_rad,
                y_rad,
            ),
        }
    }
}

/// A scan-angle bounding rectangle (radians).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScanAngleRect {
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
}

/// Project every sampled point through `forward` and take the bounding rect;
/// `None` if ANY sample fails to project (a domain past the limb cannot be
/// cropped honestly). PORTED from `sat_window.rs::window_scan_angle_rect`.
pub fn scan_angle_rect(
    samples: &[(f64, f64)],
    mut forward: impl FnMut(f64, f64) -> Option<(f64, f64)>,
) -> Option<ScanAngleRect> {
    let mut rect: Option<ScanAngleRect> = None;
    for &(lat, lon) in samples {
        let (x, y) = forward(lat, lon)?;
        if !(x.is_finite() && y.is_finite()) {
            return None;
        }
        rect = Some(match rect {
            None => ScanAngleRect {
                x_min: x,
                x_max: x,
                y_min: y,
                y_max: y,
            },
            Some(r) => ScanAngleRect {
                x_min: r.x_min.min(x),
                x_max: r.x_max.max(x),
                y_min: r.y_min.min(y),
                y_max: r.y_max.max(y),
            },
        });
    }
    rect
}

/// The 9 georeferenced sample points (corners + edge mids + center, in lat/lon)
/// whose scan angles bound a WRF domain crop. PORTED pattern from
/// `SatNativeWindow::sample_points`, indexed on the WRF grid instead of a window.
pub fn domain_sample_points(georef: &GridGeoref, nx: usize, ny: usize) -> Vec<(f64, f64)> {
    if nx < 2 || ny < 2 {
        return Vec::new();
    }
    let (mx, my) = ((nx - 1) as f64 / 2.0, (ny - 1) as f64 / 2.0);
    let (xi, yi) = ((nx - 1) as f64, (ny - 1) as f64);
    let idx = [
        (0.0, 0.0),
        (xi, 0.0),
        (0.0, yi),
        (xi, yi),
        (mx, 0.0),
        (0.0, my),
        (xi, my),
        (mx, yi),
        (mx, my),
    ];
    idx.iter()
        .filter_map(|&(i, j)| georef.inverse(i, j))
        .collect()
}

/// The scan-angle raster grid derived from a domain's scan-angle bbox at a pitch.
///
/// Row 0 is the northernmost (max scan `y`), the GOES storage convention. Pixel
/// `(px, py)` maps to scan `(x_min + px*pitch_x, y_max - py*pitch_y)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScanGrid {
    pub nx: usize,
    pub ny: usize,
    pub x_min: f64,
    pub y_max: f64,
    pub pitch_x: f64,
    pub pitch_y: f64,
}

impl ScanGrid {
    /// Build a raster covering `rect` at `pitch` (rad/pixel), clamping each axis
    /// to `<= max_axis` (coarsening the pitch if needed). `max_axis` is also
    /// clamped to [`MAX_AXIS`].
    pub fn from_rect(rect: ScanAngleRect, pitch: f64, max_axis: usize) -> Self {
        let cap = max_axis.clamp(2, MAX_AXIS);
        let span_x = (rect.x_max - rect.x_min).max(0.0);
        let span_y = (rect.y_max - rect.y_min).max(0.0);
        let raw_nx = ((span_x / pitch).ceil() as usize + 1).max(2);
        let raw_ny = ((span_y / pitch).ceil() as usize + 1).max(2);
        let nx = raw_nx.min(cap);
        let ny = raw_ny.min(cap);
        // If the axis was capped, coarsen the pitch so the raster still spans the
        // full rect (one pixel at each edge).
        let pitch_x = if nx > 1 {
            span_x / (nx - 1) as f64
        } else {
            pitch
        };
        let pitch_y = if ny > 1 {
            span_y / (ny - 1) as f64
        } else {
            pitch
        };
        Self {
            nx,
            ny,
            x_min: rect.x_min,
            y_max: rect.y_max,
            pitch_x: if pitch_x > 0.0 { pitch_x } else { pitch },
            pitch_y: if pitch_y > 0.0 { pitch_y } else { pitch },
        }
    }

    /// Build an exact crop of the global ABI 2 km fixed-grid lattice.
    ///
    /// Pixel centres obey the PUG/ABI coordinate contract
    /// `angle = (index + 1/2) * 56 urad`.  Thus zero scan angle is the corner of
    /// four pixels and the nearest centres are at +/-28 urad.  Bounds are snapped
    /// outward to the first global centres that bracket `rect`; row 0 remains north.
    /// No endpoint fitting or pitch coarsening is permitted.  `None` means the
    /// requested crop falls outside the 5424-sample full-disk lattice or exceeds the
    /// caller's axis cap, because either fallback would cease to be an ABI lattice.
    pub fn abi_2km_global_lattice(rect: ScanAngleRect, max_axis: usize) -> Option<Self> {
        if !(rect.x_min.is_finite()
            && rect.x_max.is_finite()
            && rect.y_min.is_finite()
            && rect.y_max.is_finite())
            || rect.x_min > rect.x_max
            || rect.y_min > rect.y_max
        {
            return None;
        }

        // Coordinates obtained from a NetCDF f32 axis can land a few ulps either
        // side of an exact half-pitch value.  A 1e-9-index tolerance (5.6e-14 rad)
        // makes an analytically identical crop deterministic without moving any
        // genuinely non-lattice domain bound to another pixel.
        const INDEX_TOLERANCE: f64 = 1.0e-9;
        let lower_index =
            |angle: f64| (angle / IR_PITCH_RAD - 0.5 + INDEX_TOLERANCE).floor() as i32;
        let upper_index = |angle: f64| (angle / IR_PITCH_RAD - 0.5 - INDEX_TOLERANCE).ceil() as i32;

        let x_index_min = lower_index(rect.x_min);
        let x_index_max = upper_index(rect.x_max);
        let y_index_min = lower_index(rect.y_min);
        let y_index_max = upper_index(rect.y_max);
        if x_index_min < ABI_2KM_LATTICE_INDEX_MIN
            || x_index_max > ABI_2KM_LATTICE_INDEX_MAX
            || y_index_min < ABI_2KM_LATTICE_INDEX_MIN
            || y_index_max > ABI_2KM_LATTICE_INDEX_MAX
        {
            return None;
        }

        let nx = (x_index_max - x_index_min + 1) as usize;
        let ny = (y_index_max - y_index_min + 1) as usize;
        let cap = max_axis.clamp(2, MAX_AXIS);
        if nx < 2 || ny < 2 || nx > cap || ny > cap {
            return None;
        }
        Some(Self {
            nx,
            ny,
            x_min: (x_index_min as f64 + 0.5) * IR_PITCH_RAD,
            y_max: (y_index_max as f64 + 0.5) * IR_PITCH_RAD,
            pitch_x: IR_PITCH_RAD,
            pitch_y: IR_PITCH_RAD,
        })
    }

    /// Signed global half-pitch indices `(x_min, x_max, y_min, y_max)` when this
    /// grid is an exact ABI 2 km lattice crop; `None` for native/legacy fitted grids.
    pub fn abi_2km_global_indices(&self) -> Option<(i32, i32, i32, i32)> {
        let index = |angle: f64| -> Option<i32> {
            let raw = angle / IR_PITCH_RAD - 0.5;
            let rounded = raw.round();
            ((raw - rounded).abs() <= 1.0e-8).then_some(rounded as i32)
        };
        if (self.pitch_x - IR_PITCH_RAD).abs() > 1.0e-15
            || (self.pitch_y - IR_PITCH_RAD).abs() > 1.0e-15
        {
            return None;
        }
        let x_min = index(self.x_min)?;
        let y_max = index(self.y_max)?;
        let x_max = x_min.checked_add(self.nx.checked_sub(1)? as i32)?;
        let y_min = y_max.checked_sub(self.ny.checked_sub(1)? as i32)?;
        Some((x_min, x_max, y_min, y_max))
    }

    /// Build a NATIVE-resolution raster: one output pixel per WRF grid cell, so the
    /// output axis counts equal the WRF cell counts `native_nx`/`native_ny` (each
    /// clamped to `2..=max_axis`, and `max_axis` itself clamped to [`MAX_AXIS`]). The
    /// per-axis pitch is `span / (count - 1)` = domain-scan-angle-extent /
    /// native-cell-count, so the raster spans the full rect with one pixel per cell.
    /// When the WRF grid exceeds the cap the count clamps and the pitch coarsens to
    /// still span the domain — the honest MAX_AXIS exception (the caller logs it).
    pub fn native(
        rect: ScanAngleRect,
        native_nx: usize,
        native_ny: usize,
        max_axis: usize,
    ) -> Self {
        let cap = max_axis.clamp(2, MAX_AXIS);
        let span_x = (rect.x_max - rect.x_min).max(0.0);
        let span_y = (rect.y_max - rect.y_min).max(0.0);
        let nx = native_nx.clamp(2, cap);
        let ny = native_ny.clamp(2, cap);
        // Fall back to a nonzero pitch only for a degenerate (zero-span) rect.
        let fallback = VISIBLE_PITCH_RAD;
        let raw_x = span_x / (nx - 1) as f64;
        let raw_y = span_y / (ny - 1) as f64;
        let pitch_x = if raw_x > 0.0 { raw_x } else { fallback };
        let pitch_y = if raw_y > 0.0 { raw_y } else { fallback };
        Self {
            nx,
            ny,
            x_min: rect.x_min,
            y_max: rect.y_max,
            pitch_x,
            pitch_y,
        }
    }

    /// Scan angles `(x, y)` (rad) at output pixel `(px, py)`; row 0 = north.
    pub fn scan_angle(&self, px: usize, py: usize) -> (f64, f64) {
        (
            self.x_min + px as f64 * self.pitch_x,
            self.y_max - py as f64 * self.pitch_y,
        )
    }
}

/// The per-pixel surface lookup for one frame: geodetic lat/lon and fractional
/// WRF index `(i, j)` for every output pixel. Off-earth pixels (PUG visibility
/// fails) carry `NaN`. Row-major, row 0 = north (`nx * ny` each).
#[derive(Debug, Clone)]
pub struct SurfaceRaster {
    pub nx: usize,
    pub ny: usize,
    pub scan: ScanGrid,
    /// Geodetic latitude (deg) per pixel; `NaN` off-earth.
    pub lat: Vec<f32>,
    /// Geodetic longitude (deg) per pixel; `NaN` off-earth.
    pub lon: Vec<f32>,
    /// Fractional WRF index i (0-based) per pixel; `NaN` off-earth.
    pub grid_i: Vec<f32>,
    /// Fractional WRF index j (0-based) per pixel; `NaN` off-earth.
    pub grid_j: Vec<f32>,
    /// Optional per-pixel scan angles on SimSat's existing spherical physics
    /// camera.  `None` is the byte-for-byte historical path where sensor and
    /// physics scans are identical.  ABI ellipsoid navigation fills this after
    /// ellipsoid scan -> geodetic -> model-sphere conversion.
    pub model_scan: Option<Vec<[f32; 2]>>,
    /// Sensor-navigation metadata for geostationary rasters; `None` for top-down
    /// and free-perspective rasters.
    pub navigation_geometry: Option<GeoNavigationGeometry>,
}

impl SurfaceRaster {
    fn empty(scan: ScanGrid) -> Self {
        let n = scan.nx * scan.ny;
        Self {
            nx: scan.nx,
            ny: scan.ny,
            scan,
            lat: vec![f32::NAN; n],
            lon: vec![f32::NAN; n],
            grid_i: vec![f32::NAN; n],
            grid_j: vec![f32::NAN; n],
            model_scan: None,
            navigation_geometry: None,
        }
    }

    /// Ray scan angles for the existing spherical model physics at one pixel.
    pub fn model_scan_angle(&self, px: usize, py: usize) -> (f64, f64) {
        let idx = py * self.nx + px;
        self.model_scan
            .as_ref()
            .and_then(|v| v.get(idx))
            .filter(|v| v[0].is_finite() && v[1].is_finite())
            .map(|v| (v[0] as f64, v[1] as f64))
            .unwrap_or_else(|| self.scan.scan_angle(px, py))
    }

    /// Bounding rectangle of the ray angles consumed by the model atmosphere and
    /// cloud marches.  This differs slightly from the public sensor fixed grid in
    /// ABI ellipsoid mode.
    pub fn model_scan_rect(&self) -> (f64, f64, f64, f64) {
        let Some(values) = &self.model_scan else {
            let x_max = self.scan.x_min + self.scan.nx.saturating_sub(1) as f64 * self.scan.pitch_x;
            let y_min = self.scan.y_max - self.scan.ny.saturating_sub(1) as f64 * self.scan.pitch_y;
            return (self.scan.x_min, x_max, y_min, self.scan.y_max);
        };
        let mut rect: Option<(f64, f64, f64, f64)> = None;
        for value in values {
            let (x, y) = (value[0] as f64, value[1] as f64);
            if !(x.is_finite() && y.is_finite()) {
                continue;
            }
            rect = Some(match rect {
                None => (x, x, y, y),
                Some((xmin, xmax, ymin, ymax)) => {
                    (xmin.min(x), xmax.max(x), ymin.min(y), ymax.max(y))
                }
            });
        }
        rect.unwrap_or_else(|| {
            let x_max = self.scan.x_min + self.scan.nx.saturating_sub(1) as f64 * self.scan.pitch_x;
            let y_min = self.scan.y_max - self.scan.ny.saturating_sub(1) as f64 * self.scan.pitch_y;
            (self.scan.x_min, x_max, y_min, self.scan.y_max)
        })
    }

    /// The domain lat/lon bounding box over on-earth pixels, `(lat_min, lat_max,
    /// lon_min, lon_max)`, or `None` if no pixel is on earth. Longitude is taken
    /// on the naive numeric range (no antimeridian handling in M1).
    pub fn lat_lon_bbox(&self) -> Option<(f32, f32, f32, f32)> {
        let mut it = self
            .lat
            .iter()
            .zip(self.lon.iter())
            .filter(|(la, lo)| la.is_finite() && lo.is_finite());
        let (&la0, &lo0) = it.next()?;
        let (mut la_min, mut la_max, mut lo_min, mut lo_max) = (la0, la0, lo0, lo0);
        for (&la, &lo) in it {
            la_min = la_min.min(la);
            la_max = la_max.max(la);
            lo_min = lo_min.min(lo);
            lo_max = lo_max.max(lo);
        }
        Some((la_min, la_max, lo_min, lo_max))
    }
}

/// Build the per-pixel surface raster for a domain seen from a camera: choose the
/// scan-angle bbox from the domain corners, pick the pitch (clamped `<= max_axis`
/// per axis), then for each pixel run scan -> lat/lon (inverse) -> georef.forward
/// -> `(i, j)`. Returns `None` when the domain is not fully visible from the
/// satellite (any sample past the limb).
pub fn build_surface_raster(
    camera: &GeoCamera,
    georef: &GridGeoref,
    nx: usize,
    ny: usize,
    pitch: f64,
    max_axis: usize,
) -> Option<SurfaceRaster> {
    let rect = domain_scan_rect(camera, georef, nx, ny)?;
    let scan = ScanGrid::from_rect(rect, pitch, max_axis);
    Some(fill_surface_raster(camera, georef, nx, ny, scan))
}

/// Grow a scan-angle bbox OUTWARD by `margin_frac` of its span on EACH side (the
/// zoom-out / domain-margin feature). `margin_frac` is a fraction of the domain span:
/// `0.0` returns `rect` unchanged (identity — the domain edge-to-edge); `0.20` adds
/// 20% of the span on every side, so the original domain occupies the center
/// `1/(1 + 2*margin_frac)` of the grown rect (for `0.20`, ~`1/1.4`). Growing the bbox in
/// scan-angle space is the design-doc approved way to extend the geostationary extent —
/// the grown pixels sample the real earth around the domain (Blue Marble ground + clear
/// sky), because every out-of-domain sampler returns clear / flat ground.
pub fn grow_scan_rect(rect: ScanAngleRect, margin_frac: f64) -> ScanAngleRect {
    let m = margin_frac.max(0.0);
    if m <= 0.0 {
        return rect;
    }
    let mx = (rect.x_max - rect.x_min) * m;
    let my = (rect.y_max - rect.y_min) * m;
    ScanAngleRect {
        x_min: rect.x_min - mx,
        x_max: rect.x_max + mx,
        y_min: rect.y_min - my,
        y_max: rect.y_max + my,
    }
}

/// The NATIVE-resolution output pixel counts for a domain of `nx * ny` WRF cells with a
/// zoom-out `margin_frac`: each axis grows by `1 + 2*margin_frac` (the extent grew by
/// that factor and we keep ~one output pixel per WRF cell across the whole extent, so the
/// domain portion stays native pitch). `0.0` returns `(nx, ny)` exactly (identity). Each
/// axis is `>= 2`; the per-axis MAX_AXIS clamp is applied later by the scan-grid builder.
pub fn extended_native_counts(nx: usize, ny: usize, margin_frac: f64) -> (usize, usize) {
    let f = 1.0 + 2.0 * margin_frac.max(0.0);
    (
        ((nx as f64 * f).round() as usize).max(2),
        ((ny as f64 * f).round() as usize).max(2),
    )
}

/// Build the per-pixel surface raster for a domain at a [`ResolutionMode`] (owner
/// native-resolution fix) with an optional zoom-out `margin_frac` (the domain-margin
/// feature). `Native` (the studio default) sizes the raster to the WRF grid — one output
/// pixel per cell — so full native detail is preserved; the ABI modes use their fixed
/// pitch. `margin_frac > 0` GROWS the domain scan-angle bbox by that fraction on each side
/// ([`grow_scan_rect`]) and, for `Native`, grows the pixel count to keep native pitch
/// ([`extended_native_counts`]), so the raster covers domain + a real-earth margin; the
/// out-of-domain margin pixels render clear-sky ground (their `(i, j)` falls outside the
/// domain, which the samplers treat as clear / flat ground). `margin_frac = 0.0`
/// reproduces the edge-to-edge extent EXACTLY. Returns `None` when the domain is not fully
/// visible from the satellite (identical to [`build_surface_raster`]). When `Native` must
/// clamp against `max_axis` (the extended target exceeds the per-axis cap) it logs the
/// honest exception to stderr.
pub fn build_surface_raster_mode(
    camera: &GeoCamera,
    georef: &GridGeoref,
    nx: usize,
    ny: usize,
    mode: ResolutionMode,
    margin_frac: f64,
    max_axis: usize,
) -> Option<SurfaceRaster> {
    let rect = grow_scan_rect(domain_scan_rect(camera, georef, nx, ny)?, margin_frac);
    let (target_nx, target_ny) = extended_native_counts(nx, ny, margin_frac);
    // Exact GOES-R navigation + ABI 2 km uses the actual global 56-urad sample
    // lattice.  The historical model-sphere ABI crop and both Native paths retain
    // their prior endpoint-fit/count behavior byte-for-byte.
    let scan = if camera.navigation == GeoNavigation::GoesRAbiFixedGrid
        && mode == ResolutionMode::Abi2km
    {
        ScanGrid::abi_2km_global_lattice(rect, max_axis)?
    } else {
        mode.scan_grid(rect, target_nx, target_ny, max_axis)
    };
    if mode == ResolutionMode::Native && (scan.nx < target_nx || scan.ny < target_ny) {
        let cap = max_axis.min(MAX_AXIS);
        eprintln!(
            "simsat: Native resolution clamped to {}x{} px (target {}x{} for WRF grid \
             {}x{} + margin {:.2} exceeds the {}-px per-axis cap); scan pitch coarsened \
             to span the full extent.",
            scan.nx,
            scan.ny,
            target_nx,
            target_ny,
            nx,
            ny,
            margin_frac.max(0.0),
            cap
        );
    }
    Some(fill_surface_raster(camera, georef, nx, ny, scan))
}

/// The scan-angle bounding rect of a WRF domain seen from `camera`, or `None` when
/// the domain is not fully visible (any sampled corner/edge past the limb). Shared
/// by [`build_surface_raster`] and [`build_surface_raster_mode`].
fn domain_scan_rect(
    camera: &GeoCamera,
    georef: &GridGeoref,
    nx: usize,
    ny: usize,
) -> Option<ScanAngleRect> {
    let samples = domain_sample_points(georef, nx, ny);
    if samples.len() < 9 {
        return None; // a corner/edge is off-disk -> not honestly croppable
    }
    scan_angle_rect(&samples, |lat, lon| camera.forward(lat, lon))
}

/// Fill the per-pixel lat/lon + fractional-`(i, j)` surface raster for a fixed
/// `scan` grid: for each pixel run scan -> lat/lon (inverse) -> georef.forward ->
/// `(i, j)`. Off-earth pixels stay `NaN`.
fn fill_surface_raster(
    camera: &GeoCamera,
    georef: &GridGeoref,
    nx: usize,
    ny: usize,
    scan: ScanGrid,
) -> SurfaceRaster {
    let mut raster = SurfaceRaster::empty(scan);
    raster.navigation_geometry = Some(camera.geometry());
    let model_camera = camera.model_sphere();
    if camera.navigation != GeoNavigation::ModelSphere {
        raster.model_scan = Some(vec![[f32::NAN; 2]; scan.nx * scan.ny]);
    }
    for py in 0..scan.ny {
        for px in 0..scan.nx {
            let (x, y) = scan.scan_angle(px, py);
            let Some((lat, lon)) = camera.inverse(x, y) else {
                continue; // off-earth -> stays NaN (space)
            };
            let idx = py * scan.nx + px;
            raster.lat[idx] = lat as f32;
            raster.lon[idx] = lon as f32;
            if let Some(model_scan) = &mut raster.model_scan
                && let Some((model_x, model_y)) = model_camera.forward(lat, lon)
            {
                model_scan[idx] = [model_x as f32, model_y as f32];
            }
            let (fi, fj) = georef.forward(lat, lon);
            // Keep (i, j) only when it lands inside the WRF domain; outside-domain
            // but on-earth pixels still get Blue Marble albedo (no WRF terrain).
            let in_domain =
                (0.0..=(nx - 1) as f64).contains(&fi) && (0.0..=(ny - 1) as f64).contains(&fj);
            if in_domain {
                raster.grid_i[idx] = fi as f32;
                raster.grid_j[idx] = fj as f32;
            }
        }
    }
    raster
}

// ── top-down map-registered view (the WRF-Runner integration product) ─────────

/// The synthetic top-down camera altitude above the ground sphere (m). Placed ABOVE
/// the top of the atmosphere shell ([`ATMOSPHERE_HEIGHT_M`], 100 km) so every nadir
/// ray integrates the full atmospheric column exactly like the geostationary path's
/// on-earth pixels do. For a nadir ray (the camera on the local vertical, looking
/// straight down) the sampled column is independent of this height — it only sets the
/// ray's start distance — so any value above the atmosphere top is equivalent; this
/// fixed 300 km keeps the ray/sphere intersection well-conditioned.
pub const TOPDOWN_CAMERA_ALTITUDE_M: f64 = ATMOSPHERE_HEIGHT_M + 200_000.0;

/// The nadir (straight-down) ECEF ray for a geodetic `(lat, lon)`: the camera sits
/// on the LOCAL VERTICAL at [`TOPDOWN_CAMERA_ALTITUDE_M`] above the ground point and
/// looks straight down (view = `-local_up`). Returns `(camera_ecef, view_dir)` with a
/// unit `view_dir`. This is the per-pixel ray the top-down map view feeds into the
/// SAME surface/atmosphere/cloud march the geostationary path uses (M2/M3/M4/M5 are
/// ray-direction-agnostic): the ray descends the local vertical and hits the ground
/// exactly at `(lat, lon)`, so the output registers with a top-down Lambert map.
pub fn topdown_nadir_ray(lat_deg: f64, lon_deg: f64) -> ([f64; 3], [f64; 3]) {
    let (la, lo) = (lat_deg.to_radians(), lon_deg.to_radians());
    let (sla, cla) = la.sin_cos();
    let (slo, clo) = lo.sin_cos();
    // Unit local up = the ECEF radial at (lat, lon) = the surface-point direction.
    let up = [cla * clo, cla * slo, sla];
    let cam_r = R_EARTH + TOPDOWN_CAMERA_ALTITUDE_M;
    let cam = [up[0] * cam_r, up[1] * cam_r, up[2] * cam_r];
    let view = [-up[0], -up[1], -up[2]]; // nadir: straight down the local vertical
    (cam, view)
}

/// A top-down, north-up, map-registered output raster over the WRF domain's OWN
/// Lambert map extent (design-doc top-down addendum). Each output pixel samples a
/// fractional WRF grid index `(i, j)` uniformly across the domain — which, because the
/// georeference is affine `(i, j) <-> (u, v)`, is exactly the domain's projected x/y
/// (Lambert map) plane — with row 0 the NORTHERN domain edge (max `j`) and column 0
/// the western (min `i`). Per pixel it carries the geodetic lat/lon and the fractional
/// `(i, j)` (always in-domain by construction); the top-down render path
/// ([`crate::topdown`]) then generates a nadir ray per pixel via [`topdown_nadir_ray`]
/// and marches the same surface/atmosphere/cloud pipeline. The result lines up with
/// other top-down WRF field plots (same projection, same spherical earth).
#[derive(Debug, Clone)]
pub struct MapRaster {
    /// Output raster width (columns; west -> east).
    pub nx: usize,
    /// Output raster height (rows; row 0 = north).
    pub ny: usize,
    /// The WRF grid dims the raster samples across.
    pub domain_nx: usize,
    pub domain_ny: usize,
    /// Geodetic latitude (deg) per output pixel; `NaN` only if the projection inverse
    /// fails (never for an interior domain sample).
    pub lat: Vec<f32>,
    /// Geodetic longitude (deg) per output pixel; `NaN` as above.
    pub lon: Vec<f32>,
    /// Fractional WRF index i (0-based) per output pixel.
    pub grid_i: Vec<f32>,
    /// Fractional WRF index j (0-based) per output pixel.
    pub grid_j: Vec<f32>,
}

/// The OUTER PIXEL-EDGE fractional WRF grid-index bounds a top-down map raster spans,
/// `(i_min, i_max, j_min, j_max)`, given the domain dims, the (margin-extended, possibly
/// MAX_AXIS-clamped) output raster dims, and the zoom-out `margin_frac`. The map raster
/// samples grid indices at the SAMPLE CENTRES `[-m*(nx-1), (nx-1)*(1+m)]` (`m` =
/// `margin_frac`); the outer pixel EDGES lie a further half-pixel beyond, which is what
/// an `imshow` extent needs. For `margin_frac = 0.0` at native output (`out == domain`)
/// this is `(-0.5, domain_nx-0.5, -0.5, domain_ny-0.5)` — the domain's half-cell-padded
/// box (byte-identical to the pre-margin extent).
pub fn map_pixel_edge_index_bounds(
    domain_nx: usize,
    domain_ny: usize,
    out_nx: usize,
    out_ny: usize,
    margin_frac: f64,
) -> (f64, f64, f64, f64) {
    let m = margin_frac.max(0.0);
    let (di, dj) = ((domain_nx.max(2) - 1) as f64, (domain_ny.max(2) - 1) as f64);
    let (i_lo, i_hi) = (-m * di, di + m * di);
    let (j_lo, j_hi) = (-m * dj, dj + m * dj);
    let half_i = if out_nx > 1 {
        (i_hi - i_lo) / (2.0 * (out_nx - 1) as f64)
    } else {
        0.5
    };
    let half_j = if out_ny > 1 {
        (j_hi - j_lo) / (2.0 * (out_ny - 1) as f64)
    } else {
        0.5
    };
    (i_lo - half_i, i_hi + half_i, j_lo - half_j, j_hi + half_j)
}

/// Build a top-down [`MapRaster`] over a domain's own Lambert extent at an output
/// resolution `out_nx * out_ny` (pass the WRF grid `domain_nx * domain_ny` for the
/// native one-pixel-per-cell map), with an optional zoom-out `margin_frac` (the
/// domain-margin feature). North-up: row 0 samples the northern (extended) edge.
///
/// `margin_frac > 0` GROWS the sampled grid-index range to `[-m*(nx-1), (nx-1)*(1+m)]`
/// (`m = margin_frac`) on each axis and scales the output pixel count by `1 + 2m` so the
/// domain keeps ~native pitch, capped at [`MAX_AXIS`] per axis (coarsening the pitch, with
/// a stderr log). Margin samples (grid index outside `[0, nx-1] x [0, ny-1]`) get their
/// geodetic `lat/lon` from the global projection inverse but leave `grid_i/grid_j` as
/// `NaN`, so the surface treats them as flat real-earth ground and the cloud/IR samplers
/// return clear (no weather outside the WRF domain). `margin_frac = 0.0` reproduces the
/// edge-to-edge map EXACTLY (same dims + same per-pixel values). `None` for a degenerate
/// grid/output size.
/// EDGE-SAMPLE INSET (grid cells) for the top-down map raster (WS2 row-0 speckle fix).
/// The native map's outermost row/columns sample EXACTLY on the domain-inclusion
/// boundary (`fi = 0 / nx-1`, `fj = 0 / ny-1`). The cloud/IR marches and the ground
/// cloud-shadow then RECONSTRUCT the grid index from the raster's f32 lat/lon
/// (ECEF -> lat/lon -> projection forward), and the f32 quantization (~2.5e-3 cells at
/// a 250 m grid) oscillates the recovered index across the inclusive in-domain test
/// per pixel — the sampler returns cloud for one pixel and CLEAR for its neighbour,
/// which rendered as hard dash SPECKLE along row 0 / col 0 / col nx-1 (measured
/// roughness 50-300x the interior; clouds-off renders were clean, isolating the
/// scene reprojection). In-domain edge sample centres are inset by this amount so the
/// round trip can never cross the boundary. `0.01` cells = 2.5 m at a 250 m grid (and
/// still sub-pixel at any WRF resolution) — far below anything visible, while 4x the
/// f32 quantization error at the owner's finest (250 m) grids. Kept strictly BELOW the
/// api-layer georef-mesh round-trip tolerance (0.02 cells), which pins the mesh to the
/// exact corner indices.
pub const MAP_EDGE_INSET_CELLS: f64 = 0.01;

/// Nominal ground sample distances for the two ABI-class resolution choices when
/// rendering a map-registered top-down frame.  Unlike the geostationary view, a
/// top-down raster has no scan-angle geometry: its output counts are therefore
/// derived directly from the WRF domain's physical projected-grid spacing.
const ABI_1KM_GROUND_PITCH_M: f64 = 1_000.0;
const ABI_2KM_GROUND_PITCH_M: f64 = 2_000.0;

/// Base (zero-margin) top-down output counts for `mode`.
///
/// `Native` deliberately returns `(domain_nx, domain_ny)` without inspecting the
/// spacings, preserving the historical map raster byte-for-byte.  ABI modes cover
/// the full centre-to-centre domain span, `(n - 1) * d`, with a sample on each edge;
/// `ceil(span / pitch) + 1` guarantees that the effective pitch is never coarser
/// than the selected 1 km / 2 km class.  If either axis would exceed [`MAX_AXIS`],
/// both axes are rescaled with ONE common physical pitch.  The constrained axis
/// gets `MAX_AXIS - 1` edge-to-edge intervals and the other axis is rounded from
/// that same pitch, preserving the domain's physical aspect ratio instead of
/// independently stretching x and y. [`build_map_raster`] still spans the complete
/// domain and applies its existing margin growth/cap semantics.
pub fn topdown_resolution_counts(
    domain_nx: usize,
    domain_ny: usize,
    dx_m: f64,
    dy_m: f64,
    mode: ResolutionMode,
) -> (usize, usize) {
    if mode == ResolutionMode::Native {
        return (domain_nx, domain_ny);
    }
    let pitch_m = match mode {
        ResolutionMode::Abi1km => ABI_1KM_GROUND_PITCH_M,
        ResolutionMode::Abi2km => ABI_2KM_GROUND_PITCH_M,
        ResolutionMode::Native => unreachable!(),
    };
    if domain_nx < 2
        || domain_ny < 2
        || !dx_m.is_finite()
        || !dy_m.is_finite()
        || dx_m <= 0.0
        || dy_m <= 0.0
    {
        // Invalid source metadata must not create a zero/astronomical raster.
        // Falling back to source counts keeps the frame usable; build_map_raster
        // owns the final safety cap exactly as it did before this selector existed.
        return (domain_nx, domain_ny);
    }

    let span_x_m = (domain_nx - 1) as f64 * dx_m;
    let span_y_m = (domain_ny - 1) as f64 * dy_m;
    let raw_ix = (span_x_m / pitch_m).ceil().max(1.0) as usize;
    let raw_iy = (span_y_m / pitch_m).ceil().max(1.0) as usize;
    let max_intervals = MAX_AXIS - 1;
    if raw_ix <= max_intervals && raw_iy <= max_intervals {
        return (raw_ix + 1, raw_iy + 1);
    }

    // One common coarsened pitch keeps square physical output pixels.  Counts are
    // edge-sample counts, hence MAX_AXIS pixels provide MAX_AXIS-1 intervals.
    let capped_pitch_m = (span_x_m / max_intervals as f64)
        .max(span_y_m / max_intervals as f64)
        .max(pitch_m);
    let capped_count = |span_m: f64| {
        ((span_m / capped_pitch_m).round() as usize)
            .clamp(1, max_intervals)
            .saturating_add(1)
    };
    (capped_count(span_x_m), capped_count(span_y_m))
}

/// Build a top-down map raster at a selected [`ResolutionMode`].  `dx_m` and
/// `dy_m` are the physical WRF grid spacings in metres (callers convert geographic
/// degree grids before entering here).  Margin growth and the per-axis cap remain
/// owned by [`build_map_raster`], so every mode spans the same physical extent.
pub fn build_map_raster_mode(
    georef: &GridGeoref,
    domain_nx: usize,
    domain_ny: usize,
    dx_m: f64,
    dy_m: f64,
    mode: ResolutionMode,
    margin_frac: f64,
) -> Option<MapRaster> {
    let (out_nx, out_ny) = topdown_resolution_counts(domain_nx, domain_ny, dx_m, dy_m, mode);
    build_map_raster(georef, domain_nx, domain_ny, out_nx, out_ny, margin_frac)
}

pub fn build_map_raster(
    georef: &GridGeoref,
    domain_nx: usize,
    domain_ny: usize,
    out_nx: usize,
    out_ny: usize,
    margin_frac: f64,
) -> Option<MapRaster> {
    if domain_nx < 2 || domain_ny < 2 || out_nx < 1 || out_ny < 1 {
        return None;
    }
    let m = margin_frac.max(0.0);
    // Scale the output pixel count with the extent so the domain keeps native pitch; cap
    // each axis at MAX_AXIS (the pitch coarsens to still span domain + margin), logged.
    let grow = 1.0 + 2.0 * m;
    let want_nx = ((out_nx as f64 * grow).round() as usize).max(1);
    let want_ny = ((out_ny as f64 * grow).round() as usize).max(1);
    let ext_nx = want_nx.min(MAX_AXIS);
    let ext_ny = want_ny.min(MAX_AXIS);
    if m > 0.0 && (ext_nx < want_nx || ext_ny < want_ny) {
        eprintln!(
            "simsat: top-down margin raster clamped to {ext_nx}x{ext_ny} px (target \
             {want_nx}x{want_ny} at margin {m:.2} exceeds the {MAX_AXIS}-px per-axis cap); \
             map pitch coarsened to span the domain + margin."
        );
    }
    let (di, dj) = ((domain_nx - 1) as f64, (domain_ny - 1) as f64);
    // Extended grid-index range (sample centres): the domain [0, di] plus m*di on each side.
    let (i_lo, i_hi) = (-m * di, di + m * di);
    let (j_lo, j_hi) = (-m * dj, dj + m * dj);
    let n = ext_nx * ext_ny;
    let mut lat = vec![f32::NAN; n];
    let mut lon = vec![f32::NAN; n];
    let mut grid_i = vec![f32::NAN; n];
    let mut grid_j = vec![f32::NAN; n];
    for py in 0..ext_ny {
        // North-up: py = 0 -> the northern (max j) extended edge; py = ext_ny-1 -> south.
        let fj = if ext_ny > 1 {
            j_hi + (j_lo - j_hi) * (py as f64 / (ext_ny - 1) as f64)
        } else {
            0.5 * (j_lo + j_hi)
        };
        // Inset an IN-DOMAIN sample centre off the exact inclusion boundary (see
        // [`MAP_EDGE_INSET_CELLS`]); margin samples (outside the domain) keep their raw
        // position — they are NaN-gridded and never reconstructed against the boundary.
        let fj_s = if (0.0..=dj).contains(&fj) {
            fj.clamp(MAP_EDGE_INSET_CELLS, dj - MAP_EDGE_INSET_CELLS)
        } else {
            fj
        };
        for px in 0..ext_nx {
            let fi = if ext_nx > 1 {
                i_lo + (i_hi - i_lo) * (px as f64 / (ext_nx - 1) as f64)
            } else {
                0.5 * (i_lo + i_hi)
            };
            let fi_s = if (0.0..=di).contains(&fi) {
                fi.clamp(MAP_EDGE_INSET_CELLS, di - MAP_EDGE_INSET_CELLS)
            } else {
                fi
            };
            let idx = py * ext_nx + px;
            if let Some((la, lo)) = georef.inverse(fi_s, fj_s) {
                lat[idx] = la as f32;
                lon[idx] = lo as f32;
                // Only tag an IN-DOMAIN (i, j); the margin stays NaN so the surface reads
                // flat real-earth ground and the cloud/IR samplers return clear. The
                // stored index is the INSET position (consistent with the lat/lon).
                if (0.0..=di).contains(&fi) && (0.0..=dj).contains(&fj) {
                    grid_i[idx] = fi_s as f32;
                    grid_j[idx] = fj_s as f32;
                }
            }
        }
    }
    Some(MapRaster {
        nx: ext_nx,
        ny: ext_ny,
        domain_nx,
        domain_ny,
        lat,
        lon,
        grid_i,
        grid_j,
    })
}

impl MapRaster {
    /// Adapt this map raster to a [`SurfaceRaster`] so the shared per-pixel LUT builder
    /// (`gpu::build_luts`) and the studio's / CLI's assemble closure work UNCHANGED for
    /// both view modes. The [`ScanGrid`] is a benign PLACEHOLDER: the top-down render
    /// path generates nadir rays from the per-pixel lat/lon ([`topdown_nadir_ray`]), so
    /// no scan angle is ever read from it — it exists only to satisfy the shared
    /// `SurfaceRaster` shape (which `build_luts` consumes via `.lat/.lon/.grid_i/.grid_j`
    /// only, never `.scan`).
    pub fn as_surface_raster(&self) -> SurfaceRaster {
        SurfaceRaster {
            nx: self.nx,
            ny: self.ny,
            scan: ScanGrid {
                nx: self.nx,
                ny: self.ny,
                x_min: 0.0,
                y_max: 0.0,
                pitch_x: VISIBLE_PITCH_RAD,
                pitch_y: VISIBLE_PITCH_RAD,
            },
            lat: self.lat.clone(),
            lon: self.lon.clone(),
            grid_i: self.grid_i.clone(),
            grid_j: self.grid_j.clone(),
            model_scan: None,
            navigation_geometry: None,
        }
    }
}

// ── free PERSPECTIVE camera (tier 2: prerendered angled flyovers) ──────────────

/// Geodetic `(lat, lon)` (deg) + altitude above the sphere (m) -> ECEF metres on the
/// WRF sphere (`R = 6.37e6`, owner decision 5). The geodetic up on a sphere is the
/// radial direction, so this is exact (no ellipsoid normal).
pub fn geodetic_to_ecef(lat_deg: f64, lon_deg: f64, alt_m: f64) -> [f64; 3] {
    let (la, lo) = (lat_deg.to_radians(), lon_deg.to_radians());
    let (sla, cla) = la.sin_cos();
    let (slo, clo) = lo.sin_cos();
    let r = R_EARTH + alt_m;
    [r * cla * clo, r * cla * slo, r * sla]
}

/// A FREE PERSPECTIVE camera (the broadcast-visuals "angled 3D scene" product): an eye
/// at `(lat, lon, alt)` looking at a target `(lat, lon, alt)` with a HORIZONTAL field
/// of view `fov_deg` across a `width x height` image. Produces per-pixel ECEF rays
/// ([`PerspectiveBasis::pixel_ray`]) that feed the SAME ray-agnostic marches the
/// geostationary and top-down products use (`render::surface_toa_radiance`,
/// `clouds::march_cloud`) — the [`topdown_nadir_ray`] pattern generalized from one
/// fixed nadir ray per pixel to a pinhole ray fan from one eye.
///
/// CONVENTIONS: `fov_deg` is the HORIZONTAL FOV (the vertical follows from the aspect,
/// `tan(v/2) = tan(h/2) * height/width`); the image up is the LOCAL VERTICAL at the eye
/// projected perpendicular to the view (a horizon-level camera keeps the horizon
/// level); when the view is within ~0.8 deg of straight up/down the up-hint degenerates
/// and LOCAL NORTH is used instead (a nadir shot comes out north-up, matching the
/// top-down map). A yaw/pitch parameterization is deliberately NOT built this wave —
/// a flyover path is naturally scripted as eye/look pairs (see the binding README);
/// deriving one from yaw/pitch is a trivial caller-side conversion.
///
/// HONESTY: perspective renders carry the camera on the render result
/// (`api::Georef::camera_pose`) and in the render log — the what-if labeling
/// discipline. Parallax displacement of high cloud against the ground is PHYSICAL and
/// intended (the rays are true 3-D lines through the volume; that is the 3D look).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerspectiveCamera {
    pub eye_lat_deg: f64,
    pub eye_lon_deg: f64,
    /// Eye altitude above the sphere (m); must be > 0 (the eye is airborne/space).
    pub eye_alt_m: f64,
    pub look_lat_deg: f64,
    pub look_lon_deg: f64,
    /// Look-target altitude above the sphere (m); >= 0 (a ground point or a level).
    pub look_alt_m: f64,
    /// HORIZONTAL field of view (deg), across the image width. `1..=160`.
    pub fov_deg: f64,
    /// Output image dims (px); each `2..=`[`MAX_AXIS`].
    pub width: usize,
    pub height: usize,
}

impl PerspectiveCamera {
    /// Validate the camera parameters (finite coordinates, an airborne eye, a sane
    /// FOV, in-cap dims, and a non-degenerate eye->look direction).
    pub fn validate(&self) -> Result<(), String> {
        let vals = [
            self.eye_lat_deg,
            self.eye_lon_deg,
            self.eye_alt_m,
            self.look_lat_deg,
            self.look_lon_deg,
            self.look_alt_m,
            self.fov_deg,
        ];
        if vals.iter().any(|v| !v.is_finite()) {
            return Err("perspective camera has a non-finite parameter".to_string());
        }
        if self.eye_lat_deg.abs() > 90.0 || self.look_lat_deg.abs() > 90.0 {
            return Err("perspective camera latitude out of [-90, 90]".to_string());
        }
        if self.eye_alt_m <= 0.0 {
            return Err(format!(
                "perspective eye altitude must be > 0 m (got {})",
                self.eye_alt_m
            ));
        }
        if self.look_alt_m < 0.0 {
            return Err(format!(
                "perspective look altitude must be >= 0 m (got {})",
                self.look_alt_m
            ));
        }
        if !(1.0..=160.0).contains(&self.fov_deg) {
            return Err(format!(
                "perspective fov must be 1..=160 deg (got {})",
                self.fov_deg
            ));
        }
        if !(2..=MAX_AXIS).contains(&self.width) || !(2..=MAX_AXIS).contains(&self.height) {
            return Err(format!(
                "perspective dims must be 2..={MAX_AXIS} px per axis (got {}x{})",
                self.width, self.height
            ));
        }
        let eye = geodetic_to_ecef(self.eye_lat_deg, self.eye_lon_deg, self.eye_alt_m);
        let look = geodetic_to_ecef(self.look_lat_deg, self.look_lon_deg, self.look_alt_m);
        let d = sub3(look, eye);
        if len3(d) < 1.0 {
            return Err("perspective eye and look-at coincide".to_string());
        }
        Ok(())
    }

    /// Build the ECEF pinhole basis (validating first). See the type docs for the
    /// up-vector and FOV conventions.
    pub fn basis(&self) -> Result<PerspectiveBasis, String> {
        self.validate()?;
        let eye = geodetic_to_ecef(self.eye_lat_deg, self.eye_lon_deg, self.eye_alt_m);
        let look = geodetic_to_ecef(self.look_lat_deg, self.look_lon_deg, self.look_alt_m);
        let forward = norm3(sub3(look, eye));
        // Up hint: the local vertical at the eye (radial); local north when the view is
        // near-vertical (the nadir/zenith degeneracy — a nadir shot renders north-up).
        let up_world = norm3(eye);
        let up_hint = if dot3(forward, up_world).abs() > 0.9999 {
            let (sla, cla) = self.eye_lat_deg.to_radians().sin_cos();
            let (slo, clo) = self.eye_lon_deg.to_radians().sin_cos();
            [-sla * clo, -sla * slo, cla] // local north (ENU north in ECEF)
        } else {
            up_world
        };
        let right = norm3(cross3(forward, up_hint));
        let up = cross3(right, forward); // unit by construction (right ⊥ forward)
        let tan_half_h = (self.fov_deg.to_radians() * 0.5).tan();
        let tan_half_v = tan_half_h * self.height as f64 / self.width as f64;
        Ok(PerspectiveBasis {
            eye,
            forward,
            right,
            up,
            tan_half_h,
            tan_half_v,
            width: self.width,
            height: self.height,
        })
    }

    /// A short human-readable pose label for logs / frame provenance.
    pub fn label(&self) -> String {
        format!(
            "eye=({:.4},{:.4},{:.0} m) look=({:.4},{:.4},{:.0} m) fov={:.1} deg {}x{}",
            self.eye_lat_deg,
            self.eye_lon_deg,
            self.eye_alt_m,
            self.look_lat_deg,
            self.look_lon_deg,
            self.look_alt_m,
            self.fov_deg,
            self.width,
            self.height
        )
    }
}

/// The resolved ECEF pinhole basis of a [`PerspectiveCamera`]: the eye position and
/// the orthonormal `(forward, right, up)` frame plus the half-FOV tangents.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerspectiveBasis {
    pub eye: [f64; 3],
    pub forward: [f64; 3],
    pub right: [f64; 3],
    pub up: [f64; 3],
    pub tan_half_h: f64,
    pub tan_half_v: f64,
    pub width: usize,
    pub height: usize,
}

impl PerspectiveBasis {
    /// The unit ECEF view ray through the CENTRE of pixel `(px, py)` (row 0 = image
    /// top). NDC runs over pixel centres, so for odd dims the middle pixel's ray is
    /// exactly `forward`.
    pub fn pixel_ray(&self, px: usize, py: usize) -> [f64; 3] {
        let ndc_x = ((px as f64 + 0.5) / self.width as f64) * 2.0 - 1.0;
        let ndc_y = 1.0 - ((py as f64 + 0.5) / self.height as f64) * 2.0;
        let mut v = self.forward;
        v = madd3(v, self.right, ndc_x * self.tan_half_h);
        v = madd3(v, self.up, ndc_y * self.tan_half_v);
        norm3(v)
    }
}

/// Build the per-pixel SURFACE raster for a perspective camera: for every pixel,
/// intersect its ray with the ground sphere (`R_EARTH`); a hit yields the geodetic
/// lat/lon (+ the fractional WRF `(i, j)` when inside the domain), a miss stays `NaN`
/// (sky/space — the render composites the existing limb/space handling there, and the
/// cloud march still runs, since near-horizon rays can cross the cloud shell without
/// touching the ground). The returned [`SurfaceRaster`]'s `ScanGrid` is the same
/// benign placeholder [`MapRaster::as_surface_raster`] uses — the perspective render
/// path derives its rays from the basis, never from scan angles — so the shared LUT
/// builder (`gpu::build_luts`) and assemble machinery work unchanged.
pub fn build_perspective_raster(
    basis: &PerspectiveBasis,
    georef: &GridGeoref,
    domain_nx: usize,
    domain_ny: usize,
) -> SurfaceRaster {
    let scan = ScanGrid {
        nx: basis.width,
        ny: basis.height,
        x_min: 0.0,
        y_max: 0.0,
        pitch_x: VISIBLE_PITCH_RAD,
        pitch_y: VISIBLE_PITCH_RAD,
    };
    let mut raster = SurfaceRaster::empty(scan);
    let (di, dj) = ((domain_nx.max(1) - 1) as f64, (domain_ny.max(1) - 1) as f64);
    for py in 0..basis.height {
        for px in 0..basis.width {
            let view = basis.pixel_ray(px, py);
            let Some(t) = ray_sphere_first_hit(basis.eye, view, R_EARTH) else {
                continue; // sky/space: stays NaN
            };
            let p = madd3(basis.eye, view, t);
            let r = len3(p);
            let lat = (p[2] / r).clamp(-1.0, 1.0).asin().to_degrees();
            let lon = p[1].atan2(p[0]).to_degrees();
            let idx = py * basis.width + px;
            raster.lat[idx] = lat as f32;
            raster.lon[idx] = lon as f32;
            let (fi, fj) = georef.forward(lat, lon);
            if (0.0..=di).contains(&fi) && (0.0..=dj).contains(&fj) {
                raster.grid_i[idx] = fi as f32;
                raster.grid_j[idx] = fj as f32;
            }
        }
    }
    raster
}

/// The nearest POSITIVE ray/sphere intersection distance, or `None` (miss, or the
/// sphere is entirely behind the origin). `dir` unit; origin outside the sphere for
/// the validated camera (eye altitude > 0).
fn ray_sphere_first_hit(origin: [f64; 3], dir: [f64; 3], radius: f64) -> Option<f64> {
    let b = dot3(origin, dir);
    let c = dot3(origin, origin) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let s = disc.sqrt();
    let t0 = -b - s;
    let t1 = -b + s;
    if t0 > 0.0 {
        Some(t0)
    } else if t1 > 0.0 {
        Some(t1)
    } else {
        None
    }
}

// Small ECEF vector helpers for the perspective section (camera.rs had no vector
// math before; kept private and minimal).
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn madd3(a: [f64; 3], b: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s]
}
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn len3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}
fn norm3(a: [f64; 3]) -> [f64; 3] {
    let l = len3(a);
    if l > 0.0 {
        [a[0] / l, a[1] / l, a[2] / l]
    } else {
        [0.0, 0.0, 1.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sampled from the exact HRRR target grid in
    /// `outputs/v017-goes19-hrrr-reference/abi-reference-aligned.npz`; expected
    /// scans were produced by the pyproj-cross-checked pure-NumPy PUG reference
    /// in `scripts/fetch-goes-abi-reference.py` using the source GOES-19 CMI
    /// metadata (a=6378137, b=6356752.31414, h=35786023, lon0=-75, sweep=x).
    const GOES19_REFERENCE_SAMPLES: &[(f64, f64, f64, f64)] = &[
        (
            47.838623047,
            -134.095474243,
            -0.091206125744050,
            0.117195748662799,
        ),
        (
            52.071998596,
            -110.278320312,
            -0.057703857631825,
            0.127844062729958,
        ),
        (
            47.842193604,
            -60.917194366,
            0.027244018901465,
            0.123178030156049,
        ),
        (
            47.975654602,
            -97.506889343,
            -0.042504359652495,
            0.122770868590927,
        ),
        (
            43.475685120,
            -63.440551758,
            0.024521421060113,
            0.115531056518809,
        ),
        (
            43.239986420,
            -97.506401062,
            -0.046704881037136,
            0.114301587493284,
        ),
        (
            39.009223938,
            -65.663246155,
            0.021477975760447,
            0.106740871278062,
        ),
        (
            38.497245789,
            -97.505973816,
            -0.050633448397189,
            0.104785258914001,
        ),
        (
            34.528316498,
            -67.612884521,
            0.018216969529581,
            0.096991540533419,
        ),
        (
            33.754348755,
            -97.505607605,
            -0.054238440265365,
            0.094293485514624,
        ),
        (
            30.040863037,
            -69.344619751,
            0.014795790802460,
            0.086360146227933,
        ),
        (
            29.016033173,
            -97.505279541,
            -0.057473015525938,
            0.082913531458819,
        ),
        (
            25.552131653,
            -70.900314331,
            0.011270172680640,
            0.074936498021740,
        ),
        (
            24.364576340,
            -97.504989624,
            -0.060251889852725,
            0.070956353199642,
        ),
        (
            22.911014557,
            -114.541458130,
            -0.098904774797383,
            0.065480536419709,
        ),
        (
            21.140546799,
            -72.289718628,
            0.007754280861710,
            0.063029512605121,
        ),
    ];

    #[test]
    fn goes_r_forward_matches_validated_goes19_reference_grid() {
        let camera =
            GeoCamera::for_navigation(SatellitePreset::GoesEast, GeoNavigation::GoesRAbiFixedGrid)
                .unwrap();
        let mut sum_sq = 0.0;
        let mut max_error = 0.0_f64;
        for &(lat, lon, expected_x, expected_y) in GOES19_REFERENCE_SAMPLES {
            let (x, y) = camera.forward(lat, lon).expect("reference point visible");
            let error = (x - expected_x).hypot(y - expected_y);
            max_error = max_error.max(error);
            sum_sq += error * error;
        }
        let rms = (sum_sq / GOES19_REFERENCE_SAMPLES.len() as f64).sqrt();
        // The fixture's target lat/lon are the aligned NPZ's stored f32 values and
        // expected scans are decimal text, so ~1e-12 rad is the serialization floor
        // (still over seven orders below one 28 urad ABI visible pixel).
        assert!(max_error < 2.0e-12, "max scan error {max_error:e}");
        assert!(rms < 1.0e-12, "RMS scan error {rms:e}");
    }

    #[test]
    fn goes_r_inverse_roundtrips_reference_grid_and_rejects_space() {
        let camera =
            GeoCamera::for_navigation(SatellitePreset::GoesEast, GeoNavigation::GoesRAbiFixedGrid)
                .unwrap();
        let mut max_deg = 0.0_f64;
        for &(lat, lon, x, y) in GOES19_REFERENCE_SAMPLES {
            let (got_lat, got_lon) = camera.inverse(x, y).expect("reference scan on disk");
            max_deg = max_deg.max((got_lat - lat).hypot(got_lon - lon));
        }
        assert!(max_deg < 3.0e-9, "max roundtrip error {max_deg:e} deg");
        assert!(camera.inverse(0.1519, 0.1519).is_none());
        assert!(camera.forward(0.0, 120.0).is_none());
    }

    #[test]
    fn goes_r_navigation_is_opt_in_and_himawari_is_rejected() {
        let default = GeoCamera::new(SatellitePreset::GoesEast);
        assert_eq!(default.navigation, GeoNavigation::ModelSphere);
        assert_eq!(default.sub_lon_deg, GOES_EAST_SUB_LON_DEG);
        let abi =
            GeoCamera::for_navigation(SatellitePreset::GoesEast, GeoNavigation::GoesRAbiFixedGrid)
                .unwrap();
        assert_eq!(abi.geometry().sweep_angle_axis, "x");
        assert_eq!(abi.sub_lon_deg, -75.0);
        assert!(
            GeoCamera::for_navigation(SatellitePreset::Himawari, GeoNavigation::GoesRAbiFixedGrid)
                .is_err()
        );
    }

    #[test]
    fn goes_r_pug_example_and_axis_limbs() {
        let camera =
            GeoCamera::for_navigation(SatellitePreset::GoesEast, GeoNavigation::GoesRAbiFixedGrid)
                .unwrap();
        let (x, y) = camera.forward(33.846162, -84.690932).unwrap();
        assert!((x - -0.024_051_999_804).abs() < 1.0e-9);
        assert!((y - 0.095_339_999_332).abs() < 1.0e-9);
        let (lat, lon) = camera.inverse(-0.024052, 0.095340).unwrap();
        assert!((lat - 33.846_162_290_6).abs() < 1.0e-9);
        assert!((lon - -84.690_932_118_8).abs() < 1.0e-9);

        let x_limb = 0.151_852_080_189_958;
        let y_limb = 0.151_350_701_197_250;
        assert!(camera.inverse(x_limb - 1.0e-10, 0.0).is_some());
        assert!(camera.inverse(x_limb + 1.0e-8, 0.0).is_none());
        assert!(camera.inverse(0.0, y_limb - 1.0e-10).is_some());
        assert!(camera.inverse(0.0, y_limb + 1.0e-8).is_none());
    }
    use crate::frame::MapProjection;

    fn spherical_forward(lat: f64, lon: f64) -> Option<(f64, f64)> {
        lat_lon_to_scan_angles(PERSPECTIVE_HEIGHT_M, R_EARTH, R_EARTH, -75.2, lat, lon)
    }
    fn spherical_inverse(x: f64, y: f64) -> Option<(f64, f64)> {
        scan_angles_to_lat_lon(PERSPECTIVE_HEIGHT_M, R_EARTH, R_EARTH, -75.2, x, y)
    }

    /// Tripwire: the ported forward and ported inverse must round-trip (the whole
    /// point of porting the inverse instead of trusting rw_sat sweep=y).
    #[test]
    fn ported_forward_and_inverse_round_trip() {
        for (lat, lon) in [
            (0.0, -75.2),   // sub-satellite point
            (25.0, -80.0),  // CONUS
            (47.0, -100.0), // Enderlin-ish
            (-15.0, -60.0), // southern hemisphere, on disk
            (10.0, -45.0),  // toward the eastern limb
        ] {
            let (x, y) = spherical_forward(lat, lon).expect("on disk");
            let (blat, blon) = spherical_inverse(x, y).expect("round trip");
            assert!((blat - lat).abs() < 1.0e-4, "lat {lat} -> {blat}");
            let dlon = (blon - lon + 180.0).rem_euclid(360.0) - 180.0;
            assert!(dlon.abs() < 1.0e-4, "lon {lon} -> {blon}");
        }
    }

    #[test]
    fn sub_satellite_point_maps_to_scan_origin() {
        let (x, y) = spherical_forward(0.0, -75.2).unwrap();
        assert!(x.abs() < 1.0e-9 && y.abs() < 1.0e-9, "({x}, {y})");
    }

    #[test]
    fn far_side_of_globe_is_rejected() {
        // Antipode of -75.2 is ~ +104.8; well past the limb from GOES-East.
        assert!(spherical_forward(0.0, 104.8).is_none());
        // Himawari cannot see North America.
        let himawari = |lat, lon| {
            lat_lon_to_scan_angles(PERSPECTIVE_HEIGHT_M, R_EARTH, R_EARTH, 140.7, lat, lon)
        };
        assert!(himawari(40.0, -100.0).is_none());
    }

    #[test]
    fn scan_grid_spans_rect_and_respects_cap() {
        let rect = ScanAngleRect {
            x_min: -0.01,
            x_max: 0.01,
            y_min: -0.005,
            y_max: 0.005,
        };
        // 0.02 rad / 28 urad ~= 715 -> 716 px, under the cap.
        let g = ScanGrid::from_rect(rect, VISIBLE_PITCH_RAD, MAX_AXIS);
        assert!(g.nx > 700 && g.nx < 720, "nx={}", g.nx);
        // Corners land on the rect edges.
        let (x0, y0) = g.scan_angle(0, 0);
        assert!((x0 - rect.x_min).abs() < 1e-12);
        assert!((y0 - rect.y_max).abs() < 1e-12);
        let (x1, y1) = g.scan_angle(g.nx - 1, g.ny - 1);
        assert!((x1 - rect.x_max).abs() < 1e-9);
        assert!((y1 - rect.y_min).abs() < 1e-9);
        // A tiny cap coarsens the pitch but still spans the rect.
        let capped = ScanGrid::from_rect(rect, VISIBLE_PITCH_RAD, 64);
        assert_eq!(capped.nx, 64);
        let (cx, _) = capped.scan_angle(capped.nx - 1, 0);
        assert!((cx - rect.x_max).abs() < 1e-9);
    }

    #[test]
    fn abi_2km_global_lattice_has_pug_half_pitch_origin_and_full_disk_endpoints() {
        let center = |index: i32| (index as f64 + 0.5) * IR_PITCH_RAD;
        assert_eq!(ABI_2KM_FULL_DISK_AXIS, 5424);
        assert!((center(ABI_2KM_LATTICE_INDEX_MIN) - -0.151_844).abs() < 1.0e-15);
        assert!((center(ABI_2KM_LATTICE_INDEX_MAX) - 0.151_844).abs() < 1.0e-15);

        // A crop straddling SSP must contain four pixels.  No pixel centre is at
        // scan (0, 0): SSP is their shared corner, exactly as on the ABI fixed grid.
        let around_ssp = ScanGrid::abi_2km_global_lattice(
            ScanAngleRect {
                x_min: -1.0e-12,
                x_max: 1.0e-12,
                y_min: -1.0e-12,
                y_max: 1.0e-12,
            },
            MAX_AXIS,
        )
        .unwrap();
        assert_eq!((around_ssp.nx, around_ssp.ny), (2, 2));
        assert_eq!(around_ssp.abi_2km_global_indices(), Some((-1, 0, -1, 0)));
        assert_eq!(around_ssp.scan_angle(0, 0), (-28.0e-6, 28.0e-6));
        assert_eq!(around_ssp.scan_angle(1, 1), (28.0e-6, -28.0e-6));
    }

    #[test]
    fn abi_2km_global_lattice_matches_official_conus_coordinate_crop() {
        // ABI CONUS 2-km coordinates carried by the GOES-19 M6 CMI reference:
        // x = -0.101332..0.038612 (2500 columns),
        // y =  0.128212..0.044268 (1500 north-first rows).  These are analytic
        // half-pitch coordinates, not values fitted to this test rectangle.
        let grid = ScanGrid::abi_2km_global_lattice(
            ScanAngleRect {
                x_min: -0.101_332,
                x_max: 0.038_612,
                y_min: 0.044_268,
                y_max: 0.128_212,
            },
            MAX_AXIS,
        )
        .unwrap();
        assert_eq!((grid.nx, grid.ny), (2500, 1500));
        assert_eq!(grid.abi_2km_global_indices(), Some((-1810, 689, 790, 2289)));
        assert_eq!(grid.pitch_x, 56.0e-6);
        assert_eq!(grid.pitch_y, 56.0e-6);
        assert!((grid.x_min - -0.101_332).abs() < 1.0e-15);
        assert!((grid.y_max - 0.128_212).abs() < 1.0e-15);
        let (x_last, y_last) = grid.scan_angle(grid.nx - 1, grid.ny - 1);
        assert!((x_last - 0.038_612).abs() < 1.0e-15);
        assert!((y_last - 0.044_268).abs() < 1.0e-15);
    }

    #[test]
    fn exact_goes_abi2km_snaps_but_model_sphere_and_native_stay_legacy() {
        let proj = MapProjection::lambert(30.0, 60.0, -97.5);
        let (nx, ny) = (120usize, 90usize);
        let georef = GridGeoref::new(proj, 59.5, 44.5, 39.0, -97.5, 3000.0, 3000.0);

        let abi =
            GeoCamera::for_navigation(SatellitePreset::GoesEast, GeoNavigation::GoesRAbiFixedGrid)
                .unwrap();
        let abi_rect = domain_scan_rect(&abi, &georef, nx, ny).unwrap();
        let snapped =
            build_surface_raster_mode(&abi, &georef, nx, ny, ResolutionMode::Abi2km, 0.0, MAX_AXIS)
                .unwrap();
        assert_eq!(
            snapped.scan,
            ScanGrid::abi_2km_global_lattice(abi_rect, MAX_AXIS).unwrap()
        );
        assert!(snapped.scan.abi_2km_global_indices().is_some());

        let model = GeoCamera::new(SatellitePreset::GoesEast);
        let model_rect = domain_scan_rect(&model, &georef, nx, ny).unwrap();
        let legacy = build_surface_raster_mode(
            &model,
            &georef,
            nx,
            ny,
            ResolutionMode::Abi2km,
            0.0,
            MAX_AXIS,
        )
        .unwrap();
        assert_eq!(
            legacy.scan,
            ScanGrid::from_rect(model_rect, IR_PITCH_RAD, MAX_AXIS)
        );

        let native =
            build_surface_raster_mode(&abi, &georef, nx, ny, ResolutionMode::Native, 0.0, MAX_AXIS)
                .unwrap();
        assert_eq!(native.scan, ScanGrid::native(abi_rect, nx, ny, MAX_AXIS));
    }

    #[test]
    fn build_surface_raster_over_a_conus_domain() {
        // A small Lambert CONUS-ish domain, center-anchored like ingest does.
        let proj = MapProjection::lambert(30.0, 60.0, -97.5);
        let (nx, ny) = (120usize, 90usize);
        let georef = GridGeoref::new(
            proj,
            (nx - 1) as f64 / 2.0,
            (ny - 1) as f64 / 2.0,
            39.0,
            -97.5,
            3000.0,
            3000.0,
        );
        let camera = GeoCamera::new(SatellitePreset::GoesEast);
        let raster =
            build_surface_raster(&camera, &georef, nx, ny, VISIBLE_PITCH_RAD, MAX_AXIS).unwrap();
        assert_eq!(raster.lat.len(), raster.nx * raster.ny);
        // The domain center pixel must be on earth and inside the domain.
        let on_earth = raster.lat.iter().filter(|v| v.is_finite()).count();
        assert!(on_earth > 0, "some pixels on earth");
        let in_domain = raster.grid_i.iter().filter(|v| v.is_finite()).count();
        assert!(in_domain > 0, "some pixels inside the WRF domain");
        // The bbox must bracket the anchor latitude.
        let (la_min, la_max, _, _) = raster.lat_lon_bbox().unwrap();
        assert!(la_min < 39.0 && la_max > 39.0, "bbox {la_min}..{la_max}");
    }

    #[test]
    fn himawari_domain_is_not_visible_from_conus() {
        let proj = MapProjection::lambert(30.0, 60.0, -97.5);
        let (nx, ny) = (60usize, 60usize);
        let georef = GridGeoref::new(proj, 29.5, 29.5, 39.0, -97.5, 3000.0, 3000.0);
        let camera = GeoCamera::new(SatellitePreset::Himawari);
        assert!(
            build_surface_raster(&camera, &georef, nx, ny, VISIBLE_PITCH_RAD, MAX_AXIS).is_none()
        );
    }

    #[test]
    fn presets_carry_expected_sub_lons() {
        assert_eq!(SatellitePreset::GoesEast.sub_lon_deg(), -75.2);
        assert_eq!(SatellitePreset::GoesWest.sub_lon_deg(), -137.0);
        assert_eq!(SatellitePreset::Himawari.sub_lon_deg(), 140.7);
    }

    #[test]
    fn native_scan_grid_is_one_pixel_per_cell() {
        let rect = ScanAngleRect {
            x_min: -0.02,
            x_max: 0.02,
            y_min: -0.015,
            y_max: 0.015,
        };
        // A 783x669 WRF grid (the owner's 500 m run) must produce a 783x669 raster:
        // one output pixel per cell, NOT the coarse fixed-pitch count.
        let (nx, ny) = (783usize, 669usize);
        let g = ScanGrid::native(rect, nx, ny, MAX_AXIS);
        assert_eq!(g.nx, nx, "native nx should equal the WRF cell count");
        assert_eq!(g.ny, ny, "native ny should equal the WRF cell count");
        // The per-axis pitch is span / (count - 1); the raster spans the full rect.
        let span_x = rect.x_max - rect.x_min;
        let span_y = rect.y_max - rect.y_min;
        assert!((g.pitch_x - span_x / (nx - 1) as f64).abs() < 1e-18);
        assert!((g.pitch_y - span_y / (ny - 1) as f64).abs() < 1e-18);
        let (x0, y0) = g.scan_angle(0, 0);
        assert!((x0 - rect.x_min).abs() < 1e-12 && (y0 - rect.y_max).abs() < 1e-12);
        let (x1, y1) = g.scan_angle(g.nx - 1, g.ny - 1);
        assert!((x1 - rect.x_max).abs() < 1e-9, "east edge {x1}");
        assert!((y1 - rect.y_min).abs() < 1e-9, "south edge {y1}");
    }

    #[test]
    fn native_scan_grid_clamps_oversized_domain() {
        let rect = ScanAngleRect {
            x_min: -0.05,
            x_max: 0.05,
            y_min: -0.05,
            y_max: 0.05,
        };
        // A WRF grid wider than the 4096 cap clamps to MAX_AXIS and coarsens the
        // pitch (the honest exception), still spanning the full rect.
        let g = ScanGrid::native(rect, 6000, 6000, MAX_AXIS);
        assert_eq!(g.nx, MAX_AXIS);
        assert_eq!(g.ny, MAX_AXIS);
        let (x1, _) = g.scan_angle(g.nx - 1, 0);
        assert!((x1 - rect.x_max).abs() < 1e-9, "still spans the rect: {x1}");
        // A smaller explicit cap engages too.
        let capped = ScanGrid::native(rect, 6000, 6000, 512);
        assert_eq!(capped.nx, 512);
        assert_eq!(capped.ny, 512);
    }

    #[test]
    fn resolution_modes_expose_expected_pitches() {
        assert_eq!(ResolutionMode::Native.fixed_pitch_rad(), None);
        assert_eq!(
            ResolutionMode::Abi1km.fixed_pitch_rad(),
            Some(VISIBLE_PITCH_RAD)
        );
        assert_eq!(ResolutionMode::Abi2km.fixed_pitch_rad(), Some(IR_PITCH_RAD));
        assert_eq!(VISIBLE_PITCH_RAD, 28.0e-6);
        assert_eq!(IR_PITCH_RAD, 56.0e-6);
        // The ABI modes still produce the OLD coarser fixed-pitch grids (unchanged).
        let rect = ScanAngleRect {
            x_min: -0.01,
            x_max: 0.01,
            y_min: -0.005,
            y_max: 0.005,
        };
        let abi1 = ResolutionMode::Abi1km.scan_grid(rect, 1000, 1000, MAX_AXIS);
        let abi1_ref = ScanGrid::from_rect(rect, VISIBLE_PITCH_RAD, MAX_AXIS);
        assert_eq!(abi1.nx, abi1_ref.nx);
        assert_eq!(abi1.ny, abi1_ref.ny);
        // 2 km pitch is coarser (fewer pixels) than 1 km on the same rect.
        let abi2 = ResolutionMode::Abi2km.scan_grid(rect, 1000, 1000, MAX_AXIS);
        assert!(
            abi2.nx < abi1.nx,
            "2 km ({}) coarser than 1 km ({})",
            abi2.nx,
            abi1.nx
        );
    }

    #[test]
    fn native_pitch_from_count_round_trips() {
        let rect = ScanAngleRect {
            x_min: 0.0,
            x_max: 0.037,
            y_min: 0.0,
            y_max: 0.021,
        };
        let span_x = rect.x_max - rect.x_min;
        let span_y = rect.y_max - rect.y_min;
        for &(nx, ny) in &[(783usize, 669usize), (199, 199), (2, 2)] {
            // native() sets the count exactly and the pitch is span / (count - 1).
            let g = ScanGrid::native(rect, nx, ny, MAX_AXIS);
            assert_eq!(g.nx, nx);
            assert_eq!(g.ny, ny);
            assert!((g.pitch_x * (nx - 1) as f64 - span_x).abs() < 1e-12);
            assert!((g.pitch_y * (ny - 1) as f64 - span_y).abs() < 1e-12);
            // Feeding that pitch back through from_rect recovers the count within 1 px
            // (the float ceil boundary), confirming pitch ~= extent / native_count.
            let back = ScanGrid::from_rect(rect, g.pitch_x, MAX_AXIS);
            assert!(
                back.nx.abs_diff(nx) <= 1,
                "from_rect nx {} vs native {}",
                back.nx,
                nx
            );
        }
    }

    #[test]
    fn build_surface_raster_mode_native_matches_grid() {
        // Native mode over a CONUS-ish Lambert domain must produce a raster whose
        // pixel counts equal the WRF cell counts (one pixel per cell), whereas the
        // fixed ABI 1 km pitch undersamples the same fine grid.
        let proj = MapProjection::lambert(30.0, 60.0, -97.5);
        let (nx, ny) = (400usize, 300usize);
        let georef = GridGeoref::new(
            proj,
            (nx - 1) as f64 / 2.0,
            (ny - 1) as f64 / 2.0,
            39.0,
            -97.5,
            500.0, // 500 m cells (finer than the 1 km ABI pitch)
            500.0,
        );
        let camera = GeoCamera::new(SatellitePreset::GoesEast);
        let native = build_surface_raster_mode(
            &camera,
            &georef,
            nx,
            ny,
            ResolutionMode::Native,
            0.0,
            MAX_AXIS,
        )
        .unwrap();
        assert_eq!(native.nx, nx, "native raster nx == WRF nx");
        assert_eq!(native.ny, ny, "native raster ny == WRF ny");
        // The fixed 1 km ABI raster undersamples this 500 m domain: fewer pixels.
        let abi = build_surface_raster_mode(
            &camera,
            &georef,
            nx,
            ny,
            ResolutionMode::Abi1km,
            0.0,
            MAX_AXIS,
        )
        .unwrap();
        assert!(
            abi.nx < native.nx,
            "ABI 1 km ({}) undersamples the 500 m grid vs native ({})",
            abi.nx,
            native.nx
        );
        // Native still lands on-earth / in-domain pixels.
        let in_domain = native.grid_i.iter().filter(|v| v.is_finite()).count();
        assert!(in_domain > 0, "native raster has in-domain pixels");
    }

    #[test]
    fn view_mode_labels_and_slugs() {
        assert_eq!(ViewMode::ALL.len(), 2);
        assert_eq!(ViewMode::Geostationary.slug(), "geo");
        assert_eq!(ViewMode::TopDownMap.slug(), "topdown");
        assert!(
            ViewMode::TopDownMap
                .label()
                .to_ascii_lowercase()
                .contains("top-down"),
            "label: {}",
            ViewMode::TopDownMap.label()
        );
    }

    #[test]
    fn topdown_nadir_ray_points_straight_down_at_the_location() {
        for &(lat, lon) in &[(0.0, 0.0), (45.0, -100.0), (-30.0, 60.0), (39.0, -97.5)] {
            let (cam, view) = topdown_nadir_ray(lat, lon);
            // The camera sits on the local vertical at the expected altitude.
            let cam_r = (cam[0] * cam[0] + cam[1] * cam[1] + cam[2] * cam[2]).sqrt();
            assert!(
                (cam_r - (R_EARTH + TOPDOWN_CAMERA_ALTITUDE_M)).abs() < 1.0,
                "cam radius {cam_r}"
            );
            // The view is straight down: view . up == -1 (up = the normalized camera).
            let up = [cam[0] / cam_r, cam[1] / cam_r, cam[2] / cam_r];
            let vdotup = view[0] * up[0] + view[1] * up[1] + view[2] * up[2];
            assert!((vdotup + 1.0).abs() < 1e-9, "view.up {vdotup}");
            // The camera is above the top of the atmosphere (full-column integration).
            assert!(cam_r > R_EARTH + 100_000.0, "cam not above the atmosphere");
            // Marching straight down to the ground sphere lands exactly at (lat, lon).
            let t = cam_r - R_EARTH;
            let g = [
                cam[0] + view[0] * t,
                cam[1] + view[1] * t,
                cam[2] + view[2] * t,
            ];
            let gr = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
            assert!((gr - R_EARTH).abs() < 1.0, "ground radius {gr}");
            let glat = (g[2] / gr).clamp(-1.0, 1.0).asin().to_degrees();
            let glon = g[1].atan2(g[0]).to_degrees();
            assert!((glat - lat).abs() < 1e-6, "lat {lat} -> {glat}");
            let dlon = (glon - lon + 180.0).rem_euclid(360.0) - 180.0;
            assert!(dlon.abs() < 1e-6, "lon {lon} -> {glon}");
        }
    }

    #[test]
    fn map_raster_round_trips_to_grid_indices() {
        // A CONUS-ish Lambert domain: build the native top-down map, then map each
        // pixel (i, j) -> lat/lon -> forward -> (i, j) and confirm it recovers the
        // pixel's own grid index far tighter than the 0.05-cell projection ratchet.
        let proj = MapProjection::lambert(30.0, 60.0, -97.5);
        let (nx, ny) = (120usize, 90usize);
        let georef = GridGeoref::new(
            proj,
            (nx - 1) as f64 / 2.0,
            (ny - 1) as f64 / 2.0,
            39.0,
            -97.5,
            3000.0,
            3000.0,
        );
        let map = build_map_raster(&georef, nx, ny, nx, ny, 0.0).unwrap();
        assert_eq!(map.nx, nx);
        assert_eq!(map.ny, ny);
        let mut worst = 0.0f64;
        for py in (0..ny).step_by(7) {
            for px in (0..nx).step_by(7) {
                let idx = py * nx + px;
                let (la, lo) = (map.lat[idx], map.lon[idx]);
                assert!(la.is_finite() && lo.is_finite());
                let (fi, fj) = georef.forward(la as f64, lo as f64);
                worst = worst
                    .max((fi - map.grid_i[idx] as f64).abs())
                    .max((fj - map.grid_j[idx] as f64).abs());
            }
        }
        assert!(worst < 0.02, "map round-trip worst {worst} cells");
    }

    #[test]
    fn topdown_native_mode_is_byte_exact_with_legacy_map_builder() {
        let proj = MapProjection::lambert(30.0, 60.0, -97.5);
        let (nx, ny) = (37usize, 23usize);
        let georef = GridGeoref::new(
            proj,
            (nx - 1) as f64 / 2.0,
            (ny - 1) as f64 / 2.0,
            39.0,
            -97.5,
            3000.0,
            3000.0,
        );
        let legacy = build_map_raster(&georef, nx, ny, nx, ny, 0.0).unwrap();
        let selected =
            build_map_raster_mode(&georef, nx, ny, 3000.0, 3000.0, ResolutionMode::Native, 0.0)
                .unwrap();
        assert_eq!((selected.nx, selected.ny), (legacy.nx, legacy.ny));
        assert_eq!(selected.lat, legacy.lat);
        assert_eq!(selected.lon, legacy.lon);
        assert_eq!(selected.grid_i, legacy.grid_i);
        assert_eq!(selected.grid_j, legacy.grid_j);
    }

    #[test]
    fn topdown_abi_modes_use_physical_three_km_domain_span() {
        // 11 x 7 points at 3 km span 30 x 18 km centre-to-centre.  A sample at
        // both edges therefore requires 31 x 19 pixels at 1 km, and 16 x 10 at
        // 2 km.  This also proves x/y are sized independently.
        assert_eq!(
            topdown_resolution_counts(11, 7, 3000.0, 3000.0, ResolutionMode::Abi1km),
            (31, 19)
        );
        assert_eq!(
            topdown_resolution_counts(11, 7, 3000.0, 3000.0, ResolutionMode::Abi2km),
            (16, 10)
        );

        // Fractional pitch ratios round upward so the selected resolution still
        // spans the complete domain without becoming coarser than requested.
        assert_eq!(
            topdown_resolution_counts(10, 6, 3000.0, 3000.0, ResolutionMode::Abi2km),
            (15, 9)
        );
        // A huge fixed-resolution request is capped with ONE scale factor. The
        // shorter axis shrinks proportionally instead of being independently
        // clamped to 4096 (which would make rectangular physical pixels).
        assert_eq!(
            topdown_resolution_counts(5000, 4000, 3000.0, 3000.0, ResolutionMode::Abi1km),
            (MAX_AXIS, 3277)
        );

        // The actual 3-km HRRR native-grid geometry: uncapped ABI1km would be
        // 5394 x 3175. X hits MAX_AXIS, while Y uses the same ~1.317 km effective
        // pitch and preserves the physical aspect ratio.
        let hrrr_dx = 2_999.421_304_743_559;
        let hrrr = topdown_resolution_counts(1799, 1059, hrrr_dx, hrrr_dx, ResolutionMode::Abi1km);
        assert_eq!(hrrr, (MAX_AXIS, 2411));
        let pitch_x = (1799 - 1) as f64 * hrrr_dx / (hrrr.0 - 1) as f64;
        let pitch_y = (1059 - 1) as f64 * hrrr_dx / (hrrr.1 - 1) as f64;
        assert!(
            (pitch_x - pitch_y).abs() / pitch_x < 2.0e-4,
            "uniform-cap pitches differ: {pitch_x} m vs {pitch_y} m"
        );
    }

    #[test]
    fn topdown_fixed_resolution_keeps_existing_margin_growth() {
        let proj = MapProjection::lambert(30.0, 60.0, -97.5);
        let georef = GridGeoref::new(proj, 5.0, 3.0, 39.0, -97.5, 3000.0, 3000.0);
        let map =
            build_map_raster_mode(&georef, 11, 7, 3000.0, 3000.0, ResolutionMode::Abi1km, 0.25)
                .unwrap();
        // Base 31 x 19, grown by the pre-existing (1 + 2m) rule.
        assert_eq!((map.nx, map.ny), (47, 29));
        assert!(map.grid_i.iter().any(|v| v.is_nan()));
        assert!(map.grid_i.iter().any(|v| v.is_finite()));
    }

    #[test]
    fn map_raster_is_north_up_and_fills_the_domain() {
        let proj = MapProjection::lambert(30.0, 60.0, -97.5);
        let (nx, ny) = (60usize, 40usize);
        let georef = GridGeoref::new(
            proj,
            (nx - 1) as f64 / 2.0,
            (ny - 1) as f64 / 2.0,
            39.0,
            -97.5,
            3000.0,
            3000.0,
        );
        // A canvas-sized output (NOT the native grid) still fills the frame with the
        // domain, north-up — the whole raster is in-domain (no space).
        let (ox, oy) = (100usize, 80usize);
        let map = build_map_raster(&georef, nx, ny, ox, oy, 0.0).unwrap();
        assert!(map.lat.iter().all(|v| v.is_finite()), "all pixels on earth");
        assert!(
            map.grid_i.iter().all(|v| v.is_finite()),
            "all pixels in domain"
        );
        // Row 0 is north of the last row (a mid-latitude northern-hemisphere domain).
        let top_mid = map.lat[ox / 2];
        let bot_mid = map.lat[(oy - 1) * ox + ox / 2];
        assert!(
            top_mid > bot_mid,
            "row 0 lat {top_mid} not north of last row {bot_mid}"
        );
        // North-up grid indices: row 0 = max j, last row = min j; col 0 = min i, last = max i.
        assert!(
            map.grid_j[0] > (ny - 1) as f32 - 0.5,
            "north row j {}",
            map.grid_j[0]
        );
        assert!(
            map.grid_j[(oy - 1) * ox] < 0.5,
            "south row j {}",
            map.grid_j[(oy - 1) * ox]
        );
        assert!(map.grid_i[0] < 0.5, "west edge i {}", map.grid_i[0]);
        assert!(
            map.grid_i[ox - 1] > (nx - 1) as f32 - 0.5,
            "east edge i {}",
            map.grid_i[ox - 1]
        );
    }

    // ── zoom-out / domain-margin feature ────────────────────────────────────────

    #[test]
    fn grow_scan_rect_is_identity_at_zero_and_symmetric_growth() {
        let rect = ScanAngleRect {
            x_min: -0.02,
            x_max: 0.03,
            y_min: -0.01,
            y_max: 0.04,
        };
        // margin 0 (and any non-positive) is a byte-identical no-op.
        assert_eq!(grow_scan_rect(rect, 0.0), rect);
        assert_eq!(grow_scan_rect(rect, -0.5), rect);
        // margin 0.2 adds 20% of each span on EACH side.
        let g = grow_scan_rect(rect, 0.2);
        let (sx, sy) = (rect.x_max - rect.x_min, rect.y_max - rect.y_min);
        assert!((g.x_min - (rect.x_min - 0.2 * sx)).abs() < 1e-15);
        assert!((g.x_max - (rect.x_max + 0.2 * sx)).abs() < 1e-15);
        assert!((g.y_min - (rect.y_min - 0.2 * sy)).abs() < 1e-15);
        assert!((g.y_max - (rect.y_max + 0.2 * sy)).abs() < 1e-15);
        // The grown span is (1 + 2*margin) x the original, so the domain occupies the
        // center 1/(1+2*margin) of the frame.
        assert!(((g.x_max - g.x_min) - 1.4 * sx).abs() < 1e-15);
        assert!(((g.y_max - g.y_min) - 1.4 * sy).abs() < 1e-15);
    }

    #[test]
    fn extended_native_counts_scale_with_margin() {
        assert_eq!(extended_native_counts(800, 600, 0.0), (800, 600));
        assert_eq!(extended_native_counts(800, 600, -1.0), (800, 600));
        // 0.25 margin -> 1 + 2*0.25 = 1.5x on each axis.
        assert_eq!(extended_native_counts(800, 600, 0.25), (1200, 900));
        // Never below 2.
        assert_eq!(extended_native_counts(2, 2, 0.0), (2, 2));
    }

    /// A CONUS-ish Lambert georef + a GOES-East camera for the margin raster tests.
    fn margin_test_scene() -> (GeoCamera, GridGeoref, usize, usize) {
        let proj = MapProjection::lambert(30.0, 60.0, -97.5);
        let (nx, ny) = (200usize, 150usize);
        let georef = GridGeoref::new(
            proj,
            (nx - 1) as f64 / 2.0,
            (ny - 1) as f64 / 2.0,
            39.0,
            -97.5,
            3000.0,
            3000.0,
        );
        (GeoCamera::new(SatellitePreset::GoesEast), georef, nx, ny)
    }

    #[test]
    fn geo_margin_zero_is_identical_and_margin_extends_the_extent() {
        let (camera, georef, nx, ny) = margin_test_scene();
        // margin 0.0 reproduces the pre-margin native raster EXACTLY (dims + scan grid).
        let base = build_surface_raster_mode(
            &camera,
            &georef,
            nx,
            ny,
            ResolutionMode::Native,
            0.0,
            MAX_AXIS,
        )
        .unwrap();
        assert_eq!(
            (base.nx, base.ny),
            (nx, ny),
            "margin 0 is native one-px-per-cell"
        );

        // margin 0.3 grows the raster (Native keeps ~native pitch) and widens the lat/lon
        // span (more surrounding earth), while the domain still lands in-domain pixels.
        let m = build_surface_raster_mode(
            &camera,
            &georef,
            nx,
            ny,
            ResolutionMode::Native,
            0.3,
            MAX_AXIS,
        )
        .unwrap();
        let (exp_nx, exp_ny) = extended_native_counts(nx, ny, 0.3);
        assert_eq!(
            (m.nx, m.ny),
            (exp_nx, exp_ny),
            "margin native count = 1+2*0.3"
        );
        // The margin raster's lat/lon bbox is strictly LARGER than the edge-to-edge one
        // (the Blue Marble crop, derived from this bbox, then covers the extended extent).
        let (bla0, bla1, blo0, blo1) = base.lat_lon_bbox().unwrap();
        let (mla0, mla1, mlo0, mlo1) = m.lat_lon_bbox().unwrap();
        assert!(
            mla0 < bla0 && mla1 > bla1,
            "lat span not extended: {mla0}..{mla1} vs {bla0}..{bla1}"
        );
        assert!(
            mlo0 < blo0 && mlo1 > blo1,
            "lon span not extended: {mlo0}..{mlo1} vs {blo0}..{blo1}"
        );
        // The margin raster carries BOTH in-domain pixels (grid_i finite) AND out-of-domain
        // margin pixels (on earth but grid_i NaN): the margin is real earth, no WRF data.
        let in_domain = m.grid_i.iter().filter(|v| v.is_finite()).count();
        let on_earth = m.lat.iter().filter(|v| v.is_finite()).count();
        assert!(in_domain > 0, "no in-domain pixels");
        assert!(
            on_earth > in_domain,
            "no out-of-domain margin pixels (on earth, no WRF)"
        );
    }

    #[test]
    fn geo_margin_native_clamps_at_max_axis() {
        // A grid near the cap plus a large margin forces the Native MAX_AXIS clamp (the
        // honest exception). 3000x3000 + 0.5 margin -> target 6000x6000 -> clamped to 4096.
        let proj = MapProjection::lambert(30.0, 60.0, -97.5);
        let (nx, ny) = (3000usize, 3000usize);
        let georef = GridGeoref::new(
            proj,
            (nx - 1) as f64 / 2.0,
            (ny - 1) as f64 / 2.0,
            39.0,
            -97.5,
            1000.0,
            1000.0,
        );
        let camera = GeoCamera::new(SatellitePreset::GoesEast);
        let (target_nx, _) = extended_native_counts(nx, ny, 0.5);
        assert!(
            target_nx > MAX_AXIS,
            "target {target_nx} should exceed the cap"
        );
        let m = build_surface_raster_mode(
            &camera,
            &georef,
            nx,
            ny,
            ResolutionMode::Native,
            0.5,
            MAX_AXIS,
        )
        .unwrap();
        assert_eq!(m.nx, MAX_AXIS, "clamped to the per-axis cap");
        assert_eq!(m.ny, MAX_AXIS);
    }

    #[test]
    fn map_raster_margin_zero_is_identical_and_margin_extends() {
        let proj = MapProjection::lambert(30.0, 60.0, -97.5);
        let (nx, ny) = (120usize, 90usize);
        let georef = GridGeoref::new(
            proj,
            (nx - 1) as f64 / 2.0,
            (ny - 1) as f64 / 2.0,
            39.0,
            -97.5,
            3000.0,
            3000.0,
        );
        // margin 0 -> byte-identical to the edge-to-edge native map (dims + every value).
        let base = build_map_raster(&georef, nx, ny, nx, ny, 0.0).unwrap();
        assert_eq!((base.nx, base.ny), (nx, ny));
        assert!(
            base.grid_i.iter().all(|v| v.is_finite()),
            "margin 0 = all in-domain"
        );

        // margin 0.25 grows the output dims (native pitch kept) and adds an out-of-domain
        // real-earth margin: margin pixels have finite lat/lon but NaN grid_i (no WRF data).
        let m = build_map_raster(&georef, nx, ny, nx, ny, 0.25).unwrap();
        assert!(
            m.nx > nx && m.ny > ny,
            "margin map not extended: {}x{}",
            m.nx,
            m.ny
        );
        assert!(
            m.lat.iter().all(|v| v.is_finite()),
            "map margin pixels are on earth"
        );
        let in_domain = m.grid_i.iter().filter(|v| v.is_finite()).count();
        assert!(in_domain > 0, "no in-domain pixels");
        assert!(
            in_domain < m.nx * m.ny,
            "no out-of-domain margin pixels (all {} in-domain)",
            in_domain
        );
        // The corners are the extended edge: NW corner samples grid index i < 0, j > ny-1
        // (outside the domain) -> NaN grid, but a valid lat/lon.
        assert!(m.grid_i[0].is_nan(), "NW corner should be out-of-domain");
        assert!(
            m.lat[0].is_finite() && m.lon[0].is_finite(),
            "NW corner still on earth"
        );
        // The domain centre pixel is still in-domain.
        let cidx = (m.ny / 2) * m.nx + m.nx / 2;
        assert!(m.grid_i[cidx].is_finite(), "centre should be in-domain");
    }

    #[test]
    fn map_raster_edge_samples_are_inset_off_the_inclusion_boundary() {
        // WS2 row-0 speckle fix: the native map's outermost row/columns must NOT sample
        // exactly on the domain-inclusion boundary — the cloud/IR marches reconstruct
        // the grid index from the raster's f32 lat/lon, and the quantization error
        // oscillates the recovered index across the inclusive in-domain test (hard dash
        // speckle along row 0 / col 0 / col nx-1, measured 50-300x interior roughness).
        // In-domain edge sample centres are inset by MAP_EDGE_INSET_CELLS; interior
        // samples are untouched; margin (out-of-domain) samples keep raw positions.
        let proj = MapProjection::lambert(30.0, 60.0, -97.5);
        let (nx, ny) = (120usize, 90usize);
        let georef = GridGeoref::new(
            proj,
            (nx - 1) as f64 / 2.0,
            (ny - 1) as f64 / 2.0,
            39.0,
            -97.5,
            3000.0,
            3000.0,
        );
        let map = build_map_raster(&georef, nx, ny, nx, ny, 0.0).unwrap();
        let (di, dj) = ((nx - 1) as f64, (ny - 1) as f64);
        let e = MAP_EDGE_INSET_CELLS;
        let close = |a: f32, b: f64| (a as f64 - b).abs() < 1e-4;
        // The outermost samples sit exactly at the inset, not on the boundary.
        assert!(close(map.grid_i[0], e), "west edge i {}", map.grid_i[0]);
        assert!(
            close(map.grid_i[nx - 1], di - e),
            "east edge i {}",
            map.grid_i[nx - 1]
        );
        assert!(
            close(map.grid_j[0], dj - e),
            "north (row 0) j {}",
            map.grid_j[0]
        );
        assert!(
            close(map.grid_j[(ny - 1) * nx], e),
            "south (last row) j {}",
            map.grid_j[(ny - 1) * nx]
        );
        // A row-0 interior column is inset in j ONLY (i keeps its linear position).
        let mid = nx / 2;
        assert!(close(map.grid_i[mid], mid as f64), "row-0 mid i untouched");
        assert!(close(map.grid_j[mid], dj - e), "row-0 mid j inset");
        // An interior sample is untouched entirely (north-up: row py samples
        // j = ny-1-py; i is unflipped).
        let cidx = (ny / 2) * nx + nx / 2;
        assert!(close(map.grid_i[cidx], (nx / 2) as f64), "interior i");
        assert!(
            close(map.grid_j[cidx], (ny - 1 - ny / 2) as f64),
            "interior j"
        );
        // The reconstruction the marches perform (f32 lat/lon -> forward) now lands
        // STRICTLY inside the domain for every edge sample — the speckle mechanism
        // (crossing the inclusive boundary) is impossible.
        for &idx in &[0usize, nx - 1, (ny - 1) * nx, ny * nx - 1] {
            let (fi, fj) = georef.forward(map.lat[idx] as f64, map.lon[idx] as f64);
            assert!(
                fi > 0.0 && fi < di && fj > 0.0 && fj < dj,
                "edge sample {idx} must reconstruct strictly in-domain: ({fi}, {fj})"
            );
        }
        // Margin: out-of-domain samples keep raw (un-inset) positions — NaN grid,
        // finite lat/lon (the no-weather margin contract is unchanged).
        let m = build_map_raster(&georef, nx, ny, nx, ny, 0.25).unwrap();
        assert!(m.grid_i[0].is_nan(), "margin corner stays out-of-domain");
        assert!(m.lat[0].is_finite() && m.lon[0].is_finite());
    }

    #[test]
    fn map_pixel_edge_bounds_reduce_to_the_half_cell_box_at_margin_zero() {
        // At native output (out == domain) with margin 0 the edge bounds are exactly the
        // domain's half-cell-padded box (-0.5 .. n-0.5) — the pre-margin imshow extent.
        let (b0, b1, b2, b3) = map_pixel_edge_index_bounds(120, 90, 120, 90, 0.0);
        assert!((b0 - -0.5).abs() < 1e-12, "i_min {b0}");
        assert!((b1 - 119.5).abs() < 1e-12, "i_max {b1}");
        assert!((b2 - -0.5).abs() < 1e-12, "j_min {b2}");
        assert!((b3 - 89.5).abs() < 1e-12, "j_max {b3}");
        // With a margin the bounds extend beyond the domain box on every side.
        let (m0, m1, m2, m3) = map_pixel_edge_index_bounds(120, 90, 156, 117, 0.3);
        assert!(
            m0 < -0.5 && m1 > 119.5 && m2 < -0.5 && m3 > 89.5,
            "bounds not extended"
        );
    }

    // ── free perspective camera (tier 2) ───────────────────────────────────────

    /// An eye directly above a look point (a pure nadir shot). Odd dims so the
    /// middle pixel's ray is exactly the optical axis.
    fn nadir_camera(lat: f64, lon: f64, alt_m: f64) -> PerspectiveCamera {
        PerspectiveCamera {
            eye_lat_deg: lat,
            eye_lon_deg: lon,
            eye_alt_m: alt_m,
            look_lat_deg: lat,
            look_lon_deg: lon,
            look_alt_m: 0.0,
            fov_deg: 40.0,
            width: 641,
            height: 481,
        }
    }

    #[test]
    fn perspective_nadir_centre_ray_matches_the_topdown_nadir_ray() {
        // The nadir-degenerate perspective camera must reproduce the top-down ray at
        // the same point: the middle pixel's ray equals `topdown_nadir_ray`'s view
        // (straight down the local vertical), and the basis is orthonormal despite
        // the up-hint degeneracy (local north takes over).
        let cam = nadir_camera(46.9, -97.6, 150_000.0);
        let basis = cam.basis().expect("valid camera");
        let centre = basis.pixel_ray(cam.width / 2, cam.height / 2);
        let (_, nadir_view) = topdown_nadir_ray(46.9, -97.6);
        let dot = centre[0] * nadir_view[0] + centre[1] * nadir_view[1] + centre[2] * nadir_view[2];
        assert!(dot > 1.0 - 1e-12, "centre ray != nadir view (dot {dot})");
        // Orthonormal basis.
        assert!((len3(basis.forward) - 1.0).abs() < 1e-12);
        assert!((len3(basis.right) - 1.0).abs() < 1e-12);
        assert!((len3(basis.up) - 1.0).abs() < 1e-12);
        assert!(dot3(basis.forward, basis.right).abs() < 1e-12);
        assert!(dot3(basis.forward, basis.up).abs() < 1e-12);
        assert!(dot3(basis.right, basis.up).abs() < 1e-12);
        // North-up nadir image: image up projects onto local north (positive z-ish
        // component toward the pole at this northern latitude).
        let north = {
            let (sla, cla) = 46.9f64.to_radians().sin_cos();
            let (slo, clo) = (-97.6f64).to_radians().sin_cos();
            [-sla * clo, -sla * slo, cla]
        };
        assert!(
            dot3(basis.up, north) > 0.999,
            "nadir image should be north-up: {}",
            dot3(basis.up, north)
        );
    }

    #[test]
    fn perspective_rays_are_unit_and_span_the_horizontal_fov() {
        // An oblique camera: every pixel ray is unit length, and the angle between
        // the row-centre edge pixels' rays matches the pixel-centre FOV
        // (2 * atan(tan(fov/2) * (1 - 1/width))) — the pure-geometry FOV invariant.
        let cam = PerspectiveCamera {
            eye_lat_deg: 45.5,
            eye_lon_deg: -99.0,
            eye_alt_m: 150_000.0,
            look_lat_deg: 46.9,
            look_lon_deg: -97.6,
            look_alt_m: 0.0,
            fov_deg: 50.0,
            width: 640,
            height: 360,
        };
        let basis = cam.basis().expect("valid camera");
        for &(px, py) in &[
            (0usize, 0usize),
            (639, 0),
            (0, 359),
            (639, 359),
            (320, 180),
            (17, 251),
        ] {
            let v = basis.pixel_ray(px, py);
            assert!((len3(v) - 1.0).abs() < 1e-12, "ray ({px},{py}) not unit");
        }
        let mid_row = cam.height / 2;
        let left = basis.pixel_ray(0, mid_row);
        let right = basis.pixel_ray(cam.width - 1, mid_row);
        let angle = dot3(left, right).clamp(-1.0, 1.0).acos().to_degrees();
        let tan_half = (cam.fov_deg.to_radians() * 0.5).tan();
        let edge = tan_half * (1.0 - 1.0 / cam.width as f64);
        let expected = 2.0 * edge.atan().to_degrees();
        // The mid row sits half a pixel off the optical axis (even height), so the
        // edge-to-edge angle is a hair under the in-plane value; a loose 0.1-deg
        // tolerance covers it while still pinning the FOV scale.
        assert!(
            (angle - expected).abs() < 0.1,
            "edge-to-edge angle {angle} != expected {expected}"
        );
        // Left/right symmetry about the forward axis.
        assert!(
            (dot3(left, basis.forward) - dot3(right, basis.forward)).abs() < 1e-9,
            "rays not symmetric about forward"
        );
    }

    #[test]
    fn perspective_centre_pixel_hits_the_look_point() {
        // The camera looks AT a ground point: the middle pixel's ground intersection
        // must be that point (the ray passes through it and it is the first sphere
        // hit for a below-horizon target). Pure geometry via the raster builder.
        let proj = MapProjection::lambert(30.0, 60.0, -97.5);
        let (nx, ny) = (120usize, 90usize);
        let georef = GridGeoref::new(
            proj,
            (nx - 1) as f64 / 2.0,
            (ny - 1) as f64 / 2.0,
            39.0,
            -97.5,
            3000.0,
            3000.0,
        );
        let cam = PerspectiveCamera {
            eye_lat_deg: 37.8,
            eye_lon_deg: -99.1,
            eye_alt_m: 150_000.0,
            look_lat_deg: 39.0,
            look_lon_deg: -97.5,
            look_alt_m: 0.0,
            fov_deg: 45.0,
            width: 321,
            height: 241,
        };
        let basis = cam.basis().unwrap();
        let raster = build_perspective_raster(&basis, &georef, nx, ny);
        let idx = (cam.height / 2) * cam.width + cam.width / 2;
        let (la, lo) = (raster.lat[idx] as f64, raster.lon[idx] as f64);
        assert!(
            (la - 39.0).abs() < 1e-6 && (lo - -97.5).abs() < 1e-6,
            "centre pixel hit ({la}, {lo}) != the look point (39, -97.5)"
        );
        // And that point is the domain centre -> the fractional grid index is the
        // centre cell.
        let (fi, fj) = (raster.grid_i[idx] as f64, raster.grid_j[idx] as f64);
        assert!(
            (fi - (nx - 1) as f64 / 2.0).abs() < 1e-3 && (fj - (ny - 1) as f64 / 2.0).abs() < 1e-3,
            "centre grid index ({fi}, {fj})"
        );
    }

    #[test]
    fn perspective_raster_marks_sky_above_the_horizon() {
        // A LOW-oblique camera aimed at the horizon: the top image rows are SKY
        // (NaN — no ground intersection), the bottom rows are ground, and the sky
        // fraction is substantial. This is the raster contract the render relies on
        // (sky pixels composite the limb/space path; ground pixels the surface).
        let proj = MapProjection::lambert(30.0, 60.0, -97.5);
        let georef = GridGeoref::new(proj, 59.5, 44.5, 39.0, -97.5, 3000.0, 3000.0);
        let cam = PerspectiveCamera {
            eye_lat_deg: 38.0,
            eye_lon_deg: -99.0,
            eye_alt_m: 20_000.0,
            // Look far past the horizon at altitude: the upper half is sky.
            look_lat_deg: 43.0,
            look_lon_deg: -92.0,
            look_alt_m: 150_000.0,
            fov_deg: 60.0,
            width: 160,
            height: 120,
        };
        let basis = cam.basis().unwrap();
        let raster = build_perspective_raster(&basis, &georef, 120, 90);
        let row_finite = |py: usize| {
            (0..raster.nx)
                .filter(|px| raster.lat[py * raster.nx + px].is_finite())
                .count()
        };
        assert_eq!(row_finite(0), 0, "the top row must be sky (no ground hit)");
        assert_eq!(
            row_finite(raster.ny - 1),
            raster.nx,
            "the bottom row must be ground"
        );
        let sky = raster.lat.iter().filter(|v| !v.is_finite()).count();
        let frac = sky as f64 / (raster.nx * raster.ny) as f64;
        assert!(
            (0.2..=0.95).contains(&frac),
            "sky fraction {frac} implausible for a horizon shot"
        );
    }

    #[test]
    fn perspective_camera_validation_rejects_bad_input() {
        let good = nadir_camera(46.9, -97.6, 150_000.0);
        assert!(good.validate().is_ok());
        let mut c = good;
        c.eye_alt_m = 0.0;
        assert!(c.validate().is_err(), "grounded eye must be rejected");
        c = good;
        c.fov_deg = 0.5;
        assert!(c.validate().is_err(), "sub-degree fov must be rejected");
        c = good;
        c.fov_deg = 179.0;
        assert!(c.validate().is_err(), "near-180 fov must be rejected");
        c = good;
        c.width = 1;
        assert!(c.validate().is_err(), "degenerate width must be rejected");
        c = good;
        c.eye_lon_deg = f64::NAN;
        assert!(c.validate().is_err(), "NaN must be rejected");
        c = good;
        c.look_lat_deg = c.eye_lat_deg;
        c.look_lon_deg = c.eye_lon_deg;
        c.look_alt_m = c.eye_alt_m;
        assert!(c.validate().is_err(), "eye == look must be rejected");
        c = good;
        c.eye_lat_deg = 91.0;
        assert!(c.validate().is_err(), "latitude out of range");
    }
}
