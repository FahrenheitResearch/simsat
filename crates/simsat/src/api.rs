//! High-level render API — the reusable, PURE-RUST assembly that turns a wrfout / cached
//! brick into the frame DATA (owned Rust arrays) plus a georeference, WITHOUT a GPU, a
//! GUI, or a PNG.
//!
//! This is the single code path behind BOTH the headless examples (`render_frame` /
//! `render_ir`) AND the Python binding (`crates/simsat_py`, exposed as `import simsat`).
//! It is where the render assembly and its correctness tests live (gated normally on the
//! build nodes — pure Rust, no Python). The thin PyO3 wrapper only marshals what
//! [`render`] returns into numpy.
//!
//! [`render`] takes a [`RenderParams`] + a [`Product`] and returns a [`RenderResult`]
//! carrying:
//! - the product data ([`FrameData`]): `VisibleRgb` = the finished true-color RGB (H x W x
//!   3 u8) — byte-identical to what the studio/`render_frame` PNG shows, because it calls
//!   the same shipped [`crate::clouds::render_cloud_frame_rgba`] /
//!   [`crate::topdown::render_topdown_frame_rgba`]; `VisibleBands` = the RAW per-channel
//!   reflectance (H x W x 3 f32, pre-tonemap, in `[0, 1]`); `Ir` = the RAW brightness
//!   temperature in KELVIN (H x W f32) plus an optional colored RGB enhancement.
//! - a [`Georef`]: the projection params + an `extent` (for `imshow`) AND the H x W lat/lon
//!   mesh (for `pcolormesh`), so the array can be placed on a cartopy map.
//!
//! Default view is [`ViewMode::TopDownMap`] (map-registered, north-up, the natural fit for
//! top-down Lambert plotting suites); [`ViewMode::Geostationary`] is the from-space
//! product. The refined true-color appearance is UNCHANGED — this module only wraps the
//! existing render path; no appearance constant is touched.

use std::path::PathBuf;

use crate::asset_pack;
use crate::atmosphere::{
    self, AERIAL_FROXEL_DIM, AtmosphereLuts, AtmosphereParams, CameraGeometry, OutputTransform,
    SkyShTable, sun_enu_to_ecef,
};
use crate::bluemarble;
use crate::bricks::{self, RunManifest, VolumeBrick};
use crate::camera::{
    GeoCamera, MAX_AXIS, PerspectiveCamera, ResolutionMode, SatellitePreset, SurfaceRaster,
    ViewMode, build_map_raster, build_perspective_raster, build_surface_raster_mode,
    extended_native_counts, map_pixel_edge_index_bounds,
};
use crate::clouds::{
    self, CloudFrameStats, CloudScene, DecodedVolume, MarchConfig, OccupancyMip, StepQuality,
    render_cloud_frame_reflectance, render_cloud_frame_rgba, scan_rect_of,
};
use crate::derived::{self, DerivedField};
use crate::frame::{GridGeoref, WrfProjectionParams};
use crate::geocolor;
use crate::gpu;
use crate::horizon::HorizonMap;
use crate::ingest::{self, IngestConfig};
use crate::ingest_grib;
use crate::ir::{self, IrConfig, IrScene, IrVolume};
use crate::ir_enhance::{IrEnhancement, render_ir_rgba};
use crate::render::{
    DEFAULT_EXPOSURE, FLAT_ALBEDO_SRGB, FrameContext, SurfacePixel, WATER_ALBEDO_SCALE, blend_snow,
    normals_from_hgt, reflectance_from_radiance, shade_surface, snow_fraction,
    surface_toa_radiance,
};
use crate::sandwich;
use crate::solar::SolarFrame;
use crate::topdown::{render_topdown_frame_reflectance, render_topdown_frame_rgba};
use crate::web_layer;
use crate::wv::WvBand;

/// The sun optical-depth map resolution (matches the studio / `render_frame` constant).
const SUN_OD_RESOLUTION: usize = 512;
/// Blue Marble crop resample cap (max output axis in texels).
const BLUEMARBLE_MAX_AXIS: u32 = 4096;

/// Which product to render.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Product {
    /// The finished true-color RGB (H x W x 3 u8): the tonemapped display frame.
    VisibleRgb,
    /// The RAW per-channel reflectance (H x W x 3 f32, pre-tonemap, `[0, 1]`).
    VisibleBands,
    /// The RAW 10.3 um (ABI band 13) brightness temperature in KELVIN (H x W f32) +
    /// optional colored RGB.
    Ir,
    /// A water-vapor band (ABI 8/9/10 = 6.2/6.9/7.3 um): the RAW brightness temperature
    /// in KELVIN (H x W f32) + optional colored RGB, returned as [`FrameData::Ir`] (WV is
    /// thermal IR — the same data shape as band 13, just a different weighting function).
    WaterVapor { band: WvBand },
    /// The GeoColor day/night blend (GOES's flagship product): true-color visible by day,
    /// colored band-13 IR by night, crossfaded across the terminator by the PER-PIXEL solar
    /// elevation. Renders the visible frame ([`Product::VisibleRgb`]) AND the IR frame (band
    /// 13 through [`crate::geocolor::GEOCOLOR_NIGHT_ENHANCEMENT`]) through the SAME render
    /// paths and blends them per pixel (see [`crate::geocolor`] for the thresholds). Returned
    /// as [`FrameData::Visible`] — a baked RGB composite, like the visible product. Cost is
    /// approximately the visible march plus the IR march.
    GeoColor,
    /// The Sandwich composite (the classic severe-convection view): the visible true-color RGB
    /// as the base everywhere, with a COLOR-enhanced band-13 IR overlaid ONLY on the COLD (high)
    /// cloud tops, at an alpha that ramps with coldness (see [`crate::sandwich`] for the BT
    /// thresholds + max alpha + the overlay enhancement). Renders the visible frame
    /// ([`Product::VisibleRgb`]) AND the band-13 IR frame (raw BT + colored through
    /// [`crate::sandwich::SANDWICH_ENHANCEMENT`]) through the SAME render paths and composites
    /// them per pixel by the BT. Returned as [`FrameData::Visible`] — a baked RGB composite,
    /// like the visible product. A DAYTIME-convection product (the visible base needs daylight).
    /// Cost is approximately the visible march plus the IR march.
    Sandwich,
    /// A DERIVED scalar-field map product (precipitable water mm, cloud-top temperature K, or
    /// cloud optical depth): a per-column vertical integral / march through the brick,
    /// resampled onto the output raster. The RAW physical `f32` field is the primary
    /// deliverable (for the meteorologist's own colormaps); an optional basic studio colormap
    /// RGB is produced when [`RenderParams::derived_colormap`] is set. Returned as
    /// [`FrameData::Scalar`]. Purely a column computation — it ignores the sun / exposure /
    /// atmosphere / cloud / Blue Marble controls (like the thermal IR product). See
    /// [`crate::derived`] for the definitions + units + assumptions.
    Derived { field: DerivedField },
    /// The web-map CLOUD LAYER pair (the broadcast-visuals / Mapbox-class integration):
    /// the CLOUD FIELD ONLY as premultiplied-alpha RGBA (color = the tonemapped cloud
    /// radiance through the shipped display seam; alpha = `1 - T_cloud`) plus the ground
    /// cloud-shadow MULTIPLY layer (1.0 = no shadow), BOTH delivered on a north-up
    /// Web-Mercator-aligned grid with the four corner lon/lats a Mapbox `ImageSource`
    /// needs (on [`Georef::mercator_corners_lonlat`]; extent in EPSG:3857 metres,
    /// [`ExtentKind::WebMercatorMeters`]). NO Blue Marble / surface / ground atmosphere —
    /// the HOST map is the ground. TOP-DOWN by definition ([`RenderParams::view`] is
    /// ignored); the sun / exposure / steps / multiscatter / beer-powder / margin / granulation
    /// controls drive the cloud march like the visible product. Returned as
    /// [`FrameData::CloudLayer`]. See [`crate::web_layer`] for the alpha model + datum
    /// notes and [`crate::topdown::render_cloud_layer_frame`] for the render.
    CloudLayer,
    /// A FREE-PERSPECTIVE frame (the broadcast "angled 3D scene" product): the camera
    /// on [`RenderParams::perspective`] (eye lat/lon/alt + look-at + horizontal FOV +
    /// dims — REQUIRED for this product) feeds per-pixel pinhole rays through the SAME
    /// surface + cloud marches. `cloud_layer_only = false` renders the FULL COMPOSITE
    /// over our Blue Marble ground (the hero shot; sky rays composite the existing
    /// limb/space handling); `true` renders the CLOUD FIELD ONLY as premultiplied-alpha
    /// RGBA (alpha = `1 - T_cloud`) for compositing over a HOST 3-D map with a matching
    /// camera (no ground-shadow plane — see
    /// [`crate::topdown::render_perspective_cloud_layer`]). Returned as
    /// [`FrameData::Visible`] (`rgba` carries the real alpha in both modes; for the
    /// layer-only mode it is the premultiplied cloud image). The camera pose is
    /// recorded on [`Georef::camera_pose`] + logged (the what-if labeling discipline);
    /// [`RenderParams::view`] / `resolution` / `margin_frac` are ignored (the camera IS
    /// the view). A FLYOVER is the caller rendering N frames along its own eye/look
    /// path — no path scripting here.
    Perspective { cloud_layer_only: bool },
}

/// The ground-texture source for a visible render.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlueMarble {
    /// The seasonal day-of-year blend (the default): the two bracketing monthly 2 km
    /// composites lerped for the timestep's date, `download` fetching missing months
    /// (SHA-256-gated; else the vendored 8 km fallback), `month_override` forcing a month.
    Seasonal {
        month_override: Option<u32>,
        download: bool,
    },
    /// A single-file Blue Marble override (one JPEG for the whole frame).
    SingleFile(PathBuf),
    /// No ground texture — flat albedo (fast; no I/O; used by tests + a plain option).
    FlatAlbedo,
}

impl Default for BlueMarble {
    fn default() -> Self {
        Self::Seasonal {
            month_override: None,
            download: true,
        }
    }
}

/// A synthetic sun position override (a "what-if" light). An unset component keeps the
/// timestep's real value; setting either places the sun over the domain centre at the
/// chosen elevation / azimuth regardless of the file time.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SunOverride {
    pub elev_deg: Option<f64>,
    pub az_deg: Option<f64>,
}

/// The inputs to a render. Use [`RenderParams::new`] for the shipped defaults and set only
/// what you need.
#[derive(Debug, Clone)]
pub struct RenderParams {
    /// A wrfout file (ingested to a brick if not cached under `cache`) OR a cached
    /// `run.json`.
    pub input: PathBuf,
    /// The satellite preset (geostationary view only; ignored for top-down).
    pub satellite: SatellitePreset,
    /// Time index into the wrfout / manifest.
    pub timestep: usize,
    /// The output view: top-down map (default) or from-space geostationary.
    pub view: ViewMode,
    /// The output resolution mode (geostationary only; top-down is native one-px-per-cell).
    pub resolution: ResolutionMode,
    /// Zoom-out / domain MARGIN as a FRACTION of the domain size added on EACH side
    /// (`0.0` = the domain edge-to-edge, the pre-margin behavior; `0.20` = +20% of the
    /// domain span on every side, so the domain occupies the center `1/1.4` of the frame).
    /// The margin shows the real Blue Marble ground + clear sky AROUND the domain — WRF has
    /// no data outside the domain, so no clouds/weather render there (honest context/
    /// orientation). Applies to BOTH views. Kept as a fraction so a future km-based UI is a
    /// trivial swap. See [`crate::camera::build_surface_raster_mode`] /
    /// [`crate::camera::build_map_raster`].
    pub margin_frac: f32,
    /// Aerosol optical depth for the visible atmosphere. The shipped default is
    /// [`crate::atmosphere::DEFAULT_AOD`]. This controls the Mie aerosol column only;
    /// molecular Rayleigh scattering remains present at zero AOD.
    pub aerosol_optical_depth: f32,
    /// Apply the documented 1.5x relative-humidity swelling multiplier to the aerosol
    /// extinction. Off by default so a requested AOD is used literally.
    pub rh_aerosol_swelling: bool,
    /// Apply the product-facing daytime aerial-veil correction. Disable this to retain
    /// the full modeled path airlight; display/low-sun transforms remain independently
    /// controlled by their existing product paths.
    pub atmosphere_correction: bool,
    /// Shorten surface atmosphere columns to the WRF terrain elevation. Disable this to
    /// reproduce the legacy mean-sea-level atmosphere geometry.
    pub terrain_atmosphere: bool,
    /// Display exposure gain (visible only). [`DEFAULT_EXPOSURE`] is the shipped default.
    pub exposure: f64,
    /// M5 Wrenninge multi-scatter octaves (visible only).
    pub multiscatter: bool,
    /// Schneider beer-powder shaping of the direct cloud-sun term (visible only).
    /// Off by default; this is an appearance/QA switch and does not change cloud
    /// transmittance or the quantitative derived cloud-optical-depth product.
    pub beer_powder: bool,
    /// Cloud march step quality (visible only).
    pub steps: StepQuality,
    /// Composite volumetric clouds (visible only; `false` = surface-only for RGB/bands,
    /// and a transparent/neutral [`Product::CloudLayer`]).
    pub clouds: bool,
    /// Use the model cloud-fraction field for fractional/subcolumn cloud rendering when
    /// the source carries one (visible-family products only). `true` is the physical
    /// default; a source without cloud fraction falls back to full-cell coverage. Set
    /// `false` for the legacy behavior that treats every non-zero cloudy grid cell as
    /// horizontally full. Thermal and quantitative derived products are unchanged.
    pub fractional_clouds: bool,
    /// Multiplicative cloud optical-depth calibration (visible only). The shipped default
    /// is [`crate::clouds::DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE`]; `1.0` consumes the
    /// model-derived extinction unchanged, while `0.0` makes cloud extinction transparent.
    /// Values up to `4.0` support bounded sensitivity tests. Applied consistently to the
    /// view, secondary-sun, ambient, and ground-shadow optical depths. The quantitative
    /// derived cloud-optical-depth product remains the unscaled physical input.
    pub cloud_optical_depth_scale: f32,
    /// Sub-grid cloud GRANULATION (edge-erosion detail noise — see the granulation
    /// section of [`crate::clouds`]). `None` (the default) and `Some(false)` keep it
    /// OFF; `Some(true)` enables it for DISPLAY products ([`Product::VisibleRgb`], and
    /// through it the GeoColor day half and the Sandwich visible base). Quantitative
    /// raw-reflectance [`Product::VisibleBands`] stays OFF. The raw-Kelvin thermal products
    /// ([`Product::Ir`] / [`Product::WaterVapor`]) and [`Product::Derived`] never
    /// granulate — they read the un-eroded brick by construction, so quantitative BT
    /// verification always reflects model skill, not display texturing. `Some(bool)`
    /// forces the visible-family behavior either way. Recorded on
    /// [`RenderResult::granulation`] (the what-if-label pattern).
    pub granulation: Option<bool>,
    /// Optional synthetic sun override (visible only).
    pub sun_override: Option<SunOverride>,
    /// Brick cache root (read/write) + seasonal Blue Marble cache.
    pub cache: PathBuf,
    /// The ground-texture source (visible only).
    pub bluemarble: BlueMarble,
    /// For the IR product: also produce a colored RGB via this enhancement (the raw Kelvin
    /// BT plane is always returned). `None` = BT only.
    pub ir_enhancement: Option<IrEnhancement>,
    /// For the [`Product::Derived`] products: also produce the basic studio colormap RGB (the
    /// raw physical `f32` field is always returned). `false` (the default) = raw field only —
    /// the binding's primary deliverable; the studio / CLI set it `true` for the coloured map.
    pub derived_colormap: bool,
    /// An explicit raster to render at instead of the one built from `view`/`resolution`
    /// (the `render_frame` supersample QA path; geostationary visible only). `None` =
    /// build it. NEVER set this from the Python binding.
    pub raster_override: Option<SurfaceRaster>,
    /// Optional display-only GROUND LIFT override ([`crate::render::GROUND_DAY_LIFT`]).
    /// `None` (default) = the baked constant; `1.0` is neutral. Visible RGB-family
    /// products consume it; raw visible bands, thermal, and derived products ignore it.
    pub ground_gain: Option<f64>,
    /// Optional display-only highlight soft-clip knee
    /// ([`crate::render::CLOUD_SOFTCLIP_KNEE`]; `1.0` disables the shoulder). `None`
    /// (default) = the baked constant. Visible RGB-family products consume it.
    pub cloud_softclip: Option<f64>,
    /// Optional physical reflectance-factor ceiling mapped to display white by the
    /// bounded highlight shoulder ([`crate::render::RHO_HIGHLIGHT_MAX`]). `None` keeps
    /// the baked calibration. Display-only; raw visible bands, IR, and derived fields
    /// are unaffected.
    pub cloud_highlight_max: Option<f64>,
    /// Optional `render_frame` override for the TOP-DOWN CLOUD NORMALIZATION
    /// ([`crate::topdown::TOPDOWN_CLOUD_NORM`]; `1.0` = no normalization). `None` (default)
    /// = the baked constant. The `render_frame` `topdown-cloudnorm=` knob sets it.
    pub topdown_cloud_norm: Option<f64>,
    /// The free-perspective camera (eye / look-at / fov / dims) — REQUIRED by (and only
    /// read by) [`Product::Perspective`]. `None` (the default) for every other product.
    pub perspective: Option<PerspectiveCamera>,
}

impl RenderParams {
    /// Shipped defaults: GOES-East, timestep 0, TOP-DOWN map (the integration default),
    /// native resolution, neutral ABI display exposure, multi-scatter + offline steps +
    /// clouds + model cloud fraction on, beer-powder + granulation off, the seasonal Blue
    /// Marble (download on), no sun override, no IR enhancement, the studio cache dir.
    pub fn new(input: PathBuf) -> Self {
        Self {
            input,
            satellite: SatellitePreset::GoesEast,
            timestep: 0,
            view: ViewMode::TopDownMap,
            resolution: ResolutionMode::Native,
            margin_frac: 0.0,
            aerosol_optical_depth: atmosphere::DEFAULT_AOD as f32,
            rh_aerosol_swelling: false,
            atmosphere_correction: true,
            terrain_atmosphere: true,
            exposure: DEFAULT_EXPOSURE,
            multiscatter: true,
            beer_powder: false,
            steps: StepQuality::Offline,
            clouds: true,
            fractional_clouds: true,
            cloud_optical_depth_scale: clouds::DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE,
            granulation: None,
            sun_override: None,
            cache: ingest::default_cache_dir(),
            bluemarble: BlueMarble::default(),
            ir_enhancement: None,
            derived_colormap: false,
            raster_override: None,
            ground_gain: None,
            cloud_softclip: None,
            cloud_highlight_max: None,
            topdown_cloud_norm: None,
            perspective: None,
        }
    }
}

/// What kind of coordinate the [`Georef::extent`] `(x0, x1, y0, y1)` is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtentKind {
    /// Projection-plane METRES (Lambert/PS/Mercator) — the top-down map's `imshow` extent
    /// in the WRF Lambert CRS (place with `transform=<the WRF CRS>`). For a `MAP_PROJ = 6`
    /// lat/lon grid the "metres" are actually degrees (the plane unit), documented.
    ProjectionMeters,
    /// Longitude/latitude DEGREES bounding box — the from-space geostationary frame's
    /// extent (place with `PlateCarree`, or prefer the lat/lon mesh via `pcolormesh`).
    LonLatDegrees,
    /// EPSG:3857 Web Mercator METRES — the cloud-layer delivery grid's extent (its
    /// rows/columns are exact constant-y/constant-x lines of EPSG:3857; a web map
    /// places the image by [`Georef::mercator_corners_lonlat`] instead). See
    /// [`crate::web_layer`] for the datum note (WRF-sphere lat/lon through the
    /// standard 3857 formulas).
    WebMercatorMeters,
}

/// The georeference returned with every frame: the projection params, an `extent` for
/// `imshow`, and the H x W (row-major, row 0 = north) lat/lon mesh for `pcolormesh`.
#[derive(Debug, Clone)]
pub struct Georef {
    pub view: ViewMode,
    /// The WRF projection parameters (build a cartopy/pyproj CRS from these).
    pub projection: WrfProjectionParams,
    /// Raster width (columns; west -> east).
    pub nx: usize,
    /// Raster height (rows; row 0 = north).
    pub ny: usize,
    /// Per-pixel geodetic latitude (deg), `nx*ny` row-major; `NaN` for space/padding.
    pub lat: Vec<f32>,
    /// Per-pixel geodetic longitude (deg), `nx*ny` row-major; `NaN` for space/padding.
    pub lon: Vec<f32>,
    /// `imshow` extent `(x0, x1, y0, y1)` = `(left, right, bottom, top)`; row 0 (north)
    /// aligns to `top` (use `origin='upper'`). Units per [`ExtentKind`].
    pub extent: [f64; 4],
    pub extent_kind: ExtentKind,
    /// The four image corner `[lon, lat]` pairs in the Mapbox GL `ImageSource`
    /// `coordinates` order (top-left/NW, top-right/NE, bottom-right/SE,
    /// bottom-left/SW). `Some` only for the Web-Mercator-delivered
    /// [`Product::CloudLayer`]; `None` for every other product.
    pub mercator_corners_lonlat: Option<[[f64; 2]; 4]>,
    /// The free-perspective camera pose the frame was rendered with — `Some` only for
    /// [`Product::Perspective`] (the what-if labeling discipline: a perspective frame
    /// always carries its camera). `None` for every other product.
    pub camera_pose: Option<PerspectiveCamera>,
}

impl Georef {
    /// A short proj kind string for the projection (`lcc`/`stere`/`merc`/`latlon`), the
    /// PROJ name a cartopy/pyproj CRS is built from.
    pub fn proj_kind(&self) -> &'static str {
        match self.projection.map_proj {
            1 => "lcc",
            2 => "stere",
            3 => "merc",
            6 => "latlon",
            _ => "unknown",
        }
    }

    /// A PROJ.4 string for the [`Self::extent`] coordinate system, so a caller can build a
    /// cartopy / pyproj CRS that is EXACTLY consistent with the extent (place the image
    /// with `ax.imshow(rgb, extent=geo.extent, transform=crs, origin='upper')`).
    ///
    /// For the TOP-DOWN map (extent in projection metres) this is the WRF projection on the
    /// spherical earth `R = 6.37e6`, with `x_0 = y_0 = 0` and — crucially for Lambert /
    /// polar-stereographic — `lat_0 = ±90` (the projection pole). That choice makes PROJ's
    /// `(x, y)` reproduce SimSat's internal plane coordinate ([`GridGeoref::plane_uv`],
    /// which the extent is computed from) EXACTLY, so the extent and this CRS agree (cartopy
    /// then warps the image onto the viewer's map). For the GEOSTATIONARY frame (extent in
    /// lon/lat degrees) it is a plain `longlat` (PlateCarree) — but the geostationary raster
    /// is not rectilinear in lon/lat, so prefer `pcolormesh(geo.lon, geo.lat, data)` there.
    pub fn proj4(&self) -> String {
        let r = crate::optics::EARTH_RADIUS_M;
        if self.extent_kind == ExtentKind::WebMercatorMeters {
            // The standard EPSG:3857 definition (the extent is on the 6378137 sphere).
            return "+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 \
                    +k=1 +units=m +nadgrids=@null +no_defs"
                .to_string();
        }
        if self.extent_kind == ExtentKind::LonLatDegrees {
            return format!("+proj=longlat +R={r} +no_defs");
        }
        let p = &self.projection;
        let pole = if p.cen_lat_deg >= 0.0 { 90.0 } else { -90.0 };
        match p.map_proj {
            1 => format!(
                "+proj=lcc +lat_1={} +lat_2={} +lat_0={} +lon_0={} +x_0=0 +y_0=0 +R={} +units=m +no_defs",
                p.truelat1_deg, p.truelat2_deg, pole, p.stand_lon_deg, r
            ),
            2 => format!(
                "+proj=stere +lat_0={} +lat_ts={} +lon_0={} +x_0=0 +y_0=0 +R={} +units=m +no_defs",
                pole, p.truelat1_deg, p.stand_lon_deg, r
            ),
            3 => format!(
                "+proj=merc +lon_0={} +lat_ts={} +x_0=0 +y_0=0 +R={} +units=m +no_defs",
                p.stand_lon_deg, p.truelat1_deg, r
            ),
            _ => format!("+proj=longlat +R={r} +no_defs"),
        }
    }
}

/// The rendered product data (exactly one variant, per the requested [`Product`]).
#[derive(Debug, Clone)]
pub enum FrameData {
    /// The finished true-color display: `rgb` = `nx*ny*3` u8 (space = black), plus the raw
    /// `rgba` = `nx*ny*4` u8 (space alpha 0) the store writer / supersample downsample use.
    Visible { rgb: Vec<u8>, rgba: Vec<u8> },
    /// The RAW per-channel reflectance factor: `nx*ny*3` f32 in `[0, 1]` (space = 0).
    Bands { reflectance: Vec<f32> },
    /// The RAW brightness temperature (`nx*ny` f32, KELVIN; `NaN` off-domain) plus an
    /// optional colored `rgb` = `nx*ny*3` u8 (present iff an enhancement was requested).
    Ir {
        bt_kelvin: Vec<f32>,
        rgb: Option<Vec<u8>>,
    },
    /// A DERIVED scalar field: the RAW physical values (`nx*ny` f32; `NaN` = no-data /
    /// off-domain, or a clear column for cloud-top temperature) plus an optional basic studio
    /// colormap `rgb` = `nx*ny*3` u8 (present iff [`RenderParams::derived_colormap`] was set).
    /// `field` carries which field these values are (for the units / label / colormap).
    Scalar {
        values: Vec<f32>,
        rgb: Option<Vec<u8>>,
        field: DerivedField,
    },
    /// The web-map CLOUD LAYER pair on the Web-Mercator delivery grid (`nx*ny` from the
    /// [`RenderResult`]; row 0 = north): `rgba_premul` = the PREMULTIPLIED-alpha cloud
    /// image (`nx*ny*4`; see [`crate::topdown::CloudLayerFrame`] for the exact
    /// semantics; convert with [`crate::web_layer::unpremultiply_rgba`] for straight-
    /// alpha delivery) and `shadow` = the ground cloud-shadow MULTIPLY field
    /// (`nx*ny` f32 in `[0,1]`, 1.0 = no shadow; out-of-coverage = 1.0). The placement
    /// corners ride on [`Georef::mercator_corners_lonlat`].
    CloudLayer {
        rgba_premul: Vec<u8>,
        shadow: Vec<f32>,
    },
}

/// The valid time of the rendered frame (for store naming / reporting).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameTime {
    pub year: i32,
    pub month: u32,
    pub day: u32,
    /// UTC hour as a fraction (e.g. 2.25 = 02:15).
    pub ut: f64,
}

/// Where the visible frame's ground pixels actually came from (integration-API
/// honesty: a quality downgrade is invisible in the output array alone, so it is
/// reported instead of silently swallowed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroundSource {
    /// The full-quality seasonal 2 km monthly composite(s).
    TwoKm,
    /// At least one contributing month used the vendored 8 km emergency fallback
    /// (offline / download failed) — a visibly coarser ground.
    EightKmFallback,
    /// No ground texture — the flat-albedo constant. The string says WHY (requested,
    /// or the seasonal load failure that degraded to it).
    FlatAlbedo(String),
    /// A user-named single-file ground texture ([`BlueMarble::SingleFile`]).
    SingleFile,
}

impl GroundSource {
    /// A short stable slug for bindings/logs: `2km` / `8km-fallback` / `flat-albedo`
    /// / `single-file`.
    pub fn slug(&self) -> &'static str {
        match self {
            Self::TwoKm => "2km",
            Self::EightKmFallback => "8km-fallback",
            Self::FlatAlbedo(_) => "flat-albedo",
            Self::SingleFile => "single-file",
        }
    }
}

/// The result of a render: the product data + georef + the raster used + metadata.
#[derive(Debug, Clone)]
pub struct RenderResult {
    pub data: FrameData,
    pub georef: Georef,
    /// The raster the frame was rendered at (for a store write / stats in the CLI).
    pub raster: SurfaceRaster,
    pub nx: usize,
    pub ny: usize,
    /// The sun elevation (deg) at the domain centre (visible; `0` for IR — thermal).
    pub sun_elev_deg: f64,
    /// Whether a Native-resolution geostationary raster was clamped against the axis cap.
    pub res_clamped: bool,
    /// Cloud reflectance coverage stats (geostationary visible only; `None` otherwise).
    pub cloud_stats: Option<CloudFrameStats>,
    pub time: FrameTime,
    /// TRUE when the source carried no parseable valid time and [`Self::time`] is the
    /// FABRICATED fallback date (2004-06-21 12:00 UT — unchanged, so existing QA
    /// frames are byte-identical). The sun position and the Blue Marble season of a
    /// visible frame rendered under this flag are NOT the run's real conditions;
    /// consumers can now see that instead of silently trusting the frame.
    pub time_is_fallback: bool,
    /// Where the ground pixels came from (visible-family products; `None` for the
    /// thermal / derived products, which use no ground texture).
    pub ground_source: Option<GroundSource>,
    /// The collected ground-resolution status lines (seasonal Blue Marble download /
    /// fallback progress — previously discarded by a no-op callback). Empty when
    /// nothing noteworthy happened.
    pub ground_status: Vec<String>,
    /// TRUE when sub-grid cloud GRANULATION (edge-erosion detail noise) was applied to
    /// this frame's cloud field (see [`RenderParams::granulation`] for the product
    /// scoping). Recorded like the fake-sun what-if label so a consumer can see the
    /// display texturing was active; always `false` for the thermal / derived products.
    pub granulation: bool,
}

/// Resolved scene inputs (a wrfout ingest-if-needed, or a cached run.json).
struct SceneSource {
    brick: VolumeBrick,
    georef: GridGeoref,
    params: WrfProjectionParams,
    time_iso: Option<String>,
}

/// Render one frame. The single high-level entry point behind the examples and the Python
/// binding. Resolves the source (ingesting the wrfout timestep to a brick if needed),
/// assembles the scene, marches the requested product, and returns the frame data + georef.
pub fn render(params: &RenderParams, product: Product) -> Result<RenderResult, String> {
    let src = resolve_source(params)?;
    match product {
        // The IR window (band 13) and the WV bands (8/9/10) share ONE thermal march; they
        // differ only in the `IrConfig` (wavelength + WV mass-absorption / band selector).
        Product::Ir => render_ir_scene(&src, params, IrConfig::band13()),
        Product::WaterVapor { band } => render_ir_scene(&src, params, band.ir_config()),
        Product::VisibleRgb | Product::VisibleBands => render_visible_scene(&src, params, product),
        // GeoColor renders the visible + IR frames through the SAME paths and blends them.
        Product::GeoColor => render_geocolor_scene(&src, params),
        // Sandwich also renders visible + IR through the SAME paths, then overlays the colored
        // IR on the cold tops of the visible base.
        Product::Sandwich => render_sandwich_scene(&src, params),
        // Derived scalar fields are a per-column brick computation resampled onto the raster
        // (no sun / atmosphere / cloud march).
        Product::Derived { field } => render_derived_scene(&src, params, field),
        // The web-map cloud layer: cloud-only + ground shadow on a Web-Mercator grid.
        Product::CloudLayer => render_cloud_layer_scene(&src, params),
        // The free-perspective frame (full composite, or the cloud-layer-only variant).
        Product::Perspective { cloud_layer_only } => {
            render_perspective_scene(&src, params, cloud_layer_only)
        }
    }
}

// ── visible assembly (mirrors the studio clouds-on pipeline / render_frame) ─────

fn render_visible_scene(
    src: &SceneSource,
    params: &RenderParams,
    product: Product,
) -> Result<RenderResult, String> {
    let SceneSource {
        brick,
        georef,
        params: proj,
        time_iso,
    } = src;
    let (nx, ny) = (brick.nx, brick.ny);
    let is_topdown = params.view == ViewMode::TopDownMap;

    // Output raster: the from-space scan raster, the top-down map raster, or an explicit
    // override (the supersample QA raster). The top-down map is adapted to a SurfaceRaster
    // so the shared LUT + Blue Marble + assemble machinery is identical for both views.
    let camera = GeoCamera::new(params.satellite);
    let margin = params.margin_frac as f64;
    let raster = match &params.raster_override {
        Some(r) => r.clone(),
        None if is_topdown => build_map_raster(georef, nx, ny, nx, ny, margin)
            .ok_or_else(|| "the domain is too small to build a top-down map".to_string())?
            .as_surface_raster(),
        None => {
            build_surface_raster_mode(&camera, georef, nx, ny, params.resolution, margin, MAX_AXIS)
                .ok_or_else(|| {
                    format!(
                        "the domain is not fully visible from {}; try a different satellite",
                        params.satellite.label()
                    )
                })?
        }
    };
    // Native clamped against the axis cap? Compare against the margin-extended target.
    let (target_nx, target_ny) = extended_native_counts(nx, ny, margin);
    let res_clamped = !is_topdown
        && params.raster_override.is_none()
        && params.resolution == ResolutionMode::Native
        && (raster.nx < target_nx || raster.ny < target_ny);

    // Solar geometry + the single ECEF sun vector (sun at infinity).
    let (time, time_is_fallback) = resolve_frame_time(time_iso.as_deref());
    let solar = SolarFrame::new(time.year, time.month, time.day, time.ut);
    let use_sun_override = params
        .sun_override
        .is_some_and(|s| s.elev_deg.is_some() || s.az_deg.is_some());
    let (sun_ecef, center_sun_elev) = resolve_frame_sun(params, &raster, &solar, proj);

    // Ground texture (a SingleFile that fails to load is a HARD error; a seasonal
    // failure degrades to flat albedo but is REPORTED, never silent).
    let (bluemarble, ground_source, ground_status) =
        resolve_bluemarble(params, &raster, time.month, time.day)?;

    // Per-pixel LUTs (geo lookup + light); override the light plane for a synthetic sun.
    let (lut_geo, mut lut_light) = gpu::build_luts(&raster, bluemarble.as_ref(), nx, ny, &solar);
    if use_sun_override {
        override_light_lut(&mut lut_light, &raster, sun_ecef);
    }

    // Terrain normals + march pitch + the M3 horizon map.
    let normals = normals_from_hgt(&brick.hgt, nx, ny, proj.dx_m, proj.dy_m);
    let horiz_pitch_m = horiz_pitch_m(proj);
    let (dx_m_m, dy_m_m) = dx_dy_metres(proj);
    let horizon_map = HorizonMap::build(&brick.hgt, nx, ny, dx_m_m, dy_m_m);

    // M2 atmosphere.
    let pw_ratio = atmosphere::pw_ratio_from_brick(brick);
    let atmo_params = AtmosphereParams {
        aod: params.aerosol_optical_depth as f64,
        pw_ratio,
        aerosol_swelling: if params.rh_aerosol_swelling { 1.5 } else { 1.0 },
        ground_albedo: atmosphere::GROUND_ALBEDO,
    };
    let luts = AtmosphereLuts::build(&atmo_params);
    let sky_sh = SkyShTable::build(&luts, &atmo_params, 48);
    let cam_geo = CameraGeometry::from_sub_lon(params.satellite.sub_lon_deg());

    // M4/M5 cloud volume + scene.
    let fractional_requested = params.clouds && params.fractional_clouds;
    let fractional_clouds = fractional_requested && brick.has_cloud_fraction;
    let mut vol = if fractional_clouds {
        DecodedVolume::from_brick(brick, horiz_pitch_m)
    } else {
        DecodedVolume::from_brick_legacy(brick, horiz_pitch_m)
    };
    apply_fractional_clouds_for_visible(
        &mut vol,
        fractional_clouds,
        fractional_requested,
        "visible",
    );
    let mip = OccupancyMip::build(&vol, clouds::OCCUPANCY_MIP_FACTOR);
    // Sub-grid cloud GRANULATION (edge-erosion detail noise): OPT-IN (default OFF)
    // as of v0.1.1 — the round-1 default-ON look was owner-rejected on coarse-grid
    // decks ("cheese grater": uniform pinhole pepper on deck margins + grey
    // half-thinned transitions); the tune-2 rework (deck-coherence gate, bimodal
    // carve, domain-warped cells) re-earns the default in a later release. Only
    // meaningful for the DISPLAY product (VisibleRgb — and through it the GeoColor
    // day half / Sandwich visible base); the quantitative raw-reflectance bands and
    // thermal products never granulate regardless. The SAME Option feeds the sun-OD
    // accumulation AND rides MarchConfig into the view + sun marches, so every march
    // of this composite samples ONE eroded field.
    let gran_on = params.clouds
        && params.granulation.unwrap_or(false)
        && matches!(product, Product::VisibleRgb);
    let granulation = if gran_on {
        Some(clouds::Granulation::for_grid(horiz_pitch_m))
    } else {
        None
    };
    if let Some(g) = granulation {
        // The what-if-label pattern: the display texturing is logged, never silent.
        crate::log_line!(
            "simsat api: cloud granulation ON (sub-grid edge erosion; dx {horiz_pitch_m:.0} m \
             -> amplitude {:.3})",
            g.amplitude
        );
    }
    let sun_od = clouds::accumulate_sun_od_granulated(
        &vol,
        georef,
        sun_ecef,
        SUN_OD_RESOLUTION,
        clouds::SUN_OD_EDGE_FEATHER_TEXELS,
        granulation,
    );
    let scan_rect = scan_rect_of(&raster.scan);
    let froxel = atmosphere::build_aerial_froxel(
        &luts,
        &atmo_params,
        &cam_geo,
        sun_ecef,
        scan_rect,
        AERIAL_FROXEL_DIM,
    );
    let mut cfg = MarchConfig {
        beer_powder: params.beer_powder,
        cloud_optical_depth_scale: params.cloud_optical_depth_scale,
        octaves: if params.multiscatter {
            clouds::DEFAULT_OCTAVES
        } else {
            1
        },
        granulation,
        ..MarchConfig::new(params.steps, vol.voxel_pitch_m())
    };
    // Appearance-pass wiring: the EDGE FEATHER activates only under a zoom-out margin
    // (byte-identical no-op at margin 0); the ground-lift / soft-clip / top-down-cloud-norm
    // levers keep their baked defaults unless a `render_frame` CLI knob overrides them.
    cfg.edge_feather_cells = clouds::edge_feather_cells_for_margin(margin, nx, ny);
    // Display-only appearance overrides are deliberately ignored by VisibleBands.
    // That product remains the stable pre-tonemap diagnostic even when the same
    // RenderParams is reused for an RGB A/B render.
    if matches!(product, Product::VisibleRgb) {
        if let Some(g) = params.ground_gain {
            cfg.ground_day_lift = g;
        }
        if let Some(k) = params.cloud_softclip {
            cfg.cloud_softclip_knee = k;
        }
        if let Some(m) = params.cloud_highlight_max {
            cfg.cloud_highlight_max = m;
        }
        if let Some(n) = params.topdown_cloud_norm {
            cfg.topdown_cloud_norm = n;
        }
    }
    let scene = CloudScene {
        vol: &vol,
        mip: &mip,
        sun_od: &sun_od,
        georef,
        luts: &luts,
        sky_sh: &sky_sh,
        sun_ecef,
        cfg,
    };
    let surf = FrameContext {
        luts: &luts,
        params: &atmo_params,
        sky_sh: &sky_sh,
        cam: cam_geo,
        sun_ecef,
        output_transform: OutputTransform::AbiReflectance,
        bm_present: bluemarble.is_some(),
        water_scale: WATER_ALBEDO_SCALE as f64,
        flat_albedo_srgb: FLAT_ALBEDO_SRGB as f64,
        raymarch_steps: 16,
        exposure: params.exposure,
        ground_day_lift: cfg.ground_day_lift,
        cloud_softclip_knee: cfg.cloud_softclip_knee,
        cloud_highlight_max: cfg.cloud_highlight_max,
        atmosphere_correction: params.atmosphere_correction,
        terrain_atmosphere: params.terrain_atmosphere,
    };

    // The per-pixel surface assembler (twin of the studio's `assemble`).
    let assemble = make_assemble(
        brick,
        &lut_geo,
        &lut_light,
        bluemarble.as_ref(),
        &normals,
        &horizon_map,
        raster.nx,
        nx,
        ny,
    );

    // Render the requested product. RGB uses the shipped display path (byte-identical to
    // the studio/PNG); Bands uses the pre-tonemap reflectance path. Both views and both
    // visible products honor the clouds flag (`false` = surface only).
    let (rnx, rny) = (raster.nx, raster.ny);
    let data = match product {
        Product::VisibleRgb => {
            let rgba = if is_topdown {
                // Top-down honors the clouds flag (clouds-off = the clean surface basemap):
                // `None` renders surface-only, matching the geostationary clouds-off path.
                let topdown_scene = if params.clouds { Some(&scene) } else { None };
                render_topdown_frame_rgba(
                    &surf,
                    topdown_scene,
                    &raster.lat,
                    &raster.lon,
                    rnx,
                    rny,
                    assemble,
                )
            } else if params.clouds {
                render_cloud_frame_rgba(&scene, &surf, &froxel, &raster.scan, assemble)
            } else {
                render_geo_surface_rgba(&surf, &raster, assemble)
            };
            let rgb = rgba_to_rgb_black_space(&rgba, rnx, rny);
            FrameData::Visible { rgb, rgba }
        }
        Product::VisibleBands => {
            let reflectance = if is_topdown {
                // Top-down honors the clouds flag (clouds-off = surface-only reflectance).
                let topdown_scene = if params.clouds { Some(&scene) } else { None };
                render_topdown_frame_reflectance(
                    &surf,
                    topdown_scene,
                    &raster.lat,
                    &raster.lon,
                    rnx,
                    rny,
                    assemble,
                )
            } else if params.clouds {
                render_cloud_frame_reflectance(&scene, &surf, &froxel, &raster.scan, assemble)
            } else {
                render_geo_surface_reflectance(&surf, &raster, assemble)
            };
            FrameData::Bands { reflectance }
        }
        Product::Ir | Product::WaterVapor { .. } => {
            unreachable!("thermal products are dispatched to render_ir_scene")
        }
        Product::GeoColor | Product::Sandwich => {
            unreachable!("composite products are assembled by their own scene fn, not here")
        }
        Product::Derived { .. } => {
            unreachable!("derived products are assembled by render_derived_scene")
        }
        Product::CloudLayer => {
            unreachable!("the cloud layer is assembled by render_cloud_layer_scene")
        }
        Product::Perspective { .. } => {
            unreachable!("perspective frames are assembled by render_perspective_scene")
        }
    };

    // Cloud reflectance stats (geostationary visible only; strided, cheap).
    let cloud_stats = if is_topdown {
        None
    } else {
        let stride = (rnx.max(rny) / 128).max(1);
        Some(clouds::cloud_frame_stats(
            &scene,
            &cam_geo,
            &raster.lat,
            &raster.lon,
            &raster.grid_i,
            &raster.scan,
            stride,
            0.98,
        ))
    };

    let georef_out = build_georef(params.view, proj, georef, &raster, nx, ny, margin);
    Ok(RenderResult {
        data,
        georef: georef_out,
        raster,
        nx: rnx,
        ny: rny,
        sun_elev_deg: center_sun_elev,
        res_clamped,
        cloud_stats,
        time,
        time_is_fallback,
        ground_source: Some(ground_source),
        ground_status,
        granulation: gran_on,
    })
}

// ── IR / WV assembly (mirrors render_ir; the ONE thermal path) ──────────────────

/// Render a thermal brightness-temperature product (the 10.3 um window band 13 OR a WV
/// band 8/9/10) — the SAME slant-ray Planck-emission march, parameterised by `cfg`
/// (wavelength + WV mass-absorption + band selector). `cfg.band` keys the enhancement.
fn render_ir_scene(
    src: &SceneSource,
    params: &RenderParams,
    cfg: IrConfig,
) -> Result<RenderResult, String> {
    let SceneSource {
        brick,
        georef,
        params: proj,
        time_iso,
    } = src;
    let (nx, ny) = (brick.nx, brick.ny);
    let is_topdown = params.view == ViewMode::TopDownMap;

    let camera = GeoCamera::new(params.satellite);
    let margin = params.margin_frac as f64;
    let raster = if is_topdown {
        build_map_raster(georef, nx, ny, nx, ny, margin)
            .ok_or_else(|| "the domain is too small to build a top-down map".to_string())?
            .as_surface_raster()
    } else {
        build_surface_raster_mode(&camera, georef, nx, ny, params.resolution, margin, MAX_AXIS)
            .ok_or_else(|| {
                format!(
                    "the domain is not fully visible from {}; try a different satellite",
                    params.satellite.label()
                )
            })?
    };
    let (target_nx, target_ny) = extended_native_counts(nx, ny, margin);
    let res_clamped = !is_topdown
        && params.resolution == ResolutionMode::Native
        && (raster.nx < target_nx || raster.ny < target_ny);

    let horiz_pitch_m = horiz_pitch_m(proj);
    let vol = IrVolume::from_brick(brick, horiz_pitch_m);
    let dv = DecodedVolume::from_brick_legacy(brick, horiz_pitch_m);
    let mip = OccupancyMip::build(&dv, clouds::OCCUPANCY_MIP_FACTOR);
    let cam_geo = CameraGeometry::from_sub_lon(params.satellite.sub_lon_deg());
    let band = cfg.band;
    let scene = IrScene {
        vol: &vol,
        mip: &mip,
        georef,
        cfg,
    };

    let bt = if is_topdown {
        crate::topdown::render_topdown_ir_bt_frame(
            &scene,
            &raster.lat,
            &raster.lon,
            &raster.grid_i,
            raster.nx,
            raster.ny,
        )
    } else {
        ir::render_ir_bt_frame(&scene, &cam_geo, &raster.scan, &raster.grid_i)
    };
    let rgb = params
        .ir_enhancement
        .map(|enh| rgba_to_rgb_black_space(&render_ir_rgba(&bt, band, enh), raster.nx, raster.ny));

    let (time, time_is_fallback) = resolve_frame_time(time_iso.as_deref());
    let georef_out = build_georef(params.view, proj, georef, &raster, nx, ny, margin);
    Ok(RenderResult {
        nx: raster.nx,
        ny: raster.ny,
        data: FrameData::Ir { bt_kelvin: bt, rgb },
        georef: georef_out,
        raster,
        sun_elev_deg: 0.0,
        res_clamped,
        cloud_stats: None,
        time,
        time_is_fallback,
        ground_source: None,
        ground_status: Vec::new(),
        // The raw-Kelvin thermal march reads the un-eroded brick by construction.
        granulation: false,
    })
}

// ── GeoColor day/night blend (true-color by day, colored IR by night) ───────────

/// Render the GeoColor day/night blend. Renders the finished true-color visible frame
/// ([`render_visible_scene`] / [`Product::VisibleRgb`]) AND the band-13 colored-IR frame
/// ([`render_ir_scene`] through [`geocolor::GEOCOLOR_NIGHT_ENHANCEMENT`]) through the SAME
/// render paths, then blends them per pixel by the PER-PIXEL solar elevation (see
/// [`crate::geocolor`] for the thresholds). The day side is the refined true-color; the
/// night side is the colored IR (clouds in IR at night — no city lights, no data). Cost is
/// approximately the visible march + the IR march. Returns [`FrameData::Visible`] — a baked
/// RGB composite, stored like the visible product.
fn render_geocolor_scene(src: &SceneSource, params: &RenderParams) -> Result<RenderResult, String> {
    // Clear any explicit raster override so the two sub-renders share a bit-identical raster
    // (the blend is per-pixel; the visible + IR frames MUST be pixel-aligned). GeoColor is
    // never driven with a supersample raster.
    let mut base = params.clone();
    base.raster_override = None;

    // 1. The refined true-color visible frame (the day side).
    let vis = render_visible_scene(src, &base, Product::VisibleRgb)?;
    let vis_rgba = match &vis.data {
        FrameData::Visible { rgba, .. } => rgba,
        other => return Err(format!("geocolor: expected a visible frame, got {other:?}")),
    };

    // 2. The colored band-13 IR frame (the night side), through the GeoColor night
    //    enhancement. Reuses the SAME IR march the Ir product uses.
    let mut ir_params = base.clone();
    ir_params.ir_enhancement = Some(geocolor::GEOCOLOR_NIGHT_ENHANCEMENT);
    let ir = render_ir_scene(src, &ir_params, IrConfig::band13())?;
    let ir_rgb = match &ir.data {
        FrameData::Ir { rgb: Some(rgb), .. } => rgb,
        _ => return Err("geocolor: the IR frame is missing its enhancement RGB".to_string()),
    };
    if (vis.nx, vis.ny) != (ir.nx, ir.ny) {
        return Err(format!(
            "geocolor: visible {}x{} and IR {}x{} rasters disagree",
            vis.nx, vis.ny, ir.nx, ir.ny
        ));
    }

    // 3. The per-pixel solar elevation for the blend — the SAME per-pixel elevation the
    //    visible pass lit each terminator pixel with (`build_luts` writes it into `l[3]`;
    //    it is independent of the Blue Marble, so bm = None here reproduces it exactly). A
    //    sun override rewrites it from the single overridden ECEF sun, matching the visible
    //    pass so an overridden day/night sun blends consistently.
    let t = vis.time;
    let solar = SolarFrame::new(t.year, t.month, t.day, t.ut);
    let (_lut_geo, mut lut_light) = gpu::build_luts(&vis.raster, None, vis.nx, vis.ny, &solar);
    let use_sun_override = base
        .sun_override
        .is_some_and(|s| s.elev_deg.is_some() || s.az_deg.is_some());
    if use_sun_override {
        let (sun_ecef, _) = resolve_frame_sun(&base, &vis.raster, &solar, &src.params);
        override_light_lut(&mut lut_light, &vis.raster, sun_ecef);
    }

    // 4. Blend per pixel: full visible in day, full colored-IR at night, smoothstep across
    //    the terminator. Space pixels (visible alpha 0) stay black.
    let (rgb, rgba) = geocolor::blend_rgba(vis_rgba, ir_rgb, vis.nx * vis.ny, |i| {
        lut_light[i * 4 + 3] as f64
    });

    Ok(RenderResult {
        data: FrameData::Visible { rgb, rgba },
        georef: vis.georef,
        raster: vis.raster,
        nx: vis.nx,
        ny: vis.ny,
        sun_elev_deg: vis.sun_elev_deg,
        res_clamped: vis.res_clamped,
        cloud_stats: vis.cloud_stats,
        time: vis.time,
        time_is_fallback: vis.time_is_fallback,
        ground_source: vis.ground_source,
        ground_status: vis.ground_status,
        // The day half is the visible product; the IR night half never granulates.
        granulation: vis.granulation,
    })
}

// ── Sandwich composite (visible base + colored IR on the cold tops) ─────────────

/// Render the Sandwich composite (the classic severe-convection view). Renders the finished
/// true-color visible frame ([`render_visible_scene`] / [`Product::VisibleRgb`]) as the BASE
/// AND the band-13 IR frame ([`render_ir_scene`] through [`sandwich::SANDWICH_ENHANCEMENT`],
/// which returns the RAW BT plane AND the colored RGB) through the SAME render paths, then
/// overlays the colored IR on the COLD tops by the per-pixel [`sandwich::sandwich_alpha`] of
/// the brightness temperature (see [`crate::sandwich`] for the thresholds). The visible gives
/// the fine cloud texture; the IR color highlights the coldest overshooting tops. A DAYTIME
/// product (the visible base needs daylight). Cost is approximately the visible march + the IR
/// march. Returns [`FrameData::Visible`] — a baked RGB composite, stored like the visible
/// product.
fn render_sandwich_scene(src: &SceneSource, params: &RenderParams) -> Result<RenderResult, String> {
    // Clear any explicit raster override so the two sub-renders share a bit-identical raster
    // (the composite is per-pixel; the visible + IR frames MUST be pixel-aligned).
    let mut base = params.clone();
    base.raster_override = None;

    // 1. The refined true-color visible frame (the base everywhere).
    let vis = render_visible_scene(src, &base, Product::VisibleRgb)?;
    let vis_rgba = match &vis.data {
        FrameData::Visible { rgba, .. } => rgba,
        other => return Err(format!("sandwich: expected a visible frame, got {other:?}")),
    };

    // 2. The band-13 IR frame through the sandwich enhancement — both the RAW BT plane (drives
    //    the coldness alpha) and the colored RGB (the overlay). Reuses the SAME IR march the
    //    Ir product uses.
    let mut ir_params = base.clone();
    ir_params.ir_enhancement = Some(sandwich::SANDWICH_ENHANCEMENT);
    let ir = render_ir_scene(src, &ir_params, IrConfig::band13())?;
    let (bt, ir_rgb) = match &ir.data {
        FrameData::Ir {
            bt_kelvin,
            rgb: Some(rgb),
        } => (bt_kelvin, rgb),
        _ => return Err("sandwich: the IR frame is missing its enhancement RGB".to_string()),
    };
    if (vis.nx, vis.ny) != (ir.nx, ir.ny) {
        return Err(format!(
            "sandwich: visible {}x{} and IR {}x{} rasters disagree",
            vis.nx, vis.ny, ir.nx, ir.ny
        ));
    }

    // 3. Composite: visible base, colored IR overlaid on the cold tops (alpha ramps with the
    //    BT coldness). Space pixels (visible alpha 0) stay black.
    let (rgb, rgba) = sandwich::blend_rgba(vis_rgba, ir_rgb, bt, vis.nx * vis.ny);

    Ok(RenderResult {
        data: FrameData::Visible { rgb, rgba },
        georef: vis.georef,
        raster: vis.raster,
        nx: vis.nx,
        ny: vis.ny,
        sun_elev_deg: vis.sun_elev_deg,
        res_clamped: vis.res_clamped,
        cloud_stats: vis.cloud_stats,
        time: vis.time,
        time_is_fallback: vis.time_is_fallback,
        ground_source: vis.ground_source,
        ground_status: vis.ground_status,
        // The visible base carries the granulation; the IR overlay never granulates.
        granulation: vis.granulation,
    })
}

// ── derived scalar-field maps (per-column brick integrals, map-registered) ──────

/// Render a DERIVED scalar-field map product (precipitable water, cloud-top temperature, or
/// cloud optical depth). Computes the field as a per-column native 2-D field directly from the
/// brick ([`derived::compute_field`]) — a vertical integral / march, with NO sun / atmosphere /
/// cloud state — then resamples it onto the output raster (top-down map or from-space
/// geostationary) via the raster's fractional WRF indices ([`derived::resample_field`], `NaN`
/// off-domain), and optionally colours it with the basic studio colormap
/// ([`derived::colorize`]). Returns [`FrameData::Scalar`]; the RAW physical field is the
/// primary deliverable (`sun_elev_deg = 0`, thermal-like — no light input).
fn render_derived_scene(
    src: &SceneSource,
    params: &RenderParams,
    field: DerivedField,
) -> Result<RenderResult, String> {
    let SceneSource {
        brick,
        georef,
        params: proj,
        time_iso,
    } = src;
    let (nx, ny) = (brick.nx, brick.ny);
    let is_topdown = params.view == ViewMode::TopDownMap;

    let camera = GeoCamera::new(params.satellite);
    let margin = params.margin_frac as f64;
    let raster = if is_topdown {
        build_map_raster(georef, nx, ny, nx, ny, margin)
            .ok_or_else(|| "the domain is too small to build a top-down map".to_string())?
            .as_surface_raster()
    } else {
        build_surface_raster_mode(&camera, georef, nx, ny, params.resolution, margin, MAX_AXIS)
            .ok_or_else(|| {
                format!(
                    "the domain is not fully visible from {}; try a different satellite",
                    params.satellite.label()
                )
            })?
    };
    let (target_nx, target_ny) = extended_native_counts(nx, ny, margin);
    let res_clamped = !is_topdown
        && params.resolution == ResolutionMode::Native
        && (raster.nx < target_nx || raster.ny < target_ny);

    // The native per-column field, resampled onto the output raster (NaN off-domain).
    let native = derived::compute_field(brick, field);
    let values = derived::resample_field(
        &native,
        nx,
        ny,
        &raster.grid_i,
        &raster.grid_j,
        raster.nx,
        raster.ny,
    );
    let rgb = if params.derived_colormap {
        Some(derived::colorize(&values, field))
    } else {
        None
    };

    let (time, time_is_fallback) = resolve_frame_time(time_iso.as_deref());
    let georef_out = build_georef(params.view, proj, georef, &raster, nx, ny, margin);
    Ok(RenderResult {
        nx: raster.nx,
        ny: raster.ny,
        data: FrameData::Scalar { values, rgb, field },
        georef: georef_out,
        raster,
        sun_elev_deg: 0.0,
        res_clamped,
        cloud_stats: None,
        time,
        time_is_fallback,
        ground_source: None,
        ground_status: Vec::new(),
        // Derived fields are per-column brick integrals — never granulated.
        granulation: false,
    })
}

// ── web-map cloud layer (cloud-only + ground shadow on a Web-Mercator grid) ─────

/// Render the [`Product::CloudLayer`] pair. Assembles the SAME cloud scene the visible
/// product marches (atmosphere LUTs + SH sky ambient + decoded volume + occupancy mip +
/// granulated sun-OD map), renders the CLOUD-ONLY layer + the ground cloud-shadow field
/// on the domain's native Lambert map raster
/// ([`crate::topdown::render_cloud_layer_frame`] — no Blue Marble, no surface: the host
/// map is the ground), then resamples both onto a north-up Web-Mercator delivery grid
/// ([`crate::web_layer::reproject_cloud_layer`]) sized at ~native pitch over the
/// raster's (margin-extended) geodetic bbox. TOP-DOWN by definition —
/// [`RenderParams::view`] is ignored (documented on the Product). The returned
/// [`Georef`] describes the MERCATOR grid: extent in EPSG:3857 metres, the per-pixel
/// lat/lon mesh of that grid, and the Mapbox `ImageSource` corner lon/lats on
/// [`Georef::mercator_corners_lonlat`]. NOTE [`RenderResult::raster`] is the NATIVE
/// Lambert map raster the marches ran on (its dims differ from the delivered
/// `nx`/`ny`, which are the Mercator grid's).
fn render_cloud_layer_scene(
    src: &SceneSource,
    params: &RenderParams,
) -> Result<RenderResult, String> {
    let SceneSource {
        brick,
        georef,
        params: proj,
        time_iso,
    } = src;
    let (nx, ny) = (brick.nx, brick.ny);
    let margin = params.margin_frac as f64;
    let raster = build_map_raster(georef, nx, ny, nx, ny, margin)
        .ok_or_else(|| "the domain is too small to build a top-down map".to_string())?
        .as_surface_raster();

    // Solar geometry + the single ECEF sun (the layer's clouds are sun-lit like the
    // visible product; the sun override is honored for QA what-ifs).
    let (time, time_is_fallback) = resolve_frame_time(time_iso.as_deref());
    let solar = SolarFrame::new(time.year, time.month, time.day, time.ut);
    let (sun_ecef, center_sun_elev) = resolve_frame_sun(params, &raster, &solar, proj);

    // M2 atmosphere (sun transmittance + the SH sky ambient the cloud march reads).
    let pw_ratio = atmosphere::pw_ratio_from_brick(brick);
    let atmo_params = AtmosphereParams {
        aod: params.aerosol_optical_depth as f64,
        pw_ratio,
        aerosol_swelling: if params.rh_aerosol_swelling { 1.5 } else { 1.0 },
        ground_albedo: atmosphere::GROUND_ALBEDO,
    };
    let luts = AtmosphereLuts::build(&atmo_params);
    let sky_sh = SkyShTable::build(&luts, &atmo_params, 48);

    // M4/M5 cloud volume + scene (granulation scoping: the layer is a DISPLAY product —
    // the same opt-in rule as VisibleRgb).
    let horiz_pitch = horiz_pitch_m(proj);
    let fractional_requested = params.clouds && params.fractional_clouds;
    let fractional_clouds = fractional_requested && brick.has_cloud_fraction;
    let mut vol = if fractional_clouds {
        DecodedVolume::from_brick(brick, horiz_pitch)
    } else {
        DecodedVolume::from_brick_legacy(brick, horiz_pitch)
    };
    apply_fractional_clouds_for_visible(
        &mut vol,
        fractional_clouds,
        fractional_requested,
        "cloud layer",
    );
    let mip = OccupancyMip::build(&vol, clouds::OCCUPANCY_MIP_FACTOR);
    let gran_on = params.clouds && params.granulation.unwrap_or(false);
    let granulation = if gran_on {
        Some(clouds::Granulation::for_grid(horiz_pitch))
    } else {
        None
    };
    if let Some(g) = granulation {
        crate::log_line!(
            "simsat api: cloud-layer granulation ON (sub-grid edge erosion; dx \
             {horiz_pitch:.0} m -> amplitude {:.3})",
            g.amplitude
        );
    }
    let sun_od = clouds::accumulate_sun_od_granulated(
        &vol,
        georef,
        sun_ecef,
        SUN_OD_RESOLUTION,
        clouds::SUN_OD_EDGE_FEATHER_TEXELS,
        granulation,
    );
    let mut cfg = MarchConfig {
        beer_powder: params.beer_powder,
        cloud_optical_depth_scale: params.cloud_optical_depth_scale,
        octaves: if params.multiscatter {
            clouds::DEFAULT_OCTAVES
        } else {
            1
        },
        granulation,
        ..MarchConfig::new(params.steps, vol.voxel_pitch_m())
    };
    cfg.edge_feather_cells = clouds::edge_feather_cells_for_margin(margin, nx, ny);
    if let Some(k) = params.cloud_softclip {
        cfg.cloud_softclip_knee = k;
    }
    if let Some(m) = params.cloud_highlight_max {
        cfg.cloud_highlight_max = m;
    }
    if let Some(n) = params.topdown_cloud_norm {
        cfg.topdown_cloud_norm = n;
    }
    let scene = CloudScene {
        vol: &vol,
        mip: &mip,
        sun_od: &sun_od,
        georef,
        luts: &luts,
        sky_sh: &sky_sh,
        sun_ecef,
        cfg,
    };

    // 1. The native cloud layer + shadow on the Lambert map raster.
    let native = if params.clouds {
        crate::topdown::render_cloud_layer_frame(
            &scene,
            OutputTransform::AbiReflectance,
            params.exposure,
            &raster.lat,
            &raster.lon,
            raster.nx,
            raster.ny,
        )
    } else {
        crate::topdown::CloudLayerFrame {
            nx: raster.nx,
            ny: raster.ny,
            rgba_premul: vec![0; raster.nx * raster.ny * 4],
            shadow: vec![1.0; raster.nx * raster.ny],
        }
    };

    // 2. The Web-Mercator delivery grid over the raster's geodetic bbox, ~native pitch.
    let (la0, la1, lo0, lo1) = raster
        .lat_lon_bbox()
        .ok_or_else(|| "no on-earth pixels in the map raster".to_string())?;
    let grid = web_layer::mercator_grid_for_bbox(
        la0 as f64,
        la1 as f64,
        lo0 as f64,
        lo1 as f64,
        horiz_pitch,
        MAX_AXIS,
    )
    .ok_or_else(|| "degenerate Web-Mercator grid for the domain bbox".to_string())?;
    let (rgba_premul, shadow) =
        web_layer::reproject_cloud_layer(&native, georef, nx, ny, margin, &grid);

    // 3. The Mercator grid's georef: extent in 3857 metres + its per-pixel lat/lon mesh
    //    (lat varies only by row, lon only by column — exact Mercator alignment) + the
    //    Mapbox ImageSource corners.
    let n_out = grid.nx * grid.ny;
    let mut lat_mesh = vec![f32::NAN; n_out];
    let mut lon_mesh = vec![f32::NAN; n_out];
    let row_lat: Vec<f32> = (0..grid.ny)
        .map(|py| grid.pixel_lonlat(0, py).0 as f32)
        .collect();
    let col_lon: Vec<f32> = (0..grid.nx)
        .map(|px| grid.pixel_lonlat(px, 0).1 as f32)
        .collect();
    for (row, &la) in lat_mesh.chunks_exact_mut(grid.nx).zip(row_lat.iter()) {
        row.fill(la);
    }
    for row in lon_mesh.chunks_exact_mut(grid.nx) {
        row.copy_from_slice(&col_lon);
    }
    let georef_out = Georef {
        view: ViewMode::TopDownMap,
        projection: *proj,
        nx: grid.nx,
        ny: grid.ny,
        lat: lat_mesh,
        lon: lon_mesh,
        extent: grid.extent_3857(),
        extent_kind: ExtentKind::WebMercatorMeters,
        mercator_corners_lonlat: Some(grid.corners_lonlat()),
        camera_pose: None,
    };
    Ok(RenderResult {
        nx: grid.nx,
        ny: grid.ny,
        data: FrameData::CloudLayer {
            rgba_premul,
            shadow,
        },
        georef: georef_out,
        raster,
        sun_elev_deg: center_sun_elev,
        res_clamped: false,
        cloud_stats: None,
        time,
        time_is_fallback,
        // No ground pixels are rendered — the host map is the ground.
        ground_source: None,
        ground_status: Vec::new(),
        granulation: gran_on,
    })
}

// ── free-perspective frame (the angled-3D hero shot / host-3D cloud layer) ──────

/// Render the [`Product::Perspective`] frame. Builds the pinhole basis from
/// [`RenderParams::perspective`] (REQUIRED — a clear error otherwise), the per-pixel
/// ground raster ([`build_perspective_raster`]: earth hits carry lat/lon + WRF `(i, j)`,
/// sky rays stay `NaN`), then assembles the SAME scene the visible product marches
/// (Blue Marble ground + terrain normals + horizon map + atmosphere LUTs + SH ambient +
/// cloud volume/mip/sun-OD) and renders through
/// [`crate::topdown::render_perspective_frame_rgba`] (full composite; honors
/// [`RenderParams::clouds`]) or [`crate::topdown::render_perspective_cloud_layer`]
/// (the cloud-only premultiplied variant). `view` / `resolution` / `margin_frac` are
/// ignored — the camera IS the view (documented on the Product).
///
/// GEOREF: the lat/lon mesh is the per-pixel ground intersections (`NaN` sky) and the
/// extent is the on-earth lon/lat bbox ([`ExtentKind::LonLatDegrees`] — like the
/// geostationary frame, and like it NOT rectilinear: prefer the mesh for any
/// georeferencing; a perspective frame is a picture, not a map). `Georef::view` reuses
/// the [`ViewMode::Geostationary`] tag (a from-a-point-in-space perspective — the
/// closest of the two view modes; consumers distinguish the product by
/// [`Georef::camera_pose`], which is ALWAYS `Some` here). The camera pose is also
/// logged — the what-if labeling discipline.
fn render_perspective_scene(
    src: &SceneSource,
    params: &RenderParams,
    cloud_layer_only: bool,
) -> Result<RenderResult, String> {
    let SceneSource {
        brick,
        georef,
        params: proj,
        time_iso,
    } = src;
    let (nx, ny) = (brick.nx, brick.ny);
    let camera = params
        .perspective
        .ok_or_else(|| "Product::Perspective requires RenderParams::perspective".to_string())?;
    let basis = camera.basis()?;
    crate::log_line!("simsat api: PERSPECTIVE camera {}", camera.label());
    let raster = build_perspective_raster(&basis, georef, nx, ny);

    // Solar geometry + the ECEF sun (the sun override honored, as everywhere).
    let (time, time_is_fallback) = resolve_frame_time(time_iso.as_deref());
    let solar = SolarFrame::new(time.year, time.month, time.day, time.ut);
    let use_sun_override = params
        .sun_override
        .is_some_and(|s| s.elev_deg.is_some() || s.az_deg.is_some());
    let (sun_ecef, center_sun_elev) = resolve_frame_sun(params, &raster, &solar, proj);

    // Ground texture (full composite only — the layer-only variant renders no ground).
    let (bluemarble, ground_source, ground_status) = if cloud_layer_only {
        (
            None,
            GroundSource::FlatAlbedo("cloud-layer-only".into()),
            Vec::new(),
        )
    } else {
        resolve_bluemarble(params, &raster, time.month, time.day)?
    };

    // Per-pixel LUTs over the perspective raster (sky pixels carry the space sentinel).
    let (lut_geo, mut lut_light) = gpu::build_luts(&raster, bluemarble.as_ref(), nx, ny, &solar);
    if use_sun_override {
        override_light_lut(&mut lut_light, &raster, sun_ecef);
    }

    // Terrain + atmosphere + cloud scene (the visible product's assembly).
    let normals = normals_from_hgt(&brick.hgt, nx, ny, proj.dx_m, proj.dy_m);
    let horiz_pitch = horiz_pitch_m(proj);
    let (dx_m_m, dy_m_m) = dx_dy_metres(proj);
    let horizon_map = HorizonMap::build(&brick.hgt, nx, ny, dx_m_m, dy_m_m);
    let pw_ratio = atmosphere::pw_ratio_from_brick(brick);
    let atmo_params = AtmosphereParams {
        aod: params.aerosol_optical_depth as f64,
        pw_ratio,
        aerosol_swelling: if params.rh_aerosol_swelling { 1.5 } else { 1.0 },
        ground_albedo: atmosphere::GROUND_ALBEDO,
    };
    let luts = AtmosphereLuts::build(&atmo_params);
    let sky_sh = SkyShTable::build(&luts, &atmo_params, 48);
    let fractional_requested = params.clouds && params.fractional_clouds;
    let fractional_clouds = fractional_requested && brick.has_cloud_fraction;
    let mut vol = if fractional_clouds {
        DecodedVolume::from_brick(brick, horiz_pitch)
    } else {
        DecodedVolume::from_brick_legacy(brick, horiz_pitch)
    };
    apply_fractional_clouds_for_visible(
        &mut vol,
        fractional_clouds,
        fractional_requested,
        "perspective",
    );
    let mip = OccupancyMip::build(&vol, clouds::OCCUPANCY_MIP_FACTOR);
    // Granulation scoping: a DISPLAY product (the same opt-in rule as VisibleRgb).
    let gran_on = params.clouds && params.granulation.unwrap_or(false);
    let granulation = if gran_on {
        Some(clouds::Granulation::for_grid(horiz_pitch))
    } else {
        None
    };
    if let Some(g) = granulation {
        crate::log_line!(
            "simsat api: perspective granulation ON (sub-grid edge erosion; dx \
             {horiz_pitch:.0} m -> amplitude {:.3})",
            g.amplitude
        );
    }
    let sun_od = clouds::accumulate_sun_od_granulated(
        &vol,
        georef,
        sun_ecef,
        SUN_OD_RESOLUTION,
        clouds::SUN_OD_EDGE_FEATHER_TEXELS,
        granulation,
    );
    let mut cfg = MarchConfig {
        beer_powder: params.beer_powder,
        cloud_optical_depth_scale: params.cloud_optical_depth_scale,
        octaves: if params.multiscatter {
            clouds::DEFAULT_OCTAVES
        } else {
            1
        },
        granulation,
        ..MarchConfig::new(params.steps, vol.voxel_pitch_m())
    };
    if let Some(k) = params.cloud_softclip {
        cfg.cloud_softclip_knee = k;
    }
    if let Some(m) = params.cloud_highlight_max {
        cfg.cloud_highlight_max = m;
    }
    if let Some(g) = params.ground_gain {
        cfg.ground_day_lift = g;
    }
    let scene = CloudScene {
        vol: &vol,
        mip: &mip,
        sun_od: &sun_od,
        georef,
        luts: &luts,
        sky_sh: &sky_sh,
        sun_ecef,
        cfg,
    };
    // The frame context: ONE camera — the EYE (the perspective render contract).
    let mut cam_geo = CameraGeometry::from_sub_lon(params.satellite.sub_lon_deg());
    cam_geo.camera = basis.eye;
    let surf = FrameContext {
        luts: &luts,
        params: &atmo_params,
        sky_sh: &sky_sh,
        cam: cam_geo,
        sun_ecef,
        output_transform: OutputTransform::AbiReflectance,
        bm_present: bluemarble.is_some(),
        water_scale: WATER_ALBEDO_SCALE as f64,
        flat_albedo_srgb: FLAT_ALBEDO_SRGB as f64,
        raymarch_steps: 16,
        exposure: params.exposure,
        ground_day_lift: cfg.ground_day_lift,
        cloud_softclip_knee: cfg.cloud_softclip_knee,
        cloud_highlight_max: cfg.cloud_highlight_max,
        atmosphere_correction: params.atmosphere_correction,
        terrain_atmosphere: params.terrain_atmosphere,
    };
    let assemble = make_assemble(
        brick,
        &lut_geo,
        &lut_light,
        bluemarble.as_ref(),
        &normals,
        &horizon_map,
        raster.nx,
        nx,
        ny,
    );

    let rgba = if cloud_layer_only && params.clouds {
        crate::topdown::render_perspective_cloud_layer(
            &scene,
            &basis,
            OutputTransform::AbiReflectance,
            params.exposure,
        )
    } else if cloud_layer_only {
        vec![0; raster.nx * raster.ny * 4]
    } else {
        let persp_scene = if params.clouds { Some(&scene) } else { None };
        crate::topdown::render_perspective_frame_rgba(&surf, persp_scene, &basis, assemble)
    };
    let rgb = rgba_to_rgb_black_space(&rgba, raster.nx, raster.ny);

    // The perspective georef: the ground-intersection mesh + the on-earth lon/lat bbox.
    let (extent, extent_kind) = match raster.lat_lon_bbox() {
        Some((la0, la1, lo0, lo1)) => (
            [lo0 as f64, lo1 as f64, la0 as f64, la1 as f64],
            ExtentKind::LonLatDegrees,
        ),
        None => (
            [
                proj.cen_lon_deg,
                proj.cen_lon_deg,
                proj.cen_lat_deg,
                proj.cen_lat_deg,
            ],
            ExtentKind::LonLatDegrees,
        ),
    };
    let georef_out = Georef {
        view: ViewMode::Geostationary,
        projection: *proj,
        nx: raster.nx,
        ny: raster.ny,
        lat: raster.lat.clone(),
        lon: raster.lon.clone(),
        extent,
        extent_kind,
        mercator_corners_lonlat: None,
        camera_pose: Some(camera),
    };
    Ok(RenderResult {
        nx: raster.nx,
        ny: raster.ny,
        data: FrameData::Visible { rgb, rgba },
        georef: georef_out,
        raster,
        sun_elev_deg: center_sun_elev,
        res_clamped: false,
        cloud_stats: None,
        time,
        time_is_fallback,
        ground_source: if cloud_layer_only {
            None
        } else {
            Some(ground_source)
        },
        ground_status,
        granulation: gran_on,
    })
}

// ── helpers ─────────────────────────────────────────────────────────────────────

/// Parse the source's ISO valid time into a [`FrameTime`]. When the time is absent or
/// unparseable the FIXED fallback date 2004-06-21 12:00 UT is returned (byte-identical
/// to the historic behavior — existing QA frames do not change) and the fallback is
/// FLAGGED: the sun position and Blue Marble season derived from a fabricated date are
/// not the run's real conditions, and [`RenderResult::time_is_fallback`] lets consumers
/// see that instead of silently trusting the frame.
fn resolve_frame_time(time_iso: Option<&str>) -> (FrameTime, bool) {
    match time_iso.and_then(crate::solar::parse_iso_utc) {
        Some((year, month, day, ut)) => (
            FrameTime {
                year,
                month,
                day,
                ut,
            },
            false,
        ),
        None => (
            FrameTime {
                year: 2004,
                month: 6,
                day: 21,
                ut: 12.0,
            },
            true,
        ),
    }
}

/// Resolve a frame's single ECEF sun vector (sun at infinity) + the domain-centre solar
/// elevation, honoring a synthetic sun override (an unset component keeps the timestep's
/// real value). Shared by the visible pass and the GeoColor blend so the blend's per-pixel
/// elevation is derived from the SAME sun that lit the visible half.
fn resolve_frame_sun(
    params: &RenderParams,
    raster: &SurfaceRaster,
    solar: &SolarFrame,
    proj: &WrfProjectionParams,
) -> ([f64; 3], f64) {
    let (clat, clon) = raster
        .lat_lon_bbox()
        .map(|(la0, la1, lo0, lo1)| (((la0 + la1) * 0.5) as f64, ((lo0 + lo1) * 0.5) as f64))
        .unwrap_or((proj.cen_lat_deg, proj.cen_lon_deg));
    let real_sun = solar.at(clat, clon);
    let use_override = params
        .sun_override
        .is_some_and(|s| s.elev_deg.is_some() || s.az_deg.is_some());
    let sun_ecef = if use_override {
        let o = params.sun_override.unwrap_or_default();
        let elev_deg = o.elev_deg.unwrap_or(real_sun.elevation_deg);
        let az_deg = o.az_deg.unwrap_or(real_sun.azimuth_deg);
        let (e, az) = (elev_deg.to_radians(), az_deg.to_radians());
        let sun_enu = [e.cos() * az.sin(), e.cos() * az.cos(), e.sin()];
        sun_enu_to_ecef(sun_enu, clat, clon)
    } else {
        sun_enu_to_ecef(real_sun.enu_direction(), clat, clon)
    };
    let center_elev = sun_enu_and_elev(sun_ecef, clat, clon).1;
    (sun_ecef, center_elev)
}

/// The horizontal march pitch (m): min(dx, dy), converting a `MAP_PROJ = 6` lat/lon grid's
/// degree spacing to metres.
fn horiz_pitch_m(proj: &WrfProjectionParams) -> f64 {
    if proj.map_proj == 6 {
        proj.dx_m.min(proj.dy_m) * 111_195.0
    } else {
        proj.dx_m.min(proj.dy_m)
    }
}

/// Resolve the visible-only fractional-cloud switch before any acceleration or
/// lighting structure is built, so every consumer sees the same extinction field.
fn apply_fractional_clouds_for_visible(
    vol: &mut DecodedVolume,
    effective: bool,
    requested: bool,
    label: &str,
) {
    if !effective {
        if requested {
            crate::log_line!(
                "simsat api: {label} model cloud fraction unavailable; using legacy full-cell clouds"
            );
        }
        return;
    }
    let stats = vol.apply_fractional_clouds();
    let ratio = if stats.raw_fractional_tau > 0.0 {
        stats.effective_fractional_tau / stats.raw_fractional_tau
    } else {
        1.0
    };
    crate::log_line!(
        "simsat api: {label} model cloud fraction ON ({} / {} columns adjusted; {} partial layers; \
         {} invalid zero-fraction layers repaired; partial-column tau ratio {ratio:.3})",
        stats.columns_modified,
        stats.columns_total,
        stats.fractional_layer_count,
        stats.repaired_zero_count,
    );
}

/// `(dx, dy)` in metres (a lat/lon grid stores them in degrees).
fn dx_dy_metres(proj: &WrfProjectionParams) -> (f64, f64) {
    if proj.map_proj == 6 {
        (proj.dx_m * 111_195.0, proj.dy_m * 111_195.0)
    } else {
        (proj.dx_m, proj.dy_m)
    }
}

/// Build the georef returned to the caller: for the top-down map the `imshow` extent is
/// the domain's projection-plane (Lambert metres) outer edges; for the geostationary frame
/// it is the on-earth lon/lat bounding box. The lat/lon mesh is the raster's per-pixel
/// geodetic coordinates (for `pcolormesh`).
fn build_georef(
    view: ViewMode,
    proj: &WrfProjectionParams,
    georef: &GridGeoref,
    raster: &SurfaceRaster,
    domain_nx: usize,
    domain_ny: usize,
    margin_frac: f64,
) -> Georef {
    let (nx, ny) = (raster.nx, raster.ny);
    let (extent, extent_kind) = match view {
        ViewMode::TopDownMap => {
            // imshow pixel-edge extent = the plane coord at the raster's OUTER half-pixel
            // edges, in WRF grid-index space. With a zoom-out margin the raster spans
            // beyond the domain [0, n-1] box, so the edge bounds come from
            // `map_pixel_edge_index_bounds` (domain + margin); at margin 0 they reduce to
            // the domain's half-cell box (-0.5 .. n-0.5), so this is byte-identical to the
            // pre-margin extent. Row 0 = north (max j) -> `top`.
            let (i_min, i_max, j_min, j_max) =
                map_pixel_edge_index_bounds(domain_nx, domain_ny, nx, ny, margin_frac);
            let (x0, _) = georef.plane_uv(i_min, 0.0);
            let (x1, _) = georef.plane_uv(i_max, 0.0);
            let (_, y_south) = georef.plane_uv(0.0, j_min);
            let (_, y_north) = georef.plane_uv(0.0, j_max);
            ([x0, x1, y_south, y_north], ExtentKind::ProjectionMeters)
        }
        ViewMode::Geostationary => {
            let (lo0, lo1, la0, la1) = match raster.lat_lon_bbox() {
                Some((la_min, la_max, lo_min, lo_max)) => {
                    (lo_min as f64, lo_max as f64, la_min as f64, la_max as f64)
                }
                None => (
                    proj.cen_lon_deg,
                    proj.cen_lon_deg,
                    proj.cen_lat_deg,
                    proj.cen_lat_deg,
                ),
            };
            ([lo0, lo1, la0, la1], ExtentKind::LonLatDegrees)
        }
    };
    Georef {
        view,
        projection: *proj,
        nx,
        ny,
        lat: raster.lat.clone(),
        lon: raster.lon.clone(),
        extent,
        extent_kind,
        mercator_corners_lonlat: None,
        camera_pose: None,
    }
}

/// Resolve the ground texture for a visible frame: `(crop, source, status lines)`.
///
/// Honesty rules (WS3 — no more silent downgrades):
/// - [`BlueMarble::SingleFile`] that fails to load is a HARD `Err` (behavior change vs
///   the pre-WS3 silent flat-albedo fallback): the user explicitly named the file, so
///   silently rendering something else is the silent-wrong-output class.
/// - [`BlueMarble::Seasonal`] keeps its NEVER-hard-fail offline contract: any failure
///   degrades to flat albedo, but the degradation is REPORTED via [`GroundSource`] and
///   the collected asset-pack status lines (previously discarded by a no-op callback).
fn resolve_bluemarble(
    params: &RenderParams,
    raster: &SurfaceRaster,
    month: u32,
    day: u32,
) -> Result<
    (
        Option<bluemarble::BlueMarbleCrop>,
        GroundSource,
        Vec<String>,
    ),
    String,
> {
    let Some((la0, la1, lo0, lo1)) = raster.lat_lon_bbox() else {
        return Ok((
            None,
            GroundSource::FlatAlbedo("no on-earth pixels in the raster".to_string()),
            Vec::new(),
        ));
    };
    match &params.bluemarble {
        BlueMarble::FlatAlbedo => Ok((
            None,
            GroundSource::FlatAlbedo("flat albedo requested".to_string()),
            Vec::new(),
        )),
        BlueMarble::SingleFile(path) => {
            let crop = bluemarble::load_crop(path, la0, la1, lo0, lo1, 1.0, BLUEMARBLE_MAX_AXIS)
                .map_err(|e| format!("blue marble file {}: {e}", path.display()))?;
            Ok((Some(crop), GroundSource::SingleFile, Vec::new()))
        }
        BlueMarble::Seasonal {
            month_override,
            download,
        } => {
            let manifest = asset_pack::embedded_manifest();
            let mut lines: Vec<String> = Vec::new();
            let mut status = |s: String| lines.push(s);
            match asset_pack::load_season_ground(
                &params.cache,
                &manifest,
                month,
                day,
                *month_override,
                *download,
                la0,
                la1,
                lo0,
                lo1,
                1.0,
                BLUEMARBLE_MAX_AXIS,
                &mut status,
            ) {
                Ok(g) => {
                    lines.push(g.status_line());
                    let source = if g.used_fallback() {
                        GroundSource::EightKmFallback
                    } else {
                        GroundSource::TwoKm
                    };
                    Ok((Some(g.crop), source, lines))
                }
                Err(e) => {
                    let reason =
                        format!("seasonal blue marble failed ({e}); rendering flat albedo");
                    lines.push(reason.clone());
                    Ok((None, GroundSource::FlatAlbedo(reason), lines))
                }
            }
        }
    }
}

/// Build the per-pixel surface `assemble` closure (twin of the studio's / render_frame's).
#[allow(clippy::too_many_arguments)]
fn make_assemble<'a>(
    brick: &'a VolumeBrick,
    lut_geo: &'a [f32],
    lut_light: &'a [f32],
    bm: Option<&'a bluemarble::BlueMarbleCrop>,
    normals: &'a [[f32; 3]],
    horizon_map: &'a HorizonMap,
    rnx: usize,
    nx: usize,
    ny: usize,
) -> impl Fn(usize, usize) -> SurfacePixel + Sync + 'a {
    move |px: usize, py: usize| -> SurfacePixel {
        let idx = py * rnx + px;
        let g = &lut_geo[idx * 4..idx * 4 + 4];
        if g[0] < 0.0 {
            return SurfacePixel {
                on_earth: false,
                ..Default::default()
            };
        }
        let l = &lut_light[idx * 4..idx * 4 + 4];
        let sun_enu = [l[0], l[1], l[2]];
        let mut base = match bm {
            Some(bm) if bm.width > 0 && bm.height > 0 => bm.sample_bilinear(g[0], g[1]),
            _ => [FLAT_ALBEDO_SRGB, FLAT_ALBEDO_SRGB, FLAT_ALBEDO_SRGB],
        };
        let (
            normal_enu,
            is_water,
            terrain_horizon_rad,
            sky_openness,
            bent_normal_enu,
            wind_speed,
            surface_elevation_m,
        ) = if g[2] >= 0.0 {
            let fi_f = (g[2] as f64 * nx as f64 - 0.5).clamp(0.0, (nx - 1) as f64);
            let fj_f = (g[3] as f64 * ny as f64 - 0.5).clamp(0.0, (ny - 1) as f64);
            let cell = fj_f.round() as usize * nx + fi_f.round() as usize;
            let is_water = brick.landmask[cell] < 0.5;
            let sun_az = (sun_enu[1] as f64).atan2(sun_enu[0] as f64);
            let horizon = horizon_map.horizon_angle_at(fi_f, fj_f, sun_az) as f32;
            let (openness, bent) = horizon_map.aperture_at(fi_f, fj_f);
            let wind = (brick.u10[cell].powi(2) + brick.v10[cell].powi(2)).sqrt();
            let elevation = brick.hgt[cell];
            let elevation = if elevation.is_finite() {
                elevation
            } else {
                0.0
            };
            if !is_water {
                let snow = brick
                    .snowh
                    .as_ref()
                    .map_or(0.0, |s| snow_fraction(s[cell] as f64));
                base = blend_snow(base, snow);
            }
            (
                normals[cell],
                is_water,
                horizon,
                openness as f32,
                [bent[0] as f32, bent[1] as f32, bent[2] as f32],
                wind,
                elevation,
            )
        } else {
            ([0.0, 0.0, 1.0], false, 0.0, 1.0, [0.0, 0.0, 1.0], 0.0, 0.0)
        };
        SurfacePixel {
            on_earth: true,
            base_srgb: base,
            normal_enu,
            sun_enu,
            sun_elev_deg: l[3],
            is_water,
            view_dir: [0.0, 0.0, 1.0],
            terrain_horizon_rad,
            sky_openness,
            bent_normal_enu,
            wind_speed,
            surface_elevation_m,
        }
    }
}

/// Render the geostationary frame SURFACE-ONLY (clouds off): the M2 atmosphere + M3
/// terrain shadows / glint / snow, no cloud march. Rows in parallel; space -> alpha 0.
fn render_geo_surface_rgba(
    surf: &FrameContext,
    raster: &SurfaceRaster,
    assemble: impl Fn(usize, usize) -> SurfacePixel + Sync,
) -> Vec<u8> {
    use rayon::prelude::*;
    let scan = &raster.scan;
    let (nx, ny) = (scan.nx, scan.ny);
    let rows: Vec<Vec<u8>> = (0..ny)
        .into_par_iter()
        .map(|py| {
            let mut row = Vec::with_capacity(nx * 4);
            for px in 0..nx {
                let (sx, sy) = scan.scan_angle(px, py);
                let mut pixel = assemble(px, py);
                pixel.view_dir = surf.cam.view_dir(sx, sy);
                let rgba = shade_surface(surf, &pixel);
                for &v in &rgba {
                    row.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
                }
            }
            row
        })
        .collect();
    rows.into_iter().flatten().collect()
}

/// Render geostationary pre-tonemap reflectance with the cloud feature explicitly off.
/// This is the quantitative twin of [`render_geo_surface_rgba`], so the public
/// `RenderParams::clouds` switch has the same meaning for RGB and visible bands.
fn render_geo_surface_reflectance(
    surf: &FrameContext,
    raster: &SurfaceRaster,
    assemble: impl Fn(usize, usize) -> SurfacePixel + Sync,
) -> Vec<f32> {
    use rayon::prelude::*;
    let scan = &raster.scan;
    let (nx, ny) = (scan.nx, scan.ny);
    let rows: Vec<Vec<f32>> = (0..ny)
        .into_par_iter()
        .map(|py| {
            let mut row = Vec::with_capacity(nx * 3);
            for px in 0..nx {
                let (sx, sy) = scan.scan_angle(px, py);
                let mut pixel = assemble(px, py);
                pixel.view_dir = surf.cam.view_dir(sx, sy);
                let reflectance =
                    surface_toa_radiance(surf, &pixel, 1.0, crate::render::GROUND_DAY_LIFT)
                        .map(reflectance_from_radiance)
                        .unwrap_or([0.0; 3]);
                row.extend_from_slice(&reflectance);
            }
            row
        })
        .collect();
    rows.into_iter().flatten().collect()
}

/// Reduce an RGBA frame (row 0 = north; alpha 0 = space) to an RGB8 buffer (`nx*ny*3`),
/// rendering space (alpha 0) as black.
pub fn rgba_to_rgb_black_space(rgba: &[u8], nx: usize, ny: usize) -> Vec<u8> {
    let mut out = vec![0u8; nx * ny * 3];
    for (i, px) in rgba.chunks_exact(4).enumerate().take(nx * ny) {
        if px[3] != 0 {
            out[i * 3] = px[0];
            out[i * 3 + 1] = px[1];
            out[i * 3 + 2] = px[2];
        }
    }
    out
}

/// Rewrite the per-pixel light LUT so every on-earth pixel's sun comes from the single
/// global `sun_ecef` (the synthetic sun override). Inverse of `sun_enu_to_ecef`.
fn override_light_lut(light: &mut [f32], raster: &SurfaceRaster, sun_ecef: [f64; 3]) {
    let n = raster.nx * raster.ny;
    for idx in 0..n {
        let (lat, lon) = (raster.lat[idx], raster.lon[idx]);
        if !lat.is_finite() || !lon.is_finite() {
            continue;
        }
        let (enu, elev) = sun_enu_and_elev(sun_ecef, lat as f64, lon as f64);
        let l = &mut light[idx * 4..idx * 4 + 4];
        l[0] = enu[0] as f32;
        l[1] = enu[1] as f32;
        l[2] = enu[2] as f32;
        l[3] = elev as f32;
    }
}

/// Project a global ECEF sun direction into the local ENU basis at `(lat, lon)`, returning
/// `(sun_enu, elevation_deg)`. Inverse of `atmosphere::sun_enu_to_ecef`.
fn sun_enu_and_elev(sun_ecef: [f64; 3], lat_deg: f64, lon_deg: f64) -> ([f64; 3], f64) {
    let (la, lo) = (lat_deg.to_radians(), lon_deg.to_radians());
    let (sla, cla) = la.sin_cos();
    let (slo, clo) = lo.sin_cos();
    let east = [-slo, clo, 0.0];
    let north = [-sla * clo, -sla * slo, cla];
    let up = [cla * clo, cla * slo, sla];
    let dot = |a: [f64; 3]| a[0] * sun_ecef[0] + a[1] * sun_ecef[1] + a[2] * sun_ecef[2];
    let (e, n, u) = (dot(east), dot(north), dot(up));
    let elev = u.clamp(-1.0, 1.0).asin().to_degrees();
    ([e, n, u], elev)
}

// ── source resolution (wrfout ingest-if-needed, or a cached run.json) ──────────

fn resolve_source(params: &RenderParams) -> Result<SceneSource, String> {
    let is_json = params
        .input
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("json"))
        .unwrap_or(false);
    if is_json {
        resolve_cached(params)
    } else if ingest_grib::is_grib_input(&params.input) {
        resolve_grib(params)
    } else {
        resolve_wrfout(params)
    }
}

fn resolve_wrfout(params: &RenderParams) -> Result<SceneSource, String> {
    let path = &params.input;
    if !path.is_file() {
        return Err(format!("wrfout not found: {}", path.display()));
    }
    let geom = ingest::read_grid_geometry(path, params.timestep)
        .map_err(|e| format!("read geometry: {e}"))?;
    let georef = geom.georef().map_err(|e| format!("georef: {e}"))?;
    let run_id = ingest::default_run_id(path);
    let brick_file = bricks::brick_file_name_for(geom.time_iso.as_deref(), geom.hhmm);
    let brick_path = bricks::run_dir(&params.cache, &run_id).join(&brick_file);
    // Cache-hit STALENESS gate (WS3): a brick on disk is only reused when the run
    // manifest recorded the SAME source-file identity (byte length + mtime) it was
    // ingested from. Re-running WRF over the same wrfout path previously rendered
    // the OLD cached data silently; a pre-WS3 manifest (no identity fields) is
    // stale-once and self-heals here.
    let needs_ingest = if !brick_path.is_file() {
        true
    } else {
        let (src_bytes, src_mtime) = ingest::source_identity(path);
        let fresh = match (src_bytes, src_mtime) {
            (Some(b), Some(m)) => RunManifest::load(&RunManifest::path(&params.cache, &run_id))
                .ok()
                .and_then(|man| man.timesteps.iter().find(|t| t.file == brick_file).cloned())
                .is_some_and(|t| bricks::cache_entry_is_fresh(&t, b, m)),
            _ => false,
        };
        if !fresh {
            crate::log_line!(
                "simsat api: cached brick {} is STALE against {} (source bytes/mtime \
                 changed, or a pre-staleness-gate manifest); re-ingesting",
                brick_path.display(),
                path.display()
            );
        }
        !fresh
    };
    if needs_ingest {
        let mut config = IngestConfig::new(params.cache.clone());
        config.run_id = Some(run_id.clone());
        config.timestep = params.timestep;
        ingest::ingest_timestep(path, &config).map_err(|e| format!("ingest: {e}"))?;
    }
    let brick = bricks::read_ssb(&brick_path).map_err(|e| format!("read brick: {e}"))?;
    Ok(SceneSource {
        brick,
        georef,
        params: geom.params,
        time_iso: geom.time_iso,
    })
}

/// GRIB2 sibling of [`resolve_wrfout`] (the deferred grib-ingest integration seam):
/// probe carries the run-id + single valid time (a GRIB file is one forecast hour, so
/// `timestep` must be 0), the same WS3 staleness gate guards the cached brick, and
/// ingest goes through [`ingest_grib::ingest_grib_timestep`]. NOTE: no crop parameter is
/// plumbed yet — an RRFS full-file open refuses with the crop remedy message (HRRR needs
/// no crop); plumbing a crop field through `RenderParams` is the recorded open decision.
fn resolve_grib(params: &RenderParams) -> Result<SceneSource, String> {
    let path = &params.input;
    if !path.is_file() {
        return Err(format!("grib input not found: {}", path.display()));
    }
    if params.timestep != 0 {
        return Err(format!(
            "timestep {} out of range: a GRIB file carries a single valid time \
             (forecast hours are separate files; use timestep 0)",
            params.timestep
        ));
    }
    let probe = ingest_grib::probe_grib(path).map_err(|e| format!("probe grib: {e}"))?;
    let geom = ingest_grib::read_grib_geometry(path).map_err(|e| format!("read geometry: {e}"))?;
    let georef = geom.georef().map_err(|e| format!("georef: {e}"))?;
    let run_id = probe.default_run_id.clone();
    let brick_file = bricks::brick_file_name_for(geom.time_iso.as_deref(), geom.hhmm);
    let brick_path = bricks::run_dir(&params.cache, &run_id).join(&brick_file);
    let needs_ingest = if !brick_path.is_file() {
        true
    } else {
        let (src_bytes, src_mtime) = ingest::source_identity(path);
        let fresh = match (src_bytes, src_mtime) {
            (Some(b), Some(m)) => RunManifest::load(&RunManifest::path(&params.cache, &run_id))
                .ok()
                .and_then(|man| man.timesteps.iter().find(|t| t.file == brick_file).cloned())
                .is_some_and(|t| bricks::cache_entry_is_fresh(&t, b, m)),
            _ => false,
        };
        if !fresh {
            crate::log_line!(
                "simsat api: cached grib brick {} is STALE against {}; re-ingesting",
                brick_path.display(),
                path.display()
            );
        }
        !fresh
    };
    if needs_ingest {
        let mut config = IngestConfig::new(params.cache.clone());
        config.run_id = Some(run_id.clone());
        ingest_grib::ingest_grib_timestep(path, &config)
            .map_err(|e| format!("grib ingest: {e}"))?;
    }
    let brick = bricks::read_ssb(&brick_path).map_err(|e| format!("read brick: {e}"))?;
    Ok(SceneSource {
        brick,
        georef,
        params: geom.params,
        time_iso: geom.time_iso,
    })
}

fn resolve_cached(params: &RenderParams) -> Result<SceneSource, String> {
    let manifest = RunManifest::load(&params.input).map_err(|e| format!("read run.json: {e}"))?;
    let cache_dir = params
        .input
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| params.cache.clone());
    let ts = manifest
        .timesteps
        .get(params.timestep)
        .ok_or_else(|| {
            format!(
                "timestep {} out of range (run has {} timesteps)",
                params.timestep,
                manifest.timesteps.len()
            )
        })?
        .clone();
    let brick_path = bricks::run_dir(&cache_dir, &manifest.run_id).join(&ts.file);
    if !brick_path.is_file() {
        return Err(format!("cached brick not found: {}", brick_path.display()));
    }
    let brick = bricks::read_ssb(&brick_path).map_err(|e| format!("read brick: {e}"))?;
    let p = &manifest.projection;
    let projp = WrfProjectionParams {
        map_proj: p.map_proj,
        truelat1_deg: p.truelat1_deg,
        truelat2_deg: p.truelat2_deg,
        stand_lon_deg: p.stand_lon_deg,
        cen_lat_deg: p.cen_lat_deg,
        cen_lon_deg: p.cen_lon_deg,
        dx_m: p.dx_m,
        dy_m: p.dy_m,
    };
    // Prefer the persisted per-timestep anchor: it rebuilds the wrfout path's georef
    // BIT-IDENTICALLY (closes deferred M1 NOTE-4 — no duplicate store-run forks) and
    // places a MOVING NEST's timestep where that timestep's domain actually sits.
    // A pre-WS3 manifest (no anchor) keeps the CEN_LAT/CEN_LON approximation.
    let georef = match &ts.anchor {
        Some(a) => GridGeoref::from_anchor(
            &projp,
            a.ref_i,
            a.ref_j,
            a.ref_lat_deg,
            a.ref_lon_deg,
            a.dx,
            a.dy,
        )
        .map_err(|e| format!("georef: {e}"))?,
        None => GridGeoref::from_params_center(&projp, brick.nx, brick.ny)
            .map_err(|e| format!("georef: {e}"))?,
    };
    Ok(SceneSource {
        brick,
        georef,
        params: projp,
        time_iso: ts.time_iso,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bricks::{ChannelQuant, LogQuant, encode_temperature_celsius};
    use crate::frame::MapProjection;
    use std::collections::BTreeMap;

    #[test]
    fn render_params_default_visible_controls_are_intentional() {
        let p = RenderParams::new(PathBuf::from("input"));
        assert_eq!(
            p.aerosol_optical_depth.to_bits(),
            (atmosphere::DEFAULT_AOD as f32).to_bits()
        );
        assert!(!p.rh_aerosol_swelling);
        assert!(p.atmosphere_correction);
        assert!(p.terrain_atmosphere);
        assert_eq!(
            p.cloud_optical_depth_scale,
            clouds::DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE
        );
        assert!(p.multiscatter);
        assert!(!p.beer_powder);
        assert!(p.clouds);
        assert!(p.fractional_clouds);
        assert!(p.granulation.is_none());
        assert!(p.ground_gain.is_none());
        assert!(p.cloud_softclip.is_none());
        assert!(p.cloud_highlight_max.is_none());
    }

    /// A tiny synthetic CONUS-centred Lambert scene: a clear brick (no cloud) with a
    /// realistic lapse-rate temperature profile + warm ground, so the visible render is a
    /// lit surface and the IR reads ~ground skin temperature. No file, no network.
    fn synthetic_source(nx: usize, ny: usize, nz: usize, dx: f64) -> SceneSource {
        let n3 = nx * ny * nz;
        let n2 = nx * ny;
        let dz = 250.0f64;
        // Kelvin lapse 290 K at the ground, -6.5 K/km up.
        let mut kelvin = vec![0.0f32; n3];
        for k in 0..nz {
            let t = 290.0 - 6.5 * (k as f64 * dz / 1000.0);
            kelvin[k * n2..(k + 1) * n2].fill(t as f32);
        }
        let zero = LogQuant {
            vmin: 0.0,
            vmax: 0.0,
        };
        let mut q = BTreeMap::new();
        for name in [
            "ext_liquid",
            "ext_ice",
            "ext_snow",
            "ext_precip",
            "tau_up",
            "qvapor",
        ] {
            q.insert(name.to_string(), zero);
        }
        let brick = VolumeBrick {
            nx,
            ny,
            nz,
            z_min_m: 0.0,
            dz_m: dz,
            time_iso: None,
            quant: ChannelQuant(q),
            ext_liquid: vec![0u8; n3],
            ext_ice: vec![0u8; n3],
            ext_snow: vec![0u8; n3],
            ext_precip: vec![0u8; n3],
            tau_up: vec![0u8; n3],
            qvapor: vec![0u8; n3],
            cloud_fraction: vec![255u8; n3],
            has_cloud_fraction: false,
            temperature_f16: encode_temperature_celsius(&kelvin),
            hgt: vec![0.0f32; n2],
            landmask: vec![1.0f32; n2], // all land
            tsk: vec![291.0f32; n2],
            u10: vec![0.0f32; n2],
            v10: vec![0.0f32; n2],
            snowh: None,
            ivgtyp: None,
        };
        let proj = WrfProjectionParams {
            map_proj: 1,
            truelat1_deg: 30.0,
            truelat2_deg: 60.0,
            stand_lon_deg: -97.5,
            cen_lat_deg: 39.0,
            cen_lon_deg: -97.5,
            dx_m: dx,
            dy_m: dx,
        };
        let georef = GridGeoref::new(
            MapProjection::lambert(30.0, 60.0, -97.5),
            (nx - 1) as f64 / 2.0,
            (ny - 1) as f64 / 2.0,
            39.0,
            -97.5,
            dx,
            dx,
        );
        SceneSource {
            brick,
            georef,
            params: proj,
            time_iso: None,
        }
    }

    /// Params for the synthetic scene: flat albedo (no I/O), a forced 45-deg sun so the
    /// visible frame is deterministically lit, the requested view.
    fn synthetic_params(view: ViewMode) -> RenderParams {
        let mut p = RenderParams::new(PathBuf::from("<synthetic>"));
        p.view = view;
        p.bluemarble = BlueMarble::FlatAlbedo;
        p.sun_override = Some(SunOverride {
            elev_deg: Some(45.0),
            az_deg: Some(180.0),
        });
        p
    }

    #[test]
    fn visible_rgb_topdown_shape_and_lit() {
        let src = synthetic_source(24, 24, 24, 3000.0);
        let p = synthetic_params(ViewMode::TopDownMap);
        let res = render_visible_scene(&src, &p, Product::VisibleRgb).unwrap();
        assert_eq!((res.nx, res.ny), (24, 24));
        match &res.data {
            FrameData::Visible { rgb, rgba } => {
                assert_eq!(rgb.len(), 24 * 24 * 3);
                assert_eq!(rgba.len(), 24 * 24 * 4);
                // Top-down: every pixel is on the domain (opaque) and the surface is lit.
                assert!(rgba.chunks_exact(4).all(|px| px[3] == 255));
                let peak = rgb.iter().copied().max().unwrap();
                assert!(peak > 0, "the lit daytime surface should not be black");
            }
            other => panic!("expected Visible, got {other:?}"),
        }
        // Georef: top-down extent is projection metres; the mesh is nx*ny.
        assert_eq!(res.georef.extent_kind, ExtentKind::ProjectionMeters);
        assert_eq!(res.georef.lat.len(), 24 * 24);
        assert_eq!(res.georef.lon.len(), 24 * 24);
        assert_eq!(res.georef.proj_kind(), "lcc");
        // The extent spans a real domain (x1 > x0; top(north) != bottom(south)).
        let [x0, x1, y0, y1] = res.georef.extent;
        assert!(x1 > x0 && (y1 - y0).abs() > 0.0);
    }

    #[test]
    fn visible_bands_reflectance_is_finite_in_zero_one() {
        let src = synthetic_source(24, 24, 24, 3000.0);
        let p = synthetic_params(ViewMode::TopDownMap);
        let res = render_visible_scene(&src, &p, Product::VisibleBands).unwrap();
        match &res.data {
            FrameData::Bands { reflectance } => {
                assert_eq!(reflectance.len(), 24 * 24 * 3);
                assert!(
                    reflectance
                        .iter()
                        .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
                );
                assert!(
                    reflectance.iter().cloned().fold(0.0f32, f32::max) > 0.0,
                    "the lit surface should have positive reflectance"
                );
            }
            other => panic!("expected Bands, got {other:?}"),
        }
    }

    #[test]
    fn visible_bands_ignore_display_only_ground_and_highlight_controls() {
        let src = synthetic_source(16, 16, 16, 3000.0);
        let baseline_params = synthetic_params(ViewMode::TopDownMap);
        let baseline = render_visible_scene(&src, &baseline_params, Product::VisibleBands)
            .expect("baseline bands");

        let mut adjusted_params = baseline_params;
        adjusted_params.ground_gain = Some(0.5);
        adjusted_params.cloud_softclip = Some(0.4);
        adjusted_params.cloud_highlight_max = Some(2.0);
        adjusted_params.topdown_cloud_norm = Some(0.5);
        let adjusted = render_visible_scene(&src, &adjusted_params, Product::VisibleBands)
            .expect("bands with display overrides");

        let bits = |result: &RenderResult| match &result.data {
            FrameData::Bands { reflectance } => {
                reflectance.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            }
            other => panic!("expected Bands, got {other:?}"),
        };
        assert_eq!(
            bits(&baseline),
            bits(&adjusted),
            "display calibration controls must not mutate raw reflectance bands"
        );
    }

    #[test]
    fn ir_topdown_is_kelvin_near_skin_temperature() {
        let src = synthetic_source(20, 20, 32, 3000.0);
        let mut p = synthetic_params(ViewMode::TopDownMap);
        p.ir_enhancement = Some(IrEnhancement::Grayscale);
        let res = render_ir_scene(&src, &p, IrConfig::band13()).unwrap();
        match &res.data {
            FrameData::Ir { bt_kelvin, rgb } => {
                assert_eq!(bt_kelvin.len(), 20 * 20);
                // Clear warm ground -> BT ~ TSK (291 K); every finite value is a plausible
                // Kelvin brightness temperature (200-320 K).
                let finite: Vec<f32> = bt_kelvin
                    .iter()
                    .copied()
                    .filter(|v| v.is_finite())
                    .collect();
                assert!(!finite.is_empty(), "IR BT plane has no in-domain pixels");
                assert!(
                    finite.iter().all(|&v| (200.0..=320.0).contains(&v)),
                    "IR BT out of Kelvin range: {finite:?}"
                );
                let centre = bt_kelvin[(20 / 2) * 20 + 20 / 2];
                assert!(
                    (centre - 291.0).abs() < 3.0,
                    "clear IR BT {centre} should be near TSK 291 K"
                );
                // With an enhancement requested, a colored RGB is also returned.
                assert_eq!(rgb.as_ref().unwrap().len(), 20 * 20 * 3);
            }
            other => panic!("expected Ir, got {other:?}"),
        }
        // IR reports no sun (thermal) and a lonlat / projection extent per the view.
        assert_eq!(res.sun_elev_deg, 0.0);
        assert_eq!(res.georef.extent_kind, ExtentKind::ProjectionMeters);
    }

    /// The synthetic source with an exponential moisture column baked into the qvapor
    /// channel (`q0 exp(-z/2500)`), so the WV march has a real absorber to emit from.
    fn moist_source(nx: usize, ny: usize, nz: usize, dx: f64, q0: f64) -> SceneSource {
        let mut src = synthetic_source(nx, ny, nz, dx);
        let dz = src.brick.dz_m;
        let n2 = nx * ny;
        let mut q = vec![0f32; nx * ny * nz];
        for k in 0..nz {
            let v = (q0 * (-(k as f64 * dz) / 2500.0).exp()) as f32;
            q[k * n2..(k + 1) * n2].fill(v);
        }
        let (lq, codes) = crate::bricks::encode_log_channel(&q);
        src.brick.qvapor = codes;
        src.brick.quant.0.insert("qvapor".to_string(), lq);
        src
    }

    #[test]
    fn water_vapor_is_kelvin_shaped_and_moist_reads_colder_in_62() {
        // api::render(Product::WaterVapor) returns the right shape + a plausible Kelvin
        // range, and a MOIST column reads a COLDER 6.2 um BT than a DRY one (the WV
        // weighting function tracks upper-level moisture). Dispatch is via the public
        // `render` entry so the Product routing is exercised end to end.
        let mut p = synthetic_params(ViewMode::TopDownMap);
        p.ir_enhancement = Some(IrEnhancement::Cimss); // = the classic WV moisture palette
        let cfg = WvBand::Upper.ir_config();
        let dry = render_ir_scene(&synthetic_source(20, 20, 40, 3000.0), &p, cfg).unwrap();
        let moist = render_ir_scene(&moist_source(20, 20, 40, 3000.0, 0.014), &p, cfg).unwrap();
        let centre = |r: &RenderResult| match &r.data {
            FrameData::Ir { bt_kelvin, rgb } => {
                assert_eq!(bt_kelvin.len(), 20 * 20);
                assert!(
                    bt_kelvin
                        .iter()
                        .filter(|v| v.is_finite())
                        .all(|&v| (180.0..=320.0).contains(&v)),
                    "WV BT out of Kelvin range"
                );
                assert_eq!(rgb.as_ref().unwrap().len(), 20 * 20 * 3);
                bt_kelvin[(20 / 2) * 20 + 20 / 2] as f64
            }
            other => panic!("expected Ir data, got {other:?}"),
        };
        let (cd, cm) = (centre(&dry), centre(&moist));
        assert!(
            cm < cd - 3.0,
            "moist 6.2 BT {cm} should be colder than dry {cd}"
        );
        assert!(cm < 260.0, "moist 6.2 BT {cm} should be cold-troposphere");
        assert_eq!(moist.sun_elev_deg, 0.0);
        assert_eq!(moist.georef.extent_kind, ExtentKind::ProjectionMeters);
    }

    #[test]
    fn georef_mesh_round_trips_against_the_grid_georef() {
        // The top-down georef's lat/lon mesh must forward (through the SAME GridGeoref)
        // back onto the grid indices the map raster sampled: corner (row 0, col 0) = the
        // NW domain corner (i=0, j=ny-1), the SE (last row, last col) = (i=nx-1, j=0).
        let (nx, ny) = (24usize, 18usize);
        let src = synthetic_source(nx, ny, 16, 3000.0);
        let p = synthetic_params(ViewMode::TopDownMap);
        let res = render_visible_scene(&src, &p, Product::VisibleRgb).unwrap();
        let g = &res.georef;
        let check = |px: usize, py: usize, want_i: f64, want_j: f64| {
            let idx = py * g.nx + px;
            let (fi, fj) = src.georef.forward(g.lat[idx] as f64, g.lon[idx] as f64);
            assert!(
                (fi - want_i).abs() < 0.02 && (fj - want_j).abs() < 0.02,
                "mesh ({px},{py}) -> grid ({fi:.3},{fj:.3}) != ({want_i},{want_j})"
            );
        };
        // Native map raster (out dims == domain dims): grid_i == px, grid_j == ny-1-py.
        check(0, 0, 0.0, (ny - 1) as f64); // NW: row 0 is north (max j), col 0 is west (i=0)
        check(nx - 1, ny - 1, (nx - 1) as f64, 0.0); // SE
        check(nx / 2, ny / 2, (nx / 2) as f64, (ny - 1 - ny / 2) as f64); // interior sample
    }

    #[test]
    fn georef_proj4_matches_the_extent_frame() {
        let src = synthetic_source(20, 20, 12, 3000.0);
        // Top-down: a Lambert (lcc) proj4 with the pole as lat_0 (so PROJ x/y reproduce
        // SimSat's plane coords the extent is computed from) on the spherical earth.
        let td = render_visible_scene(
            &src,
            &synthetic_params(ViewMode::TopDownMap),
            Product::VisibleRgb,
        )
        .unwrap();
        let s = td.georef.proj4();
        assert!(s.starts_with("+proj=lcc"), "{s}");
        assert!(
            s.contains("+lat_0=90"),
            "northern cone should pin lat_0 at the pole: {s}"
        );
        assert!(
            s.contains("+lon_0=-97.5") && s.contains("+R=6370000"),
            "{s}"
        );
        assert_eq!(td.georef.proj_kind(), "lcc");
        // Geostationary: extent is lon/lat, so the CRS is plain longlat (PlateCarree).
        let geo = render_visible_scene(
            &src,
            &synthetic_params(ViewMode::Geostationary),
            Product::VisibleRgb,
        )
        .unwrap();
        assert!(
            geo.georef.proj4().starts_with("+proj=longlat"),
            "{}",
            geo.georef.proj4()
        );
    }

    #[test]
    fn topdown_and_geo_produce_the_expected_raster_shapes() {
        // Top-down native map raster == the WRF grid; the geostationary scan raster is a
        // (different) camera raster that renders successfully and carries a lonlat extent.
        let src = synthetic_source(28, 22, 16, 3000.0);
        let td = render_visible_scene(
            &src,
            &synthetic_params(ViewMode::TopDownMap),
            Product::VisibleRgb,
        )
        .unwrap();
        assert_eq!((td.nx, td.ny), (28, 22));
        assert_eq!(td.georef.extent_kind, ExtentKind::ProjectionMeters);

        let geo = render_visible_scene(
            &src,
            &synthetic_params(ViewMode::Geostationary),
            Product::VisibleRgb,
        )
        .unwrap();
        assert!(geo.nx > 0 && geo.ny > 0);
        assert_eq!(geo.georef.extent_kind, ExtentKind::LonLatDegrees);
        // The geostationary lonlat extent brackets the domain centre longitude.
        let [lo0, lo1, _la0, _la1] = geo.georef.extent;
        assert!(
            lo0 <= -97.5 && lo1 >= -97.5,
            "geo extent {lo0}..{lo1} misses -97.5"
        );
        // The geostationary frame also computes cloud stats (top-down does not).
        assert!(geo.cloud_stats.is_some() && td.cloud_stats.is_none());
    }

    #[test]
    fn geocolor_daytime_equals_the_visible_rgb() {
        // A synthetic all-daylight scene (sun_override 45 deg): GeoColor is 100% the refined
        // true-color visible frame (the day side), byte-for-byte. The whole small domain is
        // well above the +5 deg day threshold, so every pixel's blend weight is exactly 1.
        let src = synthetic_source(24, 24, 24, 3000.0);
        let p = synthetic_params(ViewMode::TopDownMap); // sun_override elev 45 -> full day
        let vis = render_visible_scene(&src, &p, Product::VisibleRgb).unwrap();
        let gc = render_geocolor_scene(&src, &p).unwrap();
        let vis_rgb = match &vis.data {
            FrameData::Visible { rgb, .. } => rgb,
            other => panic!("expected Visible, got {other:?}"),
        };
        let gc_rgb = match &gc.data {
            FrameData::Visible { rgb, rgba } => {
                assert_eq!(rgb.len(), 24 * 24 * 3);
                assert_eq!(rgba.len(), 24 * 24 * 4);
                rgb
            }
            other => panic!("expected Visible, got {other:?}"),
        };
        assert_eq!((gc.nx, gc.ny), (24, 24));
        assert_eq!(
            gc_rgb, vis_rgb,
            "daytime GeoColor must equal the visible RGB"
        );
        assert!(
            gc_rgb.iter().copied().max().unwrap() > 0,
            "the lit daytime surface should not be black"
        );
        // Sun elevation is reported (the day side uses the sun); the georef is the visible's.
        assert!(gc.sun_elev_deg > 5.0);
        assert_eq!(gc.georef.extent_kind, ExtentKind::ProjectionMeters);
    }

    #[test]
    fn geocolor_night_equals_the_ir_enhanced_rgb() {
        // A synthetic all-night scene (sun_override -30 deg): GeoColor is 100% the colored
        // band-13 IR (the night side), byte-for-byte with the GeoColor night enhancement —
        // so a scene that is BLACK in plain visible shows the storm/ground in IR. Every pixel
        // is well below the -6 deg night threshold, so every blend weight is exactly 0.
        let src = synthetic_source(20, 20, 32, 3000.0);
        let mut p = synthetic_params(ViewMode::TopDownMap);
        p.sun_override = Some(SunOverride {
            elev_deg: Some(-30.0),
            az_deg: Some(180.0),
        });
        // The reference IR frame through the SAME night enhancement.
        let mut ir_params = p.clone();
        ir_params.ir_enhancement = Some(geocolor::GEOCOLOR_NIGHT_ENHANCEMENT);
        let ir = render_ir_scene(&src, &ir_params, IrConfig::band13()).unwrap();
        let ir_rgb = match &ir.data {
            FrameData::Ir { rgb: Some(rgb), .. } => rgb,
            other => panic!("expected an enhanced IR frame, got {other:?}"),
        };
        let gc = render_geocolor_scene(&src, &p).unwrap();
        let gc_rgb = match &gc.data {
            FrameData::Visible { rgb, .. } => rgb,
            other => panic!("expected Visible, got {other:?}"),
        };
        assert_eq!(
            gc_rgb, ir_rgb,
            "night GeoColor must equal the IR-enhanced RGB"
        );
        assert!(
            gc_rgb.iter().copied().max().unwrap() > 0,
            "the night IR (warm ground on the grayscale ramp) should not be black"
        );
    }

    #[test]
    fn geocolor_render_dispatch_returns_the_visible_shape() {
        // The public `render(Product::GeoColor)` dispatches to the blend and returns a Visible
        // frame (a baked RGB composite) of the right shape + georef. A twilight-band sun
        // (0 deg) so the blend weight is strictly in (0, 1) across the domain (a real
        // crossfade, not a pinned endpoint).
        let src = synthetic_source(18, 18, 20, 3000.0);
        let mut p = synthetic_params(ViewMode::TopDownMap);
        p.sun_override = Some(SunOverride {
            elev_deg: Some(0.0),
            az_deg: Some(180.0),
        });
        let gc = render_geocolor_scene(&src, &p).unwrap();
        match &gc.data {
            FrameData::Visible { rgb, rgba } => {
                assert_eq!(rgb.len(), 18 * 18 * 3);
                assert_eq!(rgba.len(), 18 * 18 * 4);
                // On-earth pixels are opaque; every top-down pixel is on the domain.
                assert!(rgba.chunks_exact(4).all(|px| px[3] == 255));
            }
            other => panic!("expected Visible, got {other:?}"),
        }
        assert_eq!((gc.nx, gc.ny), (18, 18));
        assert_eq!(gc.georef.lat.len(), 18 * 18);
        assert_eq!(gc.georef.lon.len(), 18 * 18);
        assert_eq!(gc.georef.extent_kind, ExtentKind::ProjectionMeters);
        // The reported centre elevation is the twilight sun (0 deg).
        assert!(
            gc.sun_elev_deg.abs() < 1.0,
            "sun {} not ~0",
            gc.sun_elev_deg
        );
    }

    // ── Sandwich composite (visible base + colored IR on the cold tops) ──────────

    /// The synthetic source with a THICK, HIGH ice cloud baked into the top of the column, so
    /// the band-13 IR reads a COLD cloud-top brightness temperature (well below the sandwich
    /// warm threshold) and the visible shows a bright cloud — a cold-top convection scene.
    fn cold_top_source(nx: usize, ny: usize, nz: usize, dx: f64) -> SceneSource {
        let mut src = synthetic_source(nx, ny, nz, dx);
        let n2 = nx * ny;
        // A thick ice cloud over the whole domain in the TOP quarter of the column (high +
        // cold). 0.1 1/m over several 250 m layers is IR-opaque, so the BT reads the cold
        // cloud-top temperature (the top layers are ~210-225 K on the lapse profile).
        let mut ext = vec![0f32; nx * ny * nz];
        let top_start = nz * 3 / 4;
        for k in top_start..nz {
            ext[k * n2..(k + 1) * n2].fill(0.1);
        }
        let (lq, codes) = crate::bricks::encode_log_channel(&ext);
        src.brick.ext_ice = codes;
        src.brick.quant.0.insert("ext_ice".to_string(), lq);
        src
    }

    #[test]
    fn sandwich_warm_daytime_equals_the_visible_rgb() {
        // A clear, warm, all-daylight scene (BT ~ the warm surface, above the 260 K warm
        // threshold everywhere): the sandwich alpha is 0 for every pixel, so the composite is
        // the refined true-color visible frame byte-for-byte (no cold tops -> no IR overlay).
        let src = synthetic_source(24, 24, 24, 3000.0);
        let p = synthetic_params(ViewMode::TopDownMap); // sun_override 45 -> full daylight
        let vis = render_visible_scene(&src, &p, Product::VisibleRgb).unwrap();
        let sw = render_sandwich_scene(&src, &p).unwrap();
        let vis_rgb = match &vis.data {
            FrameData::Visible { rgb, .. } => rgb,
            other => panic!("expected Visible, got {other:?}"),
        };
        match &sw.data {
            FrameData::Visible { rgb, rgba } => {
                assert_eq!(rgb.len(), 24 * 24 * 3);
                assert_eq!(rgba.len(), 24 * 24 * 4);
                assert_eq!(
                    rgb, vis_rgb,
                    "warm sandwich (no cold tops) must equal the visible RGB"
                );
                assert!(
                    rgb.iter().copied().max().unwrap() > 0,
                    "the lit daytime surface should not be black"
                );
            }
            other => panic!("expected Visible, got {other:?}"),
        }
        assert_eq!((sw.nx, sw.ny), (24, 24));
        assert!(sw.sun_elev_deg > 5.0);
        assert_eq!(sw.georef.extent_kind, ExtentKind::ProjectionMeters);
    }

    #[test]
    fn sandwich_cold_top_overlays_ir_color_but_keeps_the_visible() {
        // A cold-top scene (a thick high ice cloud): the band-13 IR reads a COLD BT at the
        // cloud, so the sandwich OVERLAYS the colored IR there — the centre pixel differs from
        // the plain visible (the overlay is applied) AND is a genuine BLEND of the visible base
        // and the IR color (the visible still contributes, alpha < 1).
        let (nx, ny, nz) = (16usize, 16usize, 48usize);
        let src = cold_top_source(nx, ny, nz, 3000.0);
        let p = synthetic_params(ViewMode::TopDownMap); // daylight -> a meaningful visible base
        let vis = render_visible_scene(&src, &p, Product::VisibleRgb).unwrap();
        // The reference IR frame through the SAME sandwich enhancement (BT + colored RGB).
        let mut ir_params = p.clone();
        ir_params.ir_enhancement = Some(sandwich::SANDWICH_ENHANCEMENT);
        let ir = render_ir_scene(&src, &ir_params, IrConfig::band13()).unwrap();
        let sw = render_sandwich_scene(&src, &p).unwrap();

        let vis_rgb = match &vis.data {
            FrameData::Visible { rgb, .. } => rgb,
            other => panic!("expected Visible, got {other:?}"),
        };
        let (bt, ir_rgb) = match &ir.data {
            FrameData::Ir {
                bt_kelvin,
                rgb: Some(rgb),
            } => (bt_kelvin, rgb),
            other => panic!("expected an enhanced IR frame, got {other:?}"),
        };
        let sw_rgb = match &sw.data {
            FrameData::Visible { rgb, .. } => rgb,
            other => panic!("expected Visible, got {other:?}"),
        };
        assert_eq!((sw.nx, sw.ny), (nx, ny));

        let c = (ny / 2) * nx + nx / 2; // a central (cloudy) pixel
        // The IR at the cloud is cold (below the 260 K warm threshold), so the overlay applies.
        assert!(
            bt[c].is_finite() && (bt[c] as f64) < 250.0,
            "cold-top BT {} should be cold (< 250 K)",
            bt[c]
        );
        let v = [vis_rgb[c * 3], vis_rgb[c * 3 + 1], vis_rgb[c * 3 + 2]];
        let ircol = [ir_rgb[c * 3], ir_rgb[c * 3 + 1], ir_rgb[c * 3 + 2]];
        let s = [sw_rgb[c * 3], sw_rgb[c * 3 + 1], sw_rgb[c * 3 + 2]];
        // The overlay actually changed the pixel (it is not the plain visible) ...
        assert_ne!(s, v, "the cold-top overlay was not applied (== visible)");
        // ... but the visible still contributes (it is not the pure IR color either).
        assert_ne!(s, ircol, "the sandwich lost the visible base (== pure IR)");
        // Each channel is a blend of the visible base and the IR color (within [min,max]+/-1
        // for rounding), and at least one channel is STRICTLY interior (a real partial blend).
        let mut strict_interior = false;
        for ch in 0..3 {
            let (lo, hi) = (v[ch].min(ircol[ch]), v[ch].max(ircol[ch]));
            assert!(
                s[ch] as i16 >= lo as i16 - 1 && s[ch] as i16 <= hi as i16 + 1,
                "channel {ch}: {} not between visible {} and IR {}",
                s[ch],
                v[ch],
                ircol[ch]
            );
            if s[ch] > lo && s[ch] < hi {
                strict_interior = true;
            }
        }
        assert!(
            strict_interior,
            "no channel is a partial blend (visible {v:?}, IR {ircol:?}, sandwich {s:?})"
        );
    }

    #[test]
    fn sandwich_render_dispatch_returns_the_visible_shape() {
        // The public `render(Product::Sandwich)` path dispatches to the composite and returns a
        // Visible frame (a baked RGB composite) of the right shape + georef, with every
        // on-earth (top-down) pixel opaque.
        let src = cold_top_source(18, 18, 40, 3000.0);
        let p = synthetic_params(ViewMode::TopDownMap);
        let sw = render_sandwich_scene(&src, &p).unwrap();
        match &sw.data {
            FrameData::Visible { rgb, rgba } => {
                assert_eq!(rgb.len(), 18 * 18 * 3);
                assert_eq!(rgba.len(), 18 * 18 * 4);
                assert!(rgba.chunks_exact(4).all(|px| px[3] == 255));
            }
            other => panic!("expected Visible, got {other:?}"),
        }
        assert_eq!((sw.nx, sw.ny), (18, 18));
        assert_eq!(sw.georef.lat.len(), 18 * 18);
        assert_eq!(sw.georef.lon.len(), 18 * 18);
        assert_eq!(sw.georef.extent_kind, ExtentKind::ProjectionMeters);
    }

    // ── derived scalar-field maps (per-column integrals, map-registered) ─────────

    #[test]
    fn derived_precipitable_water_shape_range_and_moist_gt_dry() {
        // Product::Derived{PW} on a moist vs dry exponential column: the right Scalar shape, a
        // physically-plausible mm range, and a MOISTER column reads a LARGER PW. Derived is
        // thermal-like (no sun, projection georef, no cloud stats).
        let p = synthetic_params(ViewMode::TopDownMap);
        let field = DerivedField::PrecipitableWater;
        let moist =
            render_derived_scene(&moist_source(20, 20, 40, 3000.0, 0.014), &p, field).unwrap();
        let dry =
            render_derived_scene(&moist_source(20, 20, 40, 3000.0, 0.002), &p, field).unwrap();
        assert_eq!((moist.nx, moist.ny), (20, 20));
        assert_eq!(moist.sun_elev_deg, 0.0);
        assert_eq!(moist.georef.extent_kind, ExtentKind::ProjectionMeters);
        assert!(moist.cloud_stats.is_none());
        let center = |r: &RenderResult| match &r.data {
            FrameData::Scalar {
                values,
                rgb,
                field: f,
            } => {
                assert_eq!(values.len(), 20 * 20);
                assert_eq!(*f, DerivedField::PrecipitableWater);
                assert!(rgb.is_none(), "no colormap unless requested");
                assert!(values.iter().all(|v| v.is_finite() && *v >= 0.0));
                values[(20 / 2) * 20 + 20 / 2] as f64
            }
            other => panic!("expected Scalar, got {other:?}"),
        };
        let (cm, cd) = (center(&moist), center(&dry));
        assert!(cm > cd + 3.0, "moist PW {cm} not > dry PW {cd}");
        assert!(
            (5.0..60.0).contains(&cm),
            "moist PW {cm} not a plausible mm value"
        );
    }

    #[test]
    fn derived_cloud_top_temp_cold_over_cloud_and_nan_clear() {
        // A thick high ice cloud reads a COLD cloud-top temperature at the covered pixels; a
        // clear column reads NaN (no cloud top).
        let p = synthetic_params(ViewMode::TopDownMap);
        let field = DerivedField::CloudTopTemp;
        let cloudy = render_derived_scene(&cold_top_source(16, 16, 48, 3000.0), &p, field).unwrap();
        let clear = render_derived_scene(&synthetic_source(16, 16, 48, 3000.0), &p, field).unwrap();
        match &cloudy.data {
            FrameData::Scalar {
                values, field: f, ..
            } => {
                assert_eq!(*f, DerivedField::CloudTopTemp);
                let c = values[(16 / 2) * 16 + 16 / 2];
                assert!(
                    c.is_finite() && (c as f64) < 250.0,
                    "cloud-top T {c} not cold"
                );
                assert!((c as f64) > 190.0, "cloud-top T {c} implausibly cold");
            }
            other => panic!("expected Scalar, got {other:?}"),
        }
        match &clear.data {
            FrameData::Scalar { values, .. } => {
                assert!(
                    values.iter().all(|v| v.is_nan()),
                    "clear cloud-top must be all NaN"
                );
            }
            other => panic!("expected Scalar, got {other:?}"),
        }
    }

    #[test]
    fn derived_cloud_optical_depth_clear_zero_storm_positive() {
        // Clear column -> COD 0; a thick storm column -> a large positive optical depth.
        let p = synthetic_params(ViewMode::TopDownMap);
        let field = DerivedField::CloudOpticalDepth;
        let clear = render_derived_scene(&synthetic_source(16, 16, 24, 3000.0), &p, field).unwrap();
        let storm_source = cold_top_source(16, 16, 48, 3000.0);
        let storm = render_derived_scene(&storm_source, &p, field).unwrap();
        let vals = |r: &RenderResult| match &r.data {
            FrameData::Scalar { values, .. } => values.clone(),
            other => panic!("expected Scalar, got {other:?}"),
        };
        assert!(
            vals(&clear).iter().all(|&v| v == 0.0),
            "clear COD must be 0"
        );
        let sv = vals(&storm);
        assert!(sv[(16 / 2) * 16 + 16 / 2] > 5.0, "storm COD not thick");

        // The visible what-if scale is applied only by visible cloud consumers. The
        // quantitative derived COD must remain byte-for-byte physical/unscaled.
        let mut scaled_params = p;
        scaled_params.cloud_optical_depth_scale = 1.0;
        let scaled = render_derived_scene(&storm_source, &scaled_params, field).unwrap();
        assert_eq!(
            sv,
            vals(&scaled),
            "visible cloud scale altered raw derived COD"
        );
    }

    #[test]
    fn derived_colormap_toggle_and_geo_view() {
        // derived_colormap on -> a colormap RGB of the right shape; the geo view renders + a
        // lonlat extent (the secondary product path).
        let field = DerivedField::PrecipitableWater;
        let mut p = synthetic_params(ViewMode::TopDownMap);
        p.derived_colormap = true;
        let r = render_derived_scene(&moist_source(12, 12, 30, 3000.0, 0.012), &p, field).unwrap();
        match &r.data {
            FrameData::Scalar {
                rgb: Some(rgb),
                values,
                ..
            } => assert_eq!(rgb.len(), values.len() * 3),
            other => panic!("expected Scalar with rgb, got {other:?}"),
        }
        let pg = synthetic_params(ViewMode::Geostationary);
        let geo =
            render_derived_scene(&moist_source(12, 12, 30, 3000.0, 0.012), &pg, field).unwrap();
        assert!(geo.nx > 0 && geo.ny > 0);
        assert_eq!(geo.georef.extent_kind, ExtentKind::LonLatDegrees);
        match &geo.data {
            FrameData::Scalar { values, .. } => {
                assert!(values.iter().any(|v| v.is_finite()));
            }
            other => panic!("expected Scalar, got {other:?}"),
        }
    }

    // ── zoom-out / domain-margin feature ────────────────────────────────────────

    #[test]
    fn margin_zero_is_identity_and_margin_extends_both_views() {
        let src = synthetic_source(40, 30, 16, 3000.0);

        // Top-down: margin 0.0 reproduces the native domain dims; margin 0.3 grows both
        // axes AND widens the projection-metres imshow extent (surrounding earth added).
        let td0 = render_visible_scene(
            &src,
            &synthetic_params(ViewMode::TopDownMap),
            Product::VisibleRgb,
        )
        .unwrap();
        assert_eq!(
            (td0.nx, td0.ny),
            (40, 30),
            "margin 0 top-down = native dims"
        );
        let mut pm = synthetic_params(ViewMode::TopDownMap);
        pm.margin_frac = 0.3;
        let td = render_visible_scene(&src, &pm, Product::VisibleRgb).unwrap();
        assert!(
            td.nx > 40 && td.ny > 30,
            "top-down margin raster not extended: {}x{}",
            td.nx,
            td.ny
        );
        let span0 = td0.georef.extent[1] - td0.georef.extent[0];
        let spanm = td.georef.extent[1] - td.georef.extent[0];
        assert!(
            spanm > span0 * 1.2,
            "top-down imshow extent not widened by the margin: {spanm} vs {span0}"
        );
        // Every top-down pixel (domain AND margin) is opaque real earth; the NW margin
        // corner carries a finite lat/lon (on earth) — no black frame around the domain.
        match &td.data {
            FrameData::Visible { rgba, .. } => {
                assert!(rgba.chunks_exact(4).all(|px| px[3] == 255));
            }
            other => panic!("expected Visible, got {other:?}"),
        }
        assert!(
            td.georef.lat[0].is_finite() && td.georef.lon[0].is_finite(),
            "top-down margin corner should be on earth"
        );

        // Geostationary: margin 0.3 grows the scan raster AND widens the lon/lat extent.
        let geo0 = render_visible_scene(
            &src,
            &synthetic_params(ViewMode::Geostationary),
            Product::VisibleRgb,
        )
        .unwrap();
        let mut pg = synthetic_params(ViewMode::Geostationary);
        pg.margin_frac = 0.3;
        let geo = render_visible_scene(&src, &pg, Product::VisibleRgb).unwrap();
        assert!(
            geo.nx > geo0.nx && geo.ny > geo0.ny,
            "geostationary margin raster not extended: {}x{} vs {}x{}",
            geo.nx,
            geo.ny,
            geo0.nx,
            geo0.ny
        );
        let [glo0, glo1, gla0, gla1] = geo0.georef.extent;
        let [mlo0, mlo1, mla0, mla1] = geo.georef.extent;
        assert!(
            mlo0 < glo0 && mlo1 > glo1 && mla0 < gla0 && mla1 > gla1,
            "geostationary lon/lat extent not widened by the margin"
        );
    }

    // ── WS3: cache/manifest honesty (time fallback, anchor, ground source) ───────

    use crate::bricks::{ManifestAnchor, ManifestProjection, ManifestTimestep};
    use std::path::Path;

    fn api_temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("simsat-api-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Write a synthetic scene into a real on-disk cached run (brick + run.json) so the
    /// public `render` entry exercises `resolve_cached` end to end.
    fn write_cached_run(
        cache: &Path,
        run_id: &str,
        src: &SceneSource,
        time_iso: Option<String>,
        anchor: Option<ManifestAnchor>,
    ) -> PathBuf {
        let brick = &src.brick;
        let stamp = bricks::time_stamp(time_iso.as_deref(), 0);
        let file = bricks::brick_file_name(&stamp);
        let brick_path = bricks::run_dir(cache, run_id).join(&file);
        bricks::write_ssb(&brick_path, brick).unwrap();
        let p = &src.params;
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
        let manifest_path = RunManifest::path(cache, run_id);
        let mut manifest = RunManifest::load_or_new(
            &manifest_path,
            run_id,
            brick.nx,
            brick.ny,
            brick.nz,
            brick.z_min_m,
            brick.dz_m,
            brick.planes_2d_names(),
            projection,
        )
        .unwrap();
        manifest.register_timestep(ManifestTimestep {
            key: stamp,
            hhmm: 0,
            file,
            time_iso,
            quant: brick.quant.clone(),
            has_cloud_fraction: brick.has_cloud_fraction,
            ssb_bytes: 1,
            source_bytes: None,
            source_mtime_unix: None,
            anchor,
        });
        manifest.save(&manifest_path).unwrap();
        manifest_path
    }

    #[test]
    fn cached_run_with_corrupt_time_iso_flags_the_time_fallback() {
        // A cached run whose manifest time_iso is garbage: the render still succeeds on
        // the FIXED fallback date (2004-06-21, unchanged) but the fabrication is now
        // FLAGGED; a valid time parses and is NOT flagged (the control).
        let dir = api_temp_dir("timefallback");
        let src = synthetic_source(12, 12, 12, 3000.0);
        let bad = write_cached_run(&dir, "runbad", &src, Some("not a time".to_string()), None);
        let good = write_cached_run(
            &dir,
            "rungood",
            &src,
            Some("2025-06-21T02:15:00Z".to_string()),
            None,
        );

        let mut p = RenderParams::new(bad);
        p.view = ViewMode::TopDownMap;
        let r = render(&p, Product::Ir).expect("corrupt time must still render");
        assert!(r.time_is_fallback, "corrupt time_iso must set the flag");
        assert_eq!(
            (r.time.year, r.time.month, r.time.day),
            (2004, 6, 21),
            "the fallback date itself is unchanged"
        );

        let mut p2 = RenderParams::new(good);
        p2.view = ViewMode::TopDownMap;
        let r2 = render(&p2, Product::Ir).expect("valid time renders");
        assert!(!r2.time_is_fallback);
        assert_eq!((r2.time.year, r2.time.month, r2.time.day), (2025, 6, 21));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn cached_run_reconstructs_the_georef_from_the_persisted_anchor() {
        // Two cached runs of the SAME brick, one without an anchor (the pre-WS3
        // CEN_LAT/CEN_LON fallback) and one whose persisted anchor is deliberately
        // shifted north by 0.1 deg: the rendered top-down extents must differ — proof
        // that resolve_cached reads the anchor rather than re-deriving from the centre
        // attributes. (Bit-identity of from_anchor itself is proven in frame.rs.)
        let dir = api_temp_dir("anchor");
        let src = synthetic_source(12, 12, 12, 3000.0);
        let (nx, ny) = (src.brick.nx, src.brick.ny);
        let plain = write_cached_run(&dir, "runa", &src, None, None);
        let shifted_anchor = ManifestAnchor {
            ref_i: (nx - 1) as f64 / 2.0,
            ref_j: (ny - 1) as f64 / 2.0,
            ref_lat_deg: src.params.cen_lat_deg + 0.1,
            ref_lon_deg: src.params.cen_lon_deg,
            dx: src.params.dx_m,
            dy: src.params.dy_m,
        };
        let anchored = write_cached_run(&dir, "runb", &src, None, Some(shifted_anchor));

        let render_extent = |input: PathBuf| {
            let mut p = RenderParams::new(input);
            p.view = ViewMode::TopDownMap;
            render(&p, Product::Ir).expect("render").georef.extent
        };
        let e_plain = render_extent(plain);
        let e_anchored = render_extent(anchored);
        assert!(
            (e_plain[2] - e_anchored[2]).abs() > 1.0,
            "a shifted persisted anchor must move the projection-plane extent: \
             {e_plain:?} vs {e_anchored:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn single_file_bluemarble_that_fails_to_load_is_a_hard_error() {
        // The user explicitly named a ground-texture file; failing to load it must be
        // a hard error naming the path (BEHAVIOR CHANGE: it used to silently render
        // flat albedo).
        let src = synthetic_source(16, 16, 12, 3000.0);
        let mut p = synthetic_params(ViewMode::TopDownMap);
        let missing = std::env::temp_dir().join("simsat-ws3-definitely-missing-bm.jpg");
        p.bluemarble = BlueMarble::SingleFile(missing.clone());
        let err = render_visible_scene(&src, &p, Product::VisibleRgb).unwrap_err();
        assert!(
            err.contains("blue marble file"),
            "error should name the ground file: {err}"
        );
    }

    #[test]
    fn offline_seasonal_bluemarble_reports_the_8k_fallback() {
        // Seasonal with downloads OFF and an empty cache: the render succeeds on the
        // vendored 8 km fallback (the never-hard-fail contract) and the downgrade is
        // REPORTED on the result instead of being swallowed.
        let dir = api_temp_dir("season8k");
        let src = synthetic_source(16, 16, 12, 3000.0);
        let mut p = synthetic_params(ViewMode::TopDownMap);
        p.cache = dir.clone();
        p.bluemarble = BlueMarble::Seasonal {
            month_override: Some(6),
            download: false,
        };
        let r = render_visible_scene(&src, &p, Product::VisibleRgb).expect("offline render");
        assert_eq!(r.ground_source, Some(GroundSource::EightKmFallback));
        assert!(
            r.ground_status.iter().any(|l| l.contains("8km")),
            "the status lines should say the 8 km fallback was used: {:?}",
            r.ground_status
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn flat_albedo_ground_and_missing_time_are_reported_not_silent() {
        // The synthetic scene (no time, flat albedo requested): both honesty fields
        // are populated — the ground source says flat albedo WITH the reason, and the
        // absent valid time flags the fabricated sun date.
        let src = synthetic_source(16, 16, 12, 3000.0);
        let p = synthetic_params(ViewMode::TopDownMap); // FlatAlbedo + no time_iso
        let r = render_visible_scene(&src, &p, Product::VisibleRgb).unwrap();
        match &r.ground_source {
            Some(GroundSource::FlatAlbedo(reason)) => {
                assert!(reason.contains("requested"), "reason: {reason}");
            }
            other => panic!("expected FlatAlbedo, got {other:?}"),
        }
        assert!(r.ground_status.is_empty());
        assert!(
            r.time_is_fallback,
            "a source with no valid time must flag the fabricated date"
        );
    }

    // ── sub-grid cloud granulation (edge-erosion detail noise) ───────────────────

    /// The synthetic source with a scattered BOUNDARY-LAYER LIQUID popcorn-cu field
    /// (single cloudy cells, k 2..7 ~ 0.5-1.75 km) — the granulation target regime.
    fn low_liquid_source(nx: usize, ny: usize, nz: usize, dx: f64) -> SceneSource {
        let mut src = synthetic_source(nx, ny, nz, dx);
        let n2 = nx * ny;
        let mut ext = vec![0f32; nx * ny * nz];
        for k in 2..7.min(nz) {
            for j in 0..ny {
                for i in 0..nx {
                    if i % 4 == 1 && j % 3 == 1 {
                        ext[k * n2 + j * nx + i] = 1.5e-2;
                    }
                }
            }
        }
        let (lq, codes) = crate::bricks::encode_log_channel(&ext);
        src.brick.ext_liquid = codes;
        src.brick.quant.0.insert("ext_liquid".to_string(), lq);
        src
    }

    /// A spatially coherent, optically moderate liquid layer for the beer-powder
    /// control. Its sun optical depths sit in the regime where beer-powder differs
    /// materially from pure Beer (unlike a saturated storm core).
    fn moderate_liquid_source(nx: usize, ny: usize, nz: usize, dx: f64) -> SceneSource {
        let mut src = synthetic_source(nx, ny, nz, dx);
        let n2 = nx * ny;
        let mut ext = vec![0f32; nx * ny * nz];
        for k in 2..5.min(nz) {
            ext[k * n2..(k + 1) * n2].fill(8.0e-4);
        }
        let (lq, codes) = crate::bricks::encode_log_channel(&ext);
        src.brick.ext_liquid = codes;
        src.brick.quant.0.insert("ext_liquid".to_string(), lq);
        src
    }

    /// A model-fraction-aware cloud deck: the condensate is grid-mean and each cloudy
    /// layer covers exactly 20% of the cell (code 51). This gives the public visible
    /// path a deterministic on/off target without any file or network dependency.
    fn fractional_liquid_source(nx: usize, ny: usize, nz: usize, dx: f64) -> SceneSource {
        let mut src = moderate_liquid_source(nx, ny, nz, dx);
        let n2 = nx * ny;
        src.brick.cloud_fraction.fill(0);
        for k in 2..5.min(nz) {
            src.brick.cloud_fraction[k * n2..(k + 1) * n2].fill(51);
        }
        src.brick.has_cloud_fraction = true;
        src
    }

    #[test]
    fn fractional_cloud_switch_changes_visible_only_and_missing_field_falls_back_exactly() {
        let src = fractional_liquid_source(10, 10, 10, 3000.0);
        let mut on_params = synthetic_params(ViewMode::TopDownMap);
        on_params.steps = StepQuality::Interactive;
        on_params.fractional_clouds = true;
        let mut off_params = on_params.clone();
        off_params.fractional_clouds = false;

        let on = render_visible_scene(&src, &on_params, Product::VisibleBands).unwrap();
        let off = render_visible_scene(&src, &off_params, Product::VisibleBands).unwrap();
        let band_bits = |r: &RenderResult| match &r.data {
            FrameData::Bands { reflectance } => {
                reflectance.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            }
            other => panic!("expected Bands, got {other:?}"),
        };
        assert_ne!(
            band_bits(&on),
            band_bits(&off),
            "model cloud fraction must reach the visible march"
        );

        let ir_on = render_ir_scene(&src, &on_params, IrConfig::band13()).unwrap();
        let ir_off = render_ir_scene(&src, &off_params, IrConfig::band13()).unwrap();
        let bt_bits = |r: &RenderResult| match &r.data {
            FrameData::Ir { bt_kelvin, .. } => {
                bt_kelvin.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            }
            other => panic!("expected Ir, got {other:?}"),
        };
        assert_eq!(bt_bits(&ir_on), bt_bits(&ir_off));

        let field = DerivedField::CloudOpticalDepth;
        let derived_on = render_derived_scene(&src, &on_params, field).unwrap();
        let derived_off = render_derived_scene(&src, &off_params, field).unwrap();
        let scalar_bits = |r: &RenderResult| match &r.data {
            FrameData::Scalar { values, .. } => {
                values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            }
            other => panic!("expected Scalar, got {other:?}"),
        };
        assert_eq!(scalar_bits(&derived_on), scalar_bits(&derived_off));

        // A source with no trusted coverage field is an exact legacy fallback even
        // though the physical switch defaults on.
        let fallback = moderate_liquid_source(10, 10, 10, 3000.0);
        let fallback_on =
            render_visible_scene(&fallback, &on_params, Product::VisibleBands).unwrap();
        let fallback_off =
            render_visible_scene(&fallback, &off_params, Product::VisibleBands).unwrap();
        assert_eq!(band_bits(&fallback_on), band_bits(&fallback_off));
    }

    #[test]
    fn beer_powder_render_param_reaches_the_visible_cloud_march() {
        let src = moderate_liquid_source(12, 12, 12, 3000.0);
        let mut off_params = synthetic_params(ViewMode::TopDownMap);
        off_params.steps = StepQuality::Interactive;
        let mut on_params = off_params.clone();
        on_params.beer_powder = true;
        let off = render_visible_scene(&src, &off_params, Product::VisibleBands).unwrap();
        let on = render_visible_scene(&src, &on_params, Product::VisibleBands).unwrap();
        let center_sum = |r: &RenderResult| match &r.data {
            FrameData::Bands { reflectance } => {
                let o = ((r.ny / 2) * r.nx + r.nx / 2) * 3;
                reflectance[o..o + 3].iter().sum::<f32>()
            }
            other => panic!("expected Bands, got {other:?}"),
        };
        let (off_sum, on_sum) = (center_sum(&off), center_sum(&on));
        assert!(
            on_sum < off_sum - 1.0e-6,
            "beer-powder should darken the moderate cloud sun term: on={on_sum}, off={off_sum}"
        );
    }

    #[test]
    fn granulation_scopes_by_product_and_is_live_in_the_display_rgb() {
        // v0.1.1 OPT-IN scoping: granulation is OFF unless explicitly requested
        // (owner-rejected round-1 default look; tune-2 re-earns the default later),
        // and even when requested it is live ONLY for the display VisibleRgb —
        // quantitative bands / thermal / derived products never granulate.
        let src = low_liquid_source(24, 24, 16, 3000.0);
        let p_default = synthetic_params(ViewMode::TopDownMap); // granulation None
        let default_off = render_visible_scene(&src, &p_default, Product::VisibleRgb).unwrap();
        assert!(
            !default_off.granulation,
            "VisibleRgb must NOT granulate by default (opt-in as of v0.1.1)"
        );
        let mut p = p_default.clone();
        p.granulation = Some(true);
        let on = render_visible_scene(&src, &p, Product::VisibleRgb).unwrap();
        assert!(
            on.granulation,
            "an explicit opt-in must granulate the display product"
        );
        let mut p_off = p_default.clone();
        p_off.granulation = Some(false);
        let off = render_visible_scene(&src, &p_off, Product::VisibleRgb).unwrap();
        assert!(!off.granulation, "the explicit off must win");
        let (rgb_on, rgb_off) = match (&on.data, &off.data) {
            (FrameData::Visible { rgb: a, .. }, FrameData::Visible { rgb: b, .. }) => (a, b),
            other => panic!("expected Visible frames, got {other:?}"),
        };
        assert_ne!(
            rgb_on, rgb_off,
            "granulation should visibly change a coarse-grid liquid cu field"
        );

        let bands = render_visible_scene(&src, &p, Product::VisibleBands).unwrap();
        assert!(
            !bands.granulation,
            "the quantitative raw-reflectance bands must stay OFF even when requested"
        );
        let ir = render_ir_scene(&src, &p, IrConfig::band13()).unwrap();
        assert!(!ir.granulation, "raw-Kelvin IR never granulates");
        let derived = render_derived_scene(&src, &p, DerivedField::CloudOpticalDepth).unwrap();
        assert!(!derived.granulation, "derived fields never granulate");
    }

    #[test]
    fn ir_bt_is_byte_identical_with_granulation_requested() {
        // The raw-Kelvin contract: requesting granulation cannot touch a thermal
        // product — the BT plane is byte-identical whether the flag is forced on or
        // off (the IR march reads the un-eroded brick by construction), so
        // quantitative BT verification always reflects model skill.
        let src = low_liquid_source(20, 20, 16, 3000.0);
        let mut p_on = synthetic_params(ViewMode::TopDownMap);
        p_on.granulation = Some(true);
        let mut p_off = p_on.clone();
        p_off.granulation = Some(false);
        let r_on = render_ir_scene(&src, &p_on, IrConfig::band13()).unwrap();
        let r_off = render_ir_scene(&src, &p_off, IrConfig::band13()).unwrap();
        let bt = |r: &RenderResult| match &r.data {
            FrameData::Ir { bt_kelvin, .. } => bt_kelvin.clone(),
            other => panic!("expected Ir, got {other:?}"),
        };
        let (a, b) = (bt(&r_on), bt(&r_off));
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "IR BT must be byte-identical with granulation available"
            );
        }
        assert!(!r_on.granulation && !r_off.granulation);
    }

    #[test]
    fn margin_leaves_the_domain_centred_and_lit() {
        // With a margin the DOMAIN weather sits in the centre framed by clear surrounding
        // earth. A clear synthetic scene: the centre pixel is lit ground, and the margin
        // corner is ALSO lit ground (real earth, clear sky) — proving the margin renders the
        // surrounding earth rather than black/space.
        let src = synthetic_source(30, 30, 16, 3000.0);
        let mut p = synthetic_params(ViewMode::TopDownMap);
        p.margin_frac = 0.4;
        let res = render_visible_scene(&src, &p, Product::VisibleRgb).unwrap();
        let (nx, ny) = (res.nx, res.ny);
        assert!(nx > 30 && ny > 30);
        match &res.data {
            FrameData::Visible { rgb, .. } => {
                let lum = |px: usize, py: usize| {
                    let o = (py * nx + px) * 3;
                    rgb[o] as u32 + rgb[o + 1] as u32 + rgb[o + 2] as u32
                };
                assert!(lum(nx / 2, ny / 2) > 0, "domain centre should be lit");
                assert!(
                    lum(2, 2) > 0,
                    "margin corner should be lit real earth, not black"
                );
            }
            other => panic!("expected Visible, got {other:?}"),
        }
    }

    // ── web-map cloud layer (Product::CloudLayer) ──────────────────────────────

    #[test]
    fn cloud_layer_clear_scene_is_transparent_on_a_mercator_grid() {
        // A CLEAR brick through Product::CloudLayer: the delivered pair is on a
        // Web-Mercator grid (extent kind + EPSG:3857 proj4 + ordered Mapbox corners
        // bracketing the domain), the cloud image is EXACTLY transparent, and the
        // shadow layer exactly neutral — a host compositing it changes nothing.
        let src = synthetic_source(24, 24, 24, 3000.0);
        let p = synthetic_params(ViewMode::TopDownMap);
        let res = render_cloud_layer_scene(&src, &p).unwrap();
        let (nx, ny) = (res.nx, res.ny);
        assert!(nx >= 2 && ny >= 2);
        match &res.data {
            FrameData::CloudLayer {
                rgba_premul,
                shadow,
            } => {
                assert_eq!(rgba_premul.len(), nx * ny * 4);
                assert_eq!(shadow.len(), nx * ny);
                assert!(
                    rgba_premul.iter().all(|&v| v == 0),
                    "a clear scene must deliver a fully transparent cloud layer"
                );
                assert!(
                    shadow.iter().all(|&s| s == 1.0),
                    "a clear scene must deliver a neutral shadow layer"
                );
            }
            other => panic!("expected CloudLayer, got {other:?}"),
        }
        // The georef describes the Mercator delivery grid.
        let g = &res.georef;
        assert_eq!(g.extent_kind, ExtentKind::WebMercatorMeters);
        assert_eq!((g.nx, g.ny), (nx, ny));
        assert_eq!(g.lat.len(), nx * ny);
        assert!(g.proj4().contains("6378137"), "{}", g.proj4());
        let c = g.mercator_corners_lonlat.expect("mapbox corners");
        assert!(c[0][0] < c[1][0], "TL lon < TR lon: {c:?}");
        assert!(c[0][1] > c[3][1], "TL lat > BL lat: {c:?}");
        // The corners bracket the domain centre (39 N, -97.5 E).
        assert!(c[0][0] < -97.5 && c[1][0] > -97.5, "{c:?}");
        assert!(c[0][1] > 39.0 && c[3][1] < 39.0, "{c:?}");
        // The mesh matches the corners: row 0 is the northern edge.
        assert!(g.lat[0] > g.lat[(ny - 1) * nx] && g.lon[0] < g.lon[nx - 1]);
        assert!(res.ground_source.is_none(), "no ground is rendered");
    }

    #[test]
    fn cloud_layer_cloudy_scene_delivers_cloud_and_shadow() {
        // The cold-top source (a thick high ice cloud over the whole domain) through
        // Product::CloudLayer under a 45-deg sun: the delivered cloud image has real
        // alpha + lit color over the domain, and the shadow layer darkens the ground.
        let src = cold_top_source(24, 24, 24, 3000.0);
        let p = synthetic_params(ViewMode::TopDownMap);
        let res = render_cloud_layer_scene(&src, &p).unwrap();
        let (nx, ny) = (res.nx, res.ny);
        match &res.data {
            FrameData::CloudLayer {
                rgba_premul,
                shadow,
            } => {
                let o = ((ny / 2) * nx + nx / 2) * 4;
                assert!(
                    rgba_premul[o + 3] > 200,
                    "thick cloud centre alpha {} not near-opaque",
                    rgba_premul[o + 3]
                );
                assert!(
                    rgba_premul[o] > 0 && rgba_premul[o + 1] > 0 && rgba_premul[o + 2] > 0,
                    "sun-lit cloud should have positive color: {:?}",
                    &rgba_premul[o..o + 4]
                );
                assert!(
                    shadow[(ny / 2) * nx + nx / 2] < 0.95,
                    "a whole-domain thick cloud must shadow the ground: {}",
                    shadow[(ny / 2) * nx + nx / 2]
                );
                // The straight-alpha delivery conversion keeps alpha and stays in range.
                let straight = crate::web_layer::unpremultiply_rgba(rgba_premul);
                assert_eq!(straight.len(), rgba_premul.len());
                assert_eq!(straight[o + 3], rgba_premul[o + 3]);
            }
            other => panic!("expected CloudLayer, got {other:?}"),
        }
        assert!(
            res.sun_elev_deg > 40.0,
            "the 45-deg override drives the sun"
        );

        // The public clouds toggle has the same explicit-bypass meaning for this
        // layer product as it does for RGB/bands: transparent cloud + neutral shadow.
        let mut off_params = p;
        off_params.clouds = false;
        let off = render_cloud_layer_scene(&src, &off_params).unwrap();
        match &off.data {
            FrameData::CloudLayer {
                rgba_premul,
                shadow,
            } => {
                assert!(rgba_premul.iter().all(|&v| v == 0));
                assert!(shadow.iter().all(|&v| v == 1.0));
            }
            other => panic!("expected CloudLayer, got {other:?}"),
        }
        assert!(!off.granulation);
    }

    // ── free-perspective frame (Product::Perspective) ──────────────────────────

    /// A low-oblique camera south of the synthetic domain centre (39 N, -97.5 E),
    /// looking at it from 150 km altitude. ODD dims so the middle pixel is exactly
    /// the optical axis (the centre-mesh-hits-the-look-point assertion; even dims
    /// put the centre pixel half a pixel off axis ~1.6 km on the ground).
    fn oblique_camera() -> PerspectiveCamera {
        PerspectiveCamera {
            eye_lat_deg: 37.8,
            eye_lon_deg: -97.5,
            eye_alt_m: 150_000.0,
            look_lat_deg: 39.0,
            look_lon_deg: -97.5,
            look_alt_m: 0.0,
            fov_deg: 45.0,
            width: 65,
            height: 49,
        }
    }

    #[test]
    fn perspective_requires_the_camera_params() {
        let src = synthetic_source(20, 20, 16, 3000.0);
        let p = synthetic_params(ViewMode::TopDownMap); // no perspective set
        let err = render_perspective_scene(&src, &p, false).unwrap_err();
        assert!(
            err.contains("RenderParams::perspective"),
            "unhelpful error: {err}"
        );
    }

    #[test]
    fn perspective_full_composite_shape_pose_and_lit_ground() {
        let src = synthetic_source(24, 24, 24, 3000.0);
        let mut p = synthetic_params(ViewMode::TopDownMap);
        let cam = oblique_camera();
        p.perspective = Some(cam);
        let res = render_perspective_scene(&src, &p, false).unwrap();
        // The frame is CAMERA-sized (not domain-sized) and carries its pose.
        assert_eq!((res.nx, res.ny), (cam.width, cam.height));
        assert_eq!(
            res.georef.camera_pose,
            Some(cam),
            "the pose must ride along"
        );
        assert_eq!(res.georef.extent_kind, ExtentKind::LonLatDegrees);
        assert_eq!(res.georef.lat.len(), cam.width * cam.height);
        match &res.data {
            FrameData::Visible { rgb, rgba } => {
                assert_eq!(rgb.len(), cam.width * cam.height * 3);
                assert_eq!(rgba.len(), cam.width * cam.height * 4);
                // The centre pixel looks at the lit domain-centre ground: opaque + lit.
                let c = ((cam.height / 2) * cam.width + cam.width / 2) * 4;
                assert_eq!(rgba[c + 3], 255, "centre pixel should be on earth");
                assert!(
                    rgba[c] > 0 || rgba[c + 1] > 0 || rgba[c + 2] > 0,
                    "the 45-deg-sun ground should be lit"
                );
            }
            other => panic!("expected Visible, got {other:?}"),
        }
        // The centre pixel's mesh position is the look point (the camera aims there).
        let idx = (cam.height / 2) * cam.width + cam.width / 2;
        assert!(
            (res.georef.lat[idx] - 39.0).abs() < 1e-4 && (res.georef.lon[idx] - -97.5).abs() < 1e-4,
            "centre mesh ({}, {}) != the look point",
            res.georef.lat[idx],
            res.georef.lon[idx]
        );
    }

    #[test]
    fn perspective_cloud_layer_only_is_alpha_true_and_clear_transparent() {
        // The cold-top source (a thick high cloud over the domain): the layer-only
        // variant is premultiplied cloud RGBA — real alpha where rays cross the cloud.
        let mut p = synthetic_params(ViewMode::TopDownMap);
        p.perspective = Some(oblique_camera());
        let cloudy =
            render_perspective_scene(&cold_top_source(24, 24, 24, 3000.0), &p, true).unwrap();
        match &cloudy.data {
            FrameData::Visible { rgba, .. } => {
                let max_alpha = rgba.chunks_exact(4).map(|px| px[3]).max().unwrap();
                assert!(
                    max_alpha > 200,
                    "a thick cloud should be near-opaque in the layer: {max_alpha}"
                );
            }
            other => panic!("expected Visible (layer rgba), got {other:?}"),
        }
        assert!(
            cloudy.ground_source.is_none(),
            "layer-only renders no ground"
        );
        assert_eq!(cloudy.georef.camera_pose, Some(oblique_camera()));
        // A CLEAR scene through the same camera: exactly transparent everywhere.
        let clear =
            render_perspective_scene(&synthetic_source(24, 24, 24, 3000.0), &p, true).unwrap();
        match &clear.data {
            FrameData::Visible { rgba, .. } => {
                assert!(
                    rgba.iter().all(|&v| v == 0),
                    "a clear layer-only frame must be exactly transparent"
                );
            }
            other => panic!("expected Visible (layer rgba), got {other:?}"),
        }
    }
}
