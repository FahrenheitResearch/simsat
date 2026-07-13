//! `simsat` — the Python binding (`import simsat`).
//!
//! A THIN PyO3 + numpy wrapper over the pure-Rust [`simsat::api`] render layer. It parses
//! Python keyword arguments, calls [`simsat::api::render`] (releasing the GIL for the
//! CPU-heavy march), and marshals the returned owned Rust arrays into numpy. NO render
//! logic lives here — the physics + the tests are in the `simsat` crate.
//!
//! The functions each return `(numpy_array, georef)` so a meteorologist can plot the
//! result on a cartopy map:
//! - [`render_visible_rgb`]  -> `(H x W x 3 uint8, Georef)`  the finished true-color RGB.
//! - [`render_geocolor`] -> `(H x W x 3 uint8, Georef)`  SimSat Day/Night Color, a
//!   GeoColor-style composite (true-color by day, colored band-13 IR by night). It is not
//!   yet sensor-derived ABI GeoColor.
//! - [`render_sandwich`] -> `(H x W x 3 uint8, Georef)`  the Sandwich composite (visible
//!   true-color base + color-enhanced band-13 IR overlaid on the cold cloud tops) — the
//!   classic severe-convection view; a daytime-convection product.
//! - [`render_rgb_reflectance`] -> `(H x W x 3 float32, Georef)` RAW broad-RGB reflectance
//!   (pre-tonemap, `[0, 1]`), for custom RGB / reflectance math. The deprecated
//!   [`render_visible_bands`] name remains as a compatibility alias.
//! - [`render_ir`] -> `(H x W float32, Georef)` RAW brightness temperature in KELVIN; with
//!   `enhancement=` it instead returns `(H x W float32, H x W x 3 uint8, Georef)`.
//! - [`render_water_vapor`] -> `(H x W float32, Georef)` RAW water-vapor band (6.2/6.9/7.3
//!   um) brightness temperature in KELVIN; `band=` picks the level, `enhancement=` adds the
//!   colored RGB (as `render_ir`). The WV weighting function samples upper/mid/lower moisture.
//! - [`render_precipitable_water`] -> `(H x W float32, Georef)` RAW precipitable water in mm
//!   (the vertically-integrated water-vapor column).
//! - [`render_cloud_top_temp`] -> `(H x W float32, Georef)` RAW cloud-top temperature in
//!   KELVIN at the visible tau~1 level (`NaN` where there is no cloud).
//! - [`render_cloud_optical_depth`] -> `(H x W float32, Georef)` RAW total-column visible
//!   cloud optical depth (dimensionless; clear = 0).
//!   These three DERIVED-FIELD functions return raw physical scalar arrays for plotting with
//!   your own colormaps; `colormap=True` adds a basic `H x W x 3` uint8 colormap image too.
//! - [`render_cloud_layer`] -> `(H x W x 4 uint8, H x W float32, Georef)` — the WEB-MAP
//!   cloud layer pair (cloud-only RGBA + ground-shadow multiply layer) on a Web-Mercator
//!   grid with Mapbox ImageSource corner lon/lats on `georef.mercator_corners`.
//! - [`render_perspective`] -> `(H x W x 3 uint8, Georef)` — a FREE-PERSPECTIVE frame
//!   (eye/look/fov pinhole camera through the same marches; the angled-3D flyover
//!   product); `cloud_layer_only=True` -> `(H x W x 4 uint8, Georef)` cloud field only.
//!   The camera rides on `georef.camera_pose`.
//! - [`solar_elevation_grid`] -> `H x W float32` — the exact per-pixel surface-light
//!   elevation used for low-sun QA, including the renderer's partial-override behavior.
//!
//! Default `view='topdown'` (map-registered, north-up — the natural fit for a top-down
//! Lambert map); `view='geo'` gives the from-space geostationary view.
//! `resolution='native'` means one output pixel per source-model grid cell, not the
//! highest possible output resolution. `abi1km` / `abi2km` select 1 km / 2 km output
//! sampling and may upsample a coarse model or downsample a fine WRF grid.
//!
//! Every render function also takes `threads=` (cap the render worker threads; the rayon pool
//! is GLOBAL and built once per process — the first render call's value wins, see
//! [`apply_thread_cap`]) and returns honesty metadata on the [`Georef`]
//! (`time_is_fallback`, `ground_source`, `ground_status`), with `UserWarning`s raised
//! on a fabricated-date or downgraded-ground render ([`warn_downgrades`]).
//!
//! The binding is QUIET BY DEFAULT: the engine's diagnostic stderr lines (ingest
//! progress / warnings) are disabled at module init unless `SIMSAT_LOG=1`; flip at
//! runtime with [`set_verbose`]. Honesty metadata / `UserWarning`s are unaffected.

use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::path::PathBuf;

use simsat_engine::api::{
    self, BlueMarble, ExtentKind, FractionalCloudMode, FrameData, GroundSource, Product,
    RenderBackend, RenderIntent, RenderParams, RenderResult, SunOverride,
};
use simsat_engine::bricks::StorageProfile;
use simsat_engine::camera::{GeoNavigation, ResolutionMode, SatellitePreset, ViewMode};
use simsat_engine::clouds::{CloudMultiscatterMode, StepQuality};
use simsat_engine::derived::DerivedField;
use simsat_engine::instrument_footprint::InstrumentFootprint;
use simsat_engine::ir_enhance::IrEnhancement;
use simsat_engine::optics::CloudOpticsMode;
use simsat_engine::render::{SurfacePostlightToeConfig, TwilightSurfaceRecoveryConfig};
use simsat_engine::solar::SolarFrame;
use simsat_engine::thermal_sensor::ThermalSensor;
use simsat_engine::topdown::{configure_global_rayon, effective_thread_count};
use simsat_engine::web_layer;
use simsat_engine::wv::WvBand;

/// The georeference returned with every frame: the projection params, an `extent` for
/// `imshow`, and the H x W lat/lon mesh for `pcolormesh`.
///
/// Attributes:
///   view (str): 'topdown' or 'geo'.
///   extent (tuple[float, float, float, float]): (x0, x1, y0, y1) = (left, right, bottom,
///       top). Place with `ax.imshow(arr, extent=geo.extent, transform=crs,
///       origin='upper')`. For 'topdown' the units are projection metres (build `crs` from
///       `geo.proj4`); for 'geo' they are lon/lat degrees (use PlateCarree, or prefer the
///       lat/lon mesh).
///   extent_kind (str): 'projection_meters' (topdown) or 'lonlat_degrees' (geo).
///   proj4 (str): a PROJ.4 string for `geo.extent`'s coordinate system, EXACTLY consistent
///       with the extent (build a cartopy CRS with
///       `cartopy.crs.Projection(pyproj.CRS.from_proj4(geo.proj4))`).
///   proj_kind (str): 'lcc' | 'stere' | 'merc' | 'latlon'.
///   crs_params (dict): the projection parameters (PROJ keys + the raw WRF attributes).
///   lat (numpy H x W float32): per-pixel geodetic latitude (NaN off-domain).
///   lon (numpy H x W float32): per-pixel geodetic longitude.
///
/// Render-honesty metadata (rides on this object; also raised as UserWarnings):
///   time_is_fallback (bool): True when the source carried no parseable valid time and
///       the render used the FABRICATED fallback date 2004-06-21 12:00 UT — the sun
///       position / Blue Marble season of a visible frame are then not the run's real
///       conditions.
///   ground_source (str): where the visible ground pixels came from — '2km' |
///       '8km-fallback' | 'flat-albedo' | 'single-file' | 'none' (thermal / derived
///       products use no ground texture).
///   ground_status (list[str]): the ground-resolution status lines (seasonal Blue
///       Marble download / fallback progress); empty when nothing noteworthy happened.
///   intent (str): `display` or `sensor-fast-gray`.
///   observation_operator (str): stable provenance slug (`simsat-display-v1` or
///       `simsat-fast-gray-v1`).
///   intent_adjustments (list[str]): every automatic strict-intent substitution.
///   intent_limitations (list[str]): explicit scientific limitations of that operator.
///   thermal_sensor (str | None): selected response for IR or an IR-containing composite.
///   instrument_footprint (str): complete-radiance spatial-response stage (`off` by default).
///   instrument_footprint_metadata (dict | None): source/limitation provenance when active.
///   abi_fixed_grid_crop (dict | None): signed global half-pitch crop indices for an exact
///       ABI 2-km lattice. SSP is the shared corner of the four +/-28-urad pixels.
///   science_warnings (list[str]): explicit observation-operator limitations.
#[pyclass(module = "simsat", frozen)]
pub struct Georef {
    #[pyo3(get)]
    view: String,
    #[pyo3(get)]
    extent: (f64, f64, f64, f64),
    #[pyo3(get)]
    extent_kind: String,
    #[pyo3(get)]
    proj4: String,
    #[pyo3(get)]
    proj_kind: String,
    #[pyo3(get)]
    crs_params: Py<PyDict>,
    #[pyo3(get)]
    lat: Py<PyArray2<f32>>,
    #[pyo3(get)]
    lon: Py<PyArray2<f32>>,
    #[pyo3(get)]
    time_is_fallback: bool,
    #[pyo3(get)]
    ground_source: String,
    #[pyo3(get)]
    ground_status: Vec<String>,
    /// Requested render intent (`display` or `sensor-fast-gray`).
    #[pyo3(get)]
    intent: String,
    /// Stable operator provenance (`simsat-display-v1` or `simsat-fast-gray-v1`).
    #[pyo3(get)]
    observation_operator: String,
    /// Exact automatic strict-intent substitutions, in application order.
    #[pyo3(get)]
    intent_adjustments: Vec<String>,
    /// Honest limitations of the selected observation operator.
    #[pyo3(get)]
    intent_limitations: Vec<String>,
    /// Thermal response slug for IR / IR-containing products; `None` otherwise.
    #[pyo3(get)]
    thermal_sensor: Option<String>,
    /// Instrument spatial-response slug (`off` unless explicitly selected).
    #[pyo3(get)]
    instrument_footprint: String,
    /// Instrument source/domain/limitation provenance when active.
    #[pyo3(get)]
    instrument_footprint_metadata: Option<Py<PyDict>>,
    /// Exact ABI 2-km global-lattice crop metadata when the output uses that grid.
    #[pyo3(get)]
    abi_fixed_grid_crop: Option<Py<PyDict>>,
    /// Explicit observation-operator limitations for programmatic provenance.
    #[pyo3(get)]
    science_warnings: Vec<String>,
    /// Brick extinction storage profile actually decoded.
    #[pyo3(get)]
    storage_profile: String,
    /// The four image corner `(lon, lat)` pairs in the Mapbox GL ImageSource
    /// `coordinates` order (top-left/NW, top-right/NE, bottom-right/SE,
    /// bottom-left/SW). Only set by `render_cloud_layer` (the Web-Mercator delivery);
    /// `None` for every other product.
    #[pyo3(get)]
    mercator_corners: Option<Vec<(f64, f64)>>,
    /// The free-perspective camera pose dict (`eye_lat`/`eye_lon`/`eye_alt_m`/
    /// `look_lat`/`look_lon`/`look_alt_m`/`fov_deg`/`width`/`height`) — only set by
    /// `render_perspective` (a perspective frame always states its camera); `None`
    /// for every other product.
    #[pyo3(get)]
    camera_pose: Option<Py<PyDict>>,
    /// Geostationary sensor-grid navigation slug; `None` for non-geo products.
    #[pyo3(get)]
    geo_navigation: Option<String>,
    /// Exact sensor/model geometry provenance dictionary for a geo product.
    #[pyo3(get)]
    geo_navigation_geometry: Option<Py<PyDict>>,
}

#[pymethods]
impl Georef {
    fn __repr__(&self) -> String {
        format!(
            "Georef(view='{}', extent_kind='{}', proj_kind='{}', extent={:?})",
            self.view, self.extent_kind, self.proj_kind, self.extent
        )
    }
}

/// Render the finished true-color RGB.
///
/// `margin` (default 0.0) is a zoom-out / domain margin as a FRACTION of the domain size
/// added on each side: 0.0 = the domain edge-to-edge; e.g. 0.3 frames the domain with real
/// surrounding earth (Blue Marble ground + clear sky, no WRF weather outside the domain).
/// It applies to both views and to every render function here.
///
/// Visible-family atmosphere/cloud controls are shared by RGB, bands, GeoColor,
/// Sandwich, cloud-layer, and perspective renders: `aerosol_optical_depth` (default
/// 0.05), `rh_aerosol_swelling` (1.5x when true), `atmosphere_correction`,
/// `terrain_atmosphere`, `fractional_clouds` (default true),
/// `fractional_cloud_mode` (finished display bindings default to the reviewed
/// `deterministic-2`; `effective-od` remains the explicit fast/sensor-compatible
/// choice, with `deterministic-4/8/16` as higher-cost references), and
/// `cloud_optical_depth_scale` (0..=4, shipped default 0.15 by owner cross-file visual
/// calibration; 1.0 is unscaled model extinction), and the default-on
/// `feather_exposed_domain_edges` finite-domain presentation control. Fractional clouds
/// use the model cloud
/// fraction when present; false restores legacy horizontally-full cells. The
/// deterministic mode performs four fixed shared-u CPU marches, averages linear
/// radiance, then tonemaps once. The OD scale is
/// a visible sensitivity control and does not alter the quantitative
/// `render_cloud_optical_depth` product. `clouds` remains the explicit feature bypass;
/// `multiscatter` controls the established higher scattering octaves without changing
/// transmittance. `cloud_multiscatter` is an explicit override accepting
/// `legacy-octaves`, `single-scatter`, or opt-in experimental `delta-flux-v1` /
/// `delta-flux-v2b` / `delta-flux-v3-memory`; leaving it unset preserves the historical
/// boolean behavior exactly.
/// `beer_powder` enables the optional direct-sun shaping, and `granulation` enables
/// display-only sub-grid cloud-edge erosion; both default off. Finished visible display
/// products also expose `topdown_stratiform_regularization`, an opt-in/default-off
/// low/liquid-deck reconstruction used only by the top-down finished-visible path.
/// They also expose `topdown_cloud_footprint`, an opt-in/default-off seven-tap
/// pre-tonemap footprint applied only to the cloud radiance residual, leaving terrain
/// sharp, and `topdown_shadow_antialias`, a default-on filter for top-down ground-shadow
/// map aliasing. Geostationary and raw visible bands ignore these controls. Finished visible display
/// products also accept `ground_gain`, `cloud_softclip`, and `cloud_highlight_max` as
/// optional calibration overrides. The land-only controls
/// `land_sza_normalization` / `land_sza_max_gain` and `land_dark_toe` plus its
/// knee/gamma/max-gain parameters are independently switchable and default on in the
/// owner-selected current display preset. Passing both booleans false is the exact legacy
/// identity. `surface_postlight_toe` is a separate default-off display experiment over
/// LAND after lighting and view transmittance but before atmospheric airlight/cloud
/// compositing; its knee/gamma/max-gain defaults are 0.18/0.80/1.35. The independent
/// `twilight_surface_recovery` uses the tighter -6..+12 degree low-sun gate and is enabled
/// for finished visible-family displays with the owner-selected 0.30/0.50/4.0 controls.
/// Both run on CPU and GPU. Sensor Fast Gray substitutes both off and reports the
/// adjustment. Raw visible bands deliberately expose neither display-only control.
///
/// `threads` (default None = all cores, or RAYON_NUM_THREADS) caps the render worker
/// threads for THIS PROCESS. The pool is global and built ONCE — the first render call's
/// value wins; later calls cannot change it. Available on every render function here.
///
/// Returns `(rgb, georef)` where `rgb` is a numpy `H x W x 3` uint8 array (row 0 = north).
#[pyfunction]
#[pyo3(signature = (
    input, *, storage_profile="compact-u8", backend="cpu", intent="display", sat="goes-east", geo_navigation="model-sphere", view="topdown", timestep=0, resolution="native", margin=0.0,
    aerosol_optical_depth=0.05, rh_aerosol_swelling=false, atmosphere_correction=true,
    terrain_atmosphere=true, land_sza_normalization=true, land_sza_max_gain=4.0,
    land_dark_toe=true, land_dark_toe_knee=0.08, land_dark_toe_gamma=0.65,
    land_dark_toe_max_gain=1.5, surface_postlight_toe=false,
    surface_postlight_toe_knee=0.18, surface_postlight_toe_gamma=0.80,
    surface_postlight_toe_max_gain=1.35, twilight_surface_recovery=true,
    twilight_surface_recovery_knee=0.30, twilight_surface_recovery_gamma=0.50,
    twilight_surface_recovery_max_gain=4.0, exposure=None, ground_gain=None, cloud_softclip=None,
    cloud_highlight_max=None, multiscatter=true, cloud_multiscatter=None, beer_powder=false,
    steps="offline", clouds=true, fractional_clouds=true, fractional_cloud_mode="deterministic-2", cloud_optical_depth_scale=0.15,
    cloud_optics="fixed",
    feather_exposed_domain_edges=true, granulation=false,
    topdown_stratiform_regularization=false,
    topdown_cloud_footprint=false,
    topdown_shadow_antialias=true,
    sun_elev=None, sun_az=None, cache=None, bluemarble=None,
    bluemarble_month=None, bluemarble_download=true, threads=None
))]
#[allow(clippy::too_many_arguments)]
fn render_visible_rgb<'py>(
    py: Python<'py>,
    input: String,
    storage_profile: &str,
    backend: &str,
    intent: &str,
    sat: &str,
    geo_navigation: &str,
    view: &str,
    timestep: usize,
    resolution: &str,
    margin: f64,
    aerosol_optical_depth: f32,
    rh_aerosol_swelling: bool,
    atmosphere_correction: bool,
    terrain_atmosphere: bool,
    land_sza_normalization: bool,
    land_sza_max_gain: f64,
    land_dark_toe: bool,
    land_dark_toe_knee: f64,
    land_dark_toe_gamma: f64,
    land_dark_toe_max_gain: f64,
    surface_postlight_toe: bool,
    surface_postlight_toe_knee: f64,
    surface_postlight_toe_gamma: f64,
    surface_postlight_toe_max_gain: f64,
    twilight_surface_recovery: bool,
    twilight_surface_recovery_knee: f64,
    twilight_surface_recovery_gamma: f64,
    twilight_surface_recovery_max_gain: f64,
    exposure: Option<f64>,
    ground_gain: Option<f64>,
    cloud_softclip: Option<f64>,
    cloud_highlight_max: Option<f64>,
    multiscatter: bool,
    cloud_multiscatter: Option<&str>,
    beer_powder: bool,
    steps: &str,
    clouds: bool,
    fractional_clouds: bool,
    fractional_cloud_mode: &str,
    cloud_optical_depth_scale: f32,
    cloud_optics: &str,
    feather_exposed_domain_edges: bool,
    granulation: bool,
    topdown_stratiform_regularization: bool,
    topdown_cloud_footprint: bool,
    topdown_shadow_antialias: bool,
    sun_elev: Option<f64>,
    sun_az: Option<f64>,
    cache: Option<String>,
    bluemarble: Option<String>,
    bluemarble_month: Option<u32>,
    bluemarble_download: bool,
    threads: Option<usize>,
) -> PyResult<(Bound<'py, PyArray3<u8>>, Georef)> {
    let mut params = build_visible_params(
        input,
        storage_profile,
        intent,
        sat,
        geo_navigation,
        view,
        timestep,
        resolution,
        margin,
        aerosol_optical_depth,
        rh_aerosol_swelling,
        atmosphere_correction,
        terrain_atmosphere,
        land_sza_normalization,
        land_sza_max_gain,
        land_dark_toe,
        land_dark_toe_knee,
        land_dark_toe_gamma,
        land_dark_toe_max_gain,
        surface_postlight_toe,
        surface_postlight_toe_knee,
        surface_postlight_toe_gamma,
        surface_postlight_toe_max_gain,
        twilight_surface_recovery,
        twilight_surface_recovery_knee,
        twilight_surface_recovery_gamma,
        twilight_surface_recovery_max_gain,
        exposure,
        ground_gain,
        cloud_softclip,
        cloud_highlight_max,
        multiscatter,
        cloud_multiscatter,
        beer_powder,
        steps,
        clouds,
        fractional_clouds,
        fractional_cloud_mode,
        cloud_optical_depth_scale,
        cloud_optics,
        feather_exposed_domain_edges,
        granulation,
        topdown_stratiform_regularization,
        topdown_cloud_footprint,
        topdown_shadow_antialias,
        sun_elev,
        sun_az,
        cache,
        bluemarble,
        bluemarble_month,
        bluemarble_download,
    )?;
    params.backend = parse_backend(backend)?;
    apply_thread_cap(threads);
    let result = render_or_err(py, params, Product::VisibleRgb)?;
    warn_downgrades(py, &result, true);
    let (ny, nx) = (result.ny, result.nx);
    let geo = build_georef(py, &result)?;
    let rgb = match result.data {
        FrameData::Visible { rgb, .. } => rgb,
        _ => return Err(runtime_err("expected a visible frame")),
    };
    let arr = Array3::from_shape_vec((ny, nx, 3), rgb)
        .map_err(|e| runtime_err(&format!("rgb reshape: {e}")))?
        .into_pyarray(py);
    Ok((arr, geo))
}

/// Render SimSat Day/Night Color, a GeoColor-style composite: true-color visible by day,
/// colored band-13 IR by night, crossfaded across the terminator by per-pixel solar elevation.
/// This is not yet sensor-derived ABI GeoColor: its visible side uses SimSat's broad RGB
/// operator and its IR side uses the selected Band 13 response. It is always meaningful day
/// or night; the night side shows storms/clouds in IR rather than city lights. Sun, exposure,
/// and cloud controls apply to the visible (day) half.
///
/// Returns `(rgb, georef)` where `rgb` is a numpy `H x W x 3` uint8 array (row 0 = north),
/// exactly like [`render_visible_rgb`].
#[pyfunction]
#[pyo3(signature = (
    input, *, storage_profile="compact-u8", intent="display", sat="goes-east", geo_navigation="model-sphere", view="topdown", timestep=0, resolution="native", margin=0.0,
    sensor="fast-gray", instrument_footprint="off",
    aerosol_optical_depth=0.05, rh_aerosol_swelling=false, atmosphere_correction=true,
    terrain_atmosphere=true, land_sza_normalization=true, land_sza_max_gain=4.0,
    land_dark_toe=true, land_dark_toe_knee=0.08, land_dark_toe_gamma=0.65,
    land_dark_toe_max_gain=1.5, surface_postlight_toe=false,
    surface_postlight_toe_knee=0.18, surface_postlight_toe_gamma=0.80,
    surface_postlight_toe_max_gain=1.35, twilight_surface_recovery=true,
    twilight_surface_recovery_knee=0.30, twilight_surface_recovery_gamma=0.50,
    twilight_surface_recovery_max_gain=4.0, exposure=None, ground_gain=None, cloud_softclip=None,
    cloud_highlight_max=None, multiscatter=true, cloud_multiscatter=None, beer_powder=false,
    steps="offline", clouds=true, fractional_clouds=true, fractional_cloud_mode="deterministic-2", cloud_optical_depth_scale=0.15,
    cloud_optics="fixed",
    feather_exposed_domain_edges=true, granulation=false,
    topdown_stratiform_regularization=false,
    topdown_cloud_footprint=false,
    topdown_shadow_antialias=true,
    sun_elev=None, sun_az=None, cache=None, bluemarble=None,
    bluemarble_month=None, bluemarble_download=true, threads=None
))]
#[allow(clippy::too_many_arguments)]
fn render_geocolor<'py>(
    py: Python<'py>,
    input: String,
    storage_profile: &str,
    intent: &str,
    sat: &str,
    geo_navigation: &str,
    view: &str,
    timestep: usize,
    resolution: &str,
    margin: f64,
    sensor: &str,
    instrument_footprint: &str,
    aerosol_optical_depth: f32,
    rh_aerosol_swelling: bool,
    atmosphere_correction: bool,
    terrain_atmosphere: bool,
    land_sza_normalization: bool,
    land_sza_max_gain: f64,
    land_dark_toe: bool,
    land_dark_toe_knee: f64,
    land_dark_toe_gamma: f64,
    land_dark_toe_max_gain: f64,
    surface_postlight_toe: bool,
    surface_postlight_toe_knee: f64,
    surface_postlight_toe_gamma: f64,
    surface_postlight_toe_max_gain: f64,
    twilight_surface_recovery: bool,
    twilight_surface_recovery_knee: f64,
    twilight_surface_recovery_gamma: f64,
    twilight_surface_recovery_max_gain: f64,
    exposure: Option<f64>,
    ground_gain: Option<f64>,
    cloud_softclip: Option<f64>,
    cloud_highlight_max: Option<f64>,
    multiscatter: bool,
    cloud_multiscatter: Option<&str>,
    beer_powder: bool,
    steps: &str,
    clouds: bool,
    fractional_clouds: bool,
    fractional_cloud_mode: &str,
    cloud_optical_depth_scale: f32,
    cloud_optics: &str,
    feather_exposed_domain_edges: bool,
    granulation: bool,
    topdown_stratiform_regularization: bool,
    topdown_cloud_footprint: bool,
    topdown_shadow_antialias: bool,
    sun_elev: Option<f64>,
    sun_az: Option<f64>,
    cache: Option<String>,
    bluemarble: Option<String>,
    bluemarble_month: Option<u32>,
    bluemarble_download: bool,
    threads: Option<usize>,
) -> PyResult<(Bound<'py, PyArray3<u8>>, Georef)> {
    let mut params = build_visible_params(
        input,
        storage_profile,
        intent,
        sat,
        geo_navigation,
        view,
        timestep,
        resolution,
        margin,
        aerosol_optical_depth,
        rh_aerosol_swelling,
        atmosphere_correction,
        terrain_atmosphere,
        land_sza_normalization,
        land_sza_max_gain,
        land_dark_toe,
        land_dark_toe_knee,
        land_dark_toe_gamma,
        land_dark_toe_max_gain,
        surface_postlight_toe,
        surface_postlight_toe_knee,
        surface_postlight_toe_gamma,
        surface_postlight_toe_max_gain,
        twilight_surface_recovery,
        twilight_surface_recovery_knee,
        twilight_surface_recovery_gamma,
        twilight_surface_recovery_max_gain,
        exposure,
        ground_gain,
        cloud_softclip,
        cloud_highlight_max,
        multiscatter,
        cloud_multiscatter,
        beer_powder,
        steps,
        clouds,
        fractional_clouds,
        fractional_cloud_mode,
        cloud_optical_depth_scale,
        cloud_optics,
        feather_exposed_domain_edges,
        granulation,
        topdown_stratiform_regularization,
        topdown_cloud_footprint,
        topdown_shadow_antialias,
        sun_elev,
        sun_az,
        cache,
        bluemarble,
        bluemarble_month,
        bluemarble_download,
    )?;
    params.thermal_sensor = parse_thermal_sensor(sensor)?;
    params.instrument_footprint = parse_instrument_footprint(instrument_footprint)?;
    apply_thread_cap(threads);
    let result = render_or_err(py, params, Product::GeoColor)?;
    warn_downgrades(py, &result, true);
    let (ny, nx) = (result.ny, result.nx);
    let geo = build_georef(py, &result)?;
    let rgb = match result.data {
        FrameData::Visible { rgb, .. } => rgb,
        _ => return Err(runtime_err("expected a visible (GeoColor) frame")),
    };
    let arr = Array3::from_shape_vec((ny, nx, 3), rgb)
        .map_err(|e| runtime_err(&format!("geocolor rgb reshape: {e}")))?
        .into_pyarray(py);
    Ok((arr, geo))
}

/// Render the Sandwich composite (the classic severe-convection view): the visible true-color
/// RGB as the base everywhere, with a color-enhanced band-13 IR overlaid ONLY on the COLD (high)
/// cloud tops at an alpha that ramps with coldness. The visible gives the fine cloud texture;
/// the IR color highlights the coldest overshooting tops. A DAYTIME-convection product (the
/// visible base needs daylight; at night it degrades to ~IR). The sun / exposure / clouds
/// controls drive the visible base.
///
/// Returns `(rgb, georef)` where `rgb` is a numpy `H x W x 3` uint8 array (row 0 = north),
/// exactly like [`render_visible_rgb`].
#[pyfunction]
#[pyo3(signature = (
    input, *, storage_profile="compact-u8", intent="display", sat="goes-east", geo_navigation="model-sphere", view="topdown", timestep=0, resolution="native", margin=0.0,
    sensor="fast-gray", instrument_footprint="off",
    aerosol_optical_depth=0.05, rh_aerosol_swelling=false, atmosphere_correction=true,
    terrain_atmosphere=true, land_sza_normalization=true, land_sza_max_gain=4.0,
    land_dark_toe=true, land_dark_toe_knee=0.08, land_dark_toe_gamma=0.65,
    land_dark_toe_max_gain=1.5, surface_postlight_toe=false,
    surface_postlight_toe_knee=0.18, surface_postlight_toe_gamma=0.80,
    surface_postlight_toe_max_gain=1.35, twilight_surface_recovery=true,
    twilight_surface_recovery_knee=0.30, twilight_surface_recovery_gamma=0.50,
    twilight_surface_recovery_max_gain=4.0, exposure=None, ground_gain=None, cloud_softclip=None,
    cloud_highlight_max=None, multiscatter=true, cloud_multiscatter=None, beer_powder=false,
    steps="offline", clouds=true, fractional_clouds=true, fractional_cloud_mode="deterministic-2", cloud_optical_depth_scale=0.15,
    cloud_optics="fixed",
    feather_exposed_domain_edges=true, granulation=false,
    topdown_stratiform_regularization=false,
    topdown_cloud_footprint=false,
    topdown_shadow_antialias=true,
    sun_elev=None, sun_az=None, cache=None, bluemarble=None,
    bluemarble_month=None, bluemarble_download=true, threads=None
))]
#[allow(clippy::too_many_arguments)]
fn render_sandwich<'py>(
    py: Python<'py>,
    input: String,
    storage_profile: &str,
    intent: &str,
    sat: &str,
    geo_navigation: &str,
    view: &str,
    timestep: usize,
    resolution: &str,
    margin: f64,
    sensor: &str,
    instrument_footprint: &str,
    aerosol_optical_depth: f32,
    rh_aerosol_swelling: bool,
    atmosphere_correction: bool,
    terrain_atmosphere: bool,
    land_sza_normalization: bool,
    land_sza_max_gain: f64,
    land_dark_toe: bool,
    land_dark_toe_knee: f64,
    land_dark_toe_gamma: f64,
    land_dark_toe_max_gain: f64,
    surface_postlight_toe: bool,
    surface_postlight_toe_knee: f64,
    surface_postlight_toe_gamma: f64,
    surface_postlight_toe_max_gain: f64,
    twilight_surface_recovery: bool,
    twilight_surface_recovery_knee: f64,
    twilight_surface_recovery_gamma: f64,
    twilight_surface_recovery_max_gain: f64,
    exposure: Option<f64>,
    ground_gain: Option<f64>,
    cloud_softclip: Option<f64>,
    cloud_highlight_max: Option<f64>,
    multiscatter: bool,
    cloud_multiscatter: Option<&str>,
    beer_powder: bool,
    steps: &str,
    clouds: bool,
    fractional_clouds: bool,
    fractional_cloud_mode: &str,
    cloud_optical_depth_scale: f32,
    cloud_optics: &str,
    feather_exposed_domain_edges: bool,
    granulation: bool,
    topdown_stratiform_regularization: bool,
    topdown_cloud_footprint: bool,
    topdown_shadow_antialias: bool,
    sun_elev: Option<f64>,
    sun_az: Option<f64>,
    cache: Option<String>,
    bluemarble: Option<String>,
    bluemarble_month: Option<u32>,
    bluemarble_download: bool,
    threads: Option<usize>,
) -> PyResult<(Bound<'py, PyArray3<u8>>, Georef)> {
    let mut params = build_visible_params(
        input,
        storage_profile,
        intent,
        sat,
        geo_navigation,
        view,
        timestep,
        resolution,
        margin,
        aerosol_optical_depth,
        rh_aerosol_swelling,
        atmosphere_correction,
        terrain_atmosphere,
        land_sza_normalization,
        land_sza_max_gain,
        land_dark_toe,
        land_dark_toe_knee,
        land_dark_toe_gamma,
        land_dark_toe_max_gain,
        surface_postlight_toe,
        surface_postlight_toe_knee,
        surface_postlight_toe_gamma,
        surface_postlight_toe_max_gain,
        twilight_surface_recovery,
        twilight_surface_recovery_knee,
        twilight_surface_recovery_gamma,
        twilight_surface_recovery_max_gain,
        exposure,
        ground_gain,
        cloud_softclip,
        cloud_highlight_max,
        multiscatter,
        cloud_multiscatter,
        beer_powder,
        steps,
        clouds,
        fractional_clouds,
        fractional_cloud_mode,
        cloud_optical_depth_scale,
        cloud_optics,
        feather_exposed_domain_edges,
        granulation,
        topdown_stratiform_regularization,
        topdown_cloud_footprint,
        topdown_shadow_antialias,
        sun_elev,
        sun_az,
        cache,
        bluemarble,
        bluemarble_month,
        bluemarble_download,
    )?;
    params.thermal_sensor = parse_thermal_sensor(sensor)?;
    params.instrument_footprint = parse_instrument_footprint(instrument_footprint)?;
    apply_thread_cap(threads);
    let result = render_or_err(py, params, Product::Sandwich)?;
    warn_downgrades(py, &result, true);
    let (ny, nx) = (result.ny, result.nx);
    let geo = build_georef(py, &result)?;
    let rgb = match result.data {
        FrameData::Visible { rgb, .. } => rgb,
        _ => return Err(runtime_err("expected a visible (sandwich) frame")),
    };
    let arr = Array3::from_shape_vec((ny, nx, 3), rgb)
        .map_err(|e| runtime_err(&format!("sandwich rgb reshape: {e}")))?
        .into_pyarray(py);
    Ok((arr, geo))
}

/// Render raw broad-RGB reflectance (pre-tonemap) for custom RGB or reflectance math.
///
/// Returns `(reflectance, georef)` where `reflectance` is a numpy `H x W x 3` float32
/// array in `[0, 1]` (broad R, G, B reflectance factors; row 0 = north). These channels
/// are not yet sensor-response-integrated ABI visible bands.
#[pyfunction]
#[pyo3(signature = (
    input, *, storage_profile="compact-u8", intent="display", sat="goes-east", geo_navigation="model-sphere", view="topdown", timestep=0, resolution="native", margin=0.0,
    aerosol_optical_depth=0.05, rh_aerosol_swelling=false, atmosphere_correction=true,
    terrain_atmosphere=true, multiscatter=true, cloud_multiscatter=None, beer_powder=false,
    steps="offline", clouds=true,
    fractional_clouds=true, fractional_cloud_mode="effective-od", cloud_optical_depth_scale=0.15,
    cloud_optics="fixed", granulation=false, sun_elev=None,
    sun_az=None, cache=None,
    bluemarble=None,
    bluemarble_month=None, bluemarble_download=true, threads=None
))]
#[allow(clippy::too_many_arguments)]
fn render_rgb_reflectance<'py>(
    py: Python<'py>,
    input: String,
    storage_profile: &str,
    intent: &str,
    sat: &str,
    geo_navigation: &str,
    view: &str,
    timestep: usize,
    resolution: &str,
    margin: f64,
    aerosol_optical_depth: f32,
    rh_aerosol_swelling: bool,
    atmosphere_correction: bool,
    terrain_atmosphere: bool,
    multiscatter: bool,
    cloud_multiscatter: Option<&str>,
    beer_powder: bool,
    steps: &str,
    clouds: bool,
    fractional_clouds: bool,
    fractional_cloud_mode: &str,
    cloud_optical_depth_scale: f32,
    cloud_optics: &str,
    granulation: bool,
    sun_elev: Option<f64>,
    sun_az: Option<f64>,
    cache: Option<String>,
    bluemarble: Option<String>,
    bluemarble_month: Option<u32>,
    bluemarble_download: bool,
    threads: Option<usize>,
) -> PyResult<(Bound<'py, PyArray3<f32>>, Georef)> {
    render_visible_bands(
        py,
        input,
        storage_profile,
        intent,
        sat,
        geo_navigation,
        view,
        timestep,
        resolution,
        margin,
        aerosol_optical_depth,
        rh_aerosol_swelling,
        atmosphere_correction,
        terrain_atmosphere,
        multiscatter,
        cloud_multiscatter,
        beer_powder,
        steps,
        clouds,
        fractional_clouds,
        fractional_cloud_mode,
        cloud_optical_depth_scale,
        cloud_optics,
        granulation,
        sun_elev,
        sun_az,
        cache,
        bluemarble,
        bluemarble_month,
        bluemarble_download,
        threads,
    )
}

/// Deprecated compatibility alias for [`render_rgb_reflectance`].
///
/// `render_visible_bands` returns exactly the same broad-RGB reflectance array and remains
/// available for existing code. New code should use `render_rgb_reflectance`; the old name
/// could be mistaken for discrete, sensor-response-integrated visible bands.
///
/// Returns `(bands, georef)` where `bands` is a numpy `H x W x 3` float32 array in `[0, 1]`
/// (R, G, B reflectance factors; row 0 = north).
#[pyfunction]
#[pyo3(signature = (
    input, *, storage_profile="compact-u8", intent="display", sat="goes-east", geo_navigation="model-sphere", view="topdown", timestep=0, resolution="native", margin=0.0,
    aerosol_optical_depth=0.05, rh_aerosol_swelling=false, atmosphere_correction=true,
    terrain_atmosphere=true, multiscatter=true, cloud_multiscatter=None, beer_powder=false,
    steps="offline", clouds=true,
    fractional_clouds=true, fractional_cloud_mode="effective-od", cloud_optical_depth_scale=0.15,
    cloud_optics="fixed", granulation=false, sun_elev=None,
    sun_az=None, cache=None,
    bluemarble=None,
    bluemarble_month=None, bluemarble_download=true, threads=None
))]
#[allow(clippy::too_many_arguments)]
fn render_visible_bands<'py>(
    py: Python<'py>,
    input: String,
    storage_profile: &str,
    intent: &str,
    sat: &str,
    geo_navigation: &str,
    view: &str,
    timestep: usize,
    resolution: &str,
    margin: f64,
    aerosol_optical_depth: f32,
    rh_aerosol_swelling: bool,
    atmosphere_correction: bool,
    terrain_atmosphere: bool,
    multiscatter: bool,
    cloud_multiscatter: Option<&str>,
    beer_powder: bool,
    steps: &str,
    clouds: bool,
    fractional_clouds: bool,
    fractional_cloud_mode: &str,
    cloud_optical_depth_scale: f32,
    cloud_optics: &str,
    granulation: bool,
    sun_elev: Option<f64>,
    sun_az: Option<f64>,
    cache: Option<String>,
    bluemarble: Option<String>,
    bluemarble_month: Option<u32>,
    bluemarble_download: bool,
    threads: Option<usize>,
) -> PyResult<(Bound<'py, PyArray3<f32>>, Georef)> {
    let params = build_visible_params(
        input,
        storage_profile,
        intent,
        sat,
        geo_navigation,
        view,
        timestep,
        resolution,
        margin,
        aerosol_optical_depth,
        rh_aerosol_swelling,
        atmosphere_correction,
        terrain_atmosphere,
        false,
        simsat_engine::render::LAND_SZA_MAX_GAIN,
        false,
        simsat_engine::render::LAND_DARK_TOE_KNEE,
        simsat_engine::render::LAND_DARK_TOE_GAMMA,
        simsat_engine::render::LAND_DARK_TOE_MAX_GAIN,
        false,
        simsat_engine::render::SURFACE_POSTLIGHT_TOE_KNEE,
        simsat_engine::render::SURFACE_POSTLIGHT_TOE_GAMMA,
        simsat_engine::render::SURFACE_POSTLIGHT_TOE_MAX_GAIN,
        false,
        simsat_engine::render::TWILIGHT_SURFACE_RECOVERY_KNEE,
        simsat_engine::render::TWILIGHT_SURFACE_RECOVERY_GAMMA,
        simsat_engine::render::TWILIGHT_SURFACE_RECOVERY_MAX_GAIN,
        None,
        None,
        None,
        None,
        multiscatter,
        cloud_multiscatter,
        beer_powder,
        steps,
        clouds,
        fractional_clouds,
        fractional_cloud_mode,
        cloud_optical_depth_scale,
        cloud_optics,
        false,
        granulation,
        false,
        false,
        false,
        sun_elev,
        sun_az,
        cache,
        bluemarble,
        bluemarble_month,
        bluemarble_download,
    )?;
    apply_thread_cap(threads);
    let result = render_or_err(py, params, Product::RgbReflectance)?;
    warn_downgrades(py, &result, true);
    let (ny, nx) = (result.ny, result.nx);
    let geo = build_georef(py, &result)?;
    let refl = match result.data {
        FrameData::Bands { reflectance } => reflectance,
        _ => return Err(runtime_err("expected a bands frame")),
    };
    let arr = Array3::from_shape_vec((ny, nx, 3), refl)
        .map_err(|e| runtime_err(&format!("bands reshape: {e}")))?
        .into_pyarray(py);
    Ok((arr, geo))
}

/// Render the RAW infrared brightness temperature (ABI band 13, 10.3 um) in KELVIN.
///
/// Returns `(bt, georef)` where `bt` is a numpy `H x W` float32 array in Kelvin (NaN
/// off-domain; row 0 = north). If `enhancement` is given (one of 'natural', 'cimss',
/// 'bd', 'avn', 'funktop', 'rainbow', 'gray') the return is instead
/// `(bt, rgb, georef)` with `rgb` a numpy `H x W x 3` uint8 colored image. 'natural'
/// remains the continuous NOAA heritage Band-13 grayscale; 'cimss' is the recommended
/// false-color display. `sensor='fast-gray'` preserves the historical center-wavelength
/// response; `sensor='goes-r-abi-band13-fm4'` applies NOAA's official
/// FM4/GOES-19 Band 13 SRF and emits a warning that absorption remains gray.
#[pyfunction]
#[pyo3(signature = (
    input, *, storage_profile="compact-u8", sat="goes-east", geo_navigation="model-sphere", view="topdown", timestep=0, resolution="native", margin=0.0,
    sensor="fast-gray", instrument_footprint="off", enhancement=None, cache=None, threads=None
))]
#[allow(clippy::too_many_arguments)]
fn render_ir<'py>(
    py: Python<'py>,
    input: String,
    storage_profile: &str,
    sat: &str,
    geo_navigation: &str,
    view: &str,
    timestep: usize,
    resolution: &str,
    margin: f64,
    sensor: &str,
    instrument_footprint: &str,
    enhancement: Option<String>,
    cache: Option<String>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut params = RenderParams::new(PathBuf::from(&input));
    force_surface_recovery_identity(&mut params);
    params.storage_profile = parse_storage_profile(storage_profile)?;
    params.satellite = parse_sat(sat)?;
    params.geo_navigation = parse_geo_navigation(geo_navigation)?;
    params.view = parse_view(view)?;
    params.timestep = timestep;
    params.resolution = parse_resolution(resolution)?;
    params.margin_frac = parse_margin(margin)?;
    params.thermal_sensor = parse_thermal_sensor(sensor)?;
    params.instrument_footprint = parse_instrument_footprint(instrument_footprint)?;
    params.bluemarble = BlueMarble::FlatAlbedo; // IR is thermal; no ground texture needed
    params.ir_enhancement = match &enhancement {
        Some(e) => Some(parse_enhancement(e)?),
        None => None,
    };
    if let Some(c) = cache {
        params.cache = PathBuf::from(c);
    }
    apply_thread_cap(threads);
    let result = render_or_err(py, params, Product::Ir)?;
    warn_downgrades(py, &result, false);
    let (ny, nx) = (result.ny, result.nx);
    let geo = build_georef(py, &result)?;
    let (bt, rgb) = match result.data {
        FrameData::Ir { bt_kelvin, rgb } => (bt_kelvin, rgb),
        _ => return Err(runtime_err("expected an IR frame")),
    };
    let bt_arr = Array2::from_shape_vec((ny, nx), bt)
        .map_err(|e| runtime_err(&format!("bt reshape: {e}")))?
        .into_pyarray(py);
    let geo_obj = Bound::new(py, geo)?;
    let tuple = match rgb {
        Some(rgb) => {
            let rgb_arr = Array3::from_shape_vec((ny, nx, 3), rgb)
                .map_err(|e| runtime_err(&format!("ir rgb reshape: {e}")))?
                .into_pyarray(py);
            PyTuple::new(
                py,
                [bt_arr.into_any(), rgb_arr.into_any(), geo_obj.into_any()],
            )?
        }
        None => PyTuple::new(py, [bt_arr.into_any(), geo_obj.into_any()])?,
    };
    Ok(tuple.into_any())
}

/// Render a RAW water-vapor band brightness temperature (ABI band 8/9/10 = 6.2/6.9/7.3
/// um) in KELVIN.
///
/// `band` selects the level: '6.2' (upper), '6.9' (mid), or '7.3' (lower). Returns
/// `(bt, georef)` where `bt` is a numpy `H x W` float32 array in Kelvin (NaN off-domain;
/// row 0 = north). With `enhancement` given (one of 'natural', 'cimss', 'bd', 'avn',
/// 'funktop', 'rainbow', 'gray') the return is instead `(bt, rgb, georef)` with `rgb` a numpy
/// `H x W x 3` uint8 colored image — for WV, 'cimss' is the classic WV moisture palette
/// and 'gray' is a WV-scaled grayscale (cold/moist white). Thermal — works day AND night.
#[pyfunction]
#[pyo3(signature = (
    input, *, storage_profile="compact-u8", band="6.2", sat="goes-east", geo_navigation="model-sphere", view="topdown", timestep=0, resolution="native",
    margin=0.0, enhancement=None, cache=None, threads=None
))]
#[allow(clippy::too_many_arguments)]
fn render_water_vapor<'py>(
    py: Python<'py>,
    input: String,
    storage_profile: &str,
    band: &str,
    sat: &str,
    geo_navigation: &str,
    view: &str,
    timestep: usize,
    resolution: &str,
    margin: f64,
    enhancement: Option<String>,
    cache: Option<String>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let wv_band = parse_wv_band(band)?;
    let mut params = RenderParams::new(PathBuf::from(&input));
    force_surface_recovery_identity(&mut params);
    params.storage_profile = parse_storage_profile(storage_profile)?;
    params.satellite = parse_sat(sat)?;
    params.geo_navigation = parse_geo_navigation(geo_navigation)?;
    params.view = parse_view(view)?;
    params.timestep = timestep;
    params.resolution = parse_resolution(resolution)?;
    params.margin_frac = parse_margin(margin)?;
    params.bluemarble = BlueMarble::FlatAlbedo; // WV is thermal; no ground texture needed
    params.ir_enhancement = match &enhancement {
        Some(e) => Some(parse_enhancement(e)?),
        None => None,
    };
    if let Some(c) = cache {
        params.cache = PathBuf::from(c);
    }
    apply_thread_cap(threads);
    let result = render_or_err(py, params, Product::WaterVapor { band: wv_band })?;
    warn_downgrades(py, &result, false);
    let (ny, nx) = (result.ny, result.nx);
    let geo = build_georef(py, &result)?;
    let (bt, rgb) = match result.data {
        FrameData::Ir { bt_kelvin, rgb } => (bt_kelvin, rgb),
        _ => return Err(runtime_err("expected a water-vapor frame")),
    };
    let bt_arr = Array2::from_shape_vec((ny, nx), bt)
        .map_err(|e| runtime_err(&format!("bt reshape: {e}")))?
        .into_pyarray(py);
    let geo_obj = Bound::new(py, geo)?;
    let tuple = match rgb {
        Some(rgb) => {
            let rgb_arr = Array3::from_shape_vec((ny, nx, 3), rgb)
                .map_err(|e| runtime_err(&format!("wv rgb reshape: {e}")))?
                .into_pyarray(py);
            PyTuple::new(
                py,
                [bt_arr.into_any(), rgb_arr.into_any(), geo_obj.into_any()],
            )?
        }
        None => PyTuple::new(py, [bt_arr.into_any(), geo_obj.into_any()])?,
    };
    Ok(tuple.into_any())
}

/// Render the RAW precipitable water (the vertically-integrated water-vapor column) in
/// MILLIMETRES (`mm == kg m^-2`).
///
/// Returns `(pw, georef)` where `pw` is a numpy `H x W` float32 array in mm (NaN off-domain;
/// row 0 = north). With `colormap=True` the return is instead `(pw, rgb, georef)` with `rgb` a
/// numpy `H x W x 3` uint8 basic moisture-ramp image (the raw `pw` array is the primary
/// deliverable — plot it with your own colormap/cartopy). A per-column integral, day AND night
/// (no sun input). Density is the standard-atmosphere exponential (the brick carries no
/// pressure).
#[pyfunction]
#[pyo3(signature = (
    input, *, storage_profile="compact-u8", sat="goes-east", view="topdown", timestep=0, resolution="native", margin=0.0,
    colormap=false, cache=None, threads=None
))]
#[allow(clippy::too_many_arguments)]
fn render_precipitable_water<'py>(
    py: Python<'py>,
    input: String,
    storage_profile: &str,
    sat: &str,
    view: &str,
    timestep: usize,
    resolution: &str,
    margin: f64,
    colormap: bool,
    cache: Option<String>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    render_derived_impl(
        py,
        DerivedField::PrecipitableWater,
        input,
        storage_profile,
        sat,
        view,
        timestep,
        resolution,
        margin,
        colormap,
        cache,
        threads,
    )
}

/// Render the RAW cloud-top temperature (the temperature at the effective cloud top, the
/// visible optical-depth ~1 level a satellite sees) in KELVIN.
///
/// Returns `(ctt, georef)` where `ctt` is a numpy `H x W` float32 array in Kelvin; a CLEAR /
/// optically-thin column is `NaN` (no cloud top). With `colormap=True` the return is instead
/// `(ctt, rgb, georef)` with `rgb` a numpy `H x W x 3` uint8 thermal (IR-rainbow) image. A
/// per-column march, day AND night (no sun input).
#[pyfunction]
#[pyo3(signature = (
    input, *, storage_profile="compact-u8", sat="goes-east", view="topdown", timestep=0, resolution="native", margin=0.0,
    colormap=false, cache=None, threads=None
))]
#[allow(clippy::too_many_arguments)]
fn render_cloud_top_temp<'py>(
    py: Python<'py>,
    input: String,
    storage_profile: &str,
    sat: &str,
    view: &str,
    timestep: usize,
    resolution: &str,
    margin: f64,
    colormap: bool,
    cache: Option<String>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    render_derived_impl(
        py,
        DerivedField::CloudTopTemp,
        input,
        storage_profile,
        sat,
        view,
        timestep,
        resolution,
        margin,
        colormap,
        cache,
        threads,
    )
}

/// Render the RAW cloud optical depth (the total-column visible optical depth,
/// dimensionless).
///
/// Returns `(cod, georef)` where `cod` is a numpy `H x W` float32 array (NaN off-domain;
/// clear column = 0; a thick storm core = tens+; row 0 = north). With `colormap=True` the
/// return is instead `(cod, rgb, georef)` with `rgb` a numpy `H x W x 3` uint8 white-to-dark
/// image. A per-column integral, day AND night (no sun input).
#[pyfunction]
#[pyo3(signature = (
    input, *, storage_profile="compact-u8", sat="goes-east", view="topdown", timestep=0, resolution="native", margin=0.0,
    colormap=false, cache=None, threads=None
))]
#[allow(clippy::too_many_arguments)]
fn render_cloud_optical_depth<'py>(
    py: Python<'py>,
    input: String,
    storage_profile: &str,
    sat: &str,
    view: &str,
    timestep: usize,
    resolution: &str,
    margin: f64,
    colormap: bool,
    cache: Option<String>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    render_derived_impl(
        py,
        DerivedField::CloudOpticalDepth,
        input,
        storage_profile,
        sat,
        view,
        timestep,
        resolution,
        margin,
        colormap,
        cache,
        threads,
    )
}

/// Render the WEB-MAP CLOUD LAYER pair (the Mapbox-class compositing product): the
/// CLOUD FIELD ONLY as an RGBA image plus the ground cloud-shadow MULTIPLY layer, both
/// on a north-up Web-Mercator (EPSG:3857) grid a web map can place directly.
///
/// Returns `(rgba, shadow, georef)`:
///   rgba: numpy `H x W x 4` uint8 (row 0 = north). STRAIGHT alpha by default (what a
///       PNG / browser / Mapbox raster source expects: composite `src*a + dst*(1-a)`).
///       Pass `premultiplied=True` for the engine's premultiplied array (composite
///       `src + dst*(1-a)` — the physically-exact additive form; see the simsat
///       `web_layer` docs for the difference on thin bright wisps).
///   shadow: numpy `H x W` float32 in `[0, 1]` — multiply your basemap by it
///       (1.0 = no shadow; out-of-domain = 1.0).
///   georef: `extent` in EPSG:3857 metres (`extent_kind='webmercator_meters'`,
///       `proj4` = the standard 3857 string) and `mercator_corners` = the four
///       `(lon, lat)` corner pairs in the Mapbox ImageSource `coordinates` order
///       (NW, NE, SE, SW).
///
/// TOP-DOWN by definition (there is no `view=` — the host map is the ground; no Blue
/// Marble is rendered). The sun / aerosol / exposure / steps / multiscatter /
/// beer-powder / granulation / fractional-cloud / cloud-optical-depth controls and the
/// default-on `feather_exposed_domain_edges` finite-domain presentation control drive the cloud
/// march exactly like `render_visible_rgb`;
/// `clouds=False` returns a transparent cloud image and neutral shadow. Datum note:
/// lat/lon are on the WRF
/// sphere fed through standard EPSG:3857 — the usual WRF-on-a-web-map approximation.
#[pyfunction]
#[pyo3(signature = (
    input, *, storage_profile="compact-u8", intent="display", sat="goes-east", timestep=0, margin=0.0, aerosol_optical_depth=0.05,
    rh_aerosol_swelling=false, atmosphere_correction=true, terrain_atmosphere=true,
    exposure=None, ground_gain=None, cloud_softclip=None, cloud_highlight_max=None,
    multiscatter=true, cloud_multiscatter=None, beer_powder=false,
    steps="offline", clouds=true,
    fractional_clouds=true, fractional_cloud_mode="effective-od", cloud_optical_depth_scale=0.15,
    cloud_optics="fixed",
    feather_exposed_domain_edges=true, granulation=false, sun_elev=None, sun_az=None, cache=None,
    premultiplied=false, threads=None
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn render_cloud_layer<'py>(
    py: Python<'py>,
    input: String,
    storage_profile: &str,
    intent: &str,
    sat: &str,
    timestep: usize,
    margin: f64,
    aerosol_optical_depth: f32,
    rh_aerosol_swelling: bool,
    atmosphere_correction: bool,
    terrain_atmosphere: bool,
    exposure: Option<f64>,
    ground_gain: Option<f64>,
    cloud_softclip: Option<f64>,
    cloud_highlight_max: Option<f64>,
    multiscatter: bool,
    cloud_multiscatter: Option<&str>,
    beer_powder: bool,
    steps: &str,
    clouds: bool,
    fractional_clouds: bool,
    fractional_cloud_mode: &str,
    cloud_optical_depth_scale: f32,
    cloud_optics: &str,
    feather_exposed_domain_edges: bool,
    granulation: bool,
    sun_elev: Option<f64>,
    sun_az: Option<f64>,
    cache: Option<String>,
    premultiplied: bool,
    threads: Option<usize>,
) -> PyResult<(Bound<'py, PyArray3<u8>>, Bound<'py, PyArray2<f32>>, Georef)> {
    let mut params = RenderParams::new(PathBuf::from(&input));
    force_surface_recovery_identity(&mut params);
    params.storage_profile = parse_storage_profile(storage_profile)?;
    params.intent = parse_intent(intent)?;
    params.satellite = parse_sat(sat)?;
    params.timestep = timestep;
    params.margin_frac = parse_margin(margin)?;
    apply_visible_physics_controls(
        &mut params,
        aerosol_optical_depth,
        rh_aerosol_swelling,
        atmosphere_correction,
        terrain_atmosphere,
        fractional_clouds,
        fractional_cloud_mode,
        cloud_optical_depth_scale,
        cloud_optics,
        feather_exposed_domain_edges,
        beer_powder,
        granulation,
    )?;
    apply_visible_display_controls(
        &mut params,
        exposure,
        ground_gain,
        cloud_softclip,
        cloud_highlight_max,
    );
    params.multiscatter = multiscatter;
    params.cloud_multiscatter = cloud_multiscatter
        .map(parse_cloud_multiscatter)
        .transpose()?;
    params.steps = parse_steps(steps)?;
    params.clouds = clouds;
    params.sun_override = if sun_elev.is_some() || sun_az.is_some() {
        Some(SunOverride {
            elev_deg: sun_elev,
            az_deg: sun_az,
        })
    } else {
        None
    };
    if let Some(c) = cache {
        params.cache = PathBuf::from(c);
    }
    // No ground is rendered (the host map is the ground) — never touch the Blue Marble.
    params.bluemarble = BlueMarble::FlatAlbedo;
    apply_thread_cap(threads);
    let result = render_or_err(py, params, Product::CloudLayer)?;
    warn_downgrades(py, &result, true);
    let (ny, nx) = (result.ny, result.nx);
    let geo = build_georef(py, &result)?;
    let (rgba_premul, shadow) = match result.data {
        FrameData::CloudLayer {
            rgba_premul,
            shadow,
        } => (rgba_premul, shadow),
        _ => return Err(runtime_err("expected a cloud-layer frame")),
    };
    let rgba = if premultiplied {
        rgba_premul
    } else {
        web_layer::unpremultiply_rgba(&rgba_premul)
    };
    let rgba_arr = Array3::from_shape_vec((ny, nx, 4), rgba)
        .map_err(|e| runtime_err(&format!("rgba reshape: {e}")))?
        .into_pyarray(py);
    let shadow_arr = Array2::from_shape_vec((ny, nx), shadow)
        .map_err(|e| runtime_err(&format!("shadow reshape: {e}")))?
        .into_pyarray(py);
    Ok((rgba_arr, shadow_arr, geo))
}

/// Render a FREE-PERSPECTIVE frame (the angled-3D "flyover" product): a pinhole camera
/// at `eye=(lat, lon, alt_m)` looking at `look=(lat, lon, alt_m)` with a HORIZONTAL
/// field of view `fov` (deg) over a `size=(width, height)` image, fed through the SAME
/// surface + cloud marches as every other product.
///
/// Returns `(rgb, georef)` — the FULL COMPOSITE over the Blue Marble ground (numpy
/// `H x W x 3` uint8; sky rays composite the atmosphere limb, space is black). With
/// `cloud_layer_only=True` it instead returns `(rgba, georef)` — the CLOUD FIELD ONLY
/// as premultiplied-alpha `H x W x 4` uint8 (alpha = 1 - cloud transmittance), for
/// compositing over a host 3-D map rendered with a matching camera.
///
/// `georef.camera_pose` always carries the camera dict (a perspective frame states its
/// camera — the what-if discipline); `georef.lat`/`lon` are the per-pixel GROUND
/// intersections (NaN for sky rays) — a perspective frame is a picture, not a map, so
/// use the mesh for georeferencing rather than the extent. Parallax displacement of
/// high cloud against the ground is physical and intended (the 3D look). A FLYOVER is
/// simply N calls along your own eye/look path (each frame is an independent render).
#[pyfunction]
#[pyo3(signature = (
    input, *, eye, look, storage_profile="compact-u8", intent="display", fov=40.0, size=(1280, 720), timestep=0,
    aerosol_optical_depth=0.05, rh_aerosol_swelling=false, atmosphere_correction=true,
    terrain_atmosphere=true, land_sza_normalization=true, land_sza_max_gain=4.0,
    land_dark_toe=true, land_dark_toe_knee=0.08, land_dark_toe_gamma=0.65,
    land_dark_toe_max_gain=1.5, surface_postlight_toe=false,
    surface_postlight_toe_knee=0.18, surface_postlight_toe_gamma=0.80,
    surface_postlight_toe_max_gain=1.35, twilight_surface_recovery=true,
    twilight_surface_recovery_knee=0.30, twilight_surface_recovery_gamma=0.50,
    twilight_surface_recovery_max_gain=4.0, exposure=None, ground_gain=None, cloud_softclip=None,
    cloud_highlight_max=None, multiscatter=true, cloud_multiscatter=None, beer_powder=false,
    steps="offline", clouds=true, fractional_clouds=true, fractional_cloud_mode="effective-od", cloud_optical_depth_scale=0.15,
    cloud_optics="fixed",
    feather_exposed_domain_edges=true, granulation=false,
    cloud_layer_only=false, sun_elev=None, sun_az=None,
    cache=None, bluemarble=None, bluemarble_month=None, bluemarble_download=true, threads=None
))]
#[allow(clippy::too_many_arguments)]
fn render_perspective<'py>(
    py: Python<'py>,
    input: String,
    eye: (f64, f64, f64),
    look: (f64, f64, f64),
    storage_profile: &str,
    intent: &str,
    fov: f64,
    size: (usize, usize),
    timestep: usize,
    aerosol_optical_depth: f32,
    rh_aerosol_swelling: bool,
    atmosphere_correction: bool,
    terrain_atmosphere: bool,
    land_sza_normalization: bool,
    land_sza_max_gain: f64,
    land_dark_toe: bool,
    land_dark_toe_knee: f64,
    land_dark_toe_gamma: f64,
    land_dark_toe_max_gain: f64,
    surface_postlight_toe: bool,
    surface_postlight_toe_knee: f64,
    surface_postlight_toe_gamma: f64,
    surface_postlight_toe_max_gain: f64,
    twilight_surface_recovery: bool,
    twilight_surface_recovery_knee: f64,
    twilight_surface_recovery_gamma: f64,
    twilight_surface_recovery_max_gain: f64,
    exposure: Option<f64>,
    ground_gain: Option<f64>,
    cloud_softclip: Option<f64>,
    cloud_highlight_max: Option<f64>,
    multiscatter: bool,
    cloud_multiscatter: Option<&str>,
    beer_powder: bool,
    steps: &str,
    clouds: bool,
    fractional_clouds: bool,
    fractional_cloud_mode: &str,
    cloud_optical_depth_scale: f32,
    cloud_optics: &str,
    feather_exposed_domain_edges: bool,
    granulation: bool,
    cloud_layer_only: bool,
    sun_elev: Option<f64>,
    sun_az: Option<f64>,
    cache: Option<String>,
    bluemarble: Option<String>,
    bluemarble_month: Option<u32>,
    bluemarble_download: bool,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut params = RenderParams::new(PathBuf::from(&input));
    params.storage_profile = parse_storage_profile(storage_profile)?;
    params.intent = parse_intent(intent)?;
    params.timestep = timestep;
    apply_visible_physics_controls(
        &mut params,
        aerosol_optical_depth,
        rh_aerosol_swelling,
        atmosphere_correction,
        terrain_atmosphere,
        fractional_clouds,
        fractional_cloud_mode,
        cloud_optical_depth_scale,
        cloud_optics,
        feather_exposed_domain_edges,
        beer_powder,
        granulation,
    )?;
    apply_visible_display_controls(
        &mut params,
        exposure,
        ground_gain,
        cloud_softclip,
        cloud_highlight_max,
    );
    apply_land_appearance_controls(
        &mut params,
        land_sza_normalization,
        land_sza_max_gain,
        land_dark_toe,
        land_dark_toe_knee,
        land_dark_toe_gamma,
        land_dark_toe_max_gain,
    )?;
    apply_surface_postlight_toe_controls(
        &mut params,
        surface_postlight_toe,
        surface_postlight_toe_knee,
        surface_postlight_toe_gamma,
        surface_postlight_toe_max_gain,
    )?;
    apply_twilight_surface_recovery_controls(
        &mut params,
        twilight_surface_recovery,
        twilight_surface_recovery_knee,
        twilight_surface_recovery_gamma,
        twilight_surface_recovery_max_gain,
    )?;
    if cloud_layer_only {
        force_surface_recovery_identity(&mut params);
    }
    params.multiscatter = multiscatter;
    params.cloud_multiscatter = cloud_multiscatter
        .map(parse_cloud_multiscatter)
        .transpose()?;
    params.steps = parse_steps(steps)?;
    params.clouds = clouds;
    params.sun_override = if sun_elev.is_some() || sun_az.is_some() {
        Some(SunOverride {
            elev_deg: sun_elev,
            az_deg: sun_az,
        })
    } else {
        None
    };
    if let Some(c) = cache {
        params.cache = PathBuf::from(c);
    }
    params.bluemarble = match bluemarble {
        Some(path) => BlueMarble::SingleFile(PathBuf::from(path)),
        None => BlueMarble::Seasonal {
            month_override: bluemarble_month,
            download: bluemarble_download,
        },
    };
    params.perspective = Some(simsat_engine::camera::PerspectiveCamera {
        eye_lat_deg: eye.0,
        eye_lon_deg: eye.1,
        eye_alt_m: eye.2,
        look_lat_deg: look.0,
        look_lon_deg: look.1,
        look_alt_m: look.2,
        fov_deg: fov,
        width: size.0,
        height: size.1,
    });
    apply_thread_cap(threads);
    let result = render_or_err(py, params, Product::Perspective { cloud_layer_only })?;
    warn_downgrades(py, &result, true);
    let (ny, nx) = (result.ny, result.nx);
    let geo = build_georef(py, &result)?;
    let (rgb, rgba) = match result.data {
        FrameData::Visible { rgb, rgba } => (rgb, rgba),
        _ => return Err(runtime_err("expected a perspective (visible) frame")),
    };
    let geo_obj = Bound::new(py, geo)?;
    let tuple = if cloud_layer_only {
        let arr = Array3::from_shape_vec((ny, nx, 4), rgba)
            .map_err(|e| runtime_err(&format!("perspective rgba reshape: {e}")))?
            .into_pyarray(py);
        PyTuple::new(py, [arr.into_any(), geo_obj.into_any()])?
    } else {
        let arr = Array3::from_shape_vec((ny, nx, 3), rgb)
            .map_err(|e| runtime_err(&format!("perspective rgb reshape: {e}")))?
            .into_pyarray(py);
        PyTuple::new(py, [arr.into_any(), geo_obj.into_any()])?
    };
    Ok(tuple.into_any())
}

/// Enable or disable the engine's diagnostic stderr lines (ingest progress / warnings)
/// for this process at runtime.
///
/// The binding is SILENT BY DEFAULT — a library must not chatter uninvited — so the
/// engine's diagnostic sink is disabled at `import simsat` unless the `SIMSAT_LOG`
/// environment variable is truthy (`1`/`true`). Call `simsat.set_verbose(True)` to see
/// the lines (they go to STDERR), `set_verbose(False)` to silence them again.
///
/// This gates DIAGNOSTIC chatter only. Render-honesty surfacing is data, not logs, and
/// is unaffected: `georef.time_is_fallback` / `ground_source` / `ground_status` and
/// their `UserWarning`s are raised regardless of this switch.
#[pyfunction]
fn set_verbose(enabled: bool) {
    simsat_engine::log::set_enabled(enabled);
}

/// Return the renderer's frame-sun elevation at every latitude/longitude pixel.
///
/// `lat` and `lon` must be same-shaped two-dimensional `float32` numpy arrays. The
/// calculation deliberately matches the surface-light LUT that drives terrain and
/// cloud compositing: per-pixel NOAA elevation normally; with an override, NOAA at the
/// finite-coordinate bounding-box centre is partially overridden, converted once to
/// ECEF, then projected into each pixel's local ENU frame. An omitted override component
/// keeps its real centre value. Invalid/off-earth coordinate pairs return `NaN`. The
/// atmosphere/cloud frame sun is a separate single ECEF vector.
#[pyfunction]
#[pyo3(signature = (time_iso, lat, lon, sun_elev=None, sun_az=None))]
fn solar_elevation_grid<'py>(
    py: Python<'py>,
    time_iso: &str,
    lat: PyReadonlyArray2<'py, f32>,
    lon: PyReadonlyArray2<'py, f32>,
    sun_elev: Option<f64>,
    sun_az: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let lat_view = lat.as_array();
    let lon_view = lon.as_array();
    if lat_view.shape() != lon_view.shape() {
        return Err(value_err(format!(
            "lat and lon must have the same shape (got {:?} and {:?})",
            lat_view.shape(),
            lon_view.shape()
        )));
    }
    let shape = (lat_view.shape()[0], lat_view.shape()[1]);
    if shape.0 == 0 || shape.1 == 0 {
        return Err(value_err(
            "lat and lon grids must have non-zero height and width".to_string(),
        ));
    }

    let (year, month, day, ut) = parse_valid_utc(time_iso).ok_or_else(|| {
        value_err(format!(
            "time_iso must be a valid UTC timestamp like 1974-04-03T23:12:00Z; got '{time_iso}'"
        ))
    })?;
    let solar = SolarFrame::new(year, month, day, ut);
    let lat_values: Vec<f32> = lat_view.iter().copied().collect();
    let lon_values: Vec<f32> = lon_view.iter().copied().collect();
    let values = simsat_engine::solar::solar_elevation_grid(
        &solar,
        &lat_values,
        &lon_values,
        sun_elev,
        sun_az,
    )
    .map_err(value_err)?;
    Ok(Array2::from_shape_vec(shape, values)
        .map_err(|e| runtime_err(&format!("solar elevation reshape: {e}")))?
        .into_pyarray(py))
}

fn parse_valid_utc(value: &str) -> Option<(i32, u32, u32, f64)> {
    let value = value.trim();
    let value = value.strip_suffix('Z').unwrap_or(value);
    let (date, time) = value.split_once(['T', '_'])?;
    let date: Vec<_> = date.split('-').collect();
    let clock: Vec<_> = time.split(':').collect();
    if date.len() != 3 || !(2..=3).contains(&clock.len()) {
        return None;
    }
    let year: i32 = date[0].parse().ok()?;
    let month: u32 = date[1].parse().ok()?;
    let day: u32 = date[2].parse().ok()?;
    let hour: u32 = clock[0].parse().ok()?;
    let minute: u32 = clock[1].parse().ok()?;
    let second: f64 = clock.get(2).map_or(Some(0.0), |v| v.parse().ok())?;
    if hour >= 24 || minute >= 60 || !second.is_finite() || !(0.0..60.0).contains(&second) {
        return None;
    }
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return None,
    };
    if !(1..=days).contains(&day) {
        return None;
    }
    Some((
        year,
        month,
        day,
        hour as f64 + minute as f64 / 60.0 + second / 3600.0,
    ))
}

/// `SIMSAT_LOG` truthiness: `1` or `true` (case-insensitive) opts the process into the
/// engine's diagnostic stderr lines at import time.
fn env_verbose_opt_in() -> bool {
    std::env::var("SIMSAT_LOG")
        .map(|v| {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true")
        })
        .unwrap_or(false)
}

// ── helpers ─────────────────────────────────────────────────────────────────────

/// Shared body for the three derived-field bindings: parse args, run
/// [`Product::Derived`], and marshal `(values, [rgb,] georef)` into numpy. The raw `H x W`
/// float32 field is always returned; the basic colormap RGB is added only when `colormap`.
#[allow(clippy::too_many_arguments)]
fn render_derived_impl<'py>(
    py: Python<'py>,
    field: DerivedField,
    input: String,
    storage_profile: &str,
    sat: &str,
    view: &str,
    timestep: usize,
    resolution: &str,
    margin: f64,
    colormap: bool,
    cache: Option<String>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut params = RenderParams::new(PathBuf::from(&input));
    force_surface_recovery_identity(&mut params);
    params.storage_profile = parse_storage_profile(storage_profile)?;
    params.satellite = parse_sat(sat)?;
    params.view = parse_view(view)?;
    params.timestep = timestep;
    params.resolution = parse_resolution(resolution)?;
    params.margin_frac = parse_margin(margin)?;
    params.bluemarble = BlueMarble::FlatAlbedo; // a column integral; no ground texture needed
    params.derived_colormap = colormap;
    if let Some(c) = cache {
        params.cache = PathBuf::from(c);
    }
    apply_thread_cap(threads);
    let result = render_or_err(py, params, Product::Derived { field })?;
    warn_downgrades(py, &result, false);
    let (ny, nx) = (result.ny, result.nx);
    let geo = build_georef(py, &result)?;
    let (values, rgb) = match result.data {
        FrameData::Scalar { values, rgb, .. } => (values, rgb),
        _ => return Err(runtime_err("expected a derived scalar frame")),
    };
    let vals_arr = Array2::from_shape_vec((ny, nx), values)
        .map_err(|e| runtime_err(&format!("values reshape: {e}")))?
        .into_pyarray(py);
    let geo_obj = Bound::new(py, geo)?;
    let tuple = match rgb {
        Some(rgb) => {
            let rgb_arr = Array3::from_shape_vec((ny, nx, 3), rgb)
                .map_err(|e| runtime_err(&format!("derived rgb reshape: {e}")))?
                .into_pyarray(py);
            PyTuple::new(
                py,
                [vals_arr.into_any(), rgb_arr.into_any(), geo_obj.into_any()],
            )?
        }
        None => PyTuple::new(py, [vals_arr.into_any(), geo_obj.into_any()])?,
    };
    Ok(tuple.into_any())
}

/// Run the render with the GIL released (the CPU march is pure Rust), mapping a render
/// error to a Python `RuntimeError`.
fn render_or_err(py: Python<'_>, params: RenderParams, product: Product) -> PyResult<RenderResult> {
    py.allow_threads(move || api::render(&params, product))
        .map_err(|e| runtime_err(&e))
}

#[allow(clippy::too_many_arguments)]
fn build_visible_params(
    input: String,
    storage_profile: &str,
    intent: &str,
    sat: &str,
    geo_navigation: &str,
    view: &str,
    timestep: usize,
    resolution: &str,
    margin: f64,
    aerosol_optical_depth: f32,
    rh_aerosol_swelling: bool,
    atmosphere_correction: bool,
    terrain_atmosphere: bool,
    land_sza_normalization: bool,
    land_sza_max_gain: f64,
    land_dark_toe: bool,
    land_dark_toe_knee: f64,
    land_dark_toe_gamma: f64,
    land_dark_toe_max_gain: f64,
    surface_postlight_toe: bool,
    surface_postlight_toe_knee: f64,
    surface_postlight_toe_gamma: f64,
    surface_postlight_toe_max_gain: f64,
    twilight_surface_recovery: bool,
    twilight_surface_recovery_knee: f64,
    twilight_surface_recovery_gamma: f64,
    twilight_surface_recovery_max_gain: f64,
    exposure: Option<f64>,
    ground_gain: Option<f64>,
    cloud_softclip: Option<f64>,
    cloud_highlight_max: Option<f64>,
    multiscatter: bool,
    cloud_multiscatter: Option<&str>,
    beer_powder: bool,
    steps: &str,
    clouds: bool,
    fractional_clouds: bool,
    fractional_cloud_mode: &str,
    cloud_optical_depth_scale: f32,
    cloud_optics: &str,
    feather_exposed_domain_edges: bool,
    granulation: bool,
    topdown_stratiform_regularization: bool,
    topdown_cloud_footprint: bool,
    topdown_shadow_antialias: bool,
    sun_elev: Option<f64>,
    sun_az: Option<f64>,
    cache: Option<String>,
    bluemarble: Option<String>,
    bluemarble_month: Option<u32>,
    bluemarble_download: bool,
) -> PyResult<RenderParams> {
    let mut params = RenderParams::new(PathBuf::from(&input));
    params.storage_profile = parse_storage_profile(storage_profile)?;
    params.intent = parse_intent(intent)?;
    params.satellite = parse_sat(sat)?;
    params.geo_navigation = parse_geo_navigation(geo_navigation)?;
    params.view = parse_view(view)?;
    params.timestep = timestep;
    params.resolution = parse_resolution(resolution)?;
    params.margin_frac = parse_margin(margin)?;
    apply_visible_physics_controls(
        &mut params,
        aerosol_optical_depth,
        rh_aerosol_swelling,
        atmosphere_correction,
        terrain_atmosphere,
        fractional_clouds,
        fractional_cloud_mode,
        cloud_optical_depth_scale,
        cloud_optics,
        feather_exposed_domain_edges,
        beer_powder,
        granulation,
    )?;
    apply_visible_display_controls(
        &mut params,
        exposure,
        ground_gain,
        cloud_softclip,
        cloud_highlight_max,
    );
    apply_land_appearance_controls(
        &mut params,
        land_sza_normalization,
        land_sza_max_gain,
        land_dark_toe,
        land_dark_toe_knee,
        land_dark_toe_gamma,
        land_dark_toe_max_gain,
    )?;
    apply_surface_postlight_toe_controls(
        &mut params,
        surface_postlight_toe,
        surface_postlight_toe_knee,
        surface_postlight_toe_gamma,
        surface_postlight_toe_max_gain,
    )?;
    apply_twilight_surface_recovery_controls(
        &mut params,
        twilight_surface_recovery,
        twilight_surface_recovery_knee,
        twilight_surface_recovery_gamma,
        twilight_surface_recovery_max_gain,
    )?;
    params.multiscatter = multiscatter;
    params.cloud_multiscatter = cloud_multiscatter
        .map(parse_cloud_multiscatter)
        .transpose()?;
    params.steps = parse_steps(steps)?;
    params.clouds = clouds;
    params.topdown_stratiform_regularization = topdown_stratiform_regularization;
    params.topdown_cloud_footprint = topdown_cloud_footprint;
    params.topdown_shadow_antialias = topdown_shadow_antialias;
    params.sun_override = if sun_elev.is_some() || sun_az.is_some() {
        Some(SunOverride {
            elev_deg: sun_elev,
            az_deg: sun_az,
        })
    } else {
        None
    };
    if let Some(c) = cache {
        params.cache = PathBuf::from(c);
    }
    params.bluemarble = match bluemarble {
        Some(path) => BlueMarble::SingleFile(PathBuf::from(path)),
        None => BlueMarble::Seasonal {
            month_override: bluemarble_month,
            download: bluemarble_download,
        },
    };
    Ok(params)
}

/// Validate and apply the common visible-atmosphere/cloud calibration controls. Keeping
/// this in one helper makes the Python defaults and bounds identical across RGB, raw bands,
/// GeoColor, Sandwich, cloud-layer, and perspective entry points.
#[allow(clippy::too_many_arguments)]
fn apply_visible_physics_controls(
    params: &mut RenderParams,
    aerosol_optical_depth: f32,
    rh_aerosol_swelling: bool,
    atmosphere_correction: bool,
    terrain_atmosphere: bool,
    fractional_clouds: bool,
    fractional_cloud_mode: &str,
    cloud_optical_depth_scale: f32,
    cloud_optics: &str,
    feather_exposed_domain_edges: bool,
    beer_powder: bool,
    granulation: bool,
) -> PyResult<()> {
    if !aerosol_optical_depth.is_finite() || !(0.0..=0.6).contains(&aerosol_optical_depth) {
        return Err(value_err(format!(
            "aerosol_optical_depth must be finite and in 0.0..=0.6, got \
             {aerosol_optical_depth}"
        )));
    }
    if !cloud_optical_depth_scale.is_finite() || !(0.0..=4.0).contains(&cloud_optical_depth_scale) {
        return Err(value_err(format!(
            "cloud_optical_depth_scale must be finite and in 0.0..=4.0, got \
             {cloud_optical_depth_scale}"
        )));
    }
    params.aerosol_optical_depth = aerosol_optical_depth;
    params.rh_aerosol_swelling = rh_aerosol_swelling;
    params.atmosphere_correction = atmosphere_correction;
    params.terrain_atmosphere = terrain_atmosphere;
    let mode = parse_fractional_cloud_mode(fractional_cloud_mode)?;
    params.fractional_clouds = fractional_clouds && mode != FractionalCloudMode::Off;
    params.fractional_cloud_mode = if params.fractional_clouds {
        mode
    } else {
        FractionalCloudMode::Off
    };
    params.cloud_optical_depth_scale = cloud_optical_depth_scale;
    params.cloud_optics = CloudOpticsMode::parse(cloud_optics).ok_or_else(|| {
        value_err(format!(
            "cloud_optics must be 'fixed', 'nssl-native', or 'hrrr-thompson-native', got '{cloud_optics}'"
        ))
    })?;
    params.feather_exposed_domain_edges = feather_exposed_domain_edges;
    params.beer_powder = beer_powder;
    params.granulation = Some(granulation);
    Ok(())
}

/// Apply optional finished-display calibration controls. `None` preserves the engine's
/// shipped constant, while an explicit value is recorded on `RenderParams`. These are
/// intentionally separate from the physical controls because raw visible bands expose
/// none of these arguments and pass `None` for every value, remaining pre-tonemap diagnostics.
fn apply_visible_display_controls(
    params: &mut RenderParams,
    exposure: Option<f64>,
    ground_gain: Option<f64>,
    cloud_softclip: Option<f64>,
    cloud_highlight_max: Option<f64>,
) {
    if let Some(value) = exposure {
        params.exposure = value;
    }
    params.ground_gain = ground_gain;
    params.cloud_softclip = cloud_softclip;
    params.cloud_highlight_max = cloud_highlight_max;
}

/// Validate and apply the display-only land controls. The shipped preset enables both;
/// explicitly disabling both is the exact legacy path. Bounds match the CLI and Studio.
#[allow(clippy::too_many_arguments)]
fn apply_land_appearance_controls(
    params: &mut RenderParams,
    sza_normalization: bool,
    sza_max_gain: f64,
    dark_toe: bool,
    dark_toe_knee: f64,
    dark_toe_gamma: f64,
    dark_toe_max_gain: f64,
) -> PyResult<()> {
    let bounded = |name: &str, value: f64, lo: f64, hi: f64| -> PyResult<f64> {
        if !value.is_finite() || !(lo..=hi).contains(&value) {
            return Err(value_err(format!(
                "{name} must be finite and in {lo}..={hi}, got {value}"
            )));
        }
        Ok(value)
    };
    params.land_appearance = simsat_engine::render::LandAppearanceConfig {
        sza_normalization,
        sza_max_gain: bounded("land_sza_max_gain", sza_max_gain, 1.0, 4.0)?,
        dark_toe,
        dark_toe_knee: bounded("land_dark_toe_knee", dark_toe_knee, 1.0e-6, 1.0)?,
        dark_toe_gamma: bounded("land_dark_toe_gamma", dark_toe_gamma, 0.05, 1.0)?,
        dark_toe_max_gain: bounded("land_dark_toe_max_gain", dark_toe_max_gain, 1.0, 4.0)?,
    };
    Ok(())
}

/// Validate and apply the default-off display-only terrain signal experiment. Bounds
/// match the CLI and Studio; disabled still validates its persisted/ready A/B values.
fn apply_surface_postlight_toe_controls(
    params: &mut RenderParams,
    enabled: bool,
    knee: f64,
    gamma: f64,
    max_gain: f64,
) -> PyResult<()> {
    let bounded = |name: &str, value: f64, lo: f64, hi: f64| -> PyResult<f64> {
        if !value.is_finite() || !(lo..=hi).contains(&value) {
            return Err(value_err(format!(
                "{name} must be finite and in {lo}..={hi}, got {value}"
            )));
        }
        Ok(value)
    };
    params.surface_postlight_toe = SurfacePostlightToeConfig {
        enabled,
        knee: bounded("surface_postlight_toe_knee", knee, 1.0e-6, 1.0)?,
        gamma: bounded("surface_postlight_toe_gamma", gamma, 0.05, 1.0)?,
        max_gain: bounded("surface_postlight_toe_max_gain", max_gain, 1.0, 4.0)?,
    };
    Ok(())
}

/// Validate and apply the independent, tightly gated low-sun terrain recovery.
fn apply_twilight_surface_recovery_controls(
    params: &mut RenderParams,
    enabled: bool,
    knee: f64,
    gamma: f64,
    max_gain: f64,
) -> PyResult<()> {
    let bounded = |name: &str, value: f64, lo: f64, hi: f64| -> PyResult<f64> {
        if !value.is_finite() || !(lo..=hi).contains(&value) {
            return Err(value_err(format!(
                "{name} must be finite and in {lo}..={hi}, got {value}"
            )));
        }
        Ok(value)
    };
    params.twilight_surface_recovery = TwilightSurfaceRecoveryConfig {
        enabled,
        knee: bounded("twilight_surface_recovery_knee", knee, 1.0e-6, 1.0)?,
        gamma: bounded("twilight_surface_recovery_gamma", gamma, 0.05, 1.0)?,
        max_gain: bounded("twilight_surface_recovery_max_gain", max_gain, 1.0, 4.0)?,
    };
    Ok(())
}

/// Keep products that do not consume finished-visible terrain appearance on an explicit
/// identity request, even though the engine product seams also ignore these controls.
fn force_surface_recovery_identity(params: &mut RenderParams) {
    params.surface_postlight_toe = SurfacePostlightToeConfig::off();
    params.twilight_surface_recovery = TwilightSurfaceRecoveryConfig::off();
}

/// Build the Python [`Georef`] from a render result: scalars + the projection dict + the
/// lat/lon numpy mesh.
fn build_georef(py: Python<'_>, result: &RenderResult) -> PyResult<Georef> {
    let g = &result.georef;
    let (ny, nx) = (g.ny, g.nx);
    let lat = Array2::from_shape_vec((ny, nx), g.lat.clone())
        .map_err(|e| runtime_err(&format!("lat reshape: {e}")))?
        .into_pyarray(py)
        .unbind();
    let lon = Array2::from_shape_vec((ny, nx), g.lon.clone())
        .map_err(|e| runtime_err(&format!("lon reshape: {e}")))?
        .into_pyarray(py)
        .unbind();
    let extent_kind = match g.extent_kind {
        ExtentKind::ProjectionMeters => "projection_meters",
        ExtentKind::LonLatDegrees => "lonlat_degrees",
        ExtentKind::WebMercatorMeters => "webmercator_meters",
    };
    let crs_params = build_crs_params(py, g)?;
    let camera_pose = match &g.camera_pose {
        Some(c) => {
            let d = PyDict::new(py);
            d.set_item("eye_lat", c.eye_lat_deg)?;
            d.set_item("eye_lon", c.eye_lon_deg)?;
            d.set_item("eye_alt_m", c.eye_alt_m)?;
            d.set_item("look_lat", c.look_lat_deg)?;
            d.set_item("look_lon", c.look_lon_deg)?;
            d.set_item("look_alt_m", c.look_alt_m)?;
            d.set_item("fov_deg", c.fov_deg)?;
            d.set_item("width", c.width)?;
            d.set_item("height", c.height)?;
            Some(d.unbind())
        }
        None => None,
    };
    let geo_navigation = g
        .geo_navigation
        .map(|nav| nav.navigation.slug().to_string());
    let geo_navigation_geometry = match g.geo_navigation {
        Some(nav) => {
            let d = PyDict::new(py);
            d.set_item("navigation", nav.navigation.slug())?;
            d.set_item("perspective_point_height_m", nav.perspective_point_height_m)?;
            d.set_item("semi_major_axis_m", nav.semi_major_axis_m)?;
            d.set_item("semi_minor_axis_m", nav.semi_minor_axis_m)?;
            d.set_item("fixed_grid_origin_lon_deg", nav.sub_lon_deg)?;
            d.set_item("model_camera_sub_lon_deg", nav.model_sub_lon_deg)?;
            d.set_item("sweep_angle_axis", nav.sweep_angle_axis)?;
            Some(d.unbind())
        }
        None => None,
    };
    let footprint_meta = result.instrument_footprint.metadata();
    let instrument_footprint_metadata = if result.instrument_footprint == InstrumentFootprint::Off {
        None
    } else {
        let d = PyDict::new(py);
        d.set_item("slug", footprint_meta.slug)?;
        d.set_item("label", footprint_meta.label)?;
        d.set_item("channel", footprint_meta.channel)?;
        d.set_item("domain", footprint_meta.domain)?;
        d.set_item("sample_angle_urad", footprint_meta.sample_angle_urad)?;
        d.set_item("source_url", footprint_meta.source_url)?;
        d.set_item("limitation", footprint_meta.limitation)?;
        Some(d.unbind())
    };
    let abi_fixed_grid_crop = result
        .raster
        .scan
        .abi_2km_global_indices()
        .map(|(x_min, x_max, y_min, y_max)| -> PyResult<Py<PyDict>> {
            let d = PyDict::new(py);
            d.set_item("sample_angle_urad", 56.0)?;
            d.set_item("ssp_is_four_pixel_corner", true)?;
            d.set_item("x_index_min", x_min)?;
            d.set_item("x_index_max", x_max)?;
            d.set_item("y_index_min", y_min)?;
            d.set_item("y_index_max", y_max)?;
            d.set_item("nx", result.raster.scan.nx)?;
            d.set_item("ny", result.raster.scan.ny)?;
            d.set_item("x_min_rad", result.raster.scan.x_min)?;
            d.set_item("y_max_rad", result.raster.scan.y_max)?;
            Ok(d.unbind())
        })
        .transpose()?;
    Ok(Georef {
        // A perspective frame is its own view kind (internally it reuses the geo
        // extent semantics; the pose dict is the discriminator).
        view: if g.camera_pose.is_some() {
            "perspective".to_string()
        } else {
            g.view.slug().to_string()
        },
        extent: (g.extent[0], g.extent[1], g.extent[2], g.extent[3]),
        extent_kind: extent_kind.to_string(),
        proj4: g.proj4(),
        proj_kind: g.proj_kind().to_string(),
        crs_params,
        lat,
        lon,
        time_is_fallback: result.time_is_fallback,
        ground_source: result
            .ground_source
            .as_ref()
            .map(|s| s.slug().to_string())
            .unwrap_or_else(|| "none".to_string()),
        ground_status: result.ground_status.clone(),
        intent: result.intent.slug().to_string(),
        observation_operator: result.observation_operator.to_string(),
        intent_adjustments: result
            .intent_adjustments
            .iter()
            .map(|a| a.label().to_string())
            .collect(),
        intent_limitations: result
            .intent
            .limitations()
            .iter()
            .map(|s| (*s).to_string())
            .collect(),
        thermal_sensor: result.thermal_sensor.map(|s| s.slug().to_string()),
        instrument_footprint: result.instrument_footprint.slug().to_string(),
        instrument_footprint_metadata,
        abi_fixed_grid_crop,
        science_warnings: result.science_warnings.clone(),
        storage_profile: result.storage_profile.slug().to_string(),
        mercator_corners: g
            .mercator_corners_lonlat
            .map(|c| c.iter().map(|p| (p[0], p[1])).collect()),
        camera_pose,
        geo_navigation,
        geo_navigation_geometry,
    })
}

/// Raise Python `UserWarning`s for honesty downgrades the caller would otherwise never
/// see in the returned arrays. The same facts ride on the [`Georef`] attributes
/// (`time_is_fallback`, `ground_source`, `ground_status`) for programmatic use.
fn warn_downgrades(py: Python<'_>, result: &RenderResult, sun_dependent: bool) {
    let mut msgs: Vec<String> = Vec::new();
    for adjustment in &result.diagnostics {
        msgs.push(format!(
            "simsat: GPU preview temporary adjustment: {}",
            adjustment.label()
        ));
    }
    for adjustment in &result.intent_adjustments {
        msgs.push(format!(
            "simsat: intent={} temporary adjustment: {}",
            result.intent.slug(),
            adjustment.label()
        ));
    }
    msgs.extend(
        result
            .science_warnings
            .iter()
            .map(|warning| format!("simsat: science limitation: {warning}")),
    );
    if sun_dependent && result.time_is_fallback {
        msgs.push(
            "simsat: the source carried no parseable valid time; the sun position and \
             Blue Marble season use the FABRICATED fallback date 2004-06-21 12:00 UT \
             (georef.time_is_fallback is True)"
                .to_string(),
        );
    }
    match &result.ground_source {
        Some(GroundSource::EightKmFallback) => msgs.push(
            "simsat: seasonal Blue Marble 2 km month(s) unavailable; the ground uses the \
             coarser vendored 8 km fallback (georef.ground_source = '8km-fallback')"
                .to_string(),
        ),
        Some(GroundSource::FlatAlbedo(reason)) => msgs.push(format!(
            "simsat: ground texture downgraded to flat albedo: {reason} \
             (georef.ground_source = 'flat-albedo')"
        )),
        _ => {}
    }
    if msgs.is_empty() {
        return;
    }
    if let Ok(warnings) = py.import("warnings") {
        for m in msgs {
            let _ = warnings.call_method1("warn", (m,));
        }
    }
}

/// Apply the optional `threads=` cap for this render, via the engine's
/// `effective_thread_count` / `configure_global_rayon` (the CLI's mechanism).
///
/// HONEST SEMANTICS (rayon global pool): the pool is built ONCE per process — the
/// FIRST render call in a process fixes the worker-thread count, from `threads=` (wins)
/// or the `RAYON_NUM_THREADS` environment variable; a different `threads=` on a LATER
/// call in the same process has NO effect. Pass `threads=` on the first call (or set
/// the env var before the first render) — e.g. `threads=1` per worker under a 16-way
/// `ProcessPoolExecutor` so 16 concurrent renders do not each grab every core.
fn apply_thread_cap(threads: Option<usize>) {
    let env = std::env::var("RAYON_NUM_THREADS").ok();
    configure_global_rayon(effective_thread_count(threads, env.as_deref()));
}

/// Build the `crs_params` dict: the PROJ parameters (consistent with `proj4`) plus the raw
/// WRF projection attributes for reference.
fn build_crs_params(py: Python<'_>, g: &api::Georef) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    let p = &g.projection;
    let r = 6_370_000.0f64; // WRF spherical earth (design decision 5)
    d.set_item("proj", g.proj_kind())?;
    d.set_item("R", r)?;
    // Raw WRF attributes (for reference / a fully custom CRS).
    d.set_item("map_proj", p.map_proj)?;
    d.set_item("truelat1", p.truelat1_deg)?;
    d.set_item("truelat2", p.truelat2_deg)?;
    d.set_item("stand_lon", p.stand_lon_deg)?;
    d.set_item("cen_lat", p.cen_lat_deg)?;
    d.set_item("cen_lon", p.cen_lon_deg)?;
    d.set_item("dx", p.dx_m)?;
    d.set_item("dy", p.dy_m)?;
    // PROJ keys consistent with `geo.proj4` (for the projection-metres / topdown extent).
    if g.extent_kind == ExtentKind::ProjectionMeters {
        let pole = if p.cen_lat_deg >= 0.0 { 90.0 } else { -90.0 };
        d.set_item("lon_0", p.stand_lon_deg)?;
        d.set_item("x_0", 0.0)?;
        d.set_item("y_0", 0.0)?;
        match p.map_proj {
            1 => {
                d.set_item("lat_1", p.truelat1_deg)?;
                d.set_item("lat_2", p.truelat2_deg)?;
                d.set_item("lat_0", pole)?;
            }
            2 => {
                d.set_item("lat_0", pole)?;
                d.set_item("lat_ts", p.truelat1_deg)?;
            }
            3 => {
                d.set_item("lat_ts", p.truelat1_deg)?;
            }
            _ => {}
        }
    }
    Ok(d.unbind())
}

fn runtime_err(msg: &str) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(msg.to_string())
}

fn value_err(msg: String) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(msg)
}

fn parse_sat(v: &str) -> PyResult<SatellitePreset> {
    match v.to_ascii_lowercase().replace(['-', '_', ' '], "").as_str() {
        "goeseast" | "goese" | "east" => Ok(SatellitePreset::GoesEast),
        "goeswest" | "goesw" | "west" => Ok(SatellitePreset::GoesWest),
        "himawari" | "ahi" => Ok(SatellitePreset::Himawari),
        _ => Err(value_err(format!(
            "unknown sat '{v}' (goes-east|goes-west|himawari)"
        ))),
    }
}

fn parse_backend(v: &str) -> PyResult<RenderBackend> {
    match v.to_ascii_lowercase().replace(['_', ' '], "-").as_str() {
        "cpu" => Ok(RenderBackend::Cpu),
        "gpu" | "gpu-preview" | "preview" => Ok(RenderBackend::GpuPreview),
        _ => Err(value_err(format!(
            "unknown backend '{v}' (cpu|gpu-preview)"
        ))),
    }
}

fn parse_storage_profile(v: &str) -> PyResult<StorageProfile> {
    StorageProfile::parse(v).ok_or_else(|| {
        value_err(format!(
            "unknown storage_profile '{v}' (compact-u8|science-cloud-f16)"
        ))
    })
}

fn parse_intent(v: &str) -> PyResult<RenderIntent> {
    match v.to_ascii_lowercase().replace(['_', ' '], "-").as_str() {
        "display" => Ok(RenderIntent::Display),
        "sensor" | "sensor-fast-gray" | "fast-gray" | "simsat-fast-gray-v1" => {
            Ok(RenderIntent::SensorFastGray)
        }
        _ => Err(value_err(format!(
            "unknown intent '{v}' (expected 'display' or 'sensor-fast-gray')"
        ))),
    }
}

fn parse_view(v: &str) -> PyResult<ViewMode> {
    match v.to_ascii_lowercase().replace(['-', '_', ' '], "").as_str() {
        "geo" | "geostationary" | "fromspace" | "space" => Ok(ViewMode::Geostationary),
        "topdown" | "top" | "map" | "topdownmap" | "nadir" => Ok(ViewMode::TopDownMap),
        _ => Err(value_err(format!("unknown view '{v}' (topdown|geo)"))),
    }
}

fn parse_geo_navigation(v: &str) -> PyResult<GeoNavigation> {
    match v.to_ascii_lowercase().replace(['_', ' '], "-").as_str() {
        "model-sphere" | "sphere" | "model" | "default" => Ok(GeoNavigation::ModelSphere),
        "goes-r-abi" | "goes-r" | "abi" | "ellipsoid" => Ok(GeoNavigation::GoesRAbiFixedGrid),
        _ => Err(value_err(format!(
            "unknown geo_navigation '{v}' (expected 'model-sphere' or 'goes-r-abi')"
        ))),
    }
}

/// Validate a zoom-out / domain margin fraction (added on each side). `0.0` = the domain
/// edge-to-edge (the default); a positive fraction frames the domain with surrounding real
/// earth. Kept as a fraction so a future km-based UI is a trivial swap.
fn parse_margin(v: f64) -> PyResult<f32> {
    if !(0.0..=4.0).contains(&v) || !v.is_finite() {
        return Err(value_err(format!(
            "margin must be a finite fraction in 0.0..=4.0 (0.0 = edge-to-edge), got {v}"
        )));
    }
    Ok(v as f32)
}

fn parse_resolution(v: &str) -> PyResult<ResolutionMode> {
    match v.to_ascii_lowercase().replace(['-', '_', ' '], "").as_str() {
        "native" => Ok(ResolutionMode::Native),
        "abi1km" | "1km" => Ok(ResolutionMode::Abi1km),
        "abi2km" | "2km" => Ok(ResolutionMode::Abi2km),
        _ => Err(value_err(format!(
            "unknown resolution '{v}' (native|abi1km|abi2km)"
        ))),
    }
}

fn parse_steps(v: &str) -> PyResult<StepQuality> {
    match v.to_ascii_lowercase().as_str() {
        "offline" | "full" | "384" => Ok(StepQuality::Offline),
        "interactive" | "preview" | "192" => Ok(StepQuality::Interactive),
        _ => Err(value_err(format!(
            "unknown steps '{v}' (offline|interactive)"
        ))),
    }
}

fn parse_fractional_cloud_mode(v: &str) -> PyResult<FractionalCloudMode> {
    match v.to_ascii_lowercase().replace('_', "-").as_str() {
        "on" | "true" | "effective" | "effective-od" => Ok(FractionalCloudMode::EffectiveOd),
        "deterministic-2" | "deterministic2" | "ica-2" | "mcica-2" => {
            Ok(FractionalCloudMode::Deterministic2)
        }
        "deterministic-4" | "deterministic4" | "ica-4" | "mcica-4" => {
            Ok(FractionalCloudMode::Deterministic4)
        }
        "deterministic-8" | "deterministic8" => Ok(FractionalCloudMode::Deterministic8),
        "deterministic-16" | "deterministic16" => Ok(FractionalCloudMode::Deterministic16),
        "off" | "false" | "legacy" => Ok(FractionalCloudMode::Off),
        _ => Err(value_err(format!(
            "unknown fractional_cloud_mode '{v}' (off|effective-od|deterministic-2|deterministic-4|deterministic-8|deterministic-16)"
        ))),
    }
}

fn parse_cloud_multiscatter(v: &str) -> PyResult<CloudMultiscatterMode> {
    match v.to_ascii_lowercase().replace('_', "-").as_str() {
        "legacy" | "legacy-octaves" | "octaves" => Ok(CloudMultiscatterMode::LegacyOctaves),
        "single" | "single-scatter" | "off" => Ok(CloudMultiscatterMode::SingleScatter),
        "delta-flux-v1" | "delta-flux" | "stage2" => Ok(CloudMultiscatterMode::DeltaFluxV1),
        "delta-flux-v2b" | "delta-flux-v2" | "stage2-p1" => Ok(CloudMultiscatterMode::DeltaFluxV2),
        "delta-flux-v3-memory" | "delta-flux-v3" | "stage2-memory" => {
            Ok(CloudMultiscatterMode::DeltaFluxV3)
        }
        _ => Err(value_err(format!(
            "unknown cloud_multiscatter '{v}' \
             (legacy-octaves|single-scatter|delta-flux-v1|delta-flux-v2b|delta-flux-v3-memory)"
        ))),
    }
}

fn parse_wv_band(v: &str) -> PyResult<WvBand> {
    WvBand::parse(v).ok_or_else(|| {
        value_err(format!(
            "unknown water-vapor band '{v}' (6.2|6.9|7.3, or upper|mid|low)"
        ))
    })
}

fn parse_thermal_sensor(v: &str) -> PyResult<ThermalSensor> {
    ThermalSensor::parse(v).ok_or_else(|| {
        value_err(format!(
            "unknown sensor '{v}' (fast-gray|goes-r-abi-band13-fm4)"
        ))
    })
}

fn parse_instrument_footprint(v: &str) -> PyResult<InstrumentFootprint> {
    InstrumentFootprint::parse(v).ok_or_else(|| {
        value_err(format!(
            "unknown instrument_footprint '{v}' (off|goes-r-abi-band13-mtf-prototype)"
        ))
    })
}

fn parse_enhancement(v: &str) -> PyResult<IrEnhancement> {
    IrEnhancement::parse_strict(v).ok_or_else(|| {
        value_err(format!(
            "unknown enhancement '{v}' \
             (natural|cimss|bd|avn|funktop|rainbow|gray; \
             noaa/heritage accepted for natural; grayscale/greyscale accepted for gray)"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_parser_defaults_and_tokens_match_rust_api() {
        assert_eq!(
            RenderParams::new(PathBuf::from("input")).backend,
            RenderBackend::Cpu
        );
        assert_eq!(parse_backend("cpu").unwrap(), RenderBackend::Cpu);
        assert_eq!(
            parse_backend("gpu-preview").unwrap(),
            RenderBackend::GpuPreview
        );
        assert_eq!(parse_backend("gpu").unwrap(), RenderBackend::GpuPreview);
        assert!(parse_backend("magic").is_err());
    }

    #[test]
    fn storage_profile_parser_defaults_compact_and_exposes_science() {
        assert_eq!(
            RenderParams::new(PathBuf::from("input")).storage_profile,
            StorageProfile::CompactU8
        );
        assert_eq!(
            parse_storage_profile("science-cloud-f16").unwrap(),
            StorageProfile::ScienceCloudF16
        );
        assert!(parse_storage_profile("raw-f32").is_err());
    }

    #[test]
    fn render_intent_parser_exposes_strict_sensor_fast_gray_semantics() {
        assert_eq!(
            RenderParams::new(PathBuf::from("input")).intent,
            RenderIntent::Display
        );
        assert_eq!(parse_intent("display").unwrap(), RenderIntent::Display);
        assert_eq!(
            parse_intent("sensor-fast-gray").unwrap(),
            RenderIntent::SensorFastGray
        );
        assert_eq!(
            parse_intent("simsat-fast-gray-v1").unwrap(),
            RenderIntent::SensorFastGray
        );
        assert!(parse_intent("abi-band-2").is_err());

        let mut params = RenderParams::new(PathBuf::from("input"));
        assert!(params.topdown_shadow_antialias);
        params.intent = parse_intent("sensor").unwrap();
        params.granulation = Some(true);
        let (effective, changes) = api::plan_render_intent(&params);
        assert_eq!(effective.cloud_optical_depth_scale, 1.0);
        assert!(effective.fractional_clouds);
        assert_eq!(effective.granulation, Some(false));
        assert!(!effective.topdown_shadow_antialias);
        assert!(changes.contains(&simsat_engine::api::RenderIntentAdjustment::GranulationOff));
        assert!(
            changes
                .contains(&simsat_engine::api::RenderIntentAdjustment::TopdownShadowAntialiasOff)
        );
    }

    #[test]
    fn fractional_mode_parser_exposes_reference_and_preserves_effective_alias() {
        for token in ["on", "effective", "effective_od", "effective-od"] {
            assert_eq!(
                parse_fractional_cloud_mode(token).unwrap(),
                FractionalCloudMode::EffectiveOd
            );
        }
        assert_eq!(
            parse_fractional_cloud_mode("deterministic-2").unwrap(),
            FractionalCloudMode::Deterministic2
        );
        assert_eq!(
            parse_fractional_cloud_mode("deterministic-4").unwrap(),
            FractionalCloudMode::Deterministic4
        );
        assert_eq!(
            parse_fractional_cloud_mode("deterministic-8").unwrap(),
            FractionalCloudMode::Deterministic8
        );
        assert_eq!(
            parse_fractional_cloud_mode("deterministic_16").unwrap(),
            FractionalCloudMode::Deterministic16
        );
        assert_eq!(
            parse_fractional_cloud_mode("off").unwrap(),
            FractionalCloudMode::Off
        );
        assert!(parse_fractional_cloud_mode("random-4").is_err());
        assert!(parse_fractional_cloud_mode("mcica-8").is_err());
    }

    #[test]
    fn thermal_sensor_parser_preserves_default_and_accepts_official_srf() {
        assert_eq!(
            RenderParams::new(PathBuf::from("input")).thermal_sensor,
            ThermalSensor::FastGray
        );
        assert_eq!(
            parse_thermal_sensor("fast-gray").unwrap(),
            ThermalSensor::FastGray
        );
        assert_eq!(
            parse_thermal_sensor("goes-r-abi-band13-fm4").unwrap(),
            ThermalSensor::GoesRAbiBand13Fm4
        );
        assert!(parse_thermal_sensor("gaussian-ish").is_err());
    }

    #[test]
    fn instrument_footprint_parser_is_explicit_and_default_off() {
        assert_eq!(
            RenderParams::new(PathBuf::from("input")).instrument_footprint,
            InstrumentFootprint::Off
        );
        assert_eq!(
            parse_instrument_footprint("goes-r-abi-band13-mtf-prototype").unwrap(),
            InstrumentFootprint::GoesRAbiBand13Mtf
        );
        assert!(parse_instrument_footprint("generic-blur").is_err());
    }

    #[test]
    fn enhancement_parser_exposes_natural_and_preserves_legacy_tokens() {
        assert_eq!(
            parse_enhancement("natural").unwrap(),
            IrEnhancement::Natural
        );
        assert_eq!(
            parse_enhancement("NOAA heritage").unwrap(),
            IrEnhancement::Natural
        );
        for (token, expected) in [
            ("cimss", IrEnhancement::Cimss),
            ("bd", IrEnhancement::Bd),
            ("avn", IrEnhancement::Avn),
            ("funktop", IrEnhancement::Funktop),
            ("rainbow", IrEnhancement::Rainbow),
            ("gray", IrEnhancement::Grayscale),
            ("greyscale", IrEnhancement::Grayscale),
        ] {
            assert_eq!(parse_enhancement(token).unwrap(), expected, "{token}");
        }
        assert!(parse_enhancement("smooth-rainbow").is_err());
    }

    #[test]
    fn visible_physics_helper_assigns_every_binding_control() {
        let mut params = RenderParams::new(PathBuf::from("input"));
        apply_visible_physics_controls(
            &mut params,
            0.2,
            true,
            false,
            false,
            false,
            "deterministic-4",
            0.5,
            "fixed",
            true,
            true,
            true,
        )
        .unwrap();
        assert_eq!(params.aerosol_optical_depth, 0.2);
        assert!(params.rh_aerosol_swelling);
        assert!(!params.atmosphere_correction);
        assert!(!params.terrain_atmosphere);
        assert!(!params.fractional_clouds);
        assert_eq!(params.fractional_cloud_mode, FractionalCloudMode::Off);
        assert_eq!(params.cloud_optical_depth_scale, 0.5);
        assert!(params.feather_exposed_domain_edges);
        assert!(params.beer_powder);
        assert_eq!(params.granulation, Some(true));
    }

    #[test]
    fn visible_physics_helper_honors_explicit_edge_off() {
        let mut params = RenderParams::new(PathBuf::from("input"));
        assert!(params.feather_exposed_domain_edges);
        apply_visible_physics_controls(
            &mut params,
            simsat_engine::atmosphere::DEFAULT_AOD as f32,
            false,
            true,
            true,
            true,
            "effective-od",
            simsat_engine::clouds::DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE,
            "fixed",
            false,
            false,
            false,
        )
        .unwrap();
        assert_eq!(
            params.aerosol_optical_depth.to_bits(),
            (simsat_engine::atmosphere::DEFAULT_AOD as f32).to_bits()
        );
        assert!(!params.rh_aerosol_swelling);
        assert!(params.atmosphere_correction);
        assert!(params.terrain_atmosphere);
        assert!(params.fractional_clouds);
        assert_eq!(
            params.fractional_cloud_mode,
            FractionalCloudMode::EffectiveOd
        );
        assert!(params.fractional_clouds);
        assert_eq!(
            params.cloud_optical_depth_scale,
            simsat_engine::clouds::DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE
        );
        assert!(!params.feather_exposed_domain_edges);
        assert!(!params.beer_powder);
        assert_eq!(params.granulation, Some(false));
    }

    #[test]
    fn visible_display_helper_preserves_defaults_when_omitted() {
        let mut params = RenderParams::new(PathBuf::from("input"));
        let default_exposure = params.exposure;
        apply_visible_display_controls(&mut params, None, None, None, None);
        assert_eq!(params.exposure, default_exposure);
        assert!(params.ground_gain.is_none());
        assert!(params.cloud_softclip.is_none());
        assert!(params.cloud_highlight_max.is_none());
    }

    #[test]
    fn visible_display_helper_assigns_every_override() {
        let mut params = RenderParams::new(PathBuf::from("input"));
        apply_visible_display_controls(&mut params, Some(1.4), Some(1.6), Some(0.65), Some(1.25));
        assert_eq!(params.exposure, 1.4);
        assert_eq!(params.ground_gain, Some(1.6));
        assert_eq!(params.cloud_softclip, Some(0.65));
        assert_eq!(params.cloud_highlight_max, Some(1.25));
    }

    #[test]
    fn land_appearance_helper_assigns_every_control_and_explicit_identity() {
        let mut params = RenderParams::new(PathBuf::from("input"));
        assert_eq!(
            params.land_appearance,
            simsat_engine::render::LandAppearanceConfig::shipped()
        );
        apply_land_appearance_controls(
            &mut params,
            false,
            simsat_engine::render::LAND_SZA_MAX_GAIN,
            false,
            0.08,
            0.65,
            1.5,
        )
        .unwrap();
        assert_eq!(
            params.land_appearance,
            simsat_engine::render::LandAppearanceConfig::identity()
        );
        apply_land_appearance_controls(&mut params, true, 1.7, true, 0.07, 0.6, 1.4).unwrap();
        assert_eq!(
            params.land_appearance,
            simsat_engine::render::LandAppearanceConfig {
                sza_normalization: true,
                sza_max_gain: 1.7,
                dark_toe: true,
                dark_toe_knee: 0.07,
                dark_toe_gamma: 0.6,
                dark_toe_max_gain: 1.4,
            }
        );
    }

    #[test]
    fn land_appearance_helper_rejects_nonfinite_and_out_of_range_values() {
        let cases = [
            (0.9, 0.08, 0.65, 1.5),
            (1.6, 0.0, 0.65, 1.5),
            (1.6, 0.08, 1.1, 1.5),
            (1.6, 0.08, 0.65, f64::NAN),
        ];
        for (sza, knee, gamma, toe_max) in cases {
            let mut params = RenderParams::new(PathBuf::from("input"));
            assert!(
                apply_land_appearance_controls(&mut params, true, sza, true, knee, gamma, toe_max,)
                    .is_err()
            );
        }
    }

    #[test]
    fn surface_postlight_toe_helper_is_default_off_and_assigns_bounded_controls() {
        let mut params = RenderParams::new(PathBuf::from("input"));
        assert!(params.surface_postlight_toe.is_identity());
        apply_surface_postlight_toe_controls(&mut params, true, 0.18, 0.80, 1.35).unwrap();
        assert_eq!(
            params.surface_postlight_toe,
            SurfacePostlightToeConfig {
                enabled: true,
                knee: 0.18,
                gamma: 0.80,
                max_gain: 1.35,
            }
        );
    }

    #[test]
    fn surface_postlight_toe_helper_rejects_invalid_values() {
        for (knee, gamma, max_gain) in
            [(0.0, 0.80, 1.35), (0.18, 1.1, 1.35), (0.18, 0.80, f64::NAN)]
        {
            let mut params = RenderParams::new(PathBuf::from("input"));
            assert!(
                apply_surface_postlight_toe_controls(&mut params, true, knee, gamma, max_gain,)
                    .is_err()
            );
        }
    }

    #[test]
    fn twilight_surface_recovery_helper_keeps_shipped_default_and_supports_explicit_off() {
        let mut params = RenderParams::new(PathBuf::from("input"));
        assert_eq!(
            params.twilight_surface_recovery,
            TwilightSurfaceRecoveryConfig::shipped()
        );
        apply_twilight_surface_recovery_controls(&mut params, false, 0.30, 0.50, 4.0).unwrap();
        assert_eq!(
            params.twilight_surface_recovery,
            TwilightSurfaceRecoveryConfig::off()
        );
        apply_twilight_surface_recovery_controls(&mut params, true, 0.30, 0.50, 4.0).unwrap();
        assert_eq!(
            params.twilight_surface_recovery,
            TwilightSurfaceRecoveryConfig::shipped()
        );
    }

    #[test]
    fn non_visible_product_helper_forces_both_postlight_controls_to_identity() {
        let mut params = RenderParams::new(PathBuf::from("input"));
        params.surface_postlight_toe = SurfacePostlightToeConfig {
            enabled: true,
            ..SurfacePostlightToeConfig::default()
        };
        force_surface_recovery_identity(&mut params);
        assert_eq!(
            params.surface_postlight_toe,
            SurfacePostlightToeConfig::off()
        );
        assert_eq!(
            params.twilight_surface_recovery,
            TwilightSurfaceRecoveryConfig::off()
        );
    }

    #[test]
    fn twilight_surface_recovery_helper_rejects_invalid_values() {
        for (knee, gamma, max_gain) in [(0.0, 0.50, 4.0), (0.30, 1.1, 4.0), (0.30, 0.50, f64::NAN)]
        {
            let mut params = RenderParams::new(PathBuf::from("input"));
            assert!(
                apply_twilight_surface_recovery_controls(&mut params, true, knee, gamma, max_gain,)
                    .is_err()
            );
        }
    }

    #[test]
    fn cloud_multiscatter_parser_matches_rust_and_cli_tokens() {
        assert_eq!(
            parse_cloud_multiscatter("legacy-octaves").unwrap(),
            CloudMultiscatterMode::LegacyOctaves
        );
        assert_eq!(
            parse_cloud_multiscatter("single_scatter").unwrap(),
            CloudMultiscatterMode::SingleScatter
        );
        assert_eq!(
            parse_cloud_multiscatter("delta-flux-v1").unwrap(),
            CloudMultiscatterMode::DeltaFluxV1
        );
        assert_eq!(
            parse_cloud_multiscatter("delta-flux-v2b").unwrap(),
            CloudMultiscatterMode::DeltaFluxV2
        );
        assert_eq!(
            parse_cloud_multiscatter("delta-flux-v3-memory").unwrap(),
            CloudMultiscatterMode::DeltaFluxV3
        );
        assert!(parse_cloud_multiscatter("unknown").is_err());
    }

    #[test]
    fn omitted_cloud_multiscatter_preserves_the_legacy_boolean_contract() {
        let defaults = RenderParams::new(PathBuf::from("input"));
        assert!(defaults.multiscatter);
        assert_eq!(defaults.cloud_multiscatter, None);

        let mut legacy_single = defaults;
        legacy_single.multiscatter = false;
        assert_eq!(legacy_single.cloud_multiscatter, None);
    }
}

/// The `simsat` Python module.
#[pymodule]
fn simsat(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // QUIET BY DEFAULT (module init): disable the engine's diagnostic stderr sink for
    // this process unless SIMSAT_LOG opts in. The engine defaults the sink ON so the
    // CLI/studio are untouched; the LIBRARY personality is silent. Runtime flip:
    // simsat.set_verbose(True/False).
    simsat_engine::log::set_enabled(env_verbose_opt_in());
    m.add(
        "__doc__",
        "SimSat — physically-based simulated visible/IR satellite imagery from WRF output. \
         render_visible_rgb / render_geocolor / render_sandwich / render_rgb_reflectance / \
         render_ir / render_water_vapor / render_precipitable_water / render_cloud_top_temp / \
         render_cloud_optical_depth each return a numpy array plus a Georef (projection params \
         + imshow extent + lat/lon mesh + render-honesty metadata: time_is_fallback, \
         ground_source, ground_status). The three derived-field functions return RAW physical \
         scalar arrays (mm / Kelvin / dimensionless) to plot with your own colormaps. Every \
         function takes threads= (per-process rayon cap; the FIRST render call's value wins). \
         render_cloud_layer returns the web-map cloud + shadow layer pair on a Web-Mercator \
         grid (georef.mercator_corners = the Mapbox ImageSource corner lon/lats). \
         render_perspective renders a free eye/look/fov pinhole camera (full composite, or \
         cloud_layer_only=True for the cloud field alone; georef.camera_pose = the camera).",
    )?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(render_visible_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(render_geocolor, m)?)?;
    m.add_function(wrap_pyfunction!(render_sandwich, m)?)?;
    m.add_function(wrap_pyfunction!(render_rgb_reflectance, m)?)?;
    m.add_function(wrap_pyfunction!(render_visible_bands, m)?)?;
    m.add_function(wrap_pyfunction!(render_ir, m)?)?;
    m.add_function(wrap_pyfunction!(render_water_vapor, m)?)?;
    m.add_function(wrap_pyfunction!(render_precipitable_water, m)?)?;
    m.add_function(wrap_pyfunction!(render_cloud_top_temp, m)?)?;
    m.add_function(wrap_pyfunction!(render_cloud_optical_depth, m)?)?;
    m.add_function(wrap_pyfunction!(render_cloud_layer, m)?)?;
    m.add_function(wrap_pyfunction!(render_perspective, m)?)?;
    m.add_function(wrap_pyfunction!(solar_elevation_grid, m)?)?;
    m.add_function(wrap_pyfunction!(set_verbose, m)?)?;
    m.add_class::<Georef>()?;
    Ok(())
}
