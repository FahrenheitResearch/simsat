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
//! - [`render_geocolor`] -> `(H x W x 3 uint8, Georef)`  the GeoColor day/night blend
//!   (true-color by day, colored band-13 IR by night) — always meaningful day OR night.
//! - [`render_sandwich`] -> `(H x W x 3 uint8, Georef)`  the Sandwich composite (visible
//!   true-color base + color-enhanced band-13 IR overlaid on the cold cloud tops) — the
//!   classic severe-convection view; a daytime-convection product.
//! - [`render_visible_bands`] -> `(H x W x 3 float32, Georef)`  RAW per-channel reflectance
//!   (pre-tonemap, `[0, 1]`), for building a custom RGB / operating on bands.
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
//!
//! Default `view='topdown'` (map-registered, north-up — the natural fit for a top-down
//! Lambert map); `view='geo'` gives the from-space geostationary view.
//! `resolution='native'` means one output pixel per source-model grid cell, not the
//! highest possible output resolution. `abi1km` / `abi2km` select 1 km / 2 km output
//! sampling and may upsample a coarse model or downsample a fine WRF grid.
//!
//! Every function also takes `threads=` (cap the render worker threads; the rayon pool
//! is GLOBAL and built once per process — the first render call's value wins, see
//! [`apply_thread_cap`]) and returns honesty metadata on the [`Georef`]
//! (`time_is_fallback`, `ground_source`, `ground_status`), with `UserWarning`s raised
//! on a fabricated-date or downgraded-ground render ([`warn_downgrades`]).
//!
//! The binding is QUIET BY DEFAULT: the engine's diagnostic stderr lines (ingest
//! progress / warnings) are disabled at module init unless `SIMSAT_LOG=1`; flip at
//! runtime with [`set_verbose`]. Honesty metadata / `UserWarning`s are unaffected.

use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::path::PathBuf;

use simsat_engine::api::{
    self, BlueMarble, ExtentKind, FrameData, GroundSource, Product, RenderBackend, RenderParams,
    RenderResult, SunOverride,
};
use simsat_engine::camera::{ResolutionMode, SatellitePreset, ViewMode};
use simsat_engine::clouds::{CloudMultiscatterMode, StepQuality};
use simsat_engine::derived::DerivedField;
use simsat_engine::ir_enhance::IrEnhancement;
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
/// `terrain_atmosphere`, `fractional_clouds` (default true), and
/// `cloud_optical_depth_scale` (0..=4, shipped default 0.15 by owner cross-file visual
/// calibration; 1.0 is unscaled model extinction), and the default-on
/// `feather_exposed_domain_edges` finite-domain presentation control. Fractional clouds
/// use the model cloud
/// fraction when present; false restores legacy horizontally-full cells. The OD scale is
/// a visible sensitivity control and does not alter the quantitative
/// `render_cloud_optical_depth` product. `clouds` remains the explicit feature bypass;
/// `multiscatter` controls the established higher scattering octaves without changing
/// transmittance. `cloud_multiscatter` is an explicit override accepting
/// `legacy-octaves`, `single-scatter`, or opt-in experimental `delta-flux-v1` /
/// `delta-flux-v2b`; leaving it unset preserves the historical boolean behavior exactly.
/// `beer_powder` enables the optional direct-sun shaping, and `granulation` enables
/// display-only sub-grid cloud-edge erosion; both default off. Finished visible display
/// products also expose `topdown_stratiform_regularization`, an opt-in/default-off
/// low/liquid-deck reconstruction used only by the top-down finished-visible path.
/// Geostationary and raw visible bands ignore it. Finished visible display
/// products also accept `ground_gain`, `cloud_softclip`, and `cloud_highlight_max` as
/// optional calibration overrides. The land-only controls
/// `land_sza_normalization` / `land_sza_max_gain` and `land_dark_toe` plus its
/// knee/gamma/max-gain parameters are independently switchable and default on in the
/// owner-selected v0.1.5 display preset. Passing both booleans false is the exact legacy
/// identity. Raw
/// visible bands deliberately expose none of these display-only land controls.
///
/// `threads` (default None = all cores, or RAYON_NUM_THREADS) caps the render worker
/// threads for THIS PROCESS. The pool is global and built ONCE — the first render call's
/// value wins; later calls cannot change it. Available on every render function here.
///
/// Returns `(rgb, georef)` where `rgb` is a numpy `H x W x 3` uint8 array (row 0 = north).
#[pyfunction]
#[pyo3(signature = (
    input, *, backend="cpu", sat="goes-east", view="topdown", timestep=0, resolution="native", margin=0.0,
    aerosol_optical_depth=0.05, rh_aerosol_swelling=false, atmosphere_correction=true,
    terrain_atmosphere=true, land_sza_normalization=true, land_sza_max_gain=1.6,
    land_dark_toe=true, land_dark_toe_knee=0.08, land_dark_toe_gamma=0.65,
    land_dark_toe_max_gain=1.5, exposure=None, ground_gain=None, cloud_softclip=None,
    cloud_highlight_max=None, multiscatter=true, cloud_multiscatter=None, beer_powder=false,
    steps="offline", clouds=true, fractional_clouds=true, cloud_optical_depth_scale=0.15,
    feather_exposed_domain_edges=true, granulation=false,
    topdown_stratiform_regularization=false,
    sun_elev=None, sun_az=None, cache=None, bluemarble=None,
    bluemarble_month=None, bluemarble_download=true, threads=None
))]
#[allow(clippy::too_many_arguments)]
fn render_visible_rgb<'py>(
    py: Python<'py>,
    input: String,
    backend: &str,
    sat: &str,
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
    cloud_optical_depth_scale: f32,
    feather_exposed_domain_edges: bool,
    granulation: bool,
    topdown_stratiform_regularization: bool,
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
        sat,
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
        cloud_optical_depth_scale,
        feather_exposed_domain_edges,
        granulation,
        topdown_stratiform_regularization,
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

/// Render the GeoColor day/night blend: true-color visible by day, colored band-13 IR by
/// night, crossfaded across the terminator by the per-pixel solar elevation (GOES's flagship
/// product). Always meaningful day OR night — the night side shows the storm/clouds in IR
/// where a plain visible frame would be black (no city lights; our night side is the colored
/// IR, honestly). The sun / exposure / clouds controls apply to the visible (day) half.
///
/// Returns `(rgb, georef)` where `rgb` is a numpy `H x W x 3` uint8 array (row 0 = north),
/// exactly like [`render_visible_rgb`].
#[pyfunction]
#[pyo3(signature = (
    input, *, sat="goes-east", view="topdown", timestep=0, resolution="native", margin=0.0,
    aerosol_optical_depth=0.05, rh_aerosol_swelling=false, atmosphere_correction=true,
    terrain_atmosphere=true, land_sza_normalization=true, land_sza_max_gain=1.6,
    land_dark_toe=true, land_dark_toe_knee=0.08, land_dark_toe_gamma=0.65,
    land_dark_toe_max_gain=1.5, exposure=None, ground_gain=None, cloud_softclip=None,
    cloud_highlight_max=None, multiscatter=true, cloud_multiscatter=None, beer_powder=false,
    steps="offline", clouds=true, fractional_clouds=true, cloud_optical_depth_scale=0.15,
    feather_exposed_domain_edges=true, granulation=false,
    topdown_stratiform_regularization=false,
    sun_elev=None, sun_az=None, cache=None, bluemarble=None,
    bluemarble_month=None, bluemarble_download=true, threads=None
))]
#[allow(clippy::too_many_arguments)]
fn render_geocolor<'py>(
    py: Python<'py>,
    input: String,
    sat: &str,
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
    cloud_optical_depth_scale: f32,
    feather_exposed_domain_edges: bool,
    granulation: bool,
    topdown_stratiform_regularization: bool,
    sun_elev: Option<f64>,
    sun_az: Option<f64>,
    cache: Option<String>,
    bluemarble: Option<String>,
    bluemarble_month: Option<u32>,
    bluemarble_download: bool,
    threads: Option<usize>,
) -> PyResult<(Bound<'py, PyArray3<u8>>, Georef)> {
    let params = build_visible_params(
        input,
        sat,
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
        cloud_optical_depth_scale,
        feather_exposed_domain_edges,
        granulation,
        topdown_stratiform_regularization,
        sun_elev,
        sun_az,
        cache,
        bluemarble,
        bluemarble_month,
        bluemarble_download,
    )?;
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
    input, *, sat="goes-east", view="topdown", timestep=0, resolution="native", margin=0.0,
    aerosol_optical_depth=0.05, rh_aerosol_swelling=false, atmosphere_correction=true,
    terrain_atmosphere=true, land_sza_normalization=true, land_sza_max_gain=1.6,
    land_dark_toe=true, land_dark_toe_knee=0.08, land_dark_toe_gamma=0.65,
    land_dark_toe_max_gain=1.5, exposure=None, ground_gain=None, cloud_softclip=None,
    cloud_highlight_max=None, multiscatter=true, cloud_multiscatter=None, beer_powder=false,
    steps="offline", clouds=true, fractional_clouds=true, cloud_optical_depth_scale=0.15,
    feather_exposed_domain_edges=true, granulation=false,
    topdown_stratiform_regularization=false,
    sun_elev=None, sun_az=None, cache=None, bluemarble=None,
    bluemarble_month=None, bluemarble_download=true, threads=None
))]
#[allow(clippy::too_many_arguments)]
fn render_sandwich<'py>(
    py: Python<'py>,
    input: String,
    sat: &str,
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
    cloud_optical_depth_scale: f32,
    feather_exposed_domain_edges: bool,
    granulation: bool,
    topdown_stratiform_regularization: bool,
    sun_elev: Option<f64>,
    sun_az: Option<f64>,
    cache: Option<String>,
    bluemarble: Option<String>,
    bluemarble_month: Option<u32>,
    bluemarble_download: bool,
    threads: Option<usize>,
) -> PyResult<(Bound<'py, PyArray3<u8>>, Georef)> {
    let params = build_visible_params(
        input,
        sat,
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
        cloud_optical_depth_scale,
        feather_exposed_domain_edges,
        granulation,
        topdown_stratiform_regularization,
        sun_elev,
        sun_az,
        cache,
        bluemarble,
        bluemarble_month,
        bluemarble_download,
    )?;
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

/// Render the RAW per-channel reflectance (pre-tonemap), for building a custom RGB / band
/// math.
///
/// Returns `(bands, georef)` where `bands` is a numpy `H x W x 3` float32 array in `[0, 1]`
/// (R, G, B reflectance factors; row 0 = north).
#[pyfunction]
#[pyo3(signature = (
    input, *, sat="goes-east", view="topdown", timestep=0, resolution="native", margin=0.0,
    aerosol_optical_depth=0.05, rh_aerosol_swelling=false, atmosphere_correction=true,
    terrain_atmosphere=true, multiscatter=true, cloud_multiscatter=None, beer_powder=false,
    steps="offline", clouds=true,
    fractional_clouds=true, cloud_optical_depth_scale=0.15, granulation=false, sun_elev=None,
    sun_az=None, cache=None,
    bluemarble=None,
    bluemarble_month=None, bluemarble_download=true, threads=None
))]
#[allow(clippy::too_many_arguments)]
fn render_visible_bands<'py>(
    py: Python<'py>,
    input: String,
    sat: &str,
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
    cloud_optical_depth_scale: f32,
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
        sat,
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
        cloud_optical_depth_scale,
        false,
        granulation,
        false,
        sun_elev,
        sun_az,
        cache,
        bluemarble,
        bluemarble_month,
        bluemarble_download,
    )?;
    apply_thread_cap(threads);
    let result = render_or_err(py, params, Product::VisibleBands)?;
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
/// off-domain; row 0 = north). If `enhancement` is given (one of 'cimss', 'bd', 'avn',
/// 'funktop', 'rainbow', 'gray') the return is instead `(bt, rgb, georef)` with `rgb` a
/// numpy `H x W x 3` uint8 colored image.
#[pyfunction]
#[pyo3(signature = (
    input, *, sat="goes-east", view="topdown", timestep=0, resolution="native", margin=0.0,
    enhancement=None, cache=None, threads=None
))]
#[allow(clippy::too_many_arguments)]
fn render_ir<'py>(
    py: Python<'py>,
    input: String,
    sat: &str,
    view: &str,
    timestep: usize,
    resolution: &str,
    margin: f64,
    enhancement: Option<String>,
    cache: Option<String>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut params = RenderParams::new(PathBuf::from(&input));
    params.satellite = parse_sat(sat)?;
    params.view = parse_view(view)?;
    params.timestep = timestep;
    params.resolution = parse_resolution(resolution)?;
    params.margin_frac = parse_margin(margin)?;
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
/// row 0 = north). With `enhancement` given (one of 'cimss', 'bd', 'avn', 'funktop',
/// 'rainbow', 'gray') the return is instead `(bt, rgb, georef)` with `rgb` a numpy
/// `H x W x 3` uint8 colored image — for WV, 'cimss' is the classic WV moisture palette
/// and 'gray' is a WV-scaled grayscale (cold/moist white). Thermal — works day AND night.
#[pyfunction]
#[pyo3(signature = (
    input, *, band="6.2", sat="goes-east", view="topdown", timestep=0, resolution="native",
    margin=0.0, enhancement=None, cache=None, threads=None
))]
#[allow(clippy::too_many_arguments)]
fn render_water_vapor<'py>(
    py: Python<'py>,
    input: String,
    band: &str,
    sat: &str,
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
    params.satellite = parse_sat(sat)?;
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
    input, *, sat="goes-east", view="topdown", timestep=0, resolution="native", margin=0.0,
    colormap=false, cache=None, threads=None
))]
#[allow(clippy::too_many_arguments)]
fn render_precipitable_water<'py>(
    py: Python<'py>,
    input: String,
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
    input, *, sat="goes-east", view="topdown", timestep=0, resolution="native", margin=0.0,
    colormap=false, cache=None, threads=None
))]
#[allow(clippy::too_many_arguments)]
fn render_cloud_top_temp<'py>(
    py: Python<'py>,
    input: String,
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
    input, *, sat="goes-east", view="topdown", timestep=0, resolution="native", margin=0.0,
    colormap=false, cache=None, threads=None
))]
#[allow(clippy::too_many_arguments)]
fn render_cloud_optical_depth<'py>(
    py: Python<'py>,
    input: String,
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
    input, *, sat="goes-east", timestep=0, margin=0.0, aerosol_optical_depth=0.05,
    rh_aerosol_swelling=false, atmosphere_correction=true, terrain_atmosphere=true,
    exposure=None, ground_gain=None, cloud_softclip=None, cloud_highlight_max=None,
    multiscatter=true, cloud_multiscatter=None, beer_powder=false,
    steps="offline", clouds=true,
    fractional_clouds=true, cloud_optical_depth_scale=0.15,
    feather_exposed_domain_edges=true, granulation=false, sun_elev=None, sun_az=None, cache=None,
    premultiplied=false, threads=None
))]
#[allow(clippy::too_many_arguments)]
fn render_cloud_layer<'py>(
    py: Python<'py>,
    input: String,
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
    cloud_optical_depth_scale: f32,
    feather_exposed_domain_edges: bool,
    granulation: bool,
    sun_elev: Option<f64>,
    sun_az: Option<f64>,
    cache: Option<String>,
    premultiplied: bool,
    threads: Option<usize>,
) -> PyResult<(Bound<'py, PyArray3<u8>>, Bound<'py, PyArray2<f32>>, Georef)> {
    let mut params = RenderParams::new(PathBuf::from(&input));
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
        cloud_optical_depth_scale,
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
    input, *, eye, look, fov=40.0, size=(1280, 720), timestep=0,
    aerosol_optical_depth=0.05, rh_aerosol_swelling=false, atmosphere_correction=true,
    terrain_atmosphere=true, land_sza_normalization=true, land_sza_max_gain=1.6,
    land_dark_toe=true, land_dark_toe_knee=0.08, land_dark_toe_gamma=0.65,
    land_dark_toe_max_gain=1.5, exposure=None, ground_gain=None, cloud_softclip=None,
    cloud_highlight_max=None, multiscatter=true, cloud_multiscatter=None, beer_powder=false,
    steps="offline", clouds=true, fractional_clouds=true, cloud_optical_depth_scale=0.15,
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
    cloud_optical_depth_scale: f32,
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
    params.timestep = timestep;
    apply_visible_physics_controls(
        &mut params,
        aerosol_optical_depth,
        rh_aerosol_swelling,
        atmosphere_correction,
        terrain_atmosphere,
        fractional_clouds,
        cloud_optical_depth_scale,
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
    sat: &str,
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
    cloud_optical_depth_scale: f32,
    feather_exposed_domain_edges: bool,
    granulation: bool,
    topdown_stratiform_regularization: bool,
    sun_elev: Option<f64>,
    sun_az: Option<f64>,
    cache: Option<String>,
    bluemarble: Option<String>,
    bluemarble_month: Option<u32>,
    bluemarble_download: bool,
) -> PyResult<RenderParams> {
    let mut params = RenderParams::new(PathBuf::from(&input));
    params.satellite = parse_sat(sat)?;
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
        cloud_optical_depth_scale,
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
    params.multiscatter = multiscatter;
    params.cloud_multiscatter = cloud_multiscatter
        .map(parse_cloud_multiscatter)
        .transpose()?;
    params.steps = parse_steps(steps)?;
    params.clouds = clouds;
    params.topdown_stratiform_regularization = topdown_stratiform_regularization;
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
fn apply_visible_physics_controls(
    params: &mut RenderParams,
    aerosol_optical_depth: f32,
    rh_aerosol_swelling: bool,
    atmosphere_correction: bool,
    terrain_atmosphere: bool,
    fractional_clouds: bool,
    cloud_optical_depth_scale: f32,
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
    params.fractional_clouds = fractional_clouds;
    params.cloud_optical_depth_scale = cloud_optical_depth_scale;
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
        mercator_corners: g
            .mercator_corners_lonlat
            .map(|c| c.iter().map(|p| (p[0], p[1])).collect()),
        camera_pose,
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

fn parse_view(v: &str) -> PyResult<ViewMode> {
    match v.to_ascii_lowercase().replace(['-', '_', ' '], "").as_str() {
        "geo" | "geostationary" | "fromspace" | "space" => Ok(ViewMode::Geostationary),
        "topdown" | "top" | "map" | "topdownmap" | "nadir" => Ok(ViewMode::TopDownMap),
        _ => Err(value_err(format!("unknown view '{v}' (topdown|geo)"))),
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

fn parse_cloud_multiscatter(v: &str) -> PyResult<CloudMultiscatterMode> {
    match v.to_ascii_lowercase().replace('_', "-").as_str() {
        "legacy" | "legacy-octaves" | "octaves" => Ok(CloudMultiscatterMode::LegacyOctaves),
        "single" | "single-scatter" | "off" => Ok(CloudMultiscatterMode::SingleScatter),
        "delta-flux-v1" | "delta-flux" | "stage2" => Ok(CloudMultiscatterMode::DeltaFluxV1),
        "delta-flux-v2b" | "delta-flux-v2" | "stage2-p1" => Ok(CloudMultiscatterMode::DeltaFluxV2),
        _ => Err(value_err(format!(
            "unknown cloud_multiscatter '{v}' \
             (legacy-octaves|single-scatter|delta-flux-v1|delta-flux-v2b)"
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

fn parse_enhancement(v: &str) -> PyResult<IrEnhancement> {
    // `IrEnhancement::parse` is total (falls back to a default); reject an unknown token
    // explicitly so a typo surfaces rather than silently picking grayscale.
    match v.to_ascii_lowercase().replace(['-', '_', ' '], "").as_str() {
        "cimss" | "bd" | "avn" | "funktop" | "rainbow" | "gray" | "grayscale" | "grey" => {
            Ok(IrEnhancement::parse(v))
        }
        _ => Err(value_err(format!(
            "unknown enhancement '{v}' (cimss|bd|avn|funktop|rainbow|gray)"
        ))),
    }
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
    fn visible_physics_helper_assigns_every_binding_control() {
        let mut params = RenderParams::new(PathBuf::from("input"));
        apply_visible_physics_controls(
            &mut params,
            0.2,
            true,
            false,
            false,
            false,
            0.5,
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
            simsat_engine::clouds::DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE,
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
        apply_land_appearance_controls(&mut params, false, 1.6, false, 0.08, 0.65, 1.5).unwrap();
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
         render_visible_rgb / render_geocolor / render_sandwich / render_visible_bands / \
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
    m.add_function(wrap_pyfunction!(render_visible_bands, m)?)?;
    m.add_function(wrap_pyfunction!(render_ir, m)?)?;
    m.add_function(wrap_pyfunction!(render_water_vapor, m)?)?;
    m.add_function(wrap_pyfunction!(render_precipitable_water, m)?)?;
    m.add_function(wrap_pyfunction!(render_cloud_top_temp, m)?)?;
    m.add_function(wrap_pyfunction!(render_cloud_optical_depth, m)?)?;
    m.add_function(wrap_pyfunction!(render_cloud_layer, m)?)?;
    m.add_function(wrap_pyfunction!(render_perspective, m)?)?;
    m.add_function(wrap_pyfunction!(set_verbose, m)?)?;
    m.add_class::<Georef>()?;
    Ok(())
}
