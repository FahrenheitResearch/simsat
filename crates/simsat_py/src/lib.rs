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
//!
//! Default `view='topdown'` (map-registered, north-up — the natural fit for a top-down
//! Lambert map); `view='geo'` gives the from-space geostationary view.
//!
//! Every function also takes `threads=` (cap the render worker threads; the rayon pool
//! is GLOBAL and built once per process — the first render call's value wins, see
//! [`apply_thread_cap`]) and returns honesty metadata on the [`Georef`]
//! (`time_is_fallback`, `ground_source`, `ground_status`), with `UserWarning`s raised
//! on a fabricated-date or downgraded-ground render ([`warn_downgrades`]).

use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::path::PathBuf;

use simsat_engine::api::{
    self, BlueMarble, ExtentKind, FrameData, GroundSource, Product, RenderParams, RenderResult,
    SunOverride,
};
use simsat_engine::camera::{ResolutionMode, SatellitePreset, ViewMode};
use simsat_engine::clouds::StepQuality;
use simsat_engine::derived::DerivedField;
use simsat_engine::ir_enhance::IrEnhancement;
use simsat_engine::topdown::{configure_global_rayon, effective_thread_count};
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
/// `threads` (default None = all cores, or RAYON_NUM_THREADS) caps the render worker
/// threads for THIS PROCESS. The pool is global and built ONCE — the first render call's
/// value wins; later calls cannot change it. Available on every render function here.
///
/// Returns `(rgb, georef)` where `rgb` is a numpy `H x W x 3` uint8 array (row 0 = north).
#[pyfunction]
#[pyo3(signature = (
    input, *, sat="goes-east", view="topdown", timestep=0, resolution="native", margin=0.0,
    exposure=None, multiscatter=true, steps="offline", clouds=true, sun_elev=None,
    sun_az=None, cache=None, bluemarble=None, bluemarble_month=None, bluemarble_download=true,
    threads=None
))]
#[allow(clippy::too_many_arguments)]
fn render_visible_rgb<'py>(
    py: Python<'py>,
    input: String,
    sat: &str,
    view: &str,
    timestep: usize,
    resolution: &str,
    margin: f64,
    exposure: Option<f64>,
    multiscatter: bool,
    steps: &str,
    clouds: bool,
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
        exposure,
        multiscatter,
        steps,
        clouds,
        sun_elev,
        sun_az,
        cache,
        bluemarble,
        bluemarble_month,
        bluemarble_download,
    )?;
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
    exposure=None, multiscatter=true, steps="offline", clouds=true, sun_elev=None,
    sun_az=None, cache=None, bluemarble=None, bluemarble_month=None, bluemarble_download=true,
    threads=None
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
    exposure: Option<f64>,
    multiscatter: bool,
    steps: &str,
    clouds: bool,
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
        exposure,
        multiscatter,
        steps,
        clouds,
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
    exposure=None, multiscatter=true, steps="offline", clouds=true, sun_elev=None,
    sun_az=None, cache=None, bluemarble=None, bluemarble_month=None, bluemarble_download=true,
    threads=None
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
    exposure: Option<f64>,
    multiscatter: bool,
    steps: &str,
    clouds: bool,
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
        exposure,
        multiscatter,
        steps,
        clouds,
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
    multiscatter=true, steps="offline", sun_elev=None, sun_az=None, cache=None,
    bluemarble=None, bluemarble_month=None, bluemarble_download=true, threads=None
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
    multiscatter: bool,
    steps: &str,
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
        None,
        multiscatter,
        steps,
        true,
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
    exposure: Option<f64>,
    multiscatter: bool,
    steps: &str,
    clouds: bool,
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
    if let Some(e) = exposure {
        params.exposure = e;
    }
    params.multiscatter = multiscatter;
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
    Ok(params)
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
    };
    let crs_params = build_crs_params(py, g)?;
    Ok(Georef {
        view: g.view.slug().to_string(),
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
    })
}

/// Raise Python `UserWarning`s for honesty downgrades the caller would otherwise never
/// see in the returned arrays. The same facts ride on the [`Georef`] attributes
/// (`time_is_fallback`, `ground_source`, `ground_status`) for programmatic use.
fn warn_downgrades(py: Python<'_>, result: &RenderResult, sun_dependent: bool) {
    let mut msgs: Vec<String> = Vec::new();
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

/// The `simsat` Python module.
#[pymodule]
fn simsat(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "__doc__",
        "SimSat — physically-based simulated visible/IR satellite imagery from WRF output. \
         render_visible_rgb / render_geocolor / render_sandwich / render_visible_bands / \
         render_ir / render_water_vapor / render_precipitable_water / render_cloud_top_temp / \
         render_cloud_optical_depth each return a numpy array plus a Georef (projection params \
         + imshow extent + lat/lon mesh + render-honesty metadata: time_is_fallback, \
         ground_source, ground_status). The three derived-field functions return RAW physical \
         scalar arrays (mm / Kelvin / dimensionless) to plot with your own colormaps. Every \
         function takes threads= (per-process rayon cap; the FIRST render call's value wins).",
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
    m.add_class::<Georef>()?;
    Ok(())
}
