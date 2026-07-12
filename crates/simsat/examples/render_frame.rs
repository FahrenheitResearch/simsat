//! `render_frame` — headless "render one full composited visible frame to PNG".
//!
//! A headless render harness: CPU by default, with an explicit synchronous GPU-preview
//! backend for visual QA without the Studio GUI. It is a THIN wrapper over
//! [`simsat::api::render`] (the one shared
//! render assembly behind the studio, this example, and the Python binding): parse the
//! CLI, call `api::render`, then do the output glue (optional canvas letterbox, PNG
//! write, optional sat-store write, the `SUMMARY` line). Because the RGB comes from the
//! same `api::render` -> `clouds::render_cloud_frame_rgba` /
//! `topdown::render_topdown_frame_rgba` the studio uses, the PNG is byte-identical to
//! what the studio displays.
//!
//! USAGE (key=value args, any order):
//!
//!   render_frame input=<path> out=<file.png> [key=value ...]
//!
//!   input=<path>       REQUIRED. A wrfout file (ingested to a brick if not cached under
//!                      `cache=`) OR a cached run's `run.json`.
//!   out=<file.png>     REQUIRED. Output PNG path (RGB8, row 0 = north).
//!   backend=<mode>     cpu | gpu-preview (default cpu). GPU preview temporarily selects
//!                      a compatible Visible/clouds-on configuration and reports every change.
//!   sat=<preset>       goes-east | goes-west | himawari   (default goes-east)
//!   timestep=<n>       Time index (default 0).
//!   resolution=<mode>  native | abi1km | abi2km           (default native)
//!                      native = one output pixel per source grid cell. ABI 1/2 km are
//!                      output sampling choices and may upsample coarse or downsample fine data.
//!   margin=<frac>      zoom-out margin, a FRACTION of the domain added on each side
//!                      (default 0.0 = edge-to-edge; 0.3 = the domain in a 30% earth margin).
//!   aerosol-optical-depth=<f>  aerosol AOD, 0.0..=0.6 (default DEFAULT_AOD = 0.05).
//!   rh-aerosol-swelling=<b>    on | off — apply the documented 1.5x aerosol swelling
//!                              multiplier (default off).
//!   atmosphere-correction=<b>  on | off — product-facing daytime aerial-veil
//!                              correction (default on; off = full airlight).
//!   terrain-atmosphere=<b>     on | off — shorten atmospheric columns to WRF terrain
//!                              elevation (default on; off = legacy sea-level geometry).
//!   multiscatter=<b>   on | off  — M5 Wrenninge octaves   (default on).
//!   beer-powder=<b>    on | off  — Schneider direct-sun shaping (default off).
//!   clouds=<b>         on | off  — off = surface only (QA terrain/glint)  (default on).
//!   fractional-clouds=<b> on | off — use model cloud fraction; off = legacy full cells
//!                              (default on; falls back safely when the field is absent).
//!   cloud-optical-depth-scale=<f>  cloud OD calibration, 0.0..=4.0 (shipped default 0.15;
//!                      1.0 = unscaled model extinction).
//!   feather-exposed-domain-edges=<b> on | off — fade finished clouds at camera-exposed
//!                      finite WRF boundaries (owner-selected v0.1.5 default on).
//!   granulation=<b>    on | off  — sub-grid cloud edge erosion (default off).
//!   steps=<quality>    offline | interactive              (default offline).
//!   sun-elev=<deg>     OPTIONAL sun-elevation override (else true solar geometry).
//!   sun-az=<deg>       OPTIONAL sun-azimuth override (deg from north).
//!   exposure=<f>       Display gain before the ABI stretch (default DEFAULT_EXPOSURE).
//!   land-sza-normalization=<b> on | off - bounded land-only solar-zenith display
//!                              normalization (owner-selected default on).
//!   land-sza-max-gain=<f>      upper bound for that normalization (default 1.6).
//!   land-dark-toe=<b>          on | off - bounded dark-land reflectance lift
//!                              (owner-selected default on).
//!   land-dark-toe-knee=<f>     linear-reflectance identity knee (default 0.08).
//!   land-dark-toe-gamma=<f>    toe exponent, 0.05..=1 (default 0.65; 1 = identity).
//!   land-dark-toe-max-gain=<f> upper bound for the toe lift (default 1.5).
//!   cache=<dir>        Brick cache root + seasonal Blue Marble cache.
//!   bluemarble=<path>  OPTIONAL single-file Blue Marble override (default seasonal pack).
//!   bluemarble-month=<MM>  Force month 1..=12 (what-if; default day-of-year blend).
//!   bluemarble-download=<b>  on|off lazy month fetch (default on).
//!   view=<mode>        geo | topdown  (default geo). topdown = the map-registered view.
//!   canvas=<WxH>       OPTIONAL letterbox into a fixed figure size (black pad).
//!   threads=<N>        OPTIONAL rayon thread cap (else honor RAYON_NUM_THREADS).
//!   store=<dir>        ALSO write the visible frame to this sat-store root (geo only).
//!   sector=<token>     Store run sector token (default: the input's run id).
//!   geocolor=<b>       on | off  — render the GeoColor day/night blend (true-color day +
//!                      colored band-13 IR night) instead of plain visible  (default off).
//!   sandwich=<b>       on | off  — render the Sandwich composite (true-color visible base +
//!                      color-enhanced band-13 IR overlaid on the cold cloud tops)  (default off).
//!   product=<p>        visible | geocolor | sandwich | cloud-layer. `cloud-layer` renders the
//!                      WEB-MAP CLOUD LAYER pair (the Mapbox-class compositing product): `out`
//!                      becomes the cloud RGBA PNG (straight alpha), plus `<stem>_shadow.png`
//!                      (grayscale multiply layer, 255 = no shadow) and `<stem>.json` (the
//!                      Mapbox ImageSource corner lon/lats + EPSG:3857 extent). Top-down by
//!                      definition (view= is ignored for it).
//!   composite-out=<p>  With product=cloud-layer: ALSO write the in-process synthetic composite
//!                      proof (shadow multiply + cloud over a checkerboard basemap) to this PNG.
//!   eye=lat,lon,alt    FREE-PERSPECTIVE camera eye (deg, deg, metres above the sphere).
//!   look=lat,lon,alt   The perspective look-at target. eye= + look= together switch the
//!                      render to Product::Perspective (the angled-3D full composite over
//!                      our Blue Marble ground; sky rays composite the limb/space).
//!   fov=<deg>          Perspective HORIZONTAL field of view (default 40).
//!   camsize=<WxH>      Perspective image dims (default 1280x720).
//!   perspective-layer=<b>  on | off (default off): render the CLOUD FIELD ONLY through the
//!                      perspective camera; `out` becomes a straight-alpha RGBA PNG (for
//!                      compositing over a host 3-D map with a matching camera).
//!                      A FLYOVER is N invocations along your own eye/look path.
//!   ground-gain=<f>    OVERRIDE the shipped GROUND LIFT (default 1.0 = neutral).
//!   cloud-softclip=<f> OVERRIDE the shipped highlight knee (default 0.65);
//!                      1.0 = disable the shoulder (hard clamp).
//!   cloud-highlight-max=<f>  OVERRIDE the physical reflectance ceiling mapped to white
//!                      (default 1.25); larger values preserve more detail.
//!   topdown-cloudnorm=<f>  OVERRIDE the baked TOP-DOWN CLOUD NORMALIZATION
//!                      (topdown::TOPDOWN_CLOUD_NORM); 1.0 = no normalization.
//!   synthetic-green=<b> on | off (default off) — the ABI SYNTHETIC-GREEN display mode
//!                      prototype (low-sun visible pass): display green becomes
//!                      G' = 0.45*R + 0.45*B + 0.10*G (Bah et al. 2018), the real
//!                      GOES-R true-color green arithmetic (khaki/mauve casts are
//!                      impossible in it by construction). A/B judgment flag; applies
//!                      to the whole process (all products rendered by this run).
//!   bands-out=<path.bin>  QA/diagnostic: ALSO render the SAME scene through
//!                      `Product::VisibleBands` and write the RAW pre-tonemap,
//!                      pre-exposure reflectance (`nx*ny*3` little-endian f32, row 0 =
//!                      north, band order R,G,B per pixel) + print a `BANDS ...` line.
//!                      Splits the pipeline for offline analysis: physics color/texture
//!                      is in the .bin; the display transform's contribution is the
//!                      delta between the .bin pushed through the (pure) tonemap and
//!                      the PNG. Costs a second full render.
//!
//! On completion it prints a one-line `SUMMARY ...` (dims, on-earth fraction, centre sun
//! elevation, exposure, multiscatter, wall time, peak/median display luminance, the
//! strided physical cloud coverage/peak reflectance for the geostationary view (`n/a`
//! for top-down, where that separate diagnostic march is not computed), and
//! `cloud_lum_p90_p10` — the WS2 bright-cloud contrast metric: P90-P10 display
//! luminance over strongly-cloudy output pixels, computed from the frame itself so it
//! works for BOTH geo and top-down views) to stdout.
//!
//! NOTE: the old `supersample=` QA flag was REMOVED with the api refactor — it was a
//! documented, tested-and-REJECTED anti-alias experiment (the cloud march is already
//! trilinear, so a box average changed nothing at N^2 cost). The single shared
//! `api::render` path has no supersample raster; nothing in the shipping pipeline used it.

use std::path::{Path, PathBuf};
use std::time::Instant;

use image::{GrayImage, RgbImage, RgbaImage};
use simsat::api::{self, BlueMarble, FrameData, Product, RenderBackend, RenderParams, SunOverride};
use simsat::atmosphere::DEFAULT_AOD;
use simsat::camera::{PerspectiveCamera, ResolutionMode, SatellitePreset, ViewMode};
use simsat::clouds::{
    CloudFrameStats, CloudMultiscatterMode, DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE, StepQuality,
};
use simsat::gpu::RenderedFrame;
use simsat::ingest;
use simsat::render::{
    DEFAULT_EXPOSURE, LAND_DARK_TOE_GAMMA, LAND_DARK_TOE_KNEE, LAND_DARK_TOE_MAX_GAIN,
    LAND_SZA_MAX_GAIN, LandAppearanceConfig,
};
use simsat::store_out::{self, VisibleFrame};
use simsat::topdown;
use simsat::web_layer;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = run(&args) {
        eprintln!("render_frame: {e}");
        eprintln!("run with no args (or `--help`) for usage.");
        std::process::exit(1);
    }
}

/// Parsed command-line options.
struct Opts {
    input: PathBuf,
    out: PathBuf,
    backend: RenderBackend,
    sat: SatellitePreset,
    timestep: usize,
    resolution: ResolutionMode,
    /// Zoom-out / domain margin as a FRACTION added on each side (0.0 = edge-to-edge).
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
    multiscatter: bool,
    cloud_multiscatter: Option<CloudMultiscatterMode>,
    beer_powder: bool,
    steps: StepQuality,
    sun_elev_override: Option<f64>,
    sun_az_override: Option<f64>,
    clouds: bool,
    fractional_clouds: bool,
    cloud_optical_depth_scale: f32,
    feather_exposed_domain_edges: bool,
    granulation: bool,
    topdown_stratiform_regularization: bool,
    exposure: f64,
    cache: PathBuf,
    bluemarble: Option<PathBuf>,
    bluemarble_month: Option<u32>,
    bluemarble_download: bool,
    view: ViewMode,
    canvas: Option<(usize, usize)>,
    threads: Option<usize>,
    store: Option<PathBuf>,
    sector: Option<String>,
    /// Render the GeoColor day/night blend (true-color day + colored band-13 IR night)
    /// instead of the plain visible frame. The output is still a baked RGB composite.
    geocolor: bool,
    /// Render the Sandwich composite (true-color visible base + color-enhanced band-13 IR
    /// overlaid on the cold cloud tops) instead of the plain visible frame. Takes precedence
    /// over `geocolor`. The output is still a baked RGB composite.
    sandwich: bool,
    /// Render the WEB-MAP CLOUD LAYER pair (`product=cloud-layer`): the cloud-only RGBA
    /// (straight alpha in the PNG) + the ground cloud-shadow multiply PNG + a JSON
    /// sidecar with the Mapbox ImageSource corner lon/lats, on a Web-Mercator grid.
    /// Takes precedence over `sandwich`/`geocolor`.
    cloud_layer: bool,
    /// With `product=cloud-layer`: ALSO write the in-process SYNTHETIC COMPOSITE PROOF
    /// (shadow-multiply then cloud-over on a checkerboard basemap) to this PNG — the
    /// no-Mapbox registration/appearance check.
    composite_out: Option<PathBuf>,
    /// FREE-PERSPECTIVE camera (tier 2): `eye=lat,lon,alt_m` + `look=lat,lon,alt_m`
    /// trigger the perspective product (both required together).
    eye: Option<(f64, f64, f64)>,
    look: Option<(f64, f64, f64)>,
    /// Horizontal field of view (deg) for the perspective camera (default 40).
    fov: f64,
    /// Perspective image dims `camsize=WxH` (default 1280x720).
    camsize: (usize, usize),
    /// Perspective CLOUD-LAYER-ONLY mode: `out` becomes a straight-alpha RGBA PNG of
    /// the cloud field alone (for compositing over a host 3-D map).
    perspective_layer: bool,
    /// Appearance-pass tuning overrides (None = the baked engine defaults). Future tuning
    /// knobs; the shipped look already comes from the baked constants.
    ground_gain: Option<f64>,
    cloud_softclip: Option<f64>,
    cloud_highlight_max: Option<f64>,
    topdown_cloud_norm: Option<f64>,
    /// QA/diagnostic: also dump the raw pre-tonemap reflectance bands to this path.
    bands_out: Option<PathBuf>,
    /// ABI synthetic-green display mode (prototype A/B; default off).
    synthetic_green: bool,
}

fn run(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return Ok(());
    }
    let opts = parse_opts(args)?;
    eprintln!(
        "render_frame: input={} product={} view={} sat={} ts={} res={} margin={:.2} \
         aod={:.3} rh-swelling={} atmosphere-correction={} terrain-atmosphere={} \
         land-sza={}({:.2}) land-dark-toe={}({:.3}/{:.2}/{:.2}) \
         cloud-multiscatter={} beer-powder={} clouds={} fractional-clouds={} \
         cloud-od-scale={:.3} feather-exposed-edges={} granulation={} \
         topdown-stratiform-regularization={} \
         steps={} sun-elev={} \
         exposure={:.3} backend={} canvas={} threads={}",
        opts.input.display(),
        product_label(&opts),
        opts.view.slug(),
        opts.sat.slug(),
        opts.timestep,
        opts.resolution.label(),
        opts.margin,
        opts.aerosol_optical_depth,
        opts.rh_aerosol_swelling,
        opts.atmosphere_correction,
        opts.terrain_atmosphere,
        opts.land_sza_normalization,
        opts.land_sza_max_gain,
        opts.land_dark_toe,
        opts.land_dark_toe_knee,
        opts.land_dark_toe_gamma,
        opts.land_dark_toe_max_gain,
        resolved_cloud_multiscatter(&opts).slug(),
        opts.beer_powder,
        opts.clouds,
        opts.fractional_clouds,
        opts.cloud_optical_depth_scale,
        opts.feather_exposed_domain_edges,
        opts.granulation,
        opts.topdown_stratiform_regularization,
        if opts.steps == StepQuality::Offline {
            "offline"
        } else {
            "interactive"
        },
        opts.sun_elev_override
            .map(|e| format!("{e:.1}"))
            .unwrap_or_else(|| "actual".to_string()),
        opts.exposure,
        opts.backend.slug(),
        opts.canvas
            .map(|(w, h)| format!("{w}x{h}"))
            .unwrap_or_else(|| "native".to_string()),
        opts.threads
            .map(|n| n.to_string())
            .unwrap_or_else(|| "auto".to_string()),
    );

    // Rayon thread cap (before ANY parallel work): the explicit `threads=` override, else
    // `RAYON_NUM_THREADS`, else all cores. Keeps a 16-way pool from oversubscribing.
    topdown::configure_global_rayon(topdown::effective_thread_count(
        opts.threads,
        std::env::var("RAYON_NUM_THREADS").ok().as_deref(),
    ));
    simsat::platform::lower_ingest_thread_priority();
    // ABI synthetic-green display mode (prototype A/B; process-global, default off).
    simsat::render::set_synthetic_green(opts.synthetic_green);
    if opts.synthetic_green {
        eprintln!("render_frame: ABI SYNTHETIC-GREEN display mode ON (G' = 0.45R + 0.45B + 0.10G)");
    }

    // ── the one shared render assembly ──
    let params = render_params(&opts);
    // The web-map cloud layer is a different delivery (RGBA + shadow + sidecar, not one
    // RGB PNG) — its own output path.
    if opts.cloud_layer && opts.backend == RenderBackend::Cpu {
        return run_cloud_layer(&opts, &params);
    }
    // Perspective (eye= + look=) wins; then Sandwich > GeoColor > plain visible. All are
    // baked RGB(A) composites (FrameData::Visible).
    let product = if params.perspective.is_some() {
        Product::Perspective {
            cloud_layer_only: opts.perspective_layer,
        }
    } else if opts.cloud_layer {
        // CPU cloud-layer delivery returned above. GPU-preview deliberately receives
        // the requested product here so the API reports its temporary -> Visible
        // substitution rather than the CLI hiding it.
        Product::CloudLayer
    } else if opts.sandwich {
        Product::Sandwich
    } else if opts.geocolor {
        Product::GeoColor
    } else {
        Product::VisibleRgb
    };
    let t0 = Instant::now();
    let result = api::render(&params, product)?;
    if let Some(adapter) = &result.gpu_adapter {
        eprintln!("render_frame: GPU preview adapter: {adapter}");
    }
    for adjustment in &result.diagnostics {
        eprintln!(
            "render_frame: GPU preview temporary adjustment: {}",
            adjustment.label()
        );
    }
    let wall = t0.elapsed();
    let (rgb, rgba) = match &result.data {
        FrameData::Visible { rgb, rgba } => (rgb, rgba),
        _ => return Err("expected a visible frame".to_string()),
    };
    let (rnx, rny) = (result.nx, result.ny);

    eprintln!(
        "render_frame: rendered {}x{} ({}, {}){} in {:.3}s",
        rnx,
        rny,
        opts.view.slug(),
        opts.resolution.label(),
        if result.res_clamped {
            " [clamped to cap]"
        } else {
            ""
        },
        wall.as_secs_f64(),
    );
    // The what-if labeling discipline: a perspective frame always states its camera.
    if let Some(cam) = &result.georef.camera_pose {
        println!(
            "PERSPECTIVE {} layer_only={} ground_frac={:.3}",
            cam.label(),
            opts.perspective_layer,
            result.georef.lat.iter().filter(|v| v.is_finite()).count() as f64
                / (rnx * rny).max(1) as f64,
        );
    }

    // ── output: RGBA (perspective cloud layer) or RGB (+ optional canvas letterbox) ──
    let (final_nx, final_ny) = if opts.perspective_layer {
        // The cloud-layer-only perspective frame delivers STRAIGHT-alpha RGBA (the
        // canvas letterbox is RGB-only and skipped for it).
        if opts.canvas.is_some() {
            eprintln!("render_frame: canvas= skipped for the RGBA perspective layer.");
        }
        write_rgba8_png(&opts.out, rnx, rny, &web_layer::unpremultiply_rgba(rgba))?;
        (rnx, rny)
    } else {
        let (fnx, fny, final_rgb) = match opts.canvas {
            Some((cw, ch)) => (cw, ch, topdown::letterbox_rgb(rgb, rnx, rny, cw, ch)),
            None => (rnx, rny, rgb.clone()),
        };
        write_rgb8_png(&opts.out, fnx, fny, &final_rgb)?;
        (fnx, fny)
    };

    // ── optional sat-store write (geostationary only — the store carries the scan mesh) ──
    if opts.store.is_some() {
        if opts.backend == RenderBackend::GpuPreview {
            eprintln!(
                "render_frame: store write skipped (GPU previews are display-only; store output stays CPU)."
            );
        } else if opts.view == ViewMode::TopDownMap || params.perspective.is_some() {
            eprintln!("render_frame: store write skipped (store= is for the geostationary view).");
        } else {
            write_store(&opts, &result, rgba)?;
        }
    }

    // ── optional raw-bands diagnostic dump (a second render through VisibleBands) ──
    if let Some(bands_path) = &opts.bands_out {
        let mut bands_params = params.clone();
        if bands_params.backend == RenderBackend::GpuPreview {
            eprintln!(
                "render_frame: bands-out uses CPU because GPU preview is a finished Visible display backend."
            );
            bands_params.backend = RenderBackend::Cpu;
        }
        let bands_result = api::render(&bands_params, Product::VisibleBands)?;
        let reflectance = match &bands_result.data {
            FrameData::Bands { reflectance } => reflectance,
            _ => return Err("expected a bands frame".to_string()),
        };
        let mut bytes = Vec::with_capacity(reflectance.len() * 4);
        for v in reflectance {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(bands_path, &bytes)
            .map_err(|e| format!("write bands {}: {e}", bands_path.display()))?;
        println!(
            "BANDS file={} dims={}x{} channels=3 dtype=f32le rows=north-first",
            bands_path.display(),
            bands_result.nx,
            bands_result.ny,
        );
    }

    // ── stats for the manifest ──
    let (on_earth, peak_lum, median_lum) = display_luma_stats(rgba);
    let on_earth_frac = on_earth as f64 / (rnx * rny).max(1) as f64;
    let (cloud_lum_p90_p10, cloud_lum_frac) = cloud_contrast_stat(rgba);
    let (cloud_frac, peak_refl, peak_sun_refl) =
        cloud_stat_summary_fields(result.cloud_stats.as_ref());

    eprintln!("render_frame: wrote {}", opts.out.display());
    println!(
        "SUMMARY file={} backend={} view={} dims={}x{} canvas={} render_dims={}x{} res={}{} sat={} \
         sun_elev={:.1} exposure={:.3} aod={:.3} rh_aerosol_swelling={} \
         atmosphere_correction={} terrain_atmosphere={} cloud_multiscatter={} beer_powder={} \
         clouds={} fractional_clouds_requested={} cloud_optical_depth_scale={:.3} \
         feather_exposed_domain_edges={} granulation={} \
         steps={} synthetic_green={} \
         on_earth_frac={:.3} \
         peak_lum={:.3} median_lum={:.3} cloud_frac={} peak_reflectance={} \
         peak_sun_reflectance={} cloud_lum_p90_p10={:.4} cloud_lum_frac={:.3} wall_s={:.3}",
        opts.out.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
        result.backend.slug(),
        opts.view.slug(),
        final_nx,
        final_ny,
        opts.canvas
            .map(|(w, h)| format!("{w}x{h}"))
            .unwrap_or_else(|| "none".to_string()),
        rnx,
        rny,
        opts.resolution.label().replace(' ', "_"),
        if result.res_clamped { "[clamped]" } else { "" },
        opts.sat.slug(),
        result.sun_elev_deg,
        opts.exposure,
        opts.aerosol_optical_depth,
        opts.rh_aerosol_swelling,
        opts.atmosphere_correction,
        opts.terrain_atmosphere,
        resolved_cloud_multiscatter(&opts).slug(),
        opts.beer_powder,
        opts.clouds,
        opts.fractional_clouds,
        opts.cloud_optical_depth_scale,
        opts.feather_exposed_domain_edges,
        opts.granulation,
        if opts.steps == StepQuality::Offline {
            "offline"
        } else {
            "interactive"
        },
        opts.synthetic_green,
        on_earth_frac,
        peak_lum,
        median_lum,
        cloud_frac,
        peak_refl,
        peak_sun_refl,
        cloud_lum_p90_p10,
        cloud_lum_frac,
        wall.as_secs_f64(),
    );
    Ok(())
}

/// Render + deliver the WEB-MAP CLOUD LAYER pair (`product=cloud-layer`): the cloud
/// RGBA PNG (STRAIGHT alpha — the PNG/browser/Mapbox convention; the engine computes
/// premultiplied, see `web_layer`), the grayscale shadow multiply PNG
/// (`<out stem>_shadow.png`, 255 = no shadow), the JSON sidecar (`<out stem>.json`,
/// Mapbox ImageSource corners + EPSG:3857 extent), and optionally the in-process
/// synthetic composite proof (`composite-out=`): shadow-multiply then cloud-over on a
/// checkerboard basemap — the registration check that needs no Mapbox.
fn run_cloud_layer(opts: &Opts, params: &RenderParams) -> Result<(), String> {
    let t0 = Instant::now();
    let result = api::render(params, Product::CloudLayer)?;
    let wall = t0.elapsed();
    let (rgba_premul, shadow) = match &result.data {
        FrameData::CloudLayer {
            rgba_premul,
            shadow,
        } => (rgba_premul, shadow),
        _ => return Err("expected a cloud-layer frame".to_string()),
    };
    let (nx, ny) = (result.nx, result.ny);

    // 1. The cloud RGBA PNG (straight alpha for PNG delivery).
    let straight = web_layer::unpremultiply_rgba(rgba_premul);
    write_rgba8_png(&opts.out, nx, ny, &straight)?;

    // 2. The shadow multiply PNG (grayscale; 255 = no shadow).
    let shadow_path = sibling_path(&opts.out, "_shadow.png");
    write_gray8_png(&shadow_path, nx, ny, &web_layer::shadow_to_gray(shadow))?;

    // 3. The JSON sidecar (Mapbox ImageSource corners + extent + semantics).
    let corners = result
        .georef
        .mercator_corners_lonlat
        .ok_or("cloud layer missing mercator corners")?;
    let extent = result.georef.extent;
    let grid = web_layer::MercatorGrid {
        nx,
        ny,
        x_min: extent[0],
        x_max: extent[1],
        y_min: extent[2],
        y_max: extent[3],
    };
    let t = result.time;
    let mut hh = t.ut as u32;
    let mut mm = ((t.ut - hh as f64) * 60.0).round() as u32;
    if mm >= 60 {
        hh += 1;
        mm -= 60;
    }
    let time_iso = format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:00Z",
        t.year, t.month, t.day, hh, mm
    );
    let sidecar = web_layer::cloud_layer_sidecar_json(
        &grid,
        opts.out.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
        shadow_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?"),
        &time_iso,
        result.sun_elev_deg,
        result.granulation,
    );
    let sidecar_path = sibling_path(&opts.out, ".json");
    std::fs::write(&sidecar_path, &sidecar)
        .map_err(|e| format!("write sidecar {}: {e}", sidecar_path.display()))?;

    // 4. Optional synthetic composite proof over a checkerboard basemap.
    if let Some(comp_path) = &opts.composite_out {
        let base = web_layer::checker_basemap(nx, ny, 32);
        let composed = web_layer::composite_over_basemap(&base, &straight, shadow, nx * ny);
        write_rgb8_png(comp_path, nx, ny, &composed)?;
        eprintln!(
            "render_frame: wrote composite proof {}",
            comp_path.display()
        );
    }

    // 5. Layer stats: coverage (alpha > 0), mean alpha, shadow floor.
    let covered = straight.chunks_exact(4).filter(|p| p[3] > 0).count();
    let mean_alpha = straight
        .chunks_exact(4)
        .map(|p| p[3] as f64 / 255.0)
        .sum::<f64>()
        / (nx * ny).max(1) as f64;
    let shadow_min = shadow.iter().cloned().fold(1.0f32, f32::min);
    let shadow_mean = shadow.iter().map(|&s| s as f64).sum::<f64>() / shadow.len().max(1) as f64;
    let nw = corners[0];
    let se = corners[2];
    eprintln!(
        "render_frame: wrote {} + {} + {}",
        opts.out.display(),
        shadow_path.display(),
        sidecar_path.display()
    );
    println!(
        "LAYERSUMMARY file={} dims={}x{} crs=EPSG:3857 corner_nw={:.4},{:.4} \
         corner_se={:.4},{:.4} sun_elev={:.1} cover_frac={:.3} mean_alpha={:.3} \
         shadow_min={:.3} shadow_mean={:.3} beer_powder={} fractional_clouds_requested={} \
         feather_exposed_domain_edges={} granulation={} wall_s={:.3}",
        opts.out.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
        nx,
        ny,
        nw[0],
        nw[1],
        se[0],
        se[1],
        result.sun_elev_deg,
        covered as f64 / (nx * ny).max(1) as f64,
        mean_alpha,
        shadow_min,
        shadow_mean,
        opts.beer_powder,
        opts.fractional_clouds,
        opts.feather_exposed_domain_edges,
        result.granulation,
        wall.as_secs_f64(),
    );
    Ok(())
}

/// `path` with `suffix` appended to its file STEM (`clouds.png` + `_shadow.png` ->
/// `clouds_shadow.png`; `clouds.png` + `.json` -> `clouds.json`).
fn sibling_path(path: &Path, suffix: &str) -> PathBuf {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("layer");
    path.with_file_name(format!("{stem}{suffix}"))
}

/// Write RGBA8 bytes (row 0 = north, straight alpha) to a PNG.
fn write_rgba8_png(path: &Path, nx: usize, ny: usize, rgba: &[u8]) -> Result<(), String> {
    if rgba.len() != nx * ny * 4 {
        return Err(format!("rgba byte count {} != {}x{}x4", rgba.len(), nx, ny));
    }
    let img = RgbaImage::from_fn(nx as u32, ny as u32, |x, y| {
        let o = (y as usize * nx + x as usize) * 4;
        image::Rgba([rgba[o], rgba[o + 1], rgba[o + 2], rgba[o + 3]])
    });
    img.save(path)
        .map_err(|e| format!("write PNG {}: {e}", path.display()))
}

/// Write 8-bit grayscale bytes (row 0 = north) to a PNG.
fn write_gray8_png(path: &Path, nx: usize, ny: usize, gray: &[u8]) -> Result<(), String> {
    if gray.len() != nx * ny {
        return Err(format!("gray byte count {} != {}x{}", gray.len(), nx, ny));
    }
    let img = GrayImage::from_fn(nx as u32, ny as u32, |x, y| {
        image::Luma([gray[y as usize * nx + x as usize]])
    });
    img.save(path)
        .map_err(|e| format!("write PNG {}: {e}", path.display()))
}

/// Display luminance threshold above which an output pixel is counted as
/// STRONGLY CLOUDY for the [`cloud_contrast_stat`] metric. `0.70` display sits at
/// `rho' ~ 0.49` (just under the soft-clip knee at neutral exposure 1.0), so the population is
/// the bright cloud band whose texture the bounded shoulder is meant to preserve.
const CLOUD_LUM_MIN: f64 = 0.70;

/// The WS2 bright-cloud CONTRAST metric: `(P90 - P10, fraction)` of the display
/// luminance over STRONGLY-CLOUDY pixels (mean-RGB luminance >= [`CLOUD_LUM_MIN`],
/// on-earth only). Computed from the OUTPUT frame — not from the march's cloud stats —
/// so it works identically for the geostationary AND top-down views (the top-down path
/// has no physical cloud stats). A flat white square scores ~0; recovered texture
/// raises it. `(0.0, 0.0)` when fewer than 32 pixels qualify (no meaningful cloud).
fn cloud_contrast_stat(rgba: &[u8]) -> (f64, f64) {
    let mut lums: Vec<f64> = Vec::new();
    let mut on_earth = 0usize;
    for px in rgba.chunks_exact(4) {
        if px[3] == 0 {
            continue;
        }
        on_earth += 1;
        let l = (px[0] as f64 + px[1] as f64 + px[2] as f64) / 3.0 / 255.0;
        if l >= CLOUD_LUM_MIN {
            lums.push(l);
        }
    }
    if lums.len() < 32 {
        return (0.0, 0.0);
    }
    lums.sort_by(f64::total_cmp);
    let q = |p: f64| lums[((lums.len() - 1) as f64 * p).round() as usize];
    let frac = lums.len() as f64 / on_earth.max(1) as f64;
    (q(0.90) - q(0.10), frac)
}

/// Stable `SUMMARY` tokens for the optional physical cloud-march diagnostic. The API
/// deliberately computes [`CloudFrameStats`] only for the geostationary view; top-down
/// still has the view-independent output-frame metrics above. Preserve that distinction
/// instead of serializing an absent diagnostic as three plausible-but-false zeroes.
fn cloud_stat_summary_fields(stats: Option<&CloudFrameStats>) -> (String, String, String) {
    match stats {
        Some(s) => (
            format!("{:.3}", s.cloud_fraction()),
            format!("{:.4}", s.max_reflectance),
            format!("{:.4}", s.max_sun_reflectance),
        ),
        None => ("n/a".to_string(), "n/a".to_string(), "n/a".to_string()),
    }
}

/// A short product label for the log line
/// (perspective > cloud-layer > sandwich > geocolor > visible).
fn product_label(opts: &Opts) -> &'static str {
    if opts.eye.is_some() {
        if opts.perspective_layer {
            "perspective-cloud-layer"
        } else {
            "perspective"
        }
    } else if opts.cloud_layer {
        "cloud-layer"
    } else if opts.sandwich {
        "sandwich"
    } else if opts.geocolor {
        "geocolor"
    } else {
        "visible"
    }
}

/// Build the shared [`RenderParams`] from the CLI options.
fn render_params(opts: &Opts) -> RenderParams {
    let bluemarble = match &opts.bluemarble {
        Some(path) => BlueMarble::SingleFile(path.clone()),
        None => BlueMarble::Seasonal {
            month_override: opts.bluemarble_month,
            download: opts.bluemarble_download,
        },
    };
    let sun_override = if opts.sun_elev_override.is_some() || opts.sun_az_override.is_some() {
        Some(SunOverride {
            elev_deg: opts.sun_elev_override,
            az_deg: opts.sun_az_override,
        })
    } else {
        None
    };
    RenderParams {
        input: opts.input.clone(),
        backend: opts.backend,
        satellite: opts.sat,
        timestep: opts.timestep,
        view: opts.view,
        resolution: opts.resolution,
        margin_frac: opts.margin as f32,
        aerosol_optical_depth: opts.aerosol_optical_depth,
        rh_aerosol_swelling: opts.rh_aerosol_swelling,
        atmosphere_correction: opts.atmosphere_correction,
        terrain_atmosphere: opts.terrain_atmosphere,
        land_appearance: LandAppearanceConfig {
            sza_normalization: opts.land_sza_normalization,
            sza_max_gain: opts.land_sza_max_gain,
            dark_toe: opts.land_dark_toe,
            dark_toe_knee: opts.land_dark_toe_knee,
            dark_toe_gamma: opts.land_dark_toe_gamma,
            dark_toe_max_gain: opts.land_dark_toe_max_gain,
        },
        exposure: opts.exposure,
        multiscatter: opts.multiscatter,
        cloud_multiscatter: opts.cloud_multiscatter,
        beer_powder: opts.beer_powder,
        steps: opts.steps,
        clouds: opts.clouds,
        fractional_clouds: opts.fractional_clouds,
        cloud_optical_depth_scale: opts.cloud_optical_depth_scale,
        feather_exposed_domain_edges: opts.feather_exposed_domain_edges,
        granulation: Some(opts.granulation),
        topdown_stratiform_regularization: opts.topdown_stratiform_regularization,
        sun_override,
        cache: opts.cache.clone(),
        bluemarble,
        ir_enhancement: None,
        derived_colormap: false,
        raster_override: None,
        ground_gain: opts.ground_gain,
        cloud_softclip: opts.cloud_softclip,
        cloud_highlight_max: opts.cloud_highlight_max,
        topdown_cloud_norm: opts.topdown_cloud_norm,
        perspective: perspective_camera_of(opts),
    }
}

/// The free-perspective camera from the CLI options (both `eye=` and `look=` present —
/// parse_opts guarantees they come in pairs), or `None` (no perspective requested).
fn perspective_camera_of(opts: &Opts) -> Option<PerspectiveCamera> {
    match (opts.eye, opts.look) {
        (Some(eye), Some(look)) => Some(PerspectiveCamera {
            eye_lat_deg: eye.0,
            eye_lon_deg: eye.1,
            eye_alt_m: eye.2,
            look_lat_deg: look.0,
            look_lon_deg: look.1,
            look_alt_m: look.2,
            fov_deg: opts.fov,
            width: opts.camsize.0,
            height: opts.camsize.1,
        }),
        _ => None,
    }
}

/// Write the rendered visible frame into a sat-store run (the M7 loop-render QA path).
fn write_store(opts: &Opts, result: &api::RenderResult, rgba: &[u8]) -> Result<(), String> {
    let t = result.time;
    let mut hh = t.ut as u16;
    let mut mm = ((t.ut - hh as f64) * 60.0).round() as u16;
    if mm >= 60 {
        hh += 1;
        mm -= 60;
    }
    let hhmm = hh * 100 + mm;
    let sector = opts
        .sector
        .clone()
        .unwrap_or_else(|| ingest::default_run_id(&opts.input));
    let rf = RenderedFrame {
        width: result.nx as u32,
        height: result.ny as u32,
        rgba: rgba.to_vec(),
    };
    let frame = VisibleFrame::from_rendered(
        &rf,
        result.raster.lat.clone(),
        result.raster.lon.clone(),
        sector,
        opts.sat,
        t.year,
        t.month,
        t.day,
        hhmm,
    );
    let store_root = opts.store.as_ref().expect("store set");
    match store_out::write_visible_frame(store_root, &frame) {
        Ok(w) => {
            eprintln!(
                "render_frame: wrote store frame simsat/{} t{:04} ({} bytes){}",
                w.run,
                w.hhmm,
                w.bytes,
                if w.created_run { " [new run]" } else { "" }
            );
            Ok(())
        }
        Err(e) => Err(format!("store write failed: {e}")),
    }
}

/// Write RGB8 bytes (row 0 = north) to a PNG.
fn write_rgb8_png(path: &Path, nx: usize, ny: usize, rgb: &[u8]) -> Result<(), String> {
    if rgb.len() != nx * ny * 3 {
        return Err(format!("rgb byte count {} != {}x{}x3", rgb.len(), nx, ny));
    }
    let img = RgbImage::from_fn(nx as u32, ny as u32, |x, y| {
        let o = (y as usize * nx + x as usize) * 3;
        image::Rgb([rgb[o], rgb[o + 1], rgb[o + 2]])
    });
    img.save(path)
        .map_err(|e| format!("write PNG {}: {e}", path.display()))
}

/// Over on-earth pixels (RGBA alpha != 0), the count, peak, and median of the display
/// luminance `(r+g+b)/3/255` in `[0,1]` — the exposure-applied brightness.
fn display_luma_stats(rgba: &[u8]) -> (usize, f64, f64) {
    let mut lums: Vec<f64> = Vec::new();
    let mut peak = 0.0f64;
    for px in rgba.chunks_exact(4) {
        if px[3] == 0 {
            continue;
        }
        let l = (px[0] as f64 + px[1] as f64 + px[2] as f64) / 3.0 / 255.0;
        peak = peak.max(l);
        lums.push(l);
    }
    let median = if lums.is_empty() {
        0.0
    } else {
        lums.sort_by(f64::total_cmp);
        lums[lums.len() / 2]
    };
    (lums.len(), peak, median)
}

// ── argument parsing ───────────────────────────────────────────────────────────

fn parse_opts(args: &[String]) -> Result<Opts, String> {
    let mut input: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut backend = RenderBackend::Cpu;
    let mut sat = SatellitePreset::GoesEast;
    let mut timestep = 0usize;
    let mut resolution = ResolutionMode::Native;
    let mut margin = 0.0f64;
    let mut aerosol_optical_depth = DEFAULT_AOD as f32;
    let mut rh_aerosol_swelling = false;
    let mut atmosphere_correction = true;
    let mut terrain_atmosphere = true;
    let land_defaults = LandAppearanceConfig::default();
    let mut land_sza_normalization = land_defaults.sza_normalization;
    let mut land_sza_max_gain = LAND_SZA_MAX_GAIN;
    let mut land_dark_toe = land_defaults.dark_toe;
    let mut land_dark_toe_knee = LAND_DARK_TOE_KNEE;
    let mut land_dark_toe_gamma = LAND_DARK_TOE_GAMMA;
    let mut land_dark_toe_max_gain = LAND_DARK_TOE_MAX_GAIN;
    let mut multiscatter = true;
    let mut cloud_multiscatter = None;
    let mut beer_powder = false;
    let mut clouds = true;
    let mut fractional_clouds = true;
    let mut cloud_optical_depth_scale = DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE;
    let mut feather_exposed_domain_edges = true;
    let mut granulation = false;
    let mut topdown_stratiform_regularization = false;
    let mut steps = StepQuality::Offline;
    let mut sun_elev_override: Option<f64> = None;
    let mut sun_az_override: Option<f64> = None;
    let mut exposure = DEFAULT_EXPOSURE;
    let mut cache = ingest::default_cache_dir();
    let mut bluemarble: Option<PathBuf> = None;
    let mut bluemarble_month: Option<u32> = None;
    let mut bluemarble_download = true;
    let mut view = ViewMode::Geostationary;
    let mut canvas: Option<(usize, usize)> = None;
    let mut threads: Option<usize> = None;
    let mut store: Option<PathBuf> = None;
    let mut sector: Option<String> = None;
    let mut geocolor = false;
    let mut sandwich = false;
    let mut cloud_layer = false;
    let mut composite_out: Option<PathBuf> = None;
    let mut eye: Option<(f64, f64, f64)> = None;
    let mut look: Option<(f64, f64, f64)> = None;
    let mut fov = 40.0f64;
    let mut camsize = (1280usize, 720usize);
    let mut perspective_layer = false;
    let mut product_perspective = false;
    let mut ground_gain: Option<f64> = None;
    let mut cloud_softclip: Option<f64> = None;
    let mut cloud_highlight_max: Option<f64> = None;
    let mut topdown_cloud_norm: Option<f64> = None;
    let mut bands_out: Option<PathBuf> = None;
    let mut synthetic_green = false;

    for a in args {
        let (k, v) = a
            .split_once('=')
            .ok_or_else(|| format!("expected key=value, got '{a}'"))?;
        match k {
            "input" | "wrfout" | "in" => input = Some(PathBuf::from(v)),
            "out" | "output" | "png" => out = Some(PathBuf::from(v)),
            "backend" | "render-backend" | "render_backend" => backend = parse_backend(v)?,
            "sat" | "satellite" => sat = parse_sat(v)?,
            "timestep" | "ts" => timestep = v.parse().map_err(|_| format!("bad timestep '{v}'"))?,
            "resolution" | "res" => resolution = parse_resolution(v)?,
            "margin" | "zoom-out" | "zoomout" => {
                margin = v.parse().map_err(|_| format!("bad margin '{v}'"))?;
                if !(0.0..=4.0).contains(&margin) {
                    return Err(format!("margin must be 0.0..=4.0 (fraction), got {margin}"));
                }
            }
            "aerosol-optical-depth" | "aerosol_optical_depth" | "aod" => {
                aerosol_optical_depth = v
                    .parse()
                    .map_err(|_| format!("bad aerosol-optical-depth '{v}'"))?;
                if !aerosol_optical_depth.is_finite()
                    || !(0.0..=0.6).contains(&aerosol_optical_depth)
                {
                    return Err(format!(
                        "aerosol-optical-depth must be finite and in 0.0..=0.6, got \
                         {aerosol_optical_depth}"
                    ));
                }
            }
            "rh-aerosol-swelling" | "rh_aerosol_swelling" | "rh-swelling" => {
                rh_aerosol_swelling = parse_bool(v)?
            }
            "atmosphere-correction" | "atmosphere_correction" | "atmo-correction" => {
                atmosphere_correction = parse_bool(v)?
            }
            "terrain-atmosphere" | "terrain_atmosphere" | "terrain-atmo" => {
                terrain_atmosphere = parse_bool(v)?
            }
            "land-sza-normalization" | "land_sza_normalization" | "land-sza" => {
                land_sza_normalization = parse_bool(v)?
            }
            "land-sza-max-gain" | "land_sza_max_gain" => {
                land_sza_max_gain = v
                    .parse()
                    .map_err(|_| format!("bad land-sza-max-gain '{v}'"))?;
                if !land_sza_max_gain.is_finite() || !(1.0..=4.0).contains(&land_sza_max_gain) {
                    return Err(format!(
                        "land-sza-max-gain must be finite and in 1.0..=4.0, got \
                         {land_sza_max_gain}"
                    ));
                }
            }
            "land-dark-toe" | "land_dark_toe" | "dark-land-toe" => land_dark_toe = parse_bool(v)?,
            "land-dark-toe-knee" | "land_dark_toe_knee" => {
                land_dark_toe_knee = v
                    .parse()
                    .map_err(|_| format!("bad land-dark-toe-knee '{v}'"))?;
                if !land_dark_toe_knee.is_finite() || !(1.0e-6..=1.0).contains(&land_dark_toe_knee)
                {
                    return Err(format!(
                        "land-dark-toe-knee must be finite and in 1e-6..=1.0, got \
                         {land_dark_toe_knee}"
                    ));
                }
            }
            "land-dark-toe-gamma" | "land_dark_toe_gamma" => {
                land_dark_toe_gamma = v
                    .parse()
                    .map_err(|_| format!("bad land-dark-toe-gamma '{v}'"))?;
                if !land_dark_toe_gamma.is_finite() || !(0.05..=1.0).contains(&land_dark_toe_gamma)
                {
                    return Err(format!(
                        "land-dark-toe-gamma must be finite and in 0.05..=1.0, got \
                         {land_dark_toe_gamma}"
                    ));
                }
            }
            "land-dark-toe-max-gain" | "land_dark_toe_max_gain" => {
                land_dark_toe_max_gain = v
                    .parse()
                    .map_err(|_| format!("bad land-dark-toe-max-gain '{v}'"))?;
                if !land_dark_toe_max_gain.is_finite()
                    || !(1.0..=4.0).contains(&land_dark_toe_max_gain)
                {
                    return Err(format!(
                        "land-dark-toe-max-gain must be finite and in 1.0..=4.0, got \
                         {land_dark_toe_max_gain}"
                    ));
                }
            }
            "multiscatter" | "ms" => multiscatter = parse_bool(v)?,
            "cloud-multiscatter" | "cloud_multiscatter" | "cloud-ms" => {
                cloud_multiscatter = Some(parse_cloud_multiscatter(v)?)
            }
            "beer-powder" | "beer_powder" | "beerpowder" => beer_powder = parse_bool(v)?,
            "clouds" => clouds = parse_bool(v)?,
            "fractional-clouds" | "fractional_clouds" | "model-cloud-fraction" => {
                fractional_clouds = parse_bool(v)?
            }
            "cloud-optical-depth-scale" | "cloud_optical_depth_scale" | "cloud-od-scale" => {
                cloud_optical_depth_scale = v
                    .parse()
                    .map_err(|_| format!("bad cloud-optical-depth-scale '{v}'"))?;
                if !cloud_optical_depth_scale.is_finite()
                    || !(0.0..=4.0).contains(&cloud_optical_depth_scale)
                {
                    return Err(format!(
                        "cloud-optical-depth-scale must be finite and in 0.0..=4.0, got \
                         {cloud_optical_depth_scale}"
                    ));
                }
            }
            "feather-exposed-domain-edges" | "feather_exposed_domain_edges" => {
                feather_exposed_domain_edges = parse_bool(v)?
            }
            "granulation" | "granulate" | "cloud-granulation" => granulation = parse_bool(v)?,
            "topdown-stratiform-regularization"
            | "topdown_stratiform_regularization"
            | "topdown-cloud-regularization" => topdown_stratiform_regularization = parse_bool(v)?,
            "steps" | "quality" => steps = parse_steps(v)?,
            "sun-elev" | "sun_elev" | "sunelev" => {
                sun_elev_override = Some(v.parse().map_err(|_| format!("bad sun-elev '{v}'"))?)
            }
            "sun-az" | "sun_az" | "sunaz" => {
                sun_az_override = Some(v.parse().map_err(|_| format!("bad sun-az '{v}'"))?)
            }
            "exposure" | "ev" => exposure = v.parse().map_err(|_| format!("bad exposure '{v}'"))?,
            "cache" => cache = PathBuf::from(v),
            "bluemarble" | "bm" => bluemarble = Some(PathBuf::from(v)),
            "bluemarble-month" | "bm-month" | "month" => {
                let m: u32 = v
                    .parse()
                    .map_err(|_| format!("bad bluemarble-month '{v}'"))?;
                if !(1..=12).contains(&m) {
                    return Err(format!("bluemarble-month must be 1..=12, got {m}"));
                }
                bluemarble_month = Some(m);
            }
            "bluemarble-download" | "bm-download" => bluemarble_download = parse_bool(v)?,
            "view" | "mode" => view = parse_view(v)?,
            "canvas" | "figure" | "size" => canvas = Some(parse_canvas(v)?),
            "threads" | "rayon-threads" | "num-threads" => {
                threads = Some(v.parse().map_err(|_| format!("bad threads '{v}'"))?)
            }
            "store" | "sat-store" | "store-root" => store = Some(PathBuf::from(v)),
            "sector" | "run-token" => sector = Some(v.to_string()),
            "geocolor" | "gc" => geocolor = parse_bool(v)?,
            "sandwich" | "sw" => sandwich = parse_bool(v)?,
            "product" => match v.to_ascii_lowercase().replace(['-', '_', ' '], "").as_str() {
                "visible" | "vis" => {}
                "geocolor" => geocolor = true,
                "sandwich" => sandwich = true,
                "cloudlayer" | "layer" => cloud_layer = true,
                "perspective" | "persp" => product_perspective = true,
                other => {
                    return Err(format!(
                        "unknown product '{other}' \
                         (visible|geocolor|sandwich|cloud-layer|perspective)"
                    ));
                }
            },
            "composite-out" | "composite_out" | "compositeout" => {
                composite_out = Some(PathBuf::from(v))
            }
            "eye" => eye = Some(parse_triple(v)?),
            "look" | "lookat" | "look-at" => look = Some(parse_triple(v)?),
            "fov" => fov = v.parse().map_err(|_| format!("bad fov '{v}'"))?,
            "camsize" | "cam-size" | "cam_size" => camsize = parse_canvas(v)?,
            "perspective-layer" | "perspective_layer" | "persplayer" => {
                perspective_layer = parse_bool(v)?
            }
            "ground-gain" | "ground_gain" | "groundgain" => {
                ground_gain = Some(v.parse().map_err(|_| format!("bad ground-gain '{v}'"))?)
            }
            "cloud-softclip" | "cloud_softclip" | "softclip" => {
                cloud_softclip = Some(v.parse().map_err(|_| format!("bad cloud-softclip '{v}'"))?)
            }
            "cloud-highlight-max" | "cloud_highlight_max" | "highlight-max" => {
                cloud_highlight_max = Some(
                    v.parse()
                        .map_err(|_| format!("bad cloud-highlight-max '{v}'"))?,
                )
            }
            "topdown-cloudnorm" | "topdown_cloudnorm" | "cloudnorm" => {
                topdown_cloud_norm = Some(
                    v.parse()
                        .map_err(|_| format!("bad topdown-cloudnorm '{v}'"))?,
                )
            }
            "bands-out" | "bands_out" | "bandsout" => bands_out = Some(PathBuf::from(v)),
            "synthetic-green" | "synthetic_green" | "syngreen" => synthetic_green = parse_bool(v)?,
            other => return Err(format!("unknown key '{other}'")),
        }
    }
    // The perspective camera is all-or-nothing: eye + look come as a pair, and the
    // perspective-only flags demand them.
    if eye.is_some() != look.is_some() {
        return Err("perspective needs BOTH eye=lat,lon,alt_m AND look=lat,lon,alt_m".to_string());
    }
    if (product_perspective || perspective_layer) && eye.is_none() {
        return Err("product=perspective / perspective-layer= require eye= and look=".to_string());
    }
    Ok(Opts {
        input: input.ok_or("missing required input=<path>")?,
        out: out.ok_or("missing required out=<file.png>")?,
        backend,
        sat,
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
        multiscatter,
        cloud_multiscatter,
        beer_powder,
        steps,
        sun_elev_override,
        sun_az_override,
        clouds,
        fractional_clouds,
        cloud_optical_depth_scale,
        feather_exposed_domain_edges,
        granulation,
        topdown_stratiform_regularization,
        exposure,
        cache,
        bluemarble,
        bluemarble_month,
        bluemarble_download,
        view,
        canvas,
        threads,
        store,
        sector,
        geocolor,
        sandwich,
        cloud_layer,
        composite_out,
        eye,
        look,
        fov,
        camsize,
        perspective_layer,
        ground_gain,
        cloud_softclip,
        cloud_highlight_max,
        topdown_cloud_norm,
        bands_out,
        synthetic_green,
    })
}

/// Parse a `lat,lon,alt_m` triple (e.g. `eye=46.2,-98.9,150000`).
fn parse_triple(v: &str) -> Result<(f64, f64, f64), String> {
    let parts: Vec<&str> = v.split(',').map(str::trim).collect();
    if parts.len() != 3 {
        return Err(format!("expected lat,lon,alt_m, got '{v}'"));
    }
    let f = |s: &str| {
        s.parse::<f64>()
            .map_err(|_| format!("bad number '{s}' in '{v}'"))
    };
    Ok((f(parts[0])?, f(parts[1])?, f(parts[2])?))
}

fn parse_view(v: &str) -> Result<ViewMode, String> {
    match v.to_ascii_lowercase().replace(['-', '_', ' '], "").as_str() {
        "geo" | "geostationary" | "fromspace" | "space" => Ok(ViewMode::Geostationary),
        "topdown" | "top" | "map" | "topdownmap" | "nadir" => Ok(ViewMode::TopDownMap),
        _ => Err(format!("unknown view '{v}' (geo|topdown)")),
    }
}

fn parse_backend(v: &str) -> Result<RenderBackend, String> {
    match v.to_ascii_lowercase().replace(['_', ' '], "-").as_str() {
        "cpu" => Ok(RenderBackend::Cpu),
        "gpu" | "gpu-preview" | "preview" => Ok(RenderBackend::GpuPreview),
        _ => Err(format!("unknown backend '{v}' (expected cpu|gpu-preview)")),
    }
}

/// Parse a `WxH` canvas size (e.g. `1100x850`).
fn parse_canvas(v: &str) -> Result<(usize, usize), String> {
    let (w, h) = v
        .split_once(['x', 'X', '*', ','])
        .ok_or_else(|| format!("bad canvas '{v}' (expected WxH, e.g. 1100x850)"))?;
    let w: usize = w
        .trim()
        .parse()
        .map_err(|_| format!("bad canvas width '{v}'"))?;
    let h: usize = h
        .trim()
        .parse()
        .map_err(|_| format!("bad canvas height '{v}'"))?;
    if w == 0 || h == 0 {
        return Err(format!("canvas dims must be > 0, got {v}"));
    }
    Ok((w, h))
}

fn parse_sat(v: &str) -> Result<SatellitePreset, String> {
    match v.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
        "goeseast" | "goese" | "east" => Ok(SatellitePreset::GoesEast),
        "goeswest" | "goesw" | "west" => Ok(SatellitePreset::GoesWest),
        "himawari" | "ahi" => Ok(SatellitePreset::Himawari),
        _ => Err(format!(
            "unknown satellite '{v}' (goes-east|goes-west|himawari)"
        )),
    }
}

fn parse_resolution(v: &str) -> Result<ResolutionMode, String> {
    match v.to_ascii_lowercase().replace(['-', '_', ' '], "").as_str() {
        "native" => Ok(ResolutionMode::Native),
        "abi1km" | "1km" => Ok(ResolutionMode::Abi1km),
        "abi2km" | "2km" => Ok(ResolutionMode::Abi2km),
        _ => Err(format!("unknown resolution '{v}' (native|abi1km|abi2km)")),
    }
}

fn parse_steps(v: &str) -> Result<StepQuality, String> {
    match v.to_ascii_lowercase().as_str() {
        "offline" | "full" | "384" => Ok(StepQuality::Offline),
        "interactive" | "preview" | "192" => Ok(StepQuality::Interactive),
        _ => Err(format!("unknown steps '{v}' (offline|interactive)")),
    }
}

fn parse_bool(v: &str) -> Result<bool, String> {
    match v.to_ascii_lowercase().as_str() {
        "on" | "true" | "1" | "yes" => Ok(true),
        "off" | "false" | "0" | "no" => Ok(false),
        _ => Err(format!("expected on/off, got '{v}'")),
    }
}

fn parse_cloud_multiscatter(v: &str) -> Result<CloudMultiscatterMode, String> {
    match v.to_ascii_lowercase().replace('_', "-").as_str() {
        "legacy" | "legacy-octaves" | "octaves" => Ok(CloudMultiscatterMode::LegacyOctaves),
        "single" | "single-scatter" | "off" => Ok(CloudMultiscatterMode::SingleScatter),
        "delta-flux-v1" | "delta-flux" | "stage2" => Ok(CloudMultiscatterMode::DeltaFluxV1),
        "delta-flux-v2b" | "delta-flux-v2" | "stage2-p1" => Ok(CloudMultiscatterMode::DeltaFluxV2),
        _ => Err(format!(
            "expected legacy-octaves|single-scatter|delta-flux-v1|delta-flux-v2b, got '{v}'"
        )),
    }
}

fn resolved_cloud_multiscatter(opts: &Opts) -> CloudMultiscatterMode {
    opts.cloud_multiscatter.unwrap_or(if opts.multiscatter {
        CloudMultiscatterMode::LegacyOctaves
    } else {
        CloudMultiscatterMode::SingleScatter
    })
}

fn print_usage() {
    eprintln!(
        "render_frame — headless full-frame composited render to PNG (CPU or explicit GPU preview).\n\n\
         USAGE:\n  render_frame input=<wrfout|run.json> out=<file.png> [key=value ...]\n\n\
         KEYS:\n\
         \x20 input=<path>       wrfout (ingest-if-needed) or a cached run.json  [required]\n\
         \x20 out=<file.png>     output PNG (RGB8, row 0 = north)                [required]\n\
         \x20 backend=<mode>     cpu | gpu-preview (default cpu; preview reports substitutions)\n\
         \x20 sat=<preset>       goes-east | goes-west | himawari   (default goes-east)\n\
         \x20 timestep=<n>       time index (default 0)\n\
         \x20 resolution=<mode>  native | abi1km | abi2km           (default native)\n\
         \x20                    native = one pixel per source-grid cell; ABI 1/2 km may\n\
         \x20                    upsample coarse or downsample fine model grids\n\
         \x20 margin=<frac>      zoom-out margin fraction on each side (default 0.0 edge-to-edge)\n\
         \x20 aerosol-optical-depth=<f>  aerosol AOD, 0.0..=0.6 (default {DEFAULT_AOD})\n\
         \x20 rh-aerosol-swelling=<b>    on|off 1.5x aerosol swelling (default off)\n\
         \x20 atmosphere-correction=<b>  on|off daytime aerial-veil correction (default on)\n\
         \x20 terrain-atmosphere=<b>     on|off terrain-height columns (default on)\n\
         \x20 land-sza-normalization=<b> on|off bounded land SZA normalization (default on)\n\
         \x20 land-sza-max-gain=<f>      SZA normalization bound (default {LAND_SZA_MAX_GAIN})\n\
         \x20 land-dark-toe=<b>          on|off bounded dark-land toe (default on)\n\
         \x20 land-dark-toe-knee=<f>     toe identity knee (default {LAND_DARK_TOE_KNEE})\n\
         \x20 land-dark-toe-gamma=<f>    toe exponent (default {LAND_DARK_TOE_GAMMA})\n\
         \x20 land-dark-toe-max-gain=<f> toe gain bound (default {LAND_DARK_TOE_MAX_GAIN})\n\
         \x20 multiscatter=<b>   on | off  legacy compatibility toggle (default on)\n\
         \x20 cloud-multiscatter=<mode> legacy-octaves|single-scatter|delta-flux-v1|delta-flux-v2b (opt-in)\n\
         \x20 beer-powder=<b>    on | off  direct-sun shaping       (default off)\n\
         \x20 clouds=<b>         on | off  (off = surface only)     (default on)\n\
         \x20 fractional-clouds=<b> on | off use model cloud fraction (default on; off = legacy)\n\
         \x20 cloud-optical-depth-scale=<f>  cloud OD scale, 0.0..=4.0 (default {DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE}; 1.0 = unscaled)\n\
         \x20 feather-exposed-domain-edges=<b> on|off fade finished clouds at visible WRF boundaries (default on)\n\
         \x20 granulation=<b>    on | off  sub-grid cloud detail    (default off)\n\
         \x20 topdown-stratiform-regularization=<b> on|off low-deck source reconstruction (default off)\n\
         \x20 steps=<quality>    offline | interactive              (default offline)\n\
         \x20 sun-elev=<deg>     OPTIONAL sun elevation override (else true solar)\n\
         \x20 sun-az=<deg>       OPTIONAL sun azimuth override, deg from north\n\
         \x20 exposure=<f>       display gain before the ABI stretch (default {DEFAULT_EXPOSURE})\n\
         \x20 cache=<dir>        brick cache root + seasonal Blue Marble cache\n\
         \x20 bluemarble=<path>  single-file Blue Marble override (default: seasonal pack)\n\
         \x20 bluemarble-month=<MM>  force month 1..=12 (default: day-of-year blend)\n\
         \x20 bluemarble-download=<b>  on|off lazy month fetch (default on)\n\
         \x20 view=<mode>        geo | topdown  (default geo)\n\
         \x20 canvas=<WxH>       letterbox into a fixed figure size, black pad (e.g. 1100x850)\n\
         \x20 threads=<N>        rayon thread cap (else honor RAYON_NUM_THREADS)\n\
         \x20 store=<dir>        ALSO write the visible frame to this sat-store root (geo only)\n\
         \x20 sector=<token>     store run sector token (default: the input's run id)\n\
         \x20 geocolor=<b>       on|off  GeoColor day/night blend (true-color day + IR night)\n\
         \x20 sandwich=<b>       on|off  Sandwich (true-color base + color IR on cold tops)\n\
         \x20 product=<p>        visible|geocolor|sandwich|cloud-layer|perspective\n\
         \x20 composite-out=<p>  cloud-layer only: synthetic composite proof PNG\n\
         \x20 eye=lat,lon,alt    perspective camera eye (with look= switches to perspective)\n\
         \x20 look=lat,lon,alt   perspective look-at target\n\
         \x20 fov=<deg>          perspective horizontal FOV (default 40)\n\
         \x20 camsize=<WxH>      perspective image dims (default 1280x720)\n\
         \x20 perspective-layer=<b>  on|off cloud-field-only RGBA through the camera\n\
         \x20 ground-gain=<f>    override shipped 1.0 GROUND LIFT (neutral)\n\
         \x20 cloud-softclip=<f> override shipped 0.65 highlight knee (1.0 = disable)\n\
         \x20 cloud-highlight-max=<f> override shipped 1.25 ceiling mapped to white\n\
         \x20 topdown-cloudnorm=<f>  override the top-down cloud normalization (1.0 = none)\n\
         \x20 synthetic-green=<b> on|off ABI synthetic-green display mode (G'=0.45R+0.45B+0.10G)\n\
         \x20 bands-out=<path.bin>  QA: also dump raw pre-tonemap reflectance (f32le RGB)\n\n\
         SUMMARY: cloud_frac/peak[_sun]_reflectance are geostationary-only physical\n\
         diagnostics and print n/a for top-down; cloud_lum_* works in both views.\n"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts_with(extra: &[&str]) -> Opts {
        let mut args = vec!["input=input".to_string(), "out=out.png".to_string()];
        args.extend(extra.iter().map(|v| (*v).to_string()));
        parse_opts(&args).expect("parse options")
    }

    #[test]
    fn cloud_controls_have_intentional_defaults() {
        let opts = opts_with(&[]);
        assert_eq!(opts.backend, RenderBackend::Cpu);
        assert!(!opts.beer_powder);
        assert!(!opts.granulation);
        assert!(opts.multiscatter);
        assert_eq!(opts.cloud_multiscatter, None);
        assert_eq!(
            resolved_cloud_multiscatter(&opts),
            CloudMultiscatterMode::LegacyOctaves
        );
        assert!(!opts.topdown_stratiform_regularization);
        assert!(opts.fractional_clouds);
        assert!(opts.feather_exposed_domain_edges);
        assert_eq!(
            opts.cloud_optical_depth_scale,
            DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE
        );
        assert!(opts.land_sza_normalization);
        assert_eq!(opts.land_sza_max_gain, LAND_SZA_MAX_GAIN);
        assert!(opts.land_dark_toe);
        assert_eq!(opts.land_dark_toe_knee, LAND_DARK_TOE_KNEE);
        assert_eq!(opts.land_dark_toe_gamma, LAND_DARK_TOE_GAMMA);
        assert_eq!(opts.land_dark_toe_max_gain, LAND_DARK_TOE_MAX_GAIN);
        assert!(opts.ground_gain.is_none());
        assert!(opts.cloud_softclip.is_none());
        assert!(opts.cloud_highlight_max.is_none());
        let params = render_params(&opts);
        assert_eq!(params.backend, RenderBackend::Cpu);
        assert!(!params.beer_powder);
        assert_eq!(params.granulation, Some(false));
        assert!(!params.topdown_stratiform_regularization);
        assert!(params.fractional_clouds);
        assert!(params.feather_exposed_domain_edges);
        assert_eq!(
            params.cloud_optical_depth_scale,
            DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE
        );
        assert_eq!(params.land_appearance, LandAppearanceConfig::shipped());
        assert!(params.ground_gain.is_none());
        assert!(params.cloud_softclip.is_none());
        assert!(params.cloud_highlight_max.is_none());
    }

    #[test]
    fn backend_parser_is_explicit_and_reaches_render_params() {
        let opts = opts_with(&["backend=gpu-preview"]);
        assert_eq!(opts.backend, RenderBackend::GpuPreview);
        assert_eq!(render_params(&opts).backend, RenderBackend::GpuPreview);
        assert_eq!(
            opts_with(&["backend=gpu"]).backend,
            RenderBackend::GpuPreview
        );

        let args = vec![
            "input=input".to_string(),
            "out=out.png".to_string(),
            "backend=magic".to_string(),
        ];
        let err = parse_opts(&args).err().expect("invalid backend rejected");
        assert!(err.contains("cpu|gpu-preview"));
    }

    #[test]
    fn explicit_cloud_transport_tokens_reach_the_rust_api() {
        for (token, expected) in [
            ("legacy-octaves", CloudMultiscatterMode::LegacyOctaves),
            ("single-scatter", CloudMultiscatterMode::SingleScatter),
            ("delta-flux-v1", CloudMultiscatterMode::DeltaFluxV1),
            ("delta-flux-v2b", CloudMultiscatterMode::DeltaFluxV2),
        ] {
            let arg = format!("cloud-multiscatter={token}");
            let opts = opts_with(&[&arg]);
            assert_eq!(resolved_cloud_multiscatter(&opts), expected);
            assert_eq!(render_params(&opts).cloud_multiscatter, Some(expected));
        }

        let legacy_off = opts_with(&["multiscatter=off"]);
        assert_eq!(legacy_off.cloud_multiscatter, None);
        assert_eq!(
            resolved_cloud_multiscatter(&legacy_off),
            CloudMultiscatterMode::SingleScatter,
            "omitting the new override preserves the old boolean contract"
        );
    }

    #[test]
    fn cloud_controls_parse_and_explicit_edge_off_reach_render_params() {
        let opts = opts_with(&[
            "beer-powder=on",
            "granulation=yes",
            "fractional-clouds=off",
            "feather-exposed-domain-edges=off",
            "topdown-stratiform-regularization=on",
        ]);
        assert!(opts.beer_powder);
        assert!(opts.granulation);
        assert!(!opts.fractional_clouds);
        assert!(!opts.feather_exposed_domain_edges);
        assert!(opts.topdown_stratiform_regularization);
        let params = render_params(&opts);
        assert!(params.beer_powder);
        assert_eq!(params.granulation, Some(true));
        assert!(!params.fractional_clouds);
        assert!(!params.feather_exposed_domain_edges);
        assert!(params.topdown_stratiform_regularization);
    }

    #[test]
    fn display_controls_parse_and_reach_render_params() {
        let opts = opts_with(&[
            "ground-gain=1.6",
            "cloud-softclip=0.65",
            "cloud-highlight-max=1.25",
            "land-sza-normalization=on",
            "land-sza-max-gain=1.7",
            "land-dark-toe=yes",
            "land-dark-toe-knee=0.07",
            "land-dark-toe-gamma=0.6",
            "land-dark-toe-max-gain=1.4",
        ]);
        assert_eq!(opts.ground_gain, Some(1.6));
        assert_eq!(opts.cloud_softclip, Some(0.65));
        assert_eq!(opts.cloud_highlight_max, Some(1.25));
        assert!(opts.land_sza_normalization);
        assert_eq!(opts.land_sza_max_gain, 1.7);
        assert!(opts.land_dark_toe);
        assert_eq!(opts.land_dark_toe_knee, 0.07);
        assert_eq!(opts.land_dark_toe_gamma, 0.6);
        assert_eq!(opts.land_dark_toe_max_gain, 1.4);

        let params = render_params(&opts);
        assert_eq!(params.ground_gain, Some(1.6));
        assert_eq!(params.cloud_softclip, Some(0.65));
        assert_eq!(params.cloud_highlight_max, Some(1.25));
        assert_eq!(
            params.land_appearance,
            LandAppearanceConfig {
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
    fn land_controls_can_explicitly_select_legacy_identity() {
        let opts = opts_with(&["land-sza-normalization=off", "land-dark-toe=off"]);
        assert!(!opts.land_sza_normalization);
        assert!(!opts.land_dark_toe);
        assert_eq!(
            render_params(&opts).land_appearance,
            LandAppearanceConfig::identity()
        );
    }

    #[test]
    fn invalid_land_appearance_values_are_rejected() {
        for arg in [
            "land-sza-max-gain=0.9",
            "land-dark-toe-knee=0",
            "land-dark-toe-gamma=1.1",
            "land-dark-toe-max-gain=nan",
        ] {
            let args = vec![
                "input=input".to_string(),
                "out=out.png".to_string(),
                arg.to_string(),
            ];
            assert!(parse_opts(&args).is_err(), "must reject {arg}");
        }
    }

    #[test]
    fn absent_topdown_cloud_stats_are_not_serialized_as_real_zeroes() {
        assert_eq!(
            cloud_stat_summary_fields(None),
            ("n/a".to_string(), "n/a".to_string(), "n/a".to_string())
        );
    }

    #[test]
    fn present_geostationary_cloud_stats_keep_numeric_summary_precision() {
        let stats = CloudFrameStats {
            sampled: 8,
            cloudy: 3,
            all_finite: true,
            max_inscatter: 12.0,
            max_reflectance: 0.67894,
            max_sun_reflectance: 0.12346,
        };
        assert_eq!(
            cloud_stat_summary_fields(Some(&stats)),
            (
                "0.375".to_string(),
                "0.6789".to_string(),
                "0.1235".to_string()
            )
        );
    }
}
