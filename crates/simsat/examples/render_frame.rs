//! `render_frame` — headless "render one full composited visible frame to PNG".
//!
//! A CPU-only render harness so a render can be visually QA'd without a GPU or the
//! studio GUI. It is now a THIN wrapper over [`simsat::api::render`] (the one shared
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
//!   sat=<preset>       goes-east | goes-west | himawari   (default goes-east)
//!   timestep=<n>       Time index (default 0).
//!   resolution=<mode>  native | abi1km | abi2km           (default native)
//!   margin=<frac>      zoom-out margin, a FRACTION of the domain added on each side
//!                      (default 0.0 = edge-to-edge; 0.3 = the domain in a 30% earth margin).
//!   multiscatter=<b>   on | off  — M5 Wrenninge octaves   (default on).
//!   clouds=<b>         on | off  — off = surface only (QA terrain/glint)  (default on).
//!   steps=<quality>    offline | interactive              (default offline).
//!   sun-elev=<deg>     OPTIONAL sun-elevation override (else true solar geometry).
//!   sun-az=<deg>       OPTIONAL sun-azimuth override (deg from north).
//!   exposure=<f>       Display gain before the ABI stretch (default DEFAULT_EXPOSURE).
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
//!   ground-gain=<f>    OVERRIDE the baked GROUND LIFT (render::GROUND_DAY_LIFT); 1.0 = neutral.
//!   cloud-softclip=<f> OVERRIDE the baked highlight soft-clip knee (render::CLOUD_SOFTCLIP_KNEE);
//!                      1.0 = disable the shoulder (hard clamp).
//!   topdown-cloudnorm=<f>  OVERRIDE the baked TOP-DOWN CLOUD NORMALIZATION
//!                      (topdown::TOPDOWN_CLOUD_NORM); 1.0 = no normalization.
//!
//! On completion it prints a one-line `SUMMARY ...` (dims, on-earth fraction, centre sun
//! elevation, exposure, multiscatter, wall time, peak/median display luminance, the
//! strided physical peak/median cloud reflectance, and `cloud_lum_p90_p10` — the WS2
//! bright-cloud contrast metric: P90-P10 display luminance over strongly-cloudy output
//! pixels, computed from the frame itself so it works for BOTH geo and top-down views)
//! to stdout.
//!
//! NOTE: the old `supersample=` QA flag was REMOVED with the api refactor — it was a
//! documented, tested-and-REJECTED anti-alias experiment (the cloud march is already
//! trilinear, so a box average changed nothing at N^2 cost). The single shared
//! `api::render` path has no supersample raster; nothing in the shipping pipeline used it.

use std::path::{Path, PathBuf};
use std::time::Instant;

use image::RgbImage;
use simsat::api::{self, BlueMarble, FrameData, Product, RenderParams, SunOverride};
use simsat::camera::{ResolutionMode, SatellitePreset, ViewMode};
use simsat::clouds::StepQuality;
use simsat::gpu::RenderedFrame;
use simsat::ingest;
use simsat::render::DEFAULT_EXPOSURE;
use simsat::store_out::{self, VisibleFrame};
use simsat::topdown;

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
    sat: SatellitePreset,
    timestep: usize,
    resolution: ResolutionMode,
    /// Zoom-out / domain margin as a FRACTION added on each side (0.0 = edge-to-edge).
    margin: f64,
    multiscatter: bool,
    steps: StepQuality,
    sun_elev_override: Option<f64>,
    sun_az_override: Option<f64>,
    clouds: bool,
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
    /// Appearance-pass tuning overrides (None = the baked engine defaults). Future tuning
    /// knobs; the shipped look already comes from the baked constants.
    ground_gain: Option<f64>,
    cloud_softclip: Option<f64>,
    topdown_cloud_norm: Option<f64>,
}

fn run(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return Ok(());
    }
    let opts = parse_opts(args)?;
    eprintln!(
        "render_frame: input={} product={} view={} sat={} ts={} res={} margin={:.2} \
         multiscatter={} clouds={} steps={} sun-elev={} exposure={:.3} canvas={} threads={}",
        opts.input.display(),
        product_label(&opts),
        opts.view.slug(),
        opts.sat.slug(),
        opts.timestep,
        opts.resolution.label(),
        opts.margin,
        opts.multiscatter,
        opts.clouds,
        if opts.steps == StepQuality::Offline {
            "offline"
        } else {
            "interactive"
        },
        opts.sun_elev_override
            .map(|e| format!("{e:.1}"))
            .unwrap_or_else(|| "actual".to_string()),
        opts.exposure,
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

    // ── the one shared render assembly ──
    let params = render_params(&opts);
    // Sandwich = the visible base + cold-top IR overlay; GeoColor = the day/night blend; else
    // the plain refined true-color visible frame. All three are baked RGB composites
    // (FrameData::Visible). Sandwich takes precedence over GeoColor if both are set.
    let product = if opts.sandwich {
        Product::Sandwich
    } else if opts.geocolor {
        Product::GeoColor
    } else {
        Product::VisibleRgb
    };
    let t0 = Instant::now();
    let result = api::render(&params, product)?;
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

    // ── optional canvas letterbox to a fixed figure size (black pad) ──
    let (final_nx, final_ny, final_rgb) = match opts.canvas {
        Some((cw, ch)) => (cw, ch, topdown::letterbox_rgb(rgb, rnx, rny, cw, ch)),
        None => (rnx, rny, rgb.clone()),
    };
    write_rgb8_png(&opts.out, final_nx, final_ny, &final_rgb)?;

    // ── optional sat-store write (geostationary only — the store carries the scan mesh) ──
    if opts.store.is_some() {
        if opts.view == ViewMode::TopDownMap {
            eprintln!("render_frame: store write skipped (store= is geostationary only).");
        } else {
            write_store(&opts, &result, rgba)?;
        }
    }

    // ── stats for the manifest ──
    let (on_earth, peak_lum, median_lum) = display_luma_stats(rgba);
    let on_earth_frac = on_earth as f64 / (rnx * rny).max(1) as f64;
    let (cloud_lum_p90_p10, cloud_lum_frac) = cloud_contrast_stat(rgba);
    let (cloud_frac, peak_refl, peak_sun_refl) = match &result.cloud_stats {
        Some(s) => (s.cloud_fraction(), s.max_reflectance, s.max_sun_reflectance),
        None => (0.0, 0.0, 0.0),
    };

    eprintln!("render_frame: wrote {}", opts.out.display());
    println!(
        "SUMMARY file={} view={} dims={}x{} canvas={} render_dims={}x{} res={}{} sat={} \
         sun_elev={:.1} exposure={:.3} multiscatter={} steps={} on_earth_frac={:.3} \
         peak_lum={:.3} median_lum={:.3} cloud_frac={:.3} peak_reflectance={:.4} \
         peak_sun_reflectance={:.4} cloud_lum_p90_p10={:.4} cloud_lum_frac={:.3} wall_s={:.3}",
        opts.out.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
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
        opts.multiscatter,
        if opts.steps == StepQuality::Offline {
            "offline"
        } else {
            "interactive"
        },
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

/// Display luminance threshold above which an output pixel is counted as
/// STRONGLY CLOUDY for the [`cloud_contrast_stat`] metric. `0.70` display sits at
/// `rho' ~ 0.49` (just under the soft-clip knee at exposure 1.6), so the population is
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

/// A short product label for the log line (sandwich > geocolor > visible).
fn product_label(opts: &Opts) -> &'static str {
    if opts.sandwich {
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
        satellite: opts.sat,
        timestep: opts.timestep,
        view: opts.view,
        resolution: opts.resolution,
        margin_frac: opts.margin as f32,
        exposure: opts.exposure,
        multiscatter: opts.multiscatter,
        steps: opts.steps,
        clouds: opts.clouds,
        sun_override,
        cache: opts.cache.clone(),
        bluemarble,
        ir_enhancement: None,
        derived_colormap: false,
        raster_override: None,
        ground_gain: opts.ground_gain,
        cloud_softclip: opts.cloud_softclip,
        topdown_cloud_norm: opts.topdown_cloud_norm,
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
    let mut sat = SatellitePreset::GoesEast;
    let mut timestep = 0usize;
    let mut resolution = ResolutionMode::Native;
    let mut margin = 0.0f64;
    let mut multiscatter = true;
    let mut clouds = true;
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
    let mut ground_gain: Option<f64> = None;
    let mut cloud_softclip: Option<f64> = None;
    let mut topdown_cloud_norm: Option<f64> = None;

    for a in args {
        let (k, v) = a
            .split_once('=')
            .ok_or_else(|| format!("expected key=value, got '{a}'"))?;
        match k {
            "input" | "wrfout" | "in" => input = Some(PathBuf::from(v)),
            "out" | "output" | "png" => out = Some(PathBuf::from(v)),
            "sat" | "satellite" => sat = parse_sat(v)?,
            "timestep" | "ts" => timestep = v.parse().map_err(|_| format!("bad timestep '{v}'"))?,
            "resolution" | "res" => resolution = parse_resolution(v)?,
            "margin" | "zoom-out" | "zoomout" => {
                margin = v.parse().map_err(|_| format!("bad margin '{v}'"))?;
                if !(0.0..=4.0).contains(&margin) {
                    return Err(format!("margin must be 0.0..=4.0 (fraction), got {margin}"));
                }
            }
            "multiscatter" | "ms" => multiscatter = parse_bool(v)?,
            "clouds" => clouds = parse_bool(v)?,
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
            "ground-gain" | "ground_gain" | "groundgain" => {
                ground_gain = Some(v.parse().map_err(|_| format!("bad ground-gain '{v}'"))?)
            }
            "cloud-softclip" | "cloud_softclip" | "softclip" => {
                cloud_softclip = Some(v.parse().map_err(|_| format!("bad cloud-softclip '{v}'"))?)
            }
            "topdown-cloudnorm" | "topdown_cloudnorm" | "cloudnorm" => {
                topdown_cloud_norm = Some(
                    v.parse()
                        .map_err(|_| format!("bad topdown-cloudnorm '{v}'"))?,
                )
            }
            other => return Err(format!("unknown key '{other}'")),
        }
    }
    Ok(Opts {
        input: input.ok_or("missing required input=<path>")?,
        out: out.ok_or("missing required out=<file.png>")?,
        sat,
        timestep,
        resolution,
        margin,
        multiscatter,
        steps,
        sun_elev_override,
        sun_az_override,
        clouds,
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
        ground_gain,
        cloud_softclip,
        topdown_cloud_norm,
    })
}

fn parse_view(v: &str) -> Result<ViewMode, String> {
    match v.to_ascii_lowercase().replace(['-', '_', ' '], "").as_str() {
        "geo" | "geostationary" | "fromspace" | "space" => Ok(ViewMode::Geostationary),
        "topdown" | "top" | "map" | "topdownmap" | "nadir" => Ok(ViewMode::TopDownMap),
        _ => Err(format!("unknown view '{v}' (geo|topdown)")),
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

fn print_usage() {
    eprintln!(
        "render_frame — headless full-frame composited render to PNG (CPU, no GPU).\n\n\
         USAGE:\n  render_frame input=<wrfout|run.json> out=<file.png> [key=value ...]\n\n\
         KEYS:\n\
         \x20 input=<path>       wrfout (ingest-if-needed) or a cached run.json  [required]\n\
         \x20 out=<file.png>     output PNG (RGB8, row 0 = north)                [required]\n\
         \x20 sat=<preset>       goes-east | goes-west | himawari   (default goes-east)\n\
         \x20 timestep=<n>       time index (default 0)\n\
         \x20 resolution=<mode>  native | abi1km | abi2km           (default native)\n\
         \x20 margin=<frac>      zoom-out margin fraction on each side (default 0.0 edge-to-edge)\n\
         \x20 multiscatter=<b>   on | off  (M5 octaves)             (default on)\n\
         \x20 clouds=<b>         on | off  (off = surface only)     (default on)\n\
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
         \x20 ground-gain=<f>    override the baked GROUND LIFT (1.0 = neutral)\n\
         \x20 cloud-softclip=<f> override the highlight soft-clip knee (1.0 = disable)\n\
         \x20 topdown-cloudnorm=<f>  override the top-down cloud normalization (1.0 = none)\n"
    );
}
