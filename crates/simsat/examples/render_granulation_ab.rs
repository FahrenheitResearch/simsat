//! `render_granulation_ab` — headless A/B QA harness for the sub-grid cloud
//! GRANULATION (edge-erosion detail noise; the clouds.rs granulation section).
//!
//! Renders the SAME visible frame twice through [`simsat::api::render`] — granulation
//! forced OFF then forced ON — writes both PNGs (plus optional zoomed crops of a named
//! pixel rect), prints a machine-readable `GRANSUMMARY` diff line (the dx-derived
//! amplitude, changed-pixel fraction, mean/max channel delta, wall times), and by
//! default ALSO proves the raw-Kelvin contract: the band-13 IR BT plane is rendered
//! with the flag forced on AND off and verified BYTE-IDENTICAL (`GRANIR` line —
//! granulation must never touch a quantitative thermal product).
//!
//! This is a NEW QA harness so `render_frame` / `render_ir` (owned by other
//! workstreams) stay untouched; it reuses the same `api::render` path, so the ON frame
//! is byte-identical to what the studio / binding display product shows.
//!
//! USAGE (key=value args, any order):
//!
//!   render_granulation_ab input=<path> out-prefix=<prefix> [key=value ...]
//!
//!   input=<path>        REQUIRED. A wrfout file (ingested if not cached) OR a run.json.
//!   out-prefix=<p>      REQUIRED. Writes <p>_off.png and <p>_on.png (+ _crop variants).
//!   sat=<preset>        goes-east | goes-west | himawari   (default goes-east)
//!   timestep=<n>        Time index (default 0).
//!   view=<mode>         geo | topdown  (default topdown — the WRF-Runner product).
//!   resolution=<mode>   native | abi1km | abi2km           (default native, geo only)
//!   margin=<frac>       Zoom-out margin fraction            (default 0.0).
//!   steps=<quality>     offline | interactive               (default offline).
//!   sun-elev=<deg>      OPTIONAL sun-elevation override (else true solar geometry).
//!   sun-az=<deg>        OPTIONAL sun-azimuth override.
//!   exposure=<f>        Display gain (default DEFAULT_EXPOSURE).
//!   cache=<dir>         Brick cache root + seasonal Blue Marble cache.
//!   bluemarble=<path>   OPTIONAL single-file Blue Marble override.
//!   bluemarble-month=<MM>    Force month 1..=12 (default day-of-year blend).
//!   bluemarble-download=<b>  on|off lazy month fetch (default on).
//!   threads=<N>         OPTIONAL rayon thread cap (else RAYON_NUM_THREADS).
//!   crop=<X,Y,W,H>      OPTIONAL pixel rect; writes zoomed crops of both frames.
//!   zoom=<N>            Crop nearest-neighbour magnification 1..=8 (default 4).
//!   ir=<b>              on|off — the IR byte-identity proof (default on).
//!   coherence-map=<p>   OPTIONAL (run.json input only): write the round-2
//!                       deck-coherence gate as a grayscale PNG (white = open /
//!                       granulate, black = closed deck; north-up like the top-down
//!                       frame) + print a GRANCOH stats line — the tuning diagnostic
//!                       for GRAN_COHERENCE_* / GRAN_PROTECT_*.

use std::path::{Path, PathBuf};
use std::time::Instant;

use image::{GrayImage, RgbImage};
use simsat::api::{self, BlueMarble, FrameData, Product, RenderParams, SunOverride};
use simsat::bricks::{self, RunManifest};
use simsat::camera::{ResolutionMode, SatellitePreset, ViewMode};
use simsat::clouds::{DecodedVolume, GranCoherence, StepQuality, granulation_amplitude};
use simsat::ingest;
use simsat::render::DEFAULT_EXPOSURE;
use simsat::topdown;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = run(&args) {
        eprintln!("render_granulation_ab: {e}");
        eprintln!("run with no args (or `--help`) for usage.");
        std::process::exit(1);
    }
}

struct Opts {
    input: PathBuf,
    out_prefix: String,
    sat: SatellitePreset,
    timestep: usize,
    view: ViewMode,
    resolution: ResolutionMode,
    margin: f64,
    steps: StepQuality,
    sun_elev_override: Option<f64>,
    sun_az_override: Option<f64>,
    exposure: f64,
    cache: PathBuf,
    bluemarble: Option<PathBuf>,
    bluemarble_month: Option<u32>,
    bluemarble_download: bool,
    threads: Option<usize>,
    crop: Option<(usize, usize, usize, usize)>,
    zoom: usize,
    ir_check: bool,
    coherence_map: Option<PathBuf>,
}

fn run(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return Ok(());
    }
    let opts = parse_opts(args)?;
    let threads = topdown::effective_thread_count(
        opts.threads,
        std::env::var("RAYON_NUM_THREADS").ok().as_deref(),
    );
    topdown::configure_global_rayon(threads);

    // The round-2 deck-coherence tuning diagnostic (independent of the A/B pair).
    if let Some(path) = &opts.coherence_map {
        write_coherence_map(&opts.input, opts.timestep, path)?;
    }

    let params_at = |granulation: bool| -> RenderParams {
        let mut p = RenderParams::new(opts.input.clone());
        p.satellite = opts.sat;
        p.timestep = opts.timestep;
        p.view = opts.view;
        p.resolution = opts.resolution;
        p.margin_frac = opts.margin as f32;
        p.exposure = opts.exposure;
        p.steps = opts.steps;
        p.granulation = Some(granulation);
        p.cache = opts.cache.clone();
        p.bluemarble = match &opts.bluemarble {
            Some(path) => BlueMarble::SingleFile(path.clone()),
            None => BlueMarble::Seasonal {
                month_override: opts.bluemarble_month,
                download: opts.bluemarble_download,
            },
        };
        if opts.sun_elev_override.is_some() || opts.sun_az_override.is_some() {
            p.sun_override = Some(SunOverride {
                elev_deg: opts.sun_elev_override,
                az_deg: opts.sun_az_override,
            });
        }
        p
    };

    // The A/B pair: granulation OFF then ON, same params otherwise.
    let t0 = Instant::now();
    let off = api::render(&params_at(false), Product::VisibleRgb)?;
    let wall_off = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let on = api::render(&params_at(true), Product::VisibleRgb)?;
    let wall_on = t1.elapsed().as_secs_f64();
    if (off.nx, off.ny) != (on.nx, on.ny) {
        return Err(format!(
            "A/B raster mismatch: {}x{} vs {}x{}",
            off.nx, off.ny, on.nx, on.ny
        ));
    }
    let (nx, ny) = (on.nx, on.ny);
    let rgb_of = |r: &api::RenderResult| -> Result<Vec<u8>, String> {
        match &r.data {
            FrameData::Visible { rgb, .. } => Ok(rgb.clone()),
            other => Err(format!("expected a visible frame, got {other:?}")),
        }
    };
    let rgb_off = rgb_of(&off)?;
    let rgb_on = rgb_of(&on)?;
    write_png(&format!("{}_off.png", opts.out_prefix), &rgb_off, nx, ny)?;
    write_png(&format!("{}_on.png", opts.out_prefix), &rgb_on, nx, ny)?;
    if let Some((cx, cy, cw, ch)) = opts.crop {
        let crop_off = crop_zoom(&rgb_off, nx, ny, cx, cy, cw, ch, opts.zoom)?;
        let crop_on = crop_zoom(&rgb_on, nx, ny, cx, cy, cw, ch, opts.zoom)?;
        write_png(
            &format!("{}_off_crop.png", opts.out_prefix),
            &crop_off.0,
            crop_off.1,
            crop_off.2,
        )?;
        write_png(
            &format!("{}_on_crop.png", opts.out_prefix),
            &crop_on.0,
            crop_on.1,
            crop_on.2,
        )?;
    }

    // Diff stats over the RGB planes.
    let mut changed_px = 0usize;
    let mut sum_abs = 0.0f64;
    let mut max_abs = 0u32;
    for px in 0..(nx * ny) {
        let mut px_changed = false;
        for c in 0..3 {
            let a = rgb_off[px * 3 + c] as i32;
            let b = rgb_on[px * 3 + c] as i32;
            let d = (a - b).unsigned_abs();
            if d > 0 {
                px_changed = true;
            }
            sum_abs += d as f64;
            max_abs = max_abs.max(d);
        }
        if px_changed {
            changed_px += 1;
        }
    }
    let changed_frac = changed_px as f64 / (nx * ny) as f64;
    let mean_abs = sum_abs / (nx * ny * 3) as f64;

    // The dx-derived amplitude (from the returned projection; MAP_PROJ 6 stores
    // degrees, converted like the render assembly does).
    let proj = &on.georef.projection;
    let dx_m = if proj.map_proj == 6 {
        proj.dx_m.min(proj.dy_m) * 111_195.0
    } else {
        proj.dx_m.min(proj.dy_m)
    };
    let amplitude = granulation_amplitude(dx_m);

    println!(
        "GRANSUMMARY dims={nx}x{ny} view={} dx_m={dx_m:.0} amplitude={amplitude:.4} \
         gran_flag_off={} gran_flag_on={} sun_elev={:.2} changed_frac={changed_frac:.4} \
         mean_abs_delta={mean_abs:.4} max_abs_delta={max_abs} \
         wall_off={wall_off:.2}s wall_on={wall_on:.2}s",
        opts.view.slug(),
        off.granulation,
        on.granulation,
        on.sun_elev_deg,
    );

    // The raw-Kelvin contract proof: the band-13 BT plane must be BYTE-IDENTICAL with
    // the granulation flag forced on vs off (thermal products never granulate).
    if opts.ir_check {
        let bt_of = |granulation: bool| -> Result<Vec<f32>, String> {
            let p = params_at(granulation);
            match api::render(&p, Product::Ir)?.data {
                FrameData::Ir { bt_kelvin, .. } => Ok(bt_kelvin),
                other => Err(format!("expected an IR frame, got {other:?}")),
            }
        };
        let bt_off = bt_of(false)?;
        let bt_on = bt_of(true)?;
        let identical = bt_off.len() == bt_on.len()
            && bt_off
                .iter()
                .zip(bt_on.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits());
        println!("GRANIR identical={identical} n={}", bt_off.len());
        if !identical {
            return Err("IR BT plane differs with granulation on — the raw-Kelvin \
                        contract is broken"
                .to_string());
        }
    }
    Ok(())
}

/// Write the round-2 DECK-COHERENCE gate field ([`GranCoherence`]) of a cached
/// brick as a grayscale PNG (white = open / granulate, black = closed deck),
/// row-flipped to north-up so it registers with the top-down frame, and print a
/// `GRANCOH` stats line. Requires a `run.json` input (the brick must already be
/// ingested — render the A/B pair once first for a wrfout input).
fn write_coherence_map(input: &Path, timestep: usize, out: &Path) -> Result<(), String> {
    let manifest = RunManifest::load(input)
        .map_err(|e| format!("coherence-map needs a run.json input: {e}"))?;
    let cache_dir = input
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .ok_or("coherence-map: run.json has no cache root above it")?;
    let ts = manifest.timesteps.get(timestep).ok_or_else(|| {
        format!(
            "timestep {timestep} out of range (run has {} timesteps)",
            manifest.timesteps.len()
        )
    })?;
    let brick_path = bricks::run_dir(&cache_dir, &manifest.run_id).join(&ts.file);
    let brick = bricks::read_ssb(&brick_path)
        .map_err(|e| format!("read brick {}: {e}", brick_path.display()))?;
    let p = &manifest.projection;
    let pitch = if p.map_proj == 6 {
        p.dx_m.min(p.dy_m) * 111_195.0
    } else {
        p.dx_m.min(p.dy_m)
    };
    let (nx, ny) = (brick.nx, brick.ny);
    let vol = DecodedVolume::from_brick_legacy(&brick, pitch);
    let coh = GranCoherence::build(&vol);
    let (open, partial, closed) = coh.stats();
    println!(
        "GRANCOH dims={nx}x{ny} open={open} partial={partial} closed={closed} \
         open_frac={:.4} closed_frac={:.4}",
        open as f64 / (nx * ny) as f64,
        closed as f64 / (nx * ny) as f64,
    );
    let mut img = GrayImage::new(nx as u32, ny as u32);
    for y in 0..ny {
        let j = ny - 1 - y; // north-up, like the top-down frame
        for x in 0..nx {
            let g = coh.gate_at(x as f64, j as f64);
            img.put_pixel(x as u32, y as u32, image::Luma([(g * 255.0).round() as u8]));
        }
    }
    img.save(out)
        .map_err(|e| format!("write {}: {e}", out.display()))?;
    eprintln!(
        "render_granulation_ab: wrote coherence map {} ({nx}x{ny})",
        out.display()
    );
    Ok(())
}

fn write_png(path: &str, rgb: &[u8], nx: usize, ny: usize) -> Result<(), String> {
    let img = RgbImage::from_raw(nx as u32, ny as u32, rgb.to_vec())
        .ok_or_else(|| format!("bad frame dims {nx}x{ny}"))?;
    img.save(path).map_err(|e| format!("write {path}: {e}"))?;
    eprintln!("render_granulation_ab: wrote {path} ({nx}x{ny})");
    Ok(())
}

/// Crop a pixel rect out of an RGB frame and magnify it by `zoom` (nearest
/// neighbour — a pure pixel magnification for visual review, no resampling).
#[allow(clippy::too_many_arguments)]
fn crop_zoom(
    rgb: &[u8],
    nx: usize,
    ny: usize,
    cx: usize,
    cy: usize,
    cw: usize,
    ch: usize,
    zoom: usize,
) -> Result<(Vec<u8>, usize, usize), String> {
    if cw == 0 || ch == 0 || cx + cw > nx || cy + ch > ny {
        return Err(format!(
            "crop {cx},{cy},{cw},{ch} outside the {nx}x{ny} frame"
        ));
    }
    let (ow, oh) = (cw * zoom, ch * zoom);
    let mut out = vec![0u8; ow * oh * 3];
    for oy in 0..oh {
        let sy = cy + oy / zoom;
        for ox in 0..ow {
            let sx = cx + ox / zoom;
            let s = (sy * nx + sx) * 3;
            let d = (oy * ow + ox) * 3;
            out[d..d + 3].copy_from_slice(&rgb[s..s + 3]);
        }
    }
    Ok((out, ow, oh))
}

fn parse_opts(args: &[String]) -> Result<Opts, String> {
    let mut opts = Opts {
        input: PathBuf::new(),
        out_prefix: String::new(),
        sat: SatellitePreset::GoesEast,
        timestep: 0,
        view: ViewMode::TopDownMap,
        resolution: ResolutionMode::Native,
        margin: 0.0,
        steps: StepQuality::Offline,
        sun_elev_override: None,
        sun_az_override: None,
        exposure: DEFAULT_EXPOSURE,
        cache: ingest::default_cache_dir(),
        bluemarble: None,
        bluemarble_month: None,
        bluemarble_download: true,
        threads: None,
        crop: None,
        zoom: 4,
        ir_check: true,
        coherence_map: None,
    };
    for arg in args {
        let (key, value) = arg
            .split_once('=')
            .ok_or_else(|| format!("expected key=value, got `{arg}`"))?;
        match key {
            "input" => opts.input = PathBuf::from(value),
            "out-prefix" => opts.out_prefix = value.to_string(),
            "sat" => {
                opts.sat = match value {
                    "goes-east" => SatellitePreset::GoesEast,
                    "goes-west" => SatellitePreset::GoesWest,
                    "himawari" => SatellitePreset::Himawari,
                    _ => return Err(format!("unknown sat `{value}`")),
                }
            }
            "timestep" => opts.timestep = value.parse().map_err(|_| "bad timestep")?,
            "view" => {
                opts.view = match value {
                    "geo" => ViewMode::Geostationary,
                    "topdown" => ViewMode::TopDownMap,
                    _ => return Err(format!("unknown view `{value}`")),
                }
            }
            "resolution" => {
                opts.resolution = match value {
                    "native" => ResolutionMode::Native,
                    "abi1km" => ResolutionMode::Abi1km,
                    "abi2km" => ResolutionMode::Abi2km,
                    _ => return Err(format!("unknown resolution `{value}`")),
                }
            }
            "margin" => opts.margin = value.parse().map_err(|_| "bad margin")?,
            "steps" => {
                opts.steps = match value {
                    "offline" => StepQuality::Offline,
                    "interactive" => StepQuality::Interactive,
                    _ => return Err(format!("unknown steps `{value}`")),
                }
            }
            "sun-elev" => opts.sun_elev_override = Some(value.parse().map_err(|_| "bad sun-elev")?),
            "sun-az" => opts.sun_az_override = Some(value.parse().map_err(|_| "bad sun-az")?),
            "exposure" => opts.exposure = value.parse().map_err(|_| "bad exposure")?,
            "cache" => opts.cache = PathBuf::from(value),
            "bluemarble" => opts.bluemarble = Some(PathBuf::from(value)),
            "bluemarble-month" => {
                let m: u32 = value.parse().map_err(|_| "bad bluemarble-month")?;
                if !(1..=12).contains(&m) {
                    return Err("bluemarble-month must be 1..=12".to_string());
                }
                opts.bluemarble_month = Some(m);
            }
            "bluemarble-download" => opts.bluemarble_download = parse_bool(value)?,
            "threads" => opts.threads = Some(value.parse().map_err(|_| "bad threads")?),
            "crop" => {
                let parts: Vec<usize> = value
                    .split(',')
                    .map(|v| v.trim().parse::<usize>())
                    .collect::<Result<_, _>>()
                    .map_err(|_| "bad crop (want X,Y,W,H)")?;
                if parts.len() != 4 {
                    return Err("crop wants X,Y,W,H".to_string());
                }
                opts.crop = Some((parts[0], parts[1], parts[2], parts[3]));
            }
            "zoom" => {
                let z: usize = value.parse().map_err(|_| "bad zoom")?;
                if !(1..=8).contains(&z) {
                    return Err("zoom must be 1..=8".to_string());
                }
                opts.zoom = z;
            }
            "ir" => opts.ir_check = parse_bool(value)?,
            "coherence-map" => opts.coherence_map = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown key `{key}`")),
        }
    }
    if opts.input.as_os_str().is_empty() {
        return Err("input= is required".to_string());
    }
    if opts.out_prefix.is_empty() {
        return Err("out-prefix= is required".to_string());
    }
    Ok(opts)
}

fn parse_bool(v: &str) -> Result<bool, String> {
    match v {
        "on" | "true" | "1" | "yes" => Ok(true),
        "off" | "false" | "0" | "no" => Ok(false),
        _ => Err(format!("expected on|off, got `{v}`")),
    }
}

fn print_usage() {
    eprintln!(
        "render_granulation_ab input=<wrfout|run.json> out-prefix=<prefix> \
         [sat=goes-east] [timestep=0] [view=topdown] [resolution=native] [margin=0.0] \
         [steps=offline] [sun-elev=deg] [sun-az=deg] [exposure={DEFAULT_EXPOSURE}] \
         [cache=dir] [bluemarble=path] [bluemarble-month=MM] [bluemarble-download=on] \
         [threads=N] [crop=X,Y,W,H] [zoom=4] [ir=on] [coherence-map=path.png]"
    );
}
