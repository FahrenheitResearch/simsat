//! `render_ir` — headless "render one synthetic thermal frame to PNG".
//!
//! The IR/WV sibling of `render_frame`: a CPU-only harness so a thermal frame can be
//! visually QA'd without a GPU or the studio GUI. Renders the 10.3 um window (band 13,
//! default) OR a water-vapor band (`wv=6.2|6.9|7.3`). It is now a THIN wrapper over
//! [`simsat::api::render`] (the one shared render assembly): parse the CLI, call
//! `api::render(Product::Ir)`, then colour + write the PNG. IR is thermal, so it works at
//! ANY timestep including night (no sun input at all).
//!
//! USAGE (key=value args, any order):
//!
//!   render_ir input=<path> out=<file.png> [key=value ...]
//!
//!   input=<path>        REQUIRED. A wrfout file (ingested if not cached) OR a run.json.
//!   out=<file.png>      REQUIRED. Output PNG path (RGB8, row 0 = north).
//!   bt-out=<path.bin>   OPTIONAL audit dump: north-first unletterboxed f32le Kelvin plane.
//!   sat=<preset>        goes-east | goes-west | himawari | mtg-i1 (default goes-east)
//!                       MTG aliases: mtg | meteosat-12 | meteosat12. This is the 0-degree
//!                       camera with generic thermal/WV physics, not an FCI SRF or PSF.
//!   geo-navigation=<mode> model-sphere | goes-r-abi (GOES-East/West only; default model-sphere)
//!   timestep=<n>        time index (default 0).
//!   resolution=<mode>   native | abi1km | abi2km            (default native)
//!                       native = one output pixel per source grid cell. ABI 1/2 km are
//!                       output sampling choices and may upsample coarse or downsample fine data.
//!   enhancement=<name>  natural | cimss | bd | avn | funktop | rainbow | gray
//!                       (default cimss for Band 13 and WV)
//!   wv=<band>           6.2 | 6.9 | 7.3  — render a WATER-VAPOR band instead of the
//!                       10.3 um window (band 13). Thermal either way.
//!   derived=<field>     pw | ctt | cod  — render a DERIVED scalar-field MAP (precipitable
//!                       water mm / cloud-top temp K / cloud optical depth) with its basic
//!                       colormap instead of a BT band. Prints DERIVEDSUMMARY (raw min/max).
//!   cache=<dir>         brick cache root (read/write).
//!   view=<mode>         geo | topdown  (default geo). topdown = a top-down BT MAP.
//!   canvas=<WxH>        OPTIONAL letterbox into a fixed figure size (black pad).
//!   threads=<N>         OPTIONAL rayon thread cap (else honor RAYON_NUM_THREADS).
//!
//! On completion it prints a one-line `IRSUMMARY ...` with dims, on-earth fraction, the
//! coldest cloud-top BT, the warmest ground BT, the median BT, and the wall time.

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use image::RgbImage;
use simsat::api::{self, BlueMarble, FrameData, Product, RenderParams};
use simsat::bricks::StorageProfile;
use simsat::camera::{GeoNavigation, ResolutionMode, SatellitePreset, ViewMode};
use simsat::derived::{self, DerivedField};
use simsat::instrument_footprint::InstrumentFootprint;
use simsat::ir::ir_frame_stats;
use simsat::ir_enhance::IrEnhancement;
use simsat::thermal_sensor::ThermalSensor;
use simsat::topdown;
use simsat::wv::WvBand;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = run(&args) {
        eprintln!("render_ir: {e}");
        eprintln!("run with no args (or --help) for usage.");
        std::process::exit(1);
    }
}

struct Opts {
    input: PathBuf,
    storage_profile: StorageProfile,
    out: PathBuf,
    /// Optional audit dump of the raw, unletterboxed brightness-temperature plane.
    bt_out: Option<PathBuf>,
    sat: SatellitePreset,
    geo_navigation: GeoNavigation,
    timestep: usize,
    resolution: ResolutionMode,
    /// Zoom-out / domain margin as a FRACTION added on each side (0.0 = edge-to-edge). The
    /// thermal margin is NO-DATA (NaN / masked): WRF has no skin/air temperature outside the
    /// domain, so the honest thermal fallback marks the margin as no data.
    margin: f64,
    enhancement: IrEnhancement,
    cache: PathBuf,
    view: ViewMode,
    canvas: Option<(usize, usize)>,
    threads: Option<usize>,
    /// `Some(band)` renders a WATER-VAPOR band (6.2/6.9/7.3 um) instead of the 10.3 um
    /// window (band 13). Thermal either way.
    wv: Option<WvBand>,
    /// `Some(field)` renders a DERIVED scalar-field map (precipitable water / cloud-top temp /
    /// cloud optical depth) instead of a brightness-temperature band. Takes precedence over wv.
    derived: Option<DerivedField>,
    /// Spectral response for the Band 13 window product.
    sensor: ThermalSensor,
    /// Complete-radiance ABI Band 13 angular-grid footprint (default off).
    instrument_footprint: InstrumentFootprint,
}

fn run(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return Ok(());
    }
    let opts = parse_opts(args)?;
    let band = opts.wv.map(|b| b.abi_band()).unwrap_or(13);
    let band_label = opts
        .wv
        .map(|b| format!("WV {} um (band {})", b.micron(), b.abi_band()))
        .unwrap_or_else(|| "band 13".to_string());
    let product_label = opts
        .derived
        .map(|f| f.label().to_string())
        .unwrap_or_else(|| band_label.clone());
    eprintln!(
        "render_ir: input={} product={} sensor={} instrument-footprint={} storage-profile={} view={} sat={} geo-navigation={} ts={} res={} enhancement={} canvas={} threads={}",
        opts.input.display(),
        product_label,
        opts.sensor.slug(),
        opts.instrument_footprint.slug(),
        opts.storage_profile.slug(),
        opts.view.slug(),
        opts.sat.slug(),
        opts.geo_navigation.slug(),
        opts.timestep,
        opts.resolution.label(),
        opts.enhancement.slug(),
        opts.canvas
            .map(|(w, h)| format!("{w}x{h}"))
            .unwrap_or_else(|| "native".to_string()),
        opts.threads
            .map(|n| n.to_string())
            .unwrap_or_else(|| "auto".to_string()),
    );
    if opts.sat == SatellitePreset::MtgI1 {
        eprintln!(
            "render_ir: MTG-I1 / Meteosat-12 selects the 0-degree camera only; thermal \
             and water-vapor behavior remains SimSat's generic model (no FCI spectral \
             response or PSF)."
        );
    }
    topdown::configure_global_rayon(topdown::effective_thread_count(
        opts.threads,
        std::env::var("RAYON_NUM_THREADS").ok().as_deref(),
    ));
    simsat::platform::lower_ingest_thread_priority();

    let params = RenderParams {
        input: opts.input.clone(),
        storage_profile: opts.storage_profile,
        satellite: opts.sat,
        geo_navigation: opts.geo_navigation,
        timestep: opts.timestep,
        view: opts.view,
        resolution: opts.resolution,
        margin_frac: opts.margin as f32,
        cache: opts.cache.clone(),
        ir_enhancement: Some(opts.enhancement),
        thermal_sensor: opts.sensor,
        instrument_footprint: opts.instrument_footprint,
        // A derived field asks the api to also produce the basic studio colormap RGB (the
        // harness writes a coloured PNG). Ignored by the IR/WV products.
        derived_colormap: opts.derived.is_some(),
        // Visible-only fields are irrelevant to the thermal IR march.
        bluemarble: BlueMarble::FlatAlbedo,
        ..RenderParams::new(opts.input.clone())
    };

    let product = match (opts.derived, opts.wv) {
        (Some(field), _) => Product::Derived { field },
        (None, Some(band)) => Product::WaterVapor { band },
        (None, None) => Product::Ir,
    };
    let t0 = Instant::now();
    let result = api::render(&params, product)?;
    let wall = t0.elapsed();
    let abi_lattice_crop = result
        .raster
        .scan
        .abi_2km_global_indices()
        .map(|(x0, x1, y0, y1)| format!("x{x0}:{x1},y{y0}:{y1}"))
        .unwrap_or_else(|| "none".to_string());

    for warning in &result.science_warnings {
        eprintln!("render_ir: SCIENCE WARNING: {warning}");
    }

    // A DERIVED scalar-field map (precipitable water / cloud-top temp / cloud optical depth):
    // write the basic colormap PNG + a DERIVEDSUMMARY of the raw field. The RAW array is the
    // primary deliverable (the binding); this harness is the QA-frame renderer.
    if let Some(field) = opts.derived {
        let (values, rgb) = match &result.data {
            FrameData::Scalar { values, rgb, .. } => {
                (values, rgb.as_ref().expect("derived colormap rgb"))
            }
            _ => return Err("expected a derived scalar frame".to_string()),
        };
        let (rnx, rny) = (result.nx, result.ny);
        let (final_nx, final_ny, final_rgb) = match opts.canvas {
            Some((cw, ch)) => (cw, ch, topdown::letterbox_rgb(rgb, rnx, rny, cw, ch)),
            None => (rnx, rny, rgb.clone()),
        };
        write_rgb8_png(&opts.out, final_nx, final_ny, &final_rgb)?;
        let stats = derived::field_stats(values);
        let on_earth_frac = stats.finite as f64 / (rnx * rny).max(1) as f64;
        eprintln!(
            "render_ir: rendered {rnx}x{rny} ({}, {}){} {} in {:.3}s -> {}",
            opts.view.slug(),
            opts.resolution.label(),
            if result.res_clamped {
                " [clamped to cap]"
            } else {
                ""
            },
            field.label(),
            wall.as_secs_f64(),
            opts.out.display(),
        );
        println!(
            "DERIVEDSUMMARY file={} field={} units={} view={} dims={}x{} canvas={} res={}{} \
             sat={} sat_scope=camera in_domain_frac={:.3} min={:.3} max={:.3} median={:.3} wall_s={:.3}",
            opts.out.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
            field.slug(),
            if field.units().is_empty() {
                "dimensionless"
            } else {
                field.units()
            },
            opts.view.slug(),
            final_nx,
            final_ny,
            opts.canvas
                .map(|(w, h)| format!("{w}x{h}"))
                .unwrap_or_else(|| "none".to_string()),
            opts.resolution.label().replace(' ', "_"),
            if result.res_clamped { "[clamped]" } else { "" },
            opts.sat.slug(),
            on_earth_frac,
            stats.min,
            stats.max,
            stats.median,
            wall.as_secs_f64(),
        );
        return Ok(());
    }

    let (bt, rgb) = match &result.data {
        FrameData::Ir { bt_kelvin, rgb } => (bt_kelvin, rgb.as_ref().expect("enhancement RGB")),
        _ => return Err("expected a thermal (IR/WV) frame".to_string()),
    };
    let (rnx, rny) = (result.nx, result.ny);
    eprintln!(
        "render_ir: rendered {}x{} ({}, {}){} {} in {:.3}s",
        rnx,
        rny,
        opts.view.slug(),
        opts.resolution.label(),
        if result.res_clamped {
            " [clamped to cap]"
        } else {
            ""
        },
        band_label,
        wall.as_secs_f64(),
    );

    if let Some(path) = &opts.bt_out {
        write_f32le(path, bt)?;
        eprintln!(
            "render_ir: wrote north-first {}x{} f32le Kelvin audit plane {}",
            rnx,
            rny,
            path.display()
        );
    }

    // Optional canvas letterbox to a fixed figure size.
    let (final_nx, final_ny, final_rgb) = match opts.canvas {
        Some((cw, ch)) => (cw, ch, topdown::letterbox_rgb(rgb, rnx, rny, cw, ch)),
        None => (rnx, rny, rgb.clone()),
    };
    write_rgb8_png(&opts.out, final_nx, final_ny, &final_rgb)?;

    // Stats for the manifest.
    let stats = ir_frame_stats(bt);
    let on_earth_frac = stats.finite as f64 / (rnx * rny).max(1) as f64;
    eprintln!("render_ir: wrote {}", opts.out.display());
    println!(
        "IRSUMMARY file={} band={} sensor={} response={} instrument_footprint={} abi_lattice_crop={} view={} dims={}x{} canvas={} res={}{} sat={} sat_scope=camera enhancement={} \
         on_earth_frac={:.3} cold_top_bt={:.1} warm_ground_bt={:.1} median_bt={:.1} \
         all_finite={} tsk_fallback={} wall_s={:.3}",
        opts.out.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
        band,
        opts.sensor.slug(),
        opts.sensor.metadata().response.replace(' ', "_"),
        result.instrument_footprint.slug(),
        abi_lattice_crop,
        opts.view.slug(),
        final_nx,
        final_ny,
        opts.canvas
            .map(|(w, h)| format!("{w}x{h}"))
            .unwrap_or_else(|| "none".to_string()),
        opts.resolution.label().replace(' ', "_"),
        if result.res_clamped { "[clamped]" } else { "" },
        opts.sat.slug(),
        opts.enhancement.slug(),
        on_earth_frac,
        stats.min_bt,
        stats.max_bt,
        stats.median_bt,
        stats.all_finite_in_domain,
        // WS1: the process-wide TSK-fallback diagnostic (a missing/all-zero TSK
        // plane was substituted by the lowest-level air temperature).
        simsat::ir::tsk_fallback_engaged(),
        wall.as_secs_f64(),
    );
    Ok(())
}

/// Write an RGB8 buffer (`nx*ny*3`, row 0 = north) to a PNG.
fn write_rgb8_png(path: &Path, nx: usize, ny: usize, rgb: &[u8]) -> Result<(), String> {
    if rgb.len() != nx * ny * 3 {
        return Err(format!("rgb byte count {} != {nx}x{ny}x3", rgb.len()));
    }
    let img = RgbImage::from_fn(nx as u32, ny as u32, |x, y| {
        let o = (y as usize * nx + x as usize) * 3;
        image::Rgb([rgb[o], rgb[o + 1], rgb[o + 2]])
    });
    img.save(path)
        .map_err(|e| format!("write PNG {}: {e}", path.display()))
}

/// Write a north-first scalar f32le plane without changing NaN no-data values.
///
/// This is deliberately an audit-only CLI dump, not a display setting. It is written
/// before any optional canvas letterbox so its layout remains exactly `ny * nx`.
fn write_f32le(path: &Path, values: &[f32]) -> Result<(), String> {
    let file = std::fs::File::create(path)
        .map_err(|e| format!("create f32le BT dump {}: {e}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for value in values {
        writer
            .write_all(&value.to_le_bytes())
            .map_err(|e| format!("write f32le BT dump {}: {e}", path.display()))?;
    }
    writer
        .flush()
        .map_err(|e| format!("flush f32le BT dump {}: {e}", path.display()))
}

fn parse_opts(args: &[String]) -> Result<Opts, String> {
    let mut input: Option<PathBuf> = None;
    let mut storage_profile = StorageProfile::CompactU8;
    let mut out: Option<PathBuf> = None;
    let mut bt_out: Option<PathBuf> = None;
    let mut sat = SatellitePreset::GoesEast;
    let mut geo_navigation = GeoNavigation::ModelSphere;
    let mut timestep = 0usize;
    let mut resolution = ResolutionMode::Native;
    let mut margin = 0.0f64;
    let mut enhancement = IrEnhancement::default();
    let mut cache = simsat::ingest::default_cache_dir();
    let mut view = ViewMode::Geostationary;
    let mut canvas: Option<(usize, usize)> = None;
    let mut threads: Option<usize> = None;
    let mut wv: Option<WvBand> = None;
    let mut derived: Option<DerivedField> = None;
    let mut sensor = ThermalSensor::FastGray;
    let mut instrument_footprint = InstrumentFootprint::Off;
    let mut enhancement_explicit = false;

    for a in args {
        let (k, v) = a
            .split_once('=')
            .ok_or_else(|| format!("expected key=value, got '{a}'"))?;
        match k {
            "input" | "wrfout" | "in" => input = Some(PathBuf::from(v)),
            "out" | "output" | "png" => out = Some(PathBuf::from(v)),
            "bt-out" | "bt_out" | "btout" => bt_out = Some(PathBuf::from(v)),
            "sat" | "satellite" => sat = parse_sat(v)?,
            "geo-navigation" | "geo_navigation" | "navigation" => {
                geo_navigation = parse_geo_navigation(v)?
            }
            "timestep" | "ts" => timestep = v.parse().map_err(|_| format!("bad timestep '{v}'"))?,
            "resolution" | "res" => resolution = parse_resolution(v)?,
            "margin" | "zoom-out" | "zoomout" => {
                margin = v.parse().map_err(|_| format!("bad margin '{v}'"))?;
                if !(0.0..=4.0).contains(&margin) {
                    return Err(format!("margin must be 0.0..=4.0 (fraction), got {margin}"));
                }
            }
            "enhancement" | "enh" | "e" => {
                // STRICT parse (WS1): an unknown token is an ERROR, not a silent
                // fall-back to the default (a `grayscale` typo used to render CIMSS).
                enhancement = IrEnhancement::parse_strict(v).ok_or_else(|| {
                    format!(
                        "unknown enhancement '{v}' (natural|cimss|bd|avn|funktop|rainbow|gray; \
                         noaa/heritage accepted for natural; grayscale/greyscale accepted for gray)"
                    )
                })?;
                enhancement_explicit = true;
            }
            "cache" => cache = PathBuf::from(v),
            "storage-profile" | "storage_profile" | "brick-storage" => {
                storage_profile = StorageProfile::parse(v).ok_or_else(|| {
                    format!("bad storage-profile '{v}' (expected compact-u8|science-cloud-f16)")
                })?;
            }
            "view" => view = parse_view(v)?,
            "canvas" | "figure" | "size" => canvas = Some(parse_canvas(v)?),
            "threads" | "rayon-threads" | "num-threads" => {
                threads = Some(v.parse().map_err(|_| format!("bad threads '{v}'"))?)
            }
            "wv" | "band" | "water-vapor" => {
                wv = Some(
                    WvBand::parse(v).ok_or_else(|| format!("bad wv band '{v}' (6.2|6.9|7.3)"))?,
                )
            }
            "derived" | "field" => {
                derived = Some(
                    DerivedField::parse(v)
                        .ok_or_else(|| format!("bad derived field '{v}' (pw|ctt|cod)"))?,
                )
            }
            "sensor" | "response" | "spectral-response" => {
                sensor = ThermalSensor::parse(v)
                    .ok_or_else(|| format!("bad sensor '{v}' (fast-gray|goes-r-abi-band13-fm4)"))?
            }
            "instrument-footprint" | "instrument_footprint" | "channel-footprint" => {
                instrument_footprint = InstrumentFootprint::parse(v).ok_or_else(|| {
                    format!("bad instrument-footprint '{v}' (off|goes-r-abi-band13-mtf-prototype)")
                })?
            }
            other => return Err(format!("unknown key '{other}'")),
        }
    }
    // WV uses CIMSS as its classic moisture palette. Band 13 also ships with the
    // owner-selected CIMSS Style default. An explicit enhancement= wins either way.
    if wv.is_some() && !enhancement_explicit {
        enhancement = IrEnhancement::Cimss;
    }
    if sensor != ThermalSensor::FastGray && (wv.is_some() || derived.is_some()) {
        return Err("sensor=goes-r-abi-band13-fm4 applies only to the Band 13 window product (omit wv=/derived=)".to_string());
    }
    if bt_out.is_some() && derived.is_some() {
        return Err(
            "bt-out= applies only to thermal IR/WV brightness-temperature products (omit derived=)"
                .to_string(),
        );
    }
    Ok(Opts {
        input: input.ok_or("missing required input=<path>")?,
        storage_profile,
        out: out.ok_or("missing required out=<file.png>")?,
        bt_out,
        sat,
        geo_navigation,
        timestep,
        resolution,
        margin,
        enhancement,
        cache,
        view,
        canvas,
        threads,
        wv,
        derived,
        sensor,
        instrument_footprint,
    })
}

fn parse_view(v: &str) -> Result<ViewMode, String> {
    match v.to_ascii_lowercase().replace(['-', '_', ' '], "").as_str() {
        "geo" | "geostationary" | "fromspace" | "space" => Ok(ViewMode::Geostationary),
        "topdown" | "top" | "map" | "topdownmap" | "nadir" => Ok(ViewMode::TopDownMap),
        _ => Err(format!("unknown view '{v}' (geo|topdown)")),
    }
}

fn parse_geo_navigation(v: &str) -> Result<GeoNavigation, String> {
    match v.to_ascii_lowercase().replace(['_', ' '], "-").as_str() {
        "model-sphere" | "sphere" | "model" | "default" => Ok(GeoNavigation::ModelSphere),
        "goes-r-abi" | "goes-r" | "abi" | "ellipsoid" => Ok(GeoNavigation::GoesRAbiFixedGrid),
        _ => Err(format!(
            "unknown geo-navigation '{v}' (expected model-sphere|goes-r-abi)"
        )),
    }
}

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
        "mtg" | "mtgi1" | "meteosat12" => Ok(SatellitePreset::MtgI1),
        _ => Err(format!(
            "unknown satellite '{v}' (goes-east|goes-west|himawari|mtg-i1)"
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

fn print_usage() {
    eprintln!(
        "render_ir — headless synthetic-IR (band 13) render to PNG (CPU, no GPU).\n\n\
         USAGE:\n  render_ir input=<wrfout|run.json> out=<file.png> [key=value ...]\n\n\
         KEYS:\n\
         \x20 input=<path>        wrfout (ingest-if-needed) or a cached run.json  [required]\n\
         \x20 out=<file.png>      output PNG (RGB8, row 0 = north)                [required]\n\
         \x20 bt-out=<file.bin>   audit dump: north-first unletterboxed f32le Kelvin plane\n\
         \x20 storage-profile=<mode> compact-u8 | science-cloud-f16 (default compact-u8)\n\
         \x20 sat=<preset>        goes-east | goes-west | himawari | mtg-i1 (default goes-east)\n\
         \x20                     MTG aliases: mtg | meteosat-12 | meteosat12\n\
         \x20                     0-degree camera with generic IR/WV; no FCI SRF/PSF\n\
         \x20 geo-navigation=<mode> model-sphere | goes-r-abi (GOES-East/West only; default model-sphere)\n\
         \x20 timestep=<n>        time index (default 0)\n\
         \x20 resolution=<mode>   native | abi1km | abi2km           (default native)\n\
         \x20                     native = one pixel per source-grid cell; ABI 1/2 km may\n\
         \x20                     upsample coarse or downsample fine model grids\n\
         \x20 margin=<frac>       zoom-out margin fraction on each side (default 0.0; thermal margin = no-data)\n\
         \x20 enhancement=<name>  natural|cimss|bd|avn|funktop|rainbow|gray\n\
         \x20                     (default cimss for Band 13 and WV)\n\
         \x20 sensor=<response>   fast-gray (default) | goes-r-abi-band13-fm4 (official NOAA SRF)\n\
         \x20 instrument-footprint=<mode> off (default) | goes-r-abi-band13-mtf-prototype; exact global 56-urad lattice (requires FM4 + GOES-R exact nav + geo ABI2km)\n\
         \x20 wv=<band>           6.2|6.9|7.3  render a water-vapor band (else band 13)\n\
         \x20 derived=<field>     pw|ctt|cod  render a derived scalar-field map (mm/K/tau)\n\
         \x20 cache=<dir>         brick cache root (default: studio cache dir)\n\
         \x20 view=<mode>         geo | topdown  — from-space (default) OR top-down map BT\n\
         \x20 canvas=<WxH>        letterbox into a fixed figure size, black pad (e.g. 1100x850)\n\
         \x20 threads=<N>         rayon thread cap (else honor RAYON_NUM_THREADS)\n"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mtg_camera_aliases_parse_and_report_the_canonical_slug() {
        for token in ["mtg", "mtg-i1", "meteosat-12", "meteosat12"] {
            assert_eq!(parse_sat(token).unwrap(), SatellitePreset::MtgI1, "{token}");
            assert_eq!(parse_sat(token).unwrap().slug(), "mtgi1", "{token}");
        }
    }

    #[test]
    fn storage_profile_defaults_compact_and_parses_science() {
        let base = vec!["input=input".to_string(), "out=out.png".to_string()];
        assert_eq!(
            parse_opts(&base).unwrap().storage_profile,
            StorageProfile::CompactU8
        );
        let mut science = base;
        science.push("storage-profile=science-cloud-f16".to_string());
        assert_eq!(
            parse_opts(&science).unwrap().storage_profile,
            StorageProfile::ScienceCloudF16
        );
    }

    #[test]
    fn band13_and_water_vapor_default_cimss_while_explicit_natural_is_preserved() {
        let base = vec!["input=input".to_string(), "out=out.png".to_string()];
        assert_eq!(parse_opts(&base).unwrap().enhancement, IrEnhancement::Cimss);

        let mut wv = base.clone();
        wv.push("wv=6.2".to_string());
        assert_eq!(parse_opts(&wv).unwrap().enhancement, IrEnhancement::Cimss);

        let mut explicit = wv;
        explicit.push("enhancement=natural".to_string());
        assert_eq!(
            parse_opts(&explicit).unwrap().enhancement,
            IrEnhancement::Natural
        );
    }
}
