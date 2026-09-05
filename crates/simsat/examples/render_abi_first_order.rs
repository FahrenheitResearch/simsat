//! Explicit incomplete spectral ABI reference; independent of shipped display.
//! Required: input= cache= spectral-surface= config= out-dir=
//! Optional: timestep=0 sat=goes-east|goes-west threads=6
//! Config JSON declares all reference-atmosphere and numerical assumptions.
//! Output C01/C02/C03 are pi*L/E, not display RGB and not solar-zenith normalized.
use sha2::{Digest, Sha256};
use simsat::{
    abi_first_order::FirstOrderConfig,
    api::{self, RenderIntent, RenderParams},
    camera::{GeoNavigation, ResolutionMode, SatellitePreset, ViewMode},
};
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    path::{Path, PathBuf},
};
fn hash(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut b = vec![0u8; 1024 * 1024];
    loop {
        let n = f.read(&mut b)?;
        if n == 0 {
            break;
        }
        h.update(&b[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
fn plane(
    path: &Path,
    values: impl IntoIterator<Item = f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    for v in values {
        f.write_all(&v.to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = BTreeMap::new();
    for a in std::env::args().skip(1) {
        let (k,v)=a.split_once('=').ok_or("usage: input= cache= spectral-surface= config= out-dir= [timestep=0 sat=goes-east threads=6]")?;
        if args.insert(k.to_string(), v.to_string()).is_some() {
            return Err("duplicate argument".into());
        }
    }
    let mut required = |name: &str| -> Result<String, Box<dyn std::error::Error>> {
        args.remove(name)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| format!("missing {name}").into())
    };
    let input = PathBuf::from(required("input")?);
    let cache = PathBuf::from(required("cache")?);
    let surface = PathBuf::from(required("spectral-surface")?);
    let config_path = PathBuf::from(required("config")?);
    let out = PathBuf::from(required("out-dir")?);
    let timestep = args
        .remove("timestep")
        .unwrap_or_else(|| "0".into())
        .parse::<usize>()?;
    let satellite = match args.remove("sat").as_deref().unwrap_or("goes-east") {
        "goes-east" => SatellitePreset::GoesEast,
        "goes-west" => SatellitePreset::GoesWest,
        _ => return Err("first-order ABI requires goes-east or goes-west".into()),
    };
    let threads = args
        .remove("threads")
        .unwrap_or_else(|| "6".into())
        .parse::<usize>()?;
    if !(1..=6).contains(&threads) || !args.is_empty() {
        return Err("unknown arguments or threads outside 1..6".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    let cfg: FirstOrderConfig = serde_json::from_slice(&std::fs::read(&config_path)?)?;
    cfg.validate()?;
    if out.exists() {
        return Err("out-dir must be new; existing render evidence is never overwritten".into());
    }
    let mut params = RenderParams::new(input.clone());
    params.cache = cache;
    params.timestep = timestep;
    params.satellite = satellite;
    params.view = ViewMode::Geostationary;
    params.resolution = ResolutionMode::Native;
    params.geo_navigation = GeoNavigation::GoesRAbiFixedGrid;
    params.intent = RenderIntent::SensorFastGray;
    params.spectral_surface = Some(surface.clone());
    let start = std::time::Instant::now();
    let (frame, raster, time) = api::render_abi_first_order(&params, &cfg)?;
    std::fs::create_dir_all(&out)?;
    plane(
        &out.join("rho-c01-c02-c03.bin"),
        frame.reflectance.iter().copied(),
    )?;
    plane(
        &out.join("scatter-rho-c01-c02-c03.bin"),
        frame.scattered_reflectance.iter().copied(),
    )?;
    plane(
        &out.join("surface-rho-c01-c02-c03.bin"),
        frame.surface_reflectance.iter().copied(),
    )?;
    plane(&out.join("latitude.bin"), raster.lat.iter().copied())?;
    plane(&out.join("longitude.bin"), raster.lon.iter().copied())?;
    std::fs::write(out.join("support-flags.bin"), &frame.support_flags)?;
    std::fs::write(out.join("glint-mask.bin"), &frame.glint_mask)?;
    let (pw, ph) = (
        frame.nx.div_ceil(cfg.sample_stride),
        frame.ny.div_ceil(cfg.sample_stride),
    );
    for (b, name) in ["c01", "c02", "c03"].iter().enumerate() {
        plane(
            &out.join(format!("{name}-rho.bin")),
            frame.reflectance.chunks_exact(3).map(|p| p[b]),
        )?;
        let mut preview = Vec::with_capacity(pw * ph);
        for y in (0..frame.ny).step_by(cfg.sample_stride) {
            for x in (0..frame.nx).step_by(cfg.sample_stride) {
                let r = frame.reflectance[(y * frame.nx + x) * 3 + b];
                preview.push((r.clamp(0.0, 1.0).sqrt() * 255.0).round() as u8);
            }
        }
        image::GrayImage::from_raw(pw as u32, ph as u32, preview)
            .ok_or("invalid preview size")?
            .save(out.join(format!("{name}-first-order-preview.png")))?;
    }
    let mut files = BTreeMap::new();
    for p in std::fs::read_dir(&out)? {
        let p = p?.path();
        files.insert(
            p.file_name().unwrap().to_string_lossy().to_string(),
            serde_json::json!({"bytes":p.metadata()?.len(),"sha256":hash(&p)?}),
        );
    }
    let report = serde_json::json!({
        "operator":"simsat-abi-first-order-gray-cloud-v1","complete_abi_operator":false,
        "quantity":"TOA reflectance factor pi*L/E","channels":["c01","c02","c03"],
        "input":input,"input_sha256":hash(&input)?,"surface_manifest":surface,
        "surface_manifest_sha256":hash(&surface)?,
        "binary_sha256":hash(&std::env::current_exe()?)?,"config":cfg,"time":time,
        "width":frame.nx,"height":frame.ny,"rows":"north-first","layout":"row_column_c01_c02_c03_f32le",
        "preview":{"width":pw,"height":ph,"display":"clip [0,1], sqrt; no effect on raw values","sample_stride":cfg.sample_stride},
        "satellite":format!("{satellite:?}"),"navigation":"GOES-R ABI; spherical scene rays to model-grid points",
        "wavelength_um":frame.wavelength_um,"active_wavelengths":frame.active_wavelengths,
        "cloud_optical_depth_scale":1.0,"cloud_coverage":"full-cell",
        "sampled_pixels":frame.support_flags.iter().filter(|&&v|v&1!=0).count(),
        "view_exterior_pixels":frame.support_flags.iter().filter(|&&v|v&2!=0).count(),
        "sun_exterior_pixels":frame.support_flags.iter().filter(|&&v|v&4!=0).count(),
        "glint_definition":"water, illuminated, facet tan(beta)^2 <= model-wind mean square slope; one e-fold slope-PDF core",
        "limitations":[
            "First scattering order only. Missing atmospheric/cloud multiple scattering and diffuse surface illumination.",
            "Cloud extinction is unscaled but gray, conservative, with existing liquid/ice dual-HG phase; spectral particle modules not connected yet.",
            "Explicit exponential reference dry atmosphere, not WRF molecular profiles. No gases or aerosols.",
            "Lambertian approximation to HAMSTER black-sky albedo; no BRDF angular reconstruction or contemporary surface-state claim.",
            "Water is direct Cox-Munk only with existing n=1.34 and model wind; no water-leaving light, whitecaps or diffuse sky reflection.",
            "Target-height spherical boundary; no terrain cast shadows or model snow overlay.",
            "Point Sun, full-cell cloud coverage; no fractional overlap, polarization or instrument PSF.",
            "Exterior cloud volume is zero; support flags expose rays leaving the model domain.",
            "Spectral interpolation of computed normalized radiance requires convergence checks; not line-by-line RT.",
            "CPU scene reference; no GPU scene fallback or new display calibration."
        ],"files":files,"seconds":start.elapsed().as_secs_f64()
    });
    std::fs::write(out.join("render.json"), serde_json::to_vec_pretty(&report)?)?;
    println!("{report}");
    Ok(())
}
