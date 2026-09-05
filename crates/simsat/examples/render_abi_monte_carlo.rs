//! All-order scalar Monte Carlo spectral ABI reference; independent of shipped display.
//! Required: input= cache= spectral-surface= config= out-dir=
//! Optional: timestep=0 sat=goes-east|goes-west threads=6
//! Config JSON declares all reference-atmosphere and numerical assumptions.
//! Output C01/C02/C03 are pi*L/E, not display RGB and not solar-zenith normalized.
use sha2::{Digest, Sha256};
use simsat::{
    abi_monte_carlo::MonteCarloConfig,
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
        _ => return Err("Monte Carlo ABI requires goes-east or goes-west".into()),
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
    let cfg: MonteCarloConfig = serde_json::from_slice(&std::fs::read(&config_path)?)?;
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
    let (frame, raster, time) = api::render_abi_monte_carlo(&params, &cfg)?;
    std::fs::create_dir_all(&out)?;
    plane(
        &out.join("rho-c01-c02-c03.bin"),
        frame.reflectance.iter().copied(),
    )?;
    plane(
        &out.join("first-order-rho-c01-c02-c03.bin"),
        frame.first_order_reflectance.iter().copied(),
    )?;
    plane(
        &out.join("higher-order-rho-c01-c02-c03.bin"),
        frame.higher_order_reflectance.iter().copied(),
    )?;
    for (name, values) in [
        ("standard-error-rho.bin", &frame.standard_error),
        (
            "first-order-standard-error-rho.bin",
            &frame.first_order_standard_error,
        ),
        (
            "higher-order-standard-error-rho.bin",
            &frame.higher_order_standard_error,
        ),
        ("mean-events.bin", &frame.mean_events),
        ("path-exterior-fraction.bin", &frame.path_exterior_fraction),
        ("sun-exterior-fraction.bin", &frame.sun_exterior_fraction),
        (
            "surface-exterior-fraction.bin",
            &frame.surface_exterior_fraction,
        ),
    ] {
        plane(&out.join(name), values.iter().copied())?;
    }
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
            .save(out.join(format!("{name}-monte-carlo-preview.png")))?;
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
        "operator":"simsat-abi-monte-carlo-gray-cloud-v1","complete_abi_operator":false,
        "quantity":"TOA reflectance factor pi*L/E","channels":["c01","c02","c03"],
        "input":input,"input_sha256":hash(&input)?,"surface_manifest":surface,
        "surface_manifest_sha256":hash(&surface)?,"binary_sha256":hash(&std::env::current_exe()?)?,
        "config":cfg,"time":time,"width":frame.nx,"height":frame.ny,
        "rows":"north-first","layout":"row_column_c01_c02_c03_f32le",
        "preview":{"width":pw,"height":ph,"display":"clip [0,1], sqrt; raw and uncertainty unchanged","sample_stride":cfg.sample_stride},
        "satellite":format!("{satellite:?}"),"navigation":"GOES-R ABI; spherical scene rays to model-grid points",
        "wavelength_sampling":"Independent wavelength per photon, sampled from all official NOAA FM4 / TSIS-1 HSRS quadrature weights. No transfer-grid interpolation.",
        "uncertainty":"Empirical per-band standard error of independent photon-path means; rare scattering events can be undersampled, so this is not a guaranteed confidence bound. Includes path and wavelength sampling only, not numerical integration, boundary, model, or omitted-physics errors.",
        "first_order":"Mean of source terms at the first interaction; surface direct and atmospheric single scattering.",
        "higher_order":"Mean of source terms after two or more interactions; includes atmospheric/cloud multiple scattering and diffuse surface reflections.",
        "cloud_optical_depth_scale":1.0,"cloud_coverage":"full-cell",
        "sampled_pixels":frame.support_flags.iter().filter(|&&v|v&1!=0).count(),
        "support_flags":"bit0 sampled; bit1 any view/secondary path outside cloud volume; bit2 any Sun path outside cloud volume; bit3 surface interaction outside measured coverage. Per-path fractions are saved separately.",
        "path_exterior_pixels":frame.support_flags.iter().filter(|&&v|v&2!=0).count(),
        "sun_exterior_pixels":frame.support_flags.iter().filter(|&&v|v&4!=0).count(),
        "surface_exterior_pixels":frame.support_flags.iter().filter(|&&v|v&8!=0).count(),
        "glint_definition":"Primary model-water footprint, illuminated, facet tan(beta)^2 <= model-wind mean square slope.",
        "limitations":[
            "All scattering orders for this scalar reference, but gray conservative model clouds and inherited dual-HG phase; spectral particle modules are not connected.",
            "Explicit exponential reference dry atmosphere, not native WRF molecular profiles. No gases or aerosols.",
            "HAMSTER black-sky spectral albedo used as a Lambertian land approximation. No directional land BRDF or contemporary surface-state claim.",
            "Inherited isotropic Cox-Munk BRDF with n=1.34 and model wind; diffuse reflected sky is now included, but shadowing/masking, whitecaps and water-leaving light remain missing.",
            "One target-height sphere per primary pixel; no full terrain surface or terrain cast shadows, no model snow overlay.",
            "Point Sun, full-cell cloud coverage; no fractional overlap, polarization or instrument PSF.",
            "Exterior cloud is zero; surface beyond measured coverage absorbs. Flags and path fractions disclose these incomplete boundary conditions.",
            "Free paths and direct Sun optical depths use explicit finite-step volume reconstruction and spherical air-column quadrature; numerical convergence remains separate from photon standard errors.",
            "CPU 3D scene reference; sampling and all-order slab GPU counterparts are audited separately. Full-scene GPU traversal is not implemented."
        ],"files":files,"seconds":start.elapsed().as_secs_f64()
    });
    std::fs::write(out.join("render.json"), serde_json::to_vec_pretty(&report)?)?;
    println!("{report}");
    Ok(())
}
