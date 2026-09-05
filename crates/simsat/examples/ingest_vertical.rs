//! Explicit vertical-resolution convergence experiment from native WRF fields.
//! Usage: cargo run --release -p simsat --example ingest_vertical -- INPUT NEW_CACHE DZ_M NZ
//! Render the emitted run.json with the ordinary CLI. No horizontal crop, optics
//! change, or source mutation. A NEW cache directory is mandatory because the
//! historical default cache key does not encode custom vertical geometry.
//! Finer grids can exceed the default 2.5 GB memory budget; this is opt-in tooling.
use simsat::ingest::{IngestConfig, ingest_timestep};
use std::path::PathBuf;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err("usage: ingest_vertical INPUT NEW_CACHE DZ_M NZ".into());
    }
    let input = PathBuf::from(&args[0]);
    let cache = PathBuf::from(&args[1]);
    let dz: f64 = args[2].parse()?;
    let nz: usize = args[3].parse()?;
    if !input.is_file() || !dz.is_finite() || dz < 1.0 || nz < 2 || dz * (nz - 1) as f64 > 100_000.0
    {
        return Err("input must exist; require dz >= 1 m, nz >= 2 and top <= 100 km".into());
    }
    // create_dir refuses an existing destination atomically: no stale-cache reuse.
    std::fs::create_dir(&cache)?;
    let mut config = IngestConfig::new(cache);
    config.dz_m = dz;
    config.nz_brick = nz;
    simsat::topdown::configure_global_rayon(Some(6));
    let result = ingest_timestep(&input, &config)?;
    println!(
        "manifest={} dims={}x{}x{} dz_m={} top_m={} peak_rss_bytes={:?}",
        result.manifest_path.display(),
        result.nx,
        result.ny,
        result.nz,
        dz,
        dz * (nz - 1) as f64,
        result.peak_rss_bytes
    );
    Ok(())
}
