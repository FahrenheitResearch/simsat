//! Headless HRRR/RRFS GRIB2 -> `.ssb` ingest harness (QA / node use).
//!
//! Ingests ONE GRIB2 file into the brick cache and prints a machine-readable
//! `GRIBINGEST` line. The written run renders through the EXISTING harnesses by
//! pointing them at the produced `run.json` (the cached-run path):
//!
//! ```text
//! cargo run --release -p simsat --example ingest_grib_file -- \
//!     input=/path/hrrr.t20z.wrfnatf00.grib2 cache=/path/cache
//! cargo run --release -p simsat --example render_frame -- \
//!     input=/path/cache/<run_id>/run.json out=frame.png ...
//! ```
//!
//! key=value args: `input` (the .grib2 file, required), `cache` (cache root,
//! default the engine cache dir), `run-id` (override; default embeds the cycle
//! date — see `ingest_grib::default_grib_run_id`), `crop` (`conus` or
//! `lat_min,lat_max,lon_min,lon_max` — REQUIRED for oversize grids like the
//! RRFS NA rotated grid; subject to the peak-RSS admission gate).

use std::path::PathBuf;

use simsat::bricks::StorageProfile;
use simsat::ingest::{IngestConfig, default_cache_dir};
use simsat::ingest_grib::{self, GribIngestOptions, ingest_grib_timestep_with, parse_crop};

fn main() {
    let mut input: Option<PathBuf> = None;
    let mut cache: Option<PathBuf> = None;
    let mut run_id: Option<String> = None;
    let mut storage_profile = StorageProfile::CompactU8;
    let mut options = GribIngestOptions::default();
    for arg in std::env::args().skip(1) {
        let Some((key, value)) = arg.split_once('=') else {
            eprintln!("unrecognized arg (expected key=value): {arg}");
            std::process::exit(2);
        };
        match key {
            "input" => input = Some(PathBuf::from(value)),
            "cache" => cache = Some(PathBuf::from(value)),
            "run-id" => run_id = Some(value.to_string()),
            "storage-profile" => {
                storage_profile = StorageProfile::parse(value).unwrap_or_else(|| {
                    eprintln!("bad storage-profile '{value}' (compact-u8|science-cloud-f16)");
                    std::process::exit(2);
                });
            }
            "crop" => match parse_crop(value) {
                Ok(crop) => options.crop = Some(crop),
                Err(e) => {
                    eprintln!("bad crop: {e}");
                    std::process::exit(2);
                }
            },
            _ => {
                eprintln!("unknown key: {key}");
                std::process::exit(2);
            }
        }
    }
    let Some(input) = input else {
        eprintln!(
            "usage: ingest_grib_file input=<file.grib2> [cache=<dir>] [run-id=<id>] \
             [crop=conus|lat_min,lat_max,lon_min,lon_max] \
             [storage-profile=compact-u8|science-cloud-f16]"
        );
        std::process::exit(2);
    };
    if !ingest_grib::is_grib_input(&input) {
        eprintln!(
            "input does not look like GRIB2 (extension or magic): {}",
            input.display()
        );
        std::process::exit(2);
    }

    let mut config = IngestConfig::new(cache.unwrap_or_else(default_cache_dir));
    config.run_id = run_id;
    config.storage_profile = storage_profile;

    match ingest_grib_timestep_with(&input, &config, &options) {
        Ok(report) => {
            println!(
                "GRIBINGEST run={} dims={}x{}x{} hhmm={:04} wall={:.2}s peak_rss_mb={} \
                 ssb_mb={:.2} manifest={}",
                report.run_id,
                report.nx,
                report.ny,
                report.nz,
                report.hhmm,
                report.wall.as_secs_f64(),
                report
                    .peak_rss_bytes
                    .map(|b| format!("{:.1}", b as f64 / (1024.0 * 1024.0)))
                    .unwrap_or_else(|| "n/a".to_string()),
                report.ssb_bytes as f64 / (1024.0 * 1024.0),
                report.manifest_path.display(),
            );
        }
        Err(e) => {
            eprintln!("ingest failed: {e}");
            std::process::exit(1);
        }
    }
}
