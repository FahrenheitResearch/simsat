//! Audit current compact SSB precision against the post-resample f32 WRF fields.
//!
//! ```text
//! cargo run --profile release-fast -p simsat --example ssb_precision_audit -- \
//!   input=C:\path\wrfout_d02_... out=C:\path\audit-output
//! ```
//!
//! This deliberately re-ingests the requested timestep into an isolated cache so
//! the audit sees the f32 fields immediately before quantization. It does not alter
//! the SSB format or any default render setting.

use std::path::PathBuf;

use simsat::ingest::{IngestConfig, ingest_timestep_with_precision_audit};
use simsat::optics::CloudOpticsMode;
use simsat::precision_audit::PrecisionAuditConfig;

fn main() {
    if let Err(error) = run() {
        eprintln!("precision audit failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut input = None;
    let mut output = None;
    let mut timestep = 0usize;
    let mut cloud_optics = CloudOpticsMode::Fixed;
    for argument in std::env::args().skip(1) {
        let Some((key, value)) = argument.split_once('=') else {
            return Err(format!("expected key=value argument, got {argument:?}"));
        };
        match key.trim().to_ascii_lowercase().as_str() {
            "input" => input = Some(PathBuf::from(value)),
            "out" | "output" => output = Some(PathBuf::from(value)),
            "timestep" | "time-index" => {
                timestep = value
                    .parse()
                    .map_err(|_| format!("invalid timestep {value:?}"))?;
            }
            "cloud-optics" => {
                cloud_optics = CloudOpticsMode::parse(value)
                    .ok_or_else(|| format!("invalid cloud-optics {value:?}"))?;
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    let input = input.ok_or_else(|| {
        "missing input=<wrfout>; also provide out=<precision-audit-directory>".to_string()
    })?;
    let output = output.ok_or_else(|| "missing out=<precision-audit-directory>".to_string())?;
    if !input.is_file() {
        return Err(format!("input is not a file: {}", input.display()));
    }

    let mut config = IngestConfig::new(output.join("cache"));
    config.run_id = Some("precision-audit-reference".to_string());
    config.timestep = timestep;
    config.cloud_optics = cloud_optics;
    let audit = PrecisionAuditConfig::new(output.clone());

    let report =
        ingest_timestep_with_precision_audit(&input, &config, &audit).map_err(|e| e.to_string())?;
    println!(
        "PRECISION_AUDIT report={} brick={} dims={}x{}x{} ingest_s={:.3} ssb_bytes={}",
        output.join("report.json").display(),
        report.brick_path.display(),
        report.nx,
        report.ny,
        report.nz,
        report.wall.as_secs_f64(),
        report.ssb_bytes,
    );
    Ok(())
}
