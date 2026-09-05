//! Export independently integrated ABI C01/C02/C03 SURFACE albedos (not TOA imagery).
use simsat::{spectral_surface::SpectralSurface, visible_sensor::AbiReflectiveBand};
use std::{io::Write, path::PathBuf};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 3 {
        return Err("usage: audit_spectral_surface surface.json NEW_OUTPUT_DIRECTORY".into());
    }
    let source = PathBuf::from(&args[1]);
    let out = PathBuf::from(&args[2]);
    if out.exists() {
        return Err("output directory must be new".into());
    }
    let surface = SpectralSurface::load(&source)?;
    std::fs::create_dir_all(&out)?;
    let mut results = Vec::new();
    for (band, name) in AbiReflectiveBand::ALL
        .into_iter()
        .zip(["c01", "c02", "c03"])
    {
        let values = surface.band_albedo_grid(band)?;
        let path = out.join(format!("surface-albedo-{name}.bin"));
        let mut file = std::io::BufWriter::new(std::fs::File::create(&path)?);
        for v in &values {
            file.write_all(&v.to_le_bytes())?;
        }
        file.flush()?;
        let finite: Vec<_> = values.iter().copied().filter(|v| v.is_finite()).collect();
        let min = finite.iter().copied().fold(f32::INFINITY, f32::min);
        let max = finite.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        results.push(serde_json::json!({"band":name,"quantity":"solar_SRF_weighted_surface_black_sky_albedo","valid_land":finite.len(),"total":values.len(),"minimum":min,"maximum":max,"file":path.file_name().unwrap().to_string_lossy()}));
    }
    let report = serde_json::json!({"source":source,"climatology_doy":surface.climatology_doy(),"note":"These are surface albedos, not TOA ABI reflectance factors. No atmosphere, Sun-angle factor or cloud operator is included.","bands":results});
    std::fs::write(out.join("audit.json"), serde_json::to_vec_pretty(&report)?)?;
    println!("{report}");
    Ok(())
}
