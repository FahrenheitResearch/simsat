//! Verify a measured display surface against actual native WRF coordinates.
use std::path::Path;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: audit_nbar_surface surface.json wrfout".into());
    }
    let s = simsat::nbar_surface::NbarSurface::load(Path::new(&args[1]))?;
    let g = simsat::ingest::read_grid_geometry(Path::new(&args[2]), 0)?;
    let (nx, ny) = (g.nx, g.ny);
    let r = simsat::camera::SurfaceRaster {
        nx,
        ny,
        scan: simsat::camera::ScanGrid {
            nx,
            ny,
            x_min: 0.0,
            y_max: 0.0,
            pitch_x: 1.0,
            pitch_y: 1.0,
        },
        lat: vec![0.0; nx * ny],
        lon: vec![0.0; nx * ny],
        grid_i: (0..nx * ny).map(|p| (p % nx) as f32).collect(),
        grid_j: (0..nx * ny).map(|p| (p / nx) as f32).collect(),
        model_scan: None,
        navigation_geometry: None,
    };
    // Audit registration independently of a brick. Reconstruct the source land
    // mask's native order by exact coordinate identity, then perturb one coordinate.
    let h: serde_json::Value = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let directory = Path::new(&args[1]).parent().ok_or("no parent")?;
    let c = std::fs::read(directory.join("coordinates.bin"))?;
    let land = std::fs::read(directory.join("land-mask.bin"))?;
    let lookup: std::collections::BTreeMap<_, _> = g
        .xlat
        .iter()
        .zip(&g.xlong)
        .enumerate()
        .map(|(i, (&lat, &lon))| (((lat as f64).to_bits(), (lon as f64).to_bits()), i))
        .collect();
    let mut native_land = vec![0.0; nx * ny];
    for (cell, bytes) in c.chunks_exact(16).enumerate() {
        let key = (
            u64::from_le_bytes(bytes[..8].try_into().unwrap()),
            u64::from_le_bytes(bytes[8..].try_into().unwrap()),
        );
        native_land[*lookup.get(&key).ok_or("coordinate differs from WRF")?] = land[cell] as f32;
    }
    let (_, error) = s.raster_rgba(&g, &native_land, &r)?;
    let mut wrong = g.clone();
    wrong.xlat[0] += 0.0001;
    assert!(s.raster_rgba(&wrong, &native_land, &r).is_err());
    native_land[0] = 1.0 - native_land[0];
    assert!(s.raster_rgba(&g, &native_land, &r).is_err());
    println!(
        "{}",
        serde_json::json!({"grid":[nx,ny],"coordinate_matches":nx*ny,
        "max_ideal_projection_rounding_cells":error,"shifted_coordinate_rejected":true,
        "land_mask_mismatch_rejected":true,"source_date":h["source_date"],
        "full":s.full_count,"magnitude":s.magnitude_count,"fallback":s.fallback_count})
    );
    Ok(())
}
