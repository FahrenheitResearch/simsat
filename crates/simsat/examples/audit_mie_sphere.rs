//! Independent BHMIE/SciPy requests and spectral sphere reference responses.
use simsat::material_indices::{RefractiveIndex, VisibleMaterial};
use simsat::mie_sphere::MieSphere;
use std::f64::consts::PI;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut inputs = Vec::new();
    for wavelength in [0.442067, 0.47, 0.64, 0.865, 0.903751] {
        let index = VisibleMaterial::LiquidWaterSegelstein1981.at(wavelength)?;
        for radius in [0.25, 0.5, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0] {
            inputs.push(("liquid-water", index, 2.0 * PI * radius / wavelength));
        }
    }
    for (real, imaginary) in [(1.01, 0.0), (1.33, 0.0), (1.5, 0.05), (2.0, 0.01)] {
        for x in [1.0, 2.0, 10.0, 100.0, 512.0, 1024.0, 2048.0] {
            inputs.push(("generic-sphere", RefractiveIndex { real, imaginary }, x));
        }
    }
    let angles = [
        0.0f64, 0.1, 1.0, 10.0, 30.0, 60.0, 90.0, 120.0, 150.0, 170.0, 179.0, 179.9, 180.0,
    ];
    let mut cases = Vec::new();
    for (label, index, x) in inputs {
        // BHMIE's public interface takes f32. Quantize each actual request
        // identically before solving with our f64 implementation.
        let index = RefractiveIndex {
            real: index.real as f32 as f64,
            imaginary: index.imaginary as f32 as f64,
        };
        let x = x as f32 as f64;
        let sphere = MieSphere::new(index, x)?;
        let q = sphere.efficiencies();
        let phase=angles.iter().map(|degrees|Ok(serde_json::json!({"degrees":degrees,"phase_sr1":sphere.phase_sr1(degrees.to_radians().cos())?}))).collect::<Result<Vec<_>,Box<dyn std::error::Error>>>()?;
        cases.push(serde_json::json!({"id":cases.len(),"label":label,"x":x,"n_real":index.real,"n_imag_positive":index.imaginary,
            "orders":sphere.coefficients().len(),"extinction_efficiency":q.extinction,"scattering_efficiency":q.scattering,"absorption_efficiency":q.absorption,"asymmetry":q.asymmetry,"phase":phase}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"schema":"simsat-mie-sphere-reference-audit-v1","cases":cases,
        "input_contract":"BHMIE-compatible f32-rounded relative index and size parameter; all calculations in SimSat are f64",
        "phase_normalization":"unpolarized phase sr^-1, integral over 4pi equals one","limitations":["homogeneous sphere only","no particle-size-distribution averaging","not a nonspherical ice model","reference-condition water material; not temperature-resolved supercooled optics"]})
        )?
    );
    Ok(())
}
