//! Numerical audit of material-index interpolation on an actual GPU.
#[path = "support/compute_audit.rs"]
mod compute_audit;
use simsat::material_indices::VisibleMaterial;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gpu = compute_audit::ComputeAudit::new()?;
    let mut cases = Vec::new();
    let mut packed = Vec::new();
    for material in [
        VisibleMaterial::LiquidWaterSegelstein1981,
        VisibleMaterial::IceIhWarrenBrandt2008,
    ] {
        for k in 0..=1200 {
            let wavelength = 0.4f32 + 0.0005f32 * k as f32;
            let wavelength = wavelength.min(1.0);
            let [a, b] = material.bracket(wavelength as f64)?;
            cases.push((material, wavelength));
            packed.extend_from_slice(&[
                wavelength,
                0.0,
                0.0,
                0.0,
                a.wavelength_um as f32,
                a.index.real as f32,
                a.index.imaginary as f32,
                0.0,
                b.wavelength_um as f32,
                b.index.real as f32,
                b.index.imaginary as f32,
                0.0,
            ]);
        }
    }
    let source = format!(
        "{}\n{}",
        include_str!("../src/gpu/shaders/material_indices.wgsl"),
        r#"
@group(0) @binding(0) var<storage,read> inputs: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> outputs: array<vec4<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    let i=gid.x;
    if (i>=arrayLength(&outputs)) {return;}
    let n=visible_material_index(inputs[3u*i].x,inputs[3u*i+1u].xyz,inputs[3u*i+2u].xyz);
    outputs[i]=vec4<f32>(n.x,n.y*1e10,0.0,0.0);
}
"#
    );
    let got = gpu.run(source, &packed, cases.len())?;
    let mut max_relative = [0.0f64; 2];
    for ((material, wavelength), g) in cases.iter().zip(got) {
        let n = material.at(*wavelength as f64)?;
        for (k, expected) in [n.real, n.imaginary * 1e10].into_iter().enumerate() {
            if !g[k].is_finite() || g[k] <= 0.0 {
                return Err("invalid GPU material index".into());
            }
            max_relative[k] = max_relative[k].max((g[k] as f64 / expected - 1.0).abs());
        }
    }
    let tolerance = 2e-5;
    let passed = max_relative.iter().all(|v| *v <= tolerance);
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "check":"visible-material-index-cpu-wgsl","adapter":gpu.info.name,"backend":format!("{:?}",gpu.info.backend),
            "driver":gpu.info.driver,"driver_info":gpu.info.driver_info,"cases":cases.len(),"wavelength_um":[0.4,1.0],
            "materials":["liquid-water-segelstein-1981","ice-ih-warren-brandt-2008"],
            "max_relative_error":{"real_index":max_relative[0],"imaginary_index":max_relative[1]},
            "tolerance":tolerance,"passed":passed,"contract":"host selects exact native material knots; GPU performs the same log-wavelength interpolation"
        }))?
    );
    if !passed {
        return Err("material CPU/WGSL tolerance exceeded".into());
    }
    Ok(())
}
