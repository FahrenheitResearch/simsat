//! Execute molecular WGSL and compare with the independently tested f64 CPU
//! physics. GPU unavailability is an error, never a silently skipped success.
#[path = "support/compute_audit.rs"]
mod compute_audit;
use simsat::spectral_molecular::DryAirRayleigh;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gpu = compute_audit::ComputeAudit::new()?;
    let mut cases = Vec::<[f32; 4]>::new();
    for k in 0..=150 {
        for co2 in [0.0, 300.0, 360.0, 1000.0] {
            for mu in [-1.0, -0.5, 0.0, 0.5, 1.0] {
                cases.push([0.25 + 0.005 * k as f32, co2, mu, 2.0e29]);
            }
        }
    }
    let source = format!(
        "{}\n{}",
        include_str!("../src/gpu/shaders/spectral_molecular.wgsl"),
        r#"
@group(0) @binding(0) var<storage,read> cases: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> results: array<vec4<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x>=arrayLength(&cases)) {return;}
    let c=cases[gid.x];let o=dry_air_rayleigh(c.x,c.y);
    results[gid.x]=vec4<f32>(o.cross_section_m2*1e31,o.king_factor,
        dry_air_rayleigh_phase(o,c.z),dry_air_rayleigh_optical_depth(o,c.w));
}
"#
    );
    let got = gpu.run(
        source,
        &cases.iter().flatten().copied().collect::<Vec<_>>(),
        cases.len(),
    )?;
    let mut max_relative = [0.0f64; 4];
    for (c, g) in cases.iter().zip(got) {
        let o = DryAirRayleigh::new(c[0] as f64, c[1] as f64)?;
        let expected = [
            o.cross_section_m2() * 1e31,
            o.king_factor(),
            o.phase_sr1(c[2] as f64)?,
            o.optical_depth(c[3] as f64)?,
        ];
        for k in 0..4 {
            if !g[k].is_finite() {
                return Err("non-finite GPU molecular result".into());
            }
            max_relative[k] = max_relative[k].max((g[k] as f64 / expected[k] - 1.0).abs());
        }
    }
    let tolerance = 2e-5;
    let passed = max_relative.iter().all(|v| *v <= tolerance);
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "check":"spectral-molecular-cpu-wgsl","adapter":gpu.info.name,"backend":format!("{:?}",gpu.info.backend),"driver":gpu.info.driver,"driver_info":gpu.info.driver_info,
            "cases":cases.len(),"wavelength_um":[0.25,1.0],"co2_ppm":[0,300,360,1000],"scattering_cosines":[-1,-0.5,0,0.5,1],
            "max_relative_error":{"cross_section":max_relative[0],"king_factor":max_relative[1],"phase":max_relative[2],"optical_depth":max_relative[3]},"tolerance":tolerance,"passed":passed
        }))?
    );
    if !passed {
        return Err("molecular CPU/WGSL tolerance exceeded".into());
    }
    Ok(())
}
