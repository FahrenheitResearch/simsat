//! Run the first-order transport WGSL against the CPU RTE kernel, including
//! thin, opaque, and opposing sun/view paths. Requires a real GPU adapter.
#[path = "support/compute_audit.rs"]
mod compute_audit;
use simsat::spectral_transport::{
    DirectLambertianBoundary, SingleScatterSegment, SolarDepthEndpoints, integrate_single_scatter,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gpu = compute_audit::ComputeAudit::new()?;
    let mut cases = Vec::<[f32; 8]>::new();
    for ext in [0.0, 1e-8, 1e-4, 0.001, 0.02, 0.05, 0.1, 1.0, 10.0, 1000.0] {
        for foreground in [0.0, 0.2, 20.0] {
            for sn in [0.0, 0.01, 5.0, 50.0, 10000.0] {
                for sf in [0.0, 0.1, 5.0, 10000.0] {
                    cases.push([foreground, ext, 0.7 * ext, 0.13, sn, sf, 0.2, 0.4]);
                }
            }
        }
    }
    let source = format!(
        "{}\n{}",
        include_str!("../src/gpu/shaders/spectral_transport.wgsl"),
        r#"
struct Case { optics:vec4<f32>,sun_surface:vec4<f32> };
@group(0) @binding(0) var<storage,read> cases: array<Case>;
@group(0) @binding(1) var<storage,read_write> results: array<vec4<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    if(gid.x>=arrayLength(&cases)){return;}
    let a=cases[gid.x].optics;let b=cases[gid.x].sun_surface;
    results[gid.x]=vec4<f32>(spectral_single_scatter_segment(a.x,a.y,a.z,a.w,b.x,b.y),
        spectral_direct_lambertian(b.z,b.w,b.y,a.x+a.y),exp(-a.x-a.y),0.0);
}
"#
    );
    let got = gpu.run(
        source,
        &cases.iter().flatten().copied().collect::<Vec<_>>(),
        cases.len(),
    )?;
    let mut max_abs = [0.0f64; 3];
    let mut max_rel = [0.0f64; 3];
    let mut passed = true;
    let atol = 1e-8;
    let rtol = 2e-5;
    for (c, g) in cases.iter().zip(got) {
        let c = c.map(f64::from);
        let front = SingleScatterSegment::new(c[0], 0.0, 0.0, None)?;
        let s = SingleScatterSegment::new(
            c[1],
            c[2],
            c[3],
            Some(SolarDepthEndpoints::new(c[4], c[5])?),
        )?;
        let b = DirectLambertianBoundary::new(c[6], c[7], Some(c[5]))?;
        let expected = integrate_single_scatter([front, s], Some(b))?;
        let values = [
            expected.scattered_normalized_radiance_sr1,
            expected.surface_normalized_radiance_sr1,
            expected.view_transmittance,
        ];
        for k in 0..3 {
            if !g[k].is_finite() {
                return Err("non-finite GPU transport result".into());
            }
            let error = (g[k] as f64 - values[k]).abs();
            max_abs[k] = max_abs[k].max(error);
            if values[k] > 1e-8 {
                max_rel[k] = max_rel[k].max(error / values[k]);
            }
            passed &= error <= atol + rtol * values[k].abs();
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "check":"spectral-first-order-transport-cpu-wgsl","adapter":gpu.info.name,"backend":format!("{:?}",gpu.info.backend),"driver":gpu.info.driver,"driver_info":gpu.info.driver_info,
            "cases":cases.len(),"outputs":["scattered_L_over_E_sr1","surface_L_over_E_sr1","view_transmittance"],
            "max_absolute_error":max_abs,"max_relative_error_where_reference_above_1e-8":max_rel,
            "absolute_tolerance":atol,"relative_tolerance":rtol,"passed":passed,
            "limitations":"Tests direct first-order kernels; not full image or multiple-scattering parity."
        }))?
    );
    if !passed {
        return Err("transport CPU/WGSL tolerance exceeded".into());
    }
    Ok(())
}
