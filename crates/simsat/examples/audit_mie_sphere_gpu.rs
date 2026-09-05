//! Execute the scalar Mie angular sum on an actual GPU using shared coefficients.
#[path = "support/compute_audit.rs"]
mod compute_audit;
use simsat::material_indices::{RefractiveIndex, VisibleMaterial};
use simsat::mie_sphere::MieSphere;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gpu = compute_audit::ComputeAudit::new()?;
    let mut spheres = Vec::new();
    for x in [
        1.0,
        2.0,
        std::f64::consts::PI,
        10.0,
        100.0,
        512.0,
        1024.0,
        2048.0,
    ] {
        for wavelength in [0.47, 0.64, 0.865] {
            let index = VisibleMaterial::LiquidWaterSegelstein1981.at(wavelength)?;
            spheres.push(MieSphere::new(index, x as f32 as f64)?);
        }
    }
    spheres.push(MieSphere::new(
        RefractiveIndex {
            real: 1.5,
            imaginary: 0.1,
        },
        2.0,
    )?);
    let mut cases = Vec::new();
    for (sphere, _) in spheres.iter().enumerate() {
        for angle in 0..=180 {
            cases.push((sphere, (angle as f64).to_radians().cos() as f32));
        }
        cases.extend([(sphere, -0.999999), (sphere, 0.999999)]);
    }
    // Two input vec4s per ray; remaining vec4s are shared sphere coefficients.
    let mut packed = vec![0.0f32; cases.len() * 8];
    let mut offsets = Vec::new();
    for sphere in &spheres {
        offsets.push(packed.len() / 4);
        for c in sphere.coefficients() {
            packed.extend_from_slice(&[
                c.electric_real as f32,
                c.electric_imaginary as f32,
                c.magnetic_real as f32,
                c.magnetic_imaginary as f32,
            ]);
        }
    }
    for (i, (sphere, mu)) in cases.iter().enumerate() {
        let s = &spheres[*sphere];
        packed[8 * i..8 * i + 8].copy_from_slice(&[
            *mu,
            offsets[*sphere] as f32,
            s.coefficients().len() as f32,
            0.0,
            s.size_parameter() as f32,
            s.efficiencies().scattering as f32,
            0.0,
            0.0,
        ]);
    }
    let source = format!(
        "{}\n{}",
        include_str!("../src/gpu/shaders/mie_sphere.wgsl"),
        r#"
@group(0) @binding(0) var<storage,read> inputs: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> outputs: array<vec4<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    let i=gid.x;
    if (i>=arrayLength(&outputs)) {return;}
    let c=inputs[2u*i];let physical=inputs[2u*i+1u];
    let offset=u32(c.y);let orders=u32(c.z);
    var state=mie_phase_begin();
    for (var order=1u;order<=orders;order+=1u) {
        state=mie_phase_step(state,inputs[offset+order-1u],order,c.x);
    }
    outputs[i]=vec4<f32>(mie_phase_finish(state,physical.x,physical.y),0.0,0.0,0.0);
}
"#
    );
    let got = gpu.run(source, &packed, cases.len())?;
    let (absolute_tolerance, relative_tolerance) = (1e-7, 2e-3);
    let mut max_absolute = 0.0f64;
    let mut max_relative = 0.0f64;
    let mut failures = Vec::new();
    for ((sphere, mu), actual) in cases.iter().zip(got) {
        let expected = spheres[*sphere].phase_sr1(*mu as f64)?;
        let value = actual[0] as f64;
        let difference = (value - expected).abs();
        if !value.is_finite()
            || value < 0.0
            || difference > absolute_tolerance + relative_tolerance * expected
        {
            failures.push(
                serde_json::json!({"sphere":sphere,"mu":mu,"expected":expected,"actual":value}),
            );
        }
        max_absolute = max_absolute.max(difference);
        if expected > 1e-7 {
            max_relative = max_relative.max(difference / expected);
        }
    }
    let passed = failures.is_empty();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "check":"scalar-mie-angular-cpu-wgsl","adapter":gpu.info.name,"backend":format!("{:?}",gpu.info.backend),"driver":gpu.info.driver,"driver_info":gpu.info.driver_info,
            "spheres":spheres.len(),"cases":cases.len(),"size_parameter_range":[1,2048],"max_absolute_error_sr1":max_absolute,"max_relative_error_when_phase_gt_1e_minus7_sr1":max_relative,
            "absolute_tolerance_sr1":absolute_tolerance,"relative_tolerance":relative_tolerance,"failures":failures,"passed":passed,
            "contract":"shared CPU-prepared sphere coefficients; GPU evaluates angular amplitude sum and normalized scalar phase; not an independent GPU Bessel solver"
        }))?
    );
    if !passed {
        return Err("Mie angular CPU/WGSL tolerance exceeded".into());
    }
    Ok(())
}
