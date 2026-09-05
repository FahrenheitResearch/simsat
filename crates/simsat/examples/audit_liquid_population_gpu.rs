//! Actual GPU audit of shared droplet-population angular and mass coefficients.
#[path = "support/compute_audit.rs"]
mod compute_audit;
use simsat::liquid_population::{LiquidPopulationOptics, ParticleNumberNode};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gpu = compute_audit::ComputeAudit::new()?;
    let distributions: Vec<Vec<ParticleNumberNode>> = [
        vec![(2.5, 1.0)],
        vec![(10.0, 1.0)],
        vec![(5.0, 3.0), (10.0, 2.0), (30.0, 1.0)],
        vec![
            (0.25, 3.0),
            (0.5, 2.0),
            (2.0, 10.0),
            (5.0, 20.0),
            (10.0, 30.0),
            (25.0, 5.0),
            (50.0, 0.2),
            (100.0, 0.001),
        ],
    ]
    .into_iter()
    .map(|d| {
        d.into_iter()
            .map(|(r, w)| ParticleNumberNode {
                radius_m: r * 1e-6,
                number_weight: w,
            })
            .collect()
    })
    .collect();
    let mut populations = Vec::new();
    for lambda in [0.442067, 0.47, 0.64, 0.865, 0.903751] {
        for nodes in &distributions {
            populations.push(LiquidPopulationOptics::new(lambda, 1.00028, nodes)?);
        }
    }
    let mut cases = Vec::new();
    for p in 0..populations.len() {
        for degrees in 0..=180 {
            cases.push((
                p,
                (degrees as f64).to_radians().cos() as f32,
                10f32.powi(-6 + degrees % 5),
            ));
        }
        cases.extend([(p, -0.999999, 0.001), (p, 0.999999, 0.001)]);
    }
    let mut packed = vec![0f32; cases.len() * 8];
    let mut offsets = Vec::new();
    for pop in &populations {
        let meta = packed.len() / 4;
        offsets.push(meta);
        packed.resize(packed.len() + pop.components().len() * 8, 0.0);
        for (j, (sphere, weight)) in pop.components().iter().enumerate() {
            let offset = packed.len() / 4;
            for c in sphere.coefficients() {
                packed.extend_from_slice(&[
                    c.electric_real as f32,
                    c.electric_imaginary as f32,
                    c.magnetic_real as f32,
                    c.magnetic_imaginary as f32,
                ]);
            }
            packed[meta * 4 + j * 8..meta * 4 + j * 8 + 8].copy_from_slice(&[
                offset as f32,
                sphere.coefficients().len() as f32,
                sphere.size_parameter() as f32,
                sphere.efficiencies().scattering as f32,
                *weight as f32,
                0.0,
                0.0,
                0.0,
            ]);
        }
    }
    for (i, (p, mu, mass)) in cases.iter().enumerate() {
        let bulk = populations[*p].bulk();
        packed[i * 8..i * 8 + 8].copy_from_slice(&[
            *mu,
            offsets[*p] as f32,
            populations[*p].components().len() as f32,
            *mass,
            bulk.mass_extinction_m2_kg as f32,
            bulk.mass_scattering_m2_kg as f32,
            bulk.mass_absorption_m2_kg as f32,
            0.0,
        ]);
    }
    let source = format!(
        "{}\n{}\n{}",
        include_str!("../src/gpu/shaders/mie_sphere.wgsl"),
        include_str!("../src/gpu/shaders/liquid_population.wgsl"),
        r#"
@group(0) @binding(0) var<storage,read> inputs:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> outputs:array<vec4<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    let i=gid.x;
    if(i>=arrayLength(&outputs)){return;}
    let c=inputs[2u*i]; let k=inputs[2u*i+1u];
    var phase=0.0;
    for(var j=0u;j<u32(c.z);j+=1u) {
        let m=inputs[u32(c.y)+2u*j];
        let weight=inputs[u32(c.y)+2u*j+1u].x;
        var state=mie_phase_begin();
        for(var order=1u;order<=u32(m.y);order+=1u) {
            state=mie_phase_step(state,inputs[u32(m.x)+order-1u],order,c.x);
        }
        phase=liquid_population_phase_add(phase,mie_phase_finish(state,m.z,m.w),weight);
    }
    outputs[i]=vec4<f32>(phase,liquid_population_volume_coefficients(k.xyz,c.w));
}
"#
    );
    let actual = gpu.run(source, &packed, cases.len())?;
    let mut failures = Vec::new();
    let mut max_phase_relative = 0.0f64;
    let mut max_coefficient_relative = 0.0f64;
    for ((p, mu, mass), got) in cases.iter().zip(actual) {
        let pop = &populations[*p];
        let volume = pop.bulk().at_mass_density(*mass as f64)?;
        let expected = [
            pop.phase_sr1(*mu as f64)?,
            volume.extinction_m_inv,
            volume.scattering_m_inv,
            volume.absorption_m_inv,
        ];
        for c in 0..4 {
            let delta = (got[c] as f64 - expected[c]).abs();
            let relative = delta / expected[c].abs().max(1e-30);
            let (abs_tol, rel_tol) = if c == 0 { (1e-7, 2e-3) } else { (1e-15, 2e-5) };
            if c == 0 {
                max_phase_relative = max_phase_relative.max(relative);
            } else {
                max_coefficient_relative = max_coefficient_relative.max(relative);
            }
            if !got[c].is_finite() || got[c] < 0.0 || delta > abs_tol + rel_tol * expected[c].abs()
            {
                failures.push(serde_json::json!({"population":p,"mu":mu,"mass_kg_m3":mass,"component":c,"expected":expected[c],"actual":got[c]}));
            }
        }
    }
    let passed = failures.is_empty();
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"check":"liquid-population-cpu-wgsl","adapter":gpu.info.name,"backend":format!("{:?}",gpu.info.backend),"driver_info":gpu.info.driver_info,"populations":populations.len(),"cases":cases.len(),"passed":passed,"failures":failures,"max_phase_relative_error":max_phase_relative,"max_mass_coefficient_relative_error":max_coefficient_relative,"phase_tolerance":{"absolute_sr1":1e-7,"relative":2e-3},"coefficient_tolerance":{"absolute_m_inv":1e-15,"relative":2e-5},"contract":"Same host-prepared particle coefficients, scattering weights and bulk cross sections; GPU evaluates angular phase and model-mass conversion. No GPU Bessel solver or default PSD."})
        )?
    );
    if !passed {
        return Err("GPU liquid population audit failed".into());
    }
    Ok(())
}
