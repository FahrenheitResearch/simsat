//! Real-GPU sampling/math parity and optional all-order slab reference CSV.
//! `audit_spectral_path_gpu [slabs]`; slabs reads the same 13-column DISORT cases.
#[path = "support/compute_audit.rs"]
mod compute_audit;
use simsat::spectral_path::{Material, Moments, PhaseFunction, Random, direction_about, unit};
use std::io::Read;
const SHADER: &str = include_str!("../src/gpu/shaders/spectral_path.wgsl");
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gpu = compute_audit::ComputeAudit::new()?;
    if std::env::args().nth(1).as_deref() == Some("slabs") {
        return slabs(&gpu);
    }
    let mut phase_inputs = Vec::<[f32; 16]>::new();
    let mut expected = Vec::new();
    for phase in [
        PhaseFunction::rayleigh(0.014),
        PhaseFunction::dual_hg(0.85, -0.15, 0.9),
        PhaseFunction::dual_hg(0.75, -0.1, 0.9),
        PhaseFunction::dual_hg(1e-7, -0.009, 0.5),
        PhaseFunction::model_mixture(0.2, 0.3, 0.5, 0.014)?,
    ] {
        for i in 0..4096u32 {
            let mut c = [0.0f32; 16];
            c[0] = phase.rayleigh_weight as f32;
            c[1] = phase.gamma as f32;
            c[2] = -1.0 + 2.0 * i as f32 / 4095.0;
            c[3] = f32::from_bits(198271u32.wrapping_add(i.wrapping_mul(2654435761)));
            for k in 0..4 {
                c[4 + k] = phase.hg_g[k] as f32;
                c[8 + k] = phase.hg_weight[k] as f32;
            }
            let p = PhaseFunction {
                rayleigh_weight: c[0] as f64,
                gamma: c[1] as f64,
                hg_g: std::array::from_fn(|k| c[4 + k] as f64),
                hg_weight: std::array::from_fn(|k| c[8 + k] as f64),
            };
            let mut rng = Random::new(c[3].to_bits());
            let choose = rng.uniform();
            let u = rng.uniform();
            expected.push([p.sample_cosine(choose, u), p.value(c[2] as f64), choose, u]);
            phase_inputs.push(c);
        }
    }
    let src = format!(
        "{SHADER}\n{}",
        r#"
struct Case { a:vec4<f32>, g:vec4<f32>, w:vec4<f32>, reserved:vec4<f32> };
@group(0) @binding(0) var<storage,read> cases:array<Case>;
@group(0) @binding(1) var<storage,read_write> results:array<vec4<f32>>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){
 if(gid.x>=arrayLength(&cases)){return;}let c=cases[gid.x];
 var state=bitcast<u32>(c.a.w);let choose=path_uniform(&state);let u=path_uniform(&state);
 let p=PathPhase(c.a.x,c.a.y,c.g,c.w);
 results[gid.x]=vec4<f32>(path_phase_cosine(p,choose,u),path_phase_value(p,c.a.z),choose,u);
}
"#
    );
    let got = gpu.run(
        src,
        &phase_inputs.iter().flatten().copied().collect::<Vec<_>>(),
        phase_inputs.len(),
    )?;
    let mut maxima = [0.0f64; 4];
    let mut passed = true;
    for (g, e) in got.iter().zip(&expected) {
        for k in 0..4 {
            let err = (g[k] as f64 - e[k]).abs();
            maxima[k] = maxima[k].max(err);
            let tol = if k >= 2 {
                0.0
            } else {
                2e-6 + 2e-5 * e[k].abs()
            };
            passed &= g[k].is_finite() && err <= tol;
        }
    }
    let mut material_inputs = Vec::<[f32; 12]>::new();
    let mut material_expected = Vec::new();
    for normal in [
        [0.0, 0.0, 1.0],
        unit([0.3, 0.4, 0.5]),
        unit([-0.9, 0.2, 0.1]),
    ] {
        for mu in [0.05, 0.2, 0.6, 1.0] {
            for kind in [0, 1] {
                for sample in 0..1024u32 {
                    let view = direction_about(normal, mu, 0.7);
                    let mut c = [0.0f32; 12];
                    for k in 0..3 {
                        c[k] = normal[k] as f32;
                        c[4 + k] = view[k] as f32;
                    }
                    c[3] = kind as f32;
                    c[7] = if kind == 0 {
                        0.3
                    } else {
                        0.003 + 0.00512 * 7.0
                    };
                    c[8] = f32::from_bits(712371u32.wrapping_add(sample.wrapping_mul(2654435761)));
                    let material = if kind == 0 {
                        Material::Lambertian {
                            albedo: c[7] as f64,
                        }
                    } else {
                        Material::CoxMunk {
                            mean_square_slope: c[7] as f64,
                        }
                    };
                    let n = unit(std::array::from_fn(|k| c[k] as f64));
                    let v = unit(std::array::from_fn(|k| c[4 + k] as f64));
                    let mut rng = Random::new(c[8].to_bits());
                    let e = material
                        .sample(v, n, &mut rng)
                        .map_or([0.0; 4], |(d, w)| [d[0], d[1], d[2], w]);
                    material_inputs.push(c);
                    material_expected.push(e);
                }
            }
        }
    }
    let src = format!(
        "{SHADER}\n{}",
        r#"
struct Case { normal_kind:vec4<f32>,view_value:vec4<f32>,seed:vec4<f32> };
@group(0) @binding(0) var<storage,read> cases:array<Case>;
@group(0) @binding(1) var<storage,read_write> results:array<vec4<f32>>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){
 if(gid.x>=arrayLength(&cases)){return;}let c=cases[gid.x];var state=bitcast<u32>(c.seed.x);
 let n=normalize(c.normal_kind.xyz);let v=normalize(c.view_value.xyz);
 if(c.normal_kind.w<0.5){results[gid.x]=path_lambertian_sample(v,n,c.view_value.w,&state);}
 else{results[gid.x]=path_cox_munk_sample(v,n,c.view_value.w,&state);}
}
"#
    );
    let got = gpu.run(
        src,
        &material_inputs
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
        material_inputs.len(),
    )?;
    let mut material_max = [0.0f64; 4];
    let mut material_failed = 0usize;
    for (g, e) in got.iter().zip(&material_expected) {
        for k in 0..4 {
            let err = (g[k] as f64 - e[k]).abs();
            material_max[k] = material_max[k].max(err);
            if !g[k].is_finite() || err > 1e-5 + 2e-4 * e[k].abs() {
                passed = false;
                material_failed += 1;
            }
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
           "check":"spectral-path-cpu-wgsl-sampling","adapter":gpu.info.name,"backend":format!("{:?}",gpu.info.backend),
           "phase_cases":phase_inputs.len(),"phase_outputs":["cosine","phase_sr1","uniform_1","uniform_2"],"phase_max_absolute_error":maxima,
           "phase_tolerance":"2e-6 + 2e-5 abs(reference); RNG uniforms exact",
           "material_cases":material_inputs.len(),"material_outputs":["direction_x","direction_y","direction_z","bounce_weight"],"material_max_absolute_error":material_max,
           "material_tolerance":"1e-5 + 2e-4 abs(reference)","material_failed_components":material_failed,"passed":passed,
           "limitations":"Sampling/material kernels only. All-order slab tested separately; no claim of full 3D scene GPU equivalence."
        }))?
    );
    if !passed {
        return Err("spectral path CPU/WGSL sampling tolerance exceeded".into());
    }
    Ok(())
}
fn slabs(gpu: &compute_audit::ComputeAudit) -> Result<(), Box<dyn std::error::Error>> {
    const SAMPLES: usize = 65536;
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    let mut cases = Vec::new();
    let mut ids = Vec::new();
    for line in input.lines().filter(|s| !s.trim().is_empty()) {
        let c = line
            .split_whitespace()
            .map(str::parse::<f64>)
            .collect::<Result<Vec<_>, _>>()?;
        if c.len() != 13 {
            return Err("13-column DISORT case required".into());
        }
        let phase = if c[3] == 0.0 {
            PhaseFunction::rayleigh(c[7])
        } else {
            PhaseFunction::dual_hg(c[4], c[5], c[6])
        };
        simsat::spectral_path::HomogeneousSlab {
            tau: c[1],
            single_scatter_albedo: c[2],
            phase,
            solar_cosine: c[8],
            albedo: c[11],
        }
        .validate()?;
        if !(0.0 < c[9] && c[9] <= 1.0 && c[10].is_finite()) {
            return Err("invalid slab view".into());
        }
        let seed = 198271u32.wrapping_add((c[0] as u32).wrapping_mul(2654435761));
        cases.push([
            c[1] as f32,
            c[2] as f32,
            c[8] as f32,
            c[9] as f32,
            c[10].to_radians() as f32,
            c[11] as f32,
            c[3] as f32,
            f32::from_bits(seed),
            c[4] as f32,
            c[5] as f32,
            c[6] as f32,
            c[7] as f32,
        ]);
        ids.push((c[0] as usize, c[12] as usize));
    }
    eprintln!(
        "GPU slab audit: {} {:?}; {} paths/case",
        gpu.info.name, gpu.info.backend, SAMPLES
    );
    let src = format!(
        "{SHADER}\nconst PATH_SAMPLES:u32={SAMPLES}u;\n{}",
        r#"
struct Case { optics:vec4<f32>,geometry:vec4<f32>,phase:vec4<f32> };
@group(0) @binding(0) var<storage,read> cases:array<Case>;
@group(0) @binding(1) var<storage,read_write> results:array<vec4<f32>>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){
 let index=gid.x/PATH_SAMPLES;if(index>=arrayLength(&cases)){return;}let c=cases[index];
 var state=bitcast<u32>(c.geometry.w)+gid.x%PATH_SAMPLES;
 var phase=PathPhase(0.0,0.0,vec4<f32>(c.phase.xy,0.0,0.0),vec4<f32>(c.phase.z,1.0-c.phase.z,0.0,0.0));
 if(c.geometry.z==0.0){phase=PathPhase(1.0,c.phase.w,vec4<f32>(0.0),vec4<f32>(0.0));}
 results[gid.x]=path_trace_slab(c.optics.x,c.optics.y,phase,c.optics.z,c.optics.w,c.geometry.x,c.geometry.y,16u,0.95,100000u,&state);
}
"#
    );
    println!(
        "id,nstr,samples,rho_f,standard_error_rho_f,first_order_rho_f,higher_order_rho_f,mean_events"
    );
    for (batch, chunk) in cases.chunks(4).enumerate() {
        let got = gpu.run(
            src.clone(),
            &chunk.iter().flatten().copied().collect::<Vec<_>>(),
            chunk.len() * SAMPLES,
        )?;
        for (local, photons) in got.chunks_exact(SAMPLES).enumerate() {
            let mut m = [Moments::default(); 4];
            for p in photons {
                if p.iter().any(|v| !v.is_finite()) || p[3] < 0.0 {
                    return Err(
                        "GPU path safety failure or non-finite result; no truncated path accepted"
                            .into(),
                    );
                }
                for k in 0..4 {
                    m[k].push(p[k] as f64);
                }
            }
            let (id, nstr) = ids[batch * 4 + local];
            println!(
                "{id},{nstr},{SAMPLES},{:.17},{:.17},{:.17},{:.17},{:.6}",
                m[0].mean,
                m[0].standard_error().unwrap(),
                m[1].mean,
                m[2].mean,
                m[3].mean
            );
        }
    }
    Ok(())
}
