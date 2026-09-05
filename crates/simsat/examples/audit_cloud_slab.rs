//! Independent-solver requests and SimSat source-only cloud slab responses.
//! Black lower boundary, no atmosphere, no external ambient, SSA=1 direct term.
//! The experimental LUT internally retains its documented oracle SSA=0.999.
//! No image setting/default is changed. Compare against external DISORT outputs.
use simsat::spectral_transport::{
    SingleScatterSegment, SolarDepthEndpoints, integrate_single_scatter,
};
use simsat::{cloud_delta_flux, clouds};
use std::f64::consts::PI;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let steps = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "512".into())
        .parse::<usize>()?;
    if !(16..=8192).contains(&steps) {
        return Err("quadrature steps must be within 16..=8192".into());
    }
    let mut records = Vec::new();
    for (phase, g1, g2, w, el, ei) in [
        (
            "liquid",
            clouds::PHASE_LIQUID_G1,
            clouds::PHASE_LIQUID_G2,
            clouds::PHASE_LIQUID_W,
            1.0,
            0.0,
        ),
        (
            "ice",
            clouds::PHASE_ICE_G1,
            clouds::PHASE_ICE_G2,
            clouds::PHASE_ICE_W,
            0.0,
            1.0,
        ),
    ] {
        for tau in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0] {
            for mu0 in [
                0.15f64,
                0.3,
                0.42261826174069944,
                0.65,
                0.8660254037844387,
                0.98,
            ] {
                for muv in [0.35f64, 0.7, 1.0] {
                    for az in [0.0f64, 90.0, 180.0] {
                        let cos = (-mu0 * muv
                            + ((1.0 - mu0 * mu0) * (1.0 - muv * muv)).sqrt()
                                * az.to_radians().cos())
                        .clamp(-1.0, 1.0);
                        let ph = clouds::aggregate_phase(cos, el, ei);
                        let single = integrate_single_scatter(
                            [SingleScatterSegment::new(
                                tau / muv,
                                tau / muv,
                                ph,
                                Some(SolarDepthEndpoints::new(0.0, tau / mu0)?),
                            )?],
                            None,
                        )?
                        .scattered_normalized_radiance_sr1;
                        // Integrate source in viewing optical depth. The discarded
                        // tail has viewing transmittance below exp(-40)=4.25e-18.
                        let end = (tau / muv).min(40.0);
                        let dt = end / steps as f64;
                        let mut values = [0.0; 8];
                        for k in 0..steps {
                            for sign in [-1.0, 1.0] {
                                let tv = (k as f64 + 0.5 + sign / (2.0 * 3.0f64.sqrt())) * dt;
                                let depth = tv * muv / tau;
                                let sun_tau = tv * muv / mu0;
                                let old_tau = tau.max(sun_tau);
                                let weight = 0.5 * dt * (-tv).exp();
                                values[0] += weight
                                    * clouds::octave_sun_source_thin_gated(
                                        cos,
                                        el,
                                        ei,
                                        sun_tau,
                                        false,
                                        clouds::DEFAULT_OCTAVES,
                                        old_tau,
                                    );
                                values[1] += weight
                                    * clouds::octave_sun_source_thin_gated(
                                        cos,
                                        el,
                                        ei,
                                        sun_tau,
                                        true,
                                        clouds::DEFAULT_OCTAVES,
                                        old_tau,
                                    );
                                for (j, column) in [old_tau, tau].into_iter().enumerate() {
                                    values[2 + 3 * j] += weight
                                        * cloud_delta_flux::stage2_higher_order_source(
                                            column, depth, mu0, 0.0, el, ei,
                                        )
                                        .higher_isotropic;
                                    values[3 + 3 * j] += weight
                                        * cloud_delta_flux::stage2_higher_order_source_p1(
                                            column, depth, mu0, 0.0, el, ei, muv,
                                        )
                                        .higher_isotropic;
                                    values[4+3*j]+=weight*cloud_delta_flux::stage2_higher_order_source_order_memory(column,depth,mu0,0.0,el,ei,cos).higher_isotropic;
                                }
                            }
                        }
                        for v in &mut values[2..] {
                            *v += single;
                        }
                        let id = records.len();
                        records.push(serde_json::json!({
                            "id":id,"phase":phase,"tau":tau,"mu0":mu0,"muv":muv,"relative_azimuth_deg":az,"scattering_cosine":cos,
                            "within_lut_tau_mu_domain":(0.1..=30.0).contains(&tau)&&(0.42261826174069944..=0.8660254037844387).contains(&mu0),
                            "reference_request":{"tau":tau,"ssa":1.0,"kind":1,"g1":g1,"g2":g2,"weight":w,"gamma":0.0,"mu0":mu0,"muv":muv,"relative_azimuth_deg":az,"albedo":0.0},
                            "single_rho_f":PI*single,
                            "rho_f":{"legacy_beer":PI*values[0],"legacy_powder":PI*values[1],
                                "delta_v1_old_geometry":PI*values[2],"delta_v2_old_geometry":PI*values[3],"delta_v3_old_geometry":PI*values[4],
                                "delta_v1_vertical_column":PI*values[5],"delta_v2_vertical_column":PI*values[6],"delta_v3_vertical_column":PI*values[7]}
                        }));
                    }
                }
            }
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema":"simsat-cloud-slab-source-audit-v1","quadrature":"two-point Gauss per viewing optical-depth interval","intervals":steps,
            "view_optical_tail_limit":40.0,"case_count":records.len(),
            "limitations":["source-only homogeneous slab, not a full image","black ground and no atmospheric or external sky illumination",
                "SimSat direct scattering SSA=1; existing LUT source trained at SSA=0.999","vertical-column variant changes only the closure argument, not the runtime renderer"],
            "cases":records
        }))?
    );
    Ok(())
}
