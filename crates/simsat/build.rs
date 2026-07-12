use std::fs;
use std::io::Write;
use std::path::PathBuf;

use serde_json::Value;

#[path = "../../build-support/windows_version.rs"]
mod windows_version;

const TAU: [f64; 6] = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0];
const SZA: [f64; 2] = [65.0, 30.0];
const PROFILE: [&str; 2] = ["shipping-liquid-dual-hg", "shipping-ice-dual-hg"];
const BINS: usize = 32;

fn exact_index(value: f64, nodes: &[f64]) -> Option<usize> {
    nodes.iter().position(|node| (value - node).abs() < 1.0e-12)
}

fn state_index(profile: usize, tau: usize, sza: usize, albedo: usize) -> usize {
    (((profile * TAU.len() + tau) * SZA.len() + sza) * 3) + albedo
}

fn number(value: &Value, field: &str) -> f64 {
    value[field]
        .as_f64()
        .unwrap_or_else(|| panic!("Stage-2 field {field} is not numeric"))
}

fn main() {
    windows_version::embed(
        &["simsat-render-frame", "simsat-render-ir"],
        "SimSat",
        "SimSat command-line renderer",
    );

    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let source =
        manifest.join("../../experiments/cuda-cloud-oracle/stage2-results-rtx5090-v1.jsonl");
    println!("cargo:rerun-if-changed={}", source.display());

    let mut states: Vec<Option<(usize, Vec<f32>)>> =
        vec![None; PROFILE.len() * TAU.len() * SZA.len() * 3];
    let input = fs::read_to_string(&source)
        .unwrap_or_else(|error| panic!("read Stage-2 oracle {}: {error}", source.display()));
    for line in input.lines() {
        let record: Value = serde_json::from_str(line).expect("parse Stage-2 oracle row");
        let request = &record["request"];
        let Some(profile) = request["phase_profile"]
            .as_str()
            .and_then(|name| PROFILE.iter().position(|candidate| name == *candidate))
        else {
            continue;
        };
        let ssa = request["ssa"].as_str().unwrap().parse::<f64>().unwrap();
        if (ssa - 0.999).abs() > 1.0e-12 {
            continue;
        }
        let Some(tau) = exact_index(request["tau"].as_str().unwrap().parse().unwrap(), &TAU) else {
            continue;
        };
        let Some(sza) = exact_index(
            request["sun_zenith_deg"].as_str().unwrap().parse().unwrap(),
            &SZA,
        ) else {
            continue;
        };
        let albedo_value = request["surface_albedo"]
            .as_str()
            .unwrap()
            .parse::<f64>()
            .unwrap();
        let albedo = if albedo_value.abs() < 1.0e-12 {
            0
        } else if (albedo_value - 0.2).abs() < 1.0e-12 {
            1
        } else if (albedo_value - 0.6).abs() < 1.0e-12 {
            2
        } else {
            continue;
        };
        let bins = record["result"]["collision_source"]["bins"]
            .as_array()
            .expect("Stage-2 collision bins");
        assert_eq!(bins.len(), BINS, "Stage-2 depth-bin count changed");
        let values: Vec<f32> = bins
            .iter()
            .map(|bin| number(bin, "scattering_source_density") as f32)
            .collect();
        let max_scatters = request["max_scatters"]
            .as_str()
            .unwrap()
            .parse::<usize>()
            .unwrap();
        let slot = &mut states[state_index(profile, tau, sza, albedo)];
        match slot {
            Some((previous_order, previous)) if *previous_order == max_scatters => {
                assert_eq!(previous, &values, "duplicate Stage-2 state is not exact");
            }
            Some((previous_order, _)) if *previous_order > max_scatters => {}
            _ => *slot = Some((max_scatters, values)),
        }
    }

    let output = PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("stage2_cloud_lut.rs");
    let mut file = fs::File::create(&output).expect("create generated Stage-2 LUT");
    writeln!(
        file,
        "// Generated from stage2-results-rtx5090-v1.jsonl; do not edit."
    )
    .unwrap();
    writeln!(
        file,
        "#[allow(clippy::excessive_precision)]\npub(crate) const STAGE2_SOURCE_LUT: [f32; {}] = [",
        PROFILE.len() * TAU.len() * SZA.len() * 2 * BINS
    )
    .unwrap();
    for profile in 0..PROFILE.len() {
        for tau in 0..TAU.len() {
            for sza in 0..SZA.len() {
                let zero = states[state_index(profile, tau, sza, 0)]
                    .as_ref()
                    .expect("missing Stage-2 albedo-zero state")
                    .1
                    .as_slice();
                for albedo in 0..2 {
                    let values = if albedo == 0 {
                        zero.to_vec()
                    } else if let Some((_, exact)) = &states[state_index(profile, tau, sza, 1)] {
                        exact.clone()
                    } else {
                        let high = states[state_index(profile, tau, sza, 2)]
                            .as_ref()
                            .expect("missing Stage-2 albedo interpolation endpoint")
                            .1
                            .as_slice();
                        zero.iter()
                            .zip(high)
                            .map(|(&lo, &hi)| lo + (hi - lo) / 3.0)
                            .collect()
                    };
                    for value in values {
                        assert!(value.is_finite() && value >= 0.0);
                        writeln!(file, "    {value:.9e}f32,").unwrap();
                    }
                }
            }
        }
    }
    writeln!(file, "];").unwrap();
    writeln!(
        file,
        "pub const STAGE2_ORACLE_SHA256: &str = \"7ba7aee813098ee831378df6f853844d74f790fe8d15baf38744614e729404aa\";"
    )
    .unwrap();
}
