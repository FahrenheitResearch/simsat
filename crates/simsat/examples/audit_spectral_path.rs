//! Independent all-order path estimates for the exact scalar DISORT slab inputs.
//! Reads: id tau ssa kind g1 g2 weight gamma mu0 muv relaz_deg albedo nstr.
//! nstr is retained for join/provenance, not used by the Monte Carlo method.
use rayon::prelude::*;
use simsat::spectral_path::{self, HomogeneousSlab, Moments, PathConfig, PhaseFunction, Random};
use std::{f64::consts::PI, io::Read};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut samples = 500000usize;
    let mut seed = 198271u32;
    for arg in std::env::args().skip(1) {
        let (k, v) = arg.split_once('=').ok_or("expected samples= or seed=")?;
        match k {
            "samples" => samples = v.parse()?,
            "seed" => seed = v.parse()?,
            _ => return Err("unknown argument".into()),
        }
    }
    if samples < 2 {
        return Err("at least two independent paths required".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(6)
        .build_global()?;
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    let config = PathConfig {
        roulette_start_order: 16,
        roulette_weight_threshold: 0.95,
        event_safety_limit: 100000,
    };
    config.validate()?;
    println!(
        "id,nstr,samples,rho_f,standard_error_rho_f,first_order_rho_f,higher_order_rho_f,mean_events"
    );
    for line in input
        .lines()
        .filter(|s| !s.trim().is_empty() && !s.trim_start().starts_with('#'))
    {
        let f = line
            .split_whitespace()
            .map(str::parse::<f64>)
            .collect::<Result<Vec<_>, _>>()?;
        if f.len() != 13 {
            return Err("expected 13 slab columns".into());
        }
        let phase = match f[3] as usize {
            0 => PhaseFunction::rayleigh(f[7]),
            1 => PhaseFunction::dual_hg(f[4], f[5], f[6]),
            _ => return Err("invalid phase kind".into()),
        };
        let scene = HomogeneousSlab {
            tau: f[1],
            single_scatter_albedo: f[2],
            phase,
            solar_cosine: f[8],
            albedo: f[11],
        };
        scene.validate()?;
        if !f[9].is_finite() || f[9] <= 0.0 || f[9] > 1.0 || !f[10].is_finite() {
            return Err("invalid view geometry".into());
        }
        let direction = HomogeneousSlab::view_direction(f[9], f[10].to_radians());
        let chunks = (0..samples.div_ceil(4096))
            .into_par_iter()
            .map(|chunk| -> Result<_, String> {
                let mut all = Moments::default();
                let (mut first, mut higher, mut events) = (0.0, 0.0, 0usize);
                for path in chunk * 4096..((chunk + 1) * 4096).min(samples) {
                    let mut rng = Random::new(
                        seed.wrapping_add(path as u32)
                            .wrapping_add((f[0] as u32).wrapping_mul(2654435761)),
                    );
                    let result = spectral_path::trace(
                        &scene,
                        [0.0, 0.0, scene.tau],
                        direction,
                        config,
                        &mut rng,
                    )?;
                    all.push(PI * result.total());
                    first += PI * result.first_order_l_over_e;
                    higher += PI * result.higher_order_l_over_e;
                    events += result.events;
                }
                Ok((all, first, higher, events))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut all = Moments::default();
        let (mut first, mut higher, mut events) = (0.0, 0.0, 0usize);
        for (m, a, b, n) in chunks {
            let count = all.count + m.count;
            let delta = m.mean - all.mean;
            all.m2 += m.m2 + delta * delta * (all.count * m.count) as f64 / count as f64;
            all.mean += delta * m.count as f64 / count as f64;
            all.count = count;
            first += a;
            higher += b;
            events += n;
        }
        println!(
            "{},{},{},{:.17},{:.17},{:.17},{:.17},{:.6}",
            f[0] as usize,
            f[12] as usize,
            samples,
            all.mean,
            all.standard_error().unwrap(),
            first / samples as f64,
            higher / samples as f64,
            events as f64 / samples as f64
        );
    }
    Ok(())
}
