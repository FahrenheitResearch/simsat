// CPU twin: spectral_molecular.rs. Standalone scientific kernel, not wired to
// the legacy RGB display uniforms. Host MUST validate lambda 0.25..=1.0 um,
// CO2 0..=1000 ppm and finite nonnegative dry-air number density/column.
// Spectral extinction / phase consumers use these properties, not RGB constants.
const MOLECULAR_PI: f32 = 3.141592653589793;
struct DryAirRayleigh {
    cross_section_m2: f32,
    king_factor: f32,
    phase_gamma: f32,
};
fn dry_air_rayleigh(wavelength_um: f32, co2_ppm: f32) -> DryAirRayleigh {
    let inv_lambda2 = 1.0 / (wavelength_um * wavelength_um);
    let refractivity_300 = 1.0e-8 * (8060.51 + 2480990.0 / (132.274 - inv_lambda2)
        + 17455.7 / (39.32957 - inv_lambda2));
    let refractivity = refractivity_300 * (1.0 + 0.54 * (co2_ppm * 1.0e-6 - 0.0003));
    let co2_percent = co2_ppm * 1.0e-4;
    let f_n2 = 1.034 + 3.17e-4 * inv_lambda2;
    let f_o2 = 1.096 + 1.385e-3 * inv_lambda2 + 1.448e-4 * inv_lambda2 * inv_lambda2;
    let king = (78.084 * f_n2 + 20.946 * f_o2 + 0.934 + co2_percent * 1.15)
        / (78.084 + 20.946 + 0.934 + co2_percent);
    let n2_minus_one = refractivity * (2.0 + refractivity);
    let ratio = n2_minus_one / (3.0 + n2_minus_one);
    // Work in um in the denominator to avoid Ns^2 overflowing float32.
    let ns_scaled = 2.546899e13;
    let lambda2 = wavelength_um * wavelength_um;
    let sigma = 24.0 * MOLECULAR_PI * MOLECULAR_PI * MOLECULAR_PI * ratio * ratio * king
        / (lambda2 * lambda2 * ns_scaled * ns_scaled);
    let delta = 6.0 * (king - 1.0) / (7.0 * king + 3.0);
    return DryAirRayleigh(sigma, king, delta / (2.0 - delta));
}
fn dry_air_rayleigh_phase(optic: DryAirRayleigh, cos_theta: f32) -> f32 {
    let gamma = optic.phase_gamma;
    return 3.0 / (16.0 * MOLECULAR_PI * (1.0 + 2.0 * gamma))
        * ((1.0 + 3.0 * gamma) + (1.0 - gamma) * cos_theta * cos_theta);
}
fn dry_air_rayleigh_scattering_m1(optic: DryAirRayleigh, dry_number_density_m3: f32) -> f32 {
    return optic.cross_section_m2 * dry_number_density_m3;
}
fn dry_air_rayleigh_optical_depth(optic: DryAirRayleigh, dry_number_column_m2: f32) -> f32 {
    return optic.cross_section_m2 * dry_number_column_m2;
}
