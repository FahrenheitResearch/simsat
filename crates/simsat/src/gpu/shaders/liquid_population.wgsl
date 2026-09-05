// Shared population reduction after host-prepared physical Mie coefficients.
// Number/radius integration is common host preparation for both backends.
// Inputs are normalized scattering weights, not number or mass fractions.
fn liquid_population_phase_add(accumulated_sr1: f32, particle_phase_sr1: f32, scattering_weight: f32) -> f32 {
    return accumulated_sr1 + particle_phase_sr1 * scattering_weight;
}
// mass_coefficients=(extinction, scattering, absorption) in m^2 kg^-1.
fn liquid_population_volume_coefficients(mass_coefficients: vec3<f32>, liquid_kg_m3: f32) -> vec3<f32> {
    return mass_coefficients * liquid_kg_m3;
}
