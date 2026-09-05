// CPU twin: spectral_transport.rs. Input optical depths are finite/nonnegative;
// 0 <= scattering_tau <= extinction_tau and phase_sr1 >= 0 are HOST validated.
// This is only the direct single-scattering term, not a full cloud closure.
fn spectral_mean_transmittance(tau_near: f32, tau_far: f32) -> f32 {
    let span=abs(tau_far-tau_near);
    var factor: f32;
    // WGSL has no expm1. Series through x^4 keeps thin optical depths stable;
    // for x<0.05 the first omitted term is less than 4.4e-10.
    if (span < 0.05) {
        factor=1.0+span*(-0.5+span*(1.0/6.0+span*(-1.0/24.0+span/120.0)));
    } else { factor=(1.0-exp(-span))/span; }
    return exp(-min(tau_near,tau_far))*factor;
}
fn spectral_single_scatter_segment(view_tau_before: f32, extinction_tau: f32,
    scattering_tau: f32, phase_sr1: f32, solar_near: f32, solar_far: f32) -> f32 {
    return scattering_tau*spectral_mean_transmittance(view_tau_before+solar_near,
        view_tau_before+extinction_tau+solar_far)*phase_sr1;
}
fn spectral_direct_lambertian(albedo: f32, incidence_cosine: f32,
    solar_tau: f32, view_tau: f32) -> f32 {
    return albedo*incidence_cosine/3.141592653589793*exp(-solar_tau-view_tau);
}
