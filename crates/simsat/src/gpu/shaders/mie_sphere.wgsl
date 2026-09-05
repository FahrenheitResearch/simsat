// Scalar unpolarized Lorenz-Mie angular sum, paired with mie_sphere.rs.
// Host prepares the shared sphere coefficients. This is not a GPU Bessel solver.
// Positive-angle parity and a difference recurrence avoid repeatedly subtracting
// large nearly equal pi_n values near forward/backscatter. This is algebraically
// the same Legendre recurrence, with no physical/phase-angle approximation.
struct MiePhaseState {
    s1: vec2<f32>,
    s2: vec2<f32>,
    pi_previous: f32,
    pi_current: f32,
    pi_difference: f32,
};
fn mie_phase_begin() -> MiePhaseState {
    return MiePhaseState(vec2<f32>(0.0),vec2<f32>(0.0),0.0,1.0,1.0);
}
fn mie_phase_step(state: MiePhaseState, coefficient: vec4<f32>, order: u32, cosine: f32) -> MiePhaseState {
    let n=f32(order);let mu=abs(cosine);
    var pi_n=state.pi_current;
    var tau=n*mu*pi_n-(n+1.0)*state.pi_previous;
    var next=(2.0*n+1.0)/n*mu*pi_n-(n+1.0)/n*state.pi_previous;
    var difference=next-pi_n;
    if (mu>=0.5) {
        tau=(n*(mu-1.0)-1.0)*pi_n+(n+1.0)*state.pi_difference;
        difference=(1.0+1.0/n)*state.pi_difference+(2.0+1.0/n)*(mu-1.0)*pi_n;
        next=pi_n+difference;
    }
    if (mu==1.0) {pi_n=0.5*n*(n+1.0);tau=pi_n;}
    var pi_sign=1.0;var tau_sign=1.0;
    if (cosine<0.0) {
        if (order%2u==0u) {pi_sign=-1.0;}
        tau_sign=-pi_sign;
    }
    let factor=(2.0*n+1.0)/(n*(n+1.0));
    let s1=state.s1+factor*(coefficient.xy*(pi_sign*pi_n)+coefficient.zw*(tau_sign*tau));
    let s2=state.s2+factor*(coefficient.xy*(tau_sign*tau)+coefficient.zw*(pi_sign*pi_n));
    return MiePhaseState(s1,s2,pi_n,next,difference);
}
fn mie_phase_finish(state: MiePhaseState, size_parameter: f32, qsca: f32) -> f32 {
    return (dot(state.s1,state.s1)+dot(state.s2,state.s2))/(6.283185307179586*size_parameter*size_parameter*qsca);
}
