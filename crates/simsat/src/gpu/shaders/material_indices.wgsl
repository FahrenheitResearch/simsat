// Same reference-condition interpolation as material_indices.rs. Host selects
// adjacent authoritative knots and validates wavelength within 0.4..=1.0 um.
// Each knot is (wavelength_um, n_real, n_imag_positive).
fn visible_material_index(wavelength_um: f32, a: vec3<f32>, b: vec3<f32>) -> vec2<f32> {
    if (wavelength_um == a.x) {return a.yz;}
    if (wavelength_um == b.x) {return b.yz;}
    let t = log(wavelength_um/a.x) / log(b.x/a.x);
    return vec2<f32>(a.y + t*(b.y-a.y), a.z*exp(t*log(b.z/a.z)));
}
