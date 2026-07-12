//! Experimental Stage-2 Monte Carlo cloud diffuse-source closure.
//!
//! This is deliberately opt-in. It preserves the exact direct single-scatter source and
//! reconstructs only the higher-order field from the immutable Stage-2 RTX 5090 oracle.
//! The LUT axes are cloud phase regime, `log1p(column tau)`, solar cosine, Lambertian
//! surface albedo, and fractional vertical optical depth. Linear interpolation between
//! nonnegative samples is bounded and shape preserving. The v1 directional
//! reconstruction is isotropic (`1 / 4 pi`), the smallest nonnegative normalized
//! angular kernel. The opt-in v2b reconstruction adds a bounded P1 upper-boundary
//! escape moment normalized to preserve the upward-hemisphere mean source. The v3
//! candidate restores the angular memory of the second scattering order with an
//! exactly normalized, nonnegative dual-HG convolution, then decays that memory in
//! transport optical depth. Unlike an exposure adjustment, v3 changes cloud geometry
//! and phase-angle contrast while preserving the angle-integrated higher-order source.

use std::f64::consts::PI;

use crate::clouds::{
    PHASE_ICE_G1, PHASE_ICE_G2, PHASE_ICE_W, PHASE_LIQUID_G1, PHASE_LIQUID_G2, PHASE_LIQUID_W,
    henyey_greenstein,
};

include!(concat!(env!("OUT_DIR"), "/stage2_cloud_lut.rs"));

const TAU: [f64; 6] = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0];
const LOG1P_TAU: [f64; 6] = [
    0.095_310_179_804_324_87,
    0.262_364_264_467_491_06,
    std::f64::consts::LN_2,
    1.386_294_361_119_890_6,
    2.397_895_272_798_370_7,
    3.433_987_204_485_146_3,
];
const MU: [f64; 2] = [
    0.422_618_261_740_699_44, // cos(65 deg)
    0.866_025_403_784_438_7,  // cos(30 deg)
];
const ALBEDO: [f64; 2] = [0.0, 0.2];
const DEPTH_BINS: usize = 32;
const PROFILES: usize = 2;
const ORACLE_SSA: f64 = 0.999;
const INV_FOUR_PI: f64 = 1.0 / (4.0 * PI);
// First moments of the shipping dual-HG mixtures:
// liquid = .9*.85 + .1*(-.15), ice = .9*.75 + .1*(-.10).
const LIQUID_G_BAR: f64 = 0.75;
const ICE_G_BAR: f64 = 0.665;
/// Conservative half-strength of the maximum nonnegative P1 moment. V2b normalizes
/// the resulting kernel over the upward escape hemisphere, retaining angular contrast
/// without the general brightening seen in the unnormalized v2 exploratory round.
const P1_ESCAPE_STRENGTH: f64 = 0.5;
/// Successive-order angular-memory decay per transport optical depth. The immutable
/// Stage-2 calibration grid selects 0.72 (minimum directional-BRF RMSE is 0.73); this
/// is close to `ln(2)`, i.e. one transport OD approximately halves second-order memory.
const ORDER_MEMORY_DECAY: f64 = 0.72;
/// A Lambertian lower boundary destroys incident-direction memory. This bounded
/// empirical damping leaves a black boundary untouched and retains about 20% of the
/// memory at albedo 0.2, the upper boundary represented by the shipping LUT.
const LAMBERTIAN_MEMORY_DECAY: f64 = 8.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeltaFluxSource {
    /// Nonnegative higher-order local source per unit extinction and solar normal flux.
    pub higher_isotropic: f64,
    /// Interpolated all-order scattering-collision density per incident horizontal path.
    pub total_collision_density: f64,
    /// The density remaining after subtracting the analytic direct first collision.
    pub higher_collision_density: f64,
}

impl DeltaFluxSource {
    pub const ZERO: Self = Self {
        higher_isotropic: 0.0,
        total_collision_density: 0.0,
        higher_collision_density: 0.0,
    };
}

#[inline]
fn bracket(nodes: &[f64], value: f64) -> (usize, usize, f64) {
    let value = value.clamp(nodes[0], nodes[nodes.len() - 1]);
    for upper in 1..nodes.len() {
        if value <= nodes[upper] {
            let lower = upper - 1;
            let span = nodes[upper] - nodes[lower];
            return (lower, upper, (value - nodes[lower]) / span);
        }
    }
    let last = nodes.len() - 1;
    (last, last, 0.0)
}

#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

#[inline]
fn lut(profile: usize, tau: usize, mu: usize, albedo: usize, depth: usize) -> f64 {
    let index = ((((profile * TAU.len() + tau) * MU.len() + mu) * ALBEDO.len() + albedo)
        * DEPTH_BINS)
        + depth;
    STAGE2_SOURCE_LUT[index] as f64
}

fn profile_density(profile: usize, tau: f64, mu: f64, albedo: f64, fractional_depth: f64) -> f64 {
    debug_assert!(profile < PROFILES);
    let (t0, t1, ft) = bracket(&LOG1P_TAU, tau.clamp(TAU[0], TAU[TAU.len() - 1]).ln_1p());
    let (m0, m1, fm) = bracket(&MU, mu);
    let (a0, a1, fa) = bracket(&ALBEDO, albedo);
    let depth_coordinate = fractional_depth.clamp(0.0, 1.0) * DEPTH_BINS as f64 - 0.5;
    let (d0, d1, fd) = if depth_coordinate <= 0.0 {
        (0, 0, 0.0)
    } else if depth_coordinate >= (DEPTH_BINS - 1) as f64 {
        (DEPTH_BINS - 1, DEPTH_BINS - 1, 0.0)
    } else {
        let lower = depth_coordinate.floor() as usize;
        (lower, lower + 1, depth_coordinate - lower as f64)
    };

    let at = |ti: usize, mi: usize, ai: usize| {
        lerp(
            lut(profile, ti, mi, ai, d0),
            lut(profile, ti, mi, ai, d1),
            fd,
        )
    };
    let at_tau = |ti: usize| {
        let low_albedo = lerp(at(ti, m0, a0), at(ti, m0, a1), fa);
        let high_albedo = lerp(at(ti, m1, a0), at(ti, m1, a1), fa);
        lerp(low_albedo, high_albedo, fm)
    };
    lerp(at_tau(t0), at_tau(t1), ft).max(0.0)
}

/// Return the Stage-2 higher-order local source for a mixed liquid/ice sample.
///
/// `column_tau` is the display-scaled vertical whole-column cloud OD and
/// `fractional_depth` is top=0, lower boundary=1. Liquid and ice use the exact shipping
/// dual-HG oracle regimes; precipitation follows the ice regime. Below the first oracle
/// node (`tau=0.1`) the source coefficient is forced linear in tau, so its integrated
/// higher-order radiance vanishes as `O(tau^2)`.
pub fn stage2_higher_order_source(
    column_tau: f64,
    fractional_depth: f64,
    solar_cosine: f64,
    surface_albedo: f64,
    ext_liquid: f64,
    ext_ice_precip: f64,
) -> DeltaFluxSource {
    let ext_total = ext_liquid.max(0.0) + ext_ice_precip.max(0.0);
    if !column_tau.is_finite()
        || column_tau <= 0.0
        || !solar_cosine.is_finite()
        || solar_cosine <= 0.0
        || ext_total <= 0.0
    {
        return DeltaFluxSource::ZERO;
    }

    let tau_eval = column_tau.clamp(TAU[0], TAU[TAU.len() - 1]);
    let mu_eval = solar_cosine.clamp(MU[0], MU[MU.len() - 1]);
    let depth = fractional_depth.clamp(0.0, 1.0);
    let albedo = surface_albedo.clamp(ALBEDO[0], ALBEDO[ALBEDO.len() - 1]);
    let liquid = profile_density(0, tau_eval, mu_eval, albedo, depth);
    let ice = profile_density(1, tau_eval, mu_eval, albedo, depth);
    let total_density = (liquid * ext_liquid.max(0.0) + ice * ext_ice_precip.max(0.0)) / ext_total;
    let first_density = ORACLE_SSA * tau_eval / mu_eval * (-tau_eval * depth / mu_eval).exp();
    let higher_density = (total_density - first_density).clamp(0.0, total_density);
    let thin_scale = (column_tau / TAU[0]).clamp(0.0, 1.0);
    let higher_isotropic =
        (mu_eval * higher_density / tau_eval * INV_FOUR_PI * thin_scale).max(0.0);

    if !(total_density.is_finite() && higher_density.is_finite() && higher_isotropic.is_finite()) {
        return DeltaFluxSource::ZERO;
    }
    DeltaFluxSource {
        higher_isotropic,
        total_collision_density: total_density,
        higher_collision_density: higher_density,
    }
}

/// Reconstruct the Stage-2 higher-order source with a bounded P1 escape moment.
///
/// The Monte Carlo LUT is angle integrated, so v1 uses an isotropic kernel. Close to
/// a cloud's vacuum upper boundary, however, the diffuse field has an outward flux.
/// In P1 form `I = J * (1 + 3 f mu)`. We use the nonnegative Eddington bound
/// `3f = 0.5`, half the nonnegative limit, decayed by optical distance to the upper
/// boundary in *transport* optical depth `(1-g_bar) tau`. For upward directions
/// `mu in [0, 1]`, the exact uniform-solid-angle mean of `1 + q mu` is `1 + q/2`;
/// dividing by it makes the upward-hemisphere mean exactly one. This retains
/// centre-to-limb directional contrast without a blanket satellite-view brightness
/// gain, exposure compensation, or any change to the v1 closure.
pub fn stage2_higher_order_source_p1(
    column_tau: f64,
    fractional_depth: f64,
    solar_cosine: f64,
    surface_albedo: f64,
    ext_liquid: f64,
    ext_ice_precip: f64,
    mu_to_view: f64,
) -> DeltaFluxSource {
    let mut source = stage2_higher_order_source(
        column_tau,
        fractional_depth,
        solar_cosine,
        surface_albedo,
        ext_liquid,
        ext_ice_precip,
    );
    if source.higher_isotropic <= 0.0 || !mu_to_view.is_finite() {
        return source;
    }

    let liquid = ext_liquid.max(0.0);
    let ice = ext_ice_precip.max(0.0);
    let ext_total = liquid + ice;
    if ext_total <= 0.0 {
        return DeltaFluxSource::ZERO;
    }
    let g_bar = (liquid * LIQUID_G_BAR + ice * ICE_G_BAR) / ext_total;
    let transport_depth =
        (1.0 - g_bar).max(0.0) * column_tau.max(0.0) * fractional_depth.clamp(0.0, 1.0);
    let p1_moment = P1_ESCAPE_STRENGTH * (-transport_depth).exp();
    let directional_multiplier =
        (1.0 + p1_moment * mu_to_view.clamp(-1.0, 1.0)) / (1.0 + 0.5 * p1_moment);
    source.higher_isotropic *= directional_multiplier;
    source
}

/// The angle-normalized convolution of two aggregate dual-HG scattering events.
///
/// A HG phase has Legendre moments `g^l`; convolution therefore multiplies the
/// moments and maps a lobe pair `(g_i, g_j)` to another normalized HG lobe with
/// parameter `g_i*g_j`. Expanding the shipping liquid/ice mixture into four weighted
/// lobes and convolving every pair is exact for two events in a locally homogeneous
/// phase mixture. Multiplying by `4*pi` returns a dimensionless kernel whose spherical
/// mean is exactly one. It is nonnegative and has a finite maximum because every
/// shipping `|g_i*g_j| < 1`.
fn second_order_memory_kernel(cos_theta: f64, ext_liquid: f64, ext_ice_precip: f64) -> f64 {
    let liquid = ext_liquid.max(0.0);
    let ice = ext_ice_precip.max(0.0);
    let total = liquid + ice;
    if total <= 0.0 || !cos_theta.is_finite() {
        return 1.0;
    }
    let liquid_fraction = liquid / total;
    let ice_fraction = ice / total;
    let lobes = [
        (liquid_fraction * PHASE_LIQUID_W, PHASE_LIQUID_G1),
        (liquid_fraction * (1.0 - PHASE_LIQUID_W), PHASE_LIQUID_G2),
        (ice_fraction * PHASE_ICE_W, PHASE_ICE_G1),
        (ice_fraction * (1.0 - PHASE_ICE_W), PHASE_ICE_G2),
    ];
    let cosine = cos_theta.clamp(-1.0, 1.0);
    let mut phase = 0.0;
    for &(weight_a, g_a) in &lobes {
        for &(weight_b, g_b) in &lobes {
            phase += weight_a * weight_b * henyey_greenstein(cosine, g_a * g_b);
        }
    }
    (4.0 * PI * phase).max(0.0)
}

/// Reconstruct the Stage-2 higher-order source with successive-order angular memory.
///
/// The Stage-2 LUT stores an angle-integrated collision source. Treating every order
/// above the first as isotropic is especially wrong in optically thin cloud: the
/// second order dominates and retains the squared Legendre moments of the particle
/// phase function. Joseph, Wiscombe & Weinman (1976), DOI
/// 10.1175/1520-0469(1976)033<2452:TDEAFR>2.0.CO;2, likewise identify `g^2` as the
/// forward-peak moment needed by delta-Eddington transport. Hansen & Travis (1974),
/// DOI 10.1007/BF00168069, formulate multiple scattering by successive orders.
///
/// V3 uses the exact two-event convolution above and blends it toward isotropy as
///
/// `K = 1 + exp[-0.72*(1-g_bar)*tau - 8*A] * (K2 - 1)`.
///
/// Both endpoints are nonnegative normalized phase kernels, so their convex blend is
/// nonnegative, finite, and preserves the Stage-2 angle-integrated source exactly.
/// The scalar source, direct single scatter, extinction and cloud OD are unchanged.
pub fn stage2_higher_order_source_order_memory(
    column_tau: f64,
    fractional_depth: f64,
    solar_cosine: f64,
    surface_albedo: f64,
    ext_liquid: f64,
    ext_ice_precip: f64,
    cos_theta: f64,
) -> DeltaFluxSource {
    let mut source = stage2_higher_order_source(
        column_tau,
        fractional_depth,
        solar_cosine,
        surface_albedo,
        ext_liquid,
        ext_ice_precip,
    );
    if source.higher_isotropic <= 0.0 {
        return source;
    }

    let liquid = ext_liquid.max(0.0);
    let ice = ext_ice_precip.max(0.0);
    let total = liquid + ice;
    if total <= 0.0 || !cos_theta.is_finite() {
        return DeltaFluxSource::ZERO;
    }
    let g_bar = (liquid * LIQUID_G_BAR + ice * ICE_G_BAR) / total;
    let transport_tau = (1.0 - g_bar).max(0.0) * column_tau.max(0.0);
    let memory = (-ORDER_MEMORY_DECAY * transport_tau
        - LAMBERTIAN_MEMORY_DECAY * surface_albedo.clamp(0.0, 1.0))
    .exp()
    .clamp(0.0, 1.0);
    let second_order = second_order_memory_kernel(cos_theta, liquid, ice);
    let multiplier = 1.0 + memory * (second_order - 1.0);
    source.higher_isotropic *= multiplier.max(0.0);
    source
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage2_source_is_finite_nonnegative_and_energy_bounded() {
        for &tau in &[0.0, 1.0e-8, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0] {
            for &mu in &[0.01, MU[0], 0.6, MU[1], 1.0] {
                for &depth in &[0.0, 0.1, 0.5, 0.9, 1.0] {
                    for &(liquid, ice) in &[(1.0, 0.0), (0.0, 1.0), (0.4, 0.6)] {
                        let source = stage2_higher_order_source(tau, depth, mu, 0.15, liquid, ice);
                        assert!(source.higher_isotropic.is_finite());
                        assert!(source.higher_isotropic >= 0.0);
                        assert!(source.higher_collision_density >= 0.0);
                        assert!(
                            source.higher_collision_density
                                <= source.total_collision_density + 1.0e-12
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn thin_limit_integrated_higher_order_is_quadratic() {
        let source1 = stage2_higher_order_source(1.0e-3, 0.5, 0.7, 0.0, 1.0, 0.0).higher_isotropic;
        let source2 = stage2_higher_order_source(2.0e-3, 0.5, 0.7, 0.0, 1.0, 0.0).higher_isotropic;
        let integrated_ratio = (2.0e-3 * source2) / (1.0e-3 * source1);
        assert!(
            (integrated_ratio - 4.0).abs() < 1.0e-12,
            "{integrated_ratio}"
        );
    }

    #[test]
    fn oracle_identity_is_pinned() {
        assert_eq!(
            STAGE2_ORACLE_SHA256,
            "7ba7aee813098ee831378df6f853844d74f790fe8d15baf38744614e729404aa"
        );
        assert_eq!(STAGE2_SOURCE_LUT.len(), PROFILES * 6 * 2 * 2 * DEPTH_BINS);
    }

    #[test]
    fn p1_escape_reconstruction_is_nonnegative_bounded_and_upward_mean_preserving() {
        let isotropic = stage2_higher_order_source(3.0, 0.2, 0.7, 0.1, 0.6, 0.4);
        let up = stage2_higher_order_source_p1(3.0, 0.2, 0.7, 0.1, 0.6, 0.4, 1.0);
        let down = stage2_higher_order_source_p1(3.0, 0.2, 0.7, 0.1, 0.6, 0.4, -1.0);
        let side = stage2_higher_order_source_p1(3.0, 0.2, 0.7, 0.1, 0.6, 0.4, 0.0);
        assert!(down.higher_isotropic >= 0.0);
        assert!(up.higher_isotropic <= 1.2 * isotropic.higher_isotropic);
        assert!(down.higher_isotropic >= 0.4 * isotropic.higher_isotropic);
        assert!(side.higher_isotropic <= isotropic.higher_isotropic);
        assert!(
            ((up.higher_isotropic + side.higher_isotropic) * 0.5 - isotropic.higher_isotropic)
                .abs()
                < 1.0e-15
        );
        assert_eq!(
            up.total_collision_density,
            isotropic.total_collision_density
        );
        assert_eq!(
            up.higher_collision_density,
            isotropic.higher_collision_density
        );
    }

    #[test]
    fn second_order_memory_kernel_is_normalized_nonnegative_and_finite() {
        for &(liquid, ice) in &[(1.0, 0.0), (0.0, 1.0), (0.4, 0.6)] {
            let bins = 20_000;
            let dmu = 2.0 / bins as f64;
            let mut integral = 0.0;
            let mut maximum = 0.0f64;
            for index in 0..bins {
                let mu = -1.0 + (index as f64 + 0.5) * dmu;
                let kernel = second_order_memory_kernel(mu, liquid, ice);
                assert!(kernel.is_finite() && kernel >= 0.0);
                integral += 0.5 * kernel * dmu;
                maximum = maximum.max(kernel);
            }
            assert!((integral - 1.0).abs() < 2.0e-6, "mean {integral}");
            assert!(maximum < 23.0, "finite shipping-lobe bound {maximum}");
        }
    }

    #[test]
    fn order_memory_suppresses_thin_backscatter_and_relaxes_to_isotropic() {
        let isotropic_thin = stage2_higher_order_source(0.1, 0.3, 0.866, 0.0, 1.0, 0.0);
        let memory_thin =
            stage2_higher_order_source_order_memory(0.1, 0.3, 0.866, 0.0, 1.0, 0.0, -0.866);
        let ratio = memory_thin.higher_isotropic / isotropic_thin.higher_isotropic;
        assert!(
            (0.30..0.40).contains(&ratio),
            "thin backscatter ratio {ratio}"
        );

        let isotropic_thick = stage2_higher_order_source(30.0, 0.3, 0.866, 0.0, 1.0, 0.0);
        let memory_thick =
            stage2_higher_order_source_order_memory(30.0, 0.3, 0.866, 0.0, 1.0, 0.0, -0.866);
        let thick_ratio = memory_thick.higher_isotropic / isotropic_thick.higher_isotropic;
        assert!(
            (thick_ratio - 1.0).abs() < 0.004,
            "thick ratio {thick_ratio}"
        );
        assert_eq!(
            memory_thin.higher_collision_density,
            isotropic_thin.higher_collision_density
        );
    }

    #[test]
    fn lambertian_boundary_damps_but_never_reverses_order_memory() {
        let black = stage2_higher_order_source_order_memory(0.3, 0.2, 0.866, 0.0, 0.0, 1.0, -0.7);
        let bright = stage2_higher_order_source_order_memory(0.3, 0.2, 0.866, 0.2, 0.0, 1.0, -0.7);
        let bright_isotropic = stage2_higher_order_source(0.3, 0.2, 0.866, 0.2, 0.0, 1.0);
        let black_isotropic = stage2_higher_order_source(0.3, 0.2, 0.866, 0.0, 0.0, 1.0);
        assert!(black.higher_isotropic < black_isotropic.higher_isotropic);
        assert!(bright.higher_isotropic < bright_isotropic.higher_isotropic);
        assert!(
            bright.higher_isotropic / bright_isotropic.higher_isotropic
                > black.higher_isotropic / black_isotropic.higher_isotropic
        );
    }
}
