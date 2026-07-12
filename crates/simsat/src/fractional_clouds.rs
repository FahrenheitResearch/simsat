//! Fractional-cloud optical-depth closures.
//!
//! Model condensate is a grid-mean quantity.  For a layer with grid-mean optical
//! depth `tau` and cloud fraction `f`, the cloudy sub-column therefore has optical
//! depth `tau / f`.  A single layer's area-mean transmittance is
//!
//! `T = (1 - f) + f * exp(-tau / f)`.
//!
//! [`maximum_overlap_closure`] extends that transfer to a stack of layers using
//! exact maximum overlap on the linear-u8 fraction grid.  One shared uniform
//! coordinate `u` is used through the stack: a layer is cloudy when `u < f`.
//! Codes `1..=254` divide `[0, 1)` into 255 equal intervals, so the integral is an
//! exact fixed-size sum rather than a stochastic estimate.  Code 255 is a full-
//! coverage layer.  Code 0 with positive optical depth is an inconsistent source
//! value; it is conservatively repaired to full coverage and counted.
//!
//! The implementation performs no heap allocation.  It first accumulates the
//! in-cloud optical depth by fraction code, then walks the 255 nested maximum-
//! overlap intervals once.

/// The denominator of the linear-u8 cloud-fraction encoding.
pub const FRACTION_BINS: usize = 255;

/// Number of fixed midpoint samples used by the original opt-in deterministic
/// reference closure. Kept as a compatibility constant for callers that explicitly
/// request the four-member reference.
pub const DETERMINISTIC_SUBCOLUMN_COUNT: usize = 4;

/// Selectable fixed-stratified ensemble sizes. These deliberately remain a small,
/// bounded set: 4 is the interactive reference while 8 and 16 are convergence
/// experiments with correspondingly higher CPU cost.
pub const DETERMINISTIC_SUBCOLUMN_COUNTS: [usize; 3] = [4, 8, 16];

/// Midpoint coordinate of one deterministic stratified subcolumn.
///
/// The four valid coordinates are `0.125, 0.375, 0.625, 0.875`. Returning
/// `None` for an invalid index keeps callers from silently wrapping a requested
/// sample and biasing the ensemble mean.
#[inline]
pub fn deterministic_subcolumn_u(index: usize) -> Option<f64> {
    deterministic_subcolumn_u_for_count(index, DETERMINISTIC_SUBCOLUMN_COUNT)
}

/// Midpoint coordinate of one member in a selectable deterministic stratified
/// ensemble. One coordinate is shared through every vertical layer and every
/// view/sun/shadow construction for that member, preserving maximum overlap.
///
/// Only the reviewed 4/8/16 sizes are accepted. Returning `None` for an unsupported
/// count or out-of-range index prevents accidental silent wrapping or a zero-sized
/// ensemble from biasing the radiance mean.
#[inline]
pub fn deterministic_subcolumn_u_for_count(index: usize, count: usize) -> Option<f64> {
    (DETERMINISTIC_SUBCOLUMN_COUNTS.contains(&count) && index < count)
        .then_some((index as f64 + 0.5) / count as f64)
}

/// Large finite value used to keep hostile inputs from overflowing accumulators.
/// Optical depths remotely near this value already have zero representable
/// transmittance, so the clamp has no observable transfer effect.
const MAX_TRACKED_TAU: f64 = 1.0e300;

/// The exact, allocation-free closure for one maximum-overlapped layer stack.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaximumOverlapClosure {
    /// Sum of the input grid-mean optical depths whose codes are in `1..=254`.
    pub raw_fractional_tau: f64,
    /// Optical depth of the exact mean fractional transmittance.
    pub effective_fractional_tau: f64,
    /// `effective_fractional_tau / raw_fractional_tau`; `1` when there is no
    /// positive fractional optical depth.
    pub scale: f64,
    /// Optical depth applied to every sub-column: code 255 plus repaired code 0.
    pub full_tau: f64,
    /// `full_tau + raw_fractional_tau`.
    pub raw_total_tau: f64,
    /// `full_tau + effective_fractional_tau`.
    pub effective_total_tau: f64,
    /// Exact mean transmittance of only the codes in `1..=254`.
    pub mean_fractional_transmittance: f64,
    /// Mean transmittance after the full/base optical depth is also applied.
    pub mean_total_transmittance: f64,
    /// Number of positive-optical-depth code-0 layers repaired to full coverage.
    pub repaired_zero_count: usize,
    /// Number of positive-optical-depth layers with codes in `1..=254`.
    pub fractional_layer_count: usize,
    /// Number of positive-optical-depth full layers, including repaired zeroes.
    pub full_layer_count: usize,
    /// Number of zero, negative, NaN, or negative-infinite input layers ignored.
    pub ignored_layer_count: usize,
}

impl Default for MaximumOverlapClosure {
    fn default() -> Self {
        Self {
            raw_fractional_tau: 0.0,
            effective_fractional_tau: 0.0,
            scale: 1.0,
            full_tau: 0.0,
            raw_total_tau: 0.0,
            effective_total_tau: 0.0,
            mean_fractional_transmittance: 1.0,
            mean_total_transmittance: 1.0,
            repaired_zero_count: 0,
            fractional_layer_count: 0,
            full_layer_count: 0,
            ignored_layer_count: 0,
        }
    }
}

/// Area-mean transmittance of one fractional layer.
///
/// Codes `1..=254` use `(1-f) + f*exp(-tau/f)`, with `f = code/255`.
/// Code 255 and a positive-tau code 0 use the legacy/full transfer `exp(-tau)`.
/// Zero, negative, NaN, and negative-infinite optical depths are clear.  Positive
/// infinity is treated as the saturated finite limit.
pub fn fractional_layer_transmittance(grid_mean_tau: f64, fraction_code: u8) -> f64 {
    let Some(tau) = usable_tau(grid_mean_tau) else {
        return 1.0;
    };
    match fraction_code {
        0 | 255 => (-tau).exp(),
        code => {
            let f = code as f64 / FRACTION_BINS as f64;
            // `1 + f*expm1(-tau/f)` is the transfer equation written to retain
            // precision in the optically-thin limit.
            (1.0 + f * (-(tau / f)).exp_m1()).clamp(0.0, 1.0)
        }
    }
}

/// Effective optical depth corresponding to [`fractional_layer_transmittance`].
///
/// Full/repaired-full layers return their input optical depth exactly (within the
/// hostile-input saturation clamp), avoiding an unnecessary exp/log round trip.
pub fn fractional_layer_effective_tau(grid_mean_tau: f64, fraction_code: u8) -> f64 {
    let Some(tau) = usable_tau(grid_mean_tau) else {
        return 0.0;
    };
    match fraction_code {
        0 | 255 => tau,
        code => {
            let f = code as f64 / FRACTION_BINS as f64;
            let absorbed_in_cloud = -(-(tau / f)).exp_m1();
            // Jensen's inequality makes this no larger than the grid-mean tau;
            // the `min` only suppresses a possible last-bit roundoff violation.
            (-(-f * absorbed_in_cloud).ln_1p()).min(tau)
        }
    }
}

/// Compute the exact shared-`u` maximum-overlap closure for a layer stack.
///
/// Each iterator item is `(grid_mean_layer_tau, fraction_code)`.  Fractional
/// layers use their in-cloud optical depth `tau/(code/255)`.  Because all layers
/// share `u`, a code-`c` layer contributes to exactly the first `c` of the 255
/// equal intervals.  Full and repaired-zero optical depth is factored out of the
/// fractional integral and reported separately.
pub fn maximum_overlap_closure<I>(layers: I) -> MaximumOverlapClosure
where
    I: IntoIterator<Item = (f64, u8)>,
{
    // Index c holds the summed in-cloud OD of fractional layers with code c.
    // Index 255 remains zero because full layers are factored into `full_tau`.
    let mut in_cloud_by_code = [0.0f64; FRACTION_BINS + 1];
    let mut out = MaximumOverlapClosure::default();

    for (input_tau, code) in layers {
        let Some(tau) = usable_tau(input_tau) else {
            out.ignored_layer_count += 1;
            continue;
        };
        match code {
            0 => {
                out.full_tau = bounded_tau_add(out.full_tau, tau);
                out.repaired_zero_count += 1;
                out.full_layer_count += 1;
            }
            255 => {
                out.full_tau = bounded_tau_add(out.full_tau, tau);
                out.full_layer_count += 1;
            }
            code => {
                let f = code as f64 / FRACTION_BINS as f64;
                let in_cloud_tau = (tau / f).min(MAX_TRACKED_TAU);
                let slot = &mut in_cloud_by_code[code as usize];
                *slot = bounded_tau_add(*slot, in_cloud_tau);
                out.raw_fractional_tau = bounded_tau_add(out.raw_fractional_tau, tau);
                out.fractional_layer_count += 1;
            }
        }
    }

    // The overwhelming majority of real-domain columns are clear or contain only
    // full-coverage cloud. Avoid 255 transcendental evaluations for that no-op case.
    if out.raw_fractional_tau <= 0.0 {
        out.raw_total_tau = out.full_tau;
        out.effective_total_tau = out.full_tau;
        out.mean_total_transmittance = (-out.full_tau).exp();
        return out;
    }

    // Integrate the 255 equal u intervals from the clearest interval downward.
    // In interval m, precisely the layers with code > m are active.  Summing
    // absorption rather than transmittance preserves the thin-limit digits.
    let mut active_tau = 0.0f64;
    let mut absorption_sum = 0.0f64;
    let mut compensation = 0.0f64;
    for interval in (0..FRACTION_BINS).rev() {
        active_tau = bounded_tau_add(active_tau, in_cloud_by_code[interval + 1]);
        let absorbed = -(-active_tau).exp_m1();
        // Kahan sum: useful when every interval carries only tiny optical depth.
        let y = absorbed - compensation;
        let next = absorption_sum + y;
        compensation = (next - absorption_sum) - y;
        absorption_sum = next;
    }

    let max_fractional_absorption = (FRACTION_BINS - 1) as f64 / FRACTION_BINS as f64;
    let mean_absorption =
        (absorption_sum / FRACTION_BINS as f64).clamp(0.0, max_fractional_absorption);
    out.mean_fractional_transmittance = 1.0 - mean_absorption;
    // Jensen's inequality gives tau_eff <= raw tau.  Retain that invariant if
    // transcendental rounding puts the computed value an ulp above the bound.
    out.effective_fractional_tau = (-(-mean_absorption).ln_1p()).min(out.raw_fractional_tau);
    out.scale = if out.raw_fractional_tau > 0.0 {
        (out.effective_fractional_tau / out.raw_fractional_tau).clamp(0.0, 1.0)
    } else {
        1.0
    };
    out.raw_total_tau = bounded_tau_add(out.full_tau, out.raw_fractional_tau);
    out.effective_total_tau = bounded_tau_add(out.full_tau, out.effective_fractional_tau);
    out.mean_total_transmittance = (-out.full_tau).exp() * out.mean_fractional_transmittance;
    out
}

#[inline]
fn usable_tau(tau: f64) -> Option<f64> {
    if tau.is_nan() || tau <= 0.0 {
        None
    } else if tau.is_infinite() {
        if tau.is_sign_positive() {
            Some(MAX_TRACKED_TAU)
        } else {
            None
        }
    } else {
        Some(tau.min(MAX_TRACKED_TAU))
    }
}

#[inline]
fn bounded_tau_add(a: f64, b: f64) -> f64 {
    let sum = a + b;
    if sum.is_finite() {
        sum.min(MAX_TRACKED_TAU)
    } else {
        MAX_TRACKED_TAU
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance,
            "actual {actual:.17e}, expected {expected:.17e}, error {error:.3e} > {tolerance:.3e}"
        );
    }

    #[test]
    fn empty_stack_is_clear_and_neutral() {
        let c = maximum_overlap_closure(std::iter::empty());
        assert_eq!(c, MaximumOverlapClosure::default());
    }

    #[test]
    fn deterministic_four_uses_fixed_stratified_midpoints() {
        let got: Vec<f64> = (0..DETERMINISTIC_SUBCOLUMN_COUNT)
            .map(|n| deterministic_subcolumn_u(n).unwrap())
            .collect();
        assert_eq!(got, vec![0.125, 0.375, 0.625, 0.875]);
        assert_eq!(
            deterministic_subcolumn_u(DETERMINISTIC_SUBCOLUMN_COUNT),
            None
        );
    }

    #[test]
    fn selectable_deterministic_ensembles_use_reproducible_stratified_midpoints() {
        for count in DETERMINISTIC_SUBCOLUMN_COUNTS {
            let first = deterministic_subcolumn_u_for_count(0, count).unwrap();
            let last = deterministic_subcolumn_u_for_count(count - 1, count).unwrap();
            assert_eq!(first, 0.5 / count as f64);
            assert_eq!(last, 1.0 - 0.5 / count as f64);
            assert_eq!(deterministic_subcolumn_u_for_count(count, count), None);
            for index in 0..count {
                let a = deterministic_subcolumn_u_for_count(index, count).unwrap();
                let b = deterministic_subcolumn_u_for_count(index, count).unwrap();
                assert_eq!(a.to_bits(), b.to_bits());
                assert!(a > 0.0 && a < 1.0);
                if index > 0 {
                    let previous = deterministic_subcolumn_u_for_count(index - 1, count).unwrap();
                    assert_eq!((a - previous).to_bits(), (1.0 / count as f64).to_bits());
                }
            }
        }
        assert_eq!(deterministic_subcolumn_u_for_count(0, 0), None);
        assert_eq!(deterministic_subcolumn_u_for_count(0, 2), None);
        assert_eq!(deterministic_subcolumn_u_for_count(0, 32), None);
    }

    #[test]
    fn scalar_transfer_matches_the_fraction_point_two_anchor() {
        // 51/255 is exactly 0.2: in-cloud tau is 1/0.2 = 5.
        let expected_t = 0.8 + 0.2 * (-5.0f64).exp();
        let expected_tau = -expected_t.ln();
        assert_close(fractional_layer_transmittance(1.0, 51), expected_t, 1.0e-15);
        assert_close(
            fractional_layer_effective_tau(1.0, 51),
            expected_tau,
            1.0e-15,
        );
        assert_close(expected_t, 0.801_347_589_399_817_1, 1.0e-15);
        assert_close(expected_tau, 0.221_460_481_721_012_35, 1.0e-15);
    }

    #[test]
    fn scalar_thin_limit_is_the_grid_mean_optical_depth() {
        let tau = 1.0e-10;
        for code in [1u8, 17, 51, 127, 254] {
            let effective = fractional_layer_effective_tau(tau, code);
            assert!(effective > 0.0);
            assert!(
                ((effective - tau) / tau).abs() < 2.0e-8,
                "code {code}: effective {effective:.17e} vs raw {tau:.17e}"
            );
        }
    }

    #[test]
    fn scalar_transfer_is_bounded_and_monotone_in_tau() {
        for code in 1u8..=254 {
            let mut previous_tau = 0.0;
            let mut previous_t = 1.0;
            for raw in [0.0, 1.0e-8, 1.0e-4, 0.01, 0.1, 1.0, 10.0, 1.0e6] {
                let effective = fractional_layer_effective_tau(raw, code);
                let transmittance = fractional_layer_transmittance(raw, code);
                assert!(effective >= previous_tau - 1.0e-15);
                assert!((0.0..=raw.max(0.0) + 1.0e-12).contains(&effective));
                assert!((0.0..=1.0).contains(&transmittance));
                assert!(transmittance <= previous_t + 1.0e-15);
                previous_tau = effective;
                previous_t = transmittance;
            }
        }
    }

    #[test]
    fn full_and_repaired_zero_layers_are_exact_legacy_base_tau() {
        assert_close(
            fractional_layer_transmittance(1.25, 255),
            (-1.25f64).exp(),
            0.0,
        );
        assert_close(
            fractional_layer_transmittance(1.25, 0),
            (-1.25f64).exp(),
            0.0,
        );
        assert_eq!(fractional_layer_effective_tau(1.25, 255), 1.25);
        assert_eq!(fractional_layer_effective_tau(1.25, 0), 1.25);

        let c = maximum_overlap_closure([(1.25, 255), (2.75, 0)]);
        assert_eq!(c.full_tau, 4.0);
        assert_eq!(c.raw_fractional_tau, 0.0);
        assert_eq!(c.effective_fractional_tau, 0.0);
        assert_eq!(c.scale, 1.0);
        assert_eq!(c.raw_total_tau, 4.0);
        assert_eq!(c.effective_total_tau, 4.0);
        assert_eq!(c.mean_fractional_transmittance, 1.0);
        assert_close(c.mean_total_transmittance, (-4.0f64).exp(), 0.0);
        assert_eq!(c.repaired_zero_count, 1);
        assert_eq!(c.full_layer_count, 2);
    }

    #[test]
    fn saturated_fractional_layer_opacity_tends_to_coverage() {
        let fifth = maximum_overlap_closure([(1.0e6, 51)]);
        assert_close(fifth.mean_fractional_transmittance, 0.8, 1.0e-15);
        assert_close(fifth.effective_fractional_tau, -0.8f64.ln(), 1.0e-15);

        let almost_full = maximum_overlap_closure([(1.0e6, 254)]);
        assert_close(
            almost_full.mean_fractional_transmittance,
            1.0 / FRACTION_BINS as f64,
            1.0e-15,
        );
        assert_close(
            almost_full.effective_fractional_tau,
            (FRACTION_BINS as f64).ln(),
            1.0e-14,
        );
    }

    #[test]
    fn two_layer_shared_u_closure_matches_the_piecewise_analytic_result() {
        // Layer A: f=.2, in-cloud tau=5.  Layer B: f=.4, in-cloud tau=1.25.
        // 51 bins see both, 51 see only B, and 153 are clear.
        let c = maximum_overlap_closure([(1.0, 51), (0.5, 102), (0.7, 255)]);
        let expected_fractional_t =
            (51.0 * (-6.25f64).exp() + 51.0 * (-1.25f64).exp() + 153.0) / 255.0;
        assert_close(
            c.mean_fractional_transmittance,
            expected_fractional_t,
            2.0e-15,
        );
        assert_close(
            c.effective_fractional_tau,
            -expected_fractional_t.ln(),
            2.0e-15,
        );
        assert_eq!(c.raw_fractional_tau, 1.5);
        assert_eq!(c.full_tau, 0.7);
        assert_close(
            c.mean_total_transmittance,
            (-0.7f64).exp() * expected_fractional_t,
            2.0e-15,
        );
    }

    #[test]
    fn maximum_overlap_is_permutation_invariant() {
        let layers = [
            (0.7, 13),
            (1.2, 51),
            (0.09, 127),
            (3.0, 211),
            (0.4, 254),
            (0.2, 255),
            (0.3, 0),
        ];
        let forward = maximum_overlap_closure(layers);
        let reverse = maximum_overlap_closure(layers.into_iter().rev());
        assert_close(
            forward.raw_fractional_tau,
            reverse.raw_fractional_tau,
            1.0e-15,
        );
        assert_close(
            forward.effective_fractional_tau,
            reverse.effective_fractional_tau,
            1.0e-15,
        );
        assert_close(forward.full_tau, reverse.full_tau, 1.0e-15);
        assert_close(
            forward.mean_fractional_transmittance,
            reverse.mean_fractional_transmittance,
            1.0e-15,
        );
        assert_eq!(forward.repaired_zero_count, reverse.repaired_zero_count);
    }

    #[test]
    fn invalid_and_extreme_inputs_never_produce_nan_or_negative_outputs() {
        let invalid = maximum_overlap_closure([
            (f64::NAN, 51),
            (-1.0, 102),
            (f64::NEG_INFINITY, 255),
            (0.0, 0),
        ]);
        assert_eq!(
            invalid,
            MaximumOverlapClosure {
                ignored_layer_count: 4,
                ..MaximumOverlapClosure::default()
            }
        );
        assert_eq!(fractional_layer_transmittance(f64::NAN, 51), 1.0);
        assert_eq!(fractional_layer_effective_tau(-1.0, 51), 0.0);

        // Positive infinity is the saturated limit, represented internally by a
        // huge finite tau so subtraction-free accumulation cannot form Inf-Inf.
        let saturated = maximum_overlap_closure([(f64::INFINITY, 51), (f64::INFINITY, 0)]);
        for value in [
            saturated.raw_fractional_tau,
            saturated.effective_fractional_tau,
            saturated.scale,
            saturated.full_tau,
            saturated.raw_total_tau,
            saturated.effective_total_tau,
            saturated.mean_fractional_transmittance,
            saturated.mean_total_transmittance,
        ] {
            assert!(value.is_finite(), "non-finite closure value {value}");
            assert!(value >= 0.0, "negative closure value {value}");
        }
        assert_eq!(saturated.repaired_zero_count, 1);
    }

    #[test]
    fn closure_bounds_hold_for_a_mixed_deterministic_stack() {
        let mut state = 0x8a5c_13d7_u64;
        let mut layers = Vec::new();
        for _ in 0..512 {
            // A tiny deterministic generator keeps the test dependency-free.
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let code = (state >> 32) as u8;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let tau = ((state >> 11) as f64 / ((1u64 << 53) as f64)) * 3.0;
            layers.push((tau, code));
        }
        let c = maximum_overlap_closure(layers);
        assert!(c.effective_fractional_tau >= 0.0);
        assert!(c.effective_fractional_tau <= c.raw_fractional_tau);
        assert!((0.0..=1.0).contains(&c.scale));
        assert!(c.effective_total_tau <= c.raw_total_tau);
        assert!(c.mean_fractional_transmittance > 0.0);
        assert!(c.mean_fractional_transmittance <= 1.0);
    }

    #[test]
    fn analytic_closure_matches_a_dense_shared_u_reference() {
        let layers = [
            (0.03, 7),
            (0.8, 51),
            (0.2, 96),
            (1.7, 173),
            (0.4, 238),
            (0.6, 255),
            (0.25, 0),
        ];
        let analytic = maximum_overlap_closure(layers);

        // A multiple of 255 midpoint samples gives every encoded u interval the
        // same number of samples and never lands exactly on a fraction boundary.
        let samples = FRACTION_BINS * 200;
        let mut sum_t = 0.0f64;
        for sample in 0..samples {
            let u = (sample as f64 + 0.5) / samples as f64;
            let mut fractional_tau = 0.0f64;
            for &(tau, code) in &layers {
                if (1..=254).contains(&code) {
                    let f = code as f64 / FRACTION_BINS as f64;
                    if u < f {
                        fractional_tau += tau / f;
                    }
                }
            }
            sum_t += (-fractional_tau).exp();
        }
        let dense_t = sum_t / samples as f64;
        assert_close(analytic.mean_fractional_transmittance, dense_t, 1.0e-12);
        assert_close(analytic.effective_fractional_tau, -dense_t.ln(), 2.0e-12);
    }
}
