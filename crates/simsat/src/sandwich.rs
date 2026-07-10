//! Sandwich composite (the classic severe-convection view): semi-transparent COLOR-enhanced
//! band-13 IR (cold cloud tops) overlaid on the visible true-color base. The visible gives the
//! fine cloud texture / structure; the IR color highlights the coldest overshooting tops.
//!
//! This module holds the pure, node-testable composite math + the documented thresholds. The
//! render orchestration (rendering the visible frame AND the IR frame through the SAME
//! [`crate::api::render`] paths and feeding this blend) lives in [`crate::api`] under
//! [`crate::api::Product::Sandwich`]; the studio reuses [`blend_rgba`] directly over its
//! already-rendered visible composite + a band-13 IR march.
//!
//! Structurally this mirrors [`crate::geocolor`] — both render a visible frame + an IR frame
//! and composite per pixel — but the BLEND RULE differs. GeoColor crossfades day-visible vs
//! night-IR by the PER-PIXEL SOLAR ELEVATION. Sandwich alpha-OVERLAYS the colored IR over the
//! visible BASE by the PER-PIXEL BRIGHTNESS TEMPERATURE (coldness): the visible shows through
//! everywhere, the colored IR fades in only on the cold tops.
//!
//! ## Thresholds (documented, the tunable constants)
//! - BT `>= `[`SANDWICH_WARM_THRESHOLD_K`]` (260 K)` -> alpha 0: warm/low areas stay FULLY
//!   visible (true-color cloud/ground detail shows through, no IR color).
//! - BT `<= `[`SANDWICH_COLD_THRESHOLD_K`]` (200 K)` -> alpha [`SANDWICH_MAX_ALPHA`] (0.80):
//!   the coldest overshooting tops get the STRONGEST IR color — but not fully opaque, so the
//!   visible cloud texture still shows through even there.
//! - between (200..260 K) -> a linear ramp of alpha with coldness (see [`sandwich_alpha`]).
//! - The overlay enhancement is [`SANDWICH_ENHANCEMENT`] (a one-constant swap).
//!
//! ## Daytime product (honest)
//! Sandwich needs a LIT visible base to be meaningful — it is a daytime-convection product. At
//! night the visible is dark, so the composite degrades to roughly the colored IR over black
//! (still shows the cold tops, but with no true-color texture underneath). Documented; use
//! GeoColor or plain IR for a night storm.

use crate::ir_enhance::IrEnhancement;

/// BT (Kelvin) at/above which the sandwich overlay alpha is 0 — warm/low areas stay fully
/// visible (no IR color). The warm edge of the cold-top overlay.
pub const SANDWICH_WARM_THRESHOLD_K: f64 = 260.0;
/// BT (Kelvin) at/below which the sandwich overlay alpha is [`SANDWICH_MAX_ALPHA`] — the
/// coldest overshooting tops get the strongest IR color. The cold edge of the overlay.
pub const SANDWICH_COLD_THRESHOLD_K: f64 = 200.0;
/// The maximum overlay alpha (at the coldest tops). Deliberately < 1 so the visible cloud
/// TEXTURE still shows through even under the coldest, most strongly-colored tops — the point
/// of the sandwich (visible structure + IR color together, not IR alone).
pub const SANDWICH_MAX_ALPHA: f64 = 0.80;
/// The IR band-13 enhancement used for the sandwich overlay: the NOAA/SSD RB rainbow. Over the
/// 200-260 K overlay range it runs green -> yellow -> orange -> red (the coldest tops), the
/// classic cold-top severe-convection color ramp. Swappable to another `IrEnhancement` (e.g.
/// [`IrEnhancement::Cimss`] enhanced-IR, or [`IrEnhancement::Bd`]) in this one place.
pub const SANDWICH_ENHANCEMENT: IrEnhancement = IrEnhancement::Rainbow;

/// The overlay alpha for a pixel at brightness temperature `bt_k` (Kelvin): a LINEAR ramp from
/// 0 at/above [`SANDWICH_WARM_THRESHOLD_K`] (warm — fully visible) to [`SANDWICH_MAX_ALPHA`]
/// at/below [`SANDWICH_COLD_THRESHOLD_K`] (the coldest tops — strongest IR color). Monotone
/// increasing with coldness. A non-finite BT (off-domain / margin — no IR data) returns 0, so
/// those pixels stay the pure visible base.
#[inline]
pub fn sandwich_alpha(bt_k: f64) -> f64 {
    if !bt_k.is_finite() {
        // No IR data (off-domain / zoom-out margin) -> keep the visible base.
        return 0.0;
    }
    let t = ((SANDWICH_WARM_THRESHOLD_K - bt_k)
        / (SANDWICH_WARM_THRESHOLD_K - SANDWICH_COLD_THRESHOLD_K))
        .clamp(0.0, 1.0);
    t * SANDWICH_MAX_ALPHA
}

/// Composite the visible true-color frame and the colored-IR frame into the sandwich: the
/// visible is the base everywhere, the colored IR is overlaid on the cold tops by the per-pixel
/// [`sandwich_alpha`] of the brightness temperature.
///
/// - `vis_rgba` is the finished true-color frame, `n*4` bytes; its alpha is the on-earth mask
///   (alpha 0 = space -> the output stays black + transparent there regardless of the IR).
/// - `ir_rgb` is the colored-IR frame, `n*3` bytes (colored through [`SANDWICH_ENHANCEMENT`]),
///   with out-of-domain pixels rendered black.
/// - `bt_kelvin` is the RAW band-13 brightness temperature plane (`n` f32, Kelvin; `NaN`
///   off-domain), which drives the per-pixel overlay alpha.
///
/// Returns `(rgb, rgba)`: the `n*3` RGB (space = black) the store writer / display uses, and
/// the `n*4` RGBA carrying the visible on-earth alpha (space alpha 0).
pub fn blend_rgba(
    vis_rgba: &[u8],
    ir_rgb: &[u8],
    bt_kelvin: &[f32],
    n: usize,
) -> (Vec<u8>, Vec<u8>) {
    assert!(vis_rgba.len() >= n * 4, "vis_rgba too short");
    assert!(ir_rgb.len() >= n * 3, "ir_rgb too short");
    assert!(bt_kelvin.len() >= n, "bt_kelvin too short");
    let mut rgb = vec![0u8; n * 3];
    let mut rgba = vec![0u8; n * 4];
    for i in 0..n {
        let a = vis_rgba[i * 4 + 3];
        if a == 0 {
            // Space: black + transparent, matching the visible/IR off-earth convention.
            continue;
        }
        let alpha = sandwich_alpha(bt_kelvin[i] as f64);
        for c in 0..3 {
            let v = (1.0 - alpha) * vis_rgba[i * 4 + c] as f64 + alpha * ir_rgb[i * 3 + c] as f64;
            let b = v.round().clamp(0.0, 255.0) as u8;
            rgb[i * 3 + c] = b;
            rgba[i * 4 + c] = b;
        }
        rgba[i * 4 + 3] = a;
    }
    (rgb, rgba)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sandwich_alpha_is_a_monotone_ramp_zero_warm_max_cold() {
        // Warm (>= 260 K) -> 0; cold (<= 200 K) -> the max alpha (the documented ends).
        assert_eq!(sandwich_alpha(SANDWICH_WARM_THRESHOLD_K), 0.0);
        assert_eq!(sandwich_alpha(300.0), 0.0);
        assert_eq!(
            sandwich_alpha(SANDWICH_COLD_THRESHOLD_K),
            SANDWICH_MAX_ALPHA
        );
        assert_eq!(sandwich_alpha(180.0), SANDWICH_MAX_ALPHA);
        // Off-domain (NaN) keeps the visible base (alpha 0).
        assert_eq!(sandwich_alpha(f64::NAN), 0.0);
        // Ramps monotonically UP as BT drops from warm to cold, strictly between the ends.
        let span = SANDWICH_WARM_THRESHOLD_K - SANDWICH_COLD_THRESHOLD_K;
        let mut prev = -1.0;
        for k in 0..=20 {
            // walk from cold to warm; alpha should DECREASE (monotone with coldness)
            let bt = SANDWICH_COLD_THRESHOLD_K + span * (k as f64 / 20.0);
            let a = sandwich_alpha(bt);
            assert!(
                (0.0..=SANDWICH_MAX_ALPHA).contains(&a),
                "alpha {a} out of [0, max] at {bt} K"
            );
            if k > 0 {
                assert!(a <= prev + 1e-12, "not monotone (cold->warm) at {bt} K");
            }
            prev = a;
        }
        // The interior of the band is a genuine partial overlay (neither pinned end).
        let mid = sandwich_alpha((SANDWICH_WARM_THRESHOLD_K + SANDWICH_COLD_THRESHOLD_K) * 0.5);
        assert!(
            mid > 0.0 && mid < SANDWICH_MAX_ALPHA,
            "mid {mid} not partial"
        );
    }

    #[test]
    fn warm_top_pixel_equals_the_visible_rgb() {
        // A warm-top pixel (BT >= the warm threshold) has alpha 0, so the composite is the
        // visible RGB byte-for-byte — warm areas stay fully true-color.
        let vis = vec![40, 90, 160, 255]; // a lit visible surface pixel
        let ir = vec![200, 20, 20]; // a saturated IR color that must NOT bleed in here
        let bt = vec![290.0f32]; // warm surface -> alpha 0
        let (rgb, rgba) = blend_rgba(&vis, &ir, &bt, 1);
        assert_eq!(
            &rgb[0..3],
            &[40, 90, 160],
            "warm pixel should equal visible"
        );
        assert_eq!(&rgba[0..4], &[40, 90, 160, 255]);
    }

    #[test]
    fn very_cold_top_pixel_is_strongly_ir_colored_but_visible_still_contributes() {
        // A very-cold-top pixel (BT <= the cold threshold) blends at the MAX alpha (0.80): the
        // output is strongly the IR color, but the visible base STILL contributes (alpha < 1),
        // so it is neither the pure IR nor the pure visible.
        let vis = vec![250u8, 250, 250, 255]; // bright white visible cloud texture
        let ir = vec![190u8, 0, 0]; // a cold-top red (Rainbow-style)
        let bt = vec![195.0f32]; // very cold overshoot -> alpha = SANDWICH_MAX_ALPHA
        let (rgb, _) = blend_rgba(&vis, &ir, &bt, 1);
        // Expected: 0.2*visible + 0.8*ir per channel.
        let expect = |v: f64, i: f64| {
            ((1.0 - SANDWICH_MAX_ALPHA) * v + SANDWICH_MAX_ALPHA * i).round() as u8
        };
        assert_eq!(rgb[0], expect(250.0, 190.0));
        assert_eq!(rgb[1], expect(250.0, 0.0));
        assert_eq!(rgb[2], expect(250.0, 0.0));
        // Strongly IR-colored: red dominates (R clearly the largest channel), matching the IR.
        assert!(
            rgb[0] > rgb[1] + 60 && rgb[0] > rgb[2] + 60,
            "not IR-dominated: {rgb:?}"
        );
        // But the visible still contributes: green/blue are NOT the pure IR 0 (the white bled
        // in), and red is NOT the pure IR 190 either.
        assert!(
            rgb[1] > 0 && rgb[2] > 0,
            "visible did not contribute: {rgb:?}"
        );
        assert_ne!(
            [rgb[0], rgb[1], rgb[2]],
            [190, 0, 0],
            "must not be the pure IR"
        );
        assert_ne!(
            [rgb[0], rgb[1], rgb[2]],
            [250, 250, 250],
            "must not be the pure visible"
        );
    }

    #[test]
    fn mixed_frame_shows_visible_warm_and_ir_colored_cold() {
        // Two pixels: a warm one reads the visible surface; a cold one reads the IR-colored
        // overlay. The composite carries BOTH looks (the sandwich's whole point).
        let vis = vec![
            30, 120, 40, 255, // px0 warm: green land (visible)
            240, 240, 240, 255, // px1 cold-top: bright visible cloud
        ];
        let ir = vec![
            10, 10, 10, // px0 IR (warm, unused — alpha 0)
            180, 30, 0, // px1 IR (cold-top red)
        ];
        let bt = vec![285.0f32, 200.0f32]; // warm surface, then a cold overshoot
        let (rgb, _) = blend_rgba(&vis, &ir, &bt, 2);
        assert_eq!(
            &rgb[0..3],
            &[30, 120, 40],
            "warm pixel is the visible surface"
        );
        // The cold pixel is reddened by the IR overlay (R is now clearly the top channel,
        // which it was NOT in the neutral-white visible).
        assert!(
            rgb[3] > rgb[4] && rgb[3] > rgb[5],
            "cold pixel not IR-reddened: {:?}",
            &rgb[3..6]
        );
    }

    #[test]
    fn space_pixels_stay_black_and_transparent() {
        // A space pixel (visible alpha 0) must stay black + transparent regardless of the IR
        // color or the BT.
        let vis = vec![9, 9, 9, 0]; // space: alpha 0
        let ir = vec![255, 255, 255]; // a bright IR value that must NOT leak into space
        let bt = vec![190.0f32]; // a cold BT (alpha would be max) — still must not apply
        let (rgb, rgba) = blend_rgba(&vis, &ir, &bt, 1);
        assert_eq!(&rgb[0..3], &[0, 0, 0], "space stays black");
        assert_eq!(rgba[3], 0, "space stays transparent");
    }
}
