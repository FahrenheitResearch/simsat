//! GeoColor day/night blend (the GOES flagship product): true-color visible by day,
//! colored infrared by night, crossfaded across the terminator by the PER-PIXEL solar
//! elevation.
//!
//! This module holds the pure, node-testable blend math + the documented thresholds. The
//! render orchestration (rendering the visible frame and the IR frame through the SAME
//! `api::render` paths and feeding this blend) lives in [`crate::api`] under
//! [`crate::api::Product::GeoColor`]; the studio reuses [`blend_rgba`] directly over its
//! already-rendered visible composite + a band-13 IR march.
//!
//! ## Thresholds (documented)
//! - solar elevation `>= `[`GEOCOLOR_DAY_ELEV_DEG`]` (+5 deg)` -> 100% visible true-color.
//! - solar elevation `<= `[`GEOCOLOR_NIGHT_ELEV_DEG`]` (-6 deg)` -> 100% colored IR.
//! - in the twilight band (+5 .. -6 deg) -> a smoothstep crossfade, so the terminator is a
//!   graceful hand-off rather than a hard line. See [`day_weight`].
//!
//! ## Night look (honest)
//! The night side is the ABI band-13 (10.3 um) IR through the [`GEOCOLOR_NIGHT_ENHANCEMENT`]
//! (Grayscale: cold cloud tops white, warm surface dark) — real GeoColor shows clouds in IR
//! at night, and this is the clean, faithful depiction of that. We have NO city-lights /
//! night-earth texture (no data), so OUR GeoColor night is purely the colored IR — documented
//! honestly. A colored night-cloud enhancement is a trivial future swap (this one constant).

use crate::ir_enhance::IrEnhancement;

/// Solar elevation (deg) at/above which GeoColor is 100% visible true-color (day).
pub const GEOCOLOR_DAY_ELEV_DEG: f64 = 5.0;
/// Solar elevation (deg) at/below which GeoColor is 100% colored IR (night).
pub const GEOCOLOR_NIGHT_ELEV_DEG: f64 = -6.0;
/// The IR band-13 enhancement used for the GeoColor night side: a clean grayscale (cold
/// cloud tops white, warm surface dark). Honest depiction of "clouds in IR at night"; no
/// city lights (no data). Swappable to a colored night-cloud enhancement in one place.
pub const GEOCOLOR_NIGHT_ENHANCEMENT: IrEnhancement = IrEnhancement::Grayscale;

/// The VISIBLE weight for a pixel at solar elevation `elev_deg`: a smoothstep crossfade from
/// 0 (night, full colored IR) at/below [`GEOCOLOR_NIGHT_ELEV_DEG`] to 1 (day, full visible
/// true-color) at/above [`GEOCOLOR_DAY_ELEV_DEG`]. Monotone increasing through the twilight
/// band; the smoothstep gives a graceful (zero-slope-at-the-ends) hand-off at the terminator.
#[inline]
pub fn day_weight(elev_deg: f64) -> f64 {
    let t = ((elev_deg - GEOCOLOR_NIGHT_ELEV_DEG)
        / (GEOCOLOR_DAY_ELEV_DEG - GEOCOLOR_NIGHT_ELEV_DEG))
        .clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Blend a visible true-color frame and a colored-IR frame into the GeoColor day/night
/// composite, per pixel by solar elevation.
///
/// - `vis_rgba` is the finished true-color frame, `n*4` bytes; its alpha is the on-earth mask
///   (alpha 0 = space -> the output stays black + transparent there regardless of the IR).
/// - `ir_rgb` is the colored-IR frame, `n*3` bytes, with out-of-domain pixels rendered black
///   (as [`crate::api::rgba_to_rgb_black_space`] produces). At night those become black, which
///   is correct — there is no IR data outside the WRF domain.
/// - `elev_at(i)` returns the solar elevation (deg) at pixel `i` — the same per-pixel elevation
///   the visible pass used to light the terminator (`build_luts` `l[3]`).
///
/// Returns `(rgb, rgba)`: the `n*3` RGB (space = black) the store writer / display uses, and
/// the `n*4` RGBA carrying the visible on-earth alpha (space alpha 0).
pub fn blend_rgba(
    vis_rgba: &[u8],
    ir_rgb: &[u8],
    n: usize,
    elev_at: impl Fn(usize) -> f64,
) -> (Vec<u8>, Vec<u8>) {
    assert!(vis_rgba.len() >= n * 4, "vis_rgba too short");
    assert!(ir_rgb.len() >= n * 3, "ir_rgb too short");
    let mut rgb = vec![0u8; n * 3];
    let mut rgba = vec![0u8; n * 4];
    for i in 0..n {
        let a = vis_rgba[i * 4 + 3];
        if a == 0 {
            // Space: black + transparent, matching the visible/IR off-earth convention.
            continue;
        }
        let w = day_weight(elev_at(i));
        for c in 0..3 {
            let v = w * vis_rgba[i * 4 + c] as f64 + (1.0 - w) * ir_rgb[i * 3 + c] as f64;
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
    fn day_weight_is_smoothstep_one_day_zero_night_monotone() {
        // Full day at/above +5 -> 1.0; full night at/below -6 -> 0.0 (the documented ends).
        assert_eq!(day_weight(GEOCOLOR_DAY_ELEV_DEG), 1.0);
        assert_eq!(day_weight(90.0), 1.0);
        assert_eq!(day_weight(GEOCOLOR_NIGHT_ELEV_DEG), 0.0);
        assert_eq!(day_weight(-90.0), 0.0);
        // The twilight band crossfades monotonically from 0 to 1, strictly between the ends.
        let span = GEOCOLOR_DAY_ELEV_DEG - GEOCOLOR_NIGHT_ELEV_DEG;
        let mut prev = -1.0;
        for k in 0..=22 {
            let e = GEOCOLOR_NIGHT_ELEV_DEG + span * (k as f64 / 22.0);
            let w = day_weight(e);
            assert!((0.0..=1.0).contains(&w), "weight {w} out of [0,1] at {e}");
            assert!(w >= prev - 1e-12, "not monotone at {e}: {w} < {prev}");
            prev = w;
        }
        // The interior of the band is a genuine partial blend (neither pinned end).
        let mid = day_weight((GEOCOLOR_DAY_ELEV_DEG + GEOCOLOR_NIGHT_ELEV_DEG) * 0.5);
        assert!(
            mid > 0.0 && mid < 1.0,
            "twilight midpoint {mid} not a blend"
        );
    }

    #[test]
    fn full_day_pixel_is_visible_full_night_pixel_is_ir() {
        // Two on-earth pixels with distinct visible + IR colours.
        let vis = vec![
            10, 20, 30, 255, // px0 visible
            40, 50, 60, 255, // px1 visible
        ];
        let ir = vec![
            200, 100, 50, // px0 IR
            70, 80, 90, // px1 IR
        ];
        // px0 fully day (+90) -> the visible colour; px1 fully night (-90) -> the IR colour.
        let (rgb, rgba) = blend_rgba(&vis, &ir, 2, |i| if i == 0 { 90.0 } else { -90.0 });
        assert_eq!(&rgb[0..3], &[10, 20, 30], "day pixel should equal visible");
        assert_eq!(&rgb[3..6], &[70, 80, 90], "night pixel should equal IR");
        // The RGBA carries the visible on-earth alpha.
        assert_eq!(rgba[3], 255);
        assert_eq!(rgba[7], 255);
    }

    #[test]
    fn mixed_terminator_frame_contains_visible_and_ir_dominated_pixels() {
        // Four pixels: two full-day (visible bright), two full-night (IR). The blended
        // frame must contain BOTH a visible-dominated pixel and an IR-dominated one.
        let vis = vec![
            250, 250, 250, 255, // bright day surface
            250, 250, 250, 255, //
            5, 5, 5, 255, // dark night visible (near-black, as a real night scene)
            5, 5, 5, 255, //
        ];
        let ir = vec![
            10, 10, 10, // day IR (unused, weight 1)
            10, 10, 10, //
            220, 220, 220, // cold cloud top (bright IR at night)
            220, 220, 220, //
        ];
        let (rgb, _) = blend_rgba(&vis, &ir, 4, |i| if i < 2 { 30.0 } else { -30.0 });
        // Day pixels read the bright visible surface; night pixels read the bright IR cloud.
        assert_eq!(
            &rgb[0..3],
            &[250, 250, 250],
            "day pixel is the visible surface"
        );
        assert_eq!(&rgb[6..9], &[220, 220, 220], "night pixel is the IR cloud");
        // Concretely: some pixel is visible-dominated, some IR-dominated (not all identical).
        let day_bright = rgb[0] as i32;
        let night_bright = rgb[6] as i32;
        assert!(day_bright > 200 && night_bright > 200);
        // The night visible was near-black (5); that it is now bright proves IR took over.
        assert!(
            night_bright > 100,
            "night side should show the storm in IR, not black"
        );
    }

    #[test]
    fn space_pixels_stay_black_and_transparent() {
        // A space pixel (visible alpha 0) must stay black + transparent regardless of the IR
        // (and regardless of the blend weight).
        let vis = vec![9, 9, 9, 0]; // space: alpha 0
        let ir = vec![255, 255, 255]; // a bright IR value that must NOT leak into space
        let (rgb, rgba) = blend_rgba(&vis, &ir, 1, |_| -30.0); // night weight
        assert_eq!(&rgb[0..3], &[0, 0, 0], "space stays black");
        assert_eq!(rgba[3], 0, "space stays transparent");
    }
}
