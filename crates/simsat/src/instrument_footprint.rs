//! Experimental instrument spatial-response operators.
//!
//! This module is deliberately separate from the display-only top-down cloud
//! feather.  A channel footprint acts on the complete, unclipped channel radiance
//! before brightness-temperature inversion or any display enhancement.
//!
//! The ABI prototype is an angular-grid operator. ABI samples are uniformly spaced
//! in fixed scan angle, not uniformly spaced in kilometres on Earth.  The operator
//! requires the exact GOES-R sweep-x navigation and the globally anchored ABI 2-km
//! lattice: 56-urad pitch with the sub-satellite point at the corner of four pixels.
//! Sweep-x ellipsoid navigation therefore supplies the increasing surface footprint
//! away from the sub-satellite point; no additional empirical `sec(view_angle)` blur
//! is added.

use rayon::prelude::*;

/// NOAA GOES-16 ABI L1b/CMI Full Validation Product Performance Guide.
pub const ABI_MTF_VALIDATION_URL: &str = "https://www.ospo.noaa.gov/operations/goes/\
product-quality-overview/ps-pvr/goes-16/ABI/Cloud%20and%20Moisture%20Imagery/Full/\
GOES-16_ABI-L1b-CMI_Full-Validation_ProductPerformanceGuide_v2.pdf";

/// ABI Band 13 nominal fixed-grid sample spacing (56 microradians, 2 km at SSP).
pub const ABI_BAND13_SAMPLE_ANGLE_URAD: f64 = 56.0;

/// Measured GOES-16 ABI Band 13 EW MTF at 1/4, 1/2, 3/4 and 1 times Nyquist.
/// Tables 19 and 20 report the same Full Validation values for falling and rising
/// lunar edges.  Band 13 NS values were unavailable, so the prototype uses the EW
/// fit separably in both axes and reports that limitation.
pub const ABI_BAND13_EW_MTF_FULL: [f64; 4] = [0.93, 0.74, 0.50, 0.28];

/// Non-negative, energy-preserving three-tap fit used by the prototype.
///
/// A positive discrete kernel cannot match all four published MTF samples exactly
/// (matching the unusually low Nyquist value while retaining the half-Nyquist value
/// requires signed resampling lobes).  `[0.15, 0.70, 0.15]` is therefore a bounded
/// radiance-space approximation that follows the reliable mid-band response without
/// introducing negative radiance or ringing.
pub const ABI_BAND13_FIR: [f64; 3] = [0.15, 0.70, 0.15];

/// Optional channel spatial-response stage.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InstrumentFootprint {
    /// No instrument footprint.  Exact pre-experiment behavior.
    #[default]
    Off,
    /// Experimental GOES-R ABI Band 13 angular-grid MTF approximation.
    GoesRAbiBand13Mtf,
}

/// Stable footprint metadata for provenance-bearing interfaces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InstrumentFootprintMetadata {
    pub slug: &'static str,
    pub label: &'static str,
    pub channel: Option<&'static str>,
    pub domain: &'static str,
    pub sample_angle_urad: Option<f64>,
    pub source_url: Option<&'static str>,
    pub limitation: Option<&'static str>,
}

impl InstrumentFootprint {
    pub const ALL: [Self; 2] = [Self::Off, Self::GoesRAbiBand13Mtf];

    pub const fn metadata(self) -> InstrumentFootprintMetadata {
        match self {
            Self::Off => InstrumentFootprintMetadata {
                slug: "off",
                label: "Off",
                channel: None,
                domain: "none",
                sample_angle_urad: None,
                source_url: None,
                limitation: None,
            },
            Self::GoesRAbiBand13Mtf => InstrumentFootprintMetadata {
                slug: "goes-r-abi-band13-mtf-prototype",
                label: "GOES-16-informed ABI Band 13 MTF prototype",
                channel: Some("ABI Band 13 (10.3 um)"),
                domain: "complete band radiance on an exact global-lattice-snapped ABI 56-urad angular crop",
                sample_angle_urad: Some(ABI_BAND13_SAMPLE_ANGLE_URAD),
                source_url: Some(ABI_MTF_VALIDATION_URL),
                limitation: Some(
                    "experimental positive three-tap approximation to measured GOES-16 ABI Band 13 EW MTF, transferred to GOES-19/FM4 as an unvalidated flight-model hypothesis; Band 13 NS values were unavailable; temporal integration and detector variation are not modeled; the crop edge and every one-pixel invalid-mask perimeter are emitted as no-data and excluded from sensor-validation metrics",
                ),
            },
        }
    }

    pub const fn slug(self) -> &'static str {
        self.metadata().slug
    }

    pub const fn label(self) -> &'static str {
        self.metadata().label
    }

    pub fn parse(token: &str) -> Option<Self> {
        match token.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "disabled" | "false" => Some(Self::Off),
            "goes-r-abi-band13-mtf-prototype"
            | "goes-r-abi-band13-mtf"
            | "abi-band13-mtf"
            | "abi13-mtf"
            | "band13-mtf" => Some(Self::GoesRAbiBand13Mtf),
            _ => None,
        }
    }

    /// Spatial frequency response of the prototype one-dimensional FIR at a
    /// fraction of Nyquist (`0` = DC, `1` = Nyquist).
    pub fn mtf(self, nyquist_fraction: f64) -> f64 {
        match self {
            Self::Off => 1.0,
            Self::GoesRAbiBand13Mtf => {
                let phase = std::f64::consts::PI * nyquist_fraction.clamp(0.0, 1.0);
                ABI_BAND13_FIR[1] + 2.0 * ABI_BAND13_FIR[0] * phase.cos()
            }
        }
    }
}

/// Apply the ABI Band 13 prototype to a scalar radiance plane.
///
/// `NaN` is the invalid/space mask and remains `NaN`.  Missing neighbours retain
/// their tap weight on the centre sample.  Because the side taps are symmetric,
/// every one-dimensional pass is doubly stochastic over valid samples: constants
/// and the global finite radiance sum are preserved (to floating-point roundoff),
/// including at cropped-domain and space boundaries. This reflective one-pixel
/// boundary treatment is numerical, not an instrument model; sensor-validation
/// metrics must exclude valid pixels touching the invalid mask.
pub fn apply_band13_radiance_footprint(radiance: &[f64], nx: usize, ny: usize) -> Vec<f64> {
    assert_eq!(radiance.len(), nx * ny);
    let side = ABI_BAND13_FIR[0];
    let center = ABI_BAND13_FIR[1];

    let horizontal_rows: Vec<Vec<f64>> = (0..ny)
        .into_par_iter()
        .map(|y| {
            let mut row = vec![f64::NAN; nx];
            for (x, out) in row.iter_mut().enumerate() {
                let idx = y * nx + x;
                let value = radiance[idx];
                if !value.is_finite() {
                    continue;
                }
                let left = x
                    .checked_sub(1)
                    .map(|xx| radiance[y * nx + xx])
                    .filter(|v| v.is_finite())
                    .unwrap_or(value);
                let right = (x + 1 < nx)
                    .then(|| radiance[y * nx + x + 1])
                    .filter(|v| v.is_finite())
                    .unwrap_or(value);
                *out = side * left + center * value + side * right;
            }
            row
        })
        .collect();
    let horizontal: Vec<f64> = horizontal_rows.into_iter().flatten().collect();

    let rows: Vec<Vec<f64>> = (0..ny)
        .into_par_iter()
        .map(|y| {
            let mut row = vec![f64::NAN; nx];
            for (x, out) in row.iter_mut().enumerate() {
                let idx = y * nx + x;
                let value = horizontal[idx];
                if !value.is_finite() {
                    continue;
                }
                let above = y
                    .checked_sub(1)
                    .map(|yy| horizontal[yy * nx + x])
                    .filter(|v| v.is_finite())
                    .unwrap_or(value);
                let below = (y + 1 < ny)
                    .then(|| horizontal[(y + 1) * nx + x])
                    .filter(|v| v.is_finite())
                    .unwrap_or(value);
                *out = side * above + center * value + side * below;
            }
            row
        })
        .collect();
    rows.into_iter().flatten().collect()
}

/// Footprint output plus the explicit sensor-validation mask.
///
/// `validation_mask[idx] == 1` only when the complete 3x3 FIR support exists and
/// is finite. Crop borders and the one-pixel perimeter around any invalid/space
/// sample are zero. The matching output radiance is `NaN` there, so ordinary
/// finite-pixel metrics cannot accidentally score the numerical reflection used
/// to make a visually complete convolution.
#[derive(Debug, Clone, PartialEq)]
pub struct Band13FootprintOutput {
    pub radiance: Vec<f64>,
    pub validation_mask: Vec<u8>,
    /// Finite input samples deliberately changed to no-data because their FIR
    /// support touches a crop or invalid-mask boundary.
    pub excluded_finite_samples: usize,
}

/// Apply the Band 13 footprint and enforce its honest validation perimeter.
pub fn apply_band13_radiance_footprint_validated(
    radiance: &[f64],
    nx: usize,
    ny: usize,
) -> Band13FootprintOutput {
    assert_eq!(radiance.len(), nx * ny);
    let mut filtered = apply_band13_radiance_footprint(radiance, nx, ny);
    let mut validation_mask = vec![0_u8; nx * ny];
    let mut excluded_finite_samples = 0usize;
    for y in 0..ny {
        for x in 0..nx {
            let idx = y * nx + x;
            let valid = x > 0
                && x + 1 < nx
                && y > 0
                && y + 1 < ny
                && ((y - 1)..=(y + 1))
                    .all(|yy| ((x - 1)..=(x + 1)).all(|xx| radiance[yy * nx + xx].is_finite()));
            if valid {
                validation_mask[idx] = 1;
            } else {
                if radiance[idx].is_finite() {
                    excluded_finite_samples += 1;
                }
                filtered[idx] = f64::NAN;
            }
        }
    }
    Band13FootprintOutput {
        radiance: filtered,
        validation_mask,
        excluded_finite_samples,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finite_sum(values: &[f64]) -> f64 {
        values.iter().copied().filter(|v| v.is_finite()).sum()
    }

    #[test]
    fn default_is_exactly_off_and_registry_is_parseable() {
        assert_eq!(InstrumentFootprint::default(), InstrumentFootprint::Off);
        for mode in InstrumentFootprint::ALL {
            assert_eq!(InstrumentFootprint::parse(mode.slug()), Some(mode));
        }
        assert_eq!(InstrumentFootprint::parse("not-a-footprint"), None);
    }

    #[test]
    fn band13_kernel_is_positive_normalized_and_tracks_noaa_midband_mtf() {
        assert!(ABI_BAND13_FIR.iter().all(|&w| w >= 0.0));
        assert!((ABI_BAND13_FIR.iter().sum::<f64>() - 1.0).abs() < 1.0e-15);
        let mode = InstrumentFootprint::GoesRAbiBand13Mtf;
        let fitted = [0.25, 0.50, 0.75, 1.0].map(|f| mode.mtf(f));
        let rmse = (fitted
            .iter()
            .zip(ABI_BAND13_EW_MTF_FULL)
            .map(|(fit, target)| (fit - target).powi(2))
            .sum::<f64>()
            / fitted.len() as f64)
            .sqrt();
        // The bounded positive fit follows the reliable mid-band values closely;
        // its documented largest mismatch is the noisy Nyquist measurement.
        assert!((fitted[1] - ABI_BAND13_EW_MTF_FULL[1]).abs() < 0.05);
        assert!((fitted[2] - ABI_BAND13_EW_MTF_FULL[2]).abs() < 0.02);
        assert!((rmse - 0.0644).abs() < 0.001);
        assert!(fitted.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn footprint_preserves_constant_energy_and_invalid_mask() {
        let (nx, ny) = (9, 7);
        let mut input = vec![3.25; nx * ny];
        for idx in [0, 8, 27, 54, 62] {
            input[idx] = f64::NAN;
        }
        let got = apply_band13_radiance_footprint(&input, nx, ny);
        for (before, after) in input.iter().zip(&got) {
            if before.is_finite() {
                assert!((*after - *before).abs() < 1.0e-12);
            } else {
                assert!(after.is_nan());
            }
        }
        assert!((finite_sum(&got) - finite_sum(&input)).abs() < 1.0e-12);
    }

    #[test]
    fn footprint_impulse_is_separable_kernel_and_energy_preserving() {
        let (nx, ny) = (9, 9);
        let mut impulse = vec![0.0; nx * ny];
        impulse[4 * nx + 4] = 1.0;
        let got = apply_band13_radiance_footprint(&impulse, nx, ny);
        for dy in -1isize..=1 {
            for dx in -1isize..=1 {
                let idx = (4usize.checked_add_signed(dy).unwrap()) * nx
                    + 4usize.checked_add_signed(dx).unwrap();
                let expected =
                    ABI_BAND13_FIR[(dx + 1) as usize] * ABI_BAND13_FIR[(dy + 1) as usize];
                assert!((got[idx] - expected).abs() < 1.0e-15);
            }
        }
        assert!((finite_sum(&got) - 1.0).abs() < 1.0e-15);
    }

    #[test]
    fn validated_footprint_excludes_crop_and_invalid_mask_perimeters() {
        let (nx, ny) = (7, 7);
        let mut input = vec![2.0; nx * ny];
        input[3 * nx + 3] = f64::NAN;
        let got = apply_band13_radiance_footprint_validated(&input, nx, ny);
        assert_eq!(got.radiance.len(), input.len());
        assert_eq!(got.validation_mask.len(), input.len());
        for y in 0..ny {
            for x in 0..nx {
                let idx = y * nx + x;
                let touches_crop = x == 0 || x + 1 == nx || y == 0 || y + 1 == ny;
                let touches_hole = x.abs_diff(3) <= 1 && y.abs_diff(3) <= 1;
                if touches_crop || touches_hole {
                    assert_eq!(got.validation_mask[idx], 0, "({x}, {y})");
                    assert!(got.radiance[idx].is_nan(), "({x}, {y})");
                } else {
                    assert_eq!(got.validation_mask[idx], 1, "({x}, {y})");
                    assert!((got.radiance[idx] - 2.0).abs() < 1.0e-12);
                }
            }
        }
        assert_eq!(got.excluded_finite_samples, 32);
    }
}
