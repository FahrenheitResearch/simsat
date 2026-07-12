//! Thermal sensor spectral-response registry.
//!
//! The shipped thermal renderer historically evaluated Planck emission at one
//! centre wavelength.  [`ThermalSensor::FastGray`] preserves that byte-for-byte
//! fast path.  [`ThermalSensor::GoesRAbiBand13Fm4`] is an opt-in ABI Band 13
//! observation-operator prototype: it integrates Planck spectral radiance over
//! the official NOAA/NESDIS GOES-R ABI FM4 (GOES-19) channel-13 spectral response
//! and inverts the same monotone band response to brightness temperature.
//!
//! Only the source function and radiance-to-BT conversion are spectral here. The
//! current cloud and gas absorption coefficients remain band-gray; callers must
//! surface [`ThermalSensor::limitation_warning`] with ABI-mode results.

use std::sync::OnceLock;

use crate::optics::{PLANCK_C1L, PLANCK_C2, inverse_planck, planck_radiance};

/// Official NOAA/NESDIS release page for all four pre-launch ABI flight-model SRFs.
pub const ABI_FM4_SRF_RELEASE_URL: &str = "https://ncc.nesdis.noaa.gov/GOESR/ABI.php";
/// Direct NOAA/NESDIS archive containing the FM4 channel files.
pub const ABI_FM4_SRF_ARCHIVE_URL: &str =
    "https://ncc.nesdis.noaa.gov/GOESR/docs/GOES-R_ABI_FM4_SRF_CWG.zip";
/// SHA-256 of the downloaded NOAA ZIP (retrieved 2026-07-11).
pub const ABI_FM4_SRF_ARCHIVE_SHA256: &str =
    "B1482058CD63481F55E523C565A982BB2787F01088960E2833A1E4BF6286DD17";
/// SHA-256 of the NOAA channel-13 text file as distributed (CRLF bytes).
pub const ABI_FM4_CH13_SOURCE_SHA256: &str =
    "285D8D6261EAAF8653734B602160BECDA23AA51F95A4907C243FCD4CA28FE07E";
/// SHA-256 of the vendored file after the sole transformation CRLF -> LF.
pub const ABI_FM4_CH13_VENDORED_SHA256: &str =
    "4E14BD906C4B8FD2D97569C1AC502D6E32ACE5C35B9DDA76513D5E0067FA567A";

/// Current thermal source/response model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ThermalSensor {
    /// Existing centre-wavelength gray source function and inverse Planck.
    /// This remains the default and preserves prior output.
    #[default]
    FastGray,
    /// Official GOES-R ABI FM4 (GOES-19) Band 13 spectral response.
    GoesRAbiBand13Fm4,
}

/// Stable registry metadata suitable for CLI, Python, GUI, and provenance output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThermalSensorMetadata {
    pub slug: &'static str,
    pub label: &'static str,
    pub response: &'static str,
    pub radiance_units: &'static str,
    pub source_url: Option<&'static str>,
    pub source_sha256: Option<&'static str>,
}

impl ThermalSensor {
    pub const ALL: [Self; 2] = [Self::FastGray, Self::GoesRAbiBand13Fm4];

    pub const fn metadata(self) -> ThermalSensorMetadata {
        match self {
            Self::FastGray => ThermalSensorMetadata {
                slug: "fast-gray",
                label: "Fast gray (center wavelength)",
                response: "single center wavelength",
                radiance_units: "W m^-3 sr^-1",
                source_url: None,
                source_sha256: None,
            },
            Self::GoesRAbiBand13Fm4 => ThermalSensorMetadata {
                slug: "goes-r-abi-band13-fm4",
                label: "GOES-R ABI Band 13 (FM4/GOES-19 SRF)",
                response: "NOAA CWG relative SRF, normalized in wavenumber",
                radiance_units: "mW m^-2 sr^-1 (cm^-1)^-1",
                source_url: Some(ABI_FM4_SRF_ARCHIVE_URL),
                source_sha256: Some(ABI_FM4_CH13_SOURCE_SHA256),
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
            "fast-gray" | "fast" | "gray" | "legacy" | "center" | "center-wavelength" => {
                Some(Self::FastGray)
            }
            "goes-r-abi-band13-fm4"
            | "goes-r-abi-band13"
            | "abi-band13"
            | "abi13"
            | "band13-srf"
            | "fm4"
            | "goes19"
            | "goes-19" => Some(Self::GoesRAbiBand13Fm4),
            _ => None,
        }
    }

    /// Required honesty warning for the current ABI prototype.
    pub const fn limitation_warning(self) -> Option<&'static str> {
        match self {
            Self::FastGray => None,
            Self::GoesRAbiBand13Fm4 => Some(
                "ABI Band 13 SRF is applied to Planck emission and BT inversion, but cloud and water-vapor absorption remain SimSat's gray 10.3 um approximation; this is not yet a line-by-line ABI observation operator.",
            ),
        }
    }

    /// Emitted source radiance in this response model's native spectral-radiance units.
    #[inline]
    pub fn source_radiance(self, temperature_k: f64, center_wavelength_m: f64) -> f64 {
        match self {
            Self::FastGray => planck_radiance(temperature_k, center_wavelength_m),
            Self::GoesRAbiBand13Fm4 => abi13_band_radiance(temperature_k),
        }
    }

    /// Invert this response model's effective radiance to brightness temperature.
    #[inline]
    pub fn brightness_temperature(self, radiance: f64, center_wavelength_m: f64) -> f64 {
        match self {
            Self::FastGray => inverse_planck(radiance, center_wavelength_m),
            Self::GoesRAbiBand13Fm4 => inverse_abi13_band_radiance(radiance),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SrfSample {
    wavenumber_cm1: f64,
    response: f64,
}

const ABI13_FM4_SRF_TEXT: &str = include_str!("../assets/abi_srf/GOES-R_ABI_FM4_SRF_CWG_ch13.txt");

fn abi13_samples() -> &'static [SrfSample] {
    static SAMPLES: OnceLock<Vec<SrfSample>> = OnceLock::new();
    SAMPLES.get_or_init(|| {
        let samples: Vec<_> = ABI13_FM4_SRF_TEXT
            .lines()
            .filter(|line| !line.trim_start().starts_with('#') && !line.trim().is_empty())
            .map(|line| {
                let mut fields = line.split_whitespace();
                let _wavelength_um: f64 = fields
                    .next()
                    .expect("ABI SRF wavelength")
                    .parse()
                    .expect("numeric ABI SRF wavelength");
                let wavenumber_cm1 = fields
                    .next()
                    .expect("ABI SRF wavenumber")
                    .parse()
                    .expect("numeric ABI SRF wavenumber");
                let response = fields
                    .next()
                    .expect("ABI SRF response")
                    .parse()
                    .expect("numeric ABI SRF response");
                SrfSample {
                    wavenumber_cm1,
                    response,
                }
            })
            .collect();
        assert!(samples.len() > 1000, "truncated ABI Band 13 SRF asset");
        samples
    })
}

/// Planck spectral radiance per cm^-1 in the ABI/NOAA emissive-band convention:
/// mW m^-2 sr^-1 (cm^-1)^-1. `wavenumber_cm1` is in cm^-1.
#[inline]
fn planck_wavenumber_radiance(temperature_k: f64, wavenumber_cm1: f64) -> f64 {
    if temperature_k <= 0.0 || wavenumber_cm1 <= 0.0 {
        return 0.0;
    }
    let nu_m1 = 100.0 * wavenumber_cm1;
    let expo = PLANCK_C2 * nu_m1 / temperature_k;
    let denom = expo.exp_m1();
    if denom <= 0.0 || !denom.is_finite() {
        return 0.0;
    }
    // B per m^-1 -> per cm^-1 (*100), W -> mW (*1000).
    PLANCK_C1L * nu_m1.powi(3) / denom * 100_000.0
}

fn abi13_band_radiance_exact(temperature_k: f64) -> f64 {
    if temperature_k <= 0.0 {
        return 0.0;
    }
    let samples = abi13_samples();
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for pair in samples.windows(2) {
        let a = pair[0];
        let b = pair[1];
        let dx = (b.wavenumber_cm1 - a.wavenumber_cm1).abs();
        let ba = planck_wavenumber_radiance(temperature_k, a.wavenumber_cm1);
        let bb = planck_wavenumber_radiance(temperature_k, b.wavenumber_cm1);
        numerator += 0.5 * (ba * a.response + bb * b.response) * dx;
        denominator += 0.5 * (a.response + b.response) * dx;
    }
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

const LUT_MIN_K: f64 = 100.0;
const LUT_MAX_K: f64 = 400.0;
const LUT_STEP_K: f64 = 0.25;
const LUT_LEN: usize = ((LUT_MAX_K - LUT_MIN_K) / LUT_STEP_K) as usize + 1;

fn abi13_lut() -> &'static [f64] {
    static LUT: OnceLock<Vec<f64>> = OnceLock::new();
    LUT.get_or_init(|| {
        (0..LUT_LEN)
            .map(|i| abi13_band_radiance_exact(LUT_MIN_K + i as f64 * LUT_STEP_K))
            .collect()
    })
}

fn abi13_band_radiance(temperature_k: f64) -> f64 {
    if temperature_k <= 0.0 {
        return 0.0;
    }
    if !(LUT_MIN_K..=LUT_MAX_K).contains(&temperature_k) {
        return abi13_band_radiance_exact(temperature_k);
    }
    let x = (temperature_k - LUT_MIN_K) / LUT_STEP_K;
    let i = (x.floor() as usize).min(LUT_LEN - 2);
    let f = x - i as f64;
    let lut = abi13_lut();
    lut[i] + f * (lut[i + 1] - lut[i])
}

fn inverse_abi13_band_radiance(radiance: f64) -> f64 {
    if radiance <= 0.0 {
        return 0.0;
    }
    let lut = abi13_lut();
    if radiance < lut[0] || radiance > lut[LUT_LEN - 1] {
        // Stable monotone bisection outside the normal atmospheric LUT range.
        let mut lo: f64 = 1.0;
        let mut hi: f64 = 1000.0;
        for _ in 0..64 {
            let mid = 0.5 * (lo + hi);
            if abi13_band_radiance_exact(mid) < radiance {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return 0.5 * (lo + hi);
    }
    let upper = lut.partition_point(|&v| v < radiance).min(LUT_LEN - 1);
    if upper == 0 {
        return LUT_MIN_K;
    }
    let lower = upper - 1;
    let span = lut[upper] - lut[lower];
    let f = if span > 0.0 {
        (radiance - lut[lower]) / span
    } else {
        0.0
    };
    LUT_MIN_K + (lower as f64 + f) * LUT_STEP_K
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optics::IR_BAND13_WAVELENGTH_M;

    #[test]
    fn official_asset_is_complete_and_monotone_in_wavenumber() {
        let s = abi13_samples();
        assert_eq!(s.len(), 1057);
        assert!(
            s.windows(2)
                .all(|p| p[1].wavenumber_cm1 < p[0].wavenumber_cm1)
        );
        assert!(s.iter().all(|p| p.response >= 0.0 && p.response <= 1.1));
    }

    #[test]
    fn abi13_response_is_monotone_and_round_trips_180_to_330_k() {
        let sensor = ThermalSensor::GoesRAbiBand13Fm4;
        let mut previous = 0.0;
        for t in (180..=330).map(|t| t as f64) {
            let l = sensor.source_radiance(t, IR_BAND13_WAVELENGTH_M);
            assert!(l > previous, "non-monotone at {t} K: {l} <= {previous}");
            let bt = sensor.brightness_temperature(l, IR_BAND13_WAVELENGTH_M);
            assert!((bt - t).abs() < 1.0e-6, "round trip {t} K -> {bt} K");
            previous = l;
        }
    }

    #[test]
    fn official_bandpass_differs_from_center_wavelength_for_mixed_scene() {
        // A sub-pixel warm/cold mixture is non-Planckian. A finite-width response and
        // the old centre-wavelength response must therefore retrieve different BTs.
        let abi = ThermalSensor::GoesRAbiBand13Fm4;
        let fast = ThermalSensor::FastGray;
        let mix_abi = 0.55 * abi.source_radiance(205.0, IR_BAND13_WAVELENGTH_M)
            + 0.45 * abi.source_radiance(302.0, IR_BAND13_WAVELENGTH_M);
        let mix_fast = 0.55 * fast.source_radiance(205.0, IR_BAND13_WAVELENGTH_M)
            + 0.45 * fast.source_radiance(302.0, IR_BAND13_WAVELENGTH_M);
        let bt_abi = abi.brightness_temperature(mix_abi, IR_BAND13_WAVELENGTH_M);
        let bt_fast = fast.brightness_temperature(mix_fast, IR_BAND13_WAVELENGTH_M);
        assert!(
            (bt_abi - bt_fast).abs() > 0.005,
            "responses collapsed: {bt_abi} vs {bt_fast}"
        );
        assert!((bt_abi - bt_fast).abs() < 2.0, "implausible response delta");
    }

    #[test]
    fn default_and_parser_preserve_fast_gray() {
        assert_eq!(ThermalSensor::default(), ThermalSensor::FastGray);
        assert_eq!(
            ThermalSensor::parse("fast-gray"),
            Some(ThermalSensor::FastGray)
        );
        assert_eq!(
            ThermalSensor::parse("abi-band13"),
            Some(ThermalSensor::GoesRAbiBand13Fm4)
        );
        assert_eq!(ThermalSensor::parse("not-a-sensor"), None);
    }
}
