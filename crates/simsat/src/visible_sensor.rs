//! Official ABI FM4 reflective-band response quadrature; transport is NOT implemented.
//!
//! This registry supplies the NOAA C01/C02/C03 response curves and their radiometric
//! integration contract. It is not connected to RenderIntent or any image product.
//! Feeding the existing broad RGB radiances into it would NOT produce ABI channels.
//!
//! Wavelength quadrature follows the reflective-band units in NOAA's Cloud and
//! Moisture Imagery ATBD v4, section 3.4.1.2: L_lambda in W m^-2 sr^-1 um^-1,
//! E_lambda in W m^-2 um^-1, and reflectance factor rho_f = pi L_band / E_band.
//! E must be at the observation's Earth-Sun distance; the 1-AU convenience method
//! explicitly applies the inverse-square correction. There is no division by mu0,
//! display shaping, or clipping. A Lambertian surface without atmosphere therefore
//! yields rho_f = albedo * mu0, and directional glint can exceed one.
//!
//! Primary sources and the exact source/vendored hashes are also documented in
//! assets/abi_srf/README.md. The existing thermal registry owns the shared NOAA ZIP
//! provenance. Only CRLF -> LF was applied to the three new text assets.
//!
//! SCIENCE WARNING: a future caller must supply wavelength-resolved TOA radiance
//! and solar irradiance. SimSat currently lacks calibrated spectral surface
//! albedo/BRDF (especially C03 near-IR), spectral gas/cloud optics, and a validated
//! spectral transport/ABI-footprint path. This module fills none of those gaps.

use std::f64::consts::PI;
use std::sync::OnceLock;

pub use crate::thermal_sensor::{
    ABI_FM4_SRF_ARCHIVE_SHA256, ABI_FM4_SRF_ARCHIVE_URL, ABI_FM4_SRF_RELEASE_URL,
};

/// NOAA's radiance-to-CMI definition; section 3.4.1.2, equations 3-2 through 3-5.
pub const ABI_CMIP_ATBD_URL: &str = "https://www.star.nesdis.noaa.gov/goesr/documents/ATBDs/Enterprise/ATBD_Enterprise_Cloud_and_Moisture_Imagery_Product_v4_2021-01-13.pdf";

/// Reflective bands needed for the future GOES-19 visible/NIR observation operator.
/// There is deliberately no default or render-intent conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiReflectiveBand {
    C01,
    C02,
    C03,
}

/// Provenance for an individual NOAA FM4 spectral response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbiReflectiveBandMetadata {
    pub band_id: u8,
    pub slug: &'static str,
    pub source_filename: &'static str,
    /// Hash of the exact NOAA archive entry, including its original CRLF newlines.
    pub source_sha256: &'static str,
    /// Hash of the vendored asset after the sole transformation CRLF -> LF.
    pub vendored_sha256: &'static str,
}

/// One exact tabulated NOAA response sample. Both spectral coordinates are
/// retained to make their integration measure explicit and independently checkable.
#[derive(Debug, Clone, Copy)]
pub struct SrfSample {
    pub wavelength_um: f64,
    pub wavenumber_cm1: f64,
    pub response: f64,
}

/// Invalid caller data. Signed finite radiance is allowed and is never clipped;
/// solar irradiance must be nonnegative with a strictly positive band integral.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VisibleSensorError {
    NonFiniteSpectrum { wavelength_um: f64 },
    NegativeSolarIrradiance { wavelength_um: f64 },
    NonPositiveSolarIntegral,
    NonFiniteIntegral,
    InvalidEarthSunDistance,
}

impl std::fmt::Display for VisibleSensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteSpectrum { wavelength_um } => {
                write!(f, "non-finite spectrum at {wavelength_um} um")
            }
            Self::NegativeSolarIrradiance { wavelength_um } => {
                write!(f, "negative solar irradiance at {wavelength_um} um")
            }
            Self::NonPositiveSolarIntegral => {
                f.write_str("solar irradiance has no positive in-band integral")
            }
            Self::NonFiniteIntegral => f.write_str("spectral integral or reflectance overflowed"),
            Self::InvalidEarthSunDistance => {
                f.write_str("Earth-Sun distance in AU and its square must be finite and positive")
            }
        }
    }
}

impl std::error::Error for VisibleSensorError {}

impl AbiReflectiveBand {
    pub const ALL: [Self; 3] = [Self::C01, Self::C02, Self::C03];

    pub const fn metadata(self) -> AbiReflectiveBandMetadata {
        match self {
            Self::C01 => AbiReflectiveBandMetadata {
                band_id: 1,
                slug: "goes-r-abi-c01-fm4",
                source_filename: "GOES-R_ABI_FM4_SRF_CWG_ch1.txt",
                source_sha256: "940A9CB586E96BEE0B079CC3149D1AF6D9C3252B35E24D2FC7DB9BB3CD7C88DB",
                vendored_sha256: "8076BA1B487B706574B55158E02A69A9A7E6B457C6EC9D20C92180B5B32C47FE",
            },
            Self::C02 => AbiReflectiveBandMetadata {
                band_id: 2,
                slug: "goes-r-abi-c02-fm4",
                source_filename: "GOES-R_ABI_FM4_SRF_CWG_ch2.txt",
                source_sha256: "800A98FE0AA63253883F9CCDD6BA6117594A91C26DF4743E0D8A7D5579C3A24F",
                vendored_sha256: "7B025F14CB49FD02E898F0AF3DC57155211F87A8FC3A41E27F1462A1C0A53DE6",
            },
            Self::C03 => AbiReflectiveBandMetadata {
                band_id: 3,
                slug: "goes-r-abi-c03-fm4",
                source_filename: "GOES-R_ABI_FM4_SRF_CWG_ch3.txt",
                source_sha256: "5D15E53230BFF18F6F8F8C45B8885468254D66CAE50BBF0573D8330C8E09E4E3",
                vendored_sha256: "BA962645BDB8CA434C8C474AE7D5D6CF0BB337FD96E0A21E55E7CA9B9F4E8B7E",
            },
        }
    }

    /// Official wavelength-ascending response samples. The full tabulated response
    /// is used; no effective-wavelength or RGB interpolation is substituted.
    pub fn samples(self) -> &'static [SrfSample] {
        self.response().samples.as_slice()
    }

    /// Integral of relative response over wavelength, in um.
    pub fn response_integral_um(self) -> f64 {
        self.response().width_um
    }

    /// Integrate R(lambda) * spectrum(lambda) d(lambda) using trapezoidal weights
    /// on the tabulated wavelength grid. The callback receives wavelength in um.
    ///
    /// Supply spectral density per um. For radiance in W m^-2 sr^-1 um^-1, the
    /// result is a response-weighted integral in W m^-2 sr^-1. Supplying density
    /// per cm^-1 without the coordinate Jacobian is incorrect.
    pub fn integrate_wavelength(
        self,
        spectrum: impl FnMut(f64) -> f64,
    ) -> Result<f64, VisibleSensorError> {
        self.integrate_checked(spectrum, false)
    }

    /// Response-weighted mean spectral density, integral(R*f)/integral(R).
    /// Units equal the callback's spectral-density units, which must be per um.
    pub fn band_average_wavelength(
        self,
        spectrum: impl FnMut(f64) -> f64,
    ) -> Result<f64, VisibleSensorError> {
        finite(self.integrate_wavelength(spectrum)? / self.response_integral_um())
    }

    /// CMI reflectance factor at the observation's Earth-Sun distance:
    /// pi * integral(R L_lambda d_lambda) / integral(R E_lambda d_lambda).
    ///
    /// Both callbacks receive wavelength in um. L_lambda is TOA directional
    /// radiance in W m^-2 sr^-1 um^-1; E_lambda is TOA solar irradiance on a plane
    /// normal to the solar beam in W m^-2 um^-1 at the SAME distance. Do not
    /// pre-multiply E_lambda by local cos(SZA). The SRF's normalization cancels.
    ///
    /// No solar-zenith division, clipping, gamma, exposure or display transform is
    /// applied. Values above one (and finite negative radiance) are preserved.
    pub fn reflectance_factor(
        self,
        radiance_lambda: impl FnMut(f64) -> f64,
        solar_irradiance_lambda: impl FnMut(f64) -> f64,
    ) -> Result<f64, VisibleSensorError> {
        let radiance = self.integrate_wavelength(radiance_lambda)?;
        let irradiance = self.integrate_checked(solar_irradiance_lambda, true)?;
        if irradiance <= 0.0 {
            return Err(VisibleSensorError::NonPositiveSolarIntegral);
        }
        finite(PI * radiance / irradiance)
    }

    /// Convenience normalization with solar irradiance tabulated at 1 AU.
    ///
    /// Radiance is still the TOA radiance at the observation time. The supplied
    /// distance d is instantaneous Earth-Sun distance divided by 1 AU. This method
    /// evaluates E_observation = E_1au / d^2, exactly equivalent to NOAA's
    /// rho_f = pi*d^2*L_band/E_1au_band. No ephemeris or implicit distance is used.
    pub fn reflectance_factor_from_1au(
        self,
        radiance_lambda: impl FnMut(f64) -> f64,
        mut solar_irradiance_1au_lambda: impl FnMut(f64) -> f64,
        earth_sun_distance_au: f64,
    ) -> Result<f64, VisibleSensorError> {
        let distance_squared = earth_sun_distance_au * earth_sun_distance_au;
        if !earth_sun_distance_au.is_finite()
            || earth_sun_distance_au <= 0.0
            || !distance_squared.is_finite()
            || distance_squared <= 0.0
        {
            return Err(VisibleSensorError::InvalidEarthSunDistance);
        }
        self.reflectance_factor(radiance_lambda, |lambda| {
            solar_irradiance_1au_lambda(lambda) / distance_squared
        })
    }

    fn integrate_checked(
        self,
        mut spectrum: impl FnMut(f64) -> f64,
        is_solar: bool,
    ) -> Result<f64, VisibleSensorError> {
        let response = self.response();
        let mut integral = 0.0;
        for (sample, weight) in response.samples.iter().zip(&response.weights_um) {
            let value = spectrum(sample.wavelength_um);
            if !value.is_finite() {
                return Err(VisibleSensorError::NonFiniteSpectrum {
                    wavelength_um: sample.wavelength_um,
                });
            }
            if is_solar && value < 0.0 {
                return Err(VisibleSensorError::NegativeSolarIrradiance {
                    wavelength_um: sample.wavelength_um,
                });
            }
            integral += weight * value;
        }
        finite(integral)
    }

    fn text(self) -> &'static str {
        match self {
            Self::C01 => include_str!("../assets/abi_srf/GOES-R_ABI_FM4_SRF_CWG_ch1.txt"),
            Self::C02 => include_str!("../assets/abi_srf/GOES-R_ABI_FM4_SRF_CWG_ch2.txt"),
            Self::C03 => include_str!("../assets/abi_srf/GOES-R_ABI_FM4_SRF_CWG_ch3.txt"),
        }
    }

    fn response(self) -> &'static SpectralResponse {
        static C01: OnceLock<SpectralResponse> = OnceLock::new();
        static C02: OnceLock<SpectralResponse> = OnceLock::new();
        static C03: OnceLock<SpectralResponse> = OnceLock::new();
        let cell = match self {
            Self::C01 => &C01,
            Self::C02 => &C02,
            Self::C03 => &C03,
        };
        cell.get_or_init(|| SpectralResponse::from_noaa_text(self.text()))
    }
}

fn finite(value: f64) -> Result<f64, VisibleSensorError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(VisibleSensorError::NonFiniteIntegral)
    }
}

struct SpectralResponse {
    samples: Vec<SrfSample>,
    /// Response-inclusive trapezoidal weights on lambda, not wavenumber.
    weights_um: Vec<f64>,
    width_um: f64,
}

impl SpectralResponse {
    fn from_noaa_text(text: &str) -> Self {
        let samples: Vec<_> = text
            .lines()
            .filter(|line| !line.trim_start().starts_with('#') && !line.trim().is_empty())
            .map(|line| {
                let fields: Vec<f64> = line
                    .split_whitespace()
                    .map(|field| field.parse().expect("numeric vendored ABI SRF"))
                    .collect();
                assert_eq!(fields.len(), 3, "three NOAA SRF columns");
                let sample = SrfSample {
                    wavelength_um: fields[0],
                    wavenumber_cm1: fields[1],
                    response: fields[2],
                };
                assert!(sample.wavelength_um.is_finite() && sample.wavelength_um > 0.0);
                assert!(sample.wavenumber_cm1.is_finite() && sample.wavenumber_cm1 > 0.0);
                assert!(sample.response.is_finite() && sample.response >= 0.0);
                sample
            })
            .collect();
        assert!(samples.len() > 1000, "truncated ABI reflective-band SRF");
        let mut weights_um = vec![0.0; samples.len()];
        for (i, pair) in samples.windows(2).enumerate() {
            let delta = pair[1].wavelength_um - pair[0].wavelength_um;
            assert!(delta > 0.0, "NOAA SRF must ascend in wavelength");
            weights_um[i] += 0.5 * delta * pair[0].response;
            weights_um[i + 1] += 0.5 * delta * pair[1].response;
        }
        let width_um = weights_um.iter().sum::<f64>();
        assert!(width_um.is_finite() && width_um > 0.0);
        Self {
            samples,
            weights_um,
            width_um,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn sha256_hex(bytes: &[u8]) -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02X}"))
            .collect()
    }

    #[test]
    fn official_source_and_vendored_srf_hashes_match() {
        for band in AbiReflectiveBand::ALL {
            let text = band.text();
            assert!(!text.contains('\r'));
            assert_eq!(sha256_hex(text.as_bytes()), band.metadata().vendored_sha256);
            let original_crlf = text.replace('\n', "\r\n");
            assert_eq!(
                sha256_hex(original_crlf.as_bytes()),
                band.metadata().source_sha256
            );
        }
    }

    #[test]
    fn uniform_lambertian_has_albedo_times_incidence_without_sza_normalization() {
        for band in AbiReflectiveBand::ALL {
            for (albedo, mu0) in [(0.0, 1.0), (0.24, 0.31), (0.8, 1.0), (0.6, 0.0)] {
                // An arbitrary positive sloped solar spectrum exercises the
                // energy-weighted ratio; its amplitude and SRF width cancel.
                let solar = |lambda: f64| 1800.0 / lambda;
                let rho = band
                    .reflectance_factor(|lambda| albedo * mu0 * solar(lambda) / PI, solar)
                    .unwrap();
                assert!((rho - albedo * mu0).abs() < 2.0e-14);
            }
            let average = band.band_average_wavelength(|_| 173.0).unwrap();
            assert!((average - 173.0).abs() < 2.0e-11);
        }
    }

    #[test]
    fn directional_reflectance_above_one_and_signed_radiance_are_preserved() {
        for band in AbiReflectiveBand::ALL {
            for wanted in [1.7, -0.025] {
                let rho = band
                    .reflectance_factor(|_| wanted * 1000.0 / PI, |_| 1000.0)
                    .unwrap();
                assert!((rho - wanted).abs() < 2.0e-14);
            }
        }
    }

    #[test]
    fn one_au_and_current_distance_contracts_agree() {
        for band in AbiReflectiveBand::ALL {
            let distance: f64 = 1.02;
            let solar_1au = |lambda: f64| 1500.0 / lambda;
            let radiance = |lambda: f64| 0.37 * solar_1au(lambda) / (PI * distance.powi(2));
            let current = band
                .reflectance_factor(radiance, |lambda| solar_1au(lambda) / distance.powi(2))
                .unwrap();
            let from_1au = band
                .reflectance_factor_from_1au(radiance, solar_1au, distance)
                .unwrap();
            assert!((current - 0.37).abs() < 2.0e-14);
            assert!((current - from_1au).abs() < 2.0e-14);
            let wrong_distance = band.reflectance_factor(radiance, solar_1au).unwrap();
            assert!((wrong_distance - current / distance.powi(2)).abs() < 2.0e-14);
        }
    }

    #[test]
    fn wavelength_and_wavenumber_integrals_agree_with_jacobian() {
        for band in AbiReflectiveBand::ALL {
            for power in [0, 1, -4] {
                let per_um = |lambda: f64| lambda.powi(power);
                let by_wavelength = band.integrate_wavelength(per_um).unwrap();
                let by_wavenumber = band
                    .samples()
                    .windows(2)
                    .map(|pair| {
                        let value = |sample: SrfSample| {
                            // lambda[um] = 10^4 / nu[cm^-1]; the absolute
                            // Jacobian is |d lambda / d nu| = 10^4 / nu^2.
                            let nu = sample.wavenumber_cm1;
                            per_um(10_000.0 / nu) * sample.response * 10_000.0 / nu.powi(2)
                        };
                        0.5 * (value(pair[0]) + value(pair[1]))
                            * (pair[0].wavenumber_cm1 - pair[1].wavenumber_cm1)
                    })
                    .sum::<f64>();
                // NOAA rounds its wavelength column independently of the integer
                // wavenumbers; the two trapezoidal rules also have distinct
                // second-order discretization errors. This allows 5 ppm.
                let relative = (by_wavelength - by_wavenumber).abs() / by_wavelength.abs();
                assert!(relative < 5.0e-6, "{band:?} power={power}: {relative}");
            }
        }
    }

    #[test]
    fn invalid_spectra_and_distance_are_explicit_errors() {
        let band = AbiReflectiveBand::C01;
        assert_eq!(
            band.reflectance_factor(|_| 1.0, |_| 0.0),
            Err(VisibleSensorError::NonPositiveSolarIntegral)
        );
        assert!(matches!(
            band.reflectance_factor(|_| 1.0, |_| -1.0),
            Err(VisibleSensorError::NegativeSolarIrradiance { .. })
        ));
        assert!(matches!(
            band.reflectance_factor(|_| f64::NAN, |_| 1.0),
            Err(VisibleSensorError::NonFiniteSpectrum { .. })
        ));
        for distance in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                band.reflectance_factor_from_1au(|_| 1.0, |_| 1.0, distance),
                Err(VisibleSensorError::InvalidEarthSunDistance)
            );
        }
    }
}
