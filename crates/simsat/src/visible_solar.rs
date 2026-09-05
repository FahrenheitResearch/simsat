//! Measured TSIS-1 HSRS illumination integrated with official ABI FM4 responses.
//!
//! Solar Fraunhofer lines are integrated at every native 0.001-nm source knot,
//! not aliased onto the response grid. The resulting positive nodal weights
//! integrate a PIECEWISE-LINEAR normalized transfer L_lambda/E_lambda on the
//! exact SRF nodes. See `assets/solar_hsrs/README.md` and the reproducible Python
//! preparation script. Atmospheric gas lines must be resolved independently;
//! these weights do not turn a smooth transport approximation into line-by-line RT.
//!
//! This spectrum is a fixed reference at 1 AU, not contemporaneous irradiance.
//! An observation-distance correction is explicit where radiance is returned.
//! No RGB solar constants, gamma, exposure, mu0 division or clipping are used.

use crate::visible_sensor::{AbiReflectiveBand, VisibleSensorError};
use std::f64::consts::PI;
use std::sync::OnceLock;

pub const HSRS_SOURCE_URL: &str =
    "https://lasp.colorado.edu/lisird/latis/dap/tsis1_hsrs.csv?wavelength>=400&wavelength<=1000";
pub const HSRS_SOURCE_SHA256: &str =
    "ea6d0e219925a69607a485451ed2a8dcaf8b2f583b646b04a94122feef4d697f";

#[derive(Debug, Clone, Copy)]
pub struct SolarResponseNode {
    pub wavelength_um: f64,
    /// Integral E_1au(lambda) R(lambda) phi_i(lambda) d_lambda, in W m^-2.
    pub solar_response_weight_w_m2: f64,
}

pub struct AbiSolarResponse {
    band: AbiReflectiveBand,
    nodes: Vec<SolarResponseNode>,
    integral_w_m2: f64,
}

impl AbiSolarResponse {
    pub fn for_band(band: AbiReflectiveBand) -> &'static Self {
        static C01: OnceLock<AbiSolarResponse> = OnceLock::new();
        static C02: OnceLock<AbiSolarResponse> = OnceLock::new();
        static C03: OnceLock<AbiSolarResponse> = OnceLock::new();
        let cell = match band {
            AbiReflectiveBand::C01 => &C01,
            AbiReflectiveBand::C02 => &C02,
            AbiReflectiveBand::C03 => &C03,
        };
        cell.get_or_init(|| Self::from_asset(band))
    }

    pub fn band(&self) -> AbiReflectiveBand {
        self.band
    }
    pub fn nodes(&self) -> &[SolarResponseNode] {
        &self.nodes
    }
    pub fn solar_response_integral_1au_w_m2(&self) -> f64 {
        self.integral_w_m2
    }
    pub fn mean_solar_irradiance_1au_w_m2_um(&self) -> f64 {
        self.integral_w_m2 / self.band.response_integral_um()
    }

    /// ABI reflectance factor from wavelength-resolved L_lambda/E_lambda [sr^-1].
    /// The transfer is evaluated at the observation geometry. For an unattenuated
    /// Lambertian surface, supply a(lambda)*mu0/pi. Result: solar-weighted a*mu0.
    /// Positive radiative transfer is linear in E, so the Earth-Sun distance
    /// cancels here. A time-dependent change of solar spectral SHAPE does not.
    pub fn reflectance_factor_from_normalized_radiance(
        &self,
        transfer_sr1: impl FnMut(f64) -> f64,
    ) -> Result<f64, VisibleSensorError> {
        finite(PI * self.integrate_transfer(transfer_sr1)? / self.integral_w_m2)
    }

    /// Response-weighted mean TOA spectral radiance [W m^-2 sr^-1 um^-1],
    /// given normalized L_lambda/E_lambda [sr^-1] and Earth-Sun distance in AU.
    pub fn mean_radiance_from_normalized_radiance(
        &self,
        transfer_sr1: impl FnMut(f64) -> f64,
        earth_sun_distance_au: f64,
    ) -> Result<f64, VisibleSensorError> {
        let d2 = earth_sun_distance_au * earth_sun_distance_au;
        if !earth_sun_distance_au.is_finite()
            || earth_sun_distance_au <= 0.0
            || !d2.is_finite()
            || d2 <= 0.0
        {
            return Err(VisibleSensorError::InvalidEarthSunDistance);
        }
        finite(self.integrate_transfer(transfer_sr1)? / (self.band.response_integral_um() * d2))
    }

    fn integrate_transfer(
        &self,
        mut transfer: impl FnMut(f64) -> f64,
    ) -> Result<f64, VisibleSensorError> {
        let mut sum = 0.0;
        for n in &self.nodes {
            let f = transfer(n.wavelength_um);
            if !f.is_finite() {
                return Err(VisibleSensorError::NonFiniteSpectrum {
                    wavelength_um: n.wavelength_um,
                });
            }
            sum += f * n.solar_response_weight_w_m2;
        }
        finite(sum)
    }

    fn from_asset(band: AbiReflectiveBand) -> Self {
        let text = asset(band);
        let nodes: Vec<_> = text
            .lines()
            .filter(|s| !s.starts_with('#') && !s.trim().is_empty())
            .map(|s| {
                let v: Vec<f64> = s
                    .split_whitespace()
                    .map(|v| v.parse().expect("numeric HSRS response asset"))
                    .collect();
                assert_eq!(v.len(), 2);
                assert!(v[0].is_finite() && v[0] > 0.0 && v[1].is_finite() && v[1] >= 0.0);
                SolarResponseNode {
                    wavelength_um: v[0],
                    solar_response_weight_w_m2: v[1],
                }
            })
            .collect();
        assert_eq!(nodes.len(), band.samples().len());
        for (n, s) in nodes.iter().zip(band.samples()) {
            assert_eq!(n.wavelength_um, s.wavelength_um);
        }
        let integral_w_m2 = nodes
            .iter()
            .map(|n| n.solar_response_weight_w_m2)
            .sum::<f64>();
        assert!(integral_w_m2.is_finite() && integral_w_m2 > 0.0);
        Self {
            band,
            nodes,
            integral_w_m2,
        }
    }
}

fn finite(v: f64) -> Result<f64, VisibleSensorError> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(VisibleSensorError::NonFiniteIntegral)
    }
}
fn asset(band: AbiReflectiveBand) -> &'static str {
    match band {
        AbiReflectiveBand::C01 => include_str!("../assets/solar_hsrs/abi-fm4-c01-hsrs-weights.txt"),
        AbiReflectiveBand::C02 => include_str!("../assets/solar_hsrs/abi-fm4-c02-hsrs-weights.txt"),
        AbiReflectiveBand::C03 => include_str!("../assets/solar_hsrs/abi-fm4-c03-hsrs-weights.txt"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn solar_assets_and_source_match_manifest() {
        let m: serde_json::Value =
            serde_json::from_str(include_str!("../assets/solar_hsrs/manifest.json")).unwrap();
        assert_eq!(m["source_sha256"], HSRS_SOURCE_SHA256);
        for (band, record) in AbiReflectiveBand::ALL
            .into_iter()
            .zip(m["bands"].as_array().unwrap())
        {
            let hash = Sha256::digest(asset(band).as_bytes())
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>();
            assert_eq!(record["sha256"], hash);
            let solar = AbiSolarResponse::for_band(band);
            assert!(
                (solar.mean_solar_irradiance_1au_w_m2_um()
                    / record["mean_solar_irradiance_w_m2_um"].as_f64().unwrap()
                    - 1.0)
                    .abs()
                    < 1e-13
            );
        }
    }

    #[test]
    fn lambertian_normalization_and_inverse_square_distance() {
        for band in AbiReflectiveBand::ALL {
            let s = AbiSolarResponse::for_band(band);
            let f = 0.2 * 0.6 / PI;
            let rho = s
                .reflectance_factor_from_normalized_radiance(|_| f)
                .unwrap();
            assert!((rho - 0.12).abs() < 1e-13);
            let l1 = s
                .mean_radiance_from_normalized_radiance(|_| f, 1.0)
                .unwrap();
            let l2 = s
                .mean_radiance_from_normalized_radiance(|_| f, 2.0)
                .unwrap();
            assert_eq!(l1, 4.0 * l2);
            assert!((PI * l1 / s.mean_solar_irradiance_1au_w_m2_um() - rho).abs() < 1e-13);
            assert!(
                (s.reflectance_factor_from_normalized_radiance(|_| 2.0 / PI)
                    .unwrap()
                    - 2.0)
                    .abs()
                    < 1e-13
            );
            assert!(
                (s.reflectance_factor_from_normalized_radiance(|_| -0.1 / PI)
                    .unwrap()
                    + 0.1)
                    .abs()
                    < 1e-13
            );
        }
    }

    #[test]
    fn invalid_spectrum_or_distance_is_not_hidden() {
        let s = AbiSolarResponse::for_band(AbiReflectiveBand::C01);
        assert!(
            s.reflectance_factor_from_normalized_radiance(|_| f64::NAN)
                .is_err()
        );
        assert!(
            s.reflectance_factor_from_normalized_radiance(|_| f64::MAX)
                .is_err()
        );
        for d in [
            0.0,
            -1.0,
            f64::NAN,
            f64::INFINITY,
            f64::MAX,
            f64::MIN_POSITIVE,
        ] {
            assert!(
                s.mean_radiance_from_normalized_radiance(|_| 1.0, d)
                    .is_err()
            );
        }
    }
}
