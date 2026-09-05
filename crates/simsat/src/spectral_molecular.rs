//! Wavelength-resolved dry-air Rayleigh optics for the experimental ABI operator.
//!
//! Bodhaine et al. (1999), Eqs. 5, 6, 18, 19, 22 and 23. Wavelength is in
//! micrometres; cross sections are in m^2 per molecule. The refractive-index fit
//! and reference number density BOTH refer to 288.15 K and 101325 Pa. Do not
//! replace one reference constant without replacing the other.
//!
//! This is a molecular component, not a complete ABI radiative-transfer model.
//! It excludes water-vapour Rayleigh scattering, gas absorption, aerosol,
//! polarization and Raman redistribution. Density/column inputs must describe
//! DRY air. The checked range 0.25..=1.0 um covers the paper's Table 3 and the
//! complete vendored ABI C01/C02/C03 response ranges. No current CO2 value is
//! assumed: the caller provides a concentration appropriate to its case.
//!
//! The WGSL reference is `gpu/shaders/spectral_molecular.wgsl`. It is separate
//! from the legacy RGB display atmosphere; no existing image gains are changed.

use std::f64::consts::PI;

pub const BODHAINE_1999_URL: &str =
    "https://reef.atmos.colostate.edu/~odell/at721/resources/rayleighOpticalDepth.pdf";
/// Published reference for the scalar depolarizing Rayleigh phase, section 2.5.
pub const RTTOV_14_SCIENCE_URL: &str =
    "https://nwp-saf.eumetsat.int/site/download/documentation/rtm/docs_rttov14/rttov14_svr.pdf";
const REFERENCE_NUMBER_DENSITY_M3: f64 = 2.546_899e25;
const BOLTZMANN_J_K: f64 = 1.380_649e-23;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MolecularError {
    WavelengthOutOfRange,
    Co2OutOfRange,
    InvalidDensity,
    InvalidColumn,
    InvalidDryPressureOrTemperature,
    InvalidScatteringCosine,
    NonFiniteResult,
}

impl std::fmt::Display for MolecularError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::WavelengthOutOfRange => "molecular wavelength must be finite and within 0.25..=1.0 um",
            Self::Co2OutOfRange => "dry-air CO2 must be finite and within the supported 0..=1000 ppm range",
            Self::InvalidDensity => "dry-air number density must be finite and nonnegative (m^-3)",
            Self::InvalidColumn => "dry-air number column must be finite and nonnegative (m^-2)",
            Self::InvalidDryPressureOrTemperature => "dry pressure must be finite and nonnegative (Pa), and temperature finite and positive (K)",
            Self::InvalidScatteringCosine => "scattering cosine must be finite and within -1..=1",
            Self::NonFiniteResult => "molecular calculation overflowed",
        })
    }
}
impl std::error::Error for MolecularError {}

/// Validated dry-air molecular properties at a single wavelength. Fields are
/// private so density and phase calculations cannot consume unvalidated optics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DryAirRayleigh {
    wavelength_um: f64,
    co2_ppm: f64,
    king_factor: f64,
    cross_section_m2: f64,
    phase_gamma: f64,
}

impl DryAirRayleigh {
    pub fn new(wavelength_um: f64, co2_ppm: f64) -> Result<Self, MolecularError> {
        if !(0.25..=1.0).contains(&wavelength_um) {
            return Err(MolecularError::WavelengthOutOfRange);
        }
        // An explicit terrestrial input contract, not a universal bound on CO2.
        if !(0.0..=1000.0).contains(&co2_ppm) {
            return Err(MolecularError::Co2OutOfRange);
        }
        let inv_lambda2 = wavelength_um.powi(-2);
        let co2_fraction = co2_ppm * 1.0e-6;
        let co2_percent = co2_ppm * 1.0e-4;
        let refractivity_300 = 1.0e-8
            * (8060.51
                + 2_480_990.0 / (132.274 - inv_lambda2)
                + 17_455.7 / (39.32957 - inv_lambda2));
        let refractivity = refractivity_300 * (1.0 + 0.54 * (co2_fraction - 0.0003));
        let f_n2 = 1.034 + 3.17e-4 * inv_lambda2;
        let f_o2 = 1.096 + 1.385e-3 * inv_lambda2 + 1.448e-4 * inv_lambda2.powi(2);
        let king_factor = (78.084 * f_n2 + 20.946 * f_o2 + 0.934 + co2_percent * 1.15)
            / (78.084 + 20.946 + 0.934 + co2_percent);
        // (n^2-1)/(n^2+2), evaluated without subtracting nearly equal values.
        // The equivalent form matters especially in the float32 shader twin.
        let n2_minus_one = refractivity * (2.0 + refractivity);
        let refractive_ratio = n2_minus_one / (3.0 + n2_minus_one);
        // lambda_m^4 * Ns^2 = lambda_um^4 * (Ns * 1e-12)^2.
        // This form avoids overflowing float32 in the shader implementation.
        let cross_section_m2 = 24.0 * PI.powi(3) * refractive_ratio.powi(2) * king_factor
            / (wavelength_um.powi(4) * (REFERENCE_NUMBER_DENSITY_M3 * 1.0e-12).powi(2));
        let delta = 6.0 * (king_factor - 1.0) / (7.0 * king_factor + 3.0);
        let phase_gamma = delta / (2.0 - delta);
        Ok(Self {
            wavelength_um,
            co2_ppm,
            king_factor,
            cross_section_m2,
            phase_gamma,
        })
    }

    pub fn wavelength_um(self) -> f64 {
        self.wavelength_um
    }
    pub fn co2_ppm(self) -> f64 {
        self.co2_ppm
    }
    pub fn king_factor(self) -> f64 {
        self.king_factor
    }
    pub fn cross_section_m2(self) -> f64 {
        self.cross_section_m2
    }
    /// gamma = delta / (2-delta), for use in the matching scalar phase function.
    pub fn phase_gamma(self) -> f64 {
        self.phase_gamma
    }

    pub fn scattering_coefficient_m1(
        self,
        dry_number_density_m3: f64,
    ) -> Result<f64, MolecularError> {
        if !dry_number_density_m3.is_finite() || dry_number_density_m3 < 0.0 {
            return Err(MolecularError::InvalidDensity);
        }
        finite(self.cross_section_m2 * dry_number_density_m3)
    }

    pub fn optical_depth(self, dry_number_column_m2: f64) -> Result<f64, MolecularError> {
        if !dry_number_column_m2.is_finite() || dry_number_column_m2 < 0.0 {
            return Err(MolecularError::InvalidColumn);
        }
        finite(self.cross_section_m2 * dry_number_column_m2)
    }

    /// Scalar, unpolarized phase in sr^-1, normalized over 4 pi. The supplied
    /// cosine is between incoming and outgoing PHOTON propagation directions.
    /// Rayleigh is symmetric, so reversing one direction gives the same value.
    pub fn phase_sr1(self, scattering_cosine: f64) -> Result<f64, MolecularError> {
        if !(-1.0..=1.0).contains(&scattering_cosine) {
            return Err(MolecularError::InvalidScatteringCosine);
        }
        let g = self.phase_gamma;
        Ok(3.0 / (16.0 * PI * (1.0 + 2.0 * g))
            * ((1.0 + 3.0 * g) + (1.0 - g) * scattering_cosine.powi(2)))
    }
}

/// Ideal-gas dry-air number density (m^-3). Subtract water-vapour partial
/// pressure before calling; total moist pressure would double-count dry air.
pub fn dry_air_number_density_m3(
    dry_pressure_pa: f64,
    temperature_k: f64,
) -> Result<f64, MolecularError> {
    if !dry_pressure_pa.is_finite()
        || dry_pressure_pa < 0.0
        || !temperature_k.is_finite()
        || temperature_k <= 0.0
    {
        return Err(MolecularError::InvalidDryPressureOrTemperature);
    }
    finite(dry_pressure_pa / (BOLTZMANN_J_K * temperature_k))
}

fn finite(value: f64) -> Result<f64, MolecularError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(MolecularError::NonFiniteResult)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_bodhaine_table_3_at_360_ppm() {
        // Independently published values, Table 3; cm^2 -> m^2 multiplies by 1e-4.
        // Printed precision is 5 significant digits. This catches cm/um/m and
        // ppm/percent mistakes, and mismatched reference temperature/density.
        for (lambda, sigma_m2) in [
            (0.250, 1.2610e-29),
            (0.300, 5.6525e-30),
            (0.560, 4.1908e-31),
            (0.640, 2.4341e-31),
            (1.000, 4.0132e-32),
        ] {
            let got = DryAirRayleigh::new(lambda, 360.0)
                .unwrap()
                .cross_section_m2();
            assert!(
                (got / sigma_m2 - 1.0).abs() < 5.0e-5,
                "{lambda}: {got:e} vs {sigma_m2:e}"
            );
        }
        for (lambda, king) in [
            (0.25, 1.06308),
            (0.3, 1.05643),
            (0.56, 1.04873),
            (0.64, 1.04820),
        ] {
            assert!(
                (DryAirRayleigh::new(lambda, 360.0).unwrap().king_factor() - king).abs() < 5.0e-6
            );
        }
    }

    #[test]
    fn depolarizing_phase_conserves_scattered_energy_and_is_symmetric() {
        for lambda in [0.25, 0.47, 0.64, 0.86, 1.0] {
            let optic = DryAirRayleigh::new(lambda, 360.0).unwrap();
            // Simpson integration is exact for this quadratic phase in mu.
            let n = 1000;
            let mut sum = 0.0;
            for i in 0..=n {
                let mu = -1.0 + 2.0 * i as f64 / n as f64;
                let w = if i == 0 || i == n {
                    1.0
                } else if i % 2 == 0 {
                    2.0
                } else {
                    4.0
                };
                sum += w * optic.phase_sr1(mu).unwrap();
                assert_eq!(optic.phase_sr1(mu), optic.phase_sr1(-mu));
            }
            assert!((sum * (2.0 / n as f64) / 3.0 * 2.0 * PI - 1.0).abs() < 1.0e-12);
            // Anisotropic molecules scatter more at 90 degrees than ideal spheres.
            assert!(optic.phase_sr1(0.0).unwrap() > 3.0 / (16.0 * PI));
            assert!(optic.phase_sr1(1.0).unwrap() < 3.0 / (8.0 * PI));
        }
    }

    #[test]
    fn density_and_column_units_agree_for_a_uniform_layer() {
        let optic = DryAirRayleigh::new(0.56, 360.0).unwrap();
        let n = dry_air_number_density_m3(101325.0, 288.15).unwrap();
        assert!((n / REFERENCE_NUMBER_DENSITY_M3 - 1.0).abs() < 3.0e-5);
        let beta = optic.scattering_coefficient_m1(n).unwrap();
        assert!((optic.optical_depth(n * 8000.0).unwrap() - beta * 8000.0).abs() < 1.0e-15);
        assert_eq!(optic.scattering_coefficient_m1(0.0).unwrap(), 0.0);
        assert_eq!(optic.optical_depth(0.0).unwrap(), 0.0);
        assert_eq!(dry_air_number_density_m3(0.0, 288.15).unwrap(), 0.0);
    }

    #[test]
    fn invalid_inputs_are_rejected_instead_of_silently_clamped() {
        for v in [f64::NAN, f64::INFINITY, -1.0, 0.0, 0.249, 1.001] {
            assert!(DryAirRayleigh::new(v, 360.0).is_err());
        }
        for v in [f64::NAN, f64::INFINITY, -1.0, 1001.0] {
            assert!(DryAirRayleigh::new(0.64, v).is_err());
        }
        let optic = DryAirRayleigh::new(0.64, 360.0).unwrap();
        for v in [f64::NAN, f64::INFINITY, -1.0] {
            assert!(optic.optical_depth(v).is_err());
            assert!(optic.scattering_coefficient_m1(v).is_err());
        }
        for v in [f64::NAN, f64::INFINITY, 1.01, -1.01] {
            assert!(optic.phase_sr1(v).is_err());
        }
        assert!(dry_air_number_density_m3(1.0, 0.0).is_err());
        assert!(dry_air_number_density_m3(-1.0, 300.0).is_err());
        assert!(dry_air_number_density_m3(f64::MAX, f64::MIN_POSITIVE).is_err());
    }

    #[test]
    fn spectral_molecular_wgsl_validates() {
        let module =
            naga::front::wgsl::parse_str(include_str!("gpu/shaders/spectral_molecular.wgsl"))
                .unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}
