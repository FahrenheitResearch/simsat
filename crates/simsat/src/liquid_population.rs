//! Scattering by a supplied discrete liquid-droplet size distribution.
//!
//! Number weights may be measured bin counts or quadrature weights of n(r) dr.
//! They are normalized together, never fitted to an observed image. The bulk
//! extinction/scattering cross sections, mass, asymmetry and angular phase use
//! the SAME particle population. A caller converts model liquid mass density
//! (kg m^-3) to optical coefficients (m^-1); no display extinction scale enters.
//!
//! Reference-condition pure-water indices are from material_indices. This is
//! not a temperature-resolved supercooled-water model, ice-habit model, or cloud
//! multiple-scattering closure. No default size distribution is silently chosen.
use crate::material_indices::{RefractiveIndex, VisibleMaterial};
use crate::mie_sphere::{MieError, MieSphere};
use std::f64::consts::PI;

pub const LIQUID_DENSITY_KG_M3: f64 = 1000.0;

#[derive(Debug, Clone, Copy)]
pub struct ParticleNumberNode {
    pub radius_m: f64,
    /// Finite nonnegative number weight, including quadrature/bin width.
    pub number_weight: f64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PopulationError {
    InvalidPopulation,
    InvalidMedium,
    InvalidWavelength,
    InvalidMassDensity,
    Sphere(MieError),
}
impl std::fmt::Display for PopulationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPopulation => f.write_str("liquid population needs finite positive radii and nonnegative number weights with positive finite total mass"),
            Self::InvalidMedium => f.write_str("real medium refractive index must be finite within 1..=1.01"),
            Self::InvalidWavelength => f.write_str("visible liquid wavelength must be finite within 0.4..=1 micrometres"),
            Self::InvalidMassDensity => f.write_str("model liquid mass density must be finite and nonnegative"),
            Self::Sphere(e) => write!(f, "liquid sphere: {e}"),
        }
    }
}
impl std::error::Error for PopulationError {}
impl From<MieError> for PopulationError {
    fn from(e: MieError) -> Self {
        Self::Sphere(e)
    }
}
#[derive(Debug, Clone, Copy)]
pub struct LiquidBulkOptics {
    pub mass_extinction_m2_kg: f64,
    pub mass_scattering_m2_kg: f64,
    pub mass_absorption_m2_kg: f64,
    pub asymmetry: f64,
    /// M3 / M2, where Mk = integral r^k n(r) dr.
    pub effective_radius_m: f64,
    /// M4 M2 / M3^2 - 1, the area-weighted effective variance.
    pub effective_variance: f64,
}
#[derive(Debug, Clone, Copy)]
pub struct LiquidVolumeOptics {
    pub extinction_m_inv: f64,
    pub scattering_m_inv: f64,
    pub absorption_m_inv: f64,
}
impl LiquidBulkOptics {
    pub fn at_mass_density(self, kg_m3: f64) -> Result<LiquidVolumeOptics, PopulationError> {
        if [
            self.mass_extinction_m2_kg,
            self.mass_scattering_m2_kg,
            self.mass_absorption_m2_kg,
        ]
        .iter()
        .any(|c| !c.is_finite() || *c < 0.0)
        {
            return Err(PopulationError::InvalidPopulation);
        }
        if !kg_m3.is_finite() || kg_m3 < 0.0 {
            return Err(PopulationError::InvalidMassDensity);
        }
        let result = LiquidVolumeOptics {
            extinction_m_inv: self.mass_extinction_m2_kg * kg_m3,
            scattering_m_inv: self.mass_scattering_m2_kg * kg_m3,
            absorption_m_inv: self.mass_absorption_m2_kg * kg_m3,
        };
        if !result.extinction_m_inv.is_finite()
            || !result.scattering_m_inv.is_finite()
            || !result.absorption_m_inv.is_finite()
        {
            return Err(PopulationError::InvalidMassDensity);
        }
        Ok(result)
    }
}
pub struct LiquidPopulationOptics {
    bulk: LiquidBulkOptics,
    components: Vec<(MieSphere, f64)>,
}
impl LiquidPopulationOptics {
    pub fn new(
        wavelength_um: f64,
        medium_real_index: f64,
        nodes: &[ParticleNumberNode],
    ) -> Result<Self, PopulationError> {
        if !medium_real_index.is_finite() || !(1.0..=1.01).contains(&medium_real_index) {
            return Err(PopulationError::InvalidMedium);
        }
        let material = VisibleMaterial::LiquidWaterSegelstein1981
            .at(wavelength_um)
            .map_err(|_| PopulationError::InvalidWavelength)?;
        if nodes.is_empty()
            || nodes.iter().any(|n| {
                !n.radius_m.is_finite()
                    || n.radius_m <= 0.0
                    || !n.number_weight.is_finite()
                    || n.number_weight < 0.0
            })
        {
            return Err(PopulationError::InvalidPopulation);
        }
        // Rescaling all input number weights cannot change bulk optical
        // properties. Normalize by the largest first to avoid overflow when
        // callers provide large absolute number concentrations.
        let max_weight = nodes.iter().map(|n| n.number_weight).fold(0.0f64, f64::max);
        if max_weight == 0.0 {
            return Err(PopulationError::InvalidPopulation);
        }
        let total_scaled: f64 = nodes.iter().map(|n| n.number_weight / max_weight).sum();
        if !total_scaled.is_finite() {
            return Err(PopulationError::InvalidPopulation);
        }
        let index = RefractiveIndex {
            real: material.real / medium_real_index,
            imaginary: material.imaginary / medium_real_index,
        };
        let mut components = Vec::new();
        let (mut m2, mut m3, mut m4) = (0.0, 0.0, 0.0);
        let (mut ext, mut sca, mut absorption, mut moment) = (0.0, 0.0, 0.0, 0.0);
        for node in nodes {
            if node.number_weight == 0.0 {
                continue;
            }
            let weight = (node.number_weight / max_weight) / total_scaled;
            let radius = node.radius_m;
            let sphere = MieSphere::new(
                index,
                2.0 * PI * medium_real_index * radius / (wavelength_um * 1e-6),
            )?;
            let q = sphere.efficiencies();
            let area_weight = weight * PI * radius * radius;
            let scattering_weight = area_weight * q.scattering;
            ext += area_weight * q.extinction;
            sca += scattering_weight;
            absorption += area_weight * q.absorption;
            moment += scattering_weight * q.asymmetry;
            m2 += weight * radius.powi(2);
            m3 += weight * radius.powi(3);
            m4 += weight * radius.powi(4);
            components.push((sphere, scattering_weight));
        }
        let mass = 4.0 * PI / 3.0 * LIQUID_DENSITY_KG_M3 * m3;
        if !mass.is_finite() || mass <= 0.0 || !sca.is_finite() || sca <= 0.0 {
            return Err(PopulationError::InvalidPopulation);
        }
        for (_, weight) in &mut components {
            *weight /= sca;
        }
        let bulk = LiquidBulkOptics {
            mass_extinction_m2_kg: ext / mass,
            mass_scattering_m2_kg: sca / mass,
            mass_absorption_m2_kg: absorption / mass,
            asymmetry: moment / sca,
            effective_radius_m: m3 / m2,
            effective_variance: (m4 * m2 / (m3 * m3) - 1.0).max(0.0),
        };
        Ok(Self { bulk, components })
    }
    pub fn bulk(&self) -> LiquidBulkOptics {
        self.bulk
    }
    /// Shared CPU-prepared sphere coefficients and normalized scattering
    /// weights. GPU transport evaluates these same components in WGSL.
    pub fn components(&self) -> &[(MieSphere, f64)] {
        &self.components
    }
    pub fn phase_sr1(&self, cosine: f64) -> Result<f64, PopulationError> {
        let mut phase = 0.0;
        for (sphere, weight) in &self.components {
            phase += weight * sphere.phase_sr1(cosine)?;
        }
        Ok(phase)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn monodisperse_population_reproduces_cross_section_per_particle_mass() {
        let radius = 10e-6;
        let pop = LiquidPopulationOptics::new(
            0.64,
            1.0,
            &[ParticleNumberNode {
                radius_m: radius,
                number_weight: 1.0,
            }],
        )
        .unwrap();
        let index = VisibleMaterial::LiquidWaterSegelstein1981.at(0.64).unwrap();
        let sphere = MieSphere::new(index, 2.0 * PI * radius / 0.64e-6).unwrap();
        let expected =
            3.0 * sphere.efficiencies().extinction / (4.0 * LIQUID_DENSITY_KG_M3 * radius);
        assert!((pop.bulk().mass_extinction_m2_kg / expected - 1.0).abs() < 2e-13);
        assert!((pop.bulk().effective_radius_m / radius - 1.0).abs() < 2e-15);
        assert!(pop.bulk().effective_variance < 1e-15);
        for cosine in [-1.0, -0.5, 0.0, 0.5, 0.999999, 1.0] {
            assert!(
                (pop.phase_sr1(cosine).unwrap() / sphere.phase_sr1(cosine).unwrap() - 1.0).abs()
                    < 2e-13
            );
        }
    }
    #[test]
    fn number_rescaling_bin_splitting_and_mass_scaling_preserve_population() {
        let nodes = [
            ParticleNumberNode {
                radius_m: 5e-6,
                number_weight: 3.0,
            },
            ParticleNumberNode {
                radius_m: 20e-6,
                number_weight: 1.0,
            },
        ];
        let a = LiquidPopulationOptics::new(0.865, 1.00028, &nodes).unwrap();
        let split = [
            ParticleNumberNode {
                radius_m: 5e-6,
                number_weight: 1.5e300,
            },
            ParticleNumberNode {
                radius_m: 5e-6,
                number_weight: 1.5e300,
            },
            ParticleNumberNode {
                radius_m: 20e-6,
                number_weight: 1e300,
            },
        ];
        let b = LiquidPopulationOptics::new(0.865, 1.00028, &split).unwrap();
        assert!(
            (a.bulk().mass_extinction_m2_kg / b.bulk().mass_extinction_m2_kg - 1.0).abs() < 1e-13
        );
        for mu in [-1.0, -0.7, 0.0, 0.5, 0.999999, 1.0] {
            assert!((a.phase_sr1(mu).unwrap() / b.phase_sr1(mu).unwrap() - 1.0).abs() < 1e-13);
        }
        let bulk = a.bulk();
        // Independent analytical moments of this two-bin number population.
        let m2 = 3.0f64 * 5.0f64.powi(2) + 20.0f64.powi(2);
        let m3 = 3.0f64 * 5.0f64.powi(3) + 20.0f64.powi(3);
        let m4 = 3.0f64 * 5.0f64.powi(4) + 20.0f64.powi(4);
        assert!((bulk.effective_radius_m / (1e-6 * m3 / m2) - 1.0).abs() < 1e-14);
        assert!((bulk.effective_variance - (m4 * m2 / m3.powi(2) - 1.0)).abs() < 1e-14);
        let zero = bulk.at_mass_density(0.0).unwrap();
        assert_eq!(zero.extinction_m_inv, 0.0);
        let one = bulk.at_mass_density(1e-3).unwrap();
        let two = bulk.at_mass_density(2e-3).unwrap();
        assert_eq!(two.extinction_m_inv, 2.0 * one.extinction_m_inv);
        assert!((one.extinction_m_inv - one.scattering_m_inv - one.absorption_m_inv).abs() < 1e-13);
    }
    #[test]
    fn invalid_or_unresolved_populations_fail_instead_of_selecting_a_default() {
        let node = ParticleNumberNode {
            radius_m: 10e-6,
            number_weight: 1.0,
        };
        assert!(LiquidPopulationOptics::new(0.64, 1.0, &[]).is_err());
        assert!(
            LiquidPopulationOptics::new(
                0.64,
                1.0,
                &[ParticleNumberNode {
                    number_weight: 0.0,
                    ..node
                }]
            )
            .is_err()
        );
        for radius in [0.0, -1.0, f64::NAN, 1e-9, 1.0] {
            assert!(
                LiquidPopulationOptics::new(
                    0.64,
                    1.0,
                    &[ParticleNumberNode {
                        radius_m: radius,
                        ..node
                    }]
                )
                .is_err()
            );
        }
        for weight in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(
                LiquidPopulationOptics::new(
                    0.64,
                    1.0,
                    &[ParticleNumberNode {
                        number_weight: weight,
                        ..node
                    }]
                )
                .is_err()
            );
        }
        for lambda in [0.399, 1.001, f64::NAN] {
            assert!(LiquidPopulationOptics::new(lambda, 1.0, &[node]).is_err());
        }
        for medium in [0.0, 0.999, 1.011, f64::NAN] {
            assert!(LiquidPopulationOptics::new(0.64, medium, &[node]).is_err());
        }
        let bulk = LiquidPopulationOptics::new(0.64, 1.0, &[node])
            .unwrap()
            .bulk();
        for mass in [-1.0, f64::NAN, f64::INFINITY, f64::MAX] {
            assert!(bulk.at_mass_density(mass).is_err());
        }
    }
}
