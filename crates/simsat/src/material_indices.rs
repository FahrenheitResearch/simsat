//! Visible material refractive indices for independently specified cloud optics.
//!
//! Reference-condition liquid water (Segelstein 1981) and ice Ih (Warren/Brandt
//! 2008), not a temperature-dependent microphysics or ice-habit model. The
//! supported wavelength range is 0.4..=1.0 um. Tables include every native knot
//! bracketing that interval. See assets/material_indices/README.md.
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisibleMaterial {
    LiquidWaterSegelstein1981,
    IceIhWarrenBrandt2008,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefractiveIndex {
    pub real: f64,
    /// Positive imaginary part: m = n + i*k with exp(i*(k_wave*z - omega*t)).
    pub imaginary: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaterialIndexNode {
    pub wavelength_um: f64,
    pub index: RefractiveIndex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterialWavelengthError;
impl std::fmt::Display for MaterialWavelengthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("material index requires a finite wavelength within 0.4..=1.0 um")
    }
}
impl std::error::Error for MaterialWavelengthError {}

impl VisibleMaterial {
    pub fn nodes(self) -> &'static [MaterialIndexNode] {
        static WATER: OnceLock<Vec<MaterialIndexNode>> = OnceLock::new();
        static ICE: OnceLock<Vec<MaterialIndexNode>> = OnceLock::new();
        let cell = match self {
            Self::LiquidWaterSegelstein1981 => &WATER,
            Self::IceIhWarrenBrandt2008 => &ICE,
        };
        cell.get_or_init(|| {
            let nodes: Vec<MaterialIndexNode> = self
                .asset()
                .lines()
                .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
                .map(|line| {
                    let values: Vec<f64> = line
                        .split_whitespace()
                        .map(|v| v.parse().expect("checked-in material index number"))
                        .collect();
                    assert_eq!(values.len(), 3, "checked-in material index columns");
                    assert!(values.iter().all(|v| v.is_finite() && *v > 0.0));
                    MaterialIndexNode {
                        wavelength_um: values[0],
                        index: RefractiveIndex {
                            real: values[1],
                            imaginary: values[2],
                        },
                    }
                })
                .collect();
            assert!(nodes.len() >= 2);
            assert!(
                nodes
                    .windows(2)
                    .all(|v| v[0].wavelength_um < v[1].wavelength_um)
            );
            assert!(nodes[0].wavelength_um <= 0.4 && nodes[nodes.len() - 1].wavelength_um >= 1.0);
            nodes
        })
    }

    /// Adjacent authoritative knots for CPU/GPU spectral preparation. A GPU
    /// interpolation kernel consumes the same knots; it does not invent a new
    /// refractive-index fit or resample the material spectrum onto RGB channels.
    pub fn bracket(
        self,
        wavelength_um: f64,
    ) -> Result<[MaterialIndexNode; 2], MaterialWavelengthError> {
        if !wavelength_um.is_finite() || !(0.4..=1.0).contains(&wavelength_um) {
            return Err(MaterialWavelengthError);
        }
        let nodes = self.nodes();
        let upper = nodes
            .partition_point(|n| n.wavelength_um < wavelength_um)
            .clamp(1, nodes.len() - 1);
        Ok([nodes[upper - 1], nodes[upper]])
    }

    /// The published interpolation: real index is linear in log wavelength;
    /// log imaginary index is linear in log wavelength. No extrapolation.
    pub fn at(self, wavelength_um: f64) -> Result<RefractiveIndex, MaterialWavelengthError> {
        let [a, b] = self.bracket(wavelength_um)?;
        if wavelength_um == a.wavelength_um {
            return Ok(a.index);
        }
        if wavelength_um == b.wavelength_um {
            return Ok(b.index);
        }
        let t = (wavelength_um / a.wavelength_um).ln() / (b.wavelength_um / a.wavelength_um).ln();
        Ok(RefractiveIndex {
            real: a.index.real + t * (b.index.real - a.index.real),
            imaginary: a.index.imaginary * (t * (b.index.imaginary / a.index.imaginary).ln()).exp(),
        })
    }

    fn asset(self) -> &'static str {
        match self {
            Self::LiquidWaterSegelstein1981 => {
                include_str!("../assets/material_indices/water-segelstein-1981-visible.txt")
            }
            Self::IceIhWarrenBrandt2008 => {
                include_str!("../assets/material_indices/ice-ih-warren-brandt-2008-visible.txt")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    #[test]
    fn material_sources_are_pinned_and_native_knots_recover_exactly() {
        for (material, count, hash) in [
            (
                VisibleMaterial::LiquidWaterSegelstein1981,
                121,
                "08f70d712526a8cda3054b03b76187563ca851debdf52d7103bb77d77b4c2d70",
            ),
            (
                VisibleMaterial::IceIhWarrenBrandt2008,
                61,
                "cdee3ca276805f0ade063a9d3a4840c7cd20549302dcc8a6e35e8201009d09d9",
            ),
        ] {
            assert_eq!(
                format!("{:x}", Sha256::digest(material.asset().as_bytes())),
                hash
            );
            assert_eq!(material.nodes().len(), count);
            for node in material
                .nodes()
                .iter()
                .filter(|n| (0.4..=1.0).contains(&n.wavelength_um))
            {
                assert_eq!(material.at(node.wavelength_um).unwrap(), node.index);
            }
        }
        // Independently printed Warren/Brandt visible table values.
        let ice = VisibleMaterial::IceIhWarrenBrandt2008;
        assert_eq!(
            ice.at(0.47).unwrap(),
            RefractiveIndex {
                real: 1.3145,
                imaginary: 1.956e-10
            }
        );
        assert_eq!(
            ice.at(0.64).unwrap(),
            RefractiveIndex {
                real: 1.3083,
                imaginary: 1.220e-8
            }
        );
    }
    #[test]
    fn log_interpolation_is_positive_and_matches_geometric_midpoints() {
        for material in [
            VisibleMaterial::LiquidWaterSegelstein1981,
            VisibleMaterial::IceIhWarrenBrandt2008,
        ] {
            for pair in material.nodes().windows(2) {
                let [a, b] = [pair[0], pair[1]];
                let wavelength = (a.wavelength_um * b.wavelength_um).sqrt();
                if wavelength < 0.4 {
                    continue;
                }
                let index = material.at(wavelength).unwrap();
                assert!((index.real - 0.5 * (a.index.real + b.index.real)).abs() < 1e-13);
                assert!(
                    (index.imaginary / (a.index.imaginary * b.index.imaginary).sqrt() - 1.0).abs()
                        < 1e-12
                );
                assert!(index.imaginary > 0.0);
            }
        }
    }
    #[test]
    fn material_indices_cover_every_abi_visible_node_without_extrapolation() {
        use crate::visible_sensor::AbiReflectiveBand;
        use crate::visible_solar::AbiSolarResponse;
        for band in [
            AbiReflectiveBand::C01,
            AbiReflectiveBand::C02,
            AbiReflectiveBand::C03,
        ] {
            for node in AbiSolarResponse::for_band(band).nodes() {
                for material in [
                    VisibleMaterial::LiquidWaterSegelstein1981,
                    VisibleMaterial::IceIhWarrenBrandt2008,
                ] {
                    let n = material.at(node.wavelength_um).unwrap();
                    assert!((1.3..1.35).contains(&n.real) && n.imaginary > 0.0);
                }
            }
        }
        for invalid in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            0.3999,
            1.0001,
        ] {
            assert!(
                VisibleMaterial::LiquidWaterSegelstein1981
                    .at(invalid)
                    .is_err()
            );
        }
    }
    #[test]
    fn material_index_wgsl_is_valid() {
        let module =
            naga::front::wgsl::parse_str(include_str!("gpu/shaders/material_indices.wgsl"))
                .unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}
