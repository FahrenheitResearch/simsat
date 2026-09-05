//! First-order scalar transport through supplied optical ray segments.
//!
//! Each segment is homogeneous along the viewing ray, and direct-sun optical
//! depth varies linearly between its endpoints. Integrating that exponential
//! source analytically avoids midpoint bias, including when sun and viewing
//! attenuation cancel. Geometry, wavelengths and model optics are supplied by
//! the caller. They are not inferred from RGB or from observed radiances.
//!
//! This is the DIRECT single-scattering term, not a complete cloud renderer.
//! Diffuse illumination, atmospheric/surface multiple scattering, polarization
//! and the finite solar disk require separate terms. Earth/terrain shadow
//! boundaries must split the ray; a segment may be entirely lit or entirely dark.
//! This explicit decomposition is intended to make later multiple-scattering
//! closures testable against an exact first-order contribution.

use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportError {
    InvalidOpticalDepth,
    InvalidScatteringDepth,
    InvalidPhase,
    InvalidSurface,
    NonFiniteResult,
}
impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::InvalidOpticalDepth => "optical depth must be finite and nonnegative",
            Self::InvalidScatteringDepth => {
                "scattering optical depth must lie between zero and extinction optical depth"
            }
            Self::InvalidPhase => "phase density must be finite and nonnegative (sr^-1)",
            Self::InvalidSurface => {
                "Lambertian albedo and illuminated surface-incidence cosine must be within 0..=1"
            }
            Self::NonFiniteResult => "single-scattering accumulation overflowed",
        })
    }
}
impl std::error::Error for TransportError {}

/// Solar extinction depths from the Sun to the near and far segment endpoints.
/// Near/far refer to distance from the sensor, not proximity to the Sun. An
/// entirely Earth/terrain-shadowed segment is represented by `None` instead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarDepthEndpoints {
    near: f64,
    far: f64,
}
impl SolarDepthEndpoints {
    pub fn new(near: f64, far: f64) -> Result<Self, TransportError> {
        valid_tau(near)?;
        valid_tau(far)?;
        Ok(Self { near, far })
    }
    pub fn near(self) -> f64 {
        self.near
    }
    pub fn far(self) -> f64 {
        self.far
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SingleScatterSegment {
    extinction_depth: f64,
    scattering_depth: f64,
    phase_sr1: f64,
    solar: Option<SolarDepthEndpoints>,
}
impl SingleScatterSegment {
    pub fn new(
        extinction_depth: f64,
        scattering_depth: f64,
        phase_sr1: f64,
        solar: Option<SolarDepthEndpoints>,
    ) -> Result<Self, TransportError> {
        valid_tau(extinction_depth)?;
        if !scattering_depth.is_finite() || !(0.0..=extinction_depth).contains(&scattering_depth) {
            return Err(TransportError::InvalidScatteringDepth);
        }
        if !phase_sr1.is_finite() || phase_sr1 < 0.0 {
            return Err(TransportError::InvalidPhase);
        }
        Ok(Self {
            extinction_depth,
            scattering_depth,
            phase_sr1,
            solar,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectLambertianBoundary {
    albedo: f64,
    incidence_cosine: f64,
    solar_depth: Option<f64>,
}
impl DirectLambertianBoundary {
    /// `None` means the direct beam is blocked. This boundary excludes diffuse
    /// sky/cloud illumination and subsequent surface/atmosphere bounces.
    pub fn new(
        albedo: f64,
        incidence_cosine: f64,
        solar_depth: Option<f64>,
    ) -> Result<Self, TransportError> {
        if !(0.0..=1.0).contains(&albedo) || !(0.0..=1.0).contains(&incidence_cosine) {
            return Err(TransportError::InvalidSurface);
        }
        if let Some(t) = solar_depth {
            valid_tau(t)?;
        }
        Ok(Self {
            albedo,
            incidence_cosine,
            solar_depth,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SingleScatterResult {
    /// L_lambda / E_lambda (sr^-1), from photons scattered exactly once in air/cloud.
    pub scattered_normalized_radiance_sr1: f64,
    /// Directly lit Lambertian boundary, attenuated along the entire viewing ray.
    pub surface_normalized_radiance_sr1: f64,
    pub view_transmittance: f64,
}
impl SingleScatterResult {
    pub fn total_normalized_radiance_sr1(self) -> f64 {
        self.scattered_normalized_radiance_sr1 + self.surface_normalized_radiance_sr1
    }
}

/// March ordered NEAR-TO-FAR optical segments. Wavelength-specific callers can
/// feed the resulting normalized radiance directly to `visible_solar` quadrature.
pub fn integrate_single_scatter(
    segments: impl IntoIterator<Item = SingleScatterSegment>,
    boundary: Option<DirectLambertianBoundary>,
) -> Result<SingleScatterResult, TransportError> {
    let mut view_tau = 0.0;
    let mut scattered = 0.0;
    for segment in segments {
        if let Some(sun) = segment.solar {
            let a = view_tau + sun.near;
            let b = view_tau + segment.extinction_depth + sun.far;
            // Integral_0^1 exp[-lerp(a,b,x)] dx. Both endpoints are nonnegative,
            // so evaluating from the smaller avoids exp(+large)*exp(-large).
            let mean_t = mean_exponential_transmittance(a, b)?;
            scattered += segment.scattering_depth * mean_t * segment.phase_sr1;
        }
        view_tau += segment.extinction_depth;
        if !view_tau.is_finite() || !scattered.is_finite() {
            return Err(TransportError::NonFiniteResult);
        }
    }
    let mut surface = 0.0;
    if let Some(b) = boundary
        && let Some(t) = b.solar_depth
    {
        surface = b.albedo * b.incidence_cosine / PI * (-view_tau - t).exp();
    }
    Ok(SingleScatterResult {
        scattered_normalized_radiance_sr1: scattered,
        surface_normalized_radiance_sr1: surface,
        view_transmittance: (-view_tau).exp(),
    })
}

fn mean_exponential_transmittance(a: f64, b: f64) -> Result<f64, TransportError> {
    if !a.is_finite() || !b.is_finite() {
        return Err(TransportError::NonFiniteResult);
    }
    let span = (b - a).abs();
    let factor = if span == 0.0 {
        1.0
    } else {
        -(-span).exp_m1() / span
    };
    Ok((-a.min(b)).exp() * factor)
}
fn valid_tau(t: f64) -> Result<(), TransportError> {
    if t.is_finite() && t >= 0.0 {
        Ok(())
    } else {
        Err(TransportError::InvalidOpticalDepth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral_molecular::DryAirRayleigh;
    use crate::visible_sensor::AbiReflectiveBand;
    use crate::visible_solar::AbiSolarResponse;

    #[test]
    fn homogeneous_slab_matches_exact_first_order_solution() {
        // Scalar RTE analytic slab solution; also the independent MC oracle's
        // single-scatter reference. Unlike BRF=pi*L/(E*mu0), ABI rho_f=pi*L/E.
        let phase = DryAirRayleigh::new(0.47, 360.0)
            .unwrap()
            .phase_sr1(-0.5)
            .unwrap();
        for tau in [0.0, 1e-10, 0.01, 0.1, 1.0, 30.0, 10000.0] {
            for mu0 in [0.05, 0.4, 1.0] {
                for muv in [0.1, 0.7, 1.0] {
                    for n in [1, 7, 64] {
                        let s = (0..n).map(|k| {
                            SingleScatterSegment::new(
                                tau / muv / n as f64,
                                0.8 * tau / muv / n as f64,
                                phase,
                                Some(
                                    SolarDepthEndpoints::new(
                                        tau / mu0 * k as f64 / n as f64,
                                        tau / mu0 * (k + 1) as f64 / n as f64,
                                    )
                                    .unwrap(),
                                ),
                            )
                            .unwrap()
                        });
                        let got = integrate_single_scatter(s, None)
                            .unwrap()
                            .scattered_normalized_radiance_sr1;
                        let exact = 0.8 * phase * mu0 / (mu0 + muv)
                            * -(-tau * (1.0 / mu0 + 1.0 / muv)).exp_m1();
                        assert!(
                            (got - exact).abs() < 2e-14,
                            "tau {tau}, mu0 {mu0}, muv {muv}, n {n}: {got} vs {exact}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn cancellation_and_opaque_reverse_sun_paths_stay_finite() {
        let p = 0.1;
        // Solar optical depth falls by exactly the rise in view optical depth.
        let s = SingleScatterSegment::new(
            4.0,
            2.0,
            p,
            Some(SolarDepthEndpoints::new(4.0, 0.0).unwrap()),
        )
        .unwrap();
        let got = integrate_single_scatter([s], None)
            .unwrap()
            .scattered_normalized_radiance_sr1;
        assert!((got - 2.0 * p * (-4.0f64).exp()).abs() < 1e-15);
        let s = SingleScatterSegment::new(
            1.0,
            0.5,
            p,
            Some(SolarDepthEndpoints::new(10000.0, 0.0).unwrap()),
        )
        .unwrap();
        let got = integrate_single_scatter([s], None)
            .unwrap()
            .scattered_normalized_radiance_sr1;
        assert!((got - 0.5 * p * (-1.0f64).exp() / 9999.0).abs() < 1e-15);
    }

    #[test]
    fn blocked_sun_and_pure_absorption_have_no_scattered_source() {
        for s in [
            SingleScatterSegment::new(2.0, 1.0, 0.1, None).unwrap(),
            SingleScatterSegment::new(
                2.0,
                0.0,
                0.1,
                Some(SolarDepthEndpoints::new(0.0, 0.0).unwrap()),
            )
            .unwrap(),
        ] {
            let result = integrate_single_scatter([s], None).unwrap();
            assert_eq!(result.scattered_normalized_radiance_sr1, 0.0);
            assert_eq!(result.view_transmittance, (-2.0f64).exp());
        }
    }

    #[test]
    fn clear_surface_and_full_band_normalization_join_without_rgb() {
        // End-to-end spectral source -> exact first-order transport -> full-SRF
        // solar quadrature. A black atmosphere has no unrequested ambient term.
        for band in AbiReflectiveBand::ALL {
            let solar = AbiSolarResponse::for_band(band);
            let rho = solar
                .reflectance_factor_from_normalized_radiance(|_| {
                    let boundary = DirectLambertianBoundary::new(0.3, 0.5, Some(0.0)).unwrap();
                    integrate_single_scatter([], Some(boundary))
                        .unwrap()
                        .total_normalized_radiance_sr1()
                })
                .unwrap();
            assert!((rho - 0.15).abs() < 1e-13);
        }
        let s = SingleScatterSegment::new(0.2, 0.0, 0.0, None).unwrap();
        let b = DirectLambertianBoundary::new(0.3, 0.5, Some(0.4)).unwrap();
        let got = integrate_single_scatter([s], Some(b))
            .unwrap()
            .surface_normalized_radiance_sr1;
        assert!((got - 0.15 / PI * (-0.6f64).exp()).abs() < 1e-15);
        assert_eq!(
            integrate_single_scatter(
                [],
                Some(DirectLambertianBoundary::new(1.0, 1.0, None).unwrap())
            )
            .unwrap()
            .surface_normalized_radiance_sr1,
            0.0
        );
    }

    #[test]
    fn invalid_optics_are_rejected() {
        assert!(SolarDepthEndpoints::new(f64::NAN, 0.0).is_err());
        assert!(SolarDepthEndpoints::new(0.0, -1.0).is_err());
        assert!(SingleScatterSegment::new(-1.0, 0.0, 0.1, None).is_err());
        assert!(SingleScatterSegment::new(1.0, 1.1, 0.1, None).is_err());
        assert!(SingleScatterSegment::new(1.0, 0.5, f64::INFINITY, None).is_err());
        assert!(DirectLambertianBoundary::new(1.1, 0.5, None).is_err());
        assert!(DirectLambertianBoundary::new(1.0, f64::NAN, None).is_err());
        let s = SingleScatterSegment::new(f64::MAX, 0.0, 0.0, None).unwrap();
        assert!(integrate_single_scatter([s, s], None).is_err());
    }

    #[test]
    fn transport_wgsl_validates() {
        let module =
            naga::front::wgsl::parse_str(include_str!("gpu/shaders/spectral_transport.wgsl"))
                .unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}
