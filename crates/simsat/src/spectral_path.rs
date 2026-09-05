//! Scalar backward Monte Carlo transport with explicit surface reflections.
//!
//! Sources: Mayer (2009), https://elib.dlr.de/59646/1/2009-Mayer_A9RA3_tmp.pdf
//! (free optical paths, local estimation, backward transport), and the existing
//! normalized HG / depolarizing-Rayleigh and Cox-Munk models. No scattering-order
//! brightness closure is used. Russian roulette preserves expectation; hitting
//! the separate event safety bound is an error, never a silently truncated image.
//!
//! The sampling math has a WGSL counterpart. Scene traversal is supplied by the
//! caller so an independent analytic slab can test the SAME transport loop.
use crate::{clouds, optics};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn unit(a: [f64; 3]) -> [f64; 3] {
    let norm = dot(a, a).sqrt();
    a.map(|v| v / norm)
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
pub fn direction_about(axis: [f64; 3], cosine: f64, azimuth: f64) -> [f64; 3] {
    let helper = if axis[2].abs() < 0.9 {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let x = unit(cross(helper, axis));
    let y = cross(axis, x);
    let sine = (1.0 - cosine * cosine).max(0.0).sqrt();
    unit(std::array::from_fn(|i| {
        cosine * axis[i] + sine * (azimuth.cos() * x[i] + azimuth.sin() * y[i])
    }))
}

/// PCG RXS-M-XS permutation of a full-period 32-bit LCG. The 23-bit open-interval
/// conversion is exactly representable in f32, including its two endpoints.
/// CPU and WGSL consume the same sequence; no scheduling-dependent global RNG.
#[derive(Clone, Copy, Debug)]
pub struct Random {
    state: u32,
}
impl Random {
    pub fn new(seed: u32) -> Self {
        Self { state: seed }
    }
    pub fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(747796405).wrapping_add(2891336453);
        let word = ((self.state >> ((self.state >> 28) + 4)) ^ self.state).wrapping_mul(277803737);
        (word >> 22) ^ word
    }
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u32() >> 9) as f64 + 0.5) / 8388608.0
    }
    pub fn optical_depth(&mut self) -> f64 {
        -self.uniform().ln()
    }
}

/// A normalized mixture: one depolarizing Rayleigh term and up to four HG lobes.
#[derive(Clone, Copy, Debug)]
pub struct PhaseFunction {
    pub rayleigh_weight: f64,
    pub gamma: f64,
    pub hg_g: [f64; 4],
    pub hg_weight: [f64; 4],
}
impl PhaseFunction {
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=0.2).contains(&self.gamma)
            || !self.rayleigh_weight.is_finite()
            || self.rayleigh_weight < 0.0
            || self.hg_g.iter().any(|g| !g.is_finite() || g.abs() >= 1.0)
            || self.hg_weight.iter().any(|w| !w.is_finite() || *w < 0.0)
            || (self.rayleigh_weight + self.hg_weight.iter().sum::<f64>() - 1.0).abs() > 1e-10
        {
            return Err("invalid normalized scattering phase mixture".into());
        }
        Ok(())
    }
    pub fn dual_hg(g1: f64, g2: f64, w: f64) -> Self {
        Self {
            rayleigh_weight: 0.0,
            gamma: 0.0,
            hg_g: [g1, g2, 0.0, 0.0],
            hg_weight: [w, 1.0 - w, 0.0, 0.0],
        }
    }
    pub fn rayleigh(gamma: f64) -> Self {
        Self {
            rayleigh_weight: 1.0,
            gamma,
            hg_g: [0.0; 4],
            hg_weight: [0.0; 4],
        }
    }
    pub fn model_mixture(
        rayleigh: f64,
        liquid: f64,
        ice_precip: f64,
        gamma: f64,
    ) -> Result<Self, String> {
        let total = rayleigh + liquid + ice_precip;
        if [rayleigh, liquid, ice_precip]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
            || total <= 0.0
        {
            return Err("invalid spectral scattering coefficients".into());
        }
        let p = Self {
            rayleigh_weight: rayleigh / total,
            gamma,
            hg_g: [
                clouds::PHASE_LIQUID_G1,
                clouds::PHASE_LIQUID_G2,
                clouds::PHASE_ICE_G1,
                clouds::PHASE_ICE_G2,
            ],
            hg_weight: [
                liquid * clouds::PHASE_LIQUID_W / total,
                liquid * (1.0 - clouds::PHASE_LIQUID_W) / total,
                ice_precip * clouds::PHASE_ICE_W / total,
                ice_precip * (1.0 - clouds::PHASE_ICE_W) / total,
            ],
        };
        p.validate()?;
        Ok(p)
    }
    pub fn value(&self, cosine: f64) -> f64 {
        let mu = cosine.clamp(-1.0, 1.0);
        let rayleigh = 3.0 / (16.0 * PI * (1.0 + 2.0 * self.gamma))
            * ((1.0 + 3.0 * self.gamma) + (1.0 - self.gamma) * mu * mu);
        self.rayleigh_weight * rayleigh
            + self
                .hg_g
                .iter()
                .zip(&self.hg_weight)
                .map(|(&g, &w)| w * clouds::henyey_greenstein(mu, g))
                .sum::<f64>()
    }
    /// Cosine between incoming and outgoing photon PROPAGATION directions.
    /// In backward tracing this is also dot(old_path_direction,new_direction).
    pub fn sample_cosine(&self, choose: f64, u: f64) -> f64 {
        if choose < self.rayleigh_weight {
            // Rayleigh is an exact mixture of uniform mu and density 3*mu^2/2.
            let isotropic = 3.0 * (1.0 + 3.0 * self.gamma) / (4.0 * (1.0 + 2.0 * self.gamma));
            let inside = choose / self.rayleigh_weight;
            return if inside < isotropic {
                2.0 * u - 1.0
            } else {
                (2.0 * u - 1.0).cbrt()
            };
        }
        let mut cumulative = self.rayleigh_weight;
        for (&g, &w) in self.hg_g.iter().zip(&self.hg_weight) {
            cumulative += w;
            if choose < cumulative {
                if g.abs() < 0.01 {
                    // Algebraic form of the same inverse CDF, avoiding the
                    // subtraction of almost equal numbers around g = 0.
                    let x = 2.0 * u - 1.0;
                    let numerator =
                        x + 0.5 * g * (x * x + 3.0) + g * g * x + 0.5 * g * g * g * (x * x - 1.0);
                    return (numerator / (1.0 + g * x).powi(2)).clamp(-1.0, 1.0);
                }
                let q = (1.0 - g * g) / (1.0 - g + 2.0 * g * u);
                return ((1.0 + g * g - q * q) / (2.0 * g)).clamp(-1.0, 1.0);
            }
        }
        // Only roundoff at the normalized cumulative endpoint can reach here.
        1.0
    }
    pub fn sample(&self, direction: [f64; 3], rng: &mut Random) -> [f64; 3] {
        let cosine = self.sample_cosine(rng.uniform(), rng.uniform());
        direction_about(direction, cosine, 2.0 * PI * rng.uniform())
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Material {
    Lambertian {
        albedo: f64,
    },
    /// Same reciprocal isotropic Cox-Munk BRDF as the existing reference. Its
    /// missing shadowing/masking, whitecaps and water-leaving light remain explicit.
    CoxMunk {
        mean_square_slope: f64,
    },
}
impl Material {
    pub fn validate(&self) -> Result<(), String> {
        let valid = match *self {
            Self::Lambertian { albedo } => (0.0..=1.0).contains(&albedo),
            Self::CoxMunk { mean_square_slope } => {
                mean_square_slope.is_finite() && mean_square_slope > 0.0
            }
        };
        if valid {
            Ok(())
        } else {
            Err("invalid physical surface material".into())
        }
    }
    pub fn sun_source(&self, sun: [f64; 3], to_view: [f64; 3], normal: [f64; 3]) -> f64 {
        let mu0 = dot(sun, normal).max(0.0);
        match *self {
            Self::Lambertian { albedo } => albedo * mu0 / PI,
            Self::CoxMunk { mean_square_slope } => {
                optics::cox_munk_glint_reflectance(sun, to_view, normal, mean_square_slope) * mu0
                    / PI
            }
        }
    }
    /// Returns the sampled reflected direction and BRDF*cos(theta)/PDF weight.
    /// A rejected below-horizon facet contributes zero; it is NOT resampled.
    pub fn sample(
        &self,
        to_view: [f64; 3],
        normal: [f64; 3],
        rng: &mut Random,
    ) -> Option<([f64; 3], f64)> {
        if dot(to_view, normal) <= 0.0 {
            return None;
        }
        match *self {
            Self::Lambertian { albedo } => {
                if albedo <= 0.0 {
                    return None;
                }
                Some((
                    direction_about(normal, rng.uniform().sqrt(), 2.0 * PI * rng.uniform()),
                    albedo,
                ))
            }
            Self::CoxMunk {
                mean_square_slope: mss,
            } => {
                if dot(to_view, normal) <= 1e-4 {
                    return None;
                }
                let mss = mss.max(1e-4);
                let tan2 = -mss * rng.uniform().ln();
                let ch = (1.0 + tan2).sqrt().recip();
                let facet = direction_about(normal, ch, 2.0 * PI * rng.uniform());
                let vh = dot(to_view, facet);
                if vh <= 0.0 {
                    return None;
                }
                let incoming = unit(std::array::from_fn(|i| 2.0 * vh * facet[i] - to_view[i]));
                let mui = dot(incoming, normal);
                if mui <= 1e-4 {
                    return None;
                }
                let slope_pdf = (-tan2 / mss).exp() / (PI * mss);
                let pdf = slope_pdf / ch.powi(3) / (4.0 * vh);
                let brdf = optics::cox_munk_glint_reflectance(incoming, to_view, normal, mss) / PI;
                Some((incoming, brdf * mui / pdf))
            }
        }
    }
}

pub enum Event {
    Escape,
    Scatter {
        point: [f64; 3],
        phase: PhaseFunction,
        single_scatter_albedo: f64,
    },
    Surface {
        point: [f64; 3],
        normal: [f64; 3],
        material: Material,
    },
}
pub struct EventWithSupport {
    pub event: Event,
    pub flags: u8,
}
pub trait Scene {
    fn sun_direction(&self) -> [f64; 3];
    fn next_event(
        &self,
        point: [f64; 3],
        direction: [f64; 3],
        optical_depth: f64,
    ) -> Result<EventWithSupport, String>;
    fn sun_transmittance(&self, point: [f64; 3]) -> Result<(f64, u8), String>;
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PathConfig {
    pub roulette_start_order: usize,
    /// Roulette only terminates attenuated weights below this threshold.
    /// Conservative unit-weight paths are never killed/reweighted: a fixed
    /// survival cap can produce unbounded variance in optically thick clouds.
    pub roulette_weight_threshold: f64,
    /// A safety failure, NOT a scattering-order approximation or truncation.
    pub event_safety_limit: usize,
}
impl PathConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.roulette_start_order == 0
            || self.event_safety_limit <= self.roulette_start_order
            || !self.roulette_weight_threshold.is_finite()
            || self.roulette_weight_threshold <= 0.0
            || self.roulette_weight_threshold >= 1.0
        {
            return Err(
                "invalid explicit Monte Carlo roulette or event-safety configuration".into(),
            );
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct PathResult {
    pub first_order_l_over_e: f64,
    pub higher_order_l_over_e: f64,
    pub events: usize,
    pub surface_events: usize,
    pub flags: u8,
}
impl PathResult {
    pub fn total(self) -> f64 {
        self.first_order_l_over_e + self.higher_order_l_over_e
    }
}
pub fn trace(
    scene: &impl Scene,
    point: [f64; 3],
    direction: [f64; 3],
    cfg: PathConfig,
    rng: &mut Random,
) -> Result<PathResult, String> {
    cfg.validate()?;
    let mut result = PathResult::default();
    let (mut p, mut d, mut weight) = (point, direction, 1.0);
    for order in 1..=cfg.event_safety_limit {
        let step = scene.next_event(p, d, rng.optical_depth())?;
        result.flags |= step.flags;
        let (source, next) = match step.event {
            Event::Escape => return Ok(result),
            Event::Scatter {
                point,
                phase,
                single_scatter_albedo: ssa,
            } => {
                if !(0.0..=1.0).contains(&ssa) {
                    return Err("invalid single-scatter albedo".into());
                }
                phase.validate()?;
                weight *= ssa;
                if weight == 0.0 {
                    return Ok(result);
                }
                let (sun_t, flags) = scene.sun_transmittance(point)?;
                result.flags |= flags;
                p = point;
                let source = phase.value(dot(d, scene.sun_direction())) * sun_t;
                (source, Some((phase.sample(d, rng), 1.0)))
            }
            Event::Surface {
                point,
                normal,
                material,
            } => {
                let (sun_t, flags) = scene.sun_transmittance(point)?;
                result.flags |= flags;
                p = point;
                result.surface_events += 1;
                material.validate()?;
                let to_view = d.map(|v| -v);
                (
                    material.sun_source(scene.sun_direction(), to_view, normal) * sun_t,
                    material.sample(to_view, normal, rng),
                )
            }
        };
        result.events = order;
        if order == 1 {
            result.first_order_l_over_e += weight * source;
        } else {
            result.higher_order_l_over_e += weight * source;
        }
        let Some((new_direction, bounce_weight)) = next else {
            return Ok(result);
        };
        d = new_direction;
        weight *= bounce_weight;
        if !weight.is_finite() || weight < 0.0 || !result.total().is_finite() {
            return Err("non-finite Monte Carlo path weight".into());
        }
        if weight == 0.0 {
            return Ok(result);
        }
        if order >= cfg.roulette_start_order && weight < cfg.roulette_weight_threshold {
            let survival = weight;
            if rng.uniform() >= survival {
                return Ok(result);
            }
            weight /= survival;
        }
    }
    Err("Monte Carlo event safety limit reached; refusing biased truncation".into())
}

/// Online independent-path moments. Standard error is undefined for one path.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Moments {
    pub count: usize,
    pub mean: f64,
    pub m2: f64,
}
impl Moments {
    pub fn push(&mut self, value: f64) {
        self.count += 1;
        let d = value - self.mean;
        self.mean += d / self.count as f64;
        self.m2 += d * (value - self.mean);
    }
    pub fn standard_error(&self) -> Option<f64> {
        (self.count > 1).then(|| (self.m2.max(0.0) / (self.count * (self.count - 1)) as f64).sqrt())
    }
}

/// Homogeneous, plane-parallel external-reference scene, in vertical optical
/// depth units. It exercises the same all-order path loop as the 3D renderer.
#[derive(Clone, Copy, Debug)]
pub struct HomogeneousSlab {
    pub tau: f64,
    pub single_scatter_albedo: f64,
    pub phase: PhaseFunction,
    pub solar_cosine: f64,
    pub albedo: f64,
}
impl HomogeneousSlab {
    pub fn validate(&self) -> Result<(), String> {
        self.phase.validate()?;
        if !self.tau.is_finite()
            || self.tau < 0.0
            || !(0.0..=1.0).contains(&self.single_scatter_albedo)
            || !(0.0..=1.0).contains(&self.albedo)
            || self.solar_cosine <= 0.0
            || self.solar_cosine > 1.0
            || !self.solar_cosine.is_finite()
        {
            return Err("invalid explicit slab reference".into());
        }
        Ok(())
    }
    /// DISORT azimuth is measured relative to the incident beam's horizontal
    /// propagation. This gives cos(scatter)=-mu0*muv+sin0*sinv*cos(relative az).
    pub fn view_direction(view_cosine: f64, relative_azimuth_rad: f64) -> [f64; 3] {
        let s = (1.0 - view_cosine * view_cosine).max(0.0).sqrt();
        [
            s * relative_azimuth_rad.cos(),
            s * relative_azimuth_rad.sin(),
            -view_cosine,
        ]
    }
}
impl Scene for HomogeneousSlab {
    fn sun_direction(&self) -> [f64; 3] {
        [
            (1.0 - self.solar_cosine * self.solar_cosine).sqrt(),
            0.0,
            self.solar_cosine,
        ]
    }
    fn next_event(
        &self,
        p: [f64; 3],
        d: [f64; 3],
        optical_depth: f64,
    ) -> Result<EventWithSupport, String> {
        let boundary = if d[2] > 0.0 {
            (self.tau - p[2]).max(0.0) / d[2]
        } else if d[2] < 0.0 {
            -p[2].max(0.0) / d[2]
        } else {
            f64::INFINITY
        };
        let event = if optical_depth < boundary {
            Event::Scatter {
                point: std::array::from_fn(|i| p[i] + optical_depth * d[i]),
                phase: self.phase,
                single_scatter_albedo: self.single_scatter_albedo,
            }
        } else if d[2] > 0.0 {
            Event::Escape
        } else {
            Event::Surface {
                point: [p[0] + boundary * d[0], p[1] + boundary * d[1], 0.0],
                normal: [0.0, 0.0, 1.0],
                material: Material::Lambertian {
                    albedo: self.albedo,
                },
            }
        };
        Ok(EventWithSupport { event, flags: 0 })
    }
    fn sun_transmittance(&self, p: [f64; 3]) -> Result<(f64, u8), String> {
        Ok(((-(self.tau - p[2]).max(0.0) / self.solar_cosine).exp(), 0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> PathConfig {
        PathConfig {
            roulette_start_order: 16,
            roulette_weight_threshold: 0.95,
            event_safety_limit: 100000,
        }
    }
    #[test]
    fn phase_sampling_matches_independent_angular_moments() {
        let mut rng = Random::new(831709);
        for phase in [
            PhaseFunction::rayleigh(0.0),
            PhaseFunction::rayleigh(0.014),
            PhaseFunction::dual_hg(0.85, -0.15, 0.9),
            PhaseFunction::dual_hg(0.75, -0.10, 0.9),
            PhaseFunction::model_mixture(0.2, 0.3, 0.5, 0.014).unwrap(),
        ] {
            phase.validate().unwrap();
            let mut first = Moments::default();
            let mut second = Moments::default();
            for _ in 0..200000 {
                let c = phase.sample_cosine(rng.uniform(), rng.uniform());
                assert!((-1.0..=1.0).contains(&c));
                first.push(c);
                second.push(0.5 * (3.0 * c * c - 1.0));
            }
            let expected_first: f64 = phase
                .hg_g
                .iter()
                .zip(&phase.hg_weight)
                .map(|(g, w)| g * w)
                .sum();
            let expected_second: f64 = phase.rayleigh_weight * (1.0 - phase.gamma)
                / (10.0 * (1.0 + 2.0 * phase.gamma))
                + phase
                    .hg_g
                    .iter()
                    .zip(&phase.hg_weight)
                    .map(|(g, w)| g * g * w)
                    .sum::<f64>();
            assert!((first.mean - expected_first).abs() < 6.0 * first.standard_error().unwrap());
            assert!((second.mean - expected_second).abs() < 6.0 * second.standard_error().unwrap());
        }
    }
    #[test]
    fn vacuum_lambertian_has_one_solar_cosine_and_no_random_brightness() {
        let mut rng = Random::new(1297);
        for mu0 in [0.01, 0.5, 1.0] {
            let slab = HomogeneousSlab {
                tau: 0.0,
                single_scatter_albedo: 1.0,
                phase: PhaseFunction::rayleigh(0.0),
                solar_cosine: mu0,
                albedo: 0.2,
            };
            slab.validate().unwrap();
            for muv in [0.2, 0.8, 1.0] {
                for _ in 0..100 {
                    let result = trace(
                        &slab,
                        [0.0, 0.0, 0.0],
                        HomogeneousSlab::view_direction(muv, 0.0),
                        config(),
                        &mut rng,
                    )
                    .unwrap();
                    assert!((PI * result.total() - 0.2 * mu0).abs() < 1e-14);
                    assert_eq!(result.higher_order_l_over_e, 0.0);
                }
            }
        }
    }
    #[test]
    fn cox_munk_sampler_matches_independent_hemispherical_quadrature() {
        let normal = [0.0, 0.0, 1.0];
        let mss = optics::cox_munk_mean_square_slope(7.0);
        let material = Material::CoxMunk {
            mean_square_slope: mss,
        };
        let mut rng = Random::new(584219);
        for mu in [0.2, 0.6, 1.0] {
            let to_view = [(1.0_f64 - mu * mu).sqrt(), 0.0, mu];
            let mut expected = 0.0;
            let (n_mu, n_phi) = (400, 720);
            for m in 0..n_mu {
                let cosine = (m as f64 + 0.5) / n_mu as f64;
                for p in 0..n_phi {
                    let direction =
                        direction_about(normal, cosine, 2.0 * PI * (p as f64 + 0.5) / n_phi as f64);
                    expected += optics::cox_munk_glint_reflectance(direction, to_view, normal, mss)
                        / PI
                        * cosine
                        * 2.0
                        * PI
                        / (n_mu * n_phi) as f64;
                }
            }
            let mut measured = Moments::default();
            for _ in 0..200000 {
                measured.push(
                    material
                        .sample(to_view, normal, &mut rng)
                        .map_or(0.0, |(_, weight)| weight),
                );
            }
            assert!(
                (measured.mean - expected).abs() < 6.0 * measured.standard_error().unwrap() + 3e-6,
                "{mu} {} {expected}",
                measured.mean
            );
        }
    }
    #[test]
    fn rng_uniform_is_open_and_exactly_representable_in_f32() {
        let mut rng = Random::new(0);
        let mut m = Moments::default();
        for _ in 0..200000 {
            let u = rng.uniform();
            assert!(u > 0.0 && u < 1.0);
            assert_eq!(u, (u as f32) as f64);
            m.push(u);
        }
        assert!((m.mean - 0.5).abs() < 6.0 * m.standard_error().unwrap());
    }
    struct FixedCollisions {
        count: usize,
        ssa: f64,
    }
    impl Scene for FixedCollisions {
        fn sun_direction(&self) -> [f64; 3] {
            [0.0, 0.0, 1.0]
        }
        fn sun_transmittance(&self, _: [f64; 3]) -> Result<(f64, u8), String> {
            Ok((1.0, 0))
        }
        fn next_event(&self, p: [f64; 3], _: [f64; 3], _: f64) -> Result<EventWithSupport, String> {
            let order = p[0] as usize;
            Ok(EventWithSupport {
                event: if order >= self.count {
                    Event::Escape
                } else {
                    Event::Scatter {
                        point: [(order + 1) as f64, 0.0, 0.0],
                        phase: PhaseFunction::dual_hg(0.0, 0.0, 1.0),
                        single_scatter_albedo: self.ssa,
                    }
                },
                flags: 0,
            })
        }
    }
    #[test]
    fn conservative_paths_are_not_artificially_killed_by_roulette() {
        let scene = FixedCollisions {
            count: 100,
            ssa: 1.0,
        };
        let cfg = PathConfig {
            roulette_start_order: 1,
            ..config()
        };
        for seed in 0..100 {
            let r = trace(
                &scene,
                [0.0; 3],
                [0.0, 0.0, -1.0],
                cfg,
                &mut Random::new(seed),
            )
            .unwrap();
            assert_eq!(r.events, 100);
            assert!((4.0 * PI * r.total() - 100.0).abs() < 1e-10);
        }
    }
    #[test]
    fn attenuated_weight_roulette_preserves_expected_light() {
        let scene = FixedCollisions {
            count: 20,
            ssa: 0.5,
        };
        let cfg = PathConfig {
            roulette_start_order: 1,
            ..config()
        };
        let mut m = Moments::default();
        for seed in 0..200000 {
            m.push(
                4.0 * PI
                    * trace(
                        &scene,
                        [0.0; 3],
                        [0.0, 0.0, -1.0],
                        cfg,
                        &mut Random::new(seed),
                    )
                    .unwrap()
                    .total(),
            );
        }
        let expected = 1.0 - 0.5f64.powi(20);
        assert!((m.mean - expected).abs() < 6.0 * m.standard_error().unwrap());
    }
    #[test]
    fn safety_limit_refuses_a_truncated_image_and_invalid_optics() {
        let cfg = PathConfig {
            roulette_start_order: 1,
            event_safety_limit: 10,
            ..config()
        };
        let error = trace(
            &FixedCollisions {
                count: 11,
                ssa: 1.0,
            },
            [0.0; 3],
            [0.0, 0.0, -1.0],
            cfg,
            &mut Random::new(123),
        )
        .unwrap_err();
        assert!(error.contains("safety limit"));
        assert!(PhaseFunction::dual_hg(1.0, 0.0, 1.0).validate().is_err());
        assert!(PhaseFunction::rayleigh(f64::NAN).validate().is_err());
        assert!(Material::Lambertian { albedo: 1.01 }.validate().is_err());
        assert!(
            Material::CoxMunk {
                mean_square_slope: 0.0
            }
            .validate()
            .is_err()
        );
        assert!(
            trace(
                &FixedCollisions {
                    count: 2,
                    ssa: f64::NAN
                },
                [0.0; 3],
                [0.0, 0.0, -1.0],
                config(),
                &mut Random::new(12)
            )
            .is_err()
        );
    }
    #[test]
    fn gpu_path_sampling_and_slab_reference_validate() {
        let module = naga::front::wgsl::parse_str(include_str!("gpu/shaders/spectral_path.wgsl"))
            .expect("path shader parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("path shader validates");
    }
}
