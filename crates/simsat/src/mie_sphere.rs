//! Scalar Lorenz-Mie scattering by a homogeneous sphere in a real-index medium.
//!
//! Coefficients follow the Riccati-Bessel/log-derivative formulation of
//! Bohren & Huffman (1983), with downward recurrences for both D_n(m*x)
//! and psi_n(x). The latter avoids the growing complementary solution
//! contaminating the small high-order psi values. Optical efficiencies and
//! the normalized unpolarized phase are derived from the same coefficients.
//!
//! Intended as a spectral liquid-droplet building block, not an ice-habit
//! model or complete cloud closure. The caller specifies the relative index
//! and x=2*pi*n_medium*radius/vacuum_wavelength. Particle-size distribution
//! integration and multiple scattering remain separate operations.
use crate::material_indices::RefractiveIndex;
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MieError {
    InvalidSize,
    InvalidIndex,
    InvalidAngle,
    NoScattering,
    NumericalFailure,
}
impl std::fmt::Display for MieError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::InvalidSize => "sphere size parameter must be finite within 1..=2048",
            Self::InvalidIndex => "sphere relative index requires real 1..=2 and positive-imaginary absorption 0..=0.1",
            Self::InvalidAngle => "scattering cosine must be finite within -1..=1",
            Self::NoScattering => "normalized phase is undefined for an index-matched sphere",
            Self::NumericalFailure => "sphere recurrence or optical efficiency is nonphysical/nonfinite",
        })
    }
}
impl std::error::Error for MieError {}

#[derive(Debug, Clone, Copy, Default)]
struct C {
    re: f64,
    im: f64,
}
impl C {
    fn real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }
    fn add(self, b: Self) -> Self {
        Self {
            re: self.re + b.re,
            im: self.im + b.im,
        }
    }
    fn sub(self, b: Self) -> Self {
        Self {
            re: self.re - b.re,
            im: self.im - b.im,
        }
    }
    fn scale(self, b: f64) -> Self {
        Self {
            re: self.re * b,
            im: self.im * b,
        }
    }
    fn mul(self, b: Self) -> Self {
        Self {
            re: self.re * b.re - self.im * b.im,
            im: self.re * b.im + self.im * b.re,
        }
    }
    fn div(self, b: Self) -> Self {
        let norm = b.norm2();
        Self {
            re: (self.re * b.re + self.im * b.im) / norm,
            im: (self.im * b.re - self.re * b.im) / norm,
        }
    }
    fn norm2(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    fn real_conjugate_product(self, b: Self) -> f64 {
        self.re * b.re + self.im * b.im
    }
    fn finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MieEfficiencies {
    pub extinction: f64,
    pub scattering: f64,
    pub absorption: f64,
    pub asymmetry: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct MieCoefficient {
    pub electric_real: f64,
    pub electric_imaginary: f64,
    pub magnetic_real: f64,
    pub magnetic_imaginary: f64,
}
impl MieCoefficient {
    fn a(self) -> C {
        C {
            re: self.electric_real,
            im: self.electric_imaginary,
        }
    }
    fn b(self) -> C {
        C {
            re: self.magnetic_real,
            im: self.magnetic_imaginary,
        }
    }
}

pub struct MieSphere {
    size_parameter: f64,
    coefficients: Vec<MieCoefficient>,
    efficiencies: MieEfficiencies,
}
impl MieSphere {
    pub fn new(relative_index: RefractiveIndex, size_parameter: f64) -> Result<Self, MieError> {
        Self::calculate(relative_index, size_parameter, 128)
    }
    pub fn size_parameter(&self) -> f64 {
        self.size_parameter
    }
    pub fn efficiencies(&self) -> MieEfficiencies {
        self.efficiencies
    }
    /// Orders n=1 onward; shared with GPU angular evaluation/preparation.
    pub fn coefficients(&self) -> &[MieCoefficient] {
        &self.coefficients
    }

    fn calculate(index: RefractiveIndex, x: f64, padding: usize) -> Result<Self, MieError> {
        if !x.is_finite() || !(1.0..=2048.0).contains(&x) {
            return Err(MieError::InvalidSize);
        }
        if !index.real.is_finite()
            || !index.imaginary.is_finite()
            || !(1.0..=2.0).contains(&index.real)
            || !(0.0..=0.1).contains(&index.imaginary)
        {
            return Err(MieError::InvalidIndex);
        }
        if index.real == 1.0 && index.imaginary == 0.0 {
            return Ok(Self {
                size_parameter: x,
                coefficients: Vec::new(),
                efficiencies: MieEfficiencies {
                    extinction: 0.0,
                    scattering: 0.0,
                    absorption: 0.0,
                    asymmetry: 0.0,
                },
            });
        }
        let m = C {
            re: index.real,
            im: index.imaginary,
        };
        let z = m.scale(x);
        let stop = (x + 4.05 * x.cbrt() + 2.0).ceil() as usize;
        let start = stop.max(z.norm2().sqrt().ceil() as usize) + padding;
        let mut derivative = vec![C::default(); start + 1];
        for n in (1..=start).rev() {
            let nz = C::real(n as f64).div(z);
            derivative[n - 1] = nz.sub(C::real(1.0).div(derivative[n].add(nz)));
        }
        // Miller's downward recurrence, scaled before overflow. Normalize
        // using whichever exact low-order value is larger to avoid a sin(x)
        // node or a psi_1(x) node becoming the normalization denominator.
        let mut psi = vec![0.0f64; start + 2];
        psi[start] = 1.0;
        for n in (1..=start).rev() {
            psi[n - 1] = (2 * n + 1) as f64 / x * psi[n] - psi[n + 1];
            if psi[n - 1].abs() > 1e100 {
                for value in &mut psi[n - 1..] {
                    *value *= 1e-100;
                }
            }
        }
        let exact0 = x.sin();
        let exact1 = x.sin() / x - x.cos();
        let scale = if exact0.abs() >= exact1.abs() {
            exact0 / psi[0]
        } else {
            exact1 / psi[1]
        };
        for value in &mut psi[..=stop] {
            *value *= scale;
        }
        let mut chi_previous = x.cos();
        let mut chi = x.cos() / x + x.sin();
        let mut coefficients = Vec::with_capacity(stop);
        let mut scattering_sum = 0.0;
        let mut extinction_sum = 0.0;
        for n in 1..=stop {
            let nf = n as f64;
            let xi = C {
                re: psi[n],
                im: -chi,
            };
            let previous_xi = C {
                re: psi[n - 1],
                im: -chi_previous,
            };
            let electric = derivative[n].div(m).add(C::real(nf / x));
            let magnetic = m.mul(derivative[n]).add(C::real(nf / x));
            let a = electric
                .scale(psi[n])
                .sub(C::real(psi[n - 1]))
                .div(electric.mul(xi).sub(previous_xi));
            let b = magnetic
                .scale(psi[n])
                .sub(C::real(psi[n - 1]))
                .div(magnetic.mul(xi).sub(previous_xi));
            if !a.finite() || !b.finite() {
                return Err(MieError::NumericalFailure);
            }
            scattering_sum += (2.0 * nf + 1.0) * (a.norm2() + b.norm2());
            extinction_sum += (2.0 * nf + 1.0) * (a.re + b.re);
            coefficients.push(MieCoefficient {
                electric_real: a.re,
                electric_imaginary: a.im,
                magnetic_real: b.re,
                magnetic_imaginary: b.im,
            });
            let next = (2.0 * nf + 1.0) / x * chi - chi_previous;
            chi_previous = chi;
            chi = next;
        }
        let scattering = 2.0 * scattering_sum / (x * x);
        let extinction = 2.0 * extinction_sum / (x * x);
        let mut moment = 0.0;
        for (i, c) in coefficients.iter().enumerate() {
            let n = (i + 1) as f64;
            moment += (2.0 * n + 1.0) / (n * (n + 1.0)) * c.a().real_conjugate_product(c.b());
            if let Some(next) = coefficients.get(i + 1) {
                moment += n * (n + 2.0) / (n + 1.0)
                    * (c.a().real_conjugate_product(next.a())
                        + c.b().real_conjugate_product(next.b()));
            }
        }
        let asymmetry = 2.0 * moment / scattering_sum;
        if !extinction.is_finite()
            || !scattering.is_finite()
            || scattering <= 0.0
            || extinction < scattering - 1e-11 * scattering.max(1.0)
            || !asymmetry.is_finite()
            || !(-1.0..=1.0).contains(&asymmetry)
        {
            return Err(MieError::NumericalFailure);
        }
        Ok(Self {
            size_parameter: x,
            coefficients,
            efficiencies: MieEfficiencies {
                extinction,
                scattering,
                absorption: (extinction - scattering).max(0.0),
                asymmetry,
            },
        })
    }

    /// Unpolarized phase [sr^-1], normalized over the full sphere to unity.
    pub fn phase_sr1(&self, cosine: f64) -> Result<f64, MieError> {
        if !cosine.is_finite() || !(-1.0..=1.0).contains(&cosine) {
            return Err(MieError::InvalidAngle);
        }
        if self.efficiencies.scattering <= 0.0 {
            return Err(MieError::NoScattering);
        }
        let mu = cosine.abs();
        let mut pi_previous = 0.0;
        let mut pi_current = 1.0;
        let mut pi_difference = 1.0;
        let mut s1 = C::default();
        let mut s2 = C::default();
        for (i, c) in self.coefficients.iter().enumerate() {
            let n = (i + 1) as f64;
            let mut tau = n * mu * pi_current - (n + 1.0) * pi_previous;
            let mut next = (2.0 * n + 1.0) / n * mu * pi_current - (n + 1.0) / n * pi_previous;
            let mut difference = next - pi_current;
            // Algebraically identical difference recurrence; Sterbenz-exact
            // mu-1 and the slowly changing pi_n difference remain resolved near
            // the narrow large-droplet forward peak. WGSL uses the same form.
            if mu >= 0.5 {
                tau = (n * (mu - 1.0) - 1.0) * pi_current + (n + 1.0) * pi_difference;
                difference =
                    (1.0 + 1.0 / n) * pi_difference + (2.0 + 1.0 / n) * (mu - 1.0) * pi_current;
                next = pi_current + difference;
            }
            if mu == 1.0 {
                pi_current = 0.5 * n * (n + 1.0);
                tau = pi_current;
            }
            let pi_sign = if cosine < 0.0 && (i + 1) % 2 == 0 {
                -1.0
            } else {
                1.0
            };
            let tau_sign = if cosine < 0.0 { -pi_sign } else { 1.0 };
            let factor = (2.0 * n + 1.0) / (n * (n + 1.0));
            s1 = s1.add(
                c.a()
                    .scale(pi_sign * pi_current)
                    .add(c.b().scale(tau_sign * tau))
                    .scale(factor),
            );
            s2 = s2.add(
                c.a()
                    .scale(tau_sign * tau)
                    .add(c.b().scale(pi_sign * pi_current))
                    .scale(factor),
            );
            pi_previous = pi_current;
            pi_current = next;
            pi_difference = difference;
        }
        let phase = (s1.norm2() + s2.norm2())
            / (2.0 * PI * self.size_parameter * self.size_parameter * self.efficiencies.scattering);
        if phase.is_finite() && phase >= 0.0 {
            Ok(phase)
        } else {
            Err(MieError::NumericalFailure)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn direct_scipy_bessel_reference_covers_visible_particles_and_large_sizes() {
        let data: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/mie-scipy-reference.json"))
                .unwrap();
        let cases = data["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 68);
        for case in cases {
            let sphere = MieSphere::new(
                RefractiveIndex {
                    real: case["n_real"].as_f64().unwrap(),
                    imaginary: case["n_imag_positive"].as_f64().unwrap(),
                },
                case["x"].as_f64().unwrap(),
            )
            .unwrap();
            let q = sphere.efficiencies();
            for (actual, key) in [
                (q.extinction, "qext"),
                (q.scattering, "qsca"),
                (q.asymmetry, "g"),
            ] {
                let expected = case[key].as_f64().unwrap();
                assert!(
                    (actual - expected).abs() <= 1e-12 + 2e-9 * expected.abs(),
                    "case {} {key}: {actual} vs {expected}",
                    case["id"]
                );
            }
            for p in case["phase"].as_array().unwrap() {
                let expected = p["phase_sr1"].as_f64().unwrap();
                let mu = p["degrees"].as_f64().unwrap().to_radians().cos();
                let actual = sphere.phase_sr1(mu).unwrap();
                assert!(
                    (actual - expected).abs() <= 1e-10 + 1e-7 * expected.abs(),
                    "case {} phase {}: {actual} vs {expected}",
                    case["id"],
                    p["degrees"]
                );
            }
        }
    }
    #[test]
    fn sphere_matches_independent_bohren_huffman_absorbing_reference() {
        // Official libRadtran 2.0.6 BHMIE compiled unchanged with local Flang.
        // Its interface and returned efficiencies are f32; internal work is f64.
        let sphere = MieSphere::new(
            RefractiveIndex {
                real: 1.5,
                imaginary: 0.1,
            },
            2.0,
        )
        .unwrap();
        let q = sphere.efficiencies();
        for (actual, reference) in [
            (q.extinction, 1.941_478_371_620_178_2),
            (q.scattering, 1.286_167_979_240_417_5),
            (q.asymmetry, 0.657_082_974_910_736_1),
            (sphere.phase_sr1(1.0).unwrap(), 0.449_425_382_706_079_2),
            (
                sphere.phase_sr1(30.0f64.to_radians().cos()).unwrap(),
                0.313_366_716_543_789_6,
            ),
        ] {
            assert!(
                (actual / reference - 1.0).abs() < 2e-6,
                "{actual} vs {reference}"
            );
        }
        assert!(q.absorption > 0.0);
    }
    #[test]
    fn sphere_is_conservative_and_recurrence_converges_at_cloud_sizes() {
        for x in [
            1.0,
            2.0,
            PI,
            4.493409457909064,
            10.0,
            100.0,
            512.0,
            1024.0,
            2048.0,
        ] {
            for imaginary in [0.0, 1e-9, 1e-6, 0.1] {
                let m = RefractiveIndex {
                    real: 1.335,
                    imaginary,
                };
                let a = MieSphere::calculate(m, x, 128).unwrap();
                let b = MieSphere::calculate(m, x, 192).unwrap();
                let qa = a.efficiencies();
                let qb = b.efficiencies();
                if imaginary == 0.0 {
                    assert!((qa.extinction - qa.scattering).abs() < 1e-12);
                }
                for (aa, bb) in [
                    (qa.extinction, qb.extinction),
                    (qa.scattering, qb.scattering),
                    (qa.asymmetry, qb.asymmetry),
                ] {
                    assert!((aa - bb).abs() < 2e-11, "x={x}: {aa} vs {bb}");
                }
                for mu in [-1.0, -0.75, 0.0, 0.5, 0.95, 1.0] {
                    let aa = a.phase_sr1(mu).unwrap();
                    let bb = b.phase_sr1(mu).unwrap();
                    assert!(
                        (aa - bb).abs() < 2e-10 * aa.max(1e-6),
                        "x={x},mu={mu}: {aa} vs {bb}"
                    );
                }
            }
        }
    }
    #[test]
    fn angular_quadrature_recovers_scattering_normalization_and_asymmetry() {
        // Independent Gauss-Legendre integration, with enough nodes to resolve
        // the squared finite Mie angular series. Unlike uniform-angle sums,
        // it resolves the narrow forward peak at large droplet size.
        for x in [1.0, 2.0, 10.0, 100.0, 512.0] {
            let sphere = MieSphere::new(
                RefractiveIndex {
                    real: 1.335,
                    imaginary: 1e-7,
                },
                x,
            )
            .unwrap();
            let order = sphere.coefficients().len() + 2;
            let mut integral = 0.0;
            let mut moment = 0.0;
            for i in 1..=order {
                let mut z = (PI * (i as f64 - 0.25) / (order as f64 + 0.5)).cos();
                let derivative_at = |z: f64| {
                    let mut previous = 1.0;
                    let mut current = z;
                    for n in 2..=order {
                        let nf = n as f64;
                        let next = ((2.0 * nf - 1.0) * z * current - (nf - 1.0) * previous) / nf;
                        previous = current;
                        current = next;
                    }
                    (
                        current,
                        order as f64 * (z * current - previous) / (z * z - 1.0),
                    )
                };
                let mut converged = false;
                for _ in 0..32 {
                    let (pn, dpn) = derivative_at(z);
                    let correction = pn / dpn;
                    z -= correction;
                    if correction.abs() < 3e-15 {
                        converged = true;
                        break;
                    }
                }
                assert!(converged);
                let (_, dpn) = derivative_at(z);
                let weight = 2.0 / ((1.0 - z * z) * dpn * dpn);
                let contribution = 2.0 * PI * weight * sphere.phase_sr1(z).unwrap();
                integral += contribution;
                moment += z * contribution;
            }
            assert!((integral - 1.0).abs() < 2e-9, "x={x},integral={integral}");
            assert!(
                (moment - sphere.efficiencies().asymmetry).abs() < 2e-9,
                "x={x},moment={moment}"
            );
        }
    }

    #[test]
    fn mie_angular_wgsl_is_valid() {
        let module =
            naga::front::wgsl::parse_str(include_str!("gpu/shaders/mie_sphere.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }

    #[test]
    fn sphere_rejects_outside_contract_and_handles_index_matching() {
        let m = RefractiveIndex {
            real: 1.33,
            imaginary: 0.0,
        };
        for x in [0.0, 0.999, 2048.1, f64::NAN, f64::INFINITY] {
            assert!(MieSphere::new(m, x).is_err());
        }
        for index in [
            RefractiveIndex {
                real: 0.9,
                imaginary: 0.0,
            },
            RefractiveIndex {
                real: 1.3,
                imaginary: -1e-9,
            },
            RefractiveIndex {
                real: 1.3,
                imaginary: 0.1001,
            },
            RefractiveIndex {
                real: f64::NAN,
                imaginary: 0.0,
            },
        ] {
            assert!(MieSphere::new(index, 10.0).is_err());
        }
        let matched = MieSphere::new(
            RefractiveIndex {
                real: 1.0,
                imaginary: 0.0,
            },
            10.0,
        )
        .unwrap();
        assert_eq!(matched.efficiencies().extinction, 0.0);
        assert_eq!(matched.phase_sr1(0.0), Err(MieError::NoScattering));
        let sphere = MieSphere::new(m, 10.0).unwrap();
        assert_eq!(sphere.phase_sr1(1.001), Err(MieError::InvalidAngle));
    }
}
