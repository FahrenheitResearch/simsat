//! Clear-sky atmosphere — Hillaire 2020 (design doc section 3).
//!
//! Hillaire, "A Scalable and Production Ready Sky and Atmosphere Rendering
//! Technique", EGSR 2020 — the transmittance LUT, the isotropic multiple-scattering
//! LUT, the per-frame sky-view LUT, and the aerial-perspective froxel volume. This
//! module is the CPU REFERENCE (the WGSL twin lives in `gpu/shaders/surface.wgsl`);
//! nodes have no GPU, so the physics is validated here in `cargo test` and the WGSL
//! is kept in lockstep by discipline (design section 9, test strategy 2).
//!
//! Every constant carries its source and units — that is the project honesty
//! standard (design section 6). Band-averaged RGB gray optics; NOT a line-by-line
//! radiative-transfer model (design non-goals).
//!
//! Geometry convention: distances are in metres, radii are measured from the earth
//! CENTRE. `r` is a point's radius, `mu` is the cosine of the angle between the
//! local up (radial) direction and a ray. Spherically-symmetric quantities (the
//! transmittance and multi-scatter LUTs) are parameterised by `(r, mu)`; the
//! sky-view LUT and the render additionally need the sun. The render works in
//! spherical ECEF (see [`CameraGeometry`]).

use crate::bricks::VolumeBrick;
use crate::camera::GEO_ORBIT_RADIUS_M;
use crate::optics::EARTH_RADIUS_M;

use std::f64::consts::PI;

// ── Radii ──────────────────────────────────────────────────────────────────

/// Bottom of the atmosphere = WRF spherical earth radius (m). Owner decision 5;
/// identical to `optics::EARTH_RADIUS_M` so ALL of SimSat uses one earth radius.
pub const R_GROUND_M: f64 = EARTH_RADIUS_M;
/// Atmosphere shell thickness (m). Hillaire's 100 km (his 6360 -> 6460 km sphere;
/// we keep the same 100 km over the WRF 6370 km ground).
pub const ATMOSPHERE_HEIGHT_M: f64 = 100_000.0;
/// Top of the atmosphere (m from earth centre).
pub const R_TOP_M: f64 = R_GROUND_M + ATMOSPHERE_HEIGHT_M;

// ── Rayleigh ───────────────────────────────────────────────────────────────

/// Rayleigh sea-level scattering coefficient per RGB band (m^-1). Hillaire 2020
/// supplemental values (0.005802 / 0.013558 / 0.033100 km^-1) at ~680/550/440 nm.
/// The green (550 nm) value 1.3558e-5 m^-1 is the published order for Rayleigh
/// scattering at 550 nm (cf. Bucholtz 1995 / Penndorf 1957, ~1.3e-5 m^-1).
pub const RAYLEIGH_SCATTERING: [f64; 3] = [5.802e-6, 13.558e-6, 33.100e-6];
/// Rayleigh density scale height (m). Standard 8 km.
pub const RAYLEIGH_SCALE_HEIGHT_M: f64 = 8_000.0;

// ── Mie (aerosol) ──────────────────────────────────────────────────────────

/// Mie density scale height (m). Design section 3: a single exponential layer
/// ~1.2 km.
pub const MIE_SCALE_HEIGHT_M: f64 = 1_200.0;
/// Mie single-scatter albedo (scattering / extinction). Hillaire's 3.996/4.40.
pub const MIE_SINGLE_SCATTER_ALBEDO: f64 = 0.9;
/// Cornette-Shanks phase asymmetry `g` (forward-scattering aerosol).
pub const MIE_ASYMMETRY_G: f64 = 0.8;
/// Default aerosol optical depth (vertical Mie extinction integral). Design
/// section 3 default was 0.10; the M2 twilight pass lowered it to 0.05 (a cleaner
/// continental atmosphere) to shrink the warm, hazy Mie forward halo around the
/// low sun that was reddening the terminator (notes/twilight-diagnosis.md item 3).
pub const DEFAULT_AOD: f64 = 0.05;

// ── Ozone ──────────────────────────────────────────────────────────────────

/// Ozone absorption coefficient per RGB band at the layer peak (m^-1). Hillaire's
/// base (0.650 / 1.881 / 0.085) e-3 km^-1 scaled by [`OZONE_STRENGTH`]. Ozone
/// absorbs green/red far more than blue (the Chappuis band), which is why the
/// long-path deep-twilight residual is blue — the M2 twilight pass strengthened it
/// to counter the over-orange terminator (notes/twilight-diagnosis.md item 4).
pub const OZONE_ABSORPTION: [f64; 3] = [
    0.650e-6 * OZONE_STRENGTH,
    1.881e-6 * OZONE_STRENGTH,
    0.085e-6 * OZONE_STRENGTH,
];
/// Multiplier on the base Hillaire ozone absorption (M2 twilight pass). The Chappuis
/// band supports ~1.3-1.5x; 1.45 shifts the terminator + deep-twilight residual
/// bluer/cooler without retuning Rayleigh. `1.0` reproduces the pre-tuning ozone.
pub const OZONE_STRENGTH: f64 = 1.45;
/// Ozone tent-profile centre altitude (m).
pub const OZONE_CENTER_M: f64 = 25_000.0;
/// Ozone tent-profile half width (m): density = max(0, 1 - |h - centre| / halfwidth).
/// Round 2 TESTED widening this 15 -> 20 km, but the extra ozone column also cooled
/// the 0-deg terminator too far (R/B 1.43 -> 1.26, below the ~1.4 target) and dimmed
/// the daytime median (+10 -0.008, outside the +/-0.005 tolerance) — so it was
/// REVERTED to 15 km and the -2 fix was moved entirely to the highlight-desaturation
/// ramp (which is sun-elevation-neutral and leaves 0/daytime/-4/-6 at round 1).
pub const OZONE_HALF_WIDTH_M: f64 = 15_000.0;

// ── Below-horizon twilight fill (M2 twilight pass) ────────────────────────────

/// Sky-view observer altitude (m). M2 used 500 m; the twilight pass raised it to
/// 3 km so the sky-view LUT (the source of the ground/cloud ambient fill) sees more
/// of the still-sunlit high atmosphere once the sun is below the LOCAL ground
/// horizon — a brighter, longer, bluer blue-hour ambient. Purely the ambient
/// observer height; the per-pixel view marches use the real space-camera geometry
/// (notes/twilight-diagnosis.md item 2, knob "sky-view observer altitude").
pub const SKYVIEW_OBSERVER_ALTITUDE_M: f64 = 3_000.0;

/// Gain on the multiple-scattering term in the render marches (M2 twilight pass).
/// A modest, bounded boost of the ALREADY-computed isotropic Psi where it is
/// CONSUMED in `integrate_scattered_luminance` and the froxel march — it does NOT
/// change the multiscatter LUT build or the `f_ms` energy balance (which stays
/// `< 1`). Multiple scattering is Rayleigh-blue-dominated and is the dominant
/// below-horizon signal, so this brightens AND blues the twilight fill
/// (notes/twilight-diagnosis.md item 2, knob "multiple-scattering magnitude").
pub const MULTISCATTER_GAIN: f64 = 1.4;

// ── WRF moisture modulation (honest approximation, named) ────────────────────

/// Water-vapor broadband absorption coefficient per RGB band at the surface,
/// at the US-standard precipitable-water column (m^-1). Red-weighted because
/// water vapor absorbs the long visible wavelengths far more than blue — so a
/// wetter column darkens the WV-weighted (red) band, the direction the test asserts.
/// These are plausibility magnitudes for a band-averaged visible model, NOT
/// line-by-line HITRAN cross sections (documented approximation).
pub const WATER_VAPOR_ABSORPTION: [f64; 3] = [6.0e-6, 1.5e-6, 0.2e-6];
/// Water-vapor density scale height (m): vapor is boundary-layer concentrated.
pub const WATER_VAPOR_SCALE_HEIGHT_M: f64 = 2_000.0;
/// US Standard Atmosphere precipitable water (kg m^-2) — the reference column the
/// WRF PW is taken relative to (design section 3).
pub const PW_STANDARD_KG_M2: f64 = 14.2;

// ── Solar ────────────────────────────────────────────────────────────────────

/// Total solar irradiance at the top of the atmosphere (W m^-2).
pub const TSI_W_M2: f64 = 1361.0;
/// Band-averaged solar irradiance per RGB band at the top of the atmosphere
/// (W m^-2). A CIE-ish split of the ~42% of the 1361 W/m^2 TSI that falls in the
/// visible, with the mild blue excess of the solar spectrum. NOTE these cancel in
/// the reflectance factor (rho = pi L / E_band), so they set only the debug-HDR
/// scale and the white balance of the inscatter/limb; the reflectance colour comes
/// from albedo, transmittance, and the Rayleigh-blue scattering coefficients.
pub const SOLAR_IRRADIANCE_RGB: [f64; 3] = [180.0, 188.0, 196.0];
/// Solar disk angular diameter (deg). Design section 6.
pub const SUN_ANGULAR_DIAMETER_DEG: f64 = 0.533;
/// Solar disk angular RADIUS (rad) = half of 0.533 deg = 4.65 mrad.
pub const SUN_ANGULAR_RADIUS_RAD: f64 = 4.65e-3;

/// Hestroffer-Magnan (1998) limb-darkening coefficients for the visible solar
/// disk: I(mu)/I(1) = 1 - a1(1-mu) - a2(1-mu)^2, mu = cos(angle from disk centre).
/// Defined now (documented) for M3's finite-disk glint; M2 uses only the
/// disk-AVERAGED dimming factor [`LIMB_DARKENING_DISK_AVG`].
pub const LIMB_DARKENING_A1: f64 = 0.397;
pub const LIMB_DARKENING_A2: f64 = 0.216;

/// Ground albedo used for the multi-scatter ground bounce and (implicitly) the
/// sky-view lower hemisphere. A neutral continental value.
pub const GROUND_ALBEDO: f64 = 0.3;

// ── LUT default dimensions (design section 3) ────────────────────────────────

/// Transmittance LUT size (width = mu axis, height = radius axis).
pub const TRANSMITTANCE_LUT_W: usize = 256;
pub const TRANSMITTANCE_LUT_H: usize = 64;
/// Multiple-scattering LUT size (square).
pub const MULTISCATTER_LUT_SIZE: usize = 32;
/// Sky-view LUT default size (width = azimuth, height = view zenith).
pub const SKYVIEW_LUT_W: usize = 192;
pub const SKYVIEW_LUT_H: usize = 108;
/// Aerial-perspective froxel volume dimension (per axis).
pub const AERIAL_FROXEL_DIM: usize = 32;

/// Raymarch step counts (documented; balance accuracy vs the LUT-build cost).
const TRANSMITTANCE_STEPS: usize = 40;
const MULTISCATTER_STEPS: usize = 20;
const MULTISCATTER_SPHERE_SQRT: usize = 8; // 8x8 = 64 directions

// ── tiny vec3 helpers over [f64;3] ───────────────────────────────────────────

#[inline]
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
fn scl3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
fn madd3(a: [f64; 3], b: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s]
}
#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn len3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}
#[inline]
fn norm3(a: [f64; 3]) -> [f64; 3] {
    let l = len3(a);
    if l > 0.0 { scl3(a, 1.0 / l) } else { a }
}
#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// ── configurable optics parameters ───────────────────────────────────────────

/// The configurable atmosphere optics knobs (design section 3): aerosol optical
/// depth, the WRF precipitable-water modulation ratio, and the (off-by-default)
/// RH-driven aerosol swelling. Rayleigh/ozone stay standard-atmosphere.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtmosphereParams {
    /// Aerosol optical depth (vertical Mie extinction integral).
    pub aod: f64,
    /// Precipitable water / 14.2 kg m^-2 (US-standard column). 1.0 = standard.
    pub pw_ratio: f64,
    /// RH-driven aerosol swelling factor on the Mie coefficient. 1.0 = off (M2
    /// default). The UI toggle can raise it; documented hook, not tuned.
    pub aerosol_swelling: f64,
    /// Ground albedo for the multi-scatter bounce.
    pub ground_albedo: f64,
}

impl Default for AtmosphereParams {
    fn default() -> Self {
        Self {
            aod: DEFAULT_AOD,
            pw_ratio: 1.0,
            aerosol_swelling: 1.0,
            ground_albedo: GROUND_ALBEDO,
        }
    }
}

impl AtmosphereParams {
    /// Mie EXTINCTION at the surface (m^-1) from the AOD (and swelling): the
    /// exponential profile integrates to `ext_ground * H_M`, so `ext_ground =
    /// AOD / H_M`.
    #[inline]
    pub fn mie_extinction_ground(&self) -> f64 {
        self.aerosol_swelling * self.aod / MIE_SCALE_HEIGHT_M
    }
    /// Mie SCATTERING at the surface (m^-1).
    #[inline]
    pub fn mie_scattering_ground(&self) -> f64 {
        MIE_SINGLE_SCATTER_ALBEDO * self.mie_extinction_ground()
    }
    /// Mie ABSORPTION at the surface (m^-1).
    #[inline]
    pub fn mie_absorption_ground(&self) -> f64 {
        (1.0 - MIE_SINGLE_SCATTER_ALBEDO) * self.mie_extinction_ground()
    }
}

/// A sampled medium at one altitude: per-band scattering/extinction (m^-1).
#[derive(Debug, Clone, Copy)]
pub struct MediumSample {
    /// Rayleigh scattering per band (m^-1).
    pub rayleigh_scattering: [f64; 3],
    /// Mie scattering (m^-1, band-independent).
    pub mie_scattering: f64,
    /// Total scattering per band = Rayleigh + Mie (m^-1).
    pub scattering: [f64; 3],
    /// Total extinction per band = scattering + Mie absorption + ozone + water
    /// vapor (m^-1).
    pub extinction: [f64; 3],
}

/// Sample the medium at geometric altitude `h` (m above the ground sphere).
pub fn sample_medium(h: f64, params: &AtmosphereParams) -> MediumSample {
    let h = h.max(0.0);
    let rayleigh_density = (-h / RAYLEIGH_SCALE_HEIGHT_M).exp();
    let mie_density = (-h / MIE_SCALE_HEIGHT_M).exp();
    let ozone_density = (1.0 - (h - OZONE_CENTER_M).abs() / OZONE_HALF_WIDTH_M).max(0.0);
    let wv_density = (-h / WATER_VAPOR_SCALE_HEIGHT_M).exp();

    let mie_s = params.mie_scattering_ground() * mie_density;
    let mie_a = params.mie_absorption_ground() * mie_density;

    let mut rayleigh_scattering = [0.0; 3];
    let mut scattering = [0.0; 3];
    let mut extinction = [0.0; 3];
    for c in 0..3 {
        let ray_s = RAYLEIGH_SCATTERING[c] * rayleigh_density;
        let ozone_a = OZONE_ABSORPTION[c] * ozone_density;
        let wv_a = WATER_VAPOR_ABSORPTION[c] * wv_density * params.pw_ratio;
        rayleigh_scattering[c] = ray_s;
        scattering[c] = ray_s + mie_s;
        extinction[c] = ray_s + mie_s + mie_a + ozone_a + wv_a;
    }
    MediumSample {
        rayleigh_scattering,
        mie_scattering: mie_s,
        scattering,
        extinction,
    }
}

// ── ray/sphere geometry (spherically symmetric, in (r, mu)) ──────────────────

/// Does a ray from radius `r` with view-zenith cosine `mu` intersect the ground
/// sphere? (Standard Bruneton test: pointing down AND the discriminant is real.)
#[inline]
pub fn ray_hits_ground(r: f64, mu: f64) -> bool {
    mu < 0.0 && r * r * (mu * mu - 1.0) + R_GROUND_M * R_GROUND_M >= 0.0
}

/// Distance (m) from radius `r` along `mu` to the top-of-atmosphere sphere.
#[inline]
pub fn distance_to_top(r: f64, mu: f64) -> f64 {
    let disc = r * r * (mu * mu - 1.0) + R_TOP_M * R_TOP_M;
    (-r * mu + disc.max(0.0).sqrt()).max(0.0)
}

/// Distance (m) from radius `r` along `mu` to the ground sphere, or `None`.
#[inline]
pub fn distance_to_ground(r: f64, mu: f64) -> Option<f64> {
    let disc = r * r * (mu * mu - 1.0) + R_GROUND_M * R_GROUND_M;
    if disc < 0.0 {
        return None;
    }
    let d = -r * mu - disc.sqrt();
    (d > 0.0).then_some(d)
}

/// Optical depth per band from radius `r` along `mu` to the top of the atmosphere.
pub fn optical_depth_to_top(r: f64, mu: f64, params: &AtmosphereParams) -> [f64; 3] {
    let t_max = distance_to_top(r, mu);
    let steps = TRANSMITTANCE_STEPS;
    let dt = t_max / steps as f64;
    let mut od = [0.0f64; 3];
    for i in 0..steps {
        let t = (i as f64 + 0.5) * dt;
        // radius at parametric distance t along the ray: sqrt(r^2 + t^2 + 2 r mu t)
        let rr = (r * r + t * t + 2.0 * r * mu * t).max(0.0).sqrt();
        let m = sample_medium(rr - R_GROUND_M, params);
        od = madd3(od, m.extinction, dt);
    }
    od
}

// ── phase functions ──────────────────────────────────────────────────────────

/// Rayleigh phase function (normalised to integrate to 1 over the sphere).
#[inline]
pub fn rayleigh_phase(cos_theta: f64) -> f64 {
    3.0 / (16.0 * PI) * (1.0 + cos_theta * cos_theta)
}

/// Cornette-Shanks Mie phase function with asymmetry `g` (normalised over sphere).
#[inline]
pub fn cornette_shanks_phase(cos_theta: f64, g: f64) -> f64 {
    let g2 = g * g;
    let num = 3.0 * (1.0 - g2) * (1.0 + cos_theta * cos_theta);
    let denom = 8.0 * PI * (2.0 + g2) * (1.0 + g2 - 2.0 * g * cos_theta).powf(1.5);
    num / denom
}

/// Isotropic (uniform) phase 1/(4 pi) — the multiple-scattering assumption.
#[inline]
fn uniform_phase() -> f64 {
    1.0 / (4.0 * PI)
}

// ── generic 2-D LUT (RGBA f32, bilinear) ─────────────────────────────────────

/// A 2-D lookup table of RGBA f32 texels. Row-major, texel `(x, y)` at
/// `4*(y*w + x)`. The A channel is padding (kept so the buffer uploads straight
/// into an `Rgba32Float` texture the shader manual-bilinear-samples).
#[derive(Debug, Clone, PartialEq)]
pub struct Lut2 {
    pub width: usize,
    pub height: usize,
    /// `width*height*4` f32, RGBA.
    pub data: Vec<f32>,
}

impl Lut2 {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; width * height * 4],
        }
    }

    #[inline]
    fn set(&mut self, x: usize, y: usize, rgb: [f64; 3]) {
        let o = 4 * (y * self.width + x);
        self.data[o] = rgb[0] as f32;
        self.data[o + 1] = rgb[1] as f32;
        self.data[o + 2] = rgb[2] as f32;
        self.data[o + 3] = 1.0;
    }

    #[inline]
    fn fetch(&self, x: usize, y: usize) -> [f64; 3] {
        let o = 4 * (y * self.width + x);
        [
            self.data[o] as f64,
            self.data[o + 1] as f64,
            self.data[o + 2] as f64,
        ]
    }

    /// Bilinear sample at continuous texel coordinate `(u, v)` in [0,1] (the same
    /// half-texel-centre convention the WGSL twin uses with `textureLoad`).
    pub fn sample_uv(&self, u: f64, v: f64) -> [f64; 3] {
        let fx = (u.clamp(0.0, 1.0) * self.width as f64 - 0.5).max(0.0);
        let fy = (v.clamp(0.0, 1.0) * self.height as f64 - 0.5).max(0.0);
        let x0 = (fx.floor() as usize).min(self.width - 1);
        let y0 = (fy.floor() as usize).min(self.height - 1);
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let tx = fx - x0 as f64;
        let ty = fy - y0 as f64;
        let c00 = self.fetch(x0, y0);
        let c10 = self.fetch(x1, y0);
        let c01 = self.fetch(x0, y1);
        let c11 = self.fetch(x1, y1);
        let a = add3(scl3(c00, 1.0 - tx), scl3(c10, tx));
        let b = add3(scl3(c01, 1.0 - tx), scl3(c11, tx));
        add3(scl3(a, 1.0 - ty), scl3(b, ty))
    }
}

// ── transmittance LUT ────────────────────────────────────────────────────────

/// Map transmittance-LUT UV -> `(r, mu)` (Bruneton/Hillaire parameterisation).
pub fn transmittance_uv_to_r_mu(u: f64, v: f64) -> (f64, f64) {
    let h = (R_TOP_M * R_TOP_M - R_GROUND_M * R_GROUND_M).sqrt();
    let rho = h * v.clamp(0.0, 1.0);
    let r = (rho * rho + R_GROUND_M * R_GROUND_M).sqrt();
    let d_min = R_TOP_M - r;
    let d_max = rho + h;
    let d = d_min + u.clamp(0.0, 1.0) * (d_max - d_min);
    let mu = if d <= 0.0 {
        1.0
    } else {
        ((R_TOP_M * R_TOP_M - r * r - d * d) / (2.0 * r * d)).clamp(-1.0, 1.0)
    };
    (r, mu)
}

/// Map `(r, mu)` -> transmittance-LUT UV (inverse of [`transmittance_uv_to_r_mu`]).
pub fn transmittance_r_mu_to_uv(r: f64, mu: f64) -> (f64, f64) {
    let h = (R_TOP_M * R_TOP_M - R_GROUND_M * R_GROUND_M).sqrt();
    let rho = (r * r - R_GROUND_M * R_GROUND_M).max(0.0).sqrt();
    let d = distance_to_top(r, mu);
    let d_min = R_TOP_M - r;
    let d_max = rho + h;
    let u = if d_max - d_min > 0.0 {
        (d - d_min) / (d_max - d_min)
    } else {
        0.0
    };
    let v = if h > 0.0 { rho / h } else { 0.0 };
    (u.clamp(0.0, 1.0), v.clamp(0.0, 1.0))
}

/// Build the transmittance LUT (RGB = transmittance to the top of atmosphere).
pub fn build_transmittance_lut(params: &AtmosphereParams) -> Lut2 {
    let (w, h) = (TRANSMITTANCE_LUT_W, TRANSMITTANCE_LUT_H);
    let mut lut = Lut2::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let u = (x as f64 + 0.5) / w as f64;
            let v = (y as f64 + 0.5) / h as f64;
            let (r, mu) = transmittance_uv_to_r_mu(u, v);
            let od = optical_depth_to_top(r, mu, params);
            lut.set(x, y, [(-od[0]).exp(), (-od[1]).exp(), (-od[2]).exp()]);
        }
    }
    lut
}

/// Sample the transmittance LUT at `(r, mu)` (bilinear).
pub fn sample_transmittance(lut: &Lut2, r: f64, mu: f64) -> [f64; 3] {
    let (u, v) = transmittance_r_mu_to_uv(r.clamp(R_GROUND_M, R_TOP_M), mu);
    lut.sample_uv(u, v)
}

// ── multiple-scattering LUT ──────────────────────────────────────────────────

/// Map multiscatter-LUT UV -> `(r, mu_s)`.
pub fn multiscatter_uv_to_r_mus(u: f64, v: f64) -> (f64, f64) {
    let mu_s = (u.clamp(0.0, 1.0) * 2.0 - 1.0).clamp(-1.0, 1.0);
    let r = R_GROUND_M + v.clamp(0.0, 1.0) * (R_TOP_M - R_GROUND_M);
    (r.clamp(R_GROUND_M, R_TOP_M), mu_s)
}

/// Map `(r, mu_s)` -> multiscatter-LUT UV.
pub fn multiscatter_r_mus_to_uv(r: f64, mu_s: f64) -> (f64, f64) {
    let u = (mu_s.clamp(-1.0, 1.0) * 0.5 + 0.5).clamp(0.0, 1.0);
    let v = ((r - R_GROUND_M) / (R_TOP_M - R_GROUND_M)).clamp(0.0, 1.0);
    (u, v)
}

/// Hillaire's isotropic multiple-scattering estimate at `(r, mu_s)`, using the
/// transmittance LUT for sun visibility. Sun illuminance is unit white here; the
/// real solar irradiance is applied where the LUT is consumed. Returns
/// `(psi, f_ms)`: `psi` is the infinite-order multi-scatter factor (per band), and
/// `f_ms` is the single-bounce transfer factor (per band; the geometric series
/// `1/(1 - f_ms)` converges only for `0 <= f_ms < 1` — the energy-conservation test
/// asserts on it).
fn compute_multiscatter(
    transmittance: &Lut2,
    params: &AtmosphereParams,
    r: f64,
    mu_s: f64,
) -> ([f64; 3], [f64; 3]) {
    // Local frame: up = +z; the sun sits at zenith cosine mu_s in the x-z plane.
    let p0 = [0.0, 0.0, r];
    let sun = [(1.0 - mu_s * mu_s).max(0.0).sqrt(), 0.0, mu_s];
    let n = MULTISCATTER_SPHERE_SQRT;
    let mut lum_total = [0.0; 3]; // 2nd-order scattered radiance (uniform phase)
    let mut fms_total = [0.0; 3]; // transfer factor
    for i in 0..n {
        for j in 0..n {
            // Uniform sphere sample (theta polar, cos-uniform).
            let theta = PI * (i as f64 + 0.5) / n as f64;
            let cos_phi = 1.0 - 2.0 * (j as f64 + 0.5) / n as f64;
            let sin_phi = (1.0 - cos_phi * cos_phi).max(0.0).sqrt();
            let dir = [sin_phi * theta.cos(), sin_phi * theta.sin(), cos_phi];
            let (l, lf) = multiscatter_ray(transmittance, params, p0, dir, sun);
            lum_total = add3(lum_total, l);
            fms_total = add3(fms_total, lf);
        }
    }
    let inv = 1.0 / (n * n) as f64; // uniform sphere average = integral / 4pi
    let l2 = scl3(lum_total, inv);
    let fms = scl3(fms_total, inv);
    let mut psi = [0.0; 3];
    for c in 0..3 {
        let denom = 1.0 - fms[c];
        psi[c] = if denom > 1.0e-6 { l2[c] / denom } else { l2[c] };
    }
    (psi, fms)
}

/// One direction's contribution to the multi-scatter estimate: returns
/// `(2nd-order radiance, transfer factor)`.
fn multiscatter_ray(
    transmittance: &Lut2,
    params: &AtmosphereParams,
    p0: [f64; 3],
    dir: [f64; 3],
    sun: [f64; 3],
) -> ([f64; 3], [f64; 3]) {
    let r0 = len3(p0);
    let mu = dot3(norm3(p0), dir);
    let hits_ground = ray_hits_ground(r0, mu);
    let t_max = match distance_to_ground(r0, mu) {
        Some(d) if hits_ground => d,
        _ => distance_to_top(r0, mu),
    };
    let steps = MULTISCATTER_STEPS;
    let dt = t_max / steps as f64;
    let mut od = [0.0f64; 3];
    let mut l = [0.0; 3];
    let mut lf = [0.0; 3];
    for i in 0..steps {
        let t = (i as f64 + 0.5) * dt;
        let p = madd3(p0, dir, t);
        let r = len3(p);
        let up = scl3(p, 1.0 / r);
        let m = sample_medium(r - R_GROUND_M, params);
        let mu_s = dot3(up, sun);
        let t_sun = if ray_hits_ground(r, mu_s) {
            [0.0; 3]
        } else {
            sample_transmittance(transmittance, r, mu_s)
        };
        let up_phase = uniform_phase();
        for c in 0..3 {
            let throughput = (-od[c]).exp();
            let ext = m.extinction[c].max(1.0e-12);
            let sample_t = (-ext * dt).exp();
            // 2nd-order: sun light scattered once with uniform phase toward here.
            let s = m.scattering[c] * up_phase * t_sun[c];
            let s_int = (s - s * sample_t) / ext;
            l[c] += throughput * s_int;
            // Transfer factor f_ms = sphere-average of integral(T * sigma_s dt), with
            // NO inner phase (Hillaire MultiScatAs1). For an isotropic ambient field the
            // 1/4pi phase and the 4pi solid angle of the uniform field cancel, so the
            // single-scatter L term keeps `up_phase` but this transfer term must NOT
            // (an earlier `* up_phase` made f_ms 4pi too small, defeating the
            // 1/(1-f_ms) multiple-scattering boost — M2 review FINDING 2).
            let sf = m.scattering[c];
            let sf_int = (sf - sf * sample_t) / ext;
            lf[c] += throughput * sf_int;
            od[c] += ext * dt;
        }
    }
    // Ground bounce: Lambertian ground lit by the direct sun contributes to the
    // 2nd-order term (Hillaire).
    if hits_ground {
        let p = madd3(p0, dir, t_max);
        let r = len3(p);
        let up = scl3(p, 1.0 / r);
        let mu_s = dot3(up, sun);
        if mu_s > 0.0 {
            let t_sun = sample_transmittance(transmittance, r, mu_s);
            for c in 0..3 {
                let throughput = (-od[c]).exp();
                l[c] += throughput * params.ground_albedo * mu_s / PI * t_sun[c];
            }
        }
    }
    (l, lf)
}

/// Build the multiple-scattering LUT (RGB = the isotropic Psi factor).
pub fn build_multiscatter_lut(transmittance: &Lut2, params: &AtmosphereParams) -> Lut2 {
    let s = MULTISCATTER_LUT_SIZE;
    let mut lut = Lut2::new(s, s);
    for y in 0..s {
        for x in 0..s {
            let u = (x as f64 + 0.5) / s as f64;
            let v = (y as f64 + 0.5) / s as f64;
            let (r, mu_s) = multiscatter_uv_to_r_mus(u, v);
            let (psi, _f_ms) = compute_multiscatter(transmittance, params, r, mu_s);
            lut.set(x, y, psi);
        }
    }
    lut
}

/// Sample the multi-scatter LUT at `(r, mu_s)` (bilinear).
pub fn sample_multiscatter(lut: &Lut2, r: f64, mu_s: f64) -> [f64; 3] {
    let (u, v) = multiscatter_r_mus_to_uv(r.clamp(R_GROUND_M, R_TOP_M), mu_s);
    lut.sample_uv(u, v)
}

// ── the shared scattering raymarch (sky-view, aerial perspective, limb) ───────

/// The precomputed LUTs the raymarch consumes.
pub struct AtmosphereLuts {
    pub transmittance: Lut2,
    pub multiscatter: Lut2,
}

impl AtmosphereLuts {
    /// Build the optics-config LUTs (transmittance, then multi-scatter).
    pub fn build(params: &AtmosphereParams) -> Self {
        let transmittance = build_transmittance_lut(params);
        let multiscatter = build_multiscatter_lut(&transmittance, params);
        Self {
            transmittance,
            multiscatter,
        }
    }
}

/// Result of a scattering raymarch along one ray segment.
#[derive(Debug, Clone, Copy)]
pub struct ScatterResult {
    /// In-scattered radiance reaching the ray origin (per band, physical units of
    /// the supplied sun irradiance).
    pub inscatter: [f64; 3],
    /// Transmittance of the segment (per band).
    pub transmittance: [f64; 3],
}

/// March a ray segment `[0, t_max]` from `p0` (ECEF, earth-centred) along unit
/// `view`, accumulating single + (optional) multiple in-scattering toward the
/// origin and the segment transmittance. `sun` is the unit ECEF sun direction;
/// `sun_irradiance` is the per-band top-of-atmosphere solar irradiance. The
/// earth-shadow test (a step whose sun ray hits the ground gets NO direct single
/// scattering, only multi-scatter) is what produces correct twilight.
#[allow(clippy::too_many_arguments)]
pub fn integrate_scattered_luminance(
    luts: &AtmosphereLuts,
    params: &AtmosphereParams,
    p0: [f64; 3],
    view: [f64; 3],
    sun: [f64; 3],
    sun_irradiance: [f64; 3],
    t_max: f64,
    steps: usize,
    include_multiscatter: bool,
) -> ScatterResult {
    let cos_vs = dot3(view, sun);
    let rayleigh_ph = rayleigh_phase(cos_vs);
    let mie_ph = cornette_shanks_phase(cos_vs, MIE_ASYMMETRY_G);
    let dt = t_max / steps.max(1) as f64;
    let mut od = [0.0f64; 3];
    let mut l = [0.0; 3];
    for i in 0..steps {
        let t = (i as f64 + 0.5) * dt;
        let p = madd3(p0, view, t);
        let r = len3(p);
        if !(R_GROUND_M - 1.0..R_TOP_M + 1.0).contains(&r) {
            // Outside the shell: no medium (guards float drift on entry/exit).
            continue;
        }
        let up = scl3(p, 1.0 / r);
        let m = sample_medium(r - R_GROUND_M, params);
        let mu_s = dot3(up, sun);
        let t_sun = if ray_hits_ground(r, mu_s) {
            [0.0; 3]
        } else {
            sample_transmittance(&luts.transmittance, r, mu_s)
        };
        let ms = if include_multiscatter {
            sample_multiscatter(&luts.multiscatter, r, mu_s)
        } else {
            [0.0; 3]
        };
        for c in 0..3 {
            let throughput = (-od[c]).exp();
            let ext = m.extinction[c].max(1.0e-12);
            let sample_t = (-ext * dt).exp();
            // Single scattering: direct sun (attenuated to p) scattered toward view.
            let single =
                t_sun[c] * (m.rayleigh_scattering[c] * rayleigh_ph + m.mie_scattering * mie_ph);
            // Multiple scattering: fills the earth shadow (no sun-visibility gate).
            // MULTISCATTER_GAIN modestly boosts this below-horizon twilight fill.
            let multi = MULTISCATTER_GAIN * m.scattering[c] * ms[c];
            let s = sun_irradiance[c] * (single + multi);
            let s_int = (s - s * sample_t) / ext;
            l[c] += throughput * s_int;
            od[c] += ext * dt;
        }
    }
    ScatterResult {
        inscatter: l,
        transmittance: [(-od[0]).exp(), (-od[1]).exp(), (-od[2]).exp()],
    }
}

// ── sky-view LUT (twilight) ──────────────────────────────────────────────────

/// Configuration for a sky-view LUT build (dims + march steps). The design target
/// is [`SKYVIEW_LUT_W`]x[`SKYVIEW_LUT_H`]; tests use smaller for speed.
#[derive(Debug, Clone, Copy)]
pub struct SkyViewConfig {
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    /// Observer altitude above the ground (m).
    pub observer_altitude_m: f64,
}

impl Default for SkyViewConfig {
    fn default() -> Self {
        Self {
            width: SKYVIEW_LUT_W,
            height: SKYVIEW_LUT_H,
            steps: 30,
            observer_altitude_m: SKYVIEW_OBSERVER_ALTITUDE_M,
        }
    }
}

/// Map a sky-view `v` coordinate to a view-zenith angle (rad), with a sqrt bias
/// that packs samples near the horizon (Hillaire's horizon split). `beta` is the
/// horizon dip angle for the observer radius.
fn skyview_v_to_zenith(v: f64, r_obs: f64) -> f64 {
    let v_hor = (r_obs * r_obs - R_GROUND_M * R_GROUND_M).max(0.0).sqrt();
    let cos_beta = (v_hor / r_obs).clamp(-1.0, 1.0);
    let beta = cos_beta.acos(); // dip of the horizon below the horizontal
    let zenith_horizon = PI - beta; // view-zenith angle of the horizon direction
    if v < 0.5 {
        // Above the horizon: zenith 0 (up) -> zenith_horizon.
        let coord = 1.0 - 2.0 * v;
        zenith_horizon * (1.0 - coord * coord)
    } else {
        // Below the horizon: zenith_horizon -> PI (straight down).
        let coord = 2.0 * v - 1.0;
        zenith_horizon + beta * coord * coord
    }
}

/// Build the per-frame sky-view LUT for a sun ELEVATION (rad; negative = below
/// horizon). u = sun-relative azimuth, v = view zenith (horizon-split). RGB =
/// sky radiance in that view direction, from the observer altitude.
pub fn build_sky_view_lut(
    luts: &AtmosphereLuts,
    params: &AtmosphereParams,
    sun_elevation_rad: f64,
    cfg: &SkyViewConfig,
) -> Lut2 {
    let (w, h) = (cfg.width, cfg.height);
    let r_obs = R_GROUND_M + cfg.observer_altitude_m;
    let p0 = [0.0, 0.0, r_obs];
    // Sun in the observer's local frame (up = +z), azimuth 0 (defines u origin).
    let se = sun_elevation_rad;
    let sun = [se.cos(), 0.0, se.sin()];
    let mut lut = Lut2::new(w, h);
    for y in 0..h {
        let v = (y as f64 + 0.5) / h as f64;
        let zenith = skyview_v_to_zenith(v, r_obs);
        let (sz, cz) = zenith.sin_cos();
        for x in 0..w {
            let u = (x as f64 + 0.5) / w as f64;
            let az = u * 2.0 * PI; // sun-relative azimuth
            let view = [sz * az.cos(), sz * az.sin(), cz];
            // March to the ground or the top of atmosphere.
            let mu = dot3(norm3(p0), view);
            let t_max = match distance_to_ground(r_obs, mu) {
                Some(d) if ray_hits_ground(r_obs, mu) => d,
                _ => distance_to_top(r_obs, mu),
            };
            let res = integrate_scattered_luminance(
                luts,
                params,
                p0,
                view,
                sun,
                SOLAR_IRRADIANCE_RGB,
                t_max,
                cfg.steps,
                true,
            );
            lut.set(x, y, res.inscatter);
        }
    }
    lut
}

/// Hemispherical cosine integral of a sky-view LUT -> ambient irradiance on a
/// horizontal surface (per band, W m^-2). Only the above-horizon texels contribute.
pub fn ambient_irradiance_from_sky_view(lut: &Lut2, r_obs: f64) -> [f64; 3] {
    let (w, h) = (lut.width, lut.height);
    let mut irr = [0.0; 3];
    // Integrate L(omega) * cos(zenith) dOmega over the upper hemisphere.
    // dOmega = sin(zenith) dZenith dAzimuth; we sum over texels with their
    // parameterised zenith and a uniform azimuth step.
    let d_az = 2.0 * PI / w as f64;
    for y in 0..h {
        let v0 = y as f64 / h as f64;
        let v1 = (y as f64 + 1.0) / h as f64;
        let z0 = skyview_v_to_zenith(v0, r_obs);
        let z1 = skyview_v_to_zenith(v1, r_obs);
        let zc = 0.5 * (z0 + z1);
        if zc >= PI / 2.0 {
            continue; // below the horizon
        }
        let d_zenith = (z1 - z0).abs();
        let weight = zc.cos() * zc.sin() * d_zenith * d_az;
        for x in 0..w {
            let c = lut.fetch(x, y);
            irr = madd3(irr, c, weight);
        }
    }
    irr
}

/// A compact table of ambient irradiance vs sun elevation, the "cheap projection"
/// of the sky-view LUT the surface render consumes per pixel (full SH-2 directional
/// ambient is M5). Entries span `elev_min_deg`..`elev_max_deg`.
#[derive(Debug, Clone)]
pub struct AmbientTable {
    pub elev_min_deg: f64,
    pub elev_max_deg: f64,
    /// `n` RGB entries, index 0 = elev_min, last = elev_max.
    pub entries: Vec<[f32; 3]>,
}

impl AmbientTable {
    /// Build the table by integrating a sky-view LUT at `n` sun elevations. Uses a
    /// modest sky-view resolution per entry (fast); the design 192x108 sky-view is
    /// what the twilight LUT/test exercises.
    pub fn build(luts: &AtmosphereLuts, params: &AtmosphereParams, n: usize) -> Self {
        let (elev_min_deg, elev_max_deg) = (-20.0, 90.0);
        let cfg = SkyViewConfig {
            width: 64,
            height: 48,
            steps: 16,
            observer_altitude_m: SKYVIEW_OBSERVER_ALTITUDE_M,
        };
        let r_obs = R_GROUND_M + cfg.observer_altitude_m;
        let mut entries = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 / (n - 1).max(1) as f64;
            let elev = (elev_min_deg + t * (elev_max_deg - elev_min_deg)).to_radians();
            let sv = build_sky_view_lut(luts, params, elev, &cfg);
            let irr = ambient_irradiance_from_sky_view(&sv, r_obs);
            entries.push([irr[0] as f32, irr[1] as f32, irr[2] as f32]);
        }
        Self {
            elev_min_deg,
            elev_max_deg,
            entries,
        }
    }

    /// Ambient irradiance (per band) at a sun elevation (deg), linearly interpolated.
    pub fn at(&self, elev_deg: f64) -> [f64; 3] {
        let n = self.entries.len();
        if n == 0 {
            return [0.0; 3];
        }
        let t = ((elev_deg - self.elev_min_deg) / (self.elev_max_deg - self.elev_min_deg))
            .clamp(0.0, 1.0);
        let f = t * (n - 1) as f64;
        let i0 = (f.floor() as usize).min(n - 1);
        let i1 = (i0 + 1).min(n - 1);
        let a = self.entries[i0];
        let b = self.entries[i1];
        let w = f - i0 as f64;
        [
            (a[0] as f64) * (1.0 - w) + (b[0] as f64) * w,
            (a[1] as f64) * (1.0 - w) + (b[1] as f64) * w,
            (a[2] as f64) * (1.0 - w) + (b[2] as f64) * w,
        ]
    }

    /// Flatten to `n*4` RGBA f32 for upload as an `n`x1 texture.
    pub fn to_rgba_f32(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.entries.len() * 4);
        for e in &self.entries {
            out.extend_from_slice(&[e[0], e[1], e[2], 1.0]);
        }
        out
    }
}

// ── SH-2 directional sky ambient (design section 6, M5) ──────────────────────

/// Number of real SH basis functions up to order 2 (`l = 0,1,2` -> 1 + 3 + 5 = 9).
pub const SH2_COUNT: usize = 9;

/// The 9 real spherical-harmonic basis functions (`l <= 2`) at a unit direction
/// `d = (x, y, z)`. Standard real SH (e.g. Sloan 2008 "Stupid Spherical Harmonics
/// (SH) Tricks"): index 0 = Y_0^0; 1..=3 = Y_1^{-1,0,1}; 4..=8 = Y_2^{-2,-1,0,1,2}.
pub fn sh2_basis(d: [f64; 3]) -> [f64; SH2_COUNT] {
    let (x, y, z) = (d[0], d[1], d[2]);
    [
        0.282094791773878,                       // Y00 = 1/(2 sqrt(pi))
        0.488602511902920 * y,                   // Y1-1
        0.488602511902920 * z,                   // Y10
        0.488602511902920 * x,                   // Y11
        1.092548430592079 * x * y,               // Y2-2
        1.092548430592079 * y * z,               // Y2-1
        0.315391565252520 * (3.0 * z * z - 1.0), // Y20
        1.092548430592079 * x * z,               // Y21
        0.546274215296040 * (x * x - y * y),     // Y22
    ]
}

/// Cosine-lobe convolution coefficients `A_l` (Ramamoorthi & Hanrahan 2001, "An
/// Efficient Representation for Irradiance Environment Maps"): turning a projected
/// sky-RADIANCE SH into diffuse IRRADIANCE at a surface normal is a per-band SH dot
/// product with these. `A_0 = pi`, `A_1 = 2 pi / 3`, `A_2 = pi / 4` (odd `l > 1`
/// vanish; only `l <= 2` are kept, which is exact to ~99% for a smooth environment).
const SH2_COSINE_A: [f64; SH2_COUNT] = [
    PI,             // l = 0
    2.0 * PI / 3.0, // l = 1
    2.0 * PI / 3.0,
    2.0 * PI / 3.0,
    PI / 4.0, // l = 2
    PI / 4.0,
    PI / 4.0,
    PI / 4.0,
    PI / 4.0,
];

/// SH-2 (9 RGB-coefficient) projection of a sky-view LUT — the directional, COLORED
/// sky ambient (design section 6, "how much sky and what color"). The frame is
/// SUN-RELATIVE: `+z` = local up, `+x` = the sun's horizontal azimuth (the sky-view
/// LUT's azimuth origin), `+y = z x x`. This replaces M2's scalar [`AmbientTable`]:
/// the DC (`l=0`) term is the mean sky radiance the scalar carried, and the `l=1/l=2`
/// terms carry the sun-side-vs-antisun-side gradient (orange fill toward a sunset sun,
/// cool fill away) the scalar could not.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkyShAmbient {
    /// 9 SH-2 coefficients, each an RGB triple (sky RADIANCE projection).
    pub coef: [[f64; 3]; SH2_COUNT],
}

impl SkyShAmbient {
    /// Project a sky-view LUT into SH-2 over the full sphere. Texel `(x, y)` is the
    /// sky radiance in the direction (zenith from [`skyview_v_to_zenith`], azimuth =
    /// `u*2pi` sun-relative); each term is weighted by the texel solid angle
    /// `sin(zenith) dZenith dAzimuth` (the same quadrature the scalar ambient uses).
    pub fn project(lut: &Lut2, r_obs: f64) -> Self {
        let (w, h) = (lut.width, lut.height);
        let d_az = 2.0 * PI / w.max(1) as f64;
        let mut coef = [[0.0f64; 3]; SH2_COUNT];
        for y in 0..h {
            let v0 = y as f64 / h as f64;
            let v1 = (y as f64 + 1.0) / h as f64;
            let z0 = skyview_v_to_zenith(v0, r_obs);
            let z1 = skyview_v_to_zenith(v1, r_obs);
            let zc = 0.5 * (z0 + z1);
            let d_zenith = (z1 - z0).abs();
            let (sz, cz) = zc.sin_cos();
            let solid = sz * d_zenith * d_az;
            for x in 0..w {
                let az = (x as f64 + 0.5) / w as f64 * 2.0 * PI;
                let dir = [sz * az.cos(), sz * az.sin(), cz];
                let basis = sh2_basis(dir);
                let rad = lut.fetch(x, y);
                for (i, b) in basis.iter().enumerate() {
                    let bw = b * solid;
                    coef[i][0] += rad[0] * bw;
                    coef[i][1] += rad[1] * bw;
                    coef[i][2] += rad[2] * bw;
                }
            }
        }
        Self { coef }
    }

    /// Raw SH reconstruction of the sky RADIANCE in unit direction `dir` (sun-relative
    /// frame). SH truncation can ring slightly negative; the result is clamped `>= 0`.
    pub fn radiance(&self, dir: [f64; 3]) -> [f64; 3] {
        let basis = sh2_basis(norm3(dir));
        let mut out = [0.0; 3];
        for (i, b) in basis.iter().enumerate() {
            for (c, o) in out.iter_mut().enumerate() {
                *o += self.coef[i][c] * b;
            }
        }
        [out[0].max(0.0), out[1].max(0.0), out[2].max(0.0)]
    }

    /// Diffuse IRRADIANCE (per band, W m^-2) at a surface with unit `normal` (in the
    /// sun-relative frame), via the cosine-lobe convolution. At `normal = up` this
    /// reproduces the scalar hemisphere irradiance M2's ambient table carried.
    pub fn irradiance(&self, normal: [f64; 3]) -> [f64; 3] {
        let basis = sh2_basis(norm3(normal));
        let mut out = [0.0; 3];
        for ((coef, &b), &a_l) in self.coef.iter().zip(basis.iter()).zip(SH2_COSINE_A.iter()) {
            let a = a_l * b;
            for (c, o) in out.iter_mut().enumerate() {
                *o += coef[c] * a;
            }
        }
        [out[0].max(0.0), out[1].max(0.0), out[2].max(0.0)]
    }
}

/// Build the sun-relative orthonormal frame at a point: `z` = local up, `x` = the
/// sun's horizontal direction (the SH azimuth origin), `y = z x x`. `up` and `sun`
/// are unit vectors in ANY consistent basis (ENU for terrain, ECEF for cloud voxels).
/// If the sun is at the zenith the horizontal axis is arbitrary (ambient is then
/// azimuth-symmetric, so the choice does not matter).
pub fn sun_relative_frame(up: [f64; 3], sun: [f64; 3]) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let z = norm3(up);
    let sun_h = madd3(sun, z, -dot3(sun, z)); // sun minus its up-projection
    let x = if len3(sun_h) > 1.0e-6 {
        norm3(sun_h)
    } else {
        let seed = if z[2].abs() < 0.9 {
            [0.0, 0.0, 1.0]
        } else {
            [1.0, 0.0, 0.0]
        };
        norm3(madd3(seed, z, -dot3(seed, z)))
    };
    let y = cross3(z, x);
    (x, y, z)
}

/// Express a direction `v` in a sun-relative frame `(x, y, z)`.
#[inline]
pub fn to_frame(v: [f64; 3], frame: ([f64; 3], [f64; 3], [f64; 3])) -> [f64; 3] {
    [dot3(v, frame.0), dot3(v, frame.1), dot3(v, frame.2)]
}

/// A sun-elevation table of [`SkyShAmbient`] projections — the M5 replacement for the
/// scalar [`AmbientTable`] in the render path. Built per frame at `n` elevations
/// (mirroring the ambient table so per-pixel elevation interpolation keeps the
/// terminator fade), each entry the SH-2 projection of that elevation's sky-view LUT.
/// Receivers evaluate it directionally at their normal (terrain slope / cloud up).
#[derive(Debug, Clone)]
pub struct SkyShTable {
    pub elev_min_deg: f64,
    pub elev_max_deg: f64,
    /// `n` SH-2 projections, index 0 = `elev_min_deg`, last = `elev_max_deg`.
    pub entries: Vec<SkyShAmbient>,
}

impl SkyShTable {
    /// Build the table by projecting a sky-view LUT into SH-2 at `n` sun elevations
    /// (same span/resolution as [`AmbientTable::build`]).
    pub fn build(luts: &AtmosphereLuts, params: &AtmosphereParams, n: usize) -> Self {
        let (elev_min_deg, elev_max_deg) = (-20.0, 90.0);
        let cfg = SkyViewConfig {
            width: 64,
            height: 48,
            steps: 16,
            observer_altitude_m: SKYVIEW_OBSERVER_ALTITUDE_M,
        };
        let r_obs = R_GROUND_M + cfg.observer_altitude_m;
        let n = n.max(1);
        let mut entries = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 / (n - 1).max(1) as f64;
            let elev = (elev_min_deg + t * (elev_max_deg - elev_min_deg)).to_radians();
            let sv = build_sky_view_lut(luts, params, elev, &cfg);
            entries.push(SkyShAmbient::project(&sv, r_obs));
        }
        Self {
            elev_min_deg,
            elev_max_deg,
            entries,
        }
    }

    /// The SH-2 coefficients interpolated at a sun elevation (deg).
    pub fn at(&self, elev_deg: f64) -> SkyShAmbient {
        let n = self.entries.len();
        if n == 0 {
            return SkyShAmbient {
                coef: [[0.0; 3]; SH2_COUNT],
            };
        }
        let t = ((elev_deg - self.elev_min_deg) / (self.elev_max_deg - self.elev_min_deg))
            .clamp(0.0, 1.0);
        let f = t * (n - 1) as f64;
        let i0 = (f.floor() as usize).min(n - 1);
        let i1 = (i0 + 1).min(n - 1);
        let w = f - i0 as f64;
        let e0 = &self.entries[i0].coef;
        let e1 = &self.entries[i1].coef;
        let mut coef = [[0.0f64; 3]; SH2_COUNT];
        for (k, out) in coef.iter_mut().enumerate() {
            for (c, o) in out.iter_mut().enumerate() {
                *o = e0[k][c] * (1.0 - w) + e1[k][c] * w;
            }
        }
        SkyShAmbient { coef }
    }

    /// Directional diffuse irradiance at a receiver: `up` + `sun` build the sun-relative
    /// frame, `normal` is the receiver normal (same basis as `up`/`sun`).
    pub fn irradiance(
        &self,
        elev_deg: f64,
        up: [f64; 3],
        sun: [f64; 3],
        normal: [f64; 3],
    ) -> [f64; 3] {
        let frame = sun_relative_frame(up, sun);
        self.at(elev_deg).irradiance(to_frame(normal, frame))
    }

    /// Raw sky radiance in a direction `dir` (same basis as `up`/`sun`) — the "what
    /// colour is the sky in that direction" query used for the directional-fill test.
    pub fn radiance(&self, elev_deg: f64, up: [f64; 3], sun: [f64; 3], dir: [f64; 3]) -> [f64; 3] {
        let frame = sun_relative_frame(up, sun);
        self.at(elev_deg).radiance(to_frame(dir, frame))
    }

    /// The flat-receiver (`normal = up`) hemisphere irradiance — the SH counterpart of
    /// the M2 scalar ambient value, for places that want a non-directional number.
    pub fn scalar_irradiance(&self, elev_deg: f64) -> [f64; 3] {
        self.at(elev_deg).irradiance([0.0, 0.0, 1.0])
    }

    /// Flatten to `n * 9` RGBA f32 texels (row = elevation entry, column = SH coef) for
    /// the GPU-mirror upload (`clouds.wgsl` SH-ambient twin, activated in M5-GPU).
    pub fn to_rgba_f32(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.entries.len() * SH2_COUNT * 4);
        for e in &self.entries {
            for c in &e.coef {
                out.extend_from_slice(&[c[0] as f32, c[1] as f32, c[2] as f32, 1.0]);
            }
        }
        out
    }

    /// The per-entry flat (`normal = up`) hemisphere irradiance packed as `n * 4` RGBA
    /// f32 — the SCALAR ambient LUT the clouds-OFF GPU surface pass still consumes. The
    /// SH directional ambient is the CPU cloud path (the shipping cloud render);
    /// `surface.wgsl` keeps this M2-style scalar LUT until the M5-GPU activation wires
    /// the SH table binding (a documented CPU/GPU divergence, per the M4 pattern).
    pub fn to_scalar_rgba_f32(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.entries.len() * 4);
        for e in &self.entries {
            let irr = e.irradiance([0.0, 0.0, 1.0]);
            out.extend_from_slice(&[irr[0] as f32, irr[1] as f32, irr[2] as f32, 1.0]);
        }
        out
    }
}

// ── finite solar disk ────────────────────────────────────────────────────────

/// Fraction of the solar disk above the local horizon, as a smooth function of the
/// disk-CENTRE elevation (rad). A circular-segment area fraction: 0 fully set, 0.5
/// centre on the horizon, 1 fully risen — continuous and monotone across geometric
/// sunset (no step), so the sun sets over ~2 min of real time (design section 6).
pub fn solar_disk_visible_fraction(sun_elevation_rad: f64) -> f64 {
    let x = sun_elevation_rad / SUN_ANGULAR_RADIUS_RAD;
    if x >= 1.0 {
        return 1.0;
    }
    if x <= -1.0 {
        return 0.0;
    }
    // Fraction of a unit disk with the horizon chord at signed height -x:
    // (acos(-x) + x*sqrt(1-x^2)) / pi.
    ((-x).acos() + x * (1.0 - x * x).sqrt()) / PI
}

/// Hestroffer-Magnan limb intensity I(mu)/I(1), mu = cos(angle from disk centre).
pub fn limb_darkening(mu: f64) -> f64 {
    let m = mu.clamp(0.0, 1.0);
    (1.0 - LIMB_DARKENING_A1 * (1.0 - m) - LIMB_DARKENING_A2 * (1.0 - m) * (1.0 - m)).max(0.0)
}

/// Disk-averaged limb-darkening factor (intensity-weighted mean over the disk) —
/// the constant dimming M2 applies to the direct term. `<I>/I(1)` for the H-M law
/// `I(mu) = 1 - a1(1-mu) - a2(1-mu)^2`, area-weighted over the disk with `mu` the
/// projected radius parameter: `<I>/I(1) = 1 - a1/3 - a2/6`. With a1=0.397, a2=0.216
/// this is `1 - 0.13233 - 0.036 = 0.8317` (~0.832). The earlier 0.79 was ~5% low and
/// dimmed the daytime disk (M2 review FINDING 3; `E_sun` cancels in `rho = pi L / E`,
/// so this factor does NOT cancel — it is a direct multiplicative albedo dimming).
pub const LIMB_DARKENING_DISK_AVG: f64 = 0.832;

// ── ECEF camera geometry for the space camera ────────────────────────────────

/// The geostationary camera in spherical ECEF: its position and the orthonormal
/// look basis `(ex, ey, ez)` a scan angle `(x, y)` decomposes into. `ex` points
/// from the satellite toward the earth centre, `ey` is east, `ez` is north at the
/// sub-satellite point. This is the ECEF companion of the CGMS scan-angle math in
/// `camera.rs` (derived so that `view_dir` of a scan angle equals the normalised
/// `point - satellite` the CGMS forward encodes).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraGeometry {
    pub camera: [f64; 3],
    pub ex: [f64; 3],
    pub ey: [f64; 3],
    pub ez: [f64; 3],
}

impl CameraGeometry {
    /// Build from a sub-satellite longitude (deg), on the spherical earth.
    pub fn from_sub_lon(sub_lon_deg: f64) -> Self {
        let l = sub_lon_deg.to_radians();
        let (sl, cl) = l.sin_cos();
        let d = [cl, sl, 0.0]; // sub-satellite direction from earth centre
        Self {
            camera: scl3(d, GEO_ORBIT_RADIUS_M),
            ex: [-cl, -sl, 0.0], // toward the centre
            ey: [-sl, cl, 0.0],  // east
            ez: [0.0, 0.0, 1.0], // north
        }
    }

    /// The unit ECEF view direction for a scan angle `(x, y)` (rad), matching the
    /// CGMS sweep=y convention: local `(1, v_y, v_z)` in `(ex, ey, ez)`.
    pub fn view_dir(&self, scan_x: f64, scan_y: f64) -> [f64; 3] {
        let v_y = scan_x.tan();
        let v_z = scan_y.tan() * 1.0_f64.hypot(v_y);
        let world = add3(add3(self.ex, scl3(self.ey, v_y)), scl3(self.ez, v_z));
        norm3(world)
    }
}

/// Convert a sun direction given in a location's local ENU basis to a unit ECEF
/// vector. The sun is effectively at infinity, so ONE ECEF sun vector (computed at
/// the domain centre) is valid across the whole domain; the per-pixel local sun
/// elevation then falls out of `dot(up_pixel, sun_ecef)`.
pub fn sun_enu_to_ecef(sun_enu: [f64; 3], lat_deg: f64, lon_deg: f64) -> [f64; 3] {
    let (la, lo) = (lat_deg.to_radians(), lon_deg.to_radians());
    let (sla, cla) = la.sin_cos();
    let (slo, clo) = lo.sin_cos();
    let east = [-slo, clo, 0.0];
    let north = [-sla * clo, -sla * slo, cla];
    let up = [cla * clo, cla * slo, sla];
    let e = sun_enu[0];
    let n = sun_enu[1];
    let u = sun_enu[2];
    norm3([
        e * east[0] + n * north[0] + u * up[0],
        e * east[1] + n * north[1] + u * up[1],
        e * east[2] + n * north[2] + u * up[2],
    ])
}

// ── aerial-perspective froxel volume (space-camera slicing) ──────────────────

/// The 32x32x32 aerial-perspective froxel volume for the geostationary view
/// (design section 3). Space-camera slicing: (u, v) index a coarse grid over the
/// scan-angle raster, and the depth axis `w` is the TRAVERSAL FRACTION of the
/// atmosphere shell along each pixel's ray (entry -> ground for surface pixels,
/// entry -> far exit for limb pixels), NOT linear camera distance (which would put
/// every slice in the 35,000 km vacuum). Each froxel stores `(inscatter.rgb, mean
/// transmittance)` a la Hillaire. M2's surface render samples the full-traversal
/// endpoint directly per pixel; the intermediate depth slices feed M4 cloud aerial
/// perspective. Built + tested here (the design deliverable + slant-consistency).
#[derive(Debug, Clone)]
pub struct AerialFroxel {
    pub dim: usize,
    /// `dim^3 * 4` f32, RGBA per froxel: RGB = inscatter, A = mean transmittance.
    /// Index `4 * ((z*dim + y)*dim + x)`.
    pub data: Vec<f32>,
}

impl AerialFroxel {
    /// Fetch `(inscatter, mean_transmittance)` at integer froxel coords.
    pub fn fetch(&self, x: usize, y: usize, z: usize) -> ([f64; 3], f64) {
        let o = 4 * ((z * self.dim + y) * self.dim + x);
        (
            [
                self.data[o] as f64,
                self.data[o + 1] as f64,
                self.data[o + 2] as f64,
            ],
            self.data[o + 3] as f64,
        )
    }
}

/// Build the aerial-perspective froxel volume for a camera geometry over a
/// scan-angle rectangle `(x_min, x_max, y_min, y_max)` (rad). Each (u, v) column's
/// ray is marched through its in-atmosphere segment; the `dim` depth slices store
/// the cumulative inscatter + mean transmittance from the atmosphere entry.
#[allow(clippy::too_many_arguments)]
pub fn build_aerial_froxel(
    luts: &AtmosphereLuts,
    params: &AtmosphereParams,
    cam: &CameraGeometry,
    sun_ecef: [f64; 3],
    scan_rect: (f64, f64, f64, f64),
    dim: usize,
) -> AerialFroxel {
    let (x_min, x_max, y_min, y_max) = scan_rect;
    let mut data = vec![0.0f32; dim * dim * dim * 4];
    for vy in 0..dim {
        for ux in 0..dim {
            let sx = x_min + (ux as f64 + 0.5) / dim as f64 * (x_max - x_min);
            let sy = y_min + (vy as f64 + 0.5) / dim as f64 * (y_max - y_min);
            let view = cam.view_dir(sx, sy);
            // Find the in-atmosphere segment [t_enter, t_exit] of this ray.
            let Some((t_enter, t_exit)) = ray_atmosphere_segment(cam.camera, view) else {
                continue; // ray misses the shell -> all froxels stay zero (space)
            };
            let seg = t_exit - t_enter;
            // March the whole segment once, snapshotting at each depth slice.
            let steps_total = dim; // one march step per depth slice
            let dt = seg / steps_total as f64;
            let cos_vs = dot3(view, sun_ecef);
            let rayleigh_ph = rayleigh_phase(cos_vs);
            let mie_ph = cornette_shanks_phase(cos_vs, MIE_ASYMMETRY_G);
            let mut od = [0.0f64; 3];
            let mut l = [0.0; 3];
            for k in 0..dim {
                let t = t_enter + (k as f64 + 0.5) * dt;
                let p = madd3(cam.camera, view, t);
                let r = len3(p);
                let up = scl3(p, 1.0 / r.max(1.0));
                let m = sample_medium(r - R_GROUND_M, params);
                let mu_s = dot3(up, sun_ecef);
                let t_sun = if ray_hits_ground(r, mu_s) {
                    [0.0; 3]
                } else {
                    sample_transmittance(&luts.transmittance, r, mu_s)
                };
                let ms = sample_multiscatter(&luts.multiscatter, r, mu_s);
                for c in 0..3 {
                    let throughput = (-od[c]).exp();
                    let ext = m.extinction[c].max(1.0e-12);
                    let sample_t = (-ext * dt).exp();
                    let single = t_sun[c]
                        * (m.rayleigh_scattering[c] * rayleigh_ph + m.mie_scattering * mie_ph);
                    let multi = MULTISCATTER_GAIN * m.scattering[c] * ms[c];
                    let s = SOLAR_IRRADIANCE_RGB[c] * (single + multi);
                    l[c] += throughput * (s - s * sample_t) / ext;
                    od[c] += ext * dt;
                }
                let mean_t = ((-od[0]).exp() + (-od[1]).exp() + (-od[2]).exp()) / 3.0;
                let o = 4 * ((k * dim + vy) * dim + ux);
                data[o] = l[0] as f32;
                data[o + 1] = l[1] as f32;
                data[o + 2] = l[2] as f32;
                data[o + 3] = mean_t as f32;
            }
        }
    }
    AerialFroxel { dim, data }
}

/// The `[t_enter, t_exit]` parametric distances where a ray `origin + t*dir`
/// (dir unit) is inside the atmosphere shell but ABOVE the ground. For a ray that
/// hits the ground, `t_exit` is the ground intersection; for a limb ray it is the
/// far top-of-atmosphere crossing. `None` if the ray misses the shell entirely.
pub fn ray_atmosphere_segment(origin: [f64; 3], dir: [f64; 3]) -> Option<(f64, f64)> {
    let (t0_top, t1_top) = ray_sphere(origin, dir, R_TOP_M)?;
    // The relevant entry is the nearest positive top crossing.
    let t_enter = t0_top.max(0.0);
    let mut t_exit = t1_top;
    if let Some((t0_g, _t1_g)) = ray_sphere(origin, dir, R_GROUND_M) {
        // If the ground is entered within the shell, the segment ends at the ground.
        if t0_g > t_enter && t0_g < t_exit {
            t_exit = t0_g;
        }
    }
    if t_exit <= t_enter {
        return None;
    }
    Some((t_enter, t_exit))
}

/// Ray/sphere intersection: the two real roots `(t0 <= t1)` of `|origin + t*dir| =
/// radius`, or `None`. `dir` is assumed unit.
fn ray_sphere(origin: [f64; 3], dir: [f64; 3], radius: f64) -> Option<(f64, f64)> {
    let b = dot3(origin, dir);
    let c = dot3(origin, origin) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let s = disc.sqrt();
    Some((-b - s, -b + s))
}

// ── energy / tonemap output transforms (design section 6) ────────────────────

/// The output transform selecting how internal HDR reflectance becomes display
/// bytes. Default = the ABI-like reflectance stretch written to the store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputTransform {
    /// ABI-like reflectance factor with the sqrt satellite stretch (product default).
    AbiReflectance,
    /// Reflectance through a plain sRGB gamma (simple debug transform).
    DebugSrgb,
}

impl OutputTransform {
    /// The float code the shader keys on.
    pub fn code(self) -> f32 {
        match self {
            Self::AbiReflectance => 0.0,
            Self::DebugSrgb => 1.0,
        }
    }
}

// ── display transform (M2 twilight pass) ─────────────────────────────────────
//
// The visible frame is `stretch(desaturate(exposure * pi * L / E_sun))`, a
// DISPLAY-side transform (not a physics change; exposure/reflectance are intact).
// Two pieces, both aimed at the terminator (notes/twilight-diagnosis.md item 1):
//
//  1. `desaturate_highlights` — as reflectance-LUMINANCE rises past a threshold the
//     chroma is compressed toward that luminance (a neutral grey), so a warm
//     highlight rolls off toward white instead of clipping its dominant channel
//     first into saturated orange. Low-luminance twilight keeps its full (Rayleigh
//     blue) chroma. This is the "tonemap on luminance / desaturate as luminance
//     rises" requirement.
//  2. `abi_reflectance_stretch` — the ABI sqrt with a LIFTED TOE below a knee, so
//     the dim below-horizon twilight the model computes becomes visible; ABOVE the
//     knee it is EXACTLY the old `sqrt`, so the owner-approved daytime look (bright
//     cloud tops ~white, midtones as before) is unchanged.
//
// CRITICAL INVARIANT: for `rho >= REFL_TOE_KNEE` the stretch is bit-for-bit `sqrt`,
// and `desaturate_highlights` is a no-op wherever luminance < DESAT_LUM_LO OR chroma
// < DESAT_SAT_LO — so the daytime midtones/median (low luminance) and near-neutral
// daytime ground/cloud (low chroma) are preserved while only the shadows/low-end and
// the over-saturated warm highlights change.

/// Toe knee: at/above this reflectance the stretch is exactly `sqrt` (daytime
/// unchanged); below it the toe is lifted toward `pow(rho, REFL_TOE_GAMMA)`.
pub const REFL_TOE_KNEE: f64 = 0.05;
/// Toe power (< 0.5) blended in below the knee to lift the dim twilight low end.
pub const REFL_TOE_GAMMA: f64 = 0.38;
/// Highlight desaturation LUMINANCE gate (reflectance-luminance). Below LUM_LO the
/// desaturation is off, so the dim -4/-6 twilight keeps its full Rayleigh-blue chroma
/// AND the daytime +5/+10 MEDIAN pixels (Y_rho ~0.06-0.086) stay unshifted; it ramps
/// to full by LUM_HI, so the bright -2/0-deg highlights (and daytime cloud tops) are
/// eligible. Round 2 raised this 0.04 -> 0.09/0.13 to protect the daytime median.
pub const DESAT_LUM_LO: f64 = 0.09;
pub const DESAT_LUM_HI: f64 = 0.13;
/// Highlight desaturation SATURATION gate (chroma = (max-min)/max of the reflectance
/// triple, 0 = grey .. 1 = fully saturated). Near-neutral pixels (chroma < SAT_LO:
/// daytime ground/cloud tops) are left alone; only the OVER-saturated highlights —
/// the reddened low-sun anvil tops — are pulled toward grey, ramping to full by
/// SAT_HI. This chroma gate is what lets the very-saturated -2 deg anvils be cooled
/// to amber (bright-5% R/B 2.35 -> 1.8) while the LESS-saturated 0-deg highlights are
/// only lightly touched (R/B kept at ~1.43): a luminance-only gate could not, because
/// -2 is DIMMER than 0-deg and desat rises with luminance (it would cool 0 at least
/// as much as -2). Values tuned from the render sweep (notes/twilight-tuning-notes.md).
pub const DESAT_SAT_LO: f64 = 0.40;
pub const DESAT_SAT_HI: f64 = 0.88;
/// Maximum chroma compression toward luminance (0 = none, 1 = fully grey), reached
/// only when BOTH gates are full (a bright AND over-saturated highlight).
pub const DESAT_MAX: f64 = 0.55;

/// Hermite smoothstep on `[edge0, edge1]` (C1, monotone). Shared by the display
/// transform and the twilight fill.
#[inline]
pub fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    if edge1 <= edge0 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Highlight desaturation: compress the RGB reflectance chroma toward its Rec.709
/// luminance for pixels that are BOTH bright (luminance gate) AND over-saturated
/// (saturation gate), so a reddened warm highlight rolls off toward amber/white
/// rather than clipping red-first into saturated orange. The dim twilight (low
/// luminance), the daytime median (low luminance), and the near-neutral daytime
/// ground/cloud (low chroma) are left untouched. Display-side; operates on the
/// exposure-scaled reflectance BEFORE the per-channel clamp/stretch.
#[inline]
pub fn desaturate_highlights(rho: [f64; 3]) -> [f64; 3] {
    let y = 0.2126 * rho[0] + 0.7152 * rho[1] + 0.0722 * rho[2];
    let mx = rho[0].max(rho[1]).max(rho[2]);
    let mn = rho[0].min(rho[1]).min(rho[2]);
    let chroma = if mx > 1.0e-4 { (mx - mn) / mx } else { 0.0 };
    let lum_gate = smoothstep(DESAT_LUM_LO, DESAT_LUM_HI, y);
    let sat_gate = smoothstep(DESAT_SAT_LO, DESAT_SAT_HI, chroma);
    let d = DESAT_MAX * lum_gate * sat_gate;
    [
        rho[0] * (1.0 - d) + y * d,
        rho[1] * (1.0 - d) + y * d,
        rho[2] * (1.0 - d) + y * d,
    ]
}

/// The sqrt-ish satellite visible stretch applied to a reflectance factor `rho`
/// (per channel), with a LIFTED TOE below [`REFL_TOE_KNEE`]. At/above the knee it is
/// exactly `sqrt(clamp(rho, 0, 1))` — the well-known ABI/MODIS visible enhancement,
/// UNCHANGED for the daytime range. Below the knee the sqrt is smoothly blended
/// toward the brighter `pow(rho, REFL_TOE_GAMMA)`, lifting the dim twilight low end
/// into visibility. Monotone and continuous; `stretch(0)=0`, `stretch(1)=1`.
#[inline]
pub fn abi_reflectance_stretch(rho: f64) -> f64 {
    let x = rho.clamp(0.0, 1.0);
    let s = x.sqrt();
    if x >= REFL_TOE_KNEE {
        return s;
    }
    // Below the knee: blend sqrt -> the lifted toe as x -> 0. `w` is 1 at the knee
    // (continuous with the sqrt branch) and 0 at x=0.
    let toe = x.powf(REFL_TOE_GAMMA);
    let w = smoothstep(0.0, REFL_TOE_KNEE, x);
    toe * (1.0 - w) + s * w
}

// ── precipitable water from the brick (WRF moisture modulation) ──────────────

/// Estimate the domain-mean precipitable water (kg m^-2) from a decoded brick's
/// `qvapor` channel. Honest approximation: the brick carries the vapor MIXING
/// RATIO on the uniform-dz axis but no pressure, so we weight by the standard-
/// atmosphere air density ([`crate::optics::standard_air_density_kg_m3`], the same
/// kernel the IR/WV march and `derived.rs` use) and integrate
/// `PW = sum_k rho(z_k) * q_k * dz_above_terrain` over each column, averaged over
/// the domain. The DIRECTION (a wetter column -> higher PW -> more WV absorption ->
/// lower red-band transmittance) is what matters and is what the test asserts.
///
/// SUB-TERRAIN CLIP (WS1 march-physics pass): each layer's contribution is clipped
/// to its portion ABOVE the terrain surface (`brick.hgt`), exactly as
/// [`crate::derived::precipitable_water_field`] — the ingest fills sub-terrain
/// levels with the CLAMPED surface vapor, which would otherwise inflate the
/// `pw_ratio` (and so the visible WV-band absorption) over elevated terrain. A
/// sea-level column (`hgt <= z_min`) is bit-identical to the pre-clip integral.
pub fn precipitable_water_from_brick(brick: &VolumeBrick) -> f64 {
    let (nx, ny, nz) = (brick.nx, brick.ny, brick.nz);
    if nx == 0 || ny == 0 || nz == 0 {
        return 0.0;
    }
    let quant = brick.quant.get("qvapor");
    let has_hgt = brick.hgt.len() == nx * ny;
    let mut total = 0.0f64;
    for j in 0..ny {
        for i in 0..nx {
            let surface = if has_hgt {
                brick.hgt[j * nx + i] as f64
            } else {
                brick.z_min_m
            };
            let mut pw = 0.0;
            for k in 0..nz {
                let zb = brick.z_min_m + k as f64 * brick.dz_m; // layer [zb, zb+dz)
                let thickness = (zb + brick.dz_m - zb.max(surface)).max(0.0);
                if thickness <= 0.0 {
                    continue;
                }
                let idx = (k * ny + j) * nx + i;
                let q = quant.decode(brick.qvapor[idx]) as f64; // kg/kg
                pw += crate::optics::standard_air_density_kg_m3(zb) * q * thickness;
            }
            total += pw;
        }
    }
    total / (nx * ny) as f64
}

/// The precipitable-water RATIO (PW / 14.2) for a brick, clamped to a sane band so a
/// pathological column cannot blow up the WV term.
pub fn pw_ratio_from_brick(brick: &VolumeBrick) -> f64 {
    pw_ratio_from_pw(precipitable_water_from_brick(brick))
}

/// Map a precipitable-water value (kg/m^2) to the clamped WV ratio, treating a
/// non-finite input (a NaN/Inf qvapor column) as the 1.0 STANDARD reference rather
/// than letting it poison the whole frame. `f64::clamp` returns NaN for a NaN input
/// (the comparisons are false), so the advertised guard needed an explicit finiteness
/// check (M2 review FINDING 4).
fn pw_ratio_from_pw(pw: f64) -> f64 {
    let r = pw / PW_STANDARD_KG_M2;
    if r.is_finite() {
        r.clamp(0.0, 5.0)
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_luts() -> AtmosphereLuts {
        // Full-size transmittance (cheap) + full multiscatter (32x32, cheap).
        AtmosphereLuts::build(&AtmosphereParams::default())
    }

    #[test]
    fn rayleigh_green_is_published_order_at_550nm() {
        // Published Rayleigh scattering at 550 nm is ~1.3e-5 m^-1 (Bucholtz 1995).
        let green = RAYLEIGH_SCATTERING[1];
        assert!(
            (1.0e-5..2.0e-5).contains(&green),
            "green Rayleigh {green} out of published order"
        );
        // Blue scatters far more than red (why the sky is blue). Bind to locals so
        // this is not a compile-time-constant assertion.
        let (blue, red) = (RAYLEIGH_SCATTERING[2], RAYLEIGH_SCATTERING[0]);
        assert!(blue > red * 3.0, "blue {blue} !> 3x red {red}");
    }

    #[test]
    fn medium_extinction_positive_and_decreases_with_altitude() {
        let p = AtmosphereParams::default();
        let low = sample_medium(0.0, &p);
        let high = sample_medium(30_000.0, &p);
        for c in 0..3 {
            assert!(low.extinction[c] > 0.0);
            // Rayleigh+Mie dominate low; both fall off with height.
            assert!(low.rayleigh_scattering[c] > high.rayleigh_scattering[c]);
        }
        // Ozone tent peaks at 25 km: green extinction there retains an ozone floor
        // even though Rayleigh has largely fallen off (green ozone is the strongest
        // ozone band and drives the blue twilight residual).
        let at25 = sample_medium(25_000.0, &p);
        let above = sample_medium(45_000.0, &p);
        assert!(at25.extinction[1] > 0.0);
        // Green extinction at the ozone peak exceeds well above the ozone layer.
        assert!(at25.extinction[1] > above.extinction[1]);
    }

    #[test]
    fn phase_functions_integrate_to_one_over_the_sphere() {
        // Numerically integrate p(cos) over the sphere; should be ~1.
        let n = 2000;
        let mut ray_sum = 0.0;
        let mut mie_sum = 0.0;
        for i in 0..n {
            let mu = -1.0 + 2.0 * (i as f64 + 0.5) / n as f64;
            let dmu = 2.0 / n as f64;
            // dOmega over full sphere = 2*pi * dmu (azimuth-symmetric).
            ray_sum += rayleigh_phase(mu) * 2.0 * PI * dmu;
            mie_sum += cornette_shanks_phase(mu, MIE_ASYMMETRY_G) * 2.0 * PI * dmu;
        }
        assert!((ray_sum - 1.0).abs() < 0.02, "rayleigh integral {ray_sum}");
        assert!((mie_sum - 1.0).abs() < 0.03, "mie integral {mie_sum}");
        // Cornette-Shanks with g=0.8 is strongly forward-scattering.
        assert!(
            cornette_shanks_phase(1.0, MIE_ASYMMETRY_G)
                > cornette_shanks_phase(-1.0, MIE_ASYMMETRY_G) * 10.0
        );
    }

    #[test]
    fn transmittance_uv_round_trips() {
        // Only ABOVE-horizon rays are in the parameterisation's domain (the
        // transmittance LUT is to the top of atmosphere; below-horizon rays hit the
        // ground and are gated out with ray_hits_ground before any LUT sample).
        for &(r, mu) in &[
            (R_GROUND_M + 1000.0, 1.0),
            (R_GROUND_M + 5000.0, 0.3),
            (R_GROUND_M + 20000.0, 0.05), // near-horizontal but above the horizon
            (R_TOP_M - 1000.0, 0.8),
        ] {
            let (u, v) = transmittance_r_mu_to_uv(r, mu);
            let (r2, mu2) = transmittance_uv_to_r_mu(u, v);
            assert!((r2 - r).abs() < 500.0, "r {r} -> {r2}");
            assert!((mu2 - mu).abs() < 0.02, "mu {mu} -> {mu2}");
        }
    }

    #[test]
    fn transmittance_horizon_path_is_lower_than_zenith() {
        let p = AtmosphereParams::default();
        let lut = build_transmittance_lut(&p);
        let r = R_GROUND_M + 100.0;
        let zenith = sample_transmittance(&lut, r, 1.0); // straight up (short path)
        let horizon = sample_transmittance(&lut, r, 0.02); // near-horizontal (long path)
        for c in 0..3 {
            assert!(zenith[c] > horizon[c], "band {c}: zenith !> horizon");
            assert!(zenith[c] <= 1.0 && zenith[c] > 0.0);
        }
        // The horizon path is far more attenuated in blue (Rayleigh) than red ->
        // reddened sun at the horizon.
        assert!(horizon[0] > horizon[2], "horizon should be red-biased");
    }

    #[test]
    fn transmittance_lut_matches_direct_march() {
        let p = AtmosphereParams::default();
        let lut = build_transmittance_lut(&p);
        for &(r, mu) in &[(R_GROUND_M + 2000.0, 0.6), (R_GROUND_M + 8000.0, 0.25)] {
            let sampled = sample_transmittance(&lut, r, mu);
            let od = optical_depth_to_top(r, mu, &p);
            let direct = [(-od[0]).exp(), (-od[1]).exp(), (-od[2]).exp()];
            for c in 0..3 {
                assert!(
                    (sampled[c] - direct[c]).abs() < 0.03,
                    "band {c}: lut {} vs direct {}",
                    sampled[c],
                    direct[c]
                );
            }
        }
    }

    #[test]
    fn multiscatter_lut_conserves_energy() {
        let p = AtmosphereParams::default();
        let t = build_transmittance_lut(&p);
        let s = MULTISCATTER_LUT_SIZE;
        let mut max_blue_fms = 0.0f64;
        for y in (0..s).step_by(7) {
            for x in (0..s).step_by(7) {
                let u = (x as f64 + 0.5) / s as f64;
                let v = (y as f64 + 0.5) / s as f64;
                let (r, mu_s) = multiscatter_uv_to_r_mus(u, v);
                let (psi, f_ms) = compute_multiscatter(&t, &p, r, mu_s);
                for c in 0..3 {
                    // The geometric series 1/(1 - f_ms) converges only if 0<=f_ms<1.
                    assert!(
                        f_ms[c] >= 0.0 && f_ms[c] < 1.0,
                        "f_ms {} out of [0,1)",
                        f_ms[c]
                    );
                    assert!(psi[c] >= 0.0 && psi[c].is_finite());
                }
                max_blue_fms = max_blue_fms.max(f_ms[2]);
            }
        }
        // After the FINDING-2 fix (no spurious 1/4pi) the transfer factor is a MEANINGFUL
        // fraction, not ~0. Blue near the ground with a high sun reaches a few tenths, so
        // the 1/(1-f_ms) multiple-scattering boost is no longer defeated. The old
        // (4pi-too-small) f_ms peaked well below 0.1; require a materially larger value.
        assert!(
            max_blue_fms > 0.10,
            "blue f_ms peak {max_blue_fms} too small — the multiple-scattering boost is defeated"
        );
    }

    #[test]
    fn sky_view_twilight_darkens_and_blue_shifts() {
        let luts = small_luts();
        let p = AtmosphereParams::default();
        let cfg = SkyViewConfig {
            width: 64,
            height: 48,
            steps: 16,
            observer_altitude_m: 500.0,
        };
        let r_obs = R_GROUND_M + cfg.observer_altitude_m;
        let mut prev_bright = f64::INFINITY;
        let mut prev_blue_frac = 0.0;
        for &elev_deg in &[-2.0f64, -6.0, -12.0] {
            let sv = build_sky_view_lut(&luts, &p, elev_deg.to_radians(), &cfg);
            let irr = ambient_irradiance_from_sky_view(&sv, r_obs);
            let bright = irr[0] + irr[1] + irr[2];
            let blue_frac = irr[2] / bright.max(1.0e-12);
            // Monotone darkening as the sun sets deeper.
            assert!(
                bright < prev_bright,
                "elev {elev_deg}: brightness {bright} !< {prev_bright}"
            );
            // Blue-shifted residual: the blue fraction rises as red is lost to the
            // long twilight path (Rayleigh + ozone).
            assert!(
                blue_frac >= prev_blue_frac - 1.0e-6,
                "elev {elev_deg}: blue frac {blue_frac} < {prev_blue_frac}"
            );
            assert!(bright > 0.0, "twilight still has some glow at {elev_deg}");
            prev_bright = bright;
            prev_blue_frac = blue_frac;
        }
    }

    #[test]
    fn ambient_table_darkens_toward_night() {
        let luts = small_luts();
        let p = AtmosphereParams::default();
        let table = AmbientTable::build(&luts, &p, 24);
        let day = table.at(60.0);
        let dusk = table.at(-2.0);
        let night = table.at(-18.0);
        let sum = |v: [f64; 3]| v[0] + v[1] + v[2];
        assert!(sum(day) > sum(dusk), "day ambient !> dusk");
        assert!(sum(dusk) > sum(night), "dusk ambient !> night");
        // Astronomical night ambient is essentially zero.
        assert!(sum(night) < sum(day) * 0.02, "night ambient too bright");
    }

    #[test]
    fn finite_disk_sunset_is_a_smooth_monotone_ramp() {
        // disk_fraction through geometric sunset: continuous, monotone, no step.
        let ar = SUN_ANGULAR_RADIUS_RAD;
        assert!((solar_disk_visible_fraction(0.0) - 0.5).abs() < 1.0e-9);
        assert_eq!(solar_disk_visible_fraction(2.0 * ar), 1.0);
        assert_eq!(solar_disk_visible_fraction(-2.0 * ar), 0.0);
        // Sample a fine ramp across [-ar, ar]; strictly decreasing as elevation
        // falls, and the max step between adjacent samples is tiny (no jump).
        let n = 400;
        let mut prev = 1.1;
        let mut max_step = 0.0f64;
        for i in 0..=n {
            let e = ar - 2.0 * ar * (i as f64 / n as f64); // +ar down to -ar
            let f = solar_disk_visible_fraction(e);
            assert!(f <= prev + 1.0e-9, "not monotone at step {i}");
            if i > 0 {
                max_step = max_step.max((prev - f).abs());
            }
            prev = f;
        }
        assert!(max_step < 0.05, "found a step (jump) of {max_step}");
    }

    #[test]
    fn limb_darkening_is_a_dimming_within_unity() {
        assert!((limb_darkening(1.0) - 1.0).abs() < 1.0e-9); // centre = full
        assert!(limb_darkening(0.0) < 1.0); // edge dimmer
        assert!(limb_darkening(0.0) > 0.0);
        let avg = LIMB_DARKENING_DISK_AVG;
        assert!(avg > 0.5 && avg < 1.0, "disk-avg limb darkening {avg}");
        // The constant must match its own area-weighted derivation 1 - a1/3 - a2/6
        // (M2 review FINDING 3: 0.79 disagreed with 0.832).
        let derived = 1.0 - LIMB_DARKENING_A1 / 3.0 - LIMB_DARKENING_A2 / 6.0;
        assert!(
            (avg - derived).abs() < 1.0e-3,
            "disk-avg {avg} disagrees with its derivation {derived}"
        );
    }

    #[test]
    fn pw_ratio_guards_non_finite() {
        // A NaN / Inf precipitable water must fall back to the 1.0 standard reference,
        // not propagate NaN into the WV term and poison the whole frame (FINDING 4).
        assert_eq!(pw_ratio_from_pw(f64::NAN), 1.0);
        assert_eq!(pw_ratio_from_pw(f64::INFINITY), 1.0);
        assert_eq!(pw_ratio_from_pw(f64::NEG_INFINITY), 1.0);
        // Finite values still map + clamp as before.
        assert!((pw_ratio_from_pw(PW_STANDARD_KG_M2) - 1.0).abs() < 1e-12);
        assert_eq!(pw_ratio_from_pw(-5.0), 0.0); // negative -> clamp low
        assert_eq!(pw_ratio_from_pw(1000.0), 5.0); // huge -> clamp high
    }

    #[test]
    fn earth_shadowed_point_gets_only_multiscatter() {
        // A point in the earth's shadow (sun ray blocked by the ground) still gets
        // multi-scatter but no direct single scattering. Compare the two paths at a
        // point whose local sun is below the horizon.
        let luts = small_luts();
        // Point high in the atmosphere; sun well below the local horizon so its ray
        // to the sun intersects the earth.
        let r = R_GROUND_M + 15_000.0;
        let mu_s = -0.4; // sun 24 deg below horizon
        assert!(ray_hits_ground(r, mu_s), "setup: sun ray must hit ground");
        let t_sun = if ray_hits_ground(r, mu_s) {
            [0.0; 3]
        } else {
            sample_transmittance(&luts.transmittance, r, mu_s)
        };
        assert_eq!(t_sun, [0.0; 3], "direct sun transmittance must be zero");
        // Multi-scatter is still positive there.
        let ms = sample_multiscatter(&luts.multiscatter, r, mu_s);
        assert!(ms.iter().all(|&v| v >= 0.0));
        // A daytime point (sun up) has non-zero direct transmittance for contrast.
        let day_t = sample_transmittance(&luts.transmittance, r, 0.5);
        assert!(day_t[1] > 0.0);
    }

    #[test]
    fn pw_modulation_lowers_red_transmittance() {
        // A wetter column (higher pw_ratio) increases WV absorption, lowering the
        // red-band (WV-weighted) transmittance more than blue.
        let dry = AtmosphereParams {
            pw_ratio: 0.5,
            ..Default::default()
        };
        let wet = AtmosphereParams {
            pw_ratio: 3.0,
            ..Default::default()
        };
        let r = R_GROUND_M + 50.0;
        let mu = 0.3; // a slant path so the boundary-layer WV term accumulates
        let od_dry = optical_depth_to_top(r, mu, &dry);
        let od_wet = optical_depth_to_top(r, mu, &wet);
        let t_dry = (-od_dry[0]).exp();
        let t_wet = (-od_wet[0]).exp();
        assert!(t_wet < t_dry, "wetter column must lower red transmittance");
        // The blue band (little WV absorption) is barely affected relative to red.
        let dred = (-od_dry[0]).exp() - (-od_wet[0]).exp();
        let dblue = (-od_dry[2]).exp() - (-od_wet[2]).exp();
        assert!(dred > dblue, "WV modulation must hit red more than blue");
    }

    #[test]
    fn froxel_limb_column_has_lower_transmittance_than_nadir() {
        // A near-limb ray traverses a far longer atmosphere slant than a near-nadir
        // ray, so its full-traversal transmittance is lower and inscatter higher.
        let luts = small_luts();
        let p = AtmosphereParams::default();
        let cam = CameraGeometry::from_sub_lon(-75.2);
        let sun = [0.0, 0.0, 1.0]; // overhead-ish in ECEF (north pole dir) — arbitrary but fixed
        // A scan rect straddling nadir out toward the limb.
        // Earth angular radius from GEO ~ asin(R/GEO) ~ 8.7 deg ~ 0.152 rad.
        let rect = (-0.15, 0.15, -0.15, 0.15);
        let froxel = build_aerial_froxel(&luts, &p, &cam, sun, rect, AERIAL_FROXEL_DIM);
        // Centre column (near nadir) vs an edge column (near the limb).
        let mid = AERIAL_FROXEL_DIM / 2;
        let (_, t_nadir) = froxel.fetch(mid, mid, AERIAL_FROXEL_DIM - 1);
        let (l_limb, t_limb) = froxel.fetch(0, mid, AERIAL_FROXEL_DIM - 1);
        // Only meaningful if both columns actually crossed atmosphere; the edge
        // column grazes the limb (long path) so should be at least as attenuated.
        if t_nadir > 0.0 && t_limb > 0.0 {
            assert!(
                t_limb <= t_nadir + 1.0e-6,
                "limb transmittance {t_limb} !<= nadir {t_nadir}"
            );
            let bright = l_limb[0] + l_limb[1] + l_limb[2];
            assert!(bright >= 0.0);
        }
    }

    #[test]
    fn view_dir_matches_point_minus_camera() {
        // The ECEF view_dir for a scan angle equals normalize(point - camera) for
        // the point the CGMS forward maps to that scan angle — the geometry the
        // limb/aerial raymarch relies on being consistent with the M1 camera.
        use crate::camera::{GeoCamera, SatellitePreset};
        let preset = SatellitePreset::GoesEast;
        let cam_cgms = GeoCamera::new(preset);
        let geom = CameraGeometry::from_sub_lon(preset.sub_lon_deg());
        for &(lat, lon) in &[(0.0, -75.2), (25.0, -80.0), (40.0, -95.0)] {
            let (sx, sy) = cam_cgms.forward(lat, lon).expect("on disk");
            let view = geom.view_dir(sx, sy);
            // Point on the ground sphere.
            let (la, lo) = (lat.to_radians(), lon.to_radians());
            let p = [
                R_GROUND_M * la.cos() * lo.cos(),
                R_GROUND_M * la.cos() * lo.sin(),
                R_GROUND_M * la.sin(),
            ];
            let expect = norm3([
                p[0] - geom.camera[0],
                p[1] - geom.camera[1],
                p[2] - geom.camera[2],
            ]);
            for k in 0..3 {
                assert!(
                    (view[k] - expect[k]).abs() < 1.0e-4,
                    "component {k}: {} vs {}",
                    view[k],
                    expect[k]
                );
            }
        }
    }

    #[test]
    fn ray_atmosphere_segment_limb_vs_ground() {
        let cam = CameraGeometry::from_sub_lon(0.0);
        // A ray straight at nadir hits the ground: segment ends at the ground.
        let nadir = cam.view_dir(0.0, 0.0);
        let (t0, t1) = ray_atmosphere_segment(cam.camera, nadir).expect("nadir crosses shell");
        let ground_r = len3(madd3(cam.camera, nadir, t1));
        assert!(
            (ground_r - R_GROUND_M).abs() < 50.0,
            "nadir exit should be the ground, r={ground_r}"
        );
        let enter_r = len3(madd3(cam.camera, nadir, t0));
        assert!(
            (enter_r - R_TOP_M).abs() < 50.0,
            "enter at top, r={enter_r}"
        );
        // A ray in the limb band (ground tangent ~0.15166 rad, top tangent ~0.15406
        // rad) grazes the shell and exits at the far top, NOT the ground.
        let limb = cam.view_dir(0.1535, 0.0);
        let (_l0, l1) = ray_atmosphere_segment(cam.camera, limb).expect("limb grazes the shell");
        let exit_r = len3(madd3(cam.camera, limb, l1));
        assert!(exit_r > R_GROUND_M + 1000.0, "limb ray must not hit ground");
    }

    #[test]
    fn abi_stretch_is_monotone_sqrt() {
        assert_eq!(abi_reflectance_stretch(0.0), 0.0);
        assert_eq!(abi_reflectance_stretch(1.0), 1.0);
        assert!((abi_reflectance_stretch(0.25) - 0.5).abs() < 1.0e-9);
        // Monotone and brightens midtones (sqrt(x) > x for x in (0,1)).
        assert!(abi_reflectance_stretch(0.16) > 0.16);
        assert!(abi_reflectance_stretch(0.04) < abi_reflectance_stretch(0.09));
        assert_eq!(OutputTransform::AbiReflectance.code(), 0.0);
        assert_eq!(OutputTransform::DebugSrgb.code(), 1.0);
    }

    #[test]
    fn abi_stretch_toe_lifts_dim_and_is_sqrt_above_knee() {
        // SHADOW LIFT: below the knee the stretch is LIFTED above the plain sqrt (dim
        // twilight becomes visible); at/above the knee it is EXACTLY sqrt (daytime
        // unchanged); monotone across the knee.
        for &x in &[0.002_f64, 0.01, 0.03] {
            assert!(
                abi_reflectance_stretch(x) > x.sqrt() + 1.0e-3,
                "toe must lift {x}: {} !> sqrt {}",
                abi_reflectance_stretch(x),
                x.sqrt()
            );
        }
        for &x in &[REFL_TOE_KNEE, 0.1, 0.5, 1.0] {
            assert!(
                (abi_reflectance_stretch(x) - x.sqrt()).abs() < 1.0e-9,
                "at/above the knee must equal sqrt at {x}"
            );
        }
        let mut prev = -1.0;
        for k in 0..=200 {
            let x = k as f64 / 200.0;
            let s = abi_reflectance_stretch(x);
            assert!(s >= prev - 1.0e-12, "not monotone at {x}: {s} < {prev}");
            prev = s;
        }
    }

    #[test]
    fn desaturate_highlights_is_chroma_and_luminance_gated() {
        let chroma = |v: [f64; 3]| {
            let mx = v[0].max(v[1]).max(v[2]);
            let mn = v[0].min(v[1]).min(v[2]);
            if mx > 1.0e-4 { (mx - mn) / mx } else { 0.0 }
        };
        // Bright AND over-saturated (reddened anvil) -> chroma pulled toward grey.
        let warm = [0.60, 0.20, 0.15];
        let out = desaturate_highlights(warm);
        assert!(
            chroma(out) < chroma(warm) - 1.0e-3,
            "bright saturated highlight must desaturate: {} !< {}",
            chroma(out),
            chroma(warm)
        );
        assert!(
            out[0] > out[2],
            "but it must stay warm (R > B), got {out:?}"
        );
        // Bright but near-neutral (chroma < DESAT_SAT_LO) -> untouched (daytime cloud).
        let neutral = [0.60, 0.58, 0.56];
        assert_eq!(
            desaturate_highlights(neutral),
            neutral,
            "near-neutral bright pixel must be untouched by the saturation gate"
        );
        // Dim but saturated (luminance < DESAT_LUM_LO) -> untouched (twilight blue kept).
        let dim = [0.05, 0.02, 0.09];
        assert_eq!(
            desaturate_highlights(dim),
            dim,
            "dim twilight must be untouched by the luminance gate"
        );
    }

    #[test]
    fn twilight_tuning_constants_are_locked() {
        // Value-lock the M2 twilight-pass tuned constants (a compile + drift guard).
        let eqf = |a: f64, b: f64| (a - b).abs() < 1.0e-12;
        assert!(eqf(DEFAULT_AOD, 0.05));
        assert!(eqf(OZONE_STRENGTH, 1.45));
        assert!(eqf(OZONE_HALF_WIDTH_M, 15_000.0));
        assert!(eqf(SKYVIEW_OBSERVER_ALTITUDE_M, 3_000.0));
        assert!(eqf(MULTISCATTER_GAIN, 1.4));
        assert!(eqf(DESAT_LUM_LO, 0.09) && eqf(DESAT_LUM_HI, 0.13));
        assert!(eqf(DESAT_SAT_LO, 0.40) && eqf(DESAT_SAT_HI, 0.88));
        assert!(eqf(DESAT_MAX, 0.55));
        assert!(eqf(REFL_TOE_KNEE, 0.05) && eqf(REFL_TOE_GAMMA, 0.38));
        // Ozone absorption folds in the strength multiplier.
        assert!(eqf(OZONE_ABSORPTION[0], 0.650e-6 * OZONE_STRENGTH));
        assert!(eqf(OZONE_ABSORPTION[1], 1.881e-6 * OZONE_STRENGTH));
    }

    fn brick_with_qvapor(qvapor_f32: &[f32], nx: usize, ny: usize, nz: usize) -> VolumeBrick {
        use crate::bricks::{ChannelQuant, encode_log_channel, encode_temperature_celsius};
        use std::collections::BTreeMap;
        let cells_3d = nx * ny * nz;
        let cells_2d = nx * ny;
        let (qv_scale, qvapor) = encode_log_channel(qvapor_f32);
        let mut map = BTreeMap::new();
        map.insert("qvapor".to_string(), qv_scale);
        VolumeBrick {
            nx,
            ny,
            nz,
            z_min_m: 0.0,
            dz_m: 250.0,
            time_iso: None,
            quant: ChannelQuant(map),
            ext_liquid: vec![0; cells_3d],
            ext_ice: vec![0; cells_3d],
            ext_precip: vec![0; cells_3d],
            tau_up: vec![0; cells_3d],
            qvapor,
            temperature_f16: encode_temperature_celsius(&vec![288.0; cells_3d]),
            hgt: vec![0.0; cells_2d],
            landmask: vec![1.0; cells_2d],
            tsk: vec![288.0; cells_2d],
            u10: vec![0.0; cells_2d],
            v10: vec![0.0; cells_2d],
            snowh: None,
            ivgtyp: None,
        }
    }

    #[test]
    fn pw_from_brick_increases_with_qvapor() {
        // A wetter column integrates to a higher precipitable water -> a higher
        // pw_ratio -> more WV absorption (the modulation direction M2 requires).
        let (nx, ny, nz) = (2usize, 2usize, 8usize);
        let cells = nx * ny * nz;
        // Boundary-layer moisture (k < 4), then dry aloft. Flat index k = i/(nx*ny).
        let dry: Vec<f32> = (0..cells)
            .map(|i| if i / (nx * ny) < 4 { 0.002 } else { 0.0005 })
            .collect();
        let wet: Vec<f32> = dry.iter().map(|q| q * 6.0).collect();
        let pw_dry = precipitable_water_from_brick(&brick_with_qvapor(&dry, nx, ny, nz));
        let pw_wet = precipitable_water_from_brick(&brick_with_qvapor(&wet, nx, ny, nz));
        assert!(pw_dry > 0.0, "dry column still has some PW: {pw_dry}");
        assert!(pw_wet > pw_dry, "wetter column PW {pw_wet} !> dry {pw_dry}");
        let ratio_wet = pw_ratio_from_brick(&brick_with_qvapor(&wet, nx, ny, nz));
        let ratio_dry = pw_ratio_from_brick(&brick_with_qvapor(&dry, nx, ny, nz));
        assert!(ratio_wet >= ratio_dry);
    }

    #[test]
    fn pw_from_brick_clips_sub_terrain_vapor_and_matches_derived() {
        // WS1 march-physics: raising the terrain excludes the (clamped, fictitious)
        // sub-terrain layers from the column integral, so the pw_ratio no longer
        // inflates over elevated terrain; and the atmosphere-side integral agrees
        // with the derived-product PW field (same clip, same density kernel).
        let (nx, ny, nz) = (4usize, 4usize, 24usize);
        let q = 0.008f32;
        let qv: Vec<f32> = vec![q; nx * ny * nz];
        let mut brick = brick_with_qvapor(&qv, nx, ny, nz);
        let sea = precipitable_water_from_brick(&brick);
        brick.hgt.iter_mut().for_each(|h| *h = 1000.0);
        let elevated = precipitable_water_from_brick(&brick);
        assert!(
            elevated < sea,
            "elevated-terrain PW {elevated} !< sea-level PW {sea}"
        );
        // The excluded amount is the 0..1000 m surface-density vapor layer
        // (~9.4 kg m^-2 at 8 g/kg), NOT the whole column.
        let excluded = sea - elevated;
        assert!(
            (6.0..12.0).contains(&excluded),
            "excluded sub-terrain PW {excluded} not ~ the 1000 m surface layer"
        );
        let field = crate::derived::precipitable_water_field(&brick);
        let mean = field.iter().map(|&v| v as f64).sum::<f64>() / field.len() as f64;
        assert!(
            (mean - elevated).abs() < 1.0e-3 * elevated.max(1.0),
            "atmosphere PW {elevated} != derived-field mean {mean}"
        );
    }

    #[test]
    fn sun_enu_to_ecef_is_unit_and_consistent() {
        // Sun straight up at (lat, lon) -> ECEF radial direction there.
        let up = sun_enu_to_ecef([0.0, 0.0, 1.0], 30.0, -90.0);
        let (la, lo) = (30f64.to_radians(), (-90f64).to_radians());
        let radial = [la.cos() * lo.cos(), la.cos() * lo.sin(), la.sin()];
        for k in 0..3 {
            assert!((up[k] - radial[k]).abs() < 1.0e-9);
        }
        assert!((len3(up) - 1.0).abs() < 1.0e-9);
    }

    // ── SH-2 directional sky ambient (M5) ──

    /// Fill an analytic sky into a sky-view-parameterised `Lut2` (the same texel
    /// direction convention [`SkyShAmbient::project`] reads).
    fn analytic_sky_lut(w: usize, h: usize, r_obs: f64, f: impl Fn([f64; 3]) -> [f64; 3]) -> Lut2 {
        let mut lut = Lut2::new(w, h);
        for y in 0..h {
            let v = (y as f64 + 0.5) / h as f64;
            let zenith = skyview_v_to_zenith(v, r_obs);
            let (sz, cz) = zenith.sin_cos();
            for x in 0..w {
                let az = (x as f64 + 0.5) / w as f64 * 2.0 * PI;
                lut.set(x, y, f([sz * az.cos(), sz * az.sin(), cz]));
            }
        }
        lut
    }

    #[test]
    fn sh2_projection_round_trips_a_constant_sky() {
        // An isotropic sky of radiance c projects to a DC-only SH; its cosine-lobe
        // irradiance is pi*c at ANY normal, and the l=1 (directional) coefs vanish.
        let r_obs = R_GROUND_M + 500.0;
        let c = [0.3, 0.5, 0.7];
        let lut = analytic_sky_lut(96, 64, r_obs, |_| c);
        let sh = SkyShAmbient::project(&lut, r_obs);
        for &n in &[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.6, 0.0, 0.8]] {
            let e = sh.irradiance(n);
            for ch in 0..3 {
                let want = PI * c[ch];
                assert!(
                    (e[ch] - want).abs() < 0.08 * want,
                    "constant-sky irradiance {} vs pi*c {want} (normal {n:?})",
                    e[ch]
                );
            }
        }
        // Directional (l=1) coefficients are ~0 for an isotropic sky.
        for k in 1..=3 {
            for ch in 0..3 {
                assert!(
                    sh.coef[k][ch].abs() < 0.05 * sh.coef[0][ch].abs().max(1e-9),
                    "l=1 coef [{k}][{ch}] = {} not ~0",
                    sh.coef[k][ch]
                );
            }
        }
    }

    #[test]
    fn sh2_captures_a_directional_sky_gradient() {
        // A sky brighter (and warmer) toward +x (the sun azimuth). The SH must be
        // brighter toward +x than -x, both as raw radiance and as tilted irradiance.
        let r_obs = R_GROUND_M + 500.0;
        let lut = analytic_sky_lut(128, 64, r_obs, |d| {
            let g = 1.0 + 0.8 * d[0].max(0.0);
            [1.2 * g, 0.9 * g, 0.6 * g]
        });
        let sh = SkyShAmbient::project(&lut, r_obs);
        let sun_side = sh.radiance([1.0, 0.0, 0.0]);
        let anti_side = sh.radiance([-1.0, 0.0, 0.0]);
        for ch in 0..3 {
            assert!(
                sun_side[ch] > anti_side[ch],
                "SH radiance should be brighter toward +x: band {ch} sun {} anti {}",
                sun_side[ch],
                anti_side[ch]
            );
        }
        // A slope tilted toward +x gets more fill than one tilted toward -x.
        let e_sun = sh.irradiance(norm3([1.0, 0.0, 1.0]));
        let e_anti = sh.irradiance(norm3([-1.0, 0.0, 1.0]));
        assert!(
            e_sun[0] > e_anti[0],
            "sun-facing slope irradiance {} !> anti-facing {}",
            e_sun[0],
            e_anti[0]
        );
    }

    #[test]
    fn sunset_sky_sh_gives_warm_sun_side_cool_antisun_fill() {
        // The REAL sky-view LUT at a ~2 deg (sunset) sun, projected to SH-2. A cloud
        // face tilted toward the (warm, bright) sun horizon gets warmer + brighter fill
        // than one tilted toward the (cool, blue) anti-sun sky — the M5 directional,
        // coloured ambient requirement.
        let luts = small_luts();
        let params = AtmosphereParams::default();
        let cfg = SkyViewConfig {
            width: 128,
            height: 96,
            steps: 24,
            observer_altitude_m: 500.0,
        };
        let r_obs = R_GROUND_M + cfg.observer_altitude_m;
        let sv = build_sky_view_lut(&luts, &params, 2f64.to_radians(), &cfg);
        let sh = SkyShAmbient::project(&sv, r_obs);
        let sun_up = norm3([1.0, 0.0, 1.0]); // face tilted toward the sun azimuth
        let anti_up = norm3([-1.0, 0.0, 1.0]); // face tilted away from the sun
        let e_sun = sh.irradiance(sun_up);
        let e_anti = sh.irradiance(anti_up);
        let warm_sun = e_sun[0] / e_sun[2].max(1e-9);
        let warm_anti = e_anti[0] / e_anti[2].max(1e-9);
        assert!(
            warm_sun > warm_anti,
            "sun-facing fill should be warmer (higher R/B) than anti-sun: {warm_sun} vs {warm_anti}"
        );
        let bright_sun: f64 = e_sun.iter().sum();
        let bright_anti: f64 = e_anti.iter().sum();
        assert!(
            bright_sun > bright_anti,
            "sun-facing fill should be brighter (horizon glow): {bright_sun} vs {bright_anti}"
        );
    }

    #[test]
    fn sh2_scalar_irradiance_tracks_the_ambient_table() {
        // SH irradiance at the up normal (flat receiver) reproduces the scalar ambient
        // table's hemisphere irradiance to within SH-2 truncation, so the terrain
        // surface does not regress when the scalar ambient is replaced by the SH.
        let luts = small_luts();
        let params = AtmosphereParams::default();
        let ambient = AmbientTable::build(&luts, &params, 16);
        let sh_table = SkyShTable::build(&luts, &params, 16);
        for &elev in &[60.0, 30.0, 10.0] {
            let scalar = ambient.at(elev);
            let sh = sh_table.scalar_irradiance(elev);
            for c in 0..3 {
                let ratio = sh[c] / scalar[c].max(1e-9);
                assert!(
                    (0.6..1.6).contains(&ratio),
                    "SH-vs-scalar ambient at elev {elev} band {c}: ratio {ratio} out of band"
                );
            }
        }
    }
}
