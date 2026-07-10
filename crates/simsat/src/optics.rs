//! Hydrometeor optics constants and CPU reference kernels (design doc section 4).
//!
//! This is the single reviewed table of physical constants the ingest and (later)
//! the render kernels consume. Every constant carries its derivation and units in
//! a comment — that is the project honesty standard (design section 6): no tuned
//! magic numbers hide here. The extinction, air-density, and temperature kernels
//! are the CPU reference twins the shader kernels will be validated against.
//!
//! Band-averaged gray optics per hydrometeor class (visible band); NOT a
//! line-by-line radiative transfer model (design "non-goals").

/// Dry-air gas constant `R_d` (J kg^-1 K^-1). WRF `module_model_constants::r_d`.
pub const R_D: f64 = 287.0;

/// Specific heat of dry air at constant pressure `c_p` (J kg^-1 K^-1).
/// WRF defines `cp = 7*r_d/2`, so this stays exactly consistent with `R_D`.
pub const CP: f64 = 7.0 * R_D / 2.0;

/// Poisson exponent `kappa = R_d / c_p` (dimensionless), = 2/7 for an ideal
/// diatomic atmosphere. Used in the exner conversion of potential temperature.
pub const KAPPA: f64 = R_D / CP;

/// Reference pressure `p0` for potential temperature (Pa). WRF `p1000mb = 1e5`.
pub const P0: f64 = 1.0e5;

/// WRF potential-temperature offset (K): stored `T` is `theta - 300`.
pub const THETA_BASE: f64 = 300.0;

/// Gravitational acceleration `g0` (m s^-2). WRF `module_model_constants::g`.
/// Used to turn geopotential `(PH+PHB)` into geometric height `z = (PH+PHB)/g0`.
pub const G0: f64 = 9.81;

/// Bulk density of liquid water `rho_w` (kg m^-3). Appears in the geometric-optics
/// extinction denominator for every class (see `extinction_coefficient`).
pub const RHO_W: f64 = 1000.0;

/// Spherical earth radius WRF map projections use (m). Mirrored in `frame.rs`.
pub const EARTH_RADIUS_M: f64 = 6_370_000.0;

/// A single hydrometeor class with its band-averaged effective radius.
///
/// Effective radii (design section 4 + the SSB v3 snow-optics fix): cloud liquid
/// 10 um, cloud ice 40 um, snow 150 um (its OWN optics — no longer cloud-ice
/// optics), rain/graupel 1 mm. Larger particles -> smaller extinction per unit
/// mass, so precipitation-sized species read as a translucent veil rather than
/// cauliflower. Only the PRODUCT `rho_particle * r_eff` enters the
/// geometric-optics extinction (see [`extinction_coefficient`]), and the code
/// convention fixes `rho_particle = rho_w`, so every class radius here is the
/// rho_w-NORMALIZED effective radius `(rho_particle / rho_w) * r_particle`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HydrometeorClass {
    /// Cloud liquid water (WRF `QCLOUD`). r_e = 10 um (the canonical continental
    /// droplet value; mass extinction 3/(2 rho_w r_e) = 150 m^2 kg^-1).
    CloudLiquid,
    /// Cloud ice (WRF `QICE`): SMALL pristine crystals. r_e = 40 um rho_w-normalized
    /// (~44 um at rho_ice = 917 — the large end of the standard 25-40+ um cirrus
    /// range; mass extinction 37.5 m^2 kg^-1). Verified against the same
    /// geometric-optics framework as the v3 snow fix: `3/(2 rho_i r_e)` at 30-44 um
    /// of solid ice spans 41-55 m^2 kg^-1, so the M0 value stands.
    Ice,
    /// Snow (WRF `QSNOW`): precipitation-sized AGGREGATES (D ~ 0.5-3 mm at bulk
    /// densities ~50-200 kg m^-3), NOT small crystals. rho_w-normalized
    /// r_e = 150 um (equivalently ~1.5 mm aggregates near 100 kg m^-3 bulk
    /// density) -> visible mass extinction 3/(2 rho_w r_e) = 10 m^2 kg^-1, the
    /// honest mass-weighted aggregate value: falling-snow visibility studies give
    /// ~10-15 m^2 kg^-1 (Rasmussen et al. 1999, J. Appl. Meteor.), aggregate
    /// area-mass power laws (Mitchell 1996, JAS) give ~20-30 for 1-2 mm
    /// aggregates, and CRTM/RTTOV-style forward operators treat snow as its own
    /// large-particle class (effective sizes of hundreds of um), well below
    /// small-ice extinction per unit mass. Before SSB v3, snow shared cloud-ice
    /// optics (37.5 m^2 kg^-1) inside `ext_ice`, inflating snow's visible
    /// extinction 3.75x — anvil plates and stratiform shields rendered
    /// blindingly thick (the "clouds too thick" defect). Snow now enters
    /// `ext_precip` at this beta (see `ingest.rs`).
    Snow,
    /// Rain (WRF `QRAIN`). r_e = 1 mm (raindrops 0.5-2 mm; mass extinction
    /// 1.5 m^2 kg^-1).
    Rain,
    /// Graupel (WRF `QGRAUP`). r_e = 1 mm shared with rain (rho_w-normalized
    /// ~0.5-0.7 mm would be exact for 2-3 mm graupel at rho ~ 450 kg m^-3; the
    /// shared value is a documented, bounded simplification).
    Graupel,
}

impl HydrometeorClass {
    /// Effective particle radius `r_e` (m) for this class (rho_w-normalized — see
    /// the enum doc; only the product `rho_w * r_e` enters the optics).
    pub const fn effective_radius_m(self) -> f64 {
        match self {
            Self::CloudLiquid => 10.0e-6,
            Self::Ice => 40.0e-6,
            Self::Snow => 150.0e-6,
            Self::Rain | Self::Graupel => 1.0e-3,
        }
    }
}

/// Air density from pressure and temperature via the ideal gas law (kg m^-3).
///
/// `rho = p / (R_d * T)`. This uses the DRY gas constant `R_d`; the moist-air
/// correction (virtual temperature `T_v = T*(1 + 0.61*q_v)`, which would lower
/// density by up to ~2% in a saturated tropical column) is intentionally omitted
/// in M0 — it is a small, documented simplification, and the extinction it feeds
/// is itself a gray-optics approximation. `air_density_moist` is provided for
/// callers that want the correction.
#[inline]
pub fn air_density(pressure_pa: f64, temperature_k: f64) -> f64 {
    pressure_pa / (R_D * temperature_k)
}

/// Air density using virtual temperature from water-vapor mixing ratio (kg m^-3).
/// `rho = p / (R_d * T * (1 + 0.61*q_v))`. Documented alternative to `air_density`.
#[inline]
pub fn air_density_moist(pressure_pa: f64, temperature_k: f64, qvapor: f64) -> f64 {
    pressure_pa / (R_D * temperature_k * (1.0 + 0.61 * qvapor))
}

/// Absolute temperature (K) from WRF perturbation potential temperature and pressure.
///
/// `T = (theta' + 300) * (p / p0)^kappa` with `theta'` the stored `T` field,
/// `p = P + PB` the full pressure (Pa). This is the exact exner conversion WRF's
/// own diagnostics use.
#[inline]
pub fn temperature_from_theta(theta_perturbation: f64, pressure_pa: f64) -> f64 {
    (theta_perturbation + THETA_BASE) * (pressure_pa / P0).powf(KAPPA)
}

/// Geometric-optics extinction coefficient (m^-1) for one hydrometeor class.
///
/// `beta_ext = (3/2) * rho_air * q / (rho_w * r_e)`.
///
/// Derivation: for a population of spheres of radius `r_e` and bulk density
/// `rho_w`, the geometric-optics extinction (Qext -> 2, but the mass-weighted
/// cross section reduces to the 3/2 factor for the standard effective-radius
/// definition) is `beta = 3 * rho_air * q / (4 * rho_w * r_e) * Qext` with
/// `Qext = 2`, giving the `3/2` prefactor used here. `q` is the class mixing
/// ratio (kg kg^-1), `rho_air` the ambient air density (kg m^-3). Using `rho_w`
/// as the particle bulk density for ice/snow/graupel too is a documented
/// gray-optics simplification (their true bulk density is lower, which would
/// raise `beta` — but the effective radius already lumps the phase optics).
#[inline]
pub fn extinction_coefficient(rho_air: f64, q: f64, effective_radius_m: f64) -> f64 {
    if q <= 0.0 || rho_air <= 0.0 || effective_radius_m <= 0.0 {
        return 0.0;
    }
    1.5 * rho_air * q / (RHO_W * effective_radius_m)
}

/// Convenience: extinction for a named class.
#[inline]
pub fn class_extinction(class: HydrometeorClass, rho_air: f64, q: f64) -> f64 {
    extinction_coefficient(rho_air, q, class.effective_radius_m())
}

// ── IR (10.3 um, ABI band 13) thermal optics (design section 7, M6) ───────────
//
// The visible constants above give the geometric-optics EXTINCTION the brick
// stores. The synthetic-IR pass (`ir.rs`) needs the 10.3 um ABSORPTION instead,
// plus the emitted gray-body radiance. Both live here next to the visible optics
// so the whole optics table stays one reviewed file (the honesty standard,
// design section 6). None of this is line-by-line radiative transfer (design
// "non-goals"): band-averaged gray absorption/emission per hydrometeor class.
//
// ABSORPTION MODEL. In the 10.3 um window each hydrometeor class is treated as a
// gray absorber/emitter with negligible scattering (single-scatter albedo ~ 0 at
// 10.3 um for cloud particles — window-band scattering is small vs absorption, so
// we drop it: a documented simplification). The absorption coefficient is
//   beta_abs = kappa_mass * M
// with `kappa_mass` (m^2 g^-1) the band-averaged mass-absorption coefficient and
// `M` (g m^-3) the class mass concentration. The brick carries only the VISIBLE
// extinction, so `M` is recovered by inverting the geometric-optics relation
// `beta_vis = (3/2) M_kg / (rho_w r_e)` (see `extinction_coefficient`):
//   M_kg = (2/3) * rho_w * r_e * beta_vis         [kg m^-3]
//   M    = 1000 * M_kg                            [g  m^-3]
// closing the loop from the one channel the brick stores.
//
// CONSTANTS (band-averaged GRAY values at 10.3 um; NOT line-by-line). Liquid
// ~ 0.15 m^2/g and ice ~ 0.07 m^2/g are the standard longwave-window order for
// cloud liquid/ice mass absorption (liquid: Hu & Stamnes 1993; ice: Fu & Liou
// 1993 / Yang et al.; the exact value is a tuned band-average, not a fit). They
// are TUNED so an optically thick anvil is IR-opaque (tau_ir >> 1) and its
// brightness temperature equals its cloud-top temperature (the M6 proof standard,
// design section 10). With these values the ratio beta_abs/beta_vis is ~1.0 for
// liquid and ~1.87 for ice, so an anvil with a visible optical depth of even a
// few units is IR-opaque (BT = cloud-top T), while thin cirrus (tau_vis << 1)
// stays semi-transparent (BT between the cloud top and the surface) — both
// unit-tested. The large-particle species (rain/graupel at r_e = 1 mm, and — since
// SSB v3 — snow aggregates at r_e = 150 um, all in the `ext_precip` channel) get a
// much smaller mass absorption (large particles absorb little per unit mass): the
// geometric-optics `3 Q_abs / (4 rho_w r_e)` with Q_abs ~ 0.93 (millimetric
// ice/water particles are opaque internal absorbers at 10.3 um — the bulk e-folding
// depth is a few um). Per unit VISIBLE extinction that is beta_abs/beta_vis
// ~ 0.47 for every large species, because Q_abs/Q_ext is SIZE-INDEPENDENT in the
// geometric regime — which is why ONE per-channel recovery can carry the mixed
// rain + graupel + snow channel. The shipped channel recovery uses the SNOW
// coefficient (the channel's cold-relevant species) INCLUDING the documented
// small-ice-spectrum factor [`IR_SNOW_SMALL_ICE_FACTOR`], i.e. ratio ~ 0.93;
// for the rain/graupel share this over-absorbs 2x, which is BT-invisible where
// rain lives (low, warm, T ~ T_sfc: what it absorbs it re-emits at nearly the
// same temperature). A snow-dominated anvil plate is IR-opaque by sheer mass
// (tau_vis ~ 20 -> tau_ir ~ 19 with the factor, BT = cloud-top T).

/// IR (10.3 um) mass-absorption coefficient for cloud LIQUID (m^2 g^-1).
pub const IR_MASS_ABS_LIQUID_M2_G: f64 = 0.15;
/// IR (10.3 um) mass-absorption coefficient for small cloud ICE (m^2 g^-1).
pub const IR_MASS_ABS_ICE_M2_G: f64 = 0.07;
/// The PURE geometric-optics 10.3 um mass absorption of 150 um (rho_w-normalized)
/// snow aggregates (m^2 g^-1): `3 Q_abs / (4 rho_w r_e)` with Q_abs ~= 0.93,
/// i.e. an IR-absorption / visible-extinction ratio `Q_abs / Q_ext = 0.467`,
/// the size-independent large-particle value rain/graupel share.
pub const IR_MASS_ABS_SNOW_GEOMETRIC_M2_G: f64 = 4.667e-3;

/// Unresolved-size-spectrum compensation on snow's 10.3 um absorption
/// (dimensionless, applied on top of the geometric aggregate value). WRF QSNOW —
/// especially on coarse grids — physically includes small cloud-top ice crystals
/// whose 10.3 um absorption per unit VISIBLE extinction exceeds the pure
/// 150-um-aggregate geometric limit; band-averaged fast-RT operators carry
/// exactly this kind of gray compensation rather than resolving the spectrum.
/// Factor 2.0 was chosen in an orchestrator-reviewed A/B on the Michael
/// hurricane IR (geometric vs 2x vs 3x): pure geometric rendered the CDO canopy
/// mid-grey (median BT 255 K) where a real major hurricane reads broadly cold;
/// the compensation restores a cold canopy while keeping the eye / banding and
/// the honestly graded anvil edges the snow-optics fix bought. VISIBLE optics
/// are untouched by construction (this constant exists only in the IR/WV
/// recovery).
///
/// SCIENCE-REVIEW FRAMING (2026-07-09, do not cite this factor as physics
/// headroom): a fixed-size aggregate cannot exceed `Q_abs/Q_ext ~ 0.5`, so 2x
/// is NOT within single-size geometric physics. It is equivalent to a ~7%
/// small-ice mass fraction inside QSNOW, or to a Mitchell (1996) area-mass A/m
/// in the 10-15 m^2/kg band — both plausible for cold anvil canopies. It is a
/// documented STOPGAP for the real fix: Field et al. (2005) temperature-
/// dependent snow size (the UPP `EFFR('S')` moment math), which makes cold
/// snow small in BOTH bands simultaneously and retires this factor to 1.0
/// (queued as the next brick-format rev; see notes/next-level-wave.md).
pub const IR_SNOW_SMALL_ICE_FACTOR: f64 = 2.0;

/// IR (10.3 um) mass-absorption coefficient for SNOW (m^2 g^-1): the geometric
/// aggregate value times the documented small-ice-spectrum factor. This is the
/// coefficient `ir.rs` applies to the ENTIRE `ext_precip` channel (see
/// `cloud_ir_absorption` for why the channel follows its cold-relevant species).
pub const IR_MASS_ABS_SNOW_M2_G: f64 = IR_SNOW_SMALL_ICE_FACTOR * IR_MASS_ABS_SNOW_GEOMETRIC_M2_G;
/// IR (10.3 um) mass-absorption coefficient for PRECIP (rain/graupel) (m^2 g^-1).
/// Large particles (r_e = 1 mm) absorb little per unit mass at 10.3 um; this is
/// the geometric-optics order `~ 3 Q_abs / (4 rho_w r_e)` for a large absorbing
/// drop (Q_abs ~ 0.9), a documented gray value.
pub const IR_MASS_ABS_PRECIP_M2_G: f64 = 7.0e-4;

/// Longwave surface emissivity at 10.3 um (dimensionless). Land/water/ice are all
/// ~ 0.97-0.99 in the window; a single gray value is used (a documented
/// simplification — a LANDMASK/IVGTYP-dependent emissivity is a later refinement).
pub const IR_SURFACE_EMISSIVITY: f64 = 0.99;

/// ABI band-13 centre wavelength (m): 10.3 um.
pub const IR_BAND13_WAVELENGTH_M: f64 = 10.3e-6;

// ── Water-vapor bands (ABI 8 / 9 / 10 = 6.2 / 6.9 / 7.3 um; owner decision 6) ──
//
// The water-vapor bands are the SAME top-down Planck-emission march as the 10.3 um
// window (ir.rs), but the DOMINANT absorber/emitter is water vapor (from the brick
// QVAPOR channel) instead of the surface. In the 6-7 um nu2 water-vapor band the
// atmosphere is opaque to the surface: emission reaches the satellite only from the
// troposphere where the WV optical depth from space first reaches ~1 (the WEIGHTING
// FUNCTION). Making the band more strongly absorbing raises that level (colder BT);
// making it weaker lowers it (warmer BT). The three ABI WV bands sample three
// heights: 6.2 um (strong -> UPPER troposphere, cold), 6.9 um (mid), 7.3 um (weak
// -> LOWER/mid troposphere, warm).
//
// MODEL (honest, band-averaged GRAY water-vapor; NOT line-by-line — design
// "non-goals"). The WV absorption per voxel is `beta_wv = kappa_wv * rho_wv` with
// `rho_wv = rho_air(z) * qvapor` (kg m^-3) exactly as the window continuum
// (`ir_wv_continuum_absorption`), only with a per-band mass-absorption `kappa_wv`
// (m^2 kg^-1) that is ~1e2-1e3 x the weak window-continuum value — that is the whole
// difference from band 13. Cloud/ice/precip stay opaque in the WV bands too (the
// same `ir_absorption_from_ext` cloud absorption is added), so cloud tops read cold.
//
// WEIGHTING-FUNCTION TUNING. With an exponential moist column (surface mixing ratio
// q0 ~ 12 g/kg, combined vapor+air-density scale height H_e ~ 1.9 km), the column WV
// optical depth is `tau_col ~ kappa_wv * rho_air(0) * q0 * H_e ~ kappa_wv * 28
// kg m^-2`, and the weighting function `W(z) = beta_wv(z) * exp(-tau_from_top(z))`
// peaks where `tau_from_top ~ 1`, i.e. `z_peak ~ H_e * ln(tau_col)`. The three gray
// coefficients below are TUNED so those peaks sit in the upper / mid / lower
// troposphere for a typical moist column, giving band-averaged BTs in the classic
// WV range (~220-260 K, colder than the 10.3 um window). They are documented gray
// band-averages, not a fit to a specific sounding; the RELATIVE ordering
// (6.2 > 6.9 > 7.3 in absorption -> 6.2 < 6.9 < 7.3 in BT) is the physics under test.
//
// TERRAIN-CLIP INTERACTION (WS1 march-physics pass). The tuning column above is a
// SEA-LEVEL column, and `IrVolume::from_brick` now clips the ingest's clamped
// sub-terrain vapor out of the marched field, so these coefficients are unchanged by
// the clip — the sea-level weighting-function rationale never included the
// fictitious below-ground air. Over ELEVATED terrain the clip warms the WV BTs
// (most for band 10, whose weighting function sits lowest and previously absorbed
// ~tau 2-3 of nonexistent boundary-layer vapor below a ~1500 m surface; band 13's
// weak continuum shifts < 1 K). That warming is the bug fix, not a retune signal.

/// ABI band-8 (upper-level WV, 6.2 um) centre wavelength (m).
pub const WV_BAND8_WAVELENGTH_M: f64 = 6.2e-6;
/// ABI band-9 (mid-level WV, 6.9 um) centre wavelength (m).
pub const WV_BAND9_WAVELENGTH_M: f64 = 6.9e-6;
/// ABI band-10 (lower-level WV, 7.3 um) centre wavelength (m).
pub const WV_BAND10_WAVELENGTH_M: f64 = 7.3e-6;

/// Band-averaged gray WV mass-absorption at 6.2 um (m^2 kg^-1). STRONGLY absorbing:
/// the weighting function peaks in the UPPER troposphere (~8-9 km) so band 8 reads a
/// cold BT and shows upper-level moisture. Tuned (see the weighting-function note).
pub const WV_MASS_ABS_BAND8_M2_KG: f64 = 3.0;
/// Band-averaged gray WV mass-absorption at 6.9 um (m^2 kg^-1). MID-level: the
/// weighting function peaks around the mid troposphere (~5-6 km).
pub const WV_MASS_ABS_BAND9_M2_KG: f64 = 0.7;
/// Band-averaged gray WV mass-absorption at 7.3 um (m^2 kg^-1). WEAKLY absorbing:
/// the weighting function peaks in the LOWER/mid troposphere (~3 km) so band 10 reads
/// the warmest of the three WV bands (sees deeper, closer to the moist boundary layer).
pub const WV_MASS_ABS_BAND10_M2_KG: f64 = 0.18;

/// First radiation constant for SPECTRAL radiance in wavelength, `2 h c^2`
/// (W m^2 sr^-1), so `B_lambda` comes out in W m^-3 sr^-1. CODATA.
pub const PLANCK_C1L: f64 = 1.191_042_972e-16;
/// Second radiation constant `h c / k_B` (m K). CODATA.
pub const PLANCK_C2: f64 = 1.438_776_877e-2;

/// Sea-level standard air density (kg m^-3) — the reference for the exponential
/// standard-atmosphere profile the IR water-vapor continuum weight uses.
pub const RHO_AIR_SEA_LEVEL: f64 = 1.225;
/// Air-density scale height (m) for that exponential profile (the ~8.5 km density
/// scale height of the lower atmosphere).
pub const AIR_DENSITY_SCALE_HEIGHT_M: f64 = 8500.0;

/// IR water-vapor CONTINUUM mass-absorption coefficient (m^2 kg^-1) at 10.3 um.
/// A weak, documented gray approximation of the self+foreign continuum in the
/// window: the real continuum is ~ e-folding in water-vapor partial pressure and
/// spectrally structured (MT_CKD); we use one linear gray value tuned so the
/// column optical depth `tau_wv = kappa_wv * PW` is small (a 25 kg m^-2 column
/// gives tau ~ 0.125, i.e. a few-K window BT depression). It exists so a very
/// moist clear column reads a hair cooler than the skin temperature, not to model
/// the WV band (that is a future 6.2 um addition, owner decision 6).
pub const IR_WV_CONTINUUM_MASS_ABS_M2_KG: f64 = 5.0e-3;

impl HydrometeorClass {
    /// Band-averaged 10.3 um mass-absorption coefficient (m^2 g^-1) for this class.
    pub const fn ir_mass_absorption_m2_g(self) -> f64 {
        match self {
            Self::CloudLiquid => IR_MASS_ABS_LIQUID_M2_G,
            Self::Ice => IR_MASS_ABS_ICE_M2_G,
            Self::Snow => IR_MASS_ABS_SNOW_M2_G,
            Self::Rain | Self::Graupel => IR_MASS_ABS_PRECIP_M2_G,
        }
    }
}

/// Hydrometeor mass concentration (g m^-3) recovered from the brick's stored
/// VISIBLE geometric-optics extinction (m^-1): the inverse of
/// `beta_vis = (3/2) M_kg / (rho_w r_e)`, i.e. `M = 1000 * (2/3) rho_w r_e beta_vis`.
#[inline]
pub fn mass_concentration_g_m3(ext_vis_per_m: f64, effective_radius_m: f64) -> f64 {
    if ext_vis_per_m <= 0.0 || effective_radius_m <= 0.0 {
        return 0.0;
    }
    1000.0 * (2.0 / 3.0) * RHO_W * effective_radius_m * ext_vis_per_m
}

/// IR (10.3 um) absorption coefficient (m^-1) of one hydrometeor class, derived
/// from its stored visible extinction: `beta_abs = kappa_mass * M`, with `M`
/// recovered by [`mass_concentration_g_m3`]. This is the one function `ir.rs`
/// calls per class per voxel.
#[inline]
pub fn ir_absorption_from_ext(class: HydrometeorClass, ext_vis_per_m: f64) -> f64 {
    class.ir_mass_absorption_m2_g()
        * mass_concentration_g_m3(ext_vis_per_m, class.effective_radius_m())
}

/// Standard-atmosphere air density (kg m^-3) at MSL height `z_m` (an exponential
/// `rho0 exp(-z/H)` — the density weight for the IR water-vapor continuum, since
/// the brick carries no pressure). Documented approximation.
#[inline]
pub fn standard_air_density_kg_m3(z_m: f64) -> f64 {
    RHO_AIR_SEA_LEVEL * (-z_m.max(0.0) / AIR_DENSITY_SCALE_HEIGHT_M).exp()
}

/// Water-vapor absorption coefficient (m^-1) at a voxel with water-vapor mixing
/// ratio `qvapor` (kg kg^-1) at MSL height `z_m` for a mass-absorption
/// `kappa_wv_m2_kg` (m^2 kg^-1): `beta = kappa_wv * rho_wv`,
/// `rho_wv = rho_air(z) * qvapor`. This is the ONE gray water-vapor kernel shared by
/// the 10.3 um window (weak continuum kappa) and the 6.2 / 6.9 / 7.3 um water-vapor
/// bands (strong per-band kappa); only the coefficient differs (design section 7 +
/// the WV addendum). Zero for non-positive vapor / coefficient.
#[inline]
pub fn wv_absorption(qvapor_kg_kg: f64, z_m: f64, kappa_wv_m2_kg: f64) -> f64 {
    if qvapor_kg_kg <= 0.0 || kappa_wv_m2_kg <= 0.0 {
        return 0.0;
    }
    let rho_wv = standard_air_density_kg_m3(z_m) * qvapor_kg_kg; // kg m^-3
    kappa_wv_m2_kg * rho_wv
}

/// IR water-vapor CONTINUUM absorption coefficient (m^-1): [`wv_absorption`] at the
/// weak window-continuum mass-absorption [`IR_WV_CONTINUUM_MASS_ABS_M2_KG`]. The thin
/// window-band wrapper (band 13); the WV bands call [`wv_absorption`] with their own
/// strong per-band coefficient.
#[inline]
pub fn ir_wv_continuum_absorption(qvapor_kg_kg: f64, z_m: f64) -> f64 {
    wv_absorption(qvapor_kg_kg, z_m, IR_WV_CONTINUUM_MASS_ABS_M2_KG)
}

/// Planck spectral radiance `B_lambda(T)` (W m^-3 sr^-1) at wavelength
/// `wavelength_m` and absolute temperature `temperature_k` (K).
/// `B = c1L / (lambda^5 (exp(c2/(lambda T)) - 1))`. Non-positive T or lambda -> 0.
#[inline]
pub fn planck_radiance(temperature_k: f64, wavelength_m: f64) -> f64 {
    if temperature_k <= 0.0 || wavelength_m <= 0.0 {
        return 0.0;
    }
    let l5 = wavelength_m.powi(5);
    let expo = PLANCK_C2 / (wavelength_m * temperature_k);
    let denom = l5 * (expo.exp() - 1.0);
    if denom <= 0.0 || !denom.is_finite() {
        return 0.0;
    }
    PLANCK_C1L / denom
}

/// Inverse Planck: the brightness temperature (K) of a spectral radiance
/// `radiance` (W m^-3 sr^-1) at wavelength `wavelength_m`. Exact inverse of
/// [`planck_radiance`]. Non-positive radiance/lambda -> 0.
#[inline]
pub fn inverse_planck(radiance: f64, wavelength_m: f64) -> f64 {
    if radiance <= 0.0 || wavelength_m <= 0.0 {
        return 0.0;
    }
    let l5 = wavelength_m.powi(5);
    PLANCK_C2 / (wavelength_m * (1.0 + PLANCK_C1L / (l5 * radiance)).ln())
}

// ── Cox-Munk water sun-glint + Fresnel (design section 5, M3) ─────────────────
//
// The remote-sensing-standard glint model (Cox & Munk 1954, "Measurement of the
// roughness of the sea surface from photographs of the sun's glitter", JOSA 44):
// wind ruffles the sea into wave facets whose slopes are ~Gaussian, and the sun
// glint is the specular reflection off the subset of facets oriented to send the
// sun toward the viewer. NOT a line-by-line ocean BRDF (design "non-goals"): a
// band-averaged gray Fresnel x the isotropic slope PDF.

/// Refractive index of sea water in the visible (dimensionless, ~1.33-1.34; 1.34 is
/// the standard glint value). Drives the Fresnel reflectance.
pub const WATER_REFRACTIVE_INDEX_VIS: f64 = 1.34;

/// Cox-Munk isotropic mean-square-slope intercept (dimensionless): the calm-sea
/// residual slope variance. `sigma^2 = C0 + C1 * W` (Cox & Munk 1954, eq. for the
/// isotropic/combined slope variance).
pub const COX_MUNK_SLOPE_C0: f64 = 0.003;
/// Cox-Munk isotropic mean-square-slope wind coefficient (per m/s at 10 m). The
/// published isotropic combined-slope relation `sigma^2 = 0.003 + 0.00512 * W`.
pub const COX_MUNK_SLOPE_C1: f64 = 0.00512;

/// The Cox-Munk isotropic mean-square slope `sigma^2 = 0.003 + 0.00512 * W` for a
/// 10 m wind speed `W` (m/s). Monotone increasing in `W` (a windier sea has a wider
/// slope distribution -> a broader, dimmer-peaked glitter pattern). Floored small to
/// keep the slope PDF finite. Anisotropic Cox-Munk (up/cross-wind variances) is a
/// documented later refinement; isotropic is design-acceptable for v1.
#[inline]
pub fn cox_munk_mean_square_slope(wind_speed_m_s: f64) -> f64 {
    (COX_MUNK_SLOPE_C0 + COX_MUNK_SLOPE_C1 * wind_speed_m_s.max(0.0)).max(1.0e-4)
}

/// Unpolarised Fresnel reflectance for light in air (`n1 = 1`) meeting a medium of
/// relative refractive index `n_rel` at incidence-cosine `cos_incidence` (the cosine
/// of the angle between the incident ray and the surface normal). The average of the
/// s- and p-polarised power reflectances. At normal incidence on water this is
/// `((1-n)/(1+n))^2 ~= 0.021`; it rises to 1 at grazing incidence.
#[inline]
pub fn fresnel_reflectance_unpolarized(cos_incidence: f64, n_rel: f64) -> f64 {
    if n_rel <= 0.0 {
        return 0.0;
    }
    let ci = cos_incidence.clamp(0.0, 1.0);
    let sin_t2 = (1.0 - ci * ci) / (n_rel * n_rel); // Snell: sin_t = sin_i / n_rel
    if sin_t2 >= 1.0 {
        return 1.0; // total internal reflection (only if n_rel < 1)
    }
    let ct = (1.0 - sin_t2).sqrt();
    let rs = ((ci - n_rel * ct) / (ci + n_rel * ct)).powi(2);
    let rp = ((n_rel * ci - ct) / (n_rel * ci + ct)).powi(2);
    (0.5 * (rs + rp)).clamp(0.0, 1.0)
}

/// The Cox-Munk sea-surface sun-glint REFLECTANCE FACTOR `rho = pi L / E_perp`
/// (dimensionless, the same reflectance quantity the render pipeline compares to),
/// for a sun direction `to_sun`, a surface->viewer direction `to_camera`, the local
/// surface `up`, and the mean-square slope `mss` (from [`cox_munk_mean_square_slope`]).
/// All directions are unit vectors in ANY consistent basis (the caller uses ECEF with
/// `up` = the local vertical at the water point).
///
/// `rho = pi * F(omega) * P(z_x, z_y) / (4 * mu_s * mu_v * cos^4 beta)` (Cox & Munk
/// 1954; the form used in 6S / ocean-colour glint corrections), where:
///   - the wave facet that specularly reflects the sun to the viewer has normal
///     `n_f = normalize(to_sun + to_camera)`; `beta` is its tilt from `up`
///     (`cos beta = n_f . up`) and `omega` the incidence on it (`cos omega =
///     to_sun . n_f`);
///   - `P = exp(-tan^2 beta / mss) / (pi * mss)` is the isotropic slope PDF (the
///     glitter's ANGULAR EXTENT — it widens with `mss`, i.e. with wind);
///   - `mu_s = to_sun . up`, `mu_v = to_camera . up` (the sun/view zenith cosines).
///
/// Zero when the sun is below the horizon or the view grazes (guards the divide).
pub fn cox_munk_glint_reflectance(
    to_sun: [f64; 3],
    to_camera: [f64; 3],
    up: [f64; 3],
    mss: f64,
) -> f64 {
    let s = norm3(to_sun);
    let v = norm3(to_camera);
    let u = norm3(up);
    let mu_s = dot3(s, u);
    let mu_v = dot3(v, u);
    if mu_s <= 1.0e-4 || mu_v <= 1.0e-4 {
        return 0.0;
    }
    let hf = [s[0] + v[0], s[1] + v[1], s[2] + v[2]];
    let hlen = (hf[0] * hf[0] + hf[1] * hf[1] + hf[2] * hf[2]).sqrt();
    if hlen <= 1.0e-9 {
        return 0.0;
    }
    let nf = [hf[0] / hlen, hf[1] / hlen, hf[2] / hlen];
    let cos_beta = dot3(nf, u).clamp(1.0e-4, 1.0);
    let cos_omega = dot3(s, nf).clamp(0.0, 1.0);
    let tan2 = (1.0 - cos_beta * cos_beta) / (cos_beta * cos_beta);
    let mss = mss.max(1.0e-4);
    let p = (-tan2 / mss).exp() / (std::f64::consts::PI * mss);
    let f = fresnel_reflectance_unpolarized(cos_omega, WATER_REFRACTIVE_INDEX_VIS);
    let rho = std::f64::consts::PI * f * p / (4.0 * mu_s * mu_v * cos_beta.powi(4));
    if rho.is_finite() { rho.max(0.0) } else { 0.0 }
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn norm3(a: [f64; 3]) -> [f64; 3] {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if l > 0.0 {
        [a[0] / l, a[1] / l, a[2] / l]
    } else {
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kappa_is_two_sevenths() {
        assert!((KAPPA - 2.0 / 7.0).abs() < 1.0e-12);
    }

    #[test]
    fn effective_radii_match_design_table() {
        assert_eq!(HydrometeorClass::CloudLiquid.effective_radius_m(), 10.0e-6);
        assert_eq!(HydrometeorClass::Ice.effective_radius_m(), 40.0e-6);
        // SSB v3 snow-optics fix: snow is a precipitation-sized aggregate with its
        // OWN rho_w-normalized effective radius, no longer sharing cloud-ice optics.
        assert_eq!(HydrometeorClass::Snow.effective_radius_m(), 150.0e-6);
        assert_eq!(HydrometeorClass::Rain.effective_radius_m(), 1.0e-3);
        assert_eq!(HydrometeorClass::Graupel.effective_radius_m(), 1.0e-3);
    }

    /// SSB v3: the per-species VISIBLE mass-extinction coefficients are locked —
    /// `k = 3/(2 rho_w r_e)`: liquid 150, ice 37.5, snow 10, rain/graupel
    /// 1.5 m^2 kg^-1 — with the physical per-unit-mass ordering
    /// liquid > ice > snow > rain (smaller particles extinguish more per gram).
    #[test]
    fn visible_mass_extinction_per_species_locked_and_ordered() {
        // class_extinction at rho_air = 1 kg/m^3, q = 1 g/kg gives k * 1e-3 (m^-1),
        // i.e. the mass-extinction coefficient scaled by the 1 g/m^3 concentration.
        let k = |class: HydrometeorClass| class_extinction(class, 1.0, 1.0e-3) * 1.0e3;
        let liquid = k(HydrometeorClass::CloudLiquid);
        let ice = k(HydrometeorClass::Ice);
        let snow = k(HydrometeorClass::Snow);
        let rain = k(HydrometeorClass::Rain);
        let graupel = k(HydrometeorClass::Graupel);
        assert!((liquid - 150.0).abs() < 1e-9, "liquid k={liquid}");
        assert!((ice - 37.5).abs() < 1e-9, "ice k={ice}");
        assert!((snow - 10.0).abs() < 1e-9, "snow k={snow}");
        assert!((rain - 1.5).abs() < 1e-9, "rain k={rain}");
        assert!((graupel - 1.5).abs() < 1e-9, "graupel k={graupel}");
        assert!(
            liquid > ice && ice > snow && snow > rain && rain > 0.0,
            "per-unit-mass ordering violated: {liquid} > {ice} > {snow} > {rain} > 0"
        );
        // The defect factor this fix removes: snow under the old shared-ice optics
        // carried exactly ice/snow = 3.75x too much visible extinction per gram.
        assert!(((ice / snow) - 3.75).abs() < 1e-9);
    }

    /// SSB v3 + the IR lever round: snow's 10.3 um absorption per unit VISIBLE
    /// extinction is the size-independent geometric large-particle ratio
    /// `Q_abs/Q_ext ~ 0.467` (== rain/graupel — the identity that lets ONE
    /// per-channel recovery carry the mixed rain+graupel+snow `ext_precip`
    /// channel) times the NAMED unresolved-size-spectrum compensation
    /// [`IR_SNOW_SMALL_ICE_FACTOR`]. Still well below small-ice's tuned 1.87 per
    /// unit extinction AND per unit mass, yet a thick snow plate goes IR-opaque
    /// by mass.
    #[test]
    fn snow_ir_absorption_matches_the_large_particle_ratio() {
        let ext = 6.67e-3; // a realistic snow-plate visible extinction (m^-1)
        let snow_ratio = ir_absorption_from_ext(HydrometeorClass::Snow, ext) / ext;
        let rain_ratio = ir_absorption_from_ext(HydrometeorClass::Rain, ext) / ext;
        let ice_ratio = ir_absorption_from_ext(HydrometeorClass::Ice, ext) / ext;
        // The constant IS the documented product: factor x geometric aggregate value
        // (factor x Q_abs/Q_ext x k_vis in per-unit-extinction terms).
        const {
            assert!(IR_SNOW_SMALL_ICE_FACTOR == 2.0);
            assert!(
                IR_MASS_ABS_SNOW_M2_G == IR_SNOW_SMALL_ICE_FACTOR * IR_MASS_ABS_SNOW_GEOMETRIC_M2_G
            );
        };
        assert!(
            (snow_ratio - IR_SNOW_SMALL_ICE_FACTOR * 0.4667).abs() < 2e-3,
            "snow ratio {snow_ratio}"
        );
        // The geometric identity with rain survives underneath the named factor.
        assert!(
            (snow_ratio - IR_SNOW_SMALL_ICE_FACTOR * rain_ratio).abs() < 2e-3,
            "snow {snow_ratio} != factor x rain {rain_ratio}: the channel identity"
        );
        assert!(snow_ratio < ice_ratio, "snow must absorb less per unit ext");
        // Per unit MASS: snow 9.33e-3 m^2/g still well below ice 0.07 m^2/g
        // (const-asserted, matching the WV coefficient-ordering guard pattern).
        const {
            assert!(IR_MASS_ABS_SNOW_M2_G < IR_MASS_ABS_ICE_M2_G / 5.0);
        };
        // IR-opacity by mass survives: a 2 kg/m^2 snow water path over a 3 km plate
        // (M = 0.667 g/m^3 -> tau_vis = 10 m^2/kg * 2 kg/m^2 = 20) has
        // tau_ir = 0.933 * 20 ~ 19 >> 1 -> transmittance ~ 0.
        let m_g_m3 = 0.667; // g/m^3
        let beta_vis = 10.0 * (m_g_m3 / 1000.0); // k * M_kg = 6.67e-3 m^-1
        let beta_ir = ir_absorption_from_ext(HydrometeorClass::Snow, beta_vis);
        let tau_ir = beta_ir * 3000.0;
        assert!(
            tau_ir > 8.0,
            "thick snow plate not IR-opaque: tau_ir {tau_ir}"
        );
    }

    #[test]
    fn temperature_from_theta_recovers_surface() {
        // At p = p0, exner factor = 1, so T = theta' + 300 exactly.
        let t = temperature_from_theta(-10.0, P0);
        assert!((t - 290.0).abs() < 1.0e-9);
        // A warm boundary layer parcel (theta'=15) lifted to 700 hPa cools.
        let t700 = temperature_from_theta(15.0, 70_000.0);
        assert!(t700 < 315.0 && t700 > 280.0, "t700={t700}");
    }

    #[test]
    fn air_density_is_near_std_atmosphere_at_surface() {
        // Sea-level standard: p=101325 Pa, T=288.15 K -> rho ~= 1.225 kg/m^3.
        let rho = air_density(101_325.0, 288.15);
        assert!((rho - 1.225).abs() < 0.01, "rho={rho}");
    }

    #[test]
    fn moist_air_is_less_dense_than_dry() {
        let dry = air_density(90_000.0, 295.0);
        let moist = air_density_moist(90_000.0, 295.0, 0.018);
        assert!(moist < dry);
        // 18 g/kg of vapor lowers density by roughly 1%.
        assert!((dry - moist) / dry < 0.02);
    }

    #[test]
    fn extinction_scales_and_class_ordering_is_physical() {
        let rho = 1.0;
        let q = 1.0e-3; // 1 g/kg
        let liquid = class_extinction(HydrometeorClass::CloudLiquid, rho, q);
        let ice = class_extinction(HydrometeorClass::Ice, rho, q);
        let rain = class_extinction(HydrometeorClass::Rain, rho, q);
        // Same mass, smaller particles => far larger extinction.
        assert!(liquid > ice);
        assert!(ice > rain);
        // Explicit value: (3/2)*1*1e-3/(1000*10e-6) = 0.15 m^-1 for cloud liquid.
        assert!((liquid - 0.15).abs() < 1.0e-9, "liquid={liquid}");
        // Zero / negative mass yields zero extinction.
        assert_eq!(
            class_extinction(HydrometeorClass::CloudLiquid, rho, 0.0),
            0.0
        );
        assert_eq!(
            class_extinction(HydrometeorClass::CloudLiquid, rho, -1.0),
            0.0
        );
    }

    #[test]
    fn planck_inverse_planck_round_trip() {
        // Radiance -> BT -> radiance and BT -> radiance -> BT must both be exact
        // inverses across the whole plausible BT range (design section 7 test 4).
        for &t in &[180.0, 210.0, 235.0, 273.15, 300.0, 320.0] {
            let l = planck_radiance(t, IR_BAND13_WAVELENGTH_M);
            assert!(l > 0.0 && l.is_finite(), "radiance {l} at {t} K");
            let bt = inverse_planck(l, IR_BAND13_WAVELENGTH_M);
            assert!((bt - t).abs() < 1.0e-4, "BT {bt} != T {t}");
        }
        // Planck is monotone increasing in temperature (a warmer body is brighter).
        let mut prev = 0.0;
        for &t in &[180.0, 220.0, 260.0, 300.0] {
            let l = planck_radiance(t, IR_BAND13_WAVELENGTH_M);
            assert!(l > prev, "Planck not monotone at {t} K");
            prev = l;
        }
        // Guards: non-positive inputs return 0 (no NaN/Inf).
        assert_eq!(planck_radiance(0.0, IR_BAND13_WAVELENGTH_M), 0.0);
        assert_eq!(planck_radiance(250.0, 0.0), 0.0);
        assert_eq!(inverse_planck(0.0, IR_BAND13_WAVELENGTH_M), 0.0);
        assert_eq!(inverse_planck(-1.0, IR_BAND13_WAVELENGTH_M), 0.0);
    }

    #[test]
    fn ir_absorption_ratios_make_anvils_opaque_and_precip_mild() {
        // The IR absorption / visible extinction ratios are the anvil-BT-to-cloud-
        // top-T tuning: ~1.0 liquid, ~1.87 ice, ~0.47 precip (design section 7).
        for &(class, want) in &[
            (HydrometeorClass::CloudLiquid, 1.0),
            (HydrometeorClass::Ice, 1.8667),
            (HydrometeorClass::Rain, 0.4667),
        ] {
            // Ratio is independent of the extinction magnitude (both are linear in it).
            let ext = 3.0e-2;
            let ratio = ir_absorption_from_ext(class, ext) / ext;
            assert!(
                (ratio - want).abs() < 0.02,
                "{class:?} ratio {ratio} != {want}"
            );
        }
        // Ice absorbs more per unit VISIBLE extinction than liquid (larger r_e =>
        // more mass behind the same visible cross-section), so an ice anvil goes
        // IR-opaque sooner than a liquid deck of the same visible optical depth.
        assert!(
            ir_absorption_from_ext(HydrometeorClass::Ice, 1.0e-2)
                > ir_absorption_from_ext(HydrometeorClass::CloudLiquid, 1.0e-2)
        );
        // Zero / negative extinction -> zero absorption.
        assert_eq!(ir_absorption_from_ext(HydrometeorClass::Ice, 0.0), 0.0);
        assert_eq!(ir_absorption_from_ext(HydrometeorClass::Ice, -1.0), 0.0);
        // A thick anvil (visible optical depth ~ 30 over a 3 km ice column) is
        // IR-opaque: tau_ir = ratio_ice * tau_vis ~ 56 >> 1 -> transmittance ~ 0.
        let beta_ice = ir_absorption_from_ext(HydrometeorClass::Ice, 1.0e-2); // m^-1
        let tau_ir = beta_ice * 3000.0;
        assert!(tau_ir > 10.0, "anvil not IR-opaque: tau_ir {tau_ir}");
    }

    #[test]
    fn wv_continuum_is_weak_and_altitude_decaying() {
        // Column optical depth tau = kappa_wv * PW should be small (weak, design
        // section 7). Integrate an exponential q profile against the density weight.
        let q_surface = 0.014f64; // 14 g/kg boundary-layer vapor
        let (mut tau, mut pw) = (0.0f64, 0.0f64);
        let dz = 50.0f64;
        let mut z = 0.0f64;
        while z < 12000.0 {
            let q = q_surface * (-z / 2500.0).exp(); // vapor scale height ~ 2.5 km
            tau += ir_wv_continuum_absorption(q, z) * dz;
            pw += standard_air_density_kg_m3(z) * q * dz;
            z += dz;
        }
        assert!(tau > 0.0 && tau < 0.5, "WV continuum not weak: tau {tau}");
        // tau = kappa_wv * PW to within the integration slop.
        assert!(
            (tau - IR_WV_CONTINUUM_MASS_ABS_M2_KG * pw).abs() < 1.0e-3,
            "tau {tau} != kappa*PW {}",
            IR_WV_CONTINUUM_MASS_ABS_M2_KG * pw
        );
        // Density decays with height; dry air absorbs nothing.
        assert!(standard_air_density_kg_m3(0.0) > standard_air_density_kg_m3(8500.0));
        assert_eq!(ir_wv_continuum_absorption(0.0, 1000.0), 0.0);
    }

    #[test]
    fn wv_band_absorption_scales_and_orders_and_wraps_the_continuum() {
        // The WV band absorption is linear in both the mass-absorption coefficient and
        // the vapor mass, and the per-band coefficients decrease 6.2 > 6.9 > 7.3 (so the
        // 6.2 um weighting function sits highest -> coldest BT; the physics under test).
        // These are compile-time-constant ordering guards (const-asserted).
        const {
            assert!(WV_MASS_ABS_BAND8_M2_KG > WV_MASS_ABS_BAND9_M2_KG);
            assert!(WV_MASS_ABS_BAND9_M2_KG > WV_MASS_ABS_BAND10_M2_KG);
            // WV bands absorb FAR more per unit vapor than the weak window continuum
            // (that is the whole difference from band 13 — the surface disappears).
            assert!(WV_MASS_ABS_BAND10_M2_KG > 10.0 * IR_WV_CONTINUUM_MASS_ABS_M2_KG);
            // The WV wavelengths are ordered 6.2 < 6.9 < 7.3 um.
            assert!(WV_BAND8_WAVELENGTH_M < WV_BAND9_WAVELENGTH_M);
            assert!(WV_BAND9_WAVELENGTH_M < WV_BAND10_WAVELENGTH_M);
        };
        // Linear in vapor and in kappa.
        let q = 0.010;
        let z = 3000.0;
        let a1 = wv_absorption(q, z, WV_MASS_ABS_BAND8_M2_KG);
        let a2 = wv_absorption(2.0 * q, z, WV_MASS_ABS_BAND8_M2_KG);
        assert!((a2 - 2.0 * a1).abs() < 1e-12, "not linear in vapor");
        let ratio = wv_absorption(q, z, 2.0 * WV_MASS_ABS_BAND8_M2_KG) / a1;
        assert!((ratio - 2.0).abs() < 1e-12, "not linear in kappa");
        // The continuum function is EXACTLY the wrapper at the continuum coefficient
        // (so band 13 behaviour is byte-identical to before this refactor).
        assert_eq!(
            ir_wv_continuum_absorption(q, z),
            wv_absorption(q, z, IR_WV_CONTINUUM_MASS_ABS_M2_KG)
        );
        // Guards: non-positive vapor / coefficient -> zero.
        assert_eq!(wv_absorption(0.0, z, WV_MASS_ABS_BAND8_M2_KG), 0.0);
        assert_eq!(wv_absorption(q, z, 0.0), 0.0);
    }

    #[test]
    fn cox_munk_mss_is_monotone_in_wind() {
        // sigma^2 = 0.003 + 0.00512 W: rises with wind (a windier sea is rougher).
        assert!((cox_munk_mean_square_slope(0.0) - 0.003).abs() < 1e-9);
        let mut prev = -1.0;
        for &w in &[0.0f64, 2.0, 5.0, 10.0, 15.0, 20.0] {
            let m = cox_munk_mean_square_slope(w);
            assert!(m > prev, "mss not monotone at W={w}: {m} <= {prev}");
            prev = m;
        }
        // Explicit value at 10 m/s: 0.003 + 0.0512 = 0.0542.
        assert!((cox_munk_mean_square_slope(10.0) - 0.0542).abs() < 1e-9);
        // Negative wind clamps to the calm value.
        assert_eq!(
            cox_munk_mean_square_slope(-5.0),
            cox_munk_mean_square_slope(0.0)
        );
    }

    #[test]
    fn fresnel_water_is_low_at_normal_high_at_grazing() {
        let n = WATER_REFRACTIVE_INDEX_VIS;
        // Normal incidence (cos = 1): ((1-n)/(1+n))^2 ~= 0.0211 for n = 1.34.
        let normal = fresnel_reflectance_unpolarized(1.0, n);
        assert!((normal - 0.0211).abs() < 0.002, "normal Fresnel {normal}");
        // Grazing (cos -> 0): -> 1.
        let grazing = fresnel_reflectance_unpolarized(0.02, n);
        assert!(grazing > 0.7, "grazing Fresnel {grazing} should approach 1");
        // Monotone increasing from normal toward grazing (decreasing cos).
        let mut prev = 0.0;
        for &c in &[1.0f64, 0.8, 0.6, 0.4, 0.2, 0.05] {
            let r = fresnel_reflectance_unpolarized(c, n);
            assert!(
                r >= prev - 1e-9,
                "Fresnel not monotone at cos={c}: {r} < {prev}"
            );
            prev = r;
        }
        assert!((0.0..=1.0).contains(&fresnel_reflectance_unpolarized(0.5, n)));
    }

    #[test]
    fn glint_peaks_at_the_specular_direction_and_widens_with_wind() {
        // Geometry in a local ENU-like basis: up = +z. Put the sun at 40 deg elevation
        // toward +east; the SPECULAR viewer direction (mirror of the sun about up) is
        // 40 deg elevation toward −east. Off-specular is a small azimuth tilt.
        let up = [0.0, 0.0, 1.0];
        let se = 40.0f64.to_radians();
        let to_sun = [se.cos(), 0.0, se.sin()]; // east-ish, 40 deg up
        // Specular view: same elevation, mirrored horizontal (−east).
        let spec_view = [-se.cos(), 0.0, se.sin()];
        let mss = cox_munk_mean_square_slope(5.0);
        let at_spec = cox_munk_glint_reflectance(to_sun, spec_view, up, mss);
        assert!(
            at_spec > 0.0 && at_spec.is_finite(),
            "specular glint {at_spec}"
        );
        // A view well off the specular azimuth gets far less glint. 30 deg of azimuth
        // offset puts the required wave-facet tilt out in the slope-distribution tail
        // (beyond the calm-sea RMS slope), which is where wind widens the glitter.
        let off = 30.0f64.to_radians();
        let off_view = [-se.cos() * off.cos(), se.cos() * off.sin(), se.sin()];
        let at_off = cox_munk_glint_reflectance(to_sun, off_view, up, mss);
        assert!(
            at_off < at_spec,
            "off-specular {at_off} !< specular {at_spec}"
        );
        // "Widens with wind": at the SAME far-tail off-specular geometry, a windier
        // (rougher) sea spreads more energy into the tail -> more glint there than calm.
        let calm =
            cox_munk_glint_reflectance(to_sun, off_view, up, cox_munk_mean_square_slope(1.0));
        let windy =
            cox_munk_glint_reflectance(to_sun, off_view, up, cox_munk_mean_square_slope(12.0));
        assert!(
            windy > calm,
            "glitter tail should widen with wind: windy {windy} !> calm {calm}"
        );
        // Sun below the horizon -> no glint.
        let below = cox_munk_glint_reflectance([1.0, 0.0, -0.1], spec_view, up, mss);
        assert_eq!(below, 0.0);
    }
}
