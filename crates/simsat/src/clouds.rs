//! Volumetric cloud raymarch — CPU reference (design doc section 4, M4).
//!
//! This is the tested CPU REFERENCE for the M4 cloud raymarch; the WGSL twin is
//! `gpu/shaders/clouds.wgsl` (a superset of the M2 surface pass). Nodes have no GPU,
//! so the physics is validated here in `cargo test` and the shader is kept in
//! lockstep by discipline — the same twin workflow M2 established for
//! `surface.wgsl` <-> `render.rs`/`atmosphere.rs`.
//!
//! What M4 does (design section 4, EXCLUDING the M5 items):
//!   - the TRUE slant ray marched in spherical ECEF (design section 1): a straight
//!     line in ECEF, per step ECEF -> lat/lon/h -> projection forward -> fractional
//!     (i, j) -> brick sample;
//!   - adaptive stepping: coarse ~2x voxel-pitch steps through empty space using a
//!     low-res occupancy mip, refined ~0.5x pitch inside cloud (caps 192/384);
//!   - dual-lobe Henyey-Greenstein phase per class (liquid/ice, precip on the ice
//!     lobe), single-scatter albedo 1.0 (conservative);
//!   - sun transmittance = sun-OD-map fetch + short-range detail taps (Nubis);
//!   - Schneider beer-powder on the sun term only (toggleable, ON default), a named
//!     stylization that never exceeds pure Beer;
//!   - per-voxel scalar sky-irradiance ambient (M2's ambient table) attenuated by
//!     e^-tau_up from above (the brick channel) and a cheap ground-bounce from below;
//!   - composite over the M2 surface radiance, aerial perspective on the cloud from
//!     the M2 froxel, output through the ABI reflectance stretch.
//!
//! NOT M4 (these are M5): Wrenninge multi-scatter octaves, penumbral/pre-blurred
//! cloud shadows, full SH-2 directional ambient. Sub-grid noise is off (owner
//! default) and not built here.
//!
//! Geometry: distances are metres, ECEF radii are measured from the earth CENTRE.
//! The brick vertical axis is MSL height `z(k) = z_min + k*dz`; the ground sphere is
//! at `R_GROUND_M`, so an ECEF point at radius `r` has MSL height `h = r - R_GROUND_M`.

use std::f64::consts::PI;

use rayon::prelude::*;

use crate::atmosphere::{
    self, AerialFroxel, AtmosphereLuts, GROUND_ALBEDO, R_GROUND_M, SOLAR_IRRADIANCE_RGB, SkyShTable,
};
use crate::bricks::VolumeBrick;
use crate::camera::ScanGrid;
use crate::frame::GridGeoref;
use crate::render::{
    CLOUD_SOFTCLIP_KNEE, FrameContext, GROUND_DAY_LIFT, SurfacePixel, radiance_to_rgba_softclip,
    surface_toa_radiance,
};

// ── optics constants (design section 4) ──────────────────────────────────────

/// Dual-lobe Henyey-Greenstein for cloud LIQUID: forward lobe `g1`, back lobe `g2`,
/// forward weight `w` (design section 4).
pub const PHASE_LIQUID_G1: f64 = 0.85;
pub const PHASE_LIQUID_G2: f64 = -0.15;
pub const PHASE_LIQUID_W: f64 = 0.9;
/// Dual-lobe HG for cloud ICE (design section 4). No forward weight is given in the
/// spec for ice; we reuse the liquid `w = 0.9` (documented choice — ice crystals are
/// also strongly forward-scattering in the visible).
pub const PHASE_ICE_G1: f64 = 0.75;
pub const PHASE_ICE_G2: f64 = -0.10;
pub const PHASE_ICE_W: f64 = 0.9;

/// Ambient split: how much of the cloud ambient arrives from above (attenuated by
/// e^-tau_up) vs from below (a cheap ground-bounce estimate). Sum to 1.
pub const AMBIENT_W_ABOVE: f64 = 0.7;
pub const AMBIENT_W_BELOW: f64 = 0.3;

/// Default occupancy-mip downsample factor (per axis): 8x (design section 4).
pub const OCCUPANCY_MIP_FACTOR: usize = 8;

/// Secondary sun-march (design section 4, the Nubis/Frostbite light march) default
/// step count. Short by design: exponentially-spaced samples reach the top of a thick
/// anvil in a handful of taps while resolving the near field that dominates the edge.
pub const SUN_MARCH_STEPS: usize = 6;
/// Growth factor of the exponentially-spaced sun-march steps (each step is this
/// multiple of the previous). With a `voxel_pitch` base and 6 steps the reach is
/// `pitch*(growth^6 - 1)/(growth - 1)` = 63x pitch at growth 2 (~15.75 km at 250 m).
pub const SUN_MARCH_GROWTH: f64 = 2.0;
/// OFFLINE secondary sun-march step count (WS1 march-physics pass). The offline /
/// stored-frame quality tier buys a denser, slower-growing light march: 10 steps at
/// growth 1.5 reach `(1.5^10 - 1)/0.5` = ~113x pitch (~28 km at 250 m) with a much
/// finer near field than the interactive `(6, 2.0)` schedule. Selected by
/// [`MarchConfig::new`] from the [`StepQuality`]; interactive keeps `(6, 2.0)`.
pub const SUN_MARCH_STEPS_OFFLINE: usize = 10;
/// OFFLINE secondary sun-march growth factor (see [`SUN_MARCH_STEPS_OFFLINE`]).
pub const SUN_MARCH_GROWTH_OFFLINE: f64 = 1.5;
/// Default stratified-sampling jitter amplitude for the secondary sun march, in
/// `[0, 1]` ([`MarchConfig::sun_march_jitter_amp`]). When non-zero, the exponential
/// schedule samples each segment at a DETERMINISTIC hash-offset point instead of
/// the fixed midpoint (classic stratified sampling: one uniform offset per ray,
/// from [`hash01_position`]), which decorrelates the banding a fixed-phase
/// geometric schedule can imprint on smooth cloud fields. `0.0` = the fixed
/// midpoint; `1.0` = the full stratum.
///
/// DEFAULT 0.0 (a documented look decision, WS1): at full amplitude the jitter
/// turned dusk anvil faces visibly GRAINY — near-horizontal sun rays make
/// `tau_sun` extremely sensitive to the sampled offset, so the stratified noise
/// dwarfs the subtle schedule banding it was meant to remove (A/B frames in
/// `notes/qa-frames/ws1-march-physics/`: `after_dusk_actualsun.png` amp 1.0 vs
/// `probe_dusk_amp0.png`). The machinery ships tested and mirrored in the WGSL
/// twin; enabling is a one-constant change if schedule banding is ever observed.
pub const SUN_MARCH_JITTER_AMP: f64 = 0.0;

// ── Wrenninge/Oz multi-scatter octaves (design section 4, M5) ─────────────────
//
// The single dual-HG scatter of M4/fix2 lights only the thin sun-facing skin of a
// thick cloud (its forward-peaked phase throws most of the one bounce it models
// forward/down, and self-shadow kills the deep samples), so a sunlit anvil top read
// only ~0.10-0.16 reflectance instead of the 0.5-0.9 of real convective tops. That
// gap is MULTIPLE scattering. The Wrenninge/Oz "octaves" approximation (Wrenninge,
// Kulla & Sannikov, "Oz: The Great and Volumetric", SIGGRAPH 2013 talks; adopted in
// Hillaire, "Physically Based Sky, Atmosphere and Cloud Rendering in Frostbite",
// SIGGRAPH 2016) recovers the bright diffuse reflection as a SUM of `N` cheap octaves:
// octave `k` reuses the SAME sun optical depth `tau_sun` and scattering angle but with
//   - extinction scaled `a^k` in the Beer term: deeper octaves see LESS self-shadow,
//     so light penetrates a thick cloud (the dominant thick-anvil brightening);
//   - phase eccentricity scaled `g*b^k`: deeper octaves approach isotropic, boosting
//     the weak back-scatter of the GEO/sun geometry;
//   - a brightness weight `c^k`: a geometric decay so the octave sum converges to a
//     BOUNDED ceiling (required for "reflectance <= 1, monotone toward a ceiling";
//     `c = 1`, a plain unbounded sum, would grow without limit in `N`).
// Octave 0 (a^0 = b^0 = c^0 = 1) is EXACTLY the fix2 single scatter, so `octaves = 1`
// reproduces fix2 and the studio A/B is `octaves = DEFAULT_OCTAVES` vs `1`.
//
// This is an ENERGY-GAIN APPROXIMATION of multiple scattering, NOT a full solution
// (the honesty standard, design section 6). Cost: the secondary sun march runs ONCE
// per sample; the octaves are `N` cheap phase+exp evaluations of that one `tau_sun` —
// the primary march is NOT lengthened (design "do not triple the march length").

/// Default octave count. Design "start N=3"; we default to 6 so a thick anvil reaches
/// the observed convective-top reflectance (order 0.5-0.8) — with the `c < 1` weight
/// decay the octave sum is near its bounded ceiling by then. `N` is a runtime knob
/// ([`MarchConfig::octaves`]; the studio A/B and the monotone-toward-ceiling test vary
/// it; `N = 1` reproduces fix2 single scatter).
pub const DEFAULT_OCTAVES: usize = 6;
/// Per-octave EXTINCTION scale `a` (applied to `tau_sun` in the Beer self-shadow):
/// deeper octaves see `tau_sun * a^k`, i.e. less attenuation -> light penetrates.
pub const OCTAVE_EXTINCTION_SCALE: f64 = 0.5;
/// Per-octave PHASE-eccentricity scale `b` (applied to the HG `g` lobes): deeper
/// octaves approach isotropic, strengthening the back-scatter term.
pub const OCTAVE_PHASE_SCALE: f64 = 0.5;
/// Per-octave BRIGHTNESS weight `c` (`weight_k = c^k`): the geometric decay that gives
/// the octave sum a finite ceiling. Set to 0.85 because visible cloud is a NEAR-
/// CONSERVATIVE scatterer (single-scatter albedo ~1), so successive scattering orders
/// lose little energy — a high `c` is the physically-honest choice for a thick cloud
/// and is what lifts the sunlit anvil to the 0.5-0.9 real convective-top reflectance.
/// Still `< 1`, so the octave sum converges to a bounded ceiling (the energy-bound and
/// monotone-toward-ceiling tests hold).
pub const OCTAVE_BRIGHTNESS_SCALE: f64 = 0.85;

/// Solar disk angular DIAMETER (rad) = 0.533 deg — the penumbra-widening factor for
/// the ground cloud shadow (design section 6: blur radius = occluder distance x
/// tan 0.533 deg). `tan(0.533 deg) ~= 0.0093`.
pub const SUN_ANGULAR_DIAMETER_RAD: f64 = 0.533 * std::f64::consts::PI / 180.0;

// LIMB-DARKENING NOTE (WS1 march-physics decision, recorded next to the octave
// calibration it belongs to): the SURFACE direct-sun term dims by the disk-averaged
// Hestroffer-Magnan factor `atmosphere::LIMB_DARKENING_DISK_AVG = 0.832`; the CLOUD
// sun term below does NOT apply it. Applying it now would be a uniform -17% on every
// sunlit cloud — a LOOK change to the owner-approved M5 brilliance that needs an
// orchestrator visual round, not a silent physics landing. The omission is absorbed
// by the octave brightness calibration (`OCTAVE_BRIGHTNESS_SCALE` and the 0.5-0.9
// anvil reflectance target were tuned WITHOUT the factor), so current behavior is a
// documented CALIBRATION choice, kept; flagged to the orchestrator for a future
// cloud/surface consistency look-round.

// ── domain/margin edge feather (zoom-out appearance pass) ─────────────────────
//
// With a zoom-out margin (`RenderParams::margin_frac > 0`) the domain sits inside a
// larger frame of real ground + clear sky, but WRF has no cloud data outside the domain,
// so the cloud volume ends abruptly at the rectangular domain edge — a hard cloud wall
// against the clear margin, the biggest remaining "looks wrong" contributor. The EDGE
// FEATHER ramps the cloud/volume contribution down to zero over the outer band of the
// domain so clouds fade smoothly into the clear margin. It is applied per march sample by
// scaling the local extinction, so a faded cloud both scatters less light AND grows more
// transparent (the ground shows through) — the physically-consistent fade. It is GATED on
// margin: the caller passes a band width of `0.0` at margin 0 (edge-to-edge), where the
// feather is a byte-identical no-op, preserving the approved domain-fills-the-frame look.

/// EDGE FEATHER band width as a FRACTION of the smaller domain axis: the cloud
/// contribution ramps from 0 at the domain edge to full at this depth into the domain.
/// `0.04` = the outer ~4% of the domain (design "the outer ~3-5%"). Only active when a
/// zoom-out margin is present (see [`edge_feather_cells_for_margin`]).
pub const EDGE_FEATHER_BAND_FRAC: f64 = 0.04;

/// The EDGE FEATHER band width in CELLS for a given zoom-out `margin_frac` and domain
/// size: `EDGE_FEATHER_BAND_FRAC * min(nx, ny)` when `margin_frac > 0`, else `0.0` (the
/// neutral no-op — at margin 0 the domain fills the frame and no feather is applied, so
/// the render is byte-identical to before). Set into [`MarchConfig::edge_feather_cells`]
/// by the render assembly.
#[inline]
pub fn edge_feather_cells_for_margin(margin_frac: f64, nx: usize, ny: usize) -> f64 {
    if margin_frac > 0.0 {
        EDGE_FEATHER_BAND_FRAC * (nx.min(ny) as f64)
    } else {
        0.0
    }
}

/// The EDGE FEATHER weight in `[0, 1]` for a fractional brick sample `(fi, fj)` in a
/// domain of `nx * ny` cells, with a feather band of `band_cells` cells: `1.0` in the
/// interior, ramping smoothly to `0.0` at the domain edge over the outer `band_cells`.
/// `band_cells <= 0` -> `1.0` everywhere (the neutral no-op). A monotone smoothstep of the
/// distance to the nearest of the four domain edges (`0 .. n-1` box); a sample outside the
/// domain is `0.0` (though such samples already read CLEAR extinction). Non-finite -> 0.
#[inline]
pub fn edge_feather(fi: f64, fj: f64, nx: usize, ny: usize, band_cells: f64) -> f64 {
    if band_cells <= 0.0 {
        return 1.0;
    }
    if !(fi.is_finite() && fj.is_finite()) {
        return 0.0;
    }
    let hi_i = nx.saturating_sub(1) as f64;
    let hi_j = ny.saturating_sub(1) as f64;
    let d = fi.min(hi_i - fi).min(fj).min(hi_j - fj);
    if d <= 0.0 {
        return 0.0;
    }
    if d >= band_cells {
        return 1.0;
    }
    let t = d / band_cells;
    t * t * (3.0 - 2.0 * t) // smoothstep, monotone on [0, 1]
}

// ── sub-grid cloud GRANULATION (edge-erosion detail noise) ────────────────────
//
// WRF cannot resolve the sub-kilometre elements of a boundary-layer cumulus field
// (its effective resolution is ~7 dx — Skamarock 2004, MWR 132), so popcorn-cu /
// confluence-band cloud renders as hard-edged solid-white cutouts where the real sky
// is a granular carpet of sub-grid elements. GRANULATION represents that unresolved
// variability by a SUBTRACT-ONLY, deterministic EROSION of the sampled extinction:
//
//   sigma' = sigma * m(d, e),  m = remap-style multiplier (Nubis/Decima lineage),
//   d = sigma / (trilinear-corner-neighbourhood max)  — the RELATIVE density,
//   e = amplitude/GRAN_AMP_CAP * gate * noise         — the erosion threshold,
//
// where `noise` is a 3-octave 2-D WORLEY (cellular) F1 field — cumulus morphology per
// the Nubis/Decima detail-erosion lineage (Schneider, "The Real-Time Volumetric
// Cloudscapes of Horizon: Zero Dawn", SIGGRAPH 2015 Advances) — with octave weights
// following a k^-5/3 spectrum envelope (observed liquid-water spectra: Davis et al.
// 1996, JAS 53; the 3DCLOUD realism criterion: Szczap et al. 2014, GMD 7), base cell
// scale a few hundred metres to ~1 km (the dominant element scale of the observed
// cloud-size distribution, power-law exponent ~1.66: Wood & Field 2011, J. Climate 24).
// The noise is 2-D (horizontal) so grains are vertically-coherent COLUMNS, the honest
// morphology of boundary-layer elements, and it is anchored DETERMINISTICALLY in
// brick/world space (position-hash cells, the hash01_position style): the same run
// renders the same field every frame (loops do not shimmer) and the geostationary and
// top-down views agree exactly (both sample by fractional brick coordinates).
//
// HONESTY CONTRACT (each point is test-pinned):
//   - SUBTRACT-ONLY: sigma' <= sigma everywhere; sigma = 0 stays 0 — the erosion never
//     ADDS cloud where the model has none.
//   - The remap pins d = 1 to m = 1: the interior of an optically thick core (at its
//     neighbourhood max) is byte-UNTOUCHED; erosion lives where the local extinction is
//     low relative to its neighbourhood (edges, trilinear tent flanks) — which is what
//     GRANULATES a blocky cell into grains instead of merely feathering its outline.
//   - AMPLITUDE = the unresolved-variance fraction of a k^-5/3 spectrum integrated
//     from the model's effective resolution (~7 dx, Skamarock 2004) down to the finest
//     represented scale, normalised toward the observed all-scale fractional standard
//     deviation of cloud condensate ~0.75 (Shonk et al. 2010, QJRMS 136, Tripleclouds)
//     and CAPPED so the field-mean tau reduction stays within the ~0.7 plane-parallel
//     correction bound (Cahalan et al. 1994, JAS 51) — it can never over-thin. On a
//     250 m run the amplitude is naturally small (~0.17 — the model already resolves
//     the texture); on a 2-3 km run it is strong (~0.44-0.48).
//   - SPECIES/HEIGHT GATED: full strength only on LIQUID extinction in the boundary
//     layer (below ~4 km), fading to zero by ~7 km; ice anvils and cirrus are left
//     alone (their edges already render correctly; cirrus inhomogeneity is a separate,
//     weaker regime — Hogan & Kew 2005, QJRMS 131).
//   - SCOPING: display products only (VisibleRgb, the GeoColor day half, the Sandwich
//     visible base). Raw-Kelvin thermal products (IR/WV) march the separate IrVolume
//     and are byte-identical by construction; the raw-reflectance bands and derived
//     fields default OFF (quantitative outputs must reflect model skill, not display
//     texturing) — see `api::RenderParams::granulation`.
//
// HONEST LIMITATION (McICA/COSP precedent — a SUB-GRID VARIABILITY representation,
// not stochastic cloud placement): erosion can only carve what the model put there.
// It breaks up cell-scale blobs and band edges; it cannot recreate distinct sub-km
// elements deep inside a 20 km SOLID overcast slab (whose interior is honestly at its
// neighbourhood max and stays untouched).
//
// The eroded field is applied at the SAMPLER level ([`DecodedVolume::sample_granulated`])
// and the same [`Granulation`] value is threaded to the view march, the secondary sun
// march ([`MarchConfig::granulation`]) AND the sun-OD map accumulation
// ([`accumulate_sun_od_granulated`]), so every march of one composite samples the SAME
// eroded field (clouds, their self-shadows and their ground shadows stay consistent).
// The occupancy mip is built from the UN-ERODED field — erosion only reduces extinction,
// so the mip stays conservative. `tau_up` (an ingest-time column integral) is NOT eroded
// (it feeds only the cloud-ambient attenuation; a documented second-order approximation).

/// The three Worley octave cell scales (m): base ~1 km down to ~250 m (Wood & Field
/// 2011 element scales; the finest octave is [`GRAN_MIN_SCALE_M`], the render-scale
/// floor of the spectrum integral).
pub const GRAN_OCTAVE_SCALES_M: [f64; 3] = [1000.0, 500.0, 250.0];
/// Octave AMPLITUDE weights following the k^-5/3 energy-spectrum envelope: the band
/// standard deviation scales as lambda^(1/3), so `w_i = lambda_i^(1/3) / sum` =
/// cbrt(1000)/S, cbrt(500)/S, cbrt(250)/S with S = 24.2366105 (literals — cbrt is not
/// const and libm bit-variation must not enter the deterministic field).
pub const GRAN_OCTAVE_WEIGHTS: [f64; 3] = [0.412_598_7, 0.327_480_0, 0.259_921_3];
/// Per-octave hash salts (arbitrary, fixed — part of the deterministic anchoring).
pub const GRAN_OCTAVE_SALTS: [u32; 3] = [0x51A7_C0DE, 0x9BD2_A0E5, 0x2F63_D19B];
/// The finest represented granulation scale (m) — the lower limit of the k^-5/3
/// variance integral and the smallest octave cell.
pub const GRAN_MIN_SCALE_M: f64 = 250.0;
/// The outer reference scale (m) of the unresolved-variance normalisation: the
/// GCM-gridbox scale at which the Shonk et al. (2010) fractional standard deviation
/// ~0.75 was estimated. The amplitude asymptotes toward [`GRAN_SHONK_FSD`] (then the
/// Cahalan cap) as the model grid coarsens toward it.
pub const GRAN_SPECTRUM_OUTER_M: f64 = 100_000.0;
/// The observed all-scale fractional standard deviation of cloud condensate (Shonk
/// et al. 2010, QJRMS 136: the global Tripleclouds estimate).
pub const GRAN_SHONK_FSD: f64 = 0.75;
/// A model's EFFECTIVE resolution in grid cells (Skamarock 2004, MWR 132: kinetic-energy
/// spectra of WRF forecasts decay below ~7 dx).
pub const SKAMAROCK_EFFECTIVE_RES_CELLS: f64 = 7.0;
/// The plane-parallel correction bound (Cahalan et al. 1994, JAS 51: the effective
/// optical depth of inhomogeneous stratocumulus ~0.7x the mean): the field-mean tau
/// reduction of the erosion must stay within `1 - 0.7`; the amplitude cap and the
/// tail-shaped erosion field keep it there (test-pinned).
pub const CAHALAN_TAU_FACTOR: f64 = 0.7;
/// The amplitude cap enforcing the Cahalan bound: for a zero-gap binary medium at
/// fractional standard deviation `s`, the subtract-only realisation keeps `1/(1+s^2)`
/// of the mean; `s = sqrt(1/0.7 - 1) = 0.655` is the bound, capped below it at 0.6.
pub const GRAN_AMP_CAP: f64 = 0.6;
/// The erosion-threshold gain: `e = amplitude/GRAN_AMP_CAP * gate * noise`, so the
/// erosion threshold reaches full carve (1.0) exactly at the Cahalan-limit amplitude.
pub const GRAN_EROSION_GAIN: f64 = 1.0 / GRAN_AMP_CAP;
/// Erosion-threshold ceiling (keeps the remap denominator `1 - e` well-conditioned).
pub const GRAN_EROSION_MAX: f64 = 0.98;
/// Full granulation strength at/below this MSL height (m): the boundary-layer liquid
/// regime (spec: "strong on boundary-layer liquid below ~4 km").
pub const GRAN_HEIGHT_FULL_M: f64 = 4000.0;
/// Zero granulation at/above this MSL height (m): mid/high cloud (supercooled decks,
/// anvils, cirrus) is left alone (Hogan & Kew 2005). Smoothstep between the two.
pub const GRAN_HEIGHT_ZERO_M: f64 = 7000.0;
/// The tail-shaping window on the raw octave-Worley value: the erosion field is
/// `smoothstep(GRAN_CARVE_LO, GRAN_CARVE_HI, W)`, i.e. zero over the low-W GRAIN
/// interiors (most of the field survives untouched) and 1 over the high-W Voronoi
/// BORDER network (the carved gaps between grains) — the grain/gap bimodality that
/// makes the erosion granulate instead of uniformly dimming, and what keeps the mean
/// reduction inside the Cahalan bound (the eroded area fraction is the W-tail).
pub const GRAN_CARVE_LO: f64 = 0.52;
/// See [`GRAN_CARVE_LO`].
pub const GRAN_CARVE_HI: f64 = 0.62;
/// INTERIOR-PROTECTION window on the RELATIVE density `d` (round-1 QA lever): the
/// erosion threshold is scaled by `1 - smoothstep(GRAN_INTERIOR_LO,
/// GRAN_INTERIOR_HI, d)`, so a sample at `d >= GRAN_INTERIOR_HI` never erodes and
/// only genuinely boundary-relative samples (`d <= GRAN_INTERIOR_LO`: trilinear
/// tent flanks, band edges) see the full threshold. WITHOUT this, the ordinary
/// cell-to-cell LWC variability INSIDE a wide solid stratus deck (relative density
/// ~0.75-1 against the local max) read as "edge" and the pure remap peppered the
/// deck with dark pinholes (the round-1 1974 frame). Erosion is for boundaries;
/// deck interiors stay solid — consistent with the documented honest limitation.
pub const GRAN_INTERIOR_LO: f64 = 0.45;
/// See [`GRAN_INTERIOR_LO`].
pub const GRAN_INTERIOR_HI: f64 = 0.75;

/// Sub-grid granulation parameters carried by [`MarchConfig::granulation`] and
/// [`accumulate_sun_od_granulated`]. `None` anywhere = the feature fully off
/// (byte-identical to the pre-granulation render).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Granulation {
    /// The erosion amplitude in `[0, GRAN_AMP_CAP]` — the unresolved fractional
    /// standard deviation for the model grid (see [`granulation_amplitude`]).
    pub amplitude: f64,
}

impl Granulation {
    /// The granulation for a model grid spacing `dx_m` (m): the dx-derived amplitude.
    /// `Granulation::for_grid(250.0).amplitude` is ~0.17 (a 250 m run — near-neutral);
    /// `for_grid(3000.0)` is ~0.44 (a 3 km run — strong).
    pub fn for_grid(dx_m: f64) -> Self {
        Self {
            amplitude: granulation_amplitude(dx_m),
        }
    }
}

/// A clamped smoothstep of `t` (0 below 0, 1 above 1, C1 in between).
#[inline]
fn smooth01(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The dx-derived granulation amplitude: the unresolved fractional standard deviation
/// of a k^-5/3 spectrum. Variance between scales `[a, b]` of `E(k) ~ k^-5/3` is
/// proportional to `b^(2/3) - a^(2/3)` (in wavelength terms), so the fraction of the
/// all-scale (outer = [`GRAN_SPECTRUM_OUTER_M`]) variance the model leaves unresolved
/// between its effective resolution `7 dx` (Skamarock 2004) and the finest represented
/// scale [`GRAN_MIN_SCALE_M`] is `(lam_eff^(2/3) - lam_min^(2/3)) / (L^(2/3) -
/// lam_min^(2/3))`; the amplitude is [`GRAN_SHONK_FSD`] x its square root, capped at
/// [`GRAN_AMP_CAP`] (the Cahalan bound). Monotone in `dx`; ~0.17 at 250 m, ~0.29 at
/// 1 km, ~0.44 at 3 km, capped from ~7.4 km up. Non-finite / non-positive `dx` -> 0.
pub fn granulation_amplitude(dx_m: f64) -> f64 {
    if !dx_m.is_finite() || dx_m <= 0.0 {
        return 0.0;
    }
    let lam_eff = (SKAMAROCK_EFFECTIVE_RES_CELLS * dx_m).max(GRAN_MIN_SCALE_M);
    let pow23 = |x: f64| x.powf(2.0 / 3.0);
    let num = pow23(lam_eff) - pow23(GRAN_MIN_SCALE_M);
    let den = pow23(GRAN_SPECTRUM_OUTER_M) - pow23(GRAN_MIN_SCALE_M);
    let frac = (num / den).clamp(0.0, 1.0);
    (GRAN_SHONK_FSD * frac.sqrt()).min(GRAN_AMP_CAP)
}

/// Deterministic cell hash to `[0, 1)` — the hash01_position-style integer avalanche
/// over a 2-D noise-cell coordinate + salt (platform-stable pure function; the
/// granulation anchor). Twin of `gran_cell_hash01` in `clouds.wgsl`.
#[inline]
fn gran_cell_hash01(ix: i64, iy: i64, salt: u32) -> f64 {
    let mut h = (ix as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add((iy as u32).wrapping_mul(0x85EB_CA6B))
        .wrapping_add(salt.wrapping_mul(0xC2B2_AE35));
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    h as f64 / 4_294_967_296.0
}

/// 2-D Worley (cellular) F1: the distance from `(qx, qy)` (in CELL units) to the
/// nearest jittered feature point (one per integer cell, position hashed from the
/// cell coordinate + salt), clamped to `[0, 1]`. ~0 at grain centres, high on the
/// Voronoi border network between them.
fn worley2_f1(qx: f64, qy: f64, salt: u32) -> f64 {
    let bx = qx.floor() as i64;
    let by = qy.floor() as i64;
    let mut best = f64::INFINITY;
    for dy in -1i64..=1 {
        for dx in -1i64..=1 {
            let cx = bx + dx;
            let cy = by + dy;
            let fx = cx as f64 + gran_cell_hash01(cx, cy, salt);
            let fy = cy as f64 + gran_cell_hash01(cx, cy, salt ^ 0x68E3_1DA4);
            let d2 = (qx - fx) * (qx - fx) + (qy - fy) * (qy - fy);
            if d2 < best {
                best = d2;
            }
        }
    }
    best.sqrt().clamp(0.0, 1.0)
}

/// The GRANULATION EROSION FIELD at a horizontal position (m) in the brick plane:
/// the k^-5/3-weighted 3-octave Worley F1 ([`GRAN_OCTAVE_SCALES_M`] /
/// [`GRAN_OCTAVE_WEIGHTS`]) tail-shaped through `smoothstep(GRAN_CARVE_LO,
/// GRAN_CARVE_HI, W)` into `[0, 1]`: 0 over grain interiors (no erosion), 1 on the
/// carved gap network. Pure and deterministic in the position (memoised per thread
/// for the exact repeated `(u, v)` a nadir top-down column produces — a bit-exact
/// cache, never an approximation).
pub fn granulation_erosion_noise(u_m: f64, v_m: f64) -> f64 {
    thread_local! {
        static MEMO: std::cell::Cell<(u64, u64, f64)> =
            const { std::cell::Cell::new((0, 0, -1.0)) };
    }
    let key = (u_m.to_bits(), v_m.to_bits());
    let hit = MEMO.with(|m| m.get());
    if hit.2 >= 0.0 && hit.0 == key.0 && hit.1 == key.1 {
        return hit.2;
    }
    let mut w = 0.0f64;
    for i in 0..GRAN_OCTAVE_SCALES_M.len() {
        let lam = GRAN_OCTAVE_SCALES_M[i];
        w += GRAN_OCTAVE_WEIGHTS[i] * worley2_f1(u_m / lam, v_m / lam, GRAN_OCTAVE_SALTS[i]);
    }
    let e = smooth01((w.clamp(0.0, 1.0) - GRAN_CARVE_LO) / (GRAN_CARVE_HI - GRAN_CARVE_LO));
    MEMO.with(|m| m.set((key.0, key.1, e)));
    e
}

/// The SPECIES/HEIGHT gate in `[0, 1]`: the LIQUID share of the sample's extinction
/// times a smooth height ramp (1 at/below [`GRAN_HEIGHT_FULL_M`], 0 at/above
/// [`GRAN_HEIGHT_ZERO_M`]). Ice-only samples and high cloud gate to exactly 0
/// (byte-untouched); mixed-phase / mid-level liquid erodes weakly.
pub fn granulation_gate(ext_liquid: f64, ext_ice: f64, ext_precip: f64, z_msl_m: f64) -> f64 {
    let total = ext_liquid + ext_ice + ext_precip;
    if total <= 0.0 {
        return 0.0;
    }
    let liquid_frac = (ext_liquid / total).clamp(0.0, 1.0);
    let height =
        1.0 - smooth01((z_msl_m - GRAN_HEIGHT_FULL_M) / (GRAN_HEIGHT_ZERO_M - GRAN_HEIGHT_FULL_M));
    liquid_frac * height
}

/// The INTERIOR-PROTECTION factor in `[0, 1]` for a relative density `d` (see
/// [`GRAN_INTERIOR_LO`]): `1` at/below `GRAN_INTERIOR_LO` (a true boundary sample —
/// full erosion), `0` at/above `GRAN_INTERIOR_HI` (interior variability of a solid
/// deck — never eroded), a monotone smoothstep between.
pub fn granulation_interior_protection(rel_density: f64) -> f64 {
    1.0 - smooth01(
        (rel_density.clamp(0.0, 1.0) - GRAN_INTERIOR_LO) / (GRAN_INTERIOR_HI - GRAN_INTERIOR_LO),
    )
}

/// The remap-style EROSION MULTIPLIER `m in [0, 1]` for a sample at RELATIVE density
/// `d = sigma / neighbourhood-max` under erosion threshold `e` (the Nubis density
/// remap `d' = (d - e) / (1 - e)`, returned as the ratio `m = d'/d`): `m = 1` at
/// `d = 1` for ANY `e` (interiors at their neighbourhood max are untouched), `m = 0`
/// where `d <= e` (fully carved), monotone in both arguments, and `m <= 1` always
/// (subtract-only). `e <= 0` -> 1 (the neutral no-op).
pub fn granulation_multiplier(rel_density: f64, erosion: f64) -> f64 {
    if erosion <= 0.0 {
        return 1.0;
    }
    let d = rel_density.clamp(0.0, 1.0);
    if d <= 0.0 {
        return 1.0; // zero extinction: nothing to erode (callers skip this case)
    }
    let e = erosion.min(GRAN_EROSION_MAX);
    (((d - e).max(0.0)) / (d * (1.0 - e))).clamp(0.0, 1.0)
}

/// Henyey-Greenstein phase (normalised to integrate to 1 over the sphere).
#[inline]
pub fn henyey_greenstein(cos_theta: f64, g: f64) -> f64 {
    let g2 = g * g;
    (1.0 - g2) / (4.0 * PI * (1.0 + g2 - 2.0 * g * cos_theta).powf(1.5))
}

/// Dual-lobe HG: `w*HG(g1) + (1-w)*HG(g2)`. Integrates to 1 (each lobe does, and the
/// weights sum to 1).
#[inline]
pub fn dual_henyey_greenstein(cos_theta: f64, g1: f64, g2: f64, w: f64) -> f64 {
    w * henyey_greenstein(cos_theta, g1) + (1.0 - w) * henyey_greenstein(cos_theta, g2)
}

/// Liquid-cloud phase.
#[inline]
pub fn phase_liquid(cos_theta: f64) -> f64 {
    dual_henyey_greenstein(cos_theta, PHASE_LIQUID_G1, PHASE_LIQUID_G2, PHASE_LIQUID_W)
}

/// Ice-cloud phase (precip is treated on this lobe — a documented choice: rain and
/// graupel are large, strongly forward-scattering particles, well modelled by the
/// broad ice lobe; a dedicated rain phase is out of M4 scope).
#[inline]
pub fn phase_ice(cos_theta: f64) -> f64 {
    dual_henyey_greenstein(cos_theta, PHASE_ICE_G1, PHASE_ICE_G2, PHASE_ICE_W)
}

/// The scattering-weighted aggregate phase of a mixed-phase sample. Liquid uses the
/// liquid lobe; ice + precip use the ice lobe (single-scatter albedo 1, so scattering
/// == extinction per class).
#[inline]
pub fn aggregate_phase(cos_theta: f64, ext_liquid: f64, ext_ice_precip: f64) -> f64 {
    let total = ext_liquid + ext_ice_precip;
    if total <= 0.0 {
        return 1.0 / (4.0 * PI); // isotropic fallback (never actually used: sigma=0)
    }
    (ext_liquid * phase_liquid(cos_theta) + ext_ice_precip * phase_ice(cos_theta)) / total
}

/// The scattering-weighted aggregate phase with the dual-HG eccentricities scaled by
/// `g_scale` (the Wrenninge octave phase term: octave `k` uses `g_scale = b^k`, so the
/// phase relaxes toward isotropic `1/(4 pi)` with depth). At `g_scale = 1` this equals
/// [`aggregate_phase`]. Each HG lobe stays a normalised phase for any scaled `g`.
#[inline]
pub fn aggregate_phase_scaled(
    cos_theta: f64,
    ext_liquid: f64,
    ext_ice_precip: f64,
    g_scale: f64,
) -> f64 {
    let total = ext_liquid + ext_ice_precip;
    if total <= 0.0 {
        return 1.0 / (4.0 * PI);
    }
    let liq = dual_henyey_greenstein(
        cos_theta,
        PHASE_LIQUID_G1 * g_scale,
        PHASE_LIQUID_G2 * g_scale,
        PHASE_LIQUID_W,
    );
    let ice = dual_henyey_greenstein(
        cos_theta,
        PHASE_ICE_G1 * g_scale,
        PHASE_ICE_G2 * g_scale,
        PHASE_ICE_W,
    );
    (ext_liquid * liq + ext_ice_precip * ice) / total
}

/// The Wrenninge/Oz multi-scatter octave SUN SOURCE scalar (design section 4, M5): the
/// sum over `octaves` octaves of `weight_k * phase(g*b^k) * vis(tau_sun*a^k)`, where
/// `vis` is Beer (or beer-powder) and `tau_sun` is the single depth-resolved cloud sun
/// optical depth (marched once, reused by every octave). Replaces the fix2
/// `phase(cos) * vis(tau_sun)` single-scatter sun term; at `octaves = 1` it equals it
/// exactly. Bounded and monotone in `octaves` (each added octave is a positive term of
/// a geometrically-decaying series -> a finite ceiling; the `octave_reflectance_*`
/// tests assert both). See the octave-constants block for the physics + citation.
#[inline]
pub fn octave_sun_source(
    cos_theta: f64,
    ext_liquid: f64,
    ext_ice_precip: f64,
    tau_sun: f64,
    beer_powder_on: bool,
    octaves: usize,
) -> f64 {
    let mut acc = 0.0f64;
    let mut ext_scale = 1.0f64; // a^k
    let mut g_scale = 1.0f64; // b^k
    let mut weight = 1.0f64; // c^k
    for _ in 0..octaves.max(1) {
        let tau_k = tau_sun * ext_scale;
        let vis_k = if beer_powder_on {
            beer_powder(tau_k)
        } else {
            beer(tau_k)
        };
        let phase_k = aggregate_phase_scaled(cos_theta, ext_liquid, ext_ice_precip, g_scale);
        acc += weight * phase_k * vis_k;
        ext_scale *= OCTAVE_EXTINCTION_SCALE;
        g_scale *= OCTAVE_PHASE_SCALE;
        weight *= OCTAVE_BRIGHTNESS_SCALE;
    }
    acc
}

/// Pure Beer-Lambert sun transmittance `e^-tau`.
#[inline]
pub fn beer(tau: f64) -> f64 {
    (-tau).exp()
}

/// Schneider's beer-powder sugar term `e^-tau * (1 - e^-2tau)`, applied ONLY to the
/// sun term (a named STYLIZATION with a physical rationale: it approximates the
/// missing forward-scatter buildup that darkens optically-thin cloud edges). It is
/// bounded above by pure Beer for all `tau >= 0` (the `beer_powder_never_exceeds_beer`
/// test asserts this), so it can only darken, never brighten.
#[inline]
pub fn beer_powder(tau: f64) -> f64 {
    (-tau).exp() * (1.0 - (-2.0 * tau).exp())
}

/// The ambient attenuation factor for a cloud voxel: a scalar in `[0, 1]` that scales
/// M2's ambient irradiance. Sky above reaches the voxel attenuated by `e^-tau_up`
/// (the brick channel = optical depth above the voxel); a cheap ground-bounce from
/// below is attenuated by `e^-tau_down` and the ground albedo. Monotone DECREASING in
/// `tau_up` (the `ambient_factor_is_monotone_in_tau_up` test asserts this).
#[inline]
pub fn ambient_cloud_factor(tau_up: f64, tau_down: f64, ground_albedo: f64) -> f64 {
    AMBIENT_W_ABOVE * (-tau_up).exp() + AMBIENT_W_BELOW * ground_albedo * (-tau_down).exp()
}

// ── small vec3 helpers over [f64;3] ──────────────────────────────────────────

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn madd3(a: [f64; 3], b: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s]
}
#[inline]
fn scl3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
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

/// Deterministic hash of an ECEF position to `[0, 1)` — the stratified-sampling
/// jitter seed for the secondary sun march. The position is quantized to whole
/// metres (ECEF magnitudes ~6.4e6 m fit i32 comfortably) and mixed with a small
/// integer avalanche, so the value is a pure, platform-stable function of the
/// sample position: the same ray gets the same offset every render (no temporal
/// shimmer), neighbouring samples get decorrelated offsets (no banding). The WGSL
/// twin (`clouds.wgsl::hash01`) uses the same mix on f32-rounded coordinates; its
/// low-bit rounding may differ from f64 — a documented divergence (the jitter is
/// decorrelation, not physics, so bit parity is not required).
#[inline]
pub fn hash01_position(p: [f64; 3]) -> f64 {
    let q = |x: f64| x.round() as i64 as u32;
    let mut h = q(p[0])
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(q(p[1]).wrapping_mul(0x85EB_CA6B))
        .wrapping_add(q(p[2]).wrapping_mul(0xC2B2_AE35));
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    h as f64 / 4_294_967_296.0
}

/// Two orthonormal axes perpendicular to a unit direction `n` (for the sun-aligned
/// orthographic sun-OD frame).
fn perp_basis(n: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let seed = if n[2].abs() < 0.9 {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let u = norm3(cross3(seed, n));
    let v = cross3(n, u);
    (u, v)
}

/// Two real roots `(t0 <= t1)` of `|origin + t*dir| = radius` (dir unit), or `None`.
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

/// The `[t_enter, t_exit]` where a ray is inside the brick shell `[r_inner, r_outer]`
/// and above the inner sphere; `t_exit` is the inner (ground/brick-bottom) hit for a
/// downward ray, else the far outer crossing. `None` if the ray misses the shell.
pub fn ray_shell_segment(
    origin: [f64; 3],
    dir: [f64; 3],
    r_inner: f64,
    r_outer: f64,
) -> Option<(f64, f64)> {
    let (t0_out, t1_out) = ray_sphere(origin, dir, r_outer)?;
    let t_enter = t0_out.max(0.0);
    let mut t_exit = t1_out;
    if let Some((t0_in, _)) = ray_sphere(origin, dir, r_inner)
        && t0_in > t_enter
        && t0_in < t_exit
    {
        t_exit = t0_in;
    }
    if t_exit <= t_enter {
        return None;
    }
    Some((t_enter, t_exit))
}

// ── decoded cloud volume ─────────────────────────────────────────────────────

/// A brick decoded to physical extinction (m^-1) + tau_up (optical depth above),
/// ready for the march. The three extinction classes stay separate so the phase mix
/// is per sample. Index `(k*ny + j)*nx + i` (same as the brick).
#[derive(Debug, Clone)]
pub struct DecodedVolume {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub z_min_m: f64,
    pub dz_m: f64,
    /// Horizontal cell size (m) — the finest of dx/dy; drives the march step pitch.
    pub horiz_pitch_m: f64,
    pub ext_liquid: Vec<f32>,
    pub ext_ice: Vec<f32>,
    pub ext_precip: Vec<f32>,
    pub tau_up: Vec<f32>,
}

/// One trilinearly-sampled cloud voxel (physical extinction, m^-1).
#[derive(Debug, Clone, Copy, Default)]
pub struct CloudSample {
    pub ext_liquid: f64,
    pub ext_ice: f64,
    pub ext_precip: f64,
    pub tau_up: f64,
}

impl CloudSample {
    /// Total extinction = scattering (SSA = 1 in the visible).
    #[inline]
    pub fn total_ext(&self) -> f64 {
        self.ext_liquid + self.ext_ice + self.ext_precip
    }
}

impl DecodedVolume {
    /// Decode a brick's log-quantized channels to physical extinction (m^-1). The
    /// three extinction classes are decoded via the per-volume `LogQuant` scales; the
    /// tau_up channel likewise. `horiz_pitch_m` is the WRF horizontal cell size (m)
    /// used for the march step pitch (min of dx/dy; the caller passes it, since the
    /// brick itself does not carry the projection spacing).
    pub fn from_brick(brick: &VolumeBrick, horiz_pitch_m: f64) -> Self {
        let ql = brick.quant.get("ext_liquid");
        let qi = brick.quant.get("ext_ice");
        let qp = brick.quant.get("ext_precip");
        let qt = brick.quant.get("tau_up");
        Self {
            nx: brick.nx,
            ny: brick.ny,
            nz: brick.nz,
            z_min_m: brick.z_min_m,
            dz_m: brick.dz_m,
            horiz_pitch_m,
            ext_liquid: brick.ext_liquid.iter().map(|&c| ql.decode(c)).collect(),
            ext_ice: brick.ext_ice.iter().map(|&c| qi.decode(c)).collect(),
            ext_precip: brick.ext_precip.iter().map(|&c| qp.decode(c)).collect(),
            tau_up: brick.tau_up.iter().map(|&c| qt.decode(c)).collect(),
        }
    }

    #[inline]
    fn cell(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.ny + j) * self.nx + i
    }

    /// Total extinction (m^-1) at an integer cell (for the occupancy mip build).
    #[inline]
    pub fn total_ext_cell(&self, i: usize, j: usize, k: usize) -> f64 {
        let c = self.cell(i, j, k);
        self.ext_liquid[c] as f64 + self.ext_ice[c] as f64 + self.ext_precip[c] as f64
    }

    /// Trilinearly sample at fractional grid coords `(fi, fj, fk)`. Outside the brick
    /// (any axis out of `[0, n-1]`, or non-finite) returns zero extinction — the
    /// honest answer: no WRF cloud data there (design section 2 zero-extrapolation).
    /// The RAW (un-granulated) field: identical to `sample_granulated(.., None)`.
    #[inline]
    pub fn sample(&self, fi: f64, fj: f64, fk: f64) -> CloudSample {
        self.sample_granulated(fi, fj, fk, None)
    }

    /// [`Self::sample`] with the optional sub-grid GRANULATION erosion applied (see the
    /// granulation section at the top of this module). `None` is byte-identical to the
    /// raw trilinear sample (same operations, same order — the off-flag anchor test pins
    /// it). With `Some`, the three EXTINCTION channels are scaled by the remap-style
    /// erosion multiplier ([`granulation_multiplier`]) of the sample's relative density
    /// `d = total / (max total over its 8 trilinear-support corners)` under the
    /// deterministic erosion field `e = amplitude/GRAN_AMP_CAP * gate *
    /// interior_protection(d) * noise`; `tau_up` is never eroded (an ingest-time
    /// column integral — documented approximation).
    /// Subtract-only: the eroded sample never exceeds the raw one; zero stays zero.
    pub fn sample_granulated(
        &self,
        fi: f64,
        fj: f64,
        fk: f64,
        granulation: Option<Granulation>,
    ) -> CloudSample {
        if !(fi.is_finite() && fj.is_finite() && fk.is_finite())
            || fi < 0.0
            || fj < 0.0
            || fk < 0.0
            || fi > (self.nx - 1) as f64
            || fj > (self.ny - 1) as f64
            || fk > (self.nz - 1) as f64
        {
            return CloudSample::default();
        }
        let i0 = fi.floor() as usize;
        let j0 = fj.floor() as usize;
        let k0 = fk.floor() as usize;
        let i1 = (i0 + 1).min(self.nx - 1);
        let j1 = (j0 + 1).min(self.ny - 1);
        let k1 = (k0 + 1).min(self.nz - 1);
        let ti = fi - i0 as f64;
        let tj = fj - j0 as f64;
        let tk = fk - k0 as f64;
        // The 8 trilinear-support corners, in the exact order the lerp consumes them
        // (c00 pair, c10 pair, c01 pair, c11 pair) — the lerp arithmetic below is
        // bit-identical to the pre-granulation sampler.
        let idx = [
            self.cell(i0, j0, k0),
            self.cell(i1, j0, k0),
            self.cell(i0, j1, k0),
            self.cell(i1, j1, k0),
            self.cell(i0, j0, k1),
            self.cell(i1, j0, k1),
            self.cell(i0, j1, k1),
            self.cell(i1, j1, k1),
        ];
        let trilerp = |ch: &[f32]| -> f64 {
            let g = |n: usize| ch[idx[n]] as f64;
            let c00 = g(0) * (1.0 - ti) + g(1) * ti;
            let c10 = g(2) * (1.0 - ti) + g(3) * ti;
            let c01 = g(4) * (1.0 - ti) + g(5) * ti;
            let c11 = g(6) * (1.0 - ti) + g(7) * ti;
            let c0 = c00 * (1.0 - tj) + c10 * tj;
            let c1 = c01 * (1.0 - tj) + c11 * tj;
            c0 * (1.0 - tk) + c1 * tk
        };
        let mut s = CloudSample {
            ext_liquid: trilerp(&self.ext_liquid),
            ext_ice: trilerp(&self.ext_ice),
            ext_precip: trilerp(&self.ext_precip),
            tau_up: trilerp(&self.tau_up),
        };
        let Some(g) = granulation else {
            return s;
        };
        if g.amplitude <= 0.0 {
            return s;
        }
        let total = s.total_ext();
        if total <= 0.0 {
            return s;
        }
        // Species/height gate (cheap) before the noise (the expensive part).
        let z_msl = self.z_min_m + fk * self.dz_m;
        let gate = granulation_gate(s.ext_liquid, s.ext_ice, s.ext_precip, z_msl);
        if gate <= 0.0 {
            return s;
        }
        // Relative density vs the trilinear-support neighbourhood max (convexity of the
        // trilerp guarantees d <= 1; total > 0 guarantees corner_max > 0), and the
        // interior protection — both cheap, both before the noise: a solid-deck
        // interior sample (d >= GRAN_INTERIOR_HI) skips the noise entirely.
        let mut corner_max = 0.0f64;
        for &c in &idx {
            let t = self.ext_liquid[c] as f64 + self.ext_ice[c] as f64 + self.ext_precip[c] as f64;
            if t > corner_max {
                corner_max = t;
            }
        }
        if corner_max <= 0.0 {
            return s;
        }
        let d = total / corner_max;
        let protection = granulation_interior_protection(d);
        if protection <= 0.0 {
            return s;
        }
        // The deterministic erosion field, anchored in brick-plane metres (the same
        // (fi, fj) — and so the same erosion — for a physical point regardless of the
        // view/ray that sampled it).
        let pitch = self.horiz_pitch_m.max(1.0);
        let noise = granulation_erosion_noise(fi * pitch, fj * pitch);
        let e = (g.amplitude * GRAN_EROSION_GAIN * gate * protection * noise).min(GRAN_EROSION_MAX);
        if e <= 0.0 {
            return s;
        }
        let m = granulation_multiplier(d, e);
        if m < 1.0 {
            s.ext_liquid *= m;
            s.ext_ice *= m;
            s.ext_precip *= m;
        }
        s
    }

    /// Top-of-brick ECEF radius (m).
    #[inline]
    pub fn r_top(&self) -> f64 {
        R_GROUND_M + self.z_min_m + self.nz as f64 * self.dz_m
    }

    /// Bottom-of-brick ECEF radius (m).
    #[inline]
    pub fn r_bottom(&self) -> f64 {
        R_GROUND_M + self.z_min_m
    }

    /// The march step pitch (m): the finest of the vertical dz and horizontal cell.
    #[inline]
    pub fn voxel_pitch_m(&self) -> f64 {
        self.dz_m.min(self.horiz_pitch_m).max(1.0)
    }
}

/// ECEF point -> fractional brick coords `(fi, fj, fk)` + radius (the design section 1
/// per-step transform: closed-form ECEF -> spherical lat/lon/h -> projection forward).
#[inline]
pub fn ecef_to_brick(
    p: [f64; 3],
    georef: &GridGeoref,
    z_min_m: f64,
    dz_m: f64,
) -> (f64, f64, f64, f64) {
    let r = len3(p);
    let h = r - R_GROUND_M;
    let fk = (h - z_min_m) / dz_m;
    let lat = (p[2] / r).clamp(-1.0, 1.0).asin().to_degrees();
    let lon = p[1].atan2(p[0]).to_degrees();
    let (fi, fj) = georef.forward(lat, lon);
    (fi, fj, fk, r)
}

/// Fractional brick coords -> ECEF point (inverse of [`ecef_to_brick`]); `None` if the
/// projection inverse fails. The round-trip of these two is the M4 companion of the
/// M0 projection ratchet.
pub fn brick_to_ecef(
    georef: &GridGeoref,
    i: f64,
    j: f64,
    k: f64,
    z_min_m: f64,
    dz_m: f64,
) -> Option<[f64; 3]> {
    let (lat, lon) = georef.inverse(i, j)?;
    let r = R_GROUND_M + z_min_m + k * dz_m;
    let (la, lo) = (lat.to_radians(), lon.to_radians());
    Some([
        r * la.cos() * lo.cos(),
        r * la.cos() * lo.sin(),
        r * la.sin(),
    ])
}

// ── occupancy mip (coarse empty-space skipping) ──────────────────────────────

/// A low-res max-extinction mip of the volume for coarse empty-space skipping. Each
/// block holds the MAX total extinction of its voxels AND of its 26 neighbouring
/// blocks (a one-block DILATION): a block is "occupied" iff any voxel in it or in a
/// neighbouring block has extinction > 0. Dilation is what stops the march from
/// coarse-stepping over the half-voxel trilinear cloud "skirt" that leaks one voxel
/// across a block boundary onto the empty side (M4 review FINDING 2 — the faint
/// 8-voxel-periodic edge thinning). It only ever ADDS occupancy, so the conservative
/// guarantee (never skip a non-empty voxel) is preserved, and it also converges the CPU
/// path with the GPU twin, whose linear-filtered occupancy fetch already dilates
/// (FINDING 5c). The `occupancy_mip_is_conservative_and_dilated` test asserts both.
#[derive(Debug, Clone)]
pub struct OccupancyMip {
    pub mx: usize,
    pub my: usize,
    pub mz: usize,
    pub factor: usize,
    /// Max total extinction (m^-1) per block, index `(kz*my + jy)*mx + ix`.
    pub maxext: Vec<f32>,
}

impl OccupancyMip {
    /// Build a `factor`-downsampled, one-block-DILATED max-extinction mip of `vol`.
    pub fn build(vol: &DecodedVolume, factor: usize) -> Self {
        let factor = factor.max(1);
        let mx = vol.nx.div_ceil(factor);
        let my = vol.ny.div_ceil(factor);
        let mz = vol.nz.div_ceil(factor);
        // Raw per-block max extinction.
        let mut raw = vec![0.0f32; mx * my * mz];
        for k in 0..vol.nz {
            let kb = k / factor;
            for j in 0..vol.ny {
                let jb = j / factor;
                for i in 0..vol.nx {
                    let ib = i / factor;
                    let e = vol.total_ext_cell(i, j, k) as f32;
                    let o = (kb * my + jb) * mx + ib;
                    if e > raw[o] {
                        raw[o] = e;
                    }
                }
            }
        }
        // Dilate by one block (26-neighbourhood incl. self): mark a block occupied if
        // any neighbour has extinction, so the trilinear skirt on the empty-facing side
        // of an occupied boundary is fine-stepped, never coarse-skipped (FINDING 2).
        let mut maxext = vec![0.0f32; mx * my * mz];
        for kb in 0..mz {
            for jb in 0..my {
                for ib in 0..mx {
                    let mut m = 0.0f32;
                    for dk in -1i64..=1 {
                        let nk = kb as i64 + dk;
                        if nk < 0 || nk as usize >= mz {
                            continue;
                        }
                        for dj in -1i64..=1 {
                            let nj = jb as i64 + dj;
                            if nj < 0 || nj as usize >= my {
                                continue;
                            }
                            for di in -1i64..=1 {
                                let ni = ib as i64 + di;
                                if ni < 0 || ni as usize >= mx {
                                    continue;
                                }
                                let e = raw[(nk as usize * my + nj as usize) * mx + ni as usize];
                                if e > m {
                                    m = e;
                                }
                            }
                        }
                    }
                    maxext[(kb * my + jb) * mx + ib] = m;
                }
            }
        }
        Self {
            mx,
            my,
            mz,
            factor,
            maxext,
        }
    }

    /// Max extinction of the block containing fractional voxel `(fi, fj, fk)`.
    ///
    /// GUARD BAND (WS1 march-physics pass): a probe within one mip block
    /// ([`OccupancyMip::factor`] cells) OUTSIDE the volume reads the nearest EDGE
    /// block — conservative-occupied near the boundary. The pre-WS1 hard zero for
    /// any out-of-range probe (even one metre outside) let a coarse-stepping march
    /// jump across the domain boundary and skip up to a coarse step of EDGE CLOUD
    /// unsampled at a side entry (the dilation could not help: it only marks
    /// in-range blocks). The guard band exceeds every coarse step (2x pitch cloud
    /// march, 4x pitch IR march, vs `factor >= 4` cells), so entries are always
    /// fine-stepped. Beyond the guard band: 0 (far empty space coarse-skips as
    /// before). This only SIZES march steps — the volume sampler stays
    /// zero-outside, so no data smears out of the domain.
    #[inline]
    pub fn maxext_at(&self, fi: f64, fj: f64, fk: f64) -> f32 {
        if !(fi.is_finite() && fj.is_finite() && fk.is_finite()) {
            return 0.0;
        }
        let guard = self.factor as f64;
        let block = |f: f64, blocks: usize| -> Option<usize> {
            if f < -guard || f > (blocks * self.factor) as f64 + guard {
                return None;
            }
            Some((f.max(0.0) as usize / self.factor).min(blocks.saturating_sub(1)))
        };
        match (block(fi, self.mx), block(fj, self.my), block(fk, self.mz)) {
            (Some(ib), Some(jb), Some(kb)) => self.maxext[(kb * self.my + jb) * self.mx + ib],
            _ => 0.0,
        }
    }

    /// Flatten to `mx*my*mz` R8 bytes for a GPU `R8Unorm` 3-D upload: 255 where the
    /// block is occupied (any extinction), 0 where empty. Conservative by
    /// construction (the shader coarse-steps only where this is 0).
    pub fn to_r8_occupancy(&self) -> Vec<u8> {
        self.maxext
            .iter()
            .map(|&e| if e > 0.0 { 255u8 } else { 0u8 })
            .collect()
    }
}

// ── sun optical-depth map (design section 6) ─────────────────────────────────

/// A sun-aligned orthographic optical-depth map: texel `(u, v)` holds the TOTAL
/// optical depth of the brick column along the sun ray. Consumer: cloud shadows on the
/// ground `T = e^-od` (the whole column IS the cloud between the ground and the sun, so
/// the total-column value is correct here). The map is anchored at the brick centre
/// with axes `au, av` perpendicular to the sun. NOTE it is NOT used for the in-cloud
/// sun transmittance any more — a 2-D total-column scalar cannot give a per-depth
/// transmittance, which killed the direct-sun term for thick clouds (M4 review FINDING
/// 1); that now uses the depth-resolved secondary light march in
/// [`cloud_sun_optical_depth`].
///
/// M5 adds `occ_dist` (per texel: the extinction-weighted mean SLANT distance from the
/// ground to the occluding cloud along the sun ray) so [`SunOdMap::penumbral_shadow`]
/// can widen the ground shadow's penumbra with occluder height (design section 6:
/// blur radius = occluder distance x tan 0.533 deg). The map's `(au, av)` plane IS
/// perpendicular to the sun, so a blur of that radius in map metres is the physically
/// correct disk-of-sun soft edge (a named approximation: pre-blur vs disk-sampling the
/// volume).
#[derive(Debug, Clone)]
pub struct SunOdMap {
    pub width: usize,
    pub height: usize,
    pub od: Vec<f32>,
    /// Extinction-weighted mean occluder slant distance (m) per texel; 0 where clear.
    pub occ_dist: Vec<f32>,
    center: [f64; 3],
    au: [f64; 3],
    av: [f64; 3],
    u_min: f64,
    u_max: f64,
    v_min: f64,
    v_max: f64,
}

/// EDGE FEATHER width (texels) applied to the accumulated sun-OD map's outer band
/// (WS1 march-physics pass, with the out-of-extent contract on
/// [`SunOdMap::sample_uv`]): the column optical depth ramps smoothly to zero over
/// the outermost texels, so the ground-shadow field is CONTINUOUS across the map
/// boundary — outside the extent the consumers now read 0 (clear), and without the
/// feather a cloud column landing exactly in an edge texel would produce a hard
/// shadow-to-clear step at the boundary line. Interior texels (deeper than this
/// many texels from the edge) are byte-identical to the raw accumulation.
pub const SUN_OD_EDGE_FEATHER_TEXELS: f64 = 1.5;

/// The sun-OD edge-feather weight for texel `(tx, ty)` of a `width x height` map:
/// a smoothstep of the texel's distance to the nearest map edge over
/// `feather_texels`; `1.0` in the interior, `0.0` on the outermost texel ring.
/// `feather_texels <= 0` -> `1.0` everywhere (the neutral no-op).
#[inline]
fn sun_od_edge_weight(
    tx: usize,
    ty: usize,
    width: usize,
    height: usize,
    feather_texels: f64,
) -> f64 {
    if feather_texels <= 0.0 {
        return 1.0;
    }
    let d = tx
        .min(width.saturating_sub(1).saturating_sub(tx))
        .min(ty.min(height.saturating_sub(1).saturating_sub(ty))) as f64;
    let t = (d / feather_texels).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Accumulate the sun-OD map for a volume + sun direction (design section 6, consumer
/// (a)+(b)). `resolution` is the square map side (design target 1024). CPU REFERENCE
/// of the compute-pass accumulation (`sun_od.wgsl` is the naga-validated GPU twin).
/// Applies the default edge feather ([`SUN_OD_EDGE_FEATHER_TEXELS`]); the public
/// 4-arg signature is unchanged — the feather width is an internal parameter of the
/// delegating [`accumulate_sun_od_feathered`].
pub fn accumulate_sun_od(
    vol: &DecodedVolume,
    georef: &GridGeoref,
    sun_ecef: [f64; 3],
    resolution: usize,
) -> SunOdMap {
    accumulate_sun_od_feathered(
        vol,
        georef,
        sun_ecef,
        resolution,
        SUN_OD_EDGE_FEATHER_TEXELS,
    )
}

/// [`accumulate_sun_od`] with an explicit edge-feather width in texels. `0.0`
/// disables the feather — the map is then byte-identical to the raw (pre-WS1)
/// accumulation (the band-0 anchor test pins this). The feather scales only the
/// `od` channel (an ADDITIVE column quantity); `occ_dist` is an extinction-weighted
/// MEAN distance, which stays meaningful unscaled (it only sets the penumbra blur
/// radius, and a feathered od already fades the shadow itself).
pub fn accumulate_sun_od_feathered(
    vol: &DecodedVolume,
    georef: &GridGeoref,
    sun_ecef: [f64; 3],
    resolution: usize,
    feather_texels: f64,
) -> SunOdMap {
    accumulate_sun_od_granulated(vol, georef, sun_ecef, resolution, feather_texels, None)
}

/// [`accumulate_sun_od_feathered`] over the optionally-GRANULATED extinction field:
/// pass the SAME [`Granulation`] as [`MarchConfig::granulation`] so the ground cloud
/// shadows come from the SAME eroded field the view/sun marches sample (a granulated
/// cumulus field casts granulated shadows, not the solid blob's). `None` is
/// byte-identical to [`accumulate_sun_od_feathered`].
pub fn accumulate_sun_od_granulated(
    vol: &DecodedVolume,
    georef: &GridGeoref,
    sun_ecef: [f64; 3],
    resolution: usize,
    feather_texels: f64,
    granulation: Option<Granulation>,
) -> SunOdMap {
    let resolution = resolution.max(1);
    let sun = norm3(sun_ecef);
    let (au, av) = perp_basis(sun);
    let ci = (vol.nx - 1) as f64 / 2.0;
    let cj = (vol.ny - 1) as f64 / 2.0;
    let ck = (vol.nz - 1) as f64 / 2.0;
    let center =
        brick_to_ecef(georef, ci, cj, ck, vol.z_min_m, vol.dz_m).unwrap_or([R_GROUND_M, 0.0, 0.0]);

    // Extent from the 8 brick corners projected onto (au, av, sun).
    let (mut u_min, mut u_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut v_min, mut v_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut s_min, mut s_max) = (f64::INFINITY, f64::NEG_INFINITY);
    for &ki in &[0.0, (vol.nz - 1) as f64] {
        for &ji in &[0.0, (vol.ny - 1) as f64] {
            for &ii in &[0.0, (vol.nx - 1) as f64] {
                if let Some(p) = brick_to_ecef(georef, ii, ji, ki, vol.z_min_m, vol.dz_m) {
                    let d = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
                    let (u, v, s) = (dot3(d, au), dot3(d, av), dot3(d, sun));
                    u_min = u_min.min(u);
                    u_max = u_max.max(u);
                    v_min = v_min.min(v);
                    v_max = v_max.max(v);
                    s_min = s_min.min(s);
                    s_max = s_max.max(s);
                }
            }
        }
    }
    if !(u_min.is_finite() && v_min.is_finite() && s_min.is_finite()) {
        // Degenerate (projection failed at every corner): an all-zero map.
        return SunOdMap {
            width: resolution,
            height: resolution,
            od: vec![0.0; resolution * resolution],
            occ_dist: vec![0.0; resolution * resolution],
            center,
            au,
            av,
            u_min: -1.0,
            u_max: 1.0,
            v_min: -1.0,
            v_max: 1.0,
        };
    }

    let pitch = vol.voxel_pitch_m();
    let margin = pitch * 4.0;
    let s_start = s_max + margin;
    let s_len = (s_max - s_min) + 2.0 * margin;
    // Target one sample per voxel pitch along the sun ray so a thin (1-2 voxel) layer
    // is not stepped over. On a wide domain at a low sun the along-sun span is huge and
    // hits the cap; the cap is 1024 (raised from 512 — M4 review FINDING 3) so the
    // worst-case ds roughly halves. The map now feeds ONLY the ground cloud-shadow
    // (the in-cloud sun term uses the secondary light march, FINDING 1), so this
    // resolution governs ground-shadow fidelity of thin cirrus, not cloud lighting.
    let n_steps = ((s_len / pitch).ceil() as usize).clamp(1, 1024);
    let ds = s_len / n_steps as f64;

    // Rows in parallel (embarrassingly parallel over texels; on the below-normal
    // worker for the studio, and release-profile in the fixture test). Each texel
    // accumulates the column optical depth AND the extinction-weighted mean occluder
    // slant distance (for the M5 penumbra).
    let rows: Vec<(Vec<f32>, Vec<f32>)> = (0..resolution)
        .into_par_iter()
        .map(|ty| {
            let v = v_min + (ty as f64 + 0.5) / resolution as f64 * (v_max - v_min);
            let mut od_row = vec![0.0f32; resolution];
            let mut dist_row = vec![0.0f32; resolution];
            for (tx, (od_cell, dist_cell)) in od_row.iter_mut().zip(dist_row.iter_mut()).enumerate()
            {
                let u = u_min + (tx as f64 + 0.5) / resolution as f64 * (u_max - u_min);
                // Start on the sun side, march away from the sun through the column.
                let start = add3(add3(center, scl3(au, u)), scl3(av, v));
                let start = madd3(start, sun, s_start);
                let mut acc = 0.0f64;
                let mut dist_weighted = 0.0f64;
                for step in 0..n_steps {
                    let t = (step as f64 + 0.5) * ds;
                    let p = madd3(start, sun, -t);
                    let (fi, fj, fk, r) = ecef_to_brick(p, georef, vol.z_min_m, vol.dz_m);
                    let ext = vol.sample_granulated(fi, fj, fk, granulation).total_ext();
                    if ext > 0.0 {
                        acc += ext * ds;
                        // Slant distance from this occluder sample down to the ground
                        // along the sun ray ~= height above ground / sin(local sun
                        // elevation). Clamp the sine so a near-horizon sun does not blow
                        // the distance up unboundedly.
                        let h = (r - R_GROUND_M).max(0.0);
                        let mu = dot3(scl3(p, 1.0 / r), sun).max(0.05);
                        dist_weighted += ext * ds * (h / mu);
                    }
                }
                *od_cell = acc as f32;
                *dist_cell = if acc > 0.0 {
                    (dist_weighted / acc) as f32
                } else {
                    0.0
                };
            }
            (od_row, dist_row)
        })
        .collect();
    let mut od = Vec::with_capacity(resolution * resolution);
    let mut occ_dist = Vec::with_capacity(resolution * resolution);
    for (od_row, dist_row) in rows {
        od.extend(od_row);
        occ_dist.extend(dist_row);
    }
    // Edge feather (WS1): ramp the outer band's od to zero so the shadow field is
    // continuous across the map boundary (see SUN_OD_EDGE_FEATHER_TEXELS). Interior
    // texels are untouched; feather 0 leaves the whole map byte-identical.
    if feather_texels > 0.0 {
        for ty in 0..resolution {
            for tx in 0..resolution {
                let w = sun_od_edge_weight(tx, ty, resolution, resolution, feather_texels);
                if w < 1.0 {
                    od[ty * resolution + tx] *= w as f32;
                }
            }
        }
    }
    SunOdMap {
        width: resolution,
        height: resolution,
        od,
        occ_dist,
        center,
        au,
        av,
        u_min,
        u_max,
        v_min,
        v_max,
    }
}

impl SunOdMap {
    /// Bilinear sample of a channel at sun-plane coordinates `(u, v)` in metres.
    ///
    /// OUT-OF-EXTENT CONTRACT (WS1 march-physics pass): a point outside
    /// `[u_min, u_max] x [v_min, v_max]` (with a half-texel tolerance) returns `0.0`
    /// for BOTH channels — the map's extent covers the whole brick, so there is no
    /// cloud column out there. The previous clamp-to-edge read handed every
    /// out-of-extent ground point the nearest EDGE texel's column, which smeared a
    /// domain-edge cloud's shadow across the entire zoom-out margin strip.
    fn sample_uv(&self, chan: &[f32], u: f64, v: f64) -> f64 {
        let su = self.u_max - self.u_min;
        let sv = self.v_max - self.v_min;
        if su <= 0.0 || sv <= 0.0 {
            return 0.0;
        }
        let tol_u = 0.5 * su / self.width.max(1) as f64;
        let tol_v = 0.5 * sv / self.height.max(1) as f64;
        if u < self.u_min - tol_u
            || u > self.u_max + tol_u
            || v < self.v_min - tol_v
            || v > self.v_max + tol_v
        {
            return 0.0;
        }
        let fu =
            ((u - self.u_min) / su * self.width as f64 - 0.5).clamp(0.0, (self.width - 1) as f64);
        let fv =
            ((v - self.v_min) / sv * self.height as f64 - 0.5).clamp(0.0, (self.height - 1) as f64);
        let x0 = fu.floor() as usize;
        let y0 = fv.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let tx = fu - x0 as f64;
        let ty = fv - y0 as f64;
        let g = |x: usize, y: usize| chan[y * self.width + x] as f64;
        let a = g(x0, y0) * (1.0 - tx) + g(x1, y0) * tx;
        let b = g(x0, y1) * (1.0 - tx) + g(x1, y1) * tx;
        a * (1.0 - ty) + b * ty
    }

    /// The sun-plane `(u, v)` metre coordinates of an ECEF point.
    #[inline]
    fn plane_uv(&self, p: [f64; 3]) -> (f64, f64) {
        let d = [
            p[0] - self.center[0],
            p[1] - self.center[1],
            p[2] - self.center[2],
        ];
        (dot3(d, self.au), dot3(d, self.av))
    }

    /// Sample the total column optical depth at an ECEF point (bilinear; 0 outside the
    /// map extent).
    pub fn sample(&self, p: [f64; 3]) -> f64 {
        let (u, v) = self.plane_uv(p);
        self.sample_uv(&self.od, u, v)
    }

    /// Sample the extinction-weighted mean occluder slant distance (m) at an ECEF point.
    pub fn sample_occ_dist(&self, p: [f64; 3]) -> f64 {
        let (u, v) = self.plane_uv(p);
        self.sample_uv(&self.occ_dist, u, v)
    }

    /// The PENUMBRAL ground cloud-shadow transmittance at an ECEF ground point (design
    /// section 6, M5): the sun-visibility fraction with a physically soft, distance-
    /// widening edge. The penumbra blur radius = the occluder's slant distance x
    /// `tan(0.533 deg)` (the sun disk's angular diameter projected onto the sun plane,
    /// which is exactly this map's `(u, v)` plane). We average the Beer transmittance
    /// over a small disk of that radius — a named approximation (pre-blur instead of
    /// disk-sampling the volume): a higher cloud (larger `occ_dist`) yields a wider,
    /// softer penumbra; a ground-hugging cloud stays sharp; a clear column stays 1.
    pub fn penumbral_shadow(&self, p: [f64; 3]) -> f64 {
        let (u, v) = self.plane_uv(p);
        let occ_dist = self.sample_uv(&self.occ_dist, u, v);
        let radius = occ_dist * (SUN_ANGULAR_DIAMETER_RAD.tan());
        // Below ~one texel of blur there is no penumbra to resolve: sharp Beer shadow.
        let texel = ((self.u_max - self.u_min).abs() / self.width.max(1) as f64)
            .max((self.v_max - self.v_min).abs() / self.height.max(1) as f64);
        if radius <= 0.5 * texel {
            return beer(self.sample_uv(&self.od, u, v));
        }
        // A centre tap + two rings of taps over the blur disk, transmittance-averaged
        // (the penumbra is a partial occlusion of the sun DISK, so it softens in
        // transmittance, not optical-depth, space).
        const RING: usize = 8;
        let mut sum = beer(self.sample_uv(&self.od, u, v));
        let mut wsum = 1.0f64;
        for (ri, &rr) in [0.5, 1.0].iter().enumerate() {
            let w = if ri == 0 { 1.0 } else { 0.6 };
            for k in 0..RING {
                let ang = (k as f64 + 0.5) / RING as f64 * 2.0 * PI;
                let du = radius * rr * ang.cos();
                let dv = radius * rr * ang.sin();
                sum += w * beer(self.sample_uv(&self.od, u + du, v + dv));
                wsum += w;
            }
        }
        sum / wsum
    }

    /// Flatten to `width*height` R32Float for a GPU upload (the map is sampled by the
    /// cloud/surface passes for shadows + long-range sun transmittance).
    pub fn to_r32f(&self) -> Vec<f32> {
        self.od.clone()
    }

    /// Flatten the occluder-distance channel to `width*height` R32Float for the GPU
    /// penumbra mirror upload.
    pub fn occ_dist_to_r32f(&self) -> Vec<f32> {
        self.occ_dist.clone()
    }
}

// ── the march ────────────────────────────────────────────────────────────────

/// Step quality — the two design step ceilings (section 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepQuality {
    /// 192 primary steps (interactive preview).
    Interactive,
    /// 384 primary steps (offline / stored frame — full quality).
    Offline,
}

impl StepQuality {
    pub fn max_steps(self) -> usize {
        match self {
            Self::Interactive => 192,
            Self::Offline => 384,
        }
    }
    pub fn code(self) -> f32 {
        match self {
            Self::Interactive => 0.0,
            Self::Offline => 1.0,
        }
    }
}

/// Cloud march tuning (design section 4).
#[derive(Debug, Clone, Copy)]
pub struct MarchConfig {
    /// Coarse-step multiplier of the voxel pitch through empty space (~2x).
    pub coarse_mult: f64,
    /// Fine-step multiplier inside cloud (~0.5x).
    pub fine_mult: f64,
    /// Hard primary-step cap (192 interactive / 384 offline).
    pub max_steps: usize,
    /// Number of exponentially-spaced steps in the secondary sun march (design
    /// section 4). This is the DEPTH-RESOLVED cloud self-shadow: from each in-cloud
    /// sample the march accumulates the real extinction along the sun ray FROM THAT
    /// SAMPLE toward the top of the cloud, so a thick-anvil top (little cloud above it
    /// toward the sun) is near-fully sunlit while the base (whole cloud above it) is
    /// shadowed. Replaces the fix2 depth-blind total-column sun-OD-map term (M4 review
    /// FINDING 1); the orthographic sun-OD map is retained ONLY for the ground-shadow
    /// consumer, where the whole-column value is correct.
    pub sun_march_steps: usize,
    /// Base (first) sun-march step length (m); each subsequent step grows by
    /// `sun_march_growth`. Defaults to the voxel pitch so the near field is resolved.
    pub sun_march_step_m: f64,
    /// Growth factor of the exponentially-spaced sun-march steps.
    pub sun_march_growth: f64,
    /// Stratified-sampling jitter amplitude for the secondary sun march, `[0, 1]`
    /// (WS1 march-physics pass): each ray samples its exponential segments at a
    /// deterministic hash-offset point (`0.5 + amp*(hash01_position(p) - 0.5)` of
    /// the segment) instead of the fixed midpoint, decorrelating schedule banding.
    /// `0.0` reproduces the fixed-midpoint march exactly (the neutrality test pins
    /// it); default [`SUN_MARCH_JITTER_AMP`].
    pub sun_march_jitter_amp: f64,
    /// Number of Wrenninge/Oz multi-scatter octaves on the sun term (design section 4,
    /// M5). `1` = the fix2 single scatter; `DEFAULT_OCTAVES` = the multi-scatter look.
    /// See [`octave_sun_source`] and the octave-constants block.
    pub octaves: usize,
    /// Apply the Schneider beer-powder stylization to the sun term. **OFF by default
    /// as of M5**: beer-powder was a stylization that FAKED the missing forward-scatter
    /// buildup by darkening thin faces; the octaves now supply that buildup for real,
    /// so leaving powder on would double-darken the very faces the octaves brighten
    /// (design section 4, M5 beer-powder decision). Kept as a toggle; when on it is
    /// applied per octave and can only darken (bounded above by Beer for all tau).
    pub beer_powder: bool,
    /// Ground albedo for the ambient ground-bounce estimate.
    pub ground_albedo: f64,
    /// Early-out view-transmittance floor (stop when the cloud is essentially opaque).
    pub transmittance_floor: f64,
    /// GROUND LIFT (top-down/basemap appearance pass, [`crate::render::GROUND_DAY_LIFT`]):
    /// the sun-gated daytime surface-brightness lift passed to
    /// [`crate::render::surface_toa_radiance`] by the cloud/top-down composite. Default =
    /// the baked `GROUND_DAY_LIFT`; the `render_frame` `ground-gain=` knob overrides it.
    /// `1.0` = neutral no-op.
    pub ground_day_lift: f64,
    /// CLOUD/HIGHLIGHT SOFT-CLIP knee ([`crate::render::CLOUD_SOFTCLIP_KNEE`]): the
    /// Reinhard highlight shoulder knee the cloud/top-down RGB tonemap
    /// ([`crate::render::radiance_to_rgba_softclip`]) uses so bright cloud tops keep
    /// structure. Default = the baked `CLOUD_SOFTCLIP_KNEE`; the `render_frame`
    /// `cloud-softclip=` knob overrides it. `1.0` = disables the shoulder (hard clamp).
    pub cloud_softclip_knee: f64,
    /// TOP-DOWN CLOUD NORMALIZATION ([`crate::topdown::TOPDOWN_CLOUD_NORM`]): the
    /// sun-gated multiplier on the top-down cloud radiance (fixes the near-nadir "white
    /// square"; the geostationary path ignores it). Default = the baked
    /// `TOPDOWN_CLOUD_NORM`; the `render_frame` `topdown-cloudnorm=` knob overrides it.
    /// `1.0` = neutral no-op.
    pub topdown_cloud_norm: f64,
    /// EDGE FEATHER band width in CELLS (see [`edge_feather`] /
    /// [`edge_feather_cells_for_margin`]): the outer band of the domain over which the
    /// cloud contribution ramps to zero, so clouds fade into a zoom-out margin instead of
    /// a hard cutoff. `0.0` = neutral no-op (no feather — set when there is no margin, so
    /// the render is byte-identical to before). Set by the render assembly from the margin.
    pub edge_feather_cells: f64,
    /// Sub-grid cloud GRANULATION (edge-erosion detail noise; see the granulation section
    /// at the top of this module). `None` (the [`MarchConfig::new`] default) = off,
    /// byte-identical to the pre-granulation march. When `Some`, BOTH the primary view
    /// march and the secondary sun march sample the eroded field
    /// ([`DecodedVolume::sample_granulated`]); the render assembly must pass the SAME
    /// value to [`accumulate_sun_od_granulated`] so the ground shadows agree — every
    /// march of one composite samples the SAME eroded field.
    pub granulation: Option<Granulation>,
}

impl MarchConfig {
    /// Defaults for a step quality and voxel pitch.
    pub fn new(quality: StepQuality, voxel_pitch_m: f64) -> Self {
        // The secondary sun-march schedule follows the step quality (WS1
        // march-physics pass): the offline / stored-frame tier gets the denser,
        // slower-growing (10, 1.5) schedule (finer near field AND ~28 km natural
        // reach); interactive keeps the cheap (6, 2.0). Both are further EXTENDED
        // per sample to the in-shell slant reach by `cloud_sun_optical_depth`.
        let (sun_steps, sun_growth) = match quality {
            StepQuality::Interactive => (SUN_MARCH_STEPS, SUN_MARCH_GROWTH),
            StepQuality::Offline => (SUN_MARCH_STEPS_OFFLINE, SUN_MARCH_GROWTH_OFFLINE),
        };
        Self {
            coarse_mult: 2.0,
            fine_mult: 0.5,
            max_steps: quality.max_steps(),
            sun_march_steps: sun_steps,
            sun_march_step_m: voxel_pitch_m,
            sun_march_growth: sun_growth,
            sun_march_jitter_amp: SUN_MARCH_JITTER_AMP,
            octaves: DEFAULT_OCTAVES,
            beer_powder: false,
            ground_albedo: GROUND_ALBEDO,
            transmittance_floor: 0.003,
            // Appearance-pass baked defaults (the studio's `..MarchConfig::new()` inherits
            // these; the render_frame CLI knobs override them). Edge feather off by default
            // (activated only by a zoom-out margin, via `edge_feather_cells_for_margin`).
            ground_day_lift: GROUND_DAY_LIFT,
            cloud_softclip_knee: CLOUD_SOFTCLIP_KNEE,
            topdown_cloud_norm: crate::topdown::TOPDOWN_CLOUD_NORM,
            edge_feather_cells: 0.0,
            granulation: None,
        }
    }
}

/// The bundle of scene resources one cloud march reads.
pub struct CloudScene<'a> {
    pub vol: &'a DecodedVolume,
    pub mip: &'a OccupancyMip,
    pub sun_od: &'a SunOdMap,
    pub georef: &'a GridGeoref,
    pub luts: &'a AtmosphereLuts,
    /// SH-2 directional sky ambient (M5) — replaces M2's scalar ambient table. Cloud
    /// voxels evaluate its upper-hemisphere irradiance (the sky colour, warm at sunset)
    /// attenuated by `tau_up`/`tau_down`.
    pub sky_sh: &'a SkyShTable,
    /// Unit ECEF sun direction (sun at infinity).
    pub sun_ecef: [f64; 3],
    pub cfg: MarchConfig,
}

/// Result of one cloud march along a view ray.
#[derive(Debug, Clone, Copy)]
pub struct CloudMarch {
    /// In-scattered cloud radiance reaching the camera (per band).
    pub inscatter: [f64; 3],
    /// The DIRECT-SUN part of `inscatter` alone (per band) — the diagnostic that
    /// isolates the sun single-scatter term from the scalar ambient term. Before the
    /// FINDING-1 fix this was ~0 for thick clouds (the sun term was dead); it is the
    /// acceptance measure that the sunlit contribution is now alive. Not used by the
    /// composite (which uses `inscatter`); it is CPU-diagnostic only, so the GPU twin
    /// does not carry it.
    pub sun_inscatter: [f64; 3],
    /// View transmittance through the cloud (scalar — cloud extinction is gray).
    pub transmittance: f64,
    /// Transmittance-weighted mean traversal fraction of the cloud along the ray
    /// within the BRICK shell (the cloud's visual centroid, in `[0,1]`) — a diagnostic
    /// / regression value. NOTE the aerial-perspective froxel is indexed by the
    /// ATMOSPHERE-shell fraction, not this one; use `mean_t_m` for that (see
    /// `shade_cloud_pixel`).
    pub mean_w: f64,
    /// Transmittance-weighted mean ABSOLUTE distance of the cloud along the view ray
    /// from the camera (m). This is the coordinate the aerial-perspective froxel
    /// needs: converting it to the atmosphere-shell traversal fraction the froxel is
    /// indexed by (via [`atmosphere_shell_fraction`]) fixes the M4-review FINDING-4
    /// brick-vs-atmosphere depth mismatch (a 10 km cloud was read as ~50 km airlight).
    pub mean_t_m: f64,
}

impl CloudMarch {
    /// A clear (no-cloud) result.
    pub const CLEAR: CloudMarch = CloudMarch {
        inscatter: [0.0; 3],
        sun_inscatter: [0.0; 3],
        transmittance: 1.0,
        mean_w: 1.0,
        mean_t_m: 0.0,
    };
}

/// The DEPTH-RESOLVED cloud sun optical depth: the optical depth along the sun ray
/// FROM the sample `p` toward the sun (i.e. the cloud between `p` and the sun), by a
/// short secondary light march (the standard Nubis/Frostbite pattern, design section
/// 4). Exponentially-spaced steps (`sun_march_steps` of them, each `sun_march_growth`x
/// the previous, starting at `sun_march_step_m`) sample the real extinction along the
/// sun direction and accumulate `sigma_t * ds`. The near field — which dominates the
/// self-shadow of the sunlit face — is finely resolved; the far tail is cheap and only
/// matters where it has already driven the transmittance to ~0.
///
/// This REPLACES the fix2 depth-blind term (`sun_od.sample(p) * 0.5 + detail taps`),
/// which handed every sample `0.5 *` the WHOLE-column optical depth and so killed the
/// direct-sun term for the top/sun-facing samples of any thick cloud (M4 review
/// FINDING 1). A single 2-D total-column scalar fundamentally cannot give a per-depth
/// transmittance, so the map is no longer consulted here; it survives only for the
/// ground cloud-shadow ([`ground_cloud_shadow`]), where the whole column IS the cloud
/// between the ground and the sun.
fn cloud_sun_optical_depth(scene: &CloudScene, p: [f64; 3]) -> f64 {
    let cfg = &scene.cfg;
    let n = cfg.sun_march_steps.max(1);
    let growth = cfg.sun_march_growth.max(1.0);
    let base = cfg.sun_march_step_m.max(1.0);
    // Deterministic stratified jitter (see MarchConfig::sun_march_jitter_amp): one
    // hash offset per ray applied within every segment; amp 0 = the fixed midpoint.
    let amp = cfg.sun_march_jitter_amp.clamp(0.0, 1.0);
    let offset = if amp > 0.0 {
        0.5 + amp * (hash01_position(p) - 0.5)
    } else {
        0.5
    };
    let mut tau = 0.0f64;
    let mut dist = 0.0f64;
    let mut ds = base;
    for _ in 0..n {
        // Sample within the segment (dist .. dist+ds) at the stratified offset,
        // toward the sun. Samples the SAME (optionally granulated) field as the
        // primary view march, so cloud self-shadowing matches the eroded cloud.
        let pp = madd3(p, scene.sun_ecef, dist + offset * ds);
        let (fi, fj, fk, _) = ecef_to_brick(pp, scene.georef, scene.vol.z_min_m, scene.vol.dz_m);
        tau += scene
            .vol
            .sample_granulated(fi, fj, fk, cfg.granulation)
            .total_ext()
            * ds;
        dist += ds;
        ds *= growth;
    }
    // TAIL EXTENSION (WS1 march-physics pass): the fixed geometric schedule reaches
    // only `base*(g^n - 1)/(g - 1)` of slant (~63x pitch interactive, ~113x
    // offline), so an anvil 20+ km along a low sun ray cast NO shadow at all on the
    // cloud below it. Cover the REMAINING in-shell slant toward the sun (up to
    // `ray_shell_segment`'s exit) with two stratified samples. The near field keeps
    // the EXACT unextended schedule — cloud self-shadow accuracy is never degraded,
    // and a short high-sun column (exit inside the natural reach) is bit-identical
    // to the unextended march. The tail is a coarse, honest, jitter-decorrelated
    // estimate of far occlusion; its reach discontinuity at the horizon-grazing
    // ground/shell-exit switch only moves far samples, not the near field.
    if let Some((_, t_exit)) =
        ray_shell_segment(p, scene.sun_ecef, scene.vol.r_bottom(), scene.vol.r_top())
        && t_exit > dist
    {
        let half = 0.5 * (t_exit - dist);
        for _ in 0..2 {
            let pp = madd3(p, scene.sun_ecef, dist + offset * half);
            let (fi, fj, fk, _) =
                ecef_to_brick(pp, scene.georef, scene.vol.z_min_m, scene.vol.dz_m);
            tau += scene
                .vol
                .sample_granulated(fi, fj, fk, cfg.granulation)
                .total_ext()
                * half;
            dist += half;
        }
    }
    tau
}

/// The finite-disk EARTH-SHADOW sun factor for an elevated sample (WS1
/// march-physics pass): the fraction of the solar disk above the sample's LOCAL
/// GEOMETRIC HORIZON. From radius `r` the horizon dips `acos(R_ground / r)` below
/// the local horizontal, so the disk-centre elevation relative to the horizon is
/// `asin(mu_sun) + dip`; [`atmosphere::solar_disk_visible_fraction`] turns that
/// into the smooth 0..1 circular-segment fraction. This REPLACES the binary
/// `ray_hits_ground` gate on the cloud direct-sun term, which switched the whole
/// sun contribution on/off in a single step as the terminator swept an elevated
/// cloud — the hard lit/unlit line across dusk anvils. Asymptotes match the old
/// gate outside the half-degree penumbral band: 1.0 well above the horizon, 0.0
/// well below (both pinned by tests).
#[inline]
pub fn sun_horizon_disk_fraction(r: f64, mu_sun: f64) -> f64 {
    let ratio = (R_GROUND_M / r.max(R_GROUND_M)).clamp(-1.0, 1.0);
    let dip = ratio.acos();
    let elev = mu_sun.clamp(-1.0, 1.0).asin();
    atmosphere::solar_disk_visible_fraction(elev + dip)
}

/// March the cloud volume along one view ray (design section 4). Front-to-back from
/// the brick-shell entry to the ground/exit, adaptive stepping via the occupancy mip.
/// Returns the in-scattered cloud radiance, the view transmittance, and the cloud's
/// visual-centroid traversal fraction. Twin of the WGSL `march_cloud`.
pub fn march_cloud(scene: &CloudScene, cam: [f64; 3], view: [f64; 3]) -> CloudMarch {
    let vol = scene.vol;
    let Some((t_enter, t_exit)) = ray_shell_segment(cam, view, vol.r_bottom(), vol.r_top()) else {
        return CloudMarch::CLEAR;
    };
    let seg = t_exit - t_enter;
    if seg <= 0.0 {
        return CloudMarch::CLEAR;
    }
    let pitch = vol.voxel_pitch_m();
    let coarse = scene.cfg.coarse_mult * pitch;
    let fine = scene.cfg.fine_mult * pitch;
    let e_sun = SOLAR_IRRADIANCE_RGB;
    let cos_vs = dot3(view, scene.sun_ecef);

    let mut t = t_enter;
    let mut trans = 1.0f64;
    let mut inscatter = [0.0f64; 3];
    let mut sun_inscatter = [0.0f64; 3];
    let mut w_accum = 0.0f64;
    let mut w_weight = 0.0f64;
    let mut steps = 0usize;

    while t < t_exit && steps < scene.cfg.max_steps && trans > scene.cfg.transmittance_floor {
        let p = madd3(cam, view, t);
        let (fi, fj, fk, _r) = ecef_to_brick(p, scene.georef, vol.z_min_m, vol.dz_m);
        let occ = scene.mip.maxext_at(fi, fj, fk);
        // Clamp EVERY step to the shell exit and sample the segment MIDPOINT (WS1
        // march-physics pass, the march_ir pattern): the unclamped final step used
        // to integrate up to half a fine step of extinction PAST the exit (below
        // the ground / outside the brick shell), and the left-endpoint sample
        // biased every in-cloud segment.
        let mut ds = if occ > 0.0 { fine } else { coarse };
        if t + ds > t_exit {
            ds = t_exit - t;
        }
        if ds <= 0.0 {
            break;
        }
        if occ <= 0.0 {
            // Empty block: coarse skip, no sampling.
            t += ds;
            steps += 1;
            continue;
        }
        let pm = madd3(cam, view, t + 0.5 * ds);
        let (mi, mj, mk, rm) = ecef_to_brick(pm, scene.georef, vol.z_min_m, vol.dz_m);
        // The (optionally granulated) view sample — the same eroded field the sun
        // march and the sun-OD map read (MarchConfig::granulation).
        let sample = vol.sample_granulated(mi, mj, mk, scene.cfg.granulation);
        let sigma_t = sample.total_ext();
        if sigma_t <= 0.0 {
            t += ds;
            steps += 1;
            continue;
        }
        // EDGE FEATHER (zoom-out appearance pass): fade the cloud contribution to zero over
        // the outer band of the domain so clouds melt into a zoom-out margin instead of a
        // hard cutoff. `sigma_eff` scales BOTH the in-scatter source and the step opacity
        // consistently, so a faded sample scatters less light AND grows more transparent
        // (the ground shows through). No-op (feather 1.0) when there is no margin, i.e.
        // `edge_feather_cells == 0` -> `sigma_eff == sigma_t` byte-for-byte.
        let feather = edge_feather(mi, mj, vol.nx, vol.ny, scene.cfg.edge_feather_cells);
        let sigma_eff = sigma_t * feather;
        if sigma_eff <= 0.0 {
            t += ds;
            steps += 1;
            continue;
        }

        // Sun source: Wrenninge/Oz multi-scatter octaves (M5) over the SINGLE
        // depth-resolved cloud sun optical depth (marched once, reused by all octaves).
        // octaves=1 == the fix2 single dual-HG scatter `phase(g) * vis(tau_sun)`;
        // octaves=DEFAULT_OCTAVES adds the deep-penetration + back-scatter buildup that
        // makes a thick anvil brilliant. Beer-powder (OFF by default in M5) applies per
        // octave when on.
        let tau_cloud_sun = cloud_sun_optical_depth(scene, pm);
        let sun_src = octave_sun_source(
            cos_vs,
            sample.ext_liquid,
            sample.ext_ice + sample.ext_precip,
            tau_cloud_sun,
            scene.cfg.beer_powder,
            scene.cfg.octaves,
        );

        // Atmospheric sun transmittance to the sample (reddening at low sun) with the
        // FINITE-DISK EARTH-SHADOW FADE (WS1 march-physics pass): the fraction of the
        // solar disk above the sample's local geometric horizon scales the direct sun
        // smoothly through the terminator, replacing the binary ray_hits_ground gate
        // (which drew a hard lit/unlit line across dusk anvils). The transmittance-LUT
        // sample clamps mu to the horizon so the fading disk is attenuated by the
        // (defined) grazing path rather than an undefined below-horizon sample.
        let up = scl3(pm, 1.0 / rm);
        let mu_sun = dot3(up, scene.sun_ecef);
        let disk_sun = sun_horizon_disk_fraction(rm, mu_sun);
        let t_atmo_sun = if disk_sun <= 0.0 {
            [0.0; 3]
        } else {
            let ratio = (R_GROUND_M / rm.max(R_GROUND_M)).min(1.0);
            let mu_horizon = -(1.0 - ratio * ratio).max(0.0).sqrt();
            let tr = atmosphere::sample_transmittance(
                &scene.luts.transmittance,
                rm,
                mu_sun.max(mu_horizon),
            );
            [tr[0] * disk_sun, tr[1] * disk_sun, tr[2] * disk_sun]
        };

        // SH-2 directional sky ambient (M5): the sky irradiance at the voxel's local up
        // in the sun-relative frame (the sky COLOUR, warm at sunset), attenuated from
        // above by e^-tau_up (the brick channel) + a ground bounce from below by
        // e^-tau_down. Replaces M2's scalar white-balanced ambient (design section 6).
        let sun_elev_deg = mu_sun.clamp(-1.0, 1.0).asin().to_degrees();
        let e_sky = scene
            .sky_sh
            .irradiance(sun_elev_deg, up, scene.sun_ecef, up);
        let col_total = vol.sample(mi, mj, 0.0).tau_up; // OD of the whole column at (i,j)
        let tau_down = (col_total - sample.tau_up).max(0.0);
        let amb_factor = ambient_cloud_factor(sample.tau_up, tau_down, scene.cfg.ground_albedo);

        // Use the edge-feathered extinction `sigma_eff` for the step opacity + the
        // in-scatter source (the sun-OD / ambient inputs above use the TRUE field — the
        // feather only fades THIS sample's local scattering). At feather 1.0 (no margin)
        // `sigma_eff == sigma_t`, so this is byte-identical to the pre-feather march.
        let step_t = (-sigma_eff * ds).exp();
        for c in 0..3 {
            let s_sun = e_sun[c] * sigma_eff * sun_src * t_atmo_sun[c];
            let s_amb = sigma_eff * (e_sky[c] / PI) * amb_factor;
            let s = s_sun + s_amb;
            inscatter[c] += trans * (s - s * step_t) / sigma_eff;
            sun_inscatter[c] += trans * (s_sun - s_sun * step_t) / sigma_eff;
        }
        let contribution = trans * (1.0 - step_t);
        let w_frac = ((t + 0.5 * ds) - t_enter) / seg;
        w_accum += contribution * w_frac;
        w_weight += contribution;
        trans *= step_t;
        t += ds;
        steps += 1;
    }

    let mean_w = if w_weight > 0.0 {
        (w_accum / w_weight).clamp(0.0, 1.0)
    } else {
        1.0
    };
    // The absolute distance of the cloud centroid along the ray: t_enter + mean_w*seg
    // (mean_w is the brick-relative fraction, so this reconstructs the weighted mean t
    // exactly). shade_cloud_pixel converts it to the atmosphere-shell fraction.
    let mean_t_m = t_enter + mean_w * seg;
    CloudMarch {
        inscatter,
        sun_inscatter,
        transmittance: trans,
        mean_w,
        mean_t_m,
    }
}

/// The traversal fraction of the ATMOSPHERE shell (entry -> ground / far exit) at an
/// absolute distance `t` (m) along a view ray — the coordinate the aerial-perspective
/// froxel's depth axis is indexed by ([`atmosphere::build_aerial_froxel`]). Returns 1.0
/// (the far endpoint) if the ray misses the shell. This is the correct froxel depth for
/// a cloud sample (M4 review FINDING 4); the previous code passed the BRICK-shell
/// fraction, mapping a ~10 km cloud to ~50 km of airlight.
pub fn atmosphere_shell_fraction(cam: [f64; 3], view: [f64; 3], t: f64) -> f64 {
    match atmosphere::ray_atmosphere_segment(cam, view) {
        Some((t_enter, t_exit)) if t_exit > t_enter => {
            ((t - t_enter) / (t_exit - t_enter)).clamp(0.0, 1.0)
        }
        _ => 1.0,
    }
}

// ── froxel aerial perspective on the cloud ───────────────────────────────────

/// Sample the M2 aerial-perspective froxel for a cloud pixel at scan angle
/// `(scan_x, scan_y)` and traversal fraction `w` (the cloud's visual centroid). The
/// froxel was built over `scan_rect`; its depth axis is the traversal fraction. Nearest
/// sampling in M4 (documented — the froxel is a smooth low-res field). Returns
/// `(camera->cloud inscatter, camera->cloud mean transmittance)`.
pub fn froxel_at_cloud(
    froxel: &AerialFroxel,
    scan_rect: (f64, f64, f64, f64),
    scan_x: f64,
    scan_y: f64,
    w: f64,
) -> ([f64; 3], f64) {
    let (x_min, x_max, y_min, y_max) = scan_rect;
    let dim = froxel.dim;
    if dim == 0 || x_max <= x_min || y_max <= y_min {
        return ([0.0; 3], 1.0);
    }
    let u = ((scan_x - x_min) / (x_max - x_min)).clamp(0.0, 1.0);
    let v = ((scan_y - y_min) / (y_max - y_min)).clamp(0.0, 1.0);
    let x = ((u * dim as f64) as usize).min(dim - 1);
    let y = ((v * dim as f64) as usize).min(dim - 1);
    let z = ((w.clamp(0.0, 1.0) * dim as f64) as usize).min(dim - 1);
    froxel.fetch(x, y, z)
}

// ── the composite (surface + cloud) ──────────────────────────────────────────

/// Ground cloud-shadow factor (design section 6, consumer (a)): the cloud
/// sun-visibility at the ground point the view ray hits. `1.0` when the ray does not
/// reach the ground. M5 uses the PENUMBRAL shadow ([`SunOdMap::penumbral_shadow`]) —
/// a physically soft, distance-widening edge (blur radius = occluder distance x
/// tan 0.533 deg) instead of the fix2 sharp `e^-od`.
pub fn ground_cloud_shadow(scene: &CloudScene, cam: [f64; 3], view: [f64; 3]) -> f64 {
    match ray_sphere(cam, view, scene.vol.r_bottom()) {
        Some((t0, _)) if t0 > 0.0 => {
            let pg = madd3(cam, view, t0);
            scene.sun_od.penumbral_shadow(pg)
        }
        _ => 1.0,
    }
}

/// Composite one pixel: surface (M2, cloud-shadowed) + cloud march + froxel aerial
/// perspective, through the ABI reflectance stretch. Returns display `rgba` in
/// `[0,1]`; alpha `0` only for space (transparent), `1` on earth/limb. Twin of
/// `fs_main` in `clouds.wgsl`.
///
/// The composite (a NAMED approximation):
/// `L = L_toa * T_cloud + T_ac * L_cloud + I_ac * (1 - T_cloud)`
/// where `L_toa` is the M2 surface/limb radiance (which keeps its own camera->ground
/// aerial perspective) shown through the cloud view transmittance `T_cloud`, and
/// `(I_ac, T_ac)` is the froxel camera->cloud aerial perspective at the cloud's visual
/// centroid depth (the ATMOSPHERE-shell fraction, [`atmosphere_shell_fraction`]) applied
/// to the cloud's own radiance `L_cloud`. The front airlight `I_ac` is weighted by
/// `(1 - T_cloud)` rather than added whole: `L_toa` already contains the full
/// camera->ground airlight, so adding `I_ac` outright double-counted the front segment
/// (M4 review FINDING 4). With this weighting the clear case (`T_cloud = 1`) reduces to
/// `L_toa` and the opaque case (`T_cloud -> 0`) keeps the full front airlight in front
/// of the cloud — no double count at either limit.
#[allow(clippy::too_many_arguments)]
pub fn shade_cloud_pixel(
    scene: &CloudScene,
    surf: &FrameContext,
    px: &SurfacePixel,
    froxel: &AerialFroxel,
    scan_rect: (f64, f64, f64, f64),
    scan_x: f64,
    scan_y: f64,
) -> [f32; 4] {
    match composite_cloud_radiance(scene, surf, px, froxel, scan_rect, scan_x, scan_y) {
        None => [0.0, 0.0, 0.0, 0.0], // space
        // One frame exposure gains the whole composited radiance (surface + cloud)
        // uniformly; the per-scene highlight soft-clip keeps bright cloud tops from
        // clamping to a flat white (the top-down/basemap appearance pass). The low-sun
        // illuminant correction sits at the same display seam as the surface/top-down
        // paths (on-earth only; identity outside the 2-30 deg band) so the geo
        // clouds-on product matches them — the lowsun-fix integration hand-off.
        Some(l_final) => radiance_to_rgba_softclip(
            crate::render::apply_low_sun_illuminant(
                l_final,
                px.on_earth,
                px.sun_elev_deg as f64,
                surf.luts,
            ),
            surf.output_transform,
            surf.exposure,
            scene.cfg.cloud_softclip_knee,
        ),
    }
}

/// The composited top-of-atmosphere LINEAR RADIANCE of one cloud pixel (the surface, the
/// cloud, and the froxel front airlight), before any tonemap/exposure. `None` for a space
/// pixel (the surface ray misses the earth/limb). This is the shared numerator of BOTH the
/// RGB product ([`shade_cloud_pixel`] then [`radiance_to_rgba_softclip`]) and the raw-bands
/// product ([`render_cloud_frame_reflectance`] then [`crate::render::reflectance_from_radiance`]),
/// so the two products are the same physics through one composite — the RGB path is
/// byte-identical to before (this is a pure extraction of the former `shade_cloud_pixel`
/// body). See the composite note on [`shade_cloud_pixel`].
#[allow(clippy::too_many_arguments)]
pub fn composite_cloud_radiance(
    scene: &CloudScene,
    surf: &FrameContext,
    px: &SurfacePixel,
    froxel: &AerialFroxel,
    scan_rect: (f64, f64, f64, f64),
    scan_x: f64,
    scan_y: f64,
) -> Option<[f64; 3]> {
    let cam = surf.cam.camera;
    let view = px.view_dir;

    // Space (or limb) with no cloud in the path -> the M2 surface/limb result. The
    // surface radiance carries the per-scene GROUND LIFT (the basemap brightness pass).
    let shadow = ground_cloud_shadow(scene, cam, view);
    let l_toa = surface_toa_radiance(surf, px, shadow, scene.cfg.ground_day_lift)?; // None -> space

    let m = march_cloud(scene, cam, view);
    if m.transmittance >= 1.0 && m.inscatter == [0.0; 3] {
        // No cloud along the ray: the M2 surface, unmodified.
        return Some(l_toa);
    }
    // Froxel depth = the atmosphere-shell traversal fraction of the cloud centroid
    // (NOT the brick-shell fraction the froxel is not indexed by) — FINDING 4.
    let w_froxel = atmosphere_shell_fraction(cam, view, m.mean_t_m);
    let (i_ac, t_ac) = froxel_at_cloud(froxel, scan_rect, scan_x, scan_y, w_froxel);
    let mut l_final = [0.0f64; 3];
    for c in 0..3 {
        l_final[c] =
            l_toa[c] * m.transmittance + t_ac * m.inscatter[c] + i_ac[c] * (1.0 - m.transmittance);
    }
    Some(l_final)
}

// ── GPU volume packing (Texture A) ───────────────────────────────────────────

/// Interleave the brick's four u8 log-quant channels into `Rgba8Unorm` 3-D texture
/// bytes (Texture A): R = ext_liquid, G = ext_ice, B = ext_precip, A = tau_up. The
/// per-channel `LogQuant` scales go to the shader uniforms for in-shader decode. This
/// is the design section-2 / vol3d 3-D upload payload (the codes are already
/// quantized in the brick; no re-quantization). Index `(k*ny + j)*nx + i`.
pub fn pack_texture_a(brick: &VolumeBrick) -> Vec<u8> {
    let n = brick.nx * brick.ny * brick.nz;
    let mut out = Vec::with_capacity(n * 4);
    for (((&l, &ice), &p), &t) in brick
        .ext_liquid
        .iter()
        .zip(&brick.ext_ice)
        .zip(&brick.ext_precip)
        .zip(&brick.tau_up)
    {
        out.extend_from_slice(&[l, ice, p, t]);
    }
    out
}

/// A summary of a frame's cloud coverage, for the env-gated real-fixture assertion
/// (design section 9): the fraction of in-domain pixels with any cloud, and whether
/// every radiance came out finite.
#[derive(Debug, Clone, Copy)]
pub struct CloudFrameStats {
    pub sampled: usize,
    pub cloudy: usize,
    pub all_finite: bool,
    pub max_inscatter: f64,
    /// Peak cloud reflectance factor `rho = pi * L / E_band` over all sampled pixels
    /// and bands — the total (sun + ambient) peak cloud brightness.
    pub max_reflectance: f64,
    /// Peak DIRECT-SUN reflectance factor (from `sun_inscatter` alone) — the acceptance
    /// metric for FINDING 1. Before fix2 this was ~0 (the depth-blind sun-OD map killed
    /// the sun term for thick clouds); a positive value proves the sun single-scatter
    /// term is now alive on sunlit faces. NOTE the absolute value is bounded by the
    /// single-scatter forward-peaked phase in a back-scatter GEO/sun geometry — the
    /// order-0.5-0.9 anvil brightness needs the M5 multiple-scattering octaves.
    pub max_sun_reflectance: f64,
}

impl CloudFrameStats {
    pub fn cloud_fraction(&self) -> f64 {
        if self.sampled == 0 {
            0.0
        } else {
            self.cloudy as f64 / self.sampled as f64
        }
    }
}

/// March the cloud for every in-domain pixel of a scan raster and summarise coverage
/// (used by the env-gated Enderlin fixture test). Steps over the raster by `stride` to
/// keep the CPU cost bounded on big domains. A pixel is "cloudy" when its view
/// transmittance drops below `cloudy_threshold`.
#[allow(clippy::too_many_arguments)]
pub fn cloud_frame_stats(
    scene: &CloudScene,
    cam: &atmosphere::CameraGeometry,
    lat: &[f32],
    lon: &[f32],
    grid_i: &[f32],
    scan: &ScanGrid,
    stride: usize,
    cloudy_threshold: f64,
) -> CloudFrameStats {
    let stride = stride.max(1);
    let mut sampled = 0usize;
    let mut cloudy = 0usize;
    let mut all_finite = true;
    let mut max_inscatter = 0.0f64;
    let mut max_reflectance = 0.0f64;
    let mut max_sun_reflectance = 0.0f64;
    for py in (0..scan.ny).step_by(stride) {
        for px in (0..scan.nx).step_by(stride) {
            let idx = py * scan.nx + px;
            if !grid_i[idx].is_finite() || !lat[idx].is_finite() || !lon[idx].is_finite() {
                continue; // off-earth or outside the WRF domain
            }
            let (sx, sy) = scan.scan_angle(px, py);
            let view = cam.view_dir(sx, sy);
            let m = march_cloud(scene, cam.camera, view);
            sampled += 1;
            for (&ins, &e_band) in m.inscatter.iter().zip(SOLAR_IRRADIANCE_RGB.iter()) {
                if !ins.is_finite() {
                    all_finite = false;
                }
                max_inscatter = max_inscatter.max(ins);
                let rho = PI * ins / e_band;
                if rho.is_finite() {
                    max_reflectance = max_reflectance.max(rho);
                }
            }
            for (&sun_ins, &e_band) in m.sun_inscatter.iter().zip(SOLAR_IRRADIANCE_RGB.iter()) {
                let rho = PI * sun_ins / e_band;
                if rho.is_finite() {
                    max_sun_reflectance = max_sun_reflectance.max(rho);
                }
            }
            if !m.transmittance.is_finite() {
                all_finite = false;
            }
            if m.transmittance < cloudy_threshold {
                cloudy += 1;
            }
        }
    }
    CloudFrameStats {
        sampled,
        cloudy,
        all_finite,
        max_inscatter,
        max_reflectance,
        max_sun_reflectance,
    }
}

/// The scan-angle rectangle of a raster (`x_min, x_max, y_min, y_max`, rad) — the
/// extent the aerial-perspective froxel was built over.
pub fn scan_rect_of(scan: &ScanGrid) -> (f64, f64, f64, f64) {
    let x_max = scan.x_min + scan.nx.saturating_sub(1) as f64 * scan.pitch_x;
    let y_min = scan.y_max - scan.ny.saturating_sub(1) as f64 * scan.pitch_y;
    (scan.x_min, x_max, y_min, scan.y_max)
}

/// Render a full cloud-composited frame to row-major `Rgba8` bytes (row 0 = north,
/// alpha 0 only for space) — the M4 STUDIO render path (design section 4/8). The
/// per-pixel surface state (Blue Marble albedo, terrain normal, LANDMASK water, local
/// sun) is supplied by `assemble` (the studio samples its Blue Marble crop + brick
/// planes + solar), so this stays engine-side and testable; the view direction is
/// derived here from the scan grid so it always matches the camera. Rows are marched
/// in parallel (rayon) on the below-normal worker — the UI never blocks and a newer
/// render supersedes an older one (progressive/cancelable, design section 8; the CPU
/// worker has no per-dispatch TDR budget to respect).
pub fn render_cloud_frame_rgba(
    scene: &CloudScene,
    surf: &FrameContext,
    froxel: &AerialFroxel,
    scan: &ScanGrid,
    assemble: impl Fn(usize, usize) -> SurfacePixel + Sync,
) -> Vec<u8> {
    let (nx, ny) = (scan.nx, scan.ny);
    let scan_rect = scan_rect_of(scan);
    let rows: Vec<Vec<u8>> = (0..ny)
        .into_par_iter()
        .map(|py| {
            let mut row = Vec::with_capacity(nx * 4);
            for px in 0..nx {
                let (sx, sy) = scan.scan_angle(px, py);
                let mut pixel = assemble(px, py);
                pixel.view_dir = surf.cam.view_dir(sx, sy);
                let rgba = shade_cloud_pixel(scene, surf, &pixel, froxel, scan_rect, sx, sy);
                for &v in &rgba {
                    row.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
                }
            }
            row
        })
        .collect();
    rows.into_iter().flatten().collect()
}

/// Render a full cloud-composited frame to row-major RAW REFLECTANCE (`nx*ny*3` f32 in
/// `[0, 1]`, row 0 = north; space pixels are `0`) — the PRE-TONEMAP per-band product the
/// Python binding's `render_visible_bands` returns. Identical assembly to
/// [`render_cloud_frame_rgba`] (same [`composite_cloud_radiance`], same `assemble`, same
/// scan rays), but each pixel's composited radiance is converted to the reflectance factor
/// `pi*L/E_sun` ([`crate::render::reflectance_from_radiance`]) instead of the exposure +
/// ABI-stretch display transform. Rows in parallel (rayon).
pub fn render_cloud_frame_reflectance(
    scene: &CloudScene,
    surf: &FrameContext,
    froxel: &AerialFroxel,
    scan: &ScanGrid,
    assemble: impl Fn(usize, usize) -> SurfacePixel + Sync,
) -> Vec<f32> {
    let (nx, ny) = (scan.nx, scan.ny);
    let scan_rect = scan_rect_of(scan);
    let rows: Vec<Vec<f32>> = (0..ny)
        .into_par_iter()
        .map(|py| {
            let mut row = vec![0.0f32; nx * 3];
            for px in 0..nx {
                let (sx, sy) = scan.scan_angle(px, py);
                let mut pixel = assemble(px, py);
                pixel.view_dir = surf.cam.view_dir(sx, sy);
                if let Some(l) =
                    composite_cloud_radiance(scene, surf, &pixel, froxel, scan_rect, sx, sy)
                {
                    let rho = crate::render::reflectance_from_radiance(l);
                    row[px * 3..px * 3 + 3].copy_from_slice(&rho);
                }
            }
            row
        })
        .collect();
    rows.into_iter().flatten().collect()
}

/// Joint-bilateral upsample of a half-resolution RGB image (`lw*lh*3` f32) to a
/// full-resolution image (`fw*fh*3`) guided by a full-resolution scalar `guide`
/// (`fw*fh`) — the iGPU interactive-preview mechanism (design section 8: "half-res
/// march + bilateral upsample"). Range weights on the guide keep the earth/space and
/// cloud/clear boundaries sharp instead of bleeding across them; a constant guide
/// reduces to bilinear (partition of unity). The M4 studio renders the displayed and
/// stored frame at FULL resolution (stored-frame quality is never reduced, owner
/// decision); this is the tested capability the live-camera preview path uses.
pub fn bilateral_upsample(
    low: &[f32],
    lw: usize,
    lh: usize,
    guide: &[f32],
    fw: usize,
    fh: usize,
    sigma_range: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; fw * fh * 3];
    if lw == 0 || lh == 0 || fw == 0 || fh == 0 {
        return out;
    }
    let sx = lw as f64 / fw as f64;
    let sy = lh as f64 / fh as f64;
    let inv2s2 = if sigma_range > 0.0 {
        1.0 / (2.0 * (sigma_range as f64).powi(2))
    } else {
        0.0
    };
    for y in 0..fh {
        for x in 0..fw {
            let g = guide[y * fw + x] as f64;
            let flx = (x as f64 + 0.5) * sx - 0.5;
            let fly = (y as f64 + 0.5) * sy - 0.5;
            let lx0 = flx.floor() as i64;
            let ly0 = fly.floor() as i64;
            let mut wsum = 0.0f64;
            let mut acc = [0.0f64; 3];
            for dy in 0..2i64 {
                for dx in 0..2i64 {
                    let lx = (lx0 + dx).clamp(0, lw as i64 - 1) as usize;
                    let ly = (ly0 + dy).clamp(0, lh as i64 - 1) as usize;
                    // Guide value at this low-res sample's full-res centre.
                    let gx = (((lx as f64 + 0.5) / sx - 0.5).round() as i64).clamp(0, fw as i64 - 1)
                        as usize;
                    let gy = (((ly as f64 + 0.5) / sy - 0.5).round() as i64).clamp(0, fh as i64 - 1)
                        as usize;
                    let gq = guide[gy * fw + gx] as f64;
                    let wx = (1.0 - (flx - lx as f64).abs()).max(0.0);
                    let wy = (1.0 - (fly - ly as f64).abs()).max(0.0);
                    let range = (-(g - gq).powi(2) * inv2s2).exp();
                    let w = wx * wy * range + 1.0e-6;
                    for c in 0..3 {
                        acc[c] += w * low[(ly * lw + lx) * 3 + c] as f64;
                    }
                    wsum += w;
                }
            }
            for c in 0..3 {
                out[(y * fw + x) * 3 + c] = (acc[c] / wsum) as f32;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atmosphere::{AtmosphereParams, CameraGeometry};
    use crate::frame::{GridGeoref, MapProjection};

    /// A tiny analytic volume: `nx*ny*nz` with a caller-filled extinction field.
    fn build_volume(
        nx: usize,
        ny: usize,
        nz: usize,
        dz: f64,
        horiz: f64,
        fill: impl Fn(usize, usize, usize) -> (f64, f64, f64),
    ) -> DecodedVolume {
        let n = nx * ny * nz;
        let mut ext_liquid = vec![0.0f32; n];
        let mut ext_ice = vec![0.0f32; n];
        let mut ext_precip = vec![0.0f32; n];
        let tau_up = vec![0.0f32; n];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let (l, ic, p) = fill(i, j, k);
                    let c = (k * ny + j) * nx + i;
                    ext_liquid[c] = l as f32;
                    ext_ice[c] = ic as f32;
                    ext_precip[c] = p as f32;
                }
            }
        }
        DecodedVolume {
            nx,
            ny,
            nz,
            z_min_m: 0.0,
            dz_m: dz,
            horiz_pitch_m: horiz,
            ext_liquid,
            ext_ice,
            ext_precip,
            tau_up,
        }
    }

    fn shared_luts() -> &'static (AtmosphereLuts, SkyShTable) {
        static CACHE: std::sync::OnceLock<(AtmosphereLuts, SkyShTable)> =
            std::sync::OnceLock::new();
        CACHE.get_or_init(|| {
            let params = AtmosphereParams::default();
            let luts = AtmosphereLuts::build(&params);
            let sky_sh = SkyShTable::build(&luts, &params, 16);
            (luts, sky_sh)
        })
    }

    fn test_georef(nx: usize, ny: usize, dx: f64) -> GridGeoref {
        // A small Lambert CONUS-ish domain centred at 45N, 100W.
        let proj = MapProjection::lambert(30.0, 60.0, -100.0);
        GridGeoref::new(
            proj,
            (nx - 1) as f64 / 2.0,
            (ny - 1) as f64 / 2.0,
            45.0,
            -100.0,
            dx,
            dx,
        )
    }

    #[test]
    fn out_of_domain_sample_is_clear_no_edge_smear() {
        // A brick that is FULLY cloud in every voxel. Sampling INSIDE the domain returns
        // that cloud; sampling just OUTSIDE any axis returns CLEAR (zero extinction), NOT
        // the clamped edge voxel — so a zoom-out / margin pixel (whose (i, j) falls outside
        // the domain) sees clear sky, never a smear of the domain-edge cloud outward. This
        // is the honesty guarantee for the margin feature: there is no WRF data outside the
        // domain.
        let (nx, ny, nz) = (8usize, 8usize, 8usize);
        let vol = build_volume(nx, ny, nz, 250.0, 3000.0, |_, _, _| (5.0e-2, 0.0, 0.0));
        assert!(
            vol.sample(3.5, 3.5, 3.5).total_ext() > 0.0,
            "interior should be cloudy"
        );
        let eps = 1.0e-3;
        assert_eq!(
            vol.sample(-eps, 3.0, 3.0).total_ext(),
            0.0,
            "west margin not clear"
        );
        assert_eq!(
            vol.sample((nx - 1) as f64 + eps, 3.0, 3.0).total_ext(),
            0.0,
            "east margin not clear"
        );
        assert_eq!(
            vol.sample(3.0, -eps, 3.0).total_ext(),
            0.0,
            "south margin not clear"
        );
        assert_eq!(
            vol.sample(3.0, (ny - 1) as f64 + eps, 3.0).total_ext(),
            0.0,
            "north margin not clear"
        );
        // A comfortably out-of-domain sample (a real margin pixel maps far outside) is clear.
        assert_eq!(
            vol.sample(-3.0, -3.0, 3.0).total_ext(),
            0.0,
            "far margin not clear"
        );
        // The occupancy mip: probes within one block OUTSIDE the boundary read the
        // edge block (the WS1 guard band — conservative step-sizing so a coarse step
        // cannot jump over the entry into edge cloud; the SAMPLER above stays clear,
        // which is what prevents any smear); far outside reads empty (coarse-skip).
        let mip = OccupancyMip::build(&vol, OCCUPANCY_MIP_FACTOR);
        assert!(
            mip.maxext_at(-eps, 3.0, 3.0) > 0.0,
            "the guard band should read the (cloudy) edge block just outside"
        );
        assert_eq!(
            mip.maxext_at(-(OCCUPANCY_MIP_FACTOR as f64) - 1.0, 3.0, 3.0),
            0.0,
            "beyond the guard band the mip reads empty"
        );
    }

    #[test]
    fn dual_hg_phase_integrates_to_one_over_the_sphere() {
        // Numerically integrate p(cos) over the sphere for both class phases.
        let n = 4000;
        let mut liq = 0.0;
        let mut ice = 0.0;
        for i in 0..n {
            let mu = -1.0 + 2.0 * (i as f64 + 0.5) / n as f64;
            let dmu = 2.0 / n as f64;
            liq += phase_liquid(mu) * 2.0 * PI * dmu;
            ice += phase_ice(mu) * 2.0 * PI * dmu;
        }
        assert!((liq - 1.0).abs() < 0.02, "liquid phase integral {liq}");
        assert!((ice - 1.0).abs() < 0.02, "ice phase integral {ice}");
        // Strongly forward-scattering.
        assert!(phase_liquid(1.0) > phase_liquid(-1.0) * 10.0);
    }

    #[test]
    fn beer_powder_never_exceeds_beer_and_darkens_edges() {
        let mut prev_ratio = 0.0;
        for &tau in &[0.0, 0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0] {
            let b = beer(tau);
            let bp = beer_powder(tau);
            assert!(bp <= b + 1e-12, "tau {tau}: powder {bp} > beer {b}");
            assert!(bp >= 0.0);
            // Powder darkens thin cloud far more than thick (edge darkening): the
            // ratio powder/beer rises monotonically from 0 toward 1 with tau.
            if tau > 0.0 {
                let ratio = bp / b;
                assert!(ratio >= prev_ratio - 1e-12, "ratio not monotone at {tau}");
                prev_ratio = ratio;
            }
        }
        // At tau=0 both are 0-ish edge; at large tau powder -> beer.
        assert!((beer_powder(20.0) - beer(20.0)).abs() < 1e-6);
    }

    #[test]
    fn ambient_factor_is_monotone_in_tau_up() {
        let mut prev = f64::INFINITY;
        for &tau_up in &[0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let f = ambient_cloud_factor(tau_up, 0.5, GROUND_ALBEDO);
            assert!(
                f <= prev + 1e-12,
                "not monotone at tau_up {tau_up}: {f} > {prev}"
            );
            assert!(f >= 0.0);
            prev = f;
        }
        // More cloud below (tau_down) also lowers the ground-bounce contribution.
        let a = ambient_cloud_factor(0.5, 0.0, GROUND_ALBEDO);
        let b = ambient_cloud_factor(0.5, 5.0, GROUND_ALBEDO);
        assert!(a > b);
    }

    #[test]
    fn ecef_brick_round_trip_matches_the_projection_ratchet() {
        // Every sampled (i, j, k) -> ECEF -> back to (i, j, k) within the 0.05-cell
        // ratchet (the M4 companion of the M0 projection round trip).
        let (nx, ny, nz) = (60, 45, 40);
        let georef = test_georef(nx, ny, 3000.0);
        let (z_min, dz) = (0.0, 250.0);
        let mut worst = 0.0f64;
        for k in (0..nz).step_by(7) {
            for j in (0..ny).step_by(7) {
                for i in (0..nx).step_by(7) {
                    let p =
                        brick_to_ecef(&georef, i as f64, j as f64, k as f64, z_min, dz).unwrap();
                    let (fi, fj, fk, _) = ecef_to_brick(p, &georef, z_min, dz);
                    worst = worst
                        .max((fi - i as f64).abs())
                        .max((fj - j as f64).abs())
                        .max((fk - k as f64).abs());
                }
            }
        }
        assert!(worst < 0.05, "ecef<->brick round trip worst {worst} cells");
    }

    /// Fine uniform-step line integral of the total extinction along a view ray
    /// through the volume — the closed-form optical depth the march approximates.
    fn reference_optical_depth(
        vol: &DecodedVolume,
        georef: &GridGeoref,
        cam: [f64; 3],
        view: [f64; 3],
    ) -> f64 {
        let Some((t0, t1)) = ray_shell_segment(cam, view, vol.r_bottom(), vol.r_top()) else {
            return 0.0;
        };
        let n = 4000;
        let dt = (t1 - t0) / n as f64;
        let mut od = 0.0;
        for i in 0..n {
            let t = t0 + (i as f64 + 0.5) * dt;
            let p = madd3(cam, view, t);
            let (fi, fj, fk, _) = ecef_to_brick(p, georef, vol.z_min_m, vol.dz_m);
            od += vol.sample(fi, fj, fk).total_ext() * dt;
        }
        od
    }

    #[test]
    fn uniform_slab_transmittance_matches_closed_form_both_directions() {
        // A fully-filled uniform slab of extinction sigma. The adaptive march's view
        // transmittance must match e^{-tau} where tau is the closed-form line integral
        // of the SAME sampled field (the fine reference), for two different rays
        // crossing the slab (both march directions). Comparing to the fine reference
        // isolates the adaptive-stepping error from brick-boundary sampling.
        let (nx, ny, nz) = (16, 16, 24);
        let dz = 250.0;
        let sigma = 4.0e-4; // per class -> total 4e-4 m^-1
        let vol = build_volume(nx, ny, nz, dz, 3000.0, |_, _, _| (sigma, 0.0, 0.0));
        let mip = OccupancyMip::build(&vol, 4);
        let georef = test_georef(nx, ny, 3000.0);
        let (luts, sky_sh) = shared_luts();
        let sun = [0.0, 0.0, -1.0];
        let sun_od = accumulate_sun_od(&vol, &georef, sun, 32);
        let scene = CloudScene {
            vol: &vol,
            mip: &mip,
            sun_od: &sun_od,
            georef: &georef,
            luts,
            sky_sh,
            sun_ecef: sun,
            cfg: MarchConfig::new(StepQuality::Offline, vol.voxel_pitch_m()),
        };
        let cam = CameraGeometry::from_sub_lon(-100.0);
        // Two rays crossing the slab at different slant angles/positions.
        for &(gi, gj) in &[((nx - 1) as f64 / 2.0, (ny - 1) as f64 / 2.0), (3.0, 4.0)] {
            let target = brick_to_ecef(&georef, gi, gj, 0.0, 0.0, dz).unwrap();
            let view = norm3([
                target[0] - cam.camera[0],
                target[1] - cam.camera[1],
                target[2] - cam.camera[2],
            ]);
            let od_ref = reference_optical_depth(&vol, &georef, cam.camera, view);
            let expected = (-od_ref).exp();
            let m = march_cloud(&scene, cam.camera, view);
            // 0.002 (was 0.01): the WS1 final-step clamp + midpoint sampling removed
            // the up-to-half-a-voxel of below-ground extinction the old march
            // integrated past the shell exit.
            assert!(
                (m.transmittance - expected).abs() < 0.002,
                "slab transmittance {} vs closed-form e^-tau {expected} (tau={od_ref})",
                m.transmittance
            );
            // And the closed-form optical depth is genuinely Beer-Lambert: tau/sigma
            // (the covered path length) is within ~voxel-boundary slop of the full
            // shell crossing (the top voxel of the shell is above the last brick level).
            let (t0, t1) =
                ray_shell_segment(cam.camera, view, vol.r_bottom(), vol.r_top()).unwrap();
            let shell_path = t1 - t0;
            let covered = od_ref / sigma;
            assert!(
                covered > 0.8 * shell_path && covered <= 1.02 * shell_path,
                "tau/sigma {covered} not within Beer-Lambert range of shell path {shell_path}"
            );
        }
    }

    #[test]
    fn occupancy_mip_is_conservative_and_dilated() {
        // A box confined to the CENTRE block (voxels 8..16 in each axis -> block
        // (1,1,1) at factor 8) of a 32^3 volume (4 blocks/axis).
        let (nx, ny, nz) = (32, 32, 32);
        let vol = build_volume(nx, ny, nz, 250.0, 3000.0, |i, j, k| {
            let inside = (8..16).contains(&i) && (8..16).contains(&j) && (8..16).contains(&k);
            if inside {
                (1.0e-3, 0.0, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        });
        let mip = OccupancyMip::build(&vol, OCCUPANCY_MIP_FACTOR);
        // (a) Conservative: the mip must mark every non-empty voxel's block occupied.
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    if vol.total_ext_cell(i, j, k) > 0.0 {
                        assert!(
                            mip.maxext_at(i as f64, j as f64, k as f64) > 0.0,
                            "mip skipped a non-empty voxel ({i},{j},{k})"
                        );
                    }
                }
            }
        }
        // (b) Dilation: block (0,0,0) is EMPTY in the raw field but a 26-neighbour of
        // the occupied centre block, so its trilinear skirt cannot be coarse-skipped.
        assert!(
            mip.maxext_at(0.0, 0.0, 0.0) > 0.0,
            "the neighbour block should be dilated-occupied"
        );
        // (c) A block two blocks from the cloud (voxel 24..31 -> block (3,3,3)) is NOT
        // a neighbour of the centre block, so the one-block dilation leaves it empty.
        assert_eq!(
            mip.maxext_at(28.0, 28.0, 28.0),
            0.0,
            "a block two blocks away should remain empty (dilation is one block)"
        );
        // R8 occupancy packing is 255 where occupied, 0 where empty.
        let r8 = mip.to_r8_occupancy();
        assert_eq!(r8.len(), mip.mx * mip.my * mip.mz);
        assert!(r8.contains(&255) && r8.contains(&0));
    }

    #[test]
    fn sun_od_map_casts_a_shadow_column_behind_a_box() {
        // A box cloud; a texel whose sun ray passes through the box has od > 0, a
        // texel to the side has od = 0. And the ground point directly "under" the box
        // toward the sun is shadowed.
        let (nx, ny, nz) = (40, 40, 40);
        let dz = 250.0;
        let vol = build_volume(nx, ny, nz, dz, 3000.0, |i, j, k| {
            let inside = (16..24).contains(&i) && (16..24).contains(&j) && (12..28).contains(&k);
            if inside {
                (2.0e-3, 0.0, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        });
        let georef = test_georef(nx, ny, 3000.0);
        // Sun straight overhead the domain centre (local up at 45N/100W).
        let center = brick_to_ecef(&georef, 20.0, 20.0, 20.0, 0.0, dz).unwrap();
        let sun = norm3(center); // local zenith at the box
        let od = accumulate_sun_od(&vol, &georef, sun, 64);
        // The column through the box centre casts a shadow.
        let box_center = brick_to_ecef(&georef, 20.0, 20.0, 20.0, 0.0, dz).unwrap();
        assert!(
            od.sample(box_center) > 0.0,
            "the box column should have optical depth > 0"
        );
        // A ground point under the box (toward the sun) is shadowed.
        let ground_under = brick_to_ecef(&georef, 20.0, 20.0, 0.0, 0.0, dz).unwrap();
        let shadow = beer(od.sample(ground_under));
        assert!(
            shadow < 0.9,
            "ground under the box should be shadowed: T={shadow}"
        );
        // A far-corner column sees no cloud.
        let corner = brick_to_ecef(&georef, 2.0, 2.0, 20.0, 0.0, dz).unwrap();
        assert!(
            od.sample(corner) < 1e-6,
            "a clear column should have ~0 optical depth, got {}",
            od.sample(corner)
        );
    }

    #[test]
    fn sun_march_lights_cloud_top_brighter_than_base() {
        // A THICK box, sun at the local zenith over it. The depth-resolved secondary
        // sun march must see almost no cloud above a near-TOP sample (sunlit) and the
        // whole column above a near-BASE sample (shadowed) — the FINDING-1 fix that
        // makes thick anvil tops sunlit instead of flat/ambient-only.
        let (nx, ny, nz) = (24, 24, 48);
        let dz = 250.0;
        let vol = build_volume(nx, ny, nz, dz, 3000.0, |i, j, k| {
            let inside = (8..16).contains(&i) && (8..16).contains(&j) && (4..44).contains(&k);
            if inside {
                (4.0e-3, 0.0, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        });
        let georef = test_georef(nx, ny, 3000.0);
        let mip = OccupancyMip::build(&vol, OCCUPANCY_MIP_FACTOR);
        let center = brick_to_ecef(&georef, 12.0, 12.0, 24.0, 0.0, dz).unwrap();
        let sun = norm3(center); // local zenith over the box
        let sun_od = accumulate_sun_od(&vol, &georef, sun, 32);
        let (luts, sky_sh) = shared_luts();
        let scene = scene_ref(&vol, &mip, &sun_od, &georef, luts, sky_sh, sun);
        let top = brick_to_ecef(&georef, 12.0, 12.0, 43.0, 0.0, dz).unwrap();
        let base = brick_to_ecef(&georef, 12.0, 12.0, 5.0, 0.0, dz).unwrap();
        let tau_top = cloud_sun_optical_depth(&scene, top);
        let tau_base = cloud_sun_optical_depth(&scene, base);
        assert!(
            tau_top < 1.0,
            "sunlit cloud top should be near-clear toward the sun: tau_top={tau_top}"
        );
        assert!(
            tau_base > 5.0,
            "cloud base should be heavily shadowed: tau_base={tau_base}"
        );
        assert!(
            tau_base > tau_top * 5.0,
            "base {tau_base} not >> top {tau_top} (depth-blind regression)"
        );
        let vis_top = beer(tau_top);
        let vis_base = beer(tau_base);
        assert!(
            vis_top > 0.5,
            "sunlit top visibility {vis_top} should be high"
        );
        assert!(
            vis_base < 0.05,
            "shadowed base visibility {vis_base} should be near zero"
        );
    }

    #[test]
    fn sun_march_thin_cloud_is_nearly_uniformly_lit() {
        // A 2-voxel-thick cloud: every sample sees at most ~2 voxels of cloud toward
        // the sun, so the top and bottom are lit nearly equally (thin clouds are
        // ~uniform) — the counterpart to the thick self-shadow.
        let (nx, ny, nz) = (24, 24, 24);
        let dz = 250.0;
        let vol = build_volume(nx, ny, nz, dz, 3000.0, |i, j, k| {
            let inside = (8..16).contains(&i) && (8..16).contains(&j) && (11..13).contains(&k);
            if inside {
                (1.5e-3, 0.0, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        });
        let georef = test_georef(nx, ny, 3000.0);
        let mip = OccupancyMip::build(&vol, OCCUPANCY_MIP_FACTOR);
        let center = brick_to_ecef(&georef, 12.0, 12.0, 12.0, 0.0, dz).unwrap();
        let sun = norm3(center);
        let sun_od = accumulate_sun_od(&vol, &georef, sun, 32);
        let (luts, sky_sh) = shared_luts();
        let scene = scene_ref(&vol, &mip, &sun_od, &georef, luts, sky_sh, sun);
        let lower = brick_to_ecef(&georef, 12.0, 12.0, 11.0, 0.0, dz).unwrap();
        let upper = brick_to_ecef(&georef, 12.0, 12.0, 12.0, 0.0, dz).unwrap();
        let vis_lower = beer(cloud_sun_optical_depth(&scene, lower));
        let vis_upper = beer(cloud_sun_optical_depth(&scene, upper));
        assert!(
            vis_upper > 0.5 && vis_lower > 0.3,
            "thin cloud should stay bright: lower {vis_lower} upper {vis_upper}"
        );
        assert!(
            vis_lower / vis_upper > 0.5,
            "thin cloud should be nearly uniformly lit: ratio {}",
            vis_lower / vis_upper
        );
    }

    #[test]
    fn cloud_sun_optical_depth_is_monotone_and_visibility_bounded() {
        // Under a uniform full slab, the sun OD grows as the sample sinks (more cloud
        // above it toward the sun), and the sun visibility stays within [0,1].
        let (nx, ny, nz) = (16, 16, 40);
        let dz = 250.0;
        let vol = build_volume(nx, ny, nz, dz, 3000.0, |_, _, _| (2.0e-3, 0.0, 0.0));
        let georef = test_georef(nx, ny, 3000.0);
        let mip = OccupancyMip::build(&vol, OCCUPANCY_MIP_FACTOR);
        let center = brick_to_ecef(&georef, 8.0, 8.0, 20.0, 0.0, dz).unwrap();
        let sun = norm3(center); // local zenith
        let sun_od = accumulate_sun_od(&vol, &georef, sun, 16);
        let (luts, sky_sh) = shared_luts();
        let scene = scene_ref(&vol, &mip, &sun_od, &georef, luts, sky_sh, sun);
        let mut prev = -1.0f64;
        for &k in &[38.0, 30.0, 20.0, 10.0, 2.0] {
            let p = brick_to_ecef(&georef, 8.0, 8.0, k, 0.0, dz).unwrap();
            let tau = cloud_sun_optical_depth(&scene, p);
            assert!(tau >= 0.0 && tau.is_finite(), "tau {tau} at k={k}");
            let vis = beer(tau);
            assert!((0.0..=1.0).contains(&vis), "vis {vis} out of [0,1]");
            assert!(
                tau >= prev - 1e-9,
                "sun OD should grow as the sample sinks (k={k}): {tau} < {prev}"
            );
            prev = tau;
        }
    }

    #[test]
    fn froxel_depth_maps_to_atmosphere_shell_fraction() {
        // The froxel is indexed by the ATMOSPHERE-shell traversal fraction (entry
        // ~100 km -> ground), NOT the brick-shell fraction. A ~10 km cloud on a
        // near-nadir ray must map to ~0.9 of the way down the atmosphere shell, NOT the
        // ~0.5 the old brick-relative fraction handed the froxel (M4 review FINDING 4).
        let cam = CameraGeometry::from_sub_lon(-100.0);
        let georef = test_georef(40, 40, 3000.0);
        let target = brick_to_ecef(&georef, 20.0, 20.0, 0.0, 0.0, 250.0).unwrap();
        let view = norm3([
            target[0] - cam.camera[0],
            target[1] - cam.camera[1],
            target[2] - cam.camera[2],
        ]);
        let (t_enter, t_exit) =
            crate::atmosphere::ray_atmosphere_segment(cam.camera, view).unwrap();
        // The exact distance along the ray where its altitude is 10 km (the near
        // crossing of the 10 km sphere). The slant cancels in the shell FRACTION, so a
        // 10 km cloud maps to ~(100 - 10)/100 = 0.9 regardless of view obliquity.
        let (t_cloud, _) = ray_sphere(cam.camera, view, R_GROUND_M + 10_000.0).unwrap();
        let w = atmosphere_shell_fraction(cam.camera, view, t_cloud);
        assert!(
            w > 0.75 && w < 0.98,
            "a 10 km cloud should map near the ground end of the atmosphere shell \
             (~0.9), not the brick-relative ~0.5, got {w}"
        );
        // Endpoints map to 0 and 1 exactly.
        assert!(atmosphere_shell_fraction(cam.camera, view, t_enter).abs() < 1e-9);
        assert!((atmosphere_shell_fraction(cam.camera, view, t_exit) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn box_cloud_march_is_a_stable_regression() {
        // A single box cloud on a 32^3 synthetic volume, marched from the
        // geostationary camera through the domain centre. Pin the composite behaviour:
        // the cloud is visible (transmittance < 1, positive finite inscatter, sane
        // centroid) and lit by a mid-sky sun. This is the design section-9 "single box
        // cloud pinned-array regression" (pinned to physical bounds, not raw floats, so
        // it is portable across platforms while still catching a broken march).
        let (nx, ny, nz) = (32, 32, 32);
        let dz = 250.0;
        let vol = build_volume(nx, ny, nz, dz, 3000.0, |i, j, k| {
            let inside = (12..20).contains(&i) && (12..20).contains(&j) && (10..24).contains(&k);
            if inside {
                (5.0e-3, 1.0e-3, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        });
        let mip = OccupancyMip::build(&vol, OCCUPANCY_MIP_FACTOR);
        let georef = test_georef(nx, ny, 3000.0);
        let (luts, sky_sh) = shared_luts();
        // A sun ~40 deg up, to the east of the box.
        let center = brick_to_ecef(&georef, 16.0, 16.0, 17.0, 0.0, dz).unwrap();
        let up = norm3(center);
        let (east, _) = perp_basis(up);
        let e = 40.0f64.to_radians();
        let sun = norm3(add3(scl3(up, e.sin()), scl3(east, e.cos())));
        let sun_od = accumulate_sun_od(&vol, &georef, sun, 64);
        let cfg = MarchConfig::new(StepQuality::Offline, vol.voxel_pitch_m());
        let scene = CloudScene {
            vol: &vol,
            mip: &mip,
            sun_od: &sun_od,
            georef: &georef,
            luts,
            sky_sh,
            sun_ecef: sun,
            cfg,
        };
        let cam = CameraGeometry::from_sub_lon(-100.0);
        let view = norm3([
            center[0] - cam.camera[0],
            center[1] - cam.camera[1],
            center[2] - cam.camera[2],
        ]);
        let m = march_cloud(&scene, cam.camera, view);
        // The box is optically thick along the near-nadir slant: mostly opaque.
        assert!(
            m.transmittance > 0.0 && m.transmittance < 0.6,
            "box transmittance out of expected band: {}",
            m.transmittance
        );
        // Lit cloud: positive, finite inscatter in every band.
        for c in 0..3 {
            assert!(
                m.inscatter[c].is_finite() && m.inscatter[c] > 0.0,
                "band {c} inscatter {} not positive-finite",
                m.inscatter[c]
            );
        }
        // The visual centroid sits inside the cloud slab (not at an edge sentinel).
        assert!(
            (0.05..=0.95).contains(&m.mean_w),
            "cloud centroid {} outside the slab",
            m.mean_w
        );
        // Beer-powder ON must not brighten vs the M5 default (OFF): powder only darkens
        // the sun term (bounded above by Beer per octave). `m` above is the default
        // (powder off); `m_powder` turns it on.
        let cfg_powder = MarchConfig {
            beer_powder: true,
            ..cfg
        };
        let scene_powder = CloudScene {
            cfg: cfg_powder,
            ..scene_ref(&vol, &mip, &sun_od, &georef, luts, sky_sh, sun)
        };
        let m_powder = march_cloud(&scene_powder, cam.camera, view);
        let sum_default: f64 = m.inscatter.iter().sum();
        let sum_powder: f64 = m_powder.inscatter.iter().sum();
        assert!(
            sum_powder <= sum_default + 1e-9,
            "powder should not brighten the default: powder {sum_powder} > default {sum_default}"
        );
    }

    // Small helper so the powder-vs-beer comparison can rebuild the scene struct.
    // The schedule-precision tests pin the sun-march jitter OFF (deterministic
    // sample points); the jitter has its own determinism/neutrality test.
    fn scene_ref<'a>(
        vol: &'a DecodedVolume,
        mip: &'a OccupancyMip,
        sun_od: &'a SunOdMap,
        georef: &'a GridGeoref,
        luts: &'a AtmosphereLuts,
        sky_sh: &'a SkyShTable,
        sun: [f64; 3],
    ) -> CloudScene<'a> {
        CloudScene {
            vol,
            mip,
            sun_od,
            georef,
            luts,
            sky_sh,
            sun_ecef: sun,
            cfg: MarchConfig {
                sun_march_jitter_amp: 0.0,
                ..MarchConfig::new(StepQuality::Offline, vol.voxel_pitch_m())
            },
        }
    }

    #[test]
    fn empty_volume_marches_clear() {
        let (nx, ny, nz) = (16, 16, 16);
        let vol = build_volume(nx, ny, nz, 250.0, 3000.0, |_, _, _| (0.0, 0.0, 0.0));
        let mip = OccupancyMip::build(&vol, 4);
        let georef = test_georef(nx, ny, 3000.0);
        let (luts, sky_sh) = shared_luts();
        let sun = [0.0, 0.0, 1.0];
        let sun_od = accumulate_sun_od(&vol, &georef, sun, 16);
        let scene = scene_ref(&vol, &mip, &sun_od, &georef, luts, sky_sh, sun);
        let cam = CameraGeometry::from_sub_lon(-100.0);
        let center = brick_to_ecef(&georef, 8.0, 8.0, 0.0, 0.0, 250.0).unwrap();
        let view = norm3([
            center[0] - cam.camera[0],
            center[1] - cam.camera[1],
            center[2] - cam.camera[2],
        ]);
        let m = march_cloud(&scene, cam.camera, view);
        assert_eq!(m.transmittance, 1.0);
        assert_eq!(m.inscatter, [0.0; 3]);
    }

    #[test]
    fn render_cloud_frame_produces_valid_rgba() {
        // End-to-end CPU composite: a box cloud over a domain rendered from GOES-East
        // produces a well-formed Rgba8 frame (right byte count, alpha 0-or-255, some
        // on-earth pixels, and at least one visibly-clouded pixel).
        use crate::camera::{
            GeoCamera, MAX_AXIS, SatellitePreset, VISIBLE_PITCH_RAD, build_surface_raster,
        };
        let (nx, ny, nz) = (24, 24, 32);
        let dz = 250.0;
        let vol = build_volume(nx, ny, nz, dz, 3000.0, |i, j, k| {
            let inside = (8..16).contains(&i) && (8..16).contains(&j) && (8..24).contains(&k);
            if inside {
                (5.0e-3, 1.0e-3, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        });
        let mip = OccupancyMip::build(&vol, OCCUPANCY_MIP_FACTOR);
        let georef = test_georef(nx, ny, 3000.0);
        let (luts, sky_sh) = shared_luts();
        let params = AtmosphereParams::default();
        let center = brick_to_ecef(&georef, 12.0, 12.0, 16.0, 0.0, dz).unwrap();
        let sun = norm3(center);
        let sun_od = accumulate_sun_od(&vol, &georef, sun, 48);
        let scene = CloudScene {
            vol: &vol,
            mip: &mip,
            sun_od: &sun_od,
            georef: &georef,
            luts,
            sky_sh,
            sun_ecef: sun,
            cfg: MarchConfig::new(StepQuality::Interactive, vol.voxel_pitch_m()),
        };
        let cam_geo = CameraGeometry::from_sub_lon(-100.0);
        let camera = GeoCamera::new(SatellitePreset::GoesEast);
        let raster =
            build_surface_raster(&camera, &georef, nx, ny, VISIBLE_PITCH_RAD, MAX_AXIS).unwrap();
        let scan_rect = scan_rect_of(&raster.scan);
        let froxel =
            crate::atmosphere::build_aerial_froxel(luts, &params, &cam_geo, sun, scan_rect, 8);
        let surf = FrameContext {
            luts,
            params: &params,
            sky_sh,
            cam: cam_geo,
            sun_ecef: sun,
            output_transform: crate::atmosphere::OutputTransform::AbiReflectance,
            bm_present: false,
            water_scale: 0.55,
            flat_albedo_srgb: 0.5,
            raymarch_steps: 8,
            exposure: 1.0,
        };
        let rnx = raster.nx;
        let lat = raster.lat.clone();
        let assemble = move |px: usize, py: usize| SurfacePixel {
            on_earth: lat[py * rnx + px].is_finite(),
            base_srgb: [0.4, 0.4, 0.4],
            normal_enu: [0.0, 0.0, 1.0],
            sun_enu: [0.0, 0.0, 1.0],
            sun_elev_deg: 60.0,
            is_water: false,
            view_dir: [0.0, 0.0, 1.0],
            ..Default::default()
        };
        let bytes = render_cloud_frame_rgba(&scene, &surf, &froxel, &raster.scan, &assemble);
        assert_eq!(bytes.len(), raster.nx * raster.ny * 4);
        let mut earth = 0;
        for px in bytes.chunks_exact(4) {
            assert!(px[3] == 0 || px[3] == 255, "alpha must be 0 or 255");
            if px[3] == 255 {
                earth += 1;
            }
        }
        assert!(earth > 0, "some pixels should be on earth");

        // The RAW-BANDS (pre-tonemap reflectance) geostationary product over the SAME
        // scene: nx*ny*3 f32, every value finite and in [0, 1], and the lit/clouded scene
        // has a positive reflectance somewhere.
        let refl = render_cloud_frame_reflectance(&scene, &surf, &froxel, &raster.scan, &assemble);
        assert_eq!(refl.len(), raster.nx * raster.ny * 3);
        assert!(
            refl.iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
        );
        assert!(
            refl.iter().cloned().fold(0.0f32, f32::max) > 0.0,
            "the lit/clouded scene should have positive reflectance"
        );
    }

    #[test]
    fn exposure_brightens_the_whole_composited_frame_consistently() {
        // The composite exposure (FrameContext::exposure, applied in radiance_to_rgba)
        // must brighten BOTH clear-surface and clouded pixels together, and never darken
        // any on-earth pixel. Renders the same box-cloud frame at exposure 1.0 and 2.0
        // and asserts: every on-earth pixel is >= as bright, at least one strictly
        // brighter, and clouded pixels brighten too (surface + cloud consistency).
        use crate::camera::{
            GeoCamera, MAX_AXIS, SatellitePreset, VISIBLE_PITCH_RAD, build_surface_raster,
        };
        let (nx, ny, nz) = (24, 24, 32);
        let dz = 250.0;
        let vol = build_volume(nx, ny, nz, dz, 3000.0, |i, j, k| {
            let inside = (8..16).contains(&i) && (8..16).contains(&j) && (8..24).contains(&k);
            if inside {
                (5.0e-3, 1.0e-3, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        });
        let mip = OccupancyMip::build(&vol, OCCUPANCY_MIP_FACTOR);
        let georef = test_georef(nx, ny, 3000.0);
        let (luts, sky_sh) = shared_luts();
        let params = AtmosphereParams::default();
        let center = brick_to_ecef(&georef, 12.0, 12.0, 16.0, 0.0, dz).unwrap();
        let sun = norm3(center);
        let sun_od = accumulate_sun_od(&vol, &georef, sun, 48);
        let scene = CloudScene {
            vol: &vol,
            mip: &mip,
            sun_od: &sun_od,
            georef: &georef,
            luts,
            sky_sh,
            sun_ecef: sun,
            cfg: MarchConfig::new(StepQuality::Interactive, vol.voxel_pitch_m()),
        };
        let cam_geo = CameraGeometry::from_sub_lon(-100.0);
        let camera = GeoCamera::new(SatellitePreset::GoesEast);
        let raster =
            build_surface_raster(&camera, &georef, nx, ny, VISIBLE_PITCH_RAD, MAX_AXIS).unwrap();
        let scan_rect = scan_rect_of(&raster.scan);
        let froxel =
            crate::atmosphere::build_aerial_froxel(luts, &params, &cam_geo, sun, scan_rect, 8);
        let rnx = raster.nx;
        let lat = raster.lat.clone();
        let assemble = move |px: usize, py: usize| SurfacePixel {
            on_earth: lat[py * rnx + px].is_finite(),
            base_srgb: [0.4, 0.4, 0.4],
            normal_enu: [0.0, 0.0, 1.0],
            sun_enu: [0.0, 0.0, 1.0],
            sun_elev_deg: 60.0,
            is_water: false,
            view_dir: [0.0, 0.0, 1.0],
            ..Default::default()
        };
        let render_at = |exposure: f64| {
            let surf = FrameContext {
                luts,
                params: &params,
                sky_sh,
                cam: cam_geo,
                sun_ecef: sun,
                output_transform: crate::atmosphere::OutputTransform::AbiReflectance,
                bm_present: false,
                water_scale: 0.55,
                flat_albedo_srgb: 0.5,
                raymarch_steps: 8,
                exposure,
            };
            render_cloud_frame_rgba(&scene, &surf, &froxel, &raster.scan, &assemble)
        };
        let base = render_at(1.0);
        let bright = render_at(2.0);
        assert_eq!(base.len(), bright.len());
        let mut any_brighter = 0usize;
        let mut cloud_brighter = 0usize;
        for (b0, b1) in base.chunks_exact(4).zip(bright.chunks_exact(4)) {
            if b0[3] == 0 {
                continue; // space
            }
            let s0 = b0[0] as i32 + b0[1] as i32 + b0[2] as i32;
            let s1 = b1[0] as i32 + b1[1] as i32 + b1[2] as i32;
            assert!(
                s1 >= s0,
                "exposure 2.0 darkened an on-earth pixel: {s0} -> {s1}"
            );
            if s1 > s0 {
                any_brighter += 1;
                // A "clouded" pixel is meaningfully bright at exposure 2 (anvil/edge).
                if s1 > 200 {
                    cloud_brighter += 1;
                }
            }
        }
        assert!(
            any_brighter > 0,
            "exposure 2.0 should brighten some on-earth pixels"
        );
        assert!(
            cloud_brighter > 0,
            "exposure should brighten cloud pixels too (surface + cloud consistency)"
        );
    }

    #[test]
    fn bilateral_upsample_partitions_unity_and_preserves_edges() {
        // Constant guide + constant low-res -> constant output (partition of unity).
        let (lw, lh, fw, fh) = (4usize, 4usize, 8usize, 8usize);
        let low = vec![0.5f32; lw * lh * 3];
        let guide_flat = vec![1.0f32; fw * fh];
        let up = bilateral_upsample(&low, lw, lh, &guide_flat, fw, fh, 0.1);
        assert_eq!(up.len(), fw * fh * 3);
        for &v in &up {
            assert!((v - 0.5).abs() < 1e-4, "flat upsample not constant: {v}");
        }
        // A sharp vertical guide edge at x = fw/2: left guide 0, right guide 1. Low-res
        // left half red, right half blue. The upsample must not bleed across the edge.
        let mut low2 = vec![0.0f32; lw * lh * 3];
        for y in 0..lh {
            for x in 0..lw {
                let o = (y * lw + x) * 3;
                if x < lw / 2 {
                    low2[o] = 1.0;
                } else {
                    low2[o + 2] = 1.0;
                }
            }
        }
        let mut guide2 = vec![0.0f32; fw * fh];
        for y in 0..fh {
            for x in 0..fw {
                guide2[y * fw + x] = if x < fw / 2 { 0.0 } else { 1.0 };
            }
        }
        let up2 = bilateral_upsample(&low2, lw, lh, &guide2, fw, fh, 0.2);
        let left = (3 * fw + (fw / 2 - 1)) * 3;
        assert!(
            up2[left] > 0.8 && up2[left + 2] < 0.2,
            "left edge leaked blue: {:?}",
            &up2[left..left + 3]
        );
        let right = (3 * fw + fw / 2) * 3;
        assert!(
            up2[right + 2] > 0.8 && up2[right] < 0.2,
            "right edge leaked red: {:?}",
            &up2[right..right + 3]
        );
    }

    // ── M5: Wrenninge octaves, beer-powder decision, penumbra ──

    #[test]
    fn octave_sun_source_equals_single_scatter_and_converges() {
        // Back-scatter GEO/sun geometry, a thick self-shadowed sample.
        let cos = -0.7;
        let (el, ip) = (3.0e-3, 1.0e-3);
        let tau = 4.0;
        // octaves=1 reproduces the fix2 single dual-HG scatter EXACTLY.
        let single = aggregate_phase(cos, el, ip) * beer(tau);
        let s1 = octave_sun_source(cos, el, ip, tau, false, 1);
        assert!(
            (s1 - single).abs() < 1e-12,
            "octaves=1 must equal single scatter: {s1} vs {single}"
        );
        // Monotone non-decreasing in the octave count, converging to a bounded ceiling
        // (the c<1 geometric weight tail).
        let mut prev = s1;
        for n in 2..=20 {
            let s = octave_sun_source(cos, el, ip, tau, false, n);
            assert!(
                s >= prev - 1e-12,
                "octave sum not monotone at N={n}: {s} < {prev}"
            );
            prev = s;
        }
        // Converges to a bounded ceiling (c<1): the increment from N=30 to N=40 is a
        // tiny fraction of the total (the c=0.85 near-conservative weight converges more
        // slowly than a small c, but still geometrically).
        let s30 = octave_sun_source(cos, el, ip, tau, false, 30);
        let s40 = octave_sun_source(cos, el, ip, tau, false, 40);
        assert!(
            s40 - s30 < 0.02 * s40,
            "octave sum should be near its ceiling by N=30..40: {s30} -> {s40}"
        );
        // The default multi-scatter materially brightens the thick self-shadowed sample.
        let multi = octave_sun_source(cos, el, ip, tau, false, DEFAULT_OCTAVES);
        assert!(
            multi > single * 2.0,
            "octaves should multiply the thick-cloud sun term: {multi} vs single {single}"
        );
    }

    #[test]
    fn beer_powder_default_off_and_only_darkens() {
        // M5 decision: beer-powder OFF by default (octaves now supply the real
        // forward-scatter buildup it used to fake, so powder-on double-darkens).
        let cfg = MarchConfig::new(StepQuality::Offline, 250.0);
        assert!(!cfg.beer_powder, "M5 default: beer-powder must be OFF");
        assert_eq!(cfg.octaves, DEFAULT_OCTAVES, "M5 default: octaves on");
        let cos = -0.6;
        for &tau in &[0.05, 0.5, 3.0, 20.0] {
            let off = octave_sun_source(cos, 2e-3, 1e-3, tau, false, DEFAULT_OCTAVES);
            let on = octave_sun_source(cos, 2e-3, 1e-3, tau, true, DEFAULT_OCTAVES);
            assert!(
                on <= off + 1e-12,
                "powder must not brighten at tau {tau}: on {on} > off {off}"
            );
        }
        // Powder darkens a thin (low-tau) face far more than a thick one — the double-
        // darkening the octaves make unnecessary (on/off ratio smaller at small tau).
        let thin = octave_sun_source(cos, 2e-3, 1e-3, 0.1, true, DEFAULT_OCTAVES)
            / octave_sun_source(cos, 2e-3, 1e-3, 0.1, false, DEFAULT_OCTAVES);
        let thick = octave_sun_source(cos, 2e-3, 1e-3, 5.0, true, DEFAULT_OCTAVES)
            / octave_sun_source(cos, 2e-3, 1e-3, 5.0, false, DEFAULT_OCTAVES);
        assert!(
            thin < thick,
            "powder should darken thin faces more than thick: thin {thin} vs thick {thick}"
        );
    }

    #[test]
    fn multiscatter_octaves_brighten_a_thick_anvil_and_stay_bounded() {
        // A thick synthetic anvil (dense, deep), sun ~50 deg over it, GOES-East view
        // onto the sunlit top. The M5 octaves must lift the peak reflectance far above
        // single scatter (the brilliance payoff), stay energy-plausible (<= 1 at the
        // shipped default), and increase monotonically with the octave count.
        let (nx, ny, nz) = (24, 24, 64);
        let dz = 250.0;
        let vol = build_volume(nx, ny, nz, dz, 3000.0, |i, j, k| {
            let inside = (6..18).contains(&i) && (6..18).contains(&j) && (8..58).contains(&k);
            if inside {
                (6.0e-3, 4.0e-3, 0.0) // total 1e-2 m^-1 over ~12.5 km -> tau ~125
            } else {
                (0.0, 0.0, 0.0)
            }
        });
        let mip = OccupancyMip::build(&vol, OCCUPANCY_MIP_FACTOR);
        let georef = test_georef(nx, ny, 3000.0);
        let (luts, sky_sh) = shared_luts();
        let center = brick_to_ecef(&georef, 12.0, 12.0, 33.0, 0.0, dz).unwrap();
        let up = norm3(center);
        let (east, _) = perp_basis(up);
        let e = 50f64.to_radians();
        let sun = norm3(add3(scl3(up, e.sin()), scl3(east, e.cos())));
        let sun_od = accumulate_sun_od(&vol, &georef, sun, 64);
        let cam = CameraGeometry::from_sub_lon(-100.0);
        let top = brick_to_ecef(&georef, 12.0, 12.0, 56.0, 0.0, dz).unwrap();
        let view = norm3([
            top[0] - cam.camera[0],
            top[1] - cam.camera[1],
            top[2] - cam.camera[2],
        ]);
        let peak_rho = |octaves: usize| -> f64 {
            let cfg = MarchConfig {
                octaves,
                ..MarchConfig::new(StepQuality::Offline, vol.voxel_pitch_m())
            };
            let scene = CloudScene {
                vol: &vol,
                mip: &mip,
                sun_od: &sun_od,
                georef: &georef,
                luts,
                sky_sh,
                sun_ecef: sun,
                cfg,
            };
            let m = march_cloud(&scene, cam.camera, view);
            let mut r = 0.0f64;
            for (ins, e_band) in m.inscatter.iter().zip(SOLAR_IRRADIANCE_RGB.iter()) {
                r = r.max(PI * ins / e_band);
            }
            r
        };
        let single = peak_rho(1);
        let multi = peak_rho(DEFAULT_OCTAVES);
        println!(
            "ANVIL peak reflectance: single(octaves=1)={single:.4} \
             multi(octaves={DEFAULT_OCTAVES})={multi:.4} ratio={:.2}x",
            multi / single.max(1e-9)
        );
        // Monotone increasing in the octave count.
        let mut prev = 0.0;
        for n in 1..=8 {
            let r = peak_rho(n);
            assert!(
                r >= prev - 1e-9,
                "reflectance not monotone in octaves at N={n}: {r} < {prev}"
            );
            prev = r;
        }
        // Energy plausibility at the shipped default (a conservative slab reflects <= 1).
        assert!(
            multi <= 1.0,
            "peak reflectance must stay physical (<= 1) at the default: {multi}"
        );
        // The payoff: octaves multiply the sunlit face, and the anvil reads brilliant
        // (far above the fix2 single-scatter ~0.10-0.16 grey). The printed value is the
        // acceptance evidence; the real Enderlin fixture confirms on WRF data.
        assert!(
            multi > single * 2.0,
            "octaves should multiply the sunlit anvil: {multi} vs single {single}"
        );
        // Brilliance floor: the tuned octaves take this synthetic anvil to ~0.66 (the
        // printed value; 4.3x over single scatter), in the 0.5-0.9 real convective-top
        // band. The floor locks the regression well above the fix2 ~0.10-0.16 grey while
        // leaving headroom for platform float variation.
        assert!(
            multi > 0.45,
            "the multi-scatter sunlit anvil should read brilliant (order 0.5+): {multi}"
        );
    }

    #[test]
    fn penumbra_widens_with_occluder_height() {
        // Two clouds with the same horizontal footprint, one low (near ground) and one
        // high. Sun at the local zenith. The high cloud's occluder distance is larger,
        // so its ground-shadow penumbra (blur radius = occ_dist x tan 0.533 deg) is
        // wider — the EXTRA softening over the sharp e^-od shadow scales with height.
        let (nx, ny, nz) = (32, 32, 56);
        let (dx, dz) = (500.0, 250.0);
        let georef = test_georef(nx, ny, dx);
        let build = |k_lo: usize, k_hi: usize| {
            build_volume(nx, ny, nz, dz, dx, move |i, j, k| {
                let inside =
                    (12..20).contains(&i) && (12..20).contains(&j) && (k_lo..k_hi).contains(&k);
                if inside {
                    (3.0e-3, 0.0, 0.0)
                } else {
                    (0.0, 0.0, 0.0)
                }
            })
        };
        let low = build(2, 6); // ~0.5-1.5 km
        let high = build(44, 48); // ~11-12 km
        let center = brick_to_ecef(&georef, 16.0, 16.0, 28.0, 0.0, dz).unwrap();
        let sun = norm3(center); // local zenith over the box
        let res = 256;
        let od_low = accumulate_sun_od(&low, &georef, sun, res);
        let od_high = accumulate_sun_od(&high, &georef, sun, res);

        // occ_dist scales with cloud height (sampled under the cloud centre).
        let ground_c = brick_to_ecef(&georef, 16.0, 16.0, 0.0, 0.0, dz).unwrap();
        let d_low = od_low.sample_occ_dist(ground_c);
        let d_high = od_high.sample_occ_dist(ground_c);
        assert!(
            d_high > d_low * 3.0,
            "occluder distance should scale with cloud height: high {d_high} vs low {d_low}"
        );

        // Transition width (0.25 -> 0.75) across the shadow edge, for a shadow function.
        let width = |od: &SunOdMap, penumbral: bool| -> f64 {
            let (mut i25, mut i75) = (None, None);
            let mut ii = 16.0;
            while ii <= 26.0 {
                let pg = brick_to_ecef(&georef, ii, 16.0, 0.0, 0.0, dz).unwrap();
                let s = if penumbral {
                    od.penumbral_shadow(pg)
                } else {
                    beer(od.sample(pg))
                };
                if i25.is_none() && s >= 0.25 {
                    i25 = Some(ii);
                }
                if s >= 0.75 {
                    i75 = Some(ii);
                    break;
                }
                ii += 0.02;
            }
            match (i25, i75) {
                (Some(a), Some(b)) => (b - a) * dx,
                _ => 0.0,
            }
        };
        // The EXTRA softening the penumbra adds over the sharp e^-od shadow (isolates the
        // blur from the cloud-edge / map softening common to both clouds).
        let extra_high = width(&od_high, true) - width(&od_high, false);
        let extra_low = width(&od_low, true) - width(&od_low, false);
        assert!(
            extra_high > 0.0,
            "the high cloud should cast a real penumbra (extra softening {extra_high} m)"
        );
        assert!(
            extra_high > extra_low,
            "penumbra widening should scale with occluder height: high +{extra_high} m vs low +{extra_low} m"
        );
    }

    // ── edge feather (zoom-out / margin appearance pass) ──────────────────────

    #[test]
    fn edge_feather_cells_for_margin_is_gated_on_margin() {
        // No margin -> 0 (neutral no-op, byte-identical to the pre-feather march).
        assert_eq!(edge_feather_cells_for_margin(0.0, 200, 300), 0.0);
        assert_eq!(edge_feather_cells_for_margin(-0.1, 200, 300), 0.0);
        // With a margin -> the band is EDGE_FEATHER_BAND_FRAC of the SMALLER axis.
        let b = edge_feather_cells_for_margin(0.3, 200, 300);
        assert!(
            (b - EDGE_FEATHER_BAND_FRAC * 200.0).abs() < 1e-9,
            "band {b}"
        );
        assert!(b > 0.0);
    }

    #[test]
    fn edge_feather_is_a_monotone_edge_ramp_and_no_op_off() {
        let (nx, ny) = (100usize, 100usize);
        let band = 4.0;
        // Off (band 0) -> 1.0 everywhere (neutral no-op), even at the very edge.
        for &(fi, fj) in &[(0.0, 0.0), (50.0, 50.0), (99.0, 50.0)] {
            assert_eq!(
                edge_feather(fi, fj, nx, ny, 0.0),
                1.0,
                "no-op at ({fi},{fj})"
            );
        }
        // At/over the domain edge -> 0 (clouds fully faded into the margin).
        assert_eq!(edge_feather(0.0, 50.0, nx, ny, band), 0.0);
        assert_eq!(edge_feather(99.0, 50.0, nx, ny, band), 0.0);
        assert_eq!(edge_feather(50.0, 0.0, nx, ny, band), 0.0);
        assert_eq!(edge_feather(-3.0, 50.0, nx, ny, band), 0.0, "outside -> 0");
        assert_eq!(
            edge_feather(f64::NAN, 50.0, nx, ny, band),
            0.0,
            "non-finite -> 0"
        );
        // Full interior (deeper than the band from every edge) -> 1.0.
        assert_eq!(edge_feather(50.0, 50.0, nx, ny, band), 1.0);
        assert_eq!(
            edge_feather(band, 50.0, nx, ny, band),
            1.0,
            "at the band depth -> 1"
        );
        // Monotone non-decreasing as we move inward from the west edge along the band.
        let mut prev = -1.0;
        for k in 0..=8 {
            let fi = k as f64 * band / 8.0; // 0 .. band
            let w = edge_feather(fi, 50.0, nx, ny, band);
            assert!(w >= prev - 1e-12, "not monotone at fi={fi}: {w} < {prev}");
            assert!((0.0..=1.0).contains(&w));
            prev = w;
        }
        // Symmetric: the same depth from the EAST edge gives the same weight.
        let d = 1.5;
        let w_w = edge_feather(d, 50.0, nx, ny, band);
        let w_e = edge_feather((nx - 1) as f64 - d, 50.0, nx, ny, band);
        assert!((w_w - w_e).abs() < 1e-12, "edge ramp should be symmetric");
    }

    // ── WS1 march-physics: sun-march reach/schedule/jitter, the finite-disk
    // terminator fade, the final-step clamp, and the sun-OD extent contract ────

    #[test]
    fn sun_march_reaches_a_distant_occluder_through_the_shell() {
        // A dense occluder ~20 km along the sun ray from the sample: the OLD fixed
        // interactive schedule (6 steps, growth 2, base = pitch 250 m) reached only
        // ~15.75 km of slant, so this occluder cast NO shadow at all (tau == 0,
        // measured on the fail-before probe at ec80e88). The WS1 tail extension
        // covers the remaining in-shell slant toward the sun with two stratified
        // samples, so the occluder is sampled (fails before the fix).
        let (nx, ny, nz) = (100, 16, 48);
        let (dx, dz) = (3000.0, 250.0);
        let vol = build_volume(nx, ny, nz, dz, dx, |i, j, k| {
            let inside = (14..19).contains(&i) && (6..11).contains(&j) && (32..41).contains(&k);
            if inside {
                (5.0e-3, 0.0, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        });
        let georef = test_georef(nx, ny, dx);
        let mip = OccupancyMip::build(&vol, OCCUPANCY_MIP_FACTOR);
        let (luts, sky_sh) = shared_luts();
        // Sample near the ground; the sun points at a target INSIDE the occluder
        // (~19.7 km slant away, elevation ~24 deg), so the sun ray crosses it.
        let p = brick_to_ecef(&georef, 10.0, 8.0, 4.0, 0.0, dz).unwrap();
        let q = brick_to_ecef(&georef, 16.0, 8.0, 36.0, 0.0, dz).unwrap();
        let sun = norm3([q[0] - p[0], q[1] - p[1], q[2] - p[2]]);
        let sun_od = accumulate_sun_od(&vol, &georef, sun, 4);
        let cfg = MarchConfig {
            sun_march_jitter_amp: 0.0, // deterministic sample points
            ..MarchConfig::new(StepQuality::Interactive, vol.voxel_pitch_m())
        };
        let scene = CloudScene {
            vol: &vol,
            mip: &mip,
            sun_od: &sun_od,
            georef: &georef,
            luts,
            sky_sh,
            sun_ecef: sun,
            cfg,
        };
        let tau = cloud_sun_optical_depth(&scene, p);
        assert!(
            tau > 1.0,
            "the ~20 km occluder must shadow the sample: tau {tau}"
        );
    }

    #[test]
    fn offline_sun_schedule_converges_better_than_interactive() {
        // A uniform slab, sun at the local zenith: the true sampled-field optical
        // depth from a bottom sample to the field top is analytic (the trilinear
        // field is sigma up to z = (nz-1)*dz, 0 above). The denser offline (10, 1.5)
        // schedule must approximate it better than the interactive (6, 2.0) one.
        let (nx, ny, nz) = (16, 16, 40);
        let dz = 250.0;
        let sigma = 2.0e-4;
        let vol = build_volume(nx, ny, nz, dz, 3000.0, |_, _, _| (sigma, 0.0, 0.0));
        let georef = test_georef(nx, ny, 3000.0);
        let mip = OccupancyMip::build(&vol, OCCUPANCY_MIP_FACTOR);
        let center = brick_to_ecef(&georef, 8.0, 8.0, 20.0, 0.0, dz).unwrap();
        let sun = norm3(center); // local zenith over the sample column
        let sun_od = accumulate_sun_od(&vol, &georef, sun, 4);
        let (luts, sky_sh) = shared_luts();
        let p = brick_to_ecef(&georef, 8.0, 8.0, 2.0, 0.0, dz).unwrap();
        let tau_ref = sigma * ((nz - 1) as f64 - 2.0) * dz;
        let tau_at = |quality: StepQuality| {
            let cfg = MarchConfig {
                sun_march_jitter_amp: 0.0,
                ..MarchConfig::new(quality, vol.voxel_pitch_m())
            };
            let scene = CloudScene {
                vol: &vol,
                mip: &mip,
                sun_od: &sun_od,
                georef: &georef,
                luts,
                sky_sh,
                sun_ecef: sun,
                cfg,
            };
            cloud_sun_optical_depth(&scene, p)
        };
        let err_int = (tau_at(StepQuality::Interactive) - tau_ref).abs();
        let err_off = (tau_at(StepQuality::Offline) - tau_ref).abs();
        assert!(
            err_off < err_int,
            "the offline schedule must converge better: {err_off} !< {err_int} (tau_ref {tau_ref})"
        );
        assert!(
            err_off < 0.35 * tau_ref,
            "offline error should be moderate: {err_off} vs tau {tau_ref}"
        );
    }

    #[test]
    fn sun_march_jitter_is_deterministic_and_amp0_neutral() {
        // The hash is a pure, platform-stable function of the position.
        let a = hash01_position([1.0e6, -2.0e6, 5.5e6]);
        let b = hash01_position([1.0e6, -2.0e6, 5.5e6]);
        assert_eq!(a, b, "hash must be deterministic");
        assert!((0.0..1.0).contains(&a));
        let c = hash01_position([1.0e6 + 300.0, -2.0e6, 5.5e6]);
        assert!((0.0..1.0).contains(&c));
        assert_ne!(a, c, "neighbouring samples should decorrelate");

        // amp 0 reproduces the fixed-midpoint schedule exactly, and the jittered
        // march is itself deterministic (two identical calls agree bit-for-bit).
        let (nx, ny, nz) = (16, 16, 40);
        let dz = 250.0;
        let vol = build_volume(nx, ny, nz, dz, 3000.0, |_, _, _| (1.0e-3, 0.0, 0.0));
        let georef = test_georef(nx, ny, 3000.0);
        let mip = OccupancyMip::build(&vol, OCCUPANCY_MIP_FACTOR);
        let center = brick_to_ecef(&georef, 8.0, 8.0, 20.0, 0.0, dz).unwrap();
        let sun = norm3(center);
        let sun_od = accumulate_sun_od(&vol, &georef, sun, 4);
        let (luts, sky_sh) = shared_luts();
        // Sample high enough that the shell exit is closer than the natural reach,
        // so no schedule extension applies (the reference below assumes base pitch).
        let p = brick_to_ecef(&georef, 8.0, 8.0, 20.0, 0.0, dz).unwrap();
        let tau_amp = |amp: f64| {
            let cfg = MarchConfig {
                sun_march_jitter_amp: amp,
                ..MarchConfig::new(StepQuality::Interactive, vol.voxel_pitch_m())
            };
            let scene = CloudScene {
                vol: &vol,
                mip: &mip,
                sun_od: &sun_od,
                georef: &georef,
                luts,
                sky_sh,
                sun_ecef: sun,
                cfg,
            };
            cloud_sun_optical_depth(&scene, p)
        };
        // Neutrality: amp 0 == an independently-computed fixed-midpoint schedule.
        let mut tau_ref = 0.0f64;
        let (mut dist, mut ds) = (0.0f64, vol.voxel_pitch_m());
        for _ in 0..SUN_MARCH_STEPS {
            let pp = madd3(p, sun, dist + 0.5 * ds);
            let (fi, fj, fk, _) = ecef_to_brick(pp, &georef, vol.z_min_m, vol.dz_m);
            tau_ref += vol.sample(fi, fj, fk).total_ext() * ds;
            dist += ds;
            ds *= SUN_MARCH_GROWTH;
        }
        assert!(
            (tau_amp(0.0) - tau_ref).abs() < 1.0e-12,
            "amp 0 must reproduce the fixed-midpoint march: {} vs {tau_ref}",
            tau_amp(0.0)
        );
        // Determinism of the jittered march.
        assert_eq!(tau_amp(1.0).to_bits(), tau_amp(1.0).to_bits());
    }

    #[test]
    fn cloud_sun_term_survives_a_partial_disk_below_the_horizon() {
        // The WS1 finite-disk earth-shadow fade on the cloud direct sun. The
        // DISCRIMINATING defect of the old binary ray_hits_ground gate (verified on
        // the fail-before probe at ec80e88): with the disk CENTRE below the local
        // horizon but the upper disk still peeking above it, the gate zeroed the
        // sun term EXACTLY — the fade keeps the partial-disk contribution
        // (this assertion fails before the fix, pre-fix value == 0).
        //
        // HONEST FINDING from the probe: in THIS atmosphere (AOD 0.05, 1200 m Mie
        // scale height) the grazing transmittance at cloud-horizon elevations
        // decays to ~1e-5 of its value half a degree higher, so the pre-fix gate's
        // on/off step was already masked by the transmittance's own steepness for
        // elevated clouds — a sweep of the total sun term shows a smooth
        // exponential rise BOTH before and after the fix. The fade is still the
        // correct physics (partial-disk illumination; robust to lower-AOD
        // atmospheres); the "hard dusk line" a viewer may still see is NOT this
        // gate (reported as a cross-workstream finding).
        let (nx, ny, nz) = (16, 16, 40);
        let dz = 250.0;
        // A single-cell cloud so every in-cloud sample shares nearly one horizon.
        let vol = build_volume(nx, ny, nz, dz, 3000.0, |i, j, k| {
            if i == 7 && j == 7 && k == 33 {
                (1.0e-2, 0.0, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        });
        let georef = test_georef(nx, ny, 3000.0);
        let mip = OccupancyMip::build(&vol, OCCUPANCY_MIP_FACTOR);
        let (luts, sky_sh) = shared_luts();
        let center = brick_to_ecef(&georef, 7.0, 7.0, 33.0, 0.0, dz).unwrap();
        let up = norm3(center);
        let (east, _) = perp_basis(up);
        let cam = CameraGeometry::from_sub_lon(-100.0);
        let view = norm3([
            center[0] - cam.camera[0],
            center[1] - cam.camera[1],
            center[2] - cam.camera[2],
        ]);
        // The sun-OD map is not consulted by march_cloud; one dummy map suffices.
        let sun_od = accumulate_sun_od(&vol, &georef, [0.0, 0.0, 1.0], 4);
        let v_at = |e_deg: f64| -> f64 {
            let er = e_deg.to_radians();
            let sun = norm3(add3(scl3(up, er.sin()), scl3(east, er.cos())));
            let cfg = MarchConfig {
                sun_march_jitter_amp: 0.0,
                ..MarchConfig::new(StepQuality::Offline, vol.voxel_pitch_m())
            };
            let scene = CloudScene {
                vol: &vol,
                mip: &mip,
                sun_od: &sun_od,
                georef: &georef,
                luts,
                sky_sh,
                sun_ecef: sun,
                cfg,
            };
            march_cloud(&scene, cam.camera, view).sun_inscatter[0]
        };
        // The deepest horizon any in-cloud sample can have: the trilinear support
        // tops out at z = 8500 m (voxel 34).
        let dip_hi_deg = (R_GROUND_M / (R_GROUND_M + 8500.0)).acos().to_degrees();
        // (a) Whole disk below every sample's horizon: no direct sun at all —
        //     identical to the old gate.
        assert_eq!(
            v_at(-dip_hi_deg - 0.30),
            0.0,
            "fully-set sun must leave no direct term"
        );
        // (b) Disk centre below the deepest horizon, upper disk peeking above: the
        //     sun term must SURVIVE (the old gate zeroed it exactly — fails before).
        let v_peek = v_at(-dip_hi_deg - 0.03);
        assert!(
            v_peek > 0.0,
            "a partial disk above the horizon must light the cloud"
        );
        // (c) Monotone rise across the whole penumbral band into full daylight.
        let mut prev = -1.0f64;
        let mut e_deg = -dip_hi_deg - 0.4;
        while e_deg <= -dip_hi_deg + 0.5 {
            let v = v_at(e_deg);
            assert!(
                v >= prev,
                "the sun term must rise monotonically across the band: {v} < {prev} at {e_deg}"
            );
            prev = v;
            e_deg += 0.05;
        }
        assert!(
            prev > v_peek,
            "the fully-risen sun must exceed the peek value"
        );
    }

    #[test]
    fn sun_horizon_disk_fraction_asymptotes_match_the_binary_gate() {
        // Well outside the half-degree penumbral band the smooth fade equals the
        // old binary ray_hits_ground gate; at the horizon it is exactly half.
        let r = R_GROUND_M + 8000.0;
        let ratio = R_GROUND_M / r;
        let dip = ratio.acos();
        let mu_h = -(1.0 - ratio * ratio).sqrt();
        let above = (-dip + 0.02).sin();
        let below = (-dip - 0.02).sin();
        assert_eq!(sun_horizon_disk_fraction(r, above), 1.0, "full disk above");
        assert_eq!(sun_horizon_disk_fraction(r, below), 0.0, "no disk below");
        assert!(!atmosphere::ray_hits_ground(r, above));
        assert!(atmosphere::ray_hits_ground(r, below));
        assert!(
            (sun_horizon_disk_fraction(r, mu_h) - 0.5).abs() < 1.0e-3,
            "half the disk at the geometric horizon"
        );
        // Monotone across the penumbral band.
        let mut prev = -1.0f64;
        let mut e = -dip - 0.02;
        while e <= -dip + 0.02 {
            let f = sun_horizon_disk_fraction(r, e.sin());
            assert!((0.0..=1.0).contains(&f));
            assert!(f >= prev - 1.0e-12, "disk fraction must be monotone");
            prev = f;
            e += 0.001;
        }
        // At ground level the horizon is the horizontal: elevation 0 = half disk.
        assert!((sun_horizon_disk_fraction(R_GROUND_M, 0.0) - 0.5).abs() < 1.0e-9);
    }

    #[test]
    fn march_final_step_clamps_to_the_shell_exit() {
        // A coarse-voxel (fine step 1000 m) GROUND-TOUCHING layer with rays kept in
        // the domain interior: the only sharp march boundary is the shell exit at
        // the ground, so the residual error isolates the WS1 final-step clamp +
        // midpoint sampling (the layer's top fades over one voxel — a trilinear
        // ramp the midpoint rule integrates almost exactly). Before the fix the
        // unclamped final step integrated up to a full fine step of extinction
        // BELOW the ground (T errors up to 0.06 measured on the fail-before probe
        // at ec80e88) — this test fails before the fix.
        let (nx, ny, nz) = (24, 24, 10);
        let (dx, dz) = (2000.0, 2000.0);
        let sigma = 1.0e-4;
        let vol = build_volume(nx, ny, nz, dz, dx, |_, _, k| {
            if k <= 7 {
                (sigma, 0.0, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        });
        let mip = OccupancyMip::build(&vol, 4);
        let georef = test_georef(nx, ny, dx);
        let (luts, sky_sh) = shared_luts();
        let sun = [0.0, 0.0, 1.0];
        let sun_od = accumulate_sun_od(&vol, &georef, sun, 4);
        let scene = scene_ref(&vol, &mip, &sun_od, &georef, luts, sky_sh, sun);
        let cam = CameraGeometry::from_sub_lon(-100.0);
        // Targets in the NORTH-CENTRE so the slant ray from the (southern) GOES
        // camera descends fully inside the domain (no side-boundary crossings).
        for &(gi, gj) in &[
            (11.5, 17.5),
            (9.2, 18.8),
            (14.9, 16.4),
            (8.3, 19.1),
            (13.7, 17.2),
        ] {
            let target = brick_to_ecef(&georef, gi, gj, 0.0, 0.0, dz).unwrap();
            let view = norm3([
                target[0] - cam.camera[0],
                target[1] - cam.camera[1],
                target[2] - cam.camera[2],
            ]);
            let od_ref = reference_optical_depth(&vol, &georef, cam.camera, view);
            let expected = (-od_ref).exp();
            let m = march_cloud(&scene, cam.camera, view);
            assert!(
                (m.transmittance - expected).abs() < 0.002,
                "ray to ({gi},{gj}): transmittance {} vs reference {expected} (tau {od_ref})",
                m.transmittance
            );
        }
    }

    #[test]
    fn sun_od_out_of_extent_is_clear_not_smeared() {
        // A fully-cloudy volume: every map texel holds column od > 0, including the
        // edge texels. A ground point FAR OUTSIDE the map extent must read od 0
        // (clear; penumbral shadow 1.0), not the clamped edge texel — the old
        // clamp-to-edge read smeared a domain-edge shadow across the whole zoom-out
        // margin strip (fails before the fix).
        let (nx, ny, nz) = (24, 24, 24);
        let dz = 250.0;
        let vol = build_volume(nx, ny, nz, dz, 3000.0, |_, _, _| (2.0e-3, 0.0, 0.0));
        let georef = test_georef(nx, ny, 3000.0);
        let center = brick_to_ecef(&georef, 11.5, 11.5, 12.0, 0.0, dz).unwrap();
        let sun = norm3(center);
        let od = accumulate_sun_od(&vol, &georef, sun, 32);
        // Interior ground point: a real shadow.
        let inside = brick_to_ecef(&georef, 11.5, 11.5, 0.0, 0.0, dz).unwrap();
        assert!(
            od.sample(inside) > 0.5,
            "interior column should carry od: {}",
            od.sample(inside)
        );
        assert!(od.penumbral_shadow(inside) < 0.9);
        // A margin ground point far outside the domain (and the map extent).
        let outside = brick_to_ecef(&georef, -100.0, -100.0, 0.0, 0.0, dz).unwrap();
        assert_eq!(od.sample(outside), 0.0, "out-of-extent od must be clear");
        assert_eq!(
            od.sample_occ_dist(outside),
            0.0,
            "out-of-extent occ_dist is 0"
        );
        assert_eq!(
            od.penumbral_shadow(outside),
            1.0,
            "no shadow outside the map extent"
        );
    }

    #[test]
    fn sun_od_edge_feather_fades_the_outer_band_only() {
        // Default map vs feather 0: interior texels are byte-identical (the band-0
        // anchor — feather 0 IS the raw pre-WS1 accumulation), the outermost ring
        // fades fully, the in-between band never exceeds raw. occ_dist untouched.
        let (nx, ny, nz) = (24, 24, 24);
        let dz = 250.0;
        let vol = build_volume(nx, ny, nz, dz, 3000.0, |_, _, _| (2.0e-3, 0.0, 0.0));
        let georef = test_georef(nx, ny, 3000.0);
        let center = brick_to_ecef(&georef, 11.5, 11.5, 12.0, 0.0, dz).unwrap();
        let sun = norm3(center);
        let res = 32usize;
        let raw = accumulate_sun_od_feathered(&vol, &georef, sun, res, 0.0);
        let feathered = accumulate_sun_od(&vol, &georef, sun, res);
        let band = SUN_OD_EDGE_FEATHER_TEXELS.ceil() as usize;
        let mut edge_reduced = 0usize;
        for ty in 0..res {
            for tx in 0..res {
                let d = tx.min(res - 1 - tx).min(ty.min(res - 1 - ty));
                let (r, f) = (raw.od[ty * res + tx], feathered.od[ty * res + tx]);
                if d >= band {
                    assert_eq!(r, f, "interior texel ({tx},{ty}) must be untouched");
                } else if d == 0 {
                    assert_eq!(f, 0.0, "the outermost ring must fade fully");
                    if r > 0.0 {
                        edge_reduced += 1;
                    }
                } else {
                    assert!(f <= r, "feathered texel ({tx},{ty}) must not exceed raw");
                }
            }
        }
        assert!(
            edge_reduced > 0,
            "the fully-cloudy map should carry od on the edge ring"
        );
        assert_eq!(
            raw.occ_dist, feathered.occ_dist,
            "occ_dist is not feathered"
        );
    }

    // ── sub-grid cloud GRANULATION (edge-erosion detail noise) ─────────────────

    #[test]
    fn granulation_amplitude_follows_the_unresolved_spectrum_and_caps() {
        // The dx-derived amplitude: near-zero on a 250 m run (the model already
        // resolves the granulation-scale texture), strong on a 2-3 km run, monotone
        // in dx, capped at the Cahalan-bound amplitude for very coarse grids, and 0
        // for degenerate dx.
        let a250 = granulation_amplitude(250.0);
        let a1000 = granulation_amplitude(1000.0);
        let a3000 = granulation_amplitude(3000.0);
        println!("GRAN amplitude: dx250={a250:.4} dx1000={a1000:.4} dx3000={a3000:.4}");
        assert!(
            a250 > 0.0 && a250 < 0.2,
            "250 m amplitude should be near-zero: {a250}"
        );
        assert!(a3000 > 0.4, "3 km amplitude should be strong: {a3000}");
        assert!(
            a3000 > 2.5 * a250,
            "coarse grid must erode far more than fine: {a3000} vs {a250}"
        );
        let mut prev = 0.0f64;
        for dx in [30.0, 100.0, 250.0, 500.0, 1000.0, 3000.0, 8000.0, 30000.0] {
            let a = granulation_amplitude(dx);
            assert!(
                (0.0..=GRAN_AMP_CAP).contains(&a),
                "amplitude {a} out of [0, cap] at dx {dx}"
            );
            assert!(a >= prev, "amplitude not monotone at dx {dx}: {a} < {prev}");
            prev = a;
        }
        // The Cahalan-derived cap binds for very coarse grids.
        assert_eq!(granulation_amplitude(50_000.0), GRAN_AMP_CAP);
        // At/below the render-scale floor there is nothing unresolved to add.
        assert_eq!(granulation_amplitude(30.0), 0.0);
        assert_eq!(granulation_amplitude(0.0), 0.0);
        assert_eq!(granulation_amplitude(-5.0), 0.0);
        assert_eq!(granulation_amplitude(f64::NAN), 0.0);
    }

    #[test]
    fn granulation_noise_is_deterministic_shaped_and_tail_distributed() {
        // Determinism (the same brick-plane position always hashes to the same
        // erosion — no shimmer between frames, and geo == top-down by construction
        // since both sample by brick coordinates).
        let a = granulation_erosion_noise(12_345.0, -6_789.0);
        let b = granulation_erosion_noise(12_345.0, -6_789.0);
        assert_eq!(a.to_bits(), b.to_bits(), "noise must be deterministic");
        // Tail-shaped distribution over a 10 km field sampled at 62.5 m: values in
        // [0, 1]; MOST of the field is exactly-zero grain interior (untouched cloud);
        // a real minority is the fully-carved gap network; the mean is small (the
        // Cahalan premise: the eroded area fraction is the W-tail).
        let n = 160usize;
        let (mut sum, mut carved, mut zero) = (0.0f64, 0usize, 0usize);
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        for y in 0..n {
            for x in 0..n {
                let e = granulation_erosion_noise(x as f64 * 62.5, y as f64 * 62.5);
                assert!((0.0..=1.0).contains(&e), "noise {e} out of [0,1]");
                sum += e;
                if e >= 0.9 {
                    carved += 1;
                }
                if e <= 0.0 {
                    zero += 1;
                }
                min_v = min_v.min(e);
                max_v = max_v.max(e);
            }
        }
        let total = (n * n) as f64;
        let mean = sum / total;
        let carved_frac = carved as f64 / total;
        let zero_frac = zero as f64 / total;
        println!(
            "GRAN noise: mean={mean:.4} carved_frac={carved_frac:.4} zero_frac={zero_frac:.4}"
        );
        assert!(max_v > 0.9 && min_v <= 0.0, "field should span grain..gap");
        assert!(
            mean > 0.03 && mean < 0.40,
            "erosion-field mean {mean} outside the tail-shaped band"
        );
        assert!(
            (0.02..0.45).contains(&carved_frac),
            "carved-gap fraction {carved_frac} implausible"
        );
        assert!(
            zero_frac > 0.30,
            "most of the field should be untouched grain interior: {zero_frac}"
        );
    }

    #[test]
    fn granulation_multiplier_is_remap_subtract_only() {
        // e = 0 is the neutral no-op; d = 1 (a sample AT its neighbourhood max — a
        // thick-core / uniform-deck interior) is untouched for ANY erosion; d <= e is
        // fully carved; monotone in both arguments; never above 1 (subtract-only).
        assert_eq!(granulation_multiplier(0.7, 0.0), 1.0);
        for &e in &[0.0, 0.1, 0.3, 0.6, 0.98, 1.5] {
            assert_eq!(
                granulation_multiplier(1.0, e),
                1.0,
                "interior (d=1) must be untouched at e={e}"
            );
        }
        assert_eq!(granulation_multiplier(0.3, 0.3), 0.0);
        assert_eq!(granulation_multiplier(0.1, 0.5), 0.0);
        let mut prev = 1.0f64;
        for &e in &[0.0, 0.1, 0.2, 0.4, 0.6, 0.8] {
            let m = granulation_multiplier(0.7, e);
            assert!((0.0..=1.0).contains(&m), "m {m} out of range at e={e}");
            assert!(m <= prev + 1e-12, "m must fall as erosion rises");
            prev = m;
        }
        let mut prev = 0.0f64;
        for &d in &[0.05, 0.2, 0.4, 0.6, 0.8, 1.0] {
            let m = granulation_multiplier(d, 0.35);
            assert!(m >= prev - 1e-12, "m must rise with relative density");
            prev = m;
        }
    }

    #[test]
    fn granulation_off_and_zero_amplitude_are_byte_identical() {
        // (a) The restructured sampler with granulation None reproduces the original
        // trilinear formula BIT-FOR-BIT (an independent inline reference); (b) a zero
        // amplitude is byte-identical to None (the off-flag anchor).
        let (nx, ny, nz) = (12usize, 10usize, 8usize);
        let vol = build_volume(nx, ny, nz, 250.0, 3000.0, |i, j, k| {
            let v = ((i * 7 + j * 13 + k * 3) % 5) as f64 * 1.0e-3;
            let w = ((i + 2 * j) % 3) as f64 * 5.0e-4;
            (v, w, if k % 2 == 0 { 2.0e-4 } else { 0.0 })
        });
        let cell = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;
        let reference = |ch: &[f32], fi: f64, fj: f64, fk: f64| -> f64 {
            let i0 = fi.floor() as usize;
            let j0 = fj.floor() as usize;
            let k0 = fk.floor() as usize;
            let i1 = (i0 + 1).min(nx - 1);
            let j1 = (j0 + 1).min(ny - 1);
            let k1 = (k0 + 1).min(nz - 1);
            let (ti, tj, tk) = (fi - i0 as f64, fj - j0 as f64, fk - k0 as f64);
            let g = |i: usize, j: usize, k: usize| ch[cell(i, j, k)] as f64;
            let c00 = g(i0, j0, k0) * (1.0 - ti) + g(i1, j0, k0) * ti;
            let c10 = g(i0, j1, k0) * (1.0 - ti) + g(i1, j1, k0) * ti;
            let c01 = g(i0, j0, k1) * (1.0 - ti) + g(i1, j0, k1) * ti;
            let c11 = g(i0, j1, k1) * (1.0 - ti) + g(i1, j1, k1) * ti;
            let c0 = c00 * (1.0 - tj) + c10 * tj;
            let c1 = c01 * (1.0 - tj) + c11 * tj;
            c0 * (1.0 - tk) + c1 * tk
        };
        let amp0 = Some(Granulation { amplitude: 0.0 });
        let mut probes = 0usize;
        for pi in 0..23 {
            for pj in 0..19 {
                let fi = pi as f64 * (nx - 1) as f64 / 22.0;
                let fj = pj as f64 * (ny - 1) as f64 / 18.0;
                let fk = ((pi * 19 + pj) % 29) as f64 * (nz - 1) as f64 / 28.0;
                let s = vol.sample(fi, fj, fk);
                assert_eq!(
                    s.ext_liquid.to_bits(),
                    reference(&vol.ext_liquid, fi, fj, fk).to_bits(),
                    "sampler drifted from the reference trilerp at ({fi},{fj},{fk})"
                );
                assert_eq!(
                    s.ext_ice.to_bits(),
                    reference(&vol.ext_ice, fi, fj, fk).to_bits()
                );
                assert_eq!(
                    s.tau_up.to_bits(),
                    reference(&vol.tau_up, fi, fj, fk).to_bits()
                );
                let z = vol.sample_granulated(fi, fj, fk, amp0);
                assert_eq!(z.ext_liquid.to_bits(), s.ext_liquid.to_bits());
                assert_eq!(z.ext_ice.to_bits(), s.ext_ice.to_bits());
                assert_eq!(z.ext_precip.to_bits(), s.ext_precip.to_bits());
                assert_eq!(z.tau_up.to_bits(), s.tau_up.to_bits());
                probes += 1;
            }
        }
        assert!(probes > 400);
    }

    #[test]
    fn granulation_is_subtract_only_and_zero_stays_zero() {
        // Over a scattered low-liquid popcorn field at the STRONGEST amplitude, the
        // eroded sample never exceeds the raw one in ANY channel, tau_up is never
        // touched, clear air stays clear, and the erosion is live somewhere.
        let (nx, ny, nz) = (20usize, 20usize, 12usize);
        let vol = build_volume(nx, ny, nz, 250.0, 3000.0, |i, j, k| {
            let blob = (i % 5 == 2) && (j % 4 == 1) && (2..7).contains(&k);
            if blob {
                (2.0e-2, 0.0, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        });
        let gran = Some(Granulation {
            amplitude: GRAN_AMP_CAP,
        });
        let mut strictly_less = 0usize;
        let (mut p, mut q, mut r) = (0.13f64, 0.37f64, 0.71f64);
        for _ in 0..12_000 {
            // A deterministic low-discrepancy walk over the volume.
            p = (p + 0.618_033_988_749_895).fract();
            q = (q + 0.414_213_562_373_095).fract();
            r = (r + 0.324_717_957_244_746).fract();
            let fi = p * (nx - 1) as f64;
            let fj = q * (ny - 1) as f64;
            let fk = r * (nz - 1) as f64;
            let raw = vol.sample(fi, fj, fk);
            let ero = vol.sample_granulated(fi, fj, fk, gran);
            assert!(
                ero.ext_liquid <= raw.ext_liquid,
                "liquid grew at ({fi},{fj},{fk})"
            );
            assert!(ero.ext_ice <= raw.ext_ice);
            assert!(ero.ext_precip <= raw.ext_precip);
            assert_eq!(ero.tau_up.to_bits(), raw.tau_up.to_bits(), "tau_up eroded");
            if raw.total_ext() <= 0.0 {
                assert_eq!(ero.total_ext(), 0.0, "erosion added cloud to clear air");
            } else if ero.total_ext() < raw.total_ext() {
                strictly_less += 1;
            }
        }
        assert!(
            strictly_less > 50,
            "the erosion should be live on a coarse-grid liquid field: {strictly_less}"
        );
    }

    #[test]
    fn granulation_gates_ice_and_high_liquid() {
        // The species/height gate: ice-only cloud (anvils/cirrus) and liquid above
        // the boundary-layer band are byte-untouched at ANY amplitude; the same blob
        // as low liquid granulates. Plus the gate function's own anchors.
        assert_eq!(granulation_gate(1.0e-2, 0.0, 0.0, 2000.0), 1.0);
        assert_eq!(granulation_gate(0.0, 1.0e-2, 0.0, 2000.0), 0.0);
        assert_eq!(granulation_gate(0.0, 0.0, 1.0e-2, 2000.0), 0.0);
        assert_eq!(granulation_gate(1.0e-2, 0.0, 0.0, 8000.0), 0.0);
        assert_eq!(granulation_gate(0.0, 0.0, 0.0, 1000.0), 0.0);
        let mixed = granulation_gate(5.0e-3, 5.0e-3, 0.0, 2000.0);
        assert!((mixed - 0.5).abs() < 1e-12, "mixed-phase gate {mixed}");
        let mut prev = 1.0f64;
        for &z in &[3000.0, 4500.0, 5500.0, 6500.0, 7500.0] {
            let g = granulation_gate(1.0e-2, 0.0, 0.0, z);
            assert!(g <= prev + 1e-12, "height gate must fall with z");
            prev = g;
        }

        let gran = Some(Granulation {
            amplitude: GRAN_AMP_CAP,
        });
        let blob = |lo: usize, hi: usize| {
            move |i: usize, j: usize, k: usize| {
                if (6..14).contains(&i) && (6..14).contains(&j) && (lo..hi).contains(&k) {
                    (0.0, 1.5e-2, 0.0)
                } else {
                    (0.0, 0.0, 0.0)
                }
            }
        };
        let (nx, ny, nz) = (20usize, 20usize, 40usize);
        // Ice-only, low: byte-identical.
        let ice = build_volume(nx, ny, nz, 250.0, 3000.0, blob(2, 8));
        // High liquid (k 30..36 -> 7.5-9 km MSL, above the zero height): byte-identical.
        let high = build_volume(nx, ny, nz, 250.0, 3000.0, |i, j, k| {
            if (6..14).contains(&i) && (6..14).contains(&j) && (30..36).contains(&k) {
                (1.5e-2, 0.0, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        });
        // Low liquid: granulated.
        let low = build_volume(nx, ny, nz, 250.0, 3000.0, |i, j, k| {
            if (6..14).contains(&i) && (6..14).contains(&j) && (2..8).contains(&k) {
                (1.5e-2, 0.0, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        });
        let mut low_changed = 0usize;
        for pi in 0..40 {
            for pj in 0..40 {
                let fi = 5.0 + pi as f64 * 0.25;
                let fj = 5.0 + pj as f64 * 0.25;
                for &fk in &[3.5f64, 5.1, 31.5, 33.2] {
                    for (vol, changed) in [(&ice, false), (&high, false), (&low, true)] {
                        let raw = vol.sample(fi, fj, fk);
                        let ero = vol.sample_granulated(fi, fj, fk, gran);
                        if !changed {
                            assert_eq!(
                                ero.total_ext().to_bits(),
                                raw.total_ext().to_bits(),
                                "gated volume must be byte-untouched at ({fi},{fj},{fk})"
                            );
                        } else if ero.total_ext() < raw.total_ext() {
                            low_changed += 1;
                        }
                    }
                }
            }
        }
        assert!(
            low_changed > 20,
            "low liquid should granulate: {low_changed} changed samples"
        );
    }

    #[test]
    fn granulation_carves_gaps_within_the_cahalan_tau_bound() {
        // ONE blocky cell of boundary-layer liquid on a 3 km grid — the popcorn-cu
        // defect case. The erosion must GRANULATE its trilinear tent (carve real gaps
        // AND keep real grains at the SAME relative density — not a uniform ring
        // feather), while the sigma-weighted field mean stays within the Cahalan
        // plane-parallel bound (never over-thinned).
        let (nx, ny, nz) = (16usize, 16usize, 8usize);
        let vol = build_volume(nx, ny, nz, 250.0, 3000.0, |i, j, k| {
            if i == 8 && j == 8 && k == 3 {
                (2.0e-2, 0.0, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        });
        let run = |amplitude: f64| -> (f64, f64, f64, f64) {
            let gran = Some(Granulation { amplitude });
            let peak = vol.sample(8.0, 8.0, 3.0).total_ext();
            let (mut sum_raw, mut sum_ero) = (0.0f64, 0.0f64);
            let mut m_min = f64::INFINITY;
            let mut grains = 0usize;
            let mut n_cloudy = 0usize;
            let (mut band_min, mut band_max) = (f64::INFINITY, f64::NEG_INFINITY);
            for pi in 0..99 {
                for pj in 0..99 {
                    let fi = 7.02 + pi as f64 * 0.02;
                    let fj = 7.02 + pj as f64 * 0.02;
                    let raw = vol.sample(fi, fj, 3.0).total_ext();
                    let ero = vol.sample_granulated(fi, fj, 3.0, gran).total_ext();
                    sum_raw += raw;
                    sum_ero += ero;
                    if raw > 0.0 {
                        n_cloudy += 1;
                        let m = ero / raw;
                        m_min = m_min.min(m);
                        if m > 0.95 {
                            grains += 1;
                        }
                        // The fixed relative-density band d in [0.4, 0.5]: a pure edge
                        // feather would give one multiplier here; granulation varies it.
                        let d = raw / peak;
                        if (0.4..0.5).contains(&d) {
                            band_min = band_min.min(m);
                            band_max = band_max.max(m);
                        }
                    }
                }
            }
            let ratio = sum_ero / sum_raw;
            let grain_frac = grains as f64 / n_cloudy as f64;
            println!(
                "GRAN tent amp={amplitude:.3}: mean-tau ratio={ratio:.4} m_min={m_min:.4} \
                 grain_frac={grain_frac:.4} band m range=[{band_min:.3}, {band_max:.3}]"
            );
            (ratio, m_min, grain_frac, band_max - band_min)
        };
        // The 3 km amplitude (the spec's named strong case).
        let (ratio, m_min, grain_frac, band_range) = run(granulation_amplitude(3000.0));
        assert!(
            ratio >= CAHALAN_TAU_FACTOR,
            "field-mean tau reduction over-thins the Cahalan bound: {ratio}"
        );
        assert!(ratio < 0.995, "the erosion should actually erode: {ratio}");
        assert!(m_min < 0.05, "gaps must carve to ~clear: m_min {m_min}");
        assert!(
            grain_frac > 0.3,
            "grains must survive untouched: {grain_frac}"
        );
        assert!(
            band_range > 0.5,
            "at fixed relative density the multiplier must vary grain-to-gap \
             (granulation, not an outline feather): range {band_range}"
        );
        // The extreme (cap) amplitude stays near the bound too (documented margin).
        let (ratio_cap, ..) = run(GRAN_AMP_CAP);
        assert!(
            ratio_cap >= CAHALAN_TAU_FACTOR - 0.08,
            "cap-amplitude mean-tau ratio {ratio_cap} far below the Cahalan bound"
        );
    }

    #[test]
    fn granulation_interior_protection_shields_solid_deck_variability() {
        // The round-1 QA pepper defect: ordinary cell-to-cell LWC variability inside
        // a WIDE solid liquid deck read as "edge" under the pure remap and peppered
        // the deck with pinholes. The interior-protection window must leave such a
        // deck byte-UNTOUCHED (its interior relative density stays >= GRAN_INTERIOR_HI)
        // while a single-cell popcorn tent still granulates.
        assert_eq!(granulation_interior_protection(1.0), 0.0);
        assert_eq!(granulation_interior_protection(GRAN_INTERIOR_HI), 0.0);
        assert_eq!(granulation_interior_protection(GRAN_INTERIOR_LO), 1.0);
        assert_eq!(granulation_interior_protection(0.2), 1.0);
        let mut prev = 1.0f64;
        for &d in &[0.3, 0.45, 0.55, 0.65, 0.75, 0.9, 1.0] {
            let p = granulation_interior_protection(d);
            assert!((0.0..=1.0).contains(&p));
            assert!(p <= prev + 1e-12, "protection must fall with density");
            prev = p;
        }

        let (nx, ny, nz) = (24usize, 24usize, 10usize);
        // A wide deck with +/-10% deterministic cell-to-cell variability, k 2..6
        // (boundary-layer liquid — the granulation target regime by gate).
        let deck = build_volume(nx, ny, nz, 250.0, 3000.0, |i, j, k| {
            if (2..6).contains(&k) {
                let v = 1.0 + 0.1 * (((i * 5 + j * 11 + k) % 7) as f64 / 3.0 - 1.0);
                (2.0e-2 * v, 0.0, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        });
        let gran = Some(Granulation {
            amplitude: GRAN_AMP_CAP,
        });
        // Sample the deck INTERIOR (well inside every horizontal edge): byte-untouched.
        for pi in 0..30 {
            for pj in 0..30 {
                let fi = 4.0 + pi as f64 * 0.5;
                let fj = 4.0 + pj as f64 * 0.5;
                for &fk in &[3.0f64, 3.5, 4.2] {
                    let raw = deck.sample(fi, fj, fk);
                    let ero = deck.sample_granulated(fi, fj, fk, gran);
                    assert_eq!(
                        ero.total_ext().to_bits(),
                        raw.total_ext().to_bits(),
                        "deck interior peppered at ({fi},{fj},{fk}): d = {}",
                        raw.total_ext()
                            / deck.sample(fi.floor(), fj.floor(), fk.floor()).total_ext()
                    );
                }
            }
        }
        // The single-cell popcorn tent still granulates (the protection must not
        // disable the feature).
        let tent = build_volume(nx, ny, nz, 250.0, 3000.0, |i, j, k| {
            if i == 12 && j == 12 && k == 3 {
                (2.0e-2, 0.0, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        });
        let mut changed = 0usize;
        for pi in 0..60 {
            for pj in 0..60 {
                let fi = 11.05 + pi as f64 * 0.032;
                let fj = 11.05 + pj as f64 * 0.032;
                let raw = tent.sample(fi, fj, 3.0);
                let ero = tent.sample_granulated(fi, fj, 3.0, gran);
                if ero.total_ext() < raw.total_ext() {
                    changed += 1;
                }
            }
        }
        assert!(
            changed > 100,
            "the popcorn tent must still granulate under interior protection: {changed}"
        );
    }

    #[test]
    fn granulation_is_live_deterministic_and_consistent_across_marches() {
        // The eroded field must be what the PRIMARY march, the SECONDARY sun march
        // and the SUN-OD map all sample: marching with granulation can only RAISE the
        // view transmittance (subtract-only), the sun-OD map can only fall, both are
        // deterministic, and two identically-built volumes agree bit-for-bit (which
        // is exactly the geo == top-down agreement: both views sample the same
        // sampler at the same brick coordinates).
        let (nx, ny, nz) = (24usize, 24usize, 16usize);
        let build = || {
            build_volume(nx, ny, nz, 250.0, 3000.0, |i, j, k| {
                if (8..14).contains(&i) && (8..14).contains(&j) && (2..8).contains(&k) {
                    (1.5e-2, 0.0, 0.0)
                } else {
                    (0.0, 0.0, 0.0)
                }
            })
        };
        let vol = build();
        let vol2 = build();
        let gran = Some(Granulation::for_grid(3000.0));
        // Sample-level agreement across identically-built volumes (the view-agnostic
        // determinism statement).
        for &(fi, fj, fk) in &[
            (8.3f64, 9.7f64, 3.2f64),
            (10.1, 8.6, 5.5),
            (12.9, 13.4, 2.1),
        ] {
            let a = vol.sample_granulated(fi, fj, fk, gran);
            let b = vol2.sample_granulated(fi, fj, fk, gran);
            assert_eq!(a.ext_liquid.to_bits(), b.ext_liquid.to_bits());
            assert_eq!(a.ext_ice.to_bits(), b.ext_ice.to_bits());
            assert_eq!(a.ext_precip.to_bits(), b.ext_precip.to_bits());
        }
        let georef = test_georef(nx, ny, 3000.0);
        let mip = OccupancyMip::build(&vol, OCCUPANCY_MIP_FACTOR);
        let (luts, sky_sh) = shared_luts();
        let center = brick_to_ecef(&georef, 11.0, 11.0, 5.0, 0.0, 250.0).unwrap();
        let up = norm3(center);
        let (east, _) = perp_basis(up);
        let e = 40.0f64.to_radians();
        let sun = norm3(add3(scl3(up, e.sin()), scl3(east, e.cos())));
        // The sun-OD map over the eroded field never exceeds the raw one.
        let od_off = accumulate_sun_od(&vol, &georef, sun, 48);
        let od_on =
            accumulate_sun_od_granulated(&vol, &georef, sun, 48, SUN_OD_EDGE_FEATHER_TEXELS, gran);
        let mut od_less = 0usize;
        for (a, b) in od_on.od.iter().zip(od_off.od.iter()) {
            assert!(a <= b, "granulated sun-OD grew: {a} > {b}");
            if a < b {
                od_less += 1;
            }
        }
        assert!(od_less > 0, "the granulated sun-OD map should differ");
        // The march: granulation can only raise the view transmittance; at least one
        // ray differs (the feature is live end-to-end); the march is deterministic.
        let cam = CameraGeometry::from_sub_lon(-100.0);
        let base_cfg = MarchConfig {
            sun_march_jitter_amp: 0.0,
            ..MarchConfig::new(StepQuality::Offline, vol.voxel_pitch_m())
        };
        let scene_off = CloudScene {
            vol: &vol,
            mip: &mip,
            sun_od: &od_off,
            georef: &georef,
            luts,
            sky_sh,
            sun_ecef: sun,
            cfg: base_cfg,
        };
        let scene_on = CloudScene {
            vol: &vol,
            mip: &mip,
            sun_od: &od_on,
            georef: &georef,
            luts,
            sky_sh,
            sun_ecef: sun,
            cfg: MarchConfig {
                granulation: gran,
                ..base_cfg
            },
        };
        let mut any_differ = false;
        for &(gi, gj) in &[(9.0f64, 9.0f64), (10.5, 11.5), (11.0, 9.5), (12.5, 12.0)] {
            let target = brick_to_ecef(&georef, gi, gj, 0.0, 0.0, 250.0).unwrap();
            let view = norm3([
                target[0] - cam.camera[0],
                target[1] - cam.camera[1],
                target[2] - cam.camera[2],
            ]);
            let m_off = march_cloud(&scene_off, cam.camera, view);
            let m_on = march_cloud(&scene_on, cam.camera, view);
            // Subtract-only through the march — VALID ONLY when the un-granulated
            // march did NOT hit the transmittance-floor early exit: past the floor
            // the thicker march STOPS integrating while the thinner one continues,
            // so both are "effectively opaque" but their sub-floor values are not
            // ordered (both marches share the step trajectory otherwise — the mip,
            // not the sample, sizes the steps). The sampler-level subtract-only
            // invariant is pinned exactly in its own test.
            if m_off.transmittance > base_cfg.transmittance_floor {
                assert!(
                    m_on.transmittance >= m_off.transmittance - 1e-12,
                    "granulation must not thicken a ray: {} < {}",
                    m_on.transmittance,
                    m_off.transmittance
                );
            }
            if (m_on.transmittance - m_off.transmittance).abs() > 1e-9
                || m_on.inscatter != m_off.inscatter
            {
                any_differ = true;
            }
            let m_on2 = march_cloud(&scene_on, cam.camera, view);
            assert_eq!(
                m_on.transmittance.to_bits(),
                m_on2.transmittance.to_bits(),
                "the granulated march must be deterministic"
            );
            for c in 0..3 {
                assert_eq!(m_on.inscatter[c].to_bits(), m_on2.inscatter[c].to_bits());
            }
        }
        assert!(any_differ, "granulation should be live through the march");
    }
}
