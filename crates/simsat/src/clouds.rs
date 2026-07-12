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
use crate::bricks::{LogQuant, StorageProfile, VolumeBrick, decode_log2_f16};
use crate::camera::{ScanGrid, SurfaceRaster};
use crate::fractional_clouds::{
    DETERMINISTIC_SUBCOLUMN_COUNT, deterministic_subcolumn_u_for_count, maximum_overlap_closure,
};
use crate::frame::GridGeoref;
use crate::render::{
    CLOUD_SOFTCLIP_KNEE, FrameContext, GROUND_DAY_LIFT, SurfacePixel,
    radiance_to_rgba_softclip_with_synthetic_green, surface_toa_radiance,
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

/// Runtime visible-cloud optical-depth scale bounds. `0` is a deliberate QA/off
/// endpoint; values above four are capped because exponentials are already visually
/// saturated there while extreme/invalid inputs can otherwise destabilise a march.
/// The stored brick extinction and derived cloud-optical-depth products remain raw.
pub const CLOUD_OPTICAL_DEPTH_SCALE_MIN: f32 = 0.0;
pub const CLOUD_OPTICAL_DEPTH_SCALE_MAX: f32 = 4.0;
/// Shipped visible-cloud optical-depth calibration: `0.15`, selected by the owner
/// after broad cross-file visual review. This supersedes the earlier tied
/// `0.20`/`0.30` midpoint candidate and is not a claimed physical optimum. `1.0`
/// remains available as the unscaled model-extinction A/B.
pub const DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE: f32 = 0.15;

/// Validate a user-facing visible-cloud optical-depth scale. Non-finite values fall
/// back to the shipped calibration rather than silently erasing/saturating cloud.
#[inline]
pub fn validated_cloud_optical_depth_scale(scale: f32) -> f32 {
    if scale.is_finite() {
        scale.clamp(CLOUD_OPTICAL_DEPTH_SCALE_MIN, CLOUD_OPTICAL_DEPTH_SCALE_MAX)
    } else {
        DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE
    }
}

/// Probability that an additional scattering event is available in cloud optical
/// depth `tau`. Raising this once per higher octave gives the correct thin limit:
/// order `k` vanishes as `tau^k`, while optically thick cloud tends to the unchanged
/// Wrenninge/Oz octave sum. Octave zero is never gated.
#[inline]
fn multiscatter_thin_gate(tau: f64) -> f64 {
    1.0 - (-tau.max(0.0)).exp()
}

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

/// Resolve the visible cloud edge-feather band for a rendered camera raster.
///
/// With `feather_exposed_domain_edges == false`, this is EXACTLY the pre-v0.1.5
/// margin-gated behavior from [`edge_feather_cells_for_margin`]. When the shipped
/// v0.1.5 control is enabled, a raster containing any sample without a finite WRF
/// `(i, j)` also selects the existing fixed 4% band. That covers geostationary
/// bounding-box corners and perspective sky/outside-domain rays without fading a
/// fully in-domain top-down raster at zero margin.
///
/// `grid_i` and `grid_j` are expected to be the same camera-raster shape. A shape
/// mismatch is treated as exposed rather than silently disabling the safety fade.
#[inline]
pub fn edge_feather_cells_for_raster(
    margin_frac: f64,
    nx: usize,
    ny: usize,
    feather_exposed_domain_edges: bool,
    grid_i: &[f32],
    grid_j: &[f32],
) -> f64 {
    let legacy = edge_feather_cells_for_margin(margin_frac, nx, ny);
    if !feather_exposed_domain_edges || legacy > 0.0 {
        return legacy;
    }
    let exposes_outside = grid_i.len() != grid_j.len()
        || grid_i
            .iter()
            .zip(grid_j)
            .any(|(&i, &j)| !(i.is_finite() && j.is_finite()));
    if exposes_outside {
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
//   sigma' = sigma * B(m(d, e)),  m = remap-style multiplier (Nubis/Decima lineage),
//   B      = the BIMODAL carve shape (round 2): gap-or-grain, see GRAN_BIMODAL_*,
//   d = sigma / (trilinear-corner-neighbourhood max)  — the RELATIVE density,
//   e = amplitude/GRAN_AMP_CAP * gate * coherence * noise  — the erosion threshold,
//
// where `noise` is a 3-octave 2-D WORLEY (cellular) F1 field — cumulus morphology per
// the Nubis/Decima detail-erosion lineage (Schneider, "The Real-Time Volumetric
// Cloudscapes of Horizon: Zero Dawn", SIGGRAPH 2015 Advances) — with octave weights
// following a k^-5/3 spectrum envelope (observed liquid-water spectra: Davis et al.
// 1996, JAS 53; the 3DCLOUD realism criterion: Szczap et al. 2014, GMD 7), base cell
// scale a few hundred metres to ~1 km (the dominant element scale of the observed
// cloud-size distribution, power-law exponent ~1.66: Wood & Field 2011, J. Climate 24).
// The sample coordinates are DOMAIN-WARPED (round 2) by a low-frequency value noise
// from the same hash family before the Worley octaves, so cell size/spacing varies
// smoothly across the scene instead of reading as a mechanical uniform lattice at
// domain scale (see GRAN_WARP_*).
// The noise is 2-D (horizontal) so grains are vertically-coherent COLUMNS, the honest
// morphology of boundary-layer elements, and it is anchored DETERMINISTICALLY in
// brick/world space (position-hash cells, the hash01_position style): the same run
// renders the same field every frame (loops do not shimmer) and the geostationary and
// top-down views agree exactly (both sample by fractional brick coordinates).
//
// `coherence` (round 2, the DECK-COHERENCE GATE) is a per-COLUMN 2-D field built once
// per composite from the volume itself ([`GranCoherence::build`]): granulation is
// applied only where the local NEIGHBOURHOOD is genuinely BROKEN/partial cloud. A
// spatially coherent deck — near-total cloud fill AND low resolved column-tau
// variability over the 15x15-cell DECK-SCALE window (~45 km at 3 km dx; a band
// sheet is locally filled at 7x7 but never at deck scale) — is a SOLID CORE; the
// gate closes granulation over every core and (via a distance-tapered dilation)
// over the deck's MARGINS, so a stratiform sheet stays closed everywhere while
// broken-cu / confluence-band regions keep full granulation. This is the FSD literature's regime dependence made explicit (Boutle
// et al. 2014, QJRMS 140: fractional standard deviation of liquid falls toward
// overcast; Shonk et al. 2010): the represented sub-grid variability is LARGE in
// broken regimes and SMALL inside solid stratiform — round 1 applied one amplitude
// everywhere and peppered deck interiors/margins with pinholes (the owner's "cheese
// grater"), which the interior-protection window alone could not stop because
// moderate-thickness rings around an optical core legitimately sit at d < 0.75.
// The same field also carries the OPTICAL-THINNESS standdown (owner round-2
// addendum): where the neighbourhood-max eligible tau is small (a regionally
// translucent veil) granulation stands down and the honest transmittance gradient
// carries the wispy look; erosion earns its keep only on optically substantial,
// unresolved broken cloud (see GRAN_THIN_TAU_LO/HI).
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
// The DECK-COHERENCE field rides the [`SunOdMap`] (`gran_coherence`): the sun-OD
// accumulation is the one per-composite call that receives BOTH the volume and the
// granulation value from every render assembly (api + studio), so it builds the
// coherence once, applies it to its own accumulation, and carries it to the marches
// through [`CloudScene::sun_od`] — the assemblies themselves stay untouched.
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
/// Round-2 retune 0.52/0.62 -> 0.46/0.58: the bimodal carve restores the broad
/// partial-erosion halo that round 1's carpet texture leaned on, so the gap NETWORK
/// itself must be wider for grains to read as separated elements; the Cahalan
/// mean-tau bound is re-verified by test at this width.
pub const GRAN_CARVE_LO: f64 = 0.46;
/// See [`GRAN_CARVE_LO`].
pub const GRAN_CARVE_HI: f64 = 0.58;
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

// ── round-2 tuning: DECK-COHERENCE GATE / BIMODAL CARVE / DOMAIN WARP ─────────

/// The neighbourhood window side (cells) of the OPTICAL-THINNESS measure — the
/// model's unresolved scale, ~7 dx (Skamarock 2004 effective resolution). Odd;
/// radius 3.
pub const GRAN_COHERENCE_WINDOW_CELLS: usize = 7;
/// The SOLID-CORE window side (cells): a deck must be filled AND uniform over this
/// much larger scale (~15 dx = 45 km at 3 km) to close granulation. The first
/// round-2 render used the 7x7 unresolved-scale window here and the core test
/// fired INSIDE confluence-band sheets (locally filled/uniform 7x7 patches exist
/// all through a band field), stamping protection squares across the exact region
/// the owner wants granular — a real stratiform deck distinguishes itself from an
/// unresolved-cu sheet by coherence over tens of km, not one effective-resolution
/// cell. Odd; radius 7.
pub const GRAN_COHERENCE_CORE_WINDOW_CELLS: usize = 15;
/// A column counts as CLOUD for the neighbourhood fill fraction when its
/// granulation-eligible (liquid, height-gated) optical depth exceeds this. Low on
/// purpose: soft deck fringes (tau well below optical thickness) must count as part
/// of the deck's mass so a margin window reads FULL, not broken.
pub const GRAN_COHERENCE_TAU_CLOUDY: f64 = 0.25;
/// SOLID-CORE fill window: a 7x7 neighbourhood is a solid core only when its cloud
/// fill fraction rises through this band toward total ([`GRAN_COHERENCE_FILL_HI`]).
pub const GRAN_COHERENCE_FILL_LO: f64 = 0.85;
/// See [`GRAN_COHERENCE_FILL_LO`].
pub const GRAN_COHERENCE_FILL_HI: f64 = 0.97;
/// SOLID-CORE uniformity window on the RESOLVED column-tau fractional standard
/// deviation (std/mean over the 7x7 window): a filled-but-internally-variable
/// convective sheet (fsd well above this) is NOT a coherent deck and keeps
/// granulating; a smooth stratiform deck (fsd below) is. The regime split is the
/// FSD literature's (Boutle et al. 2014: overcast fsd ~0.2-0.4, broken ~0.7-1+).
pub const GRAN_COHERENCE_FSD_LO: f64 = 0.40;
/// See [`GRAN_COHERENCE_FSD_LO`].
pub const GRAN_COHERENCE_FSD_HI: f64 = 0.75;
/// OPTICAL-THINNESS standdown (owner round-2 addendum): where the neighbourhood is
/// optically THIN (7x7-window MAX eligible column tau at/below this), granulation
/// stands down entirely — a translucent veil already gets its wispy texture from the
/// honest transmittance gradient of the physics; erosion there reads as dark specks
/// in grey, never as grains. Granulation earns its keep only on optically
/// SUBSTANTIAL-but-unresolved broken cloud (window max at/above
/// [`GRAN_THIN_TAU_HI`], smoothstep between). Keyed on the neighbourhood MAX (not
/// the point column) so the thin trilinear SKIRT of a thick element still carves —
/// its neighbourhood is substantial; only regionally-thin veils stand down.
pub const GRAN_THIN_TAU_LO: f64 = 0.5;
/// See [`GRAN_THIN_TAU_LO`].
pub const GRAN_THIN_TAU_HI: f64 = 2.5;
/// CORE-REGION EROSION radius (cells, Chebyshev min-filter on the solid-core
/// strength): a core counts only if it is INTERIOR to a coherent core REGION —
/// with the core window this demands deck coherence over ~(15 + 2x4) = 23 cells
/// (~70 km at 3 km dx). An isolated deck-scale patch that barely qualifies inside a
/// broken-cu field is part of the broken REGIME (the first 15x15 retune stamped
/// protection squares on 45-60 km patches all through the confluence bands; radius
/// 2 still left the larger ones); a real stratiform deck is coherent over hundreds
/// of km and survives easily.
pub const GRAN_CORE_ERODE_CELLS: usize = 4;
/// Deck-margin protection reach (cells, Chebyshev — the core window's own metric, so
/// deck corners are covered exactly like edges): a column within this distance of an
/// ERODED solid-core centre is FULLY closed (the dilation that keeps a coherent
/// deck's soft margins closed). Eroded cores sit >= core-window-radius (7) + the
/// erosion (4) + the physical fringe width (~2 cells) inside the cloudy mask, and
/// the protection must also cover the first CLEAR column past the fringe (the
/// trilinear tent between the last cloudy column and it carries real extinction and
/// reads the gate bilinearly) — hence 7 + 4 + 2 + 1 = 14.
pub const GRAN_PROTECT_FULL_CELLS: f64 = 14.0;
/// Protection tapers smoothly from full at [`GRAN_PROTECT_FULL_CELLS`] to none here
/// (no hard granulated/closed seam next to a deck).
pub const GRAN_PROTECT_ZERO_CELLS: f64 = 19.0;
/// BIMODAL CARVE (round 2): the remap multiplier `m` is reshaped grain-or-gap by
/// `smoothstep(GRAN_BIMODAL_GAP, GRAN_BIMODAL_GRAIN, m)` — at/below the GAP point the
/// sample carves to CLEAR, at/above the GRAIN point it is restored to FULL extinction
/// (still `<=` the raw sample: subtract-only holds by construction), and the
/// half-eroded middle band that read as translucent GREY mush is squeezed into a
/// steep transition. Real high-sun cu fields are high-contrast (white grain / clear
/// gap), not translucent; round 1's plain remap left most eroded samples mid-band.
pub const GRAN_BIMODAL_GAP: f64 = 0.25;
/// See [`GRAN_BIMODAL_GAP`].
pub const GRAN_BIMODAL_GRAIN: f64 = 0.65;
/// DOMAIN WARP (round 2): the Worley sample position is displaced by a smooth
/// low-frequency value noise (same deterministic hash family) with this correlation
/// scale (m) ...
pub const GRAN_WARP_SCALE_M: f64 = 4000.0;
/// ... and this maximum displacement (m, per axis). The warp gradient (~2 x amp /
/// scale) locally stretches/compresses/shears the cell lattice ~30-60%, so grain
/// size and spacing vary across the scene — the uniform cell spacing is what read
/// as mechanical ("cheese grater") at domain scale. Deterministic, view-agnostic.
pub const GRAN_WARP_AMP_M: f64 = 1300.0;
/// Hash salts of the warp's two displacement channels (fixed, arbitrary).
pub const GRAN_WARP_SALT_U: u32 = 0x1B56_C4E9;
/// See [`GRAN_WARP_SALT_U`].
pub const GRAN_WARP_SALT_V: u32 = 0x7A99_1E3D;

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

/// Smooth 2-D VALUE NOISE in `[0, 1)`: the cell-corner hashes of the same
/// deterministic family, blended with a smoothstep interpolant (C1). Low-frequency
/// input for the domain warp.
fn gran_value_noise(qx: f64, qy: f64, salt: u32) -> f64 {
    let bx = qx.floor();
    let by = qy.floor();
    let (ix, iy) = (bx as i64, by as i64);
    let tx = smooth01(qx - bx);
    let ty = smooth01(qy - by);
    let h00 = gran_cell_hash01(ix, iy, salt);
    let h10 = gran_cell_hash01(ix + 1, iy, salt);
    let h01 = gran_cell_hash01(ix, iy + 1, salt);
    let h11 = gran_cell_hash01(ix + 1, iy + 1, salt);
    (h00 * (1.0 - tx) + h10 * tx) * (1.0 - ty) + (h01 * (1.0 - tx) + h11 * tx) * ty
}

/// The DOMAIN-WARP displacement (m) of the granulation noise at a brick-plane
/// position (m): two independent smooth value-noise channels ([`GRAN_WARP_SALT_U`] /
/// [`GRAN_WARP_SALT_V`]) at correlation scale [`GRAN_WARP_SCALE_M`], each mapped to
/// `[-GRAN_WARP_AMP_M, GRAN_WARP_AMP_M]`. Deterministic and view-agnostic (a pure
/// function of the position, like the erosion noise it feeds); its smooth gradient
/// is what varies the Worley cell size/spacing across the scene.
pub fn granulation_warp_offset(u_m: f64, v_m: f64) -> (f64, f64) {
    let qx = u_m / GRAN_WARP_SCALE_M;
    let qy = v_m / GRAN_WARP_SCALE_M;
    let du = GRAN_WARP_AMP_M * (2.0 * gran_value_noise(qx, qy, GRAN_WARP_SALT_U) - 1.0);
    let dv = GRAN_WARP_AMP_M * (2.0 * gran_value_noise(qx, qy, GRAN_WARP_SALT_V) - 1.0);
    (du, dv)
}

/// The GRANULATION EROSION FIELD at a horizontal position (m) in the brick plane:
/// the position is first DOMAIN-WARPED ([`granulation_warp_offset`] — round 2, cell
/// size/spacing varies across the scene), then the k^-5/3-weighted 3-octave Worley
/// F1 ([`GRAN_OCTAVE_SCALES_M`] / [`GRAN_OCTAVE_WEIGHTS`]) is tail-shaped through
/// `smoothstep(GRAN_CARVE_LO, GRAN_CARVE_HI, W)` into `[0, 1]`: 0 over grain
/// interiors (no erosion), 1 on the carved gap network. Pure and deterministic in
/// the position (memoised per thread for the exact repeated `(u, v)` a nadir
/// top-down column produces — a bit-exact cache, never an approximation).
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
    let (du, dv) = granulation_warp_offset(u_m, v_m);
    let (uw, vw) = (u_m + du, v_m + dv);
    let mut w = 0.0f64;
    for i in 0..GRAN_OCTAVE_SCALES_M.len() {
        let lam = GRAN_OCTAVE_SCALES_M[i];
        w += GRAN_OCTAVE_WEIGHTS[i] * worley2_f1(uw / lam, vw / lam, GRAN_OCTAVE_SALTS[i]);
    }
    let e = smooth01((w.clamp(0.0, 1.0) - GRAN_CARVE_LO) / (GRAN_CARVE_HI - GRAN_CARVE_LO));
    MEMO.with(|m| m.set((key.0, key.1, e)));
    e
}

/// Pixel-footprint-filtered granulation erosion field.
///
/// The raw field contains 250 m and 500 m octaves. Sampling those once per 500 m
/// model/output cell aliases them into square islands. This deterministic 2x2
/// Gauss-Legendre quadrature approximates the area mean over a square footprint,
/// preserving the larger-scale structure while band-limiting detail that the grid
/// cannot resolve. A non-positive or non-finite footprint is the exact legacy point
/// sample.
pub fn granulation_erosion_noise_footprint(u_m: f64, v_m: f64, footprint_m: f64) -> f64 {
    if !footprint_m.is_finite() || footprint_m <= 0.0 {
        return granulation_erosion_noise(u_m, v_m);
    }
    thread_local! {
        static MEMO: std::cell::Cell<(u64, u64, u64, f64)> =
            const { std::cell::Cell::new((0, 0, 0, -1.0)) };
    }
    let key = (u_m.to_bits(), v_m.to_bits(), footprint_m.to_bits());
    let hit = MEMO.with(|m| m.get());
    if hit.3 >= 0.0 && hit.0 == key.0 && hit.1 == key.1 && hit.2 == key.2 {
        return hit.3;
    }

    // Two-point Gauss-Legendre nodes mapped from [-1, 1] to a square of full
    // width `footprint_m`: +/- footprint/(2*sqrt(3)) on each axis.
    let d = footprint_m / (2.0 * 3.0_f64.sqrt());
    let filtered = 0.25
        * (granulation_erosion_noise(u_m - d, v_m - d)
            + granulation_erosion_noise(u_m + d, v_m - d)
            + granulation_erosion_noise(u_m - d, v_m + d)
            + granulation_erosion_noise(u_m + d, v_m + d));
    MEMO.with(|m| m.set((key.0, key.1, key.2, filtered)));
    filtered
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

/// The BIMODAL CARVE shape (round 2) applied to the remap multiplier:
/// `smoothstep(GRAN_BIMODAL_GAP, GRAN_BIMODAL_GRAIN, m)`. Monotone `[0,1] -> [0,1]`
/// with `B(0) = 0` and `B(1) = 1`; at/below the GAP point the sample carves to CLEAR,
/// at/above the GRAIN point it is RESTORED to the full raw extinction (`B(m) <= 1`,
/// so the eroded sample never exceeds the raw one — subtract-only holds), and the
/// half-eroded middle that rendered as translucent grey mush is squeezed into a steep
/// transition (grain-or-gap, the high-contrast look of a real high-sun cu field).
pub fn granulation_bimodal(m: f64) -> f64 {
    smooth01((m.clamp(0.0, 1.0) - GRAN_BIMODAL_GAP) / (GRAN_BIMODAL_GRAIN - GRAN_BIMODAL_GAP))
}

/// The per-column DECK-COHERENCE GATE (round 2): `1.0` where the neighbourhood at the
/// unresolved scale is genuinely broken/partial cloud (granulate freely), `0.0` over
/// a spatially coherent deck AND its margins (stay closed). Built ONCE per composite
/// from the volume by [`accumulate_sun_od_granulated`] and carried to the marches on
/// the [`SunOdMap`] (`gran_coherence`) so every march of the composite reads the same
/// field. Pure function of the volume: deterministic, identical for the geostationary
/// and top-down views.
///
/// Construction (see the granulation section header for the science):
/// 1. Per column: the granulation-ELIGIBLE optical depth (liquid extinction through
///    the height ramp, vertically integrated).
/// 2. Per column, over the 15x15-cell DECK-SCALE window
///    ([`GRAN_COHERENCE_CORE_WINDOW_CELLS`]): the cloud FILL fraction (columns above
///    [`GRAN_COHERENCE_TAU_CLOUDY`]) and the resolved column-tau fractional standard
///    deviation (std/mean).
/// 3. SOLID CORE strength = `smoothstep(fill)` x `1 - smoothstep(fsd)` — filled AND
///    uniform (a stratiform deck) over tens of km, not filled-but-variable or
///    merely locally-filled (a convective / unresolved-cu sheet) — then ERODED by a
///    min-filter ([`GRAN_CORE_ERODE_CELLS`]): a core must be interior to a coherent
///    core REGION, so an isolated deck-scale patch inside a broken field stays part
///    of the broken regime.
/// 4. OPTICAL-THINNESS standdown: a smooth ramp of the window MAX eligible tau
///    ([`GRAN_THIN_TAU_LO`]/[`GRAN_THIN_TAU_HI`]) — regionally-translucent veils
///    already read wispy through the honest transmittance gradient, so granulation
///    stands down there (erosion on a veil is dark specks in grey, not grains).
/// 5. Gate = thinness x (`1 -` the distance-tapered max of core strength within
///    [`GRAN_PROTECT_ZERO_CELLS`], full protection inside
///    [`GRAN_PROTECT_FULL_CELLS`]) — the dilation that keeps deck MARGINS closed.
#[derive(Debug, Clone)]
pub struct GranCoherence {
    nx: usize,
    ny: usize,
    gate: Vec<f32>,
}

impl GranCoherence {
    /// Build the gate field from a decoded volume (see the type docs).
    pub fn build(vol: &DecodedVolume) -> Self {
        let (nx, ny, nz) = (vol.nx, vol.ny, vol.nz);
        let n = nx * ny;
        if n == 0 {
            return Self {
                nx,
                ny,
                gate: Vec::new(),
            };
        }
        // 1. Granulation-eligible column optical depth (liquid x height ramp).
        let mut tau = vec![0.0f64; n];
        for k in 0..nz {
            let z = vol.z_min_m + k as f64 * vol.dz_m;
            if z >= GRAN_HEIGHT_ZERO_M {
                break; // layers are ascending in z; nothing eligible above
            }
            let ramp = 1.0
                - smooth01((z - GRAN_HEIGHT_FULL_M) / (GRAN_HEIGHT_ZERO_M - GRAN_HEIGHT_FULL_M));
            let plane = &vol.ext_liquid[k * n..(k + 1) * n];
            for (t, &e) in tau.iter_mut().zip(plane.iter()) {
                *t += e as f64 * ramp * vol.dz_m;
            }
        }
        // 2. Window stats via summed-area tables (exact box sums, edge-clamped).
        let sat = |f: &dyn Fn(usize) -> f64| -> Vec<f64> {
            let w = nx + 1;
            let mut s = vec![0.0f64; w * (ny + 1)];
            for j in 0..ny {
                let mut row = 0.0f64;
                for i in 0..nx {
                    row += f(j * nx + i);
                    s[(j + 1) * w + (i + 1)] = s[j * w + (i + 1)] + row;
                }
            }
            s
        };
        let sat_fill = sat(&|c| {
            if tau[c] >= GRAN_COHERENCE_TAU_CLOUDY {
                1.0
            } else {
                0.0
            }
        });
        let sat_tau = sat(&|c| tau[c]);
        let sat_tau2 = sat(&|c| tau[c] * tau[c]);
        // The SOLID-CORE stats window (15x15, deck scale) and the THINNESS window
        // (7x7, the unresolved scale) — see the constants for why they differ.
        let r_core = GRAN_COHERENCE_CORE_WINDOW_CELLS / 2;
        let r = GRAN_COHERENCE_WINDOW_CELLS / 2;
        let box_sum = |s: &[f64], i0: usize, i1: usize, j0: usize, j1: usize| -> f64 {
            let w = nx + 1;
            s[(j1 + 1) * w + (i1 + 1)] + s[j0 * w + i0]
                - s[j0 * w + (i1 + 1)]
                - s[(j1 + 1) * w + i0]
        };
        // 3. Solid-core strength per column (rows in parallel, flattened in order).
        let solid_rows: Vec<Vec<f32>> = (0..ny)
            .into_par_iter()
            .map(|j| {
                let j0 = j.saturating_sub(r_core);
                let j1 = (j + r_core).min(ny - 1);
                (0..nx)
                    .map(|i| {
                        let i0 = i.saturating_sub(r_core);
                        let i1 = (i + r_core).min(nx - 1);
                        let count = ((i1 + 1 - i0) * (j1 + 1 - j0)) as f64;
                        let fill = box_sum(&sat_fill, i0, i1, j0, j1) / count;
                        let fill_term = smooth01(
                            (fill - GRAN_COHERENCE_FILL_LO)
                                / (GRAN_COHERENCE_FILL_HI - GRAN_COHERENCE_FILL_LO),
                        );
                        if fill_term <= 0.0 {
                            return 0.0f32;
                        }
                        let mean = box_sum(&sat_tau, i0, i1, j0, j1) / count;
                        if mean <= 1.0e-9 {
                            return 0.0f32;
                        }
                        let var =
                            (box_sum(&sat_tau2, i0, i1, j0, j1) / count - mean * mean).max(0.0);
                        let fsd = var.sqrt() / mean;
                        let uniform_term = 1.0
                            - smooth01(
                                (fsd - GRAN_COHERENCE_FSD_LO)
                                    / (GRAN_COHERENCE_FSD_HI - GRAN_COHERENCE_FSD_LO),
                            );
                        (fill_term * uniform_term) as f32
                    })
                    .collect()
            })
            .collect();
        let mut solid_raw = Vec::with_capacity(n);
        for row in solid_rows {
            solid_raw.extend(row);
        }
        // 3b. CORE-REGION EROSION (min-filter): a core must be INTERIOR to a coherent
        // core region — isolated deck-scale stamps inside a broken field die here.
        let re = GRAN_CORE_ERODE_CELLS;
        let solid: Vec<f32> = {
            let eroded_rows: Vec<Vec<f32>> = (0..ny)
                .into_par_iter()
                .map(|j| {
                    let j0 = j.saturating_sub(re);
                    let j1 = (j + re).min(ny - 1);
                    (0..nx)
                        .map(|i| {
                            let i0 = i.saturating_sub(re);
                            let i1 = (i + re).min(nx - 1);
                            let mut m = 1.0f32;
                            for jj in j0..=j1 {
                                for ii in i0..=i1 {
                                    let s = solid_raw[jj * nx + ii];
                                    if s < m {
                                        m = s;
                                    }
                                }
                            }
                            m
                        })
                        .collect()
                })
                .collect();
            let mut v = Vec::with_capacity(n);
            for row in eroded_rows {
                v.extend(row);
            }
            v
        };
        // 4. OPTICAL-THINNESS standdown (owner round-2 addendum): the 7x7-window MAX
        // eligible tau through the smooth ramp — regionally-thin translucent veils
        // stand down (their wispy look is the honest transmittance gradient); the
        // thin skirt of a SUBSTANTIAL element keeps carving (window max is high).
        let thin_rows: Vec<Vec<f32>> = (0..ny)
            .into_par_iter()
            .map(|j| {
                let j0 = j.saturating_sub(r);
                let j1 = (j + r).min(ny - 1);
                (0..nx)
                    .map(|i| {
                        let i0 = i.saturating_sub(r);
                        let i1 = (i + r).min(nx - 1);
                        let mut wmax = 0.0f64;
                        for jj in j0..=j1 {
                            for ii in i0..=i1 {
                                let t = tau[jj * nx + ii];
                                if t > wmax {
                                    wmax = t;
                                }
                            }
                        }
                        smooth01((wmax - GRAN_THIN_TAU_LO) / (GRAN_THIN_TAU_HI - GRAN_THIN_TAU_LO))
                            as f32
                    })
                    .collect()
            })
            .collect();
        let mut thin = Vec::with_capacity(n);
        for row in thin_rows {
            thin.extend(row);
        }
        // 5. Distance-tapered dilation: protection = max of solid x taper(distance);
        // gate = thinness x (1 - protection).
        let reach = GRAN_PROTECT_ZERO_CELLS.ceil() as usize;
        let any_solid = solid.iter().any(|&s| s > 0.0);
        let gate: Vec<f32> = if !any_solid {
            thin
        } else {
            let gate_rows: Vec<Vec<f32>> = (0..ny)
                .into_par_iter()
                .map(|j| {
                    (0..nx)
                        .map(|i| {
                            let thin_here = thin[j * nx + i] as f64;
                            if thin_here <= 0.0 {
                                return 0.0f32;
                            }
                            let j_lo = j.saturating_sub(reach);
                            let j_hi = (j + reach).min(ny - 1);
                            let i_lo = i.saturating_sub(reach);
                            let i_hi = (i + reach).min(nx - 1);
                            let mut protection = 0.0f64;
                            'scan: for jj in j_lo..=j_hi {
                                for ii in i_lo..=i_hi {
                                    let s = solid[jj * nx + ii] as f64;
                                    // s is an upper bound of s * taper: skip fast.
                                    if s <= protection {
                                        continue;
                                    }
                                    let d = (ii as i64 - i as i64)
                                        .unsigned_abs()
                                        .max((jj as i64 - j as i64).unsigned_abs())
                                        as f64;
                                    let taper = 1.0
                                        - smooth01(
                                            (d - GRAN_PROTECT_FULL_CELLS)
                                                / (GRAN_PROTECT_ZERO_CELLS
                                                    - GRAN_PROTECT_FULL_CELLS),
                                        );
                                    protection = protection.max(s * taper);
                                    if protection >= 1.0 {
                                        break 'scan;
                                    }
                                }
                            }
                            (thin_here * (1.0 - protection)).clamp(0.0, 1.0) as f32
                        })
                        .collect()
                })
                .collect();
            let mut gate = Vec::with_capacity(n);
            for row in gate_rows {
                gate.extend(row);
            }
            gate
        };
        Self { nx, ny, gate }
    }

    /// Bilinear sample of the gate at fractional column coords (clamp-to-edge;
    /// out-of-domain / non-finite reads the nearest edge column — such samples are
    /// CLEAR anyway). `1.0` = granulate freely, `0.0` = closed (coherent deck).
    #[inline]
    pub fn gate_at(&self, fi: f64, fj: f64) -> f64 {
        if self.gate.is_empty() {
            return 1.0;
        }
        let fi = if fi.is_finite() {
            fi.clamp(0.0, (self.nx - 1) as f64)
        } else {
            0.0
        };
        let fj = if fj.is_finite() {
            fj.clamp(0.0, (self.ny - 1) as f64)
        } else {
            0.0
        };
        let i0 = fi.floor() as usize;
        let j0 = fj.floor() as usize;
        let i1 = (i0 + 1).min(self.nx - 1);
        let j1 = (j0 + 1).min(self.ny - 1);
        let ti = fi - i0 as f64;
        let tj = fj - j0 as f64;
        let g = |i: usize, j: usize| self.gate[j * self.nx + i] as f64;
        let g0 = g(i0, j0) * (1.0 - ti) + g(i1, j0) * ti;
        let g1 = g(i0, j1) * (1.0 - ti) + g(i1, j1) * ti;
        g0 * (1.0 - tj) + g1 * tj
    }

    /// Summary counts over the gate field `(open >= 0.9, partial, closed <= 0.1)` —
    /// the render-log / QA diagnostic.
    pub fn stats(&self) -> (usize, usize, usize) {
        let mut open = 0usize;
        let mut partial = 0usize;
        let mut closed = 0usize;
        for &g in &self.gate {
            if g >= 0.9 {
                open += 1;
            } else if g <= 0.1 {
                closed += 1;
            } else {
                partial += 1;
            }
        }
        (open, partial, closed)
    }
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
    octave_sun_source_thin_gated(
        cos_theta,
        ext_liquid,
        ext_ice_precip,
        tau_sun,
        beer_powder_on,
        octaves,
        f64::INFINITY,
    )
}

/// [`octave_sun_source`] with the physically required optically-thin limit. The
/// compatibility wrapper above retains the legacy ungated helper contract; visible
/// render marches call this function with the larger of local and column cloud OD.
/// Order zero is byte-identical to single scatter. Every higher order is weighted by
/// another `1 - exp(-support_tau)`, so it disappears when there is insufficient cloud
/// for another interaction and approaches the former result for thick cloud.
#[inline]
pub fn octave_sun_source_thin_gated(
    cos_theta: f64,
    ext_liquid: f64,
    ext_ice_precip: f64,
    tau_sun: f64,
    beer_powder_on: bool,
    octaves: usize,
    support_tau: f64,
) -> f64 {
    let mut acc = 0.0f64;
    let mut ext_scale = 1.0f64; // a^k
    let mut g_scale = 1.0f64; // b^k
    let mut weight = 1.0f64; // c^k
    let thin_gate = multiscatter_thin_gate(support_tau);
    let mut order_gate = 1.0f64;
    for k in 0..octaves.max(1) {
        if k > 0 {
            order_gate *= thin_gate;
        }
        let tau_k = tau_sun * ext_scale;
        let vis_k = if beer_powder_on {
            beer_powder(tau_k)
        } else {
            beer(tau_k)
        };
        let phase_k = aggregate_phase_scaled(cos_theta, ext_liquid, ext_ice_precip, g_scale);
        acc += order_gate * weight * phase_k * vis_k;
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

/// Project the apparent solar disk radius to an occluder's distance. A convolution
/// radius uses the disk half-angle, not its full angular diameter.
#[inline]
fn solar_penumbra_radius_m(occluder_distance_m: f64) -> f64 {
    occluder_distance_m.max(0.0) * atmosphere::SUN_ANGULAR_RADIUS_RAD.tan()
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
    /// QSNOW-only auxiliary subset of `ext_precip`. This is metadata for the
    /// fractional-cloud closure and must never be added to total extinction.
    pub ext_snow: Vec<u8>,
    /// Quantization scale for the encoded `ext_snow` auxiliary.
    pub ext_snow_quant: LogQuant,
    /// Authoritative decoded snow auxiliary for ScienceCloudF16. Empty for the
    /// compact profile, which keeps the lower-residency encoded u8 path above.
    pub science_ext_snow: Vec<f32>,
    pub ext_precip: Vec<f32>,
    pub tau_up: Vec<f32>,
    /// Linear-u8 model cloud coverage, one code per voxel.
    pub cloud_fraction: Vec<u8>,
    /// True only when `cloud_fraction` came from a trusted model field.
    pub has_cloud_fraction: bool,
}

/// Diagnostics from applying model fractional cloud cover to a decoded volume.
///
/// The closure preserves the legacy field when coverage is unavailable, when all
/// covered condensate has code 255, or when the caller leaves the feature off.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FractionalCloudStats {
    pub available: bool,
    pub columns_total: usize,
    pub columns_modified: usize,
    pub fractional_layer_count: usize,
    pub repaired_zero_count: usize,
    pub raw_fractional_tau: f64,
    pub effective_fractional_tau: f64,
}

/// Diagnostics from materializing one member of the opt-in deterministic
/// four-subcolumn maximum-overlap reference ensemble.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FractionalSubcolumnStats {
    pub available: bool,
    pub subcolumn_index: usize,
    pub subcolumn_u: f64,
    pub fractional_layer_count: usize,
    pub cloudy_fractional_layer_count: usize,
    pub repaired_zero_count: usize,
    /// Grid-mean cloud optical depth represented by the source condensate.
    pub grid_mean_cloud_tau: f64,
    /// Cloud optical depth carried by this explicit subcolumn before the render
    /// calibration multiplier. Ensemble-mean closure is diagnosed by the caller.
    pub subcolumn_cloud_tau: f64,
}

/// Diagnostics from the opt-in top-down stratiform column regularizer.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StratiformRegularizationStats {
    pub columns_total: usize,
    pub low_cloud_seeds: usize,
    pub columns_changed: usize,
    pub tau_before: f64,
    pub tau_after: f64,
    pub min_scale: f32,
    pub max_scale: f32,
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
        Self::decode_brick(brick, horiz_pitch_m, true)
    }

    /// Decode only the channels needed by legacy visible/thermal occupancy paths.
    /// This avoids cloning fractional auxiliaries when the switch is off or a caller
    /// only needs total extinction.
    pub fn from_brick_legacy(brick: &VolumeBrick, horiz_pitch_m: f64) -> Self {
        Self::decode_brick(brick, horiz_pitch_m, false)
    }

    fn decode_brick(brick: &VolumeBrick, horiz_pitch_m: f64, fractional_aux: bool) -> Self {
        let ql = brick.quant.get("ext_liquid");
        let qi = brick.quant.get("ext_ice");
        let qs = brick.quant.get("ext_snow");
        let qp = brick.quant.get("ext_precip");
        let qt = brick.quant.get("tau_up");
        let science = (brick.storage_profile == StorageProfile::ScienceCloudF16)
            .then_some(brick.science_cloud_f16.as_ref())
            .flatten();
        let decode_science = |values: &[u16]| values.iter().map(|&v| decode_log2_f16(v)).collect();
        let mut volume = Self {
            nx: brick.nx,
            ny: brick.ny,
            nz: brick.nz,
            z_min_m: brick.z_min_m,
            dz_m: brick.dz_m,
            horiz_pitch_m,
            ext_liquid: science.map_or_else(
                || brick.ext_liquid.iter().map(|&c| ql.decode(c)).collect(),
                |payload| decode_science(&payload.ext_liquid),
            ),
            ext_ice: science.map_or_else(
                || brick.ext_ice.iter().map(|&c| qi.decode(c)).collect(),
                |payload| decode_science(&payload.ext_ice),
            ),
            ext_snow: if science.is_none() && fractional_aux && brick.has_cloud_fraction {
                brick.ext_snow.clone()
            } else {
                Vec::new()
            },
            ext_snow_quant: qs,
            science_ext_snow: if fractional_aux && brick.has_cloud_fraction {
                science.map_or_else(Vec::new, |payload| decode_science(&payload.ext_snow))
            } else {
                Vec::new()
            },
            ext_precip: science.map_or_else(
                || brick.ext_precip.iter().map(|&c| qp.decode(c)).collect(),
                |payload| decode_science(&payload.ext_precip),
            ),
            tau_up: brick.tau_up.iter().map(|&c| qt.decode(c)).collect(),
            cloud_fraction: if fractional_aux && brick.has_cloud_fraction {
                brick.cloud_fraction.clone()
            } else {
                Vec::new()
            },
            has_cloud_fraction: fractional_aux && brick.has_cloud_fraction,
        };
        if science.is_some() {
            volume.rebuild_tau_up_from_extinction();
        }
        volume
    }

    #[inline]
    fn snow_extinction(&self, cell: usize) -> f32 {
        if self.science_ext_snow.is_empty() {
            self.ext_snow_quant.decode(self.ext_snow[cell])
        } else {
            self.science_ext_snow[cell]
        }
    }

    fn rebuild_tau_up_from_extinction(&mut self) {
        if self.nz == 0 {
            return;
        }
        let dz = self.dz_m.max(0.0);
        for j in 0..self.ny {
            for i in 0..self.nx {
                let top = self.cell(i, j, self.nz - 1);
                self.tau_up[top] = 0.0;
                for k in (0..self.nz - 1).rev() {
                    let c0 = self.cell(i, j, k);
                    let c1 = self.cell(i, j, k + 1);
                    let beta0 = self.ext_liquid[c0] as f64
                        + self.ext_ice[c0] as f64
                        + self.ext_precip[c0] as f64;
                    let beta1 = self.ext_liquid[c1] as f64
                        + self.ext_ice[c1] as f64
                        + self.ext_precip[c1] as f64;
                    self.tau_up[c0] = (self.tau_up[c1] as f64 + 0.5 * (beta0 + beta1) * dz) as f32;
                }
            }
        }
    }

    /// Apply the model cloud-fraction field using a maximum-overlap column closure.
    ///
    /// WRF condensate is grid-mean mass. For each `(i, j)` column, partially covered
    /// liquid, ice, and snow layers are converted to the homogeneous optical depth
    /// with the same exact area-mean vertical transmittance. A single column scale is
    /// used so species ratios and vertical structure remain intact. Rain, graupel,
    /// and other precipitation remain full-cell; `ext_snow` identifies only the snow share inside
    /// the legacy total `ext_precip` channel, avoiding double counting. Code-zero
    /// layers with positive cloud extinction are conservatively repaired to full
    /// coverage by [`maximum_overlap_closure`].
    ///
    /// This is exact for the vertical maximum-overlap transfer and an intentionally
    /// deterministic homogeneous equivalent for slant rays. `tau_up` is recomputed
    /// in each changed column so ambient light and the view/sun marches consume one
    /// consistent extinction field.
    pub fn apply_fractional_clouds(&mut self) -> FractionalCloudStats {
        let cells = self.nx.saturating_mul(self.ny).saturating_mul(self.nz);
        let available = self.has_cloud_fraction
            && self.ext_liquid.len() == cells
            && self.ext_ice.len() == cells
            && (self.ext_snow.len() == cells || self.science_ext_snow.len() == cells)
            && self.ext_precip.len() == cells
            && self.tau_up.len() == cells
            && self.cloud_fraction.len() == cells;
        let mut stats = FractionalCloudStats {
            available,
            columns_total: self.nx.saturating_mul(self.ny),
            ..FractionalCloudStats::default()
        };
        if !available || self.nz == 0 {
            return stats;
        }

        let dz = self.dz_m.max(0.0);
        for j in 0..self.ny {
            for i in 0..self.nx {
                let closure = maximum_overlap_closure((0..self.nz).map(|k| {
                    let c = self.cell(i, j, k);
                    // Independent log quantization can put the snow auxiliary a few
                    // ulps above the total precip channel. Cap it to its parent so the
                    // auxiliary can never invent extinction.
                    let snow = (self.snow_extinction(c) as f64)
                        .max(0.0)
                        .min((self.ext_precip[c] as f64).max(0.0));
                    let cloud_ext = (self.ext_liquid[c] as f64).max(0.0)
                        + (self.ext_ice[c] as f64).max(0.0)
                        + snow;
                    (cloud_ext * dz, self.cloud_fraction[c])
                }));

                stats.fractional_layer_count += closure.fractional_layer_count;
                stats.repaired_zero_count += closure.repaired_zero_count;
                stats.raw_fractional_tau += closure.raw_fractional_tau;
                stats.effective_fractional_tau += closure.effective_fractional_tau;

                if closure.raw_fractional_tau <= 0.0 || closure.scale >= 1.0 {
                    continue;
                }
                let scale = closure.scale as f32;
                let mut changed = false;
                for k in 0..self.nz {
                    let c = self.cell(i, j, k);
                    if !(1..=254).contains(&self.cloud_fraction[c]) {
                        continue;
                    }
                    let liquid = self.ext_liquid[c].max(0.0);
                    let ice = self.ext_ice[c].max(0.0);
                    let precip = self.ext_precip[c].max(0.0);
                    let snow = self.snow_extinction(c).max(0.0).min(precip);
                    if liquid + ice + snow <= 0.0 {
                        continue;
                    }
                    self.ext_liquid[c] = liquid * scale;
                    self.ext_ice[c] = ice * scale;
                    self.ext_precip[c] = (precip - snow) + snow * scale;
                    changed = true;
                }
                if !changed {
                    continue;
                }

                stats.columns_modified += 1;
                let top = self.cell(i, j, self.nz - 1);
                self.tau_up[top] = 0.0;
                for k in (0..self.nz - 1).rev() {
                    let c0 = self.cell(i, j, k);
                    let c1 = self.cell(i, j, k + 1);
                    let beta0 = self.ext_liquid[c0] as f64
                        + self.ext_ice[c0] as f64
                        + self.ext_precip[c0] as f64;
                    let beta1 = self.ext_liquid[c1] as f64
                        + self.ext_ice[c1] as f64
                        + self.ext_precip[c1] as f64;
                    self.tau_up[c0] = (self.tau_up[c1] as f64 + 0.5 * (beta0 + beta1) * dz) as f32;
                }
            }
        }
        // These encoded auxiliaries are ingest metadata, not sampled render fields.
        // Release them after the one-shot closure to keep large-domain peak residency
        // bounded during the expensive ray march.
        self.ext_snow = Vec::new();
        self.science_ext_snow = Vec::new();
        self.cloud_fraction = Vec::new();
        self.has_cloud_fraction = false;
        stats
    }

    /// Materialize one fixed, stratified, maximum-overlap cloud subcolumn.
    ///
    /// Model liquid, cloud ice, and snow are grid-mean condensates. In a layer
    /// with model fraction `f`, the selected subcolumn therefore carries
    /// `beta/f` when its shared coordinate `u < f`, and zero cloud condensate
    /// otherwise. Code 255 remains a full layer; code 0 with positive cloud
    /// condensate is repaired to full coverage, matching the established
    /// effective-OD closure. Rain/graupel (the non-snow share of `ext_precip`)
    /// remain full-cell precipitation in every subcolumn.
    ///
    /// The resulting volume is self-consistent: `tau_up` is rebuilt from the
    /// exact extinction that the view, secondary-sun, ambient and shadow paths
    /// consume. The method is deterministic and performs no stochastic draws.
    pub fn apply_deterministic_fractional_subcolumn(
        &mut self,
        subcolumn_index: usize,
    ) -> Result<FractionalSubcolumnStats, String> {
        self.apply_deterministic_fractional_subcolumn_count(
            subcolumn_index,
            DETERMINISTIC_SUBCOLUMN_COUNT,
        )
    }

    /// Materialize one member of a selectable 4/8/16 fixed-stratified ensemble.
    ///
    /// This is the counted form of [`Self::apply_deterministic_fractional_subcolumn`].
    /// The sampled `u` remains shared through every vertical layer. The caller must
    /// feed this one resulting volume to its view, sun, ambient, and shadow paths as
    /// one indivisible member before averaging linear radiance.
    pub fn apply_deterministic_fractional_subcolumn_count(
        &mut self,
        subcolumn_index: usize,
        subcolumn_count: usize,
    ) -> Result<FractionalSubcolumnStats, String> {
        let u = deterministic_subcolumn_u_for_count(subcolumn_index, subcolumn_count).ok_or_else(
            || {
                format!(
                    "deterministic fractional subcolumn index {subcolumn_index} is outside 0..{subcolumn_count}, or count is not one of 4/8/16"
                )
            },
        )?;
        let cells = self.nx.saturating_mul(self.ny).saturating_mul(self.nz);
        let available = self.has_cloud_fraction
            && self.ext_liquid.len() == cells
            && self.ext_ice.len() == cells
            && (self.ext_snow.len() == cells || self.science_ext_snow.len() == cells)
            && self.ext_precip.len() == cells
            && self.tau_up.len() == cells
            && self.cloud_fraction.len() == cells;
        let mut stats = FractionalSubcolumnStats {
            available,
            subcolumn_index,
            subcolumn_u: u,
            ..FractionalSubcolumnStats::default()
        };
        if !available || self.nz == 0 {
            return Ok(stats);
        }

        let dz = self.dz_m.max(0.0);
        for c in 0..cells {
            let liquid = self.ext_liquid[c].max(0.0);
            let ice = self.ext_ice[c].max(0.0);
            let precip = self.ext_precip[c].max(0.0);
            let snow = self.snow_extinction(c).max(0.0).min(precip);
            let other_precip = precip - snow;
            let cloud_ext = liquid + ice + snow;
            stats.grid_mean_cloud_tau += cloud_ext as f64 * dz;

            match self.cloud_fraction[c] {
                255 => {
                    stats.subcolumn_cloud_tau += cloud_ext as f64 * dz;
                }
                0 => {
                    if cloud_ext > 0.0 {
                        stats.repaired_zero_count += 1;
                        stats.subcolumn_cloud_tau += cloud_ext as f64 * dz;
                    }
                }
                code => {
                    if cloud_ext <= 0.0 {
                        continue;
                    }
                    stats.fractional_layer_count += 1;
                    let f = code as f32 / crate::fractional_clouds::FRACTION_BINS as f32;
                    if u < f as f64 {
                        let inv_f = 1.0 / f;
                        self.ext_liquid[c] = liquid * inv_f;
                        self.ext_ice[c] = ice * inv_f;
                        self.ext_precip[c] = other_precip + snow * inv_f;
                        stats.cloudy_fractional_layer_count += 1;
                        stats.subcolumn_cloud_tau += cloud_ext as f64 / f as f64 * dz;
                    } else {
                        self.ext_liquid[c] = 0.0;
                        self.ext_ice[c] = 0.0;
                        self.ext_precip[c] = other_precip;
                    }
                }
            }
        }

        // Rebuild every column, including those containing only full/repaired
        // cloud, so the explicit subcolumn has one authoritative extinction state.
        let n2 = self.nx.saturating_mul(self.ny);
        if self.nz > 0 && n2 > 0 {
            for j in 0..self.ny {
                for i in 0..self.nx {
                    let top = self.cell(i, j, self.nz - 1);
                    self.tau_up[top] = 0.0;
                    for k in (0..self.nz - 1).rev() {
                        let c0 = self.cell(i, j, k);
                        let c1 = self.cell(i, j, k + 1);
                        let beta0 = self.ext_liquid[c0] as f64
                            + self.ext_ice[c0] as f64
                            + self.ext_precip[c0] as f64;
                        let beta1 = self.ext_liquid[c1] as f64
                            + self.ext_ice[c1] as f64
                            + self.ext_precip[c1] as f64;
                        self.tau_up[c0] =
                            (self.tau_up[c1] as f64 + 0.5 * (beta0 + beta1) * dz) as f32;
                    }
                }
            }
        }

        self.ext_snow.clear();
        self.cloud_fraction.clear();
        self.has_cloud_fraction = false;
        Ok(stats)
    }

    /// Apply a bounded, top-down-only observation-operator reconstruction to broad
    /// low/moderate stratiform cloud columns.
    ///
    /// HRRR can carry grid-scale ring/dash energy in BOTH cloud fraction and native
    /// condensate. A nadir one-cell-per-pixel view exposes that numerical-scale energy
    /// directly, whereas a slant geostationary footprint mixes it away. This operator
    /// regularizes COLUMN optical depth before the cloud march, not the finished image:
    ///
    /// - only liquid-dominated columns whose cloud top is below 7 km seed a broad deck;
    /// - high/frozen or optically very thick convective columns and their two-cell
    ///   neighbourhood are excluded;
    /// - a normalized 5x5 binomial footprint estimates the local column OD;
    /// - one positive, bounded scale multiplies every extinction species/level in a
    ///   selected column, preserving its phase ratio and vertical structure;
    /// - the selected-area OD is renormalized back to its input total, and `tau_up` is
    ///   rebuilt so every light/view consumer remains consistent.
    ///
    /// It cannot invent cloud in a truly clear column. This is deliberately opt-in and
    /// is assembled only for finished top-down visible RGB; geostationary and raw-band
    /// paths never call it.
    pub fn regularize_topdown_stratiform_columns(
        &mut self,
        optical_depth_scale: f32,
    ) -> StratiformRegularizationStats {
        const TOP_MAX_M: f64 = 7000.0;
        const LIQUID_SHARE_MIN: f64 = 0.60;
        const EFFECTIVE_TAU_MIN: f64 = 0.04;
        const EFFECTIVE_TAU_CORE: f64 = 12.0;
        const MIN_COLUMN_SCALE: f64 = 0.35;
        const MAX_COLUMN_SCALE: f64 = 2.5;
        const BROAD_SUPPORT_WEIGHT: u32 = 80; // 31% of the 5x5 binomial footprint.
        const KERNEL: [u32; 5] = [1, 4, 6, 4, 1];

        let ncol = self.nx.saturating_mul(self.ny);
        let cells = ncol.saturating_mul(self.nz);
        let mut stats = StratiformRegularizationStats {
            columns_total: ncol,
            min_scale: 1.0,
            max_scale: 1.0,
            ..StratiformRegularizationStats::default()
        };
        if self.nx < 5
            || self.ny < 5
            || self.nz == 0
            || self.ext_liquid.len() != cells
            || self.ext_ice.len() != cells
            || self.ext_precip.len() != cells
            || self.tau_up.len() != cells
        {
            return stats;
        }

        let dz = self.dz_m.max(0.0);
        let od_scale = (optical_depth_scale as f64).clamp(0.0, 4.0);
        let mut tau = vec![0.0f64; ncol];
        let mut low_seed = vec![false; ncol];
        let mut protected_core = vec![false; ncol];
        for j in 0..self.ny {
            for i in 0..self.nx {
                let col = j * self.nx + i;
                let mut liquid_tau = 0.0;
                let mut frozen_tau = 0.0;
                let mut top_m = self.z_min_m;
                for k in 0..self.nz {
                    let c = self.cell(i, j, k);
                    let liquid = self.ext_liquid[c].max(0.0) as f64;
                    let frozen =
                        self.ext_ice[c].max(0.0) as f64 + self.ext_precip[c].max(0.0) as f64;
                    let total = liquid + frozen;
                    liquid_tau += liquid * dz;
                    frozen_tau += frozen * dz;
                    if total * dz > 1.0e-5 {
                        top_m = self.z_min_m + (k as f64 + 0.5) * dz;
                    }
                }
                let total_tau = liquid_tau + frozen_tau;
                tau[col] = total_tau;
                if total_tau <= 0.0 {
                    continue;
                }
                let liquid_share = liquid_tau / total_tau;
                let effective_tau = total_tau * od_scale;
                let low_like = top_m <= TOP_MAX_M && liquid_share >= LIQUID_SHARE_MIN;
                low_seed[col] =
                    low_like && (EFFECTIVE_TAU_MIN..EFFECTIVE_TAU_CORE).contains(&effective_tau);
                protected_core[col] = !low_like || effective_tau >= EFFECTIVE_TAU_CORE;
                stats.low_cloud_seeds += usize::from(low_seed[col]);
            }
        }

        let mut targets = vec![f64::NAN; ncol];
        for j in 2..self.ny - 2 {
            for i in 2..self.nx - 2 {
                let col = j * self.nx + i;
                if tau[col] <= 0.0 || protected_core[col] {
                    continue;
                }
                let mut seed_support = 0u32;
                let mut has_protected_core = false;
                let mut weighted_tau = 0.0;
                for (ky, &wy) in KERNEL.iter().enumerate() {
                    let jj = j + ky - 2;
                    for (kx, &wx) in KERNEL.iter().enumerate() {
                        let ii = i + kx - 2;
                        let n = jj * self.nx + ii;
                        let w = wy * wx;
                        seed_support += w * u32::from(low_seed[n]);
                        has_protected_core |= protected_core[n];
                        weighted_tau += w as f64 * tau[n];
                    }
                }
                if has_protected_core || seed_support < BROAD_SUPPORT_WEIGHT {
                    continue;
                }
                let smoothed = weighted_tau / 256.0;
                let scale = (smoothed / tau[col]).clamp(MIN_COLUMN_SCALE, MAX_COLUMN_SCALE);
                targets[col] = tau[col] * scale;
            }
        }

        let selected: Vec<usize> = targets
            .iter()
            .enumerate()
            .filter_map(|(idx, target)| target.is_finite().then_some(idx))
            .collect();
        if selected.is_empty() {
            return stats;
        }
        let raw_sum: f64 = selected.iter().map(|&idx| tau[idx]).sum();
        // Solve the bounded global gain as a monotone water-fill. A fixed number of
        // repeated correction/clamp passes can stop with a visible OD deficit when a
        // small selected region contains columns at both bounds. Since gain=0 gives
        // 0.35*raw_sum and gain=8 saturates every target at 2.5*raw_sum, bisection has
        // a guaranteed bracket and is independent of the field's dynamic range.
        let bounded_sum = |gain: f64| -> f64 {
            selected
                .iter()
                .map(|&idx| {
                    (targets[idx] * gain)
                        .clamp(tau[idx] * MIN_COLUMN_SCALE, tau[idx] * MAX_COLUMN_SCALE)
                })
                .sum()
        };
        let mut gain_lo = 0.0;
        let mut gain_hi = 1.0;
        while bounded_sum(gain_hi) < raw_sum {
            gain_hi *= 2.0;
        }
        // 32 halvings put the residual below 1e-9 of the feasible interval; the
        // capacity-aware residual pass below then removes the remaining sum error.
        for _ in 0..32 {
            let gain_mid = 0.5 * (gain_lo + gain_hi);
            if bounded_sum(gain_mid) < raw_sum {
                gain_lo = gain_mid;
            } else {
                gain_hi = gain_mid;
            }
        }
        let gain = 0.5 * (gain_lo + gain_hi);
        for &idx in &selected {
            targets[idx] = (targets[idx] * gain)
                .clamp(tau[idx] * MIN_COLUMN_SCALE, tau[idx] * MAX_COLUMN_SCALE);
        }

        // Remove the last few summation ulps without violating either per-column
        // bound. The stored f32 field will still conserve only to f32 roundoff, which
        // is why the diagnostic below is measured from the mutated field itself.
        let mut residual = raw_sum - selected.iter().map(|&idx| targets[idx]).sum::<f64>();
        for &idx in &selected {
            if residual == 0.0 {
                break;
            }
            let old = targets[idx];
            let new = if residual > 0.0 {
                old + residual.min(tau[idx] * MAX_COLUMN_SCALE - old)
            } else {
                old - (-residual).min(old - tau[idx] * MIN_COLUMN_SCALE)
            };
            targets[idx] = new;
            residual -= new - old;
        }

        stats.tau_before = raw_sum;
        let mut actual_tau_after = 0.0;
        for &col in &selected {
            let scale = (targets[col] / tau[col]) as f32;
            if (scale - 1.0).abs() <= 1.0e-5 {
                actual_tau_after += tau[col];
                continue;
            }
            let i = col % self.nx;
            let j = col / self.nx;
            let mut column_tau_after = 0.0;
            for k in 0..self.nz {
                let c = self.cell(i, j, k);
                self.ext_liquid[c] *= scale;
                self.ext_ice[c] *= scale;
                self.ext_precip[c] *= scale;
                column_tau_after += (self.ext_liquid[c].max(0.0) as f64
                    + self.ext_ice[c].max(0.0) as f64
                    + self.ext_precip[c].max(0.0) as f64)
                    * dz;
            }
            actual_tau_after += column_tau_after;
            stats.columns_changed += 1;
            stats.min_scale = stats.min_scale.min(scale);
            stats.max_scale = stats.max_scale.max(scale);

            let top = self.cell(i, j, self.nz - 1);
            self.tau_up[top] = 0.0;
            for k in (0..self.nz - 1).rev() {
                let c0 = self.cell(i, j, k);
                let c1 = self.cell(i, j, k + 1);
                let beta0 = self.ext_liquid[c0] as f64
                    + self.ext_ice[c0] as f64
                    + self.ext_precip[c0] as f64;
                let beta1 = self.ext_liquid[c1] as f64
                    + self.ext_ice[c1] as f64
                    + self.ext_precip[c1] as f64;
                self.tau_up[c0] = (self.tau_up[c1] as f64 + 0.5 * (beta0 + beta1) * dz) as f32;
            }
        }
        stats.tau_after = actual_tau_after;
        stats
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

    /// [`Self::sample_granulated_gated`] WITHOUT a deck-coherence field (coherence
    /// `None` = gate 1 everywhere). The shipping marches pass the per-composite
    /// coherence carried on the [`SunOdMap`]; this wrapper serves the pure
    /// sampler-level tests and any caller without a composite context.
    pub fn sample_granulated(
        &self,
        fi: f64,
        fj: f64,
        fk: f64,
        granulation: Option<Granulation>,
    ) -> CloudSample {
        self.sample_granulated_gated(fi, fj, fk, granulation, None)
    }

    /// [`Self::sample`] with the optional sub-grid GRANULATION erosion applied (see the
    /// granulation section at the top of this module). `None` is byte-identical to the
    /// raw trilinear sample (same operations, same order — the off-flag anchor test pins
    /// it). With `Some`, the three EXTINCTION channels are scaled by the BIMODAL carve
    /// ([`granulation_bimodal`]) of the remap-style erosion multiplier
    /// ([`granulation_multiplier`]) of the sample's relative density
    /// `d = total / (max total over its 8 trilinear-support corners)` under the
    /// deterministic erosion field `e = amplitude/GRAN_AMP_CAP * gate *
    /// interior_protection(d) * coherence * noise`, where `coherence` is the
    /// per-column DECK-COHERENCE gate ([`GranCoherence`]; `None` = 1 everywhere);
    /// `tau_up` is never eroded (an ingest-time column integral — documented
    /// approximation).
    /// Subtract-only: the eroded sample never exceeds the raw one; zero stays zero.
    pub fn sample_granulated_gated(
        &self,
        fi: f64,
        fj: f64,
        fk: f64,
        granulation: Option<Granulation>,
        coherence: Option<&GranCoherence>,
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
        // DECK-COHERENCE gate (round 2, cheap 2-D bilinear): a coherent deck and its
        // margins skip the erosion (and the noise cost) entirely.
        let coh = coherence.map_or(1.0, |c| c.gate_at(fi, fj));
        if coh <= 0.0 {
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
        let noise = granulation_erosion_noise_footprint(fi * pitch, fj * pitch, pitch);
        let e = (g.amplitude * GRAN_EROSION_GAIN * gate * protection * coh * noise)
            .min(GRAN_EROSION_MAX);
        if e <= 0.0 {
            return s;
        }
        // The remap multiplier through the BIMODAL carve (round 2): gap-or-grain.
        let m = granulation_bimodal(granulation_multiplier(d, e));
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
        Self::build_from_extinction(vol.nx, vol.ny, vol.nz, factor, |i, j, k| {
            vol.total_ext_cell(i, j, k) as f32
        })
    }

    /// Build the same conservative occupancy mip directly from a brick's quantized
    /// extinction channels, without materializing a [`DecodedVolume`].
    ///
    /// The 256-entry decode tables reproduce [`LogQuant::decode`] exactly. Per-voxel
    /// channel addition follows [`DecodedVolume::total_ext_cell`] (decoded `f32`
    /// values widened to `f64`, summed left-to-right, then narrowed to `f32`) and the
    /// shared builder applies the identical block maximum and one-block dilation.
    /// Consequently every `maxext` value, not only its occupied/empty classification,
    /// is bit-identical to `OccupancyMip::build(DecodedVolume::from_brick_legacy(..))`.
    pub fn from_quantized_brick(brick: &VolumeBrick, factor: usize) -> Self {
        let cells = brick
            .nx
            .checked_mul(brick.ny)
            .and_then(|n| n.checked_mul(brick.nz))
            .expect("cloud-volume dimensions overflow usize");
        assert_eq!(brick.ext_liquid.len(), cells, "ext_liquid length");
        assert_eq!(brick.ext_ice.len(), cells, "ext_ice length");
        assert_eq!(brick.ext_precip.len(), cells, "ext_precip length");

        let decode_lut = |quant: LogQuant| -> [f32; 256] {
            core::array::from_fn(|code| quant.decode(code as u8))
        };
        let liquid_lut = decode_lut(brick.quant.get("ext_liquid"));
        let ice_lut = decode_lut(brick.quant.get("ext_ice"));
        let precip_lut = decode_lut(brick.quant.get("ext_precip"));

        Self::build_from_extinction(brick.nx, brick.ny, brick.nz, factor, |i, j, k| {
            let cell = (k * brick.ny + j) * brick.nx + i;
            (liquid_lut[brick.ext_liquid[cell] as usize] as f64
                + ice_lut[brick.ext_ice[cell] as usize] as f64
                + precip_lut[brick.ext_precip[cell] as usize] as f64) as f32
        })
    }

    fn build_from_extinction(
        nx: usize,
        ny: usize,
        nz: usize,
        factor: usize,
        mut extinction: impl FnMut(usize, usize, usize) -> f32,
    ) -> Self {
        let factor = factor.max(1);
        let mx = nx.div_ceil(factor);
        let my = ny.div_ceil(factor);
        let mz = nz.div_ceil(factor);
        // Raw per-block max extinction.
        let mut raw = vec![0.0f32; mx * my * mz];
        for k in 0..nz {
            let kb = k / factor;
            for j in 0..ny {
                let jb = j / factor;
                for i in 0..nx {
                    let ib = i / factor;
                    let e = extinction(i, j, k);
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
/// [`cloud_sun_optical_depth`]. The raw total is retained as a missing-data fallback
/// for higher-order support only when a legacy/synthetic volume has no positive
/// whole-column `tau_up`; real ingested volumes use that smooth native column and
/// never let this coarse shadow raster modulate cloud radiance.
///
/// M5 adds `occ_dist` (per texel: the extinction-weighted mean SLANT distance from the
/// ground to the occluding cloud along the sun ray) so [`SunOdMap::penumbral_shadow`]
/// can widen the ground shadow's penumbra with occluder height (design section 6:
/// blur radius = occluder distance x tan 0.2665 deg). The map's `(au, av)` plane IS
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
    /// The per-composite DECK-COHERENCE gate (round-2 granulation; `Some` iff the map
    /// was accumulated with granulation on). Carried HERE because the sun-OD
    /// accumulation is the one per-composite call that receives both the volume and
    /// the granulation value from every render assembly (api + studio), and every
    /// march reads this map through [`CloudScene::sun_od`] — so all marches of one
    /// composite share the SAME coherence field with zero assembly changes.
    pub gran_coherence: Option<GranCoherence>,
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
    // The per-composite DECK-COHERENCE gate (round 2): built once from the volume,
    // applied to THIS accumulation and carried to the marches on the returned map.
    let gran_coherence = granulation
        .filter(|g| g.amplitude > 0.0)
        .map(|_| GranCoherence::build(vol));
    if let Some(c) = &gran_coherence {
        let (open, partial, closed) = c.stats();
        crate::log_line!(
            "simsat clouds: granulation deck-coherence gate: {open} open / {partial} partial / \
             {closed} closed of {} columns",
            open + partial + closed
        );
    }
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
            gran_coherence,
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
                    let ext = vol
                        .sample_granulated_gated(fi, fj, fk, granulation, gran_coherence.as_ref())
                        .total_ext();
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
        gran_coherence,
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
    /// `tan(0.2665 deg)` (the sun disk's angular radius projected onto the sun plane,
    /// which is exactly this map's `(u, v)` plane). We average the Beer transmittance
    /// over a small disk of that radius — a named approximation (pre-blur instead of
    /// disk-sampling the volume): a higher cloud (larger `occ_dist`) yields a wider,
    /// softer penumbra; a ground-hugging cloud stays sharp; a clear column stays 1.
    pub fn penumbral_shadow(&self, p: [f64; 3]) -> f64 {
        self.penumbral_shadow_scaled(p, 1.0)
    }

    /// [`Self::penumbral_shadow`] with a visible-cloud optical-depth multiplier.
    /// The stored map remains raw (and reusable by physical COD diagnostics); the
    /// Beer exponent is scaled only at this visible-render consumer. The occluder
    /// distance is an extinction-weighted mean and therefore does not change.
    pub fn penumbral_shadow_scaled(&self, p: [f64; 3], optical_depth_scale: f32) -> f64 {
        let od_scale = validated_cloud_optical_depth_scale(optical_depth_scale) as f64;
        let (u, v) = self.plane_uv(p);
        let occ_dist = self.sample_uv(&self.occ_dist, u, v);
        let radius = solar_penumbra_radius_m(occ_dist);
        // Below ~one texel of blur there is no penumbra to resolve: sharp Beer shadow.
        let texel = ((self.u_max - self.u_min).abs() / self.width.max(1) as f64)
            .max((self.v_max - self.v_min).abs() / self.height.max(1) as f64);
        if radius <= 0.5 * texel {
            return beer(od_scale * self.sample_uv(&self.od, u, v));
        }
        // A centre tap + two rings of taps over the blur disk, transmittance-averaged
        // (the penumbra is a partial occlusion of the sun DISK, so it softens in
        // transmittance, not optical-depth, space).
        const RING: usize = 8;
        let mut sum = beer(od_scale * self.sample_uv(&self.od, u, v));
        let mut wsum = 1.0f64;
        for (ri, &rr) in [0.5, 1.0].iter().enumerate() {
            let w = if ri == 0 { 1.0 } else { 0.6 };
            for k in 0..RING {
                let ang = (k as f64 + 0.5) / RING as f64 * 2.0 * PI;
                let du = radius * rr * ang.cos();
                let dv = radius * rr * ang.sin();
                sum += w * beer(od_scale * self.sample_uv(&self.od, u + du, v + dv));
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

/// Visible-cloud higher-order transport dispatch.
///
/// `LegacyOctaves` is the exact shipping v0.1.4/v0.1.5 arithmetic. The Stage-2
/// delta-flux closure is an explicitly selected research candidate and is never
/// entered by a default render.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CloudMultiscatterMode {
    #[default]
    LegacyOctaves,
    SingleScatter,
    DeltaFluxV1,
    DeltaFluxV2,
    DeltaFluxV3,
}

impl CloudMultiscatterMode {
    pub const fn slug(self) -> &'static str {
        match self {
            Self::LegacyOctaves => "legacy-octaves",
            Self::SingleScatter => "single-scatter",
            Self::DeltaFluxV1 => "delta-flux-v1",
            Self::DeltaFluxV2 => "delta-flux-v2b",
            Self::DeltaFluxV3 => "delta-flux-v3-memory",
        }
    }
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
    /// FINDING 1); the orthographic sun-OD map remains the ground-shadow source and a
    /// total-column support measure for the thin-limit multiscatter gate. It is never
    /// substituted for the depth-resolved sun transmittance.
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
    /// Higher-order cloud transport dispatch. [`CloudMultiscatterMode::LegacyOctaves`]
    /// preserves the established octave arithmetic exactly; `DeltaFluxV1` selects the
    /// opt-in Stage-2 Monte Carlo depth-source LUT and `DeltaFluxV2` its brightness-
    /// neutral, upward-hemisphere-normalized P1 directional reconstruction;
    /// `DeltaFluxV3` retains the bounded second-order phase memory in thin cloud.
    pub multiscatter_mode: CloudMultiscatterMode,
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
    /// Runtime QA/calibration multiplier for VISIBLE cloud optical depth. Applied at
    /// render consumption to view extinction, cloud self-shadow, ground shadow, and
    /// ambient attenuation. It deliberately does not mutate the decoded volume,
    /// derived COD products, or thermal-IR opacity. The shipped default is
    /// [`DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE`]; `1.0` remains the explicit unscaled
    /// model-extinction value. Validated to
    /// [`CLOUD_OPTICAL_DEPTH_SCALE_MIN`]..=[`CLOUD_OPTICAL_DEPTH_SCALE_MAX`].
    pub cloud_optical_depth_scale: f32,
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
    /// Physical reflectance-factor ceiling mapped to display white by the bounded
    /// highlight shoulder ([`crate::render::RHO_HIGHLIGHT_MAX`]). The
    /// `render_frame` `cloud-highlight-max=` knob can override it for display-only
    /// headroom A/B tests; it does not change extinction, transmittance, IR, or derived COD.
    pub cloud_highlight_max: f64,
    /// Per-frame ABI synthetic-green display arithmetic. False is the native broad-RGB
    /// path and the strict Sensor Fast Gray requirement.
    pub synthetic_green: bool,
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
    /// ([`DecodedVolume::sample_granulated_gated`]) under the DECK-COHERENCE gate
    /// carried on [`CloudScene::sun_od`]; the render assembly must pass the SAME
    /// value to [`accumulate_sun_od_granulated`] so the ground shadows (and the
    /// coherence field it builds) agree — every march of one composite samples the
    /// SAME eroded field.
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
            multiscatter_mode: CloudMultiscatterMode::LegacyOctaves,
            beer_powder: false,
            ground_albedo: GROUND_ALBEDO,
            transmittance_floor: 0.003,
            cloud_optical_depth_scale: DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE,
            // Appearance-pass baked defaults (the studio's `..MarchConfig::new()` inherits
            // these; the render_frame CLI knobs override them). Edge feather off by default
            // (activated only by a zoom-out margin, via `edge_feather_cells_for_margin`).
            ground_day_lift: GROUND_DAY_LIFT,
            cloud_softclip_knee: CLOUD_SOFTCLIP_KNEE,
            cloud_highlight_max: crate::render::RHO_HIGHLIGHT_MAX,
            synthetic_green: false,
            topdown_cloud_norm: crate::topdown::TOPDOWN_CLOUD_NORM,
            edge_feather_cells: 0.0,
            granulation: None,
        }
    }

    /// Validated visible optical-depth multiplier as the march's native `f64`.
    #[inline]
    pub fn validated_cloud_optical_depth_scale(&self) -> f64 {
        validated_cloud_optical_depth_scale(self.cloud_optical_depth_scale) as f64
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
/// transmittance, so the map is no longer consulted here; outside this function it
/// survives for the ground cloud-shadow ([`ground_cloud_shadow`]), where the whole
/// column IS the cloud between the ground and the sun, and as a missing-`tau_up`
/// support fallback for legacy/synthetic volumes.
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
    let coherence = scene.sun_od.gran_coherence.as_ref();
    for _ in 0..n {
        // Sample within the segment (dist .. dist+ds) at the stratified offset,
        // toward the sun. Samples the SAME (optionally granulated, coherence-gated)
        // field as the primary view march, so cloud self-shadowing matches the
        // eroded cloud.
        let pp = madd3(p, scene.sun_ecef, dist + offset * ds);
        let (fi, fj, fk, _) = ecef_to_brick(pp, scene.georef, scene.vol.z_min_m, scene.vol.dz_m);
        tau += scene
            .vol
            .sample_granulated_gated(fi, fj, fk, cfg.granulation, coherence)
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
                .sample_granulated_gated(fi, fj, fk, cfg.granulation, coherence)
                .total_ext()
                * half;
            dist += half;
        }
    }
    // The decoded field and cached sun-OD map stay physical/raw. The visible QA
    // multiplier is applied at consumption so derived COD and thermal IR are untouched.
    tau * cfg.validated_cloud_optical_depth_scale()
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
    let od_scale = scene.cfg.validated_cloud_optical_depth_scale();

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
        // The (optionally granulated, coherence-gated) view sample — the same eroded
        // field the sun march and the sun-OD map read (MarchConfig::granulation +
        // the coherence carried on the sun-OD map).
        let sample = vol.sample_granulated_gated(
            mi,
            mj,
            mk,
            scene.cfg.granulation,
            scene.sun_od.gran_coherence.as_ref(),
        );
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
        // At the neutral OD scale, `edge_feather_cells == 0` ->
        // `sigma_eff == sigma_t` byte-for-byte.
        let feather = edge_feather(mi, mj, vol.nx, vol.ny, scene.cfg.edge_feather_cells);
        let sigma_eff = sigma_t * feather * od_scale;
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
        // The smooth vertical whole-column OD is the best support measure for whether
        // multiple scattering is possible (including at a thick cloud's sunlit top).
        // The 512-texel sun-aligned map is a ground-shadow raster: taking its maximum
        // with a real `tau_up` imprinted that raster's dash/moire texture on HRRR cloud
        // radiance. Consult it ONLY when a legacy/synthetic volume has no positive
        // whole-column channel. It remains support-only, never sample-to-sun
        // transmittance; `tau_sun` and one local voxel are additional fallbacks.
        let col_total = vol.sample(mi, mj, 0.0).tau_up; // raw OD of whole column at (i,j)
        let column_support_tau = if col_total.is_finite() && col_total > 0.0 {
            col_total * od_scale
        } else {
            scene.sun_od.sample(pm) * od_scale
        };
        let multiscatter_support_tau = column_support_tau.max(tau_cloud_sun).max(sigma_eff * pitch);
        let up = scl3(pm, 1.0 / rm);
        let mu_sun = dot3(up, scene.sun_ecef);
        let sun_src = match scene.cfg.multiscatter_mode {
            CloudMultiscatterMode::LegacyOctaves => octave_sun_source_thin_gated(
                cos_vs,
                sample.ext_liquid,
                sample.ext_ice + sample.ext_precip,
                tau_cloud_sun,
                scene.cfg.beer_powder,
                scene.cfg.octaves,
                multiscatter_support_tau,
            ),
            CloudMultiscatterMode::SingleScatter => octave_sun_source_thin_gated(
                cos_vs,
                sample.ext_liquid,
                sample.ext_ice + sample.ext_precip,
                tau_cloud_sun,
                scene.cfg.beer_powder,
                1,
                multiscatter_support_tau,
            ),
            CloudMultiscatterMode::DeltaFluxV1
            | CloudMultiscatterMode::DeltaFluxV2
            | CloudMultiscatterMode::DeltaFluxV3 => {
                let direct = octave_sun_source_thin_gated(
                    cos_vs,
                    sample.ext_liquid,
                    sample.ext_ice + sample.ext_precip,
                    tau_cloud_sun,
                    scene.cfg.beer_powder,
                    1,
                    multiscatter_support_tau,
                );
                let fractional_depth = if col_total.is_finite() && col_total > 0.0 {
                    (sample.tau_up / col_total).clamp(0.0, 1.0)
                } else {
                    0.5
                };
                let higher = match scene.cfg.multiscatter_mode {
                    CloudMultiscatterMode::DeltaFluxV2 => {
                        // `view` points camera -> sample; the outgoing direction toward
                        // the camera is `-view`. `up` is the slab upper-boundary normal.
                        crate::cloud_delta_flux::stage2_higher_order_source_p1(
                            multiscatter_support_tau,
                            fractional_depth,
                            mu_sun,
                            scene.cfg.ground_albedo,
                            sample.ext_liquid,
                            sample.ext_ice + sample.ext_precip,
                            -dot3(view, up),
                        )
                    }
                    CloudMultiscatterMode::DeltaFluxV3 => {
                        crate::cloud_delta_flux::stage2_higher_order_source_order_memory(
                            multiscatter_support_tau,
                            fractional_depth,
                            mu_sun,
                            scene.cfg.ground_albedo,
                            sample.ext_liquid,
                            sample.ext_ice + sample.ext_precip,
                            cos_vs,
                        )
                    }
                    CloudMultiscatterMode::DeltaFluxV1 => {
                        crate::cloud_delta_flux::stage2_higher_order_source(
                            multiscatter_support_tau,
                            fractional_depth,
                            mu_sun,
                            scene.cfg.ground_albedo,
                            sample.ext_liquid,
                            sample.ext_ice + sample.ext_precip,
                        )
                    }
                    _ => unreachable!("delta-flux match arm entered for non-delta mode"),
                };
                direct + higher.higher_isotropic
            }
        };

        // Atmospheric sun transmittance to the sample (reddening at low sun) with the
        // FINITE-DISK EARTH-SHADOW FADE (WS1 march-physics pass): the fraction of the
        // solar disk above the sample's local geometric horizon scales the direct sun
        // smoothly through the terminator, replacing the binary ray_hits_ground gate
        // (which drew a hard lit/unlit line across dusk anvils). The transmittance-LUT
        // sample clamps mu to the horizon so the fading disk is attenuated by the
        // (defined) grazing path rather than an undefined below-horizon sample.
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
        let tau_down = (col_total - sample.tau_up).max(0.0);
        let amb_factor = ambient_cloud_factor(
            sample.tau_up * od_scale,
            tau_down * od_scale,
            scene.cfg.ground_albedo,
        );

        // Use the edge-feathered extinction `sigma_eff` for the step opacity + the
        // in-scatter source (the sun/ambient inputs above use the edge-unfeathered field;
        // the runtime visible OD scale still applies to every optical-depth consumer).
        // At feather 1.0, scale 1.0, `sigma_eff == sigma_t`, so this is byte-identical
        // to the pre-feather march.
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
/// tan 0.2665 deg) instead of the fix2 sharp `e^-od`.
pub fn ground_cloud_shadow(scene: &CloudScene, cam: [f64; 3], view: [f64; 3]) -> f64 {
    match ray_sphere(cam, view, scene.vol.r_bottom()) {
        Some((t0, _)) if t0 > 0.0 => {
            let pg = madd3(cam, view, t0);
            scene
                .sun_od
                .penumbral_shadow_scaled(pg, scene.cfg.cloud_optical_depth_scale)
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
        Some(l_final) => radiance_to_rgba_softclip_with_synthetic_green(
            crate::render::apply_low_sun_illuminant(
                l_final,
                px.on_earth,
                px.sun_elev_deg as f64,
                surf.luts,
            ),
            surf.output_transform,
            surf.exposure,
            scene.cfg.cloud_softclip_knee,
            scene.cfg.cloud_highlight_max,
            scene.cfg.synthetic_green,
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
    // Apply the same product-facing aerial correction to atmosphere in FRONT of cloud
    // that surface_toa_radiance applies over the ground. Leaving this unscaled made the
    // clouds-on geostationary product reintroduce a bright haze veil. Raw-physics mode
    // deliberately retains the full froxel airlight.
    let front_veil = if surf.atmosphere_correction {
        crate::render::aerial_veil_scale(px.sun_elev_deg as f64)
    } else {
        1.0
    };
    let mut l_final = [0.0f64; 3];
    for c in 0..3 {
        l_final[c] = l_toa[c] * m.transmittance
            + t_ac * m.inscatter[c]
            + front_veil * i_ac[c] * (1.0 - m.transmittance);
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
    raster: &SurfaceRaster,
    stride: usize,
    cloudy_threshold: f64,
) -> CloudFrameStats {
    let scan = &raster.scan;
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
            if !raster.grid_i[idx].is_finite()
                || !raster.lat[idx].is_finite()
                || !raster.lon[idx].is_finite()
            {
                continue; // off-earth or outside the WRF domain
            }
            let (sx, sy) = raster.model_scan_angle(px, py);
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
    raster: &SurfaceRaster,
    assemble: impl Fn(usize, usize) -> SurfacePixel + Sync,
) -> Vec<u8> {
    let (nx, ny) = (raster.nx, raster.ny);
    let scan_rect = raster.model_scan_rect();
    let rows: Vec<Vec<u8>> = (0..ny)
        .into_par_iter()
        .map(|py| {
            let mut row = Vec::with_capacity(nx * 4);
            for px in 0..nx {
                let (sx, sy) = raster.model_scan_angle(px, py);
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
/// Python binding's `render_rgb_reflectance` returns (`render_visible_bands` is the
/// deprecated compatibility alias). Identical assembly to
/// [`render_cloud_frame_rgba`] (same [`composite_cloud_radiance`], same `assemble`, same
/// scan rays), but each pixel's composited radiance is converted to the reflectance factor
/// `pi*L/E_sun` ([`crate::render::reflectance_from_radiance`]) instead of the exposure +
/// ABI-stretch display transform. Rows in parallel (rayon).
pub fn render_cloud_frame_reflectance(
    scene: &CloudScene,
    surf: &FrameContext,
    froxel: &AerialFroxel,
    raster: &SurfaceRaster,
    assemble: impl Fn(usize, usize) -> SurfacePixel + Sync,
) -> Vec<f32> {
    let (nx, ny) = (raster.nx, raster.ny);
    let scan_rect = raster.model_scan_rect();
    let rows: Vec<Vec<f32>> = (0..ny)
        .into_par_iter()
        .map(|py| {
            let mut row = vec![0.0f32; nx * 3];
            for px in 0..nx {
                let (sx, sy) = raster.model_scan_angle(px, py);
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

/// Render the untonemapped, unclamped linear-radiance numerator of a cloud frame.
///
/// This reference seam exists for deterministic ICA/McICA-style subcolumn
/// integration: callers render each explicit cloud state through the same view,
/// sun, ambient and shadow paths, average these radiances, then apply the display
/// transform exactly once. `alpha` is 255 for earth/limb and 0 for space.
pub fn render_cloud_frame_linear_radiance(
    scene: &CloudScene,
    surf: &FrameContext,
    froxel: &AerialFroxel,
    raster: &SurfaceRaster,
    assemble: impl Fn(usize, usize) -> SurfacePixel + Sync,
) -> (Vec<f64>, Vec<u8>) {
    let (nx, ny) = (raster.nx, raster.ny);
    let scan_rect = raster.model_scan_rect();
    let rows: Vec<(Vec<f64>, Vec<u8>)> = (0..ny)
        .into_par_iter()
        .map(|py| {
            let mut radiance = vec![0.0f64; nx * 3];
            let mut alpha = vec![0u8; nx];
            for px in 0..nx {
                let (sx, sy) = raster.model_scan_angle(px, py);
                let mut pixel = assemble(px, py);
                pixel.view_dir = surf.cam.view_dir(sx, sy);
                if let Some(l) =
                    composite_cloud_radiance(scene, surf, &pixel, froxel, scan_rect, sx, sy)
                {
                    radiance[px * 3..px * 3 + 3].copy_from_slice(&l);
                    alpha[px] = 255;
                }
            }
            (radiance, alpha)
        })
        .collect();
    let mut radiance = Vec::with_capacity(nx * ny * 3);
    let mut alpha = Vec::with_capacity(nx * ny);
    for (r, a) in rows {
        radiance.extend(r);
        alpha.extend(a);
    }
    (radiance, alpha)
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
    use crate::bricks::ChannelQuant;
    use crate::frame::{GridGeoref, MapProjection};
    use std::collections::BTreeMap;

    fn occupancy_test_brick(
        dims: (usize, usize, usize),
        ext_liquid: Vec<u8>,
        ext_ice: Vec<u8>,
        ext_precip: Vec<u8>,
        quantizers: (LogQuant, LogQuant, LogQuant),
    ) -> VolumeBrick {
        let (nx, ny, nz) = dims;
        let cells = nx * ny * nz;
        let plane = nx * ny;
        assert_eq!(ext_liquid.len(), cells);
        assert_eq!(ext_ice.len(), cells);
        assert_eq!(ext_precip.len(), cells);
        let zero = LogQuant {
            vmin: 0.0,
            vmax: 0.0,
        };
        let mut quant = BTreeMap::new();
        for (name, value) in [
            ("ext_liquid", quantizers.0),
            ("ext_ice", quantizers.1),
            ("ext_snow", zero),
            ("ext_precip", quantizers.2),
            ("tau_up", zero),
            ("qvapor", zero),
        ] {
            quant.insert(name.to_string(), value);
        }
        VolumeBrick {
            storage_profile: crate::bricks::StorageProfile::CompactU8,
            science_cloud_f16: None,
            nx,
            ny,
            nz,
            z_min_m: 0.0,
            dz_m: 250.0,
            time_iso: None,
            quant: ChannelQuant(quant),
            ext_liquid,
            ext_ice,
            ext_snow: vec![0; cells],
            ext_precip,
            tau_up: vec![0; cells],
            qvapor: vec![0; cells],
            cloud_fraction: vec![255; cells],
            has_cloud_fraction: false,
            temperature_f16: vec![0; cells],
            hgt: vec![0.0; plane],
            landmask: vec![1.0; plane],
            tsk: vec![300.0; plane],
            u10: vec![0.0; plane],
            v10: vec![0.0; plane],
            snowh: None,
            ivgtyp: None,
        }
    }

    fn assert_quantized_mip_matches_decoded(brick: &VolumeBrick, factor: usize, label: &str) {
        let quantized = OccupancyMip::from_quantized_brick(brick, factor);
        let decoded = DecodedVolume::from_brick_legacy(brick, 1000.0);
        let reference = OccupancyMip::build(&decoded, factor);
        assert_eq!(quantized.mx, reference.mx, "{label}: mx");
        assert_eq!(quantized.my, reference.my, "{label}: my");
        assert_eq!(quantized.mz, reference.mz, "{label}: mz");
        assert_eq!(quantized.factor, reference.factor, "{label}: factor");
        let got: Vec<u32> = quantized.maxext.iter().map(|v| v.to_bits()).collect();
        let expected: Vec<u32> = reference.maxext.iter().map(|v| v.to_bits()).collect();
        assert_eq!(got, expected, "{label}: maxext bits");
    }

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
            ext_snow: vec![0; n],
            ext_snow_quant: LogQuant {
                vmin: 0.0,
                vmax: 0.0,
            },
            science_ext_snow: Vec::new(),
            ext_precip,
            tau_up,
            cloud_fraction: vec![255; n],
            has_cloud_fraction: false,
        }
    }

    /// Rebuild the ingestion-style vertical optical-depth channel for a synthetic
    /// test volume.  A positive value at `k=0` is the marker that whole-column
    /// support is genuinely available to the visible cloud march.
    fn rebuild_test_tau_up(vol: &mut DecodedVolume) {
        if vol.nz == 0 {
            return;
        }
        for j in 0..vol.ny {
            for i in 0..vol.nx {
                let top = vol.cell(i, j, vol.nz - 1);
                vol.tau_up[top] = 0.0;
                for k in (0..vol.nz - 1).rev() {
                    let c0 = vol.cell(i, j, k);
                    let c1 = vol.cell(i, j, k + 1);
                    let beta0 = vol.ext_liquid[c0] as f64
                        + vol.ext_ice[c0] as f64
                        + vol.ext_precip[c0] as f64;
                    let beta1 = vol.ext_liquid[c1] as f64
                        + vol.ext_ice[c1] as f64
                        + vol.ext_precip[c1] as f64;
                    vol.tau_up[c0] =
                        (vol.tau_up[c1] as f64 + 0.5 * (beta0 + beta1) * vol.dz_m) as f32;
                }
            }
        }
    }

    fn test_column_tau(vol: &DecodedVolume, i: usize, j: usize) -> f64 {
        (0..vol.nz)
            .map(|k| {
                let c = vol.cell(i, j, k);
                (vol.ext_liquid[c] as f64 + vol.ext_ice[c] as f64 + vol.ext_precip[c] as f64)
                    * vol.dz_m
            })
            .sum()
    }

    #[test]
    fn topdown_stratiform_regularization_conserves_od_and_column_structure() {
        let (nx, ny, nz) = (13, 13, 6);
        let mut vol = build_volume(nx, ny, nz, 500.0, 3000.0, |i, j, k| {
            // A broad low/liquid deck with deterministic grid-scale stipple. Every
            // column is a seed; the species and vertical ratios are non-trivial.
            let pattern = match (i + 2 * j) % 4 {
                0 => 0.35,
                1 => 0.8,
                2 => 1.5,
                _ => 2.5,
            };
            let vertical = 1.0 + 0.05 * k as f64;
            (
                6.0e-4 * pattern * vertical,
                1.0e-4 * pattern * vertical,
                5.0e-5 * pattern * vertical,
            )
        });
        vol.tau_up.fill(99.0); // proves every modified column is rebuilt.
        let before = vol.clone();
        let interior = |v: &DecodedVolume| {
            (2..ny - 2)
                .flat_map(|j| (2..nx - 2).map(move |i| test_column_tau(v, i, j)))
                .collect::<Vec<_>>()
        };
        let before_tau = interior(&before);
        let variance = |values: &[f64]| {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
        };

        let stats = vol.regularize_topdown_stratiform_columns(0.15);
        let after_tau = interior(&vol);
        assert_eq!(stats.columns_total, nx * ny);
        assert_eq!(stats.low_cloud_seeds, nx * ny);
        assert!(stats.columns_changed > 70, "stats={stats:?}");
        assert!(
            variance(&after_tau) < 0.25 * variance(&before_tau),
            "stipple variance was not materially reduced"
        );

        let before_sum = before_tau.iter().sum::<f64>();
        let after_sum = after_tau.iter().sum::<f64>();
        assert!((stats.tau_before - before_sum).abs() < 1.0e-9);
        assert!(
            (stats.tau_after - stats.tau_before).abs() / stats.tau_before < 2.0e-7,
            "stats={stats:?}"
        );
        assert!(
            (after_sum - before_sum).abs() / before_sum < 2.0e-7,
            "actual selected OD changed: {before_sum} -> {after_sum}"
        );

        let mut verified_changed_column = false;
        for j in 2..ny - 2 {
            for i in 2..nx - 2 {
                let scale = test_column_tau(&vol, i, j) / test_column_tau(&before, i, j);
                if (scale - 1.0).abs() <= 1.0e-5 {
                    continue;
                }
                verified_changed_column = true;
                for k in 0..nz {
                    let c = vol.cell(i, j, k);
                    for (after, raw) in [
                        (vol.ext_liquid[c], before.ext_liquid[c]),
                        (vol.ext_ice[c], before.ext_ice[c]),
                        (vol.ext_precip[c], before.ext_precip[c]),
                    ] {
                        assert!(((after / raw) as f64 - scale).abs() < 3.0e-7);
                    }
                }
                let top = vol.cell(i, j, nz - 1);
                assert_eq!(vol.tau_up[top], 0.0);
                for k in (0..nz - 1).rev() {
                    let c0 = vol.cell(i, j, k);
                    let c1 = vol.cell(i, j, k + 1);
                    let beta0 = vol.ext_liquid[c0] as f64
                        + vol.ext_ice[c0] as f64
                        + vol.ext_precip[c0] as f64;
                    let beta1 = vol.ext_liquid[c1] as f64
                        + vol.ext_ice[c1] as f64
                        + vol.ext_precip[c1] as f64;
                    let expected = vol.tau_up[c1] as f64 + 0.5 * (beta0 + beta1) * vol.dz_m;
                    assert!((vol.tau_up[c0] as f64 - expected).abs() < 2.0e-6);
                }
            }
        }
        assert!(verified_changed_column);
    }

    #[test]
    fn topdown_stratiform_regularization_exactly_excludes_convective_columns() {
        let (nx, ny, nz) = (11, 11, 32);
        let mut vol = build_volume(nx, ny, nz, 250.0, 3000.0, |i, _, k| {
            if i < nx / 2 {
                // High frozen cloud: cloud top is above the 7-km stratiform gate.
                if k >= 28 {
                    (0.0, 2.0e-3, 5.0e-4)
                } else {
                    (0.0, 0.0, 0.0)
                }
            } else {
                // Optically thick low liquid core: effective tau is above the core gate.
                if k < 4 {
                    (0.1, 0.0, 0.0)
                } else {
                    (0.0, 0.0, 0.0)
                }
            }
        });
        vol.tau_up
            .iter_mut()
            .enumerate()
            .for_each(|(idx, v)| *v = idx as f32 * 1.0e-5);
        let before = vol.clone();
        let stats = vol.regularize_topdown_stratiform_columns(0.15);
        assert_eq!(stats.columns_changed, 0, "stats={stats:?}");
        assert_eq!(stats.tau_before, 0.0);
        assert_eq!(stats.tau_after, 0.0);
        assert_eq!(vol.ext_liquid, before.ext_liquid);
        assert_eq!(vol.ext_ice, before.ext_ice);
        assert_eq!(vol.ext_precip, before.ext_precip);
        assert_eq!(vol.tau_up, before.tau_up);
    }

    #[test]
    fn fractional_clouds_unavailable_and_full_cover_are_exact_noops() {
        let mut unavailable = build_volume(2, 1, 3, 250.0, 500.0, |i, _, k| {
            (1.0e-3 * (i + 1) as f64, 2.0e-4 * k as f64, 3.0e-4)
        });
        unavailable.tau_up = vec![1.25, 0.75, 0.0, 1.5, 0.5, 0.0];
        let before = unavailable.clone();
        let stats = unavailable.apply_fractional_clouds();
        assert!(!stats.available);
        assert_eq!(unavailable.ext_liquid, before.ext_liquid);
        assert_eq!(unavailable.ext_ice, before.ext_ice);
        assert_eq!(unavailable.ext_snow, before.ext_snow);
        assert_eq!(unavailable.ext_precip, before.ext_precip);
        assert_eq!(unavailable.tau_up, before.tau_up);

        let mut full = before;
        full.has_cloud_fraction = true;
        full.cloud_fraction.fill(255);
        let full_before = full.clone();
        let stats = full.apply_fractional_clouds();
        assert!(stats.available);
        assert_eq!(stats.columns_modified, 0);
        assert_eq!(stats.fractional_layer_count, 0);
        assert_eq!(full.ext_liquid, full_before.ext_liquid);
        assert_eq!(full.ext_ice, full_before.ext_ice);
        assert_eq!(full.ext_precip, full_before.ext_precip);
        assert_eq!(full.tau_up, full_before.tau_up);
        assert!(full.ext_snow.is_empty());
        assert!(full.cloud_fraction.is_empty());
    }

    #[test]
    fn fractional_clouds_thin_partial_layers_and_recompute_tau_up() {
        let mut vol = build_volume(1, 1, 3, 100.0, 500.0, |_, _, _| (0.01, 0.0, 0.0));
        vol.has_cloud_fraction = true;
        vol.cloud_fraction.fill(51); // exactly f = 0.2
        vol.tau_up.fill(99.0); // proves the cumulative channel is rebuilt
        let raw = vol.ext_liquid.clone();
        let stats = vol.apply_fractional_clouds();
        assert!(stats.available);
        assert_eq!(stats.columns_modified, 1);
        assert_eq!(stats.fractional_layer_count, 3);
        assert!(stats.effective_fractional_tau < stats.raw_fractional_tau);
        let scale = stats.effective_fractional_tau / stats.raw_fractional_tau;
        for (&after, &before) in vol.ext_liquid.iter().zip(&raw) {
            assert!(((after as f64 / before as f64) - scale).abs() < 2.0e-7);
        }
        assert_eq!(vol.tau_up[2], 0.0);
        let beta = vol.ext_liquid[0] as f64;
        assert!((vol.tau_up[1] as f64 - beta * 100.0).abs() < 2.0e-7);
        assert!((vol.tau_up[0] as f64 - beta * 200.0).abs() < 4.0e-7);
    }

    #[test]
    fn fractional_clouds_scale_only_the_snow_share_of_total_precip() {
        let mut vol = build_volume(1, 1, 2, 100.0, 500.0, |_, _, _| (0.0, 0.0, 0.03));
        let (snow_quant, snow_codes) = crate::bricks::encode_log_channel(&[0.01, 0.01]);
        vol.ext_snow_quant = snow_quant;
        vol.ext_snow = snow_codes;
        vol.has_cloud_fraction = true;
        vol.cloud_fraction.fill(51);
        let stats = vol.apply_fractional_clouds();
        let scale = (stats.effective_fractional_tau / stats.raw_fractional_tau) as f32;
        for c in 0..2 {
            assert!((vol.ext_precip[c] - (0.02 + 0.01 * scale)).abs() < 1.0e-8);
            assert_eq!(vol.ext_liquid[c], 0.0);
            assert_eq!(vol.ext_ice[c], 0.0);
            assert!(vol.ext_precip[c] <= 0.03);
        }
        assert!(vol.ext_snow.is_empty());
    }

    #[test]
    fn fractional_clouds_repair_zero_fraction_condensate_to_full_cover() {
        let mut vol = build_volume(1, 1, 2, 100.0, 500.0, |_, _, _| (0.01, 0.0, 0.0));
        vol.has_cloud_fraction = true;
        vol.cloud_fraction.fill(0);
        vol.tau_up = vec![7.0, 3.0];
        let before = vol.clone();
        let stats = vol.apply_fractional_clouds();
        assert_eq!(stats.repaired_zero_count, 2);
        assert_eq!(stats.columns_modified, 0);
        assert_eq!(vol.ext_liquid, before.ext_liquid);
        assert_eq!(vol.tau_up, before.tau_up);
    }

    #[test]
    fn deterministic_four_materializes_shared_u_and_preserves_precip_semantics() {
        let mut source = build_volume(1, 1, 1, 100.0, 500.0, |_, _, _| (0.02, 0.01, 0.04));
        let (snow_quant, snow_codes) = crate::bricks::encode_log_channel(&[0.01]);
        source.ext_snow_quant = snow_quant;
        source.ext_snow = snow_codes;
        source.has_cloud_fraction = true;
        source.cloud_fraction.fill(128); // f=128/255, sampled by u=.125 and .375
        source.tau_up.fill(99.0);

        let f = 128.0f64 / 255.0;
        let mut cloudy = 0usize;
        let mut mean_liquid = 0.0f64;
        let mut mean_ice = 0.0f64;
        let mut mean_snow = 0.0f64;
        for n in 0..crate::fractional_clouds::DETERMINISTIC_SUBCOLUMN_COUNT {
            let mut member = source.clone();
            let stats = member.apply_deterministic_fractional_subcolumn(n).unwrap();
            assert!(stats.available);
            assert_eq!(stats.fractional_layer_count, 1);
            assert_eq!(member.tau_up[0], 0.0);
            let other_precip = 0.03f64;
            let snow = (member.ext_precip[0] as f64 - other_precip).max(0.0);
            if stats.cloudy_fractional_layer_count == 1 {
                cloudy += 1;
                assert!((member.ext_liquid[0] as f64 - 0.02 / f).abs() < 2.0e-8);
                assert!((member.ext_ice[0] as f64 - 0.01 / f).abs() < 2.0e-8);
                assert!((snow - 0.01 / f).abs() < 2.0e-8);
            } else {
                assert_eq!(member.ext_liquid[0], 0.0);
                assert_eq!(member.ext_ice[0], 0.0);
                assert!(snow < 2.0e-8);
            }
            // Rain/graupel is the non-snow share and is full-cell in every ICA member.
            assert!((member.ext_precip[0] as f64 - snow - other_precip).abs() < 2.0e-8);
            mean_liquid += member.ext_liquid[0] as f64 / 4.0;
            mean_ice += member.ext_ice[0] as f64 / 4.0;
            mean_snow += snow / 4.0;
        }
        assert_eq!(cloudy, 2);
        let represented_coverage = cloudy as f64 / 4.0;
        assert_eq!(represented_coverage, 0.5);
        // Four-point midpoint quadrature follows the prescribed beta/f in cloudy
        // members. Its finite-sample mass is therefore beta*(N/4)/f (close, but not
        // silently renormalized to exact mass when encoded f is not a quarter).
        assert!((mean_liquid - 0.02 * represented_coverage / f).abs() < 2.0e-8);
        assert!((mean_ice - 0.01 * represented_coverage / f).abs() < 2.0e-8);
        assert!((mean_snow - 0.01 * represented_coverage / f).abs() < 2.0e-8);
    }

    #[test]
    fn deterministic_four_keeps_full_and_repairs_zero_fraction_layers() {
        let mut source = build_volume(1, 1, 2, 100.0, 500.0, |_, _, k| {
            (0.01 * (k + 1) as f64, 0.0, 0.0)
        });
        source.has_cloud_fraction = true;
        source.cloud_fraction = vec![0, 255];
        let before = source.ext_liquid.clone();
        for n in 0..4 {
            let mut member = source.clone();
            let stats = member.apply_deterministic_fractional_subcolumn(n).unwrap();
            assert_eq!(stats.repaired_zero_count, 1);
            assert_eq!(stats.fractional_layer_count, 0);
            assert_eq!(member.ext_liquid, before);
            assert_eq!(member.tau_up[1], 0.0);
            let expected = 0.5 * (before[0] + before[1]) * 100.0;
            assert!((member.tau_up[0] - expected).abs() < 1.0e-7);
        }
    }

    #[test]
    fn selectable_fixed_stratified_counts_are_deterministic_and_converge_in_coverage() {
        let mut source = build_volume(1, 1, 1, 100.0, 500.0, |_, _, _| (0.02, 0.0, 0.0));
        source.has_cloud_fraction = true;
        source.cloud_fraction.fill(77); // f ~= 0.302; deliberately not aligned to 4/8/16.
        let encoded_fraction = 77.0 / 255.0;

        let mut previous_error = f64::INFINITY;
        for count in crate::fractional_clouds::DETERMINISTIC_SUBCOLUMN_COUNTS {
            let mut cloudy = 0usize;
            let mut u_bits = Vec::with_capacity(count);
            for member_index in 0..count {
                let mut member = source.clone();
                let stats = member
                    .apply_deterministic_fractional_subcolumn_count(member_index, count)
                    .unwrap();
                u_bits.push(stats.subcolumn_u.to_bits());
                cloudy += stats.cloudy_fractional_layer_count;

                let mut repeat = source.clone();
                let repeat_stats = repeat
                    .apply_deterministic_fractional_subcolumn_count(member_index, count)
                    .unwrap();
                assert_eq!(
                    stats.subcolumn_u.to_bits(),
                    repeat_stats.subcolumn_u.to_bits()
                );
                assert_eq!(member.ext_liquid, repeat.ext_liquid);
                assert_eq!(member.ext_ice, repeat.ext_ice);
                assert_eq!(member.ext_precip, repeat.ext_precip);
                assert_eq!(member.tau_up, repeat.tau_up);
            }
            assert_eq!(u_bits.len(), count);
            let represented = cloudy as f64 / count as f64;
            let error = (represented - encoded_fraction).abs();
            assert!(error <= 0.5 / count as f64 + f64::EPSILON);
            assert!(error <= previous_error + f64::EPSILON);
            previous_error = error;
        }
    }

    #[test]
    fn topdown_stratiform_regularization_conserves_od_and_preserves_columns() {
        // A deliberately stiff 7x7 deck: the centre column is much thicker than its
        // neighbours, so both scale bounds participate in the selected 3x3 interior.
        // Twelve correction/clamp passes left this case short by about 2.8e-5.
        let mut vol = build_volume(7, 7, 2, 1.0, 3000.0, |i, j, _| {
            let tau = if i == 3 && j == 3 { 10.0 } else { 0.267 };
            (0.4 * tau, 0.1 * tau, 0.0)
        });
        let before = vol.clone();
        let total_od = |v: &DecodedVolume| -> f64 {
            v.ext_liquid
                .iter()
                .zip(&v.ext_ice)
                .zip(&v.ext_precip)
                .map(|((&l, &i), &p)| (l as f64 + i as f64 + p as f64) * v.dz_m)
                .sum()
        };
        let od_before = total_od(&vol);
        let stats = vol.regularize_topdown_stratiform_columns(0.15);
        let od_after = total_od(&vol);

        assert!(stats.columns_changed > 0);
        assert!(stats.min_scale >= 0.35 - 2.0e-7);
        assert!(stats.max_scale <= 2.5 + 2.0e-7);
        assert!((od_after - od_before).abs() / od_before < 2.0e-7);
        assert!((stats.tau_after - stats.tau_before).abs() / stats.tau_before < 2.0e-7);

        for j in 0..vol.ny {
            for i in 0..vol.nx {
                let c0 = vol.cell(i, j, 0);
                let scale = vol.ext_liquid[c0] / before.ext_liquid[c0];
                assert!(scale.is_finite() && scale > 0.0);
                assert!((0.35 - 2.0e-7..=2.5 + 2.0e-7).contains(&scale));
                // One scale applies to every level/species in the column.
                for k in 0..vol.nz {
                    let c = vol.cell(i, j, k);
                    assert!((vol.ext_liquid[c] - before.ext_liquid[c] * scale).abs() < 2.0e-7);
                    assert!((vol.ext_ice[c] - before.ext_ice[c] * scale).abs() < 2.0e-7);
                    assert_eq!(vol.ext_precip[c], 0.0);
                    assert!(vol.tau_up[c].is_finite());
                }
                // The kernel has no support outside the two-cell interior; those edge
                // columns must remain an exact no-op rather than acquiring a seam scale.
                if i < 2 || j < 2 || i + 2 >= vol.nx || j + 2 >= vol.ny {
                    assert_eq!(scale.to_bits(), 1.0f32.to_bits());
                }
            }
        }
    }

    #[test]
    fn topdown_stratiform_regularization_rejects_malformed_tau_up() {
        let mut vol = build_volume(7, 7, 2, 1.0, 3000.0, |i, j, _| {
            let tau = if i == 3 && j == 3 { 5.0 } else { 0.5 };
            (0.5 * tau, 0.0, 0.0)
        });
        vol.tau_up.clear();
        let before = vol.clone();
        let stats = vol.regularize_topdown_stratiform_columns(0.15);
        assert_eq!(stats.columns_changed, 0);
        assert_eq!(vol.ext_liquid, before.ext_liquid);
        assert_eq!(vol.ext_ice, before.ext_ice);
        assert_eq!(vol.ext_precip, before.ext_precip);
        assert!(vol.tau_up.is_empty());
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
    fn visible_cloud_optical_depth_scale_defaults_shipped_and_is_bounded() {
        let cfg = MarchConfig::new(StepQuality::Offline, 250.0);
        assert_eq!(
            cfg.cloud_optical_depth_scale,
            DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE
        );
        assert_eq!(
            cfg.validated_cloud_optical_depth_scale(),
            DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE as f64
        );
        assert_eq!(validated_cloud_optical_depth_scale(-3.0), 0.0);
        assert_eq!(validated_cloud_optical_depth_scale(99.0), 4.0);
        assert_eq!(
            validated_cloud_optical_depth_scale(f32::NAN),
            DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE
        );
        assert_eq!(
            validated_cloud_optical_depth_scale(f32::INFINITY),
            DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE
        );
        assert_eq!(validated_cloud_optical_depth_scale(1.0), 1.0);
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
            cfg: MarchConfig {
                // This analytic Beer-Lambert probe compares against raw model
                // extinction, not the shipped visible-display calibration.
                cloud_optical_depth_scale: 1.0,
                ..MarchConfig::new(StepQuality::Offline, vol.voxel_pitch_m())
            },
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

            // The runtime visible OD scale is applied to the complete march, not just
            // a display alpha: half scale follows e^(-0.5*tau), while the explicit
            // zero endpoint is optically clear and emits no cloud light.
            let march_at_scale = |scale: f32| {
                let scaled_scene = CloudScene {
                    vol: scene.vol,
                    mip: scene.mip,
                    sun_od: scene.sun_od,
                    georef: scene.georef,
                    luts: scene.luts,
                    sky_sh: scene.sky_sh,
                    sun_ecef: scene.sun_ecef,
                    cfg: MarchConfig {
                        cloud_optical_depth_scale: scale,
                        ..scene.cfg
                    },
                };
                march_cloud(&scaled_scene, cam.camera, view)
            };
            let half = march_at_scale(0.5);
            let expected_half = (-0.5 * od_ref).exp();
            assert!(
                (half.transmittance - expected_half).abs() < 0.002,
                "half-scale transmittance {} vs {expected_half}",
                half.transmittance
            );
            assert!(
                half.transmittance > m.transmittance,
                "half scale must reveal more ground: {} <= {}",
                half.transmittance,
                m.transmittance
            );
            let off = march_at_scale(0.0);
            assert_eq!(off.transmittance, 1.0);
            assert_eq!(off.inscatter, [0.0; 3]);
            assert_eq!(off.sun_inscatter, [0.0; 3]);
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
    fn quantized_occupancy_matches_every_code_and_quantizer_edge_bit_exactly() {
        let codes: Vec<u8> = (0u16..=255).map(|code| code as u8).collect();
        let zeros = vec![0; codes.len()];
        let quantizers = [
            (
                "normal",
                LogQuant {
                    vmin: 1.0e-12,
                    vmax: 1.0e-1,
                },
            ),
            (
                "zero",
                LogQuant {
                    vmin: 0.0,
                    vmax: 0.0,
                },
            ),
            (
                "negative-max",
                LogQuant {
                    vmin: -1.0,
                    vmax: -0.1,
                },
            ),
            (
                "degenerate-positive",
                LogQuant {
                    vmin: 0.25,
                    vmax: 0.25,
                },
            ),
            (
                "reversed-positive",
                LogQuant {
                    vmin: 1.0,
                    vmax: 0.25,
                },
            ),
            (
                "zero-min",
                LogQuant {
                    vmin: 0.0,
                    vmax: 1.0,
                },
            ),
            (
                "negative-min",
                LogQuant {
                    vmin: -1.0,
                    vmax: 1.0,
                },
            ),
            (
                "nan-min",
                LogQuant {
                    vmin: f64::NAN,
                    vmax: 1.0,
                },
            ),
            (
                "nan-max",
                LogQuant {
                    vmin: 1.0e-6,
                    vmax: f64::NAN,
                },
            ),
            (
                "infinite-max",
                LogQuant {
                    vmin: 1.0e-6,
                    vmax: f64::INFINITY,
                },
            ),
        ];
        let zero = LogQuant {
            vmin: 0.0,
            vmax: 0.0,
        };
        for &(quant_label, quant) in &quantizers {
            for channel in 0..3 {
                let mut channels = [zeros.clone(), zeros.clone(), zeros.clone()];
                channels[channel] = codes.clone();
                let mut scales = [zero; 3];
                scales[channel] = quant;
                let brick = occupancy_test_brick(
                    (16, 16, 1),
                    channels[0].clone(),
                    channels[1].clone(),
                    channels[2].clone(),
                    (scales[0], scales[1], scales[2]),
                );
                for factor in [0, 1, 7, 64] {
                    assert_quantized_mip_matches_decoded(
                        &brick,
                        factor,
                        &format!("{quant_label}/channel-{channel}/factor-{factor}"),
                    );
                }
            }
        }
    }

    #[test]
    fn quantized_occupancy_matches_mixed_odd_volume_and_dilation_bit_exactly() {
        let dims = (13usize, 11usize, 5usize);
        let cells = dims.0 * dims.1 * dims.2;
        let mut seed = 0x9e37_79b9u32;
        let mut next_code = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as u8
        };
        let liquid: Vec<u8> = (0..cells).map(|_| next_code()).collect();
        let ice: Vec<u8> = (0..cells).map(|_| next_code()).collect();
        let mut precip = vec![0u8; cells];
        precip[0] = 1;
        precip[cells / 2] = 127;
        precip[cells - 1] = 255;
        let normal = LogQuant {
            vmin: 1.0e-10,
            vmax: 1.0e-2,
        };
        let nan_min = LogQuant {
            vmin: f64::NAN,
            vmax: 1.0,
        };
        let reversed = LogQuant {
            vmin: 0.5,
            vmax: 1.0e-4,
        };
        for (label, quantizers) in [
            ("all-normal", (normal, normal, normal)),
            ("nan-poisons-positive", (normal, nan_min, normal)),
            ("reversed-mix", (reversed, normal, reversed)),
        ] {
            let brick = occupancy_test_brick(
                dims,
                liquid.clone(),
                ice.clone(),
                precip.clone(),
                quantizers,
            );
            for factor in [0, 1, 2, 3, 8, 32] {
                assert_quantized_mip_matches_decoded(
                    &brick,
                    factor,
                    &format!("{label}/factor-{factor}"),
                );
            }
        }
    }

    #[test]
    fn quantized_occupancy_real_fixture_benchmark() {
        use sha2::{Digest, Sha256};

        let Ok(path) = std::env::var("SIMSAT_CPU_OCCUPANCY_BENCH_BRICK") else {
            eprintln!("SIMSAT_CPU_OCCUPANCY_BENCH_BRICK unset; skipping occupancy benchmark");
            return;
        };
        let mode = std::env::var("SIMSAT_CPU_OCCUPANCY_BENCH_MODE")
            .unwrap_or_else(|_| "compare".to_string());
        let read_started = std::time::Instant::now();
        let brick = crate::bricks::read_ssb(std::path::Path::new(&path))
            .unwrap_or_else(|error| panic!("read benchmark brick {path}: {error}"));
        let read_wall = read_started.elapsed();
        let prep_started = std::time::Instant::now();
        let mip = match mode.as_str() {
            "quantized" => OccupancyMip::from_quantized_brick(&brick, OCCUPANCY_MIP_FACTOR),
            "decoded" => {
                let decoded = DecodedVolume::from_brick_legacy(&brick, 1000.0);
                OccupancyMip::build(&decoded, OCCUPANCY_MIP_FACTOR)
            }
            "compare" => {
                let quantized = OccupancyMip::from_quantized_brick(&brick, OCCUPANCY_MIP_FACTOR);
                let decoded = DecodedVolume::from_brick_legacy(&brick, 1000.0);
                let reference = OccupancyMip::build(&decoded, OCCUPANCY_MIP_FACTOR);
                let got: Vec<u32> = quantized.maxext.iter().map(|v| v.to_bits()).collect();
                let expected: Vec<u32> = reference.maxext.iter().map(|v| v.to_bits()).collect();
                assert_eq!(got, expected, "real-fixture occupancy mismatch");
                quantized
            }
            other => panic!("unknown SIMSAT_CPU_OCCUPANCY_BENCH_MODE={other}"),
        };
        let prep_wall = prep_started.elapsed();
        let digest = Sha256::digest(mip.to_r8_occupancy());
        let hash = format!("{digest:x}");
        let peak_rss = crate::platform::peak_rss_bytes()
            .map_or_else(|| "unknown".to_string(), |bytes| bytes.to_string());
        eprintln!(
            "CPU_OCCUPANCY_BENCH mode={mode} brick_dims={}x{}x{} occ_dims={}x{}x{} \
             occ_cells={} sha256={hash} read_wall_s={:.6} prep_wall_s={:.6} \
             peak_rss_bytes={peak_rss}",
            brick.nx,
            brick.ny,
            brick.nz,
            mip.mx,
            mip.my,
            mip.mz,
            mip.maxext.len(),
            read_wall.as_secs_f64(),
            prep_wall.as_secs_f64(),
        );
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
        let mut scene = scene_ref(&vol, &mip, &sun_od, &georef, luts, sky_sh, sun);
        // This is an analytic physical-depth probe, not a shipped appearance-preset
        // test. Keep the slab unscaled when the product default is recalibrated.
        scene.cfg.cloud_optical_depth_scale = 1.0;
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
            ground_day_lift: GROUND_DAY_LIFT,
            cloud_softclip_knee: CLOUD_SOFTCLIP_KNEE,
            cloud_highlight_max: crate::render::RHO_HIGHLIGHT_MAX,
            synthetic_green: false,
            atmosphere_correction: true,
            terrain_atmosphere: true,
            land_appearance: crate::render::LandAppearanceConfig::identity(),
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
        let bytes = render_cloud_frame_rgba(&scene, &surf, &froxel, &raster, &assemble);
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
        let refl = render_cloud_frame_reflectance(&scene, &surf, &froxel, &raster, &assemble);
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
                ground_day_lift: GROUND_DAY_LIFT,
                cloud_softclip_knee: CLOUD_SOFTCLIP_KNEE,
                cloud_highlight_max: crate::render::RHO_HIGHLIGHT_MAX,
                synthetic_green: false,
                atmosphere_correction: true,
                terrain_atmosphere: true,
                land_appearance: crate::render::LandAppearanceConfig::identity(),
            };
            render_cloud_frame_rgba(&scene, &surf, &froxel, &raster, &assemble)
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
    fn multiscatter_higher_orders_vanish_in_thin_limit_but_thick_cloud_is_preserved() {
        let (cos, el, ip, tau_sun) = (-0.7, 3.0e-3, 1.0e-3, 4.0);
        let single = octave_sun_source_thin_gated(cos, el, ip, tau_sun, false, 1, 0.0);
        let zero_support =
            octave_sun_source_thin_gated(cos, el, ip, tau_sun, false, DEFAULT_OCTAVES, 0.0);
        assert_eq!(
            zero_support.to_bits(),
            single.to_bits(),
            "zero cloud support must leave exactly octave zero"
        );

        let thin =
            octave_sun_source_thin_gated(cos, el, ip, tau_sun, false, DEFAULT_OCTAVES, 1.0e-3);
        assert!(
            thin > single && thin - single < 0.02 * single,
            "thin-cloud higher orders must be present but asymptotically tiny: {thin} vs {single}"
        );

        let old_thick = octave_sun_source(cos, el, ip, tau_sun, false, DEFAULT_OCTAVES);
        let gated_thick =
            octave_sun_source_thin_gated(cos, el, ip, tau_sun, false, DEFAULT_OCTAVES, 20.0);
        assert!(
            (gated_thick - old_thick).abs() < 1.0e-7 * old_thick,
            "thick-cloud gate must converge to the established octave result: {gated_thick} vs {old_thick}"
        );
    }

    #[test]
    fn real_column_multiscatter_is_invariant_to_coarse_sun_od_values() {
        // HRRR exposed a release-blocking 512-texel dash pattern when the coarse
        // ground-shadow map was also allowed to modulate higher-order cloud light.
        // A real ingested volume has a smooth whole-column `tau_up` value, so two
        // otherwise-identical cloud marches must be byte-identical even if the
        // auxiliary shadow map is changed from clear to extremely opaque.
        let (nx, ny, nz) = (16usize, 16usize, 24usize);
        let dz = 250.0;
        let mut vol = build_volume(nx, ny, nz, dz, 3000.0, |_, _, _| (5.0e-5, 0.0, 0.0));
        rebuild_test_tau_up(&mut vol);
        assert!(vol.sample(8.0, 8.0, 0.0).tau_up > 0.0);

        let georef = test_georef(nx, ny, 3000.0);
        let mip = OccupancyMip::build(&vol, OCCUPANCY_MIP_FACTOR);
        let sample = brick_to_ecef(&georef, 8.0, 8.0, 12.0, 0.0, dz).unwrap();
        let sun = norm3(sample);
        let mut clear_map = accumulate_sun_od(&vol, &georef, sun, 32);
        clear_map.od.fill(0.0);
        let mut opaque_map = clear_map.clone();
        opaque_map.od.fill(12.0);
        let (luts, sky_sh) = shared_luts();
        let cam = CameraGeometry::from_sub_lon(-100.0);
        let ground = brick_to_ecef(&georef, 8.0, 8.0, 0.0, 0.0, dz).unwrap();
        let view = norm3([
            ground[0] - cam.camera[0],
            ground[1] - cam.camera[1],
            ground[2] - cam.camera[2],
        ]);
        let cfg = MarchConfig {
            cloud_optical_depth_scale: 1.0,
            sun_march_jitter_amp: 0.0,
            octaves: DEFAULT_OCTAVES,
            ..MarchConfig::new(StepQuality::Offline, vol.voxel_pitch_m())
        };
        let march_with = |sun_od: &SunOdMap| {
            let scene = CloudScene {
                vol: &vol,
                mip: &mip,
                sun_od,
                georef: &georef,
                luts,
                sky_sh,
                sun_ecef: sun,
                cfg,
            };
            march_cloud(&scene, cam.camera, view)
        };
        let clear = march_with(&clear_map);
        let opaque = march_with(&opaque_map);
        assert_eq!(
            clear.transmittance.to_bits(),
            opaque.transmittance.to_bits()
        );
        for band in 0..3 {
            assert_eq!(
                clear.sun_inscatter[band].to_bits(),
                opaque.sun_inscatter[band].to_bits(),
                "coarse sun-OD leaked into real-column cloud light in band {band}: {} vs {}",
                clear.sun_inscatter[band],
                opaque.sun_inscatter[band]
            );
            assert_eq!(
                clear.inscatter[band].to_bits(),
                opaque.inscatter[band].to_bits()
            );
        }
    }

    #[test]
    fn zero_column_multiscatter_retains_the_legacy_sun_od_fallback() {
        // Synthetic/legacy volumes can lack `tau_up` (all zero). In that explicit
        // missing-data case the sun-aligned map remains the best available column
        // support estimate, so it must still enable higher scattering orders.
        let (nx, ny, nz) = (16usize, 16usize, 24usize);
        let dz = 250.0;
        let vol = build_volume(nx, ny, nz, dz, 3000.0, |_, _, _| (5.0e-5, 0.0, 0.0));
        assert_eq!(vol.sample(8.0, 8.0, 0.0).tau_up, 0.0);
        let georef = test_georef(nx, ny, 3000.0);
        let mip = OccupancyMip::build(&vol, OCCUPANCY_MIP_FACTOR);
        let sample = brick_to_ecef(&georef, 8.0, 8.0, 12.0, 0.0, dz).unwrap();
        let sun = norm3(sample);
        let mut clear_map = accumulate_sun_od(&vol, &georef, sun, 32);
        clear_map.od.fill(0.0);
        let mut opaque_map = clear_map.clone();
        opaque_map.od.fill(12.0);
        let (luts, sky_sh) = shared_luts();
        let cam = CameraGeometry::from_sub_lon(-100.0);
        let ground = brick_to_ecef(&georef, 8.0, 8.0, 0.0, 0.0, dz).unwrap();
        let view = norm3([
            ground[0] - cam.camera[0],
            ground[1] - cam.camera[1],
            ground[2] - cam.camera[2],
        ]);
        let cfg = MarchConfig {
            cloud_optical_depth_scale: 1.0,
            sun_march_jitter_amp: 0.0,
            octaves: DEFAULT_OCTAVES,
            ..MarchConfig::new(StepQuality::Offline, vol.voxel_pitch_m())
        };
        let march_with = |sun_od: &SunOdMap| {
            let scene = CloudScene {
                vol: &vol,
                mip: &mip,
                sun_od,
                georef: &georef,
                luts,
                sky_sh,
                sun_ecef: sun,
                cfg,
            };
            march_cloud(&scene, cam.camera, view)
        };
        let clear: f64 = march_with(&clear_map).sun_inscatter.iter().sum();
        let opaque: f64 = march_with(&opaque_map).sun_inscatter.iter().sum();
        assert!(
            opaque > clear * 1.25,
            "missing tau_up must retain sun-OD support fallback: {opaque} vs {clear}"
        );
    }

    #[test]
    fn ground_shadow_still_consumes_the_sun_od_map() {
        // The release fix is scoped only to cloud-volume higher-order support. The
        // whole sun-aligned column remains exactly the correct ground-shadow source.
        let (nx, ny, nz) = (16usize, 16usize, 24usize);
        let dz = 250.0;
        let vol = build_volume(nx, ny, nz, dz, 3000.0, |_, _, _| (5.0e-5, 0.0, 0.0));
        let georef = test_georef(nx, ny, 3000.0);
        let mip = OccupancyMip::build(&vol, OCCUPANCY_MIP_FACTOR);
        let sample = brick_to_ecef(&georef, 8.0, 8.0, 12.0, 0.0, dz).unwrap();
        let sun = norm3(sample);
        let mut clear_map = accumulate_sun_od(&vol, &georef, sun, 32);
        clear_map.od.fill(0.0);
        let mut opaque_map = clear_map.clone();
        opaque_map.od.fill(12.0);
        let (luts, sky_sh) = shared_luts();
        let cam = CameraGeometry::from_sub_lon(-100.0);
        let ground = brick_to_ecef(&georef, 8.0, 8.0, 0.0, 0.0, dz).unwrap();
        let view = norm3([
            ground[0] - cam.camera[0],
            ground[1] - cam.camera[1],
            ground[2] - cam.camera[2],
        ]);
        let cfg = MarchConfig {
            cloud_optical_depth_scale: 1.0,
            ..MarchConfig::new(StepQuality::Offline, vol.voxel_pitch_m())
        };
        let shadow_with = |sun_od: &SunOdMap| {
            let scene = CloudScene {
                vol: &vol,
                mip: &mip,
                sun_od,
                georef: &georef,
                luts,
                sky_sh,
                sun_ecef: sun,
                cfg,
            };
            ground_cloud_shadow(&scene, cam.camera, view)
        };
        let clear = shadow_with(&clear_map);
        let opaque = shadow_with(&opaque_map);
        assert!(
            clear > 0.999,
            "zero sun-OD should leave clear ground: {clear}"
        );
        assert!(
            opaque < 1.0e-4,
            "large sun-OD must still shadow the ground: {opaque}"
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
        // so its ground-shadow penumbra (blur radius = occ_dist x tan 0.2665 deg) is
        // wider — the EXTRA softening over the sharp e^-od shadow scales with height.
        let (nx, ny, nz) = (32, 32, 100);
        let (dx, dz) = (500.0, 500.0);
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
        let low = build(2, 6); // ~1-3 km
        // A deliberately high synthetic layer makes the corrected half-angle blur
        // resolvable above the map's own bilinear edge width.
        let high = build(80, 84); // ~40-42 km
        let center = brick_to_ecef(&georef, 16.0, 16.0, 50.0, 0.0, dz).unwrap();
        let sun = norm3(center); // local zenith over the box
        // Resolve the physical half-angle penumbra (about 55 m at 12 km). The former
        // diameter bug made a coarser 256 map appear sufficient by doubling the blur.
        let res = 512;
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

    #[test]
    fn penumbra_uses_the_solar_angular_radius_not_diameter() {
        let distance = 10_000.0;
        let got = solar_penumbra_radius_m(distance);
        let want = distance * atmosphere::SUN_ANGULAR_RADIUS_RAD.tan();
        let old_diameter_radius = distance * (2.0 * atmosphere::SUN_ANGULAR_RADIUS_RAD).tan();
        assert!((got - want).abs() < 1.0e-12);
        assert!((got - 46.5).abs() < 0.2, "10 km penumbra radius {got} m");
        assert!(
            got < old_diameter_radius * 0.51,
            "must not use the full diameter"
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
    fn exposed_domain_edge_feather_is_opt_in_and_preserves_margin_behavior() {
        let (nx, ny) = (200usize, 300usize);
        let all_i = vec![50.0f32; 12];
        let all_j = vec![75.0f32; 12];
        let mut exposed_i = all_i.clone();
        exposed_i[0] = f32::NAN;
        let band = EDGE_FEATHER_BAND_FRAC * nx.min(ny) as f64;

        // Default/off is the exact current margin-zero identity even if the camera
        // exposes samples beyond the finite WRF domain.
        assert_eq!(
            edge_feather_cells_for_raster(0.0, nx, ny, false, &exposed_i, &all_j),
            0.0
        );
        // On activates only when the actual raster exposes the domain boundary.
        assert_eq!(
            edge_feather_cells_for_raster(0.0, nx, ny, true, &exposed_i, &all_j),
            band
        );
        assert_eq!(
            edge_feather_cells_for_raster(0.0, nx, ny, true, &all_i, &all_j),
            0.0,
            "an all-in-domain top-down raster stays unchanged"
        );
        // Positive margin already used the same reviewed band and remains identical
        // with either experiment setting.
        assert_eq!(
            edge_feather_cells_for_raster(0.1, nx, ny, false, &all_i, &all_j),
            band
        );
        assert_eq!(
            edge_feather_cells_for_raster(0.1, nx, ny, true, &all_i, &all_j),
            band
        );
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
                // The convergence reference is the unscaled analytic optical depth.
                cloud_optical_depth_scale: 1.0,
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
                // The independently computed midpoint schedule is unscaled.
                cloud_optical_depth_scale: 1.0,
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
        // This horizon-fade test uses octave zero, so the sun-OD map's higher-order
        // support gate is irrelevant; one dummy map suffices.
        let sun_od = accumulate_sun_od(&vol, &georef, [0.0, 0.0, 1.0], 4);
        let v_at = |e_deg: f64| -> f64 {
            let er = e_deg.to_radians();
            let sun = norm3(add3(scl3(up, er.sin()), scl3(east, er.cos())));
            let cfg = MarchConfig {
                sun_march_jitter_amp: 0.0,
                octaves: 1,
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
        let mut scene = scene_ref(&vol, &mip, &sun_od, &georef, luts, sky_sh, sun);
        // The fine-step reference integrates raw model extinction.
        scene.cfg.cloud_optical_depth_scale = 1.0;
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
        let shadow = od.penumbral_shadow(inside);
        assert!(shadow < 0.9);
        assert_eq!(
            od.penumbral_shadow_scaled(inside, 0.0),
            1.0,
            "zero visible OD scale must cast no ground shadow"
        );
        assert!(
            od.penumbral_shadow_scaled(inside, 0.5) > shadow,
            "half-scale OD must make the ground shadow more transmissive"
        );
        assert!(
            od.penumbral_shadow_scaled(inside, 2.0) < shadow,
            "double-scale OD must make the ground shadow darker"
        );
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
    fn granulation_footprint_filter_reduces_500m_native_alias_energy() {
        let pitch = 500.0;
        let n = 48usize;
        let mut raw = vec![0.0f64; n * n];
        let mut filtered = vec![0.0f64; n * n];
        for y in 0..n {
            for x in 0..n {
                let u = (x as f64 + 0.5) * pitch;
                let v = (y as f64 + 0.5) * pitch;
                let c = y * n + x;
                raw[c] = granulation_erosion_noise(u, v);
                filtered[c] = granulation_erosion_noise_footprint(u, v, pitch);
                assert!((0.0..=1.0).contains(&filtered[c]));
                assert_eq!(
                    filtered[c].to_bits(),
                    granulation_erosion_noise_footprint(u, v, pitch).to_bits(),
                    "footprint filter must be deterministic"
                );
                assert_eq!(
                    granulation_erosion_noise_footprint(u, v, 0.0).to_bits(),
                    raw[c].to_bits(),
                    "zero footprint must be the exact point sample"
                );
            }
        }
        let neighbor_energy = |field: &[f64]| {
            let mut sum = 0.0;
            let mut count = 0usize;
            for y in 0..n {
                for x in 0..n {
                    let c = y * n + x;
                    if x + 1 < n {
                        let d = field[c] - field[c + 1];
                        sum += d * d;
                        count += 1;
                    }
                    if y + 1 < n {
                        let d = field[c] - field[c + n];
                        sum += d * d;
                        count += 1;
                    }
                }
            }
            sum / count as f64
        };
        let raw_energy = neighbor_energy(&raw);
        let filtered_energy = neighbor_energy(&filtered);
        assert!(
            filtered_energy < 0.5 * raw_energy,
            "500 m footprint did not suppress native-grid alias energy: raw={raw_energy}, filtered={filtered_energy}"
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
        // Liveness sweep: a dense ray grid across the blob. (Round 2 note: the
        // BIMODAL carve restores weakly-eroded samples to exactly 1.0, so only rays
        // crossing the strong carve network differ — a handful of rays can miss it.)
        let mut any_differ = false;
        for si in 0..10 {
            for sj in 0..10 {
                let gi = 8.0 + si as f64 * 0.6;
                let gj = 8.0 + sj as f64 * 0.6;
                let target = brick_to_ecef(&georef, gi, gj, 0.0, 0.0, 250.0).unwrap();
                let view = norm3([
                    target[0] - cam.camera[0],
                    target[1] - cam.camera[1],
                    target[2] - cam.camera[2],
                ]);
                let m_off = march_cloud(&scene_off, cam.camera, view);
                let m_on = march_cloud(&scene_on, cam.camera, view);
                if (m_on.transmittance - m_off.transmittance).abs() > 1e-9
                    || m_on.inscatter != m_off.inscatter
                {
                    any_differ = true;
                }
            }
        }
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

    #[test]
    fn granulation_coherence_gate_closes_a_solid_deck_and_its_margins() {
        // ROUND-2 headline regression (the owner's "cheese grater"): a spatially
        // COHERENT deck — including its soft MARGINS — must gain ZERO carved holes
        // under the deck-coherence gate. The in-test round-1 contrast (the same
        // sampler WITHOUT the coherence field) proves the margins DID erode before,
        // i.e. this test fails against the round-1 code by construction.
        let (nx, ny, nz) = (64usize, 64usize, 10usize);
        // A wide deck (i, j in 8..56) with +/-10% cell-to-cell variability and a SOFT
        // MARGIN: extinction tapers 0.6 / 0.3 over the two cells outside the rect.
        let deck = build_volume(nx, ny, nz, 250.0, 3000.0, |i, j, k| {
            if !(2..6).contains(&k) {
                return (0.0, 0.0, 0.0);
            }
            let di = if i < 8 { 8 - i } else { i.saturating_sub(55) };
            let dj = if j < 8 { 8 - j } else { j.saturating_sub(55) };
            let dist = di.max(dj);
            let factor = match dist {
                0 => 1.0,
                1 => 0.6,
                2 => 0.3,
                _ => return (0.0, 0.0, 0.0),
            };
            let var = 1.0 + 0.1 * (((i * 5 + j * 11 + k) % 7) as f64 / 3.0 - 1.0);
            (2.0e-2 * factor * var, 0.0, 0.0)
        });
        let coh = GranCoherence::build(&deck);
        let (open, partial, closed) = coh.stats();
        println!("GRAN coherence deck: open={open} partial={partial} closed={closed}");
        assert!(
            coh.gate_at(32.0, 32.0) <= 1.0e-6,
            "deck interior gate must be closed: {}",
            coh.gate_at(32.0, 32.0)
        );
        let gran = Some(Granulation::for_grid(3000.0));
        let mut probes = 0usize;
        let mut round1_eroded = 0usize;
        for pi in 0..120 {
            for pj in 0..120 {
                let fi = 5.05 + pi as f64 * 0.45;
                let fj = 5.05 + pj as f64 * 0.45;
                if fi > 58.95 || fj > 58.95 {
                    continue;
                }
                for &fk in &[2.5f64, 3.5, 4.6, 5.4] {
                    let raw = deck.sample(fi, fj, fk);
                    if raw.total_ext() <= 0.0 {
                        continue;
                    }
                    // ROUND 2: the coherence-gated sampler is byte-identical across
                    // the WHOLE deck, margins included.
                    let gated = deck.sample_granulated_gated(fi, fj, fk, gran, Some(&coh));
                    assert_eq!(
                        gated.total_ext().to_bits(),
                        raw.total_ext().to_bits(),
                        "coherent deck carved at ({fi},{fj},{fk})"
                    );
                    // ROUND-1 contrast: without the coherence field the margins erode
                    // (counted below; the pepper this round exists to kill).
                    let ungated = deck.sample_granulated(fi, fj, fk, gran);
                    if ungated.total_ext() < raw.total_ext() {
                        round1_eroded += 1;
                    }
                    probes += 1;
                }
            }
        }
        assert!(probes > 5_000, "sweep too sparse: {probes}");
        assert!(
            round1_eroded > 100,
            "the round-1 (un-gated) path should erode the deck margins — the defect \
             this gate exists to fix: {round1_eroded} eroded of {probes}"
        );
    }

    #[test]
    fn granulation_coherence_gate_keeps_a_broken_field_granulating() {
        // The regime separation in ONE volume: a solid uniform deck on the left
        // (wide enough that its ERODED core region is non-empty — a real deck), a
        // broken popcorn field on the right. The gate must stay CLOSED over the deck
        // (and its edge) and fully OPEN over the popcorn, and the gated sampler must
        // still erode the popcorn (the feature survives the gate).
        let (nx, ny, nz) = (48usize, 40usize, 10usize);
        let vol = build_volume(nx, ny, nz, 250.0, 3000.0, |i, j, k| {
            let deck = (2..26).contains(&i) && (2..38).contains(&j) && (2..6).contains(&k);
            let popcorn = i >= 34 && (i % 5 == 2) && (j % 4 == 1) && (2..7).contains(&k);
            if deck || popcorn {
                (2.0e-2, 0.0, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        });
        let coh = GranCoherence::build(&vol);
        // Deck interior + edge closed; popcorn columns fully open.
        assert!(
            coh.gate_at(12.0, 20.0) <= 1.0e-6,
            "deck interior must close"
        );
        assert!(coh.gate_at(25.9, 20.0) <= 1.0e-6, "deck edge must close");
        for &(fi, fj) in &[(37.0f64, 9.0f64), (42.0, 21.0), (47.0, 33.0)] {
            assert!(
                coh.gate_at(fi, fj) >= 0.9,
                "broken-field gate must stay open at ({fi},{fj}): {}",
                coh.gate_at(fi, fj)
            );
        }
        // The gated sampler still granulates the popcorn.
        let gran = Some(Granulation::for_grid(3000.0));
        let mut eroded = 0usize;
        let mut carved_clear = 0usize;
        for pi in 0..80 {
            for pj in 0..80 {
                let fi = 36.05 + pi as f64 * 0.14;
                let fj = 5.05 + pj as f64 * 0.42;
                let raw = vol.sample(fi, fj, 3.0);
                if raw.total_ext() <= 0.0 {
                    continue;
                }
                let gated = vol.sample_granulated_gated(fi, fj, 3.0, gran, Some(&coh));
                if gated.total_ext() < raw.total_ext() {
                    eroded += 1;
                }
                if gated.total_ext() <= 1.0e-9 {
                    carved_clear += 1;
                }
            }
        }
        println!("GRAN coherence popcorn: eroded={eroded} carved_clear={carved_clear}");
        assert!(
            eroded > 100,
            "the broken field must still granulate under the gate: {eroded}"
        );
        assert!(
            carved_clear > 20,
            "the bimodal carve should open real clear gaps: {carved_clear}"
        );
        // And the deck half of the SAME volume is byte-untouched through the same
        // gated sampler (margins included).
        for pi in 0..90 {
            for pj in 0..60 {
                let fi = 1.05 + pi as f64 * 0.3;
                let fj = 1.05 + pj as f64 * 0.62;
                let raw = vol.sample(fi, fj, 3.0);
                if raw.total_ext() <= 0.0 {
                    continue;
                }
                let gated = vol.sample_granulated_gated(fi, fj, 3.0, gran, Some(&coh));
                assert_eq!(
                    gated.total_ext().to_bits(),
                    raw.total_ext().to_bits(),
                    "deck half carved at ({fi},{fj})"
                );
            }
        }
    }

    #[test]
    fn granulation_bimodal_carve_narrows_the_half_eroded_middle() {
        // Anchors of the bimodal shape itself.
        assert_eq!(granulation_bimodal(0.0), 0.0);
        assert_eq!(granulation_bimodal(1.0), 1.0);
        assert_eq!(granulation_bimodal(GRAN_BIMODAL_GAP), 0.0);
        assert_eq!(granulation_bimodal(GRAN_BIMODAL_GRAIN), 1.0);
        assert_eq!(
            granulation_bimodal(0.1),
            0.0,
            "below the gap point carves clear"
        );
        assert_eq!(
            granulation_bimodal(0.9),
            1.0,
            "above the grain point restores full"
        );
        let mut prev = 0.0f64;
        for i in 0..=40 {
            let m = i as f64 / 40.0;
            let b = granulation_bimodal(m);
            assert!((0.0..=1.0).contains(&b));
            assert!(b >= prev - 1e-12, "bimodal shape must be monotone");
            prev = b;
        }
        // The DISTRIBUTION metric on the popcorn tent (the granulation target case):
        // among ENGAGED samples (a live erosion threshold, below the neighbourhood
        // max) the round-1 multiplier m1 leaves a broad half-eroded middle band that
        // rendered as grey mush; the bimodal carve m2 = B(m1) must (a) squeeze that
        // middle to less than half of round 1's and (b) place the bulk of eroded
        // samples at grain-or-gap. The identity shape gives mid2 == mid1, so (a)
        // FAILS against the round-1 code by construction.
        let (nx, ny, nz) = (16usize, 16usize, 8usize);
        let vol = build_volume(nx, ny, nz, 250.0, 3000.0, |i, j, k| {
            if i == 8 && j == 8 && k == 3 {
                (2.0e-2, 0.0, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        });
        let amp = granulation_amplitude(3000.0);
        let gran = Some(Granulation { amplitude: amp });
        let peak = vol.sample(8.0, 8.0, 3.0).total_ext();
        let (mut engaged, mut mid1, mut mid2, mut bulk2) = (0usize, 0usize, 0usize, 0usize);
        for pi in 0..99 {
            for pj in 0..99 {
                let fi = 7.02 + pi as f64 * 0.02;
                let fj = 7.02 + pj as f64 * 0.02;
                let raw = vol.sample(fi, fj, 3.0);
                let total = raw.total_ext();
                if total <= 0.0 {
                    continue;
                }
                // Replicate the sampler's erosion pipeline through the PUBLIC pure
                // functions (corner max == peak for this single-cell tent).
                let d = total / peak;
                let gate = granulation_gate(raw.ext_liquid, raw.ext_ice, raw.ext_precip, 750.0);
                let prot = granulation_interior_protection(d);
                let noise = granulation_erosion_noise_footprint(
                    fi * vol.horiz_pitch_m,
                    fj * vol.horiz_pitch_m,
                    vol.horiz_pitch_m,
                );
                let e = (amp * GRAN_EROSION_GAIN * gate * prot * noise).min(GRAN_EROSION_MAX);
                if e <= 0.05 || d >= 0.999 {
                    continue;
                }
                let m1 = granulation_multiplier(d, e);
                let m2 = granulation_bimodal(m1);
                engaged += 1;
                if m1 > 0.2 && m1 < 0.8 {
                    mid1 += 1;
                }
                if m2 > 0.2 && m2 < 0.8 {
                    mid2 += 1;
                }
                if !(0.2..=0.8).contains(&m2) {
                    bulk2 += 1;
                }
                // Integration lock: the sampler applies exactly B(m1).
                let expected = if m2 < 1.0 { total * m2 } else { total };
                let ero = vol.sample_granulated(fi, fj, 3.0, gran).total_ext();
                assert_eq!(
                    ero.to_bits(),
                    expected.to_bits(),
                    "sampler multiplier drifted from B(remap) at ({fi},{fj})"
                );
            }
        }
        let f_mid1 = mid1 as f64 / engaged as f64;
        let f_mid2 = mid2 as f64 / engaged as f64;
        let f_bulk2 = bulk2 as f64 / engaged as f64;
        println!(
            "GRAN bimodal: engaged={engaged} mid1={f_mid1:.4} mid2={f_mid2:.4} bulk2={f_bulk2:.4}"
        );
        assert!(engaged > 500, "engaged set too small: {engaged}");
        assert!(
            f_mid2 < 0.5 * f_mid1,
            "the bimodal carve must narrow the grey middle band to under half of \
             round 1's: mid2 {f_mid2:.4} vs mid1 {f_mid1:.4}"
        );
        assert!(
            f_bulk2 > 0.8,
            "the bulk of eroded samples must sit at grain-or-gap: {f_bulk2:.4}"
        );
    }

    #[test]
    fn granulation_thinness_gate_stands_down_on_translucent_veils() {
        // Owner round-2 addendum: a regionally-THIN broken veil (window-max eligible
        // tau below GRAN_THIN_TAU_LO) already reads wispy through the honest
        // transmittance gradient — granulation must stand down there (erosion on a
        // veil is dark specks in grey, never grains). The SAME broken pattern at
        // substantial optical depth keeps granulating: thinness, not pattern, is
        // the discriminator. The un-gated sampler erodes the veil (the round-1
        // behavior this ramp removes), so the standdown fails against round-1 code.
        let (nx, ny, nz) = (24usize, 24usize, 10usize);
        let pattern =
            |i: usize, j: usize, k: usize| (i % 5 == 2) && (j % 4 == 1) && (2..6).contains(&k);
        // Thin veil: column tau = 4 layers x 250 m x 4e-4 = 0.4 < GRAN_THIN_TAU_LO.
        let veil = build_volume(nx, ny, nz, 250.0, 3000.0, |i, j, k| {
            if pattern(i, j, k) {
                (4.0e-4, 0.0, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        });
        // Substantial: column tau = 20 >> GRAN_THIN_TAU_HI, same pattern.
        let thick = build_volume(nx, ny, nz, 250.0, 3000.0, |i, j, k| {
            if pattern(i, j, k) {
                (2.0e-2, 0.0, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        });
        let coh_veil = GranCoherence::build(&veil);
        let coh_thick = GranCoherence::build(&thick);
        assert!(
            coh_veil.gate_at(7.0, 5.0) <= 1.0e-6,
            "thin-veil gate must stand down: {}",
            coh_veil.gate_at(7.0, 5.0)
        );
        assert!(
            coh_thick.gate_at(7.0, 5.0) >= 0.9,
            "substantial broken cloud must keep granulating: {}",
            coh_thick.gate_at(7.0, 5.0)
        );
        let gran = Some(Granulation::for_grid(3000.0));
        let (mut veil_ungated_eroded, mut thick_gated_eroded) = (0usize, 0usize);
        for pi in 0..70 {
            for pj in 0..70 {
                let fi = 1.05 + pi as f64 * 0.31;
                let fj = 1.05 + pj as f64 * 0.31;
                let raw_v = veil.sample(fi, fj, 3.0);
                if raw_v.total_ext() > 0.0 {
                    // Gated: byte-identical (the standdown).
                    let g = veil.sample_granulated_gated(fi, fj, 3.0, gran, Some(&coh_veil));
                    assert_eq!(
                        g.total_ext().to_bits(),
                        raw_v.total_ext().to_bits(),
                        "thin veil eroded at ({fi},{fj})"
                    );
                    // Un-gated (round 1): erodes.
                    let u = veil.sample_granulated(fi, fj, 3.0, gran);
                    if u.total_ext() < raw_v.total_ext() {
                        veil_ungated_eroded += 1;
                    }
                }
                let raw_t = thick.sample(fi, fj, 3.0);
                if raw_t.total_ext() > 0.0 {
                    let g = thick.sample_granulated_gated(fi, fj, 3.0, gran, Some(&coh_thick));
                    if g.total_ext() < raw_t.total_ext() {
                        thick_gated_eroded += 1;
                    }
                }
            }
        }
        println!(
            "GRAN thinness: veil_ungated_eroded={veil_ungated_eroded} \
             thick_gated_eroded={thick_gated_eroded}"
        );
        assert!(
            veil_ungated_eroded > 50,
            "the un-gated path should erode the veil (the round-1 defect): \
             {veil_ungated_eroded}"
        );
        assert!(
            thick_gated_eroded > 100,
            "substantial broken cloud must still granulate: {thick_gated_eroded}"
        );
    }

    #[test]
    fn granulation_domain_warp_is_deterministic_bounded_smooth_and_wired() {
        // Determinism.
        let (a0, a1) = granulation_warp_offset(12_345.0, -6_789.0);
        let (b0, b1) = granulation_warp_offset(12_345.0, -6_789.0);
        assert_eq!(a0.to_bits(), b0.to_bits());
        assert_eq!(a1.to_bits(), b1.to_bits());
        // Bounded by the amplitude; spatially VARYING (the whole point: cell
        // size/spacing changes across the scene); smooth (no popping between
        // adjacent samples — Lipschitz bound of the smoothstep value noise).
        let mut du_min = f64::INFINITY;
        let mut du_max = f64::NEG_INFINITY;
        for s in 0..200 {
            let u = s as f64 * 137.0;
            let v = s as f64 * 89.0 - 7_000.0;
            let (du, dv) = granulation_warp_offset(u, v);
            assert!(du.abs() <= GRAN_WARP_AMP_M && dv.abs() <= GRAN_WARP_AMP_M);
            du_min = du_min.min(du);
            du_max = du_max.max(du);
        }
        assert!(
            du_max - du_min > 500.0,
            "the warp must vary across the scene: range {}",
            du_max - du_min
        );
        for s in 0..300 {
            let u = 500.0 + s as f64 * 50.0;
            let (d0, _) = granulation_warp_offset(u, 4_321.0);
            let (d1, _) = granulation_warp_offset(u + 50.0, 4_321.0);
            assert!(
                (d1 - d0).abs() < 60.0,
                "warp not smooth: step {} at u={u}",
                (d1 - d0).abs()
            );
        }
        // WIRED: the erosion noise consumes exactly this warp (bit-exact
        // reconstruction through the same private octave stack), and the warp
        // actually displaces the field somewhere.
        let mut differs = false;
        for &(u, v) in &[
            (1_234.0f64, 5_678.0f64),
            (20_000.0, 3_000.0),
            (7_500.0, 44_000.0),
            (61_000.0, 12_500.0),
        ] {
            let (du, dv) = granulation_warp_offset(u, v);
            let (uw, vw) = (u + du, v + dv);
            let mut w = 0.0f64;
            let mut w0 = 0.0f64;
            for i in 0..GRAN_OCTAVE_SCALES_M.len() {
                let lam = GRAN_OCTAVE_SCALES_M[i];
                w += GRAN_OCTAVE_WEIGHTS[i] * worley2_f1(uw / lam, vw / lam, GRAN_OCTAVE_SALTS[i]);
                w0 += GRAN_OCTAVE_WEIGHTS[i] * worley2_f1(u / lam, v / lam, GRAN_OCTAVE_SALTS[i]);
            }
            let e_manual =
                smooth01((w.clamp(0.0, 1.0) - GRAN_CARVE_LO) / (GRAN_CARVE_HI - GRAN_CARVE_LO));
            assert_eq!(
                granulation_erosion_noise(u, v).to_bits(),
                e_manual.to_bits(),
                "the erosion noise must consume the domain warp"
            );
            if (w - w0).abs() > 1e-6 {
                differs = true;
            }
        }
        assert!(
            differs,
            "the warp should displace the octave field somewhere"
        );
    }
}
