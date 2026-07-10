//! Surface shading reference kernels + terrain normals (design doc section 5/6).
//!
//! [`shade_pixel`] is the M1 no-atmosphere kernel (kept as a documented reference).
//! [`shade_surface`] is the M2 kernel — the CPU-reference twin of the WGSL surface
//! shader (`gpu/shaders/surface.wgsl`): Blue Marble albedo, the finite-disk direct
//! sun, sky-view ambient (twilight), aerial perspective (transmittance + inscatter
//! from a view-ray raymarch), the off-earth LIMB, and the ABI-like reflectance
//! output transform. Nodes have no GPU, so the physics is CPU-tested here and the
//! WGSL is kept in lockstep by discipline (design section 9, test strategy 2).
//! When you change one, change the other and keep them twins.
//!
//! M1 [`shade_pixel`] shading (honest subset; every simplification is named):
//!   - albedo = Blue Marble texel (sRGB), converted to linear;
//!   - N-dot-L point sun (finite solar disk + penumbra are M2/M3);
//!   - LANDMASK-gated flat dark water (Cox-Munk glint is M3);
//!   - sun below the horizon -> dark (no atmosphere/twilight; that is M2);
//!   - off-earth (PUG visibility fails) -> space (transparent/black);
//!   - output stretch = linear->sRGB gamma (the "sqrt-ish" satellite stretch; the
//!     full ABI reflectance factor / tonemap lands in M2).

/// Water albedo multiplier (flat dark water; glint is M3). Documented, not tuned.
/// Since WS2 this is the TWILIGHT anchor: at/below the day ramp's LO elevation the
/// water body uses exactly this scale (the locked M2 twilight look); at full day it
/// ramps to [`WATER_ALBEDO_DAY_SCALE`] (see `surface_toa_radiance`'s water branch).
pub const WATER_ALBEDO_SCALE: f32 = 0.55;

/// DAYTIME water-body albedo scale (WS2 water direct-sun pass). The water branch now
/// receives the same disk-gated, shadow-weighted DIRECT solar term the land branch has
/// (so cloud shadows exist over the ocean and ocean brightness responds to the sun);
/// that added flux would brighten the owner-approved dark-ocean look, so the water-body
/// albedo scale is simultaneously retuned DOWN at day on the SAME sun gate: effective
/// scale = `water_scale` (0.55, the twilight anchor) at/below the ramp LO, ramping to
/// this value at/above the ramp HI. Twilight is byte-identical by construction (gate =
/// 0 there); the dark-ocean/distinct-glint contrast is held by the lower day albedo.
pub const WATER_ALBEDO_DAY_SCALE: f64 = 0.35;

/// CLOUD-SHADOW FLOOR (WS2, QA item: daytime cloud ground-shadows read as very dark
/// hard navy blobs). The sun-OD ground shadow counts only EXTINCTION along the sun ray
/// — a fully-occluded ground pixel kept NO direct term and was lit by sky ambient
/// alone (blue-dominant -> navy). Physically the shadowed ground under a bright sunlit
/// cloud also receives strong DOWNSCATTERED flux from the cloud itself, which the
/// ground-shadow consumer does not model. This floor stands in for that fill: the
/// effective shadow is `f + (1 - f) * shadow` with `f = CLOUD_SHADOW_FLOOR *
/// smoothstep(LO, HI, sun_elev)` ([`effective_cloud_shadow`]) — sun-gated on the shared
/// day ramp so twilight is byte-identical, and an unshadowed pixel (`shadow = 1`) maps
/// to exactly `1` at every elevation (byte-identical). `0.0` = the old hard floor.
pub const CLOUD_SHADOW_FLOOR: f64 = 0.25;

/// Default display-side exposure gain (see [`radiance_to_rgba`]). A moderate
/// brightening over the pre-exposure implicit `1.0` — the owner reported renders
/// looked "too dark no matter the time", and this is the shipped first guess (the
/// studio slider + the `render_frame` CLI both default to it). `1.0` exactly
/// reproduces the pre-exposure output. Chosen deliberately below the level that
/// clips a bright sunlit anvil to solid white; the exact value is tuned from real
/// composited frames (notes/qa-notes.md), not from memory.
pub const DEFAULT_EXPOSURE: f64 = 1.6;

/// Neutral flat albedo (sRGB) used when the Blue Marble texture is absent, so the
/// studio still renders a lit sphere with a clear "texture missing" message.
pub const FLAT_ALBEDO_SRGB: f32 = 0.30;

// ── True-color calibration (refinement pass) ──────────────────────────────────
//
// A set of CALIBRATED display/albedo choices that push the visible frame toward the
// look of real GOES true-color RGB imagery (vivid green land, bright plains, dark
// ocean with a DISTINCT sun-glint streak), named honestly — none changes the
// underlying reflectance physics; each is a display-side calibration of the shipped
// radiance path, and each is a no-op at its identity value (so the identity set
// reproduces the pre-refinement output). Round 2 (this pass) pushed the daytime look
// further per the orchestrator's review against the real GOES references: more
// de-haze (lower veil), more vivid land (higher vibrancy), a modest LAND-only daytime
// BRIGHTNESS lift (LAND_DAY_GAIN, distinct from the global exposure), and a brighter
// + tighter sun glint (higher GLINT_STRENGTH + a narrowed Cox-Munk core GLINT_MSS_SCALE).
// All are sun-gated or surface-scoped so the M2 twilight look stays byte-identical.

/// Daytime aerial-perspective (Rayleigh) VEIL reduction. A geostationary view marches
/// the whole ~100 km atmospheric column to the ground, so a real (physically-correct)
/// Rayleigh-blue in-scatter haze is laid over the surface. Real GOES *true-color* RGB
/// products remove most of that molecular veil with a Rayleigh atmospheric correction
/// — which is exactly what reveals their vivid green land and dark ocean. We mirror
/// that: the in-scatter ADDED over an on-earth surface pixel is scaled by this factor
/// at high sun. `1.0` = the raw physical veil (no correction); `< 1` de-hazes. The
/// surface TRANSMITTANCE (the attenuation/reddening of the ground signal) is left
/// intact — only the additive haze is corrected, precisely as a Rayleigh correction
/// does — and the OFF-EARTH limb in-scatter is never touched (the space halo stays
/// physical). Gated by sun elevation (below), so twilight is untouched.
///
/// Round 2: `0.55 -> 0.40` (a bit more de-haze — the orchestrator found the round-1
/// ground still slightly hazy vs the vivid GOES true-color references).
pub const AERIAL_VEIL_DAY_SCALE: f64 = 0.40;

/// Sun-elevation gate (deg) for [`AERIAL_VEIL_DAY_SCALE`]: NO veil reduction at/below
/// LO (twilight and the terminator keep their FULL physical in-scatter — the M2
/// twilight tuning is preserved by construction, since it all lives below this
/// elevation) ramping to the full daytime reduction at/above HI. Chosen so the whole
/// twilight/terminator band (sun <= ~6 deg) is byte-unchanged and only daylight
/// de-hazes.
pub const AERIAL_VEIL_ELEV_LO_DEG: f64 = 20.0;
/// Upper end of the veil-reduction elevation ramp (deg). See [`AERIAL_VEIL_ELEV_LO_DEG`].
/// WS2 QA: `40 -> 30` — at a sun elevation of 30 deg (mid-morning) the ramp used to sit
/// at half, leaving the ground half-veiled/half-lifted ("murky"); full daytime treatment
/// now arrives by 30 deg. Frames at/above 40 deg are byte-identical (both ramps
/// saturated); the twilight band (<= 20 deg) is byte-identical as before.
pub const AERIAL_VEIL_ELEV_HI_DEG: f64 = 30.0;

/// Land-albedo VIBRANCY: a luminance-preserving saturation boost applied to the Blue
/// Marble LAND texel (in linear RGB, before shading) so vegetation reads as the vivid
/// green of real GOES true-color rather than the muted tone of the 2 km composite. It
/// flows through the full reflectance physics (direct + ambient + transmittance), so
/// it is an honest surface-albedo calibration, not a post-hoc colour push. WATER is
/// excluded (dark ocean must stay dark and its glint uncoloured). `1.0` = no change.
/// Round 2: `1.3 -> 1.45` (more vivid green vegetation vs the GOES true-color refs).
/// Luminance-preserving, so it does NOT clip land to white and does NOT change the
/// median luminance (the twilight no-regression metric); it only re-weights chroma.
pub const LAND_VIBRANCY: f64 = 1.45;

/// LAND daytime BRIGHTNESS lift (refinement pass, round 2). A modest ground-only gain
/// on the LAND surface reflectance (`l_surf`) at high sun — the orchestrator found the
/// round-1 de-hazed ground read a touch dark/muted vs the bright daylight land of the
/// GOES true-color references, so the surface signal is lifted toward that brightness.
/// This is a CALIBRATED TRUE-COLOR DISPLAY GAIN on the ground reflectance, NOT the
/// owner-approved global exposure ([`DEFAULT_EXPOSURE`] = 1.6, unchanged) and NOT an
/// albedo/physics change: it multiplies the assembled surface radiance before the
/// aerial-perspective veil is added, so only the ground signal brightens (the additive
/// haze is untouched). WATER is excluded (dark ocean stays dark; the glint has its own
/// gain). Sun-elevation-gated on the SAME ramp as the veil ([`land_day_gain`]), so at/
/// below [`AERIAL_VEIL_ELEV_LO_DEG`] it is exactly `1.0` — the whole twilight/terminator
/// band is byte-unchanged and the M2 twilight tuning is preserved by construction.
/// `1.0` = no lift. Chosen modest (below any land clip) and verified not to over-bright.
pub const LAND_DAY_GAIN: f64 = 1.20;

// ── top-down / basemap appearance pass (ground lift + highlight soft-clip) ────
//
// Two levers added to fix the reported "the ground renders too dark" (sunlit land
// peaked only ~0.53 display, ocean near-black — darker than real GOES true-color, so
// the owner had to crank exposure to 4, which then blew the storm cloud to a flat
// white square). Both are named display calibrations of the shipped radiance path (the
// LAND_DAY_GAIN / AERIAL_VEIL pattern) and are no-ops at their neutral values, so the
// owner-approved daytime + twilight looks are preserved by construction.

/// GROUND LIFT — a sun-gated daytime surface-brightness lift on BOTH land and ocean,
/// toward real-GOES true-color ground levels (the reported ground was too dark). Unlike
/// [`LAND_DAY_GAIN`] (land only, a modest vibrancy-companion lift) this lifts the WHOLE
/// surface radiance `l_surf` — land AND water — so the basemap reads bright/vivid and the
/// ocean is a visible dark blue rather than near-black, in BOTH the geostationary and
/// top-down views. It multiplies the assembled surface radiance BEFORE the aerial-
/// perspective veil (only the ground signal brightens, not the additive haze) and BEFORE
/// the cloud composite (the cloud's own radiance is not lifted — the "white square" is
/// handled separately by [`CLOUD_SOFTCLIP_KNEE`] + the top-down cloud normalization). It
/// is sun-elevation-gated on the SAME ramp as the veil / land gain ([`ground_day_lift`]),
/// so at/below [`AERIAL_VEIL_ELEV_LO_DEG`] it is exactly `1.0` — the whole twilight/
/// terminator band is byte-unchanged and the M2 twilight tuning is preserved. `1.0` = the
/// neutral no-op (reproduces the pre-lift ground). Value `2.0`: the iteration-1 experiment
/// found a ground gain of ~2.0-2.2 made the basemap vivid green / the ocean visible
/// without washing out; with the highlight soft-clip below, bright land keeps structure
/// rather than pinning to white. The `render_frame` `ground-gain=` CLI knob overrides it
/// (default = this baked value) for future tuning.
pub const GROUND_DAY_LIFT: f64 = 2.0;

/// CLOUD/HIGHLIGHT SOFT-CLIP knee — a Reinhard highlight shoulder applied in
/// [`radiance_to_rgba`] (the ONE tonemap both the surface pass and the cloud composite
/// call) so bright cloud tops (and bright ground highlights) keep STRUCTURE instead of
/// clamping to a flat white square. The reflectance factor `rho` (exposure-applied,
/// desaturated) is passed through [`soft_clip_highlight`] with this knee BEFORE the ABI
/// sqrt stretch: STRICTLY IDENTITY below the knee (so the approved mid-tones / daytime /
/// twilight are unchanged), and above the knee a smooth C1 Reinhard shoulder that maps
/// `[knee, +inf) -> [knee, 1)` — a bright anvil at `rho = 1.3` and one at `rho = 1.0`
/// land on distinct display values instead of both pinning to 1.0. `1.0` = the neutral
/// no-op (identity below 1, the old hard clamp above 1 — reproduces the pre-soft-clip
/// output). Value `0.75`: the knee below which the daytime surface / mid-tones sit, so
/// only the brightest tops are compressed. The `render_frame` `cloud-softclip=` CLI knob
/// overrides it (default = this baked value; `1.0` disables it).
pub const CLOUD_SOFTCLIP_KNEE: f64 = 0.75;

/// Sun-GLINT brightness gain. The Cox-Munk specular glint over water is physically a
/// thin, dim streak once band-averaged; real GOES shows a DISTINCT bright glint. This
/// is a calibrated display gain on the glint reflectance (its Cox-Munk angular WIDTH
/// is set by [`GLINT_MSS_SCALE`] below — this constant only lifts the peak brightness).
/// `1.0` = the raw Cox-Munk glint. Round 2: `2.0 -> 3.5` (the round-1 glint read as a
/// soft sheen; the orchestrator wants a distinct bright glitter streak toward the sun).
pub const GLINT_STRENGTH: f64 = 3.5;

/// Sun-GLINT specular-core NARROWING (refinement pass, round 2). Scales the Cox-Munk
/// mean-square slope `sigma^2` that sets the glint's angular width: `< 1` TIGHTENS the
/// sea-surface slope distribution, so the specular core is SMALLER and (energy pulled
/// into a narrower peak) BRIGHTER — a distinct glitter streak instead of a broad sheen.
/// This is physically the low-wind / calm-sea limit of the same Cox-Munk model (a
/// tighter slope PDF is exactly what a calmer sea has), applied as a calibration on the
/// modelled `sigma^2` at the glint call site; the Cox-Munk kernel itself is unchanged.
/// `1.0` = the raw modelled slope variance. Round 3: `0.6 -> 0.4` (tightened after the
/// review — round-2 confirmed the high-sun glint is a physically-correct broad wash and
/// asked to sharpen the LOW-sun streak where the specular core collapses to a tight
/// bright glitter facing the sensor; 0.5 was still blob-like at the coarse Michael d01
/// domain, so one more notch to the very-calm/glassy-sea limit). Water-only, and
/// negligible at twilight (the near-horizon specular facet is far off-normal, so the
/// tighter core makes it even smaller) — twilight-safe.
pub const GLINT_MSS_SCALE: f64 = 0.4;

// ── SNOWH snow blend (design section 5, M3) ───────────────────────────────────

/// Fresh-snow albedo (sRGB) blended over the Blue Marble ground where `SNOWH`
/// indicates snow. Deliberately below pure white (fresh snow is ~0.8 visible, and a
/// satellite-stretched full white reads as a blown-out cloud) with a slight cool
/// tint. A display/asset albedo blend, NOT a reflectance-physics change.
pub const SNOW_ALBEDO_SRGB: [f32; 3] = [0.80, 0.82, 0.85];

/// `SNOWH` (snow depth, m) below which no snow shows — thin snow does not cover the
/// vegetation/terrain the Blue Marble texel already carries.
pub const SNOW_DEPTH_MIN_M: f64 = 0.002;
/// `SNOWH` (snow depth, m) at/above which the ground reads as fully snow-covered.
pub const SNOW_DEPTH_FULL_M: f64 = 0.10;

/// Snow cover fraction `[0,1]` from `SNOWH` (snow depth, m): a monotone smoothstep
/// ramp between [`SNOW_DEPTH_MIN_M`] and [`SNOW_DEPTH_FULL_M`]. Below the min -> 0,
/// above the full depth -> 1. Non-finite input -> 0.
#[inline]
pub fn snow_fraction(snow_depth_m: f64) -> f32 {
    if !snow_depth_m.is_finite() || snow_depth_m <= SNOW_DEPTH_MIN_M {
        return 0.0;
    }
    if snow_depth_m >= SNOW_DEPTH_FULL_M {
        return 1.0;
    }
    let t = (snow_depth_m - SNOW_DEPTH_MIN_M) / (SNOW_DEPTH_FULL_M - SNOW_DEPTH_MIN_M);
    (t * t * (3.0 - 2.0 * t)) as f32 // smoothstep (monotone on [0,1])
}

/// Blend the fresh-snow albedo over a base sRGB colour by `snow_frac` `[0,1]`
/// (a linear lerp in sRGB — a subtle, monotone albedo brightening). Land only; the
/// caller gates on `!is_water`.
#[inline]
pub fn blend_snow(base_srgb: [f32; 3], snow_frac: f32) -> [f32; 3] {
    let f = snow_frac.clamp(0.0, 1.0);
    [
        base_srgb[0] * (1.0 - f) + SNOW_ALBEDO_SRGB[0] * f,
        base_srgb[1] * (1.0 - f) + SNOW_ALBEDO_SRGB[1] * f,
        base_srgb[2] * (1.0 - f) + SNOW_ALBEDO_SRGB[2] * f,
    ]
}

/// sRGB (IEC 61966-2-1) transfer, encoded -> linear. Matches the WGSL twin.
#[inline]
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear -> sRGB (IEC 61966-2-1). Matches the WGSL twin.
#[inline]
pub fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Inputs to one surface pixel's shading (the shader's per-pixel state).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShadeInputs {
    /// Whether this pixel is on the earth disk (false -> space).
    pub on_earth: bool,
    /// Base surface color (sRGB, `[0,1]`): the Blue Marble texel, or a flat albedo.
    pub base_srgb: [f32; 3],
    /// Surface normal in the local ENU basis (up if flat / outside the domain).
    pub normal_enu: [f32; 3],
    /// Unit sun direction in the local ENU basis.
    pub sun_dir_enu: [f32; 3],
    /// Sun elevation above the horizon (deg); `<= 0` -> night.
    pub sun_elev_deg: f32,
    /// Whether this pixel is water (LANDMASK-gated; only meaningful in-domain).
    pub is_water: bool,
}

/// Shade one surface pixel. Returns display-ready sRGB `rgba` in `[0,1]`; alpha is
/// `1.0` on earth and `0.0` off earth (the off-earth marker the store writer turns
/// into a transparent/NaN plane value and the display draws as black space).
pub fn shade_pixel(inp: &ShadeInputs) -> [f32; 4] {
    if !inp.on_earth {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let scale = if inp.is_water {
        WATER_ALBEDO_SCALE
    } else {
        1.0
    };
    let mut albedo = [
        srgb_to_linear(inp.base_srgb[0]) * scale,
        srgb_to_linear(inp.base_srgb[1]) * scale,
        srgb_to_linear(inp.base_srgb[2]) * scale,
    ];
    let ndotl = dot(inp.normal_enu, inp.sun_dir_enu).max(0.0) as f32;
    // Sun below the horizon -> dark (no twilight yet). The elevation is the
    // authority: a slope can face a sun that is technically grazing the horizon.
    let shade = if inp.sun_elev_deg > 0.0 { ndotl } else { 0.0 };
    for c in &mut albedo {
        *c *= shade;
    }
    [
        linear_to_srgb(albedo[0]),
        linear_to_srgb(albedo[1]),
        linear_to_srgb(albedo[2]),
        1.0,
    ]
}

#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f64 {
    a[0] as f64 * b[0] as f64 + a[1] as f64 * b[1] as f64 + a[2] as f64 * b[2] as f64
}

// ── M2 atmosphere-aware surface shading (twin of gpu/shaders/surface.wgsl) ─────

use crate::atmosphere::{
    self, AtmosphereLuts, AtmosphereParams, CameraGeometry, OutputTransform, SOLAR_IRRADIANCE_RGB,
    SkyShTable,
};
use crate::horizon;
use crate::optics;

/// Per-frame state shared by every pixel of one M2 surface render.
pub struct FrameContext<'a> {
    pub luts: &'a AtmosphereLuts,
    pub params: &'a AtmosphereParams,
    /// SH-2 directional sky ambient (M5) — replaces M2's scalar ambient table. Terrain
    /// evaluates it at the terrain normal over the full upper hemisphere (no horizon
    /// occlusion; the ambient-aperture cone from a horizon map arrives with M3).
    pub sky_sh: &'a SkyShTable,
    /// Geostationary camera geometry (ECEF).
    pub cam: CameraGeometry,
    /// Unit ECEF sun direction (constant across the domain; sun at infinity).
    pub sun_ecef: [f64; 3],
    pub output_transform: OutputTransform,
    /// Whether the Blue Marble texture is present (else the flat albedo).
    pub bm_present: bool,
    pub water_scale: f64,
    pub flat_albedo_srgb: f64,
    /// Raymarch step count (matches the shader `STEPS`).
    pub raymarch_steps: usize,
    /// Display-side exposure gain (see [`radiance_to_rgba`]): a single linear
    /// multiplier applied to the whole frame's reflectance before the ABI sqrt
    /// stretch. One exposure for the entire frame, so surface AND cloud brighten
    /// together (both composite through [`radiance_to_rgba`]). `1.0` reproduces the
    /// pre-exposure output; [`DEFAULT_EXPOSURE`] is the shipped default.
    pub exposure: f64,
}

/// One surface pixel's inputs for [`shade_surface`].
#[derive(Debug, Clone, Copy)]
pub struct SurfacePixel {
    /// On the solid earth disk (per the CGMS visibility test). False -> limb/space.
    pub on_earth: bool,
    /// Base surface colour (sRGB `[0,1]`): the Blue Marble texel or flat albedo.
    pub base_srgb: [f32; 3],
    /// Terrain normal in the local ENU basis (up if flat / outside the domain).
    pub normal_enu: [f32; 3],
    /// Unit sun direction in the local ENU basis.
    pub sun_enu: [f32; 3],
    /// Sun elevation above the local horizon (deg).
    pub sun_elev_deg: f32,
    /// LANDMASK-gated water (only meaningful in-domain).
    pub is_water: bool,
    /// Unit ECEF view direction (camera -> this pixel), valid on- and off-earth.
    pub view_dir: [f64; 3],
    /// M3 terrain cast shadow: the terrain horizon elevation angle (rad, >= 0) at the
    /// sun's azimuth for this pixel (from the [`crate::horizon::HorizonMap`]). `0` =
    /// no terrain occlusion (the flat/open no-op). Folded into the finite-disk direct
    /// term, so the sun is penumbrally occluded where a ridge rises above it.
    pub terrain_horizon_rad: f32,
    /// M3 ambient aperture: the visible-sky OPENNESS fraction `[0,1]` (1 = open
    /// ridgetop, ->0 = enclosed pocket) that scales the SH sky ambient. Default `1.0`.
    pub sky_openness: f32,
    /// M3 ambient aperture: the visible-sky BENT NORMAL (ENU unit) the SH sky ambient
    /// is evaluated at (the correctly-coloured cone axis). Default up `[0,0,1]`.
    pub bent_normal_enu: [f32; 3],
    /// M3 Cox-Munk glint: the 10 m wind speed (m/s) at this pixel (`sqrt(U10^2+V10^2)`).
    /// Drives the sea-surface slope distribution. `0` = calm. Only used when `is_water`.
    pub wind_speed: f32,
}

impl Default for SurfacePixel {
    /// The neutral surface pixel: off-earth, up-normal, no terrain occlusion, fully
    /// open sky (aperture openness 1, bent normal = up), calm sea. The M3 fields
    /// default to their NO-OP values so `..Default::default()` in a construction that
    /// predates them reproduces the pre-M3 (M5) behaviour exactly.
    fn default() -> Self {
        Self {
            on_earth: false,
            base_srgb: [0.0, 0.0, 0.0],
            normal_enu: [0.0, 0.0, 1.0],
            sun_enu: [0.0, 0.0, 1.0],
            sun_elev_deg: 0.0,
            is_water: false,
            view_dir: [0.0, 0.0, 1.0],
            terrain_horizon_rad: 0.0,
            sky_openness: 1.0,
            bent_normal_enu: [0.0, 0.0, 1.0],
            wind_speed: 0.0,
        }
    }
}

/// The shared sun-gated daytime calibration RAMP for the true-color levers: linearly
/// interpolate from `1.0` at/below [`AERIAL_VEIL_ELEV_LO_DEG`] (the whole twilight/
/// terminator band, kept byte-identical by construction) to `day_value` at/above
/// [`AERIAL_VEIL_ELEV_HI_DEG`] (full daytime), via the same monotone smoothstep. Both
/// [`aerial_veil_scale`] and [`land_day_gain`] are this one family — only `day_value`
/// differs. At the NEUTRAL `day_value = 1.0` it is exactly `1.0` at every elevation
/// (`1.0 - t * 0.0`): the identity no-op that proves each sun-gated lever is a clean
/// parameterization of the shipped radiance path (used by the refinement sanity tests).
#[inline]
pub(crate) fn day_lerp_ramp(sun_elev_deg: f64, day_value: f64) -> f64 {
    let t = atmosphere::smoothstep(
        AERIAL_VEIL_ELEV_LO_DEG,
        AERIAL_VEIL_ELEV_HI_DEG,
        sun_elev_deg,
    );
    1.0 - t * (1.0 - day_value)
}

/// Daytime aerial-perspective veil scale at a sun elevation (deg): `1.0` at/below
/// [`AERIAL_VEIL_ELEV_LO_DEG`] (twilight untouched) ramping to [`AERIAL_VEIL_DAY_SCALE`]
/// at/above [`AERIAL_VEIL_ELEV_HI_DEG`] (full daytime de-haze). Monotone via smoothstep.
#[inline]
pub fn aerial_veil_scale(sun_elev_deg: f64) -> f64 {
    day_lerp_ramp(sun_elev_deg, AERIAL_VEIL_DAY_SCALE)
}

/// LAND daytime brightness gain at a sun elevation (deg): `1.0` at/below
/// [`AERIAL_VEIL_ELEV_LO_DEG`] (twilight untouched — the same gate as the veil, so the
/// M2 twilight band is byte-unchanged) ramping to [`LAND_DAY_GAIN`] at/above
/// [`AERIAL_VEIL_ELEV_HI_DEG`] (full daytime lift). Monotone via smoothstep. See
/// [`LAND_DAY_GAIN`]. Land-only (the caller gates on `!is_water`).
#[inline]
pub fn land_day_gain(sun_elev_deg: f64) -> f64 {
    day_lerp_ramp(sun_elev_deg, LAND_DAY_GAIN)
}

/// GROUND LIFT gain at a sun elevation (deg): `1.0` at/below [`AERIAL_VEIL_ELEV_LO_DEG`]
/// (twilight untouched — the SAME gate as the veil / land gain, so the M2 twilight band
/// is byte-unchanged) ramping to `ground_lift` at/above [`AERIAL_VEIL_ELEV_HI_DEG`] (full
/// daytime lift). Monotone via smoothstep. See [`GROUND_DAY_LIFT`]. `ground_lift` is the
/// baked constant unless the `render_frame` `ground-gain=` knob overrides it; the neutral
/// `ground_lift = 1.0` is `1.0` at every elevation (an exact no-op). Applies to land AND
/// water (the whole surface radiance).
#[inline]
pub fn ground_day_lift(sun_elev_deg: f64, ground_lift: f64) -> f64 {
    day_lerp_ramp(sun_elev_deg, ground_lift)
}

/// The EFFECTIVE cloud shadow seen by the DIFFUSE direct-sun terms (land + water body):
/// `f + (1 - f) * shadow` with `f = CLOUD_SHADOW_FLOOR * smoothstep(LO, HI, sun_elev)`
/// — see [`CLOUD_SHADOW_FLOOR`]. Sun-gated on the shared day ramp, so at/below
/// [`AERIAL_VEIL_ELEV_LO_DEG`] it is exactly the input `shadow` (twilight
/// byte-identical), and `shadow = 1` maps to exactly `1` at every elevation (unshadowed
/// pixels byte-identical). Monotone in `shadow`. The SPECULAR glint keeps the RAW
/// shadow (a mirror image of an occluded solar disk is gone; the floor models diffuse
/// cloud-scattered fill, which has no specular component).
#[inline]
pub fn effective_cloud_shadow(shadow: f64, sun_elev_deg: f64) -> f64 {
    let s = shadow.clamp(0.0, 1.0);
    let f = CLOUD_SHADOW_FLOOR.clamp(0.0, 1.0)
        * atmosphere::smoothstep(
            AERIAL_VEIL_ELEV_LO_DEG,
            AERIAL_VEIL_ELEV_HI_DEG,
            sun_elev_deg,
        );
    f + (1.0 - f) * s
}

/// The PHYSICAL reflectance ceiling the bounded highlight shoulder maps to display
/// white (see [`soft_clip_highlight`]): `x_max = exposure * RHO_HIGHLIGHT_MAX` is the
/// largest exposure-applied reflectance the shoulder resolves — everything at/above it
/// pins to exactly `1.0`. Value `1.05`: measured low-sun composited peaks reach
/// `rho ~ 1.065` (slightly above), so the very brightest ~1% of a frame honestly
/// saturates to white (a real ABI bright top does too) while the whole `[knee, x_max]`
/// band below keeps a NONZERO display slope. This is what makes the shoulder
/// EXPOSURE-AWARE: the unbounded Reinhard wasted the display range asymptoting toward
/// 1.0 for inputs that can never occur, crushing the real cloud band (rho 0.6..0.9 at
/// exposure 1.6 collapsed to a 0.033 display delta — the "white square").
pub const RHO_HIGHLIGHT_MAX: f64 = 1.05;

/// BOUNDED HIGHLIGHT SOFT-CLIP of one display value `x >= 0` with knee `knee` in
/// `(0, 1]` (see [`CLOUD_SOFTCLIP_KNEE`]) and a FINITE input ceiling `x_max` (see
/// [`RHO_HIGHLIGHT_MAX`]): STRICTLY IDENTITY for `x <= knee`, and above the knee a
/// smooth bounded Mobius shoulder mapping `[knee, x_max] -> [knee, 1.0]`:
///
/// `y = knee + span * a / (a + span * (w - a) / w)`, `a = x - knee`, `span = 1 - knee`,
/// `w = x_max - knee`.
///
/// Properties (all tested): C1-continuous at the knee (slope exactly 1 from both
/// sides), strictly monotone increasing (`y' = span^2 / D^2 > 0`), reaches EXACTLY
/// `1.0` at `x_max` with a NONZERO end slope `(span/w)^2`, and hard-clamps to `1.0`
/// only above `x_max`. As `x_max -> +inf` it reduces EXACTLY to the previous unbounded
/// Reinhard shoulder `knee + span * a / (a + span)` (the regression anchor), so the
/// bounded curve is the same family with the wasted `[y(x_max), 1)` asymptote range
/// reclaimed for real contrast — distinct bright cloud tops land on distinct display
/// values instead of collapsing toward a flat white square.
///
/// At the neutral `knee = 1.0` it is the plain hard clamp (identity below 1, `1.0`
/// above), reproducing the pre-soft-clip output. Non-finite / non-positive knee falls
/// back to `1.0` (the clamp); an `x_max` at/below the knee degenerates to the clamp.
#[inline]
pub fn soft_clip_highlight(x: f64, knee: f64, x_max: f64) -> f64 {
    let k = if knee.is_finite() && knee > 0.0 {
        knee.min(1.0)
    } else {
        1.0
    };
    if x <= k {
        return x;
    }
    let a = x - k;
    let span = 1.0 - k;
    if span <= 0.0 {
        // knee == 1.0: the plain hard clamp (identity below 1, 1.0 above).
        return 1.0;
    }
    if !x_max.is_finite() {
        // Unbounded limit: the previous Reinhard shoulder (regression anchor).
        return k + span * (a / (a + span));
    }
    let w = x_max - k;
    if w <= 0.0 || x >= x_max {
        // Degenerate bound, or at/above the ceiling: display white.
        return 1.0;
    }
    k + span * a / (a + span * (w - a) / w)
}

/// Luminance-preserving saturation scale of a linear RGB triple: each channel is
/// pushed away from (`s > 1`) or toward (`s < 1`) the Rec.709 luminance. Clamped `>= 0`.
/// The [`LAND_VIBRANCY`] lever's carrier; `s = 1.0` is an exact per-channel no-op (the
/// identity that proves the vibrancy calibration is a clean parameterization — and the
/// shipping caller additionally SKIPS it entirely when `LAND_VIBRANCY == 1.0`).
#[inline]
pub(crate) fn scale_saturation(c: [f64; 3], s: f64) -> [f64; 3] {
    let y = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    [
        (y + (c[0] - y) * s).max(0.0),
        (y + (c[1] - y) * s).max(0.0),
        (y + (c[2] - y) * s).max(0.0),
    ]
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
fn norm3(a: [f64; 3]) -> [f64; 3] {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if l > 0.0 {
        [a[0] / l, a[1] / l, a[2] / l]
    } else {
        a
    }
}

/// Reflect `incident` about a unit `normal`: `incident - 2 (incident . n) n`.
#[inline]
fn reflect(incident: [f64; 3], normal: [f64; 3]) -> [f64; 3] {
    let d = 2.0 * dot3(incident, normal);
    [
        incident[0] - d * normal[0],
        incident[1] - d * normal[1],
        incident[2] - d * normal[2],
    ]
}

/// Convert an internal reflectance factor `rho` (per band) to display `[0,1]` via
/// the selected output transform (twin of the WGSL `output_transform`). For the ABI
/// product path this is `stretch(desaturate(rho))` — the M2 twilight-pass display
/// transform (highlight desaturation on the reflectance vector, then the toe-lifted
/// per-channel sqrt stretch; see `atmosphere::desaturate_highlights` /
/// `atmosphere::abi_reflectance_stretch`). The debug path stays a plain sRGB gamma.
fn apply_output_transform(
    rho: [f64; 3],
    transform: OutputTransform,
    softclip_knee: f64,
    softclip_max: f64,
) -> [f32; 3] {
    match transform {
        OutputTransform::AbiReflectance => {
            // Highlight desaturation (M2 twilight pass), THEN the bounded highlight
            // soft-clip (bright tops keep structure; the desaturate-then-shoulder ORDER
            // is load-bearing — swapping it shifts the -2 deg amber anvils), THEN the
            // toe-lifted sqrt stretch. The soft-clip is strictly identity below its knee,
            // so the desaturated daytime/twilight below the knee is byte-unchanged.
            let desat = atmosphere::desaturate_highlights(rho);
            let mut out = [0.0f32; 3];
            for c in 0..3 {
                let clipped = soft_clip_highlight(desat[c], softclip_knee, softclip_max);
                out[c] = atmosphere::abi_reflectance_stretch(clipped) as f32;
            }
            out
        }
        OutputTransform::DebugSrgb => {
            let mut out = [0.0f32; 3];
            for c in 0..3 {
                out[c] = linear_to_srgb(rho[c].clamp(0.0, 1.0) as f32);
            }
            out
        }
    }
}

/// Composite the surface radiance with the aerial-perspective (Rayleigh) in-scatter,
/// per band: `l_toa = l_surf * transmittance + veil * inscatter`. `veil` is the daytime
/// veil-reduction scale ([`aerial_veil_scale`]); the surface TRANSMITTANCE is never
/// scaled, only the additive haze — exactly as a Rayleigh atmospheric correction. At the
/// NEUTRAL `veil = 1.0` this is the raw physical composite `l_surf * transmittance +
/// inscatter` (the pre-refinement output), so the veil lever is a clean parameterization
/// of the shipped path (the identity used by the refinement sanity tests).
#[inline]
pub(crate) fn combine_aerial_veil(
    l_surf: [f64; 3],
    transmittance: [f64; 3],
    inscatter: [f64; 3],
    veil: f64,
) -> [f64; 3] {
    [
        l_surf[0] * transmittance[0] + veil * inscatter[0],
        l_surf[1] * transmittance[1] + veil * inscatter[1],
        l_surf[2] * transmittance[2] + veil * inscatter[2],
    ]
}

/// The top-of-atmosphere linear radiance of one surface (or limb) pixel, before the
/// output transform. `None` means space beyond the atmosphere (transparent). This is
/// the shared radiance path of [`shade_surface`] and the M4 cloud composite
/// ([`crate::clouds`]) — both consume the SAME linear radiance so a cloud can be laid
/// over the exact surface the atmosphere pass produces (twin of `surface_toa` in the
/// WGSL). `cloud_shadow` in `[0,1]` (1 = unshadowed) darkens the direct sun term for
/// cloud shadows on the ground (M4 sun-OD-map consumer (a)); pass `1.0` for M2.
/// `ground_lift` is the GROUND LIFT daytime brightness gain (see [`GROUND_DAY_LIFT`]) —
/// sun-gated and applied to the whole surface radiance (land AND water); pass `1.0` for
/// the neutral no-op (the pre-lift surface).
pub fn surface_toa_radiance(
    ctx: &FrameContext,
    px: &SurfacePixel,
    cloud_shadow: f64,
    ground_lift: f64,
) -> Option<[f64; 3]> {
    let cam = ctx.cam.camera;
    let view = px.view_dir;
    let sun = ctx.sun_ecef;
    let e_sun = SOLAR_IRRADIANCE_RGB;

    if !px.on_earth {
        // Off-earth: limb if the ray grazes the atmosphere shell, else space.
        let (t_enter, t_exit) = atmosphere::ray_atmosphere_segment(cam, view)?;
        let p_start = madd3(cam, view, t_enter);
        let sc = atmosphere::integrate_scattered_luminance(
            ctx.luts,
            ctx.params,
            p_start,
            view,
            sun,
            e_sun,
            t_exit - t_enter,
            ctx.raymarch_steps,
            true,
        );
        return Some(sc.inscatter);
    }

    // On-earth surface pixel.
    let scale = if px.is_water { ctx.water_scale } else { 1.0 };
    let base = if ctx.bm_present {
        px.base_srgb
    } else {
        [
            ctx.flat_albedo_srgb as f32,
            ctx.flat_albedo_srgb as f32,
            ctx.flat_albedo_srgb as f32,
        ]
    };
    let mut albedo = [
        srgb_to_linear(base[0]) as f64 * scale,
        srgb_to_linear(base[1]) as f64 * scale,
        srgb_to_linear(base[2]) as f64 * scale,
    ];
    // True-color land vibrancy (refinement pass): boost the LAND albedo saturation so
    // vegetation reads vivid green. Water is excluded (dark ocean must stay dark).
    if !px.is_water && LAND_VIBRANCY != 1.0 {
        albedo = scale_saturation(albedo, LAND_VIBRANCY);
    }

    let pi = std::f64::consts::PI;
    let sun_enu = [
        px.sun_enu[0] as f64,
        px.sun_enu[1] as f64,
        px.sun_enu[2] as f64,
    ];
    let elev_rad = px.sun_elev_deg as f64 * pi / 180.0;
    // Penumbral terrain cast shadow (M3): the direct term sees the fraction of the
    // finite solar disk above the LOCAL terrain horizon at the sun's azimuth
    // (`terrain_shadow_fraction`). `terrain_horizon = 0` reproduces the M2 astronomical
    // disk, so flat terrain is a no-op (no double count). The HGT slope N.L (below)
    // still applies — both terrain effects compose (design section 6).
    let terrain_horizon = px.terrain_horizon_rad.max(0.0) as f64;
    let disk = horizon::terrain_shadow_fraction(elev_rad, terrain_horizon);
    // Sun transmittance at the surface, evaluated at max(elev, 0) so the finite-disk
    // crossing stays smooth (the disk fraction handles the terminator, not a hard mu).
    let mu_sun = (px.sun_elev_deg.max(0.0) as f64 * pi / 180.0).sin();
    let t_sun = atmosphere::sample_transmittance(
        &ctx.luts.transmittance,
        atmosphere::R_GROUND_M + 1.0,
        mu_sun,
    );
    // Terrain ambient aperture (M3 — completes M5's SH-2 sky ambient). M5 evaluated the
    // SH sky irradiance over the FULL upper hemisphere at the slope normal; M3 evaluates
    // it at the visible-sky BENT NORMAL and scales it by the OPENNESS fraction (the
    // horizon-occluded aperture, Oat & Sander 2007). A valley/pocket gets less (and,
    // via the bent normal, correctly-coloured) sky fill than a ridgetop. Flat/open
    // ground (openness 1, bent = up) reproduces the M5 value exactly.
    let openness = (px.sky_openness as f64).clamp(0.0, 1.0);
    let bent = norm3([
        px.bent_normal_enu[0] as f64,
        px.bent_normal_enu[1] as f64,
        px.bent_normal_enu[2] as f64,
    ]);
    let e_amb_dir = ctx
        .sky_sh
        .irradiance(px.sun_elev_deg as f64, [0.0, 0.0, 1.0], sun_enu, bent);
    let e_ambient = [
        e_amb_dir[0] * openness,
        e_amb_dir[1] * openness,
        e_amb_dir[2] * openness,
    ];
    // Raw sun-OD cloud shadow (the specular glint consumer) and the FLOORED effective
    // shadow the diffuse direct terms see (cloud-scattered fill; see CLOUD_SHADOW_FLOOR).
    let shadow_raw = cloud_shadow.clamp(0.0, 1.0);
    let shadow = effective_cloud_shadow(shadow_raw, px.sun_elev_deg as f64);

    // The atmosphere shell segment (reused for the water surface-up and the aerial
    // march). For an on-earth pixel `t_exit` is the ground intersection.
    let seg = atmosphere::ray_atmosphere_segment(cam, view);

    let mut l_surf = [0.0; 3];
    if px.is_water {
        // Cox-Munk wind-ruffled SUN GLINT + Fresnel SKY REFLECTION (M3), replacing the
        // M1 flat dark water (design section 5). Geometry in ECEF using the water
        // point's local up (from the ground intersection); the glint images the
        // disk-averaged solar disk (LIMB_DARKENING_DISK_AVG, so it is never an
        // infinitesimal spike — its angular extent comes from the Cox-Munk slope PDF,
        // widening with wind), attenuated to the surface (t_sun) through the finite-disk
        // fraction (disk) and any cloud/terrain shadow.
        let surf_up = seg
            .map(|(_, t_ground)| norm3(madd3(cam, view, t_ground)))
            .unwrap_or([0.0, 0.0, 1.0]);
        let to_cam = [-view[0], -view[1], -view[2]];
        // GLINT_MSS_SCALE (refinement pass, round 2): narrow the Cox-Munk slope
        // variance so the specular core is tighter/brighter (a distinct glitter streak,
        // physically the calmer-sea limit). GLINT_STRENGTH then lifts the peak brightness.
        let mss = optics::cox_munk_mean_square_slope(px.wind_speed as f64) * GLINT_MSS_SCALE;
        let glint_rho =
            optics::cox_munk_glint_reflectance(sun, to_cam, surf_up, mss) * GLINT_STRENGTH;
        // The specular glint sees the RAW shadow (an occluded solar disk has no mirror
        // image; the CLOUD_SHADOW_FLOOR fill is diffuse-only).
        let glint_scale = disk * shadow_raw * atmosphere::LIMB_DARKENING_DISK_AVG;
        // Fresnel specular sky reflection: the sky colour in the mirror-of-view
        // direction, weighted by the Fresnel reflectance at the view zenith. Uses the
        // SH sky radiance so water reflects the (coloured) sky, not just the sun.
        let cos_view = dot3(to_cam, surf_up).max(0.0);
        let f_sky =
            optics::fresnel_reflectance_unpolarized(cos_view, optics::WATER_REFRACTIVE_INDEX_VIS);
        let sky_dir = reflect(view, surf_up); // surface -> sky specular direction
        let l_sky = ctx
            .sky_sh
            .radiance(px.sun_elev_deg as f64, surf_up, sun, sky_dir);
        // WATER DIRECT SUN (WS2): the diffuse water body now sees the same disk-gated,
        // shadow-weighted direct solar irradiance the land branch has — this is what
        // makes cloud shadows exist over the ocean and the ocean brightness respond to
        // the sun elevation. DAY-GATED on the shared ramp (`day_t`), so the whole
        // twilight band (<= LO) is BYTE-IDENTICAL to the locked M2 look; and the
        // water-body albedo is simultaneously retuned DOWN toward
        // [`WATER_ALBEDO_DAY_SCALE`] on the SAME gate so the owner-approved dark-ocean/
        // distinct-glint contrast holds (see the constant's doc).
        let day_t = atmosphere::smoothstep(
            AERIAL_VEIL_ELEV_LO_DEG,
            AERIAL_VEIL_ELEV_HI_DEG,
            px.sun_elev_deg as f64,
        );
        // Effective day albedo rescale relative to the already-applied ctx.water_scale
        // (the twilight anchor): 1.0 at/below LO -> DAY_SCALE/water_scale at/above HI.
        let scale_ratio = if ctx.water_scale > 0.0 {
            1.0 + day_t * (WATER_ALBEDO_DAY_SCALE / ctx.water_scale - 1.0)
        } else {
            1.0
        };
        let ndotl = dot(px.normal_enu, px.sun_enu).max(0.0);
        for c in 0..3 {
            let l_glint = glint_rho * e_sun[c] / pi * t_sun[c] * glint_scale;
            let e_direct = e_sun[c]
                * t_sun[c]
                * disk
                * ndotl
                * atmosphere::LIMB_DARKENING_DISK_AVG
                * shadow
                * day_t;
            // Skylight- AND (at day) sunlight-lit water body (the Blue Marble water
            // texel x the day-gated water scale) + the sun glint + the Fresnel sky
            // reflection. At day_t = 0 this is bit-for-bit the pre-WS2 expression.
            l_surf[c] = albedo[c] * scale_ratio / pi * (e_direct + e_ambient[c])
                + l_glint
                + f_sky * l_sky[c];
        }
    } else {
        // Land: Lambertian direct sun (HGT slope N.L, penumbral-shadowed disk) + the
        // aperture-occluded SH ambient. Snow (if any) is already blended into `albedo`.
        let ndotl = dot(px.normal_enu, px.sun_enu).max(0.0);
        for c in 0..3 {
            let e_direct =
                e_sun[c] * t_sun[c] * disk * ndotl * atmosphere::LIMB_DARKENING_DISK_AVG * shadow;
            l_surf[c] = albedo[c] / pi * (e_direct + e_ambient[c]);
        }
    }

    // LAND daytime brightness lift (refinement pass, round 2): a modest ground-only
    // gain on the surface reflectance at high sun (sun-gated on the veil ramp, so
    // twilight is byte-unchanged; water excluded). Applied to `l_surf` BEFORE the
    // aerial-perspective veil, so only the ground signal brightens (not the additive
    // haze). A true-color display gain, distinct from the global exposure. See
    // [`LAND_DAY_GAIN`] / [`land_day_gain`].
    if !px.is_water && LAND_DAY_GAIN != 1.0 {
        let g = land_day_gain(px.sun_elev_deg as f64);
        for v in &mut l_surf {
            *v *= g;
        }
    }

    // GROUND LIFT (top-down/basemap appearance pass): a sun-gated daytime brightness lift
    // on the WHOLE surface radiance — land AND water — toward real-GOES ground levels (the
    // reported ground was too dark). Applied BEFORE the aerial-perspective veil (only the
    // ground signal brightens, not the additive haze) and BEFORE the cloud composite (the
    // cloud radiance is not lifted). Sun-gated on the veil ramp, so at/below the twilight
    // band it is exactly `1.0` — twilight is byte-unchanged. `ground_lift = 1.0` = no-op.
    if ground_lift != 1.0 {
        let g = ground_day_lift(px.sun_elev_deg as f64, ground_lift);
        for v in &mut l_surf {
            *v *= g;
        }
    }

    // Aerial perspective: raymarch the shell from atmosphere entry to the ground.
    let mut l_toa = l_surf;
    if let Some((t_enter, t_ground)) = seg {
        let p_start = madd3(cam, view, t_enter);
        let sc = atmosphere::integrate_scattered_luminance(
            ctx.luts,
            ctx.params,
            p_start,
            view,
            sun,
            e_sun,
            t_ground - t_enter,
            ctx.raymarch_steps,
            true,
        );
        // True-color veil reduction (refinement pass): scale down the additive daytime
        // in-scatter haze laid over the surface (a Rayleigh correction; twilight is
        // untouched because the scale is 1.0 below AERIAL_VEIL_ELEV_LO_DEG). The surface
        // transmittance is left intact. Off-earth limb in-scatter (above) is never scaled.
        let veil = aerial_veil_scale(px.sun_elev_deg as f64);
        l_toa = combine_aerial_veil(l_surf, sc.transmittance, sc.inscatter, veil);
    }
    Some(l_toa)
}

/// Convert a top-of-atmosphere linear radiance to display `[0,1]` rgba (alpha 1),
/// applying the reflectance factor, the `exposure` gain, and the output transform.
/// Shared by [`shade_surface`] and the M4 cloud composite so tonemapping is identical.
///
/// EXPOSURE (the deferred "M2 MAJOR-1 tonemap" decision, now made with the owner in
/// the loop): `exposure` is a DISPLAY-SIDE linear gain, NOT a change to the physics.
/// The reflectance factor `rho = pi * L / E_sun` is the physically-based quantity;
/// exposure multiplies it — `rho' = exposure * rho` — BEFORE the ABI sqrt stretch and
/// BEFORE the `[0,1]` clamp in [`apply_output_transform`]. So the display value is
/// `stretch(clamp(exposure * pi * L / E_sun, 0, 1))`. Placement matters: because the
/// gain is applied before the clamp, `exposure > 1` brightens every sub-clip pixel
/// monotonically and pushes the brightest pixels toward (never past) white; the clamp
/// guarantees the output stays in `[0,1]`, so the `* 255` quantization can never leave
/// `0..255` (the BowEcho store contract). `exposure = 1.0` reproduces the pre-exposure
/// output exactly (the regression anchor). One `exposure` is applied per whole frame,
/// so the surface and the cloud brighten consistently. A non-finite or non-positive
/// `exposure` falls back to `1.0` (never darkens to nothing on a bad input).
pub fn radiance_to_rgba(l_toa: [f64; 3], transform: OutputTransform, exposure: f64) -> [f32; 4] {
    radiance_to_rgba_softclip(l_toa, transform, exposure, CLOUD_SOFTCLIP_KNEE)
}

/// Like [`radiance_to_rgba`] but with an explicit highlight soft-clip knee (see
/// [`CLOUD_SOFTCLIP_KNEE`] / [`soft_clip_highlight`]). [`radiance_to_rgba`] delegates
/// here with the baked default, so the plain surface path and the cloud/top-down RGB
/// paths (which read the per-scene `MarchConfig` knob, overridable by the `render_frame`
/// `cloud-softclip=` knob) all share ONE tonemap. `knee = 1.0` disables the shoulder
/// (the old hard clamp above 1.0).
///
/// EXPOSURE-AWARE SHOULDER BOUND: the shoulder's input ceiling is derived INTERNALLY
/// here as `x_max = gain * RHO_HIGHLIGHT_MAX` — the exposure gain is already applied to
/// the reflectance, so the largest input the shoulder can ever see is the exposure times
/// the physical reflectance ceiling. Deriving it here (the one seam that already
/// receives the exposure) keeps this signature and every caller (clouds.rs, topdown.rs)
/// unchanged.
pub fn radiance_to_rgba_softclip(
    l_toa: [f64; 3],
    transform: OutputTransform,
    exposure: f64,
    softclip_knee: f64,
) -> [f32; 4] {
    let e_sun = SOLAR_IRRADIANCE_RGB;
    let gain = if exposure.is_finite() && exposure > 0.0 {
        exposure
    } else {
        1.0
    };
    let mut rho = [0.0; 3];
    for c in 0..3 {
        rho[c] = gain * std::f64::consts::PI * l_toa[c] / e_sun[c];
    }
    let out = apply_output_transform(rho, transform, softclip_knee, gain * RHO_HIGHLIGHT_MAX);
    [out[0], out[1], out[2], 1.0]
}

/// Convert a top-of-atmosphere linear radiance to the RAW per-channel REFLECTANCE
/// FACTOR `rho = pi * L / E_sun`, clamped to `[0, 1]` — the pre-tonemap quantity the
/// Python binding's `render_visible_bands` returns for building custom RGB / operating
/// on bands. This is the SAME `rho` [`radiance_to_rgba`] computes internally, but WITHOUT
/// the display exposure gain and WITHOUT the ABI sqrt stretch / highlight desaturation:
/// a linear reflectance in `[0, 1]` (real ABI visible bands are reflectance factors in
/// this range; a bright sunlit cloud top saturates at 1.0). The RGB product path is
/// unchanged — this is an additional, independent conversion for the raw-bands product.
pub fn reflectance_from_radiance(l_toa: [f64; 3]) -> [f32; 3] {
    let e_sun = SOLAR_IRRADIANCE_RGB;
    let mut rho = [0.0f32; 3];
    for (c, out) in rho.iter_mut().enumerate() {
        *out = (std::f64::consts::PI * l_toa[c] / e_sun[c]).clamp(0.0, 1.0) as f32;
    }
    rho
}

/// Shade one surface pixel with the M2 atmosphere. Returns display-ready `rgba` in
/// `[0,1]`; alpha is `1.0` on earth AND on the limb (opaque), `0.0` only for space
/// beyond the atmosphere (the off-earth marker the store writer turns into a
/// transparent/NaN plane value). Twin of `fs_main` in `surface.wgsl`.
pub fn shade_surface(ctx: &FrameContext, px: &SurfacePixel) -> [f32; 4] {
    // The clouds-off surface path bakes the shipped GROUND LIFT default (the composite
    // paths pass their per-scene `MarchConfig` value); the tonemap bakes the soft-clip.
    match surface_toa_radiance(ctx, px, 1.0, GROUND_DAY_LIFT) {
        None => [0.0, 0.0, 0.0, 0.0],
        Some(l_toa) => radiance_to_rgba(l_toa, ctx.output_transform, ctx.exposure),
    }
}

/// Terrain normals in the local ENU basis from an `HGT` plane (row-major
/// `[ny][nx]`, meters MSL) with grid spacing `dx`/`dy` (m). M1 approximates
/// grid-x = east, grid-y = north (ignoring the small Lambert grid convergence
/// away from `STAND_LON` — documented; horizon-map cast shadows are M3). Flat or
/// missing terrain yields the up vector `(0, 0, 1)`. Returned packed as
/// `nx * ny` unit `[e, n, u]` triples in WRF row order (row 0 = south).
pub fn normals_from_hgt(hgt: &[f32], nx: usize, ny: usize, dx: f64, dy: f64) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0f32, 0.0, 1.0]; nx * ny];
    if hgt.len() != nx * ny || nx < 2 || ny < 2 || dx <= 0.0 || dy <= 0.0 {
        return out;
    }
    for j in 0..ny {
        for i in 0..nx {
            let (ie, iw) = (i.min(nx - 2) + 1, i.max(1) - 1);
            let (jn, js) = (j.min(ny - 2) + 1, j.max(1) - 1);
            let span_x = (ie - iw) as f64 * dx;
            let span_y = (jn - js) as f64 * dy;
            let dhdx = (hgt[j * nx + ie] as f64 - hgt[j * nx + iw] as f64) / span_x;
            let dhdy = (hgt[jn * nx + i] as f64 - hgt[js * nx + i] as f64) / span_y;
            // Upward normal of the height field z = H(x, y): (-dH/dx, -dH/dy, 1).
            let mut n = [-dhdx, -dhdy, 1.0];
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 0.0 {
                n = [n[0] / len, n[1] / len, n[2] / len];
            }
            out[j * nx + i] = [n[0] as f32, n[1] as f32, n[2] as f32];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_round_trips_within_tolerance() {
        for c in [0.0f32, 0.02, 0.1, 0.3, 0.5, 0.8, 1.0] {
            let back = linear_to_srgb(srgb_to_linear(c));
            assert!((back - c).abs() < 1e-4, "{c} -> {back}");
        }
        // Known anchors.
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-6);
        assert!(srgb_to_linear(0.0).abs() < 1e-9);
    }

    #[test]
    fn off_earth_is_transparent_black() {
        let out = shade_pixel(&ShadeInputs {
            on_earth: false,
            base_srgb: [1.0, 1.0, 1.0],
            normal_enu: [0.0, 0.0, 1.0],
            sun_dir_enu: [0.0, 0.0, 1.0],
            sun_elev_deg: 45.0,
            is_water: false,
        });
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn flat_white_land_with_overhead_sun_is_full_bright() {
        let out = shade_pixel(&ShadeInputs {
            on_earth: true,
            base_srgb: [1.0, 1.0, 1.0],
            normal_enu: [0.0, 0.0, 1.0],
            sun_dir_enu: [0.0, 0.0, 1.0],
            sun_elev_deg: 90.0,
            is_water: false,
        });
        // Overhead sun, white albedo, N.L = 1 -> full white, opaque.
        assert!((out[0] - 1.0).abs() < 1e-4);
        assert!((out[3] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn night_is_dark_even_facing_the_sun() {
        let out = shade_pixel(&ShadeInputs {
            on_earth: true,
            base_srgb: [1.0, 1.0, 1.0],
            normal_enu: [0.0, 0.0, 1.0],
            sun_dir_enu: [0.0, 0.0, 1.0],
            sun_elev_deg: -5.0, // below horizon
            is_water: false,
        });
        assert!(out[0] < 1e-4, "night should be dark, got {}", out[0]);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn water_is_darker_than_land_for_the_same_albedo() {
        let base = [0.6f32, 0.6, 0.6];
        let common = |is_water| ShadeInputs {
            on_earth: true,
            base_srgb: base,
            normal_enu: [0.0, 0.0, 1.0],
            sun_dir_enu: [0.0, 0.0, 1.0],
            sun_elev_deg: 60.0,
            is_water,
        };
        let land = shade_pixel(&common(false));
        let water = shade_pixel(&common(true));
        assert!(water[0] < land[0], "water {} !< land {}", water[0], land[0]);
    }

    #[test]
    fn a_lit_slope_is_brighter_than_a_shadowed_slope() {
        // Sun low in the east; an east-facing slope catches more light than a
        // west-facing slope.
        let sun = [0.906f32, 0.0, 0.423]; // az 90 (east), el ~25 deg
        let east_face = shade_pixel(&ShadeInputs {
            on_earth: true,
            base_srgb: [0.7, 0.7, 0.7],
            normal_enu: [0.5, 0.0, 0.866],
            sun_dir_enu: sun,
            sun_elev_deg: 25.0,
            is_water: false,
        });
        let west_face = shade_pixel(&ShadeInputs {
            on_earth: true,
            base_srgb: [0.7, 0.7, 0.7],
            normal_enu: [-0.5, 0.0, 0.866],
            sun_dir_enu: sun,
            sun_elev_deg: 25.0,
            is_water: false,
        });
        assert!(east_face[0] > west_face[0]);
    }

    #[test]
    fn normals_of_flat_terrain_point_up() {
        let hgt = vec![100.0f32; 4 * 4];
        let n = normals_from_hgt(&hgt, 4, 4, 3000.0, 3000.0);
        for v in n {
            assert!(v[0].abs() < 1e-6 && v[1].abs() < 1e-6 && (v[2] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn normals_lean_away_from_a_ridge() {
        // A slope rising toward the east: H increases with i. dH/dx > 0 -> the
        // normal's east component is negative (leans west).
        let (nx, ny) = (5usize, 3usize);
        let mut hgt = vec![0.0f32; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                hgt[j * nx + i] = (i as f32) * 300.0; // 300 m rise per cell east
            }
        }
        let n = normals_from_hgt(&hgt, nx, ny, 3000.0, 3000.0);
        let mid = n[nx + 2];
        assert!(mid[0] < 0.0, "east-rising slope leans west, got {mid:?}");
        assert!(mid[2] > 0.0);
    }

    // ── M2 shade_surface (twin of surface.wgsl) ──
    use crate::camera::{GeoCamera, SatellitePreset};

    /// Process-global optics state, built once and shared by every render test
    /// (the LUT build is the expensive part; a `OnceLock` amortises it).
    fn shared_optics() -> &'static (AtmosphereParams, AtmosphereLuts, SkyShTable) {
        static CACHE: std::sync::OnceLock<(AtmosphereParams, AtmosphereLuts, SkyShTable)> =
            std::sync::OnceLock::new();
        CACHE.get_or_init(|| {
            let params = AtmosphereParams::default();
            let luts = AtmosphereLuts::build(&params);
            let sky_sh = SkyShTable::build(&luts, &params, 16);
            (params, luts, sky_sh)
        })
    }

    fn nadir_surface_pixel(sun_elev_deg: f32) -> (FrameContext<'static>, [f64; 3]) {
        let (params, luts, sky_sh) = shared_optics();
        let cam = CameraGeometry::from_sub_lon(-75.2);
        let e = (sun_elev_deg as f64).to_radians();
        let sun_enu = [0.0, e.cos(), e.sin()]; // azimuth north; elevation e
        let sun_ecef = atmosphere::sun_enu_to_ecef(sun_enu, 0.0, -75.2);
        let ctx = FrameContext {
            luts,
            params,
            sky_sh,
            cam,
            sun_ecef,
            output_transform: OutputTransform::AbiReflectance,
            bm_present: false,
            water_scale: WATER_ALBEDO_SCALE as f64,
            flat_albedo_srgb: 0.5,
            raymarch_steps: 16,
            // exposure = 1.0 in the shading tests: they are the regression anchor for
            // the pre-exposure output (a moderated default is a display choice made in
            // the studio / CLI, not baked into the physics tests).
            exposure: 1.0,
        };
        (ctx, sun_enu)
    }

    fn nadir_view() -> [f64; 3] {
        let cam = CameraGeometry::from_sub_lon(-75.2);
        let geocam = GeoCamera::new(SatellitePreset::GoesEast);
        let (sx, sy) = geocam.forward(0.0, -75.2).unwrap();
        cam.view_dir(sx, sy)
    }

    fn brightness(rgba: [f32; 4]) -> f32 {
        rgba[0] + rgba[1] + rgba[2]
    }

    #[test]
    fn shade_surface_day_is_bright_night_is_dark() {
        let view = nadir_view();
        let (day_ctx, day_sun) = nadir_surface_pixel(90.0);
        let day = shade_surface(
            &day_ctx,
            &SurfacePixel {
                on_earth: true,
                base_srgb: [0.5, 0.5, 0.5],
                normal_enu: [0.0, 0.0, 1.0],
                sun_enu: [day_sun[0] as f32, day_sun[1] as f32, day_sun[2] as f32],
                sun_elev_deg: 90.0,
                is_water: false,
                view_dir: view,
                ..Default::default()
            },
        );
        assert_eq!(day[3], 1.0, "earth is opaque");
        let day_b = brightness(day);
        assert!(
            day_b > 0.3,
            "overhead-sun day should be bright, got {day_b}"
        );

        let (night_ctx, night_sun) = nadir_surface_pixel(-18.0);
        let night = shade_surface(
            &night_ctx,
            &SurfacePixel {
                on_earth: true,
                base_srgb: [0.5, 0.5, 0.5],
                normal_enu: [0.0, 0.0, 1.0],
                sun_enu: [
                    night_sun[0] as f32,
                    night_sun[1] as f32,
                    night_sun[2] as f32,
                ],
                sun_elev_deg: -18.0,
                is_water: false,
                view_dir: view,
                ..Default::default()
            },
        );
        assert_eq!(night[3], 1.0, "night earth is still opaque (not space)");
        let night_b = brightness(night);
        assert!(
            night_b < day_b * 0.4,
            "astronomical night should be far darker than day: night {night_b} vs day {day_b}"
        );
    }

    #[test]
    fn shade_surface_terminator_is_monotone() {
        // Brightness falls monotonically as the sun sinks through the terminator
        // (finite disk + ambient both ramp down; no step).
        let view = nadir_view();
        let mut prev = f32::INFINITY;
        for &elev in &[40.0f32, 20.0, 10.0, 5.0, 2.0, 0.0, -2.0, -5.0] {
            let (ctx, sun) = nadir_surface_pixel(elev);
            let out = shade_surface(
                &ctx,
                &SurfacePixel {
                    on_earth: true,
                    base_srgb: [0.5, 0.5, 0.5],
                    normal_enu: [0.0, 0.0, 1.0],
                    sun_enu: [sun[0] as f32, sun[1] as f32, sun[2] as f32],
                    sun_elev_deg: elev,
                    is_water: false,
                    view_dir: view,
                    ..Default::default()
                },
            );
            let b = brightness(out);
            assert!(
                b <= prev + 1.0e-3,
                "not monotone at elev {elev}: {b} > {prev}"
            );
            prev = b;
        }
    }

    #[test]
    fn shade_surface_limb_is_lit_space_is_transparent() {
        let (ctx, _sun) = nadir_surface_pixel(10.0);
        // A ray in the thin band between the ground tangent (~0.1517 rad) and the
        // atmosphere-top tangent (~0.1541 rad) grazes the shell -> the LIMB.
        let limb_view = ctx.cam.view_dir(0.1530, 0.0);
        let limb = shade_surface(
            &ctx,
            &SurfacePixel {
                on_earth: false,
                base_srgb: [0.0, 0.0, 0.0],
                normal_enu: [0.0, 0.0, 1.0],
                sun_enu: [0.0, 0.0, 1.0],
                sun_elev_deg: 0.0,
                is_water: false,
                view_dir: limb_view,
                ..Default::default()
            },
        );
        assert_eq!(limb[3], 1.0, "limb is opaque (not space)");
        assert!(brightness(limb).is_finite());

        // A ray well past the shell tangent misses the atmosphere entirely -> space.
        let space_view = ctx.cam.view_dir(0.1600, 0.0);
        let space = shade_surface(
            &ctx,
            &SurfacePixel {
                on_earth: false,
                base_srgb: [0.0, 0.0, 0.0],
                normal_enu: [0.0, 0.0, 1.0],
                sun_enu: [0.0, 0.0, 1.0],
                sun_elev_deg: 0.0,
                is_water: false,
                view_dir: space_view,
                ..Default::default()
            },
        );
        assert_eq!(space, [0.0, 0.0, 0.0, 0.0], "beyond the shell is space");
    }

    // ── exposure (display-side gain in radiance_to_rgba) ──

    #[test]
    fn exposure_one_applies_the_display_transform_at_unit_gain() {
        // The regression anchor: at exposure = 1.0 (identity gain) radiance_to_rgba must
        // equal the shipped display transform `stretch(softclip(desaturate(pi*L/E_sun)))`
        // band by band — highlight desaturation on the reflectance VECTOR, then the
        // BOUNDED highlight soft-clip at `x_max = 1.0 * RHO_HIGHLIGHT_MAX` (the
        // exposure-aware bound at unit gain), then the toe-lifted per-channel sqrt (the
        // composition + ordering the render path uses). The soft-clip is identity below
        // its knee, so the sub-knee cases are unchanged and only the bright case bends.
        let e_sun = SOLAR_IRRADIANCE_RGB;
        for l in [
            [0.0, 0.0, 0.0],
            [5.0, 8.0, 12.0],
            [40.0, 30.0, 20.0],
            [180.0, 188.0, 196.0], // rho = pi > x_max -> display white
        ] {
            let got = radiance_to_rgba(l, OutputTransform::AbiReflectance, 1.0);
            let rho = [
                std::f64::consts::PI * l[0] / e_sun[0],
                std::f64::consts::PI * l[1] / e_sun[1],
                std::f64::consts::PI * l[2] / e_sun[2],
            ];
            let desat = atmosphere::desaturate_highlights(rho);
            for c in 0..3 {
                let clipped = soft_clip_highlight(desat[c], CLOUD_SOFTCLIP_KNEE, RHO_HIGHLIGHT_MAX);
                let want = atmosphere::abi_reflectance_stretch(clipped) as f32;
                assert!(
                    (got[c] - want).abs() < 1e-6,
                    "band {c}: {} vs {want}",
                    got[c]
                );
            }
            assert_eq!(got[3], 1.0, "alpha opaque");
        }
    }

    #[test]
    fn exposure_above_one_brightens_sub_clip_pixels_monotonically() {
        // A dim (sub-clip) radiance brightens monotonically as exposure rises, until
        // it saturates at the stretch ceiling (1.0). Never darkens.
        let l = [10.0, 10.0, 10.0]; // rho ~= 0.17 at exposure 1 -> plenty of headroom
        let mut prev = -1.0f32;
        for &ev in &[0.5f32, 1.0, 1.5, 2.0, 3.0, 4.0] {
            let out = radiance_to_rgba(l, OutputTransform::AbiReflectance, ev as f64);
            let b = out[0];
            assert!(
                b + 1e-6 >= prev,
                "exposure {ev}: {b} < prev {prev} (not monotone)"
            );
            prev = b;
        }
        // A concrete brightening between two sub-clip exposures.
        let dim = radiance_to_rgba(l, OutputTransform::AbiReflectance, 1.0)[0];
        let bright = radiance_to_rgba(l, OutputTransform::AbiReflectance, 2.0)[0];
        assert!(
            bright > dim,
            "exposure 2 ({bright}) should exceed exposure 1 ({dim})"
        );
    }

    #[test]
    fn exposure_never_escapes_zero_to_one_after_the_stretch() {
        // No exposure (however large) or radiance can push the display value outside
        // [0,1], so the * 255 quantization stays in 0..255 (the store contract).
        for &ev in &[1.0f64, 2.5, 10.0, 1.0e6] {
            for l in [[0.0, 0.0, 0.0], [50.0, 90.0, 130.0], [1.0e4, 1.0e4, 1.0e4]] {
                for &t in &[OutputTransform::AbiReflectance, OutputTransform::DebugSrgb] {
                    let out = radiance_to_rgba(l, t, ev);
                    for (c, &v) in out[..3].iter().enumerate() {
                        assert!(
                            (0.0..=1.0).contains(&v),
                            "exposure {ev} L {l:?} band {c} = {v} out of [0,1]"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn exposure_preserves_reflectance_ordering() {
        // A brighter scene stays brighter than a dimmer one under the same exposure
        // (a common gain cannot reorder pixels below the clip point).
        let dim = [8.0, 8.0, 8.0];
        let bright = [24.0, 24.0, 24.0];
        for &ev in &[1.0f64, 1.6, 2.5] {
            let d = radiance_to_rgba(dim, OutputTransform::AbiReflectance, ev)[0];
            let b = radiance_to_rgba(bright, OutputTransform::AbiReflectance, ev)[0];
            assert!(b > d, "exposure {ev}: bright {b} !> dim {d}");
        }
    }

    #[test]
    fn exposure_non_finite_or_nonpositive_falls_back_to_one() {
        let l = [12.0, 12.0, 12.0];
        let anchor = radiance_to_rgba(l, OutputTransform::AbiReflectance, 1.0);
        for &bad in &[0.0f64, -3.0, f64::NAN, f64::INFINITY] {
            let out = radiance_to_rgba(l, OutputTransform::AbiReflectance, bad);
            assert_eq!(out, anchor, "bad exposure {bad} should fall back to 1.0");
        }
    }

    #[test]
    fn reflectance_from_radiance_is_the_clamped_pre_tonemap_rho() {
        // The raw-bands conversion (Python render_visible_bands) is rho = pi*L/E_sun with
        // NO exposure and NO stretch, clamped to [0, 1]. It must equal the same rho that
        // radiance_to_rgba computes internally (before the ABI stretch) at exposure 1.0.
        let e_sun = SOLAR_IRRADIANCE_RGB;
        for &target in &[0.0, 0.05, 0.25, 0.6, 0.95] {
            let l = [
                target * e_sun[0] / std::f64::consts::PI,
                target * e_sun[1] / std::f64::consts::PI,
                target * e_sun[2] / std::f64::consts::PI,
            ];
            let refl = reflectance_from_radiance(l);
            for &v in &refl {
                assert!((v as f64 - target).abs() < 1e-6, "{v} != {target}");
                assert!((0.0..=1.0).contains(&v), "reflectance out of [0,1]: {v}");
            }
        }
        // A super-bright radiance (rho > 1) clamps to exactly 1.0 (real ABI band range).
        let bright = [
            2.5 * e_sun[0] / std::f64::consts::PI,
            2.5 * e_sun[1] / std::f64::consts::PI,
            2.5 * e_sun[2] / std::f64::consts::PI,
        ];
        assert_eq!(reflectance_from_radiance(bright), [1.0, 1.0, 1.0]);
        // Non-negative floor: a (numerically) tiny/zero radiance is 0, never negative.
        assert_eq!(reflectance_from_radiance([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
    }

    // ── M2 twilight-pass display transform (toe lift + chroma-gated desaturation) ──

    #[test]
    fn tonemap_bright_reflectance_maps_near_white_unchanged() {
        // DAYTIME PRESERVED: a bright, near-neutral reflectance (a sunlit cloud top) maps
        // to near-white and is unchanged by DESATURATION and by the TOE (it is above the
        // toe knee and below the saturation gate). The baked highlight soft-clip DOES bend
        // it (that is its job — keep structure short of white), so the reference is the
        // soft-clipped-then-stretched value, NOT the plain sqrt; it must stay near-white.
        let e_sun = SOLAR_IRRADIANCE_RGB;
        // rho ~ [0.855, 0.852, 0.850] (bright, ~grey -> chroma < DESAT_SAT_LO).
        let l = [
            0.855 * e_sun[0] / std::f64::consts::PI,
            0.852 * e_sun[1] / std::f64::consts::PI,
            0.850 * e_sun[2] / std::f64::consts::PI,
        ];
        let got = radiance_to_rgba(l, OutputTransform::AbiReflectance, 1.0);
        for c in 0..3 {
            let rho = std::f64::consts::PI * l[c] / e_sun[c];
            // Soft-clip (no desat fires for this near-grey pixel), then the classic sqrt.
            let clipped = soft_clip_highlight(rho, CLOUD_SOFTCLIP_KNEE, RHO_HIGHLIGHT_MAX);
            let plain = atmosphere::abi_reflectance_stretch(clipped) as f32;
            assert!(
                (got[c] - plain).abs() < 1e-6,
                "band {c}: bright near-neutral must be unchanged by desat/toe: {} vs {plain}",
                got[c]
            );
            assert!(
                got[c] > 0.85,
                "band {c} should be near-white, got {}",
                got[c]
            );
        }
    }

    #[test]
    fn tonemap_dim_reflectance_is_toe_lifted() {
        // SHADOW LIFT: a dim below-knee reflectance (twilight) maps BRIGHTER than the
        // plain sqrt would, so the dim blue twilight the model computes becomes visible.
        let e_sun = SOLAR_IRRADIANCE_RGB;
        // rho ~ 0.01 per band (well below REFL_TOE_KNEE = 0.05), ~grey (no desat).
        let l = [
            0.01 * e_sun[0] / std::f64::consts::PI,
            0.01 * e_sun[1] / std::f64::consts::PI,
            0.01 * e_sun[2] / std::f64::consts::PI,
        ];
        let got = radiance_to_rgba(l, OutputTransform::AbiReflectance, 1.0);
        let plain_sqrt = 0.01_f64.sqrt() as f32; // 0.1
        for &v in &got[..3] {
            assert!(
                v > plain_sqrt + 1e-3,
                "dim twilight must be toe-lifted above plain sqrt {plain_sqrt}, got {v}"
            );
            assert!(v < 1.0, "still in range");
        }
    }

    #[test]
    fn tonemap_desaturates_bright_saturated_highlight_only() {
        // CHROMA GATE: a bright, strongly warm (over-saturated) highlight — the reddened
        // low-sun anvil — has its display R/B pulled toward 1 (amber, not saturated
        // orange) relative to the no-desat per-channel stretch; a bright but near-neutral
        // highlight is left unchanged.
        let e_sun = SOLAR_IRRADIANCE_RGB;
        // Warm, bright: rho ~ [1.4, 0.42, 0.24] (R clips, high chroma, high luminance).
        let warm = [
            1.40 * e_sun[0] / std::f64::consts::PI,
            0.42 * e_sun[1] / std::f64::consts::PI,
            0.24 * e_sun[2] / std::f64::consts::PI,
        ];
        let got = radiance_to_rgba(warm, OutputTransform::AbiReflectance, 1.0);
        // No-desat reference: the plain per-channel stretch of the same reflectance.
        let ref_r = atmosphere::abi_reflectance_stretch(1.40) as f32;
        let ref_b = atmosphere::abi_reflectance_stretch(0.24) as f32;
        assert!(
            got[0] / got[2] < ref_r / ref_b - 1e-3,
            "warm highlight R/B {} should drop below the no-desat {}",
            got[0] / got[2],
            ref_r / ref_b
        );
        assert!(
            got[0] / got[2] > 1.0,
            "but must stay warm (amber, not neutral)"
        );

        // Bright but near-neutral: chroma < DESAT_SAT_LO -> NOT desaturated. The baked
        // soft-clip still bends it (it is above the knee), so the reference is the
        // soft-clipped stretch (no desat); the point is that desaturation did not fire.
        let neutral = [
            0.80 * e_sun[0] / std::f64::consts::PI,
            0.78 * e_sun[1] / std::f64::consts::PI,
            0.76 * e_sun[2] / std::f64::consts::PI,
        ];
        let gotn = radiance_to_rgba(neutral, OutputTransform::AbiReflectance, 1.0);
        for c in 0..3 {
            let rho = std::f64::consts::PI * neutral[c] / e_sun[c];
            let clipped = soft_clip_highlight(rho, CLOUD_SOFTCLIP_KNEE, RHO_HIGHLIGHT_MAX);
            let plain = atmosphere::abi_reflectance_stretch(clipped) as f32;
            assert!(
                (gotn[c] - plain).abs() < 1e-6,
                "band {c}: near-neutral bright pixel must not be desaturated"
            );
        }
    }

    // ── M3 surface: SNOWH blend, penumbral terrain shadow, ambient aperture, glint ──

    #[test]
    fn snow_fraction_ramp_is_monotone_and_bounded() {
        // Below the min depth -> 0; at/above the full depth -> 1; monotone in between.
        assert_eq!(snow_fraction(0.0), 0.0);
        assert_eq!(snow_fraction(SNOW_DEPTH_MIN_M), 0.0);
        assert_eq!(snow_fraction(SNOW_DEPTH_FULL_M), 1.0);
        assert_eq!(snow_fraction(1.0), 1.0);
        assert_eq!(snow_fraction(f64::NAN), 0.0);
        let mut prev = -1.0f32;
        for k in 0..=20 {
            let d = SNOW_DEPTH_MIN_M + (SNOW_DEPTH_FULL_M - SNOW_DEPTH_MIN_M) * k as f64 / 20.0;
            let f = snow_fraction(d);
            assert!(
                f >= prev - 1e-7,
                "snow ramp not monotone at {d}: {f} < {prev}"
            );
            assert!((0.0..=1.0).contains(&f));
            prev = f;
        }
    }

    #[test]
    fn blend_snow_lerps_toward_the_snow_albedo() {
        let base = [0.10f32, 0.12, 0.08]; // dark vegetation
        assert_eq!(blend_snow(base, 0.0), base);
        assert_eq!(blend_snow(base, 1.0), SNOW_ALBEDO_SRGB);
        let half = blend_snow(base, 0.5);
        for c in 0..3 {
            // Halfway is the mean, and brighter than the dark base.
            assert!((half[c] - 0.5 * (base[c] + SNOW_ALBEDO_SRGB[c])).abs() < 1e-6);
            assert!(half[c] > base[c]);
        }
    }

    #[test]
    fn terrain_shadow_darkens_the_direct_sun_term() {
        // A lit flat land pixel: a terrain horizon above the sun (a ridge occluding it)
        // must render darker than the same pixel with no terrain occlusion (the direct
        // term is penumbrally killed; the ambient is unchanged).
        let view = nadir_view();
        let (ctx, sun) = nadir_surface_pixel(30.0);
        let base = SurfacePixel {
            on_earth: true,
            base_srgb: [0.5, 0.5, 0.5],
            normal_enu: [0.0, 0.0, 1.0],
            sun_enu: [sun[0] as f32, sun[1] as f32, sun[2] as f32],
            sun_elev_deg: 30.0,
            is_water: false,
            view_dir: view,
            ..Default::default()
        };
        let lit = brightness(shade_surface(&ctx, &base));
        let shadowed = brightness(shade_surface(
            &ctx,
            &SurfacePixel {
                terrain_horizon_rad: 40.0f32.to_radians(), // ridge well above the 30 deg sun
                ..base
            },
        ));
        assert!(
            shadowed < lit,
            "a terrain-shadowed pixel {shadowed} should be darker than the lit {lit}"
        );
        // A ridge BELOW the sun (10 deg vs 30 deg sun) leaves the disk fully clear -> lit.
        let clear = brightness(shade_surface(
            &ctx,
            &SurfacePixel {
                terrain_horizon_rad: 10.0f32.to_radians(),
                ..base
            },
        ));
        assert!(
            (clear - lit).abs() < 1e-4,
            "a low ridge should not shadow: {clear} vs {lit}"
        );
    }

    #[test]
    fn ambient_aperture_reduces_ambient_in_a_pocket() {
        // At a low sun (ambient-dominated), an occluded pocket (low openness) gets less
        // sky fill than open ground (openness 1) — the horizon-occluded terrain ambient.
        let view = nadir_view();
        let (ctx, sun) = nadir_surface_pixel(5.0);
        let open = SurfacePixel {
            on_earth: true,
            base_srgb: [0.5, 0.5, 0.5],
            normal_enu: [0.0, 0.0, 1.0],
            sun_enu: [sun[0] as f32, sun[1] as f32, sun[2] as f32],
            sun_elev_deg: 5.0,
            is_water: false,
            view_dir: view,
            ..Default::default() // openness 1, bent = up
        };
        let pocket = SurfacePixel {
            sky_openness: 0.2,
            ..open
        };
        let b_open = brightness(shade_surface(&ctx, &open));
        let b_pocket = brightness(shade_surface(&ctx, &pocket));
        assert!(
            b_pocket < b_open,
            "an occluded pocket {b_pocket} should get less ambient than open ground {b_open}"
        );
    }

    #[test]
    fn water_glint_adds_specular_brightness() {
        // Overhead sun + nadir view = the specular glint geometry (facet normal = up).
        // The glint contributes to the direct-lit case (shadow 1) but is killed when the
        // sun is occluded (shadow 0); the ambient + Fresnel sky reflection are shadow-
        // independent, so the lit water must be brighter than the occluded water.
        let view = nadir_view();
        let (ctx, sun) = nadir_surface_pixel(90.0);
        let water = SurfacePixel {
            on_earth: true,
            base_srgb: [0.05, 0.08, 0.15], // dark ocean blue
            normal_enu: [0.0, 0.0, 1.0],
            sun_enu: [sun[0] as f32, sun[1] as f32, sun[2] as f32],
            sun_elev_deg: 90.0,
            is_water: true,
            view_dir: view,
            wind_speed: 5.0,
            ..Default::default()
        };
        let lit = surface_toa_radiance(&ctx, &water, 1.0, 1.0).expect("on earth");
        let occluded = surface_toa_radiance(&ctx, &water, 0.0, 1.0).expect("on earth");
        let sum = |l: [f64; 3]| l[0] + l[1] + l[2];
        assert!(
            sum(lit) > sum(occluded) + 1e-6,
            "the sun glint should brighten lit water ({}) over occluded ({})",
            sum(lit),
            sum(occluded)
        );
        assert!(lit.iter().all(|v| v.is_finite() && *v >= 0.0));
    }

    // ── True-color refinement calibration (orchestrator-approved, frozen) ──
    //
    // The five levers (AERIAL_VEIL_DAY_SCALE, LAND_VIBRANCY, LAND_DAY_GAIN, GLINT_STRENGTH,
    // GLINT_MSS_SCALE) are display/albedo calibrations of the shipped radiance path; each is
    // a NO-OP at its identity value. These sanity tests pin the clean parameterization and the
    // intended directional effects. They do NOT re-tune the approved constants (frozen); they
    // assert PROPERTIES that hold for the shipped values, not the values themselves.

    #[test]
    fn refinement_levers_are_identity_no_ops() {
        // (a) Each lever at its NEUTRAL value reproduces the pre-refinement surface output.

        // Veil + land-day-gain are ONE daytime ramp family; at the neutral day_value 1.0 the
        // ramp is exactly 1.0 at every elevation (no daytime change), and the two shipping
        // levers ARE members of that family (only their day_value differs). Both also stay
        // exactly 1.0 through the whole twilight/terminator band (<= LO) by construction.
        for &elev in &[-10.0f64, 0.0, 10.0, 20.0, 25.0, 40.0, 60.0, 90.0] {
            assert_eq!(
                day_lerp_ramp(elev, 1.0),
                1.0,
                "neutral daytime ramp must be identity at elev {elev}"
            );
            assert_eq!(
                aerial_veil_scale(elev),
                day_lerp_ramp(elev, AERIAL_VEIL_DAY_SCALE),
                "the veil scale is the shared day ramp at AERIAL_VEIL_DAY_SCALE"
            );
            assert_eq!(
                land_day_gain(elev),
                day_lerp_ramp(elev, LAND_DAY_GAIN),
                "the land day gain is the shared day ramp at LAND_DAY_GAIN"
            );
            if elev <= AERIAL_VEIL_ELEV_LO_DEG {
                assert_eq!(aerial_veil_scale(elev), 1.0, "twilight veil untouched");
                assert_eq!(land_day_gain(elev), 1.0, "twilight land gain untouched");
            }
        }

        // Veil composite at the neutral veil = 1.0 is the raw physical composite
        // `l_surf * T + inscatter` (no de-haze), bit-for-bit per band.
        let l_surf = [12.0, 15.0, 20.0];
        let trans = [0.82, 0.86, 0.90];
        let inscat = [3.0, 4.0, 6.0];
        let neutral = combine_aerial_veil(l_surf, trans, inscat, 1.0);
        for c in 0..3 {
            assert_eq!(
                neutral[c],
                l_surf[c] * trans[c] + inscat[c],
                "band {c}: neutral veil must be the raw physical composite"
            );
        }

        // Vibrancy at the neutral s = 1.0 returns the albedo unchanged (and the shipping code
        // additionally SKIPS the call when LAND_VIBRANCY == 1.0, so it is exact there).
        let albedo = [0.05, 0.21, 0.04];
        let same = scale_saturation(albedo, 1.0);
        for c in 0..3 {
            assert!(
                (same[c] - albedo[c]).abs() < 1e-12,
                "band {c}: vibrancy 1.0 must be a no-op: {} vs {}",
                same[c],
                albedo[c]
            );
        }

        // Glint strength x1.0 and mss-scale x1.0 leave the raw Cox-Munk kernel unchanged.
        let up = [0.0, 0.0, 1.0];
        let se = 40.0f64.to_radians();
        let to_sun = [se.cos(), 0.0, se.sin()];
        let spec_view = [-se.cos(), 0.0, se.sin()];
        let raw_mss = optics::cox_munk_mean_square_slope(5.0);
        let raw = optics::cox_munk_glint_reflectance(to_sun, spec_view, up, raw_mss);
        let identity = 1.0f64;
        assert_eq!(raw * identity, raw, "glint strength 1.0 is a no-op");
        assert_eq!(
            optics::cox_munk_glint_reflectance(to_sun, spec_view, up, raw_mss * identity),
            raw,
            "glint mss-scale 1.0 is a no-op"
        );
    }

    #[test]
    fn bright_land_gets_more_saturated_and_brighter_without_clipping() {
        // (b) A bright LAND albedo becomes MORE saturated (vibrancy) and BRIGHTER (day gain)
        // yet does NOT clip: the reflectance stays < 1.0 and no display channel pins to 255.

        // Vibrancy raises chroma while PRESERVING Rec.709 luminance (a chroma-only re-weight).
        let bright_land = [0.26f64, 0.30, 0.13]; // linear ~ bright green cropland
        let vivid = scale_saturation(bright_land, LAND_VIBRANCY);
        let y = |c: [f64; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        let chroma = |c: [f64; 3]| {
            let mx = c[0].max(c[1]).max(c[2]);
            let mn = c[0].min(c[1]).min(c[2]);
            mx - mn
        };
        assert!(
            chroma(vivid) > chroma(bright_land) + 1e-6,
            "vibrancy must raise chroma: {} !> {}",
            chroma(vivid),
            chroma(bright_land)
        );
        assert!(
            (y(vivid) - y(bright_land)).abs() < 1e-6,
            "vibrancy must PRESERVE luminance: {} vs {}",
            y(vivid),
            y(bright_land)
        );

        // Day gain brightens at high sun (> 1), is a no-op at twilight, and never overshoots.
        assert!(
            land_day_gain(50.0) > 1.0,
            "high-sun land day gain must brighten"
        );
        assert_eq!(land_day_gain(0.0), 1.0, "twilight land is byte-unchanged");
        assert!(
            land_day_gain(50.0) <= LAND_DAY_GAIN + 1e-12,
            "the gain never exceeds its daytime target (no over-bright)"
        );

        // End-to-end: a bright land pixel at high sun through the full shipped radiance path
        // stays sub-clip — every reflectance channel in [0,1) and no display channel at white.
        let view = nadir_view();
        let (mut ctx, sun) = nadir_surface_pixel(50.0);
        ctx.bm_present = true; // use the (bright, coloured) base_srgb below, not the flat albedo
        let px = SurfacePixel {
            on_earth: true,
            base_srgb: [0.55, 0.60, 0.40], // bright vegetated / cropland (sRGB)
            normal_enu: [0.0, 0.0, 1.0],
            sun_enu: [sun[0] as f32, sun[1] as f32, sun[2] as f32],
            sun_elev_deg: 50.0,
            is_water: false,
            view_dir: view,
            ..Default::default()
        };
        let l = surface_toa_radiance(&ctx, &px, 1.0, 1.0).expect("on earth");
        let e_sun = SOLAR_IRRADIANCE_RGB;
        for c in 0..3 {
            let rho = std::f64::consts::PI * l[c] / e_sun[c];
            assert!(
                rho.is_finite() && (0.0..1.0).contains(&rho),
                "band {c} reflectance {rho} must stay in [0,1) (no clip)"
            );
        }
        let disp = radiance_to_rgba(l, ctx.output_transform, ctx.exposure);
        for (c, &v) in disp[..3].iter().enumerate() {
            assert!(v < 1.0, "band {c} display {v} must not pin to white (255)");
        }
    }

    #[test]
    fn glint_peak_scales_up_with_glint_strength() {
        // (c) The glint PEAK scales linearly with GLINT_STRENGTH (a display gain on the
        // Cox-Munk reflectance): a larger strength -> a brighter peak, monotonically, and the
        // shipped strength (> 1) lifts the peak above the raw Cox-Munk glint.
        let up = [0.0, 0.0, 1.0];
        let se = 35.0f64.to_radians();
        let to_sun = [se.cos(), 0.0, se.sin()];
        let spec_view = [-se.cos(), 0.0, se.sin()]; // the specular peak direction
        let mss = optics::cox_munk_mean_square_slope(4.0) * GLINT_MSS_SCALE;
        let peak = optics::cox_munk_glint_reflectance(to_sun, spec_view, up, mss);
        assert!(
            peak > 0.0,
            "there must be a specular peak to scale, got {peak}"
        );
        let mut prev = -1.0;
        for &strength in &[1.0f64, 2.0, GLINT_STRENGTH, 6.0] {
            let scaled = peak * strength;
            assert!(
                scaled > prev,
                "the peak must rise with strength {strength}: {scaled} <= {prev}"
            );
            prev = scaled;
        }
        // The shipped strength (> 1) lifts the peak above the raw Cox-Munk glint. Encoded
        // via the runtime peak (peak > 0), so `peak * GLINT_STRENGTH > peak` iff strength > 1.
        assert!(
            peak * GLINT_STRENGTH > peak,
            "the shipped glint is brighter than the raw Cox-Munk peak"
        );
    }

    #[test]
    fn glint_core_tightens_with_smaller_mss_scale() {
        // (c) A SMALLER GLINT_MSS_SCALE tightens the Cox-Munk slope PDF: energy concentrates
        // into a narrower, BRIGHTER specular core. Assert (1) the peak rises as the mss-scale
        // shrinks, and (2) the core is narrower (a fixed off-specular shoulder is a smaller
        // fraction of the peak at the tighter scale).
        let up = [0.0, 0.0, 1.0];
        let se = 35.0f64.to_radians();
        let to_sun = [se.cos(), 0.0, se.sin()];
        let spec_view = [-se.cos(), 0.0, se.sin()];
        let off = 6.0f64.to_radians(); // a small azimuth offset = the core "shoulder"
        let off_view = [-se.cos() * off.cos(), se.cos() * off.sin(), se.sin()];
        let base_mss = optics::cox_munk_mean_square_slope(4.0);

        // (1) The peak brightens as the mss-scale shrinks (a tighter PDF -> higher 1/(pi*mss)).
        let mut prev_peak = -1.0;
        for &scale in &[1.0f64, 0.6, GLINT_MSS_SCALE, 0.2] {
            let peak = optics::cox_munk_glint_reflectance(to_sun, spec_view, up, base_mss * scale);
            assert!(
                peak > prev_peak,
                "smaller mss-scale must brighten the peak at {scale}: {peak} <= {prev_peak}"
            );
            prev_peak = peak;
        }

        // (2) The core NARROWS: at the shipped tight scale the off-specular shoulder is a
        // SMALLER fraction of the peak than at the untightened (1.0) scale.
        let shoulder_ratio = |scale: f64| {
            let peak = optics::cox_munk_glint_reflectance(to_sun, spec_view, up, base_mss * scale);
            let shoulder =
                optics::cox_munk_glint_reflectance(to_sun, off_view, up, base_mss * scale);
            shoulder / peak
        };
        // The shipped mss-scale (< 1) tightens the core: its off-specular shoulder is a
        // smaller fraction of the peak than the untightened (1.0) scale (runtime-encoded).
        assert!(
            shoulder_ratio(GLINT_MSS_SCALE) < shoulder_ratio(1.0),
            "the tightened core must fall off faster: {} !< {}",
            shoulder_ratio(GLINT_MSS_SCALE),
            shoulder_ratio(1.0)
        );
    }

    // ── top-down / basemap appearance pass (ground lift + highlight soft-clip) ──

    #[test]
    fn soft_clip_highlight_identity_below_knee_bounded_and_monotone() {
        let knee = CLOUD_SOFTCLIP_KNEE;
        let x_max = DEFAULT_EXPOSURE * RHO_HIGHLIGHT_MAX; // the shipped bound at 1.6
        // Strictly identity at/below the knee (approved mid-tones/daytime/twilight).
        for &x in &[0.0, 0.1, 0.5, knee] {
            assert_eq!(
                soft_clip_highlight(x, knee, x_max),
                x,
                "identity at/below knee at {x}"
            );
        }
        // Above the knee: strictly monotone increasing, <= 1, exactly 1.0 at/above
        // x_max. Distinct bright tops below x_max land on distinct values (structure).
        let mut prev = knee;
        for &x in &[0.8, 1.0, 1.3, x_max - 1e-6] {
            let y = soft_clip_highlight(x, knee, x_max);
            assert!(y > prev, "monotone increasing at {x}: {y} <= {prev}");
            assert!(y < 1.0, "below white short of x_max at {x}: {y}");
            prev = y;
        }
        assert!(
            soft_clip_highlight(1.3, knee, x_max) > soft_clip_highlight(1.0, knee, x_max),
            "bright tops must keep structure (not both pin to white)"
        );
        // C1-continuous at the knee: the slope from just above is ~1 (matches identity).
        let h = 1e-6;
        let slope = (soft_clip_highlight(knee + h, knee, x_max) - knee) / h;
        assert!(
            (slope - 1.0).abs() < 1e-3,
            "slope at the knee should be ~1, got {slope}"
        );
    }

    #[test]
    fn soft_clip_bounded_segment_hits_one_at_xmax_with_nonzero_end_slope() {
        // The WS2 bounded shoulder: [knee, x_max] -> [knee, 1.0]. It must reach EXACTLY
        // 1.0 at x_max (no wasted asymptote range), keep a NONZERO slope approaching
        // x_max (the analytic end slope is (span/w)^2), and hard-clamp to 1.0 only above.
        let knee = CLOUD_SOFTCLIP_KNEE;
        for &exposure in &[1.0f64, DEFAULT_EXPOSURE, 2.5] {
            let x_max = exposure * RHO_HIGHLIGHT_MAX;
            let y_end = soft_clip_highlight(x_max, knee, x_max);
            assert_eq!(
                y_end, 1.0,
                "y(x_max) must be exactly 1.0 at exposure {exposure}"
            );
            // Nonzero end slope: numeric slope just below x_max matches (span/w)^2.
            let h = 1e-6;
            let span = 1.0 - knee;
            let w = x_max - knee;
            let want = (span / w) * (span / w);
            let slope = (1.0 - soft_clip_highlight(x_max - h, knee, x_max)) / h;
            assert!(
                (slope - want).abs() < 1e-3 && slope > 0.0,
                "end slope at exposure {exposure}: got {slope}, want {want} (> 0)"
            );
            // Hard white only above x_max.
            assert_eq!(soft_clip_highlight(x_max + 0.01, knee, x_max), 1.0);
            assert_eq!(soft_clip_highlight(x_max * 10.0, knee, x_max), 1.0);
        }
        // Degenerate bound at/below the knee: the hard clamp (never panics/NaN).
        assert_eq!(soft_clip_highlight(0.9, knee, knee), 1.0);
        assert_eq!(
            soft_clip_highlight(0.5, knee, knee - 0.2),
            0.5,
            "identity below knee"
        );
    }

    #[test]
    fn soft_clip_infinite_xmax_reduces_to_the_reinhard_shoulder() {
        // The regression anchor: as x_max -> inf the bounded Mobius reduces EXACTLY to
        // the previous unbounded Reinhard `knee + span * a / (a + span)` — the bounded
        // curve is the same family, so the pre-WS2 shoulder is recoverable bit-for-bit.
        let knee = CLOUD_SOFTCLIP_KNEE;
        let span = 1.0 - knee;
        for &x in &[0.2, knee, 0.8, 1.0, 1.36, 2.0, 10.0, 500.0] {
            let got = soft_clip_highlight(x, knee, f64::INFINITY);
            let want = if x <= knee {
                x
            } else {
                let a = x - knee;
                knee + span * (a / (a + span))
            };
            assert_eq!(got, want, "Reinhard limit at {x}");
        }
    }

    #[test]
    fn soft_clip_contrast_floor_recovers_the_cloud_band_at_shipped_exposure() {
        // THE WHITE-SQUARE FIX, quantified. At exposure 1.6 the cloud-texture band
        // rho 0.6..0.85 lands at x = 0.96..1.36. The old unbounded Reinhard collapsed
        // it to a 0.033 display delta (the flat white square); the bounded shoulder
        // must keep >= 0.08 of LINEAR separation across the band (shoulder domain) and
        // measurably widen the final display (post-sqrt) delta.
        //
        // NOTE (feasibility, documented in notes/ws2-tonemap-notes.md): a 0.08 POST-SQRT
        // delta is unreachable for ANY monotone C1 concave shoulder with knee 0.75 and
        // x_max = 1.68 (concavity caps it at ~0.058); the 0.08 floor is asserted on the
        // shoulder output, and the display floor is asserted at the strongest feasible
        // level (0.045, a 1.39x recovery over the 0.0334 baseline).
        let knee = CLOUD_SOFTCLIP_KNEE;
        let exposure = DEFAULT_EXPOSURE;
        let x_max = exposure * RHO_HIGHLIGHT_MAX;
        let (x_lo, x_hi) = (0.96, 1.36); // exposure-applied rho 0.6 / 0.85
        let y_lo = soft_clip_highlight(x_lo, knee, x_max);
        let y_hi = soft_clip_highlight(x_hi, knee, x_max);
        assert!(
            y_hi - y_lo >= 0.08,
            "shoulder-domain contrast floor: {} - {} = {} < 0.08",
            y_hi,
            y_lo,
            y_hi - y_lo
        );
        // End-to-end display floor through the real tonemap (neutral grey radiance so
        // desaturation is a no-op), vs the old Reinhard's 0.0334.
        let e_sun = SOLAR_IRRADIANCE_RGB;
        let l_of = |rho: f64| {
            [
                rho * e_sun[0] / std::f64::consts::PI,
                rho * e_sun[1] / std::f64::consts::PI,
                rho * e_sun[2] / std::f64::consts::PI,
            ]
        };
        let d_lo = radiance_to_rgba(l_of(0.60), OutputTransform::AbiReflectance, exposure)[0];
        let d_hi = radiance_to_rgba(l_of(0.85), OutputTransform::AbiReflectance, exposure)[0];
        assert!(
            (d_hi - d_lo) as f64 >= 0.045,
            "display contrast floor: {d_hi} - {d_lo} = {} < 0.045",
            d_hi - d_lo
        );
        // And the very top: the physical peak (rho ~ 1.065 > RHO_HIGHLIGHT_MAX) pins
        // honestly to display white.
        let d_peak = radiance_to_rgba(l_of(1.07), OutputTransform::AbiReflectance, exposure)[0];
        assert_eq!(d_peak, 1.0, "the physical peak saturates to white");
    }

    #[test]
    fn soft_clip_neutral_knee_is_the_hard_clamp() {
        // knee = 1.0 (neutral) is the pre-soft-clip hard clamp: identity below 1, clamp to
        // 1 above. Non-finite / non-positive knee falls back to the same clamp, at any
        // bound (finite or not).
        for &x_max in &[DEFAULT_EXPOSURE * RHO_HIGHLIGHT_MAX, f64::INFINITY] {
            for &knee in &[1.0f64, f64::NAN, 0.0, -2.0] {
                assert_eq!(
                    soft_clip_highlight(0.3, knee, x_max),
                    0.3,
                    "identity below 1 at knee {knee}"
                );
                assert_eq!(soft_clip_highlight(1.0, knee, x_max), 1.0);
                assert_eq!(
                    soft_clip_highlight(1.5, knee, x_max),
                    1.0,
                    "clamp above 1 at knee {knee}"
                );
                assert_eq!(soft_clip_highlight(9.0, knee, x_max), 1.0);
            }
        }
    }

    #[test]
    fn ground_day_lift_is_sun_gated_and_neutral_at_one() {
        // The neutral ground_lift = 1.0 is exactly 1.0 at every elevation (identity no-op).
        for &elev in &[-10.0f64, 0.0, 10.0, 20.0, 30.0, 40.0, 60.0, 90.0] {
            assert_eq!(ground_day_lift(elev, 1.0), 1.0, "neutral no-op at {elev}");
            // It is the shared daytime ramp family (only day_value differs).
            assert_eq!(
                ground_day_lift(elev, GROUND_DAY_LIFT),
                day_lerp_ramp(elev, GROUND_DAY_LIFT)
            );
        }
        // Sun-gated: exactly 1.0 through the whole twilight band (<= LO); the target at HI.
        assert_eq!(
            ground_day_lift(AERIAL_VEIL_ELEV_LO_DEG, GROUND_DAY_LIFT),
            1.0
        );
        assert_eq!(ground_day_lift(0.0, GROUND_DAY_LIFT), 1.0);
        assert!(
            (ground_day_lift(AERIAL_VEIL_ELEV_HI_DEG, GROUND_DAY_LIFT) - GROUND_DAY_LIFT).abs()
                < 1e-12
        );
        assert!(
            ground_day_lift(90.0, GROUND_DAY_LIFT) > 1.0,
            "daytime must lift"
        );
        // Monotone non-decreasing as the sun rises.
        let mut prev = 0.0;
        for &elev in &[-5.0f64, 10.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0] {
            let g = ground_day_lift(elev, GROUND_DAY_LIFT);
            assert!(g >= prev - 1e-12, "ground lift not monotone at {elev}");
            prev = g;
        }
    }

    #[test]
    fn ground_lift_brightens_day_surface_but_not_twilight() {
        let view = nadir_view();
        let sum = |l: [f64; 3]| l[0] + l[1] + l[2];
        // Daytime (50 deg, above the veil HI): a > 1 ground_lift brightens BOTH the land
        // and the water surface radiance (the whole surface, not just land).
        let (day_ctx, day_sun) = nadir_surface_pixel(50.0);
        for is_water in [false, true] {
            let px = SurfacePixel {
                on_earth: true,
                base_srgb: [0.5, 0.5, 0.5],
                normal_enu: [0.0, 0.0, 1.0],
                sun_enu: [day_sun[0] as f32, day_sun[1] as f32, day_sun[2] as f32],
                sun_elev_deg: 50.0,
                is_water,
                view_dir: view,
                wind_speed: 5.0,
                ..Default::default()
            };
            let base = surface_toa_radiance(&day_ctx, &px, 1.0, 1.0).expect("earth");
            let lifted = surface_toa_radiance(&day_ctx, &px, 1.0, GROUND_DAY_LIFT).expect("earth");
            assert!(
                sum(lifted) > sum(base) + 1e-6,
                "ground lift must brighten the daytime {} surface: {} !> {}",
                if is_water { "water" } else { "land" },
                sum(lifted),
                sum(base)
            );
        }
        // Twilight (5 deg, at/below the veil LO): the lift ramp is 1.0, so ANY ground_lift
        // is byte-identical to the neutral surface — the M2 twilight look is preserved.
        let (twi_ctx, twi_sun) = nadir_surface_pixel(5.0);
        let px = SurfacePixel {
            on_earth: true,
            base_srgb: [0.5, 0.5, 0.5],
            normal_enu: [0.0, 0.0, 1.0],
            sun_enu: [twi_sun[0] as f32, twi_sun[1] as f32, twi_sun[2] as f32],
            sun_elev_deg: 5.0,
            is_water: false,
            view_dir: view,
            ..Default::default()
        };
        let base = surface_toa_radiance(&twi_ctx, &px, 1.0, 1.0).expect("earth");
        let lifted = surface_toa_radiance(&twi_ctx, &px, 1.0, GROUND_DAY_LIFT).expect("earth");
        assert_eq!(
            base, lifted,
            "ground lift must be byte-identical at twilight"
        );
    }

    // ── WS2: water direct sun + day water scale + cloud-shadow floor ──

    /// A daytime water pixel in a NON-SPECULAR geometry (nadir view, calm sea, sun 40
    /// deg off zenith -> the Cox-Munk glint is ~e^-110, numerically nil), so any
    /// shadow/elevation response is the new DIFFUSE direct term, not the glint.
    fn day_water_pixel(elev: f32) -> (FrameContext<'static>, SurfacePixel) {
        let (ctx, sun) = nadir_surface_pixel(elev);
        let px = SurfacePixel {
            on_earth: true,
            base_srgb: [0.05, 0.08, 0.15], // dark ocean blue
            normal_enu: [0.0, 0.0, 1.0],
            sun_enu: [sun[0] as f32, sun[1] as f32, sun[2] as f32],
            sun_elev_deg: elev,
            is_water: true,
            view_dir: nadir_view(),
            wind_speed: 0.0,
            ..Default::default()
        };
        (ctx, px)
    }

    #[test]
    fn water_gets_direct_sun_and_cloud_shadows_at_day() {
        // Day (50 deg, above the ramp HI): a cloud shadow must darken the water BODY
        // (not just the glint — the geometry has none), monotonically in the shadow
        // fraction. Pre-WS2 the water body had NO direct term, so lit == occluded here.
        let (ctx, px) = day_water_pixel(50.0);
        let sum = |l: [f64; 3]| l[0] + l[1] + l[2];
        let occluded = sum(surface_toa_radiance(&ctx, &px, 0.0, 1.0).expect("earth"));
        let half = sum(surface_toa_radiance(&ctx, &px, 0.5, 1.0).expect("earth"));
        let lit = sum(surface_toa_radiance(&ctx, &px, 1.0, 1.0).expect("earth"));
        assert!(
            occluded + 1e-9 < half && half + 1e-9 < lit,
            "cloud shadow must darken day water monotonically: {occluded} / {half} / {lit}"
        );
        // The shadow removes a SUBSTANTIAL direct fraction (not an epsilon): with the
        // floor f the occluded direct keeps f of it, so lit/occluded > 1 clearly.
        assert!(
            lit > occluded * 1.05,
            "the direct term should be a real fraction of day water: lit {lit} vs occluded {occluded}"
        );
    }

    #[test]
    fn water_brightness_responds_to_sun_elevation_at_day() {
        // Ocean brightness must rise with the sun (the pre-WS2 water body barely moved —
        // ambient only). Both elevations are above the ramp HI so the day scale is equal;
        // the response is the direct term's t_sun * sin(elev).
        let sum = |l: [f64; 3]| l[0] + l[1] + l[2];
        let (ctx_lo, px_lo) = day_water_pixel(35.0);
        let (ctx_hi, px_hi) = day_water_pixel(75.0);
        let lo = sum(surface_toa_radiance(&ctx_lo, &px_lo, 1.0, 1.0).expect("earth"));
        let hi = sum(surface_toa_radiance(&ctx_hi, &px_hi, 1.0, 1.0).expect("earth"));
        assert!(
            hi > lo * 1.1,
            "day ocean must brighten with sun elevation: {hi} !> {lo}"
        );
    }

    #[test]
    fn water_twilight_is_shadow_invariant_and_byte_stable() {
        // At/below the ramp LO the day gate is 0: NO direct term, NO day albedo rescale
        // — the water radiance is the locked M2 twilight expression, and (in this
        // glint-free geometry) it is INVARIANT to the cloud shadow. This is the unit
        // proxy of the twilight-sweep byte-identity gate.
        for &elev in &[5.0f32, 12.0, 20.0] {
            let (ctx, px) = day_water_pixel(elev);
            let lit = surface_toa_radiance(&ctx, &px, 1.0, 1.0).expect("earth");
            let occluded = surface_toa_radiance(&ctx, &px, 0.0, 1.0).expect("earth");
            for c in 0..3 {
                assert!(
                    (lit[c] - occluded[c]).abs() < 1e-15,
                    "twilight water must not see the shadow at {elev} deg: {} vs {}",
                    lit[c],
                    occluded[c]
                );
            }
        }
    }

    #[test]
    fn water_stays_darker_than_land_at_equal_albedo_day() {
        // The dark-ocean look holds: at equal base albedo and full day, water renders
        // darker than land (the day water scale 0.35 + no land gain/vibrancy on water).
        let view = nadir_view();
        let (ctx, sun) = nadir_surface_pixel(50.0);
        let sum = |l: [f64; 3]| l[0] + l[1] + l[2];
        let mk = |is_water| SurfacePixel {
            on_earth: true,
            base_srgb: [0.3, 0.3, 0.3],
            normal_enu: [0.0, 0.0, 1.0],
            sun_enu: [sun[0] as f32, sun[1] as f32, sun[2] as f32],
            sun_elev_deg: 50.0,
            is_water,
            view_dir: view,
            wind_speed: 0.0,
            ..Default::default()
        };
        let land = sum(surface_toa_radiance(&ctx, &mk(false), 1.0, 1.0).expect("earth"));
        let water = sum(surface_toa_radiance(&ctx, &mk(true), 1.0, 1.0).expect("earth"));
        assert!(
            water < land,
            "day water {water} must stay darker than land {land} at equal albedo"
        );
    }

    #[test]
    fn effective_cloud_shadow_is_sun_gated_floored_and_monotone() {
        // Twilight (<= LO): exactly the input shadow (byte-identity gate).
        for &s in &[0.0f64, 0.3, 0.7, 1.0] {
            assert_eq!(
                effective_cloud_shadow(s, 5.0),
                s,
                "twilight identity at {s}"
            );
            assert_eq!(
                effective_cloud_shadow(s, AERIAL_VEIL_ELEV_LO_DEG),
                s,
                "identity at the ramp LO"
            );
        }
        // Full day: a fully-occluded pixel keeps the floor (cloud-scattered fill), an
        // unshadowed pixel is EXACTLY 1 (byte-identical), and the map is monotone.
        assert_eq!(effective_cloud_shadow(0.0, 50.0), CLOUD_SHADOW_FLOOR);
        assert_eq!(effective_cloud_shadow(1.0, 50.0), 1.0);
        let mut prev = -1.0;
        for &s in &[0.0f64, 0.25, 0.5, 0.75, 1.0] {
            let e = effective_cloud_shadow(s, 50.0);
            assert!(e > prev, "monotone in shadow at {s}");
            assert!((CLOUD_SHADOW_FLOOR..=1.0).contains(&e));
            prev = e;
        }
    }

    #[test]
    fn cloud_shadowed_day_land_keeps_direct_fill_above_ambient_only() {
        // End-to-end 4b: at day, a FULLY cloud-shadowed land pixel must stay brighter
        // than the same pixel with the sun terrain-occluded (disk = 0 -> a true
        // ambient-only reference with identical ambient), because the shadow floor
        // keeps a fraction of the direct term as cloud-scattered fill. This is what
        // lifts the "hard navy blob" shadows.
        let view = nadir_view();
        let (ctx, sun) = nadir_surface_pixel(50.0);
        let sum = |l: [f64; 3]| l[0] + l[1] + l[2];
        let base = SurfacePixel {
            on_earth: true,
            base_srgb: [0.4, 0.4, 0.4],
            normal_enu: [0.0, 0.0, 1.0],
            sun_enu: [sun[0] as f32, sun[1] as f32, sun[2] as f32],
            sun_elev_deg: 50.0,
            is_water: false,
            view_dir: view,
            ..Default::default()
        };
        let cloud_shadowed = sum(surface_toa_radiance(&ctx, &base, 0.0, 1.0).expect("earth"));
        let ambient_only = sum(surface_toa_radiance(
            &ctx,
            &SurfacePixel {
                terrain_horizon_rad: 85.0f32.to_radians(), // sun fully below the ridge
                ..base
            },
            1.0,
            1.0,
        )
        .expect("earth"));
        assert!(
            cloud_shadowed > ambient_only * 1.02,
            "the shadow floor must keep direct fill: shadowed {cloud_shadowed} vs ambient-only {ambient_only}"
        );
    }
}
