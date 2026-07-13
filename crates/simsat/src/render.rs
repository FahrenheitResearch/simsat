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
/// This is the horizon/night anchor: the water body uses exactly this scale at and
/// below the horizon, then reaches [`WATER_ALBEDO_DAY_SCALE`] by 12 degrees solar
/// elevation (see `surface_toa_radiance`'s water branch).
pub const WATER_ALBEDO_SCALE: f32 = 0.55;

/// DAYTIME water-body albedo scale (WS2 water direct-sun pass). The water branch now
/// receives the same disk-gated, shadow-weighted DIRECT solar term the land branch has
/// (so cloud shadows exist over the ocean and ocean brightness responds to the sun);
/// that added flux would brighten the owner-approved dark-ocean look, so the water-body
/// albedo scale is simultaneously retuned DOWN across the surface-help ramp: effective
/// scale = `water_scale` (0.55) at/below the horizon, ramping to this value by 12
/// degrees. The dark-ocean/distinct-glint contrast is held by the lower daylight albedo.
pub const WATER_ALBEDO_DAY_SCALE: f64 = 0.35;

/// CLOUD-SHADOW FLOOR (WS2, QA item: daytime cloud ground-shadows read as very dark
/// hard navy blobs). The sun-OD ground shadow counts only EXTINCTION along the sun ray
/// — a fully-occluded ground pixel kept NO direct term and was lit by sky ambient
/// alone (blue-dominant -> navy). Physically the shadowed ground under a bright sunlit
/// cloud also receives strong DOWNSCATTERED flux from the cloud itself, which the
/// ground-shadow consumer does not model. This floor stands in for that fill: the
/// effective shadow is `f + (1 - f) * shadow` with `f = CLOUD_SHADOW_FLOOR`
/// ([`effective_cloud_shadow`]). The fill is not gated by solar
/// elevation because it only affects a direct-sun term that already vanishes at night.
/// An unshadowed pixel (`shadow = 1`) maps exactly to `1`. `0.0` = the old hard floor.
pub const CLOUD_SHADOW_FLOOR: f64 = 0.45;

/// Default display-side exposure gain (see [`radiance_to_rgba`]). The v0.1.5
/// cross-case review selected `1.5`: it restores the uniformly dark HRRR 21Z frame,
/// while the exposure-aware highlight shoulder keeps the 2011 NSSL and Michael
/// cloud cases below clipping. `1.0` remains the exact neutral/identity override in
/// Studio, the CLI, Python, and the Rust API.
pub const DEFAULT_EXPOSURE: f64 = 1.5;

/// Reference solar elevation for the land-only solar-zenith display
/// normalization. At and above this elevation the correction is exactly neutral.
pub const LAND_SZA_REFERENCE_ELEV_DEG: f64 = 60.0;

/// Default upper bound for the land-only solar-zenith normalization. The correction
/// starts at the horizon and is fully enabled by 12 degrees solar elevation. Multi-time
/// low-sun QA selected `4.0`: it closely brackets the Lambertian `sin(60) / sin(12)`
/// recovery while the existing elevation ramp keeps twilight and high-sun scenes neutral.
pub const LAND_SZA_MAX_GAIN: f64 = 4.0;

/// Linear-reflectance luminance knee for the dark-land toe. Land at or above
/// this value is unchanged; darker positive reflectances receive a bounded scalar lift.
pub const LAND_DARK_TOE_KNEE: f64 = 0.08;

/// Power-law exponent for the dark-land toe. Values below one lift the dark
/// reflectance range while preserving zero and meeting the identity at the knee.
pub const LAND_DARK_TOE_GAMMA: f64 = 0.65;

/// Default upper bound for the dark-land toe's scalar lift.
pub const LAND_DARK_TOE_MAX_GAIN: f64 = 1.5;

/// Default linear-reflectance knee for the opt-in post-lighting surface toe.
pub const SURFACE_POSTLIGHT_TOE_KNEE: f64 = 0.18;

/// Default exponent for the opt-in post-lighting surface toe.
pub const SURFACE_POSTLIGHT_TOE_GAMMA: f64 = 0.80;

/// Default upper bound for the opt-in post-lighting surface toe.
pub const SURFACE_POSTLIGHT_TOE_MAX_GAIN: f64 = 1.35;

/// Owner-selected low-sun terrain-recovery parameters used by the shipped visible-display
/// paths. The explicit [`TwilightSurfaceRecoveryConfig::off`] constructor remains the
/// identity path for sensor, raw, thermal, derived, and cloud-only products.
pub const TWILIGHT_SURFACE_RECOVERY_KNEE: f64 = 0.30;
pub const TWILIGHT_SURFACE_RECOVERY_GAMMA: f64 = 0.50;
pub const TWILIGHT_SURFACE_RECOVERY_MAX_GAIN: f64 = 4.0;

/// Start/end of the civil-twilight fade-in for the independent post-light surface toe.
/// This weights a display-side transform of the already-lit surface contribution; it
/// never adds to the direct-sun irradiance term.
pub const SURFACE_TWILIGHT_IN_LO_DEG: f64 = -6.0;
pub const SURFACE_TWILIGHT_IN_HI_DEG: f64 = 0.0;

/// Tight low-sun fade-out for the separate twilight recovery. The owner-selected
/// branch stays full through +4 degrees and is exact identity by +12 degrees, so the
/// already-good higher-sun cases remain untouched.
pub const SURFACE_TWILIGHT_OUT_LO_DEG: f64 = 4.0;
pub const SURFACE_TWILIGHT_OUT_HI_DEG: f64 = 12.0;

/// Display-only terrain experiment applied to the already-lit, view-attenuated LAND surface
/// contribution before atmospheric airlight and clouds are composited. It is separate
/// from [`LandAppearanceConfig`]: this operator responds to the final surface signal,
/// not the source albedo, and is deliberately default-off.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfacePostlightToeConfig {
    pub enabled: bool,
    /// Linear reflectance-factor luminance at which the operator becomes identity.
    pub knee: f64,
    /// Toe exponent; `1.0` is identity and lower values lift dark positive signals.
    pub gamma: f64,
    /// Upper bound for the colour-preserving scalar lift; `1.0` is identity.
    pub max_gain: f64,
}

impl Default for SurfacePostlightToeConfig {
    fn default() -> Self {
        Self::off()
    }
}

impl SurfacePostlightToeConfig {
    /// Shipped/default identity path. The stored parameters remain ready for an A/B.
    pub const fn off() -> Self {
        Self {
            enabled: false,
            knee: SURFACE_POSTLIGHT_TOE_KNEE,
            gamma: SURFACE_POSTLIGHT_TOE_GAMMA,
            max_gain: SURFACE_POSTLIGHT_TOE_MAX_GAIN,
        }
    }

    #[inline]
    pub const fn is_identity(self) -> bool {
        !self.enabled
    }
}

/// Separate low-sun/civil-twilight recovery over the lit, view-attenuated LAND signal.
/// It deliberately does not retune [`SurfacePostlightToeConfig`], whose established
/// daylight behavior and parameters remain stable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwilightSurfaceRecoveryConfig {
    pub enabled: bool,
    pub knee: f64,
    pub gamma: f64,
    pub max_gain: f64,
}

impl Default for TwilightSurfaceRecoveryConfig {
    fn default() -> Self {
        Self::off()
    }
}

impl TwilightSurfaceRecoveryConfig {
    /// Owner-selected shipped visible-display calibration. [`Default`] remains the
    /// conservative identity constructor so non-display call sites must opt in explicitly.
    pub const fn shipped() -> Self {
        Self {
            enabled: true,
            knee: TWILIGHT_SURFACE_RECOVERY_KNEE,
            gamma: TWILIGHT_SURFACE_RECOVERY_GAMMA,
            max_gain: TWILIGHT_SURFACE_RECOVERY_MAX_GAIN,
        }
    }

    pub const fn off() -> Self {
        Self {
            enabled: false,
            knee: TWILIGHT_SURFACE_RECOVERY_KNEE,
            gamma: TWILIGHT_SURFACE_RECOVERY_GAMMA,
            max_gain: TWILIGHT_SURFACE_RECOVERY_MAX_GAIN,
        }
    }

    #[inline]
    pub const fn is_identity(self) -> bool {
        !self.enabled
    }
}

/// Display-only land appearance controls. The owner-selected current preset enables
/// both corrections; [`LandAppearanceConfig::identity`] is the explicit legacy/no-op
/// path. They operate on the land surface signal before aerial perspective and cloud
/// compositing. Water/glint, clouds, limb/space, thermal products, derived products,
/// cloud-only layers, and raw visible-band diagnostics do not consume them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LandAppearanceConfig {
    /// Recover the operational true-colour visibility lost when a Lambertian surface is
    /// viewed at moderate solar zenith. Exactly neutral at/below the horizon and at/
    /// above [`LAND_SZA_REFERENCE_ELEV_DEG`].
    pub sza_normalization: bool,
    /// Upper bound for [`LandAppearanceConfig::sza_normalization`].
    pub sza_max_gain: f64,
    /// Apply a bounded toe to dark positive land reflectances. Zero and reflectances at
    /// or above [`LandAppearanceConfig::dark_toe_knee`] remain unchanged.
    pub dark_toe: bool,
    /// Linear-reflectance luminance knee for the dark-land toe.
    pub dark_toe_knee: f64,
    /// Power-law exponent for the dark-land toe.
    pub dark_toe_gamma: f64,
    /// Upper bound for the dark-land toe's scalar lift.
    pub dark_toe_max_gain: f64,
}

impl Default for LandAppearanceConfig {
    fn default() -> Self {
        Self::shipped()
    }
}

impl LandAppearanceConfig {
    /// Owner-selected finished-visible land calibration.
    pub const fn shipped() -> Self {
        Self {
            sza_normalization: true,
            sza_max_gain: LAND_SZA_MAX_GAIN,
            dark_toe: true,
            dark_toe_knee: LAND_DARK_TOE_KNEE,
            dark_toe_gamma: LAND_DARK_TOE_GAMMA,
            dark_toe_max_gain: LAND_DARK_TOE_MAX_GAIN,
        }
    }

    /// Exact legacy/no-op configuration. Keep reference physics tests, raw diagnostic
    /// paths, and reproducibility profiles on this constructor instead of relying on
    /// [`Default`], whose meaning is the current shipped display preset.
    pub const fn identity() -> Self {
        Self {
            sza_normalization: false,
            sza_max_gain: LAND_SZA_MAX_GAIN,
            dark_toe: false,
            dark_toe_knee: LAND_DARK_TOE_KNEE,
            dark_toe_gamma: LAND_DARK_TOE_GAMMA,
            dark_toe_max_gain: LAND_DARK_TOE_MAX_GAIN,
        }
    }

    /// Whether the config is the exact legacy/no-op path.
    #[inline]
    pub const fn is_identity(self) -> bool {
        !self.sza_normalization && !self.dark_toe
    }
}

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
// All are sun-gated or surface-scoped. Surface help begins above the horizon so terrain
// remains visible in the same low-sun interval as the corrected atmosphere and clouds.

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

/// Sun-elevation gate (deg) for the legacy DAYTIME calibration ramp
/// ([`day_lerp_ramp`]), now retained by the top-down cloud normalization. Surface
/// visibility controls use the earlier [`SURFACE_HELP_ELEV_LO_DEG`]--
/// [`SURFACE_HELP_ELEV_HI_DEG`] ramp instead.
///
/// NOTE (low-sun visible pass): the aerial-perspective VEIL is NO LONGER a member of
/// this family — it has its own SUNRISE ramp ([`aerial_veil_scale`], constants below).
/// The 20-deg hard veil gate was backwards vs operational practice: satpy/pyspectral
/// REDUCE the Rayleigh correction approaching the terminator rather than disabling it,
/// so the sunrise band (2-20 deg) was rendering the FULL blue airlight over a nearly
/// unlit surface — the reported flat navy ground.
pub const AERIAL_VEIL_ELEV_LO_DEG: f64 = 20.0;
/// Upper end of the legacy daytime calibration ramp (deg). See
/// [`AERIAL_VEIL_ELEV_LO_DEG`].
/// WS2 QA: `40 -> 30` — at a sun elevation of 30 deg (mid-morning) the ramp used to sit
/// at half, leaving the ground half-veiled/half-lifted ("murky"); full daytime treatment
/// now arrives by 30 deg. Frames at/above 40 deg are byte-identical (both ramps
/// saturated); the twilight band (<= 20 deg) is byte-identical as before.
pub const AERIAL_VEIL_ELEV_HI_DEG: f64 = 30.0;

/// Low-sun ramp for surface-visibility corrections. The former shared 20--30 degree
/// gate made every surface control a no-op for roughly the last one to two hours of
/// daylight while the independently corrected atmosphere and elevated cloud remained
/// bright. Surface-only help now begins at the geometric horizon and is fully engaged
/// by 12 degrees. The old daytime output is unchanged because both ramps equal one by
/// 30 degrees.
pub const SURFACE_HELP_ELEV_LO_DEG: f64 = 0.0;
pub const SURFACE_HELP_ELEV_HI_DEG: f64 = 12.0;

// ── sunrise-band veil ramp (low-sun visible pass; the navy-ground fix) ─────────
//
// Real true-color products Rayleigh-correct at ALL solar angles, tapering the
// correction near the terminator (satpy/pyspectral `reduce_rayleigh` idiom) — they
// never render the full molecular airlight over a dawn surface. Our veil de-haze was
// hard-gated OFF below 20 deg, so the whole 2-20 deg sunrise band showed the raw blue
// in-scatter sheet over a nearly unlit ground: a flat navy frame (the phase-1
// `en_td_e5_cloudsoff` probe: the entire clear ground within ~0.01 display luminance,
// R/B ~0.7-0.84 vs the near-black near-neutral real-GOES dawn ground). The fix is a
// TWO-SEGMENT ramp: full physical veil at/below the terminator gate (the approved
// dusk/twilight band stays byte-identical), smoothly reducing toward the daytime
// de-haze across the sunrise band, and holding the daytime value from there up
// (daytime frames at/above [`AERIAL_VEIL_ELEV_HI_DEG`] are byte-identical since both
// old and new ramps sit at [`AERIAL_VEIL_DAY_SCALE`] there).

/// TERMINATOR gate (deg) of the sunrise veil ramp: at/below this sun elevation the
/// veil is exactly `1.0` (the full physical airlight — the twilight/terminator glow
/// and the whole approved below-horizon dusk band are byte-identical by construction).
pub const VEIL_TERMINATOR_ELEV_DEG: f64 = 2.0;
/// Upper end (deg) of the sunrise veil ramp: the full daytime de-haze
/// ([`AERIAL_VEIL_DAY_SCALE`]) is in place by this elevation and held above it.
/// Chosen inside the science-review band ("toward the daytime 0.40 by ~15-20 deg");
/// at 5 deg the smoothstep leaves a small residual de-haze (veil ~0.9 — the satpy
/// reduced-but-not-disabled behavior the review asked to evaluate).
pub const VEIL_SUNRISE_ELEV_HI_DEG: f64 = 16.0;

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

/// LAND daylight BRIGHTNESS lift (refinement pass, round 2). A modest ground-only gain
/// on the LAND surface reflectance (`l_surf`) — the orchestrator found the
/// round-1 de-hazed ground read a touch dark/muted vs the bright daylight land of the
/// GOES true-color references, so the surface signal is lifted toward that brightness.
/// This is a CALIBRATED TRUE-COLOR DISPLAY GAIN on the ground reflectance, NOT the
/// global exposure ([`DEFAULT_EXPOSURE`] = 1.5) and NOT an
/// albedo/physics change: it multiplies the assembled surface radiance before the
/// aerial-perspective veil is added, so only the ground signal brightens (the additive
/// haze is untouched). WATER is excluded (dark ocean stays dark; the glint has its own
/// gain). It uses the 0–12 degree surface-help ramp ([`land_day_gain`]), remaining
/// neutral at/below the horizon and reaching its calibrated value by 12 degrees.
/// `1.0` = no lift. Chosen modest (below any land clip) and verified not to over-bright.
pub const LAND_DAY_GAIN: f64 = 1.20;

// ── top-down / basemap appearance pass (ground lift + highlight soft-clip) ────
//
// Two levers added to fix the reported "the ground renders too dark" (sunlit land
// peaked only ~0.53 display, ocean near-black — darker than real GOES true-color, so
// the owner had to crank exposure to 4, which then blew the storm cloud to a flat
// white square). Both are named display calibrations of the shipped radiance path (the
// LAND_DAY_GAIN / AERIAL_VEIL pattern) and are no-ops at their neutral values.

/// GROUND LIFT — a sun-gated daylight surface-brightness lift on BOTH land and ocean,
/// toward real-GOES true-color ground levels (the reported ground was too dark). Unlike
/// [`LAND_DAY_GAIN`] (land only, a modest vibrancy-companion lift) this lifts the WHOLE
/// surface radiance `l_surf` — land AND water — so the basemap reads bright/vivid and the
/// ocean is a visible dark blue rather than near-black, in BOTH the geostationary and
/// top-down views. It multiplies the assembled surface radiance BEFORE the aerial-
/// perspective veil (only the ground signal brightens, not the additive haze) and BEFORE
/// the cloud composite (the cloud's own radiance is not lifted — the "white square" is
/// handled separately by [`CLOUD_SOFTCLIP_KNEE`] + the top-down cloud normalization). It
/// uses the 0–12 degree surface-help ramp ([`ground_day_lift`]), so it is neutral at/
/// below the horizon and reaches the requested value by 12 degrees. `1.0` = the
/// neutral no-op (reproduces the pre-lift ground). Owner review of the v0.2.1 low-sun
/// 1974 case selected `1.10`: a restrained surface-only lift that preserves the approved
/// cloud radiance. The former `1.6` lift compounded with the land-only gain and display
/// exposure, making terrain brighter than the visible-satellite references in the v0.1.4
/// cross-case review. The value remains overridable through the `render_frame`
/// `ground-gain=` CLI knob and equivalent Python/Studio controls.
pub const GROUND_DAY_LIFT: f64 = 1.10;

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
/// output). Value `0.65` is the cross-case v0.1.4 QA calibration: the shoulder begins
/// before bright cloud tops flatten while leaving darker terrain and cloud texture
/// unchanged. The `render_frame` `cloud-softclip=` CLI knob overrides it (default = this
/// baked value; `1.0` disables it).
pub const CLOUD_SOFTCLIP_KNEE: f64 = 0.65;

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
    /// Sun-gated daytime surface-radiance lift. [`GROUND_DAY_LIFT`] is the shipped
    /// default and `1.0` is the identity/no-lift value.
    pub ground_day_lift: f64,
    /// Highlight shoulder knee used by the visible display transform.
    /// [`CLOUD_SOFTCLIP_KNEE`] is the shipped default and `1.0` disables the shoulder.
    pub cloud_softclip_knee: f64,
    /// Physical reflectance-factor ceiling mapped to display white by the bounded
    /// shoulder. [`RHO_HIGHLIGHT_MAX`] is the shipped default.
    pub cloud_highlight_max: f64,
    /// Use the ABI synthetic-green display arithmetic for this frame. Explicitly
    /// carried with the scene so concurrent renders do not depend on process-global
    /// QA state.
    pub synthetic_green: bool,
    /// Apply the product-facing atmospheric correction (the daytime aerial-veil
    /// reduction, including the matching cloud-front correction). `false` retains the
    /// full modeled path airlight; it does not disable unrelated display transforms.
    pub atmosphere_correction: bool,
    /// Shorten the surface sun/view atmosphere columns to each pixel's terrain
    /// elevation. `false` reproduces the legacy mean-sea-level sphere behavior.
    pub terrain_atmosphere: bool,
    /// Optional display-only land corrections. The default is an exact no-op.
    pub land_appearance: LandAppearanceConfig,
    /// Optional display-only post-lighting LAND surface toe. Default-off and ignored by
    /// water/glint and raw/sensor products at the high-level API seam.
    pub surface_postlight_toe: SurfacePostlightToeConfig,
    /// Separate civil-twilight/low-sun terrain recovery. Visible-display entry points use
    /// the shipped profile; raw/sensor/thermal/derived/cloud-only paths force identity.
    pub twilight_surface_recovery: TwilightSurfaceRecoveryConfig,
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
    /// Terrain height above mean sea level (m), used only when
    /// [`FrameContext::terrain_atmosphere`] is enabled. Missing/out-of-domain values
    /// are `0`; the atmosphere geometry safely clips negative/non-finite values.
    pub surface_elevation_m: f32,
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
            surface_elevation_m: 0.0,
        }
    }
}

/// Legacy 20--30 degree daytime calibration ramp, now retained by the top-down cloud
/// normalization. It interpolates from `1.0` at/below
/// [`AERIAL_VEIL_ELEV_LO_DEG`] to `day_value` at/above
/// [`AERIAL_VEIL_ELEV_HI_DEG`] via monotone smoothstep. Surface visibility controls use
/// [`surface_help_ramp`], while the aerial veil uses [`aerial_veil_scale`]. At the
/// neutral `day_value = 1.0` this remains exactly `1.0` at every elevation.
#[inline]
pub(crate) fn day_lerp_ramp(sun_elev_deg: f64, day_value: f64) -> f64 {
    let t = atmosphere::smoothstep(
        AERIAL_VEIL_ELEV_LO_DEG,
        AERIAL_VEIL_ELEV_HI_DEG,
        sun_elev_deg,
    );
    1.0 - t * (1.0 - day_value)
}

/// Surface-only counterpart to [`day_lerp_ramp`]. This earlier ramp keeps terrain,
/// water, and their display controls synchronized with visible daylight instead of
/// withholding all correction until the sun is already 20 degrees above the horizon.
#[inline]
pub fn surface_help_ramp(sun_elev_deg: f64, day_value: f64) -> f64 {
    let t = atmosphere::smoothstep(
        SURFACE_HELP_ELEV_LO_DEG,
        SURFACE_HELP_ELEV_HI_DEG,
        sun_elev_deg,
    );
    1.0 + t * (day_value - 1.0)
}

/// Aerial-perspective veil scale at a sun elevation (deg) — the SUNRISE ramp (low-sun
/// visible pass): `1.0` at/below [`VEIL_TERMINATOR_ELEV_DEG`] (the full physical
/// airlight; the approved dusk/twilight band byte-identical by construction), smoothly
/// reducing to [`AERIAL_VEIL_DAY_SCALE`] at/above [`VEIL_SUNRISE_ELEV_HI_DEG`] and
/// holding it from there up (daytime at/above [`AERIAL_VEIL_ELEV_HI_DEG`] is
/// byte-identical to the pre-fix ramp, which also sat at the day scale there).
/// Monotone (non-increasing) via smoothstep. See the sunrise-veil module note.
#[inline]
pub fn aerial_veil_scale(sun_elev_deg: f64) -> f64 {
    let t = atmosphere::smoothstep(
        VEIL_TERMINATOR_ELEV_DEG,
        VEIL_SUNRISE_ELEV_HI_DEG,
        sun_elev_deg,
    );
    1.0 - t * (1.0 - AERIAL_VEIL_DAY_SCALE)
}

/// LAND daylight brightness gain at a sun elevation (deg): `1.0` at/below the horizon,
/// ramping to [`LAND_DAY_GAIN`] by [`SURFACE_HELP_ELEV_HI_DEG`]. Monotone via
/// smoothstep. Land-only (the caller gates on `!is_water`).
#[inline]
pub fn land_day_gain(sun_elev_deg: f64) -> f64 {
    surface_help_ramp(sun_elev_deg, LAND_DAY_GAIN)
}

/// GROUND LIFT gain at a sun elevation (deg): `1.0` at/below the horizon, ramping to
/// `ground_lift` by [`SURFACE_HELP_ELEV_HI_DEG`]. Monotone via smoothstep. See
/// [`GROUND_DAY_LIFT`]. `ground_lift` is the
/// baked constant unless the `render_frame` `ground-gain=` knob overrides it; the neutral
/// `ground_lift = 1.0` is `1.0` at every elevation (an exact no-op). Applies to land AND
/// water (the whole surface radiance).
#[inline]
pub fn ground_day_lift(sun_elev_deg: f64, ground_lift: f64) -> f64 {
    surface_help_ramp(sun_elev_deg, ground_lift)
}

/// Bounded, land-only solar-zenith display gain. The correction estimates the ratio
/// between a 60-degree reference illumination and the current horizontal direct-sun
/// cosine. It begins above the horizon, is fully enabled by 12 degrees, and becomes
/// exactly neutral again at/above the reference elevation. A disabled correction is
/// never evaluated by the caller, and `max_gain = 1` is an identity path.
#[inline]
pub fn land_sza_normalization_gain(sun_elev_deg: f64, max_gain: f64) -> f64 {
    let max_gain = if max_gain.is_finite() {
        max_gain.clamp(1.0, 4.0)
    } else {
        LAND_SZA_MAX_GAIN
    };
    if max_gain == 1.0 || sun_elev_deg >= LAND_SZA_REFERENCE_ELEV_DEG {
        return 1.0;
    }
    let mu_ref = LAND_SZA_REFERENCE_ELEV_DEG.to_radians().sin();
    let mu_floor = SURFACE_HELP_ELEV_HI_DEG.to_radians().sin();
    let mu = sun_elev_deg.clamp(0.0, 90.0).to_radians().sin();
    let target = (mu_ref / mu.max(mu_floor)).clamp(1.0, max_gain);
    surface_help_ramp(sun_elev_deg, target)
}

/// Bounded scalar lift for a dark positive land reflectance. `albedo` is the linear
/// RGB surface reflectance after snow/vibrancy handling. The scalar preserves colour
/// ratios, zero remains zero, values at/above the knee are exactly unchanged, and the
/// surface-help ramp keeps the correction neutral at/below the horizon.
#[inline]
pub fn land_dark_toe_gain(
    albedo: [f64; 3],
    sun_elev_deg: f64,
    knee: f64,
    gamma: f64,
    max_gain: f64,
) -> f64 {
    let gain = dark_toe_unweighted_gain(albedo, knee, gamma, max_gain);
    surface_help_ramp(sun_elev_deg, gain)
}

#[inline]
fn dark_toe_unweighted_gain(albedo: [f64; 3], knee: f64, gamma: f64, max_gain: f64) -> f64 {
    let knee = if knee.is_finite() {
        knee.clamp(1.0e-6, 1.0)
    } else {
        LAND_DARK_TOE_KNEE
    };
    let gamma = if gamma.is_finite() {
        gamma.clamp(0.05, 1.0)
    } else {
        LAND_DARK_TOE_GAMMA
    };
    let max_gain = if max_gain.is_finite() {
        max_gain.clamp(1.0, 4.0)
    } else {
        LAND_DARK_TOE_MAX_GAIN
    };
    let y = (0.2126 * albedo[0] + 0.7152 * albedo[1] + 0.0722 * albedo[2]).max(0.0);
    if y <= 0.0 || y >= knee || max_gain == 1.0 || gamma == 1.0 {
        return 1.0;
    }
    let power_target = knee * (y / knee).powf(gamma);
    // Blend the lifted power toe smoothly back toward the original reflectance across
    // the whole 0..knee interval. This is the documented Stage-0 candidate: it avoids
    // an overly broad midtone lift while retaining the bounded very-dark recovery.
    let w = atmosphere::smoothstep(0.0, knee, y);
    let target = power_target * (1.0 - w) + y * w;
    (target / y).clamp(1.0, max_gain)
}

/// Independent civil-twilight weight for the post-light surface toe. It fades in from
/// -6 to 0 degrees, stays fully active through +4 degrees, then fades to exact identity
/// by +12 degrees. Since this is consumed only after lighting and view attenuation, it
/// does not create direct sunlight below the horizon.
#[inline]
pub fn surface_twilight_weight(sun_elev_deg: f64) -> f64 {
    atmosphere::smoothstep(
        SURFACE_TWILIGHT_IN_LO_DEG,
        SURFACE_TWILIGHT_IN_HI_DEG,
        sun_elev_deg,
    ) * (1.0
        - atmosphere::smoothstep(
            SURFACE_TWILIGHT_OUT_LO_DEG,
            SURFACE_TWILIGHT_OUT_HI_DEG,
            sun_elev_deg,
        ))
}

/// Gain for the opt-in post-lighting surface toe. `surface_contribution` is the
/// already-lit surface radiance after camera/view transmittance but before additive
/// atmospheric airlight. It is converted to a solar-normalized reflectance factor and
/// fed through the same smooth, bounded, colour-preserving toe as the established land
/// correction. The established daylight activation is retained unchanged.
#[inline]
pub fn surface_postlight_toe_gain(
    surface_contribution: [f64; 3],
    sun_elev_deg: f64,
    config: SurfacePostlightToeConfig,
) -> f64 {
    if config.is_identity() {
        return 1.0;
    }
    let rho = [
        std::f64::consts::PI * surface_contribution[0] / SOLAR_IRRADIANCE_RGB[0],
        std::f64::consts::PI * surface_contribution[1] / SOLAR_IRRADIANCE_RGB[1],
        std::f64::consts::PI * surface_contribution[2] / SOLAR_IRRADIANCE_RGB[2],
    ];
    let knee = if config.knee.is_finite() {
        config.knee
    } else {
        SURFACE_POSTLIGHT_TOE_KNEE
    };
    let gamma = if config.gamma.is_finite() {
        config.gamma
    } else {
        SURFACE_POSTLIGHT_TOE_GAMMA
    };
    let max_gain = if config.max_gain.is_finite() {
        config.max_gain
    } else {
        SURFACE_POSTLIGHT_TOE_MAX_GAIN
    };
    land_dark_toe_gain(rho, sun_elev_deg, knee, gamma, max_gain)
}

/// Gain of the separate low-sun terrain recovery. Unlike the established post-light
/// toe, this branch is weighted only through civil twilight and low sun and is exact
/// identity again by +12 degrees.
#[inline]
pub fn twilight_surface_recovery_gain(
    surface_contribution: [f64; 3],
    sun_elev_deg: f64,
    config: TwilightSurfaceRecoveryConfig,
) -> f64 {
    if config.is_identity() {
        return 1.0;
    }
    let rho = [
        std::f64::consts::PI * surface_contribution[0] / SOLAR_IRRADIANCE_RGB[0],
        std::f64::consts::PI * surface_contribution[1] / SOLAR_IRRADIANCE_RGB[1],
        std::f64::consts::PI * surface_contribution[2] / SOLAR_IRRADIANCE_RGB[2],
    ];
    let knee = if config.knee.is_finite() {
        config.knee
    } else {
        TWILIGHT_SURFACE_RECOVERY_KNEE
    };
    let gamma = if config.gamma.is_finite() {
        config.gamma
    } else {
        TWILIGHT_SURFACE_RECOVERY_GAMMA
    };
    let max_gain = if config.max_gain.is_finite() {
        config.max_gain
    } else {
        TWILIGHT_SURFACE_RECOVERY_MAX_GAIN
    };
    let target = dark_toe_unweighted_gain(rho, knee, gamma, max_gain);
    1.0 + surface_twilight_weight(sun_elev_deg) * (target - 1.0)
}

#[inline]
pub fn combined_surface_recovery_gain(
    surface_contribution: [f64; 3],
    sun_elev_deg: f64,
    postlight: SurfacePostlightToeConfig,
    twilight: TwilightSurfaceRecoveryConfig,
) -> f64 {
    surface_postlight_toe_gain(surface_contribution, sun_elev_deg, postlight).max(
        twilight_surface_recovery_gain(surface_contribution, sun_elev_deg, twilight),
    )
}

/// Combined land-only display gain. Each correction is independently switchable and
/// bounded; [`LandAppearanceConfig::identity`] returns exactly `1.0` without touching
/// the legacy path.
#[inline]
pub fn land_appearance_gain(
    config: LandAppearanceConfig,
    sun_elev_deg: f64,
    albedo: [f64; 3],
) -> f64 {
    if config.is_identity() {
        return 1.0;
    }
    let sza = if config.sza_normalization {
        land_sza_normalization_gain(sun_elev_deg, config.sza_max_gain)
    } else {
        1.0
    };
    let toe = if config.dark_toe {
        land_dark_toe_gain(
            albedo,
            sun_elev_deg,
            config.dark_toe_knee,
            config.dark_toe_gamma,
            config.dark_toe_max_gain,
        )
    } else {
        1.0
    };
    sza * toe
}

/// Unit-preserving Lambertian surface radiance. `direct_irradiance` is the solar
/// irradiance at the surface before projection by `N.L`; `ambient_irradiance` is the
/// hemispheric diffuse term. With no atmosphere/ambient and an overhead sun, converting
/// the result back through [`reflectance_from_radiance`] returns `albedo` exactly.
#[inline]
pub fn lambert_surface_radiance(
    albedo: [f64; 3],
    direct_irradiance: [f64; 3],
    ndotl: f64,
    ambient_irradiance: [f64; 3],
) -> [f64; 3] {
    let mu = ndotl.clamp(0.0, 1.0);
    let pi = std::f64::consts::PI;
    [
        albedo[0] / pi * (direct_irradiance[0] * mu + ambient_irradiance[0]),
        albedo[1] / pi * (direct_irradiance[1] * mu + ambient_irradiance[1]),
        albedo[2] / pi * (direct_irradiance[2] * mu + ambient_irradiance[2]),
    ]
}

/// Direct surface irradiance from the disk-integrated solar bands after atmospheric,
/// finite-disk, incidence-angle, and shadow attenuation. `SOLAR_IRRADIANCE_RGB` already
/// integrates the limb-darkened disk, so the full visible disk is normalized to 1.0;
/// [`atmosphere::LIMB_DARKENING_DISK_AVG`] must not be applied a second time.
#[inline]
fn surface_direct_irradiance(
    e_sun: [f64; 3],
    t_sun: [f64; 3],
    disk_visible_fraction: f64,
    ndotl: f64,
    shadow: f64,
) -> [f64; 3] {
    let scale = disk_visible_fraction * ndotl * shadow;
    [
        e_sun[0] * t_sun[0] * scale,
        e_sun[1] * t_sun[1] * scale,
        e_sun[2] * t_sun[2] * scale,
    ]
}

/// The EFFECTIVE cloud shadow seen by the DIFFUSE direct-sun terms (land + water body):
/// `f + (1 - f) * shadow` with `f = CLOUD_SHADOW_FLOOR` — see
/// [`CLOUD_SHADOW_FLOOR`]. It is elevation-independent because the direct-sun term
/// already vanishes physically at night. `shadow = 1` maps exactly to `1` and the
/// function is monotone in `shadow`. The SPECULAR glint keeps the RAW
/// shadow (a mirror image of an occluded solar disk is gone; the floor models diffuse
/// cloud-scattered fill, which has no specular component).
#[inline]
pub fn effective_cloud_shadow(shadow: f64, _sun_elev_deg: f64) -> f64 {
    let s = shadow.clamp(0.0, 1.0);
    // The floor multiplies only the direct-sun term, whose finite disk, atmospheric
    // transmittance, and N.L already vanish at night. Gating the fill itself removed it
    // exactly when low-sun cloud shadows became longest and most visible.
    let f = CLOUD_SHADOW_FLOOR.clamp(0.0, 1.0);
    f + (1.0 - f) * s
}

/// Shadow multiplier exported with the standalone cloud overlay. Unlike the full
/// renderer, a host map applies this value to its entire already-lit basemap rather
/// than to the direct-sun term alone. Fade that approximation from neutral at/night to
/// the full effective direct shadow by [`SURFACE_HELP_ELEV_HI_DEG`].
#[inline]
pub fn effective_cloud_shadow_layer(shadow: f64, sun_elev_deg: f64) -> f64 {
    let direct_shadow = effective_cloud_shadow(shadow, sun_elev_deg);
    let daylight = atmosphere::smoothstep(
        SURFACE_HELP_ELEV_LO_DEG,
        SURFACE_HELP_ELEV_HI_DEG,
        sun_elev_deg,
    );
    if daylight <= 0.0 {
        return 1.0;
    }
    if daylight >= 1.0 {
        return direct_shadow;
    }
    1.0 - daylight * (1.0 - direct_shadow)
}

// ── low-sun ILLUMINANT CORRECTION (sunrise/dawn visible pass; the cast fix) ─────
//
// At sun elevations of ~2-20 deg the direct solar beam reaching a cloud has crossed a
// long grazing atmosphere path: Rayleigh strips blue AND the Chappuis-band ozone
// (OZONE_STRENGTH 1.45, the approved dusk stylization — see atmosphere.rs) eats a bite
// out of GREEN. Every sunlit pixel is lit by that same tinted illuminant (the octave
// multi-scatter sum multiplies the ONE reddened t_sun), so low cloud renders khaki/tan
// and thick high cloud — where the blue sky ambient is a comparable share — renders
// mauve/lavender (phase-1 diagnosis, notes/lowsun-visible-notes.md: cloud G/B 0.94-1.01
// vs the real-GOES sunrise reference 1.16). Real true-color products remove the
// illuminant tint by DIVIDING by the atmosphere-model transmittance (MODIS CREFL;
// GeoColor and satpy/pyspectral do the equivalent Rayleigh/illuminant correction), so
// the correction is derived from the SAME atmosphere model by construction and can
// never desynchronize from an ozone/AOD retune.
//
// Ours mirrors that, correcting exactly the DIAGNOSED defect: per displayed pixel,
// the GREEN channel is restored to the RAYLEIGH LINE of OUR OWN transmittance-LUT
// direct-sun illuminant sampled at a reference cloud altitude for the pixel's sun
// elevation. In reflectance space (`rho = pi L / E_sun`) the solar irradiance cancels
// per channel, so the rho-space illuminant color is exactly the transmittance triple
// `t_sun`. Under a pure lambda^-4 (Rayleigh + gray-aerosol) atmosphere the LOG
// transmittances of the three bands are collinear in the Rayleigh coefficients; the
// Chappuis ozone term is what pulls ln(t_G) BELOW that line (the green dip). The gain
// restores it: `g_G = t_G_rayleigh / t_G` with
// `ln t_G_rayleigh = lerp(ln t_R, ln t_B, a)`, `a = (beta_G - beta_R)/(beta_B -
// beta_R)` from `atmosphere::RAYLEIGH_SCATTERING` — both the sample and the expected
// value come from the same atmosphere model, so an ozone/AOD retune flows through
// automatically. The R-B (warm) axis is DELIBERATELY preserved: the real-GOES sunrise
// reference keeps R/B ~1.19 on dawn cloud, and the round-2 A/B measured that a FULL
// unit-luminance white balance (dividing by the whole illuminant color) overshoots to
// BLUE (Enderlin e5 cloud G/B 0.971 -> 0.893, R/B 1.146 -> 0.940 — worse than
// baseline on the acceptance axis) because the blue sky ambient is the second
// illuminant of the two-illuminant mix (phase-1 section 3): neutralizing the direct's
// warm slope double-blues the mix. The gains triple is renormalized to unit Rec.709
// luminance (a UNIFORM scale, ratio-preserving), so a pixel of the illuminant color
// keeps its display luminance — the approved medians cannot move. The correction is
// DISPLAY-side (the raw-bands product stays physical) and is tapered with named
// satpy-idiom gates: identity at/below the terminator gate (the approved dusk band
// byte-identical by construction), full correction across the sunrise band, identity
// again by full daytime (the LUT green dip is negligible up there anyway).

/// Reference CLOUD-TOP altitude (m) at which the low-sun illuminant color is sampled
/// from the transmittance LUT. Science-review band ~6-8 km: the mid/upper-tropospheric
/// deck the correction targets. LOWER clouds at low sun sit under a redder illuminant
/// than the reference (they keep a warm residual — matching the real product, which
/// leaves dawn stratus warm-tinted); higher anvils see slightly less.
pub const ILLUM_REF_CLOUD_ALT_M: f64 = 7_000.0;
/// Correction taper IN gate (deg): identity at/below (the amber terminator and the
/// whole approved below-horizon dusk band are byte-identical by construction).
pub const ILLUM_CORR_IN_LO_DEG: f64 = 2.0;
/// Correction taper IN gate (deg): the full correction is in place at/above this
/// elevation. The science-review guidance ("full correction ~8-10 deg") was written
/// for the full-white-balance form; the shipped GREEN-RESTORATION form preserves the
/// warm terminator axis by construction (equal R/B gains), so an earlier full
/// engagement cannot cool the amber — and the round-2 Michael real-dawn frame
/// (centre sun 3.4 deg, the reported defect band) measured only w = 0.16 under the
/// 8-deg gate, leaving the mauve CDO nearly uncorrected. 5.0 puts real sunrise
/// scenes (3-5 deg) under a meaningful-to-full correction while the <= 2 deg dusk
/// band stays exactly identity.
pub const ILLUM_CORR_IN_HI_DEG: f64 = 5.0;
/// Correction taper OUT gate (deg): the correction starts easing off here...
pub const ILLUM_CORR_OUT_LO_DEG: f64 = 20.0;
/// ...and is identity again at/above this elevation (full daytime byte-identical by
/// construction; the residual LUT gains up here are ~1 anyway).
pub const ILLUM_CORR_OUT_HI_DEG: f64 = 30.0;

/// The low-sun correction WEIGHT `[0,1]` at a sun elevation (deg): a smoothstep taper
/// in across [`ILLUM_CORR_IN_LO_DEG`]..[`ILLUM_CORR_IN_HI_DEG`], a plateau at `1.0`
/// through the sunrise band, and a smoothstep taper out across
/// [`ILLUM_CORR_OUT_LO_DEG`]..[`ILLUM_CORR_OUT_HI_DEG`]. Exactly `0.0` at/below the IN
/// gate and at/above the OUT gate (the byte-identity guarantees).
#[inline]
pub fn low_sun_illuminant_weight(sun_elev_deg: f64) -> f64 {
    atmosphere::smoothstep(ILLUM_CORR_IN_LO_DEG, ILLUM_CORR_IN_HI_DEG, sun_elev_deg)
        * (1.0 - atmosphere::smoothstep(ILLUM_CORR_OUT_LO_DEG, ILLUM_CORR_OUT_HI_DEG, sun_elev_deg))
}

/// LUT-DERIVED per-band gains of the low-sun illuminant correction at a sun elevation
/// (deg). `[1, 1, 1]` outside the correction band (weight 0). Inside: the direct-sun
/// transmittance triple `t` is sampled from OUR OWN LUT at [`ILLUM_REF_CLOUD_ALT_M`];
/// the GREEN gain restores `t_G` to the Rayleigh log-line between the measured `t_R`
/// and `t_B` (removing the Chappuis ozone green dip — the diagnosed khaki/mauve
/// driver); the triple is renormalized to unit Rec.709 luminance by a UNIFORM scale
/// (so the R/B warm axis is preserved EXACTLY and the illuminant's display luminance
/// is unchanged); and the result is blended toward identity by
/// [`low_sun_illuminant_weight`]. NEVER baked numbers: a retune of ozone/AOD/Rayleigh
/// flows through the LUT into these gains automatically (the CREFL
/// same-atmosphere-model property). See the module note for why the full white
/// balance was measured and rejected.
pub fn low_sun_illuminant_gains(sun_elev_deg: f64, luts: &AtmosphereLuts) -> [f64; 3] {
    let w = low_sun_illuminant_weight(sun_elev_deg);
    if w <= 0.0 {
        return [1.0, 1.0, 1.0];
    }
    let mu = (sun_elev_deg.to_radians()).sin();
    let t = atmosphere::sample_transmittance(
        &luts.transmittance,
        atmosphere::R_GROUND_M + ILLUM_REF_CLOUD_ALT_M,
        mu,
    );
    let degenerate = |v: f64| !v.is_finite() || v <= 0.0;
    if t.iter().any(|&c| degenerate(c)) {
        return [1.0, 1.0, 1.0]; // degenerate illuminant: no correction
    }
    // Rayleigh-expected green: the lambda^-4 log-line between the measured R and B.
    let ray = atmosphere::RAYLEIGH_SCATTERING;
    let a = (ray[1] - ray[0]) / (ray[2] - ray[0]);
    let t_g_rayleigh = ((1.0 - a) * t[0].ln() + a * t[2].ln()).exp();
    // Restore green no lower than measured (the correction only ever ADDS green back).
    let g_green = (t_g_rayleigh / t[1]).max(1.0);
    // Unit-luminance renormalization: a UNIFORM scale (ratio-preserving) that keeps
    // the corrected illuminant's Rec.709 luminance equal to the raw illuminant's.
    let y = |c: [f64; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    let y_raw = y(t);
    let y_corr = y([t[0], t[1] * g_green, t[2]]);
    if degenerate(y_raw) || degenerate(y_corr) {
        return [1.0, 1.0, 1.0];
    }
    let s = y_raw / y_corr;
    [
        1.0 + w * (s - 1.0),
        1.0 + w * (s * g_green - 1.0),
        1.0 + w * (s - 1.0),
    ]
}

/// Apply the low-sun illuminant correction to a composited top-of-atmosphere LINEAR
/// radiance, per band (gains on `L` == gains on `rho`, since `E_sun` is a per-channel
/// constant). ON-EARTH pixels only (`on_earth = false` returns the input unchanged —
/// the off-earth limb keeps its physical color). Called at the display seam of every
/// visible RGB path, on the FINAL composited radiance (surface + cloud + airlight),
/// immediately before the tonemap: [`shade_surface`] (geo clouds-off), the top-down
/// RGB path (`topdown.rs`), and the geo clouds-on composite (`clouds.rs`
/// `shade_cloud_pixel` — the one-line integration call site). The raw-bands /
/// reflectance products do NOT call this (physical, pre-display).
#[inline]
pub fn apply_low_sun_illuminant(
    l_toa: [f64; 3],
    on_earth: bool,
    sun_elev_deg: f64,
    luts: &AtmosphereLuts,
) -> [f64; 3] {
    if !on_earth {
        return l_toa;
    }
    let g = low_sun_illuminant_gains(sun_elev_deg, luts);
    [l_toa[0] * g[0], l_toa[1] * g[1], l_toa[2] * g[2]]
}

/// The PHYSICAL reflectance ceiling the bounded highlight shoulder maps to display
/// white (see [`soft_clip_highlight`]): `x_max = exposure * RHO_HIGHLIGHT_MAX` is the
/// largest exposure-applied reflectance the shoulder resolves — everything at/above it
/// pins to exactly `1.0`. Value `1.25` is the cross-case v0.1.4 QA calibration: it
/// preserves structure across a wider bright-cloud range while still allowing the
/// strongest cloud tops to reach white. This is what makes the shoulder
/// EXPOSURE-AWARE: the unbounded Reinhard wasted the display range asymptoting toward
/// 1.0 for inputs that can never occur, crushing the real cloud band (rho 0.6..0.9 at
/// exposure 1.6 collapsed to a 0.033 display delta — the "white square").
pub const RHO_HIGHLIGHT_MAX: f64 = 1.25;

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

// ── ABI SYNTHETIC-GREEN display mode (prototype; OFF by default) ────────────────
//
// Real GOES-R "true color" has NO green band: the display green is SYNTHESIZED as a
// weighted blend, G_display = 0.45*Red + 0.45*Blue + 0.10*veggie-NIR (Bah, Gunshor &
// Schmit 2018, the CIMSS "simple hybrid green"). An arithmetic consequence: any hue on
// the green-magenta axis is impossible in the real product — khaki (G excess over the
// R-B line) and mauve (G deficit) casts CANNOT occur, at any solar angle. Our G channel
// is a real radiative band, which is more physical but is exactly what lets the
// low-sun ozone green-dip show as khaki/mauve. This mode reproduces the product
// arithmetic (our G stands in for the 10% veggie-NIR term) as a DISPLAY-mode option
// for A/B judgment: the orchestrator decides default / option / drop. NOTE the
// G/B ~1.10-1.20 acceptance band of the real-GOES sunrise reference is itself a
// synthesized-green PRODUCT artifact (provenance, not physical cloud color).
//
// Mechanism: a process-global display switch (default OFF = byte-identical), set by
// the `render_frame` `synthetic-green=on` QA flag before rendering. A process-global
// (the `ir::tsk_fallback_engaged` precedent) rather than a `FrameContext` field /
// `OutputTransform` variant because the studio's settings + pickers match the enum
// exhaustively outside this pass's footprint; if the mode is adopted the switch can be
// promoted to a real studio-visible option. The WGSL twins mirror the math behind a
// module const (`SYNTHETIC_GREEN_MODE = 0.0` = off) — flip both together on adoption.

/// Synthesized-green weight of the RED reflectance (Bah et al. 2018).
pub const SYN_GREEN_W_RED: f64 = 0.45;
/// Synthesized-green weight of our native GREEN reflectance (stand-in for the 10%
/// veggie-NIR term of the real product).
pub const SYN_GREEN_W_GREEN: f64 = 0.10;
/// Synthesized-green weight of the BLUE reflectance (Bah et al. 2018).
pub const SYN_GREEN_W_BLUE: f64 = 0.45;

static SYNTHETIC_GREEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Enable/disable the ABI synthetic-green display mode process-wide (default OFF).
/// QA/prototype switch — see the module note above.
pub fn set_synthetic_green(on: bool) {
    SYNTHETIC_GREEN.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the ABI synthetic-green display mode is enabled (default `false`).
pub fn synthetic_green_enabled() -> bool {
    SYNTHETIC_GREEN.load(std::sync::atomic::Ordering::Relaxed)
}

/// Replace the green of a display reflectance triple with the ABI synthesized green
/// `G' = 0.45*R + 0.45*B + 0.10*G` (Bah et al. 2018; our G is the veggie stand-in).
/// Applied to the exposure-scaled reflectance BEFORE the highlight desaturation, i.e.
/// at the same point of the chain where the real product builds its green from the
/// (corrected) band reflectances. R and B are untouched.
#[inline]
pub fn synthesize_abi_green(rho: [f64; 3]) -> [f64; 3] {
    [
        rho[0],
        SYN_GREEN_W_RED * rho[0] + SYN_GREEN_W_GREEN * rho[1] + SYN_GREEN_W_BLUE * rho[2],
        rho[2],
    ]
}

/// Convert an internal reflectance factor `rho` (per band) to display `[0,1]` via
/// the selected output transform (twin of the WGSL `output_transform`). For the ABI
/// product path this is `stretch(desaturate(rho))` — the M2 twilight-pass display
/// transform (highlight desaturation on the reflectance vector, then the toe-lifted
/// per-channel sqrt stretch; see `atmosphere::desaturate_highlights` /
/// `atmosphere::abi_reflectance_stretch`). The debug path stays a plain sRGB gamma.
/// Reads the process-global synthetic-green switch and delegates to the pure
/// [`output_transform_with`] (tested directly, so no test ever toggles the global).
#[cfg_attr(not(test), allow(dead_code))]
fn apply_output_transform(
    rho: [f64; 3],
    transform: OutputTransform,
    softclip_knee: f64,
    softclip_max: f64,
) -> [f32; 3] {
    output_transform_with(
        rho,
        transform,
        softclip_knee,
        softclip_max,
        synthetic_green_enabled(),
    )
}

/// The PURE output transform behind [`apply_output_transform`]: `synthetic_green`
/// selects the ABI synthesized-green display mode explicitly (see
/// [`synthesize_abi_green`]). `false` reproduces the shipped transform byte-for-byte.
fn output_transform_with(
    rho: [f64; 3],
    transform: OutputTransform,
    softclip_knee: f64,
    softclip_max: f64,
    synthetic_green: bool,
) -> [f32; 3] {
    match transform {
        OutputTransform::AbiReflectance => {
            // Optional synthesized green FIRST (the real product computes its display
            // green from the corrected band reflectances before any stretch), then
            // highlight desaturation (M2 twilight pass), THEN the bounded highlight
            // soft-clip (bright tops keep structure; the desaturate-then-shoulder ORDER
            // is load-bearing — swapping it shifts the -2 deg amber anvils), THEN the
            // toe-lifted sqrt stretch. The soft-clip is strictly identity below its knee,
            // so the desaturated daytime/twilight below the knee is byte-unchanged.
            let rho = if synthetic_green {
                synthesize_abi_green(rho)
            } else {
                rho
            };
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

/// [`combine_aerial_veil`] with the opt-in post-lighting surface toe inserted at its
/// precise display seam: after view transmittance, before additive airlight. The
/// disabled path delegates to the established helper for byte-for-byte identity.
#[inline]
pub(crate) fn combine_aerial_veil_with_surface_toe(
    l_surf: [f64; 3],
    transmittance: [f64; 3],
    inscatter: [f64; 3],
    veil: f64,
    sun_elev_deg: f64,
    config: SurfacePostlightToeConfig,
    twilight: TwilightSurfaceRecoveryConfig,
) -> [f64; 3] {
    if config.is_identity() && twilight.is_identity() {
        return combine_aerial_veil(l_surf, transmittance, inscatter, veil);
    }
    let surface = [
        l_surf[0] * transmittance[0],
        l_surf[1] * transmittance[1],
        l_surf[2] * transmittance[2],
    ];
    let gain = combined_surface_recovery_gain(surface, sun_elev_deg, config, twilight);
    [
        surface[0] * gain + veil * inscatter[0],
        surface[1] * gain + veil * inscatter[1],
        surface[2] * gain + veil * inscatter[2],
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
    let surface_elevation_m = if ctx.terrain_atmosphere {
        (px.surface_elevation_m as f64).max(0.0)
    } else {
        0.0
    };
    let t_sun = atmosphere::sample_transmittance(
        &ctx.luts.transmittance,
        atmosphere::R_GROUND_M + surface_elevation_m + 1.0,
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
    let seg = atmosphere::ray_atmosphere_segment_to_surface(cam, view, surface_elevation_m);

    let mut l_surf = [0.0; 3];
    if px.is_water {
        // Cox-Munk wind-ruffled SUN GLINT + Fresnel SKY REFLECTION (M3), replacing the
        // M1 flat dark water (design section 5). Geometry in ECEF using the water
        // point's local up (from the ground intersection); the glint images the
        // disk-integrated solar irradiance (its angular extent comes from the Cox-Munk
        // slope PDF, widening with wind), attenuated to the surface (t_sun) through the
        // finite-disk fraction (disk) and any cloud/terrain shadow. Limb darkening is
        // already integrated into E_sun and is not applied as a second 0.832 dimming.
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
        let glint_scale = disk * shadow_raw;
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
        // the sun elevation. The physical direct term follows disk/Tsun/N.L without an
        // artificial appearance gate. The water-body albedo alone is retuned DOWN
        // toward [`WATER_ALBEDO_DAY_SCALE`] across the 0--12 degree surface-help ramp so
        // the owner-approved dark-ocean/distinct-glint contrast holds.
        let surface_t = atmosphere::smoothstep(
            SURFACE_HELP_ELEV_LO_DEG,
            SURFACE_HELP_ELEV_HI_DEG,
            px.sun_elev_deg as f64,
        );
        // Effective daylight albedo rescale relative to the already-applied
        // ctx.water_scale: 1.0 at/below the horizon -> DAY_SCALE/water_scale by 12 deg.
        let scale_ratio = if ctx.water_scale > 0.0 {
            1.0 + surface_t * (WATER_ALBEDO_DAY_SCALE / ctx.water_scale - 1.0)
        } else {
            1.0
        };
        let ndotl = dot(px.normal_enu, px.sun_enu).max(0.0);
        let e_direct = surface_direct_irradiance(e_sun, t_sun, disk, ndotl, shadow);
        for c in 0..3 {
            let l_glint = glint_rho * e_sun[c] / pi * t_sun[c] * glint_scale;
            // Skylight- and sunlight-lit water body (the Blue Marble water texel x the
            // surface-ramped water scale) + the sun glint + the Fresnel sky reflection.
            l_surf[c] = albedo[c] * scale_ratio / pi * (e_direct[c] + e_ambient[c])
                + l_glint
                + f_sky * l_sky[c];
        }
    } else {
        // Land: Lambertian direct sun (HGT slope N.L, penumbral-shadowed disk) + the
        // aperture-occluded SH ambient. Snow (if any) is already blended into `albedo`.
        let ndotl = dot(px.normal_enu, px.sun_enu).max(0.0);
        let e_direct = surface_direct_irradiance(e_sun, t_sun, disk, ndotl, shadow);
        for c in 0..3 {
            l_surf[c] = albedo[c] / pi * (e_direct[c] + e_ambient[c]);
        }

        // Operational-display corrections for LAND only. Both are scalar
        // gains on the surface signal, so they preserve colour ratios and cannot alter
        // water/glint or cloud radiance. The explicit identity config takes the exact
        // legacy path; the surface-help gate keeps it neutral at/below the horizon.
        let appearance_gain =
            land_appearance_gain(ctx.land_appearance, px.sun_elev_deg as f64, albedo);
        if appearance_gain != 1.0 {
            for v in &mut l_surf {
                *v *= appearance_gain;
            }
        }
    }

    // LAND daylight brightness lift (refinement pass, round 2): a modest ground-only
    // gain on the surface reflectance, neutral at/below the horizon and fully engaged
    // by 12 degrees (water excluded). Applied to `l_surf` BEFORE the
    // aerial-perspective veil, so only the ground signal brightens (not the additive
    // haze). A true-color display gain, distinct from the global exposure. See
    // [`LAND_DAY_GAIN`] / [`land_day_gain`].
    if !px.is_water && LAND_DAY_GAIN != 1.0 {
        let g = land_day_gain(px.sun_elev_deg as f64);
        for v in &mut l_surf {
            *v *= g;
        }
    }

    // GROUND LIFT (top-down/basemap appearance pass): a sun-gated daylight brightness lift
    // on the WHOLE surface radiance — land AND water — toward real-GOES ground levels (the
    // reported ground was too dark). Applied BEFORE the aerial-perspective veil (only the
    // ground signal brightens, not the additive haze) and BEFORE the cloud composite (the
    // cloud radiance is not lifted). The 0--12 degree surface-help ramp leaves it exactly
    // `1.0` at/below the horizon. `ground_lift = 1.0` is an exact no-op.
    if ground_lift != 1.0 {
        let g = ground_day_lift(px.sun_elev_deg as f64, ground_lift);
        for v in &mut l_surf {
            *v *= g;
        }
    }

    // Aerial perspective: raymarch the shell from atmosphere entry to the ground.
    // The experiment targets terrain only. Dark ocean, Fresnel sky reflection, and
    // Cox-Munk glint stay on the reviewed path exactly.
    let postlight_toe = if px.is_water {
        SurfacePostlightToeConfig::off()
    } else {
        ctx.surface_postlight_toe
    };
    let twilight_recovery = if px.is_water {
        TwilightSurfaceRecoveryConfig::off()
    } else {
        ctx.twilight_surface_recovery
    };
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
        // SUNRISE veil ramp (low-sun visible pass): scale down the additive in-scatter
        // haze laid over the surface (a Rayleigh correction; the terminator band is
        // untouched because the scale is 1.0 at/below VEIL_TERMINATOR_ELEV_DEG). The
        // surface transmittance is left intact. Off-earth limb in-scatter (above) is
        // never scaled.
        let veil = if ctx.atmosphere_correction {
            aerial_veil_scale(px.sun_elev_deg as f64)
        } else {
            1.0
        };
        l_toa = combine_aerial_veil_with_surface_toe(
            l_surf,
            sc.transmittance,
            sc.inscatter,
            veil,
            px.sun_elev_deg as f64,
            postlight_toe,
            twilight_recovery,
        );
    } else if !postlight_toe.is_identity() || !twilight_recovery.is_identity() {
        l_toa = combine_aerial_veil_with_surface_toe(
            l_surf,
            [1.0; 3],
            [0.0; 3],
            1.0,
            px.sun_elev_deg as f64,
            postlight_toe,
            twilight_recovery,
        );
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
    radiance_to_rgba_softclip(
        l_toa,
        transform,
        exposure,
        CLOUD_SOFTCLIP_KNEE,
        RHO_HIGHLIGHT_MAX,
    )
}

/// Like [`radiance_to_rgba`] but with explicit highlight soft-clip controls (see
/// [`CLOUD_SOFTCLIP_KNEE`], [`RHO_HIGHLIGHT_MAX`], and [`soft_clip_highlight`]).
/// [`radiance_to_rgba`] delegates here with the baked defaults, so the plain surface
/// path and the cloud/top-down RGB paths all share ONE tonemap. `knee = 1.0` disables
/// the shoulder (the old hard clamp above 1.0); `highlight_max` is the physical
/// reflectance-factor ceiling mapped to white.
///
/// EXPOSURE-AWARE SHOULDER BOUND: the shoulder's input ceiling is derived INTERNALLY
/// here as `x_max = gain * highlight_max` — the exposure gain is already applied to the
/// reflectance, so the largest input the shoulder can ever see is the exposure times
/// the selected physical reflectance ceiling. A non-finite or non-positive override
/// falls back to [`RHO_HIGHLIGHT_MAX`].
pub fn radiance_to_rgba_softclip(
    l_toa: [f64; 3],
    transform: OutputTransform,
    exposure: f64,
    softclip_knee: f64,
    highlight_max: f64,
) -> [f32; 4] {
    radiance_to_rgba_softclip_with_synthetic_green(
        l_toa,
        transform,
        exposure,
        softclip_knee,
        highlight_max,
        synthetic_green_enabled(),
    )
}

/// Per-render variant of [`radiance_to_rgba_softclip`]. The explicit flag is used by
/// the high-level API and Studio so Sensor Fast Gray can guarantee native broad-RGB
/// channels without changing global state; the original function remains the
/// backwards-compatible low-level QA wrapper.
pub fn radiance_to_rgba_softclip_with_synthetic_green(
    l_toa: [f64; 3],
    transform: OutputTransform,
    exposure: f64,
    softclip_knee: f64,
    highlight_max: f64,
    synthetic_green: bool,
) -> [f32; 4] {
    let e_sun = SOLAR_IRRADIANCE_RGB;
    let gain = if exposure.is_finite() && exposure > 0.0 {
        exposure
    } else {
        1.0
    };
    let highlight_max = if highlight_max.is_finite() && highlight_max > 0.0 {
        highlight_max
    } else {
        RHO_HIGHLIGHT_MAX
    };
    let mut rho = [0.0; 3];
    for c in 0..3 {
        rho[c] = gain * std::f64::consts::PI * l_toa[c] / e_sun[c];
    }
    let out = output_transform_with(
        rho,
        transform,
        softclip_knee,
        gain * highlight_max,
        synthetic_green,
    );
    [out[0], out[1], out[2], 1.0]
}

/// Convert a top-of-atmosphere linear radiance to the RAW per-channel REFLECTANCE
/// FACTOR `rho = pi * L / E_sun`, clamped to `[0, 1]` — the pre-tonemap quantity the
/// Python binding's `render_rgb_reflectance` returns for building custom RGB / operating
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
    match surface_toa_radiance(ctx, px, 1.0, ctx.ground_day_lift) {
        None => [0.0, 0.0, 0.0, 0.0],
        Some(l_toa) => {
            // Low-sun illuminant correction at the display seam (on-earth only; the
            // limb keeps its physical color). Identity outside the 2-30 deg band.
            let l = apply_low_sun_illuminant(l_toa, px.on_earth, px.sun_elev_deg as f64, ctx.luts);
            radiance_to_rgba_softclip_with_synthetic_green(
                l,
                ctx.output_transform,
                ctx.exposure,
                ctx.cloud_softclip_knee,
                ctx.cloud_highlight_max,
                ctx.synthetic_green,
            )
        }
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
    fn shipped_visible_display_calibration_is_rc_preset() {
        assert_eq!(DEFAULT_EXPOSURE, 1.5);
        assert_eq!(GROUND_DAY_LIFT, 1.10);
        assert_eq!(CLOUD_SOFTCLIP_KNEE, 0.65);
        assert_eq!(RHO_HIGHLIGHT_MAX, 1.25);
    }

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
    fn analytic_lambert_round_trips_linear_reflectance_and_ndotl_units() {
        let albedo = [0.20; 3];
        for (ndotl, expected) in [(1.0, 0.20f32), (0.5, 0.10), (0.25, 0.05)] {
            let radiance = lambert_surface_radiance(albedo, SOLAR_IRRADIANCE_RGB, ndotl, [0.0; 3]);
            let rho = reflectance_from_radiance(radiance);
            for channel in rho {
                assert!(
                    (channel - expected).abs() < 2.0e-7,
                    "N.L={ndotl}: rho={channel}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn disk_integrated_surface_sun_is_unit_normalized_and_keeps_visibility_fraction() {
        let full = surface_direct_irradiance(SOLAR_IRRADIANCE_RGB, [1.0; 3], 1.0, 1.0, 1.0);
        assert_eq!(full, SOLAR_IRRADIANCE_RGB);

        let partial = surface_direct_irradiance(SOLAR_IRRADIANCE_RGB, [1.0; 3], 0.5, 0.25, 0.8);
        for c in 0..3 {
            assert!((partial[c] - SOLAR_IRRADIANCE_RGB[c] * 0.5 * 0.25 * 0.8).abs() < 1e-12);
            assert_ne!(
                full[c],
                SOLAR_IRRADIANCE_RGB[c] * atmosphere::LIMB_DARKENING_DISK_AVG,
                "disk-integrated irradiance must not be dimmed by the centre-relative disk average"
            );
        }
    }

    #[test]
    fn analytic_lambert_surface_and_cloud_limits_preserve_reflectance_units() {
        let surface = lambert_surface_radiance([0.20; 3], SOLAR_IRRADIANCE_RGB, 1.0, [0.0; 3]);
        let cloud = lambert_surface_radiance([0.60; 3], SOLAR_IRRADIANCE_RGB, 1.0, [0.0; 3]);
        let surface_rho = reflectance_from_radiance(surface);
        assert!(surface_rho.into_iter().all(|v| (v - 0.20).abs() < 2.0e-7));

        let clear =
            crate::topdown::composite_topdown_front_column(surface, [0.0; 3], 1.0, [0.0; 3], 1.0);
        let opaque =
            crate::topdown::composite_topdown_front_column(surface, cloud, 0.0, [0.0; 3], 1.0);
        assert!(
            reflectance_from_radiance(clear)
                .into_iter()
                .all(|v| (v - 0.20).abs() < 2.0e-7),
            "clear-cloud limit must preserve the surface reflectance"
        );
        assert!(
            reflectance_from_radiance(opaque)
                .into_iter()
                .all(|v| (v - 0.60).abs() < 2.0e-7),
            "opaque-cloud limit must preserve cloud radiance units and hide surface"
        );
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
            ground_day_lift: GROUND_DAY_LIFT,
            cloud_softclip_knee: CLOUD_SOFTCLIP_KNEE,
            cloud_highlight_max: RHO_HIGHLIGHT_MAX,
            synthetic_green: false,
            atmosphere_correction: true,
            terrain_atmosphere: true,
            land_appearance: LandAppearanceConfig::identity(),
            surface_postlight_toe: SurfacePostlightToeConfig::off(),
            twilight_surface_recovery: TwilightSurfaceRecoveryConfig::off(),
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
        // The raw RGB-reflectance conversion is rho = pi*L/E_sun with
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

        // Land-day-gain is a member of the daytime ramp family; at the neutral
        // day_value 1.0 the ramp is exactly 1.0 at every elevation (no daytime change).
        // The VEIL is no longer a family member (the low-sun sunrise ramp — tested in
        // sunrise_veil_ramp_extends_the_dehaze_into_the_sunrise_band); here we pin only
        // that at/above the daytime gate it still equals the family value (daytime
        // byte-identity) and that the terminator band keeps the full physical veil.
        for &elev in &[-10.0f64, 0.0, 10.0, 20.0, 25.0, 40.0, 60.0, 90.0] {
            assert_eq!(
                day_lerp_ramp(elev, 1.0),
                1.0,
                "neutral daytime ramp must be identity at elev {elev}"
            );
            assert_eq!(
                land_day_gain(elev),
                surface_help_ramp(elev, LAND_DAY_GAIN),
                "the land day gain is the surface-help ramp at LAND_DAY_GAIN"
            );
            if elev >= AERIAL_VEIL_ELEV_HI_DEG {
                assert_eq!(
                    aerial_veil_scale(elev),
                    day_lerp_ramp(elev, AERIAL_VEIL_DAY_SCALE),
                    "full daytime veil must still equal the day-ramp value (byte-identity)"
                );
            }
            if elev <= SURFACE_HELP_ELEV_LO_DEG {
                assert_eq!(land_day_gain(elev), 1.0, "night land gain untouched");
            }
            if elev <= VEIL_TERMINATOR_ELEV_DEG {
                assert_eq!(aerial_veil_scale(elev), 1.0, "terminator veil untouched");
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

        // Day gain brightens at high sun (> 1), is a no-op at the horizon, and never overshoots.
        assert!(
            land_day_gain(50.0) > 1.0,
            "high-sun land day gain must brighten"
        );
        assert_eq!(land_day_gain(0.0), 1.0, "horizon land is unchanged");
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
        let x_max = DEFAULT_EXPOSURE * RHO_HIGHLIGHT_MAX;
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
        for &x in &[0.8, 1.0, 1.15, x_max - 1e-6] {
            let y = soft_clip_highlight(x, knee, x_max);
            assert!(y > prev, "monotone increasing at {x}: {y} <= {prev}");
            assert!(y < 1.0, "below white short of x_max at {x}: {y}");
            prev = y;
        }
        assert!(
            soft_clip_highlight(1.15, knee, x_max) > soft_clip_highlight(1.0, knee, x_max),
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
    fn soft_clip_contrast_floor_recovers_the_cloud_band_at_high_exposure_override() {
        // THE WHITE-SQUARE FIX, quantified. At exposure 1.6 the cloud-texture band
        // rho 0.6..0.85 lands at x = 0.96..1.36. The old unbounded Reinhard collapsed
        // it to a 0.033 display delta (the flat white square); the bounded shoulder
        // must keep >= 0.08 of LINEAR separation across the band (shoulder domain) and
        // measurably widen the final display (post-sqrt) delta.
        //
        // NOTE (feasibility, documented in notes/ws2-tonemap-notes.md): the 0.08 floor
        // is intentionally asserted in the linear shoulder domain; after the ABI sqrt
        // stretch the shipped 0.65-knee / 1.25-ceiling curve uses a 0.045 display floor.
        let knee = CLOUD_SOFTCLIP_KNEE;
        let exposure = 1.6;
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
        // The old ~1.07 peak now retains detail below the raised 1.25 ceiling; only a
        // stronger super-bright top at/above the shipped physical ceiling reaches white.
        let d_peak = radiance_to_rgba(l_of(1.07), OutputTransform::AbiReflectance, exposure)[0];
        assert!(d_peak < 1.0, "the ordinary bright peak keeps structure");
        let d_super = radiance_to_rgba(l_of(1.26), OutputTransform::AbiReflectance, exposure)[0];
        assert_eq!(d_super, 1.0, "a super-bright top still saturates to white");
    }

    #[test]
    fn highlight_ceiling_override_preserves_bright_structure_and_has_safe_fallbacks() {
        let e_sun = SOLAR_IRRADIANCE_RGB;
        let rho = 1.27;
        let l = [
            rho * e_sun[0] / std::f64::consts::PI,
            rho * e_sun[1] / std::f64::consts::PI,
            rho * e_sun[2] / std::f64::consts::PI,
        ];
        let default = radiance_to_rgba_softclip(
            l,
            OutputTransform::AbiReflectance,
            DEFAULT_EXPOSURE,
            CLOUD_SOFTCLIP_KNEE,
            RHO_HIGHLIGHT_MAX,
        );
        assert_eq!(default[0], 1.0, "the shipped ceiling clips this peak");

        let raised = radiance_to_rgba_softclip(
            l,
            OutputTransform::AbiReflectance,
            DEFAULT_EXPOSURE,
            CLOUD_SOFTCLIP_KNEE,
            1.50,
        );
        assert!(
            raised[0] < 1.0 && raised[0] > 0.0,
            "a raised physical ceiling should retain highlight structure: {raised:?}"
        );

        for bad in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
            assert_eq!(
                radiance_to_rgba_softclip(
                    l,
                    OutputTransform::AbiReflectance,
                    DEFAULT_EXPOSURE,
                    CLOUD_SOFTCLIP_KNEE,
                    bad,
                ),
                default,
                "invalid ceiling {bad} must use the baked calibration"
            );
        }
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
    fn ground_day_lift_uses_surface_help_ramp_and_is_neutral_at_one() {
        // The neutral ground_lift = 1.0 is exactly 1.0 at every elevation (identity no-op).
        for &elev in &[-10.0f64, 0.0, 10.0, 20.0, 30.0, 40.0, 60.0, 90.0] {
            assert_eq!(ground_day_lift(elev, 1.0), 1.0, "neutral no-op at {elev}");
            // It is the surface-help ramp family (only day_value differs).
            assert_eq!(
                ground_day_lift(elev, GROUND_DAY_LIFT),
                surface_help_ramp(elev, GROUND_DAY_LIFT)
            );
        }
        // Sun-gated: exactly 1.0 at/below the horizon; target reached by 12 degrees.
        assert_eq!(
            ground_day_lift(SURFACE_HELP_ELEV_LO_DEG, GROUND_DAY_LIFT),
            1.0
        );
        assert_eq!(ground_day_lift(0.0, GROUND_DAY_LIFT), 1.0);
        assert!(
            (ground_day_lift(SURFACE_HELP_ELEV_HI_DEG, GROUND_DAY_LIFT) - GROUND_DAY_LIFT).abs()
                < 1e-12
        );
        assert_eq!(
            ground_day_lift(90.0, GROUND_DAY_LIFT),
            GROUND_DAY_LIFT,
            "the shipped ground calibration reaches its reviewed daylight lift"
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
    fn land_appearance_default_is_shipped_and_identity_is_explicit() {
        let shipped = LandAppearanceConfig::default();
        assert_eq!(shipped, LandAppearanceConfig::shipped());
        assert!(shipped.sza_normalization);
        assert!(shipped.dark_toe);
        assert!(!shipped.is_identity());
        assert!(land_appearance_gain(shipped, 33.0, [0.02, 0.04, 0.01]) > 1.0);

        let identity = LandAppearanceConfig::identity();
        assert!(identity.is_identity());
        assert!(!identity.sza_normalization);
        assert!(!identity.dark_toe);
        assert_eq!(identity.sza_max_gain, LAND_SZA_MAX_GAIN);
        assert_eq!(identity.dark_toe_knee, LAND_DARK_TOE_KNEE);
        assert_eq!(identity.dark_toe_gamma, LAND_DARK_TOE_GAMMA);
        assert_eq!(identity.dark_toe_max_gain, LAND_DARK_TOE_MAX_GAIN);
        assert_eq!(
            land_appearance_gain(identity, 33.0, [0.02, 0.04, 0.01]),
            1.0
        );
    }

    #[test]
    fn land_sza_normalization_is_night_safe_reference_neutral_and_bounded() {
        for &elev in &[-10.0, 0.0] {
            assert_eq!(
                land_sza_normalization_gain(elev, LAND_SZA_MAX_GAIN),
                1.0,
                "night/horizon must be exact identity at {elev}"
            );
        }
        for &elev in &[LAND_SZA_REFERENCE_ELEV_DEG, 75.0, 90.0] {
            assert_eq!(
                land_sza_normalization_gain(elev, LAND_SZA_MAX_GAIN),
                1.0,
                "reference/high sun must be exact identity at {elev}"
            );
        }
        let moderate = land_sza_normalization_gain(33.0, LAND_SZA_MAX_GAIN);
        assert!(moderate > 1.0 && moderate <= LAND_SZA_MAX_GAIN);
        for &elev in &[5.0, 8.0, 10.0, 12.0] {
            let gain = land_sza_normalization_gain(elev, LAND_SZA_MAX_GAIN);
            assert!(
                gain > 1.0 && gain <= LAND_SZA_MAX_GAIN,
                "low-sun gain at {elev}"
            );
        }
        assert_eq!(land_sza_normalization_gain(33.0, 1.0), 1.0);
        assert!(land_sza_normalization_gain(30.0, 1.2) <= 1.2);
        assert!(land_sza_normalization_gain(30.0, f64::NAN).is_finite());
    }

    #[test]
    fn land_dark_toe_preserves_black_knee_high_values_and_night() {
        let gain = |rgb, elev| {
            land_dark_toe_gain(
                rgb,
                elev,
                LAND_DARK_TOE_KNEE,
                LAND_DARK_TOE_GAMMA,
                LAND_DARK_TOE_MAX_GAIN,
            )
        };
        assert_eq!(gain([0.0; 3], 40.0), 1.0, "black stays black");
        assert_eq!(
            gain([LAND_DARK_TOE_KNEE; 3], 40.0),
            1.0,
            "the knee is the identity"
        );
        assert_eq!(gain([0.4; 3], 40.0), 1.0, "bright land is unchanged");
        assert_eq!(gain([0.02, 0.04, 0.01], 0.0), 1.0, "night is unchanged");
        assert!(
            gain([0.02, 0.04, 0.01], 10.0) > 1.0,
            "low-sun toe is active"
        );
        let day = gain([0.02, 0.04, 0.01], 40.0);
        assert!(day > 1.0 && day <= LAND_DARK_TOE_MAX_GAIN);
        let mid = land_dark_toe_gain([0.04; 3], 40.0, 0.08, 0.65, 4.0);
        let q = 0.08 * (0.04f64 / 0.08).powf(0.65);
        let expected = ((q * 0.5 + 0.04 * 0.5) / 0.04).clamp(1.0, 4.0);
        assert!(
            (mid - expected).abs() < 1.0e-12,
            "the toe must use the documented smoothstep-blended target"
        );
        assert_eq!(
            land_dark_toe_gain([0.02; 3], 40.0, 0.08, 1.0, 2.0),
            1.0,
            "gamma one is an explicit identity"
        );
    }

    #[test]
    fn surface_postlight_toe_is_default_off_bounded_and_uses_recommended_grid() {
        let off = SurfacePostlightToeConfig::default();
        assert!(off.is_identity());
        assert_eq!(off.knee, 0.18);
        assert_eq!(off.gamma, 0.80);
        assert_eq!(off.max_gain, 1.35);

        let rho = 0.035;
        let surface = [
            rho * SOLAR_IRRADIANCE_RGB[0] / std::f64::consts::PI,
            rho * SOLAR_IRRADIANCE_RGB[1] / std::f64::consts::PI,
            rho * SOLAR_IRRADIANCE_RGB[2] / std::f64::consts::PI,
        ];
        assert_eq!(surface_postlight_toe_gain(surface, 25.0, off), 1.0);
        for knee in [0.15, 0.18, 0.22] {
            for gamma in [0.75, 0.80, 0.85] {
                for max_gain in [1.25, 1.35, 1.45] {
                    let config = SurfacePostlightToeConfig {
                        enabled: true,
                        knee,
                        gamma,
                        max_gain,
                    };
                    let gain = surface_postlight_toe_gain(surface, 25.0, config);
                    assert!(
                        gain > 1.0 && gain <= max_gain,
                        "grid gain {gain} outside 1..={max_gain} for {config:?}"
                    );
                    assert_eq!(surface_postlight_toe_gain(surface, 0.0, config), 1.0);
                    assert_eq!(surface_postlight_toe_gain(surface, -6.0, config), 1.0);
                    assert!(surface_postlight_toe_gain(surface, 28.0, config) > 1.0);
                }
            }
        }
    }

    #[test]
    fn surface_twilight_weight_covers_civil_twilight_and_fades_out_by_12_degrees() {
        assert_eq!(surface_twilight_weight(-7.0), 0.0);
        assert_eq!(surface_twilight_weight(-6.0), 0.0);
        assert!(surface_twilight_weight(-3.0) > 0.0);
        assert!(surface_twilight_weight(-3.0) < 1.0);
        assert_eq!(surface_twilight_weight(0.0), 1.0);
        assert_eq!(surface_twilight_weight(4.0), 1.0);
        assert!(surface_twilight_weight(8.0) > 0.0);
        assert!(surface_twilight_weight(8.0) < 1.0);
        assert_eq!(surface_twilight_weight(12.0), 0.0);
        assert_eq!(surface_twilight_weight(28.0), 0.0);
        assert_eq!(surface_twilight_weight(40.0), 0.0);
    }

    #[test]
    fn twilight_surface_recovery_config_default_is_off_while_shipped_profile_is_low_sun_only() {
        let off = TwilightSurfaceRecoveryConfig::default();
        assert!(off.is_identity());
        assert_eq!(off.knee, 0.30);
        assert_eq!(off.gamma, 0.50);
        assert_eq!(off.max_gain, 4.0);
        let on = TwilightSurfaceRecoveryConfig::shipped();
        assert_eq!(
            on,
            TwilightSurfaceRecoveryConfig {
                enabled: true,
                ..off
            }
        );
        let rho = 0.02;
        let surface = [
            rho * SOLAR_IRRADIANCE_RGB[0] / std::f64::consts::PI,
            rho * SOLAR_IRRADIANCE_RGB[1] / std::f64::consts::PI,
            rho * SOLAR_IRRADIANCE_RGB[2] / std::f64::consts::PI,
        ];
        assert_eq!(twilight_surface_recovery_gain(surface, -6.0, on), 1.0);
        assert!(twilight_surface_recovery_gain(surface, -3.0, on) > 1.0);
        assert!(twilight_surface_recovery_gain(surface, 0.0, on) > 1.0);
        assert_eq!(twilight_surface_recovery_gain(surface, 12.0, on), 1.0);
        assert_eq!(twilight_surface_recovery_gain(surface, 28.0, on), 1.0);
        assert_eq!(twilight_surface_recovery_gain(surface, 40.0, on), 1.0);
        let legacy = SurfacePostlightToeConfig {
            enabled: true,
            ..SurfacePostlightToeConfig::default()
        };
        assert_eq!(
            combined_surface_recovery_gain(surface, 4.0, legacy, on),
            surface_postlight_toe_gain(surface, 4.0, legacy)
                .max(twilight_surface_recovery_gain(surface, 4.0, on))
        );
    }

    #[test]
    fn surface_postlight_toe_sits_after_view_transmittance_before_airlight() {
        let config = SurfacePostlightToeConfig {
            enabled: true,
            ..SurfacePostlightToeConfig::default()
        };
        let l_surf = [8.0, 9.0, 7.0];
        let trans = [0.75, 0.70, 0.65];
        let inscatter = [0.4, 0.5, 0.6];
        let veil = 0.4;
        let surface = [
            l_surf[0] * trans[0],
            l_surf[1] * trans[1],
            l_surf[2] * trans[2],
        ];
        let gain = surface_postlight_toe_gain(surface, 20.0, config);
        let adjusted = combine_aerial_veil_with_surface_toe(
            l_surf,
            trans,
            inscatter,
            veil,
            20.0,
            config,
            TwilightSurfaceRecoveryConfig::off(),
        );
        for c in 0..3 {
            assert!((adjusted[c] - (surface[c] * gain + veil * inscatter[c])).abs() < 1e-12);
        }
        assert_eq!(
            combine_aerial_veil_with_surface_toe(
                l_surf,
                trans,
                inscatter,
                veil,
                20.0,
                SurfacePostlightToeConfig::off(),
                TwilightSurfaceRecoveryConfig::off(),
            ),
            combine_aerial_veil(l_surf, trans, inscatter, veil),
            "disabled experiment must preserve the established composite exactly"
        );
    }

    #[test]
    fn postlight_and_twilight_recovery_brighten_land_only_and_preserve_ocean_glint() {
        let (mut ctx, sun) = nadir_surface_pixel(4.0);
        let pixel = |is_water| SurfacePixel {
            on_earth: true,
            base_srgb: [0.08, 0.10, 0.06],
            normal_enu: [0.0, 0.0, 1.0],
            sun_enu: [sun[0] as f32, sun[1] as f32, sun[2] as f32],
            sun_elev_deg: 4.0,
            is_water,
            view_dir: nadir_view(),
            wind_speed: 5.0,
            ..Default::default()
        };
        let sum = |v: [f64; 3]| v.into_iter().sum::<f64>();
        let land_base = surface_toa_radiance(&ctx, &pixel(false), 1.0, 1.0).unwrap();
        let water_base = surface_toa_radiance(&ctx, &pixel(true), 1.0, 1.0).unwrap();
        ctx.surface_postlight_toe = SurfacePostlightToeConfig {
            enabled: true,
            ..SurfacePostlightToeConfig::default()
        };
        ctx.twilight_surface_recovery = TwilightSurfaceRecoveryConfig::shipped();
        let land_adjusted = surface_toa_radiance(&ctx, &pixel(false), 1.0, 1.0).unwrap();
        let water_adjusted = surface_toa_radiance(&ctx, &pixel(true), 1.0, 1.0).unwrap();
        assert!(sum(land_adjusted) > sum(land_base));
        assert_eq!(
            water_adjusted, water_base,
            "post-light terrain recovery must not alter dark ocean or glint"
        );
    }

    #[test]
    fn shipped_land_appearance_brightens_only_daytime_land() {
        let (mut ctx, sun) = nadir_surface_pixel(33.0);
        let pixel = |is_water| SurfacePixel {
            on_earth: true,
            base_srgb: [0.19, 0.26, 0.12],
            normal_enu: [0.0, 0.0, 1.0],
            sun_enu: [sun[0] as f32, sun[1] as f32, sun[2] as f32],
            sun_elev_deg: 33.0,
            is_water,
            view_dir: nadir_view(),
            ..Default::default()
        };
        let sum = |v: [f64; 3]| v.into_iter().sum::<f64>();

        let land_base = surface_toa_radiance(&ctx, &pixel(false), 1.0, 1.0).unwrap();
        let water_base = surface_toa_radiance(&ctx, &pixel(true), 1.0, 1.0).unwrap();
        ctx.land_appearance = LandAppearanceConfig::default();
        let land_adjusted = surface_toa_radiance(&ctx, &pixel(false), 1.0, 1.0).unwrap();
        let water_adjusted = surface_toa_radiance(&ctx, &pixel(true), 1.0, 1.0).unwrap();
        assert!(sum(land_adjusted) > sum(land_base));
        assert_eq!(
            water_adjusted, water_base,
            "land appearance controls must not alter ocean/glint"
        );
    }

    #[test]
    fn ground_lift_brightens_day_and_low_sun_surface_but_not_night() {
        const TEST_GROUND_LIFT: f64 = 1.6;
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
            let lifted = surface_toa_radiance(&day_ctx, &px, 1.0, TEST_GROUND_LIFT).expect("earth");
            assert!(
                sum(lifted) > sum(base) + 1e-6,
                "ground lift must brighten the daytime {} surface: {} !> {}",
                if is_water { "water" } else { "land" },
                sum(lifted),
                sum(base)
            );
        }
        // Low sun (5 deg): the surface-help ramp is active, so the user control must
        // materially brighten terrain instead of silently doing nothing.
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
        let lifted = surface_toa_radiance(&twi_ctx, &px, 1.0, TEST_GROUND_LIFT).expect("earth");
        assert!(sum(lifted) > sum(base), "ground lift must work at low sun");

        let (night_ctx, night_sun) = nadir_surface_pixel(-5.0);
        let night_px = SurfacePixel {
            sun_enu: [
                night_sun[0] as f32,
                night_sun[1] as f32,
                night_sun[2] as f32,
            ],
            sun_elev_deg: -5.0,
            ..px
        };
        assert_eq!(
            surface_toa_radiance(&night_ctx, &night_px, 1.0, 1.0),
            surface_toa_radiance(&night_ctx, &night_px, 1.0, TEST_GROUND_LIFT),
            "ground lift must remain a no-op at night"
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
    fn water_low_sun_receives_direct_light_and_night_is_shadow_invariant() {
        // Physical water direct sunlight now follows the disk/Tsun/N.L terms instead of
        // an artificial 20-degree display gate, so cloud shadows work at low sun.
        for &elev in &[5.0f32, 12.0, 20.0] {
            let (ctx, px) = day_water_pixel(elev);
            let lit = surface_toa_radiance(&ctx, &px, 1.0, 1.0).expect("earth");
            let occluded = surface_toa_radiance(&ctx, &px, 0.0, 1.0).expect("earth");
            assert!(
                lit.iter().sum::<f64>() > occluded.iter().sum::<f64>(),
                "shadow at {elev}"
            );
        }
        let (ctx, px) = day_water_pixel(-5.0);
        assert_eq!(
            surface_toa_radiance(&ctx, &px, 1.0, 1.0),
            surface_toa_radiance(&ctx, &px, 0.0, 1.0),
            "night water has no direct term to shadow"
        );
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
    fn effective_cloud_shadow_is_elevation_independent_floored_and_monotone() {
        for &elev in &[-10.0f64, 0.0, 5.0, 20.0, 50.0, 90.0] {
            assert_eq!(effective_cloud_shadow(0.0, elev), CLOUD_SHADOW_FLOOR);
            assert_eq!(effective_cloud_shadow(1.0, elev), 1.0);
            let mut prev = -1.0;
            for &s in &[0.0f64, 0.25, 0.5, 0.75, 1.0] {
                let e = effective_cloud_shadow(s, elev);
                assert!(e > prev, "monotone in shadow at {s}, elev {elev}");
                assert!((CLOUD_SHADOW_FLOOR..=1.0).contains(&e));
                prev = e;
            }
        }
    }

    #[test]
    fn standalone_cloud_layer_shadow_is_neutral_at_night_and_full_by_twelve_degrees() {
        for &elev in &[-90.0f64, -10.0, 0.0] {
            assert_eq!(effective_cloud_shadow_layer(0.0, elev), 1.0);
            assert_eq!(effective_cloud_shadow_layer(0.5, elev), 1.0);
        }
        for &raw in &[0.0f64, 0.25, 0.5, 1.0] {
            assert_eq!(
                effective_cloud_shadow_layer(raw, SURFACE_HELP_ELEV_HI_DEG),
                effective_cloud_shadow(raw, SURFACE_HELP_ELEV_HI_DEG)
            );
        }
        let at_five = effective_cloud_shadow_layer(0.0, 5.0);
        assert!(at_five > CLOUD_SHADOW_FLOOR && at_five < 1.0);
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

    // ── low-sun visible pass (phase 2): sunrise veil ramp, LUT-derived illuminant
    // correction, ABI synthetic-green prototype ──

    #[test]
    fn low_sun_constants_lock() {
        // The named satpy/CREFL-idiom gates of the low-sun pass (twilight-tuning
        // constants-lock pattern): changing any of these renegotiates the approved
        // dusk/daytime byte-identity guarantees and must be a conscious decision.
        assert_eq!(VEIL_TERMINATOR_ELEV_DEG, 2.0);
        assert_eq!(VEIL_SUNRISE_ELEV_HI_DEG, 16.0);
        assert_eq!(ILLUM_REF_CLOUD_ALT_M, 7_000.0);
        assert_eq!(ILLUM_CORR_IN_LO_DEG, 2.0);
        assert_eq!(ILLUM_CORR_IN_HI_DEG, 5.0);
        assert_eq!(ILLUM_CORR_OUT_LO_DEG, 20.0);
        assert_eq!(ILLUM_CORR_OUT_HI_DEG, 30.0);
        assert_eq!(SYN_GREEN_W_RED, 0.45);
        assert_eq!(SYN_GREEN_W_GREEN, 0.10);
        assert_eq!(SYN_GREEN_W_BLUE, 0.45);
        // The synthesized-green weights are a convex combination (sum 1), so a grey
        // stays grey through the synthesis.
        assert!((SYN_GREEN_W_RED + SYN_GREEN_W_GREEN + SYN_GREEN_W_BLUE - 1.0).abs() < 1e-15);
    }

    #[test]
    fn sunrise_veil_ramp_extends_the_dehaze_into_the_sunrise_band() {
        // Terminator band (<= +2 deg): the FULL physical veil — the approved dusk
        // sweep 0/-2/-4 is byte-identical by construction.
        for &e in &[-6.0f64, -4.0, -2.0, 0.0, 1.0, VEIL_TERMINATOR_ELEV_DEG] {
            assert_eq!(aerial_veil_scale(e), 1.0, "full veil at elev {e}");
        }
        // Full daytime de-haze in place at/above the sunrise HI and held from there up
        // — at/above AERIAL_VEIL_ELEV_HI_DEG this equals the pre-fix ramp value, so
        // daytime (sun 30+) is byte-identical.
        for &e in &[VEIL_SUNRISE_ELEV_HI_DEG, 20.0, 30.0, 45.0, 90.0] {
            assert_eq!(
                aerial_veil_scale(e),
                AERIAL_VEIL_DAY_SCALE,
                "daytime de-haze at elev {e}"
            );
        }
        // Monotone non-increasing across the ramp.
        let mut prev = f64::INFINITY;
        for i in 0..=100 {
            let e = -5.0 + 40.0 * i as f64 / 100.0;
            let v = aerial_veil_scale(e);
            assert!(v <= prev + 1e-12, "not monotone at {e}: {v} > {prev}");
            assert!((AERIAL_VEIL_DAY_SCALE..=1.0).contains(&v));
            prev = v;
        }
        // The science-review "residual de-haze at ~5 deg" evaluation: the smoothstep
        // leaves veil ~0.9 at 5 deg (reduced-but-not-disabled, the satpy idiom).
        let v5 = aerial_veil_scale(5.0);
        assert!(
            (0.85..0.97).contains(&v5),
            "5-deg residual de-haze out of band: {v5}"
        );
    }

    #[test]
    fn low_sun_illuminant_gains_are_identity_outside_the_band() {
        let (_params, luts, _sky) = shared_optics();
        // At/below the IN gate (the amber terminator + the whole below-horizon dusk
        // band) and at/above the OUT gate (full daytime): gains are EXACTLY [1,1,1],
        // so those approved looks are byte-identical by construction.
        for &e in &[-10.0f64, -4.0, -2.0, 0.0, ILLUM_CORR_IN_LO_DEG] {
            assert_eq!(low_sun_illuminant_gains(e, luts), [1.0, 1.0, 1.0], "at {e}");
            assert_eq!(low_sun_illuminant_weight(e), 0.0, "weight at {e}");
        }
        for &e in &[ILLUM_CORR_OUT_HI_DEG, 45.0, 90.0] {
            assert_eq!(low_sun_illuminant_gains(e, luts), [1.0, 1.0, 1.0], "at {e}");
            assert_eq!(low_sun_illuminant_weight(e), 0.0, "weight at {e}");
        }
        // apply_low_sun_illuminant: off-earth pixels are returned unchanged (the limb
        // keeps its physical color) even inside the band.
        let l = [3.0, 4.0, 5.0];
        assert_eq!(apply_low_sun_illuminant(l, false, 6.0, luts), l);
        // Inside the band an on-earth pixel IS corrected.
        assert_ne!(apply_low_sun_illuminant(l, true, 6.0, luts), l);
    }

    #[test]
    fn low_sun_illuminant_gains_correct_the_green_deficit() {
        let (_params, luts, _sky) = shared_optics();
        // The full-correction plateau spans the sunrise band.
        for &e in &[ILLUM_CORR_IN_HI_DEG, 10.0, 15.0, ILLUM_CORR_OUT_LO_DEG] {
            assert_eq!(low_sun_illuminant_weight(e), 1.0, "plateau at {e}");
        }
        // In the sunrise band the LUT illuminant carries the Chappuis ozone GREEN DIP
        // (phase-1 diagnosis), so the gains must RAISE green, preserve the R-B warm
        // axis EXACTLY (equal R and B gains — the reference keeps dawn cloud warm),
        // and pull both down slightly (the unit-luminance renormalization).
        for &e in &[5.0f64, 8.0, 12.0] {
            let g = low_sun_illuminant_gains(e, luts);
            assert!(g[1] > 1.0, "green gain at {e}: {g:?}");
            assert_eq!(g[0], g[2], "R and B gains must be equal at {e}: {g:?}");
            assert!(g[0] < 1.0, "renormalization scale at {e}: {g:?}");
            assert!(g[1] > g[0], "green must rise relative to R/B at {e}: {g:?}");
        }
        // LUMINANCE PRESERVATION (the property that protects the approved medians): at
        // full weight the corrected illuminant keeps its Rec.709 luminance exactly —
        // the green dip is repaired in chroma, not brightness. And the corrected green
        // sits ON the Rayleigh log-line between the (rescaled) R and B channels.
        let e = 10.0;
        assert_eq!(low_sun_illuminant_weight(e), 1.0);
        let t = atmosphere::sample_transmittance(
            &luts.transmittance,
            atmosphere::R_GROUND_M + ILLUM_REF_CLOUD_ALT_M,
            e.to_radians().sin(),
        );
        let g = low_sun_illuminant_gains(e, luts);
        let y = |c: [f64; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        let corrected = [t[0] * g[0], t[1] * g[1], t[2] * g[2]];
        assert!(
            (y(corrected) / y(t) - 1.0).abs() < 1e-12,
            "luminance not preserved: {} vs {}",
            y(corrected),
            y(t)
        );
        // R/B ratio preserved exactly through the correction.
        assert!(
            (corrected[0] / corrected[2] - t[0] / t[2]).abs() < 1e-12,
            "warm axis must be preserved"
        );
        // Green on the Rayleigh line: ln(G) = lerp(ln(R), ln(B), a) of the CORRECTED
        // triple (the uniform scale shifts all three logs equally, so the collinearity
        // holds for the corrected triple exactly when the restoration is applied).
        let ray = atmosphere::RAYLEIGH_SCATTERING;
        let a = (ray[1] - ray[0]) / (ray[2] - ray[0]);
        let expected_ln_g = (1.0 - a) * corrected[0].ln() + a * corrected[2].ln();
        assert!(
            (corrected[1].ln() - expected_ln_g).abs() < 1e-9,
            "corrected green must sit on the Rayleigh log-line: {} vs {expected_ln_g}",
            corrected[1].ln()
        );
    }

    #[test]
    fn synthetic_green_mode_is_off_by_default_and_kills_green_axis_casts() {
        // OFF by default: the shipped output is byte-identical.
        assert!(!synthetic_green_enabled());
        // A grey triple passes through the synthesis unchanged (weights sum to 1).
        let grey = synthesize_abi_green([0.4, 0.4, 0.4]);
        for c in 0..3 {
            assert!((grey[c] - 0.4).abs() < 1e-15, "grey drifted: {grey:?}");
        }
        // The green-magenta axis is arithmetically collapsed: G' - mid = 0.1*(G - mid)
        // with mid = (R+B)/2, i.e. the G deviation from the R-B line shrinks exactly
        // 10x — a mauve (G below the line, the phase-1 lavender CDO) rises toward it,
        // a green-excess triple falls toward it, and R/B are untouched (the R>B warm
        // SLOPE of a khaki is the illuminant gains' job, not the synthesis').
        for probe in [[0.534f64, 0.421, 0.444], [0.30, 0.40, 0.28]] {
            let syn = synthesize_abi_green(probe);
            let mid = 0.5 * (probe[0] + probe[2]);
            assert!(
                (syn[1] - mid - 0.1 * (probe[1] - mid)).abs() < 1e-12,
                "G deviation must shrink 10x: {syn:?} vs {probe:?}"
            );
            assert_eq!(syn[0], probe[0]);
            assert_eq!(syn[2], probe[2]);
        }
        let mauve = [0.534, 0.421, 0.444];
        assert!(
            synthesize_abi_green(mauve)[1] > mauve[1],
            "mauve G must rise toward the R-B line"
        );
        // The pure transform with the flag OFF is byte-identical to the shipped chain;
        // with the flag ON it equals the shipped chain fed the synthesized triple
        // (the synthesis happens BEFORE desaturation, as the real product computes its
        // display green from band reflectances before any stretch). Testing through
        // the pure function so no test ever toggles the process-global.
        for rho in [[0.2, 0.15, 0.18], [0.9, 0.7, 0.8], [0.05, 0.04, 0.06]] {
            let off = output_transform_with(
                rho,
                OutputTransform::AbiReflectance,
                CLOUD_SOFTCLIP_KNEE,
                RHO_HIGHLIGHT_MAX,
                false,
            );
            let baseline = apply_output_transform(
                rho,
                OutputTransform::AbiReflectance,
                CLOUD_SOFTCLIP_KNEE,
                RHO_HIGHLIGHT_MAX,
            );
            assert_eq!(off, baseline, "flag off must be the shipped transform");
            let on = output_transform_with(
                rho,
                OutputTransform::AbiReflectance,
                CLOUD_SOFTCLIP_KNEE,
                RHO_HIGHLIGHT_MAX,
                true,
            );
            let pre = output_transform_with(
                synthesize_abi_green(rho),
                OutputTransform::AbiReflectance,
                CLOUD_SOFTCLIP_KNEE,
                RHO_HIGHLIGHT_MAX,
                false,
            );
            assert_eq!(on, pre, "synthesis composes before desaturation");
        }
    }

    /// DIAGNOSTIC (low-sun visible pass, phase 1) — not a gate test; run by name:
    /// `cargo test -p simsat --release diag_low_sun -- --ignored --nocapture`.
    /// Prints the exact transmittance-LUT direct-sun color at cloud-sample altitudes
    /// for sunrise-band sun elevations, plus the SH sky-ambient irradiance color at an
    /// up normal — the two colors every cloud march sample mixes
    /// (`s_sun ~ e_sun * t_sun`, `s_amb ~ e_sky / pi`). The R/B ratios here bound the
    /// physics warm cast BEFORE any display transform.
    #[test]
    #[ignore]
    fn diag_low_sun_transmittance_table() {
        let (_params, luts, sky_sh) = shared_optics();
        let e_sun = SOLAR_IRRADIANCE_RGB;
        println!("== direct-sun transmittance color t_sun (LUT) ==");
        println!("alt_km elev_deg      tR      tG      tB   t_R/B  (e*t)_R/B");
        for &alt_km in &[0.0f64, 1.0, 2.0, 5.0, 10.0] {
            for &elev in &[0.0f64, 2.0, 5.0, 8.0, 12.0, 20.0, 30.0] {
                let mu = elev.to_radians().sin();
                let t = atmosphere::sample_transmittance(
                    &luts.transmittance,
                    atmosphere::R_GROUND_M + alt_km * 1000.0 + 1.0,
                    mu,
                );
                let rb = if t[2] > 0.0 { t[0] / t[2] } else { f64::NAN };
                let erb = if t[2] > 0.0 {
                    (e_sun[0] * t[0]) / (e_sun[2] * t[2])
                } else {
                    f64::NAN
                };
                println!(
                    "{alt_km:6.1} {elev:8.1} {:7.4} {:7.4} {:7.4} {rb:7.3} {erb:10.3}",
                    t[0], t[1], t[2]
                );
            }
        }
        println!("== SH sky ambient irradiance color at an up normal ==");
        println!("elev_deg      eR      eG      eB   e_R/B");
        for &elev in &[0.0f64, 2.0, 5.0, 8.0, 12.0, 20.0, 30.0] {
            let e = elev.to_radians();
            let sun_enu = [0.0, e.cos(), e.sin()];
            let up = [0.0, 0.0, 1.0];
            let irr = sky_sh.irradiance(elev, up, sun_enu, up);
            let rb = if irr[2] > 0.0 {
                irr[0] / irr[2]
            } else {
                f64::NAN
            };
            println!(
                "{elev:8.1} {:7.4} {:7.4} {:7.4} {rb:7.3}",
                irr[0], irr[1], irr[2]
            );
        }
    }
}
