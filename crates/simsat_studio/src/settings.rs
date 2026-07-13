//! Studio settings persistence + recent files (WS4 item 2).
//!
//! A hand-rolled `settings.json` under `%LOCALAPPDATA%/SimSatStudio/` — NOT
//! eframe's `persistence` feature (that would serialize opaque egui memory; this
//! file is a small, stable, human-readable contract the studio owns).
//!
//! Robustness rules (this machine crashes — hard rule 7 spirit):
//! - LOAD never fails: any missing / unreadable / corrupt file yields the defaults,
//!   and every loaded value is passed through [`StudioSettings::sanitize`] (numeric
//!   fields clamped to the UI slider ranges, unknown enum tokens reset to their
//!   defaults, the recent list deduped + capped).
//! - SAVE is atomic: write to a `settings.json.tmp` sibling, then rename over the
//!   real file, so a crash mid-write can never leave a truncated settings file.
//! - Engine enums are stored as STABLE STRING TOKENS (not derive-serialized
//!   variants), so renaming a Rust variant can never silently break or reinterpret
//!   an existing settings file.
//!
//! Deliberately NOT persisted: the fake-sun what-if override (the on/off flag AND
//! the elevation/azimuth sliders). Every session starts with the physically-honest
//! real-timestamp sun — a persisted override would silently light every frame of
//! the next session with a non-physical sun the owner may have forgotten about.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use simsat::api::{FractionalCloudMode, RenderIntent};
use simsat::atmosphere::OutputTransform;
use simsat::camera::{GeoNavigation, ResolutionMode, SatellitePreset};
use simsat::clouds::{DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE, StepQuality};
use simsat::derived::DerivedField;
use simsat::instrument_footprint::InstrumentFootprint;
use simsat::ir_enhance::IrEnhancement;
use simsat::thermal_sensor::ThermalSensor;
use simsat::wv::WvBand;

use crate::{RenderMode, StudioView};

/// Maximum entries kept in the recent-files list.
pub const RECENT_CAP: usize = 8;
/// Visible-calibration settings epoch. Epoch 1 was the diagnostic v0.1.4 RC
/// (`cloud_optical_depth_scale = 0.25`, exposure/ground = `1.6`); epoch 2 carries
/// the owner-selected `0.15` plus the neutral ABI display/ground baseline; epoch 3
/// keeps OD `0.15` and selects exposure `1.5`; epoch 4 promotes the owner-selected
/// SZA normalization and dark-land toe; epoch 5 promotes exposed-domain cloud-edge
/// feathering to the shipped visible preset; epoch 6 promotes the owner-selected
/// low-sun land SZA maximum gain from `1.6` to `4.0` and the reviewed
/// deterministic two-subcolumn Recommended closure; epoch 7 promotes the owner-selected
/// tightly gated twilight surface recovery (`0.30 / 0.50 / 4.0`) for visible displays;
/// epoch 8 promotes the reviewed, sun-gated surface-only ground lift from `1.0` to `1.10`.
/// Older settings are preserved and surfaced instead of being silently overwritten.
pub const VISIBLE_CALIBRATION_EPOCH: u32 = 8;
const LAND_APPEARANCE_CALIBRATION_EPOCH: u64 = 4;
const EXPOSED_EDGE_CALIBRATION_EPOCH: u64 = 5;
const LAND_SZA_GAIN_CALIBRATION_EPOCH: u64 = 6;
const TWILIGHT_SURFACE_RECOVERY_CALIBRATION_EPOCH: u64 = 7;
const GROUND_LIFT_CALIBRATION_EPOCH: u64 = 8;
const LEGACY_LAND_SZA_MAX_GAIN: f32 = 1.6;
const LEGACY_GROUND_DAY_LIFT: f32 = 1.0;

/// One remembered open action: what kind of open it was + the path(s) involved
/// (one path for a wrfout / cached run / sequence folder; several for a
/// multi-file sequence selection).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct RecentEntry {
    /// `"wrfout"` | `"cached"` | `"sequence"`.
    pub kind: String,
    pub paths: Vec<String>,
}

impl RecentEntry {
    /// A short menu label: the first path's file name + a kind hint.
    pub fn label(&self) -> String {
        let first = self
            .paths
            .first()
            .map(|p| {
                Path::new(p)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or(p.as_str())
                    .to_string()
            })
            .unwrap_or_else(|| "(empty)".to_string());
        let extra = if self.paths.len() > 1 {
            format!(" (+{} more)", self.paths.len() - 1)
        } else {
            String::new()
        };
        let kind = match self.kind.as_str() {
            "cached" => " [cached run]",
            "sequence" => " [sequence]",
            _ => "",
        };
        format!("{first}{extra}{kind}")
    }
}

/// Everything the studio persists between sessions. Numeric fields mirror the UI
/// slider ranges; enum-backed fields hold stable string tokens (see the module
/// doc). `#[serde(default)]` lets a partial file (older version) load field-wise;
/// a structurally-corrupt file falls back to full defaults in [`load`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StudioSettings {
    /// Calibration defaults the user has explicitly accepted. `0` denotes a settings
    /// file written before this field existed and keeps the migration banner visible.
    pub visible_calibration_epoch: u32,
    /// Sat-store root override; `None` = the app default beside the cache dir.
    pub store_root: Option<String>,
    pub sat: String,
    pub geo_navigation: String,
    pub resolution: String,
    pub view: String,
    pub mode: String,
    /// Scientific/display intent. Stored as a stable token so a Sensor Fast Gray
    /// selection survives an app restart; older files default to `display` through
    /// the struct-level `#[serde(default)]` contract.
    pub render_intent: String,
    pub ir_enhancement: String,
    pub thermal_sensor: String,
    /// Complete-radiance instrument spatial-response stage. Stable enum token;
    /// off by default and validated against the selected geometry at render time.
    pub instrument_footprint: String,
    pub output_transform: String,
    pub step_quality: String,
    pub margin_pct: f32,
    pub aod: f32,
    pub rh_swelling: bool,
    /// Daytime aerial-veil correction. `true` is the product-facing
    /// default; persisted so QA can make an exact corrected/raw-TOA A/B.
    pub atmosphere_correction: bool,
    /// Shorten the atmospheric column to the model terrain elevation. `true` is
    /// the physical default; `false` preserves the sea-level-column QA baseline.
    pub terrain_atmosphere: bool,
    /// Land-only solar-zenith display normalization (owner-selected default on).
    pub land_sza_normalization: bool,
    /// Upper bound for the land solar-zenith normalization.
    pub land_sza_max_gain: f32,
    /// Bounded dark-land reflectance toe (owner-selected default on).
    pub land_dark_toe: bool,
    pub land_dark_toe_knee: f32,
    pub land_dark_toe_gamma: f32,
    pub land_dark_toe_max_gain: f32,
    /// Experimental terrain-only toe over the lit, view-attenuated surface signal.
    /// Default-off and persisted solely for explicit display A/B work.
    pub surface_postlight_toe: bool,
    pub surface_postlight_toe_knee: f32,
    pub surface_postlight_toe_gamma: f32,
    pub surface_postlight_toe_max_gain: f32,
    /// Separate tightly gated low-sun recovery; shipped on for finished visible displays.
    pub twilight_surface_recovery: bool,
    pub twilight_surface_recovery_knee: f32,
    pub twilight_surface_recovery_gamma: f32,
    pub twilight_surface_recovery_max_gain: f32,
    pub clouds_enabled: bool,
    /// Use model cloud fraction/subcolumns when the brick provides them. `true` is
    /// the physical default; `false` preserves legacy horizontally-full cloudy cells.
    pub fractional_clouds: bool,
    /// Stable token for the CPU fractional-cloud observation operator.
    /// `deterministic-2` is the reviewed display default; effective OD remains the
    /// explicit fast/sensor-compatible closure and deterministic 4/8/16 are higher-cost
    /// fixed-stratified references. The legacy boolean remains the master switch.
    pub fractional_cloud_mode: String,
    pub multiscatter: bool,
    /// Opt-in Stage-2 Monte Carlo depth-source closure. False preserves the exact
    /// legacy octave/single-scatter dispatch.
    pub delta_flux_clouds: bool,
    /// Opt-in bounded P1 directional reconstruction over the same Stage-2 LUT.
    /// False preserves every legacy/v1 path.
    pub delta_flux_v2_clouds: bool,
    /// Opt-in successive-order angular-memory reconstruction over the Stage-2 LUT.
    /// False preserves every legacy/v1/v2 path.
    pub delta_flux_v3_clouds: bool,
    /// Visible cloud optical-depth calibration. The shipped default is `0.15`;
    /// `1.0` keeps the model-derived physical input unchanged.
    pub cloud_optical_depth_scale: f32,
    /// Opt-in high-precision extinction brick storage. False keeps CompactU8.
    pub science_cloud_f16: bool,
    /// Experimental WRF NSSL MP18 mass/number/volume-moment particle optics.
    pub nssl_native_cloud_optics: bool,
    /// Experimental HRRR Thompson/Eidhammer native particle optics.
    pub hrrr_thompson_native_cloud_optics: bool,
    /// Fade finished visible clouds at camera-exposed finite-domain boundaries.
    /// Default on in v0.1.5; false is the exact pre-feature margin-gated behavior.
    pub feather_exposed_domain_edges: bool,
    pub beer_powder: bool,
    /// Display-only sub-grid cloud-edge erosion. Explicitly opt-in/off by default.
    pub granulation: bool,
    /// Top-down-only low-stratiform column-OD reconstruction. Experimental and
    /// explicitly opt-in; geostationary/raw products ignore it.
    pub topdown_stratiform_regularization: bool,
    /// Display-only top-down pre-tonemap cloud-radiance footprint. Experimental,
    /// explicitly opt-in, and persisted for repeatable A/B review.
    pub topdown_cloud_footprint: bool,
    /// Display-only top-down ground-shadow anti-aliasing. Default on and persisted so
    /// the unfiltered diagnostic path remains available for exact A/B comparisons.
    pub topdown_shadow_antialias: bool,
    pub exposure: f32,
    /// Sun-gated daytime ground-radiance lift. `1.0` is neutral.
    pub ground_gain: f32,
    /// Finished-display highlight shoulder knee. `1.0` disables the shoulder.
    pub cloud_softclip: f32,
    /// Physical reflectance-factor ceiling mapped to display white.
    pub cloud_highlight_max: f32,
    pub bm_month_override: u32,
    pub bm_allow_download: bool,
    pub play_fps: f32,
    pub frame_cap: usize,
    /// Perspective (3-D) orbit camera: azimuth / tilt / range / fov + output dims
    /// (see `pipeline::OrbitParams`). Persisted so the owner's framing survives a
    /// restart; sanitized to the UI slider ranges on load.
    pub orbit_az_deg: f32,
    pub orbit_tilt_deg: f32,
    pub orbit_range_km: f32,
    pub orbit_fov_deg: f32,
    pub persp_width: u32,
    pub persp_height: u32,
    pub recent: Vec<RecentEntry>,
}

impl Default for StudioSettings {
    fn default() -> Self {
        // Mirrors SimSatStudioApp::new's defaults exactly (the honest baseline).
        let land = simsat::render::LandAppearanceConfig::default();
        let postlight_toe = simsat::render::SurfacePostlightToeConfig::off();
        let twilight_recovery = simsat::render::TwilightSurfaceRecoveryConfig::shipped();
        Self {
            visible_calibration_epoch: VISIBLE_CALIBRATION_EPOCH,
            store_root: None,
            sat: sat_token(SatellitePreset::GoesEast).to_string(),
            geo_navigation: geo_navigation_token(GeoNavigation::ModelSphere).to_string(),
            resolution: resolution_token(ResolutionMode::Native).to_string(),
            view: view_token(StudioView::Geostationary).to_string(),
            mode: mode_token(RenderMode::Visible).to_string(),
            render_intent: render_intent_token(RenderIntent::Display).to_string(),
            ir_enhancement: enhancement_token(IrEnhancement::default()).to_string(),
            thermal_sensor: thermal_sensor_token(ThermalSensor::FastGray).to_string(),
            instrument_footprint: instrument_footprint_token(InstrumentFootprint::Off).to_string(),
            output_transform: output_transform_token(OutputTransform::AbiReflectance).to_string(),
            step_quality: step_quality_token(StepQuality::Offline).to_string(),
            margin_pct: 0.0,
            aod: simsat::atmosphere::DEFAULT_AOD as f32,
            rh_swelling: false,
            atmosphere_correction: true,
            terrain_atmosphere: true,
            land_sza_normalization: land.sza_normalization,
            land_sza_max_gain: land.sza_max_gain as f32,
            land_dark_toe: land.dark_toe,
            land_dark_toe_knee: land.dark_toe_knee as f32,
            land_dark_toe_gamma: land.dark_toe_gamma as f32,
            land_dark_toe_max_gain: land.dark_toe_max_gain as f32,
            surface_postlight_toe: postlight_toe.enabled,
            surface_postlight_toe_knee: postlight_toe.knee as f32,
            surface_postlight_toe_gamma: postlight_toe.gamma as f32,
            surface_postlight_toe_max_gain: postlight_toe.max_gain as f32,
            twilight_surface_recovery: twilight_recovery.enabled,
            twilight_surface_recovery_knee: twilight_recovery.knee as f32,
            twilight_surface_recovery_gamma: twilight_recovery.gamma as f32,
            twilight_surface_recovery_max_gain: twilight_recovery.max_gain as f32,
            clouds_enabled: true,
            fractional_clouds: true,
            fractional_cloud_mode: fractional_cloud_mode_token(FractionalCloudMode::Deterministic2)
                .to_string(),
            multiscatter: true,
            delta_flux_clouds: false,
            delta_flux_v2_clouds: false,
            delta_flux_v3_clouds: false,
            cloud_optical_depth_scale: DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE,
            science_cloud_f16: false,
            nssl_native_cloud_optics: false,
            hrrr_thompson_native_cloud_optics: false,
            feather_exposed_domain_edges: true,
            beer_powder: false,
            granulation: false,
            topdown_stratiform_regularization: false,
            topdown_cloud_footprint: false,
            topdown_shadow_antialias: true,
            exposure: simsat::render::DEFAULT_EXPOSURE as f32,
            ground_gain: simsat::render::GROUND_DAY_LIFT as f32,
            cloud_softclip: simsat::render::CLOUD_SOFTCLIP_KNEE as f32,
            cloud_highlight_max: simsat::render::RHO_HIGHLIGHT_MAX as f32,
            bm_month_override: 0,
            bm_allow_download: true,
            play_fps: 8.0,
            frame_cap: 120,
            // Orbit defaults: from the south at a 30-deg oblique, 300 km out, a
            // 45-deg horizontal FOV, 720p output (the hero-shot framing family).
            orbit_az_deg: 180.0,
            orbit_tilt_deg: 30.0,
            orbit_range_km: 300.0,
            orbit_fov_deg: 45.0,
            persp_width: 1280,
            persp_height: 720,
            recent: Vec::new(),
        }
    }
}

/// Clamp a float to the slider range, falling back to `default` for any
/// non-finite value (JSON cannot encode NaN, but a defensive load never trusts).
fn clamp_finite(v: f32, lo: f32, hi: f32, default: f32) -> f32 {
    if v.is_finite() {
        v.clamp(lo, hi)
    } else {
        default
    }
}

impl StudioSettings {
    /// Replace only the persisted visible-appearance calibration with the current
    /// shipped preset. File/view/export/playback/recent settings remain untouched.
    /// This is the single migration seam used by the Studio's calibration banner.
    pub(crate) fn apply_shipped_visible_calibration(&mut self) {
        let d = Self::default();
        self.visible_calibration_epoch = VISIBLE_CALIBRATION_EPOCH;
        self.output_transform = d.output_transform;
        self.aod = d.aod;
        self.rh_swelling = d.rh_swelling;
        self.atmosphere_correction = d.atmosphere_correction;
        self.terrain_atmosphere = d.terrain_atmosphere;
        self.land_sza_normalization = d.land_sza_normalization;
        self.land_sza_max_gain = d.land_sza_max_gain;
        self.land_dark_toe = d.land_dark_toe;
        self.land_dark_toe_knee = d.land_dark_toe_knee;
        self.land_dark_toe_gamma = d.land_dark_toe_gamma;
        self.land_dark_toe_max_gain = d.land_dark_toe_max_gain;
        self.surface_postlight_toe = d.surface_postlight_toe;
        self.surface_postlight_toe_knee = d.surface_postlight_toe_knee;
        self.surface_postlight_toe_gamma = d.surface_postlight_toe_gamma;
        self.surface_postlight_toe_max_gain = d.surface_postlight_toe_max_gain;
        self.twilight_surface_recovery = d.twilight_surface_recovery;
        self.twilight_surface_recovery_knee = d.twilight_surface_recovery_knee;
        self.twilight_surface_recovery_gamma = d.twilight_surface_recovery_gamma;
        self.twilight_surface_recovery_max_gain = d.twilight_surface_recovery_max_gain;
        self.clouds_enabled = d.clouds_enabled;
        self.fractional_clouds = d.fractional_clouds;
        self.fractional_cloud_mode = d.fractional_cloud_mode;
        self.multiscatter = d.multiscatter;
        self.delta_flux_clouds = d.delta_flux_clouds;
        self.delta_flux_v2_clouds = d.delta_flux_v2_clouds;
        self.delta_flux_v3_clouds = d.delta_flux_v3_clouds;
        self.cloud_optical_depth_scale = d.cloud_optical_depth_scale;
        self.nssl_native_cloud_optics = d.nssl_native_cloud_optics;
        self.hrrr_thompson_native_cloud_optics = d.hrrr_thompson_native_cloud_optics;
        self.feather_exposed_domain_edges = d.feather_exposed_domain_edges;
        self.beer_powder = d.beer_powder;
        self.granulation = d.granulation;
        self.topdown_stratiform_regularization = d.topdown_stratiform_regularization;
        self.topdown_cloud_footprint = d.topdown_cloud_footprint;
        self.topdown_shadow_antialias = d.topdown_shadow_antialias;
        self.exposure = d.exposure;
        self.ground_gain = d.ground_gain;
        self.cloud_softclip = d.cloud_softclip;
        self.cloud_highlight_max = d.cloud_highlight_max;
    }

    /// Clamp every numeric field to its UI slider range, reset unknown enum
    /// tokens to their defaults, and dedupe + cap the recent list. Idempotent.
    pub fn sanitize(&mut self) {
        let d = StudioSettings::default();
        // Keep epoch 0 so the Studio can offer (rather than force) the new calibration.
        // Unknown future epochs are treated as current by this older build.
        self.visible_calibration_epoch = self
            .visible_calibration_epoch
            .min(VISIBLE_CALIBRATION_EPOCH);
        self.margin_pct = clamp_finite(self.margin_pct, 0.0, 100.0, d.margin_pct);
        self.aod = clamp_finite(self.aod, 0.0, 0.6, d.aod);
        self.land_sza_max_gain =
            clamp_finite(self.land_sza_max_gain, 1.0, 4.0, d.land_sza_max_gain);
        self.land_dark_toe_knee =
            clamp_finite(self.land_dark_toe_knee, 1.0e-6, 1.0, d.land_dark_toe_knee);
        self.land_dark_toe_gamma =
            clamp_finite(self.land_dark_toe_gamma, 0.05, 1.0, d.land_dark_toe_gamma);
        self.land_dark_toe_max_gain = clamp_finite(
            self.land_dark_toe_max_gain,
            1.0,
            4.0,
            d.land_dark_toe_max_gain,
        );
        self.surface_postlight_toe_knee = clamp_finite(
            self.surface_postlight_toe_knee,
            1.0e-6,
            1.0,
            d.surface_postlight_toe_knee,
        );
        self.surface_postlight_toe_gamma = clamp_finite(
            self.surface_postlight_toe_gamma,
            0.05,
            1.0,
            d.surface_postlight_toe_gamma,
        );
        self.surface_postlight_toe_max_gain = clamp_finite(
            self.surface_postlight_toe_max_gain,
            1.0,
            4.0,
            d.surface_postlight_toe_max_gain,
        );
        self.twilight_surface_recovery_knee = clamp_finite(
            self.twilight_surface_recovery_knee,
            1.0e-6,
            1.0,
            d.twilight_surface_recovery_knee,
        );
        self.twilight_surface_recovery_gamma = clamp_finite(
            self.twilight_surface_recovery_gamma,
            0.05,
            1.0,
            d.twilight_surface_recovery_gamma,
        );
        self.twilight_surface_recovery_max_gain = clamp_finite(
            self.twilight_surface_recovery_max_gain,
            1.0,
            4.0,
            d.twilight_surface_recovery_max_gain,
        );
        self.cloud_optical_depth_scale = clamp_finite(
            self.cloud_optical_depth_scale,
            0.0,
            4.0,
            d.cloud_optical_depth_scale,
        );
        self.exposure = clamp_finite(self.exposure, 0.25, 4.0, d.exposure);
        self.ground_gain = clamp_finite(self.ground_gain, 0.25, 4.0, d.ground_gain);
        self.cloud_softclip = clamp_finite(self.cloud_softclip, 0.05, 1.0, d.cloud_softclip);
        self.cloud_highlight_max =
            clamp_finite(self.cloud_highlight_max, 0.25, 4.0, d.cloud_highlight_max);
        self.play_fps = clamp_finite(self.play_fps, 1.0, 30.0, d.play_fps);
        self.frame_cap = self.frame_cap.clamp(8, 480);
        // Perspective orbit params: the UI slider ranges (the render-time mapping
        // additionally clamps range to the domain-derived bounds).
        self.orbit_az_deg = clamp_finite(self.orbit_az_deg, 0.0, 360.0, d.orbit_az_deg);
        self.orbit_tilt_deg = clamp_finite(self.orbit_tilt_deg, 5.0, 85.0, d.orbit_tilt_deg);
        self.orbit_range_km = clamp_finite(self.orbit_range_km, 10.0, 5000.0, d.orbit_range_km);
        self.orbit_fov_deg = clamp_finite(self.orbit_fov_deg, 15.0, 120.0, d.orbit_fov_deg);
        self.persp_width = self.persp_width.clamp(2, 4096);
        self.persp_height = self.persp_height.clamp(2, 4096);
        if self.bm_month_override > 12 {
            self.bm_month_override = 0;
        }
        self.sat = sat_from_token(&self.sat)
            .map(|preset| sat_token(preset).to_string())
            .unwrap_or(d.sat);
        if geo_navigation_from_token(&self.geo_navigation).is_none() {
            self.geo_navigation = d.geo_navigation;
        }
        if resolution_from_token(&self.resolution).is_none() {
            self.resolution = d.resolution;
        }
        if view_from_token(&self.view).is_none() {
            self.view = d.view;
        }
        if mode_from_token(&self.mode).is_none() {
            self.mode = d.mode;
        }
        if render_intent_from_token(&self.render_intent).is_none() {
            self.render_intent = d.render_intent;
        }
        if enhancement_from_token(&self.ir_enhancement).is_none() {
            self.ir_enhancement = d.ir_enhancement;
        }
        if thermal_sensor_from_token(&self.thermal_sensor).is_none() {
            self.thermal_sensor = d.thermal_sensor;
        }
        if instrument_footprint_from_token(&self.instrument_footprint).is_none() {
            self.instrument_footprint = d.instrument_footprint;
        }
        if output_transform_from_token(&self.output_transform).is_none() {
            self.output_transform = d.output_transform;
        }
        if step_quality_from_token(&self.step_quality).is_none() {
            self.step_quality = d.step_quality;
        }
        if fractional_cloud_mode_from_token(&self.fractional_cloud_mode).is_none() {
            self.fractional_cloud_mode = d.fractional_cloud_mode;
        }
        // Native optics are one ingest-time mode, even though the backward-compatible
        // settings schema stores two booleans. Match the worker's established precedence
        // (HRRR Thompson before NSSL) so a hand-edited/legacy invalid file never displays
        // two checked modes while rendering only one.
        if self.hrrr_thompson_native_cloud_optics {
            self.nssl_native_cloud_optics = false;
        }
        // Recent list: drop malformed entries, dedupe (first occurrence wins,
        // i.e. most recent since the list is newest-first), cap.
        self.recent.retain(|e| {
            !e.paths.is_empty() && matches!(e.kind.as_str(), "wrfout" | "cached" | "sequence")
        });
        let mut seen: Vec<RecentEntry> = Vec::new();
        for e in self.recent.drain(..) {
            if !seen.contains(&e) {
                seen.push(e);
            }
        }
        seen.truncate(RECENT_CAP);
        self.recent = seen;
    }
}

// ── stable string tokens for every engine enum the settings store ─────────────

pub fn sat_token(p: SatellitePreset) -> &'static str {
    match p {
        SatellitePreset::GoesEast => "goes-east",
        SatellitePreset::GoesWest => "goes-west",
        SatellitePreset::Himawari => "himawari",
        SatellitePreset::MtgI1 => "mtgi1",
    }
}

pub fn sat_from_token(t: &str) -> Option<SatellitePreset> {
    match t.to_ascii_lowercase().replace(['-', '_', ' '], "").as_str() {
        "goeseast" | "goese" | "east" => Some(SatellitePreset::GoesEast),
        "goeswest" | "goesw" | "west" => Some(SatellitePreset::GoesWest),
        "himawari" | "ahi" => Some(SatellitePreset::Himawari),
        "mtg" | "mtgi1" | "meteosat12" => Some(SatellitePreset::MtgI1),
        _ => None,
    }
}

pub fn geo_navigation_token(navigation: GeoNavigation) -> &'static str {
    navigation.slug()
}

pub fn geo_navigation_from_token(t: &str) -> Option<GeoNavigation> {
    GeoNavigation::ALL
        .into_iter()
        .find(|navigation| geo_navigation_token(*navigation) == t)
}

pub fn resolution_token(r: ResolutionMode) -> &'static str {
    match r {
        ResolutionMode::Native => "native",
        ResolutionMode::Abi1km => "abi-1km",
        ResolutionMode::Abi2km => "abi-2km",
    }
}

pub fn resolution_from_token(t: &str) -> Option<ResolutionMode> {
    ResolutionMode::ALL
        .into_iter()
        .find(|r| resolution_token(*r) == t)
}

pub fn view_token(v: StudioView) -> &'static str {
    match v {
        StudioView::Geostationary => "geo",
        StudioView::TopDownMap => "topdown",
        StudioView::Perspective => "perspective",
    }
}

pub fn view_from_token(t: &str) -> Option<StudioView> {
    StudioView::ALL.into_iter().find(|v| view_token(*v) == t)
}

pub fn mode_token(m: RenderMode) -> &'static str {
    match m {
        RenderMode::Visible => "visible",
        RenderMode::GeoColor => "geocolor",
        RenderMode::Sandwich => "sandwich",
        RenderMode::Ir => "ir-band13",
        RenderMode::WaterVapor(WvBand::Upper) => "wv-6.2",
        RenderMode::WaterVapor(WvBand::Mid) => "wv-6.9",
        RenderMode::WaterVapor(WvBand::Low) => "wv-7.3",
        RenderMode::Derived(DerivedField::PrecipitableWater) => "derived-pw",
        RenderMode::Derived(DerivedField::CloudTopTemp) => "derived-ctt",
        RenderMode::Derived(DerivedField::CloudOpticalDepth) => "derived-cod",
    }
}

pub fn mode_from_token(t: &str) -> Option<RenderMode> {
    RenderMode::ALL.into_iter().find(|m| mode_token(*m) == t)
}

pub fn render_intent_token(intent: RenderIntent) -> &'static str {
    intent.slug()
}

pub fn render_intent_from_token(t: &str) -> Option<RenderIntent> {
    match t {
        "display" => Some(RenderIntent::Display),
        "sensor-fast-gray" => Some(RenderIntent::SensorFastGray),
        _ => None,
    }
}

pub fn enhancement_token(e: IrEnhancement) -> &'static str {
    match e {
        IrEnhancement::Natural => "natural",
        IrEnhancement::Cimss => "cimss",
        IrEnhancement::Bd => "bd",
        IrEnhancement::Avn => "avn",
        IrEnhancement::Funktop => "funktop",
        IrEnhancement::Rainbow => "rainbow",
        IrEnhancement::Grayscale => "grayscale",
    }
}

pub fn enhancement_from_token(t: &str) -> Option<IrEnhancement> {
    IrEnhancement::ALL
        .into_iter()
        .find(|e| enhancement_token(*e) == t)
}

pub fn thermal_sensor_token(sensor: ThermalSensor) -> &'static str {
    sensor.slug()
}

pub fn thermal_sensor_from_token(t: &str) -> Option<ThermalSensor> {
    ThermalSensor::parse(t)
}

pub fn instrument_footprint_token(footprint: InstrumentFootprint) -> &'static str {
    footprint.slug()
}

pub fn instrument_footprint_from_token(t: &str) -> Option<InstrumentFootprint> {
    InstrumentFootprint::parse(t)
}

pub fn output_transform_token(o: OutputTransform) -> &'static str {
    match o {
        OutputTransform::AbiReflectance => "abi-reflectance",
        OutputTransform::DebugSrgb => "debug-srgb",
    }
}

pub fn output_transform_from_token(t: &str) -> Option<OutputTransform> {
    [OutputTransform::AbiReflectance, OutputTransform::DebugSrgb]
        .into_iter()
        .find(|o| output_transform_token(*o) == t)
}

pub fn step_quality_token(s: StepQuality) -> &'static str {
    match s {
        StepQuality::Offline => "offline",
        StepQuality::Interactive => "interactive",
    }
}

pub fn step_quality_from_token(t: &str) -> Option<StepQuality> {
    [StepQuality::Offline, StepQuality::Interactive]
        .into_iter()
        .find(|s| step_quality_token(*s) == t)
}

pub fn fractional_cloud_mode_token(mode: FractionalCloudMode) -> &'static str {
    mode.slug()
}

pub fn fractional_cloud_mode_from_token(t: &str) -> Option<FractionalCloudMode> {
    match t {
        "off" => Some(FractionalCloudMode::Off),
        "effective-od" | "on" => Some(FractionalCloudMode::EffectiveOd),
        "deterministic-2" => Some(FractionalCloudMode::Deterministic2),
        "deterministic-4" => Some(FractionalCloudMode::Deterministic4),
        "deterministic-8" => Some(FractionalCloudMode::Deterministic8),
        "deterministic-16" => Some(FractionalCloudMode::Deterministic16),
        _ => None,
    }
}

// ── load / save ───────────────────────────────────────────────────────────────

/// The settings file path: `%LOCALAPPDATA%/SimSatStudio/settings.json` on Windows
/// (beside the app's cache dir), with the same XDG/HOME/temp fallbacks the cache
/// dir uses on other platforms (the headless nodes' tests pass explicit paths).
pub fn settings_path() -> PathBuf {
    let nonempty = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
    if let Some(local) = nonempty("LOCALAPPDATA") {
        return PathBuf::from(local)
            .join("SimSatStudio")
            .join("settings.json");
    }
    if let Some(xdg) = nonempty("XDG_CONFIG_HOME") {
        return PathBuf::from(xdg)
            .join("simsat-studio")
            .join("settings.json");
    }
    if let Some(home) = nonempty("HOME") {
        return PathBuf::from(home)
            .join(".config")
            .join("simsat-studio")
            .join("settings.json");
    }
    std::env::temp_dir()
        .join("simsat-studio")
        .join("settings.json")
}

/// Load settings from `path`. NEVER fails: a missing / unreadable / corrupt file
/// returns the defaults, and whatever parses is sanitized (clamped + validated).
pub fn load(path: &Path) -> StudioSettings {
    let mut s = std::fs::read_to_string(path)
        .ok()
        .and_then(|text| {
            let value = serde_json::from_str::<serde_json::Value>(&text).ok()?;
            let had_calibration_epoch = value.get("visible_calibration_epoch").is_some();
            let saved_calibration_epoch = value
                .get("visible_calibration_epoch")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let had_land_sza = value.get("land_sza_normalization").is_some();
            let had_land_sza_max_gain = value.get("land_sza_max_gain").is_some();
            let had_land_toe = value.get("land_dark_toe").is_some();
            let had_exposed_edge = value.get("feather_exposed_domain_edges").is_some();
            let had_twilight_surface_recovery = value.get("twilight_surface_recovery").is_some();
            let had_ground_gain = value.get("ground_gain").is_some();
            let mut settings = serde_json::from_value::<StudioSettings>(value).ok()?;
            // Files from before epoch 4 either persisted these switches as off or did
            // not know about them. Missing legacy fields therefore mean identity, not
            // the new-install preset; the banner lets the user Apply or Keep explicitly.
            if saved_calibration_epoch < LAND_APPEARANCE_CALIBRATION_EPOCH {
                if !had_land_sza {
                    settings.land_sza_normalization = false;
                }
                if !had_land_toe {
                    settings.land_dark_toe = false;
                }
            }
            // Files from before epoch 5 either explicitly kept this switch off or
            // did not know about it. A missing legacy field means the exact former
            // margin-gated behavior; the banner offers Apply (on) versus Keep (off).
            if saved_calibration_epoch < EXPOSED_EDGE_CALIBRATION_EPOCH && !had_exposed_edge {
                settings.feather_exposed_domain_edges = false;
            }
            // Epoch 6 changes only the shipped SZA bound. If an older or hand-edited
            // settings file omitted the numeric field, preserve the former 1.6 look;
            // the calibration banner offers 4.0 explicitly instead of silently changing it.
            if saved_calibration_epoch < LAND_SZA_GAIN_CALIBRATION_EPOCH && !had_land_sza_max_gain {
                settings.land_sza_max_gain = LEGACY_LAND_SZA_MAX_GAIN;
            }
            // Epoch 7 promotes this display correction for new installs and reviewed visible
            // presets only. Older settings that predate the field retain exact identity and
            // surface the Apply/Keep banner instead of inheriting the new serde default.
            if saved_calibration_epoch < TWILIGHT_SURFACE_RECOVERY_CALIBRATION_EPOCH
                && !had_twilight_surface_recovery
            {
                settings.twilight_surface_recovery = false;
            }
            // Epoch 8 changes the shipped display ground lift. Preserve an older or
            // hand-edited file that omitted the numeric field at the previous neutral
            // value; the calibration banner offers the reviewed 1.10 explicitly.
            if saved_calibration_epoch < GROUND_LIFT_CALIBRATION_EPOCH && !had_ground_gain {
                settings.ground_gain = LEGACY_GROUND_DAY_LIFT;
            }
            if !had_calibration_epoch {
                // Preserve every old user-selected value. Epoch zero merely asks the UI
                // to offer the new shipped preset or explicitly keep those saved values.
                settings.visible_calibration_epoch = 0;
            }
            Some(settings)
        })
        .unwrap_or_default();
    s.sanitize();
    s
}

/// Atomically save settings to `path`: write a `.tmp` sibling, then rename over
/// the real file (rename replaces on Windows and POSIX), so a crash mid-write can
/// never leave a truncated settings file behind.
pub fn save(path: &Path, s: &StudioSettings) -> Result<(), String> {
    let dir = path
        .parent()
        .ok_or_else(|| format!("settings path has no parent: {}", path.display()))?;
    std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let json = serde_json::to_string_pretty(s).map_err(|e| format!("serialize settings: {e}"))?;
    let mut tmp_name = path
        .file_name()
        .map(|n| n.to_os_string())
        .ok_or_else(|| format!("settings path has no file name: {}", path.display()))?;
    tmp_name.push(".tmp");
    let tmp = dir.join(tmp_name);
    std::fs::write(&tmp, json).map_err(|e| format!("write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("rename {}: {e}", path.display()))?;
    Ok(())
}

// ── recent-files list operations ──────────────────────────────────────────────

/// Push an open action to the front of the recent list: an identical entry is
/// deduped (moved to the front), and the list is capped at [`RECENT_CAP`].
pub fn push_recent(list: &mut Vec<RecentEntry>, entry: RecentEntry) {
    list.retain(|e| e != &entry);
    list.insert(0, entry);
    list.truncate(RECENT_CAP);
}

/// Drop recent entries whose FIRST path no longer exists (`exists` injected for
/// testability). A multi-file sequence is judged by its first file.
pub fn prune_recent(list: &mut Vec<RecentEntry>, exists: &dyn Fn(&str) -> bool) {
    list.retain(|e| e.paths.first().is_some_and(|p| exists(p)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shipped_rc_defaults_match_the_reviewed_preset() {
        let s = StudioSettings::default();
        assert_eq!(s.visible_calibration_epoch, VISIBLE_CALIBRATION_EPOCH);
        assert_eq!(s.aod, simsat::atmosphere::DEFAULT_AOD as f32);
        assert!(!s.rh_swelling);
        assert!(s.atmosphere_correction);
        assert!(s.terrain_atmosphere);
        assert!(s.land_sza_normalization);
        assert_eq!(
            s.land_sza_max_gain,
            simsat::render::LAND_SZA_MAX_GAIN as f32
        );
        assert!(s.land_dark_toe);
        assert_eq!(
            s.land_dark_toe_knee,
            simsat::render::LAND_DARK_TOE_KNEE as f32
        );
        assert_eq!(
            s.land_dark_toe_gamma,
            simsat::render::LAND_DARK_TOE_GAMMA as f32
        );
        assert_eq!(
            s.land_dark_toe_max_gain,
            simsat::render::LAND_DARK_TOE_MAX_GAIN as f32
        );
        assert!(!s.surface_postlight_toe);
        assert_eq!(
            s.surface_postlight_toe_knee,
            simsat::render::SURFACE_POSTLIGHT_TOE_KNEE as f32
        );
        assert_eq!(
            s.surface_postlight_toe_gamma,
            simsat::render::SURFACE_POSTLIGHT_TOE_GAMMA as f32
        );
        assert_eq!(
            s.surface_postlight_toe_max_gain,
            simsat::render::SURFACE_POSTLIGHT_TOE_MAX_GAIN as f32
        );
        assert!(s.twilight_surface_recovery);
        assert_eq!(
            s.twilight_surface_recovery_knee,
            simsat::render::TWILIGHT_SURFACE_RECOVERY_KNEE as f32
        );
        assert_eq!(
            s.twilight_surface_recovery_gamma,
            simsat::render::TWILIGHT_SURFACE_RECOVERY_GAMMA as f32
        );
        assert_eq!(
            s.twilight_surface_recovery_max_gain,
            simsat::render::TWILIGHT_SURFACE_RECOVERY_MAX_GAIN as f32
        );
        assert_eq!(s.output_transform, "abi-reflectance");
        assert_eq!(s.render_intent, "display");
        assert_eq!(s.ir_enhancement, "cimss");
        assert_eq!(s.instrument_footprint, "off");
        assert!(s.clouds_enabled);
        assert!(s.multiscatter);
        assert!(!s.delta_flux_clouds);
        assert!(!s.delta_flux_v2_clouds);
        assert!(!s.delta_flux_v3_clouds);
        assert!(!s.beer_powder);
        assert_eq!(s.cloud_optical_depth_scale, 0.15);
        assert!(s.feather_exposed_domain_edges);
        assert_eq!(s.exposure, simsat::render::DEFAULT_EXPOSURE as f32);
        assert_eq!(s.ground_gain, simsat::render::GROUND_DAY_LIFT as f32);
        assert_eq!(s.cloud_softclip, 0.65);
        assert_eq!(s.cloud_highlight_max, 1.25);
        assert!(s.fractional_clouds);
        assert_eq!(s.fractional_cloud_mode, "deterministic-2");
        assert!(!s.granulation);
        assert!(!s.topdown_stratiform_regularization);
        assert!(!s.topdown_cloud_footprint);
        assert!(s.topdown_shadow_antialias);
    }

    #[test]
    fn shipped_visible_calibration_migration_is_complete_and_scoped() {
        let recent = vec![RecentEntry {
            kind: "wrfout".to_string(),
            paths: vec!["C:/runs/keep-me".to_string()],
        }];
        let mut s = StudioSettings {
            visible_calibration_epoch: 2,
            sat: "himawari".to_string(),
            recent: recent.clone(),
            output_transform: "debug-srgb".to_string(),
            aod: 0.6,
            rh_swelling: true,
            atmosphere_correction: false,
            terrain_atmosphere: false,
            land_sza_normalization: false,
            land_sza_max_gain: 3.0,
            land_dark_toe: false,
            land_dark_toe_knee: 0.2,
            land_dark_toe_gamma: 0.2,
            land_dark_toe_max_gain: 3.0,
            clouds_enabled: false,
            fractional_clouds: false,
            fractional_cloud_mode: "deterministic-4".to_string(),
            multiscatter: false,
            cloud_optical_depth_scale: 2.0,
            feather_exposed_domain_edges: false,
            beer_powder: true,
            granulation: true,
            topdown_stratiform_regularization: true,
            topdown_cloud_footprint: true,
            topdown_shadow_antialias: false,
            exposure: 4.0,
            ground_gain: 4.0,
            cloud_softclip: 1.0,
            cloud_highlight_max: 4.0,
            ..Default::default()
        };
        s.apply_shipped_visible_calibration();
        let d = StudioSettings::default();
        assert_eq!(s.visible_calibration_epoch, VISIBLE_CALIBRATION_EPOCH);
        assert_eq!(s.output_transform, d.output_transform);
        assert_eq!(s.aod, d.aod);
        assert_eq!(s.rh_swelling, d.rh_swelling);
        assert_eq!(s.atmosphere_correction, d.atmosphere_correction);
        assert_eq!(s.terrain_atmosphere, d.terrain_atmosphere);
        assert_eq!(s.land_sza_normalization, d.land_sza_normalization);
        assert_eq!(s.land_sza_max_gain, d.land_sza_max_gain);
        assert_eq!(s.land_dark_toe, d.land_dark_toe);
        assert_eq!(s.land_dark_toe_knee, d.land_dark_toe_knee);
        assert_eq!(s.land_dark_toe_gamma, d.land_dark_toe_gamma);
        assert_eq!(s.land_dark_toe_max_gain, d.land_dark_toe_max_gain);
        assert_eq!(s.surface_postlight_toe, d.surface_postlight_toe);
        assert_eq!(s.twilight_surface_recovery, d.twilight_surface_recovery);
        assert_eq!(s.clouds_enabled, d.clouds_enabled);
        assert_eq!(s.fractional_clouds, d.fractional_clouds);
        assert_eq!(s.fractional_cloud_mode, d.fractional_cloud_mode);
        assert_eq!(s.multiscatter, d.multiscatter);
        assert_eq!(s.delta_flux_clouds, d.delta_flux_clouds);
        assert_eq!(s.delta_flux_v2_clouds, d.delta_flux_v2_clouds);
        assert_eq!(s.delta_flux_v3_clouds, d.delta_flux_v3_clouds);
        assert_eq!(s.cloud_optical_depth_scale, d.cloud_optical_depth_scale);
        assert_eq!(
            s.feather_exposed_domain_edges,
            d.feather_exposed_domain_edges
        );
        assert_eq!(s.beer_powder, d.beer_powder);
        assert_eq!(s.granulation, d.granulation);
        assert_eq!(
            s.topdown_stratiform_regularization,
            d.topdown_stratiform_regularization
        );
        assert_eq!(s.topdown_cloud_footprint, d.topdown_cloud_footprint);
        assert_eq!(s.topdown_shadow_antialias, d.topdown_shadow_antialias);
        assert_eq!(s.exposure, d.exposure);
        assert_eq!(s.ground_gain, d.ground_gain);
        assert_eq!(s.cloud_softclip, d.cloud_softclip);
        assert_eq!(s.cloud_highlight_max, d.cloud_highlight_max);
        assert_eq!(s.sat, "himawari", "unrelated settings remain intact");
        assert_eq!(s.recent, recent, "recent files remain intact");
    }

    #[test]
    fn tokens_round_trip_every_variant() {
        for p in SatellitePreset::ALL {
            assert_eq!(sat_from_token(sat_token(p)), Some(p));
        }
        assert_eq!(sat_token(SatellitePreset::MtgI1), "mtgi1");
        assert_eq!(sat_from_token("mtgi1"), Some(SatellitePreset::MtgI1));
        for alias in ["mtg", "mtg-i1", "meteosat-12", "meteosat12"] {
            assert_eq!(sat_from_token(alias), Some(SatellitePreset::MtgI1));
        }
        for r in ResolutionMode::ALL {
            assert_eq!(resolution_from_token(resolution_token(r)), Some(r));
        }
        for v in StudioView::ALL {
            assert_eq!(view_from_token(view_token(v)), Some(v));
        }
        for m in RenderMode::ALL {
            assert_eq!(mode_from_token(mode_token(m)), Some(m));
        }
        for intent in [RenderIntent::Display, RenderIntent::SensorFastGray] {
            assert_eq!(
                render_intent_from_token(render_intent_token(intent)),
                Some(intent)
            );
        }
        for e in IrEnhancement::ALL {
            assert_eq!(enhancement_from_token(enhancement_token(e)), Some(e));
        }
        for footprint in InstrumentFootprint::ALL {
            assert_eq!(
                instrument_footprint_from_token(instrument_footprint_token(footprint)),
                Some(footprint)
            );
        }
        for o in [OutputTransform::AbiReflectance, OutputTransform::DebugSrgb] {
            assert_eq!(
                output_transform_from_token(output_transform_token(o)),
                Some(o)
            );
        }
        for s in [StepQuality::Offline, StepQuality::Interactive] {
            assert_eq!(step_quality_from_token(step_quality_token(s)), Some(s));
        }
        for mode in [
            FractionalCloudMode::Off,
            FractionalCloudMode::EffectiveOd,
            FractionalCloudMode::Deterministic2,
            FractionalCloudMode::Deterministic4,
            FractionalCloudMode::Deterministic8,
            FractionalCloudMode::Deterministic16,
        ] {
            assert_eq!(
                fractional_cloud_mode_from_token(fractional_cloud_mode_token(mode)),
                Some(mode)
            );
        }
        // Unknown tokens map to None (they reset to defaults in sanitize).
        assert_eq!(mode_from_token("does-not-exist"), None);
        assert_eq!(sat_from_token(""), None);
    }

    #[test]
    fn mtg_camera_selection_round_trips_the_settings_file() {
        let dir =
            std::env::temp_dir().join(format!("simsat-settings-mtgi1-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("settings.json");
        let selected = StudioSettings {
            sat: "meteosat-12".to_string(),
            ..Default::default()
        };

        save(&path, &selected).expect("save MTG-I1 selection");
        let loaded = load(&path);

        assert_eq!(loaded.sat, "mtgi1");
        assert_eq!(sat_from_token(&loaded.sat), Some(SatellitePreset::MtgI1));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_missing_or_corrupt_falls_back_to_defaults() {
        let dir = std::env::temp_dir().join(format!("simsat-settings-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        // Missing file -> defaults.
        let missing = dir.join("nope").join("settings.json");
        assert_eq!(load(&missing), StudioSettings::default());
        // Corrupt file -> defaults (never a panic, never a partial junk state).
        let corrupt = dir.join("corrupt.json");
        std::fs::write(&corrupt, "{ this is not json").unwrap();
        assert_eq!(load(&corrupt), StudioSettings::default());
        // A wrong-typed field fails the parse -> full defaults.
        let wrong = dir.join("wrong.json");
        std::fs::write(&wrong, r#"{ "exposure": "very bright" }"#).unwrap();
        assert_eq!(load(&wrong), StudioSettings::default());
        // A PARTIAL file keeps its good fields and defaults the rest.
        let partial = dir.join("partial.json");
        std::fs::write(&partial, r#"{ "sat": "himawari", "exposure": 2.0 }"#).unwrap();
        let s = load(&partial);
        assert_eq!(s.sat, "himawari");
        assert_eq!(s.exposure, 2.0);
        assert_eq!(s.mode, StudioSettings::default().mode);
        assert_eq!(s.render_intent, "display");
        assert_eq!(
            s.visible_calibration_epoch, 0,
            "pre-epoch files must preserve values and request an explicit migration choice"
        );
        // Backward compatibility: settings written before the land controls existed
        // retain the legacy identity until the user accepts the epoch-4 preset.
        assert!(s.atmosphere_correction);
        assert!(s.terrain_atmosphere);
        assert!(!s.land_sza_normalization);
        assert!(!s.land_dark_toe);
        assert!(!s.twilight_surface_recovery);
        assert_eq!(
            s.land_sza_max_gain, LEGACY_LAND_SZA_MAX_GAIN,
            "pre-epoch files that omit the field preserve the former shipped bound"
        );
        assert!(s.fractional_clouds);
        assert_eq!(
            s.cloud_optical_depth_scale,
            DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE
        );
        assert!(
            !s.feather_exposed_domain_edges,
            "older settings default the new experiment off"
        );
        assert!(!s.beer_powder);
        assert!(!s.granulation);
        assert!(!s.topdown_stratiform_regularization);
        assert!(
            s.topdown_shadow_antialias,
            "older settings inherit the shipped anti-alias default"
        );
        assert_eq!(
            s.ground_gain, LEGACY_GROUND_DAY_LIFT,
            "pre-epoch files that omit ground gain keep the former neutral display"
        );
        assert_eq!(s.cloud_softclip, StudioSettings::default().cloud_softclip);
        assert_eq!(
            s.cloud_highlight_max,
            StudioSettings::default().cloud_highlight_max
        );
        // IR palette selection is independently persisted. Changing the shipped
        // default must not rewrite an explicit Natural choice or require a visible-
        // calibration epoch bump.
        let explicit_natural = dir.join("explicit-natural-ir.json");
        std::fs::write(
            &explicit_natural,
            format!(
                r#"{{
                    "visible_calibration_epoch": {},
                    "mode": "ir-band13",
                    "ir_enhancement": "natural"
                }}"#,
                VISIBLE_CALIBRATION_EPOCH
            ),
        )
        .unwrap();
        let saved_natural = load(&explicit_natural);
        assert_eq!(saved_natural.ir_enhancement, "natural");
        assert_eq!(
            saved_natural.visible_calibration_epoch,
            VISIBLE_CALIBRATION_EPOCH
        );
        // A pre-epoch file's explicit calibration is never silently rewritten. The
        // Studio uses epoch zero to show its apply/keep banner.
        let legacy = dir.join("legacy-calibration.json");
        std::fs::write(
            &legacy,
            r#"{
                "cloud_optical_depth_scale": 0.5,
                "exposure": 1.5,
                "ground_gain": 2.0,
                "cloud_softclip": 0.75,
                "cloud_highlight_max": 1.05,
                "fractional_clouds": false,
                "granulation": true
            }"#,
        )
        .unwrap();
        let old = load(&legacy);
        assert_eq!(old.visible_calibration_epoch, 0);
        assert_eq!(old.cloud_optical_depth_scale, 0.5);
        assert_eq!(old.exposure, 1.5);
        assert_eq!(old.ground_gain, 2.0);
        assert_eq!(old.cloud_softclip, 0.75);
        assert_eq!(old.cloud_highlight_max, 1.05);
        assert!(!old.fractional_clouds);
        assert!(old.granulation);

        // The earlier diagnostic RC wrote epoch 1 with OD 0.25. Loading it must
        // preserve that explicit value while leaving the epoch behind the current
        // one, so the Studio offers the owner-selected 0.15 release preset.
        let diagnostic_rc = dir.join("diagnostic-rc.json");
        std::fs::write(
            &diagnostic_rc,
            r#"{
                "visible_calibration_epoch": 1,
                "cloud_optical_depth_scale": 0.25,
                "exposure": 1.6,
                "ground_gain": 1.6
            }"#,
        )
        .unwrap();
        let old_rc = load(&diagnostic_rc);
        assert_eq!(old_rc.visible_calibration_epoch, 1);
        assert_eq!(old_rc.cloud_optical_depth_scale, 0.25);
        assert_eq!(old_rc.exposure, 1.6);
        assert_eq!(old_rc.ground_gain, 1.6);
        assert!(old_rc.visible_calibration_epoch < VISIBLE_CALIBRATION_EPOCH);

        // The released v0.1.4 epoch is likewise retained verbatim and offered the
        // v0.1.5 exposure migration rather than being rewritten behind the owner.
        let v014 = dir.join("v014-calibration.json");
        std::fs::write(
            &v014,
            r#"{
                "visible_calibration_epoch": 2,
                "cloud_optical_depth_scale": 0.15,
                "exposure": 1.0,
                "ground_gain": 1.0
            }"#,
        )
        .unwrap();
        let old_v014 = load(&v014);
        assert_eq!(old_v014.visible_calibration_epoch, 2);
        assert_eq!(old_v014.cloud_optical_depth_scale, 0.15);
        assert_eq!(old_v014.exposure, 1.0);
        assert_eq!(old_v014.ground_gain, 1.0);
        assert!(old_v014.visible_calibration_epoch < VISIBLE_CALIBRATION_EPOCH);

        // Epoch 7 was the v0.2.1 RC1 calibration immediately before the reviewed
        // 1.10 ground lift. Loading RC1 settings must preserve either its former
        // neutral default or any explicit owner customization and offer the epoch-8
        // Apply/Keep choice instead of silently changing the image.
        for (name, json, expected_ground) in [
            (
                "epoch7-missing-ground.json",
                r#"{ "visible_calibration_epoch": 7 }"#,
                LEGACY_GROUND_DAY_LIFT,
            ),
            (
                "epoch7-neutral-ground.json",
                r#"{ "visible_calibration_epoch": 7, "ground_gain": 1.0 }"#,
                1.0,
            ),
            (
                "epoch7-custom-ground.json",
                r#"{ "visible_calibration_epoch": 7, "ground_gain": 1.27 }"#,
                1.27,
            ),
        ] {
            let path = dir.join(name);
            std::fs::write(&path, json).unwrap();
            let saved = load(&path);
            assert_eq!(saved.visible_calibration_epoch, 7, "{name}");
            assert_eq!(saved.ground_gain, expected_ground, "{name}");
        }

        // Epoch 3 selected exposure 1.5 while the two land operators were still off.
        // Preserve that exact saved look and leave the epoch behind 4 so the Studio
        // offers Apply (both on) versus Keep (both off) instead of silently changing it.
        let exposure_only_v015 = dir.join("v015-exposure-only-calibration.json");
        std::fs::write(
            &exposure_only_v015,
            r#"{
                "visible_calibration_epoch": 3,
                "cloud_optical_depth_scale": 0.15,
                "exposure": 1.5,
                "land_sza_normalization": false,
                "land_dark_toe": false
            }"#,
        )
        .unwrap();
        let old_v015 = load(&exposure_only_v015);
        assert_eq!(old_v015.visible_calibration_epoch, 3);
        assert_eq!(old_v015.exposure, 1.5);
        assert!(!old_v015.land_sza_normalization);
        assert!(!old_v015.land_dark_toe);
        assert!(old_v015.visible_calibration_epoch < VISIBLE_CALIBRATION_EPOCH);

        // Epoch 4 selected both land operators while exposed-domain edge feathering
        // was still off. Preserve the saved switch verbatim and leave the epoch
        // behind 5 so Studio offers Apply (edge on) versus Keep (edge off).
        let land_only_v015 = dir.join("v015-land-only-calibration.json");
        std::fs::write(
            &land_only_v015,
            r#"{
                "visible_calibration_epoch": 4,
                "cloud_optical_depth_scale": 0.15,
                "exposure": 1.5,
                "land_sza_normalization": true,
                "land_dark_toe": true,
                "feather_exposed_domain_edges": false
            }"#,
        )
        .unwrap();
        let old_land_v015 = load(&land_only_v015);
        assert_eq!(old_land_v015.visible_calibration_epoch, 4);
        assert!(old_land_v015.land_sza_normalization);
        assert_eq!(old_land_v015.land_sza_max_gain, LEGACY_LAND_SZA_MAX_GAIN);
        assert!(old_land_v015.land_dark_toe);
        assert!(!old_land_v015.feather_exposed_domain_edges);
        assert!(old_land_v015.visible_calibration_epoch < VISIBLE_CALIBRATION_EPOCH);

        // Be conservative with a hand-edited epoch-4 file that omitted the then-new
        // field: missing still means the former off behavior, never a silent rewrite.
        let land_only_missing_edge = dir.join("v015-land-only-missing-edge.json");
        std::fs::write(
            &land_only_missing_edge,
            r#"{
                "visible_calibration_epoch": 4,
                "land_sza_normalization": true,
                "land_dark_toe": true
            }"#,
        )
        .unwrap();
        let old_missing_edge = load(&land_only_missing_edge);
        assert_eq!(old_missing_edge.visible_calibration_epoch, 4);
        assert_eq!(old_missing_edge.land_sza_max_gain, LEGACY_LAND_SZA_MAX_GAIN);
        assert!(!old_missing_edge.feather_exposed_domain_edges);

        // Epoch 5 was the v0.1.9 shipped look. Preserve its explicit 1.6 SZA bound
        // and leave the epoch behind 6 so the Studio offers the owner-selected 4.0.
        let v019 = dir.join("v019-calibration.json");
        std::fs::write(
            &v019,
            r#"{
                "visible_calibration_epoch": 5,
                "land_sza_normalization": true,
                "land_sza_max_gain": 1.6,
                "land_dark_toe": true,
                "feather_exposed_domain_edges": true
            }"#,
        )
        .unwrap();
        let old_v019 = load(&v019);
        assert_eq!(old_v019.visible_calibration_epoch, 5);
        assert_eq!(old_v019.land_sza_max_gain, LEGACY_LAND_SZA_MAX_GAIN);
        assert!(old_v019.visible_calibration_epoch < VISIBLE_CALIBRATION_EPOCH);

        // A hand-edited epoch-5 file that omitted the numeric field must also keep
        // the former bound instead of inheriting the new serde default silently.
        let v019_missing_gain = dir.join("v019-missing-sza-gain.json");
        std::fs::write(
            &v019_missing_gain,
            r#"{
                "visible_calibration_epoch": 5,
                "land_sza_normalization": true,
                "land_dark_toe": true,
                "feather_exposed_domain_edges": true
            }"#,
        )
        .unwrap();
        let old_v019_missing_gain = load(&v019_missing_gain);
        assert_eq!(old_v019_missing_gain.visible_calibration_epoch, 5);
        assert_eq!(
            old_v019_missing_gain.land_sza_max_gain,
            LEGACY_LAND_SZA_MAX_GAIN
        );

        // Epoch 6 predates the shipped twilight recovery. A missing field must preserve
        // identity and leave the migration banner active; an explicit saved choice is kept.
        let v020_missing_twilight = dir.join("v020-missing-twilight-recovery.json");
        std::fs::write(
            &v020_missing_twilight,
            r#"{
                "visible_calibration_epoch": 6,
                "land_sza_normalization": true,
                "land_sza_max_gain": 4.0,
                "land_dark_toe": true,
                "feather_exposed_domain_edges": true
            }"#,
        )
        .unwrap();
        let old_v020_missing = load(&v020_missing_twilight);
        assert_eq!(old_v020_missing.visible_calibration_epoch, 6);
        assert!(!old_v020_missing.twilight_surface_recovery);
        assert!(old_v020_missing.visible_calibration_epoch < VISIBLE_CALIBRATION_EPOCH);

        let v020_explicit_twilight = dir.join("v020-explicit-twilight-recovery.json");
        std::fs::write(
            &v020_explicit_twilight,
            r#"{
                "visible_calibration_epoch": 6,
                "twilight_surface_recovery": true,
                "twilight_surface_recovery_knee": 0.25,
                "twilight_surface_recovery_gamma": 0.60,
                "twilight_surface_recovery_max_gain": 3.0
            }"#,
        )
        .unwrap();
        let old_v020_explicit = load(&v020_explicit_twilight);
        assert!(old_v020_explicit.twilight_surface_recovery);
        assert_eq!(old_v020_explicit.twilight_surface_recovery_knee, 0.25);
        assert_eq!(old_v020_explicit.twilight_surface_recovery_gamma, 0.60);
        assert_eq!(old_v020_explicit.twilight_surface_recovery_max_gain, 3.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sanitize_clamps_and_resets_unknown_tokens() {
        let mut s = StudioSettings {
            margin_pct: 250.0,
            aod: -3.0,
            land_sza_max_gain: 99.0,
            land_dark_toe_knee: 0.0,
            land_dark_toe_gamma: f32::NAN,
            land_dark_toe_max_gain: -2.0,
            surface_postlight_toe: true,
            surface_postlight_toe_knee: 0.0,
            surface_postlight_toe_gamma: f32::NAN,
            surface_postlight_toe_max_gain: 9.0,
            cloud_optical_depth_scale: 99.0,
            fractional_clouds: false,
            fractional_cloud_mode: "random-16".to_string(),
            exposure: 99.0,
            ground_gain: 99.0,
            cloud_softclip: -3.0,
            cloud_highlight_max: f32::NAN,
            play_fps: 0.0,
            frame_cap: 1,
            bm_month_override: 13,
            sat: "geostationary-9".to_string(),
            mode: "x-ray".to_string(),
            render_intent: "magic-sensor".to_string(),
            instrument_footprint: "box-blur".to_string(),
            nssl_native_cloud_optics: true,
            hrrr_thompson_native_cloud_optics: true,
            orbit_tilt_deg: 90.0,
            orbit_range_km: f32::NAN,
            persp_width: 100_000,
            persp_height: 0,
            ..Default::default()
        };
        s.sanitize();
        assert_eq!(s.margin_pct, 100.0);
        assert_eq!(s.aod, 0.0);
        assert_eq!(s.land_sza_max_gain, 4.0);
        assert_eq!(s.land_dark_toe_knee, 1.0e-6);
        assert_eq!(
            s.land_dark_toe_gamma,
            StudioSettings::default().land_dark_toe_gamma
        );
        assert_eq!(s.land_dark_toe_max_gain, 1.0);
        assert!(s.surface_postlight_toe);
        assert_eq!(s.surface_postlight_toe_knee, 1.0e-6);
        assert_eq!(
            s.surface_postlight_toe_gamma,
            StudioSettings::default().surface_postlight_toe_gamma
        );
        assert_eq!(s.surface_postlight_toe_max_gain, 4.0);
        assert_eq!(s.cloud_optical_depth_scale, 4.0);
        assert_eq!(s.fractional_cloud_mode, "deterministic-2");
        assert!(
            !s.fractional_clouds,
            "sanitize must preserve an explicit legacy A/B"
        );
        assert_eq!(s.exposure, 4.0);
        assert_eq!(s.ground_gain, 4.0);
        assert_eq!(s.cloud_softclip, 0.05);
        assert_eq!(
            s.cloud_highlight_max,
            StudioSettings::default().cloud_highlight_max
        );
        assert_eq!(s.play_fps, 1.0);
        assert_eq!(s.frame_cap, 8);
        assert_eq!(s.bm_month_override, 0);
        assert_eq!(s.sat, StudioSettings::default().sat);
        assert_eq!(s.mode, StudioSettings::default().mode);
        assert_eq!(s.render_intent, StudioSettings::default().render_intent);
        assert_eq!(
            s.instrument_footprint,
            StudioSettings::default().instrument_footprint
        );
        assert!(s.hrrr_thompson_native_cloud_optics);
        assert!(
            !s.nssl_native_cloud_optics,
            "sanitize canonicalizes the two legacy booleans with worker precedence"
        );
        // Perspective orbit fields clamp to the slider ranges (NaN -> the default).
        assert_eq!(s.orbit_tilt_deg, 85.0);
        assert_eq!(s.orbit_range_km, StudioSettings::default().orbit_range_km);
        assert_eq!(s.persp_width, 4096);
        assert_eq!(s.persp_height, 2);
        // Non-finite scale cannot enter JSON normally, but sanitize is defensive
        // for programmatic callers too and restores the shipped calibration.
        s.cloud_optical_depth_scale = f32::NAN;
        s.sanitize();
        assert_eq!(
            s.cloud_optical_depth_scale,
            DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE
        );
        // Idempotent.
        let once = s.clone();
        s.sanitize();
        assert_eq!(s, once);
    }

    #[test]
    fn recent_push_dedupes_caps_and_prune_drops_missing() {
        let entry = |p: &str| RecentEntry {
            kind: "wrfout".to_string(),
            paths: vec![p.to_string()],
        };
        let mut list = Vec::new();
        for i in 0..10 {
            push_recent(&mut list, entry(&format!("f{i}")));
        }
        assert_eq!(list.len(), RECENT_CAP, "capped at {RECENT_CAP}");
        assert_eq!(list[0].paths[0], "f9", "newest first");
        // Re-opening an existing entry moves it to the front without duplicating.
        push_recent(&mut list, entry("f5"));
        assert_eq!(list.len(), RECENT_CAP);
        assert_eq!(list[0].paths[0], "f5");
        assert_eq!(
            list.iter().filter(|e| e.paths[0] == "f5").count(),
            1,
            "deduped"
        );
        // Prune drops entries whose first path is gone.
        prune_recent(&mut list, &|p| p != "f5" && p != "f7");
        assert!(
            list.iter()
                .all(|e| e.paths[0] != "f5" && e.paths[0] != "f7")
        );
        // Sanitize also drops malformed kinds/paths and dedupes.
        let mut s = StudioSettings {
            recent: vec![
                entry("a"),
                RecentEntry {
                    kind: "bogus".to_string(),
                    paths: vec!["x".to_string()],
                },
                RecentEntry {
                    kind: "cached".to_string(),
                    paths: vec![],
                },
                entry("a"),
            ],
            ..Default::default()
        };
        s.sanitize();
        assert_eq!(s.recent.len(), 1);
        assert_eq!(s.recent[0].paths[0], "a");
    }

    #[test]
    fn save_is_atomic_and_round_trips() {
        let dir = std::env::temp_dir().join(format!("simsat-settings-save-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("settings.json");
        let mut s = StudioSettings {
            sat: "goes-west".to_string(),
            render_intent: "sensor-fast-gray".to_string(),
            instrument_footprint: "goes-r-abi-band13-mtf-prototype".to_string(),
            exposure: 2.5,
            atmosphere_correction: false,
            terrain_atmosphere: false,
            land_sza_normalization: false,
            land_sza_max_gain: 1.7,
            land_dark_toe: false,
            land_dark_toe_knee: 0.07,
            land_dark_toe_gamma: 0.6,
            land_dark_toe_max_gain: 1.4,
            fractional_clouds: false,
            cloud_optical_depth_scale: 0.5,
            feather_exposed_domain_edges: true,
            beer_powder: true,
            granulation: true,
            ground_gain: 1.6,
            cloud_softclip: 0.65,
            cloud_highlight_max: 1.25,
            recent: vec![RecentEntry {
                kind: "sequence".to_string(),
                paths: vec!["C:/runs/enderlin".to_string()],
            }],
            ..Default::default()
        };
        save(&path, &s).expect("save (creates the directory)");
        // The temp sibling is gone (renamed over the real file).
        assert!(!dir.join("settings.json.tmp").exists());
        assert_eq!(load(&path), s, "round trip");
        // Overwrite works too (rename replaces the existing file).
        s.exposure = 3.0;
        save(&path, &s).expect("resave");
        assert_eq!(load(&path).exposure, 3.0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
