//! Canonical one-click Studio preset planning.
//!
//! A preset is planned against the same persisted [`StudioSettings`] snapshot the
//! application writes to disk.  The planner returns both the resulting snapshot
//! and an exhaustive, user-facing diff; the UI never carries a second copy of the
//! policy.  Session-only GPU state rides beside the snapshot so every CPU change is
//! also visible.

use std::fmt;

use simsat::camera::{GeoNavigation, ResolutionMode, SatellitePreset};
use simsat::instrument_footprint::InstrumentFootprint;
use simsat::render::{
    GROUND_DAY_LIFT, LandAppearanceConfig, SurfacePostlightToeConfig, TwilightSurfaceRecoveryConfig,
};
use simsat::thermal_sensor::ThermalSensor;

use crate::settings::{self, StudioSettings};
use crate::{RenderMode, StudioView};

/// The three deliberately small, reviewed Studio presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StudioPreset {
    RecommendedDisplay,
    HighQualityVisible,
    SensorQa,
}

impl StudioPreset {
    pub(crate) const ALL: [Self; 3] = [
        Self::RecommendedDisplay,
        Self::HighQualityVisible,
        Self::SensorQa,
    ];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::RecommendedDisplay => "Recommended Display",
            Self::HighQualityVisible => "High Quality Visible",
            Self::SensorQa => "Sensor QA",
        }
    }

    /// Compact label for the always-visible Quick mode row. The longer `label`
    /// remains the provenance/status wording.
    pub(crate) const fn quick_label(self) -> &'static str {
        match self {
            Self::RecommendedDisplay => "Recommended",
            Self::HighQualityVisible => "High Quality",
            Self::SensorQa => "Sensor QA",
        }
    }

    pub(crate) const fn description(self) -> &'static str {
        match self {
            Self::RecommendedDisplay => {
                "For Visible/GeoColor/Sandwich: owner-reviewed CPU Offline defaults, Native \
                 resolution, OD 0.15, exposure 1.5, AOD 0.05, fixed optics, deterministic \
                 2-subcolumn closure, corrections, tightly gated twilight terrain recovery, \
                 edge feathering, and top-down shadow anti-aliasing on. For IR Band 13: selects \
                 only the recommended CIMSS Style false-color isotherm display; saved palette \
                 choices are not silently migrated."
            }
            Self::HighQualityVisible => {
                "Recommended Display plus the owner-selected deterministic 4-subcolumn \
                 reference and 0.45 cloud-highlight knee. It remains on fixed optics and does \
                 not silently enable an experimental storage, footprint, or physics mode."
            }
            Self::SensorQa => {
                "CPU Offline sensor comparison: GOES-R ABI fixed-grid navigation, sensor \
                 sampling, neutral visible transforms, and official GOES-19 FM4 Band 13 for \
                 a compatible GOES-East IR product. Top-down shadow anti-aliasing and other \
                 display-only experiments remain off."
            }
        }
    }
}

/// Session-only render switches governed by a CPU preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PresetRuntime {
    pub(crate) gpu_clouds: bool,
    pub(crate) parity_pending: bool,
}

/// One exact before/after setting shown in the Studio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PresetChange {
    pub(crate) field: &'static str,
    pub(crate) before: String,
    pub(crate) after: String,
}

impl fmt::Display for PresetChange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} -> {}", self.field, self.before, self.after)
    }
}

/// A fully validated preset result. Applying `settings` and `runtime` is the
/// complete operation represented by `changes`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PresetPlan {
    pub(crate) settings: StudioSettings,
    pub(crate) runtime: PresetRuntime,
    pub(crate) changes: Vec<PresetChange>,
}

impl PresetPlan {
    pub(crate) fn change_summary(&self) -> String {
        if self.changes.is_empty() {
            return "Already active; no settings change.".to_string();
        }
        self.changes
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("; ")
    }
}

/// Why a preset cannot honestly apply to the current product/source selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PresetUnavailable(pub(crate) String);

impl fmt::Display for PresetUnavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Plan a preset without mutating the application.
pub(crate) fn plan(
    preset: StudioPreset,
    current: &StudioSettings,
    runtime: PresetRuntime,
) -> Result<PresetPlan, PresetUnavailable> {
    let mode = settings::mode_from_token(&current.mode)
        .ok_or_else(|| PresetUnavailable(format!("Unknown product token '{}'.", current.mode)))?;
    let view = settings::view_from_token(&current.view)
        .ok_or_else(|| PresetUnavailable(format!("Unknown view token '{}'.", current.view)))?;
    let satellite = settings::sat_from_token(&current.sat)
        .ok_or_else(|| PresetUnavailable(format!("Unknown satellite token '{}'.", current.sat)))?;

    validate_scope(preset, mode, view, satellite)?;

    let mut desired = current.clone();
    // Recommended IR is deliberately palette-only. Session-only visible/GPU state is
    // unrelated to thermal recoloring and must not change when the user picks it.
    let desired_runtime = if preset == StudioPreset::RecommendedDisplay && mode == RenderMode::Ir {
        runtime
    } else {
        PresetRuntime {
            gpu_clouds: false,
            parity_pending: false,
        }
    };

    match preset {
        StudioPreset::RecommendedDisplay => {
            if mode == RenderMode::Ir {
                apply_ir_display_baseline(&mut desired);
            } else {
                apply_display_baseline(&mut desired);
            }
        }
        StudioPreset::HighQualityVisible => {
            apply_display_baseline(&mut desired);
            desired.fractional_cloud_mode = "deterministic-4".to_string();
            desired.cloud_softclip = 0.45;
        }
        StudioPreset::SensorQa => apply_sensor_qa(&mut desired, mode),
    }

    let changes = collect_changes(current, &desired, runtime, desired_runtime);
    Ok(PresetPlan {
        settings: desired,
        runtime: desired_runtime,
        changes,
    })
}

/// Identify the reviewed Quick mode represented by the current persisted and
/// session-only settings. Any manual edit to a governed field outside an exact
/// reviewed plan is deliberately reported as Custom.
pub(crate) fn active(current: &StudioSettings, runtime: PresetRuntime) -> Option<StudioPreset> {
    StudioPreset::ALL.into_iter().find(|preset| {
        plan(*preset, current, runtime)
            .map(|planned| planned.changes.is_empty())
            .unwrap_or(false)
    })
}

fn validate_scope(
    preset: StudioPreset,
    mode: RenderMode,
    view: StudioView,
    satellite: SatellitePreset,
) -> Result<(), PresetUnavailable> {
    match preset {
        StudioPreset::RecommendedDisplay => {
            if !mode.uses_visible_controls() && mode != RenderMode::Ir {
                return Err(PresetUnavailable(format!(
                    "{} has no reviewed Recommended Display path. This preset never changes \
                     Mode; select Visible, IR Band 13, GeoColor Style, or Sandwich first.",
                    mode.label()
                )));
            }
        }
        StudioPreset::HighQualityVisible => {
            if !mode.uses_visible_controls() {
                return Err(PresetUnavailable(format!(
                    "{} does not use the visible display path. This preset never changes Mode; \
                     select Visible, GeoColor Style, or Sandwich first.",
                    mode.label()
                )));
            }
            if view == StudioView::Perspective {
                return Err(PresetUnavailable(
                    "Deterministic-4 fractional clouds are not supported in Perspective. \
                     Select Geostationary or Top-down; the preset will not silently change the \
                     camera."
                        .to_string(),
                ));
            }
        }
        StudioPreset::SensorQa => {
            if !matches!(mode, RenderMode::Visible | RenderMode::Ir) {
                return Err(PresetUnavailable(format!(
                    "{} is not an honest target for this preset. Sensor QA supports Visible \
                     (Fast Gray) and IR Band 13 only; it never converts the current product.",
                    mode.label()
                )));
            }
            if satellite == SatellitePreset::Himawari {
                return Err(PresetUnavailable(
                    "GOES-R ABI fixed-grid navigation is incompatible with Himawari. Select a \
                     GOES satellite; the preset will not relabel the source."
                        .to_string(),
                ));
            }
            if mode == RenderMode::Ir && satellite != SatellitePreset::GoesEast {
                return Err(PresetUnavailable(
                    "The available official Band 13 response is FM4/GOES-19 (GOES-East), not \
                     GOES-West. Select GOES-East for this IR Sensor QA preset."
                        .to_string(),
                ));
            }
        }
    }
    Ok(())
}

/// Owner-selected finished-visible baseline. Product, satellite, view, margin,
/// timestep, paths, export, and playback choices are deliberately untouched.
fn apply_display_baseline(s: &mut StudioSettings) {
    let defaults = StudioSettings::default();
    s.visible_calibration_epoch = settings::VISIBLE_CALIBRATION_EPOCH;
    s.instrument_footprint =
        settings::instrument_footprint_token(InstrumentFootprint::Off).to_string();
    s.resolution = settings::resolution_token(ResolutionMode::Native).to_string();
    s.render_intent = "display".to_string();
    s.output_transform = defaults.output_transform;
    s.step_quality = "offline".to_string();
    s.aod = 0.05;
    s.rh_swelling = false;
    s.atmosphere_correction = true;
    s.terrain_atmosphere = true;
    let land = LandAppearanceConfig::shipped();
    s.land_sza_normalization = land.sza_normalization;
    s.land_sza_max_gain = land.sza_max_gain as f32;
    s.land_dark_toe = land.dark_toe;
    s.land_dark_toe_knee = land.dark_toe_knee as f32;
    s.land_dark_toe_gamma = land.dark_toe_gamma as f32;
    s.land_dark_toe_max_gain = land.dark_toe_max_gain as f32;
    let legacy_postlight = SurfacePostlightToeConfig::off();
    s.surface_postlight_toe = legacy_postlight.enabled;
    s.surface_postlight_toe_knee = legacy_postlight.knee as f32;
    s.surface_postlight_toe_gamma = legacy_postlight.gamma as f32;
    s.surface_postlight_toe_max_gain = legacy_postlight.max_gain as f32;
    let twilight = TwilightSurfaceRecoveryConfig::shipped();
    s.twilight_surface_recovery = twilight.enabled;
    s.twilight_surface_recovery_knee = twilight.knee as f32;
    s.twilight_surface_recovery_gamma = twilight.gamma as f32;
    s.twilight_surface_recovery_max_gain = twilight.max_gain as f32;
    s.clouds_enabled = true;
    s.fractional_clouds = true;
    s.fractional_cloud_mode = "deterministic-2".to_string();
    s.multiscatter = true;
    s.delta_flux_clouds = false;
    s.delta_flux_v2_clouds = false;
    s.delta_flux_v3_clouds = false;
    s.cloud_optical_depth_scale = 0.15;
    s.science_cloud_f16 = false;
    s.nssl_native_cloud_optics = false;
    s.hrrr_thompson_native_cloud_optics = false;
    s.feather_exposed_domain_edges = true;
    s.beer_powder = false;
    s.granulation = false;
    s.topdown_stratiform_regularization = false;
    s.topdown_cloud_footprint = false;
    s.topdown_shadow_antialias = true;
    s.exposure = 1.5;
    s.ground_gain = GROUND_DAY_LIFT as f32;
    s.cloud_softclip = 0.65;
    s.cloud_highlight_max = 1.25;
}

/// One-click Band-13 display recommendation. It changes only the palette, leaving
/// the user's sensor, navigation, sampling, camera, raw BT, and visible settings intact.
fn apply_ir_display_baseline(s: &mut StudioSettings) {
    s.ir_enhancement =
        settings::enhancement_token(simsat::ir_enhance::IrEnhancement::Cimss).to_string();
}

fn apply_sensor_qa(s: &mut StudioSettings, mode: RenderMode) {
    s.instrument_footprint =
        settings::instrument_footprint_token(InstrumentFootprint::Off).to_string();
    s.science_cloud_f16 = false;
    s.view = settings::view_token(StudioView::Geostationary).to_string();
    s.geo_navigation = settings::geo_navigation_token(GeoNavigation::GoesRAbiFixedGrid).to_string();
    s.render_intent = "sensor-fast-gray".to_string();
    s.step_quality = "offline".to_string();
    // Sensor QA is an observation-operator path, never a display terrain correction.
    let legacy_postlight = SurfacePostlightToeConfig::off();
    s.surface_postlight_toe = legacy_postlight.enabled;
    s.surface_postlight_toe_knee = legacy_postlight.knee as f32;
    s.surface_postlight_toe_gamma = legacy_postlight.gamma as f32;
    s.surface_postlight_toe_max_gain = legacy_postlight.max_gain as f32;
    let twilight = TwilightSurfaceRecoveryConfig::off();
    s.twilight_surface_recovery = twilight.enabled;
    s.twilight_surface_recovery_knee = twilight.knee as f32;
    s.twilight_surface_recovery_gamma = twilight.gamma as f32;
    s.twilight_surface_recovery_max_gain = twilight.max_gain as f32;

    match mode {
        RenderMode::Visible => {
            s.resolution = settings::resolution_token(ResolutionMode::Abi1km).to_string();
            s.output_transform = "abi-reflectance".to_string();
            s.aod = 0.05;
            s.rh_swelling = false;
            // Sensor Fast Gray retains modeled airlight; the product-facing de-haze is
            // the display transform that must be neutral here.
            s.atmosphere_correction = false;
            s.terrain_atmosphere = true;
            let land = LandAppearanceConfig::identity();
            s.land_sza_normalization = land.sza_normalization;
            s.land_sza_max_gain = land.sza_max_gain as f32;
            s.land_dark_toe = land.dark_toe;
            s.land_dark_toe_knee = land.dark_toe_knee as f32;
            s.land_dark_toe_gamma = land.dark_toe_gamma as f32;
            s.land_dark_toe_max_gain = land.dark_toe_max_gain as f32;
            s.clouds_enabled = true;
            s.fractional_clouds = true;
            s.fractional_cloud_mode = "effective-od".to_string();
            s.multiscatter = true;
            s.delta_flux_clouds = false;
            s.delta_flux_v2_clouds = false;
            s.delta_flux_v3_clouds = false;
            s.cloud_optical_depth_scale = 1.0;
            s.nssl_native_cloud_optics = false;
            s.hrrr_thompson_native_cloud_optics = false;
            s.feather_exposed_domain_edges = false;
            s.beer_powder = false;
            s.granulation = false;
            s.topdown_stratiform_regularization = false;
            s.topdown_cloud_footprint = false;
            s.topdown_shadow_antialias = false;
            s.exposure = 1.0;
            s.ground_gain = 1.0;
            s.cloud_softclip = 1.0;
            s.cloud_highlight_max = 1.0;
        }
        RenderMode::Ir => {
            s.resolution = settings::resolution_token(ResolutionMode::Abi2km).to_string();
            s.thermal_sensor =
                settings::thermal_sensor_token(ThermalSensor::GoesRAbiBand13Fm4).to_string();
            // The Studio must display RGB, so use NOAA's continuous heritage Band-13
            // grayscale. The rendered Kelvin plane remains the quantitative sensor result.
            s.ir_enhancement = "natural".to_string();
        }
        _ => unreachable!("validate_scope admits only Visible or IR"),
    }
}

fn collect_changes(
    before: &StudioSettings,
    after: &StudioSettings,
    runtime_before: PresetRuntime,
    runtime_after: PresetRuntime,
) -> Vec<PresetChange> {
    let mut changes = Vec::new();
    macro_rules! record {
        ($field:ident, $label:literal) => {
            if before.$field != after.$field {
                changes.push(PresetChange {
                    field: $label,
                    before: before.$field.to_string(),
                    after: after.$field.to_string(),
                });
            }
        };
    }

    record!(visible_calibration_epoch, "Visible calibration epoch");
    record!(geo_navigation, "Navigation");
    record!(resolution, "Resolution");
    record!(view, "View");
    record!(render_intent, "Intent");
    record!(ir_enhancement, "IR enhancement");
    record!(thermal_sensor, "Thermal sensor");
    record!(instrument_footprint, "Instrument footprint");
    record!(output_transform, "Output transform");
    record!(step_quality, "CPU quality");
    record!(aod, "Aerosol optical depth");
    record!(rh_swelling, "RH aerosol swelling");
    record!(atmosphere_correction, "Atmosphere correction");
    record!(terrain_atmosphere, "Terrain atmosphere");
    record!(land_sza_normalization, "Land SZA normalization");
    record!(land_sza_max_gain, "Land SZA max gain");
    record!(land_dark_toe, "Land dark toe");
    record!(land_dark_toe_knee, "Land dark-toe knee");
    record!(land_dark_toe_gamma, "Land dark-toe gamma");
    record!(land_dark_toe_max_gain, "Land dark-toe max gain");
    record!(surface_postlight_toe, "Legacy post-light surface toe");
    record!(surface_postlight_toe_knee, "Legacy post-light toe knee");
    record!(surface_postlight_toe_gamma, "Legacy post-light toe gamma");
    record!(
        surface_postlight_toe_max_gain,
        "Legacy post-light toe max gain"
    );
    record!(twilight_surface_recovery, "Twilight surface recovery");
    record!(twilight_surface_recovery_knee, "Twilight recovery knee");
    record!(twilight_surface_recovery_gamma, "Twilight recovery gamma");
    record!(
        twilight_surface_recovery_max_gain,
        "Twilight recovery max gain"
    );
    record!(clouds_enabled, "Clouds");
    record!(fractional_clouds, "Fractional clouds");
    record!(fractional_cloud_mode, "Fractional-cloud mode");
    record!(multiscatter, "Cloud multiscatter");
    record!(delta_flux_clouds, "Delta-flux v1");
    record!(delta_flux_v2_clouds, "Delta-flux v2b");
    record!(delta_flux_v3_clouds, "Delta-flux v3 memory");
    record!(cloud_optical_depth_scale, "Cloud OD scale");
    record!(science_cloud_f16, "ScienceCloudF16 storage");
    record!(nssl_native_cloud_optics, "NSSL native optics");
    record!(
        hrrr_thompson_native_cloud_optics,
        "HRRR Thompson native optics"
    );
    record!(feather_exposed_domain_edges, "Exposed-domain edge feather");
    record!(beer_powder, "Beer-powder shaping");
    record!(granulation, "Cloud granulation");
    record!(
        topdown_stratiform_regularization,
        "Top-down stratiform reconstruction"
    );
    record!(topdown_cloud_footprint, "Top-down cloud footprint");
    record!(topdown_shadow_antialias, "Top-down shadow anti-aliasing");
    record!(exposure, "Exposure");
    record!(ground_gain, "Ground gain");
    record!(cloud_softclip, "Cloud highlight knee");
    record!(cloud_highlight_max, "Cloud highlight ceiling");

    if runtime_before.gpu_clouds != runtime_after.gpu_clouds {
        changes.push(PresetChange {
            field: "GPU clouds",
            before: runtime_before.gpu_clouds.to_string(),
            after: runtime_after.gpu_clouds.to_string(),
        });
    }
    if runtime_before.parity_pending != runtime_after.parity_pending {
        changes.push(PresetChange {
            field: "Pending GPU parity pass",
            before: runtime_before.parity_pending.to_string(),
            after: runtime_after.parity_pending.to_string(),
        });
    }
    changes
}

#[cfg(test)]
mod tests {
    use super::*;
    use simsat::derived::DerivedField;
    use simsat::wv::WvBand;

    fn runtime_on() -> PresetRuntime {
        PresetRuntime {
            gpu_clouds: true,
            parity_pending: true,
        }
    }

    /// Prove that the user-facing list contains every persisted field changed by
    /// the plan (and neither runtime switch can disappear from that accounting).
    fn assert_diff_is_exhaustive(
        before: &StudioSettings,
        runtime_before: PresetRuntime,
        planned: &PresetPlan,
    ) {
        let before_json = serde_json::to_value(before).unwrap();
        let after_json = serde_json::to_value(&planned.settings).unwrap();
        let before_object = before_json.as_object().unwrap();
        let after_object = after_json.as_object().unwrap();
        let persisted_changes = before_object
            .iter()
            .filter(|(key, value)| after_object.get(*key) != Some(*value))
            .count();
        let runtime_changes = usize::from(runtime_before.gpu_clouds != planned.runtime.gpu_clouds)
            + usize::from(runtime_before.parity_pending != planned.runtime.parity_pending);
        assert_eq!(
            planned.changes.len(),
            persisted_changes + runtime_changes,
            "the displayed diff must account for every changed field: {}",
            planned.change_summary()
        );
    }

    fn scrambled_visible() -> StudioSettings {
        StudioSettings {
            visible_calibration_epoch: 0,
            mode: "visible".to_string(),
            sat: "goes-east".to_string(),
            geo_navigation: "model-sphere".to_string(),
            resolution: "abi-2km".to_string(),
            view: "topdown".to_string(),
            render_intent: "sensor-fast-gray".to_string(),
            instrument_footprint: "goes-r-abi-band13-mtf-prototype".to_string(),
            output_transform: "debug-srgb".to_string(),
            step_quality: "interactive".to_string(),
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
            surface_postlight_toe: true,
            surface_postlight_toe_knee: 0.22,
            surface_postlight_toe_gamma: 0.75,
            surface_postlight_toe_max_gain: 1.45,
            twilight_surface_recovery: false,
            twilight_surface_recovery_knee: 0.20,
            twilight_surface_recovery_gamma: 0.70,
            twilight_surface_recovery_max_gain: 2.0,
            clouds_enabled: false,
            fractional_clouds: false,
            fractional_cloud_mode: "off".to_string(),
            multiscatter: false,
            delta_flux_clouds: true,
            delta_flux_v2_clouds: true,
            delta_flux_v3_clouds: true,
            cloud_optical_depth_scale: 3.0,
            science_cloud_f16: true,
            nssl_native_cloud_optics: true,
            hrrr_thompson_native_cloud_optics: false,
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
            margin_pct: 0.37,
            ..Default::default()
        }
    }

    fn assert_display_settings(s: &StudioSettings, fractional_mode: &str, cloud_softclip: f32) {
        assert_eq!(
            s.visible_calibration_epoch,
            settings::VISIBLE_CALIBRATION_EPOCH
        );
        assert_eq!(s.resolution, "native");
        assert_eq!(s.render_intent, "display");
        assert_eq!(s.instrument_footprint, "off");
        assert_eq!(s.output_transform, "abi-reflectance");
        assert_eq!(s.step_quality, "offline");
        assert_eq!(s.aod, 0.05);
        assert!(!s.rh_swelling);
        assert!(s.atmosphere_correction);
        assert!(s.terrain_atmosphere);
        assert!(s.land_sza_normalization);
        assert_eq!(s.land_sza_max_gain, 4.0);
        assert!(s.land_dark_toe);
        assert_eq!(s.land_dark_toe_knee, 0.08);
        assert_eq!(s.land_dark_toe_gamma, 0.65);
        assert_eq!(s.land_dark_toe_max_gain, 1.5);
        assert!(!s.surface_postlight_toe);
        assert_eq!(s.surface_postlight_toe_knee, 0.18);
        assert_eq!(s.surface_postlight_toe_gamma, 0.80);
        assert_eq!(s.surface_postlight_toe_max_gain, 1.35);
        assert!(s.twilight_surface_recovery);
        assert_eq!(s.twilight_surface_recovery_knee, 0.30);
        assert_eq!(s.twilight_surface_recovery_gamma, 0.50);
        assert_eq!(s.twilight_surface_recovery_max_gain, 4.0);
        assert!(s.clouds_enabled);
        assert!(s.fractional_clouds);
        assert_eq!(s.fractional_cloud_mode, fractional_mode);
        assert!(s.multiscatter);
        assert!(!s.delta_flux_clouds);
        assert!(!s.delta_flux_v2_clouds);
        assert!(!s.delta_flux_v3_clouds);
        assert_eq!(s.cloud_optical_depth_scale, 0.15);
        assert!(!s.science_cloud_f16);
        assert!(!s.nssl_native_cloud_optics);
        assert!(!s.hrrr_thompson_native_cloud_optics);
        assert!(s.feather_exposed_domain_edges);
        assert!(!s.beer_powder);
        assert!(!s.granulation);
        assert!(!s.topdown_stratiform_regularization);
        assert!(!s.topdown_cloud_footprint);
        assert!(s.topdown_shadow_antialias);
        assert_eq!(s.exposure, 1.5);
        assert_eq!(s.ground_gain, GROUND_DAY_LIFT as f32);
        assert_eq!(s.cloud_softclip, cloud_softclip);
        assert_eq!(s.cloud_highlight_max, 1.25);
    }

    #[test]
    fn recommended_display_sets_every_governed_field_and_preserves_context() {
        let before = scrambled_visible();
        let plan = plan(StudioPreset::RecommendedDisplay, &before, runtime_on()).unwrap();
        assert_diff_is_exhaustive(&before, runtime_on(), &plan);
        assert_display_settings(&plan.settings, "deterministic-2", 0.65);
        assert_eq!(plan.settings.mode, before.mode, "product never changes");
        assert_eq!(plan.settings.sat, before.sat, "satellite never changes");
        assert_eq!(plan.settings.view, before.view, "camera never changes");
        assert_eq!(plan.settings.margin_pct, 0.37, "framing remains personal");
        assert_eq!(
            plan.runtime,
            PresetRuntime {
                gpu_clouds: false,
                parity_pending: false
            }
        );
        assert!(plan.changes.iter().any(|c| c.field == "Cloud OD scale"));
        assert!(
            plan.changes
                .iter()
                .any(|c| c.field == "Instrument footprint")
        );
        assert!(
            plan.changes
                .iter()
                .any(|c| c.field == "ScienceCloudF16 storage")
        );
        assert!(plan.changes.iter().any(|c| c.field == "GPU clouds"));
        assert!(
            plan.changes
                .iter()
                .any(|c| c.field == "Pending GPU parity pass")
        );
    }

    #[test]
    fn recommended_ir_selects_cimss_but_loading_preserves_saved_natural() {
        let mut before = StudioSettings {
            mode: "ir-band13".to_string(),
            ir_enhancement: "natural".to_string(),
            sat: "goes-west".to_string(),
            view: "topdown".to_string(),
            resolution: "native".to_string(),
            exposure: 2.75,
            cloud_optical_depth_scale: 0.37,
            ..Default::default()
        };
        before.sanitize();
        assert_eq!(
            before.ir_enhancement, "natural",
            "loading/sanitizing an existing setting must preserve its explicit palette token"
        );

        let runtime = runtime_on();
        let plan = plan(StudioPreset::RecommendedDisplay, &before, runtime).unwrap();
        assert_diff_is_exhaustive(&before, runtime, &plan);
        let mut expected = before.clone();
        expected.ir_enhancement = "cimss".to_string();
        assert_eq!(plan.settings, expected);
        assert_eq!(
            plan.runtime, runtime,
            "thermal recoloring must not alter GPU state"
        );
        assert_eq!(plan.changes.len(), 1);
        assert_eq!(plan.changes[0].field, "IR enhancement");
    }

    #[test]
    fn high_quality_is_display_baseline_plus_selected_cloud_geometry_and_highlight_knee() {
        let before = scrambled_visible();
        let normal = plan(StudioPreset::RecommendedDisplay, &before, runtime_on()).unwrap();
        let high = plan(StudioPreset::HighQualityVisible, &before, runtime_on()).unwrap();
        assert_diff_is_exhaustive(&before, runtime_on(), &high);
        assert_display_settings(&high.settings, "deterministic-4", 0.45);
        let mut expected = normal.settings;
        expected.fractional_cloud_mode = "deterministic-4".to_string();
        expected.cloud_softclip = 0.45;
        assert_eq!(high.settings, expected);
        assert!(!high.settings.nssl_native_cloud_optics);
        assert!(!high.settings.hrrr_thompson_native_cloud_optics);
    }

    #[test]
    fn sensor_visible_sets_exact_geometry_sampling_neutral_transforms_and_stable_physics() {
        let mut before = scrambled_visible();
        before.view = "topdown".to_string();
        let plan = plan(StudioPreset::SensorQa, &before, runtime_on()).unwrap();
        assert_diff_is_exhaustive(&before, runtime_on(), &plan);
        let s = &plan.settings;
        assert_eq!(s.mode, "visible");
        assert_eq!(s.sat, "goes-east");
        assert_eq!(s.view, "geo");
        assert_eq!(s.geo_navigation, "goes-r-abi");
        assert_eq!(s.resolution, "abi-1km");
        assert_eq!(s.render_intent, "sensor-fast-gray");
        assert_eq!(s.instrument_footprint, "off");
        assert_eq!(s.output_transform, "abi-reflectance");
        assert_eq!(s.step_quality, "offline");
        assert_eq!(s.aod, 0.05);
        assert!(!s.rh_swelling);
        assert!(!s.atmosphere_correction);
        assert!(s.terrain_atmosphere);
        assert!(!s.land_sza_normalization);
        assert_eq!(s.land_sza_max_gain, 4.0);
        assert!(!s.land_dark_toe);
        assert_eq!(s.land_dark_toe_knee, 0.08);
        assert_eq!(s.land_dark_toe_gamma, 0.65);
        assert_eq!(s.land_dark_toe_max_gain, 1.5);
        assert!(!s.surface_postlight_toe);
        assert!(!s.twilight_surface_recovery);
        assert!(s.clouds_enabled);
        assert!(s.fractional_clouds);
        assert_eq!(s.fractional_cloud_mode, "effective-od");
        assert!(s.multiscatter);
        assert!(!s.delta_flux_clouds);
        assert!(!s.delta_flux_v2_clouds);
        assert!(!s.delta_flux_v3_clouds);
        assert_eq!(s.cloud_optical_depth_scale, 1.0);
        assert!(!s.science_cloud_f16);
        assert!(!s.nssl_native_cloud_optics);
        assert!(!s.hrrr_thompson_native_cloud_optics);
        assert!(!s.feather_exposed_domain_edges);
        assert!(!s.beer_powder);
        assert!(!s.granulation);
        assert!(!s.topdown_stratiform_regularization);
        assert!(!s.topdown_cloud_footprint);
        assert!(!s.topdown_shadow_antialias);
        assert_eq!(s.exposure, 1.0);
        assert_eq!(s.ground_gain, 1.0);
        assert_eq!(s.cloud_softclip, 1.0);
        assert_eq!(s.cloud_highlight_max, 1.0);
        assert!(!plan.runtime.gpu_clouds);
        assert!(!plan.runtime.parity_pending);
        assert!(plan.changes.iter().any(|c| c.field == "View"));
        assert!(plan.changes.iter().any(|c| c.field == "Navigation"));
    }

    #[test]
    fn sensor_ir_uses_goes19_fm4_two_kilometre_natural_and_neutral_surface_recovery() {
        let before = StudioSettings {
            mode: "ir-band13".to_string(),
            sat: "goes-east".to_string(),
            view: "topdown".to_string(),
            geo_navigation: "model-sphere".to_string(),
            resolution: "native".to_string(),
            render_intent: "display".to_string(),
            ir_enhancement: "rainbow".to_string(),
            thermal_sensor: "fast-gray".to_string(),
            instrument_footprint: "goes-r-abi-band13-mtf-prototype".to_string(),
            step_quality: "interactive".to_string(),
            exposure: 3.25,
            cloud_optical_depth_scale: 0.37,
            science_cloud_f16: true,
            ..Default::default()
        };
        let original_exposure = before.exposure;
        let original_od = before.cloud_optical_depth_scale;
        let plan = plan(StudioPreset::SensorQa, &before, runtime_on()).unwrap();
        assert_diff_is_exhaustive(&before, runtime_on(), &plan);
        let s = &plan.settings;
        assert_eq!(s.mode, "ir-band13", "preset never converts the product");
        assert_eq!(s.view, "geo");
        assert_eq!(s.geo_navigation, "goes-r-abi");
        assert_eq!(s.resolution, "abi-2km");
        assert_eq!(s.render_intent, "sensor-fast-gray");
        assert_eq!(s.thermal_sensor, "goes-r-abi-band13-fm4");
        assert_eq!(s.instrument_footprint, "off");
        assert_eq!(s.ir_enhancement, "natural");
        assert_eq!(s.step_quality, "offline");
        assert!(!s.science_cloud_f16);
        assert!(!s.surface_postlight_toe);
        assert!(!s.twilight_surface_recovery);
        assert_eq!(s.exposure, original_exposure);
        assert_eq!(s.cloud_optical_depth_scale, original_od);
        assert!(!plan.runtime.gpu_clouds);
        assert!(!plan.runtime.parity_pending);
    }

    #[test]
    fn invalid_products_and_sources_are_rejected_without_a_transition() {
        for mode in [
            RenderMode::WaterVapor(WvBand::Upper),
            RenderMode::Derived(DerivedField::CloudOpticalDepth),
            RenderMode::GeoColor,
            RenderMode::Sandwich,
        ] {
            let s = StudioSettings {
                mode: settings::mode_token(mode).to_string(),
                ..Default::default()
            };
            let error = plan(
                StudioPreset::SensorQa,
                &s,
                PresetRuntime {
                    gpu_clouds: false,
                    parity_pending: false,
                },
            )
            .unwrap_err();
            assert!(error.0.contains("never converts"), "{mode:?}: {error}");
        }

        let himawari = StudioSettings {
            mode: "visible".to_string(),
            sat: "himawari".to_string(),
            ..Default::default()
        };
        assert!(
            plan(StudioPreset::SensorQa, &himawari, runtime_on())
                .unwrap_err()
                .0
                .contains("incompatible with Himawari")
        );

        let goes_west_ir = StudioSettings {
            mode: "ir-band13".to_string(),
            sat: "goes-west".to_string(),
            ..Default::default()
        };
        assert!(
            plan(StudioPreset::SensorQa, &goes_west_ir, runtime_on())
                .unwrap_err()
                .0
                .contains("GOES-East")
        );
    }

    #[test]
    fn high_quality_rejects_perspective_instead_of_hiding_a_camera_change() {
        let s = StudioSettings {
            mode: "visible".to_string(),
            view: "perspective".to_string(),
            ..Default::default()
        };
        let error = plan(StudioPreset::HighQualityVisible, &s, runtime_on()).unwrap_err();
        assert!(error.0.contains("not supported in Perspective"));
    }

    #[test]
    fn plan_is_idempotent_and_reports_no_phantom_changes() {
        let before = StudioSettings::default();
        let first = plan(
            StudioPreset::RecommendedDisplay,
            &before,
            PresetRuntime {
                gpu_clouds: false,
                parity_pending: false,
            },
        )
        .unwrap();
        assert!(first.changes.is_empty(), "{}", first.change_summary());
        let second = plan(
            StudioPreset::RecommendedDisplay,
            &first.settings,
            first.runtime,
        )
        .unwrap();
        assert!(second.changes.is_empty());
    }

    #[test]
    fn active_quick_mode_tracks_reviewed_plans_and_manual_custom_edits() {
        let runtime = PresetRuntime {
            gpu_clouds: false,
            parity_pending: false,
        };
        let base = StudioSettings::default();
        assert_eq!(
            active(&base, runtime),
            Some(StudioPreset::RecommendedDisplay)
        );

        let high_quality = plan(StudioPreset::HighQualityVisible, &base, runtime)
            .unwrap()
            .settings;
        assert_eq!(
            active(&high_quality, runtime),
            Some(StudioPreset::HighQualityVisible)
        );
        assert_eq!(high_quality.fractional_cloud_mode, "deterministic-4");
        assert_eq!(high_quality.cloud_softclip, 0.45);

        let sensor = plan(StudioPreset::SensorQa, &base, runtime)
            .unwrap()
            .settings;
        assert_eq!(active(&sensor, runtime), Some(StudioPreset::SensorQa));

        let mut custom = high_quality;
        custom.exposure = 1.6;
        assert_eq!(active(&custom, runtime), None);
        assert_eq!(active(&base, runtime_on()), None);
    }
}
