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
use simsat::atmosphere::OutputTransform;
use simsat::camera::{ResolutionMode, SatellitePreset};
use simsat::clouds::{DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE, StepQuality};
use simsat::derived::DerivedField;
use simsat::ir_enhance::IrEnhancement;
use simsat::wv::WvBand;

use crate::{RenderMode, StudioView};

/// Maximum entries kept in the recent-files list.
pub const RECENT_CAP: usize = 8;
/// Visible-calibration settings epoch. Epoch 1 was the diagnostic v0.1.4 RC
/// (`cloud_optical_depth_scale = 0.25`, exposure/ground = `1.6`); epoch 2 carries
/// the owner-selected `0.15` plus the neutral ABI display/ground baseline. Older
/// settings are preserved and surfaced instead of being silently overwritten.
pub const VISIBLE_CALIBRATION_EPOCH: u32 = 2;

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
    pub resolution: String,
    pub view: String,
    pub mode: String,
    pub ir_enhancement: String,
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
    pub clouds_enabled: bool,
    /// Use model cloud fraction/subcolumns when the brick provides them. `true` is
    /// the physical default; `false` preserves legacy horizontally-full cloudy cells.
    pub fractional_clouds: bool,
    pub multiscatter: bool,
    /// Visible cloud optical-depth calibration. The shipped default is `0.15`;
    /// `1.0` keeps the model-derived physical input unchanged.
    pub cloud_optical_depth_scale: f32,
    pub beer_powder: bool,
    /// Display-only sub-grid cloud-edge erosion. Explicitly opt-in/off by default.
    pub granulation: bool,
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
        Self {
            visible_calibration_epoch: VISIBLE_CALIBRATION_EPOCH,
            store_root: None,
            sat: sat_token(SatellitePreset::GoesEast).to_string(),
            resolution: resolution_token(ResolutionMode::Native).to_string(),
            view: view_token(StudioView::Geostationary).to_string(),
            mode: mode_token(RenderMode::Visible).to_string(),
            ir_enhancement: enhancement_token(IrEnhancement::default()).to_string(),
            output_transform: output_transform_token(OutputTransform::AbiReflectance).to_string(),
            step_quality: step_quality_token(StepQuality::Offline).to_string(),
            margin_pct: 0.0,
            aod: simsat::atmosphere::DEFAULT_AOD as f32,
            rh_swelling: false,
            atmosphere_correction: true,
            terrain_atmosphere: true,
            clouds_enabled: true,
            fractional_clouds: true,
            multiscatter: true,
            cloud_optical_depth_scale: DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE,
            beer_powder: false,
            granulation: false,
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
        if sat_from_token(&self.sat).is_none() {
            self.sat = d.sat;
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
        if enhancement_from_token(&self.ir_enhancement).is_none() {
            self.ir_enhancement = d.ir_enhancement;
        }
        if output_transform_from_token(&self.output_transform).is_none() {
            self.output_transform = d.output_transform;
        }
        if step_quality_from_token(&self.step_quality).is_none() {
            self.step_quality = d.step_quality;
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
    }
}

pub fn sat_from_token(t: &str) -> Option<SatellitePreset> {
    SatellitePreset::ALL
        .into_iter()
        .find(|p| sat_token(*p) == t)
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

pub fn enhancement_token(e: IrEnhancement) -> &'static str {
    match e {
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
            let mut settings = serde_json::from_value::<StudioSettings>(value).ok()?;
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
        assert_eq!(s.output_transform, "abi-reflectance");
        assert!(s.clouds_enabled);
        assert!(s.multiscatter);
        assert!(!s.beer_powder);
        assert_eq!(s.cloud_optical_depth_scale, 0.15);
        assert_eq!(s.exposure, 1.0);
        assert_eq!(s.ground_gain, 1.0);
        assert_eq!(s.cloud_softclip, 0.65);
        assert_eq!(s.cloud_highlight_max, 1.25);
        assert!(s.fractional_clouds);
        assert!(!s.granulation);
    }

    #[test]
    fn tokens_round_trip_every_variant() {
        for p in SatellitePreset::ALL {
            assert_eq!(sat_from_token(sat_token(p)), Some(p));
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
        for e in IrEnhancement::ALL {
            assert_eq!(enhancement_from_token(enhancement_token(e)), Some(e));
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
        // Unknown tokens map to None (they reset to defaults in sanitize).
        assert_eq!(mode_from_token("does-not-exist"), None);
        assert_eq!(sat_from_token(""), None);
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
        assert_eq!(
            s.visible_calibration_epoch, 0,
            "pre-epoch files must preserve values and request an explicit migration choice"
        );
        // Backward compatibility: settings written before these controls existed
        // receive the new shipped defaults field-by-field via `#[serde(default)]`.
        assert!(s.atmosphere_correction);
        assert!(s.terrain_atmosphere);
        assert!(s.fractional_clouds);
        assert_eq!(
            s.cloud_optical_depth_scale,
            DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE
        );
        assert!(!s.beer_powder);
        assert!(!s.granulation);
        assert_eq!(s.ground_gain, StudioSettings::default().ground_gain);
        assert_eq!(s.cloud_softclip, StudioSettings::default().cloud_softclip);
        assert_eq!(
            s.cloud_highlight_max,
            StudioSettings::default().cloud_highlight_max
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
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sanitize_clamps_and_resets_unknown_tokens() {
        let mut s = StudioSettings {
            margin_pct: 250.0,
            aod: -3.0,
            cloud_optical_depth_scale: 99.0,
            fractional_clouds: false,
            exposure: 99.0,
            ground_gain: 99.0,
            cloud_softclip: -3.0,
            cloud_highlight_max: f32::NAN,
            play_fps: 0.0,
            frame_cap: 1,
            bm_month_override: 13,
            sat: "geostationary-9".to_string(),
            mode: "x-ray".to_string(),
            orbit_tilt_deg: 90.0,
            orbit_range_km: f32::NAN,
            persp_width: 100_000,
            persp_height: 0,
            ..Default::default()
        };
        s.sanitize();
        assert_eq!(s.margin_pct, 100.0);
        assert_eq!(s.aod, 0.0);
        assert_eq!(s.cloud_optical_depth_scale, 4.0);
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
            exposure: 2.5,
            atmosphere_correction: false,
            terrain_atmosphere: false,
            fractional_clouds: false,
            cloud_optical_depth_scale: 0.5,
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
