//! SimSat Studio (M1).
//!
//! Standalone eframe/wgpu desktop app. M1 flow (design doc section 10, M1 row):
//! open a wrfout file OR an existing cached run (`run.json`); ingest the selected
//! timestep on a below-normal-priority worker with the M0 size-gate confirm for
//! large files; pick a satellite (GOES-East / GOES-West / Himawari) and timestep;
//! Render the geostationary surface frame (Blue Marble ground + HGT normals +
//! LANDMASK + point sun) and display it in-window; Write it to the sat-store so
//! BowEcho can play it (point BowEcho's sat store dir at the shown store root).
//!
//! Since M1 the studio has grown the loop timeline (M7), the product modes
//! (IR/WV/GeoColor/Sandwich/derived), and the WS4 UX tier: settings persistence +
//! recent files (`settings.rs`), the scene cache in the prepare worker, the sticky
//! error banner + log view, "Save PNG..." export, drag-and-drop + the first-run
//! CTA, and the below-normal global rayon pool.
//!
//! Threading: the heavy CPU prep (ingest if needed, brick decode, Blue Marble
//! decode/crop, LUT build) runs on a below-normal-priority worker; the GPU render
//! and readback run synchronously on the UI thread (the frame is small and this
//! keeps all wgpu work on eframe's own device/queue thread). Documented in
//! notes/m1-notes.md.
//!
//! (No `windows_subsystem = "windows"`: M1 keeps the console so the owner sees
//! ingest progress and any startup panic during acceptance.)

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui;
use eframe::egui_wgpu::wgpu;

mod pipeline;
mod presets;
mod settings;

use simsat::api::{FractionalCloudMode, RenderIntent, RenderIntentAdjustment};
use simsat::asset_pack;
use simsat::atmosphere::{
    self, AtmosphereLuts, AtmosphereParams, CameraGeometry, OutputTransform, SOLAR_IRRADIANCE_RGB,
    SkyShTable,
};
use simsat::bluemarble;
use simsat::bricks::{self, RunManifest, StorageProfile};
use simsat::camera::{
    GeoCamera, GeoNavigation, MAX_AXIS, PerspectiveBasis, PerspectiveCamera, ResolutionMode,
    SatellitePreset, SurfaceRaster, build_map_raster_mode, build_perspective_raster,
    build_surface_raster_mode, extended_native_counts,
};
use simsat::clouds::{self, CloudMultiscatterMode, MarchConfig, StepQuality};
use simsat::derived::{self, DerivedField};
use simsat::frame::{GridGeoref, WrfProjectionParams};
use simsat::geocolor;
use simsat::gpu::{
    self, CloudFrameInputs, CloudMarchParams, CloudPassResources, CloudViewMode, RenderedFrame,
    SurfaceFrameInputs, SurfaceResources, SurfaceUniforms,
};
use simsat::horizon::HorizonMap;
use simsat::ingest::{self, IngestConfig};
use simsat::ingest_grib;
use simsat::instrument_footprint::{
    InstrumentFootprint, apply_band13_radiance_footprint_validated,
};
use simsat::ir::{self, IrConfig, IrScene, IrVolume};
use simsat::ir_enhance::{IrEnhancement, render_ir_rgba};
use simsat::optics::CloudOpticsMode;
use simsat::render::{
    CLOUD_SOFTCLIP_KNEE, FLAT_ALBEDO_SRGB, FrameContext, GROUND_DAY_LIFT, LandAppearanceConfig,
    RHO_HIGHLIGHT_MAX, SurfacePixel, SurfacePostlightToeConfig, TwilightSurfaceRecoveryConfig,
    WATER_ALBEDO_SCALE, apply_low_sun_illuminant, blend_snow, normals_from_hgt,
    radiance_to_rgba_softclip, shade_surface, snow_fraction,
};
use simsat::sandwich;
use simsat::solar::{self, SolarFrame};
use simsat::store_out::{self, IrFrame, VisibleFrame, WrittenVisibleFrame};
use simsat::thermal_sensor::ThermalSensor;
use simsat::topdown;
use simsat::wv::WvBand;

/// Render mode: physically-based VISIBLE (Blue Marble + clouds + sun), synthetic IR
/// (ABI band 13, 10.3 um — thermal window), or a WATER-VAPOR band (ABI 8/9/10 =
/// 6.2/6.9/7.3 um — thermal, upper/mid/lower moisture). IR + WV are thermal (day AND
/// night; no sun input).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RenderMode {
    Visible,
    /// SimSat Day/Night Color: a GeoColor-style blend with broad-RGB visible by day and
    /// colored band-13 IR by night. It is not yet sensor-derived ABI GeoColor.
    GeoColor,
    /// Sandwich composite (the classic severe-convection view): the visible true-color RGB as
    /// the base, with a color-enhanced band-13 IR overlaid on the COLD (high) cloud tops (alpha
    /// ramps with coldness). NOT thermal — the sun + exposure light the visible base AND band-13
    /// IR is marched for the cold-top overlay. A daytime-convection product.
    Sandwich,
    Ir,
    WaterVapor(WvBand),
    /// A DERIVED scalar-field map (precipitable water / cloud-top temp / cloud optical depth):
    /// a per-column brick integral, rendered with a basic colormap. Column products — they
    /// ignore the sun / exposure / atmosphere / cloud controls (like the thermal modes).
    Derived(DerivedField),
}

/// Maximum viewport zoom, as a factor over the fit-to-window scale (display-side only).
const MAX_VIEW_ZOOM: f32 = 16.0;

/// Keep expanded configuration/detail content bounded so the rendered image never
/// gets squeezed out of an ordinary laptop-sized window. The Settings and Quick
/// mode Details sections are collapsed by default; this is their expanded height.
fn settings_scroll_max_height(window_height: f32) -> f32 {
    (window_height * 0.28).clamp(160.0, 250.0)
}

/// Fixed selector widths for the second toolbar row. The compact policy fits the
/// 900 px RC smoke-test window without scrolling; the regular policy uses the
/// extra room at 1360 px for the full selected labels.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ToolbarLayout {
    mode_width: f32,
    intent_width: f32,
    sat_width: f32,
    navigation_width: f32,
    view_width: f32,
    timestep_width: f32,
}

impl ToolbarLayout {
    /// Approximate fixed row budget: six captions plus eleven 4 px item gaps.
    #[cfg(test)]
    fn estimated_selector_width(self) -> f32 {
        self.mode_width
            + self.intent_width
            + self.sat_width
            + self.navigation_width
            + self.view_width
            + self.timestep_width
            + 220.0
    }
}

fn toolbar_layout(available_width: f32) -> ToolbarLayout {
    if available_width < 1_200.0 {
        ToolbarLayout {
            mode_width: 112.0,
            intent_width: 76.0,
            sat_width: 88.0,
            navigation_width: 100.0,
            view_width: 88.0,
            timestep_width: 132.0,
        }
    } else {
        ToolbarLayout {
            mode_width: 190.0,
            intent_width: 100.0,
            sat_width: 135.0,
            navigation_width: 190.0,
            view_width: 150.0,
            timestep_width: 180.0,
        }
    }
}

impl RenderMode {
    const ALL: [RenderMode; 10] = [
        RenderMode::Visible,
        RenderMode::GeoColor,
        RenderMode::Sandwich,
        RenderMode::Ir,
        RenderMode::WaterVapor(WvBand::Upper),
        RenderMode::WaterVapor(WvBand::Mid),
        RenderMode::WaterVapor(WvBand::Low),
        RenderMode::Derived(DerivedField::PrecipitableWater),
        RenderMode::Derived(DerivedField::CloudTopTemp),
        RenderMode::Derived(DerivedField::CloudOpticalDepth),
    ];
    fn label(self) -> &'static str {
        match self {
            RenderMode::Visible => "Visible",
            RenderMode::GeoColor => "GeoColor Style (SimSat day/night)",
            RenderMode::Sandwich => "Sandwich (vis + cold-top IR)",
            RenderMode::Ir => "IR (band 13)",
            RenderMode::WaterVapor(WvBand::Upper) => "WV 6.2 um (upper)",
            RenderMode::WaterVapor(WvBand::Mid) => "WV 6.9 um (mid)",
            RenderMode::WaterVapor(WvBand::Low) => "WV 7.3 um (lower)",
            RenderMode::Derived(DerivedField::PrecipitableWater) => "Derived: Precipitable Water",
            RenderMode::Derived(DerivedField::CloudTopTemp) => "Derived: Cloud-Top Temp",
            RenderMode::Derived(DerivedField::CloudOpticalDepth) => "Derived: Cloud Optical Depth",
        }
    }

    /// The derived scalar field for this mode, or `None` if it is not a derived-field mode.
    fn derived_field(self) -> Option<DerivedField> {
        match self {
            RenderMode::Derived(f) => Some(f),
            _ => None,
        }
    }

    /// Whether this is a thermal (brightness-temperature) mode — IR or WV. Thermal modes
    /// ignore the sun / exposure / atmosphere / cloud controls (no sun input). GeoColor is
    /// NOT thermal (it uses the sun for its day half).
    fn is_thermal(self) -> bool {
        matches!(self, RenderMode::Ir | RenderMode::WaterVapor(_))
    }

    /// Whether this is the GeoColor day/night blend (visible day + colored-IR night).
    fn is_geocolor(self) -> bool {
        matches!(self, RenderMode::GeoColor)
    }

    /// Whether this is the Sandwich composite (visible base + colored-IR cold-top overlay).
    fn is_sandwich(self) -> bool {
        matches!(self, RenderMode::Sandwich)
    }

    /// Whether this mode composites the visible render over band-13 IR (GeoColor or Sandwich):
    /// it lights the visible with the sun + exposure AND marches band-13 IR to blend in.
    fn is_visible_ir_composite(self) -> bool {
        matches!(self, RenderMode::GeoColor | RenderMode::Sandwich)
    }

    /// Whether this mode uses the visible-path controls — the Sun & Exposure, Atmosphere,
    /// Clouds, and Ground / Blue Marble drawer groups. Those apply to the physically-based
    /// visible render (Visible) and to the two visible-over-IR composites (GeoColor and
    /// Sandwich, whose day / base half IS the visible render). The thermal (IR / WV) and
    /// derived-scalar modes ignore all of them (no sun, no ground albedo), so those groups
    /// are simply hidden in those modes — the context-driven adaptive drawer.
    fn uses_visible_controls(self) -> bool {
        matches!(
            self,
            RenderMode::Visible | RenderMode::GeoColor | RenderMode::Sandwich
        )
    }

    /// The thermal march config for this mode (band 13 window, or the WV band), or `None`
    /// for the visible / GeoColor / Sandwich modes (those march band 13 internally in the
    /// blend, not as the primary thermal product).
    fn ir_config(self) -> Option<IrConfig> {
        match self {
            RenderMode::Visible
            | RenderMode::GeoColor
            | RenderMode::Sandwich
            | RenderMode::Derived(_) => None,
            RenderMode::Ir => Some(IrConfig::band13()),
            RenderMode::WaterVapor(band) => Some(band.ir_config()),
        }
    }

    /// The ABI band number the thermal frame writes / enhances through (13, or 8/9/10 for
    /// WV); 13 as a harmless placeholder in the visible mode.
    fn ir_band(self) -> u8 {
        match self {
            RenderMode::WaterVapor(band) => band.abi_band(),
            _ => 13,
        }
    }

    /// A short "band 13" / "WV 6.2 um" description for status lines.
    fn thermal_label(self) -> String {
        match self {
            RenderMode::WaterVapor(band) => format!("WV {} um", band.micron()),
            _ => "band 13".to_string(),
        }
    }
}

/// The STUDIO view picker: the engine's two `camera::ViewMode` products
/// (geostationary from-space, top-down map) plus the studio-only Perspective (3-D)
/// orbit view over the engine's free `PerspectiveCamera` (the map-layers tier-2
/// engine work, 9fd0d5e). Perspective renders on the CPU (like top-down), is
/// VISIBLE-mode only in v1 (no perspective IR march), and has no sat-store
/// contract (a picture, not a map — Save PNG is the export).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StudioView {
    Geostationary,
    TopDownMap,
    Perspective,
}

impl StudioView {
    const ALL: [StudioView; 3] = [
        StudioView::Geostationary,
        StudioView::TopDownMap,
        StudioView::Perspective,
    ];

    fn label(self) -> &'static str {
        match self {
            StudioView::Geostationary => "Geostationary (from space)",
            StudioView::TopDownMap => "Top-down map",
            StudioView::Perspective => "Perspective (3-D)",
        }
    }
}

fn main() -> eframe::Result<()> {
    // GLOBAL rayon pool, installed FIRST THING (machine-stability discipline, the
    // hard-rule-4 spirit): all cores MINUS ONE spare so the UI/desktop always has a
    // core, and every pool thread starts at BELOW-NORMAL priority (the same
    // `THREAD_PRIORITY_BELOW_NORMAL` the ingest/render workers use — the owner's
    // machine has hard-crashed under all-core load). `build_global` must precede
    // any rayon use: nothing above this line touches rayon, and the engine only
    // enters rayon inside render/ingest calls, which all happen on workers spawned
    // long after this. If the pool was somehow already initialized, keep running
    // with the default pool rather than dying.
    let threads = pipeline::pool_threads_leaving_spare(
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    );
    if let Err(e) = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .start_handler(|_| simsat::platform::lower_worker_thread_priority())
        .build_global()
    {
        eprintln!("simsat_studio: rayon global pool init failed ({e}); using rayon defaults");
    }

    let native_options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_title("SimSat Studio")
            .with_inner_size([1360.0, 860.0]),
        ..Default::default()
    };
    eframe::run_native(
        "SimSat Studio",
        native_options,
        Box::new(|cc| Ok(Box::new(SimSatStudioApp::new(cc)))),
    )
}

/// Captured GPU handles (all `Arc` internally, cheap to clone).
struct GpuCtx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    resources: SurfaceResources,
    /// The ACTIVATED GPU cloud pass (sun-OD compute + clouds.wgsl march) behind the
    /// experimental "GPU clouds" toggle. Pipelines are created once here; the CPU
    /// composite remains the shipping default and the stored-frame path.
    cloud_resources: CloudPassResources,
}

/// A selectable timestep (a wrfout time index, a cached brick hhmm, or one entry of
/// an opened multi-file sequence).
#[derive(Debug, Clone)]
struct Timestep {
    label: String,
    hhmm: u16,
    /// wrfout time index (`None` for a cached run).
    ts_index: Option<usize>,
    time_iso: Option<String>,
    /// The brick file name (`t{YYYYMMDD_HHMM}.ssb`) — from the manifest for a cached
    /// run, else derived from the time so cache paths are always reproducible.
    file: String,
    /// Index into a `Source::Sequence`'s entries (`None` for the single-file sources).
    seq_index: Option<usize>,
}

/// What is currently open.
enum Source {
    Wrfout {
        path: PathBuf,
        cache_dir: PathBuf,
        run_id: String,
        nx: usize,
        ny: usize,
        nz: usize,
        file_bytes: u64,
        needs_confirm: bool,
        confirmed: bool,
    },
    Cached {
        cache_dir: PathBuf,
        run_id: String,
        manifest: RunManifest,
    },
    /// A time SEQUENCE opened as a directory (or multi-select) of wrfout files, sorted
    /// by valid time. Every entry shares ONE `run_id` so all rendered frames land in a
    /// single store run (a proper multi-frame loop). Each entry names its own wrfout
    /// file + time index (the ingest `timestep` parameter).
    Sequence {
        entries: Vec<SeqEntry>,
        cache_dir: PathBuf,
        run_id: String,
        /// Whether any file is large enough to want the M0 size-gate confirm.
        needs_confirm: bool,
        confirmed: bool,
        total_bytes: u64,
    },
}

/// One timestep of an opened sequence: the wrfout file + time index + its valid time.
#[derive(Debug, Clone)]
struct SeqEntry {
    path: PathBuf,
    ts_index: usize,
    label: String,
    hhmm: u16,
    time_iso: Option<String>,
}

/// Why the render used (or did not use) the Blue Marble ground texture, so the
/// status chip tells the truth instead of always blaming a missing file (a decode
/// failure on a present, valid asset previously read as "texture missing").
#[derive(Clone)]
enum BmStatus {
    Loaded,
    Missing,
    Failed(String),
}

impl BmStatus {
    fn chip_label(&self) -> String {
        match self {
            BmStatus::Loaded => "Blue Marble".to_string(),
            BmStatus::Missing => "flat albedo (texture missing)".to_string(),
            BmStatus::Failed(reason) => format!("flat albedo (texture failed to decode: {reason})"),
        }
    }
}

/// The season-blended Blue Marble ground + its status line, as cached/shared by
/// the scene cache (one decode serves every frame of a sequence).
type BmGround = Arc<(bluemarble::BlueMarbleCrop, String)>;

/// Timestep-INDEPENDENT scene resources cached across the frames of a sequence
/// render (and across repeated single renders): the output raster + geo LUT, the
/// season-blended Blue Marble crop, the atmosphere LUT set, and the horizon map.
/// Each is a SINGLE-SLOT cache (`pipeline::CacheSlot`): one live entry per kind,
/// hit only on exact key equality — bounded memory, and a stale artifact can never
/// silently change a frame because every key carries the full parameter set the
/// resource depends on (see the key docs in `pipeline.rs`). Shared by the single
/// Render and the batch loop through an `Arc<Mutex<..>>` (workers are serialized
/// by the busy flag, so the lock is uncontended).
#[derive(Default)]
struct SceneCache {
    raster: pipeline::CacheSlot<pipeline::RasterCacheKey<GridGeoref>, SurfaceRaster>,
    geo_lut: pipeline::CacheSlot<pipeline::GeoLutKey<GridGeoref>, Vec<f32>>,
    bluemarble: pipeline::CacheSlot<pipeline::BmCacheKey, (bluemarble::BlueMarbleCrop, String)>,
    atmo: pipeline::CacheSlot<pipeline::AtmoLutKey, (AtmosphereLuts, SkyShTable)>,
    horizon: pipeline::CacheSlot<pipeline::HorizonCacheKey, HorizonMap>,
}

/// Which scene-cache slots hit for one prepared frame (for the per-frame log line
/// that lets the owner see the loop speedup). `None` = the resource was not needed
/// (e.g. no ground/horizon in a thermal mode).
#[derive(Clone, Copy, Default)]
struct CacheHits {
    raster: Option<bool>,
    geo_lut: Option<bool>,
    bluemarble: Option<bool>,
    atmo: Option<bool>,
    horizon: Option<bool>,
}

impl CacheHits {
    fn summary(&self) -> String {
        let s = |v: Option<bool>| match v {
            Some(true) => "hit",
            Some(false) => "miss",
            None => "n/a",
        };
        format!(
            "raster {}, geo-lut {}, ground {}, atmo {}, horizon {}",
            s(self.raster),
            s(self.geo_lut),
            s(self.bluemarble),
            s(self.atmo),
            s(self.horizon)
        )
    }
}

/// CPU-prepared inputs handed from the worker to the UI thread for GPU render.
struct PreparedRender {
    width: u32,
    height: u32,
    nx: u32,
    ny: u32,
    /// Shared with the scene cache (rebuilt only when the raster/ground changes).
    lut_geo: Arc<Vec<f32>>,
    lut_light: Vec<f32>,
    normals_rgba: Vec<u8>,
    landmask_r8: Vec<u8>,
    bluemarble: Option<BmGround>,
    bm_status: BmStatus,
    /// Seasonal ground status, e.g. `"Blue Marble: Dec/Jan blend (65% Jan)"` (M7).
    /// Empty in IR mode / flat-albedo.
    season_line: String,
    lat: Vec<f32>,
    lon: Vec<f32>,
    sector: String,
    satellite: SatellitePreset,
    /// View mode this frame was rendered in (for the header label).
    view_mode: StudioView,
    year: i32,
    month: u32,
    day: u32,
    hhmm: u16,
    on_earth_frac: f32,
    center_sun_elev: f64,
    /// Whether the fake-sun what-if override lit this frame (for the honesty label).
    sun_override: bool,
    /// Output-raster resolution mode + whether Native was clamped by MAX_AXIS.
    resolution: ResolutionMode,
    res_clamped: bool,
    // M2 atmosphere frame data.
    transmittance_lut: Vec<f32>,
    multiscatter_lut: Vec<f32>,
    ambient_lut: Vec<f32>,
    ambient_n: u32,
    uniforms: SurfaceUniforms,
    pw_ratio: f64,
    /// The M4 CPU-composited cloud frame OR the M6 coloured IR frame (row-major
    /// Rgba8, row 0 = north). `Some` when clouds are enabled OR in IR mode —
    /// `finish_prepared` displays/stores it directly (no GPU pass); `None` falls
    /// back to the M2 GPU surface pass.
    cloud_rgba: Option<Vec<u8>>,
    /// The M6 IR / WV brightness-temperature plane (Kelvin, `NaN` off-earth), `Some` in a
    /// thermal mode. Kept so the studio can re-enhance live (recolour without re-marching)
    /// and write the single-band Kelvin store frame.
    ir_bt: Option<Vec<f32>>,
    /// The IR enhancement the displayed frame was coloured with (thermal mode only).
    ir_enhancement: IrEnhancement,
    /// The ABI band of the thermal frame (13 window, or 8/9/10 for WV) — keys the
    /// enhancement palette + the single-band Kelvin store write.
    ir_band: u8,
    /// Instrument spatial response actually applied to the thermal component.
    instrument_footprint: InstrumentFootprint,
    /// The DERIVED scalar field + its RAW resampled values (`Some` only in a derived-field
    /// mode). Kept for the status line's value-range readout; the displayed frame is the basic
    /// colormap already baked into `cloud_rgba`.
    derived: Option<(DerivedField, Vec<f32>)>,
    /// The product mode this frame was rendered in (for the PNG-export file name).
    render_mode: RenderMode,
    /// The cloud toggle captured when this worker render started. A CPU top-down
    /// surface-only frame still occupies `cloud_rgba`, so buffer presence is not
    /// valid provenance for whether clouds were enabled.
    clouds_enabled: bool,
    /// The EXPERIMENTAL GPU cloud-pass inputs: `Some` when this frame is to be
    /// rendered by the GPU cloud pass on the UI thread (`cloud_rgba` is then `None`).
    /// Carries an optional CPU reference frame for the parity instrument.
    gpu_cloud: Option<Box<GpuCloudPrep>>,
    /// True only for the dedicated one-click action (the persistent manual toggle is false).
    one_click_gpu_render: bool,
    /// Exact temporary control differences selected by that action.
    gpu_preview_adjustments: GpuPreviewAdjustments,
}

/// Worker-prepared inputs of one GPU cloud render (the owned twin of
/// `gpu::CloudFrameInputs`' cloud half; the surface half comes from the existing
/// `PreparedRender` fields). Built in `prepare_render`, consumed on the UI thread.
struct GpuCloudPrep {
    view_mode: CloudViewMode,
    ray_lut: Vec<f32>,
    texture_a: Vec<u8>,
    occupancy: Vec<u8>,
    vol_nx: u32,
    vol_ny: u32,
    vol_nz: u32,
    occ_dims: (u32, u32, u32),
    ql: [f32; 4],
    qp: [f32; 4],
    z_min_m: f32,
    dz_m: f32,
    r_top_m: f32,
    r_bottom_m: f32,
    voxel_pitch_m: f32,
    geo: gpu::GeoQuads,
    march: CloudMarchParams,
    sun_od: gpu::SunOdPlan,
    froxel_dim: u32,
    froxel_data: Vec<f32>,
    sh_rows: u32,
    sh_data: Vec<f32>,
    scan_rect: [f32; 4],
    /// The CPU reference frame for the parity instrument (`Some` on a parity render):
    /// the SAME scene through the CPU composite at GPU-COMPARABLE settings
    /// (Interactive schedule, granulation off, flat/open M3 surface fields, no snow
    /// blend — the documented GPU surface model), so the delta isolates the GPU march.
    cpu_reference: Option<Vec<u8>>,
}

/// The last GPU parity report: the logged summary + the delta heatmap texture.
struct ParityReport {
    summary: String,
    texture: egui::TextureHandle,
}

/// Info about a frame rendered through the GPU cloud pass (for the status line +
/// the parity report hand-off from `render_prepared` to `finish_prepared`).
struct GpuRenderInfo {
    gpu_ms: u64,
    parity: Option<ParityReport>,
}

/// Temporary changes made by the one-click `GPU Render` action. These flags travel
/// with the prepared/rendered frame so the preview can disclose exactly how it differs
/// from the persistent Studio controls. The controls themselves are never mutated.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct GpuPreviewAdjustments(u16);

impl GpuPreviewAdjustments {
    const MODE_VISIBLE: u16 = 1 << 0;
    const VIEW_GEOSTATIONARY: u16 = 1 << 1;
    const CLOUDS_ON: u16 = 1 << 2;
    const TERRAIN_ATMOSPHERE_OFF: u16 = 1 << 3;
    const FRACTIONAL_CLOUDS_OFF: u16 = 1 << 4;
    const GRANULATION_OFF: u16 = 1 << 5;
    const EXPOSED_EDGE_FEATHER_OFF: u16 = 1 << 6;
    const SHIPPED_HIGHLIGHTS: u16 = 1 << 7;
    const INTERACTIVE_STEPS: u16 = 1 << 8;
    const LEGACY_CLOUD_TRANSPORT: u16 = 1 << 9;
    const TOPDOWN_STRATIFORM_REGULARIZATION_OFF: u16 = 1 << 10;
    const TOPDOWN_CLOUD_FOOTPRINT_OFF: u16 = 1 << 11;
    const INSTRUMENT_FOOTPRINT_OFF: u16 = 1 << 12;
    const MODEL_SPHERE_NAVIGATION: u16 = 1 << 13;

    fn insert(&mut self, flag: u16) {
        self.0 |= flag;
    }

    fn is_empty(self) -> bool {
        self.0 == 0
    }

    fn labels(self) -> Vec<&'static str> {
        let mut labels = Vec::new();
        for (flag, label) in [
            (Self::MODE_VISIBLE, "Mode -> Visible"),
            (Self::VIEW_GEOSTATIONARY, "View -> Geostationary"),
            (Self::CLOUDS_ON, "Clouds -> On"),
            (
                Self::TERRAIN_ATMOSPHERE_OFF,
                "Terrain-height atmosphere -> Off",
            ),
            (
                Self::FRACTIONAL_CLOUDS_OFF,
                "Model cloud fraction -> Off (legacy full-cell coverage)",
            ),
            (Self::GRANULATION_OFF, "Granulation -> Off"),
            (
                Self::TOPDOWN_STRATIFORM_REGULARIZATION_OFF,
                "Top-down stratiform reconstruction -> Off",
            ),
            (
                Self::TOPDOWN_CLOUD_FOOTPRINT_OFF,
                "Top-down cloud footprint -> Off",
            ),
            (
                Self::INSTRUMENT_FOOTPRINT_OFF,
                "ABI Band 13 instrument footprint -> Off",
            ),
            (
                Self::MODEL_SPHERE_NAVIGATION,
                "Navigation -> model sphere (GPU-compatible)",
            ),
            (
                Self::EXPOSED_EDGE_FEATHER_OFF,
                "Exposed-edge feather -> Off",
            ),
            (
                Self::SHIPPED_HIGHLIGHTS,
                "Highlight knee/ceiling -> shipped values",
            ),
            (Self::INTERACTIVE_STEPS, "Cloud steps -> Interactive (192)"),
            (
                Self::LEGACY_CLOUD_TRANSPORT,
                "Cloud transport -> Legacy octaves (GPU-compatible)",
            ),
        ] {
            if self.0 & flag != 0 {
                labels.push(label);
            }
        }
        labels
    }

    fn summary(self) -> String {
        self.labels().join("; ")
    }
}

/// Per-channel |delta| statistics between two same-size RGBA frames, in 8-bit counts,
/// over pixels that are ON-EARTH in either frame (alpha > 0 — space is excluded, a
/// masked-vs-masked disagreement still counts). Pure math, node-tested.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ParityStats {
    mean: [f64; 3],
    p95: [u8; 3],
    max: [u8; 3],
    compared: usize,
}

impl ParityStats {
    fn summary(&self) -> String {
        format!(
            "mean |d| R {:.2} G {:.2} B {:.2} / p95 {} {} {} / max {} {} {} (8-bit units, \
             {} px)",
            self.mean[0],
            self.mean[1],
            self.mean[2],
            self.p95[0],
            self.p95[1],
            self.p95[2],
            self.max[0],
            self.max[1],
            self.max[2],
            self.compared
        )
    }
}

/// Compute [`ParityStats`] between the CPU reference and the GPU frame (both
/// row-major RGBA8 of the same dimensions).
fn parity_stats(cpu: &[u8], gpu: &[u8]) -> ParityStats {
    let n = (cpu.len() / 4).min(gpu.len() / 4);
    let mut deltas: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut sums = [0u64; 3];
    let mut compared = 0usize;
    for i in 0..n {
        let c = &cpu[i * 4..i * 4 + 4];
        let g = &gpu[i * 4..i * 4 + 4];
        if c[3] == 0 && g[3] == 0 {
            continue; // space in both
        }
        compared += 1;
        for ch in 0..3 {
            let d = c[ch].abs_diff(g[ch]);
            sums[ch] += d as u64;
            deltas[ch].push(d);
        }
    }
    let mut mean = [0.0f64; 3];
    let mut p95 = [0u8; 3];
    let mut max = [0u8; 3];
    for ch in 0..3 {
        if compared > 0 {
            mean[ch] = sums[ch] as f64 / compared as f64;
            deltas[ch].sort_unstable();
            // Nearest-rank p95: 95% of compared pixels have |delta| <= this value.
            let idx = ((compared as f64 * 0.95).ceil() as usize).clamp(1, compared) - 1;
            p95[ch] = deltas[ch][idx];
            max[ch] = *deltas[ch].last().unwrap_or(&0);
        }
    }
    ParityStats {
        mean,
        p95,
        max,
        compared,
    }
}

/// A delta HEATMAP between the CPU reference and the GPU frame: per pixel the
/// max-channel |delta| through a gain ramp (black = identical, orange -> white =
/// growing delta; gain 4x so a 64-count delta saturates). Space-in-both stays black.
fn parity_heatmap_rgba(cpu: &[u8], gpu: &[u8]) -> Vec<u8> {
    let n = (cpu.len() / 4).min(gpu.len() / 4);
    let mut out = vec![0u8; n * 4];
    for i in 0..n {
        let c = &cpu[i * 4..i * 4 + 4];
        let g = &gpu[i * 4..i * 4 + 4];
        let px = &mut out[i * 4..i * 4 + 4];
        px[3] = 255;
        if c[3] == 0 && g[3] == 0 {
            continue;
        }
        let d = c[0]
            .abs_diff(g[0])
            .max(c[1].abs_diff(g[1]))
            .max(c[2].abs_diff(g[2])) as u32;
        px[0] = (d * 4).min(255) as u8;
        px[1] = (d * 2).min(255) as u8;
        px[2] = d.min(255) as u8;
    }
    out
}

/// The `SurfaceFrameInputs` view of a prepared frame — shared by the clouds-off GPU
/// surface pass and the surface half of the GPU cloud pass.
fn surface_inputs(prep: &PreparedRender) -> SurfaceFrameInputs<'_> {
    SurfaceFrameInputs {
        width: prep.width,
        height: prep.height,
        lut_geo: prep.lut_geo.as_slice(),
        lut_light: &prep.lut_light,
        nx: prep.nx,
        ny: prep.ny,
        normals_rgba: &prep.normals_rgba,
        landmask_r8: &prep.landmask_r8,
        bluemarble: prep.bluemarble.as_ref().map(|a| &a.0),
        transmittance_lut: &prep.transmittance_lut,
        multiscatter_lut: &prep.multiscatter_lut,
        ambient_lut: &prep.ambient_lut,
        ambient_n: prep.ambient_n,
        uniforms: prep.uniforms,
    }
}

enum WorkerMsg {
    Status(String),
    Prepared(Box<PreparedRender>),
    Error(String),
    /// One finished frame of a batch (loop) render — its index in the sequence, the
    /// total, the prepared inputs, and the worker-side prepare wall time (ms) for
    /// the per-frame throughput log. The UI thread finishes the GPU pass (if any),
    /// writes it to the store, and retains it as a `LoopFrame`.
    BatchFrame {
        index: usize,
        total: usize,
        prep: Box<PreparedRender>,
        prep_ms: u64,
    },
    /// A single sequence frame failed to prepare (a bad file / missing brick); the
    /// batch continues with the rest.
    BatchError {
        index: usize,
        message: String,
    },
    /// The batch worker finished (or was cancelled): how many frames it rendered and
    /// whether it stopped early on the cancel flag.
    BatchDone {
        rendered: usize,
        cancelled: bool,
    },
}

/// A rendered frame held for display + store write.
struct RenderedState {
    texture: egui::TextureHandle,
    rendered: RenderedFrame,
    lat: Vec<f32>,
    lon: Vec<f32>,
    sector: String,
    satellite: SatellitePreset,
    /// View mode this frame was rendered in (Geostationary, Top-down, Perspective).
    view_mode: StudioView,
    year: i32,
    month: u32,
    day: u32,
    hhmm: u16,
    bm_status: BmStatus,
    /// Seasonal Blue Marble status line (M7); empty in IR / flat-albedo.
    season_line: String,
    center_sun_elev: f64,
    sun_override: bool,
    resolution: ResolutionMode,
    res_clamped: bool,
    /// The M6 IR / WV BT plane (Kelvin, `NaN` off-earth), `Some` in a thermal mode — kept
    /// for live re-enhancement + the single-band Kelvin store write.
    ir_bt: Option<Vec<f32>>,
    /// The enhancement the currently-displayed thermal frame is coloured with.
    ir_enhancement: IrEnhancement,
    /// The ABI band of the thermal frame (13, or 8/9/10 for WV).
    ir_band: u8,
    /// Instrument spatial response actually applied to the thermal component.
    instrument_footprint: InstrumentFootprint,
    /// The DERIVED scalar field + its precomputed value-range stats (`Some` only in a
    /// derived-field mode) — the header's value-range readout, computed once at render.
    derived: Option<(DerivedField, derived::FieldStats)>,
    /// The product mode this frame was rendered in (for the PNG-export file name).
    render_mode: RenderMode,
    /// `Some(wall ms)` when this frame was rendered by the EXPERIMENTAL GPU cloud
    /// pass — such a frame is a preview and is never written to the store (the
    /// stored-frame path stays CPU for quality/provenance).
    gpu_ms: Option<u64>,
    /// True when the frame came from the dedicated one-click `GPU Render` action.
    one_click_gpu_render: bool,
    /// Temporary preview changes, shown beside a successful GPU frame.
    gpu_preview_adjustments: GpuPreviewAdjustments,
}

/// One rendered frame retained in memory for instant loop playback (the
/// prerender-then-play product). Small: an egui texture handle + display metadata;
/// the heavy `PreparedRender` is dropped once the texture is built and the frame is
/// written to the store.
struct LoopFrame {
    texture: egui::TextureHandle,
    width: u32,
    height: u32,
    /// The frame's valid-time label (shown on the timeline).
    label: String,
    /// A short per-frame summary for the status line (sun/IR info).
    summary: String,
}

/// The in-studio animation timeline over the prerendered `LoopFrame`s. Playback is a
/// pure display cycle through already-rendered textures (instant, since prerendered);
/// the frame-index/loop/fps math lives in `pipeline` and is unit-tested.
struct LoopState {
    frames: Vec<LoopFrame>,
    current: usize,
    playing: bool,
    looping: bool,
    /// Sub-frame time accumulator for the fps stepping (seconds), carried per tick.
    accumulator: f32,
    /// Total frames rendered + written to the store (may exceed `frames.len()` when the
    /// in-memory retention cap truncated the retained set — the store run is complete).
    total_rendered: usize,
    /// Whether the in-memory retention was capped (the store still has every frame).
    capped: bool,
    /// The store run the frames were written into (for the status line).
    store_run: Option<String>,
    /// Whether this loop is a thermal (IR/WV) band (labels the timeline honestly).
    is_ir: bool,
    /// The ABI band of a thermal loop (13, or 8/9/10 for WV) for the header label.
    ir_band: u8,
}

impl LoopState {
    fn new() -> Self {
        Self {
            frames: Vec::new(),
            current: 0,
            playing: false,
            looping: true,
            accumulator: 0.0,
            total_rendered: 0,
            capped: false,
            store_run: None,
            is_ir: false,
            ir_band: 13,
        }
    }
}

/// Live state of an in-flight batch (loop) render.
struct BatchState {
    total: usize,
    done: usize,
    errors: usize,
    /// Set by the Cancel button; the worker checks it at each frame boundary.
    cancel: Arc<AtomicBool>,
    /// Rolling total UI-side finish time across frames (ms), for the per-frame average
    /// reported at the end (the QA's per-frame wall time).
    total_frame_ms: u64,
}

struct SimSatStudioApp {
    gpu: Option<GpuCtx>,
    gpu_error: Option<String>,
    store_root: PathBuf,
    preset: SatellitePreset,
    /// Geostationary sensor-grid navigation; model sphere remains the default.
    geo_navigation: GeoNavigation,
    /// Output-raster resolution mode (default Native — one pixel per WRF cell).
    resolution: ResolutionMode,
    /// View mode: the from-space geostationary product (default), the top-down
    /// map-registered product (the WRF-Runner integration view), or the
    /// Perspective (3-D) orbit view. Top-down and Perspective always render on
    /// the CPU (like the shipped clouds/IR path).
    view: StudioView,
    /// Perspective orbit-camera params (see `pipeline::OrbitParams`): compass
    /// azimuth the camera sits FROM the domain centre, tilt above the horizontal,
    /// slant range (km, clamped at render to the domain-derived bounds), and the
    /// horizontal FOV. Persisted.
    orbit_az_deg: f32,
    orbit_tilt_deg: f32,
    orbit_range_km: f32,
    orbit_fov_deg: f32,
    /// Perspective output size (px per axis, 2..=4096; default 1280x720).
    persp_width: u32,
    persp_height: u32,
    /// Zoom-out / domain MARGIN as a PERCENTAGE (0-100%) of the domain size added on each
    /// side (the "Zoom out / margin" slider). 0 = the domain edge-to-edge (default). The
    /// margin frames the domain with the real surrounding earth (Blue Marble ground + clear
    /// sky; no WRF weather outside the domain). Converted to a fraction for the render param.
    margin_pct: f32,
    source: Option<Source>,
    timesteps: Vec<Timestep>,
    selected_ts: usize,
    busy: bool,
    /// When the current busy phase (render / batch) started — the status bar
    /// appends the elapsed seconds so a long march visibly makes progress.
    busy_since: Option<Instant>,
    worker_rx: Option<Receiver<WorkerMsg>>,
    status: String,
    /// The in-app log ring (info + error) with the STICKY banner error (cleared
    /// only by Dismiss or a subsequent successful render); see `pipeline::LogBuffer`.
    log: pipeline::LogBuffer,
    /// Whether the log view (a scrollable history) is expanded in the status bar.
    show_log: bool,
    /// Timestep-independent scene resources shared across renders (WS4 item 1);
    /// locked by the ONE in-flight worker at a time (serialized by `busy`).
    scene_cache: Arc<Mutex<SceneCache>>,
    rendered: Option<RenderedState>,
    /// DISPLAY-side viewport zoom, as a factor over the fit-to-window scale (1.0 = fit,
    /// clamped 1.0..=`MAX_VIEW_ZOOM`). Magnifies the already-rendered frame; no re-render.
    view_zoom: f32,
    /// DISPLAY-side viewport pan (screen-px offset of the image centre from the viewport
    /// centre), clamped so the image cannot be dragged past its own edges.
    view_pan: egui::Vec2,
    last_written: Option<String>,
    /// The prerendered animation loop (M7): `Some` once a sequence has been batch
    /// rendered. Drives the timeline + the central display when present.
    loop_state: Option<LoopState>,
    /// In-flight batch (loop) render progress + cancel handle; `Some` while rendering.
    batch: Option<BatchState>,
    /// Playback rate for the in-studio timeline (frames per second).
    play_fps: f32,
    /// Max full-res frames retained in memory for instant playback (the store run is
    /// always complete regardless). Default 120; at native Enderlin 800x800 RGBA that
    /// is ~2.56 MB/frame -> ~307 MB, within the 2 GB budget (owner decision 1). A UI
    /// knob; beyond it the batch still renders + stores every frame but stops retaining
    /// textures (a truncated in-studio timeline, flagged in the status).
    frame_cap: usize,
    /// Render mode: Visible (the M1-M5 physically-based path) or IR (M6, band 13).
    render_mode: RenderMode,
    /// Display (shipped appearance) or the strict `simsat-fast-gray-v1` operator.
    /// Session-scoped so startup remains the familiar Display mode.
    render_intent: RenderIntent,
    /// The IR enhancement (colour curve) applied to the BT plane in IR mode.
    ir_enhancement: IrEnhancement,
    /// Band-13 source/response model. Fast gray is the unchanged default; the
    /// official FM4/GOES-19 ABI SRF is an opt-in observation-operator prototype.
    thermal_sensor: ThermalSensor,
    /// Complete-radiance ABI Band 13 spatial-response stage. Experimental/default off.
    instrument_footprint: InstrumentFootprint,
    // M2 atmosphere controls (design section 3 / 6).
    aod: f32,
    rh_swelling: bool,
    /// Daytime aerial-veil correction. On is the product-facing
    /// default; the toggle exists for an exact corrected/raw-TOA QA A/B.
    atmosphere_correction: bool,
    /// Clip the atmospheric column to each surface pixel's terrain elevation.
    /// On is physical; off reproduces the old full sea-level column for QA.
    terrain_atmosphere: bool,
    /// Land-only solar-zenith display normalization (owner-selected default on).
    land_sza_normalization: bool,
    land_sza_max_gain: f32,
    /// Bounded dark-reflectance land toe (owner-selected default on).
    land_dark_toe: bool,
    land_dark_toe_knee: f32,
    land_dark_toe_gamma: f32,
    land_dark_toe_max_gain: f32,
    /// Default-off terrain-only toe after lighting/view attenuation and before airlight.
    surface_postlight_toe: bool,
    surface_postlight_toe_knee: f32,
    surface_postlight_toe_gamma: f32,
    surface_postlight_toe_max_gain: f32,
    /// Stronger, tightly gated low-sun recovery selected in visual QA; shipped on for
    /// finished visible displays and independently switchable.
    twilight_surface_recovery: bool,
    twilight_surface_recovery_knee: f32,
    twilight_surface_recovery_gamma: f32,
    twilight_surface_recovery_max_gain: f32,
    output_transform: OutputTransform,
    // M4 cloud controls (design section 4) + M5 multi-scatter (section 4/6).
    clouds_enabled: bool,
    /// Use model cloud fraction/subcolumns when available. On is the physical default;
    /// off is the legacy full-horizontal-cell cloud coverage A/B.
    fractional_clouds: bool,
    /// CPU fractional-cloud observation operator. Effective-OD is the unchanged
    /// shipped default; deterministic 4/8/16 are expensive fixed-stratified references.
    fractional_cloud_mode: FractionalCloudMode,
    multiscatter: bool,
    /// Opt-in Stage-2 Monte Carlo depth-source cloud closure. False keeps the exact
    /// established octave/single-scatter dispatch.
    delta_flux_clouds: bool,
    /// Opt-in bounded P1 directional reconstruction over the Stage-2 closure.
    delta_flux_v2_clouds: bool,
    /// Opt-in bounded successive-order angular-memory reconstruction.
    delta_flux_v3_clouds: bool,
    /// Visible cloud optical-depth calibration. The shipped 0.15 is the owner's
    /// cross-file visual selection; 1.0 uses model extinction without scaling.
    cloud_optical_depth_scale: f32,
    /// Opt-in ScienceCloudF16 on-disk extinction precision. CompactU8 remains default.
    science_cloud_f16: bool,
    /// Experimental WRF NSSL MP18 mass/number/volume-moment particle optics.
    /// False is the stable fixed-radius table; true uses a separate ingest cache.
    nssl_native_cloud_optics: bool,
    /// Experimental HRRR Thompson mass/number/temperature particle optics.
    /// False is exact fixed optics; true uses a separate versioned ingest cache.
    hrrr_thompson_native_cloud_optics: bool,
    /// Default-off presentation experiment: taper finished visible clouds where the
    /// camera exposes the finite WRF domain boundary even without a zoom-out margin.
    feather_exposed_domain_edges: bool,
    beer_powder: bool,
    /// Sub-grid cloud GRANULATION (edge-erosion detail noise; see the clouds.rs
    /// granulation section). OFF by default as of v0.1.1 — the round-1 default look
    /// was owner-rejected on coarse-grid decks ("cheese grater"); OPT-IN via the
    /// Clouds-group toggle until the tune-2 rework re-earns the default (matches the
    /// api's opt-in scoping; raw-Kelvin thermal modes never granulate regardless).
    /// The amplitude is dx-derived, so a fine (250 m) run is near-neutral and a
    /// coarse (2-3 km) run granulates strongly. Persisted as an explicit opt-in.
    granulation: bool,
    /// Top-down-only source-space reconstruction for broad low stratiform cloud.
    /// Opt-in/default off while v0.1.6 multi-case QA is in progress.
    topdown_stratiform_regularization: bool,
    /// Display-only sigma~=1.225 px cloud-radiance footprint for finished top-down
    /// visible frames. The sharp surface/base map is never filtered.
    topdown_cloud_footprint: bool,
    /// Default-on display filter for top-down ground cloud-shadow map aliasing.
    topdown_shadow_antialias: bool,
    step_quality: StepQuality,
    /// EXPERIMENTAL "GPU clouds" toggle (the M5-GPU cloud-pass activation): when on,
    /// the DISPLAYED Geostationary or Top-down Visible clouds-on frame renders through the
    /// `clouds.wgsl` GPU pass at the Interactive schedule instead of the CPU
    /// composite. The CPU path remains the shipping default and ground truth:
    /// Write-store and sequence batch renders ALWAYS use the CPU path regardless.
    /// Session-scoped (deliberately not persisted while experimental). Default OFF.
    gpu_clouds: bool,
    /// One-shot "GPU parity check" request: the next render marches BOTH paths
    /// (GPU pass + a CPU reference at GPU-comparable settings), logs the per-channel
    /// mean/p95/max |delta| and keeps a delta heatmap for review.
    parity_pending: bool,
    /// The last parity report (summary + heatmap texture), shown in the drawer.
    parity: Option<ParityReport>,
    /// Display-side exposure gain applied before the ABI stretch (see
    /// `render::radiance_to_rgba`). Affects surface + cloud together. A clear-sky
    /// configuration the GPU surface shader cannot represent routes through CPU.
    exposure: f32,
    /// Sun-gated daytime ground-radiance lift (`1.0` is neutral).
    ground_gain: f32,
    /// Finished-display highlight shoulder knee (`1.0` disables the shoulder).
    cloud_softclip: f32,
    /// Physical reflectance-factor ceiling mapped to display white.
    cloud_highlight_max: f32,
    /// "Fake sun" / what-if OVERRIDE (a deliberate, NON-PHYSICAL visualization aid): when
    /// on, the whole frame is lit by a single sun direction at `sun_override_elev` /
    /// `sun_override_az` over the domain centre (uniform sun-at-infinity, exactly like the
    /// `render_frame` sun-elev override), regardless of the file's real timestamp — e.g.
    /// "show me this night storm at noon". Off = the file's real solar geometry (the
    /// physically-honest default). When on, the status bar labels the frame a what-if so
    /// it is never mistaken for the satellite's real view at that time.
    sun_override: bool,
    sun_override_elev: f32,
    sun_override_az: f32,
    /// Seasonal Blue Marble (M7): manual month override (0 = auto/day-of-year blend,
    /// 1..=12 = force that month for a what-if) and whether to lazily download months.
    bm_month_override: u32,
    bm_allow_download: bool,
    /// Full-year-pack download worker: a simple status-string channel drained in `ui`.
    pack_rx: Option<Receiver<String>>,
    pack_busy: bool,
    /// "Save PNG..." export worker (the encode runs on a spawned thread so the UI
    /// never stalls on a large frame — the M1 NOTE-5 lesson).
    export_rx: Option<Receiver<ExportMsg>>,
    export_busy: bool,
    /// Settings persistence (WS4 item 2): the settings.json path, the last-saved
    /// snapshot (save-on-change compares against it), and the debounce timer.
    /// Epoch zero preserves a pre-v0.1.4 calibration until the user explicitly
    /// accepts the new preset or keeps their saved controls in the migration banner.
    visible_calibration_epoch: u32,
    settings_path: PathBuf,
    last_saved: settings::StudioSettings,
    settings_dirty_since: Option<Instant>,
    /// Recent open actions (newest first, capped, pruned of missing paths).
    recent: Vec<settings::RecentEntry>,
}

/// Result of the PNG-export worker thread.
enum ExportMsg {
    Ok(String),
    Err(String),
}

impl SimSatStudioApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let (gpu, gpu_error) = match cc.wgpu_render_state.as_ref() {
            Some(rs) => (
                Some(GpuCtx {
                    device: rs.device.clone(),
                    queue: rs.queue.clone(),
                    resources: SurfaceResources::init(&rs.device),
                    cloud_resources: CloudPassResources::init(&rs.device),
                }),
                None,
            ),
            None => (
                None,
                Some(
                    "No wgpu render backend available (the app was started without wgpu). \
                     Rendering is disabled."
                        .to_string(),
                ),
            ),
        };
        // Settings persistence: load (defaults on ANY error, values sanitized),
        // then apply below via `apply_settings` so the mapping lives in one place.
        let settings_path = settings::settings_path();
        let loaded = settings::load(&settings_path);
        let land_appearance = LandAppearanceConfig::default();
        let surface_postlight_toe = SurfacePostlightToeConfig::off();
        let twilight_surface_recovery = TwilightSurfaceRecoveryConfig::shipped();
        let mut app = Self {
            gpu,
            gpu_error,
            store_root: default_store_root(),
            preset: SatellitePreset::GoesEast,
            geo_navigation: GeoNavigation::ModelSphere,
            // Native (full WRF resolution) is the default the owner sees — never the
            // coarse fixed ABI pitch that undersampled and pixelated fine domains.
            resolution: ResolutionMode::Native,
            // Geostationary from-space is the default view; Top-down map is the
            // integration product; Perspective (3-D) is the orbit flyover view.
            view: StudioView::Geostationary,
            orbit_az_deg: 180.0,
            orbit_tilt_deg: 30.0,
            orbit_range_km: 300.0,
            orbit_fov_deg: 45.0,
            persp_width: 1280,
            persp_height: 720,
            // Zoom-out / margin OFF by default: the domain renders edge-to-edge (the
            // pre-margin behavior). The owner drags it up to frame the domain with earth.
            margin_pct: 0.0,
            source: None,
            timesteps: Vec::new(),
            selected_ts: 0,
            busy: false,
            busy_since: None,
            worker_rx: None,
            status: "Open a wrfout file or a cached run.json to begin.".to_string(),
            log: pipeline::LogBuffer::new(300),
            show_log: false,
            scene_cache: Arc::new(Mutex::new(SceneCache::default())),
            rendered: None,
            // Fit-to-window, no pan (reset on each new render).
            view_zoom: 1.0,
            view_pan: egui::Vec2::ZERO,
            last_written: None,
            // No loop until a sequence is batch rendered.
            loop_state: None,
            batch: None,
            // 8 fps is a readable satellite-loop default (the owner adjusts live).
            play_fps: 8.0,
            frame_cap: 120,
            // Visible is the default mode; IR (band 13) is the M6 toggle.
            render_mode: RenderMode::Visible,
            render_intent: RenderIntent::Display,
            // CIMSS Style is the default/recommended Band-13 display.
            ir_enhancement: IrEnhancement::default(),
            thermal_sensor: ThermalSensor::FastGray,
            instrument_footprint: InstrumentFootprint::Off,
            aod: atmosphere::DEFAULT_AOD as f32,
            rh_swelling: false,
            atmosphere_correction: true,
            terrain_atmosphere: true,
            land_sza_normalization: land_appearance.sza_normalization,
            land_sza_max_gain: land_appearance.sza_max_gain as f32,
            land_dark_toe: land_appearance.dark_toe,
            land_dark_toe_knee: land_appearance.dark_toe_knee as f32,
            land_dark_toe_gamma: land_appearance.dark_toe_gamma as f32,
            land_dark_toe_max_gain: land_appearance.dark_toe_max_gain as f32,
            surface_postlight_toe: surface_postlight_toe.enabled,
            surface_postlight_toe_knee: surface_postlight_toe.knee as f32,
            surface_postlight_toe_gamma: surface_postlight_toe.gamma as f32,
            surface_postlight_toe_max_gain: surface_postlight_toe.max_gain as f32,
            twilight_surface_recovery: twilight_surface_recovery.enabled,
            twilight_surface_recovery_knee: twilight_surface_recovery.knee as f32,
            twilight_surface_recovery_gamma: twilight_surface_recovery.gamma as f32,
            twilight_surface_recovery_max_gain: twilight_surface_recovery.max_gain as f32,
            output_transform: OutputTransform::AbiReflectance,
            clouds_enabled: true,
            // Model fractional cloud coverage ON by default; missing fields fall back safely.
            fractional_clouds: true,
            fractional_cloud_mode: FractionalCloudMode::Deterministic2,
            // Wrenninge multi-scatter octaves ON by default (M5): the bright-anvil look.
            multiscatter: true,
            delta_flux_clouds: false,
            delta_flux_v2_clouds: false,
            delta_flux_v3_clouds: false,
            // Owner-selected v0.1.4 cross-file visible calibration.
            cloud_optical_depth_scale: clouds::DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE,
            science_cloud_f16: false,
            nssl_native_cloud_optics: false,
            hrrr_thompson_native_cloud_optics: false,
            // Owner-selected v0.1.5 finite-domain cloud-edge presentation default.
            feather_exposed_domain_edges: true,
            // Beer-powder OFF by default (M5): octaves now supply the real forward-
            // scatter buildup, so powder-on would double-darken (design M5 decision).
            beer_powder: false,
            // Sub-grid granulation OPT-IN (off) as of v0.1.1 — see the field doc.
            granulation: false,
            topdown_stratiform_regularization: false,
            topdown_cloud_footprint: false,
            topdown_shadow_antialias: true,
            // Offline (384 steps, full quality) is the default so the displayed AND
            // stored frame is full quality (owner decision: stored quality never
            // reduced); Interactive (192) is the faster preview choice.
            step_quality: StepQuality::Offline,
            // GPU clouds OFF by default (experimental preview; CPU stays shipping).
            gpu_clouds: false,
            parity_pending: false,
            parity: None,
            // Owner-selected display gain before the ABI square-root stretch;
            // exposure 1.0 remains the exact neutral override.
            exposure: simsat::render::DEFAULT_EXPOSURE as f32,
            ground_gain: GROUND_DAY_LIFT as f32,
            cloud_softclip: CLOUD_SOFTCLIP_KNEE as f32,
            cloud_highlight_max: RHO_HIGHLIGHT_MAX as f32,
            // Fake-sun override OFF by default (honest real-timestamp sun). Defaults when
            // toggled on: a mid daytime sun from the south (a clear "what-if daylight").
            sun_override: false,
            sun_override_elev: 45.0,
            sun_override_az: 180.0,
            // Seasonal ground: auto (day-of-year blend) by default; lazy download on.
            bm_month_override: 0,
            bm_allow_download: true,
            pack_rx: None,
            pack_busy: false,
            export_rx: None,
            export_busy: false,
            visible_calibration_epoch: loaded.visible_calibration_epoch,
            settings_path,
            last_saved: loaded.clone(),
            settings_dirty_since: None,
            recent: Vec::new(),
        };
        app.apply_settings(&loaded);
        // Drop recent entries whose files vanished since last session; the next
        // autosave persists the pruned list.
        settings::prune_recent(&mut app.recent, &|p: &str| Path::new(p).exists());
        app
    }

    /// Append an INFO line to the log ring and mirror it into the status bar.
    fn logline(&mut self, msg: impl Into<String>) {
        let msg = msg.into();
        self.status = msg.clone();
        self.log.info(msg);
    }

    /// Append an ERROR line: status bar + log ring + the STICKY red banner (which
    /// only Dismiss or a subsequent successful render clears — a later status
    /// message never hides a failure).
    fn logerr(&mut self, msg: impl Into<String>) {
        let msg = msg.into();
        self.status = msg.clone();
        self.log.error(msg);
    }

    // ── settings persistence (WS4 item 2) ────────────────────────────────────

    /// Apply a loaded settings snapshot onto the app state. Unknown tokens have
    /// already been reset by `sanitize`, so every `from_token` here resolves; the
    /// `unwrap_or` defaults are pure defense. The fake-sun override is DELIBERATELY
    /// absent (never persisted — sessions always start with the honest sun).
    fn apply_settings(&mut self, s: &settings::StudioSettings) {
        self.visible_calibration_epoch = s.visible_calibration_epoch;
        if let Some(root) = &s.store_root {
            self.store_root = PathBuf::from(root);
        }
        self.preset = settings::sat_from_token(&s.sat).unwrap_or(SatellitePreset::GoesEast);
        self.geo_navigation = settings::geo_navigation_from_token(&s.geo_navigation)
            .unwrap_or(GeoNavigation::ModelSphere);
        self.resolution =
            settings::resolution_from_token(&s.resolution).unwrap_or(ResolutionMode::Native);
        self.view = settings::view_from_token(&s.view).unwrap_or(StudioView::Geostationary);
        self.orbit_az_deg = s.orbit_az_deg;
        self.orbit_tilt_deg = s.orbit_tilt_deg;
        self.orbit_range_km = s.orbit_range_km;
        self.orbit_fov_deg = s.orbit_fov_deg;
        self.persp_width = s.persp_width;
        self.persp_height = s.persp_height;
        self.render_mode = settings::mode_from_token(&s.mode).unwrap_or(RenderMode::Visible);
        self.render_intent =
            settings::render_intent_from_token(&s.render_intent).unwrap_or(RenderIntent::Display);
        self.ir_enhancement =
            settings::enhancement_from_token(&s.ir_enhancement).unwrap_or_default();
        self.thermal_sensor =
            settings::thermal_sensor_from_token(&s.thermal_sensor).unwrap_or_default();
        self.instrument_footprint =
            settings::instrument_footprint_from_token(&s.instrument_footprint).unwrap_or_default();
        self.output_transform = settings::output_transform_from_token(&s.output_transform)
            .unwrap_or(OutputTransform::AbiReflectance);
        self.step_quality =
            settings::step_quality_from_token(&s.step_quality).unwrap_or(StepQuality::Offline);
        self.margin_pct = s.margin_pct;
        self.aod = s.aod;
        self.rh_swelling = s.rh_swelling;
        self.atmosphere_correction = s.atmosphere_correction;
        self.terrain_atmosphere = s.terrain_atmosphere;
        self.land_sza_normalization = s.land_sza_normalization;
        self.land_sza_max_gain = s.land_sza_max_gain;
        self.land_dark_toe = s.land_dark_toe;
        self.land_dark_toe_knee = s.land_dark_toe_knee;
        self.land_dark_toe_gamma = s.land_dark_toe_gamma;
        self.land_dark_toe_max_gain = s.land_dark_toe_max_gain;
        self.surface_postlight_toe = s.surface_postlight_toe;
        self.surface_postlight_toe_knee = s.surface_postlight_toe_knee;
        self.surface_postlight_toe_gamma = s.surface_postlight_toe_gamma;
        self.surface_postlight_toe_max_gain = s.surface_postlight_toe_max_gain;
        self.twilight_surface_recovery = s.twilight_surface_recovery;
        self.twilight_surface_recovery_knee = s.twilight_surface_recovery_knee;
        self.twilight_surface_recovery_gamma = s.twilight_surface_recovery_gamma;
        self.twilight_surface_recovery_max_gain = s.twilight_surface_recovery_max_gain;
        self.clouds_enabled = s.clouds_enabled;
        self.fractional_clouds = s.fractional_clouds;
        self.fractional_cloud_mode =
            settings::fractional_cloud_mode_from_token(&s.fractional_cloud_mode)
                .unwrap_or(FractionalCloudMode::Deterministic2);
        self.multiscatter = s.multiscatter;
        self.delta_flux_clouds = s.delta_flux_clouds;
        self.delta_flux_v2_clouds = s.delta_flux_v2_clouds;
        self.delta_flux_v3_clouds = s.delta_flux_v3_clouds;
        self.cloud_optical_depth_scale = s.cloud_optical_depth_scale;
        self.science_cloud_f16 = s.science_cloud_f16;
        self.nssl_native_cloud_optics = s.nssl_native_cloud_optics;
        self.hrrr_thompson_native_cloud_optics = s.hrrr_thompson_native_cloud_optics;
        self.feather_exposed_domain_edges = s.feather_exposed_domain_edges;
        self.beer_powder = s.beer_powder;
        self.granulation = s.granulation;
        self.topdown_stratiform_regularization = s.topdown_stratiform_regularization;
        self.topdown_cloud_footprint = s.topdown_cloud_footprint;
        self.topdown_shadow_antialias = s.topdown_shadow_antialias;
        self.exposure = s.exposure;
        self.ground_gain = s.ground_gain;
        self.cloud_softclip = s.cloud_softclip;
        self.cloud_highlight_max = s.cloud_highlight_max;
        self.bm_month_override = s.bm_month_override;
        self.bm_allow_download = s.bm_allow_download;
        self.play_fps = s.play_fps;
        self.frame_cap = s.frame_cap;
        self.recent = s.recent.clone();
    }

    /// Capture the current persistable state (the inverse of `apply_settings`).
    fn settings_snapshot(&self) -> settings::StudioSettings {
        settings::StudioSettings {
            visible_calibration_epoch: self.visible_calibration_epoch,
            store_root: Some(self.store_root.display().to_string()),
            sat: settings::sat_token(self.preset).to_string(),
            geo_navigation: settings::geo_navigation_token(self.geo_navigation).to_string(),
            resolution: settings::resolution_token(self.resolution).to_string(),
            view: settings::view_token(self.view).to_string(),
            mode: settings::mode_token(self.render_mode).to_string(),
            render_intent: settings::render_intent_token(self.render_intent).to_string(),
            ir_enhancement: settings::enhancement_token(self.ir_enhancement).to_string(),
            thermal_sensor: settings::thermal_sensor_token(self.thermal_sensor).to_string(),
            instrument_footprint: settings::instrument_footprint_token(self.instrument_footprint)
                .to_string(),
            output_transform: settings::output_transform_token(self.output_transform).to_string(),
            step_quality: settings::step_quality_token(self.step_quality).to_string(),
            margin_pct: self.margin_pct,
            aod: self.aod,
            rh_swelling: self.rh_swelling,
            atmosphere_correction: self.atmosphere_correction,
            terrain_atmosphere: self.terrain_atmosphere,
            land_sza_normalization: self.land_sza_normalization,
            land_sza_max_gain: self.land_sza_max_gain,
            land_dark_toe: self.land_dark_toe,
            land_dark_toe_knee: self.land_dark_toe_knee,
            land_dark_toe_gamma: self.land_dark_toe_gamma,
            land_dark_toe_max_gain: self.land_dark_toe_max_gain,
            surface_postlight_toe: self.surface_postlight_toe,
            surface_postlight_toe_knee: self.surface_postlight_toe_knee,
            surface_postlight_toe_gamma: self.surface_postlight_toe_gamma,
            surface_postlight_toe_max_gain: self.surface_postlight_toe_max_gain,
            twilight_surface_recovery: self.twilight_surface_recovery,
            twilight_surface_recovery_knee: self.twilight_surface_recovery_knee,
            twilight_surface_recovery_gamma: self.twilight_surface_recovery_gamma,
            twilight_surface_recovery_max_gain: self.twilight_surface_recovery_max_gain,
            clouds_enabled: self.clouds_enabled,
            fractional_clouds: self.fractional_clouds,
            fractional_cloud_mode: settings::fractional_cloud_mode_token(
                self.fractional_cloud_mode,
            )
            .to_string(),
            multiscatter: self.multiscatter,
            delta_flux_clouds: self.delta_flux_clouds,
            delta_flux_v2_clouds: self.delta_flux_v2_clouds,
            delta_flux_v3_clouds: self.delta_flux_v3_clouds,
            cloud_optical_depth_scale: self.cloud_optical_depth_scale,
            science_cloud_f16: self.science_cloud_f16,
            nssl_native_cloud_optics: self.nssl_native_cloud_optics,
            hrrr_thompson_native_cloud_optics: self.hrrr_thompson_native_cloud_optics,
            feather_exposed_domain_edges: self.feather_exposed_domain_edges,
            beer_powder: self.beer_powder,
            granulation: self.granulation,
            topdown_stratiform_regularization: self.topdown_stratiform_regularization,
            topdown_cloud_footprint: self.topdown_cloud_footprint,
            topdown_shadow_antialias: self.topdown_shadow_antialias,
            exposure: self.exposure,
            ground_gain: self.ground_gain,
            cloud_softclip: self.cloud_softclip,
            cloud_highlight_max: self.cloud_highlight_max,
            bm_month_override: self.bm_month_override,
            bm_allow_download: self.bm_allow_download,
            play_fps: self.play_fps,
            frame_cap: self.frame_cap,
            orbit_az_deg: self.orbit_az_deg,
            orbit_tilt_deg: self.orbit_tilt_deg,
            orbit_range_km: self.orbit_range_km,
            orbit_fov_deg: self.orbit_fov_deg,
            persp_width: self.persp_width,
            persp_height: self.persp_height,
            recent: self.recent.clone(),
        }
    }

    /// Save-on-change with a short debounce (a slider drag emits one save, not
    /// hundreds), called once per UI frame; `on_exit` calls `save_settings_now`
    /// as the crash-conscious backstop.
    fn tick_settings_autosave(&mut self, ctx: &egui::Context) {
        let snap = self.settings_snapshot();
        if snap == self.last_saved {
            self.settings_dirty_since = None;
            return;
        }
        let since = *self.settings_dirty_since.get_or_insert_with(Instant::now);
        if since.elapsed() >= Duration::from_millis(750) {
            if let Err(e) = settings::save(&self.settings_path, &snap) {
                // Log ONCE per change set (last_saved advances either way so a
                // persistent disk failure cannot spam every frame; on_exit retries).
                self.log.error(format!("Settings save failed: {e}"));
            }
            self.last_saved = snap;
            self.settings_dirty_since = None;
        } else {
            // Ensure the save fires even if no further input arrives.
            ctx.request_repaint_after(Duration::from_millis(800));
        }
    }

    /// Immediate settings save (the on-exit backstop).
    fn save_settings_now(&mut self) {
        let snap = self.settings_snapshot();
        if snap != self.last_saved {
            let _ = settings::save(&self.settings_path, &snap);
            self.last_saved = snap;
        }
    }

    /// Remember a successful open action in the recent list (deduped, capped).
    fn remember_recent(&mut self, kind: &str, paths: Vec<String>) {
        settings::push_recent(
            &mut self.recent,
            settings::RecentEntry {
                kind: kind.to_string(),
                paths,
            },
        );
    }

    /// Re-run a remembered open action (the Open-menu recent list / "Reopen last").
    fn reopen_recent(&mut self, entry: &settings::RecentEntry) {
        let paths: Vec<PathBuf> = entry.paths.iter().map(PathBuf::from).collect();
        match (entry.kind.as_str(), paths.as_slice()) {
            ("wrfout", [p]) => self.open_wrfout(p.clone()),
            ("cached", [p]) => self.open_cached_run(p.clone()),
            ("sequence", _) if !paths.is_empty() => self.open_sequence(paths),
            _ => self.logerr(format!("Recent entry is malformed: {}", entry.label())),
        }
    }

    // ── open dialogs (shared by the Open menu and the first-run CTA) ─────────

    fn dialog_open_wrfout(&mut self) {
        if let Some(path) = model_input_dialog("Open a wrfout or GRIB2 file").pick_file() {
            self.open_wrfout(path);
        }
    }

    fn dialog_open_cached(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("run manifest", &["json"])
            .set_title("Open a cached run.json")
            .pick_file()
        {
            self.open_cached_run(path);
        }
    }

    fn dialog_open_sequence_folder(&mut self) {
        if let Some(dir) = rfd::FileDialog::new()
            .set_title("Open a folder of wrfout files")
            .pick_folder()
        {
            self.open_sequence(vec![dir]);
        }
    }

    fn dialog_open_sequence_files(&mut self) {
        if let Some(files) =
            model_input_dialog("Select wrfout or GRIB2 files for a sequence").pick_files()
        {
            self.open_sequence(files);
        }
    }

    // ── "Save PNG..." export (WS4 item 4) ────────────────────────────────────

    /// Export the currently-displayed rendered frame as an RGB8 PNG. The save
    /// dialog runs modally here; the RGBA->RGB conversion + PNG encode run on a
    /// spawned below-normal thread (a 4096^2 frame encodes in seconds — never on
    /// the UI thread; the M1 NOTE-5 stall lesson). Space (alpha 0) exports black.
    fn start_export(&mut self, ctx: &egui::Context) {
        if self.busy || self.export_busy {
            return;
        }
        let Some(state) = &self.rendered else {
            return;
        };
        let default_name = pipeline::build_export_filename(
            &state.sector,
            product_token(state.render_mode),
            settings::view_token(state.view_mode),
            state.year,
            state.month,
            state.day,
            state.hhmm,
            state.sun_override,
        );
        let Some(path) = rfd::FileDialog::new()
            .set_title("Save frame as PNG")
            .set_file_name(&default_name)
            .add_filter("PNG image", &["png"])
            .save_file()
        else {
            return;
        };
        let rgba = state.rendered.rgba.clone();
        let (w, h) = (state.rendered.width, state.rendered.height);
        let (tx, rx) = channel::<ExportMsg>();
        self.export_rx = Some(rx);
        self.export_busy = true;
        self.logline(format!("Saving PNG {}x{} to {} ...", w, h, path.display()));
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            simsat::platform::lower_worker_thread_priority();
            let rgb = pipeline::rgba_to_rgb_space_black(&rgba);
            let msg = match image::RgbImage::from_raw(w, h, rgb) {
                Some(img) => match img.save(&path) {
                    Ok(()) => ExportMsg::Ok(format!("Saved PNG: {}", path.display())),
                    Err(e) => ExportMsg::Err(format!("PNG save failed: {e}")),
                },
                None => ExportMsg::Err("PNG export failed: buffer size mismatch.".to_string()),
            };
            let _ = tx.send(msg);
            ctx.request_repaint();
        });
    }

    /// Drain the PNG-export worker; clear `export_busy` when its channel closes.
    fn drain_export(&mut self) {
        let mut msgs = Vec::new();
        let mut done = false;
        if let Some(rx) = &self.export_rx {
            loop {
                match rx.try_recv() {
                    Ok(m) => msgs.push(m),
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        done = true;
                        break;
                    }
                }
            }
        }
        for m in msgs {
            match m {
                ExportMsg::Ok(s) => self.logline(s),
                ExportMsg::Err(e) => self.logerr(e),
            }
        }
        if done {
            self.export_rx = None;
            self.export_busy = false;
        }
    }

    fn open_wrfout(&mut self, path: PathBuf) {
        // A GRIB2 file (HRRR wrfnat / RRFS natlev) routes to its own open path;
        // everything downstream (Source::Wrfout, the prepare worker) is shared.
        if ingest_grib::is_grib_input(&path) {
            self.open_grib(path);
            return;
        }
        // Cheap probe: dims only (no field decode) + file size, for the size gate.
        let file = match ingest::probe_wrf(&path) {
            Ok(f) => f,
            Err(e) => {
                self.logerr(format!("Failed to open wrfout: {e}"));
                return;
            }
        };
        self.remember_recent("wrfout", vec![path.display().to_string()]);
        let file_bytes = file.file_bytes;
        let cells_3d = file.nx.saturating_mul(file.ny).saturating_mul(file.nz);
        let needs_confirm =
            cells_3d >= LARGE_WRF_WARN_CELLS_3D || file_bytes >= LARGE_WRF_WARN_BYTES;

        let times = file.times.clone();
        self.timesteps = times
            .iter()
            .enumerate()
            .map(|(idx, t)| {
                let (hhmm, iso) = parse_time(t);
                Timestep {
                    label: t.clone(),
                    hhmm,
                    ts_index: Some(idx),
                    file: bricks::brick_file_name_for(Some(&iso), hhmm),
                    time_iso: Some(iso),
                    seq_index: None,
                }
            })
            .collect();
        if self.timesteps.is_empty() {
            self.timesteps.push(Timestep {
                label: "t0".to_string(),
                hhmm: 0,
                ts_index: Some(0),
                time_iso: None,
                file: bricks::brick_file_name_for(None, 0),
                seq_index: None,
            });
        }
        self.selected_ts = 0;
        self.rendered = None;
        self.loop_state = None;
        let cache_dir = ingest::default_cache_dir();
        let run_id = ingest::default_run_id(&path);
        let (nx, ny, nz) = (file.nx, file.ny, file.nz);
        self.source = Some(Source::Wrfout {
            path,
            cache_dir,
            run_id,
            nx,
            ny,
            nz,
            file_bytes,
            needs_confirm,
            confirmed: false,
        });
        if needs_confirm {
            self.logline(format!(
                "Large WRF file: {nx}x{ny}x{nz} (~{:.1}M cells), {:.2} GB. Confirm to ingest.",
                cells_3d as f64 / 1.0e6,
                file_bytes as f64 / (1u64 << 30) as f64
            ));
        } else {
            self.logline("wrfout opened. Pick a satellite + timestep, then Render.");
        }
    }

    /// GRIB2 sibling of `open_wrfout` (HRRR wrfnat / RRFS natlev): one probe pass
    /// gives dims + the single valid time + the cycle-keyed run id, then the shared
    /// `Source::Wrfout` flow (size gate, prepare worker) carries it — the worker
    /// branches to `ingest_grib_timestep` by extension. A GRIB file is ONE forecast
    /// hour, so it opens with exactly one timestep. NOTE: a full-NA RRFS file will
    /// refuse at ingest with the crop remedy message in the log/error banner (no
    /// crop UI yet — the recorded open decision).
    fn open_grib(&mut self, path: PathBuf) {
        let probe = match ingest_grib::probe_grib(&path) {
            Ok(p) => p,
            Err(e) => {
                self.logerr(format!("Failed to open GRIB2 file: {e}"));
                return;
            }
        };
        self.remember_recent("wrfout", vec![path.display().to_string()]);
        let file_bytes = probe.file_bytes;
        let cells_3d = probe.nx.saturating_mul(probe.ny).saturating_mul(probe.nz);
        let needs_confirm =
            cells_3d >= LARGE_WRF_WARN_CELLS_3D || file_bytes >= LARGE_WRF_WARN_BYTES;
        self.timesteps = vec![Timestep {
            label: probe.time_iso.clone(),
            hhmm: probe.hhmm,
            ts_index: Some(0),
            file: bricks::brick_file_name_for(Some(&probe.time_iso), probe.hhmm),
            time_iso: Some(probe.time_iso.clone()),
            seq_index: None,
        }];
        self.selected_ts = 0;
        self.rendered = None;
        self.loop_state = None;
        let cache_dir = ingest::default_cache_dir();
        let (nx, ny, nz) = (probe.nx, probe.ny, probe.nz);
        self.source = Some(Source::Wrfout {
            path,
            cache_dir,
            run_id: probe.default_run_id,
            nx,
            ny,
            nz,
            file_bytes,
            needs_confirm,
            confirmed: false,
        });
        if needs_confirm {
            self.logline(format!(
                "Large GRIB2 file: {nx}x{ny}x{nz} (~{:.1}M cells), {:.2} GB. Confirm to ingest.",
                cells_3d as f64 / 1.0e6,
                file_bytes as f64 / (1u64 << 30) as f64
            ));
        } else {
            self.logline("GRIB2 opened (one valid time). Pick a satellite, then Render.");
        }
    }

    fn open_cached_run(&mut self, run_json: PathBuf) {
        let storage_profile = if self.science_cloud_f16 {
            StorageProfile::ScienceCloudF16
        } else {
            StorageProfile::CompactU8
        };
        let manifest = match RunManifest::load_for_profile(&run_json, storage_profile) {
            Ok(m) => m,
            Err(e) => {
                self.logerr(format!(
                    "Not a valid {} run.json ({e}): {}. Select the matching storage profile before opening a cached run.",
                    storage_profile.slug(),
                    run_json.display()
                ));
                return;
            }
        };
        self.remember_recent("cached", vec![run_json.display().to_string()]);
        let cache_dir = run_json
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf())
            .unwrap_or_else(ingest::default_cache_dir);
        self.timesteps = manifest
            .timesteps
            .iter()
            .map(|t| Timestep {
                label: t
                    .time_iso
                    .clone()
                    .unwrap_or_else(|| format!("t{:04}", t.hhmm)),
                hhmm: t.hhmm,
                ts_index: None,
                time_iso: t.time_iso.clone(),
                file: t.file.clone(),
                seq_index: None,
            })
            .collect();
        if self.timesteps.is_empty() {
            self.logerr("Cached run has no timesteps.");
            return;
        }
        self.selected_ts = 0;
        self.rendered = None;
        self.loop_state = None;
        let run_id = manifest.run_id.clone();
        self.source = Some(Source::Cached {
            cache_dir,
            run_id,
            manifest,
        });
        self.logline("Cached run opened. Pick a satellite + timestep, then Render.");
    }

    /// Open a time SEQUENCE: a directory of wrfout files (a single directory path
    /// expands to its candidate wrfout files) OR a multi-selection of files. Each file
    /// is probed cheaply (header read) for its `Times`; `pipeline::build_sequence`
    /// orders every timestep by valid time (the Enderlin `01:30..06:00` naming, or a
    /// multi-time wrfout expanded over its time dimension). All frames share ONE run_id
    /// so a batch render lands them in a single multi-frame store run.
    fn open_sequence(&mut self, mut paths: Vec<PathBuf>) {
        // Remember the ORIGINAL selection (a folder stays a folder) so "Reopen"
        // replays the same action even if the folder's contents change.
        let original: Vec<String> = paths.iter().map(|p| p.display().to_string()).collect();
        if paths.len() == 1 && paths[0].is_dir() {
            paths = list_wrfout_files(&paths[0]);
        }
        if paths.is_empty() {
            self.logerr("No wrfout files found in the selection.");
            return;
        }
        let mut file_times: Vec<pipeline::FileTimes> = Vec::new();
        let mut kept: Vec<PathBuf> = Vec::new();
        let mut total_bytes: u64 = 0;
        let mut needs_confirm = false;
        for p in &paths {
            let name = p
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            match ingest::probe_wrf(p) {
                Ok(probe) => {
                    total_bytes = total_bytes.saturating_add(probe.file_bytes);
                    let cells_3d = probe.nx.saturating_mul(probe.ny).saturating_mul(probe.nz);
                    if cells_3d >= LARGE_WRF_WARN_CELLS_3D
                        || probe.file_bytes >= LARGE_WRF_WARN_BYTES
                    {
                        needs_confirm = true;
                    }
                    file_times.push(pipeline::FileTimes {
                        name,
                        times: probe.times,
                    });
                    kept.push(p.clone());
                }
                Err(e) => self.logerr(format!("Skipping {} ({e})", p.display())),
            }
        }
        if kept.is_empty() {
            self.logerr("None of the selected files are readable wrfout files.");
            return;
        }
        let seq = pipeline::build_sequence(&file_times);
        if seq.is_empty() {
            self.logerr("Could not determine valid times for any file in the sequence.");
            return;
        }
        self.remember_recent("sequence", original);
        let entries: Vec<SeqEntry> = seq
            .iter()
            .map(|it| SeqEntry {
                path: kept[it.file_index].clone(),
                ts_index: it.ts_index,
                label: it.label.clone(),
                hhmm: it.valid.hhmm(),
                time_iso: Some(it.valid.iso_utc()),
            })
            .collect();
        self.timesteps = entries
            .iter()
            .enumerate()
            .map(|(i, e)| Timestep {
                label: e.label.clone(),
                hhmm: e.hhmm,
                ts_index: Some(e.ts_index),
                time_iso: e.time_iso.clone(),
                file: String::new(),
                seq_index: Some(i),
            })
            .collect();
        self.selected_ts = 0;
        self.rendered = None;
        self.loop_state = None;
        let cache_dir = ingest::default_cache_dir();
        // ONE run_id for the whole sequence -> one multi-frame store run.
        let run_id = sequence_run_id(&kept);
        let n = entries.len();
        let first = self
            .timesteps
            .first()
            .map(|t| t.label.clone())
            .unwrap_or_default();
        let last = self
            .timesteps
            .last()
            .map(|t| t.label.clone())
            .unwrap_or_default();
        self.source = Some(Source::Sequence {
            entries,
            cache_dir,
            run_id,
            needs_confirm,
            confirmed: false,
            total_bytes,
        });
        if needs_confirm {
            self.logline(format!(
                "Sequence: {n} timesteps ({first} .. {last}), {:.2} GB total. Confirm to batch render.",
                total_bytes as f64 / (1u64 << 30) as f64
            ));
        } else {
            self.logline(format!(
                "Sequence opened: {n} timesteps ({first} .. {last}). Render sequence to prerender the loop."
            ));
        }
    }

    fn render_source_ready(&self) -> bool {
        if self.busy || self.gpu.is_none() || self.timesteps.is_empty() {
            return false;
        }
        match &self.source {
            Some(Source::Wrfout {
                needs_confirm,
                confirmed,
                ..
            }) => !needs_confirm || *confirmed,
            Some(Source::Cached { .. }) => true,
            Some(Source::Sequence {
                needs_confirm,
                confirmed,
                ..
            }) => !needs_confirm || *confirmed,
            None => false,
        }
    }

    fn can_render(&self) -> bool {
        if !self.render_source_ready() {
            return false;
        }
        // Perspective (3-D) is VISIBLE-mode only in v1 (the engine has no
        // perspective IR march); the Render button carries the "(needs Mode:
        // Visible)" hint.
        if self.view == StudioView::Perspective && self.render_mode != RenderMode::Visible {
            return false;
        }
        true
    }

    /// Actual blockers for the dedicated GPU action. Current Mode/View/cloud controls
    /// are intentionally absent: the action uses a temporary compatible preview copy.
    fn gpu_render_action_blockers(&self) -> Vec<String> {
        let mut reasons = Vec::new();
        if self.science_cloud_f16 {
            reasons.push(
                "ScienceCloudF16 is CPU-only: the GPU preview uploads CompactU8 texture codes directly and would discard the selected precision"
                    .to_string(),
            );
        }
        if self.render_intent == RenderIntent::SensorFastGray {
            reasons.push(
                "Sensor Fast Gray requires the CPU path so model cloud fraction and strict neutral highlights are preserved"
                    .to_string(),
            );
        }
        if self.gpu.is_none() {
            reasons.push(
                self.gpu_error
                    .clone()
                    .unwrap_or_else(|| "no wgpu GPU device is available".to_string()),
            );
        }
        if self.busy {
            reasons.push("another render is already in progress".to_string());
        }
        if self.source.is_none() {
            reasons.push("open a wrfout/GRIB2 file or cached run first".to_string());
        } else if self.timesteps.is_empty() {
            reasons.push("the open source has no renderable timestep".to_string());
        }
        match &self.source {
            Some(Source::Wrfout {
                needs_confirm: true,
                confirmed: false,
                ..
            })
            | Some(Source::Sequence {
                needs_confirm: true,
                confirmed: false,
                ..
            }) => reasons.push("confirm the large import first".to_string()),
            _ => {}
        }
        reasons
    }

    /// Whether a batch (loop) render can start: rendering is possible and there is more
    /// than one timestep to sweep (a single frame is just Render). Perspective view is
    /// excluded in v1 (a fixed-camera perspective loop is the queued follow-up).
    fn can_render_sequence(&self) -> bool {
        self.can_render() && self.timesteps.len() >= 2 && self.view != StudioView::Perspective
    }

    /// Build the render job for one timestep of the current source (single Render or one
    /// frame of a batch). `None` if there is no source. Sequence entries carry their own
    /// wrfout path + time index; the Sequence's shared run_id makes every rendered frame
    /// land in one store run.
    fn job_for_timestep(&self, ts: &Timestep) -> Option<JobKind> {
        match self.source.as_ref()? {
            Source::Wrfout {
                path,
                cache_dir,
                run_id,
                ..
            } => Some(JobKind::Wrfout {
                path: path.clone(),
                cache_dir: cache_dir.clone(),
                run_id: run_id.clone(),
                ts_index: ts.ts_index.unwrap_or(0),
            }),
            Source::Cached {
                cache_dir,
                run_id,
                manifest,
            } => Some(JobKind::Cached {
                brick_path: bricks::run_dir(cache_dir, run_id).join(&ts.file),
                params: manifest_params(manifest),
                run_id: run_id.clone(),
                time_iso: ts.time_iso.clone(),
                hhmm: ts.hhmm,
            }),
            Source::Sequence {
                entries,
                cache_dir,
                run_id,
                ..
            } => {
                let e = entries.get(ts.seq_index?)?;
                Some(JobKind::Wrfout {
                    path: e.path.clone(),
                    cache_dir: cache_dir.clone(),
                    run_id: run_id.clone(),
                    ts_index: e.ts_index,
                })
            }
        }
    }

    /// The full list of `(label, job)` for a batch (loop) render over EVERY timestep of
    /// the current source — a multi-time wrfout's time dimension, a cached run's bricks,
    /// or an opened sequence's files — in the picker's chronological order.
    fn build_all_jobs(&self) -> Vec<(String, JobKind)> {
        self.timesteps
            .iter()
            .filter_map(|t| self.job_for_timestep(t).map(|j| (t.label.clone(), j)))
            .collect()
    }

    fn start_render(&mut self, ctx: &egui::Context) {
        if !self.can_render() {
            return;
        }
        let atmo = self.capture_atmo();
        self.start_render_with_atmo(ctx, atmo);
    }

    /// One-click GPU cloud preview. It does not edit any persistent checkbox, picker,
    /// or quality value: only the captured per-render copy is brought into the reviewed
    /// Visible GPU envelope, and every difference is disclosed. Geostationary and
    /// Top-down retain their selected camera; Perspective falls back to geo.
    fn start_gpu_render(&mut self, ctx: &egui::Context) {
        let blockers = self.gpu_render_action_blockers();
        if !blockers.is_empty() {
            self.logerr(format!("GPU Render unavailable: {}.", blockers.join("; ")));
            return;
        }
        let mut atmo = self.capture_atmo();
        let changes = configure_one_click_gpu_preview(&mut atmo);
        if changes.is_empty() {
            self.logline(
                "GPU Render: current controls already fit the selected-view Visible GPU preview. \
                 Studio settings are unchanged; this preview is not store output.",
            );
        } else {
            self.logline(format!(
                "GPU Render temporary preview (Studio settings unchanged): {}. Stored/sequence \
                 output remains on the CPU quality path.",
                changes.summary()
            ));
        }
        self.start_render_with_atmo(ctx, atmo);
    }

    fn start_render_with_atmo(&mut self, ctx: &egui::Context, atmo: AtmoSettings) {
        let Some(ts) = self.timesteps.get(self.selected_ts).cloned() else {
            return;
        };
        let preset = self.preset;
        let resolution = self.resolution;
        // The parity request is one-shot: consumed by this render.
        self.parity_pending = false;
        if atmo.render_intent == RenderIntent::SensorFastGray {
            let changes = atmo
                .intent_adjustments
                .iter()
                .map(|a| a.label())
                .collect::<Vec<_>>()
                .join("; ");
            self.logline(format!(
                "Sensor Fast Gray ({}): {}. Limitations: {}",
                atmo.render_intent.observation_operator(),
                if changes.is_empty() {
                    "current controls already satisfy the strict operator"
                } else {
                    &changes
                },
                atmo.render_intent.limitations().join("; ")
            ));
        }
        if atmo.parity {
            self.parity = None;
            if !atmo.fractional_clouds {
                self.logline(
                    "GPU parity check: rendering BOTH paths (CPU reference + GPU pass)...",
                );
            }
        }
        if (atmo.gpu_clouds || atmo.parity) && atmo.fractional_clouds {
            self.logline(
                "GPU clouds: model cloud fraction/subcolumns are CPU-only; using the CPU \
                 composite. Turn off 'Use model cloud fraction' only for a legacy GPU preview.",
            );
        }
        if (atmo.gpu_clouds || atmo.parity) && !gpu_granulation_preview_compatible(atmo.granulation)
        {
            self.logline(
                "GPU clouds: granulation is CPU-only; using the CPU composite so the requested \
                 sub-grid detail is not ignored. Turn off Granulation only for a legacy GPU preview.",
            );
        }
        if (atmo.gpu_clouds || atmo.parity)
            && !gpu_topdown_stratiform_preview_compatible(
                atmo.view_mode == StudioView::TopDownMap,
                atmo.topdown_stratiform_regularization,
            )
        {
            self.logline(
                "GPU clouds: top-down stratiform reconstruction is CPU-only; using the CPU \
                composite so the requested reconstructed cloud field is not ignored.",
            );
        }
        if (atmo.gpu_clouds || atmo.parity)
            && !gpu_topdown_cloud_footprint_preview_compatible(
                atmo.view_mode == StudioView::TopDownMap,
                atmo.topdown_cloud_footprint,
            )
        {
            self.logline(
                "GPU clouds: top-down cloud footprint is CPU-only; using the CPU composite \
                 so the requested pre-tonemap residual filter is not ignored.",
            );
        }
        if (atmo.gpu_clouds || atmo.parity) && atmo.terrain_atmosphere {
            self.logline(
                "GPU clouds: terrain-height atmosphere requires the CPU path; disabling that \
                 physical correction re-enables the experimental GPU preview.",
            );
        }
        if (atmo.gpu_clouds || atmo.parity)
            && !gpu_cloud_tonemap_compatible(atmo.cloud_softclip, atmo.cloud_highlight_max)
        {
            self.logline(
                "GPU clouds: custom highlight knee/ceiling are CPU-only; using the CPU \
                 composite so the requested display calibration is not ignored.",
            );
        }
        if (atmo.gpu_clouds || atmo.parity)
            && matches!(
                atmo.cloud_multiscatter,
                CloudMultiscatterMode::DeltaFluxV1
                    | CloudMultiscatterMode::DeltaFluxV2
                    | CloudMultiscatterMode::DeltaFluxV3
            )
        {
            self.logline("GPU clouds: delta-flux transport is CPU-only; using the CPU composite.");
        }
        if atmo.clouds_enabled
            && atmo.render_mode.uses_visible_controls()
            && matches!(
                atmo.cloud_multiscatter,
                CloudMultiscatterMode::DeltaFluxV1
                    | CloudMultiscatterMode::DeltaFluxV2
                    | CloudMultiscatterMode::DeltaFluxV3
            )
        {
            self.logline(format!(
                "Cloud transport: {} (experimental CPU path).",
                atmo.cloud_multiscatter.slug()
            ));
        }
        if atmo.clouds_enabled
            && atmo.render_mode.uses_visible_controls()
            && !atmo.fractional_clouds
        {
            self.logline(
                "Legacy cloud coverage active: model cloud fraction disabled; every non-zero \
                 cloudy cell is horizontally full.",
            );
        }
        if atmo.clouds_enabled
            && atmo.render_mode.uses_visible_controls()
            && atmo.fractional_clouds
            && atmo.fractional_cloud_mode.is_deterministic()
        {
            self.logline(format!(
                "Fractional clouds: {} CPU reference ({} fixed-stratified shared-u marches, average linear radiance, one tonemap).",
                atmo.fractional_cloud_mode.slug(),
                atmo.fractional_cloud_mode
                    .deterministic_subcolumn_count()
                    .expect("deterministic mode count")
            ));
        }
        if atmo.clouds_enabled
            && atmo.render_mode.uses_visible_controls()
            && (atmo.cloud_optical_depth_scale - clouds::DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE).abs()
                > f32::EPSILON
        {
            self.logline(format!(
                "Visible cloud optical-depth scale {:.2} (shipped {:.2}; 1.00 = unscaled).",
                atmo.cloud_optical_depth_scale,
                clouds::DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE
            ));
        }
        let Some(job) = self.job_for_timestep(&ts) else {
            return;
        };

        let (tx, rx) = channel();
        self.worker_rx = Some(rx);
        self.busy = true;
        self.busy_since = Some(Instant::now());
        self.rendered = None;
        // A single Render shows its one frame — retire any prerendered loop view (its
        // store run is already persisted).
        self.loop_state = None;
        self.logline("Preparing render...");
        let ctx = ctx.clone();
        let cache = self.scene_cache.clone();
        std::thread::spawn(move || {
            simsat::platform::lower_worker_thread_priority();
            let t0 = Instant::now();
            match prepare_render(job, preset, resolution, atmo, &cache, &tx) {
                Ok(prep) => {
                    let _ = tx.send(WorkerMsg::Status(format!(
                        "Prepared in {} ms.",
                        t0.elapsed().as_millis()
                    )));
                    let _ = tx.send(WorkerMsg::Prepared(prep));
                }
                Err(e) => {
                    let _ = tx.send(WorkerMsg::Error(e));
                }
            }
            ctx.request_repaint();
        });
    }

    /// Turn a prepared frame into a displayable `(RenderedFrame, TextureHandle)` plus
    /// GPU-cloud info when that path rendered it: the clouds-ON / IR / top-down CPU
    /// RGBA is used directly; a `gpu_cloud` prep runs the EXPERIMENTAL GPU cloud pass;
    /// the clouds-OFF geostationary path runs the M2 GPU surface pass — all on the UI
    /// thread. Shared by the single render (`finish_prepared`) and the batch loop
    /// (`accept_batch_frame`; batches never carry `gpu_cloud`). Takes
    /// `prep.cloud_rgba`/`prep.gpu_cloud` out; leaves `prep.ir_bt` for the caller.
    fn render_prepared(
        &self,
        ctx: &egui::Context,
        prep: &mut PreparedRender,
    ) -> Result<(RenderedFrame, egui::TextureHandle, Option<GpuRenderInfo>), String> {
        let mut gpu_info = None;
        let rendered = if let Some(rgba) = prep.cloud_rgba.take() {
            RenderedFrame {
                width: prep.width,
                height: prep.height,
                rgba,
            }
        } else if let Some(gc) = prep.gpu_cloud.take() {
            // EXPERIMENTAL GPU cloud pass (the M5-GPU activation): sun-OD compute +
            // the clouds.wgsl march, offscreen + readback, on the UI thread (the M1
            // NOTE-5 pattern the surface pass already uses).
            let gpu = self
                .gpu
                .as_ref()
                .ok_or_else(|| "GPU unavailable; cannot render.".to_string())?;
            let inputs = CloudFrameInputs {
                surface: surface_inputs(prep),
                view_mode: gc.view_mode,
                ray_lut: &gc.ray_lut,
                vol_nx: gc.vol_nx,
                vol_ny: gc.vol_ny,
                vol_nz: gc.vol_nz,
                texture_a: &gc.texture_a,
                occ_dims: gc.occ_dims,
                occupancy: &gc.occupancy,
                ql: gc.ql,
                qp: gc.qp,
                z_min_m: gc.z_min_m,
                dz_m: gc.dz_m,
                r_top_m: gc.r_top_m,
                r_bottom_m: gc.r_bottom_m,
                voxel_pitch_m: gc.voxel_pitch_m,
                geo: gc.geo,
                march: gc.march,
                sun_od: gc.sun_od,
                froxel_dim: gc.froxel_dim,
                froxel_data: &gc.froxel_data,
                sh_rows: gc.sh_rows,
                sh_data: &gc.sh_data,
                scan_rect: gc.scan_rect,
            };
            let t0 = Instant::now();
            let frame = gpu.cloud_resources.render(&gpu.device, &gpu.queue, &inputs);
            let gpu_ms = t0.elapsed().as_millis() as u64;
            // Parity instrument: diff the GPU frame against the CPU reference (mean/
            // p95/max |delta| per channel + a heatmap texture the owner can view).
            let parity = gc.cpu_reference.as_ref().map(|cpu| {
                let stats = parity_stats(cpu, &frame.rgba);
                let heat = parity_heatmap_rgba(cpu, &frame.rgba);
                let color = egui::ColorImage::from_rgba_unmultiplied(
                    [frame.width as usize, frame.height as usize],
                    &heat,
                );
                let texture =
                    ctx.load_texture("simsat-parity-heatmap", color, egui::TextureOptions::LINEAR);
                ParityReport {
                    summary: stats.summary(),
                    texture,
                }
            });
            gpu_info = Some(GpuRenderInfo { gpu_ms, parity });
            frame
        } else {
            let gpu = self
                .gpu
                .as_ref()
                .ok_or_else(|| "GPU unavailable; cannot render.".to_string())?;
            let inputs = surface_inputs(prep);
            gpu.resources.render(&gpu.device, &gpu.queue, &inputs)
        };

        // Display image: force alpha opaque so space renders as black.
        let mut display = rendered.rgba.clone();
        for px in display.chunks_exact_mut(4) {
            px[3] = 255;
        }
        let color = egui::ColorImage::from_rgba_unmultiplied(
            [rendered.width as usize, rendered.height as usize],
            &display,
        );
        // LINEAR (not NEAREST): the frame renders at the WRF native resolution, so any
        // residual window magnification is smooth instead of hard blocky pixels.
        let texture = ctx.load_texture("simsat-frame", color, egui::TextureOptions::LINEAR);
        Ok((rendered, texture, gpu_info))
    }

    /// Keep the UI, worker capture, and GPU-compatibility gate on one land config.
    fn land_appearance_config(&self) -> LandAppearanceConfig {
        LandAppearanceConfig {
            sza_normalization: self.land_sza_normalization,
            sza_max_gain: self.land_sza_max_gain as f64,
            dark_toe: self.land_dark_toe,
            dark_toe_knee: self.land_dark_toe_knee as f64,
            dark_toe_gamma: self.land_dark_toe_gamma as f64,
            dark_toe_max_gain: self.land_dark_toe_max_gain as f64,
        }
    }

    fn surface_postlight_toe_config(&self) -> SurfacePostlightToeConfig {
        SurfacePostlightToeConfig {
            enabled: self.surface_postlight_toe,
            knee: self.surface_postlight_toe_knee as f64,
            gamma: self.surface_postlight_toe_gamma as f64,
            max_gain: self.surface_postlight_toe_max_gain as f64,
        }
    }

    fn twilight_surface_recovery_config(&self) -> TwilightSurfaceRecoveryConfig {
        TwilightSurfaceRecoveryConfig {
            enabled: self.twilight_surface_recovery,
            knee: self.twilight_surface_recovery_knee as f64,
            gamma: self.twilight_surface_recovery_gamma as f64,
            max_gain: self.twilight_surface_recovery_max_gain as f64,
        }
    }

    /// Snapshot the M2/M4/M5/M6 render controls into the worker-side `AtmoSettings`
    /// (shared by the single Render and the batch loop so every frame uses the current
    /// satellite/exposure/view/mode/enhancement settings).
    fn capture_atmo(&self) -> AtmoSettings {
        let mut atmo = AtmoSettings {
            render_intent: self.render_intent,
            geo_navigation: self.geo_navigation,
            intent_adjustments: Vec::new(),
            view_mode: self.view,
            orbit: pipeline::OrbitParams {
                az_deg: self.orbit_az_deg as f64,
                tilt_deg: self.orbit_tilt_deg as f64,
                range_km: self.orbit_range_km as f64,
                fov_deg: self.orbit_fov_deg as f64,
                width: self.persp_width as usize,
                height: self.persp_height as usize,
            },
            // Zoom-out / margin: the UI slider is a PERCENTAGE (0-100%); the internal render
            // param is a fraction (a future km-based UI is then a trivial swap).
            margin_frac: self.margin_pct as f64 / 100.0,
            render_mode: self.render_mode,
            ir_enhancement: self.ir_enhancement,
            thermal_sensor: self.thermal_sensor,
            instrument_footprint: self.instrument_footprint,
            aod: self.aod as f64,
            rh_swelling: self.rh_swelling,
            atmosphere_correction: self.atmosphere_correction,
            terrain_atmosphere: self.terrain_atmosphere,
            land_appearance: self.land_appearance_config(),
            surface_postlight_toe: self.surface_postlight_toe_config(),
            twilight_surface_recovery: self.twilight_surface_recovery_config(),
            output_transform: self.output_transform,
            clouds_enabled: self.clouds_enabled,
            fractional_clouds: self.fractional_clouds,
            fractional_cloud_mode: self.fractional_cloud_mode,
            cloud_multiscatter: studio_cloud_multiscatter_mode(
                self.multiscatter,
                self.delta_flux_clouds,
                self.delta_flux_v2_clouds,
                self.delta_flux_v3_clouds,
            ),
            cloud_optical_depth_scale: self.cloud_optical_depth_scale,
            storage_profile: if self.science_cloud_f16 {
                StorageProfile::ScienceCloudF16
            } else {
                StorageProfile::CompactU8
            },
            cloud_optics: if self.hrrr_thompson_native_cloud_optics
                && self.render_mode == RenderMode::Visible
            {
                CloudOpticsMode::HrrrThompsonNative
            } else if self.nssl_native_cloud_optics && self.render_mode == RenderMode::Visible {
                CloudOpticsMode::NsslNative
            } else {
                CloudOpticsMode::Fixed
            },
            feather_exposed_domain_edges: self.feather_exposed_domain_edges,
            beer_powder: self.beer_powder,
            granulation: self.granulation,
            topdown_stratiform_regularization: self.topdown_stratiform_regularization,
            topdown_cloud_footprint: self.topdown_cloud_footprint,
            topdown_shadow_antialias: self.topdown_shadow_antialias,
            step_quality: self.step_quality,
            gpu_clouds: self.gpu_clouds,
            parity: self.parity_pending,
            one_click_gpu_render: false,
            gpu_preview_adjustments: GpuPreviewAdjustments::default(),
            exposure: self.exposure as f64,
            ground_gain: self.ground_gain as f64,
            cloud_softclip: self.cloud_softclip as f64,
            cloud_highlight_max: self.cloud_highlight_max as f64,
            // Fake-sun what-if override: Some((elev_deg, az_deg)) when on, else the file's
            // real solar geometry. Uniform sun direction across the frame (sun at infinity).
            sun_override: if self.sun_override {
                Some((self.sun_override_elev as f64, self.sun_override_az as f64))
            } else {
                None
            },
            // Seasonal ground (M7): 0 = auto day-of-year blend, 1..=12 = forced month.
            bm_month_override: (1..=12)
                .contains(&self.bm_month_override)
                .then_some(self.bm_month_override),
            bm_allow_download: self.bm_allow_download,
        };
        configure_render_intent(&mut atmo);
        atmo
    }

    fn finish_prepared(&mut self, ctx: &egui::Context, mut prep: PreparedRender) {
        // M4/M6: when clouds are enabled OR in IR mode the frame is rendered on the CPU
        // worker (cloud composite / IR enhancement) and displayed/stored directly; only
        // the clear-sky visible (clouds-off) path uses the M2 GPU surface pass.
        let ir_bt = prep.ir_bt.take();
        let derived = prep.derived.take();
        let is_ir = ir_bt.is_some();
        let ir_enhancement = prep.ir_enhancement;
        let ir_band = prep.ir_band;
        let clouds_on = rendered_clouds_on(prep.clouds_enabled, is_ir, derived.is_some());
        let (rendered, texture, gpu_info) = match self.render_prepared(ctx, &mut prep) {
            Ok(v) => v,
            Err(e) => {
                self.logerr(e);
                self.busy = false;
                self.busy_since = None;
                return;
            }
        };
        // GPU-cloud preview bookkeeping: frame time for the status line, and the
        // parity report (numbers logged + heatmap kept for the drawer).
        let gpu_ms = gpu_info.as_ref().map(|i| i.gpu_ms);
        if let Some(info) = gpu_info
            && let Some(report) = info.parity
        {
            self.logline(format!("GPU parity: {}", report.summary));
            self.parity = Some(report);
        }

        // IR BT stats for the status line (coldest cloud-top vs warmest ground).
        let ir_stats = ir_bt.as_ref().map(|bt| ir::ir_frame_stats(bt));
        // Derived-field value range for the status line (before `derived` moves into state).
        let derived_summary = derived
            .as_ref()
            .map(|(field, values)| (*field, derived::field_stats(values)));
        let bm_status = prep.bm_status.clone();
        self.rendered = Some(RenderedState {
            texture,
            rendered,
            lat: prep.lat,
            lon: prep.lon,
            sector: prep.sector,
            satellite: prep.satellite,
            view_mode: prep.view_mode,
            year: prep.year,
            month: prep.month,
            day: prep.day,
            hhmm: prep.hhmm,
            bm_status: bm_status.clone(),
            season_line: prep.season_line.clone(),
            center_sun_elev: prep.center_sun_elev,
            sun_override: prep.sun_override,
            resolution: prep.resolution,
            res_clamped: prep.res_clamped,
            ir_bt,
            ir_enhancement,
            ir_band,
            instrument_footprint: prep.instrument_footprint,
            derived: derived_summary,
            render_mode: prep.render_mode,
            gpu_ms,
            one_click_gpu_render: prep.one_click_gpu_render,
            gpu_preview_adjustments: prep.gpu_preview_adjustments,
        });
        // A new render resets the display viewport to fit-to-window (no leftover zoom/pan).
        self.view_zoom = 1.0;
        self.view_pan = egui::Vec2::ZERO;
        self.busy = false;
        self.busy_since = None;
        // A successful render is the one non-explicit event that clears the sticky
        // error banner (the app demonstrably recovered).
        self.log.note_render_success();
        if let Some((field, s)) = derived_summary {
            // A derived scalar-field map: report the RAW value range (the RAW array is the
            // primary deliverable — the binding; the studio shows the basic colormap).
            self.logline(format!(
                "Rendered {} map {}x{} {}{} ({} in-domain values; min {:.2} max {:.2} \
                 median {:.2} {}).",
                field.label(),
                prep.width,
                prep.height,
                prep.resolution.label(),
                if prep.res_clamped { " [clamped]" } else { "" },
                s.finite,
                s.min,
                s.max,
                s.median,
                if field.units().is_empty() {
                    "(dimensionless)"
                } else {
                    field.units()
                },
            ));
        } else if let Some(stats) = ir_stats {
            // IR/WV is thermal: report the BT range (cold cloud/moisture tops vs warm
            // ground) and the enhancement instead of the sun/PW/Blue-Marble fields.
            self.logline(format!(
                "Rendered {} {}x{} {}{} ({:.0}% in-domain, {} enhancement, footprint {}; \
                 cold {:.1} K, warm {:.1} K, median {:.1} K).",
                band_display(ir_band),
                prep.width,
                prep.height,
                prep.resolution.label(),
                if prep.res_clamped { " [clamped]" } else { "" },
                prep.on_earth_frac * 100.0,
                ir_enhancement.label(),
                prep.instrument_footprint.slug(),
                stats.min_bt,
                stats.max_bt,
                stats.median_bt,
            ));
        } else {
            let gpu_note = match gpu_ms {
                Some(ms) => {
                    format!(" [GPU clouds {ms} ms — experimental preview; store stays CPU]")
                }
                None => String::new(),
            };
            self.logline(format!(
                "Rendered {}x{} {}{} ({:.0}% on-earth, sun {:.1} deg{}, PW x{:.2}, clouds {}, {}){}.",
                prep.width,
                prep.height,
                prep.resolution.label(),
                if prep.res_clamped { " [clamped]" } else { "" },
                prep.on_earth_frac * 100.0,
                prep.center_sun_elev,
                if prep.sun_override {
                    " OVERRIDE (what-if)"
                } else {
                    ""
                },
                prep.pw_ratio,
                if clouds_on { "on" } else { "off" },
                if prep.season_line.is_empty() {
                    bm_status.chip_label()
                } else {
                    prep.season_line.clone()
                },
                gpu_note
            ));
        }
    }

    /// Re-colour the currently-rendered IR BT plane with the current enhancement
    /// WITHOUT re-marching (cheap — a per-pixel table lookup). Called when the IR
    /// enhancement picker changes so the studio recolours live, mirroring BowEcho's
    /// live IR re-enhancement over the same true-Kelvin frame.
    fn reenhance_ir(&mut self, ctx: &egui::Context) {
        let target = self.ir_enhancement;
        // Recolour inside a scoped borrow of `self.rendered` so `logline` (which needs
        // `&mut self`) can run after the borrow ends. Returns whether a recolour happened.
        let changed = {
            let Some(state) = self.rendered.as_mut() else {
                return;
            };
            if state.ir_bt.is_none() || state.ir_enhancement == target {
                return;
            }
            let bt = state.ir_bt.as_ref().unwrap();
            let rgba = render_ir_rgba(bt, state.ir_band, target);
            // Rebuild the display texture (force opaque so out-of-domain shows black).
            let mut display = rgba.clone();
            for px in display.chunks_exact_mut(4) {
                px[3] = 255;
            }
            let color = egui::ColorImage::from_rgba_unmultiplied(
                [
                    state.rendered.width as usize,
                    state.rendered.height as usize,
                ],
                &display,
            );
            state.rendered.rgba = rgba;
            state.ir_enhancement = target;
            state.texture = ctx.load_texture("simsat-frame", color, egui::TextureOptions::LINEAR);
            true
        };
        if changed {
            let band = self.rendered.as_ref().map(|s| s.ir_band).unwrap_or(13);
            self.logline(format!(
                "Re-enhanced {} -> {} (no re-march).",
                band_display(band),
                target.label()
            ));
        }
    }

    fn write_to_store(&mut self) {
        let Some(state) = &self.rendered else {
            return;
        };
        if state.gpu_ms.is_some() {
            // Stored-frame quality/provenance stays CPU (the tested shipping path);
            // the GPU frame is an experimental preview. The button is also disabled,
            // this is defense in depth.
            self.logerr(
                "GPU-clouds preview frames are not written to the store (the stored \
                 path stays CPU). Turn off GPU clouds and Render again to write.",
            );
            return;
        }
        if state.view_mode == StudioView::Perspective {
            // No sat-store contract for a perspective frame (a picture, not a
            // georegistered map). The button is also disabled; defense in depth.
            self.logerr(
                "Perspective (3-D) frames are not written to the sat store (no store \
                 contract for a free-camera picture). Use Save PNG..., or switch View.",
            );
            return;
        }
        // IR mode writes the true-Kelvin BT plane as a SINGLE-BAND band-13 frame
        // (BowEcho re-enhances it live); visible mode writes the three baked rgb planes.
        let written = store_write_frame(
            &self.store_root,
            &state.rendered,
            state.ir_bt.as_ref(),
            state.ir_band,
            &state.lat,
            &state.lon,
            &state.sector,
            state.satellite,
            state.year,
            state.month,
            state.day,
            state.hhmm,
        );
        match written {
            Ok(w) => {
                let msg = format!(
                    "Wrote {}/{} t{:04} ({} bytes){}. Point BowEcho's sat store at: {}",
                    w.model,
                    w.run,
                    w.hhmm,
                    w.bytes,
                    if w.created_run { " [new run]" } else { "" },
                    self.store_root.display()
                );
                self.last_written = Some(w.run_dir.display().to_string());
                self.logline(msg);
            }
            Err(e) => self.logerr(format!("Store write failed: {e}")),
        }
    }

    // ── batch (loop) render: prerender a whole sequence, then play it ────────────

    /// Start a batch (loop) render over EVERY timestep of the current source. One
    /// below-normal worker thread ingests + renders each frame in chronological order,
    /// streaming a `BatchFrame` back per frame (PROGRESSIVE) and checking a cancel flag
    /// at each frame boundary (CANCELABLE). The UI thread finishes each frame (GPU pass
    /// only for the clouds-off geo path), writes it into the ONE store run, and retains
    /// it as a `LoopFrame` for instant playback (up to `frame_cap`). Reuses the exact
    /// single-frame `prepare_render` per timestep — no re-implemented render path.
    fn start_batch_render(&mut self, ctx: &egui::Context) {
        if !self.can_render_sequence() || self.view == StudioView::Perspective {
            return;
        }
        let jobs = self.build_all_jobs();
        if jobs.len() < 2 {
            return;
        }
        let total = jobs.len();
        let preset = self.preset;
        let resolution = self.resolution;
        let mut atmo = self.capture_atmo();
        // Sequence/batch renders ALWAYS take the CPU path (stored-frame quality and
        // provenance stay CPU); the experimental GPU toggle applies only to the
        // single displayed frame.
        if atmo.gpu_clouds {
            self.logline(
                "Sequence renders always use the CPU path; GPU clouds ignored for the batch.",
            );
        }
        if atmo.clouds_enabled
            && atmo.render_mode.uses_visible_controls()
            && (atmo.cloud_optical_depth_scale - clouds::DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE).abs()
                > f32::EPSILON
        {
            self.logline(format!(
                "Batch visible cloud optical-depth scale {:.2} (shipped {:.2}; \
                 1.00 = unscaled).",
                atmo.cloud_optical_depth_scale,
                clouds::DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE
            ));
        }
        atmo.gpu_clouds = false;
        atmo.parity = false;
        let cancel = Arc::new(AtomicBool::new(false));
        self.batch = Some(BatchState {
            total,
            done: 0,
            errors: 0,
            cancel: cancel.clone(),
            total_frame_ms: 0,
        });
        let mut ls = LoopState::new();
        ls.is_ir = self.render_mode.is_thermal();
        ls.ir_band = self.render_mode.ir_band();
        self.loop_state = Some(ls);
        self.rendered = None;
        let (tx, rx) = channel();
        self.worker_rx = Some(rx);
        self.busy = true;
        self.busy_since = Some(Instant::now());
        self.logline(format!(
            "Batch rendering {total} frames (prerender-then-play)..."
        ));
        let ctx = ctx.clone();
        let cache = self.scene_cache.clone();
        std::thread::spawn(move || {
            simsat::platform::lower_worker_thread_priority();
            let mut rendered = 0usize;
            let mut cancelled = false;
            for (i, (label, job)) in jobs.into_iter().enumerate() {
                if cancel.load(Ordering::Relaxed) {
                    cancelled = true;
                    break;
                }
                let _ = tx.send(WorkerMsg::Status(format!(
                    "Rendering frame {}/{}: {label}",
                    i + 1,
                    total
                )));
                let t0 = Instant::now();
                match prepare_render(job, preset, resolution, atmo.clone(), &cache, &tx) {
                    Ok(prep) => {
                        rendered += 1;
                        if tx
                            .send(WorkerMsg::BatchFrame {
                                index: i,
                                total,
                                prep,
                                prep_ms: t0.elapsed().as_millis() as u64,
                            })
                            .is_err()
                        {
                            break; // UI closed
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(WorkerMsg::BatchError {
                            index: i,
                            message: e,
                        });
                    }
                }
                ctx.request_repaint();
            }
            let _ = tx.send(WorkerMsg::BatchDone {
                rendered,
                cancelled,
            });
            ctx.request_repaint();
        });
    }

    /// Request cancellation of an in-flight batch render (takes effect at the next frame
    /// boundary — one frame may already be marching).
    fn cancel_batch(&mut self) {
        if let Some(b) = &self.batch {
            b.cancel.store(true, Ordering::Relaxed);
            self.logline("Cancelling batch render (finishing the current frame)...");
        }
    }

    /// Finish one batch frame on the UI thread: build its texture (GPU surface pass only
    /// for the clouds-off geostationary path), write it into the ONE multi-frame store
    /// run, and retain it as a `LoopFrame` for playback (bounded by `frame_cap`).
    fn accept_batch_frame(
        &mut self,
        ctx: &egui::Context,
        index: usize,
        total: usize,
        mut prep: PreparedRender,
        prep_ms: u64,
    ) {
        let started = Instant::now();
        let ir_bt = prep.ir_bt.take();
        let ir_band = prep.ir_band;
        // Batch frames never carry a GPU-cloud prep (start_batch_render forces the
        // CPU path), so the GPU info leg is always None here.
        let (rendered, texture, _gpu_info) = match self.render_prepared(ctx, &mut prep) {
            Ok(v) => v,
            Err(e) => {
                self.logerr(format!("Frame {}/{} render failed: {e}", index + 1, total));
                if let Some(b) = &mut self.batch {
                    b.errors += 1;
                }
                return;
            }
        };
        // Write this frame into the ONE multi-frame store run (visible or IR/WV).
        let written = store_write_frame(
            &self.store_root,
            &rendered,
            ir_bt.as_ref(),
            ir_band,
            &prep.lat,
            &prep.lon,
            &prep.sector,
            prep.satellite,
            prep.year,
            prep.month,
            prep.day,
            prep.hhmm,
        );
        let store_run = match &written {
            Ok(w) => {
                self.last_written = Some(w.run_dir.display().to_string());
                Some(w.run.clone())
            }
            Err(e) => {
                self.logerr(format!(
                    "Frame {}/{} store write failed: {e}",
                    index + 1,
                    total
                ));
                None
            }
        };
        let summary = if let Some(bt) = &ir_bt {
            let s = ir::ir_frame_stats(bt);
            format!(
                "{}, cold {:.0} K / warm {:.0} K",
                band_display(ir_band),
                s.min_bt,
                s.max_bt
            )
        } else {
            format!("sun {:.1} deg", prep.center_sun_elev)
        };
        let frame = LoopFrame {
            texture,
            width: rendered.width,
            height: rendered.height,
            label: frame_time_label(prep.year, prep.month, prep.day, prep.hhmm),
            summary,
        };
        let cap = self.frame_cap.max(1);
        if let Some(ls) = &mut self.loop_state {
            ls.total_rendered += 1;
            if let Some(run) = store_run {
                ls.store_run = Some(run);
            }
            if ls.frames.len() < cap {
                ls.frames.push(frame);
                // Show the newest frame as it arrives (live preview of the batch).
                ls.current = ls.frames.len() - 1;
            } else {
                // Over the retention cap: the store run still has every frame.
                ls.capped = true;
            }
        }
        let finish_ms = started.elapsed().as_millis() as u64;
        // Per-frame throughput log (the owner sees the scene-cache speedup here):
        // worker-side prepare (ingest/decode/LUTs/march) + UI-side finish (texture +
        // store write). The timeline's ~ms/frame average includes BOTH.
        self.log.info(format!(
            "Frame {}/{}: prepared {prep_ms} ms + finished {finish_ms} ms.",
            index + 1,
            total
        ));
        if let Some(b) = &mut self.batch {
            b.done = index + 1;
            b.total_frame_ms += prep_ms + finish_ms;
        }
    }

    /// Wrap up a finished/cancelled batch: report counts + per-frame wall time, arm the
    /// timeline, and start looping playback.
    fn finish_batch(&mut self, rendered: usize, cancelled: bool) {
        self.busy = false;
        self.busy_since = None;
        self.worker_rx = None;
        let (retained, capped, run) = self
            .loop_state
            .as_ref()
            .map(|ls| (ls.frames.len(), ls.capped, ls.store_run.clone()))
            .unwrap_or((0, false, None));
        let (avg_ms, errors) = self
            .batch
            .as_ref()
            .map(|b| (b.total_frame_ms / b.done.max(1) as u64, b.errors))
            .unwrap_or((0, 0));
        self.batch = None;
        if let Some(ls) = &mut self.loop_state {
            ls.current = 0;
            ls.playing = ls.frames.len() >= 2;
            ls.accumulator = 0.0;
        }
        let run_note = run
            .as_ref()
            .map(|r| format!(" Store run: simsat/{r}."))
            .unwrap_or_default();
        let cap_note = if capped {
            format!(
                " In-memory playback capped at {retained} frames; the full run is in the store."
            )
        } else {
            String::new()
        };
        let err_note = if errors > 0 {
            format!(" {errors} frame(s) failed.")
        } else {
            String::new()
        };
        if cancelled {
            self.logline(format!(
                "Batch cancelled: {rendered} rendered, {retained} retained, ~{avg_ms} ms/frame.{run_note}{cap_note}{err_note}"
            ));
        } else {
            self.logline(format!(
                "Batch complete: {rendered} frames, {retained} retained for playback, ~{avg_ms} ms/frame.{run_note}{cap_note}{err_note} Scrub/play/loop the timeline."
            ));
        }
        // A fully-clean batch counts as a successful render (clears the sticky error).
        if !cancelled && errors == 0 && rendered > 0 {
            self.log.note_render_success();
        }
    }

    /// Advance the loop play head for this UI frame using the pure fps/loop math, given
    /// the wall-clock `dt` since the last frame. Returns whether a repaint is wanted
    /// (i.e. playback is active) so `ui` can keep the animation running.
    fn tick_playback(&mut self, dt: f32) -> bool {
        let fps = self.play_fps;
        let Some(ls) = &mut self.loop_state else {
            return false;
        };
        if !ls.playing || ls.frames.len() < 2 {
            ls.accumulator = 0.0;
            return false;
        }
        let (steps, acc) = pipeline::fps_frame_step(ls.accumulator, dt, fps);
        ls.accumulator = acc;
        if steps > 0 {
            let (next, stopped) =
                pipeline::advance_index(ls.current, steps, ls.frames.len(), ls.looping);
            ls.current = next;
            if stopped {
                ls.playing = false;
            }
        }
        ls.playing
    }

    /// A short human label for the currently-open source, shown beside the Open menu in the
    /// top strip (the file name for a wrfout, or the run id for a cached run / sequence).
    fn source_display_name(&self) -> String {
        match &self.source {
            Some(Source::Wrfout { path, .. }) => path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("wrfout")
                .to_string(),
            Some(Source::Cached { run_id, .. }) => format!("cached: {run_id}"),
            Some(Source::Sequence {
                entries, run_id, ..
            }) => format!("sequence: {run_id} ({} steps)", entries.len()),
            None => "(no file open)".to_string(),
        }
    }

    /// Stable two-row toolbar. Row one keeps Open/source at left and every action pinned
    /// at right; row two uses fixed responsive selector widths. There is deliberately no
    /// horizontal ScrollArea, so keyboard focus can never retain an offset that hides Open.
    fn top_strip(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let can_render = self.can_render();
        let can_seq = self.can_render_sequence();
        let gpu_action_blockers = self.gpu_render_action_blockers();
        let can_gpu_render = gpu_action_blockers.is_empty();
        let gpu_action_disabled = format!(
            "GPU Render unavailable: {}.",
            gpu_action_blockers.join("; ")
        );
        // A GPU-clouds preview frame is never written to the store (the stored path
        // stays CPU for quality/provenance) — the button disables with a tooltip.
        // A Perspective frame has NO store contract (a picture, not a map) — same
        // disabled-with-hint pattern.
        let gpu_preview = self.rendered.as_ref().is_some_and(|s| s.gpu_ms.is_some());
        let persp_frame = self
            .rendered
            .as_ref()
            .is_some_and(|s| s.view_mode == StudioView::Perspective);
        let persp_view = self.view == StudioView::Perspective;
        // Perspective is Visible-mode only in v1: the Render button greys with the
        // "(needs Mode: Visible)" hint (the GPU-cluster discoverability pattern).
        let persp_needs_visible = persp_view && self.render_mode != RenderMode::Visible;
        let can_write = self.rendered.is_some() && !self.busy && !gpu_preview && !persp_frame;
        let batch_active = self.batch.is_some();
        let busy = self.busy;

        ui.horizontal(|ui| {
            // The right cluster claims the right edge regardless of the left content width: a
            // right-to-left layout adds widgets from the right, so the FIRST added is the
            // rightmost. Its final child is a left-to-right sub-layout that fills the remaining
            // width with the left content (Open / name / pickers).
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if busy {
                    ui.spinner();
                }
                if ui
                    .add_enabled(can_write, egui::Button::new("Write store"))
                    .on_hover_text(if gpu_preview {
                        "The displayed frame is a GPU-clouds preview (experimental); \
                         stored frames always come from the CPU path. Turn off GPU \
                         clouds and Render again to write."
                    } else if persp_frame {
                        "Perspective (3-D) frames have no sat-store contract (a \
                         free-camera picture, not a georegistered map). Use Save \
                         PNG..., or switch View to Geostationary / Top-down."
                    } else {
                        "Write the current frame into the sat store so BowEcho can play it."
                    })
                    .clicked()
                {
                    self.write_to_store();
                }
                if ui
                    .add_enabled(can_gpu_render, egui::Button::new("GPU Render"))
                    .on_hover_text(
                        "No setup required: one-click fast cloud preview. It retains Geostationary or Top-down, \
                         temporarily uses Visible + Clouds on, and substitutes only the CPU-only controls \
                         needed by the current WGSL pass. Studio settings are unchanged; every \
                         difference is shown, and stored/sequence output stays on the CPU \
                         quality path.",
                    )
                    .on_disabled_hover_text(gpu_action_disabled)
                    .clicked()
                {
                    self.start_gpu_render(ctx);
                }
                if ui
                    .add_enabled(can_render, egui::Button::new("Render"))
                    .on_disabled_hover_text(if persp_needs_visible {
                        "Perspective (3-D) renders the Visible product only in v1 \
                         (the engine has no perspective IR march). Set Mode: Visible."
                    } else {
                        "Open a source (and confirm a large import) to render."
                    })
                    .clicked()
                {
                    self.start_render(ctx);
                }
                if persp_needs_visible {
                    ui.label(egui::RichText::new("(needs Mode: Visible)").weak());
                }
                if batch_active {
                    if ui
                        .button("Cancel")
                        .on_hover_text("Stop the batch render at the next frame boundary.")
                        .clicked()
                    {
                        self.cancel_batch();
                    }
                } else if ui
                    .add_enabled(can_seq, egui::Button::new("Render sequence"))
                    .on_hover_text(
                        "Batch-render EVERY timestep into a playable loop (prerender-then-play): \
                         each frame renders on the below-normal worker, is written to the store, \
                         and is retained for instant scrub/play. Progressive + cancelable.",
                    )
                    .on_disabled_hover_text(if persp_view {
                        "Sequence rendering is not available in Perspective (3-D) view \
                         in v1 (a fixed-camera perspective loop is a queued follow-up). \
                         Switch View to Geostationary or Top-down."
                    } else {
                        "Open a multi-timestep source (sequence / multi-time wrfout) to \
                         batch-render a loop."
                    })
                    .clicked()
                {
                    self.start_batch_render(ctx);
                }
                let can_export = self.rendered.is_some() && !busy && !self.export_busy;
                if ui
                    .add_enabled(can_export, egui::Button::new("Save PNG..."))
                    .on_hover_text(
                        "Export the currently-displayed frame as an RGB PNG (space renders \
                         black). The encode runs on a background thread.",
                    )
                    .clicked()
                {
                    self.start_export(ctx);
                }
                ui.separator();

                // Open and the source label use only the remaining width. The source
                // label truncates visually but exposes the full identity on hover.
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    self.top_source_row_left(ui);
                });
            });
        });
        self.top_selector_row(ui);
    }

    /// Row-one left cluster. The right-pinned action buttons have already claimed
    /// their width before this is called.
    fn top_source_row_left(&mut self, ui: &mut egui::Ui) {
        let source_name = self.source_display_name();
        ui.horizontal(|ui| {
            ui.menu_button("Open", |ui| {
                if ui.button("Open wrfout / GRIB2...").clicked() {
                    ui.close();
                    self.dialog_open_wrfout();
                }
                if ui.button("Open cached run.json...").clicked() {
                    ui.close();
                    self.dialog_open_cached();
                }
                ui.separator();
                if ui
                    .button("Open sequence (folder)...")
                    .on_hover_text(
                        "Pick a DIRECTORY of wrfout files (e.g. the Enderlin folder). \
                                 They are ordered by valid time into a loop you batch render, \
                                 then scrub/play.",
                    )
                    .clicked()
                {
                    ui.close();
                    self.dialog_open_sequence_folder();
                }
                if ui
                    .button("Open sequence (files)...")
                    .on_hover_text("Or multi-select the wrfout files for the sequence.")
                    .clicked()
                {
                    ui.close();
                    self.dialog_open_sequence_files();
                }
                // Recent open actions (newest first, capped, pruned on load).
                if !self.recent.is_empty() {
                    ui.separator();
                    ui.label(egui::RichText::new("Recent").weak());
                    let entries = self.recent.clone();
                    for entry in &entries {
                        if ui.button(entry.label()).clicked() {
                            ui.close();
                            self.reopen_recent(entry);
                        }
                    }
                }
            });
            let source_width = ui.available_width().max(24.0);
            ui.add_sized(
                [source_width, ui.spacing().interact_size.y],
                egui::Label::new(egui::RichText::new(&source_name).strong()).truncate(),
            )
            .on_hover_text(source_name);
        });
    }

    /// Row-two product/camera selectors. At the 900 px smoke-test width compact
    /// selected labels fit in one deterministic row; menu choices retain full labels.
    fn top_selector_row(&mut self, ui: &mut egui::Ui) {
        let toolbar = toolbar_layout(ui.available_width());
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;

                    // Product mode (Visible / GeoColor / Sandwich / IR / WV / Derived); the
                    // context-driven drawer below adapts its groups to the selection.
                    ui.label("Mode:");
                    let previous_render_mode = self.render_mode;
                    egui::ComboBox::from_id_salt("mode")
                        .width(toolbar.mode_width)
                        .selected_text(self.render_mode.label())
                        .truncate()
                        .show_ui(ui, |ui| {
                            for m in RenderMode::ALL {
                                ui.selectable_value(&mut self.render_mode, m, m.label());
                            }
                        })
                        .response
                        .on_hover_text(
                            "Visible = physically-based Blue Marble + clouds + sun. GeoColor \
                             Style = SimSat Day/Night Color (broad-RGB by day, colored band-13 \
                             IR by night, crossfaded across the terminator); it is not yet \
                             sensor-derived ABI GeoColor. Sandwich = the true-color \
                             visible with color-enhanced band-13 IR overlaid on the cold cloud \
                             tops (the classic severe-convection view; a daytime product). IR = \
                             synthetic band 13 (10.3 um) window BT. Water Vapor = ABI bands \
                             8/9/10 (6.2/6.9/7.3 um) upper/mid/lower moisture BT. Derived = a \
                             per-column scalar map. IR / WV / Derived are thermal or column \
                             products (day AND night).",
                        );
                    if self.render_mode != previous_render_mode {
                        let enhancement_changed =
                            apply_product_transition_enhancement_default(
                                previous_render_mode,
                                self.render_mode,
                                &mut self.ir_enhancement,
                            );
                        let cleared = clear_incompatible_instrument_footprint(
                            self.render_mode,
                            &mut self.instrument_footprint,
                        );
                        if enhancement_changed.is_some() || cleared.is_some() {
                            self.save_settings_now();
                        }
                        if let Some(enhancement) = enhancement_changed {
                            self.logline(format!(
                                "Selected {} for {}. This product default applies only on an explicit Mode change.",
                                enhancement.label(),
                                self.render_mode.label()
                            ));
                        }
                        if let Some(cleared) = cleared {
                            self.logline(format!(
                                "Turned off {} because Mode {} has no compatible Band 13 instrument stage. The hidden setting was cleared and saved; Render is ready.",
                                cleared.label(),
                                self.render_mode.label()
                            ));
                        }
                    }
                    ui.label("Intent:");
                    egui::ComboBox::from_id_salt("render-intent")
                        .width(toolbar.intent_width)
                        .selected_text(self.render_intent.label())
                        .truncate()
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.render_intent,
                                RenderIntent::Display,
                                "Display",
                            );
                            ui.selectable_value(
                                &mut self.render_intent,
                                RenderIntent::SensorFastGray,
                                "Sensor Fast Gray",
                            );
                        })
                        .response
                        .on_hover_text(
                            "Display preserves the shipped reviewed appearance. Sensor Fast Gray \
                             uses simsat-fast-gray-v1: unscaled cloud extinction and neutral \
                             display shaping on a temporary render copy, with every adjustment \
                             logged. It is not yet an SRF-integrated ABI/AHI channel simulator.",
                        );
                    ui.label("Sat:");
                    // The satellite preset drives the geostationary scan camera; in
                    // Perspective view the orbit camera IS the view, so grey it out.
                    ui.add_enabled_ui(self.view != StudioView::Perspective, |ui| {
                        egui::ComboBox::from_id_salt("sat")
                            .width(toolbar.sat_width)
                            .selected_text(self.preset.label())
                            .truncate()
                            .show_ui(ui, |ui| {
                                for p in SatellitePreset::ALL {
                                    ui.selectable_value(&mut self.preset, p, p.label());
                                }
                            })
                            .response
                            .on_disabled_hover_text(
                                "Not used in Perspective (3-D) view — the orbit camera \
                                 (Camera group) defines the view.",
                            );
                    });
                    if self.preset == SatellitePreset::Himawari
                        && self.geo_navigation == GeoNavigation::GoesRAbiFixedGrid
                    {
                        self.geo_navigation = GeoNavigation::ModelSphere;
                    }
                    ui.label("Nav:");
                    ui.add_enabled_ui(
                        self.view == StudioView::Geostationary
                            && self.preset != SatellitePreset::Himawari,
                        |ui| {
                            egui::ComboBox::from_id_salt("geo-navigation")
                                .width(toolbar.navigation_width)
                                .selected_text(self.geo_navigation.label())
                                .truncate()
                                .show_ui(ui, |ui| {
                                    for navigation in GeoNavigation::ALL {
                                        ui.selectable_value(
                                            &mut self.geo_navigation,
                                            navigation,
                                            navigation.label(),
                                        );
                                    }
                                });
                        },
                    )
                    .response
                    .on_hover_text(
                        "Model sphere preserves the shipped WRF-consistent camera. GOES-R ABI \
                         ellipsoid uses official sweep-x fixed-grid navigation and metadata, \
                         then maps each geodetic pixel onto the existing spherical model rays. \
                         This is registration geometry, not a claim of exact ABI radiometry.",
                    );
                    // View toggle: from-space geostationary <-> the top-down map-registered
                    // product <-> the Perspective (3-D) orbit view (CPU-rendered like top-down).
                    ui.label("View:");
                    egui::ComboBox::from_id_salt("view")
                        .width(toolbar.view_width)
                        .selected_text(self.view.label())
                        .truncate()
                        .show_ui(ui, |ui| {
                            for v in StudioView::ALL {
                                ui.selectable_value(&mut self.view, v, v.label());
                            }
                        })
                        .response
                        .on_hover_text(
                            "Geostationary = the physically-authentic from-space satellite view \
                             (curved earth, limb, space). Top-down map = a synthetic north-up \
                             near-nadir map over the WRF domain's own Lambert extent, which \
                             registers with top-down field plots (the WRF-Runner integration \
                             product). Perspective (3-D) = a free camera orbiting the domain \
                             centre (azimuth / tilt / range / FOV in the Camera group) — angled \
                             3-D storm shots with true parallax; Visible mode only in v1. \
                             Top-down and Perspective render on the CPU.",
                        );
                    ui.label("Time:");
                    let current = self
                        .timesteps
                        .get(self.selected_ts)
                        .map(|t| t.label.clone())
                        .unwrap_or_else(|| "-".to_string());
                    egui::ComboBox::from_id_salt("ts")
                        .width(toolbar.timestep_width)
                        .selected_text(&current)
                        .truncate()
                        .show_ui(ui, |ui| {
                            for (i, t) in self.timesteps.iter().enumerate() {
                                ui.selectable_value(&mut self.selected_ts, i, &t.label);
                            }
                        })
                        .response
                        .on_hover_text(current);
        });
    }

    /// Non-destructive calibration migration. Settings written before the current
    /// epoch keep every saved value; this banner makes that fact visible and asks the
    /// user to either apply the current preset or deliberately keep their controls.
    fn calibration_epoch_banner(&mut self, ui: &mut egui::Ui) {
        if self.visible_calibration_epoch >= settings::VISIBLE_CALIBRATION_EPOCH {
            return;
        }
        ui.add_space(2.0);
        egui::Frame::new()
            .fill(egui::Color32::from_rgb(55, 45, 18))
            .inner_margin(egui::Margin::symmetric(8, 5))
            .corner_radius(3.0)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.colored_label(
                        egui::Color32::from_rgb(245, 210, 120),
                        "This build has a new visible calibration. Your saved controls are still active.",
                    );
                    if ui.button("Use current visible preset").clicked() {
                        let mut migrated = self.settings_snapshot();
                        migrated.apply_shipped_visible_calibration();
                        self.apply_settings(&migrated);
                    }
                    if ui.button("Keep my saved controls").clicked() {
                        self.visible_calibration_epoch = settings::VISIBLE_CALIBRATION_EPOCH;
                    }
                });
                ui.weak(
                    "Full shipped visible baseline: AOD/atmosphere/output/cloud toggles plus \
                     OD 0.15, exposure 1.50, SZA normalization on with max gain 4.00, dark-land \
                     toe on, legacy post-light toe off, twilight recovery on (0.30/0.50/4.00; \
                     -6..+12 deg), exposed-domain edge feathering on, ground 1.10, knee 0.65, \
                     ceiling 1.25, fractional clouds on, granulation off.",
                );
            });
    }

    /// Compact, always-visible entry points into the three reviewed configurations.
    /// Exact diffs/descriptions stay available in one bounded Details drawer, while
    /// an immediate Current/Custom chip makes subsequent manual edits honest.
    fn recommended_presets(&mut self, ui: &mut egui::Ui) {
        let current = self.settings_snapshot();
        let runtime = presets::PresetRuntime {
            gpu_clouds: self.gpu_clouds,
            parity_pending: self.parity_pending,
        };
        let plans: Vec<_> = presets::StudioPreset::ALL
            .into_iter()
            .map(|preset| (preset, presets::plan(preset, &current, runtime)))
            .collect();
        let active = presets::active(&current, runtime);
        let mut selected = None;

        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("Quick mode:").strong());
            for (preset, planned) in &plans {
                match planned {
                    Ok(plan) => {
                        let response = ui
                            .add_enabled(
                                !self.busy,
                                egui::Button::new(preset.quick_label())
                                    .selected(active == Some(*preset)),
                            )
                            .on_hover_text(format!(
                                "{}\n\nExact changes now:\n{}",
                                preset.description(),
                                plan.change_summary()
                            ))
                            .on_disabled_hover_text(
                                "Wait for the current render to finish before changing its saved configuration.",
                            );
                        if response.clicked() {
                            selected = Some((*preset, plan.clone()));
                        }
                    }
                    Err(reason) => {
                        ui.add_enabled(
                            false,
                            egui::Button::new(preset.quick_label()).selected(false),
                        )
                        .on_disabled_hover_text(format!(
                            "{}\n\nUnavailable: {reason}",
                            preset.description()
                        ));
                    }
                }
            }
            ui.separator();
            let (current_label, current_color) = match active {
                Some(preset) => (
                    preset.quick_label(),
                    egui::Color32::from_rgb(135, 205, 145),
                ),
                None => ("Custom", egui::Color32::from_rgb(235, 185, 110)),
            };
            ui.colored_label(current_color, format!("Current: {current_label}"));
            ui.separator();
            ui.weak("GPU Render is a one-click fast preview; no settings setup required.");
        });

        let details_height = settings_scroll_max_height(ui.ctx().content_rect().height());
        egui::CollapsingHeader::new("Details")
            .id_salt("quick-mode-details")
            .default_open(false)
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("quick-mode-details-scroll")
                    .max_height(details_height)
                    .show(ui, |ui| {
                        ui.weak(
                            "Quick modes save only their listed fields. They never hide the \
                             individual controls or silently change the selected product/source.",
                        );
                        for (index, (preset, planned)) in plans.iter().enumerate() {
                            if index != 0 {
                                ui.separator();
                            }
                            ui.horizontal_wrapped(|ui| {
                                ui.label(egui::RichText::new(preset.label()).strong());
                                if active == Some(*preset) {
                                    ui.colored_label(
                                        egui::Color32::from_rgb(135, 205, 145),
                                        "Current",
                                    );
                                }
                            });
                            ui.label(preset.description());
                            match planned {
                                Ok(plan) if plan.changes.is_empty() => {
                                    ui.weak("Already active — no settings change.");
                                }
                                Ok(plan) => {
                                    ui.weak(format!("Exact changes now ({}):", plan.changes.len()));
                                    for change in &plan.changes {
                                        ui.monospace(change.to_string());
                                    }
                                }
                                Err(reason) => {
                                    ui.colored_label(
                                        egui::Color32::from_rgb(235, 180, 105),
                                        format!("Unavailable: {reason}"),
                                    );
                                }
                            }
                        }
                    });
            });

        if let Some((preset, plan)) = selected {
            let exact = plan.change_summary();
            self.apply_settings(&plan.settings);
            self.gpu_clouds = plan.runtime.gpu_clouds;
            self.parity_pending = plan.runtime.parity_pending;
            // Presets are deliberate button actions, so persist immediately instead
            // of waiting for the normal slider-drag debounce.
            self.save_settings_now();
            self.logline(format!(
                "Applied and saved {}. Render again to update the image. {}",
                preset.label(),
                exact
            ));
        }
    }

    /// A slim, colored one-line note under the strip explaining the current product and which
    /// drawer groups apply — only shown for the modes whose controls differ from plain Visible
    /// (so the owner knows why the Sun / atmosphere / cloud groups are absent in a thermal or
    /// derived mode).
    fn context_note(&mut self, ui: &mut egui::Ui) {
        let mode = self.render_mode;
        if mode.is_thermal() {
            ui.add_space(2.0);
            ui.colored_label(
                egui::Color32::from_rgb(120, 180, 230),
                format!(
                    "{} (thermal): works day AND night. The Sun / exposure / atmosphere / cloud \
                     controls do not apply and are hidden.",
                    mode.label()
                ),
            );
        } else if mode.is_geocolor() {
            ui.add_space(2.0);
            ui.colored_label(
                egui::Color32::from_rgb(140, 200, 150),
                "GeoColor Style — SimSat Day/Night Color: broad-RGB visible by day, colored \
                 band-13 IR by night, crossfaded across the terminator. It is not yet \
                 sensor-derived ABI GeoColor. The Sun / exposure / atmosphere / cloud \
                 controls light the VISIBLE (day) half; the night half is thermal IR (no city \
                 lights — the night side is the colored IR).",
            );
        } else if mode.is_sandwich() {
            ui.add_space(2.0);
            ui.colored_label(
                egui::Color32::from_rgb(200, 170, 140),
                "Sandwich (vis + cold-top IR): the true-color visible cloud texture with the \
                 coldest overshooting tops highlighted in color-enhanced band-13 IR. The Sun / \
                 exposure / atmosphere / cloud controls light the VISIBLE base — a DAYTIME \
                 product; at night it degrades to ~IR (use IR or GeoColor for a night storm).",
            );
        } else if let Some(field) = mode.derived_field() {
            ui.add_space(2.0);
            let unit = field.units();
            ui.colored_label(
                egui::Color32::from_rgb(150, 210, 190),
                format!(
                    "{} (derived scalar map, {}): a per-column brick integral shown with a basic \
                     colormap. The RAW physical values are the plotting deliverable (the `import \
                     simsat` binding); the Sun / exposure / atmosphere / cloud controls do not \
                     apply (day AND night).",
                    field.label(),
                    if unit.is_empty() {
                        "dimensionless"
                    } else {
                        unit
                    },
                ),
            );
        }

        // Brick storage is a source/precision choice, not a visible-cloud appearance
        // control. Keep it available for every product (including IR, WV, and derived
        // fields) and even when the visible cloud layer is disabled.
        egui::CollapsingHeader::new("Storage / precision")
            .default_open(false)
            .show(ui, |ui| {
                let science_storage_response = ui
                    .checkbox(
                        &mut self.science_cloud_f16,
                        "ScienceCloudF16 precision (CPU, experimental)",
                    )
                    .on_hover_text(
                        "Store liquid, ice, snow, and total precipitation extinction as bounded \
                         log2-f16 source values in an isolated v7 cache. CompactU8 remains the \
                         production default. ScienceCloudF16 improves thin-cloud and thermal \
                         precision but uses a larger cache and requires a source re-ingest. It \
                         applies to visible, IR, water-vapor, and derived products. GPU preview \
                         stays unavailable because its textures consume CompactU8 codes directly.",
                    );
                if science_storage_response.changed() && self.science_cloud_f16 {
                    self.gpu_clouds = false;
                    self.parity_pending = false;
                }
                ui.weak(if self.science_cloud_f16 {
                    "Selected storage: ScienceCloudF16 (CPU-only experimental cache)."
                } else {
                    "Selected storage: CompactU8 (production default)."
                });
            });

        if matches!(
            mode,
            RenderMode::Ir | RenderMode::GeoColor | RenderMode::Sandwich
        ) {
            egui::CollapsingHeader::new("Thermal response")
                .default_open(false)
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Band 13 sensor:");
                        egui::ComboBox::from_id_salt("thermal-sensor")
                            .selected_text(self.thermal_sensor.label())
                            .show_ui(ui, |ui| {
                                for sensor in ThermalSensor::ALL {
                                    ui.selectable_value(
                                        &mut self.thermal_sensor,
                                        sensor,
                                        sensor.label(),
                                    );
                                }
                            })
                            .response
                            .on_hover_text(
                                "Fast gray preserves the historical 10.3 um center-wavelength path. \
                                 GOES-R ABI Band 13 integrates Planck emission through NOAA's \
                                 official FM4/GOES-19 spectral response and inverts that response \
                                 to brightness temperature.",
                            );
                    });
                    if let Some(warning) = self.thermal_sensor.limitation_warning() {
                        ui.colored_label(egui::Color32::YELLOW, format!("Science limitation: {warning}"));
                    }
                    ui.separator();
                    let mut footprint_on =
                        self.instrument_footprint == InstrumentFootprint::GoesRAbiBand13Mtf;
                    if ui
                        .checkbox(&mut footprint_on, "ABI Band 13 MTF footprint (experimental)")
                        .on_hover_text(
                            "Applies the GOES-16-measured Band 13 EW MTF-informed [0.15, 0.70, \
                             0.15] response to complete FM4 channel radiance before BT inversion. \
                             Enabling it selects GOES-East if needed, Geostationary view, GOES-R \
                             exact sweep-x navigation, ABI 2 km, and the FM4/GOES-19 response. \
                             The 56-urad grid is globally lattice-snapped with SSP at the corner \
                             of four pixels. Default off; GOES-16 MTF transfer to GOES-19 is \
                             explicitly unvalidated.",
                        )
                        .changed()
                    {
                        self.instrument_footprint = if footprint_on {
                            if self.preset == SatellitePreset::Himawari {
                                self.preset = SatellitePreset::GoesEast;
                            }
                            self.view = StudioView::Geostationary;
                            self.geo_navigation = GeoNavigation::GoesRAbiFixedGrid;
                            self.resolution = ResolutionMode::Abi2km;
                            self.thermal_sensor = ThermalSensor::GoesRAbiBand13Fm4;
                            self.gpu_clouds = false;
                            self.parity_pending = false;
                            InstrumentFootprint::GoesRAbiBand13Mtf
                        } else {
                            InstrumentFootprint::Off
                        };
                    }
                    if self.instrument_footprint != InstrumentFootprint::Off {
                        let compatible = self.view == StudioView::Geostationary
                            && self.preset != SatellitePreset::Himawari
                            && self.geo_navigation == GeoNavigation::GoesRAbiFixedGrid
                            && self.resolution == ResolutionMode::Abi2km
                            && self.thermal_sensor == ThermalSensor::GoesRAbiBand13Fm4;
                        ui.colored_label(
                            if compatible {
                                egui::Color32::from_rgb(140, 200, 150)
                            } else {
                                egui::Color32::YELLOW
                            },
                            if compatible {
                                "Exact global 56-urad lattice active; crop/mask perimeter is no-data for validation."
                            } else {
                                "Footprint needs Geostationary + GOES-R exact + ABI 2 km + FM4; toggle it off/on to reapply those requirements."
                            },
                        );
                        ui.weak(
                            "Science limitation: GOES-16 EW MTF transferred to GOES-19/FM4; \
                             Band 13 NS MTF, temporal integration, and detector variation remain unmodeled.",
                        );
                    }
                });
        }
        if let Some(state) = &self.rendered
            && state.instrument_footprint != InstrumentFootprint::Off
        {
            ui.colored_label(
                egui::Color32::from_rgb(140, 200, 150),
                format!(
                    "Rendered instrument footprint: {} (experimental, default off).",
                    state.instrument_footprint.label()
                ),
            );
        }
        // GPU-clouds preview marker: the displayed frame came from the experimental
        // GPU pass — say so (with the frame time) and that the store stays CPU.
        if let Some(state) = &self.rendered
            && let Some(ms) = state.gpu_ms
        {
            ui.add_space(2.0);
            ui.colored_label(
                egui::Color32::from_rgb(235, 185, 120),
                format!(
                    "GPU clouds (experimental preview): frame rendered on the GPU in {ms} ms. \
                     Stored frames always come from the CPU path."
                ),
            );
            if state.one_click_gpu_render {
                let changes = state.gpu_preview_adjustments;
                let detail = if changes.is_empty() {
                    "Current controls already matched the GPU envelope.".to_string()
                } else {
                    format!("Temporary preview differences: {}.", changes.summary())
                };
                ui.colored_label(
                    egui::Color32::from_rgb(235, 205, 135),
                    format!("One-click GPU Render: {detail} Studio settings were not changed."),
                );
            }
        }
    }

    /// The M0 size-gate confirm prompts for a large single wrfout file or a large sequence,
    /// kept directly under the strip (not buried in a collapsible) so the owner must confirm
    /// before the heavy ingest, exactly as before.
    fn size_gate_confirms(&mut self, ui: &mut egui::Ui) {
        let mut source_confirmed_now = false;
        if let Some(Source::Wrfout {
            needs_confirm,
            confirmed,
            nx,
            ny,
            nz,
            file_bytes,
            ..
        }) = &mut self.source
            && *needs_confirm
            && !*confirmed
        {
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                ui.colored_label(
                    egui::Color32::from_rgb(230, 170, 60),
                    format!(
                        "Large import: {nx}x{ny}x{nz}, {:.2} GB. Ingest may use significant memory.",
                        *file_bytes as f64 / (1u64 << 30) as f64
                    ),
                );
                if ui.button("Confirm and continue").clicked() {
                    *confirmed = true;
                    source_confirmed_now = true;
                }
            });
        }
        if source_confirmed_now {
            self.logline(
                "Large source confirmed. Ready to Render; an existing cache will be reused or ingest will start on demand.",
            );
        }

        let mut sequence_confirmed_now = false;
        if let Some(Source::Sequence {
            needs_confirm,
            confirmed,
            total_bytes,
            entries,
            ..
        }) = &mut self.source
            && *needs_confirm
            && !*confirmed
        {
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                ui.colored_label(
                    egui::Color32::from_rgb(230, 170, 60),
                    format!(
                        "Large sequence: {} timesteps, {:.2} GB total. Batch render ingests each \
                         file one at a time (below-normal priority).",
                        entries.len(),
                        *total_bytes as f64 / (1u64 << 30) as f64
                    ),
                );
                if ui.button("Confirm and continue").clicked() {
                    *confirmed = true;
                    sequence_confirmed_now = true;
                }
            });
        }
        if sequence_confirmed_now {
            self.logline(
                "Large sequence confirmed. Ready to Render sequence; cached frames will be reused or ingested on demand.",
            );
        }
    }

    /// The context-driven Advanced drawer: a set of collapsible group headers below the strip.
    /// ONLY the groups relevant to the current Mode are shown — Camera always; Sun & Exposure /
    /// Atmosphere / Clouds / Ground for the visible-path modes (Visible / GeoColor / Sandwich);
    /// Enhancement for the thermal IR / WV modes; Field for the derived-scalar modes; Output
    /// (sat store) always. Every control keeps the SAME wired value as before — only its
    /// grouping and context-visibility changed.
    fn advanced_drawer(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let mode = self.render_mode;

        // Camera — all modes. In Perspective (3-D) view the group shows the orbit
        // controls instead (Resolution / margin do not apply — the camera frames
        // the scene; the march still samples the full-resolution data).
        egui::CollapsingHeader::new("Camera")
            .default_open(true)
            .show(ui, |ui| {
                if self.view == StudioView::Perspective {
                    ui.horizontal_wrapped(|ui| {
                        ui.add(
                            egui::Slider::new(&mut self.orbit_az_deg, 0.0..=360.0)
                                .text("Azimuth deg"),
                        )
                        .on_hover_text(
                            "Compass direction the camera sits FROM the domain centre \
                             (0 = north of it looking south, 180 = south of it looking \
                             north).",
                        );
                        ui.add(
                            egui::Slider::new(&mut self.orbit_tilt_deg, 5.0..=85.0)
                                .text("Tilt deg"),
                        )
                        .on_hover_text(
                            "Camera elevation above the horizontal, seen from the domain \
                             centre: 85 = nearly overhead, low values = a low oblique \
                             flyover angle.",
                        );
                        ui.add(
                            egui::Slider::new(&mut self.orbit_range_km, 10.0..=5000.0)
                                .logarithmic(true)
                                .text("Range km"),
                        )
                        .on_hover_text(
                            "Slant distance from the domain centre to the camera. Clamped \
                             at render time to 0.3x-5x the domain diagonal (logged when \
                             clamped).",
                        );
                        ui.add(
                            egui::Slider::new(&mut self.orbit_fov_deg, 15.0..=120.0)
                                .text("FOV deg"),
                        )
                        .on_hover_text("Horizontal field of view of the output image.");
                    });
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Output size:");
                        ui.add(egui::Slider::new(&mut self.persp_width, 2..=4096).text("W px"));
                        ui.add(egui::Slider::new(&mut self.persp_height, 2..=4096).text("H px"));
                        ui.label(
                            egui::RichText::new(
                                "(Resolution / zoom-out margin do not apply in this view)",
                            )
                            .weak(),
                        );
                    });
                    return;
                }
                ui.horizontal_wrapped(|ui| {
                    ui.label("Resolution:");
                    egui::ComboBox::from_id_salt("res")
                        .selected_text(self.resolution.label())
                        .show_ui(ui, |ui| {
                            for r in ResolutionMode::ALL {
                                ui.selectable_value(&mut self.resolution, r, r.label());
                            }
                        })
                        .response
                        .on_hover_text(
                            "Model native = one output pixel per source grid cell (the default), \
                             not necessarily the highest output resolution. ABI 1 km / 2 km use \
                             the fixed GOES scan pitch in Geostationary view and physical 1 km / \
                             2 km map spacing in Top-down view, so they may upsample a coarse \
                             model or downsample a fine WRF grid.",
                        );
                    ui.separator();
                    ui.label("Zoom out / margin:");
                    ui.add(
                        egui::Slider::new(&mut self.margin_pct, 0.0..=100.0)
                            .suffix("%")
                            .fixed_decimals(0),
                    )
                    .on_hover_text(
                        "Zoom out so the domain sits WITHIN the frame with real earth around it, \
                         instead of running edge-to-edge into the frame. The margin is a \
                         percentage of the domain size added on each side (30% = the domain \
                         fills the center ~1/1.6 of the frame). The margin shows the real Blue \
                         Marble ground + clear sky AROUND the domain — WRF has no data outside \
                         the domain, so no clouds/weather render there (honest context). Applies \
                         to both views. 0% = the domain edge-to-edge.",
                    );
                });
            });

        if mode.uses_visible_controls() {
            // Sun & Exposure.
            egui::CollapsingHeader::new("Sun & Exposure")
                .default_open(true)
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.checkbox(&mut self.sun_override, "Override sun (what-if)")
                            .on_hover_text(
                                "NON-PHYSICAL visualization aid: light the frame with a chosen \
                                 sun elevation/azimuth over the domain centre, ignoring the \
                                 file's real time (e.g. show a night storm at noon). Off = the \
                                 file's real sun. When on, the render will NOT match the \
                                 satellite's real view at that time — the status bar marks the \
                                 frame a what-if.",
                            );
                        ui.add_enabled_ui(self.sun_override, |ui| {
                            ui.add(
                                egui::Slider::new(&mut self.sun_override_elev, -10.0..=90.0)
                                    .text("Sun elev deg"),
                            );
                            ui.add(
                                egui::Slider::new(&mut self.sun_override_az, 0.0..=360.0)
                                    .text("Sun az deg"),
                            );
                        });
                        if self.sun_override {
                            ui.colored_label(
                                egui::Color32::from_rgb(230, 170, 60),
                                "what-if (non-physical)",
                            );
                        }
                    });
                    ui.horizontal_wrapped(|ui| {
                        ui.add(egui::Slider::new(&mut self.exposure, 0.25..=4.0).text("Exposure"))
                            .on_hover_text(
                                "Display-side brightness gain applied before the ABI stretch \
                                 (1.0 = physical reflectance). Brightens surface + cloud together \
                                 on the clouds-on composite.",
                            );
                    });
                });

            // Finished-display calibration. These persisted controls change only the
            // visible RGB-family tonemap/surface lift; raw bands and thermal/derived
            // products never consume them.
            egui::CollapsingHeader::new("Advanced display calibration")
                .default_open(false)
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.add(
                            egui::Slider::new(&mut self.ground_gain, 0.25..=4.0)
                                .text("Ground lift"),
                        )
                        .on_hover_text(
                            "Sun-gated daytime surface-radiance lift. 1.0 is neutral. It does \
                             not brighten cloud radiance and does not alter raw bands.",
                        );
                        ui.add(
                            egui::Slider::new(&mut self.cloud_softclip, 0.05..=1.0)
                                .text("Highlight knee"),
                        )
                        .on_hover_text(
                            "Start of the finished-display highlight shoulder. 1.0 disables \
                             the shoulder and restores a hard clamp.",
                        );
                        ui.add(
                            egui::Slider::new(&mut self.cloud_highlight_max, 0.25..=4.0)
                                .text("Highlight ceiling"),
                        )
                        .on_hover_text(
                            "Physical reflectance factor mapped to display white. Raising it \
                             keeps more structure in bright anvils; this is display-only.",
                        );
                    });
                    ui.separator();
                    ui.label("Land visibility calibration (independent controls)");
                    ui.horizontal_wrapped(|ui| {
                        ui.checkbox(
                            &mut self.land_sza_normalization,
                            "Land solar-zenith normalization",
                        )
                        .on_hover_text(
                            "Bounded land-only operational-display correction for moderate sun \
                             angles. It is exactly neutral through twilight and at/above a \
                             60-degree sun; ocean, cloud radiance, and raw bands are untouched.",
                        );
                        ui.add_enabled_ui(self.land_sza_normalization, |ui| {
                            ui.add(
                                egui::Slider::new(&mut self.land_sza_max_gain, 1.0..=4.0)
                                    .text("SZA max gain"),
                            )
                            .on_hover_text("Upper bound; 1.0 is an identity correction.");
                        });
                    });
                    ui.horizontal_wrapped(|ui| {
                        ui.checkbox(&mut self.land_dark_toe, "Dark-land reflectance toe")
                            .on_hover_text(
                                "Bounded scalar lift below a linear-reflectance knee. It \
                                 preserves black and land colour ratios; brighter terrain, \
                                 ocean, cloud radiance, twilight, and raw bands are unchanged.",
                            );
                        ui.add_enabled_ui(self.land_dark_toe, |ui| {
                            ui.add(
                                egui::Slider::new(&mut self.land_dark_toe_knee, 0.001..=0.25)
                                    .text("Toe knee"),
                            );
                            ui.add(
                                egui::Slider::new(&mut self.land_dark_toe_gamma, 0.05..=1.0)
                                    .text("Toe gamma"),
                            )
                            .on_hover_text("1.0 is identity; lower values lift dark land.");
                            ui.add(
                                egui::Slider::new(&mut self.land_dark_toe_max_gain, 1.0..=4.0)
                                    .text("Toe max gain"),
                            );
                        });
                    });
                    ui.separator();
                    ui.label("Post-lighting terrain recovery");
                    ui.horizontal_wrapped(|ui| {
                        ui.checkbox(
                            &mut self.surface_postlight_toe,
                            "Post-light surface toe",
                        )
                        .on_hover_text(
                            "Default-off display experiment. Applies a bounded scalar toe to \
                             LAND after lighting and camera transmittance, before atmospheric \
                             airlight and clouds. Ocean/glint, cloud radiance, raw bands, and \
                             Sensor Fast Gray are unchanged.",
                        );
                        ui.add_enabled_ui(self.surface_postlight_toe, |ui| {
                            ui.add(
                                egui::Slider::new(
                                    &mut self.surface_postlight_toe_knee,
                                    0.001..=0.50,
                                )
                                .text("Post-light knee"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.surface_postlight_toe_gamma,
                                    0.05..=1.0,
                                )
                                .text("Post-light gamma"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.surface_postlight_toe_max_gain,
                                    1.0..=4.0,
                                )
                                .text("Post-light max gain"),
                            );
                        });
                    });
                    ui.horizontal_wrapped(|ui| {
                        ui.checkbox(
                            &mut self.twilight_surface_recovery,
                            "Twilight surface recovery",
                        )
                        .on_hover_text(
                            "Shipped finished-visible low-sun terrain recovery. It fades in from \
                             -6 to 0 degrees, is full through +4, fades to identity by +12, and \
                             never affects ocean/glint, clouds, raw/sensor products, or ordinary \
                             daylight. Turn it off for an exact A/B.",
                        );
                        ui.add_enabled_ui(self.twilight_surface_recovery, |ui| {
                            ui.add(
                                egui::Slider::new(
                                    &mut self.twilight_surface_recovery_knee,
                                    0.001..=0.50,
                                )
                                .text("Twilight knee"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.twilight_surface_recovery_gamma,
                                    0.05..=1.0,
                                )
                                .text("Twilight gamma"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.twilight_surface_recovery_max_gain,
                                    1.0..=4.0,
                                )
                                .text("Twilight max gain"),
                            );
                        });
                    });
                    if self.surface_postlight_toe {
                        ui.weak(
                            "Post-light toe active for finished visible LAND; CPU and GPU use \
                             the same post-view formula.",
                        );
                    }
                    if self.twilight_surface_recovery {
                        ui.weak(
                            "Twilight recovery active only in the -6 to +12 degree low-sun \
                             window; CPU and GPU gains combine it with the legacy toe by max.",
                        );
                    }
                    if self.land_sza_normalization || self.land_dark_toe {
                        ui.weak(
                            "Land visibility corrections active for finished visible land only; \
                             CPU and GPU surface paths use the same bounded formulas.",
                        );
                    }
                    if ui.button("Restore shipped land calibration").clicked() {
                        let land = LandAppearanceConfig::default();
                        self.land_sza_normalization = land.sza_normalization;
                        self.land_sza_max_gain = land.sza_max_gain as f32;
                        self.land_dark_toe = land.dark_toe;
                        self.land_dark_toe_knee = land.dark_toe_knee as f32;
                        self.land_dark_toe_gamma = land.dark_toe_gamma as f32;
                        self.land_dark_toe_max_gain = land.dark_toe_max_gain as f32;
                    }
                    let calibrated = display_calibration_is_dirty(
                        self.exposure,
                        self.ground_gain,
                        self.cloud_softclip,
                        self.cloud_highlight_max,
                    );
                    if ui
                        .add_enabled(calibrated, egui::Button::new("Restore shipped display calibration"))
                        .on_hover_text(
                            "Reset exposure, ground lift, highlight knee, and highlight ceiling \
                             to the engine's shipped constants.",
                        )
                        .clicked()
                    {
                        (
                            self.exposure,
                            self.ground_gain,
                            self.cloud_softclip,
                            self.cloud_highlight_max,
                        ) = shipped_display_calibration();
                    }
                    ui.weak(
                        "Display-only: raw visible bands, IR, water vapor, and derived fields are unchanged.",
                    );
                });

            // Atmosphere.
            egui::CollapsingHeader::new("Atmosphere")
                .default_open(false)
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Aerosol:");
                        ui.add(
                            egui::Slider::new(&mut self.aod, 0.0..=0.6)
                                .text("Aerosol optical depth (AOD)"),
                        )
                        .on_hover_text(
                            "Visible aerosol optical depth at 550 nm. This does not disable \
                             molecular (Rayleigh) scattering.",
                        );
                        ui.checkbox(&mut self.rh_swelling, "RH aerosol swelling")
                            .on_hover_text(
                                "Scales aerosol extinction by 1.5 to represent humid growth. \
                                 Off leaves AOD at the numeric value shown.",
                            );
                        if self.rh_swelling {
                            ui.weak(format!("effective aerosol AOD {:.3}", self.aod * 1.5));
                        }
                    });
                    ui.horizontal_wrapped(|ui| {
                        ui.checkbox(
                            &mut self.atmosphere_correction,
                            "Daytime aerial-veil correction",
                        )
                        .on_hover_text(
                            "Reduce the modeled daytime atmospheric veil for the true-color \
                             product. On is the product-facing default; off retains full \
                             modeled path airlight. Other display transforms remain.",
                        );
                        ui.checkbox(&mut self.terrain_atmosphere, "Terrain-height atmosphere")
                            .on_hover_text(
                                "Shorten the view and sunlight atmospheric columns to each pixel's \
                             model terrain elevation. On is physical; off reproduces the old \
                             full sea-level column for QA.",
                            );
                    });
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Output:");
                        egui::ComboBox::from_id_salt("outtx")
                            .selected_text(match self.output_transform {
                                OutputTransform::AbiReflectance => "ABI reflectance",
                                OutputTransform::DebugSrgb => "Debug sRGB",
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut self.output_transform,
                                    OutputTransform::AbiReflectance,
                                    "ABI reflectance (default)",
                                );
                                ui.selectable_value(
                                    &mut self.output_transform,
                                    OutputTransform::DebugSrgb,
                                    "Debug sRGB",
                                );
                            });
                    });
                });

            // Clouds.
            egui::CollapsingHeader::new("Clouds")
                .default_open(true)
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.checkbox(&mut self.clouds_enabled, "On").on_hover_text(
                            "Volumetric cloud raymarch (M4). Off = the M2 clear-sky surface.",
                        );
                        ui.add_enabled_ui(self.clouds_enabled, |ui| {
                            if ui
                                .checkbox(
                                    &mut self.fractional_clouds,
                                    "Use model cloud fraction",
                                )
                                .on_hover_text(
                                     "Use the source model cloud-fraction field to render \
                                     fractional subcolumns and wispy cloud edges. On is the \
                                     physical default; files without the field safely fall \
                                     back to full-cell coverage. Off reproduces the legacy \
                                     behavior where every non-zero cloudy cell fills the \
                                     horizontal cell.",
                                )
                                .changed()
                                && self.fractional_clouds
                            {
                                if self.fractional_cloud_mode == FractionalCloudMode::Off {
                                    self.fractional_cloud_mode = FractionalCloudMode::Deterministic2;
                                }
                                // The current WGSL preview has no fractional-subcolumn closure.
                                // Do not leave a now-disabled preview silently checked.
                                self.gpu_clouds = false;
                                self.parity_pending = false;
                            }
                            if self.fractional_clouds {
                                ui.label("Closure:");
                                egui::ComboBox::from_id_salt("fractional-cloud-mode")
                                    .selected_text(match self.fractional_cloud_mode {
                                        FractionalCloudMode::EffectiveOd => {
                                            "Effective OD (fast/explicit)"
                                        }
                                        FractionalCloudMode::Deterministic2 => {
                                            "Deterministic 2 (Recommended)"
                                        }
                                        FractionalCloudMode::Deterministic4 => {
                                            "Deterministic 4 (reference)"
                                        }
                                        FractionalCloudMode::Deterministic8 => {
                                            "Deterministic 8 (convergence)"
                                        }
                                        FractionalCloudMode::Deterministic16 => {
                                            "Deterministic 16 (convergence)"
                                        }
                                        FractionalCloudMode::Off => "Off",
                                    })
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(
                                            &mut self.fractional_cloud_mode,
                                            FractionalCloudMode::Deterministic2,
                                            "Deterministic 2 (Recommended, ~2x)",
                                        )
                                        .on_hover_text(
                                            "Two fixed shared-u maximum-overlap subcolumns. This \
                                             is the fastest explicit cloudy/clear radiance closure; \
                                             use Deterministic 4 when finer fraction quadrature matters.",
                                        );
                                        ui.selectable_value(
                                            &mut self.fractional_cloud_mode,
                                            FractionalCloudMode::EffectiveOd,
                                            "Effective OD (fast/explicit)",
                                        );
                                        ui.selectable_value(
                                            &mut self.fractional_cloud_mode,
                                            FractionalCloudMode::Deterministic4,
                                            "Deterministic 4 (4x CPU reference)",
                                        )
                                        .on_hover_text(
                                            "Four fixed shared-u maximum-overlap subcolumns. Each \
                                             gets its own view/sun/shadow march; linear radiance is \
                                             averaged before one tonemap. Much slower and intended \
                                             for physics QA, not the shipped default.",
                                        );
                                        ui.selectable_value(
                                            &mut self.fractional_cloud_mode,
                                            FractionalCloudMode::Deterministic8,
                                            "Deterministic 8 (8x convergence reference)",
                                        )
                                        .on_hover_text(
                                            "Eight fixed-stratified shared-u maximum-overlap \
                                             members. Higher quadrature resolution at roughly \
                                             twice the cost of Deterministic 4.",
                                        );
                                        ui.selectable_value(
                                            &mut self.fractional_cloud_mode,
                                            FractionalCloudMode::Deterministic16,
                                            "Deterministic 16 (16x convergence reference)",
                                        )
                                        .on_hover_text(
                                            "Sixteen fixed-stratified shared-u maximum-overlap \
                                             members. Expensive convergence reference; Effective \
                                             OD remains the default.",
                                        );
                                    });
                            }
                            let previous_transport = studio_cloud_multiscatter_mode(
                                self.multiscatter,
                                self.delta_flux_clouds,
                                self.delta_flux_v2_clouds,
                                self.delta_flux_v3_clouds,
                            );
                            let mut cloud_transport = previous_transport;
                            ui.horizontal_wrapped(|ui| {
                                ui.label("Cloud transport:");
                                egui::ComboBox::from_id_salt("cloud-transport")
                                    .selected_text(match cloud_transport {
                                        CloudMultiscatterMode::LegacyOctaves => {
                                            "Legacy octaves (default)"
                                        }
                                        CloudMultiscatterMode::SingleScatter => "Single scatter",
                                        CloudMultiscatterMode::DeltaFluxV1 => {
                                            "Delta-flux v1 (experimental)"
                                        }
                                        CloudMultiscatterMode::DeltaFluxV2 => {
                                            "Delta-flux v2b P1 (experimental)"
                                        }
                                        CloudMultiscatterMode::DeltaFluxV3 => {
                                            "Delta-flux v3 order memory (experimental)"
                                        }
                                    })
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(
                                            &mut cloud_transport,
                                            CloudMultiscatterMode::LegacyOctaves,
                                            "Legacy octaves (default, exact v0.1.5)",
                                        );
                                        ui.selectable_value(
                                            &mut cloud_transport,
                                            CloudMultiscatterMode::SingleScatter,
                                            "Single scatter",
                                        );
                                        ui.selectable_value(
                                            &mut cloud_transport,
                                            CloudMultiscatterMode::DeltaFluxV1,
                                            "Delta-flux v1 (experimental, CPU)",
                                        );
                                        ui.selectable_value(
                                            &mut cloud_transport,
                                            CloudMultiscatterMode::DeltaFluxV2,
                                            "Delta-flux v2b P1 (experimental, CPU)",
                                        );
                                        ui.selectable_value(
                                            &mut cloud_transport,
                                            CloudMultiscatterMode::DeltaFluxV3,
                                            "Delta-flux v3 order memory (experimental, CPU)",
                                        );
                                    });
                            });
                            self.delta_flux_clouds =
                                cloud_transport == CloudMultiscatterMode::DeltaFluxV1;
                            self.delta_flux_v2_clouds =
                                cloud_transport == CloudMultiscatterMode::DeltaFluxV2;
                            self.delta_flux_v3_clouds =
                                cloud_transport == CloudMultiscatterMode::DeltaFluxV3;
                            self.multiscatter =
                                cloud_transport == CloudMultiscatterMode::LegacyOctaves;
                            if cloud_transport != previous_transport
                                && matches!(
                                    cloud_transport,
                                    CloudMultiscatterMode::DeltaFluxV1
                                        | CloudMultiscatterMode::DeltaFluxV2
                                        | CloudMultiscatterMode::DeltaFluxV3
                                )
                            {
                                self.gpu_clouds = false;
                                self.parity_pending = false;
                            }
                            ui.weak(match cloud_transport {
                                CloudMultiscatterMode::LegacyOctaves => {
                                    "Established bright-anvil octave transport; shipped default."
                                }
                                CloudMultiscatterMode::SingleScatter => {
                                    "Direct single scattering only; dimmer diagnostic path."
                                }
                                CloudMultiscatterMode::DeltaFluxV1 => {
                                    "Research Stage-2 higher-order closure; CPU-only and opt-in."
                                }
                                CloudMultiscatterMode::DeltaFluxV2 => {
                                    "Brightness-neutral Stage-2 P1 upper-escape directionality; CPU-only and opt-in."
                                }
                                CloudMultiscatterMode::DeltaFluxV3 => {
                                    "Stage-2 source with bounded second-order angular memory; reduces thin-cloud isotropic overfill without an exposure change; CPU-only and opt-in."
                                }
                            });
                            ui.add(
                                egui::Slider::new(
                                    &mut self.cloud_optical_depth_scale,
                                    0.0..=4.0,
                                )
                                .text("Cloud optical-depth scale")
                                .fixed_decimals(2),
                            )
                            .on_hover_text(
                                "Applied consistently to visible cloud extinction, sunlight \
                                 attenuation, and shadows. The shipped 0.15 is the owner's \
                                 cross-file visual calibration; 1.00 uses model \
                                 extinction unchanged, and 0 disables visible cloud extinction. \
                                 This does not change IR or derived products.",
                            );
                            let nssl_optics_response = ui.checkbox(
                                &mut self.nssl_native_cloud_optics,
                                "NSSL native particle optics (experimental)",
                            )
                            .on_hover_text(
                                "WRF MP_PHYSICS=18 only: derive per-cell effective radii from \
                                 NSSL's saved mass, number, and graupel/hail volume moments, \
                                 including hail. Invalid cells fall back independently to the \
                                 fixed v0.1.6 table. Off is exact legacy fixed optics. The first \
                                 render after switching uses a separate cache and must re-ingest \
                                 the source. Visible mode only; thermal/GeoColor/Sandwich keep \
                                 fixed optics so their IR mass recovery stays self-consistent.",
                            );
                            if nssl_optics_response.changed() && self.nssl_native_cloud_optics {
                                self.hrrr_thompson_native_cloud_optics = false;
                            }
                            let thompson_optics_response = ui
                                .checkbox(
                                    &mut self.hrrr_thompson_native_cloud_optics,
                                    "HRRR Thompson native particle optics (experimental)",
                                )
                                .on_hover_text(
                                    "HRRR native GRIB only: use saved NCONCD/NCCICE/SPNCR number \
                                     moments for liquid, ice, and rain; the official Field/Thompson \
                                     diagnostics for snow and graupel; and per-cell fixed fallback \
                                     for invalid moments. A versioned cache keeps fixed output \
                                     untouched. Visible mode only because IR mass recovery remains \
                                     tied to fixed radii.",
                                );
                            if thompson_optics_response.changed()
                                && self.hrrr_thompson_native_cloud_optics
                            {
                                self.nssl_native_cloud_optics = false;
                            }
                            if ui
                                .add_enabled(
                                    (self.cloud_optical_depth_scale
                                        - clouds::DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE)
                                        .abs()
                                        > f32::EPSILON,
                                    egui::Button::new("Reset scale to shipped 0.15"),
                                )
                                .on_hover_text(
                                    "Restore the owner-selected shipped calibration.",
                                )
                                .clicked()
                            {
                                self.cloud_optical_depth_scale =
                                    clouds::DEFAULT_CLOUD_OPTICAL_DEPTH_SCALE;
                            }
                            if ui
                                .add_enabled(
                                    (self.cloud_optical_depth_scale - 1.0).abs()
                                        > f32::EPSILON,
                                    egui::Button::new("Use unscaled 1.00"),
                                )
                                .on_hover_text(
                                    "Use the model-derived extinction without the shipped scale.",
                                )
                                .clicked()
                            {
                                self.cloud_optical_depth_scale = 1.0;
                            }
                            if ui
                                .checkbox(
                                    &mut self.feather_exposed_domain_edges,
                                    "Feather exposed domain edges",
                                )
                                .on_hover_text(
                                    "Presentation experiment for finite regional WRF domains: \
                                     fade finished visible cloud extinction over the existing \
                                     outer 4% band when the camera reveals ground or sky beyond \
                                     the model boundary, even at 0% margin. On is the shipped \
                                     v0.1.5 preset; Off is the exact prior margin-gated behavior. \
                                     Interior clouds, raw bands, \
                                     IR/WV, and derived fields are unchanged. CPU-only until the \
                                     GPU shader has an exact twin.",
                                )
                                .changed()
                                && self.feather_exposed_domain_edges
                            {
                                self.gpu_clouds = false;
                                self.parity_pending = false;
                            }
                            ui.checkbox(&mut self.beer_powder, "Beer-powder")
                                .on_hover_text(
                                    "Schneider sugar-powder darkening of the sun term \
                                     (stylization). OFF by default in M5 — the octaves supply \
                                     the forward-scatter buildup it used to fake, so leaving it \
                                     on double-darkens.",
                                );
                            if ui
                                .checkbox(&mut self.granulation, "Granulation (sub-grid detail)")
                                .on_hover_text(
                                    "Edge-erosion detail noise: carves the unresolved sub-km \
                                     texture of boundary-layer cumulus (Worley octaves, \
                                     subtract-only — never adds cloud). Amplitude follows the \
                                     model grid: near-neutral on a 250 m run, strong on 2-3 km. \
                                     Ice anvils/cirrus and thermal IR are untouched. This \
                                     physical control is CPU-only.",
                                )
                                .changed()
                                && self.granulation
                            {
                                // The current WGSL preview has no granulation field. Do not
                                // leave a now-inapplicable preview silently checked.
                                self.gpu_clouds = false;
                                self.parity_pending = false;
                            }
                            ui.checkbox(
                                &mut self.topdown_stratiform_regularization,
                                "Top-down stratiform reconstruction",
                            )
                            .on_hover_text(
                                "Experimental v0.1.6 top-down observation operator: applies a \
                                 bounded, area-OD-conserving 5x5 column reconstruction only to \
                                 broad low/liquid stratiform cloud. It suppresses native-grid \
                                 HRRR rings while preserving each column's vertical/phase structure \
                                 and excluding high/convective cores. Geostationary, raw bands, \
                                 thermal and derived products are unchanged. Default off.",
                            );
                            if ui
                                .checkbox(
                                    &mut self.topdown_cloud_footprint,
                                    "Top-down cloud footprint",
                                )
                                .on_hover_text(
                                    "Experimental display-only observation footprint for native \
                                     top-down visible imagery. It applies a bounded seven-tap \
                                     sigma~=1.225 px filter to the PRE-TONEMAP cloud radiance \
                                     residual (cloud shadow, attenuation and in-scatter), then \
                                     adds the unchanged sharp terrain/base map and tonemaps once. \
                                     Geostationary, raw bands, thermal, derived, cloud-layer and \
                                     perspective products are unchanged. CPU-only; default off.",
                                )
                                .changed()
                                && self.topdown_cloud_footprint
                            {
                                self.gpu_clouds = false;
                                self.parity_pending = false;
                            }
                            ui.checkbox(
                                &mut self.topdown_shadow_antialias,
                                "Top-down shadow anti-aliasing",
                            )
                            .on_hover_text(
                                "Default-on display filter for the top-down ground cloud-shadow \
                                 map. It reduces fixed-grid dash/ring aliasing without blurring \
                                 the cloud radiance or terrain. Turn it off for an exact raw \
                                 shadow-map diagnostic. Geostationary and quantitative products \
                                 are unchanged; Sensor Fast Gray temporarily forces it off and \
                                 reports that substitution.",
                            );
                            ui.label("Steps:");
                            egui::ComboBox::from_id_salt("stepq")
                                .selected_text(match self.step_quality {
                                    StepQuality::Interactive => "Interactive (192)",
                                    StepQuality::Offline => "Offline (384)",
                                })
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut self.step_quality,
                                        StepQuality::Offline,
                                        "Offline (384, full quality)",
                                    );
                                    ui.selectable_value(
                                        &mut self.step_quality,
                                        StepQuality::Interactive,
                                        "Interactive (192, faster)",
                                    );
                                });
                        });
                    });
                    if self.clouds_enabled {
                        let fraction_note = if self.fractional_clouds {
                            if let Some(count) = self
                                .fractional_cloud_mode
                                .deterministic_subcolumn_count()
                            {
                                format!(
                                    "Model cloud fraction: deterministic {count}-member fixed-stratified CPU reference (about {count}x march cost)."
                                )
                            } else {
                                "Model cloud fraction: effective-OD CPU path (missing fields fall back safely)."
                                    .to_string()
                            }
                        } else {
                            "Legacy cloud coverage: each non-zero cloudy cell is horizontally full."
                                .to_string()
                        };
                        ui.weak(fraction_note);
                    }
                    ui.colored_label(
                        egui::Color32::from_rgb(120, 190, 235),
                        "Use GPU Render in the top bar for a one-click fast preview. It needs no \
                         manual setup, reports every temporary difference, and leaves saved \
                         settings unchanged.",
                    );
                    // EXPERIMENTAL GPU cloud pass (the M5-GPU activation): Geostationary
                    // or Top-down Visible clouds-on; the CPU composite stays the shipping default
                    // and the ONLY stored-frame path.
                    egui::CollapsingHeader::new("Advanced GPU controls")
                        .default_open(false)
                        .show(ui, |ui| {
                    let gpu_tonemap_ok = (self.cloud_softclip
                        - CLOUD_SOFTCLIP_KNEE as f32)
                        .abs()
                        <= f32::EPSILON
                        && (self.cloud_highlight_max - RHO_HIGHLIGHT_MAX as f32).abs()
                            <= f32::EPSILON;
                    let gpu_land_appearance_ok =
                        gpu_land_appearance_compatible(self.land_appearance_config());
                    let gpu_applicable = self.gpu.is_some()
                        && !self.science_cloud_f16
                        && matches!(self.render_mode, RenderMode::Visible)
                        && self.view != StudioView::Perspective
                        && self.clouds_enabled
                        // The current WGSL path supports the true-color correction
                        // toggle, but not per-pixel terrain elevation, fractional
                        // cloud subcolumns, or granulation.
                        && !self.terrain_atmosphere
                        && gpu_fractional_preview_compatible(self.fractional_clouds)
                        && gpu_granulation_preview_compatible(self.granulation)
                        && gpu_topdown_stratiform_preview_compatible(
                            self.view == StudioView::TopDownMap,
                            self.topdown_stratiform_regularization,
                        )
                        && gpu_topdown_cloud_footprint_preview_compatible(
                            self.view == StudioView::TopDownMap,
                            self.topdown_cloud_footprint,
                        )
                        && gpu_exposed_edge_feather_compatible(
                            self.feather_exposed_domain_edges,
                        )
                        && !self.delta_flux_clouds
                        && !self.delta_flux_v2_clouds
                        && !self.delta_flux_v3_clouds
                        && gpu_tonemap_ok
                        && gpu_land_appearance_ok;
                    // egui shows NO hover text on disabled widgets, so a greyed GPU
                    // cluster was unexplained (owner-reported) — name the unmet
                    // conditions inline and on the disabled-hover instead.
                    let gpu_hint = if self.gpu.is_none() {
                        format!(
                            "Manual GPU toggle unavailable: {}",
                            self.gpu_error
                                .as_deref()
                                .unwrap_or("no wgpu GPU device is available")
                        )
                    } else {
                        let mut unmet: Vec<String> = Vec::new();
                        if !matches!(self.render_mode, RenderMode::Visible) {
                            unmet.push(format!(
                                "Mode is {}; requires Visible",
                                self.render_mode.label()
                            ));
                        }
                        if self.view == StudioView::Perspective {
                            unmet.push(format!(
                                "View is {}; requires Geostationary or Top-down",
                                self.view.label()
                            ));
                        }
                        if !self.clouds_enabled {
                            unmet.push("Clouds is Off; requires On".to_string());
                        }
                        if self.terrain_atmosphere {
                            unmet.push("Terrain-height atmosphere is On; requires Off".to_string());
                        }
                        if self.fractional_clouds {
                            unmet.push("Use model cloud fraction is On; requires Off".to_string());
                        }
                        if self.granulation {
                            unmet.push("Granulation is On; requires Off".to_string());
                        }
                        if !gpu_topdown_stratiform_preview_compatible(
                            self.view == StudioView::TopDownMap,
                            self.topdown_stratiform_regularization,
                        ) {
                            unmet.push("Top-down stratiform reconstruction off".to_string());
                        }
                        if !gpu_topdown_cloud_footprint_preview_compatible(
                            self.view == StudioView::TopDownMap,
                            self.topdown_cloud_footprint,
                        ) {
                            unmet.push("Top-down cloud footprint off".to_string());
                        }
                        if self.feather_exposed_domain_edges {
                            unmet.push(
                                "Feather exposed domain edges is On; requires Off".to_string(),
                            );
                        }
                        if self.delta_flux_clouds
                            || self.delta_flux_v2_clouds
                            || self.delta_flux_v3_clouds
                        {
                            unmet.push(
                                "Delta-flux v1/v2b/v3 transport is CPU-only; requires Legacy octaves \
                                 or Single scatter"
                                    .to_string(),
                            );
                        }
                        if !gpu_tonemap_ok {
                            unmet.push(
                                "custom highlight knee/ceiling; requires shipped values"
                                    .to_string(),
                            );
                        }
                        if !gpu_land_appearance_ok {
                            unmet.push(
                                "current land appearance is not implemented in WGSL".to_string(),
                            );
                        }
                        if unmet.is_empty() {
                            "Manual GPU toggle ready for the current controls.".to_string()
                        } else {
                            format!("Manual GPU toggle blocked: {}.", unmet.join("; "))
                        }
                    };
                    ui.horizontal_wrapped(|ui| {
                        ui.add_enabled_ui(gpu_applicable, |ui| {
                            ui.checkbox(
                                &mut self.gpu_clouds,
                                "GPU clouds on regular Render (manual)",
                            )
                                .on_hover_text(
                                    "Render the DISPLAYED frame through the GPU cloud pass \
                                     (clouds.wgsl, Interactive sun schedule) instead of the CPU \
                                     composite — the fast live preview. Stored frames and \
                                     sequence renders ALWAYS use the CPU path. Granulation routes \
                                     to CPU. Other known preview divergences: terrain shadows / \
                                     ambient aperture / per-pixel wind / snow use flat-open \
                                     defaults; hardware-trilinear volume sampling; f32 math. \
                                     Geostationary or Top-down Visible mode only.",
                                )
                                .on_disabled_hover_text(gpu_hint.clone());
                            if ui
                                .add_enabled(
                                    !self.busy && self.can_render(),
                                    egui::Button::new("GPU parity check"),
                                )
                                .on_hover_text(
                                    "Render the current scene BOTH ways (CPU reference at \
                                     GPU-comparable settings + the GPU pass), log the \
                                     per-channel mean/p95/max |delta| and show a delta \
                                     heatmap. The GPU frame is displayed; nothing is stored.",
                                )
                                .on_disabled_hover_text(if gpu_applicable {
                                    let blockers = self.gpu_render_action_blockers();
                                    if blockers.is_empty() {
                                        "GPU parity check is ready.".to_string()
                                    } else {
                                        format!(
                                            "GPU parity check unavailable: {}.",
                                            blockers.join("; ")
                                        )
                                    }
                                } else {
                                    gpu_hint.clone()
                                })
                                .clicked()
                            {
                                self.parity_pending = true;
                                self.start_render(ctx);
                            }
                        });
                        ui.colored_label(
                            if gpu_applicable {
                                egui::Color32::from_rgb(135, 205, 150)
                            } else {
                                egui::Color32::from_rgb(235, 165, 120)
                            },
                            gpu_hint,
                        );
                    });
                    // The last parity report: numbers + the delta heatmap (black =
                    // identical; brighter = larger delta, gain 4x).
                    let mut dismiss_parity = false;
                    if let Some(report) = &self.parity {
                        ui.separator();
                        ui.label(format!("GPU parity: {}", report.summary));
                        let size = report.texture.size_vec2();
                        let scale = (360.0 / size.x.max(1.0)).min(1.0);
                        ui.add(egui::Image::new(&report.texture).fit_to_exact_size(size * scale))
                            .on_hover_text(
                                "Delta heatmap: black = identical, brighter/warmer = larger \
                                 CPU-vs-GPU delta (gain 4x — a 64-count delta saturates).",
                            );
                        dismiss_parity = ui.button("Dismiss parity report").clicked();
                    }
                    if dismiss_parity {
                        self.parity = None;
                    }
                        });
                });

            // Ground / Blue Marble.
            egui::CollapsingHeader::new("Ground / Blue Marble")
                .default_open(false)
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Ground month:");
                        let month_label = if self.bm_month_override == 0 {
                            "Auto (season)".to_string()
                        } else {
                            bluemarble::month_abbr(self.bm_month_override).to_string()
                        };
                        egui::ComboBox::from_id_salt("bmmonth")
                            .selected_text(month_label)
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut self.bm_month_override,
                                    0,
                                    "Auto (season)",
                                );
                                for m in 1..=12u32 {
                                    ui.selectable_value(
                                        &mut self.bm_month_override,
                                        m,
                                        bluemarble::month_abbr(m),
                                    );
                                }
                            })
                            .response
                            .on_hover_text(
                                "Auto = the day-of-year blend of the two bracketing Blue Marble \
                                 monthly composites, so the ground matches the WRF run's season. \
                                 A specific month FORCES that composite (a what-if, e.g. summer \
                                 ground under a winter storm).",
                            );
                        ui.checkbox(&mut self.bm_allow_download, "Download missing months")
                            .on_hover_text(
                                "Lazily fetch the 1-2 months a render needs (GitHub release, \
                                 then NASA), SHA-256 verified, into the cache. Off = use only \
                                 cached months or the bundled 8 km offline fallback.",
                            );
                        if ui
                            .add_enabled(
                                !self.pack_busy,
                                egui::Button::new("Download full-year pack"),
                            )
                            .on_hover_text(
                                "Download all 12 monthly 2 km composites (~270 MB) now, so later \
                                 renders never wait on a download.",
                            )
                            .clicked()
                        {
                            self.start_pack_download(ctx);
                        }
                        if self.pack_busy {
                            ui.spinner();
                        }
                    });
                });
        }

        if mode.is_thermal() {
            // Enhancement (IR / WV colour palette) — thermal modes only. Changing it re-colours
            // the current frame live (no re-march) via `reenhance_ir`.
            let is_wv = matches!(mode, RenderMode::WaterVapor(_));
            egui::CollapsingHeader::new("Enhancement")
                .default_open(true)
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(if is_wv {
                            "WV enhancement:"
                        } else {
                            "IR enhancement:"
                        });
                        egui::ComboBox::from_id_salt("irenh")
                            .selected_text(self.ir_enhancement.label())
                            .show_ui(ui, |ui| {
                                for e in IrEnhancement::ALL {
                                    ui.selectable_value(&mut self.ir_enhancement, e, e.label());
                                }
                            })
                            .response
                            .on_hover_text(if is_wv {
                                "The colour enhancement applied to the Kelvin WV BT plane. \
                                 CIMSS = the classic WV moisture palette (white cold/moist -> \
                                 blue -> brown warm/dry); Natural and Legacy grayscale both use \
                                 the readable WV-scaled inverted-gray range. \
                                 Changing it re-colours instantly (no re-render)."
                            } else {
                                "The display enhancement applied to the Kelvin BT plane. CIMSS Style \
                                 is the recommended false-color isotherm display. Natural is NOAA's \
                                 continuous heritage Band-13 grayscale. Changing it re-colours \
                                 instantly (no re-render)."
                            });
                    });
                });
        }

        if let Some(current_field) = mode.derived_field() {
            // Field (derived scalar field) — derived modes only. Writes the same `render_mode`
            // the top-strip Mode picker does, so the two stay in sync.
            egui::CollapsingHeader::new("Field")
                .default_open(true)
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Derived field:");
                        egui::ComboBox::from_id_salt("derivedfield")
                            .selected_text(current_field.label())
                            .show_ui(ui, |ui| {
                                for f in DerivedField::ALL {
                                    ui.selectable_value(
                                        &mut self.render_mode,
                                        RenderMode::Derived(f),
                                        f.label(),
                                    );
                                }
                            })
                            .response
                            .on_hover_text(
                                "Which per-column scalar field to compute (precipitable water / \
                                 cloud-top temperature / cloud optical depth), shown with the \
                                 basic studio colormap. The RAW values are the `import simsat` \
                                 plotting deliverable.",
                            );
                    });
                });
        }

        // Output (sat store) — all modes.
        egui::CollapsingHeader::new("Output (sat store)")
            .default_open(false)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label("Sat store root:");
                    ui.monospace(self.store_root.display().to_string());
                    if ui.button("Change...").clicked()
                        && let Some(dir) = rfd::FileDialog::new()
                            .set_title("Choose the SimSat sat-store root")
                            .pick_folder()
                    {
                        self.store_root = dir;
                    }
                });
                ui.label(
                    egui::RichText::new(
                        "In BowEcho, set the Satellite store dir to the path above to see the \
                         'simsat' run.",
                    )
                    .weak(),
                );
            });
    }

    /// The batch-progress bar + the animation timeline (frame slider / play-pause / loop
    /// toggle / fps + frame-cap knob), shown once a loop is rendering or rendered.
    fn loop_ui(&mut self, ui: &mut egui::Ui) {
        if let Some(b) = &self.batch {
            let frac = b.done as f32 / b.total.max(1) as f32;
            let per = if b.done > 0 {
                b.total_frame_ms / b.done as u64
            } else {
                0
            };
            let (done, total) = (b.done, b.total);
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                ui.add(
                    egui::ProgressBar::new(frac)
                        .desired_width(220.0)
                        .show_percentage(),
                );
                ui.label(format!("Frame {done}/{total}  ~{per} ms/frame"));
            });
        }

        let n = self
            .loop_state
            .as_ref()
            .map(|ls| ls.frames.len())
            .unwrap_or(0);
        if n == 0 {
            return;
        }
        let maxi = n - 1;
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            let playing = self
                .loop_state
                .as_ref()
                .map(|ls| ls.playing)
                .unwrap_or(false);
            if ui.button(if playing { "Pause" } else { "Play" }).clicked()
                && let Some(ls) = &mut self.loop_state
                && ls.frames.len() >= 2
            {
                ls.playing = !ls.playing;
                ls.accumulator = 0.0;
                // Restart a finished non-looping play from the top.
                if ls.playing && !ls.looping && ls.current >= ls.frames.len() - 1 {
                    ls.current = 0;
                }
            }
            if let Some(ls) = &mut self.loop_state {
                ui.checkbox(&mut ls.looping, "Loop")
                    .on_hover_text("Wrap to the first frame at the end (a satellite loop).");
            }
            ui.add(egui::Slider::new(&mut self.play_fps, 1.0..=30.0).text("fps"));
            ui.separator();
            let mut cur = self
                .loop_state
                .as_ref()
                .map(|ls| ls.current.min(maxi))
                .unwrap_or(0);
            if ui
                .add(egui::Slider::new(&mut cur, 0..=maxi).text("frame"))
                .changed()
                && let Some(ls) = &mut self.loop_state
            {
                ls.current = pipeline::clamp_scrub(cur as i64, ls.frames.len());
                ls.playing = false; // scrubbing pauses
            }
            if ui.button("|<").on_hover_text("First frame").clicked()
                && let Some(ls) = &mut self.loop_state
            {
                ls.current = 0;
                ls.playing = false;
            }
            if ui.button("<").clicked()
                && let Some(ls) = &mut self.loop_state
            {
                ls.current = ls.current.saturating_sub(1);
                ls.playing = false;
            }
            if ui.button(">").clicked()
                && let Some(ls) = &mut self.loop_state
            {
                ls.current = (ls.current + 1).min(maxi);
                ls.playing = false;
            }
            if ui.button(">|").on_hover_text("Last frame").clicked()
                && let Some(ls) = &mut self.loop_state
            {
                ls.current = maxi;
                ls.playing = false;
            }
        });

        if let Some(ls) = &self.loop_state {
            let cur = ls.current.min(ls.frames.len().saturating_sub(1));
            if let Some(f) = ls.frames.get(cur) {
                let run = ls
                    .store_run
                    .as_ref()
                    .map(|r| format!("  store: simsat/{r}"))
                    .unwrap_or_default();
                let cap = if ls.capped {
                    format!(
                        "  (retained {}/{}; full run in store)",
                        ls.frames.len(),
                        ls.total_rendered
                    )
                } else {
                    String::new()
                };
                ui.label(
                    egui::RichText::new(format!(
                        "Frame {}/{}  {}  {}{}{}",
                        cur + 1,
                        ls.frames.len(),
                        f.label,
                        f.summary,
                        run,
                        cap
                    ))
                    .weak(),
                );
            }
        }

        ui.horizontal_wrapped(|ui| {
            ui.label("In-memory frame cap:");
            ui.add(egui::Slider::new(&mut self.frame_cap, 8..=480))
                .on_hover_text(
                    "Full-res textures retained for instant in-studio playback. Beyond this the \
                     batch still renders + writes every frame to the store (BowEcho plays the \
                     full run); only in-studio scrubbing is bounded.",
                );
        });
    }

    /// Kick off the full-year Blue Marble pack download on a below-normal worker,
    /// streaming status lines back over a simple channel (drained in `drain_pack`).
    fn start_pack_download(&mut self, ctx: &egui::Context) {
        if self.pack_busy {
            return;
        }
        let (tx, rx) = channel::<String>();
        self.pack_rx = Some(rx);
        self.pack_busy = true;
        self.logline("Downloading full-year Blue Marble pack (~270 MB)...");
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            simsat::platform::lower_worker_thread_priority();
            let cache = ingest::default_cache_dir();
            let manifest = asset_pack::embedded_manifest();
            let mut status = |s: String| {
                let _ = tx.send(s);
                ctx.request_repaint();
            };
            match asset_pack::download_full_year(&cache, &manifest, &mut status) {
                Ok(n) => status(format!("Full-year pack ready: {n}/12 months at 2 km.")),
                Err(e) => status(format!("Full-year pack download failed: {e}")),
            }
            // tx drops here -> the UI sees Disconnected and clears `pack_busy`.
            ctx.request_repaint();
        });
    }

    /// Drain the full-year-pack worker's status lines; clear `pack_busy` when the worker
    /// thread ends (the channel disconnects after its final message).
    fn drain_pack(&mut self) {
        let mut msgs = Vec::new();
        let mut done = false;
        if let Some(rx) = &self.pack_rx {
            loop {
                match rx.try_recv() {
                    Ok(s) => msgs.push(s),
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        done = true;
                        break;
                    }
                }
            }
        }
        for s in msgs {
            // The pack worker streams plain strings; route its failure lines into
            // the sticky error surface (a lightweight but honest heuristic — the
            // worker's failure messages all contain "failed").
            if s.contains("failed") {
                self.logerr(s);
            } else {
                self.logline(s);
            }
        }
        if done {
            self.pack_rx = None;
            self.pack_busy = false;
        }
    }

    fn drain_worker(&mut self, ctx: &egui::Context) {
        // Drain the channel into a buffer first so message handling can freely
        // borrow `&mut self` (the receiver borrow ends here).
        let mut msgs = Vec::new();
        let mut disconnected = false;
        if let Some(rx) = &self.worker_rx {
            loop {
                match rx.try_recv() {
                    Ok(m) => msgs.push(m),
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
        }

        let mut prepared = None;
        for m in msgs {
            match m {
                WorkerMsg::Status(s) => self.logline(s),
                WorkerMsg::Prepared(p) => prepared = Some(*p),
                WorkerMsg::Error(e) => {
                    self.busy = false;
                    self.busy_since = None;
                    self.worker_rx = None;
                    self.logerr(format!("Render failed: {e}"));
                }
                WorkerMsg::BatchFrame {
                    index,
                    total,
                    prep,
                    prep_ms,
                } => {
                    self.accept_batch_frame(ctx, index, total, *prep, prep_ms);
                }
                WorkerMsg::BatchError { index, message } => {
                    if let Some(b) = &mut self.batch {
                        b.errors += 1;
                    }
                    self.logerr(format!("Frame {} failed: {message}", index + 1));
                }
                WorkerMsg::BatchDone {
                    rendered,
                    cancelled,
                } => self.finish_batch(rendered, cancelled),
            }
        }
        if let Some(prep) = prepared {
            self.worker_rx = None;
            self.finish_prepared(ctx, prep);
        } else if disconnected && self.batch.is_none() {
            self.worker_rx = None;
        }
    }
}

impl eframe::App for SimSatStudioApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.drain_worker(&ctx);
        self.drain_pack();
        self.drain_export();
        if self.busy || self.pack_busy || self.export_busy {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        // Advance loop playback over the prerendered textures (pure fps/loop math). When
        // playing, keep repainting so the animation runs at the chosen fps.
        let dt = ctx.input(|i| i.stable_dt);
        if self.tick_playback(dt) {
            ctx.request_repaint();
        }

        // Drag-and-drop: classify the dropped paths (pure, node-tested) and open.
        // Drops are REJECTED while a render is in flight (the source drives the
        // in-flight jobs); path-less drops are ignored.
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        if !dropped.is_empty() {
            if self.busy {
                self.logerr(
                    "Busy rendering — the dropped files were ignored. Drop them again when \
                     the render finishes.",
                );
            } else {
                match pipeline::classify_dropped(&dropped, &|p: &Path| p.is_dir()) {
                    Some(pipeline::DropOpen::Wrfout(p)) => self.open_wrfout(p),
                    Some(pipeline::DropOpen::CachedRun(p)) => self.open_cached_run(p),
                    Some(pipeline::DropOpen::Sequence(v)) => self.open_sequence(v),
                    None => {}
                }
            }
        }

        egui::Panel::top("controls").show_inside(ui, |ui| {
            ui.add_space(4.0);
            // Slim top bar (always one row) + the context-driven Advanced drawer below it.
            self.top_strip(ui, &ctx);
            // STICKY error banner: set by any error, cleared ONLY by Dismiss or a
            // subsequent successful render — a later status line never hides it.
            if let Some(err) = self.log.last_error().map(str::to_string) {
                ui.add_space(2.0);
                egui::Frame::new()
                    .fill(egui::Color32::from_rgb(70, 20, 20))
                    .inner_margin(egui::Margin::symmetric(8, 4))
                    .corner_radius(3.0)
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.colored_label(egui::Color32::from_rgb(255, 150, 150), &err);
                            if ui.small_button("Dismiss").clicked() {
                                self.log.dismiss_error();
                            }
                        });
                    });
            }
            self.calibration_epoch_banner(ui);
            self.recommended_presets(ui);
            self.context_note(ui);
            self.size_gate_confirms(ui);
            ui.separator();
            let settings_height = settings_scroll_max_height(ctx.content_rect().height());
            egui::CollapsingHeader::new("Settings (all controls)")
                .default_open(false)
                .show(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .id_salt("studio-settings-scroll")
                        .max_height(settings_height)
                        .show(ui, |ui| self.advanced_drawer(ui, &ctx));
                });
            ui.add_space(2.0);
        });

        // Live IR re-enhancement: if the IR picker changed since the last render, recolour
        // the current BT plane in place (no re-march). A no-op when unchanged / not IR.
        self.reenhance_ir(&ctx);

        egui::Panel::bottom("status").show_inside(ui, |ui| {
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                // The Log toggle owns the right edge; the status text fills the rest.
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.toggle_value(&mut self.show_log, "Log")
                        .on_hover_text("Show the render/status log history (errors highlighted).");
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        // Busy phases append a live elapsed-seconds counter so a long
                        // march visibly makes progress.
                        let status_line = match (self.busy, self.busy_since) {
                            (true, Some(t0)) => {
                                format!("{} ({:.0} s)", self.status, t0.elapsed().as_secs_f32())
                            }
                            _ => self.status.clone(),
                        };
                        ui.label(status_line);
                    });
                });
            });
            if let Some(err) = &self.gpu_error {
                ui.colored_label(egui::Color32::from_rgb(230, 90, 90), err);
            }
            if self.show_log {
                ui.separator();
                egui::ScrollArea::vertical()
                    .max_height(150.0)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for entry in self.log.entries() {
                            match entry.level {
                                pipeline::LogLevel::Error => ui.colored_label(
                                    egui::Color32::from_rgb(255, 140, 140),
                                    &entry.message,
                                ),
                                pipeline::LogLevel::Info => {
                                    ui.label(egui::RichText::new(&entry.message).weak())
                                }
                            };
                        }
                    });
            }
            ui.add_space(2.0);
        });

        // The batch progress + loop timeline live in a bottom bar shown only while a batch is
        // rendering or a prerendered loop exists (added AFTER the status panel so it sits just
        // above it). The pan/zoom viewport in the central panel plays the loop's current frame.
        if self.batch.is_some() || self.loop_state.is_some() {
            egui::Panel::bottom("timeline").show_inside(ui, |ui| {
                ui.add_space(2.0);
                self.loop_ui(ui);
                ui.add_space(2.0);
            });
        }

        egui::CentralPanel::default().show_inside(ui, |ui| {
            // Snapshot the frame's display info (owned) so the borrow of `self.rendered`
            // is released before the Fit button + scroll/drag below mutate the view state
            // (avoids an is_some() + unwrap() that clippy flags).
            // Prefer the animation loop's current frame when a loop is present; else the
            // single rendered frame. Both feed the SAME pan/zoom display viewport below.
            let loop_info = self
                .loop_state
                .as_ref()
                .filter(|ls| !ls.frames.is_empty())
                .map(|ls| {
                    let cur = ls.current.min(ls.frames.len() - 1);
                    let f = &ls.frames[cur];
                    let header = format!(
                        "Loop {}/{}  {}  {}  {}",
                        cur + 1,
                        ls.frames.len(),
                        f.label,
                        f.summary,
                        if ls.is_ir {
                            band_display(ls.ir_band)
                        } else {
                            "visible".to_string()
                        },
                    );
                    (
                        header,
                        false,
                        f.texture.id(),
                        f.width as f32,
                        f.height as f32,
                    )
                });
            let frame_info = loop_info.or_else(|| {
                self.rendered.as_ref().map(|state| {
                    // View label: the satellite name for the from-space product,
                    // "Top-down map" for the map-registered product, or
                    // "Perspective (3-D)" for the orbit view (honest header).
                    let view_label = match state.view_mode {
                        StudioView::TopDownMap => "Top-down map".to_string(),
                        StudioView::Perspective => "Perspective (3-D)".to_string(),
                        StudioView::Geostationary => state.satellite.label().to_string(),
                    };
                    let dims = format!(
                        "{}  {}x{} {}{}",
                        view_label,
                        state.rendered.width,
                        state.rendered.height,
                        state.resolution.label(),
                        if state.res_clamped {
                            " [clamped to cap]"
                        } else {
                            ""
                        },
                    );
                    // Derived-field header shows the field + value range; IR/WV shows the band +
                    // enhancement (thermal — no sun); visible shows the sun elevation + Blue
                    // Marble status.
                    let header = if let Some((field, s)) = &state.derived {
                        let unit = field.units();
                        format!(
                            "{dims}  {}  range {:.2}..{:.2}{}{}",
                            field.label(),
                            s.min,
                            s.max,
                            if unit.is_empty() { "" } else { " " },
                            unit,
                        )
                    } else if state.ir_bt.is_some() {
                        format!(
                            "{dims}  {}  {} enhancement",
                            band_display(state.ir_band),
                            state.ir_enhancement.label()
                        )
                    } else {
                        format!(
                            "{dims}  sun {:.1} deg  {}",
                            state.center_sun_elev,
                            if state.season_line.is_empty() {
                                state.bm_status.chip_label()
                            } else {
                                state.season_line.clone()
                            },
                        )
                    };
                    (
                        header,
                        state.sun_override && state.ir_bt.is_none() && state.derived.is_none(),
                        state.texture.id(),
                        state.rendered.width as f32,
                        state.rendered.height as f32,
                    )
                })
            });
            if let Some((header, show_override, tex_id, w, h)) = frame_info {
                ui.horizontal(|ui| {
                    ui.label(header);
                    // Fake-sun override is a NON-PHYSICAL what-if: keep it unmistakable.
                    if show_override {
                        ui.colored_label(
                            egui::Color32::from_rgb(230, 170, 60),
                            "sun OVERRIDE (what-if, non-physical)",
                        );
                    }
                    ui.separator();
                    if ui
                        .button("Fit")
                        .on_hover_text("Reset zoom + pan to fit the window")
                        .clicked()
                    {
                        self.view_zoom = 1.0;
                        self.view_pan = egui::Vec2::ZERO;
                    }
                    ui.label(format!("zoom {:.1}x", self.view_zoom))
                        .on_hover_text(
                            "Scroll to zoom (cursor-centred), drag to pan; 1.0x = fit to \
                             window. DISPLAY zoom of the already-rendered native frame: it \
                             reveals real detail down to ~1:1 pixel, beyond which it is pure \
                             magnification (a data-resolution limit, not a zoom bug).",
                        );
                });

                // ── DISPLAY-side pan + zoom viewport (no re-render, no engine change).
                // Scroll = zoom centred on the cursor; drag = pan; the Fit button resets;
                // zoom/pan reset to fit on each NEW render. The frame renders at WRF NATIVE
                // resolution (~1 output pixel per grid cell), so zooming reveals real detail
                // only down to about 1:1 pixel — beyond that it is pure magnification (no
                // more data exists; a data-resolution limit, NOT a zoom bug). A future
                // enhancement could re-render a zoomed sub-region at higher detail (a camera
                // sub-region super-sample); that is out of scope here. LINEAR texture
                // filtering (set at load) keeps the magnification smooth. ──
                let avail = ui.available_size();
                let (rect, response) = ui.allocate_exact_size(avail, egui::Sense::click_and_drag());
                // Fit scale (viewport / image); `view_zoom` is a factor over it (1.0 = fit).
                let fit = (rect.width() / w).min(rect.height() / h).clamp(0.001, 64.0);
                // Scroll to zoom, keeping the image point under the cursor fixed.
                if response.hovered() {
                    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                    if scroll.abs() > 0.0 {
                        let old = self.view_zoom;
                        let new = (old * (1.0 + scroll * 0.0015)).clamp(1.0, MAX_VIEW_ZOOM);
                        if (new - old).abs() > f32::EPSILON {
                            if let Some(cursor) = response.hover_pos() {
                                self.view_pan = pan_after_cursor_zoom(
                                    self.view_pan,
                                    cursor - rect.center(),
                                    new / old,
                                );
                            }
                            self.view_zoom = new;
                        }
                    }
                }
                // Drag to pan.
                if response.dragged() {
                    self.view_pan += response.drag_delta();
                }
                let scale = fit * self.view_zoom;
                let img = egui::vec2(w * scale, h * scale);
                // Clamp pan so the image cannot be dragged past its own edges (centred when
                // it fits the viewport, i.e. at zoom 1.0).
                self.view_pan = clamp_pan(self.view_pan, img, rect.size());
                let image_rect = egui::Rect::from_center_size(rect.center() + self.view_pan, img);
                ui.painter().with_clip_rect(rect).image(
                    tex_id,
                    image_rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
                if response.dragged() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                } else if response.hovered() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                }
            } else if self.source.is_some() {
                ui.centered_and_justified(|ui| {
                    ui.label(
                        egui::RichText::new(
                            "No frame yet. Pick a satellite + timestep, then Render.",
                        )
                        .weak(),
                    );
                });
            } else {
                // First-run / empty state: a real call-to-action instead of a bare
                // hint — the same open actions as the Open menu, plus one-click
                // reopen of the last session's source and the drag-drop affordance.
                ui.add_space((ui.available_height() * 0.22).max(12.0));
                ui.vertical_centered(|ui| {
                    ui.heading("SimSat Studio");
                    ui.label(
                        egui::RichText::new(
                            "Physically-based simulated satellite imagery from WRF output.",
                        )
                        .weak(),
                    );
                    ui.add_space(14.0);
                    let w = 260.0;
                    if ui
                        .add_sized([w, 28.0], egui::Button::new("Open wrfout..."))
                        .clicked()
                    {
                        self.dialog_open_wrfout();
                    }
                    if ui
                        .add_sized([w, 28.0], egui::Button::new("Open cached run.json..."))
                        .clicked()
                    {
                        self.dialog_open_cached();
                    }
                    if ui
                        .add_sized([w, 28.0], egui::Button::new("Open sequence (folder)..."))
                        .on_hover_text(
                            "A directory of wrfout files, ordered by valid time into a \
                             batch-renderable loop.",
                        )
                        .clicked()
                    {
                        self.dialog_open_sequence_folder();
                    }
                    if let Some(last) = self.recent.first().cloned() {
                        ui.add_space(8.0);
                        if ui
                            .add_sized(
                                [w, 28.0],
                                egui::Button::new(format!("Reopen last: {}", last.label())),
                            )
                            .clicked()
                        {
                            self.reopen_recent(&last);
                        }
                    }
                    ui.add_space(14.0);
                    ui.label(
                        egui::RichText::new(
                            "...or drop wrfout files, a run.json, or a whole folder anywhere \
                             in this window.",
                        )
                        .weak(),
                    );
                });
            }
        });

        // Translucent drop-hover overlay: shown while files are dragged over the
        // window so the drop affordance is unmistakable.
        let hovering_files = ctx.input(|i| !i.raw.hovered_files.is_empty());
        if hovering_files {
            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("simsat-drop-overlay"),
            ));
            let rect = ctx.content_rect();
            painter.rect_filled(
                rect,
                0.0,
                egui::Color32::from_rgba_unmultiplied(20, 60, 120, 90),
            );
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                if self.busy {
                    "Busy rendering — drops are ignored until the render finishes"
                } else {
                    "Drop to open: wrfout file(s), a run.json, or a folder"
                },
                egui::FontId::proportional(22.0),
                egui::Color32::WHITE,
            );
        }

        // Settings persistence: save-on-change with a short debounce (the on_exit
        // backstop saves immediately).
        self.tick_settings_autosave(&ctx);
    }

    fn on_exit(&mut self) {
        // Crash-conscious backstop: whatever changed since the last debounce save
        // is flushed on clean shutdown (an unclean crash loses at most the last
        // ~750 ms of knob motion — the atomic temp+rename save can never corrupt).
        self.save_settings_now();
    }
}

// ── worker-side preparation (no egui, runs on the below-normal thread) ─────────

/// The M2 atmosphere + M4 cloud + M6 IR controls captured from the UI for one render.
#[derive(Debug, Clone)]
struct AtmoSettings {
    /// Output semantics selected in Studio.
    render_intent: RenderIntent,
    /// Sensor-grid navigation for geostationary renders.
    geo_navigation: GeoNavigation,
    /// Exact temporary strict-intent substitutions; persistent sliders are untouched.
    intent_adjustments: Vec<RenderIntentAdjustment>,
    /// View mode (Geostationary from-space, the top-down map-registered product, or the
    /// Perspective (3-D) orbit view). Top-down and Perspective render on the CPU.
    view_mode: StudioView,
    /// The Perspective orbit-camera params (always captured; read only when
    /// `view_mode == StudioView::Perspective`). Mapped to the engine's
    /// `PerspectiveCamera` by the pure `pipeline::orbit_to_camera`.
    orbit: pipeline::OrbitParams,
    /// Zoom-out / domain MARGIN as a FRACTION of the domain size added on each side (0.0 =
    /// the domain edge-to-edge; e.g. 0.30 = +30% of the domain span on every side). The
    /// margin renders the real earth around the domain (Blue Marble ground + clear sky, no
    /// WRF weather). Applies to both views.
    margin_frac: f64,
    /// Render mode (Visible, IR band 13, or a WV band 8/9/10). In a thermal mode the
    /// sun/exposure/multi-scatter controls do not apply (thermal — no sun input).
    render_mode: RenderMode,
    /// IR/WV enhancement (colour curve) for the BT plane in a thermal mode.
    ir_enhancement: IrEnhancement,
    /// Spectral response used for Band 13 (IR/GeoColor/Sandwich only).
    thermal_sensor: ThermalSensor,
    /// Complete-radiance ABI Band 13 spatial response. Default off.
    instrument_footprint: InstrumentFootprint,
    aod: f64,
    rh_swelling: bool,
    /// Daytime aerial-veil correction (product-facing default on; off = full path airlight).
    atmosphere_correction: bool,
    /// Clip view/sun atmospheric columns to terrain elevation (physical default on).
    terrain_atmosphere: bool,
    /// Display-only land corrections. The shipped preset enables both bounded
    /// corrections; the persisted toggles can select the exact legacy identity.
    land_appearance: LandAppearanceConfig,
    /// Default-off display experiment over the lit/view-attenuated LAND contribution.
    surface_postlight_toe: SurfacePostlightToeConfig,
    /// Separate shipped, tightly gated low-sun terrain recovery for visible displays.
    twilight_surface_recovery: TwilightSurfaceRecoveryConfig,
    output_transform: OutputTransform,
    clouds_enabled: bool,
    /// Use the source's cloud-fraction/subcolumn closure when available. The CPU
    /// renderer consumes this; `false` requests the legacy horizontally-full cells.
    fractional_clouds: bool,
    /// Effective-OD (fast/default) or deterministic 4/8/16 explicit CPU members.
    fractional_cloud_mode: FractionalCloudMode,
    /// Explicit visible-cloud higher-order transport selection. The Studio's two
    /// persisted compatibility booleans are resolved to this engine enum at capture.
    cloud_multiscatter: CloudMultiscatterMode,
    /// Visible cloud optical-depth scale. Shipped = 0.15; 1.0 is unscaled.
    cloud_optical_depth_scale: f32,
    /// Authoritative brick storage precision for this render.
    storage_profile: StorageProfile,
    /// Ingest-time WRF hydrometeor particle optics.
    cloud_optics: CloudOpticsMode,
    /// Fade finished visible clouds at camera-exposed finite-domain boundaries.
    feather_exposed_domain_edges: bool,
    beer_powder: bool,
    /// Sub-grid cloud GRANULATION (edge-erosion detail noise): when on, the visible
    /// cloud march + sun march + sun-OD map all sample the SAME dx-amplitude eroded
    /// field (clouds.rs granulation section). Thermal modes never granulate.
    granulation: bool,
    /// Top-down-only low-stratiform column optical-depth reconstruction.
    topdown_stratiform_regularization: bool,
    /// Display-only pre-tonemap footprint over the cloud radiance residual.
    topdown_cloud_footprint: bool,
    /// Default-on display filter for the top-down ground cloud-shadow field.
    topdown_shadow_antialias: bool,
    step_quality: StepQuality,
    /// EXPERIMENTAL: render a Geostationary or Top-down Visible clouds-on frame through the
    /// GPU cloud pass (Interactive schedule) instead of the CPU composite. Only the
    /// DISPLAYED single-frame render honors it; batch/sequence renders force it off.
    gpu_clouds: bool,
    /// One-shot GPU parity check: march BOTH paths and return the CPU reference so
    /// the UI can diff them (works whether or not `gpu_clouds` is on).
    parity: bool,
    /// Dedicated one-click GPU action. Unlike the persisted manual toggle, this uses
    /// a temporary compatible preview configuration and never mutates Studio controls.
    one_click_gpu_render: bool,
    /// Exact temporary differences chosen by the one-click action.
    gpu_preview_adjustments: GpuPreviewAdjustments,
    /// Display-side exposure gain (before the ABI stretch) for the CPU composite.
    exposure: f64,
    /// Sun-gated daytime surface-radiance lift (`1.0` is neutral).
    ground_gain: f64,
    /// Highlight shoulder knee (`1.0` disables the shoulder).
    cloud_softclip: f64,
    /// Physical reflectance-factor ceiling mapped to display white.
    cloud_highlight_max: f64,
    /// "Fake sun" what-if OVERRIDE: `Some((elev_deg, az_deg))` places a single uniform
    /// sun over the domain centre at that elevation/azimuth (sun at infinity) regardless
    /// of the file's time; `None` uses the real timestamp solar geometry. A deliberate,
    /// clearly-labeled NON-PHYSICAL visualization aid (see the app-struct doc-comment).
    sun_override: Option<(f64, f64)>,
    /// Seasonal Blue Marble (M7): force a single month `1..=12` for the ground (a what-if
    /// override), or `None` for the day-of-year blend of the timestep date.
    bm_month_override: Option<u32>,
    /// Lazily download missing Blue Marble months (GitHub asset URL -> NASA URL, SHA-256
    /// gated) on the render worker; `false` = cached 2 km or the vendored 8 km fallback only.
    bm_allow_download: bool,
}

/// Apply the strict Sensor Fast Gray contract to a per-render copy. Display is an
/// exact no-op; persistent Studio controls remain available when the user switches
/// back. Model fractional-cloud handling and all physical/model controls not listed
/// here are deliberately retained.
fn configure_render_intent(atmo: &mut AtmoSettings) {
    if atmo.render_intent == RenderIntent::Display {
        return;
    }
    let mut changed = Vec::new();
    if (atmo.cloud_optical_depth_scale - 1.0).abs() > f32::EPSILON {
        atmo.cloud_optical_depth_scale = 1.0;
        changed.push(RenderIntentAdjustment::CloudOpticalDepthUnscaled);
    }
    if (atmo.exposure - 1.0).abs() > f64::EPSILON {
        atmo.exposure = 1.0;
        changed.push(RenderIntentAdjustment::ExposureNeutral);
    }
    if (atmo.ground_gain - 1.0).abs() > f64::EPSILON {
        atmo.ground_gain = 1.0;
        changed.push(RenderIntentAdjustment::GroundLiftNeutral);
    }
    if atmo.land_appearance != LandAppearanceConfig::identity() {
        atmo.land_appearance = LandAppearanceConfig::identity();
        changed.push(RenderIntentAdjustment::LandAppearanceIdentity);
    }
    if !atmo.surface_postlight_toe.is_identity() {
        atmo.surface_postlight_toe = SurfacePostlightToeConfig::off();
        changed.push(RenderIntentAdjustment::SurfacePostlightToeOff);
    }
    if !atmo.twilight_surface_recovery.is_identity() {
        atmo.twilight_surface_recovery = TwilightSurfaceRecoveryConfig::off();
        changed.push(RenderIntentAdjustment::TwilightSurfaceRecoveryOff);
    }
    if atmo.feather_exposed_domain_edges {
        atmo.feather_exposed_domain_edges = false;
        changed.push(RenderIntentAdjustment::ExposedEdgeFeatherOff);
    }
    if atmo.granulation {
        atmo.granulation = false;
        changed.push(RenderIntentAdjustment::GranulationOff);
    }
    if atmo.topdown_stratiform_regularization {
        atmo.topdown_stratiform_regularization = false;
        changed.push(RenderIntentAdjustment::TopdownStratiformRegularizationOff);
    }
    if atmo.topdown_cloud_footprint {
        atmo.topdown_cloud_footprint = false;
        changed.push(RenderIntentAdjustment::TopdownCloudFootprintOff);
    }
    if atmo.topdown_shadow_antialias {
        atmo.topdown_shadow_antialias = false;
        changed.push(RenderIntentAdjustment::TopdownShadowAntialiasOff);
    }
    if atmo.atmosphere_correction {
        atmo.atmosphere_correction = false;
        changed.push(RenderIntentAdjustment::AtmosphereCorrectionOff);
    }
    if (atmo.cloud_softclip - 1.0).abs() > f64::EPSILON
        || (atmo.cloud_highlight_max - 1.0).abs() > f64::EPSILON
    {
        atmo.cloud_softclip = 1.0;
        atmo.cloud_highlight_max = 1.0;
        changed.push(RenderIntentAdjustment::HighlightShoulderIdentity);
    }
    atmo.intent_adjustments = changed;
}

/// Convert a captured render configuration into the GPU cloud shader's supported
/// preview envelope. This mutates only the per-render copy: the user's persistent
/// Studio controls remain untouched. Geostationary and Top-down retain their camera;
/// Perspective has no GPU cloud ray path and is changed to a geostationary preview.
fn configure_one_click_gpu_preview(atmo: &mut AtmoSettings) -> GpuPreviewAdjustments {
    let mut changes = GpuPreviewAdjustments::default();
    if atmo.render_mode != RenderMode::Visible {
        changes.insert(GpuPreviewAdjustments::MODE_VISIBLE);
        atmo.render_mode = RenderMode::Visible;
    }
    if atmo.instrument_footprint != InstrumentFootprint::Off {
        changes.insert(GpuPreviewAdjustments::INSTRUMENT_FOOTPRINT_OFF);
        atmo.instrument_footprint = InstrumentFootprint::Off;
    }
    if atmo.geo_navigation == GeoNavigation::GoesRAbiFixedGrid {
        changes.insert(GpuPreviewAdjustments::MODEL_SPHERE_NAVIGATION);
        atmo.geo_navigation = GeoNavigation::ModelSphere;
    }
    if atmo.view_mode == StudioView::Perspective {
        changes.insert(GpuPreviewAdjustments::VIEW_GEOSTATIONARY);
        atmo.view_mode = StudioView::Geostationary;
    }
    if !atmo.clouds_enabled {
        changes.insert(GpuPreviewAdjustments::CLOUDS_ON);
        atmo.clouds_enabled = true;
    }
    if atmo.terrain_atmosphere {
        changes.insert(GpuPreviewAdjustments::TERRAIN_ATMOSPHERE_OFF);
        atmo.terrain_atmosphere = false;
    }
    if atmo.fractional_clouds {
        changes.insert(GpuPreviewAdjustments::FRACTIONAL_CLOUDS_OFF);
        atmo.fractional_clouds = false;
        atmo.fractional_cloud_mode = FractionalCloudMode::Off;
    }
    if atmo.granulation {
        changes.insert(GpuPreviewAdjustments::GRANULATION_OFF);
        atmo.granulation = false;
    }
    if atmo.topdown_stratiform_regularization {
        changes.insert(GpuPreviewAdjustments::TOPDOWN_STRATIFORM_REGULARIZATION_OFF);
        atmo.topdown_stratiform_regularization = false;
    }
    if atmo.topdown_cloud_footprint {
        changes.insert(GpuPreviewAdjustments::TOPDOWN_CLOUD_FOOTPRINT_OFF);
        atmo.topdown_cloud_footprint = false;
    }
    if atmo.feather_exposed_domain_edges {
        changes.insert(GpuPreviewAdjustments::EXPOSED_EDGE_FEATHER_OFF);
        atmo.feather_exposed_domain_edges = false;
    }
    if atmo.cloud_multiscatter != CloudMultiscatterMode::LegacyOctaves {
        changes.insert(GpuPreviewAdjustments::LEGACY_CLOUD_TRANSPORT);
        atmo.cloud_multiscatter = CloudMultiscatterMode::LegacyOctaves;
    }
    if !gpu_cloud_tonemap_compatible(atmo.cloud_softclip, atmo.cloud_highlight_max) {
        changes.insert(GpuPreviewAdjustments::SHIPPED_HIGHLIGHTS);
        atmo.cloud_softclip = CLOUD_SOFTCLIP_KNEE;
        atmo.cloud_highlight_max = RHO_HIGHLIGHT_MAX;
    }
    if atmo.step_quality != StepQuality::Interactive {
        changes.insert(GpuPreviewAdjustments::INTERACTIVE_STEPS);
        atmo.step_quality = StepQuality::Interactive;
    }
    atmo.gpu_clouds = true;
    atmo.parity = false;
    atmo.one_click_gpu_render = true;
    atmo.gpu_preview_adjustments = changes;
    changes
}

enum JobKind {
    Wrfout {
        path: PathBuf,
        cache_dir: PathBuf,
        run_id: String,
        ts_index: usize,
    },
    Cached {
        brick_path: PathBuf,
        params: WrfProjectionParams,
        run_id: String,
        time_iso: Option<String>,
        hhmm: u16,
    },
}

// ── display-side pan+zoom math (pure; unit-tested) ────────────────────────────

/// The new pan offset after a cursor-centred zoom: keep the image point under the
/// cursor fixed while the zoom scales by `ratio = new_zoom / old_zoom`. `rel` is the
/// cursor position relative to the viewport centre. Derivation: the on-screen offset
/// of a fixed image point scales by `ratio`, so `pan' = rel*(1 - ratio) + pan*ratio`
/// keeps `(rel - pan)` scaling exactly by `ratio` (the image point stays under the
/// cursor). This is a DISPLAY transform of the already-rendered frame; no re-render.
fn pan_after_cursor_zoom(pan: egui::Vec2, rel: egui::Vec2, ratio: f32) -> egui::Vec2 {
    rel * (1.0 - ratio) + pan * ratio
}

/// Clamp a pan offset so an image of size `img` cannot be dragged past its own edges
/// within a `viewport` (centred when it fits — at zoom 1.0 the image <= viewport, so
/// the allowed pan is 0 and it stays centred).
fn clamp_pan(pan: egui::Vec2, img: egui::Vec2, viewport: egui::Vec2) -> egui::Vec2 {
    let mx = ((img.x - viewport.x) * 0.5).max(0.0);
    let my = ((img.y - viewport.y) * 0.5).max(0.0);
    egui::vec2(pan.x.clamp(-mx, mx), pan.y.clamp(-my, my))
}

/// Products whose render path contains the Band 13 instrument stage.
fn mode_supports_instrument_footprint(mode: RenderMode) -> bool {
    matches!(
        mode,
        RenderMode::Ir | RenderMode::GeoColor | RenderMode::Sandwich
    )
}

/// Clear an instrument footprint before an incompatible product can hide its
/// control. Returning the previous value lets the UI explain and persist the
/// automatic safety transition exactly once.
fn clear_incompatible_instrument_footprint(
    mode: RenderMode,
    footprint: &mut InstrumentFootprint,
) -> Option<InstrumentFootprint> {
    if *footprint == InstrumentFootprint::Off || mode_supports_instrument_footprint(mode) {
        return None;
    }
    let previous = *footprint;
    *footprint = InstrumentFootprint::Off;
    Some(previous)
}

/// Select the reviewed display default only when the user explicitly enters a
/// thermal product. Startup/settings loading does not call this helper, so a
/// persisted explicit palette survives when the app opens in that same product.
///
/// Band 13 and each WV product enter their reviewed defaults (currently CIMSS).
/// Startup still preserves a persisted explicit palette because it does not call
/// this product-transition helper.
fn apply_product_transition_enhancement_default(
    previous: RenderMode,
    entered: RenderMode,
    enhancement: &mut IrEnhancement,
) -> Option<IrEnhancement> {
    if previous == entered {
        return None;
    }
    let desired = match entered {
        RenderMode::Ir => IrEnhancement::default(),
        RenderMode::WaterVapor(band) => band.default_enhancement(),
        _ => return None,
    };
    if *enhancement == desired {
        return None;
    }
    *enhancement = desired;
    Some(desired)
}

/// Validate the Studio's instrument-stage contract before touching input I/O.
/// The GUI's enable action selects these settings, while this guard catches any
/// later manual edit and prevents a silently misattributed footprint.
fn validate_studio_instrument_footprint(
    atmo: &AtmoSettings,
    preset: SatellitePreset,
    resolution: ResolutionMode,
) -> Result<(), String> {
    if atmo.instrument_footprint == InstrumentFootprint::Off {
        return Ok(());
    }
    if !mode_supports_instrument_footprint(atmo.render_mode) {
        return Err(format!(
            "Instrument footprint {} supports Band 13, GeoColor, and Sandwich only.",
            atmo.instrument_footprint.slug()
        ));
    }
    if atmo.thermal_sensor != ThermalSensor::GoesRAbiBand13Fm4 {
        return Err(format!(
            "Instrument footprint {} requires the GOES-R ABI Band 13 FM4 response.",
            atmo.instrument_footprint.slug()
        ));
    }
    if atmo.view_mode != StudioView::Geostationary
        || atmo.geo_navigation != GeoNavigation::GoesRAbiFixedGrid
        || resolution != ResolutionMode::Abi2km
        || preset == SatellitePreset::Himawari
    {
        return Err(format!(
            "Instrument footprint {} requires a GOES-East/West Geostationary render with \
             GOES-R exact navigation and ABI 2 km resolution. Toggle the footprint off/on \
             in Thermal response to apply those settings.",
            atmo.instrument_footprint.slug()
        ));
    }
    Ok(())
}

fn read_brick_for_profile(
    path: &Path,
    expected: StorageProfile,
) -> Result<simsat::bricks::VolumeBrick, bricks::BrickError> {
    let profiled = bricks::read_ssb_profiled(path)?;
    if profiled.profile != expected {
        return Err(bricks::BrickError::CacheMismatch(format!(
            "brick uses storage_profile={}, but Studio requested {}; profiles are never substituted",
            profiled.profile.slug(),
            expected.slug()
        )));
    }
    Ok(profiled.brick)
}

/// Prepare all CPU inputs for one render (ingest if needed, brick decode, camera
/// raster, solar, Blue Marble crop, normals/landmask, LUTs). Sends `Status` messages
/// over `tx` and RETURNS the prepared frame (or an error string). Returning it (rather
/// than sending a `Prepared`) lets the single-frame path and the batch loop both call
/// it — the caller tags the result as a single `Prepared` or a `BatchFrame`.
///
/// `cache` holds the timestep-INDEPENDENT scene resources (raster/geo LUT, Blue
/// Marble crop, atmosphere LUTs, horizon map) shared across the frames of a
/// sequence render and across repeated single renders — the loop-throughput work
/// (WS4 item 1). Only per-timestep work (brick decode, light LUT, cloud/IR march,
/// sun-OD, froxel) runs per frame on a cache hit.
fn prepare_render(
    job: JobKind,
    preset: SatellitePreset,
    resolution: ResolutionMode,
    atmo: AtmoSettings,
    cache: &Mutex<SceneCache>,
    tx: &Sender<WorkerMsg>,
) -> Result<Box<PreparedRender>, String> {
    let status = |s: &str| {
        let _ = tx.send(WorkerMsg::Status(s.to_string()));
    };

    validate_studio_instrument_footprint(&atmo, preset, resolution)?;
    if atmo.instrument_footprint != InstrumentFootprint::Off {
        status(&format!(
            "Instrument footprint {}: complete FM4 radiance on exact global 56-urad ABI lattice \
             (GOES-16 MTF -> GOES-19 remains experimental).",
            atmo.instrument_footprint.slug()
        ));
    }

    // Resolve: brick path, georef, params, time, sector — and, for a cached run,
    // the brick itself (it is read ONCE here to anchor the georef and passed
    // through; the second `read_ssb` of the same file is gone).
    let (brick_path, georef, params, time_iso, hhmm, run_id, peeked_brick) = match job {
        JobKind::Wrfout {
            path,
            cache_dir,
            run_id,
            ts_index,
        } => {
            // A GRIB2 source shares this whole arm; only the geometry read and the
            // ingest call differ (a GRIB file carries a single valid time).
            let is_grib = ingest_grib::is_grib_input(&path);
            let geom = if is_grib {
                ingest_grib::read_grib_geometry(&path).map_err(|e| format!("read geometry: {e}"))?
            } else {
                ingest::read_grid_geometry(&path, ts_index)
                    .map_err(|e| format!("read geometry: {e}"))?
            };
            let georef = geom.georef().map_err(|e| format!("georef: {e}"))?;
            let cloud_optics = if is_grib {
                if atmo.cloud_optics == CloudOpticsMode::HrrrThompsonNative {
                    atmo.cloud_optics
                } else {
                    CloudOpticsMode::Fixed
                }
            } else if atmo.cloud_optics == CloudOpticsMode::NsslNative {
                atmo.cloud_optics
            } else {
                CloudOpticsMode::Fixed
            };
            let brick_cache =
                ingest::brick_cache_dir(&cache_dir, cloud_optics, atmo.storage_profile);
            let brick_path = bricks::run_dir(&brick_cache, &run_id).join(
                bricks::brick_file_name_for(geom.time_iso.as_deref(), geom.hhmm),
            );
            // A cached brick counts only if it actually DECODES: an old-format or
            // corrupt .ssb (e.g. a v2 brick after the v3 snow-optics bump — the
            // owner-reported "Render failed: unsupported .ssb version: 2") is a
            // cache MISS with the wrfout right here as the source of truth, so
            // re-ingest instead of surfacing the decode error.
            let peeked = if brick_path.is_file() {
                match read_brick_for_profile(&brick_path, atmo.storage_profile) {
                    Ok(b) => Some(b),
                    Err(e) => {
                        status(&format!("Cached brick unusable ({e}); re-ingesting..."));
                        None
                    }
                }
            } else {
                None
            };
            if peeked.is_none() {
                status(&format!("Ingesting timestep {ts_index}..."));
                let mut config = IngestConfig::new(cache_dir.clone());
                config.run_id = Some(run_id.clone());
                config.timestep = ts_index;
                config.cloud_optics = cloud_optics;
                config.storage_profile = atmo.storage_profile;
                if is_grib {
                    ingest_grib::ingest_grib_timestep(&path, &config)
                        .map_err(|e| format!("grib ingest: {e}"))?;
                } else {
                    ingest::ingest_timestep(&path, &config).map_err(|e| format!("ingest: {e}"))?;
                }
                status("Ingest complete.");
            }
            (
                brick_path,
                georef,
                geom.params,
                geom.time_iso,
                geom.hhmm,
                run_id,
                peeked,
            )
        }
        JobKind::Cached {
            brick_path,
            params,
            run_id,
            time_iso,
            hhmm,
        } => {
            // nx/ny come from the brick; anchor the georef at the domain center.
            if !brick_path.is_file() {
                return Err(format!("brick not found: {}", brick_path.display()));
            }
            status("Decoding brick...");
            // A cached-run open has no source wrfout to re-ingest from, so an
            // old-format brick keeps the hard refusal — but with the remedy spelled
            // out instead of the raw decode error.
            let brick =
                read_brick_for_profile(&brick_path, atmo.storage_profile).map_err(|e| match e {
                    bricks::BrickError::UnsupportedVersion(v) => format!(
                        "this cached run's bricks are an older format (.ssb v{v}; this \
                     build reads v{}). The cache is regenerable: open the ORIGINAL \
                     wrfout (Open menu) to re-ingest, or delete the run's cache \
                     directory.",
                        atmo.storage_profile.format_version()
                    ),
                    e => format!("read brick: {e}"),
                })?;
            let georef = GridGeoref::from_params_center(&params, brick.nx, brick.ny)
                .map_err(|e| format!("georef: {e}"))?;
            (
                brick_path,
                georef,
                params,
                time_iso,
                hhmm,
                run_id,
                Some(brick),
            )
        }
    };

    let brick = match peeked_brick {
        Some(b) => b,
        None => {
            status("Decoding brick...");
            read_brick_for_profile(&brick_path, atmo.storage_profile)
                .map_err(|e| format!("read brick: {e}"))?
        }
    };
    let (nx, ny) = (brick.nx, brick.ny);

    // Output raster over the domain. Geostationary: the from-space scan raster (Native
    // sizes it to the WRF grid — one pixel per cell; the ABI modes use the fixed GOES
    // pitch). Top-down map: the north-up map raster over the domain's own Lambert extent
    // (Native is one-pixel-per-cell; ABI modes use physical 1 km / 2 km spacing), adapted
    // to the shared `SurfaceRaster` so the LUT +
    // Blue Marble + assemble machinery below is IDENTICAL — only the per-pixel ray at
    // render time diverges (nadir vs scan). `build_surface_raster_mode` logs to stderr if
    // a huge domain forces the MAX_AXIS clamp.
    let is_topdown = atmo.view_mode == StudioView::TopDownMap;
    let is_persp = atmo.view_mode == StudioView::Perspective;
    let camera = GeoCamera::for_navigation(preset, atmo.geo_navigation).map_err(str::to_string)?;
    let (map_dx_m, map_dy_m) = if params.map_proj == 6 {
        (params.dx_m * 111_195.0, params.dy_m * 111_195.0)
    } else {
        (params.dx_m, params.dy_m)
    };
    // Zoom-out / domain-margin: 0.0 = the domain edge-to-edge; > 0 grows the extent by that
    // fraction of the domain span on each side (real Blue Marble ground + clear sky around
    // the domain — no WRF weather outside it). Ignored in Perspective view (the camera
    // frames the scene; no margin extent and no edge feather — the api pattern).
    let margin = atmo.margin_frac;

    // PERSPECTIVE (3-D) view: map the orbit (azimuth/tilt/range/fov around the domain
    // centre) to the engine's free PerspectiveCamera via the pure, node-tested
    // `pipeline::orbit_to_camera` (range clamped to 0.3x-5x the domain diagonal).
    // Visible-mode only in v1 (UI-gated; guarded here as defense in depth).
    if is_persp && atmo.render_mode != RenderMode::Visible {
        return Err("Perspective (3-D) view renders the Visible product only in v1.".to_string());
    }
    let persp: Option<(PerspectiveCamera, PerspectiveBasis)> = if is_persp {
        // Domain diagonal in metres (MAP_PROJ 6 stores dx/dy in degrees).
        let (ddx, ddy) = if params.map_proj == 6 {
            (params.dx_m * 111_195.0, params.dy_m * 111_195.0)
        } else {
            (params.dx_m, params.dy_m)
        };
        let diag_m = (((nx.max(2) - 1) as f64 * ddx).powi(2)
            + ((ny.max(2) - 1) as f64 * ddy).powi(2))
        .sqrt();
        let (lo_m, hi_m) = pipeline::orbit_range_bounds_m(diag_m);
        let req_m = atmo.orbit.range_km * 1000.0;
        if !(lo_m..=hi_m).contains(&req_m) {
            status(&format!(
                "Orbit range clamped to {:.0}-{:.0} km for this domain.",
                lo_m / 1000.0,
                hi_m / 1000.0
            ));
        }
        let cam =
            pipeline::orbit_to_camera(&atmo.orbit, params.cen_lat_deg, params.cen_lon_deg, diag_m);
        let basis = cam
            .basis()
            .map_err(|e| format!("perspective camera: {e}"))?;
        // Provenance discipline: the camera pose goes into the render log (the api's
        // PERSPECTIVE log-line pattern).
        status(&format!("PERSPECTIVE camera {}", cam.label()));
        Some((cam, basis))
    } else {
        None
    };
    // The scene cache (WS4 item 1): single-slot, exact-key-equality caches for the
    // timestep-independent resources. The one in-flight worker is the only lock
    // holder (renders are serialized by the busy flag); a poisoned lock from an
    // earlier worker panic is recovered — every slot write is an atomic replace, so
    // the cache is always internally consistent.
    let mut scache = cache.lock().unwrap_or_else(|p| p.into_inner());
    let mut hits = CacheHits::default();
    let raster_key = pipeline::RasterCacheKey {
        georef,
        nx,
        ny,
        resolution: resolution_ordinal(resolution),
        margin_bits: margin.to_bits(),
        view: view_ordinal(atmo.view_mode),
        sat: sat_ordinal(preset),
        navigation: geo_navigation_ordinal(atmo.geo_navigation),
    };
    let (raster_arc, raster_hit) = if let Some((cam, basis)) = &persp {
        // Perspective rasters BYPASS the scene cache: the cache key does not carry
        // the orbit, and the per-pixel ray/sphere raster is cheap to rebuild (the
        // owner is expected to drag the orbit between renders anyway).
        status(&format!(
            "Building perspective raster ({}x{})...",
            cam.width, cam.height
        ));
        (
            Arc::new(build_perspective_raster(basis, &georef, nx, ny)),
            false,
        )
    } else {
        scache.raster.get_or_try_insert_with(
            raster_key.clone(),
            || -> Result<SurfaceRaster, String> {
                if is_topdown {
                    status("Building top-down map raster...");
                    build_map_raster_mode(&georef, nx, ny, map_dx_m, map_dy_m, resolution, margin)
                        .map(|m| m.as_surface_raster())
                        .ok_or_else(|| {
                            "The domain is too small to build a top-down map.".to_string()
                        })
                } else {
                    status("Building geostationary raster...");
                    build_surface_raster_mode(
                        &camera, &georef, nx, ny, resolution, margin, MAX_AXIS,
                    )
                    .ok_or_else(|| {
                        format!(
                            "The domain is not fully visible from {}. Try a different satellite.",
                            preset.label()
                        )
                    })
                }
            },
        )?
    };
    hits.raster = Some(raster_hit);
    let raster: &SurfaceRaster = &raster_arc;
    // Native clamped against the per-axis cap (the margin-extended target exceeds MAX_AXIS)?
    // Then the raster is coarser than native — the honest exception, surfaced in the UI.
    // (Top-down is capped separately in build_map_raster_mode; perspective dims are
    // explicit, never clamped here.)
    let (target_nx, target_ny) = extended_native_counts(nx, ny, margin);
    let res_clamped = !is_topdown
        && !is_persp
        && resolution == ResolutionMode::Native
        && (raster.nx < target_nx || raster.ny < target_ny);

    // Solar geometry from the timestep UTC (fallback: local noon-ish if unknown).
    let (year, month, day, ut) = time_iso
        .as_deref()
        .and_then(solar::parse_iso_utc)
        .unwrap_or((2004, 6, 21, 12.0));
    let solar = SolarFrame::new(year, month, day, ut);

    // Seasonal Blue Marble ground (M7): the day-of-year blend of the two bracketing
    // monthly composites (or a forced month for what-if), lazily fetched + SHA-256-gated
    // with the vendored 8 km fallback. IR mode is thermal — it needs no ground albedo at
    // all, so skip the (possibly large) decode entirely.
    let ir_mode = atmo.render_mode.is_thermal();
    let derived_mode = atmo.render_mode.derived_field();
    let is_sandwich = atmo.render_mode.is_sandwich();
    let is_visible_ir_composite = atmo.render_mode.is_visible_ir_composite();
    let ir_band = atmo.render_mode.ir_band();
    // EXPERIMENTAL GPU cloud path: Geostationary or Top-down Visible clouds-on. A parity
    // render takes it too (both paths run) even when the live toggle is off. A
    // projection the WGSL forward does not implement (rotated lat-lon = GRIB RRFS)
    // falls back to the CPU composite with a log line.
    let gpu_projection_ok =
        gpu::projection_supported(&georef) && atmo.geo_navigation == GeoNavigation::ModelSphere;
    if atmo.one_click_gpu_render && !gpu_projection_ok {
        return Err(
            "GPU Render unavailable for this source: rotated lat-lon (RRFS) has no WGSL \
             projection forward. No CPU fallback was substituted; use regular Render for the \
             exact CPU result."
                .to_string(),
        );
    }
    // The GPU shader honors the true-color-correction flag, but it has no per-pixel
    // terrain elevation. Never silently ignore that physical control.
    let gpu_atmosphere_ok = !atmo.terrain_atmosphere;
    // Fractional cloud/subcolumn closure is CPU-only until the GPU path gains the
    // cloud-fraction volume. Never silently render legacy full cells while the
    // physical control is active.
    let gpu_fractional_clouds_ok = gpu_fractional_preview_compatible(atmo.fractional_clouds);
    // Granulation changes both view extinction and the sunlight OD field. The WGSL
    // preview has neither, so never silently drop this physical control.
    let gpu_granulation_ok = gpu_granulation_preview_compatible(atmo.granulation);
    // The current GPU upload consumes the raw quantized brick. Until it accepts the
    // reconstructed decoded volume, a top-down request for that field must stay on CPU.
    // Keep this predicate explicit even while this branch's GPU path is geo-only so a
    // future top-down GPU activation cannot silently bypass the requested operator.
    let gpu_topdown_stratiform_ok = gpu_topdown_stratiform_preview_compatible(
        is_topdown,
        atmo.topdown_stratiform_regularization,
    );
    // The footprint is a CPU post-composite radiance operator. The current WGSL path
    // returns already-tonemapped pixels, so it cannot reproduce the requested seam.
    let gpu_topdown_cloud_footprint_ok =
        gpu_topdown_cloud_footprint_preview_compatible(is_topdown, atmo.topdown_cloud_footprint);
    // The exposed-domain edge control resolves against the camera raster. The WGSL
    // preview has only the legacy margin value, so keep the requested presentation exact
    // by routing it through CPU until the shader receives a reviewed twin.
    let gpu_exposed_edge_feather_ok =
        gpu_exposed_edge_feather_compatible(atmo.feather_exposed_domain_edges);
    // The cloud shader consumes exposure and ground lift through `CloudMarchParams`, but
    // its highlight knee/ceiling remain baked. Route custom values through CPU rather
    // than displaying a plausible-looking frame that silently ignored the controls.
    let gpu_cloud_tonemap_ok =
        gpu_cloud_tonemap_compatible(atmo.cloud_softclip, atmo.cloud_highlight_max);
    // The clear-sky surface shader has implicit exposure/ground calibration as well as
    // baked highlight constants. It is usable only at that exact neutral calibration.
    let gpu_surface_display_ok = gpu_surface_display_compatible(
        atmo.exposure,
        atmo.ground_gain,
        atmo.cloud_softclip,
        atmo.cloud_highlight_max,
    );
    // Both visible WGSL paths carry exact formula twins and sanitized uniforms for the
    // land corrections. Keep this predicate at the UI/worker seam so future land
    // operators cannot silently become GPU-eligible without an explicit review.
    let gpu_land_appearance_ok = gpu_land_appearance_compatible(atmo.land_appearance);
    // The current WGSL shader implements the established octave/single-scatter
    // transport only. Delta-flux is deliberately CPU-only until it has a reviewed
    // shader twin; never silently substitute a different cloud closure.
    let gpu_cloud_transport_ok = !matches!(
        atmo.cloud_multiscatter,
        CloudMultiscatterMode::DeltaFluxV1
            | CloudMultiscatterMode::DeltaFluxV2
            | CloudMultiscatterMode::DeltaFluxV3
    );
    let use_gpu_clouds = (atmo.gpu_clouds || atmo.parity)
        && atmo.clouds_enabled
        && !is_persp
        && matches!(atmo.render_mode, RenderMode::Visible)
        && gpu_projection_ok
        && gpu_atmosphere_ok
        && gpu_fractional_clouds_ok
        && gpu_granulation_ok
        && gpu_topdown_stratiform_ok
        && gpu_topdown_cloud_footprint_ok
        && gpu_exposed_edge_feather_ok
        && gpu_cloud_tonemap_ok
        && gpu_land_appearance_ok
        && gpu_cloud_transport_ok;
    if (atmo.gpu_clouds || atmo.parity) && !gpu_projection_ok {
        status("GPU clouds: rotated lat-lon (RRFS) is CPU-only; using the CPU composite.");
    }
    if (atmo.gpu_clouds || atmo.parity) && !gpu_atmosphere_ok {
        status("GPU clouds: terrain-height atmosphere is CPU-only; using the CPU composite.");
    }
    if (atmo.gpu_clouds || atmo.parity) && !gpu_fractional_clouds_ok {
        status(
            "GPU clouds: model cloud fraction/subcolumns are CPU-only; using the CPU \
             composite (turn off Use model cloud fraction for the legacy GPU preview).",
        );
    }
    if (atmo.gpu_clouds || atmo.parity) && !gpu_granulation_ok {
        status(
            "GPU clouds: granulation is CPU-only; using the CPU composite so the requested \
             sub-grid detail is not ignored.",
        );
    }
    if (atmo.gpu_clouds || atmo.parity) && !gpu_topdown_stratiform_ok {
        status(
            "GPU clouds: top-down stratiform reconstruction is CPU-only; using the CPU \
             composite so the reconstructed cloud field is not silently ignored.",
        );
    }
    if (atmo.gpu_clouds || atmo.parity) && !gpu_topdown_cloud_footprint_ok {
        status(
            "GPU clouds: top-down cloud footprint is CPU-only; using the CPU composite \
             so the requested pre-tonemap residual filter is not silently ignored.",
        );
    }
    if (atmo.gpu_clouds || atmo.parity) && !gpu_exposed_edge_feather_ok {
        status(
            "GPU clouds: exposed-domain edge feathering is CPU-only; using the CPU \
             composite so the requested boundary presentation is not ignored.",
        );
    }
    if (atmo.gpu_clouds || atmo.parity) && !gpu_cloud_tonemap_ok {
        status("GPU clouds: custom highlight knee/ceiling are CPU-only; using the CPU composite.");
    }
    if (atmo.gpu_clouds || atmo.parity) && !gpu_cloud_transport_ok {
        status("GPU clouds: delta-flux transport is CPU-only; using the CPU composite.");
    }
    if !atmo.clouds_enabled
        && matches!(atmo.render_mode, RenderMode::Visible)
        && (!gpu_surface_display_ok || !gpu_land_appearance_ok)
    {
        status(
            "Clear-sky display calibration is not representable by the GPU surface pass; \
             using the CPU surface path.",
        );
    }
    let bm_cache_dir = ingest::default_cache_dir();
    let (bluemarble, bm_status, season_line): (Option<BmGround>, BmStatus, String) = match raster
        .lat_lon_bbox()
    {
        // Thermal AND derived products are pure column computations — no ground albedo needed.
        Some(_) if ir_mode || derived_mode.is_some() => (None, BmStatus::Missing, String::new()),
        Some((la_min, la_max, lo_min, lo_max)) => {
            // The 21600x10800 global JPEG decode(s) — up to TWO per season blend —
            // are the single largest recoverable per-frame cost of a sequence
            // render, so the finished crop is cached under the full request key
            // (blend months + weight, override, download policy, bbox, max dim).
            let blend = match atmo.bm_month_override {
                Some(m) => bluemarble::MonthBlend::single(m),
                None => bluemarble::month_blend(month, day),
            };
            let bm_key = pipeline::BmCacheKey {
                month_a: blend.month_a,
                month_b: blend.month_b,
                weight_b_bits: blend.weight_b.to_bits(),
                month_override: atmo.bm_month_override,
                allow_download: atmo.bm_allow_download,
                bbox_bits: [
                    la_min.to_bits(),
                    la_max.to_bits(),
                    lo_min.to_bits(),
                    lo_max.to_bits(),
                ],
                max_dim: BM_MAX_DIM,
            };
            let manifest = asset_pack::embedded_manifest();
            let mut status_cb = |s: String| {
                let _ = tx.send(WorkerMsg::Status(s));
            };
            match scache.bluemarble.get_or_try_insert_with(bm_key, || {
                asset_pack::load_season_ground(
                    &bm_cache_dir,
                    &manifest,
                    month,
                    day,
                    atmo.bm_month_override,
                    atmo.bm_allow_download,
                    la_min,
                    la_max,
                    lo_min,
                    lo_max,
                    1.0,
                    BM_MAX_DIM,
                    &mut status_cb,
                )
                .map(|g| {
                    let line = g.status_line();
                    (g.crop, line)
                })
            }) {
                Ok((ground, hit)) => {
                    hits.bluemarble = Some(hit);
                    let line = ground.1.clone();
                    status(&line);
                    (Some(ground), BmStatus::Loaded, line)
                }
                Err(e) => {
                    // Even the vendored 8 km fallback could not be materialized (a hard
                    // disk error) — render flat albedo and surface the real reason.
                    status(&format!(
                        "Blue Marble unavailable ({e}); rendering flat albedo."
                    ));
                    (None, BmStatus::Failed(e), String::new())
                }
            }
        }
        None => {
            status("No on-earth pixels; rendering flat albedo.");
            (None, BmStatus::Missing, String::new())
        }
    };

    // Terrain normals + landmask from the brick.
    let normals = normals_from_hgt(&brick.hgt, nx, ny, params.dx_m, params.dy_m);
    let normals_rgba = gpu::normals_to_rgba8(&normals);
    let landmask_r8 = gpu::landmask_to_r8(&brick.landmask);
    // Horizontal cell size (m) for the cloud march step pitch (min of dx/dy). For a
    // lat/lon grid (MAP_PROJ 6) `dx_m` is in DEGREES, so convert to metres.
    let horiz_pitch_m = if params.map_proj == 6 {
        params.dx_m.min(params.dy_m) * 111_195.0
    } else {
        params.dx_m.min(params.dy_m)
    };
    // Per-axis cell size in METRES for the M3 horizon map (a lat/lon MAP_PROJ 6 grid
    // stores dx/dy in DEGREES). Captured here because `params` is shadowed by the
    // AtmosphereParams below, before the clouds-on branch that builds the horizon map.
    let (hgt_dx_m, hgt_dy_m) = if params.map_proj == 6 {
        (params.dx_m * 111_195.0, params.dy_m * 111_195.0)
    } else {
        (params.dx_m, params.dy_m)
    };

    // Per-pixel LUTs. `lut_light` is mutable so the fake-sun override can rewrite each
    // pixel's sun direction/elevation from the single overridden ECEF sun vector below.
    // The GEO half (BM UV + domain UV per pixel) is timestep-independent -> cached
    // under the raster key + the BM crop bounds; the LIGHT half (solar geometry)
    // changes every timestep -> rebuilt via `build_light_lut`, a unit-tested
    // bit-exact twin of the light half of `gpu::build_luts`.
    status("Building lookup textures...");
    let bm_crop = bluemarble.as_ref().map(|a| &a.0);
    let geo_key = pipeline::GeoLutKey {
        raster: raster_key.clone(),
        bm_bounds_bits: bm_crop.map(|bm| {
            [
                bm.lon_min.to_bits(),
                bm.lon_max.to_bits(),
                bm.lat_min.to_bits(),
                bm.lat_max.to_bits(),
            ]
        }),
    };
    let (lut_geo, mut lut_light) = if is_persp {
        // Perspective bypasses the geo-LUT cache too (its raster is uncached and the
        // key would not carry the orbit).
        hits.geo_lut = Some(false);
        let (g, l) = gpu::build_luts(raster, bm_crop, nx, ny, &solar);
        (Arc::new(g), l)
    } else if let Some(g) = scache.geo_lut.get(&geo_key) {
        hits.geo_lut = Some(true);
        let light = build_light_lut(raster, &solar);
        (g, light)
    } else {
        hits.geo_lut = Some(false);
        let (g, l) = gpu::build_luts(raster, bm_crop, nx, ny, &solar);
        (scache.geo_lut.put(geo_key, g), l)
    };

    // Stats for the status line. With the fake-sun override the centre elevation IS the
    // requested override elevation (the sun is placed there by construction).
    let on_earth = raster.lat.iter().filter(|v| v.is_finite()).count();
    let on_earth_frac = on_earth as f32 / (raster.nx * raster.ny).max(1) as f32;
    let center_sun_elev = match atmo.sun_override {
        Some((elev, _az)) => elev,
        None => raster
            .lat_lon_bbox()
            .map(|(la0, la1, lo0, lo1)| {
                solar
                    .at(((la0 + la1) * 0.5) as f64, ((lo0 + lo1) * 0.5) as f64)
                    .elevation_deg
            })
            .unwrap_or(0.0),
    };

    // ── M2 atmosphere frame (design section 3/6): optics-config LUTs + per-frame
    // sky-view ambient projection + the packed uniform. Built per render on the
    // worker (sub-ms class on a dGPU; the CPU reference is fast enough here). PW is
    // the domain-mean precipitable-water ratio from the brick qvapor (honest
    // approximation; documented in atmosphere.rs).
    status("Building atmosphere LUTs...");
    let pw_ratio = atmosphere::pw_ratio_from_brick(&brick);
    let params = AtmosphereParams {
        aod: atmo.aod,
        pw_ratio,
        aerosol_swelling: if atmo.rh_swelling { 1.5 } else { 1.0 },
        ground_albedo: atmosphere::GROUND_ALBEDO,
    };
    // Cached under EVERY AtmosphereParams field (raw f64 bits) — `pw_ratio` comes
    // from the brick, so a timestep whose domain-mean moisture moved rebuilds (a
    // stale LUT can never recolour a frame); the SH-2 sky ambient (M5) is built
    // from the same LUTs + params, so it lives in the same slot.
    let atmo_key = pipeline::AtmoLutKey {
        aod_bits: params.aod.to_bits(),
        pw_ratio_bits: params.pw_ratio.to_bits(),
        swelling_bits: params.aerosol_swelling.to_bits(),
        ground_albedo_bits: params.ground_albedo.to_bits(),
        sh_entries: SKY_SH_ENTRIES,
    };
    let (atmo_arc, atmo_hit) = scache.atmo.get_or_insert_with(atmo_key, || {
        let luts = AtmosphereLuts::build(&params);
        // SH-2 directional sky ambient (M5): projects the sky-view LUT into 9 RGB SH-2
        // coefficients per elevation entry (the "how much sky and what colour" ambient).
        // The clouds-OFF GPU surface pass still consumes the flat-hemisphere scalar
        // derived from it (`to_scalar_rgba_f32`); the clouds-ON CPU path evaluates the SH
        // directionally at the terrain normal + cloud up.
        let sky_sh = SkyShTable::build(&luts, &params, SKY_SH_ENTRIES);
        (luts, sky_sh)
    });
    hits.atmo = Some(atmo_hit);
    let luts = &atmo_arc.0;
    let sky_sh = &atmo_arc.1;

    let mut cam_geo = CameraGeometry::from_sub_lon(camera.model_sub_lon_deg);
    if let Some((_, basis)) = &persp {
        // ONE camera per frame — the perspective EYE (the engine's perspective
        // contract: FrameContext.cam.camera is the ray origin for the surface and
        // cloud marches; the api's render_perspective_scene does exactly this).
        cam_geo.camera = basis.eye;
    }
    // One ECEF sun vector for the frame (sun at infinity), from the domain centre. With
    // the fake-sun override, place it at the requested elevation/azimuth over the centre
    // (a uniform overridden sun direction, exactly like render_frame's sun-elev override);
    // otherwise use the file's real solar geometry.
    let sun_ecef = raster
        .lat_lon_bbox()
        .map(|(la0, la1, lo0, lo1)| {
            let clat = ((la0 + la1) * 0.5) as f64;
            let clon = ((lo0 + lo1) * 0.5) as f64;
            match atmo.sun_override {
                Some((elev, az)) => {
                    let e = elev.to_radians();
                    let a = az.to_radians();
                    let enu = [e.cos() * a.sin(), e.cos() * a.cos(), e.sin()];
                    atmosphere::sun_enu_to_ecef(enu, clat, clon)
                }
                None => {
                    atmosphere::sun_enu_to_ecef(solar.at(clat, clon).enu_direction(), clat, clon)
                }
            }
        })
        .unwrap_or([0.0, 0.0, 1.0]);
    // Fake-sun override: rewrite the per-pixel light LUT so every on-earth pixel's sun is
    // derived from this single overridden ECEF sun vector (the inverse of sun_enu_to_ecef),
    // matching how render_frame applies the override. NON-PHYSICAL what-if; labeled in the UI.
    if atmo.sun_override.is_some() {
        override_light_lut(&mut lut_light, raster, sun_ecef);
    }

    let scan = &raster.scan;
    let f3 = |v: [f64; 3]| [v[0] as f32, v[1] as f32, v[2] as f32];
    let uniforms = SurfaceUniforms {
        cam: f3(cam_geo.camera),
        r_ground: atmosphere::R_GROUND_M as f32,
        sun: f3(sun_ecef),
        r_top: atmosphere::R_TOP_M as f32,
        ex: f3(cam_geo.ex),
        x_min: scan.x_min as f32,
        ey: f3(cam_geo.ey),
        y_max: scan.y_max as f32,
        ez: f3(cam_geo.ez),
        pitch_x: scan.pitch_x as f32,
        solar: f3(SOLAR_IRRADIANCE_RGB),
        pitch_y: scan.pitch_y as f32,
        mie_sca: params.mie_scattering_ground() as f32,
        mie_ext: params.mie_extinction_ground() as f32,
        mie_g: atmosphere::MIE_ASYMMETRY_G as f32,
        pw_ratio: params.pw_ratio as f32,
        bm_present: if bluemarble.is_some() { 1.0 } else { 0.0 },
        water_scale: WATER_ALBEDO_SCALE,
        flat_albedo: FLAT_ALBEDO_SRGB,
        output_transform: atmo.output_transform.code(),
        ambient_elev_min: sky_sh.elev_min_deg as f32,
        ambient_elev_max: sky_sh.elev_max_deg as f32,
        ambient_n: sky_sh.entries.len() as f32,
        atmosphere_correction: if atmo.atmosphere_correction { 1.0 } else { 0.0 },
        land_appearance: atmo.land_appearance,
        surface_postlight_toe: atmo.surface_postlight_toe,
        twilight_surface_recovery: atmo.twilight_surface_recovery,
    };
    let transmittance_lut = luts.transmittance.data.clone();
    let multiscatter_lut = luts.multiscatter.data.clone();
    let ambient_n = sky_sh.entries.len() as u32;
    // The clouds-OFF GPU surface pass consumes the flat-hemisphere scalar of the SH
    // ambient (a documented CPU/GPU divergence; see SkyShTable::to_scalar_rgba_f32).
    let ambient_lut = sky_sh.to_scalar_rgba_f32();

    // ── M6 IR (band 13) OR the M4/M5 cloud raymarch. IR is a SEPARATE thermal pass
    // (a top-down slant-ray gray-body emission march -> true-Kelvin BT plane, coloured
    // through the enhancement); it shares only the raster + camera with the visible
    // path and ignores the sun/atmosphere/cloud state above (thermal — no sun input).
    // Both produce the one CPU-rendered RGBA frame `finish_prepared` displays/stores.
    let mut derived_out: Option<(DerivedField, Vec<f32>)> = None;
    let mut gpu_cloud_out: Option<Box<GpuCloudPrep>> = None;
    let (cloud_rgba, ir_bt) = if let Some(field) = derived_mode {
        // A DERIVED scalar-field map: a per-column brick integral (precipitable water /
        // cloud-top temperature / cloud optical depth), resampled onto the output raster and
        // coloured with the basic studio colormap. NOT a brightness-temperature product (no
        // live re-enhancement, no single-band Kelvin store). The RAW field is retained for the
        // status value range; the RAW array is the plotter's primary deliverable (the binding).
        status(&format!("Computing {} map...", field.label()));
        let native = derived::compute_field(&brick, field);
        let values = derived::resample_field(
            &native,
            nx,
            ny,
            &raster.grid_i,
            &raster.grid_j,
            raster.nx,
            raster.ny,
        );
        let rgba = derived_field_rgba(&values, field);
        derived_out = Some((field, values));
        status("Derived map complete.");
        (Some(rgba), None)
    } else if ir_mode {
        // Band 13 window OR a WV band (8/9/10) — the SAME thermal march, only the
        // `IrConfig` (wavelength + WV mass-absorption) and the enhancement band differ.
        let cfg = match atmo.render_mode {
            RenderMode::Ir => IrConfig::band13_with_sensor(atmo.thermal_sensor),
            _ => atmo
                .render_mode
                .ir_config()
                .expect("thermal mode has an IrConfig"),
        };
        status(&format!(
            "Marching {} ({})...",
            atmo.render_mode.thermal_label(),
            atmo.render_mode.label()
        ));
        let vol = IrVolume::from_brick(&brick, horiz_pitch_m);
        // The occupancy mip (from the same brick's extinction) drives coarse empty-
        // space skipping in the thermal march (conservative — no cloud is stepped over).
        let dv = clouds::DecodedVolume::from_brick_legacy(&brick, horiz_pitch_m);
        let mip = clouds::OccupancyMip::build(&dv, clouds::OCCUPANCY_MIP_FACTOR);
        let scene = IrScene {
            vol: &vol,
            mip: &mip,
            georef: &georef,
            cfg,
        };
        // Geostationary slant rays, or per-pixel nadir rays for the top-down map (same
        // thermal march — a simulated top-down brightness-temperature map).
        let bt = if is_topdown {
            topdown::render_topdown_ir_bt_frame(
                &scene,
                &raster.lat,
                &raster.lon,
                &raster.grid_i,
                raster.nx,
                raster.ny,
            )
        } else if atmo.instrument_footprint != InstrumentFootprint::Off {
            let (bt, footprint_status) =
                render_geo_band13_bt_with_footprint(&scene, &cam_geo, raster);
            status(&footprint_status);
            bt
        } else {
            ir::render_ir_bt_frame(&scene, &cam_geo, raster)
        };
        let rgba = render_ir_rgba(&bt, ir_band, atmo.ir_enhancement);
        status("Thermal march complete.");
        (Some(rgba), Some(bt))
    } else if use_gpu_clouds {
        // ── EXPERIMENTAL GPU cloud path (the M5-GPU activation). The worker packs
        // the volume + plans + upload payloads; the UI thread runs the sun-OD
        // compute and the clouds.wgsl march (gpu::CloudPassResources::render). The
        // CPU composite remains the shipping default and the ONLY stored path; a
        // parity render ALSO marches the CPU reference here for the diff.
        status("Preparing quantized GPU cloud volume...");
        // The live GPU shader consumes the brick's raw u8 codes. Build its binary
        // occupancy mip from those same codes too: decoding four full f32 volumes
        // solely for this conservative skip field used multiple GiB on large HRRR
        // bricks. The helper is byte-pinned against the decoded reference in gpu tests.
        let occupancy = gpu::quantized_occupancy_upload(&brick, clouds::OCCUPANCY_MIP_FACTOR);
        let (view_mode, ray_lut, scan_rect, froxel) = if is_topdown {
            // Top-down uses a reviewed per-pixel local-up LUT in the cloud shader.
            // Its CPU contract deliberately omits the extra geo camera->cloud front
            // froxel, so bind a neutral 1x1x1 value instead of building irrelevant
            // scan-space atmosphere data.
            (
                CloudViewMode::TopDownNadir,
                gpu::build_topdown_ray_lut(&raster.lat, &raster.lon),
                (0.0, 1.0, 0.0, 1.0),
                atmosphere::AerialFroxel {
                    dim: 1,
                    data: vec![0.0, 0.0, 0.0, 1.0],
                },
            )
        } else {
            let scan_rect = raster.model_scan_rect();
            (
                CloudViewMode::Geostationary,
                Vec::new(),
                scan_rect,
                atmosphere::build_aerial_froxel(
                    luts,
                    &params,
                    &cam_geo,
                    sun_ecef,
                    scan_rect,
                    atmosphere::AERIAL_FROXEL_DIM,
                ),
            )
        };
        // Exact twins of DecodedVolume::{voxel_pitch_m,r_bottom,r_top}; no decoded
        // volume is needed unless the explicit CPU parity render below is requested.
        let pitch = brick.dz_m.min(horiz_pitch_m).max(1.0);
        let r_bottom_m = atmosphere::R_GROUND_M + brick.z_min_m;
        let r_top_m = r_bottom_m + brick.nz as f64 * brick.dz_m;
        // The GPU pass always renders the INTERACTIVE schedule — the wgsl's
        // documented sun-march constants are the Interactive (6, 2.0) pair.
        // Eligibility above guarantees granulation is off; a requested granulated
        // render is routed to CPU rather than silently changing the cloud field.
        let icfg = MarchConfig {
            beer_powder: atmo.beer_powder,
            cloud_optical_depth_scale: atmo.cloud_optical_depth_scale,
            ground_day_lift: atmo.ground_gain,
            cloud_softclip_knee: atmo.cloud_softclip,
            cloud_highlight_max: atmo.cloud_highlight_max,
            topdown_shadow_antialias: is_topdown && atmo.topdown_shadow_antialias,
            octaves: match atmo.cloud_multiscatter {
                CloudMultiscatterMode::LegacyOctaves => clouds::DEFAULT_OCTAVES,
                CloudMultiscatterMode::SingleScatter
                | CloudMultiscatterMode::DeltaFluxV1
                | CloudMultiscatterMode::DeltaFluxV2
                | CloudMultiscatterMode::DeltaFluxV3 => 1,
            },
            edge_feather_cells: clouds::edge_feather_cells_for_margin(margin, nx, ny),
            ..MarchConfig::new(StepQuality::Interactive, pitch)
        };
        let march = CloudMarchParams {
            coarse_step_m: (icfg.coarse_mult * pitch) as f32,
            fine_step_m: (icfg.fine_mult * pitch) as f32,
            max_steps: icfg.max_steps as f32,
            exposure: atmo.exposure as f32,
            octaves: icfg.octaves as f32,
            beer_powder: icfg.beer_powder,
            ground_albedo: icfg.ground_albedo as f32,
            transmittance_floor: icfg.transmittance_floor as f32,
            edge_feather_cells: icfg.edge_feather_cells as f32,
            ground_day_lift: icfg.ground_day_lift as f32,
            cloud_optical_depth_scale: icfg.cloud_optical_depth_scale,
            topdown_shadow_antialias: icfg.topdown_shadow_antialias,
        };
        let sun_od_plan = gpu::plan_sun_od(
            &georef,
            nx,
            ny,
            brick.nz,
            brick.z_min_m,
            brick.dz_m,
            pitch,
            sun_ecef,
            SUN_OD_RESOLUTION,
        );
        let lq = brick.quant.get("ext_liquid");
        let iq = brick.quant.get("ext_ice");
        let pq = brick.quant.get("ext_precip");
        let tq = brick.quant.get("tau_up");
        // Parity: the CPU reference at GPU-COMPARABLE settings — Interactive
        // schedule, granulation OFF, FLAT/OPEN M3 surface fields, no snow blend (the
        // documented GPU surface model) — so the numbers isolate the march itself.
        // The LIVE CPU path additionally carries per-pixel terrain shadows /
        // aperture / wind / snow (an expected real difference, see the notes).
        let cpu_reference = if atmo.parity {
            status("Parity: marching the CPU reference (Interactive schedule)...");
            let vol = clouds::DecodedVolume::from_brick_legacy(&brick, horiz_pitch_m);
            let mip = clouds::OccupancyMip::build(&vol, clouds::OCCUPANCY_MIP_FACTOR);
            let sun_od = clouds::accumulate_sun_od_granulated(
                &vol,
                &georef,
                sun_ecef,
                SUN_OD_RESOLUTION,
                clouds::SUN_OD_EDGE_FEATHER_TEXELS,
                None,
            );
            let scene = clouds::CloudScene {
                vol: &vol,
                mip: &mip,
                sun_od: &sun_od,
                georef: &georef,
                luts,
                sky_sh,
                sun_ecef,
                cfg: icfg,
            };
            let surf = FrameContext {
                luts,
                params: &params,
                sky_sh,
                cam: cam_geo,
                sun_ecef,
                output_transform: atmo.output_transform,
                bm_present: bluemarble.is_some(),
                water_scale: WATER_ALBEDO_SCALE as f64,
                flat_albedo_srgb: FLAT_ALBEDO_SRGB as f64,
                raymarch_steps: 16,
                exposure: atmo.exposure,
                ground_day_lift: atmo.ground_gain,
                cloud_softclip_knee: atmo.cloud_softclip,
                cloud_highlight_max: atmo.cloud_highlight_max,
                synthetic_green: false,
                atmosphere_correction: atmo.atmosphere_correction,
                terrain_atmosphere: atmo.terrain_atmosphere,
                land_appearance: atmo.land_appearance,
                surface_postlight_toe: atmo.surface_postlight_toe,
                twilight_surface_recovery: atmo.twilight_surface_recovery,
            };
            let bm_ref = bluemarble.as_ref().map(|a| &a.0);
            let rnx = raster.nx;
            let assemble = |px: usize, py: usize| -> SurfacePixel {
                let idx = py * rnx + px;
                let g = &lut_geo[idx * 4..idx * 4 + 4];
                if g[0] < 0.0 {
                    return SurfacePixel {
                        on_earth: false,
                        ..Default::default()
                    };
                }
                let l = &lut_light[idx * 4..idx * 4 + 4];
                let base = match bm_ref {
                    Some(bm) if bm.width > 0 && bm.height > 0 => bm.sample_bilinear(g[0], g[1]),
                    _ => [FLAT_ALBEDO_SRGB, FLAT_ALBEDO_SRGB, FLAT_ALBEDO_SRGB],
                };
                let (normal_enu, is_water, surface_elevation_m) = if g[2] >= 0.0 {
                    let fi_f = (g[2] as f64 * nx as f64 - 0.5).clamp(0.0, (nx - 1) as f64);
                    let fj_f = (g[3] as f64 * ny as f64 - 0.5).clamp(0.0, (ny - 1) as f64);
                    let cell = fj_f.round() as usize * nx + fi_f.round() as usize;
                    (
                        normals[cell],
                        brick.landmask[cell] < 0.5,
                        brick.hgt.get(cell).copied().unwrap_or(0.0),
                    )
                } else {
                    ([0.0, 0.0, 1.0], false, 0.0)
                };
                SurfacePixel {
                    on_earth: true,
                    base_srgb: base,
                    normal_enu,
                    sun_enu: [l[0], l[1], l[2]],
                    sun_elev_deg: l[3],
                    is_water,
                    view_dir: [0.0, 0.0, 1.0],
                    surface_elevation_m,
                    // Flat/open M3 defaults (the GPU surface model).
                    ..Default::default()
                }
            };
            Some(if is_topdown {
                topdown::render_topdown_frame_rgba_with_cloud_footprint(
                    &surf,
                    Some(&scene),
                    &raster.lat,
                    &raster.lon,
                    raster.nx,
                    raster.ny,
                    assemble,
                    atmo.topdown_cloud_footprint,
                )
            } else {
                clouds::render_cloud_frame_rgba(&scene, &surf, &froxel, raster, assemble)
            })
        } else {
            None
        };
        status("Packing the GPU volume upload...");
        gpu_cloud_out = Some(Box::new(GpuCloudPrep {
            view_mode,
            ray_lut,
            texture_a: clouds::pack_texture_a(&brick),
            occupancy: occupancy.r8,
            vol_nx: brick.nx as u32,
            vol_ny: brick.ny as u32,
            vol_nz: brick.nz as u32,
            occ_dims: occupancy.dims,
            ql: [
                lq.vmin as f32,
                lq.vmax as f32,
                iq.vmin as f32,
                iq.vmax as f32,
            ],
            qp: [
                pq.vmin as f32,
                pq.vmax as f32,
                tq.vmin as f32,
                tq.vmax as f32,
            ],
            z_min_m: brick.z_min_m as f32,
            dz_m: brick.dz_m as f32,
            r_top_m: r_top_m as f32,
            r_bottom_m: r_bottom_m as f32,
            voxel_pitch_m: pitch as f32,
            geo: gpu::geo_quads(&georef),
            march,
            sun_od: sun_od_plan,
            froxel_dim: froxel.dim as u32,
            froxel_data: froxel.data.clone(),
            sh_rows: sky_sh.entries.len() as u32,
            sh_data: sky_sh.to_rgba_f32(),
            scan_rect: [
                scan_rect.0 as f32,
                scan_rect.1 as f32,
                scan_rect.2 as f32,
                scan_rect.3 as f32,
            ],
            cpu_reference,
        }));
        status("GPU cloud inputs ready.");
        (None, None)
    } else if atmo.clouds_enabled
        || is_topdown
        || is_persp
        || is_visible_ir_composite
        || !gpu_atmosphere_ok
        || !gpu_surface_display_ok
        || !gpu_land_appearance_ok
    {
        // ── CPU VISIBLE composite. Geostationary clouds-ON composites the M4/M5 cloud
        // march over the M2/M3 surface radiance (the tested CPU render path; the GPU
        // cloud pass is the deferred M5 activation). Regular CPU Top-down renders and
        // every PERSPECTIVE (3-D) view render here; the dedicated Top-down GPU preview
        // is handled by the branch above:
        // per-pixel nadir rays / pinhole eye rays into the SAME shading kernels
        // (`topdown::render_topdown_frame_rgba` / `render_perspective_frame_rgba`).
        // All run on the below-normal worker with rayon row-parallelism — the UI never
        // blocks. The scene resources below are built regardless (cheap) and the render
        // call selects the ray path + clouds on/off.
        status("Building horizon map...");
        // M3 horizon map (penumbral terrain shadows + the ambient aperture that
        // completes M5's SH-2 sky ambient; design section 6). Built here (clouds-on CPU
        // path only) from HGT, sun-independent. The clouds-OFF GPU surface pass does NOT
        // get the M3 per-texel terrain shadow/aperture/snow (deferred GPU activation,
        // per M4/M5/M6). `hgt_dx_m`/`hgt_dy_m` are the projection cell size in metres.
        // Cached per (run, dims, cell size): HGT is static across a run's timesteps,
        // so a sequence builds this expensive 16-azimuth scan once.
        let horizon_key = pipeline::HorizonCacheKey {
            run_id: run_id.clone(),
            nx,
            ny,
            dx_bits: hgt_dx_m.to_bits(),
            dy_bits: hgt_dy_m.to_bits(),
        };
        let (horizon_arc, horizon_hit) = scache.horizon.get_or_insert_with(horizon_key, || {
            HorizonMap::build(&brick.hgt, nx, ny, hgt_dx_m, hgt_dy_m)
        });
        hits.horizon = Some(horizon_hit);
        let horizon_map: &HorizonMap = &horizon_arc;
        // Sub-grid cloud GRANULATION (edge-erosion detail noise): ONE Option threaded
        // through the sun-OD accumulation AND MarchConfig (view + sun marches), so every
        // march of this composite samples the SAME eroded field. Off = byte-identical to
        // the pre-granulation render. dx-derived amplitude (near-neutral at 250 m).
        let granulation = if atmo.granulation {
            Some(clouds::Granulation::for_grid(horiz_pitch_m))
        } else {
            None
        };
        if atmo.clouds_enabled {
            match granulation {
                Some(g) => status(&format!(
                    "Marching clouds (granulation amp {:.2})...",
                    g.amplitude
                )),
                None => status("Marching clouds..."),
            }
        } else {
            status("Rendering clear-sky surface (clouds off)...");
        }
        let deterministic_fractional = atmo.clouds_enabled
            && atmo.fractional_clouds
            && atmo.fractional_cloud_mode.is_deterministic()
            && brick.has_cloud_fraction;
        if deterministic_fractional && is_persp {
            return Err(format!(
                "{} fractional clouds do not yet support Perspective; use Geostationary/Top-down or Effective OD",
                atmo.fractional_cloud_mode.slug()
            ));
        }
        if deterministic_fractional && is_topdown && atmo.topdown_stratiform_regularization {
            return Err(format!(
                "{} fractional clouds cannot be combined with Top-down stratiform reconstruction; disable one operator for an attributable QA render",
                atmo.fractional_cloud_mode.slug()
            ));
        }
        let fractional_clouds = atmo.clouds_enabled
            && atmo.fractional_clouds
            && atmo.fractional_cloud_mode == FractionalCloudMode::EffectiveOd
            && brick.has_cloud_fraction;
        let mut vol = if fractional_clouds {
            clouds::DecodedVolume::from_brick(&brick, horiz_pitch_m)
        } else {
            clouds::DecodedVolume::from_brick_legacy(&brick, horiz_pitch_m)
        };
        if fractional_clouds {
            let stats = vol.apply_fractional_clouds();
            let ratio = if stats.raw_fractional_tau > 0.0 {
                stats.effective_fractional_tau / stats.raw_fractional_tau
            } else {
                1.0
            };
            status(&format!(
                "Marching clouds (model fraction adjusted {}/{} columns, tau {:.2}x)...",
                stats.columns_modified, stats.columns_total, ratio
            ));
        } else if atmo.clouds_enabled && atmo.fractional_clouds && !deterministic_fractional {
            status("Marching clouds (model fraction unavailable; legacy coverage)...");
        }
        if is_topdown && atmo.clouds_enabled && atmo.topdown_stratiform_regularization {
            let stats = vol.regularize_topdown_stratiform_columns(atmo.cloud_optical_depth_scale);
            status(&format!(
                "Top-down stratiform reconstruction: {}/{} columns, OD {:.3}->{:.3}",
                stats.columns_changed, stats.columns_total, stats.tau_before, stats.tau_after,
            ));
        }
        let mip = clouds::OccupancyMip::build(&vol, clouds::OCCUPANCY_MIP_FACTOR);
        let sun_od = clouds::accumulate_sun_od_granulated(
            &vol,
            &georef,
            sun_ecef,
            SUN_OD_RESOLUTION,
            clouds::SUN_OD_EDGE_FEATHER_TEXELS,
            granulation,
        );
        let cfg = MarchConfig {
            beer_powder: atmo.beer_powder,
            cloud_optical_depth_scale: atmo.cloud_optical_depth_scale,
            ground_day_lift: atmo.ground_gain,
            cloud_softclip_knee: atmo.cloud_softclip,
            cloud_highlight_max: atmo.cloud_highlight_max,
            topdown_shadow_antialias: is_topdown && atmo.topdown_shadow_antialias,
            // Multi-scatter A/B (M5): DEFAULT_OCTAVES = the bright multiple-scatter look,
            // 1 = the fix2 single scatter.
            octaves: match atmo.cloud_multiscatter {
                CloudMultiscatterMode::LegacyOctaves => clouds::DEFAULT_OCTAVES,
                CloudMultiscatterMode::SingleScatter
                | CloudMultiscatterMode::DeltaFluxV1
                | CloudMultiscatterMode::DeltaFluxV2
                | CloudMultiscatterMode::DeltaFluxV3 => 1,
            },
            multiscatter_mode: atmo.cloud_multiscatter,
            // The legacy behavior described below remains exact when the new switch is
            // Off. When On, the shared resolver also activates the same band for an
            // actually exposed geo/perspective domain boundary at margin zero.
            // EDGE FEATHER (WS4 item 7): active only under a zoom-out margin (a
            // byte-identical no-op at margin 0) — the cloud contribution ramps to
            // zero over the outer band of the domain so clouds melt into the margin
            // instead of the hard glassy domain-edge cut seen in the QA frames.
            // Mirrors api.rs's wiring of the same engine function. Perspective ignores
            // the margin, but the opt-in still resolves from its actual camera raster.
            edge_feather_cells: studio_edge_feather_cells(
                is_persp,
                margin,
                nx,
                ny,
                atmo.feather_exposed_domain_edges,
                &raster.grid_i,
                &raster.grid_j,
            ),
            // Sub-grid granulation: the SAME value the sun-OD map above was
            // accumulated with (one eroded field per composite).
            granulation,
            ..MarchConfig::new(atmo.step_quality, vol.voxel_pitch_m())
        };
        let scene = clouds::CloudScene {
            vol: &vol,
            mip: &mip,
            sun_od: &sun_od,
            georef: &georef,
            luts,
            sky_sh,
            sun_ecef,
            cfg,
        };
        let surf = FrameContext {
            luts,
            params: &params,
            sky_sh,
            cam: cam_geo,
            sun_ecef,
            output_transform: atmo.output_transform,
            bm_present: bluemarble.is_some(),
            water_scale: WATER_ALBEDO_SCALE as f64,
            flat_albedo_srgb: FLAT_ALBEDO_SRGB as f64,
            raymarch_steps: 16,
            // One display exposure for the whole composited frame (surface + cloud).
            exposure: atmo.exposure,
            ground_day_lift: atmo.ground_gain,
            cloud_softclip_knee: atmo.cloud_softclip,
            cloud_highlight_max: atmo.cloud_highlight_max,
            synthetic_green: false,
            atmosphere_correction: atmo.atmosphere_correction,
            terrain_atmosphere: atmo.terrain_atmosphere,
            land_appearance: atmo.land_appearance,
            surface_postlight_toe: atmo.surface_postlight_toe,
            twilight_surface_recovery: atmo.twilight_surface_recovery,
        };
        let bm_ref = bluemarble.as_ref().map(|a| &a.0);
        let rnx = raster.nx;
        let assemble = |px: usize, py: usize| -> SurfacePixel {
            let idx = py * rnx + px;
            let g = &lut_geo[idx * 4..idx * 4 + 4];
            if g[0] < 0.0 {
                // Off-earth (space/limb): the surface pass handles it.
                return SurfacePixel {
                    on_earth: false,
                    ..Default::default()
                };
            }
            let l = &lut_light[idx * 4..idx * 4 + 4];
            let sun_enu = [l[0], l[1], l[2]];
            let mut base = match bm_ref {
                Some(bm) if bm.width > 0 && bm.height > 0 => bm.sample_bilinear(g[0], g[1]),
                _ => [FLAT_ALBEDO_SRGB, FLAT_ALBEDO_SRGB, FLAT_ALBEDO_SRGB],
            };
            // In-domain: terrain normal + LANDMASK water + the M3 horizon-map lookups
            // (penumbral cast shadow at the sun azimuth, ambient aperture), U10/V10
            // wind (Cox-Munk glint), and the SNOWH snow blend on land.
            let (
                normal_enu,
                is_water,
                terrain_horizon_rad,
                sky_openness,
                bent_normal_enu,
                wind_speed,
                surface_elevation_m,
            ) = if g[2] >= 0.0 {
                let fi_f = (g[2] as f64 * nx as f64 - 0.5).clamp(0.0, (nx - 1) as f64);
                let fj_f = (g[3] as f64 * ny as f64 - 0.5).clamp(0.0, (ny - 1) as f64);
                let cell = fj_f.round() as usize * nx + fi_f.round() as usize;
                let is_water = brick.landmask[cell] < 0.5;
                let sun_az = (sun_enu[1] as f64).atan2(sun_enu[0] as f64);
                let horizon = horizon_map.horizon_angle_at(fi_f, fj_f, sun_az) as f32;
                let (openness, bent) = horizon_map.aperture_at(fi_f, fj_f);
                let wind = (brick.u10[cell].powi(2) + brick.v10[cell].powi(2)).sqrt();
                if !is_water {
                    let snow = brick
                        .snowh
                        .as_ref()
                        .map_or(0.0, |s| snow_fraction(s[cell] as f64));
                    base = blend_snow(base, snow);
                }
                (
                    normals[cell],
                    is_water,
                    horizon,
                    openness as f32,
                    [bent[0] as f32, bent[1] as f32, bent[2] as f32],
                    wind,
                    brick.hgt.get(cell).copied().unwrap_or(0.0),
                )
            } else {
                ([0.0, 0.0, 1.0], false, 0.0, 1.0, [0.0, 0.0, 1.0], 0.0, 0.0)
            };
            SurfacePixel {
                on_earth: true,
                base_srgb: base,
                normal_enu,
                sun_enu,
                sun_elev_deg: l[3],
                is_water,
                view_dir: [0.0, 0.0, 1.0],
                terrain_horizon_rad,
                sky_openness,
                bent_normal_enu,
                wind_speed,
                surface_elevation_m,
            }
        };
        // Geostationary: scan-angle rays + the aerial-perspective froxel. Top-down map:
        // per-pixel nadir rays into the same shading (no froxel — the near-nadir front
        // airlight is negligible; see `topdown`). Perspective (3-D): the pinhole eye-ray
        // fan into the same shading (no froxel either — the topdown precedent the engine
        // documents; the surface keeps its full per-ray aerial perspective). Clouds
        // toggled off -> surface only in all three.
        let rgba = if deterministic_fractional {
            let froxel = if is_topdown {
                None
            } else {
                let scan_rect = raster.model_scan_rect();
                Some(atmosphere::build_aerial_froxel(
                    luts,
                    &params,
                    &cam_geo,
                    sun_ecef,
                    scan_rect,
                    atmosphere::AERIAL_FROXEL_DIM,
                ))
            };
            let members = atmo
                .fractional_cloud_mode
                .deterministic_subcolumn_count()
                .expect("deterministic worker branch requires member count");
            let mut radiance_sum = vec![0.0f64; raster.nx * raster.ny * 3];
            let mut alpha = vec![0u8; raster.nx * raster.ny];
            let mut grid_mean_tau = 0.0f64;
            let mut represented_tau_sum = 0.0f64;
            for member in 0..members {
                status(&format!(
                    "Deterministic fractional clouds: marching member {}/{}...",
                    member + 1,
                    members
                ));
                let mut member_vol = clouds::DecodedVolume::from_brick(&brick, horiz_pitch_m);
                let sub =
                    member_vol.apply_deterministic_fractional_subcolumn_count(member, members)?;
                if member == 0 {
                    grid_mean_tau = sub.grid_mean_cloud_tau;
                }
                represented_tau_sum += sub.subcolumn_cloud_tau;
                let member_mip =
                    clouds::OccupancyMip::build(&member_vol, clouds::OCCUPANCY_MIP_FACTOR);
                let member_sun_od = clouds::accumulate_sun_od_granulated(
                    &member_vol,
                    &georef,
                    sun_ecef,
                    SUN_OD_RESOLUTION,
                    clouds::SUN_OD_EDGE_FEATHER_TEXELS,
                    granulation,
                );
                let member_scene = clouds::CloudScene {
                    vol: &member_vol,
                    mip: &member_mip,
                    sun_od: &member_sun_od,
                    georef: &georef,
                    luts,
                    sky_sh,
                    sun_ecef,
                    cfg,
                };
                let (member_radiance, member_alpha) = if is_topdown {
                    topdown::render_topdown_frame_linear_radiance(
                        &surf,
                        Some(&member_scene),
                        &raster.lat,
                        &raster.lon,
                        raster.nx,
                        raster.ny,
                        assemble,
                        cfg.topdown_cloud_norm,
                    )
                } else {
                    clouds::render_cloud_frame_linear_radiance(
                        &member_scene,
                        &surf,
                        froxel.as_ref().expect("geo deterministic froxel"),
                        raster,
                        assemble,
                    )
                };
                for (sum, value) in radiance_sum.iter_mut().zip(member_radiance) {
                    *sum += value;
                }
                if member == 0 {
                    alpha = member_alpha;
                } else {
                    debug_assert_eq!(alpha, member_alpha);
                }
            }
            let members_f = members as f64;
            let represented_mean_tau = represented_tau_sum / members_f;
            status(&format!(
                "{} complete: ensemble cloud OD ratio {:.4}; averaging linear radiance then one tonemap...",
                atmo.fractional_cloud_mode.slug(),
                if grid_mean_tau > 0.0 {
                    represented_mean_tau / grid_mean_tau
                } else {
                    1.0
                }
            ));
            let mut rgba = vec![0u8; raster.nx * raster.ny * 4];
            for idx in 0..raster.nx * raster.ny {
                if alpha[idx] == 0 {
                    continue;
                }
                let mut l = [
                    radiance_sum[idx * 3] / members_f,
                    radiance_sum[idx * 3 + 1] / members_f,
                    radiance_sum[idx * 3 + 2] / members_f,
                ];
                let pixel = assemble(idx % raster.nx, idx / raster.nx);
                l = apply_low_sun_illuminant(l, pixel.on_earth, pixel.sun_elev_deg as f64, luts);
                let display = radiance_to_rgba_softclip(
                    l,
                    atmo.output_transform,
                    atmo.exposure,
                    cfg.cloud_softclip_knee,
                    cfg.cloud_highlight_max,
                );
                for c in 0..4 {
                    rgba[idx * 4 + c] = (display[c].clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
            rgba
        } else if let Some((_, basis)) = &persp {
            let scene_opt = if atmo.clouds_enabled {
                Some(&scene)
            } else {
                None
            };
            topdown::render_perspective_frame_rgba(&surf, scene_opt, basis, assemble)
        } else if is_topdown {
            let scene_opt = if atmo.clouds_enabled {
                Some(&scene)
            } else {
                None
            };
            topdown::render_topdown_frame_rgba_with_cloud_footprint(
                &surf,
                scene_opt,
                &raster.lat,
                &raster.lon,
                raster.nx,
                raster.ny,
                assemble,
                atmo.topdown_cloud_footprint,
            )
        } else {
            render_geo_visible_rgba(atmo.clouds_enabled, &surf, raster, &assemble, |assemble| {
                let scan_rect = raster.model_scan_rect();
                let froxel = atmosphere::build_aerial_froxel(
                    luts,
                    &params,
                    &cam_geo,
                    sun_ecef,
                    scan_rect,
                    atmosphere::AERIAL_FROXEL_DIM,
                );
                clouds::render_cloud_frame_rgba(&scene, &surf, &froxel, raster, assemble)
            })
        };
        status("Render complete.");
        if is_visible_ir_composite {
            // GeoColor / Sandwich: also march the band-13 IR and composite it into the visible
            // frame. GeoColor crossfades day-visible vs night-colored-IR by the PER-PIXEL solar
            // elevation (`lut_light` `l[3]`); Sandwich overlays the color-enhanced IR on the
            // COLD (high) cloud tops by the per-pixel BT (alpha ramps with coldness). Reuses the
            // cloud occupancy mip for the thermal march's empty-space skipping. Stored as the
            // visible rgb composite (ir_bt=None).
            status("Marching band-13 IR for the composite...");
            let ir_vol = IrVolume::from_brick(&brick, horiz_pitch_m);
            let ir_scene = IrScene {
                vol: &ir_vol,
                mip: &mip,
                georef: &georef,
                cfg: IrConfig::band13_with_sensor(atmo.thermal_sensor),
            };
            let bt = if is_topdown {
                topdown::render_topdown_ir_bt_frame(
                    &ir_scene,
                    &raster.lat,
                    &raster.lon,
                    &raster.grid_i,
                    raster.nx,
                    raster.ny,
                )
            } else if atmo.instrument_footprint != InstrumentFootprint::Off {
                let (bt, footprint_status) =
                    render_geo_band13_bt_with_footprint(&ir_scene, &cam_geo, raster);
                status(&footprint_status);
                bt
            } else {
                ir::render_ir_bt_frame(&ir_scene, &cam_geo, raster)
            };
            let n = raster.nx * raster.ny;
            let blended = if is_sandwich {
                // Sandwich: color the BT through the sandwich enhancement and overlay it on the
                // cold tops of the visible base by the per-pixel BT (see `sandwich`).
                let ir_rgba = render_ir_rgba(&bt, 13, sandwich::SANDWICH_ENHANCEMENT);
                let ir_rgb: Vec<u8> = ir_rgba
                    .chunks_exact(4)
                    .flat_map(|p| [p[0], p[1], p[2]])
                    .collect();
                let (_rgb, blended) = sandwich::blend_rgba(&rgba, &ir_rgb, &bt, n);
                status("Sandwich composite complete.");
                blended
            } else {
                // GeoColor: color the BT through the night enhancement and crossfade by the
                // per-pixel solar elevation (day -> true-color, night -> colored IR).
                let ir_rgba = render_ir_rgba(&bt, 13, geocolor::GEOCOLOR_NIGHT_ENHANCEMENT);
                let ir_rgb: Vec<u8> = ir_rgba
                    .chunks_exact(4)
                    .flat_map(|p| [p[0], p[1], p[2]])
                    .collect();
                let (_rgb, blended) =
                    geocolor::blend_rgba(&rgba, &ir_rgb, n, |i| lut_light[i * 4 + 3] as f64);
                status("SimSat Day/Night Color blend complete.");
                blended
            };
            (Some(blended), None)
        } else {
            (Some(rgba), None)
        }
    } else {
        (None, None)
    };

    // One compact per-frame cache report (visible in the Log view; the owner sees
    // the sequence speedup as the second frame onward turning to hits).
    status(&format!("Scene cache: {}.", hits.summary()));

    let prep = PreparedRender {
        width: raster.nx as u32,
        height: raster.ny as u32,
        nx: nx as u32,
        ny: ny as u32,
        lut_geo,
        lut_light,
        normals_rgba,
        landmask_r8,
        bluemarble,
        bm_status,
        season_line,
        lat: raster.lat.clone(),
        lon: raster.lon.clone(),
        sector: run_id,
        satellite: preset,
        view_mode: atmo.view_mode,
        year,
        month,
        day,
        hhmm,
        on_earth_frac,
        center_sun_elev,
        sun_override: atmo.sun_override.is_some(),
        resolution,
        res_clamped,
        transmittance_lut,
        multiscatter_lut,
        ambient_lut,
        ambient_n,
        uniforms,
        pw_ratio,
        cloud_rgba,
        ir_bt,
        ir_enhancement: atmo.ir_enhancement,
        ir_band,
        instrument_footprint: atmo.instrument_footprint,
        derived: derived_out,
        render_mode: atmo.render_mode,
        clouds_enabled: atmo.clouds_enabled,
        gpu_cloud: gpu_cloud_out,
        one_click_gpu_render: atmo.one_click_gpu_render,
        gpu_preview_adjustments: atmo.gpu_preview_adjustments,
    };
    Ok(Box::new(prep))
}

/// Studio-side twin of the LIGHT half of `gpu::build_luts`: per-pixel sun ENU
/// direction + elevation from the frame's solar geometry. Used on a geo-LUT cache
/// hit so only the per-timestep light is rebuilt. MUST stay bit-exact with the
/// engine (the unit test below compares against `gpu::build_luts` directly).
fn build_light_lut(raster: &SurfaceRaster, solar: &SolarFrame) -> Vec<f32> {
    let n = raster.nx * raster.ny;
    let mut light = vec![0.0f32; n * 4];
    for idx in 0..n {
        let lat = raster.lat[idx];
        let lon = raster.lon[idx];
        if !lat.is_finite() || !lon.is_finite() {
            // Space: zeros, exactly as the engine writes them.
            continue;
        }
        let pos = solar.at(lat as f64, lon as f64);
        let d = pos.enu_direction();
        let l = &mut light[idx * 4..idx * 4 + 4];
        l[0] = d[0] as f32;
        l[1] = d[1] as f32;
        l[2] = d[2] as f32;
        l[3] = pos.elevation_deg as f32;
    }
    light
}

/// Studio twin of the public API's instrument stage: march complete Band 13
/// radiance, apply the spatial response on the exact global ABI lattice, exclude
/// its crop/invalid-mask perimeter, then invert the FM4 response to BT.
fn render_geo_band13_bt_with_footprint(
    scene: &IrScene<'_>,
    cam_geo: &CameraGeometry,
    raster: &SurfaceRaster,
) -> (Vec<f32>, String) {
    debug_assert_eq!(scene.cfg.band, 13);
    debug_assert_eq!(scene.cfg.sensor, ThermalSensor::GoesRAbiBand13Fm4);
    let radiance = ir::render_ir_radiance_frame(scene, cam_geo, raster);
    let filtered = apply_band13_radiance_footprint_validated(&radiance, raster.nx, raster.ny);
    let valid = filtered.validation_mask.iter().filter(|&&v| v != 0).count();
    let (x0, x1, y0, y1) = raster
        .scan
        .abi_2km_global_indices()
        .expect("Studio footprint validation selects exact ABI 2-km navigation");
    let status = format!(
        "ABI footprint: global crop x={x0}..{x1}, y={y0}..{y1}, exact 56 urad; \
         validation support {valid}/{} px ({} finite crop/mask-perimeter px emitted no-data).",
        raster.nx * raster.ny,
        filtered.excluded_finite_samples
    );
    let bt = filtered
        .radiance
        .into_iter()
        .map(|value| {
            if value.is_finite() {
                scene.cfg.brightness_temperature(value) as f32
            } else {
                f32::NAN
            }
        })
        .collect();
    (bt, status)
}

/// Stable ordinals for the scene-cache keys (see `pipeline::RasterCacheKey`).
fn resolution_ordinal(r: ResolutionMode) -> u8 {
    match r {
        ResolutionMode::Native => 0,
        ResolutionMode::Abi1km => 1,
        ResolutionMode::Abi2km => 2,
    }
}

fn view_ordinal(v: StudioView) -> u8 {
    match v {
        StudioView::Geostationary => 0,
        StudioView::TopDownMap => 1,
        // Perspective rasters bypass the scene cache (keyed by the orbit, which is
        // not part of the key) — the ordinal exists only for key completeness.
        StudioView::Perspective => 2,
    }
}

fn sat_ordinal(p: SatellitePreset) -> u8 {
    match p {
        SatellitePreset::GoesEast => 0,
        SatellitePreset::GoesWest => 1,
        SatellitePreset::Himawari => 2,
    }
}

fn geo_navigation_ordinal(navigation: GeoNavigation) -> u8 {
    match navigation {
        GeoNavigation::ModelSphere => 0,
        GeoNavigation::GoesRAbiFixedGrid => 1,
    }
}

/// A short product token for the PNG-export file name.
fn product_token(mode: RenderMode) -> &'static str {
    match mode {
        RenderMode::Visible => "visible",
        RenderMode::GeoColor => "geocolor",
        RenderMode::Sandwich => "sandwich",
        RenderMode::Ir => "ir-band13",
        RenderMode::WaterVapor(WvBand::Upper) => "wv62",
        RenderMode::WaterVapor(WvBand::Mid) => "wv69",
        RenderMode::WaterVapor(WvBand::Low) => "wv73",
        RenderMode::Derived(DerivedField::PrecipitableWater) => "pw",
        RenderMode::Derived(DerivedField::CloudTopTemp) => "ctt",
        RenderMode::Derived(DerivedField::CloudOpticalDepth) => "cod",
    }
}

/// Colour a resampled derived scalar field (`nx*ny` f32; `NaN` = no-data / off-domain / a
/// clear cloud-top column) to row-major `Rgba8` (`nx*ny*4`): the basic studio colormap with
/// no-data pixels transparent (they display as black, like space). The RAW `f32` field is the
/// primary deliverable; this is only the in-app + store colour map.
fn derived_field_rgba(values: &[f32], field: DerivedField) -> Vec<u8> {
    let mut out = vec![0u8; values.len() * 4];
    for (i, &v) in values.iter().enumerate() {
        if v.is_finite() {
            let c = derived::value_color(v, field);
            out[i * 4] = c[0];
            out[i * 4 + 1] = c[1];
            out[i * 4 + 2] = c[2];
            out[i * 4 + 3] = 255;
        }
    }
    out
}

/// The sun-OD map resolution the studio accumulates per render (design section 6
/// target is 1024^2; a domain crop is well-shadowed at 512 and it keeps the
/// worker-side accumulation fast — the map is a coarse ground/long-range shadow).
const SUN_OD_RESOLUTION: usize = 512;

/// Max Blue Marble crop dimension the studio requests (also part of the crop's
/// scene-cache key).
const BM_MAX_DIM: u32 = 4096;

/// SH-2 sky-ambient table entry count (the M5 value; part of the atmosphere
/// LUT cache key).
const SKY_SH_ENTRIES: usize = 48;

// ── fake-sun override helpers (mirror render_frame) ──────────────────────────

/// Rewrite the per-pixel light LUT so every on-earth pixel's sun is derived from the
/// single global `sun_ecef` (the fake-sun what-if override). This is the inverse of
/// `atmosphere::sun_enu_to_ecef`: project the ECEF sun into each pixel's local ENU basis
/// and take the elevation from its up component. Space pixels keep their zeroed entry.
/// Identical to render_frame's override so the studio and the CLI agree. NON-PHYSICAL.
fn override_light_lut(light: &mut [f32], raster: &SurfaceRaster, sun_ecef: [f64; 3]) {
    let n = raster.nx * raster.ny;
    for idx in 0..n {
        let lat = raster.lat[idx];
        let lon = raster.lon[idx];
        if !lat.is_finite() || !lon.is_finite() {
            continue;
        }
        let (enu, elev) = sun_enu_and_elev(sun_ecef, lat as f64, lon as f64);
        let l = &mut light[idx * 4..idx * 4 + 4];
        l[0] = enu[0] as f32;
        l[1] = enu[1] as f32;
        l[2] = enu[2] as f32;
        l[3] = elev as f32;
    }
}

/// Project a global ECEF sun direction into the local ENU basis at `(lat, lon)`,
/// returning `(sun_enu, elevation_deg)`. Inverse of `atmosphere::sun_enu_to_ecef`.
fn sun_enu_and_elev(sun_ecef: [f64; 3], lat_deg: f64, lon_deg: f64) -> ([f64; 3], f64) {
    let (la, lo) = (lat_deg.to_radians(), lon_deg.to_radians());
    let (sla, cla) = la.sin_cos();
    let (slo, clo) = lo.sin_cos();
    let east = [-slo, clo, 0.0];
    let north = [-sla * clo, -sla * slo, cla];
    let up = [cla * clo, cla * slo, sla];
    let dot = |a: [f64; 3]| a[0] * sun_ecef[0] + a[1] * sun_ecef[1] + a[2] * sun_ecef[2];
    let elev = dot(up).clamp(-1.0, 1.0).asin().to_degrees();
    ([dot(east), dot(north), dot(up)], elev)
}

// ── small helpers ──────────────────────────────────────────────────────────

const LARGE_WRF_WARN_CELLS_3D: usize = 10_000_000;
const LARGE_WRF_WARN_BYTES: u64 = 1 << 30;

fn manifest_params(manifest: &RunManifest) -> WrfProjectionParams {
    let p = &manifest.projection;
    WrfProjectionParams {
        map_proj: p.map_proj,
        truelat1_deg: p.truelat1_deg,
        truelat2_deg: p.truelat2_deg,
        stand_lon_deg: p.stand_lon_deg,
        cen_lat_deg: p.cen_lat_deg,
        cen_lon_deg: p.cen_lon_deg,
        dx_m: p.dx_m,
        dy_m: p.dy_m,
    }
}

fn parse_time(t: &str) -> (u16, String) {
    match solar::parse_iso_utc(t) {
        Some((y, mo, d, ut)) => {
            let hh = ut as u16;
            let mm = ((ut - hh as f64) * 60.0).round() as u16;
            (
                hh * 100 + mm,
                format!("{y:04}-{mo:02}-{d:02}T{hh:02}:{mm:02}:00Z"),
            )
        }
        None => (0, t.to_string()),
    }
}

/// A short "IR band 13" / "WV 6.2 um (band 8)" description for a thermal band number,
/// for status lines + the frame header. WV bands are 8/9/10; anything else is the IR
/// window band (13).
fn band_display(band: u8) -> String {
    match band {
        8 => "WV 6.2 um (band 8)".to_string(),
        9 => "WV 6.9 um (band 9)".to_string(),
        10 => "WV 7.3 um (band 10)".to_string(),
        b => format!("IR band {b}"),
    }
}

/// Whether the visible-frame status line should report volumetric clouds as on.
/// CPU top-down/perspective surface-only renders still return an RGBA buffer, so
/// this must use the captured UI toggle rather than infer state from buffer presence.
fn rendered_clouds_on(clouds_enabled: bool, is_ir: bool, is_derived: bool) -> bool {
    clouds_enabled && !is_ir && !is_derived
}

/// Render geostationary visible pixels with the cloud toggle as a real renderer branch.
/// The disabled branch is the same surface-only row-parallel path used by the public API;
/// the cloud callback is lazy so a disabled render cannot accidentally enter the volume march.
fn render_geo_visible_rgba<F, C>(
    clouds_enabled: bool,
    surf: &FrameContext,
    raster: &SurfaceRaster,
    assemble: &F,
    render_clouds: C,
) -> Vec<u8>
where
    F: Fn(usize, usize) -> SurfacePixel + Sync,
    C: FnOnce(&F) -> Vec<u8>,
{
    if clouds_enabled {
        render_clouds(assemble)
    } else {
        render_geo_surface_rgba(surf, raster, assemble)
    }
}

/// Geostationary M2/M3 surface-only render (clouds off). Each assembled pixel gets
/// the actual scan-ray view direction before going through [`shade_surface`].
fn render_geo_surface_rgba<F>(surf: &FrameContext, raster: &SurfaceRaster, assemble: &F) -> Vec<u8>
where
    F: Fn(usize, usize) -> SurfacePixel + Sync,
{
    use rayon::prelude::*;

    let scan = &raster.scan;
    let rows: Vec<Vec<u8>> = (0..scan.ny)
        .into_par_iter()
        .map(|py| {
            let mut row = Vec::with_capacity(scan.nx * 4);
            for px in 0..scan.nx {
                let (sx, sy) = scan.scan_angle(px, py);
                let mut pixel = assemble(px, py);
                pixel.view_dir = surf.cam.view_dir(sx, sy);
                let rgba = shade_surface(surf, &pixel);
                for &value in &rgba {
                    row.push((value.clamp(0.0, 1.0) * 255.0).round() as u8);
                }
            }
            row
        })
        .collect();
    rows.into_iter().flatten().collect()
}

/// The exact engine-owned defaults restored by the Studio display-calibration button.
/// Keeping the values in one pure helper prevents the UI's dirty predicate and reset
/// action from drifting apart as controls are added.
fn shipped_display_calibration() -> (f32, f32, f32, f32) {
    (
        simsat::render::DEFAULT_EXPOSURE as f32,
        GROUND_DAY_LIFT as f32,
        CLOUD_SOFTCLIP_KNEE as f32,
        RHO_HIGHLIGHT_MAX as f32,
    )
}

fn display_calibration_is_dirty(
    exposure: f32,
    ground_gain: f32,
    cloud_softclip: f32,
    cloud_highlight_max: f32,
) -> bool {
    let shipped = shipped_display_calibration();
    (exposure - shipped.0).abs() > f32::EPSILON
        || (ground_gain - shipped.1).abs() > f32::EPSILON
        || (cloud_softclip - shipped.2).abs() > f32::EPSILON
        || (cloud_highlight_max - shipped.3).abs() > f32::EPSILON
}

/// The experimental WGSL preview does not yet carry the cloud-fraction volume or
/// subcolumn closure. Keep UI enablement and worker-side fallback on one predicate.
fn gpu_fractional_preview_compatible(fractional_clouds: bool) -> bool {
    !fractional_clouds
}

/// Resolve the Studio's backwards-compatible persisted booleans to the same explicit
/// engine transport enum used by the Rust API, CLI, and Python binding. Delta-flux wins
/// when selected; otherwise the historical multi-scatter checkbox retains its exact
/// octaves/single-scatter meaning.
fn studio_cloud_multiscatter_mode(
    multiscatter: bool,
    delta_flux_clouds: bool,
    delta_flux_v2_clouds: bool,
    delta_flux_v3_clouds: bool,
) -> CloudMultiscatterMode {
    if delta_flux_v3_clouds {
        CloudMultiscatterMode::DeltaFluxV3
    } else if delta_flux_v2_clouds {
        CloudMultiscatterMode::DeltaFluxV2
    } else if delta_flux_clouds {
        CloudMultiscatterMode::DeltaFluxV1
    } else if multiscatter {
        CloudMultiscatterMode::LegacyOctaves
    } else {
        CloudMultiscatterMode::SingleScatter
    }
}

/// The experimental WGSL preview does not sample the granulated cloud field for
/// either the camera or sunlight march. A requested granulated render must use CPU.
fn gpu_granulation_preview_compatible(granulation: bool) -> bool {
    !granulation
}

/// The GPU cloud upload currently consumes the raw quantized brick, not a decoded
/// volume after the top-down stratiform observation operator. Geo ignores that
/// top-down-only switch, but a top-down render with it enabled must remain on CPU.
fn gpu_topdown_stratiform_preview_compatible(
    is_topdown: bool,
    topdown_stratiform_regularization: bool,
) -> bool {
    !is_topdown || !topdown_stratiform_regularization
}

/// The top-down cloud footprint filters the linear-radiance cloud residual before the
/// display transform. The current GPU preview returns a finished tonemapped frame, so a
/// top-down request with this control enabled must remain on CPU.
fn gpu_topdown_cloud_footprint_preview_compatible(
    is_topdown: bool,
    topdown_cloud_footprint: bool,
) -> bool {
    !is_topdown || !topdown_cloud_footprint
}

/// The current WGSL preview accepts only the legacy margin-derived feather width;
/// it cannot inspect camera-raster coverage to activate the exposed-domain experiment.
fn gpu_exposed_edge_feather_compatible(feather_exposed_domain_edges: bool) -> bool {
    !feather_exposed_domain_edges
}

/// Studio-side call seam for the engine-owned camera-coverage resolver. Perspective
/// ignores the margin slider, while geo/top-down retain the current margin behavior.
fn studio_edge_feather_cells(
    is_perspective: bool,
    margin: f64,
    nx: usize,
    ny: usize,
    feather_exposed_domain_edges: bool,
    grid_i: &[f32],
    grid_j: &[f32],
) -> f64 {
    clouds::edge_feather_cells_for_raster(
        if is_perspective { 0.0 } else { margin },
        nx,
        ny,
        feather_exposed_domain_edges,
        grid_i,
        grid_j,
    )
}

/// The cloud WGSL path has baked highlight shoulder constants. Exposure and ground
/// lift are uniforms, so only the two baked values constrain GPU preview eligibility.
fn gpu_cloud_tonemap_compatible(cloud_softclip: f64, cloud_highlight_max: f64) -> bool {
    const EPS: f64 = 1.0e-6; // controls round-trip through persisted/UI f32 values
    (cloud_softclip - CLOUD_SOFTCLIP_KNEE).abs() <= EPS
        && (cloud_highlight_max - RHO_HIGHLIGHT_MAX).abs() <= EPS
}

/// The clear-sky surface WGSL path additionally has implicit neutral exposure and
/// ground lift. Use it only when it can reproduce every requested display control.
fn gpu_surface_display_compatible(
    exposure: f64,
    ground_gain: f64,
    cloud_softclip: f64,
    cloud_highlight_max: f64,
) -> bool {
    const EPS: f64 = 1.0e-6;
    (exposure - 1.0).abs() <= EPS
        && (ground_gain - 1.0).abs() <= EPS
        && gpu_cloud_tonemap_compatible(cloud_softclip, cloud_highlight_max)
}

/// Both visible WGSL paths consume the complete, sanitized land-appearance uniform.
/// All parameter states are representable: the packer duplicates the CPU bounds and
/// non-finite fallbacks before f32 upload, and both switches remain independent.
fn gpu_land_appearance_compatible(_config: LandAppearanceConfig) -> bool {
    true
}

/// Default sat-store root under the SimSat Studio data dir (sibling of the brick
/// cache). Shown in the UI and changeable; the owner points BowEcho here.
fn default_store_root() -> PathBuf {
    let base = ingest::default_cache_dir();
    // default_cache_dir ends in ".../cache"; put the store beside it.
    base.parent()
        .map(|p| p.join("sat-store"))
        .unwrap_or_else(|| base.join("sat-store"))
}

/// Write one rendered frame into the sat store: the IR true-Kelvin BT plane as a
/// single-band band-13 run, or the visible three baked rgb planes. Shared by the
/// single-frame Write-to-store and the batch loop (which calls it per timestep so all
/// frames land in one bit-identical-grid multi-frame run).
#[allow(clippy::too_many_arguments)]
fn store_write_frame(
    store_root: &Path,
    rendered: &RenderedFrame,
    ir_bt: Option<&Vec<f32>>,
    ir_band: u8,
    lat: &[f32],
    lon: &[f32],
    sector: &str,
    satellite: SatellitePreset,
    year: i32,
    month: u32,
    day: u32,
    hhmm: u16,
) -> Result<WrittenVisibleFrame, String> {
    if let Some(bt) = ir_bt {
        // IR window (band 13) OR a WV band (8/9/10) — the same single-band Kelvin frame,
        // keyed by `ir_band` (run `_c{band:02}_`, variable `ahi_bt_c{band:02}`).
        let frame = IrFrame::new_band(
            rendered.width as usize,
            rendered.height as usize,
            bt.clone(),
            lat.to_vec(),
            lon.to_vec(),
            sector.to_string(),
            satellite,
            ir_band,
            year,
            month,
            day,
            hhmm,
        );
        store_out::write_ir_frame(store_root, &frame)
    } else {
        let frame = VisibleFrame::from_rendered(
            rendered,
            lat.to_vec(),
            lon.to_vec(),
            sector.to_string(),
            satellite,
            year,
            month,
            day,
            hhmm,
        );
        store_out::write_visible_frame(store_root, &frame)
    }
}

/// A valid-time display label for a rendered frame (`2020-01-05 01:30 UTC`).
fn frame_time_label(year: i32, month: u32, day: u32, hhmm: u16) -> String {
    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02} UTC",
        hhmm / 100,
        hhmm % 100
    )
}

/// Native model-input picker with a useful Windows default filter.
///
/// WRF's conventional `wrfout_dNN_...` files have no extension. `rfd` models filters
/// as extensions and its Windows backend prefixes each entry with `*.`; the second
/// semicolon-delimited entry therefore deliberately expands to the native COM filter
/// `*.grib2;*.grb2;*.grib;*.grb;wrfout*`. This keeps unrelated files out of the
/// default view while retaining extensionless WRF output. "All files" remains an
/// explicit fallback for nonstandard names. Other platforms retain the extensionless-
/// safe all-files default because their backends do not share this Windows expansion.
fn model_input_dialog(title: &str) -> rfd::FileDialog {
    let dialog = rfd::FileDialog::new().set_title(title);
    #[cfg(target_os = "windows")]
    {
        dialog
            .add_filter(
                "WRF / GRIB2 model files",
                &["grib2", "grb2;*.grib;*.grb;wrfout*"],
            )
            .add_filter("GRIB2", &["grib2", "grb2", "grib", "grb"])
            .add_filter("All files", &["*"])
    }
    #[cfg(not(target_os = "windows"))]
    {
        dialog
            .add_filter("All files (wrfout / GRIB2)", &["*"])
            .add_filter("GRIB2", &["grib2", "grb2", "grib", "grb"])
    }
}

/// List candidate wrfout files in a directory: regular files whose name looks like a
/// wrfout (starts with `wrfout` or embeds a parseable valid time) and is not one of our
/// own cache/store artifacts. Sorted by name; the sequence is re-sorted by valid time
/// after probing.
fn list_wrfout_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for entry in rd.flatten() {
            let p = entry.path();
            if !p.is_file() {
                continue;
            }
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            let looks_wrf =
                name.starts_with("wrfout") || pipeline::parse_valid_time(name).is_some();
            let is_aux = name.ends_with(".json")
                || name.ends_with(".ssb")
                || name.ends_with(".rws")
                || name.ends_with(".rwg")
                || name.ends_with(".png");
            if looks_wrf && !is_aux {
                files.push(p);
            }
        }
    }
    files.sort();
    files
}

/// One run identifier for a whole sequence (the store sector + the brick cache run dir)
/// so every rendered frame lands in ONE multi-frame store run. Prefers the containing
/// directory's name (e.g. the Enderlin folder); falls back to the sanitized common
/// prefix of the file names, else `"sequence"`.
fn sequence_run_id(files: &[PathBuf]) -> String {
    if let Some(dir_name) = files
        .first()
        .and_then(|p| p.parent())
        .and_then(|d| d.file_name())
        .and_then(|s| s.to_str())
    {
        let token = store_out::sanitize_store_token(dir_name);
        if token != "unknown" {
            return token;
        }
    }
    let names: Vec<&str> = files
        .iter()
        .filter_map(|p| p.file_name().and_then(|s| s.to_str()))
        .collect();
    if let Some(first) = names.first() {
        let mut prefix_len = first.len();
        for n in &names[1..] {
            prefix_len = prefix_len.min(common_prefix_len(first, n));
        }
        let token = store_out::sanitize_store_token(&first[..prefix_len]);
        if token != "unknown" {
            return token;
        }
    }
    "sequence".to_string()
}

/// Byte length of the shared leading prefix of two strings, at a UTF-8 char boundary.
fn common_prefix_len(a: &str, b: &str) -> usize {
    let mut len = 0;
    for (ca, cb) in a.chars().zip(b.chars()) {
        if ca != cb {
            break;
        }
        len += ca.len_utf8();
    }
    len
}

#[cfg(test)]
mod tests {
    use super::*;

    fn studio_test_surface_context() -> FrameContext<'static> {
        static OPTICS: std::sync::OnceLock<(AtmosphereParams, AtmosphereLuts, SkyShTable)> =
            std::sync::OnceLock::new();
        let (params, luts, sky_sh) = OPTICS.get_or_init(|| {
            let params = AtmosphereParams::default();
            let luts = AtmosphereLuts::build(&params);
            let sky_sh = SkyShTable::build(&luts, &params, 16);
            (params, luts, sky_sh)
        });
        FrameContext {
            luts,
            params,
            sky_sh,
            cam: CameraGeometry::from_sub_lon(-75.2),
            sun_ecef: atmosphere::sun_enu_to_ecef([0.0, 0.0, 1.0], 0.0, -75.2),
            output_transform: OutputTransform::AbiReflectance,
            bm_present: true,
            water_scale: WATER_ALBEDO_SCALE as f64,
            flat_albedo_srgb: FLAT_ALBEDO_SRGB as f64,
            raymarch_steps: 8,
            exposure: simsat::render::DEFAULT_EXPOSURE,
            ground_day_lift: GROUND_DAY_LIFT,
            cloud_softclip_knee: CLOUD_SOFTCLIP_KNEE,
            cloud_highlight_max: RHO_HIGHLIGHT_MAX,
            synthetic_green: false,
            atmosphere_correction: true,
            terrain_atmosphere: false,
            land_appearance: LandAppearanceConfig::identity(),
            surface_postlight_toe: SurfacePostlightToeConfig::off(),
            twilight_surface_recovery: TwilightSurfaceRecoveryConfig::off(),
        }
    }

    fn incompatible_gpu_preview_atmo() -> AtmoSettings {
        AtmoSettings {
            render_intent: RenderIntent::Display,
            geo_navigation: GeoNavigation::ModelSphere,
            intent_adjustments: Vec::new(),
            view_mode: StudioView::TopDownMap,
            orbit: pipeline::OrbitParams {
                az_deg: 180.0,
                tilt_deg: 45.0,
                range_km: 1_000.0,
                fov_deg: 60.0,
                width: 1280,
                height: 720,
            },
            margin_frac: 0.25,
            render_mode: RenderMode::Ir,
            ir_enhancement: IrEnhancement::default(),
            thermal_sensor: ThermalSensor::GoesRAbiBand13Fm4,
            instrument_footprint: InstrumentFootprint::Off,
            aod: 0.05,
            rh_swelling: true,
            atmosphere_correction: true,
            terrain_atmosphere: true,
            land_appearance: LandAppearanceConfig::default(),
            surface_postlight_toe: SurfacePostlightToeConfig {
                enabled: true,
                ..SurfacePostlightToeConfig::default()
            },
            twilight_surface_recovery: TwilightSurfaceRecoveryConfig {
                enabled: true,
                ..TwilightSurfaceRecoveryConfig::default()
            },
            output_transform: OutputTransform::AbiReflectance,
            clouds_enabled: false,
            fractional_clouds: true,
            fractional_cloud_mode: FractionalCloudMode::Deterministic4,
            cloud_multiscatter: CloudMultiscatterMode::DeltaFluxV1,
            cloud_optical_depth_scale: 0.15,
            storage_profile: StorageProfile::CompactU8,
            cloud_optics: CloudOpticsMode::Fixed,
            feather_exposed_domain_edges: true,
            beer_powder: false,
            granulation: true,
            topdown_stratiform_regularization: true,
            topdown_cloud_footprint: true,
            topdown_shadow_antialias: true,
            step_quality: StepQuality::Offline,
            gpu_clouds: false,
            parity: true,
            one_click_gpu_render: false,
            gpu_preview_adjustments: GpuPreviewAdjustments::default(),
            exposure: 1.5,
            ground_gain: 1.0,
            cloud_softclip: 0.75,
            cloud_highlight_max: 1.05,
            sun_override: None,
            bm_month_override: None,
            bm_allow_download: false,
        }
    }

    #[test]
    fn studio_sensor_fast_gray_uses_temporary_strict_controls_and_keeps_fractional_clouds() {
        let mut atmo = incompatible_gpu_preview_atmo();
        atmo.render_intent = RenderIntent::SensorFastGray;
        let original_fractional = atmo.fractional_clouds;
        configure_render_intent(&mut atmo);

        assert_eq!(atmo.cloud_optical_depth_scale, 1.0);
        assert_eq!(atmo.exposure, 1.0);
        assert_eq!(atmo.ground_gain, 1.0);
        assert_eq!(atmo.land_appearance, LandAppearanceConfig::identity());
        assert!(atmo.surface_postlight_toe.is_identity());
        assert!(atmo.twilight_surface_recovery.is_identity());
        assert!(!atmo.feather_exposed_domain_edges);
        assert!(!atmo.granulation);
        assert!(!atmo.topdown_stratiform_regularization);
        assert!(!atmo.topdown_cloud_footprint);
        assert!(!atmo.topdown_shadow_antialias);
        assert!(!atmo.atmosphere_correction);
        assert_eq!(atmo.cloud_softclip, 1.0);
        assert_eq!(atmo.cloud_highlight_max, 1.0);
        assert_eq!(atmo.fractional_clouds, original_fractional);
        assert!(
            atmo.intent_adjustments
                .contains(&RenderIntentAdjustment::CloudOpticalDepthUnscaled)
        );
        assert!(
            atmo.intent_adjustments
                .contains(&RenderIntentAdjustment::HighlightShoulderIdentity)
        );
        assert!(
            atmo.intent_adjustments
                .contains(&RenderIntentAdjustment::SurfacePostlightToeOff)
        );
        assert!(
            atmo.intent_adjustments
                .contains(&RenderIntentAdjustment::TwilightSurfaceRecoveryOff)
        );
        assert!(
            atmo.intent_adjustments
                .contains(&RenderIntentAdjustment::TopdownShadowAntialiasOff)
        );
        assert_eq!(
            atmo.render_intent.observation_operator(),
            "simsat-fast-gray-v1"
        );
    }

    #[test]
    fn studio_abi_footprint_guard_requires_the_exact_sensor_geometry() {
        let mut atmo = incompatible_gpu_preview_atmo();
        atmo.instrument_footprint = InstrumentFootprint::GoesRAbiBand13Mtf;
        let err = validate_studio_instrument_footprint(
            &atmo,
            SatellitePreset::GoesEast,
            ResolutionMode::Abi2km,
        )
        .unwrap_err();
        assert!(err.contains("GOES-R exact navigation"), "{err}");

        atmo.view_mode = StudioView::Geostationary;
        atmo.geo_navigation = GeoNavigation::GoesRAbiFixedGrid;
        validate_studio_instrument_footprint(
            &atmo,
            SatellitePreset::GoesEast,
            ResolutionMode::Abi2km,
        )
        .unwrap();
        assert!(
            validate_studio_instrument_footprint(
                &atmo,
                SatellitePreset::Himawari,
                ResolutionMode::Abi2km,
            )
            .is_err()
        );

        atmo.instrument_footprint = InstrumentFootprint::Off;
        atmo.render_mode = RenderMode::Visible;
        validate_studio_instrument_footprint(
            &atmo,
            SatellitePreset::Himawari,
            ResolutionMode::Native,
        )
        .unwrap();
    }

    #[test]
    fn incompatible_product_switch_clears_hidden_abi_footprint() {
        for mode in [
            RenderMode::Visible,
            RenderMode::WaterVapor(WvBand::Upper),
            RenderMode::Derived(DerivedField::CloudOpticalDepth),
        ] {
            let mut footprint = InstrumentFootprint::GoesRAbiBand13Mtf;
            assert_eq!(
                clear_incompatible_instrument_footprint(mode, &mut footprint),
                Some(InstrumentFootprint::GoesRAbiBand13Mtf),
                "{mode:?}"
            );
            assert_eq!(footprint, InstrumentFootprint::Off, "{mode:?}");
        }

        for mode in [RenderMode::Ir, RenderMode::GeoColor, RenderMode::Sandwich] {
            let mut footprint = InstrumentFootprint::GoesRAbiBand13Mtf;
            assert_eq!(
                clear_incompatible_instrument_footprint(mode, &mut footprint),
                None,
                "{mode:?}"
            );
            assert_eq!(
                footprint,
                InstrumentFootprint::GoesRAbiBand13Mtf,
                "{mode:?}"
            );
        }
    }

    #[test]
    fn thermal_product_transition_selects_product_default_but_same_product_preserves_palette() {
        let mut enhancement = IrEnhancement::Natural;
        assert_eq!(
            apply_product_transition_enhancement_default(
                RenderMode::Visible,
                RenderMode::WaterVapor(WvBand::Upper),
                &mut enhancement,
            ),
            Some(IrEnhancement::Cimss)
        );
        assert_eq!(enhancement, WvBand::Upper.default_enhancement());

        enhancement = IrEnhancement::Rainbow;
        assert_eq!(
            apply_product_transition_enhancement_default(
                RenderMode::WaterVapor(WvBand::Upper),
                RenderMode::WaterVapor(WvBand::Mid),
                &mut enhancement,
            ),
            Some(IrEnhancement::Cimss)
        );
        assert_eq!(enhancement, WvBand::Mid.default_enhancement());

        enhancement = IrEnhancement::Natural;
        assert_eq!(
            apply_product_transition_enhancement_default(
                RenderMode::Visible,
                RenderMode::Ir,
                &mut enhancement,
            ),
            Some(IrEnhancement::Cimss)
        );
        assert_eq!(enhancement, IrEnhancement::Cimss);

        // Loading the app in the same persisted product does not constitute a
        // transition and therefore keeps the user's explicit palette selection.
        enhancement = IrEnhancement::Rainbow;
        assert_eq!(
            apply_product_transition_enhancement_default(
                RenderMode::WaterVapor(WvBand::Upper),
                RenderMode::WaterVapor(WvBand::Upper),
                &mut enhancement,
            ),
            None
        );
        assert_eq!(enhancement, IrEnhancement::Rainbow);

        enhancement = IrEnhancement::Cimss;
        assert_eq!(
            apply_product_transition_enhancement_default(
                RenderMode::Ir,
                RenderMode::Ir,
                &mut enhancement,
            ),
            None
        );
        assert_eq!(enhancement, IrEnhancement::Cimss);
    }

    #[test]
    fn one_click_gpu_render_selects_compatible_preview_without_touching_calibration() {
        let mut atmo = incompatible_gpu_preview_atmo();
        let original_exposure = atmo.exposure;
        let original_cloud_scale = atmo.cloud_optical_depth_scale;
        let original_margin = atmo.margin_frac;
        let original_postlight_toe = atmo.surface_postlight_toe;
        let original_twilight_recovery = atmo.twilight_surface_recovery;
        let changes = configure_one_click_gpu_preview(&mut atmo);

        assert_eq!(atmo.render_mode, RenderMode::Visible);
        assert_eq!(atmo.view_mode, StudioView::TopDownMap);
        assert!(atmo.clouds_enabled);
        assert!(!atmo.terrain_atmosphere);
        assert!(!atmo.fractional_clouds);
        assert_eq!(atmo.fractional_cloud_mode, FractionalCloudMode::Off);
        assert!(!atmo.granulation);
        assert!(!atmo.topdown_stratiform_regularization);
        assert!(!atmo.topdown_cloud_footprint);
        assert_eq!(atmo.surface_postlight_toe, original_postlight_toe);
        assert_eq!(atmo.twilight_surface_recovery, original_twilight_recovery);
        assert!(!atmo.feather_exposed_domain_edges);
        assert_eq!(
            atmo.cloud_multiscatter,
            CloudMultiscatterMode::LegacyOctaves
        );
        assert_eq!(atmo.cloud_softclip, CLOUD_SOFTCLIP_KNEE);
        assert_eq!(atmo.cloud_highlight_max, RHO_HIGHLIGHT_MAX);
        assert_eq!(atmo.step_quality, StepQuality::Interactive);
        assert!(atmo.gpu_clouds);
        assert!(!atmo.parity);
        assert!(atmo.one_click_gpu_render);
        assert_eq!(atmo.exposure, original_exposure);
        assert_eq!(atmo.cloud_optical_depth_scale, original_cloud_scale);
        assert_eq!(atmo.margin_frac, original_margin);
        assert_eq!(changes.0 & GpuPreviewAdjustments::VIEW_GEOSTATIONARY, 0);
        assert!(changes.0 & GpuPreviewAdjustments::MODE_VISIBLE != 0);
        assert!(changes.summary().contains("legacy full-cell coverage"));
        assert!(changes.summary().contains("Legacy octaves"));
        assert!(changes.summary().contains("stratiform reconstruction"));
        assert!(changes.summary().contains("cloud footprint"));
        assert!(!changes.summary().contains("Post-lighting surface toe"));
        assert!(!changes.summary().contains("Twilight surface recovery"));
    }

    #[test]
    fn one_click_gpu_render_temporarily_replaces_delta_flux_v2b_transport() {
        let mut atmo = incompatible_gpu_preview_atmo();
        atmo.render_mode = RenderMode::Visible;
        atmo.cloud_multiscatter = CloudMultiscatterMode::DeltaFluxV2;

        let changes = configure_one_click_gpu_preview(&mut atmo);

        assert_eq!(atmo.view_mode, StudioView::TopDownMap);
        assert_eq!(
            atmo.cloud_multiscatter,
            CloudMultiscatterMode::LegacyOctaves
        );
        assert_ne!(changes.0 & GpuPreviewAdjustments::LEGACY_CLOUD_TRANSPORT, 0);
        assert!(changes.summary().contains("Legacy octaves"));
    }

    #[test]
    fn one_click_gpu_render_temporarily_disables_abi_footprint_and_exact_navigation() {
        let mut persistent = incompatible_gpu_preview_atmo();
        persistent.render_mode = RenderMode::Ir;
        persistent.view_mode = StudioView::Geostationary;
        persistent.geo_navigation = GeoNavigation::GoesRAbiFixedGrid;
        persistent.instrument_footprint = InstrumentFootprint::GoesRAbiBand13Mtf;
        let mut preview = persistent.clone();

        let changes = configure_one_click_gpu_preview(&mut preview);

        assert_eq!(persistent.geo_navigation, GeoNavigation::GoesRAbiFixedGrid);
        assert_eq!(
            persistent.instrument_footprint,
            InstrumentFootprint::GoesRAbiBand13Mtf
        );
        assert_eq!(preview.render_mode, RenderMode::Visible);
        assert_eq!(preview.geo_navigation, GeoNavigation::ModelSphere);
        assert_eq!(preview.instrument_footprint, InstrumentFootprint::Off);
        assert_ne!(
            changes.0 & GpuPreviewAdjustments::INSTRUMENT_FOOTPRINT_OFF,
            0
        );
        assert_ne!(
            changes.0 & GpuPreviewAdjustments::MODEL_SPHERE_NAVIGATION,
            0
        );
        assert!(changes.summary().contains("instrument footprint -> Off"));
        assert!(changes.summary().contains("model sphere"));
    }

    #[test]
    fn one_click_gpu_render_plan_is_idempotent() {
        let mut atmo = incompatible_gpu_preview_atmo();
        configure_one_click_gpu_preview(&mut atmo);
        let second = configure_one_click_gpu_preview(&mut atmo);
        assert!(second.is_empty());
        assert!(atmo.gpu_clouds);
        assert!(atmo.one_click_gpu_render);
    }

    #[test]
    fn one_click_gpu_render_only_changes_unsupported_perspective_camera() {
        let mut atmo = incompatible_gpu_preview_atmo();
        atmo.view_mode = StudioView::Perspective;
        let changes = configure_one_click_gpu_preview(&mut atmo);
        assert_eq!(atmo.view_mode, StudioView::Geostationary);
        assert_ne!(changes.0 & GpuPreviewAdjustments::VIEW_GEOSTATIONARY, 0);
    }

    #[test]
    fn geo_clouds_off_renders_real_surface_pixels_without_entering_cloud_callback() {
        let surf = studio_test_surface_context();
        let camera = GeoCamera::new(SatellitePreset::GoesEast);
        let (sx, sy) = camera
            .forward(0.0, -75.2)
            .expect("sub-satellite point is visible");
        let scan = simsat::camera::ScanGrid {
            nx: 2,
            ny: 1,
            x_min: sx,
            y_max: sy,
            pitch_x: 1.0e-5,
            pitch_y: 1.0e-5,
        };
        let raster = SurfaceRaster {
            nx: 2,
            ny: 1,
            scan,
            lat: vec![0.0; 2],
            lon: vec![-75.2; 2],
            grid_i: vec![0.0; 2],
            grid_j: vec![0.0; 2],
            model_scan: None,
            navigation_geometry: None,
        };
        let assemble = |px: usize, _py: usize| SurfacePixel {
            on_earth: true,
            base_srgb: if px == 0 {
                [0.18, 0.32, 0.12]
            } else {
                [0.62, 0.20, 0.08]
            },
            normal_enu: [0.0, 0.0, 1.0],
            sun_enu: [0.0, 0.0, 1.0],
            sun_elev_deg: 90.0,
            sky_openness: 1.0,
            bent_normal_enu: [0.0, 0.0, 1.0],
            ..Default::default()
        };

        let expected: Vec<u8> = (0..scan.nx)
            .flat_map(|px| {
                let (scan_x, scan_y) = scan.scan_angle(px, 0);
                let mut pixel = assemble(px, 0);
                pixel.view_dir = surf.cam.view_dir(scan_x, scan_y);
                shade_surface(&surf, &pixel)
                    .into_iter()
                    .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
            })
            .collect();
        let actual = render_geo_visible_rgba(false, &surf, &raster, &assemble, |_| {
            panic!("cloud volume renderer must not run when Studio clouds are off")
        });

        assert_eq!(actual, expected);
        assert_eq!(actual.len(), 2 * 4);
        assert_ne!(&actual[0..3], &actual[4..7], "surface colors must survive");
    }

    #[test]
    fn rendered_cloud_status_uses_the_requested_toggle_not_frame_buffer_presence() {
        assert!(rendered_clouds_on(true, false, false));
        assert!(!rendered_clouds_on(false, false, false));
        assert!(!rendered_clouds_on(true, true, false));
        assert!(!rendered_clouds_on(true, false, true));
    }

    #[test]
    fn gpu_preview_requires_legacy_full_cell_cloud_coverage() {
        assert!(!gpu_fractional_preview_compatible(true));
        assert!(gpu_fractional_preview_compatible(false));
    }

    #[test]
    fn studio_cloud_transport_matches_rust_cli_and_python_dispatch() {
        assert_eq!(
            studio_cloud_multiscatter_mode(true, false, false, false),
            CloudMultiscatterMode::LegacyOctaves
        );
        assert_eq!(
            studio_cloud_multiscatter_mode(false, false, false, false),
            CloudMultiscatterMode::SingleScatter
        );
        assert_eq!(
            studio_cloud_multiscatter_mode(true, true, false, false),
            CloudMultiscatterMode::DeltaFluxV1,
            "the explicit experimental selection overrides the legacy boolean"
        );
        assert_eq!(
            studio_cloud_multiscatter_mode(false, true, false, false),
            CloudMultiscatterMode::DeltaFluxV1
        );
        assert_eq!(
            studio_cloud_multiscatter_mode(true, true, true, false),
            CloudMultiscatterMode::DeltaFluxV2,
            "the explicitly selected v2 reconstruction has highest precedence"
        );
        assert_eq!(
            studio_cloud_multiscatter_mode(true, true, true, true),
            CloudMultiscatterMode::DeltaFluxV3,
            "the explicitly selected v3 reconstruction has highest precedence"
        );
    }

    #[test]
    fn gpu_preview_never_silently_drops_granulation() {
        assert!(gpu_granulation_preview_compatible(false));
        assert!(!gpu_granulation_preview_compatible(true));
    }

    #[test]
    fn gpu_preview_never_silently_drops_topdown_stratiform_reconstruction() {
        assert!(gpu_topdown_stratiform_preview_compatible(false, false));
        assert!(gpu_topdown_stratiform_preview_compatible(false, true));
        assert!(gpu_topdown_stratiform_preview_compatible(true, false));
        assert!(!gpu_topdown_stratiform_preview_compatible(true, true));
    }

    #[test]
    fn gpu_preview_never_silently_drops_topdown_cloud_footprint() {
        assert!(gpu_topdown_cloud_footprint_preview_compatible(false, false));
        assert!(gpu_topdown_cloud_footprint_preview_compatible(false, true));
        assert!(gpu_topdown_cloud_footprint_preview_compatible(true, false));
        assert!(!gpu_topdown_cloud_footprint_preview_compatible(true, true));
    }

    #[test]
    fn gpu_preview_never_silently_drops_exposed_edge_feathering() {
        assert!(gpu_exposed_edge_feather_compatible(false));
        assert!(!gpu_exposed_edge_feather_compatible(true));
    }

    #[test]
    fn studio_cpu_edge_feather_seam_uses_camera_coverage_and_ignores_perspective_margin() {
        let all_i = vec![5.0f32; 8];
        let all_j = vec![7.0f32; 8];
        let mut exposed_i = all_i.clone();
        exposed_i[0] = f32::NAN;
        let expected = clouds::EDGE_FEATHER_BAND_FRAC * 100.0;

        assert_eq!(
            studio_edge_feather_cells(false, 0.0, 100, 120, false, &exposed_i, &all_j),
            0.0,
            "Off is the exact margin-zero legacy path"
        );
        assert_eq!(
            studio_edge_feather_cells(false, 0.0, 100, 120, true, &exposed_i, &all_j),
            expected,
            "geo coverage activates the opt-in"
        );
        assert_eq!(
            studio_edge_feather_cells(false, 0.0, 100, 120, true, &all_i, &all_j),
            0.0,
            "all-in-domain top-down coverage stays an identity"
        );
        assert_eq!(
            studio_edge_feather_cells(true, 0.5, 100, 120, false, &exposed_i, &all_j),
            0.0,
            "perspective ignores the unrelated margin slider when Off"
        );
        assert_eq!(
            studio_edge_feather_cells(true, 0.5, 100, 120, true, &exposed_i, &all_j),
            expected,
            "perspective still resolves actual exposed camera coverage when On"
        );
    }

    #[test]
    fn shipped_display_reset_and_dirty_check_include_exposure() {
        let shipped = shipped_display_calibration();
        assert_eq!(shipped.0, simsat::render::DEFAULT_EXPOSURE as f32);
        assert_eq!(shipped.1, GROUND_DAY_LIFT as f32);
        assert_eq!(shipped.2, CLOUD_SOFTCLIP_KNEE as f32);
        assert_eq!(shipped.3, RHO_HIGHLIGHT_MAX as f32);
        assert!(!display_calibration_is_dirty(
            shipped.0, shipped.1, shipped.2, shipped.3
        ));
        assert!(display_calibration_is_dirty(
            shipped.0 + 0.25,
            shipped.1,
            shipped.2,
            shipped.3
        ));
    }

    #[test]
    fn gpu_preview_never_ignores_display_calibration() {
        assert!(gpu_cloud_tonemap_compatible(
            CLOUD_SOFTCLIP_KNEE,
            RHO_HIGHLIGHT_MAX
        ));
        // Studio values pass through f32 persistence before capture.
        assert!(gpu_cloud_tonemap_compatible(
            CLOUD_SOFTCLIP_KNEE as f32 as f64,
            RHO_HIGHLIGHT_MAX as f32 as f64
        ));
        assert!(!gpu_cloud_tonemap_compatible(0.75, RHO_HIGHLIGHT_MAX));
        assert!(!gpu_cloud_tonemap_compatible(CLOUD_SOFTCLIP_KNEE, 1.05));

        // The clear-sky surface shader is still an exposure-1.0 / ground-1.0
        // reference. The owner-selected shipped calibration must therefore route to
        // CPU rather than being silently previewed at the wrong brightness.
        assert!(!gpu_surface_display_compatible(
            simsat::render::DEFAULT_EXPOSURE,
            GROUND_DAY_LIFT,
            CLOUD_SOFTCLIP_KNEE,
            RHO_HIGHLIGHT_MAX
        ));
        assert!(gpu_surface_display_compatible(
            1.0,
            1.0,
            CLOUD_SOFTCLIP_KNEE,
            RHO_HIGHLIGHT_MAX
        ));
        assert!(!gpu_surface_display_compatible(
            1.0,
            GROUND_DAY_LIFT,
            CLOUD_SOFTCLIP_KNEE,
            RHO_HIGHLIGHT_MAX
        ));
        assert!(!gpu_surface_display_compatible(
            1.1,
            1.0,
            CLOUD_SOFTCLIP_KNEE,
            RHO_HIGHLIGHT_MAX
        ));
        assert!(!gpu_surface_display_compatible(
            1.0,
            1.1,
            CLOUD_SOFTCLIP_KNEE,
            RHO_HIGHLIGHT_MAX
        ));

        assert!(gpu_land_appearance_compatible(
            LandAppearanceConfig::default()
        ));
        assert!(gpu_land_appearance_compatible(
            LandAppearanceConfig::identity()
        ));
        assert!(gpu_land_appearance_compatible(LandAppearanceConfig {
            sza_normalization: true,
            ..LandAppearanceConfig::identity()
        }));
        assert!(gpu_land_appearance_compatible(LandAppearanceConfig {
            dark_toe: true,
            ..LandAppearanceConfig::identity()
        }));
        assert!(gpu_land_appearance_compatible(LandAppearanceConfig {
            sza_max_gain: f64::NAN,
            dark_toe_knee: f64::INFINITY,
            ..LandAppearanceConfig::default()
        }));
    }

    #[test]
    fn parity_stats_identical_frames_are_zero() {
        // Two identical 3-px frames (one space pixel): zero deltas, space excluded.
        let frame = vec![
            10, 20, 30, 255, // on-earth
            0, 0, 0, 0, // space (alpha 0)
            200, 150, 100, 255, // on-earth
        ];
        let s = parity_stats(&frame, &frame);
        assert_eq!(s.compared, 2);
        assert_eq!(s.mean, [0.0, 0.0, 0.0]);
        assert_eq!(s.p95, [0, 0, 0]);
        assert_eq!(s.max, [0, 0, 0]);
        // The heatmap of identical frames is black everywhere (alpha opaque).
        let heat = parity_heatmap_rgba(&frame, &frame);
        assert_eq!(heat.len(), frame.len());
        for px in heat.chunks_exact(4) {
            assert_eq!(px, [0, 0, 0, 255]);
        }
    }

    #[test]
    fn parity_stats_report_known_deltas() {
        // Two on-earth pixels with known per-channel deltas + one space pixel.
        let cpu = vec![
            100, 100, 100, 255, //
            0, 0, 0, 0, // space in both: excluded
            200, 50, 10, 255,
        ];
        let gpu = vec![
            110, 100, 96, 255, // deltas 10, 0, 4
            0, 0, 0, 0, //
            180, 50, 10, 255, // deltas 20, 0, 0
        ];
        let s = parity_stats(&cpu, &gpu);
        assert_eq!(s.compared, 2);
        assert_eq!(s.mean, [15.0, 0.0, 2.0]);
        assert_eq!(s.max, [20, 0, 4]);
        // Nearest-rank p95 of two samples: ceil(2 * 0.95) = rank 2 = the larger.
        assert_eq!(s.p95, [20, 0, 4]);
        // Heatmap: pixel 0 max-channel delta 10 -> (40, 20, 10); space stays black.
        let heat = parity_heatmap_rgba(&cpu, &gpu);
        assert_eq!(&heat[0..4], &[40, 20, 10, 255]);
        assert_eq!(&heat[4..8], &[0, 0, 0, 255]);
        // Pixel 2 max delta 20 -> (80, 40, 20).
        assert_eq!(&heat[8..12], &[80, 40, 20, 255]);
    }

    #[test]
    fn parity_stats_count_space_disagreement() {
        // A pixel that is space on one side but rendered on the other must count
        // (a masked-vs-rendered disagreement is a real parity break).
        let cpu = vec![0, 0, 0, 0];
        let gpu = vec![60, 0, 0, 255];
        let s = parity_stats(&cpu, &gpu);
        assert_eq!(s.compared, 1);
        assert_eq!(s.max, [60, 0, 0]);
    }

    /// Cursor-centred zoom must keep the image point under the cursor FIXED on screen.
    #[test]
    fn cursor_centred_zoom_keeps_the_point_under_the_cursor() {
        // Viewport-relative cursor and an initial pan; zoom in by ratio 2.0.
        let rel = egui::vec2(120.0, -40.0);
        let pan = egui::vec2(30.0, 10.0);
        let old_scale = 1.5f32;
        for &ratio in &[2.0f32, 0.5, 1.0, 3.7] {
            let pan2 = pan_after_cursor_zoom(pan, rel, ratio);
            let new_scale = old_scale * ratio;
            // The image-space point under the cursor (relative to the image centre):
            // p = (rel - pan) / scale. It must be identical before and after the zoom.
            let p_before = (rel - pan) / old_scale;
            let p_after = (rel - pan2) / new_scale;
            assert!(
                (p_before - p_after).length() < 1e-4,
                "point moved: before {p_before:?} after {p_after:?} (ratio {ratio})"
            );
        }
    }

    /// Pan is clamped so a large image cannot be dragged past its edges, and a fitting
    /// image (<= viewport) is forced to centre (pan 0).
    #[test]
    fn pan_clamp_keeps_the_image_in_bounds() {
        let viewport = egui::vec2(1000.0, 800.0);
        // Image larger than the viewport: pan clamps to +/-(img - viewport)/2.
        let img = egui::vec2(1600.0, 1200.0);
        let clamped = clamp_pan(egui::vec2(9999.0, -9999.0), img, viewport);
        assert!((clamped.x - 300.0).abs() < 1e-4, "x {}", clamped.x); // (1600-1000)/2
        assert!((clamped.y + 200.0).abs() < 1e-4, "y {}", clamped.y); // -(1200-800)/2
        // A within-range pan is untouched.
        let ok = clamp_pan(egui::vec2(100.0, -50.0), img, viewport);
        assert_eq!(ok, egui::vec2(100.0, -50.0));
        // An image that fits the viewport is forced centred (pan 0).
        let small = egui::vec2(500.0, 400.0);
        let centred = clamp_pan(egui::vec2(123.0, -77.0), small, viewport);
        assert_eq!(centred, egui::Vec2::ZERO);
    }

    /// The zoom factor clamps to the fit..MAX_VIEW_ZOOM range.
    #[test]
    fn zoom_clamps_to_range() {
        let z = |v: f32| v.clamp(1.0, MAX_VIEW_ZOOM);
        assert_eq!(z(0.2), 1.0, "cannot zoom out past fit");
        assert_eq!(z(1.0), 1.0);
        assert_eq!(z(4.5), 4.5);
        assert_eq!(z(100.0), MAX_VIEW_ZOOM);
        assert_eq!(MAX_VIEW_ZOOM, 16.0);
    }

    #[test]
    fn expanded_settings_are_bounded_on_reviewed_window_sizes() {
        assert_eq!(settings_scroll_max_height(500.0), 160.0);
        assert!((settings_scroll_max_height(650.0) - 182.0).abs() < f32::EPSILON);
        assert!((settings_scroll_max_height(860.0) - 240.8).abs() < 1e-3);
        assert_eq!(settings_scroll_max_height(1200.0), 250.0);
    }

    #[test]
    fn responsive_toolbar_fits_the_900_and_1360_pixel_rc_widths_without_scrolling() {
        // Content width after the ordinary panel margins in the two RC smoke-test windows.
        let compact = toolbar_layout(884.0);
        assert_eq!(compact.mode_width, 112.0);
        assert!(compact.estimated_selector_width() <= 884.0);

        let regular = toolbar_layout(1344.0);
        assert_eq!(regular.mode_width, 190.0);
        assert!(regular.estimated_selector_width() <= 1344.0);
    }

    /// The studio-side light-LUT twin (used on a geo-LUT cache hit) must stay
    /// BIT-EXACT with the light half of `gpu::build_luts` — the divergence guard
    /// for the WS4 scene cache. Space (NaN) pixels stay zeroed in both.
    #[test]
    fn light_lut_twin_matches_the_engine_bit_exactly() {
        use simsat::camera::ScanGrid;
        let scan = ScanGrid {
            nx: 3,
            ny: 2,
            x_min: -0.01,
            y_max: 0.02,
            pitch_x: 1.0e-5,
            pitch_y: 1.0e-5,
        };
        let raster = SurfaceRaster {
            nx: 3,
            ny: 2,
            scan,
            lat: vec![35.0, 36.5, f32::NAN, 34.25, 40.0, 38.75],
            lon: vec![-97.0, -96.5, f32::NAN, -95.75, -100.0, -98.25],
            grid_i: vec![0.5; 6],
            grid_j: vec![0.5; 6],
            model_scan: None,
            navigation_geometry: None,
        };
        let solar = SolarFrame::new(2025, 6, 21, 2.25);
        let (_geo, engine_light) = gpu::build_luts(&raster, None, 4, 4, &solar);
        let twin = build_light_lut(&raster, &solar);
        assert_eq!(twin.len(), engine_light.len());
        for (i, (a, b)) in twin.iter().zip(engine_light.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "light LUT diverged from the engine at {i}: {a} vs {b}"
            );
        }
        // Sanity: the NaN pixel is zeroed, the finite ones carry a direction.
        assert_eq!(&twin[8..12], &[0.0, 0.0, 0.0, 0.0]);
        assert!(twin[0] != 0.0 || twin[1] != 0.0 || twin[2] != 0.0);
    }
}
