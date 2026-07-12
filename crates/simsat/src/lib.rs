//! SimSat engine.
//!
//! Physically-based simulated visible/IR satellite imagery from WRF output.
//! See `docs/simsat-engine-plan.md` for the full design; section 10 is the
//! milestone build order. M0 delivers the `.ssb` volume-brick format and the
//! streaming wrfout ingest:
//!
//! - [`optics`] — hydrometeor optics constants + CPU reference kernels.
//! - [`frame`] — analytic WRF map-projection forward/inverse + the ratchet.
//! - [`bricks`] — the `.ssb` on-disk format (log quantization, f16, manifest).
//! - [`ingest`] — streaming `WrfFile::read_var` -> brick, peak-RSS disciplined.
//! - [`platform`] — peak-RSS query + below-normal ingest thread priority.
//!
//! M1 adds the studio's first-frame path (design doc section 10, M1 row):
//!
//! - [`camera`] — geostationary fixed-grid camera (ported CGMS scan-angle math)
//!   + the per-pixel surface raster (scan -> lat/lon -> WRF `(i, j)`).
//! - [`solar`] — NOAA/Meeus solar position for the per-pixel sun direction.
//! - [`render`] — CPU-reference surface shading kernel + HGT terrain normals.
//! - [`bluemarble`] — the single-month dev ground texture loader (crop/resample).
//! - [`gpu`] — the wgpu surface pipeline + `gpu/shaders/surface.wgsl`.
//! - [`store_out`] — the sat-store visible-frame writer (`rgb_r/g/b` planes).
//!
//! M2 adds the Hillaire clear-sky atmosphere (design doc section 3):
//!
//! - [`atmosphere`] — CPU reference for the transmittance / multiple-scattering /
//!   sky-view LUTs, the aerial-perspective froxel volume, the scattering raymarch,
//!   the finite solar disk + twilight, the WRF precipitable-water modulation, and
//!   the ABI-like reflectance/tonemap. The WGSL twin is `gpu/shaders/surface.wgsl`.
//!
//! M4 adds the volumetric cloud raymarch (design doc section 4; built before M3 by
//! owner call — weather visuals first, and M4 depends only on M2's atmosphere):
//!
//! - [`clouds`] — CPU reference for the true-slant ECEF cloud march: brick decode +
//!   trilinear sampling, occupancy mip, sun optical-depth map, dual-lobe HG phase,
//!   beer-powder, scalar sky ambient, and the composite over the M2 surface with
//!   froxel aerial perspective. The WGSL twin is `gpu/shaders/clouds.wgsl` (a
//!   superset of the surface pass) + `gpu/shaders/sun_od.wgsl` (the sun-OD compute).
//!
//! M6 adds the synthetic IR band (ABI band 13, 10.3 um; design doc section 7):
//!
//! - [`ir`] — the synthetic-IR radiative-transfer pass: a top-down slant-ray
//!   gray-body emission march (`optics` Planck + per-class 10.3 um absorption +
//!   weak WV continuum + surface term) producing a true-Kelvin brightness-
//!   temperature plane. Thermal — works day AND night. CPU is the shipping path
//!   (like M4/M5); a WGSL twin is deferred to a future M6-GPU pass.
//! - [`ir_enhance`] — the BT -> RGB enhancements (Grayscale/BD/Rainbow/CIMSS/AVN/
//!   Funktop) ported from BowEcho, applied via `rw_sat::palette::anchor_color`.
//!   `store_out` writes the BT plane as a single-band Kelvin frame so BowEcho (or
//!   the studio) re-enhances it live.
//! - [`wv`] — the WATER-VAPOR bands (ABI 8/9/10 = 6.2/6.9/7.3 um): a generalization of
//!   the [`ir`] window march where the dominant emitter is water vapor (owner decision
//!   6). [`wv::WvBand`] builds the per-band [`ir::IrConfig`] (wavelength + strong per-band
//!   WV mass-absorption from [`optics`]) so the SAME march produces the WV weighting-
//!   function BT plane; `ir_enhance`'s CIMSS = the classic WV moisture palette.
//!
//! M3 adds penumbral terrain shadows, Cox-Munk water glint, the SNOWH snow blend,
//! and the terrain ambient-aperture that completes M5's SH-2 sky ambient (design
//! doc section 5/6):
//!
//! - [`horizon`] — the per-domain horizon map (16-azimuth terrain horizon elevation
//!   angles with earth curvature, CPU/rayon): the penumbral cast-shadow fraction and
//!   the Oat & Sander ambient-aperture cone. `optics` gained the Cox-Munk (1954)
//!   wind-slope glint + Fresnel constants; `render` gained the SNOWH ramp, and its
//!   `surface_toa_radiance` now folds the terrain shadow, the aperture-occluded SH
//!   ambient, and the water glint + Fresnel sky reflection (the CPU shipping path;
//!   the WGSL twins mirror it, GPU activation deferred like M4/M5/M6).
//!
//! M7 (seasonal-ground slice) adds the 12-month Blue Marble pack + day-of-year lerp:
//!
//! - [`bluemarble`] gains the mid-month-anchored day-of-year -> two-month blend
//!   ([`bluemarble::month_blend`]) and the per-texel season blend
//!   ([`bluemarble::blend_crops`] / [`bluemarble::load_season_crop`]), baked into the
//!   crop so [`bluemarble::BlueMarbleCrop::sample_bilinear`] samples the seasonal
//!   ground behind its unchanged signature.
//! - [`asset_pack`] — the downloadable 12-month asset pack: the embedded pinned
//!   manifest (month -> SHA-256 -> GitHub/NASA URLs), the streaming download + SHA-256
//!   gate (ported from BowEcho `self_update.rs`), and the vendored 8 km emergency
//!   fallback so an offline render never hard-fails.
//!
//! Top-down map view (the WRF-Runner integration product) adds the second output mode:
//!
//! - [`camera`] gains [`camera::ViewMode`] + the top-down map raster
//!   ([`camera::MapRaster`] / [`camera::build_map_raster`]) over the WRF domain's own
//!   Lambert extent, north-up, and the per-pixel nadir ray ([`camera::topdown_nadir_ray`]).
//! - [`topdown`] — the top-down RENDER GLUE: it feeds those nadir rays into the SAME
//!   shipped shading kernels (`render::surface_toa_radiance`, `clouds::march_cloud`,
//!   `ir::march_ir_bt`, `render::radiance_to_rgba`) — a camera/ray-setup path, not new
//!   shading — plus the integration output helpers (canvas letterbox + rayon thread cap).
//!
//! Derived scalar-field map products (the last of the product suite) add three top-down,
//! map-registered scalar fields for the WRF-Runner plotter:
//!
//! - [`derived`] — [`derived::DerivedField`] (precipitable water mm, cloud-top temperature K,
//!   cloud optical depth) computed as per-column vertical integrals / marches through the
//!   brick ([`derived::compute_field`]), resampled onto the output raster
//!   ([`derived::resample_field`]) via the raster's fractional WRF indices. The RAW `f32`
//!   field is the primary deliverable; [`derived::colorize`] is a basic studio colormap.
//!
//! Operational-model ingest (HRRR/RRFS GRIB2) parallels the wrfout path:
//!
//! - [`ingest_grib`] — streaming NOAA HRRR / RRFS native-level GRIB2 -> the SAME
//!   `.ssb` brick + `run.json` (an operational brick is indistinguishable from a
//!   WRF brick downstream). Seek-indexed one-message-at-a-time decode via the
//!   pinned `grib-core`, per-species extinction through the same [`optics`]
//!   constants, the same integral-conserving vertical resample, valid time from
//!   the GRIB headers.
//!
//! The high-level render API (the reusable assembly behind the examples + the Python
//! binding) sits on top of all of the above:
//!
//! - [`api`] — [`api::render`] takes a [`api::RenderParams`] + a [`api::Product`]
//!   (`VisibleRgb` / `RgbReflectance` / `Ir`) and returns the frame DATA as owned arrays +
//!   a [`api::Georef`] (projection params + `imshow` extent + the H x W lat/lon mesh),
//!   NOT a PNG. `examples/render_frame.rs` + `examples/render_ir.rs` and the PyO3 crate
//!   `simsat_py` (`import simsat`) both call it, so there is ONE render code path. The RGB
//!   product is byte-identical to the shipped display (it calls
//!   `clouds::render_cloud_frame_rgba` / `topdown::render_topdown_frame_rgba`); the bands
//!   product is the pre-tonemap reflectance; IR is the raw Kelvin BT (+ optional colored
//!   enhancement).

pub mod animation;
pub mod api;
pub mod asset_pack;
pub mod atmosphere;
pub mod bluemarble;
pub mod bricks;
pub mod camera;
pub mod cloud_delta_flux;
pub mod clouds;
pub mod derived;
pub mod fractional_clouds;
pub mod frame;
pub mod geocolor;
pub mod gpu;
pub mod horizon;
pub mod ingest;
pub mod ingest_grib;
pub mod instrument_footprint;
pub mod ir;
pub mod ir_enhance;
pub mod log;
pub mod optics;
pub mod platform;
pub mod precision_audit;
pub mod render;
pub mod sandwich;
pub mod solar;
pub mod store_out;
pub mod thermal_sensor;
pub mod topdown;
pub mod web_layer;
pub mod wv;

/// On-disk format version for `.ssb` volume bricks + the `run.json` manifest
/// (design doc section 2).
///
/// Bump this whenever the brick payload layout, channel set, quantization header,
/// or manifest schema changes in a way that older readers cannot parse. Caches are
/// regenerable dev artifacts, so there is no migration code — a version mismatch is
/// rejected on read and the cache is regenerated.
///
/// v2 (M0/M1-review fixes): the vertical resample is integral-conserving (brick
/// extinction/qvapor content changed vs v1 point-sampling), and `ManifestTimestep`
/// gained a full-datetime `key` (was `hhmm`-only, which collided across days).
///
/// v3 (the snow-optics fix): QSNOW moved OUT of `ext_ice` (where it shared
/// cloud-ice optics, inflating snow's visible extinction 3.75x — the "clouds too
/// thick" defect) and INTO `ext_precip` at its own aggregate beta (rho_w-normalized
/// r_e = 150 um, 10 m^2 kg^-1); `tau_up` is recomputed from the corrected total.
/// The channel SET and layout are unchanged — only the extinction CONTENT — but a
/// v2 brick renders wrong optical depths, so v2 caches are refused on read and
/// re-ingested from the source wrfout.
///
/// v4 (fractional-cloud foundation): `ext_snow` records a QSNOW-only auxiliary
/// subset while the legacy `ext_precip` channel remains the total large-particle
/// extinction (rain + graupel + snow). This deliberate duplicate
/// keeps legacy/GPU/IR totals byte-compatible while allowing fractional-cloud
/// rendering to scale just the snow share. A linear-u8 `cloud_fraction` channel
/// and explicit source provenance distinguish real model coverage from the
/// all-covered fallback. The payload layout therefore changes; v3 bricks are
/// refused and regenerable source-backed caches self-heal by re-ingesting.
///
/// v5 (fraction semantics): the byte layout is intentionally unchanged, but WRF
/// zero-fraction condensate cells are repaired with the scheme-consistent Xu-Randall
/// diagnostic and HRRR native cloud fractions are vertically remapped with the same
/// overlap semantics used by the renderer. A v4 brick therefore contains materially
/// different cloud-coverage content and must be regenerated even though a v5 reader
/// could parse its bytes.
///
/// v6 (GRIB snow identity): HRRR/RRFS SNMR is now preserved in the existing
/// `ext_snow` auxiliary subset instead of being discarded after its contribution
/// to total `ext_precip`. The byte layout is unchanged, but v5 GRIB bricks cannot
/// support species-correct IR/fractional optics and must be regenerated.
pub const SSB_FORMAT_VERSION: u32 = 6;

/// Four-byte magic at the head of every `.ssb` brick file: `SSB` + format epoch.
pub const SSB_MAGIC: [u8; 4] = *b"SSB1";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssb_format_version_is_v6() {
        // v6 invalidates v5 GRIB snow semantics without changing its channel layout.
        assert_eq!(SSB_FORMAT_VERSION, 6);
    }

    #[test]
    fn ssb_magic_is_four_ascii_bytes() {
        assert_eq!(SSB_MAGIC.len(), 4);
        assert_eq!(&SSB_MAGIC, b"SSB1");
        assert!(SSB_MAGIC.iter().all(u8::is_ascii));
    }
}
