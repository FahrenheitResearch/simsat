//! Top-down map-registered view + WRF-Runner integration output plumbing.
//!
//! This is the RENDER GLUE for the top-down product ([`crate::camera::ViewMode::TopDownMap`]).
//! It does NOT add any new shading: it REUSES the shipped shading kernels —
//! [`crate::render::surface_toa_radiance`] (the M2/M3 surface + atmosphere), the
//! [`crate::clouds::march_cloud`] volumetric cloud march (M4/M5), the M6 IR march
//! [`crate::ir::march_ir_bt`], and the shared tonemap [`crate::render::radiance_to_rgba`]
//! — feeding them PER-PIXEL NADIR RAYS ([`crate::camera::topdown_nadir_ray`]) instead of
//! the geostationary scan-angle rays. The atmosphere/clouds/surface/IR marches are all
//! ray-direction-agnostic (they take an explicit `(cam, view)`), so the top-down path is
//! purely a camera/ray-setup path.
//!
//! WHY a per-pixel camera: the from-space geostationary product is ONE camera (the
//! satellite) with a per-pixel scan-angle ray. The top-down map product is instead a
//! synthetic near-nadir view — each map pixel gets its own camera on the LOCAL VERTICAL
//! above that pixel's ground point, looking straight down. Every ray descends the local
//! vertical and hits the ground exactly at the pixel's lat/lon, so the output is a
//! north-up flat map of the domain that registers with other top-down Lambert WRF
//! field plots (same spherical R = 6.37e6, same projection).
//!
//! AERIAL-PERSPECTIVE NOTE (a documented near-nadir simplification): the geostationary
//! composite adds the froxel camera->cloud front airlight. For the top-down view the
//! camera sits just above the atmosphere looking straight DOWN, so the camera->cloud air
//! column is a thin near-vertical slab through the tenuous upper atmosphere — its in-
//! scatter is negligible next to the geostationary slant path. The top-down composite
//! therefore omits that front-airlight term; the surface's OWN full-column camera->ground
//! aerial perspective (integrated inside [`crate::render::surface_toa_radiance`]) is
//! retained. This is the only physics difference from the geostationary composite.
//!
//! Also holds the two integration OUTPUT helpers used by the headless CLI: the canvas
//! [`letterbox_rgb`] (pad a rendered frame into a fixed figure size, e.g. his uniform
//! 1100x850, aspect preserved, black bars) and the rayon thread cap
//! ([`effective_thread_count`] / [`configure_global_rayon`]) so a 16-way process pool
//! does not oversubscribe.

use rayon::prelude::*;

use crate::camera::topdown_nadir_ray;
use crate::clouds::{CloudScene, ground_cloud_shadow, march_cloud};
use crate::ir::{IrScene, march_ir_bt};
use crate::render::{
    CLOUD_SOFTCLIP_KNEE, FrameContext, GROUND_DAY_LIFT, SurfacePixel, day_lerp_ramp,
    radiance_to_rgba_softclip, reflectance_from_radiance, surface_toa_radiance,
};

/// TOP-DOWN CLOUD NORMALIZATION — a sun-gated multiplier on the top-down (near-nadir)
/// cloud radiance. HISTORY: baked at `0.7` by the topdown-appearance pass as a band-aid
/// for the daytime "white square" (it dimmed the physically-correct nadir cloud radiance
/// so the whole cloud population fell BELOW the tonemap's soft-clip knee — hiding the
/// contrast crush by hiding the brightness). WS2 replaced the crush's root cause with
/// the BOUNDED, exposure-aware highlight shoulder
/// ([`crate::render::soft_clip_highlight`] / [`crate::render::RHO_HIGHLIGHT_MAX`]), and
/// the round-1 A/B (Michael top-down sun40: norm 0.7 median 0.722 / contrast 0.033 vs
/// norm 1.0 median 0.861 / contrast 0.107 — matching the geo view's 0.859 / 0.099)
/// showed the un-normalized radiance + shoulder reproduces the from-space look, so the
/// baked value is now the NEUTRAL `1.0`: the top-down and geostationary views agree and
/// no physically-correct radiance is discarded. The MECHANISM is retained (the
/// `render_frame` `topdown-cloudnorm=` CLI knob still overrides it, sun-gated via
/// [`topdown_cloud_norm`] so a sub-1 override never touches night/twilight).
pub const TOPDOWN_CLOUD_NORM: f64 = 1.0;

/// TOP-DOWN CLOUD NORMALIZATION factor at a sun elevation (deg): `1.0` at/below the
/// twilight band (so night/twilight nadir clouds are byte-unchanged) ramping to
/// `norm_target` at/above the daytime band, via the SAME [`day_lerp_ramp`] the ground
/// lift / veil use. `norm_target = 1.0` is the neutral no-op (`1.0` at every elevation);
/// the baked default is [`TOPDOWN_CLOUD_NORM`]. See [`TOPDOWN_CLOUD_NORM`].
#[inline]
pub fn topdown_cloud_norm(sun_elev_deg: f64, norm_target: f64) -> f64 {
    day_lerp_ramp(sun_elev_deg, norm_target)
}

/// Clone `base` with a different camera ORIGIN (the per-pixel nadir camera). The
/// `FrameContext` holds references (`Copy`) plus small `Copy` fields, so this is cheap;
/// only `cam.camera` varies per pixel (the surface pass reads `ctx.cam.camera` +
/// `px.view_dir`, never `cam.ex/ey/ez`, so the look basis is left as the template's).
fn frame_ctx_with_camera<'a>(base: &FrameContext<'a>, camera: [f64; 3]) -> FrameContext<'a> {
    let mut cam = base.cam;
    cam.camera = camera;
    FrameContext {
        luts: base.luts,
        params: base.params,
        sky_sh: base.sky_sh,
        cam,
        sun_ecef: base.sun_ecef,
        output_transform: base.output_transform,
        bm_present: base.bm_present,
        water_scale: base.water_scale,
        flat_albedo_srgb: base.flat_albedo_srgb,
        raymarch_steps: base.raymarch_steps,
        exposure: base.exposure,
    }
}

/// Render a full TOP-DOWN VISIBLE frame to row-major `Rgba8` bytes (row 0 = north;
/// alpha 0 only for a non-finite/padding pixel, 1 on the domain). `lat`/`lon` are the
/// per-pixel geodetic coordinates of the map raster (`nx * ny`, from
/// [`crate::camera::MapRaster`]); `assemble` supplies the per-pixel surface state (Blue
/// Marble albedo, terrain normal, LANDMASK water, local sun, M3 terrain shadow/aperture,
/// wind) exactly as the geostationary path does. `scene` is `Some` to composite clouds
/// (the M4/M5 march) or `None` for a surface-only top-down (clouds toggled off).
///
/// Per pixel: a nadir ray ([`topdown_nadir_ray`]) into
/// [`surface_toa_radiance`] (surface + atmosphere) and [`march_cloud`] (cloud), composited
/// `L = L_toa * T_cloud + L_cloud` (no front airlight — see the module note) and put
/// through the shared [`radiance_to_rgba_softclip`] tonemap. Rows in parallel (rayon).
#[allow(clippy::too_many_arguments)]
pub fn render_topdown_frame_rgba(
    surf: &FrameContext,
    scene: Option<&CloudScene>,
    lat: &[f32],
    lon: &[f32],
    nx: usize,
    ny: usize,
    assemble: impl Fn(usize, usize) -> SurfacePixel + Sync,
) -> Vec<u8> {
    // Per-scene appearance-pass knobs (constant across the frame; the baked defaults when
    // clouds are off): the sun-gated top-down cloud normalization + the highlight soft-clip.
    let cloud_norm_target = scene
        .map(|s| s.cfg.topdown_cloud_norm)
        .unwrap_or(TOPDOWN_CLOUD_NORM);
    let softclip_knee = scene
        .map(|s| s.cfg.cloud_softclip_knee)
        .unwrap_or(CLOUD_SOFTCLIP_KNEE);
    let rows: Vec<Vec<u8>> = (0..ny)
        .into_par_iter()
        .map(|py| {
            let mut row = Vec::with_capacity(nx * 4);
            for px in 0..nx {
                let rgba = match topdown_pixel_radiance(
                    surf,
                    scene,
                    lat,
                    lon,
                    nx,
                    px,
                    py,
                    &assemble,
                    cloud_norm_target,
                ) {
                    // A map pixel is on-earth; None only if the ray misses the shell / padding.
                    None => [0.0, 0.0, 0.0, 0.0],
                    Some(l) => radiance_to_rgba_softclip(
                        l,
                        surf.output_transform,
                        surf.exposure,
                        softclip_knee,
                    ),
                };
                for &v in &rgba {
                    row.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
                }
            }
            row
        })
        .collect();
    rows.into_iter().flatten().collect()
}

/// Render a full TOP-DOWN VISIBLE frame to row-major RAW REFLECTANCE (`nx*ny*3` f32 in
/// `[0, 1]`, row 0 = north; space/padding pixels are `0`) — the PRE-TONEMAP per-band
/// product the Python binding's `render_visible_bands` returns for the top-down view.
/// Identical assembly to [`render_topdown_frame_rgba`] (same per-pixel nadir ray, same
/// composite via [`topdown_pixel_radiance`]); each pixel's composited radiance is converted
/// to the reflectance factor ([`reflectance_from_radiance`]) instead of the display
/// transform. Rows in parallel (rayon).
pub fn render_topdown_frame_reflectance(
    surf: &FrameContext,
    scene: Option<&CloudScene>,
    lat: &[f32],
    lon: &[f32],
    nx: usize,
    ny: usize,
    assemble: impl Fn(usize, usize) -> SurfacePixel + Sync,
) -> Vec<f32> {
    let rows: Vec<Vec<f32>> = (0..ny)
        .into_par_iter()
        .map(|py| {
            let mut row = vec![0.0f32; nx * 3];
            for px in 0..nx {
                // The RAW-bands product is physical: no top-down cloud normalization
                // (target 1.0 = neutral). The ground lift lives in `surface_toa_radiance`
                // and applies to the reflectance too (like the existing LAND_DAY_GAIN).
                if let Some(l) =
                    topdown_pixel_radiance(surf, scene, lat, lon, nx, px, py, &assemble, 1.0)
                {
                    let rho = reflectance_from_radiance(l);
                    row[px * 3..px * 3 + 3].copy_from_slice(&rho);
                }
            }
            row
        })
        .collect();
    rows.into_iter().flatten().collect()
}

/// The composited top-of-atmosphere LINEAR RADIANCE of one top-down map pixel (surface +
/// cloud, no front airlight — see the module note), before any tonemap/exposure. `None`
/// for a non-finite/padding pixel or a ray that misses the shell. The shared numerator of
/// BOTH the top-down RGB product (-> [`radiance_to_rgba_softclip`]) and the raw-bands product (->
/// [`reflectance_from_radiance`]); a pure extraction of the former per-pixel body, so the
/// RGB output is byte-identical.
#[allow(clippy::too_many_arguments)]
fn topdown_pixel_radiance(
    surf: &FrameContext,
    scene: Option<&CloudScene>,
    lat: &[f32],
    lon: &[f32],
    nx: usize,
    px: usize,
    py: usize,
    assemble: &(impl Fn(usize, usize) -> SurfacePixel + Sync),
    cloud_norm_target: f64,
) -> Option<[f64; 3]> {
    let idx = py * nx + px;
    let (la, lo) = (lat[idx], lon[idx]);
    if !la.is_finite() || !lo.is_finite() {
        return None; // padding / non-finite map pixel (never for an interior domain sample)
    }
    let (cam, view) = topdown_nadir_ray(la as f64, lo as f64);
    let ctx = frame_ctx_with_camera(surf, cam);
    let mut pixel = assemble(px, py);
    pixel.view_dir = view;
    // GROUND LIFT (basemap brightness pass): from the per-scene MarchConfig when clouds are
    // on, else the baked default (a clouds-off top-down basemap still gets the lift).
    let ground_lift = scene
        .map(|s| s.cfg.ground_day_lift)
        .unwrap_or(GROUND_DAY_LIFT);
    let shadow = match scene {
        Some(sc) => ground_cloud_shadow(sc, cam, view),
        None => 1.0,
    };
    let l_toa = surface_toa_radiance(&ctx, &pixel, shadow, ground_lift)?;
    match scene {
        Some(sc) => {
            let m = march_cloud(sc, cam, view);
            if m.transmittance >= 1.0 && m.inscatter == [0.0; 3] {
                Some(l_toa)
            } else {
                // TOP-DOWN CLOUD NORMALIZATION: sun-gated per-pixel scale on the cloud's
                // OWN radiance only (the surface behind, `l_toa * T_cloud`, is untouched).
                // At the neutral `cloud_norm_target = 1.0` (the raw-bands path) or at
                // twilight the factor is 1.0 -> byte-identical to the un-normalized cloud.
                let norm = topdown_cloud_norm(pixel.sun_elev_deg as f64, cloud_norm_target);
                let mut lf = [0.0f64; 3];
                for (c, out) in lf.iter_mut().enumerate() {
                    *out = l_toa[c] * m.transmittance + norm * m.inscatter[c];
                }
                Some(lf)
            }
        }
        None => Some(l_toa),
    }
}

/// Render a full TOP-DOWN IR brightness-temperature plane (Kelvin; `NaN` where a pixel
/// is not in the domain) for a map raster. The top-down IR is a simulated top-down
/// brightness-temperature map — the kind a WRF plotting suite usually approximates
/// from cloud-top fields, but here through the real radiative-transfer march ([`march_ir_bt`])
/// down each pixel's nadir ray. `grid_i` masks out-of-domain pixels (finite = in domain).
/// Rows in parallel; the plane is coloured by an enhancement + written like the
/// geostationary IR plane.
pub fn render_topdown_ir_bt_frame(
    scene: &IrScene,
    lat: &[f32],
    lon: &[f32],
    grid_i: &[f32],
    nx: usize,
    ny: usize,
) -> Vec<f32> {
    let rows: Vec<Vec<f32>> = (0..ny)
        .into_par_iter()
        .map(|py| {
            let mut row = vec![f32::NAN; nx];
            for (px, out) in row.iter_mut().enumerate() {
                let idx = py * nx + px;
                if !grid_i[idx].is_finite() || !lat[idx].is_finite() || !lon[idx].is_finite() {
                    continue; // out of domain -> no IR data (NaN mask)
                }
                let (cam, view) = topdown_nadir_ray(lat[idx] as f64, lon[idx] as f64);
                if let Some(bt) = march_ir_bt(scene, cam, view) {
                    *out = bt as f32;
                }
            }
            row
        })
        .collect();
    rows.into_iter().flatten().collect()
}

// ── integration output helpers (canvas letterbox + rayon thread cap) ──────────

/// The placement of a `sw x sh` image letterboxed (aspect-preserved) into a `tw x th`
/// canvas: `(scaled_w, scaled_h, offset_x, offset_y)`. The scale is `min(tw/sw, th/sh)`
/// so the image fits within the canvas; it is centred, and the leftover is the black
/// letterbox padding. Pure + deterministic (the tested unit).
pub fn letterbox_placement(
    sw: usize,
    sh: usize,
    tw: usize,
    th: usize,
) -> (usize, usize, usize, usize) {
    if sw == 0 || sh == 0 || tw == 0 || th == 0 {
        return (0, 0, 0, 0);
    }
    let scale = (tw as f64 / sw as f64).min(th as f64 / sh as f64);
    let dw = ((sw as f64 * scale).round() as usize).clamp(1, tw);
    let dh = ((sh as f64 * scale).round() as usize).clamp(1, th);
    let ox = (tw - dw) / 2;
    let oy = (th - dh) / 2;
    (dw, dh, ox, oy)
}

/// Letterbox an RGB8 image (`sw x sh`, row-major) into a fixed `tw x th` canvas: scale
/// it (aspect preserved, bilinear, clamp-to-edge) to fit, centre it, and pad the rest
/// BLACK — so a batch of domains of different sizes all become uniform `tw x th` figures
/// (a fixed plotting-suite canvas size). Returns `tw * th * 3` bytes.
pub fn letterbox_rgb(src: &[u8], sw: usize, sh: usize, tw: usize, th: usize) -> Vec<u8> {
    let mut out = vec![0u8; tw * th * 3];
    if sw == 0 || sh == 0 || tw == 0 || th == 0 || src.len() < sw * sh * 3 {
        return out;
    }
    let (dw, dh, ox, oy) = letterbox_placement(sw, sh, tw, th);
    if dw == 0 || dh == 0 {
        return out;
    }
    for y in 0..dh {
        let fy = if dh > 1 {
            y as f64 * (sh - 1) as f64 / (dh - 1) as f64
        } else {
            0.0
        };
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let ty = fy - y0 as f64;
        for x in 0..dw {
            let fx = if dw > 1 {
                x as f64 * (sw - 1) as f64 / (dw - 1) as f64
            } else {
                0.0
            };
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let tx = fx - x0 as f64;
            let o = ((oy + y) * tw + (ox + x)) * 3;
            for c in 0..3 {
                let p = |xx: usize, yy: usize| src[(yy * sw + xx) * 3 + c] as f64;
                let a = p(x0, y0) * (1.0 - tx) + p(x1, y0) * tx;
                let b = p(x0, y1) * (1.0 - tx) + p(x1, y1) * tx;
                out[o + c] = (a * (1.0 - ty) + b * ty).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Resolve the rayon thread cap for a headless batch render from an explicit CLI
/// `threads=` override and the `RAYON_NUM_THREADS` environment value. The CLI override
/// wins; otherwise the env value if it parses to `>= 1`; otherwise `None` (rayon's
/// default = all cores). This lets a driver running many SimSat processes in parallel
/// cap each one to a small share of cores (e.g. `RAYON_NUM_THREADS=1`) so the
/// concurrent processes do not each spin up all cores and thrash. Pure (tested).
pub fn effective_thread_count(cli: Option<usize>, env_val: Option<&str>) -> Option<usize> {
    if let Some(n) = cli {
        return Some(n.max(1));
    }
    match env_val {
        Some(v) => v.trim().parse::<usize>().ok().filter(|&n| n >= 1),
        None => None,
    }
}

/// Configure the GLOBAL rayon pool thread count (call ONCE, before any parallel render).
/// No-op if `cap` is `None` or the pool was already built. Used by the headless render
/// CLI to apply [`effective_thread_count`]. Not unit-tested (it mutates global state that
/// other tests share); the parse/cap logic in [`effective_thread_count`] is the test.
pub fn configure_global_rayon(cap: Option<usize>) {
    if let Some(n) = cap {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(n.max(1))
            .build_global();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atmosphere::{
        AtmosphereLuts, AtmosphereParams, CameraGeometry, OutputTransform, SkyShTable,
        sun_enu_to_ecef,
    };
    use crate::camera::build_map_raster;
    use crate::clouds::{
        CloudScene, DecodedVolume, MarchConfig, OccupancyMip, StepQuality, accumulate_sun_od,
    };
    use crate::frame::{GridGeoref, MapProjection};
    use crate::ir::{IrConfig, IrScene, IrVolume};
    use crate::render::{DEFAULT_EXPOSURE, FLAT_ALBEDO_SRGB, WATER_ALBEDO_SCALE};

    fn test_georef(nx: usize, ny: usize, dx: f64) -> GridGeoref {
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

    /// A tiny all-clear or caller-filled decoded cloud volume.
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
            tau_up: vec![0.0f32; n],
        }
    }

    /// A daytime-overhead FrameContext + assemble closure for a map raster of `nx*ny`.
    /// Sun straight up over the domain centre; flat-albedo land, up normals.
    fn overhead_surf<'a>(
        luts: &'a AtmosphereLuts,
        params: &'a AtmosphereParams,
        sky_sh: &'a SkyShTable,
        sun_ecef: [f64; 3],
    ) -> FrameContext<'a> {
        FrameContext {
            luts,
            params,
            sky_sh,
            cam: CameraGeometry::from_sub_lon(-100.0),
            sun_ecef,
            output_transform: OutputTransform::AbiReflectance,
            bm_present: false,
            water_scale: WATER_ALBEDO_SCALE as f64,
            flat_albedo_srgb: FLAT_ALBEDO_SRGB as f64,
            raymarch_steps: 16,
            exposure: DEFAULT_EXPOSURE,
        }
    }

    // ── top-down cloud normalization (sun-gated appearance pass) ──────────────

    #[test]
    fn topdown_cloud_norm_is_sun_gated_and_neutral_at_one() {
        use crate::render::{AERIAL_VEIL_ELEV_HI_DEG, AERIAL_VEIL_ELEV_LO_DEG};
        // The BAKED value is the neutral 1.0 since WS2 (the 0.7 band-aid was retired for
        // the bounded highlight shoulder): exactly 1.0 at every elevation, so the
        // top-down and geostationary cloud radiance agree.
        assert_eq!(TOPDOWN_CLOUD_NORM, 1.0, "baked norm is neutral since WS2");
        for &elev in &[-10.0f64, 0.0, 20.0, 30.0, 40.0, 90.0] {
            assert_eq!(
                topdown_cloud_norm(elev, TOPDOWN_CLOUD_NORM),
                1.0,
                "neutral no-op at {elev}"
            );
        }
        // The MECHANISM (the `topdown-cloudnorm=` CLI override) stays sun-gated: a sub-1
        // target is exactly 1.0 through the whole twilight band (<= LO — the iteration-1
        // twilight-dimming regression cannot recur), the target at/above HI, strictly
        // < 1 in full daytime, and monotone non-increasing as the sun rises.
        let target = 0.7;
        assert_eq!(topdown_cloud_norm(0.0, target), 1.0);
        assert_eq!(topdown_cloud_norm(AERIAL_VEIL_ELEV_LO_DEG, target), 1.0);
        assert!((topdown_cloud_norm(AERIAL_VEIL_ELEV_HI_DEG, target) - target).abs() < 1e-12);
        assert!(
            topdown_cloud_norm(90.0, target) < 1.0,
            "a sub-1 override must normalize daytime nadir clouds down"
        );
        let mut prev = 2.0;
        for &elev in &[-5.0f64, 15.0, 25.0, 35.0, 45.0, 90.0] {
            let n = topdown_cloud_norm(elev, target);
            assert!(
                n <= prev + 1e-12,
                "not monotone-decreasing at {elev}: {n} > {prev}"
            );
            prev = n;
        }
    }

    // ── letterbox + thread cap (pure) ─────────────────────────────────────────

    #[test]
    fn letterbox_placement_preserves_aspect_and_centres() {
        // A wide 800x400 image into a 1100x850 canvas: scale = min(1.375, 2.125) = 1.375
        // -> 1100x550, centred (ox 0, oy 150). Aspect 2:1 preserved.
        let (dw, dh, ox, oy) = letterbox_placement(800, 400, 1100, 850);
        assert_eq!((dw, dh), (1100, 550));
        assert_eq!((ox, oy), (0, 150));
        // Aspect preserved within rounding.
        let src_aspect = 800.0 / 400.0;
        let dst_aspect = dw as f64 / dh as f64;
        assert!(
            (src_aspect - dst_aspect).abs() < 0.01,
            "{src_aspect} vs {dst_aspect}"
        );
        // A tall 400x800 image into the same canvas fits by height.
        let (dw2, dh2, ox2, oy2) = letterbox_placement(400, 800, 1100, 850);
        assert_eq!(dh2, 850);
        assert!(dw2 <= 1100 && (dw2 as f64 / dh2 as f64 - 0.5).abs() < 0.01);
        assert!(ox2 > 0 && oy2 == 0);
        // A square domain into the same canvas fits by height (850) and centres in x.
        let (dw3, dh3, ox3, _oy3) = letterbox_placement(800, 800, 1100, 850);
        assert_eq!((dw3, dh3), (850, 850));
        assert_eq!(ox3, (1100 - 850) / 2);
    }

    #[test]
    fn letterbox_rgb_pads_black_and_keeps_the_image() {
        // A solid red 4x2 image into a 10x10 canvas: the scaled region is red, the
        // padding is black, the output is exactly 10*10*3 bytes.
        let src = {
            let mut v = vec![0u8; 4 * 2 * 3];
            for px in v.chunks_exact_mut(3) {
                px[0] = 200;
            }
            v
        };
        let out = letterbox_rgb(&src, 4, 2, 10, 10);
        assert_eq!(out.len(), 10 * 10 * 3);
        let (dw, dh, ox, oy) = letterbox_placement(4, 2, 10, 10);
        // A pixel at the centre of the scaled region is red.
        let cx = ox + dw / 2;
        let cy = oy + dh / 2;
        let c = (cy * 10 + cx) * 3;
        assert!(
            out[c] > 150 && out[c + 1] < 50 && out[c + 2] < 50,
            "centre not red"
        );
        // A corner pixel (outside the scaled region for this aspect) is black.
        assert_eq!(&out[0..3], &[0, 0, 0], "top-left corner not black");
    }

    #[test]
    fn effective_thread_count_prefers_cli_then_env() {
        // CLI override wins (and clamps to >= 1).
        assert_eq!(effective_thread_count(Some(4), Some("16")), Some(4));
        assert_eq!(effective_thread_count(Some(0), None), Some(1));
        // Else the env value, if it parses to >= 1.
        assert_eq!(effective_thread_count(None, Some("1")), Some(1));
        assert_eq!(effective_thread_count(None, Some("8")), Some(8));
        assert_eq!(effective_thread_count(None, Some(" 2 ")), Some(2));
        // Unset / unparseable / zero env -> None (rayon default).
        assert_eq!(effective_thread_count(None, None), None);
        assert_eq!(effective_thread_count(None, Some("")), None);
        assert_eq!(effective_thread_count(None, Some("all")), None);
        assert_eq!(effective_thread_count(None, Some("0")), None);
    }

    // ── top-down render integration (reuses the shading kernels) ──────────────

    #[test]
    fn topdown_surface_frame_is_lit_and_clear_cloud_matches_surface_only() {
        let (nx, ny, nz) = (16, 16, 24);
        let dz = 250.0;
        let georef = test_georef(nx, ny, 3000.0);
        let map = build_map_raster(&georef, nx, ny, nx, ny, 0.0).unwrap();

        let params = AtmosphereParams::default();
        let luts = AtmosphereLuts::build(&params);
        let sky_sh = SkyShTable::build(&luts, &params, 16);
        // Sun straight up over the domain centre (a clean daytime overhead sun).
        let sun_ecef = sun_enu_to_ecef([0.0, 0.0, 1.0], 45.0, -100.0);
        let surf = overhead_surf(&luts, &params, &sky_sh, sun_ecef);

        // A simple assemble: flat-albedo land, up normal, sun straight overhead.
        let assemble = |_px: usize, _py: usize| SurfacePixel {
            on_earth: true,
            base_srgb: [FLAT_ALBEDO_SRGB, FLAT_ALBEDO_SRGB, FLAT_ALBEDO_SRGB],
            normal_enu: [0.0, 0.0, 1.0],
            sun_enu: [0.0, 0.0, 1.0],
            sun_elev_deg: 90.0,
            is_water: false,
            ..Default::default()
        };

        // Surface-only top-down (clouds off).
        let surf_only =
            render_topdown_frame_rgba(&surf, None, &map.lat, &map.lon, nx, ny, assemble);
        assert_eq!(surf_only.len(), nx * ny * 4);
        // Every pixel is on the domain (alpha 255) and the daytime surface is lit.
        assert!(
            surf_only.chunks_exact(4).all(|p| p[3] == 255),
            "a pixel is space"
        );
        let peak = surf_only
            .chunks_exact(4)
            .map(|p| p[0] as u32 + p[1] as u32 + p[2] as u32)
            .max()
            .unwrap();
        assert!(peak > 0, "the daytime surface should be lit");

        // A CLEAR cloud volume adds nothing -> byte-identical to surface-only.
        let vol = build_volume(nx, ny, nz, dz, 3000.0, |_, _, _| (0.0, 0.0, 0.0));
        let mip = OccupancyMip::build(&vol, 4);
        let sun_od = accumulate_sun_od(&vol, &georef, sun_ecef, 32);
        let scene = CloudScene {
            vol: &vol,
            mip: &mip,
            sun_od: &sun_od,
            georef: &georef,
            luts: &luts,
            sky_sh: &sky_sh,
            sun_ecef,
            cfg: MarchConfig::new(StepQuality::Offline, vol.voxel_pitch_m()),
        };
        let with_clear =
            render_topdown_frame_rgba(&surf, Some(&scene), &map.lat, &map.lon, nx, ny, assemble);
        assert_eq!(
            with_clear, surf_only,
            "a clear cloud volume must not change the top-down frame"
        );
    }

    #[test]
    fn topdown_cloud_changes_the_covered_pixels() {
        let (nx, ny, nz) = (16, 16, 24);
        let dz = 250.0;
        let georef = test_georef(nx, ny, 3000.0);
        let map = build_map_raster(&georef, nx, ny, nx, ny, 0.0).unwrap();
        let params = AtmosphereParams::default();
        let luts = AtmosphereLuts::build(&params);
        let sky_sh = SkyShTable::build(&luts, &params, 16);
        let sun_ecef = sun_enu_to_ecef([0.0, 0.0, 1.0], 45.0, -100.0);
        let surf = overhead_surf(&luts, &params, &sky_sh, sun_ecef);
        let assemble = |_px: usize, _py: usize| SurfacePixel {
            on_earth: true,
            base_srgb: [FLAT_ALBEDO_SRGB, FLAT_ALBEDO_SRGB, FLAT_ALBEDO_SRGB],
            normal_enu: [0.0, 0.0, 1.0],
            sun_enu: [0.0, 0.0, 1.0],
            sun_elev_deg: 90.0,
            is_water: false,
            ..Default::default()
        };
        let clear = render_topdown_frame_rgba(&surf, None, &map.lat, &map.lon, nx, ny, assemble);

        // A thick liquid cloud filling the middle of every column (visible from directly
        // above): the top-down frame must differ from the clear surface where it covers.
        let vol = build_volume(nx, ny, nz, dz, 3000.0, |_, _, k| {
            if (nz / 2..nz / 2 + 6).contains(&k) {
                (3.0e-3, 0.0, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        });
        let mip = OccupancyMip::build(&vol, 4);
        let sun_od = accumulate_sun_od(&vol, &georef, sun_ecef, 32);
        let scene = CloudScene {
            vol: &vol,
            mip: &mip,
            sun_od: &sun_od,
            georef: &georef,
            luts: &luts,
            sky_sh: &sky_sh,
            sun_ecef,
            cfg: MarchConfig::new(StepQuality::Offline, vol.voxel_pitch_m()),
        };
        let cloudy =
            render_topdown_frame_rgba(&surf, Some(&scene), &map.lat, &map.lon, nx, ny, assemble);
        assert_eq!(cloudy.len(), clear.len());
        let differ = cloudy
            .iter()
            .zip(clear.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(differ > 0, "the cloud did not change any top-down pixel");
    }

    #[test]
    fn topdown_reflectance_is_finite_in_zero_one_and_lit() {
        // The raw-bands (pre-tonemap reflectance) top-down product: every on-domain pixel
        // is finite and in [0, 1], a lit daytime surface has a positive reflectance, and a
        // CLEAR cloud volume yields byte-identical reflectance to the surface-only path
        // (the composite reduces to the surface at T_cloud = 1).
        let (nx, ny, nz) = (16, 16, 24);
        let dz = 250.0;
        let georef = test_georef(nx, ny, 3000.0);
        let map = build_map_raster(&georef, nx, ny, nx, ny, 0.0).unwrap();
        let params = AtmosphereParams::default();
        let luts = AtmosphereLuts::build(&params);
        let sky_sh = SkyShTable::build(&luts, &params, 16);
        let sun_ecef = sun_enu_to_ecef([0.0, 0.0, 1.0], 45.0, -100.0);
        let surf = overhead_surf(&luts, &params, &sky_sh, sun_ecef);
        let assemble = |_px: usize, _py: usize| SurfacePixel {
            on_earth: true,
            base_srgb: [FLAT_ALBEDO_SRGB, FLAT_ALBEDO_SRGB, FLAT_ALBEDO_SRGB],
            normal_enu: [0.0, 0.0, 1.0],
            sun_enu: [0.0, 0.0, 1.0],
            sun_elev_deg: 90.0,
            is_water: false,
            ..Default::default()
        };
        let refl =
            render_topdown_frame_reflectance(&surf, None, &map.lat, &map.lon, nx, ny, assemble);
        assert_eq!(refl.len(), nx * ny * 3);
        assert!(
            refl.iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
        );
        assert!(
            refl.iter().cloned().fold(0.0f32, f32::max) > 0.0,
            "the lit daytime surface should have a positive reflectance"
        );

        // A clear cloud volume must not change the reflectance vs surface-only.
        let vol = build_volume(nx, ny, nz, dz, 3000.0, |_, _, _| (0.0, 0.0, 0.0));
        let mip = OccupancyMip::build(&vol, 4);
        let sun_od = accumulate_sun_od(&vol, &georef, sun_ecef, 32);
        let scene = CloudScene {
            vol: &vol,
            mip: &mip,
            sun_od: &sun_od,
            georef: &georef,
            luts: &luts,
            sky_sh: &sky_sh,
            sun_ecef,
            cfg: MarchConfig::new(StepQuality::Offline, vol.voxel_pitch_m()),
        };
        let refl_clear = render_topdown_frame_reflectance(
            &surf,
            Some(&scene),
            &map.lat,
            &map.lon,
            nx,
            ny,
            assemble,
        );
        assert_eq!(
            refl, refl_clear,
            "a clear cloud volume must not change reflectance"
        );
    }

    #[test]
    fn topdown_ir_bt_matches_skin_temperature_over_clear_ground() {
        // A top-down IR map over a warm clear ground reads BT ~ TSK at every pixel; over
        // a cold thick anvil it reads the cold cloud-top T. This exercises the top-down
        // IR renderer end-to-end (nadir rays into the M6 march).
        let (nx, ny, nz) = (12, 12, 40);
        let dz = 250.0;
        let tsk = 298.0f32;
        let georef = test_georef(nx, ny, 3000.0);
        let map = build_map_raster(&georef, nx, ny, nx, ny, 0.0).unwrap();

        // Clear warm ground, cool lapse-rate air.
        let ext_l = vec![0.0f32; nx * ny * nz];
        let mut ext_i = vec![0.0f32; nx * ny * nz];
        let ext_p = vec![0.0f32; nx * ny * nz];
        let mut temp = vec![0.0f32; nx * ny * nz];
        let qv = vec![0.0f32; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let c = (k * ny + j) * nx + i;
                    temp[c] = (288.0 - 6.5 * (k as f64 * dz / 1000.0)) as f32;
                }
            }
        }
        let clear_vol = IrVolume {
            nx,
            ny,
            nz,
            z_min_m: 0.0,
            dz_m: dz,
            horiz_pitch_m: 3000.0,
            ext_liquid: ext_l.clone(),
            ext_ice: ext_i.clone(),
            ext_precip: ext_p.clone(),
            temperature_k: temp.clone(),
            qvapor: qv.clone(),
            tsk: vec![tsk; nx * ny],
        };
        let dv = DecodedVolume {
            nx,
            ny,
            nz,
            z_min_m: 0.0,
            dz_m: dz,
            horiz_pitch_m: 3000.0,
            ext_liquid: ext_l.clone(),
            ext_ice: ext_i.clone(),
            ext_precip: ext_p.clone(),
            tau_up: vec![0.0; nx * ny * nz],
        };
        let mip = OccupancyMip::build(&dv, 8);
        let mut cfg = IrConfig::band13();
        cfg.wv_continuum = false;
        cfg.surface_emissivity = 1.0;
        let scene = IrScene {
            vol: &clear_vol,
            mip: &mip,
            georef: &georef,
            cfg,
        };
        let bt = render_topdown_ir_bt_frame(&scene, &map.lat, &map.lon, &map.grid_i, nx, ny);
        assert_eq!(bt.len(), nx * ny);
        let centre = bt[(ny / 2) * nx + nx / 2];
        assert!(
            (centre as f64 - tsk as f64).abs() < 0.5,
            "clear top-down IR BT {centre} != TSK {tsk}"
        );

        // A cold thick ice anvil in the top half: the covered pixels read the cold top.
        let cloud_top = 215.0f32;
        for k in nz / 2..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let c = (k * ny + j) * nx + i;
                    ext_i[c] = 3.0e-2;
                    temp[c] = cloud_top;
                }
            }
        }
        let anvil_vol = IrVolume {
            nx,
            ny,
            nz,
            z_min_m: 0.0,
            dz_m: dz,
            horiz_pitch_m: 3000.0,
            ext_liquid: ext_l,
            ext_ice: ext_i.clone(),
            ext_precip: ext_p,
            temperature_k: temp,
            qvapor: qv,
            tsk: vec![tsk; nx * ny],
        };
        let dv2 = DecodedVolume {
            nx,
            ny,
            nz,
            z_min_m: 0.0,
            dz_m: dz,
            horiz_pitch_m: 3000.0,
            ext_liquid: vec![0.0; nx * ny * nz],
            ext_ice: ext_i,
            ext_precip: vec![0.0; nx * ny * nz],
            tau_up: vec![0.0; nx * ny * nz],
        };
        let mip2 = OccupancyMip::build(&dv2, 8);
        let scene2 = IrScene {
            vol: &anvil_vol,
            mip: &mip2,
            georef: &georef,
            cfg,
        };
        let bt2 = render_topdown_ir_bt_frame(&scene2, &map.lat, &map.lon, &map.grid_i, nx, ny);
        let centre2 = bt2[(ny / 2) * nx + nx / 2];
        assert!(
            (centre2 as f64 - cloud_top as f64).abs() < 2.0,
            "anvil top-down IR BT {centre2} != cloud-top T {cloud_top}"
        );
    }
}
