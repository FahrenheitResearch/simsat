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

use crate::atmosphere::OutputTransform;
use crate::camera::topdown_nadir_ray;
use crate::clouds::{CloudScene, ground_cloud_shadow, march_cloud};
use crate::ir::{IrScene, march_ir_bt};
#[cfg(test)]
use crate::render::{CLOUD_SOFTCLIP_KNEE, GROUND_DAY_LIFT};
use crate::render::{
    FrameContext, SurfacePixel, apply_low_sun_illuminant, day_lerp_ramp,
    effective_cloud_shadow_layer, radiance_to_rgba_softclip_with_synthetic_green,
    reflectance_from_radiance, surface_toa_radiance,
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
        ground_day_lift: base.ground_day_lift,
        cloud_softclip_knee: base.cloud_softclip_knee,
        cloud_highlight_max: base.cloud_highlight_max,
        synthetic_green: base.synthetic_green,
        atmosphere_correction: base.atmosphere_correction,
        terrain_atmosphere: base.terrain_atmosphere,
        land_appearance: base.land_appearance,
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
        .unwrap_or(surf.cloud_softclip_knee);
    let highlight_max = scene
        .map(|s| s.cfg.cloud_highlight_max)
        .unwrap_or(surf.cloud_highlight_max);
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
                    Some((l, sun_elev_deg)) => {
                        // Low-sun illuminant correction at the display seam (every map
                        // pixel is on-earth); identity outside the 2-30 deg band. The
                        // raw-bands path below does NOT apply it (physical product).
                        let l = apply_low_sun_illuminant(l, true, sun_elev_deg as f64, surf.luts);
                        radiance_to_rgba_softclip_with_synthetic_green(
                            l,
                            surf.output_transform,
                            surf.exposure,
                            softclip_knee,
                            highlight_max,
                            scene
                                .map(|s| s.cfg.synthetic_green)
                                .unwrap_or(surf.synthetic_green),
                        )
                    }
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

/// Render the finished top-down visible frame with an optional, DISPLAY-only finite
/// cloud footprint. The surface remains sampled at the output pixel centre and is never
/// filtered. Only the cloud radiance residual is filtered, in linear radiance before the
/// shared low-sun correction and tonemap:
///
/// `L_out = L_surface_unshadowed + footprint(L_cloud_composite - L_surface_unshadowed)`.
///
/// The residual therefore includes ground cloud shadow, cloud attenuation, and cloud
/// in-scatter. The separable seven-tap binomial kernel (`[1 6 15 20 15 6 1] / 64`,
/// sigma ~= 1.225 px) is the bounded match to the owner-reviewed sigma-1.25 prototype.
/// Missing/padding neighbours retain their weight on the centre pixel; this makes each
/// one-dimensional pass symmetric and stochastic, preserving both constant fields and
/// the signed residual sum (to floating-point roundoff) without leaking cloud into map
/// padding. `enabled=false` and cloud-free renders take the exact legacy path above.
#[allow(clippy::too_many_arguments)]
pub fn render_topdown_frame_rgba_with_cloud_footprint(
    surf: &FrameContext,
    scene: Option<&CloudScene>,
    lat: &[f32],
    lon: &[f32],
    nx: usize,
    ny: usize,
    assemble: impl Fn(usize, usize) -> SurfacePixel + Sync,
    enabled: bool,
) -> Vec<u8> {
    if !enabled || scene.is_none() {
        return render_topdown_frame_rgba(surf, scene, lat, lon, nx, ny, assemble);
    }

    let scene = scene.expect("scene checked above");
    let cloud_norm_target = scene.cfg.topdown_cloud_norm;
    let softclip_knee = scene.cfg.cloud_softclip_knee;
    let highlight_max = scene.cfg.cloud_highlight_max;
    let synthetic_green = scene.cfg.synthetic_green;

    let sample_rows: Vec<Vec<TopdownRadianceComponents>> = (0..ny)
        .into_par_iter()
        .map(|py| {
            (0..nx)
                .map(|px| {
                    topdown_pixel_radiance_components(
                        surf,
                        scene,
                        lat,
                        lon,
                        nx,
                        px,
                        py,
                        &assemble,
                        cloud_norm_target,
                    )
                    .unwrap_or_default()
                })
                .collect()
        })
        .collect();
    let samples: Vec<TopdownRadianceComponents> = sample_rows.into_iter().flatten().collect();
    let valid: Vec<bool> = samples.iter().map(|s| s.valid).collect();
    let residual: Vec<[f64; 3]> = samples
        .iter()
        .map(|s| {
            if s.valid {
                [
                    s.composite[0] - s.base[0],
                    s.composite[1] - s.base[1],
                    s.composite[2] - s.base[2],
                ]
            } else {
                [0.0; 3]
            }
        })
        .collect();
    let filtered = filter_cloud_radiance_residual(&residual, &valid, nx, ny);

    let before = residual_sum(&residual, &valid);
    let after = residual_sum(&filtered, &valid);
    let max_relative_drift = (0..3)
        .map(|c| (after[c] - before[c]).abs() / before[c].abs().max(1.0e-12))
        .fold(0.0f64, f64::max);
    crate::log_line!(
        "simsat topdown: cloud radiance footprint ON (7-tap binomial, sigma 1.225 px); \
         signed residual sum rgb [{:.6e}, {:.6e}, {:.6e}] -> \
         [{:.6e}, {:.6e}, {:.6e}] (max relative drift {:.3e})",
        before[0],
        before[1],
        before[2],
        after[0],
        after[1],
        after[2],
        max_relative_drift,
    );

    let rows: Vec<Vec<u8>> = (0..ny)
        .into_par_iter()
        .map(|py| {
            let mut row = Vec::with_capacity(nx * 4);
            for px in 0..nx {
                let idx = py * nx + px;
                let s = samples[idx];
                let rgba = if !s.valid {
                    [0.0, 0.0, 0.0, 0.0]
                } else {
                    let l = [
                        s.base[0] + filtered[idx][0],
                        s.base[1] + filtered[idx][1],
                        s.base[2] + filtered[idx][2],
                    ];
                    let l = apply_low_sun_illuminant(l, true, s.sun_elev_deg as f64, surf.luts);
                    radiance_to_rgba_softclip_with_synthetic_green(
                        l,
                        surf.output_transform,
                        surf.exposure,
                        softclip_knee,
                        highlight_max,
                        synthetic_green,
                    )
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

#[derive(Clone, Copy, Debug)]
struct TopdownRadianceComponents {
    /// Unshadowed surface + atmosphere radiance. This is the sharp, immutable base.
    base: [f64; 3],
    /// Full legacy cloud composite, including cloud shadow/attenuation/in-scatter.
    composite: [f64; 3],
    sun_elev_deg: f32,
    valid: bool,
}

impl Default for TopdownRadianceComponents {
    fn default() -> Self {
        Self {
            base: [0.0; 3],
            composite: [0.0; 3],
            sun_elev_deg: 0.0,
            valid: false,
        }
    }
}

/// Compute the sharp unshadowed surface and the complete legacy cloud composite for one
/// map pixel. This is used only by the opt-in footprint path, so the default renderer
/// retains its exact single-surface-evaluation behavior and bytes.
#[allow(clippy::too_many_arguments)]
fn topdown_pixel_radiance_components(
    surf: &FrameContext,
    scene: &CloudScene,
    lat: &[f32],
    lon: &[f32],
    nx: usize,
    px: usize,
    py: usize,
    assemble: &(impl Fn(usize, usize) -> SurfacePixel + Sync),
    cloud_norm_target: f64,
) -> Option<TopdownRadianceComponents> {
    let idx = py * nx + px;
    let (la, lo) = (lat[idx], lon[idx]);
    if !la.is_finite() || !lo.is_finite() {
        return None;
    }
    let (cam, view) = topdown_nadir_ray(la as f64, lo as f64);
    let ctx = frame_ctx_with_camera(surf, cam);
    let mut pixel = assemble(px, py);
    pixel.view_dir = view;
    let ground_lift = scene.cfg.ground_day_lift;

    let base = surface_toa_radiance(&ctx, &pixel, 1.0, ground_lift)?;
    let shadow = ground_cloud_shadow(scene, cam, view);
    let shadowed_surface = if shadow == 1.0 {
        base
    } else {
        surface_toa_radiance(&ctx, &pixel, shadow, ground_lift)?
    };
    let marched = march_cloud(scene, cam, view);
    let composite = if marched.transmittance >= 1.0 && marched.inscatter == [0.0; 3] {
        shadowed_surface
    } else {
        let norm = topdown_cloud_norm(pixel.sun_elev_deg as f64, cloud_norm_target);
        [
            shadowed_surface[0] * marched.transmittance + norm * marched.inscatter[0],
            shadowed_surface[1] * marched.transmittance + norm * marched.inscatter[1],
            shadowed_surface[2] * marched.transmittance + norm * marched.inscatter[2],
        ]
    };
    Some(TopdownRadianceComponents {
        base,
        composite,
        sun_elev_deg: pixel.sun_elev_deg,
        valid: true,
    })
}

/// Separable sigma~=1.225 pixel cloud-residual footprint. At an image or padding edge,
/// the unavailable tap weight stays on the centre sample. The resulting matrix is
/// symmetric and each row sums to one, so every pass is also column-stochastic and
/// preserves signed residual sum while never bleeding into invalid padding.
fn filter_cloud_radiance_residual(
    input: &[[f64; 3]],
    valid: &[bool],
    nx: usize,
    ny: usize,
) -> Vec<[f64; 3]> {
    debug_assert_eq!(input.len(), nx * ny);
    debug_assert_eq!(valid.len(), nx * ny);
    const WEIGHTS: [f64; 7] = [
        1.0 / 64.0,
        6.0 / 64.0,
        15.0 / 64.0,
        20.0 / 64.0,
        15.0 / 64.0,
        6.0 / 64.0,
        1.0 / 64.0,
    ];

    let horizontal_rows: Vec<Vec<[f64; 3]>> = (0..ny)
        .into_par_iter()
        .map(|y| {
            let mut row = vec![[0.0; 3]; nx];
            for (x, out) in row.iter_mut().enumerate() {
                let idx = y * nx + x;
                if !valid[idx] {
                    continue;
                }
                for (tap, weight) in WEIGHTS.iter().copied().enumerate() {
                    let dx = tap as isize - 3;
                    let neighbour = x
                        .checked_add_signed(dx)
                        .filter(|&xx| xx < nx)
                        .map(|xx| y * nx + xx)
                        .filter(|&j| valid[j])
                        .unwrap_or(idx);
                    for c in 0..3 {
                        out[c] += weight * input[neighbour][c];
                    }
                }
            }
            row
        })
        .collect();
    let horizontal: Vec<[f64; 3]> = horizontal_rows.into_iter().flatten().collect();

    let vertical_rows: Vec<Vec<[f64; 3]>> = (0..ny)
        .into_par_iter()
        .map(|y| {
            let mut row = vec![[0.0; 3]; nx];
            for (x, out) in row.iter_mut().enumerate() {
                let idx = y * nx + x;
                if !valid[idx] {
                    continue;
                }
                for (tap, weight) in WEIGHTS.iter().copied().enumerate() {
                    let dy = tap as isize - 3;
                    let neighbour = y
                        .checked_add_signed(dy)
                        .filter(|&yy| yy < ny)
                        .map(|yy| yy * nx + x)
                        .filter(|&j| valid[j])
                        .unwrap_or(idx);
                    for c in 0..3 {
                        out[c] += weight * horizontal[neighbour][c];
                    }
                }
            }
            row
        })
        .collect();
    vertical_rows.into_iter().flatten().collect()
}

fn residual_sum(values: &[[f64; 3]], valid: &[bool]) -> [f64; 3] {
    values
        .iter()
        .zip(valid)
        .filter(|(_, ok)| **ok)
        .fold([0.0; 3], |mut sum, (v, _)| {
            for c in 0..3 {
                sum[c] += v[c];
            }
            sum
        })
}

/// Render a full TOP-DOWN VISIBLE frame to row-major RAW REFLECTANCE (`nx*ny*3` f32 in
/// `[0, 1]`, row 0 = north; space/padding pixels are `0`) — the PRE-TONEMAP per-band
/// product the Python binding's `render_rgb_reflectance` returns for the top-down view
/// (`render_visible_bands` is the deprecated compatibility alias).
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
                // (target 1.0 = neutral) and NO low-sun illuminant correction (that is
                // display-side). The ground lift lives in `surface_toa_radiance`
                // and applies to the reflectance too (like the existing LAND_DAY_GAIN).
                if let Some((l, _sun_elev)) =
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

/// Render unclamped linear radiance for deterministic fractional-subcolumn
/// integration. `cloud_norm_target` is the same display-only cloud normalization
/// used by [`render_topdown_frame_rgba`]; pass `1.0` for the physical raw-bands
/// product. The caller averages explicit subcolumns and tonemaps once.
#[allow(clippy::too_many_arguments)]
pub fn render_topdown_frame_linear_radiance(
    surf: &FrameContext,
    scene: Option<&CloudScene>,
    lat: &[f32],
    lon: &[f32],
    nx: usize,
    ny: usize,
    assemble: impl Fn(usize, usize) -> SurfacePixel + Sync,
    cloud_norm_target: f64,
) -> (Vec<f64>, Vec<u8>) {
    let rows: Vec<(Vec<f64>, Vec<u8>)> = (0..ny)
        .into_par_iter()
        .map(|py| {
            let mut radiance = vec![0.0f64; nx * 3];
            let mut alpha = vec![0u8; nx];
            for px in 0..nx {
                if let Some((l, _)) = topdown_pixel_radiance(
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

/// The composited top-of-atmosphere LINEAR RADIANCE of one top-down map pixel (surface +
/// cloud, no front airlight — see the module note), before any tonemap/exposure, PLUS the
/// pixel's sun elevation (deg — the low-sun illuminant correction's per-pixel input at
/// the RGB display seam). `None` for a non-finite/padding pixel or a ray that misses the
/// shell. The shared numerator of BOTH the top-down RGB product (->
/// [`radiance_to_rgba_softclip`]) and the raw-bands product (->
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
) -> Option<([f64; 3], f32)> {
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
        .unwrap_or(surf.ground_day_lift);
    let shadow = match scene {
        Some(sc) => ground_cloud_shadow(sc, cam, view),
        None => 1.0,
    };
    let l_toa = surface_toa_radiance(&ctx, &pixel, shadow, ground_lift)?;
    let elev = pixel.sun_elev_deg;
    match scene {
        Some(sc) => {
            let m = march_cloud(sc, cam, view);
            if m.transmittance >= 1.0 && m.inscatter == [0.0; 3] {
                Some((l_toa, elev))
            } else {
                // TOP-DOWN CLOUD NORMALIZATION: sun-gated per-pixel scale on the cloud's
                // OWN radiance only (the surface behind, `l_toa * T_cloud`, is untouched).
                // At the neutral `cloud_norm_target = 1.0` (the raw-bands path) or at
                // twilight the factor is 1.0 -> byte-identical to the un-normalized cloud.
                let norm = topdown_cloud_norm(elev as f64, cloud_norm_target);
                let mut lf = [0.0f64; 3];
                for (c, out) in lf.iter_mut().enumerate() {
                    *out = l_toa[c] * m.transmittance + norm * m.inscatter[c];
                }
                Some((lf, elev))
            }
        }
        None => Some((l_toa, elev)),
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

// ── cloud LAYER render (the web-map compositing product) ──────────────────────

/// The NATIVE (Lambert-map-raster) cloud layer pair one top-down render produces —
/// the cloud field ONLY (no Blue Marble, no surface, no atmosphere veil on the
/// ground: the HOST map is the ground) plus the ground cloud-shadow multiply field.
/// Both share the same raster (they came from ONE set of marches per pixel), so a
/// host that composites shadow-multiply-then-cloud-over gets registered layers.
/// [`crate::web_layer`] reprojects this onto a Web-Mercator delivery grid.
#[derive(Debug, Clone)]
pub struct CloudLayerFrame {
    pub nx: usize,
    pub ny: usize,
    /// PREMULTIPLIED-alpha RGBA (`nx*ny*4`, row 0 = north): the color channels hold
    /// the TONEMAPPED CLOUD-ONLY RADIANCE — the additive term of the shipped
    /// composite `L = L_bg*T_cloud + L_cloud` — and alpha = `1 - T_cloud` (the view
    /// transmittance the march computes). "Premultiplied" in the compositing sense:
    /// a host composites `src + dst*(1 - a)`; the color is NOT numerically
    /// multiplied by alpha (it already IS the additive term). Convert with
    /// [`crate::web_layer::unpremultiply_rgba`] for straight-alpha hosts/PNG.
    pub rgba_premul: Vec<u8>,
    /// The ground cloud-shadow MULTIPLY field (`nx*ny`, `[0,1]`, 1.0 = no shadow):
    /// the host-safe EFFECTIVE shadow from the penumbral sun-OD field. It is neutral at/
    /// below the horizon and reaches the shipped direct-term floor by 12 degrees, so a
    /// host never darkens an already-nighttime basemap. Out-of-coverage pixels are
    /// `1.0` (neutral).
    pub shadow: Vec<f32>,
}

/// Render the TOP-DOWN CLOUD LAYER (cloud field only + ground shadow) for a map
/// raster — the web-map compositing product. Per pixel: a nadir ray
/// ([`topdown_nadir_ray`]) through the SAME [`march_cloud`] the shipped composite
/// uses; alpha = `1 - T_cloud`; the cloud color is the cloud's own in-scattered
/// radiance through the SAME display seam as the shipped top-down product (the
/// sun-gated top-down cloud normalization, the low-sun illuminant correction, and
/// the exposure + softclip tonemap) — so where the cloud is opaque the layer matches
/// the shipped composite pixel-for-pixel.
///
/// ILLUMINATION HONESTY (documented decision): the cloud radiance is physical — the
/// M5 multi-octave sun term plus the SH sky ambient, whose ground-bounce component
/// assumes OUR ground albedo (`MarchConfig::ground_albedo`), not the host basemap's.
/// That second-order ambient mismatch is the honest price of compositing over a
/// ground we did not render; the sun-lit term (which dominates any visible cloud)
/// is unaffected. The tonemap is display-nonlinear, so host compositing in display
/// space approximates the linear-radiance composite — exact at alpha 0 and 1,
/// closest in between for the near-nadir view this layer is defined for.
///
/// The shadow plane is [`CloudLayerFrame::shadow`] (the surface-consumed effective
/// shadow); it is computed for EVERY on-map pixel, including clear ones (a clear
/// pixel can still sit in another cloud's shadow — that is the point of the layer).
/// Rows in parallel (rayon).
pub fn render_cloud_layer_frame(
    scene: &CloudScene,
    output_transform: OutputTransform,
    exposure: f64,
    lat: &[f32],
    lon: &[f32],
    nx: usize,
    ny: usize,
) -> CloudLayerFrame {
    let sun = scene.sun_ecef;
    let rows: Vec<(Vec<u8>, Vec<f32>)> = (0..ny)
        .into_par_iter()
        .map(|py| {
            let mut rgba_row = vec![0u8; nx * 4];
            let mut shadow_row = vec![1.0f32; nx];
            for (px, shadow_px) in shadow_row.iter_mut().enumerate() {
                let idx = py * nx + px;
                let (la, lo) = (lat[idx], lon[idx]);
                if !la.is_finite() || !lo.is_finite() {
                    continue; // padding: transparent + neutral shadow
                }
                let (cam, view) = topdown_nadir_ray(la as f64, lo as f64);
                // Per-pixel sun elevation (local up . ECEF sun) — drives the shadow
                // host-layer shadow fade, cloud-norm sun gate, and illuminant correction.
                let (lar, lor) = ((la as f64).to_radians(), (lo as f64).to_radians());
                let up = [lar.cos() * lor.cos(), lar.cos() * lor.sin(), lar.sin()];
                let mu = up[0] * sun[0] + up[1] * sun[1] + up[2] * sun[2];
                let elev = mu.clamp(-1.0, 1.0).asin().to_degrees();

                // The ground shadow field (computed for clear pixels too).
                let raw_shadow = ground_cloud_shadow(scene, cam, view);
                *shadow_px = effective_cloud_shadow_layer(raw_shadow, elev) as f32;

                let m = march_cloud(scene, cam, view);
                let alpha = (1.0 - m.transmittance).clamp(0.0, 1.0);
                if alpha <= 0.0 && m.inscatter == [0.0; 3] {
                    continue; // clear: fully transparent (the shadow still applies)
                }
                // The cloud's OWN radiance through the shipped display seam (see the
                // fn doc). The norm is the same sun-gated per-pixel factor the
                // composited top-down product applies to its cloud term.
                let norm = topdown_cloud_norm(elev, scene.cfg.topdown_cloud_norm);
                let l = [
                    norm * m.inscatter[0],
                    norm * m.inscatter[1],
                    norm * m.inscatter[2],
                ];
                let l = apply_low_sun_illuminant(l, true, elev, scene.luts);
                let disp = radiance_to_rgba_softclip_with_synthetic_green(
                    l,
                    output_transform,
                    exposure,
                    scene.cfg.cloud_softclip_knee,
                    scene.cfg.cloud_highlight_max,
                    scene.cfg.synthetic_green,
                );
                let o = px * 4;
                for c in 0..3 {
                    rgba_row[o + c] = (disp[c].clamp(0.0, 1.0) * 255.0).round() as u8;
                }
                rgba_row[o + 3] = (alpha * 255.0).round() as u8;
            }
            (rgba_row, shadow_row)
        })
        .collect();
    let mut rgba_premul = Vec::with_capacity(nx * ny * 4);
    let mut shadow = Vec::with_capacity(nx * ny);
    for (r, s) in rows {
        rgba_premul.extend_from_slice(&r);
        shadow.extend_from_slice(&s);
    }
    CloudLayerFrame {
        nx,
        ny,
        rgba_premul,
        shadow,
    }
}

// ── free PERSPECTIVE render (tier 2: the angled-3D hero shot) ──────────────────

/// Render a FULL-COMPOSITE frame (surface + atmosphere + clouds over our Blue Marble
/// ground — the hero-shot product) through a free [`PerspectiveBasis`] camera: the
/// SAME ray-agnostic marches as the geostationary/top-down products, fed the pinhole
/// ray fan ([`PerspectiveBasis::pixel_ray`]) instead of scan-angle/nadir rays.
///
/// CONTRACT: `surf.cam.camera` MUST already be the camera EYE (`basis.eye`) — the
/// surface march integrates aerial perspective along `ctx.cam.camera + t*view_dir`,
/// so the eye is set ONCE per frame by the caller (no per-pixel context clone; one
/// eye, unlike the per-pixel nadir cameras of the top-down path). `assemble` supplies
/// the per-pixel surface state from the PERSPECTIVE raster
/// ([`crate::camera::build_perspective_raster`]): ground pixels carry the Blue
/// Marble/terrain/sun state at the ray's earth intersection, sky pixels are
/// `on_earth = false` and composite the existing limb/space handling (a ray that
/// grazes the atmosphere shell renders the limb; one that misses it entirely is
/// space, alpha 0). Near-horizon sky rays can still cross the CLOUD shell — the
/// cloud march runs for every ray, so elevated cloud rises honestly above the limb.
///
/// DOCUMENTED DIVERGENCE (the top-down precedent): the froxel camera->cloud front
/// airlight is omitted — the aerial-perspective froxel is built for the geostationary
/// scan camera and does not describe a free eye. The surface term keeps its FULL
/// per-ray eye->ground aerial perspective (integrated inside `surface_toa_radiance`
/// along the actual slant ray), so the ground haze is honest; only the extra airlight
/// IN FRONT OF the cloud body is dropped: `L = L_toa * T_cloud + L_cloud`.
///
/// PARALLAX HONESTY: high cloud displaces against the ground with the view geometry
/// (the rays are true 3-D lines through the volume) — physical and intended; this is
/// the 3D look. The camera pose is recorded by the caller (api `Georef::camera_pose`
/// + the render log). Rows in parallel (rayon).
pub fn render_perspective_frame_rgba(
    surf: &FrameContext,
    scene: Option<&CloudScene>,
    basis: &crate::camera::PerspectiveBasis,
    assemble: impl Fn(usize, usize) -> SurfacePixel + Sync,
) -> Vec<u8> {
    let (nx, ny) = (basis.width, basis.height);
    let ground_lift = scene
        .map(|s| s.cfg.ground_day_lift)
        .unwrap_or(surf.ground_day_lift);
    let softclip_knee = scene
        .map(|s| s.cfg.cloud_softclip_knee)
        .unwrap_or(surf.cloud_softclip_knee);
    let highlight_max = scene
        .map(|s| s.cfg.cloud_highlight_max)
        .unwrap_or(surf.cloud_highlight_max);
    let rows: Vec<Vec<u8>> = (0..ny)
        .into_par_iter()
        .map(|py| {
            let mut row = Vec::with_capacity(nx * 4);
            for px in 0..nx {
                let view = basis.pixel_ray(px, py);
                let mut pixel = assemble(px, py);
                pixel.view_dir = view;
                let shadow = match scene {
                    Some(sc) => ground_cloud_shadow(sc, basis.eye, view),
                    None => 1.0,
                };
                let rgba = match surface_toa_radiance(surf, &pixel, shadow, ground_lift) {
                    None => [0.0, 0.0, 0.0, 0.0], // space beyond the atmosphere
                    Some(l_toa) => {
                        let l = match scene {
                            Some(sc) => {
                                let m = march_cloud(sc, basis.eye, view);
                                if m.transmittance >= 1.0 && m.inscatter == [0.0; 3] {
                                    l_toa
                                } else {
                                    // No froxel front airlight (see the fn doc); the
                                    // cloud radiance is the geo product's neutral one
                                    // (the top-down cloud norm is a nadir mechanism).
                                    [
                                        l_toa[0] * m.transmittance + m.inscatter[0],
                                        l_toa[1] * m.transmittance + m.inscatter[1],
                                        l_toa[2] * m.transmittance + m.inscatter[2],
                                    ]
                                }
                            }
                            None => l_toa,
                        };
                        // The same display seam as the geo composite: illuminant
                        // correction on-earth only (the limb keeps its physical
                        // color), then the exposure + softclip tonemap.
                        let l = apply_low_sun_illuminant(
                            l,
                            pixel.on_earth,
                            pixel.sun_elev_deg as f64,
                            surf.luts,
                        );
                        radiance_to_rgba_softclip_with_synthetic_green(
                            l,
                            surf.output_transform,
                            surf.exposure,
                            softclip_knee,
                            highlight_max,
                            scene
                                .map(|s| s.cfg.synthetic_green)
                                .unwrap_or(surf.synthetic_green),
                        )
                    }
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

/// Render the CLOUD-LAYER-ONLY variant through a free perspective camera: the cloud
/// field alone as PREMULTIPLIED-alpha RGBA (`width*height*4`; color = the tonemapped
/// cloud radiance, alpha = `1 - T_cloud` — the [`CloudLayerFrame`] semantics), for
/// compositing over a HOST 3-D map rendered with a matching camera. No ground-shadow
/// plane is produced for the perspective variant (a screen-space multiply layer only
/// describes shadows on OUR ground raster; a host 3-D scene shades its own terrain) —
/// the nadir [`render_cloud_layer_frame`] product is the shadow-bearing one.
///
/// The per-pixel display-seam sun elevation is evaluated at the CLOUD'S OWN position
/// (the march's transmittance-weighted centroid along the ray) rather than a ground
/// point — a sky-crossing ray has no ground point, and the illuminant correction
/// belongs to the cloud being lit. Rows in parallel (rayon).
pub fn render_perspective_cloud_layer(
    scene: &CloudScene,
    basis: &crate::camera::PerspectiveBasis,
    output_transform: OutputTransform,
    exposure: f64,
) -> Vec<u8> {
    let (nx, ny) = (basis.width, basis.height);
    let sun = scene.sun_ecef;
    let rows: Vec<Vec<u8>> = (0..ny)
        .into_par_iter()
        .map(|py| {
            let mut row = vec![0u8; nx * 4];
            for px in 0..nx {
                let view = basis.pixel_ray(px, py);
                let m = march_cloud(scene, basis.eye, view);
                let alpha = (1.0 - m.transmittance).clamp(0.0, 1.0);
                if alpha <= 0.0 && m.inscatter == [0.0; 3] {
                    continue; // clear/space: fully transparent
                }
                // Sun elevation at the cloud centroid (local up . ECEF sun).
                let p = [
                    basis.eye[0] + view[0] * m.mean_t_m,
                    basis.eye[1] + view[1] * m.mean_t_m,
                    basis.eye[2] + view[2] * m.mean_t_m,
                ];
                let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt().max(1.0);
                let mu = (p[0] * sun[0] + p[1] * sun[1] + p[2] * sun[2]) / r;
                let elev = mu.clamp(-1.0, 1.0).asin().to_degrees();
                let l = apply_low_sun_illuminant(m.inscatter, true, elev, scene.luts);
                let disp = radiance_to_rgba_softclip_with_synthetic_green(
                    l,
                    output_transform,
                    exposure,
                    scene.cfg.cloud_softclip_knee,
                    scene.cfg.cloud_highlight_max,
                    scene.cfg.synthetic_green,
                );
                let o = px * 4;
                for c in 0..3 {
                    row[o + c] = (disp[c].clamp(0.0, 1.0) * 255.0).round() as u8;
                }
                row[o + 3] = (alpha * 255.0).round() as u8;
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
    use crate::camera::{PerspectiveCamera, build_map_raster, build_perspective_raster};
    use crate::clouds::{
        CloudScene, DecodedVolume, MarchConfig, OccupancyMip, StepQuality, accumulate_sun_od,
    };
    use crate::frame::{GridGeoref, MapProjection};
    use crate::ir::{IrConfig, IrScene, IrVolume};
    use crate::render::{DEFAULT_EXPOSURE, FLAT_ALBEDO_SRGB, WATER_ALBEDO_SCALE};

    #[test]
    fn cloud_radiance_footprint_is_the_separable_seven_tap_binomial() {
        let (nx, ny) = (9usize, 9usize);
        let mut impulse = vec![[0.0; 3]; nx * ny];
        impulse[4 * nx + 4] = [64.0, 32.0, -16.0];
        let valid = vec![true; nx * ny];
        let got = filter_cloud_radiance_residual(&impulse, &valid, nx, ny);
        let w = [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0];
        for y in 1..=7 {
            for x in 1..=7 {
                let expected_r = w[y - 1] * w[x - 1] / 64.0;
                let px = got[y * nx + x];
                assert!((px[0] - expected_r).abs() < 1.0e-12);
                assert!((px[1] - 0.5 * expected_r).abs() < 1.0e-12);
                assert!((px[2] + 0.25 * expected_r).abs() < 1.0e-12);
            }
        }
        let sum = residual_sum(&got, &valid);
        assert!((sum[0] - 64.0).abs() < 1.0e-12);
        assert!((sum[1] - 32.0).abs() < 1.0e-12);
        assert!((sum[2] + 16.0).abs() < 1.0e-12);
    }

    #[test]
    fn cloud_radiance_footprint_preserves_constants_and_energy_at_padding() {
        let (nx, ny) = (11usize, 8usize);
        let mut valid = vec![true; nx * ny];
        // Irregular padding notch plus the usual invalid outer corner.
        for (x, y) in [(0, 0), (1, 0), (0, 1), (5, 3), (5, 4), (6, 4)] {
            valid[y * nx + x] = false;
        }
        let input: Vec<[f64; 3]> = valid
            .iter()
            .map(|ok| if *ok { [0.75, -0.25, 1.5] } else { [0.0; 3] })
            .collect();
        let got = filter_cloud_radiance_residual(&input, &valid, nx, ny);
        for (px, ok) in got.iter().zip(&valid) {
            if *ok {
                assert_eq!(*px, [0.75, -0.25, 1.5]);
            } else {
                assert_eq!(*px, [0.0; 3]);
            }
        }
        let before = residual_sum(&input, &valid);
        let after = residual_sum(&got, &valid);
        for c in 0..3 {
            assert!((after[c] - before[c]).abs() < 1.0e-10);
        }
    }

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
            ext_snow: vec![0; n],
            ext_snow_quant: crate::bricks::LogQuant {
                vmin: 0.0,
                vmax: 0.0,
            },
            science_ext_snow: Vec::new(),
            ext_precip,
            tau_up: vec![0.0f32; n],
            cloud_fraction: vec![255; n],
            has_cloud_fraction: false,
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
            ground_day_lift: GROUND_DAY_LIFT,
            cloud_softclip_knee: CLOUD_SOFTCLIP_KNEE,
            cloud_highlight_max: crate::render::RHO_HIGHLIGHT_MAX,
            synthetic_green: false,
            atmosphere_correction: true,
            terrain_atmosphere: true,
            land_appearance: crate::render::LandAppearanceConfig::identity(),
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
        let mut clear_cfg = MarchConfig::new(StepQuality::Offline, vol.voxel_pitch_m());
        // This test owns an analytic slab; do not inherit the product appearance scale.
        clear_cfg.cloud_optical_depth_scale = 1.0;
        let scene = CloudScene {
            vol: &vol,
            mip: &mip,
            sun_od: &sun_od,
            georef: &georef,
            luts: &luts,
            sky_sh: &sky_sh,
            sun_ecef,
            cfg: clear_cfg,
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
            ext_snow: vec![0.0; nx * ny * nz],
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
            ext_snow: vec![0; nx * ny * nz],
            ext_snow_quant: crate::bricks::LogQuant {
                vmin: 0.0,
                vmax: 0.0,
            },
            science_ext_snow: Vec::new(),
            ext_precip: ext_p.clone(),
            tau_up: vec![0.0; nx * ny * nz],
            cloud_fraction: vec![255; nx * ny * nz],
            has_cloud_fraction: false,
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
            ext_snow: vec![0.0; nx * ny * nz],
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
            ext_snow: vec![0; nx * ny * nz],
            ext_snow_quant: crate::bricks::LogQuant {
                vmin: 0.0,
                vmax: 0.0,
            },
            science_ext_snow: Vec::new(),
            ext_precip: vec![0.0; nx * ny * nz],
            tau_up: vec![0.0; nx * ny * nz],
            cloud_fraction: vec![255; nx * ny * nz],
            has_cloud_fraction: false,
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

    // ── cloud LAYER render (the web-map compositing product) ──────────────────

    #[test]
    fn cloud_layer_clear_volume_is_fully_transparent_and_unshadowed() {
        let (nx, ny, nz) = (16, 16, 24);
        let georef = test_georef(nx, ny, 3000.0);
        let map = build_map_raster(&georef, nx, ny, nx, ny, 0.0).unwrap();
        let params = AtmosphereParams::default();
        let luts = AtmosphereLuts::build(&params);
        let sky_sh = SkyShTable::build(&luts, &params, 16);
        let sun_ecef = sun_enu_to_ecef([0.0, 0.0, 1.0], 45.0, -100.0);
        let vol = build_volume(nx, ny, nz, 250.0, 3000.0, |_, _, _| (0.0, 0.0, 0.0));
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
        let layer = render_cloud_layer_frame(
            &scene,
            OutputTransform::AbiReflectance,
            crate::render::DEFAULT_EXPOSURE,
            &map.lat,
            &map.lon,
            nx,
            ny,
        );
        assert_eq!(layer.rgba_premul.len(), nx * ny * 4);
        assert_eq!(layer.shadow.len(), nx * ny);
        // A clear volume is a fully TRANSPARENT layer (exact zeros — a host
        // compositing it changes nothing) and a fully NEUTRAL shadow layer.
        assert!(
            layer.rgba_premul.iter().all(|&v| v == 0),
            "clear layer must be exactly transparent"
        );
        assert!(
            layer.shadow.iter().all(|&s| s == 1.0),
            "clear layer must cast no shadow"
        );
    }

    #[test]
    fn cloud_layer_thick_cloud_is_opaque_lit_and_casts_shadow() {
        let (nx, ny, nz) = (16, 16, 24);
        let georef = test_georef(nx, ny, 3000.0);
        let map = build_map_raster(&georef, nx, ny, nx, ny, 0.0).unwrap();
        let params = AtmosphereParams::default();
        let luts = AtmosphereLuts::build(&params);
        let sky_sh = SkyShTable::build(&luts, &params, 16);
        let sun_ecef = sun_enu_to_ecef([0.0, 0.0, 1.0], 45.0, -100.0);
        // A thick liquid cloud in the middle levels of the CENTRE 6x6 cells only —
        // covered pixels must be near-opaque and lit; far pixels exactly transparent.
        let lo = nx / 2 - 3;
        let hi = nx / 2 + 3;
        let vol = build_volume(nx, ny, nz, 250.0, 3000.0, |i, j, k| {
            let mid = nz / 2..nz / 2 + 6;
            if (lo..hi).contains(&i) && (lo..hi).contains(&j) && mid.contains(&k) {
                (3.0e-3, 0.0, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        });
        let mip = OccupancyMip::build(&vol, 4);
        let sun_od = accumulate_sun_od(&vol, &georef, sun_ecef, 64);
        let scene = CloudScene {
            vol: &vol,
            mip: &mip,
            sun_od: &sun_od,
            georef: &georef,
            luts: &luts,
            sky_sh: &sky_sh,
            sun_ecef,
            cfg: MarchConfig {
                // The opacity/shadow assertions below use the raw analytic OD 4.5.
                cloud_optical_depth_scale: 1.0,
                ..MarchConfig::new(StepQuality::Offline, vol.voxel_pitch_m())
            },
        };
        let layer = render_cloud_layer_frame(
            &scene,
            OutputTransform::AbiReflectance,
            crate::render::DEFAULT_EXPOSURE,
            &map.lat,
            &map.lon,
            nx,
            ny,
        );
        // The domain-centre pixel: covered, optically thick (OD = 6 cells * 250 m *
        // 3e-3 = 4.5 -> alpha ~ 0.99) and sun-lit (positive tonemapped color).
        let c = ((ny / 2) * nx + nx / 2) * 4;
        assert!(
            layer.rgba_premul[c + 3] > 240,
            "thick cloud alpha {} not near-opaque",
            layer.rgba_premul[c + 3]
        );
        assert!(
            layer.rgba_premul[c] > 0 && layer.rgba_premul[c + 1] > 0,
            "overhead-sun cloud should be lit: {:?}",
            &layer.rgba_premul[c..c + 4]
        );
        // The cloud casts a real shadow under an overhead sun (the centre column).
        assert!(
            layer.shadow[(ny / 2) * nx + nx / 2] < 0.9,
            "no ground shadow under a thick overhead cloud: {}",
            layer.shadow[(ny / 2) * nx + nx / 2]
        );
        // A far corner pixel (outside the cloud + its shadow) is exactly transparent
        // with a neutral shadow.
        let far = (nx + 1) * 4;
        assert_eq!(
            &layer.rgba_premul[far..far + 4],
            &[0, 0, 0, 0],
            "clear pixel not transparent"
        );
        assert!(
            layer.shadow[nx + 1] > 0.99,
            "clear far pixel should be unshadowed: {}",
            layer.shadow[nx + 1]
        );
    }

    // ── free perspective render (tier 2) ───────────────────────────────────────

    /// An oblique perspective camera 150 km over the test domain's south, looking at
    /// the domain-centre ground point (45 N, -100 E) with a WIDE fov, so ONE frame
    /// spans all three ray classes: ground hits (bottom), atmosphere-grazing limb
    /// (mid), and space above the shell (top — the 150 km eye sits above the 100 km
    /// atmosphere, so upward rays miss it entirely).
    fn oblique_camera() -> PerspectiveCamera {
        PerspectiveCamera {
            eye_lat_deg: 43.0,
            eye_lon_deg: -100.0,
            eye_alt_m: 150_000.0,
            look_lat_deg: 45.0,
            look_lon_deg: -100.0,
            look_alt_m: 0.0,
            fov_deg: 100.0,
            width: 96,
            height: 96,
        }
    }

    /// The raster-driven assemble for a perspective frame: ground pixels are lit
    /// flat land under an overhead sun; sky pixels are the off-earth default.
    fn perspective_assemble(
        raster: &crate::camera::SurfaceRaster,
    ) -> impl Fn(usize, usize) -> SurfacePixel + Sync + '_ {
        move |px: usize, py: usize| {
            let idx = py * raster.nx + px;
            if raster.lat[idx].is_finite() {
                SurfacePixel {
                    on_earth: true,
                    base_srgb: [FLAT_ALBEDO_SRGB, FLAT_ALBEDO_SRGB, FLAT_ALBEDO_SRGB],
                    normal_enu: [0.0, 0.0, 1.0],
                    sun_enu: [0.0, 0.0, 1.0],
                    sun_elev_deg: 90.0,
                    is_water: false,
                    ..Default::default()
                }
            } else {
                SurfacePixel::default() // sky/space: off-earth
            }
        }
    }

    #[test]
    fn perspective_full_composite_has_ground_limb_and_space() {
        let (nx, ny, nz) = (16, 16, 24);
        let georef = test_georef(nx, ny, 3000.0);
        let params = AtmosphereParams::default();
        let luts = AtmosphereLuts::build(&params);
        let sky_sh = SkyShTable::build(&luts, &params, 16);
        let sun_ecef = sun_enu_to_ecef([0.0, 0.0, 1.0], 45.0, -100.0);
        let cam = oblique_camera();
        let basis = cam.basis().expect("valid camera");
        let raster = build_perspective_raster(&basis, &georef, nx, ny);
        let mut surf = overhead_surf(&luts, &params, &sky_sh, sun_ecef);
        surf.cam.camera = basis.eye; // the perspective contract: the eye, set once

        let frame =
            render_perspective_frame_rgba(&surf, None, &basis, perspective_assemble(&raster));
        assert_eq!(frame.len(), cam.width * cam.height * 4);
        // Top-centre (row 0): SPACE (the ray leaves the 150 km eye upward, above the
        // shell).
        let top = (cam.width / 2) * 4;
        assert_eq!(frame[top + 3], 0, "top-centre should be space (alpha 0)");
        // Bottom-centre: lit ground (opaque + bright under the overhead sun).
        let bot = ((cam.height - 1) * cam.width + cam.width / 2) * 4;
        assert_eq!(frame[bot + 3], 255, "bottom-centre should be on earth");
        assert!(
            frame[bot] > 0 && frame[bot + 1] > 0,
            "daytime perspective ground should be lit: {:?}",
            &frame[bot..bot + 4]
        );
        // Some opaque pixel with NO ground hit exists (the atmosphere-grazing limb).
        let mut limb = 0usize;
        for py in 0..cam.height {
            for px in 0..cam.width {
                let idx = py * cam.width + px;
                if !raster.lat[idx].is_finite() && frame[idx * 4 + 3] == 255 {
                    limb += 1;
                }
            }
        }
        assert!(
            limb > 0,
            "an oblique wide-fov frame must contain limb pixels"
        );
        // The clear-cloud composite is byte-identical to surface-only (regression
        // anchor: adding the cloud machinery must not perturb a clear scene).
        let vol = build_volume(nx, ny, nz, 250.0, 3000.0, |_, _, _| (0.0, 0.0, 0.0));
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
        let with_clear = render_perspective_frame_rgba(
            &surf,
            Some(&scene),
            &basis,
            perspective_assemble(&raster),
        );
        assert_eq!(
            with_clear, frame,
            "a clear cloud volume must not change the perspective frame"
        );
    }

    #[test]
    fn perspective_cloud_changes_covered_pixels_with_parallax_geometry() {
        // A thick mid-level slab over the whole domain: the oblique perspective frame
        // must differ from the clear frame where rays cross the slab (the rays are
        // true 3-D lines — the slab is hit on the slant, which IS the parallax).
        let (nx, ny, nz) = (16, 16, 24);
        let georef = test_georef(nx, ny, 3000.0);
        let params = AtmosphereParams::default();
        let luts = AtmosphereLuts::build(&params);
        let sky_sh = SkyShTable::build(&luts, &params, 16);
        let sun_ecef = sun_enu_to_ecef([0.0, 0.0, 1.0], 45.0, -100.0);
        let cam = oblique_camera();
        let basis = cam.basis().unwrap();
        let raster = build_perspective_raster(&basis, &georef, nx, ny);
        let mut surf = overhead_surf(&luts, &params, &sky_sh, sun_ecef);
        surf.cam.camera = basis.eye;
        let clear =
            render_perspective_frame_rgba(&surf, None, &basis, perspective_assemble(&raster));
        let vol = build_volume(nx, ny, nz, 250.0, 3000.0, |_, _, k| {
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
        let cloudy = render_perspective_frame_rgba(
            &surf,
            Some(&scene),
            &basis,
            perspective_assemble(&raster),
        );
        let differ = cloudy
            .iter()
            .zip(clear.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(differ > 0, "the slab did not change any perspective pixel");
    }

    #[test]
    fn perspective_cloud_layer_only_is_transparent_clear_and_opaque_cloudy() {
        let (nx, ny, nz) = (16, 16, 24);
        let georef = test_georef(nx, ny, 3000.0);
        let params = AtmosphereParams::default();
        let luts = AtmosphereLuts::build(&params);
        let sky_sh = SkyShTable::build(&luts, &params, 16);
        let sun_ecef = sun_enu_to_ecef([0.0, 0.0, 1.0], 45.0, -100.0);
        let cam = oblique_camera();
        let basis = cam.basis().unwrap();
        // Clear: the layer is exactly transparent everywhere.
        let vol = build_volume(nx, ny, nz, 250.0, 3000.0, |_, _, _| (0.0, 0.0, 0.0));
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
        let layer = render_perspective_cloud_layer(
            &scene,
            &basis,
            OutputTransform::AbiReflectance,
            crate::render::DEFAULT_EXPOSURE,
        );
        assert_eq!(layer.len(), cam.width * cam.height * 4);
        assert!(
            layer.iter().all(|&v| v == 0),
            "a clear perspective layer must be exactly transparent"
        );
        // Cloudy: the slab is crossed on the slant toward the look point.
        let vol2 = build_volume(nx, ny, nz, 250.0, 3000.0, |_, _, k| {
            if (nz / 2..nz / 2 + 6).contains(&k) {
                (3.0e-3, 0.0, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        });
        let mip2 = OccupancyMip::build(&vol2, 4);
        let sun_od2 = accumulate_sun_od(&vol2, &georef, sun_ecef, 32);
        let mut cloudy_cfg = MarchConfig::new(StepQuality::Offline, vol2.voxel_pitch_m());
        cloudy_cfg.cloud_optical_depth_scale = 1.0;
        let scene2 = CloudScene {
            vol: &vol2,
            mip: &mip2,
            sun_od: &sun_od2,
            georef: &georef,
            luts: &luts,
            sky_sh: &sky_sh,
            sun_ecef,
            cfg: cloudy_cfg,
        };
        let layer2 = render_perspective_cloud_layer(
            &scene2,
            &basis,
            OutputTransform::AbiReflectance,
            crate::render::DEFAULT_EXPOSURE,
        );
        let max_alpha = layer2.chunks_exact(4).map(|p| p[3]).max().unwrap();
        assert!(
            max_alpha > 200,
            "the slant ray through a thick slab should be near-opaque: {max_alpha}"
        );
        // Space rays (top-centre) never carry cloud (the shell is below the eye's
        // upward rays): exactly transparent.
        let top = (cam.width / 2) * 4;
        assert_eq!(
            &layer2[top..top + 4],
            &[0, 0, 0, 0],
            "a space ray must stay transparent"
        );
    }
}
