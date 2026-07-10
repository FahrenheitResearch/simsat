//! Horizon maps: penumbral terrain cast shadows + the terrain ambient-aperture
//! cone (design doc section 6, M3).
//!
//! Precomputed once per domain from the brick `HGT` plane (CPU/rayon). For each
//! terrain texel we store, per azimuth bin, the maximum terrain HORIZON ELEVATION
//! angle (the angle up to the highest terrain the ray toward that azimuth crosses),
//! plus the derived ambient-aperture (a scalar visible-sky OPENNESS and a bent
//! normal — the visible-sky centroid direction). Two consumers:
//!
//!   1. **Penumbral cast shadow** ([`terrain_shadow_fraction`]): at shade time the
//!      terrain darkens the DIRECT sun term by the fraction of the finite solar disk
//!      (half-angle [`atmosphere::SUN_ANGULAR_RADIUS_RAD`] = 0.267 deg) above the
//!      local terrain horizon at the sun's azimuth. Smooth across the disk (the
//!      physically-correct penumbra), composes with the astronomical terminator with
//!      no double count (flat terrain -> the M2 disk exactly).
//!   2. **Ambient aperture** ([`HorizonMap::aperture_at`]): the horizon-occluded
//!      replacement for M5's full-hemisphere terrain sky ambient (Oat & Sander 2007
//!      ambient-aperture family) — valleys/pockets get less (and correctly-coloured)
//!      sky ambient than ridgetops.
//!
//! EARTH CURVATURE. Design section 1/6 is explicit that a naive flat tangent plane
//! is NOT acceptable at these domain sizes (the chord-vs-surface drop is ~12.5 km
//! over a 400 km half-domain, comparable to terrain relief). Along each azimuth ray
//! the target terrain's apparent height is lowered by the curvature drop
//! `d^2/(2R)` (`R = optics::EARTH_RADIUS_M`, the M0/M1 spherical earth), so distant
//! ridges correctly block LESS. See [`curvature_drop_m`].
//!
//! Grid geometry: grid-x = east, grid-y = north (the same documented M1 approximation
//! [`crate::render::normals_from_hgt`] uses; Lambert grid convergence away from
//! `STAND_LON` is ignored). Anisotropic `dx`/`dy` (metres) are handled.

use std::f64::consts::{PI, TAU};

use crate::atmosphere::solar_disk_visible_fraction;
use crate::optics::EARTH_RADIUS_M;

use rayon::prelude::*;

/// Number of azimuth bins in the horizon map (design section 6: "16 azimuths").
pub const HORIZON_AZIMUTH_BINS: usize = 16;

/// Near-field ray sampling: the first `HORIZON_NEAR_CELLS` samples step one grid
/// cell at a time (the local slope + nearby ridges dominate cast shadows), then the
/// ray switches to a geometric schedule. Documented, not tuned.
pub const HORIZON_NEAR_CELLS: usize = 24;

/// Geometric growth factor of the far-field ray schedule (past the near cells).
pub const HORIZON_FAR_GROWTH: f64 = 1.5;

/// Hard cap on the horizon search distance (m). Beyond ~60 km terrain rarely blocks
/// a low sun meaningfully and the curvature drop is large; the cap also bounds the
/// build cost. A low-sun (6 deg) shadow of a 3 km relief reaches ~28 km, well inside.
pub const HORIZON_MAX_DISTANCE_M: f64 = 60_000.0;

/// Earth-curvature drop (m) of a target at horizontal distance `distance_m` on the
/// spherical earth: `d^2 / (2 R)`. Lowers the apparent elevation of distant terrain.
#[inline]
pub fn curvature_drop_m(distance_m: f64) -> f64 {
    distance_m * distance_m / (2.0 * EARTH_RADIUS_M)
}

/// The penumbral terrain cast-shadow factor: the fraction of the finite solar disk
/// above the local terrain horizon, as a smooth function of `(sun_elev - horizon)`.
///
/// This is `solar_disk_visible_fraction(sun_elev_rad - horizon_angle_rad)` — the SAME
/// circular-segment disk math the M2 terminator uses, evaluated against the terrain
/// horizon instead of the astronomical (elevation 0) horizon. Consequences:
///   - `horizon = 0` reproduces the astronomical disk exactly, so folding this into
///     the M2 `disk` term for flat terrain is a no-op (no double count).
///   - fully lit when the whole disk clears the ridge (`sun_elev - horizon >> 0.267 deg`),
///     fully shadowed when the whole disk is below it, and 0.5 when the disk centre
///     grazes the ridge — the physically-correct penumbra across the 0.533 deg disk.
#[inline]
pub fn terrain_shadow_fraction(sun_elev_rad: f64, horizon_angle_rad: f64) -> f64 {
    solar_disk_visible_fraction(sun_elev_rad - horizon_angle_rad)
}

/// The ambient-aperture (openness, bent normal) of a texel from its 16 (or `bins`)
/// terrain horizon elevation angles (rad, each floored at 0). Oat & Sander (2007)
/// ambient-aperture family adapted to a horizon map.
///
/// For an isotropic sky of radiance `L`, the diffuse irradiance on a flat receiver
/// occluded by a horizon `h(az)` is `L * pi * <cos^2 h>` (azimuth mean) — so
/// **openness** = `<cos^2 h>` in `[0,1]` (1 = a fully open ridgetop, ->0 = an
/// enclosed pocket) is EXACTLY the isotropic-sky occlusion factor. The **bent
/// normal** is the cosine-weighted centroid direction of the visible sky (ENU:
/// x=east, y=north, z=up), which tilts toward the open side (downhill) and picks up
/// the correctly-coloured directional sky when evaluated through the SH ambient. For
/// flat terrain (all `h = 0`) this returns `(1, up)`, so the ambient reproduces M5's
/// full-hemisphere value exactly.
pub fn ambient_aperture_from_horizons(horizons: &[f64]) -> (f64, [f64; 3]) {
    let bins = horizons.len().max(1);
    let dphi = TAU / bins as f64;
    let mut mass = 0.0;
    let mut vec = [0.0f64; 3];
    for (k, &h) in horizons.iter().enumerate() {
        let h = h.max(0.0);
        let phi = k as f64 * dphi;
        let (sphi, cphi) = phi.sin_cos();
        let (sh, ch) = h.sin_cos();
        // Cosine-weighted visible "mass" of this azimuth sector (width dphi):
        // dphi * integral_h^{pi/2} sin(e) cos(e) de = dphi * 0.5 * cos^2(h).
        mass += dphi * 0.5 * ch * ch;
        // Cosine-weighted visible-direction centroid contribution:
        //   horizontal: cos(phi/sin(phi)) * integral cos^2(e) sin(e) de = cos^3(h)/3
        //   vertical  : integral sin^2(e) cos(e) de = (1 - sin^3(h))/3
        let horiz = ch * ch * ch / 3.0;
        let vert = (1.0 - sh * sh * sh) / 3.0;
        vec[0] += dphi * horiz * cphi;
        vec[1] += dphi * horiz * sphi;
        vec[2] += dphi * vert;
    }
    // Full hemisphere cosine-weighted mass is pi (all h = 0), so openness in [0,1].
    let openness = (mass / PI).clamp(0.0, 1.0);
    let len = (vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]).sqrt();
    let bent = if len > 1.0e-9 {
        [vec[0] / len, vec[1] / len, vec[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    };
    (openness, bent)
}

/// The per-domain horizon map: terrain horizon elevation angles (16 azimuth bins per
/// texel) plus the precomputed ambient-aperture (openness + bent normal). Built once
/// from the `HGT` plane; sun-independent (the same map serves any sun position).
#[derive(Debug, Clone)]
pub struct HorizonMap {
    pub nx: usize,
    pub ny: usize,
    pub bins: usize,
    /// Horizon elevation angle (rad, >= 0), index `(j*nx + i)*bins + k`.
    pub horizon: Vec<f32>,
    /// Ambient-aperture openness (`<cos^2 h>`, `[0,1]`), index `j*nx + i`.
    pub openness: Vec<f32>,
    /// Ambient-aperture bent normal (ENU unit), index `(j*nx + i)*3`.
    pub bent_normal: Vec<f32>,
    /// Whether earth curvature was applied (always true; recorded for diagnostics).
    pub with_curvature: bool,
}

impl HorizonMap {
    /// Build the horizon map from an `HGT` plane (row-major `[ny][nx]`, metres MSL)
    /// with grid spacing `dx_m`/`dy_m` (metres, east/north). Rayon-parallel over rows.
    pub fn build(hgt: &[f32], nx: usize, ny: usize, dx_m: f64, dy_m: f64) -> Self {
        let bins = HORIZON_AZIMUTH_BINS;
        let dphi = TAU / bins as f64;
        let flat = HorizonMap {
            nx,
            ny,
            bins,
            horizon: vec![0.0; nx * ny * bins],
            openness: vec![1.0; nx * ny],
            bent_normal: {
                let mut v = vec![0.0f32; nx * ny * 3];
                for c in v.chunks_exact_mut(3) {
                    c[2] = 1.0;
                }
                v
            },
            with_curvature: true,
        };
        if hgt.len() != nx * ny
            || nx < 2
            || ny < 2
            || !dx_m.is_finite()
            || !dy_m.is_finite()
            || dx_m <= 0.0
            || dy_m <= 0.0
        {
            return flat;
        }
        let pitch_near = dx_m.min(dy_m).max(1.0);
        let schedule = distance_schedule(pitch_near, nx, ny, dx_m, dy_m);

        // Per-row buffers, assembled in parallel then flattened.
        let rows: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = (0..ny)
            .into_par_iter()
            .map(|j| {
                let mut horiz_row = vec![0.0f32; nx * bins];
                let mut open_row = vec![0.0f32; nx];
                let mut bent_row = vec![0.0f32; nx * 3];
                let mut hor = vec![0.0f64; bins];
                for i in 0..nx {
                    let h0 = hgt[j * nx + i] as f64;
                    let h0 = if h0.is_finite() { h0 } else { 0.0 };
                    for (k, hk) in hor.iter_mut().enumerate() {
                        let phi = k as f64 * dphi;
                        let (sphi, cphi) = phi.sin_cos();
                        // Cell-space step per metre of horizontal distance along phi.
                        let di = cphi / dx_m;
                        let dj = sphi / dy_m;
                        let mut maxel = 0.0f64; // floored at the astronomical horizon
                        for &d in &schedule {
                            let fi = i as f64 + d * di;
                            let fj = j as f64 + d * dj;
                            if fi < 0.0 || fi > (nx - 1) as f64 || fj < 0.0 || fj > (ny - 1) as f64
                            {
                                break; // left the domain; the straight ray stays out
                            }
                            let h = bilinear_hgt(hgt, nx, ny, fi, fj);
                            let drop = curvature_drop_m(d);
                            let el = ((h - h0 - drop) / d).atan();
                            if el > maxel {
                                maxel = el;
                            }
                        }
                        *hk = maxel;
                        horiz_row[i * bins + k] = maxel as f32;
                    }
                    let (open, bent) = ambient_aperture_from_horizons(&hor);
                    open_row[i] = open as f32;
                    bent_row[i * 3] = bent[0] as f32;
                    bent_row[i * 3 + 1] = bent[1] as f32;
                    bent_row[i * 3 + 2] = bent[2] as f32;
                }
                (horiz_row, open_row, bent_row)
            })
            .collect();

        let mut horizon = Vec::with_capacity(nx * ny * bins);
        let mut openness = Vec::with_capacity(nx * ny);
        let mut bent_normal = Vec::with_capacity(nx * ny * 3);
        for (h, o, b) in rows {
            horizon.extend_from_slice(&h);
            openness.extend_from_slice(&o);
            bent_normal.extend_from_slice(&b);
        }
        HorizonMap {
            nx,
            ny,
            bins,
            horizon,
            openness,
            bent_normal,
            with_curvature: true,
        }
    }

    /// The terrain horizon elevation angle (rad, >= 0) at fractional cell `(fi, fj)`
    /// and azimuth `azimuth_rad` (from east toward north): bilinear over the 4 corner
    /// texels of the azimuth-interpolated horizon.
    pub fn horizon_angle_at(&self, fi: f64, fj: f64, azimuth_rad: f64) -> f64 {
        if self.nx < 2 || self.ny < 2 {
            return 0.0;
        }
        let (i0, j0, i1, j1, tx, ty) = self.bilinear_cells(fi, fj);
        let a = self.horizon_bin_interp(i0, j0, azimuth_rad);
        let b = self.horizon_bin_interp(i1, j0, azimuth_rad);
        let c = self.horizon_bin_interp(i0, j1, azimuth_rad);
        let d = self.horizon_bin_interp(i1, j1, azimuth_rad);
        let top = a * (1.0 - tx) + b * tx;
        let bot = c * (1.0 - tx) + d * tx;
        top * (1.0 - ty) + bot * ty
    }

    /// The ambient aperture `(openness, bent_normal)` at fractional cell `(fi, fj)`:
    /// bilinear openness, bilinear + renormalised bent normal (ENU).
    pub fn aperture_at(&self, fi: f64, fj: f64) -> (f64, [f64; 3]) {
        if self.nx < 2 || self.ny < 2 {
            return (1.0, [0.0, 0.0, 1.0]);
        }
        let (i0, j0, i1, j1, tx, ty) = self.bilinear_cells(fi, fj);
        let w00 = (1.0 - tx) * (1.0 - ty);
        let w10 = tx * (1.0 - ty);
        let w01 = (1.0 - tx) * ty;
        let w11 = tx * ty;
        let o = |i: usize, j: usize| self.openness[j * self.nx + i] as f64;
        let openness =
            (o(i0, j0) * w00 + o(i1, j0) * w10 + o(i0, j1) * w01 + o(i1, j1) * w11).clamp(0.0, 1.0);
        let b = |i: usize, j: usize, c: usize| self.bent_normal[(j * self.nx + i) * 3 + c] as f64;
        let mut bent = [0.0f64; 3];
        for (c, bc) in bent.iter_mut().enumerate() {
            *bc = b(i0, j0, c) * w00 + b(i1, j0, c) * w10 + b(i0, j1, c) * w01 + b(i1, j1, c) * w11;
        }
        let len = (bent[0] * bent[0] + bent[1] * bent[1] + bent[2] * bent[2]).sqrt();
        if len > 1.0e-9 {
            for bc in &mut bent {
                *bc /= len;
            }
        } else {
            bent = [0.0, 0.0, 1.0];
        }
        (openness, bent)
    }

    /// The azimuth-interpolated horizon angle at INTEGER texel `(i, j)`.
    fn horizon_bin_interp(&self, i: usize, j: usize, azimuth_rad: f64) -> f64 {
        let t = azimuth_rad.rem_euclid(TAU) / (TAU / self.bins as f64);
        let k0 = (t.floor() as usize) % self.bins;
        let k1 = (k0 + 1) % self.bins;
        let f = t - t.floor();
        let base = (j * self.nx + i) * self.bins;
        let a = self.horizon[base + k0] as f64;
        let b = self.horizon[base + k1] as f64;
        a * (1.0 - f) + b * f
    }

    /// Bilinear corner indices + fractions for fractional cell `(fi, fj)` (clamped).
    fn bilinear_cells(&self, fi: f64, fj: f64) -> (usize, usize, usize, usize, f64, f64) {
        let fi = fi.clamp(0.0, (self.nx - 1) as f64);
        let fj = fj.clamp(0.0, (self.ny - 1) as f64);
        let i0 = fi.floor() as usize;
        let j0 = fj.floor() as usize;
        let i1 = (i0 + 1).min(self.nx - 1);
        let j1 = (j0 + 1).min(self.ny - 1);
        (i0, j0, i1, j1, fi - i0 as f64, fj - j0 as f64)
    }
}

/// The ray distance schedule (metres) for the horizon sweep: near-field cell-by-cell
/// (captures local slope + nearby ridges), then geometric growth to the distance cap.
fn distance_schedule(pitch_near: f64, nx: usize, ny: usize, dx_m: f64, dy_m: f64) -> Vec<f64> {
    // The domain's own diagonal caps the far reach (no point sampling past it).
    let domain_diag = (((nx - 1) as f64 * dx_m).powi(2) + ((ny - 1) as f64 * dy_m).powi(2)).sqrt();
    let max_dist = HORIZON_MAX_DISTANCE_M.min(domain_diag).max(pitch_near);
    let mut out = Vec::new();
    for s in 1..=HORIZON_NEAR_CELLS {
        let d = s as f64 * pitch_near;
        if d > max_dist {
            break;
        }
        out.push(d);
    }
    let mut d = out.last().copied().unwrap_or(pitch_near) * HORIZON_FAR_GROWTH;
    while d <= max_dist {
        out.push(d);
        d *= HORIZON_FAR_GROWTH;
    }
    if out.is_empty() {
        out.push(pitch_near.min(max_dist));
    }
    out
}

/// Bilinear `HGT` (m) at fractional cell `(fi, fj)`, clamp-to-edge.
fn bilinear_hgt(hgt: &[f32], nx: usize, ny: usize, fi: f64, fj: f64) -> f64 {
    let fi = fi.clamp(0.0, (nx - 1) as f64);
    let fj = fj.clamp(0.0, (ny - 1) as f64);
    let i0 = fi.floor() as usize;
    let j0 = fj.floor() as usize;
    let i1 = (i0 + 1).min(nx - 1);
    let j1 = (j0 + 1).min(ny - 1);
    let tx = fi - i0 as f64;
    let ty = fj - j0 as f64;
    let h = |i: usize, j: usize| {
        let v = hgt[j * nx + i] as f64;
        if v.is_finite() { v } else { 0.0 }
    };
    let top = h(i0, j0) * (1.0 - tx) + h(i1, j0) * tx;
    let bot = h(i0, j1) * (1.0 - tx) + h(i1, j1) * tx;
    top * (1.0 - ty) + bot * ty
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atmosphere::{SUN_ANGULAR_DIAMETER_DEG, SUN_ANGULAR_RADIUS_RAD};

    #[test]
    fn curvature_drop_matches_design_note() {
        // ~12.5 km drop at a 400 km half-domain distance (design section 1).
        let drop = curvature_drop_m(400_000.0);
        assert!((drop - 12_558.0).abs() < 50.0, "drop {drop}");
        assert_eq!(curvature_drop_m(0.0), 0.0);
        // Monotone increasing with distance.
        assert!(curvature_drop_m(1000.0) < curvature_drop_m(2000.0));
    }

    #[test]
    fn terrain_shadow_fully_lit_shadowed_and_half_disk() {
        let deg = std::f64::consts::PI / 180.0;
        // Sun well above the horizon -> fully lit.
        assert!((terrain_shadow_fraction(30.0 * deg, 5.0 * deg) - 1.0).abs() < 1e-9);
        // Sun well below the terrain horizon -> fully shadowed.
        assert!(terrain_shadow_fraction(5.0 * deg, 30.0 * deg).abs() < 1e-9);
        // Sun centre exactly on the terrain horizon -> half the disk visible.
        assert!((terrain_shadow_fraction(10.0 * deg, 10.0 * deg) - 0.5).abs() < 1e-6);
        // Penumbra: monotone increasing as the sun rises through the disk, and the
        // transition spans the finite disk (~0.533 deg full width).
        let hor = 10.0 * deg;
        // The transition spans exactly the finite solar disk: just below the lower disk
        // edge is fully shadowed, just above the upper edge fully lit. The disk radius
        // used is half the 0.533 deg diameter (read from the shared constants so the
        // test tracks them, not hard-coded angles).
        let half_diam_rad = (SUN_ANGULAR_DIAMETER_DEG / 2.0).to_radians();
        let below = terrain_shadow_fraction(hor - half_diam_rad - 1e-4, hor);
        let above = terrain_shadow_fraction(hor + half_diam_rad + 1e-4, hor);
        assert!(below.abs() < 1e-9 && (above - 1.0).abs() < 1e-9);
        // The two constants agree: the smooth-fraction radius is the solar radius.
        assert!((half_diam_rad - SUN_ANGULAR_RADIUS_RAD).abs() < 5e-6);
        // Flat terrain (horizon 0) reproduces the astronomical disk exactly.
        for &e in &[-0.2 * deg, 0.0, 0.1 * deg, 5.0 * deg] {
            assert!(
                (terrain_shadow_fraction(e, 0.0) - solar_disk_visible_fraction(e)).abs() < 1e-12
            );
        }
    }

    #[test]
    fn aperture_open_flat_ground_is_full_sky_up() {
        // All horizons 0 -> openness 1, bent normal = up (regression to M5's full
        // hemisphere at the up normal).
        let horizons = vec![0.0f64; 16];
        let (openness, bent) = ambient_aperture_from_horizons(&horizons);
        assert!((openness - 1.0).abs() < 1e-9, "openness {openness}");
        assert!(bent[0].abs() < 1e-9 && bent[1].abs() < 1e-9 && (bent[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn aperture_enclosed_pocket_reduces_openness() {
        // A symmetric pocket walled to 45 deg all around: openness = cos^2(45) = 0.5,
        // bent normal still up (symmetric).
        let h = std::f64::consts::FRAC_PI_4;
        let horizons = vec![h; 16];
        let (openness, bent) = ambient_aperture_from_horizons(&horizons);
        assert!((openness - 0.5).abs() < 1e-6, "openness {openness}");
        assert!(bent[0].abs() < 1e-6 && bent[1].abs() < 1e-6 && bent[2] > 0.9);
        // Deeper walls -> less open, monotone.
        let deep = vec![std::f64::consts::FRAC_PI_3; 16]; // 60 deg
        let (open_deep, _) = ambient_aperture_from_horizons(&deep);
        assert!(
            open_deep < openness,
            "deeper pocket {open_deep} !< {openness}"
        );
    }

    #[test]
    fn aperture_bent_normal_tilts_toward_the_open_side() {
        // A wall to the EAST (bin 0 = azimuth 0 = +east) only: the visible-sky centroid
        // tilts WEST (away from the wall, toward open sky).
        let mut horizons = vec![0.0f64; 16];
        horizons[0] = 1.2; // ~69 deg wall due east
        let (openness, bent) = ambient_aperture_from_horizons(&horizons);
        assert!(openness < 1.0);
        assert!(bent[0] < 0.0, "bent should lean west (−east), got {bent:?}");
        assert!(bent[2] > 0.0, "bent still points up-ish");
    }

    #[test]
    fn horizon_map_on_a_synthetic_ridge_has_known_angles() {
        // A ridge: a wall rising to the east (H increases with i). A texel on the flat
        // part west of the wall sees a positive horizon toward the east and ~0 west.
        let (nx, ny) = (40usize, 8usize);
        let (dx, dy) = (100.0f64, 100.0f64);
        let ridge_i = 30usize;
        let ridge_h = 500.0f64; // 500 m wall
        let mut hgt = vec![0.0f32; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                hgt[j * nx + i] = if i >= ridge_i { ridge_h as f32 } else { 0.0 };
            }
        }
        let map = HorizonMap::build(&hgt, nx, ny, dx, dy);
        // Texel at i=20 (10 cells = 1000 m west of the wall foot). Toward the east
        // (azimuth 0) the horizon is ~atan(500/1000) minus a tiny curvature drop.
        let (i, j) = (20usize, 4usize);
        let east = map.horizon_angle_at(i as f64, j as f64, 0.0);
        let expected = ((ridge_h - curvature_drop_m(1000.0)) / 1000.0).atan();
        assert!(
            (east - expected).abs() < 0.05,
            "east horizon {east} vs expected {expected}"
        );
        // Toward the west (azimuth pi) it is flat -> ~0.
        let west = map.horizon_angle_at(i as f64, j as f64, PI);
        assert!(west.abs() < 1e-3, "west horizon {west} should be ~0");
        // A texel ON the ridge top sees ~0 east horizon (nothing higher).
        let on_ridge = map.horizon_angle_at(35.0, 4.0, 0.0);
        assert!(on_ridge.abs() < 1e-3, "ridge-top east horizon {on_ridge}");
    }

    #[test]
    fn horizon_map_flat_domain_is_open_everywhere() {
        let (nx, ny) = (16usize, 16usize);
        let hgt = vec![250.0f32; nx * ny];
        let map = HorizonMap::build(&hgt, nx, ny, 1000.0, 1000.0);
        for v in &map.horizon {
            assert!(v.abs() < 1e-4, "flat horizon should be 0, got {v}");
        }
        let (openness, bent) = map.aperture_at(8.0, 8.0);
        assert!((openness - 1.0).abs() < 1e-4);
        assert!((bent[2] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn curvature_lowers_a_distant_ridge_horizon() {
        // Same ridge relief at a far distance is blocked LESS with curvature: the
        // curvature-included horizon is below the naive flat-plane angle.
        let dist = 30_000.0f64;
        let relief = 800.0f64;
        let flat_angle = (relief / dist).atan();
        let curved_angle = ((relief - curvature_drop_m(dist)) / dist).atan();
        assert!(
            curved_angle < flat_angle,
            "curvature must lower the horizon"
        );
        // The drop at 30 km (~70 m) is a meaningful fraction of an 800 m relief.
        assert!(curvature_drop_m(dist) > 60.0);
    }
}
