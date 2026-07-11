// SimSat surface pass (design doc section 3 / 5 / 6, M2). WGSL twin of the CPU
// reference in `atmosphere.rs` (LUTs + raymarch) and `render.rs` (`shade_surface`)
// — keep them in lockstep. Validated headlessly by the `naga` shader test (nodes
// have no GPU); the physics is CPU-tested and this is the discipline-maintained
// twin the owner's exe runs.
//
// M2 shading, per output pixel:
//   - on-earth: Blue Marble albedo, finite-disk direct sun (transmittance-to-sun +
//     smooth terminator), sky-view ambient (twilight fill), and aerial perspective
//     (camera-to-ground transmittance + single/multi inscatter) applied via a view-
//     ray raymarch of the atmosphere shell.
//   - off-earth: rays that graze the atmosphere shell render the LIMB (bright
//     scattering ring); rays that miss it are space (transparent black).
//   - output transform: ABI-like reflectance factor rho = pi*L/E_band with the sqrt
//     satellite stretch (default), or reflectance through sRGB gamma (debug).
//
// Precomputed CPU lookups (uploaded):
//   lut_geo   = (bm_u, bm_v, grid_u, grid_v); bm_u < 0 -> off-earth (per CGMS);
//               grid_u < 0 -> outside the WRF domain (flat up-normal, land).
//   lut_light = (sun_e, sun_n, sun_u, sun_elev_deg)  local-ENU sun + elevation.
//   transmittance_lut (256x64), multiscatter_lut (32x32): optics-config LUTs.
//   ambient_lut (N x 1): scalar sky irradiance vs sun elevation (sky-view projection).

const PI: f32 = 3.14159265358979;
const DEG2RAD: f32 = 0.017453292519943295;

// Atmosphere constants — MUST match atmosphere.rs.
const R_GROUND: f32 = 6370000.0;
const R_TOP: f32 = 6470000.0;
const RAYLEIGH_SCA: vec3<f32> = vec3<f32>(5.802e-6, 13.558e-6, 33.1e-6);
const RAYLEIGH_H: f32 = 8000.0;
const MIE_H: f32 = 1200.0;
// Ozone base (0.65,1.881,0.085)e-6 * OZONE_STRENGTH 1.45 (M2 twilight pass; twin of
// atmosphere.rs OZONE_ABSORPTION / OZONE_STRENGTH).
const OZONE_ABS: vec3<f32> = vec3<f32>(0.65e-6 * 1.45, 1.881e-6 * 1.45, 0.085e-6 * 1.45);
const OZONE_CENTER: f32 = 25000.0;
const OZONE_HW: f32 = 15000.0;
// Multiple-scattering gain in the render march (twin of atmosphere.rs MULTISCATTER_GAIN).
const MULTISCATTER_GAIN: f32 = 1.4;
// Display transform (twin of atmosphere.rs): chroma-gated highlight desaturation +
// toe-lifted sqrt. Desat gated by BOTH luminance and saturation.
const REFL_TOE_KNEE: f32 = 0.05;
const REFL_TOE_GAMMA: f32 = 0.38;
const DESAT_LUM_LO: f32 = 0.09;
const DESAT_LUM_HI: f32 = 0.13;
const DESAT_SAT_LO: f32 = 0.40;
const DESAT_SAT_HI: f32 = 0.88;
const DESAT_MAX: f32 = 0.55;
const WV_ABS: vec3<f32> = vec3<f32>(6.0e-6, 1.5e-6, 0.2e-6);
const WV_H: f32 = 2000.0;
const SUN_ANG_R: f32 = 4.65e-3;
const LIMB_DISK_AVG: f32 = 0.832; // 1 - a1/3 - a2/6 (M2 review FINDING 3; was 0.79)
const STEPS: u32 = 32u;
const WATER_N: f32 = 1.34; // sea-water refractive index (M3 Cox-Munk glint / Fresnel)
// True-color calibration (refinement pass; twins of render.rs constants). Daytime
// aerial-perspective veil reduction (Rayleigh correction), land-albedo vibrancy, a
// LAND-only daytime brightness lift, and the sun-glint brightness gain + core
// narrowing. Round 2 values. Each is a no-op at its identity value; all are sun-gated
// or water-only so the M2 twilight look is byte-unchanged.
const AERIAL_VEIL_DAY_SCALE: f32 = 0.40;
const AERIAL_VEIL_ELEV_LO: f32 = 20.0;
const AERIAL_VEIL_ELEV_HI: f32 = 30.0; // WS2: 40 -> 30 (full daytime treatment by 30 deg; twin of render.rs)
const LAND_VIBRANCY: f32 = 1.45;
const LAND_DAY_GAIN: f32 = 1.20;   // LAND-only daytime surface-reflectance lift (not the global exposure)
const LAND_SZA_REFERENCE_ELEV: f32 = 60.0;
const GLINT_STRENGTH: f32 = 3.5;
const GLINT_MSS_SCALE: f32 = 0.4;  // < 1 tightens the Cox-Munk core -> smaller, brighter glint streak (round 3: 0.6->0.4)
// WS2 bright-cloud tonemap + water lighting (twins of render.rs). This pass has an
// implicit exposure of 1.0, so the exposure-aware shoulder bound is RHO_HIGHLIGHT_MAX
// itself (the CPU paths derive x_max = exposure * RHO_HIGHLIGHT_MAX at their seam).
const CLOUD_SOFTCLIP_KNEE: f32 = 0.65;   // identity below; bounded Mobius shoulder above
const RHO_HIGHLIGHT_MAX: f32 = 1.25;     // physical reflectance ceiling -> display 1.0
const WATER_ALBEDO_DAY_SCALE: f32 = 0.35; // daytime water-body albedo scale (twilight anchor = u.p1.y)
// Low-sun visible pass (twins of render.rs): the SUNRISE veil ramp (satpy-idiom — the
// Rayleigh de-haze is reduced toward the terminator, never hard-disabled at 20 deg)
// and the LUT-derived low-sun ILLUMINANT correction (GREEN RESTORATION: the green of
// OUR OWN transmittance-LUT direct-sun illuminant at a reference cloud altitude is
// restored to its Rayleigh log-line, removing the Chappuis ozone green dip that
// rendered dawn cloud khaki/mauve; the R-B warm axis is preserved and the triple is
// unit-luminance renormalized — see render.rs low_sun_illuminant_gains).
const VEIL_TERMINATOR_ELEV: f32 = 2.0;   // full physical veil at/below (dusk band byte-identical)
const VEIL_SUNRISE_ELEV_HI: f32 = 16.0;  // full daytime de-haze in place at/above
const ILLUM_REF_CLOUD_ALT: f32 = 7000.0; // reference cloud altitude (m) of the illuminant sample
const ILLUM_CORR_IN_LO: f32 = 2.0;       // identity at/below (dusk band byte-identical)
const ILLUM_CORR_IN_HI: f32 = 5.0;       // full correction at/above (green-restoration form; twin of render.rs)
const ILLUM_CORR_OUT_LO: f32 = 20.0;     // taper-off start
const ILLUM_CORR_OUT_HI: f32 = 30.0;     // identity at/above (daytime byte-identical)
// ABI SYNTHETIC-GREEN display mode (prototype, twin of render.rs set_synthetic_green):
// G' = 0.45*R + 0.45*B + 0.10*G (Bah et al. 2018). OFF (0.0) by default — flip together
// with the CPU process-global if the mode is adopted.
const SYNTHETIC_GREEN_MODE: f32 = 0.0;
const SYN_GREEN_W_RED: f32 = 0.45;
const SYN_GREEN_W_GREEN: f32 = 0.10;
const SYN_GREEN_W_BLUE: f32 = 0.45;

struct Uniforms {
    cam: vec4<f32>,   // xyz camera ECEF, w R_ground
    sun: vec4<f32>,   // xyz sun ECEF dir, w R_top
    ex: vec4<f32>,    // xyz look basis toward-centre, w scan x_min
    ey: vec4<f32>,    // xyz east, w scan y_max
    ez: vec4<f32>,    // xyz north, w pitch_x
    solar: vec4<f32>, // xyz band solar irradiance, w pitch_y
    p0: vec4<f32>,    // mie_sca_ground, mie_ext_ground, mie_g, pw_ratio
    p1: vec4<f32>,    // bm_present, water_scale, flat_albedo, output_transform
    p2: vec4<f32>,    // ambient_elev_min, ambient_elev_max, ambient_n, atmosphere_correction
    land0: vec4<f32>, // sza_enabled, sza_max_gain, dark_toe_enabled, dark_toe_knee
    land1: vec4<f32>, // dark_toe_gamma, dark_toe_max_gain, unused, unused
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var lut_geo: texture_2d<f32>;
@group(0) @binding(2) var lut_light: texture_2d<f32>;
@group(0) @binding(3) var bm_tex: texture_2d<f32>;
@group(0) @binding(4) var normal_tex: texture_2d<f32>;
@group(0) @binding(5) var landmask_tex: texture_2d<f32>;
@group(0) @binding(6) var samp: sampler;
@group(0) @binding(7) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(8) var multiscatter_lut: texture_2d<f32>;
@group(0) @binding(9) var ambient_lut: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(p[vi], 0.0, 1.0);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    let lo = x * 12.92;
    let hi = 1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, x <= vec3<f32>(0.0031308));
}

// ── manual bilinear LUT sampling (Rgba32Float is not guaranteed filterable on
// the iGPU floor, so we textureLoad + lerp, matching the CPU Lut2::sample_uv).
// The samplers reference their global texture directly — naga-safest (no texture
// function-parameter handles). ──
fn bilinear(c00: vec3<f32>, c10: vec3<f32>, c01: vec3<f32>, c11: vec3<f32>, tx: f32, ty: f32) -> vec3<f32> {
    return mix(mix(c00, c10, tx), mix(c01, c11, tx), ty);
}

// ── geometry (spherically symmetric in (r, mu)) ──
fn ray_hits_ground(r: f32, mu: f32) -> bool {
    return mu < 0.0 && (r * r * (mu * mu - 1.0) + R_GROUND * R_GROUND) >= 0.0;
}

fn distance_to_top(r: f32, mu: f32) -> f32 {
    let disc = r * r * (mu * mu - 1.0) + R_TOP * R_TOP;
    return max(0.0, -r * mu + sqrt(max(0.0, disc)));
}

// (t0, t1) roots of |o + t d| = rad, or (1, -1) if no real intersection.
fn ray_sphere(o: vec3<f32>, d: vec3<f32>, rad: f32) -> vec2<f32> {
    let b = dot(o, d);
    let c = dot(o, o) - rad * rad;
    let disc = b * b - c;
    if (disc < 0.0) {
        return vec2<f32>(1.0, -1.0);
    }
    let s = sqrt(disc);
    return vec2<f32>(-b - s, -b + s);
}

// ── transmittance / multiscatter LUT parameterisation ──
fn transmittance_uv(r: f32, mu: f32) -> vec2<f32> {
    let cap_r = clamp(r, R_GROUND, R_TOP);
    let bigh = sqrt(R_TOP * R_TOP - R_GROUND * R_GROUND);
    let rho = sqrt(max(0.0, cap_r * cap_r - R_GROUND * R_GROUND));
    let d = distance_to_top(cap_r, mu);
    let d_min = R_TOP - cap_r;
    let d_max = rho + bigh;
    var uu = 0.0;
    if (d_max - d_min > 0.0) {
        uu = (d - d_min) / (d_max - d_min);
    }
    var vv = 0.0;
    if (bigh > 0.0) {
        vv = rho / bigh;
    }
    return clamp(vec2<f32>(uu, vv), vec2<f32>(0.0), vec2<f32>(1.0));
}

fn sample_transmittance(r: f32, mu: f32) -> vec3<f32> {
    let uv = transmittance_uv(r, mu);
    let w = 256;
    let h = 64;
    let fx = max(0.0, uv.x * f32(w) - 0.5);
    let fy = max(0.0, uv.y * f32(h) - 0.5);
    let x0 = min(i32(floor(fx)), w - 1);
    let y0 = min(i32(floor(fy)), h - 1);
    let x1 = min(x0 + 1, w - 1);
    let y1 = min(y0 + 1, h - 1);
    let c00 = textureLoad(transmittance_lut, vec2<i32>(x0, y0), 0).rgb;
    let c10 = textureLoad(transmittance_lut, vec2<i32>(x1, y0), 0).rgb;
    let c01 = textureLoad(transmittance_lut, vec2<i32>(x0, y1), 0).rgb;
    let c11 = textureLoad(transmittance_lut, vec2<i32>(x1, y1), 0).rgb;
    return bilinear(c00, c10, c01, c11, fx - floor(fx), fy - floor(fy));
}

fn sample_multiscatter(r: f32, mu_s: f32) -> vec3<f32> {
    let uu = clamp(mu_s * 0.5 + 0.5, 0.0, 1.0);
    let vv = clamp((clamp(r, R_GROUND, R_TOP) - R_GROUND) / (R_TOP - R_GROUND), 0.0, 1.0);
    let w = 32;
    let h = 32;
    let fx = max(0.0, uu * f32(w) - 0.5);
    let fy = max(0.0, vv * f32(h) - 0.5);
    let x0 = min(i32(floor(fx)), w - 1);
    let y0 = min(i32(floor(fy)), h - 1);
    let x1 = min(x0 + 1, w - 1);
    let y1 = min(y0 + 1, h - 1);
    let c00 = textureLoad(multiscatter_lut, vec2<i32>(x0, y0), 0).rgb;
    let c10 = textureLoad(multiscatter_lut, vec2<i32>(x1, y0), 0).rgb;
    let c01 = textureLoad(multiscatter_lut, vec2<i32>(x0, y1), 0).rgb;
    let c11 = textureLoad(multiscatter_lut, vec2<i32>(x1, y1), 0).rgb;
    return bilinear(c00, c10, c01, c11, fx - floor(fx), fy - floor(fy));
}

// ── phase functions ──
fn rayleigh_phase(cos_t: f32) -> f32 {
    return 3.0 / (16.0 * PI) * (1.0 + cos_t * cos_t);
}

fn cornette_shanks(cos_t: f32, g: f32) -> f32 {
    let g2 = g * g;
    let num = 3.0 * (1.0 - g2) * (1.0 + cos_t * cos_t);
    let denom = 8.0 * PI * (2.0 + g2) * pow(1.0 + g2 - 2.0 * g * cos_t, 1.5);
    return num / denom;
}

// ── medium sample ──
struct Medium {
    rayleigh_sca: vec3<f32>,
    mie_sca: f32,
    scattering: vec3<f32>,
    extinction: vec3<f32>,
};

fn sample_medium(h_in: f32) -> Medium {
    let h = max(h_in, 0.0);
    let rd = exp(-h / RAYLEIGH_H);
    let md = exp(-h / MIE_H);
    let od = max(0.0, 1.0 - abs(h - OZONE_CENTER) / OZONE_HW);
    let wvd = exp(-h / WV_H);
    let mie_sca = u.p0.x * md;
    let mie_ext = u.p0.y * md;
    let mie_abs = mie_ext - mie_sca;
    let ray = RAYLEIGH_SCA * rd;
    let ozone = OZONE_ABS * od;
    let wv = WV_ABS * wvd * u.p0.w;
    let scattering = ray + vec3<f32>(mie_sca);
    let extinction = ray + vec3<f32>(mie_sca + mie_abs) + ozone + wv;
    return Medium(ray, mie_sca, scattering, extinction);
}

// ── scattering raymarch of a segment ──
struct Scatter {
    inscatter: vec3<f32>,
    transmittance: vec3<f32>,
};

fn raymarch(p_start: vec3<f32>, view: vec3<f32>, sun: vec3<f32>, seg_len: f32) -> Scatter {
    let cos_vs = dot(view, sun);
    let ray_ph = rayleigh_phase(cos_vs);
    let mie_ph = cornette_shanks(cos_vs, u.p0.z);
    let dt = seg_len / f32(STEPS);
    var od = vec3<f32>(0.0);
    var l = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < STEPS; i = i + 1u) {
        let t = (f32(i) + 0.5) * dt;
        let p = p_start + view * t;
        let r = length(p);
        if (r >= R_TOP + 1.0 || r < R_GROUND - 1.0) {
            continue;
        }
        let up = p / r;
        let med = sample_medium(r - R_GROUND);
        let mu_s = dot(up, sun);
        var t_sun = vec3<f32>(0.0);
        if (!ray_hits_ground(r, mu_s)) {
            t_sun = sample_transmittance(r, mu_s);
        }
        let ms = sample_multiscatter(r, mu_s);
        let throughput = exp(-od);
        let ext = max(med.extinction, vec3<f32>(1e-12));
        let sample_t = exp(-ext * dt);
        let single = t_sun * (med.rayleigh_sca * ray_ph + vec3<f32>(med.mie_sca) * mie_ph);
        let multi = MULTISCATTER_GAIN * med.scattering * ms;
        let s = u.solar.xyz * (single + multi);
        l = l + throughput * (s - s * sample_t) / ext;
        od = od + ext * dt;
    }
    return Scatter(l, exp(-od));
}

fn disk_fraction(elev: f32) -> f32 {
    let x = elev / SUN_ANG_R;
    if (x >= 1.0) {
        return 1.0;
    }
    if (x <= -1.0) {
        return 0.0;
    }
    return (acos(-x) + x * sqrt(max(0.0, 1.0 - x * x))) / PI;
}

// ── M3 penumbral terrain shadow + Cox-Munk water glint (twins of horizon.rs /
// optics.rs; naga-validated). On the ACTIVE clouds-off GPU pass the per-texel
// horizon map / wind / snow / aperture are NOT uploaded (deferred GPU activation,
// per the M4/M5/M6 GPU-deferred pattern), so the terrain shadow is called with
// horizon = 0 (a no-op that equals disk_fraction) and the glint with calm-sea wind.
// The CPU shipping path (render.rs) carries the fully per-pixel M3.

// Fraction of the finite solar disk above the local terrain horizon (rad).
fn terrain_shadow_fraction(sun_elev_rad: f32, horizon_rad: f32) -> f32 {
    return disk_fraction(sun_elev_rad - horizon_rad);
}

// Cox-Munk isotropic mean-square slope for a 10 m wind speed (m/s).
fn cox_munk_mss(wind: f32) -> f32 {
    return max(0.003 + 0.00512 * max(wind, 0.0), 1e-4);
}

// LAND daytime brightness gain (refinement pass, round 2; twin of render::land_day_gain):
// 1.0 at/below AERIAL_VEIL_ELEV_LO (twilight untouched) ramping to LAND_DAY_GAIN at/above
// AERIAL_VEIL_ELEV_HI. Land-only surface-reflectance lift, not the global exposure.
fn land_day_gain(sun_elev: f32) -> f32 {
    return 1.0 + smoothstep(AERIAL_VEIL_ELEV_LO, AERIAL_VEIL_ELEV_HI, sun_elev) * (LAND_DAY_GAIN - 1.0);
}

// Owner-selected v0.1.5 finished-visible LAND corrections. Exact f32 twin of
// render::{land_sza_normalization_gain, land_dark_toe_gain, land_appearance_gain}:
// independently switchable, bounded, scalar/colour-preserving, and exactly neutral
// through the established twilight band. The caller is the LAND branch only.
// LAND_APPEARANCE_TWIN_BEGIN
fn land_sza_normalization_gain_gpu(sun_elev: f32) -> f32 {
    let max_gain = clamp(u.land0.y, 1.0, 4.0);
    if (u.land0.x <= 0.5 || max_gain == 1.0 || sun_elev >= LAND_SZA_REFERENCE_ELEV) {
        return 1.0;
    }
    let mu_ref = sin(LAND_SZA_REFERENCE_ELEV * DEG2RAD);
    let mu_floor = sin(AERIAL_VEIL_ELEV_LO * DEG2RAD);
    let mu = sin(clamp(sun_elev, 0.0, 90.0) * DEG2RAD);
    let target_gain = clamp(mu_ref / max(mu, mu_floor), 1.0, max_gain);
    return 1.0 + smoothstep(AERIAL_VEIL_ELEV_LO, AERIAL_VEIL_ELEV_HI, sun_elev) * (target_gain - 1.0);
}

fn land_dark_toe_gain_gpu(sun_elev: f32, albedo: vec3<f32>) -> f32 {
    let knee = clamp(u.land0.w, 1e-6, 1.0);
    let gamma = clamp(u.land1.x, 0.05, 1.0);
    let max_gain = clamp(u.land1.y, 1.0, 4.0);
    if (u.land0.z <= 0.5) {
        return 1.0;
    }
    let y = max(dot(albedo, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0);
    if (y <= 0.0 || y >= knee || max_gain == 1.0 || gamma == 1.0) {
        return 1.0;
    }
    let power_target = knee * pow(y / knee, gamma);
    let w = smoothstep(0.0, knee, y);
    let target_y = power_target * (1.0 - w) + y * w;
    let gain = clamp(target_y / y, 1.0, max_gain);
    return 1.0 + smoothstep(AERIAL_VEIL_ELEV_LO, AERIAL_VEIL_ELEV_HI, sun_elev) * (gain - 1.0);
}

fn land_appearance_gain_gpu(sun_elev: f32, albedo: vec3<f32>) -> f32 {
    if (u.land0.x <= 0.5 && u.land0.z <= 0.5) {
        return 1.0;
    }
    return land_sza_normalization_gain_gpu(sun_elev) * land_dark_toe_gain_gpu(sun_elev, albedo);
}
// LAND_APPEARANCE_TWIN_END

// SUNRISE veil ramp (twin of render::aerial_veil_scale, low-sun visible pass): the full
// physical veil at/below the terminator gate, smoothly reducing to the daytime de-haze
// by VEIL_SUNRISE_ELEV_HI and held above (daytime byte-identical: the old ramp also sat
// at AERIAL_VEIL_DAY_SCALE at/above AERIAL_VEIL_ELEV_HI).
fn aerial_veil_scale(sun_elev: f32) -> f32 {
    return 1.0 - smoothstep(VEIL_TERMINATOR_ELEV, VEIL_SUNRISE_ELEV_HI, sun_elev) * (1.0 - AERIAL_VEIL_DAY_SCALE);
}

// LUT-derived low-sun illuminant gains (twin of render::low_sun_illuminant_gains):
// restore the GREEN of the reference-altitude direct-sun transmittance to its
// Rayleigh log-line (removing the Chappuis ozone green dip), preserve the R-B warm
// axis exactly, renormalize to unit Rec.709 luminance (uniform scale), taper to
// identity outside the 2-30 deg sunrise band.
fn low_sun_illuminant_gains(sun_elev: f32) -> vec3<f32> {
    let w = smoothstep(ILLUM_CORR_IN_LO, ILLUM_CORR_IN_HI, sun_elev)
        * (1.0 - smoothstep(ILLUM_CORR_OUT_LO, ILLUM_CORR_OUT_HI, sun_elev));
    if (w <= 0.0) {
        return vec3<f32>(1.0);
    }
    let mu = sin(sun_elev * DEG2RAD);
    let t = sample_transmittance(R_GROUND + ILLUM_REF_CLOUD_ALT, mu);
    if (min(t.x, min(t.y, t.z)) <= 0.0) {
        return vec3<f32>(1.0);
    }
    let a = (RAYLEIGH_SCA.y - RAYLEIGH_SCA.x) / (RAYLEIGH_SCA.z - RAYLEIGH_SCA.x);
    let t_g_ray = exp((1.0 - a) * log(t.x) + a * log(t.z));
    let g_green = max(t_g_ray / t.y, 1.0);
    let lum = vec3<f32>(0.2126, 0.7152, 0.0722);
    let y_raw = dot(t, lum);
    let y_corr = dot(vec3<f32>(t.x, t.y * g_green, t.z), lum);
    if (y_raw <= 0.0 || y_corr <= 0.0) {
        return vec3<f32>(1.0);
    }
    let s = y_raw / y_corr;
    return vec3<f32>(
        1.0 + w * (s - 1.0),
        1.0 + w * (s * g_green - 1.0),
        1.0 + w * (s - 1.0),
    );
}

// Unpolarised Fresnel reflectance, air -> medium of relative index n, at cos incidence.
fn fresnel_unpolarized(cos_i: f32, n: f32) -> f32 {
    if (n <= 0.0) {
        return 0.0;
    }
    let ci = clamp(cos_i, 0.0, 1.0);
    let sin_t2 = (1.0 - ci * ci) / (n * n);
    if (sin_t2 >= 1.0) {
        return 1.0;
    }
    let ct = sqrt(1.0 - sin_t2);
    let rs = (ci - n * ct) / (ci + n * ct);
    let rp = (n * ci - ct) / (n * ci + ct);
    return clamp(0.5 * (rs * rs + rp * rp), 0.0, 1.0);
}

// Cox-Munk sea-surface sun-glint reflectance factor rho = pi L / E_perp.
fn cox_munk_glint(to_sun: vec3<f32>, to_cam: vec3<f32>, up: vec3<f32>, mss: f32) -> f32 {
    let s = normalize(to_sun);
    let v = normalize(to_cam);
    let u = normalize(up);
    let mu_s = dot(s, u);
    let mu_v = dot(v, u);
    if (mu_s <= 1e-4 || mu_v <= 1e-4) {
        return 0.0;
    }
    let hf = s + v;
    let hlen = length(hf);
    if (hlen <= 1e-9) {
        return 0.0;
    }
    let nf = hf / hlen;
    let cos_beta = clamp(dot(nf, u), 1e-4, 1.0);
    let cos_omega = clamp(dot(s, nf), 0.0, 1.0);
    let tan2 = (1.0 - cos_beta * cos_beta) / (cos_beta * cos_beta);
    let m = max(mss, 1e-4);
    let p = exp(-tan2 / m) / (PI * m);
    let f = fresnel_unpolarized(cos_omega, WATER_N);
    let cb2 = cos_beta * cos_beta;
    let rho = PI * f * p / (4.0 * mu_s * mu_v * cb2 * cb2);
    return max(rho, 0.0);
}

fn sample_ambient(elev_deg: f32) -> vec3<f32> {
    let n = i32(u.p2.z);
    let t = clamp((elev_deg - u.p2.x) / (u.p2.y - u.p2.x), 0.0, 1.0);
    let f = t * f32(n - 1);
    let i0 = min(i32(floor(f)), n - 1);
    let i1 = min(i0 + 1, n - 1);
    let w = f - floor(f);
    let a = textureLoad(ambient_lut, vec2<i32>(i0, 0), 0).rgb;
    let b = textureLoad(ambient_lut, vec2<i32>(i1, 0), 0).rgb;
    return mix(a, b, w);
}

fn view_dir(px: f32, py: f32) -> vec3<f32> {
    let scan_x = u.ex.w + px * u.ez.w;
    let scan_y = u.ey.w - py * u.solar.w;
    let v_y = tan(scan_x);
    let v_z = tan(scan_y) * sqrt(1.0 + v_y * v_y);
    let world = u.ex.xyz + u.ey.xyz * v_y + u.ez.xyz * v_z;
    return normalize(world);
}

// Highlight desaturation: compress chroma toward Rec.709 luminance for pixels that
// are BOTH bright AND over-saturated (reddened low-sun anvils), rolling them toward
// amber/white instead of saturated orange; the dim twilight, the daytime median, and
// near-neutral daytime ground/cloud are untouched. Twin of
// atmosphere::desaturate_highlights.
fn desaturate_highlights(rho: vec3<f32>) -> vec3<f32> {
    let y = dot(rho, vec3<f32>(0.2126, 0.7152, 0.0722));
    let mx = max(rho.x, max(rho.y, rho.z));
    let mn = min(rho.x, min(rho.y, rho.z));
    var chroma = 0.0;
    if (mx > 1e-4) { chroma = (mx - mn) / mx; }
    let d = DESAT_MAX * smoothstep(DESAT_LUM_LO, DESAT_LUM_HI, y) * smoothstep(DESAT_SAT_LO, DESAT_SAT_HI, chroma);
    return mix(rho, vec3<f32>(y), d);
}

// Per-channel toe-lifted ABI sqrt stretch (twin of atmosphere::abi_reflectance_stretch):
// exactly sqrt at/above REFL_TOE_KNEE (daytime unchanged); lifted below (twilight).
fn abi_stretch_ch(rho: f32) -> f32 {
    let x = clamp(rho, 0.0, 1.0);
    let s = sqrt(x);
    if (x >= REFL_TOE_KNEE) {
        return s;
    }
    let toe = pow(x, REFL_TOE_GAMMA);
    let w = smoothstep(0.0, REFL_TOE_KNEE, x);
    return toe * (1.0 - w) + s * w;
}

// WS2 bounded highlight soft-clip (twin of render::soft_clip_highlight): strictly
// identity at/below the knee; a bounded monotone C1 Mobius shoulder mapping
// [knee, x_max] -> [knee, 1.0] with a nonzero end slope; hard 1.0 only above x_max.
// As x_max -> inf it reduces to the unbounded Reinhard shoulder (the same family).
fn soft_clip_highlight(x: f32, knee: f32, x_max: f32) -> f32 {
    if (x <= knee) {
        return x;
    }
    let span = 1.0 - knee;
    if (span <= 0.0) {
        return 1.0;
    }
    let w = x_max - knee;
    if (w <= 0.0 || x >= x_max) {
        return 1.0;
    }
    let a = x - knee;
    return knee + span * a / (a + span * (w - a) / w);
}

fn output_transform(rho: vec3<f32>) -> vec3<f32> {
    if (u.p1.w < 0.5) {
        // ABI-like reflectance factor: OPTIONAL synthesized green (the prototype ABI
        // display-green arithmetic, module const off by default — twin of the CPU
        // process-global), then desaturate highlights, then the bounded highlight
        // soft-clip (WS2 — the desaturate-then-shoulder ORDER matches the CPU path), then
        // the toe-lifted sqrt. This pass runs at implicit exposure 1.0 -> the shoulder
        // bound is RHO_HIGHLIGHT_MAX (the CPU seam derives exposure * RHO_HIGHLIGHT_MAX).
        var rr = rho;
        if (SYNTHETIC_GREEN_MODE > 0.5) {
            rr = vec3<f32>(
                rho.x,
                SYN_GREEN_W_RED * rho.x + SYN_GREEN_W_GREEN * rho.y + SYN_GREEN_W_BLUE * rho.z,
                rho.z,
            );
        }
        let ds = desaturate_highlights(rr);
        let sc = vec3<f32>(
            soft_clip_highlight(ds.x, CLOUD_SOFTCLIP_KNEE, RHO_HIGHLIGHT_MAX),
            soft_clip_highlight(ds.y, CLOUD_SOFTCLIP_KNEE, RHO_HIGHLIGHT_MAX),
            soft_clip_highlight(ds.z, CLOUD_SOFTCLIP_KNEE, RHO_HIGHLIGHT_MAX),
        );
        return vec3<f32>(abi_stretch_ch(sc.x), abi_stretch_ch(sc.y), abi_stretch_ch(sc.z));
    }
    // Debug: reflectance through sRGB gamma.
    return linear_to_srgb(clamp(rho, vec3<f32>(0.0), vec3<f32>(1.0)));
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(i32(pos.x), i32(pos.y));
    let g = textureLoad(lut_geo, coord, 0);
    let cam = u.cam.xyz;
    let sun_ecef = u.sun.xyz;
    // Integer pixel coords for the view ray, matching the CPU raster's
    // scan_angle(px, py) (built at integer px, not the +0.5 fragment centre).
    let view = view_dir(f32(coord.x), f32(coord.y));
    let e_sun = u.solar.xyz;

    if (g.x < 0.0) {
        // Off-earth (per CGMS): limb if the ray grazes the shell, else space.
        let top = ray_sphere(cam, view, R_TOP);
        if (top.y < top.x || top.y <= 0.0) {
            return vec4<f32>(0.0, 0.0, 0.0, 0.0); // space
        }
        let t_enter = max(top.x, 0.0);
        var t_exit = top.y;
        let gnd = ray_sphere(cam, view, R_GROUND);
        if (gnd.y >= gnd.x && gnd.x > t_enter && gnd.x < t_exit) {
            t_exit = gnd.x;
        }
        if (t_exit <= t_enter) {
            return vec4<f32>(0.0, 0.0, 0.0, 0.0);
        }
        let p_start = cam + view * t_enter;
        let sc = raymarch(p_start, view, sun_ecef, t_exit - t_enter);
        let rho = PI * sc.inscatter / max(e_sun, vec3<f32>(1e-6));
        return vec4<f32>(output_transform(rho), 1.0);
    }

    // On-earth surface pixel.
    let light = textureLoad(lut_light, coord, 0);

    var base: vec3<f32>;
    if (u.p1.x > 0.5) {
        base = textureSampleLevel(bm_tex, samp, vec2<f32>(g.x, g.y), 0.0).rgb;
    } else {
        base = vec3<f32>(u.p1.z, u.p1.z, u.p1.z);
    }
    var normal = vec3<f32>(0.0, 0.0, 1.0);
    var is_water = false;
    if (g.z >= 0.0) {
        let uv = vec2<f32>(g.z, g.w);
        normal = textureSampleLevel(normal_tex, samp, uv, 0.0).rgb * 2.0 - 1.0;
        let lm = textureSampleLevel(landmask_tex, samp, uv, 0.0).r;
        is_water = lm < 0.5;
    }
    normal = normalize(normal);

    var albedo = srgb_to_linear(base);
    if (is_water) {
        albedo = albedo * u.p1.y;
    } else {
        // True-color land vibrancy (refinement pass): luminance-preserving saturation.
        let y = dot(albedo, vec3<f32>(0.2126, 0.7152, 0.0722));
        albedo = max(vec3<f32>(y) + (albedo - vec3<f32>(y)) * LAND_VIBRANCY, vec3<f32>(0.0));
    }

    let sun_enu = vec3<f32>(light.x, light.y, light.z);
    let sun_elev = light.w;
    let ndotl = max(dot(normal, sun_enu), 0.0);
    // M3: the direct term sees the finite-disk fraction above the terrain horizon.
    // horizon = 0 on the active GPU pass (per-texel horizon map deferred), so this
    // equals the M2 disk_fraction; the CPU path folds the real per-pixel horizon.
    let disk = terrain_shadow_fraction(sun_elev * DEG2RAD, 0.0);
    // Transmittance to the sun at the surface; evaluated at max(elev, 0) so the
    // finite-disk crossing stays smooth (the disk fraction, not a hard mu gate,
    // handles the terminator). Below the whole disk, disk == 0 anyway.
    let mu_sun = sin(max(sun_elev, 0.0) * DEG2RAD);
    let t_sun = sample_transmittance(R_GROUND + 1.0, mu_sun);
    let e_ambient = sample_ambient(sun_elev);
    var l_surf: vec3<f32>;
    if (is_water) {
        // M3 Cox-Munk sun glint + Fresnel sky reflection (design section 5), replacing
        // M1 flat dark water. Calm-sea wind = 0 on the GPU pass (per-pixel U10/V10
        // upload deferred); the sky reflection uses the scalar ambient as a gray sky
        // (the directional SH radiance is a CPU-path feature) — documented divergence.
        let gnd2 = ray_sphere(cam, view, R_GROUND);
        let up_e = normalize(cam + view * max(gnd2.x, 0.0));
        let to_cam = -view;
        // GLINT_MSS_SCALE narrows the Cox-Munk core (round 2); GLINT_STRENGTH lifts the peak.
        let glint = cox_munk_glint(sun_ecef, to_cam, up_e, cox_munk_mss(0.0) * GLINT_MSS_SCALE) * GLINT_STRENGTH;
        let cos_view = max(dot(to_cam, up_e), 0.0);
        let f_sky = fresnel_unpolarized(cos_view, WATER_N);
        let l_glint = e_sun * (glint / PI) * t_sun * (disk * LIMB_DISK_AVG);
        // WS2 water direct sun (twin of render.rs): the water BODY sees the same
        // disk-gated direct term as land, DAY-GATED so twilight is byte-unchanged, with
        // the water albedo simultaneously retuned toward WATER_ALBEDO_DAY_SCALE on the
        // same gate. No cloud shadow on this clouds-off pass (shadow = 1 implicitly).
        let day_t = smoothstep(AERIAL_VEIL_ELEV_LO, AERIAL_VEIL_ELEV_HI, sun_elev);
        var scale_ratio = 1.0;
        if (u.p1.y > 0.0) {
            scale_ratio = 1.0 + day_t * (WATER_ALBEDO_DAY_SCALE / u.p1.y - 1.0);
        }
        let e_direct_w = e_sun * t_sun * (disk * ndotl * LIMB_DISK_AVG * day_t);
        l_surf = albedo * scale_ratio / PI * (e_direct_w + e_ambient) + l_glint + f_sky * (e_ambient / PI);
    } else {
        let e_direct = e_sun * t_sun * (disk * ndotl * LIMB_DISK_AVG);
        l_surf = albedo / PI * (e_direct + e_ambient);
        // Finished-visible appearance controls are LAND-only and precede the legacy
        // land gain/aerial veil, matching render::surface_toa_radiance exactly.
        l_surf = l_surf * land_appearance_gain_gpu(sun_elev, albedo);
        // LAND daytime brightness lift (round 2): ground-only surface-reflectance gain,
        // sun-gated so twilight is byte-unchanged. Applied before the aerial veil below.
        l_surf = l_surf * land_day_gain(sun_elev);
    }

    // Aerial perspective: raymarch the shell from atmosphere entry to the ground.
    let top = ray_sphere(cam, view, R_TOP);
    let gnd = ray_sphere(cam, view, R_GROUND);
    let t_enter = max(top.x, 0.0);
    var t_ground = top.y;
    if (gnd.y >= gnd.x && gnd.x > t_enter && gnd.x < t_ground) {
        t_ground = gnd.x;
    }
    var l_toa = l_surf;
    if (t_ground > t_enter) {
        let p_start = cam + view * t_enter;
        let sc = raymarch(p_start, view, sun_ecef, t_ground - t_enter);
        // SUNRISE veil ramp (low-sun visible pass): scale the additive surface
        // in-scatter; the terminator band (<= VEIL_TERMINATOR_ELEV) keeps the full
        // physical veil, daytime keeps the refinement de-haze.
        let veil = select(1.0, aerial_veil_scale(sun_elev), u.p2.w > 0.5);
        l_toa = l_surf * sc.transmittance + veil * sc.inscatter;
    }
    // Low-sun illuminant correction at the display seam (on-earth pixels only; the
    // off-earth limb above keeps its physical color). Identity outside 2-30 deg.
    let illum = low_sun_illuminant_gains(sun_elev);
    let rho = illum * PI * l_toa / max(e_sun, vec3<f32>(1e-6));
    return vec4<f32>(output_transform(rho), 1.0);
}
