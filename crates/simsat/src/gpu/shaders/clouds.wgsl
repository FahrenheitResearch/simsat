// SimSat cloud raymarch pass (design doc section 4, M4). GPU twin of the CPU
// reference in `clouds.rs` (the march/optics) + `render.rs`/`atmosphere.rs` (the
// surface radiance it composites over). A SUPERSET of `surface.wgsl`: it shades the
// M2 surface (with cloud shadows on the ground from the sun-OD map), marches the WRF
// cloud volume along the true ECEF slant ray, composites, applies the froxel aerial
// perspective, and tonemaps — in one fragment pass.
//
// Camera rays are selected explicitly: the historical geostationary scan fan is
// unchanged, while Top-down loads a reviewed per-pixel local-up vector and constructs
// the CPU-equivalent nadir camera/ray. GEO samples its scan-space front froxel;
// Top-down directly marches its per-pixel camera->cloud atmospheric column.
//
// M4 ships the CPU render path (`clouds::shade_cloud_pixel`, tested on the headless
// nodes); this shader is the naga-validated GPU render twin activated in M5. Kept in
// lockstep with `clouds.rs` by discipline (the same twin workflow M2 established).
//
// Approximations named in code: hardware trilinear on the u8 codes then decode (vs
// the CPU's trilerp of decoded extinction, a ~quantization-granularity difference);
// nearest froxel depth. The in-cloud sun transmittance is a short DEPTH-RESOLVED
// secondary light march toward the sun (mirrors clouds.rs after M4 review FINDING 1);
// the sun-OD map drives ground cloud-shadow and the support-only thin-multiscatter gate,
// never sample-to-sun transmittance. The froxel is indexed by the
// ATMOSPHERE-shell traversal fraction of the cloud centroid (FINDING 4), and the front
// airlight is weighted by (1 - T_cloud) so it is not double-counted.
//
// M5 mirrors (parity for the M5-GPU activation; the CPU reference in clouds.rs is the
// shipping path): the sun term is the Wrenninge/Oz multi-scatter OCTAVE sum; the ground
// cloud shadow is PENUMBRAL (blur radius = occluder distance x tan 0.2665 deg); the sky
// ambient is the SH-2 directional projection (bindings 14/15). surface.wgsl (the active
// clouds-OFF pass) still carries the M2 scalar ambient — a documented CPU/GPU divergence.
//
// ACTIVE as of the gpu-clouds pass (feat/gpu-clouds): this pass is dispatched by
// `gpu::CloudPassResources` behind the studio's "GPU clouds (experimental)" toggle —
// the INTERACTIVE-schedule live preview. The CPU composite remains the shipping
// default, the stored-frame path, and the parity ground truth. Reconciled to the
// current CPU behavior in that pass: EXPOSURE (u.m1.x), multi-scatter OCTAVES
// (u.m1.y), GROUND_DAY_LIFT (u.frx2.z), the WS2 CLOUD_SHADOW_FLOOR effective shadow,
// and the zoom-out-margin EDGE FEATHER (u.frx2.y). The remaining divergences are
// enumerated in notes/gpu-clouds-notes.md (granulation off, M3 per-texel uploads
// deferred, interactive sun schedule, trilinear-on-codes sampling, f32 ALU).

const PI: f32 = 3.14159265358979;
const DEG2RAD: f32 = 0.017453292519943295;

// Atmosphere constants — MUST match atmosphere.rs / surface.wgsl.
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
const STEPS: u32 = 32u;
// True-color calibration (refinement pass; twins of render.rs constants). Round 2:
// more de-haze, more land vibrancy, a LAND-only daylight brightness lift, and a
// brighter + tighter sun glint. Surface help begins above the horizon.
const AERIAL_VEIL_DAY_SCALE: f32 = 0.40;
const AERIAL_VEIL_ELEV_LO: f32 = 20.0;
const AERIAL_VEIL_ELEV_HI: f32 = 30.0; // WS2: 40 -> 30 (full daytime treatment by 30 deg; twin of render.rs)
const SURFACE_HELP_ELEV_LO: f32 = 0.0;
const SURFACE_HELP_ELEV_HI: f32 = 12.0;
const SURFACE_TWILIGHT_IN_LO: f32 = -6.0;
const SURFACE_TWILIGHT_IN_HI: f32 = 0.0;
const SURFACE_TWILIGHT_OUT_LO: f32 = 4.0;
const SURFACE_TWILIGHT_OUT_HI: f32 = 12.0;
const LAND_VIBRANCY: f32 = 1.45;
const LAND_DAY_GAIN: f32 = 1.20;   // LAND-only daytime surface-reflectance lift (not the global exposure)
const LAND_SZA_REFERENCE_ELEV: f32 = 60.0;
const GLINT_STRENGTH: f32 = 3.5;
const GLINT_MSS_SCALE: f32 = 0.4;  // < 1 tightens the Cox-Munk core -> smaller, brighter glint streak (round 3: 0.6->0.4)
// WS2 bright-cloud tonemap (twins of render.rs). EXPOSURE is wired as u.m1.x
// (gpu-clouds activation): rho = exposure * pi * L / E_sun, and the shoulder bound is
// x_max = exposure * RHO_HIGHLIGHT_MAX — the exact seam of the CPU
// render::radiance_to_rgba_softclip. A non-positive uniform falls back to 1.0.
const CLOUD_SOFTCLIP_KNEE: f32 = 0.65;   // identity below; bounded Mobius shoulder above
const RHO_HIGHLIGHT_MAX: f32 = 1.25;     // physical reflectance ceiling -> display 1.0
// WS2 diffuse cloud-shadow floor (twin of render::CLOUD_SHADOW_FLOOR /
// effective_cloud_shadow): elevation-independent because it multiplies only direct
// sunlight, which already vanishes at night. The specular glint keeps the RAW shadow.
const CLOUD_SHADOW_FLOOR: f32 = 0.45;
// Low-sun visible pass (twins of render.rs; see surface.wgsl for the full notes):
// the SUNRISE veil ramp + the LUT-derived low-sun ILLUMINANT correction.
const VEIL_TERMINATOR_ELEV: f32 = 2.0;   // full physical veil at/below (dusk band byte-identical)
const VEIL_SUNRISE_ELEV_HI: f32 = 16.0;  // full daytime de-haze in place at/above
const ILLUM_REF_CLOUD_ALT: f32 = 7000.0; // reference cloud altitude (m) of the illuminant sample
const ILLUM_CORR_IN_LO: f32 = 2.0;       // identity at/below (dusk band byte-identical)
const ILLUM_CORR_IN_HI: f32 = 5.0;       // full correction at/above (green-restoration form; twin of render.rs)
const ILLUM_CORR_OUT_LO: f32 = 20.0;     // taper-off start
const ILLUM_CORR_OUT_HI: f32 = 30.0;     // identity at/above (daytime byte-identical)
// ABI SYNTHETIC-GREEN display mode (prototype; OFF by default — twin of the CPU
// process-global render::set_synthetic_green; flip together on adoption).
const SYNTHETIC_GREEN_MODE: f32 = 0.0;
const SYN_GREEN_W_RED: f32 = 0.45;
const SYN_GREEN_W_GREEN: f32 = 0.10;
const SYN_GREEN_W_BLUE: f32 = 0.45;
const WATER_ALBEDO_DAY_SCALE: f32 = 0.35; // daylight water scale (horizon/night anchor = u.p1.y)

// Cloud optics — MUST match clouds.rs.
// SCHEDULE NOTE (WS1): this GPU twin keeps the INTERACTIVE sun-march schedule
// (6 steps, growth 2.0) — it is the deferred interactive-preview activation; the
// CPU shipping path selects (10, 1.5) for Offline via MarchConfig::new. Both sides
// share the WS1 shell-exit extension + stratified jitter below (documented
// divergence, per the M4/M5 pattern).
const SUN_MARCH_STEPS: i32 = 6;    // secondary sun-march steps (depth-resolved shadow)
const SUN_MARCH_GROWTH: f32 = 2.0; // exponential step growth
const SUN_MARCH_JITTER: f32 = 0.0; // stratified-jitter amplitude (clouds.rs SUN_MARCH_JITTER_AMP; 0 = fixed midpoint, see the look note there)
const PHASE_LIQUID_G1: f32 = 0.85;
const PHASE_LIQUID_G2: f32 = -0.15;
const PHASE_LIQUID_W: f32 = 0.9;
const PHASE_ICE_G1: f32 = 0.75;
const PHASE_ICE_G2: f32 = -0.10;
const PHASE_ICE_W: f32 = 0.9;
const AMBIENT_W_ABOVE: f32 = 0.7;
const AMBIENT_W_BELOW: f32 = 0.3;

// Wrenninge/Oz multi-scatter octaves (M5) — MUST match clouds.rs. octave 0 == fix2
// single scatter; deeper octaves scale sun-Beer extinction a^k, phase eccentricity b^k,
// brightness weight c^k. See clouds.rs octave-constants block for the physics/citation.
const OCTAVES: i32 = 6;               // DEFAULT_OCTAVES (the max + fallback; live count = u.m1.y)
const OCTAVE_EXTINCTION_SCALE: f32 = 0.5;
const OCTAVE_PHASE_SCALE: f32 = 0.5;
const OCTAVE_BRIGHTNESS_SCALE: f32 = 0.85;
// Tangent of the solar angular RADIUS (0.2665 deg), used as a disk-convolution radius.
const SUN_ANG_RADIUS_TAN: f32 = 0.00465003;

// Sub-grid cloud GRANULATION (edge-erosion detail noise) — MUST match the clouds.rs
// granulation section (constants, hash, Worley octaves, gate, remap multiplier).
// DEFERRED-ACTIVATION mirror (the M4/M5-GPU family): the math is mirrored and
// naga-validated, but GRAN_AMPLITUDE is a shader constant 0.0 so this twin's output
// is byte-unchanged until the GPU cloud-pass activation wires the per-frame
// amplitude (clouds.rs Granulation::amplitude, dx-derived) as a uniform. NOTE for
// the activation: sun_od.wgsl (the sun-OD compute twin) must receive the SAME
// erosion so the ground shadows match — the CPU path threads one Granulation value
// through the view march, sun march and sun-OD accumulation.
const GRAN_AMPLITUDE: f32 = 0.0;      // activation: uniform = Granulation::amplitude
const GRAN_AMP_CAP: f32 = 0.6;        // Cahalan-bound amplitude cap (clouds.rs GRAN_AMP_CAP)
const GRAN_EROSION_MAX: f32 = 0.98;
const GRAN_HEIGHT_FULL_M: f32 = 4000.0;
const GRAN_HEIGHT_ZERO_M: f32 = 7000.0;
const GRAN_CARVE_LO: f32 = 0.46;  // round-2 retune (clouds.rs: wider gap network)
const GRAN_CARVE_HI: f32 = 0.58;
const GRAN_INTERIOR_LO: f32 = 0.45; // interior protection window on the relative density
const GRAN_INTERIOR_HI: f32 = 0.75; // (solid-deck variability never erodes; clouds.rs twin)
const GRAN_SCALE0_M: f32 = 1000.0;    // Worley octave cell scales (k^-5/3 envelope)
const GRAN_SCALE1_M: f32 = 500.0;
const GRAN_SCALE2_M: f32 = 250.0;
const GRAN_W0: f32 = 0.4125987;       // lambda^(1/3) weights, normalised (clouds.rs)
const GRAN_W1: f32 = 0.3274800;
const GRAN_W2: f32 = 0.2599213;
const GRAN_SALT0: u32 = 0x51A7C0DEu;
const GRAN_SALT1: u32 = 0x9BD2A0E5u;
const GRAN_SALT2: u32 = 0x2F63D19Bu;
// Round-2 tuning (clouds.rs twins): the BIMODAL carve shape (gap-or-grain remap
// reshaping) and the DOMAIN WARP (low-frequency value-noise displacement of the
// Worley sample position — cell size/spacing varies across the scene).
const GRAN_BIMODAL_GAP: f32 = 0.25;
const GRAN_BIMODAL_GRAIN: f32 = 0.65;
const GRAN_WARP_SCALE_M: f32 = 4000.0;
const GRAN_WARP_AMP_M: f32 = 1300.0;
const GRAN_WARP_SALT_U: u32 = 0x1B56C4E9u;
const GRAN_WARP_SALT_V: u32 = 0x7A991E3Du;

struct Uniforms {
    // --- surface (M2, layout verbatim from surface.wgsl) ---
    cam: vec4<f32>,
    sun: vec4<f32>,
    ex: vec4<f32>,
    ey: vec4<f32>,
    ez: vec4<f32>,
    solar: vec4<f32>,
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
    // --- cloud (M4) ---
    dims: vec4<f32>,   // nx, ny, nz, voxel_pitch
    vert: vec4<f32>,   // z_min, dz, r_top(brick), r_bottom(brick)
    geo0: vec4<f32>,   // proj_kind, ref_i, ref_j, dx
    geo1: vec4<f32>,   // ref_u, ref_v, dy, central_meridian_deg
    geo2: vec4<f32>,   // lambert_n, lambert_f, ps_k, merc_scale
    geo3: vec4<f32>,   // south_pole, unused, unused, unused
    ql: vec4<f32>,     // ext_liquid vmin,vmax ; ext_ice vmin,vmax
    qp: vec4<f32>,     // ext_precip vmin,vmax ; tau_up vmin,vmax
    m0: vec4<f32>,     // coarse_step_m, fine_step_m, max_steps, unused (was detail_taps)
    m1: vec4<f32>,     // exposure, octaves, beer_powder, ground_albedo (sun march uses u.dims.w)
    sod_c: vec4<f32>,  // sun_od center xyz, transmittance_floor
    sod_u: vec4<f32>,  // au xyz, u_min
    sod_v: vec4<f32>,  // av xyz, u_max
    sod_e: vec4<f32>,  // v_min, v_max, sunod_dim, clouds_enabled
    frx: vec4<f32>,    // scan x_min, x_max, y_min, y_max
    frx2: vec4<f32>,   // froxel_dim, edge_feather_cells, ground_day_lift, visible cloud OD scale
    land0: vec4<f32>,  // sza_enabled, sza_max_gain, dark_toe_enabled, dark_toe_knee
    land1: vec4<f32>,  // dark_toe_gamma, dark_toe_max_gain, unused, unused
    toe0: vec4<f32>,   // enabled, knee, gamma, max_gain (post-view surface toe)
    twi0: vec4<f32>,   // enabled, knee, gamma, max_gain (tight low-sun recovery)
    ray0: vec4<f32>,   // view_mode (0 geo, 1 per-pixel nadir), topdown camera radius, shadow-AA, unused
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
@group(0) @binding(10) var volume: texture_3d<f32>;
@group(0) @binding(11) var occupancy: texture_3d<f32>;
@group(0) @binding(12) var sun_od: texture_2d<f32>;
@group(0) @binding(13) var froxel: texture_3d<f32>;
// M5 mirrors (activated in M5-GPU): SH-2 directional sky ambient (n rows x 9 coef
// columns of RGB) + the sun-OD occluder-distance channel (for the penumbra).
@group(0) @binding(14) var sh_ambient: texture_2d<f32>;
@group(0) @binding(15) var sun_od_dist: texture_2d<f32>;
@group(0) @binding(16) var topdown_ray_lut: texture_2d<f32>;

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

fn bilinear(c00: vec3<f32>, c10: vec3<f32>, c01: vec3<f32>, c11: vec3<f32>, tx: f32, ty: f32) -> vec3<f32> {
    return mix(mix(c00, c10, tx), mix(c01, c11, tx), ty);
}

fn ray_hits_ground(r: f32, mu: f32) -> bool {
    return mu < 0.0 && (r * r * (mu * mu - 1.0) + R_GROUND * R_GROUND) >= 0.0;
}

fn distance_to_top(r: f32, mu: f32) -> f32 {
    let disc = r * r * (mu * mu - 1.0) + R_TOP * R_TOP;
    return max(0.0, -r * mu + sqrt(max(0.0, disc)));
}

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

fn rayleigh_phase(cos_t: f32) -> f32 {
    return 3.0 / (16.0 * PI) * (1.0 + cos_t * cos_t);
}

fn cornette_shanks(cos_t: f32, g: f32) -> f32 {
    let g2 = g * g;
    let num = 3.0 * (1.0 - g2) * (1.0 + cos_t * cos_t);
    let denom = 8.0 * PI * (2.0 + g2) * pow(1.0 + g2 - 2.0 * g * cos_t, 1.5);
    return num / denom;
}

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
// optics.rs; naga-validated). This is the DEFERRED GPU cloud twin (activated with the
// M5-GPU cloud pass): the per-texel horizon map / wind / snow / aperture are not yet
// uploaded, so the terrain shadow uses horizon = 0 and the glint calm-sea wind, and
// the ambient keeps the M5 full-hemisphere SH (the aperture upload is deferred). The
// CPU shipping path (render.rs) carries the fully per-pixel M3.
const WATER_N: f32 = 1.34;

fn terrain_shadow_fraction(sun_elev_rad: f32, horizon_rad: f32) -> f32 {
    return disk_fraction(sun_elev_rad - horizon_rad);
}

fn cox_munk_mss(wind: f32) -> f32 {
    return max(0.003 + 0.00512 * max(wind, 0.0), 1e-4);
}

// LAND daytime brightness gain (refinement pass, round 2; twin of render::land_day_gain).
fn land_day_gain(sun_elev: f32) -> f32 {
    return 1.0 + smoothstep(SURFACE_HELP_ELEV_LO, SURFACE_HELP_ELEV_HI, sun_elev) * (LAND_DAY_GAIN - 1.0);
}

// Owner-selected v0.1.5 finished-visible LAND corrections. Exact f32 twin of
// render::{land_sza_normalization_gain, land_dark_toe_gain, land_appearance_gain}.
// These scale only the surface signal in the LAND branch; cloud source radiance,
// water/glint, limb/space, raw bands, and thermal/derived paths never consume them.
// LAND_APPEARANCE_TWIN_BEGIN
fn land_sza_normalization_gain_gpu(sun_elev: f32) -> f32 {
    let max_gain = clamp(u.land0.y, 1.0, 4.0);
    if (u.land0.x <= 0.5 || max_gain == 1.0 || sun_elev >= LAND_SZA_REFERENCE_ELEV) {
        return 1.0;
    }
    let mu_ref = sin(LAND_SZA_REFERENCE_ELEV * DEG2RAD);
    let mu_floor = sin(SURFACE_HELP_ELEV_HI * DEG2RAD);
    let mu = sin(clamp(sun_elev, 0.0, 90.0) * DEG2RAD);
    let target_gain = clamp(mu_ref / max(mu, mu_floor), 1.0, max_gain);
    return 1.0 + smoothstep(SURFACE_HELP_ELEV_LO, SURFACE_HELP_ELEV_HI, sun_elev) * (target_gain - 1.0);
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
    return 1.0 + smoothstep(SURFACE_HELP_ELEV_LO, SURFACE_HELP_ELEV_HI, sun_elev) * (gain - 1.0);
}

fn land_appearance_gain_gpu(sun_elev: f32, albedo: vec3<f32>) -> f32 {
    if (u.land0.x <= 0.5 && u.land0.z <= 0.5) {
        return 1.0;
    }
    return land_sza_normalization_gain_gpu(sun_elev) * land_dark_toe_gain_gpu(sun_elev, albedo);
}
// LAND_APPEARANCE_TWIN_END

// Post-light terrain recovery. Both branches consume only the already-lit,
// view-attenuated LAND contribution and run before additive atmospheric airlight.
// The established toe retains its daylight gate; the stronger independent recovery
// uses only the tight -6..+12 degree twilight/low-sun window. Their gains combine by
// max so enabling both cannot compound them multiplicatively.
// SURFACE_POSTLIGHT_TOE_TWIN_BEGIN
fn postlight_dark_gain_gpu(surface: vec3<f32>, config: vec4<f32>) -> f32 {
    if (config.x <= 0.5) {
        return 1.0;
    }
    let knee = clamp(config.y, 1e-6, 1.0);
    let gamma = clamp(config.z, 0.05, 1.0);
    let max_gain = clamp(config.w, 1.0, 4.0);
    let rho = PI * surface / max(u.solar.xyz, vec3<f32>(1e-6));
    let y = max(dot(rho, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0);
    if (y <= 0.0 || y >= knee || max_gain == 1.0 || gamma == 1.0) {
        return 1.0;
    }
    let power_target = knee * pow(y / knee, gamma);
    let w = smoothstep(0.0, knee, y);
    let target_y = power_target * (1.0 - w) + y * w;
    return clamp(target_y / y, 1.0, max_gain);
}

fn surface_postlight_toe_gain_gpu(surface: vec3<f32>, sun_elev: f32) -> f32 {
    let target_gain = postlight_dark_gain_gpu(surface, u.toe0);
    let weight = smoothstep(SURFACE_HELP_ELEV_LO, SURFACE_HELP_ELEV_HI, sun_elev);
    return 1.0 + weight * (target_gain - 1.0);
}

fn twilight_surface_recovery_gain_gpu(surface: vec3<f32>, sun_elev: f32) -> f32 {
    let target_gain = postlight_dark_gain_gpu(surface, u.twi0);
    let weight = smoothstep(SURFACE_TWILIGHT_IN_LO, SURFACE_TWILIGHT_IN_HI, sun_elev)
        * (1.0 - smoothstep(SURFACE_TWILIGHT_OUT_LO, SURFACE_TWILIGHT_OUT_HI, sun_elev));
    return 1.0 + weight * (target_gain - 1.0);
}

fn combined_surface_recovery_gain_gpu(surface: vec3<f32>, sun_elev: f32) -> f32 {
    return max(
        surface_postlight_toe_gain_gpu(surface, sun_elev),
        twilight_surface_recovery_gain_gpu(surface, sun_elev),
    );
}
// SURFACE_POSTLIGHT_TOE_TWIN_END

// GROUND LIFT daylight gain on the WHOLE surface radiance — land AND water (twin of
// render::ground_day_lift; the top-down/basemap appearance pass). The lift value is
// u.frx2.z (the CPU MarchConfig::ground_day_lift, default render::GROUND_DAY_LIFT);
// 1.0 at/below the horizon, reaching the requested lift by 12 degrees.
fn ground_day_lift_gain(sun_elev: f32) -> f32 {
    let lift = max(u.frx2.z, 0.0);
    if (lift <= 0.0) {
        return 1.0;
    }
    return 1.0 + smoothstep(SURFACE_HELP_ELEV_LO, SURFACE_HELP_ELEV_HI, sun_elev) * (lift - 1.0);
}

// The EFFECTIVE cloud shadow the DIFFUSE direct-sun terms see (twin of
// render::effective_cloud_shadow, WS2): f + (1-f)*shadow with
// f = CLOUD_SHADOW_FLOOR. The specular glint keeps the
// RAW shadow (an occluded solar disk has no mirror image; the floor is diffuse fill).
fn effective_cloud_shadow_gpu(shadow: f32, sun_elev: f32) -> f32 {
    let s = clamp(shadow, 0.0, 1.0);
    let f = CLOUD_SHADOW_FLOOR;
    return f + (1.0 - f) * s;
}

// Zoom-out-margin EDGE FEATHER (twin of clouds::edge_feather): the cloud contribution
// ramps to zero over the outer u.frx2.y cells of the domain so clouds melt into a
// margin instead of a hard cutoff. band <= 0 -> 1.0 everywhere (the no-op at margin 0).
fn edge_feather(fi: f32, fj: f32) -> f32 {
    let band = u.frx2.y;
    if (band <= 0.0) {
        return 1.0;
    }
    let hi_i = u.dims.x - 1.0;
    let hi_j = u.dims.y - 1.0;
    let d = min(min(fi, hi_i - fi), min(fj, hi_j - fj));
    if (d <= 0.0) {
        return 0.0;
    }
    if (d >= band) {
        return 1.0;
    }
    let t = d / band;
    return t * t * (3.0 - 2.0 * t);
}

// Validated on CPU before packing; clamp again at the consumption seam so the shader
// remains bounded if an alternate caller builds uniforms directly.
fn cloud_od_scale() -> f32 {
    return clamp(u.frx2.w, 0.0, 4.0);
}

// SUNRISE veil ramp (twin of render::aerial_veil_scale / surface.wgsl).
fn aerial_veil_scale(sun_elev: f32) -> f32 {
    return 1.0 - smoothstep(VEIL_TERMINATOR_ELEV, VEIL_SUNRISE_ELEV_HI, sun_elev) * (1.0 - AERIAL_VEIL_DAY_SCALE);
}

// LUT-derived low-sun illuminant gains (twin of render::low_sun_illuminant_gains /
// surface.wgsl): restore GREEN to the Rayleigh log-line (the ozone dip), preserve the
// R-B warm axis, unit-luminance renormalize, taper to identity outside 2-30 deg.
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

fn cox_munk_glint(to_sun: vec3<f32>, to_cam: vec3<f32>, up: vec3<f32>, mss: f32) -> f32 {
    let s = normalize(to_sun);
    let v = normalize(to_cam);
    let uu = normalize(up);
    let mu_s = dot(s, uu);
    let mu_v = dot(v, uu);
    if (mu_s <= 1e-4 || mu_v <= 1e-4) {
        return 0.0;
    }
    let hf = s + v;
    let hlen = length(hf);
    if (hlen <= 1e-9) {
        return 0.0;
    }
    let nf = hf / hlen;
    let cos_beta = clamp(dot(nf, uu), 1e-4, 1.0);
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

fn geostationary_view_dir(px: f32, py: f32) -> vec3<f32> {
    let scan_x = u.ex.w + px * u.ez.w;
    let scan_y = u.ey.w - py * u.solar.w;
    let v_y = tan(scan_x);
    let v_z = tan(scan_y) * sqrt(1.0 + v_y * v_y);
    let world = u.ex.xyz + u.ey.xyz * v_y + u.ez.xyz * v_z;
    return normalize(world);
}

struct CameraRay {
    cam: vec3<f32>,
    view: vec3<f32>,
    valid: f32,
};

// Reviewed twin of camera::topdown_nadir_ray. The CPU uploads the local ECEF up
// vector derived by that exact function; this shader only normalizes and applies
// the shared synthetic camera radius. Geo takes the historical scan-ray branch.
fn camera_ray(coord: vec2<i32>) -> CameraRay {
    if (u.ray0.x < 0.5) {
        return CameraRay(
            u.cam.xyz,
            geostationary_view_dir(f32(coord.x), f32(coord.y)),
            1.0,
        );
    }
    let r = textureLoad(topdown_ray_lut, coord, 0);
    if (r.w <= 0.5 || dot(r.xyz, r.xyz) <= 0.0) {
        return CameraRay(vec3<f32>(0.0), vec3<f32>(0.0), 0.0);
    }
    let up = normalize(r.xyz);
    return CameraRay(up * u.ray0.y, -up, 1.0);
}

// Highlight desaturation: compress chroma toward Rec.709 luminance for pixels that
// are BOTH bright AND over-saturated (reddened low-sun anvils); the dim twilight, the
// daytime median, and near-neutral daytime ground/cloud are untouched. Twin of
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

// Per-channel toe-lifted ABI sqrt stretch (twin of atmosphere::abi_reflectance_stretch).
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

fn synthesize_abi_green(rho: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        rho.x,
        SYN_GREEN_W_RED * rho.x + SYN_GREEN_W_GREEN * rho.y + SYN_GREEN_W_BLUE * rho.z,
        rho.z,
    );
}

// The display EXPOSURE gain (u.m1.x; Studio default 1.5, exact neutral override 1.0). Twin of
// render::radiance_to_rgba_softclip's gain guard: a
// non-finite/non-positive uniform falls back to 1.0 (never darkens to nothing).
fn exposure_gain() -> f32 {
    if (u.m1.x > 0.0) {
        return u.m1.x;
    }
    return 1.0;
}

fn output_transform(rho: vec3<f32>) -> vec3<f32> {
    if (u.p1.w < 0.5) {
        // ABI-like reflectance factor: OPTIONAL synthesized green (prototype, module
        // const off by default — twin of the CPU process-global), then desaturate
        // highlights, then the bounded highlight soft-clip (WS2 — the desaturate-then-
        // shoulder ORDER matches the CPU path), then the toe-lifted sqrt. The caller
        // has ALREADY applied the exposure gain to rho, so the exposure-aware shoulder
        // bound is exposure * RHO_HIGHLIGHT_MAX — the exact CPU seam
        // (render::radiance_to_rgba_softclip).
        let x_max = exposure_gain() * RHO_HIGHLIGHT_MAX;
        var rr = rho;
        if (SYNTHETIC_GREEN_MODE > 0.5) {
            rr = synthesize_abi_green(rho);
        }
        let ds = desaturate_highlights(rr);
        let sc = vec3<f32>(
            soft_clip_highlight(ds.x, CLOUD_SOFTCLIP_KNEE, x_max),
            soft_clip_highlight(ds.y, CLOUD_SOFTCLIP_KNEE, x_max),
            soft_clip_highlight(ds.z, CLOUD_SOFTCLIP_KNEE, x_max),
        );
        return vec3<f32>(abi_stretch_ch(sc.x), abi_stretch_ch(sc.y), abi_stretch_ch(sc.z));
    }
    return linear_to_srgb(clamp(rho, vec3<f32>(0.0), vec3<f32>(1.0)));
}

// ── cloud volume helpers (twin of clouds.rs) ──
fn decode_channel(v_norm: f32, vmin: f32, vmax: f32) -> f32 {
    let code = v_norm * 255.0;
    if (code < 0.5 || vmax <= 0.0) {
        return 0.0;
    }
    let t = (code - 1.0) / 254.0;
    return vmin * pow(vmax / vmin, t);
}

// WRF projection forward: geodetic (lat, lon) deg -> projection plane (u, v).
fn project(lat_deg: f32, lon_deg: f32) -> vec2<f32> {
    let kind = i32(u.geo0.x + 0.5);
    let cm = u.geo1.w;
    let phi = clamp(lat_deg, -89.999, 89.999) * DEG2RAD;
    var dlon = lon_deg - cm;
    dlon = dlon - 360.0 * floor((dlon + 180.0) / 360.0);
    let dlon_r = dlon * DEG2RAD;
    if (kind == 0) {
        let n = u.geo2.x;
        let f = u.geo2.y;
        let rho = R_GROUND * f / pow(tan(PI * 0.25 + phi * 0.5), n);
        let theta = n * dlon_r;
        return vec2<f32>(rho * sin(theta), -rho * cos(theta));
    } else if (kind == 1) {
        let k = u.geo2.z;
        if (u.geo3.x > 0.5) {
            let rho = 2.0 * R_GROUND * k * tan(PI * 0.25 + phi * 0.5);
            return vec2<f32>(rho * sin(dlon_r), rho * cos(dlon_r));
        }
        let rho = 2.0 * R_GROUND * k * tan(PI * 0.25 - phi * 0.5);
        return vec2<f32>(rho * sin(dlon_r), -rho * cos(dlon_r));
    } else if (kind == 2) {
        let scale = u.geo2.w;
        return vec2<f32>(R_GROUND * scale * dlon_r, R_GROUND * scale * log(tan(PI * 0.25 + phi * 0.5)));
    }
    return vec2<f32>(dlon, clamp(lat_deg, -89.999, 89.999));
}

// ECEF point -> fractional brick coords (fi, fj, fk).
fn ecef_to_brick(p: vec3<f32>) -> vec3<f32> {
    let r = length(p);
    let h = r - R_GROUND;
    let fk = (h - u.vert.x) / u.vert.y;
    let lat = degrees(asin(clamp(p.z / r, -1.0, 1.0)));
    let lon = degrees(atan2(p.y, p.x));
    let uv = project(lat, lon);
    let fi = u.geo0.y + (uv.x - u.geo1.x) / u.geo0.w;
    let fj = u.geo0.z + (uv.y - u.geo1.y) / u.geo1.z;
    return vec3<f32>(fi, fj, fk);
}

// ── sub-grid cloud GRANULATION (twin of the clouds.rs granulation section) ──

// Deterministic cell hash to [0, 1) (twin of clouds.rs gran_cell_hash01 — the
// hash01_position-style integer avalanche; u32 arithmetic wraps in WGSL).
fn gran_cell_hash01(ix: i32, iy: i32, salt: u32) -> f32 {
    var h: u32 = bitcast<u32>(ix) * 0x9E3779B9u
        + bitcast<u32>(iy) * 0x85EBCA6Bu
        + salt * 0xC2B2AE35u;
    h = h ^ (h >> 16u);
    h = h * 0x7FEB352Du;
    h = h ^ (h >> 15u);
    h = h * 0x846CA68Bu;
    h = h ^ (h >> 16u);
    return f32(h) / 4294967296.0;
}

// 2-D Worley F1 in cell units, clamped to [0, 1] (twin of clouds.rs worley2_f1).
fn gran_worley_f1(qx: f32, qy: f32, salt: u32) -> f32 {
    let bx = i32(floor(qx));
    let by = i32(floor(qy));
    var best = 1.0e30;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let cx = bx + dx;
            let cy = by + dy;
            let fx = f32(cx) + gran_cell_hash01(cx, cy, salt);
            let fy = f32(cy) + gran_cell_hash01(cx, cy, salt ^ 0x68E31DA4u);
            let d2 = (qx - fx) * (qx - fx) + (qy - fy) * (qy - fy);
            best = min(best, d2);
        }
    }
    return clamp(sqrt(best), 0.0, 1.0);
}

// Smooth 2-D value noise in [0, 1) (twin of clouds.rs gran_value_noise): cell-corner
// hashes blended with the smoothstep interpolant — the domain-warp input.
fn gran_value_noise(qx: f32, qy: f32, salt: u32) -> f32 {
    let bx = floor(qx);
    let by = floor(qy);
    let ix = i32(bx);
    let iy = i32(by);
    let tx = smoothstep(0.0, 1.0, qx - bx);
    let ty = smoothstep(0.0, 1.0, qy - by);
    let h00 = gran_cell_hash01(ix, iy, salt);
    let h10 = gran_cell_hash01(ix + 1, iy, salt);
    let h01 = gran_cell_hash01(ix, iy + 1, salt);
    let h11 = gran_cell_hash01(ix + 1, iy + 1, salt);
    return (h00 * (1.0 - tx) + h10 * tx) * (1.0 - ty) + (h01 * (1.0 - tx) + h11 * tx) * ty;
}

// The tail-shaped erosion field at a brick-plane position (m) — twin of
// clouds.rs granulation_erosion_noise: the position is DOMAIN-WARPED (round 2, twin
// of granulation_warp_offset — cell size/spacing varies across the scene), then the
// k^-5/3-weighted octaves are tail-shaped by the carve smoothstep.
fn gran_erosion_noise(u_m: f32, v_m: f32) -> f32 {
    let qx = u_m / GRAN_WARP_SCALE_M;
    let qy = v_m / GRAN_WARP_SCALE_M;
    let uw = u_m + GRAN_WARP_AMP_M * (2.0 * gran_value_noise(qx, qy, GRAN_WARP_SALT_U) - 1.0);
    let vw = v_m + GRAN_WARP_AMP_M * (2.0 * gran_value_noise(qx, qy, GRAN_WARP_SALT_V) - 1.0);
    var w = GRAN_W0 * gran_worley_f1(uw / GRAN_SCALE0_M, vw / GRAN_SCALE0_M, GRAN_SALT0);
    w = w + GRAN_W1 * gran_worley_f1(uw / GRAN_SCALE1_M, vw / GRAN_SCALE1_M, GRAN_SALT1);
    w = w + GRAN_W2 * gran_worley_f1(uw / GRAN_SCALE2_M, vw / GRAN_SCALE2_M, GRAN_SALT2);
    return smoothstep(GRAN_CARVE_LO, GRAN_CARVE_HI, clamp(w, 0.0, 1.0));
}

// Species/height gate (twin of clouds.rs granulation_gate): liquid share x a smooth
// boundary-layer height ramp.
fn gran_gate(ext_liquid: f32, ext_ice: f32, ext_precip: f32, z_msl_m: f32) -> f32 {
    let total = ext_liquid + ext_ice + ext_precip;
    if (total <= 0.0) {
        return 0.0;
    }
    let liquid_frac = clamp(ext_liquid / total, 0.0, 1.0);
    let height = 1.0 - smoothstep(GRAN_HEIGHT_FULL_M, GRAN_HEIGHT_ZERO_M, z_msl_m);
    return liquid_frac * height;
}

// Interior protection (twin of clouds.rs granulation_interior_protection): 1 at/
// below GRAN_INTERIOR_LO (true boundary), 0 at/above GRAN_INTERIOR_HI (deck interior).
fn gran_interior_protection(rel_density: f32) -> f32 {
    return 1.0 - smoothstep(GRAN_INTERIOR_LO, GRAN_INTERIOR_HI, clamp(rel_density, 0.0, 1.0));
}

// Remap-style erosion multiplier (twin of clouds.rs granulation_multiplier):
// m = (d - e)+ / (d (1 - e)); 1 at d = 1 (interiors untouched), 0 where d <= e.
fn gran_multiplier(rel_density: f32, erosion: f32) -> f32 {
    if (erosion <= 0.0) {
        return 1.0;
    }
    let d = clamp(rel_density, 0.0, 1.0);
    if (d <= 0.0) {
        return 1.0;
    }
    let e = min(erosion, GRAN_EROSION_MAX);
    return clamp(max(d - e, 0.0) / (d * (1.0 - e)), 0.0, 1.0);
}

// BIMODAL carve shape (round 2, twin of clouds.rs granulation_bimodal): gap-or-grain
// — at/below GAP the sample carves clear, at/above GRAIN it restores to full raw
// extinction (still subtract-only vs the raw sample), squeezing the grey middle.
fn gran_bimodal(m: f32) -> f32 {
    return smoothstep(GRAN_BIMODAL_GAP, GRAN_BIMODAL_GRAIN, clamp(m, 0.0, 1.0));
}

// DECK-COHERENCE gate (round 2, twin of clouds.rs GranCoherence::gate_at).
// ACTIVATION NOTE: the CPU builds this per-composite 2-D per-column field
// (GranCoherence::build — fill fraction + column-tau fsd over the ~7 dx window,
// distance-tapered dilation) and the GPU activation must upload it as an R8/R32
// texture sampled bilinearly at the column coords `b.xy`; until then this stub
// returns 1.0 (open) and the whole granulate() path is inert anyway behind
// GRAN_AMPLITUDE = 0.0.
fn gran_coherence_gate(bx: f32, by: f32) -> f32 {
    return 1.0;
}

// Apply the granulation erosion to a decoded volume sample at brick coords `b`
// (twin of clouds.rs DecodedVolume::sample_granulated). The CPU's relative-density
// neighbourhood is the 8 trilinear-support corners; here they are 8 unfiltered
// textureLoads decoded to total extinction. tau_up (s.w) is never eroded. With
// GRAN_AMPLITUDE = 0.0 this is a byte-identical no-op (the deferred activation
// wires the real amplitude; the anchor scale mirrors the CPU min(dx, dy) — for a
// MAP_PROJ 6 lat/lon grid the activation must convert degrees to metres).
fn granulate(s: vec4<f32>, b: vec3<f32>) -> vec4<f32> {
    if (GRAN_AMPLITUDE <= 0.0) {
        return s;
    }
    let total = s.x + s.y + s.z;
    if (total <= 0.0) {
        return s;
    }
    let z_msl = u.vert.x + b.z * u.vert.y;
    let gate = gran_gate(s.x, s.y, s.z, z_msl);
    if (gate <= 0.0) {
        return s;
    }
    // Deck-coherence gate (round 2; stub 1.0 until the activation uploads the field).
    let coh = gran_coherence_gate(b.x, b.y);
    if (coh <= 0.0) {
        return s;
    }
    let hi = vec3<i32>(u.dims.xyz) - vec3<i32>(1);
    let i0 = clamp(vec3<i32>(floor(b)), vec3<i32>(0), hi);
    let i1 = min(i0 + vec3<i32>(1), hi);
    var corner_max = 0.0;
    for (var ck: i32 = 0; ck <= 1; ck = ck + 1) {
        for (var cj: i32 = 0; cj <= 1; cj = cj + 1) {
            for (var ci: i32 = 0; ci <= 1; ci = ci + 1) {
                let p = vec3<i32>(
                    select(i0.x, i1.x, ci == 1),
                    select(i0.y, i1.y, cj == 1),
                    select(i0.z, i1.z, ck == 1),
                );
                let t = textureLoad(volume, p, 0);
                let tot = decode_channel(t.r, u.ql.x, u.ql.y)
                    + decode_channel(t.g, u.ql.z, u.ql.w)
                    + decode_channel(t.b, u.qp.x, u.qp.y);
                corner_max = max(corner_max, tot);
            }
        }
    }
    if (corner_max <= 0.0) {
        return s;
    }
    let d = total / corner_max;
    let protection = gran_interior_protection(d);
    if (protection <= 0.0) {
        return s;
    }
    let pitch = max(min(u.geo0.w, u.geo1.z), 1.0);
    let noise = gran_erosion_noise(b.x * pitch, b.y * pitch);
    let e = min(GRAN_AMPLITUDE / GRAN_AMP_CAP * gate * protection * coh * noise, GRAN_EROSION_MAX);
    if (e <= 0.0) {
        return s;
    }
    // The remap multiplier through the bimodal carve (round 2): gap-or-grain.
    let m = gran_bimodal(gran_multiplier(d, e));
    return vec4<f32>(s.x * m, s.y * m, s.z * m, s.w);
}

// Trilinear volume sample (hardware filtering on the codes, then decode — see the
// header note). Returns (ext_liquid, ext_ice, ext_precip, tau_up) in m^-1, through
// the (deferred, amplitude-0) granulation erosion so the M5-GPU activation samples
// the SAME eroded field everywhere this function is called (view + sun marches).
fn sample_volume(b: vec3<f32>) -> vec4<f32> {
    if (b.x < 0.0 || b.y < 0.0 || b.z < 0.0
        || b.x > u.dims.x - 1.0 || b.y > u.dims.y - 1.0 || b.z > u.dims.z - 1.0) {
        return vec4<f32>(0.0);
    }
    let uvw = vec3<f32>((b.x + 0.5) / u.dims.x, (b.y + 0.5) / u.dims.y, (b.z + 0.5) / u.dims.z);
    let t = textureSampleLevel(volume, samp, uvw, 0.0);
    let raw = vec4<f32>(
        decode_channel(t.r, u.ql.x, u.ql.y),
        decode_channel(t.g, u.ql.z, u.ql.w),
        decode_channel(t.b, u.qp.x, u.qp.y),
        decode_channel(t.a, u.qp.z, u.qp.w),
    );
    return granulate(raw, b);
}

// GUARD BAND (WS1, twin of clouds.rs OccupancyMip::maxext_at): a probe within one
// mip block outside the volume reads the clamped edge block (conservative — a coarse
// step must not jump over the entry into edge cloud); beyond it, empty.
const OCC_GUARD_CELLS: f32 = 8.0; // one occupancy-mip block (OCCUPANCY_MIP_FACTOR)

fn occupied(b: vec3<f32>) -> bool {
    let hi = vec3<f32>(u.dims.x - 1.0, u.dims.y - 1.0, u.dims.z - 1.0);
    if (b.x < -OCC_GUARD_CELLS || b.y < -OCC_GUARD_CELLS || b.z < -OCC_GUARD_CELLS
        || b.x > hi.x + OCC_GUARD_CELLS || b.y > hi.y + OCC_GUARD_CELLS
        || b.z > hi.z + OCC_GUARD_CELLS) {
        return false;
    }
    let bc = clamp(b, vec3<f32>(0.0), hi);
    let uvw = vec3<f32>((bc.x + 0.5) / u.dims.x, (bc.y + 0.5) / u.dims.y, (bc.z + 0.5) / u.dims.z);
    return textureSampleLevel(occupancy, samp, uvw, 0.0).r > 0.001;
}

fn henyey_greenstein(cos_t: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * cos_t, 1.5));
}

fn dual_hg(cos_t: f32, g1: f32, g2: f32, w: f32) -> f32 {
    return w * henyey_greenstein(cos_t, g1) + (1.0 - w) * henyey_greenstein(cos_t, g2);
}

fn aggregate_phase(cos_t: f32, ext_liquid: f32, ext_ice_precip: f32) -> f32 {
    let total = ext_liquid + ext_ice_precip;
    if (total <= 0.0) {
        return 1.0 / (4.0 * PI);
    }
    let liq = dual_hg(cos_t, PHASE_LIQUID_G1, PHASE_LIQUID_G2, PHASE_LIQUID_W);
    let ice = dual_hg(cos_t, PHASE_ICE_G1, PHASE_ICE_G2, PHASE_ICE_W);
    return (ext_liquid * liq + ext_ice_precip * ice) / total;
}

fn beer_powder(tau: f32) -> f32 {
    return exp(-tau) * (1.0 - exp(-2.0 * tau));
}

// Aggregate phase with the dual-HG eccentricities scaled by g_scale (octave phase).
fn aggregate_phase_scaled(cos_t: f32, ext_liquid: f32, ext_ice_precip: f32, g_scale: f32) -> f32 {
    let total = ext_liquid + ext_ice_precip;
    if (total <= 0.0) {
        return 1.0 / (4.0 * PI);
    }
    let liq = dual_hg(cos_t, PHASE_LIQUID_G1 * g_scale, PHASE_LIQUID_G2 * g_scale, PHASE_LIQUID_W);
    let ice = dual_hg(cos_t, PHASE_ICE_G1 * g_scale, PHASE_ICE_G2 * g_scale, PHASE_ICE_W);
    return (ext_liquid * liq + ext_ice_precip * ice) / total;
}

// Wrenninge/Oz multi-scatter octave SUN SOURCE (M5): sum_k weight_k * phase(g*b^k) *
// vis(tau_sun*a^k). Higher orders additionally carry `(1-exp(-support_tau))^k`, the
// probability of enough cloud interactions: they vanish in the optically-thin limit,
// while octave zero and thick-cloud behavior are unchanged. CPU twin:
// clouds.rs::octave_sun_source_thin_gated.
// The octave count comes from u.m1.y (the studio Multi-scatter A/B: DEFAULT_OCTAVES vs 1),
// clamped to [1, OCTAVES]; a zero/garbage uniform falls back to the full OCTAVES.
fn octave_sun_source(cos_t: f32, ext_liquid: f32, ext_ice_precip: f32, tau_sun: f32, powder: bool, support_tau: f32) -> f32 {
    var octaves = i32(u.m1.y + 0.5);
    if (octaves < 1 || octaves > OCTAVES) {
        octaves = OCTAVES;
    }
    var acc = 0.0;
    var ext_scale = 1.0;
    var g_scale = 1.0;
    var weight = 1.0;
    let thin_gate = 1.0 - exp(-max(support_tau, 0.0));
    var order_gate = 1.0;
    for (var k: i32 = 0; k < octaves; k = k + 1) {
        if (k > 0) {
            order_gate = order_gate * thin_gate;
        }
        let tau_k = tau_sun * ext_scale;
        var vis_k = exp(-tau_k);
        if (powder) {
            vis_k = beer_powder(tau_k);
        }
        let phase_k = aggregate_phase_scaled(cos_t, ext_liquid, ext_ice_precip, g_scale);
        acc = acc + order_gate * weight * phase_k * vis_k;
        ext_scale = ext_scale * OCTAVE_EXTINCTION_SCALE;
        g_scale = g_scale * OCTAVE_PHASE_SCALE;
        weight = weight * OCTAVE_BRIGHTNESS_SCALE;
    }
    return acc;
}

// ── SH-2 directional sky ambient (M5) — twin of atmosphere.rs SkyShAmbient ──
// Interpolate SH coefficient k (RGB) at a sun elevation (deg) from the sh_ambient
// texture (row = elevation entry, col = coefficient). Reuses u.p2 = (elev_min, elev_max, n).
fn sh_coef(k: i32, elev_deg: f32) -> vec3<f32> {
    let cnt = i32(u.p2.z);
    let t = clamp((elev_deg - u.p2.x) / (u.p2.y - u.p2.x), 0.0, 1.0);
    let f = t * f32(cnt - 1);
    let i0 = min(i32(floor(f)), cnt - 1);
    let i1 = min(i0 + 1, cnt - 1);
    let w = f - floor(f);
    let a = textureLoad(sh_ambient, vec2<i32>(k, i0), 0).rgb;
    let b = textureLoad(sh_ambient, vec2<i32>(k, i1), 0).rgb;
    return mix(a, b, w);
}

// Diffuse irradiance from the SH-2 sky at a receiver normal (sun-relative frame), via
// the Ramamoorthi cosine-lobe convolution (A0=pi, A1=2pi/3, A2=pi/4).
fn sh_irradiance(elev_deg: f32, nrm: vec3<f32>) -> vec3<f32> {
    let n = normalize(nrm);
    let a1 = 2.0 * PI / 3.0;
    let a2 = PI / 4.0;
    var e = vec3<f32>(0.0);
    e = e + sh_coef(0, elev_deg) * (PI * 0.282094791773878);
    e = e + sh_coef(1, elev_deg) * (a1 * 0.488602511902920 * n.y);
    e = e + sh_coef(2, elev_deg) * (a1 * 0.488602511902920 * n.z);
    e = e + sh_coef(3, elev_deg) * (a1 * 0.488602511902920 * n.x);
    e = e + sh_coef(4, elev_deg) * (a2 * 1.092548430592079 * n.x * n.y);
    e = e + sh_coef(5, elev_deg) * (a2 * 1.092548430592079 * n.y * n.z);
    e = e + sh_coef(6, elev_deg) * (a2 * 0.315391565252520 * (3.0 * n.z * n.z - 1.0));
    e = e + sh_coef(7, elev_deg) * (a2 * 1.092548430592079 * n.x * n.z);
    e = e + sh_coef(8, elev_deg) * (a2 * 0.546274215296040 * (n.x * n.x - n.y * n.y));
    return max(e, vec3<f32>(0.0));
}

// A receiver normal expressed in the sun-relative frame (z=up, x=sun horizontal).
fn sun_frame_normal(up: vec3<f32>, sun: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let z = normalize(up);
    let sun_h = sun - z * dot(sun, z);
    var xx: vec3<f32>;
    if (length(sun_h) > 1e-6) {
        xx = normalize(sun_h);
    } else {
        var seed = vec3<f32>(0.0, 0.0, 1.0);
        if (abs(z.z) >= 0.9) {
            seed = vec3<f32>(1.0, 0.0, 0.0);
        }
        xx = normalize(seed - z * dot(seed, z));
    }
    let yy = cross(z, xx);
    return vec3<f32>(dot(normal, xx), dot(normal, yy), dot(normal, z));
}

// Bilinear sun-OD map fetch at an ECEF point (total column optical depth).
// OUT-OF-EXTENT CONTRACT (WS1, twin of clouds.rs SunOdMap::sample_uv): outside the
// map extent (half-texel tolerance) there is no cloud column — return 0, never the
// clamped edge texel (which smeared a domain-edge shadow across the zoom-out margin).
fn sun_od_sample(p: vec3<f32>) -> f32 {
    let d = p - u.sod_c.xyz;
    let uu = dot(d, u.sod_u.xyz);
    let vv = dot(d, u.sod_v.xyz);
    let u_min = u.sod_u.w;
    let u_max = u.sod_v.w;
    let v_min = u.sod_e.x;
    let v_max = u.sod_e.y;
    let dim = i32(u.sod_e.z);
    if (u_max <= u_min || v_max <= v_min || dim <= 0) {
        return 0.0;
    }
    let tol_u = 0.5 * (u_max - u_min) / f32(dim);
    let tol_v = 0.5 * (v_max - v_min) / f32(dim);
    if (uu < u_min - tol_u || uu > u_max + tol_u || vv < v_min - tol_v || vv > v_max + tol_v) {
        return 0.0;
    }
    let fu = clamp((uu - u_min) / (u_max - u_min) * f32(dim) - 0.5, 0.0, f32(dim - 1));
    let fv = clamp((vv - v_min) / (v_max - v_min) * f32(dim) - 0.5, 0.0, f32(dim - 1));
    let x0 = i32(floor(fu));
    let y0 = i32(floor(fv));
    let x1 = min(x0 + 1, dim - 1);
    let y1 = min(y0 + 1, dim - 1);
    let a = mix(textureLoad(sun_od, vec2<i32>(x0, y0), 0).r, textureLoad(sun_od, vec2<i32>(x1, y0), 0).r, fu - floor(fu));
    let b = mix(textureLoad(sun_od, vec2<i32>(x0, y1), 0).r, textureLoad(sun_od, vec2<i32>(x1, y1), 0).r, fu - floor(fu));
    return mix(a, b, fv - floor(fv));
}

// Bilinear occluder-distance fetch (m) at an ECEF point (the M5 penumbra channel).
// Same WS1 out-of-extent contract as sun_od_sample (both channels read 0 outside).
fn sun_od_dist_sample(p: vec3<f32>) -> f32 {
    let d = p - u.sod_c.xyz;
    let uu = dot(d, u.sod_u.xyz);
    let vv = dot(d, u.sod_v.xyz);
    let u_min = u.sod_u.w;
    let u_max = u.sod_v.w;
    let v_min = u.sod_e.x;
    let v_max = u.sod_e.y;
    let dim = i32(u.sod_e.z);
    if (u_max <= u_min || v_max <= v_min || dim <= 0) {
        return 0.0;
    }
    let tol_u = 0.5 * (u_max - u_min) / f32(dim);
    let tol_v = 0.5 * (v_max - v_min) / f32(dim);
    if (uu < u_min - tol_u || uu > u_max + tol_u || vv < v_min - tol_v || vv > v_max + tol_v) {
        return 0.0;
    }
    let fu = clamp((uu - u_min) / (u_max - u_min) * f32(dim) - 0.5, 0.0, f32(dim - 1));
    let fv = clamp((vv - v_min) / (v_max - v_min) * f32(dim) - 0.5, 0.0, f32(dim - 1));
    let x0 = i32(floor(fu));
    let y0 = i32(floor(fv));
    let x1 = min(x0 + 1, dim - 1);
    let y1 = min(y0 + 1, dim - 1);
    let a = mix(textureLoad(sun_od_dist, vec2<i32>(x0, y0), 0).r, textureLoad(sun_od_dist, vec2<i32>(x1, y0), 0).r, fu - floor(fu));
    let b = mix(textureLoad(sun_od_dist, vec2<i32>(x0, y1), 0).r, textureLoad(sun_od_dist, vec2<i32>(x1, y1), 0).r, fu - floor(fu));
    return mix(a, b, fv - floor(fv));
}

// Top-down ground-shadow antialias: the exact CPU 5x5 binomial kernel in
// TRANSMITTANCE space. Disabled is the historical single Beer sample exactly.
fn filtered_ground_shadow(pg: vec3<f32>) -> f32 {
    let od_scale = cloud_od_scale();
    if (u.ray0.z < 0.5) {
        return exp(-od_scale * sun_od_sample(pg));
    }
    let weights = array<f32, 5>(1.0, 4.0, 6.0, 4.0, 1.0);
    let dim = max(u.sod_e.z, 1.0);
    let texel_u = (u.sod_v.w - u.sod_u.w) / dim;
    let texel_v = (u.sod_e.y - u.sod_e.x) / dim;
    var sum = 0.0;
    for (var yi: i32 = 0; yi < 5; yi = yi + 1) {
        for (var xi: i32 = 0; xi < 5; xi = xi + 1) {
            let off = u.sod_u.xyz * (f32(xi - 2) * texel_u)
                + u.sod_v.xyz * (f32(yi - 2) * texel_v);
            sum = sum + weights[xi] * weights[yi]
                * exp(-od_scale * sun_od_sample(pg + off));
        }
    }
    return sum * (1.0 / 256.0);
}

// PENUMBRAL ground cloud-shadow (M5) — twin of clouds.rs::SunOdMap::penumbral_shadow.
// Blur radius = occluder distance x tan(0.2665 deg) in the sun-OD map's (au, av) plane;
// transmittance-averaged over a small disk (soft, distance-widening edge).
fn penumbral_shadow(pg: vec3<f32>) -> f32 {
    let occ_dist = sun_od_dist_sample(pg);
    let radius = occ_dist * SUN_ANG_RADIUS_TAN;
    let dim = max(u.sod_e.z, 1.0);
    let texel = max((u.sod_v.w - u.sod_u.w) / dim, (u.sod_e.y - u.sod_e.x) / dim);
    if (radius <= 0.5 * texel) {
        return filtered_ground_shadow(pg);
    }
    let au = u.sod_u.xyz;
    let av = u.sod_v.xyz;
    // A physically resolved solar-disk penumbra already has its historical 17-tap
    // integration; do not nest the 25-tap footprint inside every disk tap.
    let od_scale = cloud_od_scale();
    var sum = exp(-od_scale * sun_od_sample(pg));
    var wsum = 1.0;
    for (var ri: i32 = 0; ri < 2; ri = ri + 1) {
        let rr = select(1.0, 0.5, ri == 0);
        let w = select(0.6, 1.0, ri == 0);
        for (var kk: i32 = 0; kk < 8; kk = kk + 1) {
            let ang = (f32(kk) + 0.5) / 8.0 * 2.0 * PI;
            let off = au * (radius * rr * cos(ang)) + av * (radius * rr * sin(ang));
            sum = sum + w * exp(-od_scale * sun_od_sample(pg + off));
            wsum = wsum + w;
        }
    }
    return sum / wsum;
}

// Deterministic hash of an ECEF position -> [0, 1) (twin of clouds.rs::
// hash01_position; the stratified sun-march jitter seed). f32 rounding of the huge
// ECEF coordinates may differ from the CPU's f64 in the low bit — a documented
// divergence (the jitter is decorrelation, not physics; bit parity not required).
fn hash01(p: vec3<f32>) -> f32 {
    var h = bitcast<u32>(i32(round(p.x))) * 0x9E3779B9u
        + bitcast<u32>(i32(round(p.y))) * 0x85EBCA6Bu
        + bitcast<u32>(i32(round(p.z))) * 0xC2B2AE35u;
    h = h ^ (h >> 16u);
    h = h * 0x7FEB352Du;
    h = h ^ (h >> 15u);
    h = h * 0x846CA68Bu;
    h = h ^ (h >> 16u);
    return f32(h) / 4294967296.0;
}

// Depth-resolved cloud sun optical depth: a short secondary light march toward the sun
// FROM the sample (the cloud between the sample and the sun), exponentially-spaced so
// the near field that dominates the sunlit face is resolved and the far tail is cheap.
// Mirrors clouds.rs::cloud_sun_optical_depth (M4 review FINDING 1). The sun-OD map is no
// longer consulted here (a 2-D total-column scalar cannot give a per-depth partial);
// outside this function its total is used only for ground shadow and multiscatter
// support, never for the sample-to-sun Beer term.
// WS1: each ray samples its segments at a deterministic stratified hash offset instead
// of the fixed midpoint, and a two-sample TAIL covers the remaining in-shell slant
// toward the sun past the schedule's natural reach (a distant occluder along a low sun
// ray must still shadow; the near field keeps the exact unextended schedule).
fn cloud_sun_optical_depth(p: vec3<f32>, sun: vec3<f32>) -> f32 {
    let base = max(u.dims.w, 1.0); // base step = voxel pitch
    let offset = 0.5 + SUN_MARCH_JITTER * (hash01(p) - 0.5);
    var tau = 0.0;
    var dist = 0.0;
    var ds = base;
    for (var k: i32 = 0; k < SUN_MARCH_STEPS; k = k + 1) {
        let pp = p + sun * (dist + offset * ds);
        let s = sample_volume(ecef_to_brick(pp));
        tau = tau + (s.x + s.y + s.z) * ds;
        dist = dist + ds;
        ds = ds * SUN_MARCH_GROWTH;
    }
    let top = ray_sphere(p, sun, u.vert.z);
    if (top.y >= top.x && top.y > 0.0) {
        var t_exit = top.y;
        let bot = ray_sphere(p, sun, u.vert.w);
        if (bot.y >= bot.x && bot.x > 0.0 && bot.x < t_exit) {
            t_exit = bot.x;
        }
        if (t_exit > dist) {
            let half = 0.5 * (t_exit - dist);
            for (var m: i32 = 0; m < 2; m = m + 1) {
                let pp = p + sun * (dist + offset * half);
                let s = sample_volume(ecef_to_brick(pp));
                tau = tau + (s.x + s.y + s.z) * half;
                dist = dist + half;
            }
        }
    }
    // Keep the compute-generated sun-OD texture and uploaded volume raw. Scale only
    // this visible-light consumer; derived COD and thermal IR remain physical.
    return tau * cloud_od_scale();
}

// The traversal fraction of the ATMOSPHERE shell (entry -> ground / far exit) at an
// absolute distance t along the view ray — the froxel's depth coordinate. Twin of
// clouds.rs::atmosphere_shell_fraction (M4 review FINDING 4).
fn atmosphere_shell_fraction(cam: vec3<f32>, view: vec3<f32>, t: f32) -> f32 {
    let top = ray_sphere(cam, view, R_TOP);
    if (top.y < top.x || top.y <= 0.0) {
        return 1.0;
    }
    let t_enter = max(top.x, 0.0);
    var t_exit = top.y;
    let gnd = ray_sphere(cam, view, R_GROUND);
    if (gnd.y >= gnd.x && gnd.x > t_enter && gnd.x < t_exit) {
        t_exit = gnd.x;
    }
    if (t_exit <= t_enter) {
        return 1.0;
    }
    return clamp((t - t_enter) / (t_exit - t_enter), 0.0, 1.0);
}

// Direct top-down camera->cloud atmosphere, the GPU twin of
// topdown.rs::topdown_front_column. The RGB transmittance mean matches the scalar
// alpha stored by the GEO froxel. A cloud at atmosphere entry has the exact neutral
// front column (zero airlight, unit transmittance).
fn topdown_front_column(cam: vec3<f32>, view: vec3<f32>, sun: vec3<f32>, cloud_t: f32) -> vec4<f32> {
    let top = ray_sphere(cam, view, R_TOP);
    if (top.y < top.x || top.y <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let t_enter = max(top.x, 0.0);
    var t_exit = top.y;
    let gnd = ray_sphere(cam, view, R_GROUND);
    if (gnd.y >= gnd.x && gnd.x > t_enter && gnd.x < t_exit) {
        t_exit = gnd.x;
    }
    let segment_m = max(0.0, clamp(cloud_t, t_enter, t_exit) - t_enter);
    if (segment_m <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let sc = raymarch(cam + view * t_enter, view, sun, segment_m);
    let t_ac = dot(sc.transmittance, vec3<f32>(1.0 / 3.0));
    return vec4<f32>(sc.inscatter, clamp(t_ac, 0.0, 1.0));
}

// Complete front-column algebra shared with the CPU top-down path. i_ac is already
// passed through the product-facing aerial-veil scale at the call site.
fn composite_front_column(
    l_toa: vec3<f32>,
    l_cloud: vec3<f32>,
    t_cloud: f32,
    i_ac: vec3<f32>,
    t_ac: f32,
) -> vec3<f32> {
    return l_toa * t_cloud
        + t_ac * l_cloud
        + i_ac * (1.0 - t_cloud);
}

fn froxel_at(scan_x: f32, scan_y: f32, w: f32) -> vec4<f32> {
    let dim = i32(u.frx2.x);
    if (dim <= 0 || u.frx.y <= u.frx.x || u.frx.w <= u.frx.z) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let fu = clamp((scan_x - u.frx.x) / (u.frx.y - u.frx.x), 0.0, 1.0);
    let fv = clamp((scan_y - u.frx.z) / (u.frx.w - u.frx.z), 0.0, 1.0);
    let x = min(i32(fu * f32(dim)), dim - 1);
    let y = min(i32(fv * f32(dim)), dim - 1);
    let z = min(i32(clamp(w, 0.0, 1.0) * f32(dim)), dim - 1);
    return textureLoad(froxel, vec3<i32>(x, y, z), 0);
}

struct CloudResult {
    inscatter: vec3<f32>,
    transmittance: f32,
    mean_w: f32,
    mean_t: f32,
};

// March the cloud volume along the view ray (twin of clouds.rs::march_cloud).
fn march_cloud(cam: vec3<f32>, view: vec3<f32>, sun: vec3<f32>) -> CloudResult {
    let r_top = u.vert.z;
    let r_bot = u.vert.w;
    let top = ray_sphere(cam, view, r_top);
    if (top.y < top.x || top.y <= 0.0) {
        return CloudResult(vec3<f32>(0.0), 1.0, 1.0, 0.0);
    }
    let t_enter = max(top.x, 0.0);
    var t_exit = top.y;
    let bot = ray_sphere(cam, view, r_bot);
    if (bot.y >= bot.x && bot.x > t_enter && bot.x < t_exit) {
        t_exit = bot.x;
    }
    let seg = t_exit - t_enter;
    if (seg <= 0.0) {
        return CloudResult(vec3<f32>(0.0), 1.0, 1.0, 0.0);
    }
    let coarse = u.m0.x;
    let fine = u.m0.y;
    let max_steps = i32(u.m0.z);
    let floor_t = u.sod_c.w;
    let e_sun = u.solar.xyz;
    let cos_vs = dot(view, sun);
    let od_scale = cloud_od_scale();

    var t = t_enter;
    var trans = 1.0;
    var l = vec3<f32>(0.0);
    var w_accum = 0.0;
    var w_weight = 0.0;
    for (var step: i32 = 0; step < max_steps; step = step + 1) {
        if (t >= t_exit || trans <= floor_t) {
            break;
        }
        let p = cam + view * t;
        let b = ecef_to_brick(p);
        let is_occ = occupied(b);
        // Clamp EVERY step to the shell exit and sample the segment MIDPOINT (WS1,
        // the march_ir pattern): no extinction is integrated past the exit (below
        // the ground / outside the shell), and the left-endpoint bias is removed.
        var ds = coarse;
        if (is_occ) {
            ds = fine;
        }
        if (t + ds > t_exit) {
            ds = t_exit - t;
        }
        if (ds <= 0.0) {
            break;
        }
        if (!is_occ) {
            t = t + ds;
            continue;
        }
        let pm = cam + view * (t + 0.5 * ds);
        let bm = ecef_to_brick(pm);
        let s = sample_volume(bm);
        let sigma_t = s.x + s.y + s.z;
        if (sigma_t <= 0.0) {
            t = t + ds;
            continue;
        }
        // Zoom-out-margin EDGE FEATHER (twin of the CPU march): scales BOTH the
        // in-scatter source and the step opacity, so a faded sample scatters less
        // AND grows more transparent. sigma_eff == sigma_t at band 0 and OD scale 1.
        let sigma_eff = sigma_t * edge_feather(bm.x, bm.y) * od_scale;
        if (sigma_eff <= 0.0) {
            t = t + ds;
            continue;
        }
        // Sun source: Wrenninge multi-scatter octaves (M5) over the single depth-
        // resolved cloud sun optical depth (octaves=1 == fix2 single scatter).
        let tau_cloud_sun = cloud_sun_optical_depth(pm, sun);
        // Smooth vertical column OD preserves the higher-order buildup at a thick
        // cloud's sunlit top. The coarse sun-aligned map is primarily a ground-shadow
        // raster; using max(real_column, map) imprinted its texel lattice on HRRR cloud
        // light. It remains a support-only fallback for legacy/analytic zero-tau_up
        // data and never replaces sample-to-sun tau.
        let col_total = sample_volume(vec3<f32>(bm.x, bm.y, 0.0)).w;
        var column_support_tau = sun_od_sample(pm) * od_scale;
        if (col_total > 0.0) {
            column_support_tau = col_total * od_scale;
        }
        let support_tau = max(
            max(column_support_tau, tau_cloud_sun),
            sigma_eff * u.dims.w,
        );
        let sun_src = octave_sun_source(
            cos_vs,
            s.x,
            s.y + s.z,
            tau_cloud_sun,
            u.m1.z > 0.5,
            support_tau,
        );
        let r = length(pm);
        let up = pm / r;
        let mu_sun = dot(up, sun);
        // FINITE-DISK EARTH-SHADOW FADE (WS1, twin of clouds.rs::
        // sun_horizon_disk_fraction): the solar-disk fraction above the sample's
        // local geometric horizon replaces the binary ray_hits_ground gate (the hard
        // lit/unlit line across dusk anvils); the transmittance sample clamps mu to
        // the horizon so the fading disk is attenuated by the defined grazing path.
        let rr = clamp(R_GROUND / r, -1.0, 1.0);
        let dip = acos(rr);
        let disk_sun = disk_fraction(asin(clamp(mu_sun, -1.0, 1.0)) + dip);
        var t_atmo = vec3<f32>(0.0);
        if (disk_sun > 0.0) {
            let mu_h = -sqrt(max(1.0 - rr * rr, 0.0));
            t_atmo = sample_transmittance(r, max(mu_sun, mu_h)) * disk_sun;
        }
        let sun_elev = degrees(asin(clamp(mu_sun, -1.0, 1.0)));
        // SH-2 directional sky ambient (M5): sky irradiance at the voxel's local up.
        let e_sky = sh_irradiance(sun_elev, sun_frame_normal(up, sun, up));
        let tau_down = max(col_total - s.w, 0.0);
        let amb_factor = AMBIENT_W_ABOVE * exp(-s.w * od_scale)
            + AMBIENT_W_BELOW * u.m1.w * exp(-tau_down * od_scale);
        // The FEATHERED extinction drives the step opacity + the local in-scatter
        // source (sun/ambient use the edge-unfeathered but OD-scaled field — CPU twin).
        let step_t = exp(-sigma_eff * ds);
        let s_sun = e_sun * (sigma_eff * sun_src) * t_atmo;
        let s_amb = e_sky * (sigma_eff * amb_factor / PI);
        let src = s_sun + s_amb;
        l = l + trans * (src - src * step_t) / sigma_eff;
        let contribution = trans * (1.0 - step_t);
        w_accum = w_accum + contribution * (t + 0.5 * ds - t_enter) / seg;
        w_weight = w_weight + contribution;
        trans = trans * step_t;
        t = t + ds;
    }
    var mean_w = 1.0;
    if (w_weight > 0.0) {
        mean_w = clamp(w_accum / w_weight, 0.0, 1.0);
    }
    let mean_t = t_enter + mean_w * seg;
    return CloudResult(l, trans, mean_w, mean_t);
}

// M2 surface/limb radiance (linear TOA), with a cloud-shadow factor on the direct
// term. Returns rgb radiance; caller decides transparency for space.
fn surface_radiance(coord: vec2<i32>, cam: vec3<f32>, view: vec3<f32>, cloud_shadow: f32) -> vec3<f32> {
    let sun_ecef = u.sun.xyz;
    let e_sun = u.solar.xyz;
    let g = textureLoad(lut_geo, coord, 0);
    if (g.x < 0.0) {
        // Limb (space is handled by the caller): inscatter of the grazing shell.
        let top = ray_sphere(cam, view, R_TOP);
        let t_enter = max(top.x, 0.0);
        var t_exit = top.y;
        let gnd = ray_sphere(cam, view, R_GROUND);
        if (gnd.y >= gnd.x && gnd.x > t_enter && gnd.x < t_exit) {
            t_exit = gnd.x;
        }
        let sc = raymarch(cam + view * t_enter, view, sun_ecef, max(0.0, t_exit - t_enter));
        return sc.inscatter;
    }
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
        is_water = textureSampleLevel(landmask_tex, samp, uv, 0.0).r < 0.5;
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
    // M3: penumbral terrain cast shadow folded into the disk (horizon = 0 here — the
    // per-texel horizon map upload is deferred with this GPU cloud pass).
    let disk = terrain_shadow_fraction(sun_elev * DEG2RAD, 0.0);
    let mu_sun = sin(max(sun_elev, 0.0) * DEG2RAD);
    let t_sun = sample_transmittance(R_GROUND + 1.0, mu_sun);
    // Raw sun-OD cloud shadow (the specular glint consumer) and the WS2 FLOORED
    // effective shadow the diffuse direct terms see (twin of render.rs).
    let shadow_raw = clamp(cloud_shadow, 0.0, 1.0);
    let shadow = effective_cloud_shadow_gpu(shadow_raw, sun_elev);
    // SH-2 directional terrain ambient (M5): sky irradiance at the terrain normal in the
    // sun-relative frame (full upper hemisphere; the M3 aperture openness/bent-normal
    // upload is deferred with this GPU cloud pass — twin of render.rs surface_toa).
    let e_ambient = sh_irradiance(sun_elev, sun_frame_normal(vec3<f32>(0.0, 0.0, 1.0), sun_enu, normal));
    var l_surf: vec3<f32>;
    if (is_water) {
        // M3 Cox-Munk sun glint + Fresnel sky reflection (calm-sea wind = 0; per-pixel
        // U10/V10 upload deferred). Sky reflection uses e_ambient/PI as a gray sky.
        let gnd_w = ray_sphere(cam, view, R_GROUND);
        let up_e = normalize(cam + view * max(gnd_w.x, 0.0));
        let to_cam = -view;
        // GLINT_MSS_SCALE narrows the Cox-Munk core (round 2); GLINT_STRENGTH lifts the peak.
        let glint = cox_munk_glint(sun_ecef, to_cam, up_e, cox_munk_mss(0.0) * GLINT_MSS_SCALE) * GLINT_STRENGTH;
        let cos_view = max(dot(to_cam, up_e), 0.0);
        let f_sky = fresnel_unpolarized(cos_view, WATER_N);
        // The specular glint sees the RAW shadow (twin of render.rs — the WS2 floor
        // models diffuse cloud-scattered fill, which has no specular component).
        // e_sun is disk-integrated; disk is the finite-disk visibility fraction.
        let l_glint = e_sun * (glint / PI) * t_sun * (disk * shadow_raw);
        // WS2 water direct sun (twin of render.rs): the water BODY sees the same
        // physical disk/Tsun/N.L, cloud-shadow-weighted direct term as land. Only the
        // water albedo is retuned toward WATER_ALBEDO_DAY_SCALE across the 0--12 degree
        // surface-help ramp. The glint keeps the RAW shadow above.
        let surface_t = smoothstep(SURFACE_HELP_ELEV_LO, SURFACE_HELP_ELEV_HI, sun_elev);
        var scale_ratio = 1.0;
        if (u.p1.y > 0.0) {
            scale_ratio = 1.0 + surface_t * (WATER_ALBEDO_DAY_SCALE / u.p1.y - 1.0);
        }
        let e_direct_w = e_sun * t_sun * (disk * ndotl * shadow);
        l_surf = albedo * scale_ratio / PI * (e_direct_w + e_ambient) + l_glint + f_sky * (e_ambient / PI);
    } else {
        let e_direct = e_sun * t_sun * (disk * ndotl * shadow);
        l_surf = albedo / PI * (e_direct + e_ambient);
        // Finished-visible appearance controls are LAND-only and precede the legacy
        // land gain/ground lift/aerial veil, matching the CPU composite order.
        l_surf = l_surf * land_appearance_gain_gpu(sun_elev, albedo);
        // LAND daylight brightness lift (round 2): ground-only surface-reflectance gain,
        // neutral at/below the horizon. Applied before the aerial veil below.
        l_surf = l_surf * land_day_gain(sun_elev);
    }
    // GROUND LIFT (twin of render::surface_toa_radiance): the sun-gated daylight
    // brightness lift on the WHOLE surface radiance (land AND water), applied AFTER
    // the land gain and BEFORE the aerial veil — exactly the CPU order.
    l_surf = l_surf * ground_day_lift_gain(sun_elev);

    let top = ray_sphere(cam, view, R_TOP);
    let gnd = ray_sphere(cam, view, R_GROUND);
    let t_enter = max(top.x, 0.0);
    var t_ground = top.y;
    if (gnd.y >= gnd.x && gnd.x > t_enter && gnd.x < t_ground) {
        t_ground = gnd.x;
    }
    var l_toa = l_surf;
    if (t_ground > t_enter) {
        let sc = raymarch(cam + view * t_enter, view, sun_ecef, t_ground - t_enter);
        // SUNRISE veil ramp (low-sun visible pass): the terminator band keeps the full
        // physical veil, daytime keeps the refinement de-haze.
        let veil = select(1.0, aerial_veil_scale(sun_elev), u.p2.w > 0.5);
        if (!is_water && (u.toe0.x > 0.5 || u.twi0.x > 0.5)) {
            let surface = l_surf * sc.transmittance;
            l_toa = surface * combined_surface_recovery_gain_gpu(surface, sun_elev)
                + veil * sc.inscatter;
        } else {
            l_toa = l_surf * sc.transmittance + veil * sc.inscatter;
        }
    } else if (!is_water && (u.toe0.x > 0.5 || u.twi0.x > 0.5)) {
        l_toa = l_surf * combined_surface_recovery_gain_gpu(l_surf, sun_elev);
    }
    return l_toa;
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(i32(pos.x), i32(pos.y));
    let g = textureLoad(lut_geo, coord, 0);
    let sun_ecef = u.sun.xyz;
    let e_sun = u.solar.xyz;
    let ray = camera_ray(coord);
    if (ray.valid < 0.5) {
        return vec4<f32>(0.0);
    }
    let cam = ray.cam;
    let view = ray.view;

    if (g.x < 0.0) {
        // Off-earth: limb if the ray grazes the shell, else space (transparent).
        let top = ray_sphere(cam, view, R_TOP);
        if (top.y < top.x || top.y <= 0.0) {
            return vec4<f32>(0.0, 0.0, 0.0, 0.0);
        }
        let l_limb = surface_radiance(coord, cam, view, 1.0);
        let rho = exposure_gain() * PI * l_limb / max(e_sun, vec3<f32>(1e-6));
        return vec4<f32>(output_transform(rho), 1.0);
    }

    // On-earth: PENUMBRAL cloud shadow on the ground (M5 — distance-widening soft edge).
    var shadow = 1.0;
    let gnd = ray_sphere(cam, view, u.vert.w);
    if (gnd.y >= gnd.x && gnd.x > 0.0) {
        shadow = penumbral_shadow(cam + view * gnd.x);
    }
    let l_toa = surface_radiance(coord, cam, view, shadow);
    // Low-sun illuminant correction at the display seam (on-earth pixels only; the
    // limb above keeps its physical color) — twin of the CPU shade_cloud_pixel /
    // shade_surface seam. Identity outside the 2-30 deg band.
    let pixel_sun_elev = textureLoad(lut_light, coord, 0).w;
    let illum = low_sun_illuminant_gains(pixel_sun_elev);

    if (u.sod_e.w < 0.5) {
        // Clouds disabled: the M2 surface unchanged.
        let rho = exposure_gain() * illum * PI * l_toa / max(e_sun, vec3<f32>(1e-6));
        return vec4<f32>(output_transform(rho), 1.0);
    }

    let m = march_cloud(cam, view, sun_ecef);
    // GEO uses its scan-space aerial froxel. Top-down has one local camera per pixel,
    // so it directly marches the physical front column to the cloud optical centroid.
    var ap: vec4<f32>;
    if (u.ray0.x < 0.5) {
        let scan_x = u.ex.w + f32(coord.x) * u.ez.w;
        let scan_y = u.ey.w - f32(coord.y) * u.solar.w;
        // Froxel depth = the atmosphere-shell fraction of the cloud centroid (FINDING 4).
        let w_froxel = atmosphere_shell_fraction(cam, view, m.mean_t);
        ap = froxel_at(scan_x, scan_y, w_froxel);
    } else {
        ap = topdown_front_column(cam, view, sun_ecef, m.mean_t);
    }
    // Front airlight weighted by (1 - T_cloud) to avoid double-counting the front
    // segment already inside l_toa's full-column airlight (FINDING 4).
    // Match the product-facing surface atmospheric correction for froxel airlight in
    // FRONT of cloud. p2.w is the shared SurfaceUniforms atmosphere-correction flag;
    // raw-physics mode leaves the full froxel veil intact.
    let front_veil = select(1.0, aerial_veil_scale(pixel_sun_elev), u.p2.w > 0.5);
    let l_final = composite_front_column(
        l_toa,
        m.inscatter,
        m.transmittance,
        front_veil * ap.rgb,
        ap.a,
    );
    // Display seam (twin of shade_cloud_pixel -> radiance_to_rgba_softclip): the
    // low-sun illuminant gains, then rho = EXPOSURE * pi * L / E_sun into the
    // exposure-aware output transform.
    let rho = exposure_gain() * illum * PI * l_final / max(e_sun, vec3<f32>(1e-6));
    return vec4<f32>(output_transform(rho), 1.0);
}
