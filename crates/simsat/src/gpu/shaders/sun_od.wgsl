// SimSat sun optical-depth compute pass (design doc section 6, M4). GPU twin of the
// CPU reference `clouds::accumulate_sun_od`. Accumulates, per texel of a sun-aligned
// orthographic map, the TOTAL optical depth of the brick column along the sun ray
// (Texture A = the log-quantized extinction volume). Consumers: cloud shadows on the
// ground (T = e^-od) and the raymarch long-range sun transmittance.
//
// M4 ships the CPU accumulation (`clouds::accumulate_sun_od`) for tested correctness
// on the headless nodes; this compute shader is the naga-validated GPU-acceleration
// path activated in M5. One invocation per texel; no atomics, no shared memory.

const R_GROUND: f32 = 6370000.0;

struct SunOdUniforms {
    // Sun-aligned orthographic frame (ECEF, metres). center = brick centre.
    center: vec4<f32>,   // xyz, w unused
    au: vec4<f32>,       // xyz axis-u (perp to sun), w u_min
    av: vec4<f32>,       // xyz axis-v (perp to sun), w u_max
    sun: vec4<f32>,      // xyz unit sun dir, w v_min
    extent: vec4<f32>,   // v_max, s_start, s_len, n_steps
    dims: vec4<f32>,     // nx, ny, nz, map_dim
    vert: vec4<f32>,     // z_min, dz, ds, unused
    // WRF projection forward (lat/lon -> i,j): see frame.rs::MapProjection.
    geo0: vec4<f32>,     // proj_kind, ref_i, ref_j, dx
    geo1: vec4<f32>,     // ref_u, ref_v, dy, central_meridian_deg
    geo2: vec4<f32>,     // lambert_n, lambert_f, ps_k, merc_scale
    geo3: vec4<f32>,     // south_pole, unused, unused, unused
    // LogQuant scales for the three extinction channels (code 0 = 0).
    ql: vec4<f32>,       // ext_liquid vmin, vmax ; ext_ice vmin, vmax
    qp: vec4<f32>,       // ext_precip vmin, vmax ; unused, unused
};

@group(0) @binding(0) var<uniform> u: SunOdUniforms;
@group(0) @binding(1) var volume: texture_3d<f32>;
@group(0) @binding(2) var out_od: texture_storage_2d<r32float, write>;

const PI: f32 = 3.14159265358979;
const DEG2RAD: f32 = 0.017453292519943295;

// EDGE FEATHER (WS1 march-physics pass; twin of clouds.rs SUN_OD_EDGE_FEATHER_TEXELS
// + sun_od_edge_weight): the accumulated od ramps to zero over the outermost texels
// so the ground-shadow field is continuous across the map boundary, outside which
// the consumers (clouds.wgsl sun_od_sample) now read 0.
const EDGE_FEATHER_TEXELS: f32 = 1.5;

// Decode one log-quantized channel (Rgba8Unorm normalised value -> m^-1).
fn decode(v_norm: f32, vmin: f32, vmax: f32) -> f32 {
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
    var dlon = (lon_deg - cm);
    dlon = dlon - 360.0 * floor((dlon + 180.0) / 360.0);
    let dlon_r = dlon * DEG2RAD;
    if (kind == 0) {
        // Lambert conformal conic.
        let n = u.geo2.x;
        let f = u.geo2.y;
        let rho = R_GROUND * f / pow(tan(PI * 0.25 + phi * 0.5), n);
        let theta = n * dlon_r;
        return vec2<f32>(rho * sin(theta), -rho * cos(theta));
    } else if (kind == 1) {
        // Polar stereographic.
        let k = u.geo2.z;
        let south = u.geo3.x > 0.5;
        if (south) {
            let rho = 2.0 * R_GROUND * k * tan(PI * 0.25 + phi * 0.5);
            return vec2<f32>(rho * sin(dlon_r), rho * cos(dlon_r));
        }
        let rho = 2.0 * R_GROUND * k * tan(PI * 0.25 - phi * 0.5);
        return vec2<f32>(rho * sin(dlon_r), -rho * cos(dlon_r));
    } else if (kind == 2) {
        // Mercator.
        let scale = u.geo2.w;
        return vec2<f32>(R_GROUND * scale * dlon_r, R_GROUND * scale * log(tan(PI * 0.25 + phi * 0.5)));
    }
    // Geographic lat/lon (degrees plane).
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

// Nearest-voxel total extinction (m^-1) at fractional brick coords (documented: the
// sun-OD map is a coarse shadow; the CPU reference trilerps, a small difference).
fn total_ext(b: vec3<f32>) -> f32 {
    let nx = i32(u.dims.x);
    let ny = i32(u.dims.y);
    let nz = i32(u.dims.z);
    let i = i32(round(b.x));
    let j = i32(round(b.y));
    let k = i32(round(b.z));
    if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) {
        return 0.0;
    }
    let texel = textureLoad(volume, vec3<i32>(i, j, k), 0);
    let el = decode(texel.r, u.ql.x, u.ql.y);
    let ei = decode(texel.g, u.ql.z, u.ql.w);
    let ep = decode(texel.b, u.qp.x, u.qp.y);
    return el + ei + ep;
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = i32(u.dims.w);
    let tx = i32(gid.x);
    let ty = i32(gid.y);
    if (tx >= dim || ty >= dim) {
        return;
    }
    let u_min = u.au.w;
    let u_max = u.av.w;
    let v_min = u.sun.w;
    let v_max = u.extent.x;
    let uu = u_min + (f32(tx) + 0.5) / f32(dim) * (u_max - u_min);
    let vv = v_min + (f32(ty) + 0.5) / f32(dim) * (v_max - v_min);
    let center = u.center.xyz;
    let sun = u.sun.xyz;
    let start = center + u.au.xyz * uu + u.av.xyz * vv + sun * u.extent.y;
    let n_steps = i32(u.extent.w);
    let ds = u.vert.z;
    var acc = 0.0;
    for (var s: i32 = 0; s < n_steps; s = s + 1) {
        let t = (f32(s) + 0.5) * ds;
        let p = start - sun * t;
        acc = acc + total_ext(ecef_to_brick(p)) * ds;
    }
    // WS1 edge feather: smoothstep of the texel's distance to the nearest map edge.
    let dedge = f32(min(min(tx, dim - 1 - tx), min(ty, dim - 1 - ty)));
    let tf = clamp(dedge / EDGE_FEATHER_TEXELS, 0.0, 1.0);
    let w = tf * tf * (3.0 - 2.0 * tf);
    textureStore(out_od, vec2<i32>(tx, ty), vec4<f32>(acc * w, 0.0, 0.0, 1.0));
}
