//! `ssb_preview` — read a cached `.ssb` brick and write grayscale diagnostic
//! PNGs of a few planes, for a quick visual sanity check of an ingest (M0
//! real-data proof; no GPU, runs on the headless build nodes via the `image`
//! crate). This is a debug/preview tool, NOT the render path.
//!
//! Usage: `ssb_preview <path.ssb> <out_prefix>`
//!
//! All images are grayscale and north-up / west-left: the brick's `j` axis runs
//! south->north, so image rows are flipped so north is at the top. The files are
//! `{prefix}_tauup_maxz_sqrt.png` (max-over-z of `tau_up`, i.e. full-column
//! optical depth to space), `{prefix}_ext_liquid_maxz_sqrt.png` (max-over-z
//! cloud-liquid extinction), `{prefix}_ext_ice_maxz_sqrt.png` (small/pristine ice),
//! `{prefix}_ext_snow_maxz_sqrt.png` (the snow-only auxiliary subset),
//! `{prefix}_ext_precip_maxz_sqrt.png` (the total rain+graupel+snow channel), and
//! `{prefix}_temp_kmid_linear.png` (a mid-level temperature slice in K).
//!
//! Each image is auto-scaled to its own finite min/max; the scale type (sqrt or
//! linear) is in the file name and the numeric range is logged to stderr — no
//! text is rendered into the image itself.

use std::path::{Path, PathBuf};

use image::{GrayImage, Luma};
use simsat::bricks::{self, LogQuant, VolumeBrick};

#[derive(Clone, Copy)]
enum Stretch {
    Sqrt,
    Linear,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: ssb_preview <path.ssb> <out_prefix>");
        std::process::exit(2);
    }
    let path = PathBuf::from(&args[1]);
    let prefix = args[2].clone();

    let brick = match bricks::read_ssb(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("ssb_preview: failed to read {}: {e}", path.display());
            std::process::exit(1);
        }
    };
    eprintln!(
        "ssb_preview: {} -> {}x{}x{}",
        path.display(),
        brick.nx,
        brick.ny,
        brick.nz
    );

    let tauup = max_over_z(&brick, &brick.tau_up, brick.quant.get("tau_up"));
    write_gray(
        &tauup,
        brick.nx,
        brick.ny,
        Stretch::Sqrt,
        &out_path(&prefix, "tauup_maxz_sqrt"),
    );

    for (name, codes) in [
        ("ext_liquid", &brick.ext_liquid),
        ("ext_ice", &brick.ext_ice),
        ("ext_snow", &brick.ext_snow),
        ("ext_precip", &brick.ext_precip),
    ] {
        let plane = max_over_z(&brick, codes, brick.quant.get(name));
        write_gray(
            &plane,
            brick.nx,
            brick.ny,
            Stretch::Sqrt,
            &out_path(&prefix, &format!("{name}_maxz_sqrt")),
        );
    }

    // Mid-level temperature slice (K).
    let temp_k = bricks::decode_temperature_kelvin(&brick.temperature_f16);
    let mid = brick.nz / 2;
    let plane_cells = brick.nx * brick.ny;
    let slice = &temp_k[mid * plane_cells..(mid + 1) * plane_cells];
    write_gray(
        slice,
        brick.nx,
        brick.ny,
        Stretch::Linear,
        &out_path(&prefix, "temp_kmid_linear"),
    );
}

fn out_path(prefix: &str, tag: &str) -> PathBuf {
    PathBuf::from(format!("{prefix}_{tag}.png"))
}

/// Decode a quantized 3-D channel and reduce it to its per-column maximum.
fn max_over_z(brick: &VolumeBrick, codes: &[u8], quant: LogQuant) -> Vec<f32> {
    let (nx, ny, nz) = (brick.nx, brick.ny, brick.nz);
    let mut out = vec![0f32; nx * ny];
    for k in 0..nz {
        let layer = &codes[k * nx * ny..(k + 1) * nx * ny];
        for (o, &code) in out.iter_mut().zip(layer.iter()) {
            let v = quant.decode(code);
            if v > *o {
                *o = v;
            }
        }
    }
    out
}

fn write_gray(plane: &[f32], nx: usize, ny: usize, stretch: Stretch, path: &Path) {
    let (mut vmin, mut vmax) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in plane {
        if v.is_finite() {
            vmin = vmin.min(v);
            vmax = vmax.max(v);
        }
    }
    if !vmin.is_finite() || vmax <= vmin {
        vmin = 0.0;
        vmax = if vmax.is_finite() && vmax > 0.0 {
            vmax
        } else {
            1.0
        };
    }
    let span = (vmax - vmin).max(f32::MIN_POSITIVE);
    let img = GrayImage::from_fn(nx as u32, ny as u32, |x, y| {
        // North-up: image row 0 is north, which is the brick's highest j.
        let j = ny - 1 - y as usize;
        let v = plane[j * nx + x as usize];
        let t = if v.is_finite() {
            ((v - vmin) / span).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let shaped = match stretch {
            Stretch::Sqrt => t.sqrt(),
            Stretch::Linear => t,
        };
        Luma([(shaped * 255.0).round().clamp(0.0, 255.0) as u8])
    });
    match img.save(path) {
        Ok(()) => eprintln!(
            "ssb_preview: wrote {} (range [{vmin:.4}, {vmax:.4}])",
            path.display()
        ),
        Err(e) => eprintln!("ssb_preview: failed to write {}: {e}", path.display()),
    }
}
