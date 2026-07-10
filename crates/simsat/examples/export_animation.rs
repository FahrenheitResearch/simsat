//! `export_animation` — headless "completed sat-store run -> animated GIF loop".
//!
//! Drives [`simsat::animation::export_store_run_gif`]: reads every `t{HHMM}.rws`
//! frame of ONE completed visible store run (in valid-time order, one frame in
//! memory at a time) and streams them into an infinitely-looping animated GIF.
//! No GPU, no studio GUI — works for any run the studio (or `render_frame`'s
//! `store=` flag) wrote, including runs longer than the studio's in-memory
//! frame cap.
//!
//! USAGE (key=value args, any order):
//!
//!   export_animation run=<run_dir> out=<file.gif> [fps=<f>]
//!
//!   run=<run_dir>   REQUIRED. A completed store RUN directory — the folder holding
//!                   `grid.rwg` + `t{HHMM}.rws` + `run.json`, e.g.
//!                   `{store_root}/simsat/enderlin_d03_rgb_goese_20250621`.
//!   out=<file.gif>  REQUIRED. Output GIF path.
//!   fps=<f>         Playback rate (default 8, the studio timeline default). GIF
//!                   quantizes delays to centiseconds, so the effective rate is
//!                   `100/round(100/fps)` — exact at 10 fps, ~7.7 at 8.
//!
//! HONEST FORMAT NOTES: GIF is palette-quantized (<= 256 colors/frame — smooth
//! twilight/anvil gradients can band slightly vs the stored planes) and this tool
//! deliberately bundles no H.264/ffmpeg. VISIBLE-family runs only (`rgb_r/g/b`
//! planes: the plain visible, GeoColor, and Sandwich products); single-band Kelvin
//! IR runs are stored as raw BT, not RGB — enhance them to RGB first.
//!
//! On completion it prints a one-line `GIFSUMMARY ...` (frames, dims, delay,
//! effective fps, bytes, wall time) to stdout.

use std::path::PathBuf;
use std::time::Instant;

use simsat::animation::{self, DEFAULT_GIF_FPS};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = run(&args) {
        eprintln!("export_animation: {e}");
        eprintln!("run with no args (or --help) for usage.");
        std::process::exit(1);
    }
}

fn run(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return Ok(());
    }
    let mut run_dir: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut fps = DEFAULT_GIF_FPS;
    for a in args {
        let (k, v) = a
            .split_once('=')
            .ok_or_else(|| format!("expected key=value, got '{a}'"))?;
        match k {
            "run" | "run-dir" | "input" => run_dir = Some(PathBuf::from(v)),
            "out" | "output" | "gif" => out = Some(PathBuf::from(v)),
            "fps" => {
                fps = v.parse().map_err(|_| format!("bad fps '{v}'"))?;
                if !(fps > 0.0 && fps.is_finite()) {
                    return Err(format!("fps must be a positive number, got {fps}"));
                }
            }
            other => return Err(format!("unknown key '{other}'")),
        }
    }
    let run_dir = run_dir.ok_or("missing required run=<run_dir>")?;
    let out = out.ok_or("missing required out=<file.gif>")?;

    eprintln!(
        "export_animation: run={} out={} fps={fps}",
        run_dir.display(),
        out.display()
    );
    let t0 = Instant::now();
    let summary = animation::export_store_run_gif(&run_dir, &out, fps)?;
    let wall = t0.elapsed();
    eprintln!(
        "export_animation: wrote {} ({} frames, {}x{}, {} bytes)",
        out.display(),
        summary.frames,
        summary.nx,
        summary.ny,
        summary.bytes
    );
    println!(
        "GIFSUMMARY file={} frames={} dims={}x{} delay_cs={} effective_fps={:.2} \
         requested_fps={:.2} bytes={} wall_s={:.3}",
        out.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
        summary.frames,
        summary.nx,
        summary.ny,
        summary.delay_cs,
        summary.effective_fps(),
        fps,
        summary.bytes,
        wall.as_secs_f64(),
    );
    Ok(())
}

fn print_usage() {
    eprintln!(
        "export_animation — completed sat-store run -> looping animated GIF (CPU, no GPU).\n\n\
         USAGE:\n  export_animation run=<run_dir> out=<file.gif> [fps=<f>]\n\n\
         KEYS:\n\
         \x20 run=<run_dir>   store RUN dir (grid.rwg + t*.rws + run.json)   [required]\n\
         \x20 out=<file.gif>  output GIF path                                [required]\n\
         \x20 fps=<f>         playback rate (default {DEFAULT_GIF_FPS}; GIF rounds to centiseconds)\n"
    );
}
