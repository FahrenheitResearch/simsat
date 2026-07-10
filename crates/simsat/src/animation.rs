//! Animated-GIF loop export (the animation-groundwork engine half).
//!
//! Streams a sequence of rendered RGB8 frames into an animated GIF ONE FRAME AT A
//! TIME — export never holds a whole loop in memory, so it works for any completed
//! sat-store run regardless of the studio's in-memory `frame_cap`. The encoder is
//! the `image` workspace pin's GIF support (the pure-Rust `gif` crate underneath);
//! there is deliberately NO bundled H.264/ffmpeg — GIF is the universally-viewable,
//! dependency-free loop format.
//!
//! HONEST LIMITATIONS (format, not bugs):
//!
//! - **256-color palette**: each GIF frame is quantized to at most 256 colors
//!   (per-frame NeuQuant), so smooth gradients — twilight sky, anvil shading —
//!   can show mild banding relative to the stored PNG/store frames.
//! - **Centisecond timing**: GIF frame delays are quantized to 10 ms units, so the
//!   effective rate is `100 / round(100 / fps)` fps — exact at 10 fps, while the
//!   studio's 8 fps default lands on 13 cs (~7.7 fps). Delays below 2 cs are
//!   clamped up: most decoders treat 0-1 cs as "very slow", not "very fast".
//!
//! The store-run reader half lives in [`crate::store_out`] (`list_run_frames` +
//! `read_visible_frame_rgb`); the headless driver is `examples/export_animation.rs`.
//! The in-studio export button is a later slice (the studio belongs to another
//! workstream this wave).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use image::codecs::gif::{GifEncoder, Repeat};
use image::{Delay, Frame, RgbaImage};

use crate::store_out;

/// Minimum GIF frame delay in centiseconds. The format allows 0-1 cs, but most
/// decoders (browsers included) treat those as a SLOW fallback rather than a fast
/// frame, so requests faster than 50 fps clamp here.
pub const GIF_MIN_DELAY_CS: u32 = 2;

/// Default loop rate for GIF export — matches the studio timeline's 8 fps default
/// (a readable satellite-loop cadence).
pub const DEFAULT_GIF_FPS: f32 = 8.0;

/// NeuQuant encode speed 1..=30 (1 = best quality / slowest). 10 is the standard
/// quality/cost balance for full-frame satellite imagery.
const GIF_ENCODE_SPEED: i32 = 10;

/// Convert a playback rate to the GIF frame delay in CENTISECONDS (the format's
/// native unit): `round(100 / fps)`, clamped to [`GIF_MIN_DELAY_CS`]. A
/// non-positive or non-finite `fps` degrades to 100 cs (one second per frame)
/// rather than erroring — the degenerate input still produces a viewable loop.
pub fn gif_delay_cs(fps: f32) -> u32 {
    if !fps.is_finite() || fps <= 0.0 {
        return 100;
    }
    let cs = (100.0 / fps).round() as u32;
    cs.max(GIF_MIN_DELAY_CS)
}

/// A streaming animated-GIF writer: fixed frame size, fixed per-frame delay,
/// infinite loop. Frames are pushed one at a time as interleaved RGB8 (row 0 =
/// north, the store/PNG convention) and encoded immediately; the GIF trailer is
/// written when the writer is dropped.
pub struct GifAnimation<W: Write> {
    encoder: GifEncoder<W>,
    nx: usize,
    ny: usize,
    delay_cs: u32,
    frames: usize,
}

impl<W: Write> GifAnimation<W> {
    /// Start a GIF of `nx` x `ny` frames at `fps` (see [`gif_delay_cs`] for the
    /// centisecond quantization), looping forever.
    pub fn new(writer: W, nx: usize, ny: usize, fps: f32) -> Result<Self, String> {
        if nx == 0 || ny == 0 {
            return Err(format!("GIF dims must be nonzero, got {nx}x{ny}"));
        }
        if nx > u16::MAX as usize || ny > u16::MAX as usize {
            return Err(format!("GIF dims {nx}x{ny} exceed the format's u16 limit"));
        }
        let mut encoder = GifEncoder::new_with_speed(writer, GIF_ENCODE_SPEED);
        encoder
            .set_repeat(Repeat::Infinite)
            .map_err(|e| format!("GIF repeat header: {e}"))?;
        Ok(Self {
            encoder,
            nx,
            ny,
            delay_cs: gif_delay_cs(fps),
            frames: 0,
        })
    }

    /// Encode one frame from interleaved RGB8 bytes (`nx * ny * 3`, row 0 = north).
    pub fn push_rgb(&mut self, rgb: &[u8]) -> Result<(), String> {
        let n = self.nx * self.ny;
        if rgb.len() != n * 3 {
            return Err(format!(
                "frame byte count {} != {}x{}x3",
                rgb.len(),
                self.nx,
                self.ny
            ));
        }
        let mut rgba = Vec::with_capacity(n * 4);
        for px in rgb.chunks_exact(3) {
            rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
        }
        let buffer = RgbaImage::from_raw(self.nx as u32, self.ny as u32, rgba)
            .ok_or_else(|| "RGBA buffer construction failed".to_string())?;
        // delay_cs * 10 ms is always a whole multiple of the GIF's 10 ms unit, so
        // the encoded delay round-trips exactly (the quantization already happened
        // in gif_delay_cs).
        let delay = Delay::from_numer_denom_ms(self.delay_cs * 10, 1);
        self.encoder
            .encode_frame(Frame::from_parts(buffer, 0, 0, delay))
            .map_err(|e| format!("GIF frame {}: {e}", self.frames))?;
        self.frames += 1;
        Ok(())
    }

    /// Frames encoded so far.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// The quantized per-frame delay (centiseconds).
    pub fn delay_cs(&self) -> u32 {
        self.delay_cs
    }
}

/// What a store-run GIF export produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GifExportSummary {
    pub frames: usize,
    pub nx: usize,
    pub ny: usize,
    /// Per-frame delay actually encoded (centiseconds).
    pub delay_cs: u32,
    /// Size of the written GIF file in bytes.
    pub bytes: u64,
}

impl GifExportSummary {
    /// The effective playback rate after the GIF's centisecond quantization.
    pub fn effective_fps(&self) -> f32 {
        100.0 / self.delay_cs.max(1) as f32
    }
}

/// Export a completed sat-store VISIBLE run directory (the folder holding
/// `grid.rwg` + `t{HHMM}.rws` + `run.json`) to an animated GIF at `fps`. Frames
/// are read back one at a time via [`store_out::read_visible_frame_rgb`] in
/// valid-time order — the export works for ANY completed run, including runs
/// longer than the studio's in-memory `frame_cap`. Off-earth/space pixels (`NaN`
/// planes) render black, matching the studio display. One run = one UTC day (the
/// store's day-scoped run naming); a multi-day sequence is one GIF per run.
pub fn export_store_run_gif(
    run_dir: &Path,
    out: &Path,
    fps: f32,
) -> Result<GifExportSummary, String> {
    let index = store_out::list_run_frames(run_dir)?;
    if index.frames.is_empty() {
        return Err(format!(
            "no t*.rws frames found in run dir {}",
            run_dir.display()
        ));
    }
    let file = File::create(out).map_err(|e| format!("create {}: {e}", out.display()))?;
    let mut anim = GifAnimation::new(BufWriter::new(file), index.nx, index.ny, fps)?;
    for frame in &index.frames {
        let rgb = store_out::read_visible_frame_rgb(&frame.path, index.nx, index.ny)?;
        anim.push_rgb(&rgb)?;
    }
    let frames = anim.frames();
    let delay_cs = anim.delay_cs();
    // Dropping the writer emits the GIF trailer and flushes the BufWriter.
    drop(anim);
    let bytes = std::fs::metadata(out)
        .map_err(|e| format!("stat {}: {e}", out.display()))?
        .len();
    Ok(GifExportSummary {
        frames,
        nx: index.nx,
        ny: index.ny,
        delay_cs,
        bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::SatellitePreset;
    use crate::store_out::{VisibleFrame, write_visible_frame};
    use image::AnimationDecoder;
    use image::codecs::gif::GifDecoder;
    use std::io::Cursor;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static UNIQ: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> PathBuf {
        let n = UNIQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("simsat-anim-{}-{}", std::process::id(), n));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A solid-color RGB8 frame.
    fn solid_rgb(nx: usize, ny: usize, rgb: [u8; 3]) -> Vec<u8> {
        let mut out = Vec::with_capacity(nx * ny * 3);
        for _ in 0..nx * ny {
            out.extend_from_slice(&rgb);
        }
        out
    }

    #[test]
    fn delay_quantization_rounds_to_centiseconds_and_clamps() {
        // Exact at 10 fps; the studio's 8 fps default lands on 13 cs.
        assert_eq!(gif_delay_cs(10.0), 10);
        assert_eq!(gif_delay_cs(8.0), 13);
        // 50 fps = the 2 cs floor; anything faster clamps to it.
        assert_eq!(gif_delay_cs(50.0), 2);
        assert_eq!(gif_delay_cs(1000.0), GIF_MIN_DELAY_CS);
        // Degenerate rates degrade to 1 s/frame instead of erroring.
        assert_eq!(gif_delay_cs(0.0), 100);
        assert_eq!(gif_delay_cs(-3.0), 100);
        assert_eq!(gif_delay_cs(f32::NAN), 100);
    }

    #[test]
    fn three_frames_round_trip_count_dims_and_delay() {
        // Encode three solid-color frames at 10 fps (exactly 100 ms), decode with
        // the image GIF decoder, and check count / dims / per-frame delay / that
        // each frame kept its own dominant color (solid frames quantize cleanly;
        // a small tolerance guards NeuQuant's palette placement).
        let (nx, ny) = (16usize, 12usize);
        let colors = [[200u8, 30, 30], [30, 200, 30], [30, 30, 200]];
        let mut bytes: Vec<u8> = Vec::new();
        {
            let mut anim = GifAnimation::new(&mut bytes, nx, ny, 10.0).expect("writer");
            for c in colors {
                anim.push_rgb(&solid_rgb(nx, ny, c)).expect("push");
            }
            assert_eq!(anim.frames(), 3);
            assert_eq!(anim.delay_cs(), 10);
        }
        let decoder = GifDecoder::new(Cursor::new(&bytes)).expect("decode");
        let frames = decoder.into_frames().collect_frames().expect("frames");
        assert_eq!(frames.len(), 3, "frame count round-trips");
        for (k, frame) in frames.iter().enumerate() {
            assert_eq!(frame.buffer().width() as usize, nx);
            assert_eq!(frame.buffer().height() as usize, ny);
            assert_eq!(
                frame.delay().numer_denom_ms(),
                (100, 1),
                "frame {k} delay is exactly 100 ms"
            );
            let px = frame.buffer().get_pixel((nx / 2) as u32, (ny / 2) as u32);
            for c in 0..3 {
                let want = colors[k][c] as i32;
                let got = px[c] as i32;
                assert!(
                    (want - got).abs() <= 4,
                    "frame {k} channel {c}: want ~{want}, got {got}"
                );
            }
            assert_eq!(px[3], 255, "frames are opaque");
        }
    }

    #[test]
    fn a_wrong_sized_frame_is_rejected() {
        let mut bytes: Vec<u8> = Vec::new();
        let mut anim = GifAnimation::new(&mut bytes, 8, 8, 10.0).expect("writer");
        let err = anim.push_rgb(&solid_rgb(8, 7, [1, 2, 3])).unwrap_err();
        assert!(
            err.contains("byte count"),
            "size mismatch names the byte count: {err}"
        );
        assert_eq!(anim.frames(), 0, "the rejected frame was not counted");
        // Zero-sized and oversized GIFs are rejected up front.
        assert!(GifAnimation::new(Vec::<u8>::new(), 0, 8, 10.0).is_err());
        assert!(GifAnimation::new(Vec::<u8>::new(), 70_000, 8, 10.0).is_err());
    }

    /// A tiny store visible frame with a constant red channel + one NaN (space)
    /// pixel at index 0 (mirrors store_out's synthetic frame shape).
    fn store_frame(nx: usize, ny: usize, red: f32, hhmm: u16) -> VisibleFrame {
        let n = nx * ny;
        let mut lat = vec![0f32; n];
        let mut lon = vec![0f32; n];
        for j in 0..ny {
            for i in 0..nx {
                lat[j * nx + i] = 45.0 - j as f32 * 0.1;
                lon[j * nx + i] = -100.0 + i as f32 * 0.1;
            }
        }
        let mut r = vec![red; n];
        let mut g = vec![128f32; n];
        let mut b = vec![64f32; n];
        r[0] = f32::NAN;
        g[0] = f32::NAN;
        b[0] = f32::NAN;
        VisibleFrame {
            nx,
            ny,
            rgb_r: r,
            rgb_g: g,
            rgb_b: b,
            lat,
            lon,
            sector: "anim qa".to_string(),
            satellite: SatellitePreset::GoesEast,
            band: 2,
            year: 2025,
            month: 6,
            day: 21,
            hhmm,
        }
    }

    #[test]
    fn a_store_run_exports_one_gif_frame_per_timestep_in_time_order() {
        // Write a REAL multi-frame store run (the same write_visible_frame path the
        // studio loop uses) with three distinct frames, export it, and decode: 3
        // frames, run dims, time order preserved (red channel ascends with hhmm),
        // NaN space pixel black.
        let root = temp_dir();
        let (nx, ny) = (10usize, 8usize);
        // Deliberately written OUT of time order; the export must sort by hhmm.
        let mut run_dir = None;
        for (red, hhmm) in [(120f32, 215u16), (60.0, 145), (180.0, 300)] {
            let w = write_visible_frame(&root, &store_frame(nx, ny, red, hhmm)).expect("write");
            run_dir = Some(w.run_dir.clone());
        }
        let run_dir = run_dir.unwrap();
        let out = root.join("loop.gif");
        let summary = export_store_run_gif(&run_dir, &out, 10.0).expect("export");
        assert_eq!(summary.frames, 3);
        assert_eq!((summary.nx, summary.ny), (nx, ny));
        assert_eq!(summary.delay_cs, 10);
        assert!(summary.bytes > 0 && out.is_file());
        assert!((summary.effective_fps() - 10.0).abs() < 1e-6);

        let decoder =
            GifDecoder::new(std::io::BufReader::new(File::open(&out).unwrap())).expect("decode");
        let frames = decoder.into_frames().collect_frames().expect("frames");
        assert_eq!(frames.len(), 3, "one GIF frame per store timestep");
        // hhmm order 145/215/300 -> red 60/120/180 ascending.
        let mut last_red = -1i32;
        for (k, frame) in frames.iter().enumerate() {
            assert_eq!(frame.buffer().width() as usize, nx);
            assert_eq!(frame.buffer().height() as usize, ny);
            let mid = frame.buffer().get_pixel(5, 4);
            assert!(
                mid[0] as i32 > last_red,
                "frame {k} red {} not ascending (time order broken)",
                mid[0]
            );
            last_red = mid[0] as i32;
            // The NaN space pixel (index 0) exports black.
            let space = frame.buffer().get_pixel(0, 0);
            assert!(
                space[0] <= 4 && space[1] <= 4 && space[2] <= 4,
                "frame {k} space pixel should be black, got {space:?}"
            );
        }
        std::fs::remove_dir_all(&root).ok();
    }
}
