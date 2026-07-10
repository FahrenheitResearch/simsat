//! The `.ssb` on-disk volume-brick format (design doc section 2).
//!
//! One `.ssb` file holds one WRF timestep resampled onto an affine vertical axis
//! (`z(k) = z_min_m + k*dz_m`, MSL) over the native WRF `(i, j)` horizontal grid
//! (NO decimation/windowing in M0 — that is M7).
//!
//! Payload channels (Texture A) are four log-quantized u8 3-D channels with
//! per-volume scales in the header: `ext_liquid` (cloud-liquid extinction, m^-1,
//! from `QCLOUD`), `ext_ice` (small-ice extinction, `QICE` only since SSB v3),
//! `ext_precip` (the large-particle extinction: `QRAIN + QGRAUP` at rain optics
//! plus — since SSB v3, the snow-optics fix — `QSNOW` at its own aggregate beta;
//! each species converts at its own optics before the sum), and `tau_up`
//! (cumulative optical depth from brick-top down to each level, precomputed at
//! ingest; feeds cloud ambient, the IR fast path, and shadows).
//!
//! The QVAPOR channel (owner decision 6 — a full channel now, for a later 6.2 um
//! water-vapor IR band) is its OWN fifth log-quantized u8 channel, `qvapor`
//! (mixing ratio kg/kg). Representation choice (recorded in the manifest as
//! `qvapor`): a u8 log channel, not an f16 plane. Rationale — the bandwidth-lean
//! rule (owner decision 1) favors u8; a WV IR band is coarse-tolerance, and the
//! fixed log window still gives it usable dynamic range. f16 was the alternative,
//! rejected on bandwidth.
//!
//! Texture B is temperature as f16 stored in degrees CELSIUS. Storing Kelvin in
//! f16 would only resolve ~0.25 K near 300 K (aliasing the 1 K IR enhancement
//! steps); Celsius keeps the magnitude under ~90 across the whole atmosphere,
//! where f16 resolves < 0.1 K — the fidelity the design asks for. Decoders add
//! 273.15.
//!
//! 2-D domain planes (f32) are `hgt`, `landmask`, `tsk`, `u10`, `v10`, plus
//! optional `snowh` and best-effort `ivgtyp` when present in the source.
//!
//! On-disk layout of a `t{stamp}.ssb` file (`stamp` = `YYYYMMDD_HHMM`, so
//! multi-day runs never collide on wall-clock HHMM — M0-review MINOR-2)
//! (little-endian):
//! ```text
//!   [0..4]   SSB_MAGIC (b"SSB1")
//!   [4..8]   SSB_FORMAT_VERSION (u32)
//!   [8..12]  header JSON length (u32)
//!   [12..]   header JSON (BrickHeader, uncompressed, self-describing)
//!   [...]    flate2/zlib-compressed raw payload
//! ```
//! The decompressed payload is the channels concatenated in `channels_3d` order
//! (each `nx*ny*nz` bytes, index `(k*ny + y)*nx + x`), then temperature
//! (`nx*ny*nz` u16), then the 2-D planes (`nx*ny` f32 each) in `planes_2d` order.
//! A sibling `run.json` (see `RunManifest`) indexes all timesteps of a run.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use flate2::Compression;
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use serde::{Deserialize, Serialize};

use crate::{SSB_FORMAT_VERSION, SSB_MAGIC};

/// Fixed dynamic-range window for log quantization: codes 1..=255 span five
/// decades below the per-volume peak. This bounds the per-code relative step to
/// `(1e5)^(1/254) - 1 ~= 4.7%` (nearest-code error ~= 2.3%), independent of data.
pub const QUANT_DYNAMIC_RANGE: f64 = 1.0e5;

/// Kelvin<->Celsius offset used by the f16 temperature encoding.
pub const CELSIUS_OFFSET_K: f64 = 273.15;

/// Sanity ceiling on the decompressed payload size an `.ssb` header may describe
/// (64 GiB — an order of magnitude above any real brick; the Enderlin 800x800x80
/// payload is ~0.36 GB). Bounds the [`read_ssb`] decompress so corrupt dims cannot
/// balloon memory (checked, never trusted).
pub const MAX_SSB_PAYLOAD_BYTES: u64 = 64 << 30;

/// The ordered 3-D quantized channel names (Texture A + the QVAPOR channel).
pub const CHANNELS_3D: [&str; 5] = ["ext_liquid", "ext_ice", "ext_precip", "tau_up", "qvapor"];

/// Per-channel log-quantization scale (per volume). Code 0 is reserved for an
/// exact zero / below-floor value; codes 1..=255 are log-uniform in `[vmin, vmax]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LogQuant {
    pub vmin: f64,
    pub vmax: f64,
}

impl LogQuant {
    /// Derive the scale from a channel's values: `vmax` = largest finite positive
    /// value, `vmin = vmax / QUANT_DYNAMIC_RANGE`. An all-zero channel gets a
    /// zero scale (everything encodes to code 0).
    pub fn from_values(values: &[f32]) -> Self {
        let vmax = values
            .iter()
            .copied()
            .filter(|v| v.is_finite() && *v > 0.0)
            .fold(0.0f64, |acc, v| acc.max(v as f64));
        if vmax <= 0.0 {
            Self {
                vmin: 0.0,
                vmax: 0.0,
            }
        } else {
            Self {
                vmin: vmax / QUANT_DYNAMIC_RANGE,
                vmax,
            }
        }
    }

    /// Encode one value to a u8 code.
    #[inline]
    pub fn encode(&self, value: f32) -> u8 {
        let v = value as f64;
        // Code 0 for an all-zero scale, any non-finite value (NaN or +/-Inf), or a
        // non-positive value. `!is_finite()` covers +Inf, which otherwise flows to
        // `(Inf*254).round() as i64 + 1` and overflow-panics in debug builds
        // (M0-review MINOR-1). `from_values` already excludes non-finite when
        // deriving `vmax`, so an Inf sample can survive into `encode`.
        if self.vmax <= 0.0 || !v.is_finite() || v <= 0.0 {
            return 0;
        }
        if v <= self.vmin {
            return 1;
        }
        let span = (self.vmax / self.vmin).ln();
        if span <= 0.0 {
            return 1;
        }
        let t = (v / self.vmin).ln() / span;
        let code = (t * 254.0).round() as i64 + 1;
        code.clamp(1, 255) as u8
    }

    /// Decode a u8 code back to a value.
    #[inline]
    pub fn decode(&self, code: u8) -> f32 {
        if code == 0 || self.vmax <= 0.0 {
            return 0.0;
        }
        let t = (code as f64 - 1.0) / 254.0;
        (self.vmin * (self.vmax / self.vmin).powf(t)) as f32
    }
}

/// The five per-volume channel scales, keyed by channel name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelQuant(pub BTreeMap<String, LogQuant>);

impl ChannelQuant {
    /// Look up a channel's scale (a zero scale if absent).
    pub fn get(&self, name: &str) -> LogQuant {
        self.0.get(name).copied().unwrap_or(LogQuant {
            vmin: 0.0,
            vmax: 0.0,
        })
    }
}

/// Encode an f32 channel to (scale, codes) using log quantization.
pub fn encode_log_channel(values: &[f32]) -> (LogQuant, Vec<u8>) {
    let quant = LogQuant::from_values(values);
    let codes = values.iter().map(|&v| quant.encode(v)).collect();
    (quant, codes)
}

/// Encode a Kelvin temperature channel as f16 (Celsius) bits.
pub fn encode_temperature_celsius(kelvin: &[f32]) -> Vec<u16> {
    kelvin
        .iter()
        .map(|&k| f32_to_f16_bits((k as f64 - CELSIUS_OFFSET_K) as f32))
        .collect()
}

/// Decode an f16 (Celsius) temperature channel back to Kelvin.
pub fn decode_temperature_kelvin(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| (f16_bits_to_f32(b) as f64 + CELSIUS_OFFSET_K) as f32)
        .collect()
}

/// IEEE-754 binary16 encode (round to nearest, ties to even).
pub(crate) fn f32_to_f16_bits(value: f32) -> u16 {
    let x = value.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let mantissa = (x & 0x007f_ffff) as i32;
    let exp = ((x >> 23) & 0xff) as i32;
    if exp == 0xff {
        // Inf / NaN.
        return sign | 0x7c00 | if mantissa != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflow -> Inf
    }
    if e <= 0 {
        if e < -10 {
            return sign; // underflow -> signed zero
        }
        // Subnormal half.
        let m = mantissa | 0x0080_0000; // restore implicit leading 1
        let shift = 14 - e; // 14..=24
        let half_m = (m >> shift) as u16;
        let remainder = m & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        let mut result = sign | half_m;
        if remainder > halfway || (remainder == halfway && (half_m & 1) == 1) {
            result += 1;
        }
        return result;
    }
    // Normal half.
    let half_m = (mantissa >> 13) as u16;
    let remainder = mantissa & 0x1fff;
    let mut result = sign | ((e as u16) << 10) | half_m;
    if remainder > 0x1000 || (remainder == 0x1000 && (half_m & 1) == 1) {
        result += 1; // carry into exponent is correct
    }
    result
}

/// IEEE-754 binary16 decode.
pub(crate) fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal: value = mant * 2^-24, exactly representable in f32.
        let val = (mant as f32) * 2.0f32.powi(-24);
        return if sign != 0 { -val } else { val };
    }
    if exp == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (mant << 13));
    }
    let exp32 = exp + (127 - 15);
    f32::from_bits(sign | (exp32 << 23) | (mant << 13))
}

/// The self-describing header at the front of every `.ssb` file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrickHeader {
    pub format_version: u32,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub z_min_m: f64,
    pub dz_m: f64,
    pub temperature_encoding: String,
    pub quant: ChannelQuant,
    pub channels_3d: Vec<String>,
    pub planes_2d: Vec<String>,
    pub has_snowh: bool,
    pub has_ivgtyp: bool,
    pub time_iso: Option<String>,
}

/// One timestep's volume brick, in memory.
#[derive(Debug, Clone, PartialEq)]
pub struct VolumeBrick {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub z_min_m: f64,
    pub dz_m: f64,
    pub time_iso: Option<String>,
    pub quant: ChannelQuant,
    /// 3-D quantized channels, index `(k*ny + y)*nx + x`.
    pub ext_liquid: Vec<u8>,
    pub ext_ice: Vec<u8>,
    pub ext_precip: Vec<u8>,
    pub tau_up: Vec<u8>,
    pub qvapor: Vec<u8>,
    /// Temperature as f16 (Celsius) bits, same indexing as the 3-D channels.
    pub temperature_f16: Vec<u16>,
    /// 2-D planes, index `y*nx + x`.
    pub hgt: Vec<f32>,
    pub landmask: Vec<f32>,
    pub tsk: Vec<f32>,
    pub u10: Vec<f32>,
    pub v10: Vec<f32>,
    pub snowh: Option<Vec<f32>>,
    pub ivgtyp: Option<Vec<f32>>,
}

impl VolumeBrick {
    fn cells_3d(&self) -> usize {
        self.nx * self.ny * self.nz
    }
    fn cells_2d(&self) -> usize {
        self.nx * self.ny
    }

    fn channel_3d(&self, name: &str) -> &[u8] {
        match name {
            "ext_liquid" => &self.ext_liquid,
            "ext_ice" => &self.ext_ice,
            "ext_precip" => &self.ext_precip,
            "tau_up" => &self.tau_up,
            "qvapor" => &self.qvapor,
            other => panic!("unknown 3-D channel {other}"),
        }
    }

    /// The ordered 2-D plane names this brick carries (including optional
    /// `snowh`/`ivgtyp`), matching the on-disk payload order.
    pub fn planes_2d_names(&self) -> Vec<String> {
        self.planes_2d_list()
            .iter()
            .map(|(name, _)| name.to_string())
            .collect()
    }

    fn planes_2d_list(&self) -> Vec<(&'static str, &Vec<f32>)> {
        let mut planes: Vec<(&'static str, &Vec<f32>)> = vec![
            ("hgt", &self.hgt),
            ("landmask", &self.landmask),
            ("tsk", &self.tsk),
            ("u10", &self.u10),
            ("v10", &self.v10),
        ];
        if let Some(snowh) = &self.snowh {
            planes.push(("snowh", snowh));
        }
        if let Some(ivgtyp) = &self.ivgtyp {
            planes.push(("ivgtyp", ivgtyp));
        }
        planes
    }

    fn header(&self) -> BrickHeader {
        BrickHeader {
            format_version: SSB_FORMAT_VERSION,
            nx: self.nx,
            ny: self.ny,
            nz: self.nz,
            z_min_m: self.z_min_m,
            dz_m: self.dz_m,
            temperature_encoding: "celsius_f16".to_string(),
            quant: self.quant.clone(),
            channels_3d: CHANNELS_3D.iter().map(|s| s.to_string()).collect(),
            planes_2d: self
                .planes_2d_list()
                .iter()
                .map(|(name, _)| name.to_string())
                .collect(),
            has_snowh: self.snowh.is_some(),
            has_ivgtyp: self.ivgtyp.is_some(),
            time_iso: self.time_iso.clone(),
        }
    }
}

/// Errors reading/writing an `.ssb` brick or its manifest.
#[derive(Debug)]
pub enum BrickError {
    Io(std::io::Error),
    Json(serde_json::Error),
    BadMagic([u8; 4]),
    UnsupportedVersion(u32),
    /// A `run.json` written by a DIFFERENT format version (any schema): refused with a
    /// remedy — the brick cache is regenerable, never migrated in place.
    UnsupportedManifestVersion {
        found: u64,
        expected: u32,
    },
    /// An existing `run.json` describes a DIFFERENT grid / z-axis / projection shape
    /// than the file being ingested into it (a run_id collision across domains would
    /// silently mix bricks of different grids into one run).
    CacheMismatch(String),
    Truncated(String),
}

impl std::fmt::Display for BrickError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Json(e) => write!(f, "json error: {e}"),
            Self::BadMagic(m) => write!(f, "bad .ssb magic: {m:?}"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported .ssb version: {v}"),
            Self::UnsupportedManifestVersion { found, expected } => write!(
                f,
                "unsupported run.json format_version {found} (this build reads \
                 {expected}); the brick cache is regenerable — delete the run's cache \
                 directory and re-render from the source wrfout to re-ingest"
            ),
            Self::CacheMismatch(s) => write!(f, "cache mismatch: {s}"),
            Self::Truncated(s) => write!(f, "truncated .ssb: {s}"),
        }
    }
}

impl std::error::Error for BrickError {}

impl From<std::io::Error> for BrickError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<serde_json::Error> for BrickError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

/// The datetime key/stamp for a brick: `YYYYMMDD_HHMM` when the ISO time is known,
/// else the 4-digit `HHMM` fallback for source files with no `Times` variable.
///
/// Keying on the full date (M0-review MINOR-2) is the fix for silent brick/manifest
/// collisions on runs longer than 24 h, where two timesteps at the same wall-clock
/// time on different days previously mapped to the same `t{HHMM}.ssb`.
pub fn time_stamp(time_iso: Option<&str>, hhmm: u16) -> String {
    if let Some(iso) = time_iso {
        let digits: String = iso.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.len() >= 12 {
            return format!("{}_{}", &digits[0..8], &digits[8..12]);
        }
    }
    format!("{hhmm:04}")
}

/// The `t{stamp}.ssb` file name for a datetime `stamp` (see [`time_stamp`]).
pub fn brick_file_name(stamp: &str) -> String {
    format!("t{stamp}.ssb")
}

/// The `.ssb` file name for a brick given its ISO time and `HHMM`. The single path
/// resolver ingest and the studio share so a cache name is always reproducible.
pub fn brick_file_name_for(time_iso: Option<&str>, hhmm: u16) -> String {
    brick_file_name(&time_stamp(time_iso, hhmm))
}

/// The run directory `{cache_dir}/{run_id}`.
pub fn run_dir(cache_dir: &Path, run_id: &str) -> PathBuf {
    cache_dir.join(run_id)
}

/// Serialize a brick to its raw (uncompressed) payload bytes.
fn payload_bytes(brick: &VolumeBrick) -> Vec<u8> {
    let cells_3d = brick.cells_3d();
    let cells_2d = brick.cells_2d();
    let planes = brick.planes_2d_list();
    let mut out =
        Vec::with_capacity(cells_3d * (CHANNELS_3D.len() + 2) + planes.len() * cells_2d * 4);
    for name in CHANNELS_3D {
        out.extend_from_slice(brick.channel_3d(name));
    }
    for &bits in &brick.temperature_f16 {
        out.extend_from_slice(&bits.to_le_bytes());
    }
    for (_, plane) in planes {
        for &v in plane {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// Write a brick to `path` as an `.ssb` file.
pub fn write_ssb(path: &Path, brick: &VolumeBrick) -> Result<u64, BrickError> {
    let header = brick.header();
    let header_json = serde_json::to_vec(&header)?;
    let raw = payload_bytes(brick);

    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&raw)?;
    let compressed = encoder.finish()?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::File::create(path)?;
    file.write_all(&SSB_MAGIC)?;
    file.write_all(&SSB_FORMAT_VERSION.to_le_bytes())?;
    file.write_all(&(header_json.len() as u32).to_le_bytes())?;
    file.write_all(&header_json)?;
    file.write_all(&compressed)?;
    file.flush()?;
    Ok(std::fs::metadata(path)?.len())
}

fn take<'a>(
    buf: &'a [u8],
    cursor: &mut usize,
    n: usize,
    what: &str,
) -> Result<&'a [u8], BrickError> {
    let end = cursor
        .checked_add(n)
        .ok_or_else(|| BrickError::Truncated(format!("{what}: length overflow")))?;
    if end > buf.len() {
        return Err(BrickError::Truncated(format!(
            "{what}: need {n} bytes at {cursor}, have {}",
            buf.len()
        )));
    }
    let slice = &buf[*cursor..end];
    *cursor = end;
    Ok(slice)
}

fn read_plane_f32(
    raw: &[u8],
    cursor: &mut usize,
    cells_2d: usize,
    name: &str,
) -> Result<Vec<f32>, BrickError> {
    let bytes = take(raw, cursor, cells_2d * 4, name)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Read an `.ssb` file back into a `VolumeBrick`.
pub fn read_ssb(path: &Path) -> Result<VolumeBrick, BrickError> {
    let bytes = std::fs::read(path)?;
    let mut cursor = 0usize;

    let magic = take(&bytes, &mut cursor, 4, "magic")?;
    if magic != SSB_MAGIC {
        let mut m = [0u8; 4];
        m.copy_from_slice(magic);
        return Err(BrickError::BadMagic(m));
    }
    let version = u32::from_le_bytes(take(&bytes, &mut cursor, 4, "version")?.try_into().unwrap());
    if version != SSB_FORMAT_VERSION {
        return Err(BrickError::UnsupportedVersion(version));
    }
    let header_len_bytes = take(&bytes, &mut cursor, 4, "header_len")?;
    let header_len = u32::from_le_bytes(header_len_bytes.try_into().unwrap()) as usize;
    let header_bytes = take(&bytes, &mut cursor, header_len, "header")?;
    let header: BrickHeader = serde_json::from_slice(header_bytes)?;

    // Corrupt-header hardening: the dims are untrusted bytes, so the cell products are
    // computed CHECKED (the plain product overflow-panics in a debug build and wraps to
    // a bogus small size in release), the described payload size is sanity-capped, and
    // the decompress below is BOUNDED to that size (a corrupt / zip-bomb stream cannot
    // balloon memory past what the header legitimately describes).
    let cells_2d = header.nx.checked_mul(header.ny).ok_or_else(|| {
        BrickError::Truncated(format!("header dims {}x{} overflow", header.nx, header.ny))
    })?;
    let cells_3d = cells_2d.checked_mul(header.nz).ok_or_else(|| {
        BrickError::Truncated(format!(
            "header dims {}x{}x{} overflow",
            header.nx, header.ny, header.nz
        ))
    })?;
    let n_planes = 5usize + usize::from(header.has_snowh) + usize::from(header.has_ivgtyp);
    let expected_payload = cells_3d
        .checked_mul(CHANNELS_3D.len() + 2) // five u8 channels + the u16 temperature
        .and_then(|v| v.checked_add(cells_2d.checked_mul(n_planes * 4)?))
        .filter(|&v| (v as u64) <= MAX_SSB_PAYLOAD_BYTES)
        .ok_or_else(|| {
            BrickError::Truncated(format!(
                "header describes an implausible payload ({}x{}x{})",
                header.nx, header.ny, header.nz
            ))
        })?;

    let mut decoder = ZlibDecoder::new(&bytes[cursor..]).take(expected_payload as u64 + 1);
    let mut raw = Vec::new();
    decoder.read_to_end(&mut raw)?;
    if raw.len() > expected_payload {
        return Err(BrickError::Truncated(format!(
            "payload exceeds the {expected_payload} bytes the header describes"
        )));
    }
    let mut p = 0usize;

    let ext_liquid = take(&raw, &mut p, cells_3d, "ext_liquid")?.to_vec();
    let ext_ice = take(&raw, &mut p, cells_3d, "ext_ice")?.to_vec();
    let ext_precip = take(&raw, &mut p, cells_3d, "ext_precip")?.to_vec();
    let tau_up = take(&raw, &mut p, cells_3d, "tau_up")?.to_vec();
    let qvapor = take(&raw, &mut p, cells_3d, "qvapor")?.to_vec();

    let temp_bytes = take(&raw, &mut p, cells_3d * 2, "temperature")?;
    let temperature_f16: Vec<u16> = temp_bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();

    let hgt = read_plane_f32(&raw, &mut p, cells_2d, "hgt")?;
    let landmask = read_plane_f32(&raw, &mut p, cells_2d, "landmask")?;
    let tsk = read_plane_f32(&raw, &mut p, cells_2d, "tsk")?;
    let u10 = read_plane_f32(&raw, &mut p, cells_2d, "u10")?;
    let v10 = read_plane_f32(&raw, &mut p, cells_2d, "v10")?;
    let snowh = if header.has_snowh {
        Some(read_plane_f32(&raw, &mut p, cells_2d, "snowh")?)
    } else {
        None
    };
    let ivgtyp = if header.has_ivgtyp {
        Some(read_plane_f32(&raw, &mut p, cells_2d, "ivgtyp")?)
    } else {
        None
    };

    Ok(VolumeBrick {
        nx: header.nx,
        ny: header.ny,
        nz: header.nz,
        z_min_m: header.z_min_m,
        dz_m: header.dz_m,
        time_iso: header.time_iso,
        quant: header.quant,
        ext_liquid,
        ext_ice,
        ext_precip,
        tau_up,
        qvapor,
        temperature_f16,
        hgt,
        landmask,
        tsk,
        u10,
        v10,
        snowh,
        ivgtyp,
    })
}

// ── run.json manifest ──────────────────────────────────────────────────────

/// Projection attributes recorded in the manifest (mirrors the WRF globals).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ManifestProjection {
    pub map_proj: i32,
    pub truelat1_deg: f64,
    pub truelat2_deg: f64,
    pub stand_lon_deg: f64,
    pub cen_lat_deg: f64,
    pub cen_lon_deg: f64,
    pub dx_m: f64,
    pub dy_m: f64,
}

/// The per-timestep georeference anchor persisted in the manifest: the exact values
/// [`crate::frame::GridGeoref::from_wrf_center`] anchored the wrfout path's georef with,
/// so the cached-run path can rebuild it BIT-IDENTICALLY via
/// [`crate::frame::GridGeoref::from_anchor`] (closes deferred M1 NOTE-4 — the wrfout vs
/// cache paths no longer fork a duplicate store-run dir), and so a MOVING NEST (whose
/// domain re-centres between timesteps) is anchored at each timestep's own position.
///
/// `dx`/`dy` are persisted alongside the four `ref_*` fields because a `MAP_PROJ = 6`
/// lat/lon grid takes its degree increments from the STORED `XLAT`/`XLONG`, not the
/// `DX`/`DY` attributes — without them the reconstruction could not be bit-identical.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ManifestAnchor {
    /// Fractional 0-based grid index of the anchor (the grid-centre cell).
    pub ref_i: f64,
    pub ref_j: f64,
    /// Geodetic anchor read from the stored `XLAT`/`XLONG` at this timestep.
    pub ref_lat_deg: f64,
    pub ref_lon_deg: f64,
    /// Plane-unit spacing the georef used (metres; degrees for `MAP_PROJ = 6`).
    pub dx: f64,
    pub dy: f64,
}

/// One timestep entry in a run manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManifestTimestep {
    /// The unique cache key `YYYYMMDD_HHMM` (see [`time_stamp`]) — the dedup key and
    /// the `t{key}.ssb` file stem. Format v1 keyed on `hhmm` alone, which collided
    /// across days on >24 h runs (M0-review MINOR-2).
    pub key: String,
    pub hhmm: u16,
    pub file: String,
    pub time_iso: Option<String>,
    /// The per-volume quantization scales for this timestep (design: run.json
    /// carries the quantization scales).
    pub quant: ChannelQuant,
    pub ssb_bytes: u64,
    /// Source-wrfout identity at ingest time (byte length), the staleness gate: a
    /// re-run WRF writing over the same path changes it, so a cache hit can detect
    /// that the brick no longer matches the file. `None` on a pre-WS3 v2 manifest
    /// (serde default) — treated as STALE-once, self-healing on the next render.
    #[serde(default)]
    pub source_bytes: Option<u64>,
    /// Source-wrfout mtime (unix seconds) at ingest time (see `source_bytes`).
    #[serde(default)]
    pub source_mtime_unix: Option<i64>,
    /// The per-timestep georef anchor (see [`ManifestAnchor`]). `None` on a pre-WS3
    /// v2 manifest — the cached path then falls back to the CEN_LAT/CEN_LON
    /// approximation ([`crate::frame::GridGeoref::from_params_center`]) as before.
    #[serde(default)]
    pub anchor: Option<ManifestAnchor>,
}

/// Whether a cached timestep entry is FRESH against the current source wrfout: both
/// identity fields recorded at ingest time AND matching the file's current byte length
/// and mtime. `None` fields (a pre-WS3 v2 manifest) are stale — the caller re-ingests
/// once and the fields are recorded. Pure (the freshness truth table is unit-tested);
/// the caller supplies the `fs::metadata` values.
pub fn cache_entry_is_fresh(entry: &ManifestTimestep, src_bytes: u64, src_mtime_unix: i64) -> bool {
    entry.source_bytes == Some(src_bytes) && entry.source_mtime_unix == Some(src_mtime_unix)
}

/// The `run.json` manifest indexing every timestep of a run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunManifest {
    pub format_version: u32,
    pub run_id: String,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub z_min_m: f64,
    pub dz_m: f64,
    pub temperature_encoding: String,
    pub quant_dynamic_range: f64,
    pub channels_3d: Vec<String>,
    pub planes_2d: Vec<String>,
    pub projection: ManifestProjection,
    pub timesteps: Vec<ManifestTimestep>,
}

impl RunManifest {
    /// The manifest path for a run: `{cache_dir}/{run_id}/run.json`.
    pub fn path(cache_dir: &Path, run_id: &str) -> PathBuf {
        run_dir(cache_dir, run_id).join("run.json")
    }

    /// Load an existing `run.json` manifest from disk.
    ///
    /// The `format_version` is checked FIRST at the JSON-value level, so a manifest
    /// written by ANY other format version — older or newer, whatever its schema —
    /// is refused with the regenerate-the-cache remedy instead of a confusing serde
    /// field error (or, worse, a lucky schema overlap parsing to wrong data).
    pub fn load(path: &Path) -> Result<Self, BrickError> {
        let text = std::fs::read_to_string(path)?;
        let value: serde_json::Value = serde_json::from_str(&text)?;
        let found = value
            .get("format_version")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if found != SSB_FORMAT_VERSION as u64 {
            return Err(BrickError::UnsupportedManifestVersion {
                found,
                expected: SSB_FORMAT_VERSION,
            });
        }
        Ok(serde_json::from_value(value)?)
    }

    /// Load an existing manifest or build a fresh one with these run fields.
    ///
    /// An EXISTING manifest must describe the same grid, z-axis, and projection SHAPE
    /// (`map_proj`/truelats/`stand_lon`/`dx`/`dy`) as the file being ingested;
    /// otherwise a [`BrickError::CacheMismatch`] with a remedy is returned — a run_id
    /// collision across different domains would silently mix bricks of different
    /// grids into one run (the silent-wrong-output class). `CEN_LAT`/`CEN_LON` are
    /// deliberately NOT compared: a moving nest re-centres between timesteps by
    /// design, and the per-timestep [`ManifestAnchor`] carries that.
    #[allow(clippy::too_many_arguments)]
    pub fn load_or_new(
        path: &Path,
        run_id: &str,
        nx: usize,
        ny: usize,
        nz: usize,
        z_min_m: f64,
        dz_m: f64,
        planes_2d: Vec<String>,
        projection: ManifestProjection,
    ) -> Result<Self, BrickError> {
        if path.is_file() {
            let manifest = match Self::load(path) {
                Ok(m) => m,
                // An OLD-FORMAT manifest found during INGEST is a regenerable cache
                // being re-populated by a newer build: SUPERSEDE it with a fresh
                // manifest instead of erroring (the v2 -> v3 self-heal; owner-reported
                // "Render failed: unsupported .ssb version: 2"). The old-format .ssb
                // files alongside it are refused by `read_ssb` and re-ingested per
                // timestep through the same path; only the READ paths (a cached
                // run.json open) keep the hard remedy-bearing refusal, because there
                // is no source wrfout there to re-ingest from.
                Err(BrickError::UnsupportedManifestVersion { .. }) => {
                    return Ok(Self::new_manifest(
                        run_id, nx, ny, nz, z_min_m, dz_m, planes_2d, projection,
                    ));
                }
                Err(e) => return Err(e),
            };
            let p = &manifest.projection;
            let shape_ok = p.map_proj == projection.map_proj
                && p.truelat1_deg == projection.truelat1_deg
                && p.truelat2_deg == projection.truelat2_deg
                && p.stand_lon_deg == projection.stand_lon_deg
                && p.dx_m == projection.dx_m
                && p.dy_m == projection.dy_m;
            if manifest.nx != nx
                || manifest.ny != ny
                || manifest.nz != nz
                || manifest.z_min_m != z_min_m
                || manifest.dz_m != dz_m
                || !shape_ok
            {
                return Err(BrickError::CacheMismatch(format!(
                    "the existing manifest {} was built for a different grid/projection \
                     (manifest {}x{}x{} map_proj {} dx {} vs source {}x{}x{} map_proj {} \
                     dx {}); delete that run's cache directory to re-ingest, or use a \
                     different cache dir / run id",
                    path.display(),
                    manifest.nx,
                    manifest.ny,
                    manifest.nz,
                    p.map_proj,
                    p.dx_m,
                    nx,
                    ny,
                    nz,
                    projection.map_proj,
                    projection.dx_m
                )));
            }
            return Ok(manifest);
        }
        Ok(Self::new_manifest(
            run_id, nx, ny, nz, z_min_m, dz_m, planes_2d, projection,
        ))
    }

    /// A fresh empty manifest at the CURRENT format version (the no-existing-manifest
    /// and the superseded-old-format paths of [`Self::load_or_new`]).
    #[allow(clippy::too_many_arguments)]
    fn new_manifest(
        run_id: &str,
        nx: usize,
        ny: usize,
        nz: usize,
        z_min_m: f64,
        dz_m: f64,
        planes_2d: Vec<String>,
        projection: ManifestProjection,
    ) -> Self {
        Self {
            format_version: SSB_FORMAT_VERSION,
            run_id: run_id.to_string(),
            nx,
            ny,
            nz,
            z_min_m,
            dz_m,
            temperature_encoding: "celsius_f16".to_string(),
            quant_dynamic_range: QUANT_DYNAMIC_RANGE,
            channels_3d: CHANNELS_3D.iter().map(|s| s.to_string()).collect(),
            planes_2d,
            projection,
            timesteps: Vec::new(),
        }
    }

    /// Register (or replace, by full-datetime `key`) a timestep and keep the list
    /// sorted chronologically (the `YYYYMMDD_HHMM` key sorts lexicographically =
    /// chronologically).
    pub fn register_timestep(&mut self, entry: ManifestTimestep) {
        self.timesteps.retain(|t| t.key != entry.key);
        self.timesteps.push(entry);
        self.timesteps.sort_by(|a, b| a.key.cmp(&b.key));
    }

    /// Write the manifest to `path` (pretty JSON).
    pub fn save(&self, path: &Path) -> Result<(), BrickError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(path, text)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static UNIQ: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> PathBuf {
        let n = UNIQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("simsat-ssb-{}-{}", std::process::id(), n));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn log_quant_round_trip_is_code_stable_and_bounded() {
        // Values spanning the window.
        let values: Vec<f32> = (0..500)
            .map(|i| 1.0e-4 * 1.03f32.powi(i))
            .chain([0.0, -1.0])
            .collect();
        let (quant, codes) = encode_log_channel(&values);
        // Re-encoding a decoded value must reproduce the SAME code (idempotent).
        for &code in &codes {
            let v = quant.decode(code);
            assert_eq!(quant.encode(v), code, "code {code} not stable");
        }
        // Relative error bounded for values inside [vmin, vmax].
        let mut worst_rel = 0.0f64;
        for &v in &values {
            let v = v as f64;
            if v > quant.vmin && v <= quant.vmax {
                let dq = quant.decode(quant.encode(v as f32)) as f64;
                worst_rel = worst_rel.max((dq - v).abs() / v);
            }
        }
        assert!(worst_rel < 0.03, "worst relative quant error {worst_rel}");
        // Exact zero stays code 0 -> exactly 0.0.
        assert_eq!(quant.encode(0.0), 0);
        assert_eq!(quant.decode(0), 0.0);
    }

    #[test]
    fn log_quant_all_zero_channel() {
        let (quant, codes) = encode_log_channel(&[0.0, 0.0, 0.0]);
        assert_eq!(quant.vmax, 0.0);
        assert!(codes.iter().all(|&c| c == 0));
        assert_eq!(quant.decode(200), 0.0);
    }

    #[test]
    fn f16_known_bit_patterns() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-1.0), 0xbc00);
        assert_eq!(f32_to_f16_bits(2.0), 0x4000);
        assert_eq!(f32_to_f16_bits(0.5), 0x3800);
        assert_eq!(f16_bits_to_f32(0x3c00), 1.0);
        assert_eq!(f16_bits_to_f32(0x4000), 2.0);
        assert_eq!(f16_bits_to_f32(0xbc00), -1.0);
    }

    #[test]
    fn temperature_f16_meets_tenth_kelvin_fidelity() {
        // Whole atmospheric range in Celsius: -90 .. +60 C.
        let mut kelvin = Vec::new();
        let mut t = -90.0f64;
        while t <= 60.0 {
            kelvin.push((t + CELSIUS_OFFSET_K) as f32);
            t += 0.05;
        }
        let bits = encode_temperature_celsius(&kelvin);
        let back = decode_temperature_kelvin(&bits);
        let mut worst = 0.0f64;
        for (a, b) in kelvin.iter().zip(back.iter()) {
            worst = worst.max((*a as f64 - *b as f64).abs());
        }
        assert!(
            worst < 0.1,
            "worst temperature error {worst} K exceeds 0.1 K"
        );
    }

    fn tiny_brick() -> VolumeBrick {
        let (nx, ny, nz) = (3usize, 2usize, 4usize);
        let cells_3d = nx * ny * nz;
        let cells_2d = nx * ny;
        let ext_liquid_f32: Vec<f32> = (0..cells_3d).map(|i| (i as f32) * 1.0e-3).collect();
        let ext_ice_f32: Vec<f32> = (0..cells_3d).map(|i| (i as f32) * 2.0e-4).collect();
        let ext_precip_f32: Vec<f32> = vec![0.0; cells_3d];
        let tau_f32: Vec<f32> = (0..cells_3d).map(|i| 0.1 * i as f32).collect();
        let qv_f32: Vec<f32> = (0..cells_3d).map(|i| 1.0e-3 + 1.0e-5 * i as f32).collect();
        let kelvin: Vec<f32> = (0..cells_3d).map(|i| 300.0 - 0.3 * i as f32).collect();

        let (ql, ext_liquid) = encode_log_channel(&ext_liquid_f32);
        let (qi, ext_ice) = encode_log_channel(&ext_ice_f32);
        let (qp, ext_precip) = encode_log_channel(&ext_precip_f32);
        let (qt, tau_up) = encode_log_channel(&tau_f32);
        let (qv, qvapor) = encode_log_channel(&qv_f32);
        let mut map = BTreeMap::new();
        map.insert("ext_liquid".to_string(), ql);
        map.insert("ext_ice".to_string(), qi);
        map.insert("ext_precip".to_string(), qp);
        map.insert("tau_up".to_string(), qt);
        map.insert("qvapor".to_string(), qv);

        VolumeBrick {
            nx,
            ny,
            nz,
            z_min_m: 0.0,
            dz_m: 250.0,
            time_iso: Some("2018-10-10T12:00:00Z".to_string()),
            quant: ChannelQuant(map),
            ext_liquid,
            ext_ice,
            ext_precip,
            tau_up,
            qvapor,
            temperature_f16: encode_temperature_celsius(&kelvin),
            hgt: (0..cells_2d).map(|i| 10.0 * i as f32).collect(),
            landmask: vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            tsk: (0..cells_2d).map(|i| 295.0 + i as f32).collect(),
            u10: vec![1.0; cells_2d],
            v10: vec![-2.0; cells_2d],
            snowh: Some(vec![0.0; cells_2d]),
            ivgtyp: None,
        }
    }

    #[test]
    fn ssb_round_trip_reproduces_every_byte() {
        let brick = tiny_brick();
        let dir = temp_dir();
        let path = dir.join(brick_file_name("20181010_1200"));
        let bytes = write_ssb(&path, &brick).unwrap();
        assert!(bytes > 0);
        let back = read_ssb(&path).unwrap();
        // Quantized codes and f16 bits must survive bit-for-bit.
        assert_eq!(brick, back);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn ssb_rejects_bad_magic() {
        let dir = temp_dir();
        let path = dir.join("garbage.ssb");
        std::fs::write(&path, b"NOTSSBanythinggoeshere").unwrap();
        assert!(matches!(read_ssb(&path), Err(BrickError::BadMagic(_))));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn manifest_load_or_new_and_register() {
        let dir = temp_dir();
        let path = RunManifest::path(&dir, "michael_d01");
        let projection = ManifestProjection {
            map_proj: 1,
            truelat1_deg: 30.0,
            truelat2_deg: 60.0,
            stand_lon_deg: -85.0,
            cen_lat_deg: 27.0,
            cen_lon_deg: -85.0,
            dx_m: 12000.0,
            dy_m: 12000.0,
        };
        let mut manifest = RunManifest::load_or_new(
            &path,
            "michael_d01",
            10,
            10,
            80,
            0.0,
            250.0,
            vec!["hgt".into(), "landmask".into()],
            projection,
        )
        .unwrap();
        assert!(manifest.timesteps.is_empty());
        let quant = ChannelQuant(BTreeMap::new());
        let entry = |key: &str, hhmm: u16, iso: &str, bytes: u64| ManifestTimestep {
            key: key.to_string(),
            hhmm,
            file: brick_file_name(key),
            time_iso: Some(iso.to_string()),
            quant: quant.clone(),
            ssb_bytes: bytes,
            source_bytes: None,
            source_mtime_unix: None,
            anchor: None,
        };
        // Two timesteps at the SAME wall-clock HHMM (1200) but DIFFERENT days must
        // NOT collide (the M0-review MINOR-2 regression): distinct keys, both kept.
        manifest.register_timestep(entry("20181011_1200", 1200, "2018-10-11T12:00:00Z", 111));
        manifest.register_timestep(entry("20181010_1215", 1215, "2018-10-10T12:15:00Z", 100));
        manifest.register_timestep(entry("20181010_1200", 1200, "2018-10-10T12:00:00Z", 90));
        // Re-registering the SAME key replaces, not duplicates; list stays sorted.
        manifest.register_timestep(entry("20181010_1200", 1200, "2018-10-10T12:00:00Z", 95));
        manifest.save(&path).unwrap();
        let reloaded = RunManifest::load_or_new(
            &path,
            "michael_d01",
            10,
            10,
            80,
            0.0,
            250.0,
            vec![],
            projection,
        )
        .unwrap();
        // Three distinct keys survive (same-HHMM different-day did not collide),
        // sorted chronologically by key.
        assert_eq!(reloaded.timesteps.len(), 3);
        assert_eq!(reloaded.timesteps[0].key, "20181010_1200");
        assert_eq!(reloaded.timesteps[0].ssb_bytes, 95);
        assert_eq!(reloaded.timesteps[1].key, "20181010_1215");
        assert_eq!(reloaded.timesteps[2].key, "20181011_1200");
        assert_eq!(reloaded.timesteps[2].hhmm, 1200);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn time_stamp_keys_on_full_datetime_and_falls_back_to_hhmm() {
        assert_eq!(
            time_stamp(Some("2025-06-21T02:15:00Z"), 215),
            "20250621_0215"
        );
        assert_eq!(
            brick_file_name_for(Some("2025-06-21T02:15:00Z"), 215),
            "t20250621_0215.ssb"
        );
        // WRF underscore separator is handled (digits are extracted).
        assert_eq!(
            time_stamp(Some("1974-04-03_10:00:00"), 1000),
            "19740403_1000"
        );
        // No ISO time (source lacks Times) -> 4-digit HHMM fallback.
        assert_eq!(time_stamp(None, 215), "0215");
        assert_eq!(brick_file_name_for(None, 1200), "t1200.ssb");
        // Same HHMM, different day -> different keys (no collision).
        assert_ne!(
            time_stamp(Some("2025-06-21T02:15:00Z"), 215),
            time_stamp(Some("2025-06-22T02:15:00Z"), 215)
        );
    }

    fn test_projection() -> ManifestProjection {
        ManifestProjection {
            map_proj: 1,
            truelat1_deg: 30.0,
            truelat2_deg: 60.0,
            stand_lon_deg: -97.5,
            cen_lat_deg: 39.0,
            cen_lon_deg: -97.5,
            dx_m: 3000.0,
            dy_m: 3000.0,
        }
    }

    fn test_entry() -> ManifestTimestep {
        ManifestTimestep {
            key: "20250621_0215".to_string(),
            hhmm: 215,
            file: brick_file_name("20250621_0215"),
            time_iso: Some("2025-06-21T02:15:00Z".to_string()),
            quant: ChannelQuant(BTreeMap::new()),
            ssb_bytes: 100,
            source_bytes: Some(2_047_845_048),
            source_mtime_unix: Some(1_750_470_000),
            anchor: Some(ManifestAnchor {
                ref_i: 399.5,
                ref_j: 399.5,
                ref_lat_deg: 46.9,
                ref_lon_deg: -97.6,
                dx: 250.0,
                dy: 250.0,
            }),
        }
    }

    /// WS3 staleness gate: the pure freshness truth table.
    #[test]
    fn cache_entry_freshness_truth_table() {
        let entry = test_entry();
        let (b, m) = (2_047_845_048u64, 1_750_470_000i64);
        // Recorded and matching -> fresh.
        assert!(cache_entry_is_fresh(&entry, b, m));
        // Either identity moved -> stale (a re-run WRF wrote over the same path).
        assert!(!cache_entry_is_fresh(&entry, b + 1, m));
        assert!(!cache_entry_is_fresh(&entry, b, m + 1));
        assert!(!cache_entry_is_fresh(&entry, b - 1, m - 1));
        // A pre-WS3 v2 manifest (fields absent -> None) is STALE-once: never fresh.
        let mut old = entry.clone();
        old.source_bytes = None;
        assert!(!cache_entry_is_fresh(&old, b, m));
        let mut old2 = entry.clone();
        old2.source_mtime_unix = None;
        assert!(!cache_entry_is_fresh(&old2, b, m));
    }

    /// WS3: the new optional manifest fields round-trip, and a pre-WS3 v2 manifest
    /// JSON (without them) still deserializes with `None` defaults (self-heal path).
    #[test]
    fn manifest_timestep_new_fields_round_trip_and_default() {
        let entry = test_entry();
        let json = serde_json::to_string(&entry).unwrap();
        let back: ManifestTimestep = serde_json::from_str(&json).unwrap();
        assert_eq!(back, entry);
        // Strip the WS3 fields to simulate a pre-WS3 v2 manifest entry.
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = value.as_object_mut().unwrap();
        obj.remove("source_bytes");
        obj.remove("source_mtime_unix");
        obj.remove("anchor");
        let old: ManifestTimestep = serde_json::from_value(value).unwrap();
        assert_eq!(old.source_bytes, None);
        assert_eq!(old.source_mtime_unix, None);
        assert_eq!(old.anchor, None);
        assert_eq!(old.key, entry.key);
    }

    /// WS3: a run.json of ANY other format version is refused FIRST, with the
    /// regenerate remedy — before serde ever sees the schema.
    #[test]
    fn manifest_load_refuses_other_format_versions_with_a_remedy() {
        let dir = temp_dir();
        let path = dir.join("run.json");
        // A minimal v1-style manifest: only the version key matters — the check must
        // fire before any schema field is touched.
        std::fs::write(&path, r#"{ "format_version": 1, "run_id": "old" }"#).unwrap();
        let err = RunManifest::load(&path).unwrap_err();
        match &err {
            BrickError::UnsupportedManifestVersion { found, expected } => {
                assert_eq!(*found, 1);
                assert_eq!(*expected, SSB_FORMAT_VERSION);
            }
            other => panic!("expected UnsupportedManifestVersion, got {other:?}"),
        }
        let msg = err.to_string();
        assert!(
            msg.contains("regenerable") && msg.contains("delete"),
            "remedy missing from: {msg}"
        );
        // A manifest with NO version key at all reports found 0 (same refusal).
        std::fs::write(&path, r#"{ "run_id": "ancient" }"#).unwrap();
        assert!(matches!(
            RunManifest::load(&path),
            Err(BrickError::UnsupportedManifestVersion { found: 0, .. })
        ));
        // The v3 snow-optics bump: a v2 manifest (schema-compatible, but its bricks
        // carry the inflated pre-fix snow extinction) is refused the same way — the
        // remedy is a re-ingest from the source wrfout, never a silent reuse.
        std::fs::write(
            &path,
            r#"{ "format_version": 2, "run_id": "pre_snow_fix" }"#,
        )
        .unwrap();
        assert!(matches!(
            RunManifest::load(&path),
            Err(BrickError::UnsupportedManifestVersion { found: 2, .. })
        ));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The v2 -> v3 INGEST self-heal (owner-reported "Render failed: unsupported .ssb
    /// version: 2"): `load_or_new` SUPERSEDES an old-format manifest with a fresh one
    /// at the current version instead of propagating the read refusal — during ingest
    /// the source wrfout is present, so the regenerable cache regenerates. The READ
    /// paths (`RunManifest::load` direct, a cached run.json open) keep the hard
    /// remedy-bearing refusal, covered by the test above.
    #[test]
    fn load_or_new_supersedes_an_old_format_manifest() {
        let dir = temp_dir();
        let path = RunManifest::path(&dir, "old_run");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, r#"{ "format_version": 2, "run_id": "old_run" }"#).unwrap();
        let manifest = RunManifest::load_or_new(
            &path,
            "old_run",
            100,
            80,
            40,
            0.0,
            250.0,
            vec![],
            test_projection(),
        )
        .expect("an old-format manifest must be superseded, not refused, during ingest");
        assert_eq!(manifest.format_version, SSB_FORMAT_VERSION);
        assert!(
            manifest.timesteps.is_empty(),
            "fresh manifest, no timesteps"
        );
        assert_eq!(manifest.nx, 100);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// WS3: `load_or_new` refuses to hand back a manifest built for a DIFFERENT grid
    /// or projection shape (a run_id collision across domains), but tolerates a
    /// moved CEN_LAT/CEN_LON (a moving nest re-centres by design).
    #[test]
    fn load_or_new_gates_grid_and_projection_shape() {
        let dir = temp_dir();
        let path = RunManifest::path(&dir, "run_a");
        let projection = test_projection();
        let manifest =
            RunManifest::load_or_new(&path, "run_a", 100, 80, 40, 0.0, 250.0, vec![], projection)
                .unwrap();
        manifest.save(&path).unwrap();
        // Same grid + shape, DIFFERENT centre (moving nest): accepted.
        let mut moved = projection;
        moved.cen_lat_deg = 40.2;
        moved.cen_lon_deg = -96.8;
        assert!(
            RunManifest::load_or_new(&path, "run_a", 100, 80, 40, 0.0, 250.0, vec![], moved)
                .is_ok(),
            "a moved nest centre must not be a cache mismatch"
        );
        // Different grid dims: refused with a remedy.
        let err =
            RunManifest::load_or_new(&path, "run_a", 200, 80, 40, 0.0, 250.0, vec![], projection)
                .unwrap_err();
        assert!(
            matches!(&err, BrickError::CacheMismatch(_)),
            "expected CacheMismatch, got {err:?}"
        );
        assert!(err.to_string().contains("delete"), "{err}");
        // Different projection shape (dx): refused.
        let mut other_dx = projection;
        other_dx.dx_m = 12000.0;
        assert!(matches!(
            RunManifest::load_or_new(&path, "run_a", 100, 80, 40, 0.0, 250.0, vec![], other_dx),
            Err(BrickError::CacheMismatch(_))
        ));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// WS3 read_ssb hardening: a header describing huge/overflowing dims returns a
    /// clean error (the unchecked product previously overflow-panicked in a debug
    /// build), and an implausibly large described payload is refused.
    #[test]
    fn read_ssb_rejects_huge_dims_cleanly() {
        let dir = temp_dir();
        let brick = tiny_brick();
        let good_path = dir.join("good.ssb");
        write_ssb(&good_path, &brick).unwrap();
        let good = std::fs::read(&good_path).unwrap();
        // Splice a corrupt header (overflowing dims) into an otherwise valid file.
        let mut header: BrickHeader = {
            let len = u32::from_le_bytes(good[8..12].try_into().unwrap()) as usize;
            serde_json::from_slice(&good[12..12 + len]).unwrap()
        };
        header.nx = usize::MAX / 2;
        header.ny = usize::MAX / 2;
        header.nz = 4;
        let hj = serde_json::to_vec(&header).unwrap();
        let mut evil = Vec::new();
        evil.extend_from_slice(&good[0..8]);
        evil.extend_from_slice(&(hj.len() as u32).to_le_bytes());
        evil.extend_from_slice(&hj);
        let old_len = u32::from_le_bytes(good[8..12].try_into().unwrap()) as usize;
        evil.extend_from_slice(&good[12 + old_len..]);
        let evil_path = dir.join("huge.ssb");
        std::fs::write(&evil_path, &evil).unwrap();
        let err = read_ssb(&evil_path).expect_err("huge dims must be a clean error");
        assert!(matches!(err, BrickError::Truncated(_)), "{err:?}");
        // Non-overflowing but implausible (over the payload ceiling): also refused.
        header.nx = 1 << 20;
        header.ny = 1 << 20;
        header.nz = 1 << 10;
        let hj = serde_json::to_vec(&header).unwrap();
        let mut evil2 = Vec::new();
        evil2.extend_from_slice(&good[0..8]);
        evil2.extend_from_slice(&(hj.len() as u32).to_le_bytes());
        evil2.extend_from_slice(&hj);
        evil2.extend_from_slice(&good[12 + old_len..]);
        let evil2_path = dir.join("implausible.ssb");
        std::fs::write(&evil2_path, &evil2).unwrap();
        assert!(matches!(
            read_ssb(&evil2_path),
            Err(BrickError::Truncated(_))
        ));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// WS3 read_ssb hardening: a payload truncated mid-stream is a clean error.
    #[test]
    fn read_ssb_rejects_truncated_payload_cleanly() {
        let dir = temp_dir();
        let brick = tiny_brick();
        let path = dir.join("whole.ssb");
        write_ssb(&path, &brick).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        // Chop the compressed payload roughly in half (keep the header intact).
        let header_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let payload_start = 12 + header_len;
        let cut = payload_start + (bytes.len() - payload_start) / 2;
        let chopped_path = dir.join("chopped.ssb");
        std::fs::write(&chopped_path, &bytes[..cut]).unwrap();
        assert!(
            read_ssb(&chopped_path).is_err(),
            "a truncated payload must error, not panic or return a partial brick"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn log_quant_encode_guards_non_finite() {
        // A channel with a finite max and a +Inf outlier: from_values excludes the
        // Inf when deriving vmax, so the Inf survives into encode(). It must map to
        // code 0 without an overflow panic (M0-review MINOR-1).
        let (quant, codes) = encode_log_channel(&[1.0, f32::INFINITY]);
        assert!(quant.vmax > 0.0 && quant.vmax.is_finite());
        assert_eq!(codes[1], 0, "+Inf must encode to code 0");
        assert_eq!(quant.encode(f32::INFINITY), 0);
        assert_eq!(quant.encode(f32::NEG_INFINITY), 0);
        assert_eq!(quant.encode(f32::NAN), 0);
    }
}
