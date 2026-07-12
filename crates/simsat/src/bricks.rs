//! The `.ssb` on-disk volume-brick format (design doc section 2).
//!
//! One `.ssb` file holds one WRF timestep resampled onto an affine vertical axis
//! (`z(k) = z_min_m + k*dz_m`, MSL) over the native WRF `(i, j)` horizontal grid
//! (NO decimation/windowing in M0 — that is M7).
//!
//! Payload extinction channels are log-quantized u8 3-D channels with per-volume
//! scales in the header: `ext_liquid` (cloud-liquid extinction, m^-1, from
//! `QCLOUD`), `ext_ice` (small-ice extinction, `QICE` only since SSB v3),
//! `ext_snow` (since SSB v4: a QSNOW-only AUXILIARY SUBSET of `ext_precip`, not an
//! independently additive phase), `ext_precip` (the legacy total large-particle
//! extinction: `QRAIN + QGRAUP + QSNOW`, each converted at
//! its own optics before the sum), and `tau_up`
//! (cumulative optical depth from brick-top down to each level, precomputed at
//! ingest from liquid + ice + total precip only; feeds cloud ambient, the IR fast
//! path, and shadows). Keeping snow duplicated inside total precip preserves the
//! exact legacy/GPU/IR path while allowing fractional-cloud rendering to isolate
//! the snow share as `precip - snow + scaled_snow`.
//!
//! The QVAPOR channel (owner decision 6 — a full channel now, for a later 6.2 um
//! water-vapor IR band) remains its OWN log-quantized u8 channel, `qvapor`
//! (mixing ratio kg/kg). Representation choice (recorded in the manifest as
//! `qvapor`): a u8 log channel, not an f16 plane. Rationale — the bandwidth-lean
//! rule (owner decision 1) favors u8; a WV IR band is coarse-tolerance, and the
//! fixed log window still gives it usable dynamic range. f16 was the alternative,
//! rejected on bandwidth.
//!
//! SSB v4+ also carries `cloud_fraction`, a LINEAR u8 channel (`round(clamp(f,
//! 0,1)*255)`, with every finite positive value floored to code 1 so wispy tails
//! survive quantization). `has_cloud_fraction` records provenance. When a source
//! has no trusted coverage field, every byte is 255 and the flag is false, making
//! the fallback explicitly equivalent to full-cell coverage.
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
//! Experimental SSB v7 ScienceCloudF16 files append four `nx*ny*nz` u16
//! binary16(log2(beta)) channels after those planes; the header names them and the
//! explicit profile-aware reader accepts both v6 and v7. Production writes remain v6.
//! A sibling `run.json` (see `RunManifest`) indexes all timesteps of a run.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

use flate2::Compression;
use flate2::bufread::ZlibDecoder;
use flate2::write::ZlibEncoder;
use serde::{Deserialize, Serialize};

use crate::{SSB_FORMAT_VERSION, SSB_MAGIC};

/// Format epoch used only by the opt-in cloud-extinction science profile. The
/// production/default [`write_ssb`] path remains SSB v6.
pub const SCIENCE_F16_FORMAT_VERSION: u32 = 7;

/// Lowest and highest physical log2 extinction accepted by ScienceCloudF16.
/// This spans about `8.27e-25 .. 16 m^-1`, preserving the very small positive
/// NSSL advection tails seen in real files while keeping corrupt finite half-floats
/// distinguishable from valid science data. Smaller finite positives canonicalize
/// to the exact clear-sky sentinel because their column OD is physically negligible.
pub const SCIENCE_LOG2_MIN: f32 = -80.0;
pub const SCIENCE_LOG2_MAX: f32 = 4.0;
pub const SCIENCE_ZERO_BITS: u16 = 0xfc00; // binary16 negative infinity

/// Cloud-extinction storage selected by an SSB writer/reader.
///
/// `CompactU8` is the shipped SSB v6 representation. `ScienceCloudF16` is a
/// deliberately narrow experiment: the three additive cloud-extinction channels
/// plus the snow-only auxiliary required to split snow from rain/graupel in thermal
/// transfer are repeated as log2-f16 values. Temperature, water vapour, cloud
/// fraction, tau-up, and surface planes retain their compact v6 encodings.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StorageProfile {
    #[default]
    CompactU8,
    ScienceCloudF16,
}

impl StorageProfile {
    pub const ALL: [Self; 2] = [Self::CompactU8, Self::ScienceCloudF16];

    /// Stable CLI/Python/settings/cache-identity token.
    pub const fn slug(self) -> &'static str {
        match self {
            Self::CompactU8 => "compact-u8",
            Self::ScienceCloudF16 => "science-cloud-f16",
        }
    }

    pub const fn format_version(self) -> u32 {
        match self {
            Self::CompactU8 => SSB_FORMAT_VERSION,
            Self::ScienceCloudF16 => SCIENCE_F16_FORMAT_VERSION,
        }
    }

    /// Extra cache namespace. CompactU8 intentionally preserves the historical
    /// cache path; ScienceCloudF16 can never collide with it.
    pub const fn cache_namespace(self) -> Option<&'static str> {
        match self {
            Self::CompactU8 => None,
            Self::ScienceCloudF16 => Some("storage-science-cloud-f16-v1"),
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "compact" | "compact-u8" | "u8" => Some(Self::CompactU8),
            "science" | "science-f16" | "science-cloud-f16" | "f16" => Some(Self::ScienceCloudF16),
            _ => None,
        }
    }
}

/// Post-resample cloud extinction (m^-1) supplied to the experimental
/// ScienceCloudF16 writer. `ext_snow` is the snow-only subset of `ext_precip`, not
/// another additive species; carrying both is required for the v6 thermal split.
#[derive(Debug, Clone, Copy)]
pub struct ScienceCloudExtinction<'a> {
    pub ext_liquid: &'a [f32],
    pub ext_ice: &'a [f32],
    pub ext_snow: &'a [f32],
    pub ext_precip: &'a [f32],
}

/// Encoded log2-f16 extension read from a ScienceCloudF16 brick.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ScienceCloudF16Payload {
    pub ext_liquid: Vec<u16>,
    pub ext_ice: Vec<u16>,
    pub ext_snow: Vec<u16>,
    pub ext_precip: Vec<u16>,
}

impl ScienceCloudF16Payload {
    /// Encode the science channels without imposing a grid shape.
    /// Writers validate their lengths against the brick before touching disk.
    pub fn encode(science: ScienceCloudExtinction<'_>) -> Self {
        let encode = |values: &[f32]| values.iter().map(|&v| encode_log2_f16(v)).collect();
        Self {
            ext_liquid: encode(science.ext_liquid),
            ext_ice: encode(science.ext_ice),
            ext_snow: encode(science.ext_snow),
            ext_precip: encode(science.ext_precip),
        }
    }

    /// Production encoder. Exact zero is the sole clear-sky sentinel. Negative,
    /// NaN/Inf, and out-of-contract finite values are errors rather than silently
    /// becoming plausible cloud data.
    pub fn try_encode(science: ScienceCloudExtinction<'_>) -> Result<Self, BrickError> {
        let encode = |name: &str, values: &[f32]| -> Result<Vec<u16>, BrickError> {
            values
                .iter()
                .enumerate()
                .map(|(index, &value)| try_encode_log2_f16(name, index, value))
                .collect()
        };
        Ok(Self {
            ext_liquid: encode("ext_liquid", science.ext_liquid)?,
            ext_ice: encode("ext_ice", science.ext_ice)?,
            ext_snow: encode("ext_snow", science.ext_snow)?,
            ext_precip: encode("ext_precip", science.ext_precip)?,
        })
    }

    /// Decode one encoded channel back to physical extinction (m^-1).
    pub fn decode_channel(bits: &[u16]) -> Vec<f32> {
        bits.iter().map(|&bits| decode_log2_f16(bits)).collect()
    }
}

/// A brick read through the profile-aware API. Compact v6 bricks have no
/// `science_cloud_f16` extension; v7 ScienceCloudF16 bricks preserve it exactly.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfiledVolumeBrick {
    pub profile: StorageProfile,
    pub brick: VolumeBrick,
}

/// Fixed dynamic-range window for log quantization: codes 1..=255 span five
/// decades below the per-volume peak. This bounds the per-code relative step to
/// `(1e5)^(1/254) - 1 ~= 4.7%` (nearest-code error ~= 2.3%), independent of data.
pub const QUANT_DYNAMIC_RANGE: f64 = 1.0e5;

/// Kelvin<->Celsius offset used by the f16 temperature encoding.
pub const CELSIUS_OFFSET_K: f64 = 273.15;

/// Sanity ceiling on the decompressed payload size an `.ssb` header may describe
/// (64 GiB — an order of magnitude above any real brick; the Enderlin 800x800x80
/// v4/v5 payload is ~0.48 GB). Bounds the [`read_ssb`] decompress so corrupt dims cannot
/// balloon memory (checked, never trusted).
pub const MAX_SSB_PAYLOAD_BYTES: u64 = 64 << 30;

/// The ordered 3-D u8 channel names. All except `cloud_fraction` use the
/// per-volume log quantization map; cloud fraction is linear by definition.
pub const CHANNELS_3D: [&str; 7] = [
    "ext_liquid",
    "ext_ice",
    "ext_snow",
    "ext_precip",
    "tau_up",
    "qvapor",
    "cloud_fraction",
];

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

/// The per-volume log-channel scales, keyed by channel name. The linear
/// `cloud_fraction` channel deliberately has no entry.
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

/// Encode model cloud coverage to SSB v4+'s linear u8 representation.
/// Every finite positive sample is kept at code 1 or above so a wispy positive
/// tail cannot round to zero and be mistaken for missing model coverage. Non-finite
/// and non-positive samples encode as zero; source-level absence is represented
/// separately by an all-255 channel plus `has_cloud_fraction = false`.
pub fn encode_cloud_fraction(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .map(|&f| {
            if !f.is_finite() || f <= 0.0 {
                0
            } else {
                ((f.min(1.0) * 255.0).round() as u8).max(1)
            }
        })
        .collect()
}

/// Decode one SSB v4+ linear cloud-fraction code.
#[inline]
pub fn decode_cloud_fraction(code: u8) -> f32 {
    code as f32 / 255.0
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

/// Encode non-negative extinction as binary16(log2(beta)). Negative, zero, and
/// non-finite inputs become the -infinity zero sentinel. Encoding the logarithm
/// rather than beta itself retains wispy positive tails without the CompactU8
/// per-volume positive floor and without raw-f16 underflow near 1e-8 m^-1.
#[inline]
pub fn encode_log2_f16(value: f32) -> u16 {
    if value.is_finite() && value > 0.0 {
        f32_to_f16_bits(value.log2().clamp(SCIENCE_LOG2_MIN, SCIENCE_LOG2_MAX))
    } else {
        SCIENCE_ZERO_BITS
    }
}

#[inline]
pub fn try_encode_log2_f16(channel: &str, index: usize, value: f32) -> Result<u16, BrickError> {
    if value == 0.0 {
        return Ok(SCIENCE_ZERO_BITS);
    }
    if !value.is_finite() || value < 0.0 {
        return Err(BrickError::InvalidScienceValue(format!(
            "{channel}[{index}] is non-physical extinction {value:?}"
        )));
    }
    let log2 = value.log2();
    if log2 < SCIENCE_LOG2_MIN {
        return Ok(SCIENCE_ZERO_BITS);
    }
    if log2 > SCIENCE_LOG2_MAX {
        return Err(BrickError::InvalidScienceValue(format!(
            "{channel}[{index}]={value:e} m^-1 has log2 {log2:.4}, outside {SCIENCE_LOG2_MIN}..={SCIENCE_LOG2_MAX}"
        )));
    }
    Ok(f32_to_f16_bits(log2))
}

/// Decode binary16(log2(beta)); the -infinity sentinel and any malformed
/// non-finite/negative result decode to exact clear sky.
#[inline]
pub fn decode_log2_f16(bits: u16) -> f32 {
    if bits == SCIENCE_ZERO_BITS {
        return 0.0;
    }
    let log2 = f16_bits_to_f32(bits);
    if log2.is_finite() && (SCIENCE_LOG2_MIN..=SCIENCE_LOG2_MAX).contains(&log2) {
        let value = log2.exp2();
        if value.is_finite() && value > 0.0 {
            return value;
        }
    }
    0.0
}

/// The self-describing header at the front of every `.ssb` file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrickHeader {
    pub format_version: u32,
    #[serde(default)]
    pub storage_profile: StorageProfile,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub z_min_m: f64,
    pub dz_m: f64,
    pub temperature_encoding: String,
    pub quant: ChannelQuant,
    pub channels_3d: Vec<String>,
    #[serde(default)]
    pub science_channels_3d: Vec<String>,
    #[serde(default)]
    pub science_encoding: Option<String>,
    pub planes_2d: Vec<String>,
    pub has_snowh: bool,
    pub has_ivgtyp: bool,
    pub has_cloud_fraction: bool,
    pub time_iso: Option<String>,
}

/// One timestep's volume brick, in memory.
#[derive(Debug, Clone, PartialEq)]
pub struct VolumeBrick {
    /// On-disk storage profile from which this brick was decoded. Programmatic
    /// bricks and the shipping ingest default use CompactU8.
    pub storage_profile: StorageProfile,
    /// Present only for ScienceCloudF16. The compact channels remain alongside
    /// it as an explicit backward-inspection fallback, but renderers select this
    /// payload when the profile says it is authoritative.
    pub science_cloud_f16: Option<ScienceCloudF16Payload>,
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
    /// QSNOW-only auxiliary subset of `ext_precip`; do not add it to totals twice.
    pub ext_snow: Vec<u8>,
    pub ext_precip: Vec<u8>,
    pub tau_up: Vec<u8>,
    pub qvapor: Vec<u8>,
    /// Linear u8 cloud coverage. All 255 with `has_cloud_fraction == false` is the
    /// source-unavailable full-coverage fallback.
    pub cloud_fraction: Vec<u8>,
    pub has_cloud_fraction: bool,
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
    fn channel_3d(&self, name: &str) -> &[u8] {
        match name {
            "ext_liquid" => &self.ext_liquid,
            "ext_ice" => &self.ext_ice,
            "ext_snow" => &self.ext_snow,
            "ext_precip" => &self.ext_precip,
            "tau_up" => &self.tau_up,
            "qvapor" => &self.qvapor,
            "cloud_fraction" => &self.cloud_fraction,
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
        self.header_for_profile(SSB_FORMAT_VERSION, StorageProfile::CompactU8)
    }

    fn header_for_profile(&self, format_version: u32, profile: StorageProfile) -> BrickHeader {
        let science_channels_3d = match profile {
            StorageProfile::CompactU8 => Vec::new(),
            StorageProfile::ScienceCloudF16 => vec![
                "ext_liquid".to_string(),
                "ext_ice".to_string(),
                "ext_snow".to_string(),
                "ext_precip".to_string(),
            ],
        };
        BrickHeader {
            format_version,
            storage_profile: profile,
            nx: self.nx,
            ny: self.ny,
            nz: self.nz,
            z_min_m: self.z_min_m,
            dz_m: self.dz_m,
            temperature_encoding: "celsius_f16".to_string(),
            quant: self.quant.clone(),
            channels_3d: CHANNELS_3D.iter().map(|s| s.to_string()).collect(),
            science_channels_3d,
            science_encoding: (profile == StorageProfile::ScienceCloudF16)
                .then(|| "log2-f16-le".to_string()),
            planes_2d: self
                .planes_2d_list()
                .iter()
                .map(|(name, _)| name.to_string())
                .collect(),
            has_snowh: self.snowh.is_some(),
            has_ivgtyp: self.ivgtyp.is_some(),
            has_cloud_fraction: self.has_cloud_fraction,
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
    InvalidScienceValue(String),
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
            Self::InvalidScienceValue(s) => write!(f, "invalid ScienceCloudF16 payload: {s}"),
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

/// Stream a brick's raw payload in the canonical v4+ order retained by v5. Numeric planes use a
/// small conversion buffer rather than materializing a second brick-sized byte
/// vector; this is what keeps full-domain HRRR writes inside the ingest RSS budget.
fn write_payload<W: Write>(out: &mut W, brick: &VolumeBrick) -> Result<(), std::io::Error> {
    for name in CHANNELS_3D {
        out.write_all(brick.channel_3d(name))?;
    }
    const NUMERIC_CHUNK: usize = 16 * 1024;
    let mut bytes = Vec::with_capacity(NUMERIC_CHUNK * 4);
    for chunk in brick.temperature_f16.chunks(NUMERIC_CHUNK) {
        bytes.clear();
        for &bits in chunk {
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
        out.write_all(&bytes)?;
    }
    for (_, plane) in brick.planes_2d_list() {
        for chunk in plane.chunks(NUMERIC_CHUNK) {
            bytes.clear();
            for &v in chunk {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            out.write_all(&bytes)?;
        }
    }
    Ok(())
}

fn write_u16_channel<W: Write>(out: &mut W, values: &[u16]) -> Result<(), std::io::Error> {
    const NUMERIC_CHUNK: usize = 16 * 1024;
    let mut bytes = Vec::with_capacity(NUMERIC_CHUNK * 2);
    for chunk in values.chunks(NUMERIC_CHUNK) {
        bytes.clear();
        for &bits in chunk {
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
        out.write_all(&bytes)?;
    }
    Ok(())
}

fn validate_science_cloud_f16(
    nx: usize,
    ny: usize,
    nz: usize,
    science: &ScienceCloudF16Payload,
) -> Result<(), BrickError> {
    let cells = nx
        .checked_mul(ny)
        .and_then(|n| n.checked_mul(nz))
        .ok_or_else(|| BrickError::Truncated("science cloud dimensions overflow".to_string()))?;
    for (name, values) in [
        ("ext_liquid", science.ext_liquid.as_slice()),
        ("ext_ice", science.ext_ice.as_slice()),
        ("ext_snow", science.ext_snow.as_slice()),
        ("ext_precip", science.ext_precip.as_slice()),
    ] {
        if values.len() != cells {
            return Err(BrickError::Truncated(format!(
                "science {name}: expected {cells} samples, got {}",
                values.len()
            )));
        }
        for (index, &bits) in values.iter().enumerate() {
            if bits == SCIENCE_ZERO_BITS {
                continue;
            }
            let log2 = f16_bits_to_f32(bits);
            if !log2.is_finite() || !(SCIENCE_LOG2_MIN..=SCIENCE_LOG2_MAX).contains(&log2) {
                return Err(BrickError::InvalidScienceValue(format!(
                    "{name}[{index}] has noncanonical bits 0x{bits:04x} (decoded log2={log2:?})"
                )));
            }
        }
    }
    Ok(())
}

/// Validate every in-memory channel/plane length before serializing. This keeps a
/// malformed programmatic `VolumeBrick` from writing a file whose header promises
/// more (or fewer) bytes than its payload contains, and applies the same 64-GiB
/// ceiling as the untrusted reader path.
fn validate_brick_payload(brick: &VolumeBrick) -> Result<(), BrickError> {
    match (brick.storage_profile, brick.science_cloud_f16.as_ref()) {
        (StorageProfile::CompactU8, None) => {}
        (StorageProfile::ScienceCloudF16, Some(science)) => {
            validate_science_cloud_f16(brick.nx, brick.ny, brick.nz, science)?;
        }
        (StorageProfile::CompactU8, Some(_)) => {
            return Err(BrickError::CacheMismatch(
                "CompactU8 brick unexpectedly carries ScienceCloudF16 channels".to_string(),
            ));
        }
        (StorageProfile::ScienceCloudF16, None) => {
            return Err(BrickError::CacheMismatch(
                "ScienceCloudF16 brick is missing its authoritative channels".to_string(),
            ));
        }
    }
    let cells_2d = brick.nx.checked_mul(brick.ny).ok_or_else(|| {
        BrickError::Truncated(format!("brick dims {}x{} overflow", brick.nx, brick.ny))
    })?;
    let cells_3d = cells_2d.checked_mul(brick.nz).ok_or_else(|| {
        BrickError::Truncated(format!(
            "brick dims {}x{}x{} overflow",
            brick.nx, brick.ny, brick.nz
        ))
    })?;
    for name in CHANNELS_3D {
        let found = brick.channel_3d(name).len();
        if found != cells_3d {
            return Err(BrickError::Truncated(format!(
                "{name}: expected {cells_3d} bytes, got {found}"
            )));
        }
    }
    if brick.temperature_f16.len() != cells_3d {
        return Err(BrickError::Truncated(format!(
            "temperature: expected {cells_3d} samples, got {}",
            brick.temperature_f16.len()
        )));
    }
    for (name, plane) in brick.planes_2d_list() {
        if plane.len() != cells_2d {
            return Err(BrickError::Truncated(format!(
                "{name}: expected {cells_2d} samples, got {}",
                plane.len()
            )));
        }
    }
    if !brick.has_cloud_fraction && brick.cloud_fraction.iter().any(|&v| v != 255) {
        return Err(BrickError::Truncated(
            "cloud_fraction provenance is false but fallback bytes are not all 255".to_string(),
        ));
    }
    let n_planes = brick.planes_2d_list().len();
    cells_3d
        .checked_mul(CHANNELS_3D.len() + 2)
        .and_then(|v| v.checked_add(cells_2d.checked_mul(n_planes.checked_mul(4)?)?))
        .filter(|&v| (v as u64) <= MAX_SSB_PAYLOAD_BYTES)
        .ok_or_else(|| {
            BrickError::Truncated(format!(
                "brick describes an implausible payload ({}x{}x{})",
                brick.nx, brick.ny, brick.nz
            ))
        })?;
    Ok(())
}

/// Write a brick to `path` as an `.ssb` file.
pub fn write_ssb(path: &Path, brick: &VolumeBrick) -> Result<u64, BrickError> {
    if brick.storage_profile != StorageProfile::CompactU8 || brick.science_cloud_f16.is_some() {
        return Err(BrickError::CacheMismatch(
            "write_ssb is the CompactU8 writer but the brick carries a science profile; use write_ssb_profiled"
                .to_string(),
        ));
    }
    validate_brick_payload(brick)?;
    let header = brick.header();
    let header_json = serde_json::to_vec(&header)?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::File::create(path)?;
    file.write_all(&SSB_MAGIC)?;
    file.write_all(&SSB_FORMAT_VERSION.to_le_bytes())?;
    file.write_all(&(header_json.len() as u32).to_le_bytes())?;
    file.write_all(&header_json)?;
    let mut encoder = ZlibEncoder::new(file, Compression::default());
    write_payload(&mut encoder, brick)?;
    let mut file = encoder.finish()?;
    file.flush()?;
    Ok(std::fs::metadata(path)?.len())
}

/// Write the experimental SSB v7 ScienceCloudF16 profile.
///
/// The canonical compact-v6 payload is retained as a fallback and the three
/// additive post-resample extinction channels plus the snow-only thermal subset
/// are appended as little-endian
/// binary16(log2(beta)). The default [`write_ssb`] API and all manifests remain v6;
/// callers must opt in explicitly and should use [`read_ssb_profiled`] to recover
/// the extension.
pub fn write_ssb_science_f16(
    path: &Path,
    brick: &VolumeBrick,
    science: ScienceCloudExtinction<'_>,
) -> Result<u64, BrickError> {
    let science = ScienceCloudF16Payload::try_encode(science)?;
    write_ssb_science_f16_payload(path, brick, &science)
}

/// Dispatch to the writer required by the brick's explicit profile. This is the
/// ingest-facing API; it never silently substitutes CompactU8 for missing science
/// channels or writes a science payload into the compact namespace.
pub fn write_ssb_profiled(path: &Path, brick: &VolumeBrick) -> Result<u64, BrickError> {
    match brick.storage_profile {
        StorageProfile::CompactU8 => write_ssb(path, brick),
        StorageProfile::ScienceCloudF16 => {
            let science = brick.science_cloud_f16.as_ref().ok_or_else(|| {
                BrickError::CacheMismatch(
                    "ScienceCloudF16 brick is missing its four authoritative extinction channels"
                        .to_string(),
                )
            })?;
            write_ssb_science_f16_payload(path, brick, science)
        }
    }
}

/// Write an already encoded experimental ScienceCloudF16 payload. This is useful
/// to streaming ingest/audit code that encodes each source channel as it becomes
/// available and cannot retain all four native f32 volumes simultaneously.
pub fn write_ssb_science_f16_payload(
    path: &Path,
    brick: &VolumeBrick,
    science: &ScienceCloudF16Payload,
) -> Result<u64, BrickError> {
    validate_brick_payload(brick)?;
    validate_science_cloud_f16(brick.nx, brick.ny, brick.nz, science)?;
    let cells_2d = brick.nx.checked_mul(brick.ny).ok_or_else(|| {
        BrickError::Truncated("science cloud horizontal dimensions overflow".to_string())
    })?;
    let cells_3d = cells_2d.checked_mul(brick.nz).ok_or_else(|| {
        BrickError::Truncated("science cloud volume dimensions overflow".to_string())
    })?;
    let n_planes = brick.planes_2d_list().len();
    cells_3d
        .checked_mul(CHANNELS_3D.len() + 2 + 8)
        .and_then(|v| v.checked_add(cells_2d.checked_mul(n_planes.checked_mul(4)?)?))
        .filter(|&v| (v as u64) <= MAX_SSB_PAYLOAD_BYTES)
        .ok_or_else(|| {
            BrickError::Truncated("science cloud payload exceeds safety ceiling".to_string())
        })?;

    let header =
        brick.header_for_profile(SCIENCE_F16_FORMAT_VERSION, StorageProfile::ScienceCloudF16);
    let header_json = serde_json::to_vec(&header)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::File::create(path)?;
    file.write_all(&SSB_MAGIC)?;
    file.write_all(&SCIENCE_F16_FORMAT_VERSION.to_le_bytes())?;
    file.write_all(&(header_json.len() as u32).to_le_bytes())?;
    file.write_all(&header_json)?;
    let mut encoder = ZlibEncoder::new(file, Compression::default());
    write_payload(&mut encoder, brick)?;
    write_u16_channel(&mut encoder, &science.ext_liquid)?;
    write_u16_channel(&mut encoder, &science.ext_ice)?;
    write_u16_channel(&mut encoder, &science.ext_snow)?;
    write_u16_channel(&mut encoder, &science.ext_precip)?;
    let mut file = encoder.finish()?;
    file.flush()?;
    Ok(std::fs::metadata(path)?.len())
}

// Keep compressed input residency bounded while providing zlib with reasonably
// large sequential chunks. Full-domain HRRR `.ssb` files can exceed 200 MiB.
const SSB_READ_BUFFER_BYTES: usize = 64 * 1024;

fn read_file_exact<R: Read>(
    reader: &mut R,
    cursor: &mut u64,
    bytes: &mut [u8],
    what: &str,
    file_len: u64,
) -> Result<(), BrickError> {
    let n = bytes.len();
    let end = cursor
        .checked_add(n as u64)
        .ok_or_else(|| BrickError::Truncated(format!("{what}: length overflow")))?;
    if end > file_len {
        return Err(BrickError::Truncated(format!(
            "{what}: need {n} bytes at {cursor}, have {file_len}"
        )));
    }
    reader.read_exact(bytes).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            BrickError::Truncated(format!(
                "{what}: need {n} bytes at {cursor}, have {file_len}"
            ))
        } else {
            BrickError::Io(e)
        }
    })?;
    *cursor = end;
    Ok(())
}

fn check_file_bytes_available(
    cursor: u64,
    n: usize,
    file_len: u64,
    what: &str,
) -> Result<(), BrickError> {
    let end = cursor
        .checked_add(n as u64)
        .ok_or_else(|| BrickError::Truncated(format!("{what}: length overflow")))?;
    if end > file_len {
        return Err(BrickError::Truncated(format!(
            "{what}: need {n} bytes at {cursor}, have {file_len}"
        )));
    }
    Ok(())
}

fn read_payload_exact<R: Read>(
    reader: &mut R,
    bytes: &mut [u8],
    what: &str,
) -> Result<(), BrickError> {
    reader.read_exact(bytes).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            BrickError::Truncated(format!("{what}: expected {} payload bytes", bytes.len()))
        } else {
            BrickError::Io(e)
        }
    })
}

fn read_payload_bytes<R: Read>(
    reader: &mut R,
    n: usize,
    what: &str,
) -> Result<Vec<u8>, BrickError> {
    let mut out = vec![0u8; n];
    read_payload_exact(reader, &mut out, what)?;
    Ok(out)
}

fn read_temperature_u16<R: Read>(reader: &mut R, cells_3d: usize) -> Result<Vec<u16>, BrickError> {
    const CHUNK: usize = 16 * 1024;
    let mut out = Vec::with_capacity(cells_3d);
    let mut buf = vec![0u8; CHUNK * 2];
    let mut remaining = cells_3d;
    while remaining > 0 {
        let samples = remaining.min(CHUNK);
        let bytes = &mut buf[..samples * 2];
        read_payload_exact(reader, bytes, "temperature")?;
        out.extend(
            bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]])),
        );
        remaining -= samples;
    }
    Ok(out)
}

fn read_plane_f32<R: Read>(
    reader: &mut R,
    cells_2d: usize,
    name: &str,
) -> Result<Vec<f32>, BrickError> {
    const CHUNK: usize = 16 * 1024;
    let mut out = Vec::with_capacity(cells_2d);
    let mut buf = vec![0u8; CHUNK * 4];
    let mut remaining = cells_2d;
    while remaining > 0 {
        let samples = remaining.min(CHUNK);
        let bytes = &mut buf[..samples * 4];
        read_payload_exact(reader, bytes, name)?;
        out.extend(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        );
        remaining -= samples;
    }
    Ok(out)
}

/// Read an `.ssb` file back into a `VolumeBrick`.
pub fn read_ssb(path: &Path) -> Result<VolumeBrick, BrickError> {
    let file = std::fs::File::open(path)?;
    let file_len = file.metadata()?.len();
    let reader = BufReader::with_capacity(SSB_READ_BUFFER_BYTES, file);
    read_ssb_stream(reader, file_len)
}

fn read_ssb_stream<R: BufRead>(mut reader: R, file_len: u64) -> Result<VolumeBrick, BrickError> {
    read_ssb_profiled_stream(&mut reader, file_len, false).map(|profiled| profiled.brick)
}

/// Read either the production SSB v6 CompactU8 profile or the experimental SSB
/// v7 ScienceCloudF16 profile. This is the explicit opt-in backward-compatible
/// reader; [`read_ssb`] intentionally retains its strict v6 production contract.
pub fn read_ssb_profiled(path: &Path) -> Result<ProfiledVolumeBrick, BrickError> {
    let file = std::fs::File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut reader = BufReader::with_capacity(SSB_READ_BUFFER_BYTES, file);
    read_ssb_profiled_stream(&mut reader, file_len, true)
}

fn read_ssb_profiled_stream<R: BufRead>(
    mut reader: R,
    file_len: u64,
    allow_science: bool,
) -> Result<ProfiledVolumeBrick, BrickError> {
    let mut cursor = 0u64;

    let mut magic = [0u8; 4];
    read_file_exact(&mut reader, &mut cursor, &mut magic, "magic", file_len)?;
    if magic != SSB_MAGIC {
        return Err(BrickError::BadMagic(magic));
    }
    let mut version_bytes = [0u8; 4];
    read_file_exact(
        &mut reader,
        &mut cursor,
        &mut version_bytes,
        "version",
        file_len,
    )?;
    let version = u32::from_le_bytes(version_bytes);
    if version != SSB_FORMAT_VERSION && !(allow_science && version == SCIENCE_F16_FORMAT_VERSION) {
        return Err(BrickError::UnsupportedVersion(version));
    }
    let mut header_len_bytes = [0u8; 4];
    read_file_exact(
        &mut reader,
        &mut cursor,
        &mut header_len_bytes,
        "header_len",
        file_len,
    )?;
    let header_len = u32::from_le_bytes(header_len_bytes) as usize;
    check_file_bytes_available(cursor, header_len, file_len, "header")?;
    let mut header_bytes = vec![0u8; header_len];
    read_file_exact(
        &mut reader,
        &mut cursor,
        &mut header_bytes,
        "header",
        file_len,
    )?;
    let header: BrickHeader = serde_json::from_slice(&header_bytes)?;
    if header.format_version != version {
        return Err(BrickError::Truncated(format!(
            "header format_version {} disagrees with file version {version}",
            header.format_version
        )));
    }
    let expected_profile = if version == SSB_FORMAT_VERSION {
        StorageProfile::CompactU8
    } else {
        StorageProfile::ScienceCloudF16
    };
    if header.storage_profile != expected_profile {
        return Err(BrickError::Truncated(format!(
            "header storage_profile {:?} disagrees with SSB v{version}",
            header.storage_profile
        )));
    }
    let expected_science_channels = if expected_profile == StorageProfile::ScienceCloudF16 {
        vec![
            "ext_liquid".to_string(),
            "ext_ice".to_string(),
            "ext_snow".to_string(),
            "ext_precip".to_string(),
        ]
    } else {
        Vec::new()
    };
    if header.science_channels_3d != expected_science_channels {
        return Err(BrickError::Truncated(format!(
            "header science channel order {:?} does not match profile {:?} {:?}",
            header.science_channels_3d, expected_profile, expected_science_channels
        )));
    }
    let expected_science_encoding =
        (expected_profile == StorageProfile::ScienceCloudF16).then_some("log2-f16-le");
    if header.science_encoding.as_deref() != expected_science_encoding {
        return Err(BrickError::Truncated(format!(
            "header science encoding {:?} does not match profile {:?}",
            header.science_encoding, expected_profile
        )));
    }
    let expected_channels: Vec<String> = CHANNELS_3D.iter().map(|s| s.to_string()).collect();
    if header.channels_3d != expected_channels {
        return Err(BrickError::Truncated(format!(
            "header channel order {:?} does not match SSB v{version} {:?}",
            header.channels_3d, expected_channels
        )));
    }
    let mut expected_planes = vec!["hgt", "landmask", "tsk", "u10", "v10"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    if header.has_snowh {
        expected_planes.push("snowh".to_string());
    }
    if header.has_ivgtyp {
        expected_planes.push("ivgtyp".to_string());
    }
    if header.planes_2d != expected_planes {
        return Err(BrickError::Truncated(format!(
            "header plane order {:?} does not match flags {:?}",
            header.planes_2d, expected_planes
        )));
    }

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
    let science_bytes_per_cell = if expected_profile == StorageProfile::ScienceCloudF16 {
        8
    } else {
        0
    };
    let expected_payload = cells_3d
        .checked_mul(CHANNELS_3D.len() + 2) // seven u8 channels + the u16 temperature
        .and_then(|v| v.checked_add(cells_2d.checked_mul(n_planes * 4)?))
        .and_then(|v| v.checked_add(cells_3d.checked_mul(science_bytes_per_cell)?))
        .filter(|&v| (v as u64) <= MAX_SSB_PAYLOAD_BYTES)
        .ok_or_else(|| {
            BrickError::Truncated(format!(
                "header describes an implausible payload ({}x{}x{})",
                header.nx, header.ny, header.nz
            ))
        })?;

    // Decode directly into final channel/plane allocations. The pre-v4 reader
    // first materialized the whole raw payload and then cloned every field out of
    // it, temporarily doubling brick memory; seven v4+ volume channels make that
    // avoidable spike especially costly on full HRRR domains.
    let mut decoder = ZlibDecoder::new(reader).take(expected_payload as u64 + 1);
    let ext_liquid = read_payload_bytes(&mut decoder, cells_3d, "ext_liquid")?;
    let ext_ice = read_payload_bytes(&mut decoder, cells_3d, "ext_ice")?;
    let ext_snow = read_payload_bytes(&mut decoder, cells_3d, "ext_snow")?;
    let ext_precip = read_payload_bytes(&mut decoder, cells_3d, "ext_precip")?;
    let tau_up = read_payload_bytes(&mut decoder, cells_3d, "tau_up")?;
    let qvapor = read_payload_bytes(&mut decoder, cells_3d, "qvapor")?;
    let cloud_fraction = read_payload_bytes(&mut decoder, cells_3d, "cloud_fraction")?;
    if !header.has_cloud_fraction && cloud_fraction.iter().any(|&v| v != 255) {
        return Err(BrickError::Truncated(
            "cloud_fraction provenance is false but fallback bytes are not all 255".to_string(),
        ));
    }

    let temperature_f16 = read_temperature_u16(&mut decoder, cells_3d)?;

    let hgt = read_plane_f32(&mut decoder, cells_2d, "hgt")?;
    let landmask = read_plane_f32(&mut decoder, cells_2d, "landmask")?;
    let tsk = read_plane_f32(&mut decoder, cells_2d, "tsk")?;
    let u10 = read_plane_f32(&mut decoder, cells_2d, "u10")?;
    let v10 = read_plane_f32(&mut decoder, cells_2d, "v10")?;
    let snowh = if header.has_snowh {
        Some(read_plane_f32(&mut decoder, cells_2d, "snowh")?)
    } else {
        None
    };
    let ivgtyp = if header.has_ivgtyp {
        Some(read_plane_f32(&mut decoder, cells_2d, "ivgtyp")?)
    } else {
        None
    };
    let science_cloud_f16 = if expected_profile == StorageProfile::ScienceCloudF16 {
        let payload = ScienceCloudF16Payload {
            ext_liquid: read_temperature_u16(&mut decoder, cells_3d)?,
            ext_ice: read_temperature_u16(&mut decoder, cells_3d)?,
            ext_snow: read_temperature_u16(&mut decoder, cells_3d)?,
            ext_precip: read_temperature_u16(&mut decoder, cells_3d)?,
        };
        validate_science_cloud_f16(header.nx, header.ny, header.nz, &payload)?;
        Some(payload)
    } else {
        None
    };
    let mut extra = [0u8; 1];
    if decoder.read(&mut extra)? != 0 {
        return Err(BrickError::Truncated(format!(
            "payload exceeds the {expected_payload} bytes the header describes"
        )));
    }

    let brick = VolumeBrick {
        storage_profile: expected_profile,
        science_cloud_f16,
        nx: header.nx,
        ny: header.ny,
        nz: header.nz,
        z_min_m: header.z_min_m,
        dz_m: header.dz_m,
        time_iso: header.time_iso,
        quant: header.quant,
        ext_liquid,
        ext_ice,
        ext_snow,
        ext_precip,
        tau_up,
        qvapor,
        cloud_fraction,
        has_cloud_fraction: header.has_cloud_fraction,
        temperature_f16,
        hgt,
        landmask,
        tsk,
        u10,
        v10,
        snowh,
        ivgtyp,
    };
    Ok(ProfiledVolumeBrick {
        profile: expected_profile,
        brick,
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
    /// Whether `cloud_fraction` came from a trusted model field. False means the
    /// brick carries the explicit all-255 full-coverage fallback.
    #[serde(default)]
    pub has_cloud_fraction: bool,
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
    #[serde(default)]
    pub storage_profile: StorageProfile,
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
        Self::load_for_profile(path, StorageProfile::CompactU8)
    }

    /// Load a manifest only when both its format epoch and storage-profile
    /// identity match the explicitly requested profile.
    pub fn load_for_profile(path: &Path, profile: StorageProfile) -> Result<Self, BrickError> {
        let text = std::fs::read_to_string(path)?;
        let value: serde_json::Value = serde_json::from_str(&text)?;
        let found_profile = value
            .get("storage_profile")
            .cloned()
            .and_then(|value| serde_json::from_value::<StorageProfile>(value).ok());
        if let Some(found_profile) = found_profile
            && found_profile != profile
        {
            return Err(BrickError::CacheMismatch(format!(
                "run.json uses storage_profile={}, but {} was requested; select the matching profile (cached sources are never substituted)",
                found_profile.slug(),
                profile.slug()
            )));
        }
        let found = value
            .get("format_version")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if found != profile.format_version() as u64 {
            return Err(BrickError::UnsupportedManifestVersion {
                found,
                expected: profile.format_version(),
            });
        }
        let manifest: Self = serde_json::from_value(value)?;
        if manifest.storage_profile != profile {
            return Err(BrickError::CacheMismatch(format!(
                "run.json storage_profile={} but {} was requested; select the matching profile or open the original source to regenerate",
                manifest.storage_profile.slug(),
                profile.slug()
            )));
        }
        Ok(manifest)
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
        Self::load_or_new_profiled(
            path,
            StorageProfile::CompactU8,
            run_id,
            nx,
            ny,
            nz,
            z_min_m,
            dz_m,
            planes_2d,
            projection,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_or_new_profiled(
        path: &Path,
        storage_profile: StorageProfile,
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
            let manifest = match Self::load_for_profile(path, storage_profile) {
                Ok(m) => m,
                // An OLD-FORMAT manifest found during INGEST is a regenerable cache
                // being re-populated by a newer build: SUPERSEDE it with a fresh
                // manifest instead of erroring (the source-backed format self-heal).
                // The old-format .ssb
                // files alongside it are refused by `read_ssb` and re-ingested per
                // timestep through the same path; only the READ paths (a cached
                // run.json open) keep the hard remedy-bearing refusal, because there
                // is no source wrfout there to re-ingest from.
                Err(BrickError::UnsupportedManifestVersion { .. }) => {
                    return Ok(Self::new_manifest(
                        storage_profile,
                        run_id,
                        nx,
                        ny,
                        nz,
                        z_min_m,
                        dz_m,
                        planes_2d,
                        projection,
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
            storage_profile,
            run_id,
            nx,
            ny,
            nz,
            z_min_m,
            dz_m,
            planes_2d,
            projection,
        ))
    }

    /// A fresh empty manifest at the CURRENT format version (the no-existing-manifest
    /// and the superseded-old-format paths of [`Self::load_or_new`]).
    #[allow(clippy::too_many_arguments)]
    fn new_manifest(
        storage_profile: StorageProfile,
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
            format_version: storage_profile.format_version(),
            storage_profile,
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
    use std::io::Cursor;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    static UNIQ: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> PathBuf {
        let n = UNIQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("simsat-ssb-{}-{}", std::process::id(), n));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    struct CountingReader<R> {
        inner: R,
        bytes_read: Arc<AtomicU64>,
    }

    impl<R: Read> Read for CountingReader<R> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let read = self.inner.read(buf)?;
            self.bytes_read.fetch_add(read as u64, Ordering::Relaxed);
            Ok(read)
        }
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
    fn cloud_fraction_linear_u8_clamps_rounds_and_decodes() {
        let codes = encode_cloud_fraction(&[
            -1.0,
            0.0,
            0.5,
            1.0,
            2.0,
            f32::NAN,
            f32::NEG_INFINITY,
            f32::INFINITY,
        ]);
        assert_eq!(codes, vec![0, 0, 128, 255, 255, 0, 0, 0]);
        assert_eq!(decode_cloud_fraction(0), 0.0);
        assert_eq!(decode_cloud_fraction(255), 1.0);
        assert!((decode_cloud_fraction(128) - 128.0 / 255.0).abs() < f32::EPSILON);
    }

    #[test]
    fn cloud_fraction_tiny_positive_tail_never_encodes_as_clear() {
        let codes = encode_cloud_fraction(&[f32::MIN_POSITIVE, 1.0e-8, 0.49 / 255.0, 0.5 / 255.0]);
        assert_eq!(codes, vec![1, 1, 1, 1]);
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
    fn log2_f16_preserves_wispy_extinction_without_a_positive_floor() {
        for value in [1.0e-12f32, 1.0e-9, 1.0e-7, 1.0e-4, 0.02] {
            let decoded = decode_log2_f16(encode_log2_f16(value));
            let relative = ((decoded - value) / value).abs();
            assert!(relative < 0.012, "{value:e} -> {decoded:e} ({relative})");
        }
        for value in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(decode_log2_f16(encode_log2_f16(value)), 0.0);
        }
        for malformed in [0x7c00, 0xfc00, 0x7e00, 0x7bff] {
            assert_eq!(decode_log2_f16(malformed), 0.0);
        }
    }

    #[test]
    fn science_profile_identity_and_physical_encoding_policy_are_explicit() {
        assert_eq!(StorageProfile::default(), StorageProfile::CompactU8);
        for profile in StorageProfile::ALL {
            assert_eq!(StorageProfile::parse(profile.slug()), Some(profile));
        }
        assert_eq!(
            StorageProfile::parse("science-f16"),
            Some(StorageProfile::ScienceCloudF16)
        );
        assert_eq!(StorageProfile::ScienceCloudF16.format_version(), 7);
        assert_eq!(
            StorageProfile::CompactU8.format_version(),
            SSB_FORMAT_VERSION
        );
        assert_eq!(
            try_encode_log2_f16("cloud", 0, 0.0).unwrap(),
            SCIENCE_ZERO_BITS
        );
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0e-6] {
            assert!(matches!(
                try_encode_log2_f16("cloud", 3, value),
                Err(BrickError::InvalidScienceValue(_))
            ));
        }
        assert_eq!(
            try_encode_log2_f16("cloud", 4, 2.0f32.powf(SCIENCE_LOG2_MIN - 1.0)).unwrap(),
            SCIENCE_ZERO_BITS,
            "finite sub-floor tails canonicalize to exact clear sky"
        );
        assert!(matches!(
            try_encode_log2_f16("cloud", 4, 2.0f32.powf(SCIENCE_LOG2_MAX + 1.0)),
            Err(BrickError::InvalidScienceValue(_))
        ));
        assert_eq!(
            decode_log2_f16(0x7e00),
            0.0,
            "NaN half is never plausible cloud"
        );
        assert_eq!(
            decode_log2_f16(0x7c00),
            0.0,
            "+Inf half is never plausible cloud"
        );
    }

    #[test]
    fn science_log2_f16_beats_compact_u8_on_subfloor_cloud_tails() {
        let values: Vec<f32> = (0..400)
            .map(|i| 10.0f32.powf(-12.0 + 10.0 * i as f32 / 399.0))
            .collect();
        let (quant, compact) = encode_log_channel(&values);
        let compact_rmse = (values
            .iter()
            .zip(&compact)
            .map(|(&native, &code)| (quant.decode(code) as f64 - native as f64).powi(2))
            .sum::<f64>()
            / values.len() as f64)
            .sqrt();
        let science_rmse = (values
            .iter()
            .map(|&native| {
                (decode_log2_f16(encode_log2_f16(native)) as f64 - native as f64).powi(2)
            })
            .sum::<f64>()
            / values.len() as f64)
            .sqrt();
        assert!(
            science_rmse < compact_rmse * 0.2,
            "science RMSE {science_rmse:e}, compact RMSE {compact_rmse:e}"
        );
        assert!(values.iter().any(|&v| (v as f64) < quant.vmin));
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
        let ext_snow_f32: Vec<f32> = (0..cells_3d).map(|i| (i as f32) * 3.0e-5).collect();
        let ext_precip_f32: Vec<f32> = vec![0.0; cells_3d];
        let tau_f32: Vec<f32> = (0..cells_3d).map(|i| 0.1 * i as f32).collect();
        let qv_f32: Vec<f32> = (0..cells_3d).map(|i| 1.0e-3 + 1.0e-5 * i as f32).collect();
        let kelvin: Vec<f32> = (0..cells_3d).map(|i| 300.0 - 0.3 * i as f32).collect();

        let (ql, ext_liquid) = encode_log_channel(&ext_liquid_f32);
        let (qi, ext_ice) = encode_log_channel(&ext_ice_f32);
        let (qs, ext_snow) = encode_log_channel(&ext_snow_f32);
        let (qp, ext_precip) = encode_log_channel(&ext_precip_f32);
        let (qt, tau_up) = encode_log_channel(&tau_f32);
        let (qv, qvapor) = encode_log_channel(&qv_f32);
        let mut map = BTreeMap::new();
        map.insert("ext_liquid".to_string(), ql);
        map.insert("ext_ice".to_string(), qi);
        map.insert("ext_snow".to_string(), qs);
        map.insert("ext_precip".to_string(), qp);
        map.insert("tau_up".to_string(), qt);
        map.insert("qvapor".to_string(), qv);

        VolumeBrick {
            storage_profile: StorageProfile::CompactU8,
            science_cloud_f16: None,
            nx,
            ny,
            nz,
            z_min_m: 0.0,
            dz_m: 250.0,
            time_iso: Some("2018-10-10T12:00:00Z".to_string()),
            quant: ChannelQuant(map),
            ext_liquid,
            ext_ice,
            ext_snow,
            ext_precip,
            tau_up,
            qvapor,
            cloud_fraction: encode_cloud_fraction(
                &(0..cells_3d)
                    .map(|i| i as f32 / (cells_3d - 1) as f32)
                    .collect::<Vec<_>>(),
            ),
            has_cloud_fraction: true,
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
    fn science_f16_profile_round_trips_and_reader_is_backward_compatible() {
        let brick = tiny_brick();
        let cells = brick.nx * brick.ny * brick.nz;
        let liquid: Vec<f32> = (0..cells)
            .map(|i| {
                if i % 7 == 0 {
                    0.0
                } else {
                    1.0e-12 * 1.7f32.powi(i as i32)
                }
            })
            .collect();
        let ice: Vec<f32> = liquid.iter().map(|v| *v * 0.2).collect();
        let snow: Vec<f32> = liquid.iter().map(|v| *v * 0.01).collect();
        let precip: Vec<f32> = liquid.iter().map(|v| *v * 0.03).collect();
        let dir = temp_dir();
        let compact_path = dir.join("compact-v6.ssb");
        let science_path = dir.join("science-v7.ssb");
        write_ssb(&compact_path, &brick).unwrap();
        write_ssb_science_f16(
            &science_path,
            &brick,
            ScienceCloudExtinction {
                ext_liquid: &liquid,
                ext_ice: &ice,
                ext_snow: &snow,
                ext_precip: &precip,
            },
        )
        .unwrap();

        let compact = read_ssb_profiled(&compact_path).unwrap();
        assert_eq!(compact.profile, StorageProfile::CompactU8);
        assert_eq!(compact.brick, brick);
        assert!(compact.brick.science_cloud_f16.is_none());

        let science = read_ssb_profiled(&science_path).unwrap();
        assert_eq!(science.profile, StorageProfile::ScienceCloudF16);
        let mut science_brick = science.brick;
        let payload = science_brick.science_cloud_f16.take().unwrap();
        science_brick.storage_profile = StorageProfile::CompactU8;
        assert_eq!(science_brick, brick);
        assert_eq!(
            payload.ext_liquid,
            liquid
                .iter()
                .map(|&value| encode_log2_f16(value))
                .collect::<Vec<_>>()
        );
        let decoded = ScienceCloudF16Payload::decode_channel(&payload.ext_liquid);
        for (&native, &back) in liquid.iter().zip(&decoded) {
            if native == 0.0 {
                assert_eq!(back, 0.0);
            } else {
                assert!(((back - native) / native).abs() < 0.012);
            }
        }
        assert!(matches!(
            read_ssb(&science_path),
            Err(BrickError::UnsupportedVersion(SCIENCE_F16_FORMAT_VERSION))
        ));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn science_f16_writer_rejects_incomplete_native_channels() {
        let brick = tiny_brick();
        let cells = brick.nx * brick.ny * brick.nz;
        let complete = vec![1.0e-6; cells];
        let short = vec![1.0e-6; cells - 1];
        let dir = temp_dir();
        let error = write_ssb_science_f16(
            &dir.join("incomplete.ssb"),
            &brick,
            ScienceCloudExtinction {
                ext_liquid: &complete,
                ext_ice: &short,
                ext_snow: &complete,
                ext_precip: &complete,
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("science ext_ice"), "{error}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn science_f16_reader_rejects_truncated_corrupt_and_oversized_payloads() {
        let brick = tiny_brick();
        let cells = brick.nx * brick.ny * brick.nz;
        let values = vec![1.0e-6; cells];
        let science = ScienceCloudF16Payload::encode(ScienceCloudExtinction {
            ext_liquid: &values,
            ext_ice: &values,
            ext_snow: &values,
            ext_precip: &values,
        });
        let dir = temp_dir();
        let good_path = dir.join("science-good.ssb");
        write_ssb_science_f16_payload(&good_path, &brick, &science).unwrap();
        let good = std::fs::read(&good_path).unwrap();
        let header_len = u32::from_le_bytes(good[8..12].try_into().unwrap()) as usize;
        let payload_start = 12 + header_len;

        let mut impossible_header = good.clone();
        impossible_header[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        let impossible_header_path = dir.join("science-impossible-header.ssb");
        std::fs::write(&impossible_header_path, impossible_header).unwrap();
        assert!(matches!(
            read_ssb_profiled(&impossible_header_path),
            Err(BrickError::Truncated(_))
        ));

        let truncated_path = dir.join("science-truncated.ssb");
        let cut = payload_start + (good.len() - payload_start) / 2;
        std::fs::write(&truncated_path, &good[..cut]).unwrap();
        assert!(read_ssb_profiled(&truncated_path).is_err());

        let mut corrupt = good.clone();
        corrupt[payload_start] = 0;
        let corrupt_path = dir.join("science-corrupt-zlib.ssb");
        std::fs::write(&corrupt_path, corrupt).unwrap();
        assert!(matches!(
            read_ssb_profiled(&corrupt_path),
            Err(BrickError::Io(_))
        ));

        let header =
            brick.header_for_profile(SCIENCE_F16_FORMAT_VERSION, StorageProfile::ScienceCloudF16);
        let header_json = serde_json::to_vec(&header).unwrap();
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        write_payload(&mut encoder, &brick).unwrap();
        write_u16_channel(&mut encoder, &science.ext_liquid).unwrap();
        write_u16_channel(&mut encoder, &science.ext_ice).unwrap();
        write_u16_channel(&mut encoder, &science.ext_snow).unwrap();
        write_u16_channel(&mut encoder, &science.ext_precip).unwrap();
        encoder.write_all(&[0xa5]).unwrap();
        let compressed = encoder.finish().unwrap();
        let mut extra = Vec::new();
        extra.extend_from_slice(&SSB_MAGIC);
        extra.extend_from_slice(&SCIENCE_F16_FORMAT_VERSION.to_le_bytes());
        extra.extend_from_slice(&(header_json.len() as u32).to_le_bytes());
        extra.extend_from_slice(&header_json);
        extra.extend_from_slice(&compressed);
        let extra_path = dir.join("science-extra-payload.ssb");
        std::fs::write(&extra_path, extra).unwrap();
        let Err(BrickError::Truncated(message)) = read_ssb_profiled(&extra_path) else {
            panic!("one extra decompressed science byte must be rejected");
        };
        assert!(message.contains("payload exceeds"), "{message}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn science_f16_writer_rejects_noncanonical_half_payload() {
        let brick = tiny_brick();
        let cells = brick.nx * brick.ny * brick.nz;
        let canonical = vec![encode_log2_f16(1.0e-6); cells];
        let mut payload = ScienceCloudF16Payload {
            ext_liquid: canonical.clone(),
            ext_ice: canonical.clone(),
            ext_snow: canonical.clone(),
            ext_precip: canonical,
        };
        payload.ext_ice[7] = 0x7e00; // binary16 NaN is never a canonical channel code.
        let dir = temp_dir();
        let err =
            write_ssb_science_f16_payload(&dir.join("nan-half.ssb"), &brick, &payload).unwrap_err();
        assert!(matches!(err, BrickError::InvalidScienceValue(_)), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn write_ssb_rejects_bad_channel_lengths_and_false_provenance_payload() {
        let dir = temp_dir();
        let path = dir.join("bad.ssb");
        let mut short = tiny_brick();
        short.ext_snow.pop();
        assert!(matches!(
            write_ssb(&path, &short),
            Err(BrickError::Truncated(_))
        ));

        let mut bad_fallback = tiny_brick();
        bad_fallback.has_cloud_fraction = false;
        assert!(bad_fallback.cloud_fraction.iter().any(|&v| v != 255));
        assert!(matches!(
            write_ssb(&path, &bad_fallback),
            Err(BrickError::Truncated(_))
        ));
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
    fn ssb_refuses_older_and_newer_versions_without_reading_the_whole_file() {
        for version in [SSB_FORMAT_VERSION - 1, SSB_FORMAT_VERSION + 1] {
            let mut bytes = Vec::with_capacity(SSB_READ_BUFFER_BYTES * 8);
            bytes.extend_from_slice(&SSB_MAGIC);
            bytes.extend_from_slice(&version.to_le_bytes());
            bytes.resize(SSB_READ_BUFFER_BYTES * 8, 0xa5);
            let file_len = bytes.len() as u64;
            let bytes_read = Arc::new(AtomicU64::new(0));
            let counted = CountingReader {
                inner: Cursor::new(bytes),
                bytes_read: Arc::clone(&bytes_read),
            };
            let reader = BufReader::with_capacity(SSB_READ_BUFFER_BYTES, counted);

            assert!(matches!(
                read_ssb_stream(reader, file_len),
                Err(BrickError::UnsupportedVersion(found)) if found == version
            ));
            let consumed = bytes_read.load(Ordering::Relaxed);
            assert!(
                consumed <= SSB_READ_BUFFER_BYTES as u64,
                "version rejection read {consumed} bytes instead of one bounded buffer"
            );
            assert!(
                consumed < file_len,
                "version rejection must not read the whole {file_len}-byte file"
            );
        }
    }

    #[test]
    fn read_ssb_rejects_truncated_fixed_and_json_headers() {
        let dir = temp_dir();
        let mut prefix = Vec::new();
        prefix.extend_from_slice(&SSB_MAGIC);
        prefix.extend_from_slice(&SSB_FORMAT_VERSION.to_le_bytes());
        prefix.extend_from_slice(&32u32.to_le_bytes());
        prefix.extend_from_slice(b"short");

        for (cut, expected) in [
            (0usize, "magic: need 4 bytes at 0, have 0"),
            (3, "magic: need 4 bytes at 0, have 3"),
            (4, "version: need 4 bytes at 4, have 4"),
            (7, "version: need 4 bytes at 4, have 7"),
            (8, "header_len: need 4 bytes at 8, have 8"),
            (11, "header_len: need 4 bytes at 8, have 11"),
            (12, "header: need 32 bytes at 12, have 12"),
            (15, "header: need 32 bytes at 12, have 15"),
            (prefix.len(), "header: need 32 bytes at 12, have 17"),
        ] {
            let path = dir.join(format!("header-{cut}.ssb"));
            std::fs::write(&path, &prefix[..cut]).unwrap();
            let Err(BrickError::Truncated(message)) = read_ssb(&path) else {
                panic!("a header truncated at byte {cut} must be reported as truncated");
            };
            assert_eq!(message, expected);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_ssb_rejects_impossible_header_len_before_allocation() {
        let dir = temp_dir();
        let path = dir.join("impossible-header.ssb");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&SSB_MAGIC);
        bytes.extend_from_slice(&SSB_FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();

        let Err(BrickError::Truncated(message)) = read_ssb(&path) else {
            panic!("an impossible header length must be rejected before allocation");
        };
        assert_eq!(
            message,
            format!("header: need {} bytes at 12, have 12", u32::MAX)
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_ssb_accepts_trailing_compressed_garbage_like_the_legacy_reader() {
        let dir = temp_dir();
        let path = dir.join("trailing.ssb");
        let brick = tiny_brick();
        write_ssb(&path, &brick).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.extend_from_slice(b"trailing bytes after the complete zlib stream");
        std::fs::write(&path, bytes).unwrap();

        assert_eq!(read_ssb(&path).unwrap(), brick);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_ssb_refuses_header_channel_reordering() {
        let dir = temp_dir();
        let good_path = dir.join("good.ssb");
        write_ssb(&good_path, &tiny_brick()).unwrap();
        let good = std::fs::read(&good_path).unwrap();
        let old_len = u32::from_le_bytes(good[8..12].try_into().unwrap()) as usize;
        let mut header: BrickHeader = serde_json::from_slice(&good[12..12 + old_len]).unwrap();
        header.channels_3d.swap(2, 3);
        let header_json = serde_json::to_vec(&header).unwrap();
        let mut bad = Vec::new();
        bad.extend_from_slice(&good[0..8]);
        bad.extend_from_slice(&(header_json.len() as u32).to_le_bytes());
        bad.extend_from_slice(&header_json);
        bad.extend_from_slice(&good[12 + old_len..]);
        let bad_path = dir.join("reordered.ssb");
        std::fs::write(&bad_path, bad).unwrap();
        let err = read_ssb(&bad_path).unwrap_err();
        assert!(matches!(err, BrickError::Truncated(_)), "{err:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_ssb_refuses_header_version_disagreement() {
        let dir = temp_dir();
        let good_path = dir.join("good.ssb");
        write_ssb(&good_path, &tiny_brick()).unwrap();
        let good = std::fs::read(&good_path).unwrap();
        let old_len = u32::from_le_bytes(good[8..12].try_into().unwrap()) as usize;
        let mut header: BrickHeader = serde_json::from_slice(&good[12..12 + old_len]).unwrap();
        header.format_version = SSB_FORMAT_VERSION + 1;
        let header_json = serde_json::to_vec(&header).unwrap();
        let mut bad = Vec::new();
        bad.extend_from_slice(&good[0..8]);
        bad.extend_from_slice(&(header_json.len() as u32).to_le_bytes());
        bad.extend_from_slice(&header_json);
        bad.extend_from_slice(&good[12 + old_len..]);
        let bad_path = dir.join("disagrees.ssb");
        std::fs::write(&bad_path, bad).unwrap();

        let err = read_ssb(&bad_path).unwrap_err();
        assert!(matches!(err, BrickError::Truncated(_)), "{err:?}");
        assert!(err.to_string().contains("disagrees"), "{err}");
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
            has_cloud_fraction: false,
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
    fn science_manifest_cannot_be_opened_as_compact_or_share_its_identity() {
        let dir = temp_dir();
        let path = RunManifest::path(&dir, "science-run");
        let projection = ManifestProjection {
            map_proj: 1,
            truelat1_deg: 30.0,
            truelat2_deg: 60.0,
            stand_lon_deg: -85.0,
            cen_lat_deg: 35.0,
            cen_lon_deg: -97.0,
            dx_m: 3000.0,
            dy_m: 3000.0,
        };
        let manifest = RunManifest::load_or_new_profiled(
            &path,
            StorageProfile::ScienceCloudF16,
            "science-run",
            4,
            3,
            2,
            0.0,
            250.0,
            vec![
                "hgt".into(),
                "landmask".into(),
                "tsk".into(),
                "u10".into(),
                "v10".into(),
            ],
            projection,
        )
        .unwrap();
        assert_eq!(manifest.format_version, SCIENCE_F16_FORMAT_VERSION);
        assert_eq!(manifest.storage_profile, StorageProfile::ScienceCloudF16);
        manifest.save(&path).unwrap();
        let error = RunManifest::load(&path).unwrap_err();
        assert!(matches!(error, BrickError::CacheMismatch(_)), "{error}");
        assert!(error.to_string().contains("never substituted"));
        assert_eq!(
            RunManifest::load_for_profile(&path, StorageProfile::ScienceCloudF16)
                .unwrap()
                .storage_profile,
            StorageProfile::ScienceCloudF16
        );
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
            has_cloud_fraction: true,
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
        // Strip optional fields to exercise their conservative defaults.
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = value.as_object_mut().unwrap();
        obj.remove("source_bytes");
        obj.remove("source_mtime_unix");
        obj.remove("anchor");
        obj.remove("has_cloud_fraction");
        let old: ManifestTimestep = serde_json::from_value(value).unwrap();
        assert_eq!(old.source_bytes, None);
        assert_eq!(old.source_mtime_unix, None);
        assert_eq!(old.anchor, None);
        assert!(!old.has_cloud_fraction);
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
        // A v3 manifest lacks SSB v4+'s snow-subset / cloud-fraction payload and is
        // refused the same way. The remedy is a source re-ingest, never silent reuse.
        std::fs::write(
            &path,
            r#"{ "format_version": 3, "run_id": "pre_fractional_clouds" }"#,
        )
        .unwrap();
        assert!(matches!(
            RunManifest::load(&path),
            Err(BrickError::UnsupportedManifestVersion { found: 3, .. })
        ));
        // v4 has the same byte layout as v5 but predates the corrected WRF/HRRR
        // cloud-fraction semantics, so it too must be rejected and regenerated.
        std::fs::write(
            &path,
            r#"{ "format_version": 4, "run_id": "old_fraction_semantics" }"#,
        )
        .unwrap();
        assert!(matches!(
            RunManifest::load(&path),
            Err(BrickError::UnsupportedManifestVersion { found: 4, .. })
        ));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The v4 -> v5 INGEST self-heal: `load_or_new` SUPERSEDES an old-format
    /// manifest with a fresh one
    /// at the current version instead of propagating the read refusal — during ingest
    /// the source wrfout is present, so the regenerable cache regenerates. The READ
    /// paths (`RunManifest::load` direct, a cached run.json open) keep the hard
    /// remedy-bearing refusal, covered by the test above.
    #[test]
    fn load_or_new_supersedes_an_old_format_manifest() {
        let dir = temp_dir();
        let path = RunManifest::path(&dir, "old_run");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, r#"{ "format_version": 4, "run_id": "old_run" }"#).unwrap();
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
    fn read_ssb_rejects_corrupt_zlib_payload_cleanly() {
        let dir = temp_dir();
        let path = dir.join("whole.ssb");
        write_ssb(&path, &tiny_brick()).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let header_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let payload_start = 12 + header_len;
        bytes[payload_start] = 0;
        let corrupt_path = dir.join("corrupt.ssb");
        std::fs::write(&corrupt_path, bytes).unwrap();

        let err = read_ssb(&corrupt_path).unwrap_err();
        assert!(
            matches!(&err, BrickError::Io(_)),
            "a corrupt zlib stream must remain an I/O error, got {err:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_ssb_rejects_one_extra_decompressed_payload_byte() {
        let dir = temp_dir();
        let path = dir.join("extra-payload.ssb");
        let brick = tiny_brick();
        let header_json = serde_json::to_vec(&brick.header()).unwrap();
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        write_payload(&mut encoder, &brick).unwrap();
        encoder.write_all(&[0xa5]).unwrap();
        let compressed = encoder.finish().unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&SSB_MAGIC);
        bytes.extend_from_slice(&SSB_FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(header_json.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&header_json);
        bytes.extend_from_slice(&compressed);
        std::fs::write(&path, bytes).unwrap();

        let expected_payload = brick.ext_liquid.len() * (CHANNELS_3D.len() + 2)
            + brick.hgt.len() * brick.planes_2d_list().len() * 4;
        let Err(BrickError::Truncated(message)) = read_ssb(&path) else {
            panic!("one extra decompressed byte must be rejected as an oversized payload");
        };
        assert_eq!(
            message,
            format!("payload exceeds the {expected_payload} bytes the header describes")
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
