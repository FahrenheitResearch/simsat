//! Opt-in precision audit for the compact SSB volume representation.
//!
//! The production brick deliberately stores extinction and water vapour as
//! per-volume log-u8 channels and temperature as f16 Celsius.  This module is a
//! development/validation instrument, not a second brick format: WRF ingest can
//! hand it the post-resample f32 fields immediately before encoding, and it
//! compares those fields with the values the current brick actually decodes.
//!
//! Two radiance-facing diagnostics are emitted:
//!
//! * a fixed-geometry, black-surface, unit-phase single-scatter visible response,
//!   `R = 1/8 * (1 - exp(-2 tau))`, evaluated from vertical cloud optical depth;
//! * a nadir, layer-exact version of SimSat's current gray Band-13 column transfer,
//!   using the official FM4 response for Planck emission/BT inversion.
//!
//! They intentionally remove navigation, interpolation, terrain reflectance and
//! display tone mapping so their A/B difference isolates SSB representation error.
//! They are sensitivity operators, not substitutes for a full scene render.

use std::fs;
use std::path::{Path, PathBuf};

use image::{GrayImage, Luma, Rgb, RgbImage};
use serde::Serialize;

use crate::bricks::{
    LogQuant, ScienceCloudF16Payload, StorageProfile, VolumeBrick, decode_log2_f16,
    decode_temperature_kelvin, encode_log2_f16, f16_bits_to_f32, read_ssb_profiled,
    write_ssb_science_f16_payload,
};
use crate::ir::IrConfig;
use crate::optics::{
    HydrometeorClass, IR_SURFACE_EMISSIVITY, IR_WV_CONTINUUM_MASS_ABS_M2_KG,
    ir_absorption_from_ext, wv_absorption,
};
use crate::thermal_sensor::ThermalSensor;

/// User-facing configuration for one pre-quantization precision audit.
#[derive(Debug, Clone)]
pub struct PrecisionAuditConfig {
    /// Directory that receives `report.json`, a compact text summary, and PNGs.
    pub output_dir: PathBuf,
    /// Lower visible optical-depth bound for the thin-cloud subset.
    pub thin_tau_min: f64,
    /// Inclusive upper visible optical-depth bound for the thin-cloud subset.
    pub thin_tau_max: f64,
}

impl PrecisionAuditConfig {
    pub fn new(output_dir: PathBuf) -> Self {
        Self {
            output_dir,
            thin_tau_min: 0.01,
            thin_tau_max: 0.30,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct StreamingError {
    count: usize,
    sum: f64,
    sum_abs: f64,
    sum_sq: f64,
    max_abs: f64,
}

impl StreamingError {
    fn add(&mut self, error: f64) {
        if !error.is_finite() {
            return;
        }
        self.count += 1;
        self.sum += error;
        self.sum_abs += error.abs();
        self.sum_sq += error * error;
        self.max_abs = self.max_abs.max(error.abs());
    }

    fn summary(self) -> ErrorSummary {
        if self.count == 0 {
            return ErrorSummary::default();
        }
        let n = self.count as f64;
        ErrorSummary {
            count: self.count,
            bias: self.sum / n,
            mae: self.sum_abs / n,
            rmse: (self.sum_sq / n).sqrt(),
            max_abs: self.max_abs,
        }
    }
}

/// Compact field-space error summary. Relative errors are reported separately.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct ErrorSummary {
    pub count: usize,
    pub bias: f64,
    pub mae: f64,
    pub rmse: f64,
    pub max_abs: f64,
}

/// Native-f32 versus decoded-compact statistics for one SSB channel.
#[derive(Debug, Clone, Serialize)]
pub struct FieldPrecisionMetrics {
    pub name: String,
    pub encoding: String,
    pub samples: usize,
    pub finite_positive_samples: usize,
    pub encoded_zero_samples: usize,
    pub encoded_floor_samples: usize,
    pub native_positive_below_log_floor: usize,
    pub vmin: Option<f64>,
    pub vmax: Option<f64>,
    pub absolute_error: ErrorSummary,
    /// `(decoded - native) / native`, only where the native value is at or above
    /// the encoding floor. Positive sub-floor samples are counted separately;
    /// their relative error is mathematically unbounded as native approaches zero.
    pub relative_error: ErrorSummary,
}

/// Pixel-space A/B metrics, including exact empirical absolute-error quantiles.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct MapPrecisionMetrics {
    pub pixels: usize,
    pub bias: f64,
    pub mae: f64,
    pub rmse: f64,
    pub max_abs: f64,
    pub p95_abs: f64,
    pub p99_abs: f64,
}

/// Machine-readable output from one audit.
#[derive(Debug, Clone, Serialize)]
pub struct PrecisionAuditReport {
    pub schema: String,
    pub source: String,
    pub reference: String,
    pub compact_representation: String,
    pub dimensions: [usize; 3],
    pub z_min_m: f64,
    pub dz_m: f64,
    pub ssb_bytes: u64,
    pub thin_tau_range: [f64; 2],
    pub fields: Vec<FieldPrecisionMetrics>,
    pub visible_all_cloud_response_error: MapPrecisionMetrics,
    pub visible_thin_cloud_response_error: MapPrecisionMetrics,
    pub band13_bt_all_error_k: MapPrecisionMetrics,
    pub band13_bt_thin_cloud_error_k: MapPrecisionMetrics,
    pub band13_error_attribution: Band13ErrorAttribution,
    pub science_cloud_f16: ScienceCloudF16Audit,
    pub visible_difference_png_scale: f64,
    pub band13_difference_png_scale_k: f64,
    pub artifacts: Vec<String>,
    pub limitations: Vec<String>,
}

/// Precision of the narrow ScienceCloudF16 experiment. Additive cloud extinction
/// and the snow-only thermal subset change; temperature and qvapor remain compact.
#[derive(Debug, Clone, Serialize)]
pub struct ScienceCloudF16Audit {
    pub encoding: String,
    pub stored_extinction_channels: Vec<String>,
    pub added_uncompressed_bytes: u64,
    pub profiled_ssb_bytes: u64,
    pub compact_ssb_bytes: u64,
    pub visible_all_cloud_response_error: MapPrecisionMetrics,
    pub visible_thin_cloud_response_error: MapPrecisionMetrics,
    pub band13_profile_all_error_k: MapPrecisionMetrics,
    pub band13_profile_thin_cloud_error_k: MapPrecisionMetrics,
    pub band13_cloud_extinction_only_all_error_k: MapPrecisionMetrics,
    pub band13_cloud_extinction_only_thin_error_k: MapPrecisionMetrics,
}

/// One-at-a-time compact-channel substitutions against the all-f32 reference.
/// These are sensitivity attributions rather than additive terms (radiative
/// transfer is nonlinear, so the three errors need not sum to the full A/B).
#[derive(Debug, Clone, Serialize)]
pub struct Band13ErrorAttribution {
    pub compact_cloud_extinction_only_all_k: MapPrecisionMetrics,
    pub compact_temperature_only_all_k: MapPrecisionMetrics,
    pub compact_qvapor_only_all_k: MapPrecisionMetrics,
    pub compact_cloud_extinction_only_thin_k: MapPrecisionMetrics,
    pub compact_temperature_only_thin_k: MapPrecisionMetrics,
    pub compact_qvapor_only_thin_k: MapPrecisionMetrics,
}

/// Ingest-side scratch state. It exists only when a precision audit is requested;
/// the normal ingest path allocates none of these science-reference buffers.
pub(crate) struct PrecisionAuditCapture {
    config: PrecisionAuditConfig,
    nx: usize,
    ny: usize,
    nz: usize,
    z_min_m: f64,
    dz_m: f64,
    native_tau: Vec<f64>,
    compact_tau: Vec<f64>,
    native_cloud_ir_beta: Vec<f32>,
    compact_cloud_ir_beta: Vec<f32>,
    science_f16_payload: ScienceCloudF16Payload,
    native_temperature_k: Option<Vec<f32>>,
    native_qvapor: Option<Vec<f32>>,
    fields: Vec<FieldPrecisionMetrics>,
}

impl PrecisionAuditCapture {
    pub(crate) fn new(
        config: PrecisionAuditConfig,
        nx: usize,
        ny: usize,
        nz: usize,
        z_min_m: f64,
        dz_m: f64,
    ) -> Self {
        let cells_2d = nx * ny;
        let cells_3d = cells_2d * nz;
        Self {
            config,
            nx,
            ny,
            nz,
            z_min_m,
            dz_m,
            native_tau: vec![0.0; cells_2d],
            compact_tau: vec![0.0; cells_2d],
            native_cloud_ir_beta: vec![0.0; cells_3d],
            compact_cloud_ir_beta: vec![0.0; cells_3d],
            science_f16_payload: ScienceCloudF16Payload::default(),
            native_temperature_k: None,
            native_qvapor: None,
            fields: Vec::new(),
        }
    }

    /// Record one independently additive log-u8 channel. Precipitation and its
    /// snow-only auxiliary go through [`Self::record_precipitation`] together so
    /// the thermal split cannot count snow twice.
    pub(crate) fn record_extinction(
        &mut self,
        name: &str,
        class: Option<HydrometeorClass>,
        contributes_to_render_total: bool,
        native: &[f32],
        quant: LogQuant,
        codes: &[u8],
    ) {
        self.fields
            .push(log_field_metrics(name, native, quant, codes));
        match name {
            "ext_liquid" => {
                self.science_f16_payload.ext_liquid =
                    native.iter().map(|&v| encode_log2_f16(v)).collect()
            }
            "ext_ice" => {
                self.science_f16_payload.ext_ice =
                    native.iter().map(|&v| encode_log2_f16(v)).collect()
            }
            _ => {}
        }
        if !contributes_to_render_total {
            return;
        }
        debug_assert_eq!(native.len(), self.native_cloud_ir_beta.len());
        debug_assert_eq!(native.len(), codes.len());
        let plane = self.nx * self.ny;
        for (idx, (&value, &code)) in native.iter().zip(codes).enumerate() {
            let native_value = finite_nonnegative(value);
            let compact_value = quant.decode(code) as f64;
            let column = idx % plane;
            self.native_tau[column] += native_value * self.dz_m;
            self.compact_tau[column] += compact_value * self.dz_m;
            if let Some(class) = class {
                self.native_cloud_ir_beta[idx] +=
                    ir_absorption_from_ext(class, native_value) as f32;
                self.compact_cloud_ir_beta[idx] +=
                    ir_absorption_from_ext(class, compact_value) as f32;
            }
        }
    }

    /// Record the precipitation total together with its snow-only auxiliary.
    /// This mirrors `IrVolume::cloud_ir_absorption`: snow is clamped to its parent
    /// and the rain/graupel remainder receives the large-particle coefficient.
    pub(crate) fn record_precipitation(
        &mut self,
        native_snow: &[f32],
        snow_quant: LogQuant,
        snow_codes: &[u8],
        native_total: &[f32],
        total_quant: LogQuant,
        total_codes: &[u8],
    ) {
        self.fields.push(log_field_metrics(
            "ext_snow",
            native_snow,
            snow_quant,
            snow_codes,
        ));
        self.fields.push(log_field_metrics(
            "ext_precip",
            native_total,
            total_quant,
            total_codes,
        ));
        self.science_f16_payload.ext_snow = native_snow
            .iter()
            .map(|&value| encode_log2_f16(value))
            .collect();
        self.science_f16_payload.ext_precip = native_total
            .iter()
            .map(|&value| encode_log2_f16(value))
            .collect();

        debug_assert_eq!(native_snow.len(), native_total.len());
        debug_assert_eq!(native_total.len(), self.native_cloud_ir_beta.len());
        let plane = self.nx * self.ny;
        for idx in 0..native_total.len() {
            let native_precip = finite_nonnegative(native_total[idx]);
            let native_snow = finite_nonnegative(native_snow[idx]).min(native_precip);
            let compact_precip = total_quant.decode(total_codes[idx]) as f64;
            let compact_snow = (snow_quant.decode(snow_codes[idx]) as f64).min(compact_precip);
            let column = idx % plane;
            self.native_tau[column] += native_precip * self.dz_m;
            self.compact_tau[column] += compact_precip * self.dz_m;
            self.native_cloud_ir_beta[idx] +=
                (ir_absorption_from_ext(HydrometeorClass::Snow, native_snow)
                    + ir_absorption_from_ext(HydrometeorClass::Rain, native_precip - native_snow))
                    as f32;
            self.compact_cloud_ir_beta[idx] +=
                (ir_absorption_from_ext(HydrometeorClass::Snow, compact_snow)
                    + ir_absorption_from_ext(HydrometeorClass::Rain, compact_precip - compact_snow))
                    as f32;
        }
    }

    pub(crate) fn record_log_field(
        &mut self,
        name: &str,
        native: &[f32],
        quant: LogQuant,
        codes: &[u8],
    ) {
        self.fields
            .push(log_field_metrics(name, native, quant, codes));
    }

    pub(crate) fn record_temperature(&mut self, native_k: Vec<f32>, compact_bits: &[u16]) {
        self.fields
            .push(temperature_field_metrics(&native_k, compact_bits));
        self.native_temperature_k = Some(native_k);
    }

    pub(crate) fn record_qvapor(&mut self, native: Vec<f32>, quant: LogQuant, codes: &[u8]) {
        self.fields
            .push(log_field_metrics("qvapor", &native, quant, codes));
        self.native_qvapor = Some(native);
    }

    pub(crate) fn record_cloud_fraction(&mut self, native: &[f32], codes: &[u8]) {
        self.fields
            .push(linear_u8_field_metrics("cloud_fraction", native, codes));
    }

    /// Finish the audit after the ordinary brick has been assembled and written.
    pub(crate) fn finish(
        self,
        source: &Path,
        brick: &VolumeBrick,
        ssb_bytes: u64,
    ) -> Result<PathBuf, String> {
        let native_temperature_k = self
            .native_temperature_k
            .as_deref()
            .ok_or_else(|| "precision audit did not capture native temperature".to_string())?;
        let native_qvapor = self
            .native_qvapor
            .as_deref()
            .ok_or_else(|| "precision audit did not capture native qvapor".to_string())?;
        let compact_temperature_k = decode_temperature_kelvin(&brick.temperature_f16);
        let qv_quant = brick.quant.get("qvapor");
        let compact_qvapor: Vec<f32> = brick.qvapor.iter().map(|&c| qv_quant.decode(c)).collect();

        // Audit the bytes that were actually emitted, not an encoder-only in-memory
        // reconstruction. The profile-aware reader also verifies the v7 header,
        // payload length, channel order, and zlib stream before metrics are formed.
        fs::create_dir_all(&self.config.output_dir)
            .map_err(|e| format!("create precision-audit output: {e}"))?;
        let science_ssb_path = self.config.output_dir.join("science-cloud-f16-v7.ssb");
        let profiled_ssb_bytes =
            write_ssb_science_f16_payload(&science_ssb_path, brick, &self.science_f16_payload)
                .map_err(|e| format!("write ScienceCloudF16 audit brick: {e}"))?;
        let profiled = read_ssb_profiled(&science_ssb_path)
            .map_err(|e| format!("read ScienceCloudF16 audit brick: {e}"))?;
        if profiled.profile != StorageProfile::ScienceCloudF16 {
            return Err("ScienceCloudF16 readback reported the wrong storage profile".to_string());
        }
        let mut readback_brick = profiled.brick;
        let science_payload = readback_brick
            .science_cloud_f16
            .take()
            .ok_or_else(|| "ScienceCloudF16 readback omitted its extension".to_string())?;
        readback_brick.storage_profile = StorageProfile::CompactU8;
        if readback_brick != *brick {
            return Err("ScienceCloudF16 readback changed the compact fallback brick".to_string());
        }
        if science_payload != self.science_f16_payload {
            return Err(
                "ScienceCloudF16 extension changed during write/read roundtrip".to_string(),
            );
        }
        drop(readback_brick);
        drop(self.science_f16_payload);
        let cells_2d = self.nx * self.ny;
        let cells_3d = cells_2d * self.nz;
        let mut science_f16_tau = vec![0.0f64; cells_2d];
        let mut science_f16_cloud_ir_beta = vec![0.0f32; cells_3d];
        for idx in 0..cells_3d {
            let liquid = decode_log2_f16(science_payload.ext_liquid[idx]) as f64;
            let ice = decode_log2_f16(science_payload.ext_ice[idx]) as f64;
            let precip = decode_log2_f16(science_payload.ext_precip[idx]) as f64;
            let snow = (decode_log2_f16(science_payload.ext_snow[idx]) as f64).min(precip);
            science_f16_tau[idx % cells_2d] += (liquid + ice + precip) * self.dz_m;
            science_f16_cloud_ir_beta[idx] =
                (ir_absorption_from_ext(HydrometeorClass::CloudLiquid, liquid)
                    + ir_absorption_from_ext(HydrometeorClass::Ice, ice)
                    + ir_absorption_from_ext(HydrometeorClass::Snow, snow)
                    + ir_absorption_from_ext(HydrometeorClass::Rain, precip - snow))
                    as f32;
        }

        let native_visible: Vec<f32> = self
            .native_tau
            .iter()
            .map(|&tau| visible_single_scatter_response(tau) as f32)
            .collect();
        let compact_visible: Vec<f32> = self
            .compact_tau
            .iter()
            .map(|&tau| visible_single_scatter_response(tau) as f32)
            .collect();
        let science_f16_visible: Vec<f32> = science_f16_tau
            .iter()
            .map(|&tau| visible_single_scatter_response(tau) as f32)
            .collect();
        let cloud_mask: Vec<bool> = self.native_tau.iter().map(|&tau| tau > 0.0).collect();
        let thin_mask: Vec<bool> = self
            .native_tau
            .iter()
            .map(|&tau| tau >= self.config.thin_tau_min && tau <= self.config.thin_tau_max)
            .collect();

        let cfg = IrConfig::band13_with_sensor(ThermalSensor::GoesRAbiBand13Fm4);
        let native_bt = column_band13_bt(
            ColumnRtInput {
                nx: self.nx,
                ny: self.ny,
                nz: self.nz,
                z_min_m: self.z_min_m,
                dz_m: self.dz_m,
                cloud_ir_beta: &self.native_cloud_ir_beta,
                temperature_k: native_temperature_k,
                qvapor: native_qvapor,
                hgt: &brick.hgt,
                tsk: &brick.tsk,
            },
            cfg,
        );
        let compact_bt = column_band13_bt(
            ColumnRtInput {
                nx: self.nx,
                ny: self.ny,
                nz: self.nz,
                z_min_m: self.z_min_m,
                dz_m: self.dz_m,
                cloud_ir_beta: &self.compact_cloud_ir_beta,
                temperature_k: &compact_temperature_k,
                qvapor: &compact_qvapor,
                hgt: &brick.hgt,
                tsk: &brick.tsk,
            },
            cfg,
        );
        let compact_cloud_only_bt = column_band13_bt(
            ColumnRtInput {
                nx: self.nx,
                ny: self.ny,
                nz: self.nz,
                z_min_m: self.z_min_m,
                dz_m: self.dz_m,
                cloud_ir_beta: &self.compact_cloud_ir_beta,
                temperature_k: native_temperature_k,
                qvapor: native_qvapor,
                hgt: &brick.hgt,
                tsk: &brick.tsk,
            },
            cfg,
        );
        let compact_temperature_only_bt = column_band13_bt(
            ColumnRtInput {
                nx: self.nx,
                ny: self.ny,
                nz: self.nz,
                z_min_m: self.z_min_m,
                dz_m: self.dz_m,
                cloud_ir_beta: &self.native_cloud_ir_beta,
                temperature_k: &compact_temperature_k,
                qvapor: native_qvapor,
                hgt: &brick.hgt,
                tsk: &brick.tsk,
            },
            cfg,
        );
        let compact_qvapor_only_bt = column_band13_bt(
            ColumnRtInput {
                nx: self.nx,
                ny: self.ny,
                nz: self.nz,
                z_min_m: self.z_min_m,
                dz_m: self.dz_m,
                cloud_ir_beta: &self.native_cloud_ir_beta,
                temperature_k: native_temperature_k,
                qvapor: &compact_qvapor,
                hgt: &brick.hgt,
                tsk: &brick.tsk,
            },
            cfg,
        );
        let science_f16_cloud_only_bt = column_band13_bt(
            ColumnRtInput {
                nx: self.nx,
                ny: self.ny,
                nz: self.nz,
                z_min_m: self.z_min_m,
                dz_m: self.dz_m,
                cloud_ir_beta: &science_f16_cloud_ir_beta,
                temperature_k: native_temperature_k,
                qvapor: native_qvapor,
                hgt: &brick.hgt,
                tsk: &brick.tsk,
            },
            cfg,
        );
        let science_f16_profile_bt = column_band13_bt(
            ColumnRtInput {
                nx: self.nx,
                ny: self.ny,
                nz: self.nz,
                z_min_m: self.z_min_m,
                dz_m: self.dz_m,
                cloud_ir_beta: &science_f16_cloud_ir_beta,
                temperature_k: &compact_temperature_k,
                qvapor: &compact_qvapor,
                hgt: &brick.hgt,
                tsk: &brick.tsk,
            },
            cfg,
        );
        let all_mask = vec![true; self.nx * self.ny];

        let visible_all = map_metrics(&native_visible, &compact_visible, &cloud_mask);
        let visible_thin = map_metrics(&native_visible, &compact_visible, &thin_mask);
        let bt_all = map_metrics(&native_bt, &compact_bt, &all_mask);
        let bt_thin = map_metrics(&native_bt, &compact_bt, &thin_mask);
        let band13_error_attribution = Band13ErrorAttribution {
            compact_cloud_extinction_only_all_k: map_metrics(
                &native_bt,
                &compact_cloud_only_bt,
                &all_mask,
            ),
            compact_temperature_only_all_k: map_metrics(
                &native_bt,
                &compact_temperature_only_bt,
                &all_mask,
            ),
            compact_qvapor_only_all_k: map_metrics(&native_bt, &compact_qvapor_only_bt, &all_mask),
            compact_cloud_extinction_only_thin_k: map_metrics(
                &native_bt,
                &compact_cloud_only_bt,
                &thin_mask,
            ),
            compact_temperature_only_thin_k: map_metrics(
                &native_bt,
                &compact_temperature_only_bt,
                &thin_mask,
            ),
            compact_qvapor_only_thin_k: map_metrics(
                &native_bt,
                &compact_qvapor_only_bt,
                &thin_mask,
            ),
        };
        let science_cloud_f16 = ScienceCloudF16Audit {
            encoding: "binary16(log2(beta_m^-1)); -infinity is exact zero".to_string(),
            stored_extinction_channels: vec![
                "ext_liquid".to_string(),
                "ext_ice".to_string(),
                "ext_snow (auxiliary subset)".to_string(),
                "ext_precip".to_string(),
            ],
            added_uncompressed_bytes: (self.nx as u64)
                .saturating_mul(self.ny as u64)
                .saturating_mul(self.nz as u64)
                .saturating_mul(8),
            profiled_ssb_bytes,
            compact_ssb_bytes: ssb_bytes,
            visible_all_cloud_response_error: map_metrics(
                &native_visible,
                &science_f16_visible,
                &cloud_mask,
            ),
            visible_thin_cloud_response_error: map_metrics(
                &native_visible,
                &science_f16_visible,
                &thin_mask,
            ),
            band13_profile_all_error_k: map_metrics(&native_bt, &science_f16_profile_bt, &all_mask),
            band13_profile_thin_cloud_error_k: map_metrics(
                &native_bt,
                &science_f16_profile_bt,
                &thin_mask,
            ),
            band13_cloud_extinction_only_all_error_k: map_metrics(
                &native_bt,
                &science_f16_cloud_only_bt,
                &all_mask,
            ),
            band13_cloud_extinction_only_thin_error_k: map_metrics(
                &native_bt,
                &science_f16_cloud_only_bt,
                &thin_mask,
            ),
        };

        let visible_diff_scale =
            difference_scale(&native_visible, &compact_visible, &thin_mask, 1e-7);
        let bt_diff_scale = difference_scale(&native_bt, &compact_bt, &all_mask, 1e-3);
        write_visible_png(
            &self.config.output_dir.join("visible-native.png"),
            &native_visible,
            self.nx,
            self.ny,
            self.config.thin_tau_max,
        )?;
        write_visible_png(
            &self.config.output_dir.join("visible-compact-u8.png"),
            &compact_visible,
            self.nx,
            self.ny,
            self.config.thin_tau_max,
        )?;
        write_signed_difference_png(
            &self
                .config
                .output_dir
                .join("visible-compact-minus-native.png"),
            &native_visible,
            &compact_visible,
            self.nx,
            self.ny,
            visible_diff_scale,
        )?;
        write_visible_png(
            &self.config.output_dir.join("visible-science-f16.png"),
            &science_f16_visible,
            self.nx,
            self.ny,
            self.config.thin_tau_max,
        )?;
        write_signed_difference_png(
            &self
                .config
                .output_dir
                .join("visible-science-minus-native.png"),
            &native_visible,
            &science_f16_visible,
            self.nx,
            self.ny,
            visible_diff_scale,
        )?;
        write_bt_png(
            &self.config.output_dir.join("band13-bt-native.png"),
            &native_bt,
            self.nx,
            self.ny,
        )?;
        write_bt_png(
            &self.config.output_dir.join("band13-bt-compact-u8.png"),
            &compact_bt,
            self.nx,
            self.ny,
        )?;
        write_signed_difference_png(
            &self
                .config
                .output_dir
                .join("band13-bt-compact-minus-native.png"),
            &native_bt,
            &compact_bt,
            self.nx,
            self.ny,
            bt_diff_scale,
        )?;
        write_bt_png(
            &self.config.output_dir.join("band13-bt-science-f16.png"),
            &science_f16_profile_bt,
            self.nx,
            self.ny,
        )?;
        write_signed_difference_png(
            &self
                .config
                .output_dir
                .join("band13-bt-science-minus-native.png"),
            &native_bt,
            &science_f16_profile_bt,
            self.nx,
            self.ny,
            bt_diff_scale,
        )?;

        let artifacts = vec![
            "science-cloud-f16-v7.ssb".to_string(),
            "visible-native.png".to_string(),
            "visible-compact-u8.png".to_string(),
            "visible-compact-minus-native.png".to_string(),
            "visible-science-f16.png".to_string(),
            "visible-science-minus-native.png".to_string(),
            "band13-bt-native.png".to_string(),
            "band13-bt-compact-u8.png".to_string(),
            "band13-bt-compact-minus-native.png".to_string(),
            "band13-bt-science-f16.png".to_string(),
            "band13-bt-science-minus-native.png".to_string(),
        ];
        let report = PrecisionAuditReport {
            schema: "simsat-ssb-precision-audit-v3".to_string(),
            source: source.display().to_string(),
            reference: "post-resample f32 WRF fields immediately before SSB encoding"
                .to_string(),
            compact_representation:
                "current SSB v6 per-volume log-u8 extinction/qvapor + f16-Celsius temperature"
                    .to_string(),
            dimensions: [self.nx, self.ny, self.nz],
            z_min_m: self.z_min_m,
            dz_m: self.dz_m,
            ssb_bytes,
            thin_tau_range: [self.config.thin_tau_min, self.config.thin_tau_max],
            fields: self.fields,
            visible_all_cloud_response_error: visible_all,
            visible_thin_cloud_response_error: visible_thin,
            band13_bt_all_error_k: bt_all,
            band13_bt_thin_cloud_error_k: bt_thin,
            band13_error_attribution: band13_error_attribution.clone(),
            science_cloud_f16: science_cloud_f16.clone(),
            visible_difference_png_scale: visible_diff_scale,
            band13_difference_png_scale_k: bt_diff_scale,
            artifacts,
            limitations: vec![
                "The visible operator is a fixed-geometry single-scatter sensitivity test, not the full SimSat terrain/atmosphere/cloud march.".to_string(),
                "Cloud-fraction linear-u8 error is reported in field space but is not applied to the visible response; that response isolates log-u8 extinction precision.".to_string(),
                "The Band-13 operator is a nadir layer-column transfer using SimSat's current gray cloud/gas absorption; it removes slant interpolation and navigation so the A/B isolates representation error.".to_string(),
                "The audit starts after SimSat's vertical resampling and optics conversion, so it does not measure errors from source-variable selection, resampling, assumed particle size, or gray spectroscopy.".to_string(),
                "The compact brick remains unchanged; this audit does not add pressure, ozone, native hydrometeor moments, or interface heights.".to_string(),
            ],
        };
        let report_path = self.config.output_dir.join("report.json");
        let json = serde_json::to_vec_pretty(&report)
            .map_err(|e| format!("serialize precision-audit report: {e}"))?;
        fs::write(&report_path, json)
            .map_err(|e| format!("write {}: {e}", report_path.display()))?;
        let summary = format!(
            "SimSat compact SSB precision audit\n\nsource: {}\nreference: post-resample f32 before encoding\ndims: {} x {} x {}\nthin tau: {:.3}..{:.3} ({} pixels)\nvisible thin response: bias={:.8}, MAE={:.8}, RMSE={:.8}, p99_abs={:.8}, max_abs={:.8}\nScienceCloudF16 visible thin: bias={:.8}, MAE={:.8}, RMSE={:.8}, p99_abs={:.8}, max_abs={:.8}\nBand-13 BT all: bias={:.6} K, MAE={:.6} K, RMSE={:.6} K, p99_abs={:.6} K, max_abs={:.6} K\nBand-13 BT thin: bias={:.6} K, MAE={:.6} K, RMSE={:.6} K, p99_abs={:.6} K, max_abs={:.6} K\nScienceCloudF16 Band-13 profile all/thin RMSE: {:.6} / {:.6} K\nScienceCloudF16 cloud-only all/thin RMSE: {:.6} / {:.6} K\nScienceCloudF16 SSB bytes compact/profile: {} / {}\nBand-13 all RMSE attribution: cloud_u8={:.6} K, temperature_f16={:.6} K, qvapor_u8={:.6} K\nBand-13 thin RMSE attribution: cloud_u8={:.6} K, temperature_f16={:.6} K, qvapor_u8={:.6} K\n\nSee report.json for field-space metrics, formulas, image scales, and limitations.\n",
            source.display(),
            self.nx,
            self.ny,
            self.nz,
            self.config.thin_tau_min,
            self.config.thin_tau_max,
            visible_thin.pixels,
            visible_thin.bias,
            visible_thin.mae,
            visible_thin.rmse,
            visible_thin.p99_abs,
            visible_thin.max_abs,
            science_cloud_f16.visible_thin_cloud_response_error.bias,
            science_cloud_f16.visible_thin_cloud_response_error.mae,
            science_cloud_f16.visible_thin_cloud_response_error.rmse,
            science_cloud_f16.visible_thin_cloud_response_error.p99_abs,
            science_cloud_f16.visible_thin_cloud_response_error.max_abs,
            bt_all.bias,
            bt_all.mae,
            bt_all.rmse,
            bt_all.p99_abs,
            bt_all.max_abs,
            bt_thin.bias,
            bt_thin.mae,
            bt_thin.rmse,
            bt_thin.p99_abs,
            bt_thin.max_abs,
            science_cloud_f16.band13_profile_all_error_k.rmse,
            science_cloud_f16.band13_profile_thin_cloud_error_k.rmse,
            science_cloud_f16
                .band13_cloud_extinction_only_all_error_k
                .rmse,
            science_cloud_f16
                .band13_cloud_extinction_only_thin_error_k
                .rmse,
            science_cloud_f16.compact_ssb_bytes,
            science_cloud_f16.profiled_ssb_bytes,
            band13_error_attribution
                .compact_cloud_extinction_only_all_k
                .rmse,
            band13_error_attribution.compact_temperature_only_all_k.rmse,
            band13_error_attribution.compact_qvapor_only_all_k.rmse,
            band13_error_attribution
                .compact_cloud_extinction_only_thin_k
                .rmse,
            band13_error_attribution
                .compact_temperature_only_thin_k
                .rmse,
            band13_error_attribution.compact_qvapor_only_thin_k.rmse,
        );
        fs::write(self.config.output_dir.join("SUMMARY.txt"), summary)
            .map_err(|e| format!("write precision-audit summary: {e}"))?;
        Ok(report_path)
    }
}

fn finite_nonnegative(value: f32) -> f64 {
    if value.is_finite() && value > 0.0 {
        value as f64
    } else {
        0.0
    }
}

fn log_field_metrics(
    name: &str,
    native: &[f32],
    quant: LogQuant,
    codes: &[u8],
) -> FieldPrecisionMetrics {
    assert_eq!(native.len(), codes.len());
    let mut absolute = StreamingError::default();
    let mut relative = StreamingError::default();
    let mut finite_positive = 0usize;
    let mut zero = 0usize;
    let mut floor = 0usize;
    let mut below_floor = 0usize;
    for (&value, &code) in native.iter().zip(codes) {
        zero += usize::from(code == 0);
        floor += usize::from(code == 1);
        let decoded = quant.decode(code) as f64;
        let reference = if value.is_finite() { value as f64 } else { 0.0 };
        absolute.add(decoded - reference);
        if value.is_finite() && value > 0.0 {
            finite_positive += 1;
            below_floor += usize::from((value as f64) < quant.vmin);
            if value as f64 >= quant.vmin {
                relative.add((decoded - value as f64) / value as f64);
            }
        }
    }
    FieldPrecisionMetrics {
        name: name.to_string(),
        encoding: "per-volume log-u8".to_string(),
        samples: native.len(),
        finite_positive_samples: finite_positive,
        encoded_zero_samples: zero,
        encoded_floor_samples: floor,
        native_positive_below_log_floor: below_floor,
        vmin: Some(quant.vmin),
        vmax: Some(quant.vmax),
        absolute_error: absolute.summary(),
        relative_error: relative.summary(),
    }
}

fn temperature_field_metrics(native_k: &[f32], compact_bits: &[u16]) -> FieldPrecisionMetrics {
    assert_eq!(native_k.len(), compact_bits.len());
    let mut absolute = StreamingError::default();
    let mut relative = StreamingError::default();
    let mut positive = 0usize;
    for (&native, &bits) in native_k.iter().zip(compact_bits) {
        let decoded = f16_bits_to_f32(bits) as f64 + crate::bricks::CELSIUS_OFFSET_K;
        let reference = native as f64;
        absolute.add(decoded - reference);
        if native.is_finite() && native > 0.0 {
            positive += 1;
            relative.add((decoded - reference) / reference);
        }
    }
    FieldPrecisionMetrics {
        name: "temperature".to_string(),
        encoding: "f16 Celsius".to_string(),
        samples: native_k.len(),
        finite_positive_samples: positive,
        encoded_zero_samples: 0,
        encoded_floor_samples: 0,
        native_positive_below_log_floor: 0,
        vmin: None,
        vmax: None,
        absolute_error: absolute.summary(),
        relative_error: relative.summary(),
    }
}

fn linear_u8_field_metrics(name: &str, native: &[f32], codes: &[u8]) -> FieldPrecisionMetrics {
    assert_eq!(native.len(), codes.len());
    let mut absolute = StreamingError::default();
    let mut relative = StreamingError::default();
    let mut positive = 0usize;
    let mut zero = 0usize;
    let mut floor = 0usize;
    for (&native, &code) in native.iter().zip(codes) {
        zero += usize::from(code == 0);
        floor += usize::from(code == 1);
        let decoded = code as f64 / 255.0;
        absolute.add(decoded - native as f64);
        if native.is_finite() && native > 0.0 {
            positive += 1;
            if native as f64 >= 1.0 / 255.0 {
                relative.add((decoded - native as f64) / native as f64);
            }
        }
    }
    FieldPrecisionMetrics {
        name: name.to_string(),
        encoding: "linear-u8 with positive floor".to_string(),
        samples: native.len(),
        finite_positive_samples: positive,
        encoded_zero_samples: zero,
        encoded_floor_samples: floor,
        native_positive_below_log_floor: 0,
        vmin: Some(1.0 / 255.0),
        vmax: Some(1.0),
        absolute_error: absolute.summary(),
        relative_error: relative.summary(),
    }
}

/// Nadir, overhead-sun, black-surface single-scatter response with unit phase and
/// single-scatter albedo. The constant is the standard plane-parallel source term
/// at mu0=mu=1; the audit uses it only as a monotone radiance sensitivity operator.
#[inline]
fn visible_single_scatter_response(tau: f64) -> f64 {
    0.125 * (1.0 - (-2.0 * tau.max(0.0)).exp())
}

struct ColumnRtInput<'a> {
    nx: usize,
    ny: usize,
    nz: usize,
    z_min_m: f64,
    dz_m: f64,
    cloud_ir_beta: &'a [f32],
    temperature_k: &'a [f32],
    qvapor: &'a [f32],
    hgt: &'a [f32],
    tsk: &'a [f32],
}

fn column_band13_bt(input: ColumnRtInput<'_>, cfg: IrConfig) -> Vec<f32> {
    let plane = input.nx * input.ny;
    assert_eq!(input.cloud_ir_beta.len(), plane * input.nz);
    assert_eq!(input.temperature_k.len(), plane * input.nz);
    assert_eq!(input.qvapor.len(), plane * input.nz);
    let mut out = vec![f32::NAN; plane];
    for (cell, out_cell) in out.iter_mut().enumerate() {
        let surface = input.hgt.get(cell).copied().unwrap_or(0.0) as f64;
        let surface_temperature = input.tsk.get(cell).copied().unwrap_or(0.0) as f64;
        if !surface_temperature.is_finite() || surface_temperature <= 0.0 {
            continue;
        }
        let mut transmittance = 1.0f64;
        let mut radiance = 0.0f64;
        for k in (0..input.nz).rev() {
            let idx = k * plane + cell;
            let layer_bottom = input.z_min_m + k as f64 * input.dz_m;
            let layer_fraction =
                ((layer_bottom + input.dz_m - surface) / input.dz_m).clamp(0.0, 1.0);
            let qv = finite_nonnegative(input.qvapor[idx]) * layer_fraction;
            let cloud_beta = finite_nonnegative(input.cloud_ir_beta[idx]);
            let beta = cloud_beta
                + wv_absorption(
                    qv,
                    layer_bottom + 0.5 * input.dz_m,
                    IR_WV_CONTINUUM_MASS_ABS_M2_KG,
                );
            if beta <= 0.0 {
                continue;
            }
            let step_transmittance = (-beta * input.dz_m).exp();
            radiance += transmittance
                * cfg.source_radiance(input.temperature_k[idx] as f64)
                * (1.0 - step_transmittance);
            transmittance *= step_transmittance;
        }
        radiance +=
            transmittance * IR_SURFACE_EMISSIVITY * cfg.source_radiance(surface_temperature);
        *out_cell = cfg.brightness_temperature(radiance) as f32;
    }
    out
}

fn map_metrics(reference: &[f32], compact: &[f32], mask: &[bool]) -> MapPrecisionMetrics {
    assert_eq!(reference.len(), compact.len());
    assert_eq!(reference.len(), mask.len());
    let mut streaming = StreamingError::default();
    let mut abs_errors = Vec::new();
    for ((&reference, &compact), &keep) in reference.iter().zip(compact).zip(mask) {
        if keep && reference.is_finite() && compact.is_finite() {
            let error = compact as f64 - reference as f64;
            streaming.add(error);
            abs_errors.push(error.abs());
        }
    }
    let summary = streaming.summary();
    abs_errors.sort_by(f64::total_cmp);
    MapPrecisionMetrics {
        pixels: summary.count,
        bias: summary.bias,
        mae: summary.mae,
        rmse: summary.rmse,
        max_abs: summary.max_abs,
        p95_abs: empirical_quantile(&abs_errors, 0.95),
        p99_abs: empirical_quantile(&abs_errors, 0.99),
    }
}

fn empirical_quantile(sorted: &[f64], quantile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * quantile.clamp(0.0, 1.0)).round() as usize;
    sorted[index]
}

fn difference_scale(reference: &[f32], compact: &[f32], mask: &[bool], floor: f64) -> f64 {
    let mut values: Vec<f64> = reference
        .iter()
        .zip(compact)
        .zip(mask)
        .filter_map(|((&a, &b), &keep)| {
            (keep && a.is_finite() && b.is_finite()).then_some((b as f64 - a as f64).abs())
        })
        .collect();
    values.sort_by(f64::total_cmp);
    empirical_quantile(&values, 0.99).max(floor)
}

fn write_visible_png(
    path: &Path,
    values: &[f32],
    nx: usize,
    ny: usize,
    thin_tau_max: f64,
) -> Result<(), String> {
    let stretch_max = visible_single_scatter_response(thin_tau_max).max(1e-12);
    let image = GrayImage::from_fn(nx as u32, ny as u32, |x, image_y| {
        let y = ny - 1 - image_y as usize;
        let value = values[y * nx + x as usize] as f64;
        let gray = (value / stretch_max * 255.0).clamp(0.0, 255.0).round() as u8;
        Luma([gray])
    });
    image
        .save(path)
        .map_err(|e| format!("save {}: {e}", path.display()))
}

fn write_bt_png(path: &Path, values: &[f32], nx: usize, ny: usize) -> Result<(), String> {
    let image = GrayImage::from_fn(nx as u32, ny as u32, |x, image_y| {
        let y = ny - 1 - image_y as usize;
        let bt = values[y * nx + x as usize] as f64;
        let gray = if bt.is_finite() {
            ((320.0 - bt) / 140.0 * 255.0).clamp(0.0, 255.0).round() as u8
        } else {
            0
        };
        Luma([gray])
    });
    image
        .save(path)
        .map_err(|e| format!("save {}: {e}", path.display()))
}

fn write_signed_difference_png(
    path: &Path,
    reference: &[f32],
    compact: &[f32],
    nx: usize,
    ny: usize,
    scale: f64,
) -> Result<(), String> {
    let image = RgbImage::from_fn(nx as u32, ny as u32, |x, image_y| {
        let y = ny - 1 - image_y as usize;
        let idx = y * nx + x as usize;
        let delta = compact[idx] as f64 - reference[idx] as f64;
        if !delta.is_finite() {
            return Rgb([0, 0, 0]);
        }
        let amount = (delta.abs() / scale.max(f64::MIN_POSITIVE))
            .clamp(0.0, 1.0)
            .sqrt();
        let low = 16u8;
        let high = (16.0 + 239.0 * amount).round() as u8;
        if delta > 0.0 {
            Rgb([high, low, low])
        } else if delta < 0.0 {
            Rgb([low, low, high])
        } else {
            Rgb([low, low, low])
        }
    });
    image
        .save(path)
        .map_err(|e| format!("save {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bricks::{encode_log_channel, encode_temperature_celsius};

    #[test]
    fn visible_response_is_monotone_and_thin_limit_is_linear() {
        assert_eq!(visible_single_scatter_response(0.0), 0.0);
        assert!(visible_single_scatter_response(0.3) > visible_single_scatter_response(0.1));
        let tau = 1.0e-6;
        assert!((visible_single_scatter_response(tau) / tau - 0.25).abs() < 1.0e-6);
    }

    #[test]
    fn decoded_code_centres_have_zero_field_error() {
        let values = [1.0e-8f32, 3.0e-6, 2.0e-4, 0.01];
        let (quant, codes) = encode_log_channel(&values);
        let centres: Vec<f32> = codes.iter().map(|&code| quant.decode(code)).collect();
        let metrics = log_field_metrics("test", &centres, quant, &codes);
        assert!(metrics.absolute_error.max_abs < 1.0e-12);
    }

    #[test]
    fn f16_celsius_temperature_error_stays_below_a_tenth_kelvin() {
        let native = [183.27f32, 220.13, 273.15, 301.91];
        let bits = encode_temperature_celsius(&native);
        let metrics = temperature_field_metrics(&native, &bits);
        assert!(metrics.absolute_error.max_abs < 0.1);
    }

    #[test]
    fn map_metrics_reports_signed_bias_and_empirical_tail() {
        let reference = [1.0f32, 2.0, 3.0, 4.0];
        let compact = [1.1f32, 1.8, 3.3, 4.0];
        let metrics = map_metrics(&reference, &compact, &[true; 4]);
        assert_eq!(metrics.pixels, 4);
        assert!((metrics.bias - 0.05).abs() < 1.0e-6);
        assert!((metrics.max_abs - 0.3).abs() < 1.0e-6);
        assert_eq!(metrics.p99_abs, metrics.max_abs);
    }

    #[test]
    fn precipitation_audit_matches_shipping_snow_remainder_split() {
        let dir = std::env::temp_dir().join("simsat-precision-snow-split-test");
        let mut capture =
            PrecisionAuditCapture::new(PrecisionAuditConfig::new(dir), 1, 1, 1, 0.0, 250.0);
        let snow = [0.002f32];
        let total = [0.005f32];
        let (snow_quant, snow_codes) = encode_log_channel(&snow);
        let (total_quant, total_codes) = encode_log_channel(&total);
        capture.record_precipitation(
            &snow,
            snow_quant,
            &snow_codes,
            &total,
            total_quant,
            &total_codes,
        );
        let expected_native = ir_absorption_from_ext(HydrometeorClass::Snow, 0.002)
            + ir_absorption_from_ext(HydrometeorClass::Rain, 0.003);
        assert!((capture.native_cloud_ir_beta[0] as f64 - expected_native).abs() < 1.0e-9);
        assert_eq!(capture.science_f16_payload.ext_snow.len(), 1);
        assert_eq!(capture.science_f16_payload.ext_precip.len(), 1);

        let capped_snow = [0.007f32];
        let (capped_quant, capped_codes) = encode_log_channel(&capped_snow);
        let mut capped = PrecisionAuditCapture::new(
            PrecisionAuditConfig::new(std::env::temp_dir()),
            1,
            1,
            1,
            0.0,
            250.0,
        );
        capped.record_precipitation(
            &capped_snow,
            capped_quant,
            &capped_codes,
            &total,
            total_quant,
            &total_codes,
        );
        let expected_capped = ir_absorption_from_ext(HydrometeorClass::Snow, 0.005);
        assert!((capped.native_cloud_ir_beta[0] as f64 - expected_capped).abs() < 1.0e-9);
    }
}
