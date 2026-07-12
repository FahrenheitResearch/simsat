//! Water-vapor bands (ABI 8 / 9 / 10 = 6.2 / 6.9 / 7.3 um; design section 7 + the WV
//! addendum, owner decision 6).
//!
//! The water-vapor product is a GENERALIZATION of the M6 synthetic-IR window pass
//! (`ir.rs`): the SAME top-down slant-ray gray-body Planck-emission march, but the
//! dominant absorber/emitter is WATER VAPOR (from the brick QVAPOR channel) rather
//! than the surface. Owner decision 6 predicted exactly this ("QVAPOR gets a full
//! brick channel now so a 6.2 um water-vapor IR band is a shader-only addition
//! later") — so this module is small: it does NOT re-implement the march. It only
//! names the three bands and builds the per-band [`IrConfig`] (wavelength + the
//! strong per-band WV mass-absorption from `optics`); `ir::render_ir_bt_frame` /
//! `topdown::render_topdown_ir_bt_frame` then render the WV brightness-temperature
//! plane band-agnostically, and `ir_enhance` colours it (CIMSS = the classic WV
//! moisture palette; the WV-scaled grayscale is `Grayscale`).
//!
//! The WEIGHTING FUNCTION (which troposphere layer a band sees) is set by the per-band
//! WV mass-absorption coefficient in `optics.rs` (documented there): 6.2 um strongly
//! absorbing -> UPPER troposphere (cold BT, upper-level moisture); 7.3 um weakly
//! absorbing -> LOWER/mid troposphere (warmer BT); 6.9 um between. Clouds stay opaque
//! in the WV bands too (the cloud IR absorption is unchanged), so cloud tops read cold.

use crate::ir::IrConfig;
use crate::ir_enhance::IrEnhancement;
use crate::optics::{
    WV_BAND8_WAVELENGTH_M, WV_BAND9_WAVELENGTH_M, WV_BAND10_WAVELENGTH_M, WV_MASS_ABS_BAND8_M2_KG,
    WV_MASS_ABS_BAND9_M2_KG, WV_MASS_ABS_BAND10_M2_KG,
};

/// One of the three ABI water-vapor bands. `Upper`/`Mid`/`Low` name the troposphere
/// layer each band's weighting function samples (upper / mid / lower-level moisture).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WvBand {
    /// Band 8, 6.2 um — upper-level water vapor (strongest absorption, coldest BT).
    #[default]
    Upper,
    /// Band 9, 6.9 um — mid-level water vapor.
    Mid,
    /// Band 10, 7.3 um — lower/mid-level water vapor (weakest absorption, warmest BT).
    Low,
}

impl WvBand {
    /// All three bands in UI / picker order (upper -> lower).
    pub const ALL: [WvBand; 3] = [WvBand::Upper, WvBand::Mid, WvBand::Low];

    /// The ABI band number (8 / 9 / 10) — the store selector + enhancement key.
    pub fn abi_band(self) -> u8 {
        match self {
            WvBand::Upper => 8,
            WvBand::Mid => 9,
            WvBand::Low => 10,
        }
    }

    /// Band centre wavelength (m) for the Planck / inverse-Planck.
    pub fn wavelength_m(self) -> f64 {
        match self {
            WvBand::Upper => WV_BAND8_WAVELENGTH_M,
            WvBand::Mid => WV_BAND9_WAVELENGTH_M,
            WvBand::Low => WV_BAND10_WAVELENGTH_M,
        }
    }

    /// The band-averaged gray WV mass-absorption coefficient (m^2 kg^-1) that tunes the
    /// weighting-function altitude (design section 7 / `optics.rs`).
    pub fn mass_abs_m2_kg(self) -> f64 {
        match self {
            WvBand::Upper => WV_MASS_ABS_BAND8_M2_KG,
            WvBand::Mid => WV_MASS_ABS_BAND9_M2_KG,
            WvBand::Low => WV_MASS_ABS_BAND10_M2_KG,
        }
    }

    /// The centre wavelength as a short micron string (`"6.2"` / `"6.9"` / `"7.3"`).
    pub fn micron(self) -> &'static str {
        match self {
            WvBand::Upper => "6.2",
            WvBand::Mid => "6.9",
            WvBand::Low => "7.3",
        }
    }

    /// Stable slug for CLI / settings (the micron string, e.g. `"6.2"`).
    pub fn slug(self) -> &'static str {
        self.micron()
    }

    /// Human-readable picker label, e.g. `"WV 6.2 um (upper)"`.
    pub fn label(self) -> &'static str {
        match self {
            WvBand::Upper => "WV 6.2 um (upper)",
            WvBand::Mid => "WV 6.9 um (mid)",
            WvBand::Low => "WV 7.3 um (lower)",
        }
    }

    /// Parse a band from a flexible token: the micron string (`6.2`), the compact form
    /// (`62`/`wv62`), the level name (`upper`/`mid`/`low`), or the band number (`8`).
    pub fn parse(value: &str) -> Option<Self> {
        let n = value
            .trim()
            .to_ascii_lowercase()
            .replace(['-', '_', ' '], "");
        match n.as_str() {
            "6.2" | "62" | "wv62" | "upper" | "u" | "band8" | "c08" | "c8" | "8" => {
                Some(WvBand::Upper)
            }
            "6.9" | "69" | "wv69" | "mid" | "m" | "band9" | "c09" | "c9" | "9" => Some(WvBand::Mid),
            "7.3" | "73" | "wv73" | "low" | "lower" | "l" | "band10" | "c10" | "10" => {
                Some(WvBand::Low)
            }
            _ => None,
        }
    }

    /// The enhancement a WV frame renders through by default: `Cimss`, which for the WV
    /// bands routes to the pinned `rw_sat` WV production palette
    /// (`band_anchors(8/9/10)` = the classic white-cold / blue / purple / brown moisture
    /// enhancement, scaled to the WV BT range). See `ir_enhance`.
    pub fn default_enhancement(self) -> IrEnhancement {
        IrEnhancement::Cimss
    }

    /// Build the [`IrConfig`] for this WV band: the band-13 window defaults (step sizes,
    /// surface emissivity, transmittance floor) with the WV wavelength + the strong
    /// per-band WV mass-absorption substituted, so the shared `ir::march_ir` marches the
    /// water-vapor weighting function instead of the surface. This is the whole "WV is a
    /// shader-only addition" — a config, not a new march.
    pub fn ir_config(self) -> IrConfig {
        IrConfig {
            band: self.abi_band(),
            wavelength_m: self.wavelength_m(),
            sensor: crate::thermal_sensor::ThermalSensor::FastGray,
            wv_mass_abs_m2_kg: self.mass_abs_m2_kg(),
            wv_continuum: true,
            ..IrConfig::band13()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wv_band_metadata_is_consistent_and_slugs_round_trip() {
        // Band numbers 8/9/10, wavelengths ordered 6.2 < 6.9 < 7.3, absorption ordered
        // 6.2 > 6.9 > 7.3 (so 6.2 sees highest/coldest), and every slug round-trips.
        assert_eq!(
            WvBand::ALL.map(|b| b.abi_band()),
            [8, 9, 10],
            "ABI band numbers"
        );
        assert!(WvBand::Upper.wavelength_m() < WvBand::Mid.wavelength_m());
        assert!(WvBand::Mid.wavelength_m() < WvBand::Low.wavelength_m());
        assert!(WvBand::Upper.mass_abs_m2_kg() > WvBand::Mid.mass_abs_m2_kg());
        assert!(WvBand::Mid.mass_abs_m2_kg() > WvBand::Low.mass_abs_m2_kg());
        for b in WvBand::ALL {
            assert_eq!(WvBand::parse(b.slug()), Some(b), "slug {} lost", b.slug());
        }
        // The flexible parser accepts the compact / level / band-number forms too.
        assert_eq!(WvBand::parse("wv62"), Some(WvBand::Upper));
        assert_eq!(WvBand::parse("upper"), Some(WvBand::Upper));
        assert_eq!(WvBand::parse("8"), Some(WvBand::Upper));
        assert_eq!(WvBand::parse("7.3"), Some(WvBand::Low));
        assert_eq!(WvBand::parse("nonsense"), None);
        assert_eq!(WvBand::default(), WvBand::Upper);
    }

    #[test]
    fn ir_config_carries_the_band_wavelength_and_strong_wv_coefficient() {
        // The WV config inherits the band-13 window step sizes but substitutes the band
        // wavelength + a WV mass-absorption FAR stronger than the band-13 window continuum
        // (that strength is what moves the weighting function off the surface).
        let base = IrConfig::band13();
        for b in WvBand::ALL {
            let cfg = b.ir_config();
            assert_eq!(cfg.band, b.abi_band());
            assert_eq!(cfg.wavelength_m, b.wavelength_m());
            assert_eq!(cfg.wv_mass_abs_m2_kg, b.mass_abs_m2_kg());
            assert!(cfg.wv_continuum);
            // Window defaults inherited unchanged.
            assert_eq!(cfg.max_steps, base.max_steps);
            assert_eq!(cfg.surface_emissivity, base.surface_emissivity);
            assert_eq!(cfg.transmittance_floor, base.transmittance_floor);
            // WV absorbs vastly more per unit vapor than the band-13 continuum.
            assert!(cfg.wv_mass_abs_m2_kg > 10.0 * base.wv_mass_abs_m2_kg);
        }
        // The default enhancement is CIMSS (= the WV moisture palette for bands 8-10).
        assert_eq!(WvBand::Upper.default_enhancement(), IrEnhancement::Cimss);
    }
}
