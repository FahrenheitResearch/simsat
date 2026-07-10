//! IR brightness-temperature enhancements (design doc section 7, M6).
//!
//! Maps a true-Kelvin BT plane to a coloured IR image — the classic
//! cold-cloud-tops-in-colour look the real GOES/Himawari products use. This lives
//! in the ENGINE (not the studio) because SimSat Studio is standalone and cannot
//! depend on BowEcho's binary, where these enhancements live; the store writer
//! (`store_out::write_ir_frame`) also writes true Kelvin so BowEcho re-enhances the
//! same frame live with the same curves.
//!
//! ---
//! ATTRIBUTION. The `IrEnhancement` enum and every anchor table below are PORTED
//! VERBATIM (values unchanged; comments condensed) from BowEcho
//! (`crates/app_ui/src/sat_worker.rs` in the sibling rusty-weather app, the
//! rev pinned in Cargo.toml): `IrEnhancement`, `ir_enhancement_anchors`,
//! `enhanced_anchors_for_band`, and the `ENHANCED_IR` / `BD_CURVE` / `AVN_IR` /
//! `FUNKTOP_IR` / `RAINBOW_IR` / `GRAYSCALE_IR` anchor constants. The colour
//! interpolation itself is the pinned `rw_sat::palette::anchor_color` (called
//! directly — `rw-sat` is a workspace dependency), and the CIMSS/`Cimss` per-band
//! fallback uses the pinned `rw_sat::palette::band_anchors`. The anchor SCIENCE
//! (NESDIS BD / NOAA-SSD AVN / Funktop / RB rainbow breakpoints; the CIMSS ramp) is
//! documented in the BowEcho source; the citations are retained in the constants.
//!
//! `Anchors = &'static [(f32 /* Kelvin */, [u8; 3] /* rgb */)]`, applied by
//! `anchor_color` (clamp to the ends, linear between, NaN -> transparent).
//!
//! ---
//! WATER-VAPOR bands (ABI 8/9/10 = 6.2/6.9/7.3 um). The classic WV COLOUR table is the
//! pinned `rw_sat::palette` WV production palette — `band_anchors(8/9/10)` =
//! `UPPER/MID/LOWER_WATER_VAPOR` (white cold/moist -> blue -> purple -> brown warm/dry,
//! ranges 184-268 / 188-276 / 196-286 K), the same palette BowEcho itself renders these
//! bands through. It is reached via [`IrEnhancement::Cimss`] (which routes every
//! non-window band to `band_anchors(band)`), so the WV mode default IS the classic WV
//! moisture enhancement, WITH ATTRIBUTION (the pinned dep) and correctly scaled to the
//! WV BT range. [`IrEnhancement::Grayscale`] is BAND-AWARE for the WV bands: a WV-scaled
//! inverted grayscale (cold/moist WHITE -> warm/dry BLACK) over each band's WV range, so
//! the narrow ~220-270 K WV dynamic range has proper contrast (the window bands keep the
//! full 173-330 K ramp).

use rw_sat::palette::{Anchors, anchor_color, band_anchors};

/// User-selectable IR enhancement for Kelvin brightness-temperature bands (ABI /
/// AHI bands 7-16). `Cimss` keeps the per-band production behaviour ([`ENHANCED_IR`]
/// on the longwave window 13-15, production palettes elsewhere); the others are the
/// classic NOAA absolute-temperature enhancement curves applied to every IR band.
/// Ported from `sat_worker::IrEnhancement`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IrEnhancement {
    /// CIMSS-style rainbow on 13-15, production palettes elsewhere (default).
    #[default]
    Cimss,
    /// NESDIS BD curve — the stepped Dvorak tropical-cyclone enhancement.
    Bd,
    /// NOAA/SSD AVN colour IR enhancement.
    Avn,
    /// NOAA/SSD Funktop (Ted Funk precipitation) enhancement.
    Funktop,
    /// NOAA/SSD RB rainbow IR enhancement.
    Rainbow,
    /// Plain unenhanced IR grayscale (cold = white). Band-aware: the WV bands (8/9/10)
    /// use a WV-scaled range (cold/moist white -> warm/dry black); window bands use the
    /// full 173-330 K ramp.
    Grayscale,
}

impl IrEnhancement {
    /// All enhancements in UI order.
    pub const ALL: [IrEnhancement; 6] = [
        Self::Cimss,
        Self::Bd,
        Self::Avn,
        Self::Funktop,
        Self::Rainbow,
        Self::Grayscale,
    ];

    /// Stable slug (the same keys BowEcho persists, for cross-tool consistency).
    pub fn slug(self) -> &'static str {
        match self {
            Self::Cimss => "cimss",
            Self::Bd => "bd",
            Self::Avn => "avn",
            Self::Funktop => "funktop",
            Self::Rainbow => "rainbow",
            Self::Grayscale => "gray",
        }
    }

    /// Human-readable label for the picker.
    pub fn label(self) -> &'static str {
        match self {
            Self::Cimss => "CIMSS ramp (default)",
            Self::Bd => "BD curve (Dvorak)",
            Self::Avn => "AVN",
            Self::Funktop => "Funktop",
            Self::Rainbow => "Rainbow",
            Self::Grayscale => "Grayscale",
        }
    }

    /// Parse a slug LENIENTLY: unknown values keep the default enhancement. This is
    /// the total, never-failing parse (`simsat_py` wraps it with its own validity
    /// check; the pinned-default behavior is kept for that compatibility). It
    /// accepts every [`Self::parse_strict`] alias. CLI surfaces that can report an
    /// error should call [`Self::parse_strict`] instead — the WS1 QA finding was a
    /// `grayscale` typo that this lenient parse silently rendered as CIMSS.
    pub fn parse(value: &str) -> Self {
        Self::parse_strict(value).unwrap_or_default()
    }

    /// Parse a slug STRICTLY: `None` for an unknown token, so a typo surfaces as an
    /// error instead of a silently-wrong product. Case-insensitive; `-`/`_`/space
    /// insensitive. Accepts the canonical slugs plus common aliases:
    /// `grayscale` / `greyscale` / `grey` for `gray`, `bd-curve` for `bd`, and
    /// `rb` for `rainbow`.
    pub fn parse_strict(value: &str) -> Option<Self> {
        match value
            .trim()
            .to_ascii_lowercase()
            .replace(['-', '_', ' '], "")
            .as_str()
        {
            "cimss" => Some(Self::Cimss),
            "bd" | "bdcurve" => Some(Self::Bd),
            "avn" => Some(Self::Avn),
            "funktop" => Some(Self::Funktop),
            "rainbow" | "rb" => Some(Self::Rainbow),
            "gray" | "grayscale" | "grey" | "greyscale" => Some(Self::Grayscale),
            _ => None,
        }
    }
}

/// Enhanced-IR "rainbow" (CIMSS-style) over brightness temperature (Kelvin),
/// applied to the longwave window bands 13/14/15. Warm surface/low cloud reads
/// grayscale; cold tops light up green -> yellow -> orange -> red -> dark with the
/// magenta/white overshoot tips. Anchors are (Kelvin, [r,g,b]); C = K - 273.15.
/// Ported from `sat_worker::ENHANCED_IR`.
pub const ENHANCED_IR: Anchors = &[
    (173.0, [255, 255, 255]),  // -100 C: coldest overshoot
    (183.0, [232, 84, 232]),   //  -90 C: magenta
    (193.0, [188, 188, 188]),  //  -80 C: light gray
    (203.0, [70, 8, 8]),       //  -70 C: very dark red
    (213.0, [226, 26, 26]),    //  -60 C: red
    (223.0, [245, 148, 28]),   //  -50 C: orange
    (233.0, [246, 240, 42]),   //  -40 C: yellow
    (243.0, [48, 200, 66]),    //  -30 C: green (cold-cloud onset)
    (253.0, [120, 150, 120]),  //  -20 C: gray-green transition
    (263.0, [205, 205, 205]),  //  -10 C: light gray
    (273.15, [245, 245, 245]), //    0 C: white
    (293.0, [96, 96, 96]),     //  +20 C: gray
    (313.0, [46, 26, 14]),     //  +40 C: dark brown (warm surface)
    (330.0, [8, 5, 4]),        //  +57 C: near black
];

/// The longwave-window bands get the enhanced rainbow; every other band keeps its
/// production palette (`rw_sat::palette::band_anchors`). Ported from
/// `sat_worker::enhanced_anchors_for_band`.
fn enhanced_anchors_for_band(band: u8) -> Option<Anchors> {
    matches!(band, 13..=15).then_some(ENHANCED_IR)
}

/// NESDIS BD curve (Dvorak Enhanced-IR tropical-cyclone enhancement). Canonical
/// NESDIS gray-shade boundaries (Dvorak 1984 / Velden-Olander-Zehr). Steps are
/// HARD: duplicate anchor values pin each bin flat. Ported from
/// `sat_worker::BD_CURVE`.
pub const BD_CURVE: Anchors = &[
    (164.15, [88, 88, 88]),    // -109.0 C: cold dark gray floor
    (192.95, [88, 88, 88]),    //  -80.2 C: CDG (repeat gray, <= -80.2)
    (192.95, [136, 136, 136]), //  -80.2..-75.2 C: cold medium gray
    (197.95, [136, 136, 136]),
    (197.95, [255, 255, 255]), //  -75.2..-69.6 C: white
    (203.55, [255, 255, 255]),
    (203.55, [0, 0, 0]), //  -69.6..-63.2 C: black
    (209.95, [0, 0, 0]),
    (209.95, [160, 160, 160]), //  -63.2..-53.2 C: light gray
    (219.95, [160, 160, 160]),
    (219.95, [112, 112, 112]), //  -53.2..-41.2 C: medium gray
    (231.95, [112, 112, 112]),
    (231.95, [64, 64, 64]), //  -41.2..-30.2 C: dark gray
    (242.95, [64, 64, 64]),
    (242.95, [200, 200, 200]), //  -30.2 -> +9.0 C: off-white scene ramp
    (282.15, [110, 110, 110]),
    (282.15, [255, 255, 255]), //   +9.0 -> +28.0 C: warm repeat ramp
    (301.15, [0, 0, 0]),
    (330.0, [0, 0, 0]), // hottest surface stays black
];

/// NOAA/SSD AVN colour IR enhancement (aviation/tropical). Ported from
/// `sat_worker::AVN_IR`.
pub const AVN_IR: Anchors = &[
    (163.15, [255, 255, 255]), // -110.0 C
    (170.15, [255, 255, 255]), // -103.0 C: coldest overshoot white
    (170.15, [88, 88, 88]),    // -103.0..-78.0 C: anvil-core gray
    (195.15, [88, 88, 88]),
    (195.15, [240, 0, 0]), //  -78.0..-70.5 C: red
    (202.65, [240, 0, 0]),
    (202.65, [200, 118, 10]),  //  -70.5 C: step to orange...
    (218.65, [250, 183, 0]),   //  -54.5 C: ...brightening warmward
    (218.65, [250, 250, 5]),   //  -54.5 C: step to yellow...
    (234.65, [160, 158, 0]),   //  -38.5 C: ...darkening warmward
    (234.65, [0, 120, 175]),   //  -38.5 C: step to blue...
    (258.65, [0, 158, 245]),   //  -14.5 C: ...brightening warmward
    (258.65, [255, 255, 255]), // -14.5 C: step to the warm grayscale ramp
    (281.65, [130, 130, 130]), //  +8.5 C
    (305.15, [0, 0, 0]),       // +32.0 C: warm surface black
];

/// NOAA/SSD Funktop enhancement (Ted Funk, precipitation/tropical analysis). Ported
/// from `sat_worker::FUNKTOP_IR`.
pub const FUNKTOP_IR: Anchors = &[
    (163.15, [250, 250, 250]), // -110.0 C: deep-cold white
    (182.15, [235, 250, 235]), //  -91.0 C
    (195.15, [0, 255, 20]),    //  -78.0 C: bright green
    (195.15, [255, 133, 133]), //  -78.0 C: step to pink...
    (202.65, [255, 85, 85]),   //  -70.5 C: ...deepening warmward
    (202.65, [240, 0, 0]),     //  -70.5 C: step to red...
    (215.15, [75, 0, 0]),      //  -58.0 C: ...darkening warmward
    (215.15, [10, 240, 255]),  //  -58.0 C: step to cyan...
    (234.65, [5, 10, 125]),    //  -38.5 C: ...to navy warmward
    (234.65, [245, 240, 0]),   //  -38.5 C: step to yellow...
    (254.15, [100, 100, 0]),   //  -19.0 C: ...to olive warmward
    (254.15, [222, 222, 222]), //  -19.0 C: step to the warm grayscale ramp
    (305.15, [30, 30, 30]),    //  +32.0 C
    (320.15, [0, 0, 0]),       //  +47.0 C: hottest surface black
];

/// NOAA/SSD RB "rainbow" IR enhancement (magenta -> blue -> cyan -> green ->
/// yellow -> orange -> red warm-to-cold with a white cold band). Ported from
/// `sat_worker::RAINBOW_IR`.
pub const RAINBOW_IR: Anchors = &[
    (164.15, [255, 255, 255]), // -109.0 C: repeat ramp ends white
    (182.65, [10, 10, 10]),    //  -90.5 C: repeat ramp starts near black
    (182.65, [250, 250, 252]), //  -90.5..-86.5 C: white band
    (186.65, [250, 250, 252]),
    (186.65, [240, 5, 0]),   //  -86.5 C: brightest red
    (196.15, [190, 0, 2]),   //  -77.0 C
    (204.65, [122, 0, 0]),   //  -68.5 C: darkest red-brown
    (212.15, [150, 57, 0]),  //  -61.0 C
    (221.15, [185, 112, 4]), //  -52.0 C: orange-brown
    (230.15, [210, 170, 0]), //  -43.0 C
    (240.15, [252, 252, 0]), //  -33.0 C: yellow peak
    (249.15, [160, 200, 5]), //  -24.0 C
    (259.15, [12, 120, 10]), //  -14.0 C: green
    (268.15, [0, 180, 115]), //   -5.0 C
    (277.15, [0, 250, 250]), //   +4.0 C: cyan
    (289.15, [0, 105, 175]), //  +16.0 C
    (295.65, [0, 5, 120]),   //  +22.5 C: deep blue
    (305.15, [140, 0, 195]), //  +32.0 C: magenta (warm clamp)
];

/// Plain unenhanced IR grayscale: cold cloud tops white, warm surface black. Ported
/// from `sat_worker::GRAYSCALE_IR`. Used for the window bands (13-16); the WV bands use
/// a band-scaled WV grayscale (see [`wv_grayscale_for_band`]).
pub const GRAYSCALE_IR: Anchors = &[
    (173.15, [255, 255, 255]), // -100 C
    (330.0, [0, 0, 0]),        //  +57 C
];

/// WV-scaled inverted grayscale for the 6.2 um band (8): cold/moist WHITE at 184 K ->
/// warm/dry BLACK at 268 K, over band 8's WV BT range (matches `UPPER_WATER_VAPOR`).
pub const WV_GRAYSCALE_C08: Anchors = &[(184.0, [255, 255, 255]), (268.0, [0, 0, 0])];
/// WV-scaled inverted grayscale for the 6.9 um band (9): 188 K white -> 276 K black.
pub const WV_GRAYSCALE_C09: Anchors = &[(188.0, [255, 255, 255]), (276.0, [0, 0, 0])];
/// WV-scaled inverted grayscale for the 7.3 um band (10): 196 K white -> 286 K black.
pub const WV_GRAYSCALE_C10: Anchors = &[(196.0, [255, 255, 255]), (286.0, [0, 0, 0])];

/// The WV-scaled grayscale table for a WV band (8/9/10), or `None` for non-WV bands
/// (which use the full-range [`GRAYSCALE_IR`]). Cold/moist white -> warm/dry black over
/// the band's WV BT range, so the narrow ~220-270 K WV range has proper contrast.
pub fn wv_grayscale_for_band(band: u8) -> Option<Anchors> {
    match band {
        8 => Some(WV_GRAYSCALE_C08),
        9 => Some(WV_GRAYSCALE_C09),
        10 => Some(WV_GRAYSCALE_C10),
        _ => None,
    }
}

/// The anchor table an IR band renders through for the given enhancement. `Cimss`
/// preserves the per-band production behaviour; every other curve is an absolute-
/// Kelvin table shared by all IR bands. Ported from
/// `sat_worker::ir_enhancement_anchors`.
pub fn ir_enhancement_anchors(band: u8, enhancement: IrEnhancement) -> Anchors {
    match enhancement {
        IrEnhancement::Cimss => {
            enhanced_anchors_for_band(band).unwrap_or_else(|| band_anchors(band))
        }
        IrEnhancement::Bd => BD_CURVE,
        IrEnhancement::Avn => AVN_IR,
        IrEnhancement::Funktop => FUNKTOP_IR,
        IrEnhancement::Rainbow => RAINBOW_IR,
        // Band-aware: the WV bands (8/9/10) get a WV-scaled grayscale (proper contrast
        // for the narrow WV BT range); every other band keeps the full-range ramp.
        IrEnhancement::Grayscale => wv_grayscale_for_band(band).unwrap_or(GRAYSCALE_IR),
    }
}

/// Colour one brightness-temperature value (Kelvin) for a band + enhancement.
/// Returns `[r, g, b, a]`; a non-finite BT (the out-of-domain mask) is transparent.
#[inline]
pub fn bt_to_rgba(bt_k: f32, band: u8, enhancement: IrEnhancement) -> [u8; 4] {
    anchor_color(bt_k, ir_enhancement_anchors(band, enhancement))
}

/// Colour a full BT plane (Kelvin, row-major, `NaN` = out-of-domain) to row-major
/// `Rgba8` bytes (`nx*ny*4`): the coloured IR image the studio displays and the QA
/// harness writes. Off-earth / out-of-domain pixels are transparent (`a = 0`).
pub fn render_ir_rgba(bt: &[f32], band: u8, enhancement: IrEnhancement) -> Vec<u8> {
    let anchors = ir_enhancement_anchors(band, enhancement);
    let mut out = Vec::with_capacity(bt.len() * 4);
    for &v in bt {
        out.extend_from_slice(&anchor_color(v, anchors));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enhancement_slugs_round_trip_and_default_is_cimss() {
        for e in IrEnhancement::ALL {
            assert_eq!(IrEnhancement::parse(e.slug()), e, "slug {} lost", e.slug());
        }
        assert_eq!(IrEnhancement::default(), IrEnhancement::Cimss);
        // The LENIENT parse keeps its pinned total behavior (simsat_py wraps it).
        assert_eq!(IrEnhancement::parse("nonsense"), IrEnhancement::Cimss);
    }

    #[test]
    fn parse_strict_accepts_aliases_and_rejects_unknown() {
        // WS1 item: a `grayscale` typo used to silently render CIMSS. The strict
        // parse accepts the common aliases and returns None on unknown tokens.
        for e in IrEnhancement::ALL {
            assert_eq!(IrEnhancement::parse_strict(e.slug()), Some(e));
        }
        for alias in ["grayscale", "greyscale", "grey", "GRAYSCALE", "Gray Scale"] {
            assert_eq!(
                IrEnhancement::parse_strict(alias),
                Some(IrEnhancement::Grayscale),
                "alias {alias} must parse"
            );
            assert_eq!(IrEnhancement::parse(alias), IrEnhancement::Grayscale);
        }
        assert_eq!(
            IrEnhancement::parse_strict("bd-curve"),
            Some(IrEnhancement::Bd)
        );
        assert_eq!(
            IrEnhancement::parse_strict("rb"),
            Some(IrEnhancement::Rainbow)
        );
        assert_eq!(IrEnhancement::parse_strict("nonsense"), None);
        assert_eq!(IrEnhancement::parse_strict(""), None);
    }

    #[test]
    fn grayscale_is_cold_white_warm_black_and_monotone() {
        // Grayscale: cold cloud tops white (255), warm surface black (0), and the
        // luminance decreases monotonically from cold to warm (design section 7 test 5).
        // The ends clamp only at/beyond the anchor bounds (173.15 K white, 330 K black).
        let cold = bt_to_rgba(170.0, 13, IrEnhancement::Grayscale);
        let warm = bt_to_rgba(340.0, 13, IrEnhancement::Grayscale);
        assert_eq!(
            [cold[0], cold[1], cold[2]],
            [255, 255, 255],
            "cold end white"
        );
        assert_eq!([warm[0], warm[1], warm[2]], [0, 0, 0], "warm end black");
        let mut prev = 256i32;
        for &k in &[173.15, 200.0, 240.0, 273.15, 300.0, 330.0] {
            let g = bt_to_rgba(k, 13, IrEnhancement::Grayscale)[0] as i32;
            assert!(g <= prev, "grayscale not monotone at {k} K: {g} > {prev}");
            prev = g;
        }
    }

    #[test]
    fn every_enhancement_maps_cold_and_warm_to_its_curve_ends() {
        // For each enhancement, a BT at/below the coldest anchor gets the cold-end
        // colour and a BT at/above the warmest anchor gets the warm-end colour
        // (anchor_color clamps to the ends). This is the "cold BT gets the cold-end
        // colour, warm BT the warm end" check (design section 7 test 5).
        for e in [
            IrEnhancement::Bd,
            IrEnhancement::Avn,
            IrEnhancement::Funktop,
            IrEnhancement::Rainbow,
            IrEnhancement::Grayscale,
            IrEnhancement::Cimss,
        ] {
            let anchors = ir_enhancement_anchors(13, e);
            let (cold_k, cold_rgb) = anchors[0];
            let (warm_k, warm_rgb) = anchors[anchors.len() - 1];
            let cold = bt_to_rgba(cold_k - 20.0, 13, e);
            let warm = bt_to_rgba(warm_k + 20.0, 13, e);
            assert_eq!(
                [cold[0], cold[1], cold[2]],
                cold_rgb,
                "{e:?} cold end colour"
            );
            assert_eq!(
                [warm[0], warm[1], warm[2]],
                warm_rgb,
                "{e:?} warm end colour"
            );
            assert_eq!(cold[3], 255, "opaque on earth");
        }
    }

    #[test]
    fn cimss_uses_enhanced_ir_on_window_bands_and_production_elsewhere() {
        // Band 13 CIMSS is the enhanced rainbow; a non-window band (e.g. 8, WV) falls
        // back to the pinned production palette (band_anchors), not ENHANCED_IR.
        assert_eq!(
            ir_enhancement_anchors(13, IrEnhancement::Cimss),
            ENHANCED_IR
        );
        assert_eq!(
            ir_enhancement_anchors(14, IrEnhancement::Cimss),
            ENHANCED_IR
        );
        assert_ne!(ir_enhancement_anchors(8, IrEnhancement::Cimss), ENHANCED_IR);
        // BD is the same absolute-Kelvin curve regardless of band.
        assert_eq!(ir_enhancement_anchors(13, IrEnhancement::Bd), BD_CURVE);
        assert_eq!(ir_enhancement_anchors(8, IrEnhancement::Bd), BD_CURVE);
    }

    #[test]
    fn render_plane_masks_nan_and_colours_finite() {
        // A 2-pixel plane: one NaN (out-of-domain -> transparent), one cold BT
        // (opaque, coloured). Byte layout is rgba per pixel.
        let bt = vec![f32::NAN, 210.0f32];
        let rgba = render_ir_rgba(&bt, 13, IrEnhancement::Rainbow);
        assert_eq!(rgba.len(), 8);
        assert_eq!(&rgba[0..4], &[0, 0, 0, 0], "NaN -> transparent");
        assert_eq!(rgba[7], 255, "finite BT -> opaque");
        // The cold (210 K) pixel is a saturated red-ish rainbow colour, not black.
        assert!(rgba[4] as u16 + rgba[5] as u16 + rgba[6] as u16 > 0);
    }

    #[test]
    fn wv_grayscale_is_band_scaled_cold_white_warm_black() {
        // The WV bands 8/9/10 get a WV-scaled inverted grayscale (cold/moist WHITE ->
        // warm/dry BLACK) over each band's narrow WV BT range, NOT the full 173-330 K
        // window ramp — so a typical WV BT (~230 K) is a strong mid-to-light gray with
        // real contrast, not the near-white it would be on the window ramp.
        for (band, table) in [
            (8u8, WV_GRAYSCALE_C08),
            (9, WV_GRAYSCALE_C09),
            (10, WV_GRAYSCALE_C10),
        ] {
            assert_eq!(
                ir_enhancement_anchors(band, IrEnhancement::Grayscale),
                table
            );
            let (cold_k, _) = table[0];
            let (warm_k, _) = table[1];
            let cold = bt_to_rgba(cold_k - 5.0, band, IrEnhancement::Grayscale);
            let warm = bt_to_rgba(warm_k + 5.0, band, IrEnhancement::Grayscale);
            assert_eq!(
                [cold[0], cold[1], cold[2]],
                [255, 255, 255],
                "cold WV white"
            );
            assert_eq!([warm[0], warm[1], warm[2]], [0, 0, 0], "warm WV black");
            // A 230 K WV BT is clearly darker on the WV ramp than on the window ramp
            // (better use of the dynamic range).
            let wv = bt_to_rgba(230.0, band, IrEnhancement::Grayscale)[0];
            let window = anchor_color(230.0, GRAYSCALE_IR)[0];
            assert!(
                wv < window,
                "WV grayscale {wv} not darker than window {window}"
            );
        }
        // A window band (13) still uses the full-range ramp (band-13 behaviour unchanged).
        assert_eq!(
            ir_enhancement_anchors(13, IrEnhancement::Grayscale),
            GRAYSCALE_IR
        );
    }

    #[test]
    fn cimss_on_wv_bands_is_the_classic_moisture_palette() {
        // The classic WV COLOUR table = the pinned rw_sat WV production palette
        // (band_anchors 8/9/10), reached via CIMSS: cold/moist reads light, warm/dry dark.
        for band in [8u8, 9, 10] {
            assert_eq!(
                ir_enhancement_anchors(band, IrEnhancement::Cimss),
                band_anchors(band),
                "CIMSS band {band} should be the WV production palette"
            );
            // Very cold/moist tops read a light colour; warm/dry reads a dark colour.
            let cold = bt_to_rgba(185.0, band, IrEnhancement::Cimss);
            let warm = bt_to_rgba(300.0, band, IrEnhancement::Cimss);
            let lum = |p: [u8; 4]| p[0] as u16 + p[1] as u16 + p[2] as u16;
            assert!(
                lum(cold) > lum(warm),
                "WV cold {cold:?} should be lighter than warm {warm:?}"
            );
            assert_eq!(cold[3], 255, "opaque on earth");
        }
    }
}
