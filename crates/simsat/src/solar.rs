//! Solar position (design doc section 1 / section 6, M1 slice).
//!
//! Per-pixel sun elevation/azimuth from the timestep's UTC time, via the NOAA
//! Solar Calculator algorithm (Meeus, *Astronomical Algorithms*, ch. 25 + the
//! NOAA equation-of-time formulation). This is the widely-used low-precision
//! ephemeris: accuracy is ~0.01 deg in declination and a fraction of a degree in
//! horizontal coordinates over the modern era — well inside the ~0.5 deg the M1
//! shading needs. It does NOT model topocentric parallax (negligible for the sun,
//! ~0.0024 deg) and uses the standard atmospheric-refraction correction NOAA
//! applies. Higher precision (NREL SPA) is not needed for band-averaged shading.
//!
//! Frame: azimuth is degrees clockwise from true north; elevation is degrees
//! above the horizon (refraction-corrected). The sun direction is returned in a
//! local ENU basis `(east, north, up)` for the N-dot-L surface term.

use std::f64::consts::PI;

use crate::atmosphere::sun_enu_to_ecef;

const DEG2RAD: f64 = PI / 180.0;
const RAD2DEG: f64 = 180.0 / PI;

/// Sun position in the local horizontal frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarPosition {
    /// Elevation above the horizon (deg), refraction-corrected. Negative = below.
    pub elevation_deg: f64,
    /// Azimuth clockwise from true north (deg), in `[0, 360)`.
    pub azimuth_deg: f64,
}

impl SolarPosition {
    /// Unit sun direction in the local ENU basis `(east, north, up)`.
    pub fn enu_direction(&self) -> [f64; 3] {
        let el = self.elevation_deg * DEG2RAD;
        let az = self.azimuth_deg * DEG2RAD;
        let cos_el = el.cos();
        [cos_el * az.sin(), cos_el * az.cos(), el.sin()]
    }
}

/// Julian Day (including fractional UT) for a Gregorian calendar date.
pub fn julian_day(year: i32, month: u32, day: u32, ut_hours: f64) -> f64 {
    let (mut y, mut m) = (year, month as i32);
    if m <= 2 {
        y -= 1;
        m += 12;
    }
    let a = (y as f64 / 100.0).floor();
    let b = 2.0 - a + (a / 4.0).floor();
    (365.25 * (y as f64 + 4716.0)).floor() + (30.6001 * (m as f64 + 1.0)).floor() + day as f64 + b
        - 1524.5
        + ut_hours / 24.0
}

/// Julian centuries since J2000.0 from a Julian Day.
fn julian_century(jd: f64) -> f64 {
    (jd - 2_451_545.0) / 36525.0
}

/// Internal: the shared Meeus intermediates for a Julian century.
struct SunIntermediates {
    /// Geometric mean longitude (deg, `[0,360)`).
    mean_long: f64,
    /// Geometric mean anomaly (deg).
    mean_anom: f64,
    /// Orbit eccentricity.
    eccentricity: f64,
    /// Apparent ecliptic longitude (deg).
    apparent_long: f64,
    /// Corrected obliquity of the ecliptic (deg).
    obliquity: f64,
}

fn sun_intermediates(t: f64) -> SunIntermediates {
    let mean_long = (280.46646 + t * (36000.76983 + t * 0.0003032)).rem_euclid(360.0);
    let mean_anom = 357.52911 + t * (35999.05029 - 0.0001537 * t);
    let eccentricity = 0.016708634 - t * (0.000042037 + 0.0000001267 * t);
    let m = mean_anom * DEG2RAD;
    let center = m.sin() * (1.914602 - t * (0.004817 + 0.000014 * t))
        + (2.0 * m).sin() * (0.019993 - 0.000101 * t)
        + (3.0 * m).sin() * 0.000289;
    let true_long = mean_long + center;
    let omega = 125.04 - 1934.136 * t;
    let apparent_long = true_long - 0.00569 - 0.00478 * (omega * DEG2RAD).sin();
    let obliquity0 =
        23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0;
    let obliquity = obliquity0 + 0.00256 * (omega * DEG2RAD).cos();
    SunIntermediates {
        mean_long,
        mean_anom,
        eccentricity,
        apparent_long,
        obliquity,
    }
}

/// Solar declination (deg) for a Julian Day.
pub fn solar_declination_deg(jd: f64) -> f64 {
    let s = sun_intermediates(julian_century(jd));
    ((s.obliquity * DEG2RAD).sin() * (s.apparent_long * DEG2RAD).sin()).asin() * RAD2DEG
}

/// Equation of time (minutes) for a Julian Day: apparent minus mean solar time.
pub fn equation_of_time_min(jd: f64) -> f64 {
    let s = sun_intermediates(julian_century(jd));
    let eps = s.obliquity * DEG2RAD;
    let y = (eps / 2.0).tan().powi(2);
    let l0 = s.mean_long * DEG2RAD;
    let m = s.mean_anom * DEG2RAD;
    let e = s.eccentricity;
    let eq = y * (2.0 * l0).sin() - 2.0 * e * m.sin() + 4.0 * e * y * m.sin() * (2.0 * l0).cos()
        - 0.5 * y * y * (4.0 * l0).sin()
        - 1.25 * e * e * (2.0 * m).sin();
    4.0 * eq * RAD2DEG
}

/// NOAA atmospheric-refraction correction (deg) for a geometric elevation (deg).
fn refraction_deg(elevation_deg: f64) -> f64 {
    if elevation_deg > 85.0 {
        return 0.0;
    }
    let e = elevation_deg;
    let te = (e * DEG2RAD).tan();
    let arcsec = if e > 5.0 {
        58.1 / te - 0.07 / te.powi(3) + 0.000086 / te.powi(5)
    } else if e > -0.575 {
        1735.0 + e * (-518.2 + e * (103.4 + e * (-12.79 + e * 0.711)))
    } else {
        -20.772 / te
    };
    arcsec / 3600.0
}

/// A single UTC instant's solar geometry, with the date-only terms (declination,
/// equation of time) precomputed once so per-pixel evaluation over a whole raster
/// only does the location-dependent hour-angle/zenith/azimuth work.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarFrame {
    ut_hours: f64,
    decl_rad: f64,
    eqtime_min: f64,
}

impl SolarFrame {
    /// Build the frame for a UTC instant.
    pub fn new(year: i32, month: u32, day: u32, ut_hours: f64) -> Self {
        let jd = julian_day(year, month, day, ut_hours);
        Self {
            ut_hours,
            decl_rad: solar_declination_deg(jd) * DEG2RAD,
            eqtime_min: equation_of_time_min(jd),
        }
    }

    /// Sun elevation/azimuth at a location (deg East longitude).
    pub fn at(&self, lat_deg: f64, lon_deg: f64) -> SolarPosition {
        let tst = (self.ut_hours * 60.0 + self.eqtime_min + 4.0 * lon_deg).rem_euclid(1440.0);
        let mut hour_angle = tst / 4.0 - 180.0;
        if hour_angle < -180.0 {
            hour_angle += 360.0;
        }
        let ha = hour_angle * DEG2RAD;
        let lat = lat_deg * DEG2RAD;
        let decl = self.decl_rad;

        let cos_zenith =
            (lat.sin() * decl.sin() + lat.cos() * decl.cos() * ha.cos()).clamp(-1.0, 1.0);
        let zenith = cos_zenith.acos();
        let elevation_geom = 90.0 - zenith * RAD2DEG;
        let elevation = elevation_geom + refraction_deg(elevation_geom);

        let sin_zenith = zenith.sin();
        let azimuth = if sin_zenith.abs() < 1.0e-9 {
            180.0
        } else {
            let cos_az =
                ((lat.sin() * cos_zenith - decl.sin()) / (lat.cos() * sin_zenith)).clamp(-1.0, 1.0);
            let az_acos = cos_az.acos() * RAD2DEG;
            if hour_angle > 0.0 {
                (az_acos + 180.0).rem_euclid(360.0)
            } else {
                (540.0 - az_acos).rem_euclid(360.0)
            }
        };
        SolarPosition {
            elevation_deg: elevation,
            azimuth_deg: azimuth,
        }
    }
}

/// Sun elevation/azimuth at a location and UTC instant (NOAA algorithm).
///
/// `ut_hours` is the UTC time of day in hours (e.g. `2.25` for 02:15). Longitude
/// is degrees East (WRF/XLONG convention). Convenience wrapper over [`SolarFrame`].
pub fn solar_position(
    year: i32,
    month: u32,
    day: u32,
    ut_hours: f64,
    lat_deg: f64,
    lon_deg: f64,
) -> SolarPosition {
    SolarFrame::new(year, month, day, ut_hours).at(lat_deg, lon_deg)
}

/// Parse an ISO-8601 UTC timestamp like `2025-06-21T02:15:00Z` into
/// `(year, month, day, ut_hours)`. Tolerant of a trailing `Z` and of the
/// `YYYY-MM-DD_HH:MM:SS` variant. Returns `None` on a malformed string.
pub fn parse_iso_utc(s: &str) -> Option<(i32, u32, u32, f64)> {
    let s = s.trim().trim_end_matches('Z');
    let (date, time) = s.split_once(['T', '_'])?;
    let mut dp = date.split('-');
    let year: i32 = dp.next()?.parse().ok()?;
    let month: u32 = dp.next()?.parse().ok()?;
    let day: u32 = dp.next()?.parse().ok()?;
    let mut tp = time.split(':');
    let hh: f64 = tp.next()?.parse().ok()?;
    let mm: f64 = tp.next().and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let ss: f64 = tp.next().and_then(|v| v.parse().ok()).unwrap_or(0.0);
    Some((year, month, day, hh + mm / 60.0 + ss / 3600.0))
}

/// Project a global ECEF sun direction into the local ENU basis at `(lat, lon)`.
///
/// This is the inverse of [`sun_enu_to_ecef`]. The renderer uses the returned
/// elevation for a synthetic sun override after placing that override over the
/// raster bounding-box centre.
pub fn sun_enu_and_elevation(sun_ecef: [f64; 3], lat_deg: f64, lon_deg: f64) -> ([f64; 3], f64) {
    let (la, lo) = (lat_deg.to_radians(), lon_deg.to_radians());
    let (sla, cla) = la.sin_cos();
    let (slo, clo) = lo.sin_cos();
    let east = [-slo, clo, 0.0];
    let north = [-sla * clo, -sla * slo, cla];
    let up = [cla * clo, cla * slo, sla];
    let dot = |a: [f64; 3]| a[0] * sun_ecef[0] + a[1] * sun_ecef[1] + a[2] * sun_ecef[2];
    let (e, n, u) = (dot(east), dot(north), dot(up));
    let elev = u.clamp(-1.0, 1.0).asin().to_degrees();
    ([e, n, u], elev)
}

/// Resolve the renderer's single sun-at-infinity ECEF vector at a raster centre.
///
/// An unset override component keeps the NOAA value at the centre, matching
/// [`crate::api::SunOverride`]. The returned elevation is projected back from the
/// ECEF vector rather than copied from the input so this helper and every per-pixel
/// diagnostic follow the same ENU -> ECEF -> ENU path.
pub fn frame_sun_ecef(
    solar: &SolarFrame,
    center_lat_deg: f64,
    center_lon_deg: f64,
    elevation_override_deg: Option<f64>,
    azimuth_override_deg: Option<f64>,
) -> ([f64; 3], f64) {
    let real_sun = solar.at(center_lat_deg, center_lon_deg);
    let use_override = elevation_override_deg.is_some() || azimuth_override_deg.is_some();
    let sun_ecef = if use_override {
        let elev_deg = elevation_override_deg.unwrap_or(real_sun.elevation_deg);
        let az_deg = azimuth_override_deg.unwrap_or(real_sun.azimuth_deg);
        let (elev, az) = (elev_deg.to_radians(), az_deg.to_radians());
        let sun_enu = [elev.cos() * az.sin(), elev.cos() * az.cos(), elev.sin()];
        sun_enu_to_ecef(sun_enu, center_lat_deg, center_lon_deg)
    } else {
        sun_enu_to_ecef(real_sun.enu_direction(), center_lat_deg, center_lon_deg)
    };
    let center_elev = sun_enu_and_elevation(sun_ecef, center_lat_deg, center_lon_deg).1;
    (sun_ecef, center_elev)
}

/// Compute the finite-pair latitude/longitude bounding box used by a surface raster.
///
/// Coordinates are kept as `f32` through the extrema and centre calculation because
/// that is the renderer's raster representation. A pixel is valid only when both its
/// latitude and longitude are finite.
pub fn lat_lon_bbox(lat: &[f32], lon: &[f32]) -> Option<(f32, f32, f32, f32)> {
    if lat.len() != lon.len() {
        return None;
    }
    let mut pairs = lat
        .iter()
        .zip(lon.iter())
        .filter(|(la, lo)| la.is_finite() && lo.is_finite());
    let (&la0, &lo0) = pairs.next()?;
    let (mut la_min, mut la_max, mut lo_min, mut lo_max) = (la0, la0, lo0, lo0);
    for (&la, &lo) in pairs {
        la_min = la_min.min(la);
        la_max = la_max.max(la);
        lo_min = lo_min.min(lo);
        lo_max = lo_max.max(lo);
    }
    Some((la_min, la_max, lo_min, lo_max))
}

/// Exact per-pixel elevation diagnostic for the renderer's surface-light LUT.
///
/// Without an override, [`crate::gpu::build_luts`] evaluates NOAA independently at
/// every valid pixel. With either override component set, the renderer instead places
/// that partially-overridden sun over the finite-pair bounding-box centre, converts it
/// once to ECEF, then projects that vector into every pixel's local ENU frame. This
/// diagnostic mirrors that conditional behavior exactly. The frame-wide atmosphere and
/// cloud sun remains a separate single ECEF vector. Invalid pairs produce `NaN`.
pub fn solar_elevation_grid(
    solar: &SolarFrame,
    lat: &[f32],
    lon: &[f32],
    elevation_override_deg: Option<f64>,
    azimuth_override_deg: Option<f64>,
) -> Result<Vec<f32>, String> {
    if lat.len() != lon.len() {
        return Err(format!(
            "latitude and longitude lengths differ ({} != {})",
            lat.len(),
            lon.len()
        ));
    }
    if lat.is_empty() {
        return Err("latitude and longitude grids must not be empty".to_string());
    }
    for (name, value) in [
        ("sun_elev", elevation_override_deg),
        ("sun_az", azimuth_override_deg),
    ] {
        if value.is_some_and(|v| !v.is_finite()) {
            return Err(format!("{name} must be finite when provided"));
        }
    }
    let bbox = lat_lon_bbox(lat, lon)
        .ok_or_else(|| "latitude/longitude grids contain no finite coordinate pair".to_string())?;
    let use_override = elevation_override_deg.is_some() || azimuth_override_deg.is_some();
    if !use_override {
        return Ok(lat
            .iter()
            .zip(lon.iter())
            .map(|(&la, &lo)| {
                if la.is_finite() && lo.is_finite() {
                    solar.at(la as f64, lo as f64).elevation_deg as f32
                } else {
                    f32::NAN
                }
            })
            .collect());
    }

    let (la_min, la_max, lo_min, lo_max) = bbox;
    // Preserve the renderer's f32 bbox-centre arithmetic before widening to f64.
    let center_lat = ((la_min + la_max) * 0.5) as f64;
    let center_lon = ((lo_min + lo_max) * 0.5) as f64;
    let (sun_ecef, _) = frame_sun_ecef(
        solar,
        center_lat,
        center_lon,
        elevation_override_deg,
        azimuth_override_deg,
    );
    Ok(lat
        .iter()
        .zip(lon.iter())
        .map(|(&la, &lo)| {
            if la.is_finite() && lo.is_finite() {
                sun_enu_and_elevation(sun_ecef, la as f64, lo as f64).1 as f32
            } else {
                f32::NAN
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// NREL SPA published reference (Reda & Andreas 2004, the canonical Solar
    /// Position Algorithm test case): 2003-10-17 19:30:30 UTC, lat 39.742476,
    /// lon -105.1786 -> topocentric zenith 50.11162 deg (elevation 39.88838),
    /// azimuth 194.34024 deg. The NOAA algorithm agrees to well within 0.5 deg.
    #[test]
    fn matches_nrel_spa_reference() {
        let ut = 19.0 + 30.0 / 60.0 + 30.0 / 3600.0;
        let pos = solar_position(2003, 10, 17, ut, 39.742476, -105.1786);
        assert!(
            (pos.elevation_deg - 39.88838).abs() < 0.5,
            "elevation {} vs 39.888",
            pos.elevation_deg
        );
        assert!(
            (pos.azimuth_deg - 194.34024).abs() < 0.5,
            "azimuth {} vs 194.340",
            pos.azimuth_deg
        );
    }

    #[test]
    fn declination_tracks_the_solstices() {
        // Summer solstice ~2020-06-20, declination ~ +23.44 deg.
        let dec_jun = solar_declination_deg(julian_day(2020, 6, 21, 0.0));
        assert!((dec_jun - 23.44).abs() < 0.3, "june dec {dec_jun}");
        // Winter solstice, declination ~ -23.44 deg.
        let dec_dec = solar_declination_deg(julian_day(2020, 12, 21, 12.0));
        assert!((dec_dec + 23.44).abs() < 0.3, "dec dec {dec_dec}");
        // Equinox, declination ~ 0.
        let dec_mar = solar_declination_deg(julian_day(2020, 3, 20, 12.0));
        assert!(dec_mar.abs() < 0.6, "march dec {dec_mar}");
    }

    #[test]
    fn equation_of_time_has_the_right_seasonal_sign() {
        // Early November: apparent ahead of mean, eqtime ~ +16 min.
        let nov = equation_of_time_min(julian_day(2020, 11, 3, 12.0));
        assert!((nov - 16.4).abs() < 2.0, "nov eqtime {nov}");
        // Mid February: eqtime ~ -14 min.
        let feb = equation_of_time_min(julian_day(2020, 2, 12, 12.0));
        assert!((feb + 14.0).abs() < 2.0, "feb eqtime {feb}");
    }

    #[test]
    fn at_solar_noon_sun_is_due_south_at_expected_elevation() {
        // Physical truth: at solar noon in the northern hemisphere the sun is due
        // south (az=180) at elevation 90 - |lat - dec|. Drive UT to solar noon on
        // the prime meridian: tst = 720 = ut*60 + eqtime -> ut = (720 - eqtime)/60.
        for (y, mo, d, lat) in [(2025, 6, 21, 47.0), (2025, 3, 20, 20.0)] {
            // Iterate once to converge eqtime at the noon instant (it barely moves).
            let mut ut = 12.0;
            for _ in 0..3 {
                let eq = equation_of_time_min(julian_day(y, mo, d, ut));
                ut = (720.0 - eq) / 60.0;
            }
            let dec = solar_declination_deg(julian_day(y, mo, d, ut));
            let pos = solar_position(y, mo, d, ut, lat, 0.0);
            let expect_el = 90.0 - (lat - dec).abs();
            assert!(
                (pos.elevation_deg - expect_el).abs() < 0.5,
                "el {} vs {expect_el}",
                pos.elevation_deg
            );
            assert!(
                (pos.azimuth_deg - 180.0).abs() < 0.5,
                "az {} vs 180",
                pos.azimuth_deg
            );
        }
    }

    #[test]
    fn night_side_is_below_the_horizon() {
        // Enderlin at 05:15 UTC (~00:15 local) is deep night.
        let pos = solar_position(2025, 6, 21, 5.25, 47.0, -97.0);
        assert!(
            pos.elevation_deg < 0.0,
            "expected night, got {}",
            pos.elevation_deg
        );
    }

    #[test]
    fn enu_direction_is_unit_and_points_up_when_high() {
        let pos = SolarPosition {
            elevation_deg: 90.0,
            azimuth_deg: 0.0,
        };
        let d = pos.enu_direction();
        assert!((d[2] - 1.0).abs() < 1e-9);
        let norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        // Due-east sunrise: az 90, el 0 -> +east.
        let e = SolarPosition {
            elevation_deg: 0.0,
            azimuth_deg: 90.0,
        }
        .enu_direction();
        assert!((e[0] - 1.0).abs() < 1e-9 && e[1].abs() < 1e-9);
    }

    #[test]
    fn parse_iso_utc_handles_both_separators() {
        assert_eq!(
            parse_iso_utc("2025-06-21T02:15:00Z"),
            Some((2025, 6, 21, 2.25))
        );
        assert_eq!(
            parse_iso_utc("2018-10-10_12:00:00"),
            Some((2018, 10, 10, 12.0))
        );
        assert_eq!(parse_iso_utc("garbage"), None);
    }

    #[test]
    fn elevation_grid_matches_surface_light_lut_noaa_at_center_and_corners() {
        let solar = SolarFrame::new(1974, 4, 3, 23.2);
        let lat = [35.0, 35.0, 35.0, 37.5, 37.5, 37.5, 40.0, 40.0, 40.0];
        let lon = [
            -95.0, -92.5, -90.0, -95.0, -92.5, -90.0, -95.0, -92.5, -90.0,
        ];
        let got = solar_elevation_grid(&solar, &lat, &lon, None, None).unwrap();

        for idx in 0..lat.len() {
            let expected = solar.at(lat[idx] as f64, lon[idx] as f64).elevation_deg as f32;
            assert_eq!(got[idx], expected, "pixel {idx}");
        }
    }

    #[test]
    fn elevation_grid_honors_full_and_partial_overrides() {
        let solar = SolarFrame::new(1974, 4, 3, 23.2);
        let lat = [36.0, 36.0, 37.5, 37.5, 39.0, 39.0];
        let lon = [-94.0, -91.0, -94.0, -91.0, -94.0, -91.0];

        let full = solar_elevation_grid(&solar, &lat, &lon, Some(12.5), Some(210.0)).unwrap();
        let elev = 12.5_f64.to_radians();
        let az = 210.0_f64.to_radians();
        let sun_ecef = sun_enu_to_ecef(
            [elev.cos() * az.sin(), elev.cos() * az.cos(), elev.sin()],
            37.5,
            -92.5,
        );
        for idx in 0..lat.len() {
            let expected =
                sun_enu_and_elevation(sun_ecef, lat[idx] as f64, lon[idx] as f64).1 as f32;
            assert_eq!(full[idx], expected, "override pixel {idx}");
        }
        // Add the bbox centre explicitly: an override is defined at that centre and
        // must round-trip to its requested elevation through ECEF.
        let center = solar_elevation_grid(
            &solar,
            &[36.0, 37.5, 39.0],
            &[-94.0, -92.5, -91.0],
            Some(12.5),
            Some(210.0),
        )
        .unwrap();
        assert!((center[1] - 12.5).abs() <= 1.0e-5);
        assert!(full.iter().all(|v| v.is_finite()));

        let partial = solar_elevation_grid(
            &solar,
            &[36.0, 37.5, 39.0],
            &[-94.0, -92.5, -91.0],
            Some(5.0),
            None,
        )
        .unwrap();
        assert!((partial[1] - 5.0).abs() <= 1.0e-5);
    }

    #[test]
    fn elevation_grid_validates_lengths_and_finite_coordinates() {
        let solar = SolarFrame::new(2025, 1, 15, 12.0);
        assert!(
            solar_elevation_grid(&solar, &[35.0], &[90.0, 91.0], None, None)
                .unwrap_err()
                .contains("lengths differ")
        );
        assert!(
            solar_elevation_grid(&solar, &[], &[], None, None)
                .unwrap_err()
                .contains("must not be empty")
        );
        assert!(
            solar_elevation_grid(&solar, &[f32::NAN], &[f32::NAN], None, None)
                .unwrap_err()
                .contains("no finite coordinate pair")
        );
        assert!(
            solar_elevation_grid(&solar, &[35.0], &[90.0], Some(f64::NAN), None)
                .unwrap_err()
                .contains("sun_elev")
        );

        let with_space =
            solar_elevation_grid(&solar, &[35.0, f32::NAN], &[90.0, 91.0], None, None).unwrap();
        assert!(with_space[0].is_finite());
        assert!(with_space[1].is_nan());
    }
}
