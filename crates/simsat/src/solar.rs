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
}
