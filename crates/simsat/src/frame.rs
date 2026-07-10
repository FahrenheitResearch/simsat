//! Analytic WRF map-projection math (design doc section 1, M0 slice).
//!
//! Forward `(lat, lon) -> fractional grid (i, j)` and inverse `(i, j) -> (lat, lon)`
//! for the four WRF map projections (Lambert conformal, polar stereographic,
//! Mercator, lat/lon), on WRF's spherical earth `R = 6_370_000 m`. Built from the
//! `MAP_PROJ`/`TRUELAT1`/`TRUELAT2`/`STAND_LON`/`CEN_LAT` global attributes exactly
//! as `local_import.rs::wrf_projection` / `wrf_process.rs::wrf_projection` read them.
//!
//! The correctness ratchet (design section 1): project every Nth stored
//! `XLAT`/`XLONG` through the analytic forward and assert it lands on its own
//! `(i, j)` within 0.05 cell. That runs in the env-gated fixture test on real
//! XLAT/XLONG; a pure-math forward/inverse round trip runs unconditionally here.
//!
//! NOT in scope for M0: the geostationary fixed-grid camera and the ECEF ray
//! transform — those land in M1 (and M1 MUST port `sat_worker::ahi_scan_angles_to_lat_lon`
//! for the Himawari sweep=y inverse, which the pinned `rw_sat` navigates with a
//! live transpose bug; see docs/bowecho-precedents.md section 8).
//!
//! ---
//! The `project`/`unproject` (lat/lon <-> projection-plane) kernels below are
//! PORTED from `crates/rustwx-render/src/projection.rs` in the pinned
//! rusty-weather repo (rev `edb9d277cce7fe1cfa1080d159053c583ca07b6a`):
//! `LambertConformal`, `MercatorProjection`, `PolarStereographic` `project`/
//! `unproject`, on `R_EARTH = 6_370_000`. Two deliberate divergences from the
//! source, appropriate to WRF-faithful grid navigation (code wins over the
//! render-tuned source):
//!   1. The source's `stabilize_reference_latitude` (which bumps any standard
//!      parallel within 1 deg of the equator up to 10 deg) is NOT applied — that
//!      is a render-conditioning nicety that would corrupt a low-latitude WRF
//!      Mercator/Lambert grid. Raw attribute values are used, with only the
//!      genuine degeneracy guards (`truelat1==truelat2`, near-zero cone, and a
//!      Mercator scale floor) retained.
//!   2. `PolarStereographic` gains an `unproject` (the source returns `None`).

use std::f64::consts::PI;

use crate::optics::EARTH_RADIUS_M as R_EARTH;

const DEG2RAD: f64 = PI / 180.0;
const RAD2DEG: f64 = 180.0 / PI;

/// Clamp a data latitude away from the poles so `tan(pi/4 +/- phi/2)` stays finite.
/// (±89.999 deg — well outside any WRF mid-latitude domain, harmless.)
#[inline]
fn clamp_lat(lat_deg: f64) -> f64 {
    lat_deg.clamp(-89.999, 89.999)
}

/// Clamp a *reference* latitude (a standard parallel) only to avoid a literal
/// pole singularity. Unlike the ported source we do NOT snap near-equatorial
/// parallels to 10 deg — WRF's real `TRUELAT` values must be used verbatim.
#[inline]
fn clamp_ref_lat(lat_deg: f64) -> f64 {
    lat_deg.clamp(-89.9, 89.9)
}

/// Wrap a longitude difference into `(-180, 180]` degrees.
#[inline]
fn normalize_lon(lon_deg: f64) -> f64 {
    let mut lon = lon_deg % 360.0;
    if lon > 180.0 {
        lon -= 360.0;
    } else if lon <= -180.0 {
        lon += 360.0;
    }
    lon
}

/// Errors building a projection from WRF attributes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameError {
    /// `MAP_PROJ` value SimSat does not implement.
    UnsupportedProjection(i32),
    /// The grid is too small / degenerate to anchor a projection (need >= 2x2).
    DegenerateGrid,
}

impl std::fmt::Display for FrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedProjection(code) => {
                write!(f, "unsupported WRF MAP_PROJ value: {code}")
            }
            Self::DegenerateGrid => write!(f, "grid too small to anchor a projection"),
        }
    }
}

impl std::error::Error for FrameError {}

/// SimSat-internal `map_proj` code for a ROTATED lat-lon grid (GRIB2 grid
/// template 3.1 — the RRFS North America native grid). WRF itself expresses a
/// rotated pole as `MAP_PROJ = 6` plus `POLE_LAT`/`POLE_LON` attributes, but
/// adding pole fields to [`WrfProjectionParams`] would break every existing
/// struct-literal construction site (api.rs / the studio, owned by parallel
/// agents this wave), so the rotated pole rides in the EXISTING
/// unused-for-lat-lon fields instead: **`truelat1_deg` = the rotated NORTH
/// pole's geographic latitude, `truelat2_deg` = its longitude** (RRFS: 35.0 /
/// 67.0, from the GRIB south pole at (-35, 247)). `dx_m`/`dy_m` are METRES
/// (rotated-degree spacing x [`ROTATED_LATLON_M_PER_DEG`]) so every
/// metre-based consumer — the cloud-march horizontal pitch, top-down plane
/// extents — works through the generic [`GridGeoref`] interface unchanged.
/// No existing producer emits 203 and no existing arm changed, so every
/// existing projection is byte-identical by construction. The manifest
/// persists these fields verbatim (`ManifestProjection` is field-for-field),
/// so a cached RRFS run round-trips exactly.
pub const MAP_PROJ_ROTATED_LATLON: i32 = 203;

/// Metres per degree of rotated-grid arc on the WRF sphere: `R * pi / 180`
/// (~111,177 m). The [`MapProjection::RotatedLatLon`] plane is rotated
/// lat/lon SCALED to metres by this constant — see [`MAP_PROJ_ROTATED_LATLON`].
pub const ROTATED_LATLON_M_PER_DEG: f64 = R_EARTH * PI / 180.0;

/// WRF projection attributes read from the file's globals (design section 1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WrfProjectionParams {
    /// `MAP_PROJ`: 1=Lambert, 2=polar stereographic, 3=Mercator, 6=lat/lon;
    /// [`MAP_PROJ_ROTATED_LATLON`] (203) = SimSat's rotated lat-lon (see its doc
    /// for the field-reuse convention).
    pub map_proj: i32,
    pub truelat1_deg: f64,
    pub truelat2_deg: f64,
    pub stand_lon_deg: f64,
    pub cen_lat_deg: f64,
    pub cen_lon_deg: f64,
    /// Grid spacing (m) for projected grids; for lat/lon it is unused (the degree
    /// increments come from the stored coordinates).
    pub dx_m: f64,
    pub dy_m: f64,
}

/// The analytic projection SHAPE (no georeference offset). `project` returns
/// projection-plane coordinates: meters for Lambert/PS/Mercator, degrees for
/// lat/lon (matching the georeference `dx`/`dy` unit for each kind).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MapProjection {
    /// Lambert conformal conic. `n` cone constant, `f` scale factor, `stand_lon` (deg).
    Lambert { n: f64, f: f64, stand_lon_deg: f64 },
    /// Polar stereographic. `k = (1+sin(truelat))/2`, `central_meridian` (deg).
    PolarStereographic {
        k: f64,
        central_meridian_deg: f64,
        south_pole: bool,
    },
    /// Mercator. `scale = cos(truelat)`, `central_meridian` (deg).
    Mercator {
        scale: f64,
        central_meridian_deg: f64,
    },
    /// Geographic lat/lon (cylindrical equidistant). Plane units are degrees.
    LatLon { central_meridian_deg: f64 },
    /// Rotated lat-lon (GRIB2 template 3.1; the RRFS NA grid). Plane units are
    /// METRES: rotated coordinates scaled by [`ROTATED_LATLON_M_PER_DEG`], so
    /// metre-based consumers (march pitch, top-down extents) work unchanged.
    /// The rotation math mirrors grib-core's `rotated_to_geographic` (COSMO
    /// convention, rotation angle 0): with `alpha` the rotated NORTH pole's
    /// geographic latitude and the rotated SOUTH pole at `south_pole_lon_deg`,
    /// the transform is a rotation about the axis through the south-pole
    /// meridian (unit-tested against grib-core's own grid math).
    RotatedLatLon {
        /// `sin`/`cos` of the rotated north pole's geographic latitude.
        sin_alpha: f64,
        cos_alpha: f64,
        /// Longitude of the rotated SOUTH pole (deg), the reference meridian.
        south_pole_lon_deg: f64,
    },
}

impl MapProjection {
    /// Build the projection SHAPE from WRF attributes (no georeference yet).
    pub fn from_wrf(p: &WrfProjectionParams) -> Result<Self, FrameError> {
        match p.map_proj {
            1 => Ok(Self::lambert(
                p.truelat1_deg,
                p.truelat2_deg,
                p.stand_lon_deg,
            )),
            2 => Ok(Self::polar_stereographic(
                p.truelat1_deg,
                p.stand_lon_deg,
                p.cen_lat_deg < 0.0,
            )),
            3 => Ok(Self::mercator(p.truelat1_deg, p.stand_lon_deg)),
            6 => Ok(Self::LatLon {
                central_meridian_deg: p.stand_lon_deg,
            }),
            MAP_PROJ_ROTATED_LATLON => Ok(Self::rotated_latlon(p.truelat1_deg, p.truelat2_deg)),
            other => Err(FrameError::UnsupportedProjection(other)),
        }
    }

    /// Lambert conformal conic from two standard parallels + central meridian.
    pub fn lambert(truelat1_deg: f64, truelat2_deg: f64, stand_lon_deg: f64) -> Self {
        let phi1 = clamp_ref_lat(truelat1_deg) * DEG2RAD;
        let phi2 = clamp_ref_lat(truelat2_deg) * DEG2RAD;
        let mut n = if (truelat1_deg - truelat2_deg).abs() < 1.0e-10 {
            phi1.sin()
        } else {
            let num = phi1.cos().ln() - phi2.cos().ln();
            let den = (PI / 4.0 + phi2 / 2.0).tan().ln() - (PI / 4.0 + phi1 / 2.0).tan().ln();
            num / den
        };
        if n.abs() < 1.0e-8 {
            // Degenerate (both parallels ~ equator): fall back to a tangent cone.
            n = if phi1.abs() >= 1.0e-8 {
                phi1.sin()
            } else {
                (10.0 * DEG2RAD).sin()
            };
        }
        let f = phi1.cos() * (PI / 4.0 + phi1 / 2.0).tan().powf(n) / n;
        Self::Lambert {
            n,
            f,
            stand_lon_deg,
        }
    }

    /// Polar stereographic from the true-scale latitude + central meridian.
    pub fn polar_stereographic(
        true_latitude_deg: f64,
        central_meridian_deg: f64,
        south_pole: bool,
    ) -> Self {
        let lat_ts = clamp_ref_lat(true_latitude_deg) * DEG2RAD;
        Self::PolarStereographic {
            k: (1.0 + lat_ts.sin()) / 2.0,
            central_meridian_deg,
            south_pole,
        }
    }

    /// Mercator from the latitude of true scale + central meridian.
    pub fn mercator(latitude_of_true_scale_deg: f64, central_meridian_deg: f64) -> Self {
        Self::Mercator {
            scale: (clamp_ref_lat(latitude_of_true_scale_deg) * DEG2RAD)
                .cos()
                .max(1.0e-6),
            central_meridian_deg,
        }
    }

    /// Rotated lat-lon from the rotated NORTH pole's geographic position
    /// (see [`MAP_PROJ_ROTATED_LATLON`]; the pole at `(90, any)` degenerates to
    /// an unrotated grid). The construction is deterministic in the two pole
    /// values, so a manifest round-trip rebuilds the projection bit-identically.
    pub fn rotated_latlon(pole_lat_deg: f64, pole_lon_deg: f64) -> Self {
        let alpha = pole_lat_deg * DEG2RAD;
        Self::RotatedLatLon {
            sin_alpha: alpha.sin(),
            cos_alpha: alpha.cos(),
            south_pole_lon_deg: normalize_lon(pole_lon_deg + 180.0),
        }
    }

    /// Forward: geodetic `(lat, lon)` (deg) -> projection-plane `(u, v)`.
    pub fn project(&self, lat_deg: f64, lon_deg: f64) -> (f64, f64) {
        match *self {
            Self::Lambert {
                n,
                f,
                stand_lon_deg,
            } => {
                let phi = clamp_lat(lat_deg) * DEG2RAD;
                let dlon = normalize_lon(lon_deg - stand_lon_deg) * DEG2RAD;
                let rho = R_EARTH * f / (PI / 4.0 + phi / 2.0).tan().powf(n);
                let theta = n * dlon;
                (rho * theta.sin(), -rho * theta.cos())
            }
            Self::PolarStereographic {
                k,
                central_meridian_deg,
                south_pole,
            } => {
                let phi = clamp_lat(lat_deg) * DEG2RAD;
                let theta = normalize_lon(lon_deg - central_meridian_deg) * DEG2RAD;
                if south_pole {
                    let rho = 2.0 * R_EARTH * k * (PI / 4.0 + phi / 2.0).tan();
                    (rho * theta.sin(), rho * theta.cos())
                } else {
                    let rho = 2.0 * R_EARTH * k * (PI / 4.0 - phi / 2.0).tan();
                    (rho * theta.sin(), -rho * theta.cos())
                }
            }
            Self::Mercator {
                scale,
                central_meridian_deg,
            } => {
                let phi = clamp_lat(lat_deg) * DEG2RAD;
                let lambda = normalize_lon(lon_deg - central_meridian_deg) * DEG2RAD;
                let x = R_EARTH * scale * lambda;
                let y = R_EARTH * scale * (PI / 4.0 + phi / 2.0).tan().ln();
                (x, y)
            }
            Self::LatLon {
                central_meridian_deg,
            } => (
                normalize_lon(lon_deg - central_meridian_deg),
                clamp_lat(lat_deg),
            ),
            Self::RotatedLatLon {
                sin_alpha,
                cos_alpha,
                south_pole_lon_deg,
            } => {
                // Geographic -> rotated: the exact inverse (transpose) of the
                // grib-core `rotated_to_geographic` rotation about the Y axis
                // of the south-pole-meridian frame.
                let phi = clamp_lat(lat_deg) * DEG2RAD;
                let dlon = normalize_lon(lon_deg - south_pole_lon_deg) * DEG2RAD;
                let (sin_phi, cos_phi) = (phi.sin(), phi.cos());
                let x = cos_phi * dlon.cos();
                let y = cos_phi * dlon.sin();
                let z = sin_phi;
                let rlat = (-x * cos_alpha + z * sin_alpha).clamp(-1.0, 1.0).asin();
                let rlon = y.atan2(x * sin_alpha + z * cos_alpha);
                // Plane = rotated angle (rad) * R == rotated degrees * M_PER_DEG.
                (rlon * R_EARTH, rlat * R_EARTH)
            }
        }
    }

    /// Inverse: projection-plane `(u, v)` -> geodetic `(lat, lon)` (deg).
    pub fn unproject(&self, u: f64, v: f64) -> Option<(f64, f64)> {
        if !u.is_finite() || !v.is_finite() {
            return None;
        }
        match *self {
            Self::Lambert {
                n,
                f,
                stand_lon_deg,
            } => {
                if n.abs() < 1.0e-12 || f.abs() < 1.0e-12 {
                    return None;
                }
                // With internal rho0 = 0, (rho0 - y) == -v.
                let rho_abs = (u * u + v * v).sqrt();
                if rho_abs <= 0.0 {
                    return None;
                }
                let rho_sign = n.signum();
                let rho = rho_abs * rho_sign;
                let theta = (u * rho_sign).atan2((-v) * rho_sign);
                let ratio = R_EARTH * f / rho;
                if ratio <= 0.0 || !ratio.is_finite() {
                    return None;
                }
                let phi = 2.0 * ratio.powf(1.0 / n).atan() - PI / 2.0;
                let lon = stand_lon_deg + theta / n * RAD2DEG;
                Some((phi * RAD2DEG, normalize_lon(lon)))
            }
            Self::PolarStereographic {
                k,
                central_meridian_deg,
                south_pole,
            } => {
                if k <= 0.0 {
                    return None;
                }
                let rho = (u * u + v * v).sqrt();
                let scaled = rho / (2.0 * R_EARTH * k);
                let (phi, theta) = if south_pole {
                    (2.0 * scaled.atan() - PI / 2.0, u.atan2(v))
                } else {
                    (PI / 2.0 - 2.0 * scaled.atan(), u.atan2(-v))
                };
                let lon = central_meridian_deg + theta * RAD2DEG;
                Some((phi * RAD2DEG, normalize_lon(lon)))
            }
            Self::Mercator {
                scale,
                central_meridian_deg,
            } => {
                if scale <= 0.0 {
                    return None;
                }
                let lon = central_meridian_deg + u / (R_EARTH * scale) * RAD2DEG;
                let lat = (2.0 * (v / (R_EARTH * scale)).exp().atan() - PI / 2.0) * RAD2DEG;
                Some((clamp_lat(lat), normalize_lon(lon)))
            }
            Self::LatLon {
                central_meridian_deg,
            } => Some((clamp_lat(v), normalize_lon(u + central_meridian_deg))),
            Self::RotatedLatLon {
                sin_alpha,
                cos_alpha,
                south_pole_lon_deg,
            } => {
                // Rotated -> geographic: grib-core's `rotated_to_geographic`
                // formulas verbatim (rotation angle 0), plane metres -> radians.
                let rlon = u / R_EARTH;
                let rlat = v / R_EARTH;
                let (sin_rlat, cos_rlat) = (rlat.sin(), rlat.cos());
                let (sin_rlon, cos_rlon) = (rlon.sin(), rlon.cos());
                let lat = (sin_rlat * sin_alpha + cos_rlat * cos_rlon * cos_alpha)
                    .clamp(-1.0, 1.0)
                    .asin();
                let lon = (cos_rlat * sin_rlon)
                    .atan2(cos_rlat * cos_rlon * sin_alpha - sin_rlat * cos_alpha);
                Some((
                    lat * RAD2DEG,
                    normalize_lon(lon * RAD2DEG + south_pole_lon_deg),
                ))
            }
        }
    }
}

/// A projection plus its grid georeference: maps `(lat, lon)` to fractional,
/// 0-based grid indices `(i, j)` and back.
///
/// The georeference is a pure scale+translation of the projection plane, valid
/// because WRF aligns grid-north with the projection `v`-axis at `STAND_LON` (no
/// rotation). The anchor is a known grid point — in ingest, the grid CENTER read
/// from `XLAT`/`XLONG` — so the absolute offset is exact and the projection SHAPE
/// (cone constant, radii from the attributes) is what the ratchet actually tests.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridGeoref {
    projection: MapProjection,
    ref_i: f64,
    ref_j: f64,
    ref_u: f64,
    ref_v: f64,
    dx: f64,
    dy: f64,
}

impl GridGeoref {
    /// Build from a projection and an anchor point `(ref_lat, ref_lon)` known to
    /// live at fractional grid index `(ref_i, ref_j)`, with plane-unit spacing
    /// `dx`/`dy` per cell (meters for projected kinds, degrees for lat/lon).
    pub fn new(
        projection: MapProjection,
        ref_i: f64,
        ref_j: f64,
        ref_lat_deg: f64,
        ref_lon_deg: f64,
        dx: f64,
        dy: f64,
    ) -> Self {
        let (ref_u, ref_v) = projection.project(ref_lat_deg, ref_lon_deg);
        Self {
            projection,
            ref_i,
            ref_j,
            ref_u,
            ref_v,
            dx,
            dy,
        }
    }

    /// Build a center-anchored georeference from WRF attributes and the stored
    /// `XLAT`/`XLONG` planes (row-major `[ny][nx]`, degrees). The anchor is the
    /// grid center index `((nx-1)/2, (ny-1)/2)`.
    pub fn from_wrf_center(
        params: &WrfProjectionParams,
        nx: usize,
        ny: usize,
        xlat: &[f32],
        xlong: &[f32],
    ) -> Result<Self, FrameError> {
        let projection = MapProjection::from_wrf(params)?;
        let (ref_i, ref_j, ref_lat, ref_lon, dx, dy) =
            wrf_center_anchor(params, nx, ny, xlat, xlong)?;
        Ok(Self::new(
            projection, ref_i, ref_j, ref_lat, ref_lon, dx, dy,
        ))
    }

    /// Rebuild a georeference from a PERSISTED anchor (the manifest's per-timestep
    /// `ref_*`/`dx`/`dy` fields). The construction is EXACTLY [`Self::from_wrf_center`]
    /// given the same values — the projection shape comes deterministically from the
    /// same attributes and the anchor is projected identically — so the cached-run
    /// path reconstructs the wrfout path's georef BIT-IDENTICALLY (closes deferred
    /// M1 NOTE-4: no more duplicate store-run dirs forked by the low f32 bits of the
    /// two anchoring paths), and a MOVING NEST's timestep is anchored where that
    /// timestep's domain actually sits.
    #[allow(clippy::too_many_arguments)]
    pub fn from_anchor(
        params: &WrfProjectionParams,
        ref_i: f64,
        ref_j: f64,
        ref_lat_deg: f64,
        ref_lon_deg: f64,
        dx: f64,
        dy: f64,
    ) -> Result<Self, FrameError> {
        let projection = MapProjection::from_wrf(params)?;
        Ok(Self::new(
            projection,
            ref_i,
            ref_j,
            ref_lat_deg,
            ref_lon_deg,
            dx,
            dy,
        ))
    }

    /// Build a center-anchored georeference from WRF attributes alone, anchoring
    /// at the grid center `((nx-1)/2, (ny-1)/2)` with the domain center lat/lon
    /// `CEN_LAT`/`CEN_LON`. Used by the studio's cached-run flow when the wrfout
    /// (and thus `XLAT`/`XLONG`) is not open — `CEN_LAT`/`CEN_LON` is the WRF
    /// domain center, so this is a close (documented) approximation of
    /// [`Self::from_wrf_center`]. For lat/lon grids the degree increments come
    /// from `dx`/`dy` (already in degrees for `MAP_PROJ = 6`).
    pub fn from_params_center(
        params: &WrfProjectionParams,
        nx: usize,
        ny: usize,
    ) -> Result<Self, FrameError> {
        if nx < 2 || ny < 2 {
            return Err(FrameError::DegenerateGrid);
        }
        let projection = MapProjection::from_wrf(params)?;
        let ci = (nx - 1) as f64 / 2.0;
        let cj = (ny - 1) as f64 / 2.0;
        Ok(Self::new(
            projection,
            ci,
            cj,
            params.cen_lat_deg,
            params.cen_lon_deg,
            params.dx_m,
            params.dy_m,
        ))
    }

    /// The underlying projection.
    pub fn projection(&self) -> MapProjection {
        self.projection
    }

    /// The projection-PLANE coordinate `(u, v)` of a fractional 0-based grid index
    /// `(i, j)` — the affine map `u = ref_u + (i - ref_i) * dx`, `v = ref_v + (j -
    /// ref_j) * dy`. Plane units are metres for Lambert/PS/Mercator and degrees for
    /// lat/lon (matching `dx`/`dy`). This is the inverse-direction companion of
    /// [`Self::forward`]'s internal `(u, v) -> (i, j)` step, exposed so a caller can
    /// build the top-down map's imshow extent in projection metres (the coordinate a
    /// cartopy Lambert `GeoAxes` places the image in). Pure affine; no unprojection.
    pub fn plane_uv(&self, i: f64, j: f64) -> (f64, f64) {
        (
            self.ref_u + (i - self.ref_i) * self.dx,
            self.ref_v + (j - self.ref_j) * self.dy,
        )
    }

    /// Forward: `(lat, lon)` -> fractional 0-based grid `(i, j)`.
    pub fn forward(&self, lat_deg: f64, lon_deg: f64) -> (f64, f64) {
        let (u, v) = self.projection.project(lat_deg, lon_deg);
        (
            self.ref_i + (u - self.ref_u) / self.dx,
            self.ref_j + (v - self.ref_v) / self.dy,
        )
    }

    /// Inverse: fractional 0-based grid `(i, j)` -> `(lat, lon)`.
    pub fn inverse(&self, i: f64, j: f64) -> Option<(f64, f64)> {
        let u = self.ref_u + (i - self.ref_i) * self.dx;
        let v = self.ref_v + (j - self.ref_j) * self.dy;
        self.projection.unproject(u, v)
    }
}

/// The center-anchor values [`GridGeoref::from_wrf_center`] anchors with:
/// `(ref_i, ref_j, ref_lat_deg, ref_lon_deg, dx, dy)`. Exposed as the SINGLE source of
/// the anchoring rule so ingest can PERSIST the anchor per timestep in the run manifest
/// and the cached-run path can rebuild the georef bit-identically via
/// [`GridGeoref::from_anchor`]. For a `MAP_PROJ = 6` lat/lon grid the `dx`/`dy` are the
/// degree increments straight off the STORED coordinates (not the `DX`/`DY` attributes),
/// exactly as `from_wrf_center` has always taken them.
pub fn wrf_center_anchor(
    params: &WrfProjectionParams,
    nx: usize,
    ny: usize,
    xlat: &[f32],
    xlong: &[f32],
) -> Result<(f64, f64, f64, f64, f64, f64), FrameError> {
    if nx < 2 || ny < 2 || xlat.len() != nx * ny || xlong.len() != nx * ny {
        return Err(FrameError::DegenerateGrid);
    }
    let ci = (nx - 1) / 2;
    let cj = (ny - 1) / 2;
    let c = cj * nx + ci;
    let ref_lat = xlat[c] as f64;
    let ref_lon = xlong[c] as f64;
    let (dx, dy) = if params.map_proj == 6 {
        // Degree increments straight off the stored coordinates.
        let dlon = (xlong[c + 1] - xlong[c]) as f64;
        let dlat = (xlat[c + nx] - xlat[c]) as f64;
        (dlon, dlat)
    } else {
        (params.dx_m, params.dy_m)
    };
    Ok((ci as f64, cj as f64, ref_lat, ref_lon, dx, dy))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mercator project() locked against hand-computed ground truth (catches a
    /// broken port even without the fixture).
    #[test]
    fn mercator_project_matches_hand_computation() {
        let m = MapProjection::mercator(0.0, 0.0); // scale = cos(0) = 1
        let (x0, y0) = m.project(0.0, 0.0);
        assert!(x0.abs() < 1.0e-6 && y0.abs() < 1.0e-6);
        // 1 deg of longitude at the equator = R * 1deg = 6_370_000 * pi/180.
        let (x1, _) = m.project(0.0, 1.0);
        assert!((x1 - R_EARTH * DEG2RAD).abs() < 1.0e-3, "x1={x1}");
        // y at 45N = R * ln(tan(67.5 deg)).
        let (_, y45) = m.project(45.0, 0.0);
        let expect = R_EARTH * (67.5_f64 * DEG2RAD).tan().ln();
        assert!((y45 - expect).abs() < 1.0e-3, "y45={y45} expect={expect}");
    }

    fn assert_plane_roundtrip(p: &MapProjection, lat: f64, lon: f64) {
        let (u, v) = p.project(lat, lon);
        let (lat2, lon2) = p.unproject(u, v).expect("unproject");
        assert!((lat - lat2).abs() < 1.0e-6, "lat {lat} -> {lat2}");
        let dlon = normalize_lon(lon - lon2);
        assert!(dlon.abs() < 1.0e-6, "lon {lon} -> {lon2}");
    }

    #[test]
    fn projection_plane_round_trips_all_kinds() {
        let lambert = MapProjection::lambert(30.0, 60.0, -95.0);
        let merc = MapProjection::mercator(20.0, -80.0);
        let ps_n = MapProjection::polar_stereographic(60.0, -105.0, false);
        let ps_s = MapProjection::polar_stereographic(-60.0, 0.0, true);
        let latlon = MapProjection::LatLon {
            central_meridian_deg: 10.0,
        };
        for (lat, lon) in [(25.0, -100.0), (40.0, -90.0), (35.0, -85.0), (45.0, -110.0)] {
            assert_plane_roundtrip(&lambert, lat, lon);
            assert_plane_roundtrip(&merc, lat, lon);
            assert_plane_roundtrip(&ps_n, lat, lon);
            assert_plane_roundtrip(&latlon, lat, lon);
        }
        // Southern-hemisphere polar stereographic.
        for (lat, lon) in [(-30.0, 20.0), (-50.0, -40.0), (-70.0, 130.0)] {
            assert_plane_roundtrip(&ps_s, lat, lon);
        }
    }

    /// Build a synthetic georeferenced grid, walk (i,j) -> (lat,lon) -> (i,j),
    /// and assert recovery far tighter than the 0.05-cell ratchet.
    fn assert_grid_roundtrip(georef: &GridGeoref, nx: usize, ny: usize) {
        for j in (0..ny).step_by(7) {
            for i in (0..nx).step_by(7) {
                let (lat, lon) = georef.inverse(i as f64, j as f64).expect("inverse");
                let (i2, j2) = georef.forward(lat, lon);
                assert!((i2 - i as f64).abs() < 1.0e-4, "i {i} -> {i2}");
                assert!((j2 - j as f64).abs() < 1.0e-4, "j {j} -> {j2}");
            }
        }
    }

    #[test]
    fn georef_grid_round_trips_lambert() {
        // CONUS-like Lambert, 300 km cells scaled down to a 200x150 grid at 3 km.
        let proj = MapProjection::lambert(30.0, 60.0, -97.5);
        let georef = GridGeoref::new(proj, 100.0, 75.0, 39.0, -97.5, 3000.0, 3000.0);
        // Anchor maps to itself exactly.
        let (ai, aj) = georef.forward(39.0, -97.5);
        assert!((ai - 100.0).abs() < 1.0e-6 && (aj - 75.0).abs() < 1.0e-6);
        assert_grid_roundtrip(&georef, 200, 150);
    }

    #[test]
    fn georef_grid_round_trips_mercator() {
        // Low-latitude Mercator (hurricane-belt), true-scale 20N.
        let proj = MapProjection::mercator(20.0, -85.0);
        let georef = GridGeoref::new(proj, 150.0, 100.0, 25.0, -85.0, 4000.0, 4000.0);
        assert_grid_roundtrip(&georef, 300, 200);
    }

    #[test]
    fn georef_grid_round_trips_polar_and_latlon() {
        let ps = MapProjection::polar_stereographic(60.0, -100.0, false);
        let g_ps = GridGeoref::new(ps, 128.0, 128.0, 70.0, -100.0, 5000.0, 5000.0);
        assert_grid_roundtrip(&g_ps, 256, 256);

        let ll = MapProjection::LatLon {
            central_meridian_deg: 0.0,
        };
        let g_ll = GridGeoref::new(ll, 90.0, 60.0, 45.0, 10.0, 0.1, 0.1);
        assert_grid_roundtrip(&g_ll, 180, 120);
    }

    #[test]
    fn from_wrf_center_anchors_on_the_center_cell() {
        // Build a tiny synthetic Lambert grid: fill XLAT/XLONG via the inverse,
        // then confirm from_wrf_center reproduces every cell within the ratchet.
        let params = WrfProjectionParams {
            map_proj: 1,
            truelat1_deg: 30.0,
            truelat2_deg: 60.0,
            stand_lon_deg: -97.5,
            cen_lat_deg: 39.0,
            cen_lon_deg: -97.5,
            dx_m: 3000.0,
            dy_m: 3000.0,
        };
        let (nx, ny) = (61usize, 41usize);
        let truth = GridGeoref::new(
            MapProjection::lambert(30.0, 60.0, -97.5),
            ((nx - 1) / 2) as f64,
            ((ny - 1) / 2) as f64,
            39.0,
            -97.5,
            3000.0,
            3000.0,
        );
        let mut xlat = vec![0f32; nx * ny];
        let mut xlong = vec![0f32; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let (lat, lon) = truth.inverse(i as f64, j as f64).unwrap();
                xlat[j * nx + i] = lat as f32;
                xlong[j * nx + i] = lon as f32;
            }
        }
        let georef = GridGeoref::from_wrf_center(&params, nx, ny, &xlat, &xlong).unwrap();
        // Ratchet: forward every stored coord lands within 0.05 cell of its index.
        let mut worst = 0.0f64;
        for j in 0..ny {
            for i in 0..nx {
                let (fi, fj) = georef.forward(xlat[j * nx + i] as f64, xlong[j * nx + i] as f64);
                worst = worst.max((fi - i as f64).abs()).max((fj - j as f64).abs());
            }
        }
        assert!(worst < 0.05, "worst grid error {worst} cells exceeds 0.05");
    }

    #[test]
    fn from_params_center_matches_wrf_center_for_synthetic_grid() {
        // CEN_LAT/CEN_LON at the grid center should reproduce from_wrf_center.
        let params = WrfProjectionParams {
            map_proj: 1,
            truelat1_deg: 30.0,
            truelat2_deg: 60.0,
            stand_lon_deg: -97.5,
            cen_lat_deg: 39.0,
            cen_lon_deg: -97.5,
            dx_m: 3000.0,
            dy_m: 3000.0,
        };
        let (nx, ny) = (61usize, 41usize);
        let georef = GridGeoref::from_params_center(&params, nx, ny).unwrap();
        // The anchor (grid center) maps back to CEN_LAT/CEN_LON.
        let (lat, lon) = georef
            .inverse((nx - 1) as f64 / 2.0, (ny - 1) as f64 / 2.0)
            .unwrap();
        assert!((lat - 39.0).abs() < 1e-6 && (lon + 97.5).abs() < 1e-6);
        // And a forward of that anchor lands on the center index.
        let (fi, fj) = georef.forward(39.0, -97.5);
        assert!((fi - (nx - 1) as f64 / 2.0).abs() < 1e-6);
        assert!((fj - (ny - 1) as f64 / 2.0).abs() < 1e-6);
    }

    #[test]
    fn plane_uv_is_the_affine_inverse_of_forward() {
        // plane_uv(i, j) must equal the projection-plane (u, v) of that grid index —
        // i.e. project(inverse(i, j)) — and, fed back through forward, recover (i, j).
        let proj = MapProjection::lambert(30.0, 60.0, -97.5);
        let georef = GridGeoref::new(proj, 100.0, 75.0, 39.0, -97.5, 3000.0, 3000.0);
        for (i, j) in [(0.0, 0.0), (100.0, 75.0), (199.0, 149.0), (-0.5, 149.5)] {
            let (u, v) = georef.plane_uv(i, j);
            // The plane coord unprojects to a lat/lon that forwards back to (i, j).
            let (lat, lon) = proj.unproject(u, v).expect("unproject");
            let (fi, fj) = georef.forward(lat, lon);
            assert!((fi - i).abs() < 1e-6, "i {i} -> {fi}");
            assert!((fj - j).abs() < 1e-6, "j {j} -> {fj}");
            // And project(inverse(i, j)) reproduces the same plane coord.
            if let Some((la, lo)) = georef.inverse(i, j) {
                let (pu, pv) = proj.project(la, lo);
                assert!((pu - u).abs() < 1e-3 && (pv - v).abs() < 1e-3);
            }
        }
    }

    /// WS3 / M1 NOTE-4: `from_anchor` fed the values `wrf_center_anchor` extracts must
    /// reproduce `from_wrf_center` BIT-IDENTICALLY (the cached-run reconstruction), for
    /// a projected grid AND a lat/lon grid (whose dx/dy come from the stored coords,
    /// not the attributes — the reason dx/dy are persisted alongside the anchor).
    #[test]
    fn from_anchor_reconstructs_from_wrf_center_bit_identically() {
        // Lambert grid built from a known truth georef.
        let params = WrfProjectionParams {
            map_proj: 1,
            truelat1_deg: 30.0,
            truelat2_deg: 60.0,
            stand_lon_deg: -97.5,
            cen_lat_deg: 39.0,
            cen_lon_deg: -97.5,
            dx_m: 3000.0,
            dy_m: 3000.0,
        };
        let (nx, ny) = (31usize, 21usize);
        let truth = GridGeoref::new(
            MapProjection::lambert(30.0, 60.0, -97.5),
            ((nx - 1) / 2) as f64,
            ((ny - 1) / 2) as f64,
            39.0,
            -97.5,
            3000.0,
            3000.0,
        );
        let mut xlat = vec![0f32; nx * ny];
        let mut xlong = vec![0f32; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let (lat, lon) = truth.inverse(i as f64, j as f64).unwrap();
                xlat[j * nx + i] = lat as f32;
                xlong[j * nx + i] = lon as f32;
            }
        }
        let direct = GridGeoref::from_wrf_center(&params, nx, ny, &xlat, &xlong).unwrap();
        let (ri, rj, rlat, rlon, dx, dy) =
            wrf_center_anchor(&params, nx, ny, &xlat, &xlong).unwrap();
        let rebuilt = GridGeoref::from_anchor(&params, ri, rj, rlat, rlon, dx, dy).unwrap();
        // PartialEq on GridGeoref is field-exact f64 equality: bit-identical.
        assert_eq!(direct, rebuilt, "cached-path georef must not fork");

        // Lat/lon (MAP_PROJ = 6) grid: the anchor's dx/dy must come from the STORED
        // coordinate increments, which here deliberately DIFFER from the attributes.
        let ll_params = WrfProjectionParams {
            map_proj: 6,
            truelat1_deg: 0.0,
            truelat2_deg: 0.0,
            stand_lon_deg: 0.0,
            cen_lat_deg: 45.0,
            cen_lon_deg: 10.0,
            dx_m: 999.0, // NOT the degree increment — must be ignored for map_proj 6
            dy_m: 999.0,
        };
        let ll_truth = GridGeoref::new(
            MapProjection::LatLon {
                central_meridian_deg: 0.0,
            },
            ((nx - 1) / 2) as f64,
            ((ny - 1) / 2) as f64,
            45.0,
            10.0,
            0.1,
            0.1,
        );
        let mut lat2 = vec![0f32; nx * ny];
        let mut lon2 = vec![0f32; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let (lat, lon) = ll_truth.inverse(i as f64, j as f64).unwrap();
                lat2[j * nx + i] = lat as f32;
                lon2[j * nx + i] = lon as f32;
            }
        }
        let (ri, rj, rlat, rlon, dx, dy) =
            wrf_center_anchor(&ll_params, nx, ny, &lat2, &lon2).unwrap();
        assert!(
            (dx - 0.1).abs() < 1e-4 && (dy - 0.1).abs() < 1e-4,
            "lat/lon anchor dx/dy must be the stored degree increments, got ({dx}, {dy})"
        );
        let ll_direct = GridGeoref::from_wrf_center(&ll_params, nx, ny, &lat2, &lon2).unwrap();
        let ll_rebuilt = GridGeoref::from_anchor(&ll_params, ri, rj, rlat, rlon, dx, dy).unwrap();
        assert_eq!(ll_direct, ll_rebuilt);
    }

    #[test]
    fn unsupported_projection_is_reported() {
        let params = WrfProjectionParams {
            map_proj: 99,
            truelat1_deg: 30.0,
            truelat2_deg: 30.0,
            stand_lon_deg: 0.0,
            cen_lat_deg: 0.0,
            cen_lon_deg: 0.0,
            dx_m: 1000.0,
            dy_m: 1000.0,
        };
        assert_eq!(
            MapProjection::from_wrf(&params),
            Err(FrameError::UnsupportedProjection(99))
        );
    }

    #[test]
    fn rotated_latlon_degenerate_pole_matches_plain_latlon() {
        // A rotated grid whose north pole sits at the TRUE north pole with the
        // south-pole meridian at 0 (pole_lon = -180) is an unrotated grid: the
        // rotated coordinates ARE the geographic ones. Its plane differs from
        // the plain LatLon plane only by the metres-per-degree scale.
        let rot = MapProjection::rotated_latlon(90.0, -180.0);
        let plain = MapProjection::LatLon {
            central_meridian_deg: 0.0,
        };
        for &(lat, lon) in &[
            (0.0f64, 0.0f64),
            (38.5, -97.5),
            (55.0, -113.0),
            (-33.0, 151.2),
            (70.0, 179.0),
        ] {
            let (ru, rv) = rot.project(lat, lon);
            let (pu, pv) = plain.project(lat, lon);
            assert!(
                (ru / ROTATED_LATLON_M_PER_DEG - pu).abs() < 1e-9,
                "u mismatch at ({lat}, {lon})"
            );
            assert!(
                (rv / ROTATED_LATLON_M_PER_DEG - pv).abs() < 1e-9,
                "v mismatch at ({lat}, {lon})"
            );
        }
    }

    #[test]
    fn rotated_latlon_round_trips_on_the_rrfs_pole() {
        // The RRFS NA rotation: GRIB south pole (-35, 247) -> north pole (35, 67).
        // from_wrf(203) reads the pole from the reused truelat fields (the
        // MAP_PROJ_ROTATED_LATLON convention); project/unproject must round-trip,
        // and the grid centre (rotated origin) must sit at geographic (55, -113).
        let params = WrfProjectionParams {
            map_proj: MAP_PROJ_ROTATED_LATLON,
            truelat1_deg: 35.0,
            truelat2_deg: 67.0,
            stand_lon_deg: 0.0,
            cen_lat_deg: 55.0,
            cen_lon_deg: -113.0,
            dx_m: 0.025 * ROTATED_LATLON_M_PER_DEG,
            dy_m: 0.025 * ROTATED_LATLON_M_PER_DEG,
        };
        let proj = MapProjection::from_wrf(&params).unwrap();
        // Rotated origin -> the NA domain centre.
        let (lat, lon) = proj.unproject(0.0, 0.0).unwrap();
        assert!((lat - 55.0).abs() < 1e-9, "origin lat {lat}");
        assert!((lon - (-113.0)).abs() < 1e-9, "origin lon {lon}");
        // Round trips across the domain (corners from the RRFS probe).
        for &(la, lo) in &[
            (55.0f64, -113.0f64),
            (21.14, -122.72),
            (41.48, 135.80),
            (-1.61, -157.33),
            (47.84, -60.92),
        ] {
            let (u, v) = proj.project(la, lo);
            let (bla, blo) = proj.unproject(u, v).unwrap();
            let dlon = normalize_lon(blo - lo);
            assert!(
                (bla - la).abs() < 1e-9 && dlon.abs() < 1e-9,
                "round trip failed at ({la}, {lo}) -> ({bla}, {blo})"
            );
        }
    }
}
