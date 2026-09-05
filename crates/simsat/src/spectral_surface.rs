//! Native-grid floating-point land spectra, without image gamma or quantization.
//!
//! HAMSTER black-sky albedo is used as a Lambertian approximation. It is not a
//! directional BRDF, contemporary land state, or TOA ABI reflectance. The full
//! wavelength cube remains available to the spectral observation operator.
use crate::{
    camera::SurfaceRaster, visible_sensor::AbiReflectiveBand, visible_solar::AbiSolarResponse,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path};

pub const RGB_WAVELENGTHS_UM: [f64; 3] = [0.680, 0.550, 0.440];
const MAX_DATA_BYTES: usize = 512 * 1024 * 1024;
const MAX_GRID_ERROR_CELLS: f64 = 0.005;

#[derive(Deserialize)]
struct FileRecord {
    bytes: usize,
    sha256: String,
}
#[derive(Deserialize)]
struct Header {
    schema_version: u32,
    quantity: String,
    nx: usize,
    ny: usize,
    wavelength_um: Vec<f64>,
    climatology_doy: u32,
    coordinate_layout: String,
    albedo_layout: String,
    files: BTreeMap<String, FileRecord>,
}

pub struct SpectralSurface {
    header: Header,
    coordinates: Vec<[f64; 2]>,
    albedo: Vec<f32>,
    valid: Vec<u8>,
    land: Vec<u8>,
}

impl SpectralSurface {
    pub fn load(manifest: &Path) -> Result<Self, String> {
        let size = std::fs::metadata(manifest)
            .map_err(|e| e.to_string())?
            .len();
        if size > 1024 * 1024 {
            return Err("spectral surface manifest exceeds 1 MiB".into());
        }
        let h: Header =
            serde_json::from_slice(&std::fs::read(manifest).map_err(|e| e.to_string())?)
                .map_err(|e| format!("spectral surface manifest: {e}"))?;
        let n =
            h.nx.checked_mul(h.ny)
                .ok_or("surface dimensions overflow")?;
        let count = n
            .checked_mul(h.wavelength_um.len())
            .ok_or("surface cube dimensions overflow")?;
        if h.schema_version != 1
            || h.quantity != "climatological_black_sky_surface_albedo"
            || h.coordinate_layout != "latitude_longitude_interleaved_f64le"
            || h.albedo_layout != "row_column_wavelength_f32le"
            || n == 0
            || h.wavelength_um.len() < 2
            || count > MAX_DATA_BYTES / 4
            || !(1..=365).contains(&h.climatology_doy)
            || !h.wavelength_um.iter().all(|w| w.is_finite() && *w > 0.0)
            || !h.wavelength_um.windows(2).all(|w| w[1] > w[0])
        {
            return Err("unsupported or invalid spectral surface contract".into());
        }
        let directory = manifest
            .parent()
            .ok_or("surface manifest needs a parent directory")?;
        let read = |name: &str, expected: usize| -> Result<Vec<u8>, String> {
            let rec = h
                .files
                .get(name)
                .ok_or_else(|| format!("surface manifest missing {name}"))?;
            let path = directory.join(name); // Fixed local names; manifest cannot redirect paths.
            if expected > MAX_DATA_BYTES
                || rec.bytes != expected
                || std::fs::metadata(&path)
                    .map_err(|e| format!("{}: {e}", path.display()))?
                    .len()
                    != expected as u64
            {
                return Err(format!("spectral surface size mismatch: {name}"));
            }
            let data = std::fs::read(path).map_err(|e| e.to_string())?;
            if format!("{:x}", Sha256::digest(&data)) != rec.sha256 {
                return Err(format!("spectral surface checksum mismatch: {name}"));
            }
            Ok(data)
        };
        let coordinates = read(
            "coordinates.bin",
            n.checked_mul(16).ok_or("coordinate size overflow")?,
        )?
        .chunks_exact(16)
        .map(|c| {
            [
                f64::from_le_bytes(c[..8].try_into().unwrap()),
                f64::from_le_bytes(c[8..].try_into().unwrap()),
            ]
        })
        .collect::<Vec<_>>();
        let albedo = read("albedo.bin", count * 4)?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<_>>();
        let valid = read("valid.bin", n)?;
        let land = read("land-mask.bin", n)?;
        let nw = h.wavelength_um.len();
        for (i, c) in coordinates.iter().enumerate() {
            if !c[0].is_finite()
                || !c[1].is_finite()
                || c[0].abs() > 90.0
                || c[1].abs() > 180.0
                || valid[i] > 1
                || land[i] > 1
                || valid[i] != land[i]
            {
                return Err(format!(
                    "invalid coordinate, mask or incomplete land spectrum at cell {i}"
                ));
            }
            if valid[i] == 1
                && !albedo[i * nw..(i + 1) * nw]
                    .iter()
                    .all(|a| a.is_finite() && (0.0..=1.0).contains(a))
            {
                return Err(format!("invalid surface albedo at cell {i}"));
            }
        }
        Ok(Self {
            header: h,
            coordinates,
            albedo,
            valid,
            land,
        })
    }

    pub fn climatology_doy(&self) -> u32 {
        self.header.climatology_doy
    }
    pub fn cell_count(&self) -> usize {
        self.valid.len()
    }
    pub fn coordinates(&self) -> &[[f64; 2]] {
        &self.coordinates
    }
    pub fn is_land(&self, cell: usize) -> bool {
        self.land.get(cell) == Some(&1)
    }

    /// Linear interpolation of the supplied spectrum; no extrapolation or no-data fill.
    pub fn sample(&self, cell: usize, wavelength_um: f64) -> Option<f64> {
        let w = &self.header.wavelength_um;
        if self.valid.get(cell) != Some(&1)
            || !wavelength_um.is_finite()
            || wavelength_um < w[0]
            || wavelength_um > w[w.len() - 1]
        {
            return None;
        }
        let hi = w
            .partition_point(|v| *v < wavelength_um)
            .clamp(1, w.len() - 1);
        let lo = hi - 1;
        let t = (wavelength_um - w[lo]) / (w[hi] - w[lo]);
        let a = &self.albedo[cell * w.len()..(cell + 1) * w.len()];
        Some(a[lo] as f64 * (1.0 - t) + a[hi] as f64 * t)
    }

    /// Solar/SRF-weighted surface albedo. This is NOT a TOA reflectance factor.
    pub fn band_albedo(&self, cell: usize, band: AbiReflectiveBand) -> Option<f64> {
        let response = AbiSolarResponse::for_band(band);
        let mut sum = 0.0;
        for node in response.nodes() {
            sum += self.sample(cell, node.wavelength_um)? * node.solar_response_weight_w_m2;
        }
        Some(sum / response.solar_response_integral_1au_w_m2())
    }

    /// Integrate every surface spectrum with the official SRF and measured solar
    /// spectrum. Precollapsed interpolation weights avoid resampling every cell
    /// independently at thousands of response knots. Water remains NaN.
    pub fn band_albedo_grid(&self, band: AbiReflectiveBand) -> Result<Vec<f32>, String> {
        let response = AbiSolarResponse::for_band(band);
        let w = &self.header.wavelength_um;
        let mut weights = vec![0.0f64; w.len()];
        for node in response.nodes() {
            let x = node.wavelength_um;
            if x < w[0] || x > w[w.len() - 1] {
                return Err(
                    "surface wavelength coverage is narrower than the band response".into(),
                );
            }
            let hi = w.partition_point(|v| *v < x).clamp(1, w.len() - 1);
            let lo = hi - 1;
            let f = (x - w[lo]) / (w[hi] - w[lo]);
            let q = node.solar_response_weight_w_m2 / response.solar_response_integral_1au_w_m2();
            weights[lo] += (1.0 - f) * q;
            weights[hi] += f * q;
        }
        Ok(self
            .albedo
            .chunks_exact(w.len())
            .zip(&self.valid)
            .map(|(a, valid)| {
                if *valid == 0 {
                    f32::NAN
                } else {
                    a.iter()
                        .zip(&weights)
                        .map(|(v, weight)| *v as f64 * weight)
                        .sum::<f64>() as f32
                }
            })
            .collect())
    }

    /// Match each supplied coordinate to one model cell, then resolve a per-output
    /// float texture. A one-to-one assignment and identical land masks are required.
    /// Alpha is +1 for spectral land, -1 for model water, 0 outside the model.
    pub fn raster_rgba(
        &self,
        nx: usize,
        ny: usize,
        model_land: &[f32],
        raster: &SurfaceRaster,
        forward: impl Fn(f64, f64) -> (f64, f64),
    ) -> Result<(Vec<f32>, f64), String> {
        let n = nx.checked_mul(ny).ok_or("model dimensions overflow")?;
        if n != self.cell_count()
            || model_land.len() != n
            || (nx, ny) != (self.header.nx, self.header.ny)
        {
            return Err("spectral surface and model grid dimensions differ".into());
        }
        let mut native = vec![[0.0f32; 4]; n];
        let mut seen = vec![false; n];
        let mut max_error = 0.0f64;
        for (cell, &[lat, lon]) in self.coordinates.iter().enumerate() {
            let (i, j) = forward(lat, lon);
            let (ii, jj) = (i.round(), j.round());
            let error = (i - ii).abs().max((j - jj).abs());
            if !i.is_finite()
                || !j.is_finite()
                || error > MAX_GRID_ERROR_CELLS
                || ii < 0.0
                || jj < 0.0
                || ii >= nx as f64
                || jj >= ny as f64
            {
                return Err(format!(
                    "spectral surface coordinate does not match a model cell: {cell}, ({i},{j})"
                ));
            }
            let idx = jj as usize * nx + ii as usize;
            if seen[idx]
                || !model_land[idx].is_finite()
                || (model_land[idx] >= 0.5) != self.is_land(cell)
            {
                return Err(format!(
                    "spectral surface grid assignment or land-mask mismatch at {cell}"
                ));
            }
            seen[idx] = true;
            max_error = max_error.max(error);
            if self.is_land(cell) {
                for (c, wavelength) in RGB_WAVELENGTHS_UM.into_iter().enumerate() {
                    native[idx][c] = self
                        .sample(cell, wavelength)
                        .ok_or("surface spectrum does not cover RGB wavelengths")?
                        as f32;
                }
                native[idx][3] = 1.0;
            } else {
                native[idx][3] = -1.0;
            }
        }
        let mut rgba = vec![0.0f32; raster.nx * raster.ny * 4];
        for (pixel, (&i, &j)) in raster.grid_i.iter().zip(&raster.grid_j).enumerate() {
            if i.is_finite() && j.is_finite() {
                let ii = (i as f64).round().clamp(0.0, (nx - 1) as f64) as usize;
                let jj = (j as f64).round().clamp(0.0, (ny - 1) as f64) as usize;
                rgba[pixel * 4..pixel * 4 + 4].copy_from_slice(&native[jj * nx + ii]);
            }
        }
        Ok((rgba, max_error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn interpolation_and_solar_weighting_preserve_constant_spectra_and_no_data() {
        let s = SpectralSurface {
            header: Header {
                schema_version: 1,
                quantity: String::new(),
                nx: 2,
                ny: 1,
                wavelength_um: vec![0.4, 0.7, 1.0],
                climatology_doy: 247,
                coordinate_layout: String::new(),
                albedo_layout: String::new(),
                files: BTreeMap::new(),
            },
            coordinates: vec![[0.0, 0.0], [0.0, 1.0]],
            albedo: vec![0.25, 0.25, 0.25, f32::NAN, f32::NAN, f32::NAN],
            valid: vec![1, 0],
            land: vec![1, 0],
        };
        let raster = SurfaceRaster {
            nx: 2,
            ny: 1,
            scan: crate::camera::ScanGrid {
                nx: 2,
                ny: 1,
                x_min: 0.0,
                y_max: 0.0,
                pitch_x: 1.0,
                pitch_y: 1.0,
            },
            lat: vec![0.0; 2],
            lon: vec![1.0, 0.0],
            grid_i: vec![1.0, 0.0],
            grid_j: vec![0.0; 2],
            model_scan: None,
            navigation_geometry: None,
        };
        let (pixels, error) = s
            .raster_rgba(2, 1, &[1.0, 0.0], &raster, |la, lo| (lo, la))
            .unwrap();
        assert_eq!(error, 0.0);
        assert_eq!(pixels, vec![0.0, 0.0, 0.0, -1.0, 0.25, 0.25, 0.25, 1.0]);
        assert!(
            s.raster_rgba(2, 1, &[1.0, 0.0], &raster, |la, lo| (lo + 0.1, la))
                .is_err()
        );
        assert!(
            s.raster_rgba(2, 1, &[0.0, 1.0], &raster, |la, lo| (lo, la))
                .is_err()
        );
        assert_eq!(s.sample(0, 0.55), Some(0.25));
        assert_eq!(s.sample(0, 0.39), None);
        assert_eq!(s.sample(1, 0.55), None);
        for band in AbiReflectiveBand::ALL {
            assert!((s.band_albedo(0, band).unwrap() - 0.25).abs() < 1e-14);
            assert!(s.band_albedo(1, band).is_none());
            let grid = s.band_albedo_grid(band).unwrap();
            assert_eq!(grid[0], 0.25);
            assert!(grid[1].is_nan());
        }
    }
}
