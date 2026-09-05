//! MODIS NBAR RGB as an explicit Lambertian *display proxy*, not spectral albedo.
//! NBAR is evaluated at nadir/local solar noon. No arbitrary-angle BRDF correction
//! is claimed. RGB uses measured MODIS bands 1/4/3 (620-670, 545-565, 459-479 nm),
//! not the Hillaire representative wavelengths or ABI channels.
//! Sources: https://modis.gsfc.nasa.gov/about/specifications.php
//! https://www.umb.edu/spectralmass/modis-user-guide-v006-and-v0061/mcd43a4-nbar-product/
//! Quality 0 = full inversion, 1 = magnitude inversion, 255 = missing.
//! https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/61/MCD43A4_c61.fs
use crate::camera::SurfaceRaster;
use chrono::{Datelike, NaiveDate};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path};

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
    source_date: String,
    frame_date: String,
    rgb_bands: [u8; 3],
    quality_policy: String,
    missing_policy: String,
    coordinate_layout: String,
    reflectance_layout: String,
    files: BTreeMap<String, FileRecord>,
}
pub struct NbarSurface {
    header: Header,
    coordinates: Vec<[f64; 2]>,
    land: Vec<u8>,
    colors: Vec<[f32; 4]>,
    pub full_count: usize,
    pub magnitude_count: usize,
    pub fallback_count: usize,
}
impl NbarSurface {
    pub fn load(manifest: &Path) -> Result<Self, String> {
        const MAX: usize = 512 * 1024 * 1024;
        if std::fs::metadata(manifest)
            .map_err(|e| e.to_string())?
            .len()
            > 1024 * 1024
        {
            return Err("NBAR manifest exceeds 1 MiB".into());
        }
        let h: Header =
            serde_json::from_slice(&std::fs::read(manifest).map_err(|e| e.to_string())?)
                .map_err(|e| format!("NBAR manifest: {e}"))?;
        let n = h.nx.checked_mul(h.ny).ok_or("NBAR dimensions overflow")?;
        if n == 0
            || n > MAX / 16
            || h.schema_version != 1
            || h.quantity != "modis_nbar_rgb_lambertian_display_proxy"
            || h.rgb_bands != [1, 4, 3]
            || h.missing_policy != "configured-base-map"
            || !matches!(
                h.quality_policy.as_str(),
                "full-only" | "full-and-magnitude"
            )
            || h.coordinate_layout != "latitude_longitude_interleaved_f64le"
            || h.reflectance_layout != "row_column_rgb_f32le"
        {
            return Err("unsupported or invalid NBAR display contract".into());
        }
        let source = NaiveDate::parse_from_str(&h.source_date, "%Y-%m-%d")
            .map_err(|e| format!("NBAR source date: {e}"))?;
        let frame = NaiveDate::parse_from_str(&h.frame_date, "%Y-%m-%d")
            .map_err(|e| format!("NBAR frame date: {e}"))?;
        if (source.month(), source.day()) != (frame.month(), frame.day()) {
            return Err("NBAR seasonal analogue requires the same calendar month/day".into());
        }
        let directory = manifest.parent().ok_or("NBAR manifest needs a parent")?;
        let read = |name: &str, bytes: usize| -> Result<Vec<u8>, String> {
            let rec = h
                .files
                .get(name)
                .ok_or_else(|| format!("NBAR missing {name}"))?;
            let path = directory.join(name); // Fixed names, never a manifest-specified path.
            if rec.bytes != bytes
                || std::fs::metadata(&path).map_err(|e| e.to_string())?.len() != bytes as u64
            {
                return Err(format!("NBAR size mismatch: {name}"));
            }
            let data = std::fs::read(path).map_err(|e| e.to_string())?;
            if format!("{:x}", Sha256::digest(&data)) != rec.sha256 {
                return Err(format!("NBAR checksum mismatch: {name}"));
            }
            Ok(data)
        };
        let coordinates: Vec<[f64; 2]> = read("coordinates.bin", n * 16)?
            .chunks_exact(16)
            .map(|c| {
                [
                    f64::from_le_bytes(c[..8].try_into().unwrap()),
                    f64::from_le_bytes(c[8..].try_into().unwrap()),
                ]
            })
            .collect();
        let rgb: Vec<f32> = read("nbar-rgb.bin", n * 12)?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let qa = read("quality-rgb.bin", n * 3)?;
        let land = read("land-mask.bin", n)?;
        let mut colors = vec![[0.0; 4]; n];
        let (mut full_count, mut magnitude_count, mut fallback_count) = (0, 0, 0);
        for cell in 0..n {
            let [lat, lon] = coordinates[cell];
            if !lat.is_finite()
                || !lon.is_finite()
                || lat.abs() > 90.0
                || lon.abs() > 180.0
                || land[cell] > 1
                || !qa[cell * 3..cell * 3 + 3]
                    .iter()
                    .all(|q| matches!(q, 0 | 1 | 255))
            {
                return Err(format!(
                    "invalid NBAR coordinate or quality/land mask at {cell}"
                ));
            }
            if land[cell] == 0 {
                colors[cell][3] = -1.0;
                continue;
            }
            let values = &rgb[cell * 3..cell * 3 + 3];
            let q = &qa[cell * 3..cell * 3 + 3];
            let full = q.iter().all(|v| *v == 0);
            let accepted =
                full || (h.quality_policy == "full-and-magnitude" && q.iter().all(|v| *v <= 1));
            // Directional reflectance > 1 can be valid NBAR but not this Lambertian
            // albedo proxy. Preserve it in source bytes; explicitly fall back, never clip.
            if accepted
                && values
                    .iter()
                    .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
            {
                colors[cell][..3].copy_from_slice(values);
                colors[cell][3] = 1.0;
                if full {
                    full_count += 1;
                } else {
                    magnitude_count += 1;
                }
            } else {
                colors[cell][3] = 2.0; // Force model land, retain the configured base-map path.
                fallback_count += 1;
            }
        }
        if full_count + magnitude_count == 0 {
            return Err("NBAR has no usable land RGB".into());
        }
        Ok(Self {
            header: h,
            coordinates,
            land,
            colors,
            full_count,
            magnitude_count,
            fallback_count,
        })
    }
    pub fn validate_frame_date(&self, year: i32, month: u32, day: u32) -> Result<(), String> {
        let date = NaiveDate::from_ymd_opt(year, month, day).ok_or("invalid frame date")?;
        if date.to_string() != self.header.frame_date {
            return Err(format!(
                "NBAR manifest frame date {} differs from render date {date}",
                self.header.frame_date
            ));
        }
        Ok(())
    }
    pub fn source_date(&self) -> &str {
        &self.header.source_date
    }
    pub fn quality_policy(&self) -> &str {
        &self.header.quality_policy
    }
    /// Register against the actual source XLAT/XLONG, not a guessed image row
    /// order or a tolerance expanded until an idealized projection passes.
    /// The table must cover exactly the same native WRF coordinates and land mask.
    /// WRF f32 projection/coordinate rounding is reported separately in cell units.
    pub fn raster_rgba(
        &self,
        geometry: &crate::ingest::GridGeometry,
        model_land: &[f32],
        raster: &SurfaceRaster,
    ) -> Result<(Vec<f32>, f64), String> {
        let (nx, ny) = (geometry.nx, geometry.ny);
        if (nx, ny) != (self.header.nx, self.header.ny) {
            return Err("NBAR and model dimensions differ".into());
        }
        let mut lookup = BTreeMap::new();
        for (index, (&lat, &lon)) in geometry.xlat.iter().zip(&geometry.xlong).enumerate() {
            if !lat.is_finite()
                || !lon.is_finite()
                || lookup
                    .insert(
                        ((lat as f64).to_bits(), (lon as f64).to_bits()),
                        ((index % nx) as f64, (index / nx) as f64),
                    )
                    .is_some()
            {
                return Err("invalid or duplicate source WRF coordinate".into());
            }
        }
        if lookup.len() != nx * ny {
            return Err("incomplete source WRF coordinates".into());
        }
        let projected = geometry.georef().map_err(|e| e.to_string())?;
        let mut max_projection_error = 0.0f64;
        for &[lat, lon] in &self.coordinates {
            let &(i, j) = lookup
                .get(&(lat.to_bits(), lon.to_bits()))
                .ok_or("NBAR coordinate differs from the source WRF grid")?;
            let (pi, pj) = projected.forward(lat, lon);
            if !pi.is_finite() || !pj.is_finite() {
                return Err("nonfinite source WRF projection".into());
            }
            max_projection_error = max_projection_error.max((pi - i).abs().max((pj - j).abs()));
        }
        let (rgba, _) = crate::surface_grid::raster_rgba(
            (nx, ny),
            &self.coordinates,
            &self.land,
            &self.colors,
            model_land,
            raster,
            |lat, lon| lookup[&(lat.to_bits(), lon.to_bits())],
        )?;
        Ok((rgba, max_projection_error))
    }
}

/// Overlay model snow on measured land in linear reflectance space. Fallback
/// land and water retain their separately configured surface source.
pub fn apply_model_snow(
    rgba: &mut [f32],
    raster: &SurfaceRaster,
    brick: &crate::bricks::VolumeBrick,
) {
    let nx = brick.nx;
    let ny = brick.ny;
    // Model snow overlays the climatological base in linear reflectance space.
    if let Some(snowh) = &brick.snowh {
        for (pixel, texel) in rgba.chunks_exact_mut(4).enumerate() {
            if texel[3] > 0.5 && texel[3] < 1.5 {
                let i = (raster.grid_i[pixel] as f64)
                    .round()
                    .clamp(0.0, (nx - 1) as f64) as usize;
                let j = (raster.grid_j[pixel] as f64)
                    .round()
                    .clamp(0.0, (ny - 1) as f64) as usize;
                let snow = crate::render::snow_fraction(snowh[j * nx + i] as f64);
                for (value, snow_srgb) in texel[..3].iter_mut().zip(crate::render::SNOW_ALBEDO_SRGB)
                {
                    *value =
                        *value * (1.0 - snow) + crate::render::srgb_to_linear(snow_srgb) * snow;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn nbar_contract_preserves_rgb_quality_dates_and_rejects_corrupt_input() {
        let dir = std::env::temp_dir().join(format!(
            "simsat-nbar-contract-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        let coords: Vec<u8> = (0..6)
            .flat_map(|i| [0.0f64, i as f64])
            .flat_map(f64::to_le_bytes)
            .collect();
        let values = [
            0.12f32,
            0.16,
            0.05,
            0.08,
            0.12,
            0.05,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            1.2,
            0.2,
            0.1,
            0.2,
            0.2,
            0.2,
        ];
        let files = [
            ("coordinates.bin", coords),
            (
                "nbar-rgb.bin",
                values.into_iter().flat_map(f32::to_le_bytes).collect(),
            ),
            (
                "quality-rgb.bin",
                vec![0, 0, 0, 1, 0, 1, 0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 0, 0],
            ),
            ("land-mask.bin", vec![1, 1, 1, 0, 1, 1]),
        ];
        let mut records = serde_json::Map::new();
        for (name, bytes) in &files {
            std::fs::write(dir.join(name), bytes).unwrap();
            records.insert(
                (*name).into(),
                json!({"bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(bytes))}),
            );
        }
        let mut h = json!({"schema_version":1,"quantity":"modis_nbar_rgb_lambertian_display_proxy",
            "nx":6,"ny":1,"source_date":"2024-04-03","frame_date":"1974-04-03",
            "rgb_bands":[1,4,3],"quality_policy":"full-and-magnitude",
            "missing_policy":"configured-base-map",
            "coordinate_layout":"latitude_longitude_interleaved_f64le",
            "reflectance_layout":"row_column_rgb_f32le","files":records});
        let manifest = dir.join("surface.json");
        let put = |h: &serde_json::Value| {
            std::fs::write(&manifest, serde_json::to_vec(h).unwrap()).unwrap()
        };
        put(&h);
        let source = NbarSurface::load(&manifest).unwrap();
        assert_eq!(
            (
                source.full_count,
                source.magnitude_count,
                source.fallback_count
            ),
            (1, 1, 3)
        );
        assert_eq!(source.colors[0], [0.12, 0.16, 0.05, 1.0]);
        assert_eq!(
            source.colors.iter().map(|c| c[3]).collect::<Vec<_>>(),
            [1.0, 1.0, 2.0, -1.0, 2.0, 2.0]
        );
        assert!(source.validate_frame_date(1974, 4, 3).is_ok());
        assert!(source.validate_frame_date(2024, 4, 3).is_err());
        assert!(source.validate_frame_date(1974, 4, 4).is_err());
        h["quality_policy"] = json!("full-only");
        put(&h);
        let strict = NbarSurface::load(&manifest).unwrap();
        assert_eq!(
            (
                strict.full_count,
                strict.magnitude_count,
                strict.fallback_count
            ),
            (1, 0, 4)
        );
        h["frame_date"] = json!("1974-04-04");
        put(&h);
        assert!(
            NbarSurface::load(&manifest)
                .err()
                .unwrap()
                .contains("calendar")
        );
        h["frame_date"] = json!("1974-04-03");
        put(&h);
        let path = dir.join("nbar-rgb.bin");
        let original = std::fs::read(&path).unwrap();
        let mut corrupt = original.clone();
        corrupt[0] ^= 1;
        std::fs::write(&path, &corrupt).unwrap();
        assert!(
            NbarSurface::load(&manifest)
                .err()
                .unwrap()
                .contains("checksum")
        );
        std::fs::write(&path, &original[..original.len() - 1]).unwrap();
        assert!(NbarSurface::load(&manifest).err().unwrap().contains("size"));
        std::fs::write(&path, &original).unwrap();
        h["rgb_bands"] = json!([1, 2, 3]);
        put(&h);
        assert!(NbarSurface::load(&manifest).is_err());
        for (name, _) in files {
            std::fs::remove_file(dir.join(name)).unwrap();
        }
        std::fs::remove_file(manifest).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }
}
