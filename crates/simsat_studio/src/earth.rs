//! Explicit display Earth inputs and portable local test cases.
use serde::{Deserialize, Serialize};
use simsat::{bluemarble, bricks::VolumeBrick, camera::SurfaceRaster};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EarthSettings {
    pub base_map: Option<PathBuf>,
    pub nbar_surface: Option<PathBuf>,
}

impl EarthSettings {
    pub fn resolve_paths(&mut self, directory: &Path) {
        for path in [&mut self.base_map, &mut self.nbar_surface]
            .into_iter()
            .flatten()
        {
            if path.is_relative() {
                *path = directory.join(&*path);
            }
        }
    }

    pub fn base_crop(
        &self,
        raster: &SurfaceRaster,
        blend: bluemarble::MonthBlend,
        max_dim: u32,
    ) -> Result<(bluemarble::BlueMarbleCrop, String), String> {
        let dir = self
            .base_map
            .as_ref()
            .ok_or("Select an Earth base-map folder")?;
        let (la0, la1, lo0, lo1) = raster.lat_lon_bbox().ok_or("No Earth pixels")?;
        let load = |month| {
            let path = dir.join(bluemarble::base_map_month_file_2km(month));
            bluemarble::load_crop(&path, la0, la1, lo0, lo1, 1.0, max_dim)
                .map_err(|e| format!("Earth base map {}: {e}", path.display()))
        };
        let a = load(blend.month_a)?;
        let crop = if blend.is_single() {
            a
        } else {
            bluemarble::blend_crops(&a, &load(blend.month_b)?, blend.weight_b)
        };
        Ok((
            crop,
            format!(
                "NASA BMNG Base Map (without added terrain shading): {}",
                blend.label()
            ),
        ))
    }

    pub fn nbar_rgba(
        &self,
        input: Option<&(PathBuf, usize)>,
        date: (i32, u32, u32),
        brick: &VolumeBrick,
        raster: &SurfaceRaster,
    ) -> Result<(Option<Vec<f32>>, String), String> {
        let Some(path) = &self.nbar_surface else {
            return Ok((None, String::new()));
        };
        if self.base_map.is_none() {
            return Err(
                "Measured land requires an explicit Earth base map for missing pixels and water"
                    .into(),
            );
        }
        let (input, timestep) = input.ok_or(
            "Measured land requires the original WRF input; open the wrfout instead of run.json",
        )?;
        let surface = simsat::nbar_surface::NbarSurface::load(path)?;
        surface.validate_frame_date(date.0, date.1, date.2)?;
        let geometry = simsat::ingest::read_grid_geometry(input, *timestep)
            .map_err(|e| format!("Measured land WRF geometry: {e}"))?;
        let (mut rgba, error) = surface.raster_rgba(&geometry, &brick.landmask, raster)?;
        simsat::nbar_surface::apply_model_snow(&mut rgba, raster, brick);
        let line = format!(
            "MODIS NBAR display proxy: source {}; model {:04}-{:02}-{:02}; full={} magnitude={} fallback={}; exact WRF coordinates (projection rounding {error:.6} cells). Nadir/noon reflectance; no arbitrary-angle BRDF correction.",
            surface.source_date(),
            date.0,
            date.1,
            date.2,
            surface.full_count,
            surface.magnitude_count,
            surface.fallback_count
        );
        Ok((Some(rgba), line))
    }
}

pub fn measured_pixel(rgba: Option<&[f32]>, idx: usize) -> Option<[f32; 3]> {
    let p = rgba?.get(idx * 4..idx * 4 + 4)?;
    (p[3] > 0.5 && p[3] < 1.5).then(|| [p[0], p[1], p[2]])
}

/// Paths are relative to this JSON file, not the process working directory.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TestCase {
    pub name: String,
    pub input: PathBuf,
    #[serde(default)]
    pub cache: Option<PathBuf>,
    #[serde(default)]
    pub timestep: usize,
    #[serde(default)]
    pub settings: crate::settings::StudioSettings,
}
impl TestCase {
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes =
            std::fs::read(path).map_err(|e| format!("Test case {}: {e}", path.display()))?;
        if bytes.len() > 1024 * 1024 {
            return Err("Test case exceeds 1 MiB".into());
        }
        let mut case: Self =
            serde_json::from_slice(&bytes).map_err(|e| format!("Test case: {e}"))?;
        let dir = path.parent().unwrap_or(Path::new("."));
        if case.input.is_relative() {
            case.input = dir.join(case.input);
        }
        if let Some(cache) = &mut case.cache
            && cache.is_relative()
        {
            *cache = dir.join(&*cache);
        }
        case.settings.earth.resolve_paths(dir);
        case.settings.sanitize();
        if !case.input.is_file() {
            return Err(format!("Test case input missing: {}", case.input.display()));
        }
        Ok(case)
    }
}

/// Only named case manifests in the application's cases folder are listed.
/// Large model files are opened on demand, never decoded during discovery.
pub fn bundled_cases() -> Vec<(String, PathBuf)> {
    let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("cases")))
    else {
        return Vec::new();
    };
    let mut cases = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let path = e.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                let label = path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .replace('-', " ");
                cases.push((label, path));
            }
        }
    }
    cases.sort_by(|a, b| a.0.cmp(&b.0));
    cases
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn measured_rgb_excludes_water_and_missing_pixels() {
        let rgba = [0.1, 0.2, 0.3, 1.0, 0.8, 0.8, 0.8, 2.0, 0.0, 0.0, 0.0, -1.0];
        assert_eq!(measured_pixel(Some(&rgba), 0), Some([0.1, 0.2, 0.3]));
        for i in 1..4 {
            assert_eq!(measured_pixel(Some(&rgba), i), None);
        }
    }
    #[test]
    fn earth_paths_resolve_relative_to_case() {
        let mut earth = EarthSettings {
            base_map: Some("maps".into()),
            nbar_surface: Some("nbar/surface.json".into()),
        };
        earth.resolve_paths(Path::new("case-folder"));
        assert_eq!(earth.base_map, Some(PathBuf::from("case-folder/maps")));
        assert_eq!(
            earth.nbar_surface,
            Some(PathBuf::from("case-folder/nbar/surface.json"))
        );
    }
}
