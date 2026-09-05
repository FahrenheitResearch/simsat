//! Registration shared by measured RGB and spectral land inputs.
use crate::camera::SurfaceRaster;

/// Require a one-to-one assignment to native model cells; no guessed row order.
/// The 0.005-cell tolerance accommodates WRF f32 lat/lon rounding (~1.7 m at 333 m).
pub(crate) fn raster_rgba(
    shape: (usize, usize),
    coordinates: &[[f64; 2]],
    land: &[u8],
    colors: &[[f32; 4]],
    model_land: &[f32],
    raster: &SurfaceRaster,
    forward: impl Fn(f64, f64) -> (f64, f64),
) -> Result<(Vec<f32>, f64), String> {
    let (nx, ny) = shape;
    let n = nx
        .checked_mul(ny)
        .ok_or("surface model dimensions overflow")?;
    if n == 0
        || [
            coordinates.len(),
            land.len(),
            colors.len(),
            model_land.len(),
        ]
        .iter()
        .any(|&len| len != n)
    {
        return Err("surface and model grid dimensions differ".into());
    }
    let mut native = vec![[0.0f32; 4]; n];
    let mut seen = vec![false; n];
    let mut max_error = 0.0f64;
    for (cell, &[lat, lon]) in coordinates.iter().enumerate() {
        let (i, j) = forward(lat, lon);
        let (ii, jj) = (i.round(), j.round());
        let error = (i - ii).abs().max((j - jj).abs());
        if !i.is_finite()
            || !j.is_finite()
            || error > 0.005
            || ii < 0.0
            || jj < 0.0
            || ii >= nx as f64
            || jj >= ny as f64
        {
            return Err(format!(
                "surface coordinate does not match a model cell: {cell}, ({i},{j})"
            ));
        }
        let idx = jj as usize * nx + ii as usize;
        if seen[idx]
            || !model_land[idx].is_finite()
            || (model_land[idx] >= 0.5) != (land[cell] == 1)
        {
            return Err(format!(
                "surface grid assignment or land-mask mismatch at {cell}"
            ));
        }
        seen[idx] = true;
        max_error = max_error.max(error);
        native[idx] = colors[cell];
    }
    let pixels = raster
        .nx
        .checked_mul(raster.ny)
        .ok_or("surface raster dimensions overflow")?;
    if raster.grid_i.len() != pixels || raster.grid_j.len() != pixels {
        return Err("surface raster coordinate lengths differ".into());
    }
    let mut rgba = vec![
        0.0f32;
        pixels
            .checked_mul(4)
            .ok_or("surface raster size overflow")?
    ];
    for (pixel, (&i, &j)) in raster.grid_i.iter().zip(&raster.grid_j).enumerate() {
        if i.is_finite() && j.is_finite() {
            let ii = (i as f64).round().clamp(0.0, (nx - 1) as f64) as usize;
            let jj = (j as f64).round().clamp(0.0, (ny - 1) as f64) as usize;
            rgba[pixel * 4..pixel * 4 + 4].copy_from_slice(&native[jj * nx + ii]);
        }
    }
    Ok((rgba, max_error))
}
