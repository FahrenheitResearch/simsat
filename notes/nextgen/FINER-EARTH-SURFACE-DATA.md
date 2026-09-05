# Finer Earth-surface input study

The 500 m MCD43A4.061 source is now prepared on the 600 x 600 grid of the favorite 1974 333 m frame. It is a 2024-04-03 seasonal analogue, not a reconstruction of 1974. It is NOT yet connected to the renderer, and it is not a replacement for the spectral black-sky albedo input.

The [NASA MCD43A4 file specification](https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/61/MCD43A4_c61.fs) defines seven band reflectances scaled by 0.0001, fill code 32767, and per-band mandatory QA: 0 full inversion, 1 magnitude inversion, 255 missing. The [MODIS science-team guide](https://www.umb.edu/spectralmass/modis-user-guide-v006-and-v0061/mcd43a4-nbar-product/) describes NBAR as reflectance normalized to nadir and local solar noon, based on a moving 16-day retrieval window. It does not supply arbitrary-angle BRDF coefficients.

`scripts/simsat-prepare-modis-nbar.py` takes an exact target grid, saved Planetary Computer STAC items, an explicit date, and a new output directory. Selects the daily center from the granule ID rather than an overlapping STAC interval; the inspected STAC datetime was April 6 while the requested granule center was April 3. Native source metadata confirms the March 26-April 10 input window. Duplicate tile versions select the latest production ID. Nearest source pixels retain the source detail without extrapolation or gap filling. Output retains all seven reflectances, QA, full-inversion masks, coordinates, and source-tile identity. Directional reflectances above 1 are retained, not silently clipped to albedo bounds.

Example:

    python scripts/simsat-prepare-modis-nbar.py --grid target-grid.npz --items saved-stac-items.json --date 2024-04-03 --output-dir modis-surface

The target NPZ requires `lat`, `lon`, and a binary `land_mask`. The STAC JSON contains `items` or `features`. `--resume` requires an identical grid/date/granule request and matching cached-window geometry. Source windows, metadata and unsigned source URLs are saved with SHA-256 values. Provider signing is cached by storage account/container, as in the official Planetary Computer SDK; public SAS tokens stay in memory. Scripts depend on NumPy, Rasterio, requests, and affine. This step does not require NASA login because these NBAR assets are mirrored in Planetary Computer.

## Actual coverage and decision

All 358,584 model land points lie within the two source tiles. Joint visible-band availability is 354,217 land points (98.78%); all three visible bands have full-inversion quality at 188,861 points (52.67%). The remaining valid points include magnitude retrievals; 4,367 land points have missing visible data. These categories are preserved separately. The source therefore needs a declared missing-data policy and appropriate angular treatment before it can become a complete render input. It is not legitimate to rename these seven NBAR bands as a hyperspectral black-sky albedo cube.

The parameterized preparer was rerun from public source assets. All reflectance, coordinate and quality arrays match the original preparation; tile indices match when normalized by granule identity. Negative checks reject fractional target masks and an absent exact center date even when a supplied STAC interval overlaps it.

The matching [MCD43A1 coefficients](https://www.umb.edu/spectralmass/modis-user-guide-v006-and-v0061/mcd43a1-brdfalbedo-model-parameters-product/) would permit directional evaluation. NASA's catalog locates both exact tiles, but a direct access probe returns HTTP 302 to Earthdata Login. No coefficient files were downloaded, and no arbitrary-view correction is claimed. Public NBAR data and their quality study remain available while that data route is unresolved. This does not block the overall satellite-operator work.

Source DOI: [10.5067/MODIS/MCD43A4.061](https://doi.org/10.5067/MODIS/MCD43A4.061). The surface/QA figure is an input diagnostic, not simulated satellite imagery; current native simulation comparisons remain in the Earth-color gallery.
