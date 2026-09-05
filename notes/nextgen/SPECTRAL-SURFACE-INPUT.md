# Spectral surface input for the future ABI operator

`scripts/simsat-prepare-spectral-surface.py` prepares wavelength-resolved HAMSTER land albedo on an exact supplied model grid. It is an input-data stage only: it does not alter any renderer, form RGB, synthesize ABI bands, or establish radiometric acceptance.

Primary sources: [Roccetti et al. (2024)](https://amt.copernicus.org/articles/17/6025/2024/) and [HAMSTER data, CC BY 4.0](https://opendata.physik.lmu.de/04zd8-7et52/). The data combine MODIS climatology with laboratory spectral information. The original product has 0.05 degree spatial and 10 nm spectral sampling, 400-2500 nm. It is 2013-2022 climatology, not a contemporary or 1974 land reconstruction. Black-sky albedo is not a directional BRDF, nor a surface albedo at an arbitrary solar zenith. No calibrated TOA reflectance is claimed.

The preparer reads the real capitalized Latitude/Longitude/Wavelength coordinate variables (lowercase HDF dimension scales are unpopulated in the inspected release), crops before loading the hyperspectral data, preserves target row order, and retains the supplied wavelengths in micrometres. Bilinear interpolation requires four finite in-range nonzero source spectra. No wavelength extrapolation or no-data filling occurs. A mandatory binary target land_mask excludes water: inspection found positive ocean placeholder spectra in HAMSTER, so positivity cannot classify land. Coastal source cells may remain mixed; this product supplies no pixel retrieval QA in the inspected file.

Example, after obtaining a local HAMSTER file or coordinate-preserving subset:

```powershell
python -B scripts/simsat-prepare-spectral-surface.py --source path/to/DOY247.nc --grid path/to/model-grid-with-land-mask.npz --doy 247 --output-dir path/to/new-spectral-input
```

Grid NPZ keys: `lat`, `lon`, `land_mask` (0 water, 1 land), all matching 2D arrays. Source NetCDF variables: `Latitude`, `Longitude`, `Wavelength` in nm, `Black_Sky_Albedo` with dimensions latitude/longitude/wavelength. Output NPZ keys: `latitude`, `longitude`, `wavelength_um`, `black_sky_albedo`, `valid`, `land_mask`. Output albedo dimensions are row/column/wavelength. Invalid/water values are NaN with valid=0. The day index is explicitly 1..365; the caller must choose calendar/leap-day mapping. Existing output directories and conflicting declared source days are rejected. The manifest records input/output hashes and limitations.

## Measured regional input

For the original Central America case, retrieved the published DOY247 file using exact HTTP byte ranges: 13,792,530 bytes transferred from a 3,229,160,286-byte file. The preserved source subset is 240x320x61 (8-20 N, 97-81 W, 400-1000 nm). No full-file hash or immutable ETag/Last-Modified validator was available; the subset has its own SHA-256. This limitation is recorded, not represented as a verified full-source download.

Source subset SHA-256: `c09cce9619836b29db21fd0d6f5bc4f95ce6fd70c077fec0bcc004658af5f3c3`.

Target XLAT/XLONG/LANDMASK came from the 21Z native WRF frame, flipped north-first; coordinates are exactly equal to the trusted GOES reference grid. No observed reflectances or cloud masks enter preparation. Result: 380x474x61, all 69,469 model-land pixels have complete spectral input; all 110,651 model-water pixels are missing for this land-only stage. Aligned output SHA-256: `f5d99344a196bfd0f1616b8fc3470a90ea8f6fcb0c90a5ec50c84b146753eeff`.

Evidence is under the task's `work/hamster-central-america-doy247`, `work/hamster-land-target-grid.json`, and `work/hamster-central-america-land-aligned`. The earlier unmasked scratch output is superseded and must not enter transport.

Seven focused tests cover affine spectral interpolation, axis/row orientation, signed/nonfinite/no-data stencils, domain bounds, mandatory binary land masks, units and immutable output, irrelevant observation arrays, day mismatch, and unavailable coverage. All 40 Python tooling tests pass.

Next operator work still requires wavelength-dependent atmospheric/cloud transport, solar-spectrum weighting of the NOAA SRFs, directional land reflection treatment, the ocean BRDF, sensor sampling/footprint, and per-band GOES validation. Surface albedo must never be compared directly with CMI as though it were TOA reflectance.
