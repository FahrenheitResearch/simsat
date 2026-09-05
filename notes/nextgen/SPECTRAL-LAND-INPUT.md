# Spectral land input in the existing visible renderer

This opt-in input connects the prepared HAMSTER float spectra to the actual CPU visible renderer and both WGSL surface paths. It avoids treating an enhanced, 8-bit Earth appearance map as measured surface reflectance. The existing cloud transport, atmosphere, Sun geometry and display exposure remain in use. No default is promoted.

## Reproduce

Prepare the aligned source using `simsat-prepare-spectral-surface.py`, then export it:

    python scripts/simsat-export-spectral-surface.py --input spectral-surface-aligned.npz --provenance provenance.json --output-dir surface

Add to a plain visible or raw-RGB render:

    spectral-surface=surface/surface.json

The configured Blue Marble source still supplies water. For the comparison it is the unshaded seasonal NASA base map, with `output-transform=srgb`. Land receives floating-point albedo at 680/550/440 nm, the wavelengths of the current RGB atmospheric model. These are not the ABI C01/C02/C03 bands. The legacy land saturation multiplier is bypassed for this input; the remaining display controls still apply to display intent. Model snow overlays the base in linear reflectance space, using the existing snow color approximation.

The same resolved RGBA32F raster feeds CPU and GPU: alpha +1 means spectral land, -1 means model water, zero means outside the source domain. GPU classification follows this model mask when supplied. Disabled inputs use an all-zero dummy texture, preserving the old path. GUI construction supplies None; no GUI feature is added.

The reader checks schema, fixed filenames, sizes, SHA-256, wavelength order, finite bounded land spectra, coordinates, and matching land validity. It requires an exact one-to-one assignment to the model grid, with at most 0.005 cell registration error and identical model land masks. It rejects an incompatible non-leap climatology day and explicitly rejects February 29. Unsupported product paths reject the option. No spectral extrapolation or ocean fill is performed.

## Source interpretation

[Roccetti et al. (2024)](https://amt.copernicus.org/articles/17/6025/2024/) reconstruct HAMSTER spectra from MODIS and laboratory surface spectra at 0.05 degree and 10 nm spacing. This is a 2013-2022 daily climatology, not contemporary land state or a reconstruction of 1974. Black-sky albedo is a hemispheric quantity at a reference illumination geometry; using it as a Lambertian albedo here is an explicit approximation, not directional BRDF reconstruction. Source dataset: [10.57970/04zd8-7et52](https://doi.org/10.57970/04zd8-7et52), CC BY 4.0. Ocean and sub-grid landscape detail are not reconstructed. Any earlier 'not connected to a renderer' text retained inside the immutable upstream preparation provenance describes that preparation stage.

This is an integrated surface input, not the completed per-band TOA ABI operator. The current gray-RGB atmospheric/cloud transport and display-intent extinction scaling remain. Surface-band albedos below are not satellite imagery.

## Evidence

The Central America 21Z input has 474 x 380 cells and 61 wavelengths (0.4-1.0 micrometres). All 69,469 model land cells have valid spectra; 110,651 water cells remain unavailable. Maximum measured registration error is 0.000257 model cells (about 0.77 m at 3 km spacing).

A separate example exports solar/SRF-weighted SURFACE albedos using official ABI FM4 responses and the existing TSIS-1 HSRS weights:

    cargo run --release -j 6 -p simsat --example audit_spectral_surface -- surface/surface.json audit
    python scripts/simsat-verify-spectral-surface.py --manifest surface/surface.json --audit-dir audit --output check.json

The independent NumPy calculation interpolates each selected spectrum directly at every response node, instead of collapsing weights as the Rust implementation does. It checks 193 actual land spectra per band. Maximum absolute discrepancies are 1.39e-9, 2.78e-9, and 1.49e-8 for C01, C02, C03, below the 3e-8 float-output tolerance. Water remains NaN.

The unchanged 21Z raw gray-RGB diagnostic gives:

| Metric | Unshaded appearance map | Spectral land |
|---|---:|---:|
| Strict-clear bias | -0.001087416 | +0.001910799 |
| Strict-clear MAE | 0.048146112 | 0.045350799 |
| All-valid MAE | 0.145276288 | 0.143091878 |
| Both-cloudy MAE | 0.193863330 | 0.192836099 |

Clear MAE improves about 5.8%, while absolute mean bias increases slightly. This single-time, gray-RGB comparison does not establish bandwise accuracy or the four-time promotion gates. The source looks less saturated and coarser; it remains a scientific comparison, not a promoted visual default. No IR behavior is modified.

Validation: 714 workspace tests pass, 2 ignored; fmt, strict all-target Clippy, full workspace/GUI build, and release build pass with jobs=6. The registration test rejects displaced coordinates and land-mask changes. End-to-end export/load checks reject changed/truncated bytes, invalid day, out-of-range land albedo and missing land spectra. Actual cloud GPU preview renders on an RTX 3080; its documented terrain, cloud-step and fractional-cloud fallbacks remain, so whole-image CPU/GPU identity is not claimed. Final native default/pilot image verification and command/binary hashes accompany the image package.


A land/water check with coordinate arrays agreeing within 1e-6 degrees finds that, at 21Z, the 12,495 pixels clear in both the observation and model improve from MAE 0.03428167 to 0.02543951 over land (25.8%). Their mean bias improves from -0.03170201 to -0.02266532. The 30,022 observed-clear water pixels are numerically unchanged between the unshaded-map and spectral-land raw outputs. This helps isolate the surface change from forecast cloud placement; it remains an RGB diagnostic.

The final release renderer reproduces both the previous committed 21Z unshaded-map sRGB image and the spectral pilot image byte-for-byte. Final binary SHA-256: b9aac831a4b0fc42e385195596952f07b8eff6456387ac6448d2034b2ec28009. A separate CLI clouds-off GPU attempt is not counted as a surface-only test: the headless preview explicitly forces clouds on. The standalone surface WGSL is parsed and validated by the workspace tests; the real-data GPU execution evidence covers the cloud pass.

## Completed four-hour follow-up

All four current geostationary model-grid-point cases have now been rendered and scored. The reference and cloud-mask SHA-256 values are identical before and after at each hour. This table compares the accepted default topographic NASA appearance source with spectral land plus unshaded-map water. The separate 21Z controlled source comparison above holds the unshaded water source fixed.

| UTC | Clear bias before / after | Clear MAE before / after | All-valid MAE before / after | Both-cloudy MAE before / after |
|---|---:|---:|---:|---:|
| 12Z | +0.002281 / +0.002404 | 0.008934 / 0.008848 | 0.013030 / 0.012938 | 0.013610 / 0.013562 |
| 15Z | -0.026755 / -0.024197 | 0.043952 / 0.041427 | 0.122305 / 0.119506 | 0.167875 / 0.165288 |
| 18Z | -0.032228 / -0.028459 | 0.051774 / 0.048062 | 0.163240 / 0.159212 | 0.221744 / 0.218907 |
| 21Z | -0.001199 / +0.001911 | 0.048251 / 0.045351 | 0.145382 / 0.143092 | 0.193925 / 0.192836 |

Clear, all-valid and both-cloudy MAE improve at every hour. Absolute clear mean bias increases slightly at dawn and 21Z; the 21Z deterioration is 0.000711892, below 0.02 in this current gray-RGB harness. This is not independent ABI band validation or a rerun of the original nadir raster. Thermal code/defaults are unchanged; these runs do not add a new IR measurement. The source remains opt-in because spatial coarseness and the Lambertian/climatology assumptions are not resolved. No cloud parameters were tuned.

The expensive dawn display took 1177.1 seconds; display plus raw scoring took 2381.2 seconds. The 15Z and 18Z pairs took 199.7 and 146.1 seconds. Each used six threads and the same final release binary.

A follow-up direct SurfaceResources test now executes the standalone surface shader on the RTX 3080. Four distinct gray float inputs decoded with the CPU sRGB function match equivalent texture inputs with zero display-code difference. The explicit float mask correctly overrides a conflicting legacy water mask; all-zero/disabled float input is byte-identical to the old path. This verifies the new input contract, not whole-scene CPU/GPU radiative-transfer parity. The test is opt-in on machines without a GPU; run:

    cargo test --locked --offline -j 6 -p simsat surface_gpu_linear_albedo_matches_equivalent_srgb_texture -- --ignored --nocapture

The direct test and strict all-target Clippy pass. The prior 714-test workspace pass and full GUI build still apply to unchanged production code; this follow-up adds one GPU-dependent test (three tests are now ignored by default). Its runtime log accompanies this note.
