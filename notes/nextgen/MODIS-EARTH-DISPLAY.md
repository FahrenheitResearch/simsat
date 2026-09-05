# Measured Earth colors and ground-lighting comparison

This opt-in path connects prepared 500 m MCD43A4.061 RGB data to the actual visible renderer. It is a **display-only Lambertian proxy of NBAR**, not a directional BRDF, black-sky albedo cube, or per-band ABI output. Clouds, terrain geometry, Sun position, atmosphere and the default renderer remain unchanged.

The first target is the 600 x 600, 333 m historical WRF frame at 1974-04-03 22:29. The source is explicitly 2024-04-03, a modern seasonal analogue, not a reconstruction of 1974 land cover. Measured MODIS bands 1/4/3 supply RGB without image gamma, JPEG enhancement, invented wavelengths, or the old artificial land saturation gain. Spectral mismatch with the broad Hillaire RGB atmosphere remains.

## Source and quality

The [MODIS science-team guide](https://www.umb.edu/spectralmass/modis-user-guide-v006-and-v0061/mcd43a4-nbar-product/) defines NBAR at nadir/local solar noon, using a moving 16-day window. [NASA's specification](https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/61/MCD43A4_c61.fs) defines QA 0 as full inversion, 1 as magnitude inversion and 255 as missing. [MODIS bands](https://modis.gsfc.nasa.gov/about/specifications.php) 1/4/3 span 620–670, 545–565 and 459–479 nm. The display path approximates these as its RGB surface response; it does not rename them ABI bands.

The explicit full-and-magnitude policy uses 188,861 all-full RGB land cells and 165,356 cells with magnitude retrievals. The remaining 4,367 land cells (1.22%) use the configured unshaded NASA base map. Full-only is also supported. Source values outside [0,1] are retained in the source binary but excluded from the Lambertian proxy and counted as fallback, never clipped. Separate QA, source/frame dates, provenance and hashes are preserved.

Model water retains the existing water path. Model SNOWH overlays measured land in linear reflectance with the existing approximate snow color. Base-map fallback retains the legacy base-map path. No land detail is generated.

## Reproduce

After the seven-band source preparer:

    python scripts/simsat-export-nbar-surface.py --source-dir prepared-nbar --output-dir NEW_ENGINE_INPUT --frame-date 1974-04-03 --quality full-and-magnitude --missing configured-base-map

Add to the native display command:

    nbar-surface=NEW_ENGINE_INPUT/surface.json bluemarble-base-map=BMNG_DIRECTORY output-transform=srgb

NBAR and spectral-surface are mutually exclusive. NBAR requires display intent, plain visible/RGB-reflectance products, an explicit seasonal base-map directory and original WRF input. Sensor intent is rejected. Raw RGB remains a gray display-intent diagnostic, not ABI C01/C02/C03.

The exporter requires an explicit frame date whose calendar month/day matches the source. The renderer verifies the full date. Fine-grid registration matches every target coordinate exactly to original WRF XLAT/XLONG, requiring one-to-one assignment and matching brick land mask. Cached bricks still supply clouds. A bare cached manifest or native GRIB cannot supply this original-WRF check and is currently unsupported for this display source.

The first attempt used the spectral input's ideal-projection tolerance and rejected a cell at 0.005259 cells. The audit found maximum stored-versus-ideal discrepancy 0.0063414014 cells (about 2.11 m). The fix verifies all 360,000 actual source-coordinate identities; it does not widen the tolerance. A shifted coordinate and changed land mask both fail. Existing spectral registration is unchanged.

## Images and limits

1. Current unshaded-map/sRGB display control.
2. Measured NBAR land with current lighting/cloud settings held fixed.
3. The same measured land with land solar-zenith normalization and dark-land toe disabled: an isolated comparison of these display brightening controls.
4. An explicitly labeled surface-only diagnostic with model terrain and actual Sun.

Exposure remains 1.5, gray cloud extinction scaling 0.15 and broad atmosphere/cloud transport remains. No default is promoted. These are native appearance comparisons; no GOES accuracy improvement is claimed for the historical scene.

## Verification

Final verification: 716 workspace tests pass, 3 intentionally ignored; strict all-target Clippy, formatting, GUI-inclusive build and the opt-in actual GPU surface test pass. The 2 Python exporter tests also pass. The favorite control is byte-identical to the prior image (SHA-256 90b07248a48175100e2a9f3de15e4531faee75117baf52cbb93e7dc5eba40473). The final check record accompanies finished images.

Rust/Python boundary checks cover band order, untouched source values, missing and magnitude QA, full-only selection, dates, hashes, truncation and unsupported sensor/product requests. The shared CPU/WGSL float contract uses alpha +1 for measured land, +2 for base-map fallback land, -1 for model water, 0 for legacy. Actual RTX 3080 execution verifies contradictory water masks are overridden, and +2 fallback/disabled inputs match the legacy gray-patch reference byte-for-byte. This checks the surface input, not whole-scene CPU/GPU equality.

An additional Central America public MODIS catalog lookup was rejected by automatic approval review because it would disclose the geographic bounds. Approval was requested; no request was sent. Prior four-time spectral-land scores concern another source and are not NBAR validation. Full ABI operator and original acceptance gates remain open.

The finished gallery is SimSat-Finer-Earth-2026-09-05 in Downloads, mirrored in the task outputs. It contains three native 22:29 displays, their before/after sheets, a surface-only diagnostic, and the unchanged daylight appearances for Central America and Texas. All four native renders completed successfully; commands and per-image binary hashes are recorded. No background render remains from this pass.
