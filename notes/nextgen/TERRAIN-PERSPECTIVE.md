# Native-terrain perspective camera

Use `perspective-terrain=on` with `eye=lat,lon,alt_msl_m`, `look=lat,lon,alt_msl_m`, `fov=`, and `camsize=`. This is an explicit CPU full-composite option; legacy perspective and all satellite/topdown defaults remain unchanged. The API setting is `RenderParams::perspective_terrain`.

The eye must be above bilinearly interpolated native HGT. The height field is traced on the WRF sphere with sub-cell steps and near-surface refinement. Terrain clips the surface atmosphere integral and the cloud primary ray, and the cloud shadow is evaluated at the actual terrain hit. Outside the finite model footprint the fallback is sea level. Negative terrain is clamped to the atmosphere sphere. No additional terrain texture, structures, trees, or debris are generated.

The foreground atmosphere is integrated along the actual eye ray to the cloud optical centroid using the existing front-column composition. This is an approximation to coupled air/cloud transport; it does not resolve multiply scattered lateral illumination under a deep storm. The optical transport kernels retain their CPU/WGSL counterparts. Only the finite surface boundary and perspective composition are CPU-only; no GPU perspective implementation exists.

The `*.geometry.json` sidecar records the camera, terrain mode, opacity, exposure and interpretation. Exact source/binary hashes and experiment commands are recorded separately by the case scripts.

## Vertical convergence experiment

The development example `ingest_vertical INPUT NEW_CACHE DZ_M NZ` resamples the original native WRF fields with the existing ingestion equations. It leaves horizontal extent and optical assumptions intact. Render its emitted `run.json` using the usual CLI. The output directory must be new because the legacy cache key does not encode custom vertical geometry.

For the full 333 m 1974 grid, `DZ_M=50 NZ=396` retains the default 19,750 m highest sample and all 600x600 horizontal points. Ingest peaked near 3.55 GB (the default ingest memory budget is exceeded only by this explicit experiment). A 50 m grid does not supply sub-333 m horizontal information. The matched 22:29 camera comparison did not recover a distinct funnel, so finer sampling is not promoted as a general visual fix.

## Validation

Constant 300 m terrain agrees with the analytic sphere at a 302 m eye; a buried eye is rejected. A foreground ridge occludes a more distant elevated look point and converges under step halving. An opaque terrain boundary prevents clouds behind it from contributing. The default camera behavior remains covered by existing perspective tests. Full suite: 618 passed, 2 ignored; workspace including GUI builds; final CLI suite 19 passed.

The finite-path boundary follows the transmittance integral described in PBRT v4, Volume Scattering / Transmittance: https://pbr-book.org/4ed/Volume_Scattering/Transmittance . Clear-atmosphere coefficients and integration remain the existing sourced Hillaire implementation in atmosphere.rs. Numerical ray tolerances are documented in camera.rs; no meteorological case fitting is introduced.
