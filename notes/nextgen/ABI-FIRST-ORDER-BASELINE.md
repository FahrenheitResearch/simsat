# End-to-end first-order ABI lighting reference

The new `simsat-render-abi-first-order` CLI connects actual model-grid satellite rays, the cached unscaled 3D model cloud field, spectral land and molecular air to independent C01/C02/C03 reflectance factors. This provides a measured baseline for the Earth-lighting work. It is an explicitly incomplete CPU reference, not a finished ABI operator, display replacement or realism improvement. The existing visible/IR/GUI paths and defaults are unchanged.

## Calculation

For each native model-grid target, the reference traces the satellite viewing ray through the volume and the direct Sun ray from every segment endpoint. It uses the existing unscaled liquid/ice/precipitation extinction and normalized liquid/ice dual-HG phase. Cloud coverage is full-cell. There is no exposure, land toe, solar-zenith division, cloud opacity calibration or image clipping in the raw product.

Dry-air cross sections and depolarizing scalar Rayleigh phase come from the existing tested [Bodhaine/RTTOV implementation](SPECTRAL-KERNELS.md). This experiment explicitly supplies an exponential reference atmosphere: 101325 Pa dry surface pressure, 288.15 K reference temperature, 8000 m scale height and 360 ppm CO2. These are reference-atmosphere assumptions, not native WRF profiles or a claim about present-day CO2. No gas absorption or aerosols are included.

HAMSTER's complete 400-1000 nm surface spectra supply a Lambertian approximation to climatological black-sky land albedo. Source dates, provenance, coordinate assignment, complete land spectra and model land mask are checked. Water uses only the existing direct Cox-Munk reflection and model wind, with its existing refractive index of 1.34. The kernel is pi times BRDF; direct surface light includes one solar cosine. Its directional value can exceed one without being clipped.

The exact first-order segment integral combines view attenuation with linearly varying Sun optical depth. The source and attenuated direct surface terms remain separate. Spherical dry-air solar columns use Simpson integration; point-Sun Earth-shadow crossings are split down to at most 1 m. Rays that leave the model cloud domain are flagged, with exterior cloud explicitly zero.

Transfer is evaluated on a regular wavelength grid, interpolating computed L_lambda/E_lambda, and integrated through all of the official NOAA FM4 / measured TSIS-1 HSRS response nodes. It does not turn old RGB into ABI channels. Raw rho_f = pi L/E matches the [independently audited CMI convention](ABI-CALIBRATION-AUDIT.md); no additional mu0 divisor is applied. The current output is reflectance factor, not a physical-unit radiance plane.

## Reproduction and output contract

Build with six jobs:

    cargo build --release --bin simsat-render-abi-first-order -j 6

Run with explicit paths and a new output directory:

    simsat-render-abi-first-order input=WRF_FILE cache=CACHE_DIRECTORY spectral-surface=SURFACE_JSON config=CONFIG_JSON out-dir=NEW_DIRECTORY sat=goes-east threads=6

All configuration fields are required; unknown fields and unsupported ray steps are rejected. The baseline configuration is:

```json
{
  "dry_pressure_pa": 101325.0,
  "reference_temperature_k": 288.15,
  "dry_scale_height_m": 8000.0,
  "co2_ppm": 360.0,
  "spectral_step_um": 0.005,
  "view_step_m": 125.0,
  "sun_step_m": 125.0,
  "air_step_m": 1000.0,
  "air_column_intervals": 128,
  "sample_stride": 1
}
```

Raw output is north-first f32 little-endian. `rho-c01-c02-c03.bin`, `scatter-rho-c01-c02-c03.bin` and `surface-rho-c01-c02-c03.bin` have layout row/column/band. The three `c0N-rho.bin` files are scalar planes. `latitude.bin` and `longitude.bin` record the target coordinates. `support-flags.bin`: bit 0 sampled, bit 1 view outside cloud domain, bit 2 Sun outside cloud domain. `glint-mask.bin` records the model-water specular-facet core where tan(beta)^2 is no greater than the model-wind mean-square slope. This geometric mask is independent of observations and may contain cloudy pixels.

`render.json` records input, surface, binary and output hashes, frame time, geometry, all configuration, sampling counts, wavelength grid, runtime and missing physics. Input cache construction still uses the existing ingest and resolved brick contract. Each raw plane preserves values above one. Per-band PNGs clip to [0,1] then square-root for display only. With stride greater than one, full-grid raw planes contain explicit NaN holes and the preview shows only the selected samples; no coarse preview is presented as native full output.

The API wrapper requires the CPU/native/model-grid geostationary configuration, actual Sun, and spectral surface with matching nonleap climatology day. Its existing RenderParams intent is SensorFastGray to reject display-only surface inputs; all calculation is performed by this separate first-order function. No new GPU scene fallback is implied.

## Numerical checks and scientific limits

Seven targeted tests cover response integration against constant, affine and nonlinear analytic spectra; spherical molecular columns; mixed cloud/air analytic transfer; direct-surface normalization; invalid numerical input; and a full ray through an independently specified 10-km 3D slab including its day/night cases. The slab test exercises actual sphere intersections, georeferencing, occupancy, volume samples and both view/Sun paths, instead of supplying optical depth directly.

Native rays are checked against the identical sampled rays in stride-16 scouts. Spectral scouts compare 0.010 versus 0.005 micrometre spacing. Geometric scouts halve view/Sun/air steps and double molecular-column intervals. Their differences are numerical sensitivity estimates on 720 samples per case, not all-pixel error bounds or proof of physical accuracy.

The component transport and Rayleigh/HG expressions already have tested WGSL twins and actual adapter evidence, but the new scene orchestration is CPU-only. Whole-scene CPU/GPU lockstep remains required before this becomes a GPU product. The fixed display and IR kernels are untouched.

The leading omissions are atmospheric/cloud multiple scattering, diffuse illumination/reflection at the surface, spectral cloud particles, native molecular profiles, gas absorption, aerosols, directional land BRDF, water-leaving light/whitecaps, fractional cloud overlap, terrain casting, finite solar disk, polarization and instrument PSF. The current target-height sphere is not a 3D terrain intersection. The measured MODIS NBAR display source is separate and is not accepted as full spectral ABI land input.

## Evidence

The 12/15/18/21Z runs all completed at 474 x 380 (180120 native samples each). C01/C02/C03 are independently response-integrated and compared with the strict per-band GOES-19 references and unchanged original T3 masks. There is no prior complete independent-band operator to put in a legitimate before column; old RGB channels are not substituted for one. These scores establish an incomplete reference baseline, not a before/after improvement claim.

| UTC | Strict-clear bias C01 | C02 | C03 | Both-cloudy bias C01 | C02 | C03 |
|---|---:|---:|---:|---:|---:|---:|
| 12Z | -0.008014 | -0.002126 | -0.004175 | -0.016174 | -0.013232 | -0.020113 |
| 15Z | -0.044518 | -0.016654 | -0.038130 | -0.203500 | -0.188938 | -0.234425 |
| 18Z | -0.052608 | -0.023157 | -0.048188 | -0.269797 | -0.254832 | -0.302691 |
| 21Z | -0.035918 | -0.013818 | -0.023795 | -0.241102 | -0.226052 | -0.257383 |

All values are raw reflectance factor, simulated minus observed. Thick-cloud brightness is severely deficient, as expected for a first-order-only source. The lower dawn absolute error is not evidence that dawn physics is solved. The full JSON retains MAE/RMSE and every regime, including observed-only and synthetic-only clouds, cloud-temperature proxies and the geometry-derived glint mask.

| UTC | Both-clear model land bias C01 | C02 | C03 | Both-clear model ocean bias C01 | C02 | C03 |
|---|---:|---:|---:|---:|---:|---:|
| 12Z | -0.007086 | -0.001639 | -0.001916 | -0.008718 | -0.002815 | -0.004697 |
| 15Z | -0.035331 | -0.021846 | -0.051893 | -0.045259 | -0.016312 | -0.015033 |
| 18Z | -0.051114 | -0.038844 | -0.062672 | -0.046779 | -0.012196 | -0.012032 |
| 21Z | -0.029707 | -0.013995 | -0.005634 | -0.038851 | -0.015390 | -0.014939 |

These supplemental Earth-lighting splits use official ACM == 0 and the unchanged T3 vertical model cloud mask == 0, separated by original model LANDMASK. All source land-mask coordinates exactly equal the aligned reference coordinates. A vertical clear column can still have a neighboring cloud on its slanted view or Sun path; this is a diagnostic split, not a cloud-free atmosphere counterfactual.

| UTC | Wavelength-step max change C01/C02/C03 | Ray/column-step max change C01/C02/C03 | Native seconds | View / Sun exterior count |
|---|---|---|---:|---|
| 12Z | 0.0000029 / 0.0000018 / 0.0000005 | 0.0011181 / 0.0025823 / 0.0032988 | 1200.59 | 1581 / 22240 |
| 15Z | 0.0000161 / 0.0000048 / 0.0000009 | 0.0021576 / 0.0027354 / 0.0029321 | 98.62 | 1581 / 2927 |
| 18Z | 0.0000197 / 0.0000053 / 0.0000010 | 0.0008023 / 0.0009612 / 0.0010353 | 82.84 | 1581 / 1528 |
| 21Z | 0.0000112 / 0.0000033 / 0.0000006 | 0.0015858 / 0.0020289 / 0.0021806 | 114.36 | 1581 / 3796 |

All 720 native-versus-scout samples are byte-identical for the same numerical settings at every time. Numerical refinement is sampled, not a full-image error bound. The existing ideal projected raster differs from the original stored WRF coordinates by at most approximately 43.535 m (mean 0.667 m); no new image registration or score-fitted shift was applied.

Primary scores include flagged exterior support rather than silently dropping edge pixels. At 18Z the strict quality masks retain 180062 / 180104 / 179846 samples for C01/C02/C03; the other three hours retain 180120 each. Source/reference/mask/raw/binary hashes and full configuration are preserved.

The four completed native renders used executable SHA-256 `505fca49e1a24998bece706c1198f0a04354cb6fab26f50f2c315b9ddcc1b6f3`; that exact binary is archived locally with the run evidence. A final-source scout checks the subsequent lint-only cleanup against the saved native values.

The new gallery is `SimSat-Earth-Lighting-Checks-2026-09-05` in Downloads and the task outputs. Its current measured-MODIS appearance images are explicitly copied from the completed preceding Earth-color pass; its band-lighting sheets are new. The diagnostic does not replace Current Best.

Next required work is physical diffuse illumination/multiple scattering and the other missing terms, with independent reference checks. Do not rerender this unchanged baseline or apply an empirical land brightness correction to conceal its known omissions. Original display/IR promotion gates, native-profile/particle/PSF work, complete CPU/GPU scene equivalence and broader goal remain unfinished.

Final verification: 723 workspace tests pass, 3 intentionally ignored; strict all-target Clippy, formatting, GUI-inclusive workspace build and release build pass. The final-source release produces byte-identical scientific planes, support masks and previews on all 720 sampled rays per hour at all four times after the lint-only cleanup. Full details and binary hashes are in `abi-first-order-checks.json`.
