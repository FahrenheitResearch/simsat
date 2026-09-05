# All-order scalar ABI lighting reference

The new `simsat-render-abi-monte-carlo` command follows repeated scattering and ground/sky reflections through the actual unscaled 3D model cloud volume. It addresses the missing diffuse light in the committed first-order reference. This is an experimental CPU scene, separate from the shipped display, IR and GUI. It does not yet constitute the complete ABI operator or a promoted appearance change.

## Transport and sources

The backward estimator samples exponential optical free paths, estimates the point-Sun contribution at every interaction, and samples the normalized scattering phase or physical surface reflection to continue the path. The implementation follows the transport construction in [Mayer (2009)](https://elib.dlr.de/59646/1/2009-Mayer_A9RA3_tmp.pdf). The phase mixture combines the existing depolarizing Rayleigh distribution with the normalized liquid/ice dual-HG functions. Direction convention is photon propagation, so positive HG asymmetry remains forward scattering. The [PBRT phase-function discussion](https://pbr-book.org/4ed/Volume_Scattering/Phase_Functions) uses two directions pointing away from an interaction; its sign convention must not be copied blindly.

The Rayleigh sampler uses its exact mixture of a uniform cosine and a cosine-squared density. HG uses an analytic inverse CDF, with an algebraically equivalent stable expression near g=0. The PCG-based 32-bit streams are deterministic per pixel/path. Their 23-bit open-interval uniforms match WGSL exactly. Wavelength is sampled from all official NOAA FM4 / TSIS-1 HSRS solar-response quadrature nodes; transfer is not interpolated onto a reduced wavelength grid. Three bands use common path seeds with different wavelength distributions, preserving each marginal estimator while correlating color noise.

Lambertian land uses the complete HAMSTER albedo spectrum and cosine-weighted reflection. Water samples Gaussian Cox-Munk facets and the corresponding BRDF*cos/PDF weight. Below-horizon facets contribute zero and are never resampled. The inherited kernel returns pi times BRDF; direct collimated illumination includes one incidence cosine. GPU facet directions use the equivalent slope-based sine to avoid single-precision cancellation near a flat facet. This change corrected a measured GPU audit failure without relaxing tolerances.

Russian roulette applies only to attenuated path weights below an explicit threshold. Conservative unit-weight paths are not killed and reweighted by a fixed survival cap, which can produce pathological variance in thick conservative clouds. An event or segment safety failure aborts the render; it never contributes a silently truncated radiance. The high-detail test initially hit 100000 events on one path and saved no partial image. Increasing that explicit safety budget is a computational change, not an opacity or brightness adjustment. Failures now report pixel, band and path index.

## Actual model scene and limitations

`abi_monte_carlo.rs` supplies the same core path loop with actual spherical rays, decoded volume samples, occupancy, satellite geometry, source-grid registration and spectral surface materials. Free paths use finite-step midpoint extinction, with both start and midpoint optical-depth checks to resolve entry into denser cloud. Solar optical depth uses the established finite-step cloud integral and spherical molecular column. The numerical settings are explicit and must be assessed separately from photon noise.

The atmosphere remains an exponential dry-air reference with P=101325 Pa, T=288.15 K, H=8000 m and CO2=360 ppm. These are declared reference values, not contemporary CO2 or native WRF molecular profiles. Clouds remain gray, conservative, full-cell and use inherited dual-HG phases. Spectral particle modules are not connected. Gases, aerosols, finite solar disk, polarization, fractional overlap and instrument PSF remain absent.

Land remains climatological black-sky spectral albedo approximated as Lambertian reflection, not directional BRDF. The measured MODIS NBAR appearance source is separate and is not accepted as a full spectral ABI surface. Water still lacks shadowing/masking, whitecaps and water-leaving radiance. Each primary pixel uses its target-height sphere; full terrain intersections and terrain shadows are not implemented. Exterior clouds are zero, and surface outside measured coverage absorbs. Both are incomplete boundary conditions, disclosed in flags and per-path fractions rather than hidden with edge colors.

## Reproduction and output

Build with six jobs:

    cargo build --release --bin simsat-render-abi-monte-carlo -j 6

Run with explicit paths and a new output directory:

    simsat-render-abi-monte-carlo input=WRF_FILE cache=CACHE_DIRECTORY spectral-surface=SURFACE_JSON config=CONFIG_JSON out-dir=NEW_DIRECTORY sat=goes-east threads=6

Example reference configuration (sample spacing and photon count are experimental controls, not native/full-quality defaults):

```json
{
  "dry_pressure_pa": 101325.0,
  "reference_temperature_k": 288.15,
  "dry_scale_height_m": 8000.0,
  "co2_ppm": 360.0,
  "view_step_m": 125.0,
  "sun_step_m": 125.0,
  "air_step_m": 1000.0,
  "air_column_intervals": 128,
  "collision_step_optical_depth": 0.25,
  "sample_stride": 8,
  "samples_per_band": 256,
  "seed": 198271,
  "path": {
    "roulette_start_order": 16,
    "roulette_weight_threshold": 0.95,
    "event_safety_limit": 1000000
  }
}
```

Raw planes are north-first little-endian f32 on the original model grid. `rho-c01-c02-c03.bin` contains pi L/E, with matching first-order and higher-order planes. Standard-error planes accompany all three estimates; `mean-events.bin` reports average interaction counts. C01/C02/C03 scalar planes and clipped square-root grayscale previews are also saved. Stride>1 leaves explicit NaN holes in the raw full-grid arrays; the compact previews show only selected rays. No spatial denoising or invented small-scale detail is applied.

Support bits: 0 sampled; 1 any primary or secondary cloud path leaves the volume; 2 any direct Sun path leaves the volume; 3 any surface interaction leaves measured coverage. OR flags become more common as more photons are sampled. The separate per-band path/Sun/surface exterior fractions are the useful measure of frequency. These flags are not directly comparable to the primary-only first-order flags. `glint-mask.bin` retains the observation-independent primary-water wind-facet definition.

`render.json` records input, surface-manifest, binary and output hashes, geometry, time, configuration, runtime, and limitations. Independent photon empirical SEM includes path and wavelength sampling only. Rare events can be missed at low sample counts, producing deceptively tiny SEM. It is not a guaranteed confidence interval and does not include numerical, model, boundary or omitted-physics errors.

## Independent checks

The same CPU `trace` loop is exercised by the analytic homogeneous slab adapter and the actual model scene. The external reference is the independently compiled C-DISORT 2.1.3 executable, unchanged from the earlier cloud study. The comparison spans 88 cases: Rayleigh optical depths 0.02/0.2/1; liquid/ice optical depths 0.1/1/10/100; two solar cosines, two view/azimuth configurations, absorbing and conservative scattering, and black/reflecting ground. DISORT 68 versus 96 streams differ by at most 3.09e-8 in reflectance factor.

CPU: 250000 paths/case, all 88 within six empirical standard errors plus independent reference convergence and the declared numerical tolerance; largest absolute z=3.154. GPU slab: 65536 paths/case, all 88 pass the same criterion; largest absolute z=2.819. RTX 3080 Vulkan sampling checks cover 20480 phase cases and 24576 material cases; RNG uniforms are exact, maximum direction error is below 4.48e-7, and the configured phase/material tolerances pass. Full reports retain actual absolute errors and tolerances rather than presenting statistical agreement as exact equality.

Eight core tests cover independent analytic phase moments, hemispherical water quadrature, exact vacuum Lambertian normalization, RNG endpoints, conservative-path roulette, attenuated-weight expectation, safety/invalid-input rejection, and WGSL validation. The workspace has 731 passing tests and three intentionally ignored tests at this increment. Strict all-target Clippy, the GUI-inclusive workspace build and formatting checks pass. Full scene GPU traversal remains unimplemented: the verified WGSL sampling/material and all-order slab reference are not a full-image GPU counterpart.

## Interpretation and promotion

The same-sample GOES comparisons use official per-band validity/DQF, aligned C01/C02/C03 references, and the unchanged T3 masks. Supplemental Earth splits use original model LANDMASK and official/model both-clear masks. No color gain, exposure, opacity fitting, image relocation or case tuning is introduced. A model column labeled clear may still have a neighboring cloud on a slanted path.

At the initial 21Z 2880-ray, 256-path preview, both-clear land biases change from -0.028995 to -0.003891 (C01), -0.012926 to +0.002097 (C02), and -0.006832 to +0.023139 (C03). Blue/red improve; near infrared worsens. Both-cloudy mean biases are much closer to zero. Observed-clear areas with model cloud disagreements become too bright. This is mixed evidence, not an across-the-board success.

The before column is the unfinished first-order spectral reference at identical sample positions, not the v0.2.1 display. The original four-hour display/IR promotion gates remain open. Current measured-Earth appearance images are retained unchanged; the new scalar Monte Carlo operator is not promoted over them. The overall realism goal remains active, with spectral particles, native air/gases/aerosols, surface BRDF/water color, instrument footprint, and practical full-scene GPU work still outstanding.

## Four-time same-sample scoreboard

Each row uses 720 selected native model-grid rays and 64 independent paths per band, with the same input masks for both estimators. These are coarse regression scouts, not complete native frame gates. Values are reflectance-factor bias, simulated minus observed; every before/after pair is first-order -> repeated scattering.

| UTC | Strict clear C01 | C02 | C03 | Both cloudy C01 | C02 | C03 |
|---|---:|---:|---:|---:|---:|---:|
| 12Z | -0.007828 -> +0.003650 | -0.001959 -> +0.007715 | -0.003862 -> +0.005206 | -0.014414 -> -0.003246 | -0.011319 -> +0.000017 | -0.017463 -> -0.006713 |
| 15Z | -0.045065 -> +0.016240 | -0.016997 -> +0.031422 | -0.041950 -> +0.014814 | -0.204917 -> -0.010762 | -0.190231 -> -0.004809 | -0.232144 -> -0.012055 |
| 18Z | -0.052236 -> +0.024022 | -0.021750 -> +0.043462 | -0.048028 -> +0.027807 | -0.269750 -> +0.017996 | -0.254667 -> +0.038409 | -0.304227 -> +0.020577 |
| 21Z | -0.035939 -> +0.036864 | -0.012811 -> +0.052877 | -0.022185 -> +0.053161 | -0.240796 -> -0.003368 | -0.225966 -> +0.010011 | -0.256432 -> -0.004562 |

The 21Z detailed image has 45030 sampled rays (237 x 190, every second original model-grid point) and 64 paths/band. It completed in 987.13 seconds at six threads with the explicit one-million-event safety bound. It is visibly noisy and is not presented as a native full-grid product. The 720 shared rays match all ten scientific mean/error/support planes in the coarse scout exactly. A final-source 32-path scout preserves 20 original scientific/coordinate/mask/preview files byte-for-byte.

Halving view/Sun/air steps and collision-step optical depth, while doubling column quadrature intervals, changes the sampled mean by -0.001923/-0.000676/+0.000228 in C01/C02/C03. Individual path histories can diverge and sampled pixel differences reach 0.13961/0.20629/0.30138; the low-count refinement is not an all-pixel numerical bound or proof of convergence. It mixes numerical sensitivity with stochastic path changes. The complete numerical-check report preserves those limits.

The checked-in JSON scoreboard contains all regimes, MAE/RMSE, both-clear land/ocean splits, path uncertainty, source hashes and exterior-support fractions. It must not be reduced to the favorable both-cloudy mean bias. Observed-clear and near-infrared regressions prevent promotion. The original display and IR code paths were not changed.
