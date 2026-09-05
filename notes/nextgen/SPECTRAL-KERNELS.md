# Spectral physics increment: molecular air, measured Sun, exact first order

The active objective remains the most realistic overhead simulated satellite renderer we can build and validate. This increment implements scientific components for task A. It does not activate a new render intent or claim complete ABI C01/C02/C03 transport. Existing visible and IR products continue through their established code paths. No image opacity or display default changes in this increment; no merge or push.

## Components

`crates/simsat/src/spectral_molecular.rs` implements dry-air Rayleigh cross sections from [Bodhaine et al. (1999)](https://reef.atmos.colostate.edu/~odell/at721/resources/rayleighOpticalDepth.pdf), equations 5, 6, 18, 19, 22 and 23. All units are explicit. Wavelength support is 0.25-1.0 um; CO2 is a required caller input with a declared terrestrial contract of 0-1000 ppm. The refractive-index fit and standard number density both use 288.15 K and 101325 Pa. Tests match independently printed cross sections and King factors in Table 3 at 360 ppm, and integrate the depolarizing phase to unity. Dry-air density is explicit; water-vapour scattering, gas absorption, aerosols, polarization and Raman effects are excluded.

The normalized scalar phase follows the formula in [RTTOV v14 Science and Validation Report](https://nwp-saf.eumetsat.int/site/download/documentation/rtm/docs_rttov14/rttov14_svr.pdf), version 1.0.1, 31 January 2025, section 2.5 (page 34), divided by 4 pi to use sr^-1. RTTOV is a source for that equation, not an assertion that SimSat now matches RTTOV accuracy or capabilities. The float32 WGSL twin uses stable refractivity algebra and a scaled denominator to avoid cancellation and overflow.

`crates/simsat/src/visible_solar.rs` supplies measured solar-response quadrature from the official LASP TSIS-1 HSRS source and existing official NOAA FM4 responses. The high-resolution solar source is integrated conservatively, preserving all 0.001-nm samples and narrow solar lines. See `crates/simsat/assets/solar_hsrs/README.md` and its manifest. Normalized transfer L_lambda/E_lambda joins directly to raw band radiance and reflectance factor; the API does not use RGB constants, clipping, exposure or a mu0 divisor. Full atmospheric gas spectroscopy and operational CMI calibration-convention checks remain required.

`crates/simsat/src/spectral_transport.rs` integrates the direct single-scattering contribution analytically for homogeneous viewing-ray segments with linearly varying solar optical depth. The endpoint exponential integral remains stable when sun and view attenuation cancel, for thin optical depths, and for opaque reverse-sun paths. Entire shadow segments have no direct source. An optional direct Lambertian boundary is attenuated on both paths. Tests cover the exact scalar slab solution across optical depths 0 through 10000, varied solar/view cosines and segment subdivisions, pure absorption, shadow, a vacuum Lambertian full-band normalization, and invalid inputs. This is the first-order term, not a full cloud closure. Geometry/model mapping, multiple scattering, diffuse boundary illumination and finite-disk integration remain separate work.

Both molecular and transport kernels have separate WGSL source twins. They do not alter the legacy surface/cloud shader uniform layout. Two reusable offscreen audit examples execute these kernels on an actual adapter and fail explicitly if a GPU is unavailable; there is no CPU fallback masquerading as a GPU check. Their JSON results are checked in alongside this note.

## Controlled visual comparison

`SimSat-cloud-lighting-topdown-comparison.png` is saved in Downloads/SimSat-Satellite-Renders and task outputs. Both columns use identical inputs, view, sun, opacity and display transform; only the existing cloud closure changes. Rows are Central America 2026-09-04 21Z at 3 km and the independent 1974-04-03 21Z 333 m run. The 1974 image has no contemporaneous observed reference. The experimental closure retains its limited tabulated solar-angle, optical-depth and albedo ranges, including clamping outside them.

21Z gray RGB diagnostics against aligned ABI reflectance composite:

| Metric | Legacy octaves | delta-flux-v2 |
|---|---:|---:|
| Clear bias | -0.005305 | +0.005019 |
| Both-cloudy bias | -0.139465 | -0.101129 |
| Both-cloudy MAE | 0.201143 | 0.197631 |
| All-valid bias | -0.113550 | -0.097137 |
| All-valid MAE | 0.149656 | 0.151410 |
| All-valid correlation | 0.181979 | 0.194815 |

The result is mixed: cloudy bias improves, all-valid MAE slightly worsens, and synthetic-only cloud brightness/error increases. It is not promoted. These remain gray composite diagnostics, not per-band operator validation. No four-hour improvement is claimed for this one-hour experiment.

## Work remaining for task A

Connect the spectral kernels to native model optical state and actual satellite rays; introduce spectral cloud optics, gas absorption, spectral/directional land and ocean reflection, and a independently benchmarked multiple-scattering closure. Integrate the instrument response and footprint; produce C01/C02/C03 radiance and reflectance planes with per-band provenance and valid masks. Audit the reference product's calibration convention. Then run the full four-hour/holdout scoreboard and inspect output images. The already prepared HAMSTER surface is a boundary input, not observed TOA radiance. The active goal is not complete.

## Verification of this increment

697 workspace tests passed, 2 ignored (including 65 Studio tests). All 45 Python tests pass. The full workspace, including the unchanged GUI, builds. `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets --locked --offline -j 6 -- -D warnings` pass. A pre-existing ozone unit-test range loop was converted to an iterator for the current strict lint; no runtime atmosphere code changed. The two tests affected by final lint cleanup were rerun successfully.

On the NVIDIA RTX 3080/Vulkan adapter, 3020 molecular cases pass with maximum relative error 7.72e-7 across cross section, King factor, phase and optical depth. All 600 direct-transport cases pass the declared absolute/relative float32 tolerance. The greatest relative discrepancy for non-negligible transport outputs is 9.33e-7. These are numerical component checks, not complete-image CPU/GPU equivalence or an external scientific validation of the full renderer.
