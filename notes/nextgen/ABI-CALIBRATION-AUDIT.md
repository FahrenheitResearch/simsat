# ABI visible radiometric convention: direct official metadata audit

The new normalized spectral transfer integrates measured TSIS-1 HSRS illumination through official NOAA FM4 response functions. The remaining question was whether its reflectance-factor convention matched the actual GOES-19 CMI reference product. The matching L1B C01/C02/C03 files provide an independent check of `esun`, Earth-Sun distance and `kappa0`.

For the 2026-09-04 21:00 full-disk scan (start 21:00:20.634112 UTC, matching the CMI scan), the official L1B metadata give:

| Band | Official esun (W m-2 um-1 at mean distance) | HSRS/SRF result | Relative difference |
|---|---:|---:|---:|
| C01 | 2041.962036 | 2041.942572 | -0.000953% |
| C02 | 1622.087891 | 1622.111852 | +0.001477% |
| C03 | 953.177124 | 953.194154 | +0.001787% |

The independent band integrals agree within 0.002%. No empirical correction was applied to obtain that agreement. The exact numerical source objects and scalar attributes are preserved in `official-l1b-calibration-21z.json`; computed comparison values are in `solar-calibration-comparison.json`.

At the reported distance ratio 1.008389949798584 AU, the reported L1B `kappa0` values are 0.0015644000377506018, 0.001969400094822049 and 0.0033515000250190496, respectively. Their relative residual against `pi*d^2/esun` is -2.627e-5, +3.331e-6 and +1.370e-5. These are small coefficient-consistency residuals; their cause is not inferred from the metadata alone. If the exact reported operational coefficient is applied to our band radiance, the reflectance ratio to our natural HSRS-normalized reflectance is 0.999964199, 1.000018104 and 1.000031563. A future exact-product conversion can consume the reported coefficient explicitly; no hidden tuning is warranted.

The source CMI variables identify their quantity as Lambertian-equivalent TOA albedo multiplied by the solar-zenith cosine. Thus the comparator is reflectance factor `rho_f`, not BRF with an extra division by mu0. Raw values and observations must not receive display exposure, land toes, synthetic green, or clipping before per-band scores. The current display-intent cloud lighting comparisons remain separate gray-composite diagnostics.

Spatial metadata also matters: these MCMIPF reference bands are distributed on the common 56-microradian grid. The product reports averaging from 28-microradian sampling for C01/C03 and 14-microradian sampling for C02. A future instrument-footprint comparison must account for this existing product downsampling. It must not attribute all spatial smoothing to a new renderer filter or treat the common CMI grid as native C02 resolution.

## Provenance and bounded reads

Official objects were selected from the public [NOAA GOES-19 bucket](https://noaa-goes19.s3.amazonaws.com/index.html) with the exact scan-start prefix; full source URLs are in the evidence JSON. Their sizes total 514775843 bytes. Only 514851 bytes were read to obtain scalar metadata through HDF5. Every requested range required status 206, an exact Content-Range, the expected object length, and the listed ETag; requests used If-Match. Each returned range has its own recorded SHA-256. Full-file hashes were not computed and are explicitly null. No image radiance arrays were downloaded or used to adjust the model output.

The range reader is a scratch audit utility (`work/audit_abi_calibration.py` in the active task); it limits each source to 8 MiB and refuses unbounded reads. Scientific evidence is checked into the repository; remote image files are not. Original aligned GOES references are unchanged. Separate stricter per-band DQF references are complete at `work/abi-per-band-references/{12,15,18,21}z` in the active task. Their source objects were rehashed and read through scratch hard links; none was modified. Exact original target-grid hash and all jointly finite CMI values are unchanged. At 18Z, strict four-corner DQF support excludes 58 C01, 16 C02 and 274 C03 pixels; C13 and the other times retain all 180120 pixels. These separate references are ready for per-band scoring and do not alter the historical gray-composite baseline. See `abi-per-band-quality-audit.json`.

This audit confirms the normalization input and product semantics. It does not validate the cloud, surface, atmosphere, scattering closure, satellite-ray geometry or full-image accuracy of the unfinished per-band operator.
