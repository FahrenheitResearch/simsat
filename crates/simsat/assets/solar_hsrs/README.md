# TSIS-1 HSRS solar weights for ABI FM4

These assets integrate measured solar illumination with the complete tabulated NOAA C01/C02/C03 responses. They contain no simulated or observed weather imagery. The source is the [LASP TSIS-1 HSRS dataset](https://lasp.colorado.edu/lisird/latis/dap/tsis1_hsrs), described by [Coddington et al. (2021)](https://doi.org/10.1029/2020GL091709). The endpoint content is pinned by its retrieval hash rather than assuming a version number from the current website.

`manifest.json` records the exact query, retrieval time, source/response/output hashes, units, band integrals and limitations. The retrieved regional CSV is 28,797,583 bytes with SHA-256 `ea6d0e219925a69607a485451ed2a8dcaf8b2f583b646b04a94122feef4d697f`. It covers 400-1000 nm in 600,001 samples, exactly 0.001 nm apart. The raw reference remains in task work storage; the compact derived weights are checked in.

For normalized spectral transfer `f(lambda) = L_lambda / E_lambda` in sr^-1, the assets provide positive weights `w_i = integral E_1au(lambda) R(lambda) phi_i(lambda) d_lambda`, where `phi_i` are linear interpolation functions at the exact NOAA SRF nodes. Thus `sum(w_i f_i)` integrates a piecewise-linear transfer with the measured Sun and full response. It does not sample or discard solar lines on the response grid. Every solar and response knot is retained in the preparation integral; two-point Gauss quadrature exactly integrates the cubic product of linear E, R and phi on each subinterval, up to floating precision. Source E in W m^-2 nm^-1 is integrated in nm, yielding weights in W m^-2. Callback wavelengths are in um.

| Band | Nodes | Integral E R (W m^-2) | Mean spectral E at 1 AU (W m^-2 um^-1) |
|---|---:|---:|---:|
| C01 | 2628 | 76.71185465 | 2041.94257167 |
| C02 | 3167 | 134.15676468 | 1622.11185241 |
| C03 | 1027 | 33.14784313 | 953.19415410 |

The fixed reference is not contemporaneous irradiance. Radiance uses an explicit inverse-square Earth-Sun distance correction. ABI reflectance factor is pi times the integrated normalized radiance divided by the solar integral, with no solar-zenith division. Matching operational CMI also requires auditing its irradiance/calibration convention; introducing HSRS does not prove agreement with operational kappa0.

The transfer itself must resolve atmospheric absorption structure. These weights are appropriate for a smooth, resolved transfer on the response nodes; they do not replace correlated-k or line-by-line gas integration. The source uncertainty column is retained in the original CSV but is not propagated into independent nodal error bars because covariance is unavailable. Unprovided response tails outside the NOAA table are not invented. Existing broad RGB must never be passed here as independent ABI spectra.

Reproduction (offline after retrieving the exact source query in the manifest):

```text
python -B scripts/simsat-prepare-solar-weights.py --solar-csv <retrieved-csv> --output-dir <new-output-directory>
python -B -m unittest discover -s scripts -p test_simsat_solar_weights.py -v
```

The preparation rejects an unreviewed source hash and refuses existing output files. Tests include analytic cubic integration, narrow-line energy conservation, grid subdivision, nm/um units, and invalid/partial spectral coverage.
