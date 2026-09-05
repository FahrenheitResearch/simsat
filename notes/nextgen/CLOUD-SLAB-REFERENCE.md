# Independent cloud-slab reference and vertical-column correction

The experimental DeltaFlux closures expect the vertical optical thickness of a homogeneous column. The renderer supplied `max(vertical column, sample-to-sun slant depth, local voxel depth)` instead. That maximum is a support heuristic for the legacy octave method; it is not a vertical geometric coordinate. In an obliquely illuminated slab, it varies with depth and illumination even while the actual whole column remains unchanged.

The correction supplies the stored vertical whole-column optical depth, scaled by the selected intent, to all three experimental closures. Fractional vertical depth uses the same column. A missing/nonpositive vertical channel disables their higher-order contribution; direct single scattering still uses the explicitly marched sun path. Legacy octaves and their missing-column shadow-map fallback are unchanged. No default intent or opacity is changed.

This column approximation still cannot predict the full three-dimensional diffuse response to a neighboring cloud. The new regression isolates that limitation explicitly: a lateral occluder changes the direct beam but cannot silently redefine the local vertical slab. Subtracting a single-scatter march isolates the modeled higher-order term. The old geometry fails this assertion (DeltaFluxV1 source 5.272974317 versus 4.326823425); the corrected geometry passes for V1/V2/V3. Missing vertical metadata also leaves exactly the single-scatter result.

## Independent reference

The external reference is the official standalone [cdisort 2.1.3 distribution](https://libradtran.org/doku.php?id=download), retrieved from `https://www.libradtran.org/lib/exe/fetch.php?media=download%3Acdisort-2.1.3.tar.gz` on 2026-09-05. Archive SHA-256: `7d90ad5fefe155e62729505d5b24271a77adbd15d5800d85a290a94146314ad5` (144524 bytes). The vendor solver is downloaded separately, not embedded in the renderer. The checked-in `scripts/reference/cdisort_slab.c` supplies only benchmark inputs and extracts results.

Cases are a scalar, unpolarized, plane-parallel, conservative cloud slab above a black boundary. There is no atmospheric or external sky illumination. The angular phase is exactly the renderer's dual-HG liquid/ice mixture. Legendre coefficients are `beta_l = w*g1^l + (1-w)*g2^l`, with 512 moments. The incoming/outgoing scattering cosine is `-mu0*muv + sqrt(1-mu0^2)*sqrt(1-muv^2)*cos(relative azimuth)`. The incident normal irradiance is unity; reported reflectance factor is `pi*L/E`, without dividing by mu0. BRF is separately reported as `pi*L/(mu0*E)`.

The 864 geometries span liquid and ice; vertical tau 0.03, 0.1, 0.3, 1, 3, 10, 30, 100; solar cosine 0.15, 0.3, cos(65 degrees), 0.65, cos(30 degrees), 0.98; viewing cosine 0.35, 0.7, 1; relative azimuth 0, 90, 180 degrees. No values are selected from an observed weather case. Only 324 geometries lie within the existing LUT's tau and solar-cosine bounds. Its source still retains the original oracle SSA=0.999; the reference and exact direct term use SSA=1. That difference is disclosed, not corrected with a fitted offset.

Every reference was evaluated at 32 and 68 streams. The original 64-stream run rejected a solar direction too close to a quadrature ordinate; following the solver's explicit diagnostic, the higher quadrature was changed to 68 for the entire set. Solar geometry was preserved. All 864 cases satisfy the declared reflectance-factor convergence tolerance `2e-5 + 1e-3*abs(reference_68)`. Maximum conservative flux residual at 68 streams is 1.294e-9. This is a convergence check at those two quadratures, not an exact-error guarantee.

The original distribution's `disotest` was also run. Its multi-layer absorption-cutoff comparison 12b reports discrepancies in deep, faint fields; the source deliberately discards radiation below absorption optical depth 10. The extra two-stream-versus-four-stream comparison 14b also reports differences. Those are retained in the work logs; the complete vendor test suite is not described as a clean pass. The current one-layer, conservative cloud benchmark has zero absorption optical depth and does not exercise that cutoff. Smoke checks also recover a near-vacuum Lambertian boundary and converge a Rayleigh slab.

## Source comparison

`cargo run --release -p simsat --example audit_cloud_slab --locked --offline -j 6 -- 512` emits the exact first-order term plus integrated source approximations. Higher-order integration uses two Gauss points per viewing optical-depth interval and truncates only beyond viewing optical depth 40 (transmittance 4.25e-18). Repeating at 1024 intervals changes each reported reflectance factor by at most 6.36e-6; the largest DeltaFlux change is 4.68e-6.

RMSE in reflectance factor against the 68-stream reference:

| Source approximation | Within LUT tau/mu domain (324) | Entire set (864) |
|---|---:|---:|
| Legacy octaves, pure Beer | 0.135604 | 0.179166 |
| Legacy octaves, powder | 0.257786 | 0.280390 |
| DeltaFlux V1, old geometry | 0.072165 | 0.120893 |
| DeltaFlux V1, vertical column | 0.072011 | 0.120801 |
| DeltaFlux V2, old geometry | 0.080658 | 0.122536 |
| DeltaFlux V2, vertical column | 0.080579 | 0.122643 |
| DeltaFlux V3, old geometry | 0.067699 | 0.116314 |
| DeltaFlux V3, vertical column | 0.067721 | 0.115989 |

The geometry correction is justified by the input contract, not a uniform improvement in these aggregate scores. MAE decreases in all three corrected variants, while RMSE changes are mixed. Even the corrected candidates retain substantial directional errors and clamp outside their small LUT domain. They are not promoted to default and are not a complete ABI radiometric operator.

## Reproduction

Obtain the pinned external archive, verify its hash, and unpack it into a scratch directory. Compile its `cdisort.c` and `locate.c` together with `scripts/reference/cdisort_slab.c`, using a C compiler supporting GNU11 variable-length arrays and statement expressions. The local Windows run used Clang with the MSVC build environment and flags `-O2 -std=gnu11 -D_CRT_SECURE_NO_WARNINGS`, plus the external source directory as an include path. No vendor source modifications were made.

Save audit JSON using stdout from the release example. Then run:

```text
python scripts/simsat-validate-cloud-slab.py --simulated cloud-slab.json --reference-exe /path/to/simsat-reference.exe --output-dir /new/output/directory
```

Or replay the checked-in CSV and its exact request transcript:

```text
python scripts/simsat-validate-cloud-slab.py --simulated cloud-slab.json --reference-csv notes/nextgen/cloud-slab-reference-32-68.csv --reference-input notes/nextgen/cloud-slab-reference-input.txt --output-dir /new/output/directory
```

The validator refuses an existing output directory, duplicate/missing/non-finite rows, or a replay transcript with different geometry/physics. Unconverged cases are explicitly excluded and cause a nonzero exit. Tests cover successful signed scoring and each corruption class. Full joined comparisons remain generated outputs; checked-in compact evidence is `cloud-slab-reference-summary.json`, the reference CSV and request transcript. The actual live driver reproduced all 1728 rows successfully.

## Rendering and verification

698 workspace Rust tests pass, with 2 ignored; all 49 Python tests pass. Full workspace and GUI builds pass, as do formatting and strict Clippy for all targets. Jobs remain 6. The existing GPU preview planner explicitly substitutes legacy transport and reports an adjustment for the experimental modes; these modes do not yet have shader equivalents. The unchanged legacy shader is not a GPU implementation of the corrected DeltaFlux closure. Separate new spectral molecular/first-order kernels have their own previously verified WGSL twins.

Controlled images and the four-time gray-composite diagnostic scoreboard are complete with frozen release executable SHA-256 `2d951298debde059bbd101d0702860044d3f8891649bc1f236b8e2adf1de71fa`. See `cloud-column-image-scoreboard.json` and the user-facing `SimSat-four-time-cloud-lighting-comparison.png` / `SimSat-cloud-lighting-scoreboard.md` in Downloads. Default 21Z PNG/raw output is byte-identical. The corrected candidate is not promoted: both-cloudy MAE worsens at 12Z/18Z, all-valid MAE worsens at 12Z/18Z/21Z, and the 1974 22:29 anvil becomes flatter with contour-like structure. The 21Z absolute clear-bias gate passes (deterioration -0.000424 against tolerance 0.02), which is insufficient to claim overall improvement. Thermal rendering was unchanged and not rerun in this display-only experiment. These display-intent comparisons retain the existing 0.15 opacity and 1.5 exposure; they are not per-band physical operator validation. The 1974 22:29 solar elevation is outside the LUT's represented range and must be labeled as such. No merge or push.
