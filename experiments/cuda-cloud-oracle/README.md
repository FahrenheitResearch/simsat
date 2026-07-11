# CUDA cloud-slab oracle

This experiment is a deterministic scientific reference for visible directional reflectance from
a homogeneous, plane-parallel cloud slab. It exists to calibrate or replace the renderer's
non-energy-normalized octave/cloud-opacity heuristic. It is not linked by the Rust workspace, is
not production code, and does not make an NVIDIA GPU a product requirement.

## Physical problem

The legacy v1 problem has vertical extinction optical depth `tau`, single-scattering albedo
`ssa`, a normalized Henyey-Greenstein phase function with asymmetry `g`, and a black lower
boundary. Explicit Stage-2 options select result schema v2 and add a Lambertian lower boundary,
a normalized two-lobe HG mixture, matched forward flux, and 32-bin collision-source diagnostics.
The top is vacuum. Illumination is one collimated solar beam. The calculation is scalar,
monochromatic, and unpolarized.

The phase function integrates to one over solid angle:

```text
p(cos(theta)) = (1 - g^2) / [4 pi (1 + g^2 - 2 g cos(theta))^(3/2)].
```

The reported bidirectional reflectance factor is:

```text
BRF = pi I / (F0 mu0),  mu0 = cos(sun zenith),
```

where `F0` is irradiance normal to the incident beam. `relative_azimuth_deg` is explicitly the
angle between the incoming photon's horizontal propagation direction and the outgoing viewing
direction.

Each GPU thread traces one path backward from the sensor. At every collision, a direct-sun
next-event estimate is accumulated; a reciprocal HG direction is then sampled for the preceding
segment. Philox4x32-10 uses `(seed, path index, draw block)` as its counter/key identity, so paths
do not share mutable RNG state. GPU reductions and host aggregation use a fixed order. Identical
inputs repeat exactly on the tested build/device.

For Stage 2, the phase function and its first moment are

```text
p_mix = w p_HG(g1) + (1-w) p_HG(g2),
g_bar = w g1 + (1-w) g2.
```

The CLI checks a supplied `--phase-first-moment` against the lobes. Mixture sampling selects a
lobe with its exact mixture weight and then samples that normalized HG lobe. The `w=1`, black
surface identity branch consumes the same Philox draws and produces the same aggregates as v1.

At the lower boundary, backward paths add the directly illuminated Lambertian source and sample
the diffuse incident field with a cosine-weighted hemisphere. The forward analog calculation
uses the same atmosphere, phase mixture, albedo, seed, sample count, and scatter-order cap. It
reports `R` at the top, `T` as lower-boundary loss, `A` as volume absorption, and truncation. For
an opaque physical surface, `T` can equivalently be read as surface absorption. Their sum closes
to one by integer path accounting.

Each of 32 equal fractional optical-depth bins reports extinction-collision, scattering-source,
and absorption counts. `scattering_source_density` is the analog scattering-collision count per
incident path per unit fractional optical depth, integrated over angle. This is a depth-resolved
reference diagnostic; it is not a production directional closure.

The first-scattering-only analytic answer used by the self-test is:

```text
BRF_1 = pi ssa p(cos(theta)) / (mu0 + muv)
        * [1 - exp(-tau (1/mu0 + 1/muv))].
```

## Build and reproduce on Windows

Requirements: CUDA 13, a CUDA-capable GPU, and Visual Studio 2022 C++ Build Tools.

```powershell
Set-Location <repo-root>
& .\experiments\cuda-cloud-oracle\run_baseline.ps1
```

The script compiles for `sm_120`, runs the self-test, runs the fixed sweep, and prints SHA-256
hashes. Build products stay in ignored `build/`. Override `-Arch`, `-Samples`, `-BatchSamples`,
or `-CudaRoot` as needed. The baseline intentionally does not use `--use-fast-math`.

Direct examples:

```powershell
$Oracle = ".\experiments\cuda-cloud-oracle\build\slab_oracle.exe"
& $Oracle --tau 0.15 --ssa 0.999 --g 0.85 --samples 1048576
& $Oracle --backend cpu --tau 2 --ssa 0.995 --g 0.8 --format csv
& $Oracle --self-test
& $Oracle --tau 3 --ssa .999 --phase-model mixture-hg `
    --phase-lobe1-g .85 --phase-lobe2-g -.15 --phase-lobe1-weight .9 `
    --phase-first-moment .75 --surface-albedo .6 --lower-boundary lambertian `
    --report-forward-flux true --depth-bins 32
```

`--help` lists all controls. Single-case output can be JSON or CSV. Sweep output is CSV. Every
result includes the sample count, seed, sample variance, standard error, 95% Monte Carlo
confidence interval, path-order truncation fraction, wall time, total kernel time, and longest
batch time.

Legacy/default requests retain `simsat.cuda-cloud-oracle.result.v1` and the existing CSV shape.
Explicit Stage-2 options use JSON-only `simsat.cuda-cloud-oracle.result.v2`, capability set
`stage2-reference-v1`, and version every one of the four capabilities independently in the
`capabilities` object.

The exact 800-row request from commit `17a9e95` is run with:

```powershell
python .\experiments\cuda-cloud-oracle\run_stage2_grid.py --force
& .\experiments\cuda-cloud-oracle\run_stage2_validation.ps1
```

Timing-bearing per-case JSON stays in ignored `build/` directories. The runner validates all
inputs and output identities, strips only timing fields, writes the canonical RTX 5090 JSONL and
summary, and applies the predeclared 1024/2048-order tau-30 convergence acceptance. Add
`--repeat` when a full independent repeat is wanted. The validation script rebuilds the legacy
baseline and runs CUDA memcheck, racecheck, and initcheck; use `-RunGrid -RepeatGrid` to include
both complete grids.

## Stage-2 request result

The exact commit-`17a9e95` request was executed on the RTX 5090:

- 800/800 results passed schema, input, capability, finite-statistic, forward energy, and all
  32-bin accounting checks;
- 3,690,987,520 backward plus 3,690,987,520 matched forward paths were traced;
- forward `R + T + A + truncation` closed with zero reported error in every row;
- all 40 tau-30 order pairs passed the BRF-difference test; and
- 144 independently repeated grid rows were exact after timing removal.

The full request is nevertheless **blocked**, not accepted: 16/40 tau-30 pairs exceed the
predeclared `1e-4` truncated-fraction ceiling at 2048 orders. All failures are the bright
`surface_albedo=0.6` subset for the matched `g_bar=0.665/0.75` single/dual profiles. The maximum
is `0.0023890734`. The separate `stage2-remediation-request-v1.csv` changes only those 16 rows to
4096 orders. All 16 remediation rows pass: maximum truncation is `6.79493e-6` and maximum BRF
change from the immutable 2048-order result is `7.26461e-8`. The smallest blocker was therefore
the original request's 2048-order cap for those bright, thick states. No production closure is
promoted.

Canonical evidence is in `stage2-results-rtx5090-v1.jsonl`,
`stage2-results-rtx5090-v1-summary.json`, `stage2-remediation-results-rtx5090-v1.jsonl`,
`stage2-remediation-results-rtx5090-v1-summary.json`, and
`stage2-validation-rtx5090-v1.json`.

## Acceptance checks

The built-in suite covers:

- the published all-zero Philox4x32-10 vector;
- exact zero radiance at `tau=0` and `ssa=0` on CPU and GPU;
- CPU and GPU agreement with the analytic first-scattering result for an optically thin slab;
- CPU/GPU statistical agreement for multiple scattering;
- exact repeated GPU aggregates for an identical seed;
- finite, nonnegative directional output under a thick/grazing stress case; and
- an independent forward analog CPU calculation where reflected, transmitted, absorbed, and
  order-truncated photon fractions close to one;
- exact v1/v2 identity for a black surface and single HG;
- the zero-optical-depth Lambertian analytic result `BRF = surface_albedo`;
- a dual-HG first-scatter analytic result with a checked matched first moment;
- exact CPU/GPU Stage-2 R/T/A and per-bin collision identities; and
- exact repeated GPU R/T/A and all 32 depth-bin integer aggregates.

CUDA work is split into synchronized 65,536-path batches by default. The committed RTX 5090
baseline's longest batch is far below the Windows watchdog timescale.

## Limits that matter

- Single and dual HG are low-order phase-function proxies. The labels `liquid`, `ice`,
  `NSSL-like`, and `HRRR-like` are sensitivity probes, not microphysics-scheme retrievals.
- There is no 3-D cloud geometry, side escape, broken-cloud illumination, terrain, gas/aerosol
  atmosphere, spectral response, polarization, non-Lambertian surface, or camera response.
- Fixed `max_scatters` makes runtime bounded. `truncated_fraction` exposes the affected paths;
  the reported sampling error does not include residual order-truncation bias or model error.
- Directional BRF can legitimately exceed one for anisotropic scattering. Energy conservation
  is checked by hemispheric photon accounting, not by imposing `BRF <= 1`.
- Exact repeatability is promised for the recorded executable/device path, not bitwise across
  every compiler, CUDA version, CPU, or GPU architecture.

## Primary references

- L. G. Henyey and J. L. Greenstein, “Diffuse radiation in the Galaxy” (1941),
  [doi:10.1086/144246](https://doi.org/10.1086/144246).
- B. Mayer, “Radiative transfer in the cloudy atmosphere” (2009),
  [doi:10.1140/epjconf/e2009-00912-1](https://doi.org/10.1140/epjconf/e2009-00912-1),
  and the author's [MYSTIC publication page](https://www.bmayer.de/mystic.html).
- J. K. Salmon et al., “Parallel Random Numbers: As Easy as 1, 2, 3” (SC11),
  [author-hosted paper](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf).
- S. Chandrasekhar, “On the diffuse reflection of a pencil of radiation by a plane-parallel
  atmosphere” (1958), [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC528671/).
- NVIDIA, [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
  and [Windows TDR guidance](https://docs.nvidia.com/nsight-visual-studio-edition/2024.1/reference/index.html#timeout-detection-recovery-tdr).
