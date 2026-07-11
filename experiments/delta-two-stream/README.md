# Stage-0 delta-scaled two-stream slab experiment

This directory is an isolated cloud-radiative-transfer experiment. It does not
change the SimSat production CPU marcher, WGSL shaders, SSB cache, CLI, Python
binding, or Studio.

The prototype solves **hemispheric fluxes**, not a view-direction satellite
radiance. Its purpose is to provide a deterministic, energy-auditable baseline
for replacing the current unnormalised scatter-octave source. It is not yet a
production source term.

## Method

`delta_two_stream.py` applies the Joseph-Wiscombe-Weinman delta-Eddington
similarity transform

```text
f      = g^2
tau'   = (1 - omega*f) tau
omega' = (1 - f) omega / (1 - omega*f)
g'     = (g - f) / (1 - f)
```

and then solves the homogeneous plane-parallel slab with two Gaussian diffuse
ordinates at `mu = +/-1/sqrt(3)`. Optical depth increases downward. The incident
horizontal direct flux at the top is one. The lower boundary is Lambertian.

Each numerically tame sublayer is propagated exactly with a 2x2 matrix
exponential and an analytic exponentially attenuated direct-beam source. Layer
reflection, transmission, and source operators are combined with the stable
adding method. Splitting a thick slab into sublayers is therefore for numerical
conditioning, not an optical-depth discretisation approximation.

The direct-source partition uses the first phase moment after delta scaling.
This keeps the two stream sources nonnegative for the cloud-like `g >= 0` domain
and injects exactly `omega' * F_direct / mu0` diffuse flux per unit scaled optical
depth. In the conservative case, intrinsic diffuse `R + T = 1`.

## Run

Only Python's standard library is required.

```powershell
python experiments/delta-two-stream/delta_two_stream.py check
python experiments/delta-two-stream/delta_two_stream.py generate
python experiments/delta-two-stream/delta_two_stream.py all
```

`all` runs the checks before writing:

- `fixtures/stage0-slab-flux-v1.csv`
- `fixtures/stage0-slab-flux-v1.json`
- `fixtures/stage0-self-checks-v1.json`

No timestamp is embedded. The JSON is canonicalised, the row order is fixed,
and the output records the generator SHA-256.

## Fixture grid

- Optical depth: `0`, `1e-4`, `2e-4`, `1e-3`, `0.01`, `0.03`, `0.1`,
  `0.3`, `1`, `3`, `10`, `30`, `100`.
- Solar zenith: `0`, `30`, `50`, `65`, `75` degrees.
- Phase first moment:
  - isotropic control `g=0`;
  - current ice/precip dual-HG mean `g=0.665`;
  - current liquid dual-HG mean `g=0.75`;
  - forward-scattering stress case `g=0.85`.
- Single-scatter albedo: `1`, `0.999`, `0.99`, `0.95`, `0.8`, and the
  pure-absorption control `0`.
- Lambert surface albedo: `0`, `0.1`, `0.3`, `0.8`.

There are 6,240 rows. Visible cloud is represented mainly by `omega=1` and
`0.999`; the lower values are absorption and stability tests.

## Checks

The program exits nonzero if any check fails:

1. `tau=0` identity, including exact Lambert surface reflection.
2. Conservative atmospheric and intrinsic diffuse energy conservation.
3. Pure-absorption comparison with the analytic direct-down/surface-up result.
4. The atmospheric diffuse component beyond first scattering scales as
   `O(tau^2)` in the thin limit.
5. Exact adding is invariant to coarse versus fine stable-layer partitioning.
6. Every grid output is finite and obeys energy or reflective-cavity bounds.
7. Full-row JSON generation is byte-repeatable.

Internal down/up boundary flux may exceed the unit incident flux when a bright
surface and conservative slab recycle photons. Those values are checked against
the Lambert-cavity ceiling rather than incorrectly clamped to one. TOA
reflectance, absorption fractions, intrinsic slab coefficients, and black-surface
escape fractions remain bounded by one.

## Legacy comparison columns

Every row carries the current production octave inputs:

```text
octaves             = 6
extinction scale a  = 0.5
phase scale b       = 0.5
brightness scale c  = 0.85
sum(c^k, k=0..5)    = 4.1523365625
thin gate           = 1 - exp(-input tau)
```

They are join keys and comparison metadata only. The legacy code computes a
directional, phase-weighted source at a marcher sample. This slab prototype does
not reproduce or silently reinterpret that quantity as hemispheric flux.

## What this does not solve

- No view zenith, relative azimuth, or directional TOA radiance.
- No internal diffuse source for an arbitrary point along a camera ray.
- No liquid Mie or roughened-ice phase table.
- No vertical variation of optical properties.
- No fractional subcolumns or 3-D horizontal photon transport.
- No ABI spectral-response convolution.
- No production OD calibration, atmosphere, terrain, or display transform.

The production experiment must not be wired into the marcher until a depth-
resolved diffuse-source contract has been compared with DISORT and directional
Monte Carlo radiance.

## Reference comparison contract

Future DISORT fixtures must use the same scaled/unscaled optical properties,
normalisation, surface albedo, and geometry and report:

- TOA hemispheric reflectance;
- direct and diffuse flux at slab bottom;
- atmospheric absorptance;
- up/down diffuse flux at matched internal optical-depth levels.

The separate CUDA Monte Carlo oracle should additionally report uncertainty and:

- directional TOA radiance versus view zenith and relative azimuth;
- scattering-order histograms;
- photon path-length distributions;
- 3-D cube, broken-cumulus, and anvil-edge radiance.

Acceptance is based on reference error, not agreement between the two fast
approximations. The two-stream result may be energy-correct and still have an
unacceptable ABI viewing-angle error.

## Primary references

- Joseph, Wiscombe, and Weinman (1976), delta-Eddington:
  <https://doi.org/10.1175/1520-0469(1976)033%3C2452:TDEAFR%3E2.0.CO;2>
- Stamnes et al. (1988), DISORT:
  <https://doi.org/10.1364/AO.27.002502>
- libRadtran documentation:
  <https://www.libradtran.org/doku.php?id=documentation>
- NASA I3RC:
  <https://earth.gsfc.nasa.gov/climate/model/i3rc>
