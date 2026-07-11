# Stage-0 cloud-closure comparison and fit harness

This isolated experiment joins the committed delta-two-stream and CUDA slab
fixtures, evaluates the current Wrenninge/Oz octave source in a controlled
homogeneous slab, and emits the missing reference grid for a bounded Stage-1
experiment. It does not modify or select a production renderer mode.

## Observable contract

The three columns are deliberately not treated as interchangeable:

- delta-two-stream `toa_reflectance` is an upward **hemispheric flux fraction**;
- CUDA `brf` is a **directional** bidirectional reflectance factor,
  `pi I/(F0 mu0)`;
- the legacy slab diagnostic is also a directional BRF, but it comes from
  integrating the current local octave source through an idealised black slab.
  It is a diagnostic of the heuristic, not a radiative-transfer solution.

The comparison never divides delta flux by CUDA BRF and never calls their
difference an error. The exact common inputs may be placed on one row, but they
remain different observables.

## Run

Only the Python standard library is required:

```powershell
python experiments/cloud-closure-fit/cloud_closure_fit.py check
python experiments/cloud-closure-fit/cloud_closure_fit.py generate
python experiments/cloud-closure-fit/cloud_closure_fit.py all
python experiments/cloud-closure-fit/analyze_stage1.py all
python experiments/cloud-closure-fit/generate_stage2_requests.py all
```

`all` validates both upstream self-check files, recomputes the comparison,
writes LF/UTF-8 fixtures, and rejects stale/non-repeatable output.

Committed outputs:

- `fixtures/stage0-comparison-v1.csv`;
- `fixtures/stage0-summary-v1.json`;
- `fixtures/stage0-self-checks-v1.json`; and
- `fixtures/stage1-request-grid-v1.csv`.

After running the CUDA request grid, Stage 1 additionally commits canonical,
timing-independent evidence:

- `fixtures/stage1-oracle-manifest-v1.csv`;
- `fixtures/stage1-exact-join-v1.csv`;
- `fixtures/stage1-summary-v1.json`; and
- `fixtures/stage1-self-checks-v1.json`.

The raw one-case CUDA files remain ignored. A second complete 603-case run
reproduced all 603 physics rows exactly; only the three elapsed/kernel timing
fields changed. `analyze_stage1.py` deliberately excludes those timing fields
from its canonical hashes and fixtures, so both full runs generate identical
Stage-1 output bytes.

Every output records or is covered by SHA-256 hashes of the exact committed
inputs. No timestamp is embedded.

## Defensible common cases

The committed fixtures have four exact black-surface input joins:

- the zero-optical-depth control; and
- `tau=0.3, 1, 3`, `g=0.75`, `SSA=0.999`, `SZA=30 degrees`.

Only the latter three carry nonzero information, and all use the same nadir
view and relative azimuth. A constrained one-parameter diagnostic therefore
fits

```text
BRF = analytic HG single-scatter BRF
    + alpha * legacy HG-proxy higher-order BRF,  0 <= alpha <= 1.
```

This is a same-observable directional comparison: the legacy proxy uses the
same single-HG input as CUDA solely to isolate the octave transport math. It is
not the shipping liquid/ice dual-HG phase. Both exact shipping phase variants
are reported separately and are not included in the fit.

Three optical depths at one geometry are insufficient to identify a production
closure. The result is labelled diagnostic-only and is not proposed as a
default or a hidden tuning coefficient.

## Legacy source-energy diagnostic

At a slab depth, every normalized phase integrates to one over solid angle.
The local six-octave source therefore integrates to

```text
sum_k gate^k * 0.85^k * exp(-tau_sun * 0.5^k).
```

The unattenuated weights sum to `4.1523365625`. Values above one are not, by
themselves, proof of an energy violation: a real multiply scattered diffuse
field can exceed the local direct beam. The scientifically relevant limitation
is that the legacy source is not derived from any conserved diffuse-energy
state. The top/middle/bottom columns quantify that behavior without assigning
it a false hemispheric meaning.

## Missing reference grid and command plan

The generated request grid contains only states that have an exact committed
delta-flux row. It adds the directional dimensions that the current overlap
lacks:

- optical depth `0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30`;
- HG asymmetry `0.665, 0.75, 0.85`;
- near-conservative `SSA=0.999`, plus a small `SSA=0.95` holdout;
- solar zenith `30, 50, 65 degrees`;
- view zenith `0, 40, 65 degrees`; and
- relative azimuth `0, 90, 180 degrees` where meaningful.

The `50-degree` solar slice and selected angular combinations are held out.
Run the request without editing the CUDA oracle:

```powershell
& .\experiments\cloud-closure-fit\run-request-grid.ps1
```

The runner writes one CSV per case under the ignored `requested-results/`
directory. The exact 603-case grid has now been executed twice on the RTX 5090.
The first sequential run spanned 148 seconds and wrote 391,387 bytes of raw
CSV. Use `analyze_stage1.py all` to validate, join, and canonicalize those
results. The harness never interpolates or fabricates an oracle value.

## Stage-1 result: constant alpha is rejected

The calibration-only bounded fit is:

```text
BRF_pred = analytic single-HG BRF
         + 0.631799 * legacy HG-proxy higher-order BRF
```

It passes only 177/324 calibration cases, 137/243 angular holdouts, and 9/36
absorbing holdouts under the predeclared BRF tolerance after removing the CUDA
95% Monte Carlo interval. The effective per-case alpha changes materially with
tau, HG asymmetry, SSA, and geometry; a scalar is not a closure.

All 603 cases meet the requested uncertainty target (maximum 95% half-width
`0.004071` BRF). Every `tau=30` result is provisional because the 512-order
cap truncates up to `0.007825` of paths. The conclusion does not depend on that
slice: after excluding `tau=30` and refitting alpha, only 306/528 cases pass.

Surface-albedo stability is not tested: the Stage-1 directional oracle and all
exact joins use a black lower boundary. The delta columns in the joined fixture
remain hemispheric flux; no delta-flux/CUDA-BRF ratio is used as an error.

## Explicit Stage-2 request

`generate_stage2_requests.py all` produces an 800-row request contract in
`fixtures/stage2-request-grid-v1.csv`. It adds Lambertian surface albedos
`0/0.2/0.6/0.85`, matched forward `R/T/A`, 32 depth bins, absorption holdouts,
and two dual-HG phase-shape holdouts. The dual-HG liquid and ice profiles are
paired against single HG profiles with the same first moments (`0.75` and
`0.665`), so the grid can detect information that one `g` value loses.

The request is explicitly not runnable with the current oracle. The smallest
oracle extension is recorded in
`fixtures/stage2-request-summary-v1.json`: Lambertian lower-boundary transport,
mixture-HG sampling, matched forward flux accounting, and depth-binned collision
source output. Forty `tau=30` states are paired at 1024/2048 orders. Effective
radius is not used as a phase surrogate until a wavelength-specific liquid/ice
phase table is chosen and recorded by immutable hash.

## Stage-1 experiment interface

Recommended mode, default off:

```text
cloud_multiscatter = legacy-octaves | single-scatter | delta-flux-v1
cloud_diffuse_blend = [0,1]       # experiment only; default 0
cloud_closure_lut_version = hash  # recorded in render metadata
```

`delta-flux-v1` should index an immutable LUT by total vertical optical depth,
fractional depth, solar cosine, single-scatter albedo, effective asymmetry, and
surface albedo. It returns upward/downward diffuse flux, absorption, and bounds
flags. Exact direct single scatter stays unchanged. The higher-order directional
reconstruction must use a nonnegative kernel normalized over the hemisphere;
it must not turn a hemispheric flux into directional brightness with an
unbounded gain.

The current Stage-0 boundary fluxes do not define the needed internal source.
The mode remains an interface proposal until depth-resolved reference fixtures
exist.

## Acceptance criteria

- Zero: `tau=0` and `SSA=0` give exactly zero cloud BRF/source.
- Thin limit: higher-order contribution scales as `O(tau^2)`, accepted slope
  `2 +/- 0.05`.
- Conservation: `R+T+A=1` within `1e-4`; all three are nonnegative.
- Monotonicity: conservative black-surface hemispheric reflectance is
  nondecreasing with optical depth. Directional monotonicity is asserted only
  for declared geometry slices, not universally for anisotropic scattering.
- Directional reference: error below the larger of `0.005` absolute or `5%`
  relative outside Monte Carlo uncertainty on both calibration and held-out
  cases.
- Identity: `legacy-v014` and `cloud_multiscatter=legacy-octaves` dispatch to
  unchanged arithmetic and retain the exact v0.1.4 CPU output hash.

The interface, toggle, LUT identity, and render metadata must eventually be
surfaced together in Rust CLI, Python, and Studio. None is added by this
experiment.
