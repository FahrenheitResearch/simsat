# Stage-0 bounded land surface-response lab

This directory is an isolated, dependency-free experiment. It does not change
the SimSat renderer, CLI, Python bindings, Studio, shaders, product defaults, or
raw diagnostic products.

The lab compares two proposed **land-display** operators with exact identity:

1. bounded solar-elevation (SZA) normalization; and
2. a bounded dark-land toe.

Both operators multiply all three linear-RGB channels by one bounded scalar,
so they preserve color ratios without per-channel clipping. The selected grid
also remains inside unit RGB; production output mapping is outside this lab.

## Candidate definitions

For solar elevation `e`, the SZA candidate is

```text
mu       = sin(clamp(e, 0, 90 deg))
mu_ref   = sin(60 deg)
mu_floor = sin(20 deg)
raw      = clamp(mu_ref / max(mu, mu_floor), 1, 1.6)
day      = smoothstep(20 deg, 30 deg, e)
gain     = 1 + day * (raw - 1)
```

It returns the input tuple directly at `e <= 20 deg` and `e >= 60 deg`.

For unilluminated linear surface-albedo luminance `Y`, the dark-land candidate
is

```text
knee   = 0.08
curved = knee * (Y / knee)^0.65
blend  = smoothstep(0, knee, Y)
target = mix(curved, Y, blend)
raw    = clamp(target / Y, 1, 1.5)
day    = smoothstep(20 deg, 30 deg, e)
gain   = 1 + day * (raw - 1)
```

It returns a unit gain for black, at or above the knee, and at `e <= 20 deg`.
The gain scales the illuminated surface RGB; the toe is not re-evaluated from
that darker signal. The combined case evaluates SZA from elevation and the toe
from the same base albedo, then multiplies the two independent gains, matching
the production experiment's independently switchable controls.

These constants are experiment coordinates, not recommended production
defaults.

## Synthetic grid

The 1,440 rows span:

- four RGB ratios: forest, neutral, dry soil, and cool rock;
- ten base luminances from black and `1e-6` through `0.5`;
- twelve solar elevations from `-5` through `75 deg`, including exact identity
  boundaries and the HRRR/Maine-like `33 deg` point; and
- terrain slopes of `-20`, `0`, and `+20 deg` in the solar vertical plane.

Each row records both the base linear albedo RGB and an illuminated input RGB.
The latter uses a simple Lambert cosine plus a fixed 0.18 diffuse-floor proxy.
That proxy only exercises response behavior over a broader range; it is not a
SimSat atmosphere, BRDF, or terrain-lighting model.

## Placement and lighting contrast

The lab mirrors the production experiment's placement:

```text
A       = linear surface albedo after snow/vibrancy handling
L       = illuminate(A, terrain normal, horizon, sun, diffuse sky)
g_toe   = toe_gain(luminance(A), sun elevation)
L_toe   = g_toe * L
```

For one material viewed on differently illuminated slopes, `g_toe` is the same
scalar. Therefore `L_toe(slope 1) / L_toe(slope 2)` equals the original
lighting ratio. Deriving the toe from `luminance(L)` instead would assign a
larger lift to the darker slope or cast shadow, flattening terrain contrast and
turning a material-response control into a shadow fill. The self-check holds
SZA, toe, and combined gain spread across the three slope proxies at exact zero.

## Run

Only Python's standard library is required.

```powershell
python experiments/surface-response/surface_response_lab.py check
python experiments/surface-response/surface_response_lab.py generate
python experiments/surface-response/surface_response_lab.py all
```

`all` runs the self-checks before writing deterministic fixtures:

- `fixtures/stage0-surface-response-v1.csv`: every identity/SZA/toe/combined
  response row;
- `fixtures/stage0-surface-response-summary-v1.json`: configuration, hashes,
  aggregate maxima, a moderate-sun forest anchor, and structured checks; and
- `fixtures/stage0-surface-response-checks-v1.txt`: human-readable check report.

There are no timestamps or random inputs. The summary records the generator,
CSV, and check-report SHA-256 values.

## Enforced invariants

The program exits nonzero unless all of these hold:

1. disabled is bit-for-bit identity;
2. black is bit-for-bit identity for both operators and their combination;
3. the toe is bit-for-bit identity at and above `Y=0.08`;
4. every candidate is bit-for-bit identity at and below `20 deg` solar
   elevation;
5. SZA normalization is bit-for-bit identity at and above `60 deg`;
6. every RGB value is finite and in `[0, 1]`;
7. chromaticity is preserved to `1e-12`, and SZA/toe/combined gains are
   identical across slopes for one albedo/elevation;
8. SZA, toe, and combined gains obey their declared bounds;
9. no operator darkens its input;
10. the grid actually activates both operators; and
11. rebuilding the CSV is byte-repeatable.

## Scope boundary

This lab verifies the candidate's algebra and mirrors its intended pipeline
placement; it does not select a default or validate a rendered product. A
default decision still needs real frame A/Bs, raw-product identity tests,
cloud/water/snow scope tests, CPU/GPU parity, and the full CLI/Python/Studio
control contract. The companion research and dataset plan is in
`notes/v015-surface-physics.md`.
