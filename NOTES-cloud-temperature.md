# Cloud-temperature audit: 4 September 2026

The requested first check is negative: SimSat's `derived=ctt` does not reach
193 K at any of the four hours. Native WRF air also never reaches 193 K.
However, visible tau=1 CTT is not an absolute thermal source-temperature floor,
and the existing both-cloudy mask does not isolate observation-operator error.
These measurements refine the brief's proposed binary test; they do not resolve
the complete +8 to +11 K both-cloudy residual.

No optics, extinction constants, ingest, rendering, or baseline outputs changed.
This is a reproducible diagnostic of the existing v0.2.1 baseline on one WRF
forecast, using the four trusted aligned GOES-19 references. Sources of formulas
are the repository's `derived.rs`, `bricks.rs`, `ingest.rs`, and `optics.rs`;
the operator limitations are stated in `ir.rs`. No case-fitted constants enter
the audit.

## What was measured

All temperatures below are Kelvin. "Native cloudy" means the minimum native
mass-level temperature among cells whose sum of available positive hydrometeor
mixing ratios exceeds 1e-6 kg/kg. This is a support diagnostic, not an emissivity
or cloud-top retrieval. The JSON also reports thresholds 0, 1e-8 and 1e-5.

| UTC | ABI C13 minimum | Baseline BT minimum | Visible tau=1 CTT minimum | Native cloudy minimum | Native air minimum, any cell | Both-cloudy baseline bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 199.835 | 209.575 | 213.806 | 198.221 | 195.035 | +7.597 |
| 15 | 196.331 | 211.059 | 212.087 | 196.425 | 195.584 | +9.592 |
| 18 | 192.132 | 213.009 | 214.181 | 196.536 | 195.798 | +10.803 |
| 21 | 192.987 | 206.417 | 207.712 | 196.881 | 195.656 | +8.959 |

At 18 and 21 UTC the native forecast contains no air as cold as the observed
192–193 K tail. That is missing cold thermodynamic support; changing only
cloud absorption does not create a new colder model cloud. It is **not** a
universal theorem that a brightness temperature must be bounded by local air
temperature under every scattering treatment or background boundary condition.
The all-domain minimum is also not a collocation test.

The observed <=235 K area versus baseline area remains 4.902/2.709,
4.350/3.185, 7.299/3.071, and 12.045/6.930 percent at the four hours.
CTT <=235 K areas are 2.433, 3.076, 2.924, and 7.124 percent, counting thin
NaN columns as outside the cold area. These are diagnostics; none is a proposed
replacement for modeled band radiance.

## Co-location changes the interpretation

Within **both-cloudy** columns whose observed C13 BT is <=205 K:

| UTC | Both-cloudy cold pixels | Pixels with native condensate >1e-6 | Native cloudy minimum at least as cold as the co-located observed BT |
| --- | ---: | ---: | ---: |
| 12 | 311 | 311 | 0 |
| 15 | 99 | 99 | 0 |
| 18 | 351 | 351 | 3 |
| 21 | 386 | 385 | 0 |

At 21 UTC the observed minimum, 192.987 K at 14.61395 N, 91.24892 W
(north-first row 163, column 152), is marked both cloudy. But native model
condensate above the threshold reaches only about 1.930 km, at 290.045 K.
The baseline BT is 299.161 K and visible tau=1 CTT is absent. This is a
low model cloud co-located with observed deep convection, not evidence that
the thermal operator alone owes the roughly 106 K pixel error.

At 18 UTC the observed minimum, 192.132 K at 11.58197 N, 92.30524 W
(row 275, column 114), has no native condensate above 1e-6 kg/kg.
Baseline BT is 298.319 K, and visible tau=1 CTT is absent.

Even the baseline's coldest pixels are displaced: at its minimum-temperature
locations, ABI C13 is 272.907, 282.564, 277.223 and 292.314 K respectively.
Thus comparison of independent global minima cannot quantify operator error.
The JSON records the locations and co-located values.

A two-dimensional condensate mask is valuable for excluding observed-only
clouds, but it cannot distinguish low cloud from high cloud, displacement,
cloud-top height, or optical thickness. Both-cloudy bias remains a mixture of
forecast structure and observation-operator error. The native-threshold
comparison also cannot assign an optical depth to the small cold condensate
it finds. None of these tests proves the remaining operator is correct.

## Definitions and numerical checks

- `derived::cloud_top_temp_field` descends through the **unscaled visible**
  extinction, adding `(ext_liquid + ext_ice + ext_precip) * dz`. It returns
  the stored temperature at the first whole layer crossing tau=1, without
  crossing-height interpolation. Full-column visible OD below 1 produces
  NaN, even when condensate is present.
- The four baseline caches are SSB v6 compact-u8, 474 x 380 x 80, with
  z_min=0 and dz=250 m. The script decodes the per-channel log LUT to f32
  and the stored binary16 Celsius temperatures, then performs the CTT sum
  in f64 as Rust does.
- The raw WRF files have 49 native mass levels, Thompson MP_PHYSICS=8,
  QCLOUD/QICE/QSNOW/QRAIN/QGRAUP, and **no CLDFRA variable**. Every cache
  reports `has_cloud_fraction=false`. `ir::IrVolume` also has no cloud
  fraction field, and the API explicitly ignores the visible fractional
  setting for thermal products. A thermal subcolumn implementation is an
  open feature, but this case cannot attribute its residual to mishandling
  trusted native CLDFRA.
- Native temperature follows the ingest operation order: add P+PB as f32;
  evaluate `(T + 300) * ((P+PB)/100000)^(2/7)` as f64; store the result
  as f32 before vertical resampling. These constants come from WRF through
  `optics::{THETA_BASE,P0,KAPPA}`. PH and PHB are added as f64, divided
  by the repository's g=9.81, vertically destaggered, then stored as f32.
  Relative to an all-f64 reconstruction, the largest temperature difference
  over any of the four full volumes is 0.000015259 K. NumPy and Rust libm
  need not be bit-identical at every transcendental operation; this scale
  is immaterial to the Kelvin-size conclusions.
- An "any-positive brick extinction" minimum is included only as a
  sensitivity diagnostic. Compact quantization floors positive values
  below vmin upward to code 1; tiny condensate tails are therefore not a
  physically meaningful cloud-top criterion. The native threshold sweep
  avoids mistaking that minimum for optically thick cold cloud.
- The script reproduces T3's cloud mask with its documented standard-density
  inverse and compares it with the existing raw T3 u8 planes when
  `--mask-pattern` is supplied: **zero mismatches at every hour**.
  Slight native-mask differences are expected because T3 uses resampled
  and quantized extinction plus standard density, while native thresholds
  use source mixing ratios.
- Reversing native WRF rows to north-first gives **exact** XLAT/XLONG
  agreement with the aligned references at every hour. Baseline/observation
  validity and the official BCM cloud mask match the existing validator.
- With `--verify-bin`, decoded CTT min, max and median match the unchanged
  release executable's `DERIVEDSUMMARY` to its printed 0.001 K precision
  at all four hours. The original CLI rejects `bt-out` with `derived`;
  this script obtains the physical array by decoding the existing cache.

## Reproduce

From this worktree, with Python containing numpy and netCDF4:

```powershell
python scripts/simsat-ctt-audit.py --self-test

python scripts/simsat-ctt-audit.py --data-root C:/Users/drew/soma-render-work --out out/ctt-audit --date 2026-09-04 --hours 12,15,18,21 --mask-pattern "out/simsat-validator-both-cloudy/cloudmask_d01_{hour:02d}Z.bin" --verify-bin C:/Users/drew/simsat/target/release/simsat-render-ir.exe
```

Both commands passed. The four-hour data run took about 8.5 seconds locally,
including four small derived renders; this is an audit runtime, not a renderer
performance benchmark. No compilation was needed. The script's self-test
covers a cold translucent layer over a warmer tau=1 crossing, a thin NaN column,
the Poisson surface case, and real miniature SSB channel/temperature decoding.

`--data-root`, `--date`, `--hours`, and the four `--*-pattern` arguments
make the inputs relocatable. Patterns accept `{date}`, `{hour:02d}`,
`{compact_date}` and `{cache_date}`. The source WRF time, SSB time, dimensions,
and target lat/lon are checked. This script handles native-grid WRF and compact
SSB v6 only; it does not ingest HRRR or claim support for science-f16 caches.
`--save-arrays` optionally writes NPZ planes for follow-up work. No source
NetCDF, SSB, NPZ planes, or generated PNG is checked in. The small numerical
snapshot is `notes/nextgen/ctt-audit.json`.

## Next discriminating measurement

After the radiometric/view-geometry correction in Task A, retain both-cloudy
reporting but add cloud-height/optical-thickness strata and measured thermal
contribution profiles. For columns where the forecast supplies cold condensate
at a credible top and sufficient optical depth, compare native versus 250 m
resampled extinction/temperature, absorption optical depth above each level,
surface transmittance, and the emergent radiance. That separates transparent
cold tops or vertical resampling effects from displaced and shallow forecast
clouds before changing scattering, ice absorption, or continuum assumptions.
Keep the existing negative ice-absorption result as evidence; this audit
justifies no retuning and claims no improvement to the baseline.
