# v0.1.7 render intent

SimSat now separates the shipped visualization preset from a strict, explicitly limited
observation-operator path.

## Interfaces

- Rust: `RenderParams::intent = RenderIntent::Display | SensorFastGray`.
- CLI: `intent=display|sensor-fast-gray` (`mode=` remains the camera/view alias).
- Python: visible-family functions accept `intent="display"|"sensor-fast-gray"`.
  Provenance is returned on `Georef.intent`, `Georef.observation_operator`,
  `Georef.intent_adjustments`, and `Georef.intent_limitations`.
- Studio: the top strip has an **Intent** selector. The strict controls are applied to a
  temporary render snapshot; the user's Display sliders are not overwritten.

`Display` is the default and is an exact no-op. Its operator slug is
`simsat-display-v1`.

`Sensor Fast Gray` uses the stable provenance slug `simsat-fast-gray-v1`. On a clone of
the request it deterministically selects:

- cloud optical-depth scale `1.0` (unscaled model extinction),
- exposure `1.0` and ground lift `1.0`,
- identity land appearance,
- exposed-domain feather, granulation, and top-down stratiform reconstruction off,
- product-facing atmosphere correction off (full modeled path airlight retained),
- identity highlight shoulder / hard clamp,
- synthetic green off.

Model fractional-cloud handling stays on when requested. The current GPU-preview envelope
cannot honor that contract because it substitutes legacy full cells and shipped display
highlights, so Sensor Fast Gray fails clearly on `backend=gpu-preview` and Studio uses CPU.
Every value that actually changed is reported; nothing is silently overridden.

## Honest scope

This is a semantics/provenance release, not a claim of ABI/AHI channel equivalence.
`simsat-fast-gray-v1` still uses broad band-averaged RGB/gray optics, the compact brick,
and fixed-radius cloud optics unless a source path explicitly supplies scheme-aware optics.
It does not yet integrate instrument spectral-response functions, PSF/MTF, or temporal
footprints. Finished RGB is an inspection transform; quantitative callers should use
`Product::RgbReflectance` / Python `render_rgb_reflectance` until real `SensorBand`
products land. Deprecated `VisibleBands` / `render_visible_bands` aliases remain compatible.

## Real smoke pair

Both frames use the cached HRRR 2026-07-10 21Z field, top-down native 1799x1059,
Interactive steps, and model fractional clouds:

- `outputs/render-intent/hrrr-display.png`
- `outputs/render-intent/hrrr-sensor-fast-gray.png`

The strict result is intentionally much thicker/brighter in cloud because it consumes the
unscaled model extinction instead of the owner-selected Display scale `0.15`. That visible
difference is the reason these modes must carry distinct names and provenance.

## Exact-grid GOES-19 check

The two intents were also rendered clouds-off as raw `f32le` RGB reflectance on the exact
1799x1059 HRRR target grid and compared with the aligned 2026-07-10 21:01 UTC GOES-19 ABI
reference using `scripts/simsat-validate-goes.py`. The official ACM strict-clear mask
contains 822,261 pixels. This isolates the surface/atmosphere operator from forecast cloud
displacement; it is not a forecast-skill score.

| Strict-clear luminance | Display | Sensor Fast Gray |
| --- | ---: | ---: |
| Bias | -0.032703 | -0.006808 |
| RMSE | 0.045452 | 0.033173 |
| MAE | 0.039440 | 0.024751 |
| Correlation | 0.817489 | 0.818868 |

Blue-channel RMSE falls from `0.059413` to `0.026136`; retaining modeled path airlight is
therefore a measured radiometric improvement on this case, not just a provenance change.
The remaining error and nearly unchanged spatial correlation still require multi-case
spectral surface/atmosphere work before any quantitative ABI-equivalence claim.
