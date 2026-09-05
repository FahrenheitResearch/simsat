# Earth colors and daylight display — 5 September 2026

The user asked to focus on Earth color and lighting while retaining the current cloud model. This checkpoint adds an explicit NASA BMNG seasonal base-map source and exposes the existing standard sRGB display transform in the headless API/CLI. Both remain opt-in. The shipped defaults are unchanged.

## Reproduce

Fetch the twelve pinned 2 km unshaded monthly composites (about 270 MB total):

    python scripts/simsat-fetch-base-map.py --output-dir /data/bmng-base

Add these arguments to an existing visible render command:

    bluemarble-base-map=/data/bmng-base output-transform=srgb

The base-map loader uses exactly the existing month/day interpolation, crop, resolution limit and sampling. The same crop feeds CPU and WGSL. It fails on an absent/corrupt contributing month instead of silently switching source, season or resolution. `bluemarble-month=MM` still forces one month. Explicit single-file and base-map directory overrides are mutually exclusive. Logs report the selected source, month blend and output transform.

`output-transform=abi-reflectance` retains the existing lifted-shadow square-root display. `output-transform=srgb` uses the pre-existing DebugSrgb CPU/WGSL path. This is a display choice, not a new observation operator. It retains physical scene parameters, cloud transport, ground gains and exposure. Unlike the existing ABI display it has no custom shadow lift, highlight desaturation or highlight shoulder; values beyond the display range clip. It is a daylight candidate, not a validated replacement for twilight enhancement.

## What the controlled comparisons establish

* Current shaded map versus unshaded base map: small changes over terrain; no broad visual breakthrough.
* Base map with/without the existing land dark-toe correction: subtle darkening of land; not promoted.
* Base map with existing display versus standard sRGB: visibly deeper ocean blue and greater dark-region contrast. This changes presentation, not cloud structure or radiometric accuracy. Native renderer outputs are the comparison deliverables.
* Four daylight presentations are rendered: Central America 21Z and 18Z, 1974 333 m 22:29, and Texas 19:30Z. Source dates, native grid sizes, commands and binary hashes accompany the images. The original Current Best collection is untouched.

The preliminary numeric sRGB preview used RgbReflectance, which intentionally bypasses display-only land gains; it was not an exact re-encoding of the finished display scene. It is archived in the task's work directory and excluded from the final gallery. The final sRGB images are direct simulator outputs with the full display scene.

## Source interpretation

NASA's [BMNG user manual, sections 2.5 and 5.3](https://assets.science.nasa.gov/content/dam/science/esd/eo/content-feature/bluemarble/bmng.pdf) documents cubic-spline contrast enhancement and optional added relief shading. The [base maps](https://science.nasa.gov/earth/earth-observatory/blue-marble-next-generation/base-map/) avoid the added relief shading, allowing WRF terrain normals and actual Sun geometry to supply it.

The source's enhancement is not standard sRGB encoding. At a documented control point, 0.25 surface reflectance maps to RGB code 179; a standard sRGB decoder instead yields 0.450786. Thus the existing sRGB-to-linear texture path is an appearance proxy, not a calibrated MODIS/ABI albedo retrieval. No inverse spline is asserted here: the manual does not specify enough spline boundary details for an exact inversion, and JPEG quantization, compositing and gap filling impose further limits.

These are monthly 2004 composites, not contemporaneous 1974 or 2026 land conditions. NASA documents uniform substituted deep-ocean color and residual coast/snow/cloud artifacts. Credit: NASA Earth Observatory. The complete source URL/size/SHA-256 manifest is checked in; image assets remain outside git.

The next physical surface step should consume measured spectral reflectance/BRDF with explicit date and quality provenance (the prepared HAMSTER data are one available starting point), rather than fitting another brightness curve to these cases.

## Verification and limits

713 workspace tests pass, 2 ignored. Formatting, strict all-target Clippy, full workspace/GUI build, and release renderer build pass with jobs=6. The source test checks missing-partner failure, date blending, single-month selection and provenance. The display test verifies the option reaches the final PNG, and the existing raw-output invariance test now covers it. The GPU receives the same texture and the existing output-transform uniform; no new shader math is introduced.

The unchanged 21Z raw gray-RGB harness gives:

| Metric | Current topo map | Unshaded base map |
|---|---:|---:|
| Strict-clear bias | -0.001198907 | -0.001087416 |
| Strict-clear MAE | 0.048251087 | 0.048146112 |
| All-valid MAE | 0.145381955 | 0.145276288 |
| Both-cloudy MAE | 0.193924561 | 0.193863330 |

These differences are small. The raw diagnostic deliberately disables the display-only land controls; its identical hashes with land-dark-toe on/off prove that separation, not that the finished images have identical radiometry. The sRGB choice also leaves raw values unchanged. No per-band C01/C02/C03 improvement is claimed. This one-time source comparison does not replace the required four-time acceptance table. No default is promoted, and no new IR behavior is introduced.
