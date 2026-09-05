# SimSat next-gen: verified radiometry foundation

Worktree: `C:\Users\drew\simsat-wt-nextgen-abi`, branch `codex/nextgen-abi`.
Based on main `22dd23b`. All changes are local commits; nothing was pushed or merged.

Task A is **not complete**: this increment fixes the gray sensor contract and adds
validated official-band response and scoring interfaces. There is no `sensor-abi`
render product yet. Calibrated spectral surface inputs (especially C03 NIR), solar
spectrum, spectral transport, satellite-view resampling and instrument/glint
validation remain required. Current gray scores are diagnostics, not ABI per-band
acceptance. Tasks C (new HRRR/night cases and fetch/align library) and D remain pending.

## Changes and verified scope

- Applied T1 and T3 as requested. T1 required three SHE-context conflict resolutions;
  retained main's SH-2 code and applied unscaled Cox-Munk in CPU and both WGSL twins.
  The SHE branch and its known failing test were not imported.
- Sensor Fast Gray v2 preserves raw reflectance above one and requested fractional
  cloud closure. It removes inherited ozone/multiple-scattering gains, land
  saturation/day gain, water daylight rescaling and boundary extinction/shadow fades.
  Display defaults retain the T1-corrected appearance. GPU preview remains display-only.
- Vendored official NOAA FM4 C01/C02/C03 SRFs with exact hashes and wavelength-domain
  quadrature. This utility requires actual spectral radiance; RGB is never relabeled.
- Added a repository-owned runner with immutable output, worktree validator,
  reference-date guards, both intents, both-cloudy scores, stage logs/checkpoints,
  and regression gates. Added unclipped RGB and independently keyed per-band NPZ
  validation; each band reports its masks/DQF and any supplied geometric glint mask.
- Audited native WRF and cached CTT without changing thermal physics. Corrected the
  compact-quantization comment: tiny positive values floor upward, not to zero.

## Four-hour results

| UTC | IR clear bias, before -> after K | IR both-cloudy bias K | Visible display-intent clear bias, before -> after | Sensor v2 clear bias |
|---|---:|---:|---:|---:|
| 12 | -2.985 -> -2.985 | +7.597 | +0.0013 -> +0.0013 | +0.0130 |
| 15 | -2.643 -> -2.643 | +9.592 | -0.0289 -> -0.0278 | +0.0248 |
| 18 | -1.123 -> -1.123 | +10.803 | +0.3550 -> +0.0363 | +0.0799 |
| 21 | -1.824 -> -1.824 | +8.959 | -0.0055 -> -0.0051 | +0.0451 |

Every raw IR Kelvin plane is byte-identical to v0.2.1; hashes are in
the external verification.json. The declared clear-IR 0.5 K and 21Z
clear-display-RGB 0.02 regression gates pass. The full metric comparison is
the external comparison.txt, with [scores.json](notes/nextgen/scores.json) and the external run.json.
All other reported metrics remain available for review; those two gates alone do
not certify physical improvement. The noon display improvement is T1's unscaled glint.

The brief overstated part of the original visible comparison: `rgb-reflectance-out`
is already pre-exposure/pre-tonemap, and raw paths bypass the land SZA/toe controls.
Its display intent still scales extinction by 0.15 and retains other upstream
appearance assumptions. Thus the mismatch is real, but exposure 1.5 was not in the
scored raw plane. The new sensor diagnostic uses unclipped observed/raw values;
legacy display scoring is preserved separately for exact baseline comparisons.

| UTC | Gray sensor clear bias v1 -> v2 | Gray overall bias v1 -> v2 | Gray correlation v1 -> v2 |
|---|---:|---:|---:|
| 12 | +0.0119 -> +0.0130 | +0.0071 -> +0.0087 | 0.3959 -> 0.3832 |
| 15 | +0.0281 -> +0.0248 | -0.0374 -> -0.0404 | 0.2175 -> 0.2140 |
| 18 | +0.3691 -> +0.0799 | +0.2679 -> -0.0223 | 0.0312 -> 0.1155 |
| 21 | +0.0486 -> +0.0451 | -0.0502 -> -0.0530 | 0.1661 -> 0.1652 |

Sensor-v1 and v2 metric conventions and regime sample counts are identical.
Dawn worsens slightly; 15/21Z overall absolute bias and correlation also worsen
slightly. This is a mixed result, with the large gain concentrated at noon.

These sensor results include top-down/GOES view mismatch and gray surface/optics
error. In particular v2's noon +0.080 clear bias misses the requested few-hundredths
accuracy. No case-specific constant was adjusted in response. Remaining inherited
approximations include the diffuse cloud-shadow floor, Blue Marble water/snow proxies,
fixed-radius/gray hydrometeor optics and broad RGB gas coefficients. Legacy references
also share a C02-derived validity mask; independent per-band DQF cannot recover pixels
already excluded by that common mask.

## Cloud-temperature finding

Native air minima are 195.035, 195.584, 195.798 and 195.656 K; visible tau=1 CTT
minima are 213.806, 212.087, 214.181 and 207.712 K. GOES reaches 192-193 K at
18/21Z. Co-location exposes major vertical/displacement error even within
both-cloudy: the 21Z observed coldest pixel has model cloud only near 1.93 km
and 290 K. This is evidence of forecast-state mismatch, not proof that all
+8-11 K is forecast or that scattering can never lower brightness temperature.
See [NOTES-cloud-temperature.md](NOTES-cloud-temperature.md) and [ctt-audit.json](notes/nextgen/ctt-audit.json).
The case has no native CLDFRA and no trusted cached cloud-fraction field.

## Verification and reproduction

`cargo test -p simsat --locked --offline`: 581 library tests passed, 2 ignored;
all CLI/integration tests passed (603 passed in total). `cargo build --workspace`
and the release CLI build passed sequentially. The GUI was only adjusted for
its new struct initializer/operator metadata; no GUI features were changed or launched.
The original jobs=6 configuration remains unchanged and main remains clean.
Validator self-check, 13 harness tests, and the CTT self-test passed. The CTT
four-hour check reproduced T3 masks exactly and executable CTT summaries to 0.001 K.
Logs accompany this report.

```powershell
cd C:\Users\drew\simsat-wt-nextgen-abi
python -B scripts/simsat-case-score.py --data-root C:/Users/drew/soma-render-work --bin target/release --out C:/Users/drew/soma-render-work/out/<fresh-name> --hours 12,15,18,21
python -B scripts/simsat-case-score.py --compare C:/Users/drew/soma-render-work/out/simsat-main C:/Users/drew/soma-render-work/out/simsat-nextgen-abi
```

The current runner consumes already aligned cases; it is not yet Task C's
one-command download/align/HRRR/night case library. `NOTES-cloud-temperature.md`
in the worktree contains the standalone native-temperature audit command.

The next implementation gate is a wavelength-resolved transport path with
independent spectral surface and solar inputs, using the actual satellite view
and a documented ABI sampling/resampling contract. Then score C01/C02/C03,
including clear/glint and both-cloudy strata. Only after that should thermal
contribution profiles be used to test genuinely collocated cold cloud tops.

Sources: [NOAA CMIP ATBD, section 3.4.1.2](https://www.star.nesdis.noaa.gov/goesr/documents/ATBDs/Enterprise/ATBD_Enterprise_Cloud_and_Moisture_Imagery_Product_v4_2021-01-13.pdf),
[official FM4 SRFs](https://ncc.nesdis.noaa.gov/GOESR/docs/GOES-R_ABI_FM4_SRF_CWG.zip).
Physics provenance and exact asset hashes are also in `crates/simsat/assets/abi_srf/README.md`.

External comparison images and the shareable report are in C:/Users/drew/Documents/Codex/2026-09-04/ta/outputs.
