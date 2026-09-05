# SimSat realism and 3D progress

Historical milestone; active work targets the highest achievable realism in overhead simulated satellite imagery. Cloud-seeding support is one application. See [SATELLITE-ONLY.md](SATELLITE-ONLY.md).

Worktree: `C:\Users\drew\simsat-wt-nextgen-abi`, branch `codex/nextgen-abi`. Main is untouched; nothing merged or pushed. Cargo jobs remains 6.

Images are mirrored to `C:\Users\drew\Downloads\SimSat-Cloud-Renders`. The wide 1974 cloud scene is the strongest current 3D output. Ground tornado images are diagnostic experiments; they do not yet resolve a convincing visible funnel.

## Four-hour regression

All four original topdown cases completed. Both declared gates pass against original v0.2.1 and the first next-gen increment: clear IR absolute bias deterioration <=0.5 K at every hour; 21Z display clear bias deterioration <=0.02. This does not establish full radiometric accuracy.

| UTC | Clear IR K | Both-cloudy IR K | Display clear: original / first increment / now | Sensor gray clear: first / now |
|---|---:|---:|---|---|
| 12Z | -2.9850 | +7.5966 | +0.00135 / +0.00135 / +0.00135 | +0.01304 / +0.01278 |
| 15Z | -2.6430 | +9.5917 | -0.02894 / -0.02781 / -0.02790 | +0.02484 / +0.02433 |
| 18Z | -1.1228 | +10.8027 | +0.35502 / +0.03634 / +0.03624 | +0.07990 / +0.07969 |
| 21Z | -1.8241 | +8.9589 | -0.00551 / -0.00515 / -0.00531 | +0.04510 / +0.04411 |

All four thermal planes are byte-identical to the first increment. The +7.6 to +10.8 K collocated-cloud warm residual remains. The native CTT audit previously found minima 207.7–214.2 K, far warmer than the coldest observed pixels. No global cloud-cooling offset was introduced.

## Geometry and case library

Satellite model-grid rays now use the physical satellite viewpoint and preserve the trusted target grid. The official C01/C02/C03 response registry is present, but an independent per-band transport/surface operator is still unfinished: current visible scores are gray RGB diagnostics. Do not call these ABI band-reflectance acceptance results.

Independent Texas daytime and tropical night cases completed using verified aligned GOES-19 references. Texas is HRRR-prepared WRF, not a native GRIB ingest test. Its clear IR bias is +9.02 K and both-cloudy bias +22.44 K; native skin temperatures over observed-clear pixels average 320.05 K versus ABI 305.11 K and lowest model air 306.59 K. Night clear IR bias is -13.35 K, both-cloudy -3.76 K; these holdouts reveal unresolved forecast/operator differences. All four satellite-view expanded cases are now complete; current results are in SATELLITE-ONLY.md.

## Actual 3D clouds and the tornado

The 1974 1 km and 333 m files are fully copied from weather-node-2 with matching hashes. They are different simulations, so their comparison is not a resolution-only experiment. The user-selected 22:29 333 m file also has a matching full SHA-256 (69275867c5d8ee9723b1e64613b566ed9be5a14120ce0a14eefe80d9515d3926).

At 22:29 the diagnosed near-surface rotation maximum is about 39.47088 N, 84.85352 W (i355, j327). Its local pressure anomaly relative to a 5 km Gaussian background is -9.18 hPa; this is not an absolute tornado central-pressure deficit. Cameras use actual native HGT + 2 m and the model timestamp sun.

The new opt-in terrain camera intersects native bilinear HGT, stops surface/cloud paths at foreground hills, evaluates ground shadows at the actual hit, and adds a direct camera-to-cloud atmosphere integral. The latter uses the existing optical-centroid approximation; it is not full coupled path tracing. The default satellite/topdown and legacy perspective paths are unchanged.

Controlled 250 m vs 50 m vertical sampling keeps the full 600x600 footprint and 19,750 m top, fixed optics, camera, sun, and exposure. The finer ingest takes 116.68 s and peaks near 3.55 GB, exceeding the default ingest budget only in this explicit experiment. Matched 160x90 previews show no decisive recovery of a visible funnel. Finer horizontal detail was not invented.

Ground camera tests at 22:29 and 22:27 remain heavily obscured. Exposure 4 and the existing delta-flux-v2 approximation are labeled display/lighting experiments, not promoted defaults. Native-condensate geometry checks omit rain and radiative transfer and are explicitly diagnostic. Debris, vegetation-scale geometry, and a resolved sub-grid condensation funnel are not present in these outputs.

## Verification

618 Rust tests passed, 2 ignored; the full workspace including GUI builds. The final render CLI tests (19) and native vertical-ingest example compile check pass. Previous reference/case tooling checks: 33 Python tests plus validator self-check passed. CPU/WGSL sun and primary ray corrections remain paired; terrain perspective is a CPU-only boundary/compositing path because GPU perspective is unsupported.

The corrected complete sun march can be expensive, especially twilight and fine vertical volumes. The high-resolution fine-grid trial was stopped in favor of matched small convergence images, and this is recorded in work/tornado-fine-preview-budget.json. Regression jobs were preserved.
