# SimSat satellite-only progress

The primary goal is the most realistic simulated satellite renderer we can build: visually convincing cloud structure and physically faithful radiometry, assessed against actual satellite imagery across multiple simulations and conditions. The user's ambition is the "worlds best most realistic sim sat"; this is a development goal, not an achieved or independently benchmarked claim. Cloud-seeding support is one application and does not define the main product or displace overall realism. The current scope remains overhead simulated satellite imagery, with image outputs saved in Downloads as work progresses. Ground and free-perspective work is set aside. Worktree: C:/Users/drew/simsat-wt-nextgen-abi; branch codex/nextgen-abi. Main is unchanged, nothing merged or pushed, Cargo jobs remains 6.

## Images

Images are in C:/Users/drew/Downloads/SimSat-Satellite-Renders and mirrored in this task's outputs folder. The new 1974-333m-2229-topdown-visible.png is the 600x600 native 333 m model at 1974-04-03 22:29 UTC, timestamp sun, original display intent. It contains no added procedural cloud detail. Its matching topdown-ir13 image uses the NOAA FM4 band-13 response, the existing gray thermal march, and CIMSS enhancement; the raw plane is finite and spans approximately 201.0-295.3 K (median 212.5 K). These are simulations, not observed 1974 satellite images.

Visible remains a gray RGB display approximation (extinction scale 0.15, exposure 1.5). Full independent ABI C01/C02/C03 transport is unfinished. The before/after/GOES mosaic distinguishes original topdown, corrected topdown, corrected actual-satellite viewpoint, and aligned observations. Actual satellite rays introduce physical cloud parallax; they are not a free-perspective cloud scene.

## Thin-cloud investigation

Native 3 km liquid water paths already contain narrow bands and cells; the ice/snow fields have smoother anvils and faint extended veils. Model spacing limits recoverable structure. Opacity cannot create missing filaments, and finer image output cannot recreate unresolved model detail.

A controlled 21Z topdown test changed only display cloud optical-depth scale from 0.15 to 1.0. It made clouds broader and whiter, with no recovery of wisps. Gray RGB diagnostic results:

| Metric | Default 0.15 | Test 1.0 |
|---|---:|---:|
| Clear bias | -0.005305 | +0.025433 |
| Both-cloudy bias | -0.139465 | -0.047170 |
| All-pixel MAE | 0.149656 | 0.158201 |
| All-pixel correlation | 0.181979 | 0.166950 |

Clear absolute bias deteriorates by 0.020128, just above the declared 0.02 tolerance. All-pixel error and correlation also worsen. The parameter change is not promoted. These are gray luminance diagnostics, not ABI per-band validation; this single-hour negative does not claim a four-hour improvement.

## Completed validation

The original four-hour topdown regression is complete. Both declared gates pass against v0.2.1 and the first next-gen increment. All four IR planes remain byte-identical to the first increment. A small paired CPU/WGSL optimization now skips unused direct-sun cloud marches when Earth blocks the entire finite solar disk. Eight small-image A/Bs and the full 21Z topdown PNG and raw RGB plane are byte-identical. No controlled speedup is claimed.

All four satellite-view model-grid cases are also complete. The table retains the original topdown IR product; visible columns use the actual satellite view:

| UTC | Clear IR K | Both-cloudy IR K | Display gray clear bias | Sensor gray clear bias |
|---|---:|---:|---:|---:|
| 12Z | -2.985027 | +7.596553 | +0.002281 | +0.013846 |
| 15Z | -2.643033 | +9.591748 | -0.026755 | +0.028953 |
| 18Z | -1.122776 | +10.802652 | -0.032228 | +0.023869 |
| 21Z | -1.824095 | +8.958941 | -0.001199 | +0.053861 |

The +7.6 to +10.8 K both-cloudy thermal residual remains. The native CTT audit cannot reach the observed coldest core; no global cooling offset was added. Both-cloudy also retains model cloud-height/structure errors. For satellite-view visible, the split uses a vertical condensate-column proxy, so it is not a true slant-ray cloud intersection mask.

Texas daytime and tropical night cases are complete and recorded in notes/nextgen/case-scoreboard.md. Texas is HRRR-prepared WRF, not native GRIB validation. Its clear/both-cloudy IR biases are +9.02/+22.44 K; night has -13.35/-3.76 K. These holdout errors remain unresolved and are not hidden by the original-case regression pass. The twilight expanded 12Z run took about 89 minutes across all stages with older frozen binaries; full secondary sun integration and separate PNG/raw render passes remain expensive. This is a measured run duration, not a performance benchmark.

## Next ABI input completed

Added a tested preparation stage for published HAMSTER spectral land albedo. A bounded regional retrieval now supplies 61 spectral samples from 0.4 to 1.0 micrometres at every model-land pixel on the exact original grid. It explicitly excludes water using the model land mask. See notes/nextgen/SPECTRAL-SURFACE-INPUT.md for source links, hashes, scientific limitations, and commands. It is a climatological surface input, not TOA reflectance, and is not yet connected to rendering. Independent spectral transport and directional surface reflection remain required for task A.

Verification: 618 Rust tests pass, 2 ignored. All 40 Python tooling tests pass, including 7 new spectral-surface tests. The whole workspace including the out-of-scope GUI builds after the final paired cloud change (21.97 seconds). No GUI feature changes were made.

## One application: cloud-seeding support

One intended application is helping meteorologists supporting both glaciogenic and hygroscopic cloud-seeding missions, across planning, in-mission interpretation, and post-mission review. The user subsequently clarified that this is not the main reason for SimSat. This context should inform useful diagnostics without narrowing the broader satellite-realism goal. The product remains overhead simulated satellite imagery; ground-camera and GUI development remain out of scope.

Where model diagnostics accompany the imagery, use explicitly labeled model quantities (liquid/ice phase, water content/path, temperature/height, and vertical structure) and collocated observations. Preserve original species and units. A cloud's rendered appearance must not be used to infer a seeding opportunity or a seeded-versus-unseeded response. No seeding treatment operator or mission effectiveness validation has been implemented.

True-color C01/C02/C03 alone is not the whole mission product. NOAA's [Day Cloud Phase Distinction guide](https://www.star.nesdis.noaa.gov/goes/documents/ABIQuickGuide_DayNightCloudMicroCombo.pdf) uses C13 (10.3 um), C02 (0.64 um), and C05 (1.6 um) to help distinguish cloud-top phase. That is a possible future spectral target for this application; it cannot be synthesized honestly by coloring the native ice/liquid mass maps or reusing existing RGB. Cloud-top remote sensing must be interpreted alongside the model's vertical fields, not presented as direct observation of all water below cloud top.

The [WMO weather-modification statement](https://public.wmo.int/content/wmo-statement-weather-modification) emphasizes quantifying model uncertainty and constraining simulations with observations. Accordingly, mission-oriented testing needs representative mission cases and available radar/in-situ context; the 1974 tornado remains a renderer stress test. Current single-run model diagnostics do not supply ensemble uncertainty or flight guidance.
