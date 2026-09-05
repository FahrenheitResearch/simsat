# SimSat case scoreboard

Gray RGB values are diagnostic luminance, not independent ABI reflectance bands.

Visible both-cloudy = observed cloud and vertical model condensate-column proxy; satellite rays can cross cloud in neighboring columns. IR both-cloudy is the original topdown column split. Neither split removes forecast cloud-height/structure error.

<!-- simsat-case:7214201bfd47bdb89c79 scores:d2164a4c00a844337be1c816bfc8cfdc02f670eacb6b0363bd7be0873ad6c042 -->
## 2026-09-04T18:00:00Z / Original 18Z satellite-view model grid

Input kind: wrf-netcdf. Original aligned d01 WRF case; satellite-view direct model-grid visible sampling
Case ID: `7214201bfd47bdb89c79`. Output: `C:\Users\drew\soma-render-work\out\simsat-nextgen-realism-geo\18z`. Satellite/sector: goes19/full-disk.

| Product / sampling | Clear bias | Both-cloudy bias | Observed-cloudy bias | All-valid bias | Correlation |
|---|---:|---:|---:|---:|---:|
| IR C13 K / topdown | -1.122776 | +10.802652 | +20.024585 | +12.228229 | +0.187166 |
| Display gray RGB / satellite model-grid | -0.032228 | -0.189023 | -0.208209 | -0.144207 | +0.177660 |
| Sensor gray RGB / satellite model-grid | +0.023869 | -0.070751 | -0.131849 | -0.074485 | +0.166158 |

GOES19 official FM4 C13 response

Provenance: `C:\Users\drew\soma-render-work\out\simsat-nextgen-realism-geo\18z/run.json` (input/tool/binary hashes, commands, reference scans and stage timing).

<!-- simsat-case:f3795c936eaef2118d45 scores:520224d20e506ba7fb3f642c39770925e8df5f8ca9a1d10dd589a04960f68614 -->
## 2026-09-04T19:30:00Z / texas-1930z

Input kind: wrf-netcdf. HRRR-prepared WRF, not native GRIB
Case ID: `f3795c936eaef2118d45`. Output: `C:\Users\drew\soma-render-work\out\simsat-nextgen-realism-library\texas-1930z`. Satellite/sector: goes19/conus.

| Product / sampling | Clear bias | Both-cloudy bias | Observed-cloudy bias | All-valid bias | Correlation |
|---|---:|---:|---:|---:|---:|
| IR C13 K / topdown | +9.017681 | +22.439750 | +26.950482 | +16.886570 | +0.561573 |
| Display gray RGB / satellite model-grid | -0.052974 | -0.284145 | -0.209044 | -0.122636 | +0.262912 |
| Sensor gray RGB / satellite model-grid | -0.032867 | -0.171646 | -0.165144 | -0.092104 | +0.383426 |

GOES19 official FM4 C13 response

Provenance: `C:\Users\drew\soma-render-work\out\simsat-nextgen-realism-library\texas-1930z/run.json` (input/tool/binary hashes, commands, reference scans and stage timing).

<!-- simsat-case:c94838b3457c5761c50c scores:6988e0cd1939408cce1eb1b1bba85159247aa6fab064aa923523b470fb35d8a8 -->
## 2026-09-04T03:15:00Z / night-0315z

Input kind: wrf-netcdf. Independent tropical nighttime WRF forecast
Case ID: `c94838b3457c5761c50c`. Output: `C:\Users\drew\soma-render-work\out\simsat-nextgen-realism-library\night-0315z`. Satellite/sector: goes19/full-disk.

| Product / sampling | Clear bias | Both-cloudy bias | Observed-cloudy bias | All-valid bias | Correlation |
|---|---:|---:|---:|---:|---:|
| IR C13 K / topdown | -13.349343 | -3.758178 | +1.524594 | -3.342857 | +0.167835 |

GOES19 official FM4 C13 response

Night IR mode: visible rendering and scoring deliberately skipped.

Provenance: `C:\Users\drew\soma-render-work\out\simsat-nextgen-realism-library\night-0315z/run.json` (input/tool/binary hashes, commands, reference scans and stage timing).

<!-- simsat-case:a3d481cdee8786b5add6 scores:c7d63a72af105337f42a50f31a8af0d8e6654c1f7cc0214dc1a59a82a4b88343 -->
## 2026-09-04T21:00:00Z / Original 21Z satellite-view model grid

Input kind: wrf-netcdf. wrf-netcdf
Case ID: `a3d481cdee8786b5add6`. Output: `C:\Users\drew\soma-render-work\out\simsat-nextgen-realism-geo\21z`. Satellite/sector: goes19/full-disk.

| Product / sampling | Clear bias | Both-cloudy bias | Observed-cloudy bias | All-valid bias | Correlation |
|---|---:|---:|---:|---:|---:|
| IR C13 K / topdown | -1.824095 | +8.958941 | +21.071637 | +13.496285 | +0.186223 |
| Display gray RGB / satellite model-grid | -0.001199 | -0.124128 | -0.155467 | -0.104745 | +0.217701 |
| Sensor gray RGB / satellite model-grid | +0.053861 | -0.010383 | -0.085976 | -0.039680 | +0.193296 |

GOES19 official FM4 C13 response

Provenance: `C:\Users\drew\soma-render-work\out\simsat-nextgen-realism-geo\21z/run.json` (input/tool/binary hashes, commands, reference scans and stage timing).

<!-- simsat-case:1c464b1ed50b1c3de4aa scores:2179ceb72169c23c06e347090b48d16237edc60c3b3f69a7857c9fe2cddb1879 -->
## 2026-09-04T15:00:00Z / Original 15Z satellite-view model grid

Input kind: wrf-netcdf. wrf-netcdf
Case ID: `1c464b1ed50b1c3de4aa`. Output: `C:\Users\drew\soma-render-work\out\simsat-nextgen-realism-geo\15z`. Satellite/sector: goes19/full-disk.

| Product / sampling | Clear bias | Both-cloudy bias | Observed-cloudy bias | All-valid bias | Correlation |
|---|---:|---:|---:|---:|---:|
| IR C13 K / topdown | -2.643033 | +9.591748 | +16.820444 | +8.729783 | +0.116172 |
| Display gray RGB / satellite model-grid | -0.026755 | -0.137563 | -0.160971 | -0.104804 | +0.219156 |
| Sensor gray RGB / satellite model-grid | +0.028953 | -0.016994 | -0.084177 | -0.035651 | +0.230155 |

GOES19 official FM4 C13 response

Provenance: `C:\Users\drew\soma-render-work\out\simsat-nextgen-realism-geo\15z/run.json` (input/tool/binary hashes, commands, reference scans and stage timing).

<!-- simsat-case:5498e155401a564e11ec scores:efd3935c585cc76b4b3a8b9ca22fc6854d5216dc837461ad5f3dd5543e6a3e69 -->
## 2026-09-04T12:00:00Z / Original 12Z satellite-view model grid

Input kind: wrf-netcdf. wrf-netcdf
Case ID: `5498e155401a564e11ec`. Output: `C:\Users\drew\soma-render-work\out\simsat-nextgen-realism-geo\12z`. Satellite/sector: goes19/full-disk.

| Product / sampling | Clear bias | Both-cloudy bias | Observed-cloudy bias | All-valid bias | Correlation |
|---|---:|---:|---:|---:|---:|
| IR C13 K / topdown | -2.985027 | +7.596553 | +11.212738 | +6.301522 | +0.180188 |
| Display gray RGB / satellite model-grid | +0.002281 | -0.001010 | -0.006300 | -0.003270 | +0.413800 |
| Sensor gray RGB / satellite model-grid | +0.013846 | +0.015953 | +0.005817 | +0.008708 | +0.404227 |

GOES19 official FM4 C13 response

Provenance: `C:\Users\drew\soma-render-work\out\simsat-nextgen-realism-geo\12z/run.json` (input/tool/binary hashes, commands, reference scans and stage timing).
