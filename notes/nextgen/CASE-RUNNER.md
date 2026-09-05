# Running an ABI collocation case

`python scripts/simsat-case-run.py --help` documents the single-case entry point.
Every run requires an input file, its exact valid time, a release binary directory,
and a fresh output directory. Existing outputs are refused before any rendering.

Example using an existing aligned reference bundle:

```powershell
python scripts/simsat-case-run.py --input <wrfout-or-hrrr.grib2> --time 2026-09-04T19:30:00Z --bin target/release --out out/cases/texas-1930 --satellite goes19 --sector conus --reference <aligned-reference-directory> --label "West Texas" --input-provenance "HRRR-prepared WRF; native wrfnat GRIB not tested"
```

Omit `--reference` to select, download, align and preview official NOAA imagery.
Supported sectors are `conus`, `full-disk`, `meso1`, and `meso2`; Meso COD uses
Full Disk through the repository fetcher. `goes19` selects the GOES-East renderer
preset and `goes18` selects GOES-West. GOES18 requires explicit
`--allow-goes18-fm4-approximation` because the available official FM4 response
belongs to GOES19; its IR scores are labeled as a spectral-response approximation.
`--threads` defaults to six workers, passed to both renderer CLIs and inherited
through `RAYON_NUM_THREADS`. Add `--night-ir` to render and score only
C13, explicitly recording that visible products were skipped.

The runner verifies the input time against WRF `Times` or GRIB valid-time metadata.
For a WRF file with multiple timesteps it selects the unique exact match; explicit
`--timestep` must agree. A prepared HRRR-derived WRF is recorded as `wrf-netcdf`,
with its title and supplied provenance; a successful prepared-WRF case is not a
native GRIB ingest test. The GRIB adapter supports Lambert template 3.30 and the
same spherical Earth codes as SimSat. Rotated grids are rejected explicitly.

The target mesh reproduces `frame.rs`'s integer-center-cell anchor and
`camera.rs`'s native north-first map sampling, including its 0.01-cell boundary
inset, with f32 coordinates. Stored source coordinates must agree with the
analytical model grid within the engine's 0.05-cell check. Reused references must
have a matching target time, platform, sector, content hash, dimensions and mesh.
The aligned companion must bind the exact source-manifest name and SHA256.
Legacy manifests without a sector may identify it through a unique selected
MCMIP C/F/M1/M2 product or object key; ambiguous identities are rejected and
original references are never changed. Object platforms must match the bucket;
the maximum mesh offset is recorded and must stay below 0.02 grid cells. Existing
references aligned on the original cell centers can therefore differ at the
inset boundary by about 0.01 cell; this difference is reported, not hidden.

IR uses the official ABI C13 FM4 SRF, topdown/native sampling and a mandatory
vertical condensate mask. Visible display and SensorFastGray both use
`view=geo raster=model-grid`, retaining the preset satellite subpoint and direct
model-ground-point sampling. The geometry sidecar states that this is neither an
ABI fixed lattice nor a PSF integration. Visible scores remain gray RGB luminance
diagnostics; SensorFastGray uses unclipped reflectance. The visible both-cloudy
regime intersects observed cloud with the **vertical model-column** condensate
proxy, while an oblique ray can cross cloud in neighboring columns. Cloud height
and structure errors remain in both-cloudy statistics.

`run.json` records source/binary/tool hashes, commands, timestamps, stage timings,
reference provenance and geometry checks. `scores.json` checkpoints completed
products after each validator; only `run.json` status `complete` denotes success.
The scoreboard defaults to `notes/nextgen/case-scoreboard.md`; `--scoreboard`
selects another path. Completed cases append a provenance row with clear,
both-cloudy, observed-cloudy and all-valid biases and correlation. Identical
case/score repeats are idempotent; conflicting results with the same identity
fail instead of silently rewriting history. A concurrent writer's lock is never
removed by another process.

Focused tests (no renderer, downloads or Rust build):

```powershell
python -B -m unittest discover -s scripts -p test_simsat_case_run.py -v
```
