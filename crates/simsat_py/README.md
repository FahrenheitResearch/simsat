# simsat — Python binding

Physically-based simulated visible/IR satellite imagery from WRF output, as numpy arrays
you can plot with matplotlib/cartopy. A thin PyO3 wrapper over the Rust `simsat` engine.

## Build / install (Linux or WSL)

```
pip install maturin
cd crates/simsat_py
maturin build --release          # produces a `simsat-*.whl` in target/wheels/
pip install target/wheels/simsat-*.whl
# or, for local development:
maturin develop --release
```

The wheel is `abi3` (built against the CPython stable ABI), so ONE wheel works on any
CPython >= 3.8 — no per-version rebuild. `numpy` is the only runtime dependency.

## The nine render functions

Each takes a wrfout file OR a cached `run.json` as `input` (a wrfout is ingested to a
cached brick on first use, and the cache re-ingests automatically if the wrfout is
re-written over the same path) and returns `(array(s), georef)`:

| function | returns | what it is |
|---|---|---|
| `render_visible_rgb(input, ...)` | `H x W x 3` uint8 | the finished true-color visible image (the shipped display look) |
| `render_geocolor(input, ...)` | `H x W x 3` uint8 | GeoColor day/night blend: true-color by day, colored band-13 IR by night, crossfaded across the terminator (always meaningful, day or night) |
| `render_sandwich(input, ...)` | `H x W x 3` uint8 | Sandwich severe-convection composite: visible base + color-enhanced band-13 IR overlaid on the cold cloud tops (a daytime product) |
| `render_visible_bands(input, ...)` | `H x W x 3` float32 `[0,1]` | RAW per-channel reflectance (pre-tonemap) for custom RGB / band math |
| `render_ir(input, ...)` | `H x W` float32 KELVIN | RAW band-13 (10.3 um) brightness temperature; `enhancement=` adds a colored `H x W x 3` uint8 (`(bt, rgb, geo)`) |
| `render_water_vapor(input, band='6.2'\|'6.9'\|'7.3', ...)` | `H x W` float32 KELVIN | RAW water-vapor band 8/9/10 BT (upper/mid/lower-level moisture); `enhancement=` as `render_ir` |
| `render_precipitable_water(input, ...)` | `H x W` float32 mm | RAW precipitable water (column-integrated vapor); `colormap=True` adds a basic RGB |
| `render_cloud_top_temp(input, ...)` | `H x W` float32 KELVIN | RAW cloud-top temperature at the visible tau~1 level (`NaN` = clear); `colormap=True` as above |
| `render_cloud_optical_depth(input, ...)` | `H x W` float32 | RAW total-column visible cloud optical depth (clear = 0); `colormap=True` as above |

Common keyword args (all optional): `sat` (`goes-east`/`goes-west`/`himawari`), `view`
(`topdown` default / `geo`), `timestep=0`, `resolution` (`native` default), `margin=0.0`
(zoom-out fraction added on each side — real surrounding earth, clear sky, frames the
domain), `cache=<dir>`, `threads=<n>`. The visible-family functions additionally take
`exposure=1.6`, `multiscatter=True`, `steps` (`offline`/`interactive`), `clouds=True`,
`sun_elev`/`sun_az` (what-if sun override), `bluemarble=<path>` (single-file ground),
`bluemarble_month`, `bluemarble_download=True`. Thermal functions (`render_ir`,
`render_water_vapor`) take `enhancement=` (`cimss`/`bd`/`avn`/`funktop`/`rainbow`/`gray`);
the derived-field functions take `colormap=`.

### `threads=` (per-process render-thread cap)

`threads=N` caps the render worker threads via rayon. HONEST SEMANTICS: rayon's pool is
GLOBAL and built ONCE per process — the FIRST render call in a process fixes the count
(from `threads=`, or else the `RAYON_NUM_THREADS` environment variable); a different
`threads=` on a later call in the same process has NO effect. Under a
`ProcessPoolExecutor(max_workers=16)` pass `threads=1` (each worker process gets its own
pool) so 16 concurrent renders do not each grab every core.

## Logging

The binding is **silent by default**: the engine's diagnostic stderr lines (e.g.
`simsat ingest: run=... dims=... wall=...` progress, MAP_PROJ / moving-nest warnings)
are suppressed when the module is imported, so batch runs are not spammed. To see them:

- `simsat.set_verbose(True)` — enable at runtime (`set_verbose(False)` silences again), or
- set the environment variable `SIMSAT_LOG=1` (or `true`) before `import simsat`.

The messages go to **STDERR** and their text is unchanged from the CLI's, so existing
log-parsing scripts work when enabled. This switch gates diagnostic chatter only —
render-honesty reporting (`georef.time_is_fallback` / `ground_source` / `ground_status`
and their `UserWarning`s) is data, not logs, and is always active.

## The `georef` object

| attribute | value |
|---|---|
| `geo.view` | `'topdown'` or `'geo'` |
| `geo.extent` | `(x0, x1, y0, y1)` for `imshow` (row 0 = north; use `origin='upper'`) |
| `geo.extent_kind` | `'projection_meters'` (topdown) or `'lonlat_degrees'` (geo) |
| `geo.proj4` | a PROJ.4 string EXACTLY consistent with the extent's CRS |
| `geo.proj_kind` | `'lcc'` / `'stere'` / `'merc'` / `'latlon'` |
| `geo.crs_params` | dict of the PROJ keys + the raw WRF attributes |
| `geo.lat`, `geo.lon` | `float32` `H x W` geodetic mesh (for `pcolormesh`) |
| `geo.time_is_fallback` | `True` when the source had no parseable valid time and the render used the fabricated fallback date 2004-06-21 12:00 UT (the sun position / ground season are then NOT the run's real conditions) |
| `geo.ground_source` | `'2km'` / `'8km-fallback'` / `'flat-albedo'` / `'single-file'` / `'none'` — where the visible ground pixels came from (`'none'` for thermal / derived products) |
| `geo.ground_status` | list of ground-resolution status lines (downloads / fallbacks) |

Honesty behavior: a fabricated-date or downgraded-ground render also raises a Python
`UserWarning`; and a `bluemarble=<file>` that fails to load is a hard `RuntimeError`
(you named the file — silently rendering something else would be wrong output). An
out-of-range `timestep=` is a `RuntimeError` naming the file's valid range.

## Quick start

```python
import simsat
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pyproj

wrfout = "/path/to/wrfout_d01_2020-06-21_18:00:00"

# 1) Finished true-color RGB (H x W x 3 uint8), top-down and map-registered by default.
rgb, geo = simsat.render_visible_rgb(wrfout, sat="goes-east", view="topdown")
crs = ccrs.Projection(pyproj.CRS.from_proj4(geo.proj4))
ax = plt.axes(projection=crs)
ax.imshow(rgb, extent=geo.extent, transform=crs, origin="upper")

# 2) RAW brightness temperature in KELVIN (H x W float32) for your own colormap.
bt, geo = simsat.render_ir(wrfout)
ax.pcolormesh(geo.lon, geo.lat, bt, transform=ccrs.PlateCarree())
# or let SimSat color it:
bt, ir_rgb, geo = simsat.render_ir(wrfout, enhancement="cimss")

# 3) Day/night-safe composite + a derived moisture field, capped to one worker thread.
gc, geo = simsat.render_geocolor(wrfout, threads=1)
pw, geo = simsat.render_precipitable_water(wrfout, threads=1)
```

Pass `view="geo"` for the from-space geostationary view. See the repo's
`notes/wrf-runner-glue.md` for the full WRF-Runner integration.
