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

## The eleven render functions

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
| `render_cloud_layer(input, ...)` | `(H x W x 4 uint8, H x W float32, geo)` | the WEB-MAP cloud layer pair: cloud-only RGBA (straight alpha; `premultiplied=True` for the additive form) + the ground cloud-shadow MULTIPLY layer, on a Web-Mercator grid with `geo.mercator_corners` for a Mapbox ImageSource (top-down by definition; no ground is rendered — the host map is the ground) |
| `render_perspective(input, eye=(lat,lon,alt_m), look=(lat,lon,alt_m), fov=40, size=(1280,720), ...)` | `H x W x 3` uint8 | a FREE-PERSPECTIVE frame: an eye/look/fov pinhole camera through the same marches — the angled-3D flyover product (full composite over the Blue Marble ground; sky rays composite the atmosphere limb). `cloud_layer_only=True` returns `H x W x 4` uint8 (the cloud field alone, premultiplied alpha) for a host 3-D map with a matching camera. `geo.camera_pose` always carries the camera; a FLYOVER is N calls along your own eye/look path |

Common keyword args (all optional): `sat` (`goes-east`/`goes-west`/`himawari`), `view`
(`topdown` default / `geo`), `timestep=0`, `resolution` (`native` default), `margin=0.0`
(zoom-out fraction added on each side — real surrounding earth, clear sky, frames the
domain), `cache=<dir>`, `threads=<n>`. The visible-family functions additionally take
`exposure=1.0`, `multiscatter=True`, `beer_powder=False`, `granulation=False`,
`steps` (`offline`/`interactive`), `clouds=True`, `fractional_clouds=True`,
`sun_elev`/`sun_az` (what-if sun override), `bluemarble=<path>` (single-file ground),
`bluemarble_month`, `bluemarble_download=True`. Thermal functions (`render_ir`,
`render_water_vapor`) take `enhancement=` (`cimss`/`bd`/`avn`/`funktop`/`rainbow`/`gray`);
the derived-field functions take `colormap=`.

The visible-family functions (including raw visible bands, cloud layer, and perspective)
also expose the atmosphere/cloud QA controls directly:

| keyword | default | effect |
|---|---:|---|
| `aerosol_optical_depth` | `0.05` | aerosol AOD only; Rayleigh remains present at zero |
| `rh_aerosol_swelling` | `False` | apply the documented 1.5x aerosol-extinction multiplier |
| `atmosphere_correction` | `True` | product-facing daytime aerial-veil correction; `False` retains full modeled path airlight (other display transforms remain) |
| `terrain_atmosphere` | `True` | shorten atmosphere columns to the WRF terrain elevation |
| `fractional_clouds` | `True` | use model cloud fraction/subcolumns when available; `False` restores legacy horizontally-full cloudy cells |
| `cloud_optical_depth_scale` | `0.15` | owner-selected cross-file visible calibration, applied consistently in view/sun/ambient/shadow paths (`0.0..=4.0`; `1.0` = unscaled model extinction) |
| `beer_powder` | `False` | optional Schneider shaping of the direct cloud-sun term; does not change transmittance |
| `granulation` | `False` | display-only sub-grid cloud-edge erosion; quantitative bands/thermal/derived products remain unmodified |

Finished visible display products (`render_visible_rgb`, `render_geocolor`,
`render_sandwich`, `render_cloud_layer`, and `render_perspective`) also accept these
optional display-calibration overrides. Omitting them preserves the shipped engine
constants; they are intentionally absent from `render_visible_bands` so raw reflectance
cannot be changed by a tonemap choice.

| keyword | omitted behavior | effect |
|---|---:|---|
| `ground_gain` | shipped `1.0` | sun-gated daytime surface-radiance lift (`1.0` is neutral; accepted but irrelevant for ground-free cloud layers) |
| `cloud_softclip` | shipped `0.65` | highlight shoulder knee (`1.0` disables the shoulder/hard-clamps) |
| `cloud_highlight_max` | shipped `1.25` | physical reflectance factor mapped to display white; raising it retains structure in brighter cloud tops |

`cloud_optical_depth_scale` is a labeled calibration/sensitivity control: the shipped
`0.15` is an owner-selected visual calibration after broad cross-file review, not a
claimed physical optimum. It supersedes the earlier tied `0.20`/`0.30` midpoint candidate.
`1.0` preserves the model-derived extinction unchanged, and
`0.0` makes its visible optical effects transparent. It does not alter
`render_cloud_optical_depth`, which intentionally returns the unscaled physical input.
`fractional_clouds=True` consumes model cloud fraction when the input supplies it
and safely falls back to full-cell coverage otherwise; set it false for the legacy A/B.
`clouds=False` remains the explicit feature bypass, while `multiscatter=False`
disables the higher cloud-scattering octaves without changing cloud transmittance.
`beer_powder` and `granulation` are explicit opt-in appearance controls and remain off
unless requested.
Layer-only products accept the shared keywords for call-site consistency, but
`atmosphere_correction` and `terrain_atmosphere` have no surface atmosphere to modify there.

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
| `geo.extent_kind` | `'projection_meters'` (topdown), `'lonlat_degrees'` (geo), or `'webmercator_meters'` (cloud layer) |
| `geo.proj4` | a PROJ.4 string EXACTLY consistent with the extent's CRS |
| `geo.proj_kind` | `'lcc'` / `'stere'` / `'merc'` / `'latlon'` |
| `geo.crs_params` | dict of the PROJ keys + the raw WRF attributes |
| `geo.lat`, `geo.lon` | `float32` `H x W` geodetic mesh (for `pcolormesh`) |
| `geo.time_is_fallback` | `True` when the source had no parseable valid time and the render used the fabricated fallback date 2004-06-21 12:00 UT (the sun position / ground season are then NOT the run's real conditions) |
| `geo.ground_source` | `'2km'` / `'8km-fallback'` / `'flat-albedo'` / `'single-file'` / `'none'` — where the visible ground pixels came from (`'none'` for thermal / derived products) |
| `geo.ground_status` | list of ground-resolution status lines (downloads / fallbacks) |
| `geo.mercator_corners` | `render_cloud_layer` only: the four `(lon, lat)` image corners in the Mapbox ImageSource `coordinates` order (NW, NE, SE, SW); `None` for other products |
| `geo.camera_pose` | `render_perspective` only: the camera dict (`eye_lat`/`eye_lon`/`eye_alt_m`/`look_lat`/`look_lon`/`look_alt_m`/`fov_deg`/`width`/`height`); `geo.view` reads `'perspective'`. `None` for other products |

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

## Compositing the cloud layer over a Mapbox GL map

`render_cloud_layer` produces exactly what a Mapbox GL (or MapLibre GL) `image` source
needs: a north-up Web-Mercator-aligned RGBA image + its four corner lon/lats. Export the
arrays to PNGs (straight alpha — the default) and wire them up like this.

> UNTESTED-BY-US: the snippet below is written from Mapbox GL JS API knowledge, not from
> a run we performed against a live Mapbox map — validate in your own map before relying
> on it. The in-process composite proof (`web_layer::composite_over_basemap` / the
> `render_frame product=cloud-layer composite-out=` PNG) is what we verified.

```python
import imageio.v3 as iio
rgba, shadow, geo = simsat.render_cloud_layer(wrfout, sat="goes-east")
iio.imwrite("clouds.png", rgba)                                   # straight-alpha RGBA
iio.imwrite("clouds_shadow.png", (shadow * 255).astype("uint8"))  # 255 = no shadow
print(geo.mercator_corners)  # [(lon,lat) NW, NE, SE, SW] -> paste/serve to the map
```

```js
// Mapbox GL JS (or MapLibre): the shadow layer multiplies the basemap, then the cloud
// layer composites over it. `coordinates` is the ImageSource order: NW, NE, SE, SW.
const corners = [[lonNW, latNW], [lonNE, latNE], [lonSE, latSE], [lonSW, latSW]];
map.addSource('simsat-shadow', { type: 'image', url: 'clouds_shadow.png', coordinates: corners });
map.addLayer({
  id: 'simsat-shadow', type: 'raster', source: 'simsat-shadow',
  paint: { 'raster-opacity': 1.0 },
});
map.addSource('simsat-clouds', { type: 'image', url: 'clouds.png', coordinates: corners });
map.addLayer({
  id: 'simsat-clouds', type: 'raster', source: 'simsat-clouds',
  paint: { 'raster-opacity': 1.0, 'raster-fade-duration': 0 },
});
```

NOTE on the multiply blend: core Mapbox GL raster layers composite source-over only —
a grayscale shadow PNG drawn source-over will WASH the basemap toward gray, not darken
it. Hosts that support blend modes (MapLibre >= 3 style `raster-...` extensions, deck.gl
`BitmapLayer` with `parameters: {blend: true, blendFunc: [GL.DST_COLOR, GL.ZERO]}`, or
any custom-layer/canvas pipeline) should use a true MULTIPLY for `clouds_shadow.png`.
Where only source-over is available, an acceptable stand-in is an INVERTED shadow
(`1 - shadow`) as a black image with alpha = darkness: `rgba_shadow = [0, 0, 0,
(1 - shadow) * 255]` drawn source-over — that is mathematically the same darkening as
the multiply for a black overlay. The animation loop is N timesteps -> N image pairs ->
`source.updateImage({url, coordinates})` per frame.

## Perspective flyovers

A flyover is N independent `render_perspective` calls along YOUR camera path — there is
deliberately no path-scripting DSL (each frame is one render; script the path in Python):

```python
import numpy as np
for i, t in enumerate(np.linspace(0.0, 1.0, 120)):
    eye = (36.5 + 1.5 * t, -98.5 + 1.0 * t, 150_000 - 50_000 * t)  # your own path
    rgb, geo = simsat.render_perspective(
        wrfout, eye=eye, look=(39.0, -97.5, 0.0), fov=45, size=(1280, 720)
    )
    iio.imwrite(f"flyover_{i:04d}.png", rgb)
```

Parallax of high cloud against the ground is physical (true 3-D rays) — that is the 3D
look. `geo.camera_pose` records the camera on every frame.
