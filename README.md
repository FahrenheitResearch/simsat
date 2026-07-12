# SimSat Studio

Physically-based simulated satellite imagery from WRF model output.

SimSat renders what a geostationary weather satellite would see looking at your
WRF run: NASA Blue Marble ground with terrain shadows and seasonal blending, a
finite-disk sun with real twilight, volumetric clouds with multiple scattering,
Cox-Munk water sun-glint — and synthetic thermal infrared through the same
true-Kelvin enhancement pipeline the real GOES/Himawari products use. It ships
as a desktop app (`simsat_studio`, Rust/egui/wgpu), a headless CLI, and a
numpy-returning Python binding (`import simsat`).

## Gallery

All frames below are real SimSat renders of real WRF output (a North Dakota
supercell case and Hurricane Michael), thumbnailed for the README.

| | |
|---|---|
| ![True-color visible](docs/images/visible_geo.png) **True-color visible** — from-space geostationary view, volumetric clouds with cloud shadows | ![Hurricane + margin](docs/images/visible_hurricane.png) **Hurricane Michael** — with the zoom-out margin: the WRF domain framed by the real surrounding earth |
| ![Top-down map view](docs/images/visible_topdown.png) **Top-down map view** — north-up, registered to the WRF domain's own map projection | ![IR band 13](docs/images/ir_band13.png) **Infrared band 13 (10.3 um)** — true-Kelvin brightness temperature, Rainbow enhancement; works day and night |
| ![Water vapor](docs/images/wv_62.png) **Water vapor 6.2 um (band 8)** — upper-level moisture, classic WV palette | ![GeoColor](docs/images/geocolor.png) **GeoColor** — true-color day, IR night, blended across the real terminator (shown here at sunset) |
| ![Sandwich](docs/images/sandwich.png) **Sandwich** — visible texture with color-enhanced cold cloud tops (severe-convection view) | ![Precipitable water](docs/images/derived_pw.png) **Derived fields** — precipitable water, cloud-top temperature, cloud optical depth as raw map-registered arrays |

## Products

- **Visible true-color** — the full physical pipeline: Hillaire clear-sky
  atmosphere (transmittance / multiple-scattering / sky-view LUTs), volumetric
  cloud raymarch with Wrenninge multi-scatter octaves, SH-2 directional sky
  ambient, penumbral cloud and terrain shadows, seasonal Blue Marble ground,
  Cox-Munk glint, snow blend, ABI-style display transform.
- **Infrared band 13 (10.3 um)** — a real radiative-transfer march (gray-body
  Planck emission per voxel + surface term) inverted to true-Kelvin brightness
  temperature; enhancements: Grayscale, BD, Rainbow, CIMSS, AVN, Funktop.
- **Water vapor 6.2 / 6.9 / 7.3 um (bands 8/9/10)** — the same thermal march
  with water vapor as the dominant emitter; upper/mid/lower-level moisture.
- **GeoColor** — day/night composite: true-color by day, IR by night, crossfaded
  per pixel across the terminator.
- **Sandwich** — color-enhanced IR overlaid on the visible base over cold cloud
  tops (a daytime severe-convection product).
- **Derived fields** — precipitable water (mm), cloud-top temperature (K), and
  cloud optical depth as raw `f32` map-registered arrays.

Three camera modes: the **from-space geostationary view** (GOES-East /
GOES-West / Himawari presets, CGMS fixed-grid scan geometry), a **top-down map
view** registered to the WRF domain's own projection (drops straight onto
matplotlib/cartopy axes), and a **free perspective camera** (arbitrary
eye/look/FOV through the same physics — angled 3-D storm shots and flyovers;
interactive orbit controls in the studio, `eye=/look=/fov=` in the CLI,
`render_perspective` in Python). A **web map layer** product renders the cloud
field as a transparent EPSG:3857 overlay (straight-alpha clouds + a multiply
shadow layer) for Mapbox-class basemaps.

New in v0.1.6: **faster GPU preview and improved top-down rendering**. Studio has
a one-click GPU Render action, while Rust, the CLI, and Python expose the same
`gpu-preview` backend with every temporary compatibility adjustment reported.
Top-down renders now honor Model native, ABI 1 km, and ABI 2 km resolution, and
preserve the physical aspect ratio when capped. An opt-in, default-off
stratiform reconstruction control can reduce source-grid cloud rings in affected
HRRR fields without changing geostationary or raw-band output. Optional bounded
delta-flux cloud-transport experiments are also available across the interfaces.

New in v0.1.5: **brighter terrain and natural finite-domain cloud edges**.
The visible preset raises exposure while retaining highlight control, adds
sun-angle-aware land normalization, and feathers clouds only where the finite
model boundary is exposed. All presentation controls remain independently
switchable in Studio, the CLI, Python, and Rust.

New in v0.1.4: **model-aware fractional-cloud coverage and visible calibration**.
WRF `CLDFRA` is preserved where supplied; condensate cells with missing or
contradictory zero coverage are repaired with WRF's Xu-Randall diagnostic. HRRR
`wrfnat` now ingests its native 50-level cloud-fraction field. Both sources use
maximum-overlap vertical remapping, so anvil margins and cloud tails can fade
instead of filling every model cell as an opaque slab. Fractional coverage is on
by default and has an explicit legacy off switch in Rust, the CLI, Python, and
Studio. Sources without a complete trusted field retain the conservative
full-cell fallback.

The current visible preset uses cloud-OD scale `0.15`, exposure `1.5`, neutral ground lift `1.0`,
highlight knee `0.65`, and highlight ceiling `1.25`. The OD value is the owner's
cross-file visual selection, superseding the earlier tied `0.20`/`0.30` midpoint
candidate. It is not a claimed physical optimum; every value remains overridable. Raw visible bands, thermal
products, and derived cloud optical depth do not consume these display controls.
The SSB format is v5, so source-backed v0.1.3/v4 caches re-ingest once to acquire
the corrected cloud-fraction semantics; a cached-only run needs its original
source file to upgrade.

New in v0.1.3: **atmosphere and cloud fidelity controls** — terrain-height
atmospheric columns, consistent daytime aerial-veil correction across the
surface and cloud-front airlight, optically-thin multi-scatter gating, and a
bounded visible cloud optical-depth sensitivity scale. AOD, RH swelling,
atmosphere correction, terrain atmosphere, clouds, multiscatter, Beer-powder,
granulation, and cloud-OD controls are available in Studio, the CLI, and Python.

New in v0.1.2: **operational-model ingest** — NOAA **HRRR** native-level GRIB2
(`wrfnat`) opens directly in the studio/CLI/Python exactly like a wrfout;
**RRFS** (rotated lat-lon, `natlev`) ingests via the CLI with a regional crop.
Plus an experimental opt-in GPU cloud renderer (preview-only; stored frames
always use the tested CPU path).

## Quickstart (desktop app)

1. Launch SimSat Studio and open a `wrfout` file (or a whole sequence folder).
2. Confirm the ingest — the file is streamed into a compact quantized volume
   brick (an 800x800x80 ~2 GB wrfout ingests in ~15 s and caches for reuse).
3. Pick a satellite, view, and product; press **Render**.
4. For animations: open a sequence, press **Render sequence**, then scrub/play
   the in-studio timeline.

Frames and loops can also be written to a sat-store directory (a simple
grid + per-time frame layout) for downstream viewers.

## Headless CLI

Two named binaries render without a GPU or GUI (`cargo build --release --bins`):

```
simsat-render-frame input=wrfout_d03_2025-06-21_02:15:00 out=frame.png \
    sat=goes-east view=geo aod=0.05 rh-swelling=off \
    atmosphere-correction=on terrain-atmosphere=on fractional-clouds=on cloud-od-scale=0.15 \
    multiscatter=on beer-powder=off granulation=off feather-exposed-domain-edges=on clouds=on
simsat-render-ir input=wrfout_d03_2025-06-21_02:15:00 out=ir.png \
    enhancement=rainbow
```

`simsat-render-frame` renders the visible/GeoColor/Sandwich composites;
`simsat-render-ir` renders IR, water vapor (`wv=6.2|6.9|7.3`), and the derived
fields (`derived=pw|ctt|cod`). Both take `key=value` arguments (run with
`--help` for the full list) and print a machine-readable `SUMMARY` line.
Visible renders expose the same atmosphere/cloud QA controls as Studio and Python:
numeric aerosol AOD (`0` disables aerosol), RH swelling, reduced-versus-full
aerial airlight, terrain-height atmosphere, model fractional clouds, multiscatter,
beer-powder, granulation, exposed-domain edge feathering, clouds, and a shipped `0.15`
cloud optical-depth scale
(`1.0` is unscaled model extinction; `0` disables visible cloud extinction; valid
range `0.0..=4.0`). The `0.15` default is an owner-selected cross-file visual
calibration, not a claimed physical optimum. Finished RGB products also expose
`exposure=`, `ground-gain=`, `cloud-softclip=`, and `cloud-highlight-max=`;
omitting them keeps the shipped `1.5` exposure, neutral `1.0` ground gain, `0.65`
highlight knee, and `1.25` highlight ceiling.
`fractional-clouds=off` restores legacy horizontally-full cloudy cells. The scale
is an explicit sensitivity control and does not alter the raw derived cloud-optical-
depth product; beer-powder and granulation remain opt-in/off by default. Owner-selected
v0.1.5 edge feathering defaults on. `feather-exposed-domain-edges=on` reuses the fixed 4% cloud-edge
band when a finished visible or cloud-layer camera raster exposes the WRF boundary;
with it off, the pre-v0.1.5 positive-margin behavior is unchanged, and raw visible bands,
IR/WV, and derived products ignore the control.

Single-frame visible previews can explicitly select the shared Studio wgpu cloud pass
with `backend=gpu-preview` (`cpu` remains the default). The preview preserves `view=geo`
or `view=topdown`, reports every temporary compatibility substitution on stderr, and
never writes a sat-store frame. A missing adapter or unsupported rotated-lat/lon source
is an error rather than a silent CPU fallback.

Animated GIF loops are exported from a completed store run:

```
cargo run --release -p simsat --example export_animation -- \
    run=/path/to/store/simsat/mystorm_rgb_goese_20250621 out=loop.gif fps=8
```

GIF is palette-quantized (256 colors/frame) and its frame delays are quantized
to centiseconds — documented, honest format limits; no ffmpeg is bundled.

## Python binding

`import simsat` returns numpy arrays plus georeferencing (extent + PROJ string +
lat/lon mesh) for every product, ready for matplotlib/cartopy:

```python
import simsat
rgb, geo = simsat.render_visible_rgb(
    "wrfout_d03_...",
    backend="cpu",  # or "gpu-preview" for the synchronous Studio wgpu path
    view="topdown",
    aerosol_optical_depth=0.05,
    rh_aerosol_swelling=False,
    atmosphere_correction=True,
    terrain_atmosphere=True,
    land_sza_normalization=True,   # owner-selected v0.1.5 display default
    land_dark_toe=True,            # independently switchable; both false = legacy identity
    fractional_clouds=True,
    cloud_optical_depth_scale=0.15,
    beer_powder=False,
    granulation=False,
    feather_exposed_domain_edges=True,  # owner-selected v0.1.5 finite-domain default
    topdown_stratiform_regularization=False,  # opt-in top-down low-deck reconstruction
)
ax.imshow(rgb, extent=geo.extent, origin="upper")
```

See [crates/simsat_py/README.md](crates/simsat_py/README.md) for the full API
(`render_visible_rgb`, `render_visible_bands`, `render_ir`,
`render_water_vapor`, `render_geocolor`, `render_sandwich`,
`render_precipitable_water`, `render_cloud_top_temp`,
`render_cloud_optical_depth`) and wheel-building instructions.

The experimental `topdown_stratiform_regularization` switch is off by default. It is
a bounded, optical-depth-conserving observation-operator approximation for coarse-grid
low stratiform decks, not a literal satellite footprint or new microphysics, and it
cannot recover unresolved cloud/clear structure. Geostationary and raw-band products
ignore it; Studio falls back to CPU if a GPU preview cannot consume the reconstructed
field exactly.

## Honest limitations

Clouds and weather exist only inside your WRF domain — the zoom-out margin shows
the real surrounding earth under a clear sky, not extrapolated weather. The
earth is spherical (R = 6370 km, WRF's own geometry): the standard is physical
plausibility, not pixel-level registration against real ABI imagery. GeoColor's
night side is the IR composite only — no city-lights layer yet. The IR and
water-vapor bands use gray band-averaged absorption coefficients (documented in
the code) rather than line-by-line radiative transfer. The shipping render path
is CPU (rayon-parallel; a native-resolution 800x800 composite renders in about a
second on a desktop CPU). An explicit, display-only GPU cloud preview is available for
geostationary and top-down Visible frames; CPU remains the stored/batch quality path.

## Building from source

Requires a recent stable Rust toolchain (edition 2024, Rust 1.85+).

```
cargo build --release -p simsat_studio   # the desktop app
cargo build --release -p simsat --bins   # the headless CLI binaries
cargo test --workspace                   # the test suite (CPU-only, no GPU needed)
```

On Linux, the GUI/dialog dependencies need basic desktop development headers
(Debian/Ubuntu): `libxkbcommon-dev libwayland-dev libxcb-render0-dev
libxcb-shape0-dev libxcb-xfixes0-dev`.

The Python wheel builds from the standalone `crates/simsat_py` workspace:
`pip install maturin && cd crates/simsat_py && maturin build --release`.

## Ground imagery

The ground texture is NASA's Blue Marble Next Generation (2 km monthly
composites, blended to the day of year). Months download lazily at runtime and
are verified by SHA-256; a bundled 8 km composite is the offline fallback.
Imagery courtesy NASA Earth Observatory.

## License

Licensed under either of the [MIT license](LICENSE-MIT) or the
[Apache License, Version 2.0](LICENSE-APACHE), at your option. Third-party
license notices for the full dependency closure are in
[THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt).
