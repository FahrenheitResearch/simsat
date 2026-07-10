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

Two camera modes for every product: the **from-space geostationary view**
(GOES-East / GOES-West / Himawari presets, CGMS fixed-grid scan geometry) and a
**top-down map view** registered to the WRF domain's own projection (drops
straight onto matplotlib/cartopy axes).

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
    sat=goes-east view=geo sun-elev=30
simsat-render-ir input=wrfout_d03_2025-06-21_02:15:00 out=ir.png \
    enhancement=rainbow
```

`simsat-render-frame` renders the visible/GeoColor/Sandwich composites;
`simsat-render-ir` renders IR, water vapor (`wv=6.2|6.9|7.3`), and the derived
fields (`derived=pw|ctt|cod`). Both take `key=value` arguments (run with
`--help` for the full list) and print a machine-readable `SUMMARY` line.

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
rgb, geo = simsat.render_visible_rgb("wrfout_d03_...", view="topdown")
ax.imshow(rgb, extent=geo.extent, origin="upper")
```

See [crates/simsat_py/README.md](crates/simsat_py/README.md) for the full API
(`render_visible_rgb`, `render_visible_bands`, `render_ir`,
`render_water_vapor`, `render_geocolor`, `render_sandwich`,
`render_precipitable_water`, `render_cloud_top_temp`,
`render_cloud_optical_depth`) and wheel-building instructions.

## Honest limitations

Clouds and weather exist only inside your WRF domain — the zoom-out margin shows
the real surrounding earth under a clear sky, not extrapolated weather. The
earth is spherical (R = 6370 km, WRF's own geometry): the standard is physical
plausibility, not pixel-level registration against real ABI imagery. GeoColor's
night side is the IR composite only — no city-lights layer yet. The IR and
water-vapor bands use gray band-averaged absorption coefficients (documented in
the code) rather than line-by-line radiative transfer. The shipping render path
is CPU (rayon-parallel; a native-resolution 800x800 composite renders in about a
second on a desktop CPU); GPU activation is in progress.

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
