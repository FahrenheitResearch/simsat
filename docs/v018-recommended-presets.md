# v0.1.8 recommended settings

SimSat Studio exposes three presets in the compact **Quick mode** row:
**Recommended**, **High Quality**, and **Sensor QA**. The row always names the
current reviewed mode, or **Custom** after any manual change. A preset is not a
hidden render profile: its tooltip and the single collapsed **Details** section
show the exact current before/after diff; after applying it, the same fields remain
editable and the resulting settings are saved immediately. The full control set is
under **Settings (all controls)** in a bounded scroller so opening configuration
does not squeeze the rendered image out of a laptop-sized window.

## Scope

- **Recommended Display** applies the owner-reviewed visible baseline to Visible,
  GeoColor Style, or Sandwich without changing the selected product, satellite,
  view, margin, timestep, or source.
- **High Quality Visible** is the same baseline with the owner-selected
  deterministic-4 fractional subcolumns and `0.45` cloud-highlight knee. It
  refuses Perspective because that operator is unsupported there; it does not
  silently change the camera or enable native/experimental optics.
- **Sensor QA** supports Visible and IR Band 13 only. It selects a geostationary
  GOES-R fixed grid and ABI sampling. Visible uses the explicitly limited Sensor
  Fast Gray operator with neutral display transforms. GOES-East IR uses the
  official FM4/GOES-19 Band 13 response, 2 km sampling, and a grayscale preview.
  It refuses Himawari, GOES-West Band 13, GeoColor Style, Sandwich, water-vapor,
  and derived products instead of mislabeling them as this sensor configuration.

All three presets select the CPU path and cancel a pending GPU parity pass. The
Studio's fake-sun override remains deliberately session-scoped and is not part of
the saved preset. For a reproducible production-safe baseline, all three also
select the default `CompactU8` brick storage and turn the experimental ABI Band 13
instrument footprint off. Those controls remain visible and can be enabled again
after applying a preset for an explicit experiment.

The Studio's **Storage / precision** section is product-independent. Its
ScienceCloudF16 switch remains available in visible, IR, water-vapor, and derived
modes and does not depend on the visible Clouds checkbox. ScienceCloudF16 is an
explicit CPU-only experiment; changing it selects an isolated cache and re-ingests
the original source.

The experimental ABI Band 13 footprint is available only on products with a Band
13 instrument stage. If the product is changed to Visible, water vapor, or a
derived field while that footprint is active, Studio turns it off immediately,
saves the safe state, and explains the change in status instead of leaving a
hidden incompatible control that can fail the next render.

## CLI equivalents

There is intentionally no second `preset=` parser in the CLI. The canonical
planner operates on Studio's persisted settings plus its session-only GPU state;
duplicating that policy in another parser would allow the two definitions to
drift. Use the equivalent explicit options below.

Recommended Display (retain the desired `sat=`, `view=`, and
`geo-navigation=` selections):

```powershell
simsat-render-frame input=<run> out=<image.png> product=visible backend=cpu `
  storage-profile=compact-u8 instrument-footprint=off `
  intent=display resolution=native steps=offline `
  aerosol-optical-depth=0.05 rh-aerosol-swelling=off `
  atmosphere-correction=on terrain-atmosphere=on `
  land-sza-normalization=on land-sza-max-gain=1.6 `
  land-dark-toe=on land-dark-toe-knee=0.08 land-dark-toe-gamma=0.65 `
  land-dark-toe-max-gain=1.5 clouds=on fractional-clouds=effective-od `
  cloud-optics=fixed cloud-optical-depth-scale=0.15 multiscatter=on `
  beer-powder=off granulation=off feather-exposed-domain-edges=on `
  topdown-stratiform-regularization=off topdown-cloud-footprint=off `
  exposure=1.5 ground-gain=1.0 cloud-softclip=0.65 cloud-highlight-max=1.25
```

For High Quality Visible, replace `fractional-clouds=effective-od` with
`fractional-clouds=deterministic-4` and set `cloud-softclip=0.45`.

Visible Sensor QA:

```powershell
simsat-render-frame input=<run> out=<image.png> product=visible backend=cpu `
  storage-profile=compact-u8 instrument-footprint=off `
  sat=goes-east view=geo geo-navigation=goes-r-abi resolution=abi1km `
  intent=sensor-fast-gray steps=offline aerosol-optical-depth=0.05 `
  rh-aerosol-swelling=off terrain-atmosphere=on `
  clouds=on fractional-clouds=effective-od cloud-optics=fixed `
  cloud-optical-depth-scale=1.0 multiscatter=on beer-powder=off `
  granulation=off feather-exposed-domain-edges=off `
  topdown-stratiform-regularization=off topdown-cloud-footprint=off `
  atmosphere-correction=off land-sza-normalization=off land-dark-toe=off `
  exposure=1.0 ground-gain=1.0 cloud-softclip=1.0 cloud-highlight-max=1.0
```

GOES-East IR Sensor QA (the CLI emits raw Kelvin with `bt-out=` when requested):

```powershell
simsat-render-ir input=<run> out=<image.png> bt-out=<kelvin.bin> `
  storage-profile=compact-u8 instrument-footprint=off `
  sat=goes-east view=geo geo-navigation=goes-r-abi resolution=abi2km `
  sensor=goes-r-abi-band13-fm4 enhancement=gray
```

## Python equivalents

Python keeps the same controls explicit rather than embedding a second preset
policy. The key differences are:

```python
rgb, geo = simsat.render_visible_rgb(
    input_path,
    storage_profile="compact-u8",
    backend="cpu",
    intent="display",
    resolution="native",
    aerosol_optical_depth=0.05,
    rh_aerosol_swelling=False,
    atmosphere_correction=True,
    terrain_atmosphere=True,
    land_sza_normalization=True,
    land_sza_max_gain=1.6,
    land_dark_toe=True,
    land_dark_toe_knee=0.08,
    land_dark_toe_gamma=0.65,
    land_dark_toe_max_gain=1.5,
    exposure=1.5,
    ground_gain=1.0,
    cloud_softclip=0.65,  # 0.45 for High Quality Visible
    cloud_highlight_max=1.25,
    steps="offline",
    clouds=True,
    fractional_clouds=True,
    fractional_cloud_mode="effective-od",  # "deterministic-4" for High Quality
    cloud_optical_depth_scale=0.15,
    cloud_optics="fixed",
    multiscatter=True,
    beer_powder=False,
    feather_exposed_domain_edges=True,
    granulation=False,
    topdown_stratiform_regularization=False,
    topdown_cloud_footprint=False,
)
```

For visible Sensor QA, use `intent="sensor-fast-gray"`,
`sat="goes-east"`, `view="geo"`, `geo_navigation="goes-r-abi"`,
`resolution="abi1km"`, `atmosphere_correction=False`, OD/exposure/highlight
values of `1.0`, and disable the land/feather/appearance switches as in the CLI
example. For raw GOES-East Band 13:

```python
bt, geo = simsat.render_ir(
    input_path,
    storage_profile="compact-u8",
    sat="goes-east",
    view="geo",
    geo_navigation="goes-r-abi",
    resolution="abi2km",
    sensor="goes-r-abi-band13-fm4",
    instrument_footprint="off",
)
```

Passing `enhancement="gray"` additionally returns the Studio-like grayscale RGB
preview; the first array remains the quantitative Kelvin plane.
