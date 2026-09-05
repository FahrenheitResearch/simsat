# Studio Earth colors and test cases

Open a model with **Open > Open wrfout / GRIB2**. Compatible inputs are WRF wrfout files, supported native-level GRIB2 files, and cached `run.json` manifests. A generic NetCDF or surface-only GRIB file may not have the 3D fields the renderer needs.

In **Settings (all controls) > Earth colors / ground**, choose the unshaded NASA monthly base-map folder, or extract the optional Earth-map pack beside the EXE and select **Use bundled Earth maps**. **Natural color (sRGB)** reproduces the reviewed natural display transform; **ABI display colors** retains the established display transform. Click Render to apply a change. The legacy map option remains available.

**Choose measured land JSON** loads a prepared MODIS NBAR `surface.json`. Use Visible + Display intent and the original WRF input. All source coordinates, model land mask, file checksums, and declared frame date are verified before rendering. Missing measured pixels use the selected base map; model snow overlays measured land in linear reflectance. The render log states the measured source date and quality counts. NBAR represents nadir/local-noon reflectance approximated as Lambertian RGB; historical cases can explicitly use a modern seasonal analogue. It is not a full directional or per-band ABI reflectance measurement.

## Local cases

Put named JSON files in `cases` beside the EXE to list them under **Open > Test cases**. Or choose **Open case JSON** from that menu. No model fields are decoded during listing. Relative input, cache, and Earth paths resolve beside the case JSON. Missing settings use Studio defaults. The JSON uses the stable settings tokens documented in `settings.rs`.

```json
{
  "name": "My daytime simulation",
  "input": "../models/wrfout_d01_2026-09-04_18_00_00",
  "cache": "../cache",
  "timestep": 0,
  "settings": {
    "view": "topdown",
    "resolution": "native",
    "mode": "visible",
    "render_intent": "display",
    "output_transform": "debug-srgb",
    "bm_allow_download": false,
    "earth": {"base_map": "../earth-basemap"}
  }
}
```

For night IR use `mode: "ir-band13"`; Earth inputs are unused. For measured land add `nbar_surface` inside `earth`. Source files are read; derived bricks are written into the cache directory. Large-file confirmation stays available in interactive mode.

```
simsat_studio --case cases/my-case.json
simsat_studio --case cases/my-case.json --render-and-save my-frame.png
```

The second command explicitly requests ingestion and rendering, including large files. It runs through the actual Studio window/render/export path, then exits; it does not save its batch settings into the interactive profile. Limit concurrent renders yourself; each Studio instance uses at most six CPU workers.

Cloud multi-order Monte Carlo work is an experimental research CLI. It is not enabled by opening a normal Studio case. Current research accuracy limits and numerical checks are recorded in `notes/nextgen/ABI-MONTE-CARLO-LIGHTING.md`.
