#!/usr/bin/env python3
"""Align a calibrated GridSat-CONUS visible observation to a WRF grid.

The GridSat ``ch1`` variable is visible-channel reflectance. This QA helper
bilinearly samples it at WRF ``XLAT``/``XLONG``, flips the south-to-north WRF
row order into the renderer's north-up image order, and writes:

* an 8-bit grayscale PNG using the ABI ``sqrt(reflectance)`` display transfer;
* a little-endian f32 raw reflectance plane beside it; and
* a JSON sidecar with provenance, observation-time offsets, and percentiles.

It is deliberately a private QA helper, not a production ingest path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gridsat", type=Path, help="GridSat-CONUS NetCDF input")
    parser.add_argument("wrfout", type=Path, help="WRF file providing XLAT/XLONG")
    parser.add_argument("output", type=Path, help="north-up grayscale PNG output")
    return parser.parse_args()


def wrf_lat_lon(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Dataset(path) as ds:
        lat_var = ds.variables["XLAT"]
        lon_var = ds.variables["XLONG"]
        lat = np.asarray(lat_var[0] if lat_var.ndim == 3 else lat_var[:], dtype=np.float64)
        lon = np.asarray(lon_var[0] if lon_var.ndim == 3 else lon_var[:], dtype=np.float64)
    if lat.shape != lon.shape or lat.ndim != 2:
        raise ValueError(f"XLAT/XLONG must be matching 2-D grids, got {lat.shape}/{lon.shape}")
    return lat, lon


def ensure_ascending(
    lat: np.ndarray,
    lon: np.ndarray,
    *fields: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    result = [np.asarray(field) for field in fields]
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        result = [field[::-1, :] for field in result]
    if lon[0] > lon[-1]:
        lon = lon[::-1]
        result = [field[:, ::-1] for field in result]
    return lat, lon, result


def bilinear_sample(
    field: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    if len(src_lat) < 2 or len(src_lon) < 2:
        raise ValueError("source latitude/longitude axes need at least two cells")
    lat_step = float(src_lat[1] - src_lat[0])
    lon_step = float(src_lon[1] - src_lon[0])
    if lat_step <= 0.0 or lon_step <= 0.0:
        raise ValueError("source latitude/longitude axes must be ascending")

    jf = (target_lat - float(src_lat[0])) / lat_step
    ix = (target_lon - float(src_lon[0])) / lon_step
    inside = (
        (jf >= 0.0)
        & (jf <= len(src_lat) - 1)
        & (ix >= 0.0)
        & (ix <= len(src_lon) - 1)
    )
    j0 = np.clip(np.floor(jf).astype(np.int64), 0, len(src_lat) - 2)
    i0 = np.clip(np.floor(ix).astype(np.int64), 0, len(src_lon) - 2)
    wy = np.clip(jf - j0, 0.0, 1.0)
    wx = np.clip(ix - i0, 0.0, 1.0)

    q00 = field[j0, i0]
    q10 = field[j0, i0 + 1]
    q01 = field[j0 + 1, i0]
    q11 = field[j0 + 1, i0 + 1]
    sampled = (
        q00 * (1.0 - wx) * (1.0 - wy)
        + q10 * wx * (1.0 - wy)
        + q01 * (1.0 - wx) * wy
        + q11 * wx * wy
    )
    sampled[~inside] = np.nan
    return sampled.astype(np.float32)


def finite_percentiles(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0}
    points = np.percentile(finite, [1, 5, 50, 95, 99])
    return {
        "count": int(finite.size),
        "p01": float(points[0]),
        "p05": float(points[1]),
        "p50": float(points[2]),
        "p95": float(points[3]),
        "p99": float(points[4]),
        "max": float(np.max(finite)),
    }


def main() -> None:
    args = parse_args()
    wrf_lat, wrf_lon = wrf_lat_lon(args.wrfout)

    with Dataset(args.gridsat) as ds:
        src_lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        src_lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        reflectance = np.ma.filled(ds.variables["ch1"][0], np.nan).astype(np.float64)
        delta_minutes = np.ma.filled(ds.variables["delta_time"][0], np.nan).astype(np.float64)
        coverage_start = str(getattr(ds, "time_coverage_start", ""))
        coverage_end = str(getattr(ds, "time_coverage_end", ""))
        platform = str(getattr(ds, "platform", ""))
        product_id = str(getattr(ds, "id", args.gridsat.name))

    src_lat, src_lon, fields = ensure_ascending(
        src_lat,
        src_lon,
        reflectance,
        delta_minutes,
    )
    reflectance, delta_minutes = fields
    aligned = bilinear_sample(reflectance, src_lat, src_lon, wrf_lat, wrf_lon)
    aligned_delta = bilinear_sample(delta_minutes, src_lat, src_lon, wrf_lat, wrf_lon)

    # WRF j increases south-to-north; SimSat PNG row zero is north.
    aligned = aligned[::-1, :]
    aligned_delta = aligned_delta[::-1, :]
    display = np.sqrt(np.clip(aligned, 0.0, 1.0))
    png_bytes = np.rint(np.nan_to_num(display, nan=0.0) * 255.0).astype(np.uint8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(png_bytes, mode="L").save(args.output)
    raw_path = args.output.with_suffix(".f32")
    np.asarray(aligned, dtype="<f4").tofile(raw_path)

    finite_reflectance = aligned[np.isfinite(aligned)]
    metadata = {
        "source": str(args.gridsat.resolve()),
        "wrf_grid": str(args.wrfout.resolve()),
        "product_id": product_id,
        "platform": platform,
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
        "width": int(aligned.shape[1]),
        "height": int(aligned.shape[0]),
        "domain": {
            "lat_min": float(np.nanmin(wrf_lat)),
            "lat_max": float(np.nanmax(wrf_lat)),
            "lon_min": float(np.nanmin(wrf_lon)),
            "lon_max": float(np.nanmax(wrf_lon)),
        },
        "reflectance": finite_percentiles(aligned),
        "display_sqrt_reflectance": finite_percentiles(display),
        "saturated_fraction": float(np.mean(finite_reflectance >= 1.0)),
        "delta_time_minutes": finite_percentiles(aligned_delta),
        "raw_reflectance": str(raw_path.resolve()),
        "display_transform": "round(255 * sqrt(clamp(ch1, 0, 1)))",
    }
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
