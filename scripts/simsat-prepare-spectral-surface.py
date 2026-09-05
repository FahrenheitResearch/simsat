#!/usr/bin/env python3
"""Prepare HAMSTER climatological surface spectra on an existing model grid.

This is an input-data stage for future spectral transport, not a render intent.
Output is dimensionless black-sky *surface albedo*, never ABI TOA reflectance or
RGB. HAMSTER (Roccetti et al. 2024, doi:10.5194/amt-17-6025-2024) reconstructs
400-2500 nm spectra from MODIS climatology and laboratory spectra. It does not
provide a directional BRDF, contemporary land state, or fine cloud structure.

The source may be a full published day or a coordinate-preserving regional
NetCDF subset. Read only a bounding spatial window around the target coordinates.
No observation radiances, cloud masks, or reference valid masks enter this stage.
The target must supply an explicit binary land_mask on the same grid. HAMSTER
contains positive placeholder ocean spectra, so positivity is not a land test.
An all-zero source spectrum is unavailable here (not a black Lambertian ocean);
water must be handled by the observation operator's own ocean surface model.
All four interpolation neighbors must have valid nonzero spectra. This avoids
filling source no-data; it does not remove mixed coastlines in the source grid.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

DATASET_DOI = "10.57970/04zd8-7et52"
PAPER_URL = "https://amt.copernicus.org/articles/17/6025/2024/"
LIMITATIONS = [
    "Surface black-sky albedo, not TOA reflectance; not connected to a renderer.",
    "2013-2022 climatology; no contemporary or 1974 land-state reconstruction.",
    "Black-sky albedo is not a directional BRDF or an albedo at arbitrary solar zenith.",
    "HAMSTER spectral reconstruction is not a measurement at every wavelength.",
    "No spectral extrapolation, ocean filling, coastal filling, or spatial detail synthesis.",
    "No source per-pixel retrieval quality flags are available in this HAMSTER file contract.",
    "Target ocean is masked by the supplied land_mask; source coastal pixels may be mixed at 0.05 degree resolution.",
]


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def ascending_axis(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite 1D coordinate with at least two samples")
    if np.all(np.diff(values) > 0):
        return values, False
    if np.all(np.diff(values) < 0):
        return values[::-1], True
    raise ValueError(f"{name} must be strictly monotonic")


def bracket_window(axis, target):
    finite = np.asarray(target)[np.isfinite(target)]
    if not finite.size:
        raise ValueError("target contains no finite coordinates")
    lo = max(0, int(np.searchsorted(axis, finite.min(), side="right")) - 1)
    hi = min(len(axis), int(np.searchsorted(axis, finite.max(), side="left")) + 1)
    if hi - lo < 2:
        raise ValueError("source does not overlap the target coordinate range")
    return lo, hi


def source_slice(window, length, reversed_axis):
    lo, hi = window
    return slice(length-hi, length-lo) if reversed_axis else slice(lo, hi)


def resample_spectra(latitude, longitude, spectra, target_lat, target_lon):
    """Bilinear surface spectra with strict all-neighbor validity, preserving order."""
    y, yr = ascending_axis(latitude, "latitude")
    x, xr = ascending_axis(longitude, "longitude")
    values = np.asarray(spectra, dtype=np.float64)
    if values.ndim != 3 or values.shape[:2] != (len(y), len(x)):
        raise ValueError("spectra must have shape latitude, longitude, wavelength")
    if yr:
        values = values[::-1]
    if xr:
        values = values[:, ::-1]
    tlat, tlon = np.asarray(target_lat, dtype=float), np.asarray(target_lon, dtype=float)
    if tlat.ndim != 2 or tlat.shape != tlon.shape:
        raise ValueError("target latitude and longitude must be matching 2D arrays")
    finite = np.isfinite(tlat) & np.isfinite(tlon)
    valid = finite & (tlat >= y[0]) & (tlat <= y[-1]) & (tlon >= x[0]) & (tlon <= x[-1])
    safe_y, safe_x = np.where(finite, tlat, y[0]), np.where(finite, tlon, x[0])
    iy = np.clip(np.searchsorted(y, safe_y, side="right")-1, 0, len(y)-2)
    ix = np.clip(np.searchsorted(x, safe_x, side="right")-1, 0, len(x)-2)
    fy = (safe_y-y[iy]) / (y[iy+1]-y[iy])
    fx = (safe_x-x[ix]) / (x[ix+1]-x[ix])
    source_valid = np.all(np.isfinite(values) & (values >= 0) & (values <= 1), axis=2) & np.any(values > 0, axis=2)
    result = np.zeros((*tlat.shape, values.shape[2]), dtype=np.float64)
    for dy, dx, weight in ((0,0,(1-fy)*(1-fx)), (0,1,(1-fy)*fx), (1,0,fy*(1-fx)), (1,1,fy*fx)):
        valid &= source_valid[iy+dy, ix+dx]
        result += np.nan_to_num(values[iy+dy, ix+dx], nan=0, posinf=0, neginf=0)*weight[...,None]
    result[~valid] = np.nan
    return result.astype(np.float32), valid.astype(np.uint8)


def prepare(source, grid, output, doy):
    source, grid, output = Path(source), Path(grid), Path(output)
    if output.exists():
        raise ValueError("output directory must be new")
    if not 1 <= doy <= 365:
        raise ValueError("HAMSTER climatology day must be in 1..365")
    with np.load(grid, allow_pickle=False) as target:
        lat = np.asarray(target["lat"], dtype=np.float64)
        lon = np.asarray(target["lon"], dtype=np.float64)
        land = np.asarray(target["land_mask"])
    if lat.ndim != 2 or lat.shape != lon.shape:
        raise ValueError("target latitude and longitude must be matching 2D arrays")
    if land.shape != lat.shape or not np.isin(land, [0,1]).all():
        raise ValueError("land_mask must be binary and have the exact target shape")
    with Dataset(source) as f:
        declared_doy = getattr(f, "source_doy", None)
        if declared_doy is not None and int(declared_doy) != doy:
            raise ValueError("source day does not match requested climatology day")
        # Published coordinate variables have capitalized names; lowercase names
        # in the HDF5 file are dimension scales, not populated geolocation arrays.
        sy, yr = ascending_axis(f["Latitude"][:], "latitude")
        sx, xr = ascending_axis(f["Longitude"][:], "longitude")
        wl, wr = ascending_axis(f["Wavelength"][:], "wavelength")
        if wr or wl[0] < 400 or wl[-1] > 2500:
            raise ValueError("HAMSTER wavelength axis must increase within 400..2500 nm")
        wy, wx = bracket_window(sy, lat), bracket_window(sx, lon)
        var = f["Black_Sky_Albedo"]
        if var.shape != (len(sy), len(sx), len(wl)):
            raise ValueError("source variable shape does not match coordinates")
        values = np.ma.filled(var[source_slice(wy,len(sy),yr),source_slice(wx,len(sx),xr),:], np.nan)
        if yr: values = values[::-1]
        if xr: values = values[:,::-1]
        mapped, valid = resample_spectra(sy[slice(*wy)], sx[slice(*wx)], values, lat, lon)
        valid &= (land == 1).astype(np.uint8)
        mapped[valid == 0] = np.nan
    if not valid.any():
        raise ValueError("no target pixels have a complete valid surface spectrum")
    output.mkdir(parents=True)
    np.savez_compressed(output/"spectral-surface-aligned.npz", latitude=lat, longitude=lon,
                        wavelength_um=wl/1000, black_sky_albedo=mapped, valid=valid,
                        land_mask=land.astype(np.uint8))
    report = dict(schema_version=1, quantity="climatological_black_sky_surface_albedo",
                  created_utc=datetime.now(timezone.utc).isoformat(), climatology_doy=doy,
                  dataset_doi=DATASET_DOI, paper=PAPER_URL, license="CC BY 4.0",
                  source=dict(path=str(source.resolve()), sha256=sha256(source)),
                  target_grid=dict(path=str(grid.resolve()), sha256=sha256(grid),
                                   fields_used=["lat","lon","land_mask"], orientation="preserved exactly from target"),
                  sampling="bilinear with all four neighbors valid across every supplied wavelength",
                  shape=list(mapped.shape), wavelength_um=[float(wl[0]/1000), float(wl[-1]/1000)],
                  valid_count=int(valid.sum()), land_count=int(land.sum()), total_count=int(valid.size), limitations=LIMITATIONS,
                  output_sha256=sha256(output/"spectral-surface-aligned.npz"))
    (output/"provenance.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="local HAMSTER day or coordinate-preserving subset")
    p.add_argument("--grid", type=Path, required=True, help="NPZ containing matching 2D lat, lon, and binary land_mask")
    p.add_argument("--doy", type=int, required=True, help="365-day climatology index; caller resolves calendar mapping")
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    try:
        r = prepare(a.source, a.grid, a.output_dir, a.doy)
    except (ValueError, OSError, KeyError) as e:
        p.exit(2, f"spectral surface preparation failed: {e}\n")
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
