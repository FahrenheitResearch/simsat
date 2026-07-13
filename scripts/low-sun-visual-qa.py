#!/usr/bin/env python3
"""Render and measure low-sun terrain/cloud regressions on real SimSat cases.

The report is deliberately distributional: it bins pixels by the *engine-exact*
local solar elevation, cloud optical depth, and cloud-top-temperature proxy.  It
does not treat a forecast as a pixel-perfect observation.  Display RGB is paired
with clouds-off RGB and raw pre-tonemap reflectance so exposure/tone changes can
be separated from radiance-physics changes.

Execution requires an installed ``simsat`` wheel plus numpy and Pillow.  The
default mode validates and prints the plan; ``--execute`` is required to render.
Outputs are immutable per directory and the website manifest contains no private
absolute paths.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
SOLAR_BINS = (
    ("sun-below-0", -90.0, 0.0),
    ("sun-0-2", 0.0, 2.0),
    ("sun-2-5", 2.0, 5.0),
    ("sun-5-8", 5.0, 8.0),
    ("sun-8-12", 8.0, 12.0),
    ("sun-12-16", 12.0, 16.0),
    ("sun-16-20", 16.0, 20.0),
    ("sun-20-30", 20.0, 30.0),
    ("sun-above-30", 30.0, 90.0),
)
COD_BINS = (
    ("cod-0-0p05", 0.0, 0.05),
    ("cod-0p05-0p3", 0.05, 0.3),
    ("cod-0p3-1", 0.3, 1.0),
    ("cod-1-3", 1.0, 3.0),
    ("cod-3-10", 3.0, 10.0),
    ("cod-above-10", 10.0, math.inf),
)

RAW_REFLECTANCE_KEYS = {
    "storage_profile",
    "intent",
    "sat",
    "geo_navigation",
    "view",
    "timestep",
    "resolution",
    "margin",
    "aerosol_optical_depth",
    "rh_aerosol_swelling",
    "atmosphere_correction",
    "terrain_atmosphere",
    "multiscatter",
    "cloud_multiscatter",
    "beer_powder",
    "steps",
    "fractional_clouds",
    "fractional_cloud_mode",
    "cloud_optical_depth_scale",
    "cloud_optics",
    "granulation",
    "sun_elev",
    "sun_az",
    "cache",
    "bluemarble",
    "bluemarble_month",
    "bluemarble_download",
}
DERIVED_KEYS = {
    "storage_profile",
    "sat",
    "view",
    "timestep",
    "resolution",
    "margin",
    "cache",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="low-sun QA JSON configuration")
    parser.add_argument(
        "--output-dir", type=Path, help="new immutable report directory"
    )
    parser.add_argument("--execute", action="store_true", help="render and analyze")
    parser.add_argument("--jobs", type=int, default=1, help="parallel case workers")
    parser.add_argument(
        "--threads", type=int, default=4, help="Rayon threads in each worker process"
    )
    parser.add_argument(
        "--implementation-id",
        default="working-tree",
        help="sanitized renderer implementation id stored in the report",
    )
    parser.add_argument(
        "--implementation-label",
        default="Working tree",
        help="human-readable renderer implementation label",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        dest="variants",
        help="limit execution to a configured variant id (repeatable)",
    )
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        return args
    if args.config is None:
        parser.error("--config is required")
    if args.execute and args.output_dir is None:
        parser.error("--output-dir is required with --execute")
    if args.jobs <= 0 or args.threads <= 0:
        parser.error("--jobs and --threads must be positive")
    return args


def require_dependencies() -> tuple[Any, Any, Any]:
    missing: list[str] = []
    try:
        import numpy as np
    except ImportError:
        np = None
        missing.append("numpy")
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        Image = ImageDraw = None
        missing.append("Pillow")
    if missing:
        raise RuntimeError("missing QA dependencies: " + ", ".join(missing))
    return np, Image, ImageDraw


def validate_id(value: str, context: str) -> str:
    if not ID_RE.fullmatch(value):
        raise ValueError(f"{context} must match {ID_RE.pattern}: {value!r}")
    return value


def expand_path(value: str, base: Path) -> Path:
    expanded = os.path.expanduser(os.path.expandvars(value))
    if re.search(r"%[^%]+%|\$\{[^}]+\}", expanded):
        raise ValueError(f"unresolved environment variable in {value!r}")
    path = Path(expanded)
    return path if path.is_absolute() else (base / path)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pretty_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def immutable_write(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite mismatched {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("config must be an object with schema_version 1")
    validate_id(str(config.get("id", "")), "config id")
    cases = config.get("cases")
    variants = config.get("variants")
    if not isinstance(cases, list) or len(cases) < 2:
        raise ValueError("config needs at least two chronological cases")
    if not isinstance(variants, list) or not variants:
        raise ValueError("config needs at least one display variant")
    seen: set[str] = set()
    prior_time: datetime | None = None
    for case in cases:
        case_id = validate_id(str(case.get("id", "")), "case id")
        if case_id in seen:
            raise ValueError(f"duplicate case id {case_id}")
        seen.add(case_id)
        stamp = parse_time(str(case.get("valid_time", "")))
        if prior_time is not None and stamp <= prior_time:
            raise ValueError("cases must be strictly chronological")
        prior_time = stamp
    seen.clear()
    for variant in variants:
        variant_id = validate_id(str(variant.get("id", "")), "variant id")
        if variant_id in seen:
            raise ValueError(f"duplicate variant id {variant_id}")
        seen.add(variant_id)
        exposure = float(variant.get("exposure"))
        if not math.isfinite(exposure) or exposure <= 0:
            raise ValueError(f"variant {variant_id} exposure must be positive")
    reference_case = str(config.get("reference_case", ""))
    if reference_case not in {str(item["id"]) for item in cases}:
        raise ValueError("reference_case must name one configured case")
    return config


def parse_time(value: str) -> datetime:
    text = value.strip().replace("_", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    stamp = datetime.fromisoformat(text)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def solar_terms(stamp: datetime) -> tuple[float, float, float]:
    """Return UTC hours, declination radians, and equation-of-time minutes."""

    stamp = stamp.astimezone(timezone.utc)
    year = stamp.year
    month = stamp.month
    if month <= 2:
        year -= 1
        month += 12
    a = math.floor(year / 100.0)
    b = 2.0 - a + math.floor(a / 4.0)
    ut_hours = (
        stamp.hour
        + stamp.minute / 60.0
        + (stamp.second + stamp.microsecond / 1.0e6) / 3600.0
    )
    jd = (
        math.floor(365.25 * (year + 4716.0))
        + math.floor(30.6001 * (month + 1.0))
        + stamp.day
        + b
        - 1524.5
        + ut_hours / 24.0
    )
    t = (jd - 2_451_545.0) / 36525.0
    mean_long = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0
    mean_anom = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    eccentricity = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    m = math.radians(mean_anom)
    center = (
        math.sin(m) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + math.sin(2.0 * m) * (0.019993 - 0.000101 * t)
        + math.sin(3.0 * m) * 0.000289
    )
    omega = 125.04 - 1934.136 * t
    apparent_long = (
        mean_long + center - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    )
    obliquity0 = (
        23.0
        + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0
    )
    obliquity = obliquity0 + 0.00256 * math.cos(math.radians(omega))
    obliquity_rad = math.radians(obliquity)
    declination = math.asin(
        math.sin(obliquity_rad) * math.sin(math.radians(apparent_long))
    )
    y = math.tan(obliquity_rad / 2.0) ** 2
    l0 = math.radians(mean_long)
    eq = (
        y * math.sin(2.0 * l0)
        - 2.0 * eccentricity * math.sin(m)
        + 4.0 * eccentricity * y * math.sin(m) * math.cos(2.0 * l0)
        - 0.5 * y * y * math.sin(4.0 * l0)
        - 1.25 * eccentricity * eccentricity * math.sin(2.0 * m)
    )
    return ut_hours, declination, 4.0 * math.degrees(eq)


def refraction_grid(np: Any, elevation: Any) -> Any:
    result = np.zeros_like(elevation, dtype=np.float64)
    finite = np.isfinite(elevation)
    middle = finite & (elevation <= 85.0) & (elevation > 5.0)
    low = finite & (elevation <= 5.0) & (elevation > -0.575)
    below = finite & (elevation <= -0.575)
    tangent = np.tan(np.radians(elevation))
    result[middle] = (
        58.1 / tangent[middle]
        - 0.07 / tangent[middle] ** 3
        + 0.000086 / tangent[middle] ** 5
    ) / 3600.0
    e = elevation[low]
    result[low] = (
        1735.0 + e * (-518.2 + e * (103.4 + e * (-12.79 + e * 0.711)))
    ) / 3600.0
    result[below] = (-20.772 / tangent[below]) / 3600.0
    return result


def noaa_solar_grid(np: Any, stamp: datetime, lat: Any, lon: Any) -> Any:
    ut_hours, declination, eqtime = solar_terms(stamp)
    valid = np.isfinite(lat) & np.isfinite(lon)
    hour_angle = np.mod(ut_hours * 60.0 + eqtime + 4.0 * lon, 1440.0) / 4.0 - 180.0
    lat_rad = np.radians(lat)
    ha_rad = np.radians(hour_angle)
    cos_zenith = np.clip(
        np.sin(lat_rad) * math.sin(declination)
        + np.cos(lat_rad) * math.cos(declination) * np.cos(ha_rad),
        -1.0,
        1.0,
    )
    geometric = 90.0 - np.degrees(np.arccos(cos_zenith))
    result = geometric + refraction_grid(np, geometric)
    return np.where(valid, result, np.nan)


def refraction_scalar(elevation: float) -> float:
    if elevation > 85.0:
        return 0.0
    tangent = math.tan(math.radians(elevation))
    if elevation > 5.0:
        arcsec = 58.1 / tangent - 0.07 / tangent**3 + 0.000086 / tangent**5
    elif elevation > -0.575:
        arcsec = 1735.0 + elevation * (
            -518.2 + elevation * (103.4 + elevation * (-12.79 + elevation * 0.711))
        )
    else:
        arcsec = -20.772 / tangent
    return arcsec / 3600.0


def noaa_solar_scalar(stamp: datetime, lat: float, lon: float) -> tuple[float, float]:
    ut_hours, declination, eqtime = solar_terms(stamp)
    hour_angle = (ut_hours * 60.0 + eqtime + 4.0 * lon) % 1440.0 / 4.0 - 180.0
    lat_rad = math.radians(lat)
    ha_rad = math.radians(hour_angle)
    cos_zenith = max(
        -1.0,
        min(
            1.0,
            math.sin(lat_rad) * math.sin(declination)
            + math.cos(lat_rad) * math.cos(declination) * math.cos(ha_rad),
        ),
    )
    zenith = math.acos(cos_zenith)
    geometric = 90.0 - math.degrees(zenith)
    refraction = refraction_scalar(geometric)
    sin_zenith = math.sin(zenith)
    if abs(sin_zenith) < 1.0e-9:
        azimuth = 180.0
    else:
        cos_az = max(
            -1.0,
            min(
                1.0,
                (math.sin(lat_rad) * cos_zenith - math.sin(declination))
                / (math.cos(lat_rad) * sin_zenith),
            ),
        )
        az_acos = math.degrees(math.acos(cos_az))
        azimuth = (
            (az_acos + 180.0) % 360.0 if hour_angle > 0.0 else (540.0 - az_acos) % 360.0
        )
    return geometric + refraction, azimuth


def solar_elevation_grid_fallback(
    np: Any,
    time_iso: str,
    lat: Any,
    lon: Any,
    sun_elev: float | None,
    sun_az: float | None,
) -> Any:
    """Exact Python fallback for the renderer surface-light LUT contract."""

    stamp = parse_time(time_iso)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if lat.shape != lon.shape or lat.ndim != 2 or not lat.size:
        raise ValueError("lat/lon must be same-shaped non-empty 2-D arrays")
    valid = np.isfinite(lat) & np.isfinite(lon)
    if not np.any(valid):
        raise ValueError("lat/lon grid has no finite coordinate pair")
    if sun_elev is None and sun_az is None:
        return noaa_solar_grid(np, stamp, lat, lon).astype(np.float32)
    clat = 0.5 * (float(np.min(lat[valid])) + float(np.max(lat[valid])))
    clon = 0.5 * (float(np.min(lon[valid])) + float(np.max(lon[valid])))
    real_elev, real_az = noaa_solar_scalar(stamp, clat, clon)
    elev = real_elev if sun_elev is None else float(sun_elev)
    az = real_az if sun_az is None else float(sun_az)
    e = math.radians(elev)
    a = math.radians(az)
    enu = np.array([math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e)])
    la = math.radians(clat)
    lo = math.radians(clon)
    east = np.array([-math.sin(lo), math.cos(lo), 0.0])
    north = np.array(
        [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)]
    )
    up = np.array(
        [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]
    )
    ecef = enu[0] * east + enu[1] * north + enu[2] * up
    ecef /= np.linalg.norm(ecef)
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    local_up_x = np.cos(lat_rad) * np.cos(lon_rad)
    local_up_y = np.cos(lat_rad) * np.sin(lon_rad)
    local_up_z = np.sin(lat_rad)
    dot_up = np.clip(
        local_up_x * ecef[0] + local_up_y * ecef[1] + local_up_z * ecef[2],
        -1.0,
        1.0,
    )
    return np.where(valid, np.degrees(np.arcsin(dot_up)), np.nan).astype(np.float32)


def git_state(repo: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "tracked_dirty": bool(git("status", "--short", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "branch": "unknown", "tracked_dirty": None}


def slug_number(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def raster_paths(root: Path, case_id: str, variant_id: str) -> dict[str, Path]:
    base = root / "renders" / case_id
    return {
        "on": base / f"{case_id}--{variant_id}--clouds-on.png",
        "off": base / f"{case_id}--{variant_id}--clouds-off.png",
    }


def render_case(payload: dict[str, Any]) -> dict[str, Any]:
    # Imported in the worker so Windows spawn gives each process its own one-time
    # Rayon pool and the parent can still run dependency-light plan/self-check modes.
    import numpy as np
    from PIL import Image
    import simsat

    case = payload["case"]
    config = payload["config"]
    output_dir = Path(payload["output_dir"])
    threads = int(payload["threads"])
    input_path = Path(case["input_path"])
    base_settings = dict(config.get("settings", {}))
    base_settings.pop("exposure", None)
    base_settings.pop("clouds", None)
    base_settings.pop("threads", None)
    for path_key in ("cache", "bluemarble"):
        if isinstance(base_settings.get(path_key), str):
            base_settings[path_key] = os.path.expanduser(
                os.path.expandvars(base_settings[path_key])
            )

    derived_kwargs = {
        key: value for key, value in base_settings.items() if key in DERIVED_KEYS
    }
    cod, georef = simsat.render_cloud_optical_depth(
        str(input_path), **derived_kwargs, threads=threads
    )
    ctt, _ = simsat.render_cloud_top_temp(
        str(input_path), **derived_kwargs, threads=threads
    )
    cod = np.asarray(cod, dtype=np.float32)
    ctt = np.asarray(ctt, dtype=np.float32)
    lat = np.asarray(georef.lat, dtype=np.float32)
    lon = np.asarray(georef.lon, dtype=np.float32)
    solar_function = getattr(simsat, "solar_elevation_grid", None)
    if callable(solar_function):
        solar_result = solar_function(
            str(case["valid_time"]),
            lat,
            lon,
            sun_elev=base_settings.get("sun_elev"),
            sun_az=base_settings.get("sun_az"),
        )
        solar_implementation = "engine-binding"
    else:
        solar_result = solar_elevation_grid_fallback(
            np,
            str(case["valid_time"]),
            lat,
            lon,
            base_settings.get("sun_elev"),
            base_settings.get("sun_az"),
        )
        solar_implementation = "python-parity-fallback"
    solar = np.asarray(
        solar_result[0] if isinstance(solar_result, tuple) else solar_result,
        dtype=np.float32,
    )
    if not (cod.shape == ctt.shape == lat.shape == lon.shape == solar.shape):
        raise RuntimeError(
            f"{case['id']} diagnostic shape mismatch: "
            f"cod={cod.shape} ctt={ctt.shape} lat={lat.shape} lon={lon.shape} sun={solar.shape}"
        )

    fields_dir = output_dir / "fields"
    fields_dir.mkdir(parents=True, exist_ok=True)
    fields_path = fields_dir / f"{case['id']}--fields.npz"
    np.savez_compressed(
        fields_path, cod=cod, ctt=ctt, lat=lat, lon=lon, solar_elevation=solar
    )

    raw_kwargs = {
        key: value
        for key, value in base_settings.items()
        if key in RAW_REFLECTANCE_KEYS
    }
    raw_on, _ = simsat.render_rgb_reflectance(
        str(input_path), **raw_kwargs, clouds=True, threads=threads
    )
    raw_off, _ = simsat.render_rgb_reflectance(
        str(input_path), **raw_kwargs, clouds=False, threads=threads
    )
    raw_path = fields_dir / f"{case['id']}--raw-reflectance.npz"
    np.savez_compressed(
        raw_path,
        clouds_on=np.asarray(raw_on, dtype=np.float32),
        clouds_off=np.asarray(raw_off, dtype=np.float32),
    )

    images: list[dict[str, Any]] = []
    for variant in config["variants"]:
        variant_id = str(variant["id"])
        kwargs = dict(base_settings)
        kwargs.update(variant.get("settings", {}))
        kwargs["exposure"] = float(variant["exposure"])
        paths = raster_paths(output_dir, str(case["id"]), variant_id)
        paths["on"].parent.mkdir(parents=True, exist_ok=True)
        rgb_on, _ = simsat.render_visible_rgb(
            str(input_path), **kwargs, clouds=True, threads=threads
        )
        rgb_off, _ = simsat.render_visible_rgb(
            str(input_path), **kwargs, clouds=False, threads=threads
        )
        on_array = np.asarray(rgb_on, dtype=np.uint8)
        off_array = np.asarray(rgb_off, dtype=np.uint8)
        Image.fromarray(on_array, mode="RGB").save(paths["on"], format="PNG")
        Image.fromarray(off_array, mode="RGB").save(paths["off"], format="PNG")
        images.append(
            {
                "variant_id": variant_id,
                "on": str(paths["on"]),
                "off": str(paths["off"]),
                "width": int(on_array.shape[1]),
                "height": int(on_array.shape[0]),
            }
        )

    run_manifest = json.loads(input_path.read_text(encoding="utf-8"))
    brick_name = str(
        run_manifest["timesteps"][int(base_settings.get("timestep", 0))]["file"]
    )
    brick_path = input_path.parent / brick_name
    return {
        "case_id": case["id"],
        "engine_version": str(simsat.__version__),
        "solar_implementation": solar_implementation,
        "fields": str(fields_path),
        "raw": str(raw_path),
        "images": images,
        "input_basename": input_path.parent.name,
        "manifest_sha256": hash_file(input_path),
        "brick_basename": brick_path.name,
        "brick_sha256": hash_file(brick_path),
    }


def rgb_and_luma(np: Any, Image: Any, path: Path) -> tuple[Any, Any]:
    with Image.open(path) as loaded:
        rgb = np.asarray(loaded.convert("RGB"), dtype=np.float64) / 255.0
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return rgb, luma


def stats(np: Any, values: Any) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0}
    q = np.quantile(values, [0.1, 0.5, 0.9])
    return {
        "count": int(values.size),
        "p10": float(q[0]),
        "p50": float(q[1]),
        "p90": float(q[2]),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def normalized_region_mask(
    np: Any, shape: tuple[int, int], crop: dict[str, Any]
) -> Any:
    ny, nx = shape
    x0 = max(0, min(nx - 1, int(math.floor(float(crop["x"]) * nx))))
    y0 = max(0, min(ny - 1, int(math.floor(float(crop["y"]) * ny))))
    x1 = max(
        x0 + 1, min(nx, int(math.ceil((float(crop["x"]) + float(crop["width"])) * nx)))
    )
    y1 = max(
        y0 + 1, min(ny, int(math.ceil((float(crop["y"]) + float(crop["height"])) * ny)))
    )
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def interval_mask(values: Any, lower: float, upper: float, first: bool = False) -> Any:
    if first:
        return values <= upper
    if math.isinf(upper):
        return values > lower
    return (values > lower) & (values <= upper)


def class_masks(np: Any, cod: Any, ctt: Any, valid: Any) -> dict[str, Any]:
    cloudy = valid & np.isfinite(cod) & (cod >= 1.0) & np.isfinite(ctt)
    return {
        "clear": valid & np.isfinite(cod) & (cod <= 0.05),
        "thin-unclassified": valid
        & np.isfinite(cod)
        & (((cod > 0.05) & (cod < 1.0)) | ((cod >= 1.0) & ~np.isfinite(ctt))),
        "warm-top": cloudy & (ctt >= 270.0),
        "mid-top": cloudy & (ctt >= 250.0) & (ctt < 270.0),
        "cold-top": cloudy & (ctt < 250.0),
    }


def masked_rgb_metrics(
    np: Any, rgb: Any, luma: Any, delta: Any, mask: Any
) -> dict[str, Any]:
    count = int(np.sum(mask))
    if count == 0:
        return {"count": 0}
    means = [float(np.mean(rgb[..., channel][mask])) for channel in range(3)]
    return {
        "count": count,
        "luminance": stats(np, luma[mask]),
        "cloud_excess": stats(np, delta[mask]),
        "darkening_below_neg0p02_fraction": float(np.mean(delta[mask] < -0.02)),
        "channel_mean": {"r": means[0], "g": means[1], "b": means[2]},
        "r_over_b": means[0] / means[2] if means[2] else None,
        "g_over_b": means[1] / means[2] if means[2] else None,
        "any_channel_clipped_fraction": float(
            np.mean(np.max(rgb[mask], axis=1) >= 254.0 / 255.0)
        ),
    }


def gaussian_high_pass(np: Any, tile: Any, sigma: float = 3.0) -> Any:
    radius = int(math.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(tile, ((0, 0), (radius, radius)), mode="reflect")
    horizontal = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"), 1, padded
    )
    padded = np.pad(horizontal, ((radius, radius), (0, 0)), mode="reflect")
    blurred = np.apply_along_axis(
        lambda column: np.convolve(column, kernel, mode="valid"), 0, padded
    )
    return tile - blurred


def tile_directionality(np: Any, tile: Any) -> dict[str, float]:
    hp = gaussian_high_pass(np, np.asarray(tile, dtype=np.float64))
    hp_std = float(np.std(hp))
    if hp_std < 1.0e-12:
        return {"high_pass_std": hp_std, "max_over_mean": 0.0, "angular_entropy": 1.0}
    ny, nx = hp.shape
    window = np.outer(np.hanning(ny), np.hanning(nx))
    power = np.abs(np.fft.fftshift(np.fft.fft2(hp * window))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(ny))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(nx))[None, :]
    radial = np.sqrt(fx * fx + fy * fy)
    band = (radial >= 1.0 / 64.0) & (radial <= 1.0 / 4.0)
    angles = np.mod(np.arctan2(fy, fx), math.pi)
    bins = np.floor(angles / math.pi * 36.0).astype(int).clip(0, 35)
    angular = np.bincount(bins[band], weights=power[band], minlength=36).astype(float)
    mean = float(np.mean(angular))
    ratio = float(np.max(angular) / mean) if mean > 0 else 0.0
    total = float(np.sum(angular))
    if total <= 0:
        entropy = 1.0
    else:
        probabilities = angular[angular > 0] / total
        entropy = float(-np.sum(probabilities * np.log(probabilities)) / math.log(36.0))
    return {"high_pass_std": hp_std, "max_over_mean": ratio, "angular_entropy": entropy}


def directional_artifacts(
    np: Any, values: Any, tile_size: int = 128, stride: int = 64
) -> dict[str, Any]:
    ny, nx = values.shape
    tiles: list[dict[str, Any]] = []
    ys = list(range(0, max(1, ny - tile_size + 1), stride))
    xs = list(range(0, max(1, nx - tile_size + 1), stride))
    if not ys or ys[-1] != ny - tile_size:
        ys.append(max(0, ny - tile_size))
    if not xs or xs[-1] != nx - tile_size:
        xs.append(max(0, nx - tile_size))
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            tile = values[y : y + tile_size, x : x + tile_size]
            if tile.shape != (tile_size, tile_size) or not np.isfinite(tile).all():
                continue
            metric = tile_directionality(np, tile)
            metric.update(
                {"x": int(x), "y": int(y), "width": tile_size, "height": tile_size}
            )
            metric["flagged"] = bool(
                metric["high_pass_std"] >= 0.005
                and metric["max_over_mean"] > 6.0
                and metric["angular_entropy"] < 0.85
            )
            tiles.append(metric)
    if not tiles:
        return {"tiles": 0, "flagged": False}
    ratios = np.asarray([item["max_over_mean"] for item in tiles])
    worst = max(tiles, key=lambda item: item["max_over_mean"])
    return {
        "tiles": len(tiles),
        "flagged": any(bool(item["flagged"]) for item in tiles),
        "ratio_p95": float(np.quantile(ratios, 0.95)),
        "worst": worst,
    }


def relative_asset(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def warning(
    warnings: list[dict[str, Any]],
    *,
    code: str,
    frame_id: str,
    summary: str,
    observed: float,
    operator: str,
    threshold: float,
    severity: str = "warning",
    variant_id: str | None = None,
    solar_bin: str | None = None,
    cloud_class: str | None = None,
    cod_bin: str | None = None,
) -> None:
    parts = [code.lower(), frame_id]
    for value in (variant_id, solar_bin, cloud_class, cod_bin):
        if value:
            parts.append(value)
    warnings.append(
        {
            "id": "--".join(parts),
            "code": code,
            "severity": severity,
            "summary": summary,
            "frameId": frame_id,
            "variantId": variant_id,
            "solarBin": solar_bin,
            "cloudClass": cloud_class,
            "codBin": cod_bin,
            "observed": observed,
            "operator": operator,
            "threshold": threshold,
        }
    )


def analyze(
    config: dict[str, Any], results: list[dict[str, Any]], output_dir: Path, repo: Path
) -> dict[str, Any]:
    np, Image, _ = require_dependencies()
    result_by_case = {str(item["case_id"]): item for item in results}
    minimum_samples = int(config.get("minimum_samples", 512))
    regions = config.get("regions", {})
    frames: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    loaded: dict[tuple[str, str], dict[str, Any]] = {}

    for case in config["cases"]:
        case_id = str(case["id"])
        rendered = result_by_case[case_id]
        with np.load(rendered["fields"]) as fields:
            cod = np.asarray(fields["cod"])
            ctt = np.asarray(fields["ctt"])
            solar = np.asarray(fields["solar_elevation"])
            lat = np.asarray(fields["lat"])
            lon = np.asarray(fields["lon"])
        with np.load(rendered["raw"]) as raw:
            raw_on = np.asarray(raw["clouds_on"], dtype=np.float64)
            raw_off = np.asarray(raw["clouds_off"], dtype=np.float64)
        valid = (
            np.isfinite(cod) & np.isfinite(solar) & np.isfinite(lat) & np.isfinite(lon)
        )
        masks = class_masks(np, cod, ctt, valid)
        raw_y_on = (
            0.2126 * raw_on[..., 0] + 0.7152 * raw_on[..., 1] + 0.0722 * raw_on[..., 2]
        )
        raw_y_off = (
            0.2126 * raw_off[..., 0]
            + 0.7152 * raw_off[..., 1]
            + 0.0722 * raw_off[..., 2]
        )
        raw_delta = raw_y_on - raw_y_off
        raw_metrics: dict[str, Any] = {"solarBins": {}}
        for bin_id, lower, upper in SOLAR_BINS:
            sun_mask = interval_mask(solar, lower, upper, first=bin_id == "sun-below-0")
            cell: dict[str, Any] = {}
            for class_id, class_mask in masks.items():
                combined = class_mask & sun_mask
                metric = masked_rgb_metrics(np, raw_on, raw_y_on, raw_delta, combined)
                metric["codBins"] = {
                    cod_id: masked_rgb_metrics(
                        np,
                        raw_on,
                        raw_y_on,
                        raw_delta,
                        combined
                        & interval_mask(
                            cod,
                            cod_lower,
                            cod_upper,
                            first=cod_id == "cod-0-0p05",
                        ),
                    )
                    for cod_id, cod_lower, cod_upper in COD_BINS
                }
                cell[class_id] = metric
            if lower >= 5.0:
                for cod_id, _, _ in COD_BINS:
                    warm = cell.get("warm-top", {}).get("codBins", {}).get(cod_id, {})
                    mid = cell.get("mid-top", {}).get("codBins", {}).get(cod_id, {})
                    warm_excess = warm.get("cloud_excess", {}).get("p50")
                    mid_excess = mid.get("cloud_excess", {}).get("p50")
                    if (
                        warm.get("count", 0) >= minimum_samples
                        and mid.get("count", 0) >= minimum_samples
                        and isinstance(warm_excess, float)
                        and isinstance(mid_excess, float)
                        and mid_excess > 0
                    ):
                        ratio = warm_excess / mid_excess
                        if ratio < 0.75:
                            warning(
                                warnings,
                                code="RAW_LOW_CLOUD_RELATIVE_DEFICIT",
                                frame_id=case_id,
                                solar_bin=bin_id,
                                cod_bin=cod_id,
                                summary="Pre-tonemap warm-cloud excess is small relative to mid cloud at matched COD",
                                observed=ratio,
                                operator="<",
                                threshold=0.75,
                            )
            raw_metrics["solarBins"][bin_id] = cell

        frame: dict[str, Any] = {
            "id": case_id,
            "timestampUtc": str(case["valid_time"]),
            "label": str(case.get("label", case_id)),
            "solarElevation": stats(np, solar[valid]),
            "input": {
                "manifest": rendered["input_basename"] + "/run.json",
                "manifestSha256": rendered["manifest_sha256"],
                "brick": rendered["brick_basename"],
                "brickSha256": rendered["brick_sha256"],
            },
            "rawReflectance": raw_metrics,
            "variants": [],
            "warningIds": [],
        }
        for image_record in rendered["images"]:
            variant_id = str(image_record["variant_id"])
            on_path = Path(image_record["on"])
            off_path = Path(image_record["off"])
            rgb_on, y_on = rgb_and_luma(np, Image, on_path)
            rgb_off, y_off = rgb_and_luma(np, Image, off_path)
            delta = y_on - y_off
            loaded[(case_id, variant_id)] = {
                "on": y_on,
                "off": y_off,
                "solar": solar,
                "valid": valid,
                "cod": cod,
            }
            variant: dict[str, Any] = {
                "id": variant_id,
                "label": next(
                    str(item.get("label", variant_id))
                    for item in config["variants"]
                    if item["id"] == variant_id
                ),
                "exposure": next(
                    float(item["exposure"])
                    for item in config["variants"]
                    if item["id"] == variant_id
                ),
                "image": relative_asset(on_path, output_dir),
                "surfaceImage": relative_asset(off_path, output_dir),
                "sha256": hash_file(on_path),
                "surfaceSha256": hash_file(off_path),
                "width": int(image_record["width"]),
                "height": int(image_record["height"]),
                "solarBins": {},
                "regions": {},
                "clippedFraction": float(
                    np.mean(np.max(rgb_on, axis=2) >= 254.0 / 255.0)
                ),
            }
            for region_id, region in regions.items():
                validate_id(str(region_id), "region id")
                region_mask = normalized_region_mask(np, y_on.shape, region["crop"])
                clear_region = valid & region_mask & masks["clear"]
                variant["regions"][region_id] = {
                    "label": str(region.get("label", region_id)),
                    "display": stats(np, y_on[valid & region_mask]),
                    "surfaceOnly": stats(np, y_off[valid & region_mask]),
                    "cloudExcess": stats(np, delta[valid & region_mask]),
                    "clearDisplay": stats(np, y_on[clear_region]),
                    "clearSurfaceOnly": stats(np, y_off[clear_region]),
                    "warmTopDisplay": stats(
                        np, y_on[valid & region_mask & masks["warm-top"]]
                    ),
                    "midTopDisplay": stats(
                        np, y_on[valid & region_mask & masks["mid-top"]]
                    ),
                    "coldTopDisplay": stats(
                        np, y_on[valid & region_mask & masks["cold-top"]]
                    ),
                }
            for bin_id, lower, upper in SOLAR_BINS:
                sun_mask = interval_mask(
                    solar, lower, upper, first=bin_id == "sun-below-0"
                )
                classes: dict[str, Any] = {}
                for class_id, class_mask in masks.items():
                    combined = class_mask & sun_mask
                    metric = masked_rgb_metrics(np, rgb_on, y_on, delta, combined)
                    cod_cells: dict[str, Any] = {}
                    for cod_id, cod_lower, cod_upper in COD_BINS:
                        cod_mask = interval_mask(
                            cod, cod_lower, cod_upper, first=cod_id == "cod-0-0p05"
                        )
                        cod_metric = masked_rgb_metrics(
                            np, rgb_on, y_on, delta, combined & cod_mask
                        )
                        cod_cells[cod_id] = cod_metric
                        csv_rows.append(
                            {
                                "frame": case_id,
                                "variant": variant_id,
                                "solar_bin": bin_id,
                                "cod_bin": cod_id,
                                "cloud_class": class_id,
                                "count": cod_metric.get("count", 0),
                                "luma_p50": cod_metric.get("luminance", {}).get("p50"),
                                "cloud_excess_p50": cod_metric.get(
                                    "cloud_excess", {}
                                ).get("p50"),
                                "darkening_fraction": cod_metric.get(
                                    "darkening_below_neg0p02_fraction"
                                ),
                            }
                        )
                    metric["codBins"] = cod_cells
                    classes[class_id] = metric
                    csv_rows.append(
                        {
                            "frame": case_id,
                            "variant": variant_id,
                            "solar_bin": bin_id,
                            "cod_bin": "all",
                            "cloud_class": class_id,
                            "count": metric.get("count", 0),
                            "luma_p50": metric.get("luminance", {}).get("p50"),
                            "cloud_excess_p50": metric.get("cloud_excess", {}).get(
                                "p50"
                            ),
                            "darkening_fraction": metric.get(
                                "darkening_below_neg0p02_fraction"
                            ),
                        }
                    )
                    if metric.get("count", 0) >= minimum_samples and lower >= 5.0:
                        excess = metric.get("cloud_excess", {}).get("p50")
                        dark_fraction = metric.get("darkening_below_neg0p02_fraction")
                        if (
                            class_id in {"warm-top", "mid-top"}
                            and isinstance(excess, float)
                            and excess <= 0.0
                        ):
                            warning(
                                warnings,
                                code="LOW_CLOUD_NEGATIVE_EXCESS",
                                frame_id=case_id,
                                variant_id=variant_id,
                                solar_bin=bin_id,
                                cloud_class=class_id,
                                summary="Warm/mid cloud median is no brighter than the clouds-off surface",
                                observed=excess,
                                operator="<=",
                                threshold=0.0,
                            )
                        if (
                            class_id in {"warm-top", "mid-top"}
                            and isinstance(dark_fraction, float)
                            and dark_fraction > 0.10
                        ):
                            warning(
                                warnings,
                                code="LOW_CLOUD_DARK_PIXEL_FRACTION",
                                frame_id=case_id,
                                variant_id=variant_id,
                                solar_bin=bin_id,
                                cloud_class=class_id,
                                summary="Too many warm/mid cloud pixels darken the underlying surface",
                                observed=dark_fraction,
                                operator=">",
                                threshold=0.10,
                            )
                warm = classes.get("warm-top", {})
                mid = classes.get("mid-top", {})
                warm_excess = warm.get("cloud_excess", {}).get("p50")
                mid_excess = mid.get("cloud_excess", {}).get("p50")
                if (
                    warm.get("count", 0) >= minimum_samples
                    and mid.get("count", 0) >= minimum_samples
                    and isinstance(warm_excess, float)
                    and isinstance(mid_excess, float)
                    and mid_excess > 0
                ):
                    ratio = warm_excess / mid_excess
                    if ratio < 0.75:
                        warning(
                            warnings,
                            code="LOW_CLOUD_RELATIVE_DEFICIT",
                            frame_id=case_id,
                            variant_id=variant_id,
                            solar_bin=bin_id,
                            summary="Warm-cloud linear excess is small relative to mid cloud",
                            observed=ratio,
                            operator="<",
                            threshold=0.75,
                        )
                for cod_id, _, _ in COD_BINS:
                    warm_cod = warm.get("codBins", {}).get(cod_id, {})
                    mid_cod = mid.get("codBins", {}).get(cod_id, {})
                    warm_cod_excess = warm_cod.get("cloud_excess", {}).get("p50")
                    mid_cod_excess = mid_cod.get("cloud_excess", {}).get("p50")
                    if (
                        warm_cod.get("count", 0) >= minimum_samples
                        and mid_cod.get("count", 0) >= minimum_samples
                        and isinstance(warm_cod_excess, float)
                        and isinstance(mid_cod_excess, float)
                        and mid_cod_excess > 0
                    ):
                        cod_ratio = warm_cod_excess / mid_cod_excess
                        if cod_ratio < 0.75:
                            warning(
                                warnings,
                                code="LOW_CLOUD_RELATIVE_DEFICIT_COD_MATCHED",
                                frame_id=case_id,
                                variant_id=variant_id,
                                solar_bin=bin_id,
                                cod_bin=cod_id,
                                summary="Warm-cloud excess is small relative to mid cloud at matched COD",
                                observed=cod_ratio,
                                operator="<",
                                threshold=0.75,
                            )
                variant["solarBins"][bin_id] = {
                    "lowerDeg": lower,
                    "upperDeg": upper,
                    "classes": classes,
                }
            if variant["clippedFraction"] > 0.001:
                warning(
                    warnings,
                    code="CLOUD_TOP_CLIPPING",
                    frame_id=case_id,
                    variant_id=variant_id,
                    summary="More than 0.1% of pixels clip at least one channel",
                    observed=variant["clippedFraction"],
                    operator=">",
                    threshold=0.001,
                )
            if bool(case.get("artifact_analysis", False)):
                artifacts = {
                    "display": directional_artifacts(np, y_on),
                    "surfaceOnly": directional_artifacts(np, y_off),
                    "logCod": directional_artifacts(np, np.log1p(np.nan_to_num(cod))),
                }
                variant["directionalArtifacts"] = artifacts
                if artifacts["display"].get("flagged"):
                    warning(
                        warnings,
                        code="DIRECTIONAL_ARTIFACT",
                        frame_id=case_id,
                        variant_id=variant_id,
                        summary="A 4–64 px directional pattern exceeds the automated tile threshold",
                        observed=float(artifacts["display"]["worst"]["max_over_mean"]),
                        operator=">",
                        threshold=6.0,
                    )
            frame["variants"].append(variant)
        frames.append(frame)

    # Same-grid surface-only ratios against the high-sun guardrail control for
    # geography/albedo. This answers whether lighting collapses, not whether a
    # dark forest or water pixel is intrinsically bright.
    reference_case = str(config["reference_case"])
    ratio_minima = config.get("surface_ratio_minimum", {})
    surface_curves: list[dict[str, Any]] = []
    for variant in config["variants"]:
        variant_id = str(variant["id"])
        reference = loaded[(reference_case, variant_id)]["off"]
        series: list[dict[str, Any]] = []
        for case in config["cases"]:
            case_id = str(case["id"])
            item = loaded[(case_id, variant_id)]
            stable = item["valid"] & np.isfinite(reference) & (reference > 0.03)
            ratio = np.divide(
                item["off"],
                reference,
                out=np.full_like(reference, np.nan),
                where=stable,
            )
            for bin_id, lower, upper in SOLAR_BINS:
                sun_mask = interval_mask(
                    item["solar"], lower, upper, first=bin_id == "sun-below-0"
                )
                measurement = stats(np, ratio[stable & sun_mask])
                point = {
                    "frameId": case_id,
                    "binId": bin_id,
                    "lowerDeg": lower,
                    "upperDeg": upper,
                    **measurement,
                }
                series.append(point)
                minimum = ratio_minima.get(bin_id)
                observed = measurement.get("p50")
                if (
                    case_id != reference_case
                    and isinstance(minimum, (int, float))
                    and measurement.get("count", 0) >= minimum_samples
                    and isinstance(observed, float)
                    and observed < float(minimum)
                ):
                    warning(
                        warnings,
                        code="TERRAIN_BELOW_REFERENCE",
                        frame_id=case_id,
                        variant_id=variant_id,
                        solar_bin=bin_id,
                        summary="Surface-only luminance collapsed relative to the same pixels at high sun",
                        observed=observed,
                        operator="<",
                        threshold=float(minimum),
                    )
        surface_curves.append({"variantId": variant_id, "points": series})

    temporal_steps: list[dict[str, Any]] = []
    max_rate = float(config.get("max_temporal_surface_step_per_10min", 0.08))
    temporal_region_id = config.get("temporal_region")
    for variant in config["variants"]:
        variant_id = str(variant["id"])
        for before_case, after_case in zip(config["cases"], config["cases"][1:]):
            before_id = str(before_case["id"])
            after_id = str(after_case["id"])
            elapsed_min = (
                parse_time(str(after_case["valid_time"]))
                - parse_time(str(before_case["valid_time"]))
            ).total_seconds() / 60.0
            before = loaded[(before_id, variant_id)]
            after = loaded[(after_id, variant_id)]
            shared = before["valid"] & after["valid"]
            if temporal_region_id:
                region = regions.get(str(temporal_region_id))
                if region is None:
                    raise ValueError(
                        f"temporal_region {temporal_region_id!r} is not configured"
                    )
                shared &= normalized_region_mask(
                    np, before["off"].shape, region["crop"]
                )
            delta = after["off"] - before["off"]
            measurement = stats(np, delta[shared])
            median_delta = measurement.get("p50")
            rate = (
                float(median_delta) * 10.0 / elapsed_min
                if isinstance(median_delta, float) and elapsed_min > 0
                else None
            )
            step = {
                "variantId": variant_id,
                "beforeFrameId": before_id,
                "afterFrameId": after_id,
                "elapsedMinutes": elapsed_min,
                "surfaceOnlyDelta": measurement,
                "medianDeltaPer10Minutes": rate,
                "regionId": temporal_region_id,
            }
            temporal_steps.append(step)
            if isinstance(rate, float) and abs(rate) > max_rate:
                warning(
                    warnings,
                    code="TEMPORAL_LIGHTING_STEP",
                    frame_id=after_id,
                    variant_id=variant_id,
                    summary="Surface-only luminance changes too abruptly for the elapsed time",
                    observed=abs(rate),
                    operator=">",
                    threshold=max_rate,
                )

    warning_ids_by_frame: dict[str, list[str]] = {}
    for item in warnings:
        warning_ids_by_frame.setdefault(str(item["frameId"]), []).append(
            str(item["id"])
        )
    for frame in frames:
        frame["warningIds"] = sorted(set(warning_ids_by_frame.get(frame["id"], [])))

    manifest = {
        "schemaVersion": 1,
        "id": config["id"],
        "status": "automated-diagnostic",
        "title": config.get("title", config["id"]),
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "assetRoot": "/renders/v020-low-sun",
        "engineVersion": sorted({str(item["engine_version"]) for item in results}),
        "implementation": config["implementation"],
        "sourceState": git_state(repo),
        "maskDefinition": {
            "version": "low-sun-v1",
            "clear": "COD <= 0.05",
            "thinUnclassified": "0.05 < COD < 1 or missing CTT",
            "warmTop": "COD >= 1 and CTT >= 270 K",
            "midTop": "COD >= 1 and 250 <= CTT < 270 K",
            "coldTop": "COD >= 1 and CTT < 250 K",
            "minimumSamples": minimum_samples,
            "solarGeometry": "engine surface-light LUT: per-pixel NOAA for actual sun; bbox-centre partial override -> ECEF -> per-pixel ENU for overrides",
            "solarImplementation": sorted(
                {str(item["solar_implementation"]) for item in results}
            ),
        },
        "variants": [
            {
                "id": item["id"],
                "label": item.get("label", item["id"]),
                "exposure": float(item["exposure"]),
                "role": item.get("role", "diagnostic"),
            }
            for item in config["variants"]
        ],
        "solarBins": [
            {"id": item[0], "lowerDeg": item[1], "upperDeg": item[2]}
            for item in SOLAR_BINS
        ],
        "codBins": [
            {
                "id": item[0],
                "lower": item[1],
                "upper": None if math.isinf(item[2]) else item[2],
            }
            for item in COD_BINS
        ],
        "surfaceReferenceCase": reference_case,
        "surfaceRatioCurves": surface_curves,
        "temporalSteps": temporal_steps,
        "warnings": warnings,
        "frames": frames,
    }
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    immutable_write(output_dir / "manifest.json", pretty_json(manifest))
    return manifest


def plan(config: dict[str, Any], config_path: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for case in config["cases"]:
        input_path = expand_path(str(case["input"]), config_path.parent)
        if not input_path.is_file():
            raise FileNotFoundError(f"missing case input: {input_path}")
        tasks.append({**case, "input_path": str(input_path)})
    return tasks


def self_check() -> None:
    np, _, _ = require_dependencies()
    smooth = np.tile(np.linspace(0.1, 0.9, 128), (128, 1))
    smooth_metric = tile_directionality(np, smooth)
    x = np.arange(128, dtype=float)[None, :]
    stripes = 0.4 + 0.05 * np.sin(2.0 * math.pi * x / 12.0)
    stripes = np.repeat(stripes, 128, axis=0)
    stripe_metric = tile_directionality(np, stripes)
    assert stripe_metric["high_pass_std"] >= 0.005
    assert stripe_metric["max_over_mean"] > 6.0
    assert stripe_metric["angular_entropy"] < 0.85
    assert smooth_metric["high_pass_std"] < stripe_metric["high_pass_std"]
    cod = np.array([[0.0, 0.1, 1.0, 2.0, 3.0]])
    ctt = np.array([[np.nan, np.nan, 275.0, 260.0, 230.0]])
    masks = class_masks(np, cod, ctt, np.ones_like(cod, dtype=bool))
    assert [
        int(np.sum(masks[name]))
        for name in ("clear", "thin-unclassified", "warm-top", "mid-top", "cold-top")
    ] == [1, 1, 1, 1, 1]
    lat = np.array([[30.0, 30.0, 30.0], [35.0, 35.0, 35.0], [40.0, 40.0, 40.0]])
    lon = np.array([[-100.0, -90.0, -80.0]] * 3)
    actual = solar_elevation_grid_fallback(
        np, "1974-04-03T23:12:00Z", lat, lon, None, None
    )
    overridden = solar_elevation_grid_fallback(
        np, "1974-04-03T23:12:00Z", lat, lon, 15.0, 180.0
    )
    assert np.isfinite(actual).all()
    assert 8.0 < float(actual[1, 1]) < 16.0
    assert abs(float(overridden[1, 1]) - 15.0) < 1.0e-5
    print(
        "low-sun QA self-check OK: "
        f"stripe ratio={stripe_metric['max_over_mean']:.2f} "
        f"entropy={stripe_metric['angular_entropy']:.3f}"
    )


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0
    assert args.config is not None
    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.variants:
        requested = set(args.variants)
        configured = {str(item["id"]) for item in config["variants"]}
        missing = sorted(requested - configured)
        if missing:
            raise ValueError(f"unknown --variant values: {missing}")
        config["variants"] = [
            item for item in config["variants"] if str(item["id"]) in requested
        ]
    config["implementation"] = {
        "id": validate_id(args.implementation_id, "implementation id"),
        "label": args.implementation_label,
    }
    tasks = plan(config, config_path)
    print(
        json.dumps(
            {
                "cases": len(tasks),
                "variants": len(config["variants"]),
                "visible_renders": len(tasks) * len(config["variants"]) * 2,
                "raw_reflectance_renders": len(tasks) * 2,
                "derived_fields": len(tasks) * 2,
                "jobs": args.jobs,
                "threads_per_job": args.threads,
            },
            indent=2,
        )
    )
    if not args.execute:
        return 0
    assert args.output_dir is not None
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = [
        {
            "case": task,
            "config": config,
            "output_dir": str(output_dir),
            "threads": args.threads,
        }
        for task in tasks
    ]
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(render_case, item): item["case"]["id"] for item in payloads
        }
        for future in concurrent.futures.as_completed(futures):
            case_id = futures[future]
            result = future.result()
            results.append(result)
            print(f"completed {case_id}", flush=True)
    order = {str(case["id"]): index for index, case in enumerate(config["cases"])}
    results.sort(key=lambda item: order[str(item["case_id"])])
    manifest = analyze(config, results, output_dir, Path(__file__).resolve().parents[1])
    print(
        json.dumps(
            {
                "manifest": str(output_dir / "manifest.json"),
                "frames": len(manifest["frames"]),
                "warnings": len(manifest["warnings"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"low-sun QA: {error}", file=sys.stderr)
        raise
