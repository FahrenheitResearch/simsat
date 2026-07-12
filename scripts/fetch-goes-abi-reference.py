#!/usr/bin/env python3
"""Select and optionally fetch an official NOAA GOES ABI QA reference.

The default mode is read-only: it lists the nearest objects in NOAA's public
S3 bucket and prints a JSON plan.  Nothing is downloaded until ``--download``
is supplied.  Existing files are immutable inputs: size/ETag/hash conflicts
are errors, never overwrite requests.

With ``--align``, the selected CMI/cloud-mask/COD products can be sampled onto
an exact north-up SimSat target latitude/longitude mesh.  Alignment is an
optional QA feature and clearly checks for numpy, netCDF4, pyproj, and (for a
GRIB target) eccodes.  It is not a production ingest path.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable


S3_XML_NAMESPACE = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
DEFAULT_PRODUCTS = ("ABI-L2-MCMIPC", "ABI-L2-ACMC", "ABI-L2-CODC")
SCAN_RE = re.compile(
    r"_s(?P<year>\d{4})(?P<doy>\d{3})(?P<hour>\d{2})"
    r"(?P<minute>\d{2})(?P<second>\d{2})(?P<fraction>\d*)"
    r"_e(?P<end_year>\d{4})(?P<end_doy>\d{3})(?P<end_hour>\d{2})"
    r"(?P<end_minute>\d{2})(?P<end_second>\d{2})(?P<end_fraction>\d*)"
)


@dataclasses.dataclass(frozen=True)
class S3Object:
    product: str
    key: str
    size: int
    etag: str
    last_modified: str
    scan_start: dt.datetime
    scan_end: dt.datetime

    @property
    def scan_midpoint(self) -> dt.datetime:
        return self.scan_start + (self.scan_end - self.scan_start) / 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--time",
        required=True,
        help="target valid time as RFC3339 UTC, for example 2026-07-10T21:00:00Z",
    )
    parser.add_argument(
        "--satellite",
        default="goes19",
        help="NOAA bucket satellite token (default: goes19)",
    )
    parser.add_argument(
        "--products",
        nargs="+",
        default=list(DEFAULT_PRODUCTS),
        help="ABI S3 product prefixes",
    )
    parser.add_argument(
        "--max-offset-seconds",
        type=float,
        default=600.0,
        help="maximum target-to-scan-midpoint separation (default: 600)",
    )
    parser.add_argument(
        "--selection-time",
        choices=("start", "midpoint"),
        default="start",
        help="rank scans by start or midpoint time (default: start)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="immutable destination directory (required for --download/--align)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="explicitly download the selected objects",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="explicitly write an aligned NPZ after verifying local source objects",
    )
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument(
        "--target-grid",
        type=Path,
        help="NPZ containing matching 2-D lat/lon arrays in north-up row order",
    )
    targets.add_argument(
        "--target-grib",
        type=Path,
        help="GRIB2 target grid (requires eccodes; normalized to north-up)",
    )
    parser.add_argument(
        "--aligned-name",
        default="abi-reference-aligned.npz",
        help="aligned NPZ basename (default: abi-reference-aligned.npz)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="write a natural-sqrt RGB preview beside the aligned NPZ (Pillow optional)",
    )
    parser.add_argument(
        "--manifest-name",
        default="source-manifest.json",
        help="relative immutable provenance filename",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="network timeout in seconds (default: 60)",
    )
    args = parser.parse_args()
    if (args.download or args.align) and args.output_dir is None:
        parser.error("--output-dir is required with --download or --align")
    if args.align and args.target_grid is None and args.target_grib is None:
        parser.error("--align requires --target-grid or --target-grib")
    if args.preview and not args.align:
        parser.error("--preview requires --align")
    if Path(args.manifest_name).name != args.manifest_name:
        parser.error("--manifest-name must be a basename")
    if Path(args.aligned_name).name != args.aligned_name:
        parser.error("--aligned-name must be a basename")
    if args.max_offset_seconds < 0:
        parser.error("--max-offset-seconds must be non-negative")
    return args


def parse_utc(value: str) -> dt.datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    result = dt.datetime.fromisoformat(text)
    if result.tzinfo is None:
        raise ValueError("target time needs an explicit UTC offset or Z suffix")
    return result.astimezone(dt.timezone.utc)


def scan_time(match: re.Match[str], prefix: str = "") -> dt.datetime:
    fraction = match.group(prefix + "fraction") or ""
    microsecond = int((fraction + "000000")[:6]) if fraction else 0
    year = int(match.group(prefix + "year"))
    doy = int(match.group(prefix + "doy"))
    return dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(
        days=doy - 1,
        hours=int(match.group(prefix + "hour")),
        minutes=int(match.group(prefix + "minute")),
        seconds=int(match.group(prefix + "second")),
        microseconds=microsecond,
    )


def object_from_xml(product: str, element: ET.Element) -> S3Object | None:
    key = element.findtext("s3:Key", namespaces=S3_XML_NAMESPACE) or ""
    match = SCAN_RE.search(key)
    if match is None:
        return None
    return S3Object(
        product=product,
        key=key,
        size=int(element.findtext("s3:Size", namespaces=S3_XML_NAMESPACE) or 0),
        etag=(element.findtext("s3:ETag", namespaces=S3_XML_NAMESPACE) or "").strip(
            '"'
        ),
        last_modified=element.findtext("s3:LastModified", namespaces=S3_XML_NAMESPACE)
        or "",
        scan_start=scan_time(match),
        scan_end=scan_time(match, "end_"),
    )


def hour_prefix(product: str, when: dt.datetime) -> str:
    return f"{product}/{when:%Y}/{when.timetuple().tm_yday:03d}/{when:%H}/"


def list_prefix(
    bucket: str, product: str, when: dt.datetime, timeout: float
) -> list[S3Object]:
    result: list[S3Object] = []
    continuation: str | None = None
    while True:
        parameters = {
            "list-type": "2",
            "prefix": hour_prefix(product, when),
            "max-keys": "1000",
        }
        if continuation is not None:
            parameters["continuation-token"] = continuation
        query = urllib.parse.urlencode(parameters)
        url = f"https://{bucket}.s3.amazonaws.com/?{query}"
        with urllib.request.urlopen(url, timeout=timeout) as response:
            root = ET.fromstring(response.read())
        for item in root.findall("s3:Contents", S3_XML_NAMESPACE):
            parsed = object_from_xml(product, item)
            if parsed is not None:
                result.append(parsed)
        truncated = (
            root.findtext("s3:IsTruncated", namespaces=S3_XML_NAMESPACE) or "false"
        ).lower() == "true"
        if not truncated:
            break
        continuation = root.findtext(
            "s3:NextContinuationToken", namespaces=S3_XML_NAMESPACE
        )
        if not continuation:
            raise RuntimeError(
                f"S3 listing for {product} was truncated without a continuation token"
            )
    return result


def choose_objects(
    bucket: str,
    products: Iterable[str],
    target: dt.datetime,
    platform: str,
    max_offset: float,
    timeout: float,
    selection_time: str,
) -> list[S3Object]:
    selected: list[S3Object] = []
    platform_token = f"_{platform.upper()}_"
    for product in products:
        candidates: dict[str, S3Object] = {}
        for delta_hours in (-1, 0, 1):
            hour = target + dt.timedelta(hours=delta_hours)
            for item in list_prefix(bucket, product, hour, timeout):
                if platform_token in item.key:
                    candidates[item.key] = item
        if not candidates:
            raise RuntimeError(f"no {product} objects found near {target.isoformat()}")
        instant = (
            (lambda item: item.scan_start)
            if selection_time == "start"
            else (lambda item: item.scan_midpoint)
        )
        nearest = min(
            candidates.values(),
            key=lambda item: abs((instant(item) - target).total_seconds()),
        )
        offset = abs((instant(nearest) - target).total_seconds())
        if offset > max_offset:
            raise RuntimeError(
                f"nearest {product} scan is {offset:.1f}s from target, "
                f"over the {max_offset:.1f}s limit"
            )
        selected.append(nearest)
    return selected


def hashes(path: Path) -> tuple[str, str]:
    sha = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), md5.hexdigest()


def verify_existing(path: Path, item: S3Object) -> tuple[str, str]:
    if path.stat().st_size != item.size:
        raise RuntimeError(
            f"refusing to replace {path.name}: existing size {path.stat().st_size} "
            f"does not match source size {item.size}"
        )
    sha, md5 = hashes(path)
    if item.etag and "-" not in item.etag and md5.lower() != item.etag.lower():
        raise RuntimeError(
            f"refusing to replace {path.name}: existing MD5 {md5} "
            f"does not match source ETag {item.etag}"
        )
    return sha, md5


def download_object(
    bucket: str,
    item: S3Object,
    destination: Path,
    timeout: float,
) -> tuple[str, str]:
    if destination.exists():
        return verify_existing(destination, item)
    url = f"https://{bucket}.s3.amazonaws.com/{urllib.parse.quote(item.key)}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=destination.name + ".",
            suffix=".part",
            dir=destination.parent,
            delete=False,
        ) as output:
            temp_path = Path(output.name)
            with urllib.request.urlopen(url, timeout=timeout) as response:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
                    sha.update(block)
                    md5.update(block)
        if temp_path.stat().st_size != item.size:
            raise RuntimeError(
                f"downloaded {temp_path.stat().st_size} bytes for {destination.name}; "
                f"expected {item.size}"
            )
        if (
            item.etag
            and "-" not in item.etag
            and md5.hexdigest().lower() != item.etag.lower()
        ):
            raise RuntimeError(
                f"download MD5 {md5.hexdigest()} does not match ETag {item.etag}"
            )
        os.replace(temp_path, destination)
        temp_path = None
        return sha.hexdigest(), md5.hexdigest()
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_immutable(path: Path, content: bytes) -> str:
    digest = hashlib.sha256(content).hexdigest()
    if path.exists():
        existing = path.read_bytes()
        if existing != content:
            raise RuntimeError(
                f"refusing to overwrite mismatched immutable output {path.name}; "
                "use a new empty output directory"
            )
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=path.name + ".", suffix=".part", dir=path.parent, delete=False
    ) as stream:
        temp = Path(stream.name)
        stream.write(content)
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return digest


def commit_generated(temp: Path, destination: Path) -> str:
    digest, _ = hashes(temp)
    if destination.exists():
        existing, _ = hashes(destination)
        if existing != digest:
            temp.unlink(missing_ok=True)
            raise RuntimeError(
                f"refusing to overwrite mismatched generated output {destination.name}"
            )
        temp.unlink(missing_ok=True)
        return digest
    os.replace(temp, destination)
    return digest


def require_alignment_dependencies(target_grib: bool) -> tuple[Any, Any, Any]:
    missing: list[str] = []
    try:
        import numpy as np
    except ImportError:
        np = None
        missing.append("numpy")
    try:
        from netCDF4 import Dataset
    except ImportError:
        Dataset = None
        missing.append("netCDF4")
    eccodes = None
    if target_grib:
        try:
            import eccodes
        except ImportError:
            missing.append("eccodes")
    if missing:
        raise RuntimeError(
            "--align optional dependencies are missing: " + ", ".join(missing)
        )
    return np, Dataset, eccodes


def geos_forward_numpy(
    np: Any, lon_deg: Any, lat_deg: Any, attrs: dict[str, Any]
) -> tuple[Any, Any]:
    """GOES-R geodetic lon/lat -> fixed-grid scan angles (radians).

    This is the forward form of the GOES-R PUG ellipsoidal equations and keeps
    node QA independent of pyproj.  The optional pyproj cross-check in
    ``align_reference`` verifies it where pyproj happens to be installed.
    """

    sweep = str(attrs.get("sweep_angle_axis", "x"))
    if sweep != "x":
        raise RuntimeError(f"only GOES sweep_angle_axis=x is supported, got {sweep!r}")
    req = float(attrs["semi_major_axis"])
    rpol = float(attrs["semi_minor_axis"])
    satellite_distance = float(attrs["perspective_point_height"]) + req
    lon0 = np.deg2rad(float(attrs["longitude_of_projection_origin"]))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    geocentric_lat = np.arctan((rpol * rpol) / (req * req) * np.tan(lat))
    eccentricity_sq = (req * req - rpol * rpol) / (req * req)
    radius = rpol / np.sqrt(1.0 - eccentricity_sq * np.cos(geocentric_lat) ** 2)
    delta_lon = lon - lon0
    earth_x = radius * np.cos(geocentric_lat) * np.cos(delta_lon)
    earth_y = radius * np.cos(geocentric_lat) * np.sin(delta_lon)
    earth_z = radius * np.sin(geocentric_lat)
    ray_x = satellite_distance - earth_x
    ray_y = -earth_y
    ray_z = earth_z
    ray_length = np.sqrt(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z)
    visible = (
        satellite_distance * earth_x
        > earth_x * earth_x
        + earth_y * earth_y
        + (req * req / (rpol * rpol)) * earth_z * earth_z
    )
    # GOES sweep-x instrument angles (PUG Vol. 3): x is the E/W scan
    # angle around the full ray norm; y is the N/S elevation around ray_x.
    scan_x = np.arcsin(np.clip(-ray_y / ray_length, -1.0, 1.0))
    scan_y = np.arctan2(ray_z, ray_x)
    finite = visible & np.isfinite(lon) & np.isfinite(lat)
    return np.where(finite, scan_x, np.nan), np.where(finite, scan_y, np.nan)


def masked_float(np: Any, value: Any) -> Any:
    return np.asarray(
        np.ma.asarray(value, dtype=np.float32).filled(np.nan), dtype=np.float32
    )


def normalize_grib_grid(
    np: Any, lat: Any, lon: Any, landmask: Any | None
) -> tuple[Any, Any, Any | None]:
    lon = ((lon + 180.0) % 360.0) - 180.0
    # SimSat top-down row zero is north.  ecCodes HRRR arrays start at the south.
    if float(np.nanmean(lat[0])) < float(np.nanmean(lat[-1])):
        lat = lat[::-1]
        lon = lon[::-1]
        if landmask is not None:
            landmask = landmask[::-1]
    middle = lat.shape[0] // 2
    if float(lon[middle, 0]) > float(lon[middle, -1]):
        lat = lat[:, ::-1]
        lon = lon[:, ::-1]
        if landmask is not None:
            landmask = landmask[:, ::-1]
    return lat, lon, landmask


def grid_from_grib(np: Any, eccodes: Any, path: Path) -> tuple[Any, Any, Any | None]:
    with path.open("rb") as stream:
        first = eccodes.codes_grib_new_from_file(stream)
        if first is None:
            raise RuntimeError(f"no GRIB messages in {path.name}")
        try:
            nx = int(eccodes.codes_get(first, "Nx"))
            ny = int(eccodes.codes_get(first, "Ny"))
            lat = np.asarray(
                eccodes.codes_get_array(first, "latitudes"), dtype=np.float64
            ).reshape(ny, nx)
            lon = np.asarray(
                eccodes.codes_get_array(first, "longitudes"), dtype=np.float64
            ).reshape(ny, nx)
        finally:
            eccodes.codes_release(first)
        landmask = None
        while True:
            message = eccodes.codes_grib_new_from_file(stream)
            if message is None:
                break
            try:
                if eccodes.codes_get(message, "shortName") == "lsm":
                    landmask = np.asarray(
                        eccodes.codes_get_array(message, "values"), dtype=np.float32
                    ).reshape(ny, nx)
                    break
            finally:
                eccodes.codes_release(message)
    return normalize_grib_grid(np, lat, lon, landmask)


def load_target_grid(
    np: Any, eccodes: Any, args: argparse.Namespace
) -> tuple[Any, Any, Any | None, dict[str, Any]]:
    if args.target_grid is not None:
        with np.load(args.target_grid) as data:
            if "lat" not in data or "lon" not in data:
                raise RuntimeError("target grid NPZ needs lat and lon arrays")
            lat = np.asarray(data["lat"], dtype=np.float64)
            lon = np.asarray(data["lon"], dtype=np.float64)
            landmask = (
                np.asarray(data["landmask"], dtype=np.float32)
                if "landmask" in data
                else None
            )
        source = args.target_grid
        source_type = "npz-lat-lon"
    else:
        lat, lon, landmask = grid_from_grib(np, eccodes, args.target_grib)
        source = args.target_grib
        source_type = "grib-lat-lon"
    if lat.ndim != 2 or lat.shape != lon.shape:
        raise RuntimeError(
            f"target lat/lon must be matching 2-D arrays, got {lat.shape}/{lon.shape}"
        )
    if landmask is not None and landmask.shape != lat.shape:
        raise RuntimeError("target landmask shape does not match lat/lon")
    source_sha, _ = hashes(source)
    metadata = {
        "source_basename": source.name,
        "source_sha256": source_sha,
        "source_type": source_type,
        "width": int(lat.shape[1]),
        "height": int(lat.shape[0]),
        "row_order": "north-first",
        "has_landmask": landmask is not None,
    }
    return lat, lon, landmask, metadata


def axis_and_field(np: Any, axis: Any, field: Any, dimension: int) -> tuple[Any, Any]:
    axis = np.asarray(axis, dtype=np.float64)
    if axis[0] > axis[-1]:
        axis = axis[::-1]
        field = np.flip(field, axis=dimension)
    return axis, field


def sample_regular(
    np: Any,
    field: Any,
    x_axis: Any,
    y_axis: Any,
    target_x: Any,
    target_y: Any,
    nearest: bool = False,
) -> Any:
    values = masked_float(np, field)
    x_axis, values = axis_and_field(np, x_axis, values, 1)
    y_axis, values = axis_and_field(np, y_axis, values, 0)
    inside = (
        np.isfinite(target_x)
        & np.isfinite(target_y)
        & (target_x >= x_axis[0])
        & (target_x <= x_axis[-1])
        & (target_y >= y_axis[0])
        & (target_y <= y_axis[-1])
    )
    ix = np.searchsorted(x_axis, target_x, side="right") - 1
    iy = np.searchsorted(y_axis, target_y, side="right") - 1
    ix = np.clip(ix, 0, len(x_axis) - 2)
    iy = np.clip(iy, 0, len(y_axis) - 2)
    if nearest:
        choose_x = np.abs(target_x - x_axis[ix]) > np.abs(target_x - x_axis[ix + 1])
        choose_y = np.abs(target_y - y_axis[iy]) > np.abs(target_y - y_axis[iy + 1])
        result = values[iy + choose_y.astype(np.intp), ix + choose_x.astype(np.intp)]
    else:
        x0 = x_axis[ix]
        x1 = x_axis[ix + 1]
        y0 = y_axis[iy]
        y1 = y_axis[iy + 1]
        wx = np.divide(
            target_x - x0, x1 - x0, out=np.zeros_like(target_x), where=x1 != x0
        )
        wy = np.divide(
            target_y - y0, y1 - y0, out=np.zeros_like(target_y), where=y1 != y0
        )
        q00 = values[iy, ix]
        q10 = values[iy, ix + 1]
        q01 = values[iy + 1, ix]
        q11 = values[iy + 1, ix + 1]
        result = (
            q00 * (1.0 - wx) * (1.0 - wy)
            + q10 * wx * (1.0 - wy)
            + q01 * (1.0 - wx) * wy
            + q11 * wx * wy
        )
    return np.where(inside, result, np.nan).astype(np.float32)


def find_selected(selected: list[S3Object], product: str) -> S3Object | None:
    return next((item for item in selected if item.product == product), None)


def align_reference(
    args: argparse.Namespace,
    selected: list[S3Object],
) -> dict[str, Any]:
    np, Dataset, eccodes = require_alignment_dependencies(args.target_grib is not None)
    lat, lon, landmask, target_metadata = load_target_grid(np, eccodes, args)
    output_dir = args.output_dir

    mcm_item = find_selected(selected, "ABI-L2-MCMIPC")
    if mcm_item is None:
        raise RuntimeError("--align requires ABI-L2-MCMIPC in --products")
    mcm_path = output_dir / Path(mcm_item.key).name
    with Dataset(mcm_path) as dataset:
        projection = dataset.variables["goes_imager_projection"]
        projection_attrs = {
            name: getattr(projection, name) for name in projection.ncattrs()
        }
        target_x, target_y = geos_forward_numpy(np, lon, lat, projection_attrs)
        projection_cross_check = None
        try:
            from pyproj import CRS, Transformer

            height = float(projection_attrs["perspective_point_height"])
            crs = CRS.from_cf(projection_attrs)
            transformer = Transformer.from_crs(4326, crs, always_xy=True)
            projected_x, projected_y = transformer.transform(lon, lat)
            check_x = projected_x / height
            check_y = projected_y / height
            finite = (
                np.isfinite(target_x)
                & np.isfinite(target_y)
                & np.isfinite(check_x)
                & np.isfinite(check_y)
            )
            projection_cross_check = {
                "engine": "pyproj optional cross-check",
                "finite_points": int(np.sum(finite)),
                "max_abs_x_rad": float(
                    np.max(np.abs(target_x[finite] - check_x[finite]))
                ),
                "max_abs_y_rad": float(
                    np.max(np.abs(target_y[finite] - check_y[finite]))
                ),
            }
        except ImportError:
            projection_cross_check = {
                "engine": "pyproj unavailable; pure NumPy PUG path used"
            }
        x_axis = np.asarray(dataset.variables["x"][:], dtype=np.float64)
        y_axis = np.asarray(dataset.variables["y"][:], dtype=np.float64)
        aligned: dict[str, Any] = {}
        for name in ("CMI_C01", "CMI_C02", "CMI_C03", "CMI_C13"):
            aligned[name.lower()] = sample_regular(
                np,
                dataset.variables[name][:],
                x_axis,
                y_axis,
                target_x,
                target_y,
            )
        dqf = sample_regular(
            np,
            dataset.variables["DQF_C02"][:],
            x_axis,
            y_axis,
            target_x,
            target_y,
            nearest=True,
        )
        aligned["dqf_c02"] = dqf

    acm_item = find_selected(selected, "ABI-L2-ACMC")
    if acm_item is not None:
        with Dataset(output_dir / Path(acm_item.key).name) as dataset:
            x_axis = np.asarray(dataset.variables["x"][:], dtype=np.float64)
            y_axis = np.asarray(dataset.variables["y"][:], dtype=np.float64)
            for source, destination in (
                ("BCM", "bcm"),
                ("ACM", "acm"),
                ("Cloud_Probabilities", "cloud_probability"),
            ):
                aligned[destination] = sample_regular(
                    np,
                    dataset.variables[source][:],
                    x_axis,
                    y_axis,
                    target_x,
                    target_y,
                    nearest=source in ("BCM", "ACM"),
                )

    cod_item = find_selected(selected, "ABI-L2-CODC")
    if cod_item is not None:
        with Dataset(output_dir / Path(cod_item.key).name) as dataset:
            x_axis = np.asarray(dataset.variables["x"][:], dtype=np.float64)
            y_axis = np.asarray(dataset.variables["y"][:], dtype=np.float64)
            aligned["cod"] = sample_regular(
                np,
                dataset.variables["COD"][:],
                x_axis,
                y_axis,
                target_x,
                target_y,
            )

    aligned["valid"] = (
        np.isfinite(aligned["cmi_c02"]) & (aligned["dqf_c02"] == 0)
    ).astype(np.uint8)
    aligned["lat"] = lat.astype(np.float32)
    aligned["lon"] = lon.astype(np.float32)
    if landmask is not None:
        aligned["landmask"] = landmask.astype(np.float32)

    destination = output_dir / args.aligned_name
    with tempfile.NamedTemporaryFile(
        prefix=destination.stem + ".",
        suffix=".npz.part",
        dir=output_dir,
        delete=False,
    ) as stream:
        temp = Path(stream.name)
    try:
        with temp.open("wb") as stream:
            np.savez_compressed(stream, **aligned)
        aligned_sha = commit_generated(temp, destination)
    finally:
        temp.unlink(missing_ok=True)

    preview_record = None
    if args.preview:
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError(
                "--preview optional dependency is missing: Pillow"
            ) from error
        red = np.clip(aligned["cmi_c02"], 0.0, 1.0)
        blue = np.clip(aligned["cmi_c01"], 0.0, 1.0)
        green = np.clip(0.45 * red + 0.10 * aligned["cmi_c03"] + 0.45 * blue, 0.0, 1.0)
        rgb = np.sqrt(np.stack([red, green, blue], axis=-1))
        rgb = np.rint(np.nan_to_num(rgb, nan=0.0) * 255.0).astype(np.uint8)
        preview = destination.with_suffix(".png")
        with tempfile.NamedTemporaryFile(
            prefix=preview.stem + ".", suffix=".png.part", dir=output_dir, delete=False
        ) as stream:
            preview_temp = Path(stream.name)
        try:
            Image.fromarray(rgb, mode="RGB").save(preview_temp, format="PNG")
            preview_sha = commit_generated(preview_temp, preview)
        finally:
            preview_temp.unlink(missing_ok=True)
        preview_record = {"file": preview.name, "sha256": preview_sha}

    result = {
        "file": destination.name,
        "sha256": aligned_sha,
        "target_grid": target_metadata,
        "variables": sorted(aligned),
        "method": {
            "projection": "pure NumPy GOES-R PUG ellipsoidal forward fixed-grid equations",
            "continuous": "bilinear on ABI fixed-grid scan coordinates",
            "categorical": "nearest on ABI fixed-grid scan coordinates",
            "synthetic_green": "0.45*C02 + 0.10*C03 + 0.45*C01",
        },
        "projection_cross_check": projection_cross_check,
    }
    if preview_record is not None:
        result["preview"] = preview_record
    return result


def public_record(bucket: str, item: S3Object, target: dt.datetime) -> dict[str, Any]:
    return {
        "product": item.product,
        "file": Path(item.key).name,
        "key": item.key,
        "url": f"https://{bucket}.s3.amazonaws.com/{urllib.parse.quote(item.key)}",
        "bytes": item.size,
        "etag": item.etag,
        "last_modified": item.last_modified,
        "scan_start": item.scan_start.isoformat().replace("+00:00", "Z"),
        "scan_end": item.scan_end.isoformat().replace("+00:00", "Z"),
        "scan_midpoint": item.scan_midpoint.isoformat().replace("+00:00", "Z"),
        "target_to_scan_start_seconds": (item.scan_start - target).total_seconds(),
        "target_to_scan_midpoint_seconds": (
            item.scan_midpoint - target
        ).total_seconds(),
    }


def main() -> int:
    args = parse_args()
    try:
        target = parse_utc(args.time)
        digits = re.search(r"(\d+)$", args.satellite)
        if digits is None:
            raise ValueError("--satellite must end in a GOES platform number")
        platform = "G" + digits.group(1)
        bucket = "noaa-" + args.satellite.lower()
        selected = choose_objects(
            bucket,
            args.products,
            target,
            platform,
            args.max_offset_seconds,
            args.timeout,
            args.selection_time,
        )
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "purpose": "SimSat observational QA; forecast clouds are compared distributionally, not pixel-matched",
            "provider": "NOAA/NESDIS official GOES open-data bucket",
            "bucket": bucket,
            "platform": platform,
            "target_time": target.isoformat().replace("+00:00", "Z"),
            "selection_time": args.selection_time,
            "objects": [public_record(bucket, item, target) for item in selected],
        }
        if not args.download and not args.align:
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0

        args.output_dir.mkdir(parents=True, exist_ok=True)
        for item, record in zip(selected, manifest["objects"], strict=True):
            path = args.output_dir / Path(item.key).name
            if args.download:
                sha, md5 = download_object(bucket, item, path, args.timeout)
            elif path.exists():
                sha, md5 = verify_existing(path, item)
            else:
                raise RuntimeError(
                    f"{path.name} is absent; rerun with explicit --download before --align"
                )
            record["sha256"] = sha
            record["md5"] = md5

        # fetched_at is intentionally omitted: an immutable source directory has a
        # byte-stable manifest, so a safe rerun can verify rather than overwrite it.
        manifest_path = args.output_dir / args.manifest_name
        manifest_sha = write_immutable(manifest_path, json_bytes(manifest))

        alignment_manifest = None
        alignment_manifest_sha = None
        if args.align:
            alignment_record = align_reference(args, selected)
            alignment_manifest = Path(args.aligned_name).with_suffix(".json").name
            alignment_manifest_sha = write_immutable(
                args.output_dir / alignment_manifest,
                json_bytes(
                    {
                        "schema_version": 1,
                        "source_manifest": args.manifest_name,
                        "source_manifest_sha256": manifest_sha,
                        "alignment": alignment_record,
                    }
                ),
            )
        print(
            json.dumps(
                {
                    "manifest": manifest_path.name,
                    "manifest_sha256": manifest_sha,
                    "objects": len(selected),
                    "aligned": bool(args.align),
                    "alignment_manifest": alignment_manifest,
                    "alignment_manifest_sha256": alignment_manifest_sha,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, RuntimeError, ValueError, urllib.error.URLError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
