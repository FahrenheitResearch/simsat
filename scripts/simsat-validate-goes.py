#!/usr/bin/env python3
"""Validate an exact-grid SimSat visible or ABI Band 13 render against GOES.

This command consumes the immutable ``abi-reference-aligned.npz`` produced by
``fetch-goes-abi-reference.py --align``.  It does not fetch or resample data.
Visible synthetic input may be a finished RGB PNG or the north-first,
interleaved f32le RGB reflectance dump written by ``render_frame
bands-out=...``. ABI Band 13 input is the north-first scalar f32le Kelvin plane
written by ``render_ir bt-out=...``.

The resulting pixel statistics and FSS values are collocation diagnostics, not
forecast-skill claims.  Forecast displacement, timing, and model-state error
can dominate an exact-grid comparison even when the observation operator is
well behaved.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


DISPLAY_THRESHOLDS = (0.20, 0.35, 0.50, 0.65, 0.80)
RAW_THRESHOLDS = (0.05, 0.10, 0.20, 0.35, 0.50)
DEFAULT_FSS_SCALES = (1, 3, 9, 27)
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
THERMAL_THRESHOLDS_K = (260.0, 235.0, 220.0, 205.0)
THERMAL_ENHANCEMENT_K = (180.0, 320.0)
THERMAL_DIFFERENCE_CLIP_K = 40.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--product",
        choices=("visible", "abi-band13"),
        default="visible",
        help="observation product (default: visible; preserves the original validator)",
    )
    parser.add_argument(
        "--synthetic", type=Path, help="SimSat PNG, f32le RGB, or scalar f32le Kelvin dump"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="aligned ABI NPZ from fetch-goes-abi-reference.py",
    )
    parser.add_argument("--output-dir", type=Path, help="immutable output directory")
    parser.add_argument(
        "--input-kind",
        choices=("auto", "png", "f32le-rgb", "f32le-scalar"),
        default="auto",
        help="synthetic encoding (default: infer from product and suffix)",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="optional fetch source-manifest.json (auto-detected beside the NPZ)",
    )
    parser.add_argument(
        "--fss-thresholds",
        type=float,
        nargs="+",
        help="visible luminance >= thresholds or Band 13 cold-event <= thresholds K",
    )
    parser.add_argument(
        "--fss-scales",
        type=int,
        nargs="+",
        default=list(DEFAULT_FSS_SCALES),
        help="odd square neighborhood widths in grid pixels (default: 1 3 9 27)",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="run dependency, metric, mask, raw-layout, and mismatch checks",
    )
    args = parser.parse_args()
    if not args.self_check:
        missing = [
            flag
            for flag, value in (
                ("--synthetic", args.synthetic),
                ("--reference", args.reference),
                ("--output-dir", args.output_dir),
            )
            if value is None
        ]
        if missing:
            parser.error("required unless --self-check: " + ", ".join(missing))
    if any(scale <= 0 or scale % 2 == 0 for scale in args.fss_scales):
        parser.error("--fss-scales must contain positive odd integers")
    if args.fss_thresholds:
        if args.product == "visible" and any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in args.fss_thresholds
        ):
            parser.error("visible --fss-thresholds must be finite values in [0, 1]")
        if args.product == "abi-band13" and any(
            not math.isfinite(value) or value < 100.0 or value > 400.0
            for value in args.fss_thresholds
        ):
            parser.error("abi-band13 --fss-thresholds must be finite Kelvin values in [100, 400]")
    if args.product == "visible" and args.input_kind == "f32le-scalar":
        parser.error("visible product does not accept --input-kind f32le-scalar")
    if args.product == "abi-band13" and args.input_kind in ("png", "f32le-rgb"):
        parser.error("abi-band13 product requires --input-kind auto|f32le-scalar")
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
        raise RuntimeError(
            "GOES validation dependencies are missing: "
            + ", ".join(missing)
            + ". Install them in the Python environment running this command."
        )
    return np, Image, ImageDraw


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def immutable_write(path: Path, content: bytes) -> str:
    digest = hashlib.sha256(content).hexdigest()
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(
                f"refusing to overwrite mismatched immutable output {path.name}; "
                "use a new output directory"
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


def png_bytes(Image: Any, pixels: Any) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(stream, format="PNG", optimize=True)
    return stream.getvalue()


def load_reference(np: Any, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"aligned ABI reference is not a file: {path.name}")
    try:
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    except Exception as error:
        raise RuntimeError(
            f"could not read aligned ABI NPZ {path.name}: {error}"
        ) from error
    required = ("cmi_c01", "cmi_c02", "cmi_c03", "valid")
    missing = [name for name in required if name not in arrays]
    if missing:
        raise RuntimeError(
            f"aligned ABI NPZ {path.name} is missing required arrays: {', '.join(missing)}"
        )
    shape = arrays["cmi_c02"].shape
    if len(shape) != 2:
        raise RuntimeError(f"aligned cmi_c02 must be 2-D, got shape {shape}")
    for name, values in arrays.items():
        if values.shape != shape:
            raise RuntimeError(
                f"aligned ABI array {name} shape {values.shape} does not match {shape}"
            )
    return arrays


def infer_input_kind(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    return "png" if path.suffix.lower() == ".png" else "f32le-rgb"


def load_synthetic(
    np: Any, Image: Any, path: Path, kind: str, shape: tuple[int, int]
) -> Any:
    if not path.is_file():
        raise RuntimeError(f"synthetic input is not a file: {path.name}")
    height, width = shape
    if kind == "png":
        try:
            with Image.open(path) as loaded:
                pixels = np.asarray(loaded.convert("RGB"), dtype=np.float64) / 255.0
        except Exception as error:
            raise RuntimeError(
                f"could not read synthetic PNG {path.name}: {error}"
            ) from error
        if pixels.shape != (height, width, 3):
            raise RuntimeError(
                f"synthetic PNG shape {pixels.shape} does not match aligned ABI grid "
                f"({height}, {width}, 3)"
            )
        return pixels
    expected_values = height * width * 3
    expected_bytes = expected_values * 4
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"f32le RGB size mismatch for {path.name}: got {actual_bytes} bytes, "
            f"expected {expected_bytes} for {width}x{height}x3"
        )
    pixels = np.fromfile(path, dtype="<f4").reshape(height, width, 3).astype(np.float64)
    if not np.all(np.isfinite(pixels)):
        raise RuntimeError(f"f32le RGB input {path.name} contains non-finite values")
    low = float(np.min(pixels))
    high = float(np.max(pixels))
    if low < -1.0e-5 or high > 1.00001:
        raise RuntimeError(
            f"f32le RGB values must be reflectance in [0,1]; observed range is "
            f"[{low:.6g}, {high:.6g}]"
        )
    return np.clip(pixels, 0.0, 1.0)


def load_synthetic_scalar(
    np: Any, path: Path, shape: tuple[int, int]
) -> Any:
    """Load the documented north-first f32le Kelvin plane exactly on the ABI grid."""
    if not path.is_file():
        raise RuntimeError(f"synthetic input is not a file: {path.name}")
    height, width = shape
    expected_values = height * width
    expected_bytes = expected_values * 4
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"f32le scalar size mismatch for {path.name}: got {actual_bytes} bytes, "
            f"expected {expected_bytes} for north-first {width}x{height}"
        )
    values = np.fromfile(path, dtype="<f4").reshape(height, width).astype(np.float64)
    if np.any(np.isinf(values)):
        raise RuntimeError(f"f32le scalar input {path.name} contains infinite values")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise RuntimeError(f"f32le scalar input {path.name} has no finite Kelvin values")
    low = float(np.min(finite))
    high = float(np.max(finite))
    if low < 100.0 or high > 400.0:
        raise RuntimeError(
            f"f32le scalar values must be plausible brightness temperatures in "
            f"[100,400] K or NaN no-data; observed finite range is [{low:.3f}, {high:.3f}]"
        )
    return values


def observed_rgb(
    np: Any, arrays: dict[str, Any], kind: str
) -> tuple[Any, dict[str, Any]]:
    red = np.clip(np.asarray(arrays["cmi_c02"], dtype=np.float64), 0.0, 1.0)
    blue = np.clip(np.asarray(arrays["cmi_c01"], dtype=np.float64), 0.0, 1.0)
    green = np.clip(
        0.45 * red
        + 0.10 * np.asarray(arrays["cmi_c03"], dtype=np.float64)
        + 0.45 * blue,
        0.0,
        1.0,
    )
    linear = np.stack((red, green, blue), axis=-1)
    if kind == "png":
        return np.sqrt(linear), {
            "name": "finished-visible-display",
            "synthetic": "SimSat finished RGB PNG normalized from uint8 to [0,1]",
            "observed": "sqrt(clip([C02, 0.45*C02 + 0.10*C03 + 0.45*C01, C01], 0, 1))",
            "luminance": "Rec.709 coefficients 0.2126/0.7152/0.0722 in display-value space",
        }
    return linear, {
        "name": "raw-visible-reflectance",
        "synthetic": "SimSat RgbReflectance north-first interleaved f32le RGB reflectance",
        "observed": "clip([C02, 0.45*C02 + 0.10*C03 + 0.45*C01, C01], 0, 1)",
        "luminance": "Rec.709 coefficients 0.2126/0.7152/0.0722 in linear reflectance space",
    }


def luminance(np: Any, rgb: Any) -> Any:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def quantile_record(np: Any, values: Any) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    points = np.quantile(values, QUANTILES)
    result: dict[str, Any] = {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
    for quantile, value in zip(QUANTILES, points):
        result[f"p{int(round(quantile * 100)):02d}"] = float(value)
    return result


def safe_correlation(np: Any, observed: Any, synthetic: Any) -> float | None:
    if (
        observed.size < 2
        or float(np.std(observed)) <= 1.0e-15
        or float(np.std(synthetic)) <= 1.0e-15
    ):
        return None
    value = float(np.corrcoef(observed, synthetic)[0, 1])
    return value if math.isfinite(value) else None


def comparison_metrics(
    np: Any, observed: Any, synthetic: Any, mask: Any
) -> dict[str, Any]:
    count = int(np.sum(mask))
    if count == 0:
        return {"count": 0}
    obs = observed[mask].astype(np.float64, copy=False)
    sim = synthetic[mask].astype(np.float64, copy=False)
    delta = sim - obs
    return {
        "count": count,
        "observed": quantile_record(np, obs),
        "synthetic": quantile_record(np, sim),
        "synthetic_minus_observed": quantile_record(np, delta),
        "bias": float(np.mean(delta)),
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "correlation": safe_correlation(np, obs, sim),
    }


def rgb_metrics(np: Any, observed: Any, synthetic: Any, mask: Any) -> dict[str, Any]:
    count = int(np.sum(mask))
    if count == 0:
        return {"count": 0}
    obs = observed[mask].astype(np.float64, copy=False)
    sim = synthetic[mask].astype(np.float64, copy=False)
    delta = sim - obs
    channels = {}
    for index, name in enumerate(("red", "green", "blue")):
        channels[name] = comparison_metrics(
            np, observed[..., index], synthetic[..., index], mask
        )
    return {
        "count": count,
        "mae_all_channels": float(np.mean(np.abs(delta))),
        "rmse_all_channels": float(np.sqrt(np.mean(delta * delta))),
        "channels": channels,
    }


def regime_masks(
    np: Any, arrays: dict[str, Any], finite: Any
) -> dict[str, dict[str, Any]]:
    valid = (np.asarray(arrays["valid"]) > 0) & finite
    result: dict[str, dict[str, Any]] = {
        "valid": {
            "available": True,
            "definition": "aligned valid>0 and finite observed/synthetic RGB",
            "mask": valid,
        }
    }

    def add(name: str, source: str, definition: str, predicate: Any) -> None:
        if source not in arrays:
            result[name] = {
                "available": False,
                "definition": definition,
                "unavailable_reason": f"aligned ABI NPZ has no {source} array",
            }
            return
        result[name] = {
            "available": True,
            "definition": definition,
            "mask": valid & predicate(np.asarray(arrays[source])),
        }

    add("strict_clear", "acm", "valid and official ACM == 0", lambda value: value == 0)
    add("broad_clear", "bcm", "valid and official BCM == 0", lambda value: value == 0)
    add(
        "land",
        "landmask",
        "valid and target-grid landmask >= 0.5",
        lambda value: value >= 0.5,
    )
    add(
        "ocean",
        "landmask",
        "valid and target-grid landmask < 0.5",
        lambda value: value < 0.5,
    )

    cloud_source = "bcm" if "bcm" in arrays else ("acm" if "acm" in arrays else None)
    if cloud_source is None:
        cloudy = None
        result["cloudy"] = {
            "available": False,
            "definition": "valid and official BCM == 1 (ACM != 0 fallback)",
            "unavailable_reason": "aligned ABI NPZ has neither bcm nor acm",
        }
    else:
        cloud_values = np.asarray(arrays[cloud_source])
        cloudy = valid & (
            (cloud_values == 1) if cloud_source == "bcm" else (cloud_values != 0)
        )
        result["cloudy"] = {
            "available": True,
            "definition": f"valid and official {cloud_source.upper()} "
            + ("== 1" if cloud_source == "bcm" else "!= 0"),
            "mask": cloudy,
        }

    thermal = (
        np.asarray(arrays["cmi_c13"], dtype=np.float64) if "cmi_c13" in arrays else None
    )
    thermal_specs = (
        (
            "warm_cloud_bt_gt_235k",
            "cloudy and ABI C13 brightness temperature > 235 K",
            lambda bt: bt > 235.0,
        ),
        (
            "ice_cloud_proxy_bt_le_235k",
            "cloudy and ABI C13 brightness temperature <= 235 K; thermal proxy, not a phase retrieval",
            lambda bt: bt <= 235.0,
        ),
        (
            "deep_convection_proxy_bt_le_220k",
            "cloudy and ABI C13 brightness temperature <= 220 K; cold-top proxy",
            lambda bt: bt <= 220.0,
        ),
        (
            "extreme_deep_convection_proxy_bt_le_205k",
            "cloudy and ABI C13 brightness temperature <= 205 K; very-cold-top proxy",
            lambda bt: bt <= 205.0,
        ),
    )
    for name, definition, predicate in thermal_specs:
        if cloudy is None or thermal is None:
            missing = "cmi_c13" if thermal is None else "official cloud mask"
            result[name] = {
                "available": False,
                "definition": definition,
                "unavailable_reason": f"aligned ABI NPZ has no {missing}",
            }
        else:
            result[name] = {
                "available": True,
                "definition": definition,
                "mask": cloudy & np.isfinite(thermal) & predicate(thermal),
            }
    return result


def thermal_regime_masks(
    np: Any, arrays: dict[str, Any], jointly_finite: Any
) -> dict[str, dict[str, Any]]:
    """Masks appropriate to scalar Band 13 BT comparisons.

    Temperature regimes are defined from the *observed* ABI C13 field. They are
    stratifiers, not a claim that the synthetic forecast should match each pixel.
    """
    valid = (np.asarray(arrays["valid"]) > 0) & jointly_finite
    result: dict[str, dict[str, Any]] = {
        "valid": {
            "available": True,
            "definition": "aligned valid>0 and finite observed/synthetic Band 13 BT",
            "mask": valid,
        }
    }

    def add(name: str, source: str, definition: str, predicate: Any) -> None:
        if source not in arrays:
            result[name] = {
                "available": False,
                "definition": definition,
                "unavailable_reason": f"aligned ABI NPZ has no {source} array",
            }
            return
        result[name] = {
            "available": True,
            "definition": definition,
            "mask": valid & predicate(np.asarray(arrays[source])),
        }

    add("strict_clear", "acm", "valid and official ACM == 0", lambda value: value == 0)
    add("broad_clear", "bcm", "valid and official BCM == 0", lambda value: value == 0)
    add(
        "land",
        "landmask",
        "valid and target-grid landmask >= 0.5",
        lambda value: value >= 0.5,
    )
    add(
        "ocean",
        "landmask",
        "valid and target-grid landmask < 0.5",
        lambda value: value < 0.5,
    )
    cloud_source = "bcm" if "bcm" in arrays else ("acm" if "acm" in arrays else None)
    if cloud_source is None:
        result["cloudy"] = {
            "available": False,
            "definition": "valid and official BCM == 1 (ACM != 0 fallback)",
            "unavailable_reason": "aligned ABI NPZ has neither bcm nor acm",
        }
    else:
        values = np.asarray(arrays[cloud_source])
        result["cloudy"] = {
            "available": True,
            "definition": f"valid and official {cloud_source.upper()} "
            + ("== 1" if cloud_source == "bcm" else "!= 0"),
            "mask": valid & ((values == 1) if cloud_source == "bcm" else (values != 0)),
        }
    observed = np.asarray(arrays["cmi_c13"], dtype=np.float64)
    for threshold in THERMAL_THRESHOLDS_K:
        slug = str(int(threshold))
        result[f"observed_bt_le_{slug}k"] = {
            "available": True,
            "definition": f"valid and observed ABI C13 brightness temperature <= {threshold:g} K",
            "mask": valid & (observed <= threshold),
        }
    return result


def box_sum(np: Any, values: Any, size: int) -> Any:
    radius = size // 2
    padded = np.pad(values, ((radius, radius), (radius, radius)), mode="constant")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
    integral = np.cumsum(np.cumsum(integral, axis=0), axis=1)
    return (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )


def fractions_skill_scores(
    np: Any,
    observed: Any,
    synthetic: Any,
    mask: Any,
    thresholds: list[float],
    scales: list[int],
) -> dict[str, Any]:
    mask_float = mask.astype(np.float64)
    mask_counts = {scale: box_sum(np, mask_float, scale) for scale in scales}
    records = []
    for threshold in thresholds:
        observed_event = ((observed >= threshold) & mask).astype(np.float64)
        synthetic_event = ((synthetic >= threshold) & mask).astype(np.float64)
        scale_records = []
        for scale in scales:
            counts = mask_counts[scale]
            usable = mask & (counts > 0.0)
            observed_fraction = np.divide(
                box_sum(np, observed_event, scale),
                counts,
                out=np.zeros_like(counts),
                where=counts > 0.0,
            )
            synthetic_fraction = np.divide(
                box_sum(np, synthetic_event, scale),
                counts,
                out=np.zeros_like(counts),
                where=counts > 0.0,
            )
            obs = observed_fraction[usable]
            sim = synthetic_fraction[usable]
            mse = float(np.mean((sim - obs) ** 2)) if obs.size else None
            denominator = float(np.mean(sim * sim + obs * obs)) if obs.size else 0.0
            fss = (
                1.0 - mse / denominator
                if mse is not None and denominator > 0.0
                else None
            )
            scale_records.append(
                {
                    "neighborhood_width_pixels": scale,
                    "sample_count": int(obs.size),
                    "fss": float(fss) if fss is not None else None,
                    "fraction_mse": mse,
                    "reference_denominator": denominator,
                }
            )
        records.append(
            {
                "threshold": float(threshold),
                "observed_event_fraction": float(np.mean(observed_event[mask]))
                if np.any(mask)
                else None,
                "synthetic_event_fraction": float(np.mean(synthetic_event[mask]))
                if np.any(mask)
                else None,
                "scales": scale_records,
            }
        )
    return {
        "event": "luminance >= threshold",
        "mask": "valid",
        "method": "standard FSS on locally valid square-neighborhood event fractions",
        "interpretation": "collocation/displacement diagnostic only; not a forecast-skill claim",
        "thresholds": records,
    }


def cold_threshold_diagnostics(
    np: Any,
    observed: Any,
    synthetic: Any,
    mask: Any,
    thresholds: list[float],
    scales: list[int],
) -> dict[str, Any]:
    """Categorical areas and FSS for cold-cloud BT <= threshold events."""
    mask_float = mask.astype(np.float64)
    mask_counts = {scale: box_sum(np, mask_float, scale) for scale in scales}
    mask_count = int(np.sum(mask))
    records = []
    for threshold in thresholds:
        observed_bool = (observed <= threshold) & mask
        synthetic_bool = (synthetic <= threshold) & mask
        hits = int(np.sum(observed_bool & synthetic_bool))
        misses = int(np.sum(observed_bool & ~synthetic_bool & mask))
        false_alarms = int(np.sum(~observed_bool & synthetic_bool & mask))
        correct_negatives = mask_count - hits - misses - false_alarms
        observed_count = hits + misses
        synthetic_count = hits + false_alarms
        fss_records = []
        observed_event = observed_bool.astype(np.float64)
        synthetic_event = synthetic_bool.astype(np.float64)
        for scale in scales:
            counts = mask_counts[scale]
            usable = mask & (counts > 0.0)
            observed_fraction = np.divide(
                box_sum(np, observed_event, scale),
                counts,
                out=np.zeros_like(counts),
                where=counts > 0.0,
            )
            synthetic_fraction = np.divide(
                box_sum(np, synthetic_event, scale),
                counts,
                out=np.zeros_like(counts),
                where=counts > 0.0,
            )
            obs = observed_fraction[usable]
            sim = synthetic_fraction[usable]
            mse = float(np.mean((sim - obs) ** 2)) if obs.size else None
            denominator = float(np.mean(sim * sim + obs * obs)) if obs.size else 0.0
            fss = 1.0 - mse / denominator if mse is not None and denominator > 0.0 else None
            fss_records.append(
                {
                    "neighborhood_width_pixels": scale,
                    "sample_count": int(obs.size),
                    "fss": float(fss) if fss is not None else None,
                    "fraction_mse": mse,
                    "reference_denominator": denominator,
                }
            )
        records.append(
            {
                "threshold_kelvin": float(threshold),
                "event": "brightness_temperature <= threshold_kelvin",
                "area": {
                    "valid_pixels": mask_count,
                    "observed_pixels": observed_count,
                    "synthetic_pixels": synthetic_count,
                    "observed_fraction": observed_count / mask_count if mask_count else None,
                    "synthetic_fraction": synthetic_count / mask_count if mask_count else None,
                    "synthetic_over_observed": synthetic_count / observed_count
                    if observed_count
                    else None,
                },
                "contingency": {
                    "hits": hits,
                    "misses": misses,
                    "false_alarms": false_alarms,
                    "correct_negatives": correct_negatives,
                    "probability_of_detection": hits / observed_count if observed_count else None,
                    "false_alarm_ratio": false_alarms / synthetic_count
                    if synthetic_count
                    else None,
                    "critical_success_index": hits / (hits + misses + false_alarms)
                    if hits + misses + false_alarms
                    else None,
                },
                "fss": fss_records,
            }
        )
    return {
        "event": "Band 13 brightness temperature <= threshold Kelvin",
        "mask": "valid",
        "method": "pixel contingency plus standard FSS on locally valid square-neighborhood event fractions",
        "interpretation": "collocation/displacement diagnostic only; not an observation-operator or forecast-skill claim",
        "thresholds": records,
    }


def connected_components_rle(
    np: Any, event: Any, minimum_area_pixels: int = 4
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Label 8-connected objects with row runs; requires only NumPy.

    The run-length union-find keeps the million-pixel target grid practical while
    avoiding a SciPy dependency. Objects smaller than ``minimum_area_pixels`` are
    excluded from object summaries but remain in pixel-area/FSS diagnostics.
    """
    height, _ = event.shape
    parent: list[int] = []
    run_stats: list[tuple[int, float, float, int, int, int, int]] = []

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    previous: list[tuple[int, int, int]] = []
    for y in range(height):
        row = np.asarray(event[y], dtype=np.int8)
        transitions = np.diff(np.pad(row, (1, 1), mode="constant"))
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1) - 1
        current: list[tuple[int, int, int]] = []
        previous_start = 0
        for x0_raw, x1_raw in zip(starts, ends):
            x0 = int(x0_raw)
            x1 = int(x1_raw)
            label = len(parent)
            parent.append(label)
            count = x1 - x0 + 1
            run_stats.append(
                (count, 0.5 * (x0 + x1) * count, float(y * count), x0, x1, y, y)
            )
            while previous_start < len(previous) and previous[previous_start][1] < x0 - 1:
                previous_start += 1
            cursor = previous_start
            while cursor < len(previous) and previous[cursor][0] <= x1 + 1:
                union(label, previous[cursor][2])
                cursor += 1
            current.append((x0, x1, label))
        previous = current

    aggregated: dict[int, list[float]] = {}
    for label, stats in enumerate(run_stats):
        root = find(label)
        count, sum_x, sum_y, x0, x1, y0, y1 = stats
        if root not in aggregated:
            aggregated[root] = [count, sum_x, sum_y, x0, x1, y0, y1]
        else:
            value = aggregated[root]
            value[0] += count
            value[1] += sum_x
            value[2] += sum_y
            value[3] = min(value[3], x0)
            value[4] = max(value[4], x1)
            value[5] = min(value[5], y0)
            value[6] = max(value[6], y1)

    objects = []
    excluded_count = 0
    excluded_area = 0
    for value in aggregated.values():
        area = int(value[0])
        if area < minimum_area_pixels:
            excluded_count += 1
            excluded_area += area
            continue
        objects.append(
            {
                "area_pixels": area,
                "centroid_x_pixels": float(value[1] / area),
                "centroid_y_pixels": float(value[2] / area),
                "bbox_xyxy_pixels": [int(value[3]), int(value[5]), int(value[4]), int(value[6])],
            }
        )
    objects.sort(key=lambda item: item["area_pixels"], reverse=True)
    areas = np.asarray([item["area_pixels"] for item in objects], dtype=np.float64)
    summary = {
        "connectivity": 8,
        "minimum_area_pixels": minimum_area_pixels,
        "object_count": len(objects),
        "object_area_pixels": quantile_record(np, areas),
        "excluded_smaller_object_count": excluded_count,
        "excluded_smaller_object_area_pixels": excluded_area,
        "largest_objects": objects[:25],
    }
    return summary, objects


def object_displacement_summary(
    np: Any, observed_objects: list[dict[str, Any]], synthetic_objects: list[dict[str, Any]]
) -> dict[str, Any]:
    """Greedy one-to-one centroid displacement for the 64 largest objects."""
    observed = observed_objects[:64]
    synthetic = synthetic_objects[:64]
    if not observed or not synthetic:
        return {
            "available": False,
            "reason": "one or both fields have no retained cold-cloud objects",
        }
    unused = set(range(len(synthetic)))
    matches = []
    for obs_index, obs in enumerate(observed):
        if not unused:
            break
        ox = obs["centroid_x_pixels"]
        oy = obs["centroid_y_pixels"]
        sim_index = min(
            unused,
            key=lambda index: (synthetic[index]["centroid_x_pixels"] - ox) ** 2
            + (synthetic[index]["centroid_y_pixels"] - oy) ** 2,
        )
        unused.remove(sim_index)
        sim = synthetic[sim_index]
        dx = sim["centroid_x_pixels"] - ox
        dy = sim["centroid_y_pixels"] - oy
        matches.append(
            {
                "observed_rank": obs_index + 1,
                "synthetic_rank": sim_index + 1,
                "distance_grid_pixels": float(math.hypot(dx, dy)),
                "dx_grid_pixels": float(dx),
                "dy_grid_pixels": float(dy),
                "observed_area_pixels": obs["area_pixels"],
                "synthetic_area_pixels": sim["area_pixels"],
            }
        )
    distances = np.asarray([item["distance_grid_pixels"] for item in matches])
    return {
        "available": True,
        "method": "greedy one-to-one nearest centroid among each field's 64 largest retained objects",
        "coordinate_convention": "x eastward; y southward from north-first row 0; units are target-grid pixels",
        "matched_count": len(matches),
        "distance_grid_pixels": quantile_record(np, distances),
        "matches": matches[:25],
        "caveat": "object displacement mixes forecast timing/location error with observation-operator differences",
    }


def cold_cloud_objects(
    np: Any,
    observed: Any,
    synthetic: Any,
    mask: Any,
    thresholds: list[float],
) -> dict[str, Any]:
    records = []
    for threshold in thresholds:
        observed_summary, observed_objects = connected_components_rle(
            np, (observed <= threshold) & mask
        )
        synthetic_summary, synthetic_objects = connected_components_rle(
            np, (synthetic <= threshold) & mask
        )
        records.append(
            {
                "threshold_kelvin": float(threshold),
                "observed": observed_summary,
                "synthetic": synthetic_summary,
                "centroid_displacement": object_displacement_summary(
                    np, observed_objects, synthetic_objects
                ),
            }
        )
    return {
        "event": "Band 13 brightness temperature <= threshold Kelvin",
        "implementation": "NumPy-only row-run union-find; no SciPy dependency",
        "thresholds": records,
    }


def spectrum(np: Any, values: Any, mask: Any) -> tuple[Any, dict[str, Any]]:
    count = int(np.sum(mask))
    if count < 4:
        return None, {"available": False, "reason": "fewer than four valid pixels"}
    mean = float(np.mean(values[mask]))
    height, width = values.shape
    window = np.hanning(height)[:, None] * np.hanning(width)[None, :]
    # NaN * 0 is still NaN, so replace off-mask values before the FFT rather
    # than relying on multiplication by the validity mask.
    weighted = np.where(mask, values - mean, 0.0) * window
    normalization = float(np.sum((window * mask) ** 2))
    power = np.abs(np.fft.rfft2(weighted)) ** 2 / max(normalization, 1.0e-30)
    power[0, 0] = 0.0
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.rfftfreq(width)[None, :]
    frequency = np.sqrt(fx * fx + fy * fy)
    total = float(np.sum(power))
    bands = (
        ("domain_scale", 0.0, 0.02),
        ("broad", 0.02, 0.08),
        ("mesoscale", 0.08, 0.25),
        ("fine", 0.25, 0.50),
        ("corner", 0.50, float(np.sqrt(0.5))),
    )
    summary: dict[str, Any] = {
        "available": total > 0.0,
        "mean_removed": mean,
        "window": "separable Hann multiplied by valid mask",
        "frequency_units": "cycles per target-grid pixel",
        "total_power": total,
        "spectral_centroid_cycles_per_pixel": float(np.sum(power * frequency) / total)
        if total > 0.0
        else None,
        "bands": {},
    }
    for name, low, high in bands:
        selected = (frequency >= low) & (frequency < high) & (frequency > 0.0)
        band_power = float(np.sum(power[selected]))
        summary["bands"][name] = {
            "min_cycles_per_pixel": low,
            "max_cycles_per_pixel": high,
            "power": band_power,
            "power_fraction": band_power / total if total > 0.0 else None,
        }
    return (power, frequency), summary


def spatial_spectrum_summary(
    np: Any, observed: Any, synthetic: Any, mask: Any
) -> dict[str, Any]:
    observed_fields, observed_summary = spectrum(np, observed, mask)
    synthetic_fields, synthetic_summary = spectrum(np, synthetic, mask)
    result: dict[str, Any] = {
        "method": "2-D radial power spectrum of mean-removed luminance on the exact target grid",
        "mask": "valid",
        "observed": observed_summary,
        "synthetic": synthetic_summary,
        "band_power_ratio_synthetic_over_observed": {},
        "radial_profile": [],
    }
    for name in observed_summary.get("bands", {}):
        obs = observed_summary["bands"][name]["power"]
        sim = synthetic_summary.get("bands", {}).get(name, {}).get("power")
        result["band_power_ratio_synthetic_over_observed"][name] = (
            float(sim / obs) if obs and sim is not None else None
        )
    if observed_fields is None or synthetic_fields is None:
        return result
    observed_power, frequency = observed_fields
    synthetic_power, _ = synthetic_fields
    positive = frequency > 0.0
    minimum = float(np.min(frequency[positive]))
    maximum = float(np.max(frequency))
    edges = np.geomspace(minimum, maximum, 25)
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (frequency >= low) & (frequency < high)
        count = int(np.sum(selected))
        obs = float(np.mean(observed_power[selected])) if count else None
        sim = float(np.mean(synthetic_power[selected])) if count else None
        result["radial_profile"].append(
            {
                "min_cycles_per_pixel": float(low),
                "max_cycles_per_pixel": float(high),
                "sample_count": count,
                "observed_mean_power": obs,
                "synthetic_mean_power": sim,
                "synthetic_over_observed": float(sim / obs)
                if obs and sim is not None
                else None,
            }
        )
    return result


def rgb_u8(np: Any, rgb: Any, valid: Any, raw: bool) -> Any:
    shown = np.sqrt(np.clip(rgb, 0.0, 1.0)) if raw else np.clip(rgb, 0.0, 1.0)
    shown = np.where(valid[..., None], shown, 0.0)
    return np.rint(shown * 255.0).astype(np.uint8)


def difference_u8(
    np: Any, observed_y: Any, synthetic_y: Any, valid: Any
) -> tuple[Any, float]:
    delta = synthetic_y - observed_y
    finite_values = np.abs(delta[valid])
    scale = (
        max(0.02, float(np.quantile(finite_values, 0.99)))
        if finite_values.size
        else 0.02
    )
    normalized = np.clip(delta / scale, -1.0, 1.0)
    magnitude = np.abs(normalized)
    image = np.ones((*delta.shape, 3), dtype=np.float64)
    positive = normalized >= 0.0
    image[..., 1] = np.where(positive, 1.0 - magnitude, 1.0 - magnitude)
    image[..., 0] = np.where(positive, 1.0, 1.0 - magnitude)
    image[..., 2] = np.where(positive, 1.0 - magnitude, 1.0)
    image = np.where(valid[..., None], image, 0.0)
    return np.rint(image * 255.0).astype(np.uint8), scale


def mask_u8(np: Any, mask: Any) -> Any:
    plane = np.where(mask, 255, 0).astype(np.uint8)
    return np.repeat(plane[..., None], 3, axis=2)


def joint_histogram_png(
    np: Any, Image: Any, ImageDraw: Any, observed: Any, synthetic: Any, mask: Any
) -> bytes:
    histogram, _, _ = np.histogram2d(
        observed[mask], synthetic[mask], bins=256, range=((0.0, 1.0), (0.0, 1.0))
    )
    density = np.log1p(histogram.T)
    if float(np.max(density)) > 0.0:
        density /= float(np.max(density))
    red = np.clip(3.0 * density - 1.25, 0.0, 1.0)
    green = np.clip(3.0 * density - 0.50, 0.0, 1.0)
    blue = np.clip(2.5 * density, 0.0, 1.0)
    pixels = np.rint(np.stack((red, green, blue), axis=-1) * 255.0).astype(np.uint8)
    resampling = getattr(Image, "Resampling", Image)
    plot = Image.fromarray(np.flipud(pixels), mode="RGB").resize(
        (512, 512), resample=resampling.NEAREST
    )
    canvas = Image.new("RGB", (620, 610), (18, 20, 24))
    canvas.paste(plot, (72, 28))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((71, 27, 584, 540), outline=(220, 220, 220), width=1)
    draw.line((72, 540, 584, 28), fill=(160, 160, 160), width=1)
    draw.text((264, 566), "Observed luminance", fill=(235, 235, 235))
    draw.text((8, 12), "Synthetic luminance (vertical)", fill=(235, 235, 235))
    draw.text((67, 545), "0", fill=(220, 220, 220))
    draw.text((577, 545), "1", fill=(220, 220, 220))
    draw.text((54, 25), "1", fill=(220, 220, 220))
    stream = io.BytesIO()
    canvas.save(stream, format="PNG", optimize=True)
    return stream.getvalue()


def thermal_bt_u8(np: Any, bt_kelvin: Any, valid: Any) -> Any:
    """Fixed 180--320 K cold-white/warm-black Band 13 enhancement."""
    cold, warm = THERMAL_ENHANCEMENT_K
    level = np.clip((warm - bt_kelvin) / (warm - cold), 0.0, 1.0)
    level = np.where(valid, level, 0.0)
    plane = np.rint(level * 255.0).astype(np.uint8)
    return np.repeat(plane[..., None], 3, axis=2)


def thermal_difference_u8(np: Any, observed: Any, synthetic: Any, valid: Any) -> Any:
    """Fixed signed synthetic-minus-observed BT difference, clipped at +/-40 K."""
    delta = synthetic - observed
    normalized = np.clip(delta / THERMAL_DIFFERENCE_CLIP_K, -1.0, 1.0)
    magnitude = np.abs(normalized)
    image = np.ones((*delta.shape, 3), dtype=np.float64)
    positive = normalized >= 0.0
    image[..., 1] = 1.0 - magnitude
    image[..., 0] = np.where(positive, 1.0, 1.0 - magnitude)
    image[..., 2] = np.where(positive, 1.0 - magnitude, 1.0)
    image = np.where(valid[..., None], image, 0.0)
    return np.rint(image * 255.0).astype(np.uint8)


def thermal_joint_histogram_png(
    np: Any, Image: Any, ImageDraw: Any, observed: Any, synthetic: Any, mask: Any
) -> bytes:
    cold, warm = THERMAL_ENHANCEMENT_K
    observed_values = np.clip(observed[mask], cold, warm)
    synthetic_values = np.clip(synthetic[mask], cold, warm)
    histogram, _, _ = np.histogram2d(
        observed_values,
        synthetic_values,
        bins=256,
        range=((cold, warm), (cold, warm)),
    )
    density = np.log1p(histogram.T)
    if float(np.max(density)) > 0.0:
        density /= float(np.max(density))
    red = np.clip(3.0 * density - 1.25, 0.0, 1.0)
    green = np.clip(3.0 * density - 0.50, 0.0, 1.0)
    blue = np.clip(2.5 * density, 0.0, 1.0)
    pixels = np.rint(np.stack((red, green, blue), axis=-1) * 255.0).astype(np.uint8)
    resampling = getattr(Image, "Resampling", Image)
    plot = Image.fromarray(np.flipud(pixels), mode="RGB").resize(
        (512, 512), resample=resampling.NEAREST
    )
    canvas = Image.new("RGB", (620, 610), (18, 20, 24))
    canvas.paste(plot, (72, 28))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((71, 27, 584, 540), outline=(220, 220, 220), width=1)
    draw.line((72, 540, 584, 28), fill=(160, 160, 160), width=1)
    draw.text((260, 566), "Observed ABI C13 BT (K)", fill=(235, 235, 235))
    draw.text((8, 12), "Synthetic Band 13 BT (K), vertical", fill=(235, 235, 235))
    draw.text((57, 545), f"{int(cold)}", fill=(220, 220, 220))
    draw.text((567, 545), f"{int(warm)}", fill=(220, 220, 220))
    draw.text((45, 25), f"{int(warm)}", fill=(220, 220, 220))
    stream = io.BytesIO()
    canvas.save(stream, format="PNG", optimize=True)
    return stream.getvalue()


def public_source_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        source = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"could not read source manifest {path.name}: {error}"
        ) from error
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "provider": source.get("provider"),
        "platform": source.get("platform"),
        "target_time": source.get("target_time"),
        "objects": [
            {
                key: item.get(key)
                for key in (
                    "product",
                    "file",
                    "url",
                    "sha256",
                    "scan_start",
                    "scan_end",
                )
            }
            for item in source.get("objects", [])
        ],
    }


def validate(
    synthetic_path: Path,
    reference_path: Path,
    output_dir: Path,
    requested_kind: str,
    thresholds: list[float] | None,
    scales: list[int],
    source_manifest: Path | None,
) -> dict[str, Any]:
    np, Image, ImageDraw = require_dependencies()
    arrays = load_reference(np, reference_path)
    shape = arrays["cmi_c02"].shape
    kind = infer_input_kind(synthetic_path, requested_kind)
    synthetic_rgb = load_synthetic(np, Image, synthetic_path, kind, shape)
    observed, metric_space = observed_rgb(np, arrays, kind)
    observed_y = luminance(np, observed)
    synthetic_y = luminance(np, synthetic_rgb)
    finite = np.all(np.isfinite(observed), axis=-1) & np.all(
        np.isfinite(synthetic_rgb), axis=-1
    )
    regimes = regime_masks(np, arrays, finite)
    valid = regimes["valid"]["mask"]
    if not np.any(valid):
        raise RuntimeError(
            "aligned ABI reference and synthetic input have no jointly valid pixels"
        )
    thresholds = list(
        thresholds or (DISPLAY_THRESHOLDS if kind == "png" else RAW_THRESHOLDS)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Any] = {}
    image_specs = {
        "observed": rgb_u8(np, observed, valid, raw=kind == "f32le-rgb"),
        "synthetic": rgb_u8(np, synthetic_rgb, valid, raw=kind == "f32le-rgb"),
    }
    difference, difference_scale = difference_u8(np, observed_y, synthetic_y, valid)
    image_specs["difference"] = difference
    for name, pixels in image_specs.items():
        relative = Path(name + ".png")
        digest = immutable_write(output_dir / relative, png_bytes(Image, pixels))
        artifacts[name] = {"file": relative.as_posix(), "sha256": digest}
    histogram_bytes = joint_histogram_png(
        np, Image, ImageDraw, observed_y, synthetic_y, valid
    )
    artifacts["joint_histogram"] = {
        "file": "joint-histogram.png",
        "sha256": immutable_write(output_dir / "joint-histogram.png", histogram_bytes),
    }

    regime_records: dict[str, Any] = {}
    valid_count = int(np.sum(valid))
    for name, spec in regimes.items():
        record = {key: value for key, value in spec.items() if key != "mask"}
        if spec["available"]:
            mask = spec["mask"]
            count = int(np.sum(mask))
            relative = Path("masks") / f"{name}.png"
            mask_digest = immutable_write(
                output_dir / relative, png_bytes(Image, mask_u8(np, mask))
            )
            record.update(
                {
                    "count": count,
                    "fraction_of_valid": count / valid_count,
                    "mask_artifact": {
                        "file": relative.as_posix(),
                        "sha256": mask_digest,
                    },
                    "luminance": comparison_metrics(np, observed_y, synthetic_y, mask),
                    "rgb": rgb_metrics(np, observed, synthetic_rgb, mask),
                }
            )
        regime_records[name] = record

    source_path = source_manifest
    if source_path is None:
        candidate = reference_path.parent / "source-manifest.json"
        source_path = candidate if candidate.is_file() else None
    script_path = Path(__file__).resolve()
    companion = reference_path.with_suffix(".json")
    report: dict[str, Any] = {
        "schema_version": 1,
        "command": "simsat-validate-goes",
        "interpretation": (
            "Exact-grid observation-operator/collocation diagnostics only. These values do not "
            "claim forecast pixel-match skill; forecast displacement, timing, and state error can dominate."
        ),
        "grid": {
            "width": int(shape[1]),
            "height": int(shape[0]),
            "rows": "north-first",
        },
        "metric_space": metric_space,
        "inputs": {
            "synthetic": {
                "file": synthetic_path.name,
                "sha256": sha256_file(synthetic_path),
                "bytes": synthetic_path.stat().st_size,
                "kind": kind,
            },
            "aligned_abi": {
                "file": reference_path.name,
                "sha256": sha256_file(reference_path),
                "bytes": reference_path.stat().st_size,
                "variables": sorted(arrays),
            },
        },
        "source": public_source_manifest(source_path),
        "artifacts": artifacts,
        "difference_visualization": {
            "quantity": "synthetic minus observed luminance",
            "colors": "blue negative, white zero, red positive, black invalid",
            "symmetric_clip": [-difference_scale, difference_scale],
            "note": "presentation image only; metrics use unclipped values",
        },
        "regimes": regime_records,
        "fractions_skill_score": fractions_skill_scores(
            np, observed_y, synthetic_y, valid, thresholds, scales
        ),
        "spatial_spectrum": spatial_spectrum_summary(
            np, observed_y, synthetic_y, valid
        ),
        "provenance": {
            "script_file": script_path.name,
            "script_sha256": sha256_file(script_path),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pillow": getattr(Image, "__version__", "unknown"),
            "aligned_companion": (
                {"file": companion.name, "sha256": sha256_file(companion)}
                if companion.is_file()
                else None
            ),
            "fss_thresholds": thresholds,
            "fss_scales_pixels": scales,
        },
    }
    report_bytes = json_bytes(report)
    report_sha = immutable_write(output_dir / "validation.json", report_bytes)
    return {
        "status": "complete",
        "report": str(output_dir / "validation.json"),
        "report_sha256": report_sha,
        "artifacts": {
            name: str(output_dir / value["file"]) for name, value in artifacts.items()
        },
        "valid_count": valid_count,
        "metric_space": metric_space["name"],
    }


def validate_thermal(
    synthetic_path: Path,
    reference_path: Path,
    output_dir: Path,
    thresholds: list[float] | None,
    scales: list[int],
    source_manifest: Path | None,
) -> dict[str, Any]:
    """Validate a scalar SimSat Band 13 BT plane against aligned ABI C13."""
    np, Image, ImageDraw = require_dependencies()
    arrays = load_reference(np, reference_path)
    if "cmi_c13" not in arrays:
        raise RuntimeError(
            f"aligned ABI NPZ {reference_path.name} is missing required array: cmi_c13"
        )
    shape = arrays["cmi_c02"].shape
    synthetic = load_synthetic_scalar(np, synthetic_path, shape)
    observed = np.asarray(arrays["cmi_c13"], dtype=np.float64)
    jointly_finite = np.isfinite(observed) & np.isfinite(synthetic)
    regimes = thermal_regime_masks(np, arrays, jointly_finite)
    valid = regimes["valid"]["mask"]
    if not np.any(valid):
        raise RuntimeError(
            "aligned ABI C13 reference and synthetic BT input have no jointly valid pixels"
        )
    thresholds = list(thresholds or THERMAL_THRESHOLDS_K)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Any] = {}
    image_specs = {
        "observed": thermal_bt_u8(np, observed, valid),
        "synthetic": thermal_bt_u8(np, synthetic, valid),
        "difference": thermal_difference_u8(np, observed, synthetic, valid),
    }
    for name, pixels in image_specs.items():
        relative = Path(name + ".png")
        digest = immutable_write(output_dir / relative, png_bytes(Image, pixels))
        artifacts[name] = {"file": relative.as_posix(), "sha256": digest}
    histogram_bytes = thermal_joint_histogram_png(
        np, Image, ImageDraw, observed, synthetic, valid
    )
    artifacts["joint_histogram"] = {
        "file": "joint-histogram.png",
        "sha256": immutable_write(output_dir / "joint-histogram.png", histogram_bytes),
    }

    regime_records: dict[str, Any] = {}
    valid_count = int(np.sum(valid))
    for name, spec in regimes.items():
        record = {key: value for key, value in spec.items() if key != "mask"}
        if spec["available"]:
            mask = spec["mask"]
            count = int(np.sum(mask))
            relative = Path("masks") / f"{name}.png"
            mask_digest = immutable_write(
                output_dir / relative, png_bytes(Image, mask_u8(np, mask))
            )
            record.update(
                {
                    "count": count,
                    "fraction_of_valid": count / valid_count,
                    "mask_artifact": {
                        "file": relative.as_posix(),
                        "sha256": mask_digest,
                    },
                    "brightness_temperature_kelvin": comparison_metrics(
                        np, observed, synthetic, mask
                    ),
                }
            )
        regime_records[name] = record

    source_path = source_manifest
    if source_path is None:
        candidate = reference_path.parent / "source-manifest.json"
        source_path = candidate if candidate.is_file() else None
    script_path = Path(__file__).resolve()
    companion = reference_path.with_suffix(".json")
    report: dict[str, Any] = {
        "schema_version": 1,
        "command": "simsat-validate-goes",
        "product": "abi-band13",
        "interpretation": (
            "Exact-grid collocation diagnostics only. Forecast displacement, timing, and state "
            "error can dominate. These metrics are not observation-operator skill and are not "
            "forecast-skill scores."
        ),
        "grid": {
            "width": int(shape[1]),
            "height": int(shape[0]),
            "rows": "north-first",
        },
        "metric_space": {
            "name": "abi-band13-brightness-temperature",
            "units": "kelvin",
            "synthetic": "SimSat north-first scalar f32le brightness-temperature plane",
            "observed": "aligned GOES ABI CMI C13 brightness temperature",
        },
        "inputs": {
            "synthetic": {
                "file": synthetic_path.name,
                "sha256": sha256_file(synthetic_path),
                "bytes": synthetic_path.stat().st_size,
                "kind": "f32le-scalar",
                "layout": "north-first row-major float32 little-endian Kelvin; NaN is no-data",
            },
            "aligned_abi": {
                "file": reference_path.name,
                "sha256": sha256_file(reference_path),
                "bytes": reference_path.stat().st_size,
                "variable": "cmi_c13",
                "variables": sorted(arrays),
            },
        },
        "source": public_source_manifest(source_path),
        "artifacts": artifacts,
        "brightness_temperature_visualization": {
            "enhancement": "fixed linear grayscale; cold white, warm black, invalid black",
            "range_kelvin": list(THERMAL_ENHANCEMENT_K),
            "note": "presentation image only; metrics use unclipped Kelvin values",
        },
        "difference_visualization": {
            "quantity": "synthetic minus observed Band 13 brightness temperature",
            "units": "kelvin",
            "colors": "blue negative/cold bias, white zero, red positive/warm bias, black invalid",
            "symmetric_clip_kelvin": [
                -THERMAL_DIFFERENCE_CLIP_K,
                THERMAL_DIFFERENCE_CLIP_K,
            ],
            "note": "presentation image only; metrics use unclipped Kelvin differences",
        },
        "regimes": regime_records,
        "cold_threshold_diagnostics": cold_threshold_diagnostics(
            np, observed, synthetic, valid, thresholds, scales
        ),
        "cold_cloud_objects": cold_cloud_objects(
            np, observed, synthetic, valid, thresholds
        ),
        "spatial_spectrum": spatial_spectrum_summary(
            np, observed, synthetic, valid
        ),
        "provenance": {
            "script_file": script_path.name,
            "script_sha256": sha256_file(script_path),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pillow": getattr(Image, "__version__", "unknown"),
            "aligned_companion": (
                {"file": companion.name, "sha256": sha256_file(companion)}
                if companion.is_file()
                else None
            ),
            "cold_thresholds_kelvin": thresholds,
            "fss_scales_pixels": scales,
        },
    }
    report_bytes = json_bytes(report)
    report_sha = immutable_write(output_dir / "validation.json", report_bytes)
    return {
        "status": "complete",
        "report": str(output_dir / "validation.json"),
        "report_sha256": report_sha,
        "artifacts": {
            name: str(output_dir / value["file"]) for name, value in artifacts.items()
        },
        "valid_count": valid_count,
        "metric_space": "abi-band13-brightness-temperature",
    }


def self_check() -> int:
    np, Image, _ = require_dependencies()
    with tempfile.TemporaryDirectory(prefix="simsat-goes-validation-") as directory:
        root = Path(directory)
        height, width = 24, 32
        yy, xx = np.mgrid[0:height, 0:width]
        c02 = (0.05 + 0.65 * xx / (width - 1)).astype(np.float32)
        c01 = (0.08 + 0.35 * yy / (height - 1)).astype(np.float32)
        c03 = (0.04 + 0.45 * (xx + yy) / (width + height - 2)).astype(np.float32)
        valid = np.ones((height, width), dtype=np.uint8)
        valid[:2, :3] = 0
        bcm = (xx > width // 2).astype(np.float32)
        acm = bcm.copy()
        c13 = np.where(
            xx > 27, 200.0, np.where(xx > 23, 218.0, np.where(xx > 16, 230.0, 285.0))
        ).astype(np.float32)
        landmask = (yy < height // 2).astype(np.float32)
        reference = root / "aligned.npz"
        np.savez_compressed(
            reference,
            cmi_c01=c01,
            cmi_c02=c02,
            cmi_c03=c03,
            cmi_c13=c13,
            valid=valid,
            bcm=bcm,
            acm=acm,
            landmask=landmask,
        )
        arrays = load_reference(np, reference)
        raw_observed, _ = observed_rgb(np, arrays, "f32le-rgb")
        raw = root / "synthetic.bin"
        raw_observed.astype("<f4").tofile(raw)
        raw_result = validate(
            raw, reference, root / "raw-out", "f32le-rgb", [0.1, 0.3], [1, 3], None
        )
        raw_report = json.loads(Path(raw_result["report"]).read_text(encoding="utf-8"))
        assert raw_report["regimes"]["valid"]["luminance"]["rmse"] < 1.0e-7
        assert raw_report["regimes"]["strict_clear"]["count"] > 0
        assert raw_report["regimes"]["warm_cloud_bt_gt_235k"]["count"] == 0
        assert raw_report["regimes"]["ice_cloud_proxy_bt_le_235k"]["count"] > 0
        assert all(
            scale["fss"] is None or abs(scale["fss"] - 1.0) < 1.0e-10
            for threshold in raw_report["fractions_skill_score"]["thresholds"]
            for scale in threshold["scales"]
        )

        display_observed, _ = observed_rgb(np, arrays, "png")
        png = root / "synthetic.png"
        Image.fromarray(
            np.rint(display_observed * 255.0).astype(np.uint8), mode="RGB"
        ).save(png)
        png_result = validate(
            png, reference, root / "png-out", "png", [0.2], [1, 3], None
        )
        png_report = json.loads(Path(png_result["report"]).read_text(encoding="utf-8"))
        assert png_report["regimes"]["valid"]["luminance"]["rmse"] <= 1.0 / 255.0
        assert Path(png_result["artifacts"]["joint_histogram"]).is_file()

        scalar = root / "synthetic-bt.bin"
        c13.astype("<f4").tofile(scalar)
        thermal_result = validate_thermal(
            scalar,
            reference,
            root / "thermal-out",
            list(THERMAL_THRESHOLDS_K),
            [1, 3],
            None,
        )
        thermal_report = json.loads(
            Path(thermal_result["report"]).read_text(encoding="utf-8")
        )
        assert (
            thermal_report["regimes"]["valid"]["brightness_temperature_kelvin"][
                "rmse"
            ]
            < 1.0e-7
        )
        assert all(
            scale["fss"] is None or abs(scale["fss"] - 1.0) < 1.0e-10
            for threshold in thermal_report["cold_threshold_diagnostics"]["thresholds"]
            for scale in threshold["fss"]
        )
        assert any(
            threshold["observed"]["object_count"] > 0
            for threshold in thermal_report["cold_cloud_objects"]["thresholds"]
        )
        assert all(
            not threshold["centroid_displacement"]["available"]
            or threshold["centroid_displacement"]["distance_grid_pixels"]["max"]
            < 1.0e-10
            for threshold in thermal_report["cold_cloud_objects"]["thresholds"]
        )

        broken = root / "broken.bin"
        broken.write_bytes(raw.read_bytes()[:-4])
        try:
            load_synthetic(np, Image, broken, "f32le-rgb", (height, width))
        except RuntimeError as error:
            assert "size mismatch" in str(error) and "expected" in str(error)
        else:
            raise AssertionError("truncated f32le input did not fail")
        broken_scalar = root / "broken-scalar.bin"
        broken_scalar.write_bytes(scalar.read_bytes()[:-4])
        try:
            load_synthetic_scalar(np, broken_scalar, (height, width))
        except RuntimeError as error:
            assert "size mismatch" in str(error) and "north-first" in str(error)
        else:
            raise AssertionError("truncated scalar f32le input did not fail")
    print(
        json.dumps(
            {
                "self_check": "passed",
                "png_exact_grid": "passed",
                "f32le_layout_and_identity": "passed",
                "regime_masks": "passed",
                "fss_identity": "passed",
                "shape_mismatch_error": "passed",
                "abi_band13_scalar_layout_and_identity": "passed",
                "abi_band13_threshold_fss_and_objects": "passed",
                "abi_band13_malformed_input": "passed",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    args = parse_args()
    try:
        if args.self_check:
            return self_check()
        if args.product == "abi-band13":
            result = validate_thermal(
                args.synthetic.resolve(),
                args.reference.resolve(),
                args.output_dir.resolve(),
                args.fss_thresholds,
                args.fss_scales,
                args.source_manifest.resolve() if args.source_manifest else None,
            )
        else:
            result = validate(
                args.synthetic.resolve(),
                args.reference.resolve(),
                args.output_dir.resolve(),
                args.input_kind,
                args.fss_thresholds,
                args.fss_scales,
                args.source_manifest.resolve() if args.source_manifest else None,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, RuntimeError, ValueError, AssertionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
