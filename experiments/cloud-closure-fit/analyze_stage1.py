#!/usr/bin/env python3
"""Join the exact Stage-1 CUDA request grid to delta flux and test the bounded closure.

The CUDA and legacy quantities in this file are directional BRF.  The joined
delta-two-stream quantities are hemispheric fluxes and are never used as a
directional residual, ratio, or fit target.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = EXPERIMENT_DIR / "fixtures"
DEFAULT_REQUEST_GRID = FIXTURE_DIR / "stage1-request-grid-v1.csv"
DEFAULT_RESULTS_DIR = EXPERIMENT_DIR / "requested-results"
DEFAULT_DELTA_CSV = (
    ROOT / "experiments/delta-two-stream/fixtures/stage0-slab-flux-v1.csv"
)
DEFAULT_STAGE0_SCRIPT = EXPERIMENT_DIR / "cloud_closure_fit.py"
DEFAULT_ORACLE_SOURCE = ROOT / "experiments/cuda-cloud-oracle/slab_oracle.cu"
DEFAULT_ORACLE_EXE = ROOT / "experiments/cuda-cloud-oracle/build/slab_oracle.exe"

MANIFEST_FIELDS = ["case", "canonical_physics_sha256"]
NONPHYSICS_CUDA_FIELDS = {"elapsed_ms", "kernel_ms", "max_batch_ms"}

JOIN_FIELDS = [
    "case",
    "split",
    "tau",
    "ssa",
    "hg_g",
    "sun_zenith_deg",
    "view_zenith_deg",
    "relative_azimuth_deg",
    "surface_albedo_for_delta_join",
    "samples",
    "seed",
    "max_scatters",
    "batch_samples",
    "cuda_backend",
    "cuda_device",
    "cuda_directional_brf",
    "cuda_standard_error_brf",
    "cuda_ci95_low_brf",
    "cuda_ci95_high_brf",
    "cuda_ci95_halfwidth_brf",
    "cuda_truncated_paths",
    "cuda_truncated_fraction",
    "analytic_hg_single_directional_brf",
    "cuda_directional_higher_brf",
    "legacy_hg_proxy_directional_brf_octave1",
    "legacy_hg_proxy_directional_brf_octave6",
    "legacy_hg_proxy_directional_higher_brf",
    "effective_alpha_unclamped",
    "effective_alpha_informative",
    "calibration_alpha_bounded",
    "predicted_directional_brf",
    "directional_residual_brf",
    "directional_abs_residual_brf",
    "directional_abs_residual_outside_mc95",
    "directional_acceptance_tolerance_brf",
    "directional_acceptance_pass",
    "delta_phase_case",
    "delta_hemispheric_toa_reflectance",
    "delta_hemispheric_first_reflectance",
    "delta_hemispheric_higher_reflectance",
    "delta_hemispheric_bottom_direct_transmittance",
    "delta_hemispheric_bottom_diffuse_transmittance",
    "delta_hemispheric_atmosphere_absorptance",
    "delta_hemispheric_energy_sum",
    "observable_contract",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stage0(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("simsat_cloud_closure_stage0", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load Stage-0 helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(row: Mapping[str, object], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field} in {row.get('case', '<unknown>')}: {value}")
    return value


def exact_numeric(left: object, right: object) -> bool:
    return round(float(left), 12) == round(float(right), 12)


def csv_cell(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"cannot serialize non-finite value {value}")
        return format(value, ".17g")
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def csv_bytes(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(fields),
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: csv_cell(row.get(field, "")) for field in fields})
    return output.getvalue().encode("utf-8")


def json_bytes(document: object) -> bytes:
    return (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def canonical_cuda_physics_sha256(row: Mapping[str, str]) -> str:
    """Hash every oracle field except nondeterministic wall/kernel timings."""

    canonical = {
        field: row[field]
        for field in sorted(row)
        if field not in NONPHYSICS_CUDA_FIELDS
    }
    payload = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_requests(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = read_csv(path)
    if len(rows) != 603:
        raise ValueError(f"expected 603 request rows, found {len(rows)}")
    cases = [row["case"] for row in rows]
    if len(set(cases)) != len(cases):
        raise ValueError("request grid has duplicate cases")
    if any(row["delta_exact_row_available"].lower() != "true" for row in rows):
        raise ValueError("request grid contains a case without an exact delta join")
    return rows


def load_cuda_results(
    requests: Sequence[Mapping[str, str]], results_dir: Path
) -> tuple[dict[str, dict[str, str]], list[dict[str, object]]]:
    if not results_dir.is_dir():
        raise FileNotFoundError(results_dir)
    expected_cases = {row["case"] for row in requests}
    actual_paths = sorted(results_dir.glob("ccf_v1_*.csv"))
    actual_cases = {path.stem for path in actual_paths}
    missing = sorted(expected_cases - actual_cases)
    extra = sorted(actual_cases - expected_cases)
    if missing or extra:
        raise ValueError(
            f"CUDA result set mismatch: missing={missing[:5]} ({len(missing)}), "
            f"extra={extra[:5]} ({len(extra)})"
        )

    results: dict[str, dict[str, str]] = {}
    manifest: list[dict[str, object]] = []
    for path in actual_paths:
        rows = read_csv(path)
        if len(rows) != 1:
            raise ValueError(f"expected one CUDA row in {path}, found {len(rows)}")
        row = rows[0]
        case = row.get("case", "")
        if case != path.stem:
            raise ValueError(f"CUDA case/path mismatch: {case!r} != {path.stem!r}")
        if case in results:
            raise ValueError(f"duplicate CUDA case {case}")
        results[case] = row
        manifest.append(
            {
                "case": case,
                "canonical_physics_sha256": canonical_cuda_physics_sha256(row),
            }
        )
    return results, manifest


def validate_cuda_request_match(
    request: Mapping[str, str], result: Mapping[str, str]
) -> None:
    numeric_fields = (
        "tau",
        "ssa",
        "hg_g",
        "sun_zenith_deg",
        "view_zenith_deg",
        "relative_azimuth_deg",
        "samples",
        "seed",
        "max_scatters",
        "batch_samples",
    )
    for field in numeric_fields:
        if not exact_numeric(request[field], result[field]):
            raise ValueError(
                f"CUDA/request mismatch for {request['case']} {field}: "
                f"{request[field]} != {result[field]}"
            )
    if result.get("backend") != "gpu":
        raise ValueError(f"non-GPU result for {request['case']}: {result.get('backend')}")


def delta_index(
    stage0: ModuleType, path: Path
) -> dict[tuple[float, float, float, float, float], dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    index: dict[tuple[float, float, float, float, float], dict[str, str]] = {}
    for row in read_csv(path):
        key = stage0.slab_key(
            finite_float(row, "tau"),
            finite_float(row, "solar_zenith_deg"),
            finite_float(row, "asymmetry_g"),
            finite_float(row, "single_scatter_albedo"),
            finite_float(row, "surface_albedo"),
        )
        if key in index:
            raise ValueError(f"duplicate delta key {key}")
        index[key] = row
    return index


def make_base_rows(
    stage0: ModuleType,
    requests: Sequence[Mapping[str, str]],
    cuda_results: Mapping[str, Mapping[str, str]],
    deltas: Mapping[tuple[float, float, float, float, float], Mapping[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    legacy_cache: dict[tuple[float, float, float, float, float], tuple[float, float]] = {}
    contract = (
        "CUDA/analytic/legacy columns are directional BRF. Delta columns are "
        "hemispheric flux fractions and are not a directional fit target."
    )
    for request in requests:
        cuda = cuda_results[request["case"]]
        validate_cuda_request_match(request, cuda)
        tau = finite_float(request, "tau")
        ssa = finite_float(request, "ssa")
        g = finite_float(request, "hg_g")
        sun = finite_float(request, "sun_zenith_deg")
        view = finite_float(request, "view_zenith_deg")
        azimuth = finite_float(request, "relative_azimuth_deg")
        surface = finite_float(request, "surface_albedo_for_delta_join")

        brf = finite_float(cuda, "brf")
        standard_error = finite_float(cuda, "standard_error_brf")
        ci_low = finite_float(cuda, "ci95_low_brf")
        ci_high = finite_float(cuda, "ci95_high_brf")
        if brf < 0.0 or standard_error < 0.0 or ci_low > brf or ci_high < brf:
            raise ValueError(f"invalid CUDA uncertainty tuple for {request['case']}")
        ci_halfwidth = max(brf - ci_low, ci_high - brf)

        analytic = stage0.analytic_single_scatter_brf(
            tau, ssa, g, sun, view, azimuth
        )
        legacy_key = tuple(round(value, 12) for value in (tau, g, sun, view, azimuth))
        if legacy_key not in legacy_cache:
            octave_one = stage0.analytic_single_scatter_brf(
                tau, 1.0, g, sun, view, azimuth
            )
            octave_six = stage0.legacy_slab_brf(
                tau, g, sun, view, azimuth, stage0.LEGACY_OCTAVES, "hg-proxy"
            )
            legacy_cache[legacy_key] = (octave_one, octave_six)
        octave_one, octave_six = legacy_cache[legacy_key]
        legacy_higher = octave_six - octave_one
        cuda_higher = brf - analytic
        effective_alpha: object = ""
        informative = False
        if tau > 0.0 and legacy_higher > 1.0e-15:
            effective_alpha = cuda_higher / legacy_higher
            informative = cuda_higher - ci_halfwidth > 0.0

        dkey = stage0.slab_key(tau, sun, g, ssa, surface)
        delta = deltas.get(dkey)
        if delta is None:
            raise ValueError(f"missing exact delta row for {request['case']}: {dkey}")
        delta_total = finite_float(delta, "toa_reflectance")
        delta_first = finite_float(delta, "black_surface_single_scatter_reflectance")

        rows.append(
            {
                "case": request["case"],
                "split": request["split"],
                "tau": tau,
                "ssa": ssa,
                "hg_g": g,
                "sun_zenith_deg": sun,
                "view_zenith_deg": view,
                "relative_azimuth_deg": azimuth,
                "surface_albedo_for_delta_join": surface,
                "samples": int(request["samples"]),
                "seed": int(request["seed"]),
                "max_scatters": int(request["max_scatters"]),
                "batch_samples": int(request["batch_samples"]),
                "cuda_backend": cuda["backend"],
                "cuda_device": cuda["device"],
                "cuda_directional_brf": brf,
                "cuda_standard_error_brf": standard_error,
                "cuda_ci95_low_brf": ci_low,
                "cuda_ci95_high_brf": ci_high,
                "cuda_ci95_halfwidth_brf": ci_halfwidth,
                "cuda_truncated_paths": int(cuda["truncated_paths"]),
                "cuda_truncated_fraction": finite_float(cuda, "truncated_fraction"),
                "analytic_hg_single_directional_brf": analytic,
                "cuda_directional_higher_brf": cuda_higher,
                "legacy_hg_proxy_directional_brf_octave1": octave_one,
                "legacy_hg_proxy_directional_brf_octave6": octave_six,
                "legacy_hg_proxy_directional_higher_brf": legacy_higher,
                "effective_alpha_unclamped": effective_alpha,
                "effective_alpha_informative": informative,
                "delta_phase_case": delta["phase_case"],
                "delta_hemispheric_toa_reflectance": delta_total,
                "delta_hemispheric_first_reflectance": delta_first,
                "delta_hemispheric_higher_reflectance": delta_total - delta_first,
                "delta_hemispheric_bottom_direct_transmittance": finite_float(
                    delta, "direct_transmittance_bottom"
                ),
                "delta_hemispheric_bottom_diffuse_transmittance": finite_float(
                    delta, "diffuse_down_bottom"
                ),
                "delta_hemispheric_atmosphere_absorptance": finite_float(
                    delta, "atmosphere_absorptance"
                ),
                "delta_hemispheric_energy_sum": finite_float(delta, "energy_sum"),
                "observable_contract": contract,
            }
        )
    return rows


def bounded_alpha(
    rows: Sequence[Mapping[str, object]], weighted: bool = False
) -> float:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        x = finite_float(row, "legacy_hg_proxy_directional_higher_brf")
        y = finite_float(row, "cuda_directional_higher_brf")
        if x <= 0.0:
            continue
        weight = 1.0
        if weighted:
            error = finite_float(row, "cuda_standard_error_brf")
            weight = 1.0 / max(error * error, 1.0e-18)
        numerator += weight * x * y
        denominator += weight * x * x
    if denominator <= 0.0:
        raise ValueError("no nonzero calibration leverage for bounded alpha")
    return max(0.0, min(1.0, numerator / denominator))


def evaluate_rows(rows: list[dict[str, object]], alpha: float) -> None:
    for row in rows:
        prediction = finite_float(row, "analytic_hg_single_directional_brf") + alpha * finite_float(
            row, "legacy_hg_proxy_directional_higher_brf"
        )
        residual = prediction - finite_float(row, "cuda_directional_brf")
        absolute = abs(residual)
        outside_mc = max(
            0.0, absolute - finite_float(row, "cuda_ci95_halfwidth_brf")
        )
        tolerance = max(0.005, 0.05 * abs(finite_float(row, "cuda_directional_brf")))
        row.update(
            {
                "calibration_alpha_bounded": alpha,
                "predicted_directional_brf": prediction,
                "directional_residual_brf": residual,
                "directional_abs_residual_brf": absolute,
                "directional_abs_residual_outside_mc95": outside_mc,
                "directional_acceptance_tolerance_brf": tolerance,
                "directional_acceptance_pass": outside_mc <= tolerance,
            }
        )


def percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    residuals = [finite_float(row, "directional_residual_brf") for row in rows]
    outside = [
        finite_float(row, "directional_abs_residual_outside_mc95") for row in rows
    ]
    informative_alphas = [
        finite_float(row, "effective_alpha_unclamped")
        for row in rows
        if row["effective_alpha_informative"] is True
    ]
    passes = sum(row["directional_acceptance_pass"] is True for row in rows)
    has_fit_leverage = any(
        finite_float(row, "legacy_hg_proxy_directional_higher_brf") > 0.0
        for row in rows
    )
    return {
        "row_count": len(rows),
        "nonzero_tau_count": sum(finite_float(row, "tau") > 0.0 for row in rows),
        "informative_alpha_count": len(informative_alphas),
        "effective_alpha_unclamped": {
            "minimum": min(informative_alphas) if informative_alphas else None,
            "p10": percentile(informative_alphas, 0.10),
            "median": percentile(informative_alphas, 0.50),
            "p90": percentile(informative_alphas, 0.90),
            "maximum": max(informative_alphas) if informative_alphas else None,
            "fraction_inside_unit_interval": (
                sum(0.0 <= value <= 1.0 for value in informative_alphas)
                / len(informative_alphas)
                if informative_alphas
                else None
            ),
        },
        "best_alpha_bounded_unweighted_for_group": (
            bounded_alpha(rows, weighted=False) if has_fit_leverage else None
        ),
        "directional_brf_error": {
            "rmse": math.sqrt(sum(value * value for value in residuals) / len(residuals)),
            "mean_absolute": sum(abs(value) for value in residuals) / len(residuals),
            "maximum_absolute": max(abs(value) for value in residuals),
            "maximum_absolute_outside_mc95": max(outside),
        },
        "acceptance": {
            "pass_count": passes,
            "fail_count": len(rows) - passes,
            "pass_fraction": passes / len(rows),
        },
        "cuda_quality": {
            "maximum_ci95_halfwidth_brf": max(
                finite_float(row, "cuda_ci95_halfwidth_brf") for row in rows
            ),
            "maximum_truncated_fraction": max(
                finite_float(row, "cuda_truncated_fraction") for row in rows
            ),
            "truncated_case_count": sum(
                finite_float(row, "cuda_truncated_fraction") > 0.0 for row in rows
            ),
        },
    }


def model_metrics_at_alpha(
    rows: Sequence[Mapping[str, object]], alpha: float
) -> dict[str, object]:
    copies = [dict(row) for row in rows]
    evaluate_rows(copies, alpha)
    return metrics(copies)


def group_metrics(
    rows: Sequence[Mapping[str, object]],
    getter: Callable[[Mapping[str, object]], object],
) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        value = getter(row)
        key = json.dumps(value, separators=(",", ":"), sort_keys=True)
        grouped[key].append(row)
    output = []
    for key in sorted(grouped):
        output.append({"value": json.loads(key), "metrics": metrics(grouped[key])})
    return output


def split_counts(rows: Iterable[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["split"])] += 1
    return dict(sorted(counts.items()))


def build_artifacts(args: argparse.Namespace) -> tuple[dict[Path, bytes], dict[str, object]]:
    stage0 = load_stage0(args.stage0_script)
    requests = load_requests(args.request_grid)
    cuda_results, manifest = load_cuda_results(requests, args.results_dir)
    deltas = delta_index(stage0, args.delta_csv)
    rows = make_base_rows(stage0, requests, cuda_results, deltas)

    calibration = [row for row in rows if row["split"] == "calibration"]
    alpha = bounded_alpha(calibration, weighted=False)
    alpha_weighted = bounded_alpha(calibration, weighted=True)
    evaluate_rows(rows, alpha)

    ci_target_failures = [
        row["case"]
        for row in rows
        if finite_float(row, "cuda_ci95_halfwidth_brf")
        > finite_float(requests[int(str(row["case"])[-4:]) - 1], "target_ci95_halfwidth_brf")
        + 1.0e-15
    ]
    zero_rows = [row for row in rows if finite_float(row, "tau") == 0.0]
    if any(finite_float(row, "cuda_directional_brf") != 0.0 for row in zero_rows):
        raise AssertionError("tau-zero CUDA BRF is not exact zero")
    max_delta_energy_error = max(
        abs(finite_float(row, "delta_hemispheric_energy_sum") - 1.0) for row in rows
    )
    surface_values = sorted(
        {finite_float(row, "surface_albedo_for_delta_join") for row in rows}
    )
    devices = sorted({str(row["cuda_device"]) for row in rows})
    maximum_truncation = max(finite_float(row, "cuda_truncated_fraction") for row in rows)

    by_dimension = {
        "tau": group_metrics(rows, lambda row: finite_float(row, "tau")),
        "hg_asymmetry": group_metrics(rows, lambda row: finite_float(row, "hg_g")),
        "single_scatter_albedo": group_metrics(rows, lambda row: finite_float(row, "ssa")),
        "solar_zenith_deg": group_metrics(
            rows, lambda row: finite_float(row, "sun_zenith_deg")
        ),
        "view_zenith_deg": group_metrics(
            rows, lambda row: finite_float(row, "view_zenith_deg")
        ),
        "relative_azimuth_deg": group_metrics(
            rows, lambda row: finite_float(row, "relative_azimuth_deg")
        ),
        "geometry": group_metrics(
            rows,
            lambda row: {
                "sun_zenith_deg": finite_float(row, "sun_zenith_deg"),
                "view_zenith_deg": finite_float(row, "view_zenith_deg"),
                "relative_azimuth_deg": finite_float(row, "relative_azimuth_deg"),
            },
        ),
        "surface_albedo": group_metrics(
            rows, lambda row: finite_float(row, "surface_albedo_for_delta_join")
        ),
        "split": group_metrics(rows, lambda row: str(row["split"])),
    }

    split_summary = {
        name: metrics([row for row in rows if row["split"] == name])
        for name in sorted({str(row["split"]) for row in rows})
    }
    holdout_rows = [row for row in rows if row["split"] != "calibration"]
    calibration_passed = split_summary["calibration"]["acceptance"]["fail_count"] == 0
    holdout_passed = all(row["directional_acceptance_pass"] is True for row in holdout_rows)
    surface_testable = len(surface_values) > 1

    tau_below_30 = [row for row in rows if finite_float(row, "tau") < 30.0]
    tau_below_30_calibration = [
        row for row in tau_below_30 if row["split"] == "calibration"
    ]
    tau_below_30_holdout = [
        row for row in tau_below_30 if row["split"] != "calibration"
    ]
    alpha_tau_below_30 = bounded_alpha(tau_below_30_calibration)
    zero_truncation = [
        row for row in rows if finite_float(row, "cuda_truncated_fraction") == 0.0
    ]
    zero_truncation_calibration = [
        row for row in zero_truncation if row["split"] == "calibration"
    ]
    zero_truncation_holdout = [
        row for row in zero_truncation if row["split"] != "calibration"
    ]
    alpha_zero_truncation = bounded_alpha(zero_truncation_calibration)
    tau_30 = [row for row in rows if finite_float(row, "tau") == 30.0]

    checks = {
        "schema": "simsat.cloud-closure-fit.stage1-self-checks.v1",
        "schema_version": 1,
        "status": "pass",
        "checks": {
            "exact_request_result_join": {
                "request_count": len(requests),
                "result_count": len(rows),
                "unique_case_count": len({str(row["case"]) for row in rows}),
            },
            "split_counts": split_counts(rows),
            "tau_zero": {
                "row_count": len(zero_rows),
                "all_directional_brf_exact_zero": True,
            },
            "cuda_precision_target": {
                "failure_count": len(ci_target_failures),
                "failure_cases": ci_target_failures,
            },
            "cuda_order_truncation": {
                "maximum_truncated_fraction": maximum_truncation,
                "affected_case_count": sum(
                    finite_float(row, "cuda_truncated_fraction") > 0.0 for row in rows
                ),
            },
            "delta_exact_join_energy": {
                "row_count": len(rows),
                "maximum_abs_energy_sum_error": max_delta_energy_error,
            },
            "surface_albedo_coverage": {
                "unique_values": surface_values,
                "stability_testable": surface_testable,
            },
            "observable_separation": {
                "directional_fit_target": "CUDA BRF",
                "hemispheric_flux_used_in_directional_fit": False,
            },
        },
    }

    summary = {
        "schema": "simsat.cloud-closure-fit.stage1-summary.v1",
        "schema_version": 1,
        "input_sha256": {
            str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"): sha256_file(
                Path(__file__).resolve()
            ),
            str(args.request_grid.relative_to(ROOT)).replace("\\", "/"): sha256_file(
                args.request_grid
            ),
            str(args.delta_csv.relative_to(ROOT)).replace("\\", "/"): sha256_file(
                args.delta_csv
            ),
            str(args.stage0_script.relative_to(ROOT)).replace("\\", "/"): sha256_file(
                args.stage0_script
            ),
            str(args.oracle_source.relative_to(ROOT)).replace("\\", "/"): sha256_file(
                args.oracle_source
            ),
            str(args.oracle_exe.relative_to(ROOT)).replace("\\", "/"): sha256_file(
                args.oracle_exe
            ),
        },
        "cuda": {
            "devices": devices,
            "case_count": len(rows),
            "total_paths": sum(int(row["samples"]) for row in rows),
            "maximum_ci95_halfwidth_brf": max(
                finite_float(row, "cuda_ci95_halfwidth_brf") for row in rows
            ),
            "ci95_target_failure_count": len(ci_target_failures),
            "maximum_truncated_fraction": maximum_truncation,
        },
        "observable_contract": {
            "directional": (
                "CUDA, analytic single scatter, and legacy proxy are BRF = pi*I/(F0*mu0)."
            ),
            "hemispheric": (
                "Delta-two-stream columns are upward/downward hemispheric flux fractions."
            ),
            "forbidden_inference": (
                "No directional error, ratio, fit, or production decision mixes delta flux "
                "with CUDA BRF."
            ),
        },
        "bounded_higher_order_model": {
            "formula": (
                "BRF_pred = analytic_single_HG_BRF + alpha * "
                "legacy_HG_proxy_higher_BRF"
            ),
            "fit_split": "calibration only",
            "alpha_bounded_unweighted": alpha,
            "alpha_bounded_inverse_mc_variance_diagnostic": alpha_weighted,
            "acceptance_rule": (
                "max(0, abs(pred-CUDA)-CUDA_95pct_halfwidth) <= "
                "max(0.005, 0.05*abs(CUDA_BRF))"
            ),
            "calibration_all_cases_pass": calibration_passed,
            "all_reserved_holdouts_pass": holdout_passed,
            "production_decision": "do-not-promote",
            "production_decision_reasons": [
                "constant-alpha stability is evaluated below rather than assumed",
                "the directional oracle has only a black lower boundary",
                "the grid has no surface-albedo variation",
                "slab-boundary flux/BRF does not define a depth-resolved marcher source",
                "the single-HG slab omits 3-D cloud geometry and spectral microphysics",
            ],
        },
        "overall_metrics": metrics(rows),
        "split_metrics": split_summary,
        "stability_by_dimension": by_dimension,
        "surface_albedo_coverage": {
            "unique_values": surface_values,
            "stability_testable": surface_testable,
            "conclusion": (
                "black-surface-only; no cross-albedo directional stability claim is possible"
            ),
        },
        "order_truncation_stratification": {
            "tau_below_30": {
                "status": "primary-untruncated-or-negligibly-truncated-sensitivity",
                "alpha_refit_on_calibration": alpha_tau_below_30,
                "all": model_metrics_at_alpha(tau_below_30, alpha_tau_below_30),
                "calibration": model_metrics_at_alpha(
                    tau_below_30_calibration, alpha_tau_below_30
                ),
                "holdout": model_metrics_at_alpha(
                    tau_below_30_holdout, alpha_tau_below_30
                ),
            },
            "exact_zero_truncation_only": {
                "alpha_refit_on_calibration": alpha_zero_truncation,
                "all": model_metrics_at_alpha(zero_truncation, alpha_zero_truncation),
                "calibration": model_metrics_at_alpha(
                    zero_truncation_calibration, alpha_zero_truncation
                ),
                "holdout": model_metrics_at_alpha(
                    zero_truncation_holdout, alpha_zero_truncation
                ),
            },
            "tau_30_provisional": {
                "reason": (
                    "all tau=30 paths have nonzero max-order truncation; reported Monte "
                    "Carlo intervals do not include this bias"
                ),
                "metrics_using_full_grid_alpha_for_diagnosis_only": metrics(tau_30),
                "required_followup": (
                    "rerun paired max_scatters=1024/2048 and require BRF convergence "
                    "inside max(0.002, two pooled standard errors)"
                ),
            },
        },
        "self_checks": checks,
    }

    manifest_payload = csv_bytes(manifest, MANIFEST_FIELDS)
    joined_payload = csv_bytes(rows, JOIN_FIELDS)
    summary["cuda"]["canonical_physics_manifest_sha256"] = hashlib.sha256(
        manifest_payload
    ).hexdigest()
    artifacts = {
        args.output_dir / "stage1-oracle-manifest-v1.csv": manifest_payload,
        args.output_dir / "stage1-exact-join-v1.csv": joined_payload,
        args.output_dir / "stage1-summary-v1.json": json_bytes(summary),
        args.output_dir / "stage1-self-checks-v1.json": json_bytes(checks),
    }
    repeated = {
        args.output_dir / "stage1-oracle-manifest-v1.csv": csv_bytes(
            manifest, MANIFEST_FIELDS
        ),
        args.output_dir / "stage1-exact-join-v1.csv": csv_bytes(rows, JOIN_FIELDS),
        args.output_dir / "stage1-summary-v1.json": json_bytes(summary),
        args.output_dir / "stage1-self-checks-v1.json": json_bytes(checks),
    }
    if artifacts != repeated:
        raise AssertionError("Stage-1 serialization is not byte-repeatable")
    return artifacts, summary


def write_artifacts(artifacts: Mapping[Path, bytes]) -> None:
    for path, payload in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def verify_artifacts(artifacts: Mapping[Path, bytes]) -> None:
    for path, expected in artifacts.items():
        if not path.is_file():
            raise AssertionError(f"missing generated fixture: {path}")
        if path.read_bytes() != expected:
            raise AssertionError(f"generated fixture is stale: {path}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("check", "generate", "all"))
    result.add_argument("--request-grid", type=Path, default=DEFAULT_REQUEST_GRID)
    result.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    result.add_argument("--delta-csv", type=Path, default=DEFAULT_DELTA_CSV)
    result.add_argument("--stage0-script", type=Path, default=DEFAULT_STAGE0_SCRIPT)
    result.add_argument("--oracle-source", type=Path, default=DEFAULT_ORACLE_SOURCE)
    result.add_argument("--oracle-exe", type=Path, default=DEFAULT_ORACLE_EXE)
    result.add_argument("--output-dir", type=Path, default=FIXTURE_DIR)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    required = (args.oracle_source, args.oracle_exe, args.stage0_script)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    artifacts, summary = build_artifacts(args)
    if args.command in ("generate", "all"):
        write_artifacts(artifacts)
    if args.command in ("check", "all"):
        verify_artifacts(artifacts)
    model = summary["bounded_higher_order_model"]
    holdout = summary["split_metrics"]["angular-holdout"]["acceptance"]
    print(
        "cloud-closure-stage1: PASS; "
        f"cases={summary['cuda']['case_count']} "
        f"alpha={model['alpha_bounded_unweighted']:.6f} "
        f"angular_holdout={holdout['pass_count']}/{holdout['pass_count'] + holdout['fail_count']} "
        f"surface_albedo_testable={summary['surface_albedo_coverage']['stability_testable']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, FileNotFoundError, ValueError) as error:
        print(f"cloud-closure-stage1: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
