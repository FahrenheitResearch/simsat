#!/usr/bin/env python3
"""Run and validate the exact 800-row Stage-2 CUDA cloud-oracle request.

Raw, timing-bearing per-case results stay under the ignored build directory.
The committed JSONL strips only wall/kernel timing fields; every requested
input, path count, directional statistic, forward outcome, and depth-bin count
is retained exactly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_GRID = ROOT / "experiments/cloud-closure-fit/fixtures/stage2-request-grid-v1.csv"
DEFAULT_ORACLE = HERE / "build/slab_oracle.exe"
DEFAULT_RAW = HERE / "build/stage2-requested-results"
DEFAULT_REPEAT_RAW = HERE / "build/stage2-repeat-results"
DEFAULT_RESULTS = HERE / "stage2-results-rtx5090-v1.jsonl"
DEFAULT_SUMMARY = HERE / "stage2-results-rtx5090-v1-summary.json"

CAPABILITIES = {
    "lambertian_lower_boundary": 1,
    "mixture_hg_sampling": 1,
    "matched_forward_rta": 1,
    "depth_binned_collision_source": 1,
}
TIMING_FIELDS = {"elapsed_ms", "kernel_ms", "max_batch_ms"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_grid(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 800:
        raise ValueError(f"expected 800 Stage-2 requests, found {len(rows)}")
    cases = [row["case"] for row in rows]
    if len(set(cases)) != len(cases):
        raise ValueError("duplicate case identifiers in Stage-2 request")
    for row in rows:
        if row["required_oracle_features"] != (
            "lambertian-lower-boundary;mixture-hg-sampling;matched-forward-RTA;"
            "depth-binned-collision-source"
        ):
            raise ValueError(f"unexpected capability contract in {row['case']}")
        if row["lower_boundary"] != "lambertian":
            raise ValueError(f"unsupported lower boundary in {row['case']}")
        if row["depth_bins"] != "32" or row["report_forward_flux"] != "true":
            raise ValueError(f"incomplete Stage-2 diagnostics request in {row['case']}")
    return rows


def oracle_arguments(row: Mapping[str, str], output: Path) -> list[str]:
    return [
        "--backend", "gpu",
        "--format", "json",
        "--output", str(output),
        "--case", row["case"],
        "--tau", row["tau"],
        "--ssa", row["ssa"],
        "--phase-model", row["phase_model"],
        "--phase-lobe1-g", row["phase_lobe1_g"],
        "--phase-lobe2-g", row["phase_lobe2_g"],
        "--phase-lobe1-weight", row["phase_lobe1_weight"],
        "--phase-first-moment", row["phase_first_moment"],
        "--surface-albedo", row["surface_albedo"],
        "--lower-boundary", row["lower_boundary"],
        "--sun-zenith-deg", row["sun_zenith_deg"],
        "--view-zenith-deg", row["view_zenith_deg"],
        "--relative-azimuth-deg", row["relative_azimuth_deg"],
        "--samples", row["samples"],
        "--seed", row["seed"],
        "--max-scatters", row["max_scatters"],
        "--batch-samples", row["batch_samples"],
        "--depth-bins", row["depth_bins"],
        "--report-forward-flux", row["report_forward_flux"],
    ]


def finite_tree(value: Any, path: str = "result") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite number at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_tree(child, f"{path}[{index}]")


def same_float(actual: Any, expected: str) -> bool:
    return float(actual) == float(expected)


def validate_result(row: Mapping[str, str], result: Mapping[str, Any]) -> dict[str, float]:
    finite_tree(result)
    case = row["case"]
    if result.get("schema") != "simsat.cuda-cloud-oracle.result.v2":
        raise ValueError(f"{case}: wrong result schema")
    if result.get("schema_version") != 2 or result.get("capability_set") != "stage2-reference-v1":
        raise ValueError(f"{case}: wrong schema/capability version")
    if result.get("capabilities") != CAPABILITIES:
        raise ValueError(f"{case}: wrong capability declaration")
    if result.get("request_contract") != "simsat.cloud-closure-fit.stage2-request-grid.v1":
        raise ValueError(f"{case}: wrong request contract")
    if result.get("case") != case or result.get("backend") != "gpu":
        raise ValueError(f"{case}: case/backend mismatch")
    if "RTX 5090" not in str(result.get("device")):
        raise ValueError(f"{case}: result is not from the RTX 5090")

    numeric_inputs = {
        "tau": "tau",
        "ssa": "ssa",
        "phase_lobe1_g": "phase_lobe1_g",
        "phase_lobe2_g": "phase_lobe2_g",
        "phase_lobe1_weight": "phase_lobe1_weight",
        "phase_first_moment": "phase_first_moment",
        "surface_albedo": "surface_albedo",
        "sun_zenith_deg": "sun_zenith_deg",
        "view_zenith_deg": "view_zenith_deg",
        "relative_azimuth_deg": "relative_azimuth_deg",
    }
    for result_field, request_field in numeric_inputs.items():
        if not same_float(result[result_field], row[request_field]):
            raise ValueError(f"{case}: {result_field} does not match request")
    integer_inputs = ("samples", "seed", "max_scatters", "batch_samples", "depth_bins")
    for field in integer_inputs:
        if int(result[field]) != int(row[field]):
            raise ValueError(f"{case}: {field} does not match request")
    if result["phase_model"] != row["phase_model"]:
        raise ValueError(f"{case}: phase model mismatch")
    if result["lower_boundary"] != "lambertian" or not result["report_forward_flux"]:
        raise ValueError(f"{case}: lower-boundary/forward request mismatch")

    directional = result["directional"]
    if directional["brf"] < 0.0 or directional["standard_error_brf"] < 0.0:
        raise ValueError(f"{case}: invalid directional statistic")
    if directional["truncated_paths"] > result["samples"]:
        raise ValueError(f"{case}: invalid directional truncation count")

    flux = result["forward_flux"]
    if flux is None or int(flux["samples"]) != int(row["samples"]):
        raise ValueError(f"{case}: missing or unmatched forward flux")
    outcome_sum = sum(
        int(flux[field])
        for field in ("reflected_paths", "transmitted_paths", "absorbed_paths", "truncated_paths")
    )
    if outcome_sum != int(row["samples"]):
        raise ValueError(f"{case}: forward outcomes do not classify every path")
    if abs(float(flux["closure"]) - 1.0) > 4.0e-15:
        raise ValueError(f"{case}: forward R/T/A/truncation does not close")

    source = result["collision_source"]
    bins = source["bins"]
    if source["bin_count"] != 32 or len(bins) != 32:
        raise ValueError(f"{case}: expected 32 collision-source bins")
    source_total = 0
    collision_total = 0
    absorption_total = 0
    for index, bin_result in enumerate(bins):
        if bin_result["index"] != index:
            raise ValueError(f"{case}: noncanonical depth-bin index")
        collisions = int(bin_result["collision_count"])
        scattered = int(bin_result["scattering_source_count"])
        absorbed = int(bin_result["absorption_count"])
        if collisions != scattered + absorbed:
            raise ValueError(f"{case}: depth-bin collision accounting mismatch")
        expected_per_path = scattered / int(row["samples"])
        expected_density = expected_per_path * 32.0
        if float(bin_result["scattering_source_per_incident_path"]) != expected_per_path:
            raise ValueError(f"{case}: depth-bin source normalization mismatch")
        if float(bin_result["scattering_source_density"]) != expected_density:
            raise ValueError(f"{case}: depth-bin source density mismatch")
        collision_total += collisions
        source_total += scattered
        absorption_total += absorbed
    if absorption_total != int(flux["absorbed_paths"]):
        raise ValueError(f"{case}: volume absorption bins do not match A")

    return {
        "ci95_halfwidth": 1.96 * float(directional["standard_error_brf"]),
        "directional_truncated_fraction": float(directional["truncated_fraction"]),
        "forward_truncated_fraction": float(flux["truncated_fraction"]),
        "closure_error": abs(float(flux["closure"]) - 1.0),
        "collision_total": float(collision_total),
        "source_total": float(source_total),
    }


def canonical_result(result: Mapping[str, Any]) -> dict[str, Any]:
    canonical = json.loads(json.dumps(result))
    for section in ("directional", "forward_flux"):
        if canonical.get(section) is not None:
            for field in TIMING_FIELDS:
                canonical[section].pop(field, None)
    return canonical


def canonical_bytes(rows: Iterable[Mapping[str, str]], results: Iterable[Mapping[str, Any]]) -> bytes:
    lines = []
    for row, result in zip(rows, results, strict=True):
        record = {"request": dict(row), "result": canonical_result(result)}
        lines.append(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    return ("\n".join(lines) + "\n").encode("utf-8")


def run_grid(
    oracle: Path,
    rows: list[dict[str, str]],
    raw_dir: Path,
    force: bool,
) -> tuple[list[dict[str, Any]], float, int]:
    if force and raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    executed_cases = 0
    for index, row in enumerate(rows, start=1):
        output = raw_dir / f"{row['case']}.json"
        if not output.exists():
            temporary = output.with_suffix(".tmp")
            subprocess.run(
                [str(oracle), *oracle_arguments(row, temporary)],
                cwd=ROOT,
                check=True,
            )
            temporary.replace(output)
            executed_cases += 1
        with output.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        validate_result(row, result)
        results.append(result)
        if index == 1 or index % 25 == 0 or index == len(rows):
            elapsed = time.perf_counter() - started
            print(
                f"stage2-grid: {index}/{len(rows)} validated in {elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )
    return results, time.perf_counter() - started, executed_cases


def validate_partial_repeat(
    rows: list[dict[str, str]],
    primary: list[dict[str, Any]],
    repeat_dir: Path,
) -> dict[str, Any] | None:
    if not repeat_dir.is_dir():
        return None
    compared = 0
    for row, expected in zip(rows, primary, strict=True):
        path = repeat_dir / f"{row['case']}.json"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            repeated = json.load(handle)
        validate_result(row, repeated)
        if canonical_result(repeated) != canonical_result(expected):
            raise ValueError(f"{row['case']}: partial repeat differs after timing removal")
        compared += 1
    if compared == 0:
        return None
    return {
        "scope": "partial-grid-independent-rerun",
        "completed_rows": compared,
        "exact_matches_after_timing_removal": compared,
        "all_compared_rows_exact": True,
    }


def convergence_checks(
    rows: list[dict[str, str]], results: list[dict[str, Any]]
) -> dict[str, Any]:
    groups: dict[str, list[tuple[dict[str, str], dict[str, Any]]]] = defaultdict(list)
    for row, result in zip(rows, results, strict=True):
        if row["convergence_pair"]:
            groups[row["convergence_pair"]].append((row, result))
    if len(groups) != 40 or any(len(group) != 2 for group in groups.values()):
        raise ValueError("expected 40 paired tau=30 convergence groups")

    failures = []
    brf_failures = 0
    truncation_failures = 0
    max_brf_delta = 0.0
    max_2048_truncated = 0.0
    for name, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda item: int(item[0]["max_scatters"]))
        low_row, low_result = ordered[0]
        high_row, high_result = ordered[1]
        if (int(low_row["max_scatters"]), int(high_row["max_scatters"])) != (1024, 2048):
            raise ValueError(f"{name}: expected 1024/2048 order pair")
        low = low_result["directional"]
        high = high_result["directional"]
        delta = abs(float(low["brf"]) - float(high["brf"]))
        pooled = math.sqrt(
            float(low["standard_error_brf"]) ** 2
            + float(high["standard_error_brf"]) ** 2
        )
        tolerance = max(0.002, 2.0 * pooled)
        high_truncated = float(high["truncated_fraction"])
        max_brf_delta = max(max_brf_delta, delta)
        max_2048_truncated = max(max_2048_truncated, high_truncated)
        brf_failed = delta > tolerance
        truncation_failed = high_truncated > 1.0e-4
        brf_failures += int(brf_failed)
        truncation_failures += int(truncation_failed)
        if brf_failed or truncation_failed:
            failures.append(
                {
                    "pair": name,
                    "brf_delta": delta,
                    "tolerance": tolerance,
                    "truncated_fraction_2048": high_truncated,
                    "brf_failed": brf_failed,
                    "truncation_failed": truncation_failed,
                }
            )
    return {
        "group_count": len(groups),
        "passed": len(groups) - len(failures),
        "failed": len(failures),
        "max_brf_delta": max_brf_delta,
        "max_2048_truncated_fraction": max_2048_truncated,
        "brf_convergence_failures": brf_failures,
        "truncation_failures": truncation_failures,
        "failures": failures,
    }


def write_outputs(
    rows: list[dict[str, str]],
    results: list[dict[str, Any]],
    result_bytes: bytes,
    output: Path,
    summary_path: Path,
    grid: Path,
    oracle: Path,
    elapsed_seconds: float,
    executed_cases: int,
    repeat: dict[str, Any] | None,
) -> bool:
    metrics = [validate_result(row, result) for row, result in zip(rows, results, strict=True)]
    convergence = convergence_checks(rows, results)
    summary = {
        "schema": "simsat.cuda-cloud-oracle.stage2-grid-summary.v1",
        "schema_version": 1,
        "request_contract": "simsat.cloud-closure-fit.stage2-request-grid.v1",
        "result_schema": "simsat.cuda-cloud-oracle.result.v2",
        "capability_set": "stage2-reference-v1",
        "capabilities": CAPABILITIES,
        "device": results[0]["device"],
        "rows": len(rows),
        "requested_paths_per_direction": sum(int(row["samples"]) for row in rows),
        "executed_backward_paths": sum(int(row["samples"]) for row in rows),
        "executed_forward_paths": sum(int(row["samples"]) for row in rows),
        "depth_bins_per_case": 32,
        "max_ci95_halfwidth_brf": max(metric["ci95_halfwidth"] for metric in metrics),
        "max_directional_truncated_fraction": max(
            metric["directional_truncated_fraction"] for metric in metrics
        ),
        "max_forward_truncated_fraction": max(
            metric["forward_truncated_fraction"] for metric in metrics
        ),
        "max_forward_closure_error": max(metric["closure_error"] for metric in metrics),
        "total_collision_count": int(sum(metric["collision_total"] for metric in metrics)),
        "total_scattering_source_count": int(sum(metric["source_total"] for metric in metrics)),
        "tau30_order_convergence": convergence,
        "all_checks_passed": convergence["failed"] == 0,
        "runner_elapsed_seconds": elapsed_seconds,
        "runner_executed_cases": executed_cases,
        "raw_timing_sums_ms": {
            "directional_wall": sum(
                float(result["directional"]["elapsed_ms"]) for result in results
            ),
            "directional_kernel": sum(
                float(result["directional"]["kernel_ms"]) for result in results
            ),
            "forward_wall": sum(
                float(result["forward_flux"]["elapsed_ms"]) for result in results
            ),
            "forward_kernel": sum(
                float(result["forward_flux"]["kernel_ms"]) for result in results
            ),
        },
        "repeat": repeat,
        "sha256": {
            "request_grid": sha256_file(grid),
            "grid_runner": sha256_file(Path(__file__)),
            "oracle_source": sha256_file(HERE / "slab_oracle.cu"),
            "oracle_executable": sha256_file(oracle),
            "legacy_baseline": sha256_file(HERE / "baseline-rtx5090.csv"),
            "self_test": sha256_file(HERE / "self-test-rtx5090.json"),
            "canonical_results": sha256_bytes(result_bytes),
        },
    }
    if not summary["all_checks_passed"]:
        summary["smallest_blocker"] = (
            f"{convergence['truncation_failures']} of 40 tau=30 pairs exceed the "
            "predeclared 2048-order truncated-fraction ceiling of 1e-4; "
            f"{40 - convergence['brf_convergence_failures']}/40 satisfy the BRF-delta test"
        )
    output.write_bytes(result_bytes)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return bool(summary["all_checks_passed"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--repeat-raw-dir", type=Path, default=DEFAULT_REPEAT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()

    grid = args.grid.resolve()
    oracle = args.oracle.resolve()
    if not grid.is_file() or not oracle.is_file():
        raise FileNotFoundError("Stage-2 grid or CUDA oracle executable is missing")
    rows = read_grid(grid)
    results, elapsed, executed_cases = run_grid(
        oracle, rows, args.raw_dir.resolve(), args.force
    )
    result_bytes = canonical_bytes(rows, results)

    repeat_summary = None
    if args.repeat:
        repeated, repeat_elapsed, repeat_executed_cases = run_grid(
            oracle, rows, args.repeat_raw_dir.resolve(), args.force
        )
        repeated_bytes = canonical_bytes(rows, repeated)
        if repeated_bytes != result_bytes:
            raise ValueError("full Stage-2 repeat is not byte-identical after timing removal")
        repeat_summary = {
            "byte_identical_after_timing_removal": True,
            "elapsed_seconds": repeat_elapsed,
            "executed_cases": repeat_executed_cases,
            "sha256": sha256_bytes(repeated_bytes),
        }
    else:
        repeat_summary = validate_partial_repeat(
            rows, results, args.repeat_raw_dir.resolve()
        )

    all_checks_passed = write_outputs(
        rows,
        results,
        result_bytes,
        args.output.resolve(),
        args.summary.resolve(),
        grid,
        oracle,
        elapsed,
        executed_cases,
        repeat_summary,
    )
    print(
        f"stage2-grid: {'PASS' if all_checks_passed else 'BLOCKED'}; "
        f"rows={len(rows)} canonical_sha256={sha256_bytes(result_bytes)} "
        f"repeat={bool(repeat_summary)}"
    )
    return 0 if all_checks_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
