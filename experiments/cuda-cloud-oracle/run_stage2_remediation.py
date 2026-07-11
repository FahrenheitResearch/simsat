#!/usr/bin/env python3
"""Run the versioned 4096-order remediation for failed Stage-2 tau-30 rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import run_stage2_grid as stage2


HERE = Path(__file__).resolve().parent
DEFAULT_REQUEST = HERE / "stage2-remediation-request-v1.csv"
DEFAULT_RAW = HERE / "build/stage2-remediation-results-v1"
DEFAULT_RESULTS = HERE / "stage2-remediation-results-rtx5090-v1.jsonl"
DEFAULT_SUMMARY = HERE / "stage2-remediation-results-rtx5090-v1-summary.json"

EXTRA_FIELDS = ["source_case_2048", "remediation_reason"]


def generate_request() -> tuple[list[str], list[dict[str, str]]]:
    with stage2.DEFAULT_GRID.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        original = list(reader)
    summary = json.loads(stage2.DEFAULT_SUMMARY.read_text(encoding="utf-8"))
    failed_pairs = {
        failure["pair"]
        for failure in summary["tau30_order_convergence"]["failures"]
        if failure["truncation_failed"]
    }
    if len(failed_pairs) != 16:
        raise ValueError(f"expected 16 failed truncation pairs, found {len(failed_pairs)}")

    selected = [
        row
        for row in original
        if row["convergence_pair"] in failed_pairs and row["max_scatters"] == "2048"
    ]
    selected.sort(key=lambda row: row["convergence_pair"])
    if len(selected) != 16:
        raise ValueError(f"expected 16 source rows, found {len(selected)}")

    remediation: list[dict[str, str]] = []
    for index, source in enumerate(selected, start=1):
        if float(source["tau"]) != 30.0 or float(source["surface_albedo"]) != 0.6:
            raise ValueError(f"unexpected remediation source state: {source['case']}")
        row = dict(source)
        row["source_case_2048"] = source["case"]
        row["case"] = f"ccf_v2r1_{index:04d}"
        row["split"] = "order-cap-remediation"
        row["max_scatters"] = "4096"
        row["runnable_with_current_oracle"] = "true"
        row["remediation_reason"] = (
            "failed-v1-tau30-2048-truncated-fraction-ceiling"
        )
        remediation.append(row)
    return fields + EXTRA_FIELDS, remediation


def request_bytes(fields: list[str], rows: list[dict[str, str]]) -> bytes:
    import io

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def read_original_results() -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    with stage2.DEFAULT_RESULTS.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            results[record["request"]["case"]] = record["result"]
    if len(results) != 800:
        raise ValueError(f"expected 800 immutable original results, found {len(results)}")
    return results


def run(
    oracle: Path,
    rows: list[dict[str, str]],
    raw_dir: Path,
    force: bool,
) -> tuple[list[dict[str, Any]], int, float]:
    if force and raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    executed = 0
    results = []
    for index, row in enumerate(rows, start=1):
        output = raw_dir / f"{row['case']}.json"
        if not output.exists():
            temporary = output.with_suffix(".tmp")
            subprocess.run(
                [str(oracle), *stage2.oracle_arguments(row, temporary)],
                cwd=stage2.ROOT,
                check=True,
            )
            temporary.replace(output)
            executed += 1
        result = json.loads(output.read_text(encoding="utf-8"))
        stage2.validate_result(row, result)
        results.append(result)
        print(f"stage2-remediation: {index}/{len(rows)} validated", file=sys.stderr)
    return results, executed, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, default=stage2.DEFAULT_ORACLE)
    parser.add_argument("--request", type=Path, default=DEFAULT_REQUEST)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    fields, rows = generate_request()
    serialized_request = request_bytes(fields, rows)
    args.request.write_bytes(serialized_request)
    oracle = args.oracle.resolve()
    results, executed, elapsed = run(oracle, rows, args.raw_dir.resolve(), args.force)
    original = read_original_results()

    checks = []
    canonical_lines = []
    for row, result in zip(rows, results, strict=True):
        source = original[row["source_case_2048"]]
        source_directional = source["directional"]
        directional = result["directional"]
        delta = abs(float(directional["brf"]) - float(source_directional["brf"]))
        pooled_two_sigma = 2.0 * math.sqrt(
            float(directional["standard_error_brf"]) ** 2
            + float(source_directional["standard_error_brf"]) ** 2
        )
        tolerance = max(0.002, pooled_two_sigma)
        truncated = float(directional["truncated_fraction"])
        passed = delta <= tolerance and truncated <= 1.0e-4
        checks.append(
            {
                "case": row["case"],
                "source_case_2048": row["source_case_2048"],
                "convergence_pair": row["convergence_pair"],
                "brf_2048": float(source_directional["brf"]),
                "brf_4096": float(directional["brf"]),
                "absolute_brf_delta": delta,
                "brf_tolerance": tolerance,
                "truncated_fraction_2048": float(
                    source_directional["truncated_fraction"]
                ),
                "truncated_fraction_4096": truncated,
                "passed": passed,
            }
        )
        record = {"request": row, "result": stage2.canonical_result(result)}
        canonical_lines.append(
            json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        )

    result_bytes = ("\n".join(canonical_lines) + "\n").encode("utf-8")
    passed = sum(int(check["passed"]) for check in checks)
    summary = {
        "schema": "simsat.cuda-cloud-oracle.stage2-remediation-summary.v1",
        "schema_version": 1,
        "immutable_source_results": "stage2-results-rtx5090-v1.jsonl",
        "remediation": "change only failed bright tau30 max_scatters from 2048 to 4096",
        "rows": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "all_checks_passed": passed == len(rows),
        "max_absolute_brf_delta": max(check["absolute_brf_delta"] for check in checks),
        "max_truncated_fraction_4096": max(
            check["truncated_fraction_4096"] for check in checks
        ),
        "executed_cases": executed,
        "elapsed_seconds": elapsed,
        "checks": checks,
        "production_closure_promoted": False,
        "sha256": {
            "request": stage2.sha256_bytes(serialized_request),
            "immutable_source_results": stage2.sha256_file(stage2.DEFAULT_RESULTS),
            "oracle_source": stage2.sha256_file(HERE / "slab_oracle.cu"),
            "oracle_executable": stage2.sha256_file(oracle),
            "runner": stage2.sha256_file(Path(__file__)),
            "canonical_results": stage2.sha256_bytes(result_bytes),
        },
    }
    args.output.write_bytes(result_bytes)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"stage2-remediation: {'PASS' if summary['all_checks_passed'] else 'BLOCKED'}; "
        f"rows={len(rows)} passed={passed}"
    )
    return 0 if summary["all_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
