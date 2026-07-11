#!/usr/bin/env python3
"""Generate the smallest explicit Stage-2 cloud-reference request grid.

The grid is a request contract, not fabricated oracle output.  It requires a
Lambertian lower boundary, mixture-HG phase sampling, matched forward flux, and
depth-binned source diagnostics that the current black-surface CUDA oracle does
not yet implement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
SEED = 0x53415453494D0016

FIELDS = [
    "case",
    "split",
    "tau",
    "ssa",
    "phase_profile",
    "phase_model",
    "phase_lobe1_g",
    "phase_lobe2_g",
    "phase_lobe1_weight",
    "phase_first_moment",
    "surface_albedo",
    "lower_boundary",
    "sun_zenith_deg",
    "view_zenith_deg",
    "relative_azimuth_deg",
    "samples",
    "seed",
    "max_scatters",
    "batch_samples",
    "depth_bins",
    "report_forward_flux",
    "holdout_albedo",
    "holdout_geometry",
    "holdout_phase_shape",
    "holdout_absorption",
    "convergence_pair",
    "required_oracle_features",
    "runnable_with_current_oracle",
]

PHASE_PROFILES = [
    {
        "phase_profile": "single-hg-g0p665",
        "phase_model": "single-hg",
        "phase_lobe1_g": 0.665,
        "phase_lobe2_g": 0.0,
        "phase_lobe1_weight": 1.0,
        "phase_first_moment": 0.665,
    },
    {
        "phase_profile": "single-hg-g0p75",
        "phase_model": "single-hg",
        "phase_lobe1_g": 0.75,
        "phase_lobe2_g": 0.0,
        "phase_lobe1_weight": 1.0,
        "phase_first_moment": 0.75,
    },
    {
        "phase_profile": "single-hg-g0p85",
        "phase_model": "single-hg",
        "phase_lobe1_g": 0.85,
        "phase_lobe2_g": 0.0,
        "phase_lobe1_weight": 1.0,
        "phase_first_moment": 0.85,
    },
    {
        "phase_profile": "shipping-liquid-dual-hg",
        "phase_model": "mixture-hg",
        "phase_lobe1_g": 0.85,
        "phase_lobe2_g": -0.15,
        "phase_lobe1_weight": 0.9,
        "phase_first_moment": 0.75,
    },
    {
        "phase_profile": "shipping-ice-dual-hg",
        "phase_model": "mixture-hg",
        "phase_lobe1_g": 0.75,
        "phase_lobe2_g": -0.10,
        "phase_lobe1_weight": 0.9,
        "phase_first_moment": 0.665,
    },
]

GEOMETRIES = [(0.0, 0.0), (40.0, 90.0), (65.0, 180.0)]
CONVERGENCE_GEOMETRIES = [(0.0, 0.0), (65.0, 180.0)]
REQUIRED_FEATURES = (
    "lambertian-lower-boundary;mixture-hg-sampling;matched-forward-RTA;"
    "depth-binned-collision-source"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_cell(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"cannot serialize non-finite float {value}")
        return format(value, ".17g")
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=FIELDS, lineterminator="\n", extrasaction="raise"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: csv_cell(row[field]) for field in FIELDS})
    return output.getvalue().encode("utf-8")


def json_bytes(document: object) -> bytes:
    return (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def split_name(
    *,
    albedo: float,
    view: float,
    azimuth: float,
    phase_model: str,
    ssa: float,
    convergence_pair: str,
) -> tuple[str, bool, bool, bool, bool]:
    tags: list[str] = []
    albedo_holdout = albedo in (0.2, 0.85)
    geometry_holdout = view == 40.0 and azimuth == 90.0
    phase_holdout = phase_model == "mixture-hg"
    absorption_holdout = ssa == 0.95
    if albedo == 0.2:
        tags.append("albedo-interpolation-holdout")
    elif albedo == 0.85:
        tags.append("albedo-extrapolation-holdout")
    if geometry_holdout:
        tags.append("angular-holdout")
    if phase_holdout:
        tags.append("phase-shape-holdout")
    if absorption_holdout:
        tags.append("absorption-holdout")
    if convergence_pair:
        tags.append("order-convergence-holdout")
    return (
        "+".join(tags) if tags else "calibration",
        albedo_holdout,
        geometry_holdout,
        phase_holdout,
        absorption_holdout,
    )


def add_row(
    rows: list[dict[str, object]],
    *,
    tau: float,
    ssa: float,
    phase: Mapping[str, object],
    albedo: float,
    sun: float,
    view: float,
    azimuth: float,
    max_scatters: int,
    convergence_pair: str = "",
) -> None:
    split, albedo_holdout, geometry_holdout, phase_holdout, absorption_holdout = split_name(
        albedo=albedo,
        view=view,
        azimuth=azimuth,
        phase_model=str(phase["phase_model"]),
        ssa=ssa,
        convergence_pair=convergence_pair,
    )
    rows.append(
        {
            "case": f"ccf_v2_{len(rows) + 1:04d}",
            "split": split,
            "tau": tau,
            "ssa": ssa,
            **phase,
            "surface_albedo": albedo,
            "lower_boundary": "lambertian",
            "sun_zenith_deg": sun,
            "view_zenith_deg": view,
            "relative_azimuth_deg": azimuth,
            "samples": 8388608 if tau == 30.0 else 4194304,
            "seed": SEED,
            "max_scatters": max_scatters,
            "batch_samples": 65536,
            "depth_bins": 32,
            "report_forward_flux": True,
            "holdout_albedo": albedo_holdout,
            "holdout_geometry": geometry_holdout,
            "holdout_phase_shape": phase_holdout,
            "holdout_absorption": absorption_holdout,
            "convergence_pair": convergence_pair,
            "required_oracle_features": REQUIRED_FEATURES,
            "runnable_with_current_oracle": False,
        }
    )


def request_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    # Near-conservative core: endpoints A=0/0.6 train the surface dependence;
    # A=0.2 is a reserved interpolation test.
    for tau in (0.1, 0.3, 1.0, 3.0, 10.0):
        for phase in PHASE_PROFILES:
            for albedo in (0.0, 0.2, 0.6):
                for sun in (30.0, 65.0):
                    for view, azimuth in GEOMETRIES:
                        add_row(
                            rows,
                            tau=tau,
                            ssa=0.999,
                            phase=phase,
                            albedo=albedo,
                            sun=sun,
                            view=view,
                            azimuth=azimuth,
                            max_scatters=384,
                        )

    # Absorption is entirely reserved so it tests, rather than helps identify,
    # the conservative fit's SSA dependence.
    for tau in (0.3, 3.0, 10.0):
        for phase in PHASE_PROFILES:
            for albedo in (0.0, 0.6):
                for sun in (30.0, 65.0):
                    for view, azimuth in GEOMETRIES:
                        add_row(
                            rows,
                            tau=tau,
                            ssa=0.95,
                            phase=phase,
                            albedo=albedo,
                            sun=sun,
                            view=view,
                            azimuth=azimuth,
                            max_scatters=384,
                        )

    # Snow/bright-cloud-deck boundary is an extrapolation holdout.
    for tau in (0.3, 3.0, 10.0):
        for phase in PHASE_PROFILES:
            for sun in (30.0, 65.0):
                for view, azimuth in GEOMETRIES:
                    add_row(
                        rows,
                        tau=tau,
                        ssa=0.999,
                        phase=phase,
                        albedo=0.85,
                        sun=sun,
                        view=view,
                        azimuth=azimuth,
                        max_scatters=384,
                    )

    # Every tau=30 state is paired at 1024/2048 orders; neither is allowed into a
    # fit until the pair converges inside the declared threshold.
    for phase in PHASE_PROFILES:
        for albedo in (0.0, 0.6):
            for sun in (30.0, 65.0):
                for view, azimuth in CONVERGENCE_GEOMETRIES:
                    for max_scatters in (1024, 2048):
                        add_row(
                            rows,
                            tau=30.0,
                            ssa=0.999,
                            phase=phase,
                            albedo=albedo,
                            sun=sun,
                            view=view,
                            azimuth=azimuth,
                            max_scatters=max_scatters,
                            convergence_pair=(
                                f"tau30-{phase['phase_profile']}-a{albedo:g}-"
                                f"s{sun:g}-v{view:g}-r{azimuth:g}"
                            ),
                        )

    if len(rows) != 800:
        raise AssertionError(f"expected 800 Stage-2 requests, generated {len(rows)}")
    keys = {
        (
            row["tau"],
            row["ssa"],
            row["phase_profile"],
            row["surface_albedo"],
            row["sun_zenith_deg"],
            row["view_zenith_deg"],
            row["relative_azimuth_deg"],
            row["max_scatters"],
        )
        for row in rows
    }
    if len(keys) != len(rows):
        raise AssertionError("Stage-2 grid contains duplicate physical requests")
    return rows


def true_count(rows: Iterable[Mapping[str, object]], field: str) -> int:
    return sum(row[field] is True for row in rows)


def build_artifacts(output_dir: Path) -> tuple[dict[Path, bytes], dict[str, object]]:
    rows = request_rows()
    grid = csv_bytes(rows)
    split_counts = Counter(str(row["split"]) for row in rows)
    phase_counts = Counter(str(row["phase_profile"]) for row in rows)
    convergence_groups = Counter(
        str(row["convergence_pair"])
        for row in rows
        if str(row["convergence_pair"])
    )
    if set(convergence_groups.values()) != {2}:
        raise AssertionError("each tau=30 convergence group must contain exactly two orders")

    summary = {
        "schema": "simsat.cloud-closure-fit.stage2-request-summary.v1",
        "schema_version": 1,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "grid_sha256": hashlib.sha256(grid).hexdigest(),
        "row_count": len(rows),
        "total_requested_paths": sum(int(row["samples"]) for row in rows),
        "runnable_with_current_oracle": False,
        "blocker": (
            "the current oracle has only a black lower boundary, one HG lobe, no "
            "matched per-state forward R/T/A, and no depth-binned collision source"
        ),
        "smallest_next_oracle_capabilities": REQUIRED_FEATURES.split(";"),
        "coverage": {
            "tau": sorted({float(row["tau"]) for row in rows}),
            "ssa": sorted({float(row["ssa"]) for row in rows}),
            "surface_albedo": sorted(
                {float(row["surface_albedo"]) for row in rows}
            ),
            "solar_zenith_deg": sorted(
                {float(row["sun_zenith_deg"]) for row in rows}
            ),
            "phase_profiles": dict(sorted(phase_counts.items())),
            "split_counts": dict(sorted(split_counts.items())),
            "holdout_axis_counts": {
                "albedo": true_count(rows, "holdout_albedo"),
                "geometry": true_count(rows, "holdout_geometry"),
                "phase_shape": true_count(rows, "holdout_phase_shape"),
                "absorption": true_count(rows, "holdout_absorption"),
            },
        },
        "matched_first_moment_phase_tests": [
            {
                "phase_first_moment": 0.75,
                "control": "single-hg-g0p75",
                "holdout": "shipping-liquid-dual-hg",
            },
            {
                "phase_first_moment": 0.665,
                "control": "single-hg-g0p665",
                "holdout": "shipping-ice-dual-hg",
            },
        ],
        "tau30_order_convergence": {
            "group_count": len(convergence_groups),
            "orders": [1024, 2048],
            "acceptance": (
                "absolute BRF difference <= max(0.002, two pooled Monte Carlo "
                "standard errors) and truncated fraction <= 1e-4 at 2048"
            ),
        },
        "effective_radius_policy": (
            "do not use effective radius as a phase surrogate until a wavelength-"
            "specific liquid/ice phase table is selected and recorded by immutable hash"
        ),
    }
    artifacts = {
        output_dir / "stage2-request-grid-v1.csv": grid,
        output_dir / "stage2-request-summary-v1.json": json_bytes(summary),
    }
    repeated = {
        output_dir / "stage2-request-grid-v1.csv": csv_bytes(rows),
        output_dir / "stage2-request-summary-v1.json": json_bytes(summary),
    }
    if artifacts != repeated:
        raise AssertionError("Stage-2 request serialization is not byte-repeatable")
    return artifacts, summary


def write_artifacts(artifacts: Mapping[Path, bytes]) -> None:
    for path, payload in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def verify_artifacts(artifacts: Mapping[Path, bytes]) -> None:
    for path, payload in artifacts.items():
        if not path.is_file() or path.read_bytes() != payload:
            raise AssertionError(f"missing or stale Stage-2 fixture: {path}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("check", "generate", "all"))
    result.add_argument("--output-dir", type=Path, default=FIXTURE_DIR)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    artifacts, summary = build_artifacts(args.output_dir)
    if args.command in ("generate", "all"):
        write_artifacts(artifacts)
    if args.command in ("check", "all"):
        verify_artifacts(artifacts)
    print(
        "cloud-closure-stage2-request: PASS; "
        f"rows={summary['row_count']} "
        f"paths={summary['total_requested_paths']} "
        f"runnable={summary['runnable_with_current_oracle']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, FileNotFoundError, ValueError) as error:
        print(f"cloud-closure-stage2-request: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
