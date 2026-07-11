#!/usr/bin/env python3
"""Deterministic Stage-0 cloud-closure comparison and request-grid harness.

This script intentionally compares three different quantities without pretending
they are interchangeable:

* the delta-two-stream fixture reports hemispheric slab fluxes;
* the CUDA oracle reports directional bidirectional reflectance factor (BRF);
* the legacy Wrenninge/Oz diagnostic below integrates the shipping local source
  through an idealised homogeneous slab to obtain a directional BRF analogue.

Only exact common inputs are joined.  Missing cases become an explicit request
grid; no interpolation is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = EXPERIMENT_DIR / "fixtures"

DEFAULT_DELTA_CSV = (
    ROOT / "experiments/delta-two-stream/fixtures/stage0-slab-flux-v1.csv"
)
DEFAULT_DELTA_JSON = (
    ROOT / "experiments/delta-two-stream/fixtures/stage0-slab-flux-v1.json"
)
DEFAULT_DELTA_CHECKS = (
    ROOT / "experiments/delta-two-stream/fixtures/stage0-self-checks-v1.json"
)
DEFAULT_CUDA_CSV = ROOT / "experiments/cuda-cloud-oracle/baseline-rtx5090.csv"
DEFAULT_CUDA_CHECKS = ROOT / "experiments/cuda-cloud-oracle/self-test-rtx5090.json"

PI = math.pi

# Exact production constants in crates/simsat/src/clouds.rs at the audit SHA.
LEGACY_OCTAVES = 6
LEGACY_EXTINCTION_SCALE = 0.5
LEGACY_PHASE_SCALE = 0.5
LEGACY_BRIGHTNESS_SCALE = 0.85
LEGACY_WEIGHT_SUM = sum(
    LEGACY_BRIGHTNESS_SCALE**order for order in range(LEGACY_OCTAVES)
)
LIQUID_PHASE = (0.85, -0.15, 0.9)
ICE_PHASE = (0.75, -0.10, 0.9)

# Composite Simpson resolution.  The one-dimensional source is smooth except for
# one known max() kink, and this resolution puts the one-octave result within
# 1e-10 of the analytic single-scatter expression on the reference cases.
SLAB_INTERVALS = 8192

COMPARISON_FIELDS = [
    "case",
    "match_status",
    "delta_phase_case",
    "tau",
    "ssa",
    "hg_g",
    "sun_zenith_deg",
    "view_zenith_deg",
    "relative_azimuth_deg",
    "scattering_cosine",
    "cuda_directional_brf",
    "cuda_standard_error_brf",
    "cuda_ci95_low_brf",
    "cuda_ci95_high_brf",
    "analytic_hg_single_brf",
    "cuda_directional_higher_brf",
    "legacy_hg_proxy_brf_octave1",
    "legacy_hg_proxy_brf_octave6",
    "legacy_hg_proxy_higher_brf",
    "legacy_higher_scale_to_cuda",
    "legacy_liquid_dual_hg_brf_octave1",
    "legacy_liquid_dual_hg_brf_octave6",
    "legacy_ice_dual_hg_brf_octave1",
    "legacy_ice_dual_hg_brf_octave6",
    "legacy_sphere_source_top",
    "legacy_sphere_source_mid",
    "legacy_sphere_source_bottom",
    "legacy_sphere_source_top_over_direct",
    "delta_hemispheric_toa_reflectance",
    "delta_hemispheric_first_reflectance",
    "delta_hemispheric_higher_reflectance",
    "delta_hemispheric_bottom_direct_transmittance",
    "delta_hemispheric_bottom_diffuse_transmittance",
    "delta_atmosphere_absorptance",
    "delta_energy_sum",
    "observable_contract",
]

REQUEST_FIELDS = [
    "case",
    "split",
    "tau",
    "ssa",
    "hg_g",
    "sun_zenith_deg",
    "view_zenith_deg",
    "relative_azimuth_deg",
    "surface_albedo_for_delta_join",
    "delta_exact_row_available",
    "samples",
    "seed",
    "max_scatters",
    "batch_samples",
    "target_ci95_halfwidth_brf",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def script_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: Mapping[str, object], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field}: {value!r}")
    return value


def canonical_float(value: float) -> float:
    return round(float(value), 12)


def slab_key(
    tau: float,
    sun_zenith_deg: float,
    asymmetry_g: float,
    ssa: float,
    surface_albedo: float,
) -> tuple[float, float, float, float, float]:
    return tuple(
        canonical_float(value)
        for value in (tau, sun_zenith_deg, asymmetry_g, ssa, surface_albedo)
    )  # type: ignore[return-value]


def validate_inputs(
    delta_csv_path: Path,
    delta_json_path: Path,
    delta_checks_path: Path,
    cuda_csv_path: Path,
    cuda_checks_path: Path,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    dict[tuple[float, float, float, float, float], dict[str, str]],
    dict[str, str],
]:
    required = [
        delta_csv_path,
        delta_json_path,
        delta_checks_path,
        cuda_csv_path,
        cuda_checks_path,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    delta_rows = read_csv(delta_csv_path)
    delta_document = read_json(delta_json_path)
    delta_checks = read_json(delta_checks_path)
    cuda_rows = read_csv(cuda_csv_path)
    cuda_checks = read_json(cuda_checks_path)

    if not isinstance(delta_document, dict):
        raise ValueError("delta JSON root must be an object")
    if delta_document.get("schema") != "simsat.delta-two-stream.stage0":
        raise ValueError(f"unexpected delta schema: {delta_document.get('schema')!r}")
    if delta_document.get("row_count") != len(delta_rows):
        raise ValueError("delta CSV/JSON row-count mismatch")
    json_rows = delta_document.get("rows")
    if not isinstance(json_rows, list) or len(json_rows) != len(delta_rows):
        raise ValueError("delta JSON rows are missing or inconsistent")

    if not isinstance(delta_checks, dict) or delta_checks.get("status") != "pass":
        raise ValueError("delta-two-stream self-check fixture is not passing")
    if not isinstance(cuda_checks, dict) or cuda_checks.get("all_passed") is not True:
        raise ValueError("CUDA oracle self-test fixture is not passing")

    index: dict[tuple[float, float, float, float, float], dict[str, str]] = {}
    for csv_row, json_row in zip(delta_rows, json_rows):
        if not isinstance(json_row, dict):
            raise ValueError("delta JSON row must be an object")
        key = slab_key(
            as_float(csv_row, "tau"),
            as_float(csv_row, "solar_zenith_deg"),
            as_float(csv_row, "asymmetry_g"),
            as_float(csv_row, "single_scatter_albedo"),
            as_float(csv_row, "surface_albedo"),
        )
        json_key = slab_key(
            as_float(json_row, "tau"),
            as_float(json_row, "solar_zenith_deg"),
            as_float(json_row, "asymmetry_g"),
            as_float(json_row, "single_scatter_albedo"),
            as_float(json_row, "surface_albedo"),
        )
        if key != json_key:
            raise ValueError(f"delta CSV/JSON key mismatch: {key!r} != {json_key!r}")
        csv_reflectance = as_float(csv_row, "toa_reflectance")
        json_reflectance = as_float(json_row, "toa_reflectance")
        if abs(csv_reflectance - json_reflectance) > 1.0e-15:
            raise ValueError(f"delta CSV/JSON value mismatch at {key!r}")
        if key in index:
            raise ValueError(f"duplicate delta slab key: {key!r}")
        index[key] = csv_row

    if not cuda_rows:
        raise ValueError("CUDA baseline CSV is empty")
    for row in cuda_rows:
        if as_float(row, "tau") < 0.0:
            raise ValueError("CUDA tau must be nonnegative")
        if as_float(row, "brf") < 0.0:
            raise ValueError("CUDA BRF must be nonnegative")
        if as_float(row, "standard_error_brf") < 0.0:
            raise ValueError("CUDA standard error must be nonnegative")

    hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in required}
    return delta_rows, cuda_rows, index, hashes


def hg_phase(cosine: float, g: float) -> float:
    cosine = max(-1.0, min(1.0, cosine))
    gg = g * g
    denominator = 1.0 + gg - 2.0 * g * cosine
    return (1.0 - gg) / (4.0 * PI * denominator**1.5)


def dual_hg_phase(cosine: float, g_scale: float, constants: tuple[float, float, float]) -> float:
    forward_g, backward_g, forward_weight = constants
    return forward_weight * hg_phase(cosine, forward_g * g_scale) + (
        1.0 - forward_weight
    ) * hg_phase(cosine, backward_g * g_scale)


def scattering_cosine(sun_deg: float, view_deg: float, relative_azimuth_deg: float) -> float:
    sun = math.radians(sun_deg)
    view = math.radians(view_deg)
    azimuth = math.radians(relative_azimuth_deg)
    return math.sin(sun) * math.sin(view) * math.cos(azimuth) - math.cos(sun) * math.cos(view)


def analytic_single_scatter_brf(
    tau: float,
    ssa: float,
    g: float,
    sun_deg: float,
    view_deg: float,
    relative_azimuth_deg: float,
) -> float:
    if tau <= 0.0 or ssa <= 0.0:
        return 0.0
    mu0 = math.cos(math.radians(sun_deg))
    muv = math.cos(math.radians(view_deg))
    cosine = scattering_cosine(sun_deg, view_deg, relative_azimuth_deg)
    attenuation = -math.expm1(-tau * (1.0 / mu0 + 1.0 / muv))
    return PI * ssa * hg_phase(cosine, g) * attenuation / (mu0 + muv)


def phase_function(model: str, base_g: float) -> Callable[[float, float], float]:
    if model == "hg-proxy":
        return lambda cosine, scale: hg_phase(cosine, base_g * scale)
    if model == "liquid-dual-hg":
        return lambda cosine, scale: dual_hg_phase(cosine, scale, LIQUID_PHASE)
    if model == "ice-dual-hg":
        return lambda cosine, scale: dual_hg_phase(cosine, scale, ICE_PHASE)
    raise ValueError(f"unknown phase model {model!r}")


def legacy_slab_brf(
    tau: float,
    base_g: float,
    sun_deg: float,
    view_deg: float,
    relative_azimuth_deg: float,
    octaves: int,
    phase_model: str,
    intervals: int = SLAB_INTERVALS,
) -> float:
    """Integrate the current local octave source through a black homogeneous slab.

    This is a directional diagnostic with the same BRF normalization as the CUDA
    oracle, not a claim that the legacy source solves slab radiative transfer.  It
    keeps the production support rule ``max(column_tau, tau_sun)`` and omits the
    renderer's local-pitch fallback because a homogeneous slab has no unique voxel
    pitch.  Visible SSA remains the production value of exactly one.
    """

    if tau <= 0.0:
        return 0.0
    if intervals <= 0 or intervals % 2:
        raise ValueError("Simpson intervals must be positive and even")
    mu0 = math.cos(math.radians(sun_deg))
    muv = math.cos(math.radians(view_deg))
    if mu0 <= 0.0 or muv <= 0.0:
        raise ValueError("sun and view must be in the upper hemisphere")
    cosine = scattering_cosine(sun_deg, view_deg, relative_azimuth_deg)
    phase = phase_function(phase_model, base_g)
    step = tau / intervals
    integral = 0.0
    for index in range(intervals + 1):
        vertical_depth = index * step
        sun_depth = vertical_depth / mu0
        support_tau = max(tau, sun_depth)
        thin_gate = -math.expm1(-support_tau)
        source = 0.0
        extinction_scale = 1.0
        phase_scale = 1.0
        brightness_weight = 1.0
        order_gate = 1.0
        for order in range(max(1, octaves)):
            if order > 0:
                order_gate *= thin_gate
            source += (
                order_gate
                * brightness_weight
                * phase(cosine, phase_scale)
                * math.exp(-sun_depth * extinction_scale)
            )
            extinction_scale *= LEGACY_EXTINCTION_SCALE
            phase_scale *= LEGACY_PHASE_SCALE
            brightness_weight *= LEGACY_BRIGHTNESS_SCALE
        integrand = source * math.exp(-vertical_depth / muv) / muv
        coefficient = 1 if index in (0, intervals) else 4 if index % 2 else 2
        integral += coefficient * integrand
    radiance_over_f0 = integral * step / 3.0
    return PI * radiance_over_f0 / mu0


def legacy_sphere_integrated_source(
    vertical_depth: float, total_vertical_tau: float, mu0: float, octaves: int = LEGACY_OCTAVES
) -> float:
    """Integral of the local octave source over all outgoing solid angle.

    Each HG lobe integrates to one, so phase cancels.  This exposes how much local
    source the octave heuristic injects relative to the direct-beam source.  It is
    not a slab reflectance or an energy-violation proof because the heuristic has no
    solved diffuse-energy state to compare against.
    """

    sun_depth = vertical_depth / mu0
    support_tau = max(total_vertical_tau, sun_depth)
    thin_gate = -math.expm1(-support_tau)
    total = 0.0
    order_gate = 1.0
    extinction_scale = 1.0
    weight = 1.0
    for order in range(max(1, octaves)):
        if order > 0:
            order_gate *= thin_gate
        total += order_gate * weight * math.exp(-sun_depth * extinction_scale)
        extinction_scale *= LEGACY_EXTINCTION_SCALE
        weight *= LEGACY_BRIGHTNESS_SCALE
    return total


def comparison_rows(
    cuda_rows: Sequence[Mapping[str, str]],
    delta_index: Mapping[tuple[float, float, float, float, float], Mapping[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    observable_contract = (
        "CUDA/legacy columns are directional BRF; delta columns are hemispheric flux. "
        "They are juxtaposed, never divided or fitted to one another."
    )
    for cuda in cuda_rows:
        tau = as_float(cuda, "tau")
        ssa = as_float(cuda, "ssa")
        g = as_float(cuda, "hg_g")
        sun = as_float(cuda, "sun_zenith_deg")
        view = as_float(cuda, "view_zenith_deg")
        azimuth = as_float(cuda, "relative_azimuth_deg")
        cosine = scattering_cosine(sun, view, azimuth)
        analytic = analytic_single_scatter_brf(tau, ssa, g, sun, view, azimuth)

        proxy_one = legacy_slab_brf(tau, g, sun, view, azimuth, 1, "hg-proxy")
        proxy_six = legacy_slab_brf(
            tau, g, sun, view, azimuth, LEGACY_OCTAVES, "hg-proxy"
        )
        liquid_one = legacy_slab_brf(tau, g, sun, view, azimuth, 1, "liquid-dual-hg")
        liquid_six = legacy_slab_brf(
            tau, g, sun, view, azimuth, LEGACY_OCTAVES, "liquid-dual-hg"
        )
        ice_one = legacy_slab_brf(tau, g, sun, view, azimuth, 1, "ice-dual-hg")
        ice_six = legacy_slab_brf(
            tau, g, sun, view, azimuth, LEGACY_OCTAVES, "ice-dual-hg"
        )
        cuda_brf = as_float(cuda, "brf")
        cuda_higher = cuda_brf - analytic
        legacy_higher = proxy_six - proxy_one

        mu0 = math.cos(math.radians(sun))
        source_top = legacy_sphere_integrated_source(0.0, tau, mu0)
        source_mid = legacy_sphere_integrated_source(0.5 * tau, tau, mu0)
        source_bottom = legacy_sphere_integrated_source(tau, tau, mu0)
        direct_top = 1.0

        key = slab_key(tau, sun, g, ssa, 0.0)
        delta = delta_index.get(key)
        if delta is None:
            match_status = "no_exact_delta_row"
            delta_values: dict[str, object] = {
                "delta_phase_case": "",
                "delta_hemispheric_toa_reflectance": "",
                "delta_hemispheric_first_reflectance": "",
                "delta_hemispheric_higher_reflectance": "",
                "delta_hemispheric_bottom_direct_transmittance": "",
                "delta_hemispheric_bottom_diffuse_transmittance": "",
                "delta_atmosphere_absorptance": "",
                "delta_energy_sum": "",
            }
        else:
            match_status = "exact_inputs_different_observables"
            delta_total = as_float(delta, "toa_reflectance")
            delta_first = as_float(delta, "black_surface_single_scatter_reflectance")
            delta_values = {
                "delta_phase_case": delta["phase_case"],
                "delta_hemispheric_toa_reflectance": delta_total,
                "delta_hemispheric_first_reflectance": delta_first,
                "delta_hemispheric_higher_reflectance": delta_total - delta_first,
                "delta_hemispheric_bottom_direct_transmittance": as_float(
                    delta, "direct_transmittance_bottom"
                ),
                "delta_hemispheric_bottom_diffuse_transmittance": as_float(
                    delta, "diffuse_down_bottom"
                ),
                "delta_atmosphere_absorptance": as_float(delta, "atmosphere_absorptance"),
                "delta_energy_sum": as_float(delta, "energy_sum"),
            }

        row: dict[str, object] = {
            "case": cuda["case"],
            "match_status": match_status,
            "tau": tau,
            "ssa": ssa,
            "hg_g": g,
            "sun_zenith_deg": sun,
            "view_zenith_deg": view,
            "relative_azimuth_deg": azimuth,
            "scattering_cosine": cosine,
            "cuda_directional_brf": cuda_brf,
            "cuda_standard_error_brf": as_float(cuda, "standard_error_brf"),
            "cuda_ci95_low_brf": as_float(cuda, "ci95_low_brf"),
            "cuda_ci95_high_brf": as_float(cuda, "ci95_high_brf"),
            "analytic_hg_single_brf": analytic,
            "cuda_directional_higher_brf": cuda_higher,
            "legacy_hg_proxy_brf_octave1": proxy_one,
            "legacy_hg_proxy_brf_octave6": proxy_six,
            "legacy_hg_proxy_higher_brf": legacy_higher,
            "legacy_higher_scale_to_cuda": (
                cuda_higher / legacy_higher if legacy_higher > 0.0 else ""
            ),
            "legacy_liquid_dual_hg_brf_octave1": liquid_one,
            "legacy_liquid_dual_hg_brf_octave6": liquid_six,
            "legacy_ice_dual_hg_brf_octave1": ice_one,
            "legacy_ice_dual_hg_brf_octave6": ice_six,
            "legacy_sphere_source_top": source_top,
            "legacy_sphere_source_mid": source_mid,
            "legacy_sphere_source_bottom": source_bottom,
            "legacy_sphere_source_top_over_direct": source_top / direct_top,
            "observable_contract": observable_contract,
        }
        row.update(delta_values)
        rows.append(row)
    return rows


def bounded_fit(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    common = [
        row
        for row in rows
        if row["match_status"] == "exact_inputs_different_observables"
        and as_float(row, "tau") > 0.0
        and abs(as_float(row, "hg_g") - 0.75) < 1.0e-12
        and abs(as_float(row, "ssa") - 0.999) < 1.0e-12
        and abs(as_float(row, "sun_zenith_deg") - 30.0) < 1.0e-12
        and abs(as_float(row, "view_zenith_deg")) < 1.0e-12
        and abs(as_float(row, "relative_azimuth_deg")) < 1.0e-12
    ]
    if not common:
        return {
            "status": "insufficient-no-common-nonzero-cases",
            "case_count": 0,
        }

    def solve(weights: Sequence[float]) -> float:
        numerator = 0.0
        denominator = 0.0
        for row, weight in zip(common, weights):
            legacy_higher = as_float(row, "legacy_hg_proxy_higher_brf")
            cuda_higher = as_float(row, "cuda_directional_higher_brf")
            numerator += weight * legacy_higher * cuda_higher
            denominator += weight * legacy_higher * legacy_higher
        if denominator <= 0.0:
            return 0.0
        return max(0.0, min(1.0, numerator / denominator))

    alpha_unweighted = solve([1.0] * len(common))
    inverse_variance = []
    for row in common:
        standard_error = as_float(row, "cuda_standard_error_brf")
        inverse_variance.append(1.0 / max(standard_error * standard_error, 1.0e-18))
    alpha_weighted = solve(inverse_variance)

    residuals = []
    case_scales = []
    for row in common:
        legacy_higher = as_float(row, "legacy_hg_proxy_higher_brf")
        cuda_higher = as_float(row, "cuda_directional_higher_brf")
        prediction = as_float(row, "analytic_hg_single_brf") + alpha_unweighted * legacy_higher
        residuals.append(prediction - as_float(row, "cuda_directional_brf"))
        case_scales.append(
            {
                "case": row["case"],
                "tau": as_float(row, "tau"),
                "cuda_higher_over_legacy_higher": cuda_higher / legacy_higher,
            }
        )

    return {
        "status": "diagnostic-only-insufficient-for-production",
        "reason": (
            "Only three nonzero exact cases share one HG value, SSA, solar angle, "
            "nadir view, relative azimuth, and black surface. The coefficient may "
            "describe this slice but cannot identify an angular or phase closure."
        ),
        "case_count": len(common),
        "case_names": [row["case"] for row in common],
        "alpha_bounded_unweighted": alpha_unweighted,
        "alpha_bounded_inverse_mc_variance": alpha_weighted,
        "model": "BRF_fit = analytic_single_HG_BRF + alpha * legacy_HG_proxy_higher_BRF",
        "rmse_brf_unweighted": math.sqrt(sum(value * value for value in residuals) / len(residuals)),
        "max_abs_residual_brf_unweighted": max(abs(value) for value in residuals),
        "per_case_higher_order_scales": case_scales,
        "not_fitted": "delta hemispheric flux; it is a different observable",
    }


def max_scatters_for_tau(tau: float) -> int:
    if tau <= 0.0:
        return 32
    if tau <= 0.1:
        return 64
    if tau <= 0.3:
        return 96
    if tau <= 1.0:
        return 160
    if tau <= 3.0:
        return 256
    if tau <= 10.0:
        return 384
    return 512


def request_grid(
    delta_index: Mapping[tuple[float, float, float, float, float], Mapping[str, str]]
) -> list[dict[str, object]]:
    requests: list[dict[str, object]] = []
    geometries = [
        (0.0, 0.0),
        (40.0, 0.0),
        (40.0, 90.0),
        (40.0, 180.0),
        (65.0, 0.0),
        (65.0, 90.0),
        (65.0, 180.0),
    ]
    taus = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
    asymmetries = (0.665, 0.75, 0.85)
    solar_angles = (30.0, 50.0, 65.0)
    seed = 0x53415453494D0015

    raw: list[tuple[str, float, float, float, float, float, float]] = []
    for tau in taus:
        for g in asymmetries:
            for sun in solar_angles:
                for view, azimuth in geometries:
                    split = (
                        "angular-holdout"
                        if sun == 50.0 or (view == 40.0 and azimuth == 90.0)
                        else "calibration"
                    )
                    raw.append((split, tau, 0.999, g, sun, view, azimuth))

    absorption_geometries = ((0.0, 0.0), (40.0, 90.0), (65.0, 180.0))
    for tau in (0.3, 3.0, 30.0):
        for g in (0.75, 0.85):
            for sun in (30.0, 65.0):
                for view, azimuth in absorption_geometries:
                    raw.append(("absorption-holdout", tau, 0.95, g, sun, view, azimuth))

    for ordinal, (split, tau, ssa, g, sun, view, azimuth) in enumerate(raw, start=1):
        key = slab_key(tau, sun, g, ssa, 0.0)
        requests.append(
            {
                "case": f"ccf_v1_{ordinal:04d}",
                "split": split,
                "tau": tau,
                "ssa": ssa,
                "hg_g": g,
                "sun_zenith_deg": sun,
                "view_zenith_deg": view,
                "relative_azimuth_deg": azimuth,
                "surface_albedo_for_delta_join": 0.0,
                "delta_exact_row_available": key in delta_index,
                "samples": 262144 if tau == 0.0 else 4194304,
                "seed": seed,
                "max_scatters": max_scatters_for_tau(tau),
                "batch_samples": 65536,
                "target_ci95_halfwidth_brf": 0.005,
            }
        )
    return requests


def phase_integral(g: float, intervals: int = 65536) -> float:
    step = 2.0 / intervals
    total = 0.0
    for index in range(intervals + 1):
        cosine = -1.0 + index * step
        value = 2.0 * PI * hg_phase(cosine, g)
        coefficient = 1 if index in (0, intervals) else 4 if index % 2 else 2
        total += coefficient * value
    return step * total / 3.0


def monotonic_delta_check(delta_rows: Sequence[Mapping[str, str]]) -> tuple[int, float]:
    groups: dict[tuple[float, float], list[tuple[float, float]]] = defaultdict(list)
    for row in delta_rows:
        if (
            abs(as_float(row, "single_scatter_albedo") - 1.0) < 1.0e-12
            and abs(as_float(row, "surface_albedo")) < 1.0e-12
        ):
            group = (
                canonical_float(as_float(row, "asymmetry_g")),
                canonical_float(as_float(row, "solar_zenith_deg")),
            )
            groups[group].append((as_float(row, "tau"), as_float(row, "toa_reflectance")))
    worst_drop = 0.0
    for values in groups.values():
        values.sort()
        for previous, current in zip(values, values[1:]):
            worst_drop = min(worst_drop, current[1] - previous[1])
    return len(groups), worst_drop


def run_self_checks(
    delta_rows: Sequence[Mapping[str, str]],
    cuda_rows: Sequence[Mapping[str, str]],
    comparisons: Sequence[Mapping[str, object]],
    requests: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    checks: dict[str, object] = {}

    formula_weight_sum = (1.0 - LEGACY_BRIGHTNESS_SCALE**LEGACY_OCTAVES) / (
        1.0 - LEGACY_BRIGHTNESS_SCALE
    )
    weight_error = abs(LEGACY_WEIGHT_SUM - formula_weight_sum)
    if weight_error > 1.0e-15:
        raise AssertionError(f"legacy weight sum mismatch: {weight_error}")
    checks["legacy_weight_sum"] = {
        "value": LEGACY_WEIGHT_SUM,
        "formula_error": weight_error,
    }

    phase_errors = {str(g): abs(phase_integral(g) - 1.0) for g in (0.85, 0.75, -0.15, -0.10)}
    if max(phase_errors.values()) > 2.0e-9:
        raise AssertionError(f"phase normalization error: {phase_errors!r}")
    checks["phase_normalization"] = {
        "max_abs_error": max(phase_errors.values()),
        "per_g_abs_error": phase_errors,
    }

    analytic_errors = []
    for cuda in cuda_rows:
        tau = as_float(cuda, "tau")
        if tau <= 0.0:
            continue
        g = as_float(cuda, "hg_g")
        sun = as_float(cuda, "sun_zenith_deg")
        view = as_float(cuda, "view_zenith_deg")
        azimuth = as_float(cuda, "relative_azimuth_deg")
        numeric = legacy_slab_brf(tau, g, sun, view, azimuth, 1, "hg-proxy")
        analytic_ssa_one = analytic_single_scatter_brf(tau, 1.0, g, sun, view, azimuth)
        analytic_errors.append(abs(numeric - analytic_ssa_one))
    if max(analytic_errors, default=0.0) > 1.0e-9:
        raise AssertionError(f"legacy single-scatter integration error: {max(analytic_errors)}")
    checks["single_scatter_analytic_agreement"] = {
        "case_count": len(analytic_errors),
        "max_abs_brf_error": max(analytic_errors, default=0.0),
    }

    zero_values = [
        legacy_slab_brf(0.0, 0.75, 30.0, 0.0, 0.0, LEGACY_OCTAVES, "hg-proxy"),
        analytic_single_scatter_brf(0.0, 0.999, 0.75, 30.0, 0.0, 0.0),
    ]
    zero_values.extend(
        as_float(row, "cuda_directional_brf")
        for row in comparisons
        if as_float(row, "tau") == 0.0
    )
    if any(value != 0.0 for value in zero_values):
        raise AssertionError(f"tau-zero response is not exact zero: {zero_values!r}")
    checks["tau_zero"] = {"case_count": len(zero_values), "max_abs_brf": 0.0}

    thin_a = legacy_slab_brf(1.0e-4, 0.75, 30.0, 0.0, 0.0, 6, "hg-proxy") - legacy_slab_brf(
        1.0e-4, 0.75, 30.0, 0.0, 0.0, 1, "hg-proxy"
    )
    thin_b = legacy_slab_brf(2.0e-4, 0.75, 30.0, 0.0, 0.0, 6, "hg-proxy") - legacy_slab_brf(
        2.0e-4, 0.75, 30.0, 0.0, 0.0, 1, "hg-proxy"
    )
    thin_slope = math.log(thin_b / thin_a) / math.log(2.0)
    if not (1.95 <= thin_slope <= 2.05):
        raise AssertionError(f"legacy thin higher-order slope is not O(tau^2): {thin_slope}")
    checks["legacy_thin_higher_order"] = {
        "tau_pair": [1.0e-4, 2.0e-4],
        "log_log_slope": thin_slope,
    }

    max_energy_error = max(abs(as_float(row, "energy_sum") - 1.0) for row in delta_rows)
    min_absorption = min(as_float(row, "atmosphere_absorptance") for row in delta_rows)
    if max_energy_error > 1.0e-10 or min_absorption < -1.0e-10:
        raise AssertionError(
            f"delta energy check failed: closure={max_energy_error}, min A={min_absorption}"
        )
    checks["delta_energy_conservation"] = {
        "row_count": len(delta_rows),
        "max_abs_energy_sum_error": max_energy_error,
        "minimum_atmosphere_absorptance": min_absorption,
    }

    monotonic_group_count, worst_drop = monotonic_delta_check(delta_rows)
    if worst_drop < -1.0e-10:
        raise AssertionError(f"conservative black-surface delta reflectance dropped: {worst_drop}")
    checks["delta_conservative_monotonicity"] = {
        "group_count": monotonic_group_count,
        "worst_successive_toa_reflectance_change": worst_drop,
    }

    exact = [row for row in comparisons if row["match_status"] == "exact_inputs_different_observables"]
    nonzero_common = sorted(
        (as_float(row, "tau"), as_float(row, "cuda_directional_brf"))
        for row in exact
        if as_float(row, "tau") > 0.0
    )
    if len(exact) != 4 or len(nonzero_common) != 3:
        raise AssertionError(
            f"expected 4 exact / 3 nonzero common rows, got {len(exact)} / {len(nonzero_common)}"
        )
    cuda_worst_drop = min(
        (current[1] - previous[1] for previous, current in zip(nonzero_common, nonzero_common[1:])),
        default=0.0,
    )
    if cuda_worst_drop < 0.0:
        raise AssertionError(f"common CUDA slice is not monotone: {cuda_worst_drop}")
    checks["exact_common_mapping"] = {
        "exact_case_count": len(exact),
        "nonzero_case_count": len(nonzero_common),
        "common_cuda_worst_successive_brf_change": cuda_worst_drop,
    }

    if not requests or not all(row["delta_exact_row_available"] is True for row in requests):
        raise AssertionError("request grid contains a case without an exact delta row")
    unique_request_keys = {
        (
            row["tau"],
            row["ssa"],
            row["hg_g"],
            row["sun_zenith_deg"],
            row["view_zenith_deg"],
            row["relative_azimuth_deg"],
        )
        for row in requests
    }
    if len(unique_request_keys) != len(requests):
        raise AssertionError("request grid contains duplicate directional cases")
    checks["request_grid"] = {
        "row_count": len(requests),
        "unique_row_count": len(unique_request_keys),
        "all_have_exact_delta_flux_row": True,
    }

    return {
        "schema": "simsat.cloud-closure-fit.self-checks.v1",
        "schema_version": 1,
        "status": "pass",
        "script_sha256": script_sha256(),
        "checks": checks,
    }


def csv_cell(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"cannot serialize non-finite float {value!r}")
        return format(value, ".17g")
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def csv_text(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fields), lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: csv_cell(row.get(field, "")) for field in fields})
    return output.getvalue()


def json_text(document: object) -> str:
    return json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"


def split_counts(requests: Iterable[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in requests:
        counts[str(row["split"])] += 1
    return dict(sorted(counts.items()))


def build_artifacts(args: argparse.Namespace) -> tuple[dict[Path, bytes], dict[str, object]]:
    delta_rows, cuda_rows, delta_index, hashes = validate_inputs(
        args.delta_csv,
        args.delta_json,
        args.delta_checks,
        args.cuda_csv,
        args.cuda_checks,
    )
    comparisons = comparison_rows(cuda_rows, delta_index)
    requests = request_grid(delta_index)
    fit = bounded_fit(comparisons)
    checks = run_self_checks(delta_rows, cuda_rows, comparisons, requests)

    exact = [row for row in comparisons if row["match_status"] == "exact_inputs_different_observables"]
    summary = {
        "schema": "simsat.cloud-closure-fit.summary.v1",
        "schema_version": 1,
        "script_sha256": script_sha256(),
        "input_sha256": hashes,
        "observable_contract": {
            "delta_two_stream": (
                "TOA and boundary hemispheric flux fractions from a two-stream closure."
            ),
            "cuda_oracle": "Directional BRF = pi*I/(F0*mu0) for a single-HG black slab.",
            "legacy_hg_proxy": (
                "Directional slab BRF analogue obtained by integrating the current octave "
                "source with a single-HG phase matched to the CUDA input. It isolates octave "
                "transport math but is not the shipping dual-HG phase."
            ),
            "legacy_dual_hg": (
                "Directional slab BRF analogues using the exact shipping liquid and ice phases."
            ),
            "forbidden_inference": (
                "A delta hemispheric reflectance / CUDA directional BRF ratio is not an error "
                "metric. No such ratio is generated."
            ),
        },
        "exact_common_mapping": {
            "case_count": len(exact),
            "nonzero_case_count": sum(as_float(row, "tau") > 0.0 for row in exact),
            "case_names": [row["case"] for row in exact],
            "interpolation_performed": False,
        },
        "diagnostic_directional_fit": fit,
        "legacy_source_energy": {
            "weight_sum_if_all_six_terms_are_unattenuated": LEGACY_WEIGHT_SUM,
            "interpretation": (
                "The sphere-integrated local source columns expose unconstrained source "
                "amplification. Values above one are not alone an energy violation, because "
                "a physical diffuse field can exceed the local direct beam; the defect is that "
                "the legacy value is not derived from a conserved diffuse-energy state."
            ),
            "max_top_source_over_direct_in_cuda_baseline": max(
                as_float(row, "legacy_sphere_source_top_over_direct") for row in comparisons
            ),
        },
        "stage1_recommendation": {
            "status": "experiment-only-default-off",
            "mode": "delta-flux-v1",
            "legacy_identity": (
                "cloud_multiscatter=legacy-octaves must call the unchanged v0.1.4 arithmetic; "
                "legacy-v014 remains the preset and default while the experiment is evaluated."
            ),
            "inputs": [
                "total vertical tau",
                "fractional depth",
                "mu0",
                "single-scatter albedo",
                "effective asymmetry",
                "surface albedo",
            ],
            "outputs": [
                "upward diffuse flux",
                "downward diffuse flux",
                "absorbed fraction",
                "validity/bounds flags",
            ],
            "directional_reconstruction": (
                "Keep exact direct single scatter. Initially reconstruct only the higher-order "
                "component with a nonnegative hemispherically normalized kernel; never multiply "
                "a hemispheric flux by an unconstrained directional gain. A depth-resolved "
                "DISORT/Monte-Carlo fixture is required before wiring this into the marcher."
            ),
            "controls": {
                "cloud_multiscatter": ["legacy-octaves", "single-scatter", "delta-flux-v1"],
                "cloud_diffuse_blend": "clamped [0,1], experimental only, default 0",
                "cloud_closure_lut_version": "immutable schema/hash recorded in render metadata",
            },
        },
        "acceptance_criteria": {
            "zero": "tau=0 and SSA=0 produce exactly zero cloud BRF/source contribution.",
            "thin": "higher-order contribution has log-log slope 2 +/- 0.05 as tau -> 0.",
            "conservation": (
                "For every slab, R+T+A=1 within 1e-4; R,T,A are nonnegative. "
                "Conservative black-surface R+T=1 within 1e-4."
            ),
            "monotonicity": (
                "For fixed conservative black-surface inputs, hemispheric R is nondecreasing "
                "with tau. Directional BRF monotonicity is checked only on declared geometry "
                "slices, not imposed universally on anisotropic phase functions."
            ),
            "directional_reference": (
                "Reserved CUDA/DISORT cases must meet max(0.005 absolute, 5% relative) BRF "
                "error outside reported Monte Carlo uncertainty; no production fit is accepted "
                "from the current three-case slice."
            ),
            "identity": "legacy-v014 CPU output hash remains bit-identical.",
        },
        "request_grid": {
            "row_count": len(requests),
            "split_counts": split_counts(requests),
            "all_rows_have_exact_delta_flux_join": all(
                row["delta_exact_row_available"] is True for row in requests
            ),
            "execution": "run-request-grid.ps1; results are deliberately not fabricated here",
        },
        "self_checks": checks,
    }

    comparison_bytes = csv_text(comparisons, COMPARISON_FIELDS).encode("utf-8")
    request_bytes = csv_text(requests, REQUEST_FIELDS).encode("utf-8")
    summary_bytes = json_text(summary).encode("utf-8")
    checks_bytes = json_text(checks).encode("utf-8")

    # Repeat the pure serializations before touching disk.
    if comparison_bytes != csv_text(comparisons, COMPARISON_FIELDS).encode("utf-8"):
        raise AssertionError("comparison CSV serialization is not repeatable")
    if request_bytes != csv_text(requests, REQUEST_FIELDS).encode("utf-8"):
        raise AssertionError("request-grid CSV serialization is not repeatable")
    if summary_bytes != json_text(summary).encode("utf-8"):
        raise AssertionError("summary JSON serialization is not repeatable")

    artifacts = {
        args.output_dir / "stage0-comparison-v1.csv": comparison_bytes,
        args.output_dir / "stage0-summary-v1.json": summary_bytes,
        args.output_dir / "stage0-self-checks-v1.json": checks_bytes,
        args.output_dir / "stage1-request-grid-v1.csv": request_bytes,
    }
    return artifacts, summary


def write_artifacts(artifacts: Mapping[Path, bytes]) -> None:
    for path, payload in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def verify_artifacts(artifacts: Mapping[Path, bytes]) -> None:
    for path, expected in artifacts.items():
        if not path.is_file():
            raise AssertionError(f"missing generated fixture: {path}")
        actual = path.read_bytes()
        if actual != expected:
            raise AssertionError(f"generated fixture is stale: {path}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("check", "generate", "all"))
    result.add_argument("--delta-csv", type=Path, default=DEFAULT_DELTA_CSV)
    result.add_argument("--delta-json", type=Path, default=DEFAULT_DELTA_JSON)
    result.add_argument("--delta-checks", type=Path, default=DEFAULT_DELTA_CHECKS)
    result.add_argument("--cuda-csv", type=Path, default=DEFAULT_CUDA_CSV)
    result.add_argument("--cuda-checks", type=Path, default=DEFAULT_CUDA_CHECKS)
    result.add_argument("--output-dir", type=Path, default=FIXTURE_DIR)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    artifacts, summary = build_artifacts(args)
    if args.command in ("generate", "all"):
        write_artifacts(artifacts)
    if args.command in ("check", "all"):
        verify_artifacts(artifacts)
    fit = summary["diagnostic_directional_fit"]
    print(
        "cloud-closure-fit: PASS; "
        f"exact={summary['exact_common_mapping']['case_count']} "
        f"nonzero={summary['exact_common_mapping']['nonzero_case_count']} "
        f"fit={fit['status']} "
        f"request_rows={summary['request_grid']['row_count']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, FileNotFoundError, ValueError) as error:
        print(f"cloud-closure-fit: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
