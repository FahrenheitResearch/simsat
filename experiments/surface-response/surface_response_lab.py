#!/usr/bin/env python3
"""Deterministic Stage-0 lab for bounded land display-response candidates.

This program is deliberately isolated from the SimSat production renderer.  It
compares two scalar, color-ratio-preserving display operators against identity:

* a bounded solar-elevation normalization; and
* a bounded dark-land toe.

The synthetic input grid keeps linear surface albedo separate from RGB after a
simple Lambert-plus-diffuse terrain illumination proxy.  The toe is classified
from albedo and its gain is applied to the illuminated signal, matching SimSat's
production experiment.  This is only a response-function test bed, not a
radiative-transfer or terrain model.  Python's standard library is sufficient.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


SCHEMA = "simsat.surface-response.stage0"
SCHEMA_VERSION = 1
METHOD_ID = "bounded-land-sza-and-albedo-toe-v1"

# Linear-RGB relative luminance coefficients used by the production candidate.
# The lab mirrors placement but does not select a product default.
LUMA = (0.2126, 0.7152, 0.0722)


@dataclass(frozen=True)
class SzaConfig:
    twilight_identity_end_deg: float = 20.0
    daylight_ramp_end_deg: float = 30.0
    reference_elevation_deg: float = 60.0
    elevation_floor_deg: float = 20.0
    max_gain: float = 1.6


@dataclass(frozen=True)
class ToeConfig:
    knee_luminance: float = 0.08
    gamma: float = 0.65
    max_gain: float = 1.5
    twilight_identity_end_deg: float = 20.0
    daylight_ramp_end_deg: float = 30.0


SZA = SzaConfig()
TOE = ToeConfig()

# The slope values are signed tilts in the solar vertical plane.  Positive is
# sun-facing.  The diffuse floor is a synthetic response-grid device, not a
# claim about SimSat's sky irradiance.
DIFFUSE_FLOOR = 0.18
PALETTES = (
    ("forest", (0.62, 1.00, 0.38)),
    ("neutral", (1.00, 1.00, 1.00)),
    ("dry-soil", (1.35, 0.85, 0.45)),
    ("cool-rock", (0.78, 0.95, 1.18)),
)
BASE_LUMINANCE_GRID = (
    0.0,
    1.0e-6,
    1.0e-3,
    5.0e-3,
    1.0e-2,
    2.0e-2,
    4.0e-2,
    8.0e-2,
    1.5e-1,
    5.0e-1,
)
SUN_ELEVATION_GRID_DEG = (
    -5.0,
    0.0,
    10.0,
    20.0,
    25.0,
    30.0,
    33.0,
    40.0,
    50.0,
    59.999,
    60.0,
    75.0,
)
SLOPE_TOWARD_SUN_GRID_DEG = (-20.0, 0.0, 20.0)


@dataclass(frozen=True)
class CaseResult:
    case_id: int
    palette: str
    base_luminance: float
    sun_elevation_deg: float
    slope_toward_sun_deg: float
    local_mu: float
    illumination_factor: float
    albedo_rgb: tuple[float, float, float]
    albedo_y: float
    input_rgb: tuple[float, float, float]
    input_y: float
    identity_rgb: tuple[float, float, float]
    identity_y: float
    sza_gain: float
    sza_rgb: tuple[float, float, float]
    sza_y: float
    toe_gain: float
    toe_rgb: tuple[float, float, float]
    toe_y: float
    combined_sza_gain: float
    combined_toe_gain: float
    combined_effective_gain: float
    combined_rgb: tuple[float, float, float]
    combined_y: float
    max_chroma_error: float


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    details: str


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge0 == edge1:
        return 1.0 if value >= edge1 else 0.0
    t = clamp((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def luminance(rgb: Sequence[float]) -> float:
    return sum(channel * weight for channel, weight in zip(rgb, LUMA))


def chromaticity(rgb: Sequence[float]) -> tuple[float, float, float]:
    total = sum(rgb)
    if total <= 0.0:
        return (0.0, 0.0, 0.0)
    return tuple(channel / total for channel in rgb)  # type: ignore[return-value]


def chroma_error(
    source: Sequence[float], result: Sequence[float]
) -> float:
    if sum(source) <= 0.0 and sum(result) <= 0.0:
        return 0.0
    return max(
        abs(a - b) for a, b in zip(chromaticity(source), chromaticity(result))
    )


def scale_rgb_bounded(
    rgb: tuple[float, float, float], requested_gain: float, gain_cap: float
) -> tuple[tuple[float, float, float], float]:
    """Apply one bounded, non-darkening scalar without changing color ratios."""

    if requested_gain <= 1.0:
        return rgb, 1.0
    applied = min(requested_gain, gain_cap)
    if applied <= 1.0:
        return rgb, 1.0
    return tuple(channel * applied for channel in rgb), applied  # type: ignore[return-value]


def apply_sza_normalization(
    rgb: tuple[float, float, float],
    sun_elevation_deg: float,
    *,
    enabled: bool = True,
    config: SzaConfig = SZA,
) -> tuple[tuple[float, float, float], float]:
    """Apply the bounded, daylight-only solar-elevation candidate."""

    if (
        not enabled
        or sun_elevation_deg <= config.twilight_identity_end_deg
        or sun_elevation_deg >= config.reference_elevation_deg
    ):
        return rgb, 1.0

    elevation = clamp(sun_elevation_deg, 0.0, 90.0)
    mu = math.sin(math.radians(elevation))
    mu_floor = math.sin(math.radians(config.elevation_floor_deg))
    mu_reference = math.sin(math.radians(config.reference_elevation_deg))
    normalized = clamp(mu_reference / max(mu, mu_floor), 1.0, config.max_gain)
    daylight = smoothstep(
        config.twilight_identity_end_deg,
        config.daylight_ramp_end_deg,
        sun_elevation_deg,
    )
    requested = 1.0 + daylight * (normalized - 1.0)
    return scale_rgb_bounded(rgb, requested, config.max_gain)


def dark_land_toe_gain(
    albedo_rgb: tuple[float, float, float],
    sun_elevation_deg: float,
    *,
    enabled: bool = True,
    config: ToeConfig = TOE,
) -> float:
    """Evaluate the bounded toe from unilluminated linear surface albedo."""

    y = luminance(albedo_rgb)
    if (
        not enabled
        or y <= 0.0
        or y >= config.knee_luminance
        or sun_elevation_deg <= config.twilight_identity_end_deg
    ):
        return 1.0

    ratio = y / config.knee_luminance
    curved = config.knee_luminance * ratio**config.gamma
    blend = smoothstep(0.0, config.knee_luminance, y)
    target_y = curved * (1.0 - blend) + y * blend
    full_gain = max(1.0, target_y / y)
    daylight = smoothstep(
        config.twilight_identity_end_deg,
        config.daylight_ramp_end_deg,
        sun_elevation_deg,
    )
    requested = 1.0 + daylight * (full_gain - 1.0)
    return min(requested, config.max_gain)


def apply_dark_land_toe(
    illuminated_rgb: tuple[float, float, float],
    albedo_rgb: tuple[float, float, float],
    sun_elevation_deg: float,
    *,
    enabled: bool = True,
    config: ToeConfig = TOE,
) -> tuple[tuple[float, float, float], float]:
    """Apply an albedo-derived toe gain to the illuminated surface signal."""

    gain = dark_land_toe_gain(
        albedo_rgb,
        sun_elevation_deg,
        enabled=enabled,
        config=config,
    )
    return scale_rgb_bounded(illuminated_rgb, gain, config.max_gain)


def apply_combined(
    illuminated_rgb: tuple[float, float, float],
    albedo_rgb: tuple[float, float, float],
    sun_elevation_deg: float,
) -> tuple[tuple[float, float, float], float, float]:
    _, sza_gain = apply_sza_normalization(illuminated_rgb, sun_elevation_deg)
    toe_gain = dark_land_toe_gain(albedo_rgb, sun_elevation_deg)
    combined, _ = scale_rgb_bounded(
        illuminated_rgb,
        sza_gain * toe_gain,
        SZA.max_gain * TOE.max_gain,
    )
    return combined, sza_gain, toe_gain


def palette_rgb(
    target_luminance: float, ratios: tuple[float, float, float]
) -> tuple[float, float, float]:
    if target_luminance == 0.0:
        return (0.0, 0.0, 0.0)
    scale = target_luminance / luminance(ratios)
    rgb = tuple(channel * scale for channel in ratios)
    if max(rgb) > 1.0:
        raise ValueError("palette grid exceeds unit RGB")
    return rgb  # type: ignore[return-value]


def local_solar_cosine(
    sun_elevation_deg: float, slope_toward_sun_deg: float
) -> float:
    if sun_elevation_deg <= 0.0:
        return 0.0
    elevation = math.radians(sun_elevation_deg)
    slope = math.radians(slope_toward_sun_deg)
    return clamp(
        math.sin(elevation) * math.cos(slope)
        + math.cos(elevation) * math.sin(slope),
        0.0,
        1.0,
    )


def build_cases() -> list[CaseResult]:
    cases: list[CaseResult] = []
    case_id = 0
    for palette_name, ratios in PALETTES:
        for base_y in BASE_LUMINANCE_GRID:
            base_rgb = palette_rgb(base_y, ratios)
            for sun_elevation in SUN_ELEVATION_GRID_DEG:
                for slope in SLOPE_TOWARD_SUN_GRID_DEG:
                    local_mu = local_solar_cosine(sun_elevation, slope)
                    illumination = DIFFUSE_FLOOR + (1.0 - DIFFUSE_FLOOR) * local_mu
                    source = tuple(channel * illumination for channel in base_rgb)
                    source_y = luminance(source)

                    identity_sza, _ = apply_sza_normalization(
                        source, sun_elevation, enabled=False
                    )
                    identity, _ = apply_dark_land_toe(
                        identity_sza,
                        base_rgb,
                        sun_elevation,
                        enabled=False,
                    )
                    sza_rgb, sza_gain = apply_sza_normalization(source, sun_elevation)
                    toe_rgb, toe_gain = apply_dark_land_toe(
                        source,
                        base_rgb,
                        sun_elevation,
                    )
                    combined_rgb, combined_sza_gain, combined_toe_gain = apply_combined(
                        source,
                        base_rgb,
                        sun_elevation,
                    )
                    sza_y = luminance(sza_rgb)
                    toe_y = luminance(toe_rgb)
                    combined_y = luminance(combined_rgb)
                    maximum_chroma_error = max(
                        chroma_error(source, sza_rgb),
                        chroma_error(source, toe_rgb),
                        chroma_error(source, combined_rgb),
                    )
                    cases.append(
                        CaseResult(
                            case_id=case_id,
                            palette=palette_name,
                            base_luminance=base_y,
                            sun_elevation_deg=sun_elevation,
                            slope_toward_sun_deg=slope,
                            local_mu=local_mu,
                            illumination_factor=illumination,
                            albedo_rgb=base_rgb,
                            albedo_y=luminance(base_rgb),
                            input_rgb=source,
                            input_y=source_y,
                            identity_rgb=identity,
                            identity_y=luminance(identity),
                            sza_gain=sza_gain,
                            sza_rgb=sza_rgb,
                            sza_y=sza_y,
                            toe_gain=toe_gain,
                            toe_rgb=toe_rgb,
                            toe_y=toe_y,
                            combined_sza_gain=combined_sza_gain,
                            combined_toe_gain=combined_toe_gain,
                            combined_effective_gain=(
                                combined_sza_gain * combined_toe_gain
                            ),
                            combined_rgb=combined_rgb,
                            combined_y=combined_y,
                            max_chroma_error=maximum_chroma_error,
                        )
                    )
                    case_id += 1
    return cases


CSV_FIELDS = (
    "case_id",
    "palette",
    "base_luminance",
    "sun_elevation_deg",
    "slope_toward_sun_deg",
    "local_mu",
    "illumination_factor",
    "albedo_r",
    "albedo_g",
    "albedo_b",
    "albedo_y",
    "input_r",
    "input_g",
    "input_b",
    "input_y",
    "identity_r",
    "identity_g",
    "identity_b",
    "identity_y",
    "sza_applied_gain",
    "sza_r",
    "sza_g",
    "sza_b",
    "sza_y",
    "toe_applied_gain",
    "toe_r",
    "toe_g",
    "toe_b",
    "toe_y",
    "combined_sza_gain",
    "combined_toe_gain",
    "combined_effective_gain",
    "combined_r",
    "combined_g",
    "combined_b",
    "combined_y",
    "max_chroma_error",
)


def number_text(value: float) -> str:
    if value == 0.0:
        return "0"
    return format(value, ".12g")


def case_csv_row(case: CaseResult) -> dict[str, str | int]:
    values: dict[str, str | int] = {
        "case_id": case.case_id,
        "palette": case.palette,
        "base_luminance": number_text(case.base_luminance),
        "sun_elevation_deg": number_text(case.sun_elevation_deg),
        "slope_toward_sun_deg": number_text(case.slope_toward_sun_deg),
        "local_mu": number_text(case.local_mu),
        "illumination_factor": number_text(case.illumination_factor),
        "albedo_y": number_text(case.albedo_y),
        "input_y": number_text(case.input_y),
        "identity_y": number_text(case.identity_y),
        "sza_applied_gain": number_text(case.sza_gain),
        "sza_y": number_text(case.sza_y),
        "toe_applied_gain": number_text(case.toe_gain),
        "toe_y": number_text(case.toe_y),
        "combined_sza_gain": number_text(case.combined_sza_gain),
        "combined_toe_gain": number_text(case.combined_toe_gain),
        "combined_effective_gain": number_text(case.combined_effective_gain),
        "combined_y": number_text(case.combined_y),
        "max_chroma_error": number_text(case.max_chroma_error),
    }
    for prefix, rgb in (
        ("albedo", case.albedo_rgb),
        ("input", case.input_rgb),
        ("identity", case.identity_rgb),
        ("sza", case.sza_rgb),
        ("toe", case.toe_rgb),
        ("combined", case.combined_rgb),
    ):
        values[f"{prefix}_r"] = number_text(rgb[0])
        values[f"{prefix}_g"] = number_text(rgb[1])
        values[f"{prefix}_b"] = number_text(rgb[2])
    return values


def csv_bytes(cases: Iterable[CaseResult]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for case in cases:
        writer.writerow(case_csv_row(case))
    return stream.getvalue().encode("utf-8")


def run_checks(cases: Sequence[CaseResult]) -> list[CheckResult]:
    checks: list[CheckResult] = []

    def record(name: str, passed: bool, details: str) -> None:
        checks.append(CheckResult(name, passed, details))

    disabled_ok = all(case.identity_rgb == case.input_rgb for case in cases)
    record(
        "disabled-exact-identity",
        disabled_ok,
        f"{sum(case.identity_rgb == case.input_rgb for case in cases)}/{len(cases)} rows exact",
    )

    black_cases = [case for case in cases if case.input_rgb == (0.0, 0.0, 0.0)]
    black_ok = all(
        case.sza_rgb == case.input_rgb
        and case.toe_rgb == case.input_rgb
        and case.combined_rgb == case.input_rgb
        for case in black_cases
    )
    record(
        "black-exact-identity",
        black_ok,
        f"{len(black_cases)} black rows tested",
    )

    bright_cases = [case for case in cases if case.albedo_y >= TOE.knee_luminance]
    bright_ok = all(case.toe_rgb == case.input_rgb for case in bright_cases)
    knee_albedo = (TOE.knee_luminance,) * 3
    knee_illuminated = (0.02,) * 3
    knee_output, knee_gain = apply_dark_land_toe(
        knee_illuminated,
        knee_albedo,
        33.0,
    )
    bright_ok = (
        bright_ok
        and knee_output == knee_illuminated
        and knee_gain == 1.0
    )
    record(
        "toe-at-or-above-knee-exact-identity",
        bright_ok,
        f"{len(bright_cases)} grid rows plus exact knee probe",
    )

    twilight_cases = [
        case
        for case in cases
        if case.sun_elevation_deg <= SZA.twilight_identity_end_deg
    ]
    twilight_ok = all(
        case.sza_rgb == case.input_rgb
        and case.toe_rgb == case.input_rgb
        and case.combined_rgb == case.input_rgb
        for case in twilight_cases
    )
    record(
        "twilight-at-or-below-20deg-exact-identity",
        twilight_ok,
        f"{len(twilight_cases)} rows tested",
    )

    high_sun_cases = [
        case
        for case in cases
        if case.sun_elevation_deg >= SZA.reference_elevation_deg
    ]
    high_sun_ok = all(case.sza_rgb == case.input_rgb for case in high_sun_cases)
    record(
        "sza-at-or-above-60deg-exact-identity",
        high_sun_ok,
        f"{len(high_sun_cases)} rows tested",
    )

    output_values = [
        value
        for case in cases
        for rgb in (
            case.albedo_rgb,
            case.input_rgb,
            case.sza_rgb,
            case.toe_rgb,
            case.combined_rgb,
        )
        for value in rgb
    ]
    finite_bounded = all(
        math.isfinite(value) and 0.0 <= value <= 1.0 for value in output_values
    )
    record(
        "finite-unit-bounded-rgb",
        finite_bounded,
        f"{len(output_values)} channel values; max={max(output_values):.12g}",
    )

    max_chroma_error = max(case.max_chroma_error for case in cases)
    sza_gain_groups: dict[tuple[str, float, float], list[float]] = {}
    toe_gain_groups: dict[tuple[str, float, float], list[float]] = {}
    combined_gain_groups: dict[tuple[str, float, float], list[float]] = {}
    for case in cases:
        key = (case.palette, case.base_luminance, case.sun_elevation_deg)
        sza_gain_groups.setdefault(key, []).append(case.sza_gain)
        toe_gain_groups.setdefault(key, []).append(case.toe_gain)
        combined_gain_groups.setdefault(key, []).append(case.combined_effective_gain)
    max_sza_slope_spread = max(
        max(gains) - min(gains) for gains in sza_gain_groups.values()
    )
    max_toe_slope_spread = max(
        max(gains) - min(gains) for gains in toe_gain_groups.values()
    )
    max_combined_slope_spread = max(
        max(gains) - min(gains) for gains in combined_gain_groups.values()
    )
    record(
        "color-ratio-and-lighting-contrast-preservation",
        (
            max_chroma_error <= 1.0e-12
            and max_sza_slope_spread == 0.0
            and max_toe_slope_spread == 0.0
            and max_combined_slope_spread == 0.0
        ),
        (
            f"max chromaticity component error={max_chroma_error:.3e}; "
            "SZA/toe/combined gain spread across slopes="
            f"{max_sza_slope_spread:.3e}/{max_toe_slope_spread:.3e}/"
            f"{max_combined_slope_spread:.3e}"
        ),
    )

    gain_bounds_ok = all(
        1.0 <= case.sza_gain <= SZA.max_gain
        and 1.0 <= case.toe_gain <= TOE.max_gain
        and 1.0 <= case.combined_sza_gain <= SZA.max_gain
        and 1.0 <= case.combined_toe_gain <= TOE.max_gain
        and 1.0 <= case.combined_effective_gain <= SZA.max_gain * TOE.max_gain
        for case in cases
    )
    record(
        "finite-bounded-gains",
        gain_bounds_ok,
        "SZA<=1.6, toe<=1.5, combined<=2.4",
    )

    no_darkening = all(
        case.sza_y + 1.0e-15 >= case.input_y
        and case.toe_y + 1.0e-15 >= case.input_y
        and case.combined_y + 1.0e-15 >= case.input_y
        for case in cases
    )
    record("operators-never-darken", no_darkening, f"{len(cases)} rows tested")

    active_sza = sum(case.sza_rgb != case.input_rgb for case in cases)
    active_toe = sum(case.toe_rgb != case.input_rgb for case in cases)
    record(
        "grid-exercises-both-operators",
        active_sza > 0 and active_toe > 0,
        f"SZA active rows={active_sza}; toe active rows={active_toe}",
    )

    first_csv = csv_bytes(cases)
    second_csv = csv_bytes(build_cases())
    record(
        "byte-repeatable-csv",
        first_csv == second_csv,
        f"sha256={hashlib.sha256(first_csv).hexdigest()}",
    )

    return checks


def representative_case(cases: Sequence[CaseResult]) -> CaseResult:
    return next(
        case
        for case in cases
        if case.palette == "forest"
        and case.base_luminance == 0.04
        and case.sun_elevation_deg == 33.0
        and case.slope_toward_sun_deg == 0.0
    )


def make_summary(
    cases: Sequence[CaseResult],
    checks: Sequence[CheckResult],
    generator_sha256: str,
    csv_sha256: str,
    check_report_sha256: str,
) -> dict[str, object]:
    anchor = representative_case(cases)
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "generator_sha256": generator_sha256,
        "csv_sha256": csv_sha256,
        "check_report_sha256": check_report_sha256,
        "configuration": {
            "sza": asdict(SZA),
            "dark_toe": asdict(TOE),
            "toe_evaluation_domain": (
                "linear surface albedo before terrain/direct/diffuse illumination"
            ),
            "gain_application_domain": "illuminated linear surface RGB proxy",
            "diffuse_floor_proxy": DIFFUSE_FLOOR,
            "palette_names": [name for name, _ in PALETTES],
            "base_luminance_grid": list(BASE_LUMINANCE_GRID),
            "sun_elevation_grid_deg": list(SUN_ELEVATION_GRID_DEG),
            "slope_toward_sun_grid_deg": list(SLOPE_TOWARD_SUN_GRID_DEG),
        },
        "row_count": len(cases),
        "active_row_counts": {
            "sza": sum(case.sza_rgb != case.input_rgb for case in cases),
            "dark_toe": sum(case.toe_rgb != case.input_rgb for case in cases),
            "combined": sum(case.combined_rgb != case.input_rgb for case in cases),
        },
        "maximums": {
            "sza_applied_gain": max(case.sza_gain for case in cases),
            "toe_applied_gain": max(case.toe_gain for case in cases),
            "combined_effective_gain": max(
                case.combined_effective_gain for case in cases
            ),
            "output_channel": max(
                value for case in cases for value in case.combined_rgb
            ),
            "chromaticity_component_error": max(
                case.max_chroma_error for case in cases
            ),
        },
        "representative_moderate_sun_dark_forest": {
            "base_luminance": anchor.base_luminance,
            "albedo_luminance": anchor.albedo_y,
            "sun_elevation_deg": anchor.sun_elevation_deg,
            "slope_toward_sun_deg": anchor.slope_toward_sun_deg,
            "input_luminance": anchor.input_y,
            "sza_luminance": anchor.sza_y,
            "toe_luminance": anchor.toe_y,
            "combined_luminance": anchor.combined_y,
            "sza_gain": anchor.sza_gain,
            "toe_gain": anchor.toe_gain,
            "combined_effective_gain": anchor.combined_effective_gain,
        },
        "checks": [asdict(check) for check in checks],
        "all_checks_passed": all(check.passed for check in checks),
        "interpretation": (
            "The toe is classified from unilluminated linear surface albedo and "
            "applied as a scalar to the illuminated surface signal, preserving "
            "terrain-lighting ratios. This fixture selects no production default "
            "or radiometric calibration."
        ),
    }


def checks_text(
    checks: Sequence[CheckResult], generator_sha256: str, csv_sha256: str
) -> bytes:
    passed = sum(check.passed for check in checks)
    lines = [
        f"schema: {SCHEMA}",
        f"schema_version: {SCHEMA_VERSION}",
        f"method_id: {METHOD_ID}",
        f"generator_sha256: {generator_sha256}",
        f"csv_sha256: {csv_sha256}",
        f"result: {'PASS' if passed == len(checks) else 'FAIL'} ({passed}/{len(checks)})",
        "",
    ]
    lines.extend(
        f"{'PASS' if check.passed else 'FAIL'} {check.name}: {check.details}"
        for check in checks
    )
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def write_outputs(output_dir: Path, cases: Sequence[CaseResult], checks: Sequence[CheckResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    script_bytes = Path(__file__).read_bytes()
    generator_sha256 = hashlib.sha256(script_bytes).hexdigest()
    csv_payload = csv_bytes(cases)
    csv_sha256 = hashlib.sha256(csv_payload).hexdigest()
    check_payload = checks_text(checks, generator_sha256, csv_sha256)
    check_sha256 = hashlib.sha256(check_payload).hexdigest()
    summary = make_summary(
        cases,
        checks,
        generator_sha256,
        csv_sha256,
        check_sha256,
    )
    summary_payload = (
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")

    (output_dir / "stage0-surface-response-v1.csv").write_bytes(csv_payload)
    (output_dir / "stage0-surface-response-summary-v1.json").write_bytes(
        summary_payload
    )
    (output_dir / "stage0-surface-response-checks-v1.txt").write_bytes(
        check_payload
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("check", "generate", "all"),
        default="all",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args(argv)

    cases = build_cases()
    checks = run_checks(cases)
    for check in checks:
        print(f"{'PASS' if check.passed else 'FAIL'} {check.name}: {check.details}")
    if not all(check.passed for check in checks):
        return 1

    if args.command in ("generate", "all"):
        write_outputs(args.output_dir, cases, checks)
        print(f"WROTE {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
