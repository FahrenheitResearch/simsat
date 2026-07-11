#!/usr/bin/env python3
"""Stage-0 delta-scaled two-stream slab-flux experiment.

This is deliberately not a production cloud radiance solver.  It computes
hemispheric direct and diffuse fluxes for a homogeneous plane-parallel slab,
with a Lambertian lower boundary, using a delta-Eddington similarity transform
followed by a two-discrete-ordinate (mu=+/-1/sqrt(3)) flux closure.

The implementation uses exact homogeneous-layer propagation and stable adding
of layer reflection/transmission/source operators.  It has no dependencies
beyond the Python standard library.  Generated fixtures are deterministic and
contain no wall-clock timestamp.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


SCHEMA = "simsat.delta-two-stream.stage0"
SCHEMA_VERSION = 1
METHOD_ID = "delta-eddington-s2-gauss-flux-v1"
MU_STREAM = 1.0 / math.sqrt(3.0)
MAX_SCALED_LAYER_TAU = 0.25

# Current production comparison inputs.  These are emitted only so later
# analysis can join the slab fixtures to the legacy directional-source runs.
# This prototype does not evaluate the legacy phase-weighted radiance source.
LEGACY_OCTAVES = 6
LEGACY_EXTINCTION_SCALE = 0.5
LEGACY_PHASE_SCALE = 0.5
LEGACY_BRIGHTNESS_SCALE = 0.85
LEGACY_WEIGHT_SUM = sum(
    LEGACY_BRIGHTNESS_SCALE**k for k in range(LEGACY_OCTAVES)
)

TAU_GRID = (
    0.0,
    1.0e-4,
    2.0e-4,
    1.0e-3,
    1.0e-2,
    3.0e-2,
    1.0e-1,
    3.0e-1,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
)
SZA_GRID_DEG = (0.0, 30.0, 50.0, 65.0, 75.0)
PHASE_CASES = (
    ("isotropic-control", 0.0),
    ("ice-dual-hg-moment", 0.665),
    ("liquid-dual-hg-moment", 0.75),
    ("forward-stress", 0.85),
)
SSA_GRID = (1.0, 0.999, 0.99, 0.95, 0.8, 0.0)
SURFACE_ALBEDO_GRID = (0.0, 0.1, 0.3, 0.8)


@dataclass(frozen=True)
class ScatteringOperator:
    """Two-boundary diffuse-flux operator with internal direct-beam sources.

    At the top boundary, ``down_top`` is incident and ``up_top`` is outgoing.
    At the bottom boundary, ``up_bottom`` is incident and ``down_bottom`` is
    outgoing::

        up_top     = r_top * down_top + t_up * up_bottom + source_up
        down_bottom = t_down * down_top + r_bottom * up_bottom + source_down
    """

    r_top: float
    r_bottom: float
    t_down: float
    t_up: float
    source_up: float
    source_down: float

    @staticmethod
    def identity() -> "ScatteringOperator":
        return ScatteringOperator(0.0, 0.0, 1.0, 1.0, 0.0, 0.0)


@dataclass(frozen=True)
class SlabResult:
    method: str
    phase_case: str
    tau: float
    solar_zenith_deg: float
    mu0: float
    asymmetry_g: float
    single_scatter_albedo: float
    surface_albedo: float
    delta_fraction: float
    scaled_tau: float
    scaled_asymmetry_g: float
    scaled_single_scatter_albedo: float
    direct_transmittance_bottom: float
    diffuse_down_bottom: float
    diffuse_up_surface: float
    toa_reflectance: float
    total_down_surface: float
    surface_absorptance: float
    atmosphere_absorptance: float
    energy_sum: float
    intrinsic_diffuse_reflectance: float
    intrinsic_diffuse_transmittance: float
    intrinsic_diffuse_absorptance: float
    black_surface_single_scatter_reflectance: float
    black_surface_single_scatter_diffuse_transmittance: float
    black_surface_total_diffuse_escape: float
    black_surface_multiple_scatter_escape: float
    legacy_support_tau: float
    legacy_thin_gate: float
    legacy_octaves: int
    legacy_extinction_scale: float
    legacy_phase_scale: float
    legacy_brightness_scale: float
    legacy_weight_sum: float


def script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def delta_eddington_scale(tau: float, ssa: float, g: float) -> tuple[float, float, float, float]:
    """Return ``(f, scaled_tau, scaled_ssa, scaled_g)``.

    Joseph-Wiscombe-Weinman delta-Eddington scaling uses ``f=g^2``.  This
    Stage-0 experiment is intentionally restricted to nonnegative cloud-like
    asymmetry factors.  The transform exactly preserves absorption optical
    depth: ``(1-scaled_ssa)*scaled_tau == (1-ssa)*tau`` apart from rounding.
    """

    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError(f"tau must be finite and nonnegative, got {tau!r}")
    if not math.isfinite(ssa) or not 0.0 <= ssa <= 1.0:
        raise ValueError(f"single-scatter albedo must be in [0,1], got {ssa!r}")
    if not math.isfinite(g) or not 0.0 <= g < 1.0:
        raise ValueError(f"cloud asymmetry factor must be in [0,1), got {g!r}")
    f = g * g
    denominator = 1.0 - ssa * f
    scaled_tau = denominator * tau
    scaled_ssa = ((1.0 - f) * ssa / denominator) if denominator > 0.0 else 0.0
    scaled_g = (g - f) / (1.0 - f) if f < 1.0 else 0.0
    return f, scaled_tau, scaled_ssa, scaled_g


def matrix_exp_transport(a: float, b: float, distance: float) -> tuple[float, float, float, float]:
    """Exponential of ``A*distance`` for ``A=[[-a,b],[-b,a]]``."""

    kappa_sq = a * a - b * b
    if kappa_sq < -1.0e-13:
        raise ArithmeticError(f"non-real two-stream eigenvalue: {kappa_sq}")
    kappa_sq = max(0.0, kappa_sq)
    if kappa_sq <= 1.0e-16:
        # sinh(k d)/k = d + k^2 d^3/6 + ...
        c = 1.0 + 0.5 * kappa_sq * distance * distance
        s_over_k = distance + kappa_sq * distance**3 / 6.0
    else:
        kappa = math.sqrt(kappa_sq)
        kd = kappa * distance
        c = math.cosh(kd)
        s_over_k = math.sinh(kd) / kappa
    return (
        c - a * s_over_k,
        b * s_over_k,
        -b * s_over_k,
        c + a * s_over_k,
    )


def mat_vec(matrix: tuple[float, float, float, float], vector: tuple[float, float]) -> tuple[float, float]:
    m00, m01, m10, m11 = matrix
    x, y = vector
    return m00 * x + m01 * y, m10 * x + m11 * y


def source_integral_simpson(
    a: float,
    b: float,
    distance: float,
    inverse_mu0: float,
    source: tuple[float, float],
) -> tuple[float, float]:
    """Rare near-singular fallback for the exact exponential-source integral."""

    intervals = 64
    h = distance / intervals
    acc0 = 0.0
    acc1 = 0.0
    for index in range(intervals + 1):
        s = index * h
        propagator = matrix_exp_transport(a, b, distance - s)
        v0, v1 = mat_vec(propagator, source)
        attenuation = math.exp(-inverse_mu0 * s)
        weight = 1.0 if index in (0, intervals) else (4.0 if index % 2 else 2.0)
        acc0 += weight * attenuation * v0
        acc1 += weight * attenuation * v1
    return acc0 * h / 3.0, acc1 * h / 3.0


def unit_direct_source_integral(
    a: float,
    b: float,
    distance: float,
    mu0: float,
    source: tuple[float, float],
    propagator: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Integrate ``exp(A*(d-s))*source*exp(-s/mu0)`` exactly when regular."""

    inverse_mu0 = 1.0 / mu0
    m00, m01, m10, m11 = propagator
    direct_t = math.exp(-distance * inverse_mu0)
    difference = (m00 - direct_t, m01, m10, m11 - direct_t)
    rhs0, rhs1 = mat_vec(difference, source)

    # B = A + inverse_mu0*I.  det(B) can vanish for a special stream/sun
    # geometry; Simpson integration is deterministic and accurate in that rare
    # branch because each stable layer is at most MAX_SCALED_LAYER_TAU thick.
    b00 = inverse_mu0 - a
    b01 = b
    b10 = -b
    b11 = inverse_mu0 + a
    determinant = b00 * b11 - b01 * b10
    if abs(determinant) <= 1.0e-11:
        return source_integral_simpson(a, b, distance, inverse_mu0, source)
    return (
        (b11 * rhs0 - b01 * rhs1) / determinant,
        (-b10 * rhs0 + b00 * rhs1) / determinant,
    )


def layer_operator(
    scaled_tau: float,
    mu0: float,
    scaled_ssa: float,
    scaled_g: float,
    direct_flux_at_top: float,
) -> ScatteringOperator:
    """Construct one exact, numerically tame homogeneous-layer operator."""

    a = (1.0 - 0.5 * scaled_ssa * (1.0 + scaled_g)) / MU_STREAM
    b = 0.5 * scaled_ssa * (1.0 - scaled_g) / MU_STREAM
    propagator = matrix_exp_transport(a, b, scaled_tau)

    angular = 3.0 * scaled_g * MU_STREAM * mu0
    source_down = scaled_ssa * (1.0 + angular) / (2.0 * mu0)
    source_up = scaled_ssa * (1.0 - angular) / (2.0 * mu0)
    if source_down < -1.0e-13 or source_up < -1.0e-13:
        raise ArithmeticError(
            "delta-scaled P1 direct source became negative: "
            f"down={source_down}, up={source_up}"
        )
    source_vector = (max(0.0, source_down), -max(0.0, source_up))
    q0, q1 = unit_direct_source_integral(
        a, b, scaled_tau, mu0, source_vector, propagator
    )
    q0 *= direct_flux_at_top
    q1 *= direct_flux_at_top

    m00, m01, m10, m11 = propagator
    if not math.isfinite(m11) or m11 <= 0.0:
        raise ArithmeticError(f"invalid layer transfer denominator {m11}")
    determinant = m00 * m11 - m01 * m10
    return ScatteringOperator(
        r_top=-m10 / m11,
        r_bottom=m01 / m11,
        t_down=determinant / m11,
        t_up=1.0 / m11,
        source_up=-q1 / m11,
        source_down=q0 - m01 * q1 / m11,
    )


def combine(top: ScatteringOperator, bottom: ScatteringOperator) -> ScatteringOperator:
    """Stable adding of adjacent top and bottom scattering operators."""

    denominator = 1.0 - bottom.r_top * top.r_bottom
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ArithmeticError(f"invalid adding denominator {denominator}")
    return ScatteringOperator(
        r_top=top.r_top
        + top.t_up * bottom.r_top * top.t_down / denominator,
        r_bottom=bottom.r_bottom
        + bottom.t_down * top.r_bottom * bottom.t_up / denominator,
        t_down=bottom.t_down * top.t_down / denominator,
        t_up=top.t_up * bottom.t_up / denominator,
        source_up=top.source_up
        + top.t_up
        * (bottom.r_top * top.source_down + bottom.source_up)
        / denominator,
        source_down=bottom.source_down
        + bottom.t_down
        * (top.source_down + top.r_bottom * bottom.source_up)
        / denominator,
    )


def build_slab_operator(
    scaled_tau: float,
    mu0: float,
    scaled_ssa: float,
    scaled_g: float,
    max_layer_tau: float = MAX_SCALED_LAYER_TAU,
) -> ScatteringOperator:
    if scaled_tau == 0.0:
        return ScatteringOperator.identity()
    if not math.isfinite(max_layer_tau) or max_layer_tau <= 0.0:
        raise ValueError("max_layer_tau must be finite and positive")
    layer_count = max(1, math.ceil(scaled_tau / max_layer_tau))
    layer_tau = scaled_tau / layer_count
    total = ScatteringOperator.identity()
    traversed = 0.0
    for _ in range(layer_count):
        direct_at_top = math.exp(-traversed / mu0)
        layer = layer_operator(layer_tau, mu0, scaled_ssa, scaled_g, direct_at_top)
        total = combine(total, layer)
        traversed += layer_tau
    return total


def attenuated_integral(rate: float, distance: float) -> float:
    if abs(rate) <= 1.0e-12:
        return distance
    return -math.expm1(-rate * distance) / rate


def first_order_black_surface(
    scaled_tau: float, mu0: float, scaled_ssa: float, scaled_g: float
) -> tuple[float, float]:
    """Two-stream first atmospheric scattering order for a black lower boundary."""

    if scaled_tau == 0.0 or scaled_ssa == 0.0:
        return 0.0, 0.0
    angular = 3.0 * scaled_g * MU_STREAM * mu0
    down0 = scaled_ssa * (1.0 + angular) / (2.0 * mu0)
    up0 = scaled_ssa * (1.0 - angular) / (2.0 * mu0)

    reflected = up0 * attenuated_integral(
        1.0 / mu0 + 1.0 / MU_STREAM, scaled_tau
    )
    difference = 1.0 / MU_STREAM - 1.0 / mu0
    if abs(difference) <= 1.0e-12:
        transmitted_diffuse = down0 * scaled_tau * math.exp(-scaled_tau / MU_STREAM)
    else:
        transmitted_diffuse = down0 * (
            math.exp(-scaled_tau / mu0) - math.exp(-scaled_tau / MU_STREAM)
        ) / difference
    return reflected, transmitted_diffuse


def solve_slab(
    *,
    phase_case: str,
    tau: float,
    solar_zenith_deg: float,
    g: float,
    ssa: float,
    surface_albedo: float,
) -> SlabResult:
    if not math.isfinite(solar_zenith_deg) or not 0.0 <= solar_zenith_deg < 90.0:
        raise ValueError("solar zenith must be finite and in [0,90) degrees")
    if not math.isfinite(surface_albedo) or not 0.0 <= surface_albedo <= 1.0:
        raise ValueError("surface albedo must be finite and in [0,1]")
    mu0 = math.cos(math.radians(solar_zenith_deg))
    delta_fraction, scaled_tau, scaled_ssa, scaled_g = delta_eddington_scale(tau, ssa, g)
    operator = build_slab_operator(scaled_tau, mu0, scaled_ssa, scaled_g)
    direct_bottom = math.exp(-scaled_tau / mu0)

    surface_denominator = 1.0 - surface_albedo * operator.r_bottom
    if surface_denominator <= 0.0:
        raise ArithmeticError(f"invalid surface denominator {surface_denominator}")
    diffuse_up_surface = surface_albedo * (
        operator.source_down + direct_bottom
    ) / surface_denominator
    diffuse_down_bottom = (
        operator.r_bottom * diffuse_up_surface + operator.source_down
    )
    toa_reflectance = operator.t_up * diffuse_up_surface + operator.source_up
    total_down_surface = direct_bottom + diffuse_down_bottom
    surface_absorptance = (1.0 - surface_albedo) * total_down_surface
    atmosphere_absorptance = 1.0 - toa_reflectance - surface_absorptance
    energy_sum = toa_reflectance + surface_absorptance + atmosphere_absorptance

    first_reflect, first_transmit = first_order_black_surface(
        scaled_tau, mu0, scaled_ssa, scaled_g
    )
    black_total_diffuse_escape = operator.source_up + operator.source_down
    multiple_escape = black_total_diffuse_escape - first_reflect - first_transmit
    # Suppress only last-bit negative cancellation in the optically-thin limit.
    if multiple_escape < 0.0 and abs(multiple_escape) <= 2.0e-13:
        multiple_escape = 0.0

    intrinsic_absorption = 1.0 - operator.r_top - operator.t_down
    return SlabResult(
        method=METHOD_ID,
        phase_case=phase_case,
        tau=tau,
        solar_zenith_deg=solar_zenith_deg,
        mu0=mu0,
        asymmetry_g=g,
        single_scatter_albedo=ssa,
        surface_albedo=surface_albedo,
        delta_fraction=delta_fraction,
        scaled_tau=scaled_tau,
        scaled_asymmetry_g=scaled_g,
        scaled_single_scatter_albedo=scaled_ssa,
        direct_transmittance_bottom=direct_bottom,
        diffuse_down_bottom=diffuse_down_bottom,
        diffuse_up_surface=diffuse_up_surface,
        toa_reflectance=toa_reflectance,
        total_down_surface=total_down_surface,
        surface_absorptance=surface_absorptance,
        atmosphere_absorptance=atmosphere_absorptance,
        energy_sum=energy_sum,
        intrinsic_diffuse_reflectance=operator.r_top,
        intrinsic_diffuse_transmittance=operator.t_down,
        intrinsic_diffuse_absorptance=intrinsic_absorption,
        black_surface_single_scatter_reflectance=first_reflect,
        black_surface_single_scatter_diffuse_transmittance=first_transmit,
        black_surface_total_diffuse_escape=black_total_diffuse_escape,
        black_surface_multiple_scatter_escape=multiple_escape,
        legacy_support_tau=tau,
        legacy_thin_gate=-math.expm1(-tau),
        legacy_octaves=LEGACY_OCTAVES,
        legacy_extinction_scale=LEGACY_EXTINCTION_SCALE,
        legacy_phase_scale=LEGACY_PHASE_SCALE,
        legacy_brightness_scale=LEGACY_BRIGHTNESS_SCALE,
        legacy_weight_sum=LEGACY_WEIGHT_SUM,
    )


def fixture_rows() -> list[SlabResult]:
    rows: list[SlabResult] = []
    for tau in TAU_GRID:
        for solar_zenith in SZA_GRID_DEG:
            for phase_case, g in PHASE_CASES:
                for ssa in SSA_GRID:
                    for albedo in SURFACE_ALBEDO_GRID:
                        rows.append(
                            solve_slab(
                                phase_case=phase_case,
                                tau=tau,
                                solar_zenith_deg=solar_zenith,
                                g=g,
                                ssa=ssa,
                                surface_albedo=albedo,
                            )
                        )
    return rows


def close(actual: float, expected: float, *, absolute: float, relative: float = 0.0) -> bool:
    return abs(actual - expected) <= absolute + relative * abs(expected)


def check_zero_tau() -> dict[str, object]:
    maximum_error = 0.0
    for albedo in SURFACE_ALBEDO_GRID:
        result = solve_slab(
            phase_case="liquid-dual-hg-moment",
            tau=0.0,
            solar_zenith_deg=65.0,
            g=0.75,
            ssa=1.0,
            surface_albedo=albedo,
        )
        errors = (
            abs(result.direct_transmittance_bottom - 1.0),
            abs(result.toa_reflectance - albedo),
            abs(result.surface_absorptance - (1.0 - albedo)),
            abs(result.atmosphere_absorptance),
            abs(result.diffuse_down_bottom),
        )
        maximum_error = max(maximum_error, *errors)
    if maximum_error > 2.0e-13:
        raise AssertionError(f"tau=0 identity error {maximum_error}")
    return {"status": "pass", "maximum_absolute_error": maximum_error}


def check_conservative_energy() -> dict[str, object]:
    maximum_absorption = 0.0
    maximum_energy_error = 0.0
    maximum_intrinsic_error = 0.0
    cases = 0
    for tau in (1.0e-3, 0.1, 1.0, 10.0, 100.0):
        for sza in (0.0, 50.0, 75.0):
            for _, g in PHASE_CASES:
                for albedo in (0.0, 0.3, 0.8, 1.0):
                    result = solve_slab(
                        phase_case="conservative-check",
                        tau=tau,
                        solar_zenith_deg=sza,
                        g=g,
                        ssa=1.0,
                        surface_albedo=albedo,
                    )
                    maximum_absorption = max(
                        maximum_absorption, abs(result.atmosphere_absorptance)
                    )
                    maximum_energy_error = max(
                        maximum_energy_error, abs(result.energy_sum - 1.0)
                    )
                    maximum_intrinsic_error = max(
                        maximum_intrinsic_error,
                        abs(
                            result.intrinsic_diffuse_reflectance
                            + result.intrinsic_diffuse_transmittance
                            - 1.0
                        ),
                    )
                    cases += 1
    if maximum_absorption > 2.0e-10 or maximum_intrinsic_error > 2.0e-10:
        raise AssertionError(
            "conservative-energy failure: "
            f"atmos={maximum_absorption}, intrinsic={maximum_intrinsic_error}"
        )
    return {
        "status": "pass",
        "cases": cases,
        "maximum_atmospheric_absorptance_magnitude": maximum_absorption,
        "maximum_energy_sum_error": maximum_energy_error,
        "maximum_intrinsic_r_plus_t_error": maximum_intrinsic_error,
    }


def check_pure_absorption() -> dict[str, object]:
    tau = 2.0
    sza = 60.0
    mu0 = math.cos(math.radians(sza))
    albedo = 0.3
    result = solve_slab(
        phase_case="pure-absorption",
        tau=tau,
        solar_zenith_deg=sza,
        g=0.75,
        ssa=0.0,
        surface_albedo=albedo,
    )
    direct = math.exp(-tau / mu0)
    expected_reflected = albedo * direct * math.exp(-tau / MU_STREAM)
    expected_surface_absorbed = (1.0 - albedo) * direct
    expected_atmosphere_absorbed = 1.0 - expected_reflected - expected_surface_absorbed
    maximum_error = max(
        abs(result.toa_reflectance - expected_reflected),
        abs(result.surface_absorptance - expected_surface_absorbed),
        abs(result.atmosphere_absorptance - expected_atmosphere_absorbed),
        abs(result.diffuse_down_bottom),
    )
    if maximum_error > 2.0e-12:
        raise AssertionError(f"pure-absorption analytic error {maximum_error}")
    return {
        "status": "pass",
        "maximum_absolute_error": maximum_error,
        "expected_atmosphere_absorptance": expected_atmosphere_absorbed,
    }


def check_thin_multiple_scatter_order() -> dict[str, object]:
    records: list[dict[str, float]] = []
    for g in (0.0, 0.665, 0.75, 0.85):
        small = solve_slab(
            phase_case="thin-order",
            tau=1.0e-3,
            solar_zenith_deg=50.0,
            g=g,
            ssa=1.0,
            surface_albedo=0.0,
        ).black_surface_multiple_scatter_escape
        large = solve_slab(
            phase_case="thin-order",
            tau=2.0e-3,
            solar_zenith_deg=50.0,
            g=g,
            ssa=1.0,
            surface_albedo=0.0,
        ).black_surface_multiple_scatter_escape
        if small <= 0.0 or large <= 0.0:
            raise AssertionError(f"nonpositive thin multiple-scatter escape for g={g}")
        slope = math.log(large / small) / math.log(2.0)
        records.append({"g": g, "small": small, "large": large, "log_slope": slope})
        if not 1.94 <= slope <= 2.02:
            raise AssertionError(f"thin multiple-scatter order for g={g}: {slope}")
    return {"status": "pass", "records": records}


def check_layer_partition_invariance() -> dict[str, object]:
    maximum_error = 0.0
    cases = 0
    for tau in (0.1, 3.0, 30.0, 100.0):
        for ssa in (1.0, 0.999, 0.8, 0.0):
            _, scaled_tau, scaled_ssa, scaled_g = delta_eddington_scale(tau, ssa, 0.75)
            coarse = build_slab_operator(
                scaled_tau, 0.5, scaled_ssa, scaled_g, max_layer_tau=0.5
            )
            fine = build_slab_operator(
                scaled_tau, 0.5, scaled_ssa, scaled_g, max_layer_tau=0.05
            )
            for field in asdict(coarse):
                maximum_error = max(
                    maximum_error,
                    abs(float(getattr(coarse, field)) - float(getattr(fine, field))),
                )
            cases += 1
    if maximum_error > 3.0e-11:
        raise AssertionError(f"layer partition changed exact slab solution: {maximum_error}")
    return {"status": "pass", "cases": cases, "maximum_absolute_error": maximum_error}


def check_absorption_and_bounds(rows: Sequence[SlabResult]) -> dict[str, object]:
    tolerance = 3.0e-10
    minimum = math.inf
    maximum = -math.inf
    maximum_energy_error = 0.0
    checked_values = 0
    bounded_energy_fields = (
        "direct_transmittance_bottom",
        "toa_reflectance",
        "surface_absorptance",
        "atmosphere_absorptance",
        "intrinsic_diffuse_reflectance",
        "intrinsic_diffuse_transmittance",
        "intrinsic_diffuse_absorptance",
        "black_surface_single_scatter_reflectance",
        "black_surface_single_scatter_diffuse_transmittance",
        "black_surface_total_diffuse_escape",
        "black_surface_multiple_scatter_escape",
    )
    internal_flux_fields = (
        "diffuse_down_bottom",
        "diffuse_up_surface",
        "total_down_surface",
    )
    absorbing_positive = 0
    for row in rows:
        maximum_energy_error = max(maximum_energy_error, abs(row.energy_sum - 1.0))
        if row.tau > 0.0 and row.single_scatter_albedo < 1.0 and row.atmosphere_absorptance > 0.0:
            absorbing_positive += 1
        for field in bounded_energy_fields:
            value = float(getattr(row, field))
            if not math.isfinite(value):
                raise AssertionError(f"nonfinite {field}: {row}")
            minimum = min(minimum, value)
            maximum = max(maximum, value)
            checked_values += 1
            if value < -tolerance or value > 1.0 + tolerance:
                raise AssertionError(f"out-of-bounds {field}={value}: {row}")
        # Internal boundary flux can exceed the unit incident flux because a
        # reflective surface and conservative slab recycle photons.  It remains
        # bounded by the Lambert-cavity geometric ceiling for A_s < 1.
        cavity_ceiling = 1.0 / (1.0 - row.surface_albedo)
        for field in internal_flux_fields:
            value = float(getattr(row, field))
            if not math.isfinite(value):
                raise AssertionError(f"nonfinite {field}: {row}")
            minimum = min(minimum, value)
            maximum = max(maximum, value)
            checked_values += 1
            if value < -tolerance or value > cavity_ceiling + tolerance:
                raise AssertionError(
                    f"internal flux exceeds cavity bound {field}={value} > "
                    f"{cavity_ceiling}: {row}"
                )
        if row.intrinsic_diffuse_reflectance + row.intrinsic_diffuse_transmittance > 1.0 + tolerance:
            raise AssertionError(f"intrinsic R+T exceeds one: {row}")
    if maximum_energy_error > 2.0e-15:
        raise AssertionError(f"energy bookkeeping error {maximum_energy_error}")
    if absorbing_positive == 0:
        raise AssertionError("absorption grid contains no positive atmospheric absorption")
    return {
        "status": "pass",
        "rows": len(rows),
        "checked_values": checked_values,
        "minimum_bounded_value": minimum,
        "maximum_bounded_value": maximum,
        "maximum_energy_sum_error": maximum_energy_error,
        "absorbing_rows_with_positive_atmosphere_absorptance": absorbing_positive,
    }


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def check_repeatability(rows: Sequence[SlabResult]) -> dict[str, object]:
    first = canonical_json_bytes([asdict(row) for row in rows])
    second_rows = fixture_rows()
    second = canonical_json_bytes([asdict(row) for row in second_rows])
    if first != second:
        raise AssertionError("fixture generation is not byte-repeatable")
    digest = hashlib.sha256(first).hexdigest()
    return {"status": "pass", "row_json_sha256": digest, "bytes": len(first)}


def run_checks(rows: Sequence[SlabResult]) -> dict[str, object]:
    checks = {
        "tau_zero_identity": check_zero_tau(),
        "conservative_energy": check_conservative_energy(),
        "pure_absorption_analytic": check_pure_absorption(),
        "thin_multiple_scatter_order": check_thin_multiple_scatter_order(),
        "layer_partition_invariance": check_layer_partition_invariance(),
        "finite_bounded_absorption_grid": check_absorption_and_bounds(rows),
        "repeatability": check_repeatability(rows),
    }
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "method": METHOD_ID,
        "script_sha256": script_sha256(),
        "status": "pass",
        "checks": checks,
    }


def metadata(rows: Sequence[SlabResult]) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "method": METHOD_ID,
        "script_sha256": script_sha256(),
        "row_count": len(rows),
        "normalization": {
            "incident_horizontal_direct_flux_at_toa": 1.0,
            "optical_depth_coordinate": "increases downward",
            "diffuse_stream_cosine": MU_STREAM,
            "surface": "Lambertian flux albedo",
        },
        "grid": {
            "tau": list(TAU_GRID),
            "solar_zenith_deg": list(SZA_GRID_DEG),
            "phase_cases": [
                {"name": name, "asymmetry_g": g} for name, g in PHASE_CASES
            ],
            "single_scatter_albedo": list(SSA_GRID),
            "surface_albedo": list(SURFACE_ALBEDO_GRID),
        },
        "legacy_comparison_inputs": {
            "scope": (
                "Join keys/inputs only. The legacy implementation is a directional, "
                "phase-weighted source and is not solved by this hemispheric prototype."
            ),
            "octaves": LEGACY_OCTAVES,
            "extinction_scale": LEGACY_EXTINCTION_SCALE,
            "phase_scale": LEGACY_PHASE_SCALE,
            "brightness_scale": LEGACY_BRIGHTNESS_SCALE,
            "weight_sum": LEGACY_WEIGHT_SUM,
            "thin_gate": "1-exp(-legacy_support_tau), applied once per higher order",
        },
        "limitations": [
            "Hemispheric two-stream fluxes only; no view-direction radiance.",
            "Homogeneous plane-parallel slab; no 3-D horizontal transport.",
            "P1 phase moment after delta scaling; no Mie or ice phase table.",
            "One shortwave band at a time; no ABI spectral-response convolution.",
            "The internal diffuse source at arbitrary marcher depth is not yet defined.",
            "The production cloud OD calibration and display transform are not applied.",
        ],
        "future_reference_comparisons": {
            "disort": [
                "TOA hemispheric reflectance",
                "direct and diffuse flux at slab bottom",
                "atmospheric absorptance",
                "vertical diffuse-flux profiles at matched optical-depth levels",
            ],
            "cuda_monte_carlo": [
                "The same 1-D slab flux quantities with uncertainty bars",
                "Directional TOA radiance versus view zenith and relative azimuth",
                "Scattering-order histograms and path-length distributions",
                "3-D cube/broken-cumulus/anvil-edge radiance and horizontal transport",
            ],
        },
    }


def write_outputs(output_dir: Path, rows: Sequence[SlabResult], checks: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    row_dicts = [asdict(row) for row in rows]
    csv_path = output_dir / "stage0-slab-flux-v1.csv"
    json_path = output_dir / "stage0-slab-flux-v1.json"
    checks_path = output_dir / "stage0-self-checks-v1.json"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(row_dicts[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(row_dicts)

    payload = metadata(rows)
    payload["rows"] = row_dicts
    json_path.write_bytes(canonical_json_bytes(payload))
    checks_path.write_bytes(canonical_json_bytes(checks))

    for path in (csv_path, json_path, checks_path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"wrote {path} sha256={digest}")


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("check", "generate", "all"),
        nargs="?",
        default="all",
        help="run checks, write fixtures, or do both (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="fixture directory (default: experiments/delta-two-stream/fixtures)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows = fixture_rows()
    checks: dict[str, object] | None = None
    if args.command in ("check", "all"):
        checks = run_checks(rows)
        print(json.dumps(checks, indent=2, sort_keys=True, allow_nan=False))
    if args.command in ("generate", "all"):
        if checks is None:
            checks = run_checks(rows)
        write_outputs(args.output_dir, rows, checks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
