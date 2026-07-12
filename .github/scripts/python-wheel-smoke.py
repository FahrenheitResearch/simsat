#!/usr/bin/env python3
"""Import the built wheel and exercise one tiny render without external fixtures.

The script writes a deterministic 12 x 12 x 12 clear-sky cache brick using only
the Python standard library, then calls the public ``simsat.render_ir`` binding.
It intentionally checks the installed distribution rather than importing source
from the checkout.  Keep the fixture format number in sync with the engine's
``SSB_FORMAT_VERSION``; a format bump should make this distribution contract fail
until its smoke fixture is consciously updated.
"""

from __future__ import annotations

import json
import math
import struct
import tempfile
import zlib
from pathlib import Path

import numpy as np
import simsat


FORMAT_VERSION = 5
NX = 12
NY = 12
NZ = 12
DZ_M = 250.0
TIME_ISO = "2025-01-15T12:00:00Z"
STAMP = "20250115_1200"
RUN_ID = "ci-smoke"
CHANNELS_3D = [
    "ext_liquid",
    "ext_ice",
    "ext_snow",
    "ext_precip",
    "tau_up",
    "qvapor",
    "cloud_fraction",
]
PLANES_2D = ["hgt", "landmask", "tsk", "u10", "v10"]


def f32_plane(value: float, count: int) -> bytes:
    return struct.pack(f"<{count}f", *([value] * count))


def write_tiny_cached_run(root: Path) -> Path:
    run_dir = root / RUN_ID
    run_dir.mkdir(parents=True)
    brick_path = run_dir / f"t{STAMP}.ssb"
    n3 = NX * NY * NZ
    n2 = NX * NY

    zero_quant = {name: {"vmin": 0.0, "vmax": 0.0} for name in CHANNELS_3D[:-1]}
    header = {
        "format_version": FORMAT_VERSION,
        "nx": NX,
        "ny": NY,
        "nz": NZ,
        "z_min_m": 0.0,
        "dz_m": DZ_M,
        "temperature_encoding": "celsius_f16",
        "quant": zero_quant,
        "channels_3d": CHANNELS_3D,
        "planes_2d": PLANES_2D,
        "has_snowh": False,
        "has_ivgtyp": False,
        "has_cloud_fraction": False,
        "time_iso": TIME_ISO,
    }

    payload = bytearray()
    for _ in CHANNELS_3D[:-1]:
        payload.extend(bytes(n3))
    # No model cloud-fraction field: 255 is the engine's explicit full-coverage
    # fallback. With zero hydrometeor extinction the scene remains clear.
    payload.extend(bytes([255]) * n3)
    for level in range(NZ):
        kelvin = 290.0 - 6.5 * (level * DZ_M / 1000.0)
        payload.extend(struct.pack("<e", kelvin - 273.15) * n2)
    payload.extend(f32_plane(0.0, n2))
    payload.extend(f32_plane(1.0, n2))
    payload.extend(f32_plane(291.0, n2))
    payload.extend(f32_plane(0.0, n2))
    payload.extend(f32_plane(0.0, n2))

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    brick_path.write_bytes(
        b"SSB1"
        + struct.pack("<II", FORMAT_VERSION, len(header_json))
        + header_json
        + zlib.compress(payload)
    )

    manifest = {
        "format_version": FORMAT_VERSION,
        "run_id": RUN_ID,
        "nx": NX,
        "ny": NY,
        "nz": NZ,
        "z_min_m": 0.0,
        "dz_m": DZ_M,
        "temperature_encoding": "celsius_f16",
        "quant_dynamic_range": 1.0e5,
        "channels_3d": CHANNELS_3D,
        "planes_2d": PLANES_2D,
        "projection": {
            "map_proj": 1,
            "truelat1_deg": 30.0,
            "truelat2_deg": 60.0,
            "stand_lon_deg": -97.5,
            "cen_lat_deg": 39.0,
            "cen_lon_deg": -97.5,
            "dx_m": 3000.0,
            "dy_m": 3000.0,
        },
        "timesteps": [
            {
                "key": STAMP,
                "hhmm": 1200,
                "file": brick_path.name,
                "time_iso": TIME_ISO,
                "quant": zero_quant,
                "has_cloud_fraction": False,
                "ssb_bytes": brick_path.stat().st_size,
                "source_bytes": None,
                "source_mtime_unix": None,
                "anchor": {
                    "ref_i": (NX - 1) / 2,
                    "ref_j": (NY - 1) / 2,
                    "ref_lat_deg": 39.0,
                    "ref_lon_deg": -97.5,
                    "dx": 3000.0,
                    "dy": 3000.0,
                },
            }
        ],
    }
    manifest_path = run_dir / "run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    required = (
        "render_visible_rgb",
        "render_visible_bands",
        "render_ir",
        "render_cloud_optical_depth",
    )
    missing = [name for name in required if not callable(getattr(simsat, name, None))]
    assert not missing, f"installed simsat wheel is missing public callables: {missing}"
    assert isinstance(simsat.__version__, str) and simsat.__version__

    with tempfile.TemporaryDirectory(prefix="simsat-wheel-smoke-") as temp:
        manifest = write_tiny_cached_run(Path(temp))
        brightness_temperature, georef = simsat.render_ir(
            str(manifest), view="topdown", resolution="native", threads=1
        )

    assert brightness_temperature.shape == (NY, NX)
    assert brightness_temperature.dtype == np.float32
    assert np.isfinite(brightness_temperature).all()
    median_kelvin = float(np.median(brightness_temperature))
    assert 275.0 <= median_kelvin <= 300.0, median_kelvin
    assert georef.view == "topdown"
    assert georef.extent_kind == "projection_meters"
    assert georef.lat.shape == (NY, NX)
    assert georef.lon.shape == (NY, NX)
    assert all(math.isfinite(value) for value in georef.extent)
    print(
        f"simsat {simsat.__version__}: wheel import + {NX}x{NY} IR render OK "
        f"(median {median_kelvin:.2f} K)"
    )


if __name__ == "__main__":
    main()
