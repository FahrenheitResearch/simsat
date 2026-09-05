#!/usr/bin/env python3
"""Run one immutable WRF/HRRR-to-ABI collocation case and append a scoreboard row.

Required: --input --time --out --bin. Day cases render satellite-view visible
samples on the native model grid for display and sensor-fast-gray; IR stays
native topdown. --night-ir explicitly skips both visible products. No gray RGB
channel is presented as an ABI band. The visible both-cloudy split uses a
vertical model condensate-column proxy, not an oblique ray-intersection mask.

--reference may reuse an aligned NPZ (or its containing bundle) after its time,
platform, sector, content hash and target mesh are verified. Otherwise the
repository fetcher downloads, aligns and previews the selected NOAA products.
Python dependencies: numpy, netCDF4, Pillow and pyproj (fetcher); eccodes for GRIB.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parent.parent
# The same model geometry constants/guards as camera.rs and frame.rs, not case fits.
EARTH_RADIUS_M = 6_370_000.0
BOUNDARY_INSET_CELLS = 0.01
SOURCE_GRID_TOLERANCE_CELLS = 0.05  # frame.rs stored-coordinate correctness ratchet
REFERENCE_GRID_TOLERANCE_CELLS = 0.02  # API georef round-trip tolerance
SATELLITES = {'goes19': 'goes-east', 'goes18': 'goes-west'}
CLOUD_NOTE = ('Visible both-cloudy = observed cloud and vertical model condensate-column proxy; '
              'satellite rays can cross cloud in neighboring columns. IR both-cloudy is the original '
              'topdown column split. Neither split removes forecast cloud-height/structure error.')


def import_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def helpers():
    return import_script('simsat_case_score_helpers', REPO / 'scripts/simsat-case-score.py')


def parse_time(value):
    value = value.strip().replace('_', 'T')
    parsed = datetime.fromisoformat(value[:-1] + '+00:00' if value.endswith('Z') else value)
    if parsed.tzinfo is None:
        raise ValueError('time requires an explicit UTC offset or Z')
    return parsed.astimezone(timezone.utc)


def iso_time(value):
    return value.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def select_wrf_time(times, target, requested_index=None):
    parsed = [parse_time(value.replace('_', 'T') + ('Z' if not value.endswith('Z') else '')) for value in times]
    if requested_index is not None:
        if requested_index < 0 or requested_index >= len(parsed):
            raise ValueError('WRF timestep is outside Times')
        if parsed[requested_index] != target:
            raise ValueError(f'WRF Times[{requested_index}]={iso_time(parsed[requested_index])} != requested {iso_time(target)}')
        return requested_index
    matches = [i for i, value in enumerate(parsed) if value == target]
    if len(matches) != 1:
        raise ValueError(f'requested time {iso_time(target)} must match exactly one WRF Times entry; found {len(matches)}')
    return matches[0]


class ModelProjection:
    """NumPy adapter of frame.rs MapProjection for WRF MAP_PROJ 1/2/3/6.

    Preserves engine sphere, angular wrapping, standard-parallel guards and
    integer center-cell anchor. It does not use an Earth ellipsoid for WRF rays.
    """
    def __init__(self, np, params):
        self.np, self.p = np, params
        self.kind, self.lon0 = int(params['map_proj']), float(params['stand_lon'])
        phi1 = math.radians(max(-89.9, min(89.9, params['truelat1'])))
        phi2 = math.radians(max(-89.9, min(89.9, params['truelat2'])))
        if self.kind == 1:
            self.n = (math.sin(phi1) if abs(params['truelat1'] - params['truelat2']) < 1e-10 else
                      (math.log(math.cos(phi1)) - math.log(math.cos(phi2))) /
                      (math.log(math.tan(math.pi / 4 + phi2 / 2)) - math.log(math.tan(math.pi / 4 + phi1 / 2))))
            if abs(self.n) < 1e-8:
                self.n = math.sin(phi1) if abs(phi1) >= 1e-8 else math.sin(math.radians(10))
            self.f = math.cos(phi1) * math.tan(math.pi / 4 + phi1 / 2) ** self.n / self.n
        elif self.kind == 2:
            self.k, self.south = (1 + math.sin(phi1)) / 2, params['cen_lat'] < 0
        elif self.kind == 3:
            self.scale = max(math.cos(phi1), 1e-6)
        elif self.kind != 6:
            raise ValueError(f'case grid adapter does not support MAP_PROJ={self.kind}; no approximate grid will be substituted')

    def wrap(self, value):
        result = (self.np.asarray(value) + 180.0) % 360.0 - 180.0
        return self.np.where(result == -180, 180, result)

    def forward(self, lat, lon):
        np = self.np
        lat, lon = np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)
        phi = np.radians(np.clip(lat, -89.999, 89.999))
        theta = np.radians(self.wrap(lon - self.lon0))
        if self.kind == 1:
            rho = EARTH_RADIUS_M * self.f / np.tan(math.pi / 4 + phi / 2) ** self.n
            return rho * np.sin(self.n * theta), -rho * np.cos(self.n * theta)
        if self.kind == 2:
            rho = 2 * EARTH_RADIUS_M * self.k * np.tan(math.pi / 4 + (phi / 2 if self.south else -phi / 2))
            return rho * np.sin(theta), rho * np.cos(theta) * (1 if self.south else -1)
        if self.kind == 3:
            return EARTH_RADIUS_M * self.scale * theta, EARTH_RADIUS_M * self.scale * np.log(np.tan(math.pi / 4 + phi / 2))
        return self.wrap(lon - self.lon0), np.clip(lat, -89.999, 89.999)

    def inverse(self, u, v):
        np = self.np
        if self.kind == 1:
            sign = 1 if self.n >= 0 else -1
            rho = np.sqrt(u * u + v * v) * sign
            phi = 2 * np.arctan((EARTH_RADIUS_M * self.f / rho) ** (1 / self.n)) - math.pi / 2
            lon = self.lon0 + np.degrees(np.arctan2(u * sign, -v * sign) / self.n)
            return np.degrees(phi), self.wrap(lon)
        if self.kind == 2:
            angle = 2 * np.arctan(np.sqrt(u * u + v * v) / (2 * EARTH_RADIUS_M * self.k))
            phi = angle - math.pi / 2 if self.south else math.pi / 2 - angle
            lon = self.lon0 + np.degrees(np.arctan2(u, v if self.south else -v))
            return np.degrees(phi), self.wrap(lon)
        if self.kind == 3:
            return np.degrees(2 * np.arctan(np.exp(v / (EARTH_RADIUS_M * self.scale))) - math.pi / 2), self.wrap(self.lon0 + np.degrees(u / (EARTH_RADIUS_M * self.scale)))
        return np.clip(v, -89.999, 89.999), self.wrap(u + self.lon0)


def model_grid(np, lat_south, lon_south, land_south, params):
    lat_south = np.asarray(lat_south, np.float32)
    lon_south = np.asarray(lon_south, np.float32)
    if lat_south.ndim != 2 or lat_south.shape != lon_south.shape:
        raise ValueError('source latitude/longitude must be matching 2-D arrays')
    ny, nx = lat_south.shape
    if min(nx, ny) < 2 or max(nx, ny) > 4096:
        raise ValueError(f'case requires a native grid within 2..4096 pixels per axis, got {nx}x{ny}')
    ci, cj = (nx - 1) // 2, (ny - 1) // 2  # frame.rs::wrf_center_anchor
    projection = ModelProjection(np, params)
    ref_u, ref_v = projection.forward(float(lat_south[cj, ci]), float(lon_south[cj, ci]))
    dx, dy = params['dx'], params['dy']
    if params['map_proj'] == 6:
        dx = float(np.float32(lon_south[cj, ci + 1] - lon_south[cj, ci]))
        dy = float(np.float32(lat_south[cj + 1, ci] - lat_south[cj, ci]))
    if not all(math.isfinite(value) and value > 0 for value in (dx, dy)):
        raise ValueError('source grid increments must be positive finite values')
    # Verify stored coordinates against the engine projection, before its 0.01-cell inset.
    jj, ii = np.mgrid[0:ny, 0:nx]
    u, v = projection.forward(lat_south.astype(float), lon_south.astype(float))
    worst = float(np.max(np.maximum(np.abs((u - ref_u) / dx + ci - ii), np.abs((v - ref_v) / dy + cj - jj))))
    if not math.isfinite(worst) or worst > SOURCE_GRID_TOLERANCE_CELLS:
        raise ValueError(f'stored source grid differs from engine geometry by {worst:.6f} cells (>0.05)')
    # camera.rs::build_map_raster uses the same inset for all in-domain edges.
    gi = np.clip(np.arange(nx, dtype=float), BOUNDARY_INSET_CELLS, nx - 1 - BOUNDARY_INSET_CELLS)
    gj = np.clip(np.arange(ny - 1, -1, -1, dtype=float), BOUNDARY_INSET_CELLS, ny - 1 - BOUNDARY_INSET_CELLS)
    uu, vv = np.meshgrid(ref_u + (gi - ci) * dx, ref_v + (gj - cj) * dy)
    lat, lon = projection.inverse(uu, vv)
    arrays = {'lat': lat.astype(np.float32), 'lon': lon.astype(np.float32)}
    if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
        raise ValueError('model-grid inverse produced non-finite ground points')
    if land_south is not None:
        if land_south.shape != lat.shape:
            raise ValueError('land mask shape differs from source grid')
        arrays['landmask'] = np.asarray(land_south[::-1], np.float32)
    metadata = {'width': nx, 'height': ny, 'rows': 'north-first', 'params': params,
                'anchor_i': ci, 'anchor_j': cj, 'anchor_lat': float(lat_south[cj, ci]),
                'anchor_lon': float(lon_south[cj, ci]), 'dx': float(dx), 'dy': float(dy),
                'model_earth_radius_m': EARTH_RADIUS_M, 'boundary_inset_cells': BOUNDARY_INSET_CELLS,
                'stored_grid_max_offset_cells': worst, 'method': 'frame.rs integer-center anchor and camera.rs native map sampling; float32 lat/lon'}
    return arrays, metadata, projection


def prepare_input(args, target_path):
    import numpy as np
    target = parse_time(args.time)
    with args.input.open('rb') as source:
        is_grib = source.read(4) == b'GRIB'
    if is_grib:
        import eccodes
        fetch = import_script('simsat_case_grid_fetch', REPO / 'scripts/fetch-goes-abi-reference.py')
        with args.input.open('rb') as stream:
            message = eccodes.codes_grib_new_from_file(stream)
            if message is None:
                raise ValueError('input has no GRIB messages')
            try:
                if int(eccodes.codes_get(message, 'gridDefinitionTemplateNumber')) != 30:
                    raise ValueError('case runner currently supports Lambert HRRR GRIB template 3.30 only')
                valid = datetime.strptime(f"{int(eccodes.codes_get(message, 'validityDate')):08d}{int(eccodes.codes_get(message, 'validityTime')):04d}", '%Y%m%d%H%M').replace(tzinfo=timezone.utc)
                if valid != target:
                    raise ValueError(f'GRIB valid time {iso_time(valid)} != requested {iso_time(target)}')
                sphere = {0: 6_367_470.0, 6: 6_371_229.0}.get(int(eccodes.codes_get(message, 'shapeOfTheEarth')))
                if sphere is None:
                    raise ValueError('unsupported GRIB earth shape; matches ingest_grib.rs spherical-only contract')
                params = {'map_proj': 1, 'truelat1': float(eccodes.codes_get(message, 'Latin1InDegrees')),
                          'truelat2': float(eccodes.codes_get(message, 'Latin2InDegrees')),
                          'stand_lon': float(eccodes.codes_get(message, 'LoVInDegrees')),
                          'cen_lat': 0, 'dx': float(eccodes.codes_get(message, 'DxInMetres')) * EARTH_RADIUS_M / sphere,
                          'dy': float(eccodes.codes_get(message, 'DyInMetres')) * EARTH_RADIUS_M / sphere}
            finally:
                eccodes.codes_release(message)
        lat_north, lon_north, land_north = fetch.grid_from_grib(np, eccodes, args.input)
        lat, lon = lat_north[::-1], lon_north[::-1]
        land = land_north[::-1] if land_north is not None else None
        params['cen_lat'] = float(lat[(lat.shape[0] - 1) // 2, (lat.shape[1] - 1) // 2])
        timestep, kind, source_attrs = 0, 'native-grib2', {'earth_radius_m': sphere}
        if args.timestep not in (None, 0):
            raise ValueError('GRIB input accepts timestep 0 only')
    else:
        from netCDF4 import Dataset
        with Dataset(args.input) as dataset:
            dataset.set_auto_mask(False)
            if 'Times' not in dataset.variables:
                raise ValueError('WRF input requires Times; filename time is not enough')
            times = [row.tobytes().decode('ascii').strip('\x00 ') for row in dataset['Times'][:]]
            timestep = select_wrf_time(times, target, args.timestep)
            def plane(name):
                variable = dataset[name]
                return np.asarray(variable[timestep] if variable.ndim == 3 else variable[:])
            lat, lon = plane('XLAT'), plane('XLONG')
            land = plane('LANDMASK') if 'LANDMASK' in dataset.variables else None
            attr = lambda name, fallback: float(getattr(dataset, name, fallback))
            params = {'map_proj': int(attr('MAP_PROJ', 1)), 'truelat1': attr('TRUELAT1', 30),
                      'truelat2': attr('TRUELAT2', attr('TRUELAT1', 60)),
                      'stand_lon': attr('STAND_LON', attr('CEN_LON', 0)),
                      'cen_lat': attr('CEN_LAT', 0), 'dx': attr('DX', float('nan')), 'dy': attr('DY', float('nan'))}
            source_attrs = {name: str(dataset.getncattr(name)) for name in dataset.ncattrs()
                            if name.upper() in ('TITLE', 'SOURCE', 'HISTORY', 'SIMSAT_SOURCE', 'SIMSAT_PROVENANCE')}
        kind = 'wrf-netcdf'
    arrays, geometry, projection = model_grid(np, lat, lon, land, params)
    np.savez_compressed(target_path, **arrays)
    return {'kind': kind, 'valid_time': iso_time(target), 'timestep': timestep,
            'input_provenance': args.input_provenance or kind,
            'source_attributes': source_attrs, 'geometry': geometry}, arrays, projection


def reference_identity(manifest):
    """Cross-check the platform and infer old sector records only from MCMIP."""
    bucket = manifest.get('bucket', '')
    expected_platform = {'noaa-goes19': 'G19', 'noaa-goes18': 'G18'}.get(bucket)
    if expected_platform is None or manifest.get('platform') != expected_platform:
        raise ValueError('reference platform disagrees with its satellite bucket')
    sectors = set()
    has_mcmip = False
    for item in manifest.get('objects', []):
        for field in ('key', 'file'):
            platforms = set(re.findall(r'_(G[0-9]{2})_', str(item.get(field, ''))))
            if platforms and platforms != {expected_platform}:
                raise ValueError('reference object platform disagrees with its satellite bucket')
        for field in ('product', 'key'):
            for code in re.findall(r'ABI-L2-MCMIP(M1|M2|C|F|M)(?=[/_.-]|$)', str(item.get(field, ''))):
                has_mcmip = True
                if code != 'M':  # The shared Meso prefix alone cannot identify M1 versus M2.
                    sectors.add({'C': 'conus', 'F': 'full-disk', 'M1': 'meso1', 'M2': 'meso2'}[code])
    if len(sectors) > 1:
        raise ValueError('reference MCMIP objects have ambiguous sector identities')
    declared = manifest.get('sector')
    inferred = next(iter(sectors), None)
    if declared is not None:
        if declared not in ('conus', 'full-disk', 'meso1', 'meso2') or (inferred and inferred != declared):
            raise ValueError('reference declared sector disagrees with MCMIP objects')
        return declared, 'source-manifest sector'
    if not has_mcmip or inferred is None:
        raise ValueError('legacy reference sector is ambiguous; no unique MCMIP C/F/M1/M2 product or key')
    return inferred, 'inferred from legacy selected MCMIP product/key'


def verify_reference(args, reference, target_arrays, projection, geometry):
    import numpy as np
    score = helpers()
    manifest_path, companion_path = reference.parent / 'source-manifest.json', reference.with_suffix('.json')
    manifest, companion = score.read_json(manifest_path), score.read_json(companion_path)
    manifest_hash = score.sha256(manifest_path)
    if companion.get('source_manifest') != manifest_path.name or companion.get('source_manifest_sha256') != manifest_hash:
        raise ValueError('aligned companion source-manifest name/hash binding does not match')
    reference_sector, sector_source = reference_identity(manifest)
    if parse_time(manifest['target_time']) != parse_time(args.time):
        raise ValueError('reference target_time does not match requested case time')
    if manifest.get('bucket') != 'noaa-' + args.satellite or reference_sector != args.sector:
        raise ValueError('reference satellite/sector does not match requested case')
    expected_hash = companion['alignment']['sha256']
    if score.sha256(reference) != expected_hash:
        raise ValueError('aligned reference hash differs from its immutable companion metadata')
    with np.load(reference, allow_pickle=False) as loaded:
        if 'lat' not in loaded or 'lon' not in loaded:
            raise ValueError('reused aligned reference needs lat/lon arrays for mesh verification')
        lat, lon = loaded['lat'], loaded['lon']
        if lat.shape != target_arrays['lat'].shape or lon.shape != lat.shape:
            raise ValueError('aligned reference grid shape does not match rendered native model grid')
        u, v = projection.forward(lat, lon)
        expected_u, expected_v = projection.forward(target_arrays['lat'], target_arrays['lon'])
        offset = float(np.max(np.maximum(np.abs(u - expected_u) / geometry['dx'], np.abs(v - expected_v) / geometry['dy'])))
        if not math.isfinite(offset) or offset > REFERENCE_GRID_TOLERANCE_CELLS:
            raise ValueError(f'aligned reference mesh differs by {offset:.6f} grid cells (>0.02)')
    return {'path': str(reference), 'sha256': expected_hash, 'source_manifest': str(manifest_path),
            'manifest_sha256': manifest_hash, 'companion_sha256': score.sha256(companion_path), 'grid_max_offset_cells': offset,
            'target_time': manifest['target_time'], 'sector': reference_sector, 'sector_source': sector_source, 'platform': manifest['platform']}


def append_scoreboard(path, case_id, metadata, scores):
    """Append a completed case once; conflicting repeat measurements are errors."""
    score_hash = hashlib.sha256(json.dumps(scores, sort_keys=True, allow_nan=False).encode()).hexdigest()
    marker = f'<!-- simsat-case:{case_id} scores:{score_hash} -->'
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + '.lock')
    acquired_lock = lock.open('x', encoding='utf-8')
    try:
        with acquired_lock:
            content = path.read_text(encoding='utf-8') if path.exists() else ('# SimSat case scoreboard\n\nGray RGB values are diagnostic luminance, not independent ABI reflectance bands.\n\n' + CLOUD_NOTE + '\n')
            if f'<!-- simsat-case:{case_id} ' in content:
                if marker not in content:
                    raise ValueError('duplicate case identity has different scores; retain both with distinct renderer/input provenance')
                return 'already-recorded'
            rows = [marker, f'## {metadata["valid_time"]} / {metadata["label"]}', '',
                    f'Input kind: {metadata["input_kind"]}. {metadata["input_provenance"]}',
                    f'Case ID: `{case_id}`. Output: `{metadata["output"]}`. Satellite/sector: {metadata["satellite"]}/{metadata["sector"]}.', '',
                    '| Product / sampling | Clear bias | Both-cloudy bias | Observed-cloudy bias | All-valid bias | Correlation |',
                    '|---|---:|---:|---:|---:|---:|']
            for product, values in scores.items():
                label = {'ir': 'IR C13 K / topdown', 'vis': 'Display gray RGB / satellite model-grid',
                         'vis_sensor_fast_gray': 'Sensor gray RGB / satellite model-grid'}[product]
                fmt = lambda key: f'{values[key]:+.6f}' if isinstance(values.get(key), (int, float)) else 'n/a'
                rows.append(f'| {label} | {fmt("clear_bias")} | {fmt("both_cloudy_bias")} | {fmt("cloudy_bias")} | {fmt("bias")} | {fmt("corr")} |')
            if metadata.get('ir_response_note'):
                rows += ['', metadata['ir_response_note']]
            if metadata['night_ir']:
                rows += ['', 'Night IR mode: visible rendering and scoring deliberately skipped.']
            rows += ['', f'Provenance: `{metadata["output"]}/run.json` (input/tool/binary hashes, commands, reference scans and stage timing).', '']
            temp = path.with_suffix(path.suffix + '.tmp')
            temp.write_text(content.rstrip() + '\n\n' + '\n'.join(rows), encoding='utf-8')
            temp.replace(path)
            return 'appended'
    finally:
        lock.unlink()


def update_scoreboard(path, case_id, metadata, scores):
    return append_scoreboard(path, case_id, metadata, scores)


def run(args, runner=subprocess.run, input_preparer=prepare_input, reference_verifier=verify_reference):
    score = helpers()
    if args.out.exists():
        raise FileExistsError(f'output exists: {args.out}; use a fresh directory')
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    parse_time(args.time)
    if args.threads < 1:
        raise ValueError('--threads must be positive')
    if args.satellite == 'goes18' and not args.allow_goes18_fm4_approximation:
        raise ValueError('GOES18 has no matched FM3 SRF in this runner; --allow-goes18-fm4-approximation explicitly accepts the GOES19 FM4 response approximation')
    ir_response_note = ('GOES19 official FM4 C13 response' if args.satellite == 'goes19' else
                        'APPROXIMATION: GOES18 observation scored with GOES19 FM4 C13 response; not a matched GOES18 spectral operator')
    child_environment = dict(os.environ, RAYON_NUM_THREADS=str(args.threads))
    bins = {'ir': score.binary(args.bin, 'simsat-render-ir')}
    if not args.night_ir:
        bins['visible'] = score.binary(args.bin, 'simsat-render-frame')
    scripts = {'fetch': REPO / 'scripts/fetch-goes-abi-reference.py', 'validate': REPO / 'scripts/simsat-validate-goes.py',
               'runner': Path(__file__).resolve(), 'score_extraction': REPO / 'scripts/simsat-case-score.py'}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'cache').mkdir()
    (args.out / 'logs').mkdir()
    scores = {}
    manifest = {'schema_version': 1, 'status': 'running', 'started_utc': score.utc_now(),
                'time': iso_time(parse_time(args.time)), 'satellite': args.satellite, 'sector': args.sector,
                'night_ir': args.night_ir, 'visible_cloud_regime_limitation': CLOUD_NOTE,
                'ir_response_note': ir_response_note, 'environment': {'RAYON_NUM_THREADS': str(args.threads)},
                'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                'python_version': sys.version,
                'input': {'path': str(args.input), 'sha256': score.sha256(args.input)},
                'binaries': {name: {'path': str(path), 'sha256': score.sha256(path)} for name, path in bins.items()},
                'scripts': {name: {'path': str(path), 'sha256': score.sha256(path)} for name, path in scripts.items()},
                'stages': []}
    start = time.monotonic()
    def checkpoint():
        score.write_json(args.out / 'scores.json', scores)
        score.write_json(args.out / 'run.json', manifest)
    def stage(name, command=None, callback=None):
        log = args.out / 'logs' / f'{len(manifest["stages"])+1:02d}-{name}.log'
        record = {'name': name, 'log': str(log), 'started_utc': score.utc_now(), 'status': 'running'}
        if command:
            record['command'] = [str(part) for part in command]
        manifest['stages'].append(record)
        checkpoint()
        print(f'{name}: {log}', flush=True)
        stage_start = time.monotonic()
        try:
            with log.open('x', encoding='utf-8') as output:
                if command:
                    output.write(json.dumps(record['command']) + '\n')
                    output.flush()
                    result = runner(record['command'], stdout=output, stderr=subprocess.STDOUT, cwd=REPO, check=False, env=child_environment)
                    record['returncode'] = result.returncode
                    if result.returncode:
                        raise RuntimeError(f'{name} failed with exit {result.returncode}; see {log}')
                else:
                    result = callback()
                    output.write('complete\n')
            record['status'] = 'complete'
            return result
        except BaseException as error:
            record['status'], record['error'] = 'failed', str(error)
            with log.open('a', encoding='utf-8') as output:
                output.write(str(error) + '\n')
            raise
        finally:
            record['elapsed_seconds'] = round(time.monotonic() - stage_start, 6)
            checkpoint()
    checkpoint()
    try:
        grid_path = args.out / 'target-grid.npz'
        source, grid, projection = stage('prepare-input-grid', callback=lambda: input_preparer(args, grid_path))
        manifest['source'] = source
        manifest['target_grid'] = {'path': str(grid_path), 'sha256': score.sha256(grid_path)}
        if args.reference:
            reference = args.reference / 'abi-reference-aligned.npz' if args.reference.is_dir() else args.reference
        else:
            destination = args.out / 'reference'
            stage('fetch-align-abi', [sys.executable, '-X', 'utf8', scripts['fetch'], '--time', manifest['time'],
                  '--satellite', args.satellite, '--sector', args.sector, '--target-grid', grid_path,
                  '--output-dir', destination, '--download', '--align', '--preview'])
            reference = destination / 'abi-reference-aligned.npz'
        manifest['reference'] = stage('verify-reference', callback=lambda: reference_verifier(args, reference, grid, projection, source['geometry']))
        mask, ir_plane = args.out / 'cloud-mask.bin', args.out / 'ir13.bin'
        common = [f'input={args.input}', f'timestep={source["timestep"]}', f'cache={args.out / "cache"}', f'sat={SATELLITES[args.satellite]}', 'resolution=native', f'threads={args.threads}']
        stage('render-ir', [bins['ir'], *common, f'out={args.out / "ir13.png"}', f'bt-out={ir_plane}',
              f'cloud-mask-out={mask}', 'view=topdown', 'sensor=goes-r-abi-band13-fm4', 'enhancement=cimss'])
        def validate(group, product, plane, kind):
            destination = args.out / f'validate-{group}'
            stage(f'validate-{group}', [sys.executable, '-X', 'utf8', scripts['validate'], '--product', product,
                  '--synthetic', plane, '--reference', reference, '--synthetic-cloud-mask', mask,
                  '--input-kind', kind, '--output-dir', destination])
            scores[group] = score.extract_scores(score.read_json(destination / 'validation.json'), product)
            checkpoint()
        validate('ir', 'abi-band13', ir_plane, 'f32le-scalar')
        if not args.night_ir:
            for intent, group, kind in [('display', 'vis', 'f32le-rgb'), ('sensor-fast-gray', 'vis_sensor_fast_gray', 'f32le-rgb-unclipped')]:
                plane = args.out / f'{group}-rho.bin'
                stage(f'render-{intent}', [bins['visible'], *common, f'out={args.out / (group + ".png")}',
                      f'rgb-reflectance-out={plane}', f'intent={intent}', 'view=geo', 'raster=model-grid', 'geo-navigation=goes-r-abi'])
                validate(group, 'visible', plane, kind)
        identity = {'input_sha256': manifest['input']['sha256'], 'time': manifest['time'], 'satellite': args.satellite,
                    'sector': args.sector, 'night_ir': args.night_ir, 'ir_response_note': ir_response_note, 'binaries': manifest['binaries'],
                    'validator_sha256': manifest['scripts']['validate']['sha256'], 'reference_sha256': manifest['reference']['sha256']}
        case_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        manifest['case_id'] = case_id
        manifest['scoreboard_update'] = stage('append-scoreboard', callback=lambda: update_scoreboard(args.scoreboard, case_id,
            {'valid_time': manifest['time'], 'label': args.label or args.input.stem,
             'input_kind': source['kind'], 'input_provenance': source['input_provenance'],
             'output': str(args.out), 'satellite': args.satellite, 'sector': args.sector, 'night_ir': args.night_ir,
             'ir_response_note': ir_response_note}, scores))
        manifest['status'] = 'complete'
    except BaseException as error:
        manifest['status'], manifest['error'] = 'failed', f'{type(error).__name__}: {error}'
        raise
    finally:
        manifest['finished_utc'] = score.utc_now()
        manifest['elapsed_seconds'] = round(time.monotonic() - start, 6)
        checkpoint()
    print(f'complete: {args.out / "scores.json"}; scoreboard {manifest["scoreboard_update"]}', flush=True)
    return scores


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument('--input', type=Path, required=True)
    result.add_argument('--time', required=True, help='exact model valid time, RFC3339 with Z or offset')
    result.add_argument('--out', type=Path, required=True, help='fresh output directory; existing paths are rejected')
    result.add_argument('--bin', type=Path, required=True, help='directory containing current release renderers')
    result.add_argument('--sector', choices=('conus', 'full-disk', 'meso1', 'meso2'), default='conus')
    result.add_argument('--satellite', choices=tuple(SATELLITES), default='goes19')
    result.add_argument('--threads', type=int, default=6, help='renderer Rayon worker limit; default 6, also set in child environment')
    result.add_argument('--allow-goes18-fm4-approximation', action='store_true', help='explicitly accept GOES19 FM4 IR response against GOES18 observations; recorded as an approximation')
    result.add_argument('--night-ir', action='store_true', help='skip visible render/validation; keep official SRF IR and cloud regimes')
    result.add_argument('--reference', type=Path, help='reuse existing aligned NPZ or bundle after provenance and mesh verification')
    result.add_argument('--timestep', type=int, help='WRF Times index; otherwise select exact unique matching time')
    result.add_argument('--scoreboard', type=Path, default=REPO / 'notes/nextgen/case-scoreboard.md')
    result.add_argument('--label', help='human-readable case name')
    result.add_argument('--input-provenance', help='state preparation/source, e.g. HRRR-prepared WRF; never implies native GRIB verification')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ('input', 'out', 'bin', 'scoreboard', 'reference'):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    try:
        run(args)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        print(f'simsat-case-run: {type(error).__name__}: {error}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('simsat-case-run: interrupted; completed stage logs and score checkpoints retained', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
