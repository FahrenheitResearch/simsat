#!/usr/bin/env python3
"""Compare SimSat homogeneous-slab source output with an external cdisort probe.

Build instructions and limitations: notes/nextgen/CLOUD-SLAB-REFERENCE.md.
The external solver is not part of the renderer or Python dependencies.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulated', required=True, type=Path)
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument('--reference-exe', type=Path)
    reference.add_argument('--reference-csv', type=Path)
    parser.add_argument('--reference-input', type=Path, help='Required request transcript when replaying a reference CSV')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    simulated = json.loads(args.simulated.read_text(encoding='utf-8-sig'))
    if simulated['schema'] != 'simsat-cloud-slab-source-audit-v1':
        raise ValueError('unsupported simulated source schema')
    cases = simulated['cases']
    ids = [case['id'] for case in cases]
    if not cases or len(set(ids)) != len(ids):
        raise ValueError('case ids must be nonempty and unique')
    requests = []
    for case in cases:
        r = case['reference_request']
        if r['ssa'] != 1 or r['albedo'] != 0 or r['kind'] != 1:
            raise ValueError('this benchmark requires a conservative dual-HG slab over black ground')
        for streams in [32, 68]:
            values = [case['id'], r['tau'], r['ssa'], r['kind'], r['g1'], r['g2'],
                      r['weight'], r['gamma'], r['mu0'], r['muv'],
                      r['relative_azimuth_deg'], r['albedo'], streams]
            if not all(math.isfinite(v) for v in values):
                raise ValueError('non-finite reference request')
            requests.append(' '.join(format(v, '.17g') for v in values))
    request = '\n'.join(requests) + '\n'
    if args.reference_csv:
        if not args.reference_input:
            raise ValueError('--reference-input is required when replaying a reference CSV')
        recorded = [[float(v) for v in line.split()] for line in args.reference_input.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        expected_request = [[float(v) for v in line.split()] for line in requests]
        if recorded != expected_request:
            raise ValueError('reference input geometry/physics does not match this simulated audit')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    (args.output_dir / 'reference-input.txt').write_text(request, encoding='ascii')
    if args.reference_exe:
        result = subprocess.run([str(args.reference_exe.resolve())], input=request,
                                text=True, capture_output=True)
        raw = result.stdout
        (args.output_dir / 'reference-errors.log').write_text(result.stderr, encoding='utf-8')
        (args.output_dir / 'reference.csv').write_text(raw, encoding='utf-8', newline='\n')
        result.check_returncode()
        provenance = {'exe_sha256': digest(args.reference_exe)}
    else:
        raw = args.reference_csv.read_text(encoding='utf-8-sig')
        provenance = {'csv_sha256': digest(args.reference_csv), 'request_sha256': digest(args.reference_input)}
    rows = list(csv.DictReader(io.StringIO(raw)))
    expected = {(i, streams) for i in ids for streams in [32, 68]}
    lookup = {}
    for row in rows:
        key = (int(row['id']), int(row['nstr']))
        if key in lookup:
            raise ValueError(f'duplicate reference row {key}')
        values = {k: float(v) for k, v in row.items()}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError(f'non-finite reference row {key}')
        lookup[key] = values
    if set(lookup) != expected:
        raise ValueError('reference rows do not exactly match requested cases/streams')
    joined = []
    modes = set(cases[0]['rho_f'])
    for case in cases:
        if set(case['rho_f']) != modes or not all(math.isfinite(v) for v in case['rho_f'].values()):
            raise ValueError('invalid simulated mode outputs')
        low, high = lookup[case['id'], 32], lookup[case['id'], 68]
        difference = abs(low['rho_f'] - high['rho_f'])
        joined.append({**case, 'reference_32': low, 'reference_68': high,
                       'reference_converged_32_to_68': difference <= 2e-5 + 1e-3 * abs(high['rho_f']),
                       'rho_f_32_to_68_difference': difference})
    groups = {}
    for name, condition in [('all', lambda c: True),
                            ('in_lut_domain', lambda c: c['within_lut_tau_mu_domain']),
                            ('outside_lut_domain', lambda c: not c['within_lut_tau_mu_domain'])]:
        subset = [c for c in joined if condition(c) and c['reference_converged_32_to_68']]
        scores = {}
        for mode in sorted(modes):
            errors = [c['rho_f'][mode] - c['reference_68']['rho_f'] for c in subset]
            if errors:
                scores[mode] = {'bias': sum(errors) / len(errors),
                                'mae': sum(map(abs, errors)) / len(errors),
                                'rmse': math.sqrt(sum(v*v for v in errors) / len(errors))}
        groups[name + '_converged'] = {'count': len(subset), 'scores': scores}
    summary = {'schema': 'simsat-cloud-slab-reference-comparison-v1',
               'reference': 'external cdisort 2.1.3 scalar plane-parallel one-layer probe',
               'simulated_sha256': digest(args.simulated), 'reference_provenance': provenance,
               'seconds': time.monotonic() - start, 'rows': len(rows),
               'radiance_convergence_absolute_tolerance': 2e-5,
               'radiance_convergence_relative_tolerance': 1e-3,
               'unconverged_ids': [c['id'] for c in joined if not c['reference_converged_32_to_68']],
               'max_conservative_flux_residual_68': max(abs(c['reference_68']['atmospheric_absorptance']) for c in joined),
               'groups': groups, 'limitations': simulated['limitations']}
    for name, data in [('summary.json', summary), ('comparison.json', {**summary, 'cases': joined})]:
        (args.output_dir / name).write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    if summary['unconverged_ids']:
        raise SystemExit('unconverged references excluded from scores; refine before using those cases')


if __name__ == '__main__':
    main()
