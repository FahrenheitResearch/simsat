#!/usr/bin/env python3
"""Prepare pinned visible material-index tables without executing vendor code.

Inputs are the official libRadtran 2.0.6 libsrc_f/REFWAT.f and the official
Warren/Brandt IOP_2008_ASCIItable.dat. See assets/material_indices/README.md.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import numpy as np

WATER_SHA256 = 'f80caefb0a926a2e258ca9761ba5f07f8e83867f032b0d033b57a226959d5a5f'
ICE_SHA256 = '891d0e0690cc6c6ed2dfe81291fa20a3af177fa85ef93127c58c5f86d3c370af'


def parse_water(source, count=1261):
    arrays = []
    for name in ['WLTAB', 'REREF', 'IMREF']:
        values = np.full(count, np.nan)
        expression = r'DATA\s*\(' + name + r'\(I\),\s*I\s*=\s*(\d+)\s*,\s*(\d+)\s*\)\s*/(.*?)/'
        for match in re.finditer(expression, source, re.S | re.I):
            begin, end = int(match[1])-1, int(match[2])
            if not 0 <= begin < end <= count:
                raise ValueError('invalid Fortran data range')
            body = ' '.join(line[6:72] for line in match[3].splitlines() if line.strip())
            entries = [float(v.replace('D', 'E').replace('d', 'e'))
                       for v in body.replace('&', '').split(',') if v.strip()]
            if len(entries) != end-begin or not np.isnan(values[begin:end]).all():
                raise ValueError('incomplete or overlapping Fortran data block')
            values[begin:end] = entries
        if not np.isfinite(values).all() or not (values > 0).all():
            raise ValueError('missing/nonpositive/nonfinite material data')
        arrays.append(values)
    return np.column_stack(arrays)


def visible_subset(data):
    if data.ndim != 2 or data.shape[1] != 3 or len(data) < 2:
        raise ValueError('expected three-column material table')
    if not np.isfinite(data).all() or not (data > 0).all() or not (np.diff(data[:, 0]) > 0).all():
        raise ValueError('material data must be finite, positive, and ordered')
    if data[0, 0] > .4 or data[-1, 0] < 1.:
        raise ValueError('material data must bracket 0.4..1.0 um')
    lo = max(int(np.searchsorted(data[:, 0], .4, side='right'))-1, 0)
    hi = min(int(np.searchsorted(data[:, 0], 1., side='left'))+1, len(data))
    return data[lo:hi]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--water-fortran', type=Path, required=True)
    parser.add_argument('--ice-ascii', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    raw_water, raw_ice = args.water_fortran.read_bytes(), args.ice_ascii.read_bytes()
    if hashlib.sha256(raw_water).hexdigest() != WATER_SHA256 or hashlib.sha256(raw_ice).hexdigest() != ICE_SHA256:
        raise ValueError('source hash differs from pinned authoritative material data')
    tables = [('water-segelstein-1981-visible.txt', parse_water(raw_water.decode('utf-8'))),
              ('ice-ih-warren-brandt-2008-visible.txt', np.loadtxt(args.ice_ascii))]
    subsets = [(name, visible_subset(data)) for name, data in tables]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': 'simsat-visible-material-indices-preparation-v1',
                'prepared_utc': datetime.datetime.now(datetime.UTC).isoformat(),
                'water_member_sha256': WATER_SHA256, 'ice_source_sha256': ICE_SHA256,
                'wavelength_contract_um': [.4, 1.],
                'interpolation': 'n_real linear in log wavelength; log positive n_imag linear in log wavelength',
                'source_documentation': 'crates/simsat/assets/material_indices/README.md', 'artifacts': {}}
    for name, subset in subsets:
        path = args.output_dir/name
        with path.open('w', encoding='utf-8', newline='\n') as stream:
            np.savetxt(stream, subset, fmt='%.12g', header='wavelength_um n_real n_imag_positive')
        manifest['artifacts'][name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                       'rows': len(subset), 'support_um': [float(subset[0, 0]), float(subset[-1, 0])]}
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
