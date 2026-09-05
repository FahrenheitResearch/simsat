#!/usr/bin/env python3
"""Conservative HSRS solar weights on the official ABI FM4 response grids.

For a piecewise-linear normalized transfer f(lambda)=L_lambda/E_lambda,
construct positive nodal weights w_i=int R(lambda) E(lambda) phi_i(lambda) dlam.
The source E retains every native 0.001-nm sample. Merged source/response knots
split the integral into intervals on which E, R and phi are linear; two-point
Gauss integration is exact for their cubic product (up to floating precision).
There is no sampling of narrow solar lines on the much coarser response grid.

These weights integrate smooth, wavelength-resolved transfer, NOT existing RGB.
Unresolved atmospheric gas lines still require their own spectral integration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SOURCE_SHA256 = 'ea6d0e219925a69607a485451ed2a8dcaf8b2f583b646b04a94122feef4d697f'
SOURCE_URL = 'https://lasp.colorado.edu/lisird/latis/dap/tsis1_hsrs.csv?wavelength>=400&wavelength<=1000'
SOURCE_PAPER = 'https://doi.org/10.1029/2020GL091709'


def solar_weights(solar_nm, solar_per_nm, response_um, response):
    """Return W/m^2 weights for linear transfer samples at response_um knots."""
    arrays = [np.asarray(a, dtype=np.float64) for a in (solar_nm, solar_per_nm, response_um, response)]
    x, e, t, r = arrays
    if any(a.ndim != 1 or a.size < 2 or not np.isfinite(a).all() for a in arrays):
        raise ValueError('finite one-dimensional spectra with at least two samples required')
    if x.shape != e.shape or t.shape != r.shape or np.any(np.diff(x) <= 0) or np.any(np.diff(t) <= 0):
        raise ValueError('spectral coordinates must ascend strictly and match their values')
    if np.any(e < 0) or np.any(r < 0):
        raise ValueError('solar irradiance and response must be nonnegative')
    t_nm = t * 1000.0
    if x[0] > t_nm[0] or x[-1] < t_nm[-1]:
        raise ValueError('solar source does not cover the complete response')
    knots = np.unique(np.concatenate((x[(x > t_nm[0]) & (x < t_nm[-1])], t_nm)))
    mid = (knots[1:] + knots[:-1]) * 0.5
    half = np.diff(knots) * 0.5
    weights = np.zeros_like(t)
    # Two Gauss points integrate the cubic E * R * transfer hat exactly.
    for sign in (-1.0, 1.0):
        q = mid + sign * half / np.sqrt(3.0)
        k = np.searchsorted(t_nm, q, side='right') - 1
        k = np.minimum(k, len(t_nm)-2)
        f = (q-t_nm[k])/(t_nm[k+1]-t_nm[k])
        measure = half * np.interp(q, x, e) * ((1.0-f)*r[k] + f*r[k+1])
        np.add.at(weights, k, measure*(1.0-f))
        np.add.at(weights, k+1, measure*f)
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('invalid solar response integral')
    return weights


def prepare(solar_csv, output_dir):
    raw = solar_csv.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError('solar source hash differs from the reviewed LASP retrieval; review provenance before updating')
    data = np.loadtxt(solar_csv, delimiter=',', skiprows=1)
    if data.shape != (600001, 3) or data[0, 0] != 400.0 or data[-1, 0] != 1000.0:
        raise ValueError('unexpected solar source grid')
    output_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    manifest = {'schema':'simsat-abi-solar-weights-v1', 'source_url':SOURCE_URL,
        'source_sha256':digest, 'source_bytes':len(raw), 'source_paper':SOURCE_PAPER,
        'source_retrieved_utc':'2026-09-05T05:05:39.822157+00:00',
        'source_reference':'TSIS-1 HSRS reference solar spectrum at 1 AU; not observation-date irradiance',
        'integration':'two-point Gauss per merged E and R interval; piecewise-linear L_lambda/E_lambda at exact SRF nodes',
        'units':{'wavelength':'um','weights':'W m^-2 at 1 AU'},
        'uncertainty':'source spectral uncertainty column retained in original retrieval; covariance unavailable, not propagated to weights',
        'limitations':['scalar unpolarized transfer only','transfer must resolve atmospheric absorption structure separately',
                       'response includes exactly the tabulated endpoints; unprovided out-of-band tails are not inferred'],
        'bands':[]}
    for band in (1,2,3):
        srf_path=repo/f'crates/simsat/assets/abi_srf/GOES-R_ABI_FM4_SRF_CWG_ch{band}.txt'
        srf_raw=srf_path.read_bytes()
        srf=np.loadtxt(srf_path)
        weights=solar_weights(data[:,0],data[:,1],srf[:,0],srf[:,2])
        dest=output_dir/f'abi-fm4-c{band:02d}-hsrs-weights.txt'
        if dest.exists(): raise ValueError(f'refusing to overwrite {dest}')
        with dest.open('w',encoding='ascii',newline='\n') as f:
            f.write('# wavelength_um solar_response_weight_W_m2_at_1au\n')
            for wavelength,weight in zip(srf[:,0],weights): f.write(f'{wavelength:.9f} {weight:.17e}\n')
        area=float(np.trapezoid(srf[:,2],srf[:,0]))
        manifest['bands'].append({'band':band,'file':dest.name,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
            'srf_sha256':hashlib.sha256(srf_raw).hexdigest(),'nodes':len(weights),
            'response_integral_um':area,'solar_response_integral_w_m2':float(weights.sum()),
            'mean_solar_irradiance_w_m2_um':float(weights.sum()/area)})
    dest=output_dir/'manifest.json'
    if dest.exists(): raise ValueError(f'refusing to overwrite {dest}')
    dest.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--solar-csv',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(prepare(args.solar_csv,args.output_dir),indent=2))

if __name__=='__main__': main()
