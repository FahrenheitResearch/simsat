from pathlib import Path
import argparse, hashlib, json
import numpy as np


def main():
    p=argparse.ArgumentParser(description='Independently verify integrated SURFACE albedos against direct NumPy interpolation at each official solar/SRF response knot.')
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--audit-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    h=json.loads(a.manifest.read_text(encoding='utf-8'))
    src=a.manifest.parent
    for name, rec in h['files'].items():
        assert name in {'coordinates.bin','albedo.bin','valid.bin','land-mask.bin'},name
        data=(src/name).read_bytes()
        assert len(data)==rec['bytes'] and hashlib.sha256(data).hexdigest()==rec['sha256'],name
    n=h['nx']*h['ny']; w=np.asarray(h['wavelength_um'])
    albedo=np.fromfile(src/'albedo.bin',dtype='<f4').reshape(n,len(w))
    valid=np.fromfile(src/'valid.bin',dtype='u1')==1
    selected=np.unique(np.linspace(0,valid.sum()-1,min(193,valid.sum()),dtype=int))
    indices=np.flatnonzero(valid)[selected]
    assets=Path(__file__).resolve().parents[1]/'crates/simsat/assets/solar_hsrs'
    results=[]
    for band in ['c01','c02','c03']:
        nodes=np.loadtxt(assets/f'abi-fm4-{band}-hsrs-weights.txt')
        values=np.fromfile(a.audit_dir/f'surface-albedo-{band}.bin',dtype='<f4')
        assert values.size==n and np.isnan(values[~valid]).all()
        assert np.isfinite(values[valid]).all()
        expected=np.array([np.dot(np.interp(nodes[:,0],w,albedo[i]),nodes[:,1])/nodes[:,1].sum() for i in indices])
        error=float(np.abs(values[indices]-expected).max())
        assert error<3e-8,(band,error)
        results.append(dict(band=band,checked_spectra=len(indices),max_absolute_error=error,water_is_nan=True,land_count=int(valid.sum())))
    a.output.write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(results))

if __name__=='__main__':main()
