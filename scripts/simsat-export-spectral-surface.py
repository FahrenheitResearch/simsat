#!/usr/bin/env python3
"""Export an aligned HAMSTER spectrum cube to the SimSat float input contract.

The output retains all wavelengths, coordinates and validity, plus provenance.
This does not turn black-sky albedo into a BRDF or into ABI TOA reflectance.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np


def export(source, provenance, output):
    source, provenance, output = map(Path, (source, provenance, output))
    if output.exists(): raise ValueError('Output directory must be new')
    report=json.loads(provenance.read_text(encoding='utf-8'))
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if report['output_sha256']!=digest: raise ValueError('Source NPZ does not match its provenance hash')
    with np.load(source,allow_pickle=False) as f:
        lat=np.asarray(f['latitude'],dtype='<f8');lon=np.asarray(f['longitude'],dtype='<f8')
        wl=np.asarray(f['wavelength_um'],dtype=float);albedo=np.asarray(f['black_sky_albedo'],dtype='<f4')
        valid=np.asarray(f['valid']);land=np.asarray(f['land_mask'])
    if lat.ndim!=2 or lon.shape!=lat.shape or albedo.shape!=(*lat.shape,len(wl)):raise ValueError('Shape mismatch')
    if not np.isfinite(lat).all() or not np.isfinite(lon).all() or np.any(abs(lat)>90) or np.any(abs(lon)>180):raise ValueError('Invalid coordinates')
    if valid.shape!=lat.shape or land.shape!=lat.shape or not np.isin(valid,[0,1]).all() or not np.isin(land,[0,1]).all():raise ValueError('Invalid masks')
    if np.any(valid>land) or np.any(land>valid):raise ValueError('Every land cell must have a valid spectrum; ocean must remain unavailable')
    if not np.isfinite(wl).all() or not np.all(np.diff(wl)>0) or wl[0]>.4 or wl[-1]<1.:raise ValueError('Require increasing wavelengths covering 0.4-1.0 um')
    good=albedo[valid==1]
    if not np.isfinite(good).all() or np.any((good<0)|(good>1)):raise ValueError('Invalid albedo')
    # Canonical no-data bytes, separate from the valid/land masks.
    albedo[valid==0]=np.nan
    output.mkdir(parents=True)
    arrays={'coordinates.bin':np.stack((lat,lon),axis=-1).astype('<f8'),'albedo.bin':albedo,'valid.bin':valid.astype('uint8'),'land-mask.bin':land.astype('uint8')}
    files={}
    for name,a in arrays.items():
        path=output/name;a.tofile(path)
        files[name]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    header={'schema_version':1,'quantity':'climatological_black_sky_surface_albedo','nx':lat.shape[1],'ny':lat.shape[0],'wavelength_um':wl.tolist(),'climatology_doy':report['climatology_doy'],'coordinate_layout':'latitude_longitude_interleaved_f64le','albedo_layout':'row_column_wavelength_f32le','row_order':'preserved_from_aligned_input','files':files,'source_npz_sha256':digest,'provenance':report,'limitations':report['limitations']}
    (output/'surface.json').write_text(json.dumps(header,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return header


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--provenance',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();r=export(a.input,a.provenance,a.output_dir);print(json.dumps({'shape':[r['ny'],r['nx'],len(r['wavelength_um'])],'manifest':str(a.output_dir/'surface.json')}))
if __name__=='__main__':main()
