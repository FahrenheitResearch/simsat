#!/usr/bin/env python3
"""Export measured MODIS 1/4/3 RGB for an explicitly approximate display input."""
from pathlib import Path
import argparse, datetime, hashlib, json
import numpy as np

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def export(source_dir, output_dir, frame_date, quality):
    provenance_path = source_dir / 'provenance.json'
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    source = source_dir / 'aligned-nbar.npz'
    if provenance['quantity'] != 'MODIS_7_band_nadir_BRDF_adjusted_surface_reflectance' or sha(source) != provenance['output_sha256']:
        raise ValueError('NBAR source quantity or checksum mismatch')
    source_date = datetime.date.fromisoformat(provenance['date'])
    if (source_date.month, source_date.day) != (frame_date.month, frame_date.day):
        raise ValueError('Seasonal analogue must use the same calendar month/day')
    if quality not in ['full-only', 'full-and-magnitude']:
        raise ValueError('Explicit quality policy required')
    with np.load(source, allow_pickle=False) as f:
        lat=f['latitude'];lon=f['longitude'];land=f['land_mask']
        values=f['nbar'];qa=f['mandatory_quality']
        if lat.ndim != 2 or lat.size == 0 or lon.shape != lat.shape or land.shape != lat.shape or values.shape != (*lat.shape,7) or qa.shape != values.shape:
            raise ValueError('Invalid NBAR target shape')
        if not np.array_equal(f['band'],np.arange(1,8)) or not np.isin(land,[0,1]).all() or not np.isin(qa,[0,1,255]).all():
            raise ValueError('Invalid band order or quality/land mask')
        if not np.isfinite(lat).all() or not np.isfinite(lon).all() or np.any(abs(lat)>90) or np.any(abs(lon)>180):
            raise ValueError('Invalid NBAR coordinates')
        # Native measured bands, no interpolation to invented monochromatic samples.
        rgb=np.asarray(values[...,[0,3,2]],dtype='<f4')
        rgb_qa=np.asarray(qa[...,[0,3,2]],dtype='u1')
        coordinates=np.stack([lat,lon],axis=-1).astype('<f8')
    physical=np.isfinite(rgb).all(axis=-1)&((rgb>=0)&(rgb<=1)).all(axis=-1)
    full=(rgb_qa==0).all(axis=-1)&physical&(land==1)
    magnitude=(rgb_qa<=1).all(axis=-1)&physical&(land==1)&~full
    if quality=='full-only':magnitude[:]=False
    accepted=full|magnitude
    if not accepted.any():raise ValueError('No usable NBAR RGB land')
    header=dict(schema_version=1,quantity='modis_nbar_rgb_lambertian_display_proxy',
        nx=lat.shape[1],ny=lat.shape[0],source_date=source_date.isoformat(),
        frame_date=frame_date.isoformat(),rgb_bands=[1,4,3],quality_policy=quality,
        missing_policy='configured-base-map',
        coordinate_layout='latitude_longitude_interleaved_f64le',
        reflectance_layout='row_column_rgb_f32le',
        files={},source_provenance=provenance,
        source_provenance_sha256=sha(provenance_path),
        counts=dict(full=int(full.sum()),magnitude=int(magnitude.sum()),
                    fallback=int(((land==1)&~accepted).sum())),
        limitations=[
            'Display-only Lambertian proxy of nadir/local-noon directional reflectance.',
            'No arbitrary-angle BRDF correction. Not black-sky albedo or ABI channels.',
            'MODIS bands 1/4/3 used for RGB; Hillaire atmosphere remains broad gray RGB.',
            'Explicit source/frame dates; a different source year is a seasonal analogue, not historical land reconstruction.',
            'Missing, rejected quality, and values outside [0,1] use the configured base map; never clipped or fabricated.',
            'Cloud/terrain shading is computed by the renderer. Existing display gains remain separately configurable.'
        ])
    output_dir.mkdir(parents=True,exist_ok=False)
    for name,data in [('coordinates.bin',coordinates),('nbar-rgb.bin',rgb),
                      ('quality-rgb.bin',rgb_qa),('land-mask.bin',land.astype('u1'))]:
        path=output_dir/name;data.tofile(path)
        header['files'][name]=dict(bytes=path.stat().st_size,sha256=sha(path))
    (output_dir/'surface.json').write_text(json.dumps(header,indent=2)+'\n',encoding='utf-8')
    return header

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--frame-date',type=datetime.date.fromisoformat,required=True,
                   help='Explicit intended simulation date; month/day must match source')
    p.add_argument('--quality',choices=['full-only','full-and-magnitude'],required=True)
    p.add_argument('--missing',choices=['configured-base-map'],required=True)
    a=p.parse_args()
    h=export(a.source_dir,a.output_dir,a.frame_date,a.quality)
    print(json.dumps(dict(manifest=str(a.output_dir/'surface.json'),counts=h['counts'])))
