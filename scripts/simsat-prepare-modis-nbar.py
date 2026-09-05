#!/usr/bin/env python3
from pathlib import Path
import argparse, datetime, hashlib, json, re
import numpy as np
import requests, rasterio
from rasterio.warp import transform
from rasterio.windows import Window

parser=argparse.ArgumentParser(description='Prepare seven-band MODIS MCD43A4.061 NBAR and separate quality masks on an exact target grid. Not a spectral albedo or satellite renderer.')
parser.add_argument('--grid',type=Path,required=True,help='NPZ with lat, lon and binary land_mask arrays')
parser.add_argument('--items',type=Path,required=True,help='Saved Planetary Computer STAC JSON with items or features')
parser.add_argument('--date',type=datetime.date.fromisoformat,required=True,help='Daily center date, selected from granule IDs')
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--resume',action='store_true',help='Reuse matching, georeferenced source windows in the same output directory')
args=parser.parse_args()
out=args.output_dir;grid_path=args.grid
catalog=json.loads(args.items.read_text(encoding='utf-8'))
stamp=args.date.strftime('%Y%j');by_tile={}
for item in catalog.get('items',catalog.get('features',[])):
    match=re.fullmatch(r'MCD43A4\.A'+stamp+r'\.(h[0-9]{2}v[0-9]{2})\.061\.[0-9]+',item['id'])
    if match and item['id']>by_tile.get(match[1],{}).get('id',''):
        by_tile[match[1]]=item
items=sorted(by_tile.values(),key=lambda item:item['id'])
if not items:raise ValueError('No exact-center-date MCD43A4.061 granules in supplied STAC items')
request=dict(grid_sha256=hashlib.sha256(grid_path.read_bytes()).hexdigest(),date=args.date.isoformat(),items=[item['id'] for item in items],sampling='nearest source pixel')
if out.exists():
    if not args.resume or json.loads((out/'request.json').read_text(encoding='utf-8'))!=request:
        raise ValueError('Existing output requires --resume and an identical grid/date/item request')
else:
    out.mkdir(parents=True)
    (out/'request.json').write_text(json.dumps(request,indent=2)+'\n',encoding='utf-8')
with np.load(grid_path,allow_pickle=False) as f:
    lat=f['lat'];lon=f['lon'];land_mask=f['land_mask']
    if not np.isin(land_mask,[0,1]).all():raise ValueError('Target land mask must be binary')
    land=land_mask==1
if lat.ndim!=2 or lat.size==0 or lon.shape!=lat.shape or land.shape!=lat.shape or not np.isfinite(lat).all() or not np.isfinite(lon).all() or np.any(abs(lat)>90) or np.any(abs(lon)>180):
    raise ValueError('Invalid target coordinates or grid shape')
shape=lat.shape
raw=np.full((*shape,7),32767,dtype='<i2');qa=np.full((*shape,7),255,dtype='u1');coverage=np.zeros(shape,dtype='u1')
records=[];session=requests.Session();tokens={}
from urllib.parse import urlparse
from affine import Affine
mapping_cache={}
def signed_url(url):
    u=urlparse(url)
    if u.scheme!='https' or u.netloc!='modiseuwest.blob.core.windows.net' or not u.path.startswith('/modis-061-cogs/MCD43A4/') or u.query:
        raise ValueError('Expected an unsigned MODIS COG asset from the documented Planetary Computer storage account')
    account=u.netloc.split('.')[0];container=u.path.split('/')[1];key=(account,container)
    if key not in tokens or (tokens[key][1]-datetime.datetime.now(datetime.timezone.utc)).total_seconds()<60:
        # Same container-scoped cache used by the official Planetary Computer SDK.
        response=session.get(f'https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}',timeout=30)
        response.raise_for_status();signed=response.json();tokens[key]=(signed['token'],datetime.datetime.fromisoformat(signed['msft:expiry'].replace('Z','+00:00')))
    return url+'?'+tokens[key][0]
def mapping(crs,affine,width=2400,height=2400):
    key=(str(crs),tuple(affine))
    if key not in mapping_cache:
        x,y=transform('EPSG:4326',crs,lon.ravel(),lat.ravel())
        cols,rows=(~affine)*(np.asarray(x),np.asarray(y));cols=np.floor(cols).astype(int).reshape(shape);rows=np.floor(rows).astype(int).reshape(shape)
        inside=(rows>=0)&(cols>=0)&(rows<height)&(cols<width)
        if not inside.any():raise ValueError('Selected tile has no target cells')
        r0=int(rows[inside].min());r1=int(rows[inside].max())+1;c0=int(cols[inside].min());c1=int(cols[inside].max())+1
        mapping_cache[key]=(inside,rows,cols,Window(c0,r0,c1-c0,r1-r0))
    return mapping_cache[key]
for item_index,item in enumerate(items):
    assert f'.A{stamp}.' in item['id'] and '.061.' in item['id']
    for band in range(1,8):
        for kind in ['Nadir_Reflectance','BRDF_Albedo_Band_Mandatory_Quality']:
            key=f'{kind}_Band{band}';asset=item['assets'][key]
            path=out/f'{item["id"]}-{key}.npz'
            if path.exists():
                with np.load(path,allow_pickle=False) as f:
                    data=f['raw'];tags=json.loads(str(f['metadata_json']));crs=rasterio.crs.CRS.from_wkt(str(f['crs_wkt']));affine=Affine(*f['transform']);saved_window=f['window']
                inside,rows,cols,window=mapping(crs,affine)
                assert np.array_equal(saved_window,[window.col_off,window.row_off,window.width,window.height])
            else:
                try:
                    with rasterio.open(signed_url(asset['href'])) as src:
                        tags=src.tags();crs=src.crs;affine=src.transform
                        inside,rows,cols,window=mapping(crs,affine,src.width,src.height)
                        data=src.read(1,window=window)
                        np.savez_compressed(path,raw=data,window=np.array([window.col_off,window.row_off,window.width,window.height]),transform=np.array(affine),crs_wkt=str(crs),metadata_json=json.dumps(tags))
                except Exception as error:
                    raise RuntimeError(f'Could not read public MODIS asset {key}: {type(error).__name__}') from None
            assert tags['LOCALGRANULEID'].startswith(item['id'])
            nodata=float(tags['_FillValue']);scale=float(tags.get('scale_factor','1'))
            if kind=='Nadir_Reflectance':assert scale==0.0001 and float(tags['add_offset'])==0 and nodata==32767
            else:assert nodata==255
            c0,r0=int(window.col_off),int(window.row_off)
            if band==1 and kind=='Nadir_Reflectance':
                assert not np.any(inside&(coverage>0)),'Overlapping target assignments'
                coverage[inside]=item_index+1
            target=raw if kind=='Nadir_Reflectance' else qa
            target[...,band-1][inside]=data[rows[inside]-r0,cols[inside]-c0]
            records.append(dict(item=item['id'],asset=key,url=asset['href'],file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),window=[c0,r0,int(window.width),int(window.height)],nodata=nodata,scale=scale,range_begin=tags.get('RANGEBEGINNINGDATE'),range_end=tags.get('RANGEENDINGDATE')))
        print(json.dumps(dict(item=item['id'],band=band,land_full_inversion=int(np.sum(land&inside&(qa[...,band-1]==0)&(raw[...,band-1]!=32767))))),flush=True)
    (out/'source-windows.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
valid=(raw>=0)&(raw<=32766)&land[...,None]&(coverage[...,None]>0)&np.isin(qa,[0,1])
full=valid&(qa==0);magnitude=valid&(qa==1)
nbar=raw.astype('f4')*np.float32(.0001);nbar[~valid]=np.nan
np.savez_compressed(out/'aligned-nbar.npz',latitude=lat,longitude=lon,land_mask=land.astype('u1'),band=np.arange(1,8,dtype='u1'),nbar=nbar,mandatory_quality=qa,valid=valid.astype('u1'),full_inversion=full.astype('u1'),tile_index=coverage)
report=dict(quantity='MODIS_7_band_nadir_BRDF_adjusted_surface_reflectance',date=args.date.isoformat(),tiles=[dict(index=i+1,granule_id=item['id']) for i,item in enumerate(items)],grid_sha256=hashlib.sha256(grid_path.read_bytes()).hexdigest(),sampling='nearest source pixel; no spatial, spectral or missing-data filling',shape=list(nbar.shape),land_count=int(land.sum()),covered_land=int((land&(coverage>0)).sum()),bands=[dict(band=i+1,valid_land=int(valid[...,i].sum()),full_inversion_land=int(full[...,i].sum()),magnitude_inversion_land=int(magnitude[...,i].sum()),maximum=float(np.nanmax(nbar[...,i]))) for i in range(7)],joint_rgb_full_inversion=int(np.all(full[...,[0,3,2]],axis=-1).sum()),joint_rgb_valid=int(np.all(valid[...,[0,3,2]],axis=-1).sum()),source_doi='10.5067/MODIS/MCD43A4.061',source_spec='https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/61/MCD43A4_c61.fs',limitations=['Source date is explicit; these observations do not reconstruct a different historical year.','NBAR is evaluated at nadir/local solar noon, not at arbitrary view or illumination.','Seven band reflectances, not a hyperspectral albedo curve.','QA==0 is full inversion; QA==1 is magnitude inversion. Both are retained separately, never silently promoted.','Scaled directional reflectance can exceed 1; it is not silently clipped to albedo bounds.','Source date selected from the granule ID; STAC interval metadata are not treated as the daily center date.'],output_sha256=hashlib.sha256((out/'aligned-nbar.npz').read_bytes()).hexdigest())
(out/'provenance.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report),flush=True)
