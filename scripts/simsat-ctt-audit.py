#!/usr/bin/env python3
"""Audit native WRF / cached SimSat cloud temperatures against aligned ABI C13.

Requires numpy and netCDF4. Reads WRF and compact-u8 SSB v6; never ingests or
edits source data. JSON is the default output. Large arrays are opt-in.

python scripts/simsat-ctt-audit.py --data-root C:/Users/drew/soma-render-work --out out/ctt-audit
python scripts/simsat-ctt-audit.py --self-test

CTT means VISIBLE tau=1, not a thermal contribution function or minimum cloudy
voxel temperature. A column mask can label low model cloud and deep observed
convection "both cloudy"; that regime does not isolate operator error.
"""
from pathlib import Path
import argparse, json, struct, subprocess, tempfile, zlib, warnings
import numpy as np
from netCDF4 import Dataset

NATIVE_CLOUD_THRESHOLDS = (0.0, 1e-8, 1e-6, 1e-5)


def resolve_path(args, pattern, hour):
    path = Path(pattern.format(date=args.date, hour=hour,
                               compact_date=args.date.replace('-', ''),
                               cache_date=args.date.replace('-', '_')))
    return path if path.is_absolute() else args.data_root / path


def clean_json(value):
    """Standards-compliant JSON: absent cloud diagnostics become null."""
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False)+'\n', encoding='utf-8')


def native_temperature(theta, p_perturbation, p_base):
    # ingest.rs adds P+PB as f32, then promotes. optics.rs uses the WRF
    # theta offset 300 K, p0=100000 Pa, and Rd/Cp=2/7. Temperature is stored
    # f32 before vertical resampling; this reproduces that operation order.
    p = np.asarray(p_perturbation, np.float32) + np.asarray(p_base, np.float32)
    return ((np.asarray(theta, np.float32).astype(np.float64)+300.0)
            * (p.astype(np.float64)/100000.0)**(2.0/7.0)).astype(np.float32)


def stats(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return dict(n=int(x.size), min=float(x.min()), p01=float(np.percentile(x,1)), median=float(np.median(x)), mean=float(x.mean()), max=float(x.max())) if x.size else dict(n=0)

def ssb(path):
    with path.open('rb') as f:
        if f.read(4) != b'SSB1':
            raise ValueError(f'not an SSB cache: {path}')
        ver, n = struct.unpack('<II',f.read(8))
        header = json.loads(f.read(n))
        if ver != 6 or header['storage_profile'] != 'compact-u8':
            raise ValueError('this audit supports compact-u8 SSB v6 only')
        raw = zlib.decompress(f.read())
    shape = (header['nz'],header['ny'],header['nx'])
    count = int(np.prod(shape)); off = 0; channels = {}
    for name in header['channels_3d']:
        code = np.frombuffer(raw,np.uint8,count,off).reshape(shape); off += count
        if name == 'cloud_fraction':
            channels[name] = code
        elif name in ['ext_liquid','ext_ice','ext_snow','ext_precip']:
            q = header['quant'][name]
            lut = np.zeros(256,np.float32)
            if q['vmax']>0:
                lut[1:] = q['vmin']*(q['vmax']/q['vmin'])**(np.arange(255,dtype=np.float64)/254)
            channels[name] = lut[code]
    # Rust decoder adds in f64 then casts to f32.
    temp = (np.frombuffer(raw,'<f2',count,off).reshape(shape).astype(np.float64)+273.15).astype(np.float32)
    expected = count*(len(header['channels_3d'])+2) + header['nx']*header['ny']*len(header['planes_2d'])*4
    if len(raw) != expected:
        raise ValueError(f'cache payload size {len(raw)} != expected {expected}')
    return header, channels, temp

def derived_arrays(header, ch, bt):
    # derived::cloud_top_temp_field: decoded f32 extinctions are summed as
    # f64, then integrated top-down by whole dz layers, with no interpolation.
    ext = ch['ext_liquid'].astype(np.float64)+ch['ext_ice'].astype(np.float64)+ch['ext_precip'].astype(np.float64)
    cum = np.cumsum(ext[::-1]*header['dz_m'],axis=0)
    ctt_present = cum[-1]>=1
    ktop = header['nz']-1-np.argmax(cum>=1,axis=0)
    ctt = np.take_along_axis(bt,ktop[None],axis=0)[0]
    ctt[~ctt_present]=np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        bmin = np.nanmin(np.where(ext>0,bt,np.nan),axis=0)
    dz = header['dz_m']; z0 = header['z_min_m']
    ctt_h = z0+ktop*dz
    ctt_h = np.where(ctt_present,ctt_h,np.nan)
    arrays = dict(ctt=ctt[::-1],ctt_height_m=ctt_h[::-1],brick_min_cloudy_voxel_k=bmin[::-1],brick_cod=cum[-1][::-1])
    # Reproduce T3 derived::condensate_cloud_mask_field. The standard density
    # is an acknowledged approximation to the true WRF density used at ingest.
    # Constants are optics::{standard_air_density_kg_m3,HydrometeorClass,RHO_W}:
    # rho0=1.225 kg/m3, H=8500 m; radii=10/40/150/1000 um; rho_water=1000.
    z = z0+np.arange(header['nz'])*dz
    rho = 1.225*np.exp(-np.maximum(z,0)/8500.0)
    q = ((ch['ext_liquid'].astype(np.float64)*10e-6
          +ch['ext_ice'].astype(np.float64)*40e-6
          +ch['ext_snow'].astype(np.float64)*150e-6
          +np.maximum(ch['ext_precip'].astype(np.float64)-ch['ext_snow'],0)*1e-3)
         *1000.0/(1.5*rho[:,None,None]))
    arrays['brick_condensate_mask'] = np.any(q>1e-6,axis=0)[::-1]
    return arrays


def run(args, hour):
    wrf_path = resolve_path(args,args.wrf_pattern,hour)
    brick_path = resolve_path(args,args.brick_pattern,hour)
    ref_path = resolve_path(args,args.reference_pattern,hour)
    baseline_path = resolve_path(args,args.baseline_pattern,hour)
    header, ch, bt = ssb(brick_path)
    expected_time = f'{args.date}T{hour:02d}:00:00Z'
    if header.get('time_iso') != expected_time:
        raise ValueError(f'SSB time {header.get("time_iso")} != requested {expected_time}')
    arrays = derived_arrays(header,ch,bt)
    del ch,bt
    with Dataset(wrf_path) as d:
        d.set_auto_mask(False)
        step = args.timestep
        if 'Times' in d.variables:
            native_time = d['Times'][step].tobytes().decode('ascii').strip('\x00 ').replace('_','T')+'Z'
            if native_time != expected_time:
                raise ValueError(f'WRF time {native_time} != requested {expected_time}')
        theta,p,pb = (d[name][step] for name in ('T','P','PB'))
        temp = native_temperature(theta,p,pb)
        full_f64 = (theta.astype(np.float64)+300)*((p.astype(np.float64)+pb.astype(np.float64))/1e5)**(2/7)
        numerical_difference = float(np.max(np.abs(temp.astype(np.float64)-full_f64)))
        del theta,p,pb,full_f64
        # ingest reads PH/PHB as f64, then stores destaggered mass heights f32.
        geopot = (d['PH'][step].astype(np.float64)+d['PHB'][step].astype(np.float64))/9.81
        z = ((geopot[1:]+geopot[:-1])*0.5).astype(np.float32)
        native_lat = d['XLAT'][step][::-1]
        native_lon = d['XLONG'][step][::-1]
        q = np.zeros(temp.shape,np.float64)
        species = [n for n in ['QCLOUD','QICE','QSNOW','QRAIN','QGRAUP','QHAIL'] if n in d.variables]
        for name in species:
            q += np.maximum(d[name][step].astype(np.float64),0)
        native = {}
        for threshold in NATIVE_CLOUD_THRESHOLDS:
            cloudy = q>threshold
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                tmin = np.nanmin(np.where(cloudy,temp,np.nan),axis=0)
            top = temp.shape[0]-1-np.argmax(cloudy[::-1],axis=0)
            top_t = np.take_along_axis(temp,top[None],axis=0)[0]
            top_z = np.take_along_axis(z,top[None],axis=0)[0]
            present = cloudy.any(axis=0)
            top_t[~present]=np.nan; top_z[~present]=np.nan
            slug = str(threshold)
            arrays['native_min_cloudy_k_'+slug]=tmin[::-1]
            arrays['native_highest_cloudy_k_'+slug]=top_t[::-1]
            arrays['native_highest_cloudy_height_m_'+slug]=top_z[::-1]
            native[slug] = dict(minimum_cloudy_voxel=stats(tmin),highest_cloudy_temperature=stats(top_t),cloud_column_count=int(present.sum()))
        arrays['native_min_air_temperature_k'] = temp.min(axis=0)[::-1]
        # Samples of all source profiles at globally coldest cloudy voxels and obs minima.
        native_meta = dict(has_CLDFRA='CLDFRA' in d.variables,MP_PHYSICS=int(getattr(d,'MP_PHYSICS',-1)),species=species,nz=temp.shape[0],all_temperature=stats(temp),all_height=stats(z),
                           temperature_reconstruction='f32 P+PB; f64 Poisson at kappa=2/7 and theta offset 300 K; f32 temperature before resampling, matching ingest.rs',
                           difference_from_all_f64_max_abs_kelvin=numerical_difference)
    ref = np.load(ref_path)
    obs=ref['cmi_c13']; sim=np.fromfile(baseline_path,'<f4').reshape(obs.shape)
    if native_lat.shape != obs.shape or arrays['ctt'].shape != obs.shape:
        raise ValueError('WRF, native SSB and aligned ABI shapes must agree')
    lat_error = float(np.max(np.abs(native_lat-ref['lat'])))
    lon_error = float(np.max(np.abs(native_lon-ref['lon'])))
    if max(lat_error,lon_error)>1e-5:
        raise ValueError('aligned ABI does not match north-first native WRF within 1e-5 degrees')
    valid=(ref['valid']>0)&np.isfinite(obs)&np.isfinite(sim)
    modelmask=arrays['brick_condensate_mask']
    mask_evidence={'source':'reproduced from the cached SSB using the T3 formula'}
    if args.mask_pattern:
        mask_path=resolve_path(args,args.mask_pattern,hour)
        supplied=np.fromfile(mask_path,np.uint8).reshape(obs.shape)
        checked=valid&(supplied!=255)
        mismatches=int(np.count_nonzero(modelmask[checked]!=(supplied[checked]==1)))
        mask_evidence.update(supplied_path=str(mask_path),mismatch_count=mismatches)
        if mismatches:
            raise ValueError(f'reproduced T3 mask disagrees at {mismatches} pixels')
        valid &= supplied!=255
    if not np.any(valid):
        raise ValueError('no common finite valid baseline/reference pixels')
    both=valid&(ref['bcm']==1)&modelmask
    rows={}
    for name,mask in [('valid',valid),('both_cloudy',both),('obs_le235',valid&(obs<=235)),('obs_le205',valid&(obs<=205)),('both_cloudy_obs_le205',both&(obs<=205)),('obs_le200',valid&(obs<=200))]:
        row=dict(n=int(mask.sum()),observed=stats(obs[mask]),baseline_bt=stats(sim[mask]),baseline_bias=float(np.mean(sim[mask]-obs[mask])) if mask.any() else None)
        for field in ['ctt','brick_min_cloudy_voxel_k','native_min_cloudy_k_1e-06','native_highest_cloudy_k_1e-06','native_min_air_temperature_k']:
            val=arrays[field]
            finite=mask&np.isfinite(val)
            row[field]=stats(val[mask])
            if finite.any():
                row[field]['minus_observed_mean']=float((val[finite]-obs[finite]).mean())
                row[field]['fraction_at_or_colder_than_observed']=float((val[finite]<=obs[finite]).mean())
        rows[name]=row
    cold_area={str(t):dict(observed=float(np.mean(obs[valid]<=t)),baseline=float(np.mean(sim[valid]<=t)),ctt=float(np.mean(arrays['ctt'][valid]<=t)),brick_min_cloudy=float(np.mean(arrays['brick_min_cloudy_voxel_k'][valid]<=t)),native_min_q1e6=float(np.mean(arrays['native_min_cloudy_k_1e-06'][valid]<=t))) for t in [193,200,205,220,235]}
    minima={}
    for field,plane in [('observed',obs),('baseline',sim),('ctt',arrays['ctt']),('brick_min_cloudy_voxel_k',arrays['brick_min_cloudy_voxel_k']),('native_min_cloudy_k_1e-06',arrays['native_min_cloudy_k_1e-06'])]:
        masked=np.where(valid,plane,np.nan)
        if not np.isfinite(masked).any():
            minima[field]=None
            continue
        j,i=np.unravel_index(np.nanargmin(masked),masked.shape)
        jj=masked.shape[0]-1-j
        minima[field]=dict(north_first_row=int(j),column=int(i),lat=float(ref['lat'][j,i]),lon=float(ref['lon'][j,i]),observed=float(obs[j,i]),baseline=float(sim[j,i]),ctt=float(arrays['ctt'][j,i]),brick_min_cloudy=float(arrays['brick_min_cloudy_voxel_k'][j,i]),native_min_q1e6=float(arrays['native_min_cloudy_k_1e-06'][j,i]),native_highest_q1e6=float(arrays['native_highest_cloudy_k_1e-06'][j,i]),native_highest_height_q1e6=float(arrays['native_highest_cloudy_height_m_1e-06'][j,i]),native_min_air=float(arrays['native_min_air_temperature_k'][j,i]))
    if args.save_arrays:
        np.savez_compressed(args.out/f'ctt-audit-{hour:02d}z.npz',**arrays,observed_c13=obs,baseline_c13=sim,valid=valid,both_cloudy=both)
    result=dict(hour=hour,date=args.date,wrf=str(wrf_path),brick=str(brick_path),reference=str(ref_path),baseline=str(baseline_path),
                brick_header=header,native_metadata=native_meta,grid_check=dict(latitude_max_abs_degrees=lat_error,longitude_max_abs_degrees=lon_error),
                mask_evidence=mask_evidence,native_threshold_sensitivity=native,regimes=rows,cold_area=cold_area,minima_locations=minima)
    if args.verify_bin:
        manifest_path = brick_path.parent / 'run.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest_step = next(index for index,item in enumerate(manifest['timesteps']) if item['file']==brick_path.name)
        command=[str(args.verify_bin),f'input={brick_path.parent / "run.json"}','derived=ctt','view=topdown','resolution=native',f'ts={manifest_step}','threads=6',f'out={args.out / f"ctt-check-{hour:02d}z.png"}']
        proc=subprocess.run(command,capture_output=True,text=True,check=True)
        line=next(line for line in proc.stdout.splitlines() if line.startswith('DERIVEDSUMMARY '))
        parsed=dict(item.split('=',1) for item in line.split() if '=' in item)
        computed=stats(arrays['ctt'])
        for key in ('min','max','median'):
            if abs(float(parsed[key])-computed.get(key,float("nan")))>0.001:
                raise ValueError(f'CTT executable check disagrees for {key}: {parsed[key]} vs {computed[key]}')
        result['executable_check']=dict(command=command,stdout=line,passed=True)
    write_json(args.out/f'ctt-audit-{hour:02d}z.json',result)
    print(f'{hour:02d}Z min K: ABI {rows["valid"]["observed"]["min"]:.3f}; BT {rows["valid"]["baseline_bt"]["min"]:.3f}; '
          f'CTT {rows["valid"]["ctt"].get("min",float("nan")):.3f}; native cloudy {native["1e-06"]["minimum_cloudy_voxel"].get("min",float("nan")):.3f}; '
          f'native air {native_meta["all_temperature"]["min"]:.3f}; both-cloudy bias {rows["both_cloudy"]["baseline_bias"] if rows["both_cloudy"]["baseline_bias"] is not None else float("nan"):+.3f}',flush=True)
    return result


def self_test():
    # The cold upper layer has tau .5; the warmer next layer crosses tau 1.
    # A second column never crosses, so CTT stays NaN despite cold condensate.
    header=dict(nz=3,ny=1,nx=2,z_min_m=0.0,dz_m=250.0)
    liquid=np.array([[[0,0]],[[.004,0]],[[.002,.001]]],np.float32)
    zero=np.zeros_like(liquid)
    ch=dict(ext_liquid=liquid,ext_ice=zero,ext_precip=zero,ext_snow=zero)
    temp=np.array([[[280,280]],[[250,250]],[[210,210]]],np.float32)
    a=derived_arrays(header,ch,temp)
    assert a['ctt'][0,0]==250 and np.isnan(a['ctt'][0,1])
    assert a['brick_min_cloudy_voxel_k'][0,0]==210
    assert a['ctt_height_m'][0,0]==250
    assert native_temperature(np.array([-10],np.float32),np.array([1000],np.float32),np.array([99000],np.float32))[0]==290
    assert clean_json(dict(missing=float('nan')))==dict(missing=None)
    # A miniature real SSB checks channel order, the exact-zero code, log LUT
    # and binary16 Celsius decode, then derives CTT from its decoded payload.
    with tempfile.TemporaryDirectory() as directory:
        h={**header,'format_version':6,'storage_profile':'compact-u8','channels_3d':['ext_liquid','ext_ice','ext_snow','ext_precip','tau_up','qvapor','cloud_fraction'],
           'quant':{name:dict(vmin=.002,vmax=.004) for name in ['ext_liquid','ext_ice','ext_snow','ext_precip']},'planes_2d':['hgt']}
        cells=6
        codes=np.array([0,0,255,0,1,0],np.uint8)
        raw=codes.tobytes()+bytes(cells*5)+bytes([255])*cells+(temp-273.15).astype('<f2').tobytes()+bytes(8)
        hb=json.dumps(h).encode()
        path=Path(directory)/'test.ssb'
        path.write_bytes(b'SSB1'+struct.pack('<II',6,len(hb))+hb+zlib.compress(raw))
        hh,cc,tt=ssb(path)
        result=derived_arrays(hh,cc,tt)
        assert abs(float(result['ctt'][0,0])-250)<.02
        assert np.isnan(result['ctt'][0,1])
    print('CTT audit self-test passed')


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root',type=Path,default=Path('.'))
    parser.add_argument('--out',type=Path)
    parser.add_argument('--date',default='2026-09-04')
    parser.add_argument('--hours',default='12,15,18,21')
    parser.add_argument('--timestep',type=int,default=0,help='WRF time index; supplied SSB must describe this same time')
    parser.add_argument('--wrf-pattern',default='data/d01v/wrfout_d01_{date}_{hour:02d}_00_00')
    parser.add_argument('--brick-pattern',default='out/simsat-main/cache/wrfout_d01_{cache_date}_{hour:02d}_00_00/t{compact_date}_{hour:02d}00.ssb')
    parser.add_argument('--reference-pattern',default='out/simsat/goesfd-{hour:02d}z/abi-reference-aligned.npz')
    parser.add_argument('--baseline-pattern',default='out/simsat-main/ir13_d01_{hour:02d}Z.bin')
    parser.add_argument('--mask-pattern',help='optional existing T3 u8 plane; must match the reproduced mask exactly')
    parser.add_argument('--verify-bin',type=Path,help='optional simsat-render-ir executable; validates CTT from run.json and writes small PNGs')
    parser.add_argument('--save-arrays',action='store_true',help='also save per-pixel NPZ evidence; not for version control')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.out is None:
        parser.error('--out is required unless --self-test is used')
    hours=[int(value) for value in args.hours.split(',')]
    if not hours or any(h<0 or h>23 for h in hours):
        parser.error('--hours must contain UTC hours 0..23')
    args.out.mkdir(parents=True,exist_ok=True)
    results=[run(args,h) for h in hours]
    write_json(args.out/'ctt-audit.json',dict(schema_version=1,diagnostic='native WRF / cached visible tau=1 / colocated ABI C13',
                                           limitations=['CTT is visible tau=1; thin columns are NaN.','Both-cloudy column masks do not match cloud height or optical thickness.','Cloudy voxel thresholds measure support, not emissivity.','A temperature minimum alone cannot validate absorption/scattering physics.'],
                                           hours=results))


if __name__=='__main__':
    main()
