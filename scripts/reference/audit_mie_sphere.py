"""Independent reference audit; does not import SimSat numerical kernels."""
import csv, hashlib, io, json, math, subprocess, sys
from pathlib import Path
import numpy as np
import scipy
from scipy.special import spherical_jn, spherical_yn
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--simulated',type=Path,required=True)
parser.add_argument('--bhmie-exe',type=Path,required=True)
parser.add_argument('--bhmie-source',type=Path,required=True)
parser.add_argument('--bhmie-wrapper',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
OUT=args.output_dir;OUT.mkdir(parents=True,exist_ok=False)
request=args.simulated
exe=args.bhmie_exe
cases=json.loads(request.read_text())['cases']
transcript=''.join(f"{c['id']} {c['x']:.17g} {c['n_real']:.17g} {c['n_imag_positive']:.17g} 901\n" for c in cases)
(OUT/'bhmie-input.txt').write_text(transcript,encoding='ascii',newline='\n')
with (OUT/'bhmie-full.csv').open('w',encoding='ascii',newline='') as stdout, (OUT/'bhmie-stderr.txt').open('w') as stderr:
    proc=subprocess.run([str(exe)],input=transcript,text=True,stdout=stdout,stderr=stderr,timeout=180)
if proc.returncode: raise RuntimeError(f'BHMIE exited {proc.returncode}')
rows={}
for row in csv.DictReader((OUT/'bhmie-full.csv').open()):
    row={k:float(v) for k,v in row.items()}
    rows[(int(row['id']),round(row['angle_deg'],4))]=row
fields=['id','x','n_real','n_imag','angle_deg','qext','qsca','g','phase_sr1']
selected=[]; scipy_rows=[]; failures=[]; records=[]
tols={'bhmie':{'phase_abs':1e-8,'phase_rel':2e-5,'efficiency_abs':1e-7,'efficiency_rel':1e-5},'scipy':{'phase_abs':1e-10,'phase_rel':1e-7,'efficiency_abs':1e-12,'efficiency_rel':2e-9}}
worst={}
def check(source,metric,sim,ref,case,angle=None):
    if not math.isfinite(ref) or not math.isfinite(sim): raise ValueError('nonfinite reference')
    t=tols[source]; typ='phase' if metric=='phase_sr1' else 'efficiency'
    absolute=abs(sim-ref); relative=absolute/max(abs(ref),1e-30)
    ratio=absolute/(t[typ+'_abs']+t[typ+'_rel']*abs(ref))
    key=source+':'+metric
    rec={'source':source,'metric':metric,'id':case['id'],'x':case['x'],'n_real':case['n_real'],'n_imag_positive':case['n_imag_positive'],'angle_deg':angle,'simulated':sim,'reference':ref,'absolute_error':absolute,'relative_error':relative,'tolerance_ratio':ratio}
    if key not in worst or ratio>worst[key]['tolerance_ratio']: worst[key]=rec
    if ratio>1: failures.append(rec)
for c in cases:
    x=c['x']; m=complex(c['n_real'],c['n_imag_positive'])
    # SciPy special functions supply every Bessel value directly. No SimSat
    # downward derivative/Miller recurrence or coefficients are reused.
    n=np.arange(1,c['orders']+1,dtype=float)
    z=m*x
    jz=spherical_jn(n,z); dz=spherical_jn(n,z,True)/jz+1/z
    psi=x*spherical_jn(n,x); prev=x*spherical_jn(n-1,x)
    xi=psi+1j*x*spherical_yn(n,x); prevxi=prev+1j*x*spherical_yn(n-1,x)
    aa=dz/m+n/x; bb=m*dz+n/x
    a=(aa*psi-prev)/(aa*xi-prevxi); b=(bb*psi-prev)/(bb*xi-prevxi)
    qsca=2*np.sum((2*n+1)*(abs(a)**2+abs(b)**2))/x**2
    qext=2*np.sum((2*n+1)*(a+b).real)/x**2
    g=4/(x*x*qsca)*(np.sum(n[:-1]*(n[:-1]+2)/(n[:-1]+1)*(a[:-1]*a[1:].conjugate()+b[:-1]*b[1:].conjugate()).real)+np.sum((2*n+1)/(n*(n+1))*(a*b.conjugate()).real))
    first=rows[(c['id'],0.0)]
    for metric,key,ref in [('qext','extinction_efficiency',qext),('qsca','scattering_efficiency',qsca),('g','asymmetry',g)]:
        check('scipy',metric,c[key],float(ref),c)
        check('bhmie',metric,c[key],first[metric],c)
    for p in c['phase']:
        angle=p['degrees']; mu=math.cos(math.radians(angle))
        # Independent straightforward f64 angular recurrence, unlike the
        # cancellation-controlled positive-mu recurrence in the renderer.
        previous=0.; current=1.; s1=0j; s2=0j
        for j in range(len(n)):
            order=j+1
            tau=order*mu*current-(order+1)*previous
            factor=(2*order+1)/(order*(order+1))
            s1+=factor*(a[j]*current+b[j]*tau)
            s2+=factor*(a[j]*tau+b[j]*current)
            previous,current=current,((2*order+1)*mu*current-(order+1)*previous)/order
        phase=(abs(s1)**2+abs(s2)**2)/(2*math.pi*x*x*qsca)
        row=rows[(c['id'],round(angle,4))]
        selected.append({k:row[k] for k in fields})
        scipy_rows.append(dict(id=c['id'],x=x,n_real=m.real,n_imag=m.imag,angle_deg=angle,qext=float(qext),qsca=float(qsca),g=float(g),phase_sr1=float(phase)))
        check('scipy','phase_sr1',p['phase_sr1'],float(phase),c,angle)
        check('bhmie','phase_sr1',p['phase_sr1'],row['phase_sr1'],c,angle)
    print(f"case {c['id']+1}/{len(cases)} x={x:g} done",flush=True)
for filename,values in [('bhmie-selected.csv',selected),('scipy-selected.csv',scipy_rows)]:
    with (OUT/filename).open('w',encoding='ascii',newline='') as f:
        w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(values)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result={'schema':'simsat-mie-independent-reference-v1','cases':len(cases),'angles_per_case':13,'tolerances':tols,'worst_by_source_metric':worst,'failures':failures,'passed':not failures,'provenance':{'request_sha256':sha(request),'bhmie_source_sha256':sha(args.bhmie_source),'wrapper_sha256':sha(args.bhmie_wrapper),'executable_sha256':sha(exe),'bhmie_input_sha256':sha(OUT/'bhmie-input.txt'),'bhmie_full_output_sha256':sha(OUT/'bhmie-full.csv'),'bhmie_selected_sha256':sha(OUT/'bhmie-selected.csv'),'scipy_selected_sha256':sha(OUT/'scipy-selected.csv'),'scipy_version':scipy.__version__,'numpy_version':np.__version__},'limitations':['BHMIE public inputs/output and angular amplitudes are f32; all input requests are quantized identically.','SciPy uses direct Bessel special functions; its scalar amplitude recurrence is independent f64, not a separate radiation-transfer solver.','No distribution-averaged cloud optical properties or nonspherical ice validation.']}
(OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'passed':result['passed'],'failures':len(failures),'worst':{k:v['tolerance_ratio'] for k,v in worst.items()}},indent=2))

sys.exit(0 if result["passed"] else 1)
