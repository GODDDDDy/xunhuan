"""Descriptive CPU analysis: paired baseline changes and regional error algebra."""
import argparse,csv,hashlib,json,os
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument('--require-baseline',action='store_true')
parser.add_argument('--source27')
parser.add_argument('--baseline')
parser.add_argument('--output')
parser.add_argument('--no-plots',action='store_true')
args=parser.parse_args()
P=Path('F:/WORK/projects/pidsr_recurrent_joint_k2')
OUT=Path(args.output) if args.output else P/'analysis/polar_quality_mechanism_audit_20261015'
SOURCE=Path(args.source27) if args.source27 else P/'_dl/polar_quality_matched_candidate_20261015/artifacts'
BASE=Path(args.baseline) if args.baseline else P/'_dl/polar_quality_mechanism_audit_20261015/artifacts'
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'figures').mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR']=str(OUT/'runtime_cache/matplotlib')
if not args.no_plots:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def pool(array,scale):
    h,w=array.shape;h0=h//scale*scale;w0=w//scale*scale
    return array[:h0,:w0].reshape(h0//scale,scale,w0//scale,scale).mean((1,3))
def corr(a,b):
    def ranks(x):
        _,inv,n=np.unique(x.ravel(),return_inverse=True,return_counts=True)
        return (np.cumsum(n)-n+(n-1)/2+1)[inv]
    a,b=ranks(a),ranks(b)
    return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None
def dissect(first,proposal,target,scale):
    r=target.astype(np.float64)-first.astype(np.float64);d=proposal.astype(np.float64)-first.astype(np.float64)
    dot=pool(np.mean(r*d,axis=0),scale);energy=pool(np.mean(d*d,axis=0),scale)
    gain=pool(np.mean(r*r-(r-d)**2,axis=0),scale)
    assert np.max(np.abs(gain-(2*dot-energy)))<1e-12
    active=energy>1e-16;wrong=active&(dot<=0);excess=active&(dot>0)&(gain<=0);good=active&(gain>0)
    harm=np.maximum(-gain,0);total=float(harm.sum())
    return {'regions':gain.size,'covered_shape':[first.shape[-2]//scale*scale,first.shape[-1]//scale*scale],
        'mean_mse_improvement':float(gain.mean()),'positive_region_fraction':float(good.mean()),
        'wrong_direction_region_fraction':float(wrong.mean()),'positive_direction_overstep_fraction':float(excess.mean()),
        'no_change_fraction':float((~active).mean()),
        'wrong_direction_harm_share':float(harm[wrong].sum()/total) if total else 0.,
        'overstep_harm_share':float(harm[excess].sum()/total) if total else 0.},gain

endpoint=json.loads((SOURCE/'metrics/final_validation.json').read_text(encoding='utf-8'))
final={r['sample_id']:r for r in endpoint['rows']}
baseline_path=BASE/'metrics/baseline_validation.json'
baseline=json.loads(baseline_path.read_text(encoding='utf-8')) if baseline_path.exists() else None
if args.require_baseline:assert baseline and len(baseline['rows'])==872
payload={'training':False,'test_used':False,'baseline_complete':baseline is not None,
    'regional_frames':8,'regional_scope':'Only first pre-fixed validation image per8scenes; not872 regional outputs',
    'pixel_independence_claim':False,'thresholds_fitted':False,'regions':[]}
if baseline:
    before={r['sample_id']:r for r in baseline['rows']}
    assert set(before)==set(final) and all(before[k]['t1_sha256']==final[k]['t1_sha256'] for k in final)
    result=[]
    for sid,after in final.items():
        prior=before[sid];entry={'sample_id':sid,'scene':after['scene']}
        for arm in ('real_quality','fixed_quality'):
            entry[arm]={m:after['arms'][arm][m]-prior['arms'][arm][m] for m in ('psnr','ssim')}
        result.append(entry)
    per_scene=[]
    for scene in sorted({r['scene'] for r in result}):
        rows=[r for r in result if r['scene']==scene]
        per_scene.append({'scene':scene,'frames':len(rows),'changes':{
            a:{m:float(np.mean([r[a][m] for r in rows])) for m in ('psnr','ssim')} for a in ('real_quality','fixed_quality')}})
    payload['paired_training_changes']={'frames':872,'T1_hashes_exact':True,'per_scene':per_scene,
        'image_means':{a:{m:float(np.mean([r[a][m] for r in result])) for m in ('psnr','ssim')} for a in ('real_quality','fixed_quality')},
        'per_image':result}

for path in sorted((SOURCE/'examples/validation').glob('*.npz')):
    sid=next(k for k in final if k.replace('/','_')==path.stem)
    new=BASE/'examples/validation'/path.name
    with np.load(path) as z:
        first,target=z['t1'],z['gt'];native_c=None
        if new.exists():
            with np.load(new) as original:
                assert hashlib.sha256(np.ascontiguousarray(original['t1']).tobytes()).hexdigest()==final[sid]['t1_sha256']
                assert np.array_equal(first,original['t1']) and np.array_equal(target,original['gt'])
                native_c=original['c'][0].copy()
                initial_outputs={a:original[a].copy() for a in ('real_quality','fixed_quality')}
        stages=['endpoint']+(['initial'] if native_c is not None else [])
        for stage in stages:
            for arm in ('real_quality','fixed_quality'):
                proposal=z[arm] if stage=='endpoint' else initial_outputs[arm]
                for scale in (4,16,32):
                    row,gain=dissect(first,proposal,target,scale)
                    row.update(sample_id=sid,scene=final[sid]['scene'],stage=stage,arm=arm,scale=scale,
                        endpoint_example_sha256=sha(path))
                    e=pool(np.mean(np.abs(np.clip(first,0,1)-target),axis=0),scale)
                    row['rho_T1_MAE_gain']=corr(e,gain)
                    if native_c is not None:
                        c=native_c if scale==4 else pool(native_c,scale//4)
                        assert c.shape==gain.shape
                        row['rho_C_T1_MAE']=corr(c,e);row['rho_C_gain']=corr(c,gain)
                        order=np.argsort(c.ravel(),kind='stable');bins=[]
                        for n,ids in enumerate(np.array_split(order,5)):
                            bins.append({'quantile':n+1,'regions':len(ids),'mean_C':float(c.ravel()[ids].mean()),
                                'mean_T1_MAE':float(e.ravel()[ids].mean()),'mean_gain':float(gain.ravel()[ids].mean()),
                                'positive_region_fraction':float((gain.ravel()[ids]>0).mean())})
                        row['C_quintiles_descriptive_only']=bins
                    payload['regions'].append(row)

filename='audit_results.json' if baseline else 'interim_endpoint_decomposition.json'
(OUT/filename).write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
selected=[r for r in payload['regions'] if r['stage']=='endpoint' and r['arm']=='real_quality' and r['scale']==4]
if not args.no_plots:
    fig,ax=plt.subplots(figsize=(10,4.6),constrained_layout=True);x=np.arange(len(selected))
    ax.bar(x,[r['wrong_direction_harm_share'] for r in selected],label='Non-positive direction',color='#b95147')
    ax.bar(x,[r['overstep_harm_share'] for r in selected],bottom=[r['wrong_direction_harm_share'] for r in selected],
        label='Positive direction, excessive step',color='#d9a34c')
    ax.set(xticks=x,xticklabels=[r['scene'] for r in selected],ylim=(0,1.18),yticks=np.linspace(0,1,6),xlabel='Scene (one pre-fixed validation frame each)',
        ylabel='Share of total harmful-region MSE',title='Endpoint real-quality U: local squared-error harm decomposition (4x4)')
    ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,loc='upper center',ncol=2)
    fig.savefig(OUT/'figures/harm_decomposition.png',dpi=180);plt.close(fig)
print(json.dumps({'baseline_complete':baseline is not None,'regional_rows':len(payload['regions']),
    'real_endpoint_scale4':[{'scene':r['scene'],'wrong_direction_harm_share':r['wrong_direction_harm_share'],
        'overstep_harm_share':r['overstep_harm_share'],'mean_mse_improvement':r['mean_mse_improvement']} for r in selected]},indent=2))
