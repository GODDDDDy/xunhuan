"""CPU-only paired C effects; needed because prior analyzer only compared U to T1.

No fitting, no new forward, no test access. All experiment arrays remain remote.
"""
from pathlib import Path
import hashlib,json,collections
import numpy as np
P=Path('/media/li/data3/25thstudent/DY/929work/pidsr_k2_929')
R=P/'polar_quality_mechanism_audit_20261015'
OUT=R/'reports/c_focused_pairing';OUT.mkdir(parents=True,exist_ok=True)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pool(a,k):
    h,w=a.shape;h=h//k*k;w=w//k*k
    return a[:h,:w].reshape(h//k,k,w//k,k).mean((1,3))
def corr(a,b):
    def rank(x):
        _,inv,n=np.unique(x.ravel(),return_inverse=True,return_counts=True)
        return (np.cumsum(n)-n+(n-1)/2)[inv]
    a,b=rank(a),rank(b)
    return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None
rows=[];sources={};pairing=[]
def compare(label,path,t1,gt,c,real,control,kind):
    assert real.shape==control.shape==t1.shape==gt.shape
    for scale in [4,16,32]:
        err=lambda x:pool(np.mean((x.astype(np.float64)-gt.astype(np.float64))**2,axis=0),scale)
        e1,er,ec=map(err,[t1,real,control]);gc=ec-er;gr=e1-er;gf=e1-ec
        conf=c[0] if scale==4 else pool(c[0],scale//4)
        assert conf.shape==gc.shape
        order=np.argsort(conf.ravel(),kind='stable');bins=[]
        for i,ids in enumerate(np.array_split(order,5)):
            bins.append(dict(low_to_high_bin=i+1,mean_C=float(conf.ravel()[ids].mean()),
                mean_C_increment=float(gc.ravel()[ids].mean()),
                mean_real_gain_vs_T1=float(gr.ravel()[ids].mean()),
                mean_control_gain_vs_T1=float(gf.ravel()[ids].mean()),
                real_better_fraction=float((gc.ravel()[ids]>0).mean())))
        rows.append(dict(comparison=label,interpretation=kind,file=path.name,scene=path.stem.split('_')[-2],scale=scale,
            regions=int(gc.size),covered_shape=[gc.shape[0]*scale,gc.shape[1]*scale],
            mean_C_increment=float(gc.mean()),mean_real_gain_vs_T1=float(gr.mean()),mean_control_gain_vs_T1=float(gf.mean()),
            real_better_fraction=float((gc>0).mean()),rho_C_increment=corr(conf,gc),
            real_rescues_control_harm_fraction=float(((gr>0)&(gf<=0)).mean()),
            real_destroys_control_benefit_fraction=float(((gr<=0)&(gf>0)).mean()),
            both_harmful_fraction=float(((gr<=0)&(gf<=0)).mean()),
            both_improve_fraction=float(((gr>0)&(gf>0)).mean()),C_quintiles=bins))

D26=P/'polar_confidence_candidate_diagnostic_20261014'
for path in sorted((D26/'examples').glob('*.npz')):
    sources[str(path)]=sha(path)
    with np.load(path) as z:
        assert np.array_equal(np.sort(z['c_pred'].ravel()),np.sort(z['c_shifted_quality'].ravel()))
        compare('26_frozen_real_minus_fixed',path,z['t1'],z['target'],z['c_pred'],z['u_real_quality'],z['u_fixed_quality'],'same weights, joint b/C intervention; not trained ablation')
        compare('26_frozen_real_minus_shifted',path,z['t1'],z['target'],z['c_pred'],z['u_real_quality'],z['u_shifted_quality'],'same weights; b/C jointly shifted, identical C marginal distribution; inference distribution caveat')

D27=P/'polar_quality_matched_candidate_20261015'
for path in sorted((R/'examples/validation').glob('*.npz')):
    other=D27/'examples/validation'/path.name;sources[str(path)]=sha(path);sources[str(other)]=sha(other)
    with np.load(path) as a,np.load(other) as b:
        assert np.array_equal(a['t1'],b['t1']) and np.array_equal(a['gt'],b['gt'])
        pairing.append(dict(file=path.name,first_and_GT_equal=True))
        compare('28_initial_real_minus_fixed',path,a['t1'],a['gt'],a['c'],a['real_quality'],a['fixed_quality'],'same common initial weights; numeric quality condition effect')
        compare('27_endpoint_real_minus_fixed',path,b['t1'],b['gt'],a['c'],b['real_quality'],b['fixed_quality'],'different matched-trained endpoints; includes training adaptation, not pure condition effect')

macro=[]
for label in sorted({r['comparison'] for r in rows}):
    for scale in [4,16,32]:
        part=[r for r in rows if r['comparison']==label and r['scale']==scale]
        assert len(part)==8 and len({r['scene'] for r in part})==8
        macro.append(dict(comparison=label,scale=scale,frames=8,scenes=8,
            equal_scene_mean_C_increment=float(np.mean([r['mean_C_increment'] for r in part])),
            scenes_real_better=sum(r['mean_C_increment']>0 for r in part),
            low_C_bin_positive_scenes=sum(r['C_quintiles'][0]['mean_C_increment']>0 for r in part),
            low_C_equal_scene_increment=float(np.mean([r['C_quintiles'][0]['mean_C_increment'] for r in part])),
            high_C_equal_scene_increment=float(np.mean([r['C_quintiles'][-1]['mean_C_increment'] for r in part]))))
result=dict(training=False,new_GPU_forward=False,test_used=False,threshold_fitting=False,regional_frames_per_comparison=8,
    independence='8 fixed images per comparison, one per scene; not872 regional images or independent pixels',
    definition='G_C=mean((U_control-GT)^2)-mean((U_real-GT)^2); positive favors real condition; not necessarily above T1',
    source_hashes=sources,pairing=pairing,rows=rows,scene_equal_summaries=macro)
target=OUT/'C_increment_results.json';target.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
manifest=dict(files=[dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p)) for p in [target,Path(__file__)]],primary_archive_sha256='1be6003531063ca3eedfd1f27221811414b9d939058b28b0721967d17270dfa7',source_hashes=sources)
mp=OUT/'manifest.json';mp.write_text(json.dumps(manifest,indent=2))
for item in manifest['files']:assert sha(R/item['path'])==item['sha256']
(OUT/'verification_receipt.json').write_text(json.dumps(dict(passed=True,manifest_sha256=sha(mp),server_only=True,source_files=len(sources))))
print(json.dumps(dict(completed=True,new_training=False,frames_per_comparison=8,scale4=[r for r in macro if r['scale']==4]),ensure_ascii=True))
