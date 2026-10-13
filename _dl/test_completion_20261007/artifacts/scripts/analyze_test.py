"""Audit inference and compare predeclared contrasts separately on both splits."""
import csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
lock=read(ROOT/'protocol_lock.json')
assert read(ROOT/'status.json')['state']=='complete'
assert all(sha(p)==h for p,h in lock['locked_file_sha256'].items())
assert all(sha(p)==h for p,h in lock['test_data_sha256'].items())
out=ROOT/'analysis';refs=out/'references';refs.mkdir(exist_ok=True)
historical=ROOT.parent.parent/'k1_ablation_20260930/runs/G0_official/seed_2026/metrics/step_0_test.json'
old=read(historical);new=read(ROOT/'units/G0_official/metrics/test.json')
assert [r['sample_id'] for r in old['per_image']]==[r['sample_id'] for r in new['per_image']]
official_replay=dict(historical_metrics_sha256=sha(historical),
    max_per_image_metric_diff={m:max(abs(a[m]-b[m]) for a,b in zip(old['per_image'],new['per_image'])) for m in ('psnr','ssim')})
shutil.copy2(historical,refs/'G0_historical_test.json')
seeds=[2026,2027,2028];families=['G6','K1_aligned','K2_aligned','K1_no_alignment','frozen_G6_adapter']
data={};audits={};fingerprints={};timing={}
def aggregate(rows):
    scenes=sorted({r['scene'] for r in rows})
    return {f'image_mean_{m}':float(np.mean([r[m] for r in rows])) for m in ('psnr','ssim')}|{
        f'scene_macro_{m}':float(np.mean([np.mean([r[m] for r in rows if r['scene']==s]) for s in scenes])) for m in ('psnr','ssim')}
def normalize(d,which='round2'):
    return [dict(sample_id=r['sample_id'],scene=r['scene'],**(r[which] if which in r else {m:r[m] for m in ('psnr','ssim')})) for r in d['per_image']]
for u in lock['units']:
    uid=u['id'];root=ROOT/'units'/uid
    d=read(root/'metrics/test.json');v=read(u['validation_metrics'])
    assert sha(u['validation_metrics'])==u['validation_metrics_sha256']
    if u['checkpoint']:assert sha(u['checkpoint'])==u['checkpoint_sha256']==d['checkpoint_sha256']
    audit=read(root/'diagnostics/inference_audit.json')
    assert audit['passed'] and audit['all_tensor_hashes_before']==audit['all_tensor_hashes_after']
    assert audit['actual_forward_calls']['tasknet']==188*u['actual_rounds']
    assert d['evaluation_seed']==2026 and d['frames']==188 and d['scenes']==8
    assert [r['sample_id'] for r in d['per_image']]==lock['test_sample_order']
    for split,source in [('test',d),('validation',v)]:
        rows=normalize(source)
        assert len(rows)==(188 if split=='test' else 872)
        assert all(np.isfinite(r[m]) for r in rows for m in ('psnr','ssim'))
        name=u['family'];seed=u['seed']
        if name=='G0':
            for s in seeds:data[(split,'G0',s)]=rows
        else:data[(split,name,seed)]=rows
        if name=='K2_aligned':
            raw=read(root/'metrics/inherited_test_raw.json') if split=='test' else source
            data[(split,'K2_T1',seed)]=normalize(raw,'round1')
    shutil.copy2(u['validation_metrics'],refs/(uid+'_validation.json'))
    shutil.copy2(u['source_config'],refs/(uid+'_source_config.yaml'))
    audits[uid]=dict(passed=True,tensors_checked=audit['tensors_checked'],actual_forward_calls=audit['actual_forward_calls'])
    fingerprints[uid]=dict(checkpoint_sha256=u['checkpoint_sha256'],test_metrics_sha256=sha(root/'metrics/test.json'),
        validation_metrics_sha256=u['validation_metrics_sha256'])
    timing[uid]=d['evaluation_seconds']
tables={f'{split}_{name}_seed{seed}':aggregate(rows) for (split,name,seed),rows in data.items()}
def stats(values):
    return dict(values=values,mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)),
        positive_seeds=sum(x>0 for x in values),min=min(values),max=max(values))
comparisons={}
pairs=[('G6','G0'),('K1_aligned','G6'),('K1_no_alignment','K1_aligned'),('K1_no_alignment','G6'),
    ('K2_aligned','K1_aligned'),('K2_aligned','G6'),('frozen_G6_adapter','G6'),('K2_aligned','K2_T1')]
for a,b in pairs:
    comparison={}
    for split in ('test','validation'):
        per_seed={}
        for seed in seeds:
            left,right=data[(split,a,seed)],data[(split,b,seed)]
            assert [(r['sample_id'],r['scene']) for r in left]==[(r['sample_id'],r['scene']) for r in right]
            rows=[dict(sample_id=x['sample_id'],scene=x['scene'],**{m:x[m]-y[m] for m in ('psnr','ssim')}) for x,y in zip(left,right)]
            scenes=sorted({r['scene'] for r in rows})
            per_seed[str(seed)]=dict(aggregate=aggregate(rows),per_image=rows,
                per_scene={s:dict(n=sum(r['scene']==s for r in rows),**{m:float(np.mean([r[m] for r in rows if r['scene']==s])) for m in ('psnr','ssim')}) for s in scenes},
                negative_images={m:sum(r[m]<0 for r in rows) for m in ('psnr','ssim')},
                positive_images={m:sum(r[m]>0 for r in rows) for m in ('psnr','ssim')})
        comparison[split]=dict(per_seed=per_seed,
            paired_seed_summary={k:stats([per_seed[str(seed)]['aggregate'][k] for seed in seeds]) for k in aggregate(rows)},
            scene_seed_summary={s:{m:stats([per_seed[str(seed)]['per_scene'][s][m] for seed in seeds]) for m in ('psnr','ssim')} for s in scenes})
    comparison['cross_split_mean_sign_changed']={k:bool(comparison['test']['paired_seed_summary'][k]['mean']*comparison['validation']['paired_seed_summary'][k]['mean']<0) for k in aggregate(rows)}
    comparisons[a+'_minus_'+b]=comparison
payload=dict(seeds=seeds,test_frames=188,validation_frames=872,scenes_per_split=8,evaluation_seed=2026,
    tables=tables,comparisons=comparisons,audits=audits,source_fingerprints=fingerprints,evaluation_seconds=timing,
    model_seed_summary={split:{name:{k:stats([tables[f'{split}_{name}_seed{seed}'][k] for seed in seeds]) for k in aggregate(rows)} for name in families+['G0','K2_T1']} for split in ('test','validation')},
    limits=lock['limits'],official_reference_replay=official_replay,protocol_lock_sha256=sha(ROOT/'protocol_lock.json'))
(out/'test_completion_summary.json').write_text(json.dumps(payload,indent=2))
print(json.dumps({n:{split:c[split]['paired_seed_summary'] for split in ('test','validation')} for n,c in comparisons.items()},indent=2))
