"""Audit three lambda=0 runs and summarize within-seed contrasts to locked controls."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
lock=read(ROOT/'protocol_lock.json')
assert read(ROOT/'status.json')['state']=='complete'
assert all(sha(p)==h for p,h in lock['locked_file_sha256'].items())
OUT=ROOT/'analysis'
OUT.mkdir(exist_ok=True)
REF=OUT/'references'
REF.mkdir(exist_ok=True)
seeds=lock['seeds']
data,tables,fingerprints,audits={},{},{},{}
ids=None
for seed in seeds:
    paired_logs={}
    for name in ('K1_no_alignment','K1_aligned','G6'):
        if name=='K1_no_alignment':
            run=ROOT/'runs/K1'/f'seed_{seed}'
            path=run/'metrics/step_3000_validation.json'
            train=run/'logs/train.csv'
            ck=run/'checkpoints/step_3000.pt'
            d=read(path)
            assert d['training_seed']==seed and d['evaluation_seed']==2026
            assert d['actual_tasknet_forwards']==872 and d['actual_rounds']==1
            assert d['effective_task_weights']==[1.0]
            assert read(run/'diagnostics/gate_step_3000.json')['passed']
            init=read(run/'diagnostics/initial_pair_audit.json')
            assert init['paired_aligned_initialization_matched']
            pf=read(run/'diagnostics/recurrent_preflight.json')
            assert pf['passed'] and pf['lambda_feature_alignment']==0 and pf['grad_norm_projections']==0
            for step in range(200,3001,200):
                assert read(run/f'diagnostics/frozen_step_{step}.json')['passed']
                assert read(run/f'diagnostics/projection_unchanged_step_{step}.json')['passed']
            audits[f'{name}_seed{seed}']={'passed':True,'projection_checked_save_points':15,
                'task_gradient_max_diff_at_preflight':pf['task_only_vs_total_r2p_gradient_max_diff']}
        else:
            reference=lock['references'][f'{"K1" if name=="K1_aligned" else "G6"}_seed{seed}']
            assert all(sha(v['path'])==v['sha256'] for v in reference.values())
            path,train,ck=[Path(reference[k]['path']) for k in ('metrics','training_log','checkpoint')]
            d=read(path)
            for key,v in reference.items():
                if key!='checkpoint':
                    target=REF/f'{name}_seed{seed}_{key}{Path(v["path"]).suffix}'
                    shutil.copy2(v['path'],target)
        assert d['checkpoint_step']==3000 and d['phase_weight']==0.1
        assert d['frames']==872 and d['scenes']==8
        assert sha(ck)==d['checkpoint_sha256']
        this_ids=[(r['sample_id'],r['scene']) for r in d['per_image']]
        assert len(this_ids)==len(set(this_ids))==872
        if ids is None:ids=this_ids
        assert this_ids==ids
        rows=list(csv.DictReader(train.open()))
        assert [int(r['step']) for r in rows]==list(range(1,3001))
        assert all(np.isfinite(float(v)) for r in rows for v in r.values())
        paired_logs[name]=[(r['step'],r['diffusion_seed']) for r in rows]
        data[(name,seed)]=d['per_image']
        fingerprints[f'{name}_seed{seed}']={'metrics_path':str(path),'metrics_sha256':sha(path),'checkpoint_sha256':sha(ck)}
    assert paired_logs['K1_no_alignment']==paired_logs['K1_aligned']==paired_logs['G6']
scenes=sorted({scene for _,scene in ids})
keys=['image_mean_psnr','scene_macro_psnr','image_mean_ssim','scene_macro_ssim']
def aggregate(rows):
    return {f'image_mean_{m}':float(np.mean([r[m] for r in rows])) for m in ('psnr','ssim')}|{
        f'scene_macro_{m}':float(np.mean([np.mean([r[m] for r in rows if r['scene']==scene]) for scene in scenes])) for m in ('psnr','ssim')}
def stats(v):
    return {'values':v,'mean':float(np.mean(v)),'sample_sd':float(np.std(v,ddof=1)),
            'positive_seeds':sum(x>0 for x in v),'min':min(v),'max':max(v)}
for key,rows in data.items():tables[key]=aggregate(rows)
comparisons={}
for a,b in [('K1_no_alignment','K1_aligned'),('K1_no_alignment','G6'),('K1_aligned','G6')]:
    per_seed={}
    for seed in seeds:
        rows=[{'sample_id':x['sample_id'],'scene':x['scene'],**{m:x[m]-y[m] for m in ('psnr','ssim')}}
              for x,y in zip(data[(a,seed)],data[(b,seed)])]
        per_seed[str(seed)]={'aggregate':aggregate(rows),'per_image':rows,
           'per_scene':{scene:{'n':sum(r['scene']==scene for r in rows),**{m:float(np.mean([r[m] for r in rows if r['scene']==scene])) for m in ('psnr','ssim')}} for scene in scenes},
           'negative_images':{m:sum(r[m]<0 for r in rows) for m in ('psnr','ssim')},
           'positive_images':{m:sum(r[m]>0 for r in rows) for m in ('psnr','ssim')},
           'difficulty_posthoc':{level:{'n':sum('/'+level+'/' in r['sample_id'] for r in rows),**{m:float(np.mean([r[m] for r in rows if '/'+level+'/' in r['sample_id']])) for m in ('psnr','ssim')}} for level in ('easy','hard')}}
    comparisons[a+'_minus_'+b]={'per_seed':per_seed,
        'paired_seed_summary':{k:stats([per_seed[str(seed)]['aggregate'][k] for seed in seeds]) for k in keys},
        'scene_seed_summary':{scene:{m:stats([per_seed[str(seed)]['per_scene'][scene][m] for seed in seeds]) for m in ('psnr','ssim')} for scene in scenes}}
payload={'seeds':seeds,'frames':872,'scenes':8,'evaluation_seed':2026,
         'scope':'Three paired training seeds, one-factor alignment-supervision ablation, exploratory validation',
         'per_seed_tables':{f'{name}_seed{seed}':t for (name,seed),t in tables.items()},
         'model_seed_summary':{name:{k:stats([tables[(name,seed)][k] for seed in seeds]) for k in keys} for name in ('K1_no_alignment','K1_aligned','G6')},
         'comparisons':comparisons,'audits':audits,'source_fingerprints':fingerprints,
         'protocol_sha256':sha(ROOT/'protocol_lock.json'),'limits':lock['limits']}
(OUT/'alignment_paired_summary.json').write_text(json.dumps(payload,indent=2))
print(json.dumps({'model_seed_summary':payload['model_seed_summary'],
                  'paired_differences':{n:c['paired_seed_summary'] for n,c in comparisons.items()}},indent=2))
