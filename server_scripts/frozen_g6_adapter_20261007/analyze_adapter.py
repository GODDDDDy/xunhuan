"""Audit frozen controls and retain every paired image difference, including negatives."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parent.parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
lock=read(ROOT/'protocol_lock.json')
assert read(ROOT/'status.json')['state']=='complete'
assert all(sha(p)==h for p,h in lock['locked_file_sha256'].items())
OUT=ROOT/'analysis';OUT.mkdir(exist_ok=True)
REF=OUT/'references';REF.mkdir(exist_ok=True)
seeds=lock['seeds'];data={};tables={};fingerprints={};audits={};timing={}
ids=None
for seed in seeds:
    run=ROOT/'runs/K1'/f'seed_{seed}'
    pf=read(run/'diagnostics/recurrent_preflight.json')
    assert pf['passed'] and pf['actual_forward_calls']=={'r2p':1,'tasknet':1}
    assert pf['initial_polar_identity'] and pf['initial_delta_max']==0
    assert pf['grad_norm_netg']==0 and pf['no_frozen_task_gradients'] and pf['other_model_gradients_absent']
    assert pf['proxy_step0_parity']['n']==28 and pf['proxy_step0_parity']['all_passed']
    assert read(run/'diagnostics/gate_step_200.json')['passed'] and read(run/'diagnostics/gate_step_3000.json')['passed']
    checked=[]
    for step in range(0,3001,200):
        assert read(run/f'diagnostics/full_frozen_step_{step}.json')['passed']
        assert read(run/f'diagnostics/frozen_step_{step}.json')['passed']
        checked.append(step)
    rows=list(csv.DictReader((run/'logs/train.csv').open()))
    assert [int(r['step']) for r in rows]==list(range(1,3001))
    assert all(np.isfinite(float(v)) for r in rows for v in r.values())
    assert all(float(r['tasknet_grad_norm'])==float(r['projection_grad_norm'])==float(r['feature_alignment_loss'])==0 for r in rows)
    history=read(run/'metrics/proxy_validation.json')
    assert all(r['tasknet_relative_l2_drift']==0 for r in history)
    audits[str(seed)]={'passed':True,'all_frozen_checked_steps':checked,'step0_full_resolution_replay':pf['proxy_step0_parity'],
        'input_gradient_norm_at_step0':pf['input_gradient_norm'],'r2p_gradient_norm_at_step0':pf['grad_norm_r2p'],
        'adam_states':18,'TaskNet_parameters_and_buffers_unchanged':True,'netg_drift_all_proxy_points':0}
    ref=lock['references'][f'G6_seed{seed}']
    assert all(sha(v['path'])==v['sha256'] for v in ref.values())
    for k,v in ref.items():
        if k!='checkpoint':shutil.copy2(v['path'],REF/f'G6_seed{seed}_{k}{Path(v["path"]).suffix}')
    for name,path,ck in [('adapter',run/'metrics/step_3000_validation.json',run/'checkpoints/step_3000.pt'),
                         ('G6',Path(ref['metrics']['path']),Path(ref['checkpoint']['path']))]:
        d=read(path)
        assert d['frames']==872 and d['scenes']==8 and d['checkpoint_step']==3000 and d['phase_weight']==0.1
        assert sha(ck)==d['checkpoint_sha256']
        if (name,seed)==('G6',2026):
            # Original evaluator records a single seed used for both training and validation.
            # Preserve that old schema; substantiate interpretation with its actual source/config.
            assert d['seed']==2026 and d['split']=='validation' and d['group']=='G6_phase01'
            old_study=Path(ref['checkpoint']['path']).parents[4]
            config=old_study/'configs/G6_phase01.yaml'
            assert yaml.safe_load(config.read_text())['seed']==2026
            provenance={}
            for p in (config,old_study/'scripts/evaluate_detailed.py',old_study/'scripts/run_g6_phase.py',
                      old_study/'scripts/reference/k1_ablation_snapshot.py'):
                target=REF/('G6_seed2026_provenance_'+p.name)
                shutil.copy2(p,target);provenance[str(p)]=sha(p)
            audits[str(seed)]['legacy_G6_evaluation_seed_provenance']=provenance
        else:
            assert d['evaluation_seed']==2026 and d['training_seed']==seed
        if name=='adapter':
            assert d['experiment']=='frozen_G6_identity_adapter' and not d['netg_trained_this_stage']
            assert d['frozen_G6_checkpoint_sha256']==ref['checkpoint']['sha256']
            assert d['actual_tasknet_forwards']==872 and d['actual_rounds']==1 and d['alignment_weight']==0
            timing[str(seed)]={'full_validation_seconds':d['evaluation_seconds'],
                'summed_training_step_seconds':sum(float(r['step_seconds']) for r in rows),
                'initial_preflight_peak_vram_mb':pf['peak_vram_mb'],
                'initial_and_final_crop_delta_abs_mean':[float(rows[0]['delta_abs_mean']),float(rows[-1]['delta_abs_mean'])]}
            raw=read(run/'metrics/inherited_evaluator_K1_aliases.json')
            timing[str(seed)]['full_validation_effective_polar_update']=raw['polar_update']['polar1']
        this_ids=[(r['sample_id'],r['scene']) for r in d['per_image']]
        assert len(this_ids)==len(set(this_ids))==872
        if ids is None:ids=this_ids
        assert this_ids==ids
        data[(name,seed)]=d['per_image']
        fingerprints[f'{name}_seed{seed}']={'metrics_path':str(path),'metrics_sha256':sha(path),'checkpoint_sha256':sha(ck)}
    old_rows=list(csv.DictReader(Path(ref['training_log']['path']).open()))
    assert [(r['step'],r['diffusion_seed']) for r in rows]==[(r['step'],r['diffusion_seed']) for r in old_rows]
scenes=sorted({scene for _,scene in ids})
keys=['image_mean_psnr','scene_macro_psnr','image_mean_ssim','scene_macro_ssim']
def aggregate(rows):
    return {f'image_mean_{m}':float(np.mean([r[m] for r in rows])) for m in ('psnr','ssim')}|{
        f'scene_macro_{m}':float(np.mean([np.mean([r[m] for r in rows if r['scene']==scene]) for scene in scenes])) for m in ('psnr','ssim')}
def stats(v):
    return {'values':v,'mean':float(np.mean(v)),'sample_sd':float(np.std(v,ddof=1)),
        'positive_seeds':sum(x>0 for x in v),'min':min(v),'max':max(v)}
for k,rows in data.items():tables[k]=aggregate(rows)
paired={}
for seed in seeds:
    rows=[{'sample_id':a['sample_id'],'scene':a['scene'],**{m:a[m]-b[m] for m in ('psnr','ssim')}}
        for a,b in zip(data[('adapter',seed)],data[('G6',seed)])]
    paired[str(seed)]={'aggregate':aggregate(rows),'per_image':rows,
        'per_scene':{scene:{'n':sum(r['scene']==scene for r in rows),**{m:float(np.mean([r[m] for r in rows if r['scene']==scene])) for m in ('psnr','ssim')}} for scene in scenes},
        'negative_images':{m:sum(r[m]<0 for r in rows) for m in ('psnr','ssim')},
        'positive_images':{m:sum(r[m]>0 for r in rows) for m in ('psnr','ssim')},
        'difficulty_posthoc':{level:{'n':sum('/'+level+'/' in r['sample_id'] for r in rows),**{m:float(np.mean([r[m] for r in rows if '/'+level+'/' in r['sample_id']])) for m in ('psnr','ssim')}} for level in ('easy','hard')}}
comparison={'per_seed':paired,'paired_seed_summary':{k:stats([paired[str(seed)]['aggregate'][k] for seed in seeds]) for k in keys},
    'scene_seed_summary':{scene:{m:stats([paired[str(seed)]['per_scene'][scene][m] for seed in seeds]) for m in ('psnr','ssim')} for scene in scenes}}
payload={'seeds':seeds,'frames':872,'scenes':8,'evaluation_seed':2026,
    'scope':'Staged frozen trained G6 adaptation; additional budget; exploratory validation',
    'per_seed_tables':{f'{name}_seed{seed}':t for (name,seed),t in tables.items()},
    'model_seed_summary':{name:{k:stats([tables[(name,seed)][k] for seed in seeds]) for k in keys} for name in ('adapter','G6')},
    'comparison_adapter_minus_G6':comparison,'audits':audits,'source_fingerprints':fingerprints,
    'timing':timing,'protocol_sha256':sha(ROOT/'protocol_lock.json'),'limits':lock['limits']}
(OUT/'frozen_adapter_paired_summary.json').write_text(json.dumps(payload,indent=2))
print(json.dumps({'paired_differences':comparison['paired_seed_summary'],
    'scenes':comparison['scene_seed_summary'],'difficulty':{s:p['difficulty_posthoc'] for s,p in paired.items()},
    'timing':timing},indent=2))
