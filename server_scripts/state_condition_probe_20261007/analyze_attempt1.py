"""Pair all locked pilot outcomes without choosing a seed or checkpoint."""
import csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent.parent
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
lock=read(ROOT/'training_lock.json');assert read(ROOT/'pilot_status.json')['state']=='complete'
assert all(sha(p)==h for p,h in lock['locked_file_sha256'].items())
assert sha(ROOT/'diagnostics/diagnostic_result.json')==lock['diagnostic_result_sha256']
arms=lock['conditions'];keys=['image_mean_psnr','scene_macro_psnr','image_mean_ssim','scene_macro_ssim']
source=lock['G6_explicit_source_receipt'];g6=read(source['validation_metrics'])
data={'G6':g6['per_image']};audits={};metrics={};logs={};initials={};polar_summaries={}
for arm in arms:
    run=ROOT/'runs'/arm/'seed_2026';m=read(run/'metrics/step_1000_validation.json')
    assert m['frames']==872 and m['scenes']==8 and m['actual_tasknet_forwards']==1744 and m['actual_r2p_forwards']==872
    assert m['checkpoint_step']==1000 and m['evaluation_seed']==2026 and m['training_seed']==2026
    assert m['condition_probe']==arm and m['full_task_frozen'] and m['inference_r2p_hashes_unchanged']
    assert m['baseline_replay_max_diff']['psnr']<=1e-5 and m['baseline_replay_max_diff']['ssim']<=1e-7
    cp_path=run/'checkpoints/step_1000.pt';assert sha(cp_path)==m['checkpoint_sha256']
    cp=torch.load(cp_path,map_location='cpu',weights_only=False)
    g6_cp=torch.load(source['checkpoint'],map_location='cpu',weights_only=False)
    assert all(torch.equal(t,g6_cp['tasknet_net_g'][n]) for n,t in cp['tasknet_net_g'].items())
    assert cp['effective_task_weights']==[0.0,1.0] and cp['condition_probe']==arm
    assert len(cp['optimizer']['state'])==18 and all(int(v['step'])==1000 for v in cp['optimizer']['state'].values())
    initial=read(run/'diagnostics/initial_pair_audit.json');initials[arm]=initial
    assert all(read(p)['passed'] for p in (run/'diagnostics').glob('full_frozen_step_*.json'))
    assert read(run/'diagnostics/recurrent_preflight.json')['passed']
    assert read(run/'diagnostics/gate_step_200.json')['passed'] and read(run/'diagnostics/gate_step_1000.json')['passed']
    rows=list(csv.DictReader((run/'logs/train.csv').open()));assert [int(r['step']) for r in rows]==list(range(1,1001))
    logs[arm]=rows;data[arm]=m['per_image'];metrics[arm]=m
    assert [r['sample_id'] for r in data[arm]]==[r['sample_id'] for r in data['G6']]
    assert all(np.isfinite(r[k]) for r in data[arm] for k in ('psnr','ssim'))
    audits[arm]=dict(passed=True,netg_exact_source_G6=True,all_saved_frozen_audits_passed=True,
        optimizer_active_tensors=18,initial_zero_output=True,baseline_replay_max_diff=m['baseline_replay_max_diff'],
        actual_tasknet_forwards=1744,actual_r2p_forwards=872,checkpoint_sha256=m['checkpoint_sha256'],
        source_G6_sha256=source['checkpoint_sha256'],condition_clip_fraction=m['condition_prediction_clip_fraction_mean'])
    polar_summaries[arm]={k:float(np.mean([r[k] for r in m['polar_update']]))
        for k in ('mean_abs','relative_signal_change','ratio_gt_0_02','ratio_gt_0_05')}
    del cp,g6_cp
assert initials[arms[0]]['model']==initials[arms[1]]['model']
assert initials[arms[0]]['netg']==initials[arms[1]]['netg']
assert [r['diffusion_seed'] for r in logs[arms[0]]]==[r['diffusion_seed'] for r in logs[arms[1]]]
def aggregate(rows):
    scenes=sorted({r['scene'] for r in rows})
    return {f'image_mean_{m}':float(np.mean([r[m] for r in rows])) for m in ('psnr','ssim')}|{
        f'scene_macro_{m}':float(np.mean([np.mean([r[m] for r in rows if r['scene']==s]) for s in scenes])) for m in ('psnr','ssim')}
comparisons={}
for a,b in [('mixed_rgb','G6'),('task_prediction','G6'),('task_prediction','mixed_rgb')]:
    left,right=data[a],data[b]
    assert [(r['sample_id'],r['scene']) for r in left]==[(r['sample_id'],r['scene']) for r in right]
    rows=[dict(sample_id=x['sample_id'],scene=x['scene'],difficulty=x['sample_id'].split('/')[1],
        **{m:x[m]-y[m] for m in ('psnr','ssim')}) for x,y in zip(left,right)]
    scenes=sorted({r['scene'] for r in rows})
    comparisons[a+'_minus_'+b]=dict(aggregate=aggregate(rows),per_image=rows,
        negative_images={m:sum(r[m]<0 for r in rows) for m in ('psnr','ssim')},
        positive_images={m:sum(r[m]>0 for r in rows) for m in ('psnr','ssim')},
        per_scene={s:dict(n=sum(r['scene']==s for r in rows),**{m:float(np.mean([r[m] for r in rows if r['scene']==s])) for m in ('psnr','ssim')}) for s in scenes},
        per_difficulty={d:dict(n=sum(r['difficulty']==d for r in rows),**{m:float(np.mean([r[m] for r in rows if r['difficulty']==d])) for m in ('psnr','ssim')}) for d in ('easy','hard')})
promotion=all(comparisons['task_prediction_minus_'+b]['aggregate'][k]>0 for b in ('G6','mixed_rgb') for k in keys)
diag=read(ROOT/'diagnostics/diagnostic_result.json')
round_updates={str(i):{k:float(np.mean([r[f'round{i}_delta'][k] for r in diag['per_image']]))
    for k in ('raw_delta_mean_abs','effective_mean_abs','clamp_fraction')}
    | {'clamped_identity_all_exact':all(r[f'round{i}_delta']['clamped_identity_max_diff']==0 for r in diag['per_image'])}
    for i in (1,2)}
summary=dict(scope='one paired training seed2026, additional1000 adapter steps, validation872 only',
    aggregates={a:aggregate(v) for a,v in data.items()},comparisons=comparisons,audits=audits,
    pilot_effective_polar_update_image_means=polar_summaries,
    paired_initial_tensors_equal=True,all1000_logged_RNG_seeds_equal=True,promotion_gate_passed=promotion,
    diagnostic=dict(delta_summary=diag['delta_summary'],stage_changed_images=diag['stage_changed_images'],
        training_gate_passed=diag['training_gate_passed'],independent_endpoint_checks=56,
        actual_forward_calls=diag['actual_forward_calls'],all_frozen=diag['all_frozen'],
        difficulty_counts=diag['difficulty_counts'],elapsed_seconds=diag['elapsed_seconds'],
        round_update_image_means=round_updates,
        stage_mean_abs={k:float(np.mean([r['second_round_path'][k]['mean_abs'] for r in diag['per_image']])) for k in diag['stage_changed_images']},
        gray_changed_fraction_mean=float(np.mean([r['second_round_path']['gray']['changed_fraction'] for r in diag['per_image']])),
    source_G6=source,protocol_lock_sha256=sha(ROOT/'protocol_lock.json'),training_lock_sha256=sha(ROOT/'training_lock.json'),
    limitations=['One training seed; no variance/CI/equivalence/significance estimate.',
        'Exploratory validation used in earlier work; not independent confirmation.',
        'Only1000 steps, no convergence or architecture-wide invalidity conclusion.',
        'Different condition distribution and [0,1] clamp are part of the representation intervention.',
        'Only mean gains are promotion screen; preserve individual scenes and SSIM.'])
out=ROOT/'analysis';refs=out/'references';refs.mkdir(exist_ok=True)
shutil.copy2(source['validation_metrics'],refs/'G6_seed2026_validation.json')
(out/'condition_probe_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ('comparisons','audits')},indent=2))
