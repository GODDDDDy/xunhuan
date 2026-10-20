"""Prepare one pre-registered two-arm screen; never mutate an old run."""
import hashlib,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=ROOT.parent
OLD=P/'polar_quality_matched_candidate_20261015';AUDIT=P/'polar_quality_mechanism_audit_20261015'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
assert not (ROOT/'config/plan.json').exists(),'Refuse overwrite'
assert (AUDIT/'reports/completion_receipt.json').exists(),'Complete review first'
for d in ['config','logs','diagnostics','reports','weights','pretraining/checkpoints','downstream','arms/proxy_only','arms/task_coupled']:(ROOT/d).mkdir(parents=True,exist_ok=True)
shutil.copytree(OLD/'scripts',ROOT/'scripts',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copytree(OLD/'source',ROOT/'source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
# Unique entry names avoid the historical generic-controller copy collision.
for n in ['common_initial21.pt','migration21_final.pt']:(ROOT/'weights'/n).symlink_to((OLD/'weights'/n).resolve())
(ROOT/'pretraining/checkpoints/selected.pt').symlink_to((OLD/'pretraining/checkpoints/selected.pt').resolve())
cfg=json.loads((OLD/'config/execution_config.json').read_text());cfg['paths']['output_root']=str(ROOT/'downstream')
cfg['experiment']='polar_task_guidance_gradient_short_20261015';cfg['scope']='exploratory Q gradient coupling; explicit plan overrides inherited genealogy fields'
cfg['evaluation']=dict(validation_frames=64,test_frames=0,original_size=True,tiles=False,exploratory=True)
cfg['downstream'].update(steps=2000,minimum_steps=2000)
cfg['downstream'].pop('deadline_unix',None);cfg.pop('budget',None)
cfg['optimization']=dict(candidate_lr=1e-5,guidance_lr=1e-4,core_lr=0.,task_lr=0.)
cfg['staged']['candidate_loss']='MSE+.02 edge MAE+.1 positive local MSE change+.1 Q BCE'
cfg['staged']['steps']['candidate']=2000;cfg['staged']['steps']['judgement']=0
cfg['error_guidance']['stop_candidate_gradients_to_q']='arm-specific; see plan'
write(ROOT/'config/execution_config.json',cfg)
shutil.copy2(OLD/'config/fixed_calibration.json',ROOT/'config/fixed_calibration.json')
shutil.copy2(AUDIT/'reports/fast_screen64_preregistered.json',ROOT/'config/screen64.json')
assert sha(ROOT/'config/screen64.json')=='257a0b2abd704b9b1cbc7ef35e4a057b12631ab96b832097d17a334d18258c60'
shutil.copy2(OLD/'config/plan.json',ROOT/'config/source27_plan.json')
s=ROOT/'scripts/error_guidance.py';text=s.read_text()
before='logits=self.predict(torch.cat(maps,1)); need=logits.sigmoid().detach()'
after='logits=self.predict(torch.cat(maps,1)); probability=logits.sigmoid()\n        need=probability if getattr(self,"task_coupled",False) else probability.detach()'
assert text.count(before)==1;text=text.replace(before,after)
text=text.replace('# Q cannot alter its labels or obtain candidate-loss gradients by changing\n        # its confidence; GT appears only in the supervised loss, outside forward.','# Arm flag changes only the backward connection. Inputs and GT labels stay detached.\n        # Coupled Q is a task control map with auxiliary error supervision.')
s.write_text(text)
plan=dict(kind='exploratory paired Q gradient screen, not formal confirmation',arms=['proxy_only','task_coupled'],seed=2026,
    source_initial_sha256='af2367fd9f96871cad3b94230638cf5c75acd3b8f8c186cdf210a37b730e82d8',
    source_initial_protocol='921ae3dc9fe004949efab2afdb434719a11a8268cb80bd44718bc59a9a99787d',
    source_quality_training_history=True,optimizer_reused=False,RNG_reused=False,
    steps_per_arm=2000,candidate_training_images=3640,crop=128,effective_batch=2,microbatch=1,
    single_change='Q sigmoid detach versus connected, all other numeric forward and losses identical',
    lr=dict(feedback=1e-5,error_guidance=1e-4),loss='unchanged MSE+.02 edge MAE+.1 harmful local MSE penalty+.1 Q BCE',
    conditions='both arms real H b/C on all paths; H and T1 frozen; no M execution',
    budget_seconds=14400,evaluation_reserve_seconds=3600,
    evaluation=dict(validation=64,test=0,original_size=True,tiling=False,seed_phase='matched_eval_validation',original_full_validation_index=True),
    preliminary_success='scene-equal coupled U beats T1 and proxy PSNR, SSIM not lower than either; coupled-minus-proxy PSNR positive in at least6of8 scenes; nonzero output change',
    partial_signal='beats proxy but not T1 is partial only; do not declare confidence effective',
    endpoint='fixed2000 per arm; if deadline below2000 explicitly incomplete; never select from validation',
    screen64_sha256=sha(ROOT/'config/screen64.json'),candidate_schedule_sha256=sha(Path(cfg['staged']['schedules']['candidate'])),
    execution_config_sha256=sha(ROOT/'config/execution_config.json'),
    resource_rule='1.5*(4000*step_seconds+64*paired_frame_seconds)+900 <=4h; discard preflight updates',
    literature='MPRNet CVPR2021 section3 eq1 and3.3 support task-supervised multi-stage features, not this hypothesis',
    limitations=['single seed','short continuation no convergence proof','inherited quality training','repeated validation','no formal test in screen','not independent H value test'])
plan['code_manifest']=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted((ROOT/'scripts').rglob('*.py'))]
plan['protocol_id']=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
write(ROOT/'config/plan.json',plan);write(ROOT/'status.json',dict(stage='prepared_cpu_pending',protocol_id=plan['protocol_id'],time_unix=time.time()))
print(json.dumps(dict(prepared=True,protocol_id=plan['protocol_id'])))
