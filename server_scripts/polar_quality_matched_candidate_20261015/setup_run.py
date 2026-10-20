"""Independent matched candidate continuation; archived experiment22 construction."""
import hashlib,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=ROOT.parent
SOURCE=P/'polar_gain_judgement_short_20261012';INITIAL=P/'polar_error_guided_short_20261012/downstream/checkpoints/initial.pt'
EXPECTED='af2367fd9f96871cad3b94230638cf5c75acd3b8f8c186cdf210a37b730e82d8'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
assert not (ROOT/'config/plan.json').exists(),'Refuse overwrite'
for d in ('config','diagnostics','logs','weights','reports','pretraining/checkpoints','arms/real_quality','arms/fixed_quality'):(ROOT/d).mkdir(parents=True,exist_ok=True)
shutil.copytree(SOURCE/'scripts',ROOT/'scripts',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','controller.py','archive.py','setup_run.py','matched_run.py','summarize.py','quality_forward.py','repair_preflight.py','verify_input_gradient.py'))
shutil.copytree(SOURCE/'source',ROOT/'source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
assert sha(INITIAL)==EXPECTED
(ROOT/'weights/common_initial21.pt').symlink_to(INITIAL)
(ROOT/'weights/migration21_final.pt').symlink_to(SOURCE/'weights/reused_final.pt')
(ROOT/'pretraining/checkpoints/selected.pt').symlink_to(SOURCE/'pretraining/checkpoints/selected.pt')
cfg=json.loads((SOURCE/'config/locked_run.json').read_text());cfg['paths']['output_root']=str(ROOT/'downstream');cfg['data']['workers']=0
write(ROOT/'config/execution_config.json',cfg)
for name in ('locked_run.json','manifest.csv','validation_proxy.json','candidate_schedule.json','judgement_schedule.json','base_server.yaml'):
    shutil.copy2(SOURCE/'config'/name,ROOT/'config'/('source_'+name))
plan=dict(kind='matched candidate continuation, real vs fixed explicit H quality',arms=['real_quality','fixed_quality'],seed=2026,steps_per_arm=2000,
          crop=128,effective_batch=2,microbatch=1,source_initial_sha256=EXPECTED,source_initial_protocol='921ae3dc9fe004949efab2afdb434719a11a8268cb80bd44718bc59a9a99787d',
          trainable=['feedback','error_guidance'],lr=dict(feedback=1e-5,error_guidance=1e-4),
          frozen=['first round','H','enhancer core and old adapters','net_g','prior','projections','both M heads'],
          loss='original staged candidate_loss: MSE + .02 edge MAE + .1 positive local MSE change penalty + .1 Q BCE',
          conditions='all candidate b/C paths; Q recomputed; fixed b0 from common first16 judgement training H predictions; C0=exp(-b0/tau)',
          fixed_calibration=dict(batches=8,phase='quality_constant_calibration'),candidate_schedule_sha256=sha(SOURCE/'config/candidate_schedule.json'),
          candidate_training_images=3640,all5440_seen_in_inherited_history=True,
          checkpoint_interval=500,monitor_interval=500,monitor_fixed_training_crops=4,
          evaluation=dict(validation=872,test=188,outputs=['common frozen T1','real U','fixed U'],original_size=True,tiling=False,shared_first=True,seed_phase='matched_eval'),
          endpoint='fixed2000 per arm or recorded immutable stage cutoff; incomplete if either below2000',
          budget_seconds=43200,evaluation_reserve_seconds=14400,resource_rule='1.5*(4000*median_train_step + 1060*paired_full_frame_seconds) plus1800s must fit12h; preflight discarded',
          success_interpretation='real-control AND real-T1 PSNR/SSIM, scene directions; one seed short continuation not stability or convergence proof',
          source_quality_training_history=True,matched_from_scratch=False,test_used_for_selection=False,validation_used_for_selection=False,
          execution_config_sha256=sha(ROOT/'config/execution_config.json'))
plan['code_manifest']=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted((ROOT/'scripts').rglob('*.py'))]
plan['protocol_id']=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
write(ROOT/'config/plan.json',plan);write(ROOT/'status.json',dict(stage='prepared_cpu_pending',time_unix=time.time(),protocol_id=plan['protocol_id']))
print(json.dumps(dict(prepared=True,protocol_id=plan['protocol_id'])))
