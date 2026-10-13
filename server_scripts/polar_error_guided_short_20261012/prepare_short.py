"""Copy audited dependencies; never alter the completed source experiment."""
import copy,json,math,shutil,hashlib
from pathlib import Path
BASE=Path('/media/li/data3/25thstudent/DY/929work/pidsr_k2_929')
OLD=BASE/'polar_staged_acceptance_joint_20261012'
ROOT=BASE/'polar_error_guided_short_20261012'
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def main():
 assert not (ROOT/'budget.json').exists() and not (ROOT/'config/short_preflight.json').exists()
 for name in ('config','logs','diagnostics','runtime','weights','downstream/checkpoints','downstream/config','downstream/logs','downstream/metrics'):(ROOT/name).mkdir(parents=True,exist_ok=True)
 cfg=json.loads((OLD/'config/locked_run.json').read_text())
 for k in ('code_sha256','configuration_provenance_sha256','runtime_source_evidence','budget','protocol_id'):cfg.pop(k,None)
 for name in ('base_server.yaml','manifest.csv','training_schedule_16000.json','validation_proxy.json','training_role_manifest.json'):
  p=OLD/'config'/name
  (ROOT/'config'/name).write_text(p.read_text().replace(str(OLD),str(ROOT)))
 shutil.copytree(OLD/'source',ROOT/'source',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
 for name in ('checkpoints/selected.pt','metrics/final_test.json','convergence.json'):
  original=OLD/'pretraining'/name
  if not original.exists():original=BASE/'polar_enhancer_v2_20261009'/'pretraining'/name
  if original.exists():
   p=ROOT/'pretraining'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,p)
 shutil.copytree(OLD/'weights/source_polarfree',ROOT/'weights/source_polarfree',dirs_exist_ok=True)
 source=ROOT/'weights/reused_final.pt';shutil.copy2(OLD/'downstream/checkpoints/final.pt',source)
 expected='e79af90811baa46aadfb354a682d0c5dad93886e2dfe1eabeaceb407e6cae112';assert digest(source)==expected
 cfg['paths']={k:v.replace(str(OLD),str(ROOT)) if isinstance(v,str) else v for k,v in cfg['paths'].items()}
 cfg['reuse'].update(checkpoint=str(source),sha256=expected,source_run=str(OLD),source_step=6000,source_protocol_id='081f080ec86179ba1148e64ac470613d182ba4cec377af280b7edc7e38af0dd6',new_initialization_only=['repair_need_head','zero_second_round_state_injections'])
 cfg['task_initialization']['archived_source_dir']=str(ROOT/'weights/source_polarfree')
 counts=cfg['staged']['sample_counts'];steps=dict(candidate=2000,judgement=1000)
 for role,path in cfg['staged']['schedules'].items():
  payload=json.loads(Path(path).read_text())
  if role in steps:
   payload['items']=payload['items'][:2*steps[role]];payload['total_steps']=steps[role]
   assert len({x['sample_index'] for x in payload['items']})==counts[role]
  payload.pop('sha256',None);payload['sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  destination=ROOT/'config'/f'{role}_schedule.json';destination.write_text(json.dumps(payload));cfg['staged']['schedules'][role]=str(destination)
 steps.update({r:json.loads(Path(cfg['staged']['schedules'][r]).read_text())['total_steps'] for r in ('bias','safety')})
 cfg['staged'].update(steps=steps,minimum_steps=dict(candidate=2000,judgement=1000),deadlines={},candidate_loss='same source RGB MSE+.02 edge MAE+.1 harmful local MSE; add .1 repair-need BCE',fixed_steps=True)
 cfg['downstream'].update(steps=3000,minimum_steps=3000);cfg['downstream'].pop('deadline_unix',None)
 cfg['error_guidance']=dict(target='inherited train-only local-error threshold; repair need proxy, not reflection GT',stop_candidate_gradients_to_q=True,zero_state_injections=True,guidance_lr=1e-4,only_round2=True)
 cfg['short_trial']=dict(total_steps=3000,gpu_budget_seconds=28800,evaluation_reserve_seconds=14400,source_oracle_validation=872,source_oracle_scales=[4,16,32],formal_success='T2-T1 validation PSNR>0 and SSIM>=0 with nonzero update coverage; report scenes and all test regardless',pilot_not_convergence_proof=True,matched_continued_training_control=False)
 cfg['protocol_id']='preflight'
 cfg['experiment']='polar_error_guided_short_20261012'
 (ROOT/'config/short_preflight.json').write_text(json.dumps(cfg,indent=2))
 shutil.copy2(OLD/'diagnostics/source_fixed_t1_reference.json',ROOT/'diagnostics/source_fixed_t1_reference.json')
 (ROOT/'status.json').write_text(json.dumps(dict(stage='prepared',source_sha256=expected)))
 print(json.dumps(dict(prepared=True,source_sha256=expected,steps=steps,sample_counts=counts)))
if __name__=='__main__':main()
