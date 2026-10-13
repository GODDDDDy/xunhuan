"""Prepare a new fixed-candidate study; never mutate the completed source."""
import hashlib,json,shutil
from pathlib import Path
BASE=Path('/media/li/data3/25thstudent/DY/929work/pidsr_k2_929')
OLD=BASE/'polar_error_guided_short_20261012'
ROOT=BASE/'polar_gain_judgement_short_20261012'
EXPECTED='cecdcfa47d290b0a2d6b662b96bb4407a7043ef5789e9e87d70768f2c683916d'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (ROOT/'budget.json').exists() and not (ROOT/'config/gain_preflight.json').exists()
 for name in ('config','logs','diagnostics','runtime','weights','downstream/checkpoints','downstream/config','downstream/logs','downstream/metrics'):(ROOT/name).mkdir(parents=True,exist_ok=True)
 original=json.loads((OLD/'config/locked_run.json').read_text())
 cfg=json.loads(json.dumps(original).replace(str(OLD),str(ROOT)))
 for k in ('code_sha256','configuration_provenance_sha256','runtime_source_evidence','budget','protocol_id'):cfg.pop(k,None)
 cfg['experiment']='polar_gain_judgement_short_20261012';cfg['protocol_id']='preflight'
 cfg['scope']='fixed experiment21 candidate; train-only predictability diagnosis, conditional one3000-step float-information judgement'
 for name in ('base_server.yaml','manifest.csv','training_schedule_16000.json','validation_proxy.json','training_role_manifest.json'):
  (ROOT/'config'/name).write_text((OLD/'config'/name).read_text().replace(str(OLD),str(ROOT)))
 shutil.copytree(OLD/'source',ROOT/'source',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
 shutil.copytree(OLD/'weights/source_polarfree',ROOT/'weights/source_polarfree',dirs_exist_ok=True)
 for name in ('checkpoints/selected.pt','metrics/final_test.json','convergence.json'):
  p=OLD/'pretraining'/name
  if not p.exists():p=BASE/'polar_enhancer_v2_20261009'/'pretraining'/name
  if p.exists():dest=ROOT/'pretraining'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 dest=ROOT/'weights/reused_final.pt';shutil.copy2(OLD/'downstream/checkpoints/final.pt',dest);assert sha(dest)==EXPECTED
 cfg['reuse'].update(checkpoint=str(dest),sha256=EXPECTED,source_run=str(OLD),source_step=3000,source_candidate_steps=2000,source_judgement_steps=1000,source_protocol_id=original['protocol_id'],new_initialization_only=['seven extra judgement input-channel weights initialized zero'],frozen_entire_run=['all model except action_gate','all PolarFree and prior parameters'],reuse_optimizer=False)
 cfg['downstream'].update(steps=3000,minimum_steps=3000);cfg['downstream'].pop('deadline_unix',None)
 cfg['optimization'].update(candidate_lr=0.,action_judgement_lr=1e-4,core_lr=0.,task_lr=0.)
 for role,path in original['staged']['schedules'].items():
  payload=json.loads(Path(path).read_text())
  if role=='judgement':
   items=payload['items'];payload['items']=(items*3)[:6000];payload['total_steps']=3000
   assert len({x['sample_index'] for x in payload['items']})==1140
  payload.pop('sha256',None);payload['sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  target=ROOT/'config'/f'{role}_schedule.json';target.write_text(json.dumps(payload));cfg['staged']['schedules'][role]=str(target)
 cfg['staged'].update(steps={'candidate':0,'judgement':3000,'bias':original['staged']['steps']['bias'],'safety':original['staged']['steps']['safety']},minimum_steps={'candidate':0,'judgement':3000},deadlines={},candidate_loss='not used; candidate completely frozen',fixed_steps=True)
 cfg['gain_study']=dict(gpu_budget_seconds=28800,evaluation_reserve_seconds=14400,diagnostic_steps={'judgement':128,'bias':64,'safety':128},sampled_cells_per_crop=256,ridge_regularization=.01,coverage_grid=[.01,.05,.1,.2,.5,1.],scales=[4,16,32],decision_coverage=.1,train_gate_rule='raw float ridge top10% safety cost improvement>0 and exceeds same-action-distribution random expectation; at least3of4 safety scenes positive; no validation/test in trigger',formal_change='raw floating T1/proposal/delta plus7 region increment statistics; same inherited body, objective,lr and safety rule',fixed_candidate=True,fixed_q=True,probe_interval=500,training_probe_role='bias, fixed32 crops; monitoring only',no_matched_training_control=True,diagnostic_probes_not_formal_arms=True)
 cfg['experiment_rationale'].update(user_authorization='2026-10-12 开始: fixed-candidate diagnosis then conditional3000 judgement steps',new_run_budget_hours_cap=8,planned_step_cap=3000,matched_control=False)
 cfg.pop('short_trial',None)
 (ROOT/'config/gain_preflight.json').write_text(json.dumps(cfg,indent=2))
 shutil.copy2(OLD/'diagnostics/source_fixed_t1_reference.json',ROOT/'diagnostics/source_fixed_t1_reference.json')
 shutil.copy2(OLD/'diagnostics/final_oracle_validation872.json',ROOT/'diagnostics/source_oracle_validation872.json')
 (ROOT/'status.json').write_text(json.dumps(dict(stage='prepared',source_sha256=EXPECTED)))
 print(json.dumps(dict(prepared=True,source_sha256=EXPECTED,steps=3000,conditional=True)))
if __name__=='__main__':main()
