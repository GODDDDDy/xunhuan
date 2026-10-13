"""Immutable eight-hour budget; diagnose before allowing the one formal fit."""
import hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from common import ROOT,json_save,status,sha
env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT.parent.parent/'tmp_pgj'),XDG_CACHE_HOME=str(ROOT/'runtime/cache'),TORCH_HOME=str(ROOT.parent/'runtime/cache/torch'))
env['PYTHONPATH']=str(ROOT/'source')+os.pathsep+env.get('PYTHONPATH','')
Path(env['TMPDIR']).mkdir(parents=True,exist_ok=True)
def stage(action):
 if (ROOT/'budget.json').exists():assert time.time()<json.loads((ROOT/'budget.json').read_text())['deadline_unix']
 local=dict(env)
 if action=='verify':local['CUDA_VISIBLE_DEVICES']=''
 with (ROOT/'logs'/f'{action}.log').open('a') as f:
  p=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/run_gain.py'),action],cwd=ROOT,env=local,stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError(f'{action} failed with exit {p.returncode}')
def run():
 assert not (ROOT/'budget.json').exists() and not (ROOT/'config/locked_run.json').exists(),'Do not duplicate or reset this experiment'
 stage('verify')
 now=time.time();budget=dict(start_unix=now,deadline_unix=now+28800,pgid=os.getpgrp(),process_group=os.getpgrp(),evaluation_reserve_seconds=14400,scope='diagnosis and conditional one3000-step judgement; source21 candidate frozen; immutable8h')
 json_save(ROOT/'budget.json',budget)
 with (ROOT/'logs/watchdog.log').open('a') as f:wd=subprocess.Popen([sys.executable,str(ROOT/'scripts/watchdog.py')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 json_save(ROOT/'watchdog.json',dict(pid=wd.pid,registered_pgid=budget['pgid'],deadline_unix=budget['deadline_unix']))
 stage('benchmark');measured=json.loads((ROOT/'diagnostics/gain_benchmark.json').read_text())
 cfg=json.loads((ROOT/'config/gain_preflight.json').read_text());reserve=max(14400.,1.3*1300*measured['full_image_seconds'])
 training_end=budget['deadline_unix']-reserve
 required=1.4*(872*measured['full_image_seconds']+3000*measured['judgement_step_seconds'])+1800
 if training_end-time.time()<required:status('limited',reason='Cannot fit fixed diagnosis,3000 steps and full original-size evaluation; no formal training');return
 cfg['staged']['deadlines']=dict(candidate=training_end,judgement=training_end-600);cfg['downstream']['deadline_unix']=training_end
 cfg['budget']={**budget,'evaluation_reserve_seconds':reserve,'benchmark':measured}
 snapshots=ROOT/'diagnostics/runtime_source_snapshot';snapshots.mkdir(exist_ok=True);cfg['runtime_source_evidence']={}
 for p in (ROOT/'runtime').rglob('*.py'):
  target=snapshots/p.relative_to(ROOT/'runtime');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
  cfg['runtime_source_evidence'][str(p.relative_to(ROOT))]=dict(sha256=sha(p),archived_copy=str(target.relative_to(ROOT)))
 cfg['code_sha256']={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts and 'runtime' not in p.relative_to(ROOT).parts}
 cfg['configuration_provenance_sha256']={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'config').glob('*') if p.is_file() and p.name not in ('locked_run.json','lock_receipt.json')}
 cfg['protocol_id']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest();json_save(ROOT/'config/locked_run.json',cfg)
 json_save(ROOT/'config/lock_receipt.json',dict(protocol_id=cfg['protocol_id'],locked_sha256=sha(ROOT/'config/locked_run.json'),judgement_steps=3000,diagnostic_trigger_train_only=True))
 status('plan_locked',protocol_id=cfg['protocol_id'],judgement_steps=3000,deadline_unix=budget['deadline_unix']);stage('diagnostic')
 trigger=json.loads((ROOT/'diagnostics/training_trigger.json').read_text())
 if not trigger['training_allowed']:
  status('diagnostic_complete_no_training',reason='Predeclared train-only predictability trigger failed; no extra recovery training',formal_training_started=False);return
 for action in ('initialize','judgement','calibration','mechanism','validation','test'):stage(action)
 status('completed',formal_training_started=True)
if __name__=='__main__':
 try:run()
 except Exception:
  (ROOT/'logs/failure.log').write_text(traceback.format_exc());status('failed',error=traceback.format_exc())
 finally:subprocess.run([sys.executable,str(ROOT/'scripts/archive.py')],cwd=ROOT,env=env)
