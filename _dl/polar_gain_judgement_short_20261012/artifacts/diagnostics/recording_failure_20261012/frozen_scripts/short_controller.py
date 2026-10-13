"""Independent eight-hour pilot, immutable clocks, original-size evaluation."""
import hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from common import ROOT,json_save,status,sha
env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT.parent.parent/'tmp_pegs'),XDG_CACHE_HOME=str(ROOT/'runtime/cache'),TORCH_HOME=str(ROOT.parent/'runtime/cache/torch'))
Path(env['TMPDIR']).mkdir(parents=True,exist_ok=True)
def stage(action):
 if (ROOT/'budget.json').exists():assert time.time()<json.loads((ROOT/'budget.json').read_text())['deadline_unix']
 local=dict(env)
 if action=='verify':local['CUDA_VISIBLE_DEVICES']=''
 with (ROOT/'logs'/f'{action}.log').open('a') as f:
  p=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/run_short.py'),action],cwd=ROOT,env=local,stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError(f'{action} failed with exit {p.returncode}')
def run():
 resume='--resume-preflight' in sys.argv
 assert not (ROOT/'config/locked_run.json').exists(), 'Formal code/configuration is frozen; use a documented same-protocol recovery'
 assert not (ROOT/'archive_ready.json').exists(), 'Preserve failed attempt before recovery; stale archive receipt must not stop watchdog'
 assert not (ROOT/'budget.json').exists() or resume, 'Existing clocks must not be reset'
 stage('verify')
 if (ROOT/'budget.json').exists():
  budget=json.loads((ROOT/'budget.json').read_text());assert time.time()<budget['deadline_unix']
  budget['previous_pgid']=budget['pgid'];budget.update(pgid=os.getpgrp(),process_group=os.getpgrp())
 else:
  now=time.time();budget=dict(start_unix=now,deadline_unix=now+28800,pgid=os.getpgrp(),process_group=os.getpgrp(),evaluation_reserve_seconds=14400,scope='single authorized 3000-step pilot plus fixed diagnostics and full evaluation; immutable8h')
 json_save(ROOT/'budget.json',budget)
 with (ROOT/'logs/watchdog.log').open('a') as f:wd=subprocess.Popen([sys.executable,str(ROOT/'scripts/watchdog.py')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 json_save(ROOT/'watchdog.json',dict(pid=wd.pid,registered_pgid=budget['pgid'],deadline_unix=budget['deadline_unix']))
 stage('benchmark');measured=json.loads((ROOT/'diagnostics/staged_benchmark.json').read_text())
 cfg=json.loads((ROOT/'config/short_preflight.json').read_text());reserve=max(14400.,1.5*1300*measured['full_image_seconds'])
 required=1.5*(872*measured['full_image_seconds']+sum(cfg['staged']['steps'][p]*measured['step_seconds'][p] for p in ('candidate','judgement')))+1800
 training_end=budget['deadline_unix']-reserve
 if training_end-time.time()<required:status('limited',reason='Cannot fit fixed source diagnostic, fixed3000 steps, and full evaluation; no training started');return
 candidate_end=training_end-1.5*cfg['staged']['steps']['judgement']*measured['step_seconds']['judgement']-900
 cfg['staged']['deadlines']=dict(candidate=candidate_end,judgement=training_end-600)
 cfg['budget']={**budget,'evaluation_reserve_seconds':reserve,'benchmark':measured};cfg['downstream']['deadline_unix']=training_end
 snapshots=ROOT/'diagnostics/runtime_source_snapshot';snapshots.mkdir(exist_ok=True);cfg['runtime_source_evidence']={}
 for p in (ROOT/'runtime').rglob('*.py'):
  target=snapshots/p.relative_to(ROOT/'runtime');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
  cfg['runtime_source_evidence'][str(p.relative_to(ROOT))]=dict(sha256=sha(p),archived_copy=str(target.relative_to(ROOT)))
 cfg['code_sha256']={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts and 'runtime' not in p.relative_to(ROOT).parts}
 cfg['configuration_provenance_sha256']={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'config').glob('*') if p.is_file() and p.name not in ('locked_run.json','lock_receipt.json')}
 cfg['protocol_id']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest();json_save(ROOT/'config/locked_run.json',cfg)
 json_save(ROOT/'config/lock_receipt.json',dict(protocol_id=cfg['protocol_id'],locked_sha256=sha(ROOT/'config/locked_run.json'),fixed_steps=cfg['staged']['steps'],no_test_used=True,source_diagnostic_does_not_change_plan=True))
 status('plan_locked',protocol_id=cfg['protocol_id'],candidate_steps=2000,judgement_steps=1000,deadline_unix=budget['deadline_unix'])
 for action in ('source_diagnostic','initialize','candidate','judgement','calibration','mechanism','validation','test'):stage(action)
 status('completed')
if __name__=='__main__':
 try:run()
 except Exception:
  (ROOT/'logs/failure.log').write_text(traceback.format_exc());status('failed',error=traceback.format_exc())
 finally:subprocess.run([sys.executable,str(ROOT/'scripts/archive.py')],cwd=ROOT,env=env)
