"""Resume the same fixed protocol and deadline after a recording-only failure."""
import hashlib,json,os,pathlib,subprocess,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common import json_save,status,sha
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'source'),CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',TMPDIR=str(ROOT.parent.parent/'tmp_pgj'),XDG_CACHE_HOME=str(ROOT/'runtime/cache'),TORCH_HOME=str(ROOT.parent/'runtime/cache/torch'))
def stage(action):
 assert time.time()<json.loads((ROOT/'budget.json').read_text())['deadline_unix']
 with (ROOT/'logs'/f'{action}.log').open('a') as f:p=subprocess.run([sys.executable,'-u',str(ROOT/'recovery/run_gain_recovery.py'),action],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError(f'{action} failed:{p.returncode}')
def run():
 assert not (ROOT/'archive_ready.json').exists()
 cfg=json.loads((ROOT/'config/locked_run.json').read_text());budget=json.loads((ROOT/'budget.json').read_text())
 assert time.time()<budget['deadline_unix']
 for group in ('code_sha256','configuration_provenance_sha256'):
  for name,expected in cfg[group].items():assert sha(ROOT/name)==expected,name
 budget['previous_pgid']=budget['pgid'];budget.update(pgid=os.getpgrp(),process_group=os.getpgrp());json_save(ROOT/'budget.json',budget)
 with (ROOT/'logs/watchdog.log').open('a') as f:wd=subprocess.Popen([sys.executable,str(ROOT/'scripts/watchdog.py')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 json_save(ROOT/'watchdog.json',dict(pid=wd.pid,registered_pgid=budget['pgid'],deadline_unix=budget['deadline_unix']))
 json_save(ROOT/'recovery/recovery_receipt.json',dict(protocol_id=cfg['protocol_id'],original_start_unix=budget['start_unix'],original_deadline_unix=budget['deadline_unix'],process_group=budget['pgid'],frozen_code_unchanged=True,algorithm_unchanged=True,source_and_candidate_weights_unchanged=True,recording_only_fix='cast NumPy scalar values to identical native JSON numbers',cached_diagnostic_arrays_hash_checked=True,scripts={p.name:sha(p) for p in (ROOT/'recovery').glob('*.py')}))
 stage('diagnostic')
 trigger=json.loads((ROOT/'diagnostics/training_trigger.json').read_text())
 if not trigger['training_allowed']:status('diagnostic_complete_no_training',reason='Predeclared training-only signal failed; no formal training');return
 for action in ('initialize','judgement','calibration','mechanism','validation','test'):stage(action)
 status('completed',formal_training_started=True)
if __name__=='__main__':
 try:run()
 except Exception:
  (ROOT/'logs/recovery_failure.log').write_text(traceback.format_exc());status('failed',error=traceback.format_exc())
 finally:subprocess.run([sys.executable,str(ROOT/'scripts/archive.py')],cwd=ROOT,env=env)
