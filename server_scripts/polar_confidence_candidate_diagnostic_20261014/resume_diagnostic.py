"""Execution-only allocator recovery; original frozen diagnostic code is unchanged."""
import json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(p)
def state(stage,**kw):write(ROOT/'status.json',dict(stage=stage,time_unix=time.time(),**kw))
budget=json.loads((ROOT/'budget.json').read_text())
assert time.time()<budget['deadline_unix']
attempt=1
while (ROOT/f'diagnostics/gpu_memory_attempt{attempt}').exists():attempt+=1
old=ROOT/f'diagnostics/gpu_memory_attempt{attempt}';old.mkdir(exist_ok=False)
for name in ('budget.json','watchdog.json','watchdog_registration.json','watchdog_exit.json','status.json','diagnostics/execution_failure.json','diagnostics/recovery_failure.json','logs/controller.log','logs/gpu_diagnostic.log','logs/gpu_diagnostic_recovery.log','logs/recovery_controller.log','config/fixed_quality_calibration.json'):
    p=ROOT/name
    if p.exists():q=old/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
original=(budget['start_unix'],budget['deadline_unix'])
budget.update(controller_pid=os.getpid(),controller_pgid=os.getpgrp());write(ROOT/'budget.json',budget)
assert original==(budget['start_unix'],budget['deadline_unix'])
cpuenv=dict(os.environ,CUDA_VISIBLE_DEVICES='')
watchlog=(ROOT/'logs/watchdog_recovery.log').open('a')
watcher=subprocess.Popen([sys.executable,str(ROOT/'scripts/controller_diagnostic.py'),'--watchdog'],env=cpuenv,stdout=watchlog,stderr=subprocess.STDOUT,start_new_session=True)
write(ROOT/'watchdog_registration.json',dict(pid=watcher.pid,pgid=watcher.pid,controller_pgid=os.getpgrp(),deadline_unix=budget['deadline_unix'],recovery=True))
write(ROOT/'diagnostics/memory_recovery_receipt.json',dict(start_unix=budget['start_unix'],deadline_unix=budget['deadline_unix'],budget_not_reset=True,frozen_algorithm_unchanged=True,allocator_environment=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),pid=os.getpid(),pgid=os.getpgrp()))
success=False
try:
    log=(ROOT/'logs/gpu_diagnostic_recovery.log').open('a')
    proc=subprocess.run([sys.executable,str(ROOT/'scripts/run_diagnostic.py'),'--mode','gpu_run'],stdout=log,stderr=subprocess.STDOUT)
    if proc.returncode:raise RuntimeError('GPU recovery failed: '+str(proc.returncode))
    subprocess.run([sys.executable,str(ROOT/'scripts/analyze_results.py'),'--root',str(ROOT)],env=cpuenv,check=True)
    state('completed',diagnostic_only=True,training_steps=0,test_frames=0,protocol_id=budget['protocol_id']);success=True
except Exception:
    write(ROOT/'diagnostics/recovery_failure.json',dict(time_unix=time.time(),traceback=traceback.format_exc()));state('failed',error=traceback.format_exc());print(traceback.format_exc(),flush=True)
finally:
    if watcher.poll() is None:watcher.terminate();watcher.wait(timeout=15)
    write(ROOT/'watchdog_exit.json',dict(time_unix=time.time(),pid=watcher.pid,stopped=True,deadline_unix=budget['deadline_unix']))
if success:subprocess.run([sys.executable,str(ROOT/'scripts/archive_diagnostic.py')],env=cpuenv,check=True)
