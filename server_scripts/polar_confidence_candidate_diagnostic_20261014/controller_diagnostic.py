"""Register a fresh immutable two-hour budget and independent CPU watchdog."""
import hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def state(stage,**kw):write(ROOT/'status.json',dict(stage=stage,time_unix=time.time(),**kw))

def watchdog():
    budget=json.loads((ROOT/'budget.json').read_text());pgid=budget['controller_pgid']
    while True:
        now=time.time()
        write(ROOT/'watchdog.json',dict(pid=os.getpid(),pgid=os.getpgrp(),registered_controller_pgid=pgid,deadline_unix=budget['deadline_unix'],heartbeat_unix=now))
        if now>=budget['deadline_unix']:
            state('deadline_reached',gpu_training_and_evaluation_prohibited=True)
            try:os.killpg(pgid,signal.SIGTERM)
            except ProcessLookupError:pass
            time.sleep(5)
            try:os.killpg(pgid,signal.SIGKILL)
            except ProcessLookupError:pass
            return
        if (ROOT/'archive_ready.json').exists():return
        try:os.killpg(pgid,0)
        except ProcessLookupError:return
        time.sleep(5)

def main():
    if (ROOT/'budget.json').exists():raise RuntimeError('GPU budget exists; refuse reset/relaunch')
    assert json.loads((ROOT/'diagnostics/cpu_checks.json').read_text())['passed']
    now=time.time();plan=json.loads((ROOT/'config/locked_run.json').read_text())
    write(ROOT/'budget.json',dict(start_unix=now,deadline_unix=now+plan['gpu_budget_seconds'],gpu_budget_seconds=plan['gpu_budget_seconds'],immutable=True,controller_pid=os.getpid(),controller_pgid=os.getpgrp(),protocol_id=plan['protocol_id']))
    cpuenv=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    watchlog=(ROOT/'logs/watchdog.log').open('a')
    watcher=subprocess.Popen([sys.executable,__file__,'--watchdog'],env=cpuenv,stdout=watchlog,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'watchdog_registration.json',dict(pid=watcher.pid,pgid=watcher.pid,controller_pgid=os.getpgrp(),deadline_unix=now+plan['gpu_budget_seconds']))
    try:
        log=(ROOT/'logs/gpu_diagnostic.log').open('a')
        proc=subprocess.run([sys.executable,str(ROOT/'scripts/run_diagnostic.py'),'--mode','gpu_run'],stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:raise RuntimeError('GPU diagnostic failed: '+str(proc.returncode))
        state('analysing_cpu')
        subprocess.run([sys.executable,str(ROOT/'scripts/analyze_results.py'),'--root',str(ROOT)],env=cpuenv,check=True)
        state('completed',diagnostic_only=True,training_steps=0,test_frames=0,protocol_id=plan['protocol_id'])
        subprocess.run([sys.executable,str(ROOT/'scripts/archive_diagnostic.py')],env=cpuenv,check=True)
    except Exception:
        write(ROOT/'diagnostics/execution_failure.json',dict(time_unix=time.time(),traceback=traceback.format_exc()))
        state('failed',error=traceback.format_exc())
        print(traceback.format_exc(),flush=True)
    finally:
        if watcher.poll() is None:
            watcher.terminate();watcher.wait(timeout=15)
        write(ROOT/'watchdog_exit.json',dict(time_unix=time.time(),pid=watcher.pid,stopped=True,deadline_unix=now+plan['gpu_budget_seconds']))

if __name__=='__main__':
    if '--watchdog' in sys.argv:watchdog()
    else:main()
