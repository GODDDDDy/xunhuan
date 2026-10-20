"""CPU coordinator and independent immutable-deadline watchdog for read-only audit."""
import json, os, signal, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY=sys.executable

def save(name, data):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2));temp.replace(p)

def run(mode, log):
    with (ROOT/'logs'/log).open('a') as f:
        process=subprocess.Popen([PY,str(ROOT/'scripts/audit_run.py'),mode],stdout=f,stderr=subprocess.STDOUT)
        code=process.wait()
    if code:raise RuntimeError(f'{mode} exited {code}; see {log}')

def watchdog():
    while not (ROOT/'budget.json').exists():
        if (ROOT/'status.json').exists() and json.loads((ROOT/'status.json').read_text()).get('stage')=='failed':return
        time.sleep(1)
    b=json.loads((ROOT/'budget.json').read_text())
    while Path('/proc',str(b['controller_pid'])).exists():
        if (ROOT/'status.json').exists() and json.loads((ROOT/'status.json').read_text()).get('stage') in ('completed','failed'):return
        save('watchdog.json',dict(pid=os.getpid(),pgid=os.getpgrp(),controller_pgid=b['controller_pgid'],
             deadline_unix=b['deadline_unix'],heartbeat_unix=time.time()))
        if time.time()>=b['deadline_unix']:
            save('status.json',dict(stage='deadline_exceeded',time_unix=time.time(),training=False,completed=False))
            os.killpg(b['controller_pgid'],signal.SIGTERM);return
        time.sleep(2)

def main():
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    child=None
    try:
        run('cpu','cpu.log')
        plan=json.loads((ROOT/'config/locked_run.json').read_text())
        if not (ROOT/'budget.json').exists():
            start=time.time()
            save('budget.json',dict(start_unix=start,deadline_unix=start+10800,immutable=True,
                 hours=3,training=False,protocol_id=plan['protocol_id'],controller_pid=os.getpid(),controller_pgid=os.getpgrp()))
        budget=json.loads((ROOT/'budget.json').read_text())
        assert time.time()<budget['deadline_unix']
        budget.update(controller_pid=os.getpid(),controller_pgid=os.getpgrp(),protocol_id=plan['protocol_id'])
        save('budget.json',budget)
        with (ROOT/'logs/watchdog.log').open('a') as f:
            child=subprocess.Popen([PY,str(Path(__file__)),'--watchdog'],start_new_session=True,stdout=f,stderr=subprocess.STDOUT)
        run('evaluate','baseline_validation.log')
        save('status.json',dict(stage='completed',time_unix=time.time(),training=False,test_used=False,
             validation_frames=872,protocol_id=plan['protocol_id'],cpu_analysis_pending=True))
    except Exception:
        save('status.json',dict(stage='failed',time_unix=time.time(),training=False,completed=False,error=traceback.format_exc()))
        traceback.print_exc()
    finally:
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=10)
        with (ROOT/'logs/archive.log').open('a') as f:
            subprocess.run([PY,str(ROOT/'scripts/archive.py')],stdout=f,stderr=subprocess.STDOUT,check=True)

if __name__=='__main__':
    watchdog() if '--watchdog' in sys.argv else main()
