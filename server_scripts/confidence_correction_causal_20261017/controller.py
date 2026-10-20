"""Own-process supervision, one controller, absolute deadline, immutable endpoints."""
import fcntl,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False));t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def main():
    lock=(ROOT/'controller.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    budget=json.loads((ROOT/'budget.json').read_text());deadline=budget['deadline_unix']
    write(ROOT/'controller_registration.json',dict(pid=os.getpid(),pgid=os.getpgrp(),start_unix=budget['start_unix'],deadline_unix=deadline,single_instance=True,time_unix=time.time()))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPYCACHEPREFIX=str(ROOT/'cache/pycache'),MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    while True:
        if time.time()>=deadline:raise RuntimeError('36h deadline reached before completed evaluation')
        while True:
            if time.time()>=deadline:raise RuntimeError('36h elapsed waiting for GPU')
            try:
                free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()[0])
            except Exception:free=0
            write(ROOT/'resource_wait.json',dict(time_unix=time.time(),free_MiB=free,minimum_MiB=11500,deadline_unix=deadline))
            if free>=11500:break
            time.sleep(2)
        with (ROOT/'logs/worker.log').open('a') as f:
            child=subprocess.Popen([sys.executable,str(ROOT/'scripts/causal_run.py')],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            write(ROOT/'worker_registration.json',dict(pid=child.pid,pgid=child.pid,time_unix=time.time(),deadline_unix=deadline))
            while child.poll() is None:
                write(ROOT/'controller_heartbeat.json',dict(time_unix=time.time(),worker_pid=child.pid,deadline_unix=deadline))
                # Reserve 10 minutes for orderly final evidence/summary; no partial endpoint promoted.
                if time.time()>=deadline-600:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise RuntimeError('36h limit: own worker stopped with preserved checkpoints; experiment incomplete')
                time.sleep(5)
            code=child.returncode
        if code==0:break
        fail=sorted((ROOT/'failures').glob('worker_*.json'),key=lambda p:p.stat().st_mtime)
        message=fail[-1].read_text() if fail else ''
        if 'out of memory' not in message.lower():raise RuntimeError('Worker execution failed; preserved receipt requires repair')
        time.sleep(2)
    assert (ROOT/'reports/summary.json').exists() and (ROOT/'reports/diagnostic_summary.json').exists()
    for split,n in [('validation872',872),('test188',188)]:
        data=json.loads((ROOT/'metrics'/f'{split}.json').read_text());assert data['frames']==n
    write(ROOT/'status.json',dict(stage='archiving',time_unix=time.time()))
    files=[]
    for folder in ['config','scripts','weights','metrics','diagnostics','reports','examples','failures','logs']:
        for p in sorted((ROOT/folder).rglob('*')):
            if not p.is_file() or p.name in ['controller.log'] or p.suffix in ['.pyc','.tmp']:continue
            files.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
    write(ROOT/'artifact_manifest.json',dict(files=files,server_only=True))
    write(ROOT/'closeout.json',dict(completed=True,time_unix=time.time(),files_hashed=len(files),deadline_met=time.time()<=deadline,server_only=True))
    write(ROOT/'status.json',dict(stage='completed',time_unix=time.time(),steps=6000,validation=872,test=188))
if __name__=='__main__':
    try:main()
    except Exception:
        write(ROOT/'controller_failure.json',dict(traceback=traceback.format_exc(),time_unix=time.time(),completed=False));raise
