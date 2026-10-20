"""Single owned run; checkpoint resume on OOM; no deadline or extra training."""
import fcntl,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def write(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False));tmp.replace(p)

def invoke(name,args,env=None):
    with (ROOT/'logs'/name).open('a') as f:
        return subprocess.Popen([sys.executable,*args],stdout=f,stderr=subprocess.STDOUT,env=env)

def main():
    lock=(ROOT/'controller.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'closeout.json').exists():
        assert not json.loads((ROOT/'closeout.json').read_text())['completed'],'Already completed'
    write(ROOT/'controller_registration.json',dict(pid=os.getpid(),pgid=os.getpgrp(),time_unix=time.time(),time_budget_enforced=False,steps=6000))
    cpu_env=dict(os.environ,CUDA_VISIBLE_DEVICES='',MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),PYTHONPYCACHEPREFIX=str(ROOT/'cache/pycache'))
    offline=None
    if not (ROOT/'reports/offline_summary.json').exists():offline=invoke('offline.log',[str(ROOT/'scripts/offline_diagnostics.py')],cpu_env)
    if not (ROOT/'diagnostics/cpu_checks.json').exists():
        check=invoke('cpu_checks.log',[str(ROOT/'scripts/oracle_study.py'),'cpu'],cpu_env)
        if check.wait()!=0:raise RuntimeError('CPU verification failed; see logs/cpu_checks.log')
    while not (ROOT/'reports/oracle_summary.json').exists():
        while True:
            try:
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=15)
                free=int(raw.splitlines()[0])
            except Exception:free=None
            write(ROOT/'resource_wait.json',dict(time_unix=time.time(),free_MiB=free,minimum_MiB=11500,deadline_unix=None,poll_seconds=2))
            if free is not None and free>=11500:break
            time.sleep(2)
        env=dict(os.environ,PYTHONPYCACHEPREFIX=str(ROOT/'cache/pycache'),MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        worker=invoke('gpu.log',[str(ROOT/'scripts/oracle_study.py'),'gpu'],env)
        write(ROOT/'worker_registration.json',dict(pid=worker.pid,time_unix=time.time()))
        code=worker.wait()
        if code==0:break
        latest=sorted((ROOT/'failures').glob('worker_*.json'),key=lambda p:p.stat().st_mtime)
        message=latest[-1].read_text() if latest else ''
        if 'out of memory' not in message.lower():raise RuntimeError('GPU worker failed; preserved failure needs inspection')
        time.sleep(2)
    if offline is not None and offline.wait()!=0:raise RuntimeError('Offline diagnostic failed; see logs/offline.log')
    assert (ROOT/'reports/offline_summary.json').exists() and (ROOT/'reports/oracle_summary.json').exists()
    # Sanity checks before hashing; no metrics are used to select a checkpoint.
    for split,count in [('validation872',872),('test188',188)]:
        a=json.loads((ROOT/'metrics'/f'{split}.json').read_text())
        assert a['frames']==count and len({r['sample_id'] for r in a['rows']})==count
    write(ROOT/'status.json',dict(stage='archiving',validation=872,test=188,steps=6000,time_unix=time.time()))
    files=[]
    for folder in ('config','scripts','weights','metrics','examples','diagnostics','reports','failures','logs'):
        for p in sorted((ROOT/folder).rglob('*')):
            if not p.is_file() or p.suffix=='.pyc' or p.name=='controller.log':continue
            h=hashlib.sha256()
            with p.open('rb') as f:
                for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
            files.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=h.hexdigest()))
    write(ROOT/'artifact_manifest.json',dict(files=files,server_only=True))
    write(ROOT/'closeout.json',dict(completed=True,process_returncode=0,files_hashed=len(files),server_only=True,time_unix=time.time()))
    write(ROOT/'status.json',dict(stage='completed',validation=872,test=188,steps=6000,time_unix=time.time()))

if __name__=='__main__':
    try:main()
    except Exception:
        write(ROOT/'controller_failure.json',dict(traceback=traceback.format_exc(),time_unix=time.time()))
        raise
