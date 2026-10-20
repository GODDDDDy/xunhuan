"""Bounded process ownership and server-only artifact verification."""
import hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False));tmp.replace(path)

def watch():
    b=json.loads((ROOT/'budget.json').read_text());group=b['controller_pgid']
    while time.time()<b['deadline_unix']:
        try:os.killpg(group,0)
        except ProcessLookupError:return
        time.sleep(5)
    write(ROOT/'deadline_stop.json',dict(time_unix=time.time(),controller_pgid=group))
    try:os.killpg(group,signal.SIGTERM)
    except ProcessLookupError:return
    time.sleep(5)
    try:os.killpg(group,signal.SIGKILL)
    except ProcessLookupError:pass

def main():
    assert json.loads((ROOT/'diagnostics/cpu_checks.json').read_text())['passed']
    plan=json.loads((ROOT/'config/plan.json').read_text());now=time.time()
    if (ROOT/'budget.json').exists():
        assert '--resume-preflight' in sys.argv,'Never reset an existing run budget'
        assert not (ROOT/'weights/initial.pt').exists() and not (ROOT/'logs/training.csv').exists(),'No restarting formal training'
        previous=json.loads((ROOT/'budget.json').read_text());start=previous['start_unix'];end=previous['deadline_unix']
    else:start=now;end=start+plan['budget_seconds']
    assert now<end
    write(ROOT/'budget.json',dict(start_unix=start,deadline_unix=end,immutable=True,
              protocol_id=plan['protocol_id'],controller_pid=os.getpid(),controller_pgid=os.getpgrp(),
              execution_recovery='--resume-preflight' in sys.argv))
    wlog=(ROOT/'logs/watchdog.log').open('a')
    watcher=subprocess.Popen([sys.executable,__file__,'watch'],stdout=wlog,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'watchdog_registration.json',dict(pid=watcher.pid,controller_pgid=os.getpgrp(),deadline_unix=end))
    code=1
    try:
        with (ROOT/'logs/run.log').open('a') as log:
            code=subprocess.call([sys.executable,str(ROOT/'scripts/c_loop_run.py'),'run'],stdout=log,stderr=subprocess.STDOUT)
        if code:raise RuntimeError('Training/evaluation process failed: '+str(code))
    except Exception:
        write(ROOT/'controller_failure.json',dict(time_unix=time.time(),traceback=traceback.format_exc()))
    finally:
        if watcher.poll() is None:watcher.terminate();watcher.wait(timeout=15)
        write(ROOT/'watchdog_exit.json',dict(pid=watcher.pid,stopped=True,time_unix=time.time()))
    files=[]
    for folder in ('config','scripts','weights','metrics','examples','diagnostics','reports','logs'):
        for p in sorted((ROOT/folder).rglob('*')):
            if not p.is_file() or '__pycache__' in str(p):continue
            h=hashlib.sha256()
            with p.open('rb') as f:
                for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
            files.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=h.hexdigest()))
    write(ROOT/'artifact_manifest.json',dict(files=files,server_only=True))
    write(ROOT/'closeout.json',dict(process_returncode=code,completed=code==0,files_verified=len(files),
                                   time_unix=time.time(),server_only=True))
    sys.exit(code)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='watch':watch()
    else:main()
