import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
lock=json.loads((ROOT/'protocol_lock.json').read_text())
logs=ROOT/'logs';logs.mkdir(exist_ok=True)
completed=[]
def status(u,state,**extra):
    p=dict(unit=u['id'],family=u['family'],seed=u['seed'],state=state,stage='test_188',
        completed_units=completed,total_units=len(lock['units']),time_unix=time.time(),**extra)
    tmp=ROOT/'status.json.tmp';tmp.write_text(json.dumps(p,indent=2));tmp.replace(ROOT/'status.json')
for u in lock['units']:
    logfile=logs/(u['id']+'_test.log')
    assert not logfile.exists()
    print('START '+u['id'],flush=True)
    with logfile.open('w') as log:
        child=subprocess.Popen([sys.executable,str(ROOT/'scripts/evaluate_unit.py'),u['id']],stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONUNBUFFERED='1',
            PYTHONDONTWRITEBYTECODE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'))
        status(u,'running',pid=child.pid,logfile=str(logfile));rc=child.wait()
    if rc:
        status(u,'failed',returncode=rc,logfile=str(logfile));raise RuntimeError(str(logfile))
    completed.append(dict(unit=u['id'],finished_unix=time.time()));print('DONE '+u['id'],flush=True)
status(lock['units'][-1],'complete');print('ALL SIXTEEN EVALUATIONS COMPLETED',flush=True)
