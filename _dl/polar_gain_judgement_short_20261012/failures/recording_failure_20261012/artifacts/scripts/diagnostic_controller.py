import json,os,subprocess,sys,time,traceback
from pathlib import Path
from common import ROOT,json_save,status
env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT.parent.parent/'tmp_psa'),XDG_CACHE_HOME=str(ROOT/'runtime/cache'))
Path(env['TMPDIR']).mkdir(parents=True,exist_ok=True)
try:
    assert not (ROOT/'budget.json').exists(), 'Budget cannot be restarted'
    start=time.time();budget=dict(start_unix=start,deadline_unix=start+86400,pgid=os.getpgrp(),evaluation_reserve_seconds=14400,scope='single new diagnostic + staged formal experiment; immutable 24 hours')
    json_save(ROOT/'budget.json',budget)
    with (ROOT/'logs/watchdog.log').open('a') as log:
        wd=subprocess.Popen([sys.executable,str(ROOT/'scripts/watchdog.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    json_save(ROOT/'watchdog.json',dict(pid=wd.pid,registered_pgid=budget['pgid'],deadline_unix=budget['deadline_unix']))
    status('candidate_diagnostic',deadline_unix=budget['deadline_unix'])
    with (ROOT/'logs/candidate_diagnostic.log').open('a') as log:
        code=subprocess.call([sys.executable,'-u',str(ROOT/'scripts/candidate_diagnostic.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if code:raise RuntimeError(f'Candidate diagnostic failed: {code}')
except Exception:
    (ROOT/'logs/diagnostic_failure.log').write_text(traceback.format_exc());status('failed',error=traceback.format_exc())
