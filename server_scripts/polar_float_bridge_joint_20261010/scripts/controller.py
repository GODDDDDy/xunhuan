"""One latest training arm; reuse pretraining and preserve original GPU cutoff."""
import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEADLINE=1791605047.4916465
START=1791518647.4916465
os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                  PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT.parent.parent/'tmp_pe2'),
                  XDG_CACHE_HOME=str(ROOT/'runtime/cache'),TORCH_HOME=str(ROOT.parent/'runtime/cache/torch'))

def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(p)

def status(stage,**kw):
    write(ROOT/'status.json',dict(stage=stage,time_unix=time.time(),**kw));print(stage,kw,flush=True)

def stage(action):
    if time.time()>=DEADLINE:raise RuntimeError('Original GPU deadline passed')
    with (ROOT/'logs'/f'{action}.log').open('a') as f:
        subprocess.run([sys.executable,str(ROOT/'scripts/run_stage.py'),action],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)

def run():
    assert time.time()<DEADLINE
    assert not (ROOT/'downstream/checkpoints/initial.pt').exists()
    budget=json.loads((ROOT/'budget.json').read_text())
    assert budget['start_unix']==START and budget['deadline_unix']==DEADLINE
    assert budget['process_group']==os.getpgrp()
    with (ROOT/'logs/watchdog.log').open('a') as f:
        w=subprocess.Popen([sys.executable,str(ROOT/'scripts/watchdog.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'watchdog.json',dict(pid=w.pid,process_group=os.getpgrp(),deadline_unix=DEADLINE))
    status('preflight');stage('benchmark')
    measured=json.loads((ROOT/'diagnostics/downstream_benchmark.json').read_text())
    reserve=max(14400,1.5*1060*measured['full_image_seconds'])
    preparation=1200
    remaining=DEADLINE-time.time()-reserve-preparation
    steps=min(6000,int(max(remaining,0)/(1.1*measured['step_seconds']))//500*500)
    if steps<3000:
        status('limited',reason='Remaining original budget cannot fit >=3000 steps plus full evaluation; no formal training started')
        return
    cfg=json.loads((ROOT/'config/preflight.json').read_text())
    cfg['downstream'].update(steps=steps,deadline_unix=DEADLINE-reserve)
    cfg['budget']=dict(**budget,evaluation_reserve_seconds=reserve,preparation_reserve_seconds=preparation,
                       benchmark=measured,formal_feasible=True)
    cfg['code_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts}
    cfg['protocol_id']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    write(ROOT/'config/locked_run.json',cfg)
    status('plan_locked',steps=steps,evaluation_reserve_seconds=reserve,deadline_unix=DEADLINE)
    stage('initialize');stage('training');stage('validation');stage('test')
    status('completed')

if __name__=='__main__':
    try:run()
    except Exception:
        (ROOT/'logs/failure.log').write_text(traceback.format_exc());status('failed',error=traceback.format_exc())
    finally:
        subprocess.run([sys.executable,str(ROOT/'scripts/archive.py')],cwd=ROOT)
