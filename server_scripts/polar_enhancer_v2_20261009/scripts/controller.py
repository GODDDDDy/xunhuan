"""24h bounded single-arm experiment; all GPU workers exit between phases."""
import hashlib,json,math,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
short_tmp=ROOT.parent.parent/'tmp_pe2'
short_tmp.mkdir(parents=True,exist_ok=True)
os.environ['TMPDIR']=str(short_tmp)
os.environ['XDG_CACHE_HOME']=str(ROOT/'runtime/cache')
# Existing downloaded VGG belongs to the project runtime; avoid a fresh cache.
os.environ['TORCH_HOME']=str(ROOT.parent/'runtime/cache/torch')

def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(p)
def status(stage,**kw):
    write(ROOT/'status.json',dict(stage=stage,time_unix=time.time(),**kw));print(stage,kw,flush=True)
def stage(action):
    with (ROOT/'logs'/f'{action}.log').open('a') as f:
        subprocess.run([sys.executable,str(ROOT/'scripts/run_stage.py'),action],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)

def resume_amended_run():
    """CPU-only coordinator; preserve the original GPU clock and saved lineage."""
    budget=json.loads((ROOT/'budget.json').read_text())
    assert budget['start_unix']==1791518647.4916465 and budget['deadline_unix']==1791605047.4916465
    if time.time()>=budget['deadline_unix']:raise RuntimeError('Original GPU deadline passed')
    cfg=json.loads((ROOT/'config/locked_run.json').read_text())
    assert cfg['task_initialization']['kind']=='official_polarfree'
    assert cfg['amendment']['test_seen_before_amendment'] is False
    assert not (ROOT/'downstream/checkpoints/initial.pt').exists(), 'Explicit phase recovery required after initialization'
    lock=json.loads((ROOT/'protocol_lock.json').read_text())
    for rel,digest in lock['code_sha256'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
    for path,digest in lock['sources_sha256'].items():
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        assert h.hexdigest()==digest,path
    old=json.loads((ROOT/'watchdog.json').read_text());proc=Path('/proc')/str(old['pid'])
    if proc.exists():
        cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        if cmd:
            assert str(ROOT/'scripts/watchdog.py') in cmd
            os.kill(old['pid'],signal.SIGTERM)
    budget['process_group']=os.getpgrp();write(ROOT/'budget.json',budget)
    write(ROOT/'controller.json',dict(pid=os.getpid(),process_group=os.getpgrp(),mode='authorized amendment recovery'))
    with (ROOT/'logs/watchdog.log').open('a') as f:
        watchdog=subprocess.Popen([sys.executable,str(ROOT/'scripts/watchdog.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'watchdog.json',dict(pid=watchdog.pid,deadline_unix=budget['deadline_unix']))
    status('authorized_amendment_resumed',deadline_unix=budget['deadline_unix'],pretrain_minimum=16000,pretrain_maximum=20000)
    stage('pretraining');stage('initialize');stage('training');stage('validation');stage('test')
    status('completed')

def run():
    if '--resume-run' in sys.argv:
        resume_amended_run();return
    previous_budget=None
    if (ROOT/'budget.json').exists():
        if '--resume-preflight' not in sys.argv or (ROOT/'config/locked_run.json').exists():
            raise RuntimeError('Budget already started; explicit checkpoint recovery required, never restart timer')
        previous_budget=json.loads((ROOT/'budget.json').read_text())
        if time.time()>=previous_budget['deadline_unix']:raise RuntimeError('GPU deadline passed')
    lock=json.loads((ROOT/'protocol_lock.json').read_text())
    for rel,digest in lock['code_sha256'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
    for path,digest in lock['sources_sha256'].items():
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        assert h.hexdigest()==digest,path
    cfg=json.loads((ROOT/'config/proposed_run.json').read_text())
    t0=previous_budget['start_unix'] if previous_budget else time.time()
    budget=dict(start_unix=t0,deadline_unix=t0+86400,process_group=os.getpgrp(),
              device='server RTX3080Ti',hours=24,starts_at='first GPU preflight')
    write(ROOT/'budget.json',budget)
    with (ROOT/'logs/watchdog.log').open('a') as f:
        watchdog=subprocess.Popen([sys.executable,str(ROOT/'scripts/watchdog.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'watchdog.json',dict(pid=watchdog.pid,deadline_unix=budget['deadline_unix']))
    status('gpu_preflight',deadline_unix=budget['deadline_unix'])
    import torch,pretrain,study
    torch.set_num_threads(4);device=torch.device('cuda')
    pre=pretrain.benchmark(cfg,device)
    down=study.benchmark(cfg,device)
    # Reserve full final 872+188 evaluation, pretrain 30, 256 train calibration,
    # 4 gradient calibration batches and stage transitions; no test benchmark.
    estimated_eval=1060*down['full_image_seconds']+30*pre['validation_scene_seconds']
    reserve=max(14400,1.5*estimated_eval)
    preparation=256*.5*down['step_seconds']+8*down['step_seconds']+600
    remaining=budget['deadline_unix']-time.time()-reserve-preparation
    pre_unit=pre['step_seconds']+10*pre['validation_scene_seconds']/1000
    presteps=min(12000,int(max(remaining*.55,0)/pre_unit)//1000*1000)
    downsteps=min(8000,int(max(remaining*.45,0)/down['step_seconds'])//500*500)
    can_formal=presteps>=6000 and downsteps>=3000
    if not can_formal:presteps=min(12000,int(max(remaining,0)/pre_unit)//1000*1000)
    if presteps<1000:raise RuntimeError('Budget insufficient even for evaluated pretraining checkpoint')
    pre_end=time.time()+remaining*.55 if can_formal else budget['deadline_unix']-reserve
    cfg['pretrain'].update(steps=presteps,deadline_unix=pre_end)
    cfg['downstream'].update(steps=downsteps,deadline_unix=budget['deadline_unix']-reserve)
    cfg['budget']=dict(**budget,evaluation_reserve_seconds=reserve,estimated_eval_seconds=estimated_eval,
                       preparation_reserve_seconds=preparation,formal_feasible=can_formal)
    cfg['code_sha256']=lock['code_sha256']
    cfg['protocol_id']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    write(ROOT/'config/locked_run.json',cfg)
    write(ROOT/'diagnostics/budget_plan.json',dict(pretrain=pre,downstream=down,configuration=cfg))
    status('plan_locked',pretrain_steps=presteps,downstream_steps=downsteps,formal_feasible=can_formal)
    # Parent preflight imports do not leave any live CUDA tensors.
    stage('pretraining')
    if not can_formal:
        status('limited',reason='Measured budget cannot meet 6000 pretrain and 3000 downstream steps; only correction pretraining completed')
        return
    stage('initialize');stage('training');stage('validation');stage('test')
    status('completed')
if __name__=='__main__':
    try:run()
    except Exception:
        (ROOT/'logs/failure.log').write_text(traceback.format_exc());status('failed',error=traceback.format_exc())
    finally:
        subprocess.run([sys.executable,str(ROOT/'scripts/archive.py')],cwd=ROOT)
