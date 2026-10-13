import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
from common import ROOT,json_save,status,sha
env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT.parent.parent/'tmp_psa'),XDG_CACHE_HOME=str(ROOT/'runtime/cache'),TORCH_HOME=str(ROOT.parent/'runtime/cache/torch'))
Path(env['TMPDIR']).mkdir(parents=True,exist_ok=True)

def stage(action):
    budget=json.loads((ROOT/'budget.json').read_text());assert time.time()<budget['deadline_unix']
    with (ROOT/'logs'/f'{action}.log').open('a') as f:
        result=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/run_staged.py'),action],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{action} failed with exit {result.returncode}')

def run():
    budget=json.loads((ROOT/'budget.json').read_text());assert time.time()<budget['deadline_unix']
    # Re-register only the current group; retain the immutable original clocks.
    old=json.loads((ROOT/'watchdog.json').read_text())
    if Path('/proc',str(old['pid'])).exists():os.kill(old['pid'],15)
    budget['previous_pgid']=budget['pgid'];budget['pgid']=os.getpgrp();json_save(ROOT/'budget.json',budget)
    with (ROOT/'logs/watchdog.log').open('a') as f:wd=subprocess.Popen([sys.executable,str(ROOT/'scripts/watchdog.py')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    json_save(ROOT/'watchdog.json',dict(pid=wd.pid,registered_pgid=budget['pgid'],deadline_unix=budget['deadline_unix']))
    stage('verify');stage('benchmark')
    measured=json.loads((ROOT/'diagnostics/staged_benchmark.json').read_text());reserve=max(14400.,1.5*1300*measured['full_image_seconds'])
    cfg=json.loads((ROOT/'config/staged_preflight.json').read_text());remain=budget['deadline_unix']-time.time()-reserve-3600
    expected=1.25*sum(cfg['staged']['steps'][p]*measured['step_seconds'][p] for p in ('candidate','judgement'))
    if remain<expected:
        status('limited',reason='Cannot fit the planned3000+3000 stages and complete evaluation in remaining budget; no formal training started');return
    training_end=budget['deadline_unix']-reserve
    # Allocate candidate time according to measured training speeds, with a common
    # final training deadline. Phase cutoffs cannot consume the evaluation reserve.
    fraction=measured['step_seconds']['candidate']/(measured['step_seconds']['candidate']+measured['step_seconds']['judgement'])
    candidate_end=time.time()+max(1.5*3000*measured['step_seconds']['candidate'],(training_end-time.time()-1800)*fraction)
    candidate_end=min(candidate_end,training_end-1.5*3000*measured['step_seconds']['judgement']-900)
    cfg['staged']['deadlines']=dict(candidate=candidate_end,judgement=training_end-900)
    cfg['budget']=dict(**budget,evaluation_reserve_seconds=reserve,benchmark=measured)
    cfg['downstream'].update(steps=6000,deadline_unix=training_end)
    cfg['code_sha256']={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts and 'runtime' not in p.relative_to(ROOT).parts}
    cfg['configuration_provenance_sha256']={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'config').glob('*') if p.is_file() and p.name not in ('locked_run.json','calibrated_lock_receipt.json')}
    cfg['protocol_id']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest();json_save(ROOT/'config/locked_run.json',cfg)
    json_save(ROOT/'config/lock_receipt.json',dict(protocol_id=cfg['protocol_id'],locked_sha256=sha(ROOT/'config/locked_run.json'),fixed_steps=cfg['staged']['steps'],no_test_used=True,training_calibration_rules_locked=True))
    status('plan_locked',protocol_id=cfg['protocol_id'],candidate_steps=3000,judgement_steps=3000,deadline_unix=budget['deadline_unix'])
    for action in ('initialize','candidate','judgement','calibration','mechanism','validation','test'):stage(action)
    status('completed')

if __name__=='__main__':
    try:run()
    except Exception:
        (ROOT/'logs/failure.log').write_text(traceback.format_exc());status('failed',error=traceback.format_exc())
    finally:
        subprocess.run([sys.executable,str(ROOT/'scripts/archive.py')],cwd=ROOT,env=env)
