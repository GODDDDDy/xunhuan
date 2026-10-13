"""Two-arm 1000-step pilot with matched 200-step freeze/optimizer gates."""
import csv,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent.parent
lock=json.loads((ROOT/'training_lock.json').read_text());arms=lock['conditions']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
assert all(sha(p)==h for p,h in lock['locked_file_sha256'].items())
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
done=[]
def status(arm,stage,state,**extra):
    p=ROOT/'pilot_status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(arm=arm,stage=stage,state=state,completed_stages=done,time_unix=time.time(),**extra),indent=2));tmp.replace(p)
def call(arm,stage,args):
    log=ROOT/'logs'/f'{arm}_{stage}.log';assert not log.exists()
    command=[sys.executable,str(ROOT/'scripts/run_probe.py'),'--config',str(ROOT/'configs'/f'{arm}.yaml'),*args]
    with log.open('w') as f:
        child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        status(arm,stage,'running',child_pid=child.pid,logfile=str(log));rc=child.wait()
    if rc:
        status(arm,stage,'failed',returncode=rc,logfile=str(log));raise RuntimeError(f'{stage}: {log}')
    done.append(f'{arm}/{stage}');print(f'DONE {arm} {stage}',flush=True)
def gate(arm,step):
    run=ROOT/'runs'/arm/'seed_2026';rows=list(csv.DictReader((run/'logs/train.csv').open()))
    assert [int(r['step']) for r in rows]==list(range(1,step+1))
    assert all(math.isfinite(float(v)) for r in rows for v in r.values())
    assert all(float(r['tasknet_grad_norm'])==float(r['projection_grad_norm'])==float(r['feedback_adapter_grad_norm'])==0 for r in rows)
    ck=torch.load(run/f'checkpoints/step_{step}.pt',map_location='cpu',weights_only=False)
    init=json.loads((run/'diagnostics/initial_pair_audit.json').read_text())
    assert ck['step']==step and ck['condition_probe']==arm and ck['effective_task_weights']==[0.0,1.0]
    assert ck['full_frozen_audit_passed'] and ck['baseline_task_not_trained'] and ck['identity_output_initialization']
    adam=ck['optimizer']['state'];assert len(adam)==18 and set(adam)==set(init['trainable_model_parameter_indices'])
    assert all(int(v['step'])==step for v in adam.values())
    assert all(torch.isfinite(t).all() for v in adam.values() for t in v.values() if torch.is_tensor(t))
    assert all(torch.isfinite(t).all() for t in ck['r2p_and_projections'].values())
    assert json.loads((run/f'diagnostics/full_frozen_step_{step}.json').read_text())['passed']
    assert json.loads((run/'diagnostics/recurrent_preflight.json').read_text())['passed']
    payload=dict(step=step,adam_tensors=18,all_frozen=True,continuous_finite_rows=len(rows),passed=True)
    (run/f'diagnostics/gate_step_{step}.json').write_text(json.dumps(payload,indent=2))
    plan=run/'RUN_PLAN.md'
    if plan.exists():plan.replace(run/f'RUN_PLAN_INHERITED_STAGE_{step}.md')
    plan.write_text('# Actual condition-representation pilot\n\nFrozen G6 first prediction; R2P condition='+arm+'.\n'
        'One R2P call and two frozen TaskNet calls; same restored diffusion RNG.\n'
        'Only final corrected task loss supervised, phase0.1; lambda0.\n'
        'R2P e42 features, zero residual head; only18 R2P tensors trained.\n'
        'Both arms seed2026, same initialization, data schedule and 1000 additional steps.\n'
        'Use fixed step1000 validation; no test or checkpoint selection.\n')
for arm in arms:
    call(arm,'preflight',['preflight'])
for arm in arms:
    call(arm,'train_0_200',['train','--steps','200']);gate(arm,200)
left,right=[ROOT/'runs'/a/'seed_2026' for a in arms]
l,r=[json.loads((p/'diagnostics/initial_pair_audit.json').read_text()) for p in (left,right)]
assert l['model']==r['model'] and l['netg']==r['netg'] and l['full_frozen_state']==r['full_frozen_state']
assert l['source_schedule_sha256']==r['source_schedule_sha256']
ls,rs=[list(csv.DictReader((p/'logs/train.csv').open())) for p in (left,right)]
assert [v['diffusion_seed'] for v in ls]==[v['diffusion_seed'] for v in rs]
(ROOT/'diagnostics/paired_200_gate.json').write_text(json.dumps(dict(passed=True,initial_tensors_equal=True,schedule_rng_equal=True),indent=2))
for arm in arms:
    call(arm,'train_200_1000',['train','--steps','1000','--resume']);gate(arm,1000)
for arm in arms:
    call(arm,'validation_872',['evaluate','--checkpoint','step_1000','--split','validation'])
status('both','complete','complete');print('PAIRED PILOT COMPLETE',flush=True)
