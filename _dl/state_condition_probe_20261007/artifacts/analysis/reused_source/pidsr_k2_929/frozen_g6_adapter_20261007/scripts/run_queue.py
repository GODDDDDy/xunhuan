"""Sequential frozen-G6 adapter queue, gated after 200 and 3000 updates."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
lock=json.loads((ROOT/'protocol_lock.json').read_text())
LOGS=ROOT/'logs';LOGS.mkdir(exist_ok=True)
env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
completed=[]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert all(sha(p)==h for p,h in lock['locked_file_sha256'].items())
assert all(sha(v['path'])==v['sha256'] for ref in lock['references'].values() for v in ref.values())
def status(arm,seed,stage,state,**extra):
    payload=dict(arm=arm,seed=seed,stage=stage,state=state,completed_runs=completed,new_runs_total=3,time_unix=time.time(),**extra)
    temp=ROOT/'status.json.tmp';temp.write_text(json.dumps(payload,indent=2));temp.replace(ROOT/'status.json')
def call(arm,seed,stage,args,evaluator=False):
    logfile=LOGS/f'{arm}_seed{seed}_{stage}.log'
    attempt=1
    while logfile.exists():
        attempt+=1;logfile=LOGS/f'{arm}_seed{seed}_{stage}_attempt{attempt}.log'
    script=HERE/('evaluate_matched.py' if evaluator else 'run_frozen_adapter.py')
    command=[sys.executable,str(script),'--config',str(ROOT/'configs'/f'{arm}_seed{seed}.yaml'),*args]
    print(f'START {seed} {stage}',flush=True)
    with logfile.open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        status(arm,seed,stage,'running',child_pid=child.pid,logfile=str(logfile))
        rc=child.wait()
    if rc:
        status(arm,seed,stage,'failed',returncode=rc,logfile=str(logfile))
        raise RuntimeError(f'Failed {stage}; {logfile}')
    print(f'DONE {seed} {stage}',flush=True)
def gate(arm,seed,step):
    run=ROOT/'runs'/arm/f'seed_{seed}'
    rows=list(csv.DictReader((run/'logs/train.csv').open()))
    assert [int(r['step']) for r in rows]==list(range(1,step+1))
    assert all(math.isfinite(float(v)) for r in rows for v in r.values())
    assert all(float(r['tasknet_grad_norm'])==0 and float(r['projection_grad_norm'])==0 for r in rows)
    ck=torch.load(run/f'checkpoints/step_{step}.pt',map_location='cpu',weights_only=False)
    assert ck['step']==step and ck['phase_weight']==0.1 and ck['matched_arm']=='K1'
    assert ck['effective_task_weights']==[1.0] and ck['lambda_feature_alignment']==0.0
    assert ck['experiment']=='frozen_G6_identity_adapter' and ck['full_frozen_audit_passed']
    assert ck['optimizer_updates_only_r2p'] and ck['identity_output_initialization']
    ref=lock['references'][f'G6_seed{seed}']['checkpoint']
    assert ck['frozen_G6_checkpoint_sha256']==ref['sha256']
    source=torch.load(ref['path'],map_location='cpu',weights_only=False)['tasknet_net_g']
    assert ck['tasknet_net_g'].keys()==source.keys()
    assert all(torch.equal(t,source[n]) for n,t in ck['tasknet_net_g'].items())
    initial=json.loads((run/'diagnostics/initial_pair_audit.json').read_text())
    adam=ck['optimizer']['state']
    assert len(adam)==18 and set(adam)==set(initial['trainable_model_parameter_indices'])
    assert all(int(v['step'])==step for v in adam.values())
    assert all(torch.isfinite(t).all() for v in adam.values() for t in v.values() if torch.is_tensor(t))
    assert all(torch.isfinite(t).all() for t in ck['r2p_and_projections'].values())
    for n in ('frozen','full_frozen'):
        assert json.loads((run/f'diagnostics/{n}_step_{step}.json').read_text())['passed']
    initial=json.loads((run/'diagnostics/initial_pair_audit.json').read_text())
    assert initial['all_tasknet_parameters_frozen'] and initial['trainable_r2p_parameter_tensors']==18
    pf=json.loads((run/'diagnostics/recurrent_preflight.json').read_text())
    assert pf['passed'] and pf['proxy_step0_parity']['n']==28 and pf['proxy_step0_parity']['all_passed']
    result=dict(arm=arm,seed=seed,step=step,finite_continuous_rows=len(rows),adam_state_tensors=len(adam),
        frozen_G6_exact=True,full_frozen_passed=True,passed=True)
    (run/f'diagnostics/gate_step_{step}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    plan=run/'RUN_PLAN.md'
    if plan.is_file():plan.replace(run/f'RUN_PLAN_INHERITED_STAGE_{step}.md')
    plan.write_text('# Actual frozen G6 adapter protocol\n\nOnly R2P trained, all TaskNet frozen; no alignment.\n'
        'R2P pretrained e42 trunk with zero residual output head; initial identity.\n'
        'Source G6 trained 3000 steps; additional adapter 3000 steps, fixed final checkpoint.\n'
        'One actual forward; inherited round2 columns are aliases.\n')
for seed,arm in lock['new_run_order']:
    call(arm,seed,'preflight',['preflight'])
    call(arm,seed,'train_0_200',['train','--steps','200']);gate(arm,seed,200)
    call(arm,seed,'train_200_3000',['train','--steps','3000','--resume']);gate(arm,seed,3000)
    call(arm,seed,'validation_872',[],True)
    completed.append(dict(arm=arm,seed=seed,finished_unix=time.time()))
    status(arm,seed,'run_complete','running')
status('all',0,'complete','complete')
print('ALL THREE FROZEN G6 ADAPTER RUNS COMPLETED',flush=True)
