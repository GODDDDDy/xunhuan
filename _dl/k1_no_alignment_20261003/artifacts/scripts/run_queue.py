"""Sequential durable queue; resume exact checkpoint after each successful 200-step gate."""
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
lock = json.loads((ROOT / 'protocol_lock.json').read_text())
LOGS = ROOT / 'logs'
LOGS.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
           PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
completed = []


def status(arm, seed, stage, state, **extra):
    payload = {'arm': arm, 'seed': seed, 'stage': stage, 'state': state,
               'completed_runs': completed, 'new_runs_total': 3, 'time_unix': time.time(), **extra}
    temporary = ROOT / 'status.json.tmp'
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(ROOT / 'status.json')


def call(arm, seed, stage, args, evaluator=False):
    logfile = LOGS / f'{arm}_seed{seed}_{stage}.log'
    attempt = 1
    while logfile.exists():
        attempt += 1
        logfile = LOGS / f'{arm}_seed{seed}_{stage}_attempt{attempt}.log'
    script = HERE / ('evaluate_matched.py' if evaluator else 'run_no_alignment.py')
    config = ROOT / 'configs' / f'{arm}_seed{seed}.yaml'
    command = [sys.executable, str(script), '--config', str(config), *args]
    print(f'START {arm} seed{seed} {stage}', flush=True)
    with logfile.open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
        status(arm, seed, stage, 'running', child_pid=child.pid, logfile=str(logfile))
        returncode = child.wait()
    if returncode:
        status(arm, seed, stage, 'failed', returncode=returncode, logfile=str(logfile))
        raise RuntimeError(f'{arm} seed{seed} {stage} failed; see {logfile}')
    print(f'DONE {arm} seed{seed} {stage}', flush=True)


def gate(arm, seed, step):
    run = ROOT / 'runs' / arm / f'seed_{seed}'
    rows = list(csv.DictReader((run / 'logs/train.csv').open()))
    assert [int(r['step']) for r in rows] == list(range(1, step + 1))
    assert all(math.isfinite(float(v)) for row in rows for v in row.values())
    ck = torch.load(run / f'checkpoints/step_{step}.pt', map_location='cpu', weights_only=False)
    assert ck['step'] == step and ck['phase_weight'] == 0.1 and ck['matched_arm'] == arm
    assert ck['effective_task_weights'] == ([1 / 3, 2 / 3] if arm == 'K2' else [1.0])
    assert ck['lambda_feature_alignment'] == (0.0 if arm == 'G6' else lock['K1_K2_alignment_lambda'])
    adam = ck['optimizer']['state']
    assert len(adam) == (436 if arm == 'G6' else 457)
    assert all(int(v['step']) == step for v in adam.values())
    assert all(torch.isfinite(t).all() for v in adam.values() for t in v.values() if torch.is_tensor(t))
    assert all(torch.isfinite(t).all() for t in ck['tasknet_net_g'].values())
    assert all(torch.isfinite(t).all() for t in ck['r2p_and_projections'].values())
    audit = json.loads((run / f'diagnostics/frozen_step_{step}.json').read_text())
    assert audit['passed']
    assert ck['alignment_ablation'] == 'lambda0_same_graph_optimizer' and ck['projections_unchanged']
    assert json.loads((run / f'diagnostics/projection_unchanged_step_{step}.json').read_text())['passed']
    initial = json.loads((run / 'diagnostics/initial_pair_audit.json').read_text())
    assert initial['official_netg_matched']
    counterpart = ROOT / 'runs/K1' / f'seed_{seed}/diagnostics/initial_pair_audit.json'
    if arm == 'K2' and counterpart.is_file():
        reference = json.loads(counterpart.read_text())
        assert initial['model'] == reference['model'] and initial['netg'] == reference['netg']
    result = {'arm': arm, 'seed': seed, 'step': step, 'finite_continuous_rows': len(rows),
              'adam_state_tensors': len(adam), 'frozen_passed': True,
              'netg_clip_triggered_steps': sum(float(r['tasknet_grad_norm']) > 1 for r in rows),
              'passed': True}
    (run / f'diagnostics/gate_step_{step}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    # The inherited plan describes the old K2 experiment; replace only its new-run copy.
    plan = run / 'RUN_PLAN.md'
    if plan.is_file():
        archive = run / f'RUN_PLAN_INHERITED_STAGE_{step}.md'
        plan.replace(archive)
    plan.write_text(f'# Actual matched protocol\n\nArm={arm}; seed={seed}; total steps=3000; phase=0.1.\n'
                    f'Task weights={ck["effective_task_weights"]}; alignment={ck["lambda_feature_alignment"]}.\n'
                    'Fixed step3000 selection; inherited best.pt is not used.\n'
                    'K1 round2 log/metric columns are aliases of the one actual forward.\n')


for seed, arm in lock['new_run_order']:
    call(arm, seed, 'preflight', ['preflight'])
    call(arm, seed, 'train_0_200', ['train', '--steps', '200'])
    gate(arm, seed, 200)
    call(arm, seed, 'train_200_3000', ['train', '--steps', '3000', '--resume'])
    gate(arm, seed, 3000)
    call(arm, seed, 'validation_872', [], evaluator=True)
    completed.append({'arm': arm, 'seed': seed, 'finished_unix': time.time()})
    status(arm, seed, 'run_complete', 'running')
status('all', 0, 'complete', 'complete')
print('ALL THREE NO-ALIGNMENT RUNS COMPLETED', flush=True)
