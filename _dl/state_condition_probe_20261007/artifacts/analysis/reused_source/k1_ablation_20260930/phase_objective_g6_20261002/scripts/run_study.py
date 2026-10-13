"""Sequential preflight, step200 gate, resume3000 and full872 evaluation."""
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG = ROOT / 'configs/G6_phase01.yaml'
RUN = ROOT / 'runs/G6_phase01/seed_2026'
LOGS = ROOT / 'logs'
LOGS.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTHONUNBUFFERED='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')


def status(stage, state):
    (ROOT / 'status.json').write_text(json.dumps({'stage': stage, 'state': state, 'utc_time': time.time()}))


def call(label, script, args):
    status(label, 'running')
    print('START', label, flush=True)
    with (LOGS / (label + '.log')).open('w') as log:
        subprocess.run([sys.executable, str(HERE / script), '--config', str(CONFIG), *args],
                       stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
    print('DONE', label, flush=True)


def gate(step):
    rows = list(csv.DictReader((RUN / 'logs/train.csv').open()))
    assert [int(r['step']) for r in rows] == list(range(1, step + 1))
    assert all(math.isfinite(float(v)) for r in rows for v in r.values())
    ck = torch.load(RUN / f'checkpoints/step_{step}.pt', map_location='cpu', weights_only=False)
    assert ck['step'] == step and ck['phase_weight'] == 0.1
    assert ck['lambda_feature_alignment'] == 0
    groups = ck['optimizer']['param_groups']
    assert len(groups) == 1 and groups[0]['name'] == 'tasknet_net_g' and groups[0]['lr'] == 1e-5
    adam = ck['optimizer']['state']
    assert len(adam) == len(groups[0]['params'])
    assert all(int(s['step']) == step for s in adam.values())
    assert all(torch.isfinite(v).all() for s in adam.values() for v in s.values() if torch.is_tensor(v))
    assert all(torch.isfinite(v).all() for v in ck['tasknet_net_g'].values())
    audit = json.loads((RUN / f'diagnostics/frozen_step_{step}.json').read_text())
    assert audit['passed']
    result = {'step': step, 'continuous_finite_steps': len(rows), 'optimizer_param_tensors': len(adam),
              'netg_clip_triggered_steps': sum(float(r['tasknet_grad_norm']) > 1 for r in rows),
              'frozen_audit_passed': True, 'passed': True}
    (ROOT / f'audit_step_{step}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


try:
    call('preflight', 'run_g6_phase.py', ['preflight'])
    call('step0_consistency', 'run_g6_phase.py', ['forwardcheck', '--others',
         str(HERE / 'reference/G6_original.yaml'), '--limit', '3'])
    # Forwardcheck prints actual differences; enforce its numerical assertions.
    output = (LOGS / 'step0_consistency.log').read_text()
    result = json.loads(output[output.find('{\n  "group"'):])
    assert result['n_images'] == 3
    assert all(r['passed'] and r['max_abs_diff_over_images'] == 0 for r in result['comparisons'].values())
    (ROOT / 'step0_consistency.json').write_text(json.dumps(result, indent=2))
    call('train_0_200', 'run_g6_phase.py', ['train', '--steps', '200'])
    gate(200)
    call('train_200_3000', 'run_g6_phase.py', ['train', '--steps', '3000', '--resume'])
    gate(3000)
    call('validation_872', 'evaluate_detailed.py', ['evaluate', '--checkpoint', 'step_3000', '--split', 'validation'])
    status('complete', 'complete')
except Exception:
    status('failed', 'failed')
    raise
