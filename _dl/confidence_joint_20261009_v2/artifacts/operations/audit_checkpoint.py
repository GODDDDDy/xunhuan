"""CPU-only audit of the first saved confidence checkpoint and locked inputs."""
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path('/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/confidence_joint_20261009_v2')
checkpoint = ROOT / 'arms/A2_dual/run/checkpoints/step_200.pt'
state = torch.load(checkpoint, map_location='cpu', weights_only=False)
steps = [float(x['step']) for x in state['optimizer']['state'].values()]
groups = [g['name'] for g in state['optimizer']['param_groups']]
assert state['step'] == 200 and min(steps) == max(steps) == 200
assert groups == ['r2p_and_alignment', 'tasknet_net_g', 'quality_head']
assert state['confidence_enabled'] and state['confidence_mode'] == 'dual'
assert state['actual_rounds'] == 2 and state['source_training_steps'] == 13000
assert state['source_sha256'] == 'eb17a5537bcd26583b73670a34a258f948056935f851b42c7a44eb62e1dd15ea'
lock = json.loads((ROOT / 'protocol_lock.json').read_text())
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert all(digest(ROOT / n) == h for n, h in lock['code_sha256'].items())
assert any(n.startswith('confidence_head.') for n in state['r2p_and_projections'])
value = dict(passed=True, step=200, optimizer_groups=groups, optimizer_state_count=len(steps),
             all_adam_steps=200, existing_modules_inherited=True,
             confidence_parameters_checkpointed=True, locked_code_unchanged=True,
             checkpoint_sha256=digest(checkpoint))
(ROOT / 'operations/first_checkpoint_audit.json').write_text(json.dumps(value, indent=2))
print(json.dumps(value), flush=True)
