"""Call the original evaluator; count actual TaskNet forwards and label K1 aliases."""
import argparse
import json
from pathlib import Path
import time
import torch
import yaml
import run_matched

ap = argparse.ArgumentParser()
ap.add_argument('--config', required=True)
args = ap.parse_args()
cfg = yaml.safe_load(Path(args.config).read_text())
arm = cfg['study']['arm']
base = run_matched.configure(cfg)
base.add_source_path(cfg)
original_validate = base.validate
def fixed_noise_validate(model, task, loader, device, seed, *extra, **kwargs):
    # Isolate training-seed variability from stochastic diffusion evaluation.
    return original_validate(model, task, loader, device, 2026, *extra, **kwargs)
base.validate = fixed_noise_validate
build = base.build_components
expected_calls = 2 if arm == 'K2' else 1
calls = 0


def build_with_counter(config, device):
    result = build(config, device)
    def hook(*unused):
        global calls
        calls += 1
        if calls % (50 * expected_calls) == 0:
            print(f'Evaluated {calls // expected_calls} images', flush=True)
    result[1].register_forward_hook(hook)
    return result


base.build_components = build_with_counter
tick = time.time()
base.run_evaluation(cfg, torch.device('cuda:0'), 'step_3000', 'validation')
root = Path(cfg['paths']['output_root'])
path = root / 'metrics/step_3000_validation.json'
metrics = json.loads(path.read_text())
assert int(metrics['frames']) == 872 and int(metrics['scenes']) == 8
assert calls == 872 * expected_calls
if arm == 'K1':
    metrics['alias_notice'] = 'Inherited round2 fields alias the single real output; there was no second forward.'
    base.atomic_json(root / 'metrics/inherited_evaluator_K1_aliases.json', metrics)
    assert all(r['round1'] == r['round2'] for r in metrics['per_image'])
    metrics['per_image'] = [dict(sample_id=r['sample_id'], scene=r['scene'], **r['round1']) for r in metrics['per_image']]
    for key in ('round1', 'round2', 'iteration_gain', 'polar_update', 'feedback_scale', 'alias_notice'):
        metrics.pop(key, None)
checkpoint = root / 'checkpoints/step_3000.pt'
metrics.update({'checkpoint_sha256': base.sha256_file(checkpoint), 'arm': arm,
                'seed': cfg['seed'], 'phase_weight': 0.1,
                'training_seed': cfg['seed'], 'evaluation_seed': 2026,
                'effective_task_weights': cfg['study']['effective_task_weights'],
                'actual_tasknet_forwards': calls, 'actual_rounds': expected_calls,
                'evaluation_seconds': time.time() - tick,
                'scope': 'full872 validation; exploratory; three paired seeds'})
base.atomic_json(path, metrics)
print(json.dumps({k: v for k, v in metrics.items() if k not in ('per_image', 'scene_summary')}, indent=2))
