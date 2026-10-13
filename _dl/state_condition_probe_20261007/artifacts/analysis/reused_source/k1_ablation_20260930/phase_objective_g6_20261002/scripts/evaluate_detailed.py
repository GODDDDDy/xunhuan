"""Retain losses using a hook on the original G6 evaluator; no extra TaskNet calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import run_g6_phase as study

ap = argparse.ArgumentParser()
ap.add_argument('--config', required=True)
ap.add_argument('operation', choices=['evaluate'])
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--split', required=True)
args = ap.parse_args()
cfg = yaml.safe_load(Path(args.config).read_text())
base = study.base
base.add_source_path(cfg)
root = Path(cfg['paths']['output_root'])
old_build = base.build_components
old_unpack = base.unpack_batch
rows = []
context = {}


def unpack(raw, device):
    batch = old_unpack(raw, device)
    context['batch'] = batch
    return batch


def build(config, device):
    model, task, criterion, base_config = old_build(config, device)

    def hook(module, inputs, output):
        batch = context['batch']
        prediction = output[0]
        target = batch['target']
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = [s.clone() for s in torch.cuda.get_rng_state_all()]
        l1 = float(F.l1_loss(prediction, target))
        st = float(1 - sys.modules['polarapp.losses'].ssim(prediction, target))
        ph = float(criterion.phase(prediction, target))
        pe = float(criterion.perceptual(prediction, target))
        raw = ph / 0.1
        parts = {'l1': l1, 'ssim_term': st, 'phase_raw': raw, 'phase_weighted': ph,
                 'perceptual_weighted': pe, 'total_training_coefficient': l1 + st + ph + pe,
                 'total_common_phase1': l1 + st + raw + pe,
                 'total_common_phase01': l1 + st + 0.1 * raw + pe}
        assert all(np.isfinite(v) for v in parts.values())
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
        rows.append({'sample_id': batch['sample_id'][0], 'scene': batch['scene'][0], 'loss_parts': parts})
        if len(rows) % 25 == 0:
            print(f'Evaluated {len(rows)} images', flush=True)

    task.register_forward_hook(hook)
    return model, task, criterion, base_config


base.build_components = build
base.unpack_batch = unpack
tick = time.time()
result = base.run_evaluation(cfg, torch.device('cuda'), args.checkpoint, args.split)
metrics_path = Path(result['written'])
metrics = json.loads(metrics_path.read_text())
assert len(rows) == int(metrics['frames']) == 872
assert [(r['sample_id'], r['scene']) for r in rows] == [(r['sample_id'], r['scene']) for r in metrics['per_image']]
checkpoint_sha = base.sha256_file(base.resolve_checkpoint(cfg, args.checkpoint))
metrics['checkpoint_sha256'] = checkpoint_sha
metrics['phase_weight'] = 0.1
base.atomic_json(metrics_path, metrics)
scenes = sorted({r['scene'] for r in rows})
keys = rows[0]['loss_parts']
payload = {'frames': len(rows), 'scenes': len(scenes), 'phase_weight': 0.1,
           'checkpoint_sha256': checkpoint_sha, 'elapsed_seconds': time.time() - tick,
           'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'no_additional_tasknet_calls': True, 'rng_unchanged_by_loss_arithmetic': True,
           'image_mean': {k: float(np.mean([r['loss_parts'][k] for r in rows])) for k in keys},
           'scene_macro': {k: float(np.mean([np.mean([r['loss_parts'][k] for r in rows if r['scene'] == s]) for s in scenes])) for k in keys},
           'per_image': rows}
base.atomic_json(root / 'metrics/step_3000_validation_loss_parts.json', payload)
print(json.dumps({k: v for k, v in metrics.items() if k not in ('per_image', 'scene_summary')}, indent=2))
