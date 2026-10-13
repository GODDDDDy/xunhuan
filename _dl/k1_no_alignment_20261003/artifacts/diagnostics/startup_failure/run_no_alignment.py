"""One-factor K1 ablation: lambda=0, retaining the exact model/graph/optimizer."""
import hashlib
import json
from pathlib import Path
import sys
import torch
import yaml
import run_matched

HERE = Path(__file__).resolve().parent


def configure(cfg):
    assert cfg['study']['arm'] == 'K1' and cfg['study']['fixed_lambda'] == 0.0
    assert cfg['study']['phase_weight'] == 0.1
    base = run_matched.configure(cfg)
    parent_protocol = base.protocol_id
    def protocol(config):
        obj = json.loads(json.dumps(config))
        obj['no_alignment_extension_sha256'] = base.sha256_file(Path(__file__))
        return parent_protocol(obj)
    base.protocol_id = protocol
    parent_build, parent_checkpoint = base.build_components, base.checkpoint_payload

    def build(config, device):
        model, task, criterion, raw = parent_build(config, device)
        model._projection_start = {n: v.detach().cpu().clone() for n, v in model.projections.state_dict().items()}
        root = Path(config['paths']['output_root'])
        init = json.loads((root / 'diagnostics/initial_pair_audit.json').read_text())
        reference = json.loads(Path(config['study']['aligned_initial_audit']).read_text())
        assert init['model'] == reference['model'] and init['netg'] == reference['netg']
        assert init['source_schedule_sha256'] == reference['source_schedule_sha256']
        init['paired_aligned_initialization_matched'] = True
        base.atomic_json(root / 'diagnostics/initial_pair_audit.json', init)
        return model, task, criterion, raw

    def checkpoint(model, task, optimizer, step, coefficient, metrics=None, *extra):
        value = parent_checkpoint(model, task, optimizer, step, coefficient, metrics, *extra)
        assert coefficient == 0.0
        current = model.projections.state_dict()
        changed = [n for n, v in current.items() if not torch.equal(v.detach().cpu(), model._projection_start[n])]
        assert not changed, changed
        audit = {'step': step, 'checked_tensors': len(current), 'changed_keys': changed, 'passed': not changed,
                 'reason': 'zero alignment coefficient; projections have zero gradient; weight_decay=0'}
        base.atomic_json(model._study_output_root / 'diagnostics' / f'projection_unchanged_step_{step}.json', audit)
        value['alignment_ablation'] = 'lambda0_same_graph_optimizer'
        value['projections_unchanged'] = True
        return value

    base.build_components = build
    base.checkpoint_payload = checkpoint
    base.run_preflight = lambda config, device: preflight(base, config, device)
    return base


def preflight(base, cfg, device):
    root = Path(cfg['paths']['output_root'])
    assert not any((root / 'checkpoints').glob('*.pt'))
    calibration = base.calibrate(cfg, device)
    assert calibration['lambda_feature_alignment'] == 0.0
    base.set_seed(cfg['seed'])
    model, task, criterion, raw = base.build_components(cfg, device)
    batch = base.unpack_batch(next(iter(base.scheduled_loader(cfg, raw, 1))), device)
    calls = {'r2p': 0, 'tasknet': 0}
    def count(key):
        def hook(*args):
            calls[key] += 1
        return hook
    hooks = [model.register_forward_hook(count('r2p')), task.register_forward_hook(count('tasknet'))]
    base.set_step_seed(cfg['seed'], 'preflight', 1, batch['sample_id'])
    out, _, tl, al = base.forward_losses(model, task, criterion, batch)
    for hook in hooks:
        hook.remove()
    # Keep the original alignment branch and 0*al graph: no model/optimizer change.
    loss = tl + 0.0 * al
    params = tuple(model.r2p.parameters())
    only_task = torch.autograd.grad(tl, params, retain_graph=True)
    with_zero_alignment = torch.autograd.grad(loss, params, retain_graph=True)
    gradient_max_diff = max(float((x-y).abs().max()) for x, y in zip(only_task, with_zero_alignment))
    loss.backward()
    netg_ids = {id(p) for p in task.net_g.parameters()}
    prior = [p for p in task.model.parameters() if id(p) not in netg_ids]
    payload = {'protocol_id': base.protocol_id(cfg), 'actual_forward_calls': calls,
               'single_round_aliases_verified': out.t1 is out.t2,
               'lambda_feature_alignment': 0.0, 'alignment_scalar_retained_as_diagnostic': float(al.detach()),
               'task_only_vs_total_r2p_gradient_max_diff': gradient_max_diff,
               'gradient_parity_tolerance': 1e-7,
               'grad_norm_r2p': base.gradient_norm(model.r2p.parameters()),
               'grad_norm_projections': base.gradient_norm(model.projections.parameters()),
               'grad_norm_netg': base.gradient_norm(task.net_g.parameters()),
               'prior_frozen': all(not p.requires_grad and p.grad is None for p in prior),
               'unused_feedback_has_grad': any(p.grad is not None for p in model.feedback_adapter.parameters()) or model.feedback_scale.grad is not None,
               'all_gradients_finite': all(bool(torch.isfinite(p.grad).all()) for p in list(model.parameters()) + list(task.parameters()) if p.grad is not None),
               'loss_finite': bool(torch.isfinite(loss)), 'effective_task_weights': [1.0]}
    payload['passed'] = (calls == {'r2p':1,'tasknet':1} and payload['single_round_aliases_verified']
                         and gradient_max_diff <= 1e-7 and payload['grad_norm_projections'] == 0
                         and payload['grad_norm_r2p'] > 0 and payload['grad_norm_netg'] > 0
                         and payload['prior_frozen'] and not payload['unused_feedback_has_grad']
                         and payload['all_gradients_finite'] and payload['loss_finite'])
    base.atomic_json(root / 'diagnostics/recurrent_preflight.json', payload)
    assert payload['passed'], payload
    return payload


if __name__ == '__main__':
    cfg = yaml.safe_load(Path(sys.argv[sys.argv.index('--config')+1]).read_text())
    configure(cfg).main()
