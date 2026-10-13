"""Matched phase0.1 G6/K1/K2; inherit data, optimizer, clipping and evaluators.

K2 task weights are normalized to (1/3,2/3); K1 has one actual forward.
All inherited two-round fields on K1 are aliases, never a second prediction.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import torch
import yaml

HERE = Path(__file__).resolve().parent


def configure(config):
    arm = config['study']['arm']
    if arm == 'G6':
        import run_g6_phase as extension
        base = extension.base
    else:
        import run_phase_objective as extension
        base = extension.base
    original_build = base.build_components
    original_checkpoint = base.checkpoint_payload
    original_losses = base.forward_losses
    original_forward = getattr(base, 'recurrent_forward', None)
    original_protocol = getattr(base, 'protocol_id', None)

    if original_protocol:
        def read_config(path):
            cfg = yaml.safe_load(Path(path).read_text())
            assert cfg['schema_version'] == 2 and cfg['recurrent']['feedback'] is False
            assert cfg['data']['condition'] == 'raw_mixed_rgb'
            expected = [1.0] if cfg['study']['arm'] == 'K1' else [1 / 3, 2 / 3]
            assert cfg['recurrent']['round_loss_weights'] == expected
            return cfg
        base.read_yaml = read_config
        def protocol(cfg):
            obj = copy.deepcopy(cfg)
            obj['matched_extension_sha256'] = base.sha256_file(HERE / 'run_matched.py')
            return original_protocol(obj)
        base.protocol_id = protocol

    def build(cfg, device):
        model, task, criterion, raw = original_build(cfg, device)
        model._matched_arm = cfg['study']['arm']
        hashes = lambda state: {n: hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                                for n, t in state.items()}
        initial = {'model': hashes(model.state_dict()), 'netg': hashes(task.net_g.state_dict())}
        reference = torch.load(cfg['study']['official_netg_reference'], map_location='cpu', weights_only=True)
        actual = {n: p.detach().cpu().float() for n, p in task.net_g.named_parameters()}
        assert actual.keys() == reference.keys()
        assert all(torch.equal(actual[n], reference[n]) for n in actual)
        initial.update({'seed': cfg['seed'], 'arm': cfg['study']['arm'],
                        'official_netg_matched': True,
                        'source_schedule_sha256': base.sha256_file(Path(cfg['paths']['schedule']))})
        base.atomic_json(Path(cfg['paths']['output_root']) / 'diagnostics/initial_pair_audit.json', initial)
        return model, task, criterion, raw

    def single_forward(model, task, batch, need_alignment=False):
        first = model(batch['polar'], batch['condition'], project_features=need_alignment)
        if not torch.is_grad_enabled():
            first.raw_features.clear()
        if torch.is_grad_enabled() and model.activation_checkpoint:
            t1, f1 = base.checkpoint(task, first.refined_polar, batch['official_rgb'], None,
                                     use_reentrant=False, preserve_rng_state=True)
        else:
            t1, f1 = task(first.refined_polar, batch['official_rgb'], None)
        if not torch.is_grad_enabled() and not need_alignment:
            f1.clear()
        out = copy.copy(first)
        out.first = first
        out.t1 = out.t2 = t1
        out.first_task_features = f1
        return out

    def losses(model, task, criterion, batch):
        if arm == 'K1':
            out = single_forward(model, task, batch, need_alignment=True)
            out.loss1 = out.loss2 = criterion(out.t1, batch['target'])
            task_loss = out.loss1
            al = base.feature_alignment_loss(out.first.projected_features, out.first_task_features)
            assert torch.isfinite(task_loss) and torch.isfinite(al)
            return out, out.t1, task_loss, al
        out, pred, tl, al = original_losses(model, task, criterion, batch)
        return out, pred, tl, al

    def checkpoint(model, task, optimizer, step, coefficient, metrics=None, *extra):
        value = original_checkpoint(model, task, optimizer, step, coefficient, metrics, *extra)
        value['matched_arm'] = arm
        value['effective_task_weights'] = [1 / 3, 2 / 3] if arm == 'K2' else [1.0]
        value['actual_rounds'] = 2 if arm == 'K2' else 1
        return value

    base.build_components = build
    base.checkpoint_payload = checkpoint
    if arm != 'G6':
        base.forward_losses = losses
        if arm == 'K1':
            base.recurrent_forward = single_forward
            base.run_preflight = lambda cfg, device: preflight_single(base, cfg, device)
    return base


def preflight_single(base, cfg, device):
    root = Path(cfg['paths']['output_root'])
    assert not any((root / 'checkpoints').glob('*.pt'))
    calibration = base.calibrate(cfg, device)
    base.set_seed(cfg['seed'])
    model, task, criterion, raw = base.build_components(cfg, device)
    batch = base.unpack_batch(next(iter(base.scheduled_loader(cfg, raw, 1))), device)
    torch.cuda.reset_peak_memory_stats(device)
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
    loss = tl + calibration['lambda_feature_alignment'] * al
    assert out.t1 is out.t2 and out.first.refined_polar is out.refined_polar
    loss.backward()
    netg_ids = {id(p) for p in task.net_g.parameters()}
    prior = [p for p in task.model.parameters() if id(p) not in netg_ids]
    payload = {'protocol_id': base.protocol_id(cfg), 'actual_forward_calls': calls,
               'single_round_aliases_verified': True,
               'grad_norm_r2p': base.gradient_norm(model.r2p.parameters()),
               'grad_norm_projections': base.gradient_norm(model.projections.parameters()),
               'grad_norm_netg': base.gradient_norm(task.net_g.parameters()),
               'prior_frozen': all(not p.requires_grad and p.grad is None for p in prior),
               'unused_feedback_has_grad': any(p.grad is not None for p in model.feedback_adapter.parameters()) or model.feedback_scale.grad is not None,
               'all_gradients_finite': all(torch.isfinite(p.grad).all() for p in list(model.parameters()) + list(task.parameters()) if p.grad is not None),
               'loss_finite': bool(torch.isfinite(loss)),
               'peak_vram_mb': torch.cuda.max_memory_allocated(device) / 2**20,
               'effective_task_weights': [1.0], 'lambda_feature_alignment': calibration['lambda_feature_alignment']}
    payload['all_gradients_finite'] = bool(payload['all_gradients_finite'])
    payload['passed'] = (calls == {'r2p': 1, 'tasknet': 1} and payload['prior_frozen']
                         and not payload['unused_feedback_has_grad'] and payload['all_gradients_finite']
                         and payload['loss_finite'] and all(payload[k] > 0 for k in ('grad_norm_r2p', 'grad_norm_projections', 'grad_norm_netg')))
    base.atomic_json(root / 'diagnostics/recurrent_preflight.json', payload)
    assert payload['passed'], payload
    return payload


if __name__ == '__main__':
    path = Path(sys.argv[sys.argv.index('--config') + 1])
    config = yaml.safe_load(path.read_text())
    configure(config).main()
