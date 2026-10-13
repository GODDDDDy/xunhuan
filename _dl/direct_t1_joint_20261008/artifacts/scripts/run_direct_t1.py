"""Extend the existing matched trainer with a direct T1-image second condition.

All data, losses, optimizer, clipping and evaluation remain inherited. Load the
entire trained aligned K1 (R2P, projections, net_g); do not reset its output head.
The new study starts a fresh Adam optimizer, then resumes it across its stages.
"""
import copy
import gc
import hashlib
from pathlib import Path
import torch


def state_hashes(state):
    return {n: hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for n, t in state.items()}


def configure(cfg):
    import run_matched
    assert cfg['study']['arm'] == 'K2'
    assert cfg['study']['experiment'] == 'K1_warmstart_direct_T1_joint'
    assert cfg['recurrent']['condition_mode'] == 'direct_T1_image'
    assert cfg['recurrent']['feedback'] is False  # Legacy internal-feature adapter.
    base = run_matched.configure(cfg)
    parent_build = base.build_components
    parent_protocol = base.protocol_id
    parent_checkpoint = base.checkpoint_payload

    def protocol(config):
        obj = copy.deepcopy(config)
        obj['direct_T1_extension_sha256'] = base.sha256_file(Path(__file__))
        return parent_protocol(obj)
    base.protocol_id = protocol

    def build(config, device):
        model, task, criterion, raw = parent_build(config, device)
        source = Path(config['study']['source_K1_checkpoint'])
        assert base.sha256_file(source) == config['study']['source_K1_sha256']
        ck = torch.load(source, map_location='cpu', weights_only=False)
        assert ck['step'] == 3000 and ck['actual_rounds'] == 1 and ck['phase_weight'] == 0.1
        model.load_state_dict(ck['r2p_and_projections'], strict=True)
        task.net_g.load_state_dict(ck['tasknet_net_g'], strict=True)
        assert all(torch.equal(t.detach().cpu(), ck['r2p_and_projections'][n])
                   for n, t in model.state_dict().items())
        assert all(torch.equal(t.detach().cpu(), ck['tasknet_net_g'][n])
                   for n, t in task.net_g.state_dict().items())
        model.feedback_adapter.requires_grad_(False)
        model.feedback_scale.requires_grad_(False)
        model._image_feedback_enabled = True
        base.atomic_json(Path(config['paths']['output_root'])/'diagnostics/initial_pair_audit.json', {
            'seed': config['seed'], 'source_K1_checkpoint': str(source),
            'source_K1_sha256': config['study']['source_K1_sha256'],
            'r2p_and_projections_exactly_loaded': True, 'net_g_exactly_loaded': True,
            'r2p_output_head_reset': False, 'optimizer_initialization': 'fresh_Adam',
            'model': state_hashes(model.state_dict()), 'netg': state_hashes(task.net_g.state_dict()),
            'source_schedule_sha256': base.sha256_file(Path(config['paths']['schedule'])),
            'image_feedback_enabled': True, 'internal_feature_adapter_enabled': False})
        del ck
        return model, task, criterion, raw

    def forward(model, task, batch, need_alignment=False):
        first = model(batch['polar'], batch['condition'], project_features=need_alignment)
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        def task_forward(polar):
            if torch.is_grad_enabled() and model.activation_checkpoint:
                return base.checkpoint(task, polar, batch['official_rgb'], None,
                                       use_reentrant=False, preserve_rng_state=True)
            return task(polar, batch['official_rgb'], None)
        if not torch.is_grad_enabled():
            first.raw_features.clear()
        t1, f1 = task_forward(first.refined_polar)
        condition2 = t1.clamp(0, 1)  # Remains connected to T1/net_g.
        second = model(first.refined_polar, condition2, project_features=False)
        if not torch.is_grad_enabled():
            second.raw_features.clear()
            if not need_alignment:
                f1.clear()
        devices = list(range(torch.cuda.device_count())) if cuda_rng is not None else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            t2, f2 = task_forward(second.refined_polar)
        f2.clear()
        second.first, second.t1, second.t2 = first, t1, t2
        second.first_task_features = f1
        second.second_condition = condition2
        return second

    def checkpoint(model, task, optimizer, step, coefficient, metrics=None, *extra):
        value = parent_checkpoint(model, task, optimizer, step, coefficient, metrics, *extra)
        value.update(method='shared K2 direct T1 image feedback',
                     experiment='K1_warmstart_direct_T1_joint', image_feedback_enabled=True,
                     internal_feature_feedback_enabled=False, confidence_enabled=False,
                     source_K1_sha256=cfg['study']['source_K1_sha256'],
                     source_training_steps=3000, additional_training_steps=int(step),
                     total_training_steps_from_official=3000+int(step),
                     optimizer_warmstart=False, actual_rounds=2)
        return value

    base.build_components = build
    base.recurrent_forward = forward
    base.checkpoint_payload = checkpoint
    base.run_preflight = lambda config, device: preflight(base, config, device)
    return base


def preflight(base, cfg, device):
    """Startup integrity check, including gradient through the T1 image itself."""
    root = Path(cfg['paths']['output_root'])
    assert not any((root/'checkpoints').glob('*.pt'))
    calibration = base.calibrate(cfg, device)
    base.set_seed(cfg['seed'])
    model, task, criterion, raw = base.build_components(cfg, device)
    batch = base.unpack_batch(next(iter(base.scheduled_loader(cfg, raw, 1))), device)
    torch.cuda.reset_peak_memory_stats(device)
    calls = {'r2p': 0, 'tasknet': 0}
    def count(name):
        def hook(*args):
            calls[name] += 1
        return hook
    hooks = [model.register_forward_hook(count('r2p')), task.register_forward_hook(count('tasknet'))]
    base.set_step_seed(cfg['seed'], 'preflight', 1, batch['sample_id'])
    out, _, tl, al = base.forward_losses(model, task, criterion, batch)
    for hook in hooks:
        hook.remove()
    grad = torch.autograd.grad(out.loss2, out.t1, retain_graph=True)[0]
    payload = {'protocol_id': base.protocol_id(cfg), 'actual_forward_calls': calls,
               'image_feedback_enabled': True, 'confidence_enabled': False,
               'source_K1_sha256': cfg['study']['source_K1_sha256'],
               'second_condition_equals_clamped_T1': torch.equal(out.second_condition.detach(), out.t1.detach().clamp(0, 1)),
               'round2_loss_to_T1_image_grad_norm': float(grad.detach().norm()),
               'round2_loss_to_T1_image_grad_finite': bool(torch.isfinite(grad).all()),
               'polar_round_difference': float((out.refined_polar-out.first.refined_polar).detach().abs().mean()),
               'task_round_difference': float((out.t2-out.t1).detach().abs().mean()),
               'lambda_feature_alignment': calibration['lambda_feature_alignment']}
    del grad
    expected = [out.t1.detach().clone(), out.t2.detach().clone()]
    (tl+calibration['lambda_feature_alignment']*al).backward()
    payload.update(grad_norm_r2p=base.gradient_norm(model.r2p.parameters()),
                   grad_norm_projections=base.gradient_norm(model.projections.parameters()),
                   grad_norm_net_g=base.gradient_norm(task.net_g.parameters()),
                   forward_finite=bool(torch.isfinite(tl) and torch.isfinite(al)),
                   backward_finite=all(bool(torch.isfinite(p.grad).all()) for p in list(model.parameters())+list(task.parameters()) if p.grad is not None))
    prior = [p for p in task.parameters() if id(p) not in {id(q) for q in task.net_g.parameters()}]
    payload['diffusion_prior_frozen'] = all(not p.requires_grad and p.grad is None for p in prior)
    payload['unused_feature_feedback_has_no_grad'] = all(p.grad is None for p in model.feedback_adapter.parameters()) and model.feedback_scale.grad is None
    del out, tl, al
    model.zero_grad(set_to_none=True)
    task.zero_grad(set_to_none=True)
    with torch.no_grad():
        base.set_step_seed(cfg['seed'], 'preflight', 1, batch['sample_id'])
        repeat = base.recurrent_forward(model, task, batch)
        payload['deterministic_forward'] = torch.equal(expected[0], repeat.t1) and torch.equal(expected[1], repeat.t2)
    payload['peak_vram_mb'] = torch.cuda.max_memory_allocated(device)/2**20
    required = ['second_condition_equals_clamped_T1', 'round2_loss_to_T1_image_grad_finite',
                'forward_finite', 'backward_finite', 'diffusion_prior_frozen',
                'unused_feature_feedback_has_no_grad', 'deterministic_forward']
    payload['passed'] = (calls == {'r2p': 2, 'tasknet': 2} and all(payload[k] for k in required)
                         and all(payload[k] > 0 for k in ['round2_loss_to_T1_image_grad_norm',
                                  'grad_norm_r2p', 'grad_norm_projections', 'grad_norm_net_g']))
    base.atomic_json(root/'diagnostics/recurrent_preflight.json', payload)
    assert payload['passed'], payload
    del repeat, expected, batch, model, task, criterion
    gc.collect()
    torch.cuda.empty_cache()
    return payload
