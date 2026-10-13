"""Quality-supervised confidence extension; reuse the archived matched trainer."""
import copy
import gc
import math
from pathlib import Path
import types
import torch
from torch import nn
from torch.nn import functional as F


class QualityHead(nn.Module):
    def __init__(self, initial_error):
        super().__init__()
        self.projections = nn.ModuleDict({f'T{i}': nn.Conv2d(c, 8, 1)
                                         for i, c in enumerate((96, 96, 192), 1)})
        self.body = nn.Sequential(nn.Conv2d(38, 32, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.constant_(self.body[-1].bias,
                          math.log(math.expm1(max(float(initial_error), .002) - .001)))
        assert sum(p.numel() for p in self.parameters()) == 23353

    def forward(self, image, prediction, polar_features, task_features):
        image, prediction = image.detach(), prediction.detach().clamp(0, 1)
        inputs = [F.avg_pool2d(x, 4) for x in
                  (image, prediction, (image - prediction).abs(), polar_features.detach())]
        size = inputs[0].shape[-2:]
        inputs += [head(F.interpolate(task_features[name].detach(), size=size,
                                     mode='bilinear', align_corners=False))
                   for name, head in self.projections.items()]
        return .001 + F.softplus(self.body(torch.cat(inputs, dim=1)))


def polarization_features(polar):
    from rgb_final_confirmation.polarfree import opencv_rgb_to_gray_ste
    with torch.no_grad():
        gray = [opencv_rgb_to_gray_ste(x) for x in polar.detach().chunk(4, dim=1)]
        s0 = sum(gray) / 2
        q, u = (gray[0] - gray[2]) / s0.clamp_min(.001), (gray[1] - gray[3]) / s0.clamp_min(.001)
        dolp = torch.sqrt(q.square() + u.square()).clamp(0, 1)
        usable = sum(((x > 2 / 255) & (x < 253 / 255)).float() for x in gray) / 4
        return torch.cat((s0 / 2, dolp, q.clamp(-1, 1), u.clamp(-1, 1), usable), dim=1)


def from_feature(self, polar, rgb, project_features=False):
    r = self.r2p
    pol = r.polar_encoder(polar)
    gate = torch.sigmoid(r.gate(torch.cat((pol, rgb), dim=1)))
    fused = gate * rgb + (1 - gate) * pol
    gamma, beta = r.film(rgb).chunk(2, dim=1)
    modulated = pol * (1 + .10 * torch.tanh(gamma)) + .10 * torch.tanh(beta)
    delta = r.max_residual * torch.tanh(r.body(torch.cat((modulated, fused), dim=1)))
    refined = (polar + delta).clamp(0, 1)
    features = dict(rgb=rgb, polar=pol, gate=gate, fused=fused,
                    modulated_polar=modulated, delta=delta)
    import run_phase_objective
    return run_phase_objective.base.JointOutput(refined, {}, features, delta, gate)


def configure(cfg):
    import run_matched
    import run_phase_objective
    base = run_matched.configure(cfg)
    parent_build, parent_protocol = base.build_components, base.protocol_id
    parent_checkpoint, parent_validate = base.checkpoint_payload, base.validate

    def protocol(config):
        obj = copy.deepcopy(config)
        obj['confidence_extension_sha256'] = base.sha256_file(Path(__file__))
        return parent_protocol(obj)
    base.protocol_id = protocol

    def build(config, device):
        model, task, criterion, raw = parent_build(config, device)
        source = Path(config['study']['source_checkpoint'])
        assert base.sha256_file(source) == config['study']['source_sha256']
        ck = torch.load(source, map_location='cpu', weights_only=False)
        assert ck['step'] == 10000 and ck['actual_rounds'] == 2 and not ck['confidence_enabled']
        model.load_state_dict(ck['r2p_and_projections'], strict=True)
        task.net_g.load_state_dict(ck['tasknet_net_g'], strict=True)
        assert all(torch.equal(t.detach().cpu(), ck['r2p_and_projections'][n])
                   for n, t in model.state_dict().items())
        assert all(torch.equal(t.detach().cpu(), ck['tasknet_net_g'][n])
                   for n, t in task.net_g.state_dict().items())
        model.feedback_adapter.requires_grad_(False)
        model.feedback_scale.requires_grad_(False)
        model._study_frozen_start = run_phase_objective.frozen_state(model, task)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config['seed'] + 120000)
            model.confidence_head = QualityHead(config['confidence']['initial_error'])
        model.to(device)
        model._confidence_forward = types.MethodType(from_feature, model)
        model._confidence_step = int(config['optimization']['total_steps'])
        model._confidence_mode = config['confidence']['mode']
        model._confidence_tau = float(config['confidence']['tau'])
        model._confidence_warmup = int(config['confidence']['warmup_steps'])
        base.atomic_json(Path(config['paths']['output_root']) / 'diagnostics/source_load.json', {
            'source_sha256': config['study']['source_sha256'], 'full_R2P_alignment_net_g_loaded': True,
            'existing_head_reset': False, 'new_parameters': 23353,
            'optimizer_initialization': 'fresh_Adam', 'data_condition': config['data']['condition']})
        del ck
        return model, task, criterion, raw

    def forward(model, task, batch, need_alignment=False):
        first = model(batch['polar'], batch['condition'], project_features=need_alignment)
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        def run_task(polar):
            if torch.is_grad_enabled() and model.activation_checkpoint:
                return base.checkpoint(task, polar, batch['official_rgb'], None,
                                       use_reentrant=False, preserve_rng_state=True)
            return task(polar, batch['official_rgb'], None)
        if not torch.is_grad_enabled():
            first.raw_features.clear()
        t1, features = run_task(first.refined_polar)
        align = base.feature_alignment_loss(first.projected_features, features) if need_alignment else None
        b = model.confidence_head(batch['official_rgb'], t1,
                                  polarization_features(batch['polar']), features)
        features.clear()
        c = F.interpolate(torch.exp(-b.detach() / model._confidence_tau),
                          size=t1.shape[-2:], mode='bilinear', align_corners=False)
        if hasattr(model, '_confidence_override'):
            c = torch.full_like(c, float(model._confidence_override))
        gamma = min(1., model._confidence_step / max(model._confidence_warmup, 1))
        mode = model._confidence_mode
        if mode == 'direct':
            second = model(first.refined_polar, t1.clamp(0, 1), project_features=False)
            w = torch.ones_like(c)
        else:
            cf = 1 - gamma * (1 - c)
            rgb = model.r2p.rgb_encoder(t1.clamp(0, 1))
            if torch.is_grad_enabled():
                rgb = cf * rgb + (1 - cf) * model.r2p.rgb_encoder(batch['official_rgb'])
            else:
                rgb.mul_(cf)
                rgb.add_(model.r2p.rgb_encoder(batch['official_rgb']).mul_(1 - cf))
            second = model(first.refined_polar, None, project_features=False, rgb_feature=rgb)
            w = 1 - gamma * c if mode == 'dual' else torch.ones_like(c)
            second.refined_polar = first.refined_polar + w * (second.refined_polar - first.refined_polar)
            del rgb
        if not torch.is_grad_enabled():
            second.raw_features.clear()
        devices = list(range(torch.cuda.device_count())) if cuda_rng is not None else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            t2, f2 = run_task(second.refined_polar)
        f2.clear()
        second.first, second.t1, second.t2 = first, t1, t2
        second.alignment_first = align
        second.error_scale, second.confidence_map = b, c
        second.update_weight, second.confidence_gamma = w, gamma
        return second

    def losses(model, task, criterion, batch):
        out = forward(model, task, batch, need_alignment=True)
        out.loss1, out.loss2 = criterion(out.t1, batch['target']), criterion(out.t2, batch['target'])
        task_loss = out.loss1 / 3 + out.loss2 * 2 / 3
        e = F.avg_pool2d((out.t1.detach().clamp(0, 1) - batch['target']).abs().mean(1, keepdim=True), 4).detach()
        out.confidence_loss = (e / out.error_scale + out.error_scale.log()).mean()
        assert all(torch.isfinite(x) for x in (task_loss, out.alignment_first, out.confidence_loss))
        return out, out.t2, task_loss, out.alignment_first

    def checkpoint(model, task, optimizer, step, coefficient, metrics=None, *extra):
        result = parent_checkpoint(model, task, optimizer, step, coefficient, metrics, *extra)
        result.update(method='T1 local quality confidence feedback', confidence_enabled=True,
                      confidence_mode=cfg['confidence']['mode'], confidence_config=cfg['confidence'],
                      source_sha256=cfg['study']['source_sha256'], source_training_steps=13000,
                      additional_training_steps=int(step), total_training_steps_from_official=13000 + int(step),
                      actual_rounds=2, image_feedback_enabled=True, internal_feature_feedback_enabled=False)
        return result

    def offload(model, task, optimizer, loader, device, seed):
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        task.zero_grad(set_to_none=True)
        locations = []
        for state in optimizer.state.values():
            for key in list(state):
                value = state[key]
                if torch.is_tensor(value) and value.device.type == 'cuda':
                    locations.append((state, key, value.device))
                    state[key] = value.detach().cpu()
        value = None
        gc.collect(); torch.cuda.empty_cache()
        try:
            return base.validate(model, task, loader, device, seed)
        finally:
            for state, key, location in locations:
                state[key] = state[key].to(location)

    def validate(model, task, loader, device, seed):
        from quality_metrics import QualityCollector
        total = len(loader.dataset)
        collector = QualityCollector(Path(cfg['paths']['output_root']), total, model._confidence_tau)
        original = base.recurrent_forward
        def observed(m, t, batch, need_alignment=False):
            out = original(m, t, batch, need_alignment)
            collector.observe(out, batch)
            return out
        base.recurrent_forward = observed
        try:
            metrics = parent_validate(model, task, loader, device, seed)
            metrics['quality_confidence'] = collector.finish()
            return metrics
        finally:
            base.recurrent_forward = original

    base.build_components, base.recurrent_forward, base.forward_losses = build, forward, losses
    base.checkpoint_payload, base.validate_with_optimizer_offload = checkpoint, offload
    base.validate = validate
    return base
