"""One isolated arm: calibration, integrity, continuous training, or full evaluation."""
import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import torch
from torch.nn import functional as F
import yaml
import confidence

ROOT = Path(__file__).resolve().parents[1]


def preflight(base, cfg, device):
    root = Path(cfg['paths']['output_root'])
    calibration = base.calibrate(cfg, device)
    base.set_seed(cfg['seed'])
    model, task, criterion, raw = base.build_components(cfg, device)
    batch = base.unpack_batch(next(iter(base.scheduled_loader(cfg, raw, 1))), device)
    assert torch.equal(batch['condition'], batch['official_rgb'])
    torch.cuda.reset_peak_memory_stats(device)
    calls = {'r2p': 0, 'tasknet': 0}
    def count(name):
        def hook(*args):
            calls[name] += 1
        return hook
    hooks = [model.register_forward_hook(count('r2p')), task.register_forward_hook(count('tasknet'))]
    base.set_step_seed(cfg['seed'], 'preflight', 1, batch['sample_id'])
    out, _, tl, al = base.forward_losses(model, task, criterion, batch)
    for hook in hooks: hook.remove()
    main = tl + calibration['lambda_feature_alignment'] * al
    head = tuple(model.confidence_head.parameters())
    restoration = tuple(model.r2p.parameters()) + tuple(model.projections.parameters()) + tuple(task.net_g.parameters())
    head_from_main = torch.autograd.grad(main, head, retain_graph=True, allow_unused=True)
    restoration_from_conf = torch.autograd.grad(out.confidence_loss, restoration, retain_graph=True, allow_unused=True)
    feedback = torch.autograd.grad(out.loss2, out.t1, retain_graph=True)[0]
    payload = dict(protocol_id=base.protocol_id(cfg), actual_forward_calls=calls,
                   confidence_mode=cfg['confidence']['mode'], new_parameters=sum(p.numel() for p in head),
                   main_does_not_update_H=all(g is None for g in head_from_main),
                   confidence_does_not_update_restoration=all(g is None for g in restoration_from_conf),
                   round2_to_T1_gradient=float(feedback.norm()), raw_rgb_first_condition=True)
    (main + out.confidence_loss).backward()
    payload.update(r2p_grad_norm=base.gradient_norm(model.r2p.parameters()),
                   alignment_grad_norm=base.gradient_norm(model.projections.parameters()),
                   net_g_grad_norm=base.gradient_norm(task.net_g.parameters()),
                   H_grad_norm=base.gradient_norm(head),
                   gradients_finite=all(torch.isfinite(p.grad).all() for p in
                                        list(model.parameters()) + list(task.parameters()) if p.grad is not None))
    prior = [p for p in task.parameters() if id(p) not in {id(q) for q in task.net_g.parameters()}]
    payload['prior_frozen'] = all(not p.requires_grad and p.grad is None for p in prior)
    del out, tl, al, main, feedback, head_from_main, restoration_from_conf
    model.zero_grad(set_to_none=True); task.zero_grad(set_to_none=True)
    model._confidence_step = 0
    with torch.no_grad():
        base.set_step_seed(cfg['seed'], 'preflight', 1, batch['sample_id'])
        zero = base.recurrent_forward(model, task, batch)
        expected = (zero.t1.clone(), zero.t2.clone(), zero.confidence_map.clone())
        del zero
        mode = model._confidence_mode
        model._confidence_mode = 'direct'
        base.set_step_seed(cfg['seed'], 'preflight', 1, batch['sample_id'])
        direct = base.recurrent_forward(model, task, batch)
        payload['gamma0_equals_direct'] = torch.equal(direct.t1, expected[0]) and torch.equal(direct.t2, expected[1])
        del direct
        model._confidence_mode = mode
        gt_free = {k: v for k, v in batch.items() if k != 'target'}
        base.set_step_seed(cfg['seed'], 'preflight', 1, batch['sample_id'])
        repeat = base.recurrent_forward(model, task, gt_free)
        payload['deterministic_GT_free_inference'] = (torch.equal(repeat.t1, expected[0]) and
                                                       torch.equal(repeat.t2, expected[1]) and
                                                       torch.equal(repeat.confidence_map, expected[2]))
        del repeat, expected
        if mode == 'dual':
            model._confidence_step = 500
            model._confidence_override = 1.
            base.set_step_seed(cfg['seed'], 'preflight', 1, batch['sample_id'])
            endpoint = base.recurrent_forward(model, task, gt_free)
            payload['all_C1_endpoint_identity'] = torch.equal(endpoint.refined_polar, endpoint.first.refined_polar) and torch.equal(endpoint.t1, endpoint.t2)
            del endpoint
        else:
            payload['all_C1_endpoint_identity'] = True
    payload['peak_vram_mb'] = torch.cuda.max_memory_allocated(device) / 2 ** 20
    flags = ['main_does_not_update_H', 'confidence_does_not_update_restoration', 'gradients_finite',
             'prior_frozen', 'gamma0_equals_direct', 'deterministic_GT_free_inference', 'all_C1_endpoint_identity']
    norms = ['round2_to_T1_gradient', 'r2p_grad_norm', 'alignment_grad_norm', 'net_g_grad_norm', 'H_grad_norm']
    payload['passed'] = calls == {'r2p': 2, 'tasknet': 2} and all(payload[k] for k in flags) and all(payload[k] > 0 for k in norms)
    base.atomic_json(root / 'diagnostics/recurrent_preflight.json', payload)
    assert payload['passed'], payload
    del model, task, criterion, batch, head, restoration
    gc.collect(); torch.cuda.empty_cache()
    print(json.dumps(payload), flush=True)


def calibrate(base, cfg, device):
    base.set_seed(cfg['seed'])
    model, task, criterion, raw = base.build_components(cfg, device)
    model.eval(); task.eval()
    values, ids = [], []
    with torch.no_grad():
        for i, raw_batch in enumerate(base.scheduled_loader(cfg, raw, 128), 1):
            batch = base.unpack_batch(raw_batch, device)
            base.set_step_seed(cfg['seed'], 'confidence_calibration', i, batch['sample_id'])
            first = model(batch['polar'], batch['condition'], project_features=False)
            first.raw_features.clear()
            prediction, features = task(first.refined_polar, batch['official_rgb'], None)
            error = F.avg_pool2d((prediction.clamp(0, 1) - batch['target']).abs().mean(1, keepdim=True), 4)
            values.append(error.cpu().flatten()); ids.extend(batch['sample_id'])
            del first, prediction, features, error, batch
    m = float(torch.cat(values).median())
    payload = dict(median_local_error=m, tau=max(m / 0.6931471805599453, .001), initial_error=max(m, .002),
                   training_crops=256, sample_ids=ids, source_sha256=cfg['study']['source_sha256'],
                   subset='first 256 crops of shared frozen training schedule, no validation/test')
    base.atomic_json(ROOT / 'config/quality_calibration.json', payload)
    for path in sorted((ROOT / 'config').glob('A*.yaml')):
        config = yaml.safe_load(path.read_text())
        config['confidence'].update(tau=payload['tau'], initial_error=payload['initial_error'])
        path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps({k:v for k,v in payload.items() if k != 'sample_ids'}), flush=True)


def evaluate(base, cfg, device, step):
    root = Path(cfg['paths']['output_root'])
    lock = json.loads((ROOT / 'protocol_lock.json').read_text())
    assert base.sha256_file(Path(cfg['paths']['manifest'])) == lock['manifest_sha256']
    assert base.sha256_file(Path(cfg['paths']['schedule'])) == lock['schedule_sha256']
    for name, digest in lock['code_sha256'].items():
        assert base.sha256_file(ROOT / name) == digest, name
    assert base.sha256_file(ROOT / 'config' / f"{cfg['study']['unit']}.yaml") == lock['config_sha256'][cfg['study']['unit']]
    result = dict(completed=False, test_required=True, fixed_additional_step=step,
                  source_sha256=cfg['study']['source_sha256'], unit=cfg['study']['unit'],
                  historical_steps=13000 + step, confidence_config=cfg['confidence'])
    for split, expected in [('validation', 872), ('test', 188)]:
        calls = {'r2p': 0, 'tasknet': 0}
        original_build = base.build_components
        def build(config, location):
            model, task, criterion, raw = original_build(config, location)
            def counter(name):
                def hook(*args): calls[name] += 1
                return hook
            model.register_forward_hook(counter('r2p')); task.register_forward_hook(counter('tasknet'))
            return model, task, criterion, raw
        base.build_components = build
        started = time.time()
        try:
            metrics = base.run_evaluation(cfg, device, f'step_{step}', split)
        finally:
            base.build_components = original_build
        ids = {r['sample_id'] for r in metrics['per_image']}
        from rgb_final_confirmation.data import load_manifest, select_rows
        expected_ids = {r.sample_id for r in select_rows(load_manifest(cfg['paths']['manifest']), split)}
        assert metrics['frames'] == len(metrics['per_image']) == len(ids) == expected
        assert ids == expected_ids and calls == {'r2p': expected * 2, 'tasknet': expected * 2}
        metrics.update(actual_forward_calls=calls, evaluation_seconds=time.time() - started,
                       manifest_sha256=lock['manifest_sha256'], test_used_for_selection=False)
        result[split] = metrics
        base.atomic_json(root / 'metrics' / f'{split}_results.json', metrics)
        base.atomic_json(root / 'metrics/experiment_results.json', result)
        del metrics
        gc.collect(); torch.cuda.empty_cache()
    assert result['validation']['checkpoint_sha256'] == result['test']['checkpoint_sha256']
    result['completed'] = True
    base.atomic_json(root / 'metrics/experiment_results.json', result)
    print(json.dumps(dict(unit=cfg['study']['unit'], completed=True, step=step,
                          test_round1=result['test']['round1'], test_round2=result['test']['round2'])), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['calibrate', 'train', 'evaluate'])
    parser.add_argument('--unit', default='A2_dual')
    parser.add_argument('--step', type=int, default=16000)
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / 'config' / f'{args.unit}.yaml').read_text())
    base = confidence.configure(cfg)
    base.add_source_path(cfg)
    device = torch.device('cuda:0')
    if args.action == 'calibrate':
        calibrate(base, cfg, device)
    elif args.action == 'evaluate':
        evaluate(base, cfg, device, args.step)
    else:
        root = Path(cfg['paths']['output_root'])
        for directory in ('checkpoints', 'config', 'diagnostics', 'logs', 'metrics'):
            (root / directory).mkdir(parents=True, exist_ok=True)
        if not (root / 'checkpoints/last.pt').exists():
            preflight(base, cfg, device)
            base.run_training(cfg, device, 200, False)
            gc.collect(); torch.cuda.empty_cache()
        base.run_training(cfg, device, 16000, True)
        rows = list(csv.DictReader((root / 'logs/train.csv').open()))
        steps = [int(row['step']) for row in rows]
        assert steps == list(range(1, steps[-1] + 1))
        ck = torch.load(root / 'checkpoints/last.pt', map_location='cpu', weights_only=False)
        adam_steps = [float(s['step']) for s in ck['optimizer']['state'].values()]
        assert ck['step'] == steps[-1] and min(adam_steps) == max(adam_steps) == steps[-1]
        base.atomic_json(root / 'diagnostics/training_continuity.json', dict(passed=True,
                         updates=steps[-1], all_adam_steps=steps[-1], optimizer_groups=len(ck['optimizer']['param_groups'])))
        print(json.dumps(dict(unit=args.unit, trained_steps=steps[-1], continuous=True)), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print(traceback.format_exc(), flush=True)
        raise
