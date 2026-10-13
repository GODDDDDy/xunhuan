"""Fixed K1 checkpoint: compare one pass, RGB repeat, and direct T1-image repeat.

Reuse the exact trained factory, manifest, evaluation RNG and metric functions.
This adapter is necessary because the existing K1 evaluator aliases round2 to
round1 and the existing K2 evaluator only supports internal-feature feedback.
No optimizer, fitting, checkpoint choice or alpha scan is performed.
"""
import csv
import hashlib
import importlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
UID = 'K1_aligned_seed2026'
ARMS = ('K1_T1', 'repeat_RGB_T2', 'direct_T1_image_T2')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def hashes(model, task):
    return {prefix + name: hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for prefix, obj in [('model.', model), ('task.', task)]
            for name, tensor in obj.state_dict().items()}


def main():
    start = time.time()
    for folder in ('metrics', 'diagnostics', 'logs', 'analysis'):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    assert not (ROOT / 'metrics/results.json').exists()
    source_lock = json.loads((PROJECT / 'test_completion_20261007/protocol_lock.json').read_text())
    unit = next(u for u in source_lock['units'] if u['id'] == UID)
    assert sha(unit['checkpoint']) == unit['checkpoint_sha256']
    assert unit['actual_rounds'] == 1
    cfg = yaml.safe_load(Path(unit['config']).read_text())
    cfg['paths']['output_root'] = str(ROOT / 'factory_output')
    sys.path.insert(0, unit['source_scripts'])
    module = importlib.import_module(unit['extension'])
    base = module.configure(cfg)
    base.add_source_path(cfg)
    base.set_seed(2026)
    from rgb_final_confirmation.evaluation import psnr, ssim

    runtime = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                   cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(),
                   matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                   cudnn_tf32=torch.backends.cudnn.allow_tf32)
    assert runtime['matmul_tf32'] == source_lock['runtime']['matmul_tf32']
    assert runtime['cudnn_tf32'] == source_lock['runtime']['cudnn_tf32']
    device = torch.device('cuda:0')
    model, task, criterion, raw = base.build_components(cfg, device)
    cp = torch.load(unit['checkpoint'], map_location='cpu', weights_only=False)
    assert cp['step'] == 3000 and cp['phase_weight'] == 0.1 and cp['actual_rounds'] == 1
    assert cp['protocol_id'] == base.protocol_id(cfg)
    model.load_state_dict(cp['r2p_and_projections'], strict=True)
    task.net_g.load_state_dict(cp['tasknet_net_g'], strict=True)
    del cp, criterion
    model.requires_grad_(False).eval()
    task.requires_grad_(False).eval()
    assert model.feedback_enabled is False
    before = hashes(model, task)
    prior_path = PROJECT / 'test_completion_20261007/units' / UID / 'metrics/test.json'
    prior = json.loads(prior_path.read_text())
    prior_rows = {r['sample_id']: r for r in prior['per_image']}
    loader = base.split_loader(cfg, 'test')
    assert len(loader) == 188
    calls = {'r2p': 0, 'tasknet': 0}
    def counter(name):
        def hook(*unused):
            calls[name] += 1
        return hook
    handles = [model.register_forward_hook(counter('r2p')), task.register_forward_hook(counter('tasknet'))]
    protocol = dict(unit=UID, checkpoint=unit['checkpoint'], checkpoint_sha256=unit['checkpoint_sha256'],
                    source_config=unit['source_config'], source_scripts=unit['source_scripts'],
                    script_sha256=sha(__file__), split='test', frames=188,
                    sample_order=source_lock['test_sample_order'], evaluation_seed=2026, runtime=runtime,
                    branches=list(ARMS), optimizer_created=False, training_steps=0,
                    conditions='P1=R2P(P0,I); T1=Task(P1,I); P2A=R2P(P1,I); P2B=R2P(P1,clamp(T1,0,1)); T2A/B=Task(P2A/B,I)',
                    rng='Original validation-literal index/sample RNG key; restore CPU/all CUDA pre-T1 states for each task call',
                    selection='K1 aligned seed2026 fixed step3000 chosen before new outcomes',
                    scope='Exploratory reuse on an already examined test split. Not trained two-round joint optimization.',
                    historical_K1_metrics=str(prior_path), historical_K1_metrics_sha256=sha(prior_path))
    write_json(ROOT / 'protocol.json', protocol)
    rows = []
    endpoint = None
    with (ROOT / 'metrics/per_image.jsonl').open('x', encoding='utf-8') as stream:
        with torch.no_grad():
            for index, raw_batch in enumerate(loader):
                batch = base.unpack_batch(raw_batch, device)
                sid = batch['sample_id'][0]
                assert sid == protocol['sample_order'][index]
                base.set_step_seed(2026, 'validation', index, batch['sample_id'])
                first = model(batch['polar'], batch['condition'], project_features=False)
                p1 = first.refined_polar
                first.raw_features.clear()
                p1_reference = p1.clone()
                cpu_rng = torch.get_rng_state().clone()
                cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
                def infer(polar):
                    torch.set_rng_state(cpu_rng)
                    torch.cuda.set_rng_state_all(cuda_rng)
                    prediction, features = task(polar, batch['official_rgb'], None)
                    features.clear()
                    return prediction
                t1 = infer(p1)
                t1_reference = t1.clone()
                first_metrics = {'psnr': psnr(t1, batch['target']), 'ssim': ssim(t1, batch['target'])}
                history_delta = {m: first_metrics[m] - prior_rows[sid][m] for m in ('psnr', 'ssim')}
                assert all(abs(d) <= 1e-5 for d in history_delta.values()), (sid, history_delta)
                if index == 0:
                    repeated = infer(p1)
                    maximum = float((repeated - t1).abs().max())
                    endpoint = dict(sample_id=sid, same_P1_task_output_max_abs=maximum,
                                    tolerance=1e-4, passed=maximum <= 1e-4)
                    write_json(ROOT / 'diagnostics/preflight.json', endpoint)
                    assert endpoint['passed']
                    del repeated
                feedback_image = t1.clamp(0, 1)
                clamp_ratio = float(((t1 < 0) | (t1 > 1)).float().mean())
                repeated_rgb = model(p1, batch['condition'], project_features=False)
                repeated_rgb.raw_features.clear()
                t2_rgb = infer(repeated_rgb.refined_polar)
                rgb_metrics = {'psnr': psnr(t2_rgb, batch['target']), 'ssim': ssim(t2_rgb, batch['target'])}
                del repeated_rgb, t2_rgb
                direct = model(p1, feedback_image, project_features=False)
                direct.raw_features.clear()
                t2_direct = infer(direct.refined_polar)
                direct_metrics = {'psnr': psnr(t2_direct, batch['target']), 'ssim': ssim(t2_direct, batch['target'])}
                assert torch.equal(p1, p1_reference) and torch.equal(t1, t1_reference)
                row = dict(sample_id=sid, scene=batch['scene'][0], K1_T1=first_metrics,
                           repeat_RGB_T2=rgb_metrics, direct_T1_image_T2=direct_metrics,
                           T1_condition_clamp_ratio=clamp_ratio, K1_history_delta=history_delta)
                assert all(math.isfinite(row[arm][metric]) for arm in ARMS for metric in ('psnr', 'ssim'))
                rows.append(row)
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                elapsed = time.time() - start
                write_json(ROOT / 'status.json', dict(state='running', completed=len(rows), total=188,
                                                     elapsed_seconds=elapsed, last_sample=sid))
                if index == 0 or len(rows) % 10 == 0:
                    print(f'{len(rows)}/188 elapsed={elapsed:.1f}s first_pass_history_max={max(abs(v) for v in history_delta.values()):.3g}', flush=True)
                del first, p1, p1_reference, t1, t1_reference, feedback_image, direct, t2_direct, batch
    for handle in handles:
        handle.remove()
    after = hashes(model, task)
    assert before == after
    assert all(p.grad is None for p in list(model.parameters()) + list(task.parameters()))
    assert calls == {'r2p': 564, 'tasknet': 565}, calls
    scenes = sorted({r['scene'] for r in rows})
    assert len(rows) == 188 and len(scenes) == 8
    aggregates = {}
    for arm in ARMS:
        aggregates[arm] = {}
        for metric in ('psnr', 'ssim'):
            aggregates[arm]['image_mean_' + metric] = float(np.mean([r[arm][metric] for r in rows]))
            aggregates[arm]['scene_macro_' + metric] = float(np.mean([
                np.mean([r[arm][metric] for r in rows if r['scene'] == scene]) for scene in scenes]))
    comparisons = {}
    for left, right in [('repeat_RGB_T2', 'K1_T1'), ('direct_T1_image_T2', 'K1_T1'),
                        ('direct_T1_image_T2', 'repeat_RGB_T2')]:
        name = left + '_minus_' + right
        comparisons[name] = dict(aggregate_delta={k: aggregates[left][k] - aggregates[right][k]
                                                   for k in aggregates[left]},
                                 improved_images={metric: sum(r[left][metric] > r[right][metric] for r in rows)
                                                  for metric in ('psnr', 'ssim')},
                                 per_scene={scene: {metric: float(np.mean([r[left][metric] - r[right][metric]
                                                                         for r in rows if r['scene'] == scene]))
                                                     for metric in ('psnr', 'ssim')} for scene in scenes})
    results = dict(protocol=protocol, frames=188, scenes=8, aggregates=aggregates, comparisons=comparisons,
                   endpoint=endpoint, actual_calls=calls, frozen_tensor_audit_passed=True,
                   K1_historical_max_abs={m: max(abs(r['K1_history_delta'][m]) for r in rows) for m in ('psnr', 'ssim')},
                   T1_condition_clamp_ratio_image_mean=float(np.mean([r['T1_condition_clamp_ratio'] for r in rows])),
                   elapsed_seconds=time.time() - start, peak_vram_mb=torch.cuda.max_memory_allocated()/2**20,
                   per_image=rows)
    write_json(ROOT / 'metrics/results.json', results)
    write_json(ROOT / 'diagnostics/frozen_audit.json', dict(passed=True, before=before, after=after))
    report = ['# K1直接T1图像反馈：固定权重快速复用', '',
              '复用K1 aligned seed2026 step3000。只做推理，未训练、未更新任何权重。完整188张已有test，8场景，单seed。', '',
              '先计算P1/T1。第二轮两分支分别用P1+原混合RGB、P1+T1图像生成P2；T1条件截断到[0,1]。去反射网络所有调用仍接收原混合RGB。没有内部特征反馈。', '',
              '|方案|图像PSNR|场景PSNR|图像SSIM|场景SSIM|', '|---|---:|---:|---:|---:|']
    labels = dict(K1_T1='原K1一次处理', repeat_RGB_T2='第二轮继续用原RGB', direct_T1_image_T2='第二轮改用T1图像')
    for arm in ARMS:
        a = aggregates[arm]
        report.append(f'|{labels[arm]}|{a["image_mean_psnr"]:.8f}|{a["scene_macro_psnr"]:.8f}|{a["image_mean_ssim"]:.8f}|{a["scene_macro_ssim"]:.8f}|')
    report += ['', '|比较|图像ΔPSNR|场景ΔPSNR|图像ΔSSIM|场景ΔSSIM|PSNR提高图数|SSIM提高图数|',
               '|---|---:|---:|---:|---:|---:|---:|']
    for name, comparison in comparisons.items():
        d = comparison['aggregate_delta']
        n = comparison['improved_images']
        report.append(f'|{name}|{d["image_mean_psnr"]:+.8f}|{d["scene_macro_psnr"]:+.8f}|{d["image_mean_ssim"]:+.8f}|{d["scene_macro_ssim"]:+.8f}|{n["psnr"]}/188|{n["ssim"]}/188|')
    report += ['', '## 结论范围', '',
               '这只是将训练于原RGB条件的现有K1权重复用为两次推理，T1条件与重复输入均未经专门训练。可检查这种直接复用是否改善，不能代替该结构的两轮联合训练，也不能凭负值否定完整方案。', '',
               'test已经在历史工作中被查看，本轮是探索性复用，不是新的独立确认；不据此扫描修改幅度、挑seed或选择检查点。两个聚合口径和两个指标均完整报告。', '',
               f'同输入端点通过：{endpoint["passed"]}。第一轮逐图复现历史K1，最大PSNR差{results["K1_historical_max_abs"]["psnr"]:.3g}、SSIM差{results["K1_historical_max_abs"]["ssim"]:.3g}。所有模型张量前后哈希一致，无梯度或优化器更新。', '',
               f'执行耗时{results["elapsed_seconds"]/60:.2f}分钟；检查点SHA256：{unit["checkpoint_sha256"]}。', '',
               '精确定义、逐图指标、逐场景差值、随机流及冻结审计见protocol.json、metrics/results.json、diagnostics/。']
    (ROOT / 'analysis/DIRECT_T1_REUSE_RESULTS_ZH_20261008.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    write_json(ROOT / 'status.json', dict(state='complete', completed=188, total=188,
                                         elapsed_seconds=time.time()-start, actual_calls=calls))
    print(json.dumps({k: v for k, v in results.items() if k not in ('per_image', 'protocol')}, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write_json(ROOT / 'status.json', dict(state='failed', error=repr(error)))
        raise
