"""Reuse the locked evaluator for the fixed-step test, including a running study."""
import argparse
import csv
import gc
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def test_status(root, stage, **extra):
    value = {'stage': stage, 'time_unix': time.time(), **extra}
    atomic_json(root / 'test_status.json', value)
    print(json.dumps(value), flush=True)


def audit_locked_inputs(base, cfg, root):
    lock = json.loads((root / 'protocol_lock.json').read_text())
    for name, digest in lock['code_sha256'].items():
        assert base.sha256_file(root / name) == digest, name
    for path, key in [(root / 'config/study.yaml', 'config_sha256'),
                      (Path(cfg['paths']['manifest']), 'manifest_sha256'),
                      (Path(cfg['paths']['schedule']), 'schedule_sha256')]:
        assert base.sha256_file(path) == lock[key], key
    with Path(cfg['paths']['manifest']).open(newline='') as stream:
        ids = [row['sample_id'] for row in csv.DictReader(stream) if row['split'] == 'test']
    assert len(ids) == len(set(ids)) == 188
    return ids


def write_combined_report(root, cfg, validation, test):
    source_test = root.parent / 'test_completion_20261007/units/K1_aligned_seed2026/metrics/test.json'
    old_test = json.loads(source_test.read_text())
    assert old_test['checkpoint_sha256'] == cfg['study']['source_K1_sha256']
    assert {r['sample_id'] for r in old_test['per_image']} == {r['sample_id'] for r in test['per_image']}
    old_val = json.loads(Path(cfg['study']['source_K1_validation']).read_text())
    lines = ['# 直接T1反馈两轮联合训练：验证与测试结果', '',
             'seed2026；完整K1 step3000权重起步，追加10000步，fresh Adam；偏振增强与第一轮特征对齐保留，不含置信度。', '',
             '训练检查点固定为追加step10000。测试要求由用户在本次训练过程中补充，不改变训练配置或权重选择。', '']
    for title, old, metrics in [('验证集（872张）', old_val, validation), ('测试集（188张）', old_test, test)]:
        lines += ['## ' + title, '', '|输出|图像PSNR|场景PSNR|图像SSIM|场景SSIM|',
                  '|---|---:|---:|---:|---:|']
        for label, value in [('源K1', old), ('新模型T1', metrics['round1']), ('新模型T2', metrics['round2'])]:
            lines.append(f"|{label}|{value['image_mean_psnr']:.8f}|{value['scene_macro_psnr']:.8f}|{value['image_mean_ssim']:.8f}|{value['scene_macro_ssim']:.8f}|")
        gain = metrics['iteration_gain']
        lines += ['', f"同一模型T2−T1：图像PSNR {gain['image_mean_psnr']:+.8f} dB，SSIM {gain['image_mean_ssim']:+.8f}。", '']
    lines += ['## 比较范围与可核查记录', '',
              '新模型与源K1同时相差追加训练预算和反馈结构；未完成同额外10000步K1对照，不能把全部变化归因于反馈。', '',
              '本项目此前已反复查看这一测试划分，本次仍如实报告完整测试结果，不称作全新独立确认。测试未用于本轮梯度更新、调参或检查点选择。', '',
              '逐图两轮PSNR/SSIM及场景汇总位于metrics/test_results.json；验证与测试完整合并记录位于metrics/experiment_results.json。两者使用同一个step10000权重，记录其哈希、源K1哈希、清单哈希和实际调用计数。']
    path = root / 'analysis/DIRECT_T1_JOINT_RESULTS_ZH_20261008.md'
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        backup = path.with_name('DIRECT_T1_JOINT_VALIDATION_ONLY_ZH_20261008.md')
        if not backup.exists():
            backup.write_text(path.read_text(), encoding='utf-8')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def complete_test(base, cfg, root, publish_main_status=True):
    import torch
    started = time.time()
    ids = audit_locked_inputs(base, cfg, root)
    validation = json.loads((root / 'metrics/results.json').read_text())
    assert validation['frames'] == len(validation['per_image']) == 872
    assert validation['checkpoint_step'] == 10000
    test_status(root, 'full_test', target_frames=188, fixed_additional_step=10000)
    if publish_main_status:
        previous = json.loads((root / 'status.json').read_text())
        atomic_json(root / 'training_validation_status.json', previous)
        atomic_json(root / 'status.json', {'stage': 'full_test', 'time_unix': time.time(),
                                         'additional_step': 10000, 'target_frames': 188})
    original_validate, original_build = base.validate, base.build_components
    calls = {'r2p': 0, 'tasknet': 0}
    def build(config, device):
        model, task, criterion, raw = original_build(config, device)
        def counter(name):
            def hook(*args):
                calls[name] += 1
            return hook
        model.register_forward_hook(counter('r2p'))
        task.register_forward_hook(counter('tasknet'))
        return model, task, criterion, raw
    def observed_validate(model, task, loader, device, seed):
        assert len(loader.dataset) == 188
        def batches():
            for i, batch in enumerate(loader, 1):
                yield batch
                if i % 10 == 0 or i == 188:
                    atomic_json(root / 'test_progress.json', {'completed_frames': i, 'total_frames': 188,
                        'elapsed_seconds': time.time() - started})
        return original_validate(model, task, batches(), device, seed)
    base.validate, base.build_components = observed_validate, build
    try:
        test = base.run_evaluation(cfg, torch.device('cuda:0'), 'step_10000', 'test')
    finally:
        base.validate, base.build_components = original_validate, original_build
    assert test['frames'] == len(test['per_image']) == 188
    assert len({row['sample_id'] for row in test['per_image']}) == 188
    assert {row['sample_id'] for row in test['per_image']} == set(ids)
    assert calls == {'r2p': 376, 'tasknet': 376}
    assert test['checkpoint_sha256'] == validation['checkpoint_sha256']
    test.update(split='test', confidence_enabled=False, image_feedback_enabled=True,
                source_K1_sha256=cfg['study']['source_K1_sha256'], actual_forward_calls=calls,
                evaluator_sha256=base.sha256_file(Path(__file__)),
                manifest_sha256=base.sha256_file(Path(cfg['paths']['manifest'])),
                evaluation_seconds=time.time() - started,
                scope='fixed-checkpoint inference on previously examined test split')
    atomic_json(root / 'metrics/test_results.json', test)
    combined = {'completed': True, 'test_required': True, 'fixed_additional_step': 10000,
                'checkpoint_sha256': test['checkpoint_sha256'], 'validation': validation, 'test': test}
    atomic_json(root / 'metrics/experiment_results.json', combined)
    write_combined_report(root, cfg, validation, test)
    summary = {'additional_step': 10000, 'validation_frames': 872, 'test_frames': 188,
               'test_required': True,
               'test_completed': True, 'test_round1': test['round1'], 'test_round2': test['round2'],
               'test_iteration_gain': test['iteration_gain'], 'checkpoint_sha256': test['checkpoint_sha256']}
    if publish_main_status:
        atomic_json(root / 'status.json', {'stage': 'completed', 'time_unix': time.time(), **summary})
    test_status(root, 'completed', **summary)
    gc.collect()
    torch.cuda.empty_cache()
    return test


def owner_alive(pid):
    path = Path(f'/proc/{pid}/stat')
    if not path.exists():
        return False
    try:
        return path.read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--wait-for-study', action='store_true')
    args = parser.parse_args()
    import torch
    import yaml
    import run_direct_t1
    cfg = yaml.safe_load((ROOT / 'config/study.yaml').read_text())
    base = run_direct_t1.configure(cfg)
    base.add_source_path(cfg)
    ids = audit_locked_inputs(base, cfg, ROOT)
    if args.check_only:
        ck = torch.load(ROOT / 'run/checkpoints/last.pt', map_location='cpu', weights_only=False)
        assert ck['protocol_id'] == base.protocol_id(cfg)
        print(json.dumps({'passed': True, 'test_frames': len(ids), 'locked_inputs_unchanged': True,
                          'existing_checkpoint_protocol_matches': True, 'checkpoint_step': ck['step']}))
        return
    if args.wait_for_study:
        pid = int((ROOT / 'logs/run.pid').read_text())
        test_status(ROOT, 'waiting_for_training_and_validation', upstream_pid=pid, target_frames=188)
        while True:
            state = json.loads((ROOT / 'status.json').read_text())
            if state['stage'] == 'failed':
                raise RuntimeError('Upstream study failed: ' + state.get('error', ''))
            if not owner_alive(pid):
                assert state['stage'] == 'completed', state
                break
            time.sleep(15)
    complete_test(base, cfg, ROOT)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        test_status(ROOT, 'failed', error=repr(exc), traceback=traceback.format_exc())
        main_status = ROOT / 'status.json'
        if main_status.exists() and json.loads(main_status.read_text())['stage'] == 'full_test':
            atomic_json(main_status, {'stage': 'failed', 'failed_stage': 'full_test',
                'test_completed': False, 'time_unix': time.time(), 'error': repr(exc)})
        raise
