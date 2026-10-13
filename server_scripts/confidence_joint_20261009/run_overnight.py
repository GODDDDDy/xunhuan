"""Sequential GPU study with a predeclared time-only common-step fallback."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import yaml

ROOT = Path(__file__).resolve().parents[1]


def status(stage, **extra):
    value = dict(stage=stage, time_unix=time.time(), **extra)
    tmp = ROOT / 'status.json.tmp'
    tmp.write_text(json.dumps(value, indent=2)); tmp.replace(ROOT / 'status.json')
    print(json.dumps(value), flush=True)


def run(action, unit='A2_dual', step=16000):
    status(action, unit=unit, target_step=step)
    with (ROOT / 'logs' / f'{unit}_{action}.log').open('a') as stream:
        subprocess.run([sys.executable, str(ROOT / 'scripts/run_arm.py'), action,
                        '--unit', unit, '--step', str(step)], stdout=stream, stderr=subprocess.STDOUT,
                       check=True, cwd=ROOT)


def freeze():
    lock = json.loads((ROOT / 'protocol_lock.json').read_text())
    lock['code_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((ROOT / 'scripts').rglob('*.py'))}
    lock['config_sha256'] = {p.stem: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted((ROOT / 'config').glob('A*.yaml'))}
    lock['quality_calibration_sha256'] = hashlib.sha256((ROOT / 'config/quality_calibration.json').read_bytes()).hexdigest()
    (ROOT / 'protocol_lock.json').write_text(json.dumps(lock, indent=2))


def report(step):
    results = {unit: json.loads((ROOT / 'arms' / unit / 'run/metrics/experiment_results.json').read_text())
               for unit in ('A0_direct', 'A2_dual')}
    assert all(x['completed'] for x in results.values())
    lines = ['# 偏振辅助 T1 质量置信度：同预算正式实验', '',
             f'同一完整两轮检查点起步；seed2026；各组评价追加 step{step}，历史累计{13000+step}步。', '',
             '固定目标16000步；如使用较低共同步数，只因事先指定的时间与完整评价保留预算，不依据验证/测试指标选择。', '']
    for split, title in [('validation', '验证872张'), ('test', '测试188张')]:
        lines += ['## ' + title, '', '|方案|输出|图像PSNR|场景PSNR|图像SSIM|场景SSIM|',
                  '|---|---|---:|---:|---:|---:|']
        for unit, result in results.items():
            for rnd in ('round1', 'round2'):
                v = result[split][rnd]
                lines.append(f"|{unit}|{rnd}|{v['image_mean_psnr']:.8f}|{v['scene_macro_psnr']:.8f}|{v['image_mean_ssim']:.8f}|{v['scene_macro_ssim']:.8f}|")
        gain = results['A2_dual'][split]['round2']['image_mean_psnr'] - results['A0_direct'][split]['round2']['image_mean_psnr']
        lines += ['', f'置信版本T2−同预算对照T2：图像PSNR {gain:+.8f} dB。', '']
    lines += ['## 解释范围', '', '两轮均保留增强与第一轮特征对齐；置信头仅由质量监督更新。主方案同时改变反馈条件与第二轮修改幅度，尚不能分离两者贡献。', '',
              '单seed，未完成空间均值或去偏振输入消融；当前测试集此前已被多次查看，属于探索背景。全部逐图指标、质量分数与误差分箱、场景相关性、检查点和清单哈希均保存。负面或异常结果同样保留。']
    (ROOT / 'analysis').mkdir(exist_ok=True)
    (ROOT / 'analysis/CONFIDENCE_OVERNIGHT_RESULTS_ZH_20261009.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (ROOT / 'metrics').mkdir(exist_ok=True)
    combined = dict(completed=True, common_step=step, source_training_steps=13000, results=results)
    (ROOT / 'metrics/combined_results.json').write_text(json.dumps(combined, indent=2))
    return {unit: dict(test_round1=r['test']['round1'], test_round2=r['test']['round2'],
                      test_iteration_gain=r['test']['iteration_gain'], checkpoint_sha256=r['test']['checkpoint_sha256'])
            for unit, r in results.items()}


def main():
    lock = json.loads((ROOT / 'protocol_lock.json').read_text())
    status('starting', deadline_unix=lock['deadline_unix'], target_steps=16000, order=lock['order'])
    if not (ROOT / 'config/quality_calibration.json').exists():
        run('calibrate')
    freeze()
    for unit in lock['order']:
        run('train', unit)
    common = None
    trained = {}
    for unit in lock['order']:
        directory = ROOT / 'arms' / unit / 'run'
        saved = {int(p.stem.split('_')[-1]) for p in (directory / 'checkpoints').glob('step_*.pt')}
        common = saved if common is None else common.intersection(saved)
        trained[unit] = json.loads((directory / 'diagnostics/training_continuity.json').read_text())['updates']
    common.discard(200)
    assert common, 'No common formal checkpoint; preserve incomplete study and failure records.'
    step = max(common)
    selection = dict(common_step=step, target_step=16000, trained_steps=trained,
                     rule='latest common saved step under predeclared clock reserve; no metrics read',
                     selection_time_unix=time.time())
    (ROOT / 'evaluation_selection.json').write_text(json.dumps(selection, indent=2))
    for unit in lock['order']:
        run('evaluate', unit, step)
    summary = report(step)
    status('completed', common_step=step, validation_frames_per_arm=872, test_frames_per_arm=188,
           test_required=True, test_completed=True, results=summary,
           deadline_unix=lock['deadline_unix'], within_deadline=time.time() <= lock['deadline_unix'])


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=repr(exc), traceback=traceback.format_exc(), test_completed=False)
        raise
