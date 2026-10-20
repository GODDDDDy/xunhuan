"""CPU-only descriptive verification and plots; no model fitting or selection."""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project = Path('F:/WORK/projects/pidsr_recurrent_joint_k2')
analysis = project / 'analysis/polar_quality_matched_candidate_20261015'
root = project / '_dl/polar_quality_matched_candidate_20261015/artifacts'
summary = json.loads((root / 'metrics/summary.json').read_text())
receipt = {'cpu_only': True, 'selection': False, 'splits': {}, 'runtime_source_files': 0}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none'})
fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), constrained_layout=True)
for row_index, (split, expected) in enumerate([('validation', 872), ('test', 188)]):
    data = json.loads((root / 'metrics' / f'final_{split}.json').read_text())
    rows = data['rows']
    assert len(rows) == expected and len({r['sample_id'] for r in rows}) == expected
    assert data['original_size'] and not data['tiling']
    for name in ['t1', 'real_quality', 'fixed_quality']:
        for metric in ['psnr', 'ssim']:
            values = [(r['t1'] if name == 't1' else r['arms'][name])[metric] for r in rows]
            assert all(math.isfinite(x) for x in values)
            assert abs(sum(values) / expected - data['image_means'][name][metric]) < 1e-10
    paired = list(csv.DictReader((root / 'metrics' / f'paired_{split}.csv').open()))
    assert len(paired) == expected
    for a, b in zip(rows, paired):
        assert a['sample_id'] == b['sample_id']
        assert abs(float(b['real_minus_T1_PSNR']) - (a['arms']['real_quality']['psnr'] - a['t1']['psnr'])) < 1e-12
        assert abs(float(b['real_minus_fixed_PSNR']) - (a['arms']['real_quality']['psnr'] - a['arms']['fixed_quality']['psnr'])) < 1e-12
        assert all(arm['t1_equal'] for arm in a['arms'].values())
    examples = list((root / 'examples' / split).glob('*.npz'))
    assert len(examples) == 8
    for p in examples:
        record = next(r for r in rows if r['sample_id'].replace('/', '_') == p.stem)
        with np.load(p) as z:
            for name in ['t1', 'real_quality', 'fixed_quality']:
                digest = hashlib.sha256(np.ascontiguousarray(z[name]).tobytes()).hexdigest()
                expected_hash = record['t1_sha256'] if name == 't1' else record['arms'][name]['u_sha256']
                assert digest == expected_hash, (p.name, name)
    scene_rows = summary[split]['scene_differences']
    receipt['splits'][split] = {'frames': expected, 'unique_ids': expected, 'float_examples_verified': len(examples),
        'three_output_hashes_per_example_exact': True, 'means_and_paired_CSV_exact': True,
        'real_vs_T1_PSNR_positive_scenes': sum(r['real_minus_T1']['psnr'] > 0 for r in scene_rows),
        'real_vs_fixed_PSNR_positive_scenes': sum(r['real_minus_fixed']['psnr'] > 0 for r in scene_rows)}
    for col, metric in enumerate(['psnr', 'ssim']):
        ax = axes[row_index, col]
        values = [r['real_minus_fixed'][metric] for r in scene_rows]
        ax.bar([r['scene'] for r in scene_rows], values,
            color=['#238278' if x >= 0 else '#bf5949' for x in values], width=.65)
        ax.axhline(0, color='#303030', lw=.8)
        image_mean = summary[split]['image_mean_differences']['real_minus_fixed'][metric]
        ax.axhline(image_mean, color='#375b9a', ls='--', lw=1, label=f'Image mean {image_mean:+.6f}')
        ax.set(title=f'{split.title()}: real minus fixed', xlabel='Scene',
            ylabel='PSNR difference (dB)' if metric == 'psnr' else 'SSIM difference')
        ax.legend(frameon=False, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Matched 2,000-step endpoints: mixed scene-level effects (one seed)', fontsize=14)
for ext in ['png', 'svg']:
    fig.savefig(analysis / 'figures' / f'endpoint_scene_comparison.{ext}', dpi=180)
plt.close(fig)

# Pre-specified first frame, not chosen to optimize a crop or threshold.
p = next((root / 'examples/validation').glob('validation_*_30_*.npz'))
with np.load(p) as z:
    first, real, fixed, gt = [z[k] for k in ['t1', 'real_quality', 'fixed_quality', 'gt']]
    mse = lambda a: np.mean((a.astype(np.float64) - gt.astype(np.float64)) ** 2, axis=0)
    gains = [mse(first) - mse(real), mse(first) - mse(fixed)]
    limit = max(float(np.quantile(np.abs(np.concatenate([g.ravel() for g in gains])), .98)), 1e-8)
    fig, ax = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for panel, array, title in zip(ax[0], [first, real, fixed], ['T1', 'Real quality U', 'Fixed quality U']):
        panel.imshow(np.clip(array.transpose(1, 2, 0), 0, 1)); panel.set_title(title); panel.axis('off')
    ax[1, 0].imshow(np.clip(gt.transpose(1, 2, 0), 0, 1)); ax[1, 0].set_title('GT (evaluation only)'); ax[1, 0].axis('off')
    for panel, gain, title in zip(ax[1, 1:], gains, ['Real: MSE(T1) - MSE(U)', 'Fixed: MSE(T1) - MSE(U)']):
        im = panel.imshow(gain, cmap='RdBu', vmin=-limit, vmax=limit)
        panel.set_title(title); panel.axis('off')
    fig.colorbar(im, ax=list(ax[1, 1:]), shrink=.8, label='MSE gain: blue improves; red harms')
    fig.suptitle(f'{p.stem}: pre-specified first frame; post-hoc illustration\nRGB clipped for display only; gains use raw floats; color range saturates at 98th percentile', fontsize=11)
    fig.savefig(analysis / 'figures/endpoint_scene30_example.png', dpi=160)
    plt.close(fig)

plan = json.loads((root / 'config/plan.json').read_text())
for name, expected_hash in {
    'weights/common_initial21.pt': 'af2367fd9f96871cad3b94230638cf5c75acd3b8f8c186cdf210a37b730e82d8',
    'weights/migration21_final.pt': 'cecdcfa47d290b0a2d6b662b96bb4407a7043ef5789e9e87d70768f2c683916d',
    'pretraining/checkpoints/selected.pt': 'a4ecebacba5cfc25546320974a9c37f1cc220c685f1318b3cc290ae4a68e9554',
    'arms/real_quality/final.pt': 'aa7a723094ccb693e7a7b0d5f27ed9003d93bfcfd38cd44ac3d80dbfa5428efc',
    'arms/fixed_quality/final.pt': 'f5706ad0fd84f80884c2bc4acf9b8c86c046b1f594745e0ab9d21ce32218406c',
}.items():
    assert hashlib.file_digest((root / name).open('rb'), 'sha256').hexdigest() == expected_hash
receipt['three_source_and_two_final_weight_hashes_exact'] = True
locked_receipt = json.loads((root / 'config/lock_receipt.json').read_text())
assert hashlib.file_digest((root / 'config/locked_run.json').open('rb'), 'sha256').hexdigest() == locked_receipt['sha256']
assert locked_receipt['protocol_id'] == plan['protocol_id']
receipt['locked_config_receipt_exact'] = True
for item in plan['code_manifest']:
    assert hashlib.file_digest((root / item['path']).open('rb'), 'sha256').hexdigest() == item['sha256']
receipt['locked_code_manifest_exact'] = len(plan['code_manifest'])
runtime = json.loads((root / 'diagnostics/runtime_source_manifest.json').read_text())
for item in runtime['files']:
    assert hashlib.file_digest((root / item['archived']).open('rb'), 'sha256').hexdigest() == item['sha256']
receipt['runtime_source_files'] = len(runtime['files'])
(analysis / 'receipts/endpoint_data_verification.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
