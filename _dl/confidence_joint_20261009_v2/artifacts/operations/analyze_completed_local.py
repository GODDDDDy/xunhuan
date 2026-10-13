"""CPU analysis of archived full results and predetermined scene examples."""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta

parser = argparse.ArgumentParser()
parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[3])
PROJECT = parser.parse_args().project_root
ARCHIVE = PROJECT / '_dl/confidence_joint_20261009_v2/artifacts'
os.environ['MPLCONFIGDIR'] = str(PROJECT / 'tmp/matplotlib')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
result = read(ARCHIVE / 'metrics/combined_results.json')['results']
source = read(PROJECT / '_dl/direct_t1_joint_20261008_memorysafe_v2/artifacts/metrics/experiment_results.json')
status = read(ARCHIVE / 'status.json')
analysis = dict(completed=True, common_step=16000, analysis_uses_archived_outputs_only=True,
                no_new_GPU_work=True, single_seed=2026, splits={}, example_audit={})
for split in ('validation', 'test'):
    control, dual = result['A0_direct'][split], result['A2_dual'][split]
    rows0 = {r['sample_id']: r for r in control['per_image']}
    rows2 = {r['sample_id']: r for r in dual['per_image']}
    assert set(rows0) == set(rows2)
    ids = sorted(rows0)
    for values in (control, dual):
        for rnd in ('round1', 'round2'):
            for metric in ('psnr', 'ssim'):
                actual = float(np.mean([r[rnd][metric] for r in values['per_image']]))
                assert abs(actual - values[rnd]['image_mean_' + metric]) < 1e-10
    delta = np.array([rows2[i]['round2']['psnr'] - rows0[i]['round2']['psnr'] for i in ids])
    scenes = {}
    for scene in sorted({r['scene'] for r in rows0.values()}):
        scene_ids = [i for i in ids if rows0[i]['scene'] == scene]
        scenes[scene] = dict(frames=len(scene_ids), mean_paired_PSNR_delta=float(np.mean(
            [rows2[i]['round2']['psnr'] - rows0[i]['round2']['psnr'] for i in scene_ids])),
            mean_paired_SSIM_delta=float(np.mean(
                [rows2[i]['round2']['ssim'] - rows0[i]['round2']['ssim'] for i in scene_ids])))
    quality = {}
    for unit in ('A0_direct', 'A2_dual'):
        v = result[unit][split]
        q = v['quality_confidence']
        n = sum(b['pixels'] for b in q['bins'])
        weighted = lambda key: sum(b['pixels'] * b[key] for b in q['bins'] if b['pixels']) / n
        b_mean, e_mean = weighted('mean_predicted_error'), weighted('mean_actual_error1')
        quality[unit] = dict(quarter_grid_cells=n, confidence_weighted_mean=weighted('mean_confidence'),
                             predicted_error_weighted_mean=b_mean, actual_error1_weighted_mean=e_mean,
                             actual_error2_weighted_mean=weighted('mean_actual_error2'),
                             predicted_error_over_actual_error=b_mean / e_mean,
                             polar_update_weighted_mean=weighted('mean_polar_update'),
                             spearman_image_mean=q['spearman_image_mean'],
                             spearman_scene_macro=float(np.mean(list(q['spearman_by_scene'].values()))),
                             images_with_negative_spearman=sum(r['spearman_confidence_error'] is not None and
                                 r['spearman_confidence_error'] < 0 for r in q['per_image']),
                             frames=v['frames'], scenes=len(q['spearman_by_scene']),
                             low_C_bins_error2_minus_error1=[dict(lower=b['lower'], upper=b['upper'],
                                 cells=b['pixels'], delta=b['mean_actual_error2'] - b['mean_actual_error1'])
                                 for b in q['bins'] if b['pixels'] and b['upper'] <= .3])
        examples = ARCHIVE / 'arms' / unit / 'run/metrics/confidence_examples' / split
        files = sorted(examples.glob('*.npz'))
        assert len(files) == v['scenes'] == 8
        for file in files:
            with np.load(file, allow_pickle=False) as arrays:
                grids = [arrays[k] for k in ('confidence', 'predicted_error', 'error1', 'error2')]
                assert all(g.shape == grids[0].shape and np.isfinite(g).all() for g in grids)
                assert all(g.ndim == 2 for g in grids)
                assert np.allclose(grids[0], np.exp(-grids[1] / q['tau']), atol=2e-7)
                assert 0 <= grids[0].min() <= grids[0].max() <= 1
        analysis['example_audit'][unit + '/' + split] = dict(files=8, all_finite=True,
            all_shapes_matched=True, confidence_formula_verified=True, selection='fixed first frame of each scene')
    analysis['splits'][split] = dict(frames=len(ids), scenes=scenes,
        A2_T2_minus_A0_T2={k: dual['round2'][k] - control['round2'][k] for k in dual['round2']},
        paired_PSNR_delta_mean=float(delta.mean()), paired_PSNR_delta_median=float(np.median(delta)),
        images_A2_T2_PSNR_higher=int((delta > 0).sum()), images_A2_T2_PSNR_lower=int((delta < 0).sum()),
        positive_PSNR_scene_count=sum(v['mean_paired_PSNR_delta'] > 0 for v in scenes.values()),
        iteration_PSNR_improved_images={u: sum(r['round2']['psnr'] > r['round1']['psnr']
            for r in result[u][split]['per_image']) for u in result},
        T2_minus_source_T2={u: {k: result[u][split]['round2'][k] - source[split]['round2'][k]
            for k in source[split]['round2']} for u in result}, quality=quality)
(ARCHIVE / 'metrics/cpu_result_analysis.json').write_text(json.dumps(analysis, indent=2), encoding='utf-8')

figdir = ARCHIVE / 'analysis/figures'
figdir.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
for ax, split in zip(axes, ('validation', 'test')):
    for unit, label in [('A0_direct', 'Direct control'), ('A2_dual', 'Confidence gates')]:
        bins = result[unit][split]['quality_confidence']['bins']
        # Very sparse high-score bins remain in JSON; plot displays bins with >=1000 cells.
        used = [b for b in bins if b['pixels'] >= 1000]
        ax.plot([b['mean_confidence'] for b in used], [b['mean_actual_error1'] for b in used],
                marker='o', markersize=4, label=label)
    ax.set(title=split.title(), xlabel='Mean confidence in bin', ylabel='Actual T1 local RGB MAE',
           xlim=(0, 1), ylim=(0, .12))
    ax.grid(alpha=.2); ax.legend(frameon=False)
fig.savefig(figdir / 'confidence_error_bins.png', dpi=180)
plt.close(fig)

examples = []
for file in sorted((ARCHIVE / 'arms/A2_dual/run/metrics/confidence_examples/test').glob('*.npz')):
    with np.load(file, allow_pickle=False) as arrays:
        examples.append((file.stem, str(arrays['sample_id']), arrays['confidence'].copy(),
                         arrays['error1'].copy(), arrays['error2'].copy() - arrays['error1'].copy()))
vmax = float(np.quantile(np.concatenate([r[3].ravel() for r in examples]), .95))
delta_lim = float(np.quantile(np.concatenate([np.abs(r[4]).ravel() for r in examples]), .95))
fig, axes = plt.subplots(8, 3, figsize=(10, 16), layout='constrained')
for i, (_, sample_id, c, error, delta) in enumerate(examples):
    axes[i, 0].imshow(c, vmin=0, vmax=1, cmap='viridis')
    axes[i, 1].imshow(error, vmin=0, vmax=vmax, cmap='magma')
    axes[i, 2].imshow(delta, vmin=-delta_lim, vmax=delta_lim, cmap='RdBu_r')
    for ax in axes[i]: ax.set_xticks([]); ax.set_yticks([])
    axes[i, 0].set_ylabel(sample_id, fontsize=7)
axes[0, 0].set_title('Confidence C (0 to 1)')
axes[0, 1].set_title(f'T1 local MAE (0 to {vmax:.3f})')
axes[0, 2].set_title(f'T2 - T1 MAE (±{delta_lim:.5f})\nBlue: lower error; red: higher error')
fig.savefig(figdir / 'confidence_fixed_test_examples.png', dpi=150)
plt.close(fig)

finished = datetime.fromtimestamp(status['time_unix'], timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
lines = ['# T1 质量置信度：16000 步同预算完整实验结果', '',
    '**结论：本轮完成了实现、训练和完整验证/测试，但双重置信门控未带来 PSNR 收益。质量头有误差排序能力，这不足以证明第二轮修复有效。**', '',
    f'GPU 工作于北京时间 {finished} 完成，早于 08:55:33 截止；控制器和截止看守均退出，检查时 GPU 回到 220 MiB / 0%。没有使用剩余时间依据测试结果修改模型或重新训练。', '',
    '## 实验范围与可核查依据', '',
    '同一完整增强/对齐/PolarFree 两轮权重起步，两个组均追加16000步（历史累计29000步），seed2026。共同冻结32000个crop，完整5440张训练池各访问5–6次，crop128、batch2。A0也训练同一质量头，但不使用置信门控；两组恢复/对齐、net_g、H独立Adam分组与梯度裁剪。选步只依据预先固定预算；没有按测试或验证择优。', '',
    '两组各完成全部872张验证、188张测试，均为8个场景；所有逐图ID唯一且与数据清单一致，E/D在验证各调用1744次、测试各376次。每组469个有效Adam状态均达到16000步。来源、最终权重、代码、配置、校准、清单和schedule哈希均核对。', '',
    '协议见[执行协议](CONFIDENCE_OVERNIGHT_PROTOCOL_ZH_20261009.md)，方法与文献见[设计依据](CONFIDENCE_MECHANISM_DESIGN_ZH_20261009.md)。结果来自完整记录，不使用200步代理评价作科研结论。', '']
for split, title in [('validation', '验证集872张'), ('test', '测试集188张')]:
    lines += ['## ' + title, '', '|方案|输出|图像PSNR|场景PSNR|图像SSIM|场景SSIM|', '|---|---|---:|---:|---:|---:|']
    for unit in ('A0_direct', 'A2_dual'):
        for rnd in ('round1', 'round2'):
            v = result[unit][split][rnd]
            lines.append(f"|{unit}|{'T1' if rnd=='round1' else 'T2'}|{v['image_mean_psnr']:.8f}|{v['scene_macro_psnr']:.8f}|{v['image_mean_ssim']:.8f}|{v['scene_macro_ssim']:.8f}|")
    lines += ['', '|比较|图像PSNR差|场景PSNR差|图像SSIM差|场景SSIM差|', '|---|---:|---:|---:|---:|']
    for name, delta in [('A0：T2−T1', result['A0_direct'][split]['iteration_gain']),
                        ('A2：T2−T1', result['A2_dual'][split]['iteration_gain']),
                        ('A2 T2−A0 T2', analysis['splits'][split]['A2_T2_minus_A0_T2'])]:
        lines.append(f"|{name}|{delta['image_mean_psnr']:+.8f}|{delta['scene_macro_psnr']:+.8f}|{delta['image_mean_ssim']:+.8f}|{delta['scene_macro_ssim']:+.8f}|")
    a = analysis['splits'][split]
    lines += ['', f"配对逐图比较：A2 T2的PSNR高于A0为{a['images_A2_T2_PSNR_higher']}/{a['frames']}张，低于为{a['images_A2_T2_PSNR_lower']}张；差值中位数{a['paired_PSNR_delta_median']:+.8f} dB。8个场景中{a['positive_PSNR_scene_count']}个场景均值为正。单seed和相关帧限制了泛化推断；此处是描述统计，没有宣称统计显著性。", '']
lines += ['## 置信质量核验', '', '|组/划分|逐图空间Spearman均值 C与e1|场景宏均值|负相关图像|预测b/真实e1的格点均值比|', '|---|---:|---:|---:|---:|']
for unit in ('A0_direct', 'A2_dual'):
    for split in ('validation', 'test'):
        q = analysis['splits'][split]['quality'][unit]
        lines.append(f"|{unit}/{split}|{q['spearman_image_mean']:.6f}|{q['spearman_scene_macro']:.6f}|{q['images_with_negative_spearman']}/{q['frames']}|{q['predicted_error_over_actual_error']:.6f}|")
lines += ['', 'Spearman按每张图确定性抽样约4096个四分之一分辨率格点计算，再取图像/场景均值；不是全部格点的整体相关性，也不是独立样本显著性检验。质量分箱使用全部格点，权重按格点数；这与图像平均PSNR的权重不同。', '',
    'A2在测试集的10个置信区间中，真实T1局部误差整体随C提高而降低；8个场景的平均相关性均为负，但场景07很弱（约−0.067）。b在全局格点均值上低估真实误差，因此C不能解释为校准后的正确概率。验证集最高置信区间样本很少且误差不完全单调，不能只展示支持预期的分箱。', '',
    '关键负面证据：A2测试中C<0.3的三个区间，T2局部MAE均高于T1；低置信并没有使本轮第二轮修复奏效。A2在高置信区间的偏振修改幅度较小，与门控设计一致，但这是机制行为核验，不能替代恢复收益。不同组输出不同，跨组分箱不能视作同一批格点的因果比较。', '',
    '所有32份NPZ示例（2组×2划分×8场景）均通过形状、有限值及C=exp(−b/τ)核查。示例固定为每场景首帧，不按测试表现挑选。', '',
    '![完整置信分箱](../_dl/confidence_joint_20261009_v2/artifacts/analysis/figures/confidence_error_bins.png)', '',
    '图中显示格点数≥1000的区间；更稀疏区间完整保留在JSON。', '',
    '[全部8个测试场景固定首帧：C、T1误差、T2−T1误差图](../_dl/confidence_joint_20261009_v2/artifacts/analysis/figures/confidence_fixed_test_examples.png)。蓝色表示局部误差降低、红色表示增加；误差共用色轴，95%分位截断仅为显示，原值未修改。', '',
    '## 与起点比较及结论限制', '',
    '起点是上次完整两轮训练检查点，不是无增强的G6。两组继续训练后均低于起始权重；下表仅描述继续训练全过程的变化，不能把起点差异全部归于置信模块。', '',
    '|组/划分|本轮T2−起点T2：图像PSNR|图像SSIM|', '|---|---:|---:|']
for unit in ('A0_direct', 'A2_dual'):
    for split in ('validation', 'test'):
        d = analysis['splits'][split]['T2_minus_source_T2'][unit]
        lines.append(f"|{unit}/{split}|{d['image_mean_psnr']:+.8f}|{d['image_mean_ssim']:+.8f}|")
lines += ['', '实际证据支持：质量排序已学到，当前双重门控的PSNR收益没有成立，第二轮平均PSNR仍不优于第一轮。测试SSIM有很小的正差，不能据此宣称全面改善；验证SSIM也没有超过同预算对照。', '',
    '两组同时下降，说明继续训练策略的适配也是待查因素。是否过度训练、学习率/恢复与对齐目标失衡或门控压低有效修改幅度，均只是基于本轮现象的待验证解释，现有两组不能分离这些原因。没有在看到测试之后回选较优中间权重，也没有擅自调整超参数。', '',
    '单seed；没有单独反馈门控、仅更新门控、空间常数门控、质量头去偏振输入等消融，尚不能确认偏振cue的增量价值与两处门控各自作用。测试集此前被反复查看，本轮结果属于探索阶段。后续若获新GPU预算，应在训练/验证证据上先查共性退化，再检验门控变量，当前结果不足以确认方案已有效。', '',
    '## 失败记录与归档', '',
    '首次目录confidence_joint_20261009在200步后因缺少SciPy导入失败，未保存检查点；失败源码/日志单独保留。v2仅将秩相关实现换成NumPy平均并列秩定义，从共同来源重新起步，不改变网络/科学变量。最终CPU归档审核曾误把元数据confidence_enabled理解为门控启用；实际它表示扩展质量头存在，两个组均为true，confidence_mode=direct/dual才控制门控。审核判据修正后通过；模型和指标未更改。', '',
    '本地完整归档：[artifacts](../_dl/confidence_joint_20261009_v2/artifacts)，压缩原始包：[packages](../_dl/confidence_joint_20261009_v2/packages)。原始包含142项证据文件，全部与服务器源哈希/大小一致；最终两个step16000权重均已本地下载。协议快照中的“运行中”是启动时状态，以本文和completed状态为当前记录。CPU补充分析及显示图另列最终本地哈希清单。', '',
    '|内容|SHA256|', '|---|---|',
    '|来源完整两轮 step10000|eb17a5537bcd26583b73670a34a258f948056935f851b42c7a44eb62e1dd15ea|',
    '|A0 step16000|c863715330af2a9becab35d54a20565ec3a4b30f49c99c509a0a43d7fb07a555|',
    '|A2 step16000|70579d8ef3b5627fa618b1a2b391bbd13e57f71a8e5b9f4979ae2319faf005c2|',
    '|数据清单|d4f04a73e2d1b99d11a3487591b1143be79bfef2bb1ca61cc989823e190a57bb|',
    '|共享schedule|97ff589d0323047d00dbe4c2ecc77f7076992eaca2f830c18d08e378d9ba954a|',
    '|原始归档包|18ed58a5c8cef18af466fe1e2c58b4aee2d32c36dde04ac7344c75c40a9b9b00|', '',
    '核查入口：[完整联合结果JSON](../_dl/confidence_joint_20261009_v2/artifacts/metrics/combined_results.json)、[CPU比较分析](../_dl/confidence_joint_20261009_v2/artifacts/metrics/cpu_result_analysis.json)、[完成审核](../_dl/confidence_joint_20261009_v2/artifacts/operations/final_completion_audit.json)、[本地下载核查](../_dl/confidence_joint_20261009_v2/artifacts/operations/local_archive_verification.json)。']
report = PROJECT / 'analysis/CONFIDENCE_OVERNIGHT_RESULTS_ZH_20261009.md'
report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
# Preserve the server's hashed report; this is an explicitly separate CPU addendum.
archive_text = report.read_text(encoding='utf-8').replace(
    '../_dl/confidence_joint_20261009_v2/artifacts/', '../').replace(
    '../_dl/confidence_joint_20261009_v2/artifacts)', '..)').replace(
    '../_dl/confidence_joint_20261009_v2/packages)', '../../packages)')
(ARCHIVE / 'analysis/CONFIDENCE_RESULTS_ANALYSIS_ZH_20261009.md').write_text(archive_text, encoding='utf-8')
print(json.dumps({s: {k: v for k, v in a.items() if k not in {'scenes', 'quality'}}
                  for s, a in analysis['splits'].items()}, indent=2))
