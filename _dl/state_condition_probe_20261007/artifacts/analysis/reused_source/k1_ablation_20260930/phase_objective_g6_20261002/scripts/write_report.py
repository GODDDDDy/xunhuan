"""Write the completed experimental report from verified, unrounded JSON results."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = json.loads((ROOT / 'analysis/matched_g6_summary.json').read_text())
T = S['tables']
C = S['comparisons']
keys = ['image_mean_psnr', 'scene_macro_psnr', 'image_mean_ssim', 'scene_macro_ssim']
labels = {'G0_official': 'G0 官方，未微调', 'G6_phase1': 'G6，phase=1',
          'G6_phase01': 'G6，phase=0.1', 'K2_phase1_T1': 'K2 phase=1，T1',
          'K2_phase1_T2': 'K2 phase=1，T2', 'K2_phase01_T1': 'K2 phase=0.1，T1',
          'K2_phase01_T2': 'K2 phase=0.1，T2'}
fmt = lambda row: f"{row[keys[0]]:.6f} | {row[keys[1]]:.6f} | {row[keys[2]]:.8f} | {row[keys[3]]:.8f}"
lines = ['# G6 phase=0.1 匹配对照：完整 872 验证结果', '',
         '执行日期：2026-10-02。状态：3000 步训练、完整 872 validation 评估及审计完成。所有判断限于 seed2026 的本次观测；本验证集已参与探索，不能当作独立确认。', '',
         '## 1. 实验回答什么', '',
         '本轮补齐普通 net_g 微调的 phase=0.1 对照：原始偏振与原始混合 RGB 直接走官方 TaskNet 输入流程；不调用 R2P，不使用 alignment，只训练 net_g，冻结扩散先验。与历史 G6 相比，唯一训练目标改动是 Fourier phase MSE 权重从 1 降到 0.1。', '',
         '同权重的新 K2 与新 G6 的差异，是两套完整方案的差异，包含 R2P 输入/更新、两轮监督、对齐与共享权重。该比较不能拆成单一模块增益，更不能单独证明循环有效。', '',
         '## 2. 完整主表', '',
         '| 模型输出 | PSNR 图像均值 / dB | PSNR 场景宏均 / dB | SSIM 图像均值 | SSIM 场景宏均 |',
         '|---|---:|---:|---:|---:|']
for n in ('G0_official', 'G6_phase1', 'G6_phase01', 'K2_phase1_T1', 'K2_phase1_T2', 'K2_phase01_T1', 'K2_phase01_T2'):
    lines.append('| ' + labels[n] + ' | ' + fmt(T[n]) + ' |')
lines += ['', '两种聚合并列保留：图像均值回答该 872 图集合上的平均表现；场景宏均使各场景权重相等。场景样本数量不同是合法权重，不剔除样本较多或表现较差的场景来改写结论。', '',
          '## 3. 关键比较（前者减后者）', '',
          '| 比较 | ΔPSNR 图像 / dB | ΔPSNR 场景 / dB | ΔSSIM 图像 | ΔSSIM 场景 |',
          '|---|---:|---:|---:|---:|']
pair_labels = {'G6_phase01_minus_G6_phase1': 'G6 phase0.1 − G6 phase1',
               'G6_phase01_minus_G0_official': 'G6 phase0.1 − G0',
               'K2_phase01_T1_minus_G6_phase01': 'K2 phase0.1 T1 − G6 phase0.1',
               'K2_phase01_T2_minus_G6_phase01': 'K2 phase0.1 T2 − G6 phase0.1',
               'K2_phase01_T2_minus_K2_phase01_T1': 'K2 phase0.1 T2 − T1',
               'K2_phase1_T1_minus_G6_phase1': 'K2 phase1 T1 − G6 phase1',
               'K2_phase1_T2_minus_G6_phase1': 'K2 phase1 T2 − G6 phase1'}
for n, label in pair_labels.items():
    lines.append('| ' + label + ' | ' + fmt(C[n]['aggregate']) + ' |')
g = C['G6_phase01_minus_G6_phase1']['aggregate']
q = C['K2_phase01_T2_minus_G6_phase01']['aggregate']
lines += ['', '## 4. 本次证据允许的解释', '',
          f"降低 phase 权重后，普通微调的图像均值 PSNR 变化为 {g['image_mean_psnr']:+.6f} dB，SSIM 为 {g['image_mean_ssim']:+.8f}；场景宏均变化见上表。"]
if all(g[k] > 0 for k in keys):
    lines += ['本次 G6 的四项汇总均改善。因此，新损失带来的改善也出现在完全没有 R2P、循环或对齐的普通微调中；不能把此前 K2 换损失后的改善全部解释为这些模块的贡献。']
else:
    lines += ['本次 G6 的不同指标/聚合需分别判断，不能概括为所有指标都改善。']
if all(q[k] < 0 for k in keys):
    lines += ['在相同 phase=0.1 下，新 K2 的 T2 四项汇总均低于普通微调对照。本次没有证据支持叠加现有 K2 整体方案能带来额外收益；这不等同于否定其他循环或联合优化设计。']
elif all(q[k] > 0 for k in keys):
    lines += ['在相同 phase=0.1 下，新 K2 的 T2 四项汇总均高于普通微调对照。这是当前整体方案的单次正向观测，尚不能归因到 R2P、对齐或循环中的某一个环节，也不能替代多 seed 验证。']
else:
    lines += ['在相同 phase=0.1 下，新 K2 与普通微调的比较随指标或聚合口径改变方向，不能宣称统一的额外收益。']
lines += ['现有新 K2 的 T2−T1 仍只有约 +0.001 dB 的图像均值 PSNR，SSIM 微降，且多数图像第二轮退化。新 G6 对照不能改变这个循环自身的观察，也不能被用来证明或排除具体退化机制。', '',
          '必须同时保留普通微调的代价：新 G6 相对官方 G0 虽然四项汇总均提高，但 PSNR/SSIM 只有 3/8 个场景为正，516/872 张图的 SSIM 下降；hard 子集平均 PSNR −0.321382 dB、SSIM −0.00606038。场景 56 平均 PSNR −2.653447 dB、SSIM −0.02735717。因此不能把当前微调概括为普遍改善，或据此判定 PolarFree 原模型训练不足。', '',
          '这些数字不是总体效应、置信区间或等价检验。未估计 K2/G6 成对训练的种子间变异；不使用“在噪声内”“基本等价”或“几乎全部来自某模块”等表述。', '',
          '## 5. 场景结果与负向案例', '']
for name in ('G6_phase01_minus_G6_phase1', 'K2_phase01_T1_minus_G6_phase01', 'K2_phase01_T2_minus_G6_phase01', 'G6_phase01_minus_G0_official'):
    comp = C[name]
    lines += ['### ' + pair_labels[name], '', '| 场景 | n | ΔPSNR / dB | ΔSSIM |', '|---|---:|---:|---:|']
    for scene, values in comp['per_scene'].items():
        lines.append(f"| {scene} | {values['n']} | {values['psnr']:+.6f} | {values['ssim']:+.8f} |")
    lines += ['', f"正向图像：PSNR {comp['positive_images']['psnr']}/872，SSIM {comp['positive_images']['ssim']}/872；负向图像分别 {comp['negative_images']['psnr']} 和 {comp['negative_images']['ssim']}。正向场景：PSNR {comp['positive_scenes']['psnr']}/8，SSIM {comp['positive_scenes']['ssim']}/8。", '',
              '按 easy/hard 的事后描述（不是新增独立检验）：']
    for difficulty, values in comp['difficulty_posthoc'].items():
        lines.append(f"- {difficulty}，n={values['n']}：ΔPSNR {values['psnr']:+.6f} dB，ΔSSIM {values['ssim']:+.8f}。")
    lines.append('')
lines += ['## 6. 损失分项（同一次输出，无额外 TaskNet 前向）', '',
          '| 新 G6 phase0.1 分项 | 图像均值 | 场景宏均 |', '|---|---:|---:|']
for key in S['loss_parts_G6_phase01']['image_mean']:
    lines.append(f"| {key} | {S['loss_parts_G6_phase01']['image_mean'][key]:.10f} | {S['loss_parts_G6_phase01']['scene_macro'][key]:.10f} |")
lines += ['', 'phase_raw 是原实现中的角度 MSE；phase_weighted=0.1×phase_raw。common_phase1 / common_phase01 只用于相同输出在统一系数下的数值比较，不是额外训练或额外推理。未将某个分项变小直接等同于机制被证实。', '',
          '## 7. 复现与审计', '',
          '- 3000 步固定预算、seed2026、crop128、batch2、Adam LR=1e-5、betas=(0.9,0.999)、weight_decay=0、net_g clip=1；TaskNet 保持 eval 模式。',
          '- 与新 K2 的 436 个初始 net_g 参数张量逐元素一致；manifest/schedule/base_config 等锁定路径与 SHA256 保留。与新 K2、历史 G6 的 3000 个逐步 diffusion_seed 全部一致。',
          '- R2P 全参数填 NaN 后训练损失不变，确认真实旁路；3 张 step0 图的新旧 G6 输出最大/平均差均为 0。此处只核验该样本范围，不声称覆盖所有实现问题。',
          '- 200 步门禁通过后恢复同一个优化器继续到 3000；1–3000 日志连续且有限，Adam 的 436 个状态步数均为 3000，最终权重及优化器张量均有限。',
          '- 15 个保存时点分别核对 R2P 18、projections 3、prior 125 个参数/缓冲区张量，全部零变化；prior 排除内部可训练 net_g 后审计，未使用名称混淆的旧判定。',
          '- 完整评估 872 张、8 场景，逐图 ID/场景与全部比较对象顺序一致。附加损失运算保持 CPU/全部 CUDA RNG 不变；指标与损失结果绑定最终检查点 SHA256。',
          '- 训练前锁定的配置和脚本逐字节不变；汇总/报告脚本在结果分析阶段新增，单独归档。',
          f"- 最终检查点 SHA256：`{S['checkpoint_audit']['sha256']}`。", '',
          '运行计时（包括对应训练阶段的原代理验证/保存，不当作公平计算预算比较）：']
for timing in S['timing']:
    lines.append(f"- step {timing['start_step']}→{timing['end_step']}：{timing['wall_seconds']/60:.2f} min；平均更新耗时（排除前 20 步）{timing['mean_step_seconds_excluding_first_20']:.4f} s；CUDA peak allocated {timing['cuda_peak_memory_bytes']/2**30:.3f} GiB。")
lines += ['', '## 8. 限制与下一步', '',
          'G6 一轮监督权重为 1，K2 两轮为 [0.5,1]，且 K2 有额外 alignment 和 R2P 梯度；本轮没有把总损失权重或裁剪后更新强度重标定为相同数值。相同步数、相同数据顺序与 net_g 学习率，只能称匹配这些条件，不能称所有训练条件或计算量相同。', '',
          '若继续评估循环的净贡献，应补 phase=0.1 的单轮联合模型与两轮模型的成对对照，显式匹配可训练模块、对齐、初始化、学习率和监督/计算预算的比较口径；不能用旧 phase=1 的 K1 直接替代。若要确认当前整体比较的方向，再考虑同平台成对增加 seed。现有结果尚不能据此选出特定新架构或证明步长、共享权重、对齐是根因。以上为后续建议，本轮未启动这些任务。', '',
          '## 9. 证据位置', '',
          f'服务器实验根：`{ROOT}`。',
          '最终检查点：`runs/G6_phase01/seed_2026/checkpoints/step_3000.pt`。原始逐图指标/损失在该运行的 `metrics/`；日志与逐保存点审计在 `logs/`、`diagnostics/`。',
          '汇总：`analysis/matched_g6_summary.json`，含未舍入主表、逐图/逐场景差值、负向计数、难度描述、匹配检查及全部引用文件指纹。历史 G0/G6 与 K2 指标副本保留在 `analysis/references/`。',
          '本地报告：`F:\\WORK\\projects\\pidsr_joint_alignment\\analysis\\G6_PHASE01_MATCHED_REPORT_ZH_20261002.md`。本地证据归档：`F:\\WORK\\projects\\pidsr_joint_alignment\\_dl\\phase_objective_g6_20261002\\`。', '',
          '实验事实以以上实际产物和归档源码为依据。本报告没有把推论表述为文献已证明的结论，也没有新增关于 PolarFree 原训练充分性或偏振物理质量的判断。', '']
report = ROOT / 'analysis/G6_PHASE01_MATCHED_REPORT_ZH_20261002.md'
report.write_text('\n'.join(lines), encoding='utf-8')
print(report)
