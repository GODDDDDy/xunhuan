"""Render numeric evidence tables from the audited paired-seed summary."""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
s = json.loads((ROOT / 'analysis/threeway_paired_summary.json').read_text())
assert json.loads((ROOT / 'status.json').read_text())['state'] == 'complete'
day = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')
keys = ['image_mean_psnr', 'scene_macro_psnr', 'image_mean_ssim', 'scene_macro_ssim']
heads = ['图像 PSNR', '场景宏均 PSNR', '图像 SSIM', '场景宏均 SSIM']
lines = ['# 相同任务损失下的 G6 / K1 / K2 三种子配对实验', '',
         f'完成日期：{day}。实验协议于2026-10-02开始锁定；固定 step3000、872 validation、8个场景目录。', '',
         '## 实验实际比较了什么', '',
         '- G6：原始输入直接进 TaskNet，只更新 net_g，冻结扩散先验；无 R2P、无 alignment。',
         '- K1：R2P 单次处理偏振，联合更新 R2P、对齐投影、net_g。',
         '- K2：相同初始化与优化器的共享权重两轮方案，不开任务反馈，保留跨轮梯度。', '',
         '三组任务损失均为 L1+(1−SSIM)+0.1×Fourier phase MSE+0.1×VGG L1。G6/K1监督权重为1；K2两轮为[1/3,2/3]，总权重统一为1。K1/K2 alignment λ固定为0.030651881051471687。', '',
         '**新 K2 不等于此前监督权重[0.5,1]的 K2。** 旧结果只作为seed2026的归一化敏感性对照，未混入三种子汇总。K1−G6包含偏振处理、R2P更新及alignment的整套变化，不能解释为R2P单模块贡献。', '',
         '训练seed为2026/2027/2028；同seed共用取图/裁剪顺序与训练 diffusion_seed。最终推理统一evaluation_seed=2026；因此本轮估计训练随机性，不估计推理采样随机性。复用已有G6 seed2026，其余8组为新训练。', '',
         '## 全部原值', '', '|输出 / seed|' + '|'.join(heads) + '|', '|---|' + '|'.join(['---:']*4) + '|']
for arm in ['G6', 'K1', 'K2_T1', 'K2_T2']:
    for seed in s['seeds']:
        t = s['per_seed_tables'][f'{arm}_seed{seed}']
        lines.append('|' + f'{arm} / {seed}' + '|' + '|'.join(f'{t[k]:.8f}' for k in keys) + '|')
lines += ['', '## 模型的种子均值与样本标准差', '',
          '下表“±”仅表示3个训练seed的样本标准差，不是置信区间、等价界限或显著性检验。', '',
          '|输出|' + '|'.join(heads) + '|', '|---|' + '|'.join(['---:']*4) + '|']
for arm, t in s['model_seed_summary'].items():
    lines.append('|' + arm + '|' + '|'.join(f"{t[k]['mean']:.8f} ± {t[k]['sample_sd']:.8f}" for k in keys) + '|')
lines += ['', '## 必须使用配对差值判断增量', '',
          '跨模型比较用同一训练seed的差值，而非把各模型标准差当作增量的噪声阈值。', '']
for name, c in s['comparisons'].items():
    lines += [f'### {name.replace("_minus_", " − ")}', '',
              '|指标|2026差值|2027差值|2028差值|配对均值|样本SD|正向seed数|',
              '|---|---:|---:|---:|---:|---:|---:|']
    for k, label in zip(keys, heads):
        t = c['paired_seed_summary'][k]
        lines.append('|' + label + '|' + '|'.join(f'{v:+.8f}' for v in t['values']) + f"|{t['mean']:+.8f}|{t['sample_sd']:.8f}|{t['positive_seeds']}/3|")
    lines += ['', '逐场景配对差值（3seed均值±样本SD；不同聚合回答不同问题）：', '',
              '|场景目录|PSNR差|SSIM差|', '|---|---:|---:|']
    for scene, st in c['scene_seed_summary'].items():
        lines.append(f"|{scene}|{st['psnr']['mean']:+.8f} ± {st['psnr']['sample_sd']:.8f}|{st['ssim']['mean']:+.8f} ± {st['ssim']['sample_sd']:.8f}|")
    lines += ['', '难度分组属于事后描述；不据此重新选模型或排除不利样本。', '',
              '|seed|easy图数|easy PSNR差|easy SSIM差|hard图数|hard PSNR差|hard SSIM差|PSNR退化图数|SSIM退化图数|',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for seed, ps in c['per_seed'].items():
        e, h = ps['difficulty_posthoc']['easy'], ps['difficulty_posthoc']['hard']
        lines.append(f"|{seed}|{e['n']}|{e['psnr']:+.8f}|{e['ssim']:+.8f}|{h['n']}|{h['psnr']:+.8f}|{h['ssim']:+.8f}|{ps['negative_images']['psnr']}|{ps['negative_images']['ssim']}|")

lines += ['', '## 与固定官方 G0 的比较边界', '',
          'G0是一个固定官方模型参考，并非3次独立训练。三条G6/G0等差值表示3个新训练模型分别对这个固定参考的差值；没有复制G0生成3个独立样本。', '',
          '|G0固定参考|' + '|'.join(heads) + '|', '|---|' + '|'.join(['---:']*4) + '|',
          '|官方G0|' + '|'.join(f"{s['G0_fixed_table'][k]:.8f}" for k in keys) + '|', '',
          '## 损失归一化的历史敏感性参照（仅seed2026）', '',
          '|新权重[1/3,2/3] − 原权重[0.5,1]|' + '|'.join(heads) + '|', '|---|' + '|'.join(['---:']*4) + '|']
for rnd, t in s['K2_seed2026_normalization_minus_historical_05_1'].items():
    lines.append('|' + rnd + '|' + '|'.join(f'{t[k]:+.8f}' for k in keys) + '|')
lines += ['', '这只是同seed的损失权重敏感性描述，不能证明归一化效应在其他seed稳定。', '',
          '## 审计与复现', '',
          '9个单元的3000步日志均连续、有限；同seed训练随机流匹配；K1/K2全部初始模型与net_g张量哈希匹配；新增组的每200步冻结审计与最终优化器状态审计通过。实际872推理中G6/K1各调用TaskNet872次，K2为1744次；K1继承的伪第二轮别名已单独保存并从正式结果中移除。', '',
          '第一次队列启动在任何优化器更新前被协议哈希保护拦截：固定λ校准绑定缺少新包装器哈希。已保存失败代码/协议/日志，修复绑定并锁定修订；数值λ及损失未改变。最终推理随机流固定策略在首个新872评估启动前锁定，未依据评估结果选取。详见protocol_lock.json的修订记录与diagnostics/startup_failure。', '',
          '|单元|最终检查点SHA256|指标文件SHA256|', '|---|---|---|']
for name, fp in s['source_fingerprints'].items():
    if name != 'G0_fixed_official':
        lines.append(f"|{name}|{fp['checkpoint_sha256']}|{fp['metrics_sha256']}|")
lines += ['', '## 解释边界', '',
          '- 仅3seed、固定3000步、同一872验证集，属于初步探索；此前探索已使用该验证集，不能据此宣称独立验证。',
          '- 相同更新次数不等于相同计算预算；K2每步两轮。',
          '- 图像均值与场景宏均均合法，必须并列，不能选择有利口径；退化场景和hard子集均保留。',
          '- T2−T1是同检查点的推理变化；K2−K1才是训练方案比较，二者不能互相替代。',
          '- 未检验等价性；不凭小差值断言“在噪声内”，不凭3seed均值否定所有循环/联合优化方法。',
          '- 未启动K3、feedback新训练、扩大训练预算、新结构或新增test。', '',
          '方法资料：[Bouthillier et al., Accounting for Variance in Machine Learning Benchmarks, MLSys 2021](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf)讨论多种随机性来源，不保证3seed足够；[PyTorch clip_grad_norm_](https://docs.pytorch.org/docs/main/generated/torch.nn.utils.clip_grad_norm_.html)说明梯度范数裁剪定义。本轮规范化控制的理由来自实际训练代码，不是已证明的退化原因。', '',
          '机器可核查的完整逐图差值与哈希在analysis/threeway_paired_summary.json；原指标、日志、配置及修订历史在证据包。训练检查点保留服务器，未下载大型权重。', '']
path = ROOT / 'analysis' / f'NORMALIZED_THREEWAY_RESULTS_ZH_{day}.md'
path.write_text('\n'.join(lines), encoding='utf-8')
print(path)
