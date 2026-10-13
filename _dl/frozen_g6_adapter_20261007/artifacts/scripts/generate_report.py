"""Generate factual tables; interpretations reviewed against raw outputs separately."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
s=json.loads((ROOT/'analysis/frozen_adapter_paired_summary.json').read_text())
keys=['image_mean_psnr','scene_macro_psnr','image_mean_ssim','scene_macro_ssim']
heads=['图像PSNR','场景宏均PSNR','图像SSIM','场景宏均SSIM']
c=s['comparison_adapter_minus_G6']
lines=['# 固定 G6、只训练偏振适配器：三组结果','',
    '## 本轮结论（实测，限定当前协议）','',
    '这套“已训练G6全部冻结、适配器以零修改起步、额外训练3000步”的协议未显示整体增量。相对同seed G6，图像PSNR三个seed均下降，配对均值−0.00402904 dB；场景宏均PSNR均值−0.00560230 dB（1/3为正）；图像/场景SSIM均值−0.00005371/−0.00008536，均三个seed下降。差值小，但未进行等价性或显著性检验，不能称噪声内、等价或显著退化。','',
    'hard子集三个seed的PSNR/SSIM均下降；场景56也是三个seed两项均下降。按三个seed平均差，8个场景中仅场景55的PSNR为正、仅场景22的SSIM为正。这些不是需要排除的异常，均保留在主报告。','',
    '适配器确实学到了非零偏振修改（有效mean abs约0.00186–0.00216），梯度与冻结核查也通过；因此“完全没更新”不能解释这次结果。观测支持的判断是：在当前模块、目标、初始化、数据日程和预算下，冻结下游并未获得额外平均收益。不能单独定位是模块表示、优化目标、初始化或预算问题，也不能据此否定全部循环/联合优化方案。','',
    '结合此前K1去对齐仍无稳定统一优势，此结果未支持“只去掉对齐、或固定下游就能使现有模块自然产生收益”的预期；仍不是对各因素的完整因果排查。本轮到此停止，后续若修改设计，应提出具体、可被对照推翻的假设；不将额外训练或堆叠循环当作已证明的解决办法。','',
    '固定额外训练3000步；完整872 validation；8个场景目录；最终推理seed2026。','',
    '## 实验定义','',
    '分别载入已训练3000步的G6 seed2026/2027/2028，冻结整个TaskNet参数与缓冲区，只学习R2P。R2P使用e42预训练特征部分，残差输出卷积权重/偏置置零，以严格零修改起步。不计算alignment，不更新投影/反馈模块，无反馈，一轮前向。','',
    '沿用batch2、crop128、workers2、原训练取图/裁剪与diffusion随机日程；R2P LR1e-4，Adam(0.9,0.999)，weight decay0，模型梯度裁剪1。任务损失为L1+(1−SSIM)+0.1相位MSE+0.1 VGG L1。固定step3000，不以代理验证或best.pt选择。','',
    '注意新增预算：源G6已有3000步，本轮另增加3000步适配器更新并重复使用原训练日程。比较是同一冻结G6上的增量，不是与旧联合模型等总预算的排名；改变了R2P初始化、任务网络起点与训练方式，不能分解为单因素机制。','',
    '## 原始汇总值','', '|模型 / seed|'+'|'.join(heads)+'|','|---|'+'|'.join(['---:']*4)+'|']
for name in ('G6','adapter'):
    for seed in s['seeds']:
        t=s['per_seed_tables'][f'{name}_seed{seed}']
        lines.append(f'|{name} / {seed}|'+'|'.join(f'{t[k]:.8f}' for k in keys)+'|')
lines+=['','## 适配器 − 同seed冻结G6','',
    '样本SD为三个配对差值的描述性标准差，不是置信区间、显著性或等价性检验。','',
    '|指标|seed2026|seed2027|seed2028|差值均值|样本SD|正向seed数|','|---|---:|---:|---:|---:|---:|---:|']
for k,h in zip(keys,heads):
    t=c['paired_seed_summary'][k]
    lines.append('|'+h+'|'+'|'.join(f'{v:+.8f}' for v in t['values'])+f"|{t['mean']:+.8f}|{t['sample_sd']:.8f}|{t['positive_seeds']}/3|")
lines+=['','## 场景分组（描述性）','', '|场景目录|PSNR差均值±样本SD|SSIM差均值±样本SD|','|---|---:|---:|']
for scene,t in c['scene_seed_summary'].items():
    lines.append(f"|{scene}|{t['psnr']['mean']:+.8f} ± {t['psnr']['sample_sd']:.8f}|{t['ssim']['mean']:+.8f} ± {t['ssim']['sample_sd']:.8f}|")
lines+=['','## 难度与退化图数（事后描述）','',
    '|seed|easy张数|easy PSNR差|easy SSIM差|hard张数|hard PSNR差|hard SSIM差|PSNR退化图数|SSIM退化图数|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for seed,t in c['per_seed'].items():
    e,h=t['difficulty_posthoc']['easy'],t['difficulty_posthoc']['hard']
    lines.append(f"|{seed}|{e['n']}|{e['psnr']:+.8f}|{e['ssim']:+.8f}|{h['n']}|{h['psnr']:+.8f}|{h['ssim']:+.8f}|{t['negative_images']['psnr']}|{t['negative_images']['ssim']}|")
lines+=['','## 审计结果与基线来源','',
    '三组均核验：初始裁剪的偏振修改严格为0；梯度到达偏振输入和R2P而不进入冻结参数；每组28张全分辨率代理图按原完整验证索引恢复随机流，零修改输入与直接G6输入相同，输出及历史指标在锁定容差内一致。完整872张G6基线复用历史结果；本轮只在28张核验，不声称重做完整step0评估。','',
    '历史G6 seed2026使用旧指标格式，仅有seed=2026、没有后来新增的training_seed/evaluation_seed。原配置和原评估实现表明该seed同时用于训练与验证；已保存原配置及评估/封装/训练器源码的副本和哈希，并有28张原完整索引随机流复现支持。旧指标保持原样。汇总脚本第一次因直接索引旧文件不存在的evaluation_seed字段停止，已按实际旧格式修正汇总审计；没有追加前向、没有重训或改动原指标。','',
    '每组0至3000步的每200步保存点（16点）均通过整个TaskNet参数/缓冲区、投影和反馈冻结核查。200/3000步门控核验只有18个R2P Adam状态更新、net_g与源G6逐元素相同、3000条日志连续且有限；各代理评估点net_g漂移0。最终每组实际TaskNet调用872次。','',
    '|seed|训练步内耗时（分）|完整验证耗时（分）|预检训练峰值显存（GiB）|','|---|---:|---:|---:|']
for seed,t in s['timing'].items():
    lines.append(f"|{seed}|{t['summed_training_step_seconds']/60:.2f}|{t['full_validation_seconds']/60:.2f}|{t['initial_preflight_peak_vram_mb']/1024:.2f}|")
lines+=['','训练步内耗时不含代理验证、保存与重新加载；不能当作全程墙钟时间。','',
    '## 偏振输入实际变化','',
    '以下为截断后的有效修改 P−P0 的逐图统计再取图像均值；与网络原始delta分开。相对变化为逐图 mean|P−P0| / mean|P0| 再平均，不是全数据的两个均值相除。R2P相对e42的权重漂移包含初始输出头重置，不能直接当作训练更新量。','',
    '|seed|有效修改mean abs|相对信号变化|元素修改>0.02比例|元素修改>0.05比例|','|---|---:|---:|---:|---:|']
for seed,t in s['timing'].items():
    p=t['full_validation_effective_polar_update']
    lines.append(f"|{seed}|{p['mean_abs']:.8f}|{p['relative_signal_change']:.8f}|{p['ratio_gt_0_02']:.8f}|{p['ratio_gt_0_05']:.8f}|")
lines+=['',
    '|单元|检查点SHA256|指标SHA256|','|---|---|---|']
for name,t in s['source_fingerprints'].items():
    lines.append(f"|{name}|{t['checkpoint_sha256']}|{t['metrics_sha256']}|")
lines+=['','## 解释范围与文献背景','',
    '- 872 validation此前已经用于探索；本轮不是独立确认，没有增加test。',
    '- 三个seed对应三个不同的已训练G6，差值包含源模型与适配学习变异，未独立估计适配器随机性。',
    '- 源G6已有3000步，本轮再增加3000步，重复原训练日程；不能与旧联合方案作等总预算排名。',
    '- 零输出头改变了原e42初始化；本轮不是旧联合模型的单因素消融。',
    '- 历史完整G6基线复用，仅在每组28张全分辨率代理图上复现核验。',
    '- K1目录及round2日志字段继承旧实现；round2是单次实际输出的别名。',
    '- 最终推理固定seed2026，未估计推理随机采样变异。','',
    '该实验是基于项目消融证据提出的分阶段检验，不是文献已经证明有效的方法。不能单独定位初始化、冻结或结构的因果贡献；不能直接判断循环是否有效。','',
    '[PolarAPP](https://arxiv.org/html/2603.23071v2)的元学习对齐与任务驱动精炼可作为背景，但不同于本项目普通余弦对齐；[PIDSR](https://arxiv.org/html/2504.07758v1)的尺度递进也不同于本项目同尺度模块。[Bouthillier et al., MLSys2021](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf)讨论实验变异；三个seed只提供初步描述，不保证足以判定总体效应。','',
    '逐图原指标、差值、配置、审计与日志保留在证据包；模型检查点留在服务器。','']
path=ROOT/'analysis/FROZEN_G6_ADAPTER_RESULTS_ZH_20261007.md'
path.write_text('\n'.join(lines),encoding='utf-8');print(path)
