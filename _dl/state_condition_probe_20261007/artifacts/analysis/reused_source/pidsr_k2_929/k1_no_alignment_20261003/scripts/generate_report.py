"""Render all numeric results; conclusions are reviewed separately after reading them."""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parent.parent
s=json.loads((ROOT/'analysis/alignment_paired_summary.json').read_text())
day=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')
keys=['image_mean_psnr','scene_macro_psnr','image_mean_ssim','scene_macro_ssim']
heads=['图像PSNR','场景宏均PSNR','图像SSIM','场景宏均SSIM']
lines=['# K1无alignment：三个训练seed的配对消融结果','',f'完成日期：{day}。固定3000步、完整872 validation、8个场景目录。','',
 '## 比较定义','',
 '本轮新增K1 λ=0的seed2026/2027/2028。对照复用此前同seed K1 λ=0.030651881051471687及G6。任务phase系数0.1、单轮任务权重1。唯一训练目标干预为关闭alignment监督；保留原模型、投影、计算图、优化器参数组与裁剪方式。alignment标量继续记录，但乘0、不贡献梯度。','',
 '全部初始模型/net_g哈希与旧K1配对一致；manifest、取图/裁剪schedule、训练逐步diffusion_seed相同。最终推理固定evaluation_seed2026。只选step3000，不选best.pt，未新增test。','',
 '训练batch2、crop128、workers2，R2P/投影LR1e-4、net_g LR1e-5，Adam(0.9,0.999)、wd0，模型组/net_g分别clip1。冻结扩散先验但保留向偏振输入的梯度。任务损失为L1+(1−SSIM)+0.1×Fourier phase MSE+0.1×VGG L1。','',
 '## 全部原值','', '|模型 / seed|'+'|'.join(heads)+'|','|---|'+'|'.join(['---:']*4)+'|']
for name in ('K1_no_alignment','K1_aligned','G6'):
 for seed in s['seeds']:
  t=s['per_seed_tables'][f'{name}_seed{seed}']
  lines.append(f'|{name} / {seed}|'+'|'.join(f'{t[k]:.8f}' for k in keys)+'|')
lines+=['','## 训练seed均值与样本SD','', '“±”仅为三个训练seed的样本标准差，不是置信区间、显著性或等价性检验。','',
 '|模型|'+'|'.join(heads)+'|','|---|'+'|'.join(['---:']*4)+'|']
for name,t in s['model_seed_summary'].items():
 lines.append('|'+name+'|'+'|'.join(f"{t[k]['mean']:.8f} ± {t[k]['sample_sd']:.8f}" for k in keys)+'|')
for name,c in s['comparisons'].items():
 lines+=['',f'## {name.replace("_minus_"," − ")}','',
  '|指标|seed2026差值|seed2027差值|seed2028差值|配对均值|样本SD|正向seed数|','|---|---:|---:|---:|---:|---:|---:|']
 for k,label in zip(keys,heads):
  t=c['paired_seed_summary'][k]
  lines.append('|'+label+'|'+'|'.join(f'{v:+.8f}' for v in t['values'])+f"|{t['mean']:+.8f}|{t['sample_sd']:.8f}|{t['positive_seeds']}/3|")
 lines+=['','逐场景配对差值的seed均值±样本SD：','','|场景目录|PSNR差|SSIM差|','|---|---:|---:|']
 for scene,t in c['scene_seed_summary'].items():
  lines.append(f"|{scene}|{t['psnr']['mean']:+.8f} ± {t['psnr']['sample_sd']:.8f}|{t['ssim']['mean']:+.8f} ± {t['ssim']['sample_sd']:.8f}|")
 lines+=['','难度分组为事后描述，负面样本不排除：','',
  '|seed|easy图数|easy PSNR差|easy SSIM差|hard图数|hard PSNR差|hard SSIM差|PSNR退化图数|SSIM退化图数|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for seed,t in c['per_seed'].items():
  e,h=t['difficulty_posthoc']['easy'],t['difficulty_posthoc']['hard']
  lines.append(f"|{seed}|{e['n']}|{e['psnr']:+.8f}|{e['ssim']:+.8f}|{h['n']}|{h['psnr']:+.8f}|{h['ssim']:+.8f}|{t['negative_images']['psnr']}|{t['negative_images']['ssim']}|")
lines+=['','## 实施与证据核查','',
 '三组新训练均通过单次R2P/TaskNet预检，λ=0时纯task_loss与总损失对R2P梯度的差符合锁定容差1e-7，投影梯度为0；每200步保存点的prior/未用反馈冻结审计及投影完全不变审计通过。训练日志连续有限、457个Adam状态步数匹配，最终评估TaskNet实际调用872次。','',
 '首次启动因新增预检漏写继承训练器要求的peak_vram_mb字段，在任何优化更新前停止。保留失败脚本、原协议、预检与日志，补实测显存字段并锁定修订后重跑；模型、目标、配置未改变。详见protocol_lock.json的startup_amendment和diagnostics/startup_failure。','',
 '|单元|最终检查点SHA256|指标文件SHA256|','|---|---|---|']
for name,fp in s['source_fingerprints'].items():
 lines.append(f"|{name}|{fp['checkpoint_sha256']}|{fp['metrics_sha256']}|")
lines+=['','## 解释边界与参考','',
 '- 同一已用于探索的872验证集，三个训练seed只补初步训练变异信息，不是独立数据确认。','- 无对齐−有对齐回答当前监督目标的作用；无对齐−G6仍包含R2P输入修改与联合更新，不能拆为单模块或唯一机制。','- 不凭小差值称“在噪声内”或“基本等价”；两种聚合均合法且并列报告。','- 无论结果正负，都不能直接外推为所有alignment、所有循环或R2P结构有效/无效。','',
 '[PolarAPP §3.3](https://arxiv.org/html/2603.23071v2)采用元学习对齐度量，区别于本项目普通投影/余弦对齐；[Bouthillier et al., MLSys2021](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf)讨论训练随机性，不保证三个seed充分。','',
 '完整逐图差值在analysis/alignment_paired_summary.json，原日志/配置/指标/协议修订在证据包；检查点保留服务器。', '']
path=ROOT/'analysis'/f'K1_NO_ALIGNMENT_RESULTS_ZH_{day}.md'
path.write_text('\n'.join(lines),encoding='utf-8')
print(path)
