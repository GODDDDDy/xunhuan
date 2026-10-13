import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
s=json.loads((ROOT/'analysis/condition_probe_summary.json').read_text())
ba=s['comparisons']['task_prediction_minus_mixed_rgb'];bg=s['comparisons']['task_prediction_minus_G6']
keys=['image_mean_psnr','scene_macro_psnr','image_mean_ssim','scene_macro_ssim']
heads=['图像PSNR(dB)','场景PSNR(dB)','图像SSIM','场景SSIM']
lines=['# 第二轮传递诊断与任务结果条件试验','',
    '2026-10-07启动。先完成当前K2的28张固定验证诊断，再完成两组同seed2026、同初始化、同额外1000步的条件表示对照，各完整872 validation。未跑test，未更改旧训练或结果。','',
    '## 主要结论','',
    f'本次试验中，任务预测条件相对原RGB条件的四项聚合差值均正：图像/场景PSNR {ba["aggregate"]["image_mean_psnr"]:+.8f}/{ba["aggregate"]["scene_macro_psnr"]:+.8f} dB，图像/场景SSIM {ba["aggregate"]["image_mean_ssim"]:+.8f}/{ba["aggregate"]["scene_macro_ssim"]:+.8f}。这是不能抹去的单seed正向线索，不能称条件改变毫无作用。','',
    f'但两组相对G6四项聚合差值均负；任务预测条件相对G6的图像/场景PSNR为{bg["aggregate"]["image_mean_psnr"]:+.8f}/{bg["aggregate"]["scene_macro_psnr"]:+.8f} dB，SSIM为{bg["aggregate"]["image_mean_ssim"]:+.8f}/{bg["aggregate"]["scene_macro_ssim"]:+.8f}。本轮没有实现超出冻结G6的平均增量，预先规定的扩展门槛未通过。','',
    f'条件改变不是逐图普遍改善：B相对A的PSNR下降{ba["negative_images"]["psnr"]}/872张、SSIM下降{ba["negative_images"]["ssim"]}/872张；hard子集平均SSIM差{ba["per_difficulty"]["hard"]["ssim"]:+.8f}。场景07两项提高，而其他场景正负混合；它的权重合法，保留全部场景而不删除它重算。','',
    '在所查28张图中，修改确实进入了后续表示，不能用“完全没有送进去”解释。本轮仍不能把问题唯一归因于循环思想、共享权重、STE梯度代理、增强结构或训练预算。1000步只是探索预算，不是收敛证明；按协议停止本轮扩展，不追加seed或预算。','',
    '## 诊断结果','',
    '当前normalized phase0.1 K2 step3000，无显式任务特征反馈。28张覆盖8场景，easy/hard各14张；56个端点独立重算全部通过，T1/T2复现原逐图指标，冻结哈希前后一致。输入修改实际传到8位灰度、AoLP/DoLP、prior_condition及采样prior，五个层面均28/28张有非零变化。只能排除所查图像中“完全没有传递”，不能证明STE梯度代理正确，也不能证明偏振更真实。','',
    'P(alpha)=P1+alpha(P2−P1)，直接送TaskNet、保持同一随机流；下表相对同一模型T1：','',
    '|alpha|'+'|'.join(heads)+'|','|---|'+'|'.join(['---:']*4)+'|']
for alpha,t in s['diagnostic']['delta_summary'].items():lines.append('|'+alpha+'|'+'|'.join(f'{t[k]:+.8f}' for k in keys)+'|')
lines+=['',
    '这28张中的原第二轮平均PSNR有正差，而两个SSIM口径均负；所测非零幅度均没有四项聚合同时正。这是固定子集的描述，不能挑一个alpha宣布有效，也不能解释成所有幅度无效。与188张test中的第二轮PSNR负差并不矛盾：数据组成不同；旧phase=1扫描也不能代替本次结果。','',
    f'第二轮灰度变化像素比例逐图均值：{s["diagnostic"]["gray_changed_fraction_mean"]:.6%}；各层mean abs为逐图均值再平均：','']
lines+=['- '+k+': '+f'{v:.9g}' for k,v in s['diagnostic']['stage_mean_abs'].items()]
lines+=['','AoLP差值采用周期角距离，单位rad；上述各层仅比较同一图的两次实际表示，不是偏振真值误差。',
    '', '残差必须保留clamp：P1=clamp(P0+d1)，P2=clamp(P1+d2)；28张两轮截断恒等式均逐元素成立。以下每项先在每张图内计算，再对28张等权平均，不能把网络原始delta和有效修改混用：', '',
    '|轮次|原始delta mean abs|有效修改mean abs|clamp元素比例均值|', '|---|---:|---:|---:|']
for i,t in s['diagnostic']['round_update_image_means'].items():
    lines.append(f'|{i}|{t["raw_delta_mean_abs"]:.8f}|{t["effective_mean_abs"]:.8f}|{t["clamp_fraction"]:.6%}|')
lines+=['','## 两组受控试验的实际定义','',
    'A=mixed_rgb：原混合RGB作为R2P条件。B=task_prediction：第一遍冻结G6的预测detach后clamp[0,1]作为条件。两组都实际执行原偏振G6预测→一次R2P→修正偏振G6预测；原RGB仍是两次TaskNet主输入，CPU/全部CUDA随机状态恢复使两次任务调用使用同一扩散随机流。只对最终输出计算优化目标，第一遍损失仅记录。GT不参与条件生成。','',
    'A也执行第一遍G6是为了本次计算和随机流控制；实际部署时A可省去这次未参与条件生成的调用，B则需要它。因此本实验相同调用数不等于两种方案部署成本相同。此处冻结整个TaskNet，并非又一组联合更新或共享R2P两轮训练。','',
    f'B条件生成中，T0超出[0,1]并被截断的元素比例逐图均值为{s["audits"]["task_prediction"]["condition_clip_fraction"]:.6%}；不将预测分布改变和截断各自归因为独立机制。','',
    '仅条件表示改变；R2P结构/参数量、e42特征及零输出头、G6、seed、取图裁剪日程、1000步训练随机键、Adam/学习率/裁剪、phase0.1任务损失均匹配。两组初始所有模型与net_g张量一致；只有18个R2P参数张量更新，投影/特征反馈与整个TaskNet冻结。','',
    '源G6已有3000步，本轮再加1000步R2P，不作等总预算排名；旧3000步适配器不能作为本轮同预算A。任务预测仍来自同一观测，不是独立RGB模态。原R2P代码已支持用PolarFree结果作条件，早期也做过任务特征反馈，本轮不声称仅改变条件或加入反馈就是新颖贡献。','',
    '## 完整872验证原值','', '|模型|'+'|'.join(heads)+'|','|---|'+'|'.join(['---:']*4)+'|']
for name,t in s['aggregates'].items():lines.append('|'+name+'|'+'|'.join(f'{t[k]:.8f}' for k in keys)+'|')
lines+=['','适配器的有效偏振修改同样按图内统计后逐图等权平均；比例统计包含偏振和颜色通道的张量元素，相对幅度是输入信号尺度比，不是模块贡献率：','',
    '|条件|有效修改mean abs|相对输入幅度|超过0.02的元素比例|超过0.05的元素比例|', '|---|---:|---:|---:|---:|']
for name,t in s['pilot_effective_polar_update_image_means'].items():
    lines.append(f'|{name}|{t["mean_abs"]:.8f}|{t["relative_signal_change"]:.6%}|{t["ratio_gt_0_02"]:.6%}|{t["ratio_gt_0_05"]:.6%}|')
lines+=['','## 预先指定的配对差值','', '|比较|'+'|'.join(heads)+'|','|---|'+'|'.join(['---:']*4)+'|']
for name,c in s['comparisons'].items():lines.append('|'+name+'|'+'|'.join(f'{c["aggregate"][k]:+.8f}' for k in keys)+'|')
for name,c in s['comparisons'].items():
    lines+=['',f'## {name}：分组与负面结果','', '|场景|图数|ΔPSNR|ΔSSIM|','|---|---:|---:|---:|']
    for scene,t in c['per_scene'].items():lines.append(f'|{scene}|{t["n"]}|{t["psnr"]:+.8f}|{t["ssim"]:+.8f}|')
    lines+=['','|难度|图数|ΔPSNR|ΔSSIM|','|---|---:|---:|---:|']
    for difficulty,t in c['per_difficulty'].items():lines.append(f'|{difficulty}|{t["n"]}|{t["psnr"]:+.8f}|{t["ssim"]:+.8f}|')
    lines+=['',f'PSNR下降{c["negative_images"]["psnr"]}/872张，SSIM下降{c["negative_images"]["ssim"]}/872张。']
lines+=['','## 预先规定的扩展门槛','',
    'B相对同预算A和冻结G6的四项聚合差值均正，才考虑更多seed；这只是资源投入筛选，不是显著性或总体效应检验。','',
    '本轮门槛：'+('通过；单seed正向线索仍须成对seed确认及独立评估。' if s['promotion_gate_passed'] else '未通过；本轮停止扩大，不启动更多seed、K3或额外预算。'),'',
    '## 审计与限制','',
    '两组200/1000步技术门槛全部通过：1000行训练记录连续、全部有限；1000个diffusion_seed一致；初始张量相同；最终Adam只含18个更新状态；全部保存点冻结审计通过。每组完整验证实际调用TaskNet1744次、R2P872次，R2P推理前后哈希一致；第一遍G6逐图复现既有872指标，TaskNet检查点仍与源G6逐元素相同。','',
    '首次诊断在服务器另一项GPU实验占用约1.5GB显存时发生OOM；保留原脚本、锁快照与失败日志，不将并发占用判为唯一原因。修订仅释放已测张量、把观测缓存移至CPU和改善显存分配，不修改精度、模型、输入或alpha。训练G6源文件显式哈希凭据在预检开始后补入，文件仍匹配既有test锁；没有配置或权重更改。','',
    '首次汇总脚本有括号语法错误，在读取或计算结果前停止；保留analyze_attempt1.py和analysis_attempt1.log，仅修复汇总脚本并重新汇总，没有新增训练或前向。实验代码、配置和训练锁未改变。','',
    '本轮只有一个训练seed，不能估计训练变异，不报告CI、显著、等价或噪声内。1000步预算只回答本次试验，不能断言最终收敛、架构错误或所有任务状态条件无效。预测条件的分布与clamp属于本轮表示干预，不拆成独立机制。', '',
    '872 validation以前已用于探索，既有test结果也已查看并参与研究问题形成。本轮没有新增test评估或用新test分数选择alpha/检查点，但这并不恢复历史数据的独立性；本轮不构成独立确认。相关方法只能作为设计依据，不能当本模块有效性证据：[PIDSR §3.3](https://arxiv.org/html/2504.07758)，[PolarAPP](https://arxiv.org/html/2603.23071)；选择与评估分离：[Cawley & Talbot 2010](https://jmlr.org/beta/papers/v11/cawley10a.html)。','',
    '逐图结果和完整差值见analysis/condition_probe_summary.json及diagnostics/diagnostic_result.json；协议、代码、原配置、来源凭据、日志与审计在证据包，检查点保留服务器。','']
(ROOT/'analysis/STATE_CONDITION_PROBE_RESULTS_ZH_20261007.md').write_text('\n'.join(lines),encoding='utf-8')
print(ROOT/'analysis/STATE_CONDITION_PROBE_RESULTS_ZH_20261007.md')
