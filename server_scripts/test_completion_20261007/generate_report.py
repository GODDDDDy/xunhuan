import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
s=json.loads((ROOT/'analysis/test_completion_summary.json').read_text())
keys=['image_mean_psnr','scene_macro_psnr','image_mean_ssim','scene_macro_ssim']
heads=['图像PSNR','场景宏均PSNR','图像SSIM','场景宏均SSIM']
lines=['# 当前五类模型的测试集补测：完整结果','',
    '2026-10-07。15个已训练step3000检查点（五类×三个训练seed），另重算一次官方原始G0。仅推理，没有训练、优化器更新或模型选择。test188张、8场景目录，全分辨率；共同evaluation_seed2026。','',
    '## 本轮结论与必要修正','',
    '以下均是三个既定训练seed的描述性配对结果，不是显著性检验或总体效应证明。','',
    '1. 普通微调G6相对官方G0：test图像PSNR均值+1.005756 dB、SSIM+0.00889596，两项以及场景宏均均3/3 seed为正；validation图像PSNR+1.161719 dB。普通微调本身在两个划分都有较大量级平均增益，不能据此判断论文作者原训练不充分，也不作模块加法分解。test场景00/01/02的PSNR和SSIM三个seed均下降，scene01平均PSNR−1.278861 dB；整体均值提高不等于每个场景提高。','',
    '2. 前置模块加联合优化仍没有形成跨指标、跨划分的一致额外优势，但必须保留test的SSIM正值。有对齐K1−G6：test图像PSNR−0.022738 dB（1/3正）、SSIM+0.00110476（3/3正），场景PSNR3/3负、场景SSIM3/3正；validation四项均值均负。有对齐K2−G6的test两种SSIM也3/3正。因此不能把历史validation的四项负值扩展成所有划分都无正向结果。','',
    '3. alignment的作用随指标/划分改变。无对齐−有对齐：test图像PSNR+0.006963 dB（2/3正），但两种SSIM3/3负；validation图像SSIM则3/3正。test上不能称去掉alignment全面改善，更不能凭一个划分直接宣布应保留或删除对齐。无对齐K1−G6的test两种PSNR3/3负；它在validation的图像PSNR均值小正也没有直接延续。','',
    '4. 必须区分两个循环比较。独立训练的K2−匹配K1：test图像PSNR+0.012042 dB（2/3正），validation+0.005710 dB（3/3正）；但两划分图像SSIM均值都负。同一K2的T2−T1：test两种PSNR3/3负，图像PSNR均值−0.012366 dB、场景−0.013210 dB；validation对应均值为+0.002162/+0.009040 dB。每个test seed的PSNR退化图数为119/113/107（各188张）。两种比较不矛盾：独立K1的输出不等于K2模型自身的T1，K2训练也改变了第一轮权重。当前可说两轮训练方案与独立K1存在小正向PSNR线索，不能说推理时第二次修改在测试集平均有益；同一K2的test图像SSIM均值−0.00000469、场景SSIM+0.00001737，均2/3正，不能说第二轮四项都下降。','',
    '5. 固定G6后仅训练R2P适配器：test图像PSNR均值−0.005950 dB、场景−0.008033 dB，均3/3负；validation图像PSNR也3/3负。test图像SSIM虽有+0.00001563且3/3正，却未同时带来PSNR收益，validation两种SSIM仍3/3负。这不支持在当前初始化、目标和预算下获得统一增量；不能因此定位为某一个架构细节错误。','',
    '证据允许的总体判断：现有结果尚不足以支持当前循环/联合方案优于普通微调的统一主张，也不足以否定这些宏观思想。第二轮测试PSNR退化、SSIM的划分依赖和局部场景退化是应保留的具体问题；本轮仅补测，不依据test启动或选择新结构。','',
    '## 本轮模型的含义','',
    'G0是未微调的官方PolarFree。G6直接接收原始RGB与偏振输入，只微调net_g。K1有对齐/无对齐均先做一次R2P偏振修改，再做一次TaskNet去反射；两者联合更新R2P和net_g，对照变量是alignment损失。当前K2有对齐为共享权重的两轮处理，关闭显式TaskNet特征反馈；它不等于早期feedback组。固定G6适配器冻结同seed已训练G6，再从零残差输出头训练R2P。','',
    '当前匹配系列采用task phase系数0.1、固定3000步；K2两轮任务损失权重为[1/3, 2/3]。早期phase=1、不同两轮权重或显式feedback的结果不混入本次三seed统计。固定G6适配器另有3000步，上游与下游分阶段训练，不与其他模型作等总预算排名。','',
    '## 测试集全部原值','', '|模型 / seed|'+'|'.join(heads)+'|','|---|'+'|'.join(['---:']*4)+'|']
for name in ('G0','G6','K1_aligned','K1_no_alignment','K2_T1','K2_aligned','frozen_G6_adapter'):
    for seed in ([2026] if name=='G0' else s['seeds']):
        t=s['tables'][f'test_{name}_seed{seed}']
        lines.append(f'|{name} / {seed if name!="G0" else "官方固定权重"}|'+'|'.join(f'{t[k]:.8f}' for k in keys)+'|')
lines+=['','## 测试集与验证集的配对差值均值','',
    '测试集是本轮实际重新计算，验证集复用已锁定原结果。两个划分的场景目录不同，不能按同号对齐；此处只比较各自合法汇总方向。','',
    '|比较|指标|test均值|test样本SD|test正向seed数|validation均值|均值是否换号|','|---|---|---:|---:|---:|---:|---|']
for n,c in s['comparisons'].items():
    for k,h in zip(keys,heads):
        t,v=c['test']['paired_seed_summary'][k],c['validation']['paired_seed_summary'][k]
        lines.append(f'|{n}|{h}|{t["mean"]:+.8f}|{t["sample_sd"]:.8f}|{t["positive_seeds"]}/3|{v["mean"]:+.8f}|{"是" if c["cross_split_mean_sign_changed"][k] else "否"}|')
for n,c in s['comparisons'].items():
    lines+=['',f'## test：{n}','', '|指标|seed2026|seed2027|seed2028|','|---|---:|---:|---:|']
    for k,h in zip(keys,heads):lines.append('|'+h+'|'+'|'.join(f'{v:+.8f}' for v in c['test']['paired_seed_summary'][k]['values'])+'|')
    lines+=['','逐场景配对差值：','','|test场景目录|PSNR均值±样本SD|SSIM均值±样本SD|','|---|---:|---:|']
    for scene,t in c['test']['scene_seed_summary'].items():
        lines.append(f'|{scene}|{t["psnr"]["mean"]:+.8f} ± {t["psnr"]["sample_sd"]:.8f}|{t["ssim"]["mean"]:+.8f} ± {t["ssim"]["sample_sd"]:.8f}|')
    lines+=['','|seed|PSNR退化图数|SSIM退化图数|','|---|---:|---:|']
    for seed,t in c['test']['per_seed'].items():lines.append(f'|{seed}|{t["negative_images"]["psnr"]}|{t["negative_images"]["ssim"]}|')
lines+=['','## 实施审计与时间','',
    '全部16个推理单元通过：每个模型参数与缓冲区前后哈希一致，无参数梯度、无优化器；测试sample_id顺序一致。K2实际每图2次TaskNet，其余每图1次；G0/G6不调用R2P。检查点与原验证指标哈希及948个实际测试输入/GT文件哈希核验通过。原训练和评估产物未覆盖。','',
    '官方G0按当前流程重新计算后，与历史服务器G0的188张逐图PSNR/SSIM最大差值均为0.0。当前G0图像PSNR为22.4226725951 dB，SSIM为0.8684772957；此处不混用更早的本地留存基线。历史指标文件和哈希均在证据包中。','',
    '|单元|TaskNet调用数|耗时（分钟）|检查点SHA256|','|---|---:|---:|---|']
for uid,a in s['audits'].items():
    lines.append(f'|{uid}|{a["actual_forward_calls"]["tasknet"]}|{s["evaluation_seconds"][uid]/60:.2f}|{s["source_fingerprints"][uid]["checkpoint_sha256"] or "官方原始权重；见weight_audit"}|')
limits_zh=[
    'test此前已经查看过；本轮是跨划分描述性复核，不是全新独立确认。',
    '不依据test选择seed或检查点；预先列定的15个检查点全部报告。',
    '场景目录编号只在划分内有效，不按编号把validation/test当同一物理场景。',
    '三个配对训练seed、固定推理随机流；没有估计推理采样变异。',
    '固定G6适配器有额外训练预算和不同初始化，不作等总预算排名。',
    'PSNR/SSIM、图像均值/场景宏均同时保留；没有证明等价、显著或差值在噪声内。',
    '原评估随机函数对两个划分均使用validation阶段字面量；test样本ID仍进入随机键。']
lines+=['','## 解释边界','']+['- '+v for v in limits_zh]+['',
    'G0只有一个固定官方模型，表中复用为每个训练seed的共同参照，不代表训练了三次G0。三个seed样本SD仅描述训练实例差异，不是CI、显著性或等价性检验。','',
    'K2有对齐−K1有对齐才是当前匹配循环比较；K2−无对齐K1不能当循环净贡献。固定G6适配器增加了3000步和不同初始化，单列其条件性增量，不作等总预算排名。','',
    '测试集早已查看过，结果是另一划分上的复核；不得重新称为未见独立留出。模型选择偏差依据：[Cawley & Talbot, JMLR 2010](https://www.jmlr.org/beta/papers/v11/cawley10a.html)。','',
    '全部逐图原指标与差值见analysis/test_completion_summary.json；配置、来源、协议和日志在证据包；检查点仍保留原服务器位置。','']
path=ROOT/'analysis/TEST_COMPLETION_RESULTS_ZH_20261007.md';path.write_text('\n'.join(lines),encoding='utf-8');print(path)
