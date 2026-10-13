# 同目标 K1 无 alignment：三训练种子配对消融协议

日期：2026-10-03。用户授权执行上一轮建议中的首项关键拆分。新训练仅K1 λ=0，seed2026/2027/2028；复用已完成的同seed K1 λ=0.030651881051471687与G6为对照。

## 研究问题与唯一训练干预

当前单轮联合方案没有相对G6的稳定优势，但其中同时包含R2P更新和alignment。先检验关闭当前alignment监督后是否改变任务表现；不能把结果预先解释成R2P结构或所有对齐方法的结论。

每个seed复用原K1完整初始模型、官方net_g、R2P e42、manifest、sample/crop schedule、训练逐步diffusion_seed。配置逐字段比较只允许输出目录、λ、实验范围说明和初始化审计引用路径变化。唯一改变训练目标的字段为λ：0.030651881051471687→0。

保留同一R2P、投影、单次TaskNet调用、激活重计算、优化器参数组与裁剪范围；保留alignment计算图，以 `task_loss + 0*alignment_loss` 反向传播。因此是**关闭对齐监督**，不删除投影、不冻结其他模块、不额外改优化器。weight_decay=0；投影梯度必须恰为0、投影张量在每个保存点完全不变，Adam仍可保留投影的零状态。alignment标量仅作诊断，不能当成正在优化的目标。

初步前向检查还比较纯task_loss与加入0*alignment后的R2P梯度（最大差容差1e-7），检查R2P/net_g有限非零梯度、prior冻结、反馈未使用，以及同seed全部初始模型/net_g哈希与旧K1相同。

## 固定训练、评估与审计

- 单轮任务权重1；损失为L1+(1−SSIM)+0.1×Fourier phase MSE+0.1×VGG L1。
- 固定3000步，batch2、crop128、workers2；R2P/投影LR1e-4、net_g LR1e-5，Adam(0.9,0.999)、wd0；模型组/net_g分别clip1。TaskNet eval模式，冻结扩散先验但保留向输入的梯度。
- 顺序2026→2027→2028，每组preflight→200步审计→原优化器恢复至3000→完整872 validation。任一失败停止，不按结果跳过seed。
- 固定step3000，不选best.pt；最终逐图推理随机流evaluation_seed=2026，与旧K1/G6相同；实际TaskNet调用872次。K1继承的round2是别名，正式结果清除循环字段。
- 核验连续有限的3000行日志，457个Adam状态步数，冻结prior/反馈状态，投影零更新，以及最终检查点、指标文件与所有源代码/配置哈希。

## 分析

先计算每seed `K1无对齐 − K1有对齐`，再计算 `K1无对齐 − G6`。并列报告图像均值/场景宏均的PSNR、SSIM、三个配对原值、均值及样本SD；保留逐场景/逐图、easy/hard和退化样本。三个seed不构成等价检验或独立数据验证。

若关闭当前alignment后改善，只支持其在当前配置下可能不利；不证明PolarAPP的元学习对齐无效。若未改善，只说明移除该目标没有解决当前问题，不证明R2P架构无用，也不定位唯一机制。

参考：[PolarAPP §3.3](https://arxiv.org/html/2603.23071v2)采用元学习的对齐度量，与本项目普通投影/余弦目标不同；[Bouthillier et al., MLSys2021](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf)讨论训练随机性。本轮配对规范基于此前实测结果，不能据3seed声称充分验证。

## 产物与范围

服务器：`/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/k1_no_alignment_20261003/`，协议锁在新训练启动前生成。旧研究目录与产物不修改。

本地：`F:\WORK\projects\pidsr_recurrent_joint_k2\server_scripts\k1_no_alignment_20261003\`；下载归档在对应项目 `_dl\k1_no_alignment_20261003\`；报告在 `analysis\`。

不启动冻结G6适配器、重设计R2P、第二轮纠错器、feedback、K3或新test；这些后续步骤依赖本次结果另行决定。
