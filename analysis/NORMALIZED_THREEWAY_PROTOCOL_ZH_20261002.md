# phase0.1：G6/K1/K2 成对三 seed 协议

依据用户“那么去做”的授权，继续补齐单轮联合对照和成对 seed 验证。当前 GPU 为独占空闲的 RTX3080Ti；全部新增产物位于独立实验目录，旧模型、指标和原代码保留。

## 问题与三组

- G6：原始偏振与原始混合 RGB 直接输入官方 TaskNet，只微调 net_g，冻结扩散先验；不调用 R2P，不使用 alignment。
- K1：预训练 R2P 单次处理偏振，再输入 TaskNet；联合更新 R2P、对齐投影与 net_g。沿用 K2 的同一模型构造、优化器分组、激活检查点和联合裁剪方式，不沿用历史 G5 分开裁剪 R2P/投影的不同训练路径。
- K2：同一套 R2P/TaskNet 共享权重处理两轮，不开任务反馈。第二轮偏振输入来自第一轮，保留完整跨轮梯度；两轮共享固定 diffusion RNG 流。

各组 seed=2026、2027、2028，固定 3000 步。已有 G6 phase0.1 seed2026 可复用；新增 K1/K2 seed2026 及三组 seed2027/2028，共 8 次训练/完整验证。启动前锁定组别、顺序、配置与代码哈希，不按某个 seed 的结果筛选后续组别。

## 必须控制的权重差异

此前 K2 的任务损失为 0.5L(T1)+L(T2)，总权重 1.5；G6/K1 原单轮为 L(T)。本轮统一总权重 1：G6/K1 为 L(T)，K2 为 (1/3)L(T1)+(2/3)L(T2)，保留原两轮监督的 1:2 比例。因此新 K2 是**损失归一化的对照版本**，并非复现此前 [0.5,1] 的 K2；此前结果保留，只作 seed2026 的历史敏感性参照，不能混入新组三 seed 平均。

这项控制依据实际训练代码：梯度先由任务/对齐损失共同产生，再按学习率组裁剪。权重改变可能改变总梯度范数、相对 alignment 强度和裁剪触发，因此相同步数与学习率不足以控制它。PyTorch 对裁剪的定义见 [clip_grad_norm_ 官方文档](https://docs.pytorch.org/docs/main/generated/torch.nn.utils.clip_grad_norm_.html)。这是基于代码的控制变量推论，不声称已证明原循环退化由权重造成。

任务损失继续使用现有实现：L1+(1−SSIM)+0.1×Fourier phase MSE+0.1×VGG conv5_4 L1。未改 phase 的角度距离定义。K1/K2 共用固定 alignment lambda=0.030651881051471687，不按新损失、组别或 seed 重新标定。

## 配对与执行

官方 net_g 与 R2P e42 起点固定；K1/K2 每个 seed 的全部初始模型张量逐一核对，包括随机对齐投影。扩散先验与未使用反馈适配器保持冻结/不更新。

同一 seed 三组共用一个 manifest 和固定 sample/crop schedule；2026 复用此前 schedule，2027/2028 通过现有 `build_training_schedule` 在服务器从当前 manifest 生成，并与本地已有 schedule 对照，保留两者与真实哈希。不同 seed 同时改变取图/裁剪、对齐初始化及 diffusion RNG；不估计这几种随机性的单独贡献。

最终872验证统一使用 evaluation_seed=2026 的逐图 diffusion 随机流，与复用的 G6 seed2026 完全匹配，避免将推理采样变化混入训练 seed 变异。训练中的代理验证继续沿用各训练 seed 的原实现，不据其选模型。本条在第一轮完整最终评估启动前锁定；不改变训练配置或损失。

batch2、crop128、workers2；R2P/投影 LR=1e-4，net_g LR=1e-5，Adam betas=(0.9,0.999)、weight_decay=0；模型参数组与 net_g 分别做 norm clip=1，TaskNet 在训练中保持 eval。相同步数并不等于相同 FLOPs：K2 每步有两轮，必须记录实际调用次数及耗时。

各次执行 preflight→200 步审计→恢复原优化器到 3000→完整872验证。检查连续有限日志、Adam 状态步数、初始化匹配及每保存点冻结状态；任一步失败即停止队列，保留失败日志，不跳过失败组进入最终均值。

K1 的实际 R2P/TaskNet 调用各一次。复用评估器中的旧 round2 字段仅为单轮输出别名，原始继承结果另存并明确标注；最终 K1 结果移除虚假的第二轮/循环增益字段。K2 各两次，G6 的 R2P 为零次，TaskNet 一次。

继承的检查点 `method` 字符串可能仍来自 K2 原训练器；本轮以新增的 `matched_arm`、`actual_rounds`、`effective_task_weights` 与实际调用核验判定组别，不把继承字符串当作新的方法证据。

## 分析边界

只评估固定 step3000 的完整872 validation，无新增 test。逐 seed 报告图像均值、场景宏均、逐场景/逐图配对差值、easy/hard 与负面案例。以训练 seed 为重复单位汇总成对差值均值、样本标准差与三个原值；不把872张相关图像当作872次独立训练，不凭三 seed 宣称等价或充分验证。

参考：[Bouthillier et al., Accounting for Variance in Machine Learning Benchmarks, MLSys 2021](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf)，讨论数据抽样、初始化等变异来源。论文不为“3 seed 足够”提供保证，本轮只做初步变异估计。

K1−G6 是当前单轮联合整套方案的增量，包含偏振输入修改、R2P 更新与 alignment，不能进一步拆为单模块贡献。K2−K1 是固定总监督权重、同更新预算下的两轮方案比较，不代表同计算预算或同监督位置。T2−T1 是同一 K2 检查点内的推理变化，不能替代 K2−K1 训练方案对照。

验证集和 seed2026 已参与此前探索。新增 seed 能补训练变异信息，不能把同一验证集变成独立留出。未启动 K3、feedback 新训练、扩大步数或修改结构。

## 产物

服务器根：`/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/normalized_threeway_20261002/`，含 `protocol_lock.json`、`configs/`、`scripts/`、`runs/{G6,K1,K2}/seed_{2026,2027,2028}/`、队列与逐阶段日志。

本地脚本：`F:\WORK\projects\pidsr_recurrent_joint_k2\server_scripts\normalized_threeway_20261002\`；下载：对应项目 `_dl\normalized_threeway_20261002\`；报告：对应项目 `analysis\`。所有新增本地文件均放在 F 盘。
