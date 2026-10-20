# G6 phase=0.1 对照：执行协议

本轮依据用户“执行”的授权，补齐 2026-10-02 K2 损失权重试验建议的普通微调对照。协议在训练启动前锁定；不根据中途指标选择检查点。结果写入独立报告。

## 要回答的问题

把 Fourier phase 损失权重由 1 降为 0.1 后，不经过 R2P、只微调 net_g 是否也会改善？在相同新损失下，现有 K2 整体方案与这种普通微调相差多少？

第二项比较同时包含输入路径、R2P 更新、两轮监督、对齐损失等差异，不能据此单独证明循环、R2P 或对齐的因果贡献，也不能把不同实验的增益作模块加法分解。

## 固定条件

- 原 G6 训练代码按服务器原文件逐字节归档，沿用实际旁路、数据加载、梯度裁剪、优化器和评估实现。
- 唯一训练目标改动：`L1 + (1−SSIM) + 0.1 × Fourier phase MSE + 0.1 × VGG conv5_4 L1`。前一个 0.1 为本次改动，感知损失原权重不变。phase 是现有直接比较角度的 MSE 实现，并未改成圆周距离。
- 仅 net_g 可训练。原始偏振和原始混合 RGB 走官方 TaskNet 输入处理；不调用 R2P，不使用 alignment，扩散先验冻结。TaskNet 在训练中保持原 G6 的 eval 模式。
- seed=2026，官方 net_g 起点；与新 K2 phase0.1 的初始 net_g 参数逐元素核对。
- 原 manifest、完整冻结 schedule、crop=128、batch=2、workers=2；Adam，net_g LR=1e-5，betas=(0.9,0.999)，weight_decay=0；net_g 梯度裁剪阈值 1。
- 3000 步固定预算；先训练到 200 步完成有限性/优化器/冻结审计，再原样恢复优化器继续到 3000。原代理验证时点和每 200 步保存规则保留。
- 保留同平台 TF32 等精度设置，使用此前已验证的 `expandable_segments:True` 显存分配设置。
- 最终仅评估 step3000 的完整 872 validation；不新增 test 评估。评价仍属探索，不能当作独立确认。

## 审计与报告

启动前做原训练器的梯度断言和 R2P NaN 旁路探针；以 3 张 validation 图核对新旧 G6 step0 输出。有限性检查、Adam 实际步数、连续 1–3000 日志和每个保存点冻结审计均保留。冻结 prior 按参数/缓冲区身份排除内部被训练的 net_g，避免重现历史误判。

指标来自原评估器。额外损失分解复用同一次 TaskNet 输出，检查附加运算不改变 CPU/全部 CUDA RNG。报告同时给出图像均值、场景宏均、逐场景结果以及负向案例；不凭未估计的随机变异宣称等价或“噪声内”。

比较对象：新 G6 phase0.1、历史 G6 phase1、新 K2 phase0.1 的 T1/T2；必要时并列既有 K2 phase1，不重复训练。跨方案同样 3000 次优化器更新不等于同样计算量；K2 两轮任务监督权重为 [0.5,1]，G6 单轮权重为 1，未人为重标定以构造另一个训练目标。

## 路径

- 新服务器实验根：`/media/li/data3/25thstudent/DY/929work/k1_ablation_20260930/phase_objective_g6_20261002/`
- 新运行输出：该根下 `runs/G6_phase01/seed_2026/`
- 脚本及原训练器归档：该根下 `scripts/`；训练启动前配置/脚本摘要在 `protocol_lock.json`。
- 本地脚本：`F:\WORK\projects\pidsr_joint_alignment\code\phase_objective_g6_20261002\`
- 本地证据下载：`F:\WORK\projects\pidsr_joint_alignment\_dl\phase_objective_g6_20261002\`
- 本地报告：`F:\WORK\projects\pidsr_joint_alignment\analysis\`

历史 G6、K2 检查点、指标与代码均保留原状。任何实现修正需记录变更范围和是否发生优化器更新，不能覆盖失败记录。
