# 阶段A 诊断协议（gain_objective_audit_20261013_120028）

版本 1.0；2026-10-13 北京时间。依据交接文档《GAIN_JUDGEMENT_DIAGNOSIS_AND_INTERACTION_HANDOFF_ZH_20261013》的阶段A。

## 0. 任务与边界

在不削弱第一轮、不修改第二轮候选、不动历史实验的前提下，检查"收益判断为什么学不好"，并以最低必要计算量区分实现/监督/学习/泛化问题。**本阶段只做诊断**，不启动正式的单因素对照（B）或模块协同（C）；B/C 需另外的用户授权。

- `execution_scope = diagnostic_only`；`formal_training_enabled = false`；`factorial_training_enabled = false`
- 不读 188 测试；不重跑 872 来源诊断；不新建完整恢复模型。
- GT 只进入训练监督、训练内校准与评价标签，不进入可部署输入或候选生成输入。
- 历史实验目录只读；本机新增文件放 `F:/WORK` 分类目录；新运行不沿用任何旧预算/进程组。

## 1. 起点与来源

- 第22项最终联合权重：`downstream/checkpoints/final.pt`，SHA256 `279d0a32…e9a2`（第22项协议ID `8d4f7dd5…6ad2`）。
- 第21项来源权重（A2/B 共同初始化默认）：`weights/reused_final.pt`，SHA256 `cecdcfa4…916d`。
- 既有诊断缓存：`diagnostics/feature_cache/{judgement,bias,safety}.npz`（尺寸/SHA 见 A0 收据）。
- 数据角色（沿用 `locked_run.json`）：judgement 12 场景1140图；bias 4 场景237图；safety 4 场景423图；candidate 31 场景3640图（本轮不更新候选）。
- 标签尺度：缓存生成来源 `cost_scale = 9.34407114982605e-4`（已核验，见 A0）。

## 2. 统一数学定义

`actions=[0,0.25,0.5,1]`；`T(a)=T1+a·δ`，`δ=U−T1`；成本 `= MSE(T(a))−MSE(T1)`（原始浮点RGB，官方口径），成本<0 表示改善。零动作成本定义为 0。`T2=where(M==0,T1,T1+M·δ)`，`M` 按 4×4 区域最近邻扩展。

损失：
```
y = true_cost(nonzero)/s
L_fit = SmoothL1(predicted, stop_grad(y))
p = softmax(-concat(0, predicted))
L_policy = mean(sum(p * stop_grad(true_cost(all)/s)))
L_mixed = L_fit + 0.1*L_policy
L_pure  = L_fit
```
本阶段唯一要检验的单因素：**去掉 0.1 期望项**是否改善拟合/排序/实际选择。特征标准化、动作尺度、输出结构、损失权重、初始化、门槛规则**均不同时更改**。

## 3. A0 来源/缓存/数值复核（CPU）

按 §6 顺序执行，已通过。产物：`metrics/A0_numeric_audit.json`、`docs/A0_findings.md`、`receipts/source_and_cache_receipt.json`、`figures/input_scale.svg`。关键：尺度核验、通道量级、标签分布、二次结构、`L_fit` 与 `L_mixed` 在精确标签处的梯度。

## 4. A1 已有缓存上的轻量学习性诊断（CPU）

固定设置（启动前写入 `config/diagnostic_lock.json`）：

- 输入 `x` 的 50 通道；标准化统计**仅来自判断训练角色**。
- tiny-fit 采样 **2048** 个判断训练区域，按场景均衡，固定 seed 2026 并记录索引。
- MLP `50→32→32→3`，GELU；同一份初始参数复制给 mixed/pure 两副本。
- Adam `lr=1e-3`，`batch=256`，**1000 步固定终点**，每 100 步记录纯拟合误差与三动作排序。
- 两副本完全相同样本/批次顺序，只有 loss 不同。
- 报告 `L_fit`、原始单位成本 MAE（尺度核验后）、三动作 Spearman、简单基线（全零/均值/最佳常数动作）、全拒绝/全接受参照。
- 训练内风险覆盖只看预定 1%/5%/10%/20% 面积；随机参照匹配所选动作比例并使用正确面积权重。
- 若前 10% 实际净改善>0、好于同覆盖同动作随机期望、且 4 个安全场景至少 3 个改善 → 记 `signal_present`（**不是**部署安全证明）。
- 输出标记 `probe_only=true`、`restoration_method_score=false`。
- **结论限制**：两 MLP 拟合差只说明该探针/预算/表示没拟合好，**不能**直接判定原生模块坏；MLP 拟合好也不能证明空间卷积原生头已解决。

## 5. A2 原生卷积判断头小样本匹配诊断（GPU）

1. 按已核验第22项架构构建 **50 输入通道**原生模型，严格加载第22项最终权重；不得扩成 57 通道，也不得用 43 通道结构去严格加载 50 通道权重。
2. 创建两份完全相同的 `action_gate` 副本，分别训练 `L_mixed` 与 `L_pure`；开始前两头张量严格相等。
3. 所有非 `action_gate` 张量与整个任务网保持来源严格相等；`eval` 与 `requires_grad=False` 都设置。
4. `action_gate.feature_projection` 属可训练判断器；其他任务特征/状态/H/Q/反馈/候选/先验不更新。
5. 不加入标准化层、不换输出参数化、不改变 50 通道含义、不重初始化输出层。**唯一训练因素为 loss。**
6. 最小空间缓存（用原 scheduled_loader 的数据/裁剪逻辑；选 sample_id 与裁剪坐标前不读 GT/结果）：
   - judgement 12 场景×2 裁剪=**24**；bias 4×2=**8**；safety 4×4=**16**。
   - 一次冻结两轮前向提取：T1/U/δ、b、C、judgement_state、投影前原始 `judgement_feature`、4×4 池化、三非零真实成本与共同尺度归一化标签、4×4 增量平方/绝对均值/抵消比、完整 32×32 邻接。
   - 主干只跑一次；优化判断头时每次重新通过其可训练 `feature_projection`（**不得**缓存投影后的 8 通道）。
7. 缓存等价检查（固定 5 裁剪）：初始头三成本输出一致；同一 loss 数值一致；同一优化步 projection 与 body 梯度一致；更新一次后头参数一致；非判断张量严格相等；GT 替换改标签不改可部署特征；M0 严格复制 T1。阈值默认 `atol=1e-6, rtol=1e-4`；不得把 allclose 写成逐比特一致。不通过则停止优化、修复后重做。
8. 小样本优化：共同来源=第22项最终 `action_gate`；Adam `lr=1e-4`、有效 batch 2、微批 1、`clip_norm=1`；两头同一 24 裁剪循环顺序、相同 500 步、相同随机流与**新的空**优化器状态；固定 500 终点，不挑最佳监测点，不因一头领先提前停另一头。
9. 诊断级偏差与安全筛查：尺度 `s` 沿用第22项来源尺度，两头相同；分别在同一 8 个偏差裁剪、同一 16 个安全裁剪上跑旧门槛网格；门槛网格 `0,0.05,…,12.8`（归一化成本）；要求平均 MSE 不增且明显有害接受≤20%，取最大非零覆盖，否则全拒绝；全拒绝时 `safe=true, useful=false`，有害接受率写 NA。不得依据 A3 结果改偏差/网格/尺度/规则。

## 6. A3 最多32张原尺寸验证筛查（GPU）

- 名单：8 个 validation 场景各 4 张，共 ≤32。按 sample_id 排序取 `floor(j*(n-1)/3)`，j=0..3；场景不足 4 张取全部并去重。**名单、manifest SHA、选择规则在读取 GT 与性能之前锁定**；不用测试筛选。
- 每张图只执行**一次**冻结主干，得到共同 T1/U/原始特征与状态；source/mixed/pure 三个头依次作用于同一组特征；只构建一个完整主干 + 小判断头副本。
- 每个头使用自己训练内得到的偏差与门槛，不复制另一头的校准值。
- 复用官方 PSNR/SSIM 分别计算三头的真实拼接输出；保存每张 T1/U 浮点哈希并**按 sample_id** 与来源匹配。
- 不重复 normal/guidance_off/Q0/Q1/no_feedback/no_polar_correction 整套干预；保持原尺寸完整前向，不切块。
- 记录每头三动作成本 MAE/Spearman、T2−T1 的 PSNR/SSIM、按场景差、覆盖/幅度、误接受/误拒绝、固定示例；接受集为空时仍记录门控前成本排序。

## 7. 预算、监控与状态

- **阶段A 首次 GPU 工作前登记新的 2 小时绝对截止**（`stage_A_gpu_budget_seconds=7200`），不恢复、不重置、不延长任何旧预算。A0/A1 优先 CPU，不为用满预算而继续计算。
- 不重复启动：每次 GPU 工作前核对 status/预算/watchdog 与真实进程（命令行、工作目录、进程组）。
- 状态序列：`planned → source_verified → cached_numeric_audit_complete → lightweight_probe_complete → native_head_cache_verified → native_head_diagnostic_complete → original_size_screen_complete → diagnostic_complete`。
- 完成状态用 `diagnostic_complete`，**不**使用 `formal_experiment_complete`。

## 8. 交付物

见 §15.1：`docs/diagnostic_protocol.md`、`config/diagnostic_lock.json`、`receipts/source_and_cache_receipt.json`、`receipts/online_cache_equivalence.json`、`metrics/A0_numeric_audit.json`、`metrics/A1_probe_results.json`、`metrics/A2_native_fit_results.json`、`metrics/A3_screen_per_image.csv`、`metrics/A3_screen_by_scene.csv`、`logs/stage_timing.csv`、`checkpoints/diagnostic_heads/*`、`figures/input_scale.*`、`figures/train_fit_and_cost_ranking.*`、`figures/risk_coverage.*`、`figures/fixed_image_example.*`、`docs/RESULTS_ZH.md`、`archive_manifest.json`。未完成项明确标缺，不用空结果冒充完成。

## 9. A1 执行记录与结果（已完成）

- 状态：`lightweight_probe_complete`；CPU；未用 GPU；未读 188 测试；未重跑 872。
- 共同初始权重 SHA256 `c84ea6f5d9fbc4b19fae3f22899f2f0c4d27825a8154168dfc52e44649ae7c65`；样本索引 SHA256 `fae6753b60a2f8b55604781786f72a47780c2193eb8a6dde28701d225054e10c`。
- 损失等价性：torch（与 `action_gate.cost_loss` 同定义）对比独立 numpy 解析实现，最大绝对差 `1.96e-10`。核对时修正了 numpy 参照漏写负号（应为 `softmax(-costs)`）的错误。
- 训练耗时 1.52 秒（CPU）。
- 结果摘要：
  - 训练内（2048）：L_fit 0.09892 → mixed 0.02679 / pure 0.02685；成本 MAE 0.262 → 0.1085(mixed)/0.1049(pure)；训练内三动作平均 Spearman 升到约 0.42/0.41。
  - 跨角色：判断训练全集 MAE 0.2588/0.2564（Spearman≈0.21）、偏差 0.1742/0.1580（≈0.10）、安全 0.2432/0.2396（≈0.06）。全集/偏差/安全上 mixed 均未超过"最优常数动作"基线 0.1059。
  - 安全角色强制 1%–20% 覆盖：接受区平均真实净改善全为负，10% 处 mixed −2.56e-4 / pure −3.22e-4，**劣于**同动作随机参照 −5.08e-5；有害接受率约 42%。
  - `signal_present_at_10pct`：mixed=false，pure=false（4 场景仅 1 个为正）。
- 判定：散点表示在训练内可拟合（过拟合于 2048 抽样），但不跨角色/场景迁移；**去掉期望项没有实质改善**（L_fit 几乎不变、MAE 略优、排序略降）。
- 限制：本结果只说明该探针/预算/散点表示未形成可迁移信号，**不能**据此判定原生判断头坏掉；原生头判定见 A2。
- 详见 `docs/A1_findings.md`、`metrics/A1_probe_results.json`、`receipts/A1_artifacts.json`。

## 10. A2 执行记录与结果（已完成）

- 状态：`native_head_diagnostic_complete`；GPU；未读 188 测试；未用 validation 选择任何规则；未重跑 872 诊断。
- 预算：A2 首次 GPU 前登记的新 2 小时绝对预算（`budget.json`），未沿用旧实验预算。
- 判断头：生产构造路径的 50 输入 / 3 输出原生头（`feature_projection` 96→8 可训练），
  两份副本共享初始权重 SHA256 `6323d52851ad9e67c2020463eab5d3e859e9964758dfbc311f66e69ec6e200b0`（来源 `final.pt` `279d0a32…e9a2`）。
- 缓存：48 裁剪（judgement 24 / bias 8 / safety 16），一次冻结双轮前向；`feature_projection` 不缓存。
- **缓存等价：通过，且为逐比特相等**（成本输出、损失、梯度、一次更新后的参数 max|Δ| 全为 0），
  非判断张量仍严格等于来源，GT 替换只改标签，M0 精确复制 T1。
- 训练：Adam lr 1e-4、有效 batch 2、微批 1、clip_norm 1、固定 500 步；唯一变量 = 损失。
- 结果摘要：
  - 训练内 L_fit 0.139464 → mixed 0.127693 / pure 0.127553（−8.4% / −8.5%）；
    三动作成本 MAE 2.114e-4 / 2.113e-4，优于零基线 2.247e-4 与训练均值基线 2.247e-4；
    训练内 Spearman ≈ 0.24–0.27。
  - mixed 与 pure 在所有指标上同级（终点 L_fit 相对差 0.11%，**pure 略好**）→ **期望项没有带来可测收益**。
  - 跨场景角色（bias / safety）MAE 9.69e-5 / 8.58e-5，**均劣于零基线** 8.17e-5 / 8.39e-5 → 不迁移；
    幅值系统性低估（safety 上 a=1.0 预测 0.0409 vs 真实 0.0970）。
  - 旧门槛网格 0…12.8：无任何合格的非零覆盖 → **全部拒绝**（coverage 0，safe=true，useful=false，有害率 NA）；
    GT 参照（每格最优动作上界）仅 +4.575e-6（≈ 尺度的 0.49%）。
- 判定：原生头在固定小样本上**能收敛式拟合**，但**不跨场景迁移**，且**去掉期望项不改善**；
  未找到符合预定条件的非零修改。可进入 B 的信号（pure 优于 mixed / 安全角色净改善为正）**未出现**。
- 执行偏差：7 次进程尝试中 6 次失败，全部是新写脚本的执行缺陷（详见 `docs/A2_findings.md` §9 与
  `receipts/A2_execution.json`），**未改动算法/目标/数据/协议**，协议 ID 不变。
- 详见 `docs/A2_findings.md`、`metrics/A2_native_fit_results.json`、`receipts/A2_artifacts.json`。
