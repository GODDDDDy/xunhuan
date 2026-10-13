# 修订报告：alpha 端点检查（修正定义并完成 alpha=0 / alpha=1）

日期：2026-09-30
模型：seed2026 / step3000，**冻结**（未重训、未追加预算、未重跑 872 张）
子集：沿用现有固定 24 张，**同序**
产物根目录（服务器）：`/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/evaluation_872/endpoint_check/`
本地副本：`F:\WORK\projects\pidsr_recurrent_joint_k2\server_scripts\endpoint_check\`

---

## 一、修正后的 alpha 定义

**此前我用的定义是错的。** 正确的定义（本次已按此执行）：

```
P2(alpha) = P1 + alpha * (P2 - P1)
T(alpha)  = TaskNet( P2(alpha), original_RGB ; paired_RNG )
```

要点：

1. **P2 必须是原配置下完成第二轮 R2P 的输出**。feedback 组的 P2 使用 T1 产生的反馈特征；no_feedback 组保持原配置（feedback 关闭）。
2. **插值后直接送 TaskNet，不得再次通过 R2P。**
3. **关闭 feedback 不等于 alpha=0。** 关闭 feedback 后 R2P 仍会做基础残差更新（因此 `P2off ≠ P1`），这与"直接把 P1 送 TaskNet"是两件不同的事。

### 撤回的表述

- 撤回"原先 alpha=0 应复现 T1 的预期不成立"。
- 撤回"该一致性检验无法构造"。

原预期**是成立的**，且本次已在 24/24 张上**通过**（见第三节）。此前判断不成立的唯一原因，是我把 alpha=0 错误地实现成了"反馈关闭"，而那条路径不会取消 R2P 的残差更新。

### 保留但更名

现有 `alpha0_selfcheck.py` 的检查**保留**，改称 **"同一检查点的反馈关闭消融"**。它回答的是"关掉 feedback 分支时 P2/T2 如何变化"，**不是** alpha=0 幅度自检。其原有数值（`max|P2off−P1|`、`max|T2off−T1|`）只作为该消融的结果记录。

---

## 二、端点检查的执行方式（可核查）

脚本：`evaluation_872/endpoint_check.py`
日志：`evaluation_872/endpoint_check/endpoint_check.log`（`rc=0`）
逐图 JSON：`evaluation_872/endpoint_check/endpoint_check_result.json`
运行耗时：293.3 s

每个样本按**正式评估流程**先算出参考量：

1. `P1 = model(original_polar, condition)`，`T1 = TaskNet(P1, original_RGB)`；
2. `P2 = model(P1, condition, feedback=f1)`（feedback 组用 T1 产生的 `f1`；no_feedback 组 `feedback=None`），`T2 = TaskNet(P2, original_RGB)`；
3. **alpha=0**：`TaskNet(P1, original_RGB)` → 与 T1 比；
4. **alpha=1**：`TaskNet(P2, original_RGB)` → 与 T2 比。

**RNG 纪律**：`P1` 第一次 TaskNet 调用**之前**保存 CPU 与**全部 CUDA** 的随机状态；此后每一次对照调用前都用 `torch.random.fork_rng` 恢复该状态。参考 T1/T2 调用与端点调用因此看到**同一条扩散噪声流**。

**参考完整性**：TaskNet 权重、eval 模式、original RGB 及其他输入均不变；`P1`/`P2`/`T1`/`T2` 均以 `.detach().clone()` 保存，未被原地修改；端点输入亦用 clone。

**通过判定由实测差值计算**（`passed = (input_diff == 0) and (output_max_diff < tol)`），**不是硬写 True**。容差：端点输入必须**恰好为 0**（与参考张量完全一致），输出 `1e-4`。

---

## 三、结果：两个端点全部通过

| 组 | 端点 | 端点输入 | 输入最大差 | 输出最大差 | 输出平均差 | 通过 |
|---|---|---|---:|---:|---:|---:|
| no_feedback | alpha=0 | P1 | 0.000e+00 | **0.000e+00** | 0.000e+00 | **24/24** |
| no_feedback | alpha=1 | P2 | 0.000e+00 | **0.000e+00** | 0.000e+00 | **24/24** |
| feedback | alpha=0 | P1 | 0.000e+00 | **0.000e+00** | 0.000e+00 | **24/24** |
| feedback | alpha=1 | P2 | 0.000e+00 | **0.000e+00** | 0.000e+00 | **24/24** |

**可核查事实**：

- 端点输入与对应参考张量的最大绝对差为 **0**（`input_all_exact = True`）——插值在 alpha∈{0,1} 时给出与参考**逐元素相同**的输入。
- 端点在恢复同一 RNG 状态后的 TaskNet 输出，与参考 T1/T2 的最大绝对差为 **0，平均绝对差也为 0**。

**这说明（限定范围）**：在该 24 张子集、固定随机流、同一 TaskNet 权重下，TaskNet 对同一输入是**严格确定性的**；第二轮 R2P 的输出被直接送回 TaskNet 时能**逐元素复现**原 T2。

**这不能说明**：第二轮退化不是实现缺陷以外的结论。它只排除了"TaskNet 在同输入下产生不同输出"这一种可能；**不排除** R2P 路径、损失设计、训练预算、共享权重等其他候选环节。也不构成对 872 张全集的推断。

---

## 四、对上一版报告证据范围的修正

按复核意见逐条收紧：

1. **`det_test.py` 不能证明"整个实现无 bug"。** 它只读取验证集第一张图，只调用 `model(...).refined_polar` 与 `model.r2p`，**从未调用 TaskNet**。因此其 `max|Δ|=0` 只说明该图上 R2P 重复前向是确定性的，**不能**证明：
   - TaskNet 的扩散噪声配对正确；
   - 整个实现没有 bug。
   上一版"前向完全确定性、无实现 bug"的表述**撤回**。真正覆盖 TaskNet 端点的证据是本次 alpha=0/alpha=1 检查（第二节）。
2. **`real_fb_test.py` 的余弦是单图诊断。** 它只读取验证集**第一张图**。上一版把 `cos=0.99895` / `0.99935` 写成 24 张汇总，**错误**。现明确标为**单图诊断（n=1）**。
3. **补充日志留存状态**：`det_test.py`、`delta_test.py`、`real_fb_test.py` 在服务器上**只找到脚本，未找到保存的打印结果日志** → 标注为 **"未独立核验"**。按指示**不为此重跑**。（已再次确认：服务器仅 `endpoint_check/endpoint_check.log` 为本轮新日志。）
4. **残差恒等式必须保留 clamp。** 本次显式核验（单图）：`P1 = clamp(p + d1)` 严格成立（`max|Δ|=0`），而 `max|P1 − (p+d1)_未截断| = 9.70e-02`。**去掉 clamp 写成未截断恒等式是不准确的**，上一版表述撤回。
   同时**区分两个量**（必须分别标明）：
   - **网络原始 delta** `d1`：`mean = 1.5948e-02`，`max = 9.9770e-02`；
   - **截断后的有效更新** `P1 − p`：`mean = 1.5922e-02`，`max = 9.9770e-02`。
   该图上截断触发比例很小（`raw_sum<0` 0.0785%、`raw_sum>1` 0.0631%，合计 0.14%），但**非零**；两者不得混用。
5. **`diffusion_rng_stream_shared=True` 是写入的常量**，不是随机张量相等的检验结果。源码中的保存/恢复支持其配对**意图**，但该标志本身**不能充当检验**。
6. **子集构成如实标注**：24 张 = **21 张 easy + 3 张 hard**，覆盖 8 场景。它**是按场景分层**（每场景最先 3 张），**不是场景与难度共同分层**。上一版"分层子集"的措辞已修正。

---

## 五、机制解释：撤回因果表述，只保留待验证假设

**撤回**以下表述：

- ~~"已证明同方向过量外推造成退化"~~
- ~~"与反馈无关"~~
- ~~"可以排除共享权重"~~

**仅保留为待验证假设**：

> **假设 H1（待验证）**：两轮 R2P 的原始残差方向接近、平均绝对幅度接近，可能表现为对残差的重复更新。

该假设目前的证据**仅有单图诊断**（`real_fb_test.py`，n=1，且**未独立核验**），**不能**推广到 24 张，更不能据此断言"更新过量造成 PSNR 下降"。

**其他必须收紧的点**：

- `cos ≈ 1` 仅限制**方向**；即便幅度也接近，仍应检查**相对残差误差**与**截断后的有效更新**，才谈得上"重复更新"。
- 两组出现相似现象，只提示**检查共同环节**；**不能排除共享权重**，也**不能证明现象与反馈无关**。
- `feedback_scale = 0.0207` **不是**反馈项相对主分支的实际贡献比例；不能仅凭该系数或小小的输入差值，断言反馈不会影响任务结果。

---

## 六、本轮范围与产物路径

**执行范围**：仅完成上述定义修正与 **alpha=0、alpha=1 端点检查**。**未**运行 0.25 / 0.5 等幅度扫描；**未**改变模型或训练配置；**未**重训、未追加预算、未重跑 872 张。

**服务器产物**（项目根 `pidsr_k2_929/` 下分类保存）：

| 类型 | 路径 |
|---|---|
| 运行脚本 | `evaluation_872/endpoint_check.py` |
| 运行日志 | `evaluation_872/endpoint_check/endpoint_check.log` |
| 逐图 JSON | `evaluation_872/endpoint_check/endpoint_check_result.json` |
| 保留的消融脚本 | `evaluation_872/alpha0_selfcheck.py`（改称：反馈关闭消融） |
| 消融结果 | `evaluation_872/alpha0_selfcheck_result.json` |
| clamp 核验脚本 | `evaluation_872/clamp_check.py` |

**本地副本**（`F:\WORK\projects\pidsr_recurrent_joint_k2\`，未在 C 盘新建工作文件）：

| 类型 | 路径 |
|---|---|
| 运行脚本 | `server_scripts\endpoint_check.py` |
| 运行日志 | `server_scripts\endpoint_check\endpoint_check.log` |
| 逐图 JSON | `server_scripts\endpoint_check\endpoint_check_result.json` |
| 修订报告（本文件） | `analysis\endpoint_check_report_20260930.md` |

---

## 七、逐图 JSON 字段说明

`endpoint_check_result.json` 每条 `per_image` 记录包含：

- `sample_id`、`scene`、`difficulty`
- `alpha0`：`input_max_abs_diff_vs_P1`、`output_max_abs_diff_vs_T1`、`output_mean_abs_diff_vs_T1`、`tolerance`、`passed`
- `alpha1`：`input_max_abs_diff_vs_P2`、`output_max_abs_diff_vs_T2`、`output_mean_abs_diff_vs_T2`、`tolerance`、`passed`
- `psnr`：`t1`、`t2`

顶层另含 `definition`、`subset`（含 `difficulty_composition` 与 `honesty_note`）、`tolerances`、`rng_discipline`，以及各组 `alpha0`/`alpha1` 的汇总与 `n_passed`。所有 `passed` 均由实测差值计算。
