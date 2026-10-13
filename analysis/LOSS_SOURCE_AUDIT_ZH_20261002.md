# 训练损失源码只读核对

**日期**：2026-10-02
**依据**：`ALPHA_SWEEP_REVIEW_ZH_20261002.md` 第 2 条建议（"只读核对实际训练所用损失源码、系数、FFT 输入/维度、phase 归约及周期处理；比较诊断分项之和与训练 criterion 定义，避免把公开仓库现版本等同于历史运行依赖版本"）
**性质**：**只读**。未训练、未更新权重、未修改任何检查点。唯一新增计算是 4 张图上的前向（用于数值核对）。

---

## 0. 结论摘要

1. **实际训练用的损失模块已定位**：`pidsr_k2_929/deps/DfP/polarapp/losses.py` 的 `DfPTaskLoss`，由 `rgb_final_confirmation.losses.build_official_task_loss()` 导入。
2. **运行时源码与公开仓库 `PolarAPP/main` 逐字节相同**（sha256 `41c40e3fbef49243a54c1d050e4a802c07d6f2b29b15ed4f1ecca470af8ebc59`）。⇒ "公开仓库现版本 ≠ 历史运行版本"这一担忧在本项目**不成立**，本节所有源码结论对历史运行同样有效。
3. **诊断脚本的重算与官方 criterion 数值一致**：4 张图上最大绝对差 **6.147e-08**（float32 精度内），两组结果完全相同。⇒ 此前报告中的损失分项是**可信的**，反事实记账所依赖的"phase 以系数 1 线性相加"前提**成立**。
4. **关键差异确认**：本项目的任务损失**不是** PolarFree 原训练目标。差异不止 phase 权重，而是**三项**（phase 权重、SSIM 项、VGG 项），见 §3。
5. **新增发现（此前未记录）**：训练优化的是 **`0.5·L(T1) + 1.0·L(T2) + λ·对齐损失`**，即第一轮损失以 **0.5** 权重参与。这解释了两轮为何被同时监督，也意味着"第二轮退化"不能简单归因于"第二轮没被好好优化"。

---

## 1. 实际训练损失的完整定义

### 1.1 单次任务损失（`criterion`，即 `DfPTaskLoss`）

```python
def forward(self, prediction, target):
    loss  = F.l1_loss(prediction, target)          # 像素 L1，权重 1
    loss += (1 - ssim(prediction, target))         # SSIM 项，权重 1
    loss += self.phase(prediction, target)         # phase 项，权重 1（内部无额外系数）
    loss += self.perceptual(prediction, target)    # = weight(0.1) * L1(VGG conv5_4)
    return loss
```

权重取值来自 `protocol/base_server.yaml`：

| 项 | 权重 | 来源 |
|---|---|---|
| L1 | 1.0 | 代码硬编码 |
| 1 − SSIM | 1.0 | 代码硬编码 |
| phase | **1.0** | 代码硬编码（`PhaseLoss()` 无 weight 参数） |
| perceptual（VGG conv5_4, L1） | **0.1** | `loss.perceptual_weight: 0.1` |

### 1.2 phase 项的实现细节（复核意见点名要求）

`polarapp.losses.PhaseLoss`：

```python
gray_a = image_a.mean(dim=1, keepdim=True)     # 3 通道 → 1 通道（简单均值）
phase_a = torch.angle(torch.fft.fft2(gray_a + 1e-5))
phase_b = torch.angle(torch.fft.fft2(gray_b + 1e-5))
return F.mse_loss(phase_a, phase_b)
```

- **FFT 输入**：对 **3 通道 RGB 输出**在通道维取均值得到灰度单通道，**再**做 `fft2`。
  ⚠️ 这一点很重要：phase 项比较的是 **TaskNet 输出的 RGB（3 通道）**，不是 12 通道 polar。
- **FFT 维度**：`torch.fft.fft2` 对最后两维（H, W）做二维 FFT，输出复数谱，形状不变。
- **相位取值**：`torch.angle` 取辐角，落在 (−π, π]。
- **归约方式**：`F.mse_loss` 默认 `reduction='mean'`，对**全部 H×W 频率位置等权平均**（未做频率加权，也未只在低频上计算）。
- **周期处理**：`epsilon = 1e-5` 加在**空域图像**上（`gray + 1e-5`），不是加在 `atan2` 的两个分量上。
  ⚠️ **这是相位环绕问题**：MSE 直接比较两个辐角，在 ±π 边界附近的环绕差会被计为 ~2π 的大误差。实现**未做**相位解缠或环绕修正。

### 1.3 训练总目标（K=2）

```python
output.loss1 = criterion(output.t1, batch["target"])   # 第一轮
output.loss2 = criterion(output.t2, batch["target"])   # 第二轮
task_loss   = round_weights[0] * loss1 + round_weights[1] * loss2
alignment_loss = feature_alignment_loss(...)
```

`config.yaml`：`recurrent.round_loss_weights: [0.5, 1.0]`，且被硬断言锁定：

```python
if config["recurrent"]["round_loss_weights"] != [0.5, 1.0]:
    raise ValueError("Locked first-version loss weights are [0.5, 1.0]")
```

⇒ **总目标 = 0.5·L(T1) + 1.0·L(T2) + λ_feature_alignment·对齐损失**（对齐项只在 A 组路径启用；G4/G5 等消融组的 λ 不同）。

### 1.4 perceptual 项的 VGG 提取器

`PerceptualLoss` 使用项目自有的 `VGGFeatureExtractor(layer_name_list=["conv5_4"], vgg_type="vgg19", use_input_norm=True, range_norm=True)`，损失为 `weight * F.l1_loss(feat_pred, feat_target.detach())`。已实测确认这三个参数（层名、input_norm、range_norm）在两组中均为上述取值。

---

## 2. 数值一致性核对（决定性验证）

**目的**：确认诊断脚本中"手动重算分项并求和"与**官方 criterion 自身的输出**数值一致——若不一致，此前所有分项与反事实记账都不成立。

**方法**：`server_scripts/alpha_sweep/verify_loss_recompute.py`。构建官方 criterion，对 4 张 validation 图各做一次：
- 路径 A：直接调用 `criterion(pred, target)`；
- 路径 B：按诊断脚本的口径手动重算 `L1 + (1−ssim) + phase + 0.1·VGG`。

**结果**：

| 组 | 检查图数 | 最大绝对差 | float32 精度内 |
|---|---|---|---|
| no_feedback | 4 | **6.147e-08** | 是 |
| feedback | 4 | **6.147e-08** | 是 |

逐图明细（两组数值完全相同，符合预期——同一预测张量、同一 criterion）：

| sample_id | 官方 criterion | 手动重算 | 绝对差 |
|---|---|---|---|
| validation/easy/01/0000 | 1.704054594 | 1.704054656 | 6.15e-08 |
| validation/easy/01/0001 | 1.716374993 | 1.716375044 | 5.03e-08 |
| validation/easy/01/0002 | 1.752273202 | 1.752273250 | 4.84e-08 |
| validation/easy/01/0003 | 1.721842051 | 1.721842058 | 7.45e-09 |

**判读**：差异在 float32 的 1 ULP 量级，**不是**实现分歧，仅为求和顺序不同导致的浮点舍入。⇒

- 此前报告中的损失分项（§A/§C/§D）**可信**；
- 反事实记账的前提"`phase` 以系数 1 线性相加"**成立**，因此"把 phase 系数换成 w 后总损失如何变化"这一代数分解**有效**。

> ⚠️ **边界**：本核对只覆盖 4 张图、单一预测来源。它证明的是"重算公式与 criterion 定义一致"，**不**证明整条评估链路无 bug。

---

## 3. 与 PolarFree 原训练目标的差异清单

比对对象：PolarFree 公开仓库 `options/train/ours_stage2.yml` + `polarfree/models/PolarFree_S2_model.py` + `polarfree/utils/losses.py`。

### 3.1 逐项对照

| 项 | PolarFree 原目标 | 本项目实际（PolarAPP 适配） | 是否一致 |
|---|---|---|---|
| 像素 L1 | `pixel_opt: L1Loss, loss_weight=1.0` | `F.l1_loss`，系数 1 | ✅ 一致 |
| **SSIM 项** | **无**（stage2 配置里没有 SSIM 损失） | `(1 - ssim)`，系数 **1** | ❌ **本项目多出** |
| **phase 权重** | `phase_opt.loss_weight = 0.1`，且 `PhaseLoss` 内部**把 loss_weight 乘进返回值** | `PhaseLoss` **无** weight 参数，系数 **1** | ❌ **差 10 倍** |
| phase 实现 | `epsilon=1e-8`；`gray = mean(dim=1)`；`angle(fft2(gray+eps))`；`mse_loss` | `epsilon=1e-5`；其余步骤**逐步相同** | ⚠️ **仅 epsilon 不同**（1e-8 vs 1e-5） |
| **VGG 项** | `vgg_opt.loss_weight=0.02`，`VGGLoss` 用 **5 层加权**（`[1/32,1/16,1/8,1/4,1]`）+ **MSELoss** | `perceptual_weight=0.1`，**单层 conv5_4** + **L1Loss** | ❌ **权重、层数、距离度量三者都不同** |
| TV 项 | `tv_opt.loss_weight=0.0005` | **无** | ❌ 本项目缺失 |
| pixel_diff（扩散损失） | `pixel_diff_opt: L1Loss, loss_weight=1.0`，作用于 `(prior_z, prior)` | 本项目为**冻结**扩散先验，未参与训练 | ❌ 本项目未用 |
| 损失组装 | `l_total` 顺序累加上述各项 | 逐项相加，结构等价 | ✅ 结构一致 |

### 3.2 最关键的差异：phase 权重

这是复核意见预判的差异，现已在源码层确认，且比预想更明确：

```python
# PolarFree: polarfree/utils/losses.py
class PhaseLoss(nn.Module):
    def __init__(self, loss_weight=1, epsilon=1e-8):
        self.loss_weight = loss_weight
        ...
    def forward(self, img1, img2):
        ...
        return self.loss_weight * loss          # ← 权重在这里生效
```

```python
# 本项目实际使用（= PolarAPP）: deps/DfP/polarapp/losses.py
class PhaseLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon
        ...
    def forward(self, image_a, image_b):
        ...
        return F.mse_loss(phase_a, phase_b)     # ← 没有 loss_weight
```

⇒ PolarFree 的 stage2 实际 phase 权重 = **0.1**；本项目 = **1.0**。**相差 10 倍。**

### 3.3 关于命名（复核意见已采纳）

本项目应称**「当前采用的 PolarAPP 适配任务损失」**，**不应**称"PolarFree 原训练目标"。理由如前表：三项权重/结构不同，另有 TV 与扩散项缺失。

> ⚠️ **不能只恢复一个系数就叫完整官方复现**：即使把 phase 系数改成 0.1，SSIM 项（原目标没有）、VGG 项（层数/权重/度量均不同）、缺失的 TV 与 pixel_diff 项仍然存在。要谈"官方复现"，需要逐项对齐。

---

## 4. 与既有结论的关系

### 4.1 目标分歧（此前已报）现在有了机制层面的线索

此前报告：两组任务总损失随 α 单调下降，而 PSNR/SSIM 同时下降 → "优化目标与观测指标分歧"。

本次核对补充：**phase 项在总标量中占 94.9%**（§D），而 phase 的实现是**直接对相位谱做 MSE、未做环绕修正**。这提供了一条**待验证假设**（H2）：总损失被 phase 项主导，而相位 MSE 可能并不紧贴 PSNR/SSIM 所关注的重建质量。

> ⚠️ **这仍是假设，不是结论。** 占比不等于梯度占比（§D 已注明）。要判断"phase 的梯度是否主导且与 PSNR 冲突"，必须做复核意见第 3 条的**梯度诊断**，本文件**没有做**。

### 4.2 反事实记账的适用范围（更正一处）

此前 §C 写"w 从 1 降到 0.1"——现在这个取值的意义更明确了：**w=0.1 恰好就是 PolarFree 原配置的 phase 权重**。因此 §C 那张表可以这样重读：

> 把 phase 项从"本项目实际使用的系数 1"改回"PolarFree 原配置的 0.1"，两组 T2−T1 总损失就从负转正（+0.001455 / +0.000807）。

即：**现行的"T2 总损失更优"这一排序，是在 phase 系数为 1 的前提下成立的；改用原配置的 0.1 就不成立。** 这仍只是代数记账，不预测重训结果，也不证明 0.1 更优（因为还有其他差异项）。

### 4.3 新增观察：第一轮损失以 0.5 权重参与训练

`0.5·L(T1) + 1.0·L(T2)` 意味着第一轮**并非完全自由**——它被显式的任务损失监督着。此前的诊断（α 扫描）只变动了**第二轮**的输出，第一轮固定。因此：

- α 扫描回答的是"在**第一轮已固定**的前提下，第二轮步长怎么调"；
- 它**不能**回答"两轮联合优化是否次优"——那需要改变训练目标本身。

> 这与复核意见第 4 条"最小目标对照"的设计意图一致：若要做目标对照，应改的是**损失权重**，而不是仅缩放输出。这也说明此前的 α 扫描与后续目标对照是**互补**关系，不重复。

---

## 5. 本次未做的事（明确边界）

| 复核意见条目 | 状态 |
|---|---|
| 第 2 条：训练损失源码只读核对 | ✅ **已完成**（本文件） |
| 第 2 条：比较诊断分项之和与 criterion 定义 | ✅ **已完成**（§2，差 6e-08） |
| 第 3 条：梯度诊断（各分项对共享 R2P 的梯度范数/夹角） | ❌ **未做** |
| 第 4 条：最小目标对照训练 | ❌ **未做** |
| 第 5 条：独立纠错器 A/B/C（原优先级 2） | ❌ **未做** |

**因此以下事项仍**不得**断言：**
- 不得说"phase 是第二轮退化的唯一原因"；
- 不得说"phase 占比 94.9% ⇒ phase 梯度主导训练"；
- 不得说"把 phase 权重改成 0.1 就能解决问题"；
- 不得说"已经完成 PolarFree 官方目标的复现"。

---

## 附：本次核对的证据文件

| 文件 | 内容 |
|---|---|
| `_dl/loss_audit/losses_runtime.py` | 实际运行的 `polarapp/losses.py`（sha256 `41c40e3f…`） |
| `_dl/loss_audit/losses_github_main.py` | 公开仓库 PolarAPP/main 同名文件（**与上者逐字节相同**） |
| `_dl/loss_audit/pf_losses.py` | PolarFree `utils/losses.py`（PhaseLoss / VGGLoss 原文） |
| `_dl/loss_audit/pf_s2_model.py` | PolarFree `PolarFree_S2_model.py`（损失组装原文） |
| `_dl/loss_audit/polarfree_stage2.yml` | PolarFree `ours_stage2.yml`（官方权重配置） |
| `_dl/loss_audit/loss_recompute_verification.json` | §2 数值核对结果（逐图） |
| `server_scripts/alpha_sweep/verify_loss_recompute.py` | 核对脚本（只读） |
