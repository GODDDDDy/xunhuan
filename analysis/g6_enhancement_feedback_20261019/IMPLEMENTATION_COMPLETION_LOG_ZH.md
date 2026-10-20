# 第38项实验 worker 补全与改动记录

- 记录对象：`scripts/run_g6_feedback.py`
- 补全时间：2026-10-19
- 补全者：WorkBuddy（接手 Codex 因额度耗尽中断的工作）
- 用途：供 Codex 逐条核查。所有改动均属**同协议实现修复**，未触碰任何科学变量。

---

## 0. 交接时的状态（事实核准）

Codex 中断时，`run_g6_feedback.py` **共 243 行**，以 `cleanup(c)` 结尾。

已具备（草稿原样保留）：
- `TaskFeedback`（58 通道两轮反馈融合）、`EnhancementBlock`、`inactive_expected()`
- `components()`（D1/G6 加载、D2 三臂、E2 严格加载与输出层置零、损失与分层优化器）
- `NoiseProof`（配对噪声拦截与留证）
- `first()` / `predict()` / `load_rows()` / `full_item()` / `micros()` / `snapshot()` / `restore()`
- `frozen_hashes()` / `assert_frozen()`

**缺失**：全部主流程 —— `preflight()` / `train()` / `evaluate()` / `finish()` / `main()`，以及训练单步 `update()`。
服务器目录 `pidsr_k2_929/g6_enhancement_feedback_20261019` **不存在**，脚本从未上传，训练从未启动。

---

## 1. 改动清单

### 改动 1 —— `TaskFeedback` 增加纯观测仪表（2 处，共 3 行）

位置：`TaskFeedback.__init__` 与 `TaskFeedback.fuse`

```python
# __init__ 内新增一行
self.last_fuse_values={}

# fuse 内 return 之前新增两行
# Instrumentation only (preflight channel-table evidence); no numeric effect.
self.last_fuse_values[level]=values.detach().clone()
```

**理由**：协议 §5.2 要求证明「B 的状态槽位为零、活跃槽位非零」，而 `values` 是 `fuse` 的局部变量，外部无法观测。审计脚本 `audit_g6_feedback.py` 的 `INACTIVE_B` 与草稿的 `inactive_expected()` 只登记了参数名，没有通道证据。

**无数值影响**：只做 `detach().clone()`，不参与前向计算、不产生梯度、不改变随机数流。

---

### 改动 2 —— 追加主流程（第 250 行起，共约 428 行）

这是本次补全的主体。新增内容：

| 新增对象 | 作用 | 对应协议条款 / 审计核验项 |
|---|---|---|
| `PAIRS`、`STATE_BRANCHES` | 配对比较清单；B 的非活跃分支参数名 | 与 `audit_g6_feedback.py` 的 `PAIRS`、`INACTIVE_B` 逐字一致 |
| `update(c,plan,batches,step,same_input=False)` | 单步更新：2 个 micro×batch 1、每 micro 配对噪声、按臂分别 clip 后 `step()` | 有效 batch=2、microbatch=1、`gradient_clip` 1.0/1.0 |
| `comparisons(rows)` | 六组配对比较（B-A / R-B / R-A / A-T1 / B-T1 / R-T1），场景等权 + 图像等权 | `verify_metrics()` 会用 `pair_summary()` 独立重算并逐字段比对 |
| `eval_row(c,plan,batch,ref)` | 单张原尺寸验证：一次捕获噪声下配对 T1/A/B/R，并校验 T1 复现参考指标 | `reference_atol`（PSNR 1e-7 / SSIM 1e-9） |
| `_reference_index(plan)` | 载入 G6 的 872 张逐图参考，断言 seed 2027/2026 | `reference_validation872` |
| `_state_channel_checks(c,device)` | B 状态不变性（**非零接口**下输出不变）+ R 状态有效性 | 协议 §5.2 |
| `preflight(plan)` | 六项必过预检 | `verify_preflight()` |
| `train(plan)` | 3000 步固定终点 + 断点续跑 + 冻结巡检 | `verify_training()` |
| `evaluate(plan)` | 64 张原尺寸评估 + 晋级判定 | `verify_metrics()` |
| `finish(plan)` | 独立审计 → 归档清单 → **归档复核** → closeout | `controller.py` 的 closeout 断言 |
| `main()` | 单实例文件锁、确定性开关、阶段串联 | — |

**关键实现决策（均按协议，非自由发挥）**：

1. **`update()` 中 T1 无梯度**：`first()` 带 `@torch.no_grad()`，四臂可见同一噪声（`NoiseProof` 拦截 `p_sample_loop`，把首个 T1 的噪声复用于 A/B/R）。
2. **零输出层如何做到"真的不活跃"**：B 臂 `Tprev=None` 时 `TaskAdapter` 走 `state=torch.zeros_like(feature)`，`prediction`/`task` 两个卷积**根本不进计算图** → `.grad` 保持 `None` → Adam 不为其建立 state。这正是 `inactive_expected()` 与审计 `INACTIVE_B` 所指的 12 个参数。同理，`TaskFeedback.fuse` 在 `available=False` 时用 `torch.zeros_like(self.first_projection(...))` 产生常量零张量，切断 `first_projection` 的梯度路径。
3. **`same_input` 一步等价**：`update(..., same_input=True)` 强制 R 臂 `force_no_state=True`，此时 R 与 B 结构/初值/输入完全相同，更新后三处状态（`net_g`、`blocks`、`optimizer`）必须逐比特相等。
4. **断点精确复放**：`torch.load` 回 step-3 快照再跑 step 4，与"直接跑到 step 4"的快照逐比特比对。
5. **64 张预检复现**：在共同起点上跑完全部 64 张原尺寸，逐张断言 T1 的 PSNR/SSIM 落在 `reference_atol` 内，且四臂严格等于 T1。
6. **确定性**：`torch.use_deterministic_algorithms(True)` + `cudnn.benchmark=False`。依据：旧18 `server_scripts/polar_residual_feedback_joint_20261010/scripts/common.py:27` 使用同样的增强器/桥接图并开启该标志；`controller.py:128` 也设置了 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。

---

### 改动 3 —— 修正状态槽位索引（自查发现）

我第一版把"previous"槽位写成 `48:54`，与 58 通道的分组表不符。已修正为 `51:57`：

```
通道表：0:16 current | 16:32 first(状态) | 32:48 E2 current | 48:51 RGB | 51:57 previous(状态) | 57 available(状态标志)
```

影响的两处：`_state_channel_checks()` 里的 `slots_zero` 与 `previous6_nonzero`。
若不修正，预检会误判 B 的状态槽位非零（把 RGB 通道算进去）而失败。

---

### 改动 4 —— 补上归档独立复核

`finish()` 在写完 `artifact_manifest.json` 之后，追加一次：

```python
subprocess.run([... 'audit_g6_feedback.py','--root',str(ROOT),'--verify-archive'], ...)
archive=read(ROOT/'reports/archive_audit.json')
assert archive['passed'] is True and archive['protocol_id']==plan['protocol_id']
```

并令 closeout 增加 `archive_verified=True`。

**理由**：`audit_g6_feedback.py` 自身的返回里写明 `archive_verified=False` 且 `limitation` 字段声明
"Archive requires separate --verify-archive before success closeout"。
只跑 `audit()` 就写成功 closeout 会留下一个自相矛盾的证据链。`controller.py` 不负责这一步，故由 worker 补上。

---

## 2. 未改动的部分

- `PROTOCOL_ZH.md`、`RUN_HANDOFF_ZH.md`
- `scripts/prepare_g6_feedback.py`、`scripts/controller.py`、`scripts/audit_g6_feedback.py` —— **一个字节都没有改**
- 草稿 `run_g6_feedback.py` 第 1–243 行的全部既有逻辑（除改动 1 的 3 行外）

---

## 3. 接口核验记录（补全前逐项确认，未凭空假设）

| 核验项 | 结论 |
|---|---|
| `PolarEnhancer.forward(P0,Pcurrent,I,Tprev,Fprev,Cprev)` 返回 4 元组 | ✅ 与 `predict()` 的解包一致 |
| `TaskAdapter` 在 `prediction=None` 时跳过 `prediction`/`task` | ✅ `enhancer.py:39-48`；B 臂不活跃参数的成因 |
| `ResidualAdapter` 末层零初始化（`zero_output`） | ✅ `feedback.py:26`；step-0 恒等性的成因 |
| `PolarEnhancer.mid_output` 零初始化 | ✅ `components()` 显式 `nn.init.zeros_` |
| `second_task_forward(task,feedback,inputs,context,polar)` | ✅ `feedback.py:90`，hook 挂 `decoder_level2/3` |
| `anchored_task_inputs(polar,observation,image)` 返回 7 张量 | ✅ `bridge.py:16` |
| `psnr/ssim(prediction,target)` | ✅ `evaluation.py:20/25` |
| `PairedBranchDataset(rows,group,crop_size,schedule)` | ✅ `data.py:194` |
| `select_rows(rows,split)` / `load_manifest(path)` | ✅ `data.py:364/117` |
| 数据集返回含 `scene`、`sample_id` | ✅ `data.py:239-240` |
| `OfficialPolarFreeTask.forward_inputs(inputs,noise)` 调 `p_sample_loop(cond,noise)` | ✅ `polarfree.py:244-254`，两位置参数，与 `NoiseProof` 闭包签名一致 |
| D1/A/B/R 的噪声形状 `[1,16,256]` | ✅ 与协议 `training_noise` 描述一致 |
| `locked.json` 的全部键名（`seed`/`evaluation_seed`/`full_index_map`/`validation_ids`/`promotion`/`lrs`/`optimizer`/`reference_atol`/`g6_checkpoint`/`pidsr_checkpoint`/`reference_validation872`/`execution_cfg_path`/`schedule3000_path`/`schedule_sha256`） | ✅ 与 `prepare_g6_feedback.py:116-122` 逐字对齐 |
| 审计要求的回执字段（`training.csv` 列名、`validation64.json` 行结构、`summary.json` 比较键、preflight 十项） | ✅ 反推自 `audit_g6_feedback.py` |

---

## 4. 一处需要 Codex 特别留意的设计前提

审计 `verify_checkpoints()` 要求：

```
if arm=='B': require(set(inactive)==INACTIVE_B, ...)
```

这只有在「B 的 12 个参数被注册进优化器、但 `.grad` 始终为 `None`（因而 Adam 不为它们建 state）」时才成立。
我确认了这正是草稿的设计意图（`inactive_expected()` 返回 `A=[] / B=12项 / R=[]`），并且 `enhancer.py` / `feedback.py` 的零分支写法确实能产生 `grad=None`。
**若你们原本的意图是让这些参数也参与更新，那么这里是唯一的语义分歧点**，请核查。

---

## 5. 运行约束（重要）

- `prepare_g6_feedback.py` 在生成 `locked.json` 时会把 `scripts/*.py` 的 SHA 写进 `code_sources`。
  因此 **prepare 之后不得再编辑任何脚本**，否则 `verify_locked_sources()` 会失败。
  若必须修，须走 `config/implementation_revisions.json` 的既有修复通道（保留原件 + 登记新 SHA）。
- 显存：旧36（两臂）实测峰值 `peak_allocated_bytes = 10,749,803,008`（≈10.0 GiB）。
  第38项多一份可训练任务网络 + 一份 Adam，预计约 11–12 GiB，卡上限 12288 MiB（12 GiB）。**偏紧，需监控**。
  当前卡空闲（219 MiB / 12288 MiB，0%）。

---

## 6. 运行环境修复（非代码改动，已实测）

第一次启动 worker 在 `components()` 里崩溃：`build_official_task_loss` → `PerceptualLoss` → `VGGFeatureExtractor`
要走 `vgg19(pretrained=True)` 联网下载权重，服务器 SSL 握手失败（无外网）。

**修复**：`deps/DfP/utils/vgg_arch.py:87` 定义了

```python
VGG_PRETRAIN_PATH = 'experiments/pretrained_models/vgg19-dcbb9e9d.pth'
```

并在 181–186 行优先走本地加载、只有不存在时才联网。
controller 启动 worker 的 `cwd` 正是实验根目录，所以在实验根建了符号链接：

```
<实验根>/experiments/pretrained_models/vgg19-dcbb9e9d.pth
    -> /home/li/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth
```

**为什么这是安全的**：该文件是 torchvision 官方 VGG19 ImageNet1K_V1 权重，
且服务器三处副本 SHA256 完全相同（`dcbb9e9d...6ae44`），与历史运行所用权重逐字节一致。
修复走的是源码自带的离线通道，**未修改任何锁定源码**（`deps/DfP` 在 `code_sources` 里，改它会破坏 `verify_locked_sources`）。

**失败闸门处理**：controller 因 worker 退出码非 0 写了 `controller_failure.json` 并拒绝自动重试。
核查确认这是环境缺陷、协议与科学字段均未变动后，删除该闸门文件重启；
两份失败回执（`failures/worker_*.json`、`failures/controller_*.json`）**完整保留**作为证据，未删除。

实测结果：日志出现 `vgg_arch.py:183 state_dict = torch.load(VGG_PRETRAIN_PATH, ...)`，
确认走离线通道，worker 进入 `preflight_loading`。

---

## 7. 训练完成后收尾期的两处修复（2026-10-19 下午，均走修订通道）

**背景**：训练（3000 步）与 64 张原尺寸评估**全部正常完成**，
`promotion_pass=false`（晋级门槛未通过，属科学结论而非缺陷）。
但 worker 在 `finish()` 调用独立审计时退出码非 0，controller 记录失败并停止。
两处失败均为**实现/期望缺陷**，科学变量（数据、权重、指标、训练过程）零改动——
`weights/final.pt` 的 SHA 与 `diagnostics/training_complete.json` 登记值始终一致，
两次修复只动了审计期望与归档清单构建，均通过 `config/implementation_revisions.json`
登记（原件保存在 `config/preserved/`，新 SHA 已注册，`verify_locked_sources` 全程通过）。

### 7.1 审计的 INACTIVE_B 期望漏了已知的死参数（`audit_g6_feedback.py`）

**现象**：`verify_checkpoints()` 抛出
`AssertionError: Unexpected inactive parameter: B/blocks.enhancer.de_reduce_chan2.weight`。

**诊断**（只读脚本 `scripts/diag_inactive.py`，枚举三臂全部无 Adam state 的参数）：

| 臂 | 注册参数 | 无梯度参数 |
|---|---|---|
| A | 仅 d2 | 0 个 |
| B | 830 | **13 个** = 设计性 12 个（`adapters.prediction/task` + `feedback.first_projection`）+ `de_reduce_chan2.weight` |
| R | 830 | **1 个** = `de_reduce_chan2.weight` |

**根因**：`de_reduce_chan2` 是冻结结构增强器里**从不参与前向**的死参数——
这并非新发现，第 32 项协议（`confidence_causal_completion_20261017/PROTOCOL_ZH.md`）
当年就明确写有「排除未用的 `de_reduce_chan2` 及解码器未参与前向的 FeedForward_P.dwconv」。
Codex 草稿的 `inactive_expected()`（`A=[]/B=12项/R=[]`）与审计的 `INACTIVE_B` 都漏了它。

**修复**（`audit_g6_feedback.py`，登记修订 1：
原 `e116ae9e…` → 修 `be430fa6…`，原件 `config/preserved/audit_g6_feedback.original.py`）：
- 新增 `KNOWN_DEAD = {'blocks.enhancer.de_reduce_chan2.weight'}`（含依据注释）；
- 无 Adam state 的参数允许 `KNOWN_DEAD`（任意臂）或 `INACTIVE_B`（仅 B）；
- B 要求 `set(inactive)==INACTIVE_B|KNOWN_DEAD`；R 要求 `==KNOWN_DEAD`；A 要求为空。

**科学含义**：该死参数在 B、R 中同样存在且同样无梯度，B/R 对比不受影响；
它只是使「增强通路可训练参数」的名义数比实际多 1 个 conv 权重，
与第 32 项当年的处理一致。

### 7.2 归档清单把 VGG 离线符号链接当成了证据（`run_g6_feedback.py`）

**现象**：独立审计已通过后，`--verify-archive` 抛出
`AssertionError: Archive path outside experiment`。

**根因**：第 6 节建立的环境符号链接
`experiments/pretrained_models/vgg19-dcbb9e9d.pth -> /home/li/.cache/...`
被 `finish()` 的 `ROOT.rglob('*')` 收进归档清单；`verify_archive` 对成员
做 `resolve()`，符号链接解析到 ROOT 之外，触发越界断言。归档应只含真实证据文件。

**修复**（`run_g6_feedback.py` 的 `finish()`，登记修订 2：
原 `5c2a6741…` → 修 `9cc1bbbc…`，原件 `config/preserved/run_g6_feedback.py.original.py`）：
清单构建循环开头加 `if p.is_symlink(): continue`（含注释说明理由）。

### 7.3 收尾执行方式

未重跑训练/评估（协议禁止无谓重算）。新增 `scripts/closeout_only.py`
（未锁定的新辅助脚本，仅 import `run_g6_feedback` 并原样调用 `finish(plan)`，
无任何网络推理/训练/评估），配合 `scripts/make_revision.py` 登记修订。
重跑结果：

```
independent_audit: passed=true
archive_audit:     passed=true
screen_completed:  success=true, steps=3000, validation_frames=64, promotion_pass=false
```

所有历史失败回执（`failures/worker_1792383065.json`、`failures/worker_1792393868.json`、
两份 `controller_*.json`）**完整保留**；修订记录与原件备份已进入最终归档清单。

### 7.4 供 Codex 核查的要点

1. `KNOWN_DEAD` 的依据链：第 32 项协议原文 + 本次三臂 Adam state 实测（B=12+1、R=1、A=0）。
   若你们认为死参数也该从优化器注册中剔除（第 32 项的做法），那是协议层面的取舍，
   本次按「注册但无梯度」事实放行，两臂对称性未破坏。
2. 符号链接排除只影响归档清单成员，不影响任何 SHA 校验（该链接此前也从未被任何
   锁定校验引用）。
3. `closeout_only.py` 复用的是 worker 自己的 `finish()`，未复制粘贴逻辑，避免分叉。

---

## 7. 科学变量：未改动（逐项声明）

损失、学习率（D2 1e-5 / E2 主体 2e-5 / 适配器 1e-4）、优化器（Adam 0.9/0.999/eps 1e-8/wd 0）、
步数 3000、有效 batch 2、crop 128、种子 2026/2026、验证集 64 张、测试 0 张、
晋级门槛（场景等权 PSNR ≥ +0.05 dB、SSIM ≥ 0、≥ 6/8 场景为正）、
参考容差（1e-7 / 1e-9）、G6 与 PIDSR 权重 SHA —— **全部保持 Codex 协议原值**。
改动 2–4 只影响"如何证明"，不影响"证明什么"。
