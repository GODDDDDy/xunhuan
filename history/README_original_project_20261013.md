# PIDSR R2P 两轮任务反馈实验

面向用户的持续记录：[实验记录.md](../实验记录.md)。汇总已核验的历史尝试、最新增强器实验及后续诊断，用通俗语言说明每次的动机、结果与结论边界。

2026-10-09模块核查：新增[偏振增强模块定位与重构建议](../analysis/POLAR_ENHANCER_ROLE_AND_REDESIGN_ZH_20261009.md)。核对当前轻量R2P的双RGB条件入口、原PIDSR恢复目标与当前任务适配的差异，以及PolarFree实际只接收四角度灰度/AoLP/DoLP的接口。提出偏振主干、多尺度、明确任务状态和结构化灰度修正方案；仍是未验证设计，本次没有改模型或启动训练。

2026-10-09夜间实验已完成：置信主方案A2与直接反馈对照A0均追加16000步，并各完成全部872张验证和188张测试；GPU于北京时间08:04:40释放，早于08:55:33截止。测试T2 PSNR：A0=23.17500558、A2=23.15884248，A2−A0=−0.01616311 dB；验证差−0.04416503 dB。质量头有误差排序能力，但当前门控没有建立PSNR收益，两组继续训练也都低于起点。完整权重、逐图指标、置信原值与示例已归档并核对142项源文件哈希。见[完整结果与限制](../analysis/CONFIDENCE_OVERNIGHT_RESULTS_ZH_20261009.md)、[执行协议](../analysis/CONFIDENCE_OVERNIGHT_PROTOCOL_ZH_20261009.md)。首次缺少SciPy的失败保留；以下早期状态为历史记录。

2026-10-09设计阶段历史：直接T1两轮联合训练已完成追加10000步、872张验证及188张测试，见[完整结果](../analysis/DIRECT_T1_JOINT_RESULTS_ZH_20261008.md)。随后提出[文献支持的T1局部质量置信度设计](../analysis/CONFIDENCE_MECHANISM_DESIGN_ZH_20261009.md)，继承该完整两轮权重、保留增强与对齐；现已完成上述同预算实验。

2026-10-08更新：当前执行带增强的完整K1起步、直接T1反馈两轮联合训练，见[当前协议](../analysis/DIRECT_T1_JOINT_PROTOCOL_ZH_20261008.md)。用户要求每次实验必须完成全测试集评价，已记录于[项目要求](../AGENTS.md)。当前划分为训练5440、验证872、测试188；完整测试是实验完成条件。下文保留早期实现记录，其状态与默认配置不代表当前运行。

本目录是 K=1 实验的独立扩展。实现完成，尚未运行真实 PolarFree CUDA
preflight、200 步 sanity check 或正式训练。当前没有新的去反射性能结果。

## 已实现

- 原始混合 RGB 固定；R2P 和 TaskNet 两轮共享参数。
- 三层 TaskNet 特征反馈，48 通道投影后取均值，零初始化标量门控。
- 反馈特征不 detach；仅第一轮 alignment 目标 detach。
- 扩散先验参数冻结，但保留对输入的梯度；net_g 小学习率更新。
- 损失 `0.5 L(T1) + L(T2) + lambda align1`；每组独立重新校准 lambda。
- 两轮配对相同 Torch 随机流，控制扩散采样差异。
- 独立训练的 feedback / no-feedback 配置；后者跳过反馈输入。
- 按第二轮 scene-macro PSNR、SSIM、较早步数依次选 best。
- 同时保存逐帧两轮指标、两种聚合、迭代收益、实际 clamp 后偏振修改、
  R2P/net_g 漂移、反馈梯度；test 结果文件带 checkpoint 步数与权重哈希。
- 预检与 checkpoint 记录配置、代码、数据清单等摘要，拒绝无预检训练及协议变动后续训。

no-feedback 保留相同 adapter 参数初始化以匹配随机状态，但不使用这些参数；
其梯度为 None，不参与更新。新增参数为 18,433 个（含标量门控）。
每轮残差上界为 0.10，不代表两轮累计仍限制在 0.10。

## 本机检查

当前使用 `F:/3080ti/.venv-r2p/Scripts/python.exe`。

```powershell
python -m unittest discover -s tests -v
python run_experiment.py --help
```

7 项 CPU 测试通过：使用真实 PIDSR R2P 权重和小型随机任务替代网络。
覆盖第一轮一致性、零门控与无反馈等价、门控激活、完整反馈梯度、
alignment stop-gradient、checkpoint 恢复、实际修改量和预检/训练/续训流程。
这些检查不能替代真实 PolarFree GPU preflight，也不说明算法有效。

## 服务器迁移前需要提供/修改的路径

当前配置仍引用本机现有依赖，**本目录不是独立运行包**。
在服务器复制一份配置并修改：

1. `paths.source_root`：已有 `rgb_final_confirmation/src`，包含数据加载、官方模型适配器、损失和评估。
2. `paths.base_config`：基础 experiment.yaml；其内部 `official` 中的 PolarAPP DfP 源码、
   dm_config、PolarFree 官方三份 checkpoint 路径也必须改为服务器路径。
3. `paths.reference_r2p` 与 `paths.pretrained_r2p`：现有参考 R2P 源码与 PIDSR e42。
4. manifest、冻结 schedule、validation_proxy：完整复制；manifest 内每一行的图像绝对路径
   需映射到服务器数据目录，保持 sample_id、划分、裁剪 schedule 不变。
5. 两组 `output_root` 必须分开。所有路径在启动 preflight 前定稿。

复用现有官方环境依赖（PyTorch、torchvision、NumPy、PyYAML、Pillow、OpenCV、
OmegaConf，以及官方 PolarAPP/PolarFree 依赖）。GPU/CUDA 版本由服务器硬件决定。
官方感知损失需要的 VGG 权重也应预先准备；通过 TORCH_HOME 指定服务器缓存目录。
尚未生成服务器路径配置，也未宣称已验证服务器兼容性。

## 在目标机器上的执行顺序

以下示例使用当前配置名；服务器可通过 `--config` 指定迁移后的文件。

```powershell
python run_experiment.py --config config.yaml --device cuda:0 preflight
python run_experiment.py --config config.yaml --device cuda:0 train --steps 200
# 审查 200 步日志和验证结果后，从 200 续到总计 3000：
python run_experiment.py --config config.yaml --device cuda:0 train --steps 3000 --resume
```

独立 no-feedback 对照使用 `config_no_feedback.yaml`，执行相同顺序。
preflight 包含 4 批 lambda 校准及同一批的梯度、确定性和门控激活检查；
其中一次临时更新仅用于确认零门控开启后的 adapter 梯度，不保存到训练模型。
训练从 PIDSR e42 与官方 TaskNet 重新初始化。
预检会记录 GPU 峰值显存；OOM 时不得直接认为模型不可行或自动改协议。

固定设计、仅凭 validation 选模后，最终测试命令：

```powershell
python run_experiment.py --config config.yaml --device cuda:0 evaluate --checkpoint best --split test
# 如需固定步数比较，可指定 step_3000；不能用 test 决定改用哪个 checkpoint。
```

结果位于配置的 output_root。训练会写 RUN_PLAN.md；预检、校准和验证结果
分别位于 diagnostics、config、metrics。禁止混用 K=1 与 K=2 checkpoint。

## 解释与比较

需要同时比较独立 K1、K2 no-feedback、K2 feedback，以及 K2 内 T2-T1。
同 seed、同初始化来源、同样本 schedule、同训练步数和一致选模规则。
K2 的计算预算高于 K1；记录时间与显存，不宣称等计算量比较。
是否获益由真实验证/测试决定；偏振物理真实性与迭代收敛仍未得到证明。
不加入 dynamic RGB、额外物理损失、新 backbone 或 K3。

来源：现有 EXPERIMENT_HANDOFF.md、原 K=1 run_experiment.py 与用户确认的
两轮实验范围。这里记录实现协议，不把建议当作文献已证实结论。
