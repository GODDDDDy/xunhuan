# 当前两轮网络的 T1 局部质量置信度设计

**2026-10-09夜间执行更新：用户已授权8小时GPU预算，机制已实现并启动主方案和同预算直接T1对照，各目标追加16000步。最新执行状态、时间规则及失败记录以[夜间执行协议](CONFIDENCE_OVERNIGHT_PROTOCOL_ZH_20261009.md)为准；本文件保留原始科学设计与完整消融建议，尚无置信实验的正式完整测试结果。**

日期：2026-10-09。状态：**文献核查后的可实现设计，尚未写入训练网络、尚未运行置信度实验**。本文件接替 10 月 8 日的早期置信路线草稿；初始化、监督和消融以本文件为准。

## 1. 设计结论与已有证据

在当前“偏振增强 E → PolarFree D → T1 → 第二轮 E/D”的框架中增加一个轻量置信头 H。H 根据第一轮结果、任务特征及原始偏振观测，估计 T1 的局部绝对恢复误差，再转换成质量置信图 C1。C1 控制第二轮的反馈条件；其互补量 1−C1 控制第二轮偏振更新幅度。保留已经训练的增强模块、PolarFree、第一轮特征对齐及两轮联合训练。

这是本项目的待检验方案。文献支持“可靠性估计可以辅助信息传播”，但没有直接证明这套双重门控在本项目有效，也没有证明低置信位置一定可以被第二轮修好。

当前直接 T1 联合训练已完成：完整带增强 K1 起步、追加 10000 步，验证 872 张、测试 188 张。测试图像平均 PSNR：T1=23.37858336，T2=23.36154630，T2−T1=−0.01703706 dB；验证 T2−T1=−0.01118809 dB。结果尚未建立稳定迭代收益，不能仅凭这组均值断言原因就是错误反馈传播。[完整结果（本地资料）](../../../../docs/LOCAL_ARTIFACTS.md#artifact-0052)

## 2. 文献依据及迁移边界

以下均已核对原始论文正文，并在本项目保存 PDF、文本和关键页面图像。来源及 SHA256 见 [sources.json（本地资料）](../../../../docs/LOCAL_ARTIFACTS.md#artifact-0053)。

|原始工作|论文直接支持的内容|本设计借鉴与不能直接推出的结论|
|---|---|---|
|Dong et al., *Location-Aware Single Image Reflection Removal*, ICCV 2021，5017–5026，[CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Dong_Location-Aware_Single_Image_Reflection_Removal_ICCV_2021_paper.html)|反射检测模块产生 reflection confidence map，以区域反射/透射主导信息控制特征流，并在循环网络中逐步恢复；见正文 §3、Fig.2–3。|与当前课题最直接相关，因此“置信图＋循环去反射”已有先例。本文的图估计 **T1 的恢复质量**，并非复制其反射主导图。其合成与重组损失依赖论文成像模型，不直接移植到本项目。|
|Yasarla & Patel, *Uncertainty Guided Multi-Scale Residual Learning-Using a Cycle Spinning CNN for Single Image De-Raining*, CVPR 2019，8405–8414，[CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Yasarla_Uncertainty_Guided_Multi-Scale_Residual_Learning-Using_a_Cycle_Spinning_CNN_for_CVPR_2019_paper.html)|多尺度残差恢复中估计 confidence，并将其用于残差信息传播；置信加权误差及 log-confidence 项共同训练，见 §3.2–3.3、Eq.(3)。|支持按估计可靠性选择恢复反馈。任务是去雨，不能将其性能增益直接当作偏振去反射的证据；本设计也不照搬其置信加权主损失。|
|Eldesokey, Felsberg & Khan, *Confidence Propagation through CNNs for Guided Sparse Depth Regression*, TPAMI 42(10)，2423–2436，2020（online 2019），[DOI:10.1109/TPAMI.2019.2929170](https://doi.org/10.1109/TPAMI.2019.2929170)|归一化卷积传播置信信息，并以输出置信度辅助后续 RGB 引导融合；见 §4.2、§5、§7.7。|支持把可靠性作为后续网络的辅助信息。其稀疏深度置信度具有采样/传播背景，不能直接等同于 RGB 恢复质量概率。|
|Eldesokey et al., *Uncertainty-Aware CNNs for Depth Completion: Uncertainty from Beginning to End*, CVPR 2020，12014–12023，[CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Eldesokey_Uncertainty-Aware_CNNs_for_Depth_Completion_Uncertainty_from_Beginning_to_End_CVPR_2020_paper.html)|区分输入置信度与输出不确定性；Fig.1 指出先前传播的输出置信度可能与实际误差弱相关，并引入概率误差模型训练输出不确定性，见 §4。|要求明确监督语义，并检验预测尺度与实际误差的关系。本设计的正误差尺度损失是针对当前任务的改写，不是该论文的 Gaussian NLL 原式，也不据此宣称获得校准概率。|
|Yao et al., *PolarFree: Polarization-based Reflection-Free Imaging*, CVPR 2025，10890–10899，[CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Yao_PolarFree_Polarization-based_Reflection-Free_Imaging_CVPR_2025_paper.html)|四角度观测构建 Stokes、DoLP、AoLP；透射与反射的角度相关成像均进入其模型；见 §4.1、Eq.(2)–(5)。|支持将偏振量作为观测线索；**不支持 DoLP 直接等于反射比例或 T1 正确概率**。当前 LDR 预处理数据也不足以保证严格的辐射定标物理约束。|

综合判断：优先设计“有明确质量监督的置信反馈”，而不是把 DoLP 做阈值后称作置信度，或只加一张无质量约束的注意力图。

## 3. 网络组成与完整数据流

记 I 为原混合 RGB，P0 为原始四角度 RGB 偏振（12 通道），E 为 R2P 增强，D 为包含冻结先验的 PolarFree，T 为透射预测。E 和 D 两轮共享参数；D 两轮仍以原 I 为 RGB 输入。

```text
P0、I ── E ── P1 ── D(P1,I) ── T1、F1
                          │          │
         原始 h(P0)、I ────┴──── H ── C1

Φ(T1)、Φ(I) ── C1 加权融合 ── Z2
P1、Z2 ── 共享 E ── 候选 P̂2
P1 + (1−C1)·(P̂2−P1) ── P2 ── 共享 D(P2,I) ── T2
```

|模块/机制|处理方式|
|---|---|
|偏振增强 E|继承当前已训练 R2P；保留 RGB stem、偏振 stem、内部融合门、FiLM 和最大 0.10 的残差头。|
|PolarFree D|继承当前 net_g；先验参数继续冻结并保留输入梯度；两轮 net_g 联合更新。|
|第一轮特征对齐 A|原有三组 1×1 投影为 48→96、48→96、48→192；将同一 R2P fused map 投影/缩放后对齐任务特征，任务目标 detach；不重新初始化。|
|置信头 H|新增、轻量；预测第一轮局部误差尺度。其输入和监督目标 stop-gradient。|
|置信反馈融合|复用 E 的 RGB 编码器 Φ；融合后的条件同时进入 E 的内部融合门和 FiLM。|
|偏振更新门控|以 1−C1 缩放第二轮的实际偏振增量；没有额外可学习恢复网络。|
|固定偏振表示 h|从原始 P0 计算，提供观察依据；不改变 PolarFree 的官方输入预处理。|

原先的内部任务特征反馈 adapter 仍不启用；F1 仅为置信头提供辅助特征。保持两次 E、两次 D，不增加 T0、第三轮、ConvLSTM 或多次随机采样。

## 4. 置信图究竟表示什么

定义 C1(x)∈(0,1) 为 T1 在位置 x 附近的**局部质量分数**：预测局部误差越小，C1 越高。分数来自误差尺度的单调变换；未经概率校准，不解释为“该像素正确的概率”。

它不是反射强度图，不是偏振测量有效性图，也不直接预测“第二轮能改善多少”。即使 T1 不可靠，第二轮也可能同样不可靠；本机制只是提供反馈选择与修正幅度的依据。

### 4.1 偏振侧输入：提供线索，而非硬判决

沿用已有灰度转换器，在独立的 H 侧支路读取原始四角度 P0：

\[
S_0=(g_0+g_{45}+g_{90}+g_{135})/2,\quad
S_1=g_0-g_{90},\quad S_2=g_{45}-g_{135}.
\]

\[
q=S_1/\max(S_0,\epsilon_p),\quad
u=S_2/\max(S_0,\epsilon_p),\quad
\rho=\operatorname{clip}(\sqrt{q^2+u^2},0,1).
\]

采用 h(P0)=[S0/2, ρ, clip(q,−1,1), clip(u,−1,1), V]，共 5 通道。εp=10⁻³ 是归一化强度上的数值保护。V 为四个灰度观测中处于 (2/255,253/255) 的比例，用作暗部/饱和的粗提示；阈值是本设计的工程初值，不是论文证明的传感器噪声或有效性界限。

采用 q/u，避免把低 DoLP 下不稳定且存在角度周期的 AoLP 直接送入第一版。V 仅作为学习输入，不能硬乘到 C1 上：偏振弱或饱和并不必然意味着 RGB 的 T1 错误。

H 的 q/u 保留真正的零差值；不改动现有官方任务预处理的数值保护与量化路径。当前输入不是已证明经过辐射定标的 RAW，因此 h 是“偏振观测衍生特征”，不宣称它给出精确反射能量或物理真实性证明。

### 4.2 H 的输入与轻量结构

H 使用以下信息：I、clamp(T1,0,1)、|I−clamp(T1)|、h(P0)，以及第一轮任务的三个 decoder 特征 F1^(1/2/3)。|I−T1| 只表示观测与预测的差异，不能当作真实反射标签。

先统一到图像 1/4 分辨率。RGB 与 h 用 4×4 平均池化；三层任务特征先缩放至该网格，各以 1×1 Conv 压到 8 通道。适配器的任务特征通道为 96/96/192；实际坐标需沿用现有完整图像评价路径，若有 padding 则先恢复一致有效区域。

```text
RGB/偏振侧：3 + 3 + 3 + 5 = 14 通道
任务特征侧：8 + 8 + 8 = 24 通道
拼接 38 通道
Conv3×3(38,32) → GELU → Conv3×3(32,32) → GELU → Conv1×1(32,1)
```

所有卷积带 bias，无 BN/Dropout。新增投影及 H 合计 **23,353 个参数**；与现有 alignment 投影不同，不覆盖其权重。H 在 1/4 分辨率预测 b1=10⁻³+softplus(s1)，随后计算 C1=exp(−b1/τ)，再双线性上采样用于门控。τ 为固定误差尺度，见下节。

四分之一分辨率有助于成本控制及抑制单像素标签噪声，但可能模糊细小反射边缘；边缘区域须单独检查，不能预设该尺度最佳。

## 5. 训练监督、校准与梯度路径

### 5.1 有 GT 的训练阶段

仅训练时，构造第一轮局部误差目标：

\[
e_1=\operatorname{stopgrad}\!\left[
\operatorname{AvgPool}_{4\times4}
\left(\frac13\sum_{c\in RGB}|\operatorname{clamp}(T_1)_c-T_{GT,c}|\right)\right].
\]

用误差尺度目标训练 H：

\[
L_{conf}=\operatorname{mean}\left(e_1/b_1+\log b_1\right).
\]

这是本项目采用的 Laplace 形式误差尺度代理目标，不是引用论文中的 Gaussian 目标原式。固定条件下，对期望目标求导得到最优 b=E[e1|输入]（有正下限时截断）；这是数学解释，不能保证有限数据、有限网络训练出的 b 已准确或校准。

设置正下限避免零误差下 log 项无界。该损失可能为负，不代表训练异常。GT 不作为 H 的输入；验证/测试只能用于事后核验，推理完全不读取 GT。

### 5.2 固定分数尺度 τ

从共同起点检查点的 T1 上，用固定 seed 的 **256 个训练 crop** 估计 e1 的中位数 m，设 τ=max(m/ln2,10⁻³)。除下限触发外，中位误差对应 C≈0.5。保存 crop 索引、统计量与 τ；所有消融共享同一 τ，不依据测试调节。

这是训练前的标签尺度统计，不是额外的短步性能实验。头的最终卷积权重初始化为零，bias 使 b 初值为 max(m,2×10⁻³)；只有新 H 初始化，已有 E/A/D 完整继承。

### 5.3 防止置信度退化成任意注意力

- H 输入中的 T1、F1、I、h 全部 detach；e1 也 detach。
- C1 在进入反馈和更新门控前 detach；主恢复损失不反向改写 H。
- **T1 进入 E 的反馈内容不 detach**，仍保留第二轮损失经 T1 到第一轮的训练路径。
- 主恢复损失不乘 C1，不允许网络通过降低置信度屏蔽难区域损失。
- 三组 alignment 目标沿用原 detach；H 的 detach 不替代或关闭 alignment。

H 与恢复网络共同训练，但 H 的参数只由质量监督更新。这是可解释性优先的设计取舍；失去主任务对 H 的直接优化可能限制收益，需要如实报告。

保持已有恢复目标：Ltask=L1+(1−SSIM)+0.1 Lphase+0.1 LVGG；两轮权重 [1/3,2/3]；第一轮 alignment 系数 0.030651881051471687。总目标增加 Lconf，系数暂定 1。

H 用独立 Adam 参数组/优化器，LR=10⁻⁴、独立 clip-norm=1；E 与已有对齐头 LR=10⁻⁴，net_g LR=10⁻⁵，各沿用原独立裁剪。由于梯度隔离，Lconf 不影响恢复组的损失梯度和裁剪；不能把 H 加进 E 的统一裁剪组而悄悄改变基线更新。

## 6. C1 如何进入第二轮

### 6.1 反馈内容：高置信信任 T1，低置信保留观测

\[
Z_T=\Phi(\operatorname{clamp}(T_1,0,1)),\qquad Z_I=\Phi(I),
\]
\[
Z_2=C_1\odot Z_T+(1-C_1)\odot Z_I.
\]

Φ 是当前 E 的 RGB stem，权重共享；C1 为单通道并广播至 48 通道。候选第二轮增强由 E 的同一 polar encoder、融合 gate、FiLM 和 body 计算。**Z2 必须同时代替内部融合门与 FiLM 的 RGB 条件**；只改一个入口会留下未被置信机制控制的 T1 路径。

低置信区域回退 I 是保留原始观测的假设，并非认为混合 I 已去除反射。P1 同时保留第一轮状态，网络仍需学习如何使用这些信息。

### 6.2 修正幅度：保护已有结果

先得到原 E 的候选输出 P̂2=E(P1;Z2)，再计算：

\[
P_2=P_1+(1-C_1)\odot(\hat P_2-P_1),\qquad T_2=D(P_2,I).
\]

这里缩放的是经过原残差限制和 clamp 后的实际增量。P2 是两个有界偏振输入的凸组合，保持 [0,1]；逐通道 |P2−P1|≤0.10(1−C1)。这些是代数/结构性质，**不代表 T2 误差必然下降，也不保证增强偏振满足成像物理**。

整个 C1=1 且两次 D 使用相同随机流时，P2=P1，T2=T1；局部 C1=1 仅保证该处偏振不变，PolarFree 的卷积/注意力仍可能让该处 RGB 随邻域改变。不能宣称实现逐像素 RGB 锁定。

第一版最终输出仍为 T2，不再增加 T1/T2 的 RGB 输出混合，便于分离机制来源。

### 6.3 从当前模型平滑接入

正式训练前 500 步令 γ 从 0 线性升到 1，之后固定：

\[
C_f=1-\gamma(1-C_1),\quad
Z_2=C_f Z_T+(1-C_f)Z_I,\quad
w_2=1-\gamma C_1,\quad
P_2=P_1+w_2(\hat P_2-P_1).
\]

γ=0 精确回到当前直接 T1 条件、完整第二轮偏振更新；γ=1 为上述完整设计。500 是预先固定的接入日程，不是文献证明的最优值；不能测试后再择优改变。反馈单独消融只启用 Cf，w2 始终为 1。

## 7. 初始化与公平比较

本次设计默认**从最新已经完成的、带偏振增强与对齐的直接 T1 两轮检查点起步**：

- 来源：[step_10000.pt（本地资料）](../../../../docs/LOCAL_ARTIFACTS.md#artifact-0054)。
- SHA256：`eb17a5537bcd26583b73670a34a258f948056935f851b42c7a44eb62e1dd15ea`。
- 同时继承 R2P、三组 alignment 投影及 net_g；不从 G6 单独起步，不清零增强残差头。
- 所有正式组使用相同完整检查点、fresh Adam、相同 crop/sample schedule、配对扩散随机流，再追加 10000 步；crop128、batch2、workers2 沿用现有协议。
- 加载后的历史链为 K1 3000＋直接 T1 10000＋本阶段 10000；新阶段结束时累计历史更新 23000。日志须区分新阶段步数和历史步数。

因此必须有同样追加训练预算的直接 T1 对照 A0。10 月 8 日结果只提供历史参照，不能充当本阶段的等预算对照。该对照是正式实验，不是重新做 200 步的简单性能验证。

## 8. 精简且必要的正式实验

|编号|方案|对应问题|
|---|---|---|
|A0|直接 T1 反馈、完整第二轮更新；继续训练同样 10000 步|置信版本是否优于仅增加训练预算？|
|A1|有质量 C1，仅做反馈条件融合；更新幅度不门控|质量选择能否改善反馈使用？|
|A2|有质量 C1，反馈融合＋互补更新门控，主方案|加入保守更新是否产生净收益？|
|A3|与 A2 相同，但门控用每张图 C1 的空间均值广播|收益是否只是统一缩小第二轮更新，而非空间质量选择？|
|A4|与 A2 相同，H 的 5 个偏振侧通道置零|偏振线索是否给置信估计带来额外贡献？|

A4 只去掉 **H 的偏振输入**，E/D 仍使用偏振，不能称作无偏振模型。A3 保留同一个空间 H 及其质量训练，仅改变门控使用；A0 也运行同样 H 训练但不将分数用于主网络，以匹配新增辅助参数/计算且不影响恢复梯度。固定同一初始化策略、τ、主损失、优化器裁剪与评价协议；组间使用匹配随机流，H 的初始化/校准不能扰动恢复网络的采样 schedule。

本表 A0–A4 仅是新阶段实验标签，不能写入已有数据加载器的 `group`：已有数据器的 A0 表示偏振均值条件、A1 才表示原混合 RGB。各组继续保持现有原 RGB 条件配置，禁止因标签重名而顺带更换第一轮输入。

先完整完成共同 seed2026 的正式组；如显示可靠收益，再用多个训练 seed 检查稳定性。同一已有 warm-start 下更换 seed 只衡量后续训练随机性，不能伪装为完整预训练过程的独立重复。

每个正式组、每个 seed 都必须完成 **872 张验证及全部 188 张测试**。按事前固定的新阶段 step10000 检查点评价；T1/T2 均报告逐图 PSNR/SSIM、image mean、scene macro 和 T2−T1；保留检查点、清单哈希及失败记录。测试不用于 τ、超参数、日程、步数或检查点选择；已有测试集被多次查看，仍标记为探索背景。

除恢复指标外，至少核查：

1. C1 与局部 e1 的对应关系：固定十个分数区间的平均真实误差、预测 b/实际误差对照；全局与逐场景报告，并检查暗部及边缘。
2. 排序是否合理：C1 与 e1 的 Spearman 相关预期为负，b 与 e1 预期为正；不能用分数图的视觉效果代替核验。
3. 分区域迭代结果：高/低置信区域实际偏振更新量、T2−T1 局部误差变化，是否发生低置信区域继续恶化；分区只用于评价，不用于测试调参。
4. 额外成本：参数、完整图像耗时、训练/评价峰值显存，不能只凭 2.3 万参数认定显存开销可忽略。

结果解释应区分：C1 未学到质量；C1 能排序但反馈无收益；主要收益来自统一缩步；偏振输入无额外作用；或质量选择确有净收益。若 A2 只是使 T2 接近 T1，可称减少第二轮损害，不能称已建立有效迭代提升。

## 9. 实现落点与 12GB 显存约束

复用现有 `server_scripts/direct_t1_joint_20261008/run_direct_t1.py` 与评价器，只增添 H、共享 RGB 条件特征接口和两处门控。H 的低分辨率投影不同于 alignment 头；alignment 损失与加载路径保持可追踪。

现有扩展的初始化断言只接受源 K1 step3000/一轮；新阶段必须显式改为读取这里指定的 step10000/两轮源及其哈希，不能原样套用旧断言。优先严格加载原 E/A/D 后再挂载新 H；若按含 H 的新模型加载，则缺失键只允许新增 H 的明确白名单，不用未经核查的 `strict=False` 掩盖旧模块漏载。检查点记录新方法、质量头权重、τ、γ 日程、新阶段与历史步数，并将旧协议归档保留。

伪代码（说明梯度关系，不代表已实现）：

```python
first = E(I, P0)                              # 原有增强和对齐
T1, F1 = D(first.refined_polar, I, rng=paired)
b_low = H(detach(I), detach(clamp(T1)),
          detach(h(P0)), detach(F1))
e_low = detach(pool4(mean_rgb(abs(clamp(T1) - GT))))  # 仅训练
Lconf = mean(e_low / b_low + log(b_low))
C = detach(upsample(exp(-b_low / tau)))
del F1                                        # 在此之前已保存所需对齐损失
ZT, ZI = E.rgb_encoder(clamp(T1)), E.rgb_encoder(I)
Cf = 1 - gamma * (1 - C)
candidate = E.from_rgb_feature(P1, Cf * ZT + (1 - Cf) * ZI)
P2 = P1 + (1 - gamma * C) * (candidate - P1)
T2, _ = D(P2, I, rng=paired)
L = (Ltask(T1, GT) / 3 + 2 * Ltask(T2, GT) / 3
     + lambda_align * Lalign_first + Lconf)
```

H 在第一轮特征被现有代码 clear 之前运行；得到低分辨率表征后及时释放不再需要的 Python 引用。训练仍可能因原任务梯度保留相应激活，不能将 del 误当作全部释放显存的保证。完整图像评价保留既有梯度释放、Adam 状态临时 CPU offload、CPU 加载检查点等已验证处理。

原 I 的 RGB stem 特征可在第二轮重算，避免为了省一次浅层编码而在第一轮 PolarFree 期间长期保留 48 通道大图特征。不增加第三次 PolarFree 调用；新头和门控峰值仍需真实测量。

必要的启动核查仅验证 γ=0 与现有输出一致、全 C=1 的同随机流端点、主损失不更新 H、Lconf 不更新 E/D、第二轮反馈内容梯度仍存在及无 GT 推理。这些用于排除实现错误，不据短步指标判断科研收益。

## 10. 可提出的贡献与尚未成立的部分

可探索的贡献是：在偏振增强—去反射循环中，以第一轮任务质量为语义、原始偏振为辅助观测，分别控制反馈内容和后续偏振修正幅度，并在同预算实验中核验质量与性能的关联。

“首次用 confidence map 做循环去反射”不成立，ICCV 2021 已有直接相关工作。“物理置信度”“校准概率”“保证迭代收敛”“低置信区域必然改善”也不由当前设计支持。是否形成足够新颖的贡献还需更完整的相关工作检索及上述实证，不能仅以模块组合或命名确认。

## 参考文献与本地原文

1. Zheng Dong, Ke Xu, Yin Yang, Hujun Bao, Weiwei Xu, Rynson W. H. Lau. *Location-Aware Single Image Reflection Removal*. ICCV, 2021, pp. 5017–5026. [官方论文](https://openaccess.thecvf.com/content/ICCV2021/papers/Dong_Location-Aware_Single_Image_Reflection_Removal_ICCV_2021_paper.pdf)；[本地 PDF（本地资料）](../../../../docs/LOCAL_ARTIFACTS.md#artifact-0055)。
2. Rajeev Yasarla, Vishal M. Patel. *Uncertainty Guided Multi-Scale Residual Learning-Using a Cycle Spinning CNN for Single Image De-Raining*. CVPR, 2019, pp. 8405–8414. [官方论文](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yasarla_Uncertainty_Guided_Multi-Scale_Residual_Learning-Using_a_Cycle_Spinning_CNN_for_CVPR_2019_paper.pdf)；[本地 PDF（本地资料）](../../../../docs/LOCAL_ARTIFACTS.md#artifact-0056)。
3. Abdelrahman Eldesokey, Michael Felsberg, Fahad Shahbaz Khan. *Confidence Propagation through CNNs for Guided Sparse Depth Regression*. IEEE TPAMI, 42(10), 2020, pp. 2423–2436; online 2019. DOI: [10.1109/TPAMI.2019.2929170](https://doi.org/10.1109/TPAMI.2019.2929170)；[作者机构 postprint](https://www.diva-portal.org/smash/get/diva2%3A1362784/FULLTEXT01.pdf)；[本地 PDF（本地资料）](../../../../docs/LOCAL_ARTIFACTS.md#artifact-0057)。
4. Abdelrahman Eldesokey, Michael Felsberg, Karl Holmquist, Michael Persson. *Uncertainty-Aware CNNs for Depth Completion: Uncertainty from Beginning to End*. CVPR, 2020, pp. 12014–12023. [官方论文](https://openaccess.thecvf.com/content_CVPR_2020/papers/Eldesokey_Uncertainty-Aware_CNNs_for_Depth_Completion_Uncertainty_from_Beginning_to_End_CVPR_2020_paper.pdf)；[本地 PDF（本地资料）](../../../../docs/LOCAL_ARTIFACTS.md#artifact-0058)。
5. Mingde Yao, Menglu Wang, King-Man Tam, Lingen Li, Tianfan Xue, Jinwei Gu. *PolarFree: Polarization-based Reflection-Free Imaging*. CVPR, 2025, pp. 10890–10899. [官方论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Yao_PolarFree_Polarization-based_Reflection-Free_Imaging_CVPR_2025_paper.pdf)；[本地 PDF（本地资料）](../../../../docs/LOCAL_ARTIFACTS.md#artifact-0059)。
