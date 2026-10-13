# 偏振增强模块：当前结构、任务定位与重构建议

后续已按用户最终约束实施“同分辨率PIDSR纠偏预训练＋唯一最新置信反馈方案”，见 [实施协议](POLAR_ENHANCER_V2_IMPLEMENTATION_PROTOCOL_ZH_20261009.md)。本文保留为早期分析记录，以下提案不代表全部最终采用。

日期：2026-10-09。状态：代码与文献核查后的设计建议，未更改现有实验代码、权重或启动新训练。GPU夜间预算已完成并释放。本文件回答当前增强模块是否仍适合两轮去反射任务。

## 判断及证据边界

建议重构增强模块的主干、条件接口和输出表示，使其成为“偏振观测主导、任务状态引导的去反射适配器”。这仍属于原课题的技术路线调整。

用户对RGB条件路径较多的记忆有依据，但原模块并非仅以RGB生成偏振：它始终接收真实四角度偏振，预测相对于该输入的残差。结构存在两个RGB作用入口，不等于实验已经证明RGB压倒了偏振；gate值、分支贡献和因果效应未在本轮新增核验。已有负面结果也不能唯一归因于FiLM或架构。

## 当前实际执行的模块

已核对服务器 `pidsr_k2_929/deps/r2p_film.py` 与本地留存文件相同，SHA256：`6c9f4bed76046e06a4c44361ace0dd4b4d1e3d08015336f38230c2ddb62ed7e5`。当前JointR2P严格载入的是此轻量模块，不是早期另一个带CMX/窗口注意力的ConditionalR2P，也不是PIDSRNet。

本地代码：[r2p_film.py](../_dl/state_condition_probe_20261007/artifacts/analysis/reused_source/pidsr_k2_929/deps/r2p_film.py)。当前训练包装：[JointR2P快照](../_dl/confidence_joint_20261009_v2/artifacts/scripts/reference/run_experiment_snapshot.py)。置信版本：[confidence.py](../server_scripts/confidence_joint_20261009/confidence.py)。

|部分|实际结构/作用|
|---|---|
|输入|12通道四角度彩色偏振P，3通道条件图X|
|两支编码|P：12→48→48；X：3→48→48，各为两层3×3卷积＋GELU|
|融合gate|由拼接的偏振/RGB特征预测48通道空间门，融合g·Fx+(1−g)·Fp|
|FiLM|RGB特征独立预测γ、β，形成Fp·(1+0.1tanhγ)+0.1tanhβ|
|残差主干|拼接上述两路，96→48→48→12，三层3×3卷积|
|输出|clamp(P+0.10tanh(r),0,1)|

参数125004；无下采样/多尺度主干。五层3×3卷积路径对应理论空间感受野11×11，1×1门控/FiLM不扩大空间范围。这是源代码推导，不是已实测瓶颈。第一轮三个alignment投影来自同一fused特征再匹配任务特征尺寸，不能称增强网络本身已有三个尺度的编码器。

FiLM系数有0.10限制，不能笼统说其调制无限强。融合gate允许逐位置/通道侧重RGB特征；它没有“判断RGB恢复是否正确”的质量监督。RGB既进入融合内容又产生FiLM系数，因而两条入口都受条件分布影响。FiLM本身是通用条件仿射变换，不是物理偏振推断机制。[FiLM，AAAI2018](https://ojs.aaai.org/index.php/AAAI/article/view/11671)

## 原目标与当前目标的差异

交接材料记载的预训练任务是：PIDSR退化稠密偏振＋其角度平均强度引导→干净稠密偏振，并有偏振真值、Stokes和偏振量监督。其rgb接口实际使用(I0+I45+I90+I135)/4，不是独立普通RGB照片。依据：[原始交接README（本地资料）](../docs/LOCAL_ARTIFACTS.md#artifact-0029)、[训练报告（本地资料）](../docs/LOCAL_ARTIFACTS.md#artifact-0030)。这些是项目历史记录，不把旧报告“无过拟合”的判断视作本次重新验证。

当前任务是：利用真实混合偏振观测，让PolarFree输出更准确的RGB透射层。当前数据加载器只给RGB透射监督，不给增强后的透射偏振真值。任务适配输出不能自动被称为“恢复了真实干净透射偏振”。PIDSR原论文的去马赛克/超分及对应偏振监督也不等于本模块的去反射任务目标。[PIDSR，CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_PIDSR_Complementary_Polarized_Image_Demosaicing_and_Super-Resolution_CVPR_2025_paper.html)

直接反馈两轮中，原RGB和T1先后进入同一rgb_encoder。两者语义不同：前者是混合观测，后者是带误差的透射估计；现有接口没有分别声明其角色。网络可能从数据学习这种区别，但没有机制或结果保证它已学会。最新置信版将两种RGB特征按C融合后，同时进入原gate和FiLM，仍保留这一主干。

最新完整实验中，A2有质量排序能力（测试逐图空间Spearman均值约−0.473），但T2测试PSNR比同预算A0低0.01616311 dB，验证低0.04416503 dB，低置信三个区间的T2局部MAE也未优于T1。这支持重新审视“如何生成修正”，并不证明所有置信机制无效或已经定位RGB调制为原因。两组均比起点下降，训练方案的共同因素仍需区分。[完整结果](CONFIDENCE_OVERNIGHT_RESULTS_ZH_20261009.md)

## 一个直接影响重构的接口事实

当前PolarFree适配器把每角度RGB转换为单通道灰度，再计算AoLP/DoLP，返回：

`[固定原RGB, g0, g45, g90, g135, AoLP, DoLP]`。

因此增强器目前生成12通道，但下游真正接收的是四角度灰度及由它们派生的量。沿用12通道输出不是接口必需。忽略clamp/量化、只看亮度线性变换时，每角度RGB→灰度有两个不可见的局部色彩方向；这说明输出表示值得收紧，不证明所有12通道设计必然无效。彩色偏振仍可以作为编码输入提供结构/光谱线索。

本地实现：[polarfree.py](F:/AAAketi/rgb_final_confirmation/src/rgb_final_confirmation/polarfree.py)。服务器该文件SHA256为`80d7bb81b945610f387ea04a7cdd2cbffb8a621a80d5a4228ce64dcbf5b65fe1`；当前正式评价使用匹配官方OpenCV量化前向和STE代理梯度的版本。

## 建议的完整模块

### 1. 偏振承担内容主干

原始P0始终保留；另输入当前四角度灰度Gt、Gt−G0，以及由角度差分得到的S0/S1/S2、稳定归一化q/u和可用区域标记。DoLP是辅助观测，不能直接等同反射比例或修复可靠性；不必将有周期边界的裸AoLP角值作为主表示。

采用小型三级编码器—解码器，候选宽度24/48/96，尺度1、1/2、1/4，保留偏振跳接。宽度是待评估配置，不是已验证最优值；不预报未经实现测量的参数量/显存。多尺度上下文与高分辨率细节的组织参考MPRNet；迁移到本任务的收益是研究假设。[MPRNet，CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Zamir_Multi-Stage_Progressive_Image_Restoration_CVPR_2021_paper.html)

### 2. RGB观测和任务反馈分别声明

固定I作为结构/混合观测条件；上一轮Tt、任务特征Ft作为可有误差的任务状态。第一轮反馈缺省并有显式可用标记，第二轮接入T1/F1，避免让网络把I和T1当成同一种RGB条件。

保留偏振主干的恒等路径，在低分辨率层用小型投影后的任务状态产生有界、可学习的残差引导；原RGB条件也通过这一明确接口参与更新。移除旧“凸混合替换偏振特征＋RGB独立FiLM”双入口结构。不等于彻底删RGB：偏振物理观测和RGB/任务状态各有角色。

输出初始为零可使新增修正初始为恒等；有界引导不保证训练中永远有效。是否需要任务引导、是否仍存在RGB主导，都要通过匹配训练消融和实际分支响应确认。

### 3. 输出优先匹配下游偏振表示

建议最终预测“强度修正”和“角度差分修正”，而非12个互不受约束的彩色残差。按当前代码约定S1=g0−g90、S2=g45−g135，可用三个残差δS0、δS1、δS2构成：

```
δG0   = (δS0 + δS1)/2
δG45  = (δS0 + δS2)/2
δG90  = (δS0 - δS1)/2
δG135 = (δS0 - δS2)/2
```

这里的“Stokes基底”是结构化更新参数化。对零残差而言原始G0逐元素保留，不把观测强行投影为理想Stokes测量。更新前的正交对和差若非零，该残差并不会主动消除它；clamp和量化也可能破坏更新关系，须分别记录原始/有效残差。这不能称严格物理恢复。

可先保留四灰度输出作为表示对照，再比较三分量输出，区分主干改造与参数化约束的作用。匹配下游接口时复用现有灰度→AoLP/DoLP计算，核对零更新前向、量化和代理梯度；不能随意移除官方预处理并将收益归给增强结构。

Stokes表示及结构化偏振处理有原始研究依据，但具体任务、损失与数据不能照搬。[Learning to Deblur Polarized Images，作者公开稿](https://arxiv.org/html/2402.18134)假定线性相机响应，并使用相应偏振监督。当前8位数据的响应、标定、饱和与去马赛克误差尚未支持硬物理约束，故初版不加“必须保留原DoLP/AoLP”损失，也不声称满足Stokes关系就保证透射恢复正确。

### 4. 每轮回看原始观测，估计相对它的修正

使用`G_next = clamp(G0 + bounded_residual(G0, Gt, I, Tt, Ft), 0, 1)`，让第二轮可重估/撤回第一轮修改，而不是只能在Gt上继续累加相同幅度残差。这限制相对原始观测的累计漂移，不保证两轮误差单调下降。

与当前外循环兼容：

`P0/I → 新E → G1 → PolarFree(I,G1) → T1/F1 → 新E(原观测,当前G1,任务状态) → G2 → PolarFree(I,G2) → T2`。

这仍是两次E、两次D，不为了获得第一轮任务状态额外增加第三次D。跨轮ConvLSTM暂不作为这个模块重构的必需组成：显式输入当前状态与观测已经可形成完整循环；若需记忆再独立检验，避免把所有机制同时归因于一个模块。

## 置信度、对齐与训练继承

保留H作为质量估计的独立接口，但不默认沿用硬编码`1−C`乘全部修正的规则。C描述当前误差，并不同时提供正确修正方向。第一版重构主干的比较应在相同置信设置下进行，之后以配对有/无C比较它是否改进新的修正过程。

特征对齐是训练目标，任务特征回传是前向输入，两者不同。三级新E可提供真实分辨率层级进行投影，但现有三处余弦对齐的收益并未全面成立；不能把重构后对齐“自然有效”当事实。主干比较应明确对齐设置，避免同时换结构和目标却声称仅由结构带来改善。

可以复用已有PolarFree/net_g权重。新E结构与输出改变后，不能strict继承旧125004参数模块，并称整套新系统与旧强模型起点相同。零新输出只保证原始观测旁路，不保证复现旧E+D的输出。若需要保留强系统行为，可用训练集上的旧E灰度输出做初始化蒸馏，或先保留旧E并训练独立修正分支；二者各自有额外训练预算/残留旧设计的限制，应分开命名。此处不擅自选择某个测试表现较好的检查点作新实验起点。

正式比较应预先锁定初始化与选步规则，保留旧E同预算对照和直接PolarFree同预算对照；每个正式实验/seed/消融都完成全部872验证与188测试，逐图/场景汇总、T1/T2、迭代差齐全。测试不决定超参数或中间权重，探索背景如实保留。当前讨论未启动实验。

## 参考入口与迁移边界

1. Ethan Perez等，FiLM: Visual Reasoning with a General Conditioning Layer，AAAI2018，DOI [10.1609/aaai.v32i1.11671](https://doi.org/10.1609/aaai.v32i1.11671)。支持条件仿射变换的定义，不证明偏振去反射有效。
2. Syed Waqas Zamir等，Multi-Stage Progressive Image Restoration，CVPR2021，14821–14831，[官方论文](https://openaccess.thecvf.com/content/CVPR2021/papers/Zamir_Multi-Stage_Progressive_Image_Restoration_CVPR_2021_paper.pdf)。支持上下文/细节与阶段交互设计；新E不是原网络复现。
3. Shuangfan Zhou等，PIDSR: Complementary Polarized Image Demosaicing and Super-Resolution，CVPR2025，16081–16090，[官方论文入口](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_PIDSR_Complementary_Polarized_Image_Demosaicing_and_Super-Resolution_CVPR_2025_paper.html)。支持区分偏振恢复目标与下游任务适配。
4. Chu Zhou等，Learning to Deblur Polarized Images，IJCV2025，DOI [10.1007/s11263-025-02459-7](https://doi.org/10.1007/s11263-025-02459-7)，[作者公开正文](https://arxiv.org/html/2402.18134)。支持Stokes表征及假设边界，不将其去模糊实验外推成去反射保证。
5. Mingde Yao等，PolarFree: Polarization-based Reflection-Free Imaging，CVPR2025，10890–10899，[官方论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Yao_PolarFree_Polarization-based_Reflection-Free_Imaging_CVPR_2025_paper.pdf)。任务目标/数据背景；具体当前输入转换依据本项目实际代码。

上述完整新模块为本项目综合设计假设，并非某篇论文已经验证的方案，也不能仅凭模块组合确认创新性。
