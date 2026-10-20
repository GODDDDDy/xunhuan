# 当前最值得读的文献：增强、恢复与反馈（2026-10-19）

这份清单服务最新主线：偏振增强必须保留，循环/反馈优先争取，置信度可选。不是让你重新通读文件夹全部论文。先读前三篇，估计约2—3小时；熟悉PolarFree后可更快。时间是阅读安排估计。

## 1. PIDSR：先搞清楚增强器在保留什么

原文件：[P02_CVPR2025_PIDSR.pdf](C:/Users/14030/Documents/polar_reflection_literature_2026-10-17/papers/P02_CVPR2025_PIDSR.pdf)。正式题名：Complementary Polarized Image Demosaicing and Super-Resolution，CVPR2025。[原始论文](https://arxiv.org/abs/2504.07758)。

优先读§3.2—3.4、图2/4，再读§4.2表2和§4.3图6，约35—50分钟。先看图，再看公式，不必第一遍推导全部模块。

带着三个问题：为什么只把四角偏振图变清晰还不够？怎样检查AoP/DoP/Stokes信息没有被增强损坏？增强对下游任务的价值怎样验证？

迁移边界：原论文处理CPFA去马赛克及超分；我们复用其特征骨干做同分辨率任务适配，不等于完整复现PIDSR。文中去反射示例使用RSP，不能直接当成PIDSR与PolarFree组合已成功的证据。

## 2. PolarFree：读清我们正在改进的基线

原文件：[P01_CVPR2025_PolarFree.pdf](C:/Users/14030/Documents/polar_reflection_literature_2026-10-17/papers/P01_CVPR2025_PolarFree.pdf)。正式题名：PolarFree: Polarization-based Reflection-Free Imaging，CVPR2025。[原始论文](https://arxiv.org/abs/2503.18055)。

优先读§4.1—4.4、图5/6和§5.3表3，约30—45分钟。如果已经熟悉架构，集中读实际输入、训练目标和偏振消融。

带着三个问题：原RGB、偏振强度、角度/程度信息分别进入哪里？先验和恢复网络分别负责什么？论文怎样证明偏振信息有帮助？

迁移边界：论文的两阶段训练不能直接称为T1到T2的循环。G6是本项目微调所得对照，不是原论文中的新循环方法；本项目采用的损失与输入约定须看已核验执行记录。

## 3. IBCLN：看反馈怎样成为可学习的修正过程

原文件：[P08_CVPR2020_IBCLN.pdf](C:/Users/14030/Documents/polar_reflection_literature_2026-10-17/papers/P08_CVPR2020_IBCLN.pdf)。正式题名：Single Image Reflection Removal through Cascaded Refinement，CVPR2020。[原始论文](https://arxiv.org/abs/1911.06634)。

优先读§3.2—3.4、图2/3，再读§4.3表2/3和图8，约40—60分钟。

带着三个问题：为什么每轮还保留原观测？上一轮结果与隐状态分别提供什么？怎样用消融区分“多算一轮”和“真正用好反馈”？

迁移边界：不必照搬两个网络或ConvLSTM；原文有自身的透射/反射表示和线性颜色域假设，不能把我们sRGB中的I−T1直接叫真实反射层。当前38/40的独立D2是两轮反馈级联，并非严格共享权重循环。

## 有余力再读两篇

- [P12_CVPR2021_MPRNet.pdf](C:/Users/14030/Documents/polar_reflection_literature_2026-10-17/papers/P12_CVPR2021_MPRNet.pdf)：§3.1—3.3、图2—4、§4.6，约30—45分钟。看跨阶段特征融合及监督注意力；SAM不是我们H那样的质量置信度，也没有给出误差校准保证。[原始论文](https://arxiv.org/abs/2102.02808)。
- [P04_TPAMI2023_Physics_Guided_Reflection_Separation.pdf](C:/Users/14030/Documents/polar_reflection_literature_2026-10-17/papers/P04_TPAMI2023_Physics_Guided_Reflection_Separation.pdf)：§3.3、§4.1、式22—27、图1/3/5，约25—40分钟。看“先指出具体失效原因，再设计辅助修正”，而非只加一个通用置信图。其解析分离风险图依赖特定物理与成像假设，不能直接当成T1误差图。[作者稿](https://youweilyu.github.io/src/papers/2022tpami_lyu_paper.pdf)，DOI:10.1109/TPAMI.2022.3162716。

## 读完怎样用于自己的课题

每篇只记四行：它解决什么问题；输入输出和监督是什么；哪项消融支持其核心机制；迁移到我们时哪一假设不成立。先把三篇之间的关系理顺，不急着罗列模块或把每个机制都加进网络。

置信度/不确定性论文P13—P16目前选读，不是启动增强与反馈实验的前置。若后续结果指出具体需要，再围绕那个问题阅读。

章节与图表位置已结合文件夹原文和现有逐页文本核查；本页的阅读排序、时间与问题是本项目建议，不是原论文的实验结论。源资料仍在用户指定C盘目录；本页及工作副本只写F盘。
