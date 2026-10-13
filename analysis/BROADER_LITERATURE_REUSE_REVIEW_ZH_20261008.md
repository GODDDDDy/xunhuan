# 偏振辅助去反射：扩大文献检索与机制复用评估

检索与核查日期：2026-10-08。本文是研究设计评估，不是新实验结果，也不构成任何候选模块在本项目有效的证明。

## 1. 本次检索改变了什么

研究依据扩展到反射分层、位置感知、跨轮状态、分阶段恢复、任务特征适配和偏振结构建模。PIDSR、PolarAPP继续作为已有参考，但不再限定方案来源。

当前优先级判断：先探索“多尺度、依据任务状态控制更新、保留跨轮记忆”的偏振适配器，并允许它与现有去反射网络共同适配。RAR的评估与停止思想、Stokes结构建模是进一步候选。优先级来自本项目的观测和迁移条件，是本次研究判断，不是文献给出的排名。

本次筛选是有目标的初步评估，不是系统综述或所有论文的完整复现。对于重点候选核查了作者正文或实际代码；其他候选按公开方法说明和仓库状态评估。部分CVF PDF直连返回403，采用官方收录页、作者arXiv正文或作者公开稿补查。未下载模型，未运行第三方网络。

## 2. 迁移判断依据：我们实际已经知道什么

以下属于项目证据，而非外部文献结论：

- G6是只微调PolarFree的net_g、不接R2P的强基线。现有联合优化和循环方案尚未形成对G6方向一致的额外收益。见[测试补全报告](TEST_COMPLETION_RESULTS_ZH_20261007.md)。
- 最新固定28图检查中，偏振修改进入灰度、AoLP/DoLP和扩散prior，不能用“完全没有传进去”解释该子集结果。不能由此认定STE梯度可靠或偏振修改真实。
- 最新同seed、同额外1000步试验，任务预测条件相对原RGB条件的四项聚合均正，但相对冻结G6四项均负。这是条件表示的正向线索，尚未实现额外增益；也不是收敛证明。见[2026-10-07报告](STATE_CONDITION_PROBE_RESULTS_ZH_20261007.md)。
- 留存R2P由两个两层3×3分支、1×1融合/FiLM、三个3×3残差卷积组成，理论空间感受野约11×11。**已有空间融合gate**，所以“再加一个gate/attention”本身不构成实质改进；值得改变的是门控依据、作用对象、尺度与跨轮状态。源代码见[R2P（本地资料）](../docs/LOCAL_ARTIFACTS.md#artifact-0008)。该模块是自建适配器，不等于PIDSRNet。
- 当前数据读取流程使用四角度偏振输入和RGB透射真值，未读取偏振透射/反射真值。这里不声称整个服务器不存在这类数据。见[当前留存数据读取代码](../_dl/state_condition_probe_20261007/artifacts/analysis/reused_source/pidsr_k2_929/deps/rgb_final_confirmation/src/rgb_final_confirmation/data.py)。
- 项目已核验RGB来自同一偏振观测。RGB、Stokes或任务预测可以提供不同表示与学习到的结构线索，不能被描述为额外独立传感信息。

这些事实支持扩大结构探索，但不足以把失败唯一归因于小感受野、共享权重、冻结TaskNet、STE或训练预算。

## 3. 文献与可复用部件

表中的“优先”是迁移评估。公开代码仅表示查到真实源码或入口，不表示已在本项目环境验证。预训练模型链接也未实际下载核验。

|工作与正式出处|原文直接支持的机制|可迁移到我们哪里的判断|代码与迁移限制|优先级|
|---|---|---|---|---|
|[IBCLN：Single Image Reflection Removal Through Cascaded Refinement，CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Single_Image_Reflection_Removal_Through_Cascaded_Refinement_CVPR_2020_paper.html)|透射/反射估计互相帮助；ConvLSTM传递跨轮状态；残差重建监督。|借用低分辨率状态单元，让下一轮看到原始观测、已有任务估计及之前的更新状态。|[官方训练/推理代码](https://github.com/JHL-HUST/IBCLN)公开。原分层和合成重建损失有自己的数据与成像假设，不能照搬成PolaRGB真值关系。|高|
|[Location-Aware Single Image Reflection Removal，ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Dong_Location-Aware_Single_Image_Reflection_Removal_ICCV_2021_paper.html)|多尺度Laplacian特征预测反射置信图，控制两层恢复中的特征流；循环细化。|借用多尺度区域判断，让输出更新幅度依据原始输入与当前透射估计变化。与现有分支融合gate作用不同。|[官方推理与模型代码](https://github.com/zdlarr/Location-aware-SIRR)公开；本次根目录未见完整训练入口。原方法虽不需要反射区域mask真值，其合成监督仍使用T/R及混合关系。|高|
|[MPRNet：Multi-Stage Progressive Image Restoration，CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Zamir_Multi-Stage_Progressive_Image_Restoration_CVPR_2021_paper.html)|阶段间SAM监督注意力、跨阶段特征融合CSFF，以及上下文/高分辨率恢复。|借用多尺度主干和跨阶段特征传递，把任务输出或特征引入下一轮。可用RGB任务真值监督阶段输出。|[官方代码](https://github.com/swz30/MPRNet)，[实际SAM实现](https://github.com/swz30/MPRNet/blob/main/Deblurring/MPRNet.py)已核查。SAM前向输入不含GT；原实现是3通道，偏振适配需改接口，不能假定有12通道干净偏振标签。|高|
|[Trash or Treasure? An Interactive Dual-Stream Strategy for Single Image Reflection Separation，NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/cf1f78fe923afe05f7597da2be7a3da8-Abstract.html)|将一条分支被ReLU抑制的信息传递给另一条分支，逐块交互。|如果转向透射/反射双状态，可借用轻量交互块，充分使用被当前透射分支抑制的特征。|[官方训练/推理代码](https://github.com/mingcv/YTMT-Strategy)公开。原加性分层解释有条件；它不是“RGB分支和polar分支随便互换”的依据。|中高，依赖双状态设计|
|[DSRNet：Single Image Reflection Separation via Component Synergy，ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Hu_Single_Image_Reflection_Separation_via_Component_Synergy_ICCV_2023_paper.html)|学习混合模型中的残留项；MuGI互相门控；语义金字塔双流交互。|借用MuGI和多尺度交互；学习残留项提醒我们不要把输入减透射预测直接当成真实反射。|[官方训练/推理代码](https://github.com/mingcv/DSRNet)公开。完整方法依赖分层/合成监督，需要确认我们如何获得R标签或只借用特征模块。|中高|
|[RDNet：Reversible Decoupling Network for Single Image Reflection Removal，CVPR 2025](https://arxiv.org/html/2410.08063)|可逆编码、层间自适应解耦、透射率感知prompt校准特征。|TAPG提供“按图像混合条件调节特征”的参考，比给所有样本固定修改幅度更有针对性。|[官方仓库](https://github.com/lime-j/RDNet)已公开，但作者标记弃用并建议[XReflection](https://github.com/hainuo-wang/XReflection)。TAPG不是局部反射真值；整个可逆网络移植会明显改变主干。|中|
|[RAR：Restore, Assess, Repeat: A Unified Framework for Iterative Image Restoration，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Restore_Assess_Repeat_A_Unified_Framework_for_Iterative_Image_Restoration_CVPR_2026_paper.html)|潜空间中联合评估与恢复；评估反馈指导下一轮，并比较前后结果决定继续与否。|借用“状态评估→更新控制/停止”思想，给当前迭代一个明确的继续条件。|[官方代码](https://github.com/saic-fi/RAR)公开。原模型依赖VLM/扩散组件，轻量任务误差头只是我们的改造假设；论文保真度与感知指标也有权衡。|思想高，整套移植低|
|[UniRestore：Unified Perceptual and Task-Oriented Image Restoration Model Using Diffusion Prior，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_UniRestore_Unified_Perceptual_and_Task-Oriented_Image_Restoration_Model_Using_Diffusion_CVPR_2025_paper.html)|CFRM恢复编码特征；TFA根据任务prompt在多尺度解码中适配特征。|参考任务特征适配接口，可探索让偏振线索在net_g特征层被使用。|[官方代码](https://github.com/unirestore/UniRestore)公开。任务prompt不是上一轮图像预测；原文下游示例是分类/分割，不能直接当作偏振去反射证据。|中|
|[EDTR：Exploiting Diffusion Prior for Task-driven Image Restoration，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Kim_Exploiting_Diffusion_Prior_for_Task-driven_Image_Restoration_ICCV_2025_paper.pdf)|从保留观测线索的预恢复图像出发，使用轻度加噪和少量去噪步，恢复任务相关细节。|为“先验要服务具体任务，且应保留观测锚点”提供设计依据。|[官方代码](https://github.com/JaehaKim97/EDTR)公开；作者推荐至少4×40GB或2×80GB训练配置。原Stable Diffusion系统不适合作为当前低成本首个移植目标。|思想中，整套低|
|[Learning to Deblur Polarized Images，IJCV 2025](https://link.springer.com/article/10.1007/s11263-025-02459-7)|利用Stokes结构信息的SGSC多尺度跳接；先恢复非偏振图，再重建偏振图。|借用Stokes特征融合/跳接组织输入，探索更有结构的偏振更新表示。|[官方训练/推理代码](https://github.com/fourson/Learning-to-Deblur-Polarized-Images)公开。[作者正文](https://arxiv.org/html/2402.18134)中的Stokes损失使用对应真值；当前流程不能直接获得这种监督。去模糊现象不能直接推广到反射混合。|中高，物理分支候选|
|[Quality-Improved and Property-Preserved Polarimetric Imaging via Complementarily Fusing，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f543648b4332a4d8b9e1b72c6b4e2e26-Abstract-Conference.html)，及[PolFusion+，TPAMI 2026](https://doi.org/10.1109/TPAMI.2026.3665927)|强度恢复与偏振重建分阶段处理；期刊版进一步处理饱和与DoP/AoP的相互关系。|借用偏振表示与分阶段特征融合，避免把四角度当成毫无关系的普通通道。|[会议代码](https://github.com/fourson/Quality-Improved-and-Property-Preserved-Polarimetric-Imaging-via-Complementarily-Fusing)、[期刊代码](https://github.com/fourson/PolFusion_plus)均公开。完整方法需要短曝光噪声图和长曝光模糊图，当前数据输入不匹配。|表示中高，完整系统低|
|[Physics-Guided Reflection Separation From a Pair of Unpolarized and Polarized Images，TPAMI 2023](https://doi.org/10.1109/TPAMI.2022.3162716)|估计半反射面方向，用偏振成像物理约束解耦两层。|提供物理可行性和反射/折射建模参考，帮助判断哪些约束能用于本数据。|[作者公开稿](https://www.jdl.link/doc/2011/20221224_Lyu_TPAMI22.pdf)可核查。未确认期刊版完整官方训练代码；不能把同团队NeurIPS 2019代码当成该版本。成像、曝光与几何条件需另核。|中低，物理依据|
|[Reflection Separation from a Single Image via Joint Latent Diffusion，CVPR 2026](https://brian90709.github.io/diff-reflection-separation/)|共同生成透射/反射潜变量，配合交互、disjoint sampling与潜变量优化。|作为更大幅度转向显式双层恢复的候选，可研究其交互与混合约束。|[官方推理源码和模型链接](https://github.com/Brian90709/diff-reflection-separation-code)公开；本次未确认完整训练发布。需要SD2及新增适配；生成合理细节不保证提高本项目PSNR。|低成本阶段低|

### 必须单独记录的直接先例：RP2PN

[Reflection Removal Using Recurrent Polarization-to-Polarization Network](https://arxiv.org/html/2402.18178)，Bian、Monno、Okutomi，**ICASSP 2024**，DOI [10.1109/ICASSP48485.2024.10446021](https://doi.org/10.1109/ICASSP48485.2024.10446021)。它不是CVPR/ECCV论文，但和本课题非常直接相关，不能因会议类别而漏掉。

其偏振反射/透射分支通过循环反馈协调恢复，反射支路用ConvLSTM；输入包含角度之间的差分线索。值得借用角度差分特征及状态传递。完整训练使用偏振I/R/T标签，不能直接等同于当前RGB透射监督。此次查看[作者仓库](https://github.com/wjbian/RP2PN)仅有README和supp.pdf，未发现训练/推理源码。

该先例说明“偏振+循环去反射”已有明确研究。若新原型有效，需要说明它如何改善任务状态与偏振更新的协调，不能把加入循环本身当成充分创新。

## 4. 综合判断：最值得复用的五类东西

以下是文献和本项目证据的综合解释，尚待实验验证。

### A. 多尺度上下文与跨阶段特征：首个原型的基础

依据MPRNet的上下文恢复和CSFF、Location-Aware的多尺度区域线索，把目前浅层局部R2P改为小型编码器—解码器。保留高分辨率跳接，在低分辨率提取更大范围的混合结构。

推理依据：当前适配器局部性明确，而分层判断可能需要更大范围的结构。限制：我们尚未实验证明11×11是主要瓶颈；扩大主干也可能只是增加容量或普通微调收益。

### B. 根据任务状态控制实际更新：首个原型的核心

原始偏振、当前透射预测及投影后的任务特征共同参与更新计算；门控直接控制偏振残差的空间幅度。借鉴Location-Aware的区域控制和MPRNet的阶段反馈。

必须区别现有模块：原gate混合RGB/polar特征，新门控目标是控制“当前这一轮在哪里修改、修改多少”。通过RGB任务损失联合学习时，应称其为更新门控；没有区域真值或专门核验时，不能称为真实反射置信图。

原论文无mask标签不意味着本项目可原样复制其损失：它仍利用合成T/R组成监督。可只借用区域控制机制，另用我们确实具有的任务监督。

### C. 跨轮隐状态：使循环具有可累积的信息

参考IBCLN、RP2PN，在低分辨率瓶颈放置ConvLSTM，把上一轮处理状态传下去；每轮仍能看到原始观测。跨轮隐藏特征与跨尺度融合是不同机制。

推理依据：我们已有第二轮较第一轮变差的结果，需要让第二轮具备针对当前状态的处理能力。限制：记忆不是改善保证，ConvLSTM也不能自动保证收敛或逐图单调提升。仅将新旧预测相减作为辅助特征，不应把该差值当真实反射层标签。

### D. 任务适配与联合训练：给新结构共同调整的机会

结合UniRestore的任务特征适配思路，从已有G6初始化net_g，新适配器以零输出头起步，逐步允许net_g与适配器共同更新。若选择修改特征接口，则应直接监督最终RGB透射恢复，不能继续只凭偏振输出变化声称有效。

理由是最新试验只冻结整个G6，不能回答新结构与任务网络共同适配后的表现。旧G4/K2已经试过联合更新且收益有限，因此“解冻就能解决”没有依据。联合训练是候选配置，不是已定位的病因。

### E. 结构化偏振表示与可学习停止：后续候选

IJCV去模糊工作、PolFusion系列支持将强度与偏振结构分开建模，RP2PN提供角度差分输入的先例。可探索Stokes/角度差分特征；输出测量一致性约束必须符合实际图像线性化、标定和归一化条件。

需要避免两个混淆：保持混合输入的DoLP/AoLP不变可能同时保留反射；数值上满足偏振关系也不等于去反射更好。任务适配器产生的偏振输出还不一定能解释为干净透射偏振真值。

RAR支持依据当前状态评估是否继续。我们可以后续训练轻量任务相关评估头，但这不等于复现RAR的VLM潜空间模块。若用训练样本GT误差作评估头标签，GT只能出现在训练标签生成中，推理时只能使用输入/预测/特征。该停止机制只能避免部分坏更新，不能替代一个本身有效的更新模块。

## 5. 建议进入的下一条原型路线

建议先做一个逻辑完整的原型：

1. 以已有G6为去反射网络初始化，以原始四角度偏振为固定观测锚点。
2. 用小型多尺度适配器接收当前偏振、当前任务预测/特征及跨轮隐状态。
3. 在低分辨率瓶颈保留ConvLSTM状态，产生空间门控和偏振残差，保留实际clamp处理。
4. 从两次适配更新开始，任务网络在更新间重新生成任务状态；新模块与net_g允许共同适配。
5. 对各阶段RGB输出使用已有真实透射监督。初版不引入未经数据支持的偏振T/R真值损失，也不一次塞入全部候选组件。

这是一条综合设计假设，不是某篇论文原方案。首轮可以一起改变多尺度结构、更新门控和跨轮状态，以探索性能；无需马上穷举每个开关。保留初始复现和同预算继续微调G6的对照，防止把更多训练预算当成新模块贡献。有效原型出现后再逐项消融、补种子和做独立确认。

评价不能只看两轮模型高于K1或论文原始权重：首先看是否超过同条件G6，并并列观察PSNR、SSIM、图像均值、场景宏均和难度分组。已有validation和test已经多次查看，后续最终确认需另安排独立评估；不为每个小调整重复查看test。

该原型比当前方案更有文献依据，但其性能、预算和最佳训练安排尚未实测。本轮完成的是检索与路线评估，没有启动训练。

## 6. 文献定位信息

以下给出作者、正式出处和稳定入口。代码来源与迁移条件见上表。

1. Chao Li, Yixiao Yang, Kun He, Stephen Lin, John E. Hopcroft. *Single Image Reflection Removal Through Cascaded Refinement*. CVPR 2020. [CVF正式入口](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Single_Image_Reflection_Removal_Through_Cascaded_Refinement_CVPR_2020_paper.html)；[作者稿](https://arxiv.org/abs/1911.06634)。
2. Zheng Dong, Ke Xu, Yin Yang, Hujun Bao, Weiwei Xu, Rynson W.H. Lau. *Location-Aware Single Image Reflection Removal*. ICCV 2021. [CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Dong_Location-Aware_Single_Image_Reflection_Removal_ICCV_2021_paper.html)；[正文](https://arxiv.org/html/2012.07131)。
3. Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, Ling Shao. *Multi-Stage Progressive Image Restoration*. CVPR 2021. [CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Zamir_Multi-Stage_Progressive_Image_Restoration_CVPR_2021_paper.html)。
4. Qiming Hu, Xiaojie Guo. *Trash or Treasure? An Interactive Dual-Stream Strategy for Single Image Reflection Separation*. NeurIPS 2021. [正式收录](https://proceedings.neurips.cc/paper/2021/hash/cf1f78fe923afe05f7597da2be7a3da8-Abstract.html)。
5. Qiming Hu, Xiaojie Guo. *Single Image Reflection Separation via Component Synergy*. ICCV 2023. [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Hu_Single_Image_Reflection_Separation_via_Component_Synergy_ICCV_2023_paper.html)。
6. Hao Zhao, Mingjia Li, Qiming Hu, Xiaojie Guo. *Reversible Decoupling Network for Single Image Reflection Removal*. CVPR 2025. [正式论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_Reversible_Decoupling_Network_for_Single_Image_Reflection_Removal_CVPR_2025_paper.pdf)；[作者正文](https://arxiv.org/html/2410.08063)。
7. I-Hsiang Chen, Isma Hadji, Enrique Sanchez, Adrian Bulat, Sy-Yen Kuo, Radu Timofte, Georgios Tzimiropoulos, Brais Martinez. *Restore, Assess, Repeat: A Unified Framework for Iterative Image Restoration*. CVPR 2026. [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Restore_Assess_Repeat_A_Unified_Framework_for_Iterative_Image_Restoration_CVPR_2026_paper.html)；[作者正文及补充](https://arxiv.org/html/2603.26385)。
8. I-Hsiang Chen, Wei-Ting Chen, Yu-Wei Liu, Yuan-Chun Chiang, Sy-Yen Kuo, Ming-Hsuan Yang. *UniRestore: Unified Perceptual and Task-Oriented Image Restoration Model Using Diffusion Prior*. CVPR 2025. [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_UniRestore_Unified_Perceptual_and_Task-Oriented_Image_Restoration_Model_Using_Diffusion_CVPR_2025_paper.html)。
9. Jaeha Kim, Junghun Oh, Kyoung Mu Lee. *Exploiting Diffusion Prior for Task-driven Image Restoration*. ICCV 2025. [正式论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Kim_Exploiting_Diffusion_Prior_for_Task-driven_Image_Restoration_ICCV_2025_paper.pdf)；[作者稿](https://arxiv.org/abs/2507.22459)。
10. Chu Zhou, Minggui Teng, Xinyu Zhou, Chao Xu, Imari Sato, Boxin Shi. *Learning to Deblur Polarized Images*. IJCV 133, 5976–5991, 2025. DOI：[10.1007/s11263-025-02459-7](https://doi.org/10.1007/s11263-025-02459-7)。
11. Chu Zhou, Yixing Liu, Chao Xu, Boxin Shi. *Quality-Improved and Property-Preserved Polarimetric Imaging via Complementarily Fusing*. NeurIPS 2024. [正式收录](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f543648b4332a4d8b9e1b72c6b4e2e26-Abstract-Conference.html)。
12. Chu Zhou, Yixing Liu, Minggui Teng, Chao Xu, Boxin Shi, Imari Sato. *Toward a Unified Complementary Fusion Framework for Robust Polarimetric Imaging*. TPAMI 48(7), 7720–7734, 2026. DOI：[10.1109/TPAMI.2026.3665927](https://doi.org/10.1109/TPAMI.2026.3665927)；[作者公开稿](https://pub-e7bc7e5ed67b401a8e9e587c3dee22b6.r2.dev/publications/2026/TPAMI26_PolFusion.pdf)。作者稿/仓库题名使用“Towards”。
13. Youwei Lyu, Zhaopeng Cui, Si Li, Marc Pollefeys, Boxin Shi. *Physics-Guided Reflection Separation From a Pair of Unpolarized and Polarized Images*. TPAMI 45(2), 2151–2165, 2023（在线发表2022）. DOI：[10.1109/TPAMI.2022.3162716](https://doi.org/10.1109/TPAMI.2022.3162716)。
14. Zheng-Hui Huang, Zhixiang Wang, Yu-Lun Liu, Yung-Yu Chuang. *Reflection Separation from a Single Image via Joint Latent Diffusion*. CVPR 2026. [作者项目与论文](https://brian90709.github.io/diff-reflection-separation/)；[CVF正式论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_Reflection_Separation_from_a_Single_Image_via_Joint_Latent_Diffusion_CVPR_2026_paper.pdf)。
15. Wenjiao Bian, Yusuke Monno, Masatoshi Okutomi. *Reflection Removal Using Recurrent Polarization-to-Polarization Network*. ICASSP 2024. DOI：[10.1109/ICASSP48485.2024.10446021](https://doi.org/10.1109/ICASSP48485.2024.10446021)；[作者正文](https://arxiv.org/html/2402.18178)。

实验选择与独立评价的约束依据：Gavin C. Cawley, Nicola L. C. Talbot. *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*. JMLR 11, 2079–2107, 2010. [原文](https://jmlr.org/beta/papers/v11/cawley10a.html)。
