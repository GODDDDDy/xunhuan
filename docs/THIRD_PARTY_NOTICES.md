# 第三方来源、文献及许可边界

本项目是基于已有研究的探索。公开资料保留来源，不将第三方模块重新署名为本项目发明。这里不统一授予一个覆盖所有上游的许可证；使用、再分发或商业用途须核查对应上游许可。

## 直接来源

- PolarAPP的PIDNet重建、原配TaskNet和FANet/协作设计：[作者仓库](https://github.com/roydon-luo/PolarAPP)、[原文](https://arxiv.org/html/2603.23071v2)。第42/43项后置E–T协作是本项目的任务适配设计，使用task-only外层及限定inner参数范围，不称完整复现原方法；第43项FA只在训练使用。新增源码保留上游署名和对应来源，公开代码并不意味着统一获得所有上游的再许可。

- Zhou等，PIDSR: Complementary Polarized Image Demosaicing and Super-Resolution，CVPR 2025，16081–16090。[会议原文](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhou_PIDSR_Complementary_Polarized_Image_Demosaicing_and_Super-Resolution_CVPR_2025_paper.pdf)。vendor/pidsr_official.py等是本实验归档的来源/适配快照；本项目改用同尺度恢复，不是完整SR原版。
- Yao等，PolarFree: Polarization-based Reflection-Free Imaging，CVPR 2025，10890–10899。[作者仓库](https://github.com/mdyao/PolarFree)、[作者论文](https://arxiv.org/html/2503.18055v1)。官方恢复主网、条件编码、潜变量扩散等不是本项目原创。部分源码是作者依赖或本项目适配包装。
- Restormer，CVPR 2022。[作者仓库](https://github.com/swz30/Restormer)。本地官方Transformer快照标注构建于Restormer实现，保留该关系。
- PyTorch、NumPy、OpenCV、einops、OmegaConf等是运行依赖；它们及模型预训练资源不在本仓库重新许可。

## 设计启发，不等于本项目已验证成功

- Zamir等，Multi-Stage Progressive Image Restoration，CVPR 2021。[原文](https://arxiv.org/html/2102.02808v1)：跨阶段特征和监督；具体R2P状态/输出头为本项目迁移设计。
- Kendall、Gal，What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?，NeurIPS 2017。[会议记录](https://papers.nips.cc/paper_files/paper/2017/hash/2650d6089a6d640c5e85b2b88265dc2b-Abstract.html)：不确定性/误差尺度；不能为本项目C提供概率校准保证。
- Purohit等，Spatially-Adaptive Image Restoration using Distortion-Guided Networks，ICCV 2021。[原文](https://arxiv.org/html/2108.08617v1)：定位与特征调控；不直接验证本项目Q标签。
- Dong等，Location-aware Single Image Reflection Removal，ICCV 2021。[原文](https://arxiv.org/html/2012.07131v2)：反射位置与迭代，位置不是动作收益。
- Geifman、El-Yaniv，SelectiveNet，ICML 2019。[原文](https://proceedings.mlr.press/v97/geifman19a.html)：风险/覆盖；本项目的20%安全条件不是该论文的理论保证。

更多文献及迁移限制在analysis中的逐篇文档。浮点灰度桥、具体Q代理、四动作成本头和组合协议是本项目自行设计/改造，是否有效由本项目实际对照决定。

## 图像与文档

最新配图包含数据集固定验证样本的科研说明图。底层数据权利归原提供方；本仓库没有上传完整数据集。图中的GT、事后热图和理想动作只用于监督/诊断说明，不是部署输入。

未上传第三方论文PDF。上传的PPT、结构SVG、研究记录和指标是本项目整理产物，其中引用内容仍归相应作者。原署名、注释与来源引用保持。
