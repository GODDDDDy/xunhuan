# 新偏振增强器：PIDSR 纠偏预训练与置信反馈

状态：completed

仅运行最新一组，单 seed 2026。没有匹配对照，不能将与历史结果的差异归因于新增模块。
预训练采用 PIDSR 数据、CPFA/Bayer 退化及偏振损失组，训练增强器的同分辨率恢复分支；不训练超分分支、不输入干净 RGB 引导。
AoLP 周期距离、有效区域掩码、正交关系目标残差是本课题适配，不能声称严格复现完整 PIDSR 方法。

预训练完整 30 测试场景，选定步数 18000。
四角度平均 PSNR：退化输入 31.408527，增强输出 34.498360 dB。
SSIM：输入 0.885488，输出 0.941193。

预训练验证稳定性检查：True；实际检查步数 19000。此规则是预先规定的验证平台判据，不能证明全局收敛。

PolarFree 初始化：official_polarfree，源 SHA256 a230e77fb19827a1ada41e4dc86374dee092e476915d2252841f81dd8c257ca2。

固定验证探针（原始偏振绕过增强器）步数 8000，PSNR相对官方起点变化 0.960234 dB，主网相对参数L2变化 5.326644e-03。该探针只监测性能保留，不是匹配对照，也不选择权重。

validation：872 张，步数 8000；T1/T2 PSNR 25.947380/25.947427 dB，SSIM 0.904635/0.904636。
置信—局部误差空间 Spearman 逐图均值：-0.43653638029418823。

test：188 张，步数 8000；T1/T2 PSNR 23.333284/23.333351 dB，SSIM 0.879546/0.879544。
置信—局部误差空间 Spearman 逐图均值：-0.4512481764433114。

实现与实验完成：True。方法有效性须结合偏振误差、T1/T2、场景分布判断，不能从训练结束直接推断。
历史测试集已有多轮探索背景；当前测试不用于选择权重、调参或启动追加实验。
完整逐图、场景指标、置信排序与区间误差见各 metrics JSON。

参考：[PIDSR CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_PIDSR_Complementary_Polarized_Image_Demosaicing_and_Super-Resolution_CVPR_2025_paper.html)、[SGSC IJCV2025](https://arxiv.org/html/2402.18134v2)、[UniRestore CVPR2025](https://arxiv.org/html/2501.13134)。这些文献不直接证明本组合的去反射收益。