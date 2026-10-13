# 两轮残差纠错与直接特征反馈：唯一最新训练组

状态：failed

仅运行最新一组，单 seed 2026。没有匹配对照，不能将与历史结果的差异归因于新增模块。
复用已完成并核验的PIDSR纠偏预训练；其评价记录来自前一实验。PolarFree直接从官方权重开始。新增跨轮多尺度任务/偏振特征融合、可访问原始观测的纠错状态以及专用RGB残差头，T2=T1+学习到的增量。不是所有输出层共享。新预算独立于已结束的旧实验，不能将与历史结果的差异直接归因于某一接口。
AoLP 周期距离、有效区域掩码、正交关系目标残差是本课题适配，不能声称严格复现完整 PIDSR 方法。

复用的预训练已有完整 30 测试场景，选定步数 18000。
四角度平均 PSNR：退化输入 31.408527，增强输出 34.498360 dB。
SSIM：输入 0.885488，输出 0.941193。

预训练验证稳定性检查：True；实际检查步数 19000。此规则是预先规定的验证平台判据，不能证明全局收敛。

validation 全集评价未完成。

test 全集评价未完成。

实现与实验完成：False。方法有效性须结合偏振误差、T1/T2、场景分布判断，不能从训练结束直接推断。
历史测试集已有多轮探索背景；当前测试不用于选择权重、调参或启动追加实验。
完整逐图、场景指标、置信排序与区间误差见各 metrics JSON。

参考：[PIDSR CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_PIDSR_Complementary_Polarized_Image_Demosaicing_and_Super-Resolution_CVPR_2025_paper.html)、[SGSC IJCV2025](https://arxiv.org/html/2402.18134v2)、[UniRestore CVPR2025](https://arxiv.org/html/2501.13134)。这些文献不直接证明本组合的去反射收益。