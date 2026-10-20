# 第23—43项研究进展索引（2026-10-20）

本页用于定位资料，不替代原报告的指标口径、负面场景和限制。最新运行状态见根CURRENT_STATE.json；第43项未完成，旧阶段“下一步/正在运行”只是当时记录。

|研究段落|主要资料|阅读边界|
|---|---|---|
|23 单轮E–T协作迁移|[完整结果](../analysis/POLARAPP_SINGLE_COLLABORATION_RESULTS_ZH_20261013.md)|3000/872/188完成，但普通续训对照未完成，不能识别协作独立贡献|
|24 协作延长到6000|[完整结果](../analysis/POLARAPP_COLLABORATION_CONTINUE_RESULTS_ZH_20261014.md)|原模型继续训练，非新机制；均值小幅改善不等于跨场景稳定|
|25 冻结/部分训练|[协议与结果](../analysis/POLARAPP_FREEZE_VS_PARTIAL_RESULTS_ZH_20261014.md)|按实际来源、冻结范围与对照判断|
|质量条件及其接入|[探索总结](../analysis/CONFIDENCE_EXPLORATION_SUMMARY_ZH_20261015.md)、[H改进](../analysis/h_rank_quality_20261015/RESULTS_ZH.md)、[机制审查](../analysis/POLAR_QUALITY_MECHANISM_AUDIT_ZH_20261015.md)|质量排序、校准、候选收益及使用方式是不同问题|
|30 原乘法/学习融合/均值图|[6000/872/188结果](../analysis/c_feedback_polar_loop_20261015/RESULTS_ZH.md)|局部C未建立方向一致收益|
|31 Oracle质量与固定候选|[完整结果](../analysis/oracle_quality_diagnostics_20261016/RESULTS_ZH.md)|含GT诊断不能称部署上界或现成可学收益|
|32 训练范围×质量/反馈|[完整结果](../analysis/confidence_correction_causal_20261016/RESULTS_ZH.md)|扩大范围仍未证明稳定实用净修正|
|33 固定常数/预测/GT排序|[完整结果](../analysis/confidence_causal_completion_20261017/RESULTS_ZH.md)|GT纠正排序未挽救，不是所有辅助机制无效证明|
|34 裁剪/整图上下文|[配对诊断](../analysis/round2_context_diagnostic_20261018/RESULTS_ZH.md)|ROI结果不是全图指标|
|35 整图直接优化可达性|[技术终止](../analysis/round2_full_oracle_reachability_20261018/RESULTS_ZH.md)|0正式优化/0科学结果；资源失败不证明不存在修正|
|36 恢复器适应|[64图初筛](../analysis/round2_restorer_adaptation_20261018/RESULTS_ZH.md)|来源为旧联合模型，不是G6复现|
|37 一轮状态来源匹配|[64图初筛](../analysis/round2_state_source_matched_20261018/RESULTS_ZH.md)|只约束当前局部更新器与训练配置|
|38 改用强G6的增强/反馈|[结果与补全复核](../analysis/g6_enhancement_feedback_20261019/RESULTS_ZH.md)|普通续训共同退化，原归档收尾问题另行披露|
|39 G6 Adam状态诊断|[1000步结果](../analysis/g6_adam_state_diagnostic_20261019/RESULTS_ZH.md)|优化器是组间变量，短程不能外推长程|
|40 G6配方匹配的增强/反馈|[结果](../analysis/g6_matched_enhancement_feedback_20261019/RESULTS_ZH.md)|未复现38的大退化，但未建立模块收益|
|41 固定增强双通路|[64×4结果](../analysis/g6_enhancement_path_diagnostic_20261019/RESULTS_ZH.md)|关闭现有路径不是匹配重训消融|
|42 PolarAPP重建后增强|[6000步在线与64图](../analysis/polarapp_postprocess_value_20261019/RESULTS_ZH.md)|不用G6；尾段仍改善，不能认定充分收敛|
|43 延长并加task-only协作|[方案](../analysis/polarapp_collaboration_extension_20261020/PROTOCOL_ZH.md)、[有效预算](../analysis/polarapp_collaboration_extension_20261020/BUDGET_EXTENSION_ZH.md)、[当前交接](../analysis/polarapp_collaboration_extension_20261020/RUN_HANDOFF_ZH.md)、[故障/偏差](../analysis/polarapp_collaboration_extension_20261020/RECOVERY_REVIEW_ZH.md)|仍在训练，固定15000总终点；不得把中途探针写成最终结论|

相关解释：[G6与旧训练配方复核](../analysis/macro_recurrent_strategy_20261018/TRAINING_RECIPE_REVIEW_ZH_20261019.md)、[当前网络解释的历史版本](../analysis/CURRENT_NETWORK_EXPLAINED_ZH_20261019.md)、[训练充分性审查](../analysis/TRAINING_SUFFICIENCY_REVIEW_ZH_20261019.md)、[优先阅读清单](../analysis/literature_review_20261018/READING_GUIDE_ZH_20261019.md)。阅读时以文件日期和最新来源为准，不能将旧结构说明拼成一个同时启用所有模块的当前模型。
