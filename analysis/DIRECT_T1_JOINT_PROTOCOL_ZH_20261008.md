# 当前执行：带偏振增强的K1起步，直接T1反馈两轮联合训练

日期：2026-10-08。用户明确授权先执行直接T1图像反馈，不加入置信度；随后澄清要求保留带偏振增强模块的模型，并明确每次实验必须完成测试集评价。

## 初始化与结构

本次完整继承现有K1 aligned seed2026 step3000检查点：增强模块R2P、三个对齐投影头和PolarFree的net_g均加载该检查点；不重置增强模块输出头，不用G6替换它的去反射权重。扩散先验继续使用已有官方权重并保持冻结。

源文件：`/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/normalized_threeway_20261002/runs/K1/seed_2026/checkpoints/step_3000.pt`。

SHA256：`f87e8b75d1335c7ae979ee563cac14270258c130ac2953a5903b7eb7de647b64`。

```text
P0 + 原混合RGB → 已训练R2P → P1
P1 + 原混合RGB → PolarFree → T1
P1 + clamp(T1,0,1) → 同一个R2P → P2
P2 + 原混合RGB → 同一个PolarFree → T2
```

T1条件不detach，第二轮恢复损失沿图像反馈路径传到第一轮。两轮R2P和任务网络共享参数。沿用配对相同CPU/全部CUDA随机流，控制两轮扩散采样差异。没有新增置信度分支，也没有启用旧内部特征反馈adapter。

对齐保留现有第一轮多层投影＋余弦损失，目标任务特征detach。复用原系数0.030651881051471687；它是明确的历史复用起点，不称已经针对本轮收敛训练重新校准。任务损失仍为L1+(1−SSIM)+0.1 Fourier phase+0.1 VGG；两轮任务权重[1/3,2/3]。

## 训练与评价

- 追加10000步，batch2、crop128、workers2，20000次裁剪输入；训练池5440张均覆盖，等效约3.68遍。本轮没有声称遍历每张图的全部空间区域。
- 学习率R2P/投影1e-4，net_g 1e-5；沿用Adam和分别norm clip1。
- 加载模型权重后新建Adam。来源K1已有3000步，加上此次10000步，总计13000步；本次step编号从1到10000。200→10000阶段继承本次optimizer状态，是连续训练，不回滚或重置。没有声称继承旧K1的Adam状态。
- 预检只核查源权重逐元素一致、两轮实际调用、第二轮条件等于T1、L2到T1图像的非零梯度、增强/投影/net_g梯度、冻结范围和确定性。200步检查由执行器自动核查并继续，不以小预算性能作为结构有效性门槛。
- 代理验证检查点0/200/1000/3000/6000/10000，最终使用固定追加step10000，完整评价872张validation后，必须完成全部188张test。两个划分分别报告同模型T1/T2的PSNR、SSIM图像均值、场景均值和T2−T1，保存逐图指标。test不用于本轮训练、调参或检查点选择。

本轮比较新模型与源K1同时包含追加训练与结构改变，不能把全部变化归因于T1反馈。尚未运行同额外10000步的独立K1对照；如需判断反馈独立贡献，后续必须匹配这一预算。G6仅可作已有结果参照，本轮不声称与它等总训练预算。

## 现有能力复用与新增代码范围

复制并复用已有matched/phase训练器及其原始快照，保留数据加载、计划生成、损失、裁剪、优化器、检查点和评估。必要的新增适配包括：完整K1权重初始化、第二轮RGB条件改为T1、相关梯度审计、连续执行及状态/报告输出。旧训练器仅支持固定原RGB和内部特征反馈，故需要这一最小前向扩展。

为适应共享3080 Ti显存，验证前在已完成optimizer.step后释放梯度、临时将Adam状态移至CPU，验证后按原dtype/device恢复；断点续训与最终评价先在CPU读取检查点，避免将不需要的检查点副本保留在显存。未改变输入大小、精度、权重、损失或数据计划。复制的原快照另存为`run_experiment_snapshot_original.py`，修改后的执行快照与扩展均记录哈希。

本地代码：`F:/WORK/projects/pidsr_recurrent_joint_k2/server_scripts/direct_t1_joint_20261008/`。其中本地默认`run_study.py`已更新为验证后直接调用测试；当前已在运行的服务器训练代码保持启动时版本，新增独立测试接续程序，避免重启训练。

当前服务器输出：`/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/direct_t1_joint_20261008_memorysafe_v2/`，训练与评估数据位于`run/`。训练/验证状态为`status.json`与`evaluation_progress.json`；测试接续状态为`test_status.json`、`test_progress.json`。验证程序退出后，接续程序更新主状态为`full_test`，最终仅在188张测试成功后写入含`test_completed=true`的`completed`状态。

`metrics/results.json`保留验证结果；`metrics/test_results.json`保存测试原值和汇总；`metrics/experiment_results.json`保存验证与测试的完整合并结果。最终报告为`analysis/DIRECT_T1_JOINT_RESULTS_ZH_20261008.md`，验证阶段旧报告另存为`DIRECT_T1_JOINT_VALIDATION_ONLY_ZH_20261008.md`。

测试要求在训练过程中补充。启动时`protocol_lock.json`中原“只评价validation”的字段保留作历史记录，不据此省略测试；新增评价范围、固定step10000、测试程序及训练锁哈希记录于`evaluation_extension/evaluation_addendum.json`。训练配置、数据计划、训练代码和参数不变；复用同一锁定前向及评价器，确认188个测试sample_id完整、无重复，两轮实际R2P/PolarFree各调用376次，验证与测试权重哈希相同。

当前本地取回目录：`F:/WORK/projects/pidsr_recurrent_joint_k2/_dl/direct_t1_joint_20261008_memorysafe_v2/artifacts/`。代码、配置、清单、计划和源检查点哈希由`protocol_lock.json`记录。

初次尝试`direct_t1_joint_20261008/`已正常更新200步，但在随后全分辨率代理验证时显存不足，未成功保存200步检查点；日志保留。第一次显存修订`direct_t1_joint_20261008_memorysafe/`在审查中发现后续CPU读取检查点仍需补齐，主动中止并保留记录。当前v2从同一完整K1源权重重新起步，不将这些未完成尝试计入当前模型的训练步数，不把启动检查通过当作性能结果。

本次服务器实际读到GPU为RTX3080 Ti，12GB；存在另一独立Python作业占用约1.57GB，未停止或修改它。具体耗时与资源情况以本次运行记录为准。

## 已核查的执行状态与结果

当前v2的真实CUDA预检通过：R2P与PolarFree各实际调用两次，第二轮条件逐元素等于clamp(T1)，第二轮损失对T1图像的梯度范数为0.0004018320760；R2P、三层投影与net_g均有非零梯度，扩散先验冻结，重复前向一致。源K1权重加载逐元素相等。

本轮于2026-10-08北京时间20:54完整完成：训练日志连续1–10000步，完整872张validation及188张test均已评价，测试两轮R2P/PolarFree各实际调用376次；主状态和测试状态均为completed，进程均已退出。第200步检查点的457项Adam参数状态均为step200，随后连续续训通过核查。

固定最终权重SHA256为`eb17a5537bcd26583b73670a34a258f948056935f851b42c7a44eb62e1dd15ea`；validation/test使用同一权重，本地取回的检查点哈希一致。完整结果和原值链接见[结果报告](DIRECT_T1_JOINT_RESULTS_ZH_20261008.md)。

测试图像平均PSNR：T1为23.37858336 dB，T2为23.36154630 dB，T2−T1为−0.01703706 dB；SSIM差为−0.00016854。验证图像平均PSNR的T2−T1也为负（−0.01118809 dB）。相对源K1，测试均值略升，而验证均值下降；这是一组单seed混合结果，尚未显示稳定迭代收益，不能把测试提升单独归因于反馈结构。

执行器继承的`RUN_PLAN.md`默认模板含旧初始化/反馈描述；当前运行在200步接续后已用本实际协议同步该文件，旧模板原文保留于`run/diagnostics/legacy_generated_run_plan.txt`。科研复现以本协议、实际配置、清单哈希、检查点和日志为准。
