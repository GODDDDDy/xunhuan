# K2训练目标权重对照：执行协议

用户于2026-10-02授权亲自执行下一步。先完成修正后的梯度诊断，再开展本协议两组训练及验证。实验属于单seed探索，不是独立确认。

## 问题与单变量

固定原no-feedback、共享权重K2结构，比较每轮任务损失中傅里叶相位项系数1与0.1。这里phase指灰度图像FFT相位损失，不是偏振角监督。系数0.1预先选择，未通过这次验证集搜索确定。

每轮任务损失：L1 + (1−SSIM) + c×phase + 0.1×VGG感知L1。

训练总目标：0.5×任务损失(T1) + 任务损失(T2) + 0.030651881051471687×第一轮特征对齐。改变phase系数时不重新标定alignment系数；相对目标权重随之改变是该干预的结果。

两组均从原PIDSR e42 R2P和官方net_g初始化，使用全新Adam；初始模型状态SHA256一致，28张step0代理验证逐图PSNR/SSIM差为0。seed2026，固定相同数据、裁剪与随机schedule；batch2、crop128；R2P学习率1e-4，net_g学习率1e-5；两个参数组分别裁剪至范数1。原精度设置不变。

每组先200步检查，再携带完整优化器状态恢复至3000步。R2P、alignment投影与net_g参与训练；扩散先验及未使用feedback分支冻结。检查点保存时逐张量核查冻结状态。

## 评估与判读

固定step3000；对两组各评估完整872张validation。复用原正式评估函数和循环前向，在相同输出上附加损失分项计算，避免新增TaskNet前向。损失算术前后核查CPU和全部CUDA随机流未变。跨组同时报告原始phase、两种共同系数下的任务损失，不能直接拿不同权重的训练总损失比较。

主要比较新phase0.1与新phase1的T1/T2绝对PSNR、SSIM、两种聚合、T2−T1及逐场景变化。若第二轮增量改善而最终T2下降，不能称总体改善。历史no-feedback用于复现参照，历史G6用于整体方案参照；G6未以新phase系数重训，不能将对G6的差值归因于循环或R2P本身。

两组都更新net_g，因此该干预的效果也不能只归因于偏振表征模块。正面结果不能证明feedback有效，本实验关闭feedback；负面结果也不能否定所有循环或联合优化设计。validation已参与探索；旧test已反复查看，不作为新的独立确认。

## 异常与证据保存

首次训练前全分辨率验证显存不足，尚无优化器更新。保留失败日志后，两组统一使用`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`重启。此设置及梯度重组容差修订单列runtime_environment.json，不隐去失败尝试。

服务器独立输出根：`/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/phase_objective_20261002`。本地脚本：`F:/WORK/projects/pidsr_recurrent_joint_k2/server_scripts/phase_objective_20261002`。本地结果副本：`F:/WORK/projects/pidsr_recurrent_joint_k2/_dl/phase_objective_20261002`。

原trainer逐字节归档；最小扩展仅添加phase系数、固定lambda和冻结审计。protocol_lock.json记录输入/代码SHA256；最终核查连续3000条训练日志、优化器step200/3000、冻结张量及评估检查点hash。原历史产物保留。

原trainer自动生成RUN_PLAN中的“仅feedback不同”来自历史模板，不适用于本实验；本协议及protocol_lock.json定义实际比较变量。收尾时修正本次新运行的该文档，保留原trainer快照不变。
