# 第34项运行交接与自主后续研究入口

2026-10-18：用户已授权执行本诊断，并允许基于新证据自主设计和执行后续必要实验，直到收到新指令。每个新实验仍须在启动前向用户说明问题归属、唯一假设、对照、结果边界、预算和停止条件；不需要等再次审批。

当前项为round2_context_diagnostic_20261018；服务器根目录/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/round2_context_diagnostic_20261018。所有实验数据留服务器，本机只维护源代码及文字记录。

## 最终完成与继任入口

本项已于2026-10-18 15:10:48完整完成80行；64旧full匹配、噪声、冻结、独立CPU复核、summary/closeout成功，进程退出、锁释放。根代理独立重算配对统计，原worker归档20项hash/字节数均通过。主F验证crop净修正−.109991dB、full局部+.041457dB，不支持恢复小块推理补救；该分支结束。

**继任：analysis/round2_full_oracle_reachability_20261018（第35项），服务器同名目录。** 自动跟进后续读取35的PROTOCOL_ZH.md/RUN_HANDOFF_ZH.md：固定8未见整图×100步GT有限可达性，新4小时协议，不套34旧deadline、不自动扩Oracle。已向用户说明后按自主授权准备。完整34结果见RESULTS_ZH.md。

以下保留历史，不表示34仍在运行。

## 最新状态：首图复现拦截与同协议修复（2026-10-18 14:57）

**14:58数值确认更新：**显存已满足，恢复worker完成首图预检并提交2/80行。reports/preflight.json成功，T1哈希、T1/F的GPU指标与旧33严格一致，full/crop六次前向使用相同潜变量噪声。failures/metric_device_check.json实测：首图T1 CPU SSIM=0.9719605445861816，原GPU/旧值=0.9719589352607727；F CPU PSNR=32.278524458353075，原GPU/旧值=32.2785240311672，F SSIM也有约1.61e-6差异。原失败已由设备数值差异解释，修复未放宽1e-10容差。当前进入固定80图推理，尚未有完整科学结果或全局冻结/CPU复核。

14:54自动检查发现原worker已退出：曾于14:24显存满足后开始加载及首图完整前向，T1哈希核对通过，但首图完整图指标与旧33的严格核对失败；没有提交任何配对行。失败回执为failures/failure_1792304717.json，旧日志保留。异常状态写入还出现重复time_unix关键字，导致status停留loading，此为执行层错误，不是模型实验结论。

定位发现新程序把输出搬至CPU后计算指标，旧33原实现用GPU上的MSE/卷积；实现使用同函数但设备不同，存在数值差异的具体可能。已将指标计算恢复到原GPU设备，并增强首图差值回执及异常报告；尚待GPU数值复现，不能在核验前宣称问题已经完全解决。

修复前原worker、CPUaudit及locked文件保留在服务器failures/implementation_v1_before_metric_device_fix。原config/locked.json保持逐字节不变；config/implementation_revision_1.json登记原/修复源码SHA及原截止，只允许run_context.py/audit_context.py两项实现校验更新，所有科学来源、样本、ROI、噪声、判据和protocol_id不变。后续检查同时核验该修订清单，不重新prepare或重置注册时间。

已恢复单实例worker：PID1829090，首次start_unix仍1792304601.544374，deadline仍1792311801.544374（16:23:21）。当前status=waiting_memory，空闲6114MiB，配对行0/80。显存足够自动重跑未提交的首图。reports/preflight.json、metric_device_check.json和80行及独立核验齐全之前，不称数值预检或诊断完成。

以下14:24状态保留为运行历史。

当前状态（2026-10-18 14:24核查）：配对程序、来源和清单已锁定；后台worker真实启动，PID1806016，单实例worker.lock。GPU尚未开始计算，现有显存约6114MiB空闲，低于11000MiB门槛，worker每2秒自动检查并等待；已确认本组PID没有GPU占用，未终止其他任务。来源为第33项F6000和第32项B拟合1000终点。完整设置、主判据及停止边界见PROTOCOL_ZH.md。

锁定protocol_id=2cce603dfd35dd5e40912d2b73830a65e15002ad6e64c93242637b6aa7e53840。首次worker登记1792304601.544374（北京时间14:23:21），截止1792311801.544374（16:23:21），不重置。尚无推理或科学效果结果，不把排队称为完成预检。

后台跟进已创建并启用：automationId=automation，名称“二轮修正与置信度研究自主跟进”，每30分钟。旧33维持暂停。跟进需读取最新用户指令及本页；正常等待不反复通知，完成/失败/重要结论变化再通知。用户授权明确包括后续必要实验，不在34完成时机械停止整个研究；但新实验须有新证据、独立问题卡和有限范围。

实现复用了原加载/前向/指标，新增p_sample_loop局部噪声配对包装；原full数值由64旧T1/F匹配断言保护。两位独立审阅者审查了设计/统计及前向/噪声接口，未发现启动阻断。CPU独立审计程序已写好，尚未有80行数据供其正式运行；语法检查不等于数值核验通过。

后续检查服务器status.json、worker_registration.json、metrics/rows.json行数、reports/summary.json、reports/independent_audit.json与closeout.json；只管理本项进程并核验单实例锁。64验证须匹配旧33 T1哈希及F完整图指标。预算最多2小时，从第一次worker登记启动起计；不重置。

完成本项后，独立审阅结果，集中写RESULTS_ZH.md及根实验记录.md。新实验若有必要，建立新目录和事前问题卡，然后执行；本页记下继任目录供后续自动跟进读取。没有可行动证据时明确停止该分支，不无限派生。
