# 第30项运行交接

**最终状态｜2026-10-16：已完成并结束，不再等待显存，不启动新任务。** A/B/D各6000步；256验证按原规则补齐为872，188测试完整。北京时间03:29评价完成、04:10服务器归档结束；控制进程4126455已退出。CPU独立复核训练日程、第一轮输出、终点权重、逐图汇总和清单通过。主要比较与C局部比较均未达到预设投入门槛，按约定停止本实例。

结果和核验范围统一见[最终结果报告](RESULTS_ZH.md)。服务器`status.json`为`completed`，`closeout.json`为成功结束，本次核验保存在`reports/checked_results_20261016.json`。没有新训练排队；不要将下方历史PID、OOM或等待回执理解成仍需恢复。

## 以下为执行历史，所有旧等待状态均已结束

**最新指令覆盖下方全部旧时限：用户已明确取消时间预算，只要显存够就运行。**当前监控PID为4126455，每2秒检查，空闲≥11500MiB自动加载同一6000步终点继续评价，等待不设到期时间。旧23:51最晚启动和次日06:49终止均已取消；不要恢复旧限时看门进程或按旧预算杀掉本组任务。

当前入口仍为`scripts/c_loop_eval_recovery.py`，单实例锁保持生效。运行授权为`config/evaluation_runtime_policy.json`；`budget.json`的`time_budget_enforced=false`、`deadline_unix=null`，`watchdog_registration.json`标记旧时限终止器禁用。工作进程在导入原评价器前绑定新的授权检查，已验证原评价函数引用也不再受旧截止限制。原训练/评价文件和最终权重均未修改。回执为`diagnostics/unlimited_evaluation_monitor_verified.json`，旧运行信息保存在`operations/time_budget_cancelled_by_user_20261015/`。下方旧PID/预算只用于追溯。

北京时间2026-10-15 19:08正式训练启动。此文件是执行入口；科学问题、三组差异、指标与结论边界集中在[方案](../next_experiment_review_20261015/实验设计_审阅稿.md)。

**22:25之后最新状态：训练已全部完成，评价因外部GPU进程占用而OOM；评价专用恢复已在CPU等待显存。**训练终点SHA256=`c780b95df1caaa0ee8117968c8dc22995e11687437176d0e5bfa2e4441cdf0cb`。恢复控制PID为4096710，当前看门PID查看新`watchdog_registration.json`，不要把下方原启动PID当作仍在运行。失败快照在`failures/evaluation_competing_gpu/`。

**监听升级后以此为准：**用户要求更及时地自动恢复，已将30秒轮询改为2秒，并安全交接等待中的控制进程。新监控PID为4120983，看门PID仍以实时`watchdog_registration.json`为准；使用`evaluation_controller.lock`阻止重复监控。GPU查询暂时失败将重试；OOM仍保留部分结果后继续等待。已验证心跳前进、锁及进程存活、最终权重指纹不变，回执为`diagnostics/queue_monitor_verified.json`。旧监控、源码和登记信息保存在`operations/queue_poll_upgrade_20261015/`。原预算与约23:51的最晚保守开始时间不变。

恢复入口是`scripts/c_loop_eval_recovery.py`，不是原训练控制器。它加载同一协议6000步终点，原函数执行256验证、必要补616、188测试，保留部分结果继续评价。`evaluation_queue_registration.json`保存GPU空闲门槛11500MiB、原硬截止及最晚保守开始时间（约23:51）。等待不重置12小时预算，不终止其他进程。当前尚无完整验证/测试结果；若等待耗尽会写资源不足，不能称实验已完成。新增恢复源码的指纹及固定终点在`diagnostics/evaluation_recovery_authorization.json`，原训练脚本和协议未改变。

- 服务器：`li`；目录：`/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/c_feedback_polar_loop_20261015`。
- Python：`/media/li/data3/25thstudent/DY/929work/.venv-pidsr-k2/bin/python`；复用上级`env.sh`，GPU分配器`expandable_segments:True`。
- 协议：`179e35ce4dd0e48594b21592839cbacde8c4f912e082df0195bcfe8b0f0cea18`。
- 原预算起点：1792061352.751717（10月15日18:49:12）；用户明确批准12小时，最终硬截止1792104552.751717（10月16日06:49:12）。不能重置起点或自行再延长。
- 控制进程PID/PGID：3980747；实际工作及看门PID查看`budget.json`、`watchdog_registration.json`。恢复前必须核对进程是否仍活着，禁止重复启动。

## 流程与来源

CPU正确性→GPU小图/梯度/重放/计算复用等价→最大整图资源预检→三组共同6000步→256验证→按精度规则必要时补616验证→完整188测试→汇总、服务器哈希清单与结束回执。三组每一步均更新，三份独立优化器，共享固定第一轮和相同冻结编码结果。各组只训练独立E2适配器与输出头；H、第一轮和PolarFree均固定。

6000步日程SHA256：`709be7c6588525d0b76bb5a381fdcf77185b08cade14e945803009b0f03458b0`。256图ID清单SHA256：`5ac70486b177783294e53b4ea2ab07877fd24b28969e0aa5aa732670cb32dcc5`。来源initial21 SHA256：`af2367fd9f96871cad3b94230638cf5c75acd3b8f8c186cdf210a37b730e82d8`。完整来源、代码和数据指纹在服务器`config/plan.json`；不下载权重、数据、日志、指标或归档。

预检保留：`failures/preflight_rng_device/`、`preflight_io_budget/`、`preflight_full_budget/`、`preflight_cached_budget/`。最初随机状态搬运错误修正后通过整图；GPU原指标和无损未压缩浮点存储减少CPU开销。冻结编码复用在非零输出头条件下验证CPU/GPU的P2、T2及全部可训练梯度逐元素一致。全部试探更新丢弃，无正式训练重启。

最终资源预检：含加载的三组有效步1.56987秒，含加载/指标/保存的三组整图15.20704秒，最大1024×1224，已分配峰值10,815,184,384字节。常规保守估计25,156.73秒，必要全量39,208.03秒；公式含1.5倍余量及900秒。训练硬截止1792079473.5545917由最坏评价预留反推；不是新的训练步数选择规则。

## 检查状态与完成条件

读取服务器`status.json`、`logs/training.csv`尾部及`logs/run.log`；启动后已核对前23行，三组损失/梯度有限。冻结核对与检查点每500步保存，最终只使用`weights/final.pt`的6000步终点。不要根据训练中间权重反复验证或根据测试改参数。

完成应同时具备：6000行连续训练记录；5440独立训练图、12000次呈现；冻结核对通过；完整256或按规则补齐872验证；188张测试；`reports/final_summary.json`、`artifact_manifest.json`和成功`closeout.json`。只看到进程启动或训练结束不能称本实验完成。负面及场景不一致结果照常保留。

`c_loop_controller.py --resume-preflight`仅允许尚无`weights/initial.pt`和训练CSV的预检恢复。正式训练已经开始，不能用该选项重新从零跑。若发生执行错误，应先保留当前权重、日志及预算，在同协议、同原起点内定位；不得重置预算、变更科学变量或拿较早权重冒充6000终点。
