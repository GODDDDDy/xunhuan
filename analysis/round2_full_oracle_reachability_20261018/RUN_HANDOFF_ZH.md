# 第35项运行入口

用户2026-10-18自主执行授权继续有效。每项新实验先说明问题卡，随后可直接执行，无需再等审批。

## 当前终态：第35项未完成、技术终止，不自动重启

2026-10-18 15:46，原整图预检OOM；唯一一次分层激活重算修复通过128零/非零delta输出/loss/梯度逐元素门控，但整图预检仍OOM。按预先限定停止，不再尝试新内存策略或改变分辨率。正式优化更新0、完成图0/8、无可达性指标；该问题未回答，不能把OOM作为C或二轮能力证据。

服务器status=incomplete_resource_preflight，closeout success=false/terminal=true/no_automatic_retry=true；reports/technical_termination_audit.json=verified_incomplete。源/hash、失败、原截止、零变量/曲线/行、进程退出/锁释放均核对。没有成功科学independent_audit，不得把技术核验当完成。

本项与34均无运行进程；当前诊断链结束，不机械接解冻/RGB头或替代Oracle。自动跟进将暂停；若收到未来指令，先读本页终态而非下面历史，不能因原19:37截止尚未到就重新启动。全局阶段结论见../CONFIDENCE_RESEARCH_STATUS_ZH_20261018.md，35失败细节见RESULTS_ZH.md。

下文为启动及修复历史。

当前为round2_full_oracle_reachability_20261018，服务器目录/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/round2_full_oracle_reachability_20261018。事前设置见PROTOCOL_ZH.md。34完整结束，继任不恢复其旧时限。

固定8未见整图，每图零初始化有界ΔP、Adam.001恰100步，原loss/±.02/噪声固定。不训练网络、不评价C、不自动派生更多Oracle。新4小时预算首次worker登记起算，不重置。数据留服务器。

当前：协议已说明，准备最小复用程序，尚未启动GPU。启动后登记PID/protocol/首次deadline及预检。完成须8×100、来源/冻结/噪声、summary/audit/closeout，集中写RESULTS_ZH.md和实验记录；否则明确未完成。

## 实际启动与唯一内存修复（以下取代上文准备状态）

首次worker于15:37:59登记，start_unix1792309079.1161418，deadline1792323479.116142（北京时间19:37:59），4小时不重置。锁定protocol=c75d9768b2860aa90a4d01994d11b60b6cd8824e0e6b7058d7b61409457d67b8。最初PID1852089。

首次整图真实反传预检OOM：进程用11.42GiB，torch已分配10.98GiB，申请230MiB失败；回执failures/failure_1792309091.json。零正式优化更新，无优化结果，报告未完成。原代码/audit/locked备份在failures/implementation_v1_full_backward_OOM。预启动静态审阅也曾修复曲线/噪声恢复问题，其v1锁定与代码保留在operations/prelaunch_static_v1，均无GPU更新或科学结果。

主存仅32GiB，不采用整网CPU卸载。依据协议允许同协议执行修复，根代理决定仅做一次原生torch.utils.checkpoint分层激活重算（TransformerBlock、use_reentrant=False、preserve_rng_state=True），替代整网重算粒度；不改网络参数、前向公式、8图/100步/目标/幅度。并非把OOM当科学证据。先固定首图左上128的零/小非零delta与原实现逐元素比较输出、loss、输入梯度，完全一致才准许同整图资源预检；任一门控或新整图预检失败，结束本项未完成，不再尝试其他内存策略。CPU卸载只评估未执行，不是已做的修复。

实施revision_1单独记录两个脚本原/新hash并核原备份；config/locked.json原字节与protocol不变。原deadline不变。恢复PID1855292，沿原注册时间；当前正在等待/执行等价门控，不能称完整预检通过。检查reports/execution_parity.json与reports/preflight.json、worker/status/失败；不得看见原OOM后机械无限重启。若implementation_revision_1之后再有失败，先按上述唯一修复上限审查，不加第三种实现。

PyTorch检查点功能用于用重计算换激活存储：[官方文档](https://docs.pytorch.org/docs/2.14/checkpoint.html)。实际环境仍为原torch2.5.1，API也由已安装库提供；门控以本环境实测为准，不升级运行时。独立CPUaudit会核等价门控与修订清单，不能只凭文献宣称等价。
