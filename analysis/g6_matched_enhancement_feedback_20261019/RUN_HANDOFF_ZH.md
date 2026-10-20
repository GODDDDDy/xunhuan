# 第40项运行交接

2026-10-19，用户明确执行1000步三组方案。协议已锁，服务器单实例控制器启动；当前应先查实时status/预检，不能把控制器启动称训练已开始。

- 目录：服务器 /media/li/data3/25thstudent/DY/929work/pidsr_k2_929/g6_matched_enhancement_feedback_20261019；本机对应analysis目录。
- protocol_id：b216b3a83d25ddd94ed1388b0cf6be4f5273bf0ed7d16c781d81869c8b93103c。
- scientific_id：36b4b06ab86e711e94543889ce767c60fa4a7f0032246f47bae5e1d29ddb5c7c。
- B/R各1000，true batch2/crop128/seed2027原39日程/显式配对noise；A复用39 RESET，先以临时恢复器前100步精确重放检查权重和Adam。G6固定；不使用H/C。
- 复用38架构和39数据/损失/噪声，独立新实例，旧38/39不恢复不修改；无新硬截止。
- 查status.json、controller_registration/worker_registration、heartbeat、单实例锁、config/preflight.json、logs/training.csv末行、metrics/step_1000.json帧数、failures。资源不足现有controller等，不重复启动，不终止其他GPU任务。
- 技术失败保留原源码/回执，code_sources修复须登记config/implementation_revisions.json的原始/保留/修复SHA，科学变量不可改；已正式训练则沿权重/Adam/RNG/日程恢复。
- 完成须两组各1000/同64原图/G6逐图SHA同39/summary/独立CPU审核/服务器哈希归档/final_review/成功closeout。收尾后再只读核验artifact_manifest，避免回执覆盖。
- B或R对A及G6各自满足场景PSNR≥+.05dB、SSIM≥0、6/8场景正才晋级；先说明并登记固定1000终点A/B/R全872/188确认，无新增训练，不只评赢家。均失败停止不扩步/seed/LR/loss/幅度/C/Oracle。
- 数据/权重/指标/日志/图像全部留服务器。本机只文字/自编源码；结果集中RESULTS_ZH.md、此交接、根实验记录、集中研究进度页。
- 正常推进保持安静，仅关键完成、失败、结论变化或需要用户行动时通知。初筛停止/后续确认结束后删除本自动跟进，不归档对话，不派生其它机制实验。

运行登记：controller PID2647700，worker PID2647741。应用自动跟进40-g6已创建ACTIVE，每10分钟检查，仅意义变化通知；正常训练安静。最新初始阶段为preflight_A_reuse，真实B/R梯度/同输入更新、非零状态隔离和恢复检查已走过，尚须A复用及最大整图预检最终成功回执。不能把此记录当正式训练完成。

## 正式训练已确认（2026-10-19 18:48起）

预检全部通过：A临时100步的权重和完整Adam与39 RESET精确相同，故可复用；B/R同输入一步精确相同，非零接口下B无状态泄漏、R实际输出对状态有响应，断点重放和冻结通过。固定64首图恰也是最大图，因此首图/最大图去重后为1个原图，零起点精确等于G6；正式评价仍须全部64匹配39 G6 SHA。

两组已各有30步以上正式CSV，不再是排队或仅预检。热身后约1.02秒/两组匹配更新；按实际训练、64评价、归档审核估计本次初筛约35—50分钟量级，不包含有条件全量确认。首个冷启动步7.27秒生成的2.8小时预估过于保守，不能当稳定训练吞吐。

同协议审计实现修正已登记config/implementation_revisions.json：原审计误要求首图/最大图至少2个不同ID，实际二者相同；现按原规定的ID并集核对，不改样本或预检要求。保留原源码，未停训/重置/改科学变量；重新verify通过。原协议ID和scientific_id不变。

## 最终终态：初筛完成并停止（2026-10-19）

B/R各1000、同64原尺寸评价、CPU独立审计和37项原归档均齐全；19:15:33成功closeout写完。A/B/R对G6分别−.039747/−.044988/−.073650dB，B−A−.005241，R−B−.028661（仅2/8场景正）。两候选均未过门槛，停止本实例，不补872/188、不续步/扩seed/LR/loss/幅度/C/Oracle，不恢复38/39。

注意收尾实现错误：成功回执写完后status函数重复time_unix报错，worker/controller历史退出码1、两份失败必须保留；不能称全程零失败。训练、评价和科学审计未失败。原37项SHA独立重验无失配，不重写原源码、归档、指标或成功回执。另以scripts/closeout_status_review.py做只读CPU复核与最终状态补正，服务器reports/closeout_status_review_20261019存REVIEW、SUPPLEMENTARY_MANIFEST和POSTREVIEW_CLOSEOUT。原archive SHA 3df4ce2faf60ee0ef7dc6ae58f654c66d1ceeba70c5e8af1cbcb2b7743e4df49。

结论/全部负面场景集中RESULTS_ZH.md。无C不能评价H；无平均净收益不能推成所有二轮无空间；本项与38同时换了多项训练条件，未唯一定位38根因。数据全部留服务器。后台进程已退出、两锁释放；自动跟进收尾后删除，不归档对话、不派生其他实验。

收尾补充复核完成：服务器POSTREVIEW_CLOSEOUT success=true、scientific_evidence_valid=true、original_archive_valid=true（37项）、supplementary_archive_valid=true（10项）；补充清单SHA c155aa8cdd8203b3838a664cff1d3e4d7e47ec6b578f00e2e9fe98a3b73b81b1。最终状态completed_metadata_status_repaired，历史worker退出1保留，未覆盖原成功回执及失败。最终状态写完后再次核对两份清单所有SHA通过。

监控删除确认：应用已返回40-g6 deleted。第40项不再自动跟进；旧运行中段落仅为历史，不重启本实例或自动派生新实验。
