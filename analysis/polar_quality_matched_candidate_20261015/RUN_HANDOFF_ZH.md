# 第27项运行交接：质量条件的匹配候选短训练

用户在审阅ChatGPT建议后回复“执行”，授权先分析已有浮点结果，再做真实/固定质量两组候选短训练。不是增加M，不重启旧实验。

## 实时核查入口

SSH别名li；唯一目录：/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/polar_quality_matched_candidate_20261015。

当前协议e2eb5a1132f5377a18a3df8fbdf56682e4044751a21e8bcbc6f192ac6bcfaf65；config/plan.json、locked_run.json与lock_receipt.json为准。controller恢复登记PID/PGID3423439，独立CPUwatchdog3423440；实际/proc与JSON优先。当前由scripts/controller.py --resume-execution协调，日志logs/recovery_controller.log；子阶段logs/preflight.log、real_train.log、fixed_train.log、validation.log、test.log。不要把原controller.log中的失败误判成当前失败。

预算原开始1791994752.2271562，硬截止1792037952.2271562，北京时间2026-10-15 12:19:12.227156；训练截止1792023552.2271562，08:19:12；至少4小时完整评价。不可重置延长，截止后禁止本组GPU训练/评价。旧控制入口曾错误写24小时记录，但在实际CUDA分配前只缩短为原开始+12小时，原开始保留。修订和原记录均留证。

流程：preflight→real_quality训练2000→fixed_quality训练2000→完整872验证→188测试→CPU汇总→停止watchdog→归档。北京时间2026-10-15 02:33核查：两组均已完成2000步，逐步sample_id严格一致，各覆盖3640图，0/500/1000/1500/2000的固定探针T1哈希全部不变。完整验证已完成380/872，评价子进程3456368归属原控制组3423439；watchdog3423440活跃，预算与截止未变。测试尚未开始，不依据中途指标改动方案。每两小时检查，正常无重要变化保持安静，只通知失败/完成/需用户处理。

两组终点SHA256：real_quality为aa7a723094ccb693e7a7b0d5f27ed9003d93bfcfd38cd44ac3d80dbfa5428efc，fixed_quality为f5706ad0fd84f80884c2bc4acf9b8c86c046b1f594745e0ab9d21ce32218406c。核查回执在receipts/monitor_20261015_0230.json，最终归档仍须逐文件重新核验。

## 共同起点与算法

weights/common_initial21.pt只读链接到第21initial；SHA256 af2367fd9f96871cad3b94230638cf5c75acd3b8f8c186cdf210a37b730e82d8，来源协议921ae3dc9fe004949efab2afdb434719a11a8268cb80bd44718bc59a9a99787d。这个起点继承第20已有质量条件训练历史，不是从未使用C的中性初始化。构造先用第22归档配置和原43通道组件gain22._components迁移第21final，仅用于建出完整类，然后全模型/net_g严格加载第21initial。实际起点以diagnostics/source_initial_load.json为准，不被构造阶段的weight_audit误导。

两组相同模型/初始化，seed2026，crop128有效batch2微批1，原候选角色3640图和同一固定schedule前2000步/4000裁剪；两组独立fresh Adam，不复用旧优化器/RNG。仅feedback1e-5与error_guidance1e-4更新。第一轮/H/增强核心与旧适配/投影/PolarFree/先验/M全冻结eval。主网冻结保留输入梯度；当前CPU第二轮polar输入梯度L1=0.03690157，参数无梯度。

真实组H原b/C；固定组仅共同起点前16固定判断训练裁剪H预测像素均值b0，C0=exp(-b0/tau)，不读GT统计、不以验证选择常数。所有显式候选条件通路同步控制，含Q预测与注入；固定组仍有原RGB/P0/T1/跨轮特征，因此不叫移除全部质量知识。

原候选loss保持MSE+.02edgeMAE+.1局部MSE变坏惩罚+.1Q BCE。Q预测仅训练GT局部误差代理标签监督、候选损失不反向改变Q预测；GT不进入推理。保留两次真实增强和任务执行、浮点灰度桥、跨轮纠错状态、Q与RGB头；只评价未门控U，两个M不执行。每500保存检查点及固定4训练裁剪监测，T1哈希必须等于起点，不挑权重/调参。

固定2000终点，训练不足2000只能记录未完成，不能冒充完整实验。训练后严格核验两组CSV(step,sample_ids)一致及每组3640图完整覆盖。本次不能称更新全部5440；上游历史见过5440。

## 预检与时间证据

CPU通过：严格来源加载、real新/旧T1/U完全相等、替换GT不改候选、梯度仅feedback/Q、候选与Q预测梯度分离、真实优化器/RNG恢复逐张量一致、冻结参数/第一轮不变。补充实际两次E与两次任务前向、M调用0、冻结主网输入梯度通过。

GPU四个有效batch真实反传和最大1024×1224配对原尺寸前向通过；预检更新全部丢弃。有效步中位0.53617秒，配对整图9.06973秒，峰值10327.23MiB；保守资源估计19437.91秒约5.4小时，满足12小时硬限制。当前是短训练，不能预先宣布收敛。

## 已完成的CPU区域分析

本机analysis/polar_quality_matched_candidate_20261015内region_analysis.json、REGION_ANALYSIS_ZH.md、figures/9PNG与scene_gain.svg已完成。只有8张预先固定的浮点示例，不是28张全部区域。C排序对T1误差有信息，不保证改动收益；30场景easy/0000净MSE退化且最坏10%全部区域贡献71.8%的负收益总量，不能代表其余3张或证明某模块根因。所有统计用浮点；RGB显示裁到[0,1]，GT仅评价。分析不改变已锁定组/损失/步骤/常数。绘图库、缓存均置F:\WORK；训练环境未安装新依赖。

## 失败与修订（不能删除）

1. 源SHA抄写多一字符，setup阻止启动。已用本机、原manifest与服务器三方确认为同一64字符SHA，diagnostics/source_hash_attempt1留原代码/log。
2. 观察hooks在激活checkpoint内创建autograd保存张量，CPU反传检测1855/1843不一致。观察record改为no_grad，反传核验关闭只读hooks；候选算法/值不变。diagnostics/cpu_gradient_attempt1、cpu_gradient_code_attempt1和修订回执保留，CPU再检查通过。
3. copy旧scripts时同名controller/archive覆盖自有入口，旧入口在缺config/preflight.json时失败，无CUDA分配、无正式更新。diagnostics/controller_collision_attempt1、controller_collision_code_attempt1保留旧代码/状态/预算/日志/包；恢复自有入口，只缩短错误24h到原开始+12h，watchdog重登记，不改网络/loss/条件。

第一次失败压缩包已另下载至本机_dl/polar_quality_matched_candidate_20261015/failures/controller_collision_attempt1；包SHA ebcad554d2d347e630ea54aeafbe70da20a3db7975ba87f04e49e6d5b9d3b1c9、494368字节。103清单项核验：100包内普通文件+3绝对Linux链接引用。没有提取链接；三份引用从已有核验本机档案按白名单物化并验证大小/SHA。原包未改，不能声称该失败tar自包含3份权重。archive_receipt.json明确此边界。

## 后续检查与结束要求

先查/proc防重复启动。授权范围内修复执行或从同协议已核验last.pt恢复，保留所有失败，GPU预算不变；恢复登记watchdog新进程组。算法/协议改动必须追溯，不能偷偷增加训练变量。正式训练完成不得重训。若资源不足可卸载无用内容再完整CPU评价，不能分块冒充原尺寸。

最终872验证/188测试共同T1与真实U/固定U逐图/场景PSNR/SSIM、两个差值、场景等权、负面场景与固定每场景首图浮点示例必须齐全。不用测试选组/模型/门槛/常数/步数、续训或追加实验。

真实高于固定但低于T1，只支持减轻退化。单seed、短继续训练、已有C历史、冻结适应范围、反复验证与历史测试探索都保留；不能声称稳定有效或充分收敛。

结束下载artifacts.tar.gz/ready/manifest到本机_dl/polar_quality_matched_candidate_20261015，包及每个文件大小/SHA、共同初始/两组最终权重、代码配置清单、runtime_source副本、训练CSV与全部指标/示例/失败记录核验。自有archive.py对输入链接dereference；主包排除嵌套同名tar，原失败包已分类独立保留并核验。更新科研结果与人话实验记录第27项。完整评价、归档核验与文档完成后删除automation27；截止未完成明确报告并删除，禁止截止后GPU计算。

## 完成核查（2026-10-15）
两组各2000步、872验证/188测试已于04:24:32完成；CPU归档于04:25:31就绪，所有本组训练/评价/控制/watchdog进程已退出。主包190项全部大小/SHA通过。来源与两个终点、冻结代码、运行源码副本、浮点示例独立核对通过。真实−固定验证PSNR/SSIM为−0.000603/−0.000885，测试为+0.019406/−0.000260；两组验证候选均低于T1。不支持稳定收益，不据结果重训。完整报告RESULTS_ZH.md与两张终点配图已生成，完成补充包另存并绑定主包；补充包核验后删除automation27。

**已全部结束：**主包190项及完成补充包44项均核验，服务器补充副本SHA一致；原失败包103项分类保留。旧顶层失败说明的遗留问题已在README_FIRST.md和结果报告明确更正，不修改原主包。automation27已删除，所有本组进程退出。不要重启。最终说明见RESULTS_ZH.md；closeout.json记录归档与停止状态。
