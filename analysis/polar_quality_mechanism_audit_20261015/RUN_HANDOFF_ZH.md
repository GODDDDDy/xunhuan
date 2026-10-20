# 第28项只读机制审查交接（执行中，不训练）

## 最新存储要求：不再下载实验资料

用户明确要求：“以后不需要下载，数据就放在服务器上。”本条覆盖下文历史的下载/本机解包/本机浮点分析指令。第28项及后续的数据、权重、指标、日志、浮点示例、主包/失败包/补充包都留在服务器，直接在那里做CPU分析、清单及SHA核验；通过工具读取摘要后，仅在F:\WORK更新通俗实验记录和文字报告。不再scp这些实验产物到本机；已有本机资料不自动删除。

本机`analyze_audit.py`已增加可指定服务器路径和可选无绘图模式，上传到服务器reports/cpu_review后执行；`--source27`指旧27根目录，`--baseline`指当前28根目录，`--output`用当前28/reports/analysis28，`--require-baseline`要求872完整，`--no-plots`可在无绘图库时纯NumPy分析。若服务器已有Matplotlib可直接在服务器生成图，缓存置服务器本项目录。本机不再拉取浮点例来分析。归档核验需检查包/manifest大小与SHA、所有成员、来源/冻结代码及872哈希，服务器已有源文件可直接核对，不能只确认tar文件存在。

## 最新用户授权：分析完成后直接执行

用户随后明确要求：**“分析完之后，直接开始执行”**。因此，第28项仍保持只读、原3小时截止不变；但完整分析、核验和文档完成后，应自主执行最有证据支持的最小下一步，不再等用户确认，不能只给计划后停下。本文后面旧的“不自动发起训练”“完成后删除全部跟进”仅保留为历史范围，已被本段对后续阶段的最新授权覆盖。

先分析，后锁定下一项的机制假设、主要变量、必要的匹配对照、来源、固定训练预算和成败标准；原始文献依据与待验证假设分开。优先做短验证，不能凭负面结果盲目加步或堆模块。证据不足以定位根因时，直接执行最小判别实验并如实标为探索。新实验独立目录/协议/预算/watchdog，不能重置或延长28或27预算。正式训练仍须完成全部872验证和188测试的原尺寸评价、归档核验和实验记录。不得测试选模型/调参/续训，GT不输入推理。

automation28已更新为“第28项审查及后续执行（两小时）”。审查结束后先落实后续任务，再更新这份跟进或用一份替代跟进交接，避免重复；健康安静、失败/完成才通知。不应在下一步尚未执行时直接停止全部跟进。

用户2026-10-15贴ChatGPT分析并要求“参考一下，以你的意见为主，去做下一步的动作”。执行范围：共同起点补评、质量/收益关系、实际路径审查；不自动训练新网络，不改第27模型，不凭结果延长预算/改常数。

## 当前唯一运行

SSH别名li；`/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/polar_quality_mechanism_audit_20261015`。

当前协议`70c3533f000715b68aa8264539483b32b85a1b3ee7a953969967eafbf62e2794`，以config/plan.json、locked_run.json、lock_receipt.json为准。旧协议48d4c429879b574e31f3f76b2a273939fb164564d850b0b79a3028fda18551cd只对应首次GPU分配失败，已留证。当前控制组3660485、独立watchdog3660554、评价子进程3660555（实时JSON和/proc优先）。日志logs/controller.log、cpu.log、baseline_validation.log。

独立只读补评预算原开始1792024881.243787，绝对截止1792035681.243787，**北京时间2026-10-15 11:41:21.243787**，3小时不可重置延长。旧第27项已完成且自动化27已删除，绝不重启或恢复旧GPU。当前CPU控制器与watchdog登记匹配；所有GPU必须在本项截止前停止，之后只能CPU整理。用户要求两小时检查，健康无重要变化保持安静。

CPU检查通过：源526模型/436net_g张量严格加载；E=2、任务=2、M=0；GT替换输出不变；全部冻结eval；26实际质量输入路径匹配。首批原尺寸前向T1哈希严格等于第27；有效配对整图约9.12秒，保守剩余估计9001秒，完整补评约2.5小时。

首次GPU首帧因显存分配碎片OOM，无基线输出。diagnostics/gpu_initial_attempt1保留旧代码、配置、状态、原预算、日志及原失败artifacts.tar.gz/ready/manifest。修复只打开PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True、加载后gc/empty_cache、恢复登记实际PID组；算法不变、预算未重置。恢复脚本改动有协议修订追溯。正式恢复已确认前10张以上正常。

## 数据与计算

共同实际源第21initial SHA`af2367fd9f96871cad3b94230638cf5c75acd3b8f8c186cdf210a37b730e82d8`，同第27。复用第27执行源码和训练内固定b0；所有参数冻结。只补872验证，不新读188测试，不重新训练。

每图seed完全复用第27：`set_step_seed(2026,'matched_eval_validation',i,sample_ids)`。共同T1、初始真实U、初始固定U完整前向FP32；每图T1哈希必须等于第27的config/source27_final_validation.json。两种初始U分开，不能用单一U0当两组起点。结果metrics/baseline_partial.json和完成后baseline_validation.json，前8场景固定首图浮点examples/validation。M不执行，GT仅指标，候选通过原来的quality_forward（不是另造推理实现）。

旧第21source_oracle_validation872.json虽然有起点候选，但随机流不同，哈希不匹配；已明确不能直接相减。新补评是为获得可逐图匹配的基线，不是增加新训练比较。

## CPU已完成与待做

本机`analysis/polar_quality_mechanism_audit_20261015`：

- `MECHANISM_REVIEW_ZH.md`已完成代码审查，明确多通路与Q/H冗余尚属假设，C不是直接的全局修改幅度开关；普通MSE重写残差不产生新监督。
- `scripts/analyze_audit.py`已执行endpoint局部分解，`interim_endpoint_decomposition.json`和`figures/harm_decomposition.png`，仅第27项8张固定验证例（不是872空间数据）。4×4区域损害中90.08%—99.90%来自mean((GT−T1)*(U−T1))<=0，余下为方向正但步长过量；这是局部平方误差几何描述，不是根因或SSIM结论。4/16/32尺度均保留。
- `INTERIM_FINDINGS_ZH.md`已写中途结果；完整起点未结束，不得报告2000步改变结论。

补评结束后：下载主artifacts.tar.gz、archive_ready.json、archive_manifest.json到`_dl/polar_quality_mechanism_audit_20261015`，安全解包，逐文件大小/SHA/源权重/代码与runtime_source核验。可复用_dl/polar_confidence_candidate_diagnostic_20261014/operations/verify_archive.py；其根目录由脚本位置决定。主完成status.stage=completed但cpu_analysis_pending=true表示GPU补评完成，**尚不等于整个研究审查完成**。

**原失败包另下载**：diagnostics/gpu_initial_attempt1中的tar/ready/manifest放本机failures/gpu_initial_attempt1；主archive.py排除嵌套tar。失败包ready.completed=false，验证器须明确允许对应failed状态（不能伪称成功包），普通文件/大小/SHA同样严格核验。原tar不改。

然后运行本机脚本：`scripts/analyze_audit.py --require-baseline`。运行用BundledPython C:/Users/14030/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe，PYTHONPATH=F:/WORK/tools/python_libraries/plotting_py312，MPLCONFIGDIR=本分析目录/runtime_cache/matplotlib，PYTHONDONTWRITEBYTECODE=1；新本机文件/缓存全部F:/WORK。

该脚本按sample_id/T1哈希将新872起点与第27终点配对，输出audit_results.json，保存两组各自终点−起点的PSNR/SSIM变化及场景差。8固定例保存起点/终点、真实/固定、4/16/32尺度的C排序、误差/收益、C五分箱与方向/过量分解。每张分开/场景等权，不当独立像素样本；校准不改，GT只评价，分箱不部署。核对脚本字段后再解释结果，不能混淆U−T1与训练前后变化，也不能把未门控局部GT选择说成可部署成绩。

完整结果需回答：2000步各自进步还是退步、起点条件优势与新增训练差异、不同场景是否相反、质量排序与纠错收益是否同向；结构冲突/冗余没有干预证明只能列假设。若仍无有效证据不自动发起新训练或改课题。

维护实验记录第28项和完整结果文档。可复用27的完成补充包方法，把CPU报告/图表/核验作为绑定主包SHA的补充归档；原GPU记录与失败包不改。必要时生成C分箱/方向图，仅8示例范围。查看图片确认布局。

全部补评、CPU分析、主/失败/补充包核验和文档完成后删除本项两小时自动化；如果预算到期GPU未完成，如实报告未完成，禁止继续GPU，保留部分结果并结束自动化。在原截止内可以修复执行/按既有逐图结果恢复，不能改网络算法、采样条件或重置预算。

两小时跟进已创建，automationId=28；完整CPU分析、归档核验及文档完成后删除28。第27项跟进已删除，不能恢复旧任务。

原失败包已另下载到本机failures/gpu_initial_attempt1并核验81文件，包247099565字节、SHA f1491589817a49a41a8b5bace2cd9a0218396efaeb6aebcfd76a7e36c1b97e6b，archive_completed=false。如原包不变无需重复下载。08:48附近恢复补评已记录35/872，前30进度状态正常，T1均严格匹配。

用户随后提出缩减完整评价耗时。后续新步骤优先在启动前登记探索性短试验，用固定64验证（8场景各8张、原尺寸、固定同清单）初筛，不跑188测试、不称完整确认；有希望方案再锁定并运行正式872+188，正式各对照/seed要求不变，不能事后回避正式负面结果。当前28已锁定872补评及原截止不变，本项本来不重复测试。执行下一项时先读analysis/FAST_SCREENING_POLICY_ZH_20261015.md；不得让省时流程削弱正式归因所需对照或伪称稳定。

## 2026-10-15 10:46跟进补充

原GPU补评760/872且健康。已登记一次性服务器CPU接续进程3724120（CUDA_VISIBLE_DEVICES为空），入口reports/cpu_review/postprocess_when_complete.py；它等待主包ready，再流式核验主包与原失败包（不解包不下载），并立即运行analyze_audit.py --require-baseline --no-plots。该过程只做既定CPU分析，不改锁定GPU源码/预算，不训练。reports/cpu_postprocess_status.json记录等待/核验/分析/完成或失败，logs/cpu_postprocess.log、server_archive_verification.log、cpu_final_analysis.log留证。先核查此进程避免重复启动。完成后reports/analysis28/compact_summary.json可直接通过工具读取，完整audit_results.json留服务器。cpu_postprocess的“completed_analysis_ready_for_interpretation”表示数值分析就绪，仍需Agent解释、完善报告与直接落实后续授权。

后续64张验证初筛清单已在服务器reports/fast_screen64_preregistered.json固定，不读成绩：8场景各取sha256(fast_screen64_seed2026_v1|sample_id)最小8张，共64；SHA257a0b2abd704b9b1cbc7ef35e4a057b12631ab96b832097d17a334d18258c60。保存original_validation_index，子集推理须保持原验证索引而非按64重新编号，否则T1噪声改变。此清单没有修改当前872补评。

本机NEXT_HYPOTHESES_PENDING_BASELINE_ZH.md只是待完整基线后筛选的备忘。已复核Q的need=logits.sigmoid().detach()：候选恢复损失不训练Q预测，Q仅BCE代理；这是设计而非bug。可考虑单独检验任务梯度接入Q，而非立即加模块，但不能在完整分析前默认选择。若做该改变，Q应称任务控制图，不能再宣称纯误差概率；即便改善也不自动证明H真实质量图有增量价值。MPRNet原文第3节eq1为端到端恢复目标，3.3为监督注意力，支持该背景但不证明本项目有效。备忘有核验原始链接。
