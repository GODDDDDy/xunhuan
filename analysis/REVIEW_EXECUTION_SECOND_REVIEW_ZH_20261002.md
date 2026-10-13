# 执行汇报二次复核：梯度路径错误及后续决策

日期：2026-10-02。只读核对执行汇报、梯度报告、原始JSON、诊断脚本、服务器正式run_experiment.py。未运行GPU前向/反传，未训练，未修改执行方脚本或检查点。

## 核心结论

接受逐场景损失分解与当前criterion核查作为有效补充。不能接受汇报以现有梯度诊断宣称“phase没有把方向带偏”“损失对照门槛不满足”。首先存在反馈路径遗漏，其次门槛判读比较了不同问题。

正式训练及已完成alpha扫描不因本次诊断脚本错误而被推翻；问题定位在新增梯度诊断的feedback第二轮路径。

## 1. 已确认的实现问题：feedback诊断绕过反馈

本地server_scripts/gradient_diagnostics/gradient_diagnostics.py第325行附近，forward_pair的第二轮调用为：

```python
P2o, _ = model.r2p(condition, inp2, return_features=True)
```

它绕过JointR2P.forward；T1提取的f1o并未传给第二轮。JointR2P.feedback_adapter和feedback_scale不在这条第二轮路径中。

服务器正式experiment/run_experiment.py的recurrent_forward第383–385行调用：

```python
second = model(first.refined_polar, batch["condition"],
               feedback=f1 if model.feedback_enabled else None,
               project_features=False)
```

反馈特征是在JointR2P.forward内部注入的，而不在底层model.r2p里。服务器实际诊断脚本与本地副本SHA256同为cf421fa04c684b17de8fead63a263964f176a714ff806e8682d2540f0cc65a47，故不是仅本地展示版本的问题。

现有feedback梯度记录应标为“feedback训练检查点在禁用反馈路径下的诊断”，保留原数据，不用其推断正常feedback第二轮或真实训练梯度。第一轮及no-feedback分项仍可按其实际覆盖范围使用，但no-feedback正式路径的数值一致性也应补一个小批对照。

## 2. 两处关键判读不成立

### 分项冲突被错误替换成两轮冲突

此前建议的“分项训练梯度支持冲突”，指phase、L1、SSIM等分项之间的梯度关系。汇报用两轮总任务梯度只有1/10负夹角否定门槛，回答的是另一个问题。

在现有no-feedback真实链式路径的5批数据中，第一轮phase-L1为负3/5、phase-SSIM为负3/5；第二轮phase-L1为负3/5、phase-SSIM为负2/5。它们支持所查批次的分项张力线索，已经构成设计一个最小目标对照的理由；不证明跨批次/跨seed普遍冲突，更不证明导致验证集退化。L1不是PSNR的直接MSE目标，两者不能等同。

### 用第二轮总梯度证明phase无害

诊断的L2 weighted vs TASKSUM比较整个第二轮任务梯度与0.5*gL1+gL2；不是phase分项与去掉phase后的梯度。总梯度本身含phase，即使测得phase与总梯度同向，也不能证明它符合像素/结构目标。

“phase没有把更新方向带偏”应撤回。方向是否有利必须明确相对于哪个目标；参数空间梯度、Adam动量/预条件、STE代理梯度与验证表现也不能直接等同。

## 3. 范围与表述修正

- model.r2p共享块范数低于1只是全部被裁剪参数联合范数的下界，不能据此证明正式裁剪未触发。需覆盖projections、feedback_adapter、feedback_scale；net_g按正式独立参数组另算。现有文件不能将“全部批次裁剪未触发”作为已测结论。
- 所有梯度都在step3000权重上计算。200/600/1200/2000/2800是调取训练schedule的批次索引，不是这些历史时点模型的梯度，不能写成跨早中晚训练轨迹。
- real-path的gL1与gL2也可用于问实际损失项是否冲突；isolated改变了图，只适合归因额外路径。修正feedback后，若isolated要切断第一轮影响，须同时detach P1和反馈F1。正式real-path保留两者连接，不得顺手detach。
- 当前依赖文件与当前公开文件hash一致证明当前版本一致；对历史训练也有效需核对运行时保存的源码hash/快照。不要由现状自动推回历史。
- 已知0.5*L(T1)早在既有K2协议中明确，并非新增事实。标量反事实重新加权不需要链式法则；链式法则用于梯度。
- “近似正交”不能推出语义无关或构造必然如此；alignment影响仅限定本批次、当前参数块与目标系数。

## 4. 最小修复及复跑要求

不重跑872张，不扩大诊断批次，不开始训练。

1. 修正诊断forward_pair，优先复用正式recurrent_forward与已有模型wrapper；real-path第二轮包含真实反馈。若必须自己拆图，应逐项对照正式调用，包括TaskNet激活重计算与RNG恢复，不能仅复制公式。
2. 先对两组各一批比较诊断路径与正式forward_losses的P1/P2/T1/T2和损失，保存最大差及约定容差；再比较诊断分项加权和的共享参数梯度与正式目标梯度。输入相同不意味着训练路径梯度已匹配。
3. 确认反馈分支实际参与：记录传入F1、feedback_scale及注入张量；反馈关闭可作额外反事实，但不能代替正常路径。不要要求正常反馈的梯度每批都必然非零，可能存在零/退化例子，应如实报告。
4. 保持原5批、step3000检查点与精度设置，仅重跑必要诊断。除了既有夹角，再直接计算两轮加权phase合计梯度与加权非phase合计梯度，以及目标在phase权重1和0.1下的梯度范数/夹角。若加权非phase合计梯度范数接近零，夹角标为不可解释。
5. 用实际裁剪参数集合记录联合范数；保留real-path为主，isolated为补充。无optimizer.step。
6. 存为新版本并标注旧feedback路径错误。上述修复不需要新增全量推理或seed。

## 5. 下一项训练的建议（本轮未启动）

现有no-feedback分项线索和alpha扫描已经足以提出最小目标对照，不要求先证明phase是原因。为控制投入，优先no-feedback两组：phase系数1与0.1。0.1是预先选择的一个参照，不是保证最优，也不是完整PolarFree复现。

如果问题是“目标权重是否改善原K2训练”，建议两组都从原step0初始化来源重新训练至3000步，而不是仅把新组从step3000继续训练与旧组比较。严格匹配R2P/net_g初始权重、数据/裁剪schedule、seed2026、随机流、精度、优化器、训练模式、裁剪集合、round_weights=[0.5,1.0]、反馈关闭与固定alignment系数0.030651881051471687。只改变每轮任务phase系数，其他loss不变；不重新校准lambda，否则加入第二个变量。

该训练属于新的单seed目标消融。两轮结果与历史记录可并列，主要因果对照是新运行之间。评价同时看T2-T1、最终T2、PSNR/SSIM两聚合、逐场景及分项损失。不能只凭T2-T1变正宣称整体方法改进，因为可能是T1变差；需对照绝对表现与普通微调G6。

先完成小批路径修复与核对，再启动该匹配训练是推荐顺序。如果目标对照无明确收益，再进入固定第一轮的独立纠错器，不同时改结构和loss。若要研究当前检查点的继续适应，必须另行将两组定义为同step3000起点、新建同配置优化器和相同追加预算，其结果不能作为从step0训练效应。

本文件仅提出修复与实验选择，未授权或启动任何新GPU运行。
