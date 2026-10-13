# 修正梯度诊断 v2：结果与范围

日期：2026-10-02。使用既有step3000检查点，不更新权重；每组复查原5个schedule批次（每批2图）。批次编号不代表历史训练时点。

正式recurrent_forward负责real-path，feedback第二轮包含F1；isolated同时断开P1与F1。两组首批的P1/P2/T1/T2与正式forward_losses最大差均为0；全部分项损失重组核对通过，模型/TaskNet全部状态在诊断前后不变。

|组|批次|phase与非phase夹角余弦|权重1与0.1目标梯度余弦|实际model参数组裁剪前范数|梯度重组相对L2误差|
|---|---:|---:|---:|---:|---:|
|no_feedback|200|-0.968726|-0.834726|0.358767|1.563E-004|
|no_feedback|600|0.372849|0.890457|0.197998|1.446E-005|
|no_feedback|1200|0.290133|0.903023|0.205495|6.325E-005|
|no_feedback|2000|-0.917878|0.647987|0.339442|3.776E-005|
|no_feedback|2800|0.745171|0.920997|0.293422|1.160E-004|
|feedback|200|-0.212389|0.779260|0.375654|1.151E-004|
|feedback|600|0.269949|0.908886|0.167250|3.815E-005|
|feedback|1200|-0.396392|0.764537|1.011551|3.813E-004|
|feedback|2000|0.179709|0.828324|0.406498|5.974E-005|
|feedback|2800|0.407133|0.822076|0.293092|7.648E-005|

phase与非phase均指两轮加权合计在共享R2P参数块上的梯度；非phase包含L1、1−SSIM、加权感知项，不含alignment。两种目标梯度包含同一固定alignment项。夹角只是当前权重、所查批次的局部代理梯度，不能直接等同Adam实际更新或验证收益。

no-feedback的批次200、2000出现phase与非phase强负余弦；批次200将phase系数由1改为0.1后，整个目标梯度也出现负余弦。feedback实际model联合范数在批次1200为约1.01，达到裁剪阈值。因此不能保留旧报告的“phase无方向问题”或“所有model裁剪均未触发”结论。仍未证明phase是第二轮退化的原因。

数值核对：原相对误差容差1e-4在首批以1.5629e-4失败，失败日志保留。复跑明确记录容差1e-3及实测误差；精度沿用原TF32设置，未声称梯度逐位一致。JSON内clip_bookkeeping.note保留旧版仅共享块的说明，实际全组范数应读取新增actual_model_group_preclip_norm字段，不能用该旧说明覆盖新测量。

证据：`_dl/phase_objective_20261002/gradient_diagnostics_v2.json`；脚本：`server_scripts/phase_objective_20261002/gradient_diagnostics_v2.py`；服务器原始与复跑日志：`phase_objective_20261002/logs/gradient_diagnostics_v2*.log`。

下一动作：同step0初始化、固定lambda的no-feedback phase1/phase0.1单变量配对训练；这属于探索性目标消融，0.1不是已证明最优或完整PolarFree复现。
