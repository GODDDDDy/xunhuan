# 复现与外部资料

本仓库是科研代码及证据的整理副本。不能将缺少权重、数据及官方依赖的Git目录称为已验证的独立安装包。

## 原环境

- 本机原项目：F:/WORK/projects/pidsr_recurrent_joint_k2。
- 本机仓库整理副本：F:/WORK/projects/xunhuan。
- 服务器SSH别名：li。
- 原实验父目录：/media/li/data3/25thstudent/DY/929work/pidsr_k2_929。
- 既有Python：/media/li/data3/25thstudent/DY/929work/.venv-pidsr-k2/bin/python。
- 外部数据：原PIDSR与PolarFree数据清单及config中的路径；数据本体不在Git。

用户名、目录名只是历史环境定位，不是凭证。仓库没有SSH密钥、访问令牌或登录信息。

## 复现前检查

1. 阅读根AGENTS、CURRENT_STATE、最新结果和当前用户任务授权。
2. 从原归档取得缺失权重/数据；用既有SHA及manifest逐文件核验。不要下载来源不明的替代权重。
3. 保持归档配置只读，在新的run副本中映射source_root、数据、源码、输出、缓存、预算路径。
4. common.ROOT、staged.OUT以及旧verify/probe的写路径必须映射到新run。不能对Git历史快照或原归档直接执行会写文件的函数。
5. 核验实际环境、PyTorch/CUDA、einops、NumPy、OpenCV、YAML、OmegaConf及官方依赖；以原锁定环境和CPU/GPU预检为准，不宣称当前仓库已有完整版本锁。
6. 按原生第22架构构建50通道判断头再加载对应权重，避免重复扩成57通道。
7. 确认两个真实前向、GT隔离、冻结、缓存等价和M0严格复制。
8. 新GPU工作须有新任务授权和新绝对预算，不使用任何已结束的历史budget。
9. 新正式实验按预定终点/选择规则评价，不用测试选权重、调门槛或追加步数。

这些是准备条件，不是自动启动命令。当前仓库不提供会在clone后自动运行训练的流程。

## 关键权重谱系

|权重|SHA256|
|---|---|
|第22最终|279d0a329d0efebee9678c2156bf3e7b60c0646e75a5f37f8015c7b100f9e9a2|
|第21最终|cecdcfa47d290b0a2d6b662b96bb4407a7043ef5789e9e87d70768f2c683916d|
|第20最终|e79af90811baa46aadfb354a682d0c5dad93886e2dfe1eabeaceb407e6cae112|
|第19最终|e8a635e052ec86f9abddeaa6731a3cb85533420aa78d3b0ff52fbac5bd38e299|
|第18最终|ec7f0e6e79d0dcf21c536dc71958d44514bd7e78a8624bb936f16c02c37e81e8|
|PIDSR选定18000步|a4ecebacba5cfc25546320974a9c37f1cc220c685f1318b3cc290ae4a68e9554|
|PolarFree官方net_g|a230e77fb19827a1ada41e4dc86374dee092e476915d2252841f81dd8c257ca2|

net_d/net_le_dm等哈希以相应locked_run和archive_manifest为准。上传清单不意味着这些权重文件在Git中。

## 原始与公开副本

PUBLICATION_MANIFEST.csv记录原项目相对路径、源SHA、公开副本SHA及处理方式。Markdown链接迁移的副本可能与原文档SHA不同，代码/配置/CSV/JSON等按字节保留。

旧archive_manifest描述完整原归档，包含未上传的大文件。它不会与本仓库文件集合完全一致；不能直接拿它判仓库“少文件=归档失败”。核验仓库复制文件请用PUBLICATION_MANIFEST，核验完整实验请取得原归档再用原manifest。

历史图像没有全量同步；最新审查的科学配图和组会材料已包含代表性实例。8位展示图不能代替浮点误差或真实成本。第三方论文以原文链接引用，不复制PDF。
