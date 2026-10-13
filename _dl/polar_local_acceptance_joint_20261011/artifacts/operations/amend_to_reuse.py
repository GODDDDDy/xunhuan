"""Record the pre-budget change after the user asked about weight reuse."""
from pathlib import Path
import shutil
root=Path(__file__).resolve().parents[1]
assert not (root/'budget.json').exists() and not (root/'config/locked_run.json').exists()
destination=root/'diagnostics/official_initialization_cpu_setup'
destination.mkdir(parents=True,exist_ok=True)
for relative in ('config/preflight.json','diagnostics/cpu_checks.json','diagnostics/reuse_audit.json',
                 'logs/cpu_verify.log','logs/cpu_verify_attempt1.log','logs/cpu_verify_attempt2.log'):
    source=root/relative
    if source.exists():shutil.copy2(source,destination/source.name)
(destination/'README_ZH.md').write_text(
    '仅CPU核验已通过，尚未启动GPU。用户随后询问复用上一组权重，方案在正式预算和测试前改为从核验过的上一组最终权重继续训练。保留此准备记录，不将其算作正式实验。首次依赖根目录配置错误、第二次目标替换测试自身扰动随机流的失败均保留，修正后CPU核验通过。\n',encoding='utf-8')
