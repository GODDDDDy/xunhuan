"""Preflight-only path correction; preserve failed CPU check, no weight changes."""
from pathlib import Path
import json
import yaml

root=Path(__file__).resolve().parents[1]
assert not (root/'budget.json').exists()
p=root/'config/base_server.yaml'
x=yaml.safe_load(p.read_text());old=x['project_root'];x['project_root']=str(root)
p.write_text(yaml.safe_dump(x,sort_keys=False),encoding='utf-8')
log=root/'logs/cpu_verify.log'
if log.exists():log.rename(root/'logs/cpu_verify_attempt1.log')
(root/'diagnostics/setup_repair.json').write_text(json.dumps(dict(
    reason='Copied dependency requires its containing project root in base config',
    before=old,after=str(root),algorithm_changed=False,weights_updated=False,GPU_used=False),indent=2))
