"""Pre-GPU path-only repair; preserve failed lock/config/code and CPU log."""
import hashlib,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
assert not (ROOT/'budget.json').exists(),'No pre-GPU repair after budget registration'
old=ROOT/'diagnostics/cpu_preflight_attempt1';old.mkdir(exist_ok=False)
for name in ('config/locked_run.json','config/lock_receipt.json','config/execution_config.json','logs/cpu_verify.log'):
    source=ROOT/name;target=old/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
cfg=json.loads((ROOT/'config/execution_config.json').read_text())
source=json.loads((ROOT/'config/source_locked_run.json').read_text())
cfg['paths']['source_root']=source['paths']['source_root']
write(ROOT/'config/execution_config.json',cfg)
plan=json.loads((ROOT/'config/locked_run.json').read_text());previous=plan.pop('protocol_id')
plan['execution_config_sha256']=sha(ROOT/'config/execution_config.json')
plan['diagnostic_code_manifest']=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted((ROOT/'scripts').rglob('*.py'))]
plan['execution_path_amendment']=dict(previous_protocol=previous,reason='Dependency project_root validation requires original source import path; only loading path corrected before GPU',algorithm_changed=False,gpu_started=False)
protocol=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest();plan['protocol_id']=protocol
write(ROOT/'config/locked_run.json',plan)
write(ROOT/'config/lock_receipt.json',dict(protocol_id=protocol,time_unix=time.time(),configuration_sha256=sha(ROOT/'config/locked_run.json'),previous_protocol_id=previous,algorithm_changed=False))
write(old/'repair_receipt.json',dict(previous_protocol=previous,new_protocol=protocol,gpu_started=False,source_root=cfg['paths']['source_root'],network_and_weights_unchanged=True,time_unix=time.time()))
print(json.dumps(dict(repaired=True,protocol_id=protocol)))
