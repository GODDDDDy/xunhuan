"""Traceable execution-only memory scheduling amendment; same GPU deadline."""
import hashlib,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
old=ROOT/'diagnostics/memory_scheduling_amendment';old.mkdir(exist_ok=False)
for name in ('config/locked_run.json','config/lock_receipt.json','budget.json','logs/gpu_diagnostic_recovery.log','diagnostics/cpu_checks.json'):
    p=ROOT/name;q=old/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
plan=json.loads((ROOT/'config/locked_run.json').read_text());previous=plan.pop('protocol_id')
plan['diagnostic_code_manifest']=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted((ROOT/'scripts').rglob('*.py'))]
plan['memory_execution_amendment']=dict(previous_protocol=previous,algorithm_changed=False,conditions_and_metrics_unchanged=True,source_baseline='evaluate first, save five exact CPU tensors, release GPU baseline before shared-first',allocator='expandable_segments:True',source_condition_helper_unchanged=True)
protocol=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest();plan['protocol_id']=protocol
write(ROOT/'config/locked_run.json',plan);write(ROOT/'config/lock_receipt.json',dict(protocol_id=protocol,time_unix=time.time(),configuration_sha256=sha(ROOT/'config/locked_run.json'),previous_protocol_id=previous,algorithm_changed=False))
budget=json.loads((ROOT/'budget.json').read_text());before=(budget['start_unix'],budget['deadline_unix']);budget['protocol_id']=protocol;write(ROOT/'budget.json',budget);assert before==(budget['start_unix'],budget['deadline_unix'])
write(old/'amendment_receipt.json',dict(previous_protocol=previous,new_protocol=protocol,original_start_unix=before[0],original_deadline_unix=before[1],budget_not_reset=True,algorithm_changed=False))
print(json.dumps(dict(protocol_id=protocol,start_unix=before[0],deadline_unix=before[1])))
