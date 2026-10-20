"""Preserve miscopied controller failure and shorten budget before actual GPU work."""
import hashlib,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=ROOT/'diagnostics/controller_collision_attempt1';old.mkdir(exist_ok=False)
for name in ('config/plan.json','budget.json','watchdog.json','status.json','logs/controller.log','logs/benchmark.log','logs/failure.log','artifacts.tar.gz','archive_ready.json','archive_manifest.json'):
    p=ROOT/name
    if p.exists():
        q=old/name;q.parent.mkdir(parents=True,exist_ok=True)
        if p.name in ('artifacts.tar.gz','archive_ready.json','archive_manifest.json'):shutil.move(str(p),str(q))
        else:shutil.copy2(p,q)
plan=json.loads((ROOT/'config/plan.json').read_text());previous=plan.pop('protocol_id')
plan['controller_copy_repair']=dict(previous_protocol=previous,algorithm_changed=False,reason='Old source copy overwrote generic controller/archive names; old entry failed reading absent preflight config before CUDA allocation; owned entry restored')
plan['code_manifest']=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted((ROOT/'scripts').rglob('*.py'))]
plan['protocol_id']=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest();(ROOT/'config/plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
budget=json.loads((ROOT/'budget.json').read_text());original_start=budget['start_unix'];old_end=budget['deadline_unix'];budget['deadline_unix']=min(old_end,original_start+plan['budget_seconds']);budget['train_deadline_unix']=budget['deadline_unix']-plan['evaluation_reserve_seconds'];budget['protocol_id']=plan['protocol_id'];(ROOT/'budget.json').write_text(json.dumps(budget,indent=2))
(old/'repair_receipt.json').write_text(json.dumps(dict(previous_protocol=previous,new_protocol=plan['protocol_id'],original_start_unix=original_start,erroneous_recorded_deadline_unix=old_end,corrected_deadline_unix=budget['deadline_unix'],budget_not_reset=True,deadline_only_shortened=True,no_CUDA_allocation_in_failed_entry=True,formal_training_not_started=True,time_unix=time.time()),indent=2))
print(json.dumps(dict(protocol_id=plan['protocol_id'],start_unix=original_start,deadline_unix=budget['deadline_unix'])))
