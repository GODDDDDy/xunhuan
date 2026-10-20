"""Pre-GPU observer-only repair; preserve previous plan and failed CPU records."""
import hashlib,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
assert not (ROOT/'budget.json').exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=ROOT/'diagnostics/cpu_gradient_attempt1';old.mkdir(exist_ok=False)
for name in ('config/plan.json','logs/cpu_verify.log'):
    p=ROOT/name;q=old/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
plan=json.loads((ROOT/'config/plan.json').read_text());previous=plan.pop('protocol_id')
plan['preflight_observer_repair']=dict(previous_protocol=previous,algorithm_changed=False,reason='Read-only path audit must not create autograd saved tensors inside checkpoint; backward numerical checks run without observation hooks, no_grad path checks retained')
plan['code_manifest']=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted((ROOT/'scripts').rglob('*.py'))]
plan['protocol_id']=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
(ROOT/'config/plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2));(old/'repair_receipt.json').write_text(json.dumps(dict(previous_protocol=previous,new_protocol=plan['protocol_id'],gpu_started=False,training_updates_discarded=True,algorithm_changed=False,time_unix=time.time()),indent=2))
print(json.dumps(dict(protocol_id=plan['protocol_id'],gpu_started=False)))
