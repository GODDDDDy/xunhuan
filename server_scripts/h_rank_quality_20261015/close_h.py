"""Final non-GPU closeout; preserve locked evidence and original failure."""
import json,os,time
from pathlib import Path
from common import ROOT,sha,json_save
audit=json.loads((ROOT/'reports/audit_summary.json').read_text())
assert audit['screening_passed'] is False and audit['verified_files']==139
active=[]
names={str(ROOT/'scripts'/n).encode() for n in ('h_run.py','h_eval_recover.py','launch_eval_recovery.py')}
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:args=(p/'cmdline').read_bytes().split(b'\0')
    except (OSError,PermissionError):continue
    if any(a in names for a in args):active.append(int(p.name))
assert not active,active
paths=['reports/audit_summary.json','reports/RESULTS_ZH.md','reports/PROTOCOL_EXECUTION_RECORD_ZH.md',
       'artifact_manifest.json','config/plan.json','logs/training.csv','weights/final_heads.pt']
receipt=dict(stage='closed_screen_failed_keep_original_H',steps_per_arm=2000,validation64=64,
             full_validation=0,test=0,all_training_and_evaluation_processes_exited=True,
             new_training_after_screen=0,original_deadline_unchanged=True,
             original_H_replaced=False,data_downloaded=False,
             files={p:dict(bytes=(ROOT/p).stat().st_size,sha256=sha(ROOT/p)) for p in paths},
             time_unix=time.time())
json_save(ROOT/'closeout.json',receipt)
print(json.dumps(receipt,ensure_ascii=False))
