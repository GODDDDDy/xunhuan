"""Seal preparation provenance; no GPU work or modification of old runs."""
import hashlib
import json
import os
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root),str(root/"scripts")]
from runtime import atomic_json,sha
from recovery.evaluate_existing_final import patched_source
assert not (root/"budget.json").exists()
assert not (root/"config/locked_run.json").exists()
draft=json.loads((root/"config/draft_run.json").read_text())
for key in ("operational_training_deadline_unix","forecast_evaluation_reserve_seconds"):
    draft.pop(key,None)
draft["ancestry_scope_amendment"]=draft.pop("scope_amendment",None)
draft["matched_joint_updates_not_matched_total_compute"]=True
draft["source_calibration_reused_no_new_calibration"]=True
draft["protocol_document_sha256"]=sha(root/"PROTOCOL_ZH.md")
draft["cpu_setup_receipt_sha256"]=sha(root/"cpu_setup_receipt.json")
draft["cpu_functional_check_sha256"]=sha(root/"diagnostics/cpu_functional_check.json")
draft["prelock_joint_plan_refinement_sha256"]=sha(root/"diagnostics/prelock_joint_plan_refinement.json")
for folder in ("scripts","modules","operations","recovery"):
    for p in (root/folder).rglob("*.py"):
        compile(p.read_text(),str(p),"exec")
compile(patched_source((root/"scripts/run_evaluation.py").read_text()),"patched_evaluator","exec")
assert sha(root/"weights/resume_parent_6000.pt")==draft["resume_parent_final_sha256"]
for name,digest in draft["schedule_hashes"].items():
    assert sha(root/"config"/f"{name}_schedule.json")==digest
atomic_json(root/"config/draft_run.json",draft)
atomic_json(root/"diagnostics/preparation_seal.json",{
    "passed":True,"CPU_functional_check":draft["cpu_functional_check_sha256"],
    "all_python_compile_passed":True,"patched_evaluator_compile_passed":True,
    "post_functional_check_changes":"finite tail-state validation, comments, source integrity assertion and added-window coverage counters; algorithm unchanged",
    "source_sha256":draft["resume_parent_final_sha256"],"draft_sha256":sha(root/"config/draft_run.json")})
print("Preparation sealed. No budget created and no GPU initialized.")
