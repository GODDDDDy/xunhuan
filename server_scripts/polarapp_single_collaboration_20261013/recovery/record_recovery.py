"""Record execution-only repair with unchanged source, final model and protocol."""
import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "scripts"))
from runtime import atomic_json, read_budget, read_locked_config, sha, identity_matches
locked = read_locked_config(root)
budget = read_budget(root)
for name, digest in locked["code_sha256"].items():
    if sha(root / name) != digest:
        raise RuntimeError(f"Frozen code changed: {name}")
controller = json.loads((root / "controller.json").read_text())
active = json.loads((root / "active_gpu.json").read_text())
if not controller["finished"] or identity_matches(controller.get("identity")) or active.get("active"):
    raise RuntimeError("Prior controller/GPU worker still active")
final = root / "downstream/checkpoints/meta/final.pt"
if sha(final) != "f7081db214a201a61a0c67300e564c26af394b048b978bb363d6668b10c886b4":
    raise RuntimeError("Completed final weight changed")
if (root / "downstream/metrics/evaluation_context.json").exists():
    raise RuntimeError("Expected failure before final context/metrics")
spec = importlib.util.spec_from_file_location("selection_repair", root / "recovery/evaluate_existing_final.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ns = {"__name__": "CPU_selection_audit", "__file__": str(root / "recovery/evaluate_existing_final.py")}
exec(compile(module.patched_source((root / "scripts/run_evaluation.py").read_text()), "selection_repair", "exec"), ns)
with open(locked["manifest"]) as stream:
    rows = [SimpleNamespace(**{**row, "frame": int(row["frame"])}) for row in csv.DictReader(stream)]
selected = ns["example_selection"](rows)
records = [{"sample_id": r.sample_id, "scene": r.scene, "difficulty": r.difficulty, "frame": r.frame}
           for r in rows if r.sample_id in selected]
if len(records) != 16 or any(r.split != "validation" for r in rows if r.sample_id in selected):
    raise RuntimeError("Invalid deterministic fixed examples")
receipt = {"time_unix": time.time(), "protocol_id": locked["protocol_id"],
           "budget": budget, "budget_sha256": sha(root / "budget.json"),
           "final_sha256": sha(final), "completed_training_steps": 3000, "meta_updates": 300,
           "frozen_code_sha256_unchanged": True, "retraining": False,
           "model_inference_and_metric_algorithms_unchanged": True,
           "repairs": ["actual example-role availability and unique output paths",
                       "byte snapshot of mutable CPU bookkeeping during archive"],
           "example_selection_used_GT": False, "fixed_examples": records,
           "execution_wrapper_sha256": {p.relative_to(root).as_posix(): sha(p) for p in sorted((root / "recovery").glob("*.py"))},
           "failed_temporary_archive_is_unverified": True,
           "selection_fix_not_model_selection": True}
atomic_json(root / "diagnostics/evaluation_recovery_20261013/recovery_receipt.json", receipt)
print(json.dumps(receipt, ensure_ascii=False))
