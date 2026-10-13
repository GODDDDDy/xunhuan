"""Small read-only snapshot of progress and completed checkpoint audits."""
import csv
import json
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
status = json.loads((ROOT / "status.json").read_text())
out = {"stage": status["stage"], "active_arm": status["arm"], "operation": status["operation"], "status": status["status"], "arms": {}}
for name in ("phase1", "phase01"):
    root = ROOT / "runs" / name / "seed_2026"
    csvpath = root / "logs/train.csv"
    records = list(csv.DictReader(csvpath.open())) if csvpath.exists() else []
    audits = [json.loads(p.read_text()) for p in (root / "diagnostics").glob("frozen_step_*.json")]
    arm = {"logged_updates": len(records), "last_step": int(records[-1]["step"]) if records else 0,
           "frozen_audits_passed": all(a["passed"] for a in audits),
           "last_audited_step": max((a["step"] for a in audits), default=None)}
    if records:
        arm["last_task_loss"] = float(records[-1]["task_loss"])
    candidates = list((ROOT / "logs").glob(f"{name}_evaluate_*.log"))
    evallog = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else ROOT / "missing_evaluation_log"
    if evallog.exists():
        progress = re.findall(r"Evaluated (\d+) images", evallog.read_text())
        arm["validation_progress"] = int(progress[-1]) if progress else 0
    for kind in ("validation", "validation_loss_parts"):
        path = root / "metrics" / f"step_3000_{kind}.json"
        if path.exists():
            data = json.loads(path.read_text())
            arm[kind + "_frames"] = data["frames"]
    out["arms"][name] = arm
print(json.dumps(out, indent=2))
