"""Sequential, resumable execution of the locked two-arm experiment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
TRAINER = ROOT / "scripts/run_phase_objective.py"
PYTHON = sys.executable
ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=("sanity", "complete", "evaluate"), required=True)
args = ap.parse_args()

def command(name, operation, extra=()):
    log = ROOT / "logs" / f"{name}_{operation}_{args.stage}.log"
    executable = ROOT / "scripts/evaluate_detailed.py" if operation == "evaluate" else TRAINER
    call = [PYTHON, str(executable), "--config", str(ROOT / "configs" / f"{name}.yaml"),
            operation, *extra]
    status = {"stage": args.stage, "arm": name, "operation": operation,
              "command": call, "start_unix": time.time(), "status": "running"}
    (ROOT / "status.json").write_text(json.dumps(status, indent=2))
    print(f"START {name} {operation} {list(extra)}", flush=True)
    with log.open("w") as handle:
        result = subprocess.run(call, stdout=handle, stderr=subprocess.STDOUT)
    status.update({"finish_unix": time.time(), "returncode": result.returncode,
                   "status": "completed" if result.returncode == 0 else "failed"})
    (ROOT / "status.json").write_text(json.dumps(status, indent=2))
    if result.returncode:
        raise RuntimeError(f"Failed {name} {operation}; see {log}")
    print(f"DONE {name} {operation}", flush=True)

if args.stage == "sanity":
    for name in ("phase1", "phase01"):
        command(name, "preflight")
    for name in ("phase1", "phase01"):
        command(name, "train", ("--steps", "200"))
elif args.stage == "complete":
    for name in ("phase1", "phase01"):
        command(name, "train", ("--steps", "3000", "--resume"))
if args.stage in ("complete", "evaluate"):
    for name in ("phase1", "phase01"):
        command(name, "evaluate", ("--checkpoint", "step_3000", "--split", "validation"))
(ROOT / f"{args.stage}_completed.json").write_text(json.dumps({"completed": True, "time_unix": time.time()}))
print("STAGE COMPLETED", args.stage, flush=True)
