"""A2 setup: independent run root, verified source reuse, new 2h GPU budget, patched config.

Reads experiment-22 sources strictly read-only. All write targets are redirected
into this run. Records a brand-new 2-hour absolute GPU budget (no old budget reused).
"""
import hashlib, json, os, shutil, sys, time
from pathlib import Path

RUN = Path(__file__).resolve().parent
SRC = Path("/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/polar_gain_judgement_short_20261012")
FINAL = SRC / "downstream/checkpoints/final.pt"
REUSED = SRC / "weights/reused_final.pt"
EXPECT_FINAL = "279d0a329d0efebee9678c2156bf3e7b60c0646e75a5f37f8015c7b100f9e9a2"
EXPECT_REUSED = "cecdcfa47d290b0a2d6b662b96bb4407a7043ef5789e9e87d70768f2c683916d"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


report = {"run_root": str(RUN), "source_run": str(SRC), "python": sys.executable}
for name in ("docs", "config", "cache", "logs", "metrics", "figures", "diagnostics",
             "receipts", "scripts", "pretraining/checkpoints", "checkpoints/diagnostic_heads"):
    (RUN / name).mkdir(parents=True, exist_ok=True)

report["source_final_sha256"] = sha(FINAL)
report["source_reused_sha256"] = sha(REUSED)
if report["source_final_sha256"] != EXPECT_FINAL:
    raise SystemExit("experiment22 final.pt sha256 mismatch")
if report["source_reused_sha256"] != EXPECT_REUSED:
    raise SystemExit("experiment21 reused_final.pt sha256 mismatch")

copied = []
for path in sorted((SRC / "scripts").glob("*.py")):
    shutil.copy2(path, RUN / "scripts" / path.name)
    copied.append(path.name)
for sub in ("reference", "vendor"):
    base = SRC / "scripts" / sub
    if base.is_dir():
        for path in sorted(base.rglob("*.py")):
            target = RUN / "scripts" / path.relative_to(SRC / "scripts")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied.append(str(path.relative_to(SRC / "scripts")))
report["scripts_copied"] = copied

sel_src = SRC / "pretraining/checkpoints/selected.pt"
sel_dst = RUN / "pretraining/checkpoints/selected.pt"
if not sel_dst.exists():
    shutil.copy2(sel_src, sel_dst)
report["pretraining_selected_sha256"] = sha(sel_dst)

now = time.time()
budget = dict(
    start_unix=now,
    deadline_unix=now + 7200.0,
    pgid=os.getpgrp(),
    process_group=os.getpgrp(),
    scope="stage A2 native-head diagnostic and A3 fixed 32-frame screen; a single new 2h absolute GPU budget",
    stage_A_gpu_budget_seconds=7200,
    old_budget_reused=False,
    created_iso=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
)
(RUN / "budget.json").write_text(json.dumps(budget, indent=2), encoding="utf-8")
report["budget"] = budget

cfg = json.loads((SRC / "config/locked_run.json").read_text(encoding="utf-8"))
cfg["paths"]["output_root"] = str(RUN / "downstream")
cfg["audit_run"] = dict(
    run_root=str(RUN),
    source_run=str(SRC),
    source_final_sha256=report["source_final_sha256"],
    source_reused_sha256=report["source_reused_sha256"],
    created_unix=now,
    stage_A_gpu_budget_seconds=7200,
    note="read-only reuse of experiment 22 sources; every write target is redirected into this run",
)
(RUN / "config/locked_run.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
report["locked_run_sha256"] = sha(RUN / "config/locked_run.json")

(RUN / "receipts/a2_setup.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k != "scripts_copied"}, indent=2, ensure_ascii=False))
print("scripts_copied:", len(report["scripts_copied"]))
