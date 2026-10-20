"""Keep first-launch evidence before repairing pre-lock execution only."""
import json
from pathlib import Path
import shutil
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/"scripts"))
from runtime import atomic_json,sha,identity_matches
dest=root/"diagnostics/gpu_preflight_attempt1"
assert not dest.exists()
for name in ("controller.json","active_gpu.json"):
    assert not identity_matches(json.loads((root/name).read_text()).get("identity"))
dest.mkdir()
for folder in ("scripts","modules","operations","recovery","config","logs"):
    shutil.copytree(root/folder,dest/folder,ignore=shutil.ignore_patterns("__pycache__"))
for p in root.glob("*.json"):shutil.copy2(p,dest/p.name)
for p in root.glob("*.md"):shutil.copy2(p,dest/p.name)
atomic_json(dest/"failure_receipt.json",{
    "reason":"preflight reads dataset before build_system registers source paths; archive wrapper required nonexistent locked config",
    "formal_training_started":False,"protocol_locked":False,"algorithm_change":False,
    "budget_sha256":sha(root/"budget.json"),"budget_unchanged_on_recovery":True})
print("First-launch logs, code, configuration and state preserved.")
