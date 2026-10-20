"""Reuse original CPU controller/dispatch; never rerun completed training."""
import hashlib
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "scripts"))
from runtime import read_budget, read_locked_config, sha
config = read_locked_config(root)
read_budget(root)
journal = json.loads((root / "controller_stage_receipts.json").read_text())
if journal["stages"].get("train", {}).get("returncode") != 0 or "--resume" not in sys.argv:
    raise RuntimeError("Recovery only allowed after completed training with original budget")
final = root / "downstream/checkpoints/meta/final.pt"
receipt = json.loads(final.with_suffix(".sha256.json").read_text())
if sha(final) != "f7081db214a201a61a0c67300e564c26af394b048b978bb363d6668b10c886b4" or receipt["step"] != 3000:
    raise RuntimeError("Final weight changed or is not the authorized endpoint")
if receipt["protocol_id"] != config["protocol_id"]:
    raise RuntimeError("Final protocol mismatch")
for name, digest in config["code_sha256"].items():
    if sha(root / name) != digest:
        raise RuntimeError(f"Frozen code changed: {name}")
base = root / "scripts/controller.py"
source = base.read_text(encoding="utf-8")
for old, new in (( 'str(root / "scripts" / "run_evaluation.py")', 'str(root / "recovery" / "evaluate_existing_final.py")'),
                 ( 'str(root / "scripts" / "archive.py")', 'str(root / "recovery" / "archive_live_snapshot.py")')):
    if source.count(old) != 1:
        raise RuntimeError("Unexpected controller dispatch layout")
    source = source.replace(old, new)
exec(compile(source, str(base), "exec"), {"__name__": "__main__", "__file__": __file__})
