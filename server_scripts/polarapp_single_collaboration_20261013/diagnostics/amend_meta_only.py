"""User-authorized scope amendment before meta training or final evaluation."""
import hashlib
import json
import sys
import time
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(root / "scripts"))
from runtime import atomic_json, read_budget, read_locked_config, sha, identity_matches

evidence = root / "diagnostics/user_meta_only_amendment_20261013"
original = evidence / "original_config/locked_run.json"
old = json.loads(original.read_text())
if old["protocol_id"] != "81205f97bfb418b352d33f6f23339161566465054678c8c557e0d36c0f7f2703":
    raise RuntimeError("Unexpected parent protocol")
if any((root / "downstream/checkpoints/meta").rglob("*.pt")) or any(
        (root / "downstream/metrics" / name).exists() for name in
        ("evaluation_context.json", "per_image_validation.csv", "per_image_test.csv", "summary.json")):
    raise RuntimeError("Amendment must precede all meta updates and final evaluation")
controller = json.loads((root / "controller.json").read_text())
active = json.loads((root / "active_gpu.json").read_text())
if not controller.get("finished") or identity_matches(controller.get("identity")):
    raise RuntimeError("Old controller must have exited")
if active.get("active") or identity_matches(active.get("identity")):
    raise RuntimeError("GPU worker still active")
budget = read_budget(root)
if budget != old["budget"]:
    raise RuntimeError("Original budget cannot change")
for name, digest in old["code_sha256"].items():
    if not name.startswith("scripts/") and sha(root / name) != digest:
        raise RuntimeError(f"Algorithm module/source changed: {name}")
checkpoint_receipt = json.loads((evidence / "cancelled_control_checkpoints/last.sha256.json").read_text())
if sha(evidence / "cancelled_control_checkpoints/last.pt") != checkpoint_receipt["sha256"]:
    raise RuntimeError("Cancelled control checkpoint SHA mismatch")
if checkpoint_receipt["protocol_id"] != old["protocol_id"]:
    raise RuntimeError("Cancelled control checkpoint belongs to another protocol")
new = dict(old)
new.pop("protocol_id")
new.update(arms=["meta"], scope="single_round_meta_collaboration_only_user_amended",
           matched_joint_updates_not_matched_total_compute=False,
           matched_finetuning_control_available=False,
           scope_amendment={"parent_protocol_id": old["protocol_id"],
                            "authorization": "User: 只跑协作训练",
                            "time_unix": time.time(), "cancelled_control_step": checkpoint_receipt["step"],
                            "meta_started_before_amendment": False,
                            "control_checkpoint_not_used_for_meta": True,
                            "algorithm_and_meta_schedule_unchanged": True,
                            "budget_reset_or_extended": False})
new["code_sha256"] = {name: sha(root / name) for name in old["code_sha256"]}
new["protocol_id"] = hashlib.sha256(json.dumps(new, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
atomic_json(root / "config/locked_run.json", new)
atomic_json(root / "config/lock_receipt.json", {"protocol_id": new["protocol_id"],
            "locked_sha256": sha(root / "config/locked_run.json"), "locked_at_unix": time.time(),
            "parent_protocol_id": old["protocol_id"], "scope_amendment_receipt":
            "diagnostics/user_meta_only_amendment_20261013/amendment_receipt.json"})
receipt = {"parent_protocol_id": old["protocol_id"], "protocol_id": new["protocol_id"],
           "original_locked_sha256": sha(original), "new_locked_sha256": sha(root / "config/locked_run.json"),
           "original_budget_sha256": sha(root / "budget.json"), "cancelled_control": checkpoint_receipt,
           "meta_source_sha256": new["source_checkpoint_sha256"], "joint_steps": new["joint_steps"],
           "meta_updates": new["meta_updates"], "meta_optimizer_and_rng_reused": False,
           "new_code_sha256": new["code_sha256"], "authorization": "User: 只跑协作训练",
           "full_validation_test_still_required": True, "preflight_updates_reused": False}
atomic_json(evidence / "amendment_receipt.json", receipt)
read_locked_config(root)
print(json.dumps(receipt, ensure_ascii=False))
