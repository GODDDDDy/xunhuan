"""Independent, CPU-only verification of experiment 36's exploration screen.

Run on the server with --root pointing to the experiment directory.
This script reads existing scientific artifacts and writes only its audit receipt.
It never trains, evaluates a network, downloads data, or creates a closeout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import time
import traceback
from collections import Counter
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ARMS = ["A", "B"]
PAIRS = ["B-A", "B-T1", "A-T1", "F-T1", "B-F"]
TOL = 1e-10


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, description):
    if not condition:
        raise AssertionError(description)


def is_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value)


def finite(value, description):
    value = float(value)
    require(math.isfinite(value), description)
    return value


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def near(actual, expected, description):
    require(abs(finite(actual, description) - finite(expected, description)) <= TOL,
            description)


def verify_receipts(root, plan):
    # A scientific source cannot be revised under an implementation repair.
    source_records = plan["sources"]
    code_records = plan["code_sources"]
    require(isinstance(source_records, list) and source_records, "sources is empty")
    require(isinstance(code_records, list) and code_records, "code_sources is empty")
    code_index = {str(resolve(root, record["path"]).resolve()): record
                  for record in code_records}
    source_paths = {str(resolve(root, record["path"]).resolve())
                    for record in source_records}
    require(len(code_index) == len(code_records), "Duplicate locked code source")
    revisions_path = root / "config/implementation_revisions.json"
    revisions = read(revisions_path) if revisions_path.exists() else []
    require(isinstance(revisions, list), "Implementation revisions must be a list")
    replacements = {}
    preserved = []
    for revision in revisions:
        require(revision["protocol_id"] == plan["protocol_id"], "Revision protocol")
        require(isinstance(revision["replacements"], list) and revision["replacements"],
                "Empty implementation revision")
        for change in revision["replacements"]:
            path = resolve(root, change["path"]).resolve()
            key = str(path)
            require(key in code_index and key not in source_paths,
                    "Only code_sources may be repaired: " + key)
            original = code_index[key]["sha256"]
            require(change["original_sha256"] == original,
                    "Repair original differs from locked code: " + key)
            backup = resolve(root, change["preserved_original_path"]).resolve()
            require(backup != path and backup.is_file(), "Original code not preserved: " + key)
            require(sha(backup) == original, "Preserved original hash differs: " + key)
            require(is_digest(change["repaired_sha256"]), "Invalid repaired code hash")
            replacements[key] = change
            preserved.append(dict(path=key, original_sha256=original,
                                  preserved_original_path=str(backup),
                                  repaired_sha256=change["repaired_sha256"]))
    checked = {}
    for category, records in [("sources", source_records), ("code_sources", code_records)]:
        checked[category] = []
        for record in records:
            path = resolve(root, record["path"]).resolve()
            require(path.is_file(), str(path))
            require(is_digest(record["sha256"]), "Invalid digest: " + str(path))
            current = replacements.get(str(path))
            expected = current["repaired_sha256"] if current else record["sha256"]
            require(sha(path) == expected, "Source changed without verified repair: " + str(path))
            checked[category].append(dict(path=str(path), locked_sha256=record["sha256"],
                                          actual_sha256=expected,
                                          implementation_repair=current is not None))
    checked["implementation_revisions"] = dict(
        count=len(revisions), preserved_originals=preserved,
        revision_manifest_sha256=sha(revisions_path) if revisions else None)
    return checked

def verify_preflight(root):
    preflight = read(root / "config/preflight.json")
    fields = ["passed", "noise_exact", "identity_exact", "resume_exact",
              "same_input_update_exact", "source_frozen_exact"]
    for key in fields:
        require(preflight.get(key) is True, "Preflight: " + key)
    frozen = read(root / "diagnostics/frozen_final.json")
    require(frozen.get("passed") is True, "Final frozen check")
    require(isinstance(frozen.get("before"), dict) and len(frozen["before"]) > 0,
            "Frozen fingerprints missing")
    require(frozen["before"] == frozen.get("after"), "Frozen modules changed")
    return dict(preflight_fields=fields, frozen_module_count=len(frozen["before"]))


def verify_schedule(root, plan, rows):
    path = resolve(root, plan["schedule3000_path"])
    schedule_hash = sha(path)
    require(schedule_hash == plan["schedule_sha256"], "Schedule hash")
    schedule = read(path)
    items = schedule["items"]
    require(len(items) == 6000, "Need 3000 x effective batch 2 schedule items")
    cfg = read(resolve(root, plan["train_cfg_path"]))
    manifest_path = resolve(root, cfg["paths"]["manifest"])
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    train = [row for row in manifest if row["split"] == "train"]
    require(len(train) > 0, "Empty training split")
    train_ids = {row["sample_id"] for row in train}
    expected = []
    for item in items:
        index = int(item["sample_index"])
        require(0 <= index < len(train), "Schedule sample_index outside train")
        sid = train[index]["sample_id"]
        if "sample_id" in item:
            require(item["sample_id"] == sid, "Schedule sample ID/index mismatch")
        for key in ["top", "left"]:
            require(int(item[key]) >= 0, "Negative crop coordinate")
        expected.append(sid)
    actual = [row["sample_ids"] for row in rows]
    expected_pairs = [";".join(expected[2*i:2*i+2]) for i in range(3000)]
    require(actual == expected_pairs, "Training log/sample schedule sequence mismatch")
    require(all(sid in train_ids for row in actual for sid in row.split(";")),
            "Training sample outside train split")
    reference_path = plan.get("reference_schedule_path")
    if reference_path:
        original = read(resolve(root, reference_path))
        require(items == original["items"][:6000], "Original schedule prefix mismatch")
    return dict(schedule_sha256=schedule_hash, presentations_per_arm=6000,
                unique_train_images=len(set(expected)), sequence_verified=True,
                reference_prefix_verified=bool(reference_path),
                manifest_sha256=sha(manifest_path))


def tensor_finite(torch, tensor, description):
    require(isinstance(tensor, torch.Tensor), description + ": not tensor")
    require(tensor.device.type == "cpu", description + ": non-CPU tensor")
    if tensor.is_floating_point() or tensor.is_complex():
        require(bool(torch.isfinite(tensor).all()), description + ": not finite")


def same_tree(torch, left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            same_tree(torch, value, right[key]) for key, value in left.items())
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(
            same_tree(torch, x, y) for x, y in zip(left, right))
    try:
        import numpy as np
        if isinstance(left, np.ndarray):
            return isinstance(right, np.ndarray) and np.array_equal(left, right)
    except ImportError:
        pass
    return left == right


def verify_checkpoints(root, plan):
    import torch
    require(not torch.cuda.is_available(), "Audit must remain CPU-only")
    torch.set_num_threads(2)
    initial = torch.load(root / "weights/initial.pt", map_location="cpu", weights_only=False)
    final = torch.load(root / "weights/final.pt", map_location="cpu", weights_only=False)
    require(initial["step"] == 0 and final["step"] == 3000, "Checkpoint endpoints")
    for checkpoint, label in [(initial, "initial"), (final, "final")]:
        require(checkpoint["protocol_id"] == plan["protocol_id"], label + " protocol")
        require(checkpoint["schedule_sha256"] == plan["schedule_sha256"], label + " schedule")
        require(set(checkpoint["net_g"]) == set(ARMS), label + " net_g arms")
        require(set(checkpoint["optimizers"]) == set(ARMS), label + " optimizer arms")
        require(isinstance(checkpoint.get("rng"), dict) and checkpoint["rng"],
                label + " RNG state missing")
    require(same_tree(torch, initial["net_g"]["A"], initial["net_g"]["B"]),
            "A/B initial net_g differ")
    counts = {}
    for arm in ARMS:
        before, after = initial["net_g"][arm], final["net_g"][arm]
        require(before.keys() == after.keys(), arm + " net_g keys changed")
        require(len(before) > 0, arm + " empty net_g")
        changed = []
        for key, tensor in after.items():
            tensor_finite(torch, tensor, arm + "/" + key)
            require(tensor.shape == before[key].shape and tensor.dtype == before[key].dtype,
                    arm + "/" + key + ": shape/dtype changed")
            tensor_finite(torch, before[key], "initial/" + arm + "/" + key)
            if not torch.equal(tensor, before[key]):
                changed.append(key)
        require(len(changed) > 0, arm + " received no weight updates")
        old_opt, optimizer = initial["optimizers"][arm], final["optimizers"][arm]
        require(not old_opt["state"], arm + " initial Adam state must be fresh")
        require(optimizer["state"], arm + " final Adam state empty")
        require(len(old_opt["param_groups"]) == len(optimizer["param_groups"]),
                arm + " optimizer group count changed")
        state_ids = set(optimizer["state"])
        group_ids = []
        for old_group, group in zip(old_opt["param_groups"], optimizer["param_groups"]):
            require(old_group["params"] == group["params"], arm + " optimizer parameter range changed")
            require(float(group["lr"]) == float(plan["lr"]), arm + " learning rate changed")
            require(tuple(group["betas"]) == (.9, .999), arm + " Adam betas")
            require(float(group["eps"]) == 1e-8 and float(group["weight_decay"]) == 0,
                    arm + " Adam eps/weight decay")
            group_ids.extend(group["params"])
        require(len(group_ids) == len(set(group_ids)), arm + " repeated optimizer parameter")
        require(state_ids.issubset(set(group_ids)), arm + " unknown Adam state parameters")
        for key, state in optimizer["state"].items():
            require(int(state["step"]) == 3000, arm + " Adam step mismatch")
            for field in ["exp_avg", "exp_avg_sq"]:
                tensor_finite(torch, state[field], arm + "/Adam/" + field)
            require(bool((state["exp_avg_sq"] >= 0).all()), arm + " negative Adam variance")
        counts[arm] = dict(state_tensors=len(before), changed_tensors=len(changed),
                           changed_names=changed, optimizer_active_parameters=len(state_ids),
                           optimizer_registered_parameters=len(group_ids))
    # last.pt is a required resumable final state, not an independently chosen checkpoint.
    last_path = root / "weights/last.pt"
    require(last_path.is_file(), "Missing resumable last checkpoint")
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    require(last["step"] == 3000 and last["protocol_id"] == plan["protocol_id"],
            "Last checkpoint endpoint/protocol")
    require(last["schedule_sha256"] == plan["schedule_sha256"], "Last checkpoint schedule")
    for key in ["net_g", "optimizers", "rng"]:
        require(same_tree(torch, last[key], final[key]), "Last/final " + key + " mismatch")
    return dict(initial_sha256=sha(root / "weights/initial.pt"),
                final_sha256=sha(root / "weights/final.pt"),
                last_sha256=sha(last_path), arms=counts,
                initial_weights_equal=True, initial_Adam_fresh=True,
                last_final_states_equal=True)


def pair_summary(rows, left, right):
    result = {}
    scenes = sorted({row["scene"] for row in rows})
    for metric in ["psnr", "ssim"]:
        deltas = [float(row["metrics"][left][metric]) -
                  float(row["metrics"][right][metric]) for row in rows]
        by_scene = {scene: statistics.mean(
            float(row["metrics"][left][metric]) - float(row["metrics"][right][metric])
            for row in rows if row["scene"] == scene) for scene in scenes}
        result[metric] = dict(scene_equal=statistics.mean(by_scene.values()),
                              image_equal=statistics.mean(deltas), scenes=by_scene,
                              scenes_positive=sum(value > 0 for value in by_scene.values()))
    return result


def verify_summary(actual, expected, label):
    for metric in ["psnr", "ssim"]:
        for key in ["scene_equal", "image_equal"]:
            near(actual[metric][key], expected[metric][key], label + "/" + metric + "/" + key)
        require(actual[metric]["scenes_positive"] == expected[metric]["scenes_positive"],
                label + "/" + metric + " positive-scene count")
        require(set(actual[metric]["scenes"]) == set(expected[metric]["scenes"]),
                label + "/" + metric + " scene keys")
        for scene, value in expected[metric]["scenes"].items():
            near(actual[metric]["scenes"][scene], value, label + "/" + metric + "/" + scene)


def meets_reference(comparison):
    return (comparison["psnr"]["scene_equal"] >= .05 and
            comparison["ssim"]["scene_equal"] >= 0 and
            comparison["psnr"]["scenes_positive"] >= 6)


def verify_metrics(root, plan):
    payload = read(root / "metrics/validation64.json")
    rows = payload["rows"]
    ids = plan["validation_ids"]
    require(len(ids) == 64 and len(set(ids)) == 64, "Locked validation IDs")
    require(len(rows) == 64 and [row["sample_id"] for row in rows] == ids,
            "Validation count/order/list differs from lock")
    counts = Counter(row["scene"] for row in rows)
    require(len(counts) == 8 and set(counts.values()) == {8}, "Need 8 scenes x 8 images")
    reference = {row["sample_id"]: row for row in read(
        resolve(root, plan["reference_metrics_path"]))["rows"]}
    require(len(reference) == 872, "Reference must be original 872 validation rows")
    for row in rows:
        sid = row["sample_id"]
        require(sid in reference, "Validation sample outside original reference")
        old = reference[sid]
        require(row["scene"] == old["scene"] and row["shape"] == old["shape"],
                sid + " changed scene/original shape")
        require(row.get("reference_exact") is True and row.get("original_size") is True,
                sid + " missing original-size/reference flags")
        require(row["t1_sha256"] == old["t1_sha256"], sid + " changed T1")
        for key in ["t1_sha256", "a_sha256", "b_sha256", "noise_sha256"]:
            require(is_digest(row[key]), sid + " invalid " + key)
        require(set(["T1", "F", "A", "B"]).issubset(row["metrics"]), sid + " metric arms")
        old_f = old["metrics"].get("R1F", old["metrics"].get("F"))
        require(old_f is not None, "Reference lacks R1F/F")
        for metric in ["psnr", "ssim"]:
            for arm in ["T1", "F", "A", "B"]:
                finite(row["metrics"][arm][metric], sid + "/" + arm + "/" + metric)
            near(row["metrics"]["T1"][metric], old["metrics"]["T1"][metric],
                 sid + "/T1/" + metric)
            near(row["metrics"]["F"][metric], old_f[metric], sid + "/F/" + metric)
    summary = read(root / "reports/summary.json")
    require(summary.get("exploratory_only") is True, "Must remain exploratory screen")
    require(summary.get("protocol_id", plan["protocol_id"]) == plan["protocol_id"],
            "Summary protocol mismatch")
    recomputed = {}
    for label in PAIRS:
        left, right = label.split("-")
        recomputed[label] = pair_summary(rows, left, right)
        verify_summary(summary["comparisons"][label], recomputed[label], label)
    promotion = meets_reference(recomputed["B-A"]) and meets_reference(recomputed["B-T1"])
    require(summary["promotion_pass"] is promotion, "Promotion decision disagrees with fixed rule")
    interaction = {}
    for metric in ["psnr", "ssim"]:
        difference = {
            scene: recomputed["B-A"][metric]["scenes"][scene] -
                   recomputed["F-T1"][metric]["scenes"][scene] for scene in counts}
        interaction[metric] = dict(scene_equal=statistics.mean(difference.values()),
                                   image_equal=(recomputed["B-A"][metric]["image_equal"] -
                                                recomputed["F-T1"][metric]["image_equal"]),
                                   scenes=difference,
                                   scenes_positive=sum(value > 0 for value in difference.values()))
    return dict(frames=64, test_frames=0, scenes=dict(counts), comparisons=recomputed,
                relative_increment_change=interaction, promotion_pass=promotion,
                negative_scenes={
                    label: {scene: value for scene, value in comparison["psnr"]["scenes"].items()
                            if value < 0} for label, comparison in recomputed.items()},
                metric_recalculation_scope="paired summaries from saved metrics; no pixel recomputation",
                reference_T1_and_F_verified=True, exploratory_only=True)


def audit(root):
    plan = read(root / "config/locked.json")
    require(plan["steps"] == 3000 and plan["arms"] == ARMS, "Locked arms/steps changed")
    require(is_digest(plan["protocol_id"]), "Invalid protocol ID")
    require(float(plan["lr"]) == 1e-5, "Locked learning rate differs from declared 1e-5")
    near(plan["mu"], .4459386889966336, "Fixed mu changed")
    receipts = verify_receipts(root, plan)
    preflight = verify_preflight(root)
    with (root / "logs/training.csv").open(encoding="utf-8-sig", newline="") as handle:
        training = list(csv.DictReader(handle))
    require([int(row["step"]) for row in training] == list(range(1, 3001)),
            "Training steps not exactly continuous 1..3000")
    for row in training:
        require(finite(row["seconds"], "Training seconds") >= 0, "Negative training seconds")
        for arm in ARMS:
            require(finite(row[arm + "_loss"], arm + " loss") >= 0, "Negative loss")
            require(finite(row[arm + "_grad_norm"], arm + " gradient") > 0,
                    arm + " missing true gradients")
    schedule = verify_schedule(root, plan, training)
    checkpoints = verify_checkpoints(root, plan)
    metrics = verify_metrics(root, plan)
    return dict(passed=True, completed_exploratory_screen=True, time_unix=time.time(),
                protocol_id=plan["protocol_id"], steps=3000, arms=ARMS,
                cpu_only=True, validation=64, test=0, formal_confirmation_completed=False,
                receipts=receipts, preflight=preflight, schedule=schedule,
                checkpoints=checkpoints, report=metrics,
                locked_sha256=sha(root / "config/locked.json"),
                audit_script_sha256=sha(Path(__file__)),
                limitation="This verifies saved evidence and paired arithmetic, not independent network inference.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    destination = root / "reports/independent_audit.json"
    try:
        result = audit(root)
    except Exception:
        result = dict(passed=False, time_unix=time.time(), cpu_only=True,
                      completed_exploratory_screen=False,
                      traceback=traceback.format_exc(),
                      audit_script_sha256=sha(Path(__file__)))
        write(destination, result)
        raise
    write(destination, result)
    print(json.dumps(dict(passed=True, validation=64, steps=3000,
                          promotion_pass=result["report"]["promotion_pass"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
