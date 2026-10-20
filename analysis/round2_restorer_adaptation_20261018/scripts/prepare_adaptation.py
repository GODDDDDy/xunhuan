"""CPU-only preparation of experiment 36's immutable exploratory matched pair.

Run only after all experiment scripts and PROTOCOL_ZH.md have been deployed.
Existing locked.json is verified and preserved, never replaced by this script.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import copy
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
OLD33 = PROJECT / "confidence_causal_completion_20261017"
OLD32 = PROJECT / "confidence_correction_causal_20261017"
OLD30 = PROJECT / "c_feedback_polar_loop_20261015"
MU = 0.4459386889966336


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def fingerprint(path, role=None):
    path = Path(path).resolve()
    assert path.is_file(), str(path)
    record = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
    if role:
        record["role"] = role
    return record


def protocol_id(plan):
    value = dict(plan)
    value.pop("protocol_id", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def verify_existing(plan):
    assert plan["experiment"] == 36
    assert plan["protocol_id"] == protocol_id(plan), "Locked protocol ID mismatch"
    for category in ["sources", "code_sources"]:
        for receipt in plan[category]:
            assert sha(receipt["path"]) == receipt["sha256"], receipt["path"]
    assert sha(plan["schedule3000_path"]) == plan["schedule_sha256"]
    print(json.dumps(dict(prepared=True, preserved_existing_lock=True,
                          protocol_id=plan["protocol_id"]), ensure_ascii=False))


def main():
    for folder in ["config", "weights", "logs", "metrics", "reports", "diagnostics",
                   "failures", "examples", "cache"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    lock = ROOT / "config/locked.json"
    if lock.exists():
        verify_existing(read(lock))
        return

    protocol = ROOT / "PROTOCOL_ZH.md"
    assert protocol.is_file(), "Deploy registered protocol before preparing"
    for script in ["run_adaptation.py", "audit_adaptation.py", "controller.py",
                   "prepare_adaptation.py"]:
        assert (ROOT / "scripts" / script).is_file(), script
    assert not (ROOT / "weights/initial.pt").exists(), "Unregistered weights already exist"
    assert not (ROOT / "worker_registration.json").exists(), "Worker was registered before lock"

    reference_cfg_path = OLD33 / "config/execution_config_2026.json"
    reference_schedule_path = OLD33 / "config/training6000_2026.json"
    cfg = copy.deepcopy(read(reference_cfg_path))
    original = read(reference_schedule_path)
    assert original["total_steps"] == 6000
    assert original["batch_size"] == 2 and original["crop_size"] == 128
    assert len(original["items"]) == 12000
    schedule = copy.deepcopy(original)
    schedule["items"] = copy.deepcopy(original["items"][:6000])
    schedule["total_steps"] = 3000
    schedule.pop("sha256", None)
    schedule["sha256"] = hashlib.sha256(json.dumps(
        schedule, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    schedule_path = ROOT / "config/training3000.json"
    write(schedule_path, schedule)

    manifest_path = Path(cfg["paths"]["manifest"])
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    train = [row for row in manifest if row["split"] == "train"]
    assert train, "Empty train split"
    for item in schedule["items"]:
        index = int(item["sample_index"])
        assert 0 <= index < len(train), "Schedule index outside train split"
        sample_id = train[index]["sample_id"]
        assert sample_id.split("/")[0] == "train", sample_id
        if "sample_id" in item:
            assert item["sample_id"] == sample_id, "Schedule ID/index mismatch"
        assert int(item["top"]) >= 0 and int(item["left"]) >= 0

    validation_path = OLD32 / "config/validation64.json"
    validation_ids = read(validation_path)["sample_ids"]
    assert len(validation_ids) == len(set(validation_ids)) == 64
    lookup = {row["sample_id"]: row for row in manifest}
    assert all(sample_id in lookup for sample_id in validation_ids)
    val_rows = [lookup[sample_id] for sample_id in validation_ids]
    assert all(row["split"] in ["val", "validation"] for row in val_rows)
    scene_counts = Counter(row["scene"] for row in val_rows)
    assert len(scene_counts) == 8 and set(scene_counts.values()) == {8}, scene_counts
    calibration_path = OLD33 / "config/calibration.json"
    calibration = read(calibration_path)
    assert calibration["training_only"] is True
    assert calibration["uses_GT"] is False
    assert calibration["samples"] == 256
    assert float(calibration["mu"]) == MU

    cfg["seed"] = 2026
    cfg["paths"]["schedule"] = str(schedule_path)
    cfg["paths"]["output_root"] = str(ROOT / "unused_output")
    cfg["downstream"].update(steps=3000, minimum_steps=3000,
                             effective_batch=2, microbatch=1)
    removed_deadline = cfg["downstream"].pop("deadline_unix", None)
    cfg["experiment36"] = dict(
        authoritative_protocol="config/locked.json",
        exploratory_only=True, arms=["A", "B"], steps=3000,
        trainable="independent second-round net_g only; eval mode",
        effective_batch=2, microbatch=1, crop_size=128, seed=2026,
        lr=1e-5, betas=[0.9, 0.999], eps=1e-8, weight_decay=0,
        gradient_clip_norm=1.0, scheduler=None, amp=False,
        mu=MU, bound=0.02, validation=64, test=0,
        full_resolution_evaluation=True, deadline_unix=None,
        loss="MSE + .02 * .5 * (horizontal edge MAE + vertical edge MAE)")
    cfg["historical_inherited_fields_not_authoritative"] = dict(
        note="Inherited loader/provenance config only. Experiment36 protocol is authoritative; "
             "old training, selection, scope, budgets, deadlines, and C/action protocols "
             "must not be executed.",
        fields=["scope", "pretrain", "confidence", "evaluation", "selection",
                "optimization", "retention_monitor", "experiment_rationale",
                "correction_design", "mechanism_diagnostics", "reuse",
                "local_acceptance", "experiment", "staged", "error_guidance",
                "protocol_id", "gain_study", "budget"],
        removed_downstream_deadline_unix=removed_deadline)
    cfg_path = ROOT / "config/execution_config.json"
    write(cfg_path, cfg)

    fixed_f_path = OLD33 / "weights/final.pt"
    common_initial_path = OLD30 / "weights/common_initial21.pt"
    reference_metrics_path = OLD33 / "metrics/validation872.json"
    sources = [
        fingerprint(fixed_f_path, "fixed 33F 6000-step polarization correction"),
        fingerprint(common_initial_path, "frozen original first-round E1/H/D1"),
        fingerprint(calibration_path, "train-only fixed confidence mean"),
        fingerprint(reference_cfg_path, "inherited component loader configuration"),
        fingerprint(reference_schedule_path, "original 33 training schedule"),
        fingerprint(manifest_path, "dataset manifest and original split"),
        fingerprint(validation_path, "pre-existing fixed 8 x 8 validation selection"),
        fingerprint(reference_metrics_path, "old full-resolution F/T1 reference metrics"),
        fingerprint(protocol, "registered scientific protocol"),
        fingerprint(schedule_path, "3000-step schedule prefix"),
        fingerprint(cfg_path, "experiment36 component loader configuration"),
    ]
    for key in ["base_config", "reference_r2p", "pretrained_r2p", "source_checkpoint"]:
        path = Path(cfg["paths"][key])
        if path.is_file():
            sources.append(fingerprint(path, "component dependency: " + key))

    for path in [OLD30 / "config/plan.json", OLD30 / "config/execution_config.json",
                 OLD30 / "config/validation256.json", OLD30 / "weights/initial.pt"]:
        sources.append(fingerprint(path, "inherited loader verification dependency"))
    archived_weights = Path(cfg["task_initialization"]["archived_source_dir"])
    assert archived_weights.is_dir(), str(archived_weights)
    for path in sorted(archived_weights.glob("*.pth")):
        sources.append(fingerprint(path, "frozen official PolarFree initialization dependency"))

    code_paths = set((ROOT / "scripts").glob("*.py"))
    for directory in [OLD33 / "scripts", OLD32 / "scripts", OLD30 / "scripts"]:
        assert directory.is_dir(), str(directory)
        code_paths.update(directory.rglob("*.py"))
    source_root = Path(cfg["paths"]["source_root"])
    package = source_root / "rgb_final_confirmation"
    assert package.is_dir(), str(package)
    code_paths.update(package.rglob("*.py"))
    # Lock local inherited implementation files, not the Python/library runtime.
    for directory in [PROJECT / "deps/DfP/archs", PROJECT / "deps/DfP/models",
                      PROJECT / "deps/DfP/utils"]:
        if directory.exists():
            code_paths.update(directory.rglob("*.py"))
    for path in [PROJECT / "deps/r2p_film.py"]:
        if path.is_file():
            code_paths.add(path)
    mandatory = [PROJECT / "deps/DfP/archs/diffusion.py",
                 PROJECT / "deps/DfP/archs/latent_encoder_arch.py"]
    assert all(path.is_file() for path in mandatory), "Prior architecture source missing"
    restormer_paths = [p for p in (PROJECT / "deps").rglob("*.py")
                      if "restormer" in p.name.lower()]
    code_paths.update(restormer_paths)
    # Importing the inherited module chain is CPU-only; never instantiate its
    # components here. This records the files Python actually resolved, including
    # older helpers reached through their own path setup rather than OLD30 itself.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(OLD33 / "scripts"))
    importlib.import_module("completion_run")
    imported_project_files = set()
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix == ".py" and path.is_file() and path.is_relative_to(PROJECT):
            imported_project_files.add(path)
    code_paths.update(imported_project_files)
    resolved_helpers = {}
    for name in ["common", "enhancer", "quality_forward", "gain22", "study", "bridge"]:
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
        else:
            # bridge is imported dynamically only when components are built.
            # Resolve the same inherited sys.path without executing components.
            spec = importlib.util.find_spec(name)
            assert spec is not None and spec.origin, "Unresolved inherited helper: " + name
            path = Path(spec.origin).resolve()
        assert path.suffix == ".py" and path.is_file(), str(path)
        assert path.is_relative_to(PROJECT), "Helper outside project: " + str(path)
        resolved_helpers[name] = str(path)
        code_paths.add(path)
        # Dynamic helper siblings can be imported later during component setup.
        code_paths.update(path.parent.rglob("*.py"))
    code_sources = [fingerprint(path) for path in sorted(code_paths)]
    assert len(code_sources) > 3

    plan = dict(
        schema_version=1, experiment=36, exploratory_only=True,
        authorized=True, arms=["A", "B"], steps=3000, lr=1e-5,
        mu=MU, seed=2026, crop_size=128, effective_batch=2, microbatch=1,
        gradient_clip_norm=1.0, bound=0.02,
        optimizer=dict(kind="Adam", betas=[0.9, 0.999], eps=1e-8, weight_decay=0),
        scheduler=None, amp=False, model_eval_mode=True,
        trainable="independent second-round PolarFree net_g; all net_g parameters",
        validation_ids=validation_ids,
        validation_scene_counts=dict(sorted(scene_counts.items())),
        evaluation=dict(validation=64, test=0, original_size=True, tiles=False,
                        fixed_endpoint=3000, fixed_evaluation_seed=2026),
        fixed_f_path=str(fixed_f_path), common_initial_path=str(common_initial_path),
        calibration_path=str(calibration_path),
        reference_metrics_path=str(reference_metrics_path),
        schedule3000_path=str(schedule_path), schedule_sha256=sha(schedule_path),
        reference_schedule_path=str(reference_schedule_path),
        train_cfg_path=str(cfg_path),
        sources=sources, code_sources=code_sources,
        imported_project_code_count=len(imported_project_files),
        resolved_inherited_helpers=resolved_helpers,
        positive_thresholds=dict(
            comparisons=["B-A", "B-T1"], scene_equal_psnr_min=0.05,
            scene_equal_ssim_min=0.0, positive_psnr_scenes_min=6,
            scenes_total=8, role="investment gate, not statistical significance"),
        stopping_rule="Fixed 2 x 3000 updates and fixed64; no automatic extension, "
                      "hyperparameter search, or omitted failed arms.",
        confirmation_rule="Only if both gates pass: register confirmation and evaluate "
                          "the same two fixed3000 endpoints on 872 validation + 188 test; "
                          "no new training and no test-based checkpoint choice.",
        server_only=True, deadline_unix=None)
    plan["protocol_id"] = protocol_id(plan)
    # Creation is exclusive: a competing preparer can never replace a locked plan.
    with lock.open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(prepared=True, protocol_id=plan["protocol_id"],
                          arms=plan["arms"], steps=3000, validation=64,
                          schedule_sha256=plan["schedule_sha256"],
                          source_files=len(sources), code_files=len(code_sources)),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()



