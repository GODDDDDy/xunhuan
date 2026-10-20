"""CPU-only preparation of experiment 37's immutable state-source contrast.

No component instantiation, GPU probe, model update, or old-run mutation occurs.
Run after the protocol and all four experiment scripts have been deployed.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
OLD36 = PROJECT / "round2_restorer_adaptation_20261018"
OLD33 = PROJECT / "confidence_causal_completion_20261017"
OLD32 = PROJECT / "confidence_correction_causal_20261017"
OLD30 = PROJECT / "c_feedback_polar_loop_20261015"
MU = 0.4459386889966336
ROI, HALO, ALIGN = 128, 8, 8


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_id(value, field="scientific_id"):
    unsigned = dict(value)
    unsigned.pop(field, None)
    return hashlib.sha256(json.dumps(
        unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
    row = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
    if role:
        row["role"] = role
    return row


def verify(plan):
    assert plan["experiment"] == 37 and plan["exploratory_only"] is True
    assert plan["arms"] == ["A", "B"] and plan["steps"] == 3000
    assert plan["scientific_id"] == canonical_id(plan)
    codes = {record["path"]: record for record in plan["code_sources"]}
    assert len(codes) == len(plan["code_sources"])
    replacements = {}
    revision_path = ROOT / "config/implementation_revisions.json"
    if revision_path.exists():
        revisions = read(revision_path)
        assert isinstance(revisions, list)
        for revision in revisions:
            assert revision["scientific_id"] == plan["scientific_id"]
            for replacement in revision["replacements"]:
                path = replacement["path"]
                assert path in codes, "Only locked code_sources may be repaired"
                assert replacement["original_sha256"] == codes[path]["sha256"]
                assert sha(replacement["preserved_original_path"]) == replacement["original_sha256"]
                replacements[path] = replacement
    for receipt in plan["source_receipts"]:
        assert sha(receipt["path"]) == receipt["sha256"], receipt["path"]
    for receipt in plan["code_sources"]:
        expected = replacements.get(receipt["path"], {}).get("repaired_sha256", receipt["sha256"])
        assert sha(receipt["path"]) == expected, receipt["path"]
    assert sha(plan["schedule3000_path"]) == plan["schedule_sha256"]
    assert sha(plan["cache_items_path"]) == plan["cache_items_sha256"]
    return plan


def png_shape(path):
    """Read the immutable PNG IHDR without decoding image pixels."""
    with Path(path).open("rb") as handle:
        header = handle.read(24)
    assert (len(header) == 24 and header[:8] == b"\x89PNG\r\n\x1a\n"
            and header[12:16] == b"IHDR"), "Invalid PNG: " + str(path)
    width, height = struct.unpack(">II", header[16:24])
    assert width > 0 and height > 0, str(path)
    return height, width


def main():
    for folder in ["config", "weights", "logs", "metrics", "reports", "diagnostics",
                   "failures", "examples", "cache", "scripts"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    proposal_path = ROOT / "config/proposal.json"
    if proposal_path.exists():
        plan = verify(read(proposal_path))
        print(json.dumps(dict(prepared=True, preserved_existing_proposal=True,
                              scientific_id=plan["scientific_id"])))
        return

    assert not (ROOT / "worker_registration.json").exists(), "Worker registered before proposal"
    assert not (ROOT / "weights/initial.pt").exists(), "Unregistered trainable weights"
    protocol = ROOT / "PROTOCOL_ZH.md"
    assert protocol.is_file(), "Register scientific protocol first"
    for filename in ["prepare_state_source.py", "controller.py", "run_state_source.py",
                     "audit_state_source.py"]:
        assert (ROOT / "scripts" / filename).is_file(), filename

    old_lock_path = OLD36 / "config/locked.json"
    old_lock = read(old_lock_path)
    assert old_lock["experiment"] == 36
    old_unsigned = dict(old_lock)
    old_unsigned.pop("protocol_id")
    assert hashlib.sha256(json.dumps(old_unsigned, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest() == old_lock["protocol_id"]
    cfg_source = Path(old_lock["train_cfg_path"])
    schedule_source = Path(old_lock["schedule3000_path"])
    assert sha(schedule_source) == old_lock["schedule_sha256"]
    cfg = copy.deepcopy(read(cfg_source))
    schedule = copy.deepcopy(read(schedule_source))
    assert schedule["total_steps"] == 3000 and schedule["batch_size"] == 2
    assert schedule["crop_size"] == ROI and len(schedule["items"]) == 6000
    schedule_path = ROOT / "config/training3000.json"
    write(schedule_path, schedule)
    # Schedule bytes can change formatting, while the complete ordered list is identical.
    assert read(schedule_path) == read(schedule_source)

    manifest_path = Path(cfg["paths"]["manifest"])
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    train = [row for row in manifest if row["split"] == "train"]
    assert len(train) == 5440, len(train)
    shape_cache = {}
    items = []
    image_fields = ["polar_000", "polar_045", "polar_090", "polar_135", "official_rgb", "gt_rgb"]
    for position, scheduled in enumerate(schedule["items"]):
        sample_index = int(scheduled["sample_index"])
        assert 0 <= sample_index < len(train)
        row = train[sample_index]
        sample_id = row["sample_id"]
        if "sample_id" in scheduled:
            assert scheduled["sample_id"] == sample_id
        assert sample_id.startswith("train/"), sample_id
        if sample_id not in shape_cache:
            shapes = [png_shape(row[field]) for field in image_fields]
            assert len(set(shapes)) == 1, (sample_id, shapes)
            shape_cache[sample_id] = shapes[0]
        height, width = shape_cache[sample_id]
        assert height % ALIGN == width % ALIGN == 0, (sample_id, height, width)
        top, left = int(scheduled["top"]), int(scheduled["left"])
        assert 0 <= top <= height - ROI and 0 <= left <= width - ROI, (sample_id, scheduled)
        y0 = max(0, ((top - HALO) // ALIGN) * ALIGN)
        x0 = max(0, ((left - HALO) // ALIGN) * ALIGN)
        y1 = min(height, ((top + ROI + HALO + ALIGN - 1) // ALIGN) * ALIGN)
        x1 = min(width, ((left + ROI + HALO + ALIGN - 1) // ALIGN) * ALIGN)
        assert all(v % ALIGN == 0 for v in [y0, x0, y1, x1, y1-y0, x1-x0])
        assert y0 <= top and x0 <= left and y1 >= top+ROI and x1 >= left+ROI
        items.append(dict(position=position, step=position//2+1, micro=position%2,
                          sample_id=sample_id, scene=row["scene"], sample_index=sample_index,
                          top=top, left=left, height=height, width=width,
                          outer=[y0, x0, y1, x1], inner=[top-y0, left-x0, ROI, ROI]))
        if (position + 1) % 1000 == 0:
            print(json.dumps(dict(preparing_shapes=position+1, total=6000,
                                  unique_images=len(shape_cache))), flush=True)
    assert len(shape_cache) == 5440
    cache_items_path = ROOT / "config/cache_items.json"
    write(cache_items_path, dict(schema_version=1, experiment=37, count=len(items),
                                unique_images=len(shape_cache), items=items,
                                roi_size=ROI, halo=HALO, alignment=ALIGN,
                                schedule_sha256=sha(schedule_path), train_only=True,
                                image_shape_check="all four polar PNGs, original RGB and target PNG"))

    validation_path = OLD32 / "config/validation64.json"
    validation_ids = read(validation_path)["sample_ids"]
    lookup = {row["sample_id"]: row for row in manifest}
    assert len(validation_ids) == len(set(validation_ids)) == 64
    assert all(lookup[sid]["split"] == "validation" for sid in validation_ids)
    scene_counts = Counter(lookup[sid]["scene"] for sid in validation_ids)
    assert len(scene_counts) == 8 and set(scene_counts.values()) == {8}
    calibration_path = OLD33 / "config/calibration.json"
    calibration = read(calibration_path)
    assert calibration["training_only"] is True and calibration["uses_GT"] is False
    assert calibration["samples"] == 256 and float(calibration["mu"]) == MU

    cfg["seed"] = 2026
    cfg["paths"]["schedule"] = str(schedule_path)
    cfg["paths"]["output_root"] = str(ROOT / "unused_output")
    cfg["downstream"].update(steps=3000, minimum_steps=3000, effective_batch=2, microbatch=1)
    cfg["downstream"].pop("deadline_unix", None)
    cfg.pop("experiment36", None)
    cfg["experiment37"] = dict(
        authoritative_proposal="config/proposal.json", exploratory_only=True,
        arms=["A", "B"], steps=3000, roi_size=ROI, halo=HALO, alignment=ALIGN,
        seed=2026, state_source=dict(A="native outer crop", B="full then identical outer crop"),
        noise="per image fixed: set_step_seed(2026, state_source_train, 0, [sample_id]); "
              "native receives captured actual full noise",
        first_round_frozen=True, confidence="fixed train-only global mean",
        task_feature="frozen feedback.output_project of full-grid decoder_level1/TaF1",
        trainable="independent local RGB residual heads only",
        loss="MSE + .02 * .5 * (horizontal edge MAE + vertical edge MAE)",
        validation=64, test=0, full_resolution_evaluation=True)
    cfg_path = ROOT / "config/execution_config.json"
    write(cfg_path, cfg)

    # Preserve every immutable loader dependency recorded by 36. Its local runner
    # and protocol are historical provenance, not executable authority for 37.
    sources = {}
    for record in old_lock["sources"]:
        path = Path(record["path"])
        assert sha(path) == record["sha256"], str(path)
        sources[str(path.resolve())] = fingerprint(path, "inherited verified component dependency")
    for path, role in [
        (old_lock_path, "verified experiment36 dependency inventory"),
        (cfg_source, "component loader configuration source"),
        (schedule_source, "unaltered original 3000-step schedule"),
        (manifest_path, "dataset manifest and split"),
        (validation_path, "existing fixed 8 x 8 validation selection"),
        (calibration_path, "fixed train-only confidence mean"),
        (protocol, "experiment37 registered scientific protocol"),
        (cfg_path, "experiment37 loader configuration"),
        (schedule_path, "experiment37 unaltered schedule"),
        (cache_items_path, "fixed positions, shapes, halo and local coordinates"),
    ]:
        sources[str(Path(path).resolve())] = fingerprint(path, role)
    code_paths = {Path(record["path"]).resolve() for record in old_lock["code_sources"]
                  if not Path(record["path"]).resolve().is_relative_to(OLD36 / "scripts")}
    code_paths.update(path.resolve() for path in (ROOT / "scripts").glob("*.py"))
    code_sources = [fingerprint(path) for path in sorted(code_paths)]
    assert len(code_sources) > 4
    plan = dict(
        schema_version=1, experiment=37, exploratory_only=True, authorized=True,
        arms=["A", "B"], steps=3000, lr=1e-4, mu=MU, seed=2026,
        roi_size=ROI, halo=HALO, alignment=ALIGN, effective_batch=2, microbatch=1,
        gradient_clip_norm=1.0, amp=False, scheduler=None, model_eval_mode=True,
        trainable="independent local RGB residual heads only",
        head=dict(input_channels=47, hidden_channels=32,
                  architecture=["Conv1x1(47,32)", "GELU", "Conv3x3(32,32,padding=1)",
                                "GELU", "Conv3x3(32,3,padding=1)"],
                  last_layer_zero_initialized=True, receptive_field=5, radius=2,
                  amplitude=0.25, output="T1 + .25*tanh(head(inputs)); no clamp",
                  input_order=["p0:12", "official_rgb:3", "p1:12", "t1:3", "projT1:16", "constantC:1"],
                  frozen_projection="model.feedback.output_project Conv1x1(96,16)",
                  no_spatial_global_operations=True),
        optimizer=dict(kind="Adam", betas=[0.9,0.999], eps=1e-8, weight_decay=0),
        loss="MSE + .02 * .5 * (horizontal edge MAE + vertical edge MAE)",
        noise=dict(seed=2026, phase="state_source_train", step=0,
                   key="single sample_id", actual_noise_pairing=True,
                   shape=[1,16,256], fixed_per_image=True,
                   note="This intentionally replaces the old step-keyed training noise for both arms."),
        state_sources=dict(A="first round on same native aligned outer observation crop",
                           B="first round on complete original image, then same outer coordinates"),
        train_cfg_path=str(cfg_path), schedule3000_path=str(schedule_path),
        schedule_sha256=sha(schedule_path), cache_items_path=str(cache_items_path),
        cache_items_sha256=sha(cache_items_path),
        validation_ids=validation_ids, validation_scene_counts=dict(sorted(scene_counts.items())),
        reference_metrics_path=str(OLD33 / "metrics/validation872.json"),
        common_initial_path=str(OLD30 / "weights/common_initial21.pt"),
        calibration_path=str(calibration_path),
        evaluation=dict(validation=64, test=0, original_size=True, tiles=False,
                        common_full_T1=True, fixed_endpoint=3000, fixed_evaluation_seed=2026,
                        evaluation_phase="H_only_pair_eval"),
        max_wall_seconds=43200,
        budget_start="first worker registration; controller resource waiting excluded",
        source_receipts=list(sources.values()), code_sources=code_sources,
        positive_thresholds=dict(comparisons=["B-A", "B-T1"], scene_equal_psnr_min=0.05,
                                 scene_equal_ssim_min=0.0, positive_psnr_scenes_min=6,
                                 scenes_total=8, role="investment gate, not statistical significance"),
        stopping_rule="Exactly 2 x 3000 updates and fixed64; no automatic seed, LR, loss, "
                      "amplitude, scope, step or Oracle search. Worker wall limit 43200s.",
        confirmation_rule="Only after both gates pass, register confirmation on the same "
                          "fixed endpoints and complete 872 validation plus 188 test; no new training.",
        server_only=True)
    plan["scientific_id"] = canonical_id(plan)
    with proposal_path.open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2, allow_nan=False)
    verify(plan)
    print(json.dumps(dict(prepared=True, scientific_id=plan["scientific_id"],
                          proposal_sha256=sha(proposal_path), arms=plan["arms"],
                          steps=3000, cache_items=len(items), unique_images=len(shape_cache),
                          validation=64, source_files=len(sources), code_files=len(code_sources))), flush=True)


if __name__ == "__main__":
    main()