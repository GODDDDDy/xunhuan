"""Fixed-endpoint source/meta, full-image evaluation of a single-round experiment.

One frozen backbone is shared for source and the sole trained meta adapter state.
FANet is a training-only loss network and is never run in this evaluator. The
source bypasses both adapter sets. GT is read only for final metrics and figures.
No tiling, selection, calibration, optimizer update or extra training is allowed.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


EXPECTED_COUNTS = {"validation": 872, "test": 188}
NAMES = ("source", "meta")
PAIRS = (("meta", "source"),)
FIELDS = ["sample_id", "scene", "difficulty", "height", "width", "device",
          "diffusion_seed", "diffusion_noise_sha256"]
for _name in NAMES:
    FIELDS += [f"{_name}_{metric}" for metric in
               ("psnr", "ssim", "clipped_psnr", "clipped_ssim", "raw_mse",
                "raw_prediction_min", "raw_prediction_max", "outside_unit_fraction",
                "prediction_sha256", "polar_sha256")]
for _arm in ("meta",):
    FIELDS += [f"{_arm}_polar_source_mse", f"{_arm}_polar_source_mae",
               f"{_arm}_rgb_source_mse", f"{_arm}_rgb_source_mae"]
for _left, _right in PAIRS:
    FIELDS += [f"{_left}_minus_{_right}_psnr", f"{_left}_minus_{_right}_ssim"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value):
    array = value.detach().float().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def emit(stage, **values):
    print(json.dumps({"stage": stage, **values}, ensure_ascii=False,
                     allow_nan=False), flush=True)


def example_selection(manifest):
    """One middle frame per easy/hard stratum in eight fixed validation scenes."""
    strata = defaultdict(list)
    for row in manifest:
        if row.split == "validation":
            strata[(row.scene, row.difficulty)].append(row)
    scenes = sorted({scene for scene, _ in strata})
    if len(scenes) != 8:
        raise ValueError(f"Expected eight fixed validation scenes, found {scenes}")
    result = []
    for scene in scenes:
        for difficulty in ("easy", "hard"):
            values = sorted(strata[(scene, difficulty)],
                            key=lambda row: (row.frame, row.sample_id))
            if not values:
                raise ValueError(f"Missing figure stratum {scene}/{difficulty}")
            result.append(values[len(values) // 2].sample_id)
    return result


def checkpoint_adapters(path, arm, config, torch):
    """The backbone always comes from the checked source, never an arm fallback."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol_id") != config["protocol_id"]:
        raise ValueError(f"{arm} checkpoint protocol differs from locked run")
    source_hash = checkpoint.get("source_sha256", checkpoint.get("source_sha"))
    if source_hash != config["source_checkpoint_sha256"]:
        raise ValueError(f"{arm} checkpoint source differs from locked run")
    final_step = int(checkpoint.get("joint_step", checkpoint.get("step", -1)))
    if final_step != int(config["joint_steps"]):
        raise ValueError(f"{arm} checkpoint is not the fixed final step: {final_step}")
    if "step" in checkpoint and "joint_step" in checkpoint:
        if int(checkpoint["step"]) != final_step:
            raise ValueError(f"{arm} step fields disagree")
    if checkpoint.get("arm", arm) != arm:
        raise ValueError(f"Wrong arm stored in {path}")
    states = {}
    for key in ("upstream_adapters", "downstream_adapters"):
        values = checkpoint[key]
        if not isinstance(values, dict) or not values:
            raise ValueError(f"{arm}: empty {key}")
        if any(not torch.is_tensor(value) or not torch.isfinite(value).all()
               for value in values.values()):
            raise ValueError(f"{arm}: nonfinite/non-tensor adapter state")
        states[key] = {name: value.detach().clone() for name, value in values.items()}
    receipt = {"arm": arm, "path": str(path), "sha256": sha256_file(path),
               "final_step": final_step, "source_sha256": source_hash,
               "protocol_id": checkpoint["protocol_id"],
               "optimizer_loaded": False, "rng_loaded": False,
               "FANet_used_for_inference": False}
    del checkpoint
    gc.collect()
    return states, receipt


def evaluation_context(root, config, runtime, checkpoints, examples, metric_source):
    official_root = Path(runtime["official_checkpoint_dir"])
    official_hashes = {name: sha256_file(official_root / name) for name in
                       ("net_g_latest.pth", "net_d_latest.pth", "net_le_dm_latest.pth")}
    return {
        "schema_version": 1, "protocol_id": config["protocol_id"],
        "locked_run_sha256": sha256_file(root / "config/locked_run.json"),
        "prepare_config_sha256": sha256_file(root / "config/prepare.json"),
        "source_checkpoint_sha256": config["source_checkpoint_sha256"],
        "source_checkpoint": runtime["source_checkpoint"],
        "official_ancestry_weight_sha256": official_hashes,
        "manifest_sha256": sha256_file(runtime["manifest"]),
        "checkpoint_receipts": checkpoints, "seed": int(config["seed"]),
        "expected_counts": EXPECTED_COUNTS, "arms": list(NAMES),
        "full_native_size_forward": True, "tiling": False,
        "diffusion_noise": "CPU generator, sample-id seed, one shared tensor for source and meta forwards",
        "prediction_metrics": "raw float prediction, original rgb_final_confirmation.evaluation definitions",
        "clipped_metrics": "auxiliary only; does not select weights or change primary outcome",
        "metric_source_sha256": sha256_file(metric_source),
        "evaluation_code_sha256": sha256_file(Path(__file__)),
        "example_sample_ids": examples,
        "example_selection": "middle frame per scene/difficulty; frozen before reading GT",
        "examples_are_8bit_display_only": True,
        "ground_truth_usage": "metrics and figures only; never a network input",
        "training_or_weight_selection": False,
    }


def load_existing(path, expected_ids):
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Existing CSV columns differ: {path}")
        result = list(reader)
    seen = set()
    for row in result:
        if any(row.get(field) is None for field in FIELDS) or None in row:
            raise ValueError(f"Incomplete row in {path}; preserve file before repairing")
        sample_id = row["sample_id"]
        if sample_id not in expected_ids or sample_id in seen:
            raise ValueError(f"Unknown/duplicate sample ID {sample_id} in {path}")
        seen.add(sample_id)
        for field in FIELDS:
            if field.endswith(("psnr", "ssim", "mse", "mae", "min", "max", "fraction")):
                value = float(row[field])
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite existing metric {sample_id}/{field}")
                row[field] = value
        for field in ("height", "width", "diffusion_seed"):
            row[field] = int(row[field])
    return result


def append_row(path, row):
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def save_examples(root, sample, polar_source, predictions, polars, record, torch):
    """Fixed display scale and float summaries; no per-case success selection."""
    import numpy as np
    from PIL import Image

    destination = root / "downstream/examples" / record["scene"] / record["difficulty"]
    destination.mkdir(parents=True, exist_ok=True)

    def rgb_png(value, name):
        array = value[0].detach().clamp(0, 1).mul(255).round().byte()
        array = array.cpu().permute(1, 2, 0).numpy()
        Image.fromarray(array).save(destination / f"{name}.png")

    def heat_png(value, name, maximum):
        normalized = (value[0, 0].detach().float().cpu().numpy() / maximum).clip(0, 1)
        # Fixed black-to-yellow scale; values above the legend maximum saturate.
        colors = np.stack((normalized, np.sqrt(normalized), np.zeros_like(normalized)), axis=-1)
        Image.fromarray((colors * 255).round().astype("uint8")).save(destination / f"{name}.png")

    rgb_png(sample["official_rgb"], "input")
    rgb_png(sample["target"], "gt")
    for name in NAMES:
        rgb_png(predictions[name], name)
        heat_png((predictions[name] - sample["target"]).abs().mean(1, keepdim=True),
                 f"{name}_absolute_error", .1)
    for name in ("meta",):
        heat_png((predictions[name] - predictions["source"]).abs().mean(1, keepdim=True),
                 f"{name}_source_rgb_difference", .05)
        polar_mean = torch.stack(polars[name].chunk(4, dim=1)).mean(0)
        rgb_png(polar_mean, f"{name}_polar_mean")
    rgb_png(torch.stack(polar_source.chunk(4, dim=1)).mean(0), "source_polar_mean")
    atomic_json(destination / "float_summary.json", {
        "sample": record, "quantized_PNGs_are_display_only": True,
        "absolute_error_heatmap_maximum": .1, "RGB_difference_heatmap_maximum": .05,
        "metrics_calculated_from_original_float_outputs": True,
        "full_native_size": [record["height"], record["width"]],
    })


def scene_table(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["scene"]].append(row)
    result = []
    for scene, values in sorted(groups.items()):
        row = {"scene": scene, "images": len(values)}
        for name in NAMES:
            for metric in ("psnr", "ssim", "clipped_psnr", "clipped_ssim"):
                key = f"{name}_{metric}"
                row[key] = sum(float(value[key]) for value in values) / len(values)
        for left, right in PAIRS:
            for metric in ("psnr", "ssim"):
                row[f"{left}_minus_{right}_{metric}"] = row[f"{left}_{metric}"] - row[f"{right}_{metric}"]
        for arm in ("meta",):
            for metric in ("mse", "mae"):
                key = f"{arm}_polar_source_{metric}"
                row[key] = sum(float(value[key]) for value in values) / len(values)
        result.append(row)
    return result


def write_scene_csv(path, scenes):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scenes[0]))
        writer.writeheader()
        writer.writerows(scenes)


def split_summary(rows, scenes):
    result = {"images": len(rows), "scene_count": len(scenes),
              "evaluation_devices": sorted({row["device"] for row in rows}),
              "output_metrics": {}, "paired_diffs": {},
              "polar_drift_from_source": {}, "scenes": scenes}
    for name in NAMES:
        result["output_metrics"][name] = {
            f"{level}_{metric}": sum(float(row[f"{name}_{metric}"]) for row in values) / len(values)
            for level, values in (("image_mean", rows), ("scene_macro", scenes))
            for metric in ("psnr", "ssim", "clipped_psnr", "clipped_ssim")
        }
    for left, right in PAIRS:
        pair = f"{left}_minus_{right}"
        key_p, key_s = f"{pair}_psnr", f"{pair}_ssim"
        result["paired_diffs"][pair] = {
            "image_mean_psnr": sum(float(row[key_p]) for row in rows) / len(rows),
            "image_mean_ssim": sum(float(row[key_s]) for row in rows) / len(rows),
            "scene_macro_psnr": sum(float(row[key_p]) for row in scenes) / len(scenes),
            "scene_macro_ssim": sum(float(row[key_s]) for row in scenes) / len(scenes),
            "PSNR_improved_images": sum(float(row[key_p]) > 0 for row in rows),
            "PSNR_worse_images": sum(float(row[key_p]) < 0 for row in rows),
            "SSIM_drop_images": sum(float(row[key_s]) < 0 for row in rows),
            "both_improved_images": sum(float(row[key_p]) > 0 and float(row[key_s]) >= 0 for row in rows),
            "negative_PSNR_scenes": [row["scene"] for row in scenes if float(row[key_p]) < 0],
            "negative_SSIM_scenes": [row["scene"] for row in scenes if float(row[key_s]) < 0],
            "worst_20_images": [{key: row[key] for key in ("sample_id", "scene", key_p, key_s)}
                                for row in sorted(rows, key=lambda row: float(row[key_p]))[:20]],
        }
    for arm in ("meta",):
        result["polar_drift_from_source"][arm] = {
            metric: sum(float(row[f"{arm}_polar_source_{metric}"]) for row in rows) / len(rows)
            for metric in ("mse", "mae")}
    result["interpretation_limits"] = [
        "User authorized meta-only training; source comparison cannot isolate collaboration from ordinary continued training",
        "One seed and short adapter training do not establish convergence or universal benefit",
        "Raw PolarFree prediction metrics are primary; clipped PNGs and clipped metrics are auxiliary",
        "Frozen-source polar retention is a proxy, not clean polar ground truth or physical correctness",
        "Official test has historical exploratory use; it is not a fresh independent confirmation",
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0", "cpu"))
    args = parser.parse_args()
    root = (args.root or Path(os.environ.get("_POLARAPP_RUN_ROOT", Path(__file__).resolve().parents[1]))).resolve()
    # CPU completion after deadline must never query or initialize CUDA.
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts"))
    from runtime import guard
    import torch
    from modules.single_system import build_system

    device = torch.device(args.device)
    guard(stage="evaluation", root=root, cpu=device.type == "cpu")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(2026)
    config = read_json(root / "config/locked_run.json")
    if config["arms"] != ["meta"]:
        raise ValueError("Only the user-authorized meta training arm may be evaluated")
    states, checkpoints = {}, {}
    for arm in ("meta",):
        states[arm], checkpoints[arm] = checkpoint_adapters(
            root / f"downstream/checkpoints/{arm}/final.pt", arm, config, torch)

    up, down, fa, source_receipt, runtime = build_system(root)
    # FANet never goes onto GPU or into inference; no alignment-state transplant.
    del fa
    gc.collect()
    if source_receipt["source_sha256"] != config["source_checkpoint_sha256"]:
        raise ValueError("Runtime source receipt differs from locked run")
    if str(runtime["manifest"]) != str(config["manifest"]):
        raise ValueError("Preparation and formal manifests differ")
    from rgb_final_confirmation.data import load_manifest, PairedBranchDataset
    from rgb_final_confirmation.evaluation import psnr, ssim, stable_sample_seed
    from bridge import anchored_task_inputs
    manifest = load_manifest(runtime["manifest"])
    selected = {split: sorted([row for row in manifest if row.split == split],
                              key=lambda row: row.sample_id) for split in EXPECTED_COUNTS}
    all_ids = [row.sample_id for rows in selected.values() for row in rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate validation/test sample IDs")
    for split, rows in selected.items():
        if len(rows) != EXPECTED_COUNTS[split]:
            raise ValueError(f"Incomplete {split} manifest: {len(rows)}")
    examples = example_selection(manifest)
    metrics_dir = root / "downstream/metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    context = evaluation_context(root, config, runtime, checkpoints, examples,
                                 root / "source_snapshot/source/rgb_final_confirmation/evaluation.py")
    context_path = metrics_dir / "evaluation_context.json"
    existing_csvs = [metrics_dir / f"per_image_{split}.csv" for split in EXPECTED_COUNTS]
    if context_path.is_file():
        if read_json(context_path) != context:
            raise ValueError("Evaluation context changed; preserve existing evidence instead of mixing results")
    elif any(path.exists() for path in existing_csvs):
        raise ValueError("Per-image CSV exists without its immutable evaluation context")
    else:
        atomic_json(context_path, context)

    up.requires_grad_(False).eval().to(device)
    down.requires_grad_(False).eval().to(device)
    down.task.model.device = device
    # Validate both strict adapter schemas before any final data forward.
    for arm in ("meta",):
        up.adapters.load_state_dict(states[arm]["upstream_adapters"], strict=True)
        down.adapters.load_state_dict(states[arm]["downstream_adapters"], strict=True)
    started = time.time()
    all_results = {}
    for split, manifest_rows in selected.items():
        path = metrics_dir / f"per_image_{split}.csv"
        results = load_existing(path, {row.sample_id for row in manifest_rows})
        done = {row["sample_id"] for row in results}
        dataset = PairedBranchDataset(manifest_rows, group="A1", crop_size=None)
        emit("evaluation_split_started", split=split, completed=len(done), expected=len(dataset),
             device=str(device), full_native_size=True, tiling=False)
        for index, manifest_row in enumerate(manifest_rows):
            if manifest_row.sample_id in done:
                # An example is written before its CSV row, so a completed row
                # with missing example evidence indicates corruption, not a
                # reason to silently perform extra forwards or select a case.
                if manifest_row.sample_id in examples:
                    example_path = root / "downstream/examples" / manifest_row.scene / manifest_row.difficulty
                    if not (example_path / "float_summary.json").is_file():
                        raise ValueError(f"Missing completed example evidence: {manifest_row.sample_id}")
                continue
            guard(stage="evaluation", root=root, cpu=device.type == "cpu")
            sample = dataset[index]
            batch = {key: value.unsqueeze(0).to(device) for key, value in sample.items()
                     if torch.is_tensor(value) and key in ("polar", "official_rgb", "target")}
            height, width = batch["official_rgb"].shape[-2:]
            if any(value.shape[-2:] != (height, width) for value in batch.values()):
                raise ValueError(f"Native-size tensor mismatch: {manifest_row.sample_id}")
            sample_seed = stable_sample_seed(manifest_row.sample_id, base_seed=int(config["seed"]))
            record = {"sample_id": manifest_row.sample_id, "scene": manifest_row.scene,
                      "difficulty": manifest_row.difficulty, "height": height, "width": width,
                      "device": str(device), "diffusion_seed": sample_seed}
            predictions, polars = {}, {}
            with torch.no_grad():
                # One source enhancer forward, then form one device-independent
                # noise tensor of the actual frozen prior's shape.
                p_source, source_features = up(batch["polar"], batch["official_rgb"], use_adapters=False)
                del source_features
                inputs = anchored_task_inputs(p_source.float(), batch["polar"].float(),
                                               batch["official_rgb"].float())
                prior_condition = down.task.model.net_le_dm(inputs)
                noise = torch.randn(tuple(prior_condition.shape), dtype=prior_condition.dtype,
                                    generator=torch.Generator(device="cpu").manual_seed(sample_seed),
                                    device="cpu").to(device)
                del inputs, prior_condition
                record["diffusion_noise_sha256"] = tensor_hash(noise)
                for name in NAMES:
                    guard(stage="evaluation", root=root, cpu=device.type == "cpu")
                    if name == "source":
                        corrected = p_source
                    else:
                        up.adapters.load_state_dict(states[name]["upstream_adapters"], strict=True)
                        down.adapters.load_state_dict(states[name]["downstream_adapters"], strict=True)
                        corrected, features = up(batch["polar"], batch["official_rgb"])
                        del features
                    prediction, task_features = down(corrected, batch["official_rgb"], batch["polar"],
                                                       noise=noise, use_adapters=name != "source")
                    del task_features
                    guard(stage="evaluation", root=root, cpu=device.type == "cpu")
                    if prediction.shape != batch["target"].shape or corrected.shape != batch["polar"].shape:
                        raise ValueError(f"Full-forward output shape changed: {manifest_row.sample_id}/{name}")
                    if not torch.isfinite(prediction).all() or not torch.isfinite(corrected).all():
                        raise ValueError(f"Nonfinite full-forward output: {manifest_row.sample_id}/{name}")
                    clipped = prediction.clamp(0, 1)
                    record.update({
                        f"{name}_psnr": psnr(prediction, batch["target"]),
                        f"{name}_ssim": ssim(prediction, batch["target"]),
                        f"{name}_clipped_psnr": psnr(clipped, batch["target"]),
                        f"{name}_clipped_ssim": ssim(clipped, batch["target"]),
                        f"{name}_raw_mse": float((prediction - batch["target"]).square().mean()),
                        f"{name}_raw_prediction_min": float(prediction.min()),
                        f"{name}_raw_prediction_max": float(prediction.max()),
                        f"{name}_outside_unit_fraction": float(((prediction < 0) | (prediction > 1)).float().mean()),
                        f"{name}_prediction_sha256": tensor_hash(prediction),
                        f"{name}_polar_sha256": tensor_hash(corrected),
                    })
                    if name != "source":
                        record[f"{name}_polar_source_mse"] = float((corrected - p_source).square().mean())
                        record[f"{name}_polar_source_mae"] = float((corrected - p_source).abs().mean())
                    predictions[name] = prediction.detach().cpu()
                    # Only fixed figures need all polar outputs; otherwise keep
                    # no extra 12-channel image copies in CPU memory.
                    if manifest_row.sample_id in examples:
                        polars[name] = corrected.detach().cpu()
                    del corrected, prediction, clipped
                for name in ("meta",):
                    difference = predictions[name] - predictions["source"]
                    record[f"{name}_rgb_source_mse"] = float(difference.square().mean())
                    record[f"{name}_rgb_source_mae"] = float(difference.abs().mean())
                for left, right in PAIRS:
                    for metric in ("psnr", "ssim"):
                        record[f"{left}_minus_{right}_{metric}"] = record[f"{left}_{metric}"] - record[f"{right}_{metric}"]
                if any(not math.isfinite(value) for value in record.values() if isinstance(value, float)):
                    raise ValueError(f"Nonfinite final metric: {manifest_row.sample_id}")
                if manifest_row.sample_id in examples:
                    cpu_batch = {key: value.cpu() for key, value in batch.items()}
                    save_examples(root, cpu_batch, polars["source"], predictions, polars, record, torch)
            append_row(path, record)
            results.append(record)
            done.add(manifest_row.sample_id)
            emit("evaluation_image_complete", split=split, completed=len(done), expected=len(dataset),
                 sample_id=manifest_row.sample_id,
                 meta_minus_source_psnr=record["meta_minus_source_psnr"],
                 device=str(device))
            del p_source, predictions, polars, noise, batch
            if "cpu_batch" in locals():
                del cpu_batch
        if len(done) != EXPECTED_COUNTS[split]:
            raise ValueError(f"Incomplete {split} final evaluation")
        results.sort(key=lambda row: row["sample_id"])
        scenes = scene_table(results)
        write_scene_csv(metrics_dir / f"per_scene_{split}.csv", scenes)
        all_results[split] = split_summary(results, scenes)
        atomic_json(metrics_dir / f"summary_{split}.json", all_results[split])
        emit("evaluation_split_complete", split=split, images=len(done))

    summary = {"protocol_id": config["protocol_id"],
               "evaluated_counts": {split: value["images"] for split, value in all_results.items()},
               "fixed_final_checkpoints": checkpoints,
               "source_checkpoint_sha256": config["source_checkpoint_sha256"],
               "source_receipt": source_receipt, "full_native_size_forward": True,
               "tiling": False, "FANet_used_for_inference": False,
               "metric_primary": context["prediction_metrics"],
               "evaluation_context_sha256": sha256_file(context_path),
               "fixed_validation_examples": len(examples),
               "validation": all_results["validation"], "test": all_results["test"],
               "elapsed_this_execution_seconds": time.time() - started,
               "completed": True, "method_effectiveness_requires_paired_results": True,
               "matched_finetuning_control_available": False,
               "user_cancelled_control_not_evaluated": True}
    atomic_json(metrics_dir / "summary.json", summary)
    emit("evaluation_complete", evaluated_counts=summary["evaluated_counts"],
         fixed_examples=len(examples), full_native_size=True, tiling=False)


if __name__ == "__main__":
    main()
