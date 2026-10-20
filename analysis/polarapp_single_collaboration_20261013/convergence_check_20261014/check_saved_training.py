"""Post-hoc CPU audit of saved training records; no torch, inference or selection."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DOWNLOAD = PROJECT / "_dl/polarapp_single_collaboration_20261013"
ARTIFACTS = DOWNLOAD / "artifacts"
entries = {x["path"]: x for x in json.loads((DOWNLOAD / "archive_manifest.json").read_text(encoding="utf-8"))["files"]}


def verified(relative):
    path = ARTIFACTS / relative
    data = path.read_bytes()
    expected = entries[relative]
    if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
        raise RuntimeError(f"Saved input differs from verified final archive: {relative}")
    return path


training_path = verified("downstream/logs/meta_training.csv")
with training_path.open(newline="", encoding="utf-8") as stream:
    raw = list(csv.DictReader(stream))
assert [int(r["step"]) for r in raw] == list(range(1, 3001))
keys = ("task_loss", "joint_loss", "alignment_loss", "retention_loss", "adapter_gradient_norm",
        "meta_outer_raw_loss", "meta_hypergradient_raw_norm", "meta_hypergradient_scaled_norm")
assert all(math.isfinite(float(r[k])) for r in raw for k in keys)
meta_rows = [r for r in raw if int(r["step"]) % 10 == 0]
assert len(meta_rows) == 300
windows = []
for lo in range(0, 3000, 500):
    batch = raw[lo:lo+500]
    result = {"start_step": lo+1, "end_step": lo+500, "joint_observations": 500, "meta_observations": 50}
    for key in keys:
        values = [float(r[key]) for r in batch if not key.startswith("meta_") or int(r["step"]) % 10 == 0]
        result[key] = statistics.mean(values)
        result[key + "_median"] = statistics.median(values)
        result[key + "_sample_SD"] = statistics.stdev(values)
    windows.append(result)
probe_paths = [verified("downstream/metrics/meta_probe_initial.json")]
probe_paths += [verified(f"downstream/metrics/meta_probe_{step}.json") for step in range(500, 3001, 500)]
probes = []
baseline = None
for path in probe_paths:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["step"] == 0:
        baseline = [(r["sample_id"], r["source_prediction_sha256"]) for r in value["records"]]
    else:
        assert [(r["sample_id"], r["source_prediction_sha256"]) for r in value["records"]] == baseline
    assert value["training_only"] and not value["used_for_selection"]
    source = statistics.mean(r["source_RGB_MSE"] for r in value["records"])
    adapted = statistics.mean(r["adapted_RGB_MSE"] for r in value["records"])
    row = {"step": value["step"], "fixed_train_crops": len(value["records"]),
           "source_MSE": source, "adapted_MSE": adapted,
           "relative_MSE_change_from_source_percent": 100 * (adapted/source - 1),
           "mean_per_crop_PSNR_change_dB": statistics.mean(10*math.log10(r["source_RGB_MSE"]/r["adapted_RGB_MSE"]) for r in value["records"]),
           "worse_crops_than_source": sum(r["adapted_RGB_MSE"] > r["source_RGB_MSE"] for r in value["records"]),
           "individual_MSE": [r["adapted_RGB_MSE"] for r in value["records"]]}
    probes.append(row)


def percent(after, before):
    return 100 * (after/before - 1)


tail = {k: percent(windows[-1][k], windows[-2][k]) for k in keys}
fixed_tail = percent(probes[-1]["adapted_MSE"], probes[-2]["adapted_MSE"])
result = {"scope": "posthoc_read_only_saved_training_convergence_check",
          "joint_steps": 3000, "meta_updates": 300, "training_image_crops": 6000,
          "unique_training_images": int(raw[-1]["unique_joint_images"]),
          "mean_image_draws": 6000/int(raw[-1]["unique_joint_images"]),
          "all_saved_inputs_verified_against_final_archive": True,
          "source_probe_sample_and_float_output_hashes_constant": True,
          "window_definition": "nonoverlapping 500 joint steps, 50 meta updates per window",
          "last500_vs_previous500_percent": tail,
          "fixed_probe_3000_vs_2500_MSE_percent": fixed_tail,
          "fixed_probe_3000_vs_source_MSE_percent": probes[-1]["relative_MSE_change_from_source_percent"],
          "windows": windows, "fixed_training_probes": probes,
          "validation_learning_curve_available": False,
          "parameter_update_norm_history_available": False,
          "gradient_norm_is_not_parameter_update_norm": True,
          "new_GPU_work": False, "new_inference": False, "test_used_for_stopping_or_selection": False,
          "criterion_preregistered": False,
          "conclusion": "alignment_nearly_flat; task_stability_and_convergence_not_confirmed",
          "limits": ["Random mini-batch windows differ in images, crops and diffusion noise",
                     "Only four fixed training crops; no representative periodic validation trajectory",
                     "No actual historical parameter-update norm or all intermediate model checkpoints",
                     "Post-hoc diagnostic does not certify a stationary optimum or imply longer training helps"]}
(OUT / "convergence_diagnostics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
for name, rows in (("window_statistics.csv", windows), ("fixed_probe_statistics.csv", probes)):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
print(json.dumps({k: v for k, v in result.items() if k not in ("windows", "fixed_training_probes", "limits")}, ensure_ascii=False))
print(json.dumps(probes, ensure_ascii=False))
