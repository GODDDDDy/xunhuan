"""Read-only matched comparisons from retained per-image evaluation records."""
import json
import hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ORIGINAL = ROOT.parent
names = ("phase1", "phase01")
results = {name: json.loads((ROOT / "runs" / name / "seed_2026/metrics/step_3000_validation.json").read_text()) for name in names}
rows = {name: {r["sample_id"]: r for r in result["per_image"]} for name, result in results.items()}
assert rows[names[0]].keys() == rows[names[1]].keys()
ids = sorted(rows[names[0]])
assert len(ids) == 872
scenes = sorted({rows[names[0]][i]["scene"] for i in ids})
summary = {"scope": "single seed2026; exploratory validation; fixed step3000", "frames": len(ids), "scenes": len(scenes), "arms": {}, "comparisons": {}}
checkpoint_audit = json.loads((ROOT / "diagnostics/final_checkpoint_audit.json").read_text())
for name, result in results.items():
    assert result["checkpoint_sha256"] == checkpoint_audit[name]["checkpoints"]["3000"]["checkpoint_sha256"]
    summary["arms"][name] = {k: result[k] for k in ("round1", "round2", "iteration_gain", "polar_update")}
    audits = [json.loads(p.read_text()) for p in (ROOT / "runs" / name / "seed_2026/diagnostics").glob("frozen_step_*.json")]
    assert audits and all(a["passed"] for a in audits)
    assert any(a["step"] == 3000 for a in audits)
    summary["arms"][name]["frozen_audits_passed"] = True
    summary["arms"][name]["audited_steps"] = sorted(a["step"] for a in audits)
    loss = json.loads((ROOT / "runs" / name / "seed_2026/metrics/step_3000_validation_loss_parts.json").read_text())
    assert loss["frames"] == 872
    assert loss["checkpoint_sha256"] == result["checkpoint_sha256"]
    assert {r["sample_id"] for r in loss["per_image"]} == set(ids)
    summary["arms"][name]["loss_parts"] = {k: loss[k] for k in ("round1", "round2", "iteration_delta", "rng_unchanged_by_loss_arithmetic")}

def comparison(label, first, second, rounds):
    out = {}
    for ra, rb in rounds:
        metrics = {}
        for metric in ("psnr", "ssim"):
            deltas = {i: first[i][ra][metric] - second[i][rb][metric] for i in ids}
            per_scene = {s: {"n": sum(first[i]["scene"] == s for i in ids), "mean_delta": float(np.mean([d for i, d in deltas.items() if first[i]["scene"] == s]))} for s in scenes}
            metrics[metric] = {"image_mean_delta": float(np.mean(list(deltas.values()))), "scene_macro_delta": float(np.mean([v["mean_delta"] for v in per_scene.values()])), "positive_images": sum(v > 0 for v in deltas.values()), "negative_images": sum(v < 0 for v in deltas.values()), "positive_scenes": sum(v["mean_delta"] > 0 for v in per_scene.values()), "per_scene": per_scene}
        out[ra + "_minus_" + rb] = metrics
    summary["comparisons"][label] = out

comparison("phase01_minus_phase1", rows["phase01"], rows["phase1"], [("round1", "round1"), ("round2", "round2")])
summary["difficulty_descriptive_posthoc"] = {}
for difficulty in ("easy", "hard"):
    subset = [i for i in ids if i.split("/")[1] == difficulty]
    assert subset
    summary["difficulty_descriptive_posthoc"][difficulty] = {"frames": len(subset), "T2_phase01_minus_phase1": {m: float(np.mean([rows["phase01"][i]["round2"][m] - rows["phase1"][i]["round2"][m] for i in subset])) for m in ("psnr", "ssim")}, "phase01_T2_minus_T1": {m: float(np.mean([rows["phase01"][i]["round2"][m] - rows["phase01"][i]["round1"][m] for i in subset])) for m in ("psnr", "ssim")}}
for name in names:
    comparison(name + "_T2_minus_T1", rows[name], rows[name], [("round2", "round1")])
old = json.loads((ORIGINAL / "runs/no_feedback/seed_2026/metrics/step_3000_validation.json").read_text())
old_rows = {r["sample_id"]: r for r in old["per_image"]}
assert old_rows.keys() == rows["phase1"].keys()
comparison("phase1_minus_historical_no_feedback", rows["phase1"], old_rows, [("round1", "round1"), ("round2", "round2")])
control_max = max(abs(rows["phase1"][i][rnd][m] - old_rows[i][rnd][m]) for i in ids for rnd in ("round1", "round2") for m in ("psnr", "ssim"))
summary["control_reproduction"] = {"frames": 872, "max_per_image_metric_abs_difference": control_max, "all_per_image_metrics_equal": control_max == 0}
summary["historical_metric_sha256"] = hashlib.sha256((ORIGINAL / "runs/no_feedback/seed_2026/metrics/step_3000_validation.json").read_bytes()).hexdigest()
g6path = ORIGINAL.parent / "k1_ablation_20260930/runs/G6_netg_only_bypass_r2p/seed_2026/metrics/step_3000_validation.json"
summary["g6_reference_path"] = str(g6path)
if g6path.exists():
    g6 = json.loads(g6path.read_text())
    g6rows = {r["sample_id"]: {"scene": r["scene"], "g6": {"psnr": r["psnr"], "ssim": r["ssim"]}} for r in g6["per_image"]}
    assert g6rows.keys() == rows["phase1"].keys()
    summary["g6_reference"] = {k: g6[k] for k in ("image_mean_psnr", "image_mean_ssim", "scene_macro_psnr", "scene_macro_ssim")}
    summary["g6_metric_sha256"] = hashlib.sha256(g6path.read_bytes()).hexdigest()
    for name in names:
        comparison(name + "_T2_minus_G6", rows[name], g6rows, [("round2", "g6")])
(ROOT / "analysis").mkdir(exist_ok=True)
(ROOT / "analysis/matched_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({"arms": summary["arms"], "comparison_image_deltas": {name: {r: {m: v["image_mean_delta"] for m, v in metrics.items()} for r, metrics in values.items()} for name, values in summary["comparisons"].items()}}, indent=2))
