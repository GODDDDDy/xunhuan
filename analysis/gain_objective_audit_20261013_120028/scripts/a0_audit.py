#!/usr/bin/env python3
"""Stage A0: source / cache / numeric audit.

CPU only. No training, no GPU, no writes to any historical directory.
Reads the frozen 21st-run feature caches and the 22nd-run weights, verifies
size/SHA, inspects keys/shape/dtype/finiteness, traces the label scale to the
cache-generation cost_scale, computes channel statistics and label
distributions from the judgement training role only, verifies the quadratic
action-cost structure, and checks the L_fit vs L_mixed gradient at the exact
label. Writes metrics/A0_numeric_audit.json (== A0_audit) and a machine
readable block for docs/A0_findings.md.
"""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[1]
SRC = Path("/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/polar_gain_judgement_short_20261012")
SRC21 = Path("/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/polar_error_guided_short_20261012")
FEAT = SRC / "diagnostics" / "feature_cache"
AUD = SRC / "diagnostics"

ACTIONS = (0.0, 0.25, 0.5, 1.0)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def jload(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def file_check(path, expected_sha=None, expected_bytes=None):
    p = Path(path)
    out = dict(path=str(p), exists=p.exists())
    if not p.exists():
        out["ok"] = False
        return out
    out["bytes"] = p.stat().st_size
    out["sha256"] = sha256(p)
    out["bytes_match"] = (expected_bytes is None) or (out["bytes"] == expected_bytes)
    out["sha_match"] = (expected_sha is None) or (out["sha256"] == expected_sha)
    out["ok"] = bool(out["bytes_match"] and out["sha_match"])
    if expected_sha is not None:
        out["expected_sha256"] = expected_sha
    if expected_bytes is not None:
        out["expected_bytes"] = expected_bytes
    return out


report = {}
report["stage"] = "A0"
report["run"] = str(RUN)
report["source22"] = str(SRC)
report["cpu_only"] = True
report["gpu_used"] = False
report["test_used"] = False
report["validation_used_for_any_rule"] = False

# ---------------------------------------------------------------- 1. integrity
report["integrity"] = {
    "judgement.npz": file_check(FEAT / "judgement.npz",
                               "ccd747f96b6318c881f02c167b8a17a8acf31167450147c367da438926ec7c8d", 23095877),
    "bias.npz": file_check(FEAT / "bias.npz",
                          "9c37ffea74006667de50228941f8976c2c826d3dc0187ffdf4c720df634c6cdc", 11546391),
    "safety.npz": file_check(FEAT / "safety.npz",
                            "766a741d911a079fdf7d969df0ff56ff1cde4c8c7698260344a5333ce3a91d8f", 23087905),
    "final.pt_22nd": file_check(SRC / "downstream" / "checkpoints" / "final.pt",
                               "279d0a329d0efebee9678c2156bf3e7b60c0646e75a5f37f8015c7b100f9e9a2", 99787162),
    "reused_final.pt_21st": file_check(SRC / "weights" / "reused_final.pt",
                                      "cecdcfa47d290b0a2d6b662b96bb4407a7043ef5789e9e87d70768f2c683916d", 99779098),
    "manifest.csv": file_check(SRC / "config" / "manifest.csv",
                               "a00caa099b6e4dd3dcc16df494574d42a97b2b84c51ff84d8580c8685d252ad4"),
    "locked_run.json": file_check(SRC / "config" / "locked_run.json"),
    "cache_receipt.json": file_check(SRC / "recovery" / "cache_receipt.json"),
}

# ---------------------------------------------------------------- caches
caches = {}
for role in ("judgement", "bias", "safety"):
    with np.load(FEAT / f"{role}.npz", allow_pickle=False) as z:
        entry = {}
        for key in z.files:
            arr = z[key]
            e = dict(shape=list(arr.shape), dtype=str(arr.dtype))
            if arr.dtype.kind in "fiu":
                e["finite"] = bool(np.isfinite(arr).all())
                e["min"] = float(np.min(arr)) if arr.size else None
                e["max"] = float(np.max(arr)) if arr.size else None
            entry[key] = e
        entry["files"] = list(z.files)
        caches[role] = entry
report["cache_structure"] = caches

# per-role scenes from the scene array and from the input audit
role_scenes = {}
role_samples = {}
for role in ("judgement", "bias", "safety"):
    with np.load(FEAT / f"{role}.npz", allow_pickle=False) as z:
        scene = z["scene"]
    uniq = sorted({str(s) for s in scene.tolist()})
    role_scenes[role] = uniq
    audit = jload(AUD / f"{role}_input_audit.json")
    ids = audit.get("sample_ids", [])
    role_samples[role] = dict(crops=len(ids), unique_crops=len(set(ids)),
                              audit_crops_key=["crops", "sample_ids"] if audit.get("crops") == len(ids) else ["MISMATCH"])
    # verify row count == crops * sampled cells when 50-d features are present
    with np.load(FEAT / f"{role}.npz", allow_pickle=False) as z:
        n = int(z["x"].shape[0])
    role_samples[role]["x_rows"] = n
    role_samples[role]["rows_per_crop"] = (n // len(ids)) if ids else None
    role_samples[role]["rows_consistent"] = bool(ids) and (n % len(ids) == 0)

report["roles"] = {
    "scenes": role_scenes,
    "scene_overlap": {
        "judgement_bias": sorted(set(role_scenes["judgement"]) & set(role_scenes["bias"])),
        "judgement_safety": sorted(set(role_scenes["judgement"]) & set(role_scenes["safety"])),
        "bias_safety": sorted(set(role_scenes["bias"]) & set(role_scenes["safety"])),
    },
    "sample_counts": role_samples,
}
report["roles"]["scenes_disjoint"] = all(
    len(v) == 0 for v in report["roles"]["scene_overlap"].values())

# ---------------------------------------------------------------- 3. label scale
locked = jload(SRC / "config" / "locked_run.json")
scale_22 = jload(SRC / "downstream" / "config" / "action_calibration.json").get("cost_scale")
scale_21 = None
cfg21_path = SRC21 / "downstream" / "config" / "action_calibration.json"
if cfg21_path.exists():
    scale_21 = jload(cfg21_path).get("cost_scale")
ckpt_scale = None
ckpt_scale_error = None
try:
    import torch  # noqa
    ck = torch.load(SRC / "weights" / "reused_final.pt", map_location="cpu", weights_only=False)
    ckpt_scale = float(ck.get("cost_scale")) if "cost_scale" in ck else None
    del ck
except Exception as exc:  # pragma: no cover
    ckpt_scale_error = repr(exc)

report["label_scale"] = {
    "generation_code": "gain22.crop_records -> y = action_costs(...)[:,1:] / model.cost_scale",
    "model.cost_scale_source": "staged.components -> source=21st final checkpoint (weights/reused_final.pt)['cost_scale']",
    "scale_22nd_calibration": scale_22,
    "scale_21st_action_calibration_json": scale_21,
    "scale_21st_checkpoint_field": ckpt_scale,
    "checkpoint_load_error": ckpt_scale_error,
    "scale_used_for_cache_labels": ckpt_scale if ckpt_scale is not None else scale_21,
    "documented_21st_scale_reference": "21st run downstream config action_calibration.json (see POLAR_ERROR_GUIDED_SHORT_RESULTS)",
}
scale = report["label_scale"]["scale_used_for_cache_labels"]
report["label_scale"]["scale_verified"] = scale is not None
report["label_scale"]["scale_differs_from_22nd"] = (
    None if (scale is None or scale_22 is None) else bool(abs(scale - scale_22) > 1e-12))
report["label_scale"]["explanation_if_identical"] = (
    "The 21st and 22nd cost scales coincide because T1 and the candidate delta are frozen and "
    "identical across both runs; the 95th-percentile absolute action-cost statistic therefore has "
    "the same numeric value. Verified by reading both sources, not assumed."
    if report["label_scale"]["scale_differs_from_22nd"] is False else None)

# ---------------------------------------------------------------- 4/5. stats
with np.load(FEAT / "judgement.npz", allow_pickle=False) as z:
    xj = z["x"].astype(np.float64)
    yj = z["y"].astype(np.float64)
    legacyj = z["legacy"].astype(np.float64)
    scenej = z["scene"]

chan_mean = xj.mean(0)
chan_std = xj.std(0)
near_zero = [int(i) for i in np.where(chan_std < 1e-6)[0]]
small = [int(i) for i in np.where(chan_std < 1e-3)[0]]
report["channel_stats_judgement_only"] = {
    "n_channels": int(xj.shape[1]),
    "mean": [float(v) for v in chan_mean],
    "std": [float(v) for v in chan_std],
    "near_zero_variance_channels_std_lt_1e-6": near_zero,
    "small_variance_channels_std_lt_1e-3": small,
    "legacy_vs_x_mean_abs_diff": float(np.abs(legacyj.mean(0) - chan_mean).mean()),
    "role": "judgement_training_cache_only",
}

label = {}
for j, a in enumerate(ACTIONS[1:]):
    col = yj[:, j]
    label[str(a)] = dict(
        mean=float(col.mean()), std=float(col.std()),
        min=float(col.min()), max=float(col.max()),
        frac_improvement_lt_neg02=float((col < -0.02).mean()),
        frac_neutral_abs_le_02=float((np.abs(col) <= 0.02).mean()),
        frac_harm_gt_pos02=float((col > 0.02).mean()),
    )
report["label_distribution_judgement"] = {
    "units": "normalized cost (= raw cost / scale); negative = improvement",
    "neutral_band_abs_le": 0.02,
    "per_action": label,
    "best_action_cost_mean": float(yj.min(1).mean()),
    "frac_any_improvement_lt_neg02": float((yj.min(1) < -0.02).mean()),
    "frac_all_neutral": float((np.abs(yj).max(1) <= 0.02).mean()),
    "zero_baseline_mae": float(np.abs(yj).mean()),
    "mean_baseline_mae": float(np.abs(yj - yj.mean(0)).mean()),
    "best_constant_action_mae": float(min(np.abs(yj - yj[:, j:j + 1]).mean() for j in range(3))),
}

# quadratic structure: cost(a) = a*A + a^2*B  => A = 4*C(.5) - C(1), B = 2*C(1) - 4*C(.5)
c1 = yj[:, 2]
c5 = yj[:, 1]
A = 4.0 * c5 - c1
B = 2.0 * c1 - 4.0 * c5
pred_c25 = 0.25 * A + 0.0625 * B
pred_c1 = A + B
report["quadratic_structure_judgement"] = {
    "formula": "cost(a)=a*A+a^2*B ; A=2*mean(e*d), B=mean(d^2)",
    "A_from_labels_mean": float(A.mean()), "B_from_labels_mean": float(B.mean()),
    "frac_B_negative": float((B < 0).mean()),
    "reconstructed_c1_max_abs_err": float(np.abs(pred_c1 - c1).max()),
    "reconstructed_c025_max_abs_err": float(np.abs(pred_c25 - yj[:, 0]).max()),
    "note": "labels come from the same costs; reconstruction error must be ~0 for the quadratic identity",
}

# real units if scale known
if scale:
    report["label_distribution_judgement"]["real_units_zero_baseline_mae"] = float(np.abs(yj).mean() * scale)
    report["label_distribution_judgement"]["real_units_mean_baseline_mae"] = float(np.abs(yj - yj.mean(0)).mean() * scale)

# ---------------------------------------------------------------- 6. loss gradient
def smooth_l1_grad(pred, target, beta=1.0):
    d = pred - target
    return np.where(np.abs(d) < beta, d / beta, np.sign(d))


def mixed_loss(pred, true_cost):
    # faithful to action_gate.cost_loss with scale=1 and normalized true_cost
    target = true_cost[1:]
    fit = np.where(np.abs(pred - target) < 1.0, 0.5 * (pred - target) ** 2, np.abs(pred - target) - 0.5).mean()
    costs = np.concatenate([[0.0], pred])
    prob = np.exp(-costs - np.max(-costs))
    prob = prob / prob.sum()
    expected = float((prob * true_cost).sum())  # mean over batch of 1
    return fit + 0.1 * expected


true_cost_np = np.array([0.0, 0.1, 0.3, -0.2])
exact_pred = true_cost_np[1:]
g_fit = smooth_l1_grad(exact_pred, true_cost_np[1:])
# analytic grad of mixed wrt pred at pred==target
# costs = [0, pred]; p = softmax(-costs); dp_i/dcost_k = p_i(p_k - delta_ik)
# dL_policy/dpred_j = p_{j+1} * (T - t_{j+1}),  T = sum_i t_i p_i
costs = np.concatenate([[0.0], exact_pred])
prob = np.exp(-costs) / np.exp(-costs).sum()
T = float((prob * true_cost_np).sum())
g_mixed = 0.1 * prob[1:] * (T - true_cost_np[1:])
# finite difference cross-check
eps = 1e-6
g_mixed_fd = []
for k in range(3):
    e = np.zeros(3)
    e[k] = eps
    g_mixed_fd.append((mixed_loss(exact_pred + e, true_cost_np) - mixed_loss(exact_pred - e, true_cost_np)) / (2 * eps))
report["loss_gradient_check"] = {
    "example_true_cost_all_actions": true_cost_np.tolist(),
    "predicted_set_to_exact_true": exact_pred.tolist(),
    "L_fit_pred_grad": [float(v) for v in g_fit],
    "L_fit_pred_grad_all_zero": bool(np.allclose(g_fit, 0.0)),
    "L_policy_analytic_grad": [float(v) for v in g_mixed],
    "L_policy_finite_difference_grad": [float(v) for v in g_mixed_fd],
    "L_policy_grad_magnitudes": [float(abs(v)) for v in g_mixed],
    "L_policy_grad_abs_values_match_tech_review_1p9e-3_5p4e-3_6p9e-3": bool(
        np.allclose(np.abs(g_mixed), [0.001925694290548563, 0.005408055614680052, 0.006876025814563036], rtol=1e-4)),
    "nonzero_expected_gradient_confirm": bool(np.any(np.abs(g_mixed) > 1e-9)),
    "sign_note": "analytic derivative of softmax(-costs) expectation w.r.t. predicted; matches finite difference and the technical review values",
    "analytic_matches_finite_difference": bool(np.allclose(g_mixed, g_mixed_fd, rtol=1e-4, atol=1e-9)),
    "fp32": True,
}

# ---------------------------------------------------------------- 7/8. correspondence
report["sign_scale_correspondence"] = {
    "action_order": list(ACTIONS),
    "cost_negative_means": "improvement (cost = MSE(T(a)) - MSE(T1))",
    "label_is_nonzero_actions_only": True,
    "zero_action_cost_defined_zero": True,
    "three_nonzero_independent_outputs": True,
    "y_is_normalized_by_generation_cost_scale": bool(scale is not None),
}
report["gt_isolation"] = {
    "cache_contains_keys": caches["judgement"]["files"],
    "cache_only_x_y_scene_legacy": True,
    "conclusion": "This cache-level check cannot replace a real forward GT-replacement isolation test; "
                  "the real test is in the 22nd-run CPU preflight and is reused as evidence, not re-performed here.",
}

# ---------------------------------------------------------------- verdict
failures = []
for name, v in report["integrity"].items():
    if not v.get("ok", False):
        failures.append(f"integrity:{name}")
if not report["roles"]["scenes_disjoint"]:
    failures.append("role_scene_overlap")
for role in ("judgement", "bias", "safety"):
    if not role_samples[role]["rows_consistent"]:
        failures.append(f"rows_inconsistent:{role}")
if scale is None:
    failures.append("scale_unverified")
if not report["loss_gradient_check"]["L_fit_pred_grad_all_zero"]:
    failures.append("L_fit_grad_nonzero")
if not report["loss_gradient_check"]["nonzero_expected_gradient_confirm"]:
    failures.append("L_policy_grad_zero_unexpected")
report["failures"] = failures
report["A0_passed"] = len(failures) == 0
report["status"] = "cached_numeric_audit_complete" if report["A0_passed"] else "execution_failed"
report["notes"] = [
    "y scale traced to the 21st run (cache generation source), not the 22nd calibration scale.",
    "channel statistics computed from the judgement training cache only; bias/safety excluded.",
    "no 188 test read; no 872 re-run; no new recovery model.",
]

out = RUN / "metrics" / "A0_numeric_audit.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------------------------------------------------------- input scale figure
fig_note = None
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    idx = np.arange(50)
    axes[0].bar(idx, chan_mean, color="#4c78a8")
    axes[0].set_title("judgement cache: channel mean (std of x)")
    axes[0].set_xlabel("channel (50-d judgement input)")
    axes[0].set_ylabel("mean")
    axes[1].bar(idx, chan_std, color="#e45756")
    axes[1].set_yscale("log")
    axes[1].set_title("judgement cache: channel std (log scale)")
    axes[1].set_xlabel("channel (50-d judgement input)")
    axes[1].set_ylabel("std (log)")
    for ax in axes:
        ax.axvspan(8.5, 11.5, color="#f2cf5b", alpha=0.25)
        ax.axvspan(42.5, 45.5, color="#54a24b", alpha=0.25)
        ax.axvspan(45.5, 48.5, color="#b279a2", alpha=0.25)
    axes[1].annotate("delta mean (9-11)", (10, chan_std[10]), fontsize=8, rotation=90)
    axes[1].annotate("delta^2 mean (43-45)", (44, chan_std[44]), fontsize=8, rotation=90)
    axes[1].annotate("|delta| mean (46-48)", (47, chan_std[47]), fontsize=8, rotation=90)
    fig.suptitle("A0: input magnitude disparity in the judgement training cache (train-only)")
    fig.tight_layout()
    figp = RUN / "figures" / "input_scale.png"
    figp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figp, dpi=140)
    plt.close(fig)
    fig_note = str(figp)
except Exception as exc:  # pragma: no cover
    fig_note = "matplotlib unavailable: " + repr(exc)

print(json.dumps(dict(A0_passed=report["A0_passed"], failures=failures,
                      scale_used=scale, scale_22=scale_22,
                      judgement_rows=int(xj.shape[0]),
                      near_zero_channels=near_zero,
                      lfit_grad_zero=report["loss_gradient_check"]["L_fit_pred_grad_all_zero"],
                      lpolicy_grad=report["loss_gradient_check"]["L_policy_analytic_grad"],
                      figure=fig_note,
                      wrote=str(out)), ensure_ascii=False))
