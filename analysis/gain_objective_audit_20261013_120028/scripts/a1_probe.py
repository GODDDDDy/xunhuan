#!/usr/bin/env python3
"""Stage A1: lightweight learnability probe on the existing feature cache (CPU).

Purpose (handoff doc sec.7): cheap diagnosis of whether the existing scattered
representation carries a fittable signal, and how the formal mixed objective
differs from the pure fit objective.

Fixed defaults (locked before running, see config/diagnostic_lock.json):
  - input: the 50-channel x of the judgement training cache
  - standardisation statistics: judgement training cache only (all rows)
  - tiny-fit: 2048 judgement training regions, scene-balanced, seed 2026
  - MLP: 50 -> 32 -> 32 -> 3, GELU; one shared initial parameter set copied to
    both the mixed and the pure diagnostic replicas
  - Adam lr 1e-3, batch 256, 1000 fixed steps; no inherited optimiser
  - identical sample / batch order for both replicas; only the loss differs
  - report at steps 0,100,...,1000; compare at the 1000-step endpoint

The formal loss is reproduced exactly with the real cost_loss on CPU torch:
  L_fit   = SmoothL1(pred, y_normalised)
  L_policy= mean( sum( softmax(-[0,pred]) * [0, y_normalised] ) )
  L_mixed = L_fit + 0.1 * L_policy
  L_pure  = L_fit
Because the cache y is already normalised by the verified cost_scale, calling
cost_loss(pred, y*scale, scale) reduces to the above. That identity is asserted
against a direct numpy implementation below.

CPU only. No GPU, no training of the real backbone, no 188 test, no 872 rerun.
Writes metrics/A1_probe_results.json and checkpoints/diagnostic_heads/*.
"""
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

RUN = Path(__file__).resolve().parents[1]
SRC = Path("/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/polar_gain_judgement_short_20261012")
FEAT = SRC / "diagnostics" / "feature_cache"
ACTIONS = (0.0, 0.25, 0.5, 1.0)

# ---- fixed diagnostic defaults (lock) ----
SAMPLE_N = 2048
SEED = 2026
HIDDEN = 32
LR = 1e-3
BATCH = 256
STEPS = 1000
NEUTRAL_BAND = 0.02
COVERAGE_GRID = (0.01, 0.05, 0.10, 0.20)
SIGNAL_COVERAGE = 0.10

SCALE_22 = 0.000934407114982605  # verified in A0 == cache-generation scale


# ------------------------------------------------------------------ helpers
def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def state_dict_sha(module):
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rankdata(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return None
    value = float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
    return value if math.isfinite(value) else None


class MLP(nn.Module):
    def __init__(self, in_dim=50, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------- load & standardise
t0 = time.perf_counter()
with np.load(FEAT / "judgement.npz", allow_pickle=False) as z:
    xj = z["x"].astype(np.float32)
    yj = z["y"].astype(np.float32)
    scenej = np.array([str(s) for s in z["scene"].tolist()])
with np.load(FEAT / "bias.npz", allow_pickle=False) as z:
    xb, yb, sceneb = z["x"].astype(np.float32), z["y"].astype(np.float32), np.array([str(s) for s in z["scene"].tolist()])
with np.load(FEAT / "safety.npz", allow_pickle=False) as z:
    xs, ys, scenes_ = z["x"].astype(np.float32), z["y"].astype(np.float32), np.array([str(s) for s in z["scene"].tolist()])

chan_mean = xj.mean(0)
chan_std = xj.std(0)
chan_std_safe = np.where(chan_std < 1e-8, 1.0, chan_std)


def standardise(x):
    return (x - chan_mean) / chan_std_safe


xj_s, xb_s, xs_s = standardise(xj), standardise(xb), standardise(xs)

# ---------------------------------------------- scene-balanced 2048 sampling
rng = np.random.default_rng(SEED)
scenes = sorted(set(scenej.tolist()))
n_sc = len(scenes)
base = SAMPLE_N // n_sc
rem = SAMPLE_N % n_sc
chosen = []
per_scene = {}
for i, s in enumerate(scenes):
    cnt = base + (1 if i < rem else 0)
    pool = np.where(scenej == s)[0]
    sel = rng.choice(pool, size=min(cnt, len(pool)), replace=False)
    chosen.append(sel)
    per_scene[s] = int(len(sel))
idx = np.sort(np.concatenate(chosen))
idx = idx[:SAMPLE_N]
sample_sha = sha256_bytes(idx.astype(np.int64).tobytes())

x_tr = torch.from_numpy(xj_s[idx])
y_tr = torch.from_numpy(yj[idx])          # already normalised true costs
x_tr_all = torch.from_numpy(xj_s)         # full judgement for in-role eval
x_bi_all = torch.from_numpy(xb_s)
x_sf_all = torch.from_numpy(xs_s)
y_bi = torch.from_numpy(yb)
y_sf = torch.from_numpy(ys)

# ------------------------------------------------------------- fixed batch order
epochs = int(math.ceil(STEPS / (SAMPLE_N // BATCH)))
order_rng = np.random.default_rng(SEED + 101)
batch_plan = []
for _ in range(epochs):
    perm = order_rng.permutation(SAMPLE_N)
    for b in range(SAMPLE_N // BATCH):
        batch_plan.append(perm[b * BATCH:(b + 1) * BATCH])
batch_plan = batch_plan[:STEPS]

# ------------------------------------------------------------------ two replicas
torch.manual_seed(SEED)
torch.set_num_threads(4)
base_model = MLP()
init_state = {k: v.clone() for k, v in base_model.state_dict().items()}
init_sha = state_dict_sha(base_model)

mixed_model = MLP()
mixed_model.load_state_dict(init_state)
pure_model = MLP()
pure_model.load_state_dict(init_state)
assert state_dict_sha(mixed_model) == state_dict_sha(pure_model) == init_sha

opt_mixed = torch.optim.Adam(mixed_model.parameters(), lr=LR)
opt_pure = torch.optim.Adam(pure_model.parameters(), lr=LR)


def cost_loss_replica(pred, y_norm, mode):
    fit = F.smooth_l1_loss(pred, y_norm)
    if mode == "pure":
        return fit
    costs = torch.cat([torch.zeros_like(pred[:, :1]), pred], 1)
    prob = (-costs).softmax(1)
    true_all = torch.cat([torch.zeros_like(y_norm[:, :1]), y_norm], 1)
    expected = (prob * true_all).sum(1).mean()
    return fit + 0.1 * expected


def formal_loss_numpy(pred, y_norm):
    """Independent numpy reference for the formal mixed loss at scale=1.

    Mirrors action_gate.cost_loss: probability = softmax(-costs), i.e. the
    exponent uses the NEGATED cost vector [0, pred].
    """
    d = pred - y_norm
    fit = float(np.where(np.abs(d) < 1.0, 0.5 * d ** 2, np.abs(d) - 0.5).mean())
    costs = np.concatenate([np.zeros((len(pred), 1)), pred], 1)
    neg = -costs
    e = np.exp(neg - neg.max(1, keepdims=True))
    prob = e / e.sum(1, keepdims=True)
    true_all = np.concatenate([np.zeros((len(pred), 1)), y_norm], 1)
    exp_val = float((prob * true_all).sum(1).mean())
    return fit + 0.1 * exp_val


# equivalence assertion on one batch
with torch.no_grad():
    p0 = mixed_model(x_tr[:BATCH]).numpy()
    lhs = float(cost_loss_replica(mixed_model(x_tr[:BATCH]), y_tr[:BATCH], "mixed").item())
    rhs = formal_loss_numpy(p0, y_tr[:BATCH].numpy())
assert abs(lhs - rhs) < 1e-6, (lhs, rhs)


# ------------------------------------------------------------------ metrics
def role_metrics(model, x_all, y_all, scenes_all):
    with torch.no_grad():
        pred = model(x_all).numpy()
    target = y_all.numpy()
    mae_norm = float(np.abs(pred - target).mean())
    sp = [spearman(pred[:, j], target[:, j]) for j in range(3)]
    costs = np.concatenate([np.zeros((len(pred), 1)), pred], 1)
    true_all = np.concatenate([np.zeros((len(target), 1)), target], 1)
    decision = costs.argmin(1)
    chosen = true_all[np.arange(len(pred)), decision]
    sp_gain = [spearman(-pred[:, j], -target[:, j]) for j in range(3)]
    return dict(
        cost_mae_normalised=mae_norm,
        cost_mae_raw=mae_norm * SCALE_22,
        spearman_cost_per_action=sp,
        spearman_cost_mean=float(np.nanmean([v for v in sp if v is not None])) if any(v is not None for v in sp) else None,
        spearman_gain_per_action=sp_gain,
        decision_action_ratio={str(ACTIONS[k]): float((decision == k).mean()) for k in range(4)},
        natural_accepted_frac=float((decision != 0).mean()),
        natural_accepted_mean_gain_raw=float(-chosen[decision != 0].mean() * SCALE_22) if (decision != 0).any() else None,
        natural_accepted_harmful_rate=float((chosen[decision != 0] > NEUTRAL_BAND).mean()) if (decision != 0).any() else None,
    )


def baselines(y_all):
    target = y_all.numpy()
    zero = float(np.abs(target).mean())
    meanb = float(np.abs(target - target.mean(0)).mean())
    best_const = float(min(np.abs(target - target[:, j:j + 1]).mean() for j in range(3)))
    all_accept = {str(ACTIONS[j + 1]): float(-target[:, j].mean() * SCALE_22) for j in range(3)}
    return dict(zero_baseline_mae_normalised=zero, zero_baseline_mae_raw=zero * SCALE_22,
                mean_baseline_mae_normalised=meanb, best_constant_action_mae_normalised=best_const,
                all_reject_improvement_raw=0.0, all_accept_mean_improvement_raw=all_accept)


def risk_curve_forced(pred, y_all, scenes_all, coverages):
    target = y_all.numpy()
    n = len(pred)
    costs = np.concatenate([np.zeros((n, 1)), pred], 1)
    true_all = np.concatenate([np.zeros((n, 1)), target], 1)
    decision = costs.argmin(1)
    pred_gain = -costs.min(1)
    chosen = true_all[np.arange(n), decision]
    avg_cost_per_action = true_all.mean(0)          # (4,) uniform area weight
    curves = []
    for c in coverages:
        m = max(1, int(round(n * c)))
        order = np.argsort(-pred_gain, kind="stable")[:m]
        frac = np.bincount(decision[order], minlength=4) / m
        random_cost = float((avg_cost_per_action * frac).sum())
        accepted_mean = float(-chosen[order].mean() * SCALE_22)
        global_area = float(-chosen[order].sum() / n * SCALE_22)
        by_scene = {}
        for s in sorted(set(scenes_all.tolist())):
            mask = scenes_all == s
            ns = max(1, int(round(mask.sum() * c)))
            sub_all = np.where(mask)[0]
            o2 = sub_all[np.argsort(-pred_gain[sub_all], kind="stable")[:ns]]
            by_scene[s] = float(-chosen[o2].mean() * SCALE_22)
        curves.append(dict(
            coverage_target=c,
            coverage_area=float(m / n),
            accepted_region_mean_improvement_raw=accepted_mean,
            global_area_improvement_raw=global_area,
            same_action_random_accepted_mean_improvement_raw=float(-random_cost * SCALE_22),
            same_action_random_global_improvement_raw=float(-random_cost * (m / n) * SCALE_22),
            harmful_accepted_rate=float((chosen[order] > NEUTRAL_BAND).mean()),
            action_fraction={str(ACTIONS[k]): float(frac[k]) for k in range(4)},
            by_scene_accepted_mean_improvement_raw=by_scene,
            forced_coverage_is_diagnostic_not_deployment=True,
        ))
    return curves


# --------------------------------------------------------------- training loop
def train(model, opt, mode):
    curve = []

    def record(step):
        with torch.no_grad():
            pred = model(x_tr).numpy()
        target = y_tr.numpy()
        d = pred - target
        fit = float(np.where(np.abs(d) < 1.0, 0.5 * d ** 2, np.abs(d) - 0.5).mean())
        mae = float(np.abs(d).mean())
        sp = [spearman(pred[:, j], target[:, j]) for j in range(3)]
        policy = None
        if mode == "mixed":
            costs = np.concatenate([np.zeros((len(pred), 1)), pred], 1)
            neg = -costs
            e = np.exp(neg - neg.max(1, keepdims=True))
            prob = e / e.sum(1, keepdims=True)
            true_all = np.concatenate([np.zeros((len(target), 1)), target], 1)
            policy = float((prob * true_all).sum(1).mean())
        curve.append(dict(step=step, loss_fit=fit, cost_mae_normalised=mae, cost_mae_raw=mae * SCALE_22,
                          spearman_cost_per_action=sp,
                          spearman_cost_mean=float(np.nanmean([v for v in sp if v is not None])),
                          loss_policy=policy,
                          loss_total=(fit + 0.1 * policy) if policy is not None else fit))

    record(0)
    for step in range(1, STEPS + 1):
        b = batch_plan[step - 1]
        xb = x_tr[b]
        yb_ = y_tr[b]
        for model_, opt_, mode_ in ((model, opt, mode),):
            opt_.zero_grad(set_to_none=True)
            loss = cost_loss_replica(model_(xb), yb_, mode_)
            loss.backward()
            nn.utils.clip_grad_norm_(model_.parameters(), 1.0)
            opt_.step()
        if step % 100 == 0:
            record(step)
    return curve


tm = time.perf_counter()
curve_mixed = train(mixed_model, opt_mixed, "mixed")
curve_pure = train(pure_model, opt_pure, "pure")
train_seconds = time.perf_counter() - tm

# ------------------------------------------------------------- final evaluation
roles = {}
for name, x_all, y_all, sc in (("judgement_train", x_tr_all, torch.from_numpy(yj), scenej),
                               ("judgement_sampled2048", x_tr, y_tr, scenej[idx]),
                               ("bias", x_bi_all, y_bi, sceneb),
                               ("safety", x_sf_all, y_sf, scenes_)):
    roles[name] = dict(mixed=role_metrics(mixed_model, x_all, y_all, sc),
                       pure=role_metrics(pure_model, x_all, y_all, sc))

# bias-role independent additive bias estimate
with torch.no_grad():
    pred_mixed_bias = mixed_model(x_bi_all).numpy()
    pred_pure_bias = pure_model(x_bi_all).numpy()
bias_mixed = (y_bi.numpy() - pred_mixed_bias).mean(0)
bias_pure = (y_bi.numpy() - pred_pure_bias).mean(0)


def debiased_mae(model, bias, x_all, y_all):
    with torch.no_grad():
        pred = model(x_all).numpy() + bias
    return float(np.abs(pred - y_all.numpy()).mean())


bias_report = dict(
    action_bias_normalised={"mixed": bias_mixed.tolist(), "pure": bias_pure.tolist()},
    action_bias_raw={"mixed": (bias_mixed * SCALE_22).tolist(), "pure": (bias_pure * SCALE_22).tolist()},
    judgement_train_debiased_mae={"mixed": debiased_mae(mixed_model, bias_mixed, x_tr_all, torch.from_numpy(yj)),
                                  "pure": debiased_mae(pure_model, bias_pure, x_tr_all, torch.from_numpy(yj))},
    safety_debiased_mae={"mixed": debiased_mae(mixed_model, bias_mixed, x_sf_all, y_sf),
                         "pure": debiased_mae(pure_model, bias_pure, x_sf_all, y_sf)},
    note="bias estimated on the bias role only; reported as a diagnostic, not applied as a deployed offset.",
)

# ------------------------------------------------------- risk-coverage diagnostic
with torch.no_grad():
    pred_mixed_sf = mixed_model(x_sf_all).numpy()
    pred_pure_sf = pure_model(x_sf_all).numpy()
curves_mixed = risk_curve_forced(pred_mixed_sf, y_sf, scenes_, COVERAGE_GRID)
curves_pure = risk_curve_forced(pred_pure_sf, y_sf, scenes_, COVERAGE_GRID)
check_m = next(c for c in curves_mixed if abs(c["coverage_target"] - SIGNAL_COVERAGE) < 1e-9)
check_p = next(c for c in curves_pure if abs(c["coverage_target"] - SIGNAL_COVERAGE) < 1e-9)


def signal_of(check):
    pos = sum(v > 0 for v in check["by_scene_accepted_mean_improvement_raw"].values())
    ok = (check["accepted_region_mean_improvement_raw"] > 0
          and check["accepted_region_mean_improvement_raw"] > check["same_action_random_accepted_mean_improvement_raw"]
          and pos >= 3 and len(check["by_scene_accepted_mean_improvement_raw"]) == 4)
    return ok, pos


sig_m, pos_m = signal_of(check_m)
sig_p, pos_p = signal_of(check_p)

risk_report = dict(
    safety_scenes=sorted(set(scenes_.tolist())),
    coverage_grid=list(COVERAGE_GRID),
    mixed=curves_mixed, pure=curves_pure,
    signal_at_10pct={"mixed": sig_m, "pure": sig_p},
    positive_safety_scenes_at_10pct={"mixed": pos_m, "pure": pos_p},
    rule="signal_present if top-10% accepted-region mean actual improvement > 0, better than the "
         "same-action random expectation, and >=3 of 4 safety scenes improve; it is not a deployment "
         "safety proof.",
    forced_coverage_is_diagnostic_not_deployment=True,
)

# --------------------------------------------------------------------- verdict
fin_m = curve_mixed[-1]
fin_p = curve_pure[-1]
bl = baselines(torch.from_numpy(yj))
verdict = dict(
    lfit_improves_over_step0={"mixed": bool(fin_m["loss_fit"] < curve_mixed[0]["loss_fit"]),
                              "pure": bool(fin_p["loss_fit"] < curve_pure[0]["loss_fit"])},
    final_fit_beats_zero_baseline={"mixed": bool(fin_m["cost_mae_normalised"] < bl["zero_baseline_mae_normalised"]),
                                   "pure": bool(fin_p["cost_mae_normalised"] < bl["zero_baseline_mae_normalised"])},
    final_fit_beats_best_constant_baseline={"mixed": bool(fin_m["cost_mae_normalised"] < bl["best_constant_action_mae_normalised"]),
                                            "pure": bool(fin_p["cost_mae_normalised"] < bl["best_constant_action_mae_normalised"])},
    pure_better_fit_than_mixed=bool(fin_p["loss_fit"] < fin_m["loss_fit"]),
    pure_better_cost_mae_than_mixed=bool(fin_p["cost_mae_normalised"] < fin_m["cost_mae_normalised"]),
    signal_present_at_10pct={"mixed": sig_m, "pure": sig_p},
    conclusion_limit="Both probes fitting poorly only means this probe/budget/representation did not fit; "
                     "it does not prove the native module is broken. Good MLP fit would not prove the "
                     "spatial convolutional native head is solved.",
)

# ------------------------------------------------------------------- output
report = dict(
    stage="A1",
    status="lightweight_probe_complete",
    run=str(RUN),
    source22=str(SRC),
    cpu_only=True,
    gpu_used=False,
    test_used=False,
    validation_used_for_any_rule=False,
    probe_only=True,
    restoration_method_score=False,
    config=dict(sample_n=SAMPLE_N, seed=SEED, hidden=HIDDEN, lr=LR, batch=BATCH, steps=STEPS,
                arch="50-32-32-3 GELU", optimiser="Adam", endpoint=STEPS,
                standardisation_source="judgement training cache (all rows)",
                input_channels=50, action_set=list(ACTIONS), neutral_band=NEUTRAL_BAND,
                coverage_grid=list(COVERAGE_GRID)),
    standardisation=dict(mean=chan_mean.tolist(), std=chan_std.tolist(),
                         near_zero_std_channels=[int(i) for i in np.where(chan_std < 1e-6)[0]],
                         source="judgement training cache only"),
    sampling=dict(count=int(len(idx)), per_scene=per_scene, indices_sha256=sample_sha,
                  indices=idx.tolist()),
    init=dict(scheme="torch default Linear init (uniform +-1/sqrt(fan_in)), manual_seed 2026",
              common_init_sha256=init_sha,
              identical_for_both_replicas=True),
    loss_equivalence_check=dict(torch_mixed_vs_numpy_formal_abs_diff=abs(lhs - rhs),
                                torch_uses_action_gate_cost_loss_definition=True),
    training=dict(mixed=curve_mixed, pure=curve_pure, train_seconds=train_seconds,
                  batch_order_identical=True, batch_plan_seed=SEED + 101),
    roles=roles,
    baselines_judgement_train=bl,
    bias_estimate=bias_report,
    risk_coverage=risk_report,
    verdict=verdict,
    notes=[
        "probe_only diagnostic; not a native convolutional head result and not a full R2P score.",
        "cache x is the scattered 50-d projection input; no spatial adjacency, so no image reshape/3x3 head.",
        "y is already normalised by the verified cache-generation cost_scale; the raw-unit MAE multiplies by it.",
        "scattered regions are not independent images; metrics are reported as-is without a significance claim.",
    ],
)

out = RUN / "metrics" / "A1_probe_results.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

heads = RUN / "checkpoints" / "diagnostic_heads"
heads.mkdir(parents=True, exist_ok=True)
torch.save({"state_dict": mixed_model.state_dict(), "mode": "mixed", "init_sha256": init_sha}, heads / "a1_mlp_mixed.pt")
torch.save({"state_dict": pure_model.state_dict(), "mode": "pure", "init_sha256": init_sha}, heads / "a1_mlp_pure.pt")
torch.save({"state_dict": init_state, "sha256": init_sha}, heads / "a1_mlp_common_init.pt")

print(json.dumps(dict(
    wrote=str(out),
    init_sha=init_sha,
    final_fit_mixed=fin_m["loss_fit"], final_fit_pure=fin_p["loss_fit"],
    final_mae_mixed=fin_m["cost_mae_normalised"], final_mae_pure=fin_p["cost_mae_normalised"],
    best_constant_baseline=bl["best_constant_action_mae_normalised"],
    spearman_mixed=fin_m["spearman_cost_per_action"], spearman_pure=fin_p["spearman_cost_per_action"],
    signal_mixed=sig_m, signal_pure=sig_p, train_seconds=round(train_seconds, 2),
), ensure_ascii=False))
