"""Stage A2, part 2: paired native-head learning diagnostic and screening.

Two identical 50-channel native judgement heads start from the experiment-22
final action_gate. They are trained with the same 24-crop cycle, same 500-step
fixed endpoint, same (empty) optimizer state and the same random stream; the only
difference is the loss (L_mixed = L_fit + 0.1*L_policy versus L_pure = L_fit).

Diagnostic only: not a formal recovery method, not a convergence proof.
"""
import copy
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(HERE))

from common import guard_gpu, json_save, save, seed, sha, status  # noqa: E402

assert Path(__import__("common").ROOT).resolve() == RUN, "ROOT must be the new audit run"

import gain22  # noqa: E402
from action_gate import ACTIONS, action_costs, cost_loss  # noqa: E402

from a2_common import build_head_inputs, spearman  # noqa: E402

STEPS = 500
LR = 1e-4
BATCH = 2
CLIP = 1.0
MONITOR = (0, 100, 200, 300, 400, 500)
HARMFUL_GAP = 0.02  # in normalised cost units; equals 0.02 * scale in raw MSE


def load_all():
    cfg = json.loads((RUN / "config/locked_run.json").read_text(encoding="utf-8"))
    index = json.loads((RUN / "cache/a2_cache_index.json").read_text(encoding="utf-8"))
    payload = np.load(RUN / "cache/a2_spatial_cache.npz")
    return cfg, index, payload


def build_template(cfg, device):
    """Real 50-channel native head, taken from the production construction path.

    Built through gain22.components so the architecture cannot drift from
    experiment 22. LocalAcceptance() must NOT be used as the source directly:
    its body ends in Conv2d(32,4,1), whereas ActionCostGate rebuilds the head as
    Conv2d(32,3,1). Only the architecture matters here; the diagnostic loads its
    own weights.
    """
    model, task, raw = gain22.components(cfg, device)
    template = copy.deepcopy(model.action_gate).to("cpu")
    del model, task, raw
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return template


def make_head(template, state, device):
    head = copy.deepcopy(template).to(device)
    head.load_state_dict(state, strict=True)
    head.train()
    head.requires_grad_(True)
    return head


def record_tensors(payload, positions, device):
    keys = list(payload.keys())
    out = []
    for position in positions:
        out.append({k: torch.from_numpy(payload[k][position])[None].to(device) for k in keys})
    return out


def loss_of(raw, y_raw, scale, mode):
    if mode == "mixed":
        cost4 = torch.cat([torch.zeros_like(y_raw[:, :1]), y_raw], 1)
        return cost_loss(raw, cost4, scale)
    return F.smooth_l1_loss(raw, (y_raw / scale).detach())


def pure_fit(raw, y_raw, scale):
    return float(F.smooth_l1_loss(raw, (y_raw / scale).detach()))


def predict(head, records):
    head.eval()
    chunks = []
    with torch.no_grad():
        for rec in records:
            raw = head.body(build_head_inputs(rec, head))
            chunks.append(raw[0].reshape(3, -1).transpose(0, 1))
    head.train()
    return torch.cat(chunks, 0)


def target_matrix(records):
    return torch.cat([rec["y_raw"][0].reshape(3, -1).transpose(0, 1) for rec in records], 0)


def describe(head, records, scale, baseline_median=None):
    pred = predict(head, records)
    true_norm = target_matrix(records) / scale
    diff = (pred - true_norm).abs()
    entries = {}
    for j in range(3):
        entries[f"action_{ACTIONS[j + 1]}"] = dict(
            cost_mae_raw=float(diff[:, j].mean()) * scale,
            cost_mae_normalised=float(diff[:, j].mean()),
            spearman=spearman(pred[:, j].cpu().numpy(), true_norm[:, j].cpu().numpy()),
            predicted_mean=float(pred[:, j].mean()),
            actual_mean=float(true_norm[:, j].mean()),
        )
    result = dict(
        pure_fit=float(F.smooth_l1_loss(pred, true_norm.detach())),
        cost_mae_raw=float(diff.mean()) * scale,
        cost_mae_normalised=float(diff.mean()),
        spearman_by_action={k: v["spearman"] for k, v in entries.items()},
        per_action=entries,
        cells=int(pred.shape[0]),
        zero_baseline_mae_raw=float(true_norm.abs().mean()) * scale,
        train_mean_baseline_mae_raw=(None if baseline_median is None
                                     else float((baseline_median[None, :] - true_norm).abs().mean()) * scale),
    )
    return result, pred, true_norm


def screening(head, bias_records, safety_records, scale, grid, labels):
    bias_pred = predict(head, bias_records)
    bias_true = target_matrix(bias_records) / scale
    bias = (bias_true - bias_pred).mean(0)
    safety_pred = predict(head, safety_records) + bias
    safety_true = target_matrix(safety_records) / scale
    zeros = torch.zeros_like(safety_pred[:, :1])
    rows = []
    for margin in grid:
        costs = torch.cat([zeros, safety_pred + margin], 1)
        decision = costs.argmin(1)
        accepted = decision > 0
        true_selected = torch.cat([zeros, safety_true], 1).gather(1, decision[:, None])[:, 0]
        coverage = float(accepted.float().mean())
        harmful = (float((true_selected[accepted] > HARMFUL_GAP).float().mean())
                   if bool(accepted.any()) else 0.0)
        gain = -float(true_selected.mean()) * scale
        rows.append(dict(margin=float(margin), coverage=coverage,
                         accepted_harmful_rate=(harmful if bool(accepted.any()) else None),
                         mean_mse_improvement=gain,
                         eligible=bool(gain >= 0 and harmful <= 0.2)))
    eligible = [r for r in rows if r["eligible"]]
    if eligible:
        best = max(eligible, key=lambda r: (r["coverage"], -r["margin"]))
    else:
        best = dict(margin=1e9, coverage=0.0, mean_mse_improvement=0.0,
                    accepted_harmful_rate=None, eligible=False,
                    reason="no eligible threshold; reject all")
    reject_all = dict(coverage=0.0, mean_mse_improvement=0.0, accepted_harmful_rate=None)
    accept_all = dict(coverage=1.0,
                      mean_mse_improvement=-float(safety_true.min(1).values.mean()) * scale,
                      accepted_harmful_rate=float((safety_true.min(1).values > HARMFUL_GAP).float().mean()))
    deployed = bool(best["coverage"] > 0)
    return dict(
        label=labels, bias=[float(v) for v in bias], bias_crops=len(bias_records),
        safety_crops=len(safety_records), grid=rows, chosen=best,
        safe=True, useful=bool(deployed and best["mean_mse_improvement"] > 0),
        accepted_harmful_rate=None if not deployed else best["accepted_harmful_rate"],
        reference_reject_all=reject_all, reference_accept_all=accept_all,
        harmful_gap_normalised=HARMFUL_GAP, harmful_gap_raw=HARMFUL_GAP * scale,
        calibration_is_small_sample_diagnostic_not_deployable=True)


def main():
    started = time.perf_counter()
    status("A2_train_start")
    cfg, index, payload = load_all()
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise SystemExit("A2 requires CUDA")
    guard_gpu()
    seed(cfg["seed"])
    scale = float(index["scale"])
    init_path = RUN / "checkpoints/diagnostic_heads/a2_native_common_init.pt"
    common = torch.load(init_path, map_location="cpu", weights_only=False)
    common_state = common["state"]
    assert common["input_channels"] == 50

    template = build_template(cfg, device)
    architecture = dict(
        input_channels=int(template.body[0].in_channels),
        output_channels=int(template.body[-1].out_channels),
        projection_out_channels=int(template.feature_projection.out_channels),
        state_keys_equal=bool(set(template.state_dict()) == set(common_state)),
        shapes_equal=bool(all(tuple(template.state_dict()[k].shape) == tuple(v.shape)
                              for k, v in common_state.items())),
    )
    assert architecture["input_channels"] == 50 and architecture["output_channels"] == 3
    assert architecture["state_keys_equal"] and architecture["shapes_equal"], architecture
    print(json.dumps(dict(architecture=architecture)), flush=True)

    judgement_pos = index["role_positions"]["judgement"]
    bias_pos = index["role_positions"]["bias"]
    safety_pos = index["role_positions"]["safety"]
    train_records = record_tensors(payload, judgement_pos, device)
    bias_records = record_tensors(payload, bias_pos, device)
    safety_records = record_tensors(payload, safety_pos, device)
    assert len(train_records) == 24 and len(bias_records) == 8 and len(safety_records) == 16

    train_true = target_matrix(train_records) / scale
    # Deterministic: CUDA median-with-indices is not deterministic under
    # torch.use_deterministic_algorithms(True); the tensor is tiny so use CPU.
    baseline_median = train_true.cpu().median(0).values.to(device)

    heads = {mode: make_head(template, common_state, device) for mode in ("mixed", "pure")}
    shared_init_equal = all(torch.equal(a, b) for (na, a), (nb, b) in
                            zip(heads["mixed"].state_dict().items(), heads["pure"].state_dict().items()))
    assert shared_init_equal
    optimizers = {mode: torch.optim.Adam(heads[mode].parameters(), lr=LR) for mode in ("mixed", "pure")}

    monitor = {}
    for mode, head in heads.items():
        result, _, _ = describe(head, train_records, scale, baseline_median)
        monitor[mode] = {0: result}

    order = list(range(24))
    log_rows = []
    tick = time.perf_counter()
    for step in range(1, STEPS + 1):
        guard_gpu()
        for mode in ("mixed", "pure"):
            head = heads[mode]
            optimizer = optimizers[mode]
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            ids = []
            for k in range(BATCH):
                position = order[((step - 1) * BATCH + k) % 24]
                rec = train_records[position]
                raw = head.body(build_head_inputs(rec, head))
                loss = loss_of(raw, rec["y_raw"], scale, mode) / BATCH
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite loss at step {step} mode {mode}")
                loss.backward()
                step_loss += float(loss.detach())
                ids.append(index["crops"][judgement_pos[position]]["sample_id"])
            norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), CLIP))
            optimizer.step()
            log_rows.append(dict(step=step, mode=mode, loss=step_loss,
                                 grad_norm=norm, learning_rate=optimizer.param_groups[0]["lr"],
                                 finite=True, sample_ids=";".join(ids)))
        if step in MONITOR:
            for mode, head in heads.items():
                result, _, _ = describe(head, train_records, scale, baseline_median)
                monitor[mode][step] = result
            print(json.dumps(dict(step=step, mixed=monitor["mixed"][step]["pure_fit"],
                                 pure=monitor["pure"][step]["pure_fit"])), flush=True)
    train_seconds = time.perf_counter() - tick

    log_path = RUN / "logs/a2_native_train.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(log_rows[0]))
        writer.writeheader()
        writer.writerows(log_rows)

    final = {}
    for mode, head in heads.items():
        train_metrics, _, _ = describe(head, train_records, scale, baseline_median)
        bias_metrics, _, _ = describe(head, bias_records, scale, None)
        safety_metrics, _, _ = describe(head, safety_records, scale, None)
        screen = screening(head, bias_records, safety_records, scale,
                           cfg["staged"]["safety_margin_grid"], mode)
        final[mode] = dict(
            train=train_metrics, cross_scene_bias=bias_metrics, cross_scene_safety=safety_metrics,
            screening=screen, monitors=monitor[mode],
            fit_decreased=bool(monitor[mode][STEPS]["pure_fit"] < monitor[mode][0]["pure_fit"]),
            beats_train_mean_baseline=bool(train_metrics["cost_mae_raw"] <
                                           (train_metrics["train_mean_baseline_mae_raw"] or 1e9)),
            beats_zero_baseline=bool(train_metrics["cost_mae_raw"] < train_metrics["zero_baseline_mae_raw"]),
            cross_scene_ranking_positive=bool(all((v or 0) > 0 for v in
                                                  safety_metrics["spearman_by_action"].values())),
        )
        save(RUN / f"checkpoints/diagnostic_heads/a2_native_{mode}.pt",
             dict(state=head.state_dict(), mode=mode, steps=STEPS, lr=LR, batch=BATCH, clip=CLIP,
                  shared_init_sha256=common["sha256"], loss="L_fit+0.1*L_policy" if mode == "mixed" else "L_fit",
                  probe_only=True, restoration_method_score=False))

    timing = dict(train_seconds=train_seconds, total_seconds=time.perf_counter() - started,
                  steps=STEPS, crops=24)
    result = dict(
        stage="A2_native_head_diagnostic",
        status="native_head_diagnostic_complete",
        probe_only=True, restoration_method_score=False,
        formal_training=False, convergence_claimed=False,
        native_head=True, input_channels=50,
        native_head_architecture=architecture,
        shared_initialisation=dict(sha256=common["sha256"],
                                   source="experiment22 final action_gate",
                                   source_final_sha256=common["source_final_sha256"],
                                   tensors_equal_before_training=shared_init_equal),
        config=dict(steps=STEPS, learning_rate=LR, effective_batch=BATCH, microbatch=1,
                    grad_clip=CLIP, crops=24, role="judgement", order="fixed 24-crop cycle",
                    optimizer="Adam (new empty state)", seed=cfg["seed"],
                    only_variable="loss expectation term",
                    monitor_steps=list(MONITOR), harmful_gap=HARMFUL_GAP),
        scale=scale, heads=final, timing=timing,
        baseline=dict(train_mean_baseline_mae_raw=float((baseline_median[None, :] - train_true).abs().mean()) * scale,
                      zero_baseline_mae_raw=float(train_true.abs().mean()) * scale),
        limitations=[
            "single seed; inherited weights have seen these scenes",
            "24 diagnostic crops, not the 1140-frame formal coverage",
            "fixed 500-step endpoint; no convergence proof",
            "diagnostic bias/threshold only; B must recalibrate on the full original roles",
        ],
        test_used=False, validation_used=False, source_diagnostic_872_rerun=False,
        next_stage_not_auto_started=["B formal matched training", "C module synergy"],
    )
    json_save(RUN / "metrics/A2_native_fit_results.json", result)
    json_save(RUN / "logs/a2_train_timing.json", timing)
    print(json.dumps(dict(done=True,
                          mixed_final_pure_fit=final["mixed"]["monitors"][STEPS]["pure_fit"],
                          pure_final_pure_fit=final["pure"]["monitors"][STEPS]["pure_fit"],
                          mixed_train_pure_fit=final["mixed"]["train"]["pure_fit"],
                          timing=timing)), flush=True)
    status("A2_train_complete")


if __name__ == "__main__":
    main()
