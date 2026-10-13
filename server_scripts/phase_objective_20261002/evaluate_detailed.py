"""Reuse official evaluation once; retain task-loss components from its outputs.

No additional TaskNet forward and no training-code change. RNG is checked around
the extra loss arithmetic. Both objectives are computed for comparable reporting.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import run_phase_objective as study

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("operation", choices=["evaluate"])
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--split", required=True)
args = ap.parse_args()
config = yaml.safe_load(Path(args.config).read_text())
root = Path(config["paths"]["output_root"])
base = study.base
base.add_source_path(config)
old_build = base.build_components
old_forward = base.recurrent_forward
criterion = None
rows = []

def build(cfg, device):
    global criterion
    result = old_build(cfg, device)
    criterion = result[2]
    return result

def forward(model, task, batch, *extra, **kw):
    output = old_forward(model, task, batch, *extra, **kw)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = [x.clone() for x in torch.cuda.get_rng_state_all()]
    target = batch["target"]
    perceptual = criterion.perceptual
    target_features = perceptual.extractor(target.detach())["conv5_4"]
    loss_ssim = sys.modules["polarapp.losses"].ssim
    coefficient = float(config["study"]["phase_weight"])
    def parts(prediction):
        l1 = float(F.l1_loss(prediction, target))
        st = float(1 - loss_ssim(prediction, target))
        phase_weighted = float(criterion.phase(prediction, target))
        phase_raw = phase_weighted / coefficient
        features = perceptual.extractor(prediction)["conv5_4"]
        pe = float(perceptual.weight * F.l1_loss(features, target_features))
        nonphase = l1 + st + pe
        return {"l1": l1, "ssim_term": st, "phase_raw": phase_raw,
                "phase_weighted": phase_weighted, "perceptual_weighted": pe,
                "total_training_coefficient": nonphase + phase_weighted,
                "total_common_phase1": nonphase + phase_raw,
                "total_common_phase01": nonphase + 0.1 * phase_raw}
    row = {"sample_id": batch["sample_id"][0], "scene": batch["scene"][0],
           "round1": parts(output.t1), "round2": parts(output.t2)}
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    assert all(np.isfinite(v) for rnd in ("round1", "round2") for v in row[rnd].values())
    rows.append(row)
    if len(rows) % 25 == 0:
        print(f"Evaluated {len(rows)} images", flush=True)
    return output

base.build_components = build
base.recurrent_forward = forward
tick = time.time()
result = base.run_evaluation(config, torch.device("cuda"), args.checkpoint, args.split)
assert len(rows) == result["frames"]
scenes = sorted({r["scene"] for r in rows})
payload = {"scope": "same official validation forward; no extra TaskNet calls",
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "checkpoint_sha256": result["checkpoint_sha256"],
           "allocator_setting": "expandable_segments:True",
           "frames": len(rows), "elapsed_seconds": time.time() - tick,
           "phase_weight": float(config["study"]["phase_weight"]),
           "rng_unchanged_by_loss_arithmetic": True, "per_image": rows}
for rnd in ("round1", "round2"):
    keys = rows[0][rnd].keys()
    payload[rnd] = {"image_mean": {k: float(np.mean([r[rnd][k] for r in rows])) for k in keys},
                    "scene_macro": {k: float(np.mean([np.mean([r[rnd][k] for r in rows if r["scene"] == s]) for s in scenes])) for k in keys}}
payload["iteration_delta"] = {aggregation: {k: payload["round2"][aggregation][k] - payload["round1"][aggregation][k] for k in rows[0]["round1"]} for aggregation in ("image_mean", "scene_macro")}
base.atomic_json(root / "metrics" / f"step_3000_{args.split}_loss_parts.json", payload)
print(json.dumps({"frames": result["frames"], "round1": result["round1"], "round2": result["round2"], "iteration_gain": result["iteration_gain"], "loss_iteration_delta": payload["iteration_delta"]}, indent=2))
