"""CPU-only verification of the retained matched optimizer/checkpoint states."""
import hashlib
import csv
import math
import json
from pathlib import Path
import torch
import yaml
ROOT = Path(__file__).resolve().parent.parent
out = {}
for name in ("phase1", "phase01"):
    root = ROOT / "runs" / name / "seed_2026"
    config = yaml.safe_load((ROOT / "configs" / (name + ".yaml")).read_text())
    audits = {}
    records = list(csv.DictReader((root / "logs/train.csv").open()))
    assert [int(r["step"]) for r in records] == list(range(1, 3001))
    assert all(math.isfinite(float(v)) for r in records for v in r.values())
    for step in (200, 3000):
        path = root / "checkpoints" / f"step_{step}.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        assert state["step"] == step
        assert state["phase_weight"] == config["study"]["phase_weight"]
        assert state["lambda_feature_alignment"] == config["study"]["fixed_lambda"]
        groups = state["optimizer"]["param_groups"]
        assert [g["lr"] for g in groups] == [config["optimization"]["lr_r2p"], config["optimization"]["lr_tasknet"]]
        adam_steps = sorted({int(s["step"]) for s in state["optimizer"]["state"].values()})
        assert adam_steps == [step], (name, step, adam_steps)
        assert all(torch.isfinite(t).all() for k in ("r2p_and_projections", "tasknet_net_g") for t in state[k].values())
        assert all(torch.isfinite(t).all() for s in state["optimizer"]["state"].values() for t in s.values() if isinstance(t, torch.Tensor))
        audits[str(step)] = {"checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "optimizer_steps": adam_steps, "optimizer_state_count": len(state["optimizer"]["state"]), "learning_rates": [g["lr"] for g in groups], "passed": True}
        del state
    # Under no-feedback these disjoint logged blocks cover the model group.
    clip_count = sum(math.sqrt(float(r["r2p_grad_norm"]) ** 2 + float(r["projection_grad_norm"]) ** 2 + float(r["feedback_adapter_grad_norm"]) ** 2 + float(r["feedback_scale_grad"]) ** 2) > 1 for r in records)
    netg_clip_count = sum(float(r["tasknet_grad_norm"]) > 1 for r in records)
    out[name] = {"checkpoints": audits, "training_log": {"continuous_steps_1_to_3000": True, "all_logged_values_finite": True, "model_group_preclip_norm_gt1_count": clip_count, "netg_preclip_norm_gt1_count": netg_clip_count, "note": "Counts reconstructed from disjoint no-feedback parameter block norms; threshold is 1."}}
(ROOT / "diagnostics/final_checkpoint_audit.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
