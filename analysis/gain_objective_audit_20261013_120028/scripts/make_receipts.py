#!/usr/bin/env python3
"""Build receipts/source_and_cache_receipt.json and config/diagnostic_lock.json. Stdlib only."""
import json
import sys
import time
from pathlib import Path

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
a0 = json.loads((RUN / "metrics" / "A0_numeric_audit.json").read_text(encoding="utf-8"))

receipt = {
    "created_at_unix": time.time(),
    "run": a0["run"],
    "source22": a0["source22"],
    "stage": "A0",
    "verified": a0["A0_passed"],
    "files": [dict(name=k, **{kk: vv for kk, vv in v.items() if kk in ("path", "bytes", "sha256", "ok")})
              for k, v in a0["integrity"].items()],
    "role_scenes": a0["roles"]["scenes"],
    "role_scenes_disjoint": a0["roles"]["scenes_disjoint"],
    "label_scale": a0["label_scale"],
    "cache_structure": {r: {k: a0["cache_structure"][r][k] for k in ("x", "y", "scene", "files")}
                        for r in ("judgement", "bias", "safety")},
    "channel_stats_role": "judgement_training_cache_only",
    "no_test_read": True, "no_validation_read": True, "no_gpu": True,
}
(RUN / "receipts" / "source_and_cache_receipt.json").write_text(
    json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")

lock = {
    "stage": "A",
    "execution_scope": "diagnostic_only",
    "formal_training_enabled": False,
    "factorial_training_enabled": False,
    "default_stage": "A",
    "stage_A_gpu_budget_seconds": 7200,
    "stage_A_full_validation_frame_cap": 32,
    "stage_A_test_frames": 0,
    "monitor_interval_hours": 2,
    "notify_on": ["failure", "completion", "user_action_required"],
    "seed": 2026,
    "source_weights": {
        "final_22nd_sha256": "279d0a329d0efebee9678c2156bf3e7b60c0646e75a5f37f8015c7b100f9e9a2",
        "reused_21st_sha256": "cecdcfa47d290b0a2d6b662b96bb4407a7043ef5789e9e87d70768f2c683916d",
        "cache_generation_scale": 9.34407114982605e-4,
    },
    "A1_probe": {
        "representation": "existing 50-d x, no rescaling sweep",
        "standardization": "judgement training role only",
        "samples": 2048, "sampling": "scene-balanced, fixed seed 2026, indices recorded",
        "mlp": [50, 32, 32, 3], "activation": "GELU",
        "optimizer": "Adam", "lr": 1e-3, "batch": 256, "steps": 1000,
        "log_every": 100, "endpoint_selection": "1000 only, no best-midpoint",
        "copies": ["mixed", "pure"], "shared_initialisation": True, "shared_batch_order": True,
        "risk_coverage": [0.01, 0.05, 0.10, 0.20],
        "probe_only": True, "restoration_method_score": False,
    },
    "A2_native_head": {
        "model": "50-input native architecture, strict load of 22nd final.pt",
        "trainable": ["action_gate", "action_gate.feature_projection"],
        "frozen": "all other model tensors and entire task net (eval + requires_grad False)",
        "crops": {"judgement": 24, "bias": 8, "safety": 16, "per_crop": 128},
        "scene_selection": "fixed hash order over sample_id, no GT/result read",
        "cache_equivalence": {"crops": 5, "atol": 1e-6, "rtol": 1e-4},
        "optimizer": "Adam", "lr": 1e-4, "effective_batch": 2, "microbatch": 1,
        "clip_norm": 1, "steps": 500, "endpoint": 500,
        "copies": ["mixed", "pure"], "shared_order_and_rng": True,
        "bias_crops": 8, "safety_crops": 16, "safety_grid": [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8],
        "scale": "reused 22nd source scale, identical for both heads",
    },
    "A3_screen": {
        "max_frames": 32, "validation_scenes": 8, "per_scene": 4,
        "selection": "sample_id sorted, indices floor(j*(n-1)/3), j=0..3; locked before reading GT/metrics",
        "shared_backbone": True, "heads": ["source", "mixed", "pure"],
        "test_used": False,
    },
    "started_at": "2026-10-13T12:00:28+08:00",
    "new_gpu_budget": True, "old_budget_reused": False,
    "history_readonly": True,
}
(RUN / "config" / "diagnostic_lock.json").write_text(
    json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote receipts/source_and_cache_receipt.json and config/diagnostic_lock.json")
