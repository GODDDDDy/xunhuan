#!/usr/bin/env python3
"""Verify that the diagnostic script's loss-component recomputation is numerically
identical to the OFFICIAL training criterion itself.

This is the decisive check for review item #2 ("compare the sum of the diagnostic
components against the training criterion definition").

READ-ONLY: builds the criterion, runs it on a handful of images, and compares
its scalar output against a manual recomposition. No training, no weight update.
Runs on GPU only because the criterion lives there; a small image limit keeps it
to a few minutes.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

R = Path("/media/li/data3/25thstudent/DY/929work/pidsr_k2_929")
SRC = R / "deps/rgb_final_confirmation/src"
SEED = 2026
LIMIT = 4


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    device = torch.device("cuda:0")
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    out: dict = {"purpose": "verify diagnostic loss recomposition == official criterion"}

    results = {}
    for group in ("no_feedback", "feedback"):
        mod = load_module("re_mod_" + group, R / "experiment/run_experiment.py")
        cfg_path = R / ("experiment/config.yaml" if group == "feedback"
                        else "experiment/config_no_feedback.yaml")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        mod.add_source_path(cfg)
        mod.set_seed(SEED)
        model, task, criterion, _base = mod.build_components(cfg, device)

        loss_mod = sys.modules.get("polarapp.losses")
        loss_ssim = loss_mod.ssim
        perceptual = criterion.perceptual
        perc_weight = float(perceptual.weight) if perceptual is not None else 0.0

        # cached perceptual extractor identity check
        ext = perceptual.extractor

        loader = mod.split_loader(cfg, "validation")
        rows = []
        with torch.no_grad():
            for index, raw in enumerate(loader):
                if index >= LIMIT:
                    break
                batch = mod.unpack_batch(raw, device)
                mod.set_step_seed(SEED, "validation", index, batch["sample_id"])
                target = batch["target"]
                # Use the official task output as the prediction (any tensor works;
                # we only need the loss arithmetic to match).
                pred, _ = task(batch["polar"], batch["official_rgb"], None)
                pred = pred.detach()

                # (1) the OFFICIAL criterion path
                official = float(criterion(pred, target))

                # (2) the DIAGNOSTIC recomposition, exactly as in alpha_sweep_diagnosis.py
                l1 = float(F.l1_loss(pred, target))
                s_t = float(1.0 - loss_ssim(pred, target))
                ph = float(criterion.phase(pred, target))
                tgt_feat = ext(target.detach())["conv5_4"]
                pf = ext(pred)["conv5_4"]
                pe = float(perc_weight * F.l1_loss(pf, tgt_feat))
                manual = l1 + s_t + ph + pe

                rows.append({
                    "index": index,
                    "sample_id": batch["sample_id"][0],
                    "official_criterion": official,
                    "manual_recomposition": manual,
                    "abs_diff": abs(official - manual),
                    "parts": {"l1": l1, "ssim_term": s_t, "phase": ph, "perceptual": pe},
                })

        results[group] = {
            "perceptual_weight": perc_weight,
            "extractor_layer_names": list(ext.layer_name_list),
            "extractor_use_input_norm": bool(getattr(ext, "use_input_norm", None))
            if hasattr(ext, "use_input_norm") else None,
            "extractor_range_norm": bool(getattr(ext, "range_norm", None))
            if hasattr(ext, "range_norm") else None,
            "n_checked": len(rows),
            "max_abs_diff": max(r["abs_diff"] for r in rows),
            "identical_within_1e_12": all(r["abs_diff"] < 1e-12 for r in rows),
            "identical_within_float32_eps": all(
                r["abs_diff"] <= 1e-6 * max(1.0, abs(r["official_criterion"]))
                for r in rows),
            "rows": rows,
        }
        print("[%s] weight=%.6g n=%d max_abs_diff=%.3e"
              % (group, perc_weight, len(rows), results[group]["max_abs_diff"]),
              flush=True)
        del model, task, criterion
        torch.cuda.empty_cache()

    out["groups"] = results

    # R2P round weights, from config (locked by an assertion in run_experiment)
    cfg_f = yaml.safe_load((R / "experiment/config.yaml").read_text(encoding="utf-8"))
    out["round_loss_weights"] = cfg_f["recurrent"]["round_loss_weights"]
    out["training_objective"] = (
        "total = round_weights[0]*criterion(T1) + round_weights[1]*criterion(T2) "
        "+ lambda_feature_alignment*alignment_loss  (alignment only in group A paths)")

    p = Path("/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/"
             "evaluation_872/alpha_sweep/loss_recompute_verification.json")
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("WROTE", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
