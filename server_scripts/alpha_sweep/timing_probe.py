#!/usr/bin/env python3
"""Measure the per-call cost breakdown at evaluation resolution.

Purpose: decide whether a full-validation (872 images) alpha sweep is affordable,
and if not, which component dominates. Read-only; no artefacts written.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

R = Path("/media/li/data3/25thstudent/DY/929work/pidsr_k2_929")
SRC = R / "deps/rgb_final_confirmation/src"
SEED = 2026


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def read_yaml(p):
    with open(p, "r", encoding="utf-8") as h:
        return yaml.safe_load(h)


def main() -> int:
    device = torch.device("cuda:0")
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    mod = load_module("re_probe", R / "experiment/run_experiment.py")
    cfg = read_yaml(R / "experiment/config.yaml")
    mod.add_source_path(cfg)
    mod.set_seed(SEED)
    model, task, criterion, _ = mod.build_components(cfg, device)
    ck = Path(cfg["paths"]["output_root"]) / "checkpoints/step_3000.pt"
    st = torch.load(ck, map_location=device, weights_only=False)
    model.load_state_dict(st["r2p_and_projections"], strict=True)
    task.net_g.load_state_dict(st["tasknet_net_g"], strict=True)
    del st
    model.eval()
    task.eval()
    criterion.eval()
    loader = mod.split_loader(cfg, "validation")

    it = iter(loader)
    raw = next(it)
    batch = mod.unpack_batch(raw, device)
    mod.set_step_seed(SEED, "validation", 0, batch["sample_id"])
    polar = batch["polar"]
    cond = batch["condition"]
    rgb = batch["official_rgb"]
    target = batch["target"]
    loss_mod = sys.modules["polarapp.losses"]
    perceptual = criterion.perceptual

    timings = {}
    with torch.no_grad():
        # R2P round1
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = model(polar, cond, project_features=False)
        torch.cuda.synchronize()
        timings["r2p_round_s"] = time.perf_counter() - t
        p1 = out.refined_polar.detach().clone()
        del out
        torch.cuda.empty_cache()

        # TaskNet
        torch.cuda.synchronize()
        t = time.perf_counter()
        pred, _ = task(p1, rgb, None)
        torch.cuda.synchronize()
        timings["tasknet_s"] = time.perf_counter() - t
        pred = pred.detach().clone()
        torch.cuda.empty_cache()

        # loss components
        torch.cuda.synchronize()
        t = time.perf_counter()
        _ = float(F.l1_loss(pred, target))
        torch.cuda.synchronize()
        timings["loss_l1_s"] = time.perf_counter() - t

        torch.cuda.synchronize()
        t = time.perf_counter()
        _ = float(1.0 - loss_mod.ssim(pred, target))
        torch.cuda.synchronize()
        timings["loss_ssim_s"] = time.perf_counter() - t

        torch.cuda.synchronize()
        t = time.perf_counter()
        _ = float(criterion.phase(pred, target))
        torch.cuda.synchronize()
        timings["loss_phase_s"] = time.perf_counter() - t

        torch.cuda.synchronize()
        t = time.perf_counter()
        tgt_f = perceptual.extractor(target.detach())["conv5_4"]
        torch.cuda.synchronize()
        timings["perceptual_target_only_s"] = time.perf_counter() - t

        torch.cuda.synchronize()
        t = time.perf_counter()
        pf = perceptual.extractor(pred)["conv5_4"]
        _ = float(perceptual.weight * F.l1_loss(pf, tgt_f))
        torch.cuda.synchronize()
        timings["perceptual_pred_s"] = time.perf_counter() - t

    print(json.dumps(timings, indent=2))
    per_output_loss = (timings["loss_l1_s"] + timings["loss_ssim_s"]
                       + timings["loss_phase_s"] + timings["perceptual_pred_s"])
    print()
    print("per TaskNet output: tasknet=%.3f s + loss=%.3f s = %.3f s"
          % (timings["tasknet_s"], per_output_loss, timings["tasknet_s"] + per_output_loss))
    for n_calls in (5, 7):
        per_image = n_calls * (timings["tasknet_s"] + per_output_loss) + 2 * timings["r2p_round_s"]
        print("  %d TaskNet calls/image -> %.2f s/image -> 872 imgs = %.1f h/group"
              % (n_calls, per_image, per_image * 872 / 3600.0))
        per_image_no_perc = n_calls * (timings["tasknet_s"]
                                       + timings["loss_l1_s"] + timings["loss_ssim_s"]
                                       + timings["loss_phase_s"]) + 2 * timings["r2p_round_s"]
        print("     without perceptual: %.2f s/image -> 872 imgs = %.1f h/group"
              % (per_image_no_perc, per_image_no_perc * 872 / 3600.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
