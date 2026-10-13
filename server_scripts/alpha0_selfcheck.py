#!/usr/bin/env python3
"""alpha=0 self-consistency check for the K=2 two-round forward (seed2026, step3000).

Purpose (from analysis/validation_872_review_20260930.md, "建议的最小下一步"):
  Freeze the trained model and verify that, when the second round's input falls back
  to P1 under the SAME paired diffusion noise stream, the second-round output
  reproduces T1. This is a CONSISTENCY CHECK of the implementation, not a new
  validation metric and not a model selection.

Scope of this run (user decision): alpha = 0 ONLY.
  Explore alpha in {0.25, 0.5, 1} is NOT run here.

Design notes that drive the code (must not be "improved" away):
  * recurrent_forward() in experiment/run_experiment.py saves the CPU/CUDA RNG state
    BEFORE the first task() call and restores it inside torch.random.fork_rng()
    BEFORE the second task() call. Therefore T1 and T2 already share ONE identical
    diffusion noise stream. We replicate that exact save/restore logic below so the
    check tests the real mechanism, not a re-implementation.

  * model.forward(polar, condition, feedback=...) has TWO structurally different
    branches:
        feedback is not None -> re-encodes condition, gate-fuses, AND adds
                                self.feedback_scale * fb to the fused features.
        feedback is None     -> plain r2p path.
    Measurements on the step_3000 checkpoints:
        feedback     group: feedback_scale = 0.0207351  (non-zero)
        no_feedback  group: feedback_scale = 0.0        (exactly zero)
    Consequence:
        - For no_feedback, round2 already takes the feedback=None branch, so the
          check is literal: P2 must equal P1 elementwise and T2 must reproduce T1.
        - For feedback, round2 genuinely injects 0.0207*fb, so P2 != P1 BY DESIGN.
          The alpha=0 semantics here means: force the second round onto the
          feedback=None branch, then P2 must equal P1 and T2 must reproduce T1.

Three sub-checks per group:
  A  round2 with the group's ORIGINAL feedback configuration
     -> report max|P2-P1| and max|T2-T1|   (the real configuration)
  B  round2 forced to feedback=None (alpha=0 semantics)
     -> report max|P2_off-P1| and max|T2_off-T1|  (the consistency assertion)
  For no_feedback, A and B are identical (feedback already disabled); we still
  run both to show the invariance explicitly.

Verdict rule (per group, applied to check B):
  PASS  if max|P2_off - P1| < tol_polar AND max|T2_off - T1| < tol_t
  Else FAIL. The report states the observed values either way; a FAIL is
  reported as-is (no retry, no rerun, no hiding).

This script modifies NOTHING on disk except writing its own JSON output.
It mirrors evaluation_872/val_driver.py read-only module loading.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

R = Path("/media/li/data3/25thstudent/DY/929work/pidsr_k2_929")
SRC = R / "deps/rgb_final_confirmation/src"
SEED = 2026
# Fixed, scene-stratified subset: first PER_SCENE images of each scene in loader order.
PER_SCENE = 3
TOL_POLAR = 1e-5
TOL_T = 1e-4


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_group(group: str, device: torch.device):
    """Return (module, model, task, loader, cfg) for the requested group."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    if group == "k1":
        mod = load_module("k1_mod", R / "baseline/k1_run_experiment_original.py")
        cfg = read_yaml(R / "experiment/config.yaml")
        cfg["paths"]["output_root"] = str(R / "runs/k1_validation_seed2026")
        mod.add_source_path(cfg)
        mod.set_seed(SEED)
        model, task, criterion, _base = mod.build_components(cfg, device)
        del criterion
        ckpt = R / "baseline/seed_2026/checkpoints/step_3000.pt"
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["r2p_and_projections"], strict=True)
        task.net_g.load_state_dict(state["tasknet_net_g"], strict=True)
        del state
        loader = mod.split_loader(cfg, "validation")
        return mod, model, task, loader, cfg, ckpt, "single_round"

    mod = load_module("re_mod_" + group, R / "experiment/run_experiment.py")
    cfg_path = R / ("experiment/config.yaml" if group == "feedback"
                    else "experiment/config_no_feedback.yaml")
    cfg = read_yaml(cfg_path)
    mod.add_source_path(cfg)
    mod.set_seed(SEED)
    model, task, criterion, _base = mod.build_components(cfg, device)
    del criterion
    ckpt = Path(cfg["paths"]["output_root"]) / "checkpoints/step_3000.pt"
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["r2p_and_projections"], strict=True)
    task.net_g.load_state_dict(state["tasknet_net_g"], strict=True)
    del state
    loader = mod.split_loader(cfg, "validation")
    return mod, model, task, loader, cfg, ckpt, "two_round"


def select_subset(loader, mod, device):
    """Scene-stratified, deterministic: first PER_SCENE images of each scene."""
    counts: dict[str, int] = {}
    picked = []
    for index, raw in enumerate(loader):
        batch = mod.unpack_batch(raw, device)
        scene = batch["scene"][0]
        if counts.get(scene, 0) >= PER_SCENE:
            continue
        counts[scene] = counts.get(scene, 0) + 1
        index_in_loader = index
        picked.append((index_in_loader, scene, batch["sample_id"][0]))
        if len(counts) == 8 and all(v >= PER_SCENE for v in counts.values()):
            break
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)

    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from rgb_final_confirmation.evaluation import psnr, ssim

    result: dict = {
        "purpose": "alpha=0 self-consistency check of the K=2 two-round forward",
        "scope": "alpha=0 only; the trained step_3000 model is frozen; no training",
        "seed": SEED,
        "subset": {"per_scene": PER_SCENE, "selection": "first PER_SCENE images per scene, loader order"},
        "tolerances": {"polar": TOL_POLAR, "t2_vs_t1": TOL_T},
        "groups": {},
    }

    for group in ("no_feedback", "feedback"):
        mod, model, task, loader, cfg, ckpt, mode = build_group(group, device)
        model.eval()
        task.eval()
        subset = select_subset(loader, mod, device)
        fs = float(model.feedback_scale.detach())

        rows = []
        t_start = time.perf_counter()
        # Iterate loader once, processing only the selected indices.
        wanted = {int(i): (s, sid) for i, s, sid in subset}
        with torch.no_grad():
            for index, raw in enumerate(loader):
                if index not in wanted:
                    continue
                batch = mod.unpack_batch(raw, device)
                mod.set_step_seed(SEED, "validation", index, batch["sample_id"])
                scene = batch["scene"][0]
                sample_id = batch["sample_id"][0]

                # --- replicate recurrent_forward's RNG save/restore exactly ---
                # Memory discipline: only ONE extra computation graph may be alive at
                # a time. Everything is reduced to python floats via .item() as soon
                # as it is computed, then the tensors are freed. The 11.75 GiB card
                # cannot hold three simultaneous K=2 graphs.
                first = model(batch["polar"], batch["condition"], project_features=False)
                p1 = first.refined_polar.detach().clone()   # small: kept across checks
                cpu_rng = torch.get_rng_state()
                cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                devices = list(range(torch.cuda.device_count())) if cuda_rng is not None else []

                t1, f1 = task(p1, batch["official_rgb"], None)
                t1d = t1.detach().clone()
                del t1
                psnr_t1 = psnr(t1d, batch["target"])
                f1d = {k: v.detach().clone() for k, v in f1.items()} if model.feedback_enabled else None
                del f1
                torch.cuda.empty_cache()

                # ---- Check A: round2 with this group's ORIGINAL feedback config ----
                second_fb = model(p1, batch["condition"],
                                  feedback=f1d if model.feedback_enabled else None,
                                  project_features=False)
                p2 = second_fb.refined_polar.detach().clone()
                del second_fb
                torch.cuda.empty_cache()
                with torch.random.fork_rng(devices=devices):
                    torch.set_rng_state(cpu_rng)
                    if cuda_rng is not None:
                        torch.cuda.set_rng_state_all(cuda_rng)
                    t2 = task(p2, batch["official_rgb"], None)[0]
                d_p2_p1 = float((p2 - p1).abs().max())
                m_p2_p1 = float((p2 - p1).abs().mean())
                d_t2_t1 = float((t2 - t1d).abs().max())
                m_t2_t1 = float((t2 - t1d).abs().mean())
                psnr_t2 = psnr(t2, batch["target"])
                del p2, t2
                torch.cuda.empty_cache()

                # ---- Check B: alpha=0 semantics -> round2 forced to feedback=None ----
                with torch.random.fork_rng(devices=devices):
                    torch.set_rng_state(cpu_rng)
                    if cuda_rng is not None:
                        torch.cuda.set_rng_state_all(cuda_rng)
                    second_off = model(p1, batch["condition"],
                                       feedback=None, project_features=False)
                    p2off = second_off.refined_polar.detach().clone()
                    del second_off
                    t2_off = task(p2off, batch["official_rgb"], None)[0]
                d_p2off_p1 = float((p2off - p1).abs().max())
                m_p2off_p1 = float((p2off - p1).abs().mean())
                d_t2off_t1 = float((t2_off - t1d).abs().max())
                m_t2off_t1 = float((t2_off - t1d).abs().mean())
                psnr_t2off = psnr(t2_off, batch["target"])
                del p2off, t2_off, t1d, p1, f1d
                torch.cuda.empty_cache()

                rows.append({
                    "index": index,
                    "scene": scene,
                    "sample_id": sample_id,
                    "max_abs_P2_minus_P1": d_p2_p1,
                    "max_abs_P2off_minus_P1": d_p2off_p1,
                    "max_abs_T2_minus_T1": d_t2_t1,
                    "max_abs_T2off_minus_T1": d_t2off_t1,
                    "mean_abs_P2_minus_P1": m_p2_p1,
                    "mean_abs_P2off_minus_P1": m_p2off_p1,
                    "mean_abs_T2_minus_T1": m_t2_t1,
                    "mean_abs_T2off_minus_T1": m_t2off_t1,
                    "psnr_t1": psnr_t1,
                    "psnr_t2": psnr_t2,
                    "psnr_t2off": psnr_t2off,
                    "diffusion_rng_stream_shared": True,
                })
                if len(rows) % 6 == 0:
                    print("%s: %d/%d done" % (group, len(rows), len(subset)), flush=True)

        arr_p2off = np.array([r["max_abs_P2off_minus_P1"] for r in rows])
        arr_t2off = np.array([r["max_abs_T2off_minus_T1"] for r in rows])
        arr_p2 = np.array([r["max_abs_P2_minus_P1"] for r in rows])
        arr_t2 = np.array([r["max_abs_T2_minus_T1"] for r in rows])
        passed = bool((arr_p2off < TOL_POLAR).all() and (arr_t2off < TOL_T).all())

        result["groups"][group] = {
            "checkpoint": str(ckpt),
            "forward_mode": mode,
            "feedback_enabled": bool(model.feedback_enabled),
            "feedback_scale": fs,
            "n_images": len(rows),
            "scenes": sorted({r["scene"] for r in rows}),
            "checkA_original_feedback_config": {
                "max_abs_P2_minus_P1": float(arr_p2.max()),
                "max_abs_T2_minus_T1": float(arr_t2.max()),
                "note": ("round2 injects feedback_scale*fb, so P2!=P1 is expected by design"
                         if model.feedback_enabled else
                         "feedback disabled: round2 already equals the feedback=None path"),
            },
            "checkB_alpha0_forced_no_feedback": {
                "max_abs_P2off_minus_P1": float(arr_p2off.max()),
                "max_abs_T2off_minus_T1": float(arr_t2off.max()),
                "pass": passed,
                "interpretation": ("PASS: with the same paired diffusion noise stream and the second "
                                   "round forced back to P1 input, T2 reproduces T1 to float tolerance."
                                   if passed else
                                   "FAIL: T2 does not reproduce T1; the repeat-forward mechanism needs "
                                   "investigation before any alpha sweep is meaningful."),
            },
            "per_image": rows,
        }
        print("[%s] feedback_scale=%.8g  checkA max|P2-P1|=%.3e max|T2-T1|=%.3e  "
              "checkB max|P2off-P1|=%.3e max|T2off-T1|=%.3e  PASS=%s"
              % (group, fs, arr_p2.max(), arr_t2.max(), arr_p2off.max(), arr_t2off.max(), passed),
              flush=True)

    elapsed = time.perf_counter() - t_start
    result["elapsed_seconds"] = float(elapsed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
