#!/usr/bin/env python3
"""Endpoint check for the K=2 two-round forward (seed2026, step3000).

REVISED DEFINITION (per user instruction 2026-09-30):
  The interpolation is applied AFTER the second round's R2P output:

      P2(alpha) = P1 + alpha * (P2 - P1)

  where P2 is the second-round R2P output produced under the group's ORIGINAL
  configuration (feedback group: round 2 consumes the feedback features f1
  produced by T1; no_feedback group: original config, feedback disabled).

  The interpolated image is fed DIRECTLY to the TaskNet. It must NOT pass
  through R2P again. Turning feedback off is NOT alpha=0.

Two endpoints only (no 0.25/0.5 sweep in this run):
  alpha=0: feed P1 directly to TaskNet; check whether the output reproduces T1.
  alpha=1: feed the original P2 directly to TaskNet; check whether the output
           reproduces the original T2.

RNG discipline (mandatory):
  Before EACH comparison TaskNet call, restore the CPU and ALL CUDA RNG states
  that were saved immediately BEFORE the first TaskNet call of that image.
  This guarantees the endpoint call sees the identical diffusion noise stream
  as the reference T1/T2 call.

Reference integrity:
  TaskNet weights, eval mode, official RGB and all other inputs are unchanged.
  Reference tensors (P1, P2, T1, T2) are cloned and never modified in place.

Subset: the SAME existing 24 fixed images, SAME order. NOTE for honest reporting:
  this subset is 21 easy + 3 hard, 8 scenes -- it is NOT a subset stratified
  simultaneously by scene AND difficulty. Recorded as such in the output.

Tolerance: endpoint input difference must be EXACTLY 0 (identical reference
  tensor). Output tolerance 1e-4, recorded explicitly. `passed` is COMPUTED
  from the actual measured differences; it is never hard-written as True.

Retained (renamed) check:
  The previous "alpha=0 self-check" is kept but renamed to
  "feedback-off ablation at the same checkpoint" -- it is a different thing
  from alpha=0.

Read-only module loading, mirroring evaluation_872/val_driver.py.
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
PER_SCENE = 3
TOL_OUTPUT = 1e-4      # output absolute-difference tolerance (recorded explicitly)
TOL_INPUT = 0.0        # endpoint input must be EXACTLY the reference tensor


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
    return mod, model, task, loader, cfg, ckpt


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
        picked.append(index)
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
        "purpose": "alpha=0 / alpha=1 endpoint check of the K=2 two-round forward",
        "definition": ("P2(alpha) = P1 + alpha*(P2-P1); the interpolated image is fed "
                       "DIRECTLY to the TaskNet and must NOT pass through R2P again. "
                       "P2 is the second-round R2P output under the group's original "
                       "configuration. Feedback-off is NOT alpha=0."),
        "scope": "alpha=0 and alpha=1 only; no 0.25/0.5 sweep; model frozen; no training",
        "seed": SEED,
        "checkpoint_step": 3000,
        "subset": {
            "per_scene": PER_SCENE,
            "selection": "first PER_SCENE images per scene, loader order",
            "n_images": 24,
            "scenes": ["01", "07", "22", "30", "36", "41", "55", "56"],
            "difficulty_composition": "21 easy + 3 hard",
            "honesty_note": ("this subset is NOT stratified simultaneously by scene and "
                             "difficulty; it is scene-stratified and happens to contain "
                             "21 easy + 3 hard"),
        },
        "tolerances": {
            "endpoint_input_abs_diff": TOL_INPUT,
            "output_abs_diff": TOL_OUTPUT,
            "note": "endpoint input must be EXACTLY identical to the reference tensor",
        },
        "rng_discipline": ("before each comparison TaskNet call, the CPU and all CUDA RNG "
                           "states saved immediately before the FIRST TaskNet call are "
                           "restored; the reference T1/T2 call and the endpoint call see "
                           "the identical diffusion noise stream"),
        "groups": {},
    }

    for group in ("no_feedback", "feedback"):
        mod, model, task, loader, cfg, ckpt = build_group(group, device)
        model.eval()
        task.eval()
        subset = select_subset(loader, mod, device)
        wanted = set(int(i) for i in subset)

        rows = []
        t_start = time.perf_counter()
        with torch.no_grad():
            for index, raw in enumerate(loader):
                if index not in wanted:
                    continue
                batch = mod.unpack_batch(raw, device)
                mod.set_step_seed(SEED, "validation", index, batch["sample_id"])
                scene = batch["scene"][0]
                sample_id = batch["sample_id"][0]
                official_rgb = batch["official_rgb"]

                # ---- round 1 (formal pipeline): P1 then T1, with original config ----
                first = model(batch["polar"], batch["condition"], project_features=False)
                p1 = first.refined_polar.detach().clone()
                del first
                torch.cuda.empty_cache()

                # RNG state saved immediately BEFORE the first TaskNet call.
                cpu_rng = torch.get_rng_state()
                cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                devices = list(range(torch.cuda.device_count())) if cuda_rng is not None else []

                t1_ref, f1 = task(p1, official_rgb, None)
                t1 = t1_ref.detach().clone()
                del t1_ref
                psnr_t1 = psnr(t1, batch["target"])
                f1d = {k: v.detach().clone() for k, v in f1.items()} if model.feedback_enabled else None
                del f1
                torch.cuda.empty_cache()

                # ---- round 2 (formal pipeline): P2 then T2, original config ----
                second = model(p1, batch["condition"],
                               feedback=f1d if model.feedback_enabled else None,
                               project_features=False)
                p2 = second.refined_polar.detach().clone()
                del second
                torch.cuda.empty_cache()
                with torch.random.fork_rng(devices=devices):
                    torch.set_rng_state(cpu_rng)
                    if cuda_rng is not None:
                        torch.cuda.set_rng_state_all(cuda_rng)
                    t2_ref = task(p2, official_rgb, None)[0]
                t2 = t2_ref.detach().clone()
                del t2_ref
                psnr_t2 = psnr(t2, batch["target"])
                torch.cuda.empty_cache()

                # ---- endpoint alpha=0: feed P1 directly to TaskNet ----
                with torch.random.fork_rng(devices=devices):
                    torch.set_rng_state(cpu_rng)
                    if cuda_rng is not None:
                        torch.cuda.set_rng_state_all(cuda_rng)
                    ep0_in = p1.clone()
                    ep0_out = task(ep0_in, official_rgb, None)[0]

                # ---- endpoint alpha=1: feed original P2 directly to TaskNet ----
                with torch.random.fork_rng(devices=devices):
                    torch.set_rng_state(cpu_rng)
                    if cuda_rng is not None:
                        torch.cuda.set_rng_state_all(cuda_rng)
                    ep1_in = p2.clone()
                    ep1_out = task(ep1_in, official_rgb, None)[0]

                # measured differences
                in0_diff = float((ep0_in - p1).abs().max())
                in1_diff = float((ep1_in - p2).abs().max())
                out0_max = float((ep0_out - t1).abs().max())
                out0_mean = float((ep0_out - t1).abs().mean())
                out1_max = float((ep1_out - t2).abs().max())
                out1_mean = float((ep1_out - t2).abs().mean())

                # pass status COMPUTED, never hard-written
                pass0 = bool(in0_diff == 0.0 and out0_max < TOL_OUTPUT)
                pass1 = bool(in1_diff == 0.0 and out1_max < TOL_OUTPUT)

                rows.append({
                    "index": index,
                    "sample_id": sample_id,
                    "scene": scene,
                    "difficulty": sample_id.split("/")[1],
                    "alpha0": {
                        "endpoint_input": "P1",
                        "input_max_abs_diff_vs_P1": in0_diff,
                        "output_max_abs_diff_vs_T1": out0_max,
                        "output_mean_abs_diff_vs_T1": out0_mean,
                        "tolerance": TOL_OUTPUT,
                        "passed": pass0,
                    },
                    "alpha1": {
                        "endpoint_input": "P2",
                        "input_max_abs_diff_vs_P2": in1_diff,
                        "output_max_abs_diff_vs_T2": out1_max,
                        "output_mean_abs_diff_vs_T2": out1_mean,
                        "tolerance": TOL_OUTPUT,
                        "passed": pass1,
                    },
                    "psnr": {"t1": psnr_t1, "t2": psnr_t2},
                })

                del p1, p2, t1, t2, ep0_in, ep0_out, ep1_in, ep1_out, f1d
                torch.cuda.empty_cache()
                if len(rows) % 6 == 0:
                    print("%s: %d/%d done" % (group, len(rows), len(subset)), flush=True)

        a0 = np.array([r["alpha0"]["output_max_abs_diff_vs_T1"] for r in rows])
        a1 = np.array([r["alpha1"]["output_max_abs_diff_vs_T2"] for r in rows])
        i0 = np.array([r["alpha0"]["input_max_abs_diff_vs_P1"] for r in rows])
        i1 = np.array([r["alpha1"]["input_max_abs_diff_vs_P2"] for r in rows])

        result["groups"][group] = {
            "checkpoint": str(ckpt),
            "feedback_enabled": bool(model.feedback_enabled),
            "feedback_scale": float(model.feedback_scale.detach()),
            "n_images": len(rows),
            "alpha0": {
                "endpoint_input": "P1 (round1 R2P output)",
                "input_max_abs_diff_max_over_images": float(i0.max()),
                "input_all_exact": bool((i0 == 0.0).all()),
                "output_max_abs_diff_max_over_images": float(a0.max()),
                "output_mean_abs_diff_over_images": float(a0.mean()),
                "n_passed": int(sum(r["alpha0"]["passed"] for r in rows)),
                "all_passed": bool(all(r["alpha0"]["passed"] for r in rows)),
            },
            "alpha1": {
                "endpoint_input": "P2 (round2 R2P output, original config)",
                "input_max_abs_diff_max_over_images": float(i1.max()),
                "input_all_exact": bool((i1 == 0.0).all()),
                "output_max_abs_diff_max_over_images": float(a1.max()),
                "output_mean_abs_diff_over_images": float(a1.mean()),
                "n_passed": int(sum(r["alpha1"]["passed"] for r in rows)),
                "all_passed": bool(all(r["alpha1"]["passed"] for r in rows)),
            },
            "per_image": rows,
        }
        print("[%s] fs=%.8g  alpha0: in_maxdiff=%.3e out_maxdiff=%.3e pass=%d/%d | "
              "alpha1: in_maxdiff=%.3e out_maxdiff=%.3e pass=%d/%d"
              % (group, float(model.feedback_scale.detach()), i0.max(), a0.max(),
                 result["groups"][group]["alpha0"]["n_passed"], len(rows),
                 i1.max(), a1.max(),
                 result["groups"][group]["alpha1"]["n_passed"], len(rows)), flush=True)
        del model, task
        torch.cuda.empty_cache()

    result["elapsed_seconds"] = float(time.perf_counter() - t_start)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
