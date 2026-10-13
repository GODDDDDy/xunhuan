#!/usr/bin/env python3
"""Priority-1 amplitude/target diagnosis on the EXISTING K=2 checkpoints.

No training. No weight update. Read-only.

For each group (feedback, no_feedback), seed 2026, step 3000:

  * P1 and P2 are computed normally, under the group's ORIGINAL second-round
    configuration (feedback group: round 2 consumes the T1 feedback features;
    no_feedback group: feedback stays off). P1/P2 are then FIXED.
  * The sweep is
        P(alpha) = P1 + alpha * (P2 - P1),   alpha in {0, .25, .5, .75, 1}
    The interpolated tensor is fed DIRECTLY to the TaskNet. It does NOT pass
    through R2P again. Turning feedback off is NOT alpha=0.
  * For every alpha we restore the CPU and ALL CUDA RNG states that were saved
    immediately before the FIRST TaskNet call of that image, so every alpha sees
    exactly the same diffusion noise stream as the reference T1 call.
  * We report, per alpha: PSNR and SSIM (image-mean and scene-macro), the delta
    against THIS MODEL's own T1, the official task loss and its four components
    (L1, 1-SSIM, phase, perceptual), and the improved/degraded image counts.

Endpoint integrity: alpha=0 must reproduce T1 and alpha=1 must reproduce T2.
For those two endpoints the input tensor is the CLONED reference (P1 / P2) so
the input difference is exactly 0; intermediate alphas use the formula.

Clamp bookkeeping: the R2P delta is produced as
    delta = max_residual * tanh(body(...))        # raw, pre-clamp
    refined = (polar + delta).clamp(0, 1)
so we record, per round, the raw delta, the EFFECTIVE update (refined - input)
and the clamp gap max|refined - (input + delta)|.

Subset: the SAME existing 24 fixed images, SAME order, SAME selection rule
(first 3 per scene in loader order). Honest composition: 21 easy + 3 hard;
scene-stratified but NOT stratified by scene AND difficulty simultaneously.

Writes a JSON result and does not touch anything else.
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
import torch.nn.functional as F
import yaml

R = Path("/media/li/data3/25thstudent/DY/929work/pidsr_k2_929")
SRC = R / "deps/rgb_final_confirmation/src"
SEED = 2026
PER_SCENE = 3
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
TOL_OUTPUT = 1e-4     # endpoint output absolute-difference tolerance
TOL_INPUT = 0.0       # endpoint input must be EXACTLY the reference tensor


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
    ckpt = Path(cfg["paths"]["output_root"]) / "checkpoints/step_3000.pt"
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["r2p_and_projections"], strict=True)
    task.net_g.load_state_dict(state["tasknet_net_g"], strict=True)
    del state
    loader = mod.split_loader(cfg, "validation")
    return mod, model, task, criterion, loader, cfg, ckpt


def select_subset(loader, mod, device):
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


def agg(values):
    return float(np.mean(values))


def scene_macro(values, scenes):
    vals = np.asarray(values, dtype=np.float64)
    arr = np.asarray(scenes)
    out = {}
    for s in sorted(set(arr.tolist())):
        out[s] = float(vals[arr == s].mean())
    return float(np.mean(list(out.values()))), out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--full", action="store_true",
                    help="run the whole validation split instead of the fixed 24-image "
                         "subset (priority-1 step 6: expansion under locked settings)")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many images (0 = no limit); for smoke tests")
    args = ap.parse_args()
    device = torch.device(args.device)

    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from rgb_final_confirmation.evaluation import psnr, ssim as eval_ssim

    result = {
        "purpose": ("Priority-1 amplitude/target diagnosis on the existing K=2 "
                    "checkpoints: alpha sweep of the second-round update"),
        "definition": ("P(alpha) = P1 + alpha*(P2-P1). P1/P2 are computed under the "
                       "group's ORIGINAL second-round configuration and then fixed. "
                       "The interpolated tensor is fed DIRECTLY to the TaskNet and "
                       "does NOT pass through R2P again. Feedback-off is NOT alpha=0."),
        "scope": ("frozen checkpoints; no training, no weight update; 24 fixed images; "
                  "diagnostic/exploratory only"),
        "seed": SEED,
        "checkpoint_step": 3000,
        "alphas": ALPHAS,
        "subset": (
            {
                "mode": "whole validation split, loader order, no selection",
                "honesty_note": ("expansion under locked settings (priority-1 step 6); "
                                 "the subset composition is whatever the split contains"),
            } if args.full else {
                "per_scene": PER_SCENE,
                "selection": "first PER_SCENE images per scene, loader order",
                "n_images": 24,
                "scenes": ["01", "07", "22", "30", "36", "41", "55", "56"],
                "difficulty_composition": "21 easy + 3 hard",
                "honesty_note": ("scene-stratified, NOT stratified simultaneously by "
                                 "scene and difficulty; composition recorded, not chosen"),
            }),
        "endpoint_aliasing": (
            "alpha=0 and alpha=1 are NOT re-run as separate TaskNet calls. They were "
            "proven bit-identical to T1 and T2 on the 24-image run (input and output "
            "max abs diff 0.000e+00), so their rows are aliases of the T1/T2 rows. "
            "This is recorded per row as aliased=true."
            if args.full else
            "alpha=0 and alpha=1 are run as separate TaskNet calls with the cloned "
            "reference tensor as input, so the endpoint check is direct, not aliased."),
        "rng_discipline": ("before each comparison TaskNet call (T2 and every alpha), "
                           "the CPU and all CUDA RNG states saved immediately before "
                           "the FIRST TaskNet call of that image are restored, so all "
                           "of them see the identical diffusion noise stream as T1"),
        "metric_module": "rgb_final_confirmation.evaluation.{psnr,ssim}",
        "loss_components": ("official DfPTaskLoss = L1 + (1-SSIM_loss) + phase + "
                            "perceptual; components computed from the criterion's own "
                            "submodules; the target's VGG features are cached once per "
                            "image, which is identical to the module's own "
                            "extractor(target.detach()) call"),
        "tolerances": {"endpoint_input_abs_diff": TOL_INPUT,
                       "endpoint_output_abs_diff": TOL_OUTPUT},
        "groups": {},
    }

    t_start = time.perf_counter()

    for group in ("no_feedback", "feedback"):
        mod, model, task, criterion, loader, cfg, ckpt = build_group(group, device)
        model.eval()
        task.eval()
        criterion.eval()

        # the loss module's own ssim (different implementation from the eval metric)
        loss_mod = sys.modules.get("polarapp.losses")
        if loss_mod is None:
            raise RuntimeError("polarapp.losses not in sys.modules after criterion build")
        loss_ssim = loss_mod.ssim
        perceptual = criterion.perceptual
        perc_weight = float(perceptual.weight) if perceptual is not None else None

        if args.full:
            subset = list(range(len(loader)))
            if args.limit:
                subset = subset[: args.limit]
        else:
            subset = select_subset(loader, mod, device)
        wanted = set(int(i) for i in subset)

        rows = []
        with torch.no_grad():
            for index, raw in enumerate(loader):
                if index not in wanted:
                    continue
                batch = mod.unpack_batch(raw, device)
                mod.set_step_seed(SEED, "validation", index, batch["sample_id"])
                scene = batch["scene"][0]
                sample_id = batch["sample_id"][0]
                official_rgb = batch["official_rgb"]
                target = batch["target"]
                p0 = batch["polar"]

                # target VGG features cached once (identical to extractor(target.detach()))
                tgt_feat = (perceptual.extractor(target.detach())["conv5_4"]
                            if perceptual is not None else None)

                def loss_parts(pred):
                    l1 = float(F.l1_loss(pred, target))
                    s_t = float(1.0 - loss_ssim(pred, target))
                    ph = float(criterion.phase(pred, target))
                    if perceptual is not None:
                        pf = perceptual.extractor(pred)["conv5_4"]
                        pe = float(perc_weight * F.l1_loss(pf, tgt_feat))
                    else:
                        pe = None
                    total = l1 + s_t + ph + (pe if pe is not None else 0.0)
                    return {"total": total, "l1": l1, "ssim_term": s_t,
                            "phase": ph, "perceptual": pe}

                # ---- round 1 --------------------------------------------------
                first = model(p0, batch["condition"], project_features=False)
                p1 = first.refined_polar.detach().clone()
                d1_raw = first.delta.detach().clone()
                round1 = {
                    "raw_delta_abs_mean": float(d1_raw.abs().mean()),
                    "raw_delta_abs_max": float(d1_raw.abs().max()),
                    "effective_update_abs_mean": float((p1 - p0).abs().mean()),
                    "effective_update_abs_max": float((p1 - p0).abs().max()),
                    "clamp_gap_abs_max": float((p1 - (p0 + d1_raw)).abs().max()),
                    "clamp_gap_abs_mean": float((p1 - (p0 + d1_raw)).abs().mean()),
                }
                del first, d1_raw
                torch.cuda.empty_cache()

                # RNG state saved immediately BEFORE the first TaskNet call.
                cpu_rng = torch.get_rng_state()
                cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                devices = list(range(torch.cuda.device_count())) if cuda_rng is not None else []

                t1_out, f1 = task(p1, official_rgb, None)
                t1 = t1_out.detach().clone()
                del t1_out
                psnr_t1 = float(psnr(t1, target))
                ssim_t1 = float(eval_ssim(t1, target))
                loss_t1 = loss_parts(t1)
                f1d = ({k: v.detach().clone() for k, v in f1.items()}
                       if model.feedback_enabled else None)
                del f1
                torch.cuda.empty_cache()

                # ---- round 2 (original config) --------------------------------
                second = model(p1, batch["condition"],
                               feedback=f1d if model.feedback_enabled else None,
                               project_features=False)
                p2 = second.refined_polar.detach().clone()
                d2_raw = second.delta.detach().clone()
                round2 = {
                    "raw_delta_abs_mean": float(d2_raw.abs().mean()),
                    "raw_delta_abs_max": float(d2_raw.abs().max()),
                    "effective_update_abs_mean": float((p2 - p1).abs().mean()),
                    "effective_update_abs_max": float((p2 - p1).abs().max()),
                    "clamp_gap_abs_max": float((p2 - (p1 + d2_raw)).abs().max()),
                    "clamp_gap_abs_mean": float((p2 - (p1 + d2_raw)).abs().mean()),
                }
                del second, d2_raw
                torch.cuda.empty_cache()

                with torch.random.fork_rng(devices=devices):
                    torch.set_rng_state(cpu_rng)
                    if cuda_rng is not None:
                        torch.cuda.set_rng_state_all(cuda_rng)
                    t2_out = task(p2, official_rgb, None)[0]
                t2 = t2_out.detach().clone()
                del t2_out
                psnr_t2 = float(psnr(t2, target))
                ssim_t2 = float(eval_ssim(t2, target))
                loss_t2 = loss_parts(t2)
                torch.cuda.empty_cache()

                # ---- alpha sweep ----------------------------------------------
                alphas = {}
                for a in ALPHAS:
                    if args.full and a in (0.0, 1.0):
                        # Proven bit-identical to T1 / T2 on the 24-image run.
                        src_t = t1 if a == 0.0 else t2
                        p_a = psnr_t1 if a == 0.0 else psnr_t2
                        s_a = ssim_t1 if a == 0.0 else ssim_t2
                        l_a = dict(loss_t1) if a == 0.0 else dict(loss_t2)
                        src_in = p1 if a == 0.0 else p2
                        alphas[str(a)] = {
                            "aliased": True,
                            "alias_of": "T1" if a == 0.0 else "T2",
                            "input_source": "P1" if a == 0.0 else "P2",
                            "input_max_abs_diff_vs_reference": 0.0,
                            "output_max_abs_diff_vs_reference_T": 0.0,
                            "input_min": float(src_in.min()),
                            "input_max": float(src_in.max()),
                            "psnr": p_a,
                            "ssim": s_a,
                            "delta_psnr_vs_T1": p_a - psnr_t1,
                            "delta_ssim_vs_T1": s_a - ssim_t1,
                            "loss": l_a,
                            "delta_loss_total_vs_T1": l_a["total"] - loss_t1["total"],
                        }
                        continue
                    if a == 0.0:
                        pa = p1.clone()
                        ref_in, ref_name = p1, "P1"
                    elif a == 1.0:
                        pa = p2.clone()
                        ref_in, ref_name = p2, "P2"
                    else:
                        pa = p1 + a * (p2 - p1)
                        ref_in, ref_name = None, None
                    in_diff = (float((pa - ref_in).abs().max())
                               if ref_in is not None else None)

                    with torch.random.fork_rng(devices=devices):
                        torch.set_rng_state(cpu_rng)
                        if cuda_rng is not None:
                            torch.cuda.set_rng_state_all(cuda_rng)
                        out = task(pa, official_rgb, None)[0]
                    p_a = float(psnr(out, target))
                    s_a = float(eval_ssim(out, target))
                    l_a = loss_parts(out)

                    # endpoint integrity against the reference outputs
                    if a == 0.0:
                        ref_out = t1
                        out_diff = float((out - ref_out).abs().max())
                    elif a == 1.0:
                        ref_out = t2
                        out_diff = float((out - ref_out).abs().max())
                    else:
                        out_diff = None

                    alphas[str(a)] = {
                        "input_source": ref_name if ref_name else "P1 + alpha*(P2-P1)",
                        "input_max_abs_diff_vs_reference": in_diff,
                        "output_max_abs_diff_vs_reference_T": out_diff,
                        "input_min": float(pa.min()),
                        "input_max": float(pa.max()),
                        "psnr": p_a,
                        "ssim": s_a,
                        "delta_psnr_vs_T1": p_a - psnr_t1,
                        "delta_ssim_vs_T1": s_a - ssim_t1,
                        "loss": l_a,
                        "delta_loss_total_vs_T1": l_a["total"] - loss_t1["total"],
                    }
                    del out, pa
                    torch.cuda.empty_cache()

                rows.append({
                    "index": index,
                    "sample_id": sample_id,
                    "scene": scene,
                    "difficulty": sample_id.split("/")[1],
                    "round1_clamp": round1,
                    "round2_clamp": round2,
                    "T1": {"psnr": psnr_t1, "ssim": ssim_t1, "loss": loss_t1},
                    "T2": {"psnr": psnr_t2, "ssim": ssim_t2, "loss": loss_t2,
                           "delta_psnr_vs_T1": psnr_t2 - psnr_t1,
                           "delta_ssim_vs_T1": ssim_t2 - ssim_t1,
                           "delta_loss_total_vs_T1": loss_t2["total"] - loss_t1["total"]},
                    "alphas": alphas,
                })

                del p1, p2, t1, t2, f1d, tgt_feat
                torch.cuda.empty_cache()
                step = 100 if args.full else 4
                if len(rows) % step == 0:
                    print("%s: %d/%d done (%.1f s)"
                          % (group, len(rows), len(subset), time.perf_counter() - t_start),
                          flush=True)

        # ---- aggregation ---------------------------------------------------
        scenes = [r["scene"] for r in rows]
        def col(fn):
            return [fn(r) for r in rows]

        def summarize(vals):
            m, per_scene = scene_macro(vals, scenes)
            return {"image_mean": agg(vals), "scene_macro": m, "per_scene": per_scene}

        summary = {
            "checkpoint": str(ckpt),
            "feedback_enabled": bool(model.feedback_enabled),
            "feedback_scale": float(model.feedback_scale.detach()),
            "n_images": len(rows),
            "perceptual_weight": perc_weight,
            "clamp_round1": {
                "raw_delta_abs_mean_mean_over_images":
                    agg(col(lambda r: r["round1_clamp"]["raw_delta_abs_mean"])),
                "effective_update_abs_mean_mean_over_images":
                    agg(col(lambda r: r["round1_clamp"]["effective_update_abs_mean"])),
                "clamp_gap_abs_max_over_images":
                    max(col(lambda r: r["round1_clamp"]["clamp_gap_abs_max"])),
            },
            "clamp_round2": {
                "raw_delta_abs_mean_mean_over_images":
                    agg(col(lambda r: r["round2_clamp"]["raw_delta_abs_mean"])),
                "effective_update_abs_mean_mean_over_images":
                    agg(col(lambda r: r["round2_clamp"]["effective_update_abs_mean"])),
                "clamp_gap_abs_max_over_images":
                    max(col(lambda r: r["round2_clamp"]["clamp_gap_abs_max"])),
            },
            "T1": {
                "psnr": summarize(col(lambda r: r["T1"]["psnr"])),
                "ssim": summarize(col(lambda r: r["T1"]["ssim"])),
                "loss_total_image_mean":
                    agg(col(lambda r: r["T1"]["loss"]["total"])),
                "loss_components_image_mean": {
                    k: agg(col(lambda r: r["T1"]["loss"][k])) if
                    all(r["T1"]["loss"][k] is not None for r in rows) else None
                    for k in ("l1", "ssim_term", "phase", "perceptual")
                },
            },
            "T2": {
                "psnr": summarize(col(lambda r: r["T2"]["psnr"])),
                "ssim": summarize(col(lambda r: r["T2"]["ssim"])),
                "delta_psnr_vs_T1": summarize(col(lambda r: r["T2"]["delta_psnr_vs_T1"])),
                "delta_ssim_vs_T1": summarize(col(lambda r: r["T2"]["delta_ssim_vs_T1"])),
                "loss_total_image_mean":
                    agg(col(lambda r: r["T2"]["loss"]["total"])),
                "delta_loss_total_vs_T1_image_mean":
                    agg(col(lambda r: r["T2"]["delta_loss_total_vs_T1"])),
            },
            "alphas": {},
            "endpoint_integrity": {},
        }

        for a in ALPHAS:
            key = str(a)
            d_psnr = col(lambda r: r["alphas"][key]["delta_psnr_vs_T1"])
            d_ssim = col(lambda r: r["alphas"][key]["delta_ssim_vs_T1"])
            n_imp = sum(1 for v in d_psnr if v > 0.0)
            n_deg = sum(1 for v in d_psnr if v < 0.0)
            n_tie = sum(1 for v in d_psnr if v == 0.0)
            summary["alphas"][key] = {
                "psnr": summarize(col(lambda r: r["alphas"][key]["psnr"])),
                "ssim": summarize(col(lambda r: r["alphas"][key]["ssim"])),
                "delta_psnr_vs_T1": summarize(d_psnr),
                "delta_ssim_vs_T1": summarize(d_ssim),
                "loss_total_image_mean":
                    agg(col(lambda r: r["alphas"][key]["loss"]["total"])),
                "loss_components_image_mean": {
                    k: agg(col(lambda r: r["alphas"][key]["loss"][k])) if
                    all(r["alphas"][key]["loss"][k] is not None for r in rows) else None
                    for k in ("l1", "ssim_term", "phase", "perceptual")
                },
                "delta_loss_total_vs_T1_image_mean":
                    agg(col(lambda r: r["alphas"][key]["delta_loss_total_vs_T1"])),
                "image_counts_vs_T1": {
                    "improved": n_imp, "degraded": n_deg, "unchanged": n_tie,
                    "total": len(rows),
                    "improved_fraction": n_imp / len(rows),
                    "degraded_fraction": n_deg / len(rows),
                },
                "input_range_over_images": {
                    "min": min(col(lambda r: r["alphas"][key]["input_min"])),
                    "max": max(col(lambda r: r["alphas"][key]["input_max"])),
                },
            }

        for a, ref in ((0.0, "T1"), (1.0, "T2")):
            key = str(a)
            in_d = col(lambda r: r["alphas"][key]["input_max_abs_diff_vs_reference"])
            out_d = col(lambda r: r["alphas"][key]["output_max_abs_diff_vs_reference_T"])
            summary["endpoint_integrity"][key] = {
                "reference": ref,
                "input_all_exact": bool(all(v == 0.0 for v in in_d)),
                "input_max_abs_diff_max_over_images": float(max(in_d)),
                "output_max_abs_diff_max_over_images": float(max(out_d)),
                "output_within_tolerance": bool(max(out_d) < TOL_OUTPUT),
                "tolerance": TOL_OUTPUT,
            }

        summary["per_image"] = rows
        result["groups"][group] = summary

        print("[%s] fs=%.8g  T1=%.4f  T2=%.4f (d=%+.4f)  "
              "a0=%+.4f a25=%+.4f a50=%+.4f a75=%+.4f a100=%+.4f (dPSNR vs T1, image mean)"
              % (group, float(model.feedback_scale.detach()),
                 summary["T1"]["psnr"]["image_mean"],
                 summary["T2"]["psnr"]["image_mean"],
                 summary["T2"]["delta_psnr_vs_T1"]["image_mean"],
                 summary["alphas"]["0.0"]["delta_psnr_vs_T1"]["image_mean"],
                 summary["alphas"]["0.25"]["delta_psnr_vs_T1"]["image_mean"],
                 summary["alphas"]["0.5"]["delta_psnr_vs_T1"]["image_mean"],
                 summary["alphas"]["0.75"]["delta_psnr_vs_T1"]["image_mean"],
                 summary["alphas"]["1.0"]["delta_psnr_vs_T1"]["image_mean"]),
              flush=True)
        print("[%s] endpoint: a0 out_maxdiff=%.3e (vs T1)  a1 out_maxdiff=%.3e (vs T2)"
              % (group,
                 summary["endpoint_integrity"]["0.0"]["output_max_abs_diff_max_over_images"],
                 summary["endpoint_integrity"]["1.0"]["output_max_abs_diff_max_over_images"]),
              flush=True)

        del model, task, criterion
        torch.cuda.empty_cache()

    result["elapsed_seconds"] = float(time.perf_counter() - t_start)
    result["n_tasknet_calls"] = sum(
        len(g["per_image"]) * (len(ALPHAS) + 2 if not args.full else len(ALPHAS))
        for g in result["groups"].values())
    result["full_validation"] = bool(args.full)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("WROTE", args.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
