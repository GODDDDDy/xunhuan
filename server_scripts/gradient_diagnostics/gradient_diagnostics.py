#!/usr/bin/env python3
"""Review item #3 -- training gradient diagnostics on the FROZEN K=2 checkpoints.

READ-ONLY. No weight update. No `optimizer.step`. No training loop.

Question
--------
When the K=2 training objective
      total = 0.5 * L(T1) + 1.0 * L(T2) + lambda * alignment
is differentiated through the SHARED R2P parameters, do its separate terms push
those parameters in the same direction or in conflicting directions, and how
large is each term's contribution?

Structural facts respected here
-------------------------------
* R2P is used TWICE per training step (round 1 and round 2) and both uses flow
  into the same parameter tensors.  A plain `loss.backward()` accumulates both
  rounds into one `.grad`, so we separate them with independent autograd passes.
* L(T1) and L(T2) are each a SUM of four separable terms
      L1 + (1 - SSIM) + phase + perceptual_weight * L1(VGG conv5_4)
  so each term's gradient on the shared parameters is computable on its own.
* The official criterion is *re-derived and asserted* here (not assumed) to
  equal that recomposition on every batch, before any gradient is reported.

Two decompositions, both reported
---------------------------------
Training's `recurrent_forward` does NOT detach P1 before feeding it to round 2,
so in the real training graph the round-2 term also back-propagates into round 1
through P1.  We therefore run the forward TWICE, once per decomposition:

  (A) "real-path"   : the graph exactly as training builds it.  Round-2
                      gradients here include the through-P1 path.  This is the
                      decomposition whose SUM reproduces training's actual
                      update direction.
  (B) "isolated"    : round 2 is fed a DETACHED P1, so its gradient on the
                      shared parameters is attributable to round 2 alone.  This
                      is the decomposition needed to ask "does round 2 conflict
                      with round 1?".

Both are needed: (A) validates against training; (B) answers the question.
Running them as two separate forwards (rather than one retained graph) is what
keeps peak memory inside the 12 GiB card.

Gradient conventions
--------------------
For every quantity we record `unit` (the UNWEIGHTED gradient) and `weighted`
(the same vector times the weight it carries in the objective).  Pairwise angles
are computed on `unit` vectors, because a cosine is scale invariant and the
objective weights are pure rescalings -- but the weighted norms are reported
alongside so the actual scale of each contribution is visible.

NOTE ON GRADIENT CLIPPING
-------------------------
Training clips `model.parameters()` (R2P + projections + feedback adapter) to a
global norm of 1.0 AFTER backward, on the SUMMED gradient -- it rescales the
whole vector uniformly and cannot change any pairwise angle.  We record the
pre-clip norm and whether the clip would fire, but never clip here.

NOTE ON STE
-----------
The diffusion stream feeds the task input through a quantization preprocessing
that uses a straight-through estimator, so every gradient below is the
implementation's SURROGATE gradient, not a derivative of the frozen prior.

Writes one JSON.  Nothing else is touched.
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
STEP = 3000

# Predetermined training steps.  Chosen BEFORE looking at any gradient value:
# spread across the frozen schedule (early / mid / late), skipping the very
# first step (Adam warm-up) and the very last (checkpoint boundary).
PREDETERMINED_STEPS = [200, 600, 1200, 2000, 2800]

GROUPS = ("no_feedback", "feedback")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def flat(tensors, sizes):
    """Concatenate gradients into ONE float64 CPU vector, zeros for unused params.

    Entries that are None (parameter not touched by this term) become zeros of
    that parameter's size.  Skipping them instead would give different terms
    different lengths and every dot product would compare mis-aligned
    coordinates.  The cast happens on the CPU copy so the GPU never holds a
    second float64 copy of a large gradient.
    """
    parts = []
    for t, n in zip(tensors, sizes):
        if t is not None:
            parts.append(t.detach().reshape(-1).to("cpu", torch.float64))
        else:
            parts.append(torch.zeros(int(n), dtype=torch.float64))
    return torch.cat(parts)


def vnorm(v):
    return 0.0 if v is None else float(v.norm())


def cos(a, b):
    if a is None or b is None:
        return None
    na, nb = float(a.norm()), float(b.norm())
    if na == 0.0 or nb == 0.0:
        return None
    return float(torch.dot(a, b) / (na * nb))


def add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", default="")
    ap.add_argument("--groups", default=",".join(GROUPS))
    args = ap.parse_args()

    steps = ([int(x) for x in args.steps.split(",") if x.strip()]
             if args.steps else list(PREDETERMINED_STEPS))
    groups = [g for g in args.groups.split(",") if g.strip()]
    device = torch.device(args.device)
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    out = {
        "purpose": ("Review item #3: per-term training-gradient norms and pairwise "
                    "angles on the shared R2P parameters, plus round1-vs-round2 "
                    "task gradients. Frozen checkpoint; no optimizer.step; "
                    "read-only."),
        "read_only": True,
        "no_weight_update": True,
        "optimizer_step_called": False,
        "checkpoint_step": STEP,
        "seed": SEED,
        "predetermined_steps": steps,
        "step_selection_rationale": (
            "Chosen before inspecting any gradient value: spread over the frozen "
            "schedule (early/mid/late), step 200 through 2800; the very first and "
            "very last steps are avoided. Not selected by observed result."),
        "gradient_semantics": {
            "unit": "gradient of the UNWEIGHTED term",
            "weighted": "unit * the weight the term carries in the objective",
            "angles_computed_on": "unit vectors (cosine is scale invariant)",
            "clip_convention": (
                "training clips the SUMMED gradient over model.parameters() to "
                "global norm grad_clip=1.0 after backward; a uniform rescale that "
                "cannot change any pairwise angle. NOT applied here."),
            "ste_note": (
                "quantization preprocessing in the diffusion stream uses a "
                "straight-through estimator, so all vectors are the "
                "implementation's SURROGATE gradients"),
        },
        "objective": ("total = round_weights[0]*L(T1) + round_weights[1]*L(T2) + "
                      "lambda*alignment  with round_weights=[0.5, 1.0]"),
        "per_round_loss": ("L = L1 + (1-SSIM) + phase + perceptual_weight*L1(VGG conv5_4)"),
        "decompositions": {
            "real_path": ("single forward; round 2 consumes the graph-attached P1, "
                          "exactly as training builds it; round-2 gradients include "
                          "the through-P1 path. The SUM of its parts reproduces "
                          "training's actual update direction."),
            "isolated": ("round 2 consumes a DETACHED P1, so its gradient on the "
                         "shared parameters is attributable to round 2 alone. This "
                         "is the decomposition used to ask whether round 2 conflicts "
                         "with round 1."),
        },
        "groups": {},
    }

    t_start = time.perf_counter()

    for group in groups:
        mod = load_module("re_mod_" + group, R / "experiment/run_experiment.py")
        cfg_path = R / ("experiment/config.yaml" if group == "feedback"
                        else "experiment/config_no_feedback.yaml")
        cfg = read_yaml(cfg_path)
        mod.add_source_path(cfg)
        mod.set_seed(SEED)
        model, task, criterion, base = mod.build_components(cfg, device)
        ckpt = Path(cfg["paths"]["output_root"]) / f"checkpoints/step_{STEP}.pt"
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["r2p_and_projections"], strict=True)
        task.net_g.load_state_dict(state["tasknet_net_g"], strict=True)
        del state
        model.zero_grad(set_to_none=True)
        task.zero_grad(set_to_none=True)
        model.eval()
        task.eval()
        criterion.eval()

        lambda_align = float(json.loads(
            (Path(cfg["paths"]["output_root"]) / "config"
             / "alignment_calibration.json").read_text())["lambda_feature_alignment"])
        round_w = tuple(float(x) for x in cfg["recurrent"]["round_loss_weights"])
        w1, w2 = round_w

        loss_mod = sys.modules.get("polarapp.losses")
        if loss_mod is None:
            raise RuntimeError("polarapp.losses not imported by criterion construction")
        loss_ssim = loss_mod.ssim
        perceptual = criterion.perceptual
        perc_weight = float(perceptual.weight) if perceptual is not None else 0.0
        vgg = perceptual.extractor if perceptual is not None else None

        shared = tuple(model.r2p.parameters())
        shared_names = [n for n, _ in model.r2p.named_parameters()]
        shared_sizes = [p.numel() for p in shared]
        netg = tuple(task.net_g.parameters())
        netg_sizes = [p.numel() for p in netg]

        def make_grader(params, sizes):
            """Return a grader that extracts a gradient WITHOUT retaining the graph.

            `retain_graph=False` is essential here: every term's gradient is
            extracted exactly once per forward, and the backward frees the graph
            as it goes.  Retaining it would keep the whole 12 GiB forward alive
            for the rest of the step, and the next step's forward then OOMs.
            The free-variable buffer is dropped on return so nothing survives.
            """
            def grad(loss):
                if loss is None or not loss.requires_grad:
                    return None
                out_ = flat(torch.autograd.grad(
                    loss, params, retain_graph=False, allow_unused=True,
                    create_graph=False), sizes)
                return out_
            return grad

        grad_shared = make_grader(shared, shared_sizes)
        grad_netg = make_grader(netg, netg_sizes)

        records = []
        for step in steps:
            one = mod.scheduled_loader(cfg, base, step, step - 1)
            batch = mod.unpack_batch(next(iter(one)), device)
            sample_ids = batch["sample_id"]
            noise_seed = mod.set_step_seed(SEED, "train", step, sample_ids)

            model.train()
            task.eval()
            model.zero_grad(set_to_none=True)
            task.zero_grad(set_to_none=True)

            p0 = batch["polar"]
            condition = batch["condition"]
            official_rgb = batch["official_rgb"]
            target = batch["target"]

            # The diffusion RNG state that training would have at the moment of
            # the first task call of this step.  Every forward below is rewound
            # to exactly this state, so all of them see the same noise stream.
            cpu_rng = torch.get_rng_state()
            cuda_rng = (torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available() else None)

            # cached perceptual target feature (same as the criterion's own
            # extractor(target.detach()) call, done under no_grad to save memory)
            tgt_feat = None
            if vgg is not None:
                with torch.no_grad():
                    tgt_feat = vgg(target.detach())["conv5_4"]

            def restore():
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)

            def terms(pred):
                """The four separable terms of one round's task loss."""
                d = {
                    "l1": F.l1_loss(pred, target),
                    "ssim": 1.0 - loss_ssim(pred, target),
                    "phase": criterion.phase(pred, target),
                }
                if vgg is not None:
                    d["perceptual"] = perc_weight * F.l1_loss(
                        vgg(pred)["conv5_4"], tgt_feat)
                return d

            def combined(pred):
                return sum(v for v in terms(pred).values())

            def check_round(pred, label):
                crit = float(criterion(pred, target))
                rec = float(combined(pred))
                if abs(crit - rec) > 1e-5 * max(1.0, abs(crit)):
                    raise AssertionError(
                        f"{label} loss recomposition mismatch: {crit} vs {rec}")

            def forward_pair(detach_round2_input):
                """Both rounds on one graph.  Returns every handle the caller
                needs; nothing is cached globally."""
                restore()
                P1, f1_raw = model.r2p(condition, p0, return_features=True)
                proj = {name: head(f1_raw["fused"])
                        for name, head in model.projections.items()}
                T1o, f1o = task(P1, official_rgb, None)
                restore()
                inp2 = P1.detach() if detach_round2_input else P1
                P2o, _ = model.r2p(condition, inp2, return_features=True)
                T2o, _ = task(P2o, official_rgb, None)
                return P1, proj, T1o, f1o, P2o, T2o

            # ---- (1) loss values and the recomposition check ------------------
            P1, proj1, T1, f1, P2, T2 = forward_pair(False)
            check_round(T1, "real-path round1")
            check_round(T2, "real-path round2")
            t1 = terms(T1)
            t2 = terms(T2)
            l1_total = sum(v for v in t1.values())
            l2_total = sum(v for v in t2.values())
            align_term = mod.feature_alignment_loss(proj1, f1)
            recA = {
                "L1_total": float(l1_total), "L2_total": float(l2_total),
                "criterion_L1": float(criterion(T1, target)),
                "criterion_L2": float(criterion(T2, target)),
                "loss1_parts": {k: float(v) for k, v in t1.items()},
                "loss2_parts": {k: float(v) for k, v in t2.items()},
                "alignment": float(align_term),
            }
            del P1, P2, T1, T2, t1, t2, l1_total, l2_total, f1, proj1, align_term
            model.zero_grad(set_to_none=True)
            task.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            # ---- (2) gradients, one forward per quantity ----------------------
            # retain_graph=False means each autograd.grad consumes its graph, so
            # every quantity below gets a fresh forward.  Slower, but the only
            # way to keep peak memory inside 12 GiB.
            COMP = ("l1", "ssim", "phase", "perceptual")

            def grad_terms(round_tag, detach_r2):
                """Gradients of the four per-round terms w.r.t. shared params."""
                out_ = {}
                for k in COMP:
                    if k == "perceptual" and vgg is None:
                        continue
                    restore()
                    P1, proj, T1o, f1o, P2o, T2o = forward_pair(detach_r2)
                    pred = T1o if round_tag == "L1" else T2o
                    out_[round_tag + "." + k] = grad_shared(terms(pred)[k])
                    del P1, proj, T1o, f1o, P2o, T2o
                    torch.cuda.empty_cache()
                return out_

            def grad_total(round_tag, detach_r2):
                """Shared-parameter gradient of one round's summed task loss."""
                restore()
                P1, proj, T1o, f1o, P2o, T2o = forward_pair(detach_r2)
                pred = T1o if round_tag == "L1" else T2o
                g_sh = grad_shared(combined(pred))
                del P1, proj, T1o, f1o, P2o, T2o
                torch.cuda.empty_cache()
                return g_sh

            def grad_total_netg(round_tag, detach_r2):
                """net_g gradient of one round's summed task loss."""
                restore()
                P1, proj, T1o, f1o, P2o, T2o = forward_pair(detach_r2)
                pred = T1o if round_tag == "L1" else T2o
                g_ng = grad_netg(combined(pred))
                del P1, proj, T1o, f1o, P2o, T2o
                torch.cuda.empty_cache()
                return g_ng

            def grad_align():
                restore()
                P1, proj, T1o, f1o, P2o, T2o = forward_pair(False)
                g = grad_shared(mod.feature_alignment_loss(proj, f1o))
                del P1, proj, T1o, f1o, P2o, T2o
                torch.cuda.empty_cache()
                return g

            # decomposition A (real path: what training actually builds)
            gsA = {}
            gsA.update(grad_terms("L1", False))
            gsA.update(grad_terms("L2", False))
            gsA["L1.total"] = grad_total("L1", False)
            gsA["L2.total"] = grad_total("L2", False)
            gsA["align"] = grad_align()
            # net_g is the OTHER trainable group; it sees only the task losses.
            ngA_L1 = grad_total_netg("L1", False)
            ngA_L2 = grad_total_netg("L2", False)

            # decomposition B (isolated: round 2 sees a DETACHED P1)
            gsB_L1 = grad_total("L1", True)
            gsB_L2 = grad_total("L2", True)

            # ---- length consistency check ------------------------------------
            exp = int(sum(shared_sizes))
            for tag, gg in (("real_path", gsA),
                            ("isolated_L1", {"x": gsB_L1}),
                            ("isolated_L2", {"x": gsB_L2})):
                bad = {k: int(v.numel()) for k, v in gg.items()
                       if v is not None and int(v.numel()) != exp}
                if bad:
                    raise AssertionError(
                        f"{tag}: gradient vectors with wrong length: {bad}")

            # ---- assemble the record -----------------------------------------
            def wvec(gg, key, w):
                v = gg.get(key)
                return None if v is None else v * w

            gts = add(wvec(gsA, "L1.total", w1), wvec(gsA, "L2.total", w2))
            gob = add(gts, wvec(gsA, "align", lambda_align))

            focus = {}

            def put(name, a, b):
                c = cos(gsA.get(a), gsA.get(b))
                if c is not None:
                    focus[name] = c

            for nm, a, b in (
                ("L1.total vs L2.total", "L1.total", "L2.total"),
                ("L1.phase vs L2.phase", "L1.phase", "L2.phase"),
                ("L1.phase vs L1.l1", "L1.phase", "L1.l1"),
                ("L1.phase vs L1.ssim", "L1.phase", "L1.ssim"),
                ("L1.l1 vs L1.ssim", "L1.l1", "L1.ssim"),
                ("L1.phase vs L1.perceptual", "L1.phase", "L1.perceptual"),
                ("L1.l1 vs L1.perceptual", "L1.l1", "L1.perceptual"),
                ("L2.phase vs L2.l1", "L2.phase", "L2.l1"),
                ("L2.phase vs L2.ssim", "L2.phase", "L2.ssim"),
                ("L2.l1 vs L2.ssim", "L2.l1", "L2.ssim"),
                ("L2.phase vs L2.perceptual", "L2.phase", "L2.perceptual"),
                ("L2.l1 vs L2.perceptual", "L2.l1", "L2.perceptual"),
                ("align vs L1.total", "align", "L1.total"),
                ("align vs L2.total", "align", "L2.total"),
            ):
                put(nm, a, b)

            # each weighted term vs the summed task gradient
            for nm, key, w in (("L1", "L1.total", w1), ("L2", "L2.total", w2),
                               ("align", "align", lambda_align)):
                c = cos(wvec(gsA, key, w), gts)
                if c is not None:
                    focus[nm + " weighted vs TASKSUM"] = c

            # net_g: round 1 vs round 2 task gradients on the OTHER trainable group
            netg_focus = {}
            c = cos(ngA_L1, ngA_L2)
            if c is not None:
                netg_focus["L1.total vs L2.total"] = c

            rec = {
                "step": step,
                "diffusion_seed": noise_seed,
                "sample_ids": sample_ids,
                "batch_size": int(p0.shape[0]),
                "lambda_feature_alignment": lambda_align,
                "round_weights": list(round_w),
                "perceptual_weight": perc_weight,
                "shared_param_count": exp,
                "real_path": {
                    "loss_values": recA,
                    "shared_grad_norms_unit": {
                        k: (None if v is None else vnorm(v))
                        for k, v in gsA.items()},
                    "netg_grad_norms_unit": {
                        "L1.total": vnorm(ngA_L1), "L2.total": vnorm(ngA_L2)},
                    "cosines_unit": focus,
                },
                "isolated_round2_vs_round1": {
                    "L1_grad_norm_unit": vnorm(gsB_L1),
                    "L2_grad_norm_unit": vnorm(gsB_L2),
                    "L1_weighted_norm": vnorm(gsB_L1) * w1,
                    "L2_weighted_norm": vnorm(gsB_L2) * w2,
                    "cosine_L1_vs_L2": cos(gsB_L1, gsB_L2),
                },
                "weighted_vectors_on_shared": {
                    "L1.total.weighted_norm": vnorm(wvec(gsA, "L1.total", w1)),
                    "L2.total.weighted_norm": vnorm(wvec(gsA, "L2.total", w2)),
                    "align.weighted_norm": vnorm(wvec(gsA, "align", lambda_align)),
                    "TASKSUM.weighted_norm": vnorm(gts),
                    "OBJECTIVE.weighted_norm": vnorm(gob),
                },
                "netg_cosines_unit": netg_focus,
                "focused_cosines_unit_or_weighted": focus,
            }

            n_obj = vnorm(gob)
            rec["clip_bookkeeping"] = {
                "grad_clip": 1.0,
                "objective_grad_norm_on_shared_preclip": n_obj,
                "would_clip_shared_block_alone": bool(n_obj > 1.0),
                "note": ("training clips the summed gradient over ALL "
                         "model.parameters() (shared R2P + projections + feedback "
                         "adapter) jointly to global norm 1.0; this number covers "
                         "only the shared R2P block, so it is a LOWER BOUND"),
            }

            records.append(rec)
            iso = rec["isolated_round2_vs_round1"]
            print("[%s] step %d |g_TASKSUM|=%.6g |g_align|=%.6g "
                  "cos(L1,L2)real=%.4f iso=%.4f cos(ph1,ph2)=%.4f"
                  % (group, step,
                     rec["weighted_vectors_on_shared"]["TASKSUM.weighted_norm"],
                     rec["weighted_vectors_on_shared"]["align.weighted_norm"],
                     focus.get("L1.total vs L2.total", float("nan")),
                     (iso["cosine_L1_vs_L2"] if iso["cosine_L1_vs_L2"] is not None
                      else float("nan")),
                     focus.get("L1.phase vs L2.phase", float("nan"))),
                  flush=True)

            del tgt_feat, batch, gsA, gsB_L1, gsB_L2
            del ngA_L1, ngA_L2, gts, gob, p0, condition, official_rgb, target
            model.zero_grad(set_to_none=True)
            task.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        # ---- per-group aggregation ------------------------------------------
        def summarize(key):
            vals = [r["focused_cosines_unit_or_weighted"].get(key) for r in records]
            vals = [v for v in vals if v is not None]
            if not vals:
                return None
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                    "min": float(np.min(vals)), "max": float(np.max(vals)),
                    "n": len(vals)}

        def summarize_iso(key):
            vals = [r["isolated_round2_vs_round1"].get(key) for r in records]
            vals = [v for v in vals if v is not None]
            if not vals:
                return None
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                    "min": float(np.min(vals)), "max": float(np.max(vals)),
                    "n": len(vals)}

        def summarize_path(path):
            vals = []
            for r in records:
                v = r
                for k in path:
                    v = v.get(k) if isinstance(v, dict) else None
                    if v is None:
                        break
                vals.append(v)
            vals = [v for v in vals if v is not None]
            if not vals:
                return None
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                    "min": float(np.min(vals)), "max": float(np.max(vals)),
                    "n": len(vals)}

        out["groups"][group] = {
            "group": group,
            "feedback_enabled": bool(model.feedback_enabled),
            "lambda_feature_alignment": lambda_align,
            "round_loss_weights": list(round_w),
            "perceptual_weight": perc_weight,
            "shared_param_count": int(sum(shared_sizes)),
            "netg_param_count": int(sum(netg_sizes)),
            "shared_param_names": shared_names,
            "records": records,
            "cosine_summary": {k: summarize(k) for k in [
                "L1.total vs L2.total", "L1.phase vs L2.phase",
                "L1.phase vs L1.l1", "L1.phase vs L1.ssim", "L1.l1 vs L1.ssim",
                "L2.phase vs L2.l1", "L2.phase vs L2.ssim", "L2.l1 vs L2.ssim",
                "L1.phase vs L1.perceptual", "L2.phase vs L2.perceptual",
                "L1 weighted vs TASKSUM", "L2 weighted vs TASKSUM",
                "align weighted vs TASKSUM", "align vs L1.total",
                "align vs L2.total",
            ]},
            "isolated_summary": {k: summarize_iso(k) for k in [
                "cosine_L1_vs_L2", "L1_grad_norm_unit", "L2_grad_norm_unit",
                "L1_weighted_norm", "L2_weighted_norm",
            ]},
            "netg_cosine_summary": {
                "L1.total vs L2.total": None},
            "term_grad_norm_summary": {
                k: summarize_path(["real_path", "shared_grad_norms_unit", k])
                for k in ("L1.l1", "L1.ssim", "L1.phase", "L1.perceptual",
                          "L1.total", "L2.l1", "L2.ssim", "L2.phase",
                          "L2.perceptual", "L2.total", "align")},
        }
        print("[%s] done (%.1f s)" % (group, time.perf_counter() - t_start), flush=True)
        del model, task, criterion
        torch.cuda.empty_cache()

    out["elapsed_seconds"] = time.perf_counter() - t_start
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("WROTE", args.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
