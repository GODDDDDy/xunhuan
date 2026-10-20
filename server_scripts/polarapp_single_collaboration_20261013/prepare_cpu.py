"""Disposable CPU-only integration checks, never a formal training entrypoint."""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import gc
import copy
import hashlib
import json
import random
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from modules.single_system import build_system, trainable_names, joint_objective, sha256
from modules.meta import FunctionalModelState, bilevel_probe, joint_objective_gradients


def emit(stage, **values):
    print(json.dumps({"stage": stage, **values}, ensure_ascii=False), flush=True)


def tensor_hash(value):
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def fixed_forward(up, down, batch, seed=21013, source=False):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        p, pf = up(batch["polar"], batch["official_rgb"], use_adapters=not source)
        prediction, tf = down(p, batch["official_rgb"], batch["polar"], use_adapters=not source)
    return p, prediction, pf, tf


def cosine_loss(alignment, p, t):
    transformed, matched = alignment.call(p, t)
    return sum(1 - F.cosine_similarity(transformed[f"TF{i}"], matched[f"MF{i}"], dim=1).mean()
               for i in (1, 2, 3)) / 3


def selected_training_batches(runtime):
    from rgb_final_confirmation.data import load_manifest, PairedBranchDataset
    rows = load_manifest(runtime["manifest"])
    training = [r for r in rows if r.split == "train"]
    first = next(r for r in training if r.scene in runtime["meta_inner_scene_ids"])
    other = next(r for r in training if r.scene in runtime["meta_outer_scene_ids"])
    assert first.scene != other.scene
    chosen = [first, other]
    dataset = PairedBranchDataset(chosen, "A1", crop_size=32,
                                 schedule=[{"sample_index": i, "top": 0, "left": 0} for i in (0, 1)])
    batches = []
    for i in (0, 1):
        value = dataset[i]
        batches.append({k: v[None] if torch.is_tensor(v) else v for k, v in value.items()})
    return batches


def main():
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    random.seed(2026)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized() and not torch.cuda.is_available()
    start = time.perf_counter()
    up, down, fa, source_receipt, runtime = build_system(ROOT)
    emit("source_loaded", **source_receipt)
    inner, outer = selected_training_batches(runtime)
    counts = {"enhancer": 0, "PolarFree_generator": 0}
    handles = [up.core.register_forward_hook(lambda *_: counts.__setitem__("enhancer", counts["enhancer"]+1)),
               down.task.net_g.register_forward_hook(lambda *_: counts.__setitem__("PolarFree_generator", counts["PolarFree_generator"]+1))]
    with torch.no_grad():
        initial = fixed_forward(up, down, inner)
    for handle in handles:
        handle.remove()
    assert counts == {"enhancer": 1, "PolarFree_generator": 1}
    with torch.no_grad():
        source = fixed_forward(up, down, inner, source=True)
        assert torch.equal(initial[0], source[0]) and torch.equal(initial[1], source[1])
        replacement = dict(inner, target=torch.rand_like(inner["target"]))
        replaced = fixed_forward(up, down, replacement)
        assert torch.equal(initial[1], replaced[1])
        source_outer = fixed_forward(up, down, outer, source=True)[0]
    assert not any(any(token in name for token in ("confidence_head", "feedback", "error_guidance", "action_gate"))
                   for module in (up, down) for name, _ in module.named_parameters())
    receipt = {"source": source_receipt, "alignment_source": fa.load_receipt,
               "authorization": "preparation_only", "formal_training_started": False,
               "GPU_used": False, "cuda_initialized": False,
               "samples": [{"sample_id": b["sample_id"], "scene": b["scene"], "shape": list(b["polar"].shape)}
                           for b in (inner, outer)],
               "validation_images_read": 0, "test_images_read": 0,
               "actual_forward_counts": counts, "zero_adapters_equal_source_single_round": True,
               "GT_replacement_output_invariant": True,
               "initial_prediction_sha256": tensor_hash(initial[1]),
               "inactive_mechanisms_absent": True,
               "alignment_partition": fa.partition_summary()}
    emit("single_round_initial_checks_passed")

    def states():
        return (FunctionalModelState.from_module(up, trainable_names(up), 1e-4, label="enhancer"),
                FunctionalModelState.from_module(down, trainable_names(down), 1e-4, label="task"),
                FunctionalModelState.from_module(fa, trainable_names(fa), 1e-5, label="alignment"))

    def inner_alignment(u, d, a):
        _, _, pfeatures, tfeatures = fixed_forward(u.call, d.call, inner)
        return cosine_loss(a, pfeatures, tfeatures)

    def outer_objective(u, d, _a):
        polar, prediction, _, _ = fixed_forward(u.call, d.call, outer, seed=21014)
        # Fixed positive scale is for the FANet optimizer's numerical units;
        # task-vs-retention proportions and actual joint loss are unchanged.
        return runtime["meta_outer_loss_scale"] * joint_objective(
            polar, prediction, outer["target"], source_outer)

    current = states()
    meta = bilevel_probe(*current, inner_alignment, outer_objective, outer_create_graph=False)
    assert meta.audit["upstream_inner_gradient_norm"] > 0
    assert meta.audit["downstream_inner_gradient_norm"] > 0
    assert meta.audit["alignment_hypergradient_norm"] > 0
    receipt["actual_model_bilevel"] = dict(meta.audit)
    receipt["actual_model_bilevel"].update(inner_loss=float(meta.inner_loss.detach()), outer_loss=float(meta.outer_loss.detach()))
    receipt["actual_model_bilevel"]["outer_loss_scale"] = runtime["meta_outer_loss_scale"]
    emit("actual_bilevel_passed", **{k: v for k, v in meta.audit.items() if k.endswith("norm")})
    del current, meta
    gc.collect()

    def joint(u, d, a, weight):
        polar, prediction, pf, tf = fixed_forward(u.call, d.call, inner)
        value = joint_objective(polar, prediction, inner["target"], source[0])
        if weight:
            value = value + weight * cosine_loss(a, pf, tf)
        return value

    branch_gradients = {}
    for branch, weight in (("matched_no_alignment", 0.), ("meta_alignment", .01)):
        current = states()
        gradients = joint_objective_gradients(*current, lambda u, d, a: joint(u, d, a, weight))
        branch_gradients[branch] = {"loss": float(gradients.loss.detach()), **dict(gradients.audit)}
        assert sum(float(g.abs().sum()) for g in gradients.upstream_gradients.values()) > 0
        assert sum(float(g.abs().sum()) for g in gradients.downstream_gradients.values()) > 0
        del current, gradients
        gc.collect()
    receipt["matched_branch_checks"] = branch_gradients
    emit("matched_objectives_gradient_checked")

    # One discarded diagnostic optimizer update verifies the actual adapters,
    # frozen tensor invariance and exact optimizer/RNG restore. No checkpoint
    # from these updates may be used as a formal initialization.
    parameters = [p for m in (up, down) for p in m.parameters() if p.requires_grad]
    frozen = {(side, name): value.detach().clone() for side, m in (("up", up), ("down", down))
              for name, value in m.named_parameters() if not value.requires_grad}
    initial_adapters = [p.detach().clone() for p in parameters]
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    polar, prediction, _, _ = fixed_forward(up, down, inner)
    loss = joint_objective(polar, prediction, inner["target"], source[0])
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
    optimizer.step()
    changed = sum(not torch.equal(before, after) for before, after in zip(initial_adapters, parameters))
    assert changed > 0
    for side, m in (("up", up), ("down", down)):
        assert all(torch.equal(p, frozen[side, name]) for name, p in m.named_parameters() if not p.requires_grad)
    resume_parameters = [p.detach().clone() for p in parameters]
    resume_optimizer = copy.deepcopy(optimizer.state_dict())
    resume_rng = torch.get_rng_state().clone()
    def resume_update():
        optimizer.zero_grad(set_to_none=True)
        p, y, _, _ = fixed_forward(up, down, inner)
        joint_objective(p, y, inner["target"], source[0]).backward()
        optimizer.step()
        return [p.detach().clone() for p in parameters], torch.rand(7)
    expected_parameters, expected_random = resume_update()
    with torch.no_grad():
        for p, before in zip(parameters, resume_parameters):
            p.copy_(before)
    optimizer.load_state_dict(copy.deepcopy(resume_optimizer))
    torch.set_rng_state(resume_rng)
    actual_parameters, actual_random = resume_update()
    assert all(torch.equal(a, b) for a, b in zip(expected_parameters, actual_parameters))
    assert torch.equal(expected_random, actual_random)
    with torch.no_grad():
        for p, before in zip(parameters, initial_adapters):
            p.copy_(before)
        restored = fixed_forward(up, down, inner)
    assert torch.equal(restored[0], initial[0]) and torch.equal(restored[1], initial[1])
    for p in parameters:
        p.grad = None
    receipt["discarded_optimizer_check"] = {"changed_adapter_tensors": changed,
                                            "frozen_parameters_equal_source": True,
                                            "initial_output_restored_exactly": True,
                                            "optimizer_and_RNG_resume_exact": True,
                                            "updated_weights_saved": False}
    receipt["parameter_counts"] = {side: {"total": sum(p.numel() for p in m.parameters()),
                                           "trainable": sum(p.numel() for p in m.parameters() if p.requires_grad),
                                           "names": list(trainable_names(m))}
                                   for side, m in (("enhancer", up), ("task", down), ("FANet", fa))}
    receipt["source_manifest_sha256"] = sha256(runtime["manifest"])
    receipt["passed"] = True
    receipt["elapsed_seconds"] = time.perf_counter()-start
    receipt["cuda_initialized"] = torch.cuda.is_initialized()
    assert receipt["cuda_initialized"] is False
    (ROOT / "evidence/single_system_cpu_receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    emit("preparation_cpu_passed", seconds=receipt["elapsed_seconds"], formal_training_started=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        payload = {"passed": False, "error": str(error), "traceback": traceback.format_exc(),
                   "formal_training_started": False, "GPU_used": False}
        (ROOT / "evidence").mkdir(parents=True, exist_ok=True)
        (ROOT / "evidence/single_system_cpu_failure.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        raise
