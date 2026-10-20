"""Stage A2, part 1: minimal spatial cache + mandatory online/cache equivalence.

Builds the 50-input-channel native judgement head exactly as verified for
experiment 22 (gain22.FloatActionCostGate over the frozen trunk), strictly loads
the experiment-22 final action_gate, generates the 48-crop spatial cache with a
single frozen two-round forward per crop, and verifies that the cached training
path is numerically equivalent to the online path.

No 188-test frames, no re-run of the 872 source diagnostic, trunk frozen.
"""
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(HERE))

from common import guard_gpu, json_save, save, seed, sha, status  # noqa: E402

assert Path(__import__("common").ROOT).resolve() == RUN, "ROOT must be the new audit run"

import gain22  # noqa: E402  (sets staged.components to the 50-channel FloatActionCostGate builder)
import staged  # noqa: E402
import study  # noqa: E402
from acceptance import LocalAcceptance, resize  # noqa: E402
from action_gate import ACTIONS, action_costs, compose, cost_loss  # noqa: E402

from a2_common import build_head_inputs, sha_array, spearman  # noqa: E402

SRC = Path("/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/polar_gain_judgement_short_20261012")
FINAL = SRC / "downstream/checkpoints/final.pt"
REUSED = SRC / "weights/reused_final.pt"
EXPECT_FINAL = "279d0a329d0efebee9678c2156bf3e7b60c0646e75a5f37f8015c7b100f9e9a2"
EXPECT_REUSED = "cecdcfa47d290b0a2d6b662b96bb4407a7043ef5789e9e87d70768f2c683916d"

CROP = 128
PER_SCENE = {"judgement": 2, "bias": 2, "safety": 4}
ATOL, RTOL = 1e-6, 1e-4
EQUIV_CROPS = 5


def load_cfg():
    return json.loads((RUN / "config/locked_run.json").read_text(encoding="utf-8"))


def build_model(cfg, device):
    seed(cfg["seed"])
    model, task, raw = gain22.components(cfg, device)
    final = torch.load(FINAL, map_location="cpu", weights_only=False)
    gate_state = {k[len("action_gate."):]: v for k, v in final["r2p_and_projections"].items()
                  if k.startswith("action_gate.")}
    model.action_gate.load_state_dict(gate_state, strict=True)
    model.cost_scale = final["cost_scale"]
    model.cost_bias = final["cost_bias"]
    model.safety_margin = final["safety_margin"]
    model.tau = final["tau"]
    model.eval()
    task.eval()
    model.requires_grad_(False)
    task.requires_grad_(False)
    return model, task, final


def audit_source(model, task, final):
    stored = final["r2p_and_projections"]
    mismatched = [n for n, v in model.state_dict().items()
                  if not n.startswith("action_gate.") and not torch.equal(v.cpu(), stored[n])]
    net_g_mismatched = [n for n, v in task.net_g.state_dict().items()
                        if not torch.equal(v.cpu(), final["tasknet_net_g"][n])]
    body = model.action_gate.body[0]
    report = dict(
        non_action_gate_tensors=len(model.state_dict()) - sum(n.startswith("action_gate.") for n in model.state_dict()),
        mismatched_non_action_gate=mismatched,
        net_g_tensors=len(task.net_g.state_dict()),
        mismatched_net_g=net_g_mismatched,
        action_gate_body0_in_channels=int(body.in_channels),
        action_gate_body0_out_channels=int(body.out_channels),
        is_50_input_channel=bool(body.in_channels == 50),
        cost_scale=model.cost_scale,
        cost_bias=list(model.cost_bias),
        safety_margin=model.safety_margin,
        all_non_judgement_tensors_equal_source=not mismatched,
        all_net_g_tensors_equal_source=not net_g_mismatched,
    )
    if mismatched or net_g_mismatched or body.in_channels != 50:
        raise SystemExit("source audit failed: " + json.dumps(report, ensure_ascii=False))
    return report


def role_rows(cfg):
    from rgb_final_confirmation.data import load_manifest, select_rows
    rows = select_rows(load_manifest(cfg["paths"]["manifest"]), "train")
    by_scene = {}
    for row in rows:
        by_scene.setdefault(row.scene, []).append(row)
    roles = {}
    for role in ("judgement", "bias", "safety"):
        member = [r for r in rows if r.scene in set(cfg["staged"]["scene_roles"][role])]
        roles[role] = member
    return roles, by_scene


def pick_evenly(rows, k):
    ordered = sorted(rows, key=lambda r: hashlib.sha256(r.sample_id.encode()).hexdigest())
    n = len(ordered)
    if n <= k:
        return ordered
    chosen, seen = [], set()
    for j in range(k):
        index = int(round(j * (n - 1) / (k - 1)))
        if index not in seen:
            chosen.append(ordered[index])
            seen.add(index)
    return chosen


def crop_coords(sample_id, j, width, height, base_seed):
    tag = hashlib.sha256(f"{base_seed}:{sample_id}:{j}".encode()).hexdigest()[:16]
    generator = torch.Generator().manual_seed(int(tag, 16) % (2 ** 31))
    top = int(torch.randint(0, height - CROP + 1, (1,), generator=generator).item())
    left = int(torch.randint(0, width - CROP + 1, (1,), generator=generator).item())
    return top, left


def build_plan(cfg, roles, by_scene):
    plan = []
    for role in ("judgement", "bias", "safety"):
        k = PER_SCENE[role]
        index_of = {id(r): i for i, r in enumerate(roles[role])}
        for scene in cfg["staged"]["scene_roles"][role]:
            for j, row in enumerate(pick_evenly(by_scene[scene], k)):
                with Image.open(row.polar_000) as image:
                    width, height = image.size
                top, left = crop_coords(row.sample_id, j, width, height, cfg["seed"])
                plan.append(dict(role=role, scene=scene, sample_id=row.sample_id,
                                 sample_index=index_of[id(row)], orig_w=width, orig_h=height,
                                 top=top, left=left, crop=CROP, group="A1",
                                 augmentation="none", rng_tag=f"{cfg['seed']}:{row.sample_id}:{j}"))
    return plan


def load_crop(rows, entry, device):
    from rgb_final_confirmation.data import PairedBranchDataset
    dataset = PairedBranchDataset(rows, group="A1", crop_size=CROP,
                                  schedule=[dict(sample_index=entry["sample_index"],
                                                 top=entry["top"], left=entry["left"])])
    sample = dataset[0]
    return {k: (v[None].to(device) if torch.is_tensor(v) else [v]) for k, v in sample.items()}


def np1(tensor):
    return tensor[0].detach().cpu().numpy().astype(np.float32)


def pool_features(model, task, batch, token, device):
    study.base.set_step_seed(cfg_seed_holder["seed"], "a2_native_cache", token, batch["sample_id"])
    with torch.no_grad():
        out = study.recurrent_forward(model, task, batch)
    image = batch["official_rgb"].detach().float()
    polar = study.polarization_summary(batch["polar"]).detach().float()
    a = out.t1.detach().float()
    p = out.ungated_t2.detach().float()
    d = p - a
    maps = [F.avg_pool2d(x, 4) for x in (image, a, p, d, polar)]
    size = maps[0].shape[-2:]
    err = resize(out.error_scale.detach(), size)
    conf = resize(out.confidence_map.detach(), size)
    feat = resize(out.judgement_feature.detach(), size)
    state = resize(out.judgement_state.detach(), size)
    dsq = F.avg_pool2d(d.square(), 4)
    dabs = F.avg_pool2d(d.abs(), 4)
    costs = action_costs(out.t1, out.rgb_increment, batch["target"])
    return dict(
        img=maps[0], t1p=maps[1], Up=maps[2], dp=maps[3], polp=maps[4],
        err=err, conf=conf, feat=feat, state=state, dsq=dsq, dabs=dabs,
        y=costs[:, 1:] / model.cost_scale, y_raw=costs[:, 1:],
        t1f=out.t1, Uf=out.ungated_t2, dlf=out.rgb_increment,
        costs4=costs,
    ), out


cfg_seed_holder = {"seed": 2026}


def main():
    started = time.perf_counter()
    timing = {}
    cfg = load_cfg()
    cfg_seed_holder["seed"] = cfg["seed"]
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise SystemExit("A2 requires CUDA")
    status("A2_cache_start")

    tick = time.perf_counter()
    model, task, final = build_model(cfg, device)
    audit = audit_source(model, task, final)
    timing["model_build_seconds"] = time.perf_counter() - tick

    roles, by_scene = role_rows(cfg)
    plan = build_plan(cfg, roles, by_scene)
    counts = {r: sum(1 for e in plan if e["role"] == r) for r in ("judgement", "bias", "safety")}
    print(json.dumps(dict(plan_crops=len(plan), per_role=counts)), flush=True)

    tick = time.perf_counter()
    guard_gpu()
    records = []
    for index, entry in enumerate(plan):
        batch = load_crop(roles[entry["role"]], entry, device)
        pooled, _ = pool_features(model, task, batch, index, device)
        records.append({k: np1(v) for k, v in pooled.items() if k != "costs4"})
        if (index + 1) % 8 == 0:
            print(json.dumps(dict(cached=index + 1, total=len(plan))), flush=True)
    timing["cache_forward_seconds"] = time.perf_counter() - tick

    keys = [k for k in records[0].keys()]
    arrays = {k: np.stack([r[k] for r in records]) for k in keys}
    cache_path = RUN / "cache/a2_spatial_cache.npz"
    np.savez_compressed(cache_path, **arrays)
    cache_sha = sha(cache_path)
    index_of_role = {}
    for position, entry in enumerate(plan):
        index_of_role.setdefault(entry["role"], []).append(position)
    json_save(RUN / "cache/a2_cache_index.json", dict(
        crops=plan, role_positions=index_of_role, per_role=counts, crop_size=CROP,
        pooled_cells=32, scale=model.cost_scale, channels=dict(
            img=3, t1p=3, Up=3, dp=3, polp=5, err=1, conf=1, feat=96, state=16, dsq=3, dabs=3,
            y=3, y_raw=3, t1f=3, Uf=3, dlf=3),
        array_keys=keys, cache_npz_sha256=cache_sha,
        chain="frozen two-round trunk once per crop; GT enters labels only",
        full_32x32_adjacency=True, augmentation="none", selection_before_reading_target=True))

    save(RUN / "checkpoints/diagnostic_heads/a2_native_common_init.pt", dict(
        state=model.action_gate.state_dict(),
        sha256=hashlib.sha256(json.dumps({k: sha_array(v.detach().cpu().numpy())
                                          for k, v in model.action_gate.state_dict().items()},
                                         sort_keys=True).encode()).hexdigest(),
        source="experiment22 downstream/checkpoints/final.pt action_gate",
        source_final_sha256=sha(FINAL), input_channels=50))

    # ---------------- mandatory online / cache equivalence ----------------
    tick = time.perf_counter()
    equivalence = equivalence_check(cfg, model, task, roles, plan, arrays, index_of_role, device)
    timing["equivalence_seconds"] = time.perf_counter() - tick

    timing["total_seconds"] = time.perf_counter() - started
    json_save(RUN / "receipts/online_cache_equivalence.json", equivalence)
    json_save(RUN / "metrics/A2_cache_audit.json", dict(
        source_audit=audit, plan_crops=len(plan), per_role=counts,
        cache_npz_sha256=cache_sha, cache_bytes=cache_path.stat().st_size,
        equivalence_passed=equivalence["passed"], timing=timing,
        gpu_used=True, test_used=False, validation_used=False,
        source_diagnostic_872_rerun=False))
    json_save(RUN / "logs/a2_timing.json", timing)
    print(json.dumps(dict(cache_done=True, equivalence=equivalence["passed"], timing=timing)), flush=True)
    status("A2_cache_complete", equivalence_passed=equivalence["passed"])


def equivalence_check(cfg, model, task, roles, plan, arrays, index_of_role, device):
    """Compare the online gate path with the cached training path on fixed crops."""
    positions = index_of_role["judgement"][:EQUIV_CROPS]
    online = copy.deepcopy(model.action_gate).to(device).requires_grad_(True).train()
    cached = copy.deepcopy(model.action_gate).to(device).requires_grad_(True).train()
    assert all(torch.equal(a, b) for (na, a), (nb, b) in zip(online.state_dict().items(),
                                                             cached.state_dict().items()))
    results = []
    total_online = torch.zeros((), device=device)
    total_cached = torch.zeros((), device=device)
    grad_compare = {}

    for order, position in enumerate(positions):
        entry = plan[position]
        batch = load_crop(roles[entry["role"]], entry, device)
        guard_gpu()
        # The frozen two-round forward is RNG-sensitive through set_step_seed, so
        # equivalence requires the identical (phase, step, sample) triple that was
        # used when this crop was cached. `order` is only for the receipt.
        study.base.set_step_seed(cfg["seed"], "a2_native_cache", position, batch["sample_id"])
        with torch.no_grad():
            out = study.recurrent_forward(model, task, batch)
        online.zero_grad(set_to_none=True)
        cached.zero_grad(set_to_none=True)
        x_online = online.inputs(*gain22.args(out, batch))
        raw_online = online.body(x_online)
        cost4_online = action_costs(out.t1, out.rgb_increment, batch["target"])
        loss_online = cost_loss(raw_online, cost4_online, model.cost_scale) / EQUIV_CROPS

        record = {k: torch.from_numpy(arrays[k][position])[None].to(device) for k in arrays}
        x_cached = build_head_inputs(record, cached)
        raw_cached = cached.body(x_cached)
        cost4_cached = torch.cat([torch.zeros_like(record["y_raw"][:, :1]), record["y_raw"]], 1)
        loss_cached = cost_loss(raw_cached, cost4_cached, model.cost_scale) / EQUIV_CROPS

        assert raw_online.shape == raw_cached.shape, (raw_online.shape, raw_cached.shape)
        loss_online.backward()
        loss_cached.backward()
        total_online = total_online + loss_online.detach()
        total_cached = total_cached + loss_cached.detach()
        results.append(dict(
            sample_id=entry["sample_id"], scene=entry["scene"],
            raw_cost_max_abs=float((raw_online - raw_cached).abs().max()),
            raw_cost_allclose=bool(torch.allclose(raw_online, raw_cached, atol=ATOL, rtol=RTOL)),
            raw_cost_bitwise_equal=bool(torch.equal(raw_online, raw_cached)),
            loss_online=float(loss_online.detach()) * EQUIV_CROPS,
            loss_cached=float(loss_cached.detach()) * EQUIV_CROPS,
        ))

    for name in ("feature_projection.weight", "feature_projection.bias",
                 "body.0.weight", "body.0.bias", "body.2.weight", "body.2.bias",
                 "body.4.weight", "body.4.bias"):
        a = dict(online.named_parameters())[name].grad
        b = dict(cached.named_parameters())[name].grad
        if a is None or b is None:
            grad_compare[name] = dict(present=False)
            continue
        denom = b.abs().clamp_min(1e-12)
        grad_compare[name] = dict(
            max_abs=float((a - b).abs().max()),
            max_rel=float(((a - b).abs() / denom).max()),
            allclose=bool(torch.allclose(a, b, atol=ATOL, rtol=RTOL)),
            nonzero=bool(b.abs().sum() > 0),
        )

    clip = float(torch.nn.utils.clip_grad_norm_(online.parameters(), 1))
    clip2 = float(torch.nn.utils.clip_grad_norm_(cached.parameters(), 1))
    opt_online = torch.optim.Adam(online.parameters(), lr=1e-4)
    opt_cached = torch.optim.Adam(cached.parameters(), lr=1e-4)
    opt_online.step()
    opt_cached.step()
    param_diff = {}
    for (na, a), (nb, b) in zip(online.state_dict().items(), cached.state_dict().items()):
        param_diff[na] = dict(max_abs=float((a - b).abs().max()),
                              equal=bool(torch.equal(a, b)),
                              allclose=bool(torch.allclose(a, b, atol=ATOL, rtol=RTOL)))

    # GT replacement must change labels but not the deployable features.
    # The replacement target is drawn BEFORE seeding: torch.rand_like consumes the
    # global RNG stream, and the frozen two-round forward is RNG-sensitive, so any
    # draw between set_step_seed and recurrent_forward would shift the forward.
    entry = plan[positions[0]]
    batch = load_crop(roles[entry["role"]], entry, device)
    guard_gpu()
    replaced = dict(batch, target=torch.rand_like(batch["target"]))
    study.base.set_step_seed(cfg["seed"], "a2_native_cache", positions[0], batch["sample_id"])
    with torch.no_grad():
        out_replaced = study.recurrent_forward(model, task, replaced)
    original = {k: torch.from_numpy(arrays[k][positions[0]])[None].to(device) for k in arrays}
    a2 = out_replaced.t1.detach().float()
    p2 = out_replaced.ungated_t2.detach().float()
    d2 = p2 - a2
    maps2 = [F.avg_pool2d(x, 4) for x in (batch["official_rgb"].detach().float(), a2, p2, d2,
                                          study.polarization_summary(batch["polar"]).detach().float())]
    feature_keys = ("img", "t1p", "Up", "dp", "polp")
    gt_features_unchanged = all(torch.equal(maps2[i], original[k]) for i, k in enumerate(feature_keys))
    replaced_costs = action_costs(out_replaced.t1, out_replaced.rgb_increment, replaced["target"])[:, 1:]
    labels_changed = bool((replaced_costs - original["y_raw"]).abs().max() > 1e-6)
    mask_zero = compose(out_replaced.t1, out_replaced.rgb_increment, torch.zeros_like(out_replaced.t1))
    m0_exact = bool(torch.equal(mask_zero, out_replaced.t1))

    # RNG-sensitivity evidence: a foreign set_step_seed triple changes the frozen
    # forward, which is exactly why the cache must be replayed under its own triple.
    probe_batch = load_crop(roles[entry["role"]], entry, device)
    guard_gpu()
    probe = {}
    for tag, phase, step in (("cache_triple", "a2_native_cache", positions[0]),
                             ("foreign_triple", "a2_equivalence_probe", 0)):
        study.base.set_step_seed(cfg["seed"], phase, step, probe_batch["sample_id"])
        with torch.no_grad():
            out_probe = study.recurrent_forward(model, task, probe_batch)
        probe[tag] = out_probe.t1.detach()
    rng_sensitivity = dict(
        sample_id=entry["sample_id"],
        t1_max_abs_difference=float((probe["cache_triple"] - probe["foreign_triple"]).abs().max()),
        t1_identical=bool(torch.equal(probe["cache_triple"], probe["foreign_triple"])),
        note="a foreign triple yields a different frozen forward; replaying the cache triple is mandatory",
    )

    stored = torch.load(FINAL, map_location="cpu", weights_only=False)["r2p_and_projections"]
    frozen_mismatch = [n for n, v in model.state_dict().items()
                       if not n.startswith("action_gate.") and not torch.equal(v.cpu(), stored[n])]

    passed = (all(r["raw_cost_allclose"] for r in results)
              and all(v.get("allclose", False) for v in grad_compare.values() if v.get("present", True))
              and all(v["allclose"] for v in param_diff.values())
              and any(v.get("nonzero", False) for v in grad_compare.values())
              and gt_features_unchanged and labels_changed and m0_exact
              and not frozen_mismatch)
    return dict(
        passed=passed, crops=[plan[p]["sample_id"] for p in positions],
        per_crop=results, gradient_comparison=grad_compare,
        parameter_after_one_update=param_diff,
        grad_clip_online=clip, grad_clip_cached=clip2,
        gt_replacement_changes_labels=labels_changed,
        gt_replacement_keeps_deployable_features=gt_features_unchanged,
        mask_zero_exactly_copies_t1=m0_exact,
        rng_sensitivity=rng_sensitivity,
        non_judgement_tensors_still_equal_source=not frozen_mismatch,
        non_judgement_mismatches=frozen_mismatch,
        tolerance=dict(atol=ATOL, rtol=RTOL),
        dtype="float32", autocast=dict(enabled_online="disabled (frozen reuse-fixed-first-round path)",
                                       enabled_cache="disabled (float32 cache)"),
        note="hash inequality vs numeric equivalence distinguished; allclose is not claimed as bit-exact",
        seed_policy=("identical set_step_seed triple as cache generation "
                     "(phase='a2_native_cache', step=plan position); the frozen two-round "
                     "forward is RNG-sensitive, so a different triple legitimately changes t1"),
        test_used=False, validation_used=False)


if __name__ == "__main__":
    main()
