#!/usr/bin/env python3
"""K=1 six-group ablation entry (seed2026, step3000).

Adapted from the ORIGINAL K=1 trainer:
    /media/li/data3/25thstudent/DY/929work/pidsr_k2_929/baseline/k1_run_experiment_original.py
The main computation path is preserved verbatim. This file ONLY adds:

  1. an `ablation` config section with explicit switches
        train_r2p   : bool
        train_netg  : bool
        use_alignment: bool
        bypass_r2p  : bool   (G0: raw polar straight into official TaskNet)
  2. parameter freezing via requires_grad_(False) AND exclusion from the optimizer
     (never simulated by lr=0).
  3. per-group preflight gradient assertions (the original demanded R2P, projections
     and net_g gradients all be non-zero; that is invalid for single-module groups).
  4. evaluate CLI accepting an explicit checkpoint path or the name `step_3000`.
  5. an init/final frozen-parameter audit and a step-0 forward consistency check.

Data loading, schedules, metrics (psnr/ssim), the loss, the RNG rule and the
training loop itself are NOT rewritten: they are the original implementations.

Groups:
  G0_official                bypass_r2p=True,  no training
  G1_pretrained_fixed        e42 R2P + official TaskNet, no training
  G2_netg_only               train_netg only
  G3_r2p_only                train_r2p only   (task gradient still flows back
                             through the frozen TaskNet to R2P; the TaskNet
                             forward is never wrapped in no_grad nor detached)
  G4_joint_no_align          train_r2p + train_netg, alignment weight exactly 0
  G5_joint_align_reference   reuse the existing K1 seed2026 step3000 checkpoint
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def add_source_path(config: Mapping[str, Any]) -> None:
    source = Path(config["paths"]["source_root"])
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def state_sha256(state: Mapping[str, torch.Tensor]) -> dict[str, str]:
    return {name: tensor_sha256(value) for name, value in state.items()}


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def append_csv(path: Path, values: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(values)


def load_reference_r2p(path: Path) -> type[nn.Module]:
    # Identical to the original K=1 trainer: module name and sys.modules
    # registration are preserved so the reference R2P loads exactly as before.
    import importlib.util

    spec = importlib.util.spec_from_file_location("pidsr_reference_r2p", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load R2P reference module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.R2P


@dataclass
class JointOutput:
    refined_polar: torch.Tensor
    projected_features: dict[str, torch.Tensor]
    raw_features: dict[str, torch.Tensor]
    delta: torch.Tensor
    gate: torch.Tensor


class JointR2P(nn.Module):
    """Exact pretrained R2P plus external lightweight alignment projections."""

    def __init__(
        self,
        reference_path: Path,
        checkpoint_path: Path,
        width: int,
        max_residual: float,
        task_channels: tuple[int, int, int],
    ) -> None:
        super().__init__()
        r2p_type = load_reference_r2p(reference_path)
        self.r2p = r2p_type(width=width, max_residual=max_residual)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if int(checkpoint["r2p_width"]) != width:
            raise ValueError("PIDSR checkpoint width does not match the configured R2P")
        if not math.isclose(float(checkpoint["max_residual"]), max_residual):
            raise ValueError(
                "PIDSR checkpoint residual bound does not match the configured R2P"
            )
        incompatibility = self.r2p.load_state_dict(checkpoint["r2p"], strict=True)
        if incompatibility.missing_keys or incompatibility.unexpected_keys:
            raise AssertionError(f"Strict R2P load failed: {incompatibility}")
        self.projections = nn.ModuleDict(
            {
                f"P{level}": nn.Conv2d(width, channels, 1, bias=False)
                for level, channels in enumerate(task_channels, start=1)
            }
        )
        self.pretrain_epoch = int(checkpoint["epoch"])

    def forward(self, polar: torch.Tensor, condition: torch.Tensor) -> JointOutput:
        refined, features = self.r2p(condition, polar, return_features=True)
        fused = features["fused"]
        projected = {name: head(fused) for name, head in self.projections.items()}
        return JointOutput(
            refined_polar=refined,
            projected_features=projected,
            raw_features=features,
            delta=features["delta"],
            gate=features["gate"],
        )


def feature_alignment_loss(
    projected: Mapping[str, torch.Tensor], task_features: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    losses = []
    for level in (1, 2, 3):
        source = projected[f"P{level}"]
        target = task_features[f"T{level}"].detach()
        if source.shape[-2:] != target.shape[-2:]:
            source = F.interpolate(
                source, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
        if source.shape != target.shape:
            raise ValueError(
                f"Alignment shape mismatch: {source.shape} versus {target.shape}"
            )
        losses.append(1.0 - F.cosine_similarity(source, target, dim=1).mean())
    return torch.stack(losses).mean()


def gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    values = [p.grad.detach().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def has_any_grad(parameters: Iterable[nn.Parameter]) -> bool:
    return any(p.grad is not None for p in parameters)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def stable_seed(seed: int, phase: str, step: int, sample_ids: Iterable[str]) -> int:
    payload = f"{seed}:{phase}:{step}:{'|'.join(sample_ids)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31)


def set_step_seed(seed: int, phase: str, step: int, sample_ids: Iterable[str]) -> int:
    value = stable_seed(seed, phase, step, sample_ids)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    return value


def build_base_config(config: Mapping[str, Any]):
    from rgb_final_confirmation.config import ExperimentConfig, load_config

    base = load_config(config["paths"]["base_config"])
    raw = json.loads(json.dumps(base.raw))
    output_root = str(Path(config["paths"]["output_root"]).resolve())
    raw["experiment_root"] = output_root
    raw["paths"]["outputs"] = output_root
    raw["paths"]["manifest"] = str(Path(config["paths"]["manifest"]).resolve())
    raw["paths"]["schedule"] = str(Path(config["paths"]["schedule"]).resolve())
    raw["paths"]["validation_proxy"] = str(
        Path(config["paths"]["validation_proxy"]).resolve()
    )
    raw["paths"]["weight_audit"] = str(
        Path(output_root) / "config" / "weight_audit.json"
    )
    return ExperimentConfig(raw=raw, source=Path(config["paths"]["base_config"]))


def build_components(config: Mapping[str, Any], device: torch.device):
    from rgb_final_confirmation.losses import build_official_task_loss
    from rgb_final_confirmation.polarfree import OfficialPolarFreeTask

    # Construction order is IDENTICAL to the original so that any construction-time
    # RNG consumption is unchanged between groups.
    model_values = config["model"]
    model = JointR2P(
        Path(config["paths"]["reference_r2p"]),
        Path(config["paths"]["pretrained_r2p"]),
        int(model_values["r2p_width"]),
        float(model_values["max_residual"]),
        tuple(int(value) for value in model_values["task_feature_channels"]),
    ).to(device)
    base_config = build_base_config(config)
    task = OfficialPolarFreeTask(base_config, device)
    task.model.requires_grad_(False)  # diffusion prior: frozen in EVERY group
    task.net_g.requires_grad_(True)
    task.eval()
    criterion = (
        build_official_task_loss(base_config).to(device).requires_grad_(False).eval()
    )

    ablation = config["ablation"]
    train_r2p = bool(ablation["train_r2p"])
    train_netg = bool(ablation["train_netg"])
    use_alignment = bool(ablation["use_alignment"])

    if not train_r2p:
        model.r2p.requires_grad_(False)
    # Alignment projections are only trainable when alignment is actually used.
    if not use_alignment:
        model.projections.requires_grad_(False)
    if not train_netg:
        task.net_g.requires_grad_(False)

    # Assert the declared trainable set matches reality (per group, not the
    # original's rigid "only net_g" rule).
    expected: set[int] = set()
    if train_r2p:
        expected |= {id(p) for p in model.r2p.parameters()}
    if use_alignment:
        expected |= {id(p) for p in model.projections.parameters()}
    if train_netg:
        expected |= {id(p) for p in task.net_g.parameters()}
    actual = {
        id(p)
        for module in (model, task)
        for p in module.parameters()
        if p.requires_grad
    }
    if expected != actual:
        raise AssertionError(
            "Trainable parameter set differs from the declared ablation switches"
        )
    return model, task, criterion, base_config


def scheduled_loader(
    config: Mapping[str, Any], base_config, end_step: int, start_step: int = 0
):
    from rgb_final_confirmation.data import (
        PairedBranchDataset,
        load_manifest,
        load_training_schedule,
        select_rows,
    )

    rows = load_manifest(config["paths"]["manifest"])
    train_rows = select_rows(rows, "train")
    schedule = load_training_schedule(config["paths"]["schedule"])
    batch_size = int(config["data"]["batch_size"])
    items = list(schedule["items"])[start_step * batch_size : end_step * batch_size]
    if len(items) != (end_step - start_step) * batch_size:
        raise ValueError("Frozen training schedule is shorter than requested")
    dataset = PairedBranchDataset(
        train_rows,
        group="A1",
        crop_size=int(config["data"]["crop_size"]),
        schedule=items,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["data"]["workers"]),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def proxy_loader(config: Mapping[str, Any], base_config):
    from rgb_final_confirmation.data import (
        PairedBranchDataset,
        load_manifest,
        load_validation_proxy_rows,
    )

    rows = load_manifest(config["paths"]["manifest"])
    selected = load_validation_proxy_rows(base_config, rows)
    return DataLoader(
        PairedBranchDataset(selected, group="A1", crop_size=None),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )


def split_loader(config: Mapping[str, Any], split: str):
    from rgb_final_confirmation.data import (
        PairedBranchDataset,
        load_manifest,
        select_rows,
    )

    rows = select_rows(load_manifest(config["paths"]["manifest"]), split)
    return DataLoader(
        PairedBranchDataset(rows, group="A1", crop_size=None),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )


def unpack_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "polar": batch["polar"].to(device),
        "condition": batch["condition"].to(device),
        "official_rgb": batch["official_rgb"].to(device),
        "target": batch["target"].to(device),
        "sample_id": list(batch["sample_id"]),
        "scene": list(batch["scene"]),
    }


class BypassOutput:
    """G6: raw polar goes straight into the official TaskNet; R2P never runs.

    `refined_polar` IS the raw polar tensor (same object, no copy, no clamp, no
    residual). `delta`, `gate` and `projected_features` are None by construction
    because the R2P module is never invoked on this path.
    """

    __slots__ = ("refined_polar", "delta", "gate", "projected_features")

    def __init__(self, polar: torch.Tensor) -> None:
        self.refined_polar = polar
        self.delta = None
        self.gate = None
        self.projected_features = None


def forward_losses(model, task, criterion, batch: Mapping[str, Any], config=None):
    bypass = bool((config or {}).get("ablation", {}).get("bypass_r2p", False))
    if bypass:
        # R2P is not called at all, so it cannot enter the autograd graph.
        output = BypassOutput(batch["polar"])
        prediction, task_features = task(output.refined_polar, batch["official_rgb"], None)
        task_loss = criterion(prediction, batch["target"])
        if not torch.isfinite(task_loss):
            raise FloatingPointError("Non-finite training loss")
        return output, prediction, task_loss, None
    output = model(batch["polar"], batch["condition"])
    prediction, task_features = task(output.refined_polar, batch["official_rgb"], None)
    task_loss = criterion(prediction, batch["target"])
    alignment_loss = feature_alignment_loss(output.projected_features, task_features)
    if not torch.isfinite(task_loss) or not torch.isfinite(alignment_loss):
        raise FloatingPointError("Non-finite training loss")
    return output, prediction, task_loss, alignment_loss


# --------------------------------------------------------------------------- #
# Evaluation                                                                   #
# --------------------------------------------------------------------------- #


def validate(model, task, loader, device: torch.device, seed: int, config=None):
    """Original validate(), extended only with sample_id/index recording.

    The RNG rule is untouched: set_step_seed(seed, "validation", index, sample_id).
    The phase literal stays "validation" for BOTH validation and test splits, as in
    the original source (k1_run_experiment_original.py:411).
    """
    from rgb_final_confirmation.evaluation import psnr, ssim

    model.eval()
    task.eval()
    bypass = bool((config or {}).get("ablation", {}).get("bypass_r2p", False))
    rows = []
    with torch.no_grad():
        for index, raw_batch in enumerate(loader):
            batch = unpack_batch(raw_batch, device)
            set_step_seed(seed, "validation", index, batch["sample_id"])
            if bypass:
                # G0: raw polar goes straight into the official TaskNet.
                refined = batch["polar"]
            else:
                refined = model(batch["polar"], batch["condition"]).refined_polar
            prediction, _ = task(refined, batch["official_rgb"], None)
            rows.append(
                {
                    "index": index,
                    "sample_id": batch["sample_id"][0],
                    "scene": batch["scene"][0],
                    "psnr": psnr(prediction, batch["target"]),
                    "ssim": ssim(prediction, batch["target"]),
                }
            )
    by_scene: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        by_scene[str(row["scene"])].append(row)
    scene_psnr = [
        float(np.mean([row["psnr"] for row in values])) for values in by_scene.values()
    ]
    scene_ssim = [
        float(np.mean([row["ssim"] for row in values])) for values in by_scene.values()
    ]
    return {
        "frames": float(len(rows)),
        "scenes": float(len(by_scene)),
        "image_mean_psnr": float(np.mean([row["psnr"] for row in rows])),
        "image_mean_ssim": float(np.mean([row["ssim"] for row in rows])),
        "scene_macro_psnr": float(np.mean(scene_psnr)),
        "scene_macro_ssim": float(np.mean(scene_ssim)),
        "per_image": rows,
        "scene_summary": {
            scene: {
                "frames": float(len(values)),
                "image_mean_psnr": float(np.mean([r["psnr"] for r in values])),
                "image_mean_ssim": float(np.mean([r["ssim"] for r in values])),
            }
            for scene, values in sorted(by_scene.items())
        },
    }


def checkpoint_payload(
    model, task, optimizer, step: int, coefficient: float, metrics=None, group: str = ""
):
    return {
        "schema_version": 2,
        "method": f"k1_ablation:{group}",
        "step": int(step),
        "lambda_feature_alignment": float(coefficient),
        "r2p_and_projections": model.state_dict(),
        "tasknet_net_g": task.net_g.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "validation": dict(metrics or {}),
    }


# --------------------------------------------------------------------------- #
# Init / final frozen audit                                                    #
# --------------------------------------------------------------------------- #


def frozen_audit(model, task) -> dict[str, Any]:
    """Hash every parameter AND buffer of the declared-frozen modules."""

    def block(module: nn.Module) -> dict[str, str]:
        out = {f"param:{n}": tensor_sha256(p) for n, p in module.named_parameters()}
        out.update({f"buffer:{n}": tensor_sha256(b) for n, b in module.named_buffers()})
        return out

    return {
        "r2p": block(model.r2p),
        "projections": block(model.projections),
        "net_g": block(task.net_g),
        "prior": block(task.model),
    }


def diff_audit(before: dict, after: dict) -> dict[str, Any]:
    changed: dict[str, list[str]] = {}
    for module, entries in before.items():
        diffs = [k for k, v in entries.items() if after.get(module, {}).get(k) != v]
        changed[module] = sorted(diffs)
    return {
        "changed_keys": changed,
        "n_changed": {m: len(v) for m, v in changed.items()},
    }


# --------------------------------------------------------------------------- #
# Preflight (per-group gradient assertions)                                    #
# --------------------------------------------------------------------------- #


def run_preflight(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    output_root = Path(config["paths"]["output_root"])
    set_seed(int(config["seed"]))
    model, task, criterion, base_config = build_components(config, device)
    loader = scheduled_loader(config, base_config, 1)
    batch = unpack_batch(next(iter(loader)), device)
    set_step_seed(int(config["seed"]), "preflight", 1, batch["sample_id"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    bypass = bool(config["ablation"].get("bypass_r2p", False))
    output, _, task_loss, alignment_loss = forward_losses(
        model, task, criterion, batch, config
    )
    coefficient = alignment_coefficient(config, output_root)
    loss = task_loss if alignment_loss is None else task_loss + coefficient * alignment_loss
    loss.backward()

    r2p_grad = gradient_norm(model.r2p.parameters())
    proj_grad = gradient_norm(model.projections.parameters())
    netg_grad = gradient_norm(task.net_g.parameters())
    prior_grads = [
        p.grad
        for name, p in task.model.named_parameters()
        if not name.startswith("net_g.")
    ]
    r2p_has = has_any_grad(model.r2p.parameters())
    proj_has = has_any_grad(model.projections.parameters())
    netg_has = has_any_grad(task.net_g.parameters())

    # ------------------------------------------------------------------ #
    # G6 proof that the training forward really bypasses R2P.
    # Poison every R2P parameter with NaN and re-run the SAME step (same
    # explicit diffusion seed). If R2P were anywhere on the path the loss
    # would become NaN. State is restored bit-exactly afterwards.
    # ------------------------------------------------------------------ #
    bypass_probe = None
    if bypass:
        saved = {k: v.detach().clone() for k, v in model.r2p.state_dict().items()}
        try:
            with torch.no_grad():
                for p in model.r2p.parameters():
                    p.fill_(float("nan"))
            set_step_seed(int(config["seed"]), "preflight", 1, batch["sample_id"])
            _, _, probe_loss, _ = forward_losses(model, task, criterion, batch, config)
            probe_value = float(probe_loss.detach())
            reference_value = float(task_loss.detach())
            bypass_probe = {
                "task_loss_with_nan_r2p": probe_value,
                "task_loss_reference": reference_value,
                "identical": probe_value == reference_value,
                "refined_polar_is_raw_input": bool(output.refined_polar is batch["polar"]),
            }
        finally:
            with torch.no_grad():
                model.r2p.load_state_dict(saved)
        if not bypass_probe["identical"]:
            raise AssertionError(
                "bypass_r2p is a config flag only: poisoning R2P changed the loss, "
                "so R2P is still on the training path"
            )

    payload = {
        "bypass_r2p": bypass,
        "bypass_nan_probe": bypass_probe,
        "group": config["ablation"]["group"],
        "strict_pretrained_load": True,
        "pretrain_epoch": model.pretrain_epoch,
        "pretrained_sha256": sha256_file(Path(config["paths"]["pretrained_r2p"])),
        "output_shape": list(output.refined_polar.shape),
        "output_min": float(output.refined_polar.min().detach()),
        "output_max": float(output.refined_polar.max().detach()),
        "task_loss": float(task_loss.detach()),
        "alignment_loss": (
            None if alignment_loss is None else float(alignment_loss.detach())
        ),
        "lambda_feature_alignment": float(coefficient),
        "r2p_grad_norm": r2p_grad,
        "projection_grad_norm": proj_grad,
        "tasknet_grad_norm": netg_grad,
        "r2p_has_grad": r2p_has,
        "projection_has_grad": proj_has,
        "tasknet_has_grad": netg_has,
        "frozen_prior_has_grad": any(v is not None for v in prior_grads),
        "cuda_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }

    if payload["frozen_prior_has_grad"]:
        raise AssertionError("Frozen latent/diffusion prior received parameter gradients")

    # Per-group assertions (the original demanded all three be non-zero).
    train_r2p = bool(config["ablation"]["train_r2p"])
    train_netg = bool(config["ablation"]["train_netg"])
    use_alignment = bool(config["ablation"]["use_alignment"])
    problems = []
    if train_r2p and not (r2p_has and np.isfinite(r2p_grad) and r2p_grad > 0):
        problems.append("r2p is declared trainable but has no finite non-zero gradient")
    if not train_r2p and r2p_has:
        problems.append("r2p is declared frozen but received gradients")
    if train_netg and not (netg_has and np.isfinite(netg_grad) and netg_grad > 0):
        problems.append("net_g is declared trainable but has no finite non-zero gradient")
    if not train_netg and netg_has:
        problems.append("net_g is declared frozen but received gradients")
    if use_alignment and not (proj_has and proj_grad > 0):
        problems.append("alignment is on but projections have no gradient")
    if not use_alignment and proj_has:
        problems.append("alignment is off but projections received gradients")
    payload["preflight_problems"] = problems
    if problems:
        raise AssertionError("Preflight gradient assertions failed: " + "; ".join(problems))

    # step-0 initialisation audit (frozen modules must match their source of truth)
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "diagnostics" / "init_audit.json", frozen_audit(model, task))
    atomic_json(output_root / "diagnostics" / "preflight.json", payload)
    return payload


def alignment_coefficient(config: Mapping[str, Any], output_root: Path) -> float:
    """G2/G3/G4: exactly 0. G5: reuse the original calibration, never re-calibrate."""
    if not bool(config["ablation"]["use_alignment"]):
        return 0.0
    calibration_path = output_root / "config" / "alignment_calibration.json"
    if not calibration_path.is_file():
        raise FileNotFoundError("Alignment enabled but calibration JSON is missing")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    return float(calibration["lambda_feature_alignment"])


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #


def build_optimizer(config: Mapping[str, Any], model, task):
    """Only declared-trainable groups enter the optimizer (never lr=0 fakes)."""
    groups = []
    if bool(config["ablation"]["train_r2p"]):
        groups.append(
            {
                "params": list(model.r2p.parameters()),
                "lr": float(config["optimization"]["lr_r2p"]),
                "name": "r2p",
            }
        )
    if bool(config["ablation"]["use_alignment"]):
        groups.append(
            {
                "params": list(model.projections.parameters()),
                "lr": float(config["optimization"]["lr_r2p"]),
                "name": "alignment_projections",
            }
        )
    if bool(config["ablation"]["train_netg"]):
        groups.append(
            {
                "params": list(task.net_g.parameters()),
                "lr": float(config["optimization"]["lr_tasknet"]),
                "name": "tasknet_net_g",
            }
        )
    if not groups:
        raise AssertionError("No trainable groups declared")
    return torch.optim.Adam(
        groups,
        betas=tuple(float(v) for v in config["optimization"]["betas"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )


def clip_groups(config, model, task):
    clip = float(config["optimization"]["grad_clip"])
    if bool(config["ablation"]["train_r2p"]):
        torch.nn.utils.clip_grad_norm_(model.r2p.parameters(), clip)
    if bool(config["ablation"]["use_alignment"]):
        torch.nn.utils.clip_grad_norm_(model.projections.parameters(), clip)
    if bool(config["ablation"]["train_netg"]):
        torch.nn.utils.clip_grad_norm_(task.net_g.parameters(), clip)


def run_training(
    config: Mapping[str, Any], device: torch.device, steps: int, resume: bool
) -> dict[str, Any]:
    output_root = Path(config["paths"]["output_root"])
    group = config["ablation"]["group"]
    set_seed(int(config["seed"]))
    model, task, criterion, base_config = build_components(config, device)
    optimizer = build_optimizer(config, model, task)
    coefficient = alignment_coefficient(config, output_root)

    start_step = 0
    latest_path = output_root / "checkpoints" / "last.pt"
    if resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint is missing: {latest_path}")
        state = torch.load(latest_path, map_location=device, weights_only=False)
        if not math.isclose(float(state["lambda_feature_alignment"]), coefficient):
            raise ValueError("Alignment coefficient changed before resume")
        model.load_state_dict(state["r2p_and_projections"], strict=True)
        task.net_g.load_state_dict(state["tasknet_net_g"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
    elif latest_path.exists():
        raise FileExistsError(
            "Existing training checkpoint found; use --resume or a new output root"
        )
    if steps <= start_step:
        raise ValueError(
            f"Requested step {steps} must exceed checkpoint step {start_step}"
        )

    loader = scheduled_loader(config, base_config, steps, start_step)
    validation_loader = proxy_loader(config, base_config)

    initial_task = {
        name: value.detach().cpu().float().clone()
        for name, value in task.net_g.named_parameters()
    }
    if resume:
        initial_task = torch.load(
            output_root / "config" / "tasknet_initial.pt", map_location="cpu",
            weights_only=True,
        )
    else:
        atomic_torch_save(output_root / "config" / "tasknet_initial.pt", initial_task)
        atomic_json(output_root / "diagnostics" / "init_audit.json",
                    frozen_audit(model, task))

    history_path = output_root / "metrics" / "proxy_validation.json"
    history = (
        json.loads(history_path.read_text(encoding="utf-8"))
        if history_path.is_file()
        else []
    )
    validation_steps = {int(v) for v in config["optimization"]["validation_steps"]}

    log_path = output_root / "logs" / "train.csv"
    iterator = iter(loader)
    started = time.perf_counter()
    step_times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(start_step + 1, steps + 1):
        tick = time.perf_counter()
        batch = unpack_batch(next(iterator), device)
        noise_seed = set_step_seed(
            int(config["seed"]), "train", step, batch["sample_id"]
        )
        model.train()
        task.eval()  # original K1 keeps TaskNet in eval mode even when net_g trains
        optimizer.zero_grad(set_to_none=True)
        output, _, task_loss, alignment_loss = forward_losses(
            model, task, criterion, batch, config
        )
        loss = (
            task_loss if alignment_loss is None
            else task_loss + coefficient * alignment_loss
        )
        loss.backward()
        diagnostics = {
            "step": step,
            "diffusion_seed": noise_seed,
            "loss": float(loss.detach()),
            "task_loss": float(task_loss.detach()),
            "feature_alignment_loss": (
                0.0 if alignment_loss is None else float(alignment_loss.detach())
            ),
            "r2p_grad_norm": gradient_norm(model.r2p.parameters()),
            "projection_grad_norm": gradient_norm(model.projections.parameters()),
            "tasknet_grad_norm": gradient_norm(task.net_g.parameters()),
        }
        # Only meaningful when R2P is on the path (None under bypass_r2p).
        if output.delta is not None:
            diagnostics["delta_abs_mean"] = float(output.delta.detach().abs().mean())
            diagnostics["gate_mean"] = float(output.gate.detach().mean())
        clip_groups(config, model, task)
        optimizer.step()
        diagnostics["step_seconds"] = time.perf_counter() - tick
        step_times.append(float(diagnostics["step_seconds"]))
        if not all(np.isfinite(float(v)) for v in diagnostics.values()):
            raise FloatingPointError(f"Non-finite training diagnostics: {diagnostics}")
        append_csv(log_path, diagnostics)

        metrics = None
        checkpoint_interval = int(config["optimization"]["checkpoint_interval"])
        should_checkpoint = step % checkpoint_interval == 0 or step == steps
        if step in validation_steps or step == steps:
            raw_metrics = validate(
                model, task, validation_loader, device, int(config["seed"]), config
            )
            metrics = {k: v for k, v in raw_metrics.items()
                       if k not in ("per_image", "scene_summary")}
            metrics["step"] = float(step)
            history.append(metrics)
            atomic_json(history_path, history)
        if should_checkpoint:
            atomic_torch_save(
                latest_path,
                checkpoint_payload(model, task, optimizer, step, coefficient, metrics, group),
            )
            atomic_torch_save(
                output_root / "checkpoints" / f"step_{step}.pt",
                checkpoint_payload(model, task, optimizer, step, coefficient, metrics, group),
            )
        if step % 200 == 0:
            print(f"[{group}] step {step}/{steps}", flush=True)

    timing = {
        "start_step": start_step,
        "end_step": steps,
        "wall_seconds": time.perf_counter() - started,
        "mean_step_seconds_excluding_first_20": float(
            np.mean(step_times[20:] or step_times)
        ),
        "cuda_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }
    atomic_json(
        output_root / "diagnostics" / f"timing_{start_step}_{steps}.json", timing
    )
    # final frozen audit
    final_audit = frozen_audit(model, task)
    atomic_json(output_root / "diagnostics" / "final_audit.json", final_audit)
    init = json.loads(
        (output_root / "diagnostics" / "init_audit.json").read_text(encoding="utf-8")
    )
    atomic_json(
        output_root / "diagnostics" / "frozen_diff.json",
        diff_audit(init, final_audit),
    )
    return {"checkpoint": str(latest_path), "validation": history[-1], "timing": timing}


# --------------------------------------------------------------------------- #
# Evaluation CLI (accepts explicit path or step_3000)                          #
# --------------------------------------------------------------------------- #


def resolve_checkpoint(config, name: str) -> Path:
    output_root = Path(config["paths"]["output_root"])
    candidate = Path(name)
    if candidate.is_file():
        return candidate
    candidate = output_root / "checkpoints" / f"{name}.pt"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Checkpoint not found: {name}")


def run_evaluation(
    config: Mapping[str, Any],
    device: torch.device,
    checkpoint_name: str,
    split: str,
    limit: int = 0,
) -> dict[str, Any]:
    """limit>0 restricts to the first `limit` images (smoke test only).

    The RNG rule is unchanged: index is the loader index, so a limited run uses
    exactly the same stream as the first `limit` images of a full run.
    """
    output_root = Path(config["paths"]["output_root"])
    set_seed(int(config["seed"]))
    model, task, _, _ = build_components(config, device)
    if checkpoint_name in ("none", "official", "original", "step_0"):
        # G0/G1: no checkpoint to load; e42 R2P and official TaskNet are used
        # exactly as constructed from their original sources.
        checkpoint_path = Path("(no checkpoint; original e42 R2P + official TaskNet)")
        state = {"step": 0}
    else:
        checkpoint_path = resolve_checkpoint(config, checkpoint_name)
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["r2p_and_projections"], strict=True)
        task.net_g.load_state_dict(state["tasknet_net_g"], strict=True)
    loader = split_loader(config, split)
    if limit > 0:
        from itertools import islice

        loader = list(islice(loader, limit))
    metrics = validate(
        model, task, loader, device, int(config["seed"]), config
    )
    payload = {
        "group": config["ablation"]["group"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_name": checkpoint_name,
        "checkpoint_step": int(state["step"]),
        "split": split,
        "seed": int(config["seed"]),
        **metrics,
    }
    atomic_json(
        output_root / "metrics" / f"{Path(checkpoint_name).stem}_{split}.json", payload
    )
    summary = {k: v for k, v in payload.items() if k != "per_image"}
    return {"summary": summary, "written": str(
        output_root / "metrics" / f"{Path(checkpoint_name).stem}_{split}.json")}


# --------------------------------------------------------------------------- #
# step-0 forward consistency check                                             #
# --------------------------------------------------------------------------- #


def run_forwardcheck(
    config: Mapping[str, Any], device: torch.device, others: list[str], limit: int
) -> dict[str, Any]:
    """Compare this group's step-0 forward output against other configs.

    Runs on a fixed small prefix of the validation split with the standard
    set_step_seed(seed, "validation", index, sample_id) stream, same device and
    precision. Reports actual max/mean absolute differences and a tolerance.
    """
    set_seed(int(config["seed"]))
    model, task, _, _ = build_components(config, device)
    model.eval()
    task.eval()
    bypass = bool(config["ablation"]["bypass_r2p"])

    values: dict[str, list[torch.Tensor]] = {}
    ids: list[str] = []
    with torch.no_grad():
        for index, raw in enumerate(split_loader(config, "validation")):
            if index >= limit:
                break
            batch = unpack_batch(raw, device)
            set_step_seed(int(config["seed"]), "validation", index, batch["sample_id"])
            ids.append(batch["sample_id"][0])
            refined = batch["polar"] if bypass else model(
                batch["polar"], batch["condition"]
            ).refined_polar
            prediction, _ = task(refined, batch["official_rgb"], None)
            values.setdefault(config["ablation"]["group"], []).append(
                prediction.detach().clone()
            )
            del prediction, refined
            torch.cuda.empty_cache()

    result: dict[str, Any] = {
        "group": config["ablation"]["group"],
        "n_images": len(ids),
        "sample_ids": ids,
        "tolerance": 1e-6,
        "comparisons": {},
    }
    for other_path in others:
        other_cfg = read_yaml(Path(other_path))
        add_source_path(other_cfg)
        o_model, o_task, _, _ = build_components(other_cfg, device)
        o_model.eval()
        o_task.eval()
        o_bypass = bool(other_cfg["ablation"]["bypass_r2p"])
        other_name = other_cfg["ablation"]["group"]
        o_values: list[torch.Tensor] = []
        with torch.no_grad():
            for index, raw in enumerate(split_loader(other_cfg, "validation")):
                if index >= limit:
                    break
                batch = unpack_batch(raw, device)
                set_step_seed(
                    int(other_cfg["seed"]), "validation", index, batch["sample_id"]
                )
                refined = batch["polar"] if o_bypass else o_model(
                    batch["polar"], batch["condition"]
                ).refined_polar
                pred, _ = o_task(refined, batch["official_rgb"], None)
                o_values.append(pred.detach().clone())
                del pred, refined
                torch.cuda.empty_cache()
        maxes, means = [], []
        for a, b in zip(values[config["ablation"]["group"]], o_values):
            d = (a - b).abs()
            maxes.append(float(d.max()))
            means.append(float(d.mean()))
        result["comparisons"][other_name] = {
            "other_config": other_path,
            "per_image_max_abs_diff": maxes,
            "per_image_mean_abs_diff": means,
            "max_abs_diff_over_images": float(max(maxes)),
            "mean_abs_diff_over_images": float(np.mean(means)),
            "passed": bool(max(maxes) < result["tolerance"]),
        }
        del o_model, o_task
        torch.cuda.empty_cache()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    train = commands.add_parser("train")
    train.add_argument("--steps", type=int, required=True)
    train.add_argument("--resume", action="store_true")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", default="step_3000")
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate.add_argument("--limit", type=int, default=0, help="smoke test only")
    fwd = commands.add_parser("forwardcheck")
    fwd.add_argument("--others", nargs="*", default=[])
    fwd.add_argument("--limit", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = read_yaml(args.config.resolve())
    add_source_path(config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.command == "preflight":
        result = run_preflight(config, device)
    elif args.command == "train":
        result = run_training(config, device, args.steps, args.resume)
    elif args.command == "evaluate":
        result = run_evaluation(config, device, args.checkpoint, args.split, args.limit)
    elif args.command == "forwardcheck":
        result = run_forwardcheck(config, device, args.others, args.limit)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
