from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("schema_version") != 2 or not isinstance(config.get("recurrent", {}).get("feedback"), bool):
        raise ValueError("Expected K2 schema_version=2 and boolean recurrent.feedback")
    if config["data"]["condition"] != "raw_mixed_rgb":
        raise ValueError("K2 protocol requires fixed original RGB")
    if config["recurrent"]["round_loss_weights"] != [0.5, 1.0]:
        raise ValueError("Locked first-version loss weights are [0.5, 1.0]")
    return config


def add_source_path(config: Mapping[str, Any]) -> None:
    source = str(Path(config["paths"]["source_root"]).resolve())
    if source not in sys.path:
        sys.path.insert(0, source)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values))
        if not exists:
            writer.writeheader()
        writer.writerow(values)


def load_reference_r2p(path: Path) -> type[nn.Module]:
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
        self.initial_r2p = {n: p.detach().cpu().clone() for n, p in self.r2p.named_parameters()}

    def enable_recurrence(self, channels, width, enabled, weights):
        self.feedback_enabled = bool(enabled)
        self.round_weights = tuple(float(x) for x in weights)
        # Keep both variants' initialization identical. Unused adapter parameters
        # in no-feedback never receive gradients and Adam does not update them.
        self.feedback_adapter = nn.ModuleDict({
            f"T{i}": nn.Conv2d(c, width, 1, bias=False)
            for i, c in enumerate(channels, 1)
        })
        self.feedback_scale = nn.Parameter(torch.zeros(()))

    def forward(self, polar: torch.Tensor, condition: torch.Tensor,
                feedback=None, project_features=True) -> JointOutput:
        if feedback is not None:
            r = self.r2p
            r._validate_inputs(condition, polar)
            rgb = r.rgb_encoder(condition)
            pol = r.polar_encoder(polar)
            gate = torch.sigmoid(r.gate(torch.cat((pol, rgb), dim=1)))
            fused = gate * rgb + (1 - gate) * pol
            fb = sum(F.interpolate(head(feedback[name]), size=fused.shape[-2:],
                                   mode="bilinear", align_corners=False)
                     for name, head in self.feedback_adapter.items()) / 3
            fused = fused + self.feedback_scale * fb
            gamma, beta = r.film(rgb).chunk(2, dim=1)
            modulated = pol * (1 + .10 * torch.tanh(gamma)) + .10 * torch.tanh(beta)
            delta = r.max_residual * torch.tanh(r.body(torch.cat((modulated, fused), dim=1)))
            refined = (polar + delta).clamp(0, 1)
            features = {"rgb": rgb, "polar": pol, "gate": gate, "fused": fused,
                        "modulated_polar": modulated, "delta": delta}
        else:
            refined, features = self.r2p(condition, polar, return_features=True)
        fused = features["fused"]
        projected = {name: head(fused) for name, head in self.projections.items()} if project_features else {}
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


def autograd_norm(loss: torch.Tensor, parameters: tuple[nn.Parameter, ...]) -> float:
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    values = [
        gradient.detach().square().sum()
        for gradient in gradients
        if gradient is not None
    ]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


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
    task.model.requires_grad_(False)
    task.net_g.requires_grad_(True)
    task.eval()
    criterion = (
        build_official_task_loss(base_config).to(device).requires_grad_(False).eval()
    )
    trainable_task = {id(parameter) for parameter in task.net_g.parameters()}
    actual_trainable = {
        id(parameter) for parameter in task.parameters() if parameter.requires_grad
    }
    if trainable_task != actual_trainable:
        raise AssertionError("Only PolarFree net_g may be trainable")
    # Adapter initialization must not perturb original model initialization/RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(config["seed"]) + 91000)
        model.enable_recurrence(tuple(model_values["task_feature_channels"]),
                                int(model_values["r2p_width"]),
                                config["recurrent"]["feedback"],
                                config["recurrent"]["round_loss_weights"])
    model.to(device)
    model.activation_checkpoint = bool(config.get("execution", {}).get("activation_checkpoint", False))
    model.protocol_id = protocol_id(config)
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


def recurrent_forward(model, task, batch, need_alignment=False):
    first = model(batch["polar"], batch["condition"], project_features=need_alignment)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    def task_forward(polar, rgb):
        if torch.is_grad_enabled() and getattr(model, "activation_checkpoint", False):
            return checkpoint(task, polar, rgb, None, use_reentrant=False, preserve_rng_state=True)
        return task(polar, rgb, None)
    if not torch.is_grad_enabled():
        first.raw_features.clear()
    t1, f1 = task_forward(first.refined_polar, batch["official_rgb"])
    second = model(first.refined_polar, batch["condition"],
                   feedback=f1 if model.feedback_enabled else None,
                   project_features=False)
    if not torch.is_grad_enabled():
        second.raw_features.clear()
        if not need_alignment:
            f1.clear()
    # Reuse the complete diffusion RNG stream, not just one random seed.
    devices = list(range(torch.cuda.device_count())) if cuda_rng is not None else []
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        t2, _ = task_forward(second.refined_polar, batch["official_rgb"])
    second.first = first
    second.t1, second.t2 = t1, t2
    second.first_task_features = f1
    return second


def forward_losses(model, task, criterion, batch: Mapping[str, Any]):
    output = recurrent_forward(model, task, batch, need_alignment=True)
    output.loss1 = criterion(output.t1, batch["target"])
    output.loss2 = criterion(output.t2, batch["target"])
    task_loss = model.round_weights[0] * output.loss1 + model.round_weights[1] * output.loss2
    alignment_loss = feature_alignment_loss(output.first.projected_features,
                                            output.first_task_features)
    if not torch.isfinite(task_loss) or not torch.isfinite(alignment_loss):
        raise FloatingPointError("Non-finite training loss")
    return output, output.t2, task_loss, alignment_loss


def protocol_id(config):
    data = json.loads(json.dumps(config))
    data["paths"].pop("output_root", None)
    data["code_sha256"] = sha256_file(Path(__file__))
    for key in ("pretrained_r2p", "reference_r2p", "manifest", "schedule", "validation_proxy", "base_config"):
        data[key + "_sha256"] = sha256_file(Path(config["paths"][key]))
    source = Path(config["paths"]["source_root"]) / "rgb_final_confirmation"
    data["reused_source_sha256"] = {name: sha256_file(source / name) for name in
                                    ("data.py", "polarfree.py", "losses.py", "evaluation.py", "config.py")}
    base = yaml.safe_load(Path(config["paths"]["base_config"]).read_text(encoding="utf-8"))
    official_root = Path(base["official"]["polarapp_dfp_root"])
    data["official_source_sha256"] = {p.relative_to(official_root).as_posix(): sha256_file(p) for p in sorted(official_root.rglob("*.py"))}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def polar_stats(before, after):
    change = (after.detach() - before.detach()).abs()
    return {"mean_abs": float(change.mean()),
            "relative_signal_change": float(change.mean() / before.detach().abs().mean().clamp_min(1e-12)),
            "ratio_gt_0_02": float((change > .02).float().mean()),
            "ratio_gt_0_05": float((change > .05).float().mean())}


def r2p_drift(model):
    initial = model.initial_r2p
    numerator = sum(float((p.detach().cpu() - initial[n]).square().sum())
                    for n, p in model.r2p.named_parameters())
    denominator = sum(float(p.square().sum()) for p in initial.values())
    return math.sqrt(numerator / max(denominator, 1e-30))


def calibrate(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    output_root = Path(config["paths"]["output_root"])
    calibration_path = output_root / "config" / "alignment_calibration.json"
    set_seed(int(config["seed"]))
    model, task, criterion, base_config = build_components(config, device)
    batches = int(config["alignment"]["calibration_batches"])
    loader = scheduled_loader(config, base_config, batches)
    shared = tuple(model.r2p.parameters())
    records = []
    for index, raw_batch in enumerate(loader, start=1):
        batch = unpack_batch(raw_batch, device)
        set_step_seed(int(config["seed"]), "calibrate", index, batch["sample_id"])
        output, prediction, task_loss, alignment_loss = forward_losses(model, task, criterion, batch)
        records.append(
            {
                "batch": index,
                "task_loss": float(task_loss.detach()),
                "alignment_loss": float(alignment_loss.detach()),
                "task_shared_r2p_grad_norm": autograd_norm(task_loss, shared),
                "alignment_shared_r2p_grad_norm": autograd_norm(alignment_loss, shared),
            }
        )
        del output, prediction, task_loss, alignment_loss, batch
    task_median = float(
        np.median([row["task_shared_r2p_grad_norm"] for row in records])
    )
    alignment_median = float(
        np.median([row["alignment_shared_r2p_grad_norm"] for row in records])
    )
    if task_median <= 0 or alignment_median <= 0:
        raise RuntimeError(
            f"Invalid calibration gradients: task={task_median}, align={alignment_median}"
        )
    requested = (
        float(config["alignment"]["target_gradient_fraction"])
        * task_median
        / alignment_median
    )
    coefficient = float(
        np.clip(
            requested,
            float(config["alignment"]["lambda_min"]),
            float(config["alignment"]["lambda_max"]),
        )
    )
    payload = {
        "schema_version": 1,
        "protocol_id": protocol_id(config),
        "records": records,
        "task_shared_r2p_grad_norm_median": task_median,
        "alignment_shared_r2p_grad_norm_median": alignment_median,
        "target_gradient_fraction": float(
            config["alignment"]["target_gradient_fraction"]
        ),
        "requested_lambda_feature_alignment": requested,
        "lambda_feature_alignment": coefficient,
        "clamped": not math.isclose(requested, coefficient),
    }
    atomic_json(calibration_path, payload)
    return payload


def task_drift(task: nn.Module, initial: Mapping[str, torch.Tensor]) -> float:
    numerator = 0.0
    denominator = 0.0
    for name, parameter in task.net_g.named_parameters():
        current = parameter.detach().cpu().float()
        reference = initial[name]
        numerator += float((current - reference).square().sum())
        denominator += float(reference.square().sum())
    return math.sqrt(numerator / max(denominator, 1e-30))


@torch.no_grad()
def validate(model, task, loader, device: torch.device, seed: int):
    from rgb_final_confirmation.evaluation import psnr, ssim
    model.eval()
    task.eval()
    rows = []
    for index, raw_batch in enumerate(loader):
        batch = unpack_batch(raw_batch, device)
        set_step_seed(seed, "validation", index, batch["sample_id"])
        out = recurrent_forward(model, task, batch)
        rows.append({
            "sample_id": batch["sample_id"][0], "scene": batch["scene"][0],
            "round1": {"psnr": psnr(out.t1, batch["target"]), "ssim": ssim(out.t1, batch["target"])},
            "round2": {"psnr": psnr(out.t2, batch["target"]), "ssim": ssim(out.t2, batch["target"])},
            "polar1": polar_stats(batch["polar"], out.first.refined_polar),
            "polar2": polar_stats(out.first.refined_polar, out.refined_polar),
        })
        del out
    if not rows:
        raise ValueError("Empty evaluation split")
    scenes = sorted({r["scene"] for r in rows})
    result = {"frames": len(rows), "scenes": len(scenes)}
    for rnd in ("round1", "round2"):
        result[rnd] = {}
        for metric in ("psnr", "ssim"):
            result[rnd]["image_mean_" + metric] = float(np.mean([r[rnd][metric] for r in rows]))
            result[rnd]["scene_macro_" + metric] = float(np.mean([
                np.mean([r[rnd][metric] for r in rows if r["scene"] == scene]) for scene in scenes]))
    result["iteration_gain"] = {k: result["round2"][k] - result["round1"][k] for k in result["round2"]}
    result["polar_update"] = {rnd: {k: float(np.mean([r[rnd][k] for r in rows]))
                                    for k in rows[0][rnd]} for rnd in ("polar1", "polar2")}
    # Top-level aliases always refer to round2, used by the inherited best selector.
    result.update(result["round2"])
    result["r2p_relative_l2_drift_from_pidsr"] = r2p_drift(model)
    result["feedback_scale"] = float(model.feedback_scale.detach())
    result["per_image"] = rows
    return result


def checkpoint_payload(
    model, task, optimizer, step: int, coefficient: float, metrics=None
):
    return {
        "schema_version": 2,
        "method": "shared K2 task feedback" if model.feedback_enabled else "shared K2 no feedback",
        "protocol_id": model.protocol_id,
        "feedback_enabled": model.feedback_enabled,
        "step": int(step),
        "lambda_feature_alignment": float(coefficient),
        "r2p_and_projections": model.state_dict(),
        "tasknet_net_g": task.net_g.state_dict(),
        "optimizer": optimizer.state_dict(),
        "validation": dict(metrics or {}),
    }


def run_preflight(config: Mapping[str, Any], device: torch.device):
    root = Path(config["paths"]["output_root"])
    if any((root / "checkpoints").glob("*.pt")):
        raise RuntimeError("Do not overwrite preflight/calibration of an existing run")
    calibration = calibrate(config, device)
    set_seed(int(config["seed"]))
    model, task, criterion, base = build_components(config, device)
    batch = unpack_batch(next(iter(scheduled_loader(config, base, 1))), device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    def forward():
        set_step_seed(int(config["seed"]), "preflight", 1, batch["sample_id"])
        return forward_losses(model, task, criterion, batch)
    calls = {"r2p": 0, "tasknet": 0}
    def count(name):
        def hook(*args):
            calls[name] += 1
        return hook
    handles = [model.register_forward_hook(count("r2p")), task.register_forward_hook(count("tasknet"))]
    out, _, tl, al = forward()
    for h in handles:
        h.remove()
    shapes = {k: list(v.shape) for k, v in out.first_task_features.items()}
    # Feedback path must carry gradients once its initially-zero gate opens.
    if model.feedback_enabled:
        for f in out.first_task_features.values():
            f.retain_grad()
    loss = tl + calibration["lambda_feature_alignment"] * al
    loss.backward()
    def finite_grads():
        return all(torch.isfinite(p.grad).all().item() for p in list(model.parameters()) + list(task.parameters()) if p.grad is not None)
    prior = [p for p in task.model.parameters() if id(p) not in {id(q) for q in task.net_g.parameters()}]
    payload = {
        "protocol_id": protocol_id(config), "strict_pidsr_load": True,
        "pretrain_epoch": model.pretrain_epoch,
        "r2p_shared": calls["r2p"] == 2, "tasknet_shared": calls["tasknet"] == 2,
        "feedback_enabled": model.feedback_enabled,
        "feedback_layers": ["net_g.TaF1", "net_g.TaF2", "net_g.TaF3"],
        "feedback_shapes": shapes,
        "feedback_adapter_params": sum(p.numel() for p in model.feedback_adapter.parameters()) + 1,
        "round1_polar_shape": list(out.first.refined_polar.shape),
        "round2_polar_shape": list(out.refined_polar.shape),
        "round1_task_shape": list(out.t1.shape), "round2_task_shape": list(out.t2.shape),
        "polar_round_difference": float((out.refined_polar - out.first.refined_polar).detach().abs().mean()),
        "task_round_difference": float((out.t2 - out.t1).detach().abs().mean()),
        "grad_norm_r2p": gradient_norm(model.r2p.parameters()),
        "grad_norm_net_g": gradient_norm(task.net_g.parameters()),
        "initial_adapter_grad_norm": gradient_norm(model.feedback_adapter.parameters()),
        "initial_scale_grad": float(model.feedback_scale.grad) if model.feedback_scale.grad is not None else None,
        "diffusion_prior_frozen": all(not p.requires_grad for p in prior),
        "grad_present_diffusion_prior": any(p.grad is not None for p in prior),
        "forward_finite": all(torch.isfinite(x).all().item() for x in (out.t1, out.t2, out.refined_polar, loss)),
        "backward_finite": finite_grads(),
        "lambda_feature_alignment": calibration["lambda_feature_alignment"],
    }
    expected = (out.t1.detach().clone(), out.t2.detach().clone())
    del out, tl, al, loss
    with torch.no_grad():
        repeat, _, _, _ = forward()
        payload["deterministic_forward"] = torch.equal(expected[0], repeat.t1) and torch.equal(expected[1], repeat.t2)
    del repeat, expected
    # One disposable update activates beta; no checkpoint is saved from preflight.
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["optimization"]["lr_r2p"]))
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["optimization"]["grad_clip"]), error_if_nonfinite=True)
    optimizer.step()
    model.zero_grad(set_to_none=True)
    task.zero_grad(set_to_none=True)
    out, _, tl, al = forward()
    if model.feedback_enabled:
        feedback_grad = torch.autograd.grad(out.loss2, tuple(out.first_task_features.values()), retain_graph=True, allow_unused=True)
        payload["round2_to_round1_features_grad_norm"] = float(sum(g.square().sum() for g in feedback_grad if g is not None).sqrt())
    (tl + calibration["lambda_feature_alignment"] * al).backward()
    payload["adapter_grad_norm_after_gate_update"] = gradient_norm(model.feedback_adapter.parameters())
    payload["backward_finite"] &= finite_grads()
    payload["grad_present_diffusion_prior"] |= any(p.grad is not None for p in prior)
    payload["peak_vram_mb"] = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    required = [payload[k] for k in ("r2p_shared", "tasknet_shared", "diffusion_prior_frozen", "forward_finite", "backward_finite", "deterministic_forward")]
    required += [not payload["grad_present_diffusion_prior"], payload["grad_norm_r2p"] > 0, payload["grad_norm_net_g"] > 0,
                 payload["polar_round_difference"] > 0, payload["task_round_difference"] > 0]
    if model.feedback_enabled:
        required += [payload["adapter_grad_norm_after_gate_update"] > 0, payload["round2_to_round1_features_grad_norm"] > 0]
    payload["passed"] = all(required)
    atomic_json(root / "diagnostics" / "recurrent_preflight.json", payload)
    if not payload["passed"]:
        raise RuntimeError("Preflight failed; inspect recurrent_preflight.json")
    return payload


def run_training(
    config: Mapping[str, Any], device: torch.device, steps: int, resume: bool
) -> dict[str, Any]:
    output_root = Path(config["paths"]["output_root"])
    identity = protocol_id(config)
    preflight_path = output_root / "diagnostics" / "recurrent_preflight.json"
    if not preflight_path.is_file():
        raise FileNotFoundError("Run recurrent preflight first")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not preflight.get("passed") or preflight.get("protocol_id") != identity:
        raise ValueError("Missing successful preflight for this exact configuration/code")
    calibration_path = output_root / "config" / "alignment_calibration.json"
    if not calibration_path.is_file():
        raise FileNotFoundError("Run preflight first; alignment calibration is missing")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("protocol_id") != identity:
        raise ValueError("Calibration belongs to a different protocol")
    coefficient = float(calibration["lambda_feature_alignment"])
    set_seed(int(config["seed"]))
    model, task, criterion, base_config = build_components(config, device)
    optimizer = torch.optim.Adam(
        [
            {
                "params": list(model.parameters()),
                "lr": float(config["optimization"]["lr_r2p"]),
                "name": "r2p_and_alignment",
            },
            {
                "params": list(task.net_g.parameters()),
                "lr": float(config["optimization"]["lr_tasknet"]),
                "name": "tasknet_net_g",
            },
        ],
        betas=tuple(float(value) for value in config["optimization"]["betas"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )
    start_step = 0
    latest_path = output_root / "checkpoints" / "last.pt"
    if resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint is missing: {latest_path}")
        state = torch.load(latest_path, map_location=device, weights_only=False)
        if state.get("protocol_id") != identity:
            raise ValueError("Checkpoint protocol changed before resume")
        if not math.isclose(float(state["lambda_feature_alignment"]), coefficient):
            raise ValueError("Alignment coefficient changed before resume")
        model.load_state_dict(state["r2p_and_projections"], strict=True)
        task.net_g.load_state_dict(state["tasknet_net_g"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
    elif any((output_root / "checkpoints").glob("*.pt")):
        raise FileExistsError(
            "Existing training checkpoint found; use --resume or a new output root"
        )
    if steps <= start_step:
        raise ValueError(
            f"Requested step {steps} must exceed checkpoint step {start_step}"
        )
    if steps > int(config["optimization"]["total_steps"]):
        raise ValueError("Requested steps exceed the configured experiment budget")
    if steps > 200 and start_step < 200:
        raise ValueError("Run the 200-step sanity stage, inspect results, then --resume")
    atomic_json(output_root / "config" / "run_config.json", config)
    (output_root / "RUN_PLAN.md").write_text(
        "# Run plan\n\n" + f"Seed: {config['seed']}; K=2; feedback={model.feedback_enabled}; "
        + f"full gradient; start={start_step}; end={steps}.\n\n"
        + f"Loss: 0.5 L1 + L2 + {coefficient:.9g} align1.\n\n"
        + f"LR: {config['optimization']['lr_r2p']} / {config['optimization']['lr_tasknet']}; "
        + f"batch={config['data']['batch_size']}; crop={config['data']['crop_size']}.\n\n"
        + f"Preflight passed; peak VRAM={preflight['peak_vram_mb']} MiB.\n\n"
        + "Initialization: PIDSR e42 + official TaskNet; resume uses last.pt.\n"
        + "Best: round2 scene-macro PSNR, SSIM, earlier step; validation only.\n"
        + "Against matched K2 control, only feedback usage differs.\n",
        encoding="utf-8")
    loader = scheduled_loader(config, base_config, steps, start_step)
    validation_loader = proxy_loader(config, base_config)
    initial_task = {
        name: value.detach().cpu().float().clone()
        for name, value in task.net_g.named_parameters()
    }
    if resume:
        initial_path = output_root / "config" / "tasknet_initial.pt"
        initial_task = torch.load(initial_path, map_location="cpu", weights_only=True)
    else:
        atomic_torch_save(output_root / "config" / "tasknet_initial.pt", initial_task)
    validation_history_path = output_root / "metrics" / "proxy_validation.json"
    history = (
        json.loads(validation_history_path.read_text(encoding="utf-8"))
        if validation_history_path.is_file()
        else []
    )
    validation_steps = {
        int(value) for value in config["optimization"]["validation_steps"]
    }
    best_score = None
    best_path = output_root / "checkpoints" / "best.pt"
    if best_path.is_file():
        best_state = torch.load(best_path, map_location="cpu", weights_only=False)
        best_metrics = best_state.get("validation", {})
        if best_metrics:
            best_score = (
                float(best_metrics["scene_macro_psnr"]),
                float(best_metrics["scene_macro_ssim"]),
                -int(best_state["step"]),
            )
    if start_step == 0 and 0 in validation_steps:
        metrics = validate(model, task, validation_loader, device, int(config["seed"]))
        metrics["step"] = 0.0
        metrics["tasknet_relative_l2_drift"] = 0.0
        history.append(metrics)
        atomic_json(validation_history_path, history)
        best_score = (
            float(metrics["scene_macro_psnr"]),
            float(metrics["scene_macro_ssim"]),
            0,
        )
        atomic_torch_save(
            best_path,
            checkpoint_payload(model, task, optimizer, 0, coefficient, metrics),
        )
    log_path = output_root / "logs" / "train.csv"
    iterator = iter(loader)
    started = time.perf_counter()
    step_times = []
    for step in range(start_step + 1, steps + 1):
        tick = time.perf_counter()
        batch = unpack_batch(next(iterator), device)
        noise_seed = set_step_seed(
            int(config["seed"]), "train", step, batch["sample_id"]
        )
        model.train()
        task.eval()
        optimizer.zero_grad(set_to_none=True)
        output, prediction, task_loss, alignment_loss = forward_losses(
            model, task, criterion, batch
        )
        loss = task_loss + coefficient * alignment_loss
        loss.backward()
        diagnostics = {
            "step": step,
            "diffusion_seed": noise_seed,
            "loss": float(loss.detach()),
            "task_loss": float(task_loss.detach()),
            "feature_alignment_loss": float(alignment_loss.detach()),
            "r2p_grad_norm": gradient_norm(model.r2p.parameters()),
            "projection_grad_norm": gradient_norm(model.projections.parameters()),
            "tasknet_grad_norm": gradient_norm(task.net_g.parameters()),
            "delta_abs_mean": float(output.delta.detach().abs().mean()),
            "gate_mean": float(output.gate.detach().mean()),
            "round1_task_loss": float(output.loss1.detach()),
            "round2_task_loss": float(output.loss2.detach()),
            "feedback_adapter_grad_norm": gradient_norm(model.feedback_adapter.parameters()),
            "feedback_scale": float(model.feedback_scale.detach()),
            "feedback_scale_grad": float(model.feedback_scale.grad) if model.feedback_scale.grad is not None else 0.0,
            "polar1_mean_abs": polar_stats(batch["polar"], output.first.refined_polar)["mean_abs"],
            "polar2_mean_abs": polar_stats(output.first.refined_polar, output.refined_polar)["mean_abs"],
        }
        # Clip the two learning-rate groups independently. TaskNet's much
        # larger raw gradient must not rescale away the R2P learning signal.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["optimization"]["grad_clip"]), error_if_nonfinite=True
        )
        torch.nn.utils.clip_grad_norm_(
            task.net_g.parameters(), float(config["optimization"]["grad_clip"]), error_if_nonfinite=True
        )
        optimizer.step()
        diagnostics["step_seconds"] = time.perf_counter() - tick
        step_times.append(float(diagnostics["step_seconds"]))
        if not all(np.isfinite(float(value)) for value in diagnostics.values()):
            raise FloatingPointError(f"Non-finite training diagnostics: {diagnostics}")
        append_csv(log_path, diagnostics)
        del output, prediction, task_loss, alignment_loss, loss, batch
        checkpoint_interval = int(config["optimization"]["checkpoint_interval"])
        should_checkpoint = step % checkpoint_interval == 0 or step == steps
        metrics = None
        if step in validation_steps or step == steps:
            metrics = validate(
                model, task, validation_loader, device, int(config["seed"])
            )
            metrics["step"] = float(step)
            metrics["tasknet_relative_l2_drift"] = task_drift(task, initial_task)
            history.append(metrics)
            atomic_json(validation_history_path, history)
            score = (
                float(metrics["scene_macro_psnr"]),
                float(metrics["scene_macro_ssim"]),
                -step,
            )
            if best_score is None or score > best_score:
                best_score = score
                atomic_torch_save(
                    best_path,
                    checkpoint_payload(
                        model, task, optimizer, step, coefficient, metrics
                    ),
                )
        if should_checkpoint:
            state = checkpoint_payload(
                model, task, optimizer, step, coefficient, metrics
            )
            atomic_torch_save(latest_path, state)
            atomic_torch_save(output_root / "checkpoints" / f"step_{step}.pt", state)
    timing = {
        "start_step": start_step,
        "end_step": steps,
        "wall_seconds": time.perf_counter() - started,
        "mean_step_seconds_excluding_first_20": float(
            np.mean(step_times[20:] or step_times)
        ),
    }
    atomic_json(
        output_root / "diagnostics" / f"timing_{start_step}_{steps}.json", timing
    )
    return {"checkpoint": str(latest_path), "validation": history[-1], "timing": timing}


def run_evaluation(
    config: Mapping[str, Any], device: torch.device, checkpoint_name: str, split: str
) -> dict[str, Any]:
    output_root = Path(config["paths"]["output_root"])
    checkpoint_path = output_root / "checkpoints" / f"{checkpoint_name}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    set_seed(int(config["seed"]))
    model, task, _, _ = build_components(config, device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if state.get("protocol_id") != model.protocol_id:
        raise ValueError("Evaluation configuration/code differs from checkpoint")
    model.load_state_dict(state["r2p_and_projections"], strict=True)
    task.net_g.load_state_dict(state["tasknet_net_g"], strict=True)
    metrics = validate(
        model, task, split_loader(config, split), device, int(config["seed"])
    )
    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(state["step"]),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "protocol_id": model.protocol_id,
        **metrics,
    }
    atomic_json(output_root / "metrics" / f"step_{state['step']}_{split}.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    train = commands.add_parser("train")
    train.add_argument("--steps", type=int, required=True)
    train.add_argument("--resume", action="store_true")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", default="best", help="best, last, or step_3000")
    evaluate.add_argument(
        "--split", choices=("validation", "test"), default="validation"
    )
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
        result = run_evaluation(config, device, args.checkpoint, args.split)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
