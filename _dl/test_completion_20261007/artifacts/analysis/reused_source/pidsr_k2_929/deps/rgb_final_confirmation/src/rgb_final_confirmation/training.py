from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .r2p import ConditionalR2P, R2POutput


@dataclass
class Batch:
    polar: torch.Tensor
    condition: torch.Tensor
    official_rgb: torch.Tensor
    target: torch.Tensor
    sample_id: list[str]
    scene: list[str]

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any], device: torch.device) -> Batch:
        return cls(
            polar=values["polar"].to(device),
            condition=values["condition"].to(device),
            official_rgb=values["official_rgb"].to(device),
            target=values["target"].to(device),
            sample_id=list(values["sample_id"]),
            scene=list(values["scene"]),
        )


def set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def stable_step_seed(seed: int, phase: str, step: int, sample_ids: Iterable[str]) -> int:
    """Group-independent seed for PolarFree diffusion noise."""
    payload = f"{seed}:{phase}:{step}:{'|'.join(sample_ids)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31)


def set_step_seed(seed: int, phase: str, step: int, sample_ids: Iterable[str]) -> int:
    derived = stable_step_seed(seed, phase, step, sample_ids)
    torch.manual_seed(derived)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(derived)
    return derived


def gradient_norm(parameters: Iterable[nn.Parameter]) -> torch.Tensor:
    values = [
        parameter.grad.detach().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not values:
        return torch.tensor(0.0)
    return torch.stack(values).sum().sqrt()


def _autograd_norm(loss: torch.Tensor, parameters: tuple[nn.Parameter, ...]) -> float:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    values = [gradient.detach().square().sum() for gradient in gradients if gradient is not None]
    if not values:
        return 0.0
    return float(torch.stack(values).sum().sqrt())


def feature_alignment_loss(
    projected_r2p_features: Mapping[str, torch.Tensor],
    task_features: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Parameter-free cosine objective; task features are explicit stop-gradient targets."""
    losses = []
    for level in (1, 2, 3):
        source = projected_r2p_features[f"P{level}"]
        target = task_features[f"T{level}"].detach()
        if source.shape[-2:] != target.shape[-2:]:
            source = F.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=False)
        if source.shape != target.shape:
            raise ValueError(f"Feature-alignment shape mismatch: {source.shape} != {target.shape}")
        losses.append(1.0 - F.cosine_similarity(source, target, dim=1).mean())
    return torch.stack(losses).mean()


def _polar_diagnostics(original: torch.Tensor, refined: torch.Tensor) -> dict[str, float]:
    def quantities(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        p0, p45, p90, p135 = torch.chunk(value, 4, dim=1)
        s0 = 0.5 * (p0 + p45 + p90 + p135)
        s1, s2 = p0 - p90, p45 - p135
        dolp = (torch.sqrt(s1.square() + s2.square() + 1e-8) / (s0 + 1e-8)).clamp(0, 1)
        aolp = 0.5 * torch.atan2(s2, s1)
        return s0, s1, s2, dolp, aolp

    before, after = quantities(original), quantities(refined)
    result = {
        f"{name}_change_abs_mean": float((right - left).abs().mean().detach())
        for name, left, right in zip(("stokes_s0", "stokes_s1", "stokes_s2", "dolp"), before[:4], after[:4])
    }
    angle_delta = after[4] - before[4]
    wrapped = 0.5 * torch.atan2(torch.sin(2 * angle_delta), torch.cos(2 * angle_delta))
    result["aolp_wrapped_change_abs_mean"] = float(wrapped.abs().mean().detach())
    return result


class ExperimentTrainer:
    """First-order trainer whose optimizer contains R2P parameters only."""

    def __init__(
        self,
        r2p: ConditionalR2P,
        task: nn.Module,
        criterion: nn.Module,
        *,
        lr_r2p: float = 1e-4,
        lambda_feature_alignment: float = 0.0,
        lambda_residual: float = 0.0,
        grad_clip: float = 1.0,
    ):
        self.r2p = r2p
        self.task = task.requires_grad_(False).eval()
        self.criterion = criterion.requires_grad_(False).eval()
        self.lambda_feature_alignment = float(lambda_feature_alignment)
        self.lambda_residual = float(lambda_residual)
        self.grad_clip = float(grad_clip)
        self.optimizer_r2p = torch.optim.Adam(
            self.r2p.parameters(), lr=float(lr_r2p), betas=(0.9, 0.999), weight_decay=0.0
        )
        if any(parameter.requires_grad for parameter in self.task.parameters()):
            raise AssertionError("PolarFree must be fully frozen before trainer construction")

    def _forward_losses(
        self, batch: Batch
    ) -> tuple[R2POutput, torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.r2p(batch.polar, batch.condition)
        prediction, task_features = self.task(
            output.refined_polar, batch.official_rgb, None
        )
        task_loss = self.criterion(prediction, batch.target)
        alignment_loss = feature_alignment_loss(output.projected_features, task_features)
        residual_loss = F.l1_loss(output.refined_polar, batch.polar)
        for name, value in {
            "task_loss": task_loss,
            "feature_alignment_loss": alignment_loss,
            "residual_loss": residual_loss,
        }.items():
            if not torch.isfinite(value):
                raise FloatingPointError(f"Non-finite {name}")
        return output, task_loss, alignment_loss, residual_loss

    def task_only_probe_step(self, batch: Batch) -> dict[str, float]:
        self.r2p.train()
        self.task.eval()
        output, task_loss, _, _ = self._forward_losses(batch)
        self.optimizer_r2p.zero_grad(set_to_none=True)
        task_loss.backward()
        norm = gradient_norm(self.r2p.parameters())
        torch.nn.utils.clip_grad_norm_(self.r2p.parameters(), self.grad_clip)
        self.optimizer_r2p.step()
        self.task.assert_fully_frozen()  # type: ignore[attr-defined]
        return {"task_loss": float(task_loss.detach()), "r2p_grad_norm": float(norm)}

    def component_gradient_norms(self, batch: Batch) -> dict[str, float]:
        self.r2p.train()
        _, task_loss, alignment_loss, residual_loss = self._forward_losses(batch)
        parameters = self.r2p.calibration_backbone_parameters()
        return {
            "task": _autograd_norm(task_loss, parameters),
            "feature_alignment": _autograd_norm(alignment_loss, parameters),
            "residual": _autograd_norm(residual_loss, parameters),
        }

    def train_step(self, batch: Batch) -> dict[str, float]:
        if self.lambda_feature_alignment <= 0 or self.lambda_residual <= 0:
            raise RuntimeError("Auxiliary weights must be calibrated and locked before training")
        self.r2p.train()
        self.task.eval()
        output, task_loss, alignment_loss, residual_loss = self._forward_losses(batch)
        loss = (
            task_loss
            + self.lambda_feature_alignment * alignment_loss
            + self.lambda_residual * residual_loss
        )
        self.optimizer_r2p.zero_grad(set_to_none=True)
        loss.backward()
        diagnostics = {
            "loss": float(loss.detach()),
            "task_loss": float(task_loss.detach()),
            "feature_alignment_loss": float(alignment_loss.detach()),
            "residual_loss": float(residual_loss.detach()),
            "r2p_grad_norm": float(gradient_norm(self.r2p.parameters())),
            "polar_encoder_grad_norm": float(gradient_norm(self.r2p.polar_encoder.parameters())),
            "condition_encoder_grad_norm": float(gradient_norm(self.r2p.condition_encoder.parameters())),
            "residual_head_grad_norm": float(gradient_norm(self.r2p.residual_head.parameters())),
            "projection_head_grad_norm": float(gradient_norm(self.r2p.feature_projections.parameters())),
        }
        if not all(np.isfinite(value) for value in diagnostics.values()):
            raise FloatingPointError(f"Non-finite training diagnostics: {diagnostics}")
        torch.nn.utils.clip_grad_norm_(self.r2p.parameters(), self.grad_clip)
        self.optimizer_r2p.step()
        self.task.assert_fully_frozen()  # type: ignore[attr-defined]
        diagnostics.update(
            {key: float(value.detach().mean()) for key, value in output.diagnostics.items()}
        )
        diagnostics.update(_polar_diagnostics(batch.polar, output.refined_polar))
        return diagnostics

    def checkpoint_state(
        self, step: int, validation: Mapping[str, float] | None = None
    ) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "step": int(step),
            "validation": dict(validation or {}),
            "r2p": self.r2p.state_dict(),
            "optimizer_r2p": self.optimizer_r2p.state_dict(),
            "lambda_feature_alignment": self.lambda_feature_alignment,
            "lambda_residual": self.lambda_residual,
        }
