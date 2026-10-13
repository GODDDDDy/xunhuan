from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .config import ExperimentConfig, ensure_output_path


class _AdaptiveAvgPoolDeterministicBackward(torch.autograd.Function):
    """Use PyTorch's exact forward and a fixed-order deterministic backward."""

    @staticmethod
    def forward(
        ctx: Any, inputs: torch.Tensor, output_h: int, output_w: int
    ) -> torch.Tensor:
        ctx.input_shape = tuple(inputs.shape)
        ctx.output_size = (int(output_h), int(output_w))
        return F.adaptive_avg_pool2d(inputs, ctx.output_size)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        height, width = ctx.input_shape[-2:]
        output_h, output_w = ctx.output_size
        grad_input = grad_output.new_zeros(ctx.input_shape)
        # Fixed Python loop order makes accumulation deterministic when the
        # official adaptive bins overlap on non-divisible feature maps.
        for row in range(output_h):
            start_h = (row * height) // output_h
            end_h = ((row + 1) * height + output_h - 1) // output_h
            for column in range(output_w):
                start_w = (column * width) // output_w
                end_w = ((column + 1) * width + output_w - 1) // output_w
                scale = 1.0 / ((end_h - start_h) * (end_w - start_w))
                region = grad_input[..., start_h:end_h, start_w:end_w]
                contribution = grad_output[..., row : row + 1, column : column + 1] * scale
                grad_input[..., start_h:end_h, start_w:end_w] = region + contribution
        return grad_input, None, None


class _DeterministicAdaptiveAvgPool2d(nn.Module):
    """Forward-equivalent adaptive average pooling with deterministic backward.

    PyTorch 2.7 does not provide a deterministic CUDA backward for
    ``AdaptiveAvgPool2d``.  Expressing each official adaptive bin explicitly as
    a slice mean preserves PyTorch's floor/ceil bin boundaries, including the
    overlapping bins used when a full-resolution feature map is not divisible
    by the 4x4 output grid, while allowing gradients to reach R2P
    deterministically.
    """

    def __init__(self, output_size: int | tuple[int, int]):
        super().__init__()
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = tuple(output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output_h, output_w = self.output_size
        return _AdaptiveAvgPoolDeterministicBackward.apply(inputs, output_h, output_w)


def _replace_adaptive_avg_pool(module: nn.Module) -> int:
    """Replace parameter-free pooling modules without editing official code."""
    replacements = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.AdaptiveAvgPool2d):
            setattr(module, name, _DeterministicAdaptiveAvgPool2d(child.output_size))
            replacements += 1
        else:
            replacements += _replace_adaptive_avg_pool(child)
    return replacements


def refined_polar_to_task_inputs(
    refined_polar: torch.Tensor, official_rgb: torch.Tensor
) -> list[torch.Tensor]:
    """Use the same parity-gated differentiable preprocessing for refined P.

    This makes the zero-residual A-groups numerically comparable to Baseline.
    The unchanged official RGB is always the first PolarFree input.
    """
    return official_polarfree_task_inputs(refined_polar, official_rgb)


def opencv_rgb_to_gray_ste(rgb: torch.Tensor) -> torch.Tensor:
    """OpenCV uint8 RGB2GRAY forward with a differentiable surrogate gradient.

    OpenCV's 8-bit conversion uses 15-bit fixed-point coefficients. The
    detached exact-forward correction preserves those quantized values while
    gradients follow the corresponding floating-point luminance transform.
    Real-data parity must be established by the preprocessing audit before use.
    """
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape [B,3,H,W]")
    clipped = rgb.clamp(0.0, 1.0)
    rgb_u8 = clipped.mul(255.0).round().to(torch.int32)
    red, green, blue = torch.chunk(rgb_u8, 3, dim=1)
    gray_u8 = (
        (red * 9798 + green * 19235 + blue * 3735 + 16384) >> 15
    ).clamp(0, 255)
    exact = gray_u8.to(rgb.dtype) / 255.0
    weights = rgb.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    surrogate = (clipped * weights).sum(dim=1, keepdim=True)
    return surrogate + (exact - surrogate).detach()


def official_polarfree_task_inputs(
    polar: torch.Tensor, official_rgb: torch.Tensor
) -> list[torch.Tensor]:
    """Differentiable official-forward candidate; real-data parity is a hard gate."""
    if polar.ndim != 4 or polar.shape[1] != 12:
        raise ValueError("polar must have shape [B,12,H,W]")
    if official_rgb.shape != (polar.shape[0], 3, *polar.shape[-2:]):
        raise ValueError("official_rgb must have shape [B,3,H,W] matching polar")
    gray_0, gray_45, gray_90, gray_135 = [
        opencv_rgb_to_gray_ste(angle) for angle in torch.chunk(polar, 4, dim=1)
    ]
    # The released dataloader performs these NumPy operations in float64 and
    # casts the final arrays to torch.float32. Mirror that forward precision.
    g0, g45, g90, g135 = [value.to(torch.float64) for value in (gray_0, gray_45, gray_90, gray_135)]
    intensity = 0.5 * (g0 + g45 + g90 + g135) + 1e-4
    q = g0 - g90
    u = g45 - g135
    q = torch.where(q == 0, q.new_tensor(1e-4), q)
    intensity = torch.where(intensity == 0, intensity.new_tensor(1e-4), intensity)
    dolp = (torch.sqrt(q.square() + u.square()) / intensity).clamp(max=1.0).to(polar.dtype)
    aolp = (0.5 * torch.atan2(u, q)).to(polar.dtype)
    return [official_rgb, gray_0, gray_45, gray_90, gray_135, aolp, dolp]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_official_weights(config: ExperimentConfig) -> dict[str, Any]:
    checkpoint_dir = config.path("official", "checkpoint_dir")
    records = []
    for name in config.raw["official"]["expected_checkpoints"]:
        path = checkpoint_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    audit = {"strict_load": True, "weights": records}
    destination = ensure_output_path(config, config.path("paths", "weight_audit"))
    destination.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def _load_official_diffusion(config: ExperimentConfig, device: torch.device) -> nn.Module:
    from omegaconf import OmegaConf

    official_root = config.path("official", "polarapp_dfp_root").resolve()
    if str(official_root) not in sys.path:
        sys.path.insert(0, str(official_root))
    from archs.diffusion import DiffusionModel  # type: ignore[import-not-found]

    dm_config = OmegaConf.load(config.path("official", "dm_config"))
    checkpoint_dir = config.path("official", "checkpoint_dir").resolve()
    dm_config.train_dm = False
    dm_config.path.pretrain_network_le = None
    dm_config.path.pretrain_network_le_dm = str(checkpoint_dir / "net_le_dm_latest.pth")
    dm_config.path.pretrain_network_d = str(checkpoint_dir / "net_d_latest.pth")
    dm_config.path.pretrain_network_g = str(checkpoint_dir / "net_g_latest.pth")
    dm_config.path.strict_load_le_dm = True
    dm_config.path.strict_load_d = True
    dm_config.path.strict_load_g = True
    model = DiffusionModel(dm_config, device=device).to(device)
    replaced = _replace_adaptive_avg_pool(model)
    if replaced == 0:
        raise RuntimeError("Expected PolarFree AdaptiveAvgPool2d modules were not found")
    model._deterministic_pool_replacements = replaced
    return model


class OfficialPolarFreeTask(nn.Module):
    """Strict official model with every PolarFree parameter frozen.

    The official validation forward wraps the fixed prior in ``no_grad``. This
    adapter performs the same numerical operations with frozen parameters but
    without detaching inputs, so task loss can train R2P through the fixed prior.
    """

    def __init__(self, config: ExperimentConfig, device: torch.device):
        super().__init__()
        audit_official_weights(config)
        self.model = _load_official_diffusion(config, device)
        self.model.requires_grad_(False)
        self.model.eval()

    @property
    def net_g(self) -> nn.Module:
        return self.model.net_g

    def trainable_parameters(self) -> list[nn.Parameter]:
        return []

    def train(self, mode: bool = True) -> OfficialPolarFreeTask:
        # Deliberately ignore training mode. Autograd remains enabled for the
        # input tensor, while all PolarFree parameters stay frozen.
        del mode
        super().train(False)
        self.model.eval()
        return self

    def forward(
        self,
        refined_polar: torch.Tensor,
        official_rgb: torch.Tensor,
        diffusion_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        inputs = refined_polar_to_task_inputs(refined_polar, official_rgb)
        return self.forward_inputs(inputs, diffusion_noise)

    def forward_official_polarfree(
        self,
        polar: torch.Tensor,
        official_rgb: torch.Tensor,
        diffusion_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        inputs = official_polarfree_task_inputs(polar, official_rgb)
        return self.forward_inputs(inputs, diffusion_noise)

    def forward_inputs(
        self,
        inputs: list[torch.Tensor],
        diffusion_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prior_condition = self.model.net_le_dm(inputs)
        if diffusion_noise is None:
            diffusion_noise = torch.randn_like(prior_condition)
        if diffusion_noise.shape != prior_condition.shape:
            raise ValueError("diffusion_noise shape does not match the PolarFree prior")
        prior = self.model.p_sample_loop(prior_condition, diffusion_noise)
        prediction, official_features = self.model.net_g(inputs[0], prior)
        features = {
            "T1": official_features["TaF1"],
            "T2": official_features["TaF2"],
            "T3": official_features["TaF3"],
        }
        return prediction, features

    def assert_fully_frozen(self) -> None:
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise AssertionError("Every PolarFree parameter must remain frozen")
        if any(parameter.grad is not None for parameter in self.parameters()):
            raise AssertionError("Frozen PolarFree parameters unexpectedly received gradients")


class SyntheticTask(nn.Module):
    """Small task surrogate used only by synthetic code tests."""

    def __init__(self):
        super().__init__()
        self.polar1 = nn.Conv2d(12, 48, 3, padding=1)
        self.polar2 = nn.Conv2d(48, 96, 3, stride=2, padding=1)
        self.polar3 = nn.Conv2d(96, 192, 3, stride=2, padding=1)
        self.rgb = nn.Conv2d(3, 48, 3, padding=1)
        self.head = nn.Conv2d(48, 3, 3, padding=1)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.parameters())

    def assert_fully_frozen(self) -> None:
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise AssertionError("Synthetic task must be frozen by ExperimentTrainer")
        if any(parameter.grad is not None for parameter in self.parameters()):
            raise AssertionError("Synthetic frozen task unexpectedly received gradients")

    def forward(
        self,
        refined_polar: torch.Tensor,
        official_rgb: torch.Tensor,
        diffusion_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del diffusion_noise
        feature1 = torch.relu(self.polar1(refined_polar))
        feature2 = torch.relu(self.polar2(feature1))
        feature3 = torch.relu(self.polar3(feature2))
        prediction = torch.sigmoid(self.head(feature1 + torch.relu(self.rgb(official_rgb))))
        return prediction, {"T1": feature1, "T2": feature2, "T3": feature3}
