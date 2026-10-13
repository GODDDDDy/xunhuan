from __future__ import annotations

import sys

import torch
from torch import nn
from torch.nn import functional as F

from .config import ExperimentConfig


def _ssim(image_a: torch.Tensor, image_b: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    coordinates = torch.arange(window_size, device=image_a.device, dtype=image_a.dtype)
    gaussian = torch.exp(-((coordinates - window_size // 2) ** 2) / (2 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    window = torch.outer(gaussian, gaussian).expand(image_a.shape[1], 1, window_size, window_size)
    mean_a = F.conv2d(image_a, window, padding=window_size // 2, groups=image_a.shape[1])
    mean_b = F.conv2d(image_b, window, padding=window_size // 2, groups=image_b.shape[1])
    variance_a = (
        F.conv2d(image_a.square(), window, padding=window_size // 2, groups=image_a.shape[1])
        - mean_a.square()
    )
    variance_b = (
        F.conv2d(image_b.square(), window, padding=window_size // 2, groups=image_b.shape[1])
        - mean_b.square()
    )
    covariance = (
        F.conv2d(image_a * image_b, window, padding=window_size // 2, groups=image_a.shape[1])
        - mean_a * mean_b
    )
    numerator = (2 * mean_a * mean_b + 0.01**2) * (2 * covariance + 0.03**2)
    denominator = (mean_a.square() + mean_b.square() + 0.01**2) * (
        variance_a + variance_b + 0.03**2
    )
    return (numerator / denominator.clamp_min(torch.finfo(image_a.dtype).eps)).mean()


class CoreTaskLoss(nn.Module):
    """L1 + SSIM + Fourier phase; synthetic tests use this without VGG."""

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        phase_prediction = torch.angle(torch.fft.fft2(prediction.mean(dim=1, keepdim=True) + 1e-5))
        phase_target = torch.angle(torch.fft.fft2(target.mean(dim=1, keepdim=True) + 1e-5))
        return (
            F.l1_loss(prediction, target)
            + (1.0 - _ssim(prediction, target))
            + F.mse_loss(phase_prediction, phase_target)
        )


def build_official_task_loss(config: ExperimentConfig) -> nn.Module:
    """Load PolarAPP's DfPTaskLoss without importing its polar-GT loss into training."""
    root = config.path("official", "polarapp_dfp_root").resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from polarapp.losses import DfPTaskLoss  # type: ignore[import-not-found]

    return DfPTaskLoss(float(config.raw["loss"]["perceptual_weight"]))
