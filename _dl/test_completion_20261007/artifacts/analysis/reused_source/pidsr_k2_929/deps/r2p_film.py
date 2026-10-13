"""Lightweight RGB-polarization fusion module for the R2P experiments.

The implementation adapts two established ideas:

* GMU-style gated multimodal fusion (a learned spatial gate selects the
  contribution of RGB and polarization features);
* FiLM-style feature-wise affine modulation (RGB predicts scale and bias for
  the polarization stream).

It refines an already dense, four-angle polarization tensor.  It is not a
 CPFA/polarization demosaicker and does not require a polarization target.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class R2POutput:
    """Named output container returned when ``return_features=True``."""

    refined: torch.Tensor
    rgb_feature: torch.Tensor
    polar_feature: torch.Tensor
    gate: torch.Tensor
    fused_feature: torch.Tensor
    delta: torch.Tensor


class ConvStem(nn.Module):
    """Small convolutional stem kept identical for both modalities."""

    def __init__(self, in_channels: int, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class R2P(nn.Module):
    """RGB-conditioned residual refinement of a 12-channel polarization image.

    Args:
        width: Feature width used by both modality streams.
        max_residual: Maximum absolute correction per output channel.

    Inputs are normalized tensors with shapes ``(B, 3, H, W)`` and
    ``(B, 12, H, W)``.  The RGB tensor can therefore be either the original
    PolaRGB input or the offline PolarFree result; the module itself is
    unchanged between the two experiments.
    """

    def __init__(self, width: int = 48, max_residual: float = 0.10) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        if max_residual <= 0:
            raise ValueError("max_residual must be positive")

        self.max_residual = float(max_residual)
        self.rgb_encoder = ConvStem(3, width)
        self.polar_encoder = ConvStem(12, width)

        # GMU-style spatial gate.  gate=1 trusts RGB features; gate=0 trusts
        # the polarization stream.  Initial 0 logits give a neutral 50/50
        # fusion instead of a hard-coded preference for either modality.
        self.gate = nn.Conv2d(width * 2, width, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

        # FiLM parameters are predicted from RGB features and applied to the
        # polarization stream.  Small initial modulation keeps the first
        # iterations close to the identity refinement.
        self.film = nn.Conv2d(width, width * 2, 1)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        self.body = nn.Sequential(
            nn.Conv2d(width * 2, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, 12, 3, padding=1),
        )
        # Identity at initialization: Ptask == P0.
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(
        self,
        rgb: torch.Tensor,
        polar: torch.Tensor,
        *,
        return_features: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | torch.Tensor:
        self._validate_inputs(rgb, polar)
        rgb_feature = self.rgb_encoder(rgb)
        polar_feature = self.polar_encoder(polar)

        gate = torch.sigmoid(self.gate(torch.cat((polar_feature, rgb_feature), dim=1)))
        fused_feature = gate * rgb_feature + (1.0 - gate) * polar_feature

        gamma, beta = self.film(rgb_feature).chunk(2, dim=1)
        gamma = 0.10 * torch.tanh(gamma)
        beta = 0.10 * torch.tanh(beta)
        modulated_polar = polar_feature * (1.0 + gamma) + beta

        delta = self.max_residual * torch.tanh(
            self.body(torch.cat((modulated_polar, fused_feature), dim=1))
        )
        refined = (polar + delta).clamp(0.0, 1.0)

        if not return_features:
            return refined
        features = {
            "rgb": rgb_feature,
            "polar": polar_feature,
            "gate": gate,
            "fused": fused_feature,
            "modulated_polar": modulated_polar,
            "delta": delta,
        }
        return refined, features

    @staticmethod
    def _validate_inputs(rgb: torch.Tensor, polar: torch.Tensor) -> None:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"rgb must have shape (B,3,H,W), got {tuple(rgb.shape)}")
        if polar.ndim != 4 or polar.shape[1] != 12:
            raise ValueError(
                f"polar must have shape (B,12,H,W), got {tuple(polar.shape)}"
            )
        if rgb.shape[0] != polar.shape[0] or rgb.shape[-2:] != polar.shape[-2:]:
            raise ValueError("rgb and polar must have matching batch and spatial dimensions")
