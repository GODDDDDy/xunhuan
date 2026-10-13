from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 8):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=False),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels, groups=groups)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.ReLU(inplace=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.conv2(self.conv1(inputs)))


class Encoder(nn.Module):
    def __init__(self, in_channels: int, widths: tuple[int, int, int], blocks: int, groups: int):
        super().__init__()
        stages: list[nn.Module] = []
        previous = in_channels
        for index, width in enumerate(widths):
            layers: list[nn.Module] = [
                ConvNormAct(previous, width, stride=1 if index == 0 else 2, groups=groups)
            ]
            layers.extend(ResidualBlock(width, groups=groups) for _ in range(blocks))
            stages.append(nn.Sequential(*layers))
            previous = width
        self.stages = nn.ModuleList(stages)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = []
        value = inputs
        for stage in self.stages:
            value = stage(value)
            features.append(value)
        return tuple(features)  # type: ignore[return-value]


class ChannelWeights(nn.Module):
    """CMX channel gates, retaining the original two-input gate computation."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.mlp = nn.Sequential(
            nn.Linear(channels * 4, channels * 4),
            nn.ReLU(inplace=False),
            nn.Linear(channels * 4, channels * 2),
            nn.Sigmoid(),
        )

    def forward(self, polar: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        joined = torch.cat((polar, condition), dim=1)
        # ``adaptive_avg_pool2d(..., 1)`` has a non-deterministic CUDA
        # backward in PyTorch 2.7.  A spatial mean is mathematically
        # identical for a 1x1 output and keeps the strict deterministic mode
        # required by this experiment.
        average = joined.mean(dim=(-2, -1))
        maximum = joined.amax(dim=(-2, -1))
        batch = polar.shape[0]
        return self.mlp(torch.cat((average, maximum), dim=1)).view(batch, 2, self.channels, 1, 1)


class SpatialWeights(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, polar: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat((polar, condition), dim=1)).unsqueeze(2)


class AsymmetricFeatureRectifyModule(nn.Module):
    """CM-FRM gates that update only the polar stream."""

    def __init__(self, channels: int, lambda_channel: float = 0.5, lambda_spatial: float = 0.5):
        super().__init__()
        self.lambda_channel = float(lambda_channel)
        self.lambda_spatial = float(lambda_spatial)
        self.channel_weights = ChannelWeights(channels)
        self.spatial_weights = SpatialWeights(channels)

    def forward(
        self, polar: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        channel = self.channel_weights(polar, condition)
        spatial = self.spatial_weights(polar, condition)
        # CMX uses the opposite stream's gate (index 1) when updating stream 1.
        rectified = polar + self.lambda_channel * channel[:, 1] * condition
        rectified = rectified + self.lambda_spatial * spatial[:, 1] * condition
        diagnostics = {
            "channel_gate_mean": channel[:, 1].mean(),
            "channel_gate_std": channel[:, 1].std(unbiased=False),
            "spatial_gate_mean": spatial[:, 1].mean(),
            "spatial_gate_std": spatial[:, 1].std(unbiased=False),
        }
        return rectified, diagnostics


class OneWayCrossAttention(nn.Module):
    """One-way spatial attention with Q from polar and K/V from condition."""

    def __init__(self, channels: int, heads: int, window_size: int = 8):
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by attention heads")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.window_size = int(window_size)
        self.scale = self.head_dim**-0.5
        self.query = nn.Conv2d(channels, channels, 1, bias=False)
        self.key = nn.Conv2d(channels, channels, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.output = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(8, channels)

    def _window_tokens(self, value: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        batch, _, height, width = value.shape
        window = self.window_size
        pad_h = (-height) % window
        pad_w = (-width) % window
        value = F.pad(value, (0, pad_w, 0, pad_h))
        padded_h, padded_w = value.shape[-2:]
        value = value.view(
            batch,
            self.heads,
            self.head_dim,
            padded_h // window,
            window,
            padded_w // window,
            window,
        )
        value = value.permute(0, 3, 5, 1, 4, 6, 2).reshape(
            -1, self.heads, window * window, self.head_dim
        )
        return value, (batch, height, width, padded_w)

    def _restore_windows(
        self, value: torch.Tensor, shape: tuple[int, int, int, int]
    ) -> torch.Tensor:
        batch, height, width, padded_w = shape
        window = self.window_size
        padded_h = math.ceil(height / window) * window
        value = value.view(
            batch,
            padded_h // window,
            padded_w // window,
            self.heads,
            window,
            window,
            self.head_dim,
        )
        value = value.permute(0, 3, 6, 1, 4, 2, 5).reshape(batch, self.channels, padded_h, padded_w)
        return value[..., :height, :width]

    def forward(
        self, polar: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        query, shape = self._window_tokens(self.query(polar))
        key, _ = self._window_tokens(self.key(condition))
        value, _ = self._window_tokens(self.value(condition))
        logits = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        weights = logits.softmax(dim=-1)
        attended = self._restore_windows(torch.matmul(weights, value), shape)
        contribution = self.output(attended)
        result = self.norm(polar + contribution)
        entropy = -(weights.clamp_min(1e-8).log() * weights).sum(dim=-1)
        entropy = entropy / math.log(max(weights.shape[-1], 2))
        diagnostics = {
            "attention_entropy": entropy.mean(),
            "attention_contribution_ratio": contribution.norm() / polar.norm().clamp_min(1e-8),
        }
        return result, diagnostics


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, groups: int):
        super().__init__()
        self.reduce = ConvNormAct(in_channels + skip_channels, out_channels, groups=groups)
        self.block = ResidualBlock(out_channels, groups=groups)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(self.reduce(torch.cat((value, skip), dim=1)))


@dataclass
class R2POutput:
    refined_polar: torch.Tensor
    features: dict[str, torch.Tensor]
    projected_features: dict[str, torch.Tensor]
    diagnostics: dict[str, torch.Tensor]


class ConditionalR2P(nn.Module):
    def __init__(
        self,
        widths: tuple[int, int, int] = (48, 96, 192),
        blocks_per_stage: int = 2,
        groups: int = 8,
        attention_heads: int = 8,
        attention_window: int = 8,
        residual_alpha: float = 0.05,
        frm_lambda_channel: float = 0.5,
        frm_lambda_spatial: float = 0.5,
        task_feature_channels: tuple[int, int, int] = (96, 96, 192),
    ):
        super().__init__()
        self.residual_alpha = float(residual_alpha)
        self.polar_encoder = Encoder(12, widths, blocks_per_stage, groups)
        self.condition_encoder = Encoder(3, widths, blocks_per_stage, groups)
        self.rectifiers = nn.ModuleList(
            AsymmetricFeatureRectifyModule(width, frm_lambda_channel, frm_lambda_spatial)
            for width in widths
        )
        self.cross_attention = OneWayCrossAttention(widths[2], attention_heads, attention_window)
        self.decode2 = DecoderStage(widths[2], widths[1], widths[1], groups)
        self.decode1 = DecoderStage(widths[1], widths[0], widths[0], groups)
        self.residual_head = nn.Conv2d(widths[0], 12, 3, padding=1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        # These heads are part of R2P. No external alignment/FANet parameters
        # are trained in the four-group experiment.
        self.feature_projections = nn.ModuleDict(
            {
                f"P{level}": nn.Conv2d(source, target, 1, bias=False)
                for level, (source, target) in enumerate(
                    zip(widths, task_feature_channels, strict=True), start=1
                )
            }
        )

    def calibration_backbone_parameters(self) -> tuple[nn.Parameter, ...]:
        """Shared trunk used to compare task, alignment, and residual gradients.

        The loss-specific feature projections and residual head are excluded.
        """
        modules = (
            self.polar_encoder,
            self.condition_encoder,
            self.rectifiers,
            self.cross_attention,
            self.decode2,
            self.decode1,
        )
        return tuple(parameter for module in modules for parameter in module.parameters())

    def forward(self, polar: torch.Tensor, condition: torch.Tensor) -> R2POutput:
        if polar.ndim != 4 or polar.shape[1] != 12:
            raise ValueError(f"polar must be BCHW with 12 channels, got {polar.shape}")
        if condition.shape != (polar.shape[0], 3, polar.shape[2], polar.shape[3]):
            raise ValueError(
                f"condition shape {condition.shape} is incompatible with polar {polar.shape}"
            )
        polar_features = self.polar_encoder(polar)
        condition_features = self.condition_encoder(condition)
        rectified: list[torch.Tensor] = []
        diagnostics: dict[str, torch.Tensor] = {}
        for level, (module, polar_feature, condition_feature) in enumerate(
            zip(self.rectifiers, polar_features, condition_features), start=1
        ):
            value, stats = module(polar_feature, condition_feature)
            rectified.append(value)
            diagnostics.update({f"frm{level}_{key}": item for key, item in stats.items()})
        bottleneck, attention_stats = self.cross_attention(rectified[2], condition_features[2])
        diagnostics.update(attention_stats)
        decoded2 = self.decode2(bottleneck, rectified[1])
        decoded1 = self.decode1(decoded2, rectified[0])
        delta = self.residual_alpha * torch.tanh(self.residual_head(decoded1))
        refined = torch.clamp(polar + delta, 0.0, 1.0)
        effective_delta = refined - polar
        raw_features = {"P1": decoded1, "P2": decoded2, "P3": bottleneck}
        projected_features = {
            key: self.feature_projections[key](value) for key, value in raw_features.items()
        }
        diagnostics.update(
            {
                "delta_abs_mean": effective_delta.abs().mean(),
                "delta_abs_max": effective_delta.abs().amax(),
                "clamp_fraction": ((polar + delta < 0) | (polar + delta > 1)).float().mean(),
            }
        )
        for angle, part in zip(
            ("000", "045", "090", "135"), torch.chunk(effective_delta, 4, dim=1)
        ):
            diagnostics[f"delta_abs_{angle}"] = part.abs().mean()
        return R2POutput(
            refined_polar=refined,
            features=raw_features,
            projected_features=projected_features,
            diagnostics=diagnostics,
        )
