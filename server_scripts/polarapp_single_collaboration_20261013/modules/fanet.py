"""Same-scale adapter around the pinned, unmodified PolarAPP FANet classes.

This is an adaptation, not the official two-times-resolution experiment. The
official convolution/norm parameters and checkpoint key aliases are retained.
Only a 48 -> 96 P1 interface and resolution-aware FT forwarding are added.
The three-scale cosine-distance objective is the official DfP objective.

The legacy vendor ``*_meta`` attributes are not used. Differentiable trial
updates must use torch.func.functional_call on ordinary registered parameters,
as the current official polarapp/trainer.py does. No files are written here.
"""

from __future__ import annotations

import hashlib
import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


OFFICIAL_COMMIT = "1f2b4b86eb8e2b3a863a7fb53162b87fac5a3860"
OFFICIAL_WEIGHT_SHA256 = "20fe6defc1ac37b5d7011bce5b83a43c9e21e6948f36bbb98933a742a7b75a4b"
OFFICIAL_PARAMETER_COUNT = 8_466_432
OFFICIAL_STATE_KEYS = 66
OFFICIAL_INDEPENDENT_PARAMETERS = 45


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_vendor_module(source: Path):
    """Load exactly the requested file without changing sys.path/sys.modules.

    Official to_var automatically calls .cuda() during module construction.
    Override that private module's device helper only, so CPU preparation cannot
    accidentally initialize CUDA. No official source or global torch function is
    changed. Ordinary meta=False forwards and registered parameters are intact.
    """
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    module_name = "_polarapp_fanet_private_" + hashlib.sha256(
        str(source).encode("utf-8")
    ).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pinned PolarAPP source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.to_var = lambda value, requires_grad=True: torch.autograd.Variable(
        value, requires_grad=requires_grad
    )
    return module


def _checkpoint_state(payload: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError("FANet checkpoint must be a mapping")
    for key in ("model_state_dict", "state_dict", "params", "model"):
        nested = payload.get(key)
        if isinstance(nested, Mapping) and nested:
            if all(isinstance(value, torch.Tensor) for value in nested.values()):
                return nested
    if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
        return payload
    raise ValueError("Checkpoint has no recognized, tensor-only FANet state")


class SameScaleFANet(nn.Module):
    """Official FANet with a deterministic input interface and no hard x2 resize.

    Args:
        official_source: pinned vendor/PolarAPP/DfP/utils/utils_net.py.
        checkpoint: official FANet.pth; hash and exact state are checked by default.
        p1_weight: optional inherited projections.P1.weight, shape (96,48,1,1).

    Inputs use our P1/P2/P3 and T1/T2/T3 names, or official DF1/2/3 and
    TaF1/2/3 names. Outputs use official TF1/2/3 and MF1/2/3 names. Neither
    input branch is detached: both upstream and downstream need gradients.
    """

    def __init__(
        self,
        official_source: str | Path,
        checkpoint: str | Path | None = None,
        p1_weight: torch.Tensor | None = None,
        expected_sha256: str | None = OFFICIAL_WEIGHT_SHA256,
    ) -> None:
        super().__init__()
        vendor = _load_vendor_module(Path(official_source))
        self.official = vendor.FeatureAlignment()
        self.p1_adapter = nn.Conv2d(48, 96, kernel_size=1, bias=False)
        with torch.no_grad():
            # Deterministic fallback preserves total squared feature energy.
            # Formal migration should pass the inherited P1 projection instead.
            self.p1_adapter.weight.zero_()
            identity = torch.eye(48) / (2.0 ** 0.5)
            self.p1_adapter.weight[:48, :, 0, 0].copy_(identity)
            self.p1_adapter.weight[48:, :, 0, 0].copy_(identity)
        self.p1_initialization = "duplicated_identity_energy_preserving"
        if p1_weight is not None:
            if tuple(p1_weight.shape) != (96, 48, 1, 1):
                raise ValueError(f"Inherited P1 projection shape: {tuple(p1_weight.shape)}")
            if not torch.isfinite(p1_weight).all():
                raise ValueError("Inherited P1 projection contains nonfinite values")
            with torch.no_grad():
                self.p1_adapter.weight.copy_(p1_weight.detach().cpu())
            self.p1_initialization = "inherited_projections.P1.weight"
        self.load_receipt: dict[str, Any] | None = None
        if checkpoint is not None:
            self.load_official_checkpoint(checkpoint, expected_sha256)
        self.set_trainable_partition()

    def load_official_checkpoint(
        self, checkpoint: str | Path, expected_sha256: str | None = OFFICIAL_WEIGHT_SHA256
    ) -> dict[str, Any]:
        path = Path(checkpoint)
        actual_sha = _sha256(path)
        if expected_sha256 is not None and actual_sha != expected_sha256:
            raise ValueError(f"Official FANet SHA256 mismatch: {actual_sha}")
        state = _checkpoint_state(torch.load(path, map_location="cpu", weights_only=True))

        # MetaConv2d registers .weight and .conv.weight as aliases. Strict loading
        # alone would accept conflicting aliases and overwrite one silently.
        aliases: dict[int, list[str]] = {}
        for name, parameter in self.official.named_parameters(remove_duplicate=False):
            aliases.setdefault(id(parameter), []).append(name)
        for names in aliases.values():
            if len(names) > 1 and all(name in state for name in names):
                first = state[names[0]]
                if any(not torch.equal(first, state[name]) for name in names[1:]):
                    raise ValueError(f"Conflicting official checkpoint aliases: {names}")
        self.official.load_state_dict(state, strict=True)
        current = self.official.state_dict()
        if any(not torch.equal(current[name].cpu(), value.cpu()) for name, value in state.items()):
            raise RuntimeError("Strict FANet load did not preserve every source tensor")
        independent = list(self.official.parameters())
        receipt = {
            "official_commit": OFFICIAL_COMMIT,
            "checkpoint_sha256": actual_sha,
            "state_keys": len(state),
            "independent_parameter_tensors": len(independent),
            "independent_parameter_count": sum(value.numel() for value in independent),
            "strict": True,
            "all_source_tensors_equal": True,
            "input_interface_added": "P1:48->96",
            "spatial_adaptation": "resize_transformed_upstream_to_actual_task_shape",
            "legacy_meta_attributes_used": False,
        }
        if (receipt["state_keys"] != OFFICIAL_STATE_KEYS
                or receipt["independent_parameter_tensors"] != OFFICIAL_INDEPENDENT_PARAMETERS
                or receipt["independent_parameter_count"] != OFFICIAL_PARAMETER_COUNT):
            raise RuntimeError(f"Unexpected official FANet structure: {receipt}")
        self.load_receipt = receipt
        return dict(receipt)

    def set_trainable_partition(self) -> dict[str, Any]:
        """Train input adapter, official final convs and all LayerNorm parameters.

        Early feature convolutions are frozen. Partition decisions are visible
        and deduplicated, so aliased MetaConv parameters cannot enter twice.
        This is an initialization choice for preparation, not a trained claim.
        """
        for parameter in self.official.parameters():
            parameter.requires_grad_(False)
        train_modules: list[nn.Module] = []
        for level in (1, 2, 3):
            train_modules += [
                getattr(self.official.FT, f"FT{level}").conv3,
                getattr(self.official.MFG, f"MFG{level}").conv_task[7],
            ]
        # Official norm modules are unambiguous by class name and body layout.
        for module in self.official.modules():
            if module.__class__.__name__ == "WithBias_LayerNorm":
                train_modules.append(module)
        for module in train_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self.p1_adapter.requires_grad_(True)
        return self.partition_summary()

    def partition_summary(self) -> dict[str, Any]:
        trainable = [(name, value) for name, value in self.named_parameters() if value.requires_grad]
        frozen = [(name, value) for name, value in self.named_parameters() if not value.requires_grad]
        return {
            "trainable_names": [name for name, _ in trainable],
            "frozen_names": [name for name, _ in frozen],
            "trainable_count": sum(value.numel() for _, value in trainable),
            "frozen_count": sum(value.numel() for _, value in frozen),
            "interface_count": sum(value.numel() for value in self.p1_adapter.parameters()),
            "policy": "P1_interface+FT_final_convs+MFG_final_convs+all_FANet_LayerNorm",
        }

    @staticmethod
    def _feature(features: Mapping[str, torch.Tensor], local: str, official: str) -> torch.Tensor:
        if local in features:
            return features[local]
        if official in features:
            return features[official]
        raise KeyError(f"Missing {local} / {official}")

    def forward(
        self,
        upstream_features: Mapping[str, torch.Tensor],
        task_features: Mapping[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        dem, task = {}, {}
        for level, pc, tc in ((1, 48, 96), (2, 96, 96), (3, 192, 192)):
            p = self._feature(upstream_features, f"P{level}", f"DF{level}")
            t = self._feature(task_features, f"T{level}", f"TaF{level}")
            if p.ndim != 4 or t.ndim != 4 or p.shape[1] != pc or t.shape[1] != tc:
                raise ValueError(f"Level {level}: expected P{pc},T{tc}; got {p.shape},{t.shape}")
            if p.shape[0] != t.shape[0] or p.device != t.device or p.dtype != t.dtype:
                raise ValueError(f"Level {level}: batch/device/dtype mismatch")
            dem[f"DF{level}"] = self.p1_adapter(p) if level == 1 else p
            task[f"TaF{level}"] = t

        transformed = {}
        for level in (1, 2, 3):
            feature = getattr(self.official.FT, f"FT{level}")(dem[f"DF{level}"], meta=False)
            size = task[f"TaF{level}"].shape[-2:]
            if feature.shape[-2:] != size:
                feature = F.interpolate(feature, size=size, mode="bilinear", align_corners=False)
            transformed[f"TF{level}"] = feature
        matched = self.official.MFG(dem, task, meta=False)
        return transformed, matched

    def alignment_loss(
        self, upstream_features: Mapping[str, torch.Tensor], task_features: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        transformed, matched = self(upstream_features, task_features)
        return sum(
            1 - F.cosine_similarity(transformed[f"TF{level}"], matched[f"MF{level}"], dim=1).mean()
            for level in (1, 2, 3)
        ) / 3
