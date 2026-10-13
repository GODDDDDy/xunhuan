from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import torch

from .config import ExperimentConfig, ensure_output_path


def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_dict_tensor_hashes(state: Mapping[str, torch.Tensor]) -> dict[str, str]:
    return {name: tensor_sha256(state[name]) for name in sorted(state)}


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value_hash in state_dict_tensor_hashes(state).items():
        digest.update(name.encode("utf-8"))
        digest.update(value_hash.encode("ascii"))
    return digest.hexdigest()


def implementation_sha256() -> str:
    """Fingerprint the code that defines the R2P state and forward graph.

    Training orchestration, data loading, and the frozen task network do not
    determine R2P initialization. Including them would incorrectly invalidate
    the shared state when only a continuation policy or audit changes.
    """
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in ("r2p.py",):
        digest.update(name.encode("utf-8"))
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def model_config_sha256(config: ExperimentConfig) -> str:
    """Fingerprint only the fields that determine R2P initialization."""
    payload = json.dumps(
        {"seed": config.seed, "model": config.raw["model"]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_or_create_r2p_initial_state(
    r2p: torch.nn.Module, config: ExperimentConfig
) -> tuple[Path, dict[str, str]]:
    """Create/load the one complete R2P state shared by A0/A1/A2."""
    checkpoint = ensure_output_path(config, config.path("paths", "initial_state"))
    sidecar = ensure_output_path(config, config.path("paths", "initial_state_audit"))
    config_hash = model_config_sha256(config)
    implementation_hash = implementation_sha256()
    if checkpoint.exists():
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        if int(payload["seed"]) != config.seed:
            raise ValueError("Initial-state seed mismatch")
        if payload["model_config_sha256"] != config_hash:
            raise ValueError("R2P model config changed after the shared initial state was created")
        if payload["implementation_sha256"] != implementation_hash:
            raise ValueError("Implementation changed after the shared initial state was created")
    else:
        state = _cpu_state(r2p)
        payload = {
            "schema_version": 2,
            "seed": config.seed,
            "model_config_sha256": config_hash,
            "implementation_sha256": implementation_hash,
            "r2p": state,
            "tensor_hashes": state_dict_tensor_hashes(state),
            "state_sha256": state_dict_sha256(state),
        }
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(checkpoint)

    r2p.load_state_dict(payload["r2p"], strict=True)
    live_hashes = state_dict_tensor_hashes(r2p.state_dict())
    if live_hashes != payload["tensor_hashes"]:
        raise AssertionError("Loaded R2P state does not match its per-tensor hashes")
    sidecar.write_text(
        json.dumps(
            {
                "seed": config.seed,
                "checkpoint": str(checkpoint),
                "model_config_sha256": config_hash,
                "implementation_sha256": implementation_hash,
                "state_sha256": payload["state_sha256"],
                "tensor_hashes": live_hashes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return checkpoint, live_hashes
