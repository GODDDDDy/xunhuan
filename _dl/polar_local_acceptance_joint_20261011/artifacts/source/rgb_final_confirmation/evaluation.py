from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from .config import ExperimentConfig, ensure_output_path
from .data import polar_mean


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(prediction, target).item()
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


def ssim(prediction: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    coordinates = torch.arange(window_size, device=prediction.device, dtype=prediction.dtype)
    gaussian = torch.exp(-((coordinates - window_size // 2) ** 2) / (2 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    window = torch.outer(gaussian, gaussian).expand(
        prediction.shape[1], 1, window_size, window_size
    )
    mean_x = F.conv2d(prediction, window, padding=window_size // 2, groups=prediction.shape[1])
    mean_y = F.conv2d(target, window, padding=window_size // 2, groups=prediction.shape[1])
    var_x = (
        F.conv2d(prediction.square(), window, padding=window_size // 2, groups=prediction.shape[1])
        - mean_x.square()
    )
    var_y = (
        F.conv2d(target.square(), window, padding=window_size // 2, groups=prediction.shape[1])
        - mean_y.square()
    )
    covariance = (
        F.conv2d(prediction * target, window, padding=window_size // 2, groups=prediction.shape[1])
        - mean_x * mean_y
    )
    score = ((2 * mean_x * mean_y + 0.01**2) * (2 * covariance + 0.03**2)) / (
        (mean_x.square() + mean_y.square() + 0.01**2) * (var_x + var_y + 0.03**2)
    ).clamp_min(1e-8)
    return float(score.mean())


def condition_intervention(
    name: str,
    polar: torch.Tensor,
    official_rgb: torch.Tensor,
    target: torch.Tensor,
    current_condition: torch.Tensor,
    *,
    mismatch_rgb: torch.Tensor | None = None,
    seed: int = 100,
) -> torch.Tensor:
    if name == "correct_condition":
        return current_condition
    if name == "polar_mean":
        return polar_mean(polar)
    if name == "raw_rgb":
        return official_rgb
    if name == "transmission_gt":
        return target
    if name in {"shuffled_within_scene", "shuffled_cross_scene"}:
        if mismatch_rgb is None:
            raise ValueError(f"{name} intervention requires a deterministic donor")
        return mismatch_rgb
    if name == "zero_condition":
        return torch.zeros_like(official_rgb)
    if name == "matched_noise":
        generator = torch.Generator(device=official_rgb.device).manual_seed(seed)
        noise = torch.randn(
            official_rgb.shape,
            generator=generator,
            device=official_rgb.device,
            dtype=official_rgb.dtype,
        )
        mean = official_rgb.mean(dim=(-2, -1), keepdim=True)
        std = official_rgb.std(dim=(-2, -1), keepdim=True, unbiased=False)
        return (mean + std * noise).clamp(0, 1)
    raise ValueError(f"Unknown intervention: {name}")


def stable_sample_seed(sample_id: str, base_seed: int = 100) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], "little")


def save_prediction(tensor: torch.Tensor, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().clamp(0, 1).mul(255).round().byte().cpu().permute(1, 2, 0).numpy()
    Image.fromarray(array).save(destination)


@torch.no_grad()
def evaluate_loader(
    r2p: nn.Module,
    task: nn.Module,
    loader: Iterable[Mapping[str, object]],
    device: torch.device,
    output_root: Path | None = None,
    intervention: str | None = None,
    base_seed: int = 2026,
) -> list[dict[str, object]]:
    r2p.eval()
    task.eval()
    results: list[dict[str, object]] = []
    for batch in loader:
        polar = batch["polar"].to(device)  # type: ignore[union-attr]
        condition = batch["condition"].to(device)  # type: ignore[union-attr]
        official_rgb = batch["official_rgb"].to(device)  # type: ignore[union-attr]
        target = batch["target"].to(device)  # type: ignore[union-attr]
        sample_ids = list(batch["sample_id"])  # type: ignore[arg-type]
        scenes = list(batch["scene"])  # type: ignore[arg-type]
        if len(sample_ids) != 1:
            raise ValueError("Deterministic evaluation requires batch size 1")
        if intervention is not None:
            mismatch = None
            if intervention == "shuffled_within_scene":
                mismatch = batch.get("mismatch_within_condition")
            elif intervention == "shuffled_cross_scene":
                mismatch = batch.get("mismatch_cross_condition")
            if isinstance(mismatch, torch.Tensor):
                mismatch = mismatch.to(device)
            condition = condition_intervention(
                intervention,
                polar,
                official_rgb,
                target,
                condition,
                mismatch_rgb=mismatch,  # type: ignore[arg-type]
                seed=stable_sample_seed(sample_ids[0], base_seed=base_seed),
            )
        sample_seed = stable_sample_seed(sample_ids[0], base_seed=base_seed)
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)
        r2p_output = r2p(polar, condition)
        prediction, _ = task(r2p_output.refined_polar, official_rgb, None)
        for index, (sample_id, scene) in enumerate(zip(sample_ids, scenes)):
            row = {
                "sample_id": sample_id,
                "scene": scene,
                "psnr": psnr(prediction[index : index + 1], target[index : index + 1]),
                "ssim": ssim(prediction[index : index + 1], target[index : index + 1]),
            }
            results.append(row)
            if output_root is not None:
                frame = sample_id.rsplit("/", 1)[-1]
                save_prediction(prediction[index], output_root / scene / f"{frame}.png")
    return results


@torch.no_grad()
def evaluate_official_task_loader(
    task: nn.Module,
    loader: Iterable[Mapping[str, object]],
    device: torch.device,
    output_root: Path,
    base_seed: int = 2026,
) -> list[dict[str, object]]:
    """Fresh official PolarFree baseline without R2P or collaboration."""
    task.eval()
    results: list[dict[str, object]] = []
    for batch in loader:
        polar = batch["polar"].to(device)  # type: ignore[union-attr]
        official_rgb = batch["official_rgb"].to(device)  # type: ignore[union-attr]
        target = batch["target"].to(device)  # type: ignore[union-attr]
        sample_ids = list(batch["sample_id"])  # type: ignore[arg-type]
        scenes = list(batch["scene"])  # type: ignore[arg-type]
        if len(sample_ids) != 1:
            raise ValueError("Official baseline evaluation requires batch size 1")
        sample_seed = stable_sample_seed(sample_ids[0], base_seed=base_seed)
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)
        prediction, _ = task.forward_official_polarfree(polar, official_rgb, None)
        sample_id, scene = sample_ids[0], scenes[0]
        results.append(
            {
                "sample_id": sample_id,
                "scene": scene,
                "psnr": psnr(prediction, target),
                "ssim": ssim(prediction, target),
            }
        )
        frame = sample_id.rsplit("/", 1)[-1]
        save_prediction(prediction[0], output_root / scene / f"{frame}.png")
    return results


def aggregate_metrics(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    rows = list(rows)
    by_scene: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_scene[str(row["scene"])].append(row)
    scene_rows = []
    for scene, values in sorted(by_scene.items()):
        scene_rows.append(
            {
                "scene": scene,
                "frames": len(values),
                "psnr": float(np.mean([float(row["psnr"]) for row in values])),
                "ssim": float(np.mean([float(row["ssim"]) for row in values])),
            }
        )
    return {
        "image_mean_psnr": float(np.mean([float(row["psnr"]) for row in rows])),
        "image_mean_ssim": float(np.mean([float(row["ssim"]) for row in rows])),
        "scene_macro_psnr": float(np.mean([row["psnr"] for row in scene_rows])),
        "scene_macro_ssim": float(np.mean([row["ssim"] for row in scene_rows])),
        "scenes": scene_rows,
    }


def write_metric_csv(
    config: ExperimentConfig, rows: list[dict[str, object]], path: str | Path
) -> None:
    destination = ensure_output_path(config, path)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "scene", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)
