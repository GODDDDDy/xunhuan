from __future__ import annotations

import csv
import json
import hashlib
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import ExperimentConfig, ensure_output_path

ANGLES = ("000", "045", "090", "135")


@dataclass(frozen=True)
class ManifestRow:
    split: str
    difficulty: str
    scene: str
    frame: str
    sample_id: str
    polar_000: str
    polar_045: str
    polar_090: str
    polar_135: str
    official_rgb: str
    gt_rgb: str


def polar_mean(polar: torch.Tensor) -> torch.Tensor:
    """Frozen Phase-0 predictor g(P): mean of four three-channel angles."""
    if polar.shape[-3] != 12:
        raise ValueError(f"Expected 12 polar channels, got {polar.shape}")
    return torch.stack(torch.chunk(polar, 4, dim=-3), dim=0).mean(dim=0)


def _rows_for_root(root: Path, split: str, difficulty: str) -> list[ManifestRow]:
    input_root = root / "input"
    gt_root = root / "gt"
    rows: list[ManifestRow] = []
    for scene_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        scene = scene_dir.name.zfill(2)
        gt_candidates = sorted((gt_root / scene_dir.name).glob("*_rgb.png"))
        if len(gt_candidates) != 1:
            raise ValueError(
                f"Expected one static scene GT RGB for {split}/{difficulty}/{scene}, "
                f"found {len(gt_candidates)}"
            )
        gt_rgb = gt_candidates[0]
        for base in sorted(scene_dir.glob("*_000.png")):
            frame = base.name[: -len("_000.png")]
            paths = {angle: scene_dir / f"{frame}_{angle}.png" for angle in ANGLES}
            official_rgb = scene_dir / f"{frame}_rgb.png"
            required = [*paths.values(), official_rgb, gt_rgb]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Incomplete sample {split}/{difficulty}/{scene}/{frame}: {missing}"
                )
            sample_id = f"{split}/{difficulty}/{scene}/{frame}"
            rows.append(
                ManifestRow(
                    split=split,
                    difficulty=difficulty,
                    scene=scene,
                    frame=frame,
                    sample_id=sample_id,
                    polar_000=str(paths["000"].resolve()),
                    polar_045=str(paths["045"].resolve()),
                    polar_090=str(paths["090"].resolve()),
                    polar_135=str(paths["135"].resolve()),
                    official_rgb=str(official_rgb.resolve()),
                    gt_rgb=str(gt_rgb.resolve()),
                )
            )
    return rows


def build_manifest(config: ExperimentConfig) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for difficulty in ("easy", "hard"):
        rows.extend(_rows_for_root(config.dataset_root / "train" / difficulty, "train", difficulty))
    rows.extend(_rows_for_root(config.dataset_root / "test", "test", "official"))

    validation_ids = config.validation_scene_ids
    assigned = [
        ManifestRow(
            **{
                **asdict(row),
                "split": "validation",
                "sample_id": row.sample_id.replace("train/", "validation/", 1),
            }
        )
        if row.split == "train" and row.scene in validation_ids
        else row
        for row in rows
    ]
    validate_manifest(assigned, config, check_expected_counts=True)
    destination = ensure_output_path(config, config.path("paths", "manifest"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ManifestRow.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(row) for row in assigned)
    audit = manifest_audit(assigned)
    audit_path = ensure_output_path(config, config.path("paths", "split_audit"))
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return assigned


def load_manifest(path: str | Path) -> list[ManifestRow]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [ManifestRow(**row) for row in csv.DictReader(handle)]


def manifest_audit(rows: Sequence[ManifestRow]) -> dict[str, object]:
    counts: dict[str, int] = {}
    scenes: dict[str, set[str]] = {}
    for row in rows:
        counts[row.split] = counts.get(row.split, 0) + 1
        scenes.setdefault(row.split, set()).add(row.scene)
    return {
        "frame_counts": counts,
        "scene_counts": {key: len(value) for key, value in scenes.items()},
        "scene_ids": {key: sorted(value) for key, value in scenes.items()},
    }


def validate_manifest(
    rows: Sequence[ManifestRow], config: ExperimentConfig, check_expected_counts: bool
) -> None:
    ids = [row.sample_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest contains duplicate split/difficulty/scene/frame IDs")
    scenes = {
        name: {row.scene for row in rows if row.split == name}
        for name in ("train", "validation", "test")
    }
    if scenes["train"] & scenes["validation"]:
        raise ValueError("Scene leakage between train and validation")
    # Official test scene numbers are split-local and may repeat train labels,
    # so leakage is audited by source files/root rather than numeric label.
    fields = ("polar_000", "polar_045", "polar_090", "polar_135", "official_rgb", "gt_rgb")
    paths_by_split = {
        name: {
            str(Path(getattr(row, field)).resolve()).casefold()
            for row in rows
            if row.split == name
            for field in fields
        }
        for name in ("train", "validation", "test")
    }
    if paths_by_split["train"] & paths_by_split["validation"]:
        raise ValueError("File leakage between train and validation")
    if (paths_by_split["train"] | paths_by_split["validation"]) & paths_by_split["test"]:
        raise ValueError("File leakage between development data and official test")
    if check_expected_counts:
        audit = manifest_audit(rows)["frame_counts"]
        expected = config.raw["split"]
        for name, key in (
            ("train", "expected_train_frames"),
            ("validation", "expected_validation_frames"),
            ("test", "expected_test_frames"),
        ):
            if audit.get(name, 0) != int(expected[key]):
                raise ValueError(f"{name} frame count {audit.get(name, 0)} != {expected[key]}")


def _read_rgb(path: str) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def select_condition(group: str, polar: torch.Tensor, official_rgb: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Select the only experimental variable; PolarFree still receives official_rgb."""
    if group == "A0":
        return polar_mean(polar)
    if group == "A1":
        return official_rgb.clone()
    if group == "A2":
        return target.clone()
    raise ValueError("group must be A0, A1, or A2")


class PairedBranchDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Shared A0/A1/A2 dataset; only condition selection differs."""

    def __init__(
        self,
        rows: Sequence[ManifestRow],
        group: str,
        crop_size: int | None = None,
        schedule: Sequence[dict[str, int]] | None = None,
    ):
        if group not in {"A0", "A1", "A2"}:
            raise ValueError("group must be A0, A1, or A2")
        self.rows = list(rows)
        self.group = group
        self.crop_size = crop_size
        self.schedule = list(schedule) if schedule is not None else None

    def __len__(self) -> int:
        return len(self.schedule) if self.schedule is not None else len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        schedule_item = self.schedule[index] if self.schedule is not None else None
        row_index = int(schedule_item["sample_index"]) if schedule_item is not None else index
        row = self.rows[row_index]
        polar = torch.cat([_read_rgb(getattr(row, f"polar_{angle}")) for angle in ANGLES])
        official_rgb = _read_rgb(row.official_rgb)
        target = _read_rgb(row.gt_rgb)
        condition = select_condition(self.group, polar, official_rgb, target)
        if self.crop_size is not None:
            if schedule_item is None:
                polar, official_rgb, target, condition = _paired_random_crop(
                    polar, official_rgb, target, condition, size=self.crop_size
                )
            else:
                polar, official_rgb, target, condition = _paired_crop_at(
                    polar,
                    official_rgb,
                    target,
                    condition,
                    size=self.crop_size,
                    top=int(schedule_item["top"]),
                    left=int(schedule_item["left"]),
                )
        return {
            "polar": polar,
            "condition": condition,
            "official_rgb": official_rgb,
            "target": target,
            "sample_id": row.sample_id,
            "scene": row.scene,
        }


class InterventionDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Group data with deterministic within/cross-scene condition donors."""

    def __init__(self, rows: Sequence[ManifestRow], group: str):
        self.base = PairedBranchDataset(rows, group=group, crop_size=None)
        by_scene: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            by_scene.setdefault(row.scene, []).append(index)
        self.within_scene_donor: dict[int, int] = {}
        for indices in by_scene.values():
            if len(indices) < 2:
                raise ValueError("Shuffled-condition intervention needs at least two frames per scene")
            for position, index in enumerate(indices):
                self.within_scene_donor[index] = indices[(position + 1) % len(indices)]
        self.cross_scene_donor: dict[int, int] = {}
        all_indices = list(range(len(rows)))
        for index, row in enumerate(rows):
            for offset in range(1, len(rows)):
                candidate = all_indices[(index + offset) % len(rows)]
                if rows[candidate].scene != row.scene:
                    self.cross_scene_donor[index] = candidate
                    break
            if index not in self.cross_scene_donor:
                raise ValueError("Cross-scene intervention needs at least two scenes")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.base[index]
        within = self.base[self.within_scene_donor[index]]
        cross = self.base[self.cross_scene_donor[index]]
        sample["mismatch_within_condition"] = within["condition"]
        sample["mismatch_cross_condition"] = cross["condition"]
        return sample


def _paired_random_crop(*tensors: torch.Tensor, size: int) -> tuple[torch.Tensor, ...]:
    height, width = tensors[0].shape[-2:]
    if any(tensor.shape[-2:] != (height, width) for tensor in tensors[1:]):
        raise ValueError("Paired tensors must have identical spatial dimensions")
    if min(height, width) < size:
        raise ValueError(f"Crop {size} exceeds image size {(height, width)}")
    top = random.randint(0, height - size)
    left = random.randint(0, width - size)
    return tuple(tensor[..., top : top + size, left : left + size] for tensor in tensors)


def _paired_crop_at(
    *tensors: torch.Tensor, size: int, top: int, left: int
) -> tuple[torch.Tensor, ...]:
    height, width = tensors[0].shape[-2:]
    if any(tensor.shape[-2:] != (height, width) for tensor in tensors[1:]):
        raise ValueError("Paired tensors must have identical spatial dimensions")
    if top < 0 or left < 0 or top + size > height or left + size > width:
        raise ValueError(f"Scheduled crop {(top, left, size)} exceeds image size {(height, width)}")
    return tuple(tensor[..., top : top + size, left : left + size] for tensor in tensors)


def build_training_schedule(
    rows: Sequence[ManifestRow],
    *,
    seed: int,
    total_steps: int,
    batch_size: int,
    crop_size: int,
) -> dict[str, object]:
    """Create a group-independent immutable sample/crop schedule."""
    generator = torch.Generator().manual_seed(seed)
    count = total_steps * batch_size
    indices: list[int] = []
    while len(indices) < count:
        indices.extend(torch.randperm(len(rows), generator=generator).tolist())
    items: list[dict[str, int]] = []
    for sample_index in indices[:count]:
        with Image.open(rows[sample_index].polar_000) as image:
            width, height = image.size
        if min(height, width) < crop_size:
            raise ValueError(f"Crop {crop_size} exceeds image size {(height, width)}")
        top = int(torch.randint(0, height - crop_size + 1, (1,), generator=generator).item())
        left = int(torch.randint(0, width - crop_size + 1, (1,), generator=generator).item())
        items.append({"sample_index": sample_index, "top": top, "left": left})
    payload: dict[str, object] = {
        "schema_version": 1,
        "seed": seed,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "crop_size": crop_size,
        "items": items,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def write_training_schedule(config: ExperimentConfig, rows: Sequence[ManifestRow]) -> Path:
    optimization = config.raw["optimization"]
    schedule = build_training_schedule(
        rows,
        seed=config.seed,
        total_steps=int(optimization["extended_steps"]),
        batch_size=int(config.raw["data"]["batch_size"]),
        crop_size=int(config.raw["data"]["crop_size"]),
    )
    destination = ensure_output_path(config, config.path("paths", "schedule"))
    destination.write_text(json.dumps(schedule, indent=2), encoding="utf-8")
    return destination


def load_training_schedule(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    claimed = str(payload.pop("sha256"))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    actual = hashlib.sha256(canonical).hexdigest()
    payload["sha256"] = claimed
    if actual != claimed:
        raise ValueError("Training schedule hash mismatch")
    return payload


def select_rows(rows: Iterable[ManifestRow], split: str) -> list[ManifestRow]:
    return [row for row in rows if row.split == split]


def build_validation_proxy(
    rows: Sequence[ManifestRow], frames_per_stratum: int
) -> dict[str, object]:
    """Freeze an evenly spaced proxy across every scene/difficulty stratum."""
    validation_rows = [row for row in rows if row.split == "validation"]
    strata: dict[tuple[str, str], list[ManifestRow]] = {}
    for row in validation_rows:
        strata.setdefault((row.scene, row.difficulty), []).append(row)
    selected: list[ManifestRow] = []
    counts: dict[str, int] = {}
    for (scene, difficulty), values in sorted(strata.items()):
        ordered = sorted(values, key=lambda row: (row.frame, row.sample_id))
        if len(ordered) < frames_per_stratum:
            raise ValueError(
                f"Validation stratum {scene}/{difficulty} has only {len(ordered)} frames"
            )
        if frames_per_stratum == 1:
            indices = [len(ordered) // 2]
        else:
            indices = [
                round(index * (len(ordered) - 1) / (frames_per_stratum - 1))
                for index in range(frames_per_stratum)
            ]
        if len(set(indices)) != frames_per_stratum:
            raise AssertionError("Evenly spaced validation proxy indices are not unique")
        chosen = [ordered[index] for index in indices]
        selected.extend(chosen)
        counts[f"{scene}/{difficulty}"] = len(chosen)
    sample_ids = [row.sample_id for row in selected]
    canonical = json.dumps(sample_ids, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": 1,
        "selection": "evenly_spaced_within_scene_and_difficulty",
        "frames_per_stratum": frames_per_stratum,
        "sample_count": len(sample_ids),
        "stratum_counts": counts,
        "sample_ids": sample_ids,
        "sample_ids_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def write_validation_proxy(
    config: ExperimentConfig, rows: Sequence[ManifestRow]
) -> Path:
    payload = build_validation_proxy(
        rows, int(config.raw["validation"]["proxy_frames_per_scene_difficulty"])
    )
    destination = ensure_output_path(config, config.path("paths", "validation_proxy"))
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def load_validation_proxy_rows(
    config: ExperimentConfig, rows: Sequence[ManifestRow]
) -> list[ManifestRow]:
    path = config.path("paths", "validation_proxy")
    if not path.is_file():
        raise FileNotFoundError(f"Validation proxy manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = build_validation_proxy(
        rows, int(config.raw["validation"]["proxy_frames_per_scene_difficulty"])
    )
    if payload != expected:
        raise ValueError("Validation proxy manifest differs from the frozen selection")
    by_id = {row.sample_id: row for row in rows}
    return [by_id[sample_id] for sample_id in payload["sample_ids"]]
