from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any]
    source: Path

    @property
    def project_root(self) -> Path:
        return Path(self.raw["project_root"])

    @property
    def dataset_root(self) -> Path:
        return Path(self.raw["dataset_root"])

    @property
    def experiment_root(self) -> Path:
        return Path(self.raw["experiment_root"])

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(self.raw["groups"])

    @property
    def validation_scene_ids(self) -> frozenset[str]:
        return frozenset(str(x).zfill(2) for x in self.raw["split"]["validation_scene_ids"])

    def path(self, *keys: str) -> Path:
        value: Any = self.raw
        for key in keys:
            value = value[key]
        return Path(value)


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = ExperimentConfig(raw=raw, source=source)
    _validate_config(config)
    return config


def _validate_config(config: ExperimentConfig) -> None:
    if config.raw.get("schema_version") != 2:
        raise ValueError("Unsupported configuration schema")
    bundle = Path(__file__).resolve().parents[4]
    if config.project_root.resolve() != Path(__file__).resolve().parents[2]:
        raise ValueError("project_root must identify the bundled dependency")
    if not config.dataset_root.is_dir():
        raise ValueError("Configured dataset directory does not exist")
    try:
        config.experiment_root.resolve().relative_to(bundle)
    except ValueError as error:
        raise ValueError("experiment_root must remain inside this bundle") from error
    if config.seed != 2026:
        raise ValueError("The screening experiment seed is frozen to 2026")
    if config.groups != ("A0", "A1", "A2"):
        raise ValueError("The trainable groups must be exactly A0, A1, A2")
    if config.validation_scene_ids != {"01", "07", "22", "30", "36", "41", "55", "56"}:
        raise ValueError("Validation scene IDs differ from the preregistered split")
    optimization = config.raw["optimization"]
    budgets = [int(optimization[key]) for key in ("sanity_steps", "pilot_steps", "extended_steps")]
    if budgets != [200, 3000, 5000]:
        raise ValueError("Progressive budgets must remain 200/3000/5000 steps")
    validation_steps = [int(step) for step in optimization["validation_steps"]]
    if validation_steps != [200, 1000, 2000, 2500, 3000, 4000, 4500, 5000]:
        raise ValueError("Validation milestones differ from the reviewed schedule")
    validation = config.raw["validation"]
    if int(validation["proxy_frames_per_scene_difficulty"]) != 2:
        raise ValueError("The reviewed validation proxy must use two frames per stratum")
    if [int(step) for step in validation["full_steps"]] != []:
        raise ValueError("Full validation is disabled for this reviewed screening run")
    if float(config.raw["model"]["residual_alpha"]) != 0.05:
        raise ValueError("The common bounded-residual alpha must remain 0.05")
    preprocessing = config.raw["preprocessing"]
    if bool(preprocessing["allow_approximation"]):
        bound = preprocessing["approved_max_abs_error"]
        if bound is None or not float(bound) >= 0:
            raise ValueError("Approved preprocessing approximation requires a nonnegative bound")
    loss = config.raw["loss"]
    if bool(loss["locked"]):
        for key in ("lambda_feature_alignment", "lambda_residual"):
            value = loss[key]
            if value is None or not float(value) > 0:
                raise ValueError(f"Locked loss config requires a positive {key}")


def require_locked_loss_weights(config: ExperimentConfig) -> tuple[float, float]:
    loss = config.raw["loss"]
    if not bool(loss["locked"]):
        raise RuntimeError(
            "Formal training is blocked: review the calibration proposal, set both auxiliary "
            "weights, and change loss.locked to true."
        )
    return float(loss["lambda_feature_alignment"]), float(loss["lambda_residual"])


def ensure_output_path(config: ExperimentConfig, path: str | Path) -> Path:
    candidate = Path(path).resolve()
    try:
        candidate.relative_to(config.experiment_root.resolve())
    except ValueError as error:
        raise ValueError(f"Output path escapes experiment root: {candidate}") from error
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate
