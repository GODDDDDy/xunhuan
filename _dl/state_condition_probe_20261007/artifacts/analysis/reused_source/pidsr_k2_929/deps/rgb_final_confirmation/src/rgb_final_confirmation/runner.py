from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig, ensure_output_path, require_locked_loss_weights
from .data import (
    InterventionDataset,
    PairedBranchDataset,
    build_manifest,
    load_manifest,
    load_training_schedule,
    load_validation_proxy_rows,
    manifest_audit,
    select_rows,
    validate_manifest,
    write_training_schedule,
    write_validation_proxy,
    _read_rgb,
)
from .evaluation import aggregate_metrics, evaluate_loader, evaluate_official_task_loader, write_metric_csv
from .initialization import load_or_create_r2p_initial_state, state_dict_sha256
from .losses import build_official_task_loss
from .polarfree import (
    OfficialPolarFreeTask,
    official_polarfree_task_inputs,
    refined_polar_to_task_inputs,
)
from .r2p import ConditionalR2P
from .training import Batch, ExperimentTrainer, set_seed, set_step_seed


def build_r2p(config: ExperimentConfig) -> ConditionalR2P:
    values = config.raw["model"]
    return ConditionalR2P(
        widths=tuple(values["widths"]),
        task_feature_channels=tuple(values["task_feature_channels"]),
        blocks_per_stage=int(values["blocks_per_stage"]),
        groups=int(values["group_norm_groups"]),
        attention_heads=int(values["attention_heads"]),
        attention_window=int(values["attention_window"]),
        residual_alpha=float(values["residual_alpha"]),
        frm_lambda_channel=float(values["frm_lambda_channel"]),
        frm_lambda_spatial=float(values["frm_lambda_spatial"]),
    )


def build_components(
    config: ExperimentConfig,
    device: torch.device,
    *,
    require_locked: bool,
) -> ExperimentTrainer:
    require_preprocessing_parity(config)
    if require_locked:
        lambda_fa, lambda_res = require_locked_loss_weights(config)
    else:
        lambda_fa, lambda_res = 0.0, 0.0
    r2p = build_r2p(config).to(device)
    task = OfficialPolarFreeTask(config, device)
    criterion = build_official_task_loss(config).to(device)
    optimization = config.raw["optimization"]
    return ExperimentTrainer(
        r2p,
        task,
        criterion,
        lr_r2p=float(optimization["lr_r2p"]),
        lambda_feature_alignment=lambda_fa,
        lambda_residual=lambda_res,
        grad_clip=float(optimization["grad_clip"]),
    )


def require_preprocessing_parity(config: ExperimentConfig) -> None:
    path = config.path("paths", "preprocessing_parity")
    if not path.is_file():
        raise RuntimeError("Preprocessing parity audit is missing")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if bool(audit.get("all_exact")):
        return
    preprocessing = config.raw["preprocessing"]
    approved = bool(preprocessing["allow_approximation"])
    approved_bound = preprocessing["approved_max_abs_error"]
    observed = float(audit["global_max_abs_error"])
    if not approved or approved_bound is None or observed > float(approved_bound):
        raise RuntimeError(
            "Preprocessing is a measured differentiable approximation and has not been "
            "approved at its observed maximum error"
        )


def audit_preprocessing_parity(
    config: ExperimentConfig, *, samples_per_split: int = 4
) -> dict[str, object]:
    """Compare the tensor path with the released OpenCV/NumPy dataloader on real images."""
    import cv2

    if samples_per_split <= 0:
        raise ValueError("samples_per_split must be positive")

    rows = load_manifest(config.path("paths", "manifest"))
    selected = []
    for split in ("train", "validation", "test"):
        candidates = select_rows(rows, split)
        buckets: dict[str, list[Any]] = {}
        for row in candidates:
            buckets.setdefault(row.difficulty, []).append(row)
        while len([row for row in selected if row.split == split]) < samples_per_split:
            added = False
            for difficulty in sorted(buckets):
                if buckets[difficulty]:
                    selected.append(buckets[difficulty].pop(0))
                    added = True
                    if len([row for row in selected if row.split == split]) == samples_per_split:
                        break
            if not added:
                break
    component_names = ("gray0", "gray45", "gray90", "gray135", "aolp", "dolp")
    component_errors: dict[str, list[torch.Tensor]] = {name: [] for name in component_names}
    component_exact: dict[str, bool] = {name: True for name in component_names}

    for row in selected:
        color = [_read_rgb(getattr(row, f"polar_{angle}")) for angle in ("000", "045", "090", "135")]
        polar = torch.cat(color).unsqueeze(0)
        rgb = _read_rgb(row.official_rgb).unsqueeze(0)
        local = official_polarfree_task_inputs(polar, rgb)[1:]

        official_gray = []
        for angle in ("000", "045", "090", "135"):
            bgr = cv2.imread(getattr(row, f"polar_{angle}"), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(getattr(row, f"polar_{angle}"))
            rgb_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY) / 255.0
            official_gray.append(gray)
        g0, g45, g90, g135 = official_gray
        intensity = 0.5 * (g0 + g45 + g90 + g135) + 1e-4
        q, u = g0 - g90, g45 - g135
        q[q == 0] = 1e-4
        intensity[intensity == 0] = 1e-4
        dolp = np.sqrt(q**2 + u**2) / intensity
        dolp[dolp > 1] = 1
        aolp = 0.5 * np.arctan2(u, q)
        official_arrays = (*official_gray, aolp, dolp)
        official = [
            torch.from_numpy(np.ascontiguousarray(value.copy())).float().unsqueeze(0).unsqueeze(0)
            for value in official_arrays
        ]
        for name, local_value, official_value in zip(
            component_names, local, official, strict=True
        ):
            difference = (local_value.cpu() - official_value).abs()
            component_errors[name].append(difference)
            component_exact[name] = component_exact[name] and torch.equal(
                local_value.cpu(), official_value
            )

    components = {}
    for name in component_names:
        values = torch.cat([value.flatten() for value in component_errors[name]])
        components[name] = {
            "exact": component_exact[name],
            "max_abs_error": float(values.max()),
            "mean_abs_error": float(values.mean()),
            "nonzero_fraction": float((values != 0).float().mean()),
        }
    all_exact = all(component_exact.values())
    global_max = max(float(value["max_abs_error"]) for value in components.values())
    result = {
        "status": "exact" if all_exact else "differentiable_approximation",
        "all_exact": all_exact,
        "global_max_abs_error": global_max,
        "samples": [row.sample_id for row in selected],
        "components": components,
        "forward": "OpenCV uint8 fixed-point equivalent with float64 AoLP/DoLP",
        "backward": "straight-through floating-point luminance surrogate",
    }
    destination = ensure_output_path(config, config.path("paths", "preprocessing_parity"))
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def prepare_experiment(config: ExperimentConfig) -> dict[str, object]:
    rows = build_manifest(config)
    schedule = write_training_schedule(config, select_rows(rows, "train"))
    proxy = write_validation_proxy(config, rows)
    return {
        "manifest": manifest_audit(rows),
        "schedule": str(schedule),
        "validation_proxy": str(proxy),
    }


def prepare_validation_proxy(config: ExperimentConfig) -> dict[str, object]:
    rows = load_manifest(config.path("paths", "manifest"))
    validate_manifest(rows, config, check_expected_counts=True)
    path = write_validation_proxy(config, rows)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"path": str(path), **payload}


def _scheduled_loader(
    rows: list[Any],
    config: ExperimentConfig,
    group: str,
    end_step: int,
    *,
    start_step: int = 0,
) -> DataLoader:
    payload = load_training_schedule(config.path("paths", "schedule"))
    if int(payload["seed"]) != config.seed:
        raise ValueError("Training schedule seed mismatch")
    batch_size = int(config.raw["data"]["batch_size"])
    if start_step < 0 or end_step <= start_step:
        raise ValueError("Scheduled step range must satisfy 0 <= start_step < end_step")
    start_item = start_step * batch_size
    end_item = end_step * batch_size
    items = list(payload["items"])
    if len(items) < end_item:
        raise ValueError("Training schedule is shorter than the requested run")
    dataset = PairedBranchDataset(
        rows,
        group=group,
        crop_size=int(config.raw["data"]["crop_size"]),
        schedule=items[start_item:end_item],
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config.raw["data"]["workers"]),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def _evaluation_loader(rows: list[Any], group: str) -> DataLoader:
    return DataLoader(
        PairedBranchDataset(rows, group=group, crop_size=None),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )


def _append_csv(path: Path, values: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values))
        if not exists:
            writer.writeheader()
        writer.writerow(values)


def _numeric_summary(records: list[dict[str, float]], keys: tuple[str, ...]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key in keys:
        values = np.asarray([record[key] for record in records if key in record], dtype=np.float64)
        if values.size:
            result[key] = {
                "mean": float(values.mean()),
                "max": float(values.max()),
                "last": float(values[-1]),
            }
    return result


def _write_diagnostic_summaries(
    config: ExperimentConfig,
    group: str,
    steps: int,
    records: list[dict[str, float]],
) -> None:
    root = config.path("paths", "outputs") / "diagnostics"
    categories = {
        "residual_stats": (
            "delta_abs_mean",
            "delta_abs_max",
            "delta_abs_000",
            "delta_abs_045",
            "delta_abs_090",
            "delta_abs_135",
            "clamp_fraction",
        ),
        "gradient_stats": (
            "r2p_grad_norm",
            "polar_encoder_grad_norm",
            "condition_encoder_grad_norm",
            "residual_head_grad_norm",
            "projection_head_grad_norm",
        ),
        "stokes_dolp_aolp_stats": (
            "stokes_s0_change_abs_mean",
            "stokes_s1_change_abs_mean",
            "stokes_s2_change_abs_mean",
            "dolp_change_abs_mean",
            "aolp_wrapped_change_abs_mean",
        ),
    }
    feature_keys = tuple(
        sorted(
            {
                key
                for record in records
                for key in record
                if key.startswith("frm") or key in {"attention_entropy", "attention_contribution_ratio"}
            }
        )
    )
    categories["feature_stats"] = feature_keys
    for name, keys in categories.items():
        destination = ensure_output_path(config, root / f"{group}_{steps}_{name}.json")
        destination.write_text(
            json.dumps(_numeric_summary(records, keys), indent=2), encoding="utf-8"
        )


def _save_checkpoint(
    config: ExperimentConfig,
    trainer: ExperimentTrainer,
    destination: Path,
    step: int,
    validation: dict[str, float] | None = None,
) -> Path:
    path = ensure_output_path(config, destination)
    temporary = path.with_suffix(".tmp")
    torch.save(trainer.checkpoint_state(step, validation), temporary)
    temporary.replace(path)
    return path


def _validate(
    trainer: ExperimentTrainer,
    loader: DataLoader,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    rows = evaluate_loader(trainer.r2p, trainer.task, loader, device, base_seed=seed)
    summary = aggregate_metrics(rows)
    return {
        "image_mean_psnr": float(summary["image_mean_psnr"]),
        "image_mean_ssim": float(summary["image_mean_ssim"]),
        "scene_macro_psnr": float(summary["scene_macro_psnr"]),
        "scene_macro_ssim": float(summary["scene_macro_ssim"]),
    }


def run_loss_calibration(config: ExperimentConfig, device: torch.device) -> dict[str, object]:
    """Write a proposal only; never edits or locks the experiment config."""
    rows = load_manifest(config.path("paths", "manifest"))
    validate_manifest(rows, config, check_expected_counts=True)
    train_rows = select_rows(rows, "train")
    calibration = config.raw["loss"]["calibration"]
    probe_steps = int(calibration["probe_steps"])
    measurement_batches = int(calibration["measurement_batches"])
    records: list[dict[str, object]] = []

    for group in config.groups:
        set_seed(config.seed)
        trainer = build_components(config, device, require_locked=False)
        state_path, tensor_hashes = load_or_create_r2p_initial_state(trainer.r2p, config)
        loader = _scheduled_loader(
            train_rows, config, group, probe_steps + measurement_batches
        )
        iterator = iter(loader)
        for step in range(1, probe_steps + 1):
            batch = Batch.from_mapping(next(iterator), device)
            set_step_seed(config.seed, "calibration_probe", step, batch.sample_id)
            trainer.task_only_probe_step(batch)
        for measurement in range(1, measurement_batches + 1):
            batch = Batch.from_mapping(next(iterator), device)
            set_step_seed(
                config.seed,
                "calibration_measurement",
                measurement,
                batch.sample_id,
            )
            norms = trainer.component_gradient_norms(batch)
            if any(not np.isfinite(value) or value <= 0 for value in norms.values()):
                raise RuntimeError(f"Invalid calibration gradients for {group}: {norms}")
            records.append({"group": group, "measurement": measurement, **norms})

    fraction = float(calibration["auxiliary_gradient_fraction"])
    medians = {
        key: float(np.median([float(record[key]) for record in records]))
        for key in ("task", "feature_alignment", "residual")
    }
    proposal = {
        "schema_version": 1,
        "status": "REQUIRES_HUMAN_REVIEW",
        "seed": config.seed,
        "initial_state": str(state_path),
        "initial_state_tensor_hashes": tensor_hashes,
        "probe_steps_per_group": probe_steps,
        "measurement_batches_per_group": measurement_batches,
        "auxiliary_gradient_fraction": fraction,
        "gradient_parameter_scope": [
            "polar_encoder",
            "condition_encoder",
            "rectifiers",
            "cross_attention",
            "decode2",
            "decode1",
        ],
        "excluded_from_gradient_calibration": [
            "feature_projections",
            "residual_head",
        ],
        "median_gradient_norms": medians,
        "proposed_lambda_feature_alignment": fraction * medians["task"] / medians["feature_alignment"],
        "proposed_lambda_residual": fraction * medians["task"] / medians["residual"],
        "measurements": records,
    }
    destination = ensure_output_path(config, Path(calibration["proposal"]))
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite calibration proposal: {destination}")
    destination.write_text(json.dumps(proposal, indent=2), encoding="utf-8")
    return proposal


def run_training(
    config: ExperimentConfig,
    group: str,
    steps: int,
    device: torch.device,
) -> Path:
    if group not in config.groups:
        raise ValueError(f"group must be one of {config.groups}")
    allowed_steps = {
        int(config.raw["optimization"][key])
        for key in ("sanity_steps", "pilot_steps", "extended_steps")
    }
    if steps not in allowed_steps:
        raise ValueError(f"steps must be one of {sorted(allowed_steps)}")
    if steps == int(config.raw["optimization"]["extended_steps"]):
        metrics_root = config.path("paths", "outputs") / "metrics"
        extension_path = metrics_root / "extension_decision.json"
        runtime_path = metrics_root / "runtime_budget.json"
        if not extension_path.is_file() or not runtime_path.is_file():
            raise RuntimeError("5000-step training requires extension and runtime decisions")
        extension = json.loads(extension_path.read_text(encoding="utf-8"))
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if not bool(extension["extend_all_groups_to_5000"]):
            raise RuntimeError("The validation decision does not authorize 5000-step training")
        if not bool(runtime["within_five_day_total_budget_5000"]):
            raise RuntimeError("The wall-clock decision caps training at 3000 steps")
    rows = load_manifest(config.path("paths", "manifest"))
    validate_manifest(rows, config, check_expected_counts=True)
    train_rows = select_rows(rows, "train")
    validation_rows = select_rows(rows, "validation")
    proxy_rows = load_validation_proxy_rows(config, rows)
    sanity_steps = int(config.raw["optimization"]["sanity_steps"])
    pilot_steps = int(config.raw["optimization"]["pilot_steps"])
    extended_steps = int(config.raw["optimization"]["extended_steps"])
    if steps == sanity_steps:
        start_step = 0
    elif steps == pilot_steps:
        start_step = sanity_steps
    elif steps == extended_steps:
        start_step = pilot_steps
    else:  # Guarded by allowed_steps above; keeps continuation logic explicit.
        raise AssertionError("Unreachable progressive training budget")
    train_loader = _scheduled_loader(
        train_rows, config, group, steps, start_step=start_step
    )
    proxy_validation_loader = _evaluation_loader(proxy_rows, group)
    full_validation_loader = _evaluation_loader(validation_rows, group)

    set_seed(config.seed)
    trainer = build_components(config, device, require_locked=True)
    initial_state_path, initial_hashes = load_or_create_r2p_initial_state(trainer.r2p, config)
    initial_hash = state_dict_sha256(trainer.r2p.state_dict())
    preprocessing_audit = json.loads(
        config.path("paths", "preprocessing_parity").read_text(encoding="utf-8")
    )
    run_root = config.path("paths", "outputs") / "runs" / group / f"steps_{steps}"
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"Refusing to mix with an existing run: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    resume_checkpoint: Path | None = None
    if start_step:
        resume_checkpoint = (
            config.path("paths", "outputs")
            / "checkpoints"
            / f"{group}_{start_step}_last.pt"
        )
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(
                f"The {start_step}-step continuation checkpoint is missing: {resume_checkpoint}"
            )
        try:
            resume_state = torch.load(resume_checkpoint, map_location=device, weights_only=True)
        except TypeError:
            resume_state = torch.load(resume_checkpoint, map_location=device)
        if int(resume_state["step"]) != start_step:
            raise ValueError("Continuation checkpoint step mismatch")
        if float(resume_state["lambda_feature_alignment"]) != trainer.lambda_feature_alignment:
            raise ValueError("Feature-alignment weight changed before continuation")
        if float(resume_state["lambda_residual"]) != trainer.lambda_residual:
            raise ValueError("Residual weight changed before continuation")
        trainer.r2p.load_state_dict(resume_state["r2p"], strict=True)
        trainer.optimizer_r2p.load_state_dict(resume_state["optimizer_r2p"])
    (run_root / "initialization.json").write_text(
        json.dumps(
            {
                "path": str(initial_state_path),
                "state_sha256": initial_hash,
                "tensor_hashes": initial_hashes,
                "resumed_from": str(resume_checkpoint) if resume_checkpoint else None,
                "start_step": start_step,
                "end_step": steps,
                "preprocessing_status": preprocessing_audit["status"],
                "preprocessing_global_max_abs_error": preprocessing_audit[
                    "global_max_abs_error"
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    iterator = iter(train_loader)
    log_path = config.path("paths", "outputs") / "logs" / f"{group}_{steps}_train.csv"
    if start_step:
        prior_history_path = (
            config.path("paths", "outputs")
            / "metrics"
            / f"{group}_{start_step}_validation.json"
        )
        history = json.loads(prior_history_path.read_text(encoding="utf-8"))
    else:
        history = []
    full_history_path = (
        config.path("paths", "outputs")
        / "metrics"
        / f"{group}_{start_step}_full_validation.json"
    )
    full_history = (
        json.loads(full_history_path.read_text(encoding="utf-8"))
        if start_step and full_history_path.is_file()
        else []
    )
    diagnostic_records: list[dict[str, float]] = []
    best: tuple[float, float, int] | None = None
    validation_steps = {
        int(value) for value in config.raw["optimization"]["validation_steps"]
    }
    full_validation_steps = {
        int(value) for value in config.raw["validation"]["full_steps"]
    }
    best_path = config.path("paths", "outputs") / "checkpoints" / f"{group}_{steps}_best.pt"
    if start_step:
        prior_best_record = max(
            history,
            key=lambda record: (
                float(record["scene_macro_psnr"]),
                float(record["scene_macro_ssim"]),
                -int(record["step"]),
            ),
        )
        best = (
            float(prior_best_record["scene_macro_psnr"]),
            float(prior_best_record["scene_macro_ssim"]),
            -int(prior_best_record["step"]),
        )
        prior_best_path = (
            config.path("paths", "outputs")
            / "checkpoints"
            / f"{group}_{start_step}_best.pt"
        )
        if not prior_best_path.is_file():
            raise FileNotFoundError(f"Prior best checkpoint is missing: {prior_best_path}")
        ensure_output_path(config, best_path)
        shutil.copy2(prior_best_path, best_path)
    step_times: list[float] = []
    proxy_validation_times: list[float] = []
    full_validation_times: list[float] = []
    last_validation: dict[str, float] | None = None
    run_started = time.perf_counter()
    for step in range(start_step + 1, steps + 1):
        step_started = time.perf_counter()
        batch = Batch.from_mapping(next(iterator), device)
        noise_seed = set_step_seed(config.seed, "train", step, batch.sample_id)
        diagnostics = trainer.train_step(batch)
        step_seconds = time.perf_counter() - step_started
        step_times.append(step_seconds)
        diagnostic_records.append(diagnostics)
        _append_csv(
            log_path,
            {
                "step": step,
                "diffusion_seed": noise_seed,
                "step_seconds": step_seconds,
                **diagnostics,
            },
        )
        if step in validation_steps or step == steps:
            milestone_path = (
                config.path("paths", "outputs")
                / "checkpoints"
                / f"{group}_{step}_milestone.pt"
            )
            _save_checkpoint(config, trainer, milestone_path, step, None)
            validation_started = time.perf_counter()
            metrics = _validate(trainer, proxy_validation_loader, device, config.seed)
            proxy_validation_times.append(time.perf_counter() - validation_started)
            record = {"step": float(step), **metrics}
            history.append(record)
            last_validation = metrics
            score = (metrics["scene_macro_psnr"], metrics["scene_macro_ssim"], -step)
            if best is None or score > best:
                best = score
                _save_checkpoint(config, trainer, best_path, step, metrics)
            if step in full_validation_steps:
                full_started = time.perf_counter()
                full_metrics = _validate(
                    trainer, full_validation_loader, device, config.seed
                )
                full_validation_times.append(time.perf_counter() - full_started)
                full_history.append({"step": float(step), **full_metrics})
                last_validation = full_metrics

    last_path = config.path("paths", "outputs") / "checkpoints" / f"{group}_{steps}_last.pt"
    _save_checkpoint(config, trainer, last_path, steps, last_validation)
    history_path = ensure_output_path(
        config, config.path("paths", "outputs") / "metrics" / f"{group}_{steps}_validation.json"
    )
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    if full_history:
        full_path = ensure_output_path(
            config,
            config.path("paths", "outputs")
            / "metrics"
            / f"{group}_{steps}_full_validation.json",
        )
        full_path.write_text(json.dumps(full_history, indent=2), encoding="utf-8")
    _write_diagnostic_summaries(config, group, steps, diagnostic_records)
    timing_values = step_times[20:] if len(step_times) > 20 else step_times
    mean_step_seconds = float(np.mean(timing_values))
    mean_proxy_validation_seconds = float(np.mean(proxy_validation_times))
    mean_full_validation_seconds = (
        float(np.mean(full_validation_times)) if full_validation_times else None
    )
    executed_steps = steps - start_step
    proxy_validation_runs = len(proxy_validation_times)
    full_validation_runs = len(full_validation_times)
    timing = {
        "group": group,
        "start_step": start_step,
        "end_step": steps,
        "steps_executed_this_run": executed_steps,
        "total_wall_seconds": time.perf_counter() - run_started,
        "mean_step_seconds_excluding_first_20": mean_step_seconds,
        "mean_proxy_validation_seconds": mean_proxy_validation_seconds,
        "mean_full_validation_seconds": mean_full_validation_seconds,
        "estimated_three_group_seconds": float(
            3
            * (
                mean_step_seconds * executed_steps
                + mean_proxy_validation_seconds * proxy_validation_runs
                + (mean_full_validation_seconds or 0.0) * full_validation_runs
            )
        ),
    }
    timing_path = ensure_output_path(
        config, config.path("paths", "outputs") / "diagnostics" / f"{group}_{steps}_wall_clock.json"
    )
    timing_path.write_text(json.dumps(timing, indent=2), encoding="utf-8")
    return run_root


def load_checkpoint_for_evaluation(
    config: ExperimentConfig, checkpoint: Path, device: torch.device
) -> ExperimentTrainer:
    trainer = build_components(config, device, require_locked=True)
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    trainer.r2p.load_state_dict(state["r2p"], strict=True)
    return trainer


def run_proxy_checkpoint_validation(
    config: ExperimentConfig,
    group: str,
    checkpoint: Path,
    device: torch.device,
) -> dict[str, object]:
    """Evaluate one existing checkpoint on the frozen validation proxy."""
    rows = load_manifest(config.path("paths", "manifest"))
    validate_manifest(rows, config, check_expected_counts=True)
    proxy_rows = load_validation_proxy_rows(config, rows)
    trainer = load_checkpoint_for_evaluation(config, checkpoint, device)
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    step = int(state["step"])
    started = time.perf_counter()
    metrics = _validate(
        trainer, _evaluation_loader(proxy_rows, group), device, config.seed
    )
    elapsed = time.perf_counter() - started
    record = {"step": float(step), **metrics}
    metrics_path = ensure_output_path(
        config,
        config.path("paths", "outputs")
        / "metrics"
        / f"{group}_{step}_validation.json",
    )
    if metrics_path.exists():
        raise FileExistsError(f"Refusing to overwrite proxy validation: {metrics_path}")
    metrics_path.write_text(json.dumps([record], indent=2), encoding="utf-8")
    result = {
        "scope": "proxy",
        "sample_count": len(proxy_rows),
        "checkpoint": str(checkpoint.resolve()),
        "step": step,
        "elapsed_seconds": elapsed,
        "metrics": metrics,
    }
    timing_path = ensure_output_path(
        config,
        config.path("paths", "outputs")
        / "diagnostics"
        / f"{group}_{step}_proxy_validation_wall_clock.json",
    )
    timing_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def recover_interrupted_proxy_run(
    config: ExperimentConfig,
    group: str,
    final_step: int,
    device: torch.device,
) -> dict[str, object]:
    """Reconstruct a proxy-only milestone history after an interrupted run.

    This recovery path is deliberately narrow: it evaluates already-saved
    milestones, never retrains, and makes copies (rather than moving) of the
    final and best continuation checkpoints needed by a later 5000-step run.
    """
    if final_step != int(config.raw["optimization"]["pilot_steps"]):
        raise ValueError("Recovery is supported only for the 3000-step milestone")
    metrics_root = config.path("paths", "outputs") / "metrics"
    checkpoints_root = config.path("paths", "outputs") / "checkpoints"
    history_path = metrics_root / f"{group}_{final_step}_validation.json"
    if history_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed history: {history_path}")
    rows = load_manifest(config.path("paths", "manifest"))
    validate_manifest(rows, config, check_expected_counts=True)
    proxy_rows = load_validation_proxy_rows(config, rows)
    proxy_loader = _evaluation_loader(proxy_rows, group)
    milestone_steps = sorted(
        {
            int(config.raw["optimization"]["sanity_steps"]),
            *(
                int(step)
                for step in config.raw["optimization"]["validation_steps"]
                if int(step) <= final_step
            ),
            final_step,
        }
    )
    history: list[dict[str, float]] = []
    timings: list[dict[str, object]] = []
    for step in milestone_steps:
        per_step_path = metrics_root / f"{group}_{step}_validation.json"
        if per_step_path.exists():
            stored = json.loads(per_step_path.read_text(encoding="utf-8"))
            if len(stored) != 1 or int(stored[0]["step"]) != step:
                raise ValueError(f"Malformed recovery metric: {per_step_path}")
            history.append(stored[0])
            continue
        checkpoint = checkpoints_root / f"{group}_{step}_milestone.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing recovery milestone: {checkpoint}")
        trainer = load_checkpoint_for_evaluation(config, checkpoint, device)
        started = time.perf_counter()
        metrics = _validate(trainer, proxy_loader, device, config.seed)
        elapsed = time.perf_counter() - started
        record = {"step": float(step), **metrics}
        ensure_output_path(config, per_step_path)
        per_step_path.write_text(json.dumps([record], indent=2), encoding="utf-8")
        history.append(record)
        timings.append({"step": step, "elapsed_seconds": elapsed})
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    best_record = max(
        history,
        key=lambda record: (
            float(record["scene_macro_psnr"]),
            float(record["scene_macro_ssim"]),
            -int(record["step"]),
        ),
    )
    final_milestone = checkpoints_root / f"{group}_{final_step}_milestone.pt"
    if not final_milestone.is_file():
        raise FileNotFoundError(f"Missing final recovery milestone: {final_milestone}")
    best_milestone = checkpoints_root / f"{group}_{int(best_record['step'])}_milestone.pt"
    if not best_milestone.is_file():
        # The 200-step run exposes a last checkpoint instead of a milestone.
        best_milestone = checkpoints_root / f"{group}_{int(best_record['step'])}_last.pt"
    if not best_milestone.is_file():
        raise FileNotFoundError(f"Missing best recovery checkpoint: {best_milestone}")
    last_path = checkpoints_root / f"{group}_{final_step}_last.pt"
    best_path = checkpoints_root / f"{group}_{final_step}_best.pt"
    recovery_root = ensure_output_path(
        config, config.path("paths", "outputs") / "diagnostics" / "recovery"
    )
    recovery_root.mkdir(parents=True, exist_ok=True)
    if best_path.exists():
        shutil.copy2(best_path, recovery_root / f"{group}_{final_step}_best_before_recovery.pt")
    shutil.copy2(final_milestone, last_path)
    shutil.copy2(best_milestone, best_path)
    report = {
        "group": group,
        "final_step": final_step,
        "scope": "proxy_only",
        "sample_count": len(proxy_rows),
        "history_steps": [int(record["step"]) for record in history],
        "revalidated_steps": [int(item["step"]) for item in timings],
        "best_step": int(best_record["step"]),
        "last_checkpoint": str(last_path),
        "best_checkpoint": str(best_path),
    }
    (recovery_root / f"{group}_{final_step}_recovery.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def run_group_evaluation(
    config: ExperimentConfig,
    group: str,
    checkpoint: Path,
    device: torch.device,
    *,
    split: str,
    final_selection_confirmed: bool,
    intervention: str | None = None,
    main_result_approved: bool = False,
) -> dict[str, object]:
    if split == "test" and not final_selection_confirmed:
        raise RuntimeError("Test evaluation is blocked until final model selection is confirmed")
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if intervention is not None and not main_result_approved:
        raise RuntimeError("Condition intervention is blocked until the main result is approved")
    rows = select_rows(load_manifest(config.path("paths", "manifest")), split)
    if intervention is not None:
        dataset = InterventionDataset(rows, group)
    else:
        dataset = PairedBranchDataset(rows, group=group, crop_size=None)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    trainer = load_checkpoint_for_evaluation(config, checkpoint, device)
    tag = intervention or "standard"
    output_root = config.path("paths", "outputs") / "predictions" / split / group / tag
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite predictions: {output_root}")
    metric_rows = evaluate_loader(
        trainer.r2p,
        trainer.task,
        loader,
        device,
        output_root=output_root,
        intervention=intervention,
        base_seed=config.seed,
    )
    metric_name = f"{group}_frames.csv" if split == "test" and tag == "standard" else f"{group}_{split}_{tag}.csv"
    metric_path = config.path("paths", "outputs") / "metrics" / metric_name
    write_metric_csv(config, metric_rows, metric_path)
    summary = aggregate_metrics(metric_rows)
    summary_name = f"{group}.json" if split == "test" and tag == "standard" else f"{group}_{split}_{tag}.json"
    summary_path = ensure_output_path(
        config, config.path("paths", "outputs") / "metrics" / summary_name
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_official_baseline(
    config: ExperimentConfig,
    device: torch.device,
    *,
    split: str,
    final_selection_confirmed: bool,
) -> dict[str, object]:
    require_preprocessing_parity(config)
    if split == "test" and not final_selection_confirmed:
        raise RuntimeError("Test evaluation is blocked until final model selection is confirmed")
    rows = load_manifest(config.path("paths", "manifest"))
    validate_manifest(rows, config, check_expected_counts=True)
    loader = _evaluation_loader(select_rows(rows, split), "A1")
    output_root = config.path("paths", "outputs") / "predictions" / split / "Baseline"
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite baseline: {output_root}")
    task = OfficialPolarFreeTask(config, device)
    metric_rows = evaluate_official_task_loader(
        task, loader, device, output_root, base_seed=config.seed
    )
    metric_name = "baseline_frames.csv" if split == "test" else f"baseline_{split}.csv"
    metric_path = config.path("paths", "outputs") / "metrics" / metric_name
    write_metric_csv(config, metric_rows, metric_path)
    summary = aggregate_metrics(metric_rows)
    summary_name = "baseline.json" if split == "test" else f"baseline_{split}.json"
    summary_path = ensure_output_path(
        config, config.path("paths", "outputs") / "metrics" / summary_name
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def assess_extension(config: ExperimentConfig) -> dict[str, object]:
    """Evaluate the proxy-only 3000->5000 decision for this screening run."""
    groups: dict[str, object] = {}
    extend = False
    for group in config.groups:
        proxy_history = json.loads(
            (config.path("paths", "outputs") / "metrics" / f"{group}_3000_validation.json").read_text(
                encoding="utf-8"
            )
        )
        rising = len(proxy_history) >= 3 and all(
            float(proxy_history[index]["scene_macro_psnr"])
            > float(proxy_history[index - 1]["scene_macro_psnr"])
            for index in (-2, -1)
        )
        groups[group] = {
            "validation_scope": "fixed_28_frame_proxy",
            "materially_above_baseline": None,
            "two_rising_intervals": rising,
        }
        extend = extend or rising
    result = {
        "extend_all_groups_to_5000": extend,
        "groups": groups,
        "decision_scope": "proxy_only; no full-validation or baseline comparison was run",
    }
    destination = ensure_output_path(
        config, config.path("paths", "outputs") / "metrics" / "extension_decision.json"
    )
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def assess_runtime(config: ExperimentConfig) -> dict[str, object]:
    diagnostics_root = config.path("paths", "outputs") / "diagnostics"
    reports = []
    for group in config.groups:
        standard = diagnostics_root / f"{group}_200_wall_clock.json"
        archived_full = diagnostics_root / f"{group}_200_full_wall_clock.json"
        source = standard if standard.is_file() else archived_full
        reports.append(json.loads(source.read_text(encoding="utf-8")))
    mean_step = float(
        np.mean([report["mean_step_seconds_excluding_first_20"] for report in reports])
    )
    proxy_seconds = [
        float(report["mean_proxy_validation_seconds"])
        for report in reports
        if report.get("mean_proxy_validation_seconds") is not None
    ]
    for group in config.groups:
        standalone = diagnostics_root / f"{group}_200_proxy_validation_wall_clock.json"
        if standalone.is_file():
            proxy_seconds.append(
                float(json.loads(standalone.read_text(encoding="utf-8"))["elapsed_seconds"])
            )
    full_seconds = [
        float(report["mean_full_validation_seconds"])
        for report in reports
        if report.get("mean_full_validation_seconds") is not None
    ]
    if not proxy_seconds:
        raise RuntimeError("Runtime assessment requires proxy validation timing")
    mean_proxy_validation = float(np.mean(proxy_seconds))
    mean_full_validation = float(np.mean(full_seconds)) if full_seconds else 0.0
    validation_steps = [
        int(value) for value in config.raw["optimization"]["validation_steps"]
    ]

    def estimate(start_step: int, end_step: int) -> float:
        proxy_runs = sum(start_step < step <= end_step for step in validation_steps)
        full_runs = sum(
            start_step < int(step) <= end_step
            for step in config.raw["validation"]["full_steps"]
        )
        return 3 * (
            mean_step * (end_step - start_step)
            + mean_proxy_validation * proxy_runs
            + mean_full_validation * full_runs
        )

    estimate_3000 = estimate(0, 3000)
    estimate_extension = estimate(3000, 5000)
    estimate_5000_total = estimate_3000 + estimate_extension
    result = {
        "mean_step_seconds": mean_step,
        "mean_proxy_validation_seconds": mean_proxy_validation,
        "mean_full_validation_seconds": mean_full_validation,
        "estimated_three_group_seconds_3000": estimate_3000,
        "estimated_three_group_extension_seconds_3000_to_5000": estimate_extension,
        "estimated_three_group_total_seconds_5000": estimate_5000_total,
        "within_five_day_total_budget_5000": estimate_5000_total <= 5 * 24 * 60 * 60,
        "budget_policy": (
            "The extension estimate includes only steps 3001-5000; cap at 3000 when the "
            "combined 0-5000 estimate exceeds five days."
        ),
    }
    destination = ensure_output_path(
        config, config.path("paths", "outputs") / "metrics" / "runtime_budget.json"
    )
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_comparison(config: ExperimentConfig, split: str) -> dict[str, object]:
    metrics_root = config.path("paths", "outputs") / "metrics"
    baseline_name = "baseline.json" if split == "test" else f"baseline_{split}.json"
    summaries = {
        "Baseline": json.loads((metrics_root / baseline_name).read_text(encoding="utf-8"))
    }
    for group in config.groups:
        group_name = f"{group}.json" if split == "test" else f"{group}_{split}_standard.json"
        summaries[group] = json.loads(
            (metrics_root / group_name).read_text(encoding="utf-8")
        )
    keys = ("scene_macro_psnr", "scene_macro_ssim")

    def difference(left: str, right: str) -> dict[str, float]:
        return {key: float(summaries[left][key]) - float(summaries[right][key]) for key in keys}

    result = {
        "split": split,
        "preprocessing": json.loads(
            config.path("paths", "preprocessing_parity").read_text(encoding="utf-8")
        ),
        "metrics": {name: {key: value[key] for key in keys} for name, value in summaries.items()},
        "differences": {
            "A0_minus_Baseline": difference("A0", "Baseline"),
            "A1_minus_A0": difference("A1", "A0"),
            "A2_minus_A1": difference("A2", "A1"),
        },
        "interpretation_limit": (
            "Single-seed task-utility screening only; no claim of statistical significance or "
            "physical polarization accuracy."
        ),
    }
    comparison_name = "comparison.json" if split == "test" else f"comparison_{split}.json"
    destination = ensure_output_path(config, metrics_root / comparison_name)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_integration_smoke(config: ExperimentConfig, device: torch.device) -> dict[str, object]:
    """Real-data two-step gate, supplied for later execution but not run during implementation."""
    rows = load_manifest(config.path("paths", "manifest"))
    validate_manifest(rows, config, check_expected_counts=True)
    set_seed(config.seed)
    trainer = build_components(config, device, require_locked=True)
    load_or_create_r2p_initial_state(trainer.r2p, config)
    loader = _scheduled_loader(select_rows(rows, "train"), config, "A0", 2)
    iterator = iter(loader)
    first_values = next(iterator)
    first = Batch.from_mapping(first_values, device)
    trainer.r2p.eval()
    with torch.no_grad():
        identity = trainer.r2p(first.polar, first.condition)
        if not torch.equal(identity.refined_polar, first.polar):
            raise AssertionError("Zero-initialized R2P must preserve P exactly")
        inputs = refined_polar_to_task_inputs(identity.refined_polar, first.official_rgb)
        set_step_seed(config.seed, "identity", 0, first.sample_id)
        wrapped_prediction, wrapped_features = trainer.task(
            identity.refined_polar, first.official_rgb, None
        )
        set_step_seed(config.seed, "identity", 0, first.sample_id)
        direct_prediction, direct_features, _, _ = trainer.task.model(
            inputs, first.target, phase="val"
        )
        identity_error = float((wrapped_prediction - direct_prediction).abs().max())
        if identity_error > 1e-6:
            raise AssertionError(f"Frozen wrapper differs from official fresh path: {identity_error}")
        expected_shapes = {
            f"T{level}": list(direct_features[f"TaF{level}"].shape) for level in (1, 2, 3)
        }
        if {key: list(value.shape) for key, value in wrapped_features.items()} != expected_shapes:
            raise AssertionError("Wrapped task feature shapes differ from official features")
    task_hash_before = state_dict_sha256(trainer.task.state_dict())
    diagnostics = []
    for step, values in enumerate((first_values, next(iterator)), start=1):
        batch = Batch.from_mapping(values, device)
        set_step_seed(config.seed, "integration", step, batch.sample_id)
        diagnostics.append(trainer.train_step(batch))
    trainer.task.assert_fully_frozen()
    task_hash_after = state_dict_sha256(trainer.task.state_dict())
    if task_hash_before != task_hash_after:
        raise AssertionError("Frozen PolarFree state changed during integration smoke")
    required = (
        "r2p_grad_norm",
        "polar_encoder_grad_norm",
        "condition_encoder_grad_norm",
        "residual_head_grad_norm",
        "projection_head_grad_norm",
    )
    if any(diagnostics[-1][key] <= 0 for key in required):
        raise AssertionError(f"R2P gradient gate failed: {diagnostics[-1]}")
    result = {
        "status": "passed",
        "identity_wrapper_max_abs_error": identity_error,
        "task_feature_shapes": expected_shapes,
        "polarfree_state_unchanged": True,
        "steps": diagnostics,
    }
    destination = ensure_output_path(
        config, config.path("paths", "outputs") / "diagnostics" / "integration_smoke.json"
    )
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
