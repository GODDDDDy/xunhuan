from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import load_config
from .data import load_manifest, manifest_audit, validate_manifest
from .polarfree import audit_official_weights
from .runner import (
    audit_preprocessing_parity,
    assess_extension,
    assess_runtime,
    prepare_experiment,
    prepare_validation_proxy,
    run_group_evaluation,
    run_integration_smoke,
    run_loss_calibration,
    run_official_baseline,
    run_proxy_checkpoint_validation,
    recover_interrupted_proxy_run,
    run_training,
    write_comparison,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "experiment.yaml"


def _device(name: str | None) -> torch.device:
    if name is None:
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen-PolarFree R2P four-group experiment")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Build manifest, split audit, and immutable schedule")
    commands.add_parser(
        "prepare-validation-proxy", help="Freeze the reviewed stratified validation proxy"
    )
    commands.add_parser("audit-weights", help="Hash the official PolarFree checkpoints")
    commands.add_parser("verify-manifest", help="Recheck counts and train/validation leakage")
    parity = commands.add_parser(
        "preprocessing-parity", help="Audit OpenCV/NumPy parity on real images"
    )
    parity.add_argument("--samples-per-split", type=int, default=4)

    calibrate = commands.add_parser("calibrate-loss", help="Create an unlocked loss-weight proposal")
    calibrate.add_argument("--device", default=None)
    smoke = commands.add_parser("integration-smoke", help="Run the two-step real-data gate")
    smoke.add_argument("--device", default=None)

    baseline = commands.add_parser("baseline")
    baseline.add_argument("--split", choices=("validation", "test"), required=True)
    baseline.add_argument("--final-selection-confirmed", action="store_true")
    baseline.add_argument("--device", default=None)

    train = commands.add_parser("train")
    train.add_argument("--group", choices=("A0", "A1", "A2"), required=True)
    train.add_argument("--steps", choices=(200, 3000, 5000), type=int, required=True)
    train.add_argument("--device", default=None)

    proxy_validation = commands.add_parser("validate-proxy")
    proxy_validation.add_argument("--group", choices=("A0", "A1", "A2"), required=True)
    proxy_validation.add_argument("--checkpoint", type=Path, required=True)
    proxy_validation.add_argument("--device", default=None)

    recover = commands.add_parser(
        "recover-proxy-run", help="Reconstruct an interrupted 3000-step proxy-only run"
    )
    recover.add_argument("--group", choices=("A0", "A1", "A2"), required=True)
    recover.add_argument("--steps", choices=(3000,), type=int, required=True)
    recover.add_argument("--device", default=None)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--group", choices=("A0", "A1", "A2"), required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), required=True)
    evaluate.add_argument("--final-selection-confirmed", action="store_true")
    evaluate.add_argument("--device", default=None)

    intervene = commands.add_parser("intervene")
    intervene.add_argument("--group", choices=("A0", "A1", "A2"), required=True)
    intervene.add_argument("--checkpoint", type=Path, required=True)
    intervene.add_argument("--split", choices=("validation", "test"), required=True)
    intervene.add_argument("--final-selection-confirmed", action="store_true")
    intervene.add_argument("--main-result-approved", action="store_true")
    intervene.add_argument(
        "--kind",
        choices=(
            "correct_condition",
            "polar_mean",
            "raw_rgb",
            "transmission_gt",
            "shuffled_within_scene",
            "shuffled_cross_scene",
            "zero_condition",
        ),
        required=True,
    )
    intervene.add_argument("--device", default=None)
    commands.add_parser("assess-extension")
    commands.add_parser("assess-runtime")
    compare = commands.add_parser("compare")
    compare.add_argument("--split", choices=("validation", "test"), required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "prepare":
        result = prepare_experiment(config)
    elif args.command == "prepare-validation-proxy":
        result = prepare_validation_proxy(config)
    elif args.command == "audit-weights":
        result = audit_official_weights(config)
    elif args.command == "verify-manifest":
        rows = load_manifest(config.path("paths", "manifest"))
        validate_manifest(rows, config, check_expected_counts=True)
        result = manifest_audit(rows)
    elif args.command == "preprocessing-parity":
        result = audit_preprocessing_parity(
            config, samples_per_split=args.samples_per_split
        )
    elif args.command == "calibrate-loss":
        result = run_loss_calibration(config, _device(args.device))
    elif args.command == "integration-smoke":
        result = run_integration_smoke(config, _device(args.device))
    elif args.command == "baseline":
        result = run_official_baseline(
            config,
            _device(args.device),
            split=args.split,
            final_selection_confirmed=args.final_selection_confirmed,
        )
    elif args.command == "train":
        result = str(run_training(config, args.group, args.steps, _device(args.device)))
    elif args.command == "validate-proxy":
        result = run_proxy_checkpoint_validation(
            config, args.group, args.checkpoint, _device(args.device)
        )
    elif args.command == "recover-proxy-run":
        result = recover_interrupted_proxy_run(
            config, args.group, args.steps, _device(args.device)
        )
    elif args.command in {"evaluate", "intervene"}:
        result = run_group_evaluation(
            config,
            args.group,
            args.checkpoint,
            _device(args.device),
            split=args.split,
            final_selection_confirmed=args.final_selection_confirmed,
            intervention=args.kind if args.command == "intervene" else None,
            main_result_approved=(
                args.main_result_approved if args.command == "intervene" else False
            ),
        )
    elif args.command == "assess-extension":
        result = assess_extension(config)
    elif args.command == "assess-runtime":
        result = assess_runtime(config)
    elif args.command == "compare":
        result = write_comparison(config, args.split)
    else:
        raise AssertionError(f"Unhandled command: {args.command}")
    print(json.dumps(result, indent=2) if not isinstance(result, str) else result)


if __name__ == "__main__":
    main()
