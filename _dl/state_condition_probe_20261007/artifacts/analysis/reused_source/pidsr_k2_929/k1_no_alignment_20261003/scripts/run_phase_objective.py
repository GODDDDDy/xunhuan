"""Minimal extension of the archived K2 trainer: phase coefficient and fixed lambda.

All data, recurrence, initialization, optimizer, clipping and evaluation reuse the
original trainer. No old outputs or dependencies are modified.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "reference/run_experiment_snapshot.py"
spec = importlib.util.spec_from_file_location("phase_objective_original", SOURCE)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
original_build = base.build_components
original_protocol = base.protocol_id
original_checkpoint = base.checkpoint_payload


class WeightedPhase(nn.Module):
    def __init__(self, phase, coefficient):
        super().__init__()
        self.phase = phase
        self.coefficient = float(coefficient)

    def forward(self, prediction, target):
        return self.coefficient * self.phase(prediction, target)


def protocol(config):
    copied = json.loads(json.dumps(config))
    copied["phase_extension_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return original_protocol(copied)


def frozen_state(model, task):
    trainable_ids = {id(p) for p in task.net_g.parameters()}
    trainable_buffer_ids = {id(b) for b in task.net_g.buffers()}
    result = {"prior.param." + n: p.detach().cpu().clone()
              for n, p in task.model.named_parameters() if id(p) not in trainable_ids}
    result.update({"prior.buffer." + n: p.detach().cpu().clone()
                   for n, p in task.model.named_buffers() if id(p) not in trainable_buffer_ids})
    if not model.feedback_enabled:
        result.update({"unused_feedback." + n: p.detach().cpu().clone()
                       for n, p in model.feedback_adapter.state_dict().items()})
        result["unused_feedback.scale"] = model.feedback_scale.detach().cpu().clone()
    return result


def build(config, device):
    model, task, criterion, configuration = original_build(config, device)
    coefficient = float(config["study"]["phase_weight"])
    if coefficient not in (1.0, 0.1):
        raise ValueError("Only the preregistered phase coefficients 1 and 0.1 are allowed")
    if coefficient != 1.0:
        criterion.phase = WeightedPhase(criterion.phase, coefficient)
    model._study_phase_weight = coefficient
    model._study_output_root = Path(config["paths"]["output_root"])
    model._study_frozen_start = frozen_state(model, task)
    return model, task, criterion, configuration


def fixed_calibration(config, device):
    # Do not recompute a loss-dependent lambda: that would introduce a second factor.
    payload = {"schema_version": 1, "protocol_id": protocol(config),
               "lambda_feature_alignment": float(config["study"]["fixed_lambda"]),
               "mode": "fixed_from_original_no_feedback_seed2026",
               "recalibrated": False, "phase_weight": float(config["study"]["phase_weight"])}
    base.atomic_json(Path(config["paths"]["output_root"]) / "config/alignment_calibration.json", payload)
    return payload


def checkpoint(model, task, optimizer, step, coefficient, metrics=None):
    current = frozen_state(model, task)
    changed = [n for n, p in current.items() if not torch.equal(p, model._study_frozen_start[n])]
    audit = {"step": int(step), "phase_weight": model._study_phase_weight,
             "checked_tensors": len(current), "changed_keys": changed, "passed": not changed}
    base.atomic_json(model._study_output_root / "diagnostics" / f"frozen_step_{step}.json", audit)
    if changed:
        raise AssertionError(f"Frozen prior or unused feedback changed: {changed}")
    result = original_checkpoint(model, task, optimizer, step, coefficient, metrics)
    result["phase_weight"] = model._study_phase_weight
    return result


base.protocol_id = protocol
base.build_components = build
base.calibrate = fixed_calibration
base.checkpoint_payload = checkpoint

if __name__ == "__main__":
    base.main()
