"""Single-round preparation: same-scale polar enhancer -> fixed PolarFree.

No quality, repair-need, cross-round state, RGB correction or acceptance head is
constructed. New zero-output adapters act inside each frozen backbone; they
affect the features used by FANet and the actual output, not only the loss head.
"""
from __future__ import annotations

import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

SOURCE_SHA = "279d0a329d0efebee9678c2156bf3e7b60c0646e75a5f37f8015c7b100f9e9a2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FeatureResidualAdapter(nn.Module):
    def __init__(self, channels, hidden=16):
        super().__init__()
        self.down = nn.Conv2d(channels, hidden, 1)
        self.up = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, feature):
        return feature + .1 * torch.tanh(self.up(F.gelu(self.down(feature))))


@contextmanager
def residual_hooks(points, adapters, enabled=True):
    handles = []
    try:
        if enabled:
            for key, module in points.items():
                # Bind the key, but resolve the adapter at forward time so that
                # functional_call's temporary parameters are actually consumed.
                def hook(_module, _args, output, key=key):
                    return adapters[key](output)
                handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


class SingleEnhancer(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core.requires_grad_(False).eval()
        self.core.activation_checkpoint = False
        self.adapters = nn.ModuleDict({f"level{i}": FeatureResidualAdapter(c)
                                       for i, c in enumerate((48, 96, 192), 1)})

    def forward(self, polar, image, use_adapters=True):
        points = {"level1": self.core.mid_refine[-1],
                  "level2": self.core.de_decoder2[-1],
                  "level3": self.core.bottleneck[-1]}
        with residual_hooks(points, self.adapters, use_adapters):
            corrected, features, _, _ = self.core(polar, polar, image, None, None, None)
        return corrected, features


class SingleTask(nn.Module):
    def __init__(self, task):
        super().__init__()
        self.task = task.requires_grad_(False).eval()
        self.adapters = nn.ModuleDict({f"level{i}": FeatureResidualAdapter(c)
                                       for i, c in enumerate((96, 96, 192), 1)})

    def forward(self, polar, image, observation, noise=None, use_adapters=True):
        from bridge import anchored_task_inputs
        points = {f"level{i}": getattr(self.task.net_g, f"decoder_level{i}")
                  for i in (1, 2, 3)}
        inputs = anchored_task_inputs(polar.float(), observation.float(), image.float())
        with residual_hooks(points, self.adapters, use_adapters):
            return self.task.forward_inputs(inputs, noise)


def trainable_names(module):
    return tuple(name for name, parameter in module.named_parameters() if parameter.requires_grad)


def task_objective(prediction, target):
    """Draft single-round task objective shared by the two prospective arms."""
    mse = (prediction - target).square().mean()
    edge = sum((prediction.diff(dim=dim) - target.diff(dim=dim)).abs().mean()
               for dim in (-1, -2)) / 2
    return mse + .02 * edge


def joint_objective(polar, prediction, target, source_polar, retention_weight=.1):
    # This is a frozen-source retention proxy, NOT clean-polarization ground
    # truth, official demosaicking supervision, EIT, or a physical guarantee.
    return task_objective(prediction, target) + retention_weight * (
        polar - source_polar.detach()).square().mean()


def build_system(root):
    root = Path(root).resolve()
    for folder in (root / "source_snapshot/scripts", root / "source_snapshot/source"):
        sys.path.insert(0, str(folder))
    from enhancer import PolarEnhancer
    from rgb_final_confirmation.config import ExperimentConfig
    from rgb_final_confirmation.polarfree import OfficialPolarFreeTask
    from modules.fanet import SameScaleFANet
    import yaml

    runtime = json.loads((root / "config/prepare.json").read_text(encoding="utf-8"))
    checkpoint_path = Path(runtime["source_checkpoint"])
    if sha256(checkpoint_path) != SOURCE_SHA:
        raise ValueError("Latest source checkpoint hash differs; no fallback is allowed")
    # Trusted project artifact. Its optimizer/RNG are never restored.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["r2p_and_projections"]
    core_state = {name.removeprefix("r2p."): value for name, value in state.items()
                  if name.startswith("r2p.")}
    core = PolarEnhancer()
    core.load_state_dict(core_state, strict=True)
    assert all(torch.equal(value, core_state[name]) for name, value in core.state_dict().items())

    raw = yaml.safe_load((root / "config/source_base_server.yaml").read_text(encoding="utf-8"))
    raw["project_root"] = str(root / "source_snapshot")
    raw["experiment_root"] = str(root / "evidence/source_load")
    raw["paths"]["weight_audit"] = str(root / "evidence/source_load/official_weight_audit.json")
    raw["official"]["polarapp_dfp_root"] = runtime["polarfree_dependency_root"]
    raw["official"]["dm_config"] = runtime["polarfree_dm_config"]
    raw["official"]["checkpoint_dir"] = runtime["official_checkpoint_dir"]
    audit_dir = root / "evidence/source_load"
    audit_dir.mkdir(parents=True, exist_ok=True)
    # Historical config validation is not a new-run validator. Its raw data and
    # original official loader are used with explicit, isolated output paths.
    task = OfficialPolarFreeTask(ExperimentConfig(raw, root / "config/source_base_server.yaml"),
                                 torch.device("cpu"))
    task.net_g.load_state_dict(checkpoint["tasknet_net_g"], strict=True)
    assert all(torch.equal(value, checkpoint["tasknet_net_g"][name])
               for name, value in task.net_g.state_dict().items())
    torch.manual_seed(runtime["seed"])
    upstream, downstream = SingleEnhancer(core), SingleTask(task)
    alignment = SameScaleFANet(root / "vendor/PolarAPP/DfP/utils/utils_net.py",
                               root / "weights/FANet.pth", state["projections.P1.weight"])
    source_receipt = {
        "source_run": "polar_gain_judgement_short_20261012",
        "source_sha256": SOURCE_SHA,
        "source_protocol_id": checkpoint["protocol_id"],
        "core_tensors_strict_equal": len(core_state),
        "generator_tensors_strict_equal": len(checkpoint["tasknet_net_g"]),
        "optimizer_reused": False, "rng_reused": False,
        "official_polarfree_is_ancestry_not_actual_net_g_start": True,
        "inactive_source_modules_not_constructed": ["H", "Q", "cross_round_feedback", "RGB_correction_head",
                                                    "local_acceptance", "action_cost_gate"],
        "core_legacy_observation_adapters": "retained frozen; previous-task inputs absent",
    }
    return upstream, downstream, alignment, source_receipt, runtime
