"""Reuse the archived G6 trainer; change only the Fourier-phase coefficient.

Frozen audits exclude net_g by tensor identity, since it is nested in the prior.
No source trainer, dependency, or old run is edited.
"""
import importlib.util
from pathlib import Path
import sys
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("g6_original", HERE / "reference/k1_ablation_snapshot.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
original_build = base.build_components
original_checkpoint = base.checkpoint_payload
original_audit = base.frozen_audit


class WeightedPhase(nn.Module):
    def __init__(self, phase, coefficient):
        super().__init__()
        self.phase = phase
        self.coefficient = float(coefficient)

    def forward(self, prediction, target):
        return self.coefficient * self.phase(prediction, target)


def frozen_audit(model, task):
    audit = original_audit(model, task)
    netg_ids = {id(p) for p in task.net_g.parameters()} | {id(b) for b in task.net_g.buffers()}
    audit["prior"] = {"param:" + n: base.tensor_sha256(p)
                      for n, p in task.model.named_parameters() if id(p) not in netg_ids}
    audit["prior"].update({"buffer:" + n: base.tensor_sha256(b)
                            for n, b in task.model.named_buffers() if id(b) not in netg_ids})
    return audit


def build(config, device):
    assert config["ablation"]["bypass_r2p"] and config["ablation"]["train_netg"]
    assert not config["ablation"]["train_r2p"] and not config["ablation"]["use_alignment"]
    if "study" not in config:
        # Only the archived G6 is admitted for the step0 forward consistency check.
        assert config["ablation"]["group"] == "G6_netg_only_bypass_r2p"
        return original_build(config, device)
    assert float(config["study"]["phase_weight"]) == 0.1
    model, task, criterion, cfg = original_build(config, device)
    criterion.phase = WeightedPhase(criterion.phase, 0.1)
    model._study_root = Path(config["paths"]["output_root"])
    model._study_initial = frozen_audit(model, task)
    # Every construction must start from the very same official net_g as K2.
    reference = Path(config["study"]["k2_initial_netg"])
    initial = torch.load(reference, map_location="cpu", weights_only=True)
    actual = {n: p.detach().cpu().float() for n, p in task.net_g.named_parameters()}
    assert set(actual) == set(initial)
    changed = [n for n in actual if not torch.equal(actual[n], initial[n])]
    result = {"reference": str(reference), "reference_sha256": base.sha256_file(reference),
              "parameter_tensors": len(actual), "changed_keys": changed, "passed": not changed,
              "source_hashes": {n: base.sha256_file(Path(config["paths"][n]))
                                for n in ("manifest", "schedule", "base_config", "pretrained_r2p")},
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "cudnn": torch.backends.cudnn.version(),
              "cudnn_tf32": torch.backends.cudnn.allow_tf32,
              "matmul_tf32": torch.backends.cuda.matmul.allow_tf32}
    base.atomic_json(model._study_root / "diagnostics/initialization_match.json", result)
    assert not changed, "Initial net_g differs from matched K2"
    return model, task, criterion, cfg


def checkpoint(model, task, optimizer, step, coefficient, metrics=None, group=""):
    audit = frozen_audit(model, task)
    changed = {block: [n for n, value in audit[block].items()
                       if value != model._study_initial[block][n]]
               for block in ("r2p", "projections", "prior")}
    result = {"step": int(step), "checked_tensors": {k: len(audit[k]) for k in changed},
              "changed_keys": changed, "passed": not any(changed.values())}
    base.atomic_json(model._study_root / f"diagnostics/frozen_step_{step}.json", result)
    assert result["passed"], "Frozen state changed"
    output = original_checkpoint(model, task, optimizer, step, coefficient, metrics, group)
    output["phase_weight"] = 0.1
    return output


base.build_components = build
base.frozen_audit = frozen_audit
base.checkpoint_payload = checkpoint

if __name__ == "__main__":
    base.main()
