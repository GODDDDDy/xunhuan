"""Create the two locked configs and evidence files before observing new results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform

import torch
import yaml

ORIGINAL = Path("/media/li/data3/25thstudent/DY/929work/pidsr_k2_929")
ROOT = ORIGINAL / "phase_objective_20261002"
source = yaml.safe_load((ORIGINAL / "experiment/config_no_feedback.yaml").read_text())
lambda_value = json.loads((ORIGINAL / "runs/no_feedback/seed_2026/config/alignment_calibration.json").read_text())["lambda_feature_alignment"]
configs = {}
for name, weight in (("phase1", 1.0), ("phase01", 0.1)):
    config = json.loads(json.dumps(source))
    config["paths"]["output_root"] = str(ROOT / "runs" / name / "seed_2026")
    config["study"] = {"phase_weight": weight, "fixed_lambda": lambda_value,
                       "fixed_budget": 3000, "selection": "fixed_step3000",
                       "scope": "single_seed_exploratory_validation_only"}
    destination = ROOT / "configs" / f"{name}.yaml"
    destination.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    configs[name] = config
first = json.loads(json.dumps(configs["phase1"]))
second = json.loads(json.dumps(configs["phase01"]))
first["paths"].pop("output_root")
second["paths"].pop("output_root")
first["study"].pop("phase_weight")
second["study"].pop("phase_weight")
assert first == second, "Matched configs differ beyond phase weight/output path"
manifest = {"phase_weights": [1.0, 0.1], "fixed_lambda": lambda_value,
            "steps": 3000, "seed": 2026, "feedback": False,
            "same_config_except_phase_and_output": True,
            "optimizer_start": "new Adam from original step0 initial weights",
            "evaluation": "fixed step3000, full872 validation; no new test selection",
            "historical_test_usage": "already viewed; not an independent confirmation set",
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                            "gpu": torch.cuda.get_device_name(0)}}
paths = {"archived_trainer": ROOT / "scripts/reference/run_experiment_snapshot.py",
         "extension": ROOT / "scripts/run_phase_objective.py",
         "original_config": ORIGINAL / "experiment/config_no_feedback.yaml"}
paths.update({k: Path(source["paths"][k]) for k in
              ("manifest", "schedule", "pretrained_r2p", "reference_r2p", "base_config")})
manifest["file_sha256"] = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}
(ROOT / "protocol_lock.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
