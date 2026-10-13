"""Verify paired initial states and capture a hash of the entire initial mapping."""
import hashlib
import json
from pathlib import Path
import torch
import yaml
import run_phase_objective as study

ROOT = Path(__file__).resolve().parent.parent
records = {}
for name in ("phase1", "phase01"):
    cfg = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
    study.base.add_source_path(cfg)
    study.base.set_seed(cfg["seed"])
    model, task, criterion, base = study.build(cfg, torch.device("cuda:0"))
    digest = hashlib.sha256()
    for prefix, state in (("r2p", model.state_dict()), ("net_g", task.net_g.state_dict())):
        for key, value in sorted(state.items()):
            digest.update((prefix + "." + key).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    records[name] = {"state_sha256": digest.hexdigest(), "phase_weight": cfg["study"]["phase_weight"],
                     "initialization": "original pretrained R2P e42 and official net_g; no resume"}
    del model, task, criterion
    torch.cuda.empty_cache()
assert records["phase1"]["state_sha256"] == records["phase01"]["state_sha256"]
payload = {"paired_initial_weights_identical": True, "arms": records}
(ROOT / "diagnostics/initialization_check.json").write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
