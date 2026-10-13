import importlib.util, sys, json
from pathlib import Path
import torch, yaml, numpy as np
R = Path("/media/li/data3/25thstudent/DY/929work/pidsr_k2_929")
SRC = R / "deps/rgb_final_confirmation/src"
SEED = 2026
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m
def ry(p):
    return yaml.safe_load(open(p, encoding="utf-8"))
device = torch.device("cuda:0")
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
mod = load_module("re_mod", R / "experiment/run_experiment.py")
cfg = ry(R / "experiment/config.yaml")
mod.add_source_path(cfg); mod.set_seed(SEED)
model, task, crit, base = mod.build_components(cfg, device); del crit
ck = Path(cfg["paths"]["output_root"]) / "checkpoints/step_3000.pt"
st = torch.load(ck, map_location=device, weights_only=False)
model.load_state_dict(st["r2p_and_projections"], strict=True)
task.net_g.load_state_dict(st["tasknet_net_g"], strict=True); del st
loader = mod.split_loader(cfg, "validation")
model.eval(); task.eval()
raw = next(iter(loader))
batch = mod.unpack_batch(raw, device)
mod.set_step_seed(SEED, "validation", 0, batch["sample_id"])
with torch.no_grad():
    p = batch["polar"]
    c = batch["condition"]
    # 1) two identical consecutive calls, same RNG state
    torch.manual_seed(0); a = model(p, c, feedback=None, project_features=False).refined_polar.clone()
    torch.manual_seed(0); b = model(p, c, feedback=None, project_features=False).refined_polar.clone()
    print("TEST1 two identical calls (same seeded rng): max|a-b| = %.3e" % float((a-b).abs().max()))
    # 2) without reseeding
    a2 = model(p, c, feedback=None, project_features=False).refined_polar.clone()
    b2 = model(p, c, feedback=None, project_features=False).refined_polar.clone()
    print("TEST2 two identical calls (no reseed):      max|a2-b2| = %.3e" % float((a2-b2).abs().max()))
    # 3) is R2P itself stochastic? compare raw r2p path
    r1, _ = model.r2p(c, p, return_features=True)
    r2, _ = model.r2p(c, p, return_features=True)
    print("TEST3 raw r2p twice (no reseed):           max|r1-r2| = %.3e" % float((r1-r2).abs().max()))
    # 4) feeding P1 back vs original polar
    print("TEST4 max|refined - polar|        = %.3e" % float((a - p).abs().max()))
    print("TEST4b clamp saturation: frac at 0 = %.4f, frac at 1 = %.4f"
          % (float((a<=0).float().mean()), float((a>=1).float().mean())))
    # 5) delta magnitude
    out = model(p, c, feedback=None, project_features=False)
    print("TEST5 max|delta| = %.4e  mean|delta| = %.4e" % (float(out.delta.abs().max()), float(out.delta.abs().mean())))
    print("TEST5b max_residual attr =", getattr(model.r2p, "max_residual", "N/A"))
