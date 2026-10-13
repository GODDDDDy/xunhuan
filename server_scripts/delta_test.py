import importlib.util, sys
from pathlib import Path
import torch, yaml
R = Path("/media/li/data3/25thstudent/DY/929work/pidsr_k2_929")
SRC = R / "deps/rgb_final_confirmation/src"
SEED = 2026
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m
def ry(p): return yaml.safe_load(open(p, encoding="utf-8"))
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
raw = next(iter(loader)); batch = mod.unpack_batch(raw, device)
mod.set_step_seed(SEED, "validation", 0, batch["sample_id"])
with torch.no_grad():
    p, c = batch["polar"], batch["condition"]
    f1out = model(p, c, feedback=None, project_features=False)
    p1 = f1out.refined_polar.clone(); d1 = f1out.delta.clone()
    # round 2, original feedback config
    s2 = model(p1, c, feedback=None, project_features=False)  # feedback=None for diagnosis
    p2 = s2.refined_polar.clone(); d2 = s2.delta.clone()
    print("max|P1 - (p + d1)| (should be ~0 ignoring clamp) = %.3e" % float((p1-(p+d1).clamp(0,1)).abs().max()))
    print("max|P2 - (P1 + d2)| (should be ~0 ignoring clamp) = %.3e" % float((p2-(p1+d2).clamp(0,1)).abs().max()))
    print("mean|d1| = %.4e   mean|d2| = %.4e   ratio=%.3f" % (float(d1.abs().mean()), float(d2.abs().mean()), float(d2.abs().mean()/d1.abs().mean())))
    print("max|d1|  = %.4e   max|d2|  = %.4e" % (float(d1.abs().max()), float(d2.abs().max())))
    # correlation of d1 and d2 sign/magnitude
    print("cos(d1,d2) = %.4f" % float(torch.nn.functional.cosine_similarity(d1.flatten(), d2.flatten(), dim=0)))
    # both rounds' delta nearly identical? 
    print("max|d2-d1| = %.4e   mean|d2-d1| = %.4e" % (float((d2-d1).abs().max()), float((d2-d1).abs().mean())))
