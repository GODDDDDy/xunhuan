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
def run(cfgname, tag):
    mod = load_module("re_"+tag, R / "experiment/run_experiment.py")
    cfg = ry(R / "experiment" / cfgname)
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
        o1 = model(p, c, feedback=None, project_features=False)
        p1, d1 = o1.refined_polar.clone(), o1.delta.clone()
        # task features from P1 (as recurrent_forward does)
        t1, f1 = task(p1, batch["official_rgb"], None)
        # REAL round 2 with feedback (group-config behavior)
        s2 = model(p1, c, feedback=f1 if model.feedback_enabled else None, project_features=False)
        d2 = s2.delta.clone()
        print("[%s] fs=%.6g feedback_enabled=%s" % (tag, float(model.feedback_scale), model.feedback_enabled))
        print("   mean|d1|=%.4e mean|d2|=%.4e  cos(d1,d2)=%.5f  max|d2-d1|=%.4e"
              % (float(d1.abs().mean()), float(d2.abs().mean()),
                 float(torch.nn.functional.cosine_similarity(d1.flatten(), d2.flatten(), dim=0)),
                 float((d2-d1).abs().max())))
        del model, task
        torch.cuda.empty_cache()
run("config.yaml", "feedback")
run("config_no_feedback.yaml", "nofeedback")
