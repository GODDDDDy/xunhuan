import importlib.util, sys
from pathlib import Path
import torch, yaml
R = Path("/media/li/data3/25thstudent/DY/929work/pidsr_k2_929")
SRC = R / "deps/rgb_final_confirmation/src"
def load_module(n,p):
    s=importlib.util.spec_from_file_location(n,str(p)); m=importlib.util.module_from_spec(s)
    sys.modules[n]=m; s.loader.exec_module(m); return m
def ry(p): return yaml.safe_load(open(p,encoding="utf-8"))
dev=torch.device("cuda:0")
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
mod=load_module("re_c", R/"experiment/run_experiment.py")
cfg=ry(R/"experiment/config.yaml"); mod.add_source_path(cfg); mod.set_seed(2026)
model,task,crit,base=mod.build_components(cfg,dev); del crit
st=torch.load(Path(cfg["paths"]["output_root"])/"checkpoints/step_3000.pt",map_location=dev,weights_only=False)
model.load_state_dict(st["r2p_and_projections"],strict=True); task.net_g.load_state_dict(st["tasknet_net_g"],strict=True); del st
ld=mod.split_loader(cfg,"validation"); model.eval(); task.eval()
raw=next(iter(ld)); b=mod.unpack_batch(raw,dev); mod.set_step_seed(2026,"validation",0,b["sample_id"])
with torch.no_grad():
    p=b["polar"]
    o1=model(p,b["condition"],feedback=False if False else None,project_features=False)
    p1=o1.refined_polar.clone(); d1=o1.delta.clone()
    # raw sum BEFORE clamp, vs clamp result
    raw_sum = p + d1
    clamped = raw_sum.clamp(0,1)
    print("max|P1 - clamped(p+d1)| = %.3e  (must be 0: identity holds)" % float((p1-clamped).abs().max()))
    print("max|P1 - (p+d1) UNCLAMPED| = %.6e" % float((p1-raw_sum).abs().max()))
    # where does clamp actually bite?
    low = (raw_sum < 0).float().mean(); high = (raw_sum > 1).float().mean()
    print("frac raw_sum<0 = %.6f ; frac raw_sum>1 = %.6f ; total clipped = %.6f" % (float(low), float(high), float(low+high)))
    # effective update vs raw delta
    eff = (p1 - p)
    print("mean|raw delta d1| = %.6e ; mean|effective update P1-p| = %.6e ; ratio = %.6f"
          % (float(d1.abs().mean()), float(eff.abs().mean()), float(eff.abs().mean()/d1.abs().mean())))
    print("max|raw d1| = %.6e ; max|effective| = %.6e" % (float(d1.abs().max()), float(eff.abs().max())))
