"""Pre-GPU functional audit of the two actual training policies."""
import copy
import gc
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import sys
import time
import torch

root=Path(__file__).resolve().parents[1]
assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
assert not (root/"budget.json").exists(), "This audit is preparation only"
sys.path[:0]=[str(root),str(root/"scripts")]
sys.argv=["check_pair_cpu","--root",str(root),"--stage","preflight"]
spec=importlib.util.spec_from_file_location("training_under_audit",root/"scripts/run_training.py")
train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train)
from operations.pair_policy import configure,restore,tail_state,load_tail,equal,fixed_task_hash
from modules.single_system import build_system
from runtime import atomic_json,sha
from recovery.evaluate_existing_final import patched_source
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
train.DEVICE=torch.device("cpu")
train.guard=lambda *args,**kwargs:None
def cpu_forward(up,down,batch,noise_seed,source=False,features=True):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(noise_seed)
        p,pf=up(batch["polar"],batch["official_rgb"],use_adapters=not source)
        y,tf=down(p,batch["official_rgb"],batch["polar"],use_adapters=not source)
    return (p,y,pf,tf) if features else (p,y)
train.forward=cpu_forward
draft=json.loads((root/"config/draft_run.json").read_text())
parent=torch.load(root/"weights/resume_parent_6000.pt",map_location="cpu",weights_only=False)
assert sha(root/"weights/resume_parent_6000.pt")==draft["resume_parent_final_sha256"]
rows=None
results=[]
initial=None
for arm in ("frozen","partial"):
    start=time.time()
    up,down,fa,_,config=build_system(root)
    policy=configure(up,down,arm)
    opt,opt_fa=restore(up,down,fa,parent)
    rows=train.get_rows(config)
    plan=json.loads((root/"config/joint_schedule.json").read_text())
    data=train.dataset(rows,plan)
    batches=[train.to_device(data[i]) for i in (12000,12001)]
    # Use real training pixels; small CPU crop changes only this disposable audit.
    for b in batches:
        for k,v in b.items():
            if torch.is_tensor(v) and v.ndim>=3:b[k]=v[...,:16,:16]
    with torch.no_grad():
        p,y=cpu_forward(up,down,batches[0],2026,features=False)
        digest=hashlib.sha256(y.contiguous().numpy().tobytes()).hexdigest()
        if initial is None:initial=digest
        else:assert digest==initial
        replaced=dict(batches[0]);replaced["target"]=torch.randn_like(replaced["target"])
        assert torch.equal(y,cpu_forward(up,down,replaced,2026,features=False)[1])
    frozen_hash=fixed_task_hash(down)
    core_hash=train.state_hash(up.core)
    before=tail_state(down)
    metrics=train.joint_update(up,down,fa,opt,batches,6001,draft["alignment_weight"])
    changes=sum(not torch.equal(before[n],v) for n,v in tail_state(down).items())
    assert changes==0 if arm=="frozen" else changes>0
    assert fixed_task_hash(down)==frozen_hash and train.state_hash(up.core)==core_hash
    meta=train.meta_update(up,down,fa,opt_fa,batches,list(reversed(batches)),600)
    assert meta["meta_hypergradient_scaled_norm"]>0
    locked={"protocol_id":"disposable_cpu_audit","resume_parent_final_sha256":draft["resume_parent_final_sha256"],
            "source_checkpoint_sha256":draft["source_checkpoint_sha256"]}
    payload=train.checkpoint(arm,6001,601,up,down,fa,opt,opt_fa,locked,"cpu_only_discard")
    stream=io.BytesIO();torch.save(payload,stream);stream.seek(0)
    saved=torch.load(stream,map_location="cpu",weights_only=False)
    assert equal(saved["optimizer"],opt.state_dict())
    assert equal(saved["optimizer_fa"],opt_fa.state_dict())
    with torch.no_grad():expected=cpu_forward(up,down,batches[0],2026,features=False)[1]
    with torch.no_grad():
        for p in up.adapters.parameters():p.add_(.01)
        for p in down.task.net_g.output.parameters():p.add_(.01)
    random.random();torch.rand(3)
    recovered,recovered_fa=restore(up,down,fa,saved)
    assert equal(recovered.state_dict(),saved["optimizer"])
    assert equal(recovered_fa.state_dict(),saved["optimizer_fa"])
    assert torch.equal(torch.get_rng_state(),saved["torch_rng"])
    assert random.getstate()==saved["python_rng"]
    with torch.no_grad():assert torch.equal(expected,cpu_forward(up,down,batches[0],2026,features=False)[1])
    # The evaluator must reset the partial tail before source/frozen inference.
    load_tail(down,down._initial_tail);restored=restore(up,down,fa,parent)
    with torch.no_grad():
        fresh=cpu_forward(up,down,batches[0],2026,features=False)[1]
        assert hashlib.sha256(fresh.contiguous().numpy().tobytes()).hexdigest()==initial
    results.append({"arm":arm,"policy":policy,"initial_prediction_sha256":digest,
        "changed_tail_tensors_after_joint":changes,"frozen_hash_unchanged":True,
        "real_joint":metrics,"real_second_order":meta,"optimizer_rng_output_serialization_equal":True,
        "GT_replacement_output_equal":True,"source_tail_reset_output_equal":True,"seconds":time.time()-start})
    del up,down,fa,opt,opt_fa,recovered,recovered_fa,restored,saved,payload,stream,batches,p,y,expected,fresh
    gc.collect()
source=patched_source((root/"scripts/run_evaluation.py").read_text())
compile(source,str(root/"scripts/run_evaluation.py"),"exec")
namespace={"__name__":"evaluator_cpu_audit","__file__":str(root/"recovery/evaluate_existing_final.py")}
exec(compile(source,str(root/"scripts/run_evaluation.py"),"exec"),namespace)
from rgb_final_confirmation.data import load_manifest
examples=namespace["example_selection"](load_manifest(draft["manifest"]))
assert len(examples)==len(set(examples))==16
assert namespace["NAMES"]==("source","frozen","partial")
atomic_json(root/"diagnostics/cpu_functional_check.json",{
    "passed":True,"time_unix":time.time(),"source_sha256":draft["resume_parent_final_sha256"],
    "arms":results,"real_second_order_and_joint_executed":True,"trial_updates_discarded":True,
    "evaluator_wrapper_compile_passed":True,"fixed_example_ids":examples,
    "current_schedule_hashes":{n:sha(root/"config"/f"{n}_schedule.json") for n in ("joint","meta_inner","meta_outer")},
    "cpu_setup_receipt_is_pre_refinement":True,"joint_plan_refinement":"diagnostics/prelock_joint_plan_refinement.json"})
print("CPU functional audit passed for both arms; trial updates discarded.",flush=True)
