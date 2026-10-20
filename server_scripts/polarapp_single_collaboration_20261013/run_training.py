"""New single-round matched study; no old run or budget may be resumed here."""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import random
import signal
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from PIL import Image

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--root", type=Path, required=True)
PARSER.add_argument("--stage", choices=("preflight", "train"), required=True)
PARSER.add_argument("--arm", choices=("control", "meta", "both"), default="both")
ARGS = PARSER.parse_args()
ROOT = ARGS.root.resolve()
os.environ["POLARAPP_RUN_ROOT"] = str(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from runtime import atomic_json, sha, guard, read_budget, read_locked_config, status
from modules.single_system import build_system, trainable_names, joint_objective, task_objective
from modules.meta import FunctionalModelState, bilevel_probe

DEVICE = torch.device("cuda:0")
STOP = False


def stop_request(_signum, _frame):
    global STOP
    STOP = True


signal.signal(signal.SIGTERM, stop_request)
signal.signal(signal.SIGINT, stop_request)


def seed(value):
    random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def step_seed(kind, step, sample_id):
    return int.from_bytes(hashlib.sha256(f"2026/{kind}/{step}/{sample_id}".encode()).digest()[:4], "little")


def move_system():
    seed(2026)
    up, down, fa, receipt, config = build_system(ROOT)
    up.to(DEVICE).eval(); down.to(DEVICE).eval(); fa.to(DEVICE).eval()
    down.task.model.device = DEVICE
    return up, down, fa, receipt, config


def forward(up, down, batch, noise_seed, source=False, features=True):
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(noise_seed)
        polar, pf = up(batch["polar"], batch["official_rgb"], use_adapters=not source)
        if not features:
            del pf
        prediction, tf = down(polar, batch["official_rgb"], batch["polar"], use_adapters=not source)
    return (polar, prediction, pf, tf) if features else (polar, prediction)


def fa_loss(fa, pf, tf, functional=False):
    transformed, matched = fa.call(pf, tf) if functional else fa(pf, tf)
    return sum(1-F.cosine_similarity(transformed[f"TF{i}"], matched[f"MF{i}"], dim=1).mean()
               for i in (1, 2, 3))/3


def to_device(sample):
    return {k: v[None].to(DEVICE) if torch.is_tensor(v) else v for k, v in sample.items()}


def get_rows(config):
    from rgb_final_confirmation.data import load_manifest
    return [row for row in load_manifest(config["manifest"]) if row.split == "train"]


def make_schedule(rows, steps, crop, salt):
    generator = torch.Generator().manual_seed(2026 + salt)
    indices = []
    while len(indices) < steps*2:
        indices.extend(torch.randperm(len(rows), generator=generator).tolist())
    sizes = {}
    items = []
    for index in indices[:steps*2]:
        if index not in sizes:
            with Image.open(rows[index].polar_000) as image:
                sizes[index] = image.size
        width, height = sizes[index]
        items.append({"sample_index": index,
                      "top": int(torch.randint(0, height-crop+1, (1,), generator=generator)),
                      "left": int(torch.randint(0, width-crop+1, (1,), generator=generator))})
    return {"seed":2026+salt, "steps":steps, "batch_size":2, "crop_size":crop,
            "sample_ids":[r.sample_id for r in rows], "items":items}


def dataset(rows, schedule):
    from rgb_final_confirmation.data import PairedBranchDataset
    return PairedBranchDataset(rows, "A1", schedule["crop_size"], schedule["items"])


def collect_grad_norm(loss, parameters, retain_graph=True):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return math.sqrt(sum(float(g.detach().double().square().sum()) for g in gradients if g is not None))


def adapters_optimizer(up, down):
    return torch.optim.Adam([p for m in (up, down) for p in m.parameters() if p.requires_grad],
                            lr=1e-4, eps=1e-8, foreach=False)


def meta_probe(up, down, fa, inner, outer, step):
    fa.set_trainable_partition()
    states = (FunctionalModelState.from_module(up, trainable_names(up), 1e-4, label="enhancer"),
              FunctionalModelState.from_module(down, trainable_names(down), 1e-4, label="task"),
              FunctionalModelState.from_module(fa, trainable_names(fa), 1e-5, label="alignment"))
    source_polar = []
    with torch.no_grad():
        for batch in outer:
            source_polar.append(up(batch["polar"], batch["official_rgb"], use_adapters=False)[0])
    def inner_objective(u, d, a):
        value = 0.
        for batch in inner:
            guard("train", ROOT)
            _, _, pf, tf = forward(u.call, d.call, batch, step_seed("meta_inner", step, batch["sample_id"]))
            value = value + fa_loss(a, pf, tf, functional=True)/len(inner)
        return value
    def outer_objective(u, d, _a):
        value = 0.
        for batch, reference in zip(outer, source_polar):
            guard("train", ROOT)
            p, y, _, _ = forward(u.call, d.call, batch, step_seed("meta_outer", step, batch["sample_id"]))
            value = value + 1000. * joint_objective(p, y, batch["target"], reference)/len(outer)
        return value
    result = bilevel_probe(*states, inner_objective, outer_objective, outer_create_graph=False)
    return result


def meta_update(up, down, fa, optimizer, inner, outer, step):
    result = meta_probe(up, down, fa, inner, outer, step)
    optimizer.zero_grad(set_to_none=True)
    named = dict(fa.named_parameters())
    for name, gradient in result.hypergradients.items():
        named[name].grad = gradient.detach().clone()
    torch.nn.utils.clip_grad_norm_([p for p in fa.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
    optimizer.step()
    values = {"meta_inner_loss":float(result.inner_loss.detach()),
              "meta_outer_raw_loss":float(result.outer_loss.detach())/1000.,
              "meta_outer_scaled_loss":float(result.outer_loss.detach()),
              "meta_hypergradient_raw_norm":result.audit["alignment_hypergradient_norm"]/1000.,
              "meta_hypergradient_scaled_norm":result.audit["alignment_hypergradient_norm"],
              "meta_outer_scale":1000.}
    del result
    return values


def joint_update(up, down, fa, optimizer, batches, step, alignment_weight):
    fa.requires_grad_(False).eval()
    optimizer.zero_grad(set_to_none=True)
    totals = {"task_loss":0., "retention_loss":0., "alignment_loss":0., "joint_loss":0.}
    for batch in batches:
        guard("train", ROOT)
        with torch.no_grad():
            reference = up(batch["polar"], batch["official_rgb"], use_adapters=False)[0]
        p, y, pf, tf = forward(up, down, batch, step_seed("joint", step, batch["sample_id"]))
        task = task_objective(y, batch["target"])
        keep = (p-reference).square().mean()
        alignment = fa_loss(fa, pf, tf) if alignment_weight else task.new_zeros(())
        loss = task+.1*keep+alignment_weight*alignment
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite joint objective")
        (loss/len(batches)).backward()
        for key, value in zip(totals, (task,keep,alignment,loss)):
            totals[key] += float(value.detach())/len(batches)
        del p,y,pf,tf,loss
    parameters = [p for m in (up, down) for p in m.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    optimizer.step()
    totals["adapter_gradient_norm"] = float(norm)
    return totals


@torch.no_grad()
def retention_probe(up,down,data,arm,step):
    records=[]
    for i in range(4):
        guard("train",ROOT)
        batch=to_device(data[i]);fixed_seed=step_seed("fixed_training_retention_probe",0,batch["sample_id"])
        ps,ys=forward(up,down,batch,fixed_seed,source=True,features=False)
        p,y=forward(up,down,batch,fixed_seed,features=False)
        records.append({"sample_id":batch["sample_id"],"source_RGB_MSE":float((ys-batch["target"]).square().mean()),
            "adapted_RGB_MSE":float((y-batch["target"]).square().mean()),
            "source_polar_MSE_drift":float((p-ps).square().mean()),
            "source_prediction_sha256":hashlib.sha256(ys.cpu().contiguous().numpy().tobytes()).hexdigest()})
    initial_path=ROOT/"downstream/metrics"/f"{arm}_probe_initial.json"
    if step==0:
        atomic_json(initial_path,{"arm":arm,"step":step,"records":records,"training_only":True,"used_for_selection":False})
    else:
        initial=json.loads(initial_path.read_text())
        assert [(r["sample_id"],r["source_prediction_sha256"]) for r in records]==[(r["sample_id"],r["source_prediction_sha256"]) for r in initial["records"]]
        atomic_json(ROOT/"downstream/metrics"/f"{arm}_probe_{step}.json",{
            "arm":arm,"step":step,"records":records,"training_only":True,"used_for_selection":False,
            "source_predictions_fixed":True})


def state_hash(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def check_frozen(up, down, fa, reference):
    assert state_hash(up.core) == reference["enhancer"]
    assert state_hash(down.task) == reference["task"]
    for name, parameter in fa.named_parameters():
        if name in reference["FA_frozen"]:
            assert torch.equal(parameter.detach().cpu(), reference["FA_frozen"][name])


def preflight():
    guard("preflight", ROOT)
    up, down, fa, receipt, config = move_system()
    rows = get_rows(config)
    assert len(rows) == 5440
    inner_rows = [r for r in rows if r.scene in config["meta_inner_scene_ids"]]
    outer_rows = [r for r in rows if r.scene in config["meta_outer_scene_ids"]]
    crop = int(config.get("training_crop_size",128))
    joint_schedule = make_schedule(rows, 3000, crop, 0)
    inner_schedule = make_schedule(inner_rows, 300, crop, 101)
    outer_schedule = make_schedule(outer_rows, 300, crop, 202)
    joint_data, inner_data, outer_data = (dataset(rows,joint_schedule), dataset(inner_rows,inner_schedule),dataset(outer_rows,outer_schedule))
    initial = (copy.deepcopy(up.adapters.state_dict()), copy.deepcopy(down.adapters.state_dict()), copy.deepcopy(fa.state_dict()))
    optimizer = adapters_optimizer(up, down)
    optimizer_fa = torch.optim.Adam([p for p in fa.parameters() if p.requires_grad], lr=1e-5, eps=1e-8, foreach=False)
    batches = [to_device(joint_data[i]) for i in (0,1)]
    fa.set_trainable_partition()
    torch.cuda.reset_peak_memory_stats()
    tick = time.perf_counter()
    result = meta_update(up,down,fa,optimizer_fa,[to_device(inner_data[i]) for i in (0,1)],
                         [to_device(outer_data[i]) for i in (0,1)],0)
    torch.cuda.synchronize()
    meta_seconds = time.perf_counter()-tick
    meta_peak = torch.cuda.max_memory_allocated()/2**30
    assert result["meta_hypergradient_scaled_norm"] > 0
    print(json.dumps({"preflight":"meta_passed", "seconds":meta_seconds,"peak_GiB":meta_peak,**result}),flush=True)
    # Always discard this trial before calibrating the formal objective.
    up.adapters.load_state_dict(initial[0]);down.adapters.load_state_dict(initial[1]);fa.load_state_dict(initial[2])
    del optimizer,optimizer_fa
    gc.collect();torch.cuda.empty_cache()
    norms = []
    parameters = tuple(p for m in (up,down) for p in m.parameters() if p.requires_grad)
    fa.requires_grad_(False)
    for i in range(8):
        guard("preflight",ROOT)
        batch = to_device(joint_data[i])
        with torch.no_grad():reference = up(batch["polar"],batch["official_rgb"],use_adapters=False)[0]
        p,y,pf,tf = forward(up,down,batch,step_seed("calibration",i,batch["sample_id"]))
        primary = joint_objective(p,y,batch["target"],reference)
        alignment = fa_loss(fa,pf,tf)
        norms.append({"sample_id":batch["sample_id"],"task_gradient_norm":collect_grad_norm(primary,parameters),
                      "alignment_gradient_norm":collect_grad_norm(alignment,parameters,False)})
        del p,y,pf,tf,primary,alignment
    task_median = float(torch.tensor([x["task_gradient_norm"] for x in norms]).median())
    align_median = float(torch.tensor([x["alignment_gradient_norm"] for x in norms]).median())
    if not all(math.isfinite(x) and x>0 for x in (task_median,align_median)):
        raise RuntimeError("Training-only gradient calibration is degenerate")
    weight = min(.1,.1*task_median/align_median)
    atomic_json(ROOT/"config/calibration.json",{"records":norms,"target_fraction":.1,"cap":.1,
                "task_median":task_median,"alignment_median":align_median,"alignment_weight":weight,
                "used_validation_or_test":False,"preflight_updates_discarded":True})
    durations={}; peaks={}
    for arm,lambda_fa in (("control",0.),("meta",weight)):
        up.adapters.load_state_dict(initial[0]);down.adapters.load_state_dict(initial[1]);fa.load_state_dict(initial[2])
        optimizer=adapters_optimizer(up,down);torch.cuda.reset_peak_memory_stats()
        tick=time.perf_counter();joint_update(up,down,fa,optimizer,batches,0,lambda_fa);torch.cuda.synchronize()
        durations[arm]=time.perf_counter()-tick;peaks[arm]=torch.cuda.max_memory_allocated()/2**30
        del optimizer;gc.collect();torch.cuda.empty_cache()
    up.adapters.load_state_dict(initial[0]);down.adapters.load_state_dict(initial[1]);fa.load_state_dict(initial[2])
    from rgb_final_confirmation.data import PairedBranchDataset
    maximum = (-1,None)
    for row in rows:
        with Image.open(row.official_rgb) as image:
            size=image.width*image.height
        if size>maximum[0]:maximum=(size,row)
    full=to_device(PairedBranchDataset([maximum[1]],"A1",None)[0])
    torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
    with torch.no_grad():p,y=forward(up,down,full,step_seed("full_resource",0,full["sample_id"]),features=False)
    torch.cuda.synchronize();full_seconds=time.perf_counter()-tick
    full_shape=list(y.shape);full_peak=torch.cuda.max_memory_allocated()/2**30
    del full,p,y
    expected_training=3000*sum(durations.values())+300*meta_seconds
    expected_evaluation=1060*3*full_seconds
    budget=read_budget(ROOT)
    # Four hours is a minimum reserve, not a cap on full-image evaluation.
    # The operational cutoff may move earlier; the original budget never moves.
    evaluation_reserve=max(float(budget["evaluation_reserve_seconds"]),expected_evaluation*1.2+1800.)
    operational_training_deadline=min(budget["training_deadline_unix"],budget["deadline_unix"]-evaluation_reserve)
    measurement={"source":receipt,"crop_size":crop,"batch_size":2,"microbatch":1,
                 "joint_step_seconds":durations,"joint_peak_GiB":peaks,"meta_step_seconds":meta_seconds,
                 "meta_peak_GiB":meta_peak,"meta_gradient_check":result,
                 "full_image_seconds":full_seconds,"full_image_shape":full_shape,"full_peak_GiB":full_peak,
                 "full_forward":True,"tiling":False,"resource_image_split":"train",
                 "expected_training_seconds":expected_training,"expected_three_output_evaluation_seconds":expected_evaluation,
                 "evaluation_reserve_seconds":evaluation_reserve,
                 "operational_training_deadline_unix":operational_training_deadline,
                 "preflight_updates_discarded":True}
    atomic_json(ROOT/"diagnostics/gpu_preflight.json",measurement)
    print(json.dumps({"preflight":"resource_forecast",**measurement}),flush=True)
    remaining=operational_training_deadline-time.time()
    if expected_training*1.5>remaining:
        raise RuntimeError("Fixed matched training plus conservative full-image evaluation cannot fit original absolute budget")
    for name,schedule in (("joint",joint_schedule),("meta_inner",inner_schedule),("meta_outer",outer_schedule)):
        atomic_json(ROOT/"config"/f"{name}_schedule.json",schedule)
    code_files = sorted(p for folder in (ROOT/"scripts",ROOT/"modules",ROOT/"source_snapshot",ROOT/"vendor")
                        for p in folder.rglob("*.py") if "__pycache__" not in p.parts)
    code_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in code_files}
    locked={"schema_version":1,"scope":"single_round_parameter_efficient_PolarAPP_core_transfer",
            "seed":2026,"arms":["control","meta"],"joint_steps":3000,"meta_interval":10,"meta_updates":300,
            "crop_size":crop,"effective_batch_size":2,"microbatch":1,"lr_adapters":1e-4,"lr_FANet":1e-5,
            "inner_lr":1e-4,"Adam_epsilon":1e-8,"meta_outer_loss_scale":1000.,"alignment_weight":weight,
            "task_loss":"RGB MSE + .02 edge MAE","source_polar_retention_weight":.1,
            "source_checkpoint_sha256":config["source_checkpoint_sha256"],"manifest":config["manifest"],
            "manifest_sha256":sha(config["manifest"]),"source_configuration_sha256":sha(ROOT/"config/prepare.json"),
            "schedule_hashes":{n:sha(ROOT/"config"/f"{n}_schedule.json") for n in ("joint","meta_inner","meta_outer")},
            "code_sha256":code_hashes,"budget":budget,"selection":"fixed endpoint; no validation/test choice",
            "operational_training_deadline_unix":operational_training_deadline,
            "forecast_evaluation_reserve_seconds":evaluation_reserve,
            "evaluation_counts":{"validation":872,"test":188},"metrics":"raw existing PSNR/SSIM; clipped supplementary only",
            "new_budget_reused_from_old":False,"matched_joint_updates_not_matched_total_compute":True,
            "official_FANet_sha256":sha(ROOT/"weights/FANet.pth"),"calibration_sha256":sha(ROOT/"config/calibration.json")}
    locked["protocol_id"]=hashlib.sha256(json.dumps(locked,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    atomic_json(ROOT/"config/locked_run.json",locked)
    atomic_json(ROOT/"config/lock_receipt.json",{"protocol_id":locked["protocol_id"],
                "locked_sha256":sha(ROOT/"config/locked_run.json"),"locked_at_unix":time.time()})
    status("preflight_passed_protocol_locked",ROOT,protocol_id=locked["protocol_id"],alignment_weight=weight,
           projected_training_hours=expected_training/3600.,projected_evaluation_hours=expected_evaluation/3600.)


def append_csv(path,row):
    path.parent.mkdir(parents=True,exist_ok=True);exists=path.exists()
    with path.open("a",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(row))
        if not exists:writer.writeheader()
        writer.writerow(row)


def atomic_checkpoint(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp")
    torch.save(payload,temporary);temporary.replace(path)
    atomic_json(path.with_suffix(".sha256.json"),{"sha256":sha(path),"step":payload["step"],"protocol_id":payload["protocol_id"]})


def checkpoint(arm,step,meta_steps,up,down,fa,optimizer,optimizer_fa,locked,selection):
    return {"arm":arm,"step":step,"joint_step":step,"meta_steps":meta_steps,"protocol_id":locked["protocol_id"],
            "source_sha256":locked["source_checkpoint_sha256"],"upstream_adapters":up.adapters.state_dict(),
            "downstream_adapters":down.adapters.state_dict(),"alignment_state":fa.state_dict() if arm=="meta" else None,
            "optimizer":optimizer.state_dict(),"optimizer_fa":optimizer_fa.state_dict() if arm=="meta" else None,
            "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),"python_rng":random.getstate(),
            "selection":selection,"no_validation_test_used_for_selection":True}


def verify_checkpoint(path,locked):
    receipt=json.loads(path.with_suffix(".sha256.json").read_text())
    if sha(path)!=receipt["sha256"]:raise RuntimeError("Checkpoint SHA256 mismatch")
    saved=torch.load(path,map_location="cpu",weights_only=False)
    if saved["protocol_id"]!=locked["protocol_id"] or saved["source_sha256"]!=locked["source_checkpoint_sha256"]:
        raise RuntimeError("Checkpoint belongs to different locked protocol/source")
    return saved


def train_arm(arm,locked):
    target=ROOT/"downstream/checkpoints"/arm;final=target/"final.pt";last=target/"last.pt"
    if final.exists():
        saved=verify_checkpoint(final,locked)
        if saved["step"]!=locked["joint_steps"]:raise RuntimeError("Incomplete saved endpoint cannot count as a matched completed arm")
        status("arm_already_complete",ROOT,arm=arm,step=saved["step"]);return
    up,down,fa,receipt,config=move_system();fa.set_trainable_partition()
    frozen={"enhancer":state_hash(up.core),"task":state_hash(down.task),
            "FA_frozen":{n:p.detach().cpu().clone() for n,p in fa.named_parameters() if not p.requires_grad}}
    optimizer=adapters_optimizer(up,down)
    optimizer_fa=torch.optim.Adam([p for p in fa.parameters() if p.requires_grad],lr=1e-5,eps=1e-8,foreach=False)
    start=meta_steps=0
    if last.exists():
        saved=verify_checkpoint(last,locked);start=saved["step"];meta_steps=saved["meta_steps"]
        up.adapters.load_state_dict(saved["upstream_adapters"],strict=True)
        down.adapters.load_state_dict(saved["downstream_adapters"],strict=True)
        if arm=="meta":fa.load_state_dict(saved["alignment_state"],strict=True);optimizer_fa.load_state_dict(saved["optimizer_fa"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"]);torch.cuda.set_rng_state_all(saved["cuda_rng"]);random.setstate(saved["python_rng"])
        log_path=ROOT/"downstream/logs"/f"{arm}_training.csv"
        if log_path.exists():
            with log_path.open(newline="",encoding="utf-8") as stream:old_rows=list(csv.DictReader(stream))
            unsaved=[row for row in old_rows if int(row["step"])>start]
            if unsaved:
                atomic_json(ROOT/"diagnostics"/f"{arm}_discarded_unsaved_log_{int(time.time())}.json",
                    {"restored_step":start,"discarded_rows":unsaved,"reason":"restore only verified saved checkpoint"})
                kept=[row for row in old_rows if int(row["step"])<=start]
                with log_path.open("w",newline="",encoding="utf-8") as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(old_rows[0]));writer.writeheader();writer.writerows(kept)
    rows=get_rows(config);inner_rows=[r for r in rows if r.scene in config["meta_inner_scene_ids"]]
    outer_rows=[r for r in rows if r.scene in config["meta_outer_scene_ids"]]
    schedules={n:json.loads((ROOT/"config"/f"{n}_schedule.json").read_text()) for n in ("joint","meta_inner","meta_outer")}
    datasets={"joint":dataset(rows,schedules["joint"]),"meta_inner":dataset(inner_rows,schedules["meta_inner"]),
              "meta_outer":dataset(outer_rows,schedules["meta_outer"])}
    if start==0:retention_probe(up,down,datasets["joint"],arm,0)
    coverage=set(schedules["joint"]["sample_ids"][x["sample_index"]] for x in schedules["joint"]["items"][:start*2])
    selection="fixed_3000_joint_endpoint"
    for step in range(start+1,locked["joint_steps"]+1):
        if STOP or time.time()>=min(read_budget(ROOT)["training_deadline_unix"],locked["operational_training_deadline_unix"]):
            selection="immutable_training_deadline_or_interrupt_incomplete"
            atomic_checkpoint(last,checkpoint(arm,step-1,meta_steps,up,down,fa,optimizer,optimizer_fa,locked,selection))
            status("training_stopped_incomplete",ROOT,arm=arm,step=step-1)
            raise RuntimeError("Matched training incomplete; no further GPU updates allowed at training cutoff")
        guard("train",ROOT);tick=time.perf_counter();meta_values={k:0. for k in ("meta_inner_loss","meta_outer_raw_loss",
            "meta_outer_scaled_loss","meta_hypergradient_raw_norm","meta_hypergradient_scaled_norm","meta_outer_scale")}
        if arm=="meta" and step%10==0:
            index=meta_steps*2
            meta_values=meta_update(up,down,fa,optimizer_fa,
                [to_device(datasets["meta_inner"][i]) for i in (index,index+1)],
                [to_device(datasets["meta_outer"][i]) for i in (index,index+1)],meta_steps)
            meta_steps+=1
        batches=[to_device(datasets["joint"][i]) for i in ((step-1)*2,(step-1)*2+1)]
        coverage.update(b["sample_id"] for b in batches)
        values=joint_update(up,down,fa,optimizer,batches,step,locked["alignment_weight"] if arm=="meta" else 0.)
        torch.cuda.synchronize()
        row={"step":step,"arm":arm,"unique_joint_images":len(coverage),"seconds":time.perf_counter()-tick,
             "lr_adapters":optimizer.param_groups[0]["lr"],"lr_FANet":optimizer_fa.param_groups[0]["lr"] if arm=="meta" else 0.,
             "alignment_weight":locked["alignment_weight"] if arm=="meta" else 0.,**values,**meta_values}
        append_csv(ROOT/"downstream/logs"/f"{arm}_training.csv",row)
        if step%100==0:status("training",ROOT,arm=arm,step=step,total_steps=3000,unique_joint_images=len(coverage),loss=values["joint_loss"])
        if step%500==0 or step==locked["joint_steps"]:
            check_frozen(up,down,fa,frozen)
            atomic_checkpoint(last,checkpoint(arm,step,meta_steps,up,down,fa,optimizer,optimizer_fa,locked,selection))
            retention_probe(up,down,datasets["joint"],arm,step)
        del batches
    assert len(coverage)==5440
    check_frozen(up,down,fa,frozen)
    atomic_checkpoint(final,checkpoint(arm,locked["joint_steps"],meta_steps,up,down,fa,optimizer,optimizer_fa,locked,selection))
    atomic_json(target/"training_complete.json",{"arm":arm,"joint_steps":3000,"meta_updates":meta_steps,
        "unique_joint_images":len(coverage),"protocol_id":locked["protocol_id"],"final_sha256":sha(final),
        "frozen_parameters_unchanged":True,"selection":selection})
    status("arm_training_complete",ROOT,arm=arm,step=3000,meta_updates=meta_steps,source_unchanged=True)
    del up,down,fa,optimizer,optimizer_fa;gc.collect();torch.cuda.empty_cache()


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    guard("preflight" if ARGS.stage=="preflight" else "train",ROOT)
    if ARGS.stage=="preflight":
        if (ROOT/"config/locked_run.json").exists():
            raise RuntimeError("Existing locked protocol cannot be reset by rerunning preflight")
        preflight();return
    locked=read_locked_config(ROOT)
    if locked["arms"] != ["meta"] or ARGS.arm not in ("meta","both"):
        raise RuntimeError("User authorized meta-only training; cancelled control must not resume")
    for name,digest in locked["code_sha256"].items():
        if sha(ROOT/name)!=digest:raise RuntimeError(f"Frozen code changed: {name}")
    if sha(locked["manifest"])!=locked["manifest_sha256"]:raise RuntimeError("Dataset manifest changed after lock")
    for name,digest in locked["schedule_hashes"].items():
        if sha(ROOT/"config"/f"{name}_schedule.json")!=digest:raise RuntimeError("Locked sample/crop schedule changed")
    for arm in locked["arms"] if ARGS.arm=="both" else [ARGS.arm]:
        train_arm(arm,locked)
    status("training_complete",ROOT,arms=locked["arms"],joint_steps_per_arm=3000,validation_test_used_for_training=False)


if __name__=="__main__":
    main()
