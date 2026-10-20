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
PARSER.add_argument("--arm", choices=("frozen", "partial", "both"), default="both")
ARGS = PARSER.parse_args()
ROOT = ARGS.root.resolve()
os.environ["POLARAPP_RUN_ROOT"] = str(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path[:0] = [str(ROOT / "source_snapshot/scripts"), str(ROOT / "source_snapshot/source")]
from runtime import atomic_json, sha, guard, read_budget, read_locked_config, status
from modules.single_system import build_system, trainable_names, joint_objective, task_objective
from modules.meta import FunctionalModelState, bilevel_probe
from operations.pair_policy import configure, restore, tail_state, reference_tail, fixed_task_hash

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
              FunctionalModelState.from_module(down,
                  tuple(n for n in trainable_names(down) if not n.startswith("task.model.net_g.output.")),
                  {n:(1e-6 if n.startswith("task.") else 1e-4) for n in trainable_names(down)
                   if not n.startswith("task.model.net_g.output.")}, label="task"),
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
    adapter_norm=math.sqrt(sum(float(p.grad.detach().double().square().sum())
        for m in (up.adapters,down.adapters) for p in m.parameters() if p.grad is not None))
    tail_norm=math.sqrt(sum(float(p.grad.detach().double().square().sum())
        for p in down.task.parameters() if p.requires_grad and p.grad is not None))
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    optimizer.step()
    totals["adapter_gradient_norm"] = adapter_norm
    totals["task_tail_gradient_norm"] = tail_norm
    totals["all_trainable_gradient_norm"] = float(norm)
    totals["global_clip_coefficient"] = min(1.,1./(float(norm)+1e-6))
    return totals


@torch.no_grad()
def retention_probe(up,down,data,arm,step):
    records=[]
    for i in range(4):
        guard("train",ROOT)
        batch=to_device(data[i]);fixed_seed=step_seed("fixed_training_retention_probe",0,batch["sample_id"])
        with reference_tail(down):
            ps,ys=forward(up,down,batch,fixed_seed,source=True,features=False)
        p,y=forward(up,down,batch,fixed_seed,features=False)
        records.append({"sample_id":batch["sample_id"],"source_RGB_MSE":float((ys-batch["target"]).square().mean()),
            "adapted_RGB_MSE":float((y-batch["target"]).square().mean()),
            "source_polar_MSE_drift":float((p-ps).square().mean()),
            "source_prediction_sha256":hashlib.sha256(ys.cpu().contiguous().numpy().tobytes()).hexdigest()})
    initial_path=ROOT/"downstream/metrics"/f"{arm}_probe_initial.json"
    if not initial_path.exists():
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
    assert fixed_task_hash(down) == reference["task"]
    for name, parameter in fa.named_parameters():
        if name in reference["FA_frozen"]:
            assert torch.equal(parameter.detach().cpu(), reference["FA_frozen"][name])


def preflight():
    """Real two-arm CUDA audit; all trial updates and random draws discarded."""
    config=json.loads((ROOT/'config/prepare.json').read_text())
    draft=json.loads((ROOT/'config/draft_run.json').read_text())
    assert sha(ROOT/'weights/resume_parent_6000.pt')==draft['resume_parent_final_sha256']
    assert json.loads((ROOT/'diagnostics/cpu_functional_check.json').read_text())['passed']
    rows=get_rows(config)
    schedules={n:json.loads((ROOT/'config'/f'{n}_schedule.json').read_text()) for n in ('joint','meta_inner','meta_outer')}
    datasets={'joint':dataset(rows,schedules['joint'])}
    for kind,key in (('meta_inner','meta_inner_scene_ids'),('meta_outer','meta_outer_scene_ids')):
        datasets[kind]=dataset([r for r in rows if r.scene in config[key]],schedules[kind])
    observations=[]
    for arm in ('frozen','partial'):
        parent=torch.load(ROOT/'weights/resume_parent_6000.pt',map_location='cpu',weights_only=False)
        up,down,fa,_,_=move_system()
        policy=configure(up,down,arm)
        opt,opt_fa=restore(up,down,fa,parent)
        fa.set_trainable_partition()
        frozen={'enhancer':state_hash(up.core),'task':fixed_task_hash(down),
                'FA_frozen':{n:p.detach().cpu().clone() for n,p in fa.named_parameters() if not p.requires_grad}}
        before_tail=tail_state(down)
        torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
        inner=[to_device(datasets['meta_inner'][i]) for i in (1200,1201)]
        outer=[to_device(datasets['meta_outer'][i]) for i in (1200,1201)]
        meta_values=meta_update(up,down,fa,opt_fa,inner,outer,600)
        torch.cuda.synchronize();meta_seconds=time.perf_counter()-tick
        joint_tick=time.perf_counter()
        batches=[to_device(datasets['joint'][i]) for i in (12000,12001)]
        values=joint_update(up,down,fa,opt,batches,6001,draft['alignment_weight'])
        torch.cuda.synchronize();elapsed=time.perf_counter()-tick
        joint_seconds=time.perf_counter()-joint_tick
        check_frozen(up,down,fa,frozen)
        after_tail=tail_state(down)
        changed=sum(not torch.equal(before_tail[n],after_tail[n]) for n in before_tail)
        assert changed==0 if arm=='frozen' else changed>0
        assert meta_values['meta_hypergradient_scaled_norm']>0
        max_bytes=torch.cuda.max_memory_allocated()
        del inner,outer,batches,opt,opt_fa,before_tail,after_tail
        gc.collect();torch.cuda.empty_cache()
        from operations.pair_policy import load_tail
        load_tail(down,down._initial_tail)
        restored=restore(up,down,fa,parent);del restored
        from rgb_final_confirmation.data import load_manifest,PairedBranchDataset
        candidates=[r for r in load_manifest(config['manifest']) if r.split=='validation']
        def area(row):
            with Image.open(row.polar_000) as image:return image.width*image.height
        largest=max(candidates,key=area)
        batch=to_device(PairedBranchDataset([largest],'A1',None)[0])
        tick=time.perf_counter()
        with torch.no_grad():
            p,y=forward(up,down,batch,123,features=False)
            assert y.shape==batch['target'].shape and torch.isfinite(y).all()
        torch.cuda.synchronize()
        observations.append({'arm':arm,'policy':policy,'bilevel_joint_trial_seconds':elapsed,
            'joint_trial_seconds':joint_seconds,'meta_trial_seconds':meta_seconds,
            'peak_training_bytes':max_bytes,'meta_hypergradient_scaled_norm':meta_values['meta_hypergradient_scaled_norm'],
            'trial_changed_tail_tensors':changed,'frozen_audit_passed':True,
            'native_size':list(y.shape[-2:]),'native_forward_seconds':time.perf_counter()-tick,'trial_updates_discarded':True})
        del p,y,batch,up,down,fa,parent
        gc.collect();torch.cuda.empty_cache()
    budget=read_budget(ROOT)
    projected_train=sum(3000*x['joint_trial_seconds']+300*x['meta_trial_seconds'] for x in observations)*1.5+2400
    projected_eval=max(x['native_forward_seconds'] for x in observations)*1060*3*1.5+1800
    available=budget['deadline_unix']-time.time()
    if projected_train+max(14400,projected_eval)>available:
        raise RuntimeError('Conservative paired training and full evaluation forecast exceeds immutable budget')
    draft['operational_training_deadline_unix']=min(budget['training_deadline_unix'],budget['deadline_unix']-max(14400,projected_eval))
    draft['budget']=budget
    draft['preflight_receipt']={'observations':observations,'projected_training_seconds':projected_train,
                              'projected_evaluation_seconds':projected_eval,'trial_updates_discarded':True}
    atomic_json(ROOT/'diagnostics/gpu_preflight.json',draft['preflight_receipt'])
    draft['code_sha256']={p.relative_to(ROOT).as_posix():sha(p)
        for folder in ('scripts','modules','source_snapshot','vendor','recovery','operations')
        for p in sorted((ROOT/folder).rglob('*.py')) if '__pycache__' not in p.parts}
    draft['protocol_id']=hashlib.sha256(json.dumps(draft,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    atomic_json(ROOT/'config/locked_run.json',draft)
    atomic_json(ROOT/'config/lock_receipt.json',{'protocol_id':draft['protocol_id'],'locked_sha256':sha(ROOT/'config/locked_run.json'),'locked_at_unix':time.time()})
    status('preflight_passed_protocol_locked',ROOT,protocol_id=draft['protocol_id'],observations=observations)



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
            "resume_parent_final_sha256":locked["resume_parent_final_sha256"],
            "source_sha256":locked["source_checkpoint_sha256"],"upstream_adapters":up.adapters.state_dict(),
            "downstream_adapters":down.adapters.state_dict(),"alignment_state":fa.state_dict(),"task_tail":tail_state(down),
            "optimizer":optimizer.state_dict(),"optimizer_fa":optimizer_fa.state_dict(),
            "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),"python_rng":random.getstate(),
            "selection":selection,"no_validation_test_used_for_selection":True}


def verify_checkpoint(path,locked):
    receipt=json.loads(path.with_suffix(".sha256.json").read_text())
    if sha(path)!=receipt["sha256"]:raise RuntimeError("Checkpoint SHA256 mismatch")
    saved=torch.load(path,map_location="cpu",weights_only=False)
    if saved["protocol_id"]!=locked["protocol_id"] or saved["source_sha256"]!=locked["source_checkpoint_sha256"]:
        raise RuntimeError("Checkpoint belongs to different locked protocol/source")
    return saved


@torch.no_grad()
@torch.random.fork_rng(devices=[0])
def fixed_validation_probe(up,down,locked,step,arm):
    """Monitor 16 fixed original-size validation frames; never choose a weight."""
    from rgb_final_confirmation.data import load_manifest, PairedBranchDataset
    from rgb_final_confirmation.evaluation import psnr, ssim, stable_sample_seed
    from bridge import anchored_task_inputs
    wanted=locked["fixed_validation_probe_ids"]
    all_rows={r.sample_id:r for r in load_manifest(locked["manifest"]) if r.split=="validation"}
    data=PairedBranchDataset([all_rows[x] for x in wanted],"A1",None)
    with (ROOT/"config/parent_per_image_validation.csv").open(newline="",encoding="utf-8") as stream:
        parent={r["sample_id"]:r for r in csv.DictReader(stream)}
    records=[]
    for i,sample_id in enumerate(wanted):
        guard("train",ROOT)
        batch=to_device(data[i])
        p,features=up(batch["polar"],batch["official_rgb"]);del features
        inputs=anchored_task_inputs(p.float(),batch["polar"].float(),batch["official_rgb"].float())
        condition=down.task.model.net_le_dm(inputs)
        noise=torch.randn(tuple(condition.shape),dtype=condition.dtype,device="cpu",
            generator=torch.Generator(device="cpu").manual_seed(stable_sample_seed(sample_id,base_seed=2026))).to(DEVICE)
        del inputs,condition
        prediction,features=down(p,batch["official_rgb"],batch["polar"],noise=noise);del features
        digest=hashlib.sha256(prediction.float().cpu().contiguous().numpy().tobytes()).hexdigest()
        if step==6000 and digest!=parent[sample_id]["meta_prediction_sha256"]:
            raise RuntimeError("Restored parent output differs before first new update")
        records.append({"sample_id":sample_id,"PSNR":psnr(prediction,batch["target"]),
            "SSIM":ssim(prediction,batch["target"]),"parent6000_PSNR":float(parent[sample_id]["meta_psnr"]),
            "parent6000_SSIM":float(parent[sample_id]["meta_ssim"]),"prediction_sha256":digest})
        del batch,p,prediction,noise
    atomic_json(ROOT/"downstream/metrics"/f"{arm}_validation_probe_{step}.json",{
        "arm":arm,"step":step,"records":records,"full_native_size_forward":True,"validation_only":True,
        "used_for_selection":False,"PSNR_mean":sum(r["PSNR"] for r in records)/len(records),
        "SSIM_mean":sum(r["SSIM"] for r in records)/len(records)})
    gc.collect();torch.cuda.empty_cache()


def train_arm(arm,locked):
    target=ROOT/"downstream/checkpoints"/arm;final=target/"final.pt";last=target/"last.pt"
    if final.exists():
        saved=verify_checkpoint(final,locked)
        if saved["step"]!=locked["joint_steps"]:raise RuntimeError("Incomplete saved endpoint cannot count as a matched completed arm")
        status("arm_already_complete",ROOT,arm=arm,step=saved["step"]);return
    up,down,fa,receipt,config=move_system();configure(up,down,arm);fa.set_trainable_partition()
    frozen={"enhancer":state_hash(up.core),"task":fixed_task_hash(down),
            "FA_frozen":{n:p.detach().cpu().clone() for n,p in fa.named_parameters() if not p.requires_grad}}
    saved=verify_checkpoint(last,locked) if last.exists() else torch.load(ROOT/'weights/resume_parent_6000.pt',map_location='cpu',weights_only=False)
    start=saved['step'];meta_steps=saved['meta_steps']
    optimizer,optimizer_fa=restore(up,down,fa,saved)
    if last.exists():
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
    if start==6000:retention_probe(up,down,datasets["joint"],arm,6000)
    if start==6000 and not (ROOT/"downstream/metrics"/f"{arm}_validation_probe_6000.json").exists():
        fixed_validation_probe(up,down,locked,6000,arm)
    coverage=set(schedules["joint"]["sample_ids"][x["sample_index"]] for x in schedules["joint"]["items"][:start*2])
    new_coverage=set(schedules["joint"]["sample_ids"][x["sample_index"]] for x in schedules["joint"]["items"][12000:start*2])
    selection=f"user_authorized_continuation_fixed_{locked['joint_steps']}_endpoint"
    for step in range(start+1,locked["joint_steps"]+1):
        if STOP or time.time()>=min(read_budget(ROOT)["training_deadline_unix"],locked["operational_training_deadline_unix"]):
            selection="immutable_training_deadline_or_interrupt_incomplete"
            atomic_checkpoint(last,checkpoint(arm,step-1,meta_steps,up,down,fa,optimizer,optimizer_fa,locked,selection))
            status("training_stopped_incomplete",ROOT,arm=arm,step=step-1)
            raise RuntimeError("Matched training incomplete; no further GPU updates allowed at training cutoff")
        guard("train",ROOT);tick=time.perf_counter();meta_values={k:0. for k in ("meta_inner_loss","meta_outer_raw_loss",
            "meta_outer_scaled_loss","meta_hypergradient_raw_norm","meta_hypergradient_scaled_norm","meta_outer_scale")}
        if step%10==0:
            index=meta_steps*2
            meta_values=meta_update(up,down,fa,optimizer_fa,
                [to_device(datasets["meta_inner"][i]) for i in (index,index+1)],
                [to_device(datasets["meta_outer"][i]) for i in (index,index+1)],meta_steps)
            meta_steps+=1
        batches=[to_device(datasets["joint"][i]) for i in ((step-1)*2,(step-1)*2+1)]
        coverage.update(b["sample_id"] for b in batches)
        new_coverage.update(b["sample_id"] for b in batches)
        values=joint_update(up,down,fa,optimizer,batches,step,locked["alignment_weight"])
        torch.cuda.synchronize()
        row={"step":step,"arm":arm,"unique_joint_images":len(coverage),"unique_additional_joint_images":len(new_coverage),"seconds":time.perf_counter()-tick,
             "lr_adapters":optimizer.param_groups[0]["lr"],"lr_task_tail":optimizer.param_groups[1]["lr"] if len(optimizer.param_groups)>1 else 0.,"lr_FANet":optimizer_fa.param_groups[0]["lr"],
             "alignment_weight":locked["alignment_weight"],**values,**meta_values}
        append_csv(ROOT/"downstream/logs"/f"{arm}_training.csv",row)
        if step%100==0:status("training",ROOT,arm=arm,step=step,total_steps=locked["joint_steps"],unique_joint_images=len(coverage),loss=values["joint_loss"])
        if step%500==0 or step==locked["joint_steps"]:
            check_frozen(up,down,fa,frozen)
            atomic_checkpoint(last,checkpoint(arm,step,meta_steps,up,down,fa,optimizer,optimizer_fa,locked,selection))
            retention_probe(up,down,datasets["joint"],arm,step)
            atomic_checkpoint(target/f"step_{step}.pt",checkpoint(arm,step,meta_steps,up,down,fa,optimizer,optimizer_fa,locked,selection))
            if step%1000==0:fixed_validation_probe(up,down,locked,step,arm)
        del batches
    assert len(coverage)==5440
    assert len(new_coverage)==5440
    check_frozen(up,down,fa,frozen)
    atomic_checkpoint(final,checkpoint(arm,locked["joint_steps"],meta_steps,up,down,fa,optimizer,optimizer_fa,locked,selection))
    atomic_json(target/"training_complete.json",{"arm":arm,"joint_steps":locked["joint_steps"],"meta_updates":meta_steps,
        "resume_parent_final_sha256":locked["resume_parent_final_sha256"],"additional_joint_steps":3000,
        "unique_joint_images":len(coverage),"unique_additional_joint_images":len(new_coverage),"protocol_id":locked["protocol_id"],"final_sha256":sha(final),
        "frozen_parameters_unchanged":True,"selection":selection})
    status("arm_training_complete",ROOT,arm=arm,step=locked["joint_steps"],meta_updates=meta_steps,source_unchanged=True)
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
    if sha(ROOT/'weights/resume_parent_6000.pt')!=locked['resume_parent_final_sha256']:
        raise RuntimeError("Actual shared6000 source changed after locking")
    if locked["arms"] != ["frozen","partial"] or ARGS.arm not in ("frozen","partial","both"):
        raise RuntimeError("Only paired frozen/partial collaboration is authorized")
    for name,digest in locked["code_sha256"].items():
        if sha(ROOT/name)!=digest:raise RuntimeError(f"Frozen code changed: {name}")
    if sha(locked["manifest"])!=locked["manifest_sha256"]:raise RuntimeError("Dataset manifest changed after lock")
    for name,digest in locked["schedule_hashes"].items():
        if sha(ROOT/"config"/f"{name}_schedule.json")!=digest:raise RuntimeError("Locked sample/crop schedule changed")
    for arm in locked["arms"] if ARGS.arm=="both" else [ARGS.arm]:
        train_arm(arm,locked)
    status("training_complete",ROOT,arms=locked["arms"],joint_steps_per_arm=locked["joint_steps"],validation_test_used_for_training=False)


if __name__=="__main__":
    main()
