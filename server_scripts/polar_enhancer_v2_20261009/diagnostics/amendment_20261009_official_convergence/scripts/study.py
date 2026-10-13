"""One formal experiment: new polarization correction + detached quality feedback."""
import csv,gc,importlib.util,json,math,sys,time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from common import ROOT,seed,sha,json_save,save,rng_state,restore_rng,guard_gpu,status
from enhancer import PolarEnhancer,JointEnhancer
from quality_metrics import QualityCollector

spec=importlib.util.spec_from_file_location('joint_reference',ROOT/'scripts/reference/run_experiment_snapshot.py')
base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
OUT=ROOT/'downstream'

class WeightedPhase(nn.Module):
    def __init__(self,module):super().__init__();self.module=module
    def forward(self,*args,**kwargs):return .1*self.module(*args,**kwargs)

def components(cfg,device,pretrained=True):
    base.add_source_path(cfg)
    from rgb_final_confirmation.polarfree import OfficialPolarFreeTask
    from rgb_final_confirmation.losses import build_official_task_loss
    raw=base.build_base_config(cfg)
    task=OfficialPolarFreeTask(raw,device)
    source=Path(cfg['paths']['source_checkpoint'])
    assert sha(source)==cfg['source_sha256']
    ck=torch.load(source,map_location='cpu',weights_only=False)
    task.net_g.load_state_dict(ck['tasknet_net_g'],strict=True);del ck
    task.model.requires_grad_(False);task.eval()
    criterion=build_official_task_loss(raw).to(device).requires_grad_(False).eval()
    criterion.phase=WeightedPhase(criterion.phase)
    seed(cfg['seed'])
    core=PolarEnhancer()
    if pretrained:
        state=torch.load(ROOT/'pretraining/checkpoints/selected.pt',map_location='cpu',weights_only=False)
        core.load_state_dict(state['enhancer'],strict=True)
    core.downstream_parameters();core.activation_checkpoint=True
    model=JointEnhancer(core).to(device)
    model.protocol_id=cfg.get('protocol_id','preflight');model.tau=cfg.get('confidence',{}).get('tau',.03)
    model.activation_checkpoint=True
    return model,task,criterion,raw

@torch.no_grad()
def polarization_summary(polar):
    # Observation descriptors only. This quality-head input is not a confidence label.
    from rgb_final_confirmation.polarfree import opencv_rgb_to_gray_ste
    g=[opencv_rgb_to_gray_ste(x.float()) for x in polar.chunk(4,1)]
    s0=(g[0]+g[1]+g[2]+g[3])/2;q=g[0]-g[2];u=g[1]-g[3]
    dolp=(torch.sqrt(q*q+u*u+1e-12)/s0.clamp_min(1e-4)).clamp(0,1)
    usable=(s0>2/255).float()*(torch.stack(g).amax(0)<253/255).float()
    return torch.cat((s0/2,q,u,dolp,usable),1)

def task_forward(model,task,p,image):
    if torch.is_grad_enabled() and model.activation_checkpoint:
        return checkpoint(task,p.float(),image.float(),None,use_reentrant=False,preserve_rng_state=True)
    return task(p.float(),image.float(),None)

def enhance(model,*args,project=False):
    device=args[0].device
    # Keep PolarFree and perceptual/Fourier losses FP32. AMP only the enhancer.
    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda' and model.training):
        return model(*args,project=project)

def recurrent_forward(model,task,batch,need_alignment=False):
    p0,image=batch['polar'],batch['official_rgb']
    first=enhance(model,p0,p0,image,project=need_alignment)
    if not torch.is_grad_enabled():
        first.raw_features.clear()
        if p0.is_cuda:torch.cuda.empty_cache()
    cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all() if p0.is_cuda else None
    t1,f1=task_forward(model,task,first.refined_polar,image)
    b=model.confidence_head(image,t1,polarization_summary(p0),f1)
    if not torch.is_grad_enabled() and not need_alignment:
        # Full-resolution T1 features are needed only by H, not the adapters.
        f1.pop('T1',None)
    c=torch.exp(-b/model.tau).detach()
    gamma=min(max(model.step,0)/500,1) if model.training else 1.
    effective=1-gamma*(1-c)
    gate=F.interpolate(effective,size=p0.shape[-2:],mode='bilinear',align_corners=False)
    # Raw observations anchor both rounds; current-P0 and T/F remain differentiable.
    second=enhance(model,p0,first.refined_polar,image,t1.clamp(0,1),f1,gate,project=False)
    if not torch.is_grad_enabled():
        second.raw_features.clear()
        if not need_alignment:f1.clear()
        if p0.is_cuda:torch.cuda.empty_cache()
    devices=list(range(torch.cuda.device_count())) if cuda is not None else []
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(cpu)
        if cuda is not None:torch.cuda.set_rng_state_all(cuda)
        t2,_=task_forward(model,task,second.refined_polar,image)
    second.first=first;second.t1=t1;second.t2=t2;second.first_task_features=f1
    second.error_scale=b;second.confidence_map=c;second.effective_confidence=effective
    if not torch.is_grad_enabled():
        first.raw_features.clear();second.raw_features.clear()
        if not need_alignment:f1.clear()
    return second

def losses(model,task,criterion,batch):
    out=recurrent_forward(model,task,batch,True)
    l1=criterion(out.t1.float(),batch['target']);l2=criterion(out.t2.float(),batch['target'])
    main=l1/3+2*l2/3
    align=base.feature_alignment_loss(out.first.projected_features,out.first_task_features)
    e=F.avg_pool2d((out.t1.detach().clamp(0,1)-batch['target']).abs().mean(1,keepdim=True),4)
    quality=(e/out.error_scale+out.error_scale.log()).mean()
    if not all(torch.isfinite(x) for x in (main,align,quality)):raise FloatingPointError('Nonfinite joint loss')
    return out,main,align,quality

def microbatches(raw):
    for i in range(raw['polar'].shape[0]):
        yield {k:v[i:i+1] for k,v in raw.items()}

def configure_step(task,step):
    task.net_g.requires_grad_(step>500)
    allowed={id(p) for p in task.net_g.parameters()}
    assert not any(p.requires_grad and id(p) not in allowed for p in task.parameters())

def optimizer_for(model,task):
    main=[];adapters=[]
    for n,p in model.named_parameters():
        if not p.requires_grad:continue
        (adapters if n.startswith(('r2p.adapters.','projections.','confidence_head.')) else main).append(p)
    return torch.optim.Adam([dict(params=main,lr=2e-5,name='correction'),
                             dict(params=adapters,lr=1e-4,name='interfaces'),
                             dict(params=task.net_g.parameters(),lr=1e-6,name='task')])

def benchmark(cfg,device):
    model,task,criterion,rawcfg=components(cfg,device,pretrained=False)
    model.train();model.step=501;configure_step(task,501)
    optimizer=optimizer_for(model,task);scaler=torch.amp.GradScaler('cuda',enabled=False)
    data=iter(base.scheduled_loader(cfg,rawcfg,4));times=[]
    torch.cuda.reset_peak_memory_stats()
    for i in range(4):
        guard_gpu();tick=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        for raw in microbatches(next(data)):
            batch=base.unpack_batch(raw,device);base.set_step_seed(cfg['seed'],'train',i,batch['sample_id'])
            out,main,align,quality=losses(model,task,criterion,batch)
            scaler.scale((main+.03*align+quality)/2).backward();del out
        scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(task.net_g.parameters()),1,error_if_nonfinite=True)
        scaler.step(optimizer);scaler.update();torch.cuda.synchronize();times.append(time.perf_counter()-tick)
    peak=torch.cuda.max_memory_allocated()/2**20
    del optimizer,scaler,data;model.eval();gc.collect();torch.cuda.empty_cache()
    from rgb_final_confirmation.data import load_manifest,select_rows,PairedBranchDataset
    # Largest validation image by known manifest dimensions, never inspect test metrics.
    rows=select_rows(load_manifest(cfg['paths']['manifest']),'validation')
    # Manifest lacks dimensions in some versions: decode dimensions without inference.
    dataset=PairedBranchDataset(rows,group='A1',crop_size=None)
    from PIL import Image
    sizes=[]
    for i,row in enumerate(rows):
        with Image.open(row.official_rgb) as img:sizes.append((img.width*img.height,i))
    selected=max(sizes)[1];sample=dataset[selected]
    batch={k:v.unsqueeze(0).to(device) if torch.is_tensor(v) else [v] for k,v in sample.items()}
    guard_gpu();tick=time.perf_counter()
    try:
        with torch.no_grad():out=recurrent_forward(model,task,batch)
        torch.cuda.synchronize();seconds=time.perf_counter()-tick
        report=dict(step_seconds=float(np.median(times[1:])),full_image_seconds=seconds,
                    full_image_shape=list(batch['polar'].shape),sample_id=batch['sample_id'],
                    train_peak_vram_mb=peak,eval_peak_vram_mb=torch.cuda.max_memory_allocated()/2**20,
                    full_forward=True,microbatch=1,effective_batch=2)
        del out
    except torch.cuda.OutOfMemoryError as exc:
        report=dict(step_seconds=float(np.median(times[1:])),full_image_seconds=None,
                    full_image_shape=list(batch['polar'].shape),full_forward_gpu=False,error=str(exc))
        raise RuntimeError('Full-frame evaluation does not fit GPU; measure CPU fallback before locking budget') from exc
    json_save(ROOT/'diagnostics/downstream_benchmark.json',report)
    del model,task,criterion,batch;gc.collect();torch.cuda.empty_cache();return report

@torch.no_grad()
def calibrate_quality(model,task,cfg,rawcfg,device):
    model.eval();values=[];ids=[]
    for index,raw in enumerate(base.scheduled_loader(cfg,rawcfg,128)):
        guard_gpu()
        for raw1 in microbatches(raw):
            batch=base.unpack_batch(raw1,device)
            base.set_step_seed(cfg['seed'],'quality_calibration',index,batch['sample_id'])
            first=enhance(model,batch['polar'],batch['polar'],batch['official_rgb'])
            t1,_=task_forward(model,task,first.refined_polar,batch['official_rgb'])
            error=F.avg_pool2d((t1.clamp(0,1)-batch['target']).abs().mean(1,keepdim=True),4)
            values.append(error.cpu().flatten());ids+=batch['sample_id']
    pooled=torch.cat(values);initial=max(float(pooled.median()),.002)
    tau=initial/math.log(2)
    model.confidence_head.initialize(initial);model.tau=tau
    result=dict(training_crops=len(ids),sample_ids=ids,initial_error=initial,tau=tau,
                rule='median local T1 error; tau=median/log(2); initial b floor .002',
                target='4x4 pooled RGB MAE',test_used=False)
    assert len(ids)==256
    json_save(OUT/'config/quality_calibration.json',result);return result

def calibrate_alignment(model,task,criterion,cfg,rawcfg,device):
    model.train();model.step=0;configure_step(task,0);pairs=[]
    params=tuple(p for n,p in model.r2p.named_parameters() if p.requires_grad and not n.startswith('adapters.'))
    for i,raw in enumerate(base.scheduled_loader(cfg,rawcfg,4)):
        guard_gpu()
        # Effective-batch calibration follows the same two scheduled crops.
        task_sq=align_sq=0.
        for micro in microbatches(raw):
            batch=base.unpack_batch(micro,device);base.set_step_seed(cfg['seed'],'alignment_calibration',i,batch['sample_id'])
            out,main,align,_=losses(model,task,criterion,batch)
            task_sq+=base.autograd_norm(main,params)**2
            align_sq+=base.autograd_norm(align,params)**2;del out
        pairs.append(dict(task_norm=math.sqrt(task_sq),alignment_norm=math.sqrt(align_sq)))
    coefficient=float(np.clip(.1*np.mean([x['task_norm'] for x in pairs])/max(np.mean([x['alignment_norm'] for x in pairs]),1e-12),1e-4,1))
    result=dict(lambda_alignment=coefficient,pairs=pairs,training_crops=8,rule='0.1 ratio of train-only core gradient norms; clamp [1e-4,1]')
    json_save(OUT/'config/alignment_calibration.json',result);return coefficient

def initialize(cfg,device):
    model,task,criterion,rawcfg=components(cfg,device)
    ck=torch.load(ROOT/'pretraining/checkpoints/selected.pt',map_location='cpu',weights_only=False)
    mismatches=[n for n,v in model.r2p.state_dict().items() if not torch.equal(v.cpu(),ck['enhancer'][n])]
    assert not mismatches
    json_save(OUT/'config/weight_migration.json',dict(strict=True,keys=len(ck['enhancer']),mismatches=mismatches,
              enhancer_sha256=sha(ROOT/'pretraining/checkpoints/selected.pt'),task_source_sha256=cfg['source_sha256'],
              old_enhancer_inherited=False,sr_branch=False))
    quality=calibrate_quality(model,task,cfg,rawcfg,device)
    cfg['confidence']=quality;cfg['lambda_alignment']=calibrate_alignment(model,task,criterion,cfg,rawcfg,device)
    model.protocol_id=cfg['protocol_id'];model.eval()
    save(OUT/'checkpoints/initial.pt',dict(step=0,r2p_and_projections=model.state_dict(),tasknet_net_g=task.net_g.state_dict(),
                                         tau=model.tau,protocol_id=cfg['protocol_id'],rng=rng_state()))
    json_save(ROOT/'config/locked_run.json',cfg)
    # Full-gradient checks use a disposable model state, never alter formal initialization.
    raw=next(iter(base.scheduled_loader(cfg,rawcfg,1)));batch=base.unpack_batch(next(microbatches(raw)),device)
    model.train();model.step=501;configure_step(task,501)
    count={'E':0,'D':0}
    hooks=[model.r2p.register_forward_hook(lambda *args:count.__setitem__('E',count['E']+1)),
           task.register_forward_hook(lambda *args:count.__setitem__('D',count['D']+1))]
    out,main,align,q=losses(model,task,criterion,batch)
    # The initial zero adapters block feedback paths until their output layer learns.
    main.backward(retain_graph=True)
    main_head_grad=any(p.grad is not None for p in model.confidence_head.parameters())
    assert not main_head_grad
    for p in list(model.parameters())+list(task.parameters()):p.grad=None
    q.backward()
    assert not any(p.grad is not None for p in model.r2p.parameters())
    assert not any(p.grad is not None for p in task.parameters())
    for h in hooks:h.remove()
    # Checkpoint recomputation adds execution calls; count two semantic forwards before backward separately.
    json_save(ROOT/'diagnostics/joint_gradient_check.json',dict(semantic_rounds=2,checkpoint_execution_calls=count,
              quality_detached_from_E_D=True,main_detached_from_quality_head=True,
              task_prior_frozen=True,source_migration_strict=True))
    del model,task,criterion,out,ck;gc.collect();torch.cuda.empty_cache();return cfg

def train(cfg,device):
    model,task,criterion,rawcfg=components(cfg,device)
    last=OUT/'checkpoints/last.pt';path=last if last.exists() else OUT/'checkpoints/initial.pt'
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['protocol_id']==cfg['protocol_id']
    model.load_state_dict(state['r2p_and_projections'],strict=True);task.net_g.load_state_dict(state['tasknet_net_g'],strict=True)
    model.tau=state['tau'];start=state['step']
    optimizer=optimizer_for(model,task);scaler=torch.amp.GradScaler('cuda',enabled=False)
    if 'optimizer' in state:optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
    restore_rng(state['rng']);del state
    total=cfg['downstream']['steps'];iterator=iter(base.scheduled_loader(cfg,rawcfg,total,start));step=start
    while step<total and time.time()<cfg['downstream']['deadline_unix']:
        guard_gpu();step+=1;tick=time.perf_counter();model.train();model.step=step;configure_step(task,step)
        optimizer.zero_grad(set_to_none=True);metrics=np.zeros(4);ids=[]
        for raw in microbatches(next(iterator)):
            batch=base.unpack_batch(raw,device);ids+=batch['sample_id']
            base.set_step_seed(cfg['seed'],'training',step,batch['sample_id'])
            out,main,align,quality=losses(model,task,criterion,batch)
            loss=(main+cfg['lambda_alignment']*align+quality)/2
            scaler.scale(loss).backward()
            metrics+=np.array([float(x.detach()) for x in (loss,main/2,align/2,quality/2)]);del out
        scaler.unscale_(optimizer)
        norm=float(torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(task.net_g.parameters()),1,error_if_nonfinite=True))
        scaler.step(optimizer);scaler.update()
        fields=dict(step=step,loss=metrics[0],task=metrics[1],alignment=metrics[2],quality=metrics[3],
                    grad_norm=norm,seconds=time.perf_counter()-tick,sample_ids=';'.join(ids),tasknet_trainable=step>500)
        log=OUT/'logs/train.csv';log.parent.mkdir(parents=True,exist_ok=True);exists=log.exists()
        with log.open('a',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields)
            if not exists:w.writeheader()
            w.writerow(fields)
        if step%50==0:print(json.dumps(fields),flush=True)
        stop=time.time()>=cfg['downstream']['deadline_unix']
        if step%500==0 or step==total or stop:
            state=dict(step=step,r2p_and_projections=model.state_dict(),tasknet_net_g=task.net_g.state_dict(),
                       optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),rng=rng_state(),tau=model.tau,
                       protocol_id=cfg['protocol_id'],lambda_alignment=cfg['lambda_alignment'],selection='fixed steps or time only')
            save(last,state);save(OUT/'checkpoints'/f'step_{step}.pt',state)
            status('downstream_training',step=step,target_steps=total)
            if stop:break
    # Selection is the latest predeclared budget/time checkpoint, never test performance.
    if not last.exists():raise RuntimeError('No downstream training checkpoint')
    chosen=torch.load(last,map_location='cpu',weights_only=False)
    save(OUT/'checkpoints/final.pt',chosen)
    json_save(OUT/'selection.json',dict(step=chosen['step'],sha256=sha(OUT/'checkpoints/final.pt'),
              planned_steps=total,rule='planned final step or latest time cutoff save',test_used=False))
    del model,task,criterion,optimizer,scaler,chosen;gc.collect();torch.cuda.empty_cache();return step

@torch.no_grad()
def evaluate(cfg,device,split):
    from rgb_final_confirmation.evaluation import psnr,ssim
    model,task,criterion,_=components(cfg,device);del criterion
    path=OUT/'checkpoints/final.pt';state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['protocol_id']==cfg['protocol_id']
    model.load_state_dict(state['r2p_and_projections'],strict=True);task.net_g.load_state_dict(state['tasknet_net_g'],strict=True)
    model.tau=state['tau'];model.step=state['step'];model.eval();task.eval();rows=[]
    expected=872 if split=='validation' else 188
    collector=QualityCollector(OUT,expected,model.tau)
    for i,raw in enumerate(base.split_loader(cfg,split)):
        if device.type=='cuda':guard_gpu()
        batch=base.unpack_batch(raw,device);base.set_step_seed(cfg['seed'],'validation',i,batch['sample_id'])
        out=recurrent_forward(model,task,batch)
        r=dict(sample_id=batch['sample_id'][0],scene=batch['scene'][0],
               round1=dict(psnr=psnr(out.t1,batch['target']),ssim=ssim(out.t1,batch['target'])),
               round2=dict(psnr=psnr(out.t2,batch['target']),ssim=ssim(out.t2,batch['target'])))
        r['iteration_gain']={k:r['round2'][k]-r['round1'][k] for k in ('psnr','ssim')}
        rows.append(r);collector.observe(out,batch);del out
    assert len(rows)==len({r['sample_id'] for r in rows})==expected
    scenes=[]
    for name in sorted({r['scene'] for r in rows}):
        members=[r for r in rows if r['scene']==name]
        scene=dict(scene=name,frames=len(members))
        for key in ('round1','round2','iteration_gain'):
            scene[key]={m:float(np.mean([r[key][m] for r in members])) for m in ('psnr','ssim')}
        scenes.append(scene)
    result=dict(split=split,frames=len(rows),checkpoint_step=state['step'],checkpoint_sha256=sha(path),
                protocol_id=cfg['protocol_id'],per_image=rows,per_scene=scenes,confidence=collector.finish(),
                original_size=True,tiling=False,device=str(device),test_used_for_selection=False)
    for key in ('round1','round2','iteration_gain'):
        result[key]={f'image_mean_{m}':float(np.mean([r[key][m] for r in rows])) for m in ('psnr','ssim')}
        result[key].update({f'scene_macro_{m}':float(np.mean([r[key][m] for r in scenes])) for m in ('psnr','ssim')})
    json_save(OUT/'metrics'/f'final_{split}.json',result)
    del model,task,state;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return result
