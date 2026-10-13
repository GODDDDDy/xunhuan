"""One formal experiment: new polarization correction + detached quality feedback."""
import csv,gc,hashlib,importlib.util,json,math,sys,time
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
from acceptance import LocalAcceptance,AcceptanceCollector,local_error,labels,judgement_loss

spec=importlib.util.spec_from_file_location('joint_reference',ROOT/'scripts/reference/run_experiment_snapshot.py')
base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
OUT=ROOT/'downstream'

def official_kind(cfg):
    return cfg.get('task_initialization',{}).get('kind') in ('official_polarfree','verified_previous_final')

class WeightedPhase(nn.Module):
    def __init__(self,module):super().__init__();self.module=module
    def forward(self,*args,**kwargs):return .1*self.module(*args,**kwargs)

def components(cfg,device,pretrained=True,with_loss=True):
    base.add_source_path(cfg)
    from rgb_final_confirmation.polarfree import OfficialPolarFreeTask
    from rgb_final_confirmation.losses import build_official_task_loss
    raw=base.build_base_config(cfg)
    source=Path(cfg['paths']['source_checkpoint'])
    assert sha(source)==cfg['source_sha256']
    official=cfg.get('task_initialization',{}).get('kind') in ('official_polarfree','verified_previous_final')
    if official:
        assert source.resolve()==(Path(raw.raw['official']['checkpoint_dir'])/'net_g_latest.pth').resolve()
        for name,digest in cfg['task_initialization']['official_sha256'].items():
            assert sha(Path(raw.raw['official']['checkpoint_dir'])/name)==digest
    task=OfficialPolarFreeTask(raw,device)
    ck=torch.load(source,map_location='cpu',weights_only=False)
    if official:
        weights=ck['params'];live=task.net_g.state_dict()
        assert live.keys()==weights.keys()
        assert all(torch.equal(v.cpu(),weights[n]) for n,v in live.items())
        task.initialization_audit={'kind':'official_polarfree','strict':True,'source_sha256':cfg['source_sha256'],
                                   'all_net_g_tensors_equal_official':True,'prior_sha256':cfg['task_initialization']['official_sha256'],
                                   'historical_joint_weights_overwrite':False}
    else:
        task.net_g.load_state_dict(ck['tasknet_net_g'],strict=True)
        task.initialization_audit={'kind':'historical_joint','source_sha256':cfg['source_sha256']}
    del ck
    task.model.requires_grad_(False);task.eval()
    criterion=None
    if with_loss:
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
    model.good_threshold=cfg.get('local_acceptance',{}).get('good_threshold',.01)
    model.benefit_margin=cfg.get('local_acceptance',{}).get('benefit_margin',.001)
    model.reuse_fixed_first_round=cfg.get('reuse',{}).get('enabled',False)
    if model.reuse_fixed_first_round:
        path=Path(cfg['reuse']['checkpoint']);assert sha(path)==cfg['reuse']['sha256']
        previous=torch.load(path,map_location='cpu',weights_only=False)
        transfer=model.load_state_dict(previous['r2p_and_projections'],strict=False)
        assert not transfer.unexpected_keys
        assert all(n.startswith('update_gate.') for n in transfer.missing_keys)
        assert set(transfer.missing_keys)=={n for n in model.state_dict() if n.startswith('update_gate.')}
        task.net_g.load_state_dict(previous['tasknet_net_g'],strict=True)
        assert all(torch.equal(v.cpu(),previous['r2p_and_projections'][n])
                   for n,v in model.state_dict().items() if not n.startswith('update_gate.'))
        assert all(torch.equal(v.cpu(),previous['tasknet_net_g'][n]) for n,v in task.net_g.state_dict().items())
        model.tau=previous['tau']
        model.initialization_audit=dict(source_sha256=sha(path),source_step=previous['step'],
            inherited_keys=len(previous['r2p_and_projections']),new_keys=transfer.missing_keys,
            all_existing_tensors_equal=True,source_protocol_id=previous['protocol_id'])
        task.initialization_audit.update(kind='verified_previous_final',
            starting_weights_equal_previous_net_g=True,warmstart_sha256=sha(path),
            source_originally_official=True,all_net_g_tensors_equal_official=False,
            historical_joint_weights_overwrite=False)
        model.r2p.requires_grad_(False).eval();model.projections.requires_grad_(False)
        model.confidence_head.requires_grad_(False);task.net_g.requires_grad_(False)
        del previous
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

def task_forward(model,task,p,image,anchor,context=None,polar_context=None):
    from bridge import anchored_task_inputs
    def forward(p,image,anchor):
        inputs=anchored_task_inputs(p.float(),anchor.float(),image.float())
        if context is None:return task.forward_inputs(inputs,None)
        from feedback import second_task_forward
        return second_task_forward(task,model.feedback,inputs,context,polar_context)
    if torch.is_grad_enabled() and model.activation_checkpoint:
        return checkpoint(forward,p,image,anchor,use_reentrant=False,preserve_rng_state=True)
    return forward(p,image,anchor)


def enhance(model,*args,project=False,correction_state=None):
    device=args[0].device
    # Keep PolarFree and perceptual/Fourier losses FP32. AMP only the enhancer.
    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda' and model.training and not model.reuse_fixed_first_round):
        return model(*args,project=project,correction_state=correction_state)

def recurrent_forward(model,task,batch,need_alignment=False,diagnostic=None):
    p0,image=batch['polar'],batch['official_rgb']
    first=enhance(model,p0,p0,image,project=need_alignment)
    cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all() if p0.is_cuda else None
    t1,f1=task_forward(model,task,first.refined_polar,image,p0)
    b=model.confidence_head(image,t1,polarization_summary(p0),f1)
    c=torch.exp(-b/model.tau).detach()
    gamma=min(max(model.step,0)/500,1) if model.training and not model.reuse_fixed_first_round else 1.
    effective=1-gamma*(1-c)
    gate=F.interpolate(effective,size=p0.shape[-2:],mode='bilinear',align_corners=False)
    if diagnostic=='confidence_zero':gate=torch.zeros_like(gate)
    if diagnostic=='confidence_one':gate=torch.ones_like(gate)
    context=model.feedback.make_context(image,t1,polarization_summary(p0),f1,first.raw_features,gate)
    if diagnostic=='no_feedback':
        context['reference']={k:torch.zeros_like(v) for k,v in context['reference'].items()}
        context['state']={k:torch.zeros_like(v) for k,v in context['state'].items()}
    if not torch.is_grad_enabled():
        first.raw_features.clear()
        f1.pop('T1',None)
        if p0.is_cuda:torch.cuda.empty_cache()
    # Raw observations anchor both rounds; current-P0 and T/F remain differentiable.
    second=enhance(model,p0,first.refined_polar,image,
                   None if diagnostic=='no_feedback' else t1.clamp(0,1),
                   None if diagnostic=='no_feedback' else f1,gate,project=False,
                   correction_state=context['state'])
    polar_context=model.feedback.polar_context(second.raw_features)
    if diagnostic=='no_polar_correction':
        # Diagnostic removes learned polarization corrections/features; it cannot
        # erase polarization already present in T1, so it is not a causal ablation.
        second.refined_polar=p0
        polar_context={k:torch.zeros_like(v) for k,v in polar_context.items()}
    if not torch.is_grad_enabled():
        second.raw_features.clear()
        if not need_alignment:f1.clear()
        if p0.is_cuda:torch.cuda.empty_cache()
    devices=list(range(torch.cuda.device_count())) if cuda is not None else []
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(cpu)
        if cuda is not None:torch.cuda.set_rng_state_all(cuda)
        candidate,f2=task_forward(model,task,second.refined_polar,image,p0,context,polar_context)
    ungated_t2,increment=model.feedback.correct(f2,candidate,image,t1,context,polar_context)
    judgement=model.update_gate(image,polarization_summary(p0),t1,ungated_t2,b,c,
                                f2['T1'],context['state']['level1'])
    t2,update_mask=LocalAcceptance.apply(t1,increment,judgement,model.step,model.training,override=diagnostic)
    second.ungated_t2=ungated_t2;second.acceptance=judgement;second.update_mask=update_mask
    second.first=first;second.t1=t1;second.t2=t2;second.first_task_features=f1
    second.error_scale=b;second.confidence_map=c;second.effective_confidence=effective
    second.rgb_increment=increment
    if not torch.is_grad_enabled():
        first.raw_features.clear();second.raw_features.clear()
        if not need_alignment:f1.clear()
    return second

def losses(model,task,criterion,batch):
    out=recurrent_forward(model,task,batch,True)
    l1=criterion(out.t1.float(),batch['target']);l2=criterion(out.t2.float(),batch['target'])
    main=l1/3+2*l2/3
    # Train-only, bounded error-region emphasis. No extra equivalent residual L1.
    error=(out.t1.detach()-batch['target']).abs().mean(1,keepdim=True)
    weight=(1+error/(error.mean(dim=(-2,-1),keepdim=True)+1e-4)).clamp(1,3).detach()
    correction=(weight*(out.t2-batch['target']).abs().mean(1,keepdim=True)).mean()/weight.mean()
    main=main+.1*correction
    align=base.feature_alignment_loss(out.first.projected_features,out.first_task_features)
    e=F.avg_pool2d((out.t1.detach().clamp(0,1)-batch['target']).abs().mean(1,keepdim=True),4)
    quality=(e/out.error_scale+out.error_scale.log()).mean()
    need,benefit,e1,ep=labels(out.t1,out.ungated_t2,batch['target'],model.good_threshold,model.benefit_margin)
    # Candidate supervision remains active when the final gate rejects an update.
    candidate_loss=local_error(out.ungated_t2,batch['target']).mean()
    good=(e1<=model.good_threshold).float().detach()
    update=F.avg_pool2d((out.ungated_t2-out.t1.detach()).abs().mean(1,keepdim=True),4)
    preservation=(good*update).sum()/good.sum().clamp_min(1)
    main=main+.25*candidate_loss+.1*preservation
    judge=judgement_loss(out.acceptance['logits'],need,benefit)
    quality=quality+.05*judge
    out.judge_loss=judge.detach();out.candidate_loss=candidate_loss.detach()
    out.preservation_loss=preservation.detach()
    if not all(torch.isfinite(x) for x in (main,align,quality)):raise FloatingPointError('Nonfinite joint loss')
    return out,main,align,quality

def microbatches(raw):
    for i in range(raw['polar'].shape[0]):
        yield {k:v[i:i+1] for k,v in raw.items()}

def configure_step(task,step,model=None):
    if model is not None and model.reuse_fixed_first_round:
        model.r2p.requires_grad_(False).eval();model.projections.requires_grad_(False)
        model.confidence_head.requires_grad_(False);task.net_g.requires_grad_(False).eval()
        model.feedback.requires_grad_(step>500);model.update_gate.requires_grad_(True)
        return
    task.net_g.requires_grad_(step>500)
    if model is not None:
        model.r2p.downstream_parameters()
        if 0<step<=500:model.r2p.requires_grad_(False)
    allowed={id(p) for p in task.net_g.parameters()}
    assert not any(p.requires_grad and id(p) not in allowed for p in task.parameters())

def optimizer_for(model,task,cfg=None):
    if model.reuse_fixed_first_round:
        return torch.optim.Adam([
            dict(params=model.feedback.parameters(),lr=1e-5,name='interfaces'),
            dict(params=model.update_gate.parameters(),lr=1e-4,name='judgement')])
    main=[];adapters=[]
    for n,p in model.named_parameters():
        if not p.requires_grad:continue
        (adapters if n.startswith(('r2p.adapters.','projections.','confidence_head.','feedback.','update_gate.')) else main).append(p)
    rates=(cfg or {}).get('optimization',{})
    return torch.optim.Adam([dict(params=main,lr=rates.get('correction_lr',2e-5),name='correction'),
                             dict(params=adapters,lr=rates.get('interfaces_lr',1e-4),name='interfaces'),
                             dict(params=task.net_g.parameters(),lr=rates.get('task_lr',1e-6),name='task')])

def apply_learning_rates(optimizer,cfg,step):
    rates=cfg.get('optimization',{})
    if cfg.get('reuse',{}).get('enabled',False):
        values=dict(correction=0.,task=0.,interfaces=1e-5*(step>500),judgement=1e-4)
        for group in optimizer.param_groups:group['lr']=values[group['name']]
        return values
    ramp=min(max((step-500)/max(rates.get('task_unfreeze_warmup',200),1),0),1)
    values={'correction':rates.get('correction_lr',2e-5)*(step>500),'interfaces':rates.get('interfaces_lr',1e-4),
            'task':rates.get('task_lr',1e-6)*ramp}
    for group in optimizer.param_groups:group['lr']=values[group['name']]
    return values

@torch.no_grad()
def retention_probe(task,cfg,rawcfg,device,step):
    """Fixed validation inputs bypass E to monitor preservation of PolarFree itself."""
    from rgb_final_confirmation.evaluation import psnr,ssim
    before=rng_state();task.eval();rows=[]
    try:
        for i,raw in enumerate(base.proxy_loader(cfg,rawcfg)):
            if device.type=='cuda':guard_gpu()
            batch=base.unpack_batch(raw,device)
            assert batch['sample_id'][0].startswith('validation/')
            base.set_step_seed(cfg['seed'],'official_retention_probe',i,batch['sample_id'])
            t,features=task(batch['polar'].float(),batch['official_rgb'].float(),None)
            rows.append({'sample_id':batch['sample_id'][0],'psnr':psnr(t,batch['target']),'ssim':ssim(t,batch['target'])})
            del t,features,batch
        expected=json.loads(Path(cfg['paths']['validation_proxy']).read_text())['sample_count']
        assert len(rows)==len({x['sample_id'] for x in rows})==expected
        result={'step':step,'frames':len(rows),'per_image':rows,'psnr':float(np.mean([x['psnr'] for x in rows])),
                'ssim':float(np.mean([x['ssim'] for x in rows])),'input':'raw P0 and original RGB, enhancer bypassed',
                'original_size':True,'paired_noise':True,'test_used':False,'used_for_weight_selection':False}
        initial=OUT/'metrics/retention_step_0.json'
        if step and initial.exists():
            ref=json.loads(initial.read_text());assert [x['sample_id'] for x in rows]==[x['sample_id'] for x in ref['per_image']]
            result.update(psnr_change_from_official=result['psnr']-ref['psnr'],ssim_change_from_official=result['ssim']-ref['ssim'])
        weights=torch.load(cfg['paths']['source_checkpoint'],map_location='cpu',weights_only=False)['params']
        numerator=denominator=0.
        for n,p in task.net_g.named_parameters():
            ref=weights[n].double();numerator+=float((p.detach().cpu().double()-ref).square().sum());denominator+=float(ref.square().sum())
        result['net_g_relative_l2_drift']=math.sqrt(numerator/max(denominator,1e-30))
        if cfg.get('reuse',{}).get('enabled',False):
            previous=torch.load(cfg['reuse']['checkpoint'],map_location='cpu',weights_only=False)['tasknet_net_g']
            assert all(torch.equal(v.cpu(),previous[n]) for n,v in task.net_g.state_dict().items())
            result['all_net_g_tensors_unchanged_from_reused_source']=True
            result['reference_kind']='reused previous final; absolute official drift is inherited'
        elif step<=500:assert result['net_g_relative_l2_drift']==0.
        json_save(OUT/'metrics'/f'retention_step_{step}.json',result)
        return result
    finally:
        restore_rng(before);gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()

def benchmark(cfg,device):
    model,task,criterion,rawcfg=components(cfg,device,pretrained=True)
    model.train();model.step=501;configure_step(task,501,model)
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
            t1,_=task_forward(model,task,first.refined_polar,batch['official_rgb'],batch['polar'])
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


@torch.no_grad()
def calibrate_acceptance(model,task,cfg,rawcfg,device):
    model.eval();values=[];ids=[];before=rng_state()
    try:
        for index,raw in enumerate(base.scheduled_loader(cfg,rawcfg,128)):
            if device.type=='cuda':guard_gpu()
            for micro in microbatches(raw):
                batch=base.unpack_batch(micro,device)
                base.set_step_seed(cfg['seed'],'acceptance_calibration',index,batch['sample_id'])
                first=enhance(model,batch['polar'],batch['polar'],batch['official_rgb'])
                t1,features=task_forward(model,task,first.refined_polar,batch['official_rgb'],batch['polar'])
                values.append(local_error(t1,batch['target']).cpu().flatten());ids+=batch['sample_id']
                del first,t1,features
        pooled=torch.cat(values)
        threshold=float(torch.quantile(pooled,.25));margin=max(.05*float(pooled.median()),1e-4)
        model.good_threshold=threshold;model.benefit_margin=margin
        assert len(ids)==256 and all(x.startswith('train/') for x in ids)
        result=dict(training_crops=256,sample_ids=ids,good_threshold=threshold,benefit_margin=margin,
                    threshold_rule='initial T1 training local error 25th percentile',
                    margin_rule='max(.05*initial median local error,1e-4)',
                    local_error='4x4 RGB MAE + .25 edge MAE',test_used=False,validation_used=False,
                    gate_training_warmup=500,gate_transition=1000,decision_threshold=.5)
        json_save(OUT/'config/acceptance_calibration.json',result);return result
    finally:restore_rng(before)

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
    if model.reuse_fixed_first_round:
        previous=torch.load(cfg['reuse']['checkpoint'],map_location='cpu',weights_only=False)
        reference={n.removeprefix('r2p.'):v for n,v in previous['r2p_and_projections'].items() if n.startswith('r2p.')}
    else:reference=ck['enhancer']
    mismatches=[n for n,v in model.r2p.state_dict().items() if not torch.equal(v.cpu(),reference[n])]
    assert not mismatches
    json_save(OUT/'config/weight_migration.json',dict(strict=True,keys=len(ck['enhancer']),mismatches=mismatches,
              enhancer_sha256=sha(cfg['reuse']['checkpoint']) if model.reuse_fixed_first_round else sha(ROOT/'pretraining/checkpoints/selected.pt'),
              pretrain_lineage_sha256=sha(ROOT/'pretraining/checkpoints/selected.pt'),task_source_sha256=cfg['source_sha256'],
              effective_task_source_sha256=cfg['reuse']['sha256'] if model.reuse_fixed_first_round else cfg['source_sha256'],
              task_initialization=task.initialization_audit,
              enhancer_checkpoint_protocol_id=ck['protocol_id'],downstream_protocol_id=cfg['protocol_id'],
              old_enhancer_inherited=model.reuse_fixed_first_round,sr_branch=False,
              inherited_current_source=getattr(model,'initialization_audit',None)))
    if official_kind(cfg):retention_probe(task,cfg,rawcfg,device,0)
    quality=cfg['confidence'] if model.reuse_fixed_first_round else calibrate_quality(model,task,cfg,rawcfg,device)
    cfg['confidence']=quality;cfg['local_acceptance'].update(calibrate_acceptance(model,task,cfg,rawcfg,device))
    cfg['lambda_alignment']=0. if model.reuse_fixed_first_round else calibrate_alignment(model,task,criterion,cfg,rawcfg,device)
    model.protocol_id=cfg['protocol_id'];model.eval()
    save(OUT/'checkpoints/initial.pt',dict(step=0,r2p_and_projections=model.state_dict(),tasknet_net_g=task.net_g.state_dict(),
                                         tau=model.tau,protocol_id=cfg['protocol_id'],rng=rng_state()))
    json_save(ROOT/'config/locked_run.json',cfg)
    json_save(ROOT/'config/calibrated_lock_receipt.json',dict(protocol_id=cfg['protocol_id'],
        final_configuration_sha256=sha(ROOT/'config/locked_run.json'),calibration_sources='training only',
        threshold_and_margin_locked_before_training=True))
    system_probe(model,task,cfg,rawcfg,device,0)
    # Full-gradient checks use a disposable model state, never alter formal initialization.
    raw=next(iter(base.scheduled_loader(cfg,rawcfg,1)));batch=base.unpack_batch(next(microbatches(raw)),device)
    model.train();model.step=501;configure_step(task,501,model)
    count={'E':0,'D':0}
    hooks=[model.r2p.register_forward_hook(lambda *args:count.__setitem__('E',count['E']+1)),
           task.net_g.register_forward_hook(lambda *args:count.__setitem__('D',count['D']+1))]
    out,main,align,q=losses(model,task,criterion,batch)
    semantic_count=count.copy()
    assert semantic_count=={'E':2,'D':2}
    if not model.reuse_fixed_first_round:assert torch.equal(out.t1,out.t2), 'Zero correction head must preserve T1 exactly'
    # The initial zero adapters block feedback paths until their output layer learns.
    main.backward(retain_graph=True)
    main_head_grad=any(p.grad is not None for p in model.confidence_head.parameters())
    assert not main_head_grad
    assert not any(p.grad is not None for p in model.update_gate.parameters())
    assert model.feedback.output_head[-1].weight.grad.abs().sum()>0
    for p in list(model.parameters())+list(task.parameters()):p.grad=None
    q.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.update_gate.parameters())
    assert not any(p.grad is not None for p in model.r2p.parameters())
    assert not any(p.grad is not None for p in task.parameters())
    for h in hooks:h.remove()
    # Checkpoint recomputation adds execution calls; count two semantic forwards before backward separately.
    json_save(ROOT/'diagnostics/joint_gradient_check.json',dict(semantic_rounds=2,semantic_execution_calls=semantic_count,checkpoint_execution_calls=count,
              quality_detached_from_E_D=True,main_detached_from_quality_head=True,
              task_prior_frozen=True,source_migration_strict=True,zero_head_t2_equals_t1=not model.reuse_fixed_first_round,
              previous_correction_head_reused=model.reuse_fixed_first_round,
              main_detached_from_judgement=True,judgement_gradients_nonzero=True,
              correction_output_gradient_nonzero=True))
    del model,task,criterion,out,ck;gc.collect();torch.cuda.empty_cache();return cfg

def train(cfg,device):
    model,task,criterion,rawcfg=components(cfg,device)
    last=OUT/'checkpoints/last.pt';path=last if last.exists() else OUT/'checkpoints/initial.pt'
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['protocol_id']==cfg['protocol_id']
    model.load_state_dict(state['r2p_and_projections'],strict=True);task.net_g.load_state_dict(state['tasknet_net_g'],strict=True)
    model.tau=state['tau'];start=state['step']
    optimizer=optimizer_for(model,task,cfg);scaler=torch.amp.GradScaler('cuda',enabled=False)
    if 'optimizer' in state:optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
    restore_rng(state['rng']);del state
    total=cfg['downstream']['steps'];iterator=iter(base.scheduled_loader(cfg,rawcfg,total,start));step=start
    while step<total and time.time()<cfg['downstream']['deadline_unix']:
        guard_gpu();step+=1;tick=time.perf_counter();model.train();model.step=step;configure_step(task,step,model)
        learning_rates=apply_learning_rates(optimizer,cfg,step)
        optimizer.zero_grad(set_to_none=True);metrics=np.zeros(4);ids=[];gate_metrics=np.zeros(5)
        for raw in microbatches(next(iterator)):
            batch=base.unpack_batch(raw,device);ids+=batch['sample_id']
            base.set_step_seed(cfg['seed'],'training',step,batch['sample_id'])
            out,main,align,quality=losses(model,task,criterion,batch)
            loss=(main+cfg['lambda_alignment']*align+quality)/2
            scaler.scale(loss).backward()
            gate_metrics+=np.array([float(x) for x in (out.judge_loss,out.candidate_loss,out.preservation_loss,
                                                     out.update_mask.mean().detach(),(out.update_mask==0).float().mean())])/2
            metrics+=np.array([float(x.detach()) for x in (loss,main/2,align/2,quality/2)]);del out
        scaler.unscale_(optimizer)
        norm=float(torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(task.net_g.parameters()),1,error_if_nonfinite=True))
        scaler.step(optimizer);scaler.update()
        fields=dict(step=step,loss=metrics[0],task=metrics[1],alignment=metrics[2],quality=metrics[3],
                    grad_norm=norm,seconds=time.perf_counter()-tick,sample_ids=';'.join(ids),tasknet_trainable=any(p.requires_grad for p in task.net_g.parameters()),
                    correction_lr=learning_rates['correction'],interfaces_lr=learning_rates['interfaces'],task_lr=learning_rates['task'])
        fields.update(judge_loss=gate_metrics[0],candidate_loss=gate_metrics[1],preservation_loss=gate_metrics[2],
                      update_mask_mean=gate_metrics[3],protected_pixel_fraction=gate_metrics[4])
        fields.update(judgement_lr=learning_rates.get('judgement',learning_rates['interfaces']),
                      reused_first_round_frozen=model.reuse_fixed_first_round)
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
        if official_kind(cfg) and (step==500 or step%1000==0 or step==total):
            retention_probe(task,cfg,rawcfg,device,step)
            system_probe(model,task,cfg,rawcfg,device,step)
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
    model,task,criterion,_=components(cfg,device,with_loss=False);del criterion
    path=OUT/'checkpoints/final.pt';state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['protocol_id']==cfg['protocol_id']
    model.load_state_dict(state['r2p_and_projections'],strict=True);task.net_g.load_state_dict(state['tasknet_net_g'],strict=True)
    model.tau=state['tau'];model.step=state['step'];model.eval();task.eval();rows=[]
    expected=872 if split=='validation' else 188
    collector=QualityCollector(OUT,expected,model.tau)
    acceptance=AcceptanceCollector(OUT,split,model.good_threshold,model.benefit_margin)
    for i,raw in enumerate(base.split_loader(cfg,split)):
        if device.type=='cuda':guard_gpu()
        batch=base.unpack_batch(raw,device);base.set_step_seed(cfg['seed'],'validation',i,batch['sample_id'])
        out=recurrent_forward(model,task,batch)
        r=dict(sample_id=batch['sample_id'][0],scene=batch['scene'][0],
               round1=dict(psnr=psnr(out.t1,batch['target']),ssim=ssim(out.t1,batch['target'])),
               round2=dict(psnr=psnr(out.t2,batch['target']),ssim=ssim(out.t2,batch['target'])),
               ungated_candidate=dict(psnr=psnr(out.ungated_t2,batch['target']),ssim=ssim(out.ungated_t2,batch['target'])))
        r['iteration_gain']={k:r['round2'][k]-r['round1'][k] for k in ('psnr','ssim')}
        rows.append(r);collector.observe(out,batch);acceptance.observe(out,batch);del out
    assert len(rows)==len({r['sample_id'] for r in rows})==expected
    scenes=[]
    for name in sorted({r['scene'] for r in rows}):
        members=[r for r in rows if r['scene']==name]
        scene=dict(scene=name,frames=len(members))
        for key in ('round1','round2','iteration_gain','ungated_candidate'):
            scene[key]={m:float(np.mean([r[key][m] for r in members])) for m in ('psnr','ssim')}
        scenes.append(scene)
    result=dict(split=split,frames=len(rows),checkpoint_step=state['step'],checkpoint_sha256=sha(path),
                protocol_id=cfg['protocol_id'],per_image=rows,per_scene=scenes,confidence=collector.finish(),local_acceptance=acceptance.finish(),
                original_size=True,tiling=False,device=str(device),test_used_for_selection=False)
    for key in ('round1','round2','iteration_gain','ungated_candidate'):
        result[key]={f'image_mean_{m}':float(np.mean([r[key][m] for r in rows])) for m in ('psnr','ssim')}
        result[key].update({f'scene_macro_{m}':float(np.mean([r[key][m] for r in scenes])) for m in ('psnr','ssim')})
    json_save(OUT/'metrics'/f'final_{split}.json',result)
    del model,task,state;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return result


@torch.no_grad()
def system_probe(model,task,cfg,rawcfg,device,step):
    """Fixed 28 validation frames track absolute T1, not just apparent T2-T1 gain."""
    from rgb_final_confirmation.evaluation import psnr,ssim
    before=rng_state();was_training=model.training;model.eval();task.eval();rows=[]
    try:
        for i,raw in enumerate(base.proxy_loader(cfg,rawcfg)):
            if device.type=='cuda':guard_gpu()
            batch=base.unpack_batch(raw,device)
            base.set_step_seed(cfg['seed'],'fixed_system_probe',i,batch['sample_id'])
            out=recurrent_forward(model,task,batch)
            rows.append(dict(sample_id=batch['sample_id'][0],t1_psnr=psnr(out.t1,batch['target']),
                             t2_psnr=psnr(out.t2,batch['target']),t1_ssim=ssim(out.t1,batch['target']),
                             t2_ssim=ssim(out.t2,batch['target']),
                             t1_tensor_sha256=hashlib.sha256(out.t1.cpu().contiguous().numpy().tobytes()).hexdigest()))
            del out,batch
        assert len(rows)==28
        result=dict(step=step,frames=28,per_image=rows,test_used=False,selection=False,
                    t1_psnr=float(np.mean([r['t1_psnr'] for r in rows])),
                    t2_psnr=float(np.mean([r['t2_psnr'] for r in rows])))
        initial=OUT/'metrics/system_step_0.json'
        if step and initial.exists():
            reference=json.loads(initial.read_text()); result['t1_change_from_fixed_initial']=result['t1_psnr']-reference['t1_psnr']
            result['t2_change_from_fixed_initial_t1']=result['t2_psnr']-reference['t1_psnr']
            if model.reuse_fixed_first_round:
                assert [r['t1_tensor_sha256'] for r in rows]==[r['t1_tensor_sha256'] for r in reference['per_image']]
                result['first_round_outputs_exactly_unchanged']=True
        json_save(OUT/'metrics'/f'system_step_{step}.json',result)
    finally:
        model.train(was_training);restore_rng(before)
    return result


@torch.no_grad()
def mechanism_diagnostics(cfg,device):
    """Predeclared, validation-only inference interventions; no new training arm."""
    base.add_source_path(cfg)
    from rgb_final_confirmation.evaluation import psnr,ssim
    model,task,criterion,rawcfg=components(cfg,device,with_loss=False);del criterion
    ck=torch.load(OUT/'checkpoints/final.pt',map_location='cpu',weights_only=False)
    assert ck['protocol_id']==cfg['protocol_id']
    model.load_state_dict(ck['r2p_and_projections'],strict=True);task.net_g.load_state_dict(ck['tasknet_net_g'],strict=True)
    model.step=ck['step'];model.tau=ck['tau'];model.eval();del ck
    conditions=(None,'no_feedback','no_polar_correction','confidence_zero','confidence_one','mask_zero','mask_one')
    rows=[]
    for i,raw in enumerate(base.proxy_loader(cfg,rawcfg)):
        batch=base.unpack_batch(raw,device)
        assert batch['sample_id'][0].startswith('validation/')
        record=dict(sample_id=batch['sample_id'][0],conditions={})
        for condition in conditions:
            if device.type=='cuda':guard_gpu()
            base.set_step_seed(cfg['seed'],'mechanism_probe',i,batch['sample_id'])
            out=recurrent_forward(model,task,batch,diagnostic=condition)
            record['conditions'][condition or 'normal']=dict(psnr=psnr(out.t2,batch['target']),
                        ssim=ssim(out.t2,batch['target']),increment_mae=float(out.rgb_increment.abs().mean()),
                        mask_mean=float(out.update_mask.mean()))
            del out
        rows.append(record);print(json.dumps(dict(diagnostic_frame=i+1)),flush=True)
    assert len(rows)==28
    json_save(OUT/'metrics/mechanism_validation.json',dict(frames=28,checkpoint_step=model.step,per_image=rows,
              means={c or 'normal':{m:float(np.mean([r['conditions'][c or 'normal'][m] for r in rows]))
                                   for m in ('psnr','ssim','increment_mae')} for c in conditions},
              test_used=False,tuned_from_results=False,matched_training_ablation=False,
              polar_intervention='remove second-round correction/features; original P0/T1 still carry polarization'))
