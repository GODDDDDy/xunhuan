"""Matched A/B/D training; reuse verified data, E1/H and PolarFree entry points."""
import copy,csv,gc,hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from common import ROOT,sha,json_save,save,seed,rng_state,restore_rng,guard_gpu,status
from enhancer import TaskAdapter
from quality_forward import _fork_from
import gain22,study


class FeedbackAdapter(TaskAdapter):
    def forward(self,feature,p0,current,image,prediction,task_feature,confidence):
        if self.rule=='A':
            return super().forward(feature,p0,current,image,prediction,task_feature,confidence)
        size=feature.shape[-2:]
        resize=lambda x:F.interpolate(x,size=size,mode='bilinear',align_corners=False)
        available=prediction is not None
        assert available==(task_feature is not None)
        c=feature.new_ones((feature.shape[0],1,*size)) if confidence is None else resize(confidence)
        assert torch.isfinite(c).all() and (c>=0).all() and (c<=1).all()
        state=torch.zeros_like(feature)
        if available:
            state=self.prediction(resize(torch.cat((prediction,image-prediction),1)))+self.task(resize(task_feature))
        inputs=torch.cat((feature,self.observation(resize(image)),self.current(resize(current-p0)),
                          state,c,torch.full_like(c,float(available))),1)
        return feature+.1*torch.tanh(self.out(F.gelu(self.depthwise(F.gelu(self.mix(inputs))))))


def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load(device):
    plan=json.loads((ROOT/'config/plan.json').read_text())
    assert sha(ROOT/'config/execution_config.json')==plan['execution_config_sha256']
    assert sha(ROOT/'config/validation256.json')==plan['validation256_sha256']
    for e in plan['code_manifest']:assert sha(ROOT/e['path'])==e['sha256'],e['path']
    cfg=json.loads((ROOT/'config/execution_config.json').read_text())
    for k,h in plan['data_sha256'].items():assert sha(cfg['paths'][k])==h,k
    assert sha(ROOT/'weights/common_initial21.pt')==plan['source_initial_sha256']
    seed(plan['seed']);model,task,raw=gain22._components(cfg,device)
    source=torch.load(ROOT/'weights/common_initial21.pt',map_location='cpu',weights_only=False)
    assert source['protocol_id']==plan['source_initial_protocol']
    model.load_state_dict(source['r2p_and_projections'],strict=True)
    task.net_g.load_state_dict(source['tasknet_net_g'],strict=True)
    model.tau=float(source['tau']);model.eval().requires_grad_(False);task.eval().requires_grad_(False)
    arms={}
    for arm in plan['arms']:
        e2=copy.deepcopy(model.r2p).eval().requires_grad_(False)
        for level,old in list(e2.adapters.items()):
            adapter=FeedbackAdapter(old.out.in_channels,old.task.in_channels).to(device)
            adapter.load_state_dict(old.state_dict(),strict=True);adapter.rule=arm
            e2.adapters[level]=adapter.requires_grad_(True)
        torch.nn.init.zeros_(e2.mid_output.weight);e2.mid_output.requires_grad_(True)
        assert sum(p.numel() for p in e2.parameters() if p.requires_grad)==plan['trainable_per_arm']
        assert not ({id(p) for p in e2.parameters()} & {id(p) for p in model.parameters()})
        arms[arm]=e2
    for name,v in arms['A'].state_dict().items():
        assert all(torch.equal(v,e2.state_dict()[name]) for e2 in arms.values())
    audit(model,task,source,arms)
    return cfg,plan,model,task,raw,source,arms


def trainable(name):return name.startswith(('adapters.level2.','adapters.level3.','mid_output.'))


def audit(model,task,source,arms):
    assert all(torch.equal(v.cpu(),source['r2p_and_projections'][n]) for n,v in model.state_dict().items())
    assert all(torch.equal(v.cpu(),source['tasknet_net_g'][n]) for n,v in task.net_g.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert all(not p.requires_grad and p.grad is None for p in task.parameters())
    for e2 in arms.values():
        for name,p in e2.named_parameters():
            assert p.requires_grad==trainable(name),name
            if not trainable(name):
                assert p.grad is None and torch.equal(p.detach().cpu(),source['r2p_and_projections']['r2p.'+name]),name


@torch.no_grad()
def first(model,task,batch,offload_encoder=False):
    p0,image=batch['polar'],batch['official_rgb']
    # All E2 encoders are frozen, equal to E1, and receive the same raw P0.
    # Capture these pure-function outputs during the unchanged E1 execution.
    encoded={};handles=[]
    modules=dict(p=model.r2p.polar_feature_extractor,p2=model.r2p.down1,
                 x3=model.r2p.bottleneck[-1],skip2=model.r2p.sgsc_half,skip1=model.r2p.sgsc_full)
    for name,module in modules.items():
        def capture(module,args,out,name=name):
            if name not in encoded:encoded[name]=out.detach()
        handles.append(module.register_forward_hook(capture))
    try:enhanced=study.enhance(model,p0,p0,image,project=False)
    finally:
        for handle in handles:handle.remove()
    p1=enhanced.refined_polar
    del enhanced
    assert set(encoded)==set(modules)
    if offload_encoder:encoded={k:v.cpu() for k,v in encoded.items()}
    if p0.is_cuda:torch.cuda.empty_cache()
    cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all() if p0.is_cuda else None
    t1,features=study.task_forward(model,task,p1,image,p0)
    b=model.confidence_head(image,t1,study.polarization_summary(p0),features)
    c=torch.exp(-b/model.tau).detach()
    f1={k:features[k].detach() for k in ('T2','T3')};del features,b
    return dict(p0=p0,image=image,p1=p1.detach(),t1=t1.detach(),f1=f1,c=c,cpu=cpu,cuda=cuda,encoded=encoded)


def second(model,task,e2,shared,arm,use_encoder_cache=True):
    c=shared['c']
    if arm=='D':c=c.mean((-2,-1),keepdim=True).expand_as(c)
    if use_encoder_cache:
        h,w=shared['p0'].shape[-2:];assert h%4==0 and w%4==0
        v=shared['encoded'];prediction=shared['t1'].clamp(0,1)
        x3=e2.adapters['level3'](v['x3'],shared['p0'],shared['p1'],shared['image'],prediction,shared['f1']['T3'],c)
        d2=e2.de_reduce_chan3(torch.cat((e2.de_up3(x3),v['skip2']),1))
        d2=e2.sequence(e2.de_decoder2,d2,v['p2'])
        d2=e2.adapters['level2'](d2,shared['p0'],shared['p1'],shared['image'],prediction,shared['f1']['T2'],c)
        d1=e2.de_reduce_full(torch.cat((e2.de_up2(d2),v['skip1']),1))
        d1=e2.sequence(e2.de_decoder1,d1,v['p']);d1=e2.sequence(e2.mid_refine,d1)
        residual=e2.mid_output(d1);features=None
    else:
        _,features,residual=e2.decode(shared['p0'],current=shared['p1'],image=shared['image'],
                prediction=shared['t1'].clamp(0,1),task_features=shared['f1'],confidence=c,conditioning=True)
    p2=(shared['p1']+.02*residual.tanh()).clamp(0,1)
    del features,residual
    if not torch.is_grad_enabled() and p2.is_cuda:torch.cuda.empty_cache()
    with _fork_from(shared['cpu'],shared['cuda']):
        t2,features=study.task_forward(model,task,p2,shared['image'],shared['p0'])
    del features
    return p2,t2


def loss_fn(t,y):
    return (t-y).square().mean()+.02*sum((t.diff(dim=d)-y.diff(dim=d)).abs().mean()/2 for d in (-1,-2))


def optimizers(arms):
    return {a:torch.optim.Adam([p for p in e.parameters() if p.requires_grad],lr=1e-4,
                              betas=(.9,.999),eps=1e-8,weight_decay=0) for a,e in arms.items()}


def update(model,task,arms,opts,micros,step):
    values={a:[] for a in arms};ids=[];first_hash=[]
    for o in opts.values():o.zero_grad(set_to_none=True)
    for batch in micros:
        ids+=batch['sample_id'];study.base.set_step_seed(2026,'C_feedback_polar_training',step,batch['sample_id'])
        s=first(model,task,batch)
        if step in (1,500,1000,2000,4000,6000):first_hash.append(digest(s['t1']))
        for arm,e2 in arms.items():
            p2,t2=second(model,task,e2,s,arm);loss=loss_fn(t2,batch['target'])
            assert torch.isfinite(loss);(loss/2).backward();values[arm].append(float(loss.detach()))
            del p2,t2,loss
        del s
    norms={}
    for arm,e2 in arms.items():
        parameters=[p for p in e2.parameters() if p.requires_grad]
        norms[arm]=float(torch.nn.utils.clip_grad_norm_(parameters,1,error_if_nonfinite=True))
        opts[arm].step()
    return ids,{a:float(np.mean(v)) for a,v in values.items()},norms,first_hash


def checkpoint(arms,opts,step,plan):
    return dict(step=step,arms={a:{n:p.detach().cpu() for n,p in e.state_dict().items() if trainable(n)} for a,e in arms.items()},
                optimizers={a:o.state_dict() for a,o in opts.items()},rng=rng_state(),
                protocol_id=plan['protocol_id'],source_initial_sha256=plan['source_initial_sha256'])


def restore(ck,arms,opts,plan):
    assert ck['protocol_id']==plan['protocol_id']
    for a,e in arms.items():
        state=e.state_dict();state.update(ck['arms'][a]);e.load_state_dict(state,strict=True)
        opts[a].load_state_dict(ck['optimizers'][a])
    restore_rng(ck['rng'])


def check(device):
    cfg,plan,model,task,raw,source,arms=load(device)
    small=copy.deepcopy(cfg);small['data']['crop_size']=32
    rb=next(iter(study.base.scheduled_loader(small,study.base.build_base_config(small),1)))
    micros=[study.base.unpack_batch(m,device) for m in study.microbatches(rb)]
    seed(111);s=first(model,task,micros[0]);h=digest(s['t1'])
    seed(111);s2=first(model,task,dict(micros[0],target=1-micros[0]['target']))
    assert digest(s2['t1'])==h and torch.equal(s2['c'],s['c']);del s2
    with torch.no_grad():
        for a,e in arms.items():
            p,t=second(model,task,e,s,a);assert torch.equal(p,s['p1']) and torch.equal(t,s['t1'])
        e=arms['A'].adapters['level2'];old=model.r2p.adapters['level2']
        args=(s['f1']['T2'],s['p0'],s['p1'],s['image'],s['t1'].clamp(0,1),s['f1']['T2'],s['c'])
        assert torch.equal(e(*args),old(*args))
        events=[]
        hooks=[]
        for a in ('B','D'):
            for level,adapter in arms[a].adapters.items():
                hooks.append(adapter.mix.register_forward_pre_hook(lambda m,args,a=a,level=level:events.append((a,level,args[0].detach().clone()))))
            second(model,task,arms[a],s,a)
        for handle in hooks:handle.remove()
        for a,level,inputs in events:
            adapter=arms[a].adapters[level];channels=adapter.out.in_channels;size=inputs.shape[-2:]
            resize=lambda x:F.interpolate(x,size=size,mode='bilinear',align_corners=False)
            expected=adapter.prediction(resize(torch.cat((s['t1'].clamp(0,1),s['image']-s['t1'].clamp(0,1)),1)))+adapter.task(resize(s['f1']['T2' if level=='level2' else 'T3']))
            assert torch.equal(inputs[:,3*channels:4*channels],expected)
            cond=s['c'] if a=='B' else s['c'].mean((-2,-1),keepdim=True).expand_as(s['c'])
            assert torch.equal(inputs[:,-2:-1],resize(cond))
    del s,events,args,p,t,inputs,expected
    opts=optimizers(arms);initial=copy.deepcopy(checkpoint(arms,opts,0,plan))
    for i in range(1,4):update(model,task,arms,opts,micros,i)
    gradients={}
    for a,e in arms.items():
        gradients[a]={name:sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
                      for name,module in [('level2',e.adapters['level2']),('level3',e.adapters['level3']),('output',e.mid_output)]}
        assert all(v>0 and np.isfinite(v) for v in gradients[a].values()),gradients
    # Compare after nonzero-head updates so an identity output cannot hide an
    # incorrect cache. Verify both restored pixels and all trainable gradients.
    seed(111);cache_test=first(model,task,micros[0])
    for a,e in arms.items():
        opts[a].zero_grad(set_to_none=True)
        pc,tc=second(model,task,e,cache_test,a,True);loss_fn(tc,micros[0]['target']).backward()
        cached_grad={n:p.grad.detach().clone() for n,p in e.named_parameters() if p.requires_grad}
        opts[a].zero_grad(set_to_none=True)
        pu,tu=second(model,task,e,cache_test,a,False)
        assert torch.equal(pc,pu) and torch.equal(tc,tu),'Cached encoder changed predictions'
        loss_fn(tu,micros[0]['target']).backward()
        assert all(torch.equal(cached_grad[n],p.grad) for n,p in e.named_parameters() if p.requires_grad),'Cached encoder changed gradients'
        opts[a].zero_grad(set_to_none=True)
    del cache_test,pc,tc,pu,tu,cached_grad
    audit(model,task,source,arms)
    seed(111);after=first(model,task,micros[0]);assert digest(after['t1'])==h
    replay=copy.deepcopy(checkpoint(arms,opts,3,plan));update(model,task,arms,opts,micros,4)
    expected=copy.deepcopy(checkpoint(arms,opts,4,plan));restore(replay,arms,opts,plan)
    update(model,task,arms,opts,micros,4);actual=checkpoint(arms,opts,4,plan)
    assert all(torch.equal(v,actual['arms'][a][n]) for a,state in expected['arms'].items() for n,v in state.items())
    if device.type=='cuda':
        # Exercise the actual evaluation offload/reload path as well: RNG state
        # must remain a CPU ByteTensor while feature tensors move to CUDA.
        row=eval_one(cfg,model,task,arms,micros[0],save_arrays=False)
        assert row['sample_id']==micros[0]['sample_id'][0]
    json_save(ROOT/'diagnostics'/('cpu_checks.json' if device.type=='cpu' else 'gpu_small_checks.json'),
              dict(passed=True,GT_invariance=True,zero_head_identity_exact=True,A_rule_exact=True,
                   B_D_inputs_exact=True,independent_E2=True,trainable_per_arm=292896,
                   gradients=gradients,source_and_first_exact_after_updates=True,optimizer_replay_exact=True,
                   frozen_encoder_cache_outputs_and_gradients_exact=True,preflight_updates_discarded=True))
    return cfg,plan,model,task,raw,source,arms


def full_item(cfg,sid,device):
    from rgb_final_confirmation.data import load_manifest,PairedBranchDataset
    row=next(r for r in load_manifest(cfg['paths']['manifest']) if r.sample_id==sid)
    item=PairedBranchDataset([row],group='A1',crop_size=None)[0]
    return {k:v[None].to(device) if torch.is_tensor(v) else [v] for k,v in item.items()}


@torch.no_grad()
def eval_one(cfg,model,task,arms,batch,save_arrays=False,array_directory=None):
    from rgb_final_confirmation.evaluation import psnr,ssim
    study.base.set_step_seed(2026,'H_only_pair_eval',0,batch['sample_id']);s=first(model,task,batch,offload_encoder=True)
    # Reuse the original evaluator's device and FP32 operations. The complete
    # floating outputs below are transferred only for storage/local diagnostics.
    base_metrics=dict(psnr=psnr(s['t1'],batch['target']),ssim=ssim(s['t1'],batch['target']))
    # Offload immutable shared feedback between complete full-frame passes.
    for k in ('p0','image','p1','t1','c'):s[k]=s[k].cpu()
    s['f1']={k:v.cpu() for k,v in s['f1'].items()};torch.cuda.empty_cache()
    target=batch['target'].cpu();row=dict(sample_id=batch['sample_id'][0],scene=batch['scene'][0],
          shape=list(target.shape),t1_sha256=digest(s['t1']),metrics={})
    row['metrics']['T1']=base_metrics
    arrays=dict(t1=s['t1'].numpy(),p1=s['p1'].numpy(),c=s['c'].numpy(),mean_c=s['c'].mean((-2,-1),keepdim=True).expand_as(s['c']).numpy())
    target_error=(s['t1'].clamp(0,1)-target).abs().mean(1,keepdim=True)
    quality=F.interpolate(s['c'],size=target.shape[-2:],mode='bilinear',align_corners=False)
    low=quality<=torch.quantile(quality,.2);high=quality>=torch.quantile(quality,.8)
    row['local']={}
    for a,e2 in arms.items():
        current=dict(s)
        for k in ('p0','image','p1','t1','c'):current[k]=s[k].cuda()
        current['f1']={k:v.cuda() for k,v in s['f1'].items()}
        current['encoded']={k:v.cuda() for k,v in s['encoded'].items()}
        assert current['cpu'].device.type=='cpu' and current['cpu'].dtype==torch.uint8
        p2,t2=second(model,task,e2,current,a)
        row['metrics'][a]=dict(psnr=psnr(t2,batch['target']),ssim=ssim(t2,batch['target']))
        p2=p2.cpu();t2=t2.cpu();del current;torch.cuda.empty_cache()
        difference=(t2.clamp(0,1)-target).abs().mean(1,keepdim=True)-target_error
        dp=(p2-s['p1']).abs()
        row['local'][a]=dict(lowC20_mae_change=float(difference[low].mean()),highC20_mae_change=float(difference[high].mean()),
                            mean_polar_change=float(dp.mean()),bound_hit_fraction=float((dp>=.02*(1-1e-4)).float().mean()),
                            nonzero_T_change_fraction=float((t2!=s['t1']).float().mean()))
        arrays['p2_'+a]=p2.numpy();arrays['t2_'+a]=t2.numpy();arrays['local_error_'+a]=difference.numpy()
        del p2,t2,difference,dp
    if save_arrays:
        folder=Path(array_directory) if array_directory else ROOT/'examples';folder.mkdir(parents=True,exist_ok=True)
        # ZIP_STORED: identical floating arrays, no costly deflate or quantizing.
        np.savez(folder/(row['sample_id'].replace('/','_')+'.npz'),**arrays)
    return row


def summarize(rows,counts=None):
    scenes=sorted({r['scene'] for r in rows});result={}
    for a,b in [('B','A'),('B','T1'),('B','D'),('A','T1'),('D','T1')]:
        key=a+'-'+b;result[key]={}
        for metric in ('psnr','ssim'):
            byscene={s:np.array([r['metrics'][a][metric]-r['metrics'][b][metric] for r in rows if r['scene']==s]) for s in scenes}
            means={s:float(v.mean()) for s,v in byscene.items()};macro=float(np.mean(list(means.values())))
            w={s:(counts[s]/sum(counts.values()) if counts else len(byscene[s])/len(rows)) for s in scenes}
            variance=sum((1/len(scenes))**2*(1-len(v)/counts[s])*float(v.var(ddof=1))/len(v)
                         for s,v in byscene.items()) if counts else 0.
            half=1.96*max(variance,0.)**.5
            scene_half={s:(1.96*((1-len(v)/counts[s])*float(v.var(ddof=1))/len(v))**.5 if counts else 0.) for s,v in byscene.items()}
            result[key][metric]=dict(scene_equal=macro,image_equal_estimate=sum(w[s]*means[s] for s in scenes),
                                      sampling_halfwidth=half,interval=[macro-half,macro+half],
                                      scenes=means,scene_halfwidth=scene_half,scenes_positive=sum(v>0 for v in means.values()))
    return result


def need_full(stats):
    reasons=[]
    for pair in ('B-A','B-T1'):
        for metric,bound,precision in [('psnr',.05,.02),('ssim',0.,.0005)]:
            m=stats[pair][metric];lo,hi=m['interval']
            if m['sampling_halfwidth']>precision or lo<=bound<=hi:reasons.append(pair+'_'+metric+'_precision_or_boundary')
        m=stats[pair]['psnr'];lower=sum(v-m['scene_halfwidth'][s]>0 for s,v in m['scenes'].items())
        upper=sum(v+m['scene_halfwidth'][s]>0 for s,v in m['scenes'].items())
        if lower<6<=upper:reasons.append(pair+'_scene_count_uncertain')
    # B-D ambiguity is reported, not an automatic rescue experiment. A positive
    # local-C claim would require any needed complete validation separately.
    return sorted(set(reasons))


def evaluate(cfg,plan,model,task,arms,split,ids,existing=None):
    rows=list(existing or []);done={r['sample_id'] for r in rows}
    for sid in ids:
        if sid in done:continue
        guard_gpu();batch=full_item(cfg,sid,torch.device('cuda:0'))
        row=eval_one(cfg,model,task,arms,batch,save_arrays=True);rows.append(row);del batch;gc.collect();torch.cuda.empty_cache()
        json_save(ROOT/'metrics'/f'{split}_partial.json',dict(rows=rows,original_size=True,tiling=False))
        if len(rows)%8==0:status('evaluating',split=split,completed=len(rows),total=len(ids))
    counts=plan['evaluation']['scene_counts'] if split.startswith('validation') and len(rows)==256 else None
    report=dict(split=split,frames=len(rows),rows=rows,comparisons=summarize(rows,counts),
                original_size=True,tiling=False,step=6000,protocol_id=plan['protocol_id'])
    json_save(ROOT/'metrics'/f'{split}.json',report)
    return report


def run():
    assert json.loads((ROOT/'diagnostics/cpu_checks.json').read_text())['passed']
    assert (ROOT/'budget.json').exists(),'Controller must establish immutable budget first'
    cfg,plan,model,task,raw,source,arms=check(torch.device('cuda:0'));guard_gpu()
    # Fresh copies and optimizers discard every diagnostic update.
    for a,e in arms.items():
        state={n:v.clone() for n,v in model.r2p.state_dict().items()};state['mid_output.weight'].zero_();e.load_state_dict(state,strict=True)
        for p in e.parameters():p.grad=None
    opts=optimizers(arms);timings=[]
    probe_loader=iter(study.base.scheduled_loader(cfg,raw,4))
    for step in range(1,5):
        start=time.time();rb=next(probe_loader)
        micros=[study.base.unpack_batch(m,torch.device('cuda:0')) for m in study.microbatches(rb)]
        update(model,task,arms,opts,micros,step);torch.cuda.synchronize();timings.append(time.time()-start);del micros
    audit(model,task,source,arms)
    # The max-size manifest entry is selected by dimensions, never by score.
    from rgb_final_confirmation.data import load_manifest,select_rows
    from PIL import Image
    candidates=select_rows(load_manifest(cfg['paths']['manifest']),'validation')+select_rows(load_manifest(cfg['paths']['manifest']),'test')
    def area(row):
        with Image.open(row.polar_000) as im:return im.width*im.height
    largest=max(candidates,key=area)
    torch.cuda.synchronize();start=time.time();batch=full_item(cfg,largest.sample_id,torch.device('cuda:0'))
    eval_one(cfg,model,task,arms,batch,save_arrays=True,array_directory=ROOT/'diagnostics/max_frame_preflight')
    torch.cuda.synchronize();frame_seconds=time.time()-start
    del batch;gc.collect();torch.cuda.empty_cache()
    budget=json.loads((ROOT/'budget.json').read_text());step_seconds=float(np.median(timings))
    reserve=max(3600,1.5*1060*frame_seconds+900)
    estimated=1.5*(6000*step_seconds+1060*frame_seconds)+900
    standard=1.5*(6000*step_seconds+444*frame_seconds)+900
    remaining=budget['deadline_unix']-time.time()
    benchmark=dict(train_step3_seconds=step_seconds,full_frame3_seconds=frame_seconds,
                   max_size_sample=largest.sample_id,remaining_budget_seconds=remaining,
                   standard_estimated_seconds=standard,worst_estimated_seconds=estimated,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),evaluation_reserve_seconds=reserve,
                   passed=estimated<=remaining,preflight_updates_discarded=True)
    json_save(ROOT/'diagnostics/benchmark.json',benchmark)
    assert estimated<=remaining,'6000-step train plus required worst-case evaluation does not fit approved budget'
    budget['train_deadline_unix']=budget['deadline_unix']-reserve
    json_save(ROOT/'budget.json',budget)
    for a,e in arms.items():
        state={n:v.clone() for n,v in model.r2p.state_dict().items()};state['mid_output.weight'].zero_();e.load_state_dict(state,strict=True)
        for p in e.parameters():p.grad=None
    opts=optimizers(arms);seed(2026);save(ROOT/'weights/initial.pt',checkpoint(arms,opts,0,plan))
    status('training_started',steps_per_arm=6000,benchmark=benchmark)
    seen=set();train_start=time.time()
    with (ROOT/'logs/training.csv').open('w',newline='') as f:
        writer=None
        for step,rb in enumerate(study.base.scheduled_loader(cfg,raw,6000),1):
            guard_gpu();assert time.time()<budget['train_deadline_unix'],'Training cutoff reached before fixed6000 endpoint'
            start=time.time();micros=[study.base.unpack_batch(m,torch.device('cuda:0')) for m in study.microbatches(rb)]
            ids,losses,norms,hashes=update(model,task,arms,opts,micros,step);del micros;seen.update(ids)
            torch.cuda.synchronize();record=dict(step=step,sample_ids=';'.join(ids),seconds=time.time()-start)
            for a in arms:record[a+'_loss']=losses[a];record[a+'_grad_norm']=norms[a]
            if writer is None:writer=csv.DictWriter(f,fieldnames=record);writer.writeheader()
            writer.writerow(record);f.flush()
            if step%50==0 or step==1:status('training',step=step,total=6000,losses=losses,seconds=record['seconds'])
            if step%500==0:
                audit(model,task,source,arms);save(ROOT/'weights'/f'step_{step}.pt',checkpoint(arms,opts,step,plan))
                json_save(ROOT/'diagnostics'/f'frozen_step_{step}.json',dict(frozen_exact=True,shared_first=True,t1_hashes=hashes))
    assert step==6000 and len(seen)==5440;audit(model,task,source,arms)
    save(ROOT/'weights/final.pt',checkpoint(arms,opts,step,plan))
    json_save(ROOT/'diagnostics/training_complete.json',dict(steps_per_arm=6000,unique_train_images=5440,
              crop_presentations_per_arm=12000,training_seconds=time.time()-train_start,
              endpoint_sha256=sha(ROOT/'weights/final.pt'),frozen_exact=True))
    selected=json.loads((ROOT/'config/validation256.json').read_text())['sample_ids']
    validation=evaluate(cfg,plan,model,task,arms,'validation256',selected)
    reasons=need_full(validation['comparisons'])
    json_save(ROOT/'metrics/validation_precision_decision.json',dict(reasons=reasons,complete_validation_required=bool(reasons),test_read_for_decision=False))
    if reasons:
        allids=[r.sample_id for r in select_rows(load_manifest(cfg['paths']['manifest']),'validation')]
        validation=evaluate(cfg,plan,model,task,arms,'validation872',allids,validation['rows'])
    testids=[r.sample_id for r in select_rows(load_manifest(cfg['paths']['manifest']),'test')]
    test=evaluate(cfg,plan,model,task,arms,'test188',testids)
    audit(model,task,source,arms)
    json_save(ROOT/'reports/final_summary.json',dict(protocol_id=plan['protocol_id'],steps_per_arm=6000,
              validation_frames=validation['frames'],test_frames=188,validation=validation['comparisons'],
              test=test['comparisons'],source_frozen_exact=True,endpoint_sha256=sha(ROOT/'weights/final.pt'),
              validation_expansion_reasons=reasons,test_used_for_selection=False))
    status('completed',validation=validation['frames'],test=188,steps_per_arm=6000)


if __name__=='__main__':
    torch.set_num_threads(2)
    try:
        if sys.argv[1]=='cpu':check(torch.device('cpu'));status('cpu_verified')
        elif sys.argv[1]=='run':run()
        else:raise ValueError(sys.argv[1])
    except Exception:
        json_save(ROOT/'diagnostics'/f'failure_{int(time.time())}.json',dict(time_unix=time.time(),traceback=traceback.format_exc()))
        status('failed',traceback=traceback.format_exc());raise
