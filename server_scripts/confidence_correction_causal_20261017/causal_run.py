"""V2 causal diagnosis. Reuse original experiment30 loader/forward/loss/metrics."""
import copy,csv,gc,hashlib,json,math,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'c_feedback_polar_loop_20261015'
ORA=ROOT.parent/'oracle_quality_diagnostics_20261016'
sys.path.insert(0,str(OLD/'scripts'))
import c_loop_run as base
FIRST,SECOND=base.first,base.second
PREFIX=('de_up3.','de_reduce_chan3.','de_decoder2.','de_up2.','de_reduce_full.','de_decoder1.','mid_refine.')
ALL=['R1D','R1P','R1O','R0N','R1N']
DEVICE=torch.device('cuda:0')

def write(p,x):base.json_save(Path(p),x)
def status(stage,**kw):
    obj=dict(stage=stage,time_unix=time.time(),**kw);write(ROOT/'status.json',obj);print(json.dumps(obj,ensure_ascii=False),flush=True)
def guard():
    assert time.time()<json.loads((ROOT/'budget.json').read_text())['deadline_unix'],'36h deadline reached'
def verify():
    p=json.loads((ROOT/'config/proposal.json').read_text())
    for r in p['source_receipts']:assert base.sha(r['path'])==r['sha256'],r['path']
    for r in p['code_manifest']+p['data_manifest']:assert base.sha(ROOT/r['path'])==r['sha256'],r['path']
    return p
def active(name,arm):
    small=name.startswith(('adapters.level2.','adapters.level3.','mid_output.'))
    large=arm.startswith('R1') and name.startswith(PREFIX)
    return (small or large) and not ('.ffn.dwconv.' in name and not name.startswith('mid_refine.'))

class ControlledAdapter(base.FeedbackAdapter):
    def forward(self,feature,p0,current,image,prediction,task_feature,confidence):
        if self.suppress_task:prediction=task_feature=None
        return super().forward(feature,p0,current,image,prediction,task_feature,confidence)

def first(model,task,batch,offload_encoder=False):
    s=FIRST(model,task,batch,offload_encoder=offload_encoder)
    s['target']=batch['target']
    return s
def second(model,task,e,s,arm,use_encoder_cache=True):
    q=dict(s);c=s['c']
    if arm.endswith(('D','N')):c=c.mean((-2,-1),keepdim=True).expand_as(c)
    elif arm.endswith('O'):
        err=F.avg_pool2d((s['t1'].clamp(0,1)-s['target']).abs().mean(1,keepdim=True),4).clamp_min(.001)
        c=torch.exp(-err/model.tau).detach()
    q['c']=c
    return SECOND(model,task,e,q,'B',use_encoder_cache=use_encoder_cache)

def make_arms(template,names):
    out={}
    for arm in names:
        e=copy.deepcopy(template).eval().requires_grad_(False)
        for level,old in list(e.adapters.items()):
            adapter=ControlledAdapter(old.out.in_channels,old.task.in_channels).to(next(e.parameters()).device)
            adapter.load_state_dict(old.state_dict(),strict=True);adapter.rule='B';adapter.suppress_task=arm.endswith('N')
            e.adapters[level]=adapter
        torch.nn.init.zeros_(e.mid_output.weight)
        for n,p in e.named_parameters():p.requires_grad_(active(n,arm))
        out[arm]=e
    return out

def components(names,device=DEVICE):
    cfg,oldplan,model,task,raw,source,oldarms=base.load(device)
    template=oldarms.pop('B');oldarms.clear();del oldarms;gc.collect()
    arms=make_arms(template,names);del template;gc.collect()
    opts=base.optimizers(arms)
    ck=torch.load(OLD/'weights/initial.pt',map_location='cpu',weights_only=False)
    for a,e in arms.items():
        for n,v in ck['arms']['B'].items():assert torch.equal(e.state_dict()[n].cpu(),v),n
    base.restore_rng(ck['rng'])
    base.first=first;base.second=second;base.guard_gpu=guard
    return cfg,model,task,raw,source,arms,opts

def audit(model,task,source,arms):
    assert all(torch.equal(v.cpu(),source['r2p_and_projections'][n]) for n,v in model.state_dict().items())
    assert all(torch.equal(v.cpu(),source['tasknet_net_g'][n]) for n,v in task.net_g.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert all(not p.requires_grad and p.grad is None for p in task.parameters())
    for a,e in arms.items():
        for n,p in e.named_parameters():
            assert p.requires_grad==active(n,a),(a,n)
            if not active(n,a):assert p.grad is None and torch.equal(p.detach().cpu(),source['r2p_and_projections']['r2p.'+n]),(a,n)

def ckpt(arms,opts,step,protocol):
    return dict(step=step,protocol_id=protocol,arms={a:{n:p.detach().cpu().clone() for n,p in e.state_dict().items() if active(n,a)} for a,e in arms.items()},
                optimizers={a:o.state_dict() for a,o in opts.items()},rng=base.rng_state())
def restore(ck,arms,opts,protocol):
    assert ck['protocol_id']==protocol and set(ck['arms'])==set(arms)
    for a,e in arms.items():
        s=e.state_dict();s.update(ck['arms'][a]);e.load_state_dict(s,strict=True)
        opts[a].load_state_dict(ck['optimizers'][a]);opts[a].zero_grad(set_to_none=True)
    base.restore_rng(ck['rng'])
def micros(cfg,raw,start,end):
    for rb in base.study.base.scheduled_loader(cfg,raw,end,start_step=start):
        yield [base.study.base.unpack_batch(m,DEVICE) for m in base.study.microbatches(rb)]

def preflight(proposal):
    if (ROOT/'config/locked.json').exists():return json.loads((ROOT/'config/locked.json').read_text())
    status('preflight_loading')
    cfg,model,task,raw,source,arms,opts=components(ALL)
    counts={a:sum(p.numel() for p in e.parameters() if p.requires_grad) for a,e in arms.items()}
    assert counts['R0N']==292896 and len(set(counts[a] for a in ALL if a.startswith('R1')))==1
    assert not any({id(p) for p in model.parameters()} & {id(p) for p in e.parameters()} for e in arms.values())
    # Reproduce original B at two real optimizer states, before new-arm updates.
    replay=make_arms(model.r2p,['R0P']);ro=base.optimizers(replay)
    logs={int(r['step']):r for r in csv.DictReader((OLD/'logs/training.csv').open())}
    proof=[]
    for previous in [0,500]:
        old=torch.load(OLD/'weights'/('initial.pt' if previous==0 else 'step_500.pt'),map_location='cpu',weights_only=False)
        st=replay['R0P'].state_dict();st.update(old['arms']['B']);replay['R0P'].load_state_dict(st)
        ro['R0P'].load_state_dict(old['optimizers']['B']);base.restore_rng(old['rng']);ro['R0P'].zero_grad(set_to_none=True)
        batches=next(micros(cfg,raw,previous,previous+1))
        ids,loss,norm,_=base.update(model,task,replay,ro,batches,previous+1)
        expected=logs[previous+1]
        assert ids==expected['sample_ids'].split(';')
        assert loss['R0P']==float(expected['B_loss']),(previous,loss,expected['B_loss'])
        assert norm['R0P']==float(expected['B_grad_norm']),(previous,norm,expected['B_grad_norm'])
        proof.append(dict(step=previous+1,loss_exact=True,gradient_exact=True));del old,batches
    del replay,ro;gc.collect();torch.cuda.empty_cache()
    status('preflight_replay_passed',replay=proof,trainable_counts=counts)
    # New conditions: identity at zero head; oracle exactly original; N flags zero.
    batches=next(micros(cfg,raw,0,1));s=first(model,task,batches[0]);events=[];hooks=[]
    for a in ALL:
        for level,ad in arms[a].adapters.items():
            hooks.append(ad.mix.register_forward_pre_hook(lambda m,args,a=a,level=level:events.append((a,level,args[0][:,-2:].detach().clone()))))
    with torch.no_grad():
        for a,e in arms.items():
            p,t=second(model,task,e,s,a);assert torch.equal(p,s['p1']) and torch.equal(t,s['t1'])
    for h in hooks:h.remove()
    for a,level,x in events:
        c=s['c']
        if a.endswith(('N','D')):c=c.mean((-2,-1),keepdim=True).expand_as(c)
        elif a.endswith('O'):c=torch.exp(-F.avg_pool2d((s['t1'].clamp(0,1)-s['target']).abs().mean(1,keepdim=True),4).clamp_min(.001)/model.tau)
        assert torch.equal(x[:,:1],F.interpolate(c,size=x.shape[-2:],mode='bilinear',align_corners=False))
        assert torch.equal(x[:,1:],torch.full_like(x[:,1:],0. if a.endswith('N') else 1.))
    t1_before=base.digest(s['t1']);del s,events,hooks,p,t
    timings=[]
    for step,batches in enumerate(micros(cfg,raw,0,6),1):
        tick=time.time();base.update(model,task,arms,opts,batches,step);torch.cuda.synchronize();timings.append(time.time()-tick)
    grads={a:{prefix:sum(float(p.grad.abs().sum()) for n,p in e.named_parameters() if n.startswith(prefix) and p.grad is not None) for prefix in (['adapters.','mid_output.']+(['de_decoder1.','de_decoder2.','mid_refine.'] if a.startswith('R1') else []))} for a,e in arms.items()}
    assert all(all(math.isfinite(v) and v>0 for v in g.values()) for g in grads.values()),grads
    audit(model,task,source,arms)
    # Real Adam/RNG/checkpoint replay, including newly opened weights.
    snap=copy.deepcopy(ckpt(arms,opts,6,'preflight'));base.update(model,task,arms,opts,batches,7)
    expected=copy.deepcopy(ckpt(arms,opts,7,'preflight'));restore(snap,arms,opts,'preflight')
    base.update(model,task,arms,opts,batches,7);actual=ckpt(arms,opts,7,'preflight')
    assert all(torch.equal(t,actual['arms'][a][n]) for a,st in expected['arms'].items() for n,t in st.items())
    del snap,expected,actual,batches,opts
    for e in arms.values():e.zero_grad(set_to_none=True)
    gc.collect();torch.cuda.empty_cache()
    cfg_rows=json.loads((OLD/'metrics/validation872.json').read_text())['rows']
    largest=max(cfg_rows,key=lambda r:np.prod(r['shape'][-2:]));sid=largest['sample_id']
    batch=base.full_item(cfg,sid,DEVICE)
    tick=time.time();row=eval_one(cfg,model,task,arms,batch);torch.cuda.synchronize();frame_seconds=time.time()-tick
    assert row['t1_sha256']==largest['t1_sha256'] and row['metrics']['T1']==largest['metrics']['T1']
    del batch;gc.collect();torch.cuda.empty_cache()
    train_seconds=float(np.median(timings[2:]))
    budget=json.loads((ROOT/'budget.json').read_text());elapsed=time.time()-budget['start_unix']
    forecast=elapsed+1.2*(6000*train_seconds+1060*frame_seconds+7200)+7200
    # Largest-image time makes this conservative; selection precedes all formal training.
    names=list(ALL)
    if forecast>30*3600:
        names=ALL[:3]
        for a in ALL[3:]:del arms[a]
        opts=base.optimizers(arms)
        gc.collect();torch.cuda.empty_cache()
        batch=base.full_item(cfg,sid,DEVICE);tick=time.time();eval_one(cfg,model,task,arms,batch);torch.cuda.synchronize()
        f3=time.time()-tick;del batch
        timing3=[]
        for step,bs in enumerate(micros(cfg,raw,10,13),11):
            tick=time.time();base.update(model,task,arms,opts,bs,step);torch.cuda.synchronize();timing3.append(time.time()-tick)
        forecast=elapsed+1.2*(6000*float(np.median(timing3))+1060*f3+7200)+7200
        assert forecast<=30*3600,('Even 3-arm plan exceeds approved resource envelope',forecast)
    locked=dict(proposal_protocol=proposal['protocol_id'],arms=names,steps=6000,trainable_counts={a:counts[a] for a in names},
        train_step5_seconds=train_seconds,max_frame5_seconds=frame_seconds,estimated_total_hours=forecast/3600,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),original_B_replay=proof,gradients=grads,
        identity=True,conditions_verified=True,optimizer_replay_exact=True,source_frozen_exact=True,
        original_size_verified=True,largest_sample=sid,all_preflight_updates_discarded=True,time_unix=time.time())
    locked['protocol_id']=hashlib.sha256(json.dumps(locked,sort_keys=True).encode()).hexdigest()
    write(ROOT/'config/locked.json',locked);write(ROOT/'diagnostics/preflight.json',locked)
    status('preflight_passed',arms=names,estimated_total_hours=forecast/3600,trainable_counts=locked['trainable_counts'])
    del model,task,source,arms
    if 'opts' in locals():del opts
    gc.collect();torch.cuda.empty_cache()
    return locked

@torch.no_grad()
def eval_one(cfg,model,task,arms,batch,example=False):
    from rgb_final_confirmation.evaluation import psnr,ssim
    base.study.base.set_step_seed(2026,'H_only_pair_eval',0,batch['sample_id'])
    s=first(model,task,batch,offload_encoder=True)
    row=dict(sample_id=batch['sample_id'][0],scene=batch['scene'][0],shape=list(batch['target'].shape),t1_sha256=base.digest(s['t1']),
        metrics=dict(T1=dict(psnr=psnr(s['t1'],batch['target']),ssim=ssim(s['t1'],batch['target']))),local={})
    for k in ['p0','image','p1','t1','c','target']:s[k]=s[k].cpu()
    s['f1']={k:v.cpu() for k,v in s['f1'].items()};torch.cuda.empty_cache()
    arrays={k:s[k].numpy() for k in ['p1','t1','c','target']} if example else None
    for a,e in arms.items():
        q=dict(s)
        for k in ['p0','image','p1','t1','c','target']:q[k]=s[k].to(DEVICE)
        q['f1']={k:v.to(DEVICE) for k,v in s['f1'].items()};q['encoded']={k:v.to(DEVICE) for k,v in s['encoded'].items()}
        p,t=second(model,task,e,q,a)
        row['metrics'][a]=dict(psnr=psnr(t,batch['target']),ssim=ssim(t,batch['target']))
        dp=(p-q['p1']).abs();dt=(t-q['t1']).abs()
        row['local'][a]=dict(mean_polar_change=float(dp.mean()),bound_hit_fraction=float((dp>=.02*(1-1e-4)).float().mean()),mean_RGB_change=float(dt.mean()))
        if example:arrays['p2_'+a]=p.cpu().numpy();arrays['t2_'+a]=t.cpu().numpy()
        del q,p,t,dp,dt;torch.cuda.empty_cache()
    if example:np.savez(ROOT/'examples'/(row['sample_id'].replace('/','_')+'.npz'),**arrays)
    return row

def load_crops(cfg):
    from rgb_final_confirmation.data import load_manifest,select_rows,PairedBranchDataset
    selected=json.loads((ROOT/'config/diagnostic_crops.json').read_text())
    ds=PairedBranchDataset(select_rows(load_manifest(cfg['paths']['manifest']),'train'),group='A1',crop_size=128,schedule=selected)
    result=[]
    for i in range(len(ds)):
        item=ds[i];assert item['sample_id']==selected[i]['sample_id']
        result.append({k:v[None].to(DEVICE) if torch.is_tensor(v) else [v] for k,v in item.items()})
    return result

def diagnostic(locked):
    if (ROOT/'reports/diagnostic_summary.json').exists():return
    status('diagnostic_loading')
    cfg,model,task,raw,source,arms,opts=components(['R0P','R1P'])
    batches=load_crops(cfg);shared=[]
    for batch in batches:
        base.study.base.set_step_seed(2026,'causal_fixed_crop',0,batch['sample_id']);shared.append(first(model,task,batch))
    from rgb_final_confirmation.evaluation import psnr,ssim
    path=ROOT/'diagnostics/direct_rows.json';rows=json.loads(path.read_text()) if path.exists() else []
    for i in range(len(rows),16):
        guard();s=shared[i];delta=torch.zeros_like(s['p1'],requires_grad=True);opt=torch.optim.Adam([delta],lr=.001)
        def forward_delta():
            p2=(s['p1']+delta).clamp(0,1)
            with base._fork_from(s['cpu'],s['cuda']):t,_=base.study.task_forward(model,task,p2,s['image'],s['p0'])
            return t
        best=float('inf');best_delta=delta.detach().clone()
        with torch.no_grad():initial=base.loss_fn(forward_delta(),s['target']).item()
        best=initial
        for step in range(1,101):
            guard();opt.zero_grad(set_to_none=True);t=forward_delta();loss=base.loss_fn(t,s['target']);assert torch.isfinite(loss)
            value=float(loss.detach())
            if value<best:best=value;best_delta=delta.detach().clone()
            loss.backward();opt.step()
            with torch.no_grad():delta.clamp_(-.02,.02)
            del t,loss
        with torch.no_grad():
            end=forward_delta();end_loss=float(base.loss_fn(end,s['target']))
            if end_loss<best:best=end_loss;best_delta=delta.detach().clone()
            final=dict(psnr=psnr(end,s['target']),ssim=ssim(end,s['target']))
            delta.copy_(best_delta);chosen=forward_delta()
            rows.append(dict(sample_id=batches[i]['sample_id'][0],scene=batches[i]['scene'][0],steps=100,
                initial_loss=initial,best_loss=best,final_loss=end_loss,T1=dict(psnr=psnr(s['t1'],s['target']),ssim=ssim(s['t1'],s['target'])),
                best=dict(psnr=psnr(chosen,s['target']),ssim=ssim(chosen,s['target'])),final=final,
                mean_modification=float(best_delta.abs().mean()),bound_fraction=float((best_delta.abs()>=.02*(1-1e-4)).float().mean())))
        write(path,rows);status('direct_crop_diagnostic',completed=len(rows),total=16)
        del delta,opt,chosen,end,best_delta;gc.collect();torch.cuda.empty_cache()
    fit_protocol=locked['protocol_id']+'_fit';last=ROOT/'weights/fit_last.pt';start=0
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=False);restore(ck,arms,opts,fit_protocol);start=ck['step']
    curve=ROOT/'logs/fit.jsonl'
    if curve.exists():
        records=[json.loads(v) for v in curve.read_text().splitlines() if v.strip()];records=[r for r in records if r['step']<=start]
        curve.write_text(''.join(json.dumps(r)+'\n' for r in records))
    with curve.open('a') as f:
        for step in range(start+1,1001):
            guard();values={a:[] for a in arms}
            for o in opts.values():o.zero_grad(set_to_none=True)
            for ix in [(2*(step-1))%16,(2*(step-1)+1)%16]:
                s=shared[ix]
                for a,e in arms.items():
                    _,t=second(model,task,e,s,a);loss=base.loss_fn(t,s['target']);assert torch.isfinite(loss)
                    (loss/2).backward();values[a].append(float(loss.detach()));del t,loss
            for a,e in arms.items():torch.nn.utils.clip_grad_norm_([p for p in e.parameters() if p.requires_grad],1,error_if_nonfinite=True);opts[a].step()
            f.write(json.dumps(dict(step=step,loss={a:float(np.mean(v)) for a,v in values.items()}))+'\n');f.flush()
            if step%100==0:
                base.save(last,ckpt(arms,opts,step,fit_protocol));status('small_sample_fitting',step=step,total=1000)
    fit_train=[]
    with torch.no_grad():
        for batch,s in zip(batches,shared):
            r=dict(sample_id=batch['sample_id'][0],scene=batch['scene'][0],metrics=dict(T1=dict(psnr=psnr(s['t1'],s['target']),ssim=ssim(s['t1'],s['target']))))
            for a,e in arms.items():
                _,t=second(model,task,e,s,a);r['metrics'][a]=dict(psnr=psnr(t,s['target']),ssim=ssim(t,s['target']))
            fit_train.append(r)
    del batches,shared,opts
    for e in arms.values():e.zero_grad(set_to_none=True)
    gc.collect();torch.cuda.empty_cache()
    ids=json.loads((ROOT/'config/validation64.json').read_text())['sample_ids'];vp=ROOT/'metrics/fit_validation64_partial.json'
    vr=json.loads(vp.read_text())['rows'] if vp.exists() else [];done={r['sample_id'] for r in vr}
    for sid in ids:
        if sid in done:continue
        guard();b=base.full_item(cfg,sid,DEVICE);vr.append(eval_one(cfg,model,task,arms,b));del b
        write(vp,dict(rows=vr));gc.collect();torch.cuda.empty_cache()
        if len(vr)%8==0:status('fit_validation',completed=len(vr),total=64)
    audit(model,task,source,arms)
    write(ROOT/'reports/diagnostic_summary.json',dict(direct=rows,fit_train=fit_train,fit_validation64=vr,steps=1000,
        role='exploratory fitting, not a deployment result; no test; no scientific gate',formal_initialization_not_reused=True))
    status('diagnostics_completed',direct_crops=16,fit_steps=1000,validation=64)
    del model,task,source,arms;gc.collect();torch.cuda.empty_cache()

def train(locked):
    if (ROOT/'diagnostics/training_complete.json').exists():return
    names=locked['arms'];cfg,model,task,raw,source,arms,opts=components(names)
    last=ROOT/'weights/last.pt';start=0;log=ROOT/'logs/training.csv';pid=locked['protocol_id']
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=False);restore(ck,arms,opts,pid);start=ck['step']
    else:
        assert not log.exists();base.save(ROOT/'weights/initial.pt',ckpt(arms,opts,0,pid));base.save(last,ckpt(arms,opts,0,pid))
    if log.exists():
        records=list(csv.DictReader(log.open()));keep=[r for r in records if int(r['step'])<=start]
        if len(keep)!=len(records):write(ROOT/'failures'/f'orphaned_rows_{int(time.time())}.json',records[len(keep):])
        with log.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(keep)
        assert len(keep)==start
    tick0=time.time();status('training',step=start,total=6000,arms=names)
    with log.open('a',newline='') as f:
        writer=None
        for step,batches in enumerate(micros(cfg,raw,start,6000),start+1):
            guard();tick=time.time();ids,loss,norm,hashes=base.update(model,task,arms,opts,batches,step);torch.cuda.synchronize();del batches
            row=dict(step=step,sample_ids=';'.join(ids),seconds=time.time()-tick)
            for a in names:row[a+'_loss']=loss[a];row[a+'_grad_norm']=norm[a]
            if writer is None:
                writer=csv.DictWriter(f,fieldnames=list(row))
                if start==0:writer.writeheader()
            writer.writerow(row);f.flush()
            if step%25==0 or step==1:status('training',step=step,total=6000,arms=names,losses=loss,seconds=row['seconds'])
            if step%100==0:
                state=ckpt(arms,opts,step,pid);base.save(last,state)
                if step%500==0:
                    audit(model,task,source,arms);write(ROOT/'diagnostics'/f'frozen_{step}.json',dict(passed=True,t1_hashes=hashes,time_unix=time.time()))
                    base.save(ROOT/'weights'/f'step_{step}.pt',state)
                del state
    rows=list(csv.DictReader(log.open()));assert [int(r['step']) for r in rows]==list(range(1,6001))
    seen={sid for r in rows for sid in r['sample_ids'].split(';')};assert len(seen)==5440
    audit(model,task,source,arms);base.save(ROOT/'weights/final.pt',ckpt(arms,opts,6000,pid))
    write(ROOT/'diagnostics/training_complete.json',dict(steps=6000,arms=names,unique_train_images=len(seen),crop_presentations_per_arm=12000,
        final_sha256=base.sha(ROOT/'weights/final.pt'),source_frozen_exact=True,last_segment_wall_seconds=time.time()-tick0))
    del model,task,source,arms,opts;gc.collect();torch.cuda.empty_cache()

def comparisons(rows):
    pairs=[]
    arms=list(rows[0]['metrics'])
    for a in arms:
        if a!='T1':pairs.append((a,'T1'))
    for r in ['R0','R1']:
        for a,b in [('D','N'),('P','D'),('O','P')]:
            if r+a in arms and r+b in arms:pairs.append((r+a,r+b))
    for c in ['D','P','O','N']:
        if 'R1'+c in arms and 'R0'+c in arms:pairs.append(('R1'+c,'R0'+c))
    out={}
    for a,b in pairs:
        out[a+'-'+b]={}
        for metric in ['psnr','ssim']:
            scenes={s:float(np.mean([r['metrics'][a][metric]-r['metrics'][b][metric] for r in rows if r['scene']==s])) for s in sorted({r['scene'] for r in rows})}
            out[a+'-'+b][metric]=dict(scene_equal=float(np.mean(list(scenes.values()))),image_equal=float(np.mean([r['metrics'][a][metric]-r['metrics'][b][metric] for r in rows])),scenes=scenes,scenes_positive=sum(v>0 for v in scenes.values()))
    return out

def evaluate(locked):
    cfg,model,task,raw,source,arms,opts=components(locked['arms']);ck=torch.load(ROOT/'weights/final.pt',map_location='cpu',weights_only=False)
    assert ck['step']==6000;restore(ck,arms,opts,locked['protocol_id']);del ck,opts;gc.collect();torch.cuda.empty_cache()
    summaries={}
    for split,count in [('validation872',872),('test188',188)]:
        old=json.loads((OLD/'metrics'/f'{split}.json').read_text());ora=json.loads((ORA/'metrics'/f'{split}.json').read_text())
        refs={r['sample_id']:r for r in old['rows']};oset={r['sample_id']:r for r in ora['rows']};assert set(refs)==set(oset) and len(refs)==count
        fp=ROOT/'metrics'/f'{split}.json';pp=ROOT/'metrics'/f'{split}_partial.json'
        if fp.exists():summaries[split]=json.loads(fp.read_text())['comparisons'];continue
        rows=json.loads(pp.read_text())['rows'] if pp.exists() else [];done={r['sample_id'] for r in rows};assert len(done)==len(rows)
        examples={min(r['sample_id'] for r in refs.values() if r['scene']==s) for s in {r['scene'] for r in refs.values()}}
        for sid,reference in refs.items():
            if sid in done:continue
            guard();batch=base.full_item(cfg,sid,DEVICE);r=eval_one(cfg,model,task,arms,batch,example=sid in examples);del batch
            assert r['t1_sha256']==reference['t1_sha256']==oset[sid]['t1_sha256'],sid
            assert r['metrics']['T1']==reference['metrics']['T1']==oset[sid]['metrics']['T1'],sid
            r['metrics'].update(R0D=reference['metrics']['D'],R0P=reference['metrics']['B'],R0O=oset[sid]['metrics']['O'])
            rows.append(r);write(pp,dict(rows=rows,original_size=True,step=6000));gc.collect();torch.cuda.empty_cache()
            if len(rows)%8==0:status('evaluating',split=split,completed=len(rows),total=count,arms=locked['arms'])
        assert len(rows)==count and len({r['sample_id'] for r in rows})==count
        summaries[split]=comparisons(rows);write(fp,dict(rows=rows,frames=count,comparisons=summaries[split],step=6000,protocol_id=locked['protocol_id'],original_size=True,tiling=False))
    audit(model,task,source,arms)
    write(ROOT/'reports/summary.json',dict(protocol_id=locked['protocol_id'],arms=locked['arms'],steps=6000,validation_frames=872,test_frames=188,
        comparisons=summaries,source_frozen_exact=True,oracle_uses_GT=True,seed=2026,multiseed_stability_claim=False,test_used_for_selection=False,final_sha256=base.sha(ROOT/'weights/final.pt')))
    status('worker_completed',arms=locked['arms'],validation=872,test=188)

def main():
    torch.set_num_threads(2);guard();p=verify();locked=preflight(p)
    diagnostic(locked);guard();train(locked);guard();evaluate(locked)
if __name__=='__main__':
    try:main()
    except Exception:
        write(ROOT/'failures'/f'worker_{int(time.time())}.json',dict(traceback=traceback.format_exc(),time_unix=time.time()))
        raise
