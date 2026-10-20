"""Fixed 8-image, 100-update box-constrained GT search; no network training."""
import fcntl,gc,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent
P34=PROJECT/'round2_context_diagnostic_20261018';P33=PROJECT/'confidence_causal_completion_20261017'
sys.path.insert(0,str(P34/'scripts'))
import run_context as context
completion=context.completion;prior=context.prior;base=context.base;DEVICE=context.DEVICE
read=context.read;write=context.write
def status(stage,**kw):
    obj=dict(stage=stage,time_unix=time.time(),**kw);write(ROOT/'status.json',obj);print(json.dumps(obj),flush=True)
def guard():assert time.time()<read(ROOT/'worker_registration.json')['deadline_unix'],'Four-hour limit reached'
def verify(cfg):
    v=dict(cfg);pid=v.pop('protocol_id')
    assert hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()==pid
    for x in cfg['sources']+[cfg['reference']]:assert base.sha(x['path'])==x['sha256'],x['path']
    rev=read(ROOT/'config/implementation_revision_1.json') if (ROOT/'config/implementation_revision_1.json').exists() else None
    changes={} if rev is None else {r['path']:r for r in rev['replacements']}
    if rev:assert rev['protocol_id']==pid and set(changes)=={str(ROOT/'scripts/run_reachability.py'),str(ROOT/'scripts/audit_reachability.py')}
    for x in cfg['code_sources']:
        expected=x['sha256']
        if x['path'] in changes:
            change=changes[x['path']];assert change['original_sha256']==expected and base.sha(change['preserved_original_path'])==expected
            expected=change['repaired_sha256']
        assert base.sha(x['path'])==expected,x['path']
def load(cfg):
    completion.guard=guard;prior.guard=guard;base.guard_gpu=guard;completion.MU=cfg['mu']
    exe,model,task,raw,source,arms,opts=completion.components(['R1F']);del opts
    ck=torch.load(P33/'weights/final.pt',map_location='cpu',weights_only=False);assert ck['step']==6000
    state=arms['R1F'].state_dict();state.update(ck['arms']['R1F']);arms['R1F'].load_state_dict(state,strict=True);del ck
    for m in [model,task,arms['R1F']]:m.eval().requires_grad_(False)
    return exe,model,task,arms['R1F']
def first_state(exe,model,task,e,item):
    batch=base.full_item(exe,item['sample_id'],DEVICE)
    assert list(batch['target'].shape)==[1,3,item['height'],item['width']]
    base.study.base.set_step_seed(2026,'H_only_pair_eval',0,batch['sample_id'])
    with torch.no_grad():
        s=prior.first(model,task,batch,offload_encoder=True)
        p,t=completion.second(model,task,e,context.move_shared(s,DEVICE),'R1F')
        mf=context.metrics(t,s['target']);hF=base.digest(t)
        mt=context.metrics(s['t1'],s['target']);hT=base.digest(s['t1'])
    ref=REFERENCES[item['sample_id']];assert hT==ref['t1_sha256']
    for name,met in [('T1',mt),('R1F',mf)]:
        for key in ['psnr','ssim']:assert abs(met[key]-ref['metrics'][name][key])<1e-10
    # GT search needs no E2 encoder cache or task feedback features.
    q={k:s[k].detach() for k in ['p0','image','p1','t1','target']};q.update(cpu=s['cpu'],cuda=s['cuda'])
    del s,p,t,batch;gc.collect();torch.cuda.empty_cache()
    return q,dict(T1=mt,F=mf),hT,hF
def forward_delta(model,task,s,delta):
    p=(s['p1']+delta).clamp(0,1)
    with base._fork_from(s['cpu'],s['cuda']):
        t,features=base.study.task_forward(model,task,p,s['image'],s['p0'])
    del features
    return t

class BlockRecompute:
    """Only checkpoint original TransformerBlock forwards; no parameter/name changes."""
    def __init__(self,model,task):
        self.model=model;self.original_whole=model.activation_checkpoint
        self.blocks=[(m,m.forward) for m in task.model.net_g.modules() if type(m).__name__=='TransformerBlock']
        assert self.blocks,'No known original TransformerBlock modules'
    def enable(self,enabled):
        from torch.utils.checkpoint import checkpoint as replay
        self.model.activation_checkpoint=False if enabled else self.original_whole
        for module,native in self.blocks:
            if not enabled:module.forward=native;continue
            def call(*args,_native=native,**kw):
                if torch.is_grad_enabled() and any(torch.is_tensor(x) and x.requires_grad for x in args):
                    return replay(_native,*args,use_reentrant=False,preserve_rng_state=True,**kw)
                return _native(*args,**kw)
            module.forward=call

def check_recompute(exe,model,task,control,cfg):
    """One fixed tiny fixture: strict equality gate before the one full retry."""
    batch=base.full_item(exe,cfg['items'][0]['sample_id'],DEVICE)
    crop={k:v[...,:128,:128].contiguous() if torch.is_tensor(v) else v for k,v in batch.items()};del batch
    base.study.base.set_step_seed(2026,'reachability_execution_parity',0,crop['sample_id'])
    results=[]
    with context.NoisePair(task):
        s=prior.first(model,task,crop)
        for kind in ['zero','ramp']:
            initial=torch.zeros_like(s['p1']) if kind=='zero' else torch.linspace(-.005,.005,s['p1'].numel(),device=DEVICE).reshape_as(s['p1'])
            modes=[]
            for enabled in [False,True]:
                control.enable(enabled);d=initial.detach().clone().requires_grad_(True)
                t=forward_delta(model,task,s,d);loss=base.loss_fn(t,s['target']);loss.backward()
                modes.append((t.detach().cpu(),float(loss.detach()),d.grad.detach().cpu()));del d,t,loss
            a,b=modes;row=dict(delta=kind,output_equal=torch.equal(a[0],b[0]),loss_equal=a[1]==b[1],input_gradient_equal=torch.equal(a[2],b[2]),max_gradient_diff=float((a[2]-b[2]).abs().max()))
            results.append(row)
    control.enable(True)
    passed=all(x['output_equal'] and x['loss_equal'] and x['input_gradient_equal'] for x in results)
    write(ROOT/'reports/execution_parity.json',dict(passed=passed,fixture='fixed first image top-left128; no optimizer updates',blocks=len(control.blocks),cases=results,
          implementation='native torch.utils.checkpoint per TransformerBlock, use_reentrant=False, preserve_rng_state=True; no CPU offload'))
    assert passed,'One memory repair failed exact execution parity; stop without another scheme'
    del s,crop,initial,modes,a,b;gc.collect();torch.cuda.empty_cache()
def delta_stats(d):
    return dict(max_abs=float(d.abs().max()),mean_abs=float(d.abs().mean()),bound_fraction=float((d.abs()>=.02*(1-1e-4)).float().mean()),
                tv=float(sum(d.diff(dim=n).abs().mean()/2 for n in [-1,-2])))
def checkpoint(path,step,delta,opt,best_delta,best_loss,best_step,noise,first_grad,noise_receipt):
    base.save(path,dict(step=step,delta=delta.detach().cpu(),optimizer=opt.state_dict(),best_delta=best_delta.cpu(),best_loss=best_loss,
                        best_step=best_step,noise=noise.cpu(),first_grad_norm=first_grad,rng=base.rng_state(),noise_receipt=noise_receipt))
def summarize(rows,pid):
    out=dict(protocol_id=pid,n_rows=len(rows),absolute={},comparisons={})
    for name in ['T1','F','best','final']:
        out['absolute'][name]={m:float(np.mean([r['metrics'][name][m] for r in rows])) for m in ['psnr','ssim']}
    for name in ['F','best','final']:
        v={}
        for m in ['psnr','ssim']:
            vals=[r['metrics'][name][m]-r['metrics']['T1'][m] for r in rows]
            v[m]=dict(mean=float(np.mean(vals)),positive_ids=[r['sample_id'] for r,x in zip(rows,vals) if x>0])
        v['action_reference_pass']=v['psnr']['mean']>=.05 and v['ssim']['mean']>=0 and len(v['psnr']['positive_ids'])>=6
        out['comparisons'][name+'_minus_T1']=v
    return out
def run():
    lock=(ROOT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=read(ROOT/'config/locked.json');verify(cfg);rp=ROOT/'worker_registration.json'
    if rp.exists():
        reg=read(rp);reg.update(pid=os.getpid(),resumed_unix=time.time())
    else:reg=dict(pid=os.getpid(),start_unix=time.time(),deadline_unix=time.time()+cfg['max_wall_seconds'],protocol_id=cfg['protocol_id'])
    write(rp,reg)
    if (ROOT/'closeout.json').exists():assert read(ROOT/'closeout.json')['success'];return
    torch.set_num_threads(4)
    while True:
        guard();free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
        if free>11000:break
        status('waiting_memory',free_mib=free);time.sleep(2)
    status('loading');exe,model,task,e=load(cfg);modules=dict(E1_H=model,PolarFree=task,F=e)
    before={k:context.fingerprint(m) for k,m in modules.items()}
    if (ROOT/'config/implementation_revision_1.json').exists():
        control=BlockRecompute(model,task)
        if not (ROOT/'reports/execution_parity.json').exists():check_recompute(exe,model,task,control,cfg)
        else:assert read(ROOT/'reports/execution_parity.json')['passed'];control.enable(True)
    global REFERENCES
    REFERENCES={r['sample_id']:r for r in read(cfg['reference']['path'])['rows']}
    path=ROOT/'metrics/rows.json';rows=read(path)['rows'] if path.exists() else [];done={r['sample_id'] for r in rows}
    # Full-resolution real backward on the predeclared largest image, no optimizer update.
    if not (ROOT/'reports/preflight.json').exists():
        item=next(x for x in cfg['items'] if x['sample_id']==cfg['preflight_sample']);guard()
        with context.NoisePair(task) as pair:
            s,met,ht,hf=first_state(exe,model,task,e,item)
            delta=torch.zeros_like(s['p1'],requires_grad=True)
            with torch.no_grad():zero=forward_delta(model,task,s,delta);assert torch.equal(zero,s['t1']);del zero
            torch.cuda.reset_peak_memory_stats();tick=time.time();t=forward_delta(model,task,s,delta);loss=base.loss_fn(t,s['target']);loss.backward();torch.cuda.synchronize()
            seconds=time.time()-tick;norm=float(delta.grad.norm());assert norm>0 and np.isfinite(norm)
            remaining=reg['deadline_unix']-time.time();estimate=seconds*800*1.2+600
            write(ROOT/'reports/preflight.json',dict(passed=estimate<remaining,sample_id=item['sample_id'],zero_identity=True,first_grad_norm=norm,
                  backward_seconds=seconds,estimated_remaining_seconds=estimate,available_seconds=remaining,peak_mib=torch.cuda.max_memory_allocated()/2**20,updates_discarded=True))
            assert estimate<remaining,'Fixed 8x100 scope cannot fit remaining four-hour budget'
            del s,delta,t,loss
        gc.collect();torch.cuda.empty_cache()
    for ix,item in enumerate(cfg['items']):
        if item['sample_id'] in done:continue
        guard();status('optimizing',image=ix+1,total_images=8,step=0,sample_id=item['sample_id'])
        with context.NoisePair(task) as pair:
            s,met,ht,hf=first_state(exe,model,task,e,item)
            delta=torch.zeros_like(s['p1'],requires_grad=True);opt=torch.optim.Adam([delta],lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
            cp=ROOT/'variables'/f'{ix:02d}_last.pt';curvepath=ROOT/'logs'/f'{ix:02d}_curve.json';start=0;best_loss=float('inf');best_delta=delta.detach().cpu().clone();best_step=0;first_grad=None
            curve=read(curvepath) if curvepath.exists() else []
            if cp.exists():
                ck=torch.load(cp,map_location='cpu',weights_only=False);start=ck['step'];delta.data.copy_(ck['delta'].to(DEVICE));opt.load_state_dict(ck['optimizer'])
                best_delta=ck['best_delta'];best_loss=ck['best_loss'];best_step=ck['best_step'];first_grad=ck['first_grad_norm'];assert torch.equal(ck['noise'],pair.noise.cpu())
                pair.calls=ck['noise_receipt']['calls']+pair.calls;base.restore_rng(ck['rng']);del ck
                orphan=[x for x in curve if x['step']>=start]
                if orphan:write(ROOT/'failures'/f'orphan_curve_{ix:02d}_{int(time.time())}.json',orphan)
                curve=[x for x in curve if x['step']<start]
            elif curve:
                write(ROOT/'failures'/f'uncheckpointed_curve_{ix:02d}_{int(time.time())}.json',curve);curve=[]
            for step in range(start,101):
                guard();opt.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(step<100):t=forward_delta(model,task,s,delta);loss=base.loss_fn(t,s['target'])
                value=float(loss.detach());assert np.isfinite(value)
                if step==0:assert torch.equal(t,s['t1'])
                if value<best_loss:best_loss=value;best_delta=delta.detach().cpu().clone();best_step=step
                curve.append(dict(step=step,loss=value));write(curvepath,curve)
                if step==100:
                    final_metrics=context.metrics(t,s['target']);final_hash=base.digest(t);final_loss=value;del t,loss;break
                loss.backward();norm=float(delta.grad.norm());assert np.isfinite(norm)
                if step==0:first_grad=norm;assert norm>0
                opt.step()
                with torch.no_grad():delta.clamp_(-.02,.02)
                del t,loss
                if (step+1)%10==0:
                    checkpoint(cp,step+1,delta,opt,best_delta,best_loss,best_step,pair.noise,first_grad,pair.receipt())
                    status('optimizing',image=ix+1,total_images=8,step=step+1,sample_id=item['sample_id'])
            with torch.no_grad():
                best=forward_delta(model,task,s,best_delta.to(DEVICE));bm=context.metrics(best,s['target']);best_hash=base.digest(best)
                assert abs(float(base.loss_fn(best,s['target']))-best_loss)<1e-9
                ds=dict(best=delta_stats(best_delta),final=delta_stats(delta))
            checkpoint(cp,100,delta,opt,best_delta,best_loss,best_step,pair.noise,first_grad,pair.receipt())
            row=dict(sample_id=item['sample_id'],scene=item['scene'],shape=[1,3,item['height'],item['width']],steps=100,
                     t1_sha256=ht,F_sha256=hf,best_sha256=best_hash,final_sha256=final_hash,noise_sha256=base.digest(pair.noise),noise_receipt=pair.receipt(),zero_identity=True,
                     first_grad_norm=first_grad,best_step=best_step,best_loss=best_loss,initial_loss=curve[0]['loss'],final_loss=final_loss,
                     metrics=dict(**met,best=bm,final=final_metrics),delta=ds,variables_path=str(cp),curve_path=str(curvepath))
            rows.append(row);write(path,dict(protocol_id=cfg['protocol_id'],rows=rows));status('image_completed',completed=len(rows),total=8,sample_id=item['sample_id'])
            del s,delta,opt,best_delta,best,curve;gc.collect();torch.cuda.empty_cache()
    after={k:context.fingerprint(m) for k,m in modules.items()};assert before==after
    assert all(not p.requires_grad and p.grad is None for m in modules.values() for p in m.parameters())
    write(ROOT/'reports/frozen.json',dict(all_equal=True,all_grad_none=True,before=before,after=after))
    write(ROOT/'reports/summary.json',summarize(rows,cfg['protocol_id']));verify(cfg)
    subprocess.run([sys.executable,str(ROOT/'scripts/audit_reachability.py'),str(ROOT)],check=True)
    files=[dict(path=str(p.relative_to(ROOT)),sha256=base.sha(p),bytes=p.stat().st_size) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix in ['.json','.py','.md','.pt'] and p.name not in ['status.json','artifact_manifest.json','closeout.json'] and '__pycache__' not in str(p)]
    write(ROOT/'artifact_manifest.json',dict(protocol_id=cfg['protocol_id'],files=files));write(ROOT/'closeout.json',dict(success=True,completed_unix=time.time(),images=8,updates_per_image=100,new_network_training=False,protocol_id=cfg['protocol_id']))
    status('completed',images=8,updates_per_image=100)
if __name__=='__main__':
    try:run()
    except Exception:
        error=traceback.format_exc();fp=ROOT/'failures'/f'failure_{int(time.time())}.json';write(fp,dict(error=error,time_unix=time.time()));status('failed',error=error,failure_file=str(fp));raise
