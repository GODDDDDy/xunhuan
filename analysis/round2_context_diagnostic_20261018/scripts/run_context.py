"""One fixed crop/full inference diagnosis; all experimental artifacts stay server-side."""
import contextlib, fcntl, gc, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
P33=PROJECT/'confidence_causal_completion_20261017'
P32=PROJECT/'confidence_correction_causal_20261017'
sys.path.insert(0,str(P33/'scripts'))
import completion_run as completion
prior=completion.prior
base=completion.base
DEVICE=torch.device('cuda:0')

def read(p):return json.loads(Path(p).read_text())
def write(p,x):base.json_save(Path(p),x)
def status(stage,**kw):
    obj=dict(stage=stage,time_unix=time.time(),**kw)
    write(ROOT/'status.json',obj);print(json.dumps(obj,ensure_ascii=False),flush=True)

def guard():
    assert time.time()<read(ROOT/'worker_registration.json')['deadline_unix'],'Fixed two-hour diagnostic limit reached'

def verify(cfg):
    original=dict(cfg);pid=original.pop('protocol_id')
    assert hashlib.sha256(json.dumps(original,sort_keys=True,separators=(',',':')).encode()).hexdigest()==pid
    for x in cfg['sources']+[cfg['reference']]:assert base.sha(x['path'])==x['sha256'],x['path']
    revision=read(ROOT/'config/implementation_revision_1.json') if (ROOT/'config/implementation_revision_1.json').exists() else None
    changes={} if revision is None else {r['path']:r for r in revision['replacements']}
    if revision:assert revision['protocol_id']==pid and set(changes)<={str(ROOT/'scripts/run_context.py'),str(ROOT/'scripts/audit_context.py')}
    for x in cfg['code_sources']:
        expected=x['sha256']
        if x['path'] in changes:
            change=changes[x['path']];assert change['original_sha256']==expected
            assert base.sha(change['preserved_original_path'])==expected
            expected=change['repaired_sha256']
        assert base.sha(x['path'])==expected,x['path']

def fingerprint(mod):
    h=hashlib.sha256()
    for name,t in mod.state_dict().items():
        h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def move_shared(s,device):
    q=dict(s)
    for k in ['p0','image','p1','t1','c','target']:q[k]=s[k].to(device)
    for k in ['f1','encoded']:q[k]={n:v.to(device) for n,v in s[k].items()}
    return q

class NoisePair:
    """Capture native full noise; the crop consumes exactly the captured latent tensor."""
    def __init__(self,task):self.task=task;self.native=task.model.p_sample_loop;self.reset()
    def reset(self):self.noise=None;self.scope='full';self.calls=[]
    def __enter__(self):
        def forward(condition,noise):
            native_sha=base.digest(noise)
            if self.noise is None:
                assert self.scope=='full';self.noise=noise.detach().clone()
            assert list(noise.shape)==[1,16,256] and noise.shape==self.noise.shape
            if self.scope=='full':assert torch.equal(noise,self.noise),'Legacy full T1/T2 noise differed'
            self.calls.append(dict(scope=self.scope,native_sha256=native_sha,used_sha256=base.digest(self.noise)))
            return self.native(condition,self.noise)
        self.task.model.p_sample_loop=forward
        return self
    def __exit__(self,*args):self.task.model.p_sample_loop=self.native
    def receipt(self):return dict(shape=list(self.noise.shape),used_sha256=base.digest(self.noise),calls=list(self.calls))

def metrics(x,y):
    from rgb_final_confirmation.evaluation import psnr,ssim
    # Historical paired evaluation computes both reductions on the GPU.
    # CPU copies preserve pixels but do not preserve reduction/conv roundoff.
    x,y=x.to(DEVICE),y.to(DEVICE)
    return dict(psnr=psnr(x,y),ssim=ssim(x,y))

def predict(cfg,model,task,arms,batch,pair,scope):
    pair.scope=scope
    phase='causal_fixed_crop' if batch['sample_id'][0].startswith('train/') else 'H_only_pair_eval'
    base.study.base.set_step_seed(2026,phase,0,batch['sample_id'])
    s=prior.first(model,task,batch,offload_encoder=True)
    outputs={'T1':s['t1'].cpu()}
    s=move_shared(s,'cpu');torch.cuda.empty_cache()
    for label,arm in [('F','R1F'),('FitP','R1P')]:
        q=move_shared(s,DEVICE)
        p,t=completion.second(model,task,arms[arm],q,arm)
        assert torch.isfinite(p).all() and torch.isfinite(t).all()
        assert float((p-q['p1']).abs().max())<=.020001
        outputs[label]=t.cpu()
        del p,t,q;torch.cuda.empty_cache()
    del s;gc.collect();torch.cuda.empty_cache()
    return outputs

def summarize(rows,pid):
    out=dict(protocol_id=pid,n_rows=len(rows),aggregates={})
    for split in ['train16','validation64']:
        selected=[r for r in rows if r['split']==split];dest=out['aggregates'][split]={}
        for area in ['center96','all128']:
            dest[area]={}
            for model in ['F','FitP']:
                scene_values={}
                for scene in sorted({r['scene'] for r in selected}):
                    values={k:[] for k in ['g_crop','g_full','difference','q_final','t1_gap']}
                    for r in [r for r in selected if r['scene']==scene]:
                        c=r['scores'][area]['crop'];f=r['scores'][area]['full']
                        gc_={m:c[model][m]-c['T1'][m] for m in ['psnr','ssim']}
                        gf={m:f[model][m]-f['T1'][m] for m in ['psnr','ssim']}
                        d={m:gc_[m]-gf[m] for m in gc_}
                        q={m:c[model][m]-f[model][m] for m in gc_}
                        t={m:c['T1'][m]-f['T1'][m] for m in gc_}
                        for k,v in zip(values,[gc_,gf,d,q,t]):values[k].append(v)
                    scene_values[scene]={k:{m:float(np.mean([v[m] for v in seq])) for m in ['psnr','ssim']} for k,seq in values.items()}
                dest[area][model]={k:{m:float(np.mean([s[k][m] for s in scene_values.values()])) for m in ['psnr','ssim']} for k in next(iter(scene_values.values()))}
                dest[area][model]['scenes']=scene_values
    return out

def run():
    lock=(ROOT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=read(ROOT/'config/locked.json');verify(cfg)
    reg=ROOT/'worker_registration.json'
    if not reg.exists():write(reg,dict(pid=os.getpid(),start_unix=time.time(),deadline_unix=time.time()+cfg['max_wall_seconds'],protocol_id=cfg['protocol_id']))
    else:
        v=read(reg);v.update(pid=os.getpid(),resumed_unix=time.time());write(reg,v)
    if (ROOT/'closeout.json').exists():assert read(ROOT/'closeout.json')['success'];return
    torch.set_num_threads(4)
    while True:
        guard()
        used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0]
        if int(used)>11000:break
        status('waiting_memory',free_mib=int(used));time.sleep(2)
    status('loading',protocol_id=cfg['protocol_id'])
    completion.guard=guard;prior.guard=guard;base.guard_gpu=guard;completion.MU=cfg['mu']
    execution,model,task,raw,source,arms,opts=completion.components(['R1F','R1P'])
    del opts
    ck33=torch.load(P33/'weights/final.pt',map_location='cpu',weights_only=False)
    ck32=torch.load(P32/'weights/fit_last.pt',map_location='cpu',weights_only=False)
    assert ck33['step']==6000 and ck32['step']==1000
    for arm,ck in [('R1F',ck33),('R1P',ck32)]:
        state=arms[arm].state_dict();state.update(ck['arms'][arm]);arms[arm].load_state_dict(state,strict=True)
        arms[arm].eval().requires_grad_(False)
    del ck33,ck32
    model.eval().requires_grad_(False);task.eval().requires_grad_(False)
    modules=dict(E1_H=model,PolarFree=task,**arms)
    before={k:fingerprint(m) for k,m in modules.items()}
    refs={r['sample_id']:r for r in read(cfg['reference']['path'])['rows']}
    path=ROOT/'metrics/rows.json';rows=read(path)['rows'] if path.exists() else [];done={r['sample_id'] for r in rows}
    with torch.no_grad(),NoisePair(task) as pair:
        for item in cfg['items']:
            if item['sample_id'] in done:continue
            guard();tick=time.time();pair.reset()
            batch=base.full_item(execution,item['sample_id'],DEVICE)
            assert list(batch['target'].shape)==[1,3,item['height'],item['width']]
            batch_cpu={k:v.cpu() if torch.is_tensor(v) else v for k,v in batch.items()}
            full=predict(execution,model,task,arms,batch,pair,'full')
            full_metrics={k:metrics(v,batch_cpu['target']) for k,v in full.items()}
            row={k:item[k] for k in ['sample_id','split','scene','top','left']}
            row.update(shape=list(batch['target'].shape),full_t1_sha256=base.digest(full['T1']),full_metrics=full_metrics,reference_t1_match=None,weights_unchanged=None,scores={})
            if item['split']=='validation64':
                ref=refs[item['sample_id']];assert row['full_t1_sha256']==ref['t1_sha256']
                if not rows:
                    from rgb_final_confirmation.evaluation import psnr,ssim
                    cpu_scores={k:dict(psnr=psnr(v,batch_cpu['target']),ssim=ssim(v,batch_cpu['target'])) for k,v in full.items() if k in ['T1','F']}
                    write(ROOT/'failures/metric_device_check.json',dict(sample_id=item['sample_id'],cpu_scores=cpu_scores,gpu_scores=full_metrics,
                        expected={k:ref['metrics'][v] for k,v in [('T1','T1'),('F','R1F')]},t1_hash_exact=True))
                for a,b in [('T1','T1'),('F','R1F')]:
                    for metric in ['psnr','ssim']:
                        assert abs(full_metrics[a][metric]-ref['metrics'][b][metric])<1e-10,(a,metric,full_metrics[a][metric],ref['metrics'][b][metric])
                row['reference_t1_match']=True
            y,x=item['top'],item['left']
            crop={k:v[...,y:y+128,x:x+128].contiguous().to(DEVICE) if torch.is_tensor(v) else v for k,v in batch_cpu.items()}
            del batch;torch.cuda.empty_cache()
            cropped=predict(execution,model,task,arms,crop,pair,'crop')
            assert len(pair.calls)==6
            row['noise']=pair.receipt()
            for area,(dy,dx,h,w) in cfg['scoring'].items():
                target=batch_cpu['target'][...,y+dy:y+dy+h,x+dx:x+dx+w]
                row['scores'][area]={'full':{},'crop':{}}
                for label in ['T1','F','FitP']:
                    row['scores'][area]['full'][label]=metrics(full[label][...,y+dy:y+dy+h,x+dx:x+dx+w],target)
                    row['scores'][area]['crop'][label]=metrics(cropped[label][...,dy:dy+h,dx:dx+w],target)
            row['seconds']=time.time()-tick;rows.append(row)
            write(path,dict(protocol_id=cfg['protocol_id'],rows=rows))
            status('evaluating',completed=len(rows),total=80,last=item['sample_id'],seconds=row['seconds'])
            if len(rows)==1:write(ROOT/'reports/preflight.json',dict(passed=True,legacy_full_exact=True,noise_control=pair.receipt(),seconds=row['seconds'],estimated_80_seconds=80*row['seconds']))
            del batch_cpu,crop,full,cropped,row;gc.collect();torch.cuda.empty_cache()
    after={k:fingerprint(m) for k,m in modules.items()};assert before==after
    for m in modules.values():assert all(not p.requires_grad and p.grad is None for p in m.parameters())
    for r in rows:r['weights_unchanged']=True
    write(path,dict(protocol_id=cfg['protocol_id'],rows=rows))
    write(ROOT/'reports/frozen.json',dict(before=before,after=after,all_equal=True,all_grad_none=True))
    write(ROOT/'reports/summary.json',summarize(rows,cfg['protocol_id']))
    verify(cfg)
    status('cpu_audit',completed=len(rows),total=80)
    subprocess.run([sys.executable,str(ROOT/'scripts/audit_context.py'),str(ROOT)],check=True)
    artifacts=[dict(path=str(p.relative_to(ROOT)),sha256=base.sha(p),bytes=p.stat().st_size) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix in ['.json','.py','.md'] and p.name not in ['status.json','artifact_manifest.json','closeout.json'] and '__pycache__' not in str(p)]
    write(ROOT/'artifact_manifest.json',dict(protocol_id=cfg['protocol_id'],files=artifacts))
    write(ROOT/'closeout.json',dict(success=True,completed_unix=time.time(),protocol_id=cfg['protocol_id'],rows=80,training_steps=0,independent_audit_verified=True))
    status('completed',completed=80,total=80,success=True)

if __name__=='__main__':
    try:run()
    except Exception:
        obj=dict(error=traceback.format_exc(),time_unix=time.time())
        failure_path=ROOT/'failures'/f'failure_{int(time.time())}.json'
        write(failure_path,obj);status('failed',error=obj['error'],failure_file=str(failure_path));raise
