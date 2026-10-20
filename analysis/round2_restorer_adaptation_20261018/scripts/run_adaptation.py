"""Experiment36: two matched restoration adaptations; reuse frozen33 data/forward/loss."""
import contextlib,copy,csv,fcntl,gc,hashlib,json,math,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
P33=PROJECT/'confidence_causal_completion_20261017'
sys.path.insert(0,str(P33/'scripts'))
import completion_run as completion
prior=completion.prior
base=completion.base
DEVICE=torch.device('cuda:0')
def read(p):return json.loads(Path(p).read_text())
def write(p,v):base.json_save(Path(p),v)
def status(stage,**kw):
    v=dict(stage=stage,time_unix=time.time(),**kw);write(ROOT/'status.json',v);print(json.dumps(v),flush=True)
def guard():return None
def tree_equal(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(tree_equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(tree_equal(v,w) for v,w in zip(a,b))
    return a==b
def verify(cfg):
    obj=dict(cfg);pid=obj.pop('protocol_id')
    assert hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()==pid
    replacements={}
    rp=ROOT/'config/implementation_revisions.json'
    for revision in (read(rp) if rp.exists() else []):
        assert revision['protocol_id']==pid
        for r in revision['replacements']:
            assert base.sha(r['preserved_original_path'])==r['original_sha256']
            replacements[r['path']]=r
    for r in cfg['sources']+cfg['code_sources']:
        expected=r['sha256']
        if r['path'] in replacements:
            change=replacements[r['path']];assert change['original_sha256']==expected
            expected=change['repaired_sha256']
        assert base.sha(r['path'])==expected,r['path']
def fingerprint(m,frozen_only=False):
    params=dict(m.named_parameters());h=hashlib.sha256()
    for name,v in m.state_dict().items():
        if frozen_only and name in params and params[name].requires_grad:continue
        h.update(name.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def frozen_hashes(model,task,e,ds):
    return dict(E1_H=fingerprint(model),D1_prior=fingerprint(task),F=fingerprint(e),
                **{a+'_prior':fingerprint(d,True) for a,d in ds.items()})
def assert_frozen(model,task,e,ds):
    for m in (model,task,e):
        assert all(not p.requires_grad and p.grad is None for p in m.parameters())
    for d in ds.values():
        for n,p in d.named_parameters():
            assert p.requires_grad==n.startswith('model.net_g.'),n
            if not p.requires_grad:assert p.grad is None,n
def components(cfg):
    completion.guard=prior.guard=base.guard_gpu=guard
    completion.MU=cfg['mu']
    _,model,task,_,source,arms,opts=completion.components(['R1F'])
    del opts,source
    e=arms.pop('R1F')
    ck=torch.load(P33/'weights/final.pt',map_location='cpu',weights_only=False)
    assert ck['step']==6000
    state=e.state_dict();state.update(ck['arms']['R1F']);e.load_state_dict(state,strict=True)
    del ck,state
    model.eval().requires_grad_(False);task.eval().requires_grad_(False);e.eval().requires_grad_(False)
    ds={a:copy.deepcopy(task).eval().requires_grad_(False) for a in ['A','B']}
    for d in ds.values():d.net_g.requires_grad_(True)
    opts={a:torch.optim.Adam(d.net_g.parameters(),lr=cfg['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0) for a,d in ds.items()}
    execution=read(cfg['train_cfg_path'])
    raw=base.study.base.build_base_config(execution)
    return execution,raw,model,task,e,ds,opts
class NoiseProof:
    def __init__(self,task,ds):self.tasks=dict(D1=task,**ds);self.native={};self.calls=[];self.noise=None
    def __enter__(self):
        for a,t in self.tasks.items():
            self.native[a]=t.model.p_sample_loop
            def forward(condition,noise,a=a):
                if self.noise is None:
                    assert a=='D1';self.noise=noise.detach().clone()
                assert torch.equal(noise,self.noise),('Unpaired actual noise',a)
                self.calls.append(dict(stage=a,sha256=base.digest(noise)))
                return self.native[a](condition,noise)
            t.model.p_sample_loop=forward
        return self
    def __exit__(self,*args):
        for a,t in self.tasks.items():t.model.p_sample_loop=self.native[a]
def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)
def snapshot(ds,opts,step,cfg):
    return dict(step=step,protocol_id=cfg['protocol_id'],schedule_sha256=cfg['schedule_sha256'],
        net_g={a:{n:v.detach().cpu().clone() for n,v in d.net_g.state_dict().items()} for a,d in ds.items()},
        optimizers={a:cpu_tree(o.state_dict()) for a,o in opts.items()},rng=base.rng_state())
def restore(ck,ds,opts,cfg):
    assert ck['protocol_id']==cfg['protocol_id'] and ck['schedule_sha256']==cfg['schedule_sha256']
    for a,d in ds.items():
        d.net_g.load_state_dict(ck['net_g'][a],strict=True)
        opts[a].load_state_dict(ck['optimizers'][a]);opts[a].zero_grad(set_to_none=True)
    base.restore_rng(ck['rng'])
def update(execution,raw,model,task,e,ds,opts,batches,step,same_input=False):
    values={a:[] for a in ds};ids=[];noise_records=[];t1_hashes=[]
    for opt in opts.values():opt.zero_grad(set_to_none=True)
    for batch in batches:
        ids+=batch['sample_id']
        base.study.base.set_step_seed(2026,'C_feedback_polar_training',step,batch['sample_id'])
        with NoiseProof(task,ds) as proof:
            with torch.no_grad():
                s=prior.first(model,task,batch)
                p2,old=completion.second(model,task,e,s,'R1F')
            assert float((p2-s['p1']).abs().max())<=.020001
            if step==1 or step%500==0:t1_hashes.append(base.digest(s['t1']))
            p1,image,p0,cpu,cuda=s['p1'],s['image'],s['p0'],s['cpu'],s['cuda']
            del s,old
            for a,d in ds.items():
                p=p1 if a=='A' or same_input else p2
                with base._fork_from(cpu,cuda):
                    t,features=base.study.task_forward(model,d,p,image,p0)
                del features
                loss=base.loss_fn(t,batch['target']);assert torch.isfinite(loss)
                (loss/2).backward();values[a].append(float(loss.detach()))
                del loss,t
            assert set(c['stage'] for c in proof.calls)=={'D1','A','B'}
            noise_records.append(dict(sample_id=batch['sample_id'][0],noise_sha256=base.digest(proof.noise),calls=len(proof.calls)))
        del p1,p2,image,p0
    norms={}
    for a,d in ds.items():
        norms[a]=float(torch.nn.utils.clip_grad_norm_(d.net_g.parameters(),1,error_if_nonfinite=True))
        assert norms[a]>0 and math.isfinite(norms[a])
        opts[a].step()
    return dict(sample_ids=';'.join(ids),A_loss=float(np.mean(values['A'])),B_loss=float(np.mean(values['B'])),
                A_grad_norm=norms['A'],B_grad_norm=norms['B']),noise_records,t1_hashes
def move_shared(s,device):
    q=dict(s)
    for k in ['p0','image','p1','t1','c','target']:q[k]=s[k].to(device)
    for k in ['f1','encoded']:q[k]={n:v.to(device) for n,v in s[k].items()}
    return q
@torch.no_grad()
def eval_one(execution,model,task,e,ds,batch,ref,save_arrays=False,identity=False):
    from rgb_final_confirmation.evaluation import psnr,ssim
    for d in ds.values():d.cpu()
    torch.cuda.empty_cache()
    base.study.base.set_step_seed(2026,'H_only_pair_eval',0,batch['sample_id'])
    arrays={}
    with NoiseProof(task,ds) as proof:
        s=prior.first(model,task,batch,offload_encoder=True)
        t1=s['t1'].cpu()
        metrics={'T1':dict(psnr=psnr(s['t1'],batch['target']),ssim=ssim(s['t1'],batch['target']))}
        assert base.digest(t1)==ref['t1_sha256'] and metrics['T1']==ref['metrics']['T1']
        shared=move_shared(s,'cpu');del s;torch.cuda.empty_cache()
        q=move_shared(shared,DEVICE)
        p2,old=completion.second(model,task,e,q,'R1F')
        metrics['F']=dict(psnr=psnr(old,batch['target']),ssim=ssim(old,batch['target']))
        assert all(abs(metrics['F'][m]-ref['metrics']['R1F'][m])<1e-10 for m in ['psnr','ssim'])
        old_cpu=old.cpu();p2_cpu=p2.cpu();del q,p2,old;torch.cuda.empty_cache()
        images={}
        for a,d in ds.items():
            d.to(DEVICE)
            p=shared['p1'].to(DEVICE) if a=='A' else p2_cpu.to(DEVICE)
            with base._fork_from(shared['cpu'],shared['cuda']):
                t,features=base.study.task_forward(model,d,p,shared['image'].to(DEVICE),shared['p0'].to(DEVICE))
            metrics[a]=dict(psnr=psnr(t,batch['target']),ssim=ssim(t,batch['target']))
            images[a]=t.cpu()
            if identity:assert torch.equal(images[a],t1 if a=='A' else old_cpu),(a,'step0 identity')
            del t,features,p;d.cpu();torch.cuda.empty_cache()
        row=dict(sample_id=batch['sample_id'][0],scene=batch['scene'][0],shape=list(batch['target'].shape),
            t1_sha256=base.digest(t1),a_sha256=base.digest(images['A']),b_sha256=base.digest(images['B']),
            metrics=metrics,noise_sha256=base.digest(proof.noise),noise_calls=proof.calls,reference_exact=True,original_size=True)
        if save_arrays:
            arrays=dict(t1=t1.numpy(),F=old_cpu.numpy(),A=images['A'].numpy(),B=images['B'].numpy(),
                p1=shared['p1'].numpy(),p2=p2_cpu.numpy(),predicted_c=shared['c'].numpy(),
                target=batch['target'].cpu().numpy())
            np.savez(ROOT/'examples'/(row['sample_id'].replace('/','_')+'.npz'),**arrays)
    for d in ds.values():d.to(DEVICE)
    return row
def comparisons(rows):
    out={}
    for a,b in [('B','A'),('B','T1'),('A','T1'),('F','T1'),('B','F')]:
        out[a+'-'+b]={}
        for m in ['psnr','ssim']:
            by={scene:float(np.mean([r['metrics'][a][m]-r['metrics'][b][m] for r in rows if r['scene']==scene])) for scene in sorted({r['scene'] for r in rows})}
            out[a+'-'+b][m]=dict(scene_equal=float(np.mean(list(by.values()))),image_equal=float(np.mean([r['metrics'][a][m]-r['metrics'][b][m] for r in rows])),
                scenes=by,scenes_positive=sum(v>0 for v in by.values()))
    return out
def preflight(cfg):
    fp=ROOT/'config/preflight.json'
    if fp.exists():assert read(fp)['passed'];return
    status('preflight_loading')
    execution,raw,model,task,e,ds,opts=components(cfg)
    before=frozen_hashes(model,task,e,ds)
    assert tree_equal(ds['A'].net_g.state_dict(),ds['B'].net_g.state_dict())
    ptrs=[{p.data_ptr() for p in m.parameters()} for m in [task,*ds.values()]]
    assert all(not ptrs[i]&ptrs[j] for i in range(3) for j in range(i))
    initial=snapshot(ds,opts,0,cfg)
    batch=next(prior.micros(execution,raw,0,1))[0]
    base.study.base.set_step_seed(2026,'C_feedback_polar_training',1,batch['sample_id'])
    with torch.no_grad(),NoiseProof(task,ds) as proof:
        s=prior.first(model,task,batch);p2,old=completion.second(model,task,e,s,'R1F')
        for a,d in ds.items():
            with base._fork_from(s['cpu'],s['cuda']):
                t,f=base.study.task_forward(model,d,s['p1'] if a=='A' else p2,s['image'],s['p0'])
            assert torch.equal(t,s['t1'] if a=='A' else old);del t,f
        noise_sample=base.digest(proof.noise)
    del batch,s,p2,old
    batches=next(prior.micros(execution,raw,0,1))
    update(execution,raw,model,task,e,ds,opts,batches,1,same_input=True)
    assert tree_equal(ds['A'].net_g.state_dict(),ds['B'].net_g.state_dict())
    assert tree_equal(opts['A'].state_dict(),opts['B'].state_dict())
    restore(initial,ds,opts,cfg)
    times=[]
    for step,batches in enumerate(prior.micros(execution,raw,0,3),1):
        tick=time.time();update(execution,raw,model,task,e,ds,opts,batches,step)
        torch.cuda.synchronize();times.append(time.time()-tick)
    snap=snapshot(ds,opts,3,cfg)
    base.save(ROOT/'diagnostics/preflight_checkpoint.pt',snap)
    batches=next(prior.micros(execution,raw,3,4))
    update(execution,raw,model,task,e,ds,opts,batches,4)
    expected=snapshot(ds,opts,4,cfg)
    replay=torch.load(ROOT/'diagnostics/preflight_checkpoint.pt',map_location='cpu',weights_only=False)
    restore(replay,ds,opts,cfg)
    update(execution,raw,model,task,e,ds,opts,batches,4)
    assert tree_equal(expected,snapshot(ds,opts,4,cfg))
    restore(initial,ds,opts,cfg);del snap,replay,expected,batches,initial
    assert_frozen(model,task,e,ds);assert before==frozen_hashes(model,task,e,ds)
    # Empty Adam state before full inference, just as in the final evaluation.
    refs=read(cfg['reference_metrics_path'])['rows']
    ref=max(refs,key=lambda r:np.prod(r['shape'][-2:]))
    batch=base.full_item(execution,ref['sample_id'],DEVICE);tick=time.time()
    row=eval_one(execution,model,task,e,ds,batch,ref,identity=True)
    torch.cuda.synchronize();frame_seconds=time.time()-tick
    seconds=float(np.median(times[1:]))
    obj=dict(passed=True,protocol_id=cfg['protocol_id'],noise_exact=True,identity_exact=True,resume_exact=True,
        same_input_update_exact=True,source_frozen_exact=True,initial_no_alias=True,all_preflight_updates_discarded=True,
        train_step_seconds=seconds,full_frame_seconds=frame_seconds,
        estimated_compute_hours=1.2*(3000*seconds+64*frame_seconds)/3600,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),largest_shape=row['shape'],actual_noise_sha256=noise_sample,
        trainable_counts={a:sum(p.numel() for p in d.net_g.parameters()) for a,d in ds.items()})
    write(fp,obj);status('preflight_passed',**{k:obj[k] for k in ['estimated_compute_hours','train_step_seconds','full_frame_seconds','trainable_counts']})
    del model,task,e,ds,opts;gc.collect();torch.cuda.empty_cache()
def train(cfg):
    if (ROOT/'diagnostics/training_complete.json').exists():return
    status('training_loading')
    execution,raw,model,task,e,ds,opts=components(cfg)
    before=frozen_hashes(model,task,e,ds)
    last=ROOT/'weights/last.pt';start=0;log=ROOT/'logs/training.csv'
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=False);restore(ck,ds,opts,cfg);start=ck['step'];del ck
    else:
        assert not log.exists()
        ck=snapshot(ds,opts,0,cfg);base.save(ROOT/'weights/initial.pt',ck);base.save(last,ck);del ck
    if log.exists() and log.stat().st_size:
        rows=list(csv.DictReader(log.open()));keep=[r for r in rows if int(r['step'])<=start]
        if len(keep)!=len(rows):write(ROOT/'failures'/('orphaned_rows_'+str(int(time.time()))+'.json'),rows[len(keep):])
        with log.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(keep)
        assert len(keep)==start
    step=start
    status('training',step=start,total=3000)
    with log.open('a',newline='') as f:
        writer=None
        for step,batches in enumerate(prior.micros(execution,raw,start,3000),start+1):
            tick=time.time();record,noises,hashes=update(execution,raw,model,task,e,ds,opts,batches,step)
            torch.cuda.synchronize()
            record=dict(step=step,**record,seconds=time.time()-tick)
            if writer is None:
                writer=csv.DictWriter(f,fieldnames=list(record))
                if start==0:writer.writeheader()
            writer.writerow(record);f.flush()
            if step==1 or step%100==0:
                ck=snapshot(ds,opts,step,cfg);base.save(last,ck);del ck
                write(ROOT/'diagnostics'/('paired_noise_'+str(step)+'.json'),dict(step=step,rows=noises,t1_hashes=hashes))
            if step%500==0:
                assert_frozen(model,task,e,ds);assert before==frozen_hashes(model,task,e,ds)
            if step==1 or step%25==0:status('training',step=step,total=3000,seconds=record['seconds'],A_loss=record['A_loss'],B_loss=record['B_loss'])
    assert step==3000
    assert_frozen(model,task,e,ds);after=frozen_hashes(model,task,e,ds);assert before==after
    write(ROOT/'diagnostics/frozen_final.json',dict(before=before,after=after,passed=True))
    ck=snapshot(ds,opts,3000,cfg);base.save(ROOT/'weights/final.pt',ck);base.save(last,ck)
    write(ROOT/'diagnostics/training_complete.json',dict(steps=3000,protocol_id=cfg['protocol_id'],final_sha256=base.sha(ROOT/'weights/final.pt')))
    del ck,model,task,e,ds,opts;gc.collect();torch.cuda.empty_cache()
def evaluate(cfg):
    execution,raw,model,task,e,ds,opts=components(cfg)
    ck=torch.load(ROOT/'weights/final.pt',map_location='cpu',weights_only=False);assert ck['step']==3000
    restore(ck,ds,opts,cfg);del opts,ck,raw;gc.collect();torch.cuda.empty_cache()
    for d in ds.values():d.zero_grad(set_to_none=True)
    before=frozen_hashes(model,task,e,ds)
    refs={r['sample_id']:r for r in read(cfg['reference_metrics_path'])['rows']}
    path=ROOT/'metrics/validation64.json'
    rows=read(path)['rows'] if path.exists() else []
    assert [r['sample_id'] for r in rows]==cfg['validation_ids'][:len(rows)]
    for sid in cfg['validation_ids'][len(rows):]:
        tick=time.time();batch=base.full_item(execution,sid,DEVICE)
        r=eval_one(execution,model,task,e,ds,batch,refs[sid],save_arrays=True)
        r['seconds']=time.time()-tick;rows.append(r)
        write(path,dict(protocol_id=cfg['protocol_id'],step=3000,rows=rows,frames=len(rows),original_size=True,tiling=False))
        status('evaluating',split='validation64',completed=len(rows),total=64,seconds=r['seconds'])
        del batch;gc.collect();torch.cuda.empty_cache()
    comp=comparisons(rows)
    gate=lambda k:comp[k]['psnr']['scene_equal']>=.05 and comp[k]['ssim']['scene_equal']>=0 and comp[k]['psnr']['scenes_positive']>=6
    summary=dict(protocol_id=cfg['protocol_id'],steps=3000,frames=64,exploratory_only=True,test_used=False,
        comparisons=comp,promotion_pass=bool(gate('B-A') and gate('B-T1')),
        difference_of_differences={m:comp['B-A'][m]['scene_equal']-comp['F-T1'][m]['scene_equal'] for m in ['psnr','ssim']},
        limitations=['single_seed','repeated_validation_exploration','independent_D2_weights','fixed_F_trained_on_same_training_data'])
    assert before==frozen_hashes(model,task,e,ds)
    write(ROOT/'reports/summary.json',summary)
    status('screen_worker_completed',promotion_pass=summary['promotion_pass'])
def finish(cfg):
    import subprocess
    status('independent_audit')
    subprocess.run([sys.executable,str(ROOT/'scripts/audit_adaptation.py'),'--root',str(ROOT)],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    assert read(ROOT/'reports/independent_audit.json')['passed']
    verify(cfg)
    files=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and p.suffix not in ['.pyc','.tmp'] and p.name not in ['worker.log','controller.log','status.json','artifact_manifest.json','closeout.json','controller_heartbeat.json'] and 'cache' not in p.relative_to(ROOT).parts:
            files.append(dict(path=str(p.relative_to(ROOT)),sha256=base.sha(p),bytes=p.stat().st_size))
    write(ROOT/'artifact_manifest.json',dict(protocol_id=cfg['protocol_id'],files=files,server_only=True))
    assert all(base.sha(ROOT/r['path'])==r['sha256'] for r in files)
    summary=read(ROOT/'reports/summary.json')
    close=dict(success=True,exploratory_only=True,steps=3000,validation_frames=64,test_frames=0,
        independent_audit_verified=True,promotion_pass=summary['promotion_pass'],protocol_id=cfg['protocol_id'],completed_unix=time.time())
    write(ROOT/'reports/final_review_closeout.json',close);write(ROOT/'closeout.json',close)
    status('screen_completed',**{k:close[k] for k in ['success','steps','validation_frames','promotion_pass']})
def main():
    lock=(ROOT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=read(ROOT/'config/locked.json');verify(cfg);torch.set_num_threads(2)
    if (ROOT/'closeout.json').exists():
        assert read(ROOT/'closeout.json')['success'];return
    preflight(cfg);train(cfg);evaluate(cfg);finish(cfg)
if __name__=='__main__':
    try:main()
    except Exception:
        receipt=dict(error=traceback.format_exc(),time_unix=time.time())
        path=ROOT/'failures'/('worker_'+str(int(time.time()))+'.json')
        write(path,receipt);status('failed',failure_file=str(path));raise
