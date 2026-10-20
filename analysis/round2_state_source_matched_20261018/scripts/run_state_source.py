"""Experiment37: paired native-crop/full-state local refinement, fixed sources.
Only the tiny local RGB heads are optimized. Data, weights and metrics stay here.
"""
from __future__ import annotations
import copy,csv,fcntl,gc,hashlib,json,math,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'c_feedback_polar_loop_20261015/scripts'))
import c_loop_run as base
DEVICE=torch.device('cuda:0')
MU=0.4459386889966336

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):base.json_save(Path(p),v)
def sha(p):return base.sha(Path(p))
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def status(stage,**kw):
    out=dict(stage=stage,time_unix=time.time(),**kw);write(ROOT/'status.json',out);print(json.dumps(out),flush=True)
def guard():
    p=read(ROOT/'worker_registration.json')
    assert time.time()<float(p['deadline_unix']),'Experiment37 original12h worker deadline reached; preserve state'
def verify(proposal):
    q=dict(proposal);sid=q.pop('scientific_id');assert canonical(q)==sid
    revisions={}
    rp=ROOT/'config/implementation_revisions.json'
    for revision in read(rp) if rp.exists() else []:
        assert revision.get('scientific_id',revision.get('protocol_id'))==sid
        for r in revision['replacements']:
            assert sha(r['preserved_original_path'])==r['original_sha256']
            revisions[r['path']]=r
    codes={r['path']:r for r in proposal['code_sources']}
    assert set(revisions)<=set(codes)
    for r in proposal['source_receipts']:
        assert sha(r['path'])==r['sha256'],r['path']
    for path,r in codes.items():
        expected=r['sha256']
        if path in revisions:
            change=revisions[path];assert change['original_sha256']==expected;expected=change['repaired_sha256']
        assert sha(path)==expected,path

def tree_equal(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(tree_equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(tree_equal(v,w) for v,w in zip(a,b))
    return a==b

def cpu_tree(v):
    if torch.is_tensor(v):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu_tree(w) for k,w in v.items()}
    if isinstance(v,list):return [cpu_tree(w) for w in v]
    if isinstance(v,tuple):return tuple(cpu_tree(w) for w in v)
    return copy.deepcopy(v)

def fingerprint(m):
    h=hashlib.sha256()
    for n,v in m.state_dict().items():h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def frozen(model,task):
    assert all(not p.requires_grad and p.grad is None for m in [model,task] for p in m.parameters())
    return dict(E1_H_projection=fingerprint(model),D1_and_prior=fingerprint(task))

def components(p):
    guard();base.guard_gpu=guard
    _,_,model,task,_,source,oldarms=base.load(DEVICE)
    oldarms.clear();del oldarms,source
    model.eval().requires_grad_(False);task.eval().requires_grad_(False)
    execution=read(p['train_cfg_path'])
    gc.collect();torch.cuda.empty_cache()
    return execution,model,task

class LocalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(47,32,1),nn.GELU(),nn.Conv2d(32,32,3,padding=1),nn.GELU(),nn.Conv2d(32,3,3,padding=1))
        nn.init.zeros_(self.body[-1].weight);nn.init.zeros_(self.body[-1].bias)
    def forward(self,x):return x[:,27:30]+.25*torch.tanh(self.body(x))

def heads(p):
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(p['seed']);torch.cuda.manual_seed_all(p['seed'])
        a=LocalHead().to(DEVICE)
    arms=dict(A=a,B=copy.deepcopy(a))
    opts={a:torch.optim.Adam(m.parameters(),lr=p['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0) for a,m in arms.items()}
    assert not ({v.data_ptr() for v in arms['A'].parameters()}&{v.data_ptr() for v in arms['B'].parameters()})
    return arms,opts

class NoiseTrace:
    def __init__(self,task,paired=None):self.task=task;self.paired=paired;self.noise=None;self.calls=0
    def __enter__(self):
        self.native=self.task.model.p_sample_loop
        def sample(condition,noise):
            actual=noise if self.paired is None else self.paired.to(noise.device)
            assert tuple(actual.shape)==tuple(noise.shape)==(1,16,256)
            if self.noise is None:self.noise=actual.detach().cpu().clone()
            else:assert torch.equal(self.noise,actual.cpu())
            self.calls+=1
            return self.native(condition,actual)
        self.task.model.p_sample_loop=sample;return self
    def __exit__(self,*args):self.task.model.p_sample_loop=self.native

@torch.no_grad()
def source_state(model,task,batch,paired=None,projection_proof=False,compute_metrics=False):
    cap={}
    def capture(module,args,out):
        assert out.shape[1]==96 and out.shape[-2:]==batch['target'].shape[-2:]
        projected=model.feedback.output_project(out)
        assert projected.shape[1]==16
        cap['projected']=projected.detach().cpu()
        if projection_proof:
            cap['projection_exact']=torch.equal(projected,model.feedback.output_project(out))
    hook=task.net_g.decoder_level1.register_forward_hook(capture)
    try:
        with NoiseTrace(task,paired) as proof:s=base.first(model,task,batch,offload_encoder=True)
    finally:hook.remove()
    assert proof.calls==1 and proof.noise is not None
    values=[s['p0'].cpu(),s['image'].cpu(),s['p1'].cpu(),s['t1'].cpu(),cap['projected']]
    shape=values[0].shape[-2:]
    assert all(v.shape[-2:]==shape for v in values)
    x=torch.cat(values+[torch.full((1,1,*shape),MU,dtype=torch.float32)],1).contiguous()
    assert x.dtype==torch.float32 and x.shape[1]==47 and torch.isfinite(x).all()
    result=dict(x=x,noise=proof.noise,noise_sha256=base.digest(proof.noise),t1_sha256=base.digest(s['t1']),projection_exact=cap.get('projection_exact'))
    if compute_metrics:
        from rgb_final_confirmation.evaluation import psnr,ssim
        result['metrics']=dict(psnr=psnr(s['t1'],batch['target']),ssim=ssim(s['t1'],batch['target']))
    del s,values,cap;torch.cuda.empty_cache()
    return result

def train_seed(sid):base.study.base.set_step_seed(2026,'state_source_train',0,[sid])
def crop_batch(full,box):
    y,x,yy,xx=box
    return {k:v[...,y:yy,x:xx].contiguous().to(DEVICE) if torch.is_tensor(v) else v for k,v in full.items()}
def source_pair(model,task,full,fs,item):
    train_seed(item['sample_id']);nb=crop_batch(full,item['outer'])
    tick=time.time();ns=source_state(model,task,nb,paired=fs['noise'],projection_proof=True)
    native_seconds=time.time()-tick
    y,x,yy,xx=item['outer'];bx=fs['x'][...,y:yy,x:xx].contiguous()
    assert ns['noise_sha256']==fs['noise_sha256'] and torch.equal(ns['noise'],fs['noise'])
    assert torch.equal(ns['x'][:,:15],bx[:,:15]),'Original observations differ across arms'
    gt=full['target'][...,y:yy,x:xx].detach().cpu().contiguous()
    assert torch.equal(gt,nb['target'].cpu())
    output=dict(A=ns['x'],B=bx,target=gt,inner=item['inner'],noise_sha256=fs['noise_sha256'],position=item['position'],sample_id=item['sample_id'])
    dy,dx,h,w=item['inner']
    receipt=dict(noise_A_sha256=ns['noise_sha256'],noise_B_sha256=fs['noise_sha256'],raw_observations_equal=True,
        raw_sha256=base.digest(bx[:,:15]),gt_sha256=base.digest(gt),
        native_t1_sha256=base.digest(ns['x'][:,27:30,dy:dy+h,dx:dx+w]),
        full_t1crop_sha256=base.digest(bx[:,27:30,dy:dy+h,dx:dx+w]),group_full_t1_sha256=fs['t1_sha256'],
        full_shape=list(full['target'].shape),projection_exact=ns['projection_exact'],native_seconds=native_seconds)
    del nb,ns;torch.cuda.empty_cache();return output,receipt

def roi(t,inner):
    y,x,h,w=inner;return t[...,y:y+h,x:x+w]
def snapshot(arms,opts,step,pid,p):
    return dict(step=step,protocol_id=pid,schedule_sha256=p['schedule_sha256'],arms={a:cpu_tree(m.state_dict()) for a,m in arms.items()},optimizers={a:cpu_tree(o.state_dict()) for a,o in opts.items()},rng=base.rng_state())
def restore(ck,arms,opts,pid,p):
    assert ck['protocol_id']==pid and ck['schedule_sha256']==p['schedule_sha256']
    for a,m in arms.items():m.load_state_dict(ck['arms'][a]);opts[a].load_state_dict(ck['optimizers'][a]);opts[a].zero_grad(set_to_none=True)
    base.restore_rng(ck['rng'])

def update(arms,opts,batches,step,same_input=False):
    for o in opts.values():o.zero_grad(set_to_none=True)
    losses={a:[] for a in arms}
    for batch in batches:
        y=roi(batch['target'].to(DEVICE),batch['inner'])
        for a,m in arms.items():
            x=batch['A' if same_input else a].to(DEVICE)
            t=roi(m(x),batch['inner']);loss=base.loss_fn(t,y)
            assert torch.isfinite(loss);(loss/2).backward();losses[a].append(float(loss.detach()));del x,t,loss
    norms={}
    for a,m in arms.items():
        norms[a]=float(nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True));assert norms[a]>0 and math.isfinite(norms[a]);opts[a].step()
    return dict(step=step,sample_ids=';'.join(b['sample_id'] for b in batches),positions=';'.join(str(b['position']) for b in batches),
                A_loss=float(np.mean(losses['A'])),B_loss=float(np.mean(losses['B'])),A_grad_norm=norms['A'],B_grad_norm=norms['B'])

def outer_box(y,x,h,w):return [max(0,(y-8)//8*8),max(0,(x-8)//8*8),min(h,math.ceil((y+128+8)/8)*8),min(w,math.ceil((x+128+8)/8)*8)]

def equivalence(arms,x,target):
    probe=copy.deepcopy(arms['A'])
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(18737);nn.init.normal_(probe.body[-1].weight,std=.01);nn.init.constant_(probe.body[-1].bias,.002)
    h,w=x.shape[-2:];results=[];x=x.to(DEVICE);target=target.to(DEVICE)
    for label,y,xx in [('interior',(h-128)//2,(w-128)//2),('image_boundary',0,0)]:
        guard();box=outer_box(y,xx,h,w);y0,x0,y1,x1=box;inside=[y-y0,xx-x0,128,128]
        probe.zero_grad(set_to_none=True)
        full=probe(x);fr=full[...,y:y+128,xx:xx+128];gt=target[...,y:y+128,xx:xx+128]
        assert float((fr-x[:,27:30,y:y+128,xx:xx+128]).abs().max())>1e-7
        fl=base.loss_fn(fr,gt);fl.backward();fg={n:v.grad.detach().clone() for n,v in probe.named_parameters()}
        fv=fr.detach().clone();flv=fl.detach().clone();del full,fr,fl
        probe.zero_grad(set_to_none=True)
        local=roi(probe(x[...,y0:y1,x0:x1].contiguous()),inside);ll=base.loss_fn(local,gt);ll.backward()
        output_ok=torch.allclose(fv,local,atol=2e-6,rtol=2e-5)
        loss_ok=torch.allclose(flv,ll,atol=1e-7,rtol=2e-5)
        grad_ok=all(torch.allclose(fg[n],v.grad,atol=2e-6,rtol=2e-4) for n,v in probe.named_parameters())
        result=dict(location=label,output_max_abs=float((fv-local).abs().max()),loss_abs=float((flv-ll).abs()),
                    gradient_max_abs=max(float((fg[n]-v.grad).abs().max()) for n,v in probe.named_parameters()),passed=bool(output_ok and loss_ok and grad_ok))
        results.append(result);assert result['passed'],result
        del local,ll,fg,fv,gt;torch.cuda.empty_cache()
    del probe,x,target;gc.collect();torch.cuda.empty_cache();return results

def preflight(p,items):
    locked=ROOT/'config/locked.json'
    if locked.exists():
        v=read(locked);q=dict(v);pid=q.pop('protocol_id');assert canonical(q)==pid
        assert v['scientific_id']==p['scientific_id'] and v['proposal_sha256']==sha(ROOT/'config/proposal.json')
        assert v['preflight_sha256']==sha(ROOT/'config/preflight.json') and read(ROOT/'config/preflight.json')['passed'];return v
    status('preflight_loading');execution,model,task=components(p);before=frozen(model,task);arms,opts=heads(p)
    assert tree_equal(arms['A'].state_dict(),arms['B'].state_dict())
    init=snapshot(arms,opts,0,'preflight',p);batchsets=[];times_full=[];times_native=[];io_times=[];x_probe=target_probe=None
    grouped={}
    for item in items[:8]:grouped.setdefault(item['sample_id'],[]).append(item)
    hook_exact=False;projection_exact=True
    for sid,group in grouped.items():
        guard();tick=time.time();full=base.full_item(execution,sid,DEVICE);train_seed(sid)
        fs=source_state(model,task,full,projection_proof=True);times_full.append((time.time()-tick,full['target'].shape[-2]*full['target'].shape[-1]))
        if not hook_exact:
            train_seed(sid);plain=base.first(model,task,full,offload_encoder=True)
            hook_exact=base.digest(plain['t1'])==fs['t1_sha256'];assert hook_exact;del plain
            x_probe=fs['x'];target_probe=full['target'].detach().cpu()
        projection_exact &= bool(fs['projection_exact'])
        for item in group:
            batch,receipt=source_pair(model,task,full,fs,item);batchsets.append(batch);times_native.append(receipt['native_seconds'])
            projection_exact &= bool(receipt['projection_exact'])
            tick=time.time();testpath=ROOT/'diagnostics'/('preflight_input_'+str(item['position'])+'.pt');base.save(testpath,batch);sha(testpath);io_times.append(time.time()-tick)
            assert torch.equal(arms['A'](batch['A'].to(DEVICE)).cpu(),batch['A'][:,27:30])
            assert torch.equal(arms['B'](batch['B'].to(DEVICE)).cpu(),batch['B'][:,27:30])
        del full,fs;gc.collect();torch.cuda.empty_cache()
    batchsets.sort(key=lambda b:b['position']);assert len(batchsets)==8
    largest_train=max(items,key=lambda r:r['height']*r['width'])
    tick=time.time();full=base.full_item(execution,largest_train['sample_id'],DEVICE);train_seed(largest_train['sample_id'])
    largest=source_state(model,task,full);times_full.append((time.time()-tick,largest_train['height']*largest_train['width']));del full,largest
    refs={r['sample_id']:r for r in read(p['reference_metrics_path'])['rows']}
    ref=max([refs[sid] for sid in p['validation_ids']],key=lambda r:np.prod(r['shape'][-2:]))
    tick=time.time();full=base.full_item(execution,ref['sample_id'],DEVICE)
    base.study.base.set_step_seed(2026,'H_only_pair_eval',0,[ref['sample_id']])
    val=source_state(model,task,full,compute_metrics=True)
    assert val['t1_sha256']==ref['t1_sha256'] and val['metrics']==ref['metrics']['T1']
    with torch.no_grad():
        for m in arms.values():assert torch.equal(m(val['x'].to(DEVICE)).cpu(),val['x'][:,27:30])
    torch.cuda.synchronize();max_frame_seconds=time.time()-tick;largest_val_shape=list(full['target'].shape);del full,val
    # Free frozen sources for the stronger nonzero-head gradient equivalence check.
    model.cpu();task.cpu();gc.collect();torch.cuda.empty_cache()
    local=equivalence(arms,x_probe,target_probe);del x_probe,target_probe
    update(arms,opts,batchsets[:2],1,same_input=True)
    assert tree_equal(arms['A'].state_dict(),arms['B'].state_dict()) and tree_equal(opts['A'].state_dict(),opts['B'].state_dict())
    restore(init,arms,opts,'preflight',p);train_times=[]
    for step in range(1,4):
        tick=time.time();update(arms,opts,batchsets[2*(step-1):2*step],step);torch.cuda.synchronize();train_times.append(time.time()-tick)
    snap=snapshot(arms,opts,3,'preflight',p);base.save(ROOT/'diagnostics/preflight_checkpoint.pt',snap)
    update(arms,opts,batchsets[6:8],4);expected=snapshot(arms,opts,4,'preflight',p)
    restore(torch.load(ROOT/'diagnostics/preflight_checkpoint.pt',map_location='cpu',weights_only=False),arms,opts,'preflight',p)
    update(arms,opts,batchsets[6:8],4);assert tree_equal(expected,snapshot(arms,opts,4,'preflight',p))
    restore(init,arms,opts,'preflight',p);assert before==frozen(model,task)
    rate=max(t/n for t,n in times_full);unique={r['sample_id']:r for r in items}
    full_seconds=rate*sum(r['height']*r['width'] for r in unique.values())
    cache_seconds=len(items)*(float(np.median(times_native))+float(np.median(io_times)))
    step_seconds=float(np.median(train_times[1:]));seconds=1.2*(full_seconds+cache_seconds+3000*step_seconds+64*max_frame_seconds)+900
    remaining=float(read(ROOT/'worker_registration.json')['deadline_unix'])-time.time()
    timing=dict(time_unix=time.time(),scientific_id=p['scientific_id'],
        full_measurements=[dict(sample_id=sid,seconds=v[0],pixels=v[1],seconds_per_pixel=v[0]/v[1]) for sid,v in zip(list(grouped)+[largest_train['sample_id']],times_full)],
        native_seconds=times_native,save_and_hash_seconds=io_times,train_step_seconds=train_times,
        largest_validation_seconds=max_frame_seconds,total_unique_train_pixels=sum(r['height']*r['width'] for r in unique.values()),
        unique_train_images=len(unique),full_seconds=full_seconds,cache_seconds=cache_seconds,
        train_seconds=3000*step_seconds,evaluation_seconds=64*max_frame_seconds,
        forecast_seconds=seconds,remaining_seconds=remaining,wall_deadline_unix=read(ROOT/'worker_registration.json')['deadline_unix'],
        estimator='original maximum per-pixel rate over all measured full frames; 20 percent plus900sec')
    write(ROOT/'diagnostics'/('preflight_timing_'+str(time.time_ns())+'.json'),timing)
    assert seconds<remaining,('Complete fixed protocol forecast exceeds original12h remaining budget',seconds,remaining)
    result=dict(passed=True,scientific_id=p['scientific_id'],proofs=dict(hook_t1_exact=hook_exact,projection_exact=projection_exact,
        noise_exact=True,initial_identity_exact=True,source_frozen_exact=True,same_input_update_exact=True,resume_exact=True,
        all_preflight_updates_discarded=True,largest_full_passed=True,local_equivalence_passed=True),
        local_equivalence=local,head_parameters=sum(v.numel() for v in arms['A'].parameters()),
        estimated_compute_hours=seconds/3600,estimated_full_state_hours=full_seconds/3600,train_step_seconds=step_seconds,
        median_native_seconds=float(np.median(times_native)),max_validation_seconds=max_frame_seconds,largest_validation_shape=largest_val_shape,
        largest_train_sample_id=largest_train['sample_id'],largest_train_shape=[largest_train['height'],largest_train['width']],
        estimated_cache_bytes=sum(b['A'].numel()*4+b['B'].numel()*4+b['target'].numel()*4 for b in batchsets)/len(batchsets)*len(items),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),time_unix=time.time())
    assert all(result['proofs'].values());write(ROOT/'config/preflight.json',result)
    lock=dict(scientific_id=p['scientific_id'],proposal_sha256=sha(ROOT/'config/proposal.json'),preflight_sha256=sha(ROOT/'config/preflight.json'),
              locked_time=time.time(),steps=3000,arms=['A','B'])
    lock['protocol_id']=canonical(lock)
    with locked.open('x',encoding='utf-8') as f:json.dump(lock,f,ensure_ascii=False,indent=2)
    status('preflight_passed',protocol_id=lock['protocol_id'],estimated_compute_hours=result['estimated_compute_hours'],head_parameters=result['head_parameters'])
    del model,task,arms,opts,batchsets,init,snap,expected;gc.collect();torch.cuda.empty_cache();return lock

def prepare_cache(p,locked,items):
    complete=ROOT/'diagnostics/cache_complete.json'
    if complete.exists():
        rec=read(complete);assert rec['protocol_id']==locked['protocol_id'] and rec['index_sha256']==sha(ROOT/'cache/index.json');return
    execution,model,task=components(p);before=frozen(model,task);index_path=ROOT/'cache/index.json'
    rows=read(index_path)['rows'] if index_path.exists() else [];done={r['position']:r for r in rows}
    for r in rows:assert sha(ROOT/r['file'])==r['sha256']
    groups={}
    for item in items:groups.setdefault(item['sample_id'],[]).append(item)
    started=time.time();group_count=0
    for sid,group in groups.items():
        todo=[i for i in group if i['position'] not in done]
        if not todo:continue
        guard();tick=time.time();full=base.full_item(execution,sid,DEVICE);train_seed(sid)
        fs=source_state(model,task,full)
        for item in todo:
            guard();batch,receipt=source_pair(model,task,full,fs,item)
            rel='cache/inputs/'+str(item['position']).zfill(5)+'.pt';path=ROOT/rel;base.save(path,batch)
            rec={k:item[k] for k in ['position','step','micro','sample_id','scene','top','left','outer','inner']}
            rec.update(receipt,file=rel,sha256=sha(path),bytes=path.stat().st_size)
            done[item['position']]=rec;del batch
        rows=sorted(done.values(),key=lambda r:r['position'])
        write(index_path,dict(scientific_id=p['scientific_id'],protocol_id=locked['protocol_id'],count=len(rows),rows=rows))
        group_count+=1
        if group_count==1 or group_count%10==0:
            elapsed=time.time()-started
            status('preparing_states',completed=len(rows),total=len(items),groups_done=group_count,last=sid,group_seconds=time.time()-tick,
                   elapsed_seconds=elapsed,remaining_estimate_seconds=(len(items)-len(rows))*elapsed/max(1,len(rows)))
        del full,fs;gc.collect();torch.cuda.empty_cache()
    assert [r['position'] for r in rows]==list(range(6000));after=frozen(model,task);assert before==after
    write(ROOT/'diagnostics/cache_frozen.json',dict(before=before,after=after,passed=True,all_grad_none=True))
    write(complete,dict(items=len(rows),unique_images=len(groups),protocol_id=locked['protocol_id'],index_sha256=sha(index_path),source_frozen=True))
    del model,task;gc.collect();torch.cuda.empty_cache();status('states_ready',completed=len(rows),total=6000)

def train(p,locked):
    if (ROOT/'diagnostics/training_complete.json').exists():return
    arms,opts=heads(p);pid=locked['protocol_id'];last=ROOT/'weights/last.pt';log=ROOT/'logs/training.csv';start=0
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=False);restore(ck,arms,opts,pid,p);start=ck['step'];del ck
    else:
        assert not log.exists();ck=snapshot(arms,opts,0,pid,p);base.save(ROOT/'weights/initial.pt',ck);base.save(last,ck);del ck
    if log.exists() and log.stat().st_size:
        records=list(csv.DictReader(log.open()));keep=[r for r in records if int(r['step'])<=start]
        if len(keep)!=len(records):write(ROOT/'failures'/('orphaned_rows_'+str(time.time_ns())+'.json'),records[len(keep):])
        with log.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(keep)
        assert [int(r['step']) for r in keep]==list(range(1,start+1))
    rows=read(ROOT/'cache/index.json')['rows'];assert len(rows)==6000
    status('training',step=start,total=3000)
    with log.open('a',newline='') as f:
        writer=None
        for step in range(start+1,3001):
            guard();tick=time.time();pair=[]
            for r in rows[2*(step-1):2*step]:
                batch=torch.load(ROOT/r['file'],map_location='cpu',weights_only=False);assert batch['position']==r['position'];pair.append(batch)
            record=update(arms,opts,pair,step);torch.cuda.synchronize();record['seconds']=time.time()-tick
            if writer is None:
                writer=csv.DictWriter(f,fieldnames=list(record))
                if start==0:writer.writeheader()
            writer.writerow(record);f.flush()
            if step==1 or step%100==0:
                ck=snapshot(arms,opts,step,pid,p);base.save(last,ck);del ck
            if step==1 or step%25==0:status('training',step=step,total=3000,A_loss=record['A_loss'],B_loss=record['B_loss'],seconds=record['seconds'])
            del pair
    ck=snapshot(arms,opts,3000,pid,p);base.save(ROOT/'weights/final.pt',ck);base.save(last,ck)
    assert any(not torch.equal(v,torch.load(ROOT/'weights/initial.pt',map_location='cpu',weights_only=False)['arms'][a][n]) for a in arms for n,v in ck['arms'][a].items())
    cache_frozen=read(ROOT/'diagnostics/cache_frozen.json');assert cache_frozen['passed']
    write(ROOT/'diagnostics/training_complete.json',dict(steps=3000,protocol_id=pid,schedule_sha256=p['schedule_sha256'],final_sha256=sha(ROOT/'weights/final.pt')))
    del ck,arms,opts;gc.collect();torch.cuda.empty_cache();status('training_complete',step=3000,total=3000)

def comparisons(rows):
    out={}
    for a,b in [('B','A'),('B','T1'),('A','T1')]:
        out[a+'-'+b]={}
        for metric in ['psnr','ssim']:
            scenes={s:float(np.mean([r['metrics'][a][metric]-r['metrics'][b][metric] for r in rows if r['scene']==s])) for s in sorted({r['scene'] for r in rows})}
            out[a+'-'+b][metric]=dict(scene_equal=float(np.mean(list(scenes.values()))),image_equal=float(np.mean([r['metrics'][a][metric]-r['metrics'][b][metric] for r in rows])),scenes=scenes,scenes_positive=sum(v>0 for v in scenes.values()))
    return out

def evaluate(p,locked):
    from rgb_final_confirmation.evaluation import psnr,ssim
    execution,model,task=components(p);before=frozen(model,task);arms,opts=heads(p);pid=locked['protocol_id']
    ck=torch.load(ROOT/'weights/final.pt',map_location='cpu',weights_only=False);assert ck['step']==3000;restore(ck,arms,opts,pid,p);del ck,opts
    for m in arms.values():m.eval().requires_grad_(False)
    refs={r['sample_id']:r for r in read(p['reference_metrics_path'])['rows']};path=ROOT/'metrics/validation64.json'
    rows=read(path)['rows'] if path.exists() else [];assert [r['sample_id'] for r in rows]==p['validation_ids'][:len(rows)]
    with torch.no_grad():
        for sid in p['validation_ids'][len(rows):]:
            guard();tick=time.time();batch=base.full_item(execution,sid,DEVICE)
            base.study.base.set_step_seed(2026,'H_only_pair_eval',0,[sid]);state=source_state(model,task,batch,compute_metrics=True)
            ref=refs[sid];assert state['t1_sha256']==ref['t1_sha256'] and state['metrics']==ref['metrics']['T1']
            x=state['x'].to(DEVICE);scores={'T1':state['metrics']};hashes={}
            for a,m in arms.items():
                t=m(x);scores[a]=dict(psnr=psnr(t,batch['target']),ssim=ssim(t,batch['target']));hashes[a]=base.digest(t)
                if len(rows)<8:
                    np.save(ROOT/'examples'/(sid.replace('/','_')+'_'+a+'.npy'),t.cpu().numpy())
                del t
            r=dict(sample_id=sid,scene=batch['scene'][0],shape=list(batch['target'].shape),t1_sha256=state['t1_sha256'],metrics=scores,
                   noise_sha256=state['noise_sha256'],reference_exact=True,original_size=True,input_sha256=base.digest(state['x']),output_sha256=hashes,seconds=time.time()-tick)
            rows.append(r);write(path,dict(protocol_id=pid,step=3000,rows=rows,original_size=True,tiling=False))
            status('evaluating',completed=len(rows),total=64,seconds=r['seconds']);del batch,state,x;gc.collect();torch.cuda.empty_cache()
    comp=comparisons(rows)
    gate=lambda k:comp[k]['psnr']['scene_equal']>=.05 and comp[k]['ssim']['scene_equal']>=0 and comp[k]['psnr']['scenes_positive']>=6
    summary=dict(protocol_id=pid,steps=3000,frames=64,exploratory_only=True,test_used=False,comparisons=comp,promotion_pass=bool(gate('B-A') and gate('B-T1')),
                 limitations=['single_seed','repeated_validation_exploration','fixed16channel_projection','small_local_RGB_updater','in_sample_first_round_training_images'])
    after=frozen(model,task);assert before==after
    cache=read(ROOT/'diagnostics/cache_frozen.json');assert cache['before']==before
    write(ROOT/'diagnostics/frozen_final.json',dict(before=before,after=after,passed=True,all_grad_none=True))
    write(ROOT/'reports/summary.json',summary);del model,task,arms;gc.collect();torch.cuda.empty_cache();status('evaluation_complete',promotion_pass=summary['promotion_pass'])

def finish(p,locked):
    guard();status('independent_audit')
    subprocess.run([sys.executable,str(ROOT/'scripts/audit_state_source.py'),'--root',str(ROOT)],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    assert read(ROOT/'reports/independent_audit.json')['passed'];verify(p);guard()
    files=[]
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and path.suffix not in ['.pyc','.tmp'] and path.name not in ['worker.log','controller.log','status.json','artifact_manifest.json','closeout.json','controller_heartbeat.json','worker_exit.json'] and 'pycache' not in path.parts:
            files.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size))
    write(ROOT/'artifact_manifest.json',dict(protocol_id=locked['protocol_id'],files=files,server_only=True))
    summary=read(ROOT/'reports/summary.json')
    close=dict(success=True,exploratory_only=True,steps=3000,validation_frames=64,test_frames=0,independent_audit_verified=True,
               promotion_pass=summary['promotion_pass'],protocol_id=locked['protocol_id'],completed_unix=time.time(),artifact_manifest_sha256=sha(ROOT/'artifact_manifest.json'))
    write(ROOT/'reports/final_review_closeout.json',close);write(ROOT/'closeout.json',close)
    status('screen_completed',success=True,steps=3000,validation_frames=64,promotion_pass=summary['promotion_pass'])

def main():
    lock=(ROOT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=read(ROOT/'config/proposal.json');verify(p);torch.set_num_threads(2);guard()
    if (ROOT/'closeout.json').exists():assert read(ROOT/'closeout.json')['success'];return
    items=read(p['cache_items_path'])['items'];assert len(items)==6000
    locked=preflight(p,items);prepare_cache(p,locked,items);train(p,locked);evaluate(p,locked);finish(p,locked)
if __name__=='__main__':
    try:main()
    except Exception:
        record=dict(error=traceback.format_exc(),time_unix=time.time());path=ROOT/'failures'/('worker_'+str(time.time_ns())+'.json')
        write(path,record);status('failed',failure_file=str(path),error=record['error']);raise