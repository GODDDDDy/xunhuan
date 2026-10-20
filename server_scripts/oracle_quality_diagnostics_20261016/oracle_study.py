"""Single Oracle-C arm; reuse experiment30's frozen loader, updates and metrics."""
import copy,csv,gc,hashlib,json,math,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'c_feedback_polar_loop_20261015'
sys.path.insert(0,str(OLD/'scripts'))
import c_loop_run as base
ORIGINAL_FIRST=base.first

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

def status(stage,**kw):
    record=dict(stage=stage,time_unix=time.time(),**kw)
    write(ROOT/'status.json',record);print(json.dumps(record),flush=True)

def plan():
    p=json.loads((ROOT/'config/plan.json').read_text())
    for f in p['new_code_manifest']:assert base.sha(ROOT/f['path'])==f['sha256'],f['path']
    for f in p['source_receipts']:assert base.sha(OLD/f['path'])==f['sha256'],f['path']
    return p

@torch.no_grad()
def oracle_map(t1,target,tau):
    error=F.avg_pool2d((t1.detach().clamp(0,1)-target).abs().mean(1,keepdim=True),4)
    return torch.exp(-error.clamp_min(.001)/tau).detach()

@torch.no_grad()
def oracle_first(model,task,batch,offload_encoder=False):
    s=ORIGINAL_FIRST(model,task,batch,offload_encoder=offload_encoder)
    s['c']=oracle_map(s['t1'],batch['target'],model.tau)
    return s

def initial_arm(e,opt,step=0):
    name='initial.pt' if step==0 else f'step_{step}.pt'
    ck=torch.load(OLD/'weights'/name,map_location='cpu',weights_only=False)
    assert ck['step']==step
    state=e.state_dict();state.update(ck['arms']['B']);e.load_state_dict(state,strict=True)
    opt.load_state_dict(ck['optimizers']['B']);base.restore_rng(ck['rng'])
    opt.zero_grad(set_to_none=True)
    return ck

def components(device):
    cfg,oldplan,model,task,raw,source,arms=base.load(device)
    e=arms.pop('B');arms.clear();del arms;gc.collect()
    opts=base.optimizers({'O':e});initial_arm(e,opts['O'])
    return cfg,model,task,raw,source,{'O':e},opts

def micros(cfg,raw,device,start,end):
    for rb in base.study.base.scheduled_loader(cfg,raw,end,start_step=start):
        yield [base.study.base.unpack_batch(m,device) for m in base.study.microbatches(rb)]

def preflight(device):
    p=plan();cfg,model,task,raw,source,arms,opts=components(device);e=arms['O']
    proof=[]
    if device.type=='cuda':
        oldrows={int(r['step']):r for r in csv.DictReader((OLD/'logs/training.csv').open())}
        base.first=ORIGINAL_FIRST
        for previous in (0,500):
            initial_arm(e,opts['O'],previous)
            batch=next(micros(cfg,raw,device,previous,previous+1))
            ids,loss,norm,hashes=base.update(model,task,{'B':e},{'B':opts['O']},batch,previous+1)
            expected=oldrows[previous+1]
            assert ids==expected['sample_ids'].split(';')
            assert loss['B']==float(expected['B_loss']),(previous,loss,expected['B_loss'])
            assert norm['B']==float(expected['B_grad_norm']),(previous,norm,expected['B_grad_norm'])
            proof.append(dict(step=previous+1,loss_exact=True,gradient_norm_exact=True,sample_ids=ids))
            del batch
    initial_arm(e,opts['O'])
    small=copy.deepcopy(cfg);small['data']['crop_size']=32
    tiny=next(micros(small,base.study.base.build_base_config(small),device,0,1))
    b=tiny[0]
    base.seed(111);s=ORIGINAL_FIRST(model,task,b)
    base.seed(111);other=ORIGINAL_FIRST(model,task,dict(b,target=1-b['target']))
    assert torch.equal(s['t1'],other['t1']) and torch.equal(s['c'],other['c'])
    c=oracle_map(s['t1'],b['target'],model.tau)
    c_other=oracle_map(s['t1'],1-b['target'],model.tau)
    expected=F.avg_pool2d((s['t1'].clamp(0,1)-b['target']).abs().mean(1,keepdim=True),4).clamp_min(.001)
    assert torch.equal(c,torch.exp(-expected/model.tau)) and not torch.equal(c,c_other)
    s['c']=c
    events=[];hooks=[]
    for level,a in e.adapters.items():
        hooks.append(a.mix.register_forward_pre_hook(lambda m,args,level=level:events.append((level,args[0].detach()))))
    with torch.no_grad():
        pp,tt=base.second(model,task,e,s,'O')
        assert torch.equal(pp,s['p1']) and torch.equal(tt,s['t1'])
    for h in hooks:h.remove()
    assert len(events)==2
    for level,x in events:
        assert torch.equal(x[:,-2:-1],F.interpolate(c,size=x.shape[-2:],mode='bilinear',align_corners=False))
    del s,other,pp,tt,events,c,c_other,expected
    base.first=oracle_first
    for step in range(1,4):base.update(model,task,arms,opts,tiny,step)
    gradients={name:sum(float(q.grad.abs().sum()) for q in module.parameters() if q.grad is not None)
               for name,module in [('level2',e.adapters['level2']),('level3',e.adapters['level3']),('output',e.mid_output)]}
    assert all(v>0 and math.isfinite(v) for v in gradients.values())
    ck=copy.deepcopy(base.checkpoint(arms,opts,3,p))
    base.update(model,task,arms,opts,tiny,4)
    expected=copy.deepcopy(base.checkpoint(arms,opts,4,p))
    base.restore(ck,arms,opts,p);base.update(model,task,arms,opts,tiny,4)
    actual=base.checkpoint(arms,opts,4,p)
    assert all(torch.equal(t,actual['arms']['O'][n]) for n,t in expected['arms']['O'].items())
    base.audit(model,task,source,arms)
    receipt=dict(passed=True,device=str(device),oracle_definition_exact=True,oracle_changes_with_GT=True,
                 first_round_GT_invariant=True,zero_head_identity_exact=True,oracle_channel_exact=True,
                 original_B_replay=proof,optimizer_replay_exact=True,gradients=gradients,
                 frozen_exact=True,all_preflight_updates_discarded=True,time_unix=time.time())
    del tiny,ck,expected,actual;gc.collect()
    if device.type=='cuda':
        torch.cuda.empty_cache()
        # The largest archived frame is selected by dimensions, not scores.
        rows=json.loads((OLD/'metrics/validation872.json').read_text())['rows']
        sid=max(rows,key=lambda r:np.prod(r['shape'][-2:]))['sample_id']
        batch=base.full_item(cfg,sid,device);tick=time.time()
        eval_one(cfg,model,task,e,batch,p,save_arrays=False)
        torch.cuda.synchronize()
        receipt.update(max_frame_seconds=time.time()-tick,max_frame_sample=sid,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),original_size=True)
    write(ROOT/'diagnostics'/('cpu_checks.json' if device.type=='cpu' else 'gpu_checks.json'),receipt)
    status('preflight_passed',device=str(device))

@torch.no_grad()
def eval_one(cfg,model,task,e,batch,p,save_arrays=True):
    from rgb_final_confirmation.evaluation import psnr,ssim
    base.study.base.set_step_seed(2026,'H_only_pair_eval',0,batch['sample_id'])
    s=ORIGINAL_FIRST(model,task,batch,offload_encoder=True)
    c_pred=s['c'].cpu();s['c']=oracle_map(s['t1'],batch['target'],model.tau)
    t1_metrics=dict(psnr=psnr(s['t1'],batch['target']),ssim=ssim(s['t1'],batch['target']))
    for k in ('p0','image','p1','t1','c'):s[k]=s[k].cpu()
    s['f1']={k:v.cpu() for k,v in s['f1'].items()};torch.cuda.empty_cache()
    current=dict(s)
    for k in ('p0','image','p1','t1','c'):current[k]=s[k].cuda()
    current['f1']={k:v.cuda() for k,v in s['f1'].items()}
    current['encoded']={k:v.cuda() for k,v in s['encoded'].items()}
    p2,t2=base.second(model,task,e,current,'O')
    restored=dict(psnr=psnr(t2,batch['target']),ssim=ssim(t2,batch['target']))
    p2,t2=p2.cpu(),t2.cpu();del current;torch.cuda.empty_cache()
    target=batch['target'].cpu()
    error1=(s['t1'].clamp(0,1)-target).abs().mean(1,keepdim=True)
    difference=(t2.clamp(0,1)-target).abs().mean(1,keepdim=True)-error1
    quality=F.interpolate(c_pred,size=target.shape[-2:],mode='bilinear',align_corners=False)
    low=quality<=torch.quantile(quality,.2);high=quality>=torch.quantile(quality,.8)
    dp=(p2-s['p1']).abs()
    dist=lambda x:dict(mean=float(x.mean()),std=float(x.std(unbiased=False)),
                      quantiles=[float(v) for v in torch.quantile(x,torch.tensor([.01,.1,.5,.9,.99]))])
    row=dict(sample_id=batch['sample_id'][0],scene=batch['scene'][0],shape=list(target.shape),
             t1_sha256=base.digest(s['t1']),metrics=dict(T1=t1_metrics,O=restored),
             local=dict(low_predC20_mae_change=float(difference[low].mean()),high_predC20_mae_change=float(difference[high].mean()),
                        mean_polar_change=float(dp.mean()),bound_hit_fraction=float((dp>=.02*(1-1e-4)).float().mean()),
                        nonzero_T_change_fraction=float((t2!=s['t1']).float().mean())),
             quality_distribution=dict(predicted=dist(c_pred),oracle=dist(s['c'])))
    if save_arrays:
        out=ROOT/'examples'/(row['sample_id'].replace('/','_')+'.npz')
        tmp=out.with_suffix('.tmp')
        with tmp.open('wb') as f:np.savez(f,t2_O=t2.numpy(),p2_O=p2.numpy(),c_oracle=s['c'].numpy())
        tmp.replace(out)
    return row

def train(cfg,p,model,task,raw,source,arms,opts):
    base.first=oracle_first
    last=ROOT/'weights/last.pt';step0=0
    log=ROOT/'logs/training.csv'
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=False);base.restore(ck,arms,opts,p);step0=ck['step']
        records=list(csv.DictReader(log.open())) if log.exists() else []
        keep=[v for v in records if int(v['step'])<=step0]
        if len(keep)!=len(records):
            write(ROOT/'failures'/f'orphaned_training_rows_{int(time.time())}.json',records[len(keep):])
            with log.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(keep)
        assert len(keep)==step0
    else:
        assert not log.exists(),'Do not overwrite training without an accepted checkpoint'
        initial_arm(arms['O'],opts['O']);base.save(ROOT/'weights/initial.pt',base.checkpoint(arms,opts,0,p))
        base.save(last,base.checkpoint(arms,opts,0,p))
    status('training',step=step0,total=6000)
    start=time.time()
    with log.open('a',newline='') as f:
        w=None
        for step,batch in enumerate(micros(cfg,raw,torch.device('cuda:0'),step0,6000),step0+1):
            tick=time.time();ids,loss,norm,hashes=base.update(model,task,arms,opts,batch,step)
            torch.cuda.synchronize();del batch
            row=dict(step=step,sample_ids=';'.join(ids),seconds=time.time()-tick,O_loss=loss['O'],O_grad_norm=norm['O'])
            if w is None:
                w=csv.DictWriter(f,fieldnames=list(row))
                if step0==0:w.writeheader()
            w.writerow(row);f.flush()
            if step%50==0 or step==1:status('training',step=step,total=6000,loss=loss['O'],seconds=row['seconds'])
            if step%100==0:
                base.audit(model,task,source,arms)
                ck=base.checkpoint(arms,opts,step,p);base.save(last,ck)
                if step%500==0:
                    base.save(ROOT/'weights'/f'step_{step}.pt',ck)
                    write(ROOT/'diagnostics'/f'frozen_step_{step}.json',dict(frozen_exact=True,t1_hashes=hashes))
    rows=list(csv.DictReader(log.open()));assert [int(r['step']) for r in rows]==list(range(1,6001))
    seen={sid for r in rows for sid in r['sample_ids'].split(';')};assert len(seen)==5440
    base.audit(model,task,source,arms);base.save(ROOT/'weights/final.pt',base.checkpoint(arms,opts,6000,p))
    write(ROOT/'diagnostics/training_complete.json',dict(steps=6000,unique_training_images=5440,
          crop_presentations=12000,frozen_exact=True,endpoint_sha256=base.sha(ROOT/'weights/final.pt'),
          active_seconds=sum(float(r['seconds']) for r in rows),last_segment_seconds=time.time()-start))

def compare(rows):
    scenes=sorted({r['scene'] for r in rows});out={}
    for a,b in [('O','P'),('O','T1'),('O','D')]:
        out[a+'-'+b]={}
        for m in ('psnr','ssim'):
            by={s:float(np.mean([r['metrics'][a][m]-r['metrics'][b][m] for r in rows if r['scene']==s])) for s in scenes}
            out[a+'-'+b][m]=dict(scene_equal=float(np.mean(list(by.values()))),image_equal=float(np.mean([r['metrics'][a][m]-r['metrics'][b][m] for r in rows])),scenes=by,scenes_positive=sum(v>0 for v in by.values()))
    return out

def evaluate(cfg,p,model,task,e,split):
    old=json.loads((OLD/'metrics'/f'{split}.json').read_text());reference={r['sample_id']:r for r in old['rows']}
    partial=ROOT/'metrics'/f'{split}_partial.json'
    rows=json.loads(partial.read_text())['rows'] if partial.exists() else []
    done={r['sample_id'] for r in rows};assert len(done)==len(rows) and done<=set(reference)
    for sid,prior in reference.items():
        if sid in done:continue
        batch=base.full_item(cfg,sid,torch.device('cuda:0'))
        row=eval_one(cfg,model,task,e,batch,p);del batch
        assert row['t1_sha256']==prior['t1_sha256'],sid
        assert row['metrics']['T1']==prior['metrics']['T1'],(sid,row['metrics']['T1'],prior['metrics']['T1'])
        row['metrics']['P']=prior['metrics']['B'];row['metrics']['D']=prior['metrics']['D']
        rows.append(row);write(partial,dict(rows=rows,original_size=True,tiling=False))
        gc.collect();torch.cuda.empty_cache()
        if len(rows)%8==0:status('evaluating',split=split,completed=len(rows),total=len(reference))
    assert len(rows)==old['frames']
    result=dict(split=split,frames=len(rows),rows=rows,comparisons=compare(rows),step=6000,
                protocol_id=p['protocol_id'],original_size=True,tiling=False,reused_P_and_D=True)
    write(ROOT/'metrics'/f'{split}.json',result);return result

def gpu():
    p=plan();assert json.loads((ROOT/'diagnostics/cpu_checks.json').read_text())['passed']
    if not (ROOT/'diagnostics/gpu_checks.json').exists():preflight(torch.device('cuda:0'))
    gc.collect();torch.cuda.empty_cache()
    cfg,model,task,raw,source,arms,opts=components(torch.device('cuda:0'))
    if not (ROOT/'diagnostics/training_complete.json').exists():train(cfg,p,model,task,raw,source,arms,opts)
    final=ROOT/'weights/final.pt';ck=torch.load(final,map_location='cpu',weights_only=False)
    assert ck['step']==6000 and ck['protocol_id']==p['protocol_id']
    base.restore(ck,arms,opts,p);base.audit(model,task,source,arms);del ck,opts;gc.collect();torch.cuda.empty_cache()
    validation=evaluate(cfg,p,model,task,arms['O'],'validation872')
    test=evaluate(cfg,p,model,task,arms['O'],'test188')
    base.audit(model,task,source,arms)
    gates={split:{pair:bool(c[pair]['psnr']['scene_equal']>=.05 and c[pair]['ssim']['scene_equal']>=0 and c[pair]['psnr']['scenes_positive']>=6) for pair in ('O-P','O-T1')} for split,c in [('validation',validation['comparisons']),('test',test['comparisons'])]}
    write(ROOT/'reports/oracle_summary.json',dict(protocol_id=p['protocol_id'],steps=6000,validation_frames=872,test_frames=188,
          validation=validation['comparisons'],test=test['comparisons'],gates=gates,
          source_frozen_exact=True,endpoint_sha256=base.sha(final),test_used_for_selection=False,
          oracle_uses_GT=True,deployment_claim=False,additional_training=False))
    status('gpu_completed',validation=872,test=188,steps=6000)

if __name__=='__main__':
    torch.set_num_threads(2)
    try:
        if sys.argv[1]=='cpu':preflight(torch.device('cpu'))
        elif sys.argv[1]=='gpu':gpu()
        else:raise ValueError(sys.argv[1])
    except Exception:
        write(ROOT/'failures'/f'worker_{int(time.time())}.json',dict(traceback=traceback.format_exc(),time_unix=time.time()))
        raise
