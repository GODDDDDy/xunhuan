"""Fixed-constant and histogram-preserving rank controls; optional paired training seed."""
import copy,csv,gc,hashlib,json,math,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
PREV=ROOT.parent/'confidence_correction_causal_20261017'
sys.path.insert(0,str(PREV/'scripts'))
import causal_run as prior
base=prior.base;DEVICE=prior.DEVICE
ORIGINAL_SECOND=prior.SECOND
ORIGINAL_SEED=base.study.base.set_step_seed
CORE=['R1N0','R1F','R1S'];ALL=CORE+['R1P7','R1S7']
MU=None
write=prior.write
prior.ROOT=ROOT

def status(stage,**kw):
    obj=dict(stage=stage,time_unix=time.time(),**kw);write(ROOT/'status.json',obj);print(json.dumps(obj,ensure_ascii=False),flush=True)
def guard():
    assert time.time()<json.loads((ROOT/'budget.json').read_text())['deadline_unix'],'12h deadline reached'
def verify():
    p=json.loads((ROOT/'config/proposal.json').read_text())
    for x in p['source_receipts']:assert base.sha(x['path'])==x['sha256'],x['path']
    for x in p['code_manifest']+p['data_manifest']:assert base.sha(ROOT/x['path'])==x['sha256'],x['path']
    return p

def rank_condition(c,s):
    error=F.avg_pool2d((s['t1'].clamp(0,1)-s['target']).abs().mean(1,keepdim=True),4).clamp_min(.001)
    assert c.shape==error.shape
    values=torch.sort(c.flatten(1),descending=True,stable=True).values
    positions=torch.argsort(error.flatten(1),dim=1,stable=True)
    ranked=torch.empty_like(values).scatter_(1,positions,values).reshape_as(c).detach()
    assert torch.equal(torch.sort(ranked.flatten(1),descending=True,stable=True).values,values)
    return ranked
def condition(s,arm):
    if arm in ['R1N0','R1F']:
        assert MU is not None
        return torch.full_like(s['c'],MU).detach()
    if arm in ['R1S','R1S7']:return rank_condition(s['c'],s)
    return s['c']
def second(model,task,e,s,arm,use_encoder_cache=True):
    q=dict(s);q['c']=condition(s,arm)
    return ORIGINAL_SECOND(model,task,e,q,'B',use_encoder_cache=use_encoder_cache)
prior.second=second
prior.guard=guard

def components(names):
    out=prior.components(names)
    for a,e in out[-2].items():
        for ad in e.adapters.values():ad.suppress_task=a=='R1N0'
    base.second=second;base.guard_gpu=guard
    return out
def cfgs():
    return {seed:json.loads((ROOT/'config'/f'execution_config_{seed}.json').read_text()) for seed in [2026,2027]}
def update_seed(model,task,arms,opts,batches,step,seed):
    def paired_seed(original,phase,n,ids):
        return ORIGINAL_SEED(seed if phase=='C_feedback_polar_training' else original,phase,n,ids)
    base.study.base.set_step_seed=paired_seed
    try:return base.update(model,task,arms,opts,batches,step)
    finally:base.study.base.set_step_seed=ORIGINAL_SEED

def calibrate(cfg,raw,model,task):
    global MU
    path=ROOT/'config/calibration.json'
    if path.exists():
        obj=json.loads(path.read_text());MU=obj['mu'];assert obj['samples']==256 and obj['training_only'];return
    status('calibrating_global_constant',samples=256)
    rows=[]
    schedule=json.loads(Path(cfg['paths']['schedule']).read_text())['items']
    for step,batches in enumerate(prior.micros(cfg,raw,0,128),1):
        guard()
        for batch in batches:
            ORIGINAL_SEED(2026,'C_feedback_polar_training',step,batch['sample_id'])
            s=prior.first(model,task,batch);pos=len(rows)
            rows.append(dict(sample_id=batch['sample_id'][0],schedule_position=pos,**schedule[pos],c_mean=float(s['c'].double().mean())))
            del s
        if step%32==0:status('calibrating_global_constant',completed=len(rows),samples=256)
    MU=float(np.mean([r['c_mean'] for r in rows]));assert 0<MU<=1
    write(path,dict(mu=MU,samples=256,training_only=True,uses_GT=False,rows=rows))

def preflight(proposal):
    if (ROOT/'config/locked.json').exists():
        global MU
        MU=json.loads((ROOT/'config/calibration.json').read_text())['mu']
        return json.loads((ROOT/'config/locked.json').read_text())
    status('preflight_loading')
    cfg,model,task,raw,source,arms,opts=components(ALL);configs=cfgs()
    raws={s:base.study.base.build_base_config(c) for s,c in configs.items()}
    calibrate(configs[2026],raws[2026],model,task)
    counts={a:sum(p.numel() for p in e.parameters() if p.requires_grad) for a,e in arms.items()}
    assert set(counts.values())=={2003598}
    replay=prior.make_arms(model.r2p,['R1P']);ro=base.optimizers(replay)
    logs={int(r['step']):r for r in csv.DictReader((PREV/'logs/training.csv').open())};proof=[]
    for previous in [0,500]:
        ck=torch.load(PREV/'weights'/('initial.pt' if previous==0 else 'step_500.pt'),map_location='cpu',weights_only=False)
        st=replay['R1P'].state_dict();st.update(ck['arms']['R1P']);replay['R1P'].load_state_dict(st)
        ro['R1P'].load_state_dict(ck['optimizers']['R1P']);base.restore_rng(ck['rng'])
        batches=next(prior.micros(configs[2026],raws[2026],previous,previous+1))
        ids,loss,norm,_=update_seed(model,task,replay,ro,batches,previous+1,2026);expect=logs[previous+1]
        assert ids==expect['sample_ids'].split(';')
        assert loss['R1P']==float(expect['R1P_loss'])
        assert norm['R1P']==float(expect['R1P_grad_norm'])
        proof.append(dict(step=previous+1,loss_exact=True,gradient_exact=True,samples_exact=True))
        del ck,batches
    del replay,ro;gc.collect();torch.cuda.empty_cache()
    batch=next(prior.micros(configs[2026],raws[2026],0,1))[0];s=prior.first(model,task,batch)
    events=[];hooks=[]
    for a,e in arms.items():
        for level,ad in e.adapters.items():
            channels=ad.out.in_channels
            hooks.append(ad.mix.register_forward_pre_hook(lambda m,args,a=a,level=level,ch=channels:events.append((a,level,args[0][:,-2:].detach().clone(),args[0][:,3*ch:4*ch].detach().clone()))))
    c0=s['c'].clone()
    with torch.no_grad():
        for a,e in arms.items():
            p,t=second(model,task,e,s,a);assert torch.equal(p,s['p1']) and torch.equal(t,s['t1'])
        ranked=rank_condition(c0,s)
        assert torch.equal(torch.sort(ranked.flatten(1)).values,torch.sort(c0.flatten(1)).values)
    for h in hooks:h.remove()
    assert torch.equal(c0,s['c'])
    for a,level,x,guidance in events:
        expected=F.interpolate(condition(s,a),size=x.shape[-2:],mode='bilinear',align_corners=False)
        assert torch.equal(x[:,:1],expected)
        assert torch.equal(x[:,1:],torch.full_like(x[:,1:],0. if a=='R1N0' else 1.))
        if a=='R1N0':assert not torch.count_nonzero(guidance)
    del s,batch,events,hooks,p,t,c0,ranked
    timings=[]
    for step in range(1,7):
        tick=time.time()
        for seed,names in [(2026,CORE),(2027,ALL[3:])]:
            batches=next(prior.micros(configs[seed],raws[seed],step-1,step))
            update_seed(model,task,{a:arms[a] for a in names},{a:opts[a] for a in names},batches,step,seed)
        torch.cuda.synchronize();timings.append(time.time()-tick)
    gradients={a:sum(float(p.grad.abs().sum()) for n,p in e.named_parameters() if n.startswith('de_decoder1.') and p.grad is not None) for a,e in arms.items()}
    assert all(math.isfinite(x) and x>0 for x in gradients.values())
    tick=time.time();prior.audit(model,task,source,arms);audit_seconds=time.time()-tick
    snap=copy.deepcopy(prior.ckpt(arms,opts,6,'preflight'))
    tick=time.time();base.save(ROOT/'diagnostics/preflight_checkpoint.pt',snap);save_seconds=time.time()-tick
    replay_batches={seed:next(prior.micros(configs[seed],raws[seed],6,7)) for seed in [2026,2027]}
    for seed,names in [(2026,CORE),(2027,ALL[3:])]:
        update_seed(model,task,{a:arms[a] for a in names},{a:opts[a] for a in names},replay_batches[seed],7,seed)
    expected=copy.deepcopy(prior.ckpt(arms,opts,7,'preflight'));prior.restore(snap,arms,opts,'preflight')
    for seed,names in [(2026,CORE),(2027,ALL[3:])]:
        update_seed(model,task,{a:arms[a] for a in names},{a:opts[a] for a in names},replay_batches[seed],7,seed)
    actual=prior.ckpt(arms,opts,7,'preflight')
    assert all(torch.equal(v,actual['arms'][a][n]) for a,st in expected['arms'].items() for n,v in st.items())
    for a in arms:
        assert expected['optimizers'][a]['param_groups']==actual['optimizers'][a]['param_groups']
        for key,state in expected['optimizers'][a]['state'].items():
            for field,value in state.items():assert torch.equal(value,actual['optimizers'][a]['state'][key][field])
    def tree_equal(a,b):
        if torch.is_tensor(a):return torch.equal(a,b)
        if isinstance(a,np.ndarray):return np.array_equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(tree_equal(v,b[k]) for k,v in a.items())
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
        return a==b
    assert tree_equal(expected['rng'],actual['rng'])
    del snap,expected,actual,replay_batches,batches,opts
    for e in arms.values():e.zero_grad(set_to_none=True)
    gc.collect();torch.cuda.empty_cache()
    refs=json.loads((PREV/'metrics/validation872.json').read_text())['rows'];largest=max(refs,key=lambda r:np.prod(r['shape'][-2:]))
    b=base.full_item(cfg,largest['sample_id'],DEVICE);tick=time.time()
    row=prior.eval_one(cfg,model,task,arms,b);torch.cuda.synchronize();frame_seconds=time.time()-tick
    assert row['t1_sha256']==largest['t1_sha256'] and row['metrics']['T1']==largest['metrics']['T1']
    del b;gc.collect();torch.cuda.empty_cache()
    seconds=float(np.median(timings[2:]));budget=json.loads((ROOT/'budget.json').read_text());elapsed=time.time()-budget['start_unix']
    forecast=elapsed+1.2*(6000*seconds+1060*frame_seconds+60*save_seconds+12*audit_seconds)+3600
    selected=list(ALL)
    if forecast>12*3600:
        selected=list(CORE)
        for a in ALL[3:]:del arms[a]
        gc.collect();torch.cuda.empty_cache()
        op=base.optimizers(arms);ts=[]
        for step in range(10,14):
            tick=time.time();bs=next(prior.micros(configs[2026],raws[2026],step-1,step));update_seed(model,task,arms,op,bs,step,2026)
            torch.cuda.synchronize();ts.append(time.time()-tick)
        for e in arms.values():e.zero_grad(set_to_none=True)
        b=base.full_item(cfg,largest['sample_id'],DEVICE);tick=time.time();prior.eval_one(cfg,model,task,arms,b);torch.cuda.synchronize()
        f3=time.time()-tick;del b,op,bs
        seconds=float(np.median(ts[1:]));frame_seconds=f3
        elapsed=time.time()-budget['start_unix'];forecast=elapsed+1.2*(6000*seconds+1060*f3+60*save_seconds+12*audit_seconds)+3600
        assert forecast<=12*3600,('Core complete protocol exceeds12h',forecast)
    locked=dict(proposal_protocol=proposal['protocol_id'],arms=selected,steps=6000,trainable_counts={a:counts[a] for a in selected},
        original_R1P_replay=proof,conditions_verified=True,histogram_exact=True,suppression_verified=True,constant_exact=True,
        optimizer_replay_exact=True,source_frozen_exact=True,original_size_verified=True,all_preflight_updates_discarded=True,
        calibration_sha256=base.sha(ROOT/'config/calibration.json'),train_step_seconds=seconds,max_frame_seconds=frame_seconds,
        estimated_total_hours=forecast/3600,peak_allocated_bytes=torch.cuda.max_memory_allocated(),time_unix=time.time())
    locked['protocol_id']=hashlib.sha256(json.dumps(locked,sort_keys=True).encode()).hexdigest()
    write(ROOT/'config/locked.json',locked);write(ROOT/'diagnostics/preflight.json',locked)
    status('preflight_passed',arms=selected,estimated_total_hours=forecast/3600)
    del model,task,source,arms;gc.collect();torch.cuda.empty_cache()
    return locked

def train(locked):
    if (ROOT/'diagnostics/training_complete.json').exists():return
    names=locked['arms'];cfg,model,task,raw,source,arms,opts=components(names)
    configs=cfgs();raws={s:base.study.base.build_base_config(c) for s,c in configs.items()}
    seeds=[2026,2027] if 'R1P7' in names else [2026]
    groups={2026:CORE,2027:ALL[3:]};pid=locked['protocol_id'];last=ROOT/'weights/last.pt';log=ROOT/'logs/training.csv';start=0
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=False);prior.restore(ck,arms,opts,pid);start=ck['step'];del ck
    else:
        assert not log.exists();state=prior.ckpt(arms,opts,0,pid);base.save(ROOT/'weights/initial.pt',state);base.save(last,state);del state
    if log.exists():
        records=list(csv.DictReader(log.open()));keep=[r for r in records if int(r['step'])<=start]
        if len(keep)!=len(records):write(ROOT/'failures'/f'orphaned_rows_{int(time.time())}.json',records[len(keep):])
        with log.open('w',newline='') as f:
            if records:
                w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(keep)
            else:assert start==0
        assert len(keep)==start
    loaders={s:iter(prior.micros(configs[s],raws[s],start,6000)) for s in seeds}
    tick0=time.time();status('training',step=start,total=6000,arms=names)
    with log.open('a',newline='') as f:
        writer=None
        for step in range(start+1,6001):
            guard();tick=time.time();row=dict(step=step);hashes={}
            for seed in seeds:
                bs=next(loaders[seed]);group=groups[seed]
                ids,loss,norm,h=update_seed(model,task,{a:arms[a] for a in group},{a:opts[a] for a in group},bs,step,seed)
                row[f'sample_ids_{seed}']=';'.join(ids);hashes[str(seed)]=h
                for a in group:row[a+'_loss']=loss[a];row[a+'_grad_norm']=norm[a]
                del bs
            torch.cuda.synchronize();row['seconds']=time.time()-tick
            if writer is None:
                writer=csv.DictWriter(f,fieldnames=list(row))
                if start==0:writer.writeheader()
            writer.writerow(row);f.flush()
            if step%25==0 or step==1:status('training',step=step,total=6000,arms=names,seconds=row['seconds'])
            if step%100==0:
                state=prior.ckpt(arms,opts,step,pid);base.save(last,state)
                if step%500==0:
                    prior.audit(model,task,source,arms);write(ROOT/'diagnostics'/f'frozen_{step}.json',dict(passed=True,t1_hashes=hashes,time_unix=time.time()))
                    base.save(ROOT/'weights'/f'step_{step}.pt',state)
                del state
    rows=list(csv.DictReader(log.open()));assert [int(r['step']) for r in rows]==list(range(1,6001))
    for seed in seeds:assert len({sid for r in rows for sid in r[f'sample_ids_{seed}'].split(';')})==5440
    prior.audit(model,task,source,arms);base.save(ROOT/'weights/final.pt',prior.ckpt(arms,opts,6000,pid))
    write(ROOT/'diagnostics/training_complete.json',dict(steps=6000,arms=names,source_frozen_exact=True,final_sha256=base.sha(ROOT/'weights/final.pt'),last_segment_wall_seconds=time.time()-tick0))
    del model,task,source,arms,opts;gc.collect();torch.cuda.empty_cache()

def comparisons(rows):
    names=rows[0]['metrics'];pairs=[(a,'T1') for a in names if a!='T1']
    pairs += [('R1F','R1N0'),('R1P','R1F'),('R1S','R1P'),('R1F','R1D'),('R1N0','R1N')]
    if 'R1P7' in names:pairs += [('R1S7','R1P7'),('R1P7','R1P'),('R1S7','R1S')]
    output={}
    for a,b in pairs:
        output[a+'-'+b]={}
        for m in ['psnr','ssim']:
            by={s:float(np.mean([r['metrics'][a][m]-r['metrics'][b][m] for r in rows if r['scene']==s])) for s in sorted({r['scene'] for r in rows})}
            output[a+'-'+b][m]=dict(scene_equal=float(np.mean(list(by.values()))),image_equal=float(np.mean([r['metrics'][a][m]-r['metrics'][b][m] for r in rows])),scenes=by,scenes_positive=sum(v>0 for v in by.values()))
    return output

def evaluate(locked):
    cfg,model,task,raw,source,arms,opts=components(locked['arms']);ck=torch.load(ROOT/'weights/final.pt',map_location='cpu',weights_only=False)
    assert ck['step']==6000;prior.restore(ck,arms,opts,locked['protocol_id']);del ck,opts;gc.collect();torch.cuda.empty_cache()
    summaries={}
    for split,count in [('validation872',872),('test188',188)]:
        old=json.loads((PREV/'metrics'/f'{split}.json').read_text());refs={r['sample_id']:r for r in old['rows']};assert len(refs)==count
        fp=ROOT/'metrics'/f'{split}.json';pp=ROOT/'metrics'/f'{split}_partial.json'
        if fp.exists():summaries[split]=json.loads(fp.read_text())['comparisons'];continue
        rows=json.loads(pp.read_text())['rows'] if pp.exists() else [];done={r['sample_id'] for r in rows};assert len(done)==len(rows)
        examples={min(r['sample_id'] for r in refs.values() if r['scene']==s) for s in {r['scene'] for r in refs.values()}}
        for sid,ref in refs.items():
            if sid in done:continue
            guard();b=base.full_item(cfg,sid,DEVICE);r=prior.eval_one(cfg,model,task,arms,b,example=sid in examples);del b
            assert r['t1_sha256']==ref['t1_sha256'] and r['metrics']['T1']==ref['metrics']['T1'] and r['shape']==ref['shape']
            r['metrics'].update({a:ref['metrics'][a] for a in ['R1P','R1D','R1N','R1O','R0P','R0D','R0O']})
            rows.append(r);write(pp,dict(rows=rows,original_size=True,step=6000));gc.collect();torch.cuda.empty_cache()
            if len(rows)%8==0:status('evaluating',split=split,completed=len(rows),total=count,arms=locked['arms'])
        assert len(rows)==count and len({r['sample_id'] for r in rows})==count
        summaries[split]=comparisons(rows);write(fp,dict(rows=rows,frames=count,comparisons=summaries[split],step=6000,protocol_id=locked['protocol_id'],original_size=True,tiling=False))
    prior.audit(model,task,source,arms)
    write(ROOT/'reports/summary.json',dict(protocol_id=locked['protocol_id'],arms=locked['arms'],steps=6000,validation_frames=872,test_frames=188,comparisons=summaries,
        source_frozen_exact=True,oracle_uses_GT=True,training_seeds=[2026,2027] if 'R1P7' in arms else [2026],evaluation_seed=2026,
        constant_mu=MU,multiseed_scope='Only S-P has two paired training seeds when P7/S7 are locked; N0/F remain one seed',test_used_for_selection=False,final_sha256=base.sha(ROOT/'weights/final.pt')))
    status('worker_completed',arms=locked['arms'],validation=872,test=188)

def main():
    global MU
    torch.set_num_threads(2);guard();p=verify();locked=preflight(p)
    MU=json.loads((ROOT/'config/calibration.json').read_text())['mu'];assert base.sha(ROOT/'config/calibration.json')==locked['calibration_sha256']
    train(locked);guard();evaluate(locked)
if __name__=='__main__':
    try:main()
    except Exception:
        write(ROOT/'failures'/f'worker_{int(time.time())}.json',dict(traceback=traceback.format_exc(),time_unix=time.time()));raise
