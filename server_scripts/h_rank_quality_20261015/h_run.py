"""One bounded paired H-only screen using existing first-round/data components."""
import copy,csv,gc,hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from common import ROOT,sha,json_save,save,seed,rng_state,restore_rng,guard_gpu,status
import gain22,study
from quality_metrics import spearman_numpy
from h_objective import scale_loss,ranking_loss,quality_metrics


def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load(device):
    plan=json.loads((ROOT/'config/plan.json').read_text())
    assert sha(ROOT/'config/execution_config.json')==plan['execution_config_sha256']
    assert sha(ROOT/'config/validation64.json')==plan['validation64_sha256']
    for entry in plan['code_manifest']:
        assert sha(ROOT/entry['path'])==entry['sha256'],entry['path']
    cfg=json.loads((ROOT/'config/execution_config.json').read_text())
    for key,value in plan['data_sha256'].items():assert sha(cfg['paths'][key])==value,key
    assert sha(ROOT/'weights/common_initial21.pt')==plan['source_initial_sha256']
    seed(plan['seed']);model,task,raw=gain22._components(cfg,device)
    source=torch.load(ROOT/'weights/common_initial21.pt',map_location='cpu',weights_only=False)
    assert source['protocol_id']==plan['source_initial_protocol']
    model.load_state_dict(source['r2p_and_projections'],strict=True)
    task.net_g.load_state_dict(source['tasknet_net_g'],strict=True)
    model.tau=float(source['tau']);model.eval().requires_grad_(False);task.eval().requires_grad_(False)
    heads={arm:copy.deepcopy(model.confidence_head).requires_grad_(True) for arm in plan['arms']}
    audit(model,task,source)
    imported=[]
    for module in list(sys.modules.values()):
        value=getattr(module,'__file__',None)
        if value and Path(value).suffix=='.py' and Path(value).resolve().is_relative_to(ROOT.parent):
            p=Path(value).resolve();imported.append(dict(path=str(p),sha256=sha(p)))
    json_save(ROOT/'diagnostics/runtime_source.json',dict(files=imported,python=sys.version,torch=torch.__version__,numpy=np.__version__))
    return cfg,plan,model,task,raw,source,heads


def audit(model,task,source):
    assert all(torch.equal(v.cpu(),source['r2p_and_projections'][n]) for n,v in model.state_dict().items())
    assert all(torch.equal(v.cpu(),source['tasknet_net_g'][n]) for n,v in task.net_g.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert all(not p.requires_grad and p.grad is None for p in task.parameters())


@torch.no_grad()
def first_inputs(model,task,batch):
    # GT is deliberately absent from every expression here.
    p0,image=batch['polar'],batch['official_rgb']
    first=study.enhance(model,p0,p0,image,project=False)
    t1,features=study.task_forward(model,task,first.refined_polar,image,p0)
    return (image,t1,study.polarization_summary(p0),features)


def target_error(inputs,batch):
    return F.avg_pool2d((inputs[1].detach().clamp(0,1)-batch['target']).abs().mean(1,keepdim=True),4)


def opts_for(heads,plan):
    return {arm:torch.optim.Adam(head.parameters(),lr=plan['lr']) for arm,head in heads.items()}


def synthetic_checks():
    # Correct direction: a reversed pair must be worse and gradients push the
    # high-error location upward in predicted error. No valid pairs => zero.
    e=torch.tensor([[[[.01,.1]]]])
    correct=torch.tensor([[[[.01,.1]]]],requires_grad=True)
    wrong=torch.tensor([[[[.1,.01]]]],requires_grad=True)
    good,_=ranking_loss(correct,e,123);bad,_=ranking_loss(wrong,e,123)
    assert good<bad;bad.backward();assert wrong.grad[0,0,0,0]>0 and wrong.grad[0,0,0,1]<0
    flat=torch.full_like(wrong,.04,requires_grad=True)
    zero,count=ranking_loss(flat,torch.ones_like(flat)*.02,123)
    zero.backward();assert count==0 and float(zero)==0 and torch.count_nonzero(flat.grad)==0
    oracle=quality_metrics(np.arange(1,101),np.arange(1,101),spearman_numpy)
    reverse=quality_metrics(np.arange(100,0,-1),np.arange(1,101),spearman_numpy)
    assert abs(oracle['ause'])<1e-12 and reverse['ause']>oracle['ause'] and oracle['worst20_recall']==1


def cpu_check():
    synthetic_checks();cfg,plan,model,task,raw,source,heads=load(torch.device('cpu'))
    cfg=copy.deepcopy(cfg);cfg['data']['crop_size']=32
    batch=study.base.unpack_batch(next(study.microbatches(next(iter(study.base.scheduled_loader(cfg,study.base.build_base_config(cfg),1))))),torch.device('cpu'))
    seed(111);inputs=first_inputs(model,task,batch)
    seed(111);changed=first_inputs(model,task,dict(batch,target=1-batch['target']))
    assert torch.equal(inputs[1],changed[1])
    assert all(torch.equal(heads[a](*inputs),model.confidence_head(*inputs)) for a in heads)
    error=target_error(inputs,batch);opts=opts_for(heads,plan)
    snapshots={a:copy.deepcopy(h.state_dict()) for a,h in heads.items()}
    # A real paired training update verifies checkpoint/optimizer replay exactly.
    def one():
        for opt in opts.values():opt.zero_grad(set_to_none=True)
        for micro in range(2):
            for arm,head in heads.items():
                b=head(*inputs);rank,_=ranking_loss(b,error,101+micro)
                loss=scale_loss(b,error)+(.1*rank if arm=='scale_plus_rank' else 0)
                (loss/2).backward()
        for arm,head in heads.items():
            torch.nn.utils.clip_grad_norm_(head.parameters(),1,error_if_nonfinite=True);opts[arm].step()
        audit(model,task,source)
        return {a:copy.deepcopy(h.state_dict()) for a,h in heads.items()}
    expected=one()
    for a,h in heads.items():h.load_state_dict(snapshots[a])
    opts=opts_for(heads,plan);actual=one()
    assert all(torch.equal(v,actual[a][k]) for a,s in expected.items() for k,v in s.items())
    assert any(not torch.equal(v,expected[a][k]) for a,s in snapshots.items() for k,v in s.items())
    json_save(ROOT/'diagnostics/cpu_checks.json',dict(passed=True,ranking_direction=True,
             no_valid_pairs_zero_gradient=True,oracle_AUSE_zero=True,GT_invariance=True,
             original_H_exact_forward=True,frozen_source_exact=True,real_head_updates=True,
             fresh_optimizer_replay_exact=True,diagnostic_updates_discarded=True))
    status('cpu_verified')


def checkpoint(heads,opts,step,plan):
    return dict(heads={a:h.state_dict() for a,h in heads.items()},optimizers={a:o.state_dict() for a,o in opts.items()},
                rng=rng_state(),step=step,protocol_id=plan['protocol_id'],source_initial_sha256=plan['source_initial_sha256'])


def run():
    assert json.loads((ROOT/'diagnostics/cpu_checks.json').read_text())['passed']
    assert not (ROOT/'budget.json').exists(),'Inspect prior run; never reset budget'
    plan=json.loads((ROOT/'config/plan.json').read_text());start=time.time()
    json_save(ROOT/'budget.json',dict(start_unix=start,deadline_unix=start+plan['budget_seconds'],
              train_deadline_unix=start+plan['training_seconds'],immutable=True,pid=os.getpid()))
    device=torch.device('cuda:0');cfg,plan,model,task,raw,source,heads=load(device)
    guard_gpu();opts=opts_for(heads,plan)
    # Discard a real GPU update and verify the exact inherited H forward first.
    rawbatch=next(iter(study.base.scheduled_loader(cfg,raw,1)))
    batch=study.base.unpack_batch(next(study.microbatches(rawbatch)),device)
    seed(111);inputs=first_inputs(model,task,batch);error=target_error(inputs,batch)
    with torch.no_grad():
        original=model.confidence_head(*inputs)
        assert all(torch.equal(h(*inputs),original) for h in heads.values())
    for arm,h in heads.items():
        loss=scale_loss(h(*inputs),error);loss.backward();opts[arm].step()
        h.load_state_dict(model.confidence_head.state_dict())
    audit(model,task,source);opts=opts_for(heads,plan)
    del inputs,error,original,batch;gc.collect();torch.cuda.empty_cache()
    json_save(ROOT/'diagnostics/gpu_checks.json',dict(passed=True,actual_backward=True,
             original_H_exact=True,first_round_frozen=True,updates_discarded=True))
    save(ROOT/'weights/initial_heads.pt',checkpoint(heads,opts,0,plan))
    seen=set();training=ROOT/'logs/training.csv';timings=[]
    with training.open('w',newline='') as f:
        writer=None
        for step,rb in enumerate(study.base.scheduled_loader(cfg,raw,plan['steps']),1):
            guard_gpu();assert time.time()<start+plan['training_seconds'],'Fixed training deadline reached'
            tick=time.time();ids=[];values={a:[] for a in heads}
            for opt in opts.values():opt.zero_grad(set_to_none=True)
            for mi,micro in enumerate(study.microbatches(rb)):
                batch=study.base.unpack_batch(micro,device);ids+=batch['sample_id']
                assert all(x.startswith('train/') for x in batch['sample_id'])
                study.base.set_step_seed(plan['seed'],'H_only_pair_training',step,batch['sample_id'])
                inputs=first_inputs(model,task,batch);error=target_error(inputs,batch)
                # Each head sees exactly the same tensors. No optimizer/RNG state
                # is shared. GT is used only by the two objectives.
                for arm,head in heads.items():
                    head.train();b=head(*inputs);base=scale_loss(b,error)
                    rank,count=ranking_loss(b,error,plan['seed']+10007*step+mi,
                                            plan['rank_pairs'],plan['rank_min_error_gap'])
                    loss=base+(.1*rank if arm=='scale_plus_rank' else 0.)
                    assert torch.isfinite(loss);(loss/2).backward()
                    values[arm].append((float(loss.detach()),float(base.detach()),float(rank.detach()),count))
                del inputs,error,b,base,rank,loss,batch
            for arm,head in heads.items():
                torch.nn.utils.clip_grad_norm_(head.parameters(),1,error_if_nonfinite=True);opts[arm].step()
            seen.update(ids);torch.cuda.synchronize();timings.append(time.time()-tick)
            record=dict(step=step,sample_ids=';'.join(ids),seconds=timings[-1])
            for arm,v in values.items():
                for j,key in enumerate(('loss','scale_loss','rank_loss','valid_pairs')):record[arm+'_'+key]=float(np.mean(v,axis=0)[j])
            if writer is None:writer=csv.DictWriter(f,fieldnames=record);writer.writeheader()
            writer.writerow(record);f.flush()
            if step%100==0:status('training',step=step,total=plan['steps'],median_step_seconds=float(np.median(timings[-100:])))
            if step%500==0:
                audit(model,task,source);save(ROOT/'weights'/f'step_{step}.pt',checkpoint(heads,opts,step,plan))
    assert step==plan['steps'];audit(model,task,source)
    save(ROOT/'weights/final_heads.pt',checkpoint(heads,opts,step,plan))
    json_save(ROOT/'diagnostics/paired_training.json',dict(steps_per_arm=step,crops_per_arm=step*2,
              unique_training_images=len(seen),same_first_round_tensors=True,all_non_H_tensors_equal_source=True,
              endpoint_sha256=sha(ROOT/'weights/final_heads.pt'),training_seconds=time.time()-start))
    passed=evaluate(cfg,plan,model,task,heads)
    if passed:
        status('full_confirmation_starting',weights_locked=True,no_more_training=True)
        evaluate(cfg,plan,model,task,heads,split='validation')
        evaluate(cfg,plan,model,task,heads,split='test')
        json_save(ROOT/'full_confirmation.json',dict(completed=True,validation=872,test=188,
                  steps_per_arm=plan['steps'],weights_locked_before_test=True,T2_evaluated=False,
                  protocol_id=plan['protocol_id'],time_unix=time.time()))
        status('full_confirmation_completed',validation=872,test=188,steps_per_arm=plan['steps'])


@torch.no_grad()
def evaluate(cfg,plan,model,task,heads,split=None):
    from rgb_final_confirmation.data import load_manifest,select_rows,PairedBranchDataset
    from rgb_final_confirmation.evaluation import psnr,ssim
    selected=json.loads((ROOT/'config/validation64.json').read_text())['sample_ids'] if split is None else None
    byid={r.sample_id:r for r in select_rows(load_manifest(cfg['paths']['manifest']),split or 'validation')}
    selected=selected or list(byid)
    label='validation64' if split is None else split
    dataset=PairedBranchDataset([byid[s] for s in selected],group='A1',crop_size=None)
    heads=dict(original=model.confidence_head,**heads);rows=[]
    for i in range(len(dataset)):
        guard_gpu();item=dataset[i]
        batch={k:v[None].cuda() if torch.is_tensor(v) else [v] for k,v in item.items()}
        study.base.set_step_seed(plan['seed'],'H_only_pair_eval',0,batch['sample_id'])
        inputs=first_inputs(model,task,batch);error=target_error(inputs,batch)
        e=error.cpu().numpy();pred={a:h.eval()(*inputs).cpu().numpy() for a,h in heads.items()}
        row=dict(sample_id=batch['sample_id'][0],scene=batch['scene'][0],
                 t1_sha256=digest(inputs[1]),shape=list(inputs[1].shape),
                 t1_psnr=psnr(inputs[1],batch['target']),t1_ssim=ssim(inputs[1],batch['target']),
                 heads={a:quality_metrics(b,e,spearman_numpy) for a,b in pred.items()})
        rows.append(row)
        # Keep the floating quality maps for every screening image on server.
        directory=ROOT/'examples'/label;directory.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(directory/(row['sample_id'].replace('/','_')+'.npz'),
                            error=e,t1=inputs[1].cpu().numpy(),**pred)
        json_save(ROOT/'metrics'/f'{label}_partial.json',dict(rows=rows,original_size=True,tiling=False))
        if (i+1)%8==0:status('evaluating_H',split=label,completed=i+1,total=len(dataset))
        del inputs,error,batch,pred,e;gc.collect();torch.cuda.empty_cache()
    metrics=('ause','spearman_b_error','worst20_precision','worst20_recall','b_mae','scale_loss')
    scenes={s:{a:{m:float(np.mean([r['heads'][a][m] for r in rows if r['scene']==s and r['heads'][a][m] is not None]))
                       for m in metrics} for a in heads} for s in sorted({r['scene'] for r in rows})}
    macro={a:{m:float(np.mean([v[a][m] for v in scenes.values()])) for m in metrics} for a in heads}
    baseline,candidate=macro['scale_only'],macro['scale_plus_rank']
    improved=sum(v['scale_plus_rank']['ause']<v['scale_only']['ause'] for v in scenes.values())
    passed=(baseline['ause']-candidate['ause']>=.005 and improved>=6
            and candidate['worst20_recall']>=baseline['worst20_recall']
            and candidate['scale_loss']<=baseline['scale_loss']+.02)
    report=dict(kind=plan['kind'],protocol_id=plan['protocol_id'],frames=len(rows),split=label,
                rows=rows,scenes=scenes,scene_equal=macro,ause_scenes_improved=improved,
                screening_passed=bool(passed) if split is None else None,full_confirmation=split is not None,
                conclusion=('Proceed once to full confirmation with locked heads' if passed else 'No clear matched improvement under locked rule; keep original H; stop this candidate') if split is None else 'Locked endpoint evaluation; do not select/retrain using test',
                test_used_for_training_or_selection=False,T2_evaluated=False)
    json_save(ROOT/'metrics'/f'{label}.json',report)
    json_save(ROOT/f'completion_{label}.json',dict(protocol_id=plan['protocol_id'],steps_per_arm=plan['steps'],frames=len(rows),split=label,
               screening_passed=bool(passed),full_confirmation=False,source_sha256=sha(ROOT/'weights/common_initial21.pt'),
               final_heads_sha256=sha(ROOT/'weights/final_heads.pt'),time_unix=time.time()))
    files=[p for folder in ('scripts','config','weights','metrics','examples','diagnostics') for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    json_save(ROOT/'artifact_manifest.json',dict(files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]))
    status('screen_completed' if split is None else 'split_completed',steps_per_arm=plan['steps'],split=label,frames=len(rows),screening_passed=bool(passed) if split is None else None,macro=macro,ause_scenes_improved=improved)
    return bool(passed)


if __name__=='__main__':
    torch.set_num_threads(2)
    try:
        if sys.argv[1]=='cpu':cpu_check()
        elif sys.argv[1]=='run':run()
        else:raise ValueError('Expected cpu or run')
    except Exception:
        json_save(ROOT/'diagnostics'/f'failure_{int(time.time())}.json',dict(traceback=traceback.format_exc(),time_unix=time.time()))
        status('failed',traceback=traceback.format_exc());raise
