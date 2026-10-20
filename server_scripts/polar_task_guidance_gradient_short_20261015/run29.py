"""Reuse the candidate trainer; only the Q backward connection differs."""
import argparse,copy,gc,json,time
from pathlib import Path
import numpy as np
import torch
import matched_run as old
from common import ROOT,seed,sha,json_save,status,guard_gpu,rng_state,restore_rng
from quality_forward import shared_first,candidate_from_first
original_forward=old.forward
def forward(model,task,batch,arm,b0,capture=False):
    model.error_guidance.task_coupled=(arm=='task_coupled')
    return original_forward(model,task,batch,'real_quality',b0,capture)
old.forward=forward

def cpu():
    old.cpu_verify()  # Includes actual optimizer/RNG restore and all frozen tensors.
    cfg,plan,model,task,raw,source=old.load(torch.device('cpu'))
    small=copy.deepcopy(cfg);small['data']['crop_size']=32
    batch=old.study.base.unpack_batch(next(old.study.microbatches(next(iter(old.staged.loader(small,raw,'candidate',1))))),torch.device('cpu'))
    old.configure(model,task);snapshot=copy.deepcopy(model.state_dict());counts=dict(E=0,D=0,M=0)
    handles=[model.r2p.register_forward_hook(lambda *x:counts.__setitem__('E',counts['E']+1)),task.net_g.register_forward_hook(lambda *x:counts.__setitem__('D',counts['D']+1)),model.action_gate.register_forward_hook(lambda *x:counts.__setitem__('M',counts['M']+1))]
    seed(100);a=forward(model,task,batch,'proxy_only',.03)
    for h in handles:h.remove()
    assert counts==dict(E=2,D=2,M=0)
    t1=old.digest(a.t1);u=old.digest(a.ungated_t2);del a
    seed(100);a=forward(model,task,batch,'task_coupled',.03)
    assert old.digest(a.t1)==t1 and old.digest(a.ungated_t2)==u;del a
    # Initial inject outputs are zero: activate ONLY for the diagnostic and restore.
    with torch.no_grad():
        for module in model.error_guidance.inject.values():module.body[-1].weight.normal_(0,.01)
    norms={}
    for arm in plan['arms']:
        seed(100);a=forward(model,task,batch,arm,.03)
        objective=old.staged.candidate_loss(a,batch['target'])-.1*a.repair_loss
        # Explicit restoration-only expression: detached BCE cannot be subtracted to remove its gradient.
        costs=old.staged.action_costs(a.t1.detach(),a.rgb_increment,batch['target'])
        edge=sum((a.ungated_t2.diff(dim=d)-batch['target'].diff(dim=d)).abs().mean()/2 for d in [-1,-2])
        restoration=(a.ungated_t2-batch['target']).square().mean()+.02*edge+.1*costs[:,-1:].clamp_min(0).mean()
        grads=torch.autograd.grad(restoration,tuple(model.error_guidance.predict.parameters()),allow_unused=True)
        norms[arm]=sum(float(g.abs().sum()) for g in grads if g is not None)
        del a,grads,restoration,objective
    assert norms['proxy_only']==0 and norms['task_coupled']>0,norms
    model.load_state_dict(snapshot);seed(100)
    with torch.no_grad():a=forward(model,task,dict(batch,target=1-batch['target']),'task_coupled',.03)
    assert old.digest(a.ungated_t2)==u;del a
    # A real coupled optimizer step must preserve T1/core and resume exactly.
    old.configure(model,task);opt=old.staged.phase_optimizer(model,'candidate')
    snap=copy.deepcopy(old.checkpoint(model,task,opt,0,'task_coupled',plan))
    def update():
        seed(100);opt.zero_grad(set_to_none=True);out=forward(model,task,batch,'task_coupled',.03)
        assert old.digest(out.t1)==t1
        loss=old.staged.candidate_loss(out,batch['target']);loss.backward()
        assert all(p.grad is None for k,p in model.named_parameters() if not k.startswith(('feedback.','error_guidance.')))
        torch.nn.utils.clip_grad_norm_(old.staged.phase_module(model,'candidate').parameters(),1,error_if_nonfinite=True);opt.step();old.audit(model,task,source)
        return float(loss),{k:v.detach().clone() for k,v in model.state_dict().items()}
    l1,s1=update();model.load_state_dict(snap['r2p_and_projections']);opt.load_state_dict(snap['optimizer']);restore_rng(snap['rng']);l2,s2=update()
    assert l1==l2 and all(torch.equal(v,s2[k]) for k,v in s1.items())
    json_save(ROOT/'diagnostics/coupling_cpu_checks.json',dict(passed=True,initial_outputs_bit_equal=True,counts=counts,GT_invariance=True,restoration_gradient_Q_after_temporary_activation=norms,probe_perturbations_discarded=True,coupled_optimizer_rng_exact=True,frozen_T1_and_core=True))
    status('cpu_all_verified')

def preflight():
    torch.set_num_threads(2);device=torch.device('cuda:0');cfg,plan,model,task,raw,source=old.load(device)
    guard_gpu();old.configure(model,task);times=[]
    for arm in plan['arms']:
        model.load_state_dict(source['r2p_and_projections']);opt=old.staged.phase_optimizer(model,'candidate')
        for i,rb in enumerate(old.staged.loader(cfg,raw,'candidate',2)):
            guard_gpu();tick=time.time();opt.zero_grad(set_to_none=True)
            for micro in old.study.microbatches(rb):
                batch=old.study.base.unpack_batch(micro,device);old.study.base.set_step_seed(cfg['seed'],'preflight_candidate',i,batch['sample_id'])
                out=forward(model,task,batch,arm,.03);(old.staged.candidate_loss(out,batch['target'])/2).backward();del out
            torch.nn.utils.clip_grad_norm_(old.staged.phase_module(model,'candidate').parameters(),1,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();times.append(time.time()-tick)
        old.audit(model,task,source);del opt
    del model,task,source,batch;gc.collect();torch.cuda.empty_cache()
    cfg,plan,model,task,raw,source=old.load(device);gc.collect();torch.cuda.empty_cache()
    from rgb_final_confirmation.data import load_manifest,select_rows,PairedBranchDataset
    from PIL import Image
    rows=select_rows(load_manifest(cfg['paths']['manifest']),'validation');sizes=[]
    for i,row in enumerate(rows):
        with Image.open(row.official_rgb) as im:sizes.append((im.width*im.height,i))
    sample=PairedBranchDataset(rows,group='A1',crop_size=None)[max(sizes)[1]]
    batch={k:v[None].cuda() if torch.is_tensor(v) else [v] for k,v in sample.items()}
    guard_gpu();seed(2026);tick=time.time()
    with torch.no_grad():
        first=shared_first(model,task,batch,study_module=old.study)
        for arm in plan['arms']:
            model.error_guidance.task_coupled=arm=='task_coupled'
            out=candidate_from_first(model,task,first,'real_quality',capture_paths=True,study_module=old.study)
            assert out.path_audit['all_quality_paths_match'];del out
    torch.cuda.synchronize();elapsed=time.time()-tick
    estimate=1.5*(4000*float(np.median(times[1:]))+64*elapsed)+900
    report=dict(step_seconds=float(np.median(times[1:])),paired_frame_seconds=elapsed,estimated_seconds=estimate,peak_vram_mib=torch.cuda.max_memory_allocated()/2**20,shape=list(batch['polar'].shape),original_size=True,tiling=False,updates_discarded=True)
    json_save(ROOT/'diagnostics/benchmark29.json',report);assert estimate<=plan['budget_seconds'],report
    json_save(ROOT/'config/locked_run.json',dict(plan,benchmark=report));json_save(ROOT/'config/lock_receipt.json',dict(protocol_id=plan['protocol_id'],locked_sha256=sha(ROOT/'config/locked_run.json'),preflight_discarded=True));status('locked_ready_to_train')

@torch.no_grad()
def evaluate():
    torch.set_num_threads(2);device=torch.device('cuda:0');cfg,plan,model,task,raw,source=old.load(device)
    states={a:torch.load(ROOT/'arms'/a/'final.pt',map_location='cpu',weights_only=False) for a in plan['arms']}
    for arm,ck in states.items():
        assert ck['step']==2000 and ck['protocol_id']==plan['protocol_id'] and ck['arm']==arm
        for k,v in source['r2p_and_projections'].items():
            if not k.startswith(('feedback.','error_guidance.')):assert torch.equal(v,ck['r2p_and_projections'][k])
        assert all(torch.equal(v,ck['tasknet_net_g'][k]) for k,v in source['tasknet_net_g'].items())
    spec=json.loads((ROOT/'config/screen64.json').read_text());selected={r['sample_id']:r['original_validation_index'] for r in spec['rows']}
    reference=json.loads((ROOT.parent/'polar_quality_mechanism_audit_20261015/config/source27_final_validation.json').read_text());hashes={r['sample_id']:r['t1_sha256'] for r in reference['rows']}
    from rgb_final_confirmation.evaluation import psnr,ssim
    outputs=[]
    for i,rb in enumerate(old.study.base.split_loader(cfg,'validation')):
        if rb['sample_id'][0] not in selected:continue
        guard_gpu();batch=old.study.base.unpack_batch(rb,device);sid=batch['sample_id'][0];assert selected[sid]==i
        old.study.base.set_step_seed(2026,'matched_eval_validation',i,batch['sample_id']);model.eval();task.eval()
        first=shared_first(model,task,batch,study_module=old.study);assert old.digest(first.t1)==hashes[sid]
        record=dict(sample_id=sid,scene=batch['scene'][0],original_validation_index=i,t1_sha256=hashes[sid],t1=dict(psnr=psnr(first.t1,batch['target']),ssim=ssim(first.t1,batch['target'])),arms={})
        arrays=dict(image=batch['official_rgb'][0].cpu().numpy(),gt=batch['target'][0].cpu().numpy(),t1=first.t1[0].cpu().numpy(),b=first.error_scale[0].cpu().numpy(),c=first.confidence_map[0].cpu().numpy())
        for arm in plan['arms']:
            for prefix,module in [('feedback.',model.feedback),('error_guidance.',model.error_guidance)]:module.load_state_dict({k.removeprefix(prefix):v for k,v in states[arm]['r2p_and_projections'].items() if k.startswith(prefix)},strict=True)
            model.error_guidance.task_coupled=arm=='task_coupled';model.eval()
            out=candidate_from_first(model,task,first,'real_quality',capture_paths=True,study_module=old.study);assert out.path_audit['all_quality_paths_match']
            record['arms'][arm]=dict(psnr=psnr(out.ungated_t2,batch['target']),ssim=ssim(out.ungated_t2,batch['target']),u_sha256=old.digest(out.ungated_t2))
            arrays[arm]=out.ungated_t2[0].cpu().numpy();arrays[arm+'_q']=out.repair_need[0].cpu().numpy();del out
        directory=ROOT/'examples/screen64';directory.mkdir(parents=True,exist_ok=True);np.savez_compressed(directory/(sid.replace('/','_')+'.npz'),**arrays)
        outputs.append(record);json_save(ROOT/'metrics/screen64_partial.json',dict(rows=outputs,exploratory=True,original_size=True,tiling=False))
        status('evaluating_screen64',completed=len(outputs),total=64);del first,batch,arrays;gc.collect();torch.cuda.empty_cache()
    assert len(outputs)==64
    json_save(ROOT/'metrics/screen64.json',dict(rows=outputs,protocol_id=plan['protocol_id'],test_used=False,all_T1_hashes_equal=True,original_size=True,tiling=False,exploratory=True))

def summarize():
    result=json.loads((ROOT/'metrics/screen64.json').read_text());rows=result['rows'];names=['t1','proxy_only','task_coupled']
    means=lambda members:{name:{m:float(np.mean([(r['t1'] if name=='t1' else r['arms'][name])[m] for r in members])) for m in ['psnr','ssim']} for name in names}
    per=[dict(scene=s,frames=len([r for r in rows if r['scene']==s]),means=means([r for r in rows if r['scene']==s])) for s in sorted({r['scene'] for r in rows})]
    equal={n:{m:float(np.mean([s['means'][n][m] for s in per])) for m in ['psnr','ssim']} for n in names}
    # All scenes have8 sampled frames; original-scene-size weighting is an estimate, not full872 measurement.
    counts=dict(__import__('collections').Counter(r['scene'] for r in json.loads((ROOT.parent/'polar_quality_mechanism_audit_20261015/config/source27_final_validation.json').read_text())['rows']))
    weighted={n:{m:sum(s['means'][n][m]*counts[s['scene']] for s in per)/sum(counts.values()) for m in ['psnr','ssim']} for n in names}
    wins=sum(s['means']['task_coupled']['psnr']>s['means']['proxy_only']['psnr'] for s in per)
    nonzero=any(r['arms']['task_coupled']['u_sha256']!=r['t1_sha256'] for r in rows)
    success=all(equal['task_coupled']['psnr']>equal[x]['psnr'] and equal['task_coupled']['ssim']>=equal[x]['ssim'] for x in ['t1','proxy_only']) and wins>=6 and nonzero
    json_save(ROOT/'reports/screen_summary.json',dict(exploratory=True,formal_confirmation=False,test_used=False,frames=64,per_scene=per,scene_equal=equal,original_scene_count_weighted_estimate=weighted,coupled_PSNR_scene_wins=wins,nonzero_change=nonzero,preliminary_success=success,stable_efficacy_proven=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','preflight','train','evaluate','summarize']);p.add_argument('--arm');a=p.parse_args()
    if a.mode=='cpu':cpu()
    elif a.mode=='preflight':preflight()
    elif a.mode=='train':old.train(a.arm)
    elif a.mode=='evaluate':evaluate()
    else:summarize()
