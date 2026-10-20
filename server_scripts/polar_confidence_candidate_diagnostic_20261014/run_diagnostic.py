"""Frozen candidate diagnostic; original archived model and loaders are reused."""
import argparse,copy,gc,hashlib,json,os,subprocess,sys,time
from types import SimpleNamespace
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from common import ROOT,guard_gpu,json_save,seed,sha,status
import gain22,staged,study
from condition_forward import CONDITIONS,shared_first,candidate_from_first,quality_conditions

def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def rng_digest():return digest(torch.get_rng_state())
def load(device):
    cfg=json.loads((ROOT/'config/execution_config.json').read_text())
    plan=json.loads((ROOT/'config/locked_run.json').read_text())
    assert sha(ROOT/'config/execution_config.json')==plan['execution_config_sha256']
    assert sha(plan['source_checkpoint'])==plan['source_checkpoint_sha256']
    for item in plan['diagnostic_code_manifest']:
        assert sha(ROOT/item['path'])==item['sha256'],item['path']
    seed(cfg['seed'])
    model,task,raw=gain22.components(cfg,device)
    state=torch.load(plan['source_checkpoint'],map_location='cpu',weights_only=False)
    assert state['protocol_id']==plan['source_protocol_id']
    model.load_state_dict(state['r2p_and_projections'],strict=True)
    task.net_g.load_state_dict(state['tasknet_net_g'],strict=True)
    for name in ('tau','cost_scale','cost_bias','safety_margin'):setattr(model,name,state[name])
    model.step=int(state['step']);model.eval();task.eval();model.requires_grad_(False);task.requires_grad_(False)
    assert all(torch.equal(v.cpu(),state['r2p_and_projections'][k]) for k,v in model.state_dict().items())
    assert all(torch.equal(v.cpu(),state['tasknet_net_g'][k]) for k,v in task.net_g.state_dict().items())
    json_save(ROOT/'diagnostics/source_load_receipt.json',dict(source_sha256=plan['source_checkpoint_sha256'],source_protocol=state['protocol_id'],model_tensors=len(state['r2p_and_projections']),net_g_tensors=len(state['tasknet_net_g']),strict_equal=True,all_frozen=True,tau=float(model.tau),step=model.step))
    return cfg,plan,model,task,raw,state

def compare(a,b):
    result={}
    for name in ('t1','error_scale','confidence_map','ungated_t2','rgb_increment'):
        x,y=getattr(a,name).detach().cpu(),getattr(b,name).detach().cpu()
        result[name]=dict(equal=torch.equal(x,y),maxabs=float((x-y).abs().max()))
        assert torch.equal(x,y),(name,result[name])
    return result

@torch.no_grad()
def cpu_verify():
    torch.set_num_threads(2);cfg,plan,model,task,raw,state=load(torch.device('cpu'))
    small=copy.deepcopy(cfg);small['data'].update(crop_size=64,workers=0)
    rawbatch=next(iter(staged.loader(small,raw,'judgement',1)))
    batch=study.base.unpack_batch(next(study.microbatches(rawbatch)),torch.device('cpu'))
    study.base.set_step_seed(cfg['seed'],'cpu_baseline',0,batch['sample_id']);baseline=study.recurrent_forward(model,task,batch)
    study.base.set_step_seed(cfg['seed'],'cpu_baseline',0,batch['sample_id']);first=shared_first(model,task,batch,study_module=study)
    out=candidate_from_first(model,task,first,'real_quality',study_module=study)
    exact=compare(baseline,out);b0=float(first.error_scale.mean());del baseline,out
    counts=dict(E=0,D=0,M_old=0,M_new=0)
    handles=[]
    for name,module in (('E',model.r2p),('D',task.net_g),('M_old',model.update_gate),('M_new',model.action_gate)):
        handles.append(module.register_forward_hook(lambda *a,name=name:counts.__setitem__(name,counts[name]+1)))
    study.base.set_step_seed(cfg['seed'],'cpu_invariance',0,batch['sample_id']);first=shared_first(model,task,batch,study_module=study)
    entry=rng_digest();records={}
    for condition in CONDITIONS:
        out=candidate_from_first(model,task,first,condition,fixed_b0=b0,study_module=study)
        assert entry==rng_digest();assert torch.equal(first.t1,out.t1)
        assert out.path_audit['all_quality_paths_match']
        records[condition]=dict(t1_sha256=digest(out.t1),u_sha256=digest(out.ungated_t2),path_audit=out.path_audit)
        if condition=='shifted_quality':
            b,c,_=quality_conditions(first,condition)
            assert torch.equal(torch.sort(b.flatten()).values,torch.sort(first.error_scale.flatten()).values)
            assert torch.equal(torch.sort(c.flatten()).values,torch.sort(first.confidence_map.flatten()).values)
        del out
    assert counts==dict(E=4,D=4,M_old=0,M_new=0),counts
    for handle in handles:handle.remove()
    altered=dict(batch,target=1-batch['target'])
    study.base.set_step_seed(cfg['seed'],'cpu_invariance',0,batch['sample_id']);replacement=shared_first(model,task,altered,study_module=study)
    out=candidate_from_first(model,task,replacement,'real_quality',study_module=study)
    assert digest(out.ungated_t2)==records['real_quality']['u_sha256']
    assert all(torch.equal(v.cpu(),state['r2p_and_projections'][k]) for k,v in model.state_dict().items())
    receipt=dict(passed=True,protocol_id=plan['protocol_id'],shape=list(batch['polar'].shape),source_real_exact=exact,conditions=records,gt_replacement_invariant=True,all_condition_t1_equal=True,source_tensors_unchanged=True,rng_after_branch_unchanged=True,actual_forward_counts=counts,shift_histograms_exact=True,training_steps=0)
    json_save(ROOT/'diagnostics/cpu_checks.json',receipt);status('cpu_verified',protocol_id=plan['protocol_id']);print('CPU_NUMERICAL_CHECKS_PASS',flush=True)

@torch.no_grad()
def calibrate(cfg,model,task,raw,device):
    total=0.;pixels=0;rows=[]
    schedule=json.loads(Path(cfg['staged']['schedules']['judgement']).read_text())['items']
    for i,rawbatch in enumerate(staged.loader(cfg,raw,'judgement',8)):
        for j,micro in enumerate(study.microbatches(rawbatch)):
            guard_gpu();batch=study.base.unpack_batch(micro,device)
            study.base.set_step_seed(cfg['seed'],'quality_constant_calibration',i,batch['sample_id'])
            first=shared_first(model,task,batch,study_module=study)
            assert batch['sample_id'][0].startswith('train/') and batch['scene'][0] in cfg['staged']['scene_roles']['judgement']
            total+=float(first.error_scale.double().sum());pixels+=first.error_scale.numel()
            rows.append(dict(sample_id=batch['sample_id'][0],scene=batch['scene'][0],schedule=schedule[i*2+j],b_mean=float(first.error_scale.mean()),h_pixels=first.error_scale.numel()))
            del first,batch
    assert len(rows)==16
    b0=total/pixels;c0=float(np.exp(-b0/model.tau))
    assert b0>0 and np.isfinite(b0)
    json_save(ROOT/'config/fixed_quality_calibration.json',dict(b0=b0,c0=c0,tau=float(model.tau),samples=rows,training_crops=16,phase='quality_constant_calibration',crop=128,rule='global H prediction pixel mean; no GT statistic',validation_used=False,test_used=False))
    return b0

def example_path(sample_id):return ROOT/'examples'/(sample_id.replace('/','_')+'.npz')

@torch.no_grad()
def gpu_run():
    assert json.loads((ROOT/'diagnostics/cpu_checks.json').read_text())['passed']
    torch.set_num_threads(2);guard_gpu();device=torch.device('cuda:0');cfg,plan,model,task,raw,state=load(device)
    status('calibrating_constant',protocol_id=plan['protocol_id']);b0=calibrate(cfg,model,task,raw,device)
    from rgb_final_confirmation.evaluation import psnr,ssim
    source=json.loads((Path(plan['source_run'])/'downstream/metrics/mechanism_validation.json').read_text())['per_image']
    reference={r['sample_id']:r['conditions']['normal'] for r in source}
    rows=[];seen_scenes=set();maxpsnr=maxssim=0.;peak=0.;start=time.time()
    for i,rawbatch in enumerate(study.base.proxy_loader(cfg,raw)):
        guard_gpu();batch=study.base.unpack_batch(rawbatch,device);sid=batch['sample_id'][0];scene=batch['scene'][0]
        assert sid==source[i]['sample_id']
        tick=time.time();study.base.set_step_seed(cfg['seed'],'mechanism_probe',i,batch['sample_id'])
        torch.cuda.reset_peak_memory_stats()
        # Full-size baseline comparison once; it has its own legacy M after U.
        if i==0:
            baseline_gpu=study.recurrent_forward(model,task,batch)
            baseline=SimpleNamespace(**{n:getattr(baseline_gpu,n).detach().cpu() for n in ('t1','error_scale','confidence_map','ungated_t2','rgb_increment')})
            del baseline_gpu;gc.collect();torch.cuda.empty_cache()
        study.base.set_step_seed(cfg['seed'],'mechanism_probe',i,batch['sample_id'])
        first=shared_first(model,task,batch,study_module=study)
        if i==0:
            real=candidate_from_first(model,task,first,'real_quality',fixed_b0=b0,study_module=study)
            json_save(ROOT/'diagnostics/gpu_source_exact.json',dict(passed=True,sample_id=sid,source_real_exact=compare(baseline,real),original_size=True,shape=list(batch['polar'].shape)))
            del baseline,real
        t1hash=digest(first.t1);t1psnr=psnr(first.t1,batch['target']);t1ssim=ssim(first.t1,batch['target'])
        conditions={};real_u=None;example=scene not in seen_scenes
        arrays=dict(image=batch['official_rgb'][0].cpu().numpy(),target=batch['target'][0].cpu().numpy(),t1=first.t1[0].cpu().numpy(),b_pred=first.error_scale[0].cpu().numpy(),c_pred=first.confidence_map[0].cpu().numpy()) if example else None
        for name in CONDITIONS:
            guard_gpu();out=candidate_from_first(model,task,first,name,fixed_b0=b0,study_module=study)
            assert out.path_audit['all_quality_paths_match'];assert digest(out.t1)==t1hash
            u=out.ungated_t2
            if real_u is None:real_u=u.detach().clone()
            up=psnr(u,batch['target']);us=ssim(u,batch['target']);change=(u-real_u).abs()
            conditions[name]=dict(t1_psnr=t1psnr,t1_ssim=t1ssim,u_psnr=up,u_ssim=us,u_vs_t1_psnr=up-t1psnr,u_vs_t1_ssim=us-t1ssim,
                u_t1_mae=float((u-first.t1).abs().mean()),u_vs_real_mae=float(change.mean()),u_vs_real_maxabs=float(change.max()),
                t1_sha256=t1hash,u_sha256=digest(u),b_pred_mean=float(first.error_scale.mean()),c_pred_mean=float(first.confidence_map.mean()),
                b_cond_mean=float(out.conditioned_error_scale.mean()),c_cond_mean=float(out.conditioned_confidence_map.mean()),path_audit=out.path_audit)
            if arrays is not None:
                arrays['u_'+name]=u[0].cpu().numpy();arrays['c_'+name]=out.conditioned_confidence_map[0].cpu().numpy()
            if name=='real_quality':
                dp=abs(up-reference[sid]['candidate_psnr']);ds=abs(us-reference[sid]['candidate_ssim']);maxpsnr=max(maxpsnr,dp);maxssim=max(maxssim,ds)
                assert dp<=1e-6 and ds<=1e-6,(sid,dp,ds)
            del out,u,change
        if arrays is not None:
            np.savez_compressed(example_path(sid),**arrays);seen_scenes.add(scene)
        peak=max(peak,torch.cuda.max_memory_allocated()/2**20)
        rows.append(dict(sample_id=sid,scene=scene,shape=list(batch['polar'].shape),seconds=time.time()-tick,conditions=conditions))
        metadata=dict(expected_images=28,protocol_id=plan['protocol_id'],source_checkpoint_sha256=plan['source_checkpoint_sha256'],condition_labels=dict(real_quality='原始质量图',fixed_quality='训练均值固定质量',shifted_quality='空间错位质量图'),original_size=True,tiling=False,training_steps=0,test_frames=0,seed=2026,baseline_phase='mechanism_probe',quality_constant_b0=b0,source_normal_psnr_maxabs=maxpsnr,source_normal_ssim_maxabs=maxssim)
        json_save(ROOT/'metrics/per_image.json',dict(rows=rows,run_metadata=metadata))
        status('evaluating_candidates',completed=i+1,total=28,sample_id=sid,seconds=rows[-1]['seconds'])
        del first,batch,real_u;gc.collect();torch.cuda.empty_cache()
    assert len(rows)==28 and len(seen_scenes)==8
    assert all(torch.equal(v.cpu(),state['r2p_and_projections'][k]) for k,v in model.state_dict().items())
    assert all(torch.equal(v.cpu(),state['tasknet_net_g'][k]) for k,v in task.net_g.state_dict().items())
    json_save(ROOT/'diagnostics/gpu_receipt.json',dict(passed=True,frames=28,scenes=8,conditions=3,source_normal_psnr_maxabs=maxpsnr,source_normal_ssim_maxabs=maxssim,model_and_task_tensors_unchanged=True,all_quality_paths_match=True,t1_equal_all_conditions=True,peak_vram_mib=peak,evaluation_seconds=time.time()-start,test_used=False,training_steps=0))
    status('gpu_diagnostic_complete',completed=28,protocol_id=plan['protocol_id'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['cpu_verify','gpu_run'],required=True);args=parser.parse_args()
    if args.mode=='cpu_verify':cpu_verify()
    else:gpu_run()
