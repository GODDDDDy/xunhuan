"""Validation-only, actual-output diagnostic; never an inference use of GT."""
import gc,hashlib,json,time
import numpy as np
import torch
from torch.nn import functional as F
from common import ROOT,json_save,guard_gpu,status
import study

def action_outputs(t1,delta,actions):
    return torch.stack([t1+a*delta for a in actions],1)

def oracle_output(outputs,target):
    # B,A,C,H,W -> B,A,H/4,W/4. No-update is included, so total
    # raw RGB MSE cannot increase. SSIM is separately measured on the real image.
    b,a,c,h,w=outputs.shape
    assert h%4==w%4==0
    cost=F.avg_pool2d((outputs-target[:,None]).square().mean(2),4)
    decision=cost.argmin(1,keepdim=True)
    full=F.interpolate(decision.float(),size=(h,w),mode='nearest').long()
    result=outputs.gather(1,full[:,None].expand(b,1,c,h,w)).squeeze(1)
    return result,decision,cost

def cpu_check():
    torch.manual_seed(2026)
    t=torch.randn(1,3,12,16);d=torch.randn_like(t);gt=torch.randn_like(t)
    outputs=action_outputs(t,d,[0.,.25,.5,1.]);result,decision,cost=oracle_output(outputs,gt)
    assert (result-gt).square().mean()<=(t-gt).square().mean()+1e-7
    keep=F.interpolate((decision==0).float(),size=t.shape[-2:],mode='nearest').bool().expand_as(t)
    assert torch.equal(result[keep],t[keep])
    json_save(ROOT/'diagnostics/oracle_cpu_contracts.json',dict(actual_output_composition=True,includes_no_update=True,mse_nonincrease=True,protected_exact=True,gt_used_only_after_prediction=True))

@torch.no_grad()
def run():
    cpu_check()
    cfg=json.loads((ROOT/'config/diagnostic.json').read_text())
    device=torch.device('cuda');torch.set_num_threads(4)
    study.base.add_source_path(cfg)
    from rgb_final_confirmation.evaluation import psnr,ssim
    model,task,_,rawcfg=study.components(cfg,device,with_loss=False)
    model.step=6000;model.eval();task.eval()
    reference=json.loads((ROOT/'diagnostics/source_fixed_t1_reference.json').read_text())
    hashes={r['sample_id']:r['t1_tensor_sha256'] for r in reference['per_image']}
    rows=[];actions=cfg['diagnostic']['actions'];seconds=[]
    for i,raw in enumerate(study.base.proxy_loader(cfg,rawcfg)):
        guard_gpu();tick=time.perf_counter();batch=study.base.unpack_batch(raw,device)
        assert batch['sample_id'][0].startswith('validation/')
        study.base.set_step_seed(cfg['seed'],'fixed_system_probe',i,batch['sample_id'])
        out=study.recurrent_forward(model,task,batch)
        sample=batch['sample_id'][0]
        t1hash=hashlib.sha256(out.t1.cpu().contiguous().numpy().tobytes()).hexdigest()
        assert t1hash==hashes[sample], 'T1 differs from the inherited fixed first round'
        outputs=action_outputs(out.t1,out.rgb_increment,actions)
        oracle,decision,cost=oracle_output(outputs,batch['target'])
        predictions={'t1':out.t1,'inherited_gate':out.t2,'oracle':oracle}
        predictions.update({f'fixed_{a:g}':outputs[:,j] for j,a in enumerate(actions)})
        record=dict(sample_id=sample,scene=batch['scene'][0],shape=list(out.t1.shape),t1_tensor_sha256=t1hash,metrics={k:dict(psnr=psnr(v,batch['target']),ssim=ssim(v,batch['target'])) for k,v in predictions.items()},oracle_action_fractions=[float((decision==j).float().mean()) for j in range(len(actions))])
        assert record['metrics']['oracle']['psnr']>=record['metrics']['t1']['psnr']-1e-5
        rows.append(record);torch.cuda.synchronize();seconds.append(time.perf_counter()-tick)
        print(json.dumps(dict(frame=i+1,sample=sample,seconds=seconds[-1])),flush=True)
        del out,outputs,oracle,batch,predictions,cost;gc.collect();torch.cuda.empty_cache()
    assert len(rows)==28
    keys=rows[0]['metrics'].keys()
    result=dict(frames=28,per_image=rows,means={k:{m:float(np.mean([r['metrics'][k][m] for r in rows])) for m in ('psnr','ssim')} for k in keys},actions=actions,source_sha256=cfg['reuse']['sha256'],test_used=False,weight_selection=False,oracle_uses_GT=True,oracle_deployable=False,original_size=True,tiling=False,all_first_round_hashes_equal_source=True,mean_full_image_seconds=float(np.mean(seconds)))
    result['per_scene']=[dict(scene=s,frames=sum(r['scene']==s for r in rows),means={k:{m:float(np.mean([r['metrics'][k][m] for r in rows if r['scene']==s])) for m in ('psnr','ssim')} for k in keys}) for s in sorted({r['scene'] for r in rows})]
    json_save(ROOT/'diagnostics/candidate_oracle_validation28.json',result)
    status('diagnostic_completed',frames=28,source_sha256=cfg['reuse']['sha256'])

if __name__=='__main__':run()
