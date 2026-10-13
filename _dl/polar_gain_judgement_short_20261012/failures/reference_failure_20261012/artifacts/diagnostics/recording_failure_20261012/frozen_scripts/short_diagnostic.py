"""Full validation oracle diagnostics on actual float RGB; not deployed outputs."""
import gc,hashlib,json
import numpy as np
import torch
from torch.nn import functional as F
from common import ROOT,json_save,guard_gpu
import staged,study

@torch.no_grad()
def oracle(t1,delta,target,side):
    costs=torch.cat([F.avg_pool2d((t1+a*delta-target).square().mean(1,keepdim=True),side,stride=side,ceil_mode=True,count_include_pad=False) for a in staged.ACTIONS],1)
    decision=costs.argmin(1,keepdim=True)
    mask=t1.new_tensor(staged.ACTIONS)[decision].repeat_interleave(side,-2).repeat_interleave(side,-1)[...,:t1.shape[-2],:t1.shape[-1]]
    result=staged.compose(t1,delta,mask)
    assert (result-target).square().mean()<=(t1-target).square().mean()+1e-7
    return result,float((mask>0).float().mean())

@torch.no_grad()
def run(cfg,device,final=False):
    study.base.add_source_path(cfg)
    from rgb_final_confirmation.evaluation import psnr,ssim
    model,task,raw=staged.components(cfg,device)
    if final:staged.restore(model,task,staged.OUT/'checkpoints/final.pt',cfg)
    model.eval();task.eval();rows=[]
    for i,rawbatch in enumerate(study.base.split_loader(cfg,'validation')):
        guard_gpu();batch=study.base.unpack_batch(rawbatch,device);study.base.set_step_seed(cfg['seed'],'validation',i,batch['sample_id'])
        out=staged.forward(model,task,batch);predictions={'t1':out.t1,'candidate':out.ungated_t2};coverage={}
        for side in (4,16,32):predictions[f'oracle_{side}'],coverage[str(side)]=oracle(out.t1,out.rgb_increment,batch['target'],side)
        record=dict(sample_id=batch['sample_id'][0],scene=batch['scene'][0],t1_tensor_sha256=hashlib.sha256(out.t1.cpu().contiguous().numpy().tobytes()).hexdigest(),coverage=coverage,metrics={k:dict(psnr=psnr(v,batch['target']),ssim=ssim(v,batch['target'])) for k,v in predictions.items()})
        rows.append(record)
        if (i+1)%20==0:print(json.dumps(dict(source_oracle_frames=i+1,total=872)),flush=True)
        del out,predictions,batch
    assert len(rows)==len({r['sample_id'] for r in rows})==872
    result=dict(frames=872,per_image=rows,oracle_uses_GT=True,oracle_deployable=False,original_size=True,tiling=False,test_used=False,source_sha256=cfg['reuse']['sha256'],model='final' if final else 'initial_source',means={k:{m:float(np.mean([r['metrics'][k][m] for r in rows])) for m in ('psnr','ssim')} for k in rows[0]['metrics']})
    result['per_scene']=[dict(scene=s,frames=sum(r['scene']==s for r in rows),means={k:{m:float(np.mean([r['metrics'][k][m] for r in rows if r['scene']==s])) for m in ('psnr','ssim')} for k in rows[0]['metrics']}) for s in sorted({r['scene'] for r in rows})]
    json_save(ROOT/'diagnostics'/('final_oracle_validation872.json' if final else 'source_oracle_validation872.json'),result)
    del model,task;gc.collect();torch.cuda.empty_cache()
