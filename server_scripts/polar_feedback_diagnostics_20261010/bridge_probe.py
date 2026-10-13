"""Fixed-weight validation-only check of the observation-anchored bridge."""
import gc,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from diagnose import SOURCE,state_hash,difference,guard
from bridge import anchored_task_inputs
import study
from common import sha,json_save,seed

HERE=Path(__file__).resolve().parent

@torch.no_grad()
def run():
    guard();torch.set_num_threads(4);device=torch.device('cuda:0')
    cfg=json.loads((SOURCE/'config/locked_run.json').read_text());cfg['paths']['output_root']=str(HERE/'outputs')
    # Construct existing models without allocating an unused training criterion.
    study.base.add_source_path(cfg)
    from rgb_final_confirmation.polarfree import OfficialPolarFreeTask,official_polarfree_task_inputs
    from rgb_final_confirmation.evaluation import psnr,ssim
    from enhancer import PolarEnhancer,JointEnhancer
    rawcfg=study.base.build_base_config(cfg);task=OfficialPolarFreeTask(rawcfg,device)
    seed(cfg['seed'])
    model=JointEnhancer(PolarEnhancer()).to(device)
    path=SOURCE/'downstream/checkpoints/final.pt'
    assert sha(path)=='3be8b40c1f06cc0b9828f6cfc23deaaba6db9a3be1a7a28b6bb4a5cca8fb8d76'
    ck=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(ck['r2p_and_projections'],strict=True);task.net_g.load_state_dict(ck['tasknet_net_g'],strict=True)
    model.tau=ck['tau'];model.step=8000;del ck
    model.eval().requires_grad_(False);task.eval().requires_grad_(False)
    before=dict(enhancer=state_hash(model),task=state_hash(task));rows=[];tick=time.time()
    traces=[]
    h1=task.model.net_le_dm.register_forward_hook(lambda m,x,y:traces.append({'condition':y.detach().cpu()}))
    h2=task.net_g.register_forward_pre_hook(lambda m,x:traces[-1].update(prior=x[1].detach().cpu()))
    for i,raw in enumerate(study.base.proxy_loader(cfg,rawcfg)):
        guard();batch=study.base.unpack_batch(raw,device);p0,image=batch['polar'],batch['official_rgb']
        assert batch['sample_id'][0].startswith('validation/')
        ref=official_polarfree_task_inputs(p0,image);zero=anchored_task_inputs(p0,p0,image)
        assert all(torch.equal(x,y) for x,y in zip(ref,zero));del ref,zero
        study.base.set_step_seed(cfg['seed'],'feedback_signal_diagnostic',i,batch['sample_id'])
        cpu,cuda=torch.get_rng_state(),torch.cuda.get_rng_state_all()
        first=study.enhance(model,p0,p0,image);first.raw_features.clear();torch.cuda.empty_cache()
        runs={};psecond={};outputs={}
        for name in ('official_quantized','anchored_float'):
            traces.clear()
            def task_call(p):
                guard()
                with torch.random.fork_rng(devices=[0]):
                    torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda)
                    inp=official_polarfree_task_inputs(p,image) if name=='official_quantized' else anchored_task_inputs(p,p0,image)
                    return task.forward_inputs(inp)
            t1,f1=task_call(first.refined_polar)
            b=model.confidence_head(image,t1,study.polarization_summary(p0),f1)
            c=torch.exp(-b/model.tau);f1.pop('T1',None)
            gate=F.interpolate(c,size=p0.shape[-2:],mode='bilinear',align_corners=False)
            second=study.enhance(model,p0,first.refined_polar,image,t1.clamp(0,1),f1,gate)
            second.raw_features.clear();f1.clear();torch.cuda.empty_cache()
            t2,_=task_call(second.refined_polar)
            metrics=lambda t:dict(psnr=psnr(t,batch['target']),ssim=ssim(t,batch['target']),
                                  mae=float((t.clamp(0,1)-batch['target']).abs().mean()))
            in1,in2=[(official_polarfree_task_inputs(p,image) if name=='official_quantized' else anchored_task_inputs(p,p0,image))
                     for p in (first.refined_polar,second.refined_polar)]
            runs[name]=dict(round1=metrics(t1),round2=metrics(t2),confidence_mean=float(c.mean()),
                            gray=difference(torch.cat(in1[1:5],1),torch.cat(in2[1:5],1)),
                            condition=difference(traces[0]['condition'],traces[1]['condition']),
                            prior=difference(traces[0]['prior'],traces[1]['prior']),task_output=difference(t1,t2),
                            polar=difference(first.refined_polar,second.refined_polar))
            psecond[name]=second.refined_polar.detach().cpu();outputs[name]=t2.detach().cpu()
            del second,t1,t2,b,c,gate,in1,in2;traces.clear();gc.collect();torch.cuda.empty_cache()
        rows.append(dict(sample_id=batch['sample_id'][0],scene=batch['scene'][0],runs=runs,
                         zero_correction_official_inputs_exact=True,
                         candidate_vs_official_polar=difference(psecond['anchored_float'],psecond['official_quantized']),
                         candidate_vs_official_output=difference(outputs['anchored_float'],outputs['official_quantized'])))
        json_save(HERE/'outputs/bridge_progress.json',dict(frames=len(rows),seconds=time.time()-tick,last=rows[-1]))
        print(json.dumps({'frames':len(rows),'candidate_delta':runs['anchored_float']['round2']['psnr']-runs['anchored_float']['round1']['psnr']}),flush=True)
        del first,batch,p0,image,psecond,outputs;gc.collect();torch.cuda.empty_cache()
    h1.remove();h2.remove();after=dict(enhancer=state_hash(model),task=state_hash(task));assert before==after
    assert len(rows)==28
    result=dict(training=False,test_used=False,frames=28,original_size=True,tiling=False,paired_noise=True,
                source_checkpoint_sha256=sha(path),weights_before=before,weights_after=after,
                per_image=rows,seconds=time.time()-tick,candidate_is_untrained_interface=True)
    json_save(HERE/'outputs/bridge_results.json',result)
    json_save(HERE/'bridge_status.json',dict(stage='completed',time_unix=time.time()))

if __name__=='__main__':
    try:run()
    except BaseException as exc:
        json_save(HERE/'bridge_status.json',dict(stage='failed',error=repr(exc),time_unix=time.time()));raise
