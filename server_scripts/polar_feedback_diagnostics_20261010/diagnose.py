"""Read-only, fixed-validation diagnosis of the completed enhancer experiment.

Reuse its exact enhancer, task wrapper, data and metrics. No training or test read.
"""
import argparse, gc, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'polar_enhancer_v2_20261009'
sys.path.insert(0, str(SOURCE / 'scripts'))
import study
from common import sha, json_save

DEADLINE = 1791605047.4916465

def guard():
    if time.time() >= DEADLINE:
        raise RuntimeError('Original GPU authorization expired')

def state_hash(module):
    h = hashlib.sha256()
    for n, v in module.state_dict().items():
        h.update(n.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def difference(a, b, periodic=False):
    d = (a - b).abs()
    if periodic:
        d = torch.minimum(d, (torch.pi - d).abs())
    return dict(mean_abs=float(d.mean()), max_abs=float(d.max()),
                changed_fraction=float((d != 0).float().mean()))

def float_inputs(polar, image):
    # Candidate interface only; do not silently replace the official forward.
    g = [(a.clamp(0, 1) * a.new_tensor([.299,.587,.114]).view(1,3,1,1)).sum(1,keepdim=True)
         for a in polar.chunk(4,1)]
    g0,g45,g90,g135 = [x.double() for x in g]
    intensity = .5*(g0+g45+g90+g135)+1e-4
    q,u = g0-g90,g45-g135
    q = torch.where(q==0,q.new_tensor(1e-4),q)
    return [image,*g,(.5*torch.atan2(u,q)).to(polar.dtype),
            (torch.sqrt(q*q+u*u)/intensity).clamp(max=1).to(polar.dtype)]

def interface_stats(p1,p2,image):
    from rgb_final_confirmation.polarfree import official_polarfree_task_inputs
    one,two = [official_polarfree_task_inputs(p,image) for p in (p1,p2)]
    fone,ftwo = [float_inputs(p,image) for p in (p1,p2)]
    gray1,gray2 = torch.cat(one[1:5],1),torch.cat(two[1:5],1)
    float1,float2 = torch.cat(fone[1:5],1),torch.cat(ftwo[1:5],1)
    rawgray = float2-float1
    out = dict(polar=difference(p1,p2), gray=difference(gray1,gray2),
               aolp=difference(one[5],two[5],True), dolp=difference(one[6],two[6]),
               float_gray=difference(float1,float2),
               gray_update_zero_fraction=float(((gray2-gray1)==0).float().mean()),
               float_gray_nonzero_but_quantized_zero_fraction=float(((rawgray!=0)&((gray2-gray1)==0)).float().mean()),
               quantization_error_P1=difference(float1,gray1),
               quantization_error_P2=difference(float2,gray2),
               P1_clipped_fraction=float(((p1==0)|(p1==1)).float().mean()),
               P2_clipped_fraction=float(((p2==0)|(p2==1)).float().mean()))
    return out

@torch.no_grad()
def run(candidate=False):
    guard(); device=torch.device('cuda:0'); torch.set_num_threads(4)
    cfg=json.loads((SOURCE/'config/locked_run.json').read_text())
    cfg['paths']['output_root']=str(HERE/'outputs')
    model,task,criterion,rawcfg=study.components(cfg,device)
    del criterion; gc.collect(); torch.cuda.empty_cache()
    path=SOURCE/'downstream/checkpoints/final.pt'
    assert sha(path)=='3be8b40c1f06cc0b9828f6cfc23deaaba6db9a3be1a7a28b6bb4a5cca8fb8d76'
    ck=torch.load(path,map_location='cpu',weights_only=False)
    assert ck['step']==8000 and ck['protocol_id']==cfg['protocol_id']
    model.load_state_dict(ck['r2p_and_projections'],strict=True)
    task.net_g.load_state_dict(ck['tasknet_net_g'],strict=True)
    model.tau=ck['tau']; model.step=ck['step']; del ck
    model.eval().requires_grad_(False); task.eval().requires_grad_(False)
    from rgb_final_confirmation.evaluation import psnr,ssim
    before=dict(enhancer=state_hash(model),task=state_hash(task))
    traces=[]
    hook1=task.model.net_le_dm.register_forward_hook(lambda m,x,y:traces.append({'condition':y.detach().cpu()}))
    hook2=task.net_g.register_forward_pre_hook(lambda m,x:traces[-1].update(prior=x[1].detach().cpu()))
    rows=[]; tick=time.time()
    def prediction(p,image,cpu,cuda,use_float=False):
        guard()
        with torch.random.fork_rng(devices=[0]):
            torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda)
            if use_float:return task.forward_inputs(float_inputs(p,image))[0]
            return task(p.float(),image.float(),None)[0]
    for i,raw in enumerate(study.base.proxy_loader(cfg,rawcfg)):
        guard(); traces.clear(); batch=study.base.unpack_batch(raw,device)
        assert batch['sample_id'][0].startswith('validation/')
        study.base.set_step_seed(cfg['seed'],'feedback_signal_diagnostic',i,batch['sample_id'])
        p0,image=batch['polar'],batch['official_rgb']
        one=study.enhance(model,p0,p0,image);one.raw_features.clear();torch.cuda.empty_cache()
        cpu,cuda=torch.get_rng_state(),torch.cuda.get_rng_state_all()
        t1,f1=task(one.refined_polar.float(),image.float(),None)
        b=model.confidence_head(image,t1,study.polarization_summary(p0),f1)
        c=torch.exp(-b/model.tau).detach()
        f1.pop('T1',None)
        gate=F.interpolate(c,size=p0.shape[-2:],mode='bilinear',align_corners=False)
        two=study.enhance(model,p0,one.refined_polar,image,t1.clamp(0,1),f1,gate)
        two.raw_features.clear();torch.cuda.empty_cache()
        t2=prediction(two.refined_polar,image,cpu,cuda)
        metrics=lambda t:dict(psnr=psnr(t,batch['target']),ssim=ssim(t,batch['target']),
                              mae=float((t.clamp(0,1)-batch['target']).abs().mean()))
        row=dict(sample_id=batch['sample_id'][0],scene=batch['scene'][0],
                 round1=metrics(t1),round2=metrics(t2),confidence_mean=float(c.mean()),
                 interface=interface_stats(one.refined_polar,two.refined_polar,image),
                 condition=difference(traces[0]['condition'],traces[1]['condition']),
                 prior=difference(traces[0]['prior'],traces[1]['prior']),
                 task_output=difference(t1,t2),variants={})
        del b,c
        for name,custom in [('confidence_zero',torch.zeros_like(gate)),('confidence_one',torch.ones_like(gate)),('no_feedback',None)]:
            guard()
            alt=study.enhance(model,p0,one.refined_polar,image,
                              None if name=='no_feedback' else t1.clamp(0,1),
                              None if name=='no_feedback' else f1,custom)
            alt.raw_features.clear();torch.cuda.empty_cache()
            ta=prediction(alt.refined_polar,image,cpu,cuda)
            row['variants'][name]=dict(metrics=metrics(ta),polar_vs_standard=difference(alt.refined_polar,two.refined_polar),
                                      output_vs_standard=difference(ta,t2))
            del alt,ta
        if candidate:
            # Fixed-weight distribution/sensitivity check; not a trained new model.
            tf1=prediction(one.refined_polar,image,cpu,cuda,True)
            tf2=prediction(two.refined_polar,image,cpu,cuda,True)
            row['float_interface_fixed_weight']=dict(round1=metrics(tf1),round2=metrics(tf2),
                output1_vs_original=difference(tf1,t1),output2_vs_original=difference(tf2,t2))
            del tf1,tf2
        rows.append(row)
        json_save(HERE/'outputs/progress.json',dict(frames=len(rows),expected=28,last=row,seconds=time.time()-tick))
        print(json.dumps({'frame':len(rows),'id':row['sample_id'],'gray_change':row['interface']['gray']['changed_fraction'],
                          'delta_psnr':row['round2']['psnr']-row['round1']['psnr']}),flush=True)
        del one,two,t1,t2,f1,gate,batch,p0,image; traces.clear();gc.collect();torch.cuda.empty_cache()
    hook1.remove();hook2.remove()
    assert len(rows)==len({x['sample_id'] for x in rows})==28
    after=dict(enhancer=state_hash(model),task=state_hash(task));assert before==after
    result=dict(frames=28,original_size=True,tiling=False,training=False,test_used=False,
                checkpoint_sha256=sha(path),source_protocol_id=cfg['protocol_id'],paired_noise=True,
                float_candidate_evaluated=candidate,weights_before=before,weights_after=after,per_image=rows,
                seconds=time.time()-tick,deadline_unix=DEADLINE)
    json_save(HERE/'outputs/results.json',result)
    json_save(HERE/'status.json',dict(stage='completed',frames=28,time_unix=time.time()))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--float-candidate',action='store_true');a=ap.parse_args()
    try:run(a.float_candidate)
    except BaseException as exc:
        json_save(HERE/'status.json',dict(stage='failed',error=repr(exc),time_unix=time.time()));raise
