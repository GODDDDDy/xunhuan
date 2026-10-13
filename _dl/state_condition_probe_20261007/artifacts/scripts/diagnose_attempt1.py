"""Fixed-current-checkpoint alpha sweep with real preprocessing/prior tracing."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch,yaml
ROOT=Path(__file__).resolve().parent.parent
read=lambda p:json.loads(Path(p).read_text())
lock=read(ROOT/'protocol_lock.json');u=lock['diagnostic_source']
sys.path.insert(0,u['source_scripts'])
import run_matched
cfg=yaml.safe_load((ROOT/'configs/diagnostic.yaml').read_text())
base=run_matched.configure(cfg);base.add_source_path(cfg);base.set_seed(2026)
assert all(base.sha256_file(Path(p))==h for p,h in lock['locked_file_sha256'].items())
model,task,criterion,raw=base.build_components(cfg,torch.device('cuda:0'))
cp=torch.load(u['checkpoint'],map_location='cpu',weights_only=False)
assert cp['protocol_id']==base.protocol_id(cfg) and cp['phase_weight']==0.1
model.load_state_dict(cp['r2p_and_projections'],strict=True)
task.net_g.load_state_dict(cp['tasknet_net_g'],strict=True)
del cp,criterion
model.requires_grad_(False).eval();task.requires_grad_(False).eval()
from rgb_final_confirmation.evaluation import psnr,ssim
from rgb_final_confirmation.polarfree import official_polarfree_task_inputs
def hashes():
    return {prefix+n:hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for prefix,obj in [('model.',model),('task.',task)] for n,t in obj.state_dict().items()}
before=hashes();calls={'r2p':0,'tasknet':0}
def count(k):
    def hook(*a):calls[k]+=1
    return hook
handles=[model.register_forward_hook(count('r2p')),task.register_forward_hook(count('tasknet'))]
trace={}
def condition_hook(m,a,o):trace['condition']=o.detach().clone()
def prior_hook(m,a):trace['prior']=a[1].detach().clone()
handles += [task.model.net_le_dm.register_forward_hook(condition_hook),task.net_g.register_forward_pre_hook(prior_hook)]
def snapshot(p,rgb):
    inputs=official_polarfree_task_inputs(p,rgb)
    return dict(gray=torch.cat(inputs[1:5],1).detach().clone(),aolp=inputs[5].detach().clone(),
        dolp=inputs[6].detach().clone(),condition=trace['condition'].clone(),prior=trace['prior'].clone())
def difference(left,right):
    values={}
    for k in left:
        d=(right[k]-left[k]).abs()
        if k=='aolp':d=0.5*torch.atan2(torch.sin(2*(right[k]-left[k])),torch.cos(2*(right[k]-left[k]))).abs()
        values[k]=dict(mean_abs=float(d.mean()),max_abs=float(d.max()),changed_fraction=float((d>0).float().mean()))
    return values
def delta_stats(source,result):
    p=result.refined_polar;raw_sum=source+result.delta;effective=p-source
    return dict(raw_delta_mean_abs=float(result.delta.abs().mean()),effective_mean_abs=float(effective.abs().mean()),
        effective_max_abs=float(effective.abs().max()),clamp_gap_max=float((p-raw_sum).abs().max()),
        clamp_fraction=float(((raw_sum<0)|(raw_sum>1)).float().mean()),
        clamped_identity_max_diff=float((p-raw_sum.clamp(0,1)).abs().max()))
history=read(u['validation_metrics'])
lookup={r['sample_id']:(i,r) for i,r in enumerate(history['per_image'])}
picked=set(lock['proxy']['sample_ids']);rows=[];start=time.time()
assert not (ROOT/'diagnostics/diagnostic_result.json').exists()
with torch.no_grad():
    for raw_batch in base.proxy_loader(cfg,raw):
        b=base.unpack_batch(raw_batch,torch.device('cuda:0'));sid=b['sample_id'][0]
        assert sid in picked
        index,reference=lookup[sid]
        base.set_step_seed(2026,'validation',index,b['sample_id'])
        first=model(b['polar'],b['condition'],project_features=False);first.raw_features.clear()
        rng_cpu=torch.get_rng_state();rng_cuda=torch.cuda.get_rng_state_all()
        def restore():
            torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state_all(rng_cuda)
        def infer(p):
            restore();prediction,features=task(p,b['official_rgb'],None);features.clear()
            return prediction,snapshot(p,b['official_rgb'])
        t1,stage1=infer(first.refined_polar)
        second=model(first.refined_polar,b['condition'],project_features=False);second.raw_features.clear()
        t2,stage2=infer(second.refined_polar)
        metrics1=dict(psnr=psnr(t1,b['target']),ssim=ssim(t1,b['target']))
        metrics2=dict(psnr=psnr(t2,b['target']),ssim=ssim(t2,b['target']))
        historical_diff={f'{round}_{m}':abs(metrics[m]-reference[round][m])
            for round,metrics in [('round1',metrics1),('round2',metrics2)] for m in ('psnr','ssim')}
        assert all(v<=(1e-5 if k.endswith('psnr') else 1e-7) for k,v in historical_diff.items())
        row=dict(sample_id=sid,scene=b['scene'][0],difficulty=sid.split('/')[1],
            full_validation_index=index,round1=metrics1,round2=metrics2,
            historical_abs_diff=historical_diff,round1_delta=delta_stats(b['polar'],first),
            round2_delta=delta_stats(first.refined_polar,second),second_round_path=difference(stage1,stage2),alphas={})
        for alpha in lock['alphas']:
            p=first.refined_polar.clone() if alpha==0 else (second.refined_polar.clone() if alpha==1
                else first.refined_polar+alpha*(second.refined_polar-first.refined_polar))
            prediction,stages=infer(p)
            assert bool(torch.isfinite(prediction).all())
            metric=dict(psnr=psnr(prediction,b['target']),ssim=ssim(prediction,b['target']))
            item=dict(**metric,delta_vs_T1={m:metric[m]-metrics1[m] for m in metric},path_vs_T1=difference(stage1,stages))
            if alpha in (0,1):
                reference_p=first.refined_polar if alpha==0 else second.refined_polar
                reference_t=t1 if alpha==0 else t2
                item['endpoint']=dict(input_max_diff=float((p-reference_p).abs().max()),output_max_diff=float((prediction-reference_t).abs().max()))
                item['endpoint']['passed']=item['endpoint']['input_max_diff']==0 and item['endpoint']['output_max_diff']<=1e-4
                assert item['endpoint']['passed']
            row['alphas'][str(alpha)]=item
            del prediction,p,stages
        rows.append(row)
        base.atomic_json(ROOT/'status.json',dict(state='running',stage='diagnosis',completed_images=len(rows),total_images=28,sample_id=sid))
        print(f'Diagnostic {len(rows)}/28 {sid}',flush=True)
        del first,second,t1,t2,stage1,stage2,b
after=hashes()
assert before==after and len(rows)==28 and {r['sample_id'] for r in rows}==picked
assert calls=={'r2p':56,'tasknet':196}
assert all(p.grad is None for p in list(model.parameters())+list(task.parameters()))
scenes=sorted({r['scene'] for r in rows})
def aggregates(a):
    return {f'image_mean_{m}':float(np.mean([r['alphas'][str(a)]['delta_vs_T1'][m] for r in rows])) for m in ('psnr','ssim')}|{
        f'scene_macro_{m}':float(np.mean([np.mean([r['alphas'][str(a)]['delta_vs_T1'][m] for r in rows if r['scene']==s]) for s in scenes])) for m in ('psnr','ssim')}
stage_keys=['gray','aolp','dolp','condition','prior']
reached={k:sum(r['second_round_path'][k]['max_abs']>0 for r in rows) for k in stage_keys}
gate=all(v>0 for v in reached.values())
result=dict(scope='28 prelocked validation images, diagnostic only; current normalized phase0.1 K2 seed2026',
    checkpoint_sha256=u['checkpoint_sha256'],per_image=rows,delta_summary={str(a):aggregates(a) for a in lock['alphas']},
    stage_changed_images=reached,training_gate_passed=gate,actual_forward_calls=calls,
    frozen_tensor_hashes_before=before,frozen_tensor_hashes_after=after,all_frozen=True,
    independent_endpoint_checks=56,all_endpoints_passed=True,elapsed_seconds=time.time()-start,
    difficulty_counts={d:sum(r['difficulty']==d for r in rows) for d in ('easy','hard')})
base.atomic_json(ROOT/'diagnostics/diagnostic_result.json',result)
base.atomic_json(ROOT/'status.json',dict(state='complete',stage='diagnosis',completed_images=28,training_gate_passed=gate))
print(json.dumps({k:v for k,v in result.items() if k not in ('per_image','frozen_tensor_hashes_before','frozen_tensor_hashes_after')},indent=2))
