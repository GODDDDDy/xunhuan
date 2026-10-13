"""Reuse frozen-adapter trainer; only the R2P condition representation differs."""
import copy,hashlib,json,sys
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT.parent/'frozen_g6_adapter_20261007/scripts'))
import run_frozen_adapter
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def configure(cfg):
    assert cfg['study']['condition_probe'] in ('mixed_rgb','task_prediction')
    assert cfg['optimization']['total_steps']==1000 and cfg['seed']==2026
    lock=json.loads((ROOT/'training_lock.json').read_text())
    assert all(sha(p)==h for p,h in lock['locked_file_sha256'].items())
    assert json.loads((ROOT/'diagnostics/diagnostic_result.json').read_text())['training_gate_passed']
    base=run_frozen_adapter.configure(cfg)
    parent_protocol=base.protocol_id;parent_checkpoint=base.checkpoint_payload
    def protocol(config):
        obj=copy.deepcopy(config);obj['condition_probe_extension_sha256']=sha(__file__)
        return parent_protocol(obj)
    base.protocol_id=protocol
    def forward(model,task,batch,need_alignment=False):
        assert not need_alignment
        cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state_all()
        with torch.no_grad():
            baseline,features=task(batch['polar'],batch['official_rgb'],None)
            features.clear()
        condition=(baseline.detach().clamp(0,1) if cfg['study']['condition_probe']=='task_prediction'
                   else batch['condition'].detach())
        first=model(batch['polar'],condition,project_features=False)
        if not torch.is_grad_enabled():
            first.raw_features.clear();first.gate=first.gate.new_empty(0)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(gpu)
            if torch.is_grad_enabled() and model.activation_checkpoint:
                corrected,features=base.checkpoint(task,first.refined_polar,batch['official_rgb'],None,
                    use_reentrant=False,preserve_rng_state=True)
            else:corrected,features=task(first.refined_polar,batch['official_rgb'],None)
        features.clear()
        out=copy.copy(first);out.first=first;out.t1=baseline;out.t2=corrected
        out.first_task_features={};out.condition=condition
        return out
    def losses(model,task,criterion,batch):
        out=forward(model,task,batch)
        out.loss1=criterion(out.t1,batch['target'])
        out.loss2=criterion(out.t2,batch['target'])
        assert torch.isfinite(out.loss2)
        return out,out.t2,out.loss2,out.loss2.new_zeros(())
    def checkpoint(model,task,optimizer,step,coefficient,metrics=None,*extra):
        value=parent_checkpoint(model,task,optimizer,step,coefficient,metrics,*extra)
        value.update(condition_probe=cfg['study']['condition_probe'],actual_task_calls=2,
            task_condition_detached=True,task_prediction_clamped=True,
            baseline_task_not_trained=True,effective_task_weights=[0.0,1.0])
        return value
    def validate(model,task,loader,device,seed):
        from rgb_final_confirmation.evaluation import psnr,ssim,aggregate_metrics
        old=json.loads(Path(cfg['study']['frozen_G6_metrics']).read_text())
        lookup={r['sample_id']:(i,r) for i,r in enumerate(old['per_image'])}
        rows=[];baseline_rows=[];audit_rows=[];polar=[];clip=[]
        model.eval();task.eval()
        frozen_before=run_frozen_adapter.frozen_state(model,task)
        r2p_before=run_frozen_adapter.hashes(model.r2p.state_dict())
        calls={'r2p':0,'tasknet':0}
        def count(k):
            def h(*a):calls[k]+=1
            return h
        handles=[model.register_forward_hook(count('r2p')),task.register_forward_hook(count('tasknet'))]
        with torch.no_grad():
            for raw in loader:
                b=base.unpack_batch(raw,device);sid=b['sample_id'][0];i,reference=lookup[sid]
                base.set_step_seed(2026,'validation',i,b['sample_id'])
                out=forward(model,task,b)
                row=dict(sample_id=sid,scene=b['scene'][0],psnr=psnr(out.t2,b['target']),ssim=ssim(out.t2,b['target']))
                baseline_row=dict(sample_id=sid,scene=b['scene'][0],psnr=psnr(out.t1,b['target']),ssim=ssim(out.t1,b['target']))
                differences={m:abs(baseline_row[m]-reference[m]) for m in ('psnr','ssim')}
                assert differences['psnr']<=1e-5 and differences['ssim']<=1e-7
                rows.append(row);baseline_rows.append(baseline_row);audit_rows.append(dict(sample_id=sid,**differences))
                polar.append(base.polar_stats(b['polar'],out.refined_polar))
                clip.append(float(((out.t1<0)|(out.t1>1)).float().mean()))
                if len(rows)%50==0:print(f'Evaluated {len(rows)} images',flush=True)
                del b,out
        for h in handles:h.remove()
        assert calls=={'r2p':len(rows),'tasknet':2*len(rows)}
        assert frozen_before==run_frozen_adapter.frozen_state(model,task)
        assert r2p_before==run_frozen_adapter.hashes(model.r2p.state_dict())
        summary=aggregate_metrics(rows);baseline_summary=aggregate_metrics(baseline_rows)
        summary['scene_summary']=summary.pop('scenes');summary['scenes']=len(summary['scene_summary'])
        summary['frames']=len(rows)
        summary.update(per_image=rows,baseline_per_image=baseline_rows,baseline=baseline_summary,
            baseline_replay_max_diff={m:max(r[m] for r in audit_rows) for m in ('psnr','ssim')},
            training_seed=2026,evaluation_seed=2026,condition_probe=cfg['study']['condition_probe'],
            polar_update=polar,condition_prediction_clip_fraction_mean=sum(clip)/len(clip),
            actual_tasknet_forwards=calls['tasknet'],actual_r2p_forwards=calls['r2p'],full_task_frozen=True,
            inference_r2p_hashes_unchanged=True)
        return summary
    base.recurrent_forward=forward;base.forward_losses=losses;base.validate=validate;base.checkpoint_payload=checkpoint
    base.run_preflight=lambda config,device:preflight(base,config,device)
    return base
def preflight(base,cfg,device):
    run=Path(cfg['paths']['output_root']);assert not list((run/'checkpoints').glob('*.pt'))
    base.calibrate(cfg,device);base.set_seed(cfg['seed'])
    model,task,criterion,raw=base.build_components(cfg,device)
    b=base.unpack_batch(next(iter(base.scheduled_loader(cfg,raw,1))),device)
    before=run_frozen_adapter.frozen_state(model,task)
    base.set_step_seed(2026,'preflight',1,b['sample_id']);torch.cuda.reset_peak_memory_stats(device)
    calls={'r2p':0,'tasknet':0}
    def count(k):
        def h(*a):calls[k]+=1
        return h
    handles=[model.register_forward_hook(count('r2p')),task.register_forward_hook(count('tasknet'))]
    out,_,loss,_=base.forward_losses(model,task,criterion,b)
    for h in handles:h.remove()
    identity=torch.equal(out.refined_polar,b['polar'])
    output_diff=float((out.t2-out.t1).abs().max())
    loss.backward()
    gradients={n:float(p.grad.norm()) if p.grad is not None else 0.0 for n,p in model.r2p.named_parameters()}
    trainable=[n for n,p in model.r2p.named_parameters() if p.requires_grad]
    payload=dict(protocol_id=base.protocol_id(cfg),actual_forward_calls=calls,identity_polar=identity,
        zero_output_max_diff=output_diff,condition_requires_grad=out.condition.requires_grad,
        trainable_r2p_tensors=len(trainable),grad_norm_r2p=base.gradient_norm(model.r2p.parameters()),
        grad_norm_netg=base.gradient_norm(task.net_g.parameters()),gradients=gradients,
        all_gradients_finite=all(bool(torch.isfinite(p.grad).all()) for p in model.r2p.parameters() if p.grad is not None),
        frozen_hashes_unchanged=(before==run_frozen_adapter.frozen_state(model,task)),
        no_task_gradients=all(not p.requires_grad and p.grad is None for p in task.parameters()),
        peak_vram_mb=torch.cuda.max_memory_allocated(device)/2**20)
    assert calls=={'r2p':1,'tasknet':2} and identity and output_diff<=1e-4
    assert not payload['condition_requires_grad'] and len(trainable)==18
    assert payload['grad_norm_r2p']>0 and payload['grad_norm_netg']==0
    assert payload['all_gradients_finite'] and payload['frozen_hashes_unchanged'] and payload['no_task_gradients']
    del out,loss,b
    model.zero_grad(set_to_none=True)
    proxy=base.validate(model,task,base.proxy_loader(cfg,raw),device,2026)
    assert proxy['frames']==28
    assert all(abs(a[m]-b[m])<=(1e-5 if m=='psnr' else 1e-7)
        for a,b in zip(proxy['per_image'],proxy['baseline_per_image']) for m in ('psnr','ssim'))
    payload.update(passed=True,proxy_step0_parity=dict(n=28,all_passed=True,baseline_replay_max_diff=proxy['baseline_replay_max_diff']))
    base.atomic_json(run/'diagnostics/recurrent_preflight.json',payload)
    base.atomic_json(run/'diagnostics/step0_proxy.json',proxy)
    return payload
if __name__=='__main__':
    cfg=yaml.safe_load(Path(sys.argv[sys.argv.index('--config')+1]).read_text())
    base=configure(cfg);base.main()
