"""Fit only R2P to a frozen trained G6, starting at an exact identity front-end."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import torch
import yaml
import run_matched

def hashes(state):
    return {n:hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for n,t in state.items()}

def frozen_state(model,task):
    values={'task.'+n:t for n,t in task.state_dict().items()}
    values.update({'projection.'+n:t for n,t in model.projections.state_dict().items()})
    values.update({'feedback.'+n:t for n,t in model.feedback_adapter.state_dict().items()})
    values['feedback.scale']=model.feedback_scale
    return hashes(values)

def configure(cfg):
    assert cfg['study']['arm']=='K1' and cfg['study']['fixed_lambda']==0.0
    assert cfg['study']['experiment']=='frozen_G6_identity_adapter' and cfg['study']['phase_weight']==0.1
    base=run_matched.configure(cfg)
    parent_protocol=base.protocol_id
    def protocol(config):
        obj=copy.deepcopy(config)
        obj['frozen_adapter_extension_sha256']=base.sha256_file(Path(__file__))
        obj['frozen_G6_checkpoint_sha256']=config['study']['frozen_G6_checkpoint_sha256']
        return parent_protocol(obj)
    base.protocol_id=protocol
    parent_build,parent_checkpoint=base.build_components,base.checkpoint_payload

    def build(config,device):
        model,task,criterion,raw=parent_build(config,device)
        source=Path(config['study']['frozen_G6_checkpoint'])
        # Checkpoint file is bound before training; load only into the existing net_g.
        state=torch.load(source,map_location='cpu',weights_only=False)
        assert state['step']==3000 and state['phase_weight']==0.1
        task.net_g.load_state_dict(state['tasknet_net_g'],strict=True)
        assert all(torch.equal(t.detach().cpu(),state['tasknet_net_g'][n]) for n,t in task.net_g.state_dict().items())
        task.requires_grad_(False)
        task.eval()
        model.r2p.requires_grad_(True)
        model.projections.requires_grad_(False)
        model.feedback_adapter.requires_grad_(False)
        model.feedback_scale.requires_grad_(False)
        pretrained=hashes(model.r2p.state_dict())
        head=model.r2p.body[-1]
        assert isinstance(head,torch.nn.Conv2d) and head.out_channels==12
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
        actual=hashes(model.r2p.state_dict())
        reset_keys=[n for n in actual if actual[n]!=pretrained[n]]
        assert set(reset_keys)=={'body.4.weight','body.4.bias'}
        model._full_frozen_start=frozen_state(model,task)
        model._frozen_G6_checkpoint_sha256=config['study']['frozen_G6_checkpoint_sha256']
        root=Path(config['paths']['output_root'])
        initial={'seed':config['seed'],'experiment':config['study']['experiment'],
          'frozen_G6_checkpoint_sha256':model._frozen_G6_checkpoint_sha256,
          'netg':hashes(task.net_g.state_dict()),'model':hashes(model.state_dict()),
          'full_frozen_state':model._full_frozen_start,'pretrained_r2p_state':pretrained,
          'reset_r2p_output_keys':reset_keys,'source_schedule_sha256':base.sha256_file(Path(config['paths']['schedule'])),
          'trainable_r2p_parameter_tensors':sum(p.requires_grad for p in model.r2p.parameters()),
          'trainable_model_parameter_indices':[i for i,p in enumerate(model.parameters()) if p.requires_grad],
          'all_tasknet_parameters_frozen':all(not p.requires_grad for p in task.parameters())}
        assert initial['trainable_r2p_parameter_tensors']==18 and initial['all_tasknet_parameters_frozen']
        base.atomic_json(root/'diagnostics/initial_pair_audit.json',initial)
        return model,task,criterion,raw

    def losses(model,task,criterion,batch):
        out=base.recurrent_forward(model,task,batch,need_alignment=False)
        out.loss1=out.loss2=criterion(out.t1,batch['target'])
        assert torch.isfinite(out.loss1)
        return out,out.t1,out.loss1,out.loss1.new_zeros(())

    def checkpoint(model,task,optimizer,step,coefficient,metrics=None,*extra):
        current=frozen_state(model,task)
        changed=[n for n,h in current.items() if h!=model._full_frozen_start[n]]
        assert not changed,changed
        assert all(not p.requires_grad and p.grad is None for p in task.parameters())
        audit={'step':int(step),'checked_tensors':len(current),'changed_keys':changed,'passed':not changed,
               'all_tasknet_parameters_frozen':True,'G6_checkpoint_sha256':model._frozen_G6_checkpoint_sha256}
        base.atomic_json(model._study_output_root/'diagnostics'/f'full_frozen_step_{step}.json',audit)
        value=parent_checkpoint(model,task,optimizer,step,coefficient,metrics,*extra)
        value.update(experiment='frozen_G6_identity_adapter',frozen_G6_checkpoint_sha256=model._frozen_G6_checkpoint_sha256,
                     optimizer_updates_only_r2p=True,full_frozen_audit_passed=True,identity_output_initialization=True)
        return value
    base.build_components=build
    base.forward_losses=losses
    base.checkpoint_payload=checkpoint
    base.run_preflight=lambda config,device:preflight(base,config,device)
    return base

def preflight(base,cfg,device):
    root=Path(cfg['paths']['output_root'])
    assert not any((root/'checkpoints').glob('*.pt'))
    base.calibrate(cfg,device)
    base.set_seed(cfg['seed'])
    model,task,criterion,raw=base.build_components(cfg,device)
    batch=base.unpack_batch(next(iter(base.scheduled_loader(cfg,raw,1))),device)
    torch.cuda.reset_peak_memory_stats(device)
    calls={'r2p':0,'tasknet':0}
    def hook(key):
        def count(*args):calls[key]+=1
        return count
    handles=[model.register_forward_hook(hook('r2p')),task.register_forward_hook(hook('tasknet'))]
    base.set_step_seed(cfg['seed'],'preflight',1,batch['sample_id'])
    out,_,loss,al=base.forward_losses(model,task,criterion,batch)
    for h in handles:h.remove()
    out.refined_polar.retain_grad()
    identity_input=torch.equal(out.refined_polar.detach(),batch['polar'])
    loss.backward()
    payload={'protocol_id':base.protocol_id(cfg),'actual_forward_calls':calls,
             'initial_polar_identity':identity_input,'initial_delta_max':float(out.delta.detach().abs().max()),
             'input_gradient_norm':float(out.refined_polar.grad.norm()),
             'grad_norm_r2p':base.gradient_norm(model.r2p.parameters()),
             'grad_norm_netg':base.gradient_norm(task.net_g.parameters()),
             'no_frozen_task_gradients':all(not p.requires_grad and p.grad is None for p in task.parameters()),
             'r2p_parameters_with_nonzero_gradient':[n for n,p in model.r2p.named_parameters() if p.grad is not None and bool(p.grad.abs().max()>0)],
             'other_model_gradients_absent':all(p.grad is None for p in model.projections.parameters()) and all(p.grad is None for p in model.feedback_adapter.parameters()) and model.feedback_scale.grad is None,
             'all_gradients_finite':all(bool(torch.isfinite(p.grad).all()) for p in model.r2p.parameters() if p.grad is not None),
             'peak_vram_mb':torch.cuda.max_memory_allocated(device)/2**20,
             'lambda_feature_alignment':0.0,'loss_finite':bool(torch.isfinite(loss))}
    del out,loss,al,batch
    model.zero_grad(set_to_none=True)
    # Same-input/stream tensor parity plus historical full-validation metric replay on all proxy images.
    from rgb_final_confirmation.evaluation import psnr,ssim
    old=json.loads(Path(cfg['study']['frozen_G6_metrics']).read_text())
    lookup={r['sample_id']:(i,r) for i,r in enumerate(old['per_image'])}
    parity=[]
    with torch.no_grad():
        model.eval();task.eval()
        for raw_batch in base.proxy_loader(cfg,raw):
            b=base.unpack_batch(raw_batch,device)
            index,reference=lookup[b['sample_id'][0]]
            base.set_step_seed(2026,'validation',index,b['sample_id'])
            cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state_all()
            direct,direct_features=task(b['polar'],b['official_rgb'],None)
            direct_features.clear()
            torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(gpu)
            out=base.recurrent_forward(model,task,b)
            input_diff=float((out.refined_polar-b['polar']).abs().max())
            output_diff=float((direct-out.t1).abs().max())
            psnr_diff=abs(psnr(direct,b['target'])-reference['psnr'])
            ssim_diff=abs(ssim(direct,b['target'])-reference['ssim'])
            row={'sample_id':b['sample_id'][0],'scene':b['scene'][0],
                 'input_max_diff':input_diff,'output_max_diff':output_diff,
                 'historical_G6_psnr_abs_diff':psnr_diff,'historical_G6_ssim_abs_diff':ssim_diff}
            row['passed']=input_diff==0 and output_diff<=1e-4 and psnr_diff<=1e-5 and ssim_diff<=1e-7
            parity.append(row)
            del direct,out,b
    assert len(parity)==int(cfg['study']['expected_proxy_images'])
    payload['proxy_step0_parity']={'n':len(parity),'all_passed':all(r['passed'] for r in parity),
        'output_tolerance':1e-4,'historical_psnr_tolerance':1e-5,'historical_ssim_tolerance':1e-7,'per_image':parity}
    payload['passed']=(calls=={'r2p':1,'tasknet':1} and identity_input and payload['initial_delta_max']==0
       and payload['input_gradient_norm']>0 and payload['grad_norm_r2p']>0 and payload['grad_norm_netg']==0
       and payload['no_frozen_task_gradients'] and payload['other_model_gradients_absent']
       and payload['all_gradients_finite'] and payload['loss_finite'] and payload['proxy_step0_parity']['all_passed'])
    base.atomic_json(root/'diagnostics/recurrent_preflight.json',payload)
    assert payload['passed'],payload
    return payload

if __name__=='__main__':
    cfg=yaml.safe_load(Path(sys.argv[sys.argv.index('--config')+1]).read_text())
    configure(cfg).main()
