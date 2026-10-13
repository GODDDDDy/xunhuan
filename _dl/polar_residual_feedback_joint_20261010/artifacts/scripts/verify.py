"""Meaningful CPU checks for scientific validity before consuming GPU budget."""
import gc,json,sys
import torch
from common import ROOT,json_save,rng_state,restore_rng,seed,save
import study

torch.set_num_threads(4)
cfg=json.loads((ROOT/'config/preflight.json').read_text())
cfg['data']['workers']=0
device=torch.device('cpu');seed(cfg['seed'])
model,task,criterion,rawcfg=study.components(cfg,device,with_loss=False)
criterion=torch.nn.L1Loss()
model.activation_checkpoint=False;model.r2p.activation_checkpoint=False
raw=next(iter(study.base.scheduled_loader(cfg,rawcfg,1)))
batch=study.base.unpack_batch(next(study.microbatches(raw)),device)
for key,value in batch.items():
    if torch.is_tensor(value) and value.ndim==4:batch[key]=value[...,:32,:32]
model.eval();before=rng_state()
with torch.no_grad():
    out=study.recurrent_forward(model,task,batch)
    assert torch.equal(out.t1,out.t2)
    expected=out.t2.clone();del out
    altered=dict(batch,target=torch.rand_like(batch['target']))
    restore_rng(before);other=study.recurrent_forward(model,task,altered)
    assert torch.equal(other.t2,expected),'GT leaked into inference'
    del other
print('CPU zero-head and no-GT-leak checks passed',flush=True)
model.train();model.step=1;study.configure_step(task,1,model)
optimizer=study.optimizer_for(model,task,cfg)
# optimizer_for must be built before freeze in formal training; this small check
# only tests that frozen parameters do not update and the correction head can.
frozen={n:p.detach().clone() for n,p in model.r2p.named_parameters()}
official={n:p.detach().clone() for n,p in task.net_g.named_parameters()}
out,main,alignment,quality=study.losses(model,task,criterion,batch)
main.backward(retain_graph=True)
assert not any(p.grad is not None for p in model.confidence_head.parameters())
assert model.feedback.output_head[-1].weight.grad.abs().sum()>0
for p in list(model.parameters())+list(task.parameters()):p.grad=None
quality.backward()
assert not any(p.grad is not None for p in model.r2p.parameters())
assert not any(p.grad is not None for p in task.parameters())
optimizer.zero_grad(set_to_none=True)
del out,main,alignment,quality
out,main,alignment,quality=study.losses(model,task,criterion,batch)
(main+quality).backward();optimizer.step()
assert all(torch.equal(frozen[n],p) for n,p in model.r2p.named_parameters())
assert all(torch.equal(official[n],p) for n,p in task.net_g.named_parameters())
del out,frozen,official;gc.collect()
print('CPU warmup freeze and separated quality gradients passed',flush=True)
model.step=501;study.configure_step(task,501,model)
for p in list(model.parameters())+list(task.parameters()):p.grad=None
# Disposable perturbations open initial zero paths, so a nonzero gradient test
# does not mistake their mathematically expected zero gradients for a bug.
with torch.no_grad():
    for collection in (model.feedback.decoder_inject,model.feedback.polar_inject):
        for module in collection.values():module.body[-1].weight.normal_(0,.002)
    model.feedback.output_head[-1].weight.normal_(0,.002)
model.train()
out=study.recurrent_forward(model,task,batch)
out.refined_polar.retain_grad();out.first_task_features['T2'].retain_grad()
out.t2.square().mean().backward()
assert out.refined_polar.grad is not None and out.refined_polar.grad.abs().sum()>0
assert out.first_task_features['T2'].grad is not None and out.first_task_features['T2'].grad.abs().sum()>0
assert not any(p.grad is not None for p in model.confidence_head.parameters()),'Main task must not train H'
allowed={id(p) for p in task.net_g.parameters()}
assert not any(p.grad is not None and id(p) not in allowed for p in task.parameters())
del out;gc.collect();model.eval()
with torch.no_grad():
    before=rng_state();out=study.recurrent_forward(model,task,batch);expected=out.t2.clone();del out
    path=ROOT/'diagnostics/cpu_resume_check.pt'
    save(path,dict(model=model.state_dict(),task=task.net_g.state_dict(),rng=before))
    ck=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(ck['model'],strict=True);task.net_g.load_state_dict(ck['task'],strict=True);restore_rng(ck['rng'])
    out=study.recurrent_forward(model,task,batch);assert torch.equal(out.t2,expected);del out
    path.unlink()
    for condition in ('confidence_zero','confidence_one','no_feedback','no_polar_correction'):
        restore_rng(before);out=study.recurrent_forward(model,task,batch,diagnostic=condition)
        assert torch.isfinite(out.t2).all();del out
assert not task.net_g.decoder_level2._forward_hooks and not task.net_g.decoder_level3._forward_hooks
json_save(ROOT/'diagnostics/cpu_checks.json',dict(passed=True,zero_head_preserves_t1=True,
          GT_replacement_does_not_change_prediction=True,warmup_enhancer_and_task_unchanged=True,
          quality_gradient_isolation=True,second_polar_task_gradient_nonzero=True,
          first_task_feature_gradient_nonzero=True,prior_parameters_frozen=True,
          checkpoint_rng_restore_exact=True,confidence_extremes_and_missing_feedback_finite=True,
          temporary_hooks_removed=True,official_initialization=task.initialization_audit,
          note='Disposable perturbations are discarded; CPU structural checks use L1; actual task loss is verified in GPU preflight, not efficacy evidence'))
print('All CPU scientific checks passed',flush=True)
