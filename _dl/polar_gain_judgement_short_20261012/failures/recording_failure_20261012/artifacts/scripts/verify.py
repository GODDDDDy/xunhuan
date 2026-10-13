"""Meaningful CPU checks for scientific validity before consuming GPU budget."""
import gc,json,sys
import torch
from common import ROOT,json_save,rng_state,restore_rng,seed,save
from acceptance import LocalAcceptance,local_error,labels,judgement_loss
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
if main.requires_grad:main.backward(retain_graph=True)
assert not any(p.grad is not None for p in model.confidence_head.parameters())
assert not any(p.grad is not None for p in model.update_gate.parameters())
assert not any(p.grad is not None for p in model.feedback.parameters()),'First500 train only new judgement'
for p in list(model.parameters())+list(task.parameters()):p.grad=None
quality.backward()
assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.update_gate.parameters())
assert not any(p.grad is not None for p in model.r2p.parameters())
assert not any(p.grad is not None for p in task.parameters())
assert not any(p.grad is not None for p in model.feedback.parameters())
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
out.refined_polar.retain_grad()
out.ungated_t2.square().mean().backward()
assert out.refined_polar.grad is not None and out.refined_polar.grad.abs().sum()>0
assert not out.t1.requires_grad, 'T1 is fixed throughout this continuation'
assert not any(p.grad is not None for p in model.confidence_head.parameters()),'Main task must not train H'
assert not any(p.grad is not None for p in model.update_gate.parameters()),'Task must not train judgement scores'
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
    for condition in ('confidence_zero','confidence_one','no_feedback','no_polar_correction','mask_zero','mask_one'):
        restore_rng(before);out=study.recurrent_forward(model,task,batch,diagnostic=condition)
        assert torch.isfinite(out.t2).all();del out
    restore_rng(before);out=study.recurrent_forward(model,task,batch,diagnostic='mask_zero')
    assert torch.equal(out.t1,out.t2);del out
    restore_rng(before);out=study.recurrent_forward(model,task,batch,diagnostic='mask_one')
    assert torch.equal(out.t2,out.ungated_t2);del out
    # Independent gradient checks: a fully rejected candidate still learns.
    model.train();model.step=1500
with torch.enable_grad():
    for p in list(model.parameters())+list(task.parameters()):p.grad=None
    with torch.no_grad():
        model.update_gate.body[-1].weight.zero_();model.update_gate.body[-1].bias.zero_()
    out,main,align,q=study.losses(model,task,criterion,batch)
    assert (out.update_mask==0).all(), 'Initial balanced gate must reject without candidate supervision'
    main.backward()
    assert model.feedback.output_head[-1].weight.grad.abs().sum()>0
    assert not any(p.grad is not None for p in model.update_gate.parameters())
    # Updating second-round-only feedback must not change the first output.
    expected_first=out.t1.detach().clone()
    optimizer.step()
    del out,main,align,q
    # Synthetic supervision has all three benefit labels with a fixed threshold.
    t=torch.zeros(1,3,8,12);target=torch.zeros_like(t);t[...,4:8]=.2;t[...,8:]=.2
    p=t.clone();p[...,:4]=.2;p[...,4:8]=0
    need,benefit,e1,ep=labels(t,p,target,.05,.01)
    assert set(benefit.flatten().tolist())=={0,1,2}
    # neutral zero proposal and explicit better/worse examples checked separately.
    _,neutral,_,_=labels(t,t,target,.05,.01);assert (neutral==1).all()
    assert (benefit==0).any() and (benefit==2).any()
model.eval()
with torch.no_grad():
    # Nonzero candidate and open gate prevent a vacuous initial-zero leak check.
    model.update_gate.body[-1].bias.copy_(torch.tensor([5.,5.,-5.,-5.]))
    before=rng_state();out=study.recurrent_forward(model,task,batch)
    expected=out.t2.clone();mask=out.update_mask.clone();assert mask.max()>0
    changed_batch=dict(batch,target=1-batch['target'])
    restore_rng(before);changed=study.recurrent_forward(model,task,changed_batch)
    assert torch.equal(expected,changed.t2) and torch.equal(mask,changed.update_mask)
    del out,changed
    before=rng_state();first_before=study.recurrent_forward(model,task,batch).t1.clone()
    for p in model.feedback.parameters():p.add_(.001)
    restore_rng(before);first_after=study.recurrent_forward(model,task,batch).t1
    assert torch.equal(first_before,first_after), 'Second-round interfaces changed the fixed first round'
assert not task.net_g.decoder_level2._forward_hooks and not task.net_g.decoder_level3._forward_hooks
json_save(ROOT/'diagnostics/cpu_checks.json',dict(passed=True,initial_closed_gate_preserves_t1=True,
          GT_replacement_does_not_change_prediction=True,warmup_enhancer_and_task_unchanged=True,
          quality_gradient_isolation=True,second_polar_task_gradient_nonzero=True,
          acceptance_gradient_isolation=True,mask_zero_exactly_preserves_t1=True,
          mask_one_equals_ungated_candidate=True,rejected_candidate_gradient_nonzero=True,
          local_benefit_labels_better_neutral_worse=True,
          first_round_fixed_when_second_interfaces_change=True,prior_parameters_frozen=True,
          checkpoint_rng_restore_exact=True,confidence_extremes_and_missing_feedback_finite=True,
          temporary_hooks_removed=True,official_initialization=task.initialization_audit,
          warmstart_exact=getattr(model,'initialization_audit',None),
          note='Disposable perturbations are discarded; CPU structural checks use L1; actual task loss is verified in GPU preflight, not efficacy evidence'))
print('All CPU scientific checks passed',flush=True)
