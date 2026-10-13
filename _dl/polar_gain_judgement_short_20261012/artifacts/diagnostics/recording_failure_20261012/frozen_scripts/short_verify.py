"""Meaningful CPU contracts for frozen T1, GT isolation and new conditioning."""
import copy,gc,torch
from common import ROOT,seed,save,json_save,rng_state,restore_rng
import staged,study

def verify(cfg):
    seed(cfg['seed']);model,task,raw=staged.components(cfg,torch.device('cpu'))
    source=torch.load(cfg['reuse']['checkpoint'],map_location='cpu',weights_only=False)
    assert all(torch.equal(v,source['r2p_and_projections'][k]) for k,v in model.state_dict().items() if not k.startswith('error_guidance.'))
    assert all(torch.equal(v,source['tasknet_net_g'][k]) for k,v in task.net_g.state_dict().items())
    small=copy.deepcopy(cfg);small['data'].update(crop_size=32,workers=0)
    rawbatch=next(iter(staged.loader(small,raw,'candidate',1)))
    batch=study.base.unpack_batch(next(study.microbatches(rawbatch)),torch.device('cpu'))
    staged.configure(model,task,'candidate');counts={'E':0,'D':0}
    hooks=[model.r2p.register_forward_hook(lambda *a:counts.__setitem__('E',counts['E']+1)),task.net_g.register_forward_hook(lambda *a:counts.__setitem__('D',counts['D']+1))]
    seed(100);out=staged.forward(model,task,batch);rounds=counts.copy();assert rounds=={'E':2,'D':2}
    first=out.t1.detach().clone();proposal=out.ungated_t2.detach().clone()
    seed(100);off=staged.forward(model,task,batch,'guidance_off')
    assert torch.equal(first,off.t1) and torch.equal(proposal,off.ungated_t2)
    assert torch.equal(out.t2,first)
    loss=staged.candidate_loss(out,batch['target']);loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.feedback.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.error_guidance.predict.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.error_guidance.inject.parameters())
    assert all(p.grad is None for p in task.parameters())
    assert all(p.grad is None for p in model.r2p.parameters())
    assert all(p.grad is None for p in model.action_gate.parameters())
    optimizer=staged.phase_optimizer(model,'candidate');optimizer.zero_grad(set_to_none=True)
    state=staged.checkpoint(model,task,cfg,'candidate',0,optimizer);save(ROOT/'diagnostics/cpu_resume.pt',state)
    def update():
        optimizer.zero_grad(set_to_none=True);seed(101)
        current=staged.forward(model,task,batch);staged.candidate_loss(current,batch['target']).backward()
        torch.nn.utils.clip_grad_norm_(staged.phase_module(model,'candidate').parameters(),1);optimizer.step()
        return {n:v.detach().clone() for n,v in model.state_dict().items()},torch.rand(7)
    restore_rng(state['rng']);expected,expected_rng=update()
    restored=staged.restore(model,task,ROOT/'diagnostics/cpu_resume.pt',cfg);optimizer.load_state_dict(restored['optimizer']);restore_rng(restored['rng'])
    actual,actual_rng=update()
    assert all(torch.equal(v,actual[n]) for n,v in expected.items()) and torch.equal(expected_rng,actual_rng)
    with torch.no_grad():
        seed(100);after=staged.forward(model,task,batch)
        changed=dict(batch,target=torch.rand_like(batch['target']))
        seed(100);replacement=staged.forward(model,task,changed)
        assert torch.equal(after.t1,replacement.t1) and torch.equal(after.ungated_t2,replacement.ungated_t2)
        assert torch.equal(after.repair_need,replacement.repair_need)
        assert torch.equal(first,after.t1)
        seed(100);off=staged.forward(model,task,batch,'guidance_off')
        assert not torch.equal(after.ungated_t2,off.ungated_t2),'New conditioning has no influence after a real update'
    for p in model.parameters():p.grad=None
    staged.configure(model,task,'judgement')
    with torch.no_grad():seed(100);fixed=study.recurrent_forward(model,task,batch)
    gated=staged.gate(model,fixed,batch)
    staged.cost_loss(gated.action['raw_cost'],staged.action_costs(gated.t1,gated.rgb_increment,batch['target']),model.cost_scale).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.action_gate.parameters())
    assert all(p.grad is None for p in model.feedback.parameters())
    assert all(p.grad is None for p in model.error_guidance.parameters())
    for h in hooks:h.remove()
    json_save(ROOT/'diagnostics/short_cpu_checks.json',dict(passed=True,source_sha256=cfg['reuse']['sha256'],inherited_tensors_strict=len(source['r2p_and_projections']),actual_rounds=rounds,zero_injection_source_candidate_exact=True,t1_frozen_after_real_update=True,gt_replacement_no_prediction_change=True,q_loss_and_injection_gradients_present=True,candidate_loss_cannot_manipulate_q=True,judgement_gradients_isolated=True,optimizer_rng_resume_exact=True,new_guidance_influences_candidate=True))
    (ROOT/'diagnostics/cpu_resume.pt').unlink();del model,task,optimizer;gc.collect()
