"""Focused CPU checks for the authorized schedule, plateau and protected updates."""
import json,sys,tempfile,math
from pathlib import Path
import torch
from torch import nn
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'scripts'))
import pretrain,study
from enhancer import PolarEnhancer,JointEnhancer
cfg=json.loads((root/'config/locked_run.json').read_text())
assert cfg['task_initialization']['kind']=='official_polarfree'
assert cfg['source_sha256']=='a230e77fb19827a1ada41e4dc86374dee092e476915d2252841f81dd8c257ca2'
assert cfg['budget']['deadline_unix']==1791605047.4916465
assert math.isclose(pretrain.learning_rate(cfg,8000),cfg['pretrain']['continuation']['start_lr'],rel_tol=1e-12)
assert abs(pretrain.learning_rate(cfg,8001)-pretrain.learning_rate(cfg,8000))<1e-10
assert pretrain.learning_rate(cfg,20000)==1e-6
assert all(pretrain.learning_rate(cfg,k+1)<=pretrain.learning_rate(cfg,k) for k in range(8000,20000))
with tempfile.TemporaryDirectory(dir=root/'diagnostics') as tmp:
    original=pretrain.OUT;pretrain.OUT=Path(tmp)
    try:
        (pretrain.OUT/'metrics').mkdir()
        for step,score,loss in ((13000,35.00,1.02),(14000,35.01,1.01),(15000,35.02,1.01),(16000,35.03,1.00)):
            (pretrain.OUT/'metrics'/f'validation_step_{step}.json').write_text(json.dumps({'step':step,'psnr':score,'recovery_loss':loss}))
        assert pretrain.convergence_check(cfg,15000)['stable'] is False
        assert pretrain.convergence_check(cfg,16000)['stable'] is True
        (pretrain.OUT/'metrics/validation_step_16000.json').write_text(json.dumps({'step':16000,'psnr':35.4,'recovery_loss':.8}))
        assert pretrain.convergence_check(cfg,16000)['stable'] is False
    finally:pretrain.OUT=original
class ProtectedTask(nn.Module):
    def __init__(self):
        super().__init__();self.model=nn.Module();self.model.net_g=nn.Linear(3,3);self.model.prior=nn.Linear(3,3)
        self.model.requires_grad_(False)
    @property
    def net_g(self):return self.model.net_g
model=JointEnhancer(PolarEnhancer(8,(1,1,1),(1,2,4)));task=ProtectedTask()
opt=study.optimizer_for(model,task,cfg);original={n:p.clone() for n,p in task.net_g.named_parameters()}
study.configure_step(task,500);rates=study.apply_learning_rates(opt,cfg,500)
assert rates=={'correction':2e-5,'interfaces':1e-4,'task':0.}
opt.zero_grad(set_to_none=True);next(model.parameters()).square().mean().backward();opt.step()
assert all(torch.equal(p,original[n]) for n,p in task.net_g.named_parameters())
study.configure_step(task,501);rates=study.apply_learning_rates(opt,cfg,501)
assert math.isclose(rates['task'],1e-6/200)
assert not any(p.requires_grad for p in task.model.prior.parameters())
study.configure_step(task,700);rates=study.apply_learning_rates(opt,cfg,700)
assert rates['task']==1e-6 and math.isclose(rates['interfaces']/rates['task'],100)
opt.zero_grad(set_to_none=True);task.net_g(torch.ones(2,3)).square().mean().backward();opt.step()
assert any(not torch.equal(p,original[n]) for n,p in task.net_g.named_parameters())
result={'continuous_decreasing_extension_schedule':True,'minimum_step_and_validation_stability_gate':True,
        'official_source_and_original_deadline':True,'frozen_task_exactly_unchanged':True,'separate_learning_rates_and_unfreeze_ramp':True,'prior_remains_frozen':True}
(root/'diagnostics/amended_policy_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
