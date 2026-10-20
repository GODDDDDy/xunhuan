"""Supplementary CPU check: actual rounds and frozen task's input gradient."""
import copy,torch
import matched_run as run
from common import ROOT,json_save,seed
torch.set_num_threads(2)
cfg,plan,model,task,raw,source=run.load(torch.device('cpu'));small=copy.deepcopy(cfg);small['data']['crop_size']=32
batch=run.study.base.unpack_batch(next(run.study.microbatches(next(iter(run.staged.loader(small,raw,'candidate',1))))),torch.device('cpu'))
run.configure(model,task);counts=dict(E=0,D=0,M_old=0,M_new=0);handles=[]
for name,module in (('E',model.r2p),('D',task.net_g),('M_old',model.update_gate),('M_new',model.action_gate)):
    handles.append(module.register_forward_hook(lambda *a,name=name:counts.__setitem__(name,counts[name]+1)))
seed(100);out=run.forward(model,task,batch,'real_quality',.03)
for h in handles:h.remove()
assert counts==dict(E=2,D=2,M_old=0,M_new=0),counts
assert out.refined_polar.requires_grad
out.refined_polar.retain_grad();loss=run.staged.candidate_loss(out,batch['target']);loss.backward()
norm=float(out.refined_polar.grad.abs().sum());assert norm>0
assert all(p.grad is None for p in task.parameters());run.audit(model,task,source)
json_save(ROOT/'diagnostics/input_gradient_checks.json',dict(passed=True,actual_forward_counts=counts,second_polar_input_gradient_l1=norm,task_parameters_no_gradient=True,GT_only_loss_labels=True,updates_discarded=True))
print('INPUT_GRADIENT_CHECK_PASSED',norm,flush=True)
