"""CPU checks for scientific contracts; no real GPU allocation."""
import copy,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
local_source=Path('F:/AAAketi/rgb_final_confirmation/src')
sys.path.insert(0,str(local_source if local_source.exists() else Path(__file__).resolve().parents[2]/'deps/rgb_final_confirmation/src'))
import cv2,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from enhancer import PolarEnhancer,JointEnhancer,TaskAdapter
from pidsr_data import augment,degrade,PolarDataset
from polar_loss import stokes,circular,RecoveryLoss
from common import ROOT,json_save,rng_state,restore_rng
import study

torch.set_num_threads(4);torch.manual_seed(2026)
results={}
model=PolarEnhancer(dim=8,blocks=(1,1,1),heads=(1,2,4)).eval()
p=torch.rand(1,12,16,20);rgb=torch.rand(1,3,16,20)
with torch.no_grad():
    expected=model.pretrain_forward(p).clamp(0,1)
    observed,features,raw,effective=model(p,p,rgb)
assert torch.equal(expected,observed)
assert [f.shape[-2:] for f in features.values()]==[(16,20),(8,10),(4,5)]
assert not any(n.startswith('sr_') for n in model.state_dict())
results['zero_adapters_preserve_same_resolution_restorer']=True
results['three_real_scales']=True
with torch.no_grad():
    q=torch.rand(1,12,17,21);z=model(q,q,torch.rand(1,3,17,21))
assert z[0].shape==q.shape
results['padding_original_size']=True

# Finite losses/gradients in dark, unpolarized and saturated supervision.
criterion=RecoveryLoss()
for value in (0.,.5,1.):
    pred=torch.full((1,12,16,16),value,requires_grad=True)
    loss=criterion(pred,torch.full_like(pred,value));loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
assert float(circular(torch.tensor(math.pi/2-.001),torch.tensor(-math.pi/2+.001)))<.003
results['masked_polar_losses_and_pi_periodicity']=True

# Synthetic Stokes verifies analyzer permutations under mirror and rotation.
y,x=np.mgrid[:64,:64];theta=(x+y)/100
angles=np.arange(4)*np.pi/4
arr=np.stack([.45+.12*np.cos(2*(a-theta)) for a in angles],axis=2)
arr=np.repeat(arr[...,None],3,axis=3);arr=np.round(arr*65535).astype(np.uint16)
to_tensor=lambda a:torch.tensor(a.astype(np.float32)/65535).permute(2,3,0,1).reshape(1,12,*a.shape[:2])
s=stokes(to_tensor(arr))
mirror=stokes(to_tensor(augment(arr,1)));rot=stokes(to_tensor(augment(arr,2)))
assert torch.equal(mirror[1],s[1].flip(-1)) and torch.equal(mirror[2],-s[2].flip(-1))
assert torch.equal(rot[1],-torch.rot90(s[1],1,(-2,-1))) and torch.equal(rot[2],-torch.rot90(s[2],1,(-2,-1)))
results['polarization_geometry_and_stokes_signs']=True

constant=np.zeros((64,64,4,3),np.uint16);constant[:]=np.round(np.array([.1,.2,.8])*65535)
batch=degrade(constant)
assert set(batch)=={'polar','target'}
assert torch.allclose(batch['polar'][0:3,8,8],torch.tensor([.8,.2,25/255]),atol=.001)
results['canonical_RGB_CPFA_bitdepth_and_no_guide']=True

adapter=TaskAdapter(8,12)
nn.init.normal_(adapter.out.weight,std=.1)
state=torch.rand(1,3,16,20,requires_grad=True);feat=torch.rand(1,12,8,10,requires_grad=True)
args=(torch.rand(1,8,8,8),p,p,rgb,state,feat)
zero=adapter(*args,torch.zeros(1,1,16,20))
g=torch.autograd.grad(zero.sum(),(state,feat),retain_graph=True)
assert all(torch.count_nonzero(x)==0 for x in g)
one=adapter(*args,torch.ones(1,1,16,20));g=torch.autograd.grad(one.sum(),(state,feat))
assert all(torch.count_nonzero(x)>0 for x in g)
assert (zero-args[0]).abs().max()<=.100001
results['C0_removes_feedback_C1_allows_feedback_update_bounded']=True

class FakeTask(nn.Module):
    def forward(self,p,rgb,unused):
        image=(p[:,:3]+rgb)/2
        return image,{f'T{i}':F.interpolate(image.mean(1,keepdim=True).repeat(1,c,1,1),scale_factor=1/(2**(i-1)))
                      for i,c in enumerate((96,96,192),1)}

joint=JointEnhancer(model);task=FakeTask();joint.activation_checkpoint=False;joint.eval()
batch={'polar':p,'official_rgb':rgb,'target':torch.rand_like(rgb),'sample_id':['fake'],'scene':['fake']}
counts={'E':0,'D':0}
h1=joint.r2p.register_forward_hook(lambda *a:counts.__setitem__('E',counts['E']+1))
h2=task.register_forward_hook(lambda *a:counts.__setitem__('D',counts['D']+1))
with torch.no_grad():first=study.recurrent_forward(joint,task,batch)
assert counts=={'E':2,'D':2};h1.remove();h2.remove()
batch['target']=torch.rand_like(rgb)
with torch.no_grad():second=study.recurrent_forward(joint,task,batch)
assert torch.equal(first.t1,second.t1) and torch.equal(first.t2,second.t2)
results['two_actual_rounds_and_GT_independent_inference']=True
joint.train();joint.step=501;joint.r2p.activation_checkpoint=True
out=study.recurrent_forward(joint,task,batch,True)
out.t2.mean().backward(retain_graph=True)
assert not any(v.grad is not None for v in joint.confidence_head.parameters())
for v in joint.parameters():v.grad=None
out.error_scale.mean().backward()
assert not any(v.grad is not None for v in joint.r2p.parameters())
results['head_task_gradient_separation']=True

state=copy.deepcopy(joint.state_dict());clone=JointEnhancer(PolarEnhancer(8,(1,1,1),(1,2,4)))
clone.load_state_dict(state,strict=True)
assert all(torch.equal(v,clone.state_dict()[k]) for k,v in state.items())
rng=rng_state();a=torch.rand(5);restore_rng(rng);assert torch.equal(a,torch.rand(5))
results['strict_weight_reload_and_RNG_restore']=True

# A real Adam update before/after restoration must agree, including moment state.
small=PolarEnhancer(8,(1,1,1),(1,2,4));opt=torch.optim.Adam(small.parameters(),lr=1e-4)
def update(m,o):
    o.zero_grad(set_to_none=True);loss=m.pretrain_forward(p).square().mean();loss.backward();o.step()
update(small,opt);weights=copy.deepcopy(small.state_dict());moments=copy.deepcopy(opt.state_dict());r=rng_state()
update(small,opt)
resumed=PolarEnhancer(8,(1,1,1),(1,2,4));resumed.load_state_dict(weights,strict=True)
resumed_opt=torch.optim.Adam(resumed.parameters(),lr=1e-4);resumed_opt.load_state_dict(moments);restore_rng(r)
update(resumed,resumed_opt)
assert all(torch.equal(v,resumed.state_dict()[k]) for k,v in small.state_dict().items())
results['actual_Adam_checkpoint_resume_bitwise_CPU']=True
json_save(ROOT/'diagnostics/cpu_contract_checks.json',results)
print(json.dumps(results))
