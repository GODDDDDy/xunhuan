"""Read-only CPU module inventory and a mathematical loss check; no inference/training."""
import os,sys,json,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
ROOT=Path('/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/polar_gain_judgement_short_20261012')
sys.path.insert(0,str(ROOT/'scripts'))
import torch
torch.set_num_threads(2)
from enhancer import JointEnhancer
from action_gate import ActionCostGate,cost_loss
from error_guidance import ErrorGuidance
from gain22 import FloatActionCostGate
m=JointEnhancer();m.action_gate=FloatActionCostGate(ActionCostGate(m.update_gate));m.error_guidance=ErrorGuidance()
m.requires_grad_(False);m.action_gate.requires_grad_(True)
state=torch.load(ROOT/'downstream/checkpoints/final.pt',map_location='cpu',weights_only=False)
m.load_state_dict(state['r2p_and_projections'],strict=True)
modules=[dict(name=n or 'model',kind=type(v).__name__,direct_parameters=sum(p.numel() for p in v.parameters(recurse=False)),children=[k for k,_ in v.named_children()]) for n,v in m.named_modules()]
prefixes={k:dict(parameters=sum(p.numel() for p in v.parameters()),trainable=sum(p.numel() for p in v.parameters() if p.requires_grad)) for k,v in m.named_children()}
p=torch.tensor([.1,.3,-.2]).reshape(1,3,1,1).requires_grad_();truth=torch.tensor([0,.1,.3,-.2]).reshape(1,4,1,1)
loss=cost_loss(p,truth,1.);loss.backward()
# This proves the auxiliary term moves exact cost predictions; it is an intentional score objective, not a calibrated proper cost loss.
official=ROOT.parent/'deps/DfP'
sys.path.insert(0,str(official))
import yaml
from archs.Transformer_arch import Transformer
from archs.latent_encoder_arch import latent_encoder_gelu
from archs.denoising_arch import denoising
cfg=yaml.safe_load((official/'configs/dm_config.yaml').read_text())
net=Transformer(**cfg['network_g'])
net.load_state_dict(state['tasknet_net_g'],strict=True)
task_modules=[dict(name=n or 'net_g',kind=type(v).__name__,direct_parameters=sum(p.numel() for p in v.parameters(recurse=False))) for n,v in net.named_modules()]
priors={}
for key,cls in [('network_le_dm',latent_encoder_gelu),('network_d',denoising),('network_le',latent_encoder_gelu)]:
 mod=cls(**cfg[key]);priors[key]=dict(parameters=sum(p.numel() for p in mod.parameters()),modules=[dict(name=n or key,kind=type(v).__name__,direct_parameters=sum(p.numel() for p in v.parameters(recurse=False))) for n,v in mod.named_modules()]);del mod
result=dict(cpu_only=True,cuda_available=torch.cuda.is_available(),checkpoint_sha256=hashlib.sha256((ROOT/'downstream/checkpoints/final.pt').read_bytes()).hexdigest(),model=prefixes,module_inventory=modules,net_g_parameters=sum(p.numel() for p in net.parameters()),task_modules=task_modules,priors=priors,official_configuration=cfg,exact_cost_prediction_loss=float(loss.detach()),gradient_at_exact_cost_prediction=p.grad.flatten().tolist(),source_hashes={str(f.relative_to(official)):hashlib.sha256(f.read_bytes()).hexdigest() for f in list((official/'archs').glob('*.py'))+[official/'configs/dm_config.yaml']})
assert not result['cuda_available']
print(json.dumps(result,ensure_ascii=False))
