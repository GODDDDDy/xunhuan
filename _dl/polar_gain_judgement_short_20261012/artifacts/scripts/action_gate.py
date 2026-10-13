"""Predict action costs, not a product of confidence probabilities."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from acceptance import resize

ACTIONS=(0.,.25,.5,1.)

def action_costs(t1,delta,target):
    # All tensors are actual un-clamped outputs, consistent with official PSNR.
    costs=torch.stack([F.avg_pool2d((t1+a*delta-target).square().mean(1,keepdim=True),4)[:,0] for a in ACTIONS],1)
    return costs-costs[:,:1]

class ActionCostGate(nn.Module):
    def __init__(self,source):
        super().__init__()
        self.feature_projection=copy.deepcopy(source.feature_projection)
        self.body=nn.Sequential(copy.deepcopy(source.body[0]),nn.GELU(),copy.deepcopy(source.body[2]),nn.GELU(),nn.Conv2d(32,3,1))
        nn.init.zeros_(self.body[-1].weight);nn.init.zeros_(self.body[-1].bias)
    def forward(self,image,polar,t1,proposal,error_scale,confidence,feature,state,bias=None,margin=0.):
        a,p=t1.detach().clamp(0,1),proposal.detach().clamp(0,1)
        maps=[F.avg_pool2d(x.detach().float(),4) for x in (image,a,p,p-a,polar)]
        size=maps[0].shape[-2:]
        maps += [resize(error_scale.detach(),size),resize(confidence.detach(),size),self.feature_projection(resize(feature.detach(),size)),resize(state.detach(),size)]
        raw=self.body(torch.cat(maps,1))
        corrected=raw if bias is None else raw+raw.new_tensor(bias)[None,:,None,None]
        costs=torch.cat((torch.zeros_like(corrected[:,:1]),corrected+margin),1)
        decision=costs.argmin(1,keepdim=True)
        alpha=raw.new_tensor(ACTIONS)[decision]
        mask=F.interpolate(alpha,size=image.shape[-2:],mode='nearest')
        return dict(raw_cost=raw,corrected_cost=corrected,costs=costs,decision=decision,mask=mask)

def compose(t1,delta,mask):
    mask=mask.detach()
    return torch.where(mask==0,t1,t1+mask*delta)

def cost_loss(predicted,true_cost,scale):
    target=(true_cost[:,1:]/scale).detach()
    fit=F.smooth_l1_loss(predicted,target)
    costs=torch.cat((torch.zeros_like(predicted[:,:1]),predicted),1)
    probability=(-costs).softmax(1)
    expected=(probability*(true_cost/scale).detach()).sum(1).mean()
    return fit+.1*expected
