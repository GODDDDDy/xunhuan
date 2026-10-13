"""Official PIDSR loss groups with circular AoLP and target-residual constraints."""
import torch
from torch import nn
from torch.nn import functional as F

def stokes(x):
    a,b,c,d=x.float().chunk(4,1)
    s0=(a+b+c+d)/2;s1=a-c;s2=b-d
    radius=torch.sqrt(s1.square()+s2.square()+1e-12)
    dop=(radius/s0.clamp_min(1e-6)).clamp(0,1)
    # Finite surrogate at unpolarized input; masks are defined only from supervision.
    aop=.5*torch.atan2(s2+1e-6,s1+1e-6)
    valid=(s0>2/255)&(torch.stack((a,b,c,d)).amax(0)<253/255)
    angle_valid=valid&(dop>.01)
    return s0,s1,s2,dop,aop,valid,angle_valid,(a+c-b-d)

def masked_mean(x,mask):
    return (x*mask).sum()/mask.sum().clamp_min(1)

def circular(a,b):
    return .5*torch.atan2(torch.sin(2*(a-b)),torch.cos(2*(a-b))).abs()

class RecoveryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('kx',torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=torch.float32)[None,None])
        self.register_buffer('ky',torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=torch.float32)[None,None])

    def gradient_loss(self,p,t):
        c=p.shape[1]
        gx=F.conv2d(p,self.kx.expand(c,1,3,3),padding=1,groups=c)
        gy=F.conv2d(p,self.ky.expand(c,1,3,3),padding=1,groups=c)
        tx=F.conv2d(t,self.kx.expand(c,1,3,3),padding=1,groups=c)
        ty=F.conv2d(t,self.ky.expand(c,1,3,3),padding=1,groups=c)
        return .8*(F.l1_loss(gx,tx)+F.l1_loss(gy,ty))+.4*F.l1_loss(p,t)

    def forward(self,p,t):
        p,t=p.float(),t.float();a,b=stokes(p),stokes(t)
        image=self.gradient_loss(p,t)+self.gradient_loss(a[0],b[0])
        pol=F.l1_loss(a[1],b[1])+F.l1_loss(a[2],b[2])
        pol+=masked_mean((a[3]-b[3]).abs(),b[5])+masked_mean(circular(a[4],b[4]),b[6])
        pol+=F.l1_loss(a[7],b[7])
        return image+10*pol

def correction_loss(model,batch,criterion):
    prediction=model.pretrain_forward(batch['polar'])
    return criterion(prediction,batch['target']),prediction
