"""PIDSR-derived same-resolution correction + SGSC + task adapters.

PIDSR CVPR2025: https://github.com/PRIS-CV/PIDSR
SGSC IJCV2025: https://github.com/fourson/Learning-to-Deblur-Polarized-Images
Task separation is inspired by UniRestore; this adapter is our task-specific design.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from vendor.pidsr_official import PIDSRNet
from vendor.sgsc_primitives import get_norm_layer, ResBlock, SEBlock


class SGSC(nn.Sequential):
    def __init__(self, channels):
        super().__init__(nn.Conv2d(channels*2, channels, 1),
                         ResBlock(channels, get_norm_layer('instance'), False, True),
                         SEBlock(channels, 8))


class TaskAdapter(nn.Module):
    def __init__(self, channels, task_channels):
        super().__init__()
        self.observation = nn.Conv2d(3, channels, 1)
        self.current = nn.Conv2d(12, channels, 1)
        self.prediction = nn.Conv2d(6, channels, 1)
        self.task = nn.Conv2d(task_channels, channels, 1)
        self.mix = nn.Conv2d(channels*4+2, channels, 1)
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.out = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, feature, p0, current, image, prediction, task_feature, confidence):
        size = feature.shape[-2:]
        resize = lambda x: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        available = prediction is not None
        if available != (task_feature is not None):
            raise ValueError('Tprev and Fprev must be jointly present or absent')
        c = feature.new_ones((feature.shape[0],1,*size)) if confidence is None else resize(confidence)
        if not torch.isfinite(c).all() or (c < 0).any() or (c > 1).any():
            raise ValueError('Confidence must be finite and in [0,1]')
        state = torch.zeros_like(feature)
        if available:
            state = c * (self.prediction(resize(torch.cat((prediction, image-prediction),1)))
                         + self.task(resize(task_feature)))
        flag = torch.full_like(c, float(available))
        inputs = torch.cat((feature, self.observation(resize(image)),
                            self.current(resize(current-p0)), state, c, flag),1)
        return feature + .1*torch.tanh(self.out(F.gelu(self.depthwise(F.gelu(self.mix(inputs))))))


class PolarEnhancer(PIDSRNet):
    def __init__(self, dim=48, blocks=(4,6,6), heads=(1,2,4)):
        super().__init__(dim=dim, num_blocks=list(blocks), heads=list(heads))
        # User clarified that pretraining targets this correction module at the
        # observation resolution. Retain PIDSR's polarized restoration branch.
        for name in list(self._modules):
            if name.startswith('sr_') or name=='con1x1':delattr(self,name)
        self.sgsc_full, self.sgsc_half = SGSC(dim), SGSC(dim*2)
        self.de_reduce_full = nn.Conv2d(dim*2, dim, 1, bias=False)
        self.adapters = nn.ModuleDict({'level2': TaskAdapter(dim*2,96),
                                       'level3': TaskAdapter(dim*4,192)})
        self.activation_checkpoint = False

    def block(self, module, *inputs):
        if self.activation_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(module, *inputs, use_reentrant=False, preserve_rng_state=True)
        return module(*inputs)

    def sequence(self, modules, x, p=None):
        for module in modules:
            x = self.block(module, x, p) if p is not None else self.block(module, x)
        return x

    def decode(self, p0, current=None, image=None, prediction=None, task_features=None,
               confidence=None, conditioning=False):
        height,width = p0.shape[-2:]
        pad=(0,(-width)%4,0,(-height)%4)
        if any(pad):
            p0=F.pad(p0,pad,mode='reflect')
            if current is not None: current=F.pad(current,pad,mode='reflect')
            if image is not None: image=F.pad(image,pad,mode='reflect')
            if prediction is not None: prediction=F.pad(prediction,pad,mode='reflect')
            if confidence is not None: confidence=F.pad(confidence,pad,mode='replicate')
        p = self.polar_feature_extractor(self.calculate_polar(p0))
        x1=self.sequence(self.encoder1,self.img_feature_extractor(p0),p)
        p2=self.down1(p)
        x2=self.sequence(self.encoder2,self.down1(x1),p2)
        p3=self.down2(p2)
        x3=self.sequence(self.bottleneck,self.down2(x2),p3)
        if conditioning:
            x3=self.adapters['level3'](x3,p0,current,image,prediction,
                                        None if task_features is None else task_features['T3'],confidence)
        skip2=self.block(self.sgsc_half,torch.cat((x2,p2),1))
        d2=self.de_reduce_chan3(torch.cat((self.de_up3(x3),skip2),1))
        d2=self.sequence(self.de_decoder2,d2,p2)
        if conditioning:
            d2=self.adapters['level2'](d2,p0,current,image,prediction,
                                     None if task_features is None else task_features['T2'],confidence)
        skip1=self.block(self.sgsc_full,torch.cat((x1,p),1))
        d1=self.de_reduce_full(torch.cat((self.de_up2(d2),skip1),1))
        d1=self.sequence(self.de_decoder1,d1,p)
        decoded=d1
        d1=self.sequence(self.mid_refine,decoded)
        residual=self.mid_output(d1)
        mid=(p0+residual)[...,:height,:width]
        features={'P1':d1[...,:height,:width], 'P2':d2, 'P3':x3}
        return mid,features,residual[...,:height,:width]

    def pretrain_forward(self, polar):
        mid,_,_=self.decode(polar)
        return mid

    def forward(self, P0, Pcurrent, I, Tprev=None, Fprev=None, Cprev=None):
        if P0.ndim!=4 or P0.shape[1]!=12 or Pcurrent.shape!=P0.shape:
            raise ValueError('Expected aligned Bx12xHxW P0/Pcurrent')
        if I.shape!=(P0.shape[0],3,*P0.shape[-2:]):
            raise ValueError('RGB observation must align with polarization')
        mid,features,residual=self.decode(P0,Pcurrent,I,Tprev,Fprev,Cprev,conditioning=True)
        refined=mid.clamp(0,1)
        return refined,features,residual,refined-P0

    def downstream_parameters(self):
        for name,param in self.named_parameters():
            # Official FeedForward_P registers an unused dwconv; preserve author state keys.
            active=not ('.ffn.dwconv.' in name and not name.startswith('mid_refine.'))
            param.requires_grad_(active)
        return [p for p in self.parameters() if p.requires_grad]


class QualityHead(nn.Module):
    def __init__(self, initial_error=.03):
        super().__init__()
        self.projections=nn.ModuleDict({f'T{i}':nn.Conv2d(c,8,1)
                                       for i,c in enumerate((96,96,192),1)})
        self.body=nn.Sequential(nn.Conv2d(38,32,3,padding=1),nn.GELU(),
                                nn.Conv2d(32,32,3,padding=1),nn.GELU(),nn.Conv2d(32,1,1))
        self.initialize(initial_error)

    def initialize(self,error):
        nn.init.zeros_(self.body[-1].weight)
        nn.init.constant_(self.body[-1].bias,math.log(math.expm1(max(float(error),.002)-.001)))

    def forward(self,image,prediction,polar_features,task_features):
        image,prediction=image.detach(),prediction.detach().clamp(0,1)
        inputs=[F.avg_pool2d(x,4) for x in (image,prediction,(image-prediction).abs(),polar_features.detach())]
        size=inputs[0].shape[-2:]
        inputs += [head(F.interpolate(task_features[name].detach(),size=size,mode='bilinear',align_corners=False))
                   for name,head in self.projections.items()]
        return .001+F.softplus(self.body(torch.cat(inputs,1)))


@dataclass
class EnhancedOutput:
    refined_polar: torch.Tensor
    projected_features: dict
    raw_features: dict
    delta: torch.Tensor
    effective_delta: torch.Tensor


class JointEnhancer(nn.Module):
    def __init__(self, core=None):
        super().__init__()
        self.r2p=core if core is not None else PolarEnhancer()
        dim=self.r2p.img_feature_extractor.proj.out_channels
        self.projections=nn.ModuleDict({f'P{i}':nn.Conv2d(c,t,1,bias=False)
                                       for i,(c,t) in enumerate(zip((dim,dim*2,dim*4),(96,96,192)),1)})
        self.confidence_head=QualityHead()
        self.register_buffer('feedback_scale',torch.tensor(.1))
        self.initial_r2p={n:p.detach().cpu().clone() for n,p in self.r2p.named_parameters()}
        self.step=0
        self.tau=.03

    def forward(self,P0,Pcurrent,I,Tprev=None,Fprev=None,Cprev=None,project=True):
        p,features,delta,effective=self.r2p(P0,Pcurrent,I,Tprev,Fprev,Cprev)
        projected={n:self.projections[n](f) for n,f in features.items()} if project else {}
        return EnhancedOutput(p,projected,features,delta,effective)
