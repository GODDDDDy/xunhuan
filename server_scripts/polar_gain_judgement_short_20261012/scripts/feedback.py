"""Task-specific cross-round correction, inspired by MPRNet/IBCLN/RDNet.

These papers support feature/state exchange, not the efficacy of this combination.
Quality scores are detached cues; low confidence never disables observation access.
"""
import torch
from torch import nn
from torch.nn import functional as F


def resize(x, size):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


def zero_output(module):
    nn.init.zeros_(module.weight)
    if module.bias is not None: nn.init.zeros_(module.bias)


class ResidualAdapter(nn.Module):
    def __init__(self, inputs, outputs, hidden=32):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(inputs, hidden, 1), nn.GELU(),
                                  nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
                                  nn.GELU(), nn.Conv2d(hidden, outputs, 1))
        zero_output(self.body[-1])

    def forward(self, inputs):
        return .1 * torch.tanh(self.body(inputs))


class CorrectionFeedback(nn.Module):
    def __init__(self):
        super().__init__()
        self.reference = nn.ModuleDict({f'T{i}': nn.Conv2d(c, r, 1)
                                        for i, (c, r) in enumerate(((96,8),(96,16),(192,16)),1)})
        self.first_polar = nn.ModuleDict({f'P{i}': nn.Conv2d(c,8,1)
                                         for i,c in enumerate((48,96,192),1)})
        self.second_polar = nn.ModuleDict({f'P{i}': nn.Conv2d(c,r,1)
                                          for i,(c,r) in enumerate(((48,8),(96,16),(192,16)),1)})
        self.state = nn.ModuleDict({f'level{i}': nn.Sequential(
            nn.Conv2d(24+r,32,1),nn.GELU(),nn.Conv2d(32,32,3,padding=1,groups=32),
            nn.GELU(),nn.Conv2d(32,16,1),nn.GELU()) for i,r in enumerate((8,16,16),1)})
        self.polar_inject = nn.ModuleDict({'level2':ResidualAdapter(16,96),
                                            'level3':ResidualAdapter(16,192)})
        self.decoder_project = nn.ModuleDict({'level2':nn.Conv2d(96,16,1),
                                              'level3':nn.Conv2d(192,16,1)})
        self.decoder_inject = nn.ModuleDict({'level2':ResidualAdapter(66,96),
                                             'level3':ResidualAdapter(66,192)})
        self.output_project = nn.Conv2d(96,16,1)
        # Learned RGB increment relative to T1. Not a relabelled second reconstruction.
        self.output_head = nn.Sequential(nn.Conv2d(59,32,1),nn.GELU(),
                                         nn.Conv2d(32,32,3,padding=1),nn.GELU(),nn.Conv2d(32,3,3,padding=1))
        zero_output(self.output_head[-1])

    def make_context(self, image, t1, raw_summary, first_features, first_polar, confidence):
        context={'confidence':confidence.detach(),'reference':{},'state':{}}
        for i in (1,2,3):
            key=f'T{i}'; reference=self.reference[key](first_features[key])
            pol=self.first_polar[f'P{i}'](first_polar[f'P{i}'].float())
            size=reference.shape[-2:]; c=resize(confidence.detach(),size)
            # Unscaled reference/context is accessible even when C=0.
            cues=torch.cat((resize(image,size),resize(t1,size),resize(image-t1,size),
                            resize(raw_summary.detach(),size),reference,resize(pol,size),c,1-c),1)
            context['reference'][key]=reference
            context['state'][f'level{i}']=self.state[f'level{i}'](cues)
        return context

    def polar_context(self, raw_features):
        return {n:self.second_polar[n](value.float()) for n,value in raw_features.items()}

    def fuse(self, feature, level, context, polar):
        i=int(level[-1]);size=feature.shape[-2:]
        c=resize(context['confidence'],size)
        reliable=c*resize(context['reference'][f'T{i}'],size)
        inputs=torch.cat((self.decoder_project[level](feature),reliable,
                          resize(polar[f'P{i}'],size),resize(context['state'][level],size),c,1-c),1)
        return feature+self.decoder_inject[level](inputs)

    def correct(self, task_features, candidate, image, t1, context, polar):
        size=t1.shape[-2:];c=resize(context['confidence'],size)
        inputs=torch.cat((self.output_project(task_features['T1']),
                          resize(context['state']['level1'],size),
                          c*resize(context['reference']['T1'],size),resize(polar['P1'],size),
                          image,t1,candidate-t1,c,1-c),1)
        increment=.25*torch.tanh(self.output_head(inputs))
        return t1+increment,increment


def second_task_forward(task, feedback, inputs, context, polar):
    """Use the actual official forward, with two temporary decoder feature hooks.

    The complete closure is activation-checkpointed by the caller, so backward
    recomputation installs/removes the same hooks. Official state keys stay intact.
    """
    hooks=[]
    try:
        for level in ('level3','level2'):
            module=getattr(task.net_g,f'decoder_{level}')
            def fuse_hook(module,args,output,level=level):
                return feedback.fuse(output,level,context,polar)
            hooks.append(module.register_forward_hook(fuse_hook))
        return task.forward_inputs(inputs,None)
    finally:
        for hook in hooks:hook.remove()
