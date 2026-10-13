"""Predicted repair-need cues condition the second-round state, never the first.

SPAIR/LANet motivate internal localization. This adapter and its efficacy are
research hypotheses, not a reproduction or calibrated reflection detector.
"""
import torch
from torch import nn
from torch.nn import functional as F
from feedback import ResidualAdapter, resize

class ErrorGuidance(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Conv2d(96, 8, 1)
        self.predict = nn.Sequential(nn.Conv2d(24, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32), nn.GELU(), nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.predict[-1].weight); nn.init.zeros_(self.predict[-1].bias)
        self.inject = nn.ModuleDict({f'level{i}': ResidualAdapter(18+r, 16)
                                     for i,r in enumerate((8,16,16),1)})

    def forward(self, image, t1, polar, error, confidence, feature, context, override=None):
        # Q cannot alter its labels or obtain candidate-loss gradients by changing
        # its confidence; GT appears only in the supervised loss, outside forward.
        maps=[F.avg_pool2d(x.detach().float(),4) for x in (image,t1,image-t1,polar)]
        size=maps[0].shape[-2:]
        maps += [resize(error.detach(),size),resize(confidence.detach(),size),
                 self.project(resize(feature.detach(),size))]
        logits=self.predict(torch.cat(maps,1)); need=logits.sigmoid().detach()
        if override=='need_zero': need=torch.zeros_like(need)
        if override=='need_one': need=torch.ones_like(need)
        result={**context,'state':dict(context['state'])}
        if override!='guidance_off':
            for i in (1,2,3):
                key=f'level{i}'; state=context['state'][key]; size=state.shape[-2:]
                q=resize(need,size); c=resize(confidence.detach(),size)
                reference=resize(context['reference'][f'T{i}'],size)
                cues=torch.cat((q*state,(1-q)*reference,q,c),1)
                result['state'][key]=state+self.inject[key](cues)
        return result,logits,need
