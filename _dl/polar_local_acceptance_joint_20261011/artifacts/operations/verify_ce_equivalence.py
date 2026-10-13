"""Check values and gradients against weighted CE before formal training."""
import json, sys
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from acceptance import judgement_loss

torch.manual_seed(2026)
maximum_loss_difference=maximum_gradient_difference=0.
for neutral_only in (False,True):
    x=torch.randn(2,4,8,9,requires_grad=True)
    y=torch.zeros(2,8,9,dtype=torch.long) if neutral_only else torch.randint(0,3,(2,8,9))
    n=torch.randint(0,2,(2,1,8,9)).float()
    counts=torch.bincount(y.flatten(),minlength=3).float()
    weights=(counts.sum()/(3*counts.clamp_min(1))).clamp(.25,4)
    reference=F.binary_cross_entropy_with_logits(x[:,:1],n)+F.cross_entropy(x[:,1:],y,weight=weights)
    actual=judgement_loss(x,n,y)
    gr=torch.autograd.grad(reference,x,retain_graph=True)[0]
    ga=torch.autograd.grad(actual,x)[0]
    torch.testing.assert_close(actual,reference,rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(ga,gr,rtol=1e-6,atol=1e-7)
    maximum_loss_difference=max(maximum_loss_difference,float((actual-reference).abs().detach()))
    maximum_gradient_difference=max(maximum_gradient_difference,float((ga-gr).abs().max()))
result=dict(passed=True,mixed_and_single_class_checked=True,
            maximum_loss_difference=maximum_loss_difference,
            maximum_gradient_difference=maximum_gradient_difference)
(ROOT/'diagnostics/ce_equivalence.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
