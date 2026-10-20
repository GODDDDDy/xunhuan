"""Explicit paired freeze policy; source model and priors are never overwritten."""
import copy
import hashlib
from contextlib import contextmanager
import torch

TAIL_PREFIXES = ('decoder_level1.', 'output.')

def tail_state(down):
    return {n:v.detach().clone() for n,v in down.task.net_g.state_dict().items()
            if n.startswith(TAIL_PREFIXES)}

def load_tail(down,state):
    actual=down.task.net_g.state_dict()
    expected={n for n in actual if n.startswith(TAIL_PREFIXES)}
    assert set(state)==expected
    with torch.no_grad():
        for n,v in state.items(): actual[n].copy_(v)

@contextmanager
def reference_tail(down):
    current=tail_state(down)
    load_tail(down,down._initial_tail)
    try: yield
    finally: load_tail(down,current)

def configure(up,down,arm):
    assert arm in ('frozen','partial')
    up.core.requires_grad_(False).eval()
    down.task.requires_grad_(False).eval()
    up.adapters.requires_grad_(True).eval()
    down.adapters.requires_grad_(True).eval()
    if arm=='partial':
        down.task.net_g.decoder_level1.requires_grad_(True).eval()
        down.task.net_g.output.requires_grad_(True).eval()
    down._initial_tail=tail_state(down)
    return {'arm':arm,'upstream_adapters':sum(p.numel() for p in up.adapters.parameters()),
            'downstream_adapters':sum(p.numel() for p in down.adapters.parameters()),
            'new_tail_parameters':sum(p.numel() for p in down.task.parameters() if p.requires_grad),
            'new_tail_names':[n for n,p in down.named_parameters() if n.startswith('task.') and p.requires_grad],
            'all_priors_frozen':all(not p.requires_grad for m in (down.task.model.net_d,down.task.model.net_le_dm) for p in m.parameters())}

def optimizer(up,down,saved=None):
    base=[p for m in (up.adapters,down.adapters) for p in m.parameters()]
    opt=torch.optim.Adam(base,lr=1e-4,eps=1e-8,foreach=False)
    tail=[p for p in down.task.parameters() if p.requires_grad]
    old=(saved or {}).get('optimizer')
    if old and len(old['param_groups'])==1: opt.load_state_dict(old)
    if tail: opt.add_param_group({'params':tail,'lr':1e-6})
    if old and len(old['param_groups'])>1: opt.load_state_dict(old)
    assert [g['lr'] for g in opt.param_groups]==([1e-4,1e-6] if tail else [1e-4])
    return opt

def fixed_task_hash(down):
    ignored={'model.net_g.'+n for n,p in down.task.net_g.named_parameters()
             if p.requires_grad}
    h=hashlib.sha256()
    for n,v in down.task.state_dict().items():
        if n in ignored: continue
        h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def restore(up,down,fa,saved,restore_rng=True):
    up.adapters.load_state_dict(saved['upstream_adapters'],strict=True)
    down.adapters.load_state_dict(saved['downstream_adapters'],strict=True)
    fa.load_state_dict(saved['alignment_state'],strict=True)
    if 'task_tail' in saved:load_tail(down,saved['task_tail'])
    opt=optimizer(up,down,saved)
    fa.set_trainable_partition()
    opt_fa=torch.optim.Adam([p for p in fa.parameters() if p.requires_grad],lr=1e-5,eps=1e-8,foreach=False)
    opt_fa.load_state_dict(saved['optimizer_fa'])
    if restore_rng:
        import random
        torch.set_rng_state(saved['torch_rng'])
        if torch.cuda.is_available():torch.cuda.set_rng_state_all(saved['cuda_rng'])
        random.setstate(saved['python_rng'])
    return opt,opt_fa
