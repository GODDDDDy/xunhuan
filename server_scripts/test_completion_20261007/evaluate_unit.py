"""Reuse each exact trained model factory and evaluator, isolating all output writes."""
import copy,hashlib,importlib,json,sys,time
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parent.parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def tensor_hashes(model,task):
    return {prefix+n:hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for prefix,obj in [('model.',model),('task.',task)] for n,t in obj.state_dict().items()}
lock=json.loads((ROOT/'protocol_lock.json').read_text())
uid=sys.argv[1];u=next(v for v in lock['units'] if v['id']==uid)
cfg=yaml.safe_load(Path(u['config']).read_text());root=Path(cfg['paths']['output_root'])
assert not (root/'metrics/test.json').exists()
assert all(sha(p)==h for p,h in lock['locked_file_sha256'].items())
if u['checkpoint']:assert sha(u['checkpoint'])==u['checkpoint_sha256']
sys.path.insert(0,u['source_scripts'])
module=importlib.import_module('run_g6_phase' if u['extension']=='legacy_g6' else u['extension'])
base=module.base if u['extension']=='legacy_g6' else module.configure(cfg)
base.add_source_path(cfg);base.set_seed(u['seed'])
tick=time.time();model,task,criterion,raw=base.build_components(cfg,torch.device('cuda:0'))
cp=None
if u['checkpoint']:
    cp=torch.load(u['checkpoint'],map_location='cpu',weights_only=False)
    assert cp['step']==3000 and cp['phase_weight']==0.1
    if hasattr(base,'protocol_id'):assert cp['protocol_id']==base.protocol_id(cfg)
    model.load_state_dict(cp['r2p_and_projections'],strict=True)
    task.net_g.load_state_dict(cp['tasknet_net_g'],strict=True)
    assert all(torch.equal(t.detach().cpu(),cp['tasknet_net_g'][n]) for n,t in task.net_g.state_dict().items())
model.requires_grad_(False);task.requires_grad_(False);model.eval();task.eval()
del criterion,cp
before=tensor_hashes(model,task)
calls={'r2p':0,'tasknet':0}
def count(key):
    def hook(*args):
        calls[key]+=1
        if key=='tasknet' and calls[key]%(25*u['actual_rounds'])==0:
            print(f'Evaluated {calls[key]//u["actual_rounds"]} images',flush=True)
    return hook
handles=[model.register_forward_hook(count('r2p')),task.register_forward_hook(count('tasknet'))]
loader=base.split_loader(cfg,'test')
with torch.no_grad():
    if u['family'] in ('G0','G6'):metrics=base.validate(model,task,loader,torch.device('cuda:0'),2026,cfg)
    else:metrics=base.validate(model,task,loader,torch.device('cuda:0'),2026)
for h in handles:h.remove()
after=tensor_hashes(model,task)
assert before==after and all(p.grad is None for p in list(model.parameters())+list(task.parameters()))
assert calls['tasknet']==188*u['actual_rounds']
assert calls['r2p']==(0 if u['family'] in ('G0','G6') else 188*u['actual_rounds'])
assert int(metrics['frames'])==188 and int(metrics['scenes'])==8
assert [r['sample_id'] for r in metrics['per_image']]==lock['test_sample_order']
base.atomic_json(root/'metrics/inherited_test_raw.json',metrics)
canonical=copy.deepcopy(metrics)
if u['family'] not in ('G0','G6'):
    if u['actual_rounds']==1:assert all(r['round1']==r['round2'] for r in metrics['per_image'])
    canonical['per_image']=[dict(sample_id=r['sample_id'],scene=r['scene'],**r['round2']) for r in metrics['per_image']]
canonical.update(unit_id=uid,family=u['family'],training_seed=u['seed'] if u['checkpoint'] else None,
    evaluation_seed=2026,split='test',checkpoint_step=3000 if u['checkpoint'] else 0,checkpoint=u['checkpoint'],
    checkpoint_sha256=u['checkpoint_sha256'],actual_rounds=u['actual_rounds'],actual_forward_calls=calls,
    phase_weight=0.1 if u['checkpoint'] else None,evaluation_seconds=time.time()-tick,
    evaluator_sha256=sha(__file__),protocol_lock_sha256=sha(ROOT/'protocol_lock.json'),
    scope='fixed-checkpoint inference on previously examined test split')
base.atomic_json(root/'metrics/test.json',canonical)
base.atomic_json(root/'diagnostics/inference_audit.json',dict(passed=True,tensors_checked=len(before),
    all_tensor_hashes_before=before,all_tensor_hashes_after=after,actual_forward_calls=calls,
    no_parameter_or_buffer_updates=True,no_parameter_gradients=True,checkpoint_sha256=u['checkpoint_sha256']))
print(json.dumps({k:v for k,v in canonical.items() if k not in ('per_image','scene_summary')},indent=2))
