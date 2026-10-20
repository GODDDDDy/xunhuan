"""CPU regression checks for the two storage/recovery implementation repairs."""
import copy,hashlib,importlib.util,json,os,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('repaired43',ROOT/'scripts/run_study.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
out={}
for cloned in [False,True]:
 p=torch.nn.Parameter(torch.tensor([1.,2.]));opt=torch.optim.Adam([p],lr=1e-5);p.grad=torch.ones_like(p);opt.step();saved=copy.deepcopy(opt.state_dict());before=copy.deepcopy(saved)
 q=torch.nn.Parameter(p.detach().clone());other=torch.optim.Adam([q],lr=1e-5);other.load_state_dict(copy.deepcopy(saved) if cloned else saved);q.grad=torch.ones_like(q);other.step()
 unchanged=s.equal(saved,before);out['deepcopy' if cloned else 'old_shallow']=dict(saved_state_unchanged=unchanged,saved_step=int(saved['state'][0]['step']))
 assert unchanged==cloned
scratch=ROOT/'diagnostics'/('cpu_recovery_test_'+str(time.time_ns()));scratch.mkdir();s.ROOT=scratch;d=scratch/'weights_resumed';d.mkdir();last=d/'last.pt'
v=dict(step=300,global_step=6300,protocol_id='CPU_TEST',tensor=torch.tensor([1.,2.]));s.save(last,v);first=last.resolve();original=s.sha(first)
v=dict(step=400,global_step=6400,protocol_id='CPU_TEST',tensor=torch.tensor([3.,4.]));s.save(last,v);second=last.resolve();s.save(d/'step_00400.pt',v)
assert first!=second and s.sha(first)==original and len(list(d.glob('state_*.pt')))==2
assert os.stat(second).st_ino==os.stat(d/'step_00400.pt').st_ino
assert torch.equal(torch.load(last,weights_only=False)['tensor'],v['tensor'])
out.update(passed=True,immutable_old_payload_preserved=True,atomic_latest_symlink=True,milestone_is_same_immutable_inode=True,time_unix=time.time())
(ROOT/'reports/recovery_cpu_check.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
