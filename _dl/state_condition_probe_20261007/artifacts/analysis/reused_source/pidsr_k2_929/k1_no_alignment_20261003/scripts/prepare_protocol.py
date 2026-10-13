"""Preregister three paired K1 lambda=0 runs against the completed lambda>0 controls."""
import copy
import hashlib
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT.parent / 'normalized_threeway_20261002'
assert not (ROOT / 'runs').exists() and not (ROOT / 'protocol_lock.json').exists()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
old_lock = json.loads((OLD / 'protocol_lock.json').read_text())
assert json.loads((OLD / 'status.json').read_text())['state'] == 'complete'
assert all(sha(p) == h for p, h in old_lock['locked_file_sha256'].items())
baseline_summary = json.loads((OLD / 'analysis/threeway_paired_summary.json').read_text())
(ROOT / 'configs').mkdir(parents=True, exist_ok=True)
refs = {}
config_diffs = {}
for seed in (2026,2027,2028):
    src = OLD / 'configs' / f'K1_seed{seed}.yaml'
    old = yaml.safe_load(src.read_text())
    new = copy.deepcopy(old)
    new['paths']['output_root'] = str(ROOT / 'runs/K1' / f'seed_{seed}')
    new['study']['fixed_lambda'] = 0.0
    new['study']['scope'] = 'one_factor_alignment_ablation_three_paired_training_seeds'
    new['study']['aligned_initial_audit'] = str(OLD / 'runs/K1' / f'seed_{seed}/diagnostics/initial_pair_audit.json')
    target = ROOT / 'configs' / f'K1_seed{seed}.yaml'
    target.write_text(yaml.safe_dump(new, sort_keys=False))
    def differences(a,b,prefix=''):
        result={}
        for k in a.keys() | b.keys():
            name = prefix+k
            if isinstance(a.get(k),dict) and isinstance(b.get(k),dict):
                result.update(differences(a[k],b[k],name+'.'))
            elif a.get(k) != b.get(k):
                result[name]={'old':a.get(k),'new':b.get(k)}
        return result
    diffs=differences(old,new)
    assert set(diffs)=={'paths.output_root','study.fixed_lambda','study.scope','study.aligned_initial_audit'}
    config_diffs[str(seed)]=diffs
    for arm in ('K1','G6'):
        run = (Path(old_lock['reused_G6_seed2026']['root']) if (arm,seed)==('G6',2026)
               else OLD / 'runs' / arm / f'seed_{seed}')
        paths = {'metrics':run/'metrics/step_3000_validation.json',
                 'checkpoint':run/'checkpoints/step_3000.pt','training_log':run/'logs/train.csv'}
        if arm=='K1':
            paths.update(initial_audit=run/'diagnostics/initial_pair_audit.json',config=src)
        fp = baseline_summary['source_fingerprints'][f'{arm}_seed{seed}']
        assert sha(paths['metrics'])==fp['metrics_sha256'] and sha(paths['checkpoint'])==fp['checkpoint_sha256']
        refs[f'{arm}_seed{seed}']={k:{'path':str(v),'sha256':sha(v)} for k,v in paths.items()}

files=list((ROOT/'scripts').rglob('*.py'))+list((ROOT/'configs').glob('*.yaml'))
# Also bind existing data/init/runtime configuration; these are read-only shared inputs.
for seed in (2026,2027,2028):
    cfg=yaml.safe_load((ROOT/'configs'/f'K1_seed{seed}.yaml').read_text())
    files += [Path(cfg['paths'][k]) for k in ('base_config','manifest','schedule','validation_proxy','pretrained_r2p','reference_r2p')]
    files.append(Path(cfg['study']['official_netg_reference']))
lock={'seeds':[2026,2027,2028],'new_run_order':[[s,'K1'] for s in (2026,2027,2028)],
      'steps':3000,'phase_weight':0.1,'evaluation_seed':2026,'task_weight_sum':1.0,
      'K1_K2_alignment_lambda':0.0,'control_alignment_lambda':0.030651881051471687,
      'config_differences':config_diffs,'references':refs,
      'locked_file_sha256':{str(p):sha(p) for p in sorted(set(files))},
      'limits':['Three training seeds; same previously explored validation; no independent confirmation.',
                'Lambda=0 disables alignment supervision, retaining model, optimizer groups and diagnostic alignment graph.',
                'Projection zero gradients and unchanged tensors are explicitly audited; Adam state remains present.',
                'No claim that removing alignment isolates R2P architecture from all other factors.',
                'No additional test, feedback, second-round structure or freeze-G6 adaptation launched.']}
(ROOT/'protocol_lock.json').write_text(json.dumps(lock,indent=2))
print(json.dumps({'config_differences':config_diffs,'locked_files':len(lock['locked_file_sha256']),'references':list(refs)},indent=2))
