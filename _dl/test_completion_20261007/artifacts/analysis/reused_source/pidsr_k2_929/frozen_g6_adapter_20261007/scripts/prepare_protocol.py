"""Bind frozen trained controls and a fixed, additional 3000-step adapter budget."""
import copy
import hashlib
import json
from pathlib import Path
import platform
import torch
import yaml

ROOT=Path(__file__).resolve().parent.parent
OLD=ROOT.parent/'normalized_threeway_20261002'
PREV=ROOT.parent/'k1_no_alignment_20261003'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert not (ROOT/'runs').exists() and not (ROOT/'protocol_lock.json').exists()
for study in (OLD,PREV):
    assert read(study/'status.json')['state']=='complete'
    assert all(sha(p)==h for p,h in read(study/'protocol_lock.json')['locked_file_sha256'].items())
old_lock=read(OLD/'protocol_lock.json')
baseline=read(OLD/'analysis/threeway_paired_summary.json')
(ROOT/'configs').mkdir(exist_ok=True)
refs={}
for seed in (2026,2027,2028):
    run=Path(old_lock['reused_G6_seed2026']['root']) if seed==2026 else OLD/'runs/G6'/f'seed_{seed}'
    paths={'checkpoint':run/'checkpoints/step_3000.pt','metrics':run/'metrics/step_3000_validation.json','training_log':run/'logs/train.csv'}
    fp=baseline['source_fingerprints'][f'G6_seed{seed}']
    assert sha(paths['checkpoint'])==fp['checkpoint_sha256'] and sha(paths['metrics'])==fp['metrics_sha256']
    refs[f'G6_seed{seed}']={k:{'path':str(v),'sha256':sha(v)} for k,v in paths.items()}
    cfg=copy.deepcopy(yaml.safe_load((PREV/'configs'/f'K1_seed{seed}.yaml').read_text()))
    cfg['paths']['output_root']=str(ROOT/'runs/K1'/f'seed_{seed}')
    cfg['study'].pop('aligned_initial_audit',None)
    cfg['study'].update(experiment='frozen_G6_identity_adapter',scope='staged_frozen_trained_G6_adapter_feasibility',
        frozen_G6_checkpoint=str(paths['checkpoint']),frozen_G6_checkpoint_sha256=sha(paths['checkpoint']),
        frozen_G6_metrics=str(paths['metrics']),expected_proxy_images=28)
    (ROOT/'configs'/f'K1_seed{seed}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
files=list((ROOT/'scripts').rglob('*.py'))+list((ROOT/'configs').glob('*.yaml'))
for seed in (2026,2027,2028):
    cfg=yaml.safe_load((ROOT/'configs'/f'K1_seed{seed}.yaml').read_text())
    files += [Path(cfg['paths'][k]) for k in ('base_config','manifest','schedule','validation_proxy','pretrained_r2p','reference_r2p')]
    files.append(Path(cfg['study']['official_netg_reference']))
    source=Path(cfg['paths']['source_root'])/'rgb_final_confirmation'
    files += [source/n for n in ('data.py','polarfree.py','losses.py','evaluation.py','config.py')]
    base=yaml.safe_load(Path(cfg['paths']['base_config']).read_text())
    files += list(Path(base['official']['polarapp_dfp_root']).rglob('*.py'))
runtime={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
    'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(),
    'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32}
assert runtime['matmul_tf32'] is False and runtime['cudnn_tf32'] is True
lock={'seeds':[2026,2027,2028],'new_run_order':[[s,'K1'] for s in (2026,2027,2028)],
    'experiment':'frozen_G6_identity_adapter','additional_adapter_steps':3000,'source_G6_steps':3000,
    'phase_weight':0.1,'alignment_lambda':0.0,'evaluation_seed':2026,'task_weights':[1.0],
    'initialization':'e42 pretrained R2P trunk; residual output Conv2d weight+bias reset to zero',
    'frozen':'all TaskNet parameters and buffers, projections and feedback; only 18 R2P tensors optimized',
    'selection':'fixed step3000; validation proxy not used to select checkpoint',
    'references':refs,'runtime':runtime,'locked_file_sha256':{str(p):sha(p) for p in sorted(set(files))},
    'limits':['Exploratory validation, previously inspected; no new independent test.',
        'Three seeds pair three different trained G6 controls with adapters; adapter-only variability is not isolated.',
        'Extra 3000 adapter updates reuse the training schedule; source G6 already trained 3000 steps.',
        'Incremental frozen-model comparison, not equal-total-budget ranking against previous joint runs.',
        'Zero output head changes pretrained R2P initialization; not a one-factor ablation of the old joint model.',
        'Historical 872-image G6 metrics reused after 28 full-resolution proxy tensor/RNG/metric replays per seed.',
        'K1 directory/log round2 fields are inherited aliases; exactly one TaskNet forward per evaluation image.',
        'Fixed evaluation RNG seed 2026; inference sampling variability not estimated.']}
(ROOT/'protocol_lock.json').write_text(json.dumps(lock,indent=2))
print(json.dumps({'runtime':runtime,'locked_files':len(lock['locked_file_sha256']),'references':refs},indent=2))
