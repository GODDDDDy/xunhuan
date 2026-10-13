"""Lock fifteen existing final checkpoints plus an official reference before test inference."""
import copy,csv,hashlib,json,platform
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parent.parent
P=ROOT.parent
N=P/'normalized_threeway_20261002';A=P/'k1_no_alignment_20261003';F=P/'frozen_g6_adapter_20261007'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert not (ROOT/'protocol_lock.json').exists() and not (ROOT/'units').exists()
for s in (N,A,F):
    assert read(s/'status.json')['state']=='complete'
    assert all(sha(p)==h for p,h in read(s/'protocol_lock.json')['locked_file_sha256'].items())
source=read(N/'protocol_lock.json')
g6old=Path(source['reused_G6_seed2026']['root'])
summary=read(N/'analysis/threeway_paired_summary.json')
config0=g6old.parents[2]/'configs/G6_phase01.yaml'
units=[]
def unit(name,seed,config,scripts,extension,run=None):
    cfg=yaml.safe_load(Path(config).read_text());new=copy.deepcopy(cfg)
    uid=f'{name}_seed{seed}' if name!='G0' else 'G0_official'
    new['paths']['output_root']=str(ROOT/'units'/uid)
    cp=Path(run)/'checkpoints/step_3000.pt' if run else None
    vm=Path(run)/'metrics/step_3000_validation.json' if run else N/'analysis/references/G0_fixed_official_validation.json'
    assert vm.is_file()
    target=ROOT/'configs'/f'{uid}.yaml';target.parent.mkdir(exist_ok=True)
    target.write_text(yaml.safe_dump(new,sort_keys=False))
    u=dict(id=uid,family=name,seed=seed,source_config=str(config),config=str(target),source_scripts=str(scripts),
        extension=extension,checkpoint=str(cp) if cp else None,checkpoint_sha256=sha(cp) if cp else None,
        validation_metrics=str(vm),validation_metrics_sha256=sha(vm),actual_rounds=2 if name=='K2_aligned' else 1)
    if cp:
        assert read(vm)['checkpoint_sha256']==u['checkpoint_sha256']
    units.append(u)
unit('G0',2026,config0,g6old.parents[2]/'scripts','legacy_g6')
for seed in (2026,2027,2028):
    run=g6old if seed==2026 else N/'runs/G6'/f'seed_{seed}'
    unit('G6',seed,config0 if seed==2026 else N/'configs'/f'G6_seed{seed}.yaml',
         g6old.parents[2]/'scripts' if seed==2026 else N/'scripts','legacy_g6' if seed==2026 else 'run_matched',run)
    for name,arm in [('K1_aligned','K1'),('K2_aligned','K2')]:
        unit(name,seed,N/'configs'/f'{arm}_seed{seed}.yaml',N/'scripts','run_matched',N/'runs'/arm/f'seed_{seed}')
    unit('K1_no_alignment',seed,A/'configs'/f'K1_seed{seed}.yaml',A/'scripts','run_no_alignment',A/'runs/K1'/f'seed_{seed}')
    unit('frozen_G6_adapter',seed,F/'configs'/f'K1_seed{seed}.yaml',F/'scripts','run_frozen_adapter',F/'runs/K1'/f'seed_{seed}')
assert len(units)==16 and sum(u['checkpoint'] is not None for u in units)==15
files=list((ROOT/'scripts').glob('*.py'))+list((ROOT/'configs').glob('*.yaml'))
for u in units:
    files += [Path(u['source_config']),Path(u['validation_metrics'])]
    files += list(Path(u['source_scripts']).rglob('*.py'))
    cfg=yaml.safe_load(Path(u['config']).read_text())
    files += [Path(cfg['paths'][k]) for k in ('base_config','manifest','schedule','validation_proxy','pretrained_r2p','reference_r2p')]
    bc=yaml.safe_load(Path(cfg['paths']['base_config']).read_text())
    files += list(Path(bc['official']['polarapp_dfp_root']).rglob('*.py'))
    files += [Path(bc['official']['dm_config'])]
    files += [Path(bc['official']['checkpoint_dir'])/n for n in bc['official']['expected_checkpoints']]
    files += list((Path(cfg['paths']['source_root'])/'rgb_final_confirmation').glob('*.py'))
    for k in ('official_netg_reference','k2_initial_netg','frozen_G6_checkpoint','frozen_G6_metrics','aligned_initial_audit'):
        if k in cfg['study']:files.append(Path(cfg['study'][k]))
manifest=cfg['paths']['manifest']
rows=[r for r in csv.DictReader(Path(manifest).open()) if r['split']=='test']
assert len(rows)==188 and len({r['scene'] for r in rows})==8
assert len({r['sample_id'] for r in rows})==188
data_files=sorted({r[k] for r in rows for k in ('polar_000','polar_045','polar_090','polar_135','official_rgb','gt_rgb')})
runtime=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
    gpu=torch.cuda.get_device_name(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32)
assert not runtime['matmul_tf32'] and runtime['cudnn_tf32']
lock=dict(units=units,frames=188,scenes=8,evaluation_seed=2026,checkpoint_selection='fixed step3000',
    operation='inference only; no optimizer constructed; existing weights never modified',runtime=runtime,
    locked_file_sha256={str(p):sha(p) for p in sorted(set(files))},test_sample_order=[r['sample_id'] for r in rows],
    test_data_sha256={p:sha(p) for p in data_files},
    comparison_plan=['G6-G0','K1_aligned-G6','K1_no_alignment-K1_aligned','K1_no_alignment-G6',
        'K2_aligned-K1_aligned','K2_aligned-G6','frozen_G6_adapter-G6','K2_T2-K2_T1'],
    limits=['Previously examined test set; cross-split descriptive audit, not pristine independent confirmation.',
        'No selection of seed or checkpoint from test outcomes; all fifteen planned checkpoints reported.',
        'Scene directory numbers are split-local; do not equate validation/test scenes by label.',
        'Three paired training seeds; fixed inference RNG, no inference-variance estimate.',
        'Frozen-adapter study has extra training budget and different initialization; no equal-budget ranking.',
        'Keep PSNR and SSIM, image mean and scene macro; no equivalence/significance claims without tests.',
        'Original evaluation seed function uses phase literal validation for both splits; test IDs remain in RNG key.'])
(ROOT/'protocol_lock.json').write_text(json.dumps(lock,indent=2))
print(json.dumps(dict(units=[u['id'] for u in units],runtime=runtime,locked_files=len(lock['locked_file_sha256']),test_files=len(data_files)),indent=2))
