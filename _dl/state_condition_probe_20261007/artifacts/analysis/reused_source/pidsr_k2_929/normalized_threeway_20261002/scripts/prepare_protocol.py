"""Lock eight new runs plus one reused G6; generate paired seed schedules."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD = Path('/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/phase_objective_20261002')
G6ROOT = Path('/media/li/data3/25thstudent/DY/929work/k1_ablation_20260930/phase_objective_g6_20261002')
assert not (ROOT / 'runs').exists(), 'Do not replace any existing run'
k2 = yaml.safe_load((OLD / 'configs/phase01.yaml').read_text())
g6 = yaml.safe_load((G6ROOT / 'configs/G6_phase01.yaml').read_text())
sys.path.insert(0, k2['paths']['source_root'])
from rgb_final_confirmation.data import build_training_schedule, load_training_schedule, load_manifest, select_rows
rows = select_rows(load_manifest(k2['paths']['manifest']), 'train')
schedule_paths = {2026: Path(k2['paths']['schedule'])}
schedule_checks = {}
for seed in (2027, 2028):
    candidate = HERE / 'schedules' / f'training_schedule_{seed}.json'
    old = load_training_schedule(candidate)
    generated = build_training_schedule(rows, seed=seed, total_steps=3000, batch_size=2, crop_size=128)
    schedule_checks[str(seed)] = {'local_schedule_equals_server_regeneration': old == generated,
                                  'canonical_sha256': generated['sha256']}
    # Canonical server generation is the authority; the local schedule is retained.
    destination = HERE / 'schedules' / f'verified_server_schedule_{seed}.json'
    destination.write_text(json.dumps(generated, indent=2))
    schedule_paths[seed] = destination

(ROOT / 'configs').mkdir(parents=True, exist_ok=True)
order = [(2026, 'K1'), (2026, 'K2'), (2027, 'G6'), (2027, 'K1'),
         (2027, 'K2'), (2028, 'G6'), (2028, 'K1'), (2028, 'K2')]
official_initial = str(OLD / 'runs/phase01/seed_2026/config/tasknet_initial.pt')
for seed, arm in order:
    cfg = copy.deepcopy(g6 if arm == 'G6' else k2)
    cfg['seed'] = seed
    cfg['paths']['output_root'] = str(ROOT / 'runs' / arm / f'seed_{seed}')
    cfg['paths']['schedule'] = str(schedule_paths[seed])
    cfg['study'].update({'arm': arm, 'phase_weight': 0.1, 'selection': 'fixed_step3000',
                         'scope': 'three_paired_seeds_exploratory_validation',
                         'official_netg_reference': official_initial,
                         'effective_task_weights': [1 / 3, 2 / 3] if arm == 'K2' else [1.0]})
    if arm != 'G6':
        cfg['recurrent']['round_loss_weights'] = cfg['study']['effective_task_weights']
        cfg['study']['fixed_lambda'] = 0.030651881051471687
    else:
        cfg['study']['k2_initial_netg'] = official_initial
    (ROOT / 'configs' / f'{arm}_seed{seed}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
files = list(HERE.rglob('*.py')) + list(HERE.rglob('*.json')) + list((ROOT / 'configs').glob('*.yaml'))
reused = G6ROOT / 'runs/G6_phase01/seed_2026'
lock = {'seeds': [2026, 2027, 2028], 'new_run_order': order,
        'reused_G6_seed2026': {'root': str(reused), 'metrics_sha256': sha(reused / 'metrics/step_3000_validation.json'),
                             'checkpoint_sha256': sha(reused / 'checkpoints/step_3000.pt')},
        'steps': 3000, 'phase_weight': 0.1, 'task_weight_sum': 1.0,
        'K1_K2_alignment_lambda': 0.030651881051471687,
        'schedule_regeneration': schedule_checks,
        'schedule_file_sha256': {str(s): sha(p) for s, p in schedule_paths.items()},
        'locked_file_sha256': {str(p): sha(p) for p in files},
        'scope': 'Equal updates, not equal FLOPs; G6-vs-joint is whole-scheme comparison.',
        'limits': ['Three seeds are preliminary; no population significance/equivalence claim.',
                   'Validation has been used for exploration; no independent confirmation.',
                   'Normalized K2 differs from historical weights [0.5,1]; historical run is retained.',
                   'K1 and K2 use the same trainable trunk, joint clipping, initial weights and alignment.',
                   'Seed2026 exploration informed this design; new seeds are not an independent data split.']}
(ROOT / 'protocol_lock.json').write_text(json.dumps(lock, indent=2))
print(json.dumps({k: v for k, v in lock.items() if k != 'locked_file_sha256'}, indent=2))
