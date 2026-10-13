"""Lock G6 phase0.1 before training; check shared settings against completed K2."""
import copy
import hashlib
import json
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
K2 = Path('/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/phase_objective_20261002')
original = yaml.safe_load((HERE / 'reference/G6_original.yaml').read_text())
k2 = yaml.safe_load((K2 / 'configs/phase01.yaml').read_text())
cfg = copy.deepcopy(original)
cfg['description'] = 'G6 genuine R2P bypass; phase coefficient 0.1; official initialization, seed2026, step3000.'
cfg['paths']['output_root'] = str(ROOT / 'runs/G6_phase01/seed_2026')
cfg['ablation']['group'] = 'G6_phase01'
cfg['study'] = {'phase_weight': 0.1, 'selection': 'fixed_step3000',
                'scope': 'single_seed_exploratory_validation_only',
                'k2_initial_netg': str(K2 / 'runs/phase01/seed_2026/config/tasknet_initial.pt')}
checks = {}
for key in ('seed', 'data', 'optimization', 'model'):
    checks[key] = cfg[key] == k2[key]
for key in ('manifest', 'schedule', 'base_config', 'source_root', 'reference_r2p', 'pretrained_r2p'):
    checks['path_' + key] = cfg['paths'][key] == k2['paths'][key]
assert all(checks.values()), checks
assert not (ROOT / 'runs').exists(), 'Never replace a started run'
(ROOT / 'configs').mkdir(parents=True, exist_ok=True)
config_path = ROOT / 'configs/G6_phase01.yaml'
config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
hash_file = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
files = list(HERE.glob('*.py')) + list((HERE / 'reference').glob('*')) + [config_path]
protocol = {'planned_arm': 'G6_phase01', 'seed': 2026, 'steps': 3000, 'phase_weight': 0.1,
            'matched_shared_fields': checks, 'sha256': {str(p): hash_file(p) for p in files},
            'k2_reference_config_sha256': hash_file(K2 / 'configs/phase01.yaml'),
            'comparison': ['new G6 phase0.1 vs historical G6 phase1 (objective change)',
                           'K2 phase0.1 round1 and round2 vs new G6 phase0.1 (whole scheme)'],
            'limits': ['one seed; validation used for exploration',
                       'equal optimizer steps do not mean equal compute',
                       'K2 also changes alignment, trainable modules and round supervision',
                       'no additive module or isolated recurrence contribution from G6 vs K2']}
(ROOT / 'protocol_lock.json').write_text(json.dumps(protocol, indent=2))
print(json.dumps(protocol, indent=2))
