"""Bind completed results to checkpoints, compare paired images and retain negatives."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
K1 = ROOT.parent
K2 = Path('/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/phase_objective_20261002')
RUN = ROOT / 'runs/G6_phase01/seed_2026'
OUT = ROOT / 'analysis'
OUT.mkdir(parents=True, exist_ok=True)
REF = OUT / 'references'
REF.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())

paths = {'G6_phase01': RUN / 'metrics/step_3000_validation.json',
         'G6_phase1': K1 / 'runs/G6_netg_only_bypass_r2p/seed_2026/metrics/step_3000_validation.json',
         'G0_official': K1 / 'runs/G0_official/seed_2026/metrics/step_0_validation.json',
         'K2_phase01': K2 / 'runs/phase01/seed_2026/metrics/step_3000_validation.json',
         'K2_phase1': K2 / 'runs/phase1/seed_2026/metrics/step_3000_validation.json'}
data = {n: read(p) for n, p in paths.items()}
ids = [(r['sample_id'], r['scene']) for r in data['G6_phase01']['per_image']]
assert len(ids) == len(set(ids)) == 872
assert len({s for _, s in ids}) == 8
for n, d in data.items():
    assert [(r['sample_id'], r['scene']) for r in d['per_image']] == ids, n
    if n != 'G6_phase01':
        shutil.copy2(paths[n], REF / (n + '_validation.json'))

checkpoint = RUN / 'checkpoints/step_3000.pt'
checkpoint_sha = sha(checkpoint)
assert data['G6_phase01']['checkpoint_sha256'] == checkpoint_sha
loss = read(RUN / 'metrics/step_3000_validation_loss_parts.json')
assert loss['checkpoint_sha256'] == checkpoint_sha and loss['frames'] == 872
assert [(r['sample_id'], r['scene']) for r in loss['per_image']] == ids
for n in ('K2_phase01', 'K2_phase1'):
    assert data[n]['checkpoint_sha256'] == sha(data[n]['checkpoint'])

models = {}
def collect(name, d, rnd=None):
    models[name] = [dict(sample_id=r['sample_id'], scene=r['scene'],
                         psnr=(r[rnd] if rnd else r)['psnr'],
                         ssim=(r[rnd] if rnd else r)['ssim']) for r in d['per_image']]

for n, d in data.items():
    if n.startswith('K2'):
        collect(n + '_T1', d, 'round1')
        collect(n + '_T2', d, 'round2')
    else:
        collect(n, d)

scenes = sorted({s for _, s in ids})
def aggregate(rows):
    return {'image_mean_' + m: float(np.mean([r[m] for r in rows])) for m in ('psnr', 'ssim')} | {
            'scene_macro_' + m: float(np.mean([np.mean([r[m] for r in rows if r['scene'] == s]) for s in scenes]))
            for m in ('psnr', 'ssim')}

tables = {n: aggregate(r) for n, r in models.items()}
for n, d in data.items():
    for rnd in (('round1', 'round2') if n.startswith('K2') else (None,)):
        table = tables[n + ('_T1' if rnd == 'round1' else '_T2') if rnd else n]
        official = d[rnd] if rnd else d
        assert all(abs(table[k] - official[k]) < 1e-10 for k in table), (n, rnd)

pairs = [('G6_phase01', 'G6_phase1'), ('G6_phase01', 'G0_official'),
         ('K2_phase01_T1', 'G6_phase01'), ('K2_phase01_T2', 'G6_phase01'),
         ('K2_phase01_T2', 'K2_phase01_T1'), ('K2_phase1_T1', 'G6_phase1'),
         ('K2_phase1_T2', 'G6_phase1')]
comparisons = {}
for a, b in pairs:
    rows = [dict(sample_id=x['sample_id'], scene=x['scene'],
                 psnr=x['psnr'] - y['psnr'], ssim=x['ssim'] - y['ssim'])
            for x, y in zip(models[a], models[b])]
    scene_rows = {s: {'n': sum(r['scene'] == s for r in rows),
                     **{m: float(np.mean([r[m] for r in rows if r['scene'] == s])) for m in ('psnr', 'ssim')}} for s in scenes}
    comparisons[a + '_minus_' + b] = {
        'aggregate': aggregate(rows), 'per_scene': scene_rows,
        'positive_images': {m: sum(r[m] > 0 for r in rows) for m in ('psnr', 'ssim')},
        'negative_images': {m: sum(r[m] < 0 for r in rows) for m in ('psnr', 'ssim')},
        'positive_scenes': {m: sum(r[m] > 0 for r in scene_rows.values()) for m in ('psnr', 'ssim')},
        'difficulty_posthoc': {difficulty: {'n': sum('/' + difficulty + '/' in r['sample_id'] for r in rows),
                                          **{m: float(np.mean([r[m] for r in rows if '/' + difficulty + '/' in r['sample_id']])) for m in ('psnr', 'ssim')}}
                              for difficulty in ('easy', 'hard')},
        'per_image': rows}

new_config = yaml.safe_load((ROOT / 'configs/G6_phase01.yaml').read_text())
k2_config = yaml.safe_load((K2 / 'configs/phase01.yaml').read_text())
shared = {k: new_config[k] == k2_config[k] for k in ('seed', 'data', 'optimization', 'model')}
shared.update({k: new_config['paths'][k] == k2_config['paths'][k]
               for k in ('base_config', 'source_root', 'manifest', 'schedule', 'validation_proxy', 'pretrained_r2p', 'reference_r2p')})
assert all(shared.values())
logs = list(csv.DictReader((RUN / 'logs/train.csv').open()))
assert [int(r['step']) for r in logs] == list(range(1, 3001))
seed_matches = {}
for n, log_path in {'K2_phase01': K2 / 'runs/phase01/seed_2026/logs/train.csv',
                    'G6_phase1': K1 / 'runs/G6_netg_only_bypass_r2p/seed_2026/logs/train.csv'}.items():
    reference = list(csv.DictReader(log_path.open()))
    assert len(reference) == 3000
    matches = sum((r['step'], r['diffusion_seed']) == (s['step'], s['diffusion_seed']) for r, s in zip(logs, reference))
    seed_matches[n] = matches
    assert matches == 3000

frozen_files = sorted((RUN / 'diagnostics').glob('frozen_step_*.json'))
audits = [read(p) for p in frozen_files]
assert sorted(a['step'] for a in audits) == list(range(200, 3001, 200))
assert all(a['passed'] for a in audits)
assert read(ROOT / 'audit_step_3000.json')['passed']
initial = read(RUN / 'diagnostics/initialization_match.json')
assert initial['passed']
lock = read(ROOT / 'protocol_lock.json')
script_unchanged = {p: sha(p) == digest for p, digest in lock['sha256'].items()}
assert all(script_unchanged.values())
state = torch.load(checkpoint, map_location='cpu', weights_only=False)
assert state['step'] == 3000 and state['phase_weight'] == 0.1
assert state['lambda_feature_alignment'] == 0
assert all(torch.isfinite(v).all() for v in state['tasknet_net_g'].values())
checkpoint_audit = {'path': str(checkpoint), 'sha256': checkpoint_sha, 'step': 3000,
                    'netg_tensors': len(state['tasknet_net_g']),
                    'frozen_save_points': len(audits), 'checked_tensors_per_point': audits[0]['checked_tensors'],
                    'passed': True}
final = {'scope': 'single_seed_exploratory_full872_validation', 'frames': 872, 'scenes': 8,
         'tables': tables, 'comparisons': comparisons, 'checkpoint_audit': checkpoint_audit,
         'initialization_match': initial, 'shared_settings_match': shared,
         'diffusion_seed_matches_3000_steps': seed_matches, 'locked_files_unchanged': script_unchanged,
         'loss_parts_G6_phase01': {k: loss[k] for k in ('image_mean', 'scene_macro')},
         'reference_sources': {n: {'path': str(p), 'sha256': sha(p)} for n, p in paths.items()},
         'timing': [read(p) for p in sorted((RUN / 'diagnostics').glob('timing_*.json'))],
         'limitations': lock['limits']}
(OUT / 'matched_g6_summary.json').write_text(json.dumps(final, indent=2))
print(json.dumps({k: v for k, v in final.items() if k not in ('comparisons', 'locked_files_unchanged')}, indent=2))
print(json.dumps({n: {k: v for k, v in c.items() if k != 'per_image'} for n, c in comparisons.items()}, indent=2))
