"""Audit all nine results and summarize paired seed differences without pseudoreplication."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
lock = read(ROOT / 'protocol_lock.json')
assert read(ROOT / 'status.json')['state'] == 'complete'
assert all(sha(p) == h for p, h in lock['locked_file_sha256'].items())
assert lock['evaluation_seed'] == 2026
OUT = ROOT / 'analysis'
OUT.mkdir(parents=True, exist_ok=True)
REF = OUT / 'references'
REF.mkdir(exist_ok=True)
seeds = [2026, 2027, 2028]
data, tables, checkpoint_audits, source_fingerprints = {}, {}, {}, {}
global_ids = None
for seed in seeds:
    pair_logs = {}
    initials = {}
    for arm in ('G6', 'K1', 'K2'):
        run = Path(lock['reused_G6_seed2026']['root']) if (arm, seed) == ('G6', 2026) else ROOT / 'runs' / arm / f'seed_{seed}'
        path = run / 'metrics/step_3000_validation.json'
        d = read(path)
        assert d['checkpoint_step'] == 3000 and d['phase_weight'] == 0.1
        assert int(d['frames']) == 872 and int(d['scenes']) == 8
        ck_sha = sha(run / 'checkpoints/step_3000.pt')
        assert ck_sha == d['checkpoint_sha256']
        ids = [(r['sample_id'], r['scene']) for r in d['per_image']]
        assert len(ids) == len(set(ids)) == 872
        if global_ids is None:
            global_ids = ids
        assert ids == global_ids
        if (arm, seed) == ('G6', 2026):
            assert ck_sha == lock['reused_G6_seed2026']['checkpoint_sha256']
            assert sha(path) == lock['reused_G6_seed2026']['metrics_sha256']
            shutil.copy2(path, REF / 'G6_seed2026_reused_validation.json')
        else:
            assert d['training_seed'] == seed and d['evaluation_seed'] == 2026
            assert d['actual_tasknet_forwards'] == (1744 if arm == 'K2' else 872)
            assert d['effective_task_weights'] == ([1 / 3, 2 / 3] if arm == 'K2' else [1.0])
            assert read(run / 'diagnostics/gate_step_3000.json')['passed']
            frozen = [read(p) for p in (run / 'diagnostics').glob('frozen_step_*.json')]
            assert all(d['passed'] for d in frozen)
            assert set(range(200, 3001, 200)) <= {d['step'] for d in frozen}
            initials[arm] = read(run / 'diagnostics/initial_pair_audit.json')
        log_rows = list(csv.DictReader((run / 'logs/train.csv').open()))
        assert [int(r['step']) for r in log_rows] == list(range(1, 3001))
        assert all(np.isfinite(float(v)) for r in log_rows for v in r.values())
        pair_logs[arm] = [(r['step'], r['diffusion_seed']) for r in log_rows]
        source_fingerprints[f'{arm}_seed{seed}'] = {'metrics_path': str(path), 'metrics_sha256': sha(path), 'checkpoint_sha256': ck_sha}
        checkpoint_audits[f'{arm}_seed{seed}'] = {'passed': True, 'continuous_steps': 3000, 'checkpoint_sha256': ck_sha}
        data[(arm, seed)] = d
    assert pair_logs['G6'] == pair_logs['K1'] == pair_logs['K2']
    assert initials['K1']['model'] == initials['K2']['model']
    assert initials['K1']['netg'] == initials['K2']['netg']
    if seed != 2026:
        assert initials['G6']['netg'] == initials['K1']['netg']

g0_path = Path('/media/li/data3/25thstudent/DY/929work/k1_ablation_20260930/runs/G0_official/seed_2026/metrics/step_0_validation.json')
g0 = read(g0_path)
assert [(r['sample_id'], r['scene']) for r in g0['per_image']] == global_ids
shutil.copy2(g0_path, REF / 'G0_fixed_official_validation.json')
source_fingerprints['G0_fixed_official'] = {'metrics_path': str(g0_path), 'metrics_sha256': sha(g0_path), 'not_three_independent_trained_models': True}
for seed in seeds:
    data[('G0', seed)] = g0

scenes = sorted({s for _, s in global_ids})
def aggregate(rows):
    return {f'image_mean_{m}': float(np.mean([r[m] for r in rows])) for m in ('psnr', 'ssim')} | {
            f'scene_macro_{m}': float(np.mean([np.mean([r[m] for r in rows if r['scene'] == s]) for s in scenes])) for m in ('psnr', 'ssim')}

outputs = {}
for (arm, seed), d in data.items():
    for rnd in (('round1', 'round2') if arm == 'K2' else (None,)):
        name = arm + ('_T1' if rnd == 'round1' else '_T2') if rnd else arm
        rows = [dict(sample_id=r['sample_id'], scene=r['scene'], **(r[rnd] if rnd else {m: r[m] for m in ('psnr', 'ssim')})) for r in d['per_image']]
        outputs[(name, seed)] = rows
        tables[(name, seed)] = aggregate(rows)
        official = d[rnd] if rnd else d
        assert all(abs(tables[(name, seed)][k] - official[k]) < 1e-10 for k in tables[(name, seed)])

def seed_stats(values):
    return {'values': values, 'mean': float(np.mean(values)), 'sample_sd': float(np.std(values, ddof=1)),
            'min': min(values), 'max': max(values), 'positive_seeds': sum(v > 0 for v in values)}

pairs = [('K1', 'G6'), ('K2_T2', 'K1'), ('K2_T1', 'K1'), ('K2_T2', 'G6'), ('K2_T2', 'K2_T1'),
         ('G6', 'G0'), ('K1', 'G0'), ('K2_T2', 'G0')]
comparisons = {}
for a, b in pairs:
    per_seed = {}
    for seed in seeds:
        rows = [dict(sample_id=x['sample_id'], scene=x['scene'], **{m: x[m] - y[m] for m in ('psnr', 'ssim')})
                for x, y in zip(outputs[(a, seed)], outputs[(b, seed)])]
        per_seed[str(seed)] = {'aggregate': aggregate(rows),
            'per_scene': {s: {'n': sum(r['scene'] == s for r in rows), **{m: float(np.mean([r[m] for r in rows if r['scene'] == s])) for m in ('psnr', 'ssim')}} for s in scenes},
            'positive_images': {m: sum(r[m] > 0 for r in rows) for m in ('psnr', 'ssim')},
            'negative_images': {m: sum(r[m] < 0 for r in rows) for m in ('psnr', 'ssim')},
            'difficulty_posthoc': {level: {'n': sum('/' + level + '/' in r['sample_id'] for r in rows), **{m: float(np.mean([r[m] for r in rows if '/' + level + '/' in r['sample_id']])) for m in ('psnr', 'ssim')}} for level in ('easy', 'hard')},
            'per_image': rows}
    comparisons[a + '_minus_' + b] = {'per_seed': per_seed,
        'paired_seed_summary': {k: seed_stats([per_seed[str(s)]['aggregate'][k] for s in seeds]) for k in per_seed['2026']['aggregate']},
        'scene_seed_summary': {scene: {m: seed_stats([per_seed[str(s)]['per_scene'][scene][m] for s in seeds]) for m in ('psnr', 'ssim')} for scene in scenes}}

old_k2_path = Path('/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/phase_objective_20261002/runs/phase01/seed_2026/metrics/step_3000_validation.json')
old_k2 = read(old_k2_path)
assert [(r['sample_id'], r['scene']) for r in old_k2['per_image']] == global_ids
assert sha(old_k2['checkpoint']) == old_k2['checkpoint_sha256']
shutil.copy2(old_k2_path, REF / 'K2_seed2026_original_weights_05_1.json')
sensitivity = {rnd: {k: tables[('K2_T1' if rnd == 'round1' else 'K2_T2', 2026)][k] - old_k2[rnd][k]
                    for k in old_k2[rnd]} for rnd in ('round1', 'round2')}

payload = {'scope': '3 paired training seeds; common evaluation_seed2026; exploratory full872 validation',
           'seeds': seeds, 'frames': 872, 'scenes': 8,
           'per_seed_tables': {f'{name}_seed{seed}': table for (name, seed), table in tables.items()},
           'model_seed_summary': {name: {k: seed_stats([tables[(name, s)][k] for s in seeds]) for k in tables[(name, 2026)]} for name in ('G6', 'K1', 'K2_T1', 'K2_T2')},
           'comparisons': comparisons, 'checkpoint_audits': checkpoint_audits,
           'source_fingerprints': source_fingerprints, 'protocol_file_sha256': sha(ROOT / 'protocol_lock.json'),
           'all_paired_initializations_and_diffusion_seeds_matched': True,
           'G0_fixed_table': tables[('G0', 2026)],
           'K2_seed2026_normalization_minus_historical_05_1': sensitivity,
           'historical_K2_source': {'path': str(old_k2_path), 'sha256': sha(old_k2_path)},
           'limitations': lock['limits']}
(OUT / 'threeway_paired_summary.json').write_text(json.dumps(payload, indent=2))
print(json.dumps({'per_seed_tables': payload['per_seed_tables'],
                  'paired_seed_differences': {n: c['paired_seed_summary'] for n, c in comparisons.items()}}, indent=2))
