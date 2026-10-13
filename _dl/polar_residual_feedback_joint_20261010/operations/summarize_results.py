"""CPU-only interpretation of already verified, immutable evaluation records."""
from pathlib import Path
from collections import Counter
import csv
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

receipt = read(ROOT / 'archive_receipt.json')
assert receipt['all_hashes_match'] and receipt['files_verified'] == 159
manifest = {x['path']: x for x in read(ART / 'archive_manifest.json')['files']}
cfg = read(ART / 'config/locked_run.json')
selected = read(ART / 'downstream/selection.json')
assert selected['step'] == cfg['downstream']['steps'] == 12000
assert selected['test_used'] is False
assert manifest['downstream/checkpoints/final.pt']['sha256'] == selected['sha256'] == 'ec7f0e6e79d0dcf21c536dc71958d44514bd7e78a8624bb936f16c02c37e81e8'
for name, digest in cfg['task_initialization']['official_sha256'].items():
    assert manifest['weights/source_polarfree/' + name]['sha256'] == digest
assert manifest['pretraining/checkpoints/selected.pt']['sha256'] == 'a4ecebacba5cfc25546320974a9c37f1cc220c685f1318b3cc290ae4a68e9554'
runtime = read(ART / 'diagnostics/evaluation_recovery_20261010/recovery_audit.json')
snapshots = {x['frozen_path']: x['archived_copy'] for x in runtime['runtime_source_snapshots']}
for name, digest in cfg['code_sha256'].items():
    assert manifest[snapshots.get(name, name)]['sha256'] == digest, name

summary = dict(protocol_id=cfg['protocol_id'], checkpoint_step=12000,
               checkpoint_sha256=selected['sha256'], gpu_used_for_this_analysis=False,
               archive_files_verified=receipt['files_verified'], source_weights_match=True,
               frozen_code_match=True, frozen_runtime_sources_preserved=True, splits={})
with (ART / 'config/manifest.csv').open(encoding='utf-8', newline='') as stream:
    training_ids = {r['sample_id'] for r in csv.DictReader(stream) if r['split'] == 'train'}
with (ART / 'downstream/logs/train.csv').open(encoding='utf-8', newline='') as stream:
    training_rows = list(csv.DictReader(stream))
assert [int(r['step']) for r in training_rows] == list(range(1, 12001))
seen = Counter(sample for r in training_rows for sample in r['sample_ids'].split(';'))
assert set(seen) == training_ids and len(training_ids) == 5440 and sum(seen.values()) == 24000
for r in training_rows:
    assert all(math.isfinite(float(r[k])) for k in ['loss', 'task', 'alignment', 'quality', 'grad_norm'])
    assert float(r['task_lr']) <= 1e-6 + 1e-14
    if int(r['step']) <= 500:
        assert float(r['correction_lr']) == float(r['task_lr']) == 0 and r['tasknet_trainable'] == 'False'
    else:
        assert abs(float(r['correction_lr']) - 2e-5) < 1e-14 and r['tasknet_trainable'] == 'True'
    assert abs(float(r['interfaces_lr']) - 1e-4) < 1e-14
summary['training'] = dict(steps=12000, all_rows_finite=True, unique_images=len(seen),
    sampled_crops=sum(seen.values()), min_visits=min(seen.values()), max_visits=max(seen.values()),
    actual_logged_frozen_stage_and_learning_rates_match=True)
for split, expected in [('validation', 872), ('test', 188)]:
    x = read(ART / f'downstream/metrics/final_{split}.json')
    rows, scenes = x['per_image'], x['per_scene']
    assert x['frames'] == len(rows) == len({r['sample_id'] for r in rows}) == expected
    assert x['checkpoint_sha256'] == selected['sha256'] and x['protocol_id'] == cfg['protocol_id']
    assert x['original_size'] and not x['tiling'] and not x['test_used_for_selection']
    for key in ['round1', 'round2', 'iteration_gain']:
        for metric in ['psnr', 'ssim']:
            values = [r[key][metric] for r in rows]
            assert all(math.isfinite(v) for v in values)
            assert abs(statistics.mean(values) - x[key]['image_mean_' + metric]) < 1e-10
    psnr = [r['iteration_gain']['psnr'] for r in rows]
    ssim = [r['iteration_gain']['ssim'] for r in rows]
    confidence = {k: v for k, v in x['confidence'].items() if k != 'per_image'}
    summary['splits'][split] = dict(frames=expected, round1=x['round1'], round2=x['round2'],
        iteration_gain=x['iteration_gain'], psnr_improved=sum(v > 0 for v in psnr),
        psnr_worsened=sum(v < 0 for v in psnr), ssim_improved=sum(v > 0 for v in ssim),
        both_improved=sum(a > 0 and b > 0 for a, b in zip(psnr, ssim)),
        median_psnr_gain=statistics.median(psnr), max_psnr_gain=max(psnr), min_psnr_gain=min(psnr),
        scenes=len(scenes), psnr_scenes_improved=sum(r['iteration_gain']['psnr'] > 0 for r in scenes),
        per_scene=scenes, confidence=confidence)
mechanism = read(ART / 'downstream/metrics/mechanism_validation.json')
assert mechanism['frames'] == len(mechanism['per_image']) == 28 and not mechanism['test_used']
summary['mechanism'] = {k: v for k, v in mechanism.items() if k != 'per_image'}
for name, row in summary['mechanism']['means'].items():
    row['psnr_change_from_normal'] = row['psnr'] - summary['mechanism']['means']['normal']['psnr']
initial = read(ART / 'downstream/metrics/retention_step_0.json')
final = read(ART / 'downstream/metrics/retention_step_12000.json')
assert [r['sample_id'] for r in initial['per_image']] == [r['sample_id'] for r in final['per_image']]
summary['retention'] = {k: v for k, v in final.items() if k != 'per_image'}
summary['retention']['images_psnr_improved'] = sum(b['psnr'] > a['psnr'] for a, b in zip(initial['per_image'], final['per_image']))
summary['system_probe'] = {k: v for k, v in read(ART / 'downstream/metrics/system_step_12000.json').items() if k != 'per_image'}
summary['evaluation_completed_unix'] = read(ART / 'status.json')['time_unix']
(ROOT / 'interpretation/result_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: {q: v[q] for q in ['psnr_improved', 'psnr_worsened', 'ssim_improved', 'both_improved', 'psnr_scenes_improved']} for k, v in summary['splits'].items()}))
print(json.dumps(dict(source_weights_match=True, frozen_code_match=True, retention_psnr_improved=summary['retention']['images_psnr_improved'], training=summary['training'])))
