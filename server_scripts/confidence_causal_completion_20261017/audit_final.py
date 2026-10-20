"""CPU-only independent audit of the authorized 12-hour completion experiment."""
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

# This process must never load a checkpoint onto the GPU or change training.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
PREV = ROOT.parent / 'confidence_correction_causal_20261017'
OLD = ROOT.parent / 'c_feedback_polar_loop_20261015'
CORE = ['R1N0', 'R1F', 'R1S']
REPEAT = ['R1P7', 'R1S7']
BACKGROUND = ['R1P', 'R1D', 'R1N', 'R1O', 'R0P', 'R0D', 'R0O']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def path_in_root(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_receipts(proposal):
    checked = []
    for record in proposal['source_receipts']:
        path = Path(record['path'])
        assert sha(path) == record['sha256'], str(path)
        checked.append(str(path))
    for key in ['code_manifest', 'data_manifest']:
        for record in proposal[key]:
            path = path_in_root(record['path'])
            assert sha(path) == record['sha256'], str(path)
    manifest = read(ROOT / 'artifact_manifest.json')
    for record in manifest['files']:
        path = ROOT / record['path']
        assert path.is_file(), str(path)
        assert path.stat().st_size == record['bytes'], str(path)
        assert sha(path) == record['sha256'], str(path)
    return len(checked), len(manifest['files'])


def verify_schedules(proposal, names, training):
    """Use the existing generator, including its manifest-order remapping."""
    import torch
    assert not torch.cuda.is_available(), 'Audit must remain CPU-only'
    torch.set_num_threads(2)
    sys.path.insert(0, str(OLD / 'scripts'))
    import c_loop_run as base
    config26 = read(ROOT / 'config/execution_config_2026.json')
    base.study.base.add_source_path(config26)
    from rgb_final_confirmation.data import (
        build_training_schedule, load_manifest, load_training_schedule, select_rows,
    )
    all_rows = load_manifest(config26['paths']['manifest'])
    train_rows = select_rows(all_rows, 'train')
    assert len(train_rows) == 5440
    train_ids = {row.sample_id for row in train_rows}
    ordered = sorted(train_rows, key=lambda row: row.sample_id)
    lookup = {row.sample_id: index for index, row in enumerate(train_rows)}
    seeds = [2026, 2027] if set(REPEAT).issubset(names) else [2026]
    records = {}
    schedules = {}
    for seed in seeds:
        cfg = read(ROOT / f'config/execution_config_{seed}.json')
        assert int(cfg['seed']) == seed
        assert sha(Path(cfg['paths']['manifest'])) == sha(Path(config26['paths']['manifest']))
        assert int(cfg['data']['batch_size']) == 2
        assert int(cfg['data']['crop_size']) == 128
        declaration = proposal['schedules'][str(seed)]
        schedule_path = path_in_root(declaration['path'])
        assert schedule_path.resolve() == Path(cfg['paths']['schedule']).resolve()
        assert sha(schedule_path) == declaration['sha256']
        actual = load_training_schedule(schedule_path)
        expected = build_training_schedule(ordered, seed=seed, total_steps=6000,
                                           batch_size=2, crop_size=128)
        for item in expected['items']:
            item['sample_index'] = lookup[ordered[item['sample_index']].sample_id]
        expected.pop('sha256')
        expected['sha256'] = hashlib.sha256(json.dumps(
            expected, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        assert actual == expected, f'Exact immutable schedule regeneration failed: {seed}'
        assert len(actual['items']) == 12000
        expected_ids = [train_rows[item['sample_index']].sample_id for item in actual['items']]
        expected_pairs = [';'.join(expected_ids[2*i:2*i+2]) for i in range(6000)]
        key = f'sample_ids_{seed}'
        assert [row[key] for row in training] == expected_pairs, key
        assert len(set(expected_ids)) == 5440
        if seed == 2026:
            assert actual == load_training_schedule(OLD / 'config/training6000.json')
            previous = list(csv.DictReader((PREV / 'logs/training.csv').open()))
            assert expected_pairs == [row['sample_ids'] for row in previous]
        schedules[seed] = actual
        records[str(seed)] = dict(exact_regeneration=True, sequence_verified=True,
                                 presentations_per_arm=12000, unique_train_images=5440,
                                 schedule_sha256=sha(schedule_path))
    if 2027 in schedules:
        assert schedules[2026]['items'] != schedules[2027]['items']
    calibration = read(ROOT / 'config/calibration.json')
    assert calibration['training_only'] is True and int(calibration['samples']) == 256
    values = calibration['rows']
    assert len(values) == 256
    mu = float(calibration['mu'])
    assert math.isfinite(mu) and 0 <= mu <= 1
    assert all(row['sample_id'] in train_ids and math.isfinite(float(row['c_mean']))
               and 0 <= float(row['c_mean']) <= 1 for row in values)
    assert abs(statistics.mean(float(row['c_mean']) for row in values) - mu) <= 1e-7
    calibration_order = [train_rows[item['sample_index']].sample_id
                         for item in schedules[2026]['items'][:256]]
    assert [row['sample_id'] for row in values] == calibration_order
    for index, row in enumerate(values):
        assert int(row['schedule_position']) == index
        assert all(int(row[key]) == int(schedules[2026]['items'][index][key])
                   for key in ['sample_index', 'top', 'left'])
    return records, dict(mu=mu, samples=256, training_only=True,
                         calibration_sha256=sha(ROOT / 'config/calibration.json'))


def pair_summary(rows, left, right):
    result = {}
    scenes = sorted({row['scene'] for row in rows})
    for metric in ['psnr', 'ssim']:
        deltas = [row['metrics'][left][metric] - row['metrics'][right][metric] for row in rows]
        assert all(math.isfinite(value) for value in deltas), (left, right, metric)
        by_scene = {scene: statistics.mean(
            row['metrics'][left][metric] - row['metrics'][right][metric]
            for row in rows if row['scene'] == scene) for scene in scenes}
        result[metric] = dict(scene_equal=statistics.mean(by_scene.values()),
                              image_equal=statistics.mean(deltas), scenes=by_scene,
                              scenes_positive=sum(value > 0 for value in by_scene.values()))
    return result


def same_summary(actual, expected):
    for metric in ['psnr', 'ssim']:
        for key in ['scene_equal', 'image_equal']:
            assert abs(actual[metric][key] - expected[metric][key]) <= 1e-10
        assert actual[metric]['scenes_positive'] == expected[metric]['scenes_positive']
        assert set(actual[metric]['scenes']) == set(expected[metric]['scenes'])
        for scene, value in expected[metric]['scenes'].items():
            assert abs(actual[metric]['scenes'][scene] - value) <= 1e-10


def verify_evaluation(plan, names):
    summary = read(ROOT / 'reports/summary.json')
    assert summary['protocol_id'] == plan['protocol_id']
    assert summary['steps'] == 6000 and summary['validation_frames'] == 872 and summary['test_frames'] == 188
    assert summary['test_used_for_selection'] is False
    output = {}
    required_pairs = [(arm, 'T1') for arm in names]
    required_pairs += [('R1F', 'R1N0'), ('R1P', 'R1F'), ('R1S', 'R1P')]
    if set(REPEAT).issubset(names):
        required_pairs.append(('R1S7', 'R1P7'))
    for split, count in [('validation872', 872), ('test188', 188)]:
        data = read(ROOT / 'metrics' / f'{split}.json')
        rows = data['rows']
        assert len(rows) == count and len({row['sample_id'] for row in rows}) == count
        assert data['original_size'] is True and data['tiling'] is False and data['step'] == 6000
        assert data['protocol_id'] == plan['protocol_id']
        reference = {row['sample_id']: row for row in read(PREV / 'metrics' / f'{split}.json')['rows']}
        assert set(reference) == {row['sample_id'] for row in rows}
        for row in rows:
            old = reference[row['sample_id']]
            assert row['t1_sha256'] == old['t1_sha256'], row['sample_id']
            assert row['shape'] == old['shape'] and row['scene'] == old['scene']
            assert row['metrics']['T1'] == old['metrics']['T1']
            assert all(name in row['metrics'] for name in names)
            for arm in BACKGROUND:
                assert row['metrics'][arm] == old['metrics'][arm], (row['sample_id'], arm)
        pairs = data['comparisons']
        for left, right in required_pairs:
            assert f'{left}-{right}' in pairs, (split, left, right)
        independent = {}
        for label, actual in pairs.items():
            left, right = label.split('-')
            expected = pair_summary(rows, left, right)
            same_summary(actual, expected)
            same_summary(summary['comparisons'][split][label], expected)
            independent[label] = expected
        oracle = {'2026': independent['R1S-R1P']}
        if set(REPEAT).issubset(names):
            oracle['2027'] = independent['R1S7-R1P7']
        net = {}
        for arm in names:
            pair = independent[f'{arm}-T1']
            net[arm] = dict(
                psnr_scene_equal=pair['psnr']['scene_equal'],
                ssim_scene_equal=pair['ssim']['scene_equal'],
                positive_scenes=pair['psnr']['scenes_positive'],
                negative_scenes={scene: value for scene, value in pair['psnr']['scenes'].items() if value < 0},
                practical_reference_met=(pair['psnr']['scene_equal'] >= .05
                                         and pair['ssim']['scene_equal'] >= 0
                                         and pair['psnr']['scenes_positive'] >= 6))
        output[split] = dict(frames=count, comparisons=independent, net_results=net,
                             rank_oracle_by_training_seed=oracle,
                             rank_oracle_repeat_present=len(oracle) == 2,
                             repeat_scope='training order, crops and diffusion flow; common pretrained lineage')
    return output


def run():
    budget = read(ROOT / 'budget.json')
    assert 0 < budget['deadline_unix'] - budget['start_unix'] <= 12 * 3600 + 1e-6
    close = read(ROOT / 'closeout.json')
    assert close['completed'] is True and close['deadline_met'] is True
    assert close['time_unix'] <= budget['deadline_unix']
    plan = read(ROOT / 'config/locked.json')
    names = plan['arms']
    assert names == CORE or names == CORE + REPEAT, names
    proposal = read(ROOT / 'config/proposal.json')
    if 'proposal_protocol' in plan:
        assert plan['proposal_protocol'] == proposal['protocol_id']
    for flag in ['conditions_verified', 'optimizer_replay_exact', 'source_frozen_exact',
                 'original_size_verified', 'all_preflight_updates_discarded',
                 'histogram_exact', 'suppression_verified', 'constant_exact']:
        assert plan[flag] is True, flag
    replay = plan['original_R1P_replay']
    assert {int(row['step']) for row in replay} == {1, 501}
    assert all(row['loss_exact'] and row['gradient_exact'] and row['samples_exact'] for row in replay)
    sources, artifacts = verify_receipts(proposal)
    training = list(csv.DictReader((ROOT / 'logs/training.csv').open()))
    assert [int(row['step']) for row in training] == list(range(1, 6001))
    for arm in names:
        assert all(math.isfinite(float(row[arm + '_loss']))
                   and math.isfinite(float(row[arm + '_grad_norm']))
                   and float(row[arm + '_grad_norm']) >= 0 for row in training), arm
    schedules, calibration = verify_schedules(proposal, names, training)
    assert plan['calibration_sha256'] == calibration['calibration_sha256']
    for step in range(500, 6001, 500):
        assert read(ROOT / 'diagnostics' / f'frozen_{step}.json')['passed'] is True
    complete = read(ROOT / 'diagnostics/training_complete.json')
    assert complete['steps'] == 6000 and complete['source_frozen_exact'] is True
    assert set(complete['arms']) == set(names)
    assert complete['final_sha256'] == sha(ROOT / 'weights/final.pt')
    import torch
    checkpoint = torch.load(ROOT / 'weights/final.pt', map_location='cpu', weights_only=False)
    assert checkpoint['step'] == 6000 and checkpoint['protocol_id'] == plan['protocol_id']
    assert set(checkpoint['arms']) == set(names) == set(checkpoint['optimizers'])
    for arm, optimizer in checkpoint['optimizers'].items():
        assert optimizer['state'], arm
        assert all(int(value['step']) == 6000 for value in optimizer['state'].values()), arm
        assert all(group['lr'] == 1e-4 and group['betas'] == (.9, .999)
                   and group['eps'] == 1e-8 and group['weight_decay'] == 0
                   for group in optimizer['param_groups']), arm
    del checkpoint
    output = verify_evaluation(plan, names)
    record = dict(passed=True, time_unix=time.time(), steps=6000, arms=names,
                  validation=872, test=188, all_source_and_artifact_hashes_verified=True,
                  source_receipts_checked=sources, artifact_files_checked=artifacts,
                  optimizer_steps_verified=True, paired_summaries_recomputed=True,
                  original_R1P_reuse_verified=True, original_R1P_replay=replay,
                  schedules=schedules, calibration=calibration, report=output,
                  oracle_uses_GT=True, deployment_multiseed_stability_established=False,
                  rank_oracle_training_seeds=([2026, 2027] if REPEAT[0] in names else [2026]),
                  closeout_sha256=sha(ROOT / 'closeout.json'),
                  artifact_manifest_sha256=sha(ROOT / 'artifact_manifest.json'), cpu_only=True)
    write(ROOT / 'reports/independent_audit.json', record)
    write(ROOT / 'reports/final_review_closeout.json', dict(
        completed=True, time_unix=time.time(), cpu_only=True,
        independent_audit_sha256=sha(ROOT / 'reports/independent_audit.json'),
        audit_script_sha256=sha(Path(__file__)), server_only=True))
    write(ROOT / 'audit_status.json', dict(stage='verified_completed', time_unix=time.time(),
                                         validation=872, test=188, cpu_only=True))


def main():
    budget = read(ROOT / 'budget.json')
    write(ROOT / 'audit_registration.json', dict(
        pid=os.getpid(), time_unix=time.time(), deadline_unix=budget['deadline_unix'],
        role='CPU-only independent verification; no training changes'))
    while not (ROOT / 'closeout.json').exists():
        if time.time() >= budget['deadline_unix']:
            write(ROOT / 'audit_status.json', dict(stage='deadline_incomplete', time_unix=time.time()))
            return
        write(ROOT / 'audit_status.json', dict(stage='waiting_for_closeout', time_unix=time.time()))
        time.sleep(30)
    run()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        write(ROOT / 'audit_status.json', dict(stage='verification_failed', time_unix=time.time(),
                                             traceback=traceback.format_exc(), cpu_only=True))
        raise
