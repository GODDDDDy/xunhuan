"""CPU-only completion audit and archive, extending the existing evidence pattern."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import yaml

ROOT = Path('/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/confidence_joint_20261009_v2')
read = lambda p: json.loads(Path(p).read_text())

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

status = read(ROOT / 'status.json')
selection = read(ROOT / 'evaluation_selection.json')
lock = read(ROOT / 'protocol_lock.json')
assert status['stage'] == 'completed' and status['test_completed'] and status['within_deadline']
assert selection['common_step'] == 16000
assert selection['trained_steps'] == {'A2_dual': 16000, 'A0_direct': 16000}
assert all(sha(ROOT / path) == digest for path, digest in lock['code_sha256'].items())
assert all(sha(ROOT / 'config' / (unit + '.yaml')) == digest for unit, digest in lock['config_sha256'].items())
assert sha(ROOT / 'config/quality_calibration.json') == lock['quality_calibration_sha256']

audit = dict(passed=True, cpu_only=True, common_step=16000, source_sha256=lock['source_sha256'],
             manifest_sha256=lock['manifest_sha256'], schedule_sha256=lock['schedule_sha256'],
             all_locked_code_config_calibration_unchanged=True, units={})
for unit, mode in [('A0_direct', 'direct'), ('A2_dual', 'dual')]:
    cfg = yaml.safe_load((ROOT / 'config' / (unit + '.yaml')).read_text())
    assert sha(cfg['study']['source_checkpoint']) == lock['source_sha256']
    assert sha(cfg['paths']['manifest']) == lock['manifest_sha256']
    assert sha(cfg['paths']['schedule']) == lock['schedule_sha256']
    if not (ROOT / 'config/manifest.csv').exists():
        shutil.copy2(cfg['paths']['manifest'], ROOT / 'config/manifest.csv')
    with Path(cfg['paths']['manifest']).open(newline='') as stream:
        manifest = list(csv.DictReader(stream))
    path = ROOT / 'arms' / unit / 'run/checkpoints/step_16000.pt'
    state = torch.load(path, map_location='cpu', weights_only=False)
    steps = [float(x['step']) for x in state['optimizer']['state'].values()]
    groups = [g['name'] for g in state['optimizer']['param_groups']]
    assert state['step'] == 16000 and min(steps) == max(steps) == 16000
    assert groups == ['r2p_and_alignment', 'tasknet_net_g', 'quality_head']
    # The extension records head presence for both arms; mode controls gate use.
    assert state['confidence_mode'] == mode and state['confidence_enabled'] is True
    assert state['actual_rounds'] == 2 and state['source_training_steps'] == 13000
    assert state['source_sha256'] == lock['source_sha256']
    assert any(key.startswith('confidence_head.') for key in state['r2p_and_projections'])
    checkpoint_sha = sha(path)
    result = read(ROOT / 'arms' / unit / 'run/metrics/experiment_results.json')
    assert result['completed'] and result['test_required']
    assert result['fixed_additional_step'] == 16000 and result['historical_steps'] == 29000
    split_audits = {}
    for split, count in [('validation', 872), ('test', 188)]:
        values = result[split]
        expected = {r['sample_id'] for r in manifest if r['split'] == split}
        actual = [r['sample_id'] for r in values['per_image']]
        quality = values['quality_confidence']
        assert len(expected) == values['frames'] == len(actual) == count
        assert len(set(actual)) == count and set(actual) == expected
        assert {r['sample_id'] for r in quality['per_image']} == expected
        assert len(quality['per_image']) == count
        assert values['checkpoint_step'] == 16000 and values['checkpoint_sha256'] == checkpoint_sha
        assert values['manifest_sha256'] == lock['manifest_sha256']
        assert values['test_used_for_selection'] is False
        assert values['actual_forward_calls'] == {'r2p': 2 * count, 'tasknet': 2 * count}
        split_audits[split] = dict(frames=count, all_unique_manifest_ids_matched=True,
                                  actual_forward_calls=values['actual_forward_calls'], quality_rows=count)
    audit['units'][unit] = dict(checkpoint_sha256=checkpoint_sha, optimizer_groups=groups,
                               optimizer_state_count=len(steps), all_adam_steps=16000,
                               confidence_parameters_checkpointed=True, splits=split_audits)
    del state
(ROOT / 'operations/final_completion_audit.json').write_text(json.dumps(audit, indent=2))

def retain(path):
    relative = path.relative_to(ROOT)
    if not path.is_file() or any(part in {'exports', '__pycache__'} for part in relative.parts):
        return False
    if path.suffix in {'.pyc', '.tmp'} or path.name == 'evidence_sha256.json':
        return False
    return path.suffix not in {'.pt', '.pth'} or path.name == 'step_16000.pt'

files = [p for p in sorted(ROOT.rglob('*')) if retain(p)]
index = {p.relative_to(ROOT).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size) for p in files}
index_path = ROOT / 'evidence_sha256.json'
index_path.write_text(json.dumps(index, indent=2))
exports = ROOT / 'exports'
exports.mkdir(exist_ok=True)
archive = exports / 'confidence_joint_20261009_v2_completed.tar.gz'
with tarfile.open(archive, 'w:gz', compresslevel=1) as tf:
    for path in files + [index_path]:
        tf.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
receipt = dict(archive=str(archive), sha256=sha(archive), bytes=archive.stat().st_size,
               indexed_files=len(index), evidence_index_sha256=sha(index_path),
               common_step=16000, checkpoint_sha256={k: v['checkpoint_sha256'] for k, v in audit['units'].items()},
               excluded='intermediate/best checkpoints, pyc, tmp, exports; originals retained on server')
(exports / 'archive_receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2), flush=True)
