"""Verify the immutable server package and its research evidence without GPU use."""
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


ready = read(ROOT / 'archive_ready.json')
package = ROOT / 'artifacts.tar.gz'
download = ROOT / 'artifacts.tar.gz.download'
if not package.exists():
    assert download.stat().st_size == ready['bytes'], 'Incomplete download'
    assert sha(download) == ready['sha256'], 'Package hash mismatch'
    download.replace(package)
else:
    assert package.stat().st_size == ready['bytes']
    assert sha(package) == ready['sha256']
print('Package size and SHA256 verified', flush=True)
manifest = read(ROOT / 'archive_manifest.json')
expected = {x['path']: x for x in manifest['files']}
assert len(expected) == len(manifest['files'])
destination = ROOT / 'artifacts'
destination.mkdir(exist_ok=True)
seen = set()
with tarfile.open(package, 'r|gz') as tar:
    for member in tar:
        rel = PurePosixPath(member.name)
        assert not rel.is_absolute() and '..' not in rel.parts and '\\' not in member.name
        assert member.isfile() and member.name not in seen
        assert member.name in expected or member.name == 'archive_manifest.json'
        path = destination.joinpath(*rel.parts)
        assert destination.resolve() in path.resolve().parents
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.unpack')
        h = hashlib.sha256()
        source = tar.extractfile(member)
        with temporary.open('wb') as f:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                h.update(block)
                f.write(block)
        if member.name == 'archive_manifest.json':
            assert read(temporary) == manifest
        else:
            item = expected[member.name]
            assert member.size == item['bytes'] and h.hexdigest() == item['sha256'], member.name
        temporary.replace(path)
        seen.add(member.name)
        if len(seen) % 40 == 0:
            print(f'Verified {len(seen)} package files', flush=True)
assert seen == set(expected) | {'archive_manifest.json'}
print('Every extracted file matches the manifest', flush=True)

cfg = read(destination / 'config/locked_run.json')
lock = read(destination / 'protocol_lock.json')
assert cfg['protocol_id'] == lock['active_protocol_id'] == '576f788b73218610533cddb66089d3e0c0910f9aa97604a376c9692d71634007'
for name, digest in lock['code_sha256'].items():
    assert sha(destination / name) == digest, name
assert read(destination / 'budget.json')['deadline_unix'] == 1791605047.4916465
assert cfg['task_initialization']['kind'] == 'official_polarfree'
for name, digest in cfg['task_initialization']['official_sha256'].items():
    assert sha(destination / 'weights/source_polarfree' / name) == digest, name
pre = read(destination / 'pretraining/selection.json')
test_pre = read(destination / 'pretraining/metrics/final_test.json')
assert sha(destination / 'pretraining/checkpoints/selected.pt') == pre['sha256'] == test_pre['checkpoint_sha256']
assert pre['updates'] == 19000 and pre['step'] == 18000 and pre['convergence']['stable']
assert test_pre['scenes'] == 30 and len(test_pre['per_scene']) == 30 and test_pre['test_completed']
assert len({x['scene'] for x in test_pre['per_scene']}) == 30
selection = read(destination / 'downstream/selection.json')
final_hash = sha(destination / 'downstream/checkpoints/final.pt')
assert final_hash == selection['sha256'] and selection['step'] == selection['planned_steps'] == 8000
with (destination / 'config/manifest.csv').open(newline='', encoding='utf-8') as f:
    data = list(csv.DictReader(f))
counts = {}
for split, count in [('train', 5440), ('validation', 872), ('test', 188)]:
    ids = {x['sample_id'] for x in data if x['split'] == split}
    assert len(ids) == count
    counts[split] = count
    if split == 'train':
        continue
    result = read(destination / f'downstream/metrics/final_{split}.json')
    assert result['frames'] == len(result['per_image']) == count
    assert {x['sample_id'] for x in result['per_image']} == ids
    assert len({x['sample_id'] for x in result['per_image']}) == count
    assert result['checkpoint_sha256'] == final_hash and result['checkpoint_step'] == 8000
    assert result['protocol_id'] == cfg['protocol_id']
    assert result['original_size'] and not result['tiling'] and not result['test_used_for_selection']
    assert len(result['confidence']['per_image']) == count
    assert {x['sample_id'] for x in result['confidence']['per_image']} == ids
    assert sum(x['frames'] for x in result['per_scene']) == count
    for metric in ['psnr', 'ssim']:
        for x in result['per_image']:
            assert math.isclose(x['round2'][metric] - x['round1'][metric], x['iteration_gain'][metric], abs_tol=1e-12)
        for round_key in ['round1', 'round2', 'iteration_gain']:
            average = sum(x[round_key][metric] for x in result['per_image']) / count
            assert math.isclose(average, result[round_key]['image_mean_' + metric], abs_tol=1e-10)
    examples = list((destination / 'downstream/metrics/confidence_examples' / split).glob('*.npz'))
    assert len(examples) == len(result['per_scene'])
receipt = dict(verified_unix=time.time(),package_sha256=ready['sha256'],package_bytes=ready['bytes'],
               verified_manifest_files=len(expected),all_manifest_hashes_match=True,protocol_id=cfg['protocol_id'],
               source_official_sha256=cfg['task_initialization']['official_sha256'],
               pretraining_selected_sha256=pre['sha256'],downstream_final_sha256=final_hash,
               dataset_counts=counts,pretraining_test_scenes=30,downstream_original_size=True,
               gpu_used_for_this_audit=False,completed=True)
(ROOT / 'archive_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt), flush=True)
