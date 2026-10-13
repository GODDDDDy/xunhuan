"""Verify the downloaded archive and every archived file, without GPU use."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


ready = json.loads((ROOT / 'archive_ready.json').read_text())
package = ROOT / 'artifacts.tar.gz'
assert package.stat().st_size == ready['bytes']
assert sha(package) == ready['sha256']
assert ready['completed']
print('Archive hash matches.', flush=True)

destination = ROOT / 'artifacts'
destination.mkdir(exist_ok=True)
with tarfile.open(package) as archive:
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        assert target.is_relative_to(destination.resolve())
        assert not member.issym() and not member.islnk()
    archive.extractall(destination, filter='data')

manifest = json.loads((destination / 'archive_manifest.json').read_text())
for index, item in enumerate(manifest['files'], 1):
    path = destination / item['path']
    assert path.stat().st_size == item['bytes'], item['path']
    assert sha(path) == item['sha256'], item['path']
    if index % 25 == 0:
        print(f'Verified {index} files.', flush=True)

receipt = dict(archive_sha256=ready['sha256'], bytes=ready['bytes'],
               files_verified=len(manifest['files']), all_hashes_match=True,
               gpu_used=False, source_archive_unchanged=True)
(ROOT / 'archive_receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)
