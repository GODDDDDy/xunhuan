"""Archive completed study evidence; retain large trained weights on the server."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parent.parent
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read(ROOT / 'status.json')['state'] == 'complete'
assert read(ROOT / 'analysis/threeway_paired_summary.json')['all_paired_initializations_and_diffusion_seeds_matched']
lock = read(ROOT / 'protocol_lock.json')
assert all(sha(p) == h for p, h in lock['locked_file_sha256'].items())

def retain(path):
    return path.is_file() and path.suffix not in {'.pt', '.pth', '.pyc', '.tmp'} and not any(p in {'__pycache__', 'exports'} for p in path.parts)

reused = Path(lock['reused_G6_seed2026']['root'])
dest = ROOT / 'analysis/reused_G6_seed2026_evidence'
for source in sorted(reused.rglob('*')):
    if retain(source):
        target = dest / source.relative_to(reused)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

index_path = ROOT / 'evidence_sha256.json'
files = [p for p in sorted(ROOT.rglob('*')) if retain(p) and p != index_path]
index = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
index_path.write_text(json.dumps(index, indent=2))
exports = ROOT / 'exports'
exports.mkdir(exist_ok=True)
archive = exports / 'normalized_threeway_evidence_without_checkpoints.tar.gz'
with tarfile.open(archive, 'w:gz') as tf:
    for path in files + [index_path]:
        tf.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
receipt = {'archive': str(archive), 'sha256': sha(archive), 'bytes': archive.stat().st_size,
           'indexed_files': len(index), 'excluded': ['.pt', '.pth', '.pyc', '.tmp', '__pycache__', 'exports'],
           'checkpoints_retained_on_server': True, 'checkpoint_fingerprints': read(ROOT / 'analysis/threeway_paired_summary.json')['source_fingerprints']}
(exports / 'archive_receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
