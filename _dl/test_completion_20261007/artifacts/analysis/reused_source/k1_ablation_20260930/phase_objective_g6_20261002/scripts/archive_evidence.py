"""Archive all small reproducibility evidence; trained weights stay on the server."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent.parent
assert json.loads((ROOT / 'status.json').read_text())['state'] == 'complete'
assert (ROOT / 'analysis/matched_g6_summary.json').is_file()
assert (ROOT / 'analysis/G6_PHASE01_MATCHED_REPORT_ZH_20261002.md').is_file()
ARCHIVE = ROOT / 'g6_phase01_evidence_without_checkpoints_20261002.tar.gz'
INDEX = ROOT / 'evidence_file_sha256.json'
items = sorted(p for p in ROOT.rglob('*') if p.is_file()
               and p not in (ARCHIVE, INDEX)
               and '__pycache__' not in p.parts
               and p.suffix not in ('.pt', '.pth', '.pyc', '.tmp')
               and not p.name.endswith('.tar.gz'))
fingerprints = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in items}
INDEX.write_text(json.dumps(fingerprints, indent=2))
with tarfile.open(ARCHIVE, 'w:gz') as archive:
    for path in items + [INDEX]:
        archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
payload = {'path': str(ARCHIVE), 'bytes': ARCHIVE.stat().st_size, 'sha256': digest,
           'files': len(items) + 1, 'trained_weights_included': False,
           'final_checkpoint_path': str(ROOT / 'runs/G6_phase01/seed_2026/checkpoints/step_3000.pt')}
(ROOT / 'archive_receipt.json').write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
