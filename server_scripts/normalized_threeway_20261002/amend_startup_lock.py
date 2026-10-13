"""Record startup-only protocol binding repair, before any optimizer updates."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
assert not list((ROOT / 'runs').rglob('*.pt'))
assert not list((ROOT / 'runs').rglob('train.csv'))
path = ROOT / 'protocol_lock.json'
lock = json.loads(path.read_text())
old = dict(lock['locked_file_sha256'])
for filename in lock['locked_file_sha256']:
    lock['locked_file_sha256'][filename] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
amendment = {'reason': 'Fixed calibration called the old wrapper protocol helper, omitting matched extension hash.',
             'repair': 'Bind fixed unchanged lambda to base.protocol_id of the matched wrapper.',
             'optimizer_updates_before_repair': 0, 'loss_or_hyperparameter_change': False,
             'failed_logs_preserved': True,
             'changed_files': {n: {'before': old[n], 'after': v} for n, v in lock['locked_file_sha256'].items() if old[n] != v}}
lock['startup_amendment'] = amendment
path.write_text(json.dumps(lock, indent=2))
(ROOT / 'diagnostics/startup_failure/amendment.json').write_text(json.dumps(amendment, indent=2))
print(json.dumps(amendment, indent=2))
