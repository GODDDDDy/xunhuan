"""Explicitly bind the already-declared G6 files as well as their config pointers."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
prior=json.loads((ROOT.parent/'test_completion_20261007/protocol_lock.json').read_text())
u=next(u for u in prior['units'] if u['id']=='G6_seed2026')
p=ROOT/'training_lock.json';old=p.read_text();lock=json.loads(old)
assert sha(u['checkpoint'])==u['checkpoint_sha256']
assert sha(u['validation_metrics'])==u['validation_metrics_sha256']
backup=ROOT/'diagnostics/training_lock_before_G6_receipt.json';assert not backup.exists()
backup.write_text(old)
for key in ('checkpoint','validation_metrics'):lock['locked_file_sha256'][u[key]]=sha(u[key])
lock['G6_explicit_source_receipt']=dict(checkpoint=u['checkpoint'],checkpoint_sha256=u['checkpoint_sha256'],
    validation_metrics=u['validation_metrics'],validation_metrics_sha256=u['validation_metrics_sha256'],
    source_files_match_prior_test_lock=True,no_configuration_or_weight_changes=True,
    note='Extend file-receipt coverage of already-declared source after preflight began; controls and model protocol unchanged.')
p.write_text(json.dumps(lock,indent=2));print(json.dumps(lock['G6_explicit_source_receipt'],indent=2))
