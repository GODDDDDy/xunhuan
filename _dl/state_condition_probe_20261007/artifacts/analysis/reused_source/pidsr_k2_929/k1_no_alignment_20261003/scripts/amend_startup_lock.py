"""Retain the zero-update failure and lock the missing preflight metadata repair."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
assert not any((ROOT/'runs').rglob('step_*.pt'))
assert not any((ROOT/'runs').rglob('train.csv'))
assert json.loads((ROOT/'status.json').read_text())['state']=='failed'
old=ROOT/'diagnostics/startup_failure/protocol_lock.json'
lock=json.loads(old.read_text())
path=ROOT/'scripts/run_no_alignment.py'
old_sha=lock['locked_file_sha256'][str(path)]
new_sha=hashlib.sha256(path.read_bytes()).hexdigest()
assert old_sha!=new_sha
for p,h in lock['locked_file_sha256'].items():
    if p!=str(path):assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
lock['locked_file_sha256'][str(path)]=new_sha
lock['startup_amendment']={'reason':'Inherited RUN_PLAN requires peak_vram_mb; add measured preflight memory field.',
    'optimizer_updates_before_fix':0,'training_csv_or_step_checkpoint_before_fix':False,
    'loss_and_model_configuration_unchanged':True,'old_wrapper_sha256':old_sha,'new_wrapper_sha256':new_sha,
    'old_protocol_path':str(old),'failed_log_retained':str(ROOT/'logs/K1_seed2026_train_0_200.log')}
(ROOT/'protocol_lock.json').write_text(json.dumps(lock,indent=2))
print(json.dumps(lock['startup_amendment'],indent=2))
