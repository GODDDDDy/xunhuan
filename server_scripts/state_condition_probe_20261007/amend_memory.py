"""Retain the failed launch and bind a forward-equivalent memory fix."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=ROOT/'protocol_lock.json';old=p.read_text();lock=json.loads(old)
assert not (ROOT/'diagnostics/diagnostic_result.json').exists()
assert not list((ROOT/'runs').glob('*/*/checkpoints/*.pt'))
old_script=ROOT/'scripts/diagnose_attempt1.py';new_script=ROOT/'scripts/diagnose.py'
assert sha(old_script)==lock['locked_file_sha256'][str(new_script)]
(ROOT/'diagnostics/protocol_lock_before_memory_amendment.json').write_text(old)
lock['locked_file_sha256'][str(old_script)]=sha(old_script)
lock['locked_file_sha256'][str(new_script)]=sha(new_script)
lock['locked_file_sha256'][str(Path(__file__))]=sha(__file__)
lock['memory_amendment']=dict(reason='Another GPU experiment appeared (PID2570511,1570MiB); initial diagnostic OOM before results.',
    changes=['Move diagnostic traces to CPU after actual forward.','Release already-measured gate/raw-delta tensors.','Empty inactive CUDA cache between images.','Use expandable_segments allocator.'],
    no_training_performed=True,no_input_precision_weights_or_alpha_changes=True,failed_log='logs/diagnosis_attempt1.log')
p.write_text(json.dumps(lock,indent=2))
print(json.dumps(lock['memory_amendment']))
