"""Bind the small controlled pilot after the read-only diagnostic gate."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
lock=json.loads((ROOT/'protocol_lock.json').read_text())
assert all(sha(p)==h for p,h in lock['locked_file_sha256'].items())
diag=json.loads((ROOT/'diagnostics/diagnostic_result.json').read_text())
assert diag['training_gate_passed'] and diag['all_endpoints_passed'] and diag['all_frozen']
assert diag['independent_endpoint_checks']==56
files=dict(lock['locked_file_sha256'])
for p in list((ROOT/'scripts').glob('*.py'))+list((ROOT/'configs').glob('*.yaml')):
    files[str(p)]=sha(p)
assert not (ROOT/'training_lock.json').exists()
payload=dict(locked_file_sha256=files,diagnostic_result_sha256=sha(ROOT/'diagnostics/diagnostic_result.json'),
    conditions=['mixed_rgb','task_prediction'],training_seed=2026,evaluation_seed=2026,steps=1000,
    actual_task_loss_weights=[0.0,1.0],intermediate_gate=200,
    evaluated_checkpoint='step_1000',full_validation_images=872,test_evaluation=False,
    pairing='identical model and G6 initial tensors, schedule/crops/RNG, optimizer/precision; only condition representation differs',
    condition_normalization='mixed RGB already normalized; task prediction detached and clamped to [0,1]; unchanged raw RGB still enters TaskNet',
    promotion_gate=lock['promotion_gate'],scope='one paired seed exploratory pilot; no convergence claim')
(ROOT/'training_lock.json').write_text(json.dumps(payload,indent=2));print(json.dumps({k:v for k,v in payload.items() if k!='locked_file_sha256'},indent=2))
