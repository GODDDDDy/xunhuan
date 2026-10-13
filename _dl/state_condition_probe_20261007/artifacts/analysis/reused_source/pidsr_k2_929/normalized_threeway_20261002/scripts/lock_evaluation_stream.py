"""Record fixed evaluation RNG before any new final872 outcome is available."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
assert not list((ROOT / 'runs').glob('*/*/metrics/step_3000_validation.json'))
path = ROOT / 'protocol_lock.json'
text = path.read_text()
(ROOT / 'diagnostics/protocol_lock_pre_evaluation_stream.json').write_text(text)
lock = json.loads(text)
helper = ROOT / 'scripts/evaluate_matched.py'
before = lock['locked_file_sha256'][str(helper)]
after = hashlib.sha256(helper.read_bytes()).hexdigest()
lock['locked_file_sha256'][str(helper)] = after
lock['evaluation_seed'] = 2026
lock['evaluation_stream_amendment'] = {'chosen_before_new_final872_evaluations': True,
    'training_loss_or_configuration_changed': False,
    'reason': 'Fix inference RNG across all seeds to measure training pipeline seed variability.',
    'helper_sha256_before': before, 'helper_sha256_after': after}
path.write_text(json.dumps(lock, indent=2))
print(json.dumps(lock['evaluation_stream_amendment'], indent=2))
