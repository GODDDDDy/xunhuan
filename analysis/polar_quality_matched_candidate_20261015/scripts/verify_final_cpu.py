"""Read-only checkpoint verification after completion; never allocates CUDA."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import torch

root = Path(sys.argv[1])
torch.set_num_threads(2)
source = torch.load(root / 'weights/common_initial21.pt', map_location='cpu', weights_only=False)
plan = json.loads((root / 'config/plan.json').read_text())
report = {'cpu_only': True, 'checked_unix': time.time(), 'arms': {}, 'splits': {}}
sequences = []
for arm in ('real_quality', 'fixed_quality'):
    path = root / 'arms' / arm / 'final.pt'
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert ck['step'] == 2000 and ck['protocol_id'] == plan['protocol_id']
    frozen = [k for k in source['r2p_and_projections'] if not k.startswith(('feedback.', 'error_guidance.'))]
    assert all(torch.equal(source['r2p_and_projections'][k], ck['r2p_and_projections'][k]) for k in frozen)
    assert all(torch.equal(v, ck['tasknet_net_g'][k]) for k, v in source['tasknet_net_g'].items())
    rows = list(csv.DictReader((root / 'arms' / arm / 'train.csv').open()))
    assert [int(row['step']) for row in rows] == list(range(1, 2001))
    sequences.append([(row['step'], row['sample_ids']) for row in rows])
    monitors = [json.loads(p.read_text()) for p in (root / 'arms' / arm / 'monitor').glob('*.json')]
    base = next(x for x in monitors if x['step'] == 0)
    identity = lambda x: [(row['sample_id'], row['t1_sha256']) for row in x['rows']]
    assert sorted(x['step'] for x in monitors) == [0, 500, 1000, 1500, 2000]
    assert all(x['first_round_fixed'] and identity(x) == identity(base) for x in monitors)
    report['arms'][arm] = {
        'step': ck['step'], 'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest(),
        'unique_training_images': len({sid for row in rows for sid in row['sample_ids'].split(';')}),
        'frozen_model_tensors_exact': len(frozen), 'net_g_tensors_exact': len(source['tasknet_net_g']),
        'updated_feedback_Q_tensors': sum(not torch.equal(v, ck['r2p_and_projections'][k])
            for k, v in source['r2p_and_projections'].items() if k.startswith(('feedback.', 'error_guidance.'))),
        'all_monitor_T1_hashes_exact': True}
    assert report['arms'][arm]['unique_training_images'] == 3640
assert sequences[0] == sequences[1]
report['paired_training_sample_ids_exact'] = True
for split, n in [('validation', 872), ('test', 188)]:
    data = json.loads((root / 'metrics' / f'final_{split}.json').read_text())
    rows = data['rows']
    assert len(rows) == n and len({x['sample_id'] for x in rows}) == n
    assert data['original_size'] and not data['tiling']
    assert all(all(arm['t1_equal'] for arm in row['arms'].values()) for row in rows)
    report['splits'][split] = {'unique_frames': n, 'original_size': True, 'tiling': False,
        'all_arms_T1_equal': True, 'scenes': len(data['per_scene']),
        'float_examples': len(list((root / 'examples' / split).glob('*.npz')))}
receipt = root / 'reports' / 'final_checkpoint_verification.json'
receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False, indent=2))
