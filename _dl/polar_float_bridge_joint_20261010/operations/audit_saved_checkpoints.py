"""CPU verification of official initialization and the actually saved freeze boundary."""
from pathlib import Path
import csv
import hashlib
import gc
import json
import math
import torch

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts'
expected_hashes = {
    'weights/source_polarfree/net_g_latest.pth': 'a230e77fb19827a1ada41e4dc86374dee092e476915d2252841f81dd8c257ca2',
    'weights/source_polarfree/net_d_latest.pth': 'f1d0e9acb244480e5d4ba4b5cee7ba35666fb97d41bccf4659f20911f46f1fb3',
    'weights/source_polarfree/net_le_dm_latest.pth': '6b98bd2a715d3e669acd99582d9ea37594ff4a2ee7786df86764d8efdafd0546',
    'pretraining/checkpoints/selected.pt': 'a4ecebacba5cfc25546320974a9c37f1cc220c685f1318b3cc290ae4a68e9554',
    'downstream/checkpoints/final.pt': '90720752bef847f7da3ba73849e0f5aa9ff3bd5b9b9f6c7efc0638b1e4297227',
}
for relative, expected in expected_hashes.items():
    digest = hashlib.sha256()
    with (ARTIFACTS / relative).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == expected, relative

official = torch.load(ARTIFACTS / 'weights/source_polarfree/net_g_latest.pth', map_location='cpu', weights_only=False)['params']
results = []
for file, step in [('initial.pt', 0), ('step_500.pt', 500), ('final.pt', 6000)]:
    state = torch.load(ARTIFACTS / 'downstream/checkpoints' / file, map_location='cpu', weights_only=False)
    assert state['step'] == step
    assert state['protocol_id'] == 'c3cca742318e5d89bab57311c014a56a658e9c4a143024ac558bf033ab93de5f'
    weights = state['tasknet_net_g']
    assert set(weights) == set(official)
    equal = all(torch.equal(weights[k], official[k]) for k in official)
    if step <= 500:
        assert equal
    if step == 0:
        pretraining = torch.load(ARTIFACTS / 'pretraining/checkpoints/selected.pt', map_location='cpu', weights_only=False)
        assert pretraining['step'] == 18000 and pretraining['protocol_id'] == '576f788b73218610533cddb66089d3e0c0910f9aa97604a376c9692d71634007'
        assert all(torch.equal(value, state['r2p_and_projections']['r2p.' + name])
                   for name, value in pretraining['enhancer'].items())
        migration_keys = len(pretraining['enhancer'])
        del pretraining
    assert all(torch.isfinite(v).all() for v in weights.values())
    assert all(torch.isfinite(v).all() for v in state['r2p_and_projections'].values())
    results.append(dict(file=file, step=step, official_state_tensors=len(weights), equal_to_official=equal, finite=True))
    del state, weights
    gc.collect()
with (ARTIFACTS / 'downstream/logs/train.csv').open(newline='') as f:
    rows = list(csv.DictReader(f))
assert [int(x['step']) for x in rows] == list(range(1, 6001))
seen = set()
for x in rows:
    step = int(x['step'])
    expected = 0.0 if step <= 500 else 1e-6 * min((step - 500) / 200, 1)
    assert math.isclose(float(x['task_lr']), expected, abs_tol=1e-15)
    assert float(x['correction_lr']) == 2e-5 and float(x['interfaces_lr']) == 1e-4
    assert (x['tasknet_trainable'] == 'True') == (step > 500)
    assert all(math.isfinite(float(x[k])) for k in ['loss', 'task', 'quality', 'alignment', 'grad_norm'])
    seen.update(x['sample_ids'].split(';'))
assert len(seen) == 5440 and all(x.startswith('train/') for x in seen)
manifest = list(csv.DictReader((ARTIFACTS / 'config/manifest.csv').open()))
assert seen == {x['sample_id'] for x in manifest if x['split'] == 'train'}
audit = dict(checkpoints=results, all_6000_step_logs_finite=True, lr_schedule_matches=True,
             training_images_seen=5440, enhancer_migration_exact=True, enhancer_migration_state_tensors=migration_keys,
             gpu_used=False, source_and_final_hashes=expected_hashes)
(ROOT / 'interpretation/saved_checkpoint_audit.json').write_text(json.dumps(audit, indent=2))
print(json.dumps(audit))
