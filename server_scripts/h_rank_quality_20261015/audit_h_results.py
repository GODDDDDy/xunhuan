"""Read-only result checks plus a small server-side closeout receipt."""
import csv,json,math,time
import numpy as np
from common import ROOT,sha,json_save
plan=json.loads((ROOT/'config/plan.json').read_text())
report=json.loads((ROOT/'metrics/validation64.json').read_text())
training=list(csv.DictReader((ROOT/'logs/training.csv').open()))
assert len(training)==plan['steps']==2000
assert [int(r['step']) for r in training]==list(range(1,2001))
ids=[x for r in training for x in r['sample_ids'].split(';')]
assert len(ids)==len(set(ids))==4000 and all(s.startswith('train/') for s in ids)
selected=json.loads((ROOT/'config/validation64.json').read_text())['sample_ids']
assert [r['sample_id'] for r in report['rows']]==selected
assert len(report['scenes'])==8
assert all(sum(r['scene']==s for r in report['rows'])==8 for s in report['scenes'])
for arm,values in report['scene_equal'].items():
    for metric,value in values.items():
        recomputed=float(np.mean([r['heads'][arm][metric] for r in report['rows']]))
        assert math.isclose(value,recomputed,abs_tol=1e-12),(arm,metric)
manifest=json.loads((ROOT/'artifact_manifest.json').read_text())
for item in manifest['files']:
    p=ROOT/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],item['path']
assert sha(ROOT/'weights/common_initial21.pt')==plan['source_initial_sha256']
endpoint=sha(ROOT/'weights/final_heads.pt')
assert endpoint==json.loads((ROOT/'diagnostics/paired_training.json').read_text())['endpoint_sha256']
summary=dict(protocol_id=plan['protocol_id'],screening_passed=report['screening_passed'],
             steps_per_arm=2000,unique_training_images=4000,validation64=64,test=0,
             scene_equal=report['scene_equal'],ause_scenes_improved=report['ause_scenes_improved'],
             scenes=report['scenes'],endpoint_sha256=endpoint,
             source_sha256=plan['source_initial_sha256'],verified_files=len(manifest['files']),
             training_csv_sha256=sha(ROOT/'logs/training.csv'),
             evaluation_repair=json.loads((ROOT/'diagnostics/evaluation_repair.json').read_text()),
             failures=[p.name for p in (ROOT/'diagnostics').glob('*failure*.json')],
             time_unix=time.time())
if report['screening_passed']:
    if (ROOT/'full_confirmation.json').exists():
        assert json.loads((ROOT/'full_confirmation.json').read_text())['completed']
        for split,n in (('validation',872),('test',188)):
            data=json.loads((ROOT/'metrics'/f'{split}.json').read_text())
            assert len(data['rows'])==n
            summary[split]=dict(frames=n,scene_equal=data['scene_equal'],scenes=data['scenes'])
        summary['test']=188;summary['full_confirmation']=True
    else:summary['full_confirmation']=False
else:
    assert not (ROOT/'metrics/test.json').exists()
    summary['full_confirmation']=False
json_save(ROOT/'reports/audit_summary.json',summary)
print(json.dumps(summary,ensure_ascii=False,allow_nan=False))
