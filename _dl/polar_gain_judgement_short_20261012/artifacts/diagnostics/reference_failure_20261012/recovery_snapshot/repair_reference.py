"""Attach existing audited T1 hashes to reused final-oracle records by sample ID."""
import hashlib,json,pathlib,shutil
R=pathlib.Path('/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/polar_gain_judgement_short_20261012')
OLD=R.parent/'polar_error_guided_short_20261012'
D=R/'diagnostics/reference_failure_20261012'
assert not pathlib.Path('/proc/1916509').exists() and not pathlib.Path('/proc/1916537').exists()
assert (R/'archive_ready.json').exists() and not D.exists()
D.mkdir()
for name in ('status.json','archive_ready.json','artifacts.tar.gz','archive_manifest.json','RESULTS_ZH.md','logs/recovery_controller.log','logs/diagnostic.log','logs/recovery_failure.log','recovery/recovery_receipt.json','diagnostics/source_oracle_validation872.json'):
 p=R/name
 if p.exists():q=D/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q) if name.startswith(('recovery/','diagnostics/')) else p.rename(q)
shutil.copytree(R/'recovery',D/'recovery_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
reference=OLD/'diagnostics/source_oracle_validation872.json'
shutil.copy2(reference,R/'diagnostics/t1_hash_reference_source21.json')
source=json.loads(reference.read_text());final=json.loads((R/'diagnostics/source_oracle_validation872.json').read_text())
assert final['t1_all_float_hashes_equal_source'] is True
hashes={r['sample_id']:r['t1_tensor_sha256'] for r in source['per_image']}
assert len(hashes)==len(final['per_image'])==872
assert {r['sample_id'] for r in final['per_image']}==set(hashes)
for r in final['per_image']:r['t1_tensor_sha256']=hashes[r['sample_id']]
final['t1_hash_origin']=dict(path=str(reference),sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),reuse_justification='experiment21 final evaluation previously verified every T1 equal to this reference; current full forward will assert again',candidate_oracle_metrics_still_source21_final=True)
(R/'diagnostics/source_oracle_validation872.json').write_text(json.dumps(final,indent=2))
(R/'recovery/reference_repair_receipt.json').write_text(json.dumps(dict(reason='final-oracle summary lacks T1 hash field; correct hashes live in separate source-oracle file',algorithm_changed=False,weights_changed=False,formal_training_started=False,source_reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),paired_unique_samples=872,repaired_reference_sha256=hashlib.sha256((R/'diagnostics/source_oracle_validation872.json').read_bytes()).hexdigest(),budget_reset=False),indent=2))
print(json.dumps(dict(paired_T1_hashes=872,algorithm_unchanged=True)))
