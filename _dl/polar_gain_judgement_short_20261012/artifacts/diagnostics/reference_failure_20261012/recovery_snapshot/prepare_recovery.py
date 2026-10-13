"""Preserve the first recording failure and verify cached diagnostic arrays."""
import hashlib,json,pathlib,shutil
R=pathlib.Path('/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/polar_gain_judgement_short_20261012')
D=R/'diagnostics/recording_failure_20261012'
assert not pathlib.Path('/proc/1912234').exists() and not pathlib.Path('/proc/1912341').exists()
assert not D.exists()
D.mkdir()
for name in ('status.json','archive_ready.json','artifacts.tar.gz','archive_manifest.json','RESULTS_ZH.md','logs/controller.log','logs/diagnostic.log','logs/failure.log'):
 p=R/name
 if p.exists():q=D/name;q.parent.mkdir(parents=True,exist_ok=True);p.rename(q)
shutil.copytree(R/'scripts',D/'frozen_scripts',ignore=shutil.ignore_patterns('__pycache__'))
for name in ('budget.json','watchdog.json','config/locked_run.json','config/lock_receipt.json'):
 q=D/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/name,q)
hashes={}
for p in (R/'diagnostics/feature_cache').glob('*.npz'):
 hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
assert set(hashes)=={'judgement.npz','bias.npz','safety.npz'}
(R/'recovery/cache_receipt.json').write_text(json.dumps(hashes,indent=2))
(D/'reason.json').write_text(json.dumps(dict(reason='NumPy int64 positive-scene count was not JSON serializable',algorithm_changed=False,weights_changed=False,formal_training_started=False,first_full_validation_started=False,cache_hashes=hashes,budget_reset=False),indent=2))
print(json.dumps(dict(failure_preserved=str(D),cache_files=len(hashes))))
