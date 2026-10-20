"""Copy verified inference code and inputs into an isolated, read-only-source audit."""
from pathlib import Path
import hashlib,json,shutil,sys,time
BASE=Path('/media/li/data3/25thstudent/DY/929work/pidsr_k2_929')
OLD=BASE/'polar_quality_matched_candidate_20261015'
ROOT=BASE/'polar_quality_mechanism_audit_20261015'
INCOMING=BASE/'audit28_incoming_20261015'
assert not ROOT.exists(),'Inspect existing processes/state before reusing an existing run'
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
for d in ['config','diagnostics','reports','weights','logs','metrics','examples','pretraining/checkpoints']:(ROOT/d).mkdir(parents=True,exist_ok=True)
shutil.copytree(OLD/'scripts',ROOT/'scripts',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
oldplan=json.loads((OLD/'config/plan.json').read_text())
for item in oldplan['code_manifest']:
    assert sha(ROOT/item['path'])==item['sha256']
for name in ['audit_run.py','audit_controller.py']:shutil.copy2(INCOMING/name,ROOT/'scripts'/name)
for name in ['common_initial21.pt','migration21_final.pt']:(ROOT/'weights'/name).symlink_to((OLD/'weights'/name).resolve())
(ROOT/'pretraining/checkpoints/selected.pt').symlink_to((OLD/'pretraining/checkpoints/selected.pt').resolve())
cfg=json.loads((OLD/'config/execution_config.json').read_text());cfg['paths']['output_root']=str(ROOT/'downstream')
(ROOT/'config/execution_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
for source,target in [('config/fixed_calibration.json','config/fixed_calibration.json'),('config/plan.json','config/source27_plan.json'),
                      ('metrics/final_validation.json','config/source27_final_validation.json'),('archive_ready.json','config/source27_archive_ready.json')]:
    shutil.copy2(OLD/source,ROOT/target)
shutil.copy2(INCOMING/'PROTOCOL_ZH.md',ROOT/'PROTOCOL_ZH.md')
code=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted((ROOT/'scripts').rglob('*.py'))]
plan=dict(kind='Experiment27 read-only baseline completion and mechanism audit',training=False,test_used=False,
     seed=2026,phase='matched_eval_validation',validation_frames=872,arms=['real_quality','fixed_quality'],
     source27_protocol=oldplan['protocol_id'],source_initial_sha256=oldplan['source_initial_sha256'],
     code_manifest=code,execution_config_sha256=sha(ROOT/'config/execution_config.json'),
     fixed_calibration_sha256=sha(ROOT/'config/fixed_calibration.json'),
     reference_sha256=sha(ROOT/'config/source27_final_validation.json'),budget_seconds=10800,
     local_examples='first fixed validation frame per8scenes; no test examples',original_size=True,tiling=False,
     reconstruction_method='same FP32 source and per-frame seeds; exact T1 hashes required',created_unix=time.time())
plan['protocol_id']=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
for name in ['plan.json','locked_run.json']:(ROOT/'config'/name).write_text(json.dumps(plan,ensure_ascii=False,indent=2))
(ROOT/'config/lock_receipt.json').write_text(json.dumps(dict(protocol_id=plan['protocol_id'],sha256=sha(ROOT/'config/locked_run.json'),training=False),indent=2))
(ROOT/'status.json').write_text(json.dumps(dict(stage='prepared',training=False,protocol_id=plan['protocol_id']),indent=2))
print(json.dumps(dict(root=str(ROOT),protocol_id=plan['protocol_id'],source_initial_sha256=sha(ROOT/'weights/common_initial21.pt'),files=len(code)),indent=2))
