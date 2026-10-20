"""Prepare an independent, read-only-source candidate diagnostic. No GPU calls."""
import hashlib,json,os,shutil,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'polar_gain_judgement_short_20261012'
SOURCE_SHA='279d0a329d0efebee9678c2156bf3e7b60c0646e75a5f37f8015c7b100f9e9a2'
SOURCE_PROTOCOL='8d4f7dd5e12ce21b92c4d3a3413b68da8786c7cb414e55a9ce31600de8fc6ad2'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

if (ROOT/'config/locked_run.json').exists():raise RuntimeError('Run already prepared; refuse overwrite')
for name in ('config','diagnostics','metrics','logs','reports','examples','weights','pretraining/checkpoints'):
    (ROOT/name).mkdir(parents=True,exist_ok=True)
shutil.copytree(SOURCE/'scripts',ROOT/'scripts',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copytree(SOURCE/'source',ROOT/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
cfg=json.loads((SOURCE/'config/locked_run.json').read_text())
assert cfg['protocol_id']==SOURCE_PROTOCOL
shutil.copy2(SOURCE/'config/locked_run.json',ROOT/'config/source_locked_run.json')
cfg['paths']['output_root']=str(ROOT/'downstream')
# The archived dependency validates project_root against its original __file__.
# Keep its original read-only import path; ROOT/source is a reproducible snapshot.
cfg['data']['workers']=0
for name in ('manifest.csv','validation_proxy.json','judgement_schedule.json','base_server.yaml'):
    shutil.copy2(SOURCE/'config'/name,ROOT/'config'/name)
# Loader/base_config consistency is retained; read-only paths still point to SOURCE.
write(ROOT/'config/execution_config.json',cfg)
links={ROOT/'weights/source22_final.pt':SOURCE/'downstream/checkpoints/final.pt',
       ROOT/'weights/migration21_final.pt':SOURCE/'weights/reused_final.pt',
       ROOT/'pretraining/checkpoints/selected.pt':SOURCE/'pretraining/checkpoints/selected.pt'}
for new,old in links.items():new.symlink_to(old)
assert sha(ROOT/'weights/source22_final.pt')==SOURCE_SHA
proxy=json.loads((SOURCE/'config/validation_proxy.json').read_text())
plan=dict(schema_version=1,study='frozen all-path quality-to-candidate intervention',
          source_run=str(SOURCE),source_checkpoint=str(ROOT/'weights/source22_final.pt'),
          source_checkpoint_sha256=SOURCE_SHA,source_protocol_id=SOURCE_PROTOCOL,
          conditions=['real_quality','fixed_quality','shifted_quality'],
          fixed_quality='positive global mean of H predictions from first 16 judgement-training crops; C=exp(-b0/tau)',
          fixed_calibration=dict(effective_batches=8,microbatch=1,batch_size=2,crop=128,seed=2026,phase='quality_constant_calibration'),
          shifted_quality='paired torch.roll of b/C on H-native map, floor(H/2),floor(W/2)',
          evaluation=dict(frames=28,source_proxy_sha256=sha(SOURCE/'config/validation_proxy.json'),original_size=True,tiling=False,phase='mechanism_probe',seed=2026),
          shared_first_round=True,all_explicit_quality_candidate_paths_controlled=True,
          candidate_endpoint='U=T1+delta; no final M execution',training_steps=0,test_frames=0,
          gpu_budget_seconds=7200,budget_start='first GPU launch after CPU verification; immutable',
          selection_from_results=False,matched_training_ablation=False,
          limitations=['fixed checkpoint inference dependence','fixed/shift controls may be out of training distribution','28 correlated frames / 8 scenes','no causal claim of matched training benefit'],
          inputs={name:dict(path=str(SOURCE/'config'/name),sha256=sha(SOURCE/'config'/name)) for name in ('locked_run.json','manifest.csv','validation_proxy.json','judgement_schedule.json','base_server.yaml')})
code=[]
for path in sorted((ROOT/'scripts').rglob('*.py')):code.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size))
plan['diagnostic_code_manifest']=code
plan['execution_config_sha256']=sha(ROOT/'config/execution_config.json')
protocol=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
plan['protocol_id']=protocol
write(ROOT/'config/locked_run.json',plan)
write(ROOT/'config/lock_receipt.json',dict(protocol_id=protocol,time_unix=time.time(),configuration_sha256=sha(ROOT/'config/locked_run.json'),source_sha256=SOURCE_SHA))
write(ROOT/'status.json',dict(stage='prepared_cpu_pending',time_unix=time.time(),protocol_id=protocol))
print(json.dumps(dict(prepared=True,protocol_id=protocol,source_checkpoint_verified=True,code_files=len(code)),ensure_ascii=False))
