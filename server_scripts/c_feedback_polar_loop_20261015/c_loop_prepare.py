"""Prepare the user-approved three-arm study using the existing research stack."""
import hashlib,json,shutil,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'h_rank_quality_20261015'
EXPECTED='af2367fd9f96871cad3b94230638cf5c75acd3b8f8c186cdf210a37b730e82d8'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

assert not (ROOT/'config/plan.json').exists(),'Never overwrite a prepared study'
for folder in ('config','diagnostics','logs','weights','metrics','examples','reports','pretraining/checkpoints'):
    (ROOT/folder).mkdir(parents=True,exist_ok=True)
shutil.copytree(OLD/'scripts',ROOT/'scripts',dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('__pycache__','*.pyc','c_loop_*.py'))
cfg=json.loads((OLD/'config/execution_config.json').read_text())
cfg['paths']['output_root']=str(ROOT/'downstream');cfg['data'].update(crop_size=128,batch_size=2,workers=0)
for key,name in [('manifest','manifest.csv'),('base_config','base_server.yaml'),('validation_proxy','validation_proxy.json')]:
    shutil.copy2(cfg['paths'][key],ROOT/'config'/name);cfg['paths'][key]=str(ROOT/'config'/name)
for dest,src in [('weights/common_initial21.pt',OLD/'weights/common_initial21.pt'),
                 ('pretraining/checkpoints/selected.pt',OLD/'pretraining/checkpoints/selected.pt')]:
    (ROOT/dest).symlink_to(src.resolve())
assert sha(ROOT/'weights/common_initial21.pt')==EXPECTED
sys.path.insert(0,str(ROOT/'scripts'))
import study
study.base.add_source_path(cfg)
from rgb_final_confirmation.data import load_manifest,select_rows,build_training_schedule
rows=load_manifest(cfg['paths']['manifest']);train=select_rows(rows,'train');val=select_rows(rows,'validation')
assert len(train)==5440 and len(val)==872 and len(select_rows(rows,'test'))==188
# Existing loader indexes the original manifest order. Map the schedule produced
# on sorted IDs back to that order, preserving the approved epoch definition.
ordered=sorted(train,key=lambda r:r.sample_id)
schedule=build_training_schedule(ordered,seed=2026,total_steps=6000,batch_size=2,crop_size=128)
lookup={r.sample_id:i for i,r in enumerate(train)}
for item in schedule['items']:item['sample_index']=lookup[ordered[item['sample_index']].sample_id]
schedule.pop('sha256')
schedule['sha256']=hashlib.sha256(json.dumps(schedule,sort_keys=True,separators=(',',':')).encode()).hexdigest()
write(ROOT/'config/training6000.json',schedule);cfg['paths']['schedule']=str(ROOT/'config/training6000.json')
selected=[];counts={}
for scene in sorted({r.scene for r in val}):
    group=sorted([r.sample_id for r in val if r.scene==scene]);counts[scene]=len(group)
    ix=np.random.default_rng(np.random.SeedSequence([2026,int(scene)])).permutation(len(group))[:32]
    selected.extend(group[int(i)] for i in ix)
ids_sha=hashlib.sha256(('\n'.join(sorted(selected))+'\n').encode()).hexdigest()
assert ids_sha=='5ac70486b177783294e53b4ea2ab07877fd24b28969e0aa5aa732670cb32dcc5'
write(ROOT/'config/validation256.json',dict(sample_ids=selected,scene_counts=counts,ids_sha256=ids_sha))
write(ROOT/'config/execution_config.json',cfg)
plan=dict(kind='fixed-first/H pure-polar second-round C-use matched experiment',
          authorized_by='User explicitly requested starting reviewed v3 on 2026-10-15',
          arms=['A','B','D'],seed=2026,steps=6000,crop=128,effective_batch=2,microbatch=1,
          trainable_per_arm=292896,trainable=['adapters.level2','adapters.level3','mid_output'],
          lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0.,clip=1.,residual_bound=.02,
          loss='raw T2 MSE + .02 * mean(horizontal,vertical gradient MAE)',
          source_initial_sha256=EXPECTED,
          source_initial_protocol='921ae3dc9fe004949efab2afdb434719a11a8268cb80bd44718bc59a9a99787d',
          rules=dict(A='C*Z, C channel',B='Z, C channel',D='Z, per-sample spatial mean(C) channel'),
          frozen='E1, original H, D/prior, all E2 parameters except declared adapters/output; no Q/M/RGB head',
          evaluation=dict(validation=256,test=188,max_validation=872,scene_counts=counts,
                          original_size=True,tiling=False,precision='float32',
                          seed_phase='H_only_pair_eval',checkpoint='fixed6000; no intermediate validation selection'),
          decision=dict(psnr_min=.05,ssim_min=0.,scenes_positive_min=6,
                        sampling_halfwidth_psnr_max=.02,sampling_halfwidth_ssim_max=.0005),
          budget_seconds=28800,conditional_full_validation=True,second_seed=False,
          test_used_for_training_or_selection=False,
          references=['https://arxiv.org/html/2102.02808v1','https://arxiv.org/html/2108.08617v1',
                      'https://www150.statcan.gc.ca/n1/pub/12-001-x/2019002/article/00006/02-eng.htm'])
plan['data_sha256']={k:sha(cfg['paths'][k]) for k in ('manifest','schedule','base_config','validation_proxy')}
plan['execution_config_sha256']=sha(ROOT/'config/execution_config.json')
plan['validation256_sha256']=sha(ROOT/'config/validation256.json')
plan['code_manifest']=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted((ROOT/'scripts').rglob('*.py'))]
plan['protocol_id']=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
write(ROOT/'config/plan.json',plan)
print(json.dumps(dict(prepared=True,protocol_id=plan['protocol_id'],validation_ids_sha256=ids_sha,
                     training_crops=len(schedule['items']),steps_per_arm=6000),ensure_ascii=False))
