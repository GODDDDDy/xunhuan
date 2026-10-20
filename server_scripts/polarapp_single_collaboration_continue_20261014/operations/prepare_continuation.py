"""CPU-only verified continuation setup. Original completed run is never changed."""
import copy
import csv
import hashlib
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

import torch
from PIL import Image

ROOT=Path(sys.argv[1]).resolve()
PARENT=Path('/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/polarapp_single_collaboration_20261013')
sys.path.insert(0,str(ROOT/'scripts'))
from runtime import atomic_json,sha,identity_matches,read_budget
old=json.loads((PARENT/'config/locked_run.json').read_text())
if old['protocol_id']!='451bda754cbe78e5b65a124c0dde9debc2a2072683dc98f06bf097ed19285f65':
    raise RuntimeError('Unexpected completed parent protocol')
parent_budget=read_budget(PARENT)
if (ROOT/'budget.json').exists():raise RuntimeError('Prepared continuation cannot be reset')
for name in ('controller.json','active_gpu.json'):
    value=json.loads((PARENT/name).read_text())
    if identity_matches(value.get('identity')):raise RuntimeError('Parent still running')
for folder in ('modules','source_snapshot','vendor','weights','config'):
    destination=ROOT/folder
    if destination.exists():raise RuntimeError('Setup needs an unused destination')
    shutil.copytree(PARENT/folder,destination,ignore=shutil.ignore_patterns('__pycache__'))
(ROOT/'config/locked_run.json').rename(ROOT/'config/parent_locked_run.json')
(ROOT/'config/lock_receipt.json').rename(ROOT/'config/parent_lock_receipt.json')
final=PARENT/'downstream/checkpoints/meta/final.pt'
if sha(final)!='f7081db214a201a61a0c67300e564c26af394b048b978bb363d6668b10c886b4':raise RuntimeError('Parent final changed')
shutil.copy2(final,ROOT/'weights/resume_parent_3000.pt')
parent=torch.load(final,map_location='cpu',weights_only=False)
assert parent['step']==parent['joint_step']==3000 and parent['meta_steps']==300
assert parent['protocol_id']==old['protocol_id']
config=json.loads((ROOT/'config/prepare.json').read_text())
config=json.loads(json.dumps(config).replace(str(PARENT),str(ROOT)))
config.update(scope='single_round_collaboration_continuation_3000_to_6000',
              authorization='User: 在3000步的基础上，再跑3000步',
              actual_training_start_sha256=sha(final))
config['manifest']=str(ROOT/'config/dataset_manifest_archived.csv')
atomic_json(ROOT/'config/prepare.json',config)
with open(config['manifest']) as stream: rows=[r for r in csv.DictReader(stream) if r['split']=='train']
sizes={}


def extend(schedule,subset,new_steps):
    gen=torch.Generator().manual_seed(schedule['seed'])
    order=[]
    old_count=len(schedule['items'])
    while len(order)<old_count:order.extend(torch.randperm(len(subset),generator=gen).tolist())
    for i,item in enumerate(schedule['items']):
        assert item['sample_index']==order[i]
        path=subset[item['sample_index']]['polar_000']
        if path not in sizes:
            with Image.open(path) as image:sizes[path]=image.size
        width,height=sizes[path];crop=schedule['crop_size']
        assert item['top']==int(torch.randint(0,height-crop+1,(1,),generator=gen))
        assert item['left']==int(torch.randint(0,width-crop+1,(1,),generator=gen))
    remaining=order[old_count:]
    extra=new_steps*2-old_count
    while len(remaining)<extra:remaining.extend(torch.randperm(len(subset),generator=gen).tolist())
    result=copy.deepcopy(schedule)
    for index in remaining[:extra]:
        path=subset[index]['polar_000']
        if path not in sizes:
            with Image.open(path) as image:sizes[path]=image.size
        width,height=sizes[path];crop=schedule['crop_size']
        result['items'].append({'sample_index':index,
            'top':int(torch.randint(0,height-crop+1,(1,),generator=gen)),
            'left':int(torch.randint(0,width-crop+1,(1,),generator=gen))})
    result['steps']=new_steps
    assert result['items'][:old_count]==schedule['items']
    assert len(result['items'])==new_steps*2
    return result


schedule_receipts={}
for name,subset,new_steps in (
    ('joint',rows,6000),
    ('meta_inner',[r for r in rows if r['scene'] in config['meta_inner_scene_ids']],600),
    ('meta_outer',[r for r in rows if r['scene'] in config['meta_outer_scene_ids']],600)):
    path=ROOT/f'config/{name}_schedule.json'
    previous=json.loads(path.read_text())
    expected=[r['sample_id'] for r in subset]
    assert previous['sample_ids']==expected
    atomic_json(ROOT/f'config/parent_{name}_schedule.json',previous)
    expanded=extend(previous,subset,new_steps)
    atomic_json(path,expanded)
    schedule_receipts[name]={'prefix_items_unchanged':len(previous['items']),
        'new_items':len(expanded['items']),'parent_sha256':sha(PARENT/f'config/{name}_schedule.json'),'sha256':sha(path)}
    assert expanded['items'][:len(previous['items'])]==previous['items']
references={}
for split in ('validation','test'):
    path=ROOT/f'config/parent_per_image_{split}.csv'
    shutil.copy2(PARENT/f'downstream/metrics/per_image_{split}.csv',path)
    references[split]=sha(path)
context=json.loads((PARENT/'downstream/metrics/evaluation_context.json').read_text())
budget=copy.deepcopy(parent_budget)
budget.update(root=str(ROOT),run_id=str(uuid.uuid4()),scope='new user-authorized continuation within unchanged parent deadlines',
              parent_root=str(PARENT),parent_budget_sha256=sha(PARENT/'budget.json'),
              original_deadlines_shared_not_new_24h=True)
atomic_json(ROOT/'budget.json',budget)
atomic_json(ROOT/'budget_receipt.json',{'run_id':budget['run_id'],'budget_sha256':sha(ROOT/'budget.json'),
    'created_unix':time.time(),'parent_budget_sha256':sha(PARENT/'budget.json')})
locked=copy.deepcopy(old)
locked.pop('protocol_id')
locked.update(scope='single_round_collaboration_resume_3000_add_3000',joint_steps=6000,meta_updates=600,
    additional_joint_steps=3000,additional_meta_updates=300,resume_parent_protocol_id=old['protocol_id'],
    resume_parent_final_sha256=sha(final),optimizer_and_RNG_restored=True,
    manifest=config['manifest'],manifest_sha256=sha(config['manifest']),
    source_configuration_sha256=sha(ROOT/'config/prepare.json'),budget=budget,
    parent_reference_metrics_sha256=references,fixed_validation_probe_ids=context['example_sample_ids'],
    validation_probe_steps=[3000,4000,5000,6000],validation_probes_used_for_selection=False,
    historical_test_information_available_before_user_request=True,
    selection='fixed total6000 user requested after parent completion; no test checkpoint/parameter selection',
    parent_original_run_not_modified=True)
locked['schedule_hashes']={n:sha(ROOT/f'config/{n}_schedule.json') for n in schedule_receipts}
locked['code_sha256']={p.relative_to(ROOT).as_posix():sha(p)
    for folder in ('scripts','modules','source_snapshot','vendor','recovery','operations')
    for p in sorted((ROOT/folder).rglob('*.py')) if '__pycache__' not in p.parts}
locked['protocol_id']=hashlib.sha256(json.dumps(locked,sort_keys=True,separators=(',',':')).encode()).hexdigest()
atomic_json(ROOT/'config/locked_run.json',locked)
atomic_json(ROOT/'config/lock_receipt.json',{'protocol_id':locked['protocol_id'],
    'locked_sha256':sha(ROOT/'config/locked_run.json'),'locked_at_unix':time.time()})
adopted=copy.deepcopy(parent)
adopted.update(protocol_id=locked['protocol_id'],resume_parent_final_sha256=sha(final),
    selection='verified_parent3000_adoption_for_user_continuation')
target=ROOT/'downstream/checkpoints/meta'
target.mkdir(parents=True)
torch.save(adopted,target/'last.pt')
atomic_json(target/'last.sha256.json',{'sha256':sha(target/'last.pt'),'step':3000,'protocol_id':locked['protocol_id']})
loaded=torch.load(target/'last.pt',map_location='cpu',weights_only=False)


def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


for key in ('upstream_adapters','downstream_adapters','alignment_state','optimizer','optimizer_fa',
            'torch_rng','cuda_rng','python_rng','step','meta_steps'):
    assert equal(parent[key],loaded[key]),key
sys.path.insert(0,str(ROOT))
from modules.single_system import build_system
up,down,fa,_,_=build_system(ROOT)
fa.set_trainable_partition()
up.adapters.load_state_dict(parent['upstream_adapters'],strict=True)
down.adapters.load_state_dict(parent['downstream_adapters'],strict=True)
fa.load_state_dict(parent['alignment_state'],strict=True)
optimizer=torch.optim.Adam([p for m in (up,down) for p in m.parameters() if p.requires_grad],lr=1e-4,eps=1e-8,foreach=False)
optimizer_fa=torch.optim.Adam([p for p in fa.parameters() if p.requires_grad],lr=1e-5,eps=1e-8,foreach=False)
optimizer.load_state_dict(parent['optimizer']);optimizer_fa.load_state_dict(parent['optimizer_fa'])
assert equal(optimizer.state_dict(),parent['optimizer']) and equal(optimizer_fa.state_dict(),parent['optimizer_fa'])
(ROOT/'downstream/logs').mkdir(parents=True)
(ROOT/'downstream/metrics').mkdir(parents=True)
shutil.copy2(PARENT/'downstream/logs/meta_training.csv',ROOT/'downstream/logs/meta_training.csv')
shutil.copy2(PARENT/'downstream/metrics/meta_probe_initial.json',ROOT/'downstream/metrics/meta_probe_initial.json')
shutil.copy2(PARENT/'downstream/metrics/meta_probe_3000.json',ROOT/'downstream/metrics/meta_probe_3000.json')
expected_train=1.5*5431.3895+400
remaining=locked['operational_training_deadline_unix']-time.time()
if expected_train>remaining:raise RuntimeError('Continuation cannot conservatively fit unchanged training cutoff')
receipt={'protocol_id':locked['protocol_id'],'resume_parent_final_sha256':sha(final),
    'model_optimizer_RNG_exact_after_serialization':True,'optimizer_groups_and_states_strictly_restored':True,
    'start_step':3000,'start_meta_steps':300,'end_step':6000,'end_meta_steps':600,
    'all_schedule_prefixes_exactly_preserved':schedule_receipts,'budget':budget,
    'old_run_modified':False,'GPU_preflight_reused_shape_and_algorithm_not_trial_updates':True,
    'CPU_only':True,'expected_conservative_training_seconds':expected_train,'remaining_training_seconds':remaining}
atomic_json(ROOT/'continuation_setup_receipt.json',receipt)
atomic_json(ROOT/'controller_stage_receipts.json',{'run_id':budget['run_id'],'stages':{'preflight':{
    'returncode':0,'gpu':False,'reused_parent_preflight':True,'CPU_setup_receipt_sha256':sha(ROOT/'continuation_setup_receipt.json'),
    'parent_preflight_record':str(PARENT/'diagnostics/gpu_preflight.json'),'budget_sha256':sha(ROOT/'budget.json')}}})
print(json.dumps(receipt,ensure_ascii=False))
