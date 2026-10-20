"""CPU-only setup for a new paired study. Parent directories remain read-only."""
import copy,csv,hashlib,json,shutil,sys,time
from pathlib import Path
import torch
from PIL import Image

ROOT=Path(sys.argv[1]).resolve()
PARENT=Path('/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/polarapp_single_collaboration_continue_20261014')
EXPECTED='3d3b0af2b931725099a56867501d9f98f44e9ec5d7188c4898db02e014698043'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from runtime import atomic_json,sha,identity_matches
from operations.pair_policy import configure,restore,equal,tail_state,fixed_task_hash
torch.set_num_threads(4)
if (ROOT/'budget.json').exists() or (ROOT/'config').exists():
    raise RuntimeError('New setup cannot reset an existing run')
for name in ('controller.json','active_gpu.json'):
    value=json.loads((PARENT/name).read_text())
    if identity_matches(value.get('identity')):raise RuntimeError('Parent is still active')
for folder in ('source_snapshot','vendor','weights','config'):
    shutil.copytree(PARENT/folder,ROOT/folder,ignore=shutil.ignore_patterns('__pycache__'))
if not (ROOT/'modules').exists():shutil.copytree(PARENT/'modules',ROOT/'modules')
for p in (PARENT/'modules').glob('*.py'):
    assert sha(p)==sha(ROOT/'modules'/p.name)
old=json.loads((ROOT/'config/locked_run.json').read_text())
assert old['protocol_id']=='24edbc233bb5276c30fa8b5ba347fc71098424bccab45456cf29d4374e7699d2'
(ROOT/'config/locked_run.json').rename(ROOT/'config/source6000_locked_run.json')
(ROOT/'config/lock_receipt.json').rename(ROOT/'config/source6000_lock_receipt.json')
final=PARENT/'downstream/checkpoints/meta/final.pt'
assert sha(final)==EXPECTED
shutil.copy2(final,ROOT/'weights/resume_parent_6000.pt')
parent=torch.load(final,map_location='cpu',weights_only=False)
assert parent['step']==parent['joint_step']==6000 and parent['meta_steps']==600
config=json.loads((ROOT/'config/prepare.json').read_text())
config=json.loads(json.dumps(config).replace(str(PARENT),str(ROOT)))
config.update(scope='paired_freeze_policy_collaboration_from6000',
              authorization='User: 执行这个; annotation authorizes paired frozen and partial collaboration',
              actual_training_start_sha256=EXPECTED)
config['manifest']=str(ROOT/'config/dataset_manifest_archived.csv')
atomic_json(ROOT/'config/prepare.json',config)
with open(config['manifest']) as stream:rows=[r for r in csv.DictReader(stream) if r['split']=='train']
sizes={}
def crop_check(gen,item,subset):
    path=subset[item['sample_index']]['polar_000']
    if path not in sizes:
        with Image.open(path) as image:sizes[path]=image.size
    width,height=sizes[path];crop=64
    assert item['top']==int(torch.randint(0,height-crop+1,(1,),generator=gen))
    assert item['left']==int(torch.randint(0,width-crop+1,(1,),generator=gen))

def extend(name,subset,end_steps):
    base=json.loads((PARENT/f'config/parent_{name}_schedule.json').read_text())
    current=json.loads((PARENT/f'config/{name}_schedule.json').read_text())
    gen=torch.Generator().manual_seed(base['seed'])
    order=[]
    while len(order)<len(base['items']):order.extend(torch.randperm(len(subset),generator=gen).tolist())
    for i,item in enumerate(base['items']):
        assert item['sample_index']==order[i];crop_check(gen,item,subset)
    remaining=order[len(base['items']):]
    extra=len(current['items'])-len(base['items'])
    while len(remaining)<extra:remaining.extend(torch.randperm(len(subset),generator=gen).tolist())
    for index,item in zip(remaining[:extra],current['items'][len(base['items']):]):
        assert item['sample_index']==index;crop_check(gen,item,subset)
    remaining=remaining[extra:]
    needed=end_steps*2-len(current['items'])
    while len(remaining)<needed:remaining.extend(torch.randperm(len(subset),generator=gen).tolist())
    result=copy.deepcopy(current)
    for index in remaining[:needed]:
        path=subset[index]['polar_000']
        if path not in sizes:
            with Image.open(path) as image:sizes[path]=image.size
        width,height=sizes[path]
        result['items'].append({'sample_index':index,'top':int(torch.randint(0,height-64+1,(1,),generator=gen)),
                               'left':int(torch.randint(0,width-64+1,(1,),generator=gen))})
    result['steps']=end_steps
    assert result['items'][:len(current['items'])]==current['items']
    assert len(result['items'])==end_steps*2
    atomic_json(ROOT/f'config/source6000_{name}_schedule.json',current)
    atomic_json(ROOT/f'config/{name}_schedule.json',result)
    return {'source_items_unchanged':len(current['items']),'new_total_items':len(result['items']),
            'sha256':sha(ROOT/f'config/{name}_schedule.json')}
receipts={}
for name,subset,end in (('joint',rows,9000),
    ('meta_inner',[r for r in rows if r['scene'] in config['meta_inner_scene_ids']],900),
    ('meta_outer',[r for r in rows if r['scene'] in config['meta_outer_scene_ids']],900)):
    receipts[name]=extend(name,subset,end)
references={}
for split in ('validation','test'):
    path=ROOT/f'config/parent_per_image_{split}.csv'
    shutil.copy2(PARENT/f'downstream/metrics/per_image_{split}.csv',path)
    references[split]=sha(path)
context=json.loads((PARENT/'downstream/metrics/evaluation_context.json').read_text())
draft=copy.deepcopy(old)
draft.pop('protocol_id',None);draft.pop('code_sha256',None);draft.pop('budget',None)
draft.update(scope='paired_frozen_vs_partial_collaboration_new_user_authorization',
    arms=['frozen','partial'],joint_steps=9000,meta_updates=900,additional_joint_steps=3000,additional_meta_updates=300,
    resume_parent_protocol_id=old['protocol_id'],resume_parent_final_sha256=EXPECTED,
    optimizer_and_RNG_restored=True,selection='both fixed9000 endpoints; never test or probe selection',
    manifest=config['manifest'],manifest_sha256=sha(config['manifest']),
    source_configuration_sha256=sha(ROOT/'config/prepare.json'),
    schedule_hashes={n:r['sha256'] for n,r in receipts.items()},parent_reference_metrics_sha256=references,
    fixed_validation_probe_ids=context['example_sample_ids'],validation_probe_steps=[6000,7000,8000,9000],
    matched_freeze_policy_collaboration_available=True,matched_finetuning_control_available=False,
    parent_original_run_not_modified=True,new_independent_24h_budget=True,new_budget_reused_from_old=False,
    freeze_policy={'frozen':'unchanged source policy','partial':['net_g.decoder_level1','net_g.output'],
                   'new_tail_lr':1e-6,'meta_inner_tail_lr':1e-6,
                   'output_layer_alignment_inner_update':False,
                   'reason':'RGB output convolution is not used by FA features; task joint update only'},
    comparison_claim='effect of this local-unfreeze policy within collaboration; not collaboration-vs-ordinary training',
    historical_test_information_available_before_user_request=True)
draft.pop('comparison_is_6000_vs_3000',None)
atomic_json(ROOT/'config/draft_run.json',draft)
from modules.single_system import build_system
observations=[]
for arm in ('frozen','partial'):
    up,down,fa,_,_=build_system(ROOT)
    policy=configure(up,down,arm)
    opt,opt_fa=restore(up,down,fa,parent,restore_rng=False)
    assert equal(up.adapters.state_dict(),parent['upstream_adapters'])
    assert equal(down.adapters.state_dict(),parent['downstream_adapters'])
    assert equal(fa.state_dict(),parent['alignment_state'])
    actual=opt.state_dict()
    assert actual['param_groups'][0]==parent['optimizer']['param_groups'][0]
    assert equal({k:actual['state'][k] for k in parent['optimizer']['state']},parent['optimizer']['state'])
    if arm=='partial':
        assert len(actual['param_groups'])==2
        assert not any(k in actual['state'] for k in actual['param_groups'][1]['params'])
    assert equal(opt_fa.state_dict(),parent['optimizer_fa'])
    assert policy['all_priors_frozen']
    observations.append(dict(policy,existing_optimizer_moments_exact=True,new_group_has_no_source_moments=arm=='partial',
                             initial_task_fixed_hash=fixed_task_hash(down)))
    del up,down,fa,opt,opt_fa
atomic_json(ROOT/'cpu_setup_receipt.json',{'CPU_only':True,'source_sha256':EXPECTED,
    'source_model_optimizer_RNG_preserved_in_checkpoint':True,'source_step':6000,'source_meta_step':600,
    'two_policy_initial_model_states_identical':True,'policies':observations,
    'schedule_prefixes':receipts,'old_run_modified':False,'official_and_PIDSR_are_ancestry_only':True})
print(json.dumps({'setup_passed':True,'policies':observations,'schedule_prefixes':receipts},ensure_ascii=False))
