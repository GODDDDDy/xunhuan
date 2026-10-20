"""Independent CPU evidence and arithmetic audit for experiment 38.

Does not instantiate a network, run inference, train, or create a success closeout.
Default writes reports/independent_audit.json. After the worker archives it,
--verify-archive verifies that immutable archive and writes archive_audit.json.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, statistics, time, traceback
from collections import Counter
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ARMS = ['A', 'B', 'R']
PAIRS = [('B','A'),('R','B'),('R','A'),('A','T1'),('B','T1'),('R','T1')]
G6_SHA = '6d4946a5f9963ddb1d142b5f4a97f18d0eedf724769aab4b414997a15a33322f'
E_SHA = 'a4ecebacba5cfc25546320974a9c37f1cc220c685f1318b3cc290ae4a68e9554'
INACTIVE_B = {f'blocks.enhancer.adapters.{level}.{branch}.{kind}'
              for level in ['level2','level3'] for branch in ['prediction','task']
              for kind in ['weight','bias']} | {f'blocks.feedback.first_projection.{level}.{kind}'
              for level in ['level2','level3'] for kind in ['weight','bias']}
# de_reduce_chan2 never participates in the enhancer forward: documented in the
# experiment-32 protocol (confidence_causal_completion_20261017: "排除未用的
# de_reduce_chan2 及解码器未参与前向的FeedForward_P.dwconv") and re-confirmed by
# this run's Adam state (no gradient in any arm; B=12+1 inactive, R=1 inactive).
# It is a dead parameter of the frozen-architecture enhancer, identical in B and R,
# so it is excluded from the trainable-interface expectation for every arm.
KNOWN_DEAD = {'blocks.enhancer.de_reduce_chan2.weight'}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')
    temporary.replace(path)
def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()
def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                       separators=(',',':')).encode()).hexdigest()
def require(condition, message):
    if not condition: raise AssertionError(message)
def is_digest(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path
def finite(value, description):
    value = float(value); require(math.isfinite(value), description)
    return value
def near(actual, expected, description, atol=1e-10):
    require(abs(finite(actual,description)-finite(expected,description)) <= atol, description)
def proof(value, label):
    require(value is True or (isinstance(value,dict) and value.get('passed') is True),
            'Missing/failed proof: '+label)

def verify_plan(root):
    plan = read(root/'config/locked.json'); unsigned = dict(plan)
    pid = unsigned.pop('protocol_id'); require(canonical(unsigned)==pid, 'Protocol hash')
    require(canonical(plan['scientific'])==plan['scientific_id'], 'Scientific hash')
    require(plan['experiment']==38 and plan['arms']==ARMS and plan['steps']==3000, 'Locked arms/endpoint')
    require(plan['seed']==2026 and plan['evaluation_seed']==2026, 'Locked seeds')
    require(plan['crop_size']==128 and plan['effective_batch']==2 and plan['microbatch']==1, 'Crop/batch')
    require(plan['promotion']==dict(psnr=.05,ssim=0.,positive_scenes=6,total_scenes=8,
                                   comparators=['T1','A'],candidates=['B','R']), 'Promotion rule')
    require(plan['lrs']==dict(D2=1e-5,E2_core=2e-5,adapters=1e-4), 'Locked LRs')
    require(plan['optimizer']==dict(kind='Adam',betas=[.9,.999],eps=1e-8,weight_decay=0), 'Optimizer')
    require(sha(resolve(root,plan['g6_checkpoint']))==G6_SHA, 'Strong G6 source changed')
    require(sha(resolve(root,plan['pidsr_checkpoint']))==E_SHA, 'PIDSR source changed')
    require(plan['exploratory_only'] is True, 'Initial screen must remain exploratory')
    return plan

def verify_sources(root, plan):
    sources, codes = plan['sources'], plan['code_sources']
    require(bool(sources) and bool(codes), 'Missing immutable sources')
    source_index = {str(resolve(root,r['path']).resolve()):r for r in sources}
    code_index = {str(resolve(root,r['path']).resolve()):r for r in codes}
    require(len(source_index)==len(sources) and len(code_index)==len(codes), 'Duplicate source path')
    for key in source_index.keys() & code_index.keys():
        require(source_index[key]['sha256']==code_index[key]['sha256'], 'Conflicting dual-role source')
    rp=root/'config/implementation_revisions.json'
    revisions=read(rp) if rp.exists() else []; require(isinstance(revisions,list), 'Revision list')
    replacements={}; preserved=[]
    for revision in revisions:
        require(revision['protocol_id']==plan['protocol_id'] and revision['scientific_id']==plan['scientific_id'], 'Revision authority')
        require(bool(revision['replacements']), 'Empty repair')
        for change in revision['replacements']:
            key=str(resolve(root,change['path']).resolve())
            require(key in code_index and key not in source_index, 'Cannot repair a scientific source')
            old=code_index[key]['sha256']; require(change['original_sha256']==old, 'Repair original')
            backup=resolve(root,change['preserved_original_path']).resolve()
            require(str(backup)!=key and sha(backup)==old, 'Preserved original mismatch')
            require(is_digest(change['repaired_sha256']), 'Bad repaired SHA')
            replacements[key]=change; preserved.append(change)
    output={}
    for label,records in [('sources',sources),('code_sources',codes)]:
        output[label]=[]
        for record in records:
            path=resolve(root,record['path']).resolve(); key=str(path)
            expected=replacements[key]['repaired_sha256'] if key in replacements else record['sha256']
            require(path.is_file() and is_digest(expected), 'Missing source: '+key)
            require(sha(path)==expected, 'Source changed: '+key)
            if key not in replacements: require(path.stat().st_size==record['bytes'], 'Source byte count: '+key)
            output[label].append(dict(path=key,sha256=expected,bytes=path.stat().st_size))
    output['preserved_implementation_repairs']=preserved
    return output

def verify_preflight(root):
    pre=read(root/'config/preflight.json'); proof(pre.get('passed'), 'passed')
    for key in ['initial_identity','source_frozen','same_input_update_exact','resume_exact',
                'gt_invariance','noise_exact','largest_full_reference','B_state_invariance','R_state_effective']:
        proof(pre.get(key), key)
    require(isinstance(pre.get('gradient_proofs'),dict) and bool(pre['gradient_proofs']), 'Gradient evidence missing')
    frozen=read(root/'diagnostics/frozen_final.json'); proof(frozen.get('passed'), 'frozen_final')
    require(isinstance(frozen.get('before'),dict) and bool(frozen['before']), 'Frozen fingerprints missing')
    require(frozen['before']==frozen.get('after'), 'Fixed G6/prior changed')
    for key,value in frozen['before'].items(): require(is_digest(value), 'Bad frozen fingerprint: '+key)
    return dict(preflight=pre,frozen_final=frozen,preflight_sha256=sha(root/'config/preflight.json'))

def verify_training(root, plan):
    with (root/'logs/training.csv').open(encoding='utf-8-sig',newline='') as handle:
        rows=list(csv.DictReader(handle))
    require([int(r['step']) for r in rows]==list(range(1,3001)), 'Continuous 1..3000 logs required')
    for row in rows:
        require(finite(row['seconds'],'seconds')>=0, 'Negative step time')
        for arm in ARMS:
            require(finite(row[arm+'_loss'],arm+' loss')>=0, 'Negative training loss')
            require(finite(row[arm+'_d2_grad_norm'],arm+' D2 gradient')>0, 'No D2 gradient')
        for arm in ['B','R']:
            require(finite(row[arm+'_block_grad_norm'],arm+' block gradient')>0, 'No block gradient')
    sp=resolve(root,plan['schedule3000_path']); require(sha(sp)==plan['schedule_sha256'], 'Schedule SHA')
    schedule=read(sp); require(schedule['total_steps']==3000 and schedule['batch_size']==2 and schedule['crop_size']==128, 'Schedule dimensions')
    items=schedule['items']; require(len(items)==6000, 'Need 6000 presentations')
    cfg=read(resolve(root,plan['execution_cfg_path'])); mp=resolve(root,cfg['paths']['manifest'])
    with mp.open(encoding='utf-8-sig',newline='') as handle: manifest=list(csv.DictReader(handle))
    train=[r for r in manifest if r['split']=='train']; require(len(train)==5440, 'Training split count')
    expected=[]
    for item in items:
        index=int(item['sample_index']); require(0<=index<len(train), 'Sample index outside train')
        sid=train[index]['sample_id']; require(sid.startswith('train/'), 'Training split leak')
        if 'sample_id' in item: require(sid==item['sample_id'], 'Schedule ID/index mismatch')
        require(int(item['top'])>=0 and int(item['left'])>=0, 'Negative ROI')
        expected.append(sid)
    require([r['sample_ids'] for r in rows]==[';'.join(expected[i:i+2]) for i in range(0,6000,2)], 'Training order differs from lock')
    complete=read(root/'diagnostics/training_complete.json')
    require(complete['steps']==3000 and complete['protocol_id']==plan['protocol_id'], 'Training complete receipt')
    require(complete['final_sha256']==sha(root/'weights/final.pt'), 'Training final weight SHA')
    return dict(steps=3000,presentations=6000,unique_training_images=len(set(expected)),
                log_sha256=sha(root/'logs/training.csv'),schedule_sha256=sha(sp),
                finite_losses_and_nonzero_gradients=True)

def same_tree(torch,a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(same_tree(torch,v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)): return type(a) is type(b) and len(a)==len(b) and all(same_tree(torch,x,y) for x,y in zip(a,b))
    try:
        import numpy as np
        if isinstance(a,np.ndarray): return isinstance(b,np.ndarray) and np.array_equal(a,b)
    except ImportError: pass
    return a==b

def verify_checkpoints(root,plan):
    import torch
    require(not torch.cuda.is_available(), 'CPU audit only'); torch.set_num_threads(2)
    initial=torch.load(root/'weights/initial.pt',map_location='cpu',weights_only=False)
    final=torch.load(root/'weights/final.pt',map_location='cpu',weights_only=False)
    for ck,step in [(initial,0),(final,3000)]:
        require(ck['step']==step and ck['protocol_id']==plan['protocol_id'], 'Checkpoint step/protocol')
        require(ck['schedule_sha256']==plan['schedule_sha256'], 'Checkpoint schedule')
        require(set(ck['d2'])==set(ARMS) and set(ck['blocks'])=={'B','R'}, 'Checkpoint arm coverage')
        require(set(ck['optimizers'])==set(ARMS) and set(ck['optimizer_parameter_names'])==set(ARMS), 'Checkpoint optimizer coverage')
        require(isinstance(ck.get('rng'),dict) and bool(ck['rng']), 'Missing checkpoint RNG')
    source=torch.load(resolve(root,plan['g6_checkpoint']),map_location='cpu',weights_only=False)['tasknet_net_g']
    for arm in ARMS: require(same_tree(torch,initial['d2'][arm],source), arm+' initial D2 is not G6')
    require(same_tree(torch,initial['blocks']['B'],initial['blocks']['R']), 'B/R blocks not same initial weights')
    esource=torch.load(resolve(root,plan['pidsr_checkpoint']),map_location='cpu',weights_only=False)['enhancer']
    for name,value in esource.items():
        key='enhancer.'+name; require(key in initial['blocks']['B'], 'Missing PIDSR tensor: '+key)
        reset=name.startswith('mid_output.') or (name.startswith(('adapters.level2.out.','adapters.level3.out.')))
        expected=torch.zeros_like(value) if reset else value
        require(torch.equal(initial['blocks']['B'][key],expected), 'E2 source/reset mismatch: '+name)
    counts={}; lr_map={'d2':plan['lrs']['D2'],'enhancer_body':plan['lrs']['E2_core'],'adapters_feedback':plan['lrs']['adapters']}
    for arm in ARMS:
        require(initial['optimizer_parameter_names'][arm]==final['optimizer_parameter_names'][arm], 'Optimizer name map changed')
        states={}
        for part in ['d2','blocks']:
            if arm not in initial[part]: continue
            before,after=initial[part][arm],final[part][arm]; require(before.keys()==after.keys(), 'State keys changed')
            changed=[]
            for name,value in after.items():
                require(value.device.type=='cpu' and value.shape==before[name].shape and value.dtype==before[name].dtype, 'State dtype/shape')
                if value.is_floating_point() or value.is_complex(): require(bool(torch.isfinite(value).all()), 'Nonfinite trained tensor')
                if not torch.equal(value,before[name]): changed.append(name)
            require(bool(changed), arm+'/'+part+' did not update'); states[part]=dict(changed_names=changed,tensors=len(before))
        oldopt,opt=initial['optimizers'][arm],final['optimizers'][arm]
        require(not oldopt['state'] and bool(opt['state']), 'Fresh/final Adam evidence')
        maps=final['optimizer_parameter_names'][arm]
        require(len(maps)==len(opt['param_groups'])==len(oldopt['param_groups']), 'Optimizer group map')
        seen=set(); inactive=[]
        for oldgroup,group,names in zip(oldopt['param_groups'],opt['param_groups'],maps):
            require(group['name'] in lr_map and len(names)==len(group['params']), 'Optimizer names mismatch')
            require(oldgroup['params']==group['params'], 'Optimizer parameter registration changed')
            require(float(group['lr'])==float(lr_map[group['name']])==float(oldgroup['lr']), 'Optimizer LR changed')
            require(tuple(group['betas'])==(.9,.999) and float(group['eps'])==1e-8 and float(group['weight_decay'])==0, 'Adam settings changed')
            for parameter_id,name in zip(group['params'],names):
                require(name not in seen, 'Duplicate optimizer name'); seen.add(name)
                part,local=name.split('.',1); require(part in ['d2','blocks'] and local in final[part][arm], 'Optimizer includes unknown parameter')
                if parameter_id not in opt['state']:
                    require((arm=='B' and name in INACTIVE_B) or name in KNOWN_DEAD,
                            'Unexpected inactive parameter: '+arm+'/'+name)
                    inactive.append(name); continue
                state=opt['state'][parameter_id]; require(int(state['step'])==3000, 'Adam did not reach fixed step')
                for field in ['exp_avg','exp_avg_sq']:
                    value=state[field]; require(value.device.type=='cpu' and bool(torch.isfinite(value).all()), 'Nonfinite Adam state')
                    require(value.shape==final[part][arm][local].shape, 'Adam tensor shape')
                require(bool((state['exp_avg_sq']>=0).all()), 'Negative Adam variance')
        if arm=='B': require(set(inactive)==INACTIVE_B|KNOWN_DEAD, 'B inactive branch list differs from registered interface')
        if arm=='R': require(set(inactive)==KNOWN_DEAD, 'R inactive list differs from known dead set')
        if arm=='A': require(not inactive, 'A registers only D2; no inactive parameter expected')
        counts[arm]=dict(states=states,optimizer_parameters=len(seen),inactive=inactive)
    last=torch.load(root/'weights/last.pt',map_location='cpu',weights_only=False)
    require(same_tree(torch,last,final), 'last/final states differ')
    return dict(initial_sha256=sha(root/'weights/initial.pt'),final_sha256=sha(root/'weights/final.pt'),
                last_sha256=sha(root/'weights/last.pt'),same_G6_initialization=True,
                same_B_R_initial_blocks=True,last_final_equal=True,arms=counts)

def pair_summary(rows,left,right):
    out={}; scenes=sorted({r['scene'] for r in rows})
    for metric in ['psnr','ssim']:
        by_scene={s:statistics.mean(float(r['metrics'][left][metric])-float(r['metrics'][right][metric]) for r in rows if r['scene']==s) for s in scenes}
        delta=[float(r['metrics'][left][metric])-float(r['metrics'][right][metric]) for r in rows]
        out[metric]=dict(scene_equal=statistics.mean(by_scene.values()),image_equal=statistics.mean(delta),
                         scenes=by_scene,scenes_positive=sum(v>0 for v in by_scene.values()))
    return out

def verify_summary(actual,expected,label):
    for metric in ['psnr','ssim']:
        for key in ['scene_equal','image_equal']: near(actual[metric][key],expected[metric][key],label+'/'+metric+'/'+key)
        require(actual[metric]['scenes_positive']==expected[metric]['scenes_positive'],label+' positive scene count')
        require(actual[metric]['scenes'].keys()==expected[metric]['scenes'].keys(),label+' scene IDs')
        for scene,value in expected[metric]['scenes'].items(): near(actual[metric]['scenes'][scene],value,label+'/'+metric+'/'+scene)

def verify_metrics(root,plan):
    payload=read(root/'metrics/validation64.json'); rows=payload['rows']; ids=plan['validation_ids']
    require(payload['protocol_id']==plan['protocol_id'] and payload['step']==3000, 'Evaluation endpoint/protocol')
    require(len(ids)==len(set(ids))==len(rows)==64 and [r['sample_id'] for r in rows]==ids, 'Evaluation IDs/order')
    counts=Counter(r['scene'] for r in rows); require(len(counts)==8 and set(counts.values())=={8}, '8 scenes x8 required')
    reference=read(resolve(root,plan['reference_validation872']))
    require(reference['training_seed']==2027 and reference['evaluation_seed']==2026, 'G6 reference seed mismatch')
    refs={r['sample_id']:r for r in reference['per_image']}; require(len(refs)==872, 'Original G6 reference count')
    require(plan['reference_atol']==dict(psnr=1e-7,ssim=1e-9), 'G6 reference tolerances changed')
    atol_psnr=plan['reference_atol']['psnr']; atol_ssim=plan['reference_atol']['ssim']
    for row in rows:
        sid=row['sample_id']; old=refs[sid]
        require(row['scene']==old['scene'] and int(old['index'])==plan['full_index_map'][sid], 'Reference scene/index')
        require(row.get('original_size') is True and row.get('reference_match') is True, 'Reference/full-size proof')
        require(len(row['shape'])==4 and row['shape'][0:2]==[1,3] and min(row['shape'][-2:])>=128, 'Original shape')
        require(is_digest(row['t1_sha256']) and is_digest(row['noise_sha256']), 'T1/noise digest')
        require(set(row['output_sha256'])==set(ARMS), 'Output arm digests')
        for value in row['output_sha256'].values(): require(is_digest(value), 'Output digest')
        require(set(row['metrics'])=={'T1','A','B','R'}, 'Metric arms')
        for arm in ['T1']+ARMS:
            for metric in ['psnr','ssim']: finite(row['metrics'][arm][metric],sid+'/'+arm+'/'+metric)
        near(row['metrics']['T1']['psnr'],old['psnr'],sid+'/G6 PSNR',atol_psnr)
        near(row['metrics']['T1']['ssim'],old['ssim'],sid+'/G6 SSIM',atol_ssim)
        near(row['reference_psnr'],old['psnr'],sid+'/saved reference PSNR')
        near(row['reference_ssim'],old['ssim'],sid+'/saved reference SSIM')
    summary=read(root/'reports/summary.json')
    require(summary['protocol_id']==plan['protocol_id'] and summary['steps']==3000 and summary['frames']==64, 'Summary metadata')
    recomputed={a+'-'+b:pair_summary(rows,a,b) for a,b in PAIRS}
    for label,value in recomputed.items(): verify_summary(summary['comparisons'][label],value,label)
    gate=plan['promotion']; promotion={}
    for candidate in gate['candidates']:
        values=[recomputed[candidate+'-'+ref] for ref in gate['comparators']]
        promotion[candidate]=all(v['psnr']['scene_equal']>=gate['psnr'] and v['ssim']['scene_equal']>=gate['ssim'] and v['psnr']['scenes_positive']>=gate['positive_scenes'] for v in values)
    require(summary['promotion']==promotion and summary['promotion_pass'] is any(promotion.values()), 'Promotion mismatch')
    negative={label:{metric:{s:v for s,v in comp[metric]['scenes'].items() if v<0} for metric in ['psnr','ssim']} for label,comp in recomputed.items()}
    return dict(frames=64,test_frames=0,scene_counts=dict(counts),comparisons=recomputed,
                negative_scenes=negative,promotion=promotion,promotion_pass=any(promotion.values()),
                G6_reference_metrics_verified=True,G6_reference_pixel_hash_available=False,
                metric_recalculation_scope='Paired summaries from saved metrics; no independent pixel inference',
                reference_metric_atol=dict(psnr=atol_psnr,ssim=atol_ssim))

def verify_archive(root,plan):
    path=root/'artifact_manifest.json'; payload=read(path)
    records=payload['files']; require(isinstance(records,list) and bool(records), 'Empty archive manifest')
    names=set(); total_bytes=0
    for record in records:
        relative=record.get('path',record.get('relative_path')); require(isinstance(relative,str), 'Archive path missing')
        target=(root/relative).resolve(); require(target.is_relative_to(root.resolve()), 'Archive path outside experiment')
        relative=target.relative_to(root.resolve()).as_posix(); require(relative not in names, 'Duplicate archive member')
        names.add(relative); require(target.is_file(), 'Missing archive member: '+relative)
        require(target.stat().st_size==record['bytes'] and sha(target)==record['sha256'], 'Archive bytes/SHA: '+relative)
        total_bytes+=record['bytes']
    needed={'config/locked.json','config/preflight.json','weights/initial.pt','weights/final.pt','weights/last.pt',
            'logs/training.csv','metrics/validation64.json','reports/summary.json','reports/independent_audit.json',
            'diagnostics/frozen_final.json','diagnostics/training_complete.json'}
    require(needed<=names, 'Required evidence absent from archive: '+repr(sorted(needed-names)))
    existing=read(root/'reports/independent_audit.json')
    require(existing['passed'] is True and existing['protocol_id']==plan['protocol_id'], 'Archive lacks a valid scientific audit')
    return dict(passed=True,cpu_only=True,protocol_id=plan['protocol_id'],files=len(records),bytes=total_bytes,
                artifact_manifest_sha256=sha(path),independent_audit_sha256=sha(root/'reports/independent_audit.json'),time_unix=time.time())

def audit(root):
    plan=verify_plan(root); sources=verify_sources(root,plan)
    preflight=verify_preflight(root); training=verify_training(root,plan)
    checkpoints=verify_checkpoints(root,plan); metrics=verify_metrics(root,plan)
    return dict(passed=True,cpu_only=True,experiment=38,completed_exploratory_screen=True,
                formal_confirmation_completed=False,protocol_id=plan['protocol_id'],scientific_id=plan['scientific_id'],
                steps=3000,arms=ARMS,validation=64,test=0,time_unix=time.time(),
                sources=sources,preflight=preflight,training=training,checkpoints=checkpoints,report=metrics,
                locked_sha256=sha(root/'config/locked.json'),audit_script_sha256=sha(Path(__file__)),
                archive_verified=False,limitation='Evidence/hash and paired arithmetic audit; no network inference. Archive requires separate --verify-archive before success closeout.')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--verify-archive',action='store_true')
    args=parser.parse_args(); root=args.root.resolve()
    destination=root/'reports'/('archive_audit.json' if args.verify_archive else 'independent_audit.json')
    try:
        result=verify_archive(root,verify_plan(root)) if args.verify_archive else audit(root)
    except Exception:
        failure=dict(passed=False,cpu_only=True,time_unix=time.time(),traceback=traceback.format_exc(),
                     audit_script_sha256=sha(Path(__file__)))
        write(destination,failure); raise
    write(destination,result)
    print(json.dumps(dict(passed=True,archive=args.verify_archive,receipt=str(destination)),ensure_ascii=False))
if __name__=='__main__': main()