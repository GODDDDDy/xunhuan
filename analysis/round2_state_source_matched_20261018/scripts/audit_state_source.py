"""Independent CPU evidence audit of experiment 37; no inference or training.
All cache hashes/metadata and eight prespecified cached pixel samples are checked.
Saved per-image metrics are aggregated independently, without pixel recomputation.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,statistics,time,traceback
from collections import Counter
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ARMS=['A','B']
PAIRS=['B-A','B-T1','A-T1']
MU=.4459386889966336
TOL=1e-10
PROOFS=['hook_t1_exact','projection_exact','noise_exact','initial_identity_exact',
'source_frozen_exact','same_input_update_exact','resume_exact',
'all_preflight_updates_discarded','largest_full_passed','local_equivalence_passed']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');t.replace(p)
def require(c,s):
    if not c:raise AssertionError(s)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def digest(v):return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def is_digest(v):return isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v)
def finite(v,s):
    v=float(v);require(math.isfinite(v),s+': nonfinite');return v
def near(a,b,s):require(abs(finite(a,s)-finite(b,s))<=TOL,s)
def resolve(r,p):
    p=Path(p);return p if p.is_absolute() else r/p
def canonical(v,k):
    v=dict(v);v.pop(k,None)
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def same_tree(torch,a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same_tree(torch,v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return type(a) is type(b) and len(a)==len(b) and all(same_tree(torch,x,y) for x,y in zip(a,b))
    import numpy as np
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    return a==b
def finite_tensor(torch,v,s):
    require(isinstance(v,torch.Tensor) and v.device.type=='cpu',s+': CPU tensor required')
    if v.is_floating_point() or v.is_complex():require(bool(torch.isfinite(v).all()),s+': nonfinite')
def authority(r):
    p=read(r/'config/proposal.json');l=read(r/'config/locked.json')
    require(p['experiment']==37 and p['exploratory_only'] is True,'Wrong experiment/scope')
    require(p['arms']==l['arms']==ARMS and p['steps']==l['steps']==3000,'Wrong arms/endpoints')
    require(p['scientific_id']==canonical(p,'scientific_id'),'Scientific canonical ID')
    require(l['protocol_id']==canonical(l,'protocol_id'),'Protocol canonical ID')
    require(l['scientific_id']==p['scientific_id'],'Lock scientific ID')
    require(l['proposal_sha256']==sha(r/'config/proposal.json'),'Proposal bytes changed')
    require(l['preflight_sha256']==sha(r/'config/preflight.json'),'Preflight bytes changed')
    require((p['roi_size'],p['halo'],p['alignment'])==(128,8,8),'ROI/halo/grid')
    require(p['effective_batch']==2 and p['microbatch']==1,'Batch protocol')
    require(float(p['lr'])==1e-4 and float(p['mu'])==MU,'LR/constant C')
    require(p['amp'] is False and p['scheduler'] is None,'Precision/scheduler')
    h=p['head'];require(h['input_channels']==47 and h['hidden_channels']==32,'Local widths')
    require(h['radius']==2 and h['receptive_field']==5 and h['amplitude']==.25,'Head receptive field/amplitude')
    require(h['last_layer_zero_initialized'] is True and h['no_spatial_global_operations'] is True,'Head initialization/locality')
    e=p['evaluation'];require(e['validation']==64 and e['test']==0 and e['common_full_T1'] is True
        and e['original_size'] is True and e['tiles'] is False,'Common full evaluation scope')
    t=p['positive_thresholds'];require(t['comparisons']==['B-A','B-T1'] and t['scene_equal_psnr_min']==.05
        and t['scene_equal_ssim_min']==0 and t['positive_psnr_scenes_min']==6 and t['scenes_total']==8,'Promotion rule')
    return p,l
def verify_sources(r,p,l):
    sources,codes=p['source_receipts'],p['code_sources'];require(sources and codes,'Missing source receipts')
    sp={str(resolve(r,x['path']).resolve()) for x in sources}
    ci={str(resolve(r,x['path']).resolve()):x for x in codes}
    require(len(sp)==len(sources) and len(ci)==len(codes),'Duplicate receipt path within category')
    si={str(resolve(r,x['path']).resolve()):x for x in sources}
    for key in sp.intersection(ci):
        require(si[key]['sha256']==ci[key]['sha256'],'Dual-role scientific/code digest differs: '+key)
    rp=r/'config/implementation_revisions.json';revisions=read(rp) if rp.exists() else []
    require(isinstance(revisions,list),'Revision list');changes={};preserved=[]
    for rev in revisions:
        require(rev['scientific_id']==p['scientific_id'] and rev['replacements'],'Revision scientific authority')
        if 'protocol_id' in rev:require(rev['protocol_id']==l['protocol_id'],'Revision optional protocol authority')
        for c in rev['replacements']:
            k=str(resolve(r,c['path']).resolve());require(k in ci and k not in sp,'Only code_sources repairable')
            require(c['original_sha256']==ci[k]['sha256'],'Repair original hash')
            b=resolve(r,c['preserved_original_path']).resolve()
            require(b!=Path(k) and b.is_file() and sha(b)==ci[k]['sha256'],'Original code not preserved')
            require(is_digest(c['repaired_sha256']),'Repair digest');changes[k]=c;preserved.append(c)
    out={}
    for category,rows in [('source_receipts',sources),('code_sources',codes)]:
        out[category]=[]
        for x in rows:
            f=resolve(r,x['path']).resolve();require(f.is_file() and is_digest(x['sha256']),'Missing/invalid source: '+str(f))
            c=changes.get(str(f));expected=c['repaired_sha256'] if c else x['sha256']
            require(sha(f)==expected,'Source checksum differs: '+str(f))
            if not c:require(f.stat().st_size==int(x['bytes']),'Source byte count: '+str(f))
            out[category].append(dict(path=str(f),locked_sha256=x['sha256'],actual_sha256=expected,
                bytes=f.stat().st_size,implementation_repaired=c is not None))
    out['implementation_revisions']=dict(count=len(revisions),preserved_originals=preserved,
        manifest_sha256=sha(rp) if revisions else None)
    return out
def preflight(r,p,l):
    v=read(r/'config/preflight.json');require(v['passed'] is True,'Preflight not passed')
    for k in PROOFS:require(v['proofs'].get(k) is True,'Missing preflight proof: '+k)
    local=v['local_equivalence'];require(isinstance(local,list) and len(local)>=2,'Two equivalence locations required')
    require(len({str(x['location']) for x in local})>=2,'Duplicate equivalence locations')
    for x in local:
        require(x['passed'] is True,'Local equivalence failed')
        for k in ['output_max_abs','loss_abs','gradient_max_abs']:require(finite(x[k],'Equivalence/'+k)>=0,'Negative equivalence difference')
    require(int(v['head_parameters'])==11651,'Local head parameter count')
    require(finite(v['estimated_compute_hours'],'Compute estimate')>0,'Timing estimate')
    f=read(r/'diagnostics/frozen_final.json')
    require(f['passed'] is True and f['all_grad_none'] is True,'Frozen/gradient evidence')
    require(isinstance(f['before'],dict) and f['before'] and f['before']==f['after'],'Frozen fingerprints differ')
    require(all(is_digest(x) for x in f['before'].values()),'Invalid frozen fingerprints')
    cached=read(r/'diagnostics/cache_frozen.json')
    require(cached['passed'] is True and cached['all_grad_none'] is True,'Cache freeze evidence')
    require(cached['before']==cached['after']==f['before'],'Cache/final source fingerprints differ')
    return dict(proof_fields=PROOFS,local_equivalence=local,head_parameters=11651,
        frozen_before=f['before'],frozen_after=f['after'],all_frozen_gradients_none=True,
        numerical_scope='Saved nonzero-head allclose proofs; no independent GPU inference')
def schedule(r,p):
    f=resolve(r,p['schedule3000_path']);require(sha(f)==p['schedule_sha256'],'Schedule checksum')
    s=read(f);require(s['total_steps']==3000 and s['batch_size']==2 and s['crop_size']==128,'Schedule scope')
    require(len(s['items'])==6000,'Schedule item count')
    ip=resolve(r,p['cache_items_path']);require(sha(ip)==p['cache_items_sha256'],'Cache-item checksum')
    data=read(ip);require(data['count']==6000 and data['unique_images']==5440
        and data['schedule_sha256']==p['schedule_sha256'] and data['train_only'] is True,'Cache-items authority')
    require((data['roi_size'],data['halo'],data['alignment'])==(128,8,8),'Prepared geometry')
    items=data['items'];require(len(items)==6000,'Cache item length')
    cfg=read(resolve(r,p['train_cfg_path']));mp=Path(cfg['paths']['manifest'])
    with mp.open(encoding='utf-8-sig',newline='') as h:manifest=list(csv.DictReader(h))
    train=[x for x in manifest if x['split']=='train'];require(len(train)==5440,'Train split count')
    for position,(scheduled,item) in enumerate(zip(s['items'],items)):
        index=int(scheduled['sample_index']);require(0<=index<len(train),'Schedule sample index')
        source=train[index]
        require(item['position']==position and item['step']==position//2+1 and item['micro']==position%2,'Position/step/micro')
        require(item['sample_index']==index and item['sample_id']==source['sample_id']
            and item['scene']==source['scene'] and item['sample_id'].startswith('train/'),'Schedule/cache/train sample mismatch')
        if 'sample_id' in scheduled:require(scheduled['sample_id']==item['sample_id'],'Explicit schedule ID')
        require(item['top']==int(scheduled['top']) and item['left']==int(scheduled['left']),'ROI coordinates changed')
        top,left,height,width=[int(item[k]) for k in ['top','left','height','width']]
        require(height%8==width%8==0 and 0<=top<=height-128 and 0<=left<=width-128,'ROI/full geometry')
        outer=[max(0,((top-8)//8)*8),max(0,((left-8)//8)*8),
            min(height,((top+128+8+7)//8)*8),min(width,((left+128+8+7)//8)*8)]
        require(item['outer']==outer,'Outer halo geometry')
        require(item['inner']==[top-outer[0],left-outer[1],128,128],'Internal supervision geometry')
    require(len({x['sample_id'] for x in items})==5440,'Scheduled unique images')
    return items,dict(schedule_sha256=sha(f),cache_items_sha256=sha(ip),manifest_sha256=sha(mp),
        presentations_per_arm=6000,unique_images=5440,native_ROI_coordinates_preserved=True)
def cache(r,p,l,items):
    import torch
    require(not torch.cuda.is_available(),'Audit must stay CPU-only')
    ip=r/'cache/index.json';index=read(ip)
    require(index['scientific_id']==p['scientific_id'] and index['protocol_id']==l['protocol_id'],'Cache index authority')
    rows=index['rows'];require(index['count']==len(rows)==6000,'Cache row count')
    positions=[i*5999//7 for i in range(8)];samples=[];files=set();groups={};total=0
    for pos,(x,item) in enumerate(zip(rows,items)):
        for k in ['position','step','micro','sample_id','scene','top','left','outer','inner']:
            require(x[k]==item[k],'Cache '+k+' at '+str(pos))
        require(x['raw_observations_equal'] is True,'Raw metadata pairing')
        for k in ['sha256','noise_A_sha256','noise_B_sha256','raw_sha256','gt_sha256',
                  'native_t1_sha256','full_t1crop_sha256','group_full_t1_sha256']:
            require(is_digest(x[k]),'Invalid cache digest '+k)
        require(x['noise_A_sha256']==x['noise_B_sha256'],'Actual A/B noise differs')
        shape=x['full_shape'];require(list(shape[-2:])==[item['height'],item['width']],'Full shape differs')
        value=(x['noise_B_sha256'],x['group_full_t1_sha256'],tuple(shape))
        require(groups.setdefault(x['sample_id'],value)==value,'Same image full state/noise differs')
        f=resolve(r,x['file']).resolve();require(f.is_relative_to((r/'cache').resolve()),'Cache file outside cache')
        require(str(f) not in files,'Repeated cache file');files.add(str(f))
        require(f.is_file() and f.stat().st_size==int(x['bytes']),'Cache file/byte count')
        require(sha(f)==x['sha256'],'Cache file checksum at '+str(pos));total+=f.stat().st_size
        if pos in positions:
            v=torch.load(f,map_location='cpu',weights_only=False)
            require(v['position']==pos and v['sample_id']==item['sample_id'] and v['inner']==item['inner'],'Pixel-sampled cache identity')
            require(v['noise_sha256']==x['noise_A_sha256'],'Pixel-sampled actual noise')
            a,b,target=v['A'],v['B'],v['target']
            for k,t in [('A',a),('B',b),('target',target)]:
                finite_tensor(torch,t,'Cache sample/'+k);require(t.dtype==torch.float32,'FP32 cache required')
            y0,x0,y1,x1=item['outer'];require(tuple(a.shape)==tuple(b.shape)==(1,47,y1-y0,x1-x0),'Input tensor shape')
            require(tuple(target.shape)==(1,3,y1-y0,x1-x0),'Target tensor shape')
            require(torch.equal(a[:,:15],b[:,:15]),'Sampled raw observations differ')
            require(torch.equal(a[:,-1:],torch.full_like(a[:,-1:],MU)) and
                torch.equal(b[:,-1:],torch.full_like(b[:,-1:],MU)),'Sampled constant C differs')
            require(digest(a[:,:15])==x['raw_sha256'],'Sampled raw digest')
            require(digest(target)==x['gt_sha256'],'Sampled target digest')
            dy,dx,ih,iw=item['inner']
            require(digest(a[:,27:30,dy:dy+ih,dx:dx+iw])==x['native_t1_sha256'],'Sampled native T1 digest')
            require(digest(b[:,27:30,dy:dy+ih,dx:dx+iw])==x['full_t1crop_sha256'],'Sampled full T1 crop digest')
            samples.append(dict(position=pos,sample_id=item['sample_id'],raw_equal=True,constant_C=True,
                T1_channel_hashes_verified=True,sha256=x['sha256'],shape=list(a.shape)))
            del v,a,b,target
    require(len(groups)==5440 and len(files)==6000,'Cache unique files/images')
    done=read(r/'diagnostics/cache_complete.json')
    require(done['items']==6000 and done['unique_images']==5440 and done['protocol_id']==l['protocol_id']
        and done['index_sha256']==sha(ip) and done['source_frozen'] is True,'Cache completion/freeze receipt')
    return dict(count=6000,unique_images=5440,index_sha256=sha(ip),all_file_hashes_verified=True,
        all_index_geometry_verified=True,all_index_actual_noises_paired=True,same_image_full_state_consistent=True,
        total_bytes=total,prespecified_pixel_sample_positions=positions,pixel_sample_rows=samples,
        pixel_scope='8 prespecified cache files decoded; all6000 file hashes and metadata verified')
def training(r,p,l,items):
    with (r/'logs/training.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    require([int(x['step']) for x in rows]==list(range(1,3001)),'Training is not continuous1..3000')
    for step,x in enumerate(rows,1):
        pair=items[2*(step-1):2*step]
        require(x['sample_ids']==';'.join(i['sample_id'] for i in pair),'Training sample sequence differs')
        require(x['positions']==';'.join(str(i['position']) for i in pair),'Training cache position sequence')
        require(finite(x['seconds'],'Training time')>=0,'Negative training time')
        for a in ARMS:
            require(finite(x[a+'_loss'],a+' loss')>=0,'Negative loss')
            require(finite(x[a+'_grad_norm'],a+' gradient')>0,'Missing actual gradient')
    done=read(r/'diagnostics/training_complete.json')
    require(done['steps']==3000 and done['protocol_id']==l['protocol_id']
        and done['schedule_sha256']==p['schedule_sha256'],'Training complete authority')
    require(done['final_sha256']==sha(r/'weights/final.pt'),'Training weight receipt')
    return dict(steps=3000,complete_sequence_verified=True,schedule_and_cache_positions_match=True,
        finite_positive_gradient_norms=True,training_log_sha256=sha(r/'logs/training.csv'))
def checkpoints(r,p,l):
    import torch
    require(not torch.cuda.is_available(),'Checkpoint audit must stay CPU-only');torch.set_num_threads(2)
    first=r/'weights/initial.pt';last=r/'weights/final.pt'
    initial=torch.load(first,map_location='cpu',weights_only=False)
    final=torch.load(last,map_location='cpu',weights_only=False)
    for ck,step,label in [(initial,0,'initial'),(final,3000,'final')]:
        require(ck['step']==step and ck['protocol_id']==l['protocol_id']
            and ck['schedule_sha256']==p['schedule_sha256'],label+' authority')
        require(set(ck['arms'])==set(ARMS) and set(ck['optimizers'])==set(ARMS),label+' arm set')
        require(isinstance(ck.get('rng'),dict) and ck['rng'],label+' missing RNG')
    require(same_tree(torch,initial['arms']['A'],initial['arms']['B']),'Initial local heads differ')
    details={}
    for arm in ARMS:
        before,after=initial['arms'][arm],final['arms'][arm]
        require(before.keys()==after.keys() and before,arm+' parameter keys')
        count=0;changed=[]
        for k,t in after.items():
            finite_tensor(torch,t,arm+'/'+k);finite_tensor(torch,before[k],'initial/'+arm+'/'+k)
            require(t.shape==before[k].shape and t.dtype==before[k].dtype,arm+' parameter shape/dtype')
            count+=t.numel()
            if not torch.equal(t,before[k]):changed.append(k)
        require(count==11651 and changed,arm+' head parameter count/updates')
        output=[t for t in before.values() if tuple(t.shape)==(3,32,3,3)]
        bias=[t for t in before.values() if tuple(t.shape)==(3,)]
        require(len(output)==len(bias)==1 and bool((output[0]==0).all())
            and bool((bias[0]==0).all()),arm+' output not zero-initialized')
        old,opt=initial['optimizers'][arm],final['optimizers'][arm]
        require(not old['state'] and bool(opt['state']),arm+' fresh/final Adam')
        require(len(old['param_groups'])==len(opt['param_groups']),arm+' optimizer groups')
        ids=[]
        for og,g in zip(old['param_groups'],opt['param_groups']):
            require(og['params']==g['params'],arm+' optimizer range changed')
            require(float(g['lr'])==float(p['lr']) and tuple(g['betas'])==(.9,.999)
                and float(g['eps'])==1e-8 and float(g['weight_decay'])==0,arm+' Adam hyperparameters')
            ids.extend(g['params'])
        require(len(ids)==len(set(ids)) and set(opt['state'])==set(ids),arm+' Adam state/range')
        for k,v in opt['state'].items():
            require(int(v['step'])==3000,arm+' Adam endpoint')
            for name in ['exp_avg','exp_avg_sq']:finite_tensor(torch,v[name],arm+'/Adam/'+name)
            require(bool((v['exp_avg_sq']>=0).all()),arm+' negative Adam variance')
        details[arm]=dict(parameters=count,changed_names=changed,Adam_state_count=len(opt['state']))
    lp=r/'weights/last.pt';require(lp.is_file(),'Missing final resumable checkpoint')
    ck=torch.load(lp,map_location='cpu',weights_only=False)
    require(ck['step']==3000 and ck['protocol_id']==l['protocol_id']
        and ck['schedule_sha256']==p['schedule_sha256'],'Last checkpoint authority')
    for k in ['arms','optimizers','rng']:require(same_tree(torch,ck[k],final[k]),'Last/final differs: '+k)
    return dict(initial_sha256=sha(first),final_sha256=sha(last),last_sha256=sha(lp),
        initial_heads_equal=True,zero_output_initialization_verified=True,last_final_equal=True,arms=details)
def pair_summary(rows,a,b):
    result={};scenes=sorted({x['scene'] for x in rows})
    for m in ['psnr','ssim']:
        values=[float(x['metrics'][a][m])-float(x['metrics'][b][m]) for x in rows]
        by={s:statistics.mean(float(x['metrics'][a][m])-float(x['metrics'][b][m])
            for x in rows if x['scene']==s) for s in scenes}
        result[m]=dict(scene_equal=statistics.mean(by.values()),image_equal=statistics.mean(values),
            scenes=by,scenes_positive=sum(v>0 for v in by.values()))
    return result
def verify_comparison(actual,expected,label):
    for m in ['psnr','ssim']:
        for k in ['scene_equal','image_equal']:near(actual[m][k],expected[m][k],label+'/'+m+'/'+k)
        require(actual[m]['scenes_positive']==expected[m]['scenes_positive'],label+' positive-scene count')
        require(actual[m]['scenes'].keys()==expected[m]['scenes'].keys(),label+' scene keys')
        for s,v in expected[m]['scenes'].items():near(actual[m]['scenes'][s],v,label+'/'+m+'/'+s)
def gate(c):return c['psnr']['scene_equal']>=.05 and c['ssim']['scene_equal']>=0 and c['psnr']['scenes_positive']>=6
def metrics(r,p,l):
    data=read(r/'metrics/validation64.json')
    require(data['protocol_id']==l['protocol_id'] and data['step']==3000,'Validation authority')
    require(data['original_size'] is True and data['tiling'] is False,'Full original evaluation')
    rows,ids=data['rows'],p['validation_ids']
    require(len(ids)==len(set(ids))==len(rows)==64 and [x['sample_id'] for x in rows]==ids,'Validation count/order/list')
    scenes=Counter(x['scene'] for x in rows);require(len(scenes)==8 and set(scenes.values())=={8},'Validation8x8')
    old_rows=read(resolve(r,p['reference_metrics_path']))['rows'];require(len(old_rows)==872,'Old872 reference required')
    refs={x['sample_id']:x for x in old_rows};require(len(refs)==872,'Duplicate reference IDs')
    for x in rows:
        sid=x['sample_id'];require(sid in refs,'Sample outside original reference');old=refs[sid]
        require(x['scene']==old['scene'] and x['shape']==old['shape'],sid+' scene/full shape')
        require(x['reference_exact'] is True and x['original_size'] is True,sid+' missing original/reference proof')
        require(x['t1_sha256']==old['t1_sha256'],sid+' changed full T1')
        for k in ['t1_sha256','noise_sha256','input_sha256']:require(is_digest(x[k]),sid+' invalid '+k)
        require(set(x['metrics'])=={'T1','A','B'},sid+' metric arm set')
        for m in ['psnr','ssim']:
            for a in ['T1','A','B']:finite(x['metrics'][a][m],sid+'/'+a+'/'+m)
            require(float(x['metrics']['T1'][m])==float(old['metrics']['T1'][m]),sid+' exact old33 T1 metric differs/'+m)
    summary=read(r/'reports/summary.json')
    require(summary['protocol_id']==l['protocol_id'] and summary['steps']==3000 and summary['frames']==64
        and summary['exploratory_only'] is True and summary['test_used'] is False,'Summary scope/authority')
    require(set(summary['comparisons'])==set(PAIRS),'Unexpected/missing comparisons')
    computed={}
    for label in PAIRS:
        a,b=label.split('-');computed[label]=pair_summary(rows,a,b)
        verify_comparison(summary['comparisons'][label],computed[label],label)
    promotion=bool(gate(computed['B-A']) and gate(computed['B-T1']))
    require(summary['promotion_pass'] is promotion,'Promotion disagrees with fixed rule')
    negatives={label:{m:{s:v for s,v in values[m]['scenes'].items() if v<0}
        for m in ['psnr','ssim']} for label,values in computed.items()}
    return dict(frames=64,test_frames=0,scene_counts=dict(scenes),comparisons=computed,promotion_pass=promotion,
        negative_scenes=negatives,all64_old_T1_hashes_and_metrics_exact=True,original_full_size_verified=True,
        metrics_sha256=sha(r/'metrics/validation64.json'),summary_sha256=sha(r/'reports/summary.json'),
        metric_scope='Independent paired arithmetic from saved per-image metrics; no pixel metric recomputation')
def audit(r):
    p,l=authority(r);sources=verify_sources(r,p,l);pre=preflight(r,p,l)
    items,sch=schedule(r,p);cached=cache(r,p,l,items);trained=training(r,p,l,items)
    weights=checkpoints(r,p,l);report=metrics(r,p,l)
    return dict(passed=True,completed_exploratory_screen=True,formal_confirmation_completed=False,
        cpu_only=True,time_unix=time.time(),experiment=37,scientific_id=p['scientific_id'],protocol_id=l['protocol_id'],
        steps=3000,arms=ARMS,validation=64,test=0,
        checks=dict(authority=True,sources=True,preflight=True,schedule=True,cache=True,training=True,checkpoints=True,metrics=True),
        sources=sources,preflight=pre,schedule=sch,cache=cached,training=trained,checkpoints=weights,report=report,
        differences=report['comparisons'],promotion_pass=report['promotion_pass'],
        proposal_sha256=sha(r/'config/proposal.json'),locked_sha256=sha(r/'config/locked.json'),
        audit_script_sha256=sha(Path(__file__)),
        limitation='Evidence/hash and arithmetic audit; eight fixed cache pixel samples; no independent model inference.')
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args()
    root=args.root.resolve();destination=root/'reports/independent_audit.json'
    try:result=audit(root)
    except Exception:
        result=dict(passed=False,completed_exploratory_screen=False,cpu_only=True,time_unix=time.time(),
            traceback=traceback.format_exc(),audit_script_sha256=sha(Path(__file__)))
        write(destination,result);raise
    write(destination,result)
    print(json.dumps(dict(passed=True,steps=3000,validation=64,test=0,promotion_pass=result['promotion_pass']),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
