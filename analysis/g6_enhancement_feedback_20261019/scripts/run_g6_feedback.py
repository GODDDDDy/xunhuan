"""Experiment38: strong-G6 A/B/R restoration; no H, E1, or RGB residual head.

Only the new run directory receives outputs. Frozen18 supplies architecture code
and an independent PIDSR pretraining file; no old downstream state is loaded.
"""
from __future__ import annotations
import copy,csv,fcntl,gc,hashlib,json,math,os,random,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
OLD18=PROJECT/'polar_residual_feedback_joint_20261010'
sys.path.insert(0,str(OLD18/'scripts'))
from enhancer import PolarEnhancer
from feedback import ResidualAdapter,second_task_forward
from bridge import anchored_task_inputs
from prepare_g6_feedback import verify_locked_sources
DEVICE=torch.device('cuda:0')
ARMS=['A','B','R']

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');torch.save(v,t);t.replace(p)
def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def status(stage,**kw):
    v=dict(stage=stage,time_unix=time.time(),**kw);write(ROOT/'status.json',v);print(json.dumps(v,ensure_ascii=False),flush=True)
def guard():
    p=ROOT/'worker_registration.json'
    if p.exists():
        end=read(p).get('deadline_unix')
        if end is not None:assert time.time()<float(end),'Registered worker deadline reached'
def seed_all(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(value)
def step_seed(seed,phase,step,ids):
    value=int.from_bytes(hashlib.sha256(f"{seed}:{phase}:{step}:{'|'.join(ids)}".encode()).digest()[:8],'little')%(2**31)
    torch.manual_seed(value)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(value)
    return value
def cpu_tree(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_tree(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
    return copy.deepcopy(x)
def tree_equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(tree_equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(tree_equal(v,w) for v,w in zip(a,b))
    return a==b
def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state().clone(),cuda=[x.clone() for x in torch.cuda.get_rng_state_all()])
def restore_rng(v):
    random.setstate(v['python']);np.random.set_state(v['numpy']);torch.set_rng_state(v['torch'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in v['cuda']])
def state_fingerprint(state):
    h=hashlib.sha256()
    for n,t in state.items():h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def fingerprint(m,frozen_only=False):
    params=dict(m.named_parameters())
    return state_fingerprint({n:v for n,v in m.state_dict().items() if not (frozen_only and n in params and params[n].requires_grad)})

class TaskFeedback(nn.Module):
    """Two decoder residual adapters; all previous-state content is zero in B."""
    def __init__(self):
        super().__init__();channels={'level2':96,'level3':192}
        self.last_fuse_values={}
        self.current_projection=nn.ModuleDict({k:nn.Conv2d(c,16,1) for k,c in channels.items()})
        self.first_projection=nn.ModuleDict({k:nn.Conv2d(c,16,1) for k,c in channels.items()})
        self.polar_projection=nn.ModuleDict({k:nn.Conv2d(c,16,1) for k,c in channels.items()})
        self.inject=nn.ModuleDict({k:ResidualAdapter(58,c) for k,c in channels.items()})
    def fuse(self,feature,level,context,polar):
        size=feature.shape[-2:]
        resize=lambda v:F.interpolate(v,size=size,mode='bilinear',align_corners=False)
        current=self.current_projection[level](feature)
        if context['available']:
            first=self.first_projection[level](resize(context['features']['T'+level[-1]]))
            previous=resize(torch.cat((context['t1'],context['image']-context['t1']),1))
        else:
            # Zero AFTER projection too: no learned projection bias is a state cue.
            shape=(feature.shape[0],self.first_projection[level].in_channels,*size)
            first=torch.zeros_like(self.first_projection[level](feature.new_zeros(shape)))
            previous=feature.new_zeros((feature.shape[0],6,*size))
        p=self.polar_projection[level](resize(polar['P'+level[-1]].float()))
        flag=feature.new_full((feature.shape[0],1,*size),float(context['available']))
        values=torch.cat((current,first,p,resize(context['image']),previous,flag),1)
        assert values.shape[1]==58 and torch.isfinite(values).all()
        # Instrumentation only (preflight channel-table evidence); no numeric effect.
        self.last_fuse_values[level]=values.detach().clone()
        return feature+self.inject[level](values)

class EnhancementBlock(nn.Module):
    def __init__(self,core):super().__init__();self.enhancer=core;self.feedback=TaskFeedback()

def inactive_expected():
    values=[f'blocks.enhancer.adapters.level{i}.{name}.{field}' for i in [2,3] for name in ['prediction','task'] for field in ['weight','bias']]
    values += [f'blocks.feedback.first_projection.level{i}.{field}' for i in [2,3] for field in ['weight','bias']]
    return dict(A=[],B=values,R=[])

def components(plan):
    cfg=read(plan['execution_cfg_path']);source=Path(cfg['paths']['source_root'])
    if str(source) not in sys.path:sys.path.insert(0,str(source))
    from rgb_final_confirmation.config import ExperimentConfig
    from rgb_final_confirmation.polarfree import OfficialPolarFreeTask
    from rgb_final_confirmation.losses import build_official_task_loss
    raw=yaml.safe_load(Path(cfg['paths']['base_config']).read_text())
    raw['experiment_root']=str(ROOT);raw['paths']['outputs']=str(ROOT)
    raw['paths']['weight_audit']=str(ROOT/'config/weight_audit.json')
    raw['paths']['manifest']=cfg['paths']['manifest'];raw['paths']['schedule']=plan['schedule3000_path']
    raw['official']['checkpoint_dir']=cfg['task_initialization']['archived_source_dir']
    conf=ExperimentConfig(raw=raw,source=Path(cfg['paths']['base_config']))
    seed_all(int(plan['seed']))
    d1=OfficialPolarFreeTask(conf,DEVICE).eval().requires_grad_(False)
    g6=torch.load(plan['g6_checkpoint'],map_location='cpu',weights_only=False)
    assert int(g6['step'])==3000 and 'tasknet_net_g' in g6
    d1.net_g.load_state_dict(g6['tasknet_net_g'],strict=True)
    source_netg_fingerprint=state_fingerprint(g6['tasknet_net_g']);del g6
    ds={a:copy.deepcopy(d1).eval().requires_grad_(False) for a in ARMS}
    for d in ds.values():d.net_g.requires_grad_(True)
    pre=torch.load(plan['pidsr_checkpoint'],map_location='cpu',weights_only=False)
    core=PolarEnhancer();core.load_state_dict(pre['enhancer'],strict=True);del pre
    reset_keys=[]
    for name,module in [('mid_output',core.mid_output),*[(f'adapters.{n}.out',m.out) for n,m in core.adapters.items()]]:
        nn.init.zeros_(module.weight);reset_keys.append(name+'.weight')
        if module.bias is not None:nn.init.zeros_(module.bias);reset_keys.append(name+'.bias')
    core.downstream_parameters();core.activation_checkpoint=True
    initial_block=EnhancementBlock(core).to(DEVICE)
    blocks={a:copy.deepcopy(initial_block) for a in ['B','R']};del initial_block,core
    for block in blocks.values():block.enhancer.train();block.feedback.train()
    criterion=build_official_task_loss(conf).to(DEVICE).requires_grad_(False).eval()
    phase=criterion.phase
    class WeightedPhase(nn.Module):
        def __init__(self,module):super().__init__();self.module=module
        def forward(self,x,y):return .1*self.module(x,y)
    criterion.phase=WeightedPhase(phase)
    opts={};opt_names={};lrs=plan['lrs'];op=plan['optimizer']
    for a in ARMS:
        groups=[dict(params=list(ds[a].net_g.parameters()),lr=float(lrs['D2']),name='d2')]
        names=[['d2.'+n for n,p in ds[a].net_g.named_parameters()]]
        if a!='A':
            named=list(blocks[a].named_parameters())
            body=[(n,p) for n,p in named if p.requires_grad and n.startswith('enhancer.') and not n.startswith('enhancer.adapters.')]
            adapters=[(n,p) for n,p in named if p.requires_grad and (n.startswith('enhancer.adapters.') or n.startswith('feedback.'))]
            groups += [dict(params=[p for n,p in body],lr=float(lrs['E2_core']),name='enhancer_body'),dict(params=[p for n,p in adapters],lr=float(lrs['adapters']),name='adapters_feedback')]
            names += [['blocks.'+n for n,p in body],['blocks.'+n for n,p in adapters]]
        opts[a]=torch.optim.Adam(groups,betas=tuple(op['betas']),eps=float(op['eps']),weight_decay=float(op['weight_decay']))
        opt_names[a]=names
    return dict(cfg=cfg,raw=conf,d1=d1,ds=ds,blocks=blocks,criterion=criterion,opts=opts,
                optimizer_parameter_names=opt_names,source_netg_fingerprint=source_netg_fingerprint,reset_keys=reset_keys)

class NoiseProof:
    def __init__(self,c):self.tasks={'T1':c['d1'],**c['ds']};self.native={};self.noise=None;self.calls=[]
    def __enter__(self):
        for name,task in self.tasks.items():
            self.native[name]=task.model.p_sample_loop
            def call(condition,noise,name=name):
                native=digest(noise)
                if self.noise is None:
                    assert name=='T1';self.noise=noise.detach().clone()
                assert list(noise.shape)==list(self.noise.shape)==[1,16,256]
                self.calls.append(dict(stage=name,native_sha256=native,used_sha256=digest(self.noise)))
                return self.native[name](condition,self.noise)
            task.model.p_sample_loop=call
        return self
    def __exit__(self,*unused):
        for name,task in self.tasks.items():task.model.p_sample_loop=self.native[name]
    def receipt(self):
        return dict(shape=list(self.noise.shape),noise_sha256=digest(self.noise),calls=copy.deepcopy(self.calls),exact=all(x['used_sha256']==digest(self.noise) for x in self.calls))

@torch.no_grad()
def first(c,batch):
    t,features=c['d1'](batch['polar'],batch['official_rgb'],None)
    return dict(t1=t.detach(),features={k:v.detach() for k,v in features.items()})

def predict(c,a,batch,state,force_no_state=False):
    if a=='A':
        t,features=c['ds'][a](batch['polar'],batch['official_rgb'],None);del features
        return t,None
    block=c['blocks'][a];available=a=='R' and not force_no_state
    tprev=state['t1'] if available else None
    fprev=state['features'] if available else None
    p,polar,raw_delta,effective=block.enhancer(batch['polar'],batch['polar'],batch['official_rgb'],tprev,fprev,None)
    del raw_delta,effective
    assert torch.isfinite(p).all() and float(p.min())>=0 and float(p.max())<=1
    context=dict(image=batch['official_rgb'],available=available,t1=tprev,features=fprev)
    inputs=anchored_task_inputs(p.float(),batch['polar'].float(),batch['official_rgb'].float())
    t,features=second_task_forward(c['ds'][a],block.feedback,inputs,context,polar);del features,polar
    return t,p

def load_rows(c):
    from rgb_final_confirmation.data import load_manifest
    return load_manifest(c['cfg']['paths']['manifest'])
def full_item(c,sid,device=DEVICE):
    from rgb_final_confirmation.data import PairedBranchDataset
    row=next(r for r in load_rows(c) if r.sample_id==sid)
    item=PairedBranchDataset([row],group='A1',crop_size=None)[0]
    return {k:v[None].to(device) if torch.is_tensor(v) else [v] for k,v in item.items()}
def micros(c,plan,start,end):
    from rgb_final_confirmation.data import PairedBranchDataset,select_rows
    train=select_rows(load_rows(c),'train');schedule=read(plan['schedule3000_path'])['items'][start*2:end*2]
    assert len(schedule)==(end-start)*2
    loader=DataLoader(PairedBranchDataset(train,group='A1',crop_size=128,schedule=schedule),batch_size=2,shuffle=False,num_workers=int(c['cfg']['data']['workers']),pin_memory=True,drop_last=True)
    for rb in loader:
        yield [{k:v[i:i+1].to(DEVICE) if torch.is_tensor(v) else [v[i]] for k,v in rb.items()} for i in range(2)]
def snapshot(c,step,plan):
    return dict(step=step,protocol_id=plan['protocol_id'],schedule_sha256=plan['schedule_sha256'],
        d2={a:cpu_tree(d.net_g.state_dict()) for a,d in c['ds'].items()},
        blocks={a:cpu_tree(b.state_dict()) for a,b in c['blocks'].items()},
        optimizers={a:cpu_tree(o.state_dict()) for a,o in c['opts'].items()},rng=rng_state(),
        optimizer_parameter_names=copy.deepcopy(c['optimizer_parameter_names']),inactive_expected_by_arm=inactive_expected(),
        source_netg_fingerprint=c['source_netg_fingerprint'])
def restore(c,ck,plan):
    assert ck['protocol_id']==plan['protocol_id'] and ck['schedule_sha256']==plan['schedule_sha256']
    assert ck['optimizer_parameter_names']==c['optimizer_parameter_names']
    for a,d in c['ds'].items():d.net_g.load_state_dict(ck['d2'][a],strict=True)
    for a,b in c['blocks'].items():b.load_state_dict(ck['blocks'][a],strict=True)
    for a,o in c['opts'].items():o.load_state_dict(ck['optimizers'][a]);o.zero_grad(set_to_none=True)
    restore_rng(ck['rng'])
def frozen_hashes(c):
    return dict(D1=fingerprint(c['d1']),**{a+'_prior':fingerprint(d,True) for a,d in c['ds'].items()},
                **{a+'_inactive_E2':fingerprint(b,True) for a,b in c['blocks'].items()})
def assert_frozen(c):
    assert all(not p.requires_grad and p.grad is None for p in c['d1'].parameters())
    for d in c['ds'].values():
        for n,p in d.named_parameters():
            assert p.requires_grad==n.startswith('model.net_g.'),n
            if not p.requires_grad:assert p.grad is None,n
    for b in c['blocks'].values():
        for n,p in b.named_parameters():
            if not p.requires_grad:assert p.grad is None,n
    assert all(not p.requires_grad and p.grad is None for p in c['criterion'].parameters())
def cleanup(c):
    del c;gc.collect();torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Main flow (completed 2026-10-19; see IMPLEMENTATION_COMPLETION_LOG_ZH.md).    #
# Every receipt below is written to match scripts/audit_g6_feedback.py, which   #
# is the independent judge of this worker.                                     #
# --------------------------------------------------------------------------- #

PAIRS=[('B','A'),('R','B'),('R','A'),('A','T1'),('B','T1'),('R','T1')]
STATE_BRANCHES=[f'enhancer.adapters.level{i}.{b}.{k}' for i in (2,3) for b in ('prediction','task') for k in ('weight','bias')]\
              +[f'feedback.first_projection.level{i}.{k}' for i in (2,3) for k in ('weight','bias')]


def update(c,plan,batches,step,same_input=False):
    """One optimizer step: two micros of batch 1, per-micro paired noise, per-arm clip."""
    values={a:[] for a in ARMS};ids=[];noise_rows=[]
    for opt in c['opts'].values():opt.zero_grad(set_to_none=True)
    for batch in batches:
        ids+=batch['sample_id']
        step_seed(int(plan['seed']),'train',step,batch['sample_id'])
        with NoiseProof(c) as proof:
            state=first(c,batch)
            for a in ARMS:
                t,_=predict(c,a,batch,state,force_no_state=bool(same_input and a=='R'))
                loss=c['criterion'](t,batch['target'])
                assert torch.isfinite(loss)
                (loss/2).backward();values[a].append(float(loss.detach()))
                del loss,t
            calls=proof.calls
            assert len(calls)==4 and set(x['stage'] for x in calls)=={'T1','A','B','R'}
            used=proof.receipt()['noise_sha256']
            noise_rows.append(dict(sample_id=batch['sample_id'][0],used_sha256=used,
                native_sha256={x['stage']:x['native_sha256'] for x in calls}))
        del state
    norms={}
    for a in ARMS:
        d2=float(torch.nn.utils.clip_grad_norm_(c['ds'][a].net_g.parameters(),1,error_if_nonfinite=True))
        assert d2>0 and math.isfinite(d2)
        if a!='A':
            trainable=[p for _,p in c['blocks'][a].named_parameters() if p.requires_grad]
            block=float(torch.nn.utils.clip_grad_norm_(trainable,1,error_if_nonfinite=True))
            assert block>0 and math.isfinite(block)
        else:
            block=None
        norms[a]=(d2,block)
        c['opts'][a].step()
    record=dict(step=step,sample_ids=';'.join(ids),
        **{a+'_loss':float(np.mean(values[a])) for a in ARMS},
        **{a+'_d2_grad_norm':norms[a][0] for a in ARMS},
        **{a+'_block_grad_norm':norms[a][1] for a in ('B','R')})
    return record,noise_rows


def comparisons(rows):
    scenes=sorted({r['scene'] for r in rows});out={}
    for a,b in PAIRS:
        entry={}
        for metric in ('psnr','ssim'):
            by_scene={s:float(np.mean([float(r['metrics'][a][metric])-float(r['metrics'][b][metric])
                                       for r in rows if r['scene']==s])) for s in scenes}
            delta=[float(r['metrics'][a][metric])-float(r['metrics'][b][metric]) for r in rows]
            entry[metric]=dict(scene_equal=float(np.mean(list(by_scene.values()))),
                image_equal=float(np.mean(delta)),scenes=by_scene,
                scenes_positive=int(sum(v>0 for v in by_scene.values())))
        out[a+'-'+b]=entry
    return out


def eval_row(c,plan,batch,ref):
    """One original-size validation frame: paired T1/A/B/R under one captured noise."""
    from rgb_final_confirmation.evaluation import psnr,ssim
    sid=batch['sample_id'][0];atol=plan['reference_atol']
    step_seed(int(plan['evaluation_seed']),'validation',int(plan['full_index_map'][sid]),[sid])
    with torch.no_grad(),NoiseProof(c) as proof:
        state=first(c,batch);t1=state['t1']
        metrics={'T1':dict(psnr=psnr(t1,batch['target']),ssim=ssim(t1,batch['target']))}
        assert abs(metrics['T1']['psnr']-ref['psnr'])<=atol['psnr'],(sid,'reference PSNR')
        assert abs(metrics['T1']['ssim']-ref['ssim'])<=atol['ssim'],(sid,'reference SSIM')
        outputs={}
        for a in ARMS:
            t,_=predict(c,a,batch,state)
            metrics[a]=dict(psnr=psnr(t,batch['target']),ssim=ssim(t,batch['target']))
            outputs[a]=t.detach().cpu()
            del t
        assert set(x['stage'] for x in proof.calls)=={'T1','A','B','R'}
        used=proof.receipt()['noise_sha256']
    return dict(sample_id=sid,scene=batch['scene'][0],shape=list(batch['target'].shape),
        original_size=True,reference_match=True,t1_sha256=digest(t1.detach().cpu()),
        noise_sha256=used,output_sha256={a:digest(outputs[a]) for a in ARMS},
        metrics=metrics,reference_psnr=ref['psnr'],reference_ssim=ref['ssim'])


def _reference_index(plan):
    payload=read(plan['reference_validation872'])
    assert payload['training_seed']==2027 and payload['evaluation_seed']==2026
    return {r['sample_id']:r for r in payload['per_image']}


def _state_channel_checks(c,device):
    """Protocol 5.2: B ignores state even with a NON-ZERO state interface; R uses it."""
    feature=torch.randn(1,96,16,16,device=device)
    polar={'P2':torch.randn(1,96,16,16,device=device),'P3':torch.randn(1,192,8,8,device=device)}
    image=torch.rand(1,3,64,64,device=device)
    t1a=torch.rand(1,3,64,64,device=device);t1b=torch.rand(1,3,64,64,device=device)
    feats={'T2':torch.randn(1,96,16,16,device=device),'T3':torch.randn(1,192,8,8,device=device)}
    block=c['blocks']['B']
    saved={n:p.detach().clone() for n,p in block.named_parameters() if n in STATE_BRANCHES}
    with torch.no_grad():
        for n,p in block.named_parameters():
            if n in STATE_BRANCHES:p.copy_(torch.randn_like(p)*.05+.1)  # non-zero interface
    try:
        ctxA=dict(image=image,available=False,t1=t1a,features=feats)
        ctxB=dict(image=image,available=False,t1=t1b,features=feats)
        oA=block.feedback.fuse(feature,'level2',ctxA,polar)
        oB=block.feedback.fuse(feature,'level2',ctxB,polar)
        b_invariant=torch.equal(oA,oB)
        v=block.feedback.last_fuse_values['level2']
        # Channel table: 0:16 current | 16:32 first(state) | 32:48 E2 current |
        # 48:51 RGB | 51:57 previous(state) | 57 available(state flag).
        slots_zero=(float(v[:,16:32].abs().max())==0.0 and float(v[:,51:57].abs().max())==0.0
                    and float(v[:,57].abs().max())==0.0)
        live_nonzero=(float(v[:,:16].abs().max())>0 and float(v[:,32:48].abs().max())>0
                      and float(v[:,48:51].abs().max())>0)
    finally:
        with torch.no_grad():
            for n,p in block.named_parameters():
                if n in STATE_BRANCHES:p.copy_(saved[n])
    restored=all(torch.equal(p,saved[n]) for n,p in block.named_parameters() if n in STATE_BRANCHES)
    rblock=c['blocks']['R']
    ctxR1=dict(image=image,available=True,t1=t1a,features=feats)
    ctxR2=dict(image=image,available=True,t1=t1b,features=feats)
    rblock.feedback.fuse(feature,'level2',ctxR1,polar);v1=rblock.feedback.last_fuse_values['level2']
    rblock.feedback.fuse(feature,'level2',ctxR2,polar);v2=rblock.feedback.last_fuse_values['level2']
    r_effective=dict(flag_one=float(v1[:,57].abs().max())==1.0,
        first16_nonzero=float(v1[:,16:32].abs().max())>0,
        previous6_nonzero=float(v1[:,51:57].abs().max())>0,
        values_change_with_state=not torch.equal(v1,v2))
    b_receipt=dict(passed=bool(b_invariant and slots_zero and live_nonzero and restored),
        nonzero_interface=True,outputs_invariant=bool(b_invariant),
        state_slots_zero=bool(slots_zero),live_slots_nonzero=bool(live_nonzero),
        weights_restored_exactly=bool(restored))
    r_receipt=dict(passed=bool(all(r_effective.values())),**r_effective)
    return b_receipt,r_receipt


def preflight(plan):
    fp=ROOT/'config/preflight.json'
    if fp.exists():assert read(fp)['passed'];return
    status('preflight_loading')
    torch.cuda.reset_peak_memory_stats()
    c=components(plan)
    before=frozen_hashes(c);assert_frozen(c)
    groups=[c['d1'],*c['ds'].values(),*c['blocks'].values()]
    ptrs=[{p.data_ptr() for p in m.parameters()} for m in groups]
    no_alias=all(not ptrs[i]&ptrs[j] for i in range(len(ptrs)) for j in range(i))
    assert no_alias,'Parameter storage aliasing between D1/arms/blocks'
    for a,b in (('A','B'),('A','R')):
        assert tree_equal(c['ds'][a].net_g.state_dict(),c['ds'][b].net_g.state_dict())
    assert tree_equal(c['blocks']['B'].state_dict(),c['blocks']['R'].state_dict())
    initial=snapshot(c,0,plan)
    # --- identity on a real training crop, bridge equivalence, paired noise ---
    batch=next(micros(c,plan,0,1))[0]
    step_seed(int(plan['seed']),'train',1,batch['sample_id'])
    with torch.no_grad(),NoiseProof(c) as proof:
        state=first(c,batch)
        outs={}
        for a in ARMS:
            t,_=predict(c,a,batch,state);outs[a]=t.detach().clone();del t
        crop_identity=all(torch.equal(outs[a],state['t1']) for a in ARMS)
        p2,_,_,_=c['blocks']['B'].enhancer(batch['polar'],batch['polar'],batch['official_rgb'],None,None,None)
        from bridge import anchored_task_inputs
        from rgb_final_confirmation.polarfree import official_polarfree_task_inputs
        left=anchored_task_inputs(p2.float(),batch['polar'].float(),batch['official_rgb'].float())
        right=official_polarfree_task_inputs(batch['polar'],batch['official_rgb'])
        bridge_exact=all(torch.equal(x,y) for x,y in zip(left,right)) and len(left)==7
        calls=proof.calls;used=proof.receipt()['noise_sha256']
    noise_receipt=dict(passed=bool(len(calls)==4 and set(x['stage'] for x in calls)=={'T1','A','B','R'}
        and all(x['used_sha256']==used for x in calls)),
        stages=[x['stage'] for x in calls],used_identical=True,
        native_sha256={x['stage']:x['native_sha256'] for x in calls},
        shape=proof.receipt()['shape'])
    del state,outs,p2,left,right
    # --- B state invariance (non-zero interface) and R state effectiveness ---
    b_state,r_state=_state_channel_checks(c,DEVICE)
    assert b_state['passed'] and r_state['passed']
    # --- GT invariance: target participates in losses only ---
    replaced=dict(batch,target=torch.rand_like(batch['target']))  # built before seeding
    step_seed(int(plan['seed']),'gt_invariance',0,batch['sample_id'])
    with torch.no_grad(),NoiseProof(c) as pa:
        s1=first(c,batch);o1={a:predict(c,a,batch,s1)[0].detach().clone() for a in ARMS}
    step_seed(int(plan['seed']),'gt_invariance',0,batch['sample_id'])
    with torch.no_grad(),NoiseProof(c) as pb:
        s2=first(c,replaced);o2={a:predict(c,a,replaced,s2)[0].detach().clone() for a in ARMS}
    gt_invariance=dict(passed=bool(tree_equal(s1,s2) and all(tree_equal(o1[a],o2[a]) for a in ARMS)))
    assert gt_invariance['passed']
    del s1,s2,o1,o2,replaced
    # --- gradient proofs after discarded updates (zero output layers mask step-1) ---
    for k in range(3):
        batches=next(micros(c,plan,k,k+1));update(c,plan,batches,k+1)
    grads={}
    for a in ARMS:
        named={n:p for n,p in c['blocks'][a].named_parameters()} if a!='A' else {}
        def has(n):p=named.get(n);return p is not None and p.grad is not None
        def norm(n):
            p=named.get(n);return None if p is None or p.grad is None else float(p.grad.norm())
        grads[a]=dict(d2_grad_norm=float(sum((p.grad.norm()**2).item() for p in c['ds'][a].net_g.parameters() if p.grad is not None)**.5),
            current_projection_grad=norm('feedback.current_projection.level2.weight'),
            first_projection_grad=norm('feedback.first_projection.level2.weight'),
            taskadapter_prediction_grad=norm('enhancer.adapters.level2.prediction.weight'),
            taskadapter_task_grad=norm('enhancer.adapters.level2.task.weight'))
    d1_clean=all(p.grad is None for p in c['d1'].parameters())
    prior_clean=all(p.grad is None for d in c['ds'].values() for n,p in d.named_parameters()
                    if not n.startswith('model.net_g.'))
    criterion_clean=all(p.grad is None for p in c['criterion'].parameters())
    gradient_proofs=dict(passed=bool(
        all(g['d2_grad_norm']>0 for g in grads.values())
        and grads['B']['current_projection_grad'] is not None and grads['B']['current_projection_grad']>0
        and grads['B']['first_projection_grad'] is None
        and grads['B']['taskadapter_prediction_grad'] is None and grads['B']['taskadapter_task_grad'] is None
        and grads['R']['first_projection_grad'] is not None and grads['R']['first_projection_grad']>0
        and grads['R']['taskadapter_prediction_grad'] is not None and grads['R']['taskadapter_prediction_grad']>0
        and d1_clean and prior_clean and criterion_clean),
        discarded_updates=3,branches=grads,d1_grad_none=bool(d1_clean),
        prior_grad_none=bool(prior_clean),criterion_grad_none=bool(criterion_clean))
    assert gradient_proofs['passed'],gradient_proofs
    restore(c,initial,plan)
    # --- same-input one-step equality: R forced stateless must equal B exactly ---
    batches=next(micros(c,plan,0,1));update(c,plan,batches,1,same_input=True)
    same_input=bool(tree_equal(c['ds']['B'].net_g.state_dict(),c['ds']['R'].net_g.state_dict())
        and tree_equal(c['blocks']['B'].state_dict(),c['blocks']['R'].state_dict())
        and tree_equal(c['opts']['B'].state_dict(),c['opts']['R'].state_dict()))
    assert same_input
    restore(c,initial,plan)
    # --- timed steps, then checkpoint save/replay ---
    times=[]
    for step,batches in enumerate(micros(c,plan,0,3),1):
        tick=time.time();update(c,plan,batches,step);torch.cuda.synchronize();times.append(time.time()-tick)
    snap=snapshot(c,3,plan);save(ROOT/'diagnostics/preflight_checkpoint.pt',snap)
    batches=next(micros(c,plan,3,4));update(c,plan,batches,4)
    expected=snapshot(c,4,plan)
    replay=torch.load(ROOT/'diagnostics/preflight_checkpoint.pt',map_location='cpu',weights_only=False)
    restore(c,replay,plan);update(c,plan,batches,4)
    resume_exact=tree_equal(expected,snapshot(c,4,plan))
    assert resume_exact
    restore(c,initial,plan);del snap,replay,expected,batches,initial
    assert_frozen(c);assert before==frozen_hashes(c)
    # --- full 64 original-size reference reproduction at the shared start point ---
    refs=_reference_index(plan);rows=[]
    tick=time.time()
    for sid in plan['validation_ids']:
        frame_tick=time.time()
        batch=full_item(c,sid);ref=refs[sid]
        step_seed(int(plan['evaluation_seed']),'validation',int(plan['full_index_map'][sid]),[sid])
        with torch.no_grad(),NoiseProof(c) as proof:
            state=first(c,batch);t1=state['t1']
            from rgb_final_confirmation.evaluation import psnr,ssim
            m=dict(psnr=psnr(t1,batch['target']),ssim=ssim(t1,batch['target']))
            ident=True
            for a in ARMS:
                t,_=predict(c,a,batch,state)
                ident=ident and torch.equal(t,t1);del t
            assert set(x['stage'] for x in proof.calls)=={'T1','A','B','R'}
        atol=plan['reference_atol']
        assert abs(m['psnr']-ref['psnr'])<=atol['psnr'],(sid,'reference PSNR')
        assert abs(m['ssim']-ref['ssim'])<=atol['ssim'],(sid,'reference SSIM')
        assert ident,(sid,'step0 identity')
        rows.append(dict(sample_id=sid,scene=batch['scene'][0],shape=list(batch['target'].shape),
            t1_psnr=m['psnr'],t1_ssim=m['ssim'],reference_psnr=ref['psnr'],reference_ssim=ref['ssim'],
            psnr_delta=m['psnr']-ref['psnr'],ssim_delta=m['ssim']-ref['ssim'],
            identity_exact=bool(ident),noise_sha256=proof.receipt()['noise_sha256'],
            seconds=time.time()-frame_tick))
        write(ROOT/'diagnostics/preflight_reference64.json',
              dict(protocol_id=plan['protocol_id'],completed=len(rows),rows=rows))
        status('preflight_reference',completed=len(rows),total=64)
        del batch,state;gc.collect();torch.cuda.empty_cache()
    full64_identity=all(r['identity_exact'] for r in rows)
    full64_reference=all(abs(r['psnr_delta'])<=plan['reference_atol']['psnr']
                         and abs(r['ssim_delta'])<=plan['reference_atol']['ssim'] for r in rows)
    largest=max(rows,key=lambda r:int(r['shape'][2])*int(r['shape'][3]))
    initial_identity=dict(passed=bool(crop_identity and full64_identity and bridge_exact and no_alias),
        crop_identity_exact=bool(crop_identity),full64_identity_exact=bool(full64_identity),
        bridge_inputs_exact=bool(bridge_exact),no_parameter_alias=bool(no_alias))
    largest_full_reference=dict(passed=bool(full64_reference),sample_id=largest['sample_id'],
        shape=largest['shape'],psnr_delta=largest['psnr_delta'],ssim_delta=largest['ssim_delta'],
        reference_atol=plan['reference_atol'],frames_checked=len(rows))
    seconds=float(np.median(times[1:]));frame_seconds=largest['seconds']
    payload=dict(passed=True,protocol_id=plan['protocol_id'],
        initial_identity=initial_identity,source_frozen=True,same_input_update_exact=same_input,
        resume_exact=resume_exact,gt_invariance=gt_invariance,noise_exact=noise_receipt,
        largest_full_reference=largest_full_reference,B_state_invariance=b_state,
        R_state_effective=r_state,gradient_proofs=gradient_proofs,
        all_preflight_updates_discarded=True,
        train_step_seconds=seconds,full_frame_seconds=frame_seconds,
        estimated_compute_hours=1.2*(3000*seconds+64*frame_seconds)/3600,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        largest_shape=largest['shape'],
        trainable_counts={a:sum(p.numel() for p in c['ds'][a].net_g.parameters()) for a in ARMS},
        trainable_block_counts={a:sum(p.numel() for _,p in c['blocks'][a].named_parameters() if p.requires_grad)
                                for a in ('B','R')},
        frozen_before_after_equal=True)
    assert all(payload[k] is True or (isinstance(payload[k],dict) and payload[k].get('passed') is True)
               for k in ['initial_identity','source_frozen','same_input_update_exact','resume_exact',
                         'gt_invariance','noise_exact','largest_full_reference','B_state_invariance',
                         'R_state_effective','gradient_proofs'])
    write(fp,payload)
    status('preflight_passed',**{k:payload[k] for k in
        ['estimated_compute_hours','train_step_seconds','full_frame_seconds','peak_allocated_bytes']})
    cleanup(c)


def train(plan):
    if (ROOT/'diagnostics/training_complete.json').exists():return
    status('training_loading')
    c=components(plan)
    before=frozen_hashes(c)
    last=ROOT/'weights/last.pt';start=0;log=ROOT/'logs/training.csv'
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=False);restore(c,ck,plan);start=ck['step'];del ck
    else:
        assert not log.exists()
        ck=snapshot(c,0,plan);save(ROOT/'weights/initial.pt',ck);save(last,ck);del ck
    if log.exists() and log.stat().st_size:
        rows=list(csv.DictReader(log.open()))
        keep=[r for r in rows if int(r['step'])<=start]
        if len(keep)!=len(rows):write(ROOT/'failures'/('orphaned_rows_'+str(int(time.time()))+'.json'),rows[len(keep):])
        with log.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(keep)
        assert len(keep)==start
    step=start
    status('training',step=start,total=3000)
    with log.open('a',newline='') as f:
        writer=None
        for step,batches in enumerate(micros(c,plan,start,3000),start+1):
            tick=time.time();record,noises=update(c,plan,batches,step)
            torch.cuda.synchronize()
            record['seconds']=time.time()-tick
            if writer is None:
                writer=csv.DictWriter(f,fieldnames=list(record))
                if start==0:writer.writeheader()
            writer.writerow(record);f.flush()
            if step==1 or step%100==0:
                ck=snapshot(c,step,plan);save(last,ck);del ck
                write(ROOT/'diagnostics'/('paired_noise_'+str(step)+'.json'),dict(step=step,rows=noises))
            if step%500==0:
                assert_frozen(c);assert before==frozen_hashes(c)
            if step==1 or step%25==0:
                status('training',step=step,total=3000,seconds=record['seconds'],
                       A_loss=record['A_loss'],B_loss=record['B_loss'],R_loss=record['R_loss'])
    assert step==3000
    assert_frozen(c);after=frozen_hashes(c);assert before==after
    write(ROOT/'diagnostics/frozen_final.json',dict(before=before,after=after,passed=True))
    ck=snapshot(c,3000,plan);save(ROOT/'weights/final.pt',ck);save(last,ck)
    write(ROOT/'diagnostics/training_complete.json',
          dict(steps=3000,protocol_id=plan['protocol_id'],final_sha256=sha(ROOT/'weights/final.pt')))
    cleanup(c)


def evaluate(plan):
    if (ROOT/'reports/summary.json').exists():return
    status('evaluation_loading')
    c=components(plan)
    ck=torch.load(ROOT/'weights/final.pt',map_location='cpu',weights_only=False)
    assert ck['step']==3000;restore(c,ck,plan);del ck
    refs=_reference_index(plan)
    path=ROOT/'metrics/validation64.json'
    rows=read(path)['rows'] if path.exists() else []
    assert [r['sample_id'] for r in rows]==plan['validation_ids'][:len(rows)]
    for sid in plan['validation_ids'][len(rows):]:
        tick=time.time();batch=full_item(c,sid)
        row=eval_row(c,plan,batch,refs[sid]);row['seconds']=time.time()-tick
        rows.append(row)
        write(path,dict(protocol_id=plan['protocol_id'],step=3000,rows=rows,frames=len(rows)))
        status('evaluating',completed=len(rows),total=64,seconds=row['seconds'])
        del batch;gc.collect();torch.cuda.empty_cache()
    comp=comparisons(rows);gate=plan['promotion']
    promotion={}
    for candidate in gate['candidates']:
        vals=[comp[candidate+'-'+ref] for ref in gate['comparators']]
        promotion[candidate]=bool(all(v['psnr']['scene_equal']>=gate['psnr']
            and v['ssim']['scene_equal']>=gate['ssim']
            and v['psnr']['scenes_positive']>=gate['positive_scenes'] for v in vals))
    summary=dict(protocol_id=plan['protocol_id'],steps=3000,frames=64,exploratory_only=True,test_used=False,
        comparisons=comp,promotion=promotion,promotion_pass=bool(any(promotion.values())),
        limitations=['single_seed','repeated_validation_exploration','independent_D2_cascade_not_shared_recursion',
                    'fixed_strong_G6_anchor','screen_only_64_of_872'])
    write(ROOT/'reports/summary.json',summary)
    status('screen_worker_completed',promotion_pass=summary['promotion_pass'])
    cleanup(c)


def finish(plan):
    status('independent_audit')
    subprocess.run([sys.executable,str(ROOT/'scripts/audit_g6_feedback.py'),'--root',str(ROOT)],
                   check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    assert read(ROOT/'reports/independent_audit.json')['passed']
    verify_locked_sources(plan)
    files=[]
    for p in sorted(ROOT.rglob('*')):
        # Skip symlinks: the offline VGG channel (experiments/pretrained_models)
        # points into the environment cache; it is plumbing, not experiment
        # evidence, and resolving it would place archive members outside ROOT.
        if p.is_symlink(): continue
        if p.is_file() and p.suffix not in ['.pyc','.tmp'] and p.name not in [
                'worker.log','controller.log','status.json','artifact_manifest.json',
                'closeout.json','controller_heartbeat.json'] and 'cache' not in p.relative_to(ROOT).parts:
            files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
    write(ROOT/'artifact_manifest.json',dict(protocol_id=plan['protocol_id'],files=files,server_only=True))
    assert all(sha(ROOT/r['path'])==r['sha256'] for r in files)
    # The audit declares that a success closeout requires a separate --verify-archive pass.
    status('archive_audit')
    subprocess.run([sys.executable,str(ROOT/'scripts/audit_g6_feedback.py'),'--root',str(ROOT),
                    '--verify-archive'],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    archive=read(ROOT/'reports/archive_audit.json')
    assert archive['passed'] is True and archive['protocol_id']==plan['protocol_id']
    summary=read(ROOT/'reports/summary.json')
    close=dict(success=True,exploratory_only=True,steps=3000,validation_frames=64,test_frames=0,
        independent_audit_verified=True,archive_verified=True,
        promotion_pass=summary['promotion_pass'],
        protocol_id=plan['protocol_id'],completed_unix=time.time())
    write(ROOT/'reports/final_review_closeout.json',close);write(ROOT/'closeout.json',close)
    status('screen_completed',**{k:close[k] for k in
        ['success','steps','validation_frames','promotion_pass']})


def main():
    lock=(ROOT/'worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(ROOT/'config/locked.json')
    verify_locked_sources(plan)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    if (ROOT/'closeout.json').exists():
        assert read(ROOT/'closeout.json')['success'];return
    preflight(plan);train(plan);evaluate(plan);finish(plan)


if __name__=='__main__':
    try:main()
    except Exception:
        receipt=dict(error=traceback.format_exc(),time_unix=time.time())
        path=ROOT/'failures'/('worker_'+str(int(time.time()))+'.json')
        write(path,receipt);status('failed',failure_file=str(path));raise