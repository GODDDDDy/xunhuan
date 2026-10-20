"""CPU preparation and immutable source verification for authorized experiment38."""
from __future__ import annotations
import copy,csv,hashlib,json,os,sys,yaml
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
G6_SHA='6d4946a5f9963ddb1d142b5f4a97f18d0eedf724769aab4b414997a15a33322f'
E_SHA='a4ecebacba5cfc25546320974a9c37f1cc220c685f1318b3cc290ae4a68e9554'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def receipt(p,role=None):
    p=Path(p).resolve();assert p.is_file(),str(p)
    d=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
    if role:d['role']=role
    return d
def verify_locked_sources(plan):
    unsigned=dict(plan);unsigned.pop('protocol_id')
    assert digest(unsigned)==plan['protocol_id'],'Protocol changed'
    assert digest(plan['scientific'])==plan['scientific_id'],'Scientific ID changed'
    assert plan['experiment']==38 and plan['arms']==['A','B','R'] and plan['steps']==3000
    codes={r['path']:r for r in plan['code_sources']};replacements={}
    rp=ROOT/'config/implementation_revisions.json'
    for revision in (read(rp) if rp.exists() else []):
        assert revision['protocol_id']==plan['protocol_id']
        assert revision['scientific_id']==plan['scientific_id']
        for r in revision['replacements']:
            assert r['path'] in codes,'Only locked code may be repaired'
            assert r['original_sha256']==codes[r['path']]['sha256']
            assert sha(r['preserved_original_path'])==r['original_sha256']
            replacements[r['path']]=r['repaired_sha256']
    for r in plan['sources']+plan['code_sources']:
        assert sha(r['path'])==replacements.get(r['path'],r['sha256']),r['path']
    return True

def main():
    for d in ['config','logs','weights','metrics','reports','diagnostics','failures','cache','examples']:(ROOT/d).mkdir(parents=True,exist_ok=True)
    lp=ROOT/'config/locked.json'
    if lp.exists():
        p=read(lp);verify_locked_sources(p);print(json.dumps(dict(preserved_existing=True,protocol_id=p['protocol_id'])));return
    assert not (ROOT/'worker_registration.json').exists()
    for f in ['run_g6_feedback.py','prepare_g6_feedback.py','controller.py','audit_g6_feedback.py']:assert (ROOT/'scripts'/f).is_file(),f
    protocol=ROOT/'PROTOCOL_ZH.md';assert protocol.is_file()
    old36=PROJECT/'round2_restorer_adaptation_20261018'
    source_cfg=old36/'config/execution_config.json';source_schedule=old36/'config/training3000.json'
    cfg=copy.deepcopy(read(source_cfg));schedule=read(source_schedule)
    assert schedule['total_steps']==3000 and schedule['batch_size']==2 and schedule['crop_size']==128 and len(schedule['items'])==6000
    sp=ROOT/'config/training3000.json';write(sp,schedule)
    with Path(cfg['paths']['manifest']).open(encoding='utf-8-sig',newline='') as f:manifest=list(csv.DictReader(f))
    train=[r for r in manifest if r['split']=='train'];assert len(train)==5440
    for s in schedule['items']:
        assert 0<=int(s['sample_index'])<len(train)
        assert int(s['top'])>=0 and int(s['left'])>=0
        if 'sample_id' in s:assert s['sample_id']==train[int(s['sample_index'])]['sample_id']
    vp=PROJECT/'confidence_correction_causal_20261017/config/validation64.json'
    ids=read(vp)['sample_ids'];assert len(ids)==len(set(ids))==64
    lookup={r['sample_id']:r for r in manifest};counts=Counter(lookup[i]['scene'] for i in ids)
    assert len(counts)==8 and set(counts.values())=={8}
    g6=PROJECT/'normalized_threeway_20261002/runs/G6/seed_2027/checkpoints/step_3000.pt'
    e=PROJECT/'polar_residual_feedback_joint_20261010/pretraining/checkpoints/selected.pt'
    reference=PROJECT/'normalized_threeway_20261002/runs/G6/seed_2027/metrics/step_3000_validation.json'
    assert sha(g6)==G6_SHA and sha(e)==E_SHA
    ref=read(reference);assert ref['training_seed']==2027 and ref['evaluation_seed']==2026 and len(ref['per_image'])==872
    full_index={r['sample_id']:int(r['index']) for r in ref['per_image']}
    assert len(full_index)==872 and all(i in full_index for i in ids)
    scientific=dict(experiment=38,arms=['A','B','R'],steps=3000,seed=2026,evaluation_seed=2026,
        crop_size=128,effective_batch=2,microbatch=1,validation_ids=ids,test_frames=0,
        g6_sha256=G6_SHA,pidsr_sha256=E_SHA,schedule_sha256=sha(sp),reference_atol=dict(psnr=1e-7,ssim=1e-9),
        D1='fixed strong G6 raw official RGB/P0; no E1/H',D2='independent same G6 net_g full train; fixed prior',
        E2='independent PIDSR enhancer strict load; reset mid_output and TaskAdapter outputs; Pcurrent=P0; clamp(P0+mid_res,0,1), no .02 bound',
        task_feedback='decoder levels2/3: current16+first16+E2current16+RGB3+previous6+available1=58; ResidualAdapter zero output; no RGB head',
        A='raw official RGB/P0 direct D2',B='E2+D2; all previous-state slots absent/zero; no H/C',
        R='same E2/D2/interface init as B; true T1/F1 feedback; no H/C',
        loss='L1 + (1-SSIM) + 0.1*phaseMSE + 0.1*VGG_L1; inherited G6 VGG input convention',
        lrs=dict(D2=1e-5,E2_core=2e-5,adapters=1e-4),
        optimizer=dict(kind='Adam',betas=[0.9,0.999],eps=1e-8,weight_decay=0),
        gradient_clip=dict(D2=1.0,E2_and_interfaces=1.0),scheduler=None,amp=False,warmup=0,
        mode=dict(D2='eval with gradients',E2='train activation checkpoint'),
        training_noise='train phase/2026/step/micro single sample context; paired actual [1,16,256] D1,A,B,R',
        evaluation_noise='validation phase/2026/original full872 index; same G6 forward and actual paired noise',
        promotion=dict(psnr=0.05,ssim=0.0,positive_scenes=6,total_scenes=8,comparators=['T1','A'],candidates=['B','R']),
        confirmation='If B OR R passes both comparators, evaluate same final3000 A/B/R weights full872val+188test, no additional training',
        stop='Neither passes: stop instance, no extra steps/seeds/LR/loss/amplitude/H/Oracle search',deadline_unix=None,max_wall=None)
    # Keep only loader and provenance fields as authoritative through this new plan.
    cfg['seed']=2026;cfg['paths']['schedule']=str(sp);cfg['paths']['output_root']=str(ROOT/'unused_output')
    cfg['task_initialization']['kind']='verified_previous_final'
    cfg['downstream'].update(steps=3000,minimum_steps=3000,effective_batch=2,microbatch=1)
    cfg['downstream'].pop('deadline_unix',None)
    cfg['experiment38']=scientific
    cfg['historical_inherited_fields_not_authoritative']='Only loader/base paths are reused. Do not execute old initialization, H, training, losses, selection or deadlines; config/locked.json experiment38 is authoritative.'
    cp=ROOT/'config/execution_config.json';write(cp,cfg)
    sources=[receipt(p,role) for p,role in [(source_cfg,'loader provenance only'),(source_schedule,'new common training schedule source'),(sp,'locked schedule'),(cp,'new loader configuration'),(protocol,'registered experiment38 protocol'),(g6,'strong validation-selected G6 net_g'),(e,'independent PIDSR pretraining'),(reference,'original G6 872 validation and index'),(vp,'pre-existing fixed64 list'),(cfg['paths']['manifest'],'dataset manifest')]]
    for k in ['base_config','reference_r2p','source_checkpoint']:
        p=Path(cfg['paths'][k])
        if p.is_file():sources.append(receipt(p,'dependency '+k))
    base_yaml=yaml.safe_load(Path(cfg['paths']['base_config']).read_text());sources.append(receipt(base_yaml['official']['dm_config'],'official model structure configuration'))
    wd=Path(cfg['task_initialization']['archived_source_dir']);assert wd.is_dir()
    for p in sorted(wd.glob('*.pth')):sources.append(receipt(p,'official frozen prior and construction weights'))
    codes=set((ROOT/'scripts').glob('*.py'))
    for d in [PROJECT/'polar_residual_feedback_joint_20261010/scripts',Path(cfg['paths']['source_root'])/'rgb_final_confirmation',PROJECT/'deps/DfP',PROJECT/'deps/rgb_final_confirmation/src/rgb_final_confirmation']:
        assert d.is_dir(),str(d);codes.update(d.rglob('*.py'))
    # Imported helper chains are provenance only, never instantiate old experiment components.
    for d in [PROJECT/'confidence_causal_completion_20261017/scripts',PROJECT/'confidence_correction_causal_20261017/scripts',PROJECT/'c_feedback_polar_loop_20261015/scripts',old36/'scripts',PROJECT/'polar_gain_judgement_short_20261012/scripts',PROJECT/'polar_error_guided_short_20261012/scripts',PROJECT/'polar_local_acceptance_joint_20261011/scripts']:
        if d.is_dir():codes.update(d.rglob('*.py'))
    for p in [PROJECT/'deps/r2p_film.py']:
        if p.is_file():codes.add(p)
    plan=dict(schema_version=1,experiment=38,authorized=True,exploratory_only=True,
        arms=['A','B','R'],steps=3000,seed=2026,evaluation_seed=2026,crop_size=128,effective_batch=2,microbatch=1,
        deadline_unix=None,max_wall=None,validation_ids=ids,full_index_map=full_index,
        g6_checkpoint=str(g6),pidsr_checkpoint=str(e),reference_validation872=str(reference),
        execution_cfg_path=str(cp),train_cfg_path=str(cp),schedule3000_path=str(sp),schedule_sha256=sha(sp),
        scientific=scientific,scientific_id=digest(scientific),loss=scientific['loss'],lrs=scientific['lrs'],
        optimizer=scientific['optimizer'],promotion=scientific['promotion'],reference_atol=scientific['reference_atol'],sources=sources,
        code_sources=[receipt(p) for p in sorted(codes)])
    plan['protocol_id']=digest(plan);write(lp,plan);verify_locked_sources(plan)
    print(json.dumps(dict(prepared=True,protocol_id=plan['protocol_id'],scientific_id=plan['scientific_id'],sources=len(sources),code_sources=len(codes)),ensure_ascii=False))
if __name__=='__main__':main()


