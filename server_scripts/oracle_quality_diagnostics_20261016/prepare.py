"""Lock the newly authorized diagnostic and single Oracle arm; no model changes."""
import hashlib,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'c_feedback_polar_loop_20261015'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
assert not (ROOT/'config/plan.json').exists(),'Do not overwrite a locked study'
for d in ('config','logs','weights','metrics','reports','diagnostics','examples','failures','cache'):(ROOT/d).mkdir(parents=True,exist_ok=True)
old=json.loads((OLD/'config/plan.json').read_text())
assert old['protocol_id']=='179e35ce4dd0e48594b21592839cbacde8c4f912e082df0195bcfe8b0f0cea18'
rows=json.loads((OLD/'metrics/validation872.json').read_text())['rows']
scenes=sorted({r['scene'] for r in rows})
p=dict(kind='fixed-scale offline and Oracle-quality matched diagnostic',authorized_by='User 2026-10-16: directly execute both proposed experiments plus added selection diagnostic',
       steps=6000,arms=['O'],reuse_prediction_arm='experiment30 B fixed6000',reuse_reference='experiment30 D and T1',
       source_initial_sha256=old['source_initial_sha256'],source_protocol=old['protocol_id'],seed=2026,
       oracle=dict(target='4x4 mean RGB MAE between clamp(T1,0,1) and GT',b_floor=.001,tau=.02494479901222676,
                   formula='C=exp(-max(error,.001)/tau)',GT_only_in_oracle_quality_branch=True),
       trainable_per_arm=292896,optimizer='same fresh Adam state as original B at step0',
       schedule='same12000 crops;5440 unique images;128 crop;batch2 micro1; original per-sample diffusion seeds',
       loss=old['loss'],residual_bound=.02,architecture='unchanged B; fixed E1 H PolarFree; independent E2 adapters and output only',
       evaluation=dict(validation=872,test=188,original_size=True,tiling=False,checkpoint='fixed6000 only'),
       decision=dict(pairs=['O-P','O-T1'],psnr_min=.05,ssim_min=0.,positive_scenes_min=6),
       offline=dict(split='validation',frames=872,block=4,quintiles=5,quintile_ties='stable row-major rank, equal counts; tie fraction reported',
                    fixed_examples=[min(r['sample_id'] for r in rows if r['scene']==s) for s in scenes],
                    statistics='per-image then scene-equal; correlations descriptive; no cell independence assumption',
                    numerical_psnr_crosscheck_tolerance=1e-4,small_RGB_update=1e-6,
                    selection='GT region-wise lower MSE from fixed T1 or fixed B T2; only this restricted selection class',
                    selection_resource_line=.05,conditional_training_gate=False),
       stop='one O arm6000;full evaluation; no extra seed, loss, tau, bound or architecture search',
       time_budget_enforced=False,resource_wait=dict(minimum_free_MiB=11500,poll_seconds=2),
       test_used_for_selection=False,server_only=True,
       references=['https://arxiv.org/html/1802.07095','https://kuldeeppurohit.github.io/spair/'])
paths=['config/plan.json','config/execution_config.json','config/manifest.csv','config/training6000.json',
       'metrics/validation872.json','metrics/test188.json','weights/initial.pt','weights/final.pt','weights/step_500.pt']
p['source_receipts']=[dict(path=s,sha256=sha(OLD/s)) for s in paths]
p['new_code_manifest']=[dict(path=str(f.relative_to(ROOT)),sha256=sha(f)) for f in sorted((ROOT/'scripts').glob('*.py'))]
p['protocol_id']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
(ROOT/'config/plan.json').write_text(json.dumps(p,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(dict(prepared=True,protocol_id=p['protocol_id'],free_GiB=shutil.disk_usage(ROOT).free/1024**3)))
