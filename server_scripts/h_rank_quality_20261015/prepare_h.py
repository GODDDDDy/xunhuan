"""Snapshot proven components; never modify or resume any previous experiment."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'polar_quality_matched_candidate_20261015'
EXPECTED='af2367fd9f96871cad3b94230638cf5c75acd3b8f8c186cdf210a37b730e82d8'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
assert not (ROOT/'config/plan.json').exists(),'Refuse to overwrite an existing study'
for d in ('config','diagnostics','logs','weights','pretraining/checkpoints','metrics','examples'):
    (ROOT/d).mkdir(parents=True,exist_ok=True)
shutil.copytree(OLD/'scripts',ROOT/'scripts',dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('__pycache__','*.pyc','prepare_h.py','h_run.py','h_objective.py'))
cfg=json.loads((OLD/'config/execution_config.json').read_text())
cfg['paths']['output_root']=str(ROOT/'downstream');cfg['data']['workers']=0
for key,name in (('manifest','manifest.csv'),('schedule','training_schedule.json'),
                 ('base_config','base_server.yaml'),('validation_proxy','validation_proxy.json')):
    source=Path(cfg['paths'][key]);shutil.copy2(source,ROOT/'config'/name)
    cfg['paths'][key]=str(ROOT/'config'/name)
for name,source in (('weights/common_initial21.pt',OLD/'weights/common_initial21.pt'),
                    ('pretraining/checkpoints/selected.pt',OLD/'pretraining/checkpoints/selected.pt')):
    (ROOT/name).symlink_to(source.resolve())
assert sha(ROOT/'weights/common_initial21.pt')==EXPECTED
write(ROOT/'config/execution_config.json',cfg)
import sys
sys.path.insert(0,str(ROOT/'scripts'))
import study
study.base.add_source_path(cfg)
from rgb_final_confirmation.data import load_manifest,select_rows
rows=select_rows(load_manifest(cfg['paths']['manifest']),'validation')
selected=[]
for scene in sorted({r.scene for r in rows}):
    group=sorted([r for r in rows if r.scene==scene],key=lambda r:r.sample_id)
    assert len(group)>=8
    selected += [group[int((k+.5)*len(group)/8)].sample_id for k in range(8)]
assert len(selected)==len(set(selected))==64
write(ROOT/'config/validation64.json',dict(sample_ids=selected,selection='8 evenly spaced sample_id-sorted frames per scene; before predictions'))
plan=dict(kind='exploratory H-only paired screening, not full validation/test confirmation',
          seed=2026,arms=['scale_only','scale_plus_rank'],steps=2000,effective_batch=2,
          crop=128,lr=1e-4,optimizer='fresh independent Adam per head, default betas/eps',
          rank_weight=.1,rank_pairs=2048,rank_min_error_gap=.002,
          rank_margin='min(abs(log(e_i+.001)-log(e_j+.001)),1)',
          trainable='two independent copies of H only',
          frozen='all E, D, prior, feedback, Q, M; no second-round forward',
          source_initial_sha256=EXPECTED,source_initial_protocol='921ae3dc9fe004949efab2afdb434719a11a8268cb80bd44718bc59a9a99787d',
          first_round_source='experiment21 initial, same preserved first round used by experiment27',
          shared_first='one frozen first-round forward per crop, same detached features/GT target for both heads',
          endpoint='fixed step2000; no validation selection, no hyperparameter sweep',
          evaluation=dict(validation=64,test=0,original_size=True,tiling=False,
                          primary='scene-equal normalized local-MAE AUSE, removal 0:.01:.95',
                          secondary=['full-grid Spearman(b,e)','worst20 precision/recall','MAE(b,e)','original scale loss'],
                          scope='first-round quality only; no T2 benefit claim'),
          go_rule=dict(macro_ause_reduction_at_least=.005,scenes_ause_improved_at_least=6,
                       macro_worst20_recall_not_worse=True,macro_scale_loss_increase_at_most=.02),
          thresholds='project practical screening choices, not literature maturity or statistical-significance thresholds',
          stop='one candidate; failure/unclear improvement keeps original H, no automatic retry with new loss/architecture',
          budget_seconds=10800,training_seconds=3600,full_confirmation_automatic='once iff screening passes; same locked heads, full872 validation and188 test; no additional training',
          references=['https://proceedings.mlr.press/v119/moon20a.html','https://arxiv.org/abs/1707.08347','https://arxiv.org/abs/1802.07095'])
plan['data_sha256']={k:sha(cfg['paths'][k]) for k in ('manifest','schedule','base_config','validation_proxy')}
plan['validation64_sha256']=sha(ROOT/'config/validation64.json')
plan['execution_config_sha256']=sha(ROOT/'config/execution_config.json')
plan['code_manifest']=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted((ROOT/'scripts').rglob('*.py'))]
plan['protocol_id']=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
write(ROOT/'config/plan.json',plan)
print(json.dumps(dict(prepared=True,protocol_id=plan['protocol_id'],validation=64,test=0)))
