"""Register experiment33; reuse the verified loader and training schedule builder."""
import copy,hashlib,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
PREV=ROOT.parent/'confidence_correction_causal_20261017'
OLD=ROOT.parent/'c_feedback_polar_loop_20261015'
sys.path.insert(0,str(PREV/'scripts'))
import causal_run as prior

def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))

if (ROOT/'config/proposal.json').exists():
    print('Registered proposal preserved');raise SystemExit
assert json.loads((PREV/'reports/final_review_closeout.json').read_text())['completed']
cfg=json.loads((OLD/'config/execution_config.json').read_text())
prior.base.study.base.add_source_path(cfg)
from rgb_final_confirmation.data import build_training_schedule,load_manifest,select_rows
train=select_rows(load_manifest(cfg['paths']['manifest']),'train');ordered=sorted(train,key=lambda r:r.sample_id)
lookup={r.sample_id:i for i,r in enumerate(train)}
original=json.loads(pathlib.Path(cfg['paths']['schedule']).read_text())
schedules={}
for seed in [2026,2027]:
    schedule=build_training_schedule(ordered,seed=seed,total_steps=6000,batch_size=2,crop_size=128)
    for item in schedule['items']:item['sample_index']=lookup[ordered[item['sample_index']].sample_id]
    schedule.pop('sha256',None)
    schedule['sha256']=hashlib.sha256(json.dumps(schedule,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert len(schedule['items'])==12000
    if seed==2026:assert schedule['items']==original['items'],'2026 schedule not identical'
    path=ROOT/'config'/f'training6000_{seed}.json';write(path,schedule)
    schedules[str(seed)]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path))
    c=copy.deepcopy(cfg);c['seed']=seed;c['paths']['schedule']=str(path);c['paths']['output_root']=str(ROOT/'unused_output')
    write(ROOT/'config'/f'execution_config_{seed}.json',c)
assert schedules['2026']['sha256']!=schedules['2027']['sha256']
receipts=[]
for p in [PREV/'scripts/causal_run.py',PREV/'config/locked.json',PREV/'weights/initial.pt',PREV/'weights/step_500.pt',
          PREV/'weights/final.pt',PREV/'metrics/validation872.json',PREV/'metrics/test188.json',PREV/'logs/training.csv',
          OLD/'scripts/c_loop_run.py',OLD/'weights/initial.pt',OLD/'weights/common_initial21.pt',OLD/'config/plan.json',
          pathlib.Path(cfg['paths']['manifest']),pathlib.Path(cfg['paths']['schedule'])]:
    receipts.append(dict(path=str(p),sha256=sha(p)))
proposal=dict(version=1,experiment=33,authorized=True,
    authorization='User 2026-10-17: extra12h; restore removed controls; analyze and start without approval',
    proposed_arms=['R1N0','R1F','R1S','R1P7','R1S7'],core_arms=['R1N0','R1F','R1S'],
    seed_by_arm=dict(R1N0=2026,R1F=2026,R1S=2026,R1P7=2027,R1S7=2027),steps=6000,
    lr=1e-4,bound=.02,crop=128,effective_batch=2,schedules=schedules,
    calibration='first256 seed2026 scheduled training crops; predicted C only; one global scalar; no GT',
    sorting='GT pooled clamped-T1 RGB MAE ascending, predicted C values descending; stable ties, original H grid',
    evaluation=dict(validation=872,test=188,original_size=True,tiling=False,fixed_evaluation_seed=2026),
    source_receipts=receipts,budget=json.loads((ROOT/'budget.json').read_text()),
    code_manifest=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted((ROOT/'scripts').glob('*.py'))],
    data_manifest=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted((ROOT/'config').glob('*.json'))])
proposal['protocol_id']=hashlib.sha256(json.dumps(proposal,sort_keys=True).encode()).hexdigest()
write(ROOT/'config/proposal.json',proposal)
print(json.dumps(dict(prepared=True,protocol_id=proposal['protocol_id'],schedules=schedules)))
