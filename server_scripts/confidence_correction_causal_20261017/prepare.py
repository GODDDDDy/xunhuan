"""Register authorized v2 using existing frozen code, source and evaluation rows."""
import csv,hashlib,json,pathlib,shutil,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'c_feedback_polar_loop_20261015'
ORA=ROOT.parent/'oracle_quality_diagnostics_20261016'
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
if (ROOT/'config/proposal.json').exists():
    print('Already prepared; preserving original registration');raise SystemExit
for f in ['config','weights','logs','metrics','diagnostics','reports','failures','examples','cache']:(ROOT/f).mkdir(exist_ok=True)
old=json.loads((OLD/'config/plan.json').read_text())
assert old['protocol_id']=='179e35ce4dd0e48594b21592839cbacde8c4f912e082df0195bcfe8b0f0cea18'
assert json.loads((OLD/'closeout.json').read_text())['completed']
assert json.loads((ORA/'closeout.json').read_text())['completed']
cfg=json.loads((OLD/'config/execution_config.json').read_text())
cfg['paths']['output_root']=str(ROOT/'unused_output')
write(ROOT/'config/execution_config.json',cfg)
rows=list(csv.DictReader(open(cfg['paths']['manifest'])))
train=[r for r in rows if r['split']=='train']
schedule=json.loads(pathlib.Path(cfg['paths']['schedule']).read_text())
scenes=sorted({r['scene'] for r in train})[:8];chosen=[]
for scene in scenes:
    eligible=sorted([r for r in train if r['scene']==scene],key=lambda r:r['sample_id'])
    for r in eligible[:2]:
        idx=next(i for i,t in enumerate(train) if t['sample_id']==r['sample_id'])
        matches=[(i,x) for i,x in enumerate(schedule['items']) if x['sample_index']==idx]
        assert matches,r['sample_id']
        pos,item=matches[0];chosen.append(dict(sample_id=r['sample_id'],scene=scene,schedule_position=pos,**item))
val=json.loads((OLD/'metrics/validation872.json').read_text())['rows']
test=json.loads((OLD/'metrics/test188.json').read_text())['rows']
assert len(val)==872 and len(test)==188 and len(chosen)==16
# This is the already registered 64-image balanced validation selection.
ids64=[]
for s in sorted({r['scene'] for r in val}):
    ids64 += sorted(r['sample_id'] for r in val if r['scene']==s)[:8]
# Prefer exact historical list when available; selection is locked before any outputs.
for p in [ROOT.parent/'h_rank_quality_20261015/config/validation64.json',ROOT.parent/'polar_quality_matched_candidate_20261015/config/validation64.json']:
    if p.exists():
        d=json.loads(p.read_text());a=d.get('sample_ids',d.get('ids',[]))
        if len(a)==64:ids64=a;break
assert len(ids64)==64 and len(set(ids64))==64
assert all(sum(r['scene']==s and r['sample_id'] in ids64 for r in val)==8 for s in {r['scene'] for r in val})
write(ROOT/'config/diagnostic_crops.json',chosen)
write(ROOT/'config/validation64.json',dict(sample_ids=ids64,selection='existing64 if available; otherwise first8 sample IDs per scene, locked before outputs'))
receipts=[]
for root,rels in [(OLD,['config/plan.json','config/execution_config.json','config/manifest.csv','config/training6000.json','weights/common_initial21.pt','weights/initial.pt','weights/step_500.pt','weights/final.pt','metrics/validation872.json','metrics/test188.json']),
                  (ORA,['config/plan.json','weights/final.pt','metrics/validation872.json','metrics/test188.json'])]:
    for rel in rels:receipts.append(dict(path=str(root/rel),sha256=sha(root/rel)))
proposal=dict(version=2,authorized=True,authorized_by='User: start training; v2 24-36h approved 2026-10-17',
    seed=2026,steps=6000,crop=128,effective_batch=2,lr=1e-4,clip=1.,bound=.02,
    proposed_arms=['R1D','R1P','R1O','R0N','R1N'],fallback_arms=['R1D','R1P','R1O'],
    source_receipts=receipts,source_protocol=old['protocol_id'],source_initial_sha256=old['source_initial_sha256'],
    oracle='original31 GT pooled clamped T1 RGB MAE floor .001; original tau',
    reused=['R0D:old30 D','R0P:old30 B','R0O:old31 O'],
    diagnostic=dict(A_crops=16,A_steps=100,A_lr=.001,B_arms=['R0P','R1P'],B_steps=1000,no_scientific_gate=True),
    evaluation=dict(validation=872,test=188,original_size=True,tiling=False,checkpoint='fixed6000',intermediate_full_eval=False),
    budget=json.loads((ROOT/'budget.json').read_text()),
    code_manifest=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted((ROOT/'scripts').glob('*.py'))],
    data_manifest=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted((ROOT/'config').glob('*.json'))])
proposal['protocol_id']=hashlib.sha256(json.dumps(proposal,sort_keys=True).encode()).hexdigest()
write(ROOT/'config/proposal.json',proposal)
print(json.dumps(dict(prepared=True,protocol_id=proposal['protocol_id'],scenes=scenes,validation64=len(ids64))))
