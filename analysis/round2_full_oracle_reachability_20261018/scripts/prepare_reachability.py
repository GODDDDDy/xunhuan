"""Lock one pre-existing validation image per scene, before optimization."""
import hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent
P34=PROJECT/'round2_context_diagnostic_20261018';P33=PROJECT/'confidence_causal_completion_20261017'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    assert not (ROOT/'config/locked.json').exists()
    for d in ['config','metrics','reports','logs','failures','variables','operations']:(ROOT/d).mkdir(parents=True,exist_ok=True)
    old=json.loads((P34/'config/locked.json').read_text());seen=set();items=[]
    for x in old['items']:
        if x['split']=='validation64' and x['scene'] not in seen:
            items.append({k:x[k] for k in ['sample_id','scene','height','width']});seen.add(x['scene'])
    assert len(items)==8
    receipt=lambda p:dict(path=str(p),sha256=sha(p))
    paths=[Path(x['path']) for x in old['sources']]+[P34/'config/locked.json',P34/'config/implementation_revision_1.json',P34/'scripts/run_context.py',ROOT/'PROTOCOL_ZH.md']
    cfg=dict(experiment=35,created_unix=time.time(),items=items,steps=100,lr=.001,bound=.02,mu=old['mu'],max_wall_seconds=14400,
             sources=[receipt(p) for p in paths],code_sources=[receipt(p) for p in sorted((ROOT/'scripts').glob('*.py'))],
             reference=receipt(P33/'metrics/validation872.json'),preflight_sample=max(items,key=lambda x:x['height']*x['width'])['sample_id'])
    cfg['protocol_id']=hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (ROOT/'config/locked.json').write_text(json.dumps(cfg,indent=2))
    print(json.dumps(dict(protocol_id=cfg['protocol_id'],items=items)))
if __name__=='__main__':main()
