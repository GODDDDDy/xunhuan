"""Independent CPU recomputation after controller closeout; never changes training."""
import csv,hashlib,json,math,os,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def write(p,x):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False));t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def run():
    close=json.loads((ROOT/'closeout.json').read_text());assert close['completed'] and close['deadline_met']
    plan=json.loads((ROOT/'config/locked.json').read_text());names=plan['arms']
    proposal=json.loads((ROOT/'config/proposal.json').read_text())
    for rec in proposal['source_receipts']:assert sha(Path(rec['path']))==rec['sha256'],rec['path']
    manifest=json.loads((ROOT/'artifact_manifest.json').read_text())
    for rec in manifest['files']:assert sha(ROOT/rec['path'])==rec['sha256'],rec['path']
    train=list(csv.DictReader((ROOT/'logs/training.csv').open()))
    assert [int(r['step']) for r in train]==list(range(1,6001))
    assert len({s for r in train for s in r['sample_ids'].split(';')})==5440
    reference_train=list(csv.DictReader((ROOT.parent/'c_feedback_polar_loop_20261015/logs/training.csv').open()))
    assert [r['sample_ids'] for r in train]==[r['sample_ids'] for r in reference_train]
    for step in range(500,6001,500):assert json.loads((ROOT/'diagnostics'/f'frozen_{step}.json').read_text())['passed']
    completed=json.loads((ROOT/'diagnostics/training_complete.json').read_text());assert completed['source_frozen_exact']
    assert completed['final_sha256']==sha(ROOT/'weights/final.pt')
    import torch
    ck=torch.load(ROOT/'weights/final.pt',map_location='cpu',weights_only=False)
    assert ck['step']==6000 and ck['protocol_id']==plan['protocol_id'] and set(ck['arms'])==set(names)
    for a,opt in ck['optimizers'].items():
        assert opt['state'] and all(int(v['step'])==6000 for v in opt['state'].values()),a
    output={}
    for split,n in [('validation872',872),('test188',188)]:
        data=json.loads((ROOT/'metrics'/f'{split}.json').read_text());rows=data['rows'];assert len(rows)==n
        assert len({r['sample_id'] for r in rows})==n
        assert data['original_size'] and not data['tiling'] and data['step']==6000
        old=json.loads((ROOT.parent/'c_feedback_polar_loop_20261015/metrics'/f'{split}.json').read_text())['rows']
        oracle=json.loads((ROOT.parent/'oracle_quality_diagnostics_20261016/metrics'/f'{split}.json').read_text())['rows']
        old={r['sample_id']:r for r in old};oracle={r['sample_id']:r for r in oracle}
        assert set(old)==set(oracle)=={r['sample_id'] for r in rows}
        for r in rows:
            sid=r['sample_id'];assert r['t1_sha256']==old[sid]['t1_sha256']==oracle[sid]['t1_sha256']
            assert r['shape']==old[sid]['shape']
            assert r['metrics']['T1']==old[sid]['metrics']['T1']==oracle[sid]['metrics']['T1']
            assert r['metrics']['R0P']==old[sid]['metrics']['B'] and r['metrics']['R0D']==old[sid]['metrics']['D']
            assert r['metrics']['R0O']==oracle[sid]['metrics']['O']
        for pair,stats in data['comparisons'].items():
            a,b=pair.split('-')
            for metric in ['psnr','ssim']:
                ds=[r['metrics'][a][metric]-r['metrics'][b][metric] for r in rows]
                by={s:statistics.mean(r['metrics'][a][metric]-r['metrics'][b][metric] for r in rows if r['scene']==s) for s in sorted({r['scene'] for r in rows})}
                assert abs(statistics.mean(by.values())-stats[metric]['scene_equal'])<1e-10
                assert abs(statistics.mean(ds)-stats[metric]['image_equal'])<1e-10
        terms={'local_C':('R1P','R1D','R0P','R0D'),'oracle_condition':('R1O','R1P','R0O','R0P')}
        if 'R0N' in names:terms['task_feedback']=('R1D','R1N','R0D','R0N')
        interaction={}
        for label,(a,b,c,d) in terms.items():
            interaction[label]={}
            for m in ['psnr','ssim']:
                fn=lambda r:(r['metrics'][a][m]-r['metrics'][b][m])-(r['metrics'][c][m]-r['metrics'][d][m])
                by={s:statistics.mean(fn(r) for r in rows if r['scene']==s) for s in sorted({r['scene'] for r in rows})}
                interaction[label][m]=dict(scene_equal=statistics.mean(by.values()),image_equal=statistics.mean(fn(r) for r in rows),scenes=by)
        output[split]=dict(frames=n,comparisons=data['comparisons'],interaction=interaction)
    record=dict(passed=True,time_unix=time.time(),steps=6000,arms=names,validation=872,test=188,
        all_source_and_artifact_hashes_verified=True,optimizer_steps_verified=True,paired_summaries_recomputed=True,
        report=output,oracle_uses_GT=True,multiseed_claim=False,closeout_sha256=sha(ROOT/'closeout.json'),
        artifact_manifest_sha256=sha(ROOT/'artifact_manifest.json'))
    write(ROOT/'reports/independent_audit.json',record)
    write(ROOT/'reports/final_review_closeout.json',dict(completed=True,time_unix=time.time(),independent_audit_sha256=sha(ROOT/'reports/independent_audit.json'),audit_script_sha256=sha(Path(__file__)),server_only=True))
    write(ROOT/'audit_status.json',dict(stage='verified_completed',time_unix=time.time(),validation=872,test=188))
def main():
    budget=json.loads((ROOT/'budget.json').read_text())
    write(ROOT/'audit_registration.json',dict(pid=os.getpid(),time_unix=time.time(),role='CPU-only independent verification; no training changes'))
    while not (ROOT/'closeout.json').exists():
        if time.time()>=budget['deadline_unix']:
            write(ROOT/'audit_status.json',dict(stage='deadline_incomplete',time_unix=time.time()));return
        write(ROOT/'audit_status.json',dict(stage='waiting_for_closeout',time_unix=time.time()))
        time.sleep(30)
    run()
if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        write(ROOT/'audit_status.json',dict(stage='verification_failed',time_unix=time.time(),traceback=traceback.format_exc()));raise
