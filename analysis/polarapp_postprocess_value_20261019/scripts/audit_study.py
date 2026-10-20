"""Independent CPU recount, fixed-probe identities and paired arithmetic."""
import csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CHECKPOINTS=[0,1000,2000,3000,4500,6000]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');tmp.replace(p)
def archive():
 manifest=read(ROOT/'artifact_manifest.json')
 for rec in manifest['files']:
  path=ROOT/rec['path'];assert path.stat().st_size==rec['bytes'] and sha(path)==rec['sha256'],rec['path']
 index=read(ROOT/'config/cache_index.json');assert len(index['entries'])==12000
 for rec in index['entries']:assert sha(ROOT/rec['path'])==rec['sha256']
 return dict(passed=True,files=len(manifest['files']),cached_ROI_verified=12000,manifest_sha256=sha(ROOT/'artifact_manifest.json'))
def compare(data):
 scenes=sorted({r['scene'] for r in data});stats={};images={}
 for a,b in [('A','S'),('B','S'),('B','A')]:
  pair=a+'-'+b;stats[pair]={};images[pair]={}
  for m in ['psnr','ssim']:
   per={s:float(np.mean([r['metrics'][a][m]-r['metrics'][b][m] for r in data if r['scene']==s])) for s in scenes}
   stats[pair][m]=dict(scene_equal=float(np.mean(list(per.values()))),scenes=per,positive_scenes=sum(v>0 for v in per.values()))
   images[pair][m]=float(np.mean([r['metrics'][a][m]-r['metrics'][b][m] for r in data]))
 return stats,images

def main():
 if '--archive' in sys.argv:write(ROOT/'reports/archive_audit.json',archive());return
 if '--postcloseout' in sys.argv:
  value=archive();close=read(ROOT/'closeout.json');assert close['success'] and close==read(ROOT/'reports/final_review_closeout.json')
  assert close['archive_manifest_sha256']==value['manifest_sha256'];write(ROOT/'reports/postcloseout_archive_audit.json',value);return
 plan=read(ROOT/'config/locked.json');summary=read(ROOT/'reports/summary.json')
 assert plan['steps']==summary['steps']==6000 and summary['test']==0
 rows=list(csv.DictReader((ROOT/'logs/training.csv').open()));assert [int(r['step']) for r in rows]==list(range(1,6001))
 for r in rows:
  for k in ['A_loss','B_loss','A_T_grad','B_T_grad','E_grad','seconds']:assert math.isfinite(float(r[k]))
 schedule=read(plan['schedule_path'])['items'];index=read(ROOT/'config/cache_index.json');assert len(index['entries'])==len(schedule)==12000
 old=read(ROOT/'history/user_requested_budget_v1/config/training3000.json');assert schedule[:6000]==old['items']
 for i,(r,it) in enumerate(zip(index['entries'],schedule)):
  assert r['index']==i and r['top']==it['top'] and r['left']==it['left'];assert sha(ROOT/r['path'])==r['sha256']
  assert r['sample_id']==rows[i//2]['sample_ids'].split(';')[i%2]
 assert len(index['full_sources'])==5440
 assert sha(ROOT/'config/cache_index.json')==read(ROOT/'diagnostics/cache_complete.json')['index_sha256']
 for name in ['native_preflight.json','frozen_final.json','training_complete.json']:assert read(ROOT/'diagnostics'/name).get('passed',True)
 pf=read(ROOT/'config/preflight.json');assert pf['passed'] and pf['resume_exact'] and pf['E_actual_head_and_body_update'] and pf['all_preflight_updates_discarded']
 S={};train_reference={};probe_stats={};image_stats={}
 for step in CHECKPOINTS+[6001]:
  endpoint=step==6001;actual=6000 if endpoint else step;ids=plan['validation_ids'] if endpoint else plan['probe_ids']
  path=ROOT/'metrics/step_6000.json' if endpoint else ROOT/'metrics'/f'val_probe_{step:04d}.json'
  metric=read(path);data=metric['rows'];assert metric['protocol_id']==plan['protocol_id'] and metric['step']==actual
  assert [r['sample_id'] for r in data]==ids and len(data)==(64 if endpoint else 16)
  scenes=sorted({r['scene'] for r in data});assert len(scenes)==8
  assert all(sum(r['scene']==s for r in data)==(8 if endpoint else 2) for s in scenes)
  for r in data:
   assert r['original_size'] and [x['arm'] for x in r['noise_calls']]==['S','A','B']
   assert len({x['sha256'] for x in r['noise_calls']})==1 and all(x['shape']==[1,16,256] for x in r['noise_calls'])
   value=(r['output_sha256']['S'],r['PD_sha256'],r['metrics']['S'],r['noise_calls'][0]['sha256'])
   if r['sample_id'] in S:assert S[r['sample_id']]==value
   S[r['sample_id']]=value
   if actual==0:assert len(set(r['output_sha256'].values()))==1
  stats,images=compare(data)
  if endpoint:assert stats==summary['endpoint_comparisons'];final=stats;image_stats['endpoint']=images
  else:
   assert stats==summary['probe_evaluations'][str(step)];probe_stats[str(step)]=stats;image_stats[str(step)]=images
   train=read(ROOT/'metrics'/f'train_probe_{step:04d}.json');assert train['protocol_id']==plan['protocol_id'] and train['frames']==16 and train['step']==step
   assert [r['ROI_index'] for r in train['rows']]==list(range(16))
   for r in train['rows']:
    rec=index['entries'][r['ROI_index']];assert r['sample_id']==rec['sample_id'] and r['ROI_sha256']==rec['sha256']
    assert [x['arm'] for x in r['noise_calls']]==['S','A','B'] and len({x['sha256'] for x in r['noise_calls']})==1
    value=(r['ROI_sha256'],r['output_sha256']['S'],r['noise_calls'][0]['sha256'],r['metrics']['S'])
    if r['ROI_index'] in train_reference:assert train_reference[r['ROI_index']]==value
    train_reference[r['ROI_index']]=value
   for a in ['S','A','B']:
    for m in ['loss','psnr','ssim']:assert float(np.mean([r['metrics'][a][m] for r in train['rows']]))==summary['training_probe_curve'][str(step)][a][m]
   assert read(ROOT/'diagnostics'/f'probe_complete_{step:04d}.json')['passed']
 gate=all(final['B-'+b]['psnr']['scene_equal']>=.05 and final['B-'+b]['ssim']['scene_equal']>=0 and final['B-'+b]['psnr']['positive_scenes']>=6 for b in ['A','S'])
 assert gate==summary['gate_passed'];frozen=read(ROOT/'diagnostics/frozen_final.json');assert frozen['before']==frozen['after']
 write(ROOT/'reports/independent_audit.json',dict(passed=True,paired_updates=6000,cached_ROI=12000,full_train_images=5440,probe_checkpoints=CHECKPOINTS,fixed_val_probe=16,fixed_train_probe=16,validation_endpoint=64,test=0,gate_passed=gate,scene_probe_statistics=probe_stats,endpoint_comparisons=final,image_average_differences=image_stats))
if __name__=='__main__':main()