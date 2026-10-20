"""Independent CPU audit of continuation, same-start contrast and pair summaries."""
import csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'polarapp_postprocess_value_20261019'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False))
def compare(rows,a,b):
 out={}
 for metric in ['psnr','ssim']:
  scenes=sorted({r['scene'] for r in rows});per={s:float(np.mean([r['metrics'][a][metric]-r['metrics'][b][metric] for r in rows if r['scene']==s])) for s in scenes}
  out[metric]=dict(scene_equal=float(np.mean(list(per.values()))),scenes=per,positive_scenes=sum(x>0 for x in per.values()))
 return out
def archive():
 manifest=read(ROOT/'artifact_manifest.json')
 for rec in manifest['files']:assert (ROOT/rec['path']).stat().st_size==rec['bytes'] and sha(ROOT/rec['path'])==rec['sha256'],rec['path']
 return dict(passed=True,files=len(manifest['files']),manifest_sha256=sha(ROOT/'artifact_manifest.json'))
def main():
 if '--archive' in sys.argv:write(ROOT/'reports/archive_audit.json',archive());return
 if '--postcloseout' in sys.argv:
  v=archive();c=read(ROOT/'closeout.json');assert c['success'] and c==read(ROOT/'reports/final_review_closeout.json') and c['archive_manifest_sha256']==v['manifest_sha256'];write(ROOT/'reports/postcloseout_archive_audit.json',v);return
 plan=read(ROOT/'config/locked.json');total=plan['steps'];assert total in [6000,9000]
 logs=list(csv.DictReader((ROOT/'logs/training.csv').open()));assert [int(r['step']) for r in logs]==list(range(1,total+1))
 schedule=read(ROOT/'config/continuation_ROI.json')['entries']
 meta_count=0
 for r in logs:
  step=int(r['step']);assert int(r['global_step'])==6000+step
  assert r['sample_ids'].split(';')==[x['sample_id'] for x in schedule[(step-1)*2:step*2]]
  for key in ['A_task_loss','B_task_loss','C_task_loss','C_alignment_loss','meta_hypergradient_scaled_norm']:assert math.isfinite(float(r[key]))
  if step%10==0:assert float(r['meta_hypergradient_scaled_norm'])>0;meta_count+=1
 assert meta_count==total//10
 pf=read(ROOT/'config/preflight.json');assert pf['passed'] and pf['meta_and_joint_resume_exact'] and pf['probe_weights_Adam_RNG_exact'] and pf['all_preflight_updates_discarded']
 assert read(ROOT/'config/calibration.json')['lambda_value']==plan['lambda_value']>0
 fixed=read(ROOT/'diagnostics/frozen_final.json');assert fixed['passed'] and fixed['before']==fixed['after']
 checkpoints=[0,1000,3000,6000]+([9000] if total==9000 else []);S={};trainS={};probe_stats={}
 for step in checkpoints:
  values=read(ROOT/'metrics'/f'val_probe_{step:05d}.json');train=read(ROOT/'metrics'/f'train_probe_{step:05d}.json')
  assert values['protocol_id']==train['protocol_id']==plan['protocol_id'] and values['frames']==train['frames']==16
  assert [r['sample_id'] for r in values['rows']]==plan['probe_ids']
  assert read(ROOT/'diagnostics'/f'probe_complete_{step:05d}.json')['passed']
  for r in values['rows']:
   assert [x['arm'] for x in r['noise_calls']]==['S','A','B','C'] and len({x['sha256'] for x in r['noise_calls']})==1
   value=(r['output_sha256']['S'],r['PD_sha256'],r['noise_calls'][0]['sha256'])
   if r['sample_id'] in S:assert S[r['sample_id']]==value
   S[r['sample_id']]=value
  for r in train['rows']:
   assert [x['arm'] for x in r['noise_calls']]==['S','A','B','C'] and len({x['sha256'] for x in r['noise_calls']})==1
   value=(r['sample_id'],r['output_sha256']['S'],r['noise_calls'][0]['sha256'])
   if r['ROI_index'] in trainS:assert trainS[r['ROI_index']]==value
   trainS[r['ROI_index']]=value
  probe_stats[str(step)]={a+'-'+b:compare(values['rows'],a,b) for a,b in [('B','A'),('C','B'),('C','A')]}
 summary=read(ROOT/'reports/summary.json');endpoint=read(ROOT/'metrics/endpoint64.json')['rows'];assert len(endpoint)==64 and [r['sample_id'] for r in endpoint]==plan['validation_ids']
 pairs=[('A','S'),('B','S'),('C','S'),('B','A'),('C','B'),('C','A')];stats={a+'-'+b:compare(endpoint,a,b) for a,b in pairs};assert stats==summary['comparisons']
 gates={a:all(stats[a+'-'+b]['psnr']['scene_equal']>=.05 and stats[a+'-'+b]['ssim']['scene_equal']>=0 and stats[a+'-'+b]['psnr']['positive_scenes']>=6 for b in (['A','S'] if a=='B' else ['B','A','S'])) for a in ['B','C']};assert gates==summary['gates']
 p=ROOT/'reports/confirmation_summary.json'
 if p.exists():
  full=read(p);assert full['complete']
  for split,n in [('validation',872),('test',188)]:
   rows=read(ROOT/'metrics'/f'{split}_full.json')['rows'];assert len(rows)==n and {a+'-'+b:compare(rows,a,b) for a,b in pairs}==full['comparisons'][split]
 write(ROOT/'reports/independent_audit.json',dict(passed=True,additional_updates_each=total,global_updates_each=6000+total,meta_updates=meta_count,endpoint_frames=64,probe_checkpoints=checkpoints,comparisons=stats,gates=gates,probe_comparisons=probe_stats,conditional_confirmation_complete=p.exists(),strict_source_Adam_counter_conformity=False,documented_main_Adam_counter_offset=4,recovery_file_sha256=read(ROOT/'reports/recovered_checkpoint_audit.json')['sha256'],interpretation_limit='Actual executed variant: matching A/B/C counter offset4 from preflight aliasing; no silent counter rollback'))
if __name__=='__main__':main()
