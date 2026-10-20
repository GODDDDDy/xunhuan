"""Independent CPU evidence/arithmetic audit40; no independent image inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,math,statistics,sys,time,traceback
from pathlib import Path
import torch
import run_study as run
ROOT=run.ROOT;read,sha,equal=run.read,run.sha,run.equal
def require(value,msg):
 if not value:raise AssertionError(msg)
def near(a,b):
 if isinstance(b,dict):
  require(a.keys()==b.keys(),'Summary keys')
  for k,v in b.items():near(a[k],v)
 elif isinstance(b,(float,int)):require(abs(float(a)-float(b))<=1e-10,'Summary arithmetic')
 else:require(a==b,'Summary value')
def pair(rows,a,b):
 scenes=sorted({r['scene'] for r in rows});out={}
 for m in ['psnr','ssim']:
  per={s:statistics.mean(r['metrics'][a][m]-r['metrics'][b][m] for r in rows if r['scene']==s) for s in scenes}
  out[m]=dict(scene_equal=statistics.mean(per.values()),scenes=per,positive_scenes=sum(v>0 for v in per.values()))
 return out
def audit(plan):
 run.verify(plan);require(not torch.cuda.is_available(),'CPU only');torch.set_num_threads(2)
 p=read(ROOT/'config/preflight.json')
 for k in ['passed','initial_equal','no_parameter_alias','same_input_update_exact','state_check_nonzero_interfaces','B_no_state_leak','resume_exact','source_frozen','all_preflight_updates_discarded']:require(p[k] is True,k)
 require(p['R_output_sensitivity']>0 and p['A_reuse']['replayed_steps']==100 and p['A_reuse']['exact_weights'] and p['A_reuse']['exact_Adam'],'Reuse/state proof')
 z=read(ROOT/'diagnostics/initial_reference.json')['rows']
 # The fixed first image is also the largest image in this validation list.
 # Audit the required ID union, rather than demanding duplicate inference rows.
 import yaml
 from PIL import Image
 sys.path.insert(0,plan['source_root'])
 from rgb_final_confirmation.data import load_manifest
 cfg=yaml.safe_load(Path(plan['source_config']).read_text());selected=[r for r in load_manifest(cfg['paths']['manifest']) if r.sample_id in plan['validation_ids']]
 largest=max(selected,key=lambda r:Image.open(r.official_rgb).width*Image.open(r.official_rgb).height).sample_id
 require({r['sample_id'] for r in z}=={plan['validation_ids'][0],largest} and len(z)==p['zero_identity_frames'],'Initial first/largest full frame union')
 for r in z:require(len({r['output_sha256'][a] for a in ['G6','B','R']})==1,'Initial image identity')
 init=torch.load(ROOT/'weights/initial.pt',map_location='cpu',weights_only=False);final=torch.load(ROOT/'weights/final.pt',map_location='cpu',weights_only=False)
 source=torch.load(plan['g6_checkpoint'],map_location='cpu',weights_only=False)
 require(init['step']==0 and final['step']==1000 and final['global_step']==4000,'Endpoints')
 require(final['protocol_id']==plan['protocol_id'] and final['schedule_sha256']==sha(plan['schedule_path']),'Checkpoint provenance')
 require(equal(init['blocks']['B'],init['blocks']['R']),'Blocks same initialization')
 require(init['parameter_names']==final['parameter_names'],'Parameter order')
 report={}
 for a in ['B','R']:
  require(equal(init['weights'][a],source['tasknet_net_g']) and not init['optimizers'][a]['state'],'G6/fresh Adam')
  require(any(not torch.equal(v,init['weights'][a][n]) for n,v in final['weights'][a].items()),'No D2 update')
  all_tensors={**final['weights'][a],**{'blocks.'+n:v for n,v in final['blocks'][a].items()}}
  for v in all_tensors.values():require(bool(torch.isfinite(v).all()),'Nonfinite weight')
  opt=final['optimizers'][a];missing=[]
  for group,names,lr in zip(opt['param_groups'],final['parameter_names'][a],[1e-5,2e-5,1e-4]):
   require(group['lr']==lr and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==0,'Adam hyperparameters')
   require(len(group['params'])==len(names),'Parameter mapping')
   for pid,n in zip(group['params'],names):
    if pid not in opt['state']:missing.append(n);continue
    s=opt['state'][pid];require(int(s['step'])==1000,'Adam endpoint counter')
    for key in ['exp_avg','exp_avg_sq']:require(s[key].shape==all_tensors[n].shape and bool(torch.isfinite(s[key]).all()),'Adam shape/finite')
  allowed={'blocks.enhancer.de_reduce_chan2.weight'}
  if a=='B':allowed.update(run.arch.inactive_expected()['B'])
  require(set(missing)<=allowed,'Unexpected inactive '+str(missing))
  for prefix in ['enhancer.img_feature_extractor.','enhancer.mid_output.','feedback.inject.']:
   require(any(not torch.equal(v,init['blocks'][a][n]) for n,v in final['blocks'][a].items() if n.startswith(prefix)),'No block update')
  report[a]=dict(adam_states=len(opt['state']),adam_step=1000,inactive=missing)
 require(equal(torch.load(ROOT/'weights/last.pt',map_location='cpu',weights_only=False),final),'Last/final')
 with (ROOT/'logs/training.csv').open(newline='') as f:logs=list(csv.DictReader(f))
 require([int(r['step']) for r in logs]==list(range(1,1001)),'Continuous updates')
 oldlogs=list(csv.DictReader((run.OLD39/'logs/training.csv').open()))
 require([r['sample_ids'] for r in logs]==[r['sample_ids'] for r in oldlogs],'Matched39 sample order')
 for i,r in enumerate(logs,1):
  require(int(r['global_step'])==3000+i,'Global random key')
  for a in ['B','R']:
   for key in ['loss','grad_norm','block_grad_norm']:require(math.isfinite(float(r[a+'_'+key])) and float(r[a+'_'+key])>0,'Training finite/update')
 for step in [1,*range(100,1001,100)]:
  calls=read(ROOT/'diagnostics'/f'noise_{step:04d}.json')['noise_calls']
  require([c['arm'] for c in calls]==['G6','B','R'] and len({c['sha256'] for c in calls})==1 and all(c['shape']==[2,16,256] for c in calls),'Actual train noise')
 f=read(ROOT/'diagnostics/frozen_final.json');require(f['passed'] and f['before']==f['after'],'Frozen source')
 require(read(ROOT/'diagnostics/training_complete.json')['final_sha256']==sha(ROOT/'weights/final.pt'),'Final digest')
 rows=read(ROOT/'metrics/step_1000.json')['rows'];require([r['sample_id'] for r in rows]==plan['validation_ids'],'64 IDs')
 old={r['sample_id']:r for r in read(plan['reused_A_metrics'])['rows']}
 for r in rows:
  prev=old[r['sample_id']];require(r['original_size'] and r['shape']==prev['shape'] and r['scene']==prev['scene'] and r['full_index']==prev['full_index'],'Original image/scene/index')
  require(r['metrics']['A']==prev['metrics']['RESET'] and r['metrics']['G6']==prev['metrics']['G6'],'Reused values')
  require(r['output_sha256']['A']==prev['output_sha256']['RESET'] and r['output_sha256']['G6']==prev['output_sha256']['G6'],'Same T1/A hashes')
  require(len(r['noise_calls'])==3 and all(c['shape']==[1,16,256] for c in r['noise_calls']) and len({c['sha256'] for c in r['noise_calls']})==1,'Full image noise')
  for m in r['metrics'].values():require(all(math.isfinite(x) for x in m.values()),'Metric finite')
 comparisons={a+'-'+b:pair(rows,a,b) for a,b in run.PAIRS};summary=read(ROOT/'reports/summary.json');near(summary['comparisons'],comparisons)
 gates={a:all(comparisons[a+'-'+b]['psnr']['scene_equal']>=.05 and comparisons[a+'-'+b]['ssim']['scene_equal']>=0 and comparisons[a+'-'+b]['psnr']['positive_scenes']>=6 for b in ['A','G6']) for a in ['B','R']}
 require(summary['gates']==gates and summary['next_action']==('register_same_endpoint_full872_188_confirmation' if any(gates.values()) else 'stop_instance'),'Gates/stop')
 return dict(passed=True,cpu_only=True,protocol_id=plan['protocol_id'],steps=1000,validation=64,test=0,checkpoints=report,comparisons=comparisons,gates=gates,
  negative_scenes={k:{m:{s:v for s,v in d[m]['scenes'].items() if v<0} for m in ['psnr','ssim']} for k,d in comparisons.items()},
  scope='Independent source/checkpoint/arithmetic audit; not independent image inference',time_unix=time.time())
def archive(plan):
 manifest=read(ROOT/'artifact_manifest.json');require(manifest['protocol_id']==plan['protocol_id'],'Archive protocol');names=set()
 for r in manifest['files']:
  p=(ROOT/r['path']).resolve();require(p.is_relative_to(ROOT) and r['path'] not in names,'Archive boundary');names.add(r['path']);require(p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'Archive '+r['path'])
 require({'config/locked.json','config/preflight.json','weights/initial.pt','weights/final.pt','weights/last.pt','metrics/step_1000.json','reports/summary.json','reports/independent_audit.json','logs/training.csv'}<=names,'Archive evidence')
 require('reports/archive_audit.json' not in names and 'reports/final_review_closeout.json' not in names,'Self covering receipt')
 return dict(passed=True,files=len(names),protocol_id=plan['protocol_id'],manifest_sha256=sha(ROOT/'artifact_manifest.json'),time_unix=time.time())
if __name__=='__main__':
 plan=read(ROOT/'config/locked.json');dest=ROOT/'reports'/('archive_audit.json' if '--archive' in sys.argv else 'independent_audit.json')
 try:result=archive(plan) if '--archive' in sys.argv else audit(plan)
 except Exception:run.write(dest,dict(passed=False,traceback=traceback.format_exc()));raise
 run.write(dest,result);print('Audit passed',flush=True)
