"""CPU-only source, optimizer-state, endpoint, paired arithmetic and archive audit39."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,json,math,statistics,sys,time,traceback
from pathlib import Path
import torch
import run_study as run
ROOT=Path(__file__).resolve().parents[1]
read,sha,equal=run.read,run.sha,run.equal

def require(v,msg):
 if not v:raise AssertionError(msg)
def pair(rows,a,b):
 scenes=sorted({r['scene'] for r in rows});out={}
 for metric in ['psnr','ssim']:
  per={s:statistics.mean(r['metrics'][a][metric]-r['metrics'][b][metric] for r in rows if r['scene']==s) for s in scenes}
  out[metric]=dict(scene_equal=statistics.mean(per.values()),scenes=per,positive_scenes=sum(x>0 for x in per.values()))
 return out
def near(a,b):
 if isinstance(b,dict):
  require(a.keys()==b.keys(),'Summary keys')
  for k,v in b.items():near(a[k],v)
 elif isinstance(b,(float,int)):require(abs(float(a)-float(b))<=1e-10,'Summary arithmetic')
 else:require(a==b,'Summary value')
def audit(plan):
 run.verify(plan);require(not torch.cuda.is_available(),'CPU audit only');torch.set_num_threads(2)
 require(plan['steps']==1000 and plan['arms']==['KEEP','RESET'],'Scope')
 pre=read(ROOT/'config/preflight.json')
 for k in ['passed','initial_weights_exact','source_adam_3000','reset_adam_empty','no_parameter_alias','initial_gradients_equal','same_optimizer_update_exact','resume_exact','source_frozen','all_preflight_updates_discarded']:
  require(pre[k] is True,'Preflight '+k)
 zero=read(ROOT/'diagnostics/initial_reference.json');require(len(zero['rows'])>=16,'Initial originals')
 for r in zero['rows']:require(len(set(r['output_sha256'].values()))==1,'Initial output equality')
 ck0=torch.load(ROOT/'weights/initial.pt',map_location='cpu',weights_only=False)
 ck1=torch.load(ROOT/'weights/final.pt',map_location='cpu',weights_only=False)
 source=torch.load(plan['g6_checkpoint'],map_location='cpu',weights_only=False)
 require(ck0['step']==0 and ck1['step']==1000 and ck1['global_step']==4000,'Endpoints')
 require(equal(ck0['weights']['KEEP'],source['tasknet_net_g']) and equal(ck0['weights']['RESET'],source['tasknet_net_g']),'Same G6 initial weights')
 require(equal(ck0['optimizers']['KEEP'],source['optimizer']),'Original Adam not exact')
 require(not ck0['optimizers']['RESET']['state'],'RESET not fresh')
 require(ck0['parameter_names']==ck1['parameter_names'],'Parameter order changed')
 checkpoint_audit={}
 for a,end in [('KEEP',4000),('RESET',1000)]:
  require(ck1['protocol_id']==plan['protocol_id'],'Checkpoint protocol')
  require(ck1['schedule_sha256']==sha(plan['schedule_path']),'Checkpoint data schedule')
  require(len(ck1['optimizers'][a]['state'])==436,'Missing Adam states')
  require(ck0['weights'][a].keys()==ck1['weights'][a].keys(),'Weight keys')
  changed=[]
  for name,value in ck1['weights'][a].items():
   require(bool(torch.isfinite(value).all()),'Nonfinite weight')
   if not torch.equal(value,ck0['weights'][a][name]):changed.append(name)
  require(bool(changed),'No updates')
  op=ck1['optimizers'][a]
  for g in op['param_groups']:
   require(g['lr']==1e-5 and tuple(g['betas'])==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==0,'Adam hyperparameters')
   for pid,name in zip(g['params'],ck1['parameter_names'][a]):
    st=op['state'][pid];require(int(st['step'])==end,'Wrong Adam counter')
    for v in ['exp_avg','exp_avg_sq']:require(st[v].shape==ck1['weights'][a][name].shape and bool(torch.isfinite(st[v]).all()),'Adam finite/shape')
  checkpoint_audit[a]=dict(adam_states=436,adam_step=end,changed_tensors=len(changed))
 last=torch.load(ROOT/'weights/last.pt',map_location='cpu',weights_only=False);require(equal(last,ck1),'last/final differ');del last
 for step in [100,300,1000]:
  v=torch.load(ROOT/'weights'/f'step_{step:04d}.pt',map_location='cpu',weights_only=False)
  require(v['step']==step and v['protocol_id']==plan['protocol_id'],'Fixed checkpoint missing/mismatch');del v
 with (ROOT/'logs/training.csv').open(newline='') as f:logs=list(csv.DictReader(f))
 require([int(r['step']) for r in logs]==list(range(1,1001)),'Training gap')
 schedule=read(plan['schedule_path']);require(len(schedule['items'])==2000,'Presentations')
 sys.path.insert(0,plan['source_root'])
 from rgb_final_confirmation.data import load_manifest,select_rows
 import yaml
 cfg=yaml.safe_load(Path(plan['source_config']).read_text());train=select_rows(load_manifest(cfg['paths']['manifest']),'train')
 expected=[';'.join(train[schedule['items'][i+j]['sample_index']].sample_id for j in [0,1]) for i in range(0,2000,2)]
 require([r['sample_ids'] for r in logs]==expected,'Training order')
 for i,r in enumerate(logs,1):
  require(int(r['global_step'])==3000+i,'Noise/data step wrong')
  for a in ['KEEP','RESET']:
   require(math.isfinite(float(r[a+'_loss'])) and float(r[a+'_grad_norm'])>0 and math.isfinite(float(r[a+'_grad_norm'])),'Training nonfinite')
 for step in [1,100,200,300,400,500,600,700,800,900,1000]:
  n=read(ROOT/'diagnostics'/f'noise_{step:04d}.json');calls=n['noise_calls']
  require(len(calls)==2 and {r['arm'] for r in calls}=={'KEEP','RESET'},'Noise stages')
  require(all(r['shape']==[2,16,256] for r in calls) and len({r['sha256'] for r in calls})==1,'Actual noise unpaired')
  if step==1:require(n['initial_gradients_equal'] and len(set(n['gradient_hashes'].values()))==1,'First gradients')
 f=read(ROOT/'diagnostics/frozen_final.json');require(f['passed'] and f['before']==f['after'],'Prior changed')
 completion=read(ROOT/'diagnostics/training_complete.json');require(completion['steps']==1000 and completion['final_sha256']==sha(ROOT/'weights/final.pt'),'Final SHA')
 refs={r['sample_id']:r for r in read(plan['reference_validation872'])['per_image']};summ=read(ROOT/'reports/summary.json');calculations={}
 for step,ids in [(100,plan['probe_ids']),(300,plan['probe_ids']),(1000,plan['validation_ids'])]:
  v=read(ROOT/'metrics'/f'step_{step:04d}.json');rws=v['rows']
  require(v['step']==step and v['protocol_id']==plan['protocol_id'],'Metric endpoint')
  require([r['sample_id'] for r in rws]==ids and len(set(ids))==len(ids),'Evaluation IDs')
  for r in rws:
   ref=refs[r['sample_id']];require(r['original_size'] and r['full_index']==ref['index'] and r['scene']==ref['scene'],'Original/index')
   require(len(r['noise_calls'])==3 and len({x['sha256'] for x in r['noise_calls']})==1,'Evaluation noise')
   require(all(x['shape']==[1,16,256] for x in r['noise_calls']),'Evaluation noise shape')
   require(abs(r['metrics']['G6']['psnr']-ref['psnr'])<=1e-7 and abs(r['metrics']['G6']['ssim']-ref['ssim'])<=1e-9,'G6 reference')
   for m in r['metrics'].values():require(all(math.isfinite(x) for x in m.values()),'Nonfinite metrics')
  calculations[str(step)]={a+'-'+b:pair(rws,a,b) for a,b in [('KEEP','G6'),('RESET','G6'),('KEEP','RESET')]}
 near(summ['evaluations'],calculations)
 end=calculations['1000'];d,k,r=end['KEEP-RESET'],end['KEEP-G6'],end['RESET-G6']
 if d['psnr']['scene_equal']>=.1 and d['ssim']['scene_equal']>=0 and d['psnr']['positive_scenes']>=6 and k['psnr']['scene_equal']>=-.05 and k['ssim']['scene_equal']>=-.0002 and r['psnr']['scene_equal']<=-.1:interpretation='supports_Adam_state_retention_in_this_short_run'
 elif k['psnr']['scene_equal']<=-.1 and r['psnr']['scene_equal']<=-.1:interpretation='both_decline_Adam_retention_not_sufficient'
 elif d['psnr']['scene_equal']<=-.1 and d['ssim']['scene_equal']<=0 and sum(v<0 for v in d['psnr']['scenes'].values())>=6:interpretation='reset_better_does_not_support_reset_harm'
 else:interpretation='inconclusive_or_mixed_short_run'
 require(summ['interpretation']==interpretation and summ['test']==0 and summ['steps']==1000,'Classification')
 return dict(passed=True,cpu_only=True,protocol_id=plan['protocol_id'],steps=1000,validation=64,probes=[16,16],test=0,
  checkpoints=checkpoint_audit,evaluations=calculations,interpretation=interpretation,
  negative_scenes={label:{m:{s:x for s,x in v[m]['scenes'].items() if x<0} for m in ['psnr','ssim']} for label,v in end.items()},
  final_sha256=sha(ROOT/'weights/final.pt'),time_unix=time.time(),scope='source/checkpoint/Adam/arithmetic audit, not independent image inference')
def archive(plan):
 m=read(ROOT/'artifact_manifest.json');require(m['protocol_id']==plan['protocol_id'],'Archive protocol');names=set()
 for r in m['files']:
  p=(ROOT/r['path']).resolve();require(p.is_relative_to(ROOT) and r['path'] not in names,'Archive boundary/duplicate');names.add(r['path'])
  require(p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'Archive mismatch '+r['path'])
 require({'config/locked.json','config/preflight.json','weights/initial.pt','weights/final.pt','weights/last.pt','metrics/step_0100.json','metrics/step_0300.json','metrics/step_1000.json','reports/independent_audit.json','reports/summary.json','logs/training.csv'}<=names,'Missing evidence')
 require('reports/archive_audit.json' not in names and 'reports/final_review_closeout.json' not in names,'Self-overwriting receipts')
 require(read(ROOT/'reports/independent_audit.json')['passed'],'Scientific audit')
 return dict(passed=True,protocol_id=plan['protocol_id'],files=len(names),artifact_manifest_sha256=sha(ROOT/'artifact_manifest.json'),time_unix=time.time())
if __name__=='__main__':
 p=read(ROOT/'config/locked.json');dest=ROOT/'reports'/('archive_audit.json' if '--archive' in sys.argv else 'independent_audit.json')
 try:result=archive(p) if '--archive' in sys.argv else audit(p)
 except Exception:
  run.write(dest,dict(passed=False,traceback=traceback.format_exc(),time_unix=time.time()));raise
 run.write(dest,result);print(json.dumps(dict(passed=True,archive='--archive' in sys.argv)),flush=True)
