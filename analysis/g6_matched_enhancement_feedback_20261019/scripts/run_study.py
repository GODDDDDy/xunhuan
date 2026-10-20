"""40: reuse39 RESET, paired true-batch2 enhancement and previous-state feedback.

Reuse the verified39 data/loss/noise code and38 architecture without changing
either historical run. New glue is necessary to combine their two contracts.
"""
from __future__ import annotations
import copy,csv,fcntl,gc,importlib.util,json,math,os,random,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent
OLD39=PROJECT/'g6_adam_state_diagnostic_20261019'
OLD38=PROJECT/'g6_enhancement_feedback_20261019'
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
h=module('verified39',OLD39/'scripts/run_study.py')
sys.path.insert(0,str(OLD38/'scripts'))
arch=module('architecture38',OLD38/'scripts/run_g6_feedback.py')
read,write,sha,canonical,receipt=h.read,h.write,h.sha,h.canonical,h.receipt
cpu,equal,rng,set_rng,save,seed_step=h.cpu,h.equal,h.rng,h.set_rng,h.save,h.seed_step
DEVICE=h.DEVICE;ARMS=['B','R'];h.ROOT=ROOT
def status(stage,**kw):
 value=dict(stage=stage,time_unix=time.time(),**kw);write(ROOT/'status.json',value);print(json.dumps(value),flush=True)
def verify(plan):
 unsigned=dict(plan);unsigned.pop('protocol_id');assert canonical(unsigned)==plan['protocol_id']
 assert plan['experiment']==40 and plan['arms']==ARMS and plan['steps']==1000
 assert canonical(plan['scientific'])==plan['scientific_id']
 replacements={};codes={r['path']:r for r in plan['code_sources']}
 rp=ROOT/'config/implementation_revisions.json'
 for rev in read(rp) if rp.exists() else []:
  assert rev['protocol_id']==plan['protocol_id'] and rev['scientific_id']==plan['scientific_id']
  for r in rev['replacements']:
   assert r['path'] in codes and r['original_sha256']==codes[r['path']]['sha256']
   assert sha(r['preserved_original_path'])==r['original_sha256'];replacements[r['path']]=r['repaired_sha256']
 for r in plan['sources']+plan['code_sources']:assert sha(r['path'])==replacements.get(r['path'],r['sha256']),r['path']
def prepare():
 for name in ['config','logs','weights','metrics','reports','diagnostics','failures','cache']:(ROOT/name).mkdir(parents=True,exist_ok=True)
 lp=ROOT/'config/locked.json'
 if lp.exists():verify(read(lp));return
 assert not (ROOT/'worker_registration.json').exists()
 p39=read(OLD39/'config/locked.json');assert p39['protocol_id']=='a56947cbbdc21d61220668046b3ae949c1103dff9031f01712bd258e9dd0d2df'
 assert read(OLD39/'closeout.json')['success']
 archive=read(OLD39/'artifact_manifest.json')
 for r in archive['files']:assert sha(OLD39/r['path'])==r['sha256']
 sp=ROOT/'config/training1000.json';write(sp,read(p39['schedule_path']));assert sha(sp)==sha(p39['schedule_path'])
 pre=PROJECT/'polar_residual_feedback_joint_20261010/pretraining/checkpoints/selected.pt'
 assert sha(pre)=='a4ecebacba5cfc25546320974a9c37f1cc220c685f1318b3cc290ae4a68e9554'
 scientific=dict(experiment=40,arms=ARMS,reused_control='39 RESET',steps=1000,seed=2027,batch=2,crop=128,
  g6_sha256=h.G6_SHA,pidsr_sha256=sha(pre),schedule_sha256=sha(sp),global_steps=[3001,4000],
  D2_lr=1e-5,E_body_lr=2e-5,adapter_lr=1e-4,optimizer='fresh Adam(.9,.999),eps1e-8,wd0',
  clip='D2 separately1; E+adapters separately1',loss=p39['scientific']['loss'],
  task_mode='eval gradients',enhancer_mode='train checkpointing',amp=False,scheduler=None,
  architecture='unchanged38 enhancement and decoder adapters; B excludes T1/F1, R reads fixed G6 T1/F1',
  polarization='P2=clamp(P0+E_residual,0,1); observation-anchored floating bridge',
  noise='explicit paired [2,16,256], original39 per-global-step seed',H_C=False,RGB_residual_head=False,
  validation_ids=p39['validation_ids'],test=0,endpoint_only=True,
  gate=dict(psnr=.05,ssim=0,positive_scenes=6,comparators=['A','G6']),
  confirmation='if any candidate passes, same1000 weights all A/B/R872val+188test; no new training',
  stop='if neither passes stop; no extension/seeds/LR/loss/amplitude/C/Oracle search')
 sources=copy.deepcopy(p39['sources'])
 sources += [receipt(p,r) for p,r in [(ROOT/'PROTOCOL_ZH.md','registered40 protocol'),(sp,'identical39 schedule'),(pre,'independent E pretraining'),(OLD39/'config/locked.json','39 immutable plan'),(OLD39/'artifact_manifest.json','39 audited archive'),(OLD39/'weights/step_0100.pt','A100 exact replay target'),(OLD39/'weights/final.pt','fixed A1000 weights'),(OLD39/'metrics/step_1000.json','reused A and G6 same64')]]
 codes={Path(r['path']) for r in p39['code_sources']}
 codes.update((ROOT/'scripts').glob('*.py'));codes.update((OLD38/'scripts').glob('*.py'))
 codes.update((PROJECT/'polar_residual_feedback_joint_20261010/scripts').rglob('*.py'))
 plan=dict(experiment=40,arms=ARMS,steps=1000,exploratory_only=True,g6_checkpoint=p39['g6_checkpoint'],
  source_config=p39['source_config'],source_root=p39['source_root'],reference_validation872=p39['reference_validation872'],
  schedule_path=str(sp),validation_ids=p39['validation_ids'],full_index_map=p39['full_index_map'],
  pidsr_checkpoint=str(pre),reused_A_weights=str(OLD39/'weights/final.pt'),reused_A_metrics=str(OLD39/'metrics/step_1000.json'),
  scientific=scientific,scientific_id=canonical(scientific),sources=sources,code_sources=[receipt(p) for p in sorted(codes)],deadline_unix=None)
 plan['protocol_id']=canonical(plan);write(lp,plan);verify(plan)
 vgg=Path('/home/li/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth');link=ROOT/'experiments/pretrained_models'/vgg.name;link.parent.mkdir(parents=True,exist_ok=True)
 if not link.exists():link.symlink_to(vgg)
 print(json.dumps(dict(prepared=True,protocol_id=plan['protocol_id'],scientific_id=plan['scientific_id'])),flush=True)
def components(plan):
 #39 construction fixes source, loss, task eval mode and original parameter order.
 c=h.components(plan);c['tasks']={'B':c['tasks']['RESET'],'R':c['tasks']['KEEP']};del c['opts'],c['names']
 pre=torch.load(plan['pidsr_checkpoint'],map_location='cpu',weights_only=False)
 torch.manual_seed(2027);torch.cuda.manual_seed_all(2027)
 core=arch.PolarEnhancer();core.load_state_dict(pre['enhancer'],strict=True);del pre
 for m in [core.mid_output,*[a.out for a in core.adapters.values()]]:
  nn.init.zeros_(m.weight)
  if m.bias is not None:nn.init.zeros_(m.bias)
 core.downstream_parameters();core.activation_checkpoint=True
 initial=arch.EnhancementBlock(core).to(DEVICE);c['blocks']={a:copy.deepcopy(initial).train() for a in ARMS};del initial,core
 c['opts']={};c['names']={}
 for a in ARMS:
  block=c['blocks'][a];task=list(c['tasks'][a].net_g.named_parameters());named=list(block.named_parameters())
  body=[(n,p) for n,p in named if p.requires_grad and n.startswith('enhancer.') and not n.startswith('enhancer.adapters.')]
  adapter=[(n,p) for n,p in named if p.requires_grad and (n.startswith('enhancer.adapters.') or n.startswith('feedback.'))]
  groups=[dict(params=[p for n,p in task],lr=1e-5,name='tasknet_net_g'),dict(params=[p for n,p in body],lr=2e-5,name='enhancer_body'),dict(params=[p for n,p in adapter],lr=1e-4,name='adapters_feedback')]
  c['names'][a]=[[n for n,p in task],['blocks.'+n for n,p in body],['blocks.'+n for n,p in adapter]]
  c['opts'][a]=torch.optim.Adam(groups,betas=(.9,.999),eps=1e-8,weight_decay=0)
 return c
def snapshot(c,step,plan):
 return dict(step=step,global_step=3000+step,protocol_id=plan['protocol_id'],schedule_sha256=sha(plan['schedule_path']),
  weights={a:cpu(t.net_g.state_dict()) for a,t in c['tasks'].items()},blocks={a:cpu(b.state_dict()) for a,b in c['blocks'].items()},
  optimizers={a:cpu(o.state_dict()) for a,o in c['opts'].items()},parameter_names=copy.deepcopy(c['names']),rng=rng())
def restore(c,v,plan):
 assert v['protocol_id']==plan['protocol_id'] and v['schedule_sha256']==sha(plan['schedule_path']) and v['parameter_names']==c['names']
 for a in ARMS:
  c['tasks'][a].net_g.load_state_dict(v['weights'][a],strict=True);c['blocks'][a].load_state_dict(v['blocks'][a],strict=True)
  c['opts'][a].load_state_dict(v['optimizers'][a]);c['opts'][a].zero_grad(set_to_none=True)
 set_rng(v['rng'])
def frozen(c):
 return dict(**h.frozen(c),**{a+'_inactive_E':arch.fingerprint(b,True) for a,b in c['blocks'].items()})
def assert_frozen(c):
 h.assert_frozen(c)
 for b in c['blocks'].values():
  for p in b.parameters():
   if not p.requires_grad:assert p.grad is None
@torch.no_grad()
def first(c,batch,noise):
 t,f=c['baseline'](batch['polar'],batch['official_rgb'],noise)
 return dict(t1=t.detach(),features={k:v.detach() for k,v in f.items()})
def predict(c,a,batch,state,noise,no_state=False):
 block=c['blocks'][a];available=a=='R' and not no_state;t=state['t1'] if available else None;f=state['features'] if available else None
 p,polar,_,_=block.enhancer(batch['polar'],batch['polar'],batch['official_rgb'],t,f,None)
 inputs=arch.anchored_task_inputs(p,batch['polar'],batch['official_rgb'])
 context=dict(image=batch['official_rgb'],available=available,t1=t,features=f);hooks=[]
 try:
  for level in ['level3','level2']:
   def hook(module,args,output,level=level):return block.feedback.fuse(output,level,context,polar)
   hooks.append(getattr(c['tasks'][a].net_g,'decoder_'+level).register_forward_hook(hook))
  pred,features=c['tasks'][a].forward_inputs(inputs,noise)
 finally:
  for hook in hooks:hook.remove()
 return pred
def update(c,plan,batch,step,same_input=False,gradient_check=False):
 for o in c['opts'].values():o.zero_grad(set_to_none=True)
 seed_step(2027,'train',3000+step,batch['sample_id']);noise=torch.randn(2,16,256,device=DEVICE)
 rec=dict(step=step,global_step=3000+step,sample_ids=';'.join(batch['sample_id']));gradient={}
 with h.NoiseCheck(dict(G6=c['baseline'],**c['tasks']),noise) as proof:
  state=first(c,batch,noise)
  for a in ARMS:
   pred=predict(c,a,batch,state,noise,no_state=same_input);loss=c['criterion'](pred,batch['target']);assert torch.isfinite(loss)
   loss.backward();rec[a+'_loss']=float(loss.detach());del pred,loss
   if gradient_check:gradient[a]={n:h.tensor_sha(p.grad) for n,p in c['tasks'][a].net_g.named_parameters() if p.grad is not None}
   rec[a+'_grad_norm']=float(torch.nn.utils.clip_grad_norm_(c['tasks'][a].net_g.parameters(),1.,error_if_nonfinite=True))
   active=[p for p in c['blocks'][a].parameters() if p.requires_grad]
   rec[a+'_block_grad_norm']=float(torch.nn.utils.clip_grad_norm_(active,1.,error_if_nonfinite=True))
   assert rec[a+'_grad_norm']>0 and rec[a+'_block_grad_norm']>0
  assert [r['arm'] for r in proof.calls]==['G6','B','R']
  for o in c['opts'].values():o.step()
 return rec,dict(step=step,noise_calls=proof.calls,gradient_check=gradient_check,netg_gradient_equal=equal(gradient.get('B'),gradient.get('R')) if gradient_check else None)
def replay_A(c,plan):
 task=copy.deepcopy(c['baseline']).eval();task.net_g.requires_grad_(True)
 opt=torch.optim.Adam([dict(params=list(task.net_g.parameters()),lr=1e-5,name='tasknet_net_g')],betas=(.9,.999),eps=1e-8,weight_decay=0)
 tick=time.time()
 for step,batch in enumerate(h.batches(c,plan,0,100),1):
  opt.zero_grad(set_to_none=True);seed_step(2027,'train',3000+step,batch['sample_id']);noise=torch.randn(2,16,256,device=DEVICE)
  with h.NoiseCheck({'A':task},noise):pred,features=task(batch['polar'],batch['official_rgb'],noise)
  loss=c['criterion'](pred,batch['target']);loss.backward();torch.nn.utils.clip_grad_norm_(task.net_g.parameters(),1.,error_if_nonfinite=True);opt.step();del pred,features,loss
 old=torch.load(OLD39/'weights/step_0100.pt',map_location='cpu',weights_only=False)
 assert equal(task.net_g.state_dict(),old['weights']['RESET']) and equal(opt.state_dict(),old['optimizers']['RESET'])
 result=dict(passed=True,replayed_steps=100,exact_weights=True,exact_Adam=True,seconds=time.time()-tick,source_sha256=sha(OLD39/'weights/step_0100.pt'))
 del task,opt,old,batch;gc.collect();torch.cuda.empty_cache();return result
@torch.no_grad()
def evaluate_rows(c,plan,ids,path,identity=False):
 from rgb_final_confirmation.evaluation import psnr,ssim
 old={r['sample_id']:r for r in read(plan['reused_A_metrics'])['rows']};saved=rng();results=read(path)['rows'] if path.exists() else []
 assert [r['sample_id'] for r in results]==ids[:len(results)]
 try:
  for sid in ids[len(results):]:
   batch=h.full_item(c,sid);seed_step(2026,'validation',plan['full_index_map'][sid],[sid]);noise=torch.randn(1,16,256,device=DEVICE)
   metrics={'A':old[sid]['metrics']['RESET']};hashes={'A':old[sid]['output_sha256']['RESET']}
   with h.NoiseCheck(dict(G6=c['baseline'],**c['tasks']),noise) as proof:
    state=first(c,batch,noise);metrics['G6']=dict(psnr=psnr(state['t1'],batch['target']),ssim=ssim(state['t1'],batch['target']));hashes['G6']=h.tensor_sha(state['t1'])
    assert hashes['G6']==old[sid]['output_sha256']['G6'] and metrics['G6']==old[sid]['metrics']['G6']
    for a in ARMS:
     pred=predict(c,a,batch,state,noise);metrics[a]=dict(psnr=psnr(pred,batch['target']),ssim=ssim(pred,batch['target']));hashes[a]=h.tensor_sha(pred)
     if identity:assert torch.equal(pred,state['t1']),(sid,a,'zero identity')
     del pred
   results.append(dict(sample_id=sid,scene=batch['scene'][0],shape=list(batch['target'].shape),original_size=True,full_index=plan['full_index_map'][sid],metrics=metrics,output_sha256=hashes,noise_calls=proof.calls,A_reused=True))
   write(path,dict(protocol_id=plan['protocol_id'],step=0 if identity else 1000,frames=len(results),rows=results))
   status('preflight_identity' if identity else 'evaluating',completed=len(results),total=len(ids))
   del state,batch;gc.collect();torch.cuda.empty_cache()
 finally:set_rng(saved)
 return results
def preflight(plan):
 fp=ROOT/'config/preflight.json'
 if fp.exists():assert read(fp)['passed'];return
 status('preflight_loading');c=components(plan);torch.cuda.reset_peak_memory_stats();before=frozen(c);initial=snapshot(c,0,plan)
 assert equal(initial['weights']['B'],initial['weights']['R']) and equal(initial['blocks']['B'],initial['blocks']['R'])
 sets=[{p.data_ptr() for p in m.parameters()} for m in [c['baseline'],*c['tasks'].values(),*c['blocks'].values()]]
 assert all(not sets[i]&sets[j] for i in range(len(sets)) for j in range(i))
 batch=next(h.batches(c,plan,0,1));tick=time.time();rec,proof=update(c,plan,batch,1,same_input=True,gradient_check=True);torch.cuda.synchronize();seconds=time.time()-tick
 assert proof['netg_gradient_equal'] and rec['B_loss']==rec['R_loss']
 after=snapshot(c,1,plan);assert equal(after['weights']['B'],after['weights']['R']) and equal(after['blocks']['B'],after['blocks']['R']) and equal(after['optimizers']['B'],after['optimizers']['R'])
 restore(c,initial,plan)
 # Nonzero interfaces prevent zero output layers from concealing state leakage.
 with torch.no_grad():
  for a in ARMS:
   for n,p in c['blocks'][a].named_parameters():
    if ('feedback.inject.' in n and n.endswith(('weight','bias'))) or ('.out.' in n):p.add_(torch.randn_like(p)*.001)
  seed_step(2027,'train',3001,batch['sample_id']);noise=torch.randn(2,16,256,device=DEVICE);state=first(c,batch,noise)
  changed=dict(t1=state['t1']+.1,features={k:v+.1 for k,v in state['features'].items()})
  b0=predict(c,'B',batch,state,noise);b1=predict(c,'B',batch,changed,noise);assert torch.equal(b0,b1)
  for values in c['blocks']['B'].feedback.last_fuse_values.values():assert not values[:,16:32].count_nonzero() and not values[:,51:58].count_nonzero()
  r0=predict(c,'R',batch,state,noise);r1=predict(c,'R',batch,changed,noise);sensitivity=float((r1-r0).abs().max());assert sensitivity>0
  del state,changed,b0,b1,r0,r1
 restore(c,initial,plan)
 for step,batch in enumerate(h.batches(c,plan,0,3),1):update(c,plan,batch,step)
 trained=snapshot(c,3,plan)
 for a in ARMS:
  for prefix in ['enhancer.img_feature_extractor.','enhancer.mid_output.','feedback.inject.']:
   assert any(not torch.equal(v,initial['blocks'][a][n]) for n,v in trained['blocks'][a].items() if n.startswith(prefix)),(a,prefix,'no update')
 batch=next(h.batches(c,plan,3,4));save(ROOT/'diagnostics/preflight_replay.pt',trained);update(c,plan,batch,4);expected=snapshot(c,4,plan)
 restore(c,torch.load(ROOT/'diagnostics/preflight_replay.pt',map_location='cpu',weights_only=False),plan);update(c,plan,batch,4);assert equal(expected,snapshot(c,4,plan))
 restore(c,initial,plan);assert_frozen(c);assert before==frozen(c)
 status('preflight_A_reuse');reuse=replay_A(c,plan);write(ROOT/'diagnostics/A_reuse.json',reuse)
 restore(c,initial,plan)
 from PIL import Image
 selected=[r for r in h.rows(c) if r.sample_id in plan['validation_ids']];largest=max(selected,key=lambda r:Image.open(r.official_rgb).width*Image.open(r.official_rgb).height).sample_id
 ids=list(dict.fromkeys([plan['validation_ids'][0],largest]));tick=time.time();evaluate_rows(c,plan,ids,ROOT/'diagnostics/initial_reference.json',True);frame=(time.time()-tick)/len(ids)
 assert equal(initial['weights'],snapshot(c,0,plan)['weights']) and equal(initial['blocks'],snapshot(c,0,plan)['blocks'])
 write(fp,dict(passed=True,protocol_id=plan['protocol_id'],initial_equal=True,no_parameter_alias=True,same_input_update_exact=True,
  state_check_nonzero_interfaces=True,B_no_state_leak=True,R_output_sensitivity=sensitivity,resume_exact=True,
  source_frozen=True,all_preflight_updates_discarded=True,A_reuse=reuse,zero_identity_frames=len(ids),
  initial_noise=proof,train_step_seconds=seconds,frame_seconds=frame,estimated_hours=1.3*(1000*seconds+64*frame)/3600,
  peak_allocated_bytes=torch.cuda.max_memory_allocated()))
 del c,initial,after,trained,expected,batch;gc.collect();torch.cuda.empty_cache();status('preflight_passed')
def train(plan):
 if (ROOT/'diagnostics/training_complete.json').exists():return
 c=components(plan);before=frozen(c);last=ROOT/'weights/last.pt';log=ROOT/'logs/training.csv';start=0
 if last.exists():v=torch.load(last,map_location='cpu',weights_only=False);restore(c,v,plan);start=v['step'];del v
 else:v=snapshot(c,0,plan);save(ROOT/'weights/initial.pt',v);save(last,v);del v
 if log.exists():
  old=list(csv.DictReader(log.open()));keep=[r for r in old if int(r['step'])<=start]
  if len(old)>len(keep):write(ROOT/'failures'/f'orphaned_{time.time_ns()}.json',old[len(keep):])
  assert len(keep)==start
  with log.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(old[0]));w.writeheader();w.writerows(keep)
 status('training',step=start,total=1000)
 with log.open('a',newline='') as f:
  writer=None
  for step,batch in enumerate(h.batches(c,plan,start,1000),start+1):
   tick=time.time();rec,proof=update(c,plan,batch,step);torch.cuda.synchronize();rec['seconds']=time.time()-tick
   if writer is None:writer=csv.DictWriter(f,fieldnames=list(rec));writer.writeheader() if start==0 else None
   writer.writerow(rec);f.flush()
   if step==1 or step%100==0:write(ROOT/'diagnostics'/f'noise_{step:04d}.json',proof)
   if step%100==0:save(last,snapshot(c,step,plan))
   if step%10==0:status('training',step=step,total=1000,seconds=rec['seconds'])
 assert_frozen(c);after=frozen(c);assert before==after
 save(ROOT/'weights/final.pt',snapshot(c,1000,plan));write(ROOT/'diagnostics/frozen_final.json',dict(passed=True,before=before,after=after))
 write(ROOT/'diagnostics/training_complete.json',dict(steps=1000,final_sha256=sha(ROOT/'weights/final.pt')))
 del c,batch;gc.collect();torch.cuda.empty_cache()
PAIRS=[('A','G6'),('B','G6'),('R','G6'),('B','A'),('R','A'),('R','B')]
def evaluate(plan):
 if (ROOT/'reports/summary.json').exists():return
 c=components(plan);v=torch.load(ROOT/'weights/final.pt',map_location='cpu',weights_only=False);restore(c,v,plan);del v
 rows=evaluate_rows(c,plan,plan['validation_ids'],ROOT/'metrics/step_1000.json');comparisons={a+'-'+b:h.compare(rows,a,b) for a,b in PAIRS}
 gates={a:all(comparisons[a+'-'+b]['psnr']['scene_equal']>=.05 and comparisons[a+'-'+b]['ssim']['scene_equal']>=0 and comparisons[a+'-'+b]['psnr']['positive_scenes']>=6 for b in ['A','G6']) for a in ARMS}
 write(ROOT/'reports/summary.json',dict(protocol_id=plan['protocol_id'],steps=1000,validation=64,test=0,exploratory_only=True,comparisons=comparisons,gates=gates,
  next_action='register_same_endpoint_full872_188_confirmation' if any(gates.values()) else 'stop_instance',
  limitations=['single_seed','historically_explored64','B-A whole_path_not_polarization_only','independent_D2_not_shared_recurrence','no_C_comparison','not_unique_cause_of38']))
 del c;gc.collect();torch.cuda.empty_cache()
def finish(plan):
 status('independent_audit');env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
 subprocess.run([sys.executable,str(ROOT/'scripts/audit_study.py')],check=True,env=env);verify(plan)
 excluded={'artifact_manifest.json','closeout.json','status.json','controller_heartbeat.json','resource_wait.json','worker_exit.json','worker.log','controller.log','archive_audit.json','final_review_closeout.json'}
 files=[]
 for p in sorted(ROOT.rglob('*')):
  if p.is_symlink() or not p.is_file() or p.name in excluded or p.suffix in ['.pyc','.tmp'] or 'cache' in p.relative_to(ROOT).parts or '__pycache__' in p.parts:continue
  files.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
 write(ROOT/'artifact_manifest.json',dict(protocol_id=plan['protocol_id'],files=files))
 subprocess.run([sys.executable,str(ROOT/'scripts/audit_study.py'),'--archive'],check=True,env=env)
 close=dict(success=True,exploratory_only=True,steps=1000,validation=64,test=0,protocol_id=plan['protocol_id'],archive_manifest_sha256=sha(ROOT/'artifact_manifest.json'),
  independent_audit_verified=True,archive_verified=True,next_action=read(ROOT/'reports/summary.json')['next_action'],time_unix=time.time())
 write(ROOT/'reports/final_review_closeout.json',close);write(ROOT/'closeout.json',close);status('completed',**close)
def run():
 lock=(ROOT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);plan=read(ROOT/'config/locked.json');verify(plan)
 if (ROOT/'closeout.json').exists():assert read(ROOT/'closeout.json')['success'];return
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
 preflight(plan);train(plan);evaluate(plan);finish(plan)
if __name__=='__main__':
 try:prepare() if '--prepare' in sys.argv else run()
 except Exception:
  failure=dict(time_unix=time.time(),traceback=traceback.format_exc());write(ROOT/'failures'/f'worker_{time.time_ns()}.json',failure);status('failed',**failure);raise
