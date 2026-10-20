"""Experiment39: paired true-batch2 single-pass G6 KEEP/RESET Adam diagnosis."""
from __future__ import annotations
import argparse,copy,csv,fcntl,gc,hashlib,json,math,os,random,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent
ARMS=['KEEP','RESET'];DEVICE=torch.device('cuda:0')
G6_SHA='6d4946a5f9963ddb1d142b5f4a97f18d0eedf724769aab4b414997a15a33322f'
VGG_SHA='dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def receipt(p,role=None):
 p=Path(p).resolve();v=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
 if role:v['role']=role
 return v
def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def cpu(x):
 if torch.is_tensor(x):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
 if isinstance(x,list):return [cpu(v) for v in x]
 if isinstance(x,tuple):return tuple(cpu(v) for v in x)
 return copy.deepcopy(x)
def equal(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a.cpu(),b.cpu())
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
 if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def set_rng(v):
 random.setstate(v['python']);np.random.set_state(v['numpy']);torch.set_rng_state(v['torch'].cpu());torch.cuda.set_rng_state_all([t.cpu() for t in v['cuda']])
def seed_step(seed,phase,step,ids):
 v=int.from_bytes(hashlib.sha256(f"{seed}:{phase}:{step}:{'|'.join(ids)}".encode()).digest()[:8],'little')%(2**31)
 torch.manual_seed(v);torch.cuda.manual_seed_all(v);return v
def save(p,v):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');torch.save(v,tmp);tmp.replace(p)
def status(stage,**kw):
 v=dict(stage=stage,time_unix=time.time());v.update(kw);write(ROOT/'status.json',v);print(json.dumps(v),flush=True)
def verify(plan):
 unsigned=dict(plan);unsigned.pop('protocol_id');assert canonical(unsigned)==plan['protocol_id']
 assert plan['experiment']==39 and plan['arms']==ARMS and plan['steps']==1000
 assert canonical(plan['scientific'])==plan['scientific_id']
 replacements={};codes={r['path']:r for r in plan['code_sources']}
 rp=ROOT/'config/implementation_revisions.json'
 for rev in (read(rp) if rp.exists() else []):
  assert rev['protocol_id']==plan['protocol_id'] and rev['scientific_id']==plan['scientific_id']
  for r in rev['replacements']:
   assert r['path'] in codes and r['original_sha256']==codes[r['path']]['sha256']
   assert sha(r['preserved_original_path'])==r['original_sha256'];replacements[r['path']]=r['repaired_sha256']
 for r in plan['sources']+plan['code_sources']:assert sha(r['path'])==replacements.get(r['path'],r['sha256']),r['path']
def prepare():
 for name in ['config','logs','weights','metrics','reports','diagnostics','failures','cache']:(ROOT/name).mkdir(parents=True,exist_ok=True)
 lp=ROOT/'config/locked.json'
 if lp.exists():verify(read(lp));print('Existing plan verified');return
 assert not (ROOT/'worker_registration.json').exists()
 original=PROJECT/'normalized_threeway_20261002';cfgpath=original/'configs/G6_seed2027.yaml'
 cfg=yaml.safe_load(cfgpath.read_text());sys.path.insert(0,cfg['paths']['source_root'])
 from rgb_final_confirmation.data import load_manifest,select_rows,build_training_schedule
 train=select_rows(load_manifest(cfg['paths']['manifest']),'train');assert len(train)==5440
 old=read(cfg['paths']['schedule']);extended=build_training_schedule(train,seed=2027,total_steps=4000,batch_size=2,crop_size=128)
 assert extended['items'][:6000]==old['items'] and len(old['items'])==6000
 tail=dict(schema_version=1,seed=2027,total_steps=1000,batch_size=2,crop_size=128,source_global_steps=[3001,4000],items=extended['items'][6000:8000])
 tail['sha256']=canonical(tail);sp=ROOT/'config/training1000.json';write(sp,tail)
 ids=read(PROJECT/'confidence_correction_causal_20261017/config/validation64.json')['sample_ids']
 assert len(ids)==len(set(ids))==64
 refpath=original/'runs/G6/seed_2027/metrics/step_3000_validation.json';refs=read(refpath)['per_image']
 index={x['sample_id']:x['index'] for x in refs};scenes={x['sample_id']:x['scene'] for x in refs}
 order=sorted(set(scenes[x] for x in ids));probe=[]
 for scene in order:
  group=[x for x in ids if scenes[x]==scene];assert len(group)==8;probe += [group[1],group[6]]
 g6=original/'runs/G6/seed_2027/checkpoints/step_3000.pt';assert sha(g6)==G6_SHA
 raw=yaml.safe_load(Path(cfg['paths']['base_config']).read_text())
 vgg=Path('/home/li/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth');assert sha(vgg)==VGG_SHA
 link=ROOT/'experiments/pretrained_models/vgg19-dcbb9e9d.pth';link.parent.mkdir(parents=True,exist_ok=True)
 if not link.exists():link.symlink_to(vgg)
 assert link.resolve()==vgg.resolve()
 scientific=dict(experiment=39,arms=ARMS,steps=1000,source_step=3000,seed=2027,evaluation_seed=2026,
  g6_sha256=G6_SHA,optimizer_only_variable=True,true_batch=2,crop_size=128,lr=1e-5,betas=[.9,.999],eps=1e-8,weight_decay=0,
  grad_clip=1.0,scheduler=None,warmup=0,amp=False,task_mode='eval_with_netg_gradients',prior='frozen',
  loss='L1+(1-SSIM)+.1phaseMSE+.1VGGL1; inherited G6 VGG numeric convention',
  schedule_sha256=sha(sp),prefix6000_exact=True,noise='train/2027/global_step/paired_batch2_IDs; explicit same [2,16,256]',
  probes=dict(steps=[100,300],ids=probe),final=dict(step=1000,ids=ids),test=0,
  classification=dict(material_psnr=.10,preserve_psnr_floor=-.05,preserve_ssim_floor=-.0002,min_positive_scenes=6),
  stop='Stop after1000+fixed evaluations irrespective of result; no automatic extension/seeds/LR/modules',
  reference_atol=dict(psnr=1e-7,ssim=1e-9))
 src=[receipt(p,role) for p,role in [(ROOT/'PROTOCOL_ZH.md','registered protocol'),(cfgpath,'original G6 config'),(g6,'weights and original Adam'),(refpath,'G6 full872 reference'),(cfg['paths']['schedule'],'original3000 schedule'),(sp,'fixed2000 additional presentations'),(cfg['paths']['manifest'],'unchanged split'),(cfg['paths']['base_config'],'original loader/loss configuration'),(raw['official']['dm_config'],'official architecture config'),(vgg,'fixed offline VGG')]]
 for name in raw['official']['expected_checkpoints']:src.append(receipt(Path(raw['official']['checkpoint_dir'])/name,'official construction/prior'))
 codes=set((ROOT/'scripts').glob('*.py'))
 for folder in [Path(cfg['paths']['source_root'])/'rgb_final_confirmation',PROJECT/'deps/DfP']:
  codes.update(folder.rglob('*.py'))
 plan=dict(experiment=39,arms=ARMS,steps=1000,exploratory_only=True,g6_checkpoint=str(g6),source_config=str(cfgpath),
  source_root=cfg['paths']['source_root'],reference_validation872=str(refpath),schedule_path=str(sp),validation_ids=ids,
  probe_ids=probe,full_index_map=index,scientific=scientific,scientific_id=canonical(scientific),sources=src,
  code_sources=[receipt(p) for p in sorted(codes)],deadline_unix=None)
 plan['protocol_id']=canonical(plan);write(lp,plan);verify(plan)
 print(json.dumps(dict(prepared=True,protocol_id=plan['protocol_id'],scientific_id=plan['scientific_id'])),flush=True)
class WeightedPhase(nn.Module):
 def __init__(self,phase):super().__init__();self.phase=phase
 def forward(self,x,y):return .1*self.phase(x,y)
def components(plan):
 cfg=yaml.safe_load(Path(plan['source_config']).read_text());sys.path.insert(0,plan['source_root'])
 from rgb_final_confirmation.config import ExperimentConfig
 from rgb_final_confirmation.polarfree import OfficialPolarFreeTask
 from rgb_final_confirmation.losses import build_official_task_loss
 raw=yaml.safe_load(Path(cfg['paths']['base_config']).read_text());raw['experiment_root']=str(ROOT);raw['paths']['outputs']=str(ROOT)
 raw['paths']['weight_audit']=str(ROOT/'config/weight_audit.json');raw['paths']['schedule']=plan['schedule_path']
 conf=ExperimentConfig(raw=raw,source=Path(cfg['paths']['base_config']))
 random.seed(2027);np.random.seed(2027);torch.manual_seed(2027);torch.cuda.manual_seed_all(2027)
 baseline=OfficialPolarFreeTask(conf,DEVICE).eval().requires_grad_(False)
 source=torch.load(plan['g6_checkpoint'],map_location='cpu',weights_only=False);assert source['step']==3000
 baseline.net_g.load_state_dict(source['tasknet_net_g'],strict=True)
 tasks={a:copy.deepcopy(baseline).eval().requires_grad_(False) for a in ARMS}
 opts={};names={}
 for a,t in tasks.items():
  t.net_g.requires_grad_(True);names[a]=[n for n,p in t.net_g.named_parameters()]
  opts[a]=torch.optim.Adam([dict(params=list(t.net_g.parameters()),lr=1e-5,name='tasknet_net_g')],betas=(.9,.999),eps=1e-8,weight_decay=0)
 opts['KEEP'].load_state_dict(copy.deepcopy(source['optimizer']))
 assert len(opts['KEEP'].state)==436 and not opts['RESET'].state
 assert names['KEEP']==names['RESET'] and names['KEEP']==list(source['tasknet_net_g'])
 for p,s in opts['KEEP'].state.items():
  assert int(s['step'])==3000 and s['exp_avg'].shape==p.shape and s['exp_avg_sq'].shape==p.shape
 criterion=build_official_task_loss(conf).to(DEVICE).eval().requires_grad_(False);criterion.phase=WeightedPhase(criterion.phase)
 del source
 return dict(cfg=cfg,raw=conf,baseline=baseline,tasks=tasks,opts=opts,names=names,criterion=criterion)
def frozen(c):
 out={}
 for a,t in dict(G6=c['baseline'],**c['tasks']).items():
  params=dict(t.named_parameters());h=hashlib.sha256()
  for n,v in t.state_dict().items():
   if n in params and params[n].requires_grad:continue
   h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
  out[a]=h.hexdigest()
 return out
def assert_frozen(c):
 assert all(not p.requires_grad and p.grad is None for p in c['baseline'].parameters())
 for t in c['tasks'].values():
  for n,p in t.named_parameters():
   assert p.requires_grad==n.startswith('model.net_g.')
   if not p.requires_grad:assert p.grad is None
 assert all(not p.requires_grad and p.grad is None for p in c['criterion'].parameters())
def snapshot(c,step,plan):
 return dict(step=step,global_step=3000+step,protocol_id=plan['protocol_id'],schedule_sha256=sha(plan['schedule_path']),
  weights={a:cpu(t.net_g.state_dict()) for a,t in c['tasks'].items()},optimizers={a:cpu(o.state_dict()) for a,o in c['opts'].items()},
  parameter_names=copy.deepcopy(c['names']),rng=rng())
def restore(c,v,plan):
 assert v['protocol_id']==plan['protocol_id'] and v['schedule_sha256']==sha(plan['schedule_path']) and v['parameter_names']==c['names']
 for a in ARMS:
  c['tasks'][a].net_g.load_state_dict(v['weights'][a],strict=True);c['opts'][a].load_state_dict(v['optimizers'][a]);c['opts'][a].zero_grad(set_to_none=True)
 set_rng(v['rng'])
def rows(c):
 from rgb_final_confirmation.data import load_manifest
 return load_manifest(c['cfg']['paths']['manifest'])
def batches(c,plan,start,end):
 from rgb_final_confirmation.data import PairedBranchDataset,select_rows
 ds=PairedBranchDataset(select_rows(rows(c),'train'),group='A1',crop_size=128,schedule=read(plan['schedule_path'])['items'][start*2:end*2])
 for batch in DataLoader(ds,batch_size=2,shuffle=False,num_workers=2,pin_memory=True,drop_last=True):
  yield {k:v.to(DEVICE) if torch.is_tensor(v) else v for k,v in batch.items()}
def full_item(c,sid):
 from rgb_final_confirmation.data import PairedBranchDataset
 row=next(r for r in rows(c) if r.sample_id==sid);item=PairedBranchDataset([row],group='A1',crop_size=None)[0]
 return {k:v[None].to(DEVICE) if torch.is_tensor(v) else [v] for k,v in item.items()}
class NoiseCheck:
 def __init__(self,tasks,noise):self.tasks=tasks;self.noise=noise;self.original={};self.calls=[]
 def __enter__(self):
  for a,t in self.tasks.items():
   self.original[a]=t.model.p_sample_loop
   def call(condition,noise,a=a):
    assert torch.equal(noise,self.noise) and list(noise.shape)==list(self.noise.shape)
    self.calls.append(dict(arm=a,sha256=tensor_sha(noise),shape=list(noise.shape)))
    return self.original[a](condition,noise)
   t.model.p_sample_loop=call
  return self
 def __exit__(self,*args):
  for a,t in self.tasks.items():t.model.p_sample_loop=self.original[a]
def gradient_sha(t):return canonical({n:tensor_sha(p.grad) for n,p in t.net_g.named_parameters() if p.grad is not None})
def update(c,plan,batch,step,check_initial=False):
 for o in c['opts'].values():o.zero_grad(set_to_none=True)
 seed_step(2027,'train',3000+step,batch['sample_id']);noise=torch.randn(2,16,256,device=DEVICE)
 rec=dict(step=step,global_step=3000+step,sample_ids=';'.join(batch['sample_id']));grads={}
 with NoiseCheck(c['tasks'],noise) as proof:
  for a,t in c['tasks'].items():
   pred,features=t(batch['polar'],batch['official_rgb'],noise);loss=c['criterion'](pred,batch['target'])
   assert torch.isfinite(loss);loss.backward();rec[a+'_loss']=float(loss.detach())
   if check_initial:grads[a]=gradient_sha(t)
   norm=float(torch.nn.utils.clip_grad_norm_(t.net_g.parameters(),1.,error_if_nonfinite=True));assert norm>0
   rec[a+'_grad_norm']=norm;del pred,features,loss
  assert len(proof.calls)==2 and proof.calls[0]['sha256']==proof.calls[1]['sha256']
  if check_initial:assert grads['KEEP']==grads['RESET'] and rec['KEEP_loss']==rec['RESET_loss']
  for o in c['opts'].values():o.step()
 return rec,dict(step=step,noise_calls=proof.calls,initial_gradients_equal=check_initial,gradient_hashes=grads)
@torch.no_grad()
def evaluate_rows(c,plan,step,ids,path,identity=False):
 from rgb_final_confirmation.evaluation import psnr,ssim
 refs={r['sample_id']:r for r in read(plan['reference_validation872'])['per_image']};saved=rng()
 results=read(path)['rows'] if path.exists() else [];assert [r['sample_id'] for r in results]==ids[:len(results)]
 try:
  for sid in ids[len(results):]:
   batch=full_item(c,sid);seed_step(2026,'validation',plan['full_index_map'][sid],[sid]);noise=torch.randn(1,16,256,device=DEVICE)
   metrics={};hashes={};tasks=dict(G6=c['baseline'],**c['tasks']);reference=None
   with NoiseCheck(tasks,noise) as proof:
    for a,t in tasks.items():
     pred,f=t(batch['polar'],batch['official_rgb'],noise)
     metrics[a]=dict(psnr=psnr(pred,batch['target']),ssim=ssim(pred,batch['target']));hashes[a]=tensor_sha(pred)
     if a=='G6':reference=pred.detach().clone()
     elif identity:assert torch.equal(pred,reference),(sid,a,'zero identity')
     del pred,f
   assert len(proof.calls)==3
   assert abs(metrics['G6']['psnr']-refs[sid]['psnr'])<=1e-7 and abs(metrics['G6']['ssim']-refs[sid]['ssim'])<=1e-9
   results.append(dict(sample_id=sid,scene=batch['scene'][0],shape=list(batch['target'].shape),original_size=True,
    full_index=plan['full_index_map'][sid],metrics=metrics,output_sha256=hashes,noise_calls=proof.calls))
   write(path,dict(protocol_id=plan['protocol_id'],step=step,frames=len(results),rows=results))
   status('preflight_reference' if identity else 'evaluating',step=step,completed=len(results),total=len(ids))
   del batch,reference;gc.collect();torch.cuda.empty_cache()
 finally:set_rng(saved)
 return results
def preflight(plan):
 fp=ROOT/'config/preflight.json'
 if fp.exists():assert read(fp)['passed'];return
 status('preflight_loading');c=components(plan);torch.cuda.reset_peak_memory_stats();before=frozen(c)
 sets=[{p.data_ptr() for p in t.parameters()} for t in [c['baseline'],*c['tasks'].values()]]
 assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
 initial=snapshot(c,0,plan);assert equal(initial['weights']['KEEP'],initial['weights']['RESET'])
 batch=next(batches(c,plan,0,1));tick=time.time();_,proof=update(c,plan,batch,1,check_initial=True);torch.cuda.synchronize();seconds=time.time()-tick
 restore(c,initial,plan)
 c['opts']['RESET'].load_state_dict(copy.deepcopy(c['opts']['KEEP'].state_dict()))
 update(c,plan,batch,1);same=(equal(c['tasks']['KEEP'].net_g.state_dict(),c['tasks']['RESET'].net_g.state_dict()) and equal(c['opts']['KEEP'].state_dict(),c['opts']['RESET'].state_dict()));assert same
 restore(c,initial,plan)
 for i,batch in enumerate(batches(c,plan,0,2),1):update(c,plan,batch,i)
 snap=snapshot(c,2,plan);save(ROOT/'diagnostics/preflight_replay.pt',snap)
 batch=next(batches(c,plan,2,3));update(c,plan,batch,3);expected=snapshot(c,3,plan)
 restore(c,torch.load(ROOT/'diagnostics/preflight_replay.pt',map_location='cpu',weights_only=False),plan)
 update(c,plan,batch,3);assert equal(expected,snapshot(c,3,plan))
 restore(c,initial,plan);assert_frozen(c);assert before==frozen(c)
 refs={r['sample_id']:r for r in read(plan['reference_validation872'])['per_image']}
 # All dataset frames have registered source shape; take largest by real manifest image dimensions.
 from PIL import Image
 selected=[r for r in rows(c) if r.sample_id in plan['validation_ids']]
 largest=max(selected,key=lambda r:Image.open(r.official_rgb).size[0]*Image.open(r.official_rgb).size[1]).sample_id
 ids=list(dict.fromkeys(plan['probe_ids']+[largest]));tick=time.time()
 evaluate_rows(c,plan,0,ids,ROOT/'diagnostics/initial_reference.json',identity=True)
 frame_seconds=(time.time()-tick)/len(ids)
 assert equal(initial['weights'],snapshot(c,0,plan)['weights'])
 write(fp,dict(passed=True,protocol_id=plan['protocol_id'],initial_weights_exact=True,source_adam_3000=True,reset_adam_empty=True,
  no_parameter_alias=True,initial_gradients_equal=True,same_optimizer_update_exact=same,resume_exact=True,
  source_frozen=True,all_preflight_updates_discarded=True,zero_identity_reference_frames=len(ids),
  noise_proof=proof,train_step_seconds=seconds,frame_seconds=frame_seconds,
  estimated_remaining_hours=1.2*(1000*seconds+96*frame_seconds)/3600,peak_allocated_bytes=int(torch.cuda.max_memory_allocated())))
 del c,initial,snap,expected,batch;gc.collect();torch.cuda.empty_cache();status('preflight_passed')
def train(plan):
 if (ROOT/'diagnostics/training_complete.json').exists():return
 c=components(plan);before=frozen(c);last=ROOT/'weights/last.pt';log=ROOT/'logs/training.csv';start=0
 if last.exists():
  v=torch.load(last,map_location='cpu',weights_only=False);restore(c,v,plan);start=v['step'];del v
 else:
  assert not log.exists();v=snapshot(c,0,plan);save(ROOT/'weights/initial.pt',v);save(last,v);del v
 if log.exists():
  old=list(csv.DictReader(log.open()));keep=[r for r in old if int(r['step'])<=start]
  if len(old)>len(keep):write(ROOT/'failures'/('orphaned_rows_'+str(time.time_ns())+'.json'),old[len(keep):])
  assert len(keep)==start
  with log.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(old[0]));w.writeheader();w.writerows(keep)
 status('training',step=start,total=1000)
 step=start
 with log.open('a',newline='') as f:
  writer=None
  for step,batch in enumerate(batches(c,plan,start,1000),start+1):
   tick=time.time();rec,proof=update(c,plan,batch,step,check_initial=step==1);torch.cuda.synchronize();rec['seconds']=time.time()-tick
   if writer is None:
    writer=csv.DictWriter(f,fieldnames=list(rec))
    if start==0:writer.writeheader()
   writer.writerow(rec);f.flush()
   if step==1 or step%100==0:
    v=snapshot(c,step,plan);save(last,v)
    if step in [100,300,1000]:save(ROOT/'weights'/f'step_{step:04d}.pt',v)
    del v;write(ROOT/'diagnostics'/f'noise_{step:04d}.json',proof)
   if step==1 or step%25==0:status('training',step=step,total=1000,seconds=rec['seconds'],KEEP_loss=rec['KEEP_loss'],RESET_loss=rec['RESET_loss'])
   if step%100==0:assert_frozen(c);assert before==frozen(c)
 assert step==1000;after=frozen(c);assert before==after
 write(ROOT/'diagnostics/frozen_final.json',dict(passed=True,before=before,after=after))
 # The fixed1000 checkpoint is the final state; save final separately without changing it.
 v=snapshot(c,1000,plan);save(ROOT/'weights/final.pt',v);save(last,v)
 write(ROOT/'diagnostics/training_complete.json',dict(steps=1000,protocol_id=plan['protocol_id'],final_sha256=sha(ROOT/'weights/final.pt')))
 del c,v,batch;gc.collect();torch.cuda.empty_cache()
def compare(results,a,b):
 scenes=sorted({r['scene'] for r in results});out={}
 for metric in ['psnr','ssim']:
  per={s:float(np.mean([r['metrics'][a][metric]-r['metrics'][b][metric] for r in results if r['scene']==s])) for s in scenes}
  out[metric]=dict(scene_equal=float(np.mean(list(per.values()))),scenes=per,positive_scenes=sum(v>0 for v in per.values()))
 return out
def evaluate(plan):
 if (ROOT/'reports/summary.json').exists():return
 c=components(plan);summaries={}
 for step,ids in [(100,plan['probe_ids']),(300,plan['probe_ids']),(1000,plan['validation_ids'])]:
  v=torch.load(ROOT/'weights'/f'step_{step:04d}.pt',map_location='cpu',weights_only=False);restore(c,v,plan);del v
  data=evaluate_rows(c,plan,step,ids,ROOT/'metrics'/f'step_{step:04d}.json')
  summaries[str(step)]={a+'-'+b:compare(data,a,b) for a,b in [('KEEP','G6'),('RESET','G6'),('KEEP','RESET')]}
 end=summaries['1000'];delta=end['KEEP-RESET'];keep=end['KEEP-G6'];reset=end['RESET-G6']
 meaningful=delta['psnr']['scene_equal']>=.1 and delta['ssim']['scene_equal']>=0 and delta['psnr']['positive_scenes']>=6
 preserves=keep['psnr']['scene_equal']>=-.05 and keep['ssim']['scene_equal']>=-.0002
 if meaningful and preserves and reset['psnr']['scene_equal']<=-.1:interpretation='supports_Adam_state_retention_in_this_short_run'
 elif keep['psnr']['scene_equal']<=-.1 and reset['psnr']['scene_equal']<=-.1:interpretation='both_decline_Adam_retention_not_sufficient'
 elif delta['psnr']['scene_equal']<=-.1 and delta['ssim']['scene_equal']<=0 and sum(v<0 for v in delta['psnr']['scenes'].values())>=6:interpretation='reset_better_does_not_support_reset_harm'
 else:interpretation='inconclusive_or_mixed_short_run'
 write(ROOT/'reports/summary.json',dict(protocol_id=plan['protocol_id'],steps=1000,validation=64,test=0,exploratory_only=True,
  evaluations=summaries,interpretation=interpretation,automatic_extension=False,limitations=['single_seed','fixed64','short1000_not3000','no_E_H_feedback','not_unique_cause_of_experiment38']))
 del c;gc.collect();torch.cuda.empty_cache()
def finish(plan):
 status('independent_audit');env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
 subprocess.run([sys.executable,str(ROOT/'scripts/audit_study.py')],check=True,env=env)
 verify(plan)
 excluded={'artifact_manifest.json','closeout.json','status.json','controller_heartbeat.json','worker_exit.json','worker.log','controller.log','archive_audit.json','final_review_closeout.json'}
 files=[]
 for p in sorted(ROOT.rglob('*')):
  if p.is_symlink() or not p.is_file() or p.name in excluded or p.suffix in ['.pyc','.tmp'] or 'cache' in p.relative_to(ROOT).parts or '__pycache__' in p.parts:continue
  files.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
 write(ROOT/'artifact_manifest.json',dict(protocol_id=plan['protocol_id'],files=files))
 subprocess.run([sys.executable,str(ROOT/'scripts/audit_study.py'),'--archive'],check=True,env=env)
 close=dict(success=True,exploratory_only=True,steps=1000,validation=64,test=0,protocol_id=plan['protocol_id'],
  archive_manifest_sha256=sha(ROOT/'artifact_manifest.json'),independent_audit_verified=True,archive_verified=True,
  interpretation=read(ROOT/'reports/summary.json')['interpretation'],automatic_extension=False,time_unix=time.time())
 write(ROOT/'reports/final_review_closeout.json',close);write(ROOT/'closeout.json',close);status('completed',**close)
def run():
 lock=(ROOT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 plan=read(ROOT/'config/locked.json');verify(plan)
 if (ROOT/'closeout.json').exists():assert read(ROOT/'closeout.json')['success'];return
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
 preflight(plan);train(plan);evaluate(plan);finish(plan)
if __name__=='__main__':
 try:
  if '--prepare' in sys.argv:prepare()
  else:run()
 except Exception:
  failure=dict(time_unix=time.time(),traceback=traceback.format_exc());write(ROOT/'failures'/f'worker_{time.time_ns()}.json',failure);status('failed',**failure);raise

