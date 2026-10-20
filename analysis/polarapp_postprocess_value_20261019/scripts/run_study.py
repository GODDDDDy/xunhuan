"""42: fixed native PolarAPP reconstruction, paired T versus post-E/T.

Reuses project model/data/loss/evaluation primitives. This glue is required:
the older studies never ran native PIDNet before the same-resolution enhancer.
"""
from __future__ import annotations
import copy,csv,fcntl,gc,importlib.util,json,os,random,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import yaml

ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent
spec=importlib.util.spec_from_file_location('helpers39',PROJECT/'g6_adam_state_diagnostic_20261019/scripts/run_study.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
read,write,sha,canonical,receipt=h.read,h.write,h.sha,h.canonical,h.receipt
cpu,equal,rng,set_rng,save,seed_step=h.cpu,h.equal,h.rng,h.set_rng,h.save,h.seed_step
DEVICE=torch.device('cuda:0');ARMS=['A','B'];STEPS=6000
DEM_SHA='00aa162b962d7ef952cd1c8a0d3bf31ac9bbf9e65d3f76b055bb5c08a64f2bd6'
TASK_SHA='e0ec39980e77d85de471e7108df61bacd7a69733750d2bcd37d0fa9382f09096'
E_SHA='a4ecebacba5cfc25546320974a9c37f1cc220c685f1318b3cc290ae4a68e9554'

def status(stage,**kw):
 value=dict(stage=stage,time_unix=time.time());value.update(kw)
 write(ROOT/'status.json',value);print(json.dumps(value),flush=True)

def guard():
 p=ROOT/'worker_registration.json'
 if p.exists():
  deadline=read(p).get('deadline_unix')
  if deadline is not None:assert time.time()<deadline,'Fixed worker deadline reached; preserve state'

def verify(plan):
 unsigned=dict(plan);unsigned.pop('protocol_id');assert canonical(unsigned)==plan['protocol_id']
 assert plan['experiment']==42 and plan['steps']==6000 and plan['arms']==ARMS
 assert canonical(plan['scientific'])==plan['scientific_id']
 replacements={};codes={r['path']:r for r in plan['code_sources']}
 rp=ROOT/'config/implementation_revisions.json'
 for rev in read(rp) if rp.exists() else []:
  assert rev['protocol_id']==plan['protocol_id'] and rev['scientific_id']==plan['scientific_id']
  for r in rev['replacements']:
   assert r['path'] in codes and r['original_sha256']==codes[r['path']]['sha256']
   assert sha(r['preserved_original_path'])==r['original_sha256']
   replacements[r['path']]=r['repaired_sha256']
 for r in plan['sources']+plan['code_sources']:
  assert sha(r['path'])==replacements.get(r['path'],r['sha256']),r['path']

def prepare():
 for folder in ['config','weights','logs','metrics','reports','diagnostics','failures','cache']:(ROOT/folder).mkdir(parents=True,exist_ok=True)
 lp=ROOT/'config/locked.json'
 if lp.exists():verify(read(lp));return
 assert not (ROOT/'worker_registration.json').exists()
 old=read(PROJECT/'g6_adam_state_diagnostic_20261019/config/locked.json')
 cfg=yaml.safe_load(Path(old['source_config']).read_text());sys.path.insert(0,old['source_root'])
 from rgb_final_confirmation.data import load_manifest,select_rows,build_training_schedule
 rows=load_manifest(cfg['paths']['manifest']);train=select_rows(rows,'train')
 assert len(train)==5440 and len(select_rows(rows,'validation'))==872 and len(select_rows(rows,'test'))==188
 previous=read(ROOT/'history/user_requested_budget_v1/config/training3000.json')
 extension=build_training_schedule(train,seed=2028,total_steps=3000,batch_size=2,crop_size=128)
 schedule=dict(schema_version=2,seed=2027,extension_seed=2028,total_steps=6000,batch_size=2,crop_size=128,items=previous['items']+extension['items'])
 schedule['sha256']=canonical(schedule);assert len(schedule['items'])==12000 and schedule['items'][:6000]==previous['items']
 sp=ROOT/'config/training6000.json';write(sp,schedule)
 ids=old['validation_ids'];probe=old['probe_ids'];assert len(ids)==64 and len(probe)==16
 weights=PROJECT.parent.parent/'PolarAPP权重'
 # The released bundle is outside 929work, rather than under the project.
 weights=Path('/media/li/data3/25thstudent/DY/PolarAPP权重')
 dem=weights/'DfP-DemNet.pth';task=weights/'DfP-TaskNet.pth'
 e=PROJECT/'polar_residual_feedback_joint_20261010/pretraining/checkpoints/selected.pt'
 assert sha(dem)==DEM_SHA and sha(task)==TASK_SHA and sha(e)==E_SHA
 raw=yaml.safe_load(Path(cfg['paths']['base_config']).read_text())
 from PIL import Image
 shapes={r.sample_id:list(Image.open(r.polar_000).size) for r in rows if r.split!='test'}
 assert all(w%8==0 and v%8==0 and min(w,v)>=128 for w,v in shapes.values()),'Native dimensions must be valid without resizing'
 largest_train=max(train,key=lambda r:np.prod(shapes[r.sample_id])).sample_id
 largest_val=max([r for r in rows if r.sample_id in ids],key=lambda r:np.prod(shapes[r.sample_id])).sample_id
 scientific=dict(experiment=42,arms=ARMS,steps=6000,batch=2,supervised_ROI=128,seed=2027,
  D_sha=DEM_SHA,T_sha=TASK_SHA,E_sha=E_SHA,schedule_sha=sha(sp),
  input='official simulated A(full four-angle PNG), frozen native PIDNet, float32 full->crop P_D',
  RGB='unchanged P_D mean shared S/A/B; only polarization cues modified',
  E='existing pretrain_forward only; zero mid_output; clamp[0,1]; unused adapters frozen; no feature bypass',
  loss='native DfPTaskLoss: L1+(1-SSIM)+phaseMSE+.1VGGL1, inherited range_norm',
  optimizer=dict(T_lr=1e-5,E_lr=2e-5,Adam_betas=[.9,.999],eps=1e-8,wd=0,clip_each=1.,amp=False,scheduler=None),
  noise='seed2027 train / seed2026 validation, phase+step+IDs, actual paired [B,16,256]',
  evaluation=dict(checkpoints=[0,1000,2000,3000,4500,6000],fixed_val_probe=probe,fixed_train_indices=list(range(16)),endpoint6000=ids,test=0),
  gate=dict(comparators=['A','S'],PSNR=.05,SSIM=0,positive_scenes=6),
  stop='fixed6000; if no double gate stop; no C/seeds/steps/LR/loss search',worker_seconds=86400,user_amended_budget=True,
  limitations=['single seed','historically explored validation','published source may have seen validation scenes','simulated input not real raw','crop-trained E/T, full-frame evaluation'])
 sources=[receipt(p,role) for p,role in [(ROOT/'PROTOCOL_ZH.md','registered protocol'),(dem,'native D'),(task,'native net_g'),(e,'independent E pretraining'),(sp,'paired schedule'),(cfg['paths']['manifest'],'fixed split'),(cfg['paths']['base_config'],'official construction'),(raw['official']['dm_config'],'diffusion architecture'),(old['source_config'],'data configuration')]]
 for name in raw['official']['expected_checkpoints']:sources.append(receipt(Path(raw['official']['checkpoint_dir'])/name,'official prior/construction'))
 vgg=Path('/home/li/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth');assert sha(vgg)==h.VGG_SHA;sources.append(receipt(vgg,'offline VGG'))
 link=ROOT/'experiments/pretrained_models'/vgg.name;link.parent.mkdir(parents=True,exist_ok=True)
 if not link.exists():link.symlink_to(vgg)
 code=set((ROOT/'scripts').glob('*.py'));code.add(PROJECT/'g6_adam_state_diagnostic_20261019/scripts/run_study.py')
 for folder in [PROJECT/'deps/DfP',Path(old['source_root'])/'rgb_final_confirmation',PROJECT/'polar_residual_feedback_joint_20261010/scripts']:code.update(folder.rglob('*.py'))
 plan=dict(experiment=42,arms=ARMS,steps=6000,source_root=old['source_root'],source_config=old['source_config'],
  manifest=cfg['paths']['manifest'],schedule_path=str(sp),D_checkpoint=str(dem),T_checkpoint=str(task),E_checkpoint=str(e),
  validation_ids=ids,probe_ids=probe,full_index_map=old['full_index_map'],largest_train=largest_train,largest_val=largest_val,
  scientific=scientific,scientific_id=canonical(scientific),sources=sources,code_sources=[receipt(p) for p in sorted(code)])
 plan['protocol_id']=canonical(plan);write(lp,plan);verify(plan)
 status('prepared_waiting_launch',protocol_id=plan['protocol_id'],scientific_id=plan['scientific_id'])

def components(plan):
 sys.path.insert(0,plan['source_root'])
 from rgb_final_confirmation.config import ExperimentConfig
 from rgb_final_confirmation.polarfree import OfficialPolarFreeTask
 cfg=yaml.safe_load(Path(plan['source_config']).read_text());raw=yaml.safe_load(Path(cfg['paths']['base_config']).read_text())
 raw['experiment_root']=str(ROOT);raw['paths']['outputs']=str(ROOT);raw['paths']['weight_audit']=str(ROOT/'diagnostics/official_weight_audit.json')
 conf=ExperimentConfig(raw=raw,source=Path(cfg['paths']['base_config']))
 random.seed(2027);np.random.seed(2027);torch.manual_seed(2027);torch.cuda.manual_seed_all(2027)
 baseline=OfficialPolarFreeTask(conf,DEVICE).eval().requires_grad_(False)
 ck=torch.load(plan['T_checkpoint'],map_location='cpu',weights_only=False);baseline.net_g.load_state_dict(ck['model_state_dict'],strict=True);del ck
 from utils.utils_net import PIDNet
 from utils.init_interp import init_interp
 from polarapp.losses import DfPTaskLoss
 dem=PIDNet().to(DEVICE).eval().requires_grad_(False)
 ck=torch.load(plan['D_checkpoint'],map_location='cpu',weights_only=False);dem.load_state_dict(ck['model_state_dict'],strict=True);del ck
 tasks={a:copy.deepcopy(baseline).eval().requires_grad_(False) for a in ARMS}
 for task in tasks.values():task.net_g.requires_grad_(True)
 sys.path.insert(0,str(PROJECT/'polar_residual_feedback_joint_20261010/scripts'))
 from enhancer import PolarEnhancer
 enhancer=PolarEnhancer();ck=torch.load(plan['E_checkpoint'],map_location='cpu',weights_only=False)
 enhancer.load_state_dict(ck['enhancer'],strict=True);del ck
 enhancer.downstream_parameters();enhancer.activation_checkpoint=True
 for name,p in enhancer.named_parameters():
  if name.startswith('adapters.'):p.requires_grad_(False)
 nn.init.zeros_(enhancer.mid_output.weight)
 if enhancer.mid_output.bias is not None:nn.init.zeros_(enhancer.mid_output.bias)
 enhancer=enhancer.to(DEVICE).train()
 opts={a:torch.optim.Adam([dict(params=list(t.net_g.parameters()),lr=1e-5,name='T')]+([dict(params=[p for p in enhancer.parameters() if p.requires_grad],lr=2e-5,name='E')] if a=='B' else []),betas=(.9,.999),eps=1e-8,weight_decay=0) for a,t in tasks.items()}
 criterion=DfPTaskLoss(.1).to(DEVICE).eval().requires_grad_(False)
 return dict(cfg=cfg,baseline=baseline,tasks=tasks,dem=dem,imaging=init_interp(phase='train').to(DEVICE),E=enhancer,opts=opts,criterion=criterion)

def model_hash(model,frozen_only=False):
 named=dict(model.named_parameters());parts={k:v for k,v in model.state_dict().items() if not(frozen_only and k in named and named[k].requires_grad)}
 return canonical({k:h.tensor_sha(v) for k,v in parts.items()})

def frozen(c):return dict(D=model_hash(c['dem']),S=model_hash(c['baseline']),A=model_hash(c['tasks']['A'],True),B=model_hash(c['tasks']['B'],True),E_inactive=model_hash(c['E'],True),loss=model_hash(c['criterion']))

def assert_frozen(c):
 assert all(not p.requires_grad and p.grad is None for m in [c['dem'],c['baseline'],c['criterion']] for p in m.parameters())
 for t in c['tasks'].values():
  for n,p in t.named_parameters():
   assert p.requires_grad==n.startswith('model.net_g.')
   if not p.requires_grad:assert p.grad is None
 for p in c['E'].parameters():
  if not p.requires_grad:assert p.grad is None

def snapshot(c,step,plan):
 return dict(step=step,protocol_id=plan['protocol_id'],schedule_sha256=sha(plan['schedule_path']),weights={a:cpu(t.net_g.state_dict()) for a,t in c['tasks'].items()},E=cpu(c['E'].state_dict()),optimizers={a:cpu(o.state_dict()) for a,o in c['opts'].items()},rng=rng())

def restore(c,v,plan):
 assert v['protocol_id']==plan['protocol_id'] and v['schedule_sha256']==sha(plan['schedule_path'])
 for a in ARMS:c['tasks'][a].net_g.load_state_dict(v['weights'][a],strict=True);c['opts'][a].load_state_dict(v['optimizers'][a]);c['opts'][a].zero_grad(set_to_none=True)
 c['E'].load_state_dict(v['E'],strict=True);set_rng(v['rng'])

@torch.no_grad()
def full_reconstruction(c,sid):
 batch=h.full_item(c,sid);p=c['imaging'](batch['polar']);out,_=c['dem'](p,ELT_state=False)
 assert out.shape==batch['polar'].shape and out.dtype==torch.float32 and torch.isfinite(out).all(),(sid,out.shape,batch['polar'].shape)
 return out,batch

def task_inputs(polar,anchor):
 from polarapp.operations import inter_data_process
 inputs=inter_data_process(polar);inputs[0]=anchor;return inputs

def predict(c,a,pd,noise,bypass=False):
 from polarapp.operations import inter_data_process
 anchor=inter_data_process(pd)[0]
 polar=c['E'].pretrain_forward(pd).clamp(0,1) if a=='B' and not bypass else pd
 return c['tasks'][a].forward_inputs(task_inputs(polar,anchor),noise)[0]

class CachedROI(Dataset):
 def __init__(self,entries):self.entries=entries
 def __len__(self):return len(self.entries)
 def __getitem__(self,i):
  rec=self.entries[i];path=ROOT/rec['path'];assert sha(path)==rec['sha256']
  return torch.load(path,map_location='cpu',weights_only=False)

def batches(plan,start,end):
 entries=read(ROOT/'config/cache_index.json')['entries'][start*2:end*2]
 for batch in DataLoader(CachedROI(entries),batch_size=2,shuffle=False,num_workers=2,pin_memory=True,drop_last=True):
  yield {k:v.to(DEVICE) if torch.is_tensor(v) else v for k,v in batch.items()}

def update(c,plan,batch,step,bypass=False,check=False):
 for opt in c['opts'].values():opt.zero_grad(set_to_none=True)
 seed_step(2027,'train42',step,batch['sample_id']);noise=torch.randn(2,16,256,device=DEVICE)
 rec=dict(step=step,sample_ids=';'.join(batch['sample_id']));grads={}
 with h.NoiseCheck(c['tasks'],noise) as proof:
  for a in ARMS:
   pred=predict(c,a,batch['pd'],noise,bypass=bypass);loss=c['criterion'](pred,batch['target']);assert torch.isfinite(loss)
   rec[a+'_loss']=float(loss.detach());loss.backward()
   if check:grads[a]={n:h.tensor_sha(p.grad) for n,p in c['tasks'][a].net_g.named_parameters() if p.grad is not None}
   rec[a+'_T_grad']=float(torch.nn.utils.clip_grad_norm_(c['tasks'][a].net_g.parameters(),1.,error_if_nonfinite=True));assert rec[a+'_T_grad']>0
   if a=='B':rec['E_grad']=float(torch.nn.utils.clip_grad_norm_([p for p in c['E'].parameters() if p.requires_grad],1.,error_if_nonfinite=True))
   del pred,loss
  assert len(proof.calls)==2 and proof.calls[0]['sha256']==proof.calls[1]['sha256']
  for opt in c['opts'].values():opt.step()
 return rec,dict(noise_calls=proof.calls,T_gradients_equal=equal(grads.get('A'),grads.get('B')) if check else None)

@torch.no_grad()
def evaluate_rows(c,plan,step,ids,path,identity=False):
 from rgb_final_confirmation.evaluation import psnr,ssim
 from polarapp.operations import inter_data_process
 saved=rng();e_mode=c['E'].training;c['E'].eval();rows=read(path)['rows'] if path.exists() else [];assert [r['sample_id'] for r in rows]==ids[:len(rows)]
 try:
  for sid in ids[len(rows):]:
   guard();pd,batch=full_reconstruction(c,sid);seed_step(2026,'validation42',plan['full_index_map'][sid],[sid]);noise=torch.randn(1,16,256,device=DEVICE)
   metrics={};hashes={}
   with h.NoiseCheck(dict(S=c['baseline'],**c['tasks']),noise) as proof:
    original,_=c['baseline'].forward_inputs(inter_data_process(pd),noise);original=original.clamp(0,1)
    metrics['S']=dict(psnr=psnr(original,batch['target']),ssim=ssim(original,batch['target']));hashes['S']=h.tensor_sha(original)
    for a in ARMS:
     pred=predict(c,a,pd,noise).clamp(0,1)
     if identity:assert torch.equal(original,pred),(sid,a,'identity mismatch')
     metrics[a]=dict(psnr=psnr(pred,batch['target']),ssim=ssim(pred,batch['target']));hashes[a]=h.tensor_sha(pred);del pred
   rows.append(dict(sample_id=sid,scene=batch['scene'][0],shape=list(batch['target'].shape),original_size=True,metrics=metrics,output_sha256=hashes,PD_sha256=h.tensor_sha(pd),noise_calls=proof.calls))
   write(path,dict(protocol_id=plan['protocol_id'],step=step,frames=len(rows),rows=rows))
   status('preflight_identity' if identity else 'evaluating',step=step,completed=len(rows),total=len(ids));del pd,batch,original;gc.collect();torch.cuda.empty_cache()
 finally:set_rng(saved);c['E'].train(e_mode)
 return rows

def preflight_before_cache(c,plan):
 fp=ROOT/'diagnostics/native_preflight.json'
 if fp.exists():assert read(fp)['passed'];return
 from polarapp.operations import inter_data_process
 status('preflight_native_largest')
 pd,b=full_reconstruction(c,plan['largest_train']);assert pd.shape[-2:]==b['target'].shape[-2:]
 train_shape=list(pd.shape);del pd,b;gc.collect();torch.cuda.empty_cache()
 ids=list(dict.fromkeys([plan['validation_ids'][0],plan['largest_val']]))
 t=time.time();evaluate_rows(c,plan,0,ids,ROOT/'diagnostics/initial_identity.json',True);torch.cuda.synchronize();seconds=(time.time()-t)/len(ids)
 pd,b=full_reconstruction(c,plan['largest_val']);inputs=inter_data_process(pd)
 seed_step(2026,'native42',0,[plan['largest_val']]);noise=torch.randn(1,16,256,device=DEVICE)
 with torch.no_grad():
  expected,_=c['baseline'].forward_inputs(inputs,noise)
  native=c['baseline'].model;original=native.p_sample_loop;seen=[]
  def explicit(condition,generated):
   assert list(generated.shape)==[1,16,256];seen.append(h.tensor_sha(noise));return original(condition,noise)
  native.p_sample_loop=explicit
  try:actual,*_=native(inputs,phase='val')
  finally:native.p_sample_loop=original
  assert torch.equal(actual,expected) and len(seen)==1
 write(fp,dict(passed=True,full_native_D=True,largest_train_shape=train_shape,largest_val_shape=list(pd.shape),zero_E_frames=len(ids),native_T_forward_exact=True,explicit_noise_sha256=seen[0],frame_seconds=seconds))
 del pd,b,inputs,actual,expected;gc.collect();torch.cuda.empty_cache()

@torch.no_grad()
def prepare_cache(c,plan,warmup_only=False):
 done=ROOT/'diagnostics/cache_complete.json'
 if done.exists():return
 from rgb_final_confirmation.data import load_manifest,select_rows
 train=select_rows(load_manifest(plan['manifest']),'train');items=read(plan['schedule_path'])['items']
 grouped={}
 for i,item in enumerate(items):grouped.setdefault(item['sample_index'],[]).append((i,item))
 ip=ROOT/'config/cache_index.json';index=read(ip) if ip.exists() else dict(entries=[None]*12000,full_sources={})
 if len(index['entries'])<12000:
  assert len(index['entries'])==6000;index['entries'] += [None]*6000
 tick=time.time();completed=0
 for sample_index,work in grouped.items():
  guard();row=train[sample_index];completed+=1
  if warmup_only and completed>16:
   write(ip,index);return
  if all(index['entries'][i] is not None and (ROOT/index['entries'][i]['path']).exists() and sha(ROOT/index['entries'][i]['path'])==index['entries'][i]['sha256'] for i,it in work):continue
  pd,b=full_reconstruction(c,row.sample_id)
  from rgb_final_confirmation.data import ANGLES
  inputs=[getattr(row,'polar_'+angle) for angle in ANGLES]+[row.gt_rgb]
  index['full_sources'][row.sample_id]=dict(PD_sha256=h.tensor_sha(pd),shape=list(pd.shape),input_files=[receipt(p) for p in inputs])
  for i,it in work:
   y,x=it['top'],it['left'];v=dict(pd=pd[0,:,y:y+128,x:x+128].cpu().contiguous().clone(),target=b['target'][0,:,y:y+128,x:x+128].cpu().contiguous().clone(),sample_id=row.sample_id,scene=row.scene)
   assert v['pd'].shape==torch.Size([12,128,128]) and v['target'].shape==torch.Size([3,128,128])
   path=ROOT/'cache'/f'{i:05d}.pt';save(path,v)
   index['entries'][i]=dict(index=i,sample_id=row.sample_id,top=y,left=x,path=path.relative_to(ROOT).as_posix(),sha256=sha(path),PD_roi_sha256=h.tensor_sha(v['pd']))
  del pd,b;gc.collect();torch.cuda.empty_cache()
  if completed%10==0 or completed==len(grouped):
   write(ip,index);status('preparing_full_D_cache',completed=completed,total=len(grouped),ROI=sum(x is not None for x in index['entries']),elapsed_seconds=time.time()-tick)
 write(ip,index);assert len(index['entries'])==12000 and all(index['entries'])
 write(done,dict(passed=True,ROI=12000,full_images=len(grouped),index_sha256=sha(ip),seconds=time.time()-tick))

def preflight_training(c,plan):
 fp=ROOT/'config/preflight.json'
 if fp.exists():assert read(fp)['passed'];return
 status('preflight_gradients_resume');before=frozen(c);initial=snapshot(c,0,plan)
 assert equal(initial['weights']['A'],initial['weights']['B'])
 models=[c['baseline'],c['tasks']['A'],c['tasks']['B'],c['dem'],c['E']]
 stores=[{p.data_ptr() for p in m.parameters()} for m in models]
 assert all(not stores[i]&stores[j] for i in range(len(stores)) for j in range(i))
 batch=next(batches(plan,0,1));rec,proof=update(c,plan,batch,1,check=True)
 assert rec['A_loss']==rec['B_loss'] and proof['T_gradients_equal'] and rec['E_grad']>0
 assert equal(c['tasks']['A'].net_g.state_dict(),c['tasks']['B'].net_g.state_dict())
 assert equal(c['opts']['A'].state_dict()['state'],{k:v for k,v in c['opts']['B'].state_dict()['state'].items() if k<len(list(c['tasks']['B'].net_g.parameters()))})
 restore(c,initial,plan);times=[]
 for step,batch in enumerate(batches(plan,0,3),1):
  t=time.time();update(c,plan,batch,step);torch.cuda.synchronize();times.append(time.time()-t)
 trained=snapshot(c,3,plan)
 assert any(not torch.equal(v,initial['E'][n]) for n,v in trained['E'].items() if n.startswith('img_feature_extractor.'))
 assert any(not torch.equal(v,initial['E'][n]) for n,v in trained['E'].items() if n.startswith('mid_output.'))
 assert not equal(initial['weights']['A'],trained['weights']['A']) and not equal(initial['weights']['B'],trained['weights']['B'])
 save(ROOT/'diagnostics/preflight_replay.pt',trained);batch=next(batches(plan,3,4))
 update(c,plan,batch,4);expected=snapshot(c,4,plan)
 restore(c,torch.load(ROOT/'diagnostics/preflight_replay.pt',map_location='cpu',weights_only=False),plan);update(c,plan,batch,4)
 assert equal(expected,snapshot(c,4,plan))
 restore(c,initial,plan);assert_frozen(c);assert before==frozen(c)
 probe_before=snapshot(c,0,plan);online_probes(c,plan,0)
 assert equal(probe_before,snapshot(c,0,plan)),'Online probes changed weights/Adam/RNG'
 write(ROOT/'diagnostics/probe_invariance.json',dict(passed=True,weights_Adam_RNG_exact=True,training_mode_restored=c['E'].training))
 del probe_before
 frame=read(ROOT/'diagnostics/native_preflight.json')['frame_seconds']
 write(fp,dict(passed=True,protocol_id=plan['protocol_id'],no_parameter_alias=True,initial_T_gradients_and_update_equal=True,initial_Adam_T_states_equal=True,E_actual_head_and_body_update=True,source_frozen=True,resume_exact=True,all_preflight_updates_discarded=True,noise_proof=proof,train_paired_step_seconds=float(np.mean(times[1:])),frame_seconds=frame,estimated_remaining_hours=1.3*(6000*np.mean(times[1:])+160*frame+96*np.mean(times[1:]))/3600,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
 del initial,trained,expected,batch;gc.collect();torch.cuda.empty_cache();status('preflight_passed')

def train(c,plan):
 if (ROOT/'diagnostics/training_complete.json').exists():return
 before=frozen(c);last=ROOT/'weights/last.pt';log=ROOT/'logs/training.csv';start=0
 if last.exists():
  v=torch.load(last,map_location='cpu',weights_only=False);restore(c,v,plan);start=v['step'];del v
 else:
  v=snapshot(c,0,plan);save(ROOT/'weights/initial.pt',v);save(last,v);del v
 if log.exists():
  old=list(csv.DictReader(log.open()));keep=[r for r in old if int(r['step'])<=start]
  if len(old)>len(keep):write(ROOT/'failures'/f'orphaned_{time.time_ns()}.json',old[len(keep):])
  assert len(keep)==start
  with log.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(old[0]));w.writeheader();w.writerows(keep)
 online_probes(c,plan,start) if start in CHECKPOINTS else None
 status('training',step=start,total=6000)
 with log.open('a',newline='') as f:
  writer=None
  for step,batch in enumerate(batches(plan,start,6000),start+1):
   guard();t=time.time();rec,proof=update(c,plan,batch,step);torch.cuda.synchronize();rec['seconds']=time.time()-t
   if writer is None:
    writer=csv.DictWriter(f,fieldnames=list(rec))
    if start==0:writer.writeheader()
   writer.writerow(rec);f.flush()
   if step%100==0 or step==1 or step in CHECKPOINTS:
    v=snapshot(c,step,plan);save(last,v)
    if step in CHECKPOINTS:save(ROOT/'weights'/f'step_{step:04d}.pt',v)
    write(ROOT/'diagnostics'/f'noise_{step:04d}.json',proof);del v
   if step in CHECKPOINTS:online_probes(c,plan,step)
   if step%10==0:status('training',step=step,total=6000,seconds=rec['seconds'],A_loss=rec['A_loss'],B_loss=rec['B_loss'])
 assert_frozen(c);assert before==frozen(c)
 write(ROOT/'diagnostics/frozen_final.json',dict(passed=True,before=before,after=frozen(c)))
 write(ROOT/'diagnostics/training_complete.json',dict(steps=6000,protocol_id=plan['protocol_id'],final_sha256=sha(ROOT/'weights/step_6000.pt')))

CHECKPOINTS=[0,1000,2000,3000,4500,6000]

@torch.no_grad()
def training_probe(c,plan,step):
 from polarapp.operations import inter_data_process
 from rgb_final_confirmation.evaluation import psnr,ssim
 path=ROOT/'metrics'/f'train_probe_{step:04d}.json'
 if path.exists():
  v=read(path);assert v['protocol_id']==plan['protocol_id'] and v['step']==step and v['frames']==16;return
 saved=rng();e_mode=c['E'].training;c['E'].eval();rows=[]
 try:
  index=read(ROOT/'config/cache_index.json')['entries']
  for i in range(16):
   guard();rec=index[i];assert sha(ROOT/rec['path'])==rec['sha256']
   batch=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False)
   pd=batch['pd'][None].to(DEVICE);target=batch['target'][None].to(DEVICE)
   seed_step(2027,'fixed_train_probe42',i,[batch['sample_id']]);noise=torch.randn(1,16,256,device=DEVICE)
   metrics={};hashes={}
   with h.NoiseCheck(dict(S=c['baseline'],**c['tasks']),noise) as proof:
    pred,_=c['baseline'].forward_inputs(inter_data_process(pd),noise)
    for arm in ['S','A','B']:
     if arm!='S':pred=predict(c,arm,pd,noise)
     loss=float(c['criterion'](pred,target));clipped=pred.clamp(0,1)
     metrics[arm]=dict(loss=loss,psnr=psnr(clipped,target),ssim=ssim(clipped,target));hashes[arm]=h.tensor_sha(clipped)
     del pred,clipped
   rows.append(dict(ROI_index=i,sample_id=batch['sample_id'],ROI_sha256=rec['sha256'],metrics=metrics,output_sha256=hashes,noise_calls=proof.calls))
   del pd,target,batch
 finally:set_rng(saved);c['E'].train(e_mode)
 write(path,dict(protocol_id=plan['protocol_id'],step=step,frames=16,rows=rows,diagnostic_only=True))

@torch.no_grad()
def online_probes(c,plan,step):
 assert step in CHECKPOINTS
 saved=rng()
 try:
  before=frozen(c);tick=time.time()
  training_probe(c,plan,step)
  rows=evaluate_rows(c,plan,step,plan['probe_ids'],ROOT/'metrics'/f'val_probe_{step:04d}.json',identity=step==0)
  assert before==frozen(c);assert_frozen(c)
  write(ROOT/'diagnostics'/f'probe_complete_{step:04d}.json',dict(passed=True,step=step,train_frames=16,val_frames=16,seconds=time.time()-tick,optimizer_updates=False))
 finally:set_rng(saved)


def evaluate(c,plan):
 if (ROOT/'reports/summary.json').exists():return
 summaries={}
 for step in CHECKPOINTS:
  rows=read(ROOT/'metrics'/f'val_probe_{step:04d}.json')['rows']
  summaries[str(step)]={a+'-'+b:h.compare(rows,a,b) for a,b in [('A','S'),('B','S'),('B','A')]}
 rows=evaluate_rows(c,plan,6000,plan['validation_ids'],ROOT/'metrics/step_6000.json')
 end={a+'-'+b:h.compare(rows,a,b) for a,b in [('A','S'),('B','S'),('B','A')]}
 passed=all(end['B-'+b]['psnr']['scene_equal']>=.05 and end['B-'+b]['ssim']['scene_equal']>=0 and end['B-'+b]['psnr']['positive_scenes']>=6 for b in ['A','S'])
 train_curve={str(step):{arm:{metric:float(np.mean([r['metrics'][arm][metric] for r in read(ROOT/'metrics'/f'train_probe_{step:04d}.json')['rows']])) for metric in ['loss','psnr','ssim']} for arm in ['S','A','B']} for step in CHECKPOINTS}
 write(ROOT/'reports/summary.json',dict(protocol_id=plan['protocol_id'],steps=6000,validation=64,probe_checkpoints=CHECKPOINTS,probe_frames=16,test=0,exploratory_only=True,probe_evaluations=summaries,training_probe_curve=train_curve,endpoint_comparisons=end,gate_passed=passed,next_action='register_fixed_endpoint_S_A_B_full872_188_confirmation' if passed else 'stop_instance',limitations=plan['scientific']['limitations']+['6000 is finite budget, not convergence guarantee']))

def finish(plan):
 status('independent_audit');env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
 subprocess.run([sys.executable,str(ROOT/'scripts/audit_study.py')],check=True,env=env);verify(plan)
 excluded={'artifact_manifest.json','closeout.json','status.json','controller_heartbeat.json','resource_wait.json','worker_exit.json','worker.log','controller.log','archive_audit.json','final_review_closeout.json','postcloseout_archive_audit.json'}
 files=[]
 for p in sorted(ROOT.rglob('*')):
  if not p.is_file() or p.is_symlink() or p.name in excluded or p.suffix in ['.pyc','.tmp'] or 'cache' in p.relative_to(ROOT).parts or '__pycache__' in p.parts:continue
  files.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
 write(ROOT/'artifact_manifest.json',dict(protocol_id=plan['protocol_id'],cache_payloads_covered_by_index=True,files=files))
 subprocess.run([sys.executable,str(ROOT/'scripts/audit_study.py'),'--archive'],check=True,env=env)
 close=dict(success=True,exploratory_only=True,steps=6000,validation=64,probe_checkpoints=CHECKPOINTS,probe_frames=16,test=0,protocol_id=plan['protocol_id'],archive_manifest_sha256=sha(ROOT/'artifact_manifest.json'),independent_audit_verified=True,archive_verified=True,next_action=read(ROOT/'reports/summary.json')['next_action'],time_unix=time.time())
 write(ROOT/'reports/final_review_closeout.json',close);write(ROOT/'closeout.json',close)
 subprocess.run([sys.executable,str(ROOT/'scripts/audit_study.py'),'--postcloseout'],check=True,env=env)
 status('completed',**close)

def run():
 lock=(ROOT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 plan=read(ROOT/'config/locked.json');verify(plan)
 if (ROOT/'closeout.json').exists():assert read(ROOT/'closeout.json')['success'];return
 guard();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
 c=components(plan);torch.cuda.reset_peak_memory_stats()
 preflight_before_cache(c,plan);prepare_cache(c,plan,warmup_only=True);preflight_training(c,plan);prepare_cache(c,plan)
 train(c,plan);evaluate(c,plan);del c;gc.collect();torch.cuda.empty_cache();finish(plan)

if __name__=='__main__':
 try:prepare() if '--prepare' in sys.argv else run()
 except Exception:
  failure=dict(time_unix=time.time(),traceback=traceback.format_exc());write(ROOT/'failures'/f'worker_{time.time_ns()}.json',failure);status('failed',**failure);raise
