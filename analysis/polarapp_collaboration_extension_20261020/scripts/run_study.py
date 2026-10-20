"""43: resume audited42 Adam and add task-driven E/T metric adaptation.

No source run is modified. Existing native forward, cache, FANet and exact
bilevel kernel are reused; only continuation/control/budget glue is new.
"""
from __future__ import annotations
import copy,csv,fcntl,gc,importlib.util,json,math,os,random,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent
OLD=PROJECT/'polarapp_postprocess_value_20261019';OLD23=PROJECT/'polarapp_single_collaboration_20261013'
spec=importlib.util.spec_from_file_location('native42',OLD/'scripts/run_study.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
b.ROOT=ROOT
h=b.h;read,write,sha,canonical,receipt=b.read,b.write,b.sha,b.canonical,b.receipt
cpu,equal,rng,set_rng,seed_step=b.cpu,b.equal,b.rng,b.set_rng,b.seed_step
DEVICE=torch.device('cuda:0');ARMS=['A','B','C'];START=1792429665.;ORIGINAL_END=START+32400
SOURCE_SHA='a93af237aaf90914f237261dd5e0e7eb5fa6eeaff7839f08cdf2309b8c05b733'
sys.path.insert(0,str(OLD23))
from modules.fanet import SameScaleFANet
from modules.meta import FunctionalModelState,bilevel_probe


def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 # Large checkpoint payloads are immutable. Only a small symlink is replaced.
 if path.name=='last.pt' and path.parent.name=='weights_resumed':
  payload=path.parent/f"state_{value['step']:05d}_{time.time_ns()}.pt"
  with payload.open('xb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
  temporary=path.parent/f'.lastlink_{time.time_ns()}';temporary.symlink_to(payload.name);os.replace(temporary,path)
  write(ROOT/'diagnostics/last_checkpoint_pointer.json',dict(step=value['step'],global_step=value['global_step'],protocol_id=value['protocol_id'],payload=str(payload),bytes=payload.stat().st_size))
  fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)
 elif path.parent.name=='weights_resumed' and path.name.startswith('step_'):
  assert not path.exists();target=(path.parent/'last.pt').resolve();assert target.name.startswith(f"state_{value['step']:05d}_");os.link(target,path)
 else:
  assert not path.exists(),f'Immutable checkpoint already exists: {path}'
  b.save(path,value)


def authorized_deadline():
 p=ROOT/'config/user_budget_extension.json'
 if not p.exists():return ORIGINAL_END
 v=read(p);assert v['protocol_id']=='ca7f2ae7897bc6f51464a65dc6e00273b1ff1b90f538a68e16f1d70d7e1cf0af' and v['scientific_id']=='16736ccb264df16a14c0e929e8294996799c06695744aea50d1f7ace5fc5a07b'
 assert v['previous_deadline_unix']==ORIGINAL_END and v['new_deadline_unix']==1792488600 and v['authorized_by']=='human_user'
 return v['new_deadline_unix']
END=authorized_deadline()

class BudgetExpired(RuntimeError):pass
def guard():
 if time.time()>=authorized_deadline()-60:raise BudgetExpired('Nine-hour deadline; preserve last matching checkpoint and release GPU')
def status(stage,**kw):
 v=dict(stage=stage,time_unix=time.time());v.update(kw);write(ROOT/'status.json',v);print(json.dumps(v),flush=True)
def verify(plan):
 v=dict(plan);v.pop('protocol_id');assert canonical(v)==plan['protocol_id'] and plan['experiment']==43
 assert canonical(plan['scientific'])==plan['scientific_id'] and plan['deadline_unix']==ORIGINAL_END
 codes={r['path']:r for r in plan['code_sources']};rep={};p=ROOT/'config/implementation_revisions.json'
 for rev in read(p) if p.exists() else []:
  assert rev['scientific_id']==plan['scientific_id'] and rev['protocol_id']==plan['protocol_id']
  for r in rev['replacements']:
   assert r['path'] in codes and r['original_sha256']==codes[r['path']]['sha256'] and sha(r['preserved_original_path'])==r['original_sha256'];rep[r['path']]=r['repaired_sha256']
 for r in plan['sources']+plan['code_sources']:assert sha(r['path'])==rep.get(r['path'],r['sha256']),r['path']

def prepare():
 for name in ['config','logs','weights','metrics','reports','diagnostics','failures','cache']:(ROOT/name).mkdir(parents=True,exist_ok=True)
 p=ROOT/'config/methods.json'
 if p.exists():verify(read(p));return
 guard();old=read(OLD/'config/locked.json');assert read(OLD/'closeout.json')['success'] and read(OLD/'reports/postcloseout_archive_audit.json')['passed']
 assert sha(OLD/'weights/step_6000.pt')==SOURCE_SHA
 index=read(OLD/'config/cache_index.json');assert len(index['entries'])==12000 and read(OLD/'diagnostics/cache_complete.json')['index_sha256']==sha(OLD/'config/cache_index.json')
 g=torch.Generator().manual_seed(2029);order=[]
 while len(order)<18000:order.extend(torch.randperm(12000,generator=g).tolist())
 entries=[]
 for i,old_i in enumerate(order[:18000]):
  rec=dict(index['entries'][old_i]);rec['source_ROI_index']=old_i;rec['path']=str(OLD/rec['path']);rec['presentation_index']=i;entries.append(rec)
 write(ROOT/'config/continuation_ROI.json',dict(entries=entries,seed=2029))
 scenes=sorted({x['sample_id'].split('/')[2] for x in index['entries']});inner_scenes=scenes[::2];outer_scenes=scenes[1::2]
 inner=[i for i,x in enumerate(index['entries']) if x['sample_id'].split('/')[2] in inner_scenes];outer=[i for i,x in enumerate(index['entries']) if x['sample_id'].split('/')[2] in outer_scenes]
 g=torch.Generator().manual_seed(2030);meta=[]
 for i in range(900):meta.append(dict(inner=inner[int(torch.randint(len(inner),(1,),generator=g))],outer=outer[int(torch.randint(len(outer),(1,),generator=g))]))
 write(ROOT/'config/meta_ROI.json',dict(items=meta,inner_scenes=inner_scenes,outer_scenes=outer_scenes,seed=2030))
 fanet=OLD23/'weights/FANet.pth';assert sha(fanet)=='20fe6defc1ac37b5d7011bce5b83a43c9e21e6948f36bbb98933a742a7b75a4b'
 scientific=dict(experiment=43,source_step=6000,source_sha256=SOURCE_SHA,arms=ARMS,additional_candidates=[9000,6000],batch=2,crop=128,
  frozen='native D and all task priors',forward='audited42 unchanged: P_D anchored RGB, E only polarization, native float bridge',
  task_loss=old['scientific']['loss'],T_lr=1e-5,E_lr=2e-5,Adam='inherited42 including complete moments and step6000',
  cache='reuse existing12000 fixed float32 ROIs; no new image/ROI diversity',ROI_order_sha=sha(ROOT/'config/continuation_ROI.json'),
  collaboration=dict(start='copy B6000 E/T/Adam exactly',metric='SameScaleFANet pretrained official, energy-preserving P1 adapter',
   partition='P1+official FT/MFG final convs+all FANet LayerNorm',lambda_rule='0.1 median(task_global_grad_norm/alignment_global_grad_norm), first4 train minibatches, then fixed',
   meta_every=10,meta_batch_inner=1,meta_batch_outer=1,meta_source_sha=sha(ROOT/'config/meta_ROI.json'),
   inner_E='mid_output',inner_T='reduce_chan_level3',inner_E_lr=2e-5,inner_T_lr=1e-5,outer='native RGB task-only on disjoint TRAIN scene set',
   exact_second_order=True,meta_Adam_lr=1e-5,outer_numeric_scale=1000.,joint_scope='same full E/T as B',additional_compute_disclosed=True),
  checkpoints='relative0,1000,3000,6000,+9000 if chosen; same16 train/val; endpoint64',
  full_confirmation='conditional gate and remaining measured budget, same endpoint all S/A/B/C872val+188test; no new training',budget_selection='v2: select by core fit; full confirmation separately guarded, no nine-hour extension',
  gates=dict(PSNR=.05,SSIM=0,positive_scenes=6,B_comparators=['A','S'],C_comparators=['B','A','S']),
  fixed_deadline_unix=END,seed=2027,stop='no auto extension/seeds/LR/lambda/loss/Oracle/other mechanisms')
 plan={k:copy.deepcopy(old[k]) for k in ['source_root','source_config','manifest','D_checkpoint','T_checkpoint','E_checkpoint','validation_ids','probe_ids','full_index_map','largest_train','largest_val']}
 plan.update(experiment=43,scientific=scientific,scientific_id=canonical(scientific),source_checkpoint=str(OLD/'weights/step_6000.pt'),fanet_checkpoint=str(fanet),
  source_cache_index=str(OLD/'config/cache_index.json'),deadline_unix=END,start_unix=START)
 sources=copy.deepcopy(old['sources'])
 sources += [receipt(x,role) for x,role in [(ROOT/'PROTOCOL_ZH.md','43 preregistration'),(OLD/'weights/step_6000.pt','source weights complete Adam/RNG'),(OLD/'config/locked.json','42 source protocol'),(OLD/'artifact_manifest.json','audited42 archive'),(OLD/'config/cache_index.json','immutable source cache'),(OLD/'metrics/step_6000.json','42 fixed64 baseline'),(fanet,'official metric initialization'),(ROOT/'config/continuation_ROI.json','common extended training presentations'),(ROOT/'config/meta_ROI.json','training-only meta presentations')]]
 codes={Path(x['path']) for x in old['code_sources']};codes.update((ROOT/'scripts').glob('*.py'));codes.update((OLD23/'modules').glob('*.py'));codes.add(OLD23/'vendor/PolarAPP/DfP/utils/utils_net.py')
 plan.update(sources=sources,code_sources=[receipt(x) for x in sorted(codes)]);plan['protocol_id']=canonical(plan);write(p,plan);verify(plan)
 vgg=Path('/home/li/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth');link=ROOT/'experiments/pretrained_models'/vgg.name;link.parent.mkdir(parents=True,exist_ok=True)
 if not link.exists():link.symlink_to(vgg)
 status('prepared',protocol_id=plan['protocol_id'],deadline_unix=END)

class EView(nn.Module):
 def __init__(self,core):super().__init__();self.core=core
 def forward(self,pd):
  p,features,_=self.core.decode(pd);return p.clamp(0,1),features
class TView(nn.Module):
 def __init__(self,task):super().__init__();self.task=task
 def forward(self,polar,anchor,noise):return self.task.forward_inputs(b.task_inputs(polar,anchor),noise)

def components(plan):
 c=b.components(plan);v=torch.load(plan['source_checkpoint'],map_location='cpu',weights_only=False)
 for a in ['A','B']:c['tasks'][a].net_g.load_state_dict(v['weights'][a],strict=True);c['opts'][a].load_state_dict(copy.deepcopy(v['optimizers'][a]))
 c['E'].load_state_dict(v['E'],strict=True);c['tasks']['C']=copy.deepcopy(c['tasks']['B']);c['E_C']=copy.deepcopy(c['E'])
 c['opts']['C']=torch.optim.Adam([dict(params=list(c['tasks']['C'].net_g.parameters()),lr=1e-5,name='T'),dict(params=[p for p in c['E_C'].parameters() if p.requires_grad],lr=2e-5,name='E')],betas=(.9,.999),eps=1e-8,weight_decay=0)
 c['opts']['C'].load_state_dict(copy.deepcopy(v['optimizers']['B']));source_rng=cpu(v['rng']);del v
 c['FA']=SameScaleFANet(OLD23/'vendor/PolarAPP/DfP/utils/utils_net.py',plan['fanet_checkpoint']).to(DEVICE)
 partition=c['FA'].set_trainable_partition();c['FA_names']=partition['trainable_names']
 c['meta_opt']=torch.optim.Adam([p for p in c['FA'].parameters() if p.requires_grad],lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=0)
 c['E_view']=EView(c['E_C']);c['T_view']=TView(c['tasks']['C']);c['lambda']=0.
 c['FA'].requires_grad_(False).eval();set_rng(source_rng);return c

def snapshot(c,step,plan):
 return dict(step=step,global_step=6000+step,protocol_id=plan['protocol_id'],weights={a:cpu(t.net_g.state_dict()) for a,t in c['tasks'].items()},
  E=cpu(c['E'].state_dict()),E_C=cpu(c['E_C'].state_dict()),FA=cpu(c['FA'].state_dict()),optimizers={a:cpu(o.state_dict()) for a,o in c['opts'].items()},meta_opt=cpu(c['meta_opt'].state_dict()),lambda_value=c['lambda'],rng=rng())
def restore(c,v,plan):
 assert v['protocol_id']==plan['protocol_id']
 for a in ARMS:c['tasks'][a].net_g.load_state_dict(v['weights'][a],strict=True);c['opts'][a].load_state_dict(copy.deepcopy(v['optimizers'][a]));c['opts'][a].zero_grad(set_to_none=True)
 for key in ['E','E_C','FA']:c[key].load_state_dict(v[key],strict=True)
 c['meta_opt'].load_state_dict(copy.deepcopy(v['meta_opt']));c['meta_opt'].zero_grad(set_to_none=True);c['lambda']=v['lambda_value'];set_rng(v['rng'])
def frozen(c):
 value=b.frozen(c);value['C_prior']=b.model_hash(c['tasks']['C'],True);value['E_C_inactive']=b.model_hash(c['E_C'],True)
 value['FA_fixed']=canonical({n:h.tensor_sha(p) for n,p in c['FA'].named_parameters() if n not in c['FA_names']});return value
def assert_frozen(c):
 b.assert_frozen(c)
 for p in c['E_C'].parameters():
  if not p.requires_grad:assert p.grad is None
def entries():return read(ROOT/'config/continuation_ROI.json')['entries']
def batches(start,end):
 for batch in DataLoader(b.CachedROI(entries()[start*2:end*2]),batch_size=2,shuffle=False,num_workers=2,pin_memory=True,drop_last=True):
  yield {k:v.to(DEVICE) if torch.is_tensor(v) else v for k,v in batch.items()}
def meta_item(index):
 rec=read(OLD/'config/cache_index.json')['entries'][index];path=OLD/rec['path'];assert sha(path)==rec['sha256']
 batch=torch.load(path,map_location='cpu',weights_only=False);return {k:v[None].to(DEVICE) if torch.is_tensor(v) else [v] for k,v in batch.items()}
def anchor(pd):
 from polarapp.operations import inter_data_process
 return inter_data_process(pd)[0]
def forward(c,a,pd,noise,features=False):
 if a=='A':polar=pd;pf=None
 else:
  core=c['E'] if a=='B' else c['E_C']
  if features:polar,pf,_=core.decode(pd);polar=polar.clamp(0,1)
  else:polar=core.pretrain_forward(pd).clamp(0,1);pf=None
 pred,tf=c['tasks'][a].forward_inputs(b.task_inputs(polar,anchor(pd)),noise)
 return pred,pf,tf
def alignment(fa,pf,tf):return fa.alignment_loss(pf,tf)
def norm(grads):return math.sqrt(sum(float(g.detach().double().square().sum()) for g in grads if g is not None))

def calibrate(c):
 saved=rng();vals=[];parameters=[p for p in list(c['tasks']['C'].parameters())+list(c['E_C'].parameters()) if p.requires_grad]
 for step,batch in enumerate(batches(0,4),1):
  guard();seed_step(2027,'train42',6000+step,batch['sample_id']);noise=torch.randn(2,16,256,device=DEVICE)
  pred,pf,tf=forward(c,'C',batch['pd'],noise,True);task=c['criterion'](pred,batch['target']);fa=alignment(c['FA'],pf,tf)
  gt=torch.autograd.grad(task,parameters,retain_graph=True,allow_unused=True);gf=torch.autograd.grad(fa,parameters,allow_unused=True)
  nt,nf=norm(gt),norm(gf);assert nt>0 and nf>0 and math.isfinite(nt/nf);vals.append(dict(task_gradient_norm=nt,alignment_gradient_norm=nf,ratio=nt/nf))
  del pred,pf,tf,task,fa,gt,gf
 c['lambda']=.1*float(np.median([x['ratio'] for x in vals]));write(ROOT/'config/calibration.json',dict(lambda_value=c['lambda'],training_only=True,batches=vals,rule='.1 median gradient norm ratio',selected_from_validation=False));set_rng(saved);gc.collect();torch.cuda.empty_cache()

def meta_update(c,relative_step):
 guard();saved=rng();record=read(ROOT/'config/meta_ROI.json')['items'][relative_step//10-1]
 inner=meta_item(record['inner']);outer=meta_item(record['outer']);assert inner['sample_id']!=outer['sample_id']
 c['FA'].set_trainable_partition();checkpoint=c['E_C'].activation_checkpoint;c['E_C'].activation_checkpoint=False;offloaded=evaluation_optimizer_offload(c,include_meta=False)
 try:
  up_names=[n for n,p in c['E_view'].named_parameters() if n.startswith('core.mid_output.')]
  down_names=[n for n,p in c['T_view'].named_parameters() if n.startswith('task.model.net_g.reduce_chan_level3.')]
  assert up_names and down_names
  states=(FunctionalModelState.from_module(c['E_view'],up_names,2e-5,label='E_output_probe'),FunctionalModelState.from_module(c['T_view'],down_names,1e-5,label='T_fusion_probe'),FunctionalModelState.from_module(c['FA'],c['FA_names'],1e-5,label='FA'))
  seed_step(2027,'meta_inner43',6000+relative_step,inner['sample_id']);ni=torch.randn(1,16,256,device=DEVICE)
  seed_step(2027,'meta_outer43',6000+relative_step,outer['sample_id']);no=torch.randn(1,16,256,device=DEVICE)
  def inner_loss(u,t,f):
   polar,pf=u.call(inner['pd']);pred,tf=t.call(polar,anchor(inner['pd']),ni);return cosine_loss(f.call(pf,tf))
  def outer_loss(u,t,f):
   polar,_=u.call(outer['pd']);pred,_=t.call(polar,anchor(outer['pd']),no);return 1000.*c['criterion'](pred,outer['target'])
  result=bilevel_probe(*states,inner_loss,outer_loss,outer_create_graph=False)
  c['meta_opt'].zero_grad(set_to_none=True);named=dict(c['FA'].named_parameters())
  for name,g in result.hypergradients.items():named[name].grad=g.detach().clone()
  ng=float(torch.nn.utils.clip_grad_norm_([named[n] for n in c['FA_names']],1.,error_if_nonfinite=True));assert ng>0
  c['meta_opt'].step();rec=dict(inner_loss=float(result.inner_loss.detach()),outer_task_loss=float(result.outer_loss.detach())/1000.,hypergradient_scaled_norm=ng,second_order=True,live_E_T_unchanged=True,inner_id=inner['sample_id'][0],outer_id=outer['sample_id'][0])
  del result,states
 finally:
  c['meta_opt'].zero_grad(set_to_none=True);c['FA'].requires_grad_(False);c['E_C'].activation_checkpoint=checkpoint;set_rng(saved);gc.collect();torch.cuda.empty_cache();evaluation_optimizer_restore(offloaded)
 return rec
def cosine_loss(pair):
 x,y=pair;return sum(1-torch.nn.functional.cosine_similarity(x['TF'+str(i)],y['MF'+str(i)],dim=1).mean() for i in [1,2,3])/3

def joint_update(c,batch,relative_step,weight=None,check=False):
 for opt in c['opts'].values():opt.zero_grad(set_to_none=True)
 c['FA'].requires_grad_(False);seed_step(2027,'train42',6000+relative_step,batch['sample_id']);noise=torch.randn(2,16,256,device=DEVICE)
 rec=dict(step=relative_step,global_step=6000+relative_step,sample_ids=';'.join(batch['sample_id']));grads={}
 with h.NoiseCheck(c['tasks'],noise) as proof:
  for a in ARMS:
   pred,pf,tf=forward(c,a,batch['pd'],noise,features=a=='C');task=c['criterion'](pred,batch['target']);loss=task
   rec[a+'_task_loss']=float(task.detach())
   if a=='C':
    fa=alignment(c['FA'],pf,tf);loss=task+(c['lambda'] if weight is None else weight)*fa;rec['C_alignment_loss']=float(fa.detach())
   assert torch.isfinite(loss);loss.backward()
   if check and a in ['B','C']:grads[a]={n:h.tensor_sha(p.grad) for n,p in c['tasks'][a].net_g.named_parameters() if p.grad is not None}
   rec[a+'_T_grad']=float(torch.nn.utils.clip_grad_norm_(c['tasks'][a].net_g.parameters(),1.,error_if_nonfinite=True))
   if a!='A':rec[a+'_E_grad']=float(torch.nn.utils.clip_grad_norm_([p for p in c['E' if a=='B' else 'E_C'].parameters() if p.requires_grad],1.,error_if_nonfinite=True))
   del pred,pf,tf,task,loss
  assert len(proof.calls)==3 and len({x['sha256'] for x in proof.calls})==1
  for opt in c['opts'].values():opt.step();opt.zero_grad(set_to_none=True)
 return rec,dict(noise_calls=proof.calls,BC_task_gradients_equal=equal(grads.get('B'),grads.get('C')) if check else None)
def advance(c,batch,step):
 record={};tick=time.time()
 if step%10==0:record=meta_update(c,step)
 rec,proof=joint_update(c,batch,step);torch.cuda.synchronize();rec['seconds']=time.time()-tick
 for key in ['inner_loss','outer_task_loss','hypergradient_scaled_norm']:rec['meta_'+key]=record.get(key,0.)
 return rec,proof,record


def evaluation_optimizer_offload(c,include_meta=True):
 locations=[]
 for opt in list(c['opts'].values())+([c['meta_opt']] if include_meta else []):
  for state in opt.state.values():
   for key,value in list(state.items()):
    if torch.is_tensor(value) and value.is_cuda:
     locations.append((state,key,value.device));state[key]=value.cpu()
 gc.collect();torch.cuda.empty_cache();return locations

def evaluation_optimizer_restore(locations):
 for state,key,device in locations:state[key]=state[key].to(device)

@torch.no_grad()
def evaluate_rows(c,plan,step,ids,path,identity=False):
 from rgb_final_confirmation.evaluation import psnr,ssim
 from polarapp.operations import inter_data_process
 saved=rng();offloaded=evaluation_optimizer_offload(c);modes=[c['E'].training,c['E_C'].training];c['E'].eval();c['E_C'].eval();rows=read(path)['rows'] if path.exists() else []
 assert [x['sample_id'] for x in rows]==ids[:len(rows)];old={x['sample_id']:x for x in read(OLD/'metrics/step_6000.json')['rows']}
 try:
  for sid in ids[len(rows):]:
   guard();pd,batch=b.full_reconstruction(c,sid);seed_step(2026,plan.get('evaluation_phase','validation42'),plan['full_index_map'][sid],[sid]);noise=torch.randn(1,16,256,device=DEVICE);metrics={};hashes={}
   with h.NoiseCheck(dict(S=c['baseline'],**c['tasks']),noise) as proof:
    pred=c['baseline'].forward_inputs(inter_data_process(pd),noise)[0];pred=pred.clamp(0,1)
    hashes['S']=h.tensor_sha(pred);metrics['S']=dict(psnr=psnr(pred,batch['target']),ssim=ssim(pred,batch['target']));del pred
    if sid in old:assert hashes['S']==old[sid]['output_sha256']['S'] and h.tensor_sha(pd)==old[sid]['PD_sha256']
    for a in ARMS:
     pred=forward(c,a,pd,noise)[0];pred=pred.clamp(0,1);hashes[a]=h.tensor_sha(pred);metrics[a]=dict(psnr=psnr(pred,batch['target']),ssim=ssim(pred,batch['target']));del pred
     if identity:assert hashes[a]==old[sid]['output_sha256']['B' if a=='C' else a]
   rows.append(dict(sample_id=sid,scene=batch['scene'][0],original_size=True,shape=list(batch['target'].shape),PD_sha256=h.tensor_sha(pd),metrics=metrics,output_sha256=hashes,noise_calls=proof.calls))
   write(path,dict(protocol_id=plan['protocol_id'],step=step,frames=len(rows),rows=rows));status('evaluating',step=step,completed=len(rows),total=len(ids));del pd,batch;gc.collect();torch.cuda.empty_cache()
 finally:evaluation_optimizer_restore(offloaded);set_rng(saved);c['E'].train(modes[0]);c['E_C'].train(modes[1])
 return rows

@torch.no_grad()
def online_probes(c,plan,step):
 path=ROOT/'metrics'/f'train_probe_{step:05d}.json';saved=rng();modes=[c['E'].training,c['E_C'].training];c['E'].eval();c['E_C'].eval()
 try:
  if not path.exists():
   from rgb_final_confirmation.evaluation import psnr,ssim
   from polarapp.operations import inter_data_process
   rows=[]
   for i in range(16):
    guard();batch=meta_item(i);pd=batch['pd'];target=batch['target'];seed_step(2027,'fixed_train_probe42',i,batch['sample_id']);noise=torch.randn(1,16,256,device=DEVICE);metrics={};hashes={}
    with h.NoiseCheck(dict(S=c['baseline'],**c['tasks']),noise) as proof:
     pred=c['baseline'].forward_inputs(inter_data_process(pd),noise)[0]
     for a in ['S']+ARMS:
      if a!='S':pred=forward(c,a,pd,noise)[0]
      loss=float(c['criterion'](pred,target));clipped=pred.clamp(0,1);hashes[a]=h.tensor_sha(clipped);metrics[a]=dict(loss=loss,psnr=psnr(clipped,target),ssim=ssim(clipped,target));del pred,clipped
    rows.append(dict(ROI_index=i,sample_id=batch['sample_id'][0],metrics=metrics,output_sha256=hashes,noise_calls=proof.calls))
   write(path,dict(protocol_id=plan['protocol_id'],step=step,frames=16,rows=rows))
  evaluate_rows(c,plan,step,plan['probe_ids'],ROOT/'metrics'/f'val_probe_{step:05d}.json',identity=step==0)
 finally:set_rng(saved);c['E'].train(modes[0]);c['E_C'].train(modes[1])
 write(ROOT/'diagnostics'/f'probe_complete_{step:05d}.json',dict(passed=True,step=step,training_frames=16,validation_frames=16,optimizer_updated=False))

def preflight(c,draft):
 pf=ROOT/'config/preflight.json'
 if pf.exists():assert read(pf)['passed'];return
 guard();status('preflight');models=[c['baseline'].net_g]+[c['tasks'][a].net_g for a in ARMS]+[c['E'],c['E_C']]
 pointers=[{p.data_ptr() for p in m.parameters()} for m in models]
 assert all(not (pointers[i]&pointers[j]) for i in range(len(pointers)) for j in range(i)),'Unexpected arm storage alias'
 initial0=snapshot(c,0,draft);fixed=frozen(c)
 source=torch.load(draft['source_checkpoint'],map_location='cpu',weights_only=False)
 assert equal(initial0['weights']['A'],source['weights']['A']) and equal(initial0['weights']['B'],source['weights']['B'])
 assert equal(initial0['weights']['B'],initial0['weights']['C']) and equal(initial0['E'],initial0['E_C'])
 assert equal(initial0['optimizers']['B'],initial0['optimizers']['C']) and equal(initial0['optimizers']['A'],source['optimizers']['A'])
 assert equal(initial0['optimizers']['B'],source['optimizers']['B']);del source
 t=time.time();ids=list(dict.fromkeys([draft['validation_ids'][0],draft['largest_val']]))
 evaluate_rows(c,draft,0,ids,ROOT/'diagnostics/source_identity.json',True);torch.cuda.synchronize();frame=(time.time()-t)/len(ids)
 restore(c,initial0,draft);calibrate(c);initial=snapshot(c,0,draft)
 batch=next(batches(0,1));rec,proof=joint_update(c,batch,1,weight=0,check=True)
 assert proof['BC_task_gradients_equal'] and equal(c['tasks']['B'].net_g.state_dict(),c['tasks']['C'].net_g.state_dict()) and equal(c['E'].state_dict(),c['E_C'].state_dict())
 assert equal(c['opts']['B'].state_dict(),c['opts']['C'].state_dict());restore(c,initial,draft)
 times=[]
 for i,batch in enumerate(batches(0,3),1):
  tick=time.time();joint_update(c,batch,i);torch.cuda.synchronize();times.append(time.time()-tick)
 assert not equal(c['E'].state_dict(),c['E_C'].state_dict()),'FA has no actual effect on E update'
 restore(c,initial,draft);batch=next(batches(9,10));save(ROOT/'diagnostics/preflight_resume.pt',initial)
 tick=time.time();rec,proof,mrec=advance(c,batch,10);torch.cuda.synchronize();combined=time.time()-tick;expected=snapshot(c,1,draft)
 assert not equal(initial['FA'],expected['FA']) and mrec['hypergradient_scaled_norm']>0
 restore(c,torch.load(ROOT/'diagnostics/preflight_resume.pt',map_location='cpu',weights_only=False),draft);advance(c,batch,10);assert equal(expected,snapshot(c,1,draft))
 restore(c,initial,draft);before=snapshot(c,0,draft);online_probes(c,draft,0);assert equal(before,snapshot(c,0,draft));assert_frozen(c);assert fixed==frozen(c)
 pair=float(np.median(times[1:]));meta=max(.01,combined-pair);forecasts={}
 for steps in [9000,6000]:
  probes=5 if steps==9000 else 4;seconds=1.25*(steps*pair+(steps//10)*meta+(16*probes+64)*frame)+1200
  forecasts[str(steps)]=seconds
 write(ROOT/'diagnostics/performance_budget.json',dict(core_forecasts_seconds=forecasts,optional_full_confirmation_seconds=1.25*1060*frame+1200,pair_seconds=pair,meta_seconds=meta,frame_seconds=frame,remaining_seconds=END-time.time(),selection_uses_scores=False))
 choices=[x for x in [9000,6000] if forecasts[str(x)]<=END-time.time()]
 assert choices,'No candidate core fits nine-hour deadline; no training started'
 steps=choices[0];write(pf,dict(passed=True,parent_protocol_id=draft['protocol_id'],source_weights_Adam_exact=True,BC_same_start=True,zero_FA_same_gradient_update=True,FA_affects_E_update=True,nonzero_exact_hypergradient=True,meta_and_joint_resume_exact=True,probe_weights_Adam_RNG_exact=True,source_frozen=True,all_preflight_updates_discarded=True,pair_seconds=pair,meta_seconds=meta,frame_seconds=frame,forecast_seconds=forecasts,selected_additional_steps=steps,lambda_value=c['lambda'],peak_allocated_bytes=torch.cuda.max_memory_allocated()))
 del initial0,initial,expected,before,batch;gc.collect();torch.cuda.empty_cache();status('preflight_passed',additional_steps=steps)

def lock_plan(draft):
 path=ROOT/'config/locked.json'
 if path.exists():plan=read(path);verify(plan);return plan
 pf=read(ROOT/'config/preflight.json');plan=copy.deepcopy(draft);plan.pop('protocol_id');plan['parent_protocol_id']=draft['protocol_id'];plan['steps']=pf['selected_additional_steps'];plan['lambda_value']=pf['lambda_value']
 plan['scientific']['additional_steps']=plan['steps'];plan['scientific']['lambda_value']=plan['lambda_value'];plan['scientific_id']=canonical(plan['scientific'])
 plan['sources'] += [receipt(ROOT/'config/methods.json','preregistered methods'),receipt(ROOT/'config/preflight.json','performance-only selection'),receipt(ROOT/'config/calibration.json','training-only lambda calibration')]
 plan['protocol_id']=canonical(plan);write(path,plan);verify(plan);return plan


def recovery_preflight(c,plan):
 path=ROOT/'diagnostics/recovery_preflight.json'
 if path.exists():assert read(path)['passed'];return
 status('recovery_preflight');clearance=read(ROOT/'config/recovery_clearance.json');source=Path(clearance['checkpoint_path']);assert sha(source)==clearance['checkpoint_sha256']
 v=torch.load(source,map_location='cpu',weights_only=False);restore(c,v,plan);step=v['step'];before=snapshot(c,step,plan);del v
 assert step==300 and all(int(z['step'])==6304 for a in ARMS for z in before['optimizers'][a]['state'].values())
 frozen_before=frozen(c);batch=next(batches(step+9,step+10))
 advance(c,batch,step+10);expected=snapshot(c,step+1,plan)
 assert equal(before,torch.load(source,map_location='cpu',weights_only=False)), 'Saved CPU state was mutated'
 restore(c,before,plan);advance(c,batch,step+10);assert equal(expected,snapshot(c,step+1,plan))
 restore(c,before,plan);assert equal(before,snapshot(c,step,plan));assert frozen_before==frozen(c)
 evaluate_rows(c,plan,step,list(dict.fromkeys([plan['largest_val'],plan['validation_ids'][0]])),ROOT/'diagnostics/recovery_maximum.json')
 assert equal(before,snapshot(c,step,plan));write(path,dict(passed=True,step=step,actual_main_Adam_counter=6304,documented_initial_counter_offset=4,CPU_checkpoint_not_mutated=True,meta_joint_replay_exact=True,all_probe_updates_discarded=True,maximum_full_passed=True,frozen_equal=True))
 del before,expected,batch;gc.collect();torch.cuda.empty_cache()

def train(c,plan):
 total=plan['steps'];checkpoints=[0,1000,3000,6000]+([9000] if total==9000 else []);last=ROOT/'weights_resumed/last.pt';log=ROOT/'logs/training.csv';start=0;fixed=frozen(c)
 if not last.exists():
  clearance=read(ROOT/'config/recovery_clearance.json');source=Path(clearance['checkpoint_path'])
  assert clearance['protocol_id']==plan['protocol_id'] and clearance['scientific_id']==plan['scientific_id'] and clearance['preserve_actual_Adam_counter_offset']==4
  assert sha(source)==clearance['checkpoint_sha256'];v=torch.load(source,map_location='cpu',weights_only=False)
  assert v['protocol_id']==plan['protocol_id'] and v['step']==300;save(last,v);del v
 if last.exists():v=torch.load(last,map_location='cpu',weights_only=False);restore(c,v,plan);start=v['step'];del v
 else:
  v=snapshot(c,0,plan);save(ROOT/'weights_resumed/initial.pt',v);save(last,v);del v
  # Reuse the identical verified source metrics with final locked protocol metadata.
  for kind in ['train','val']:
   p=ROOT/'metrics'/f'{kind}_probe_00000.json';v=read(p);v['parent_protocol_id']=v['protocol_id'];v['protocol_id']=plan['protocol_id'];write(p,v)
 if log.exists():
  old=list(csv.DictReader(log.open()));keep=[x for x in old if int(x['step'])<=start];assert len(keep)==start
  if len(old)>len(keep):write(ROOT/'failures'/f'orphaned_{time.time_ns()}.json',old[len(keep):])
  with log.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(old[0]));w.writeheader();w.writerows(keep)
 # Finish an interrupted fixed checkpoint probe before any further update.
 if start in checkpoints and not (ROOT/'diagnostics'/f'probe_complete_{start:05d}.json').exists():
  before=snapshot(c,start,plan);online_probes(c,plan,start)
  assert equal(before,snapshot(c,start,plan)), 'Resumed probe changed weights/Adam/RNG'
  write(ROOT/'diagnostics'/f'probe_resume_exact_{start:05d}.json',dict(passed=True,step=start,weights_Adam_RNG_unchanged=True))
  del before;gc.collect();torch.cuda.empty_cache()
 with log.open('a',newline='') as f:
  writer=None
  for step,batch in enumerate(batches(start,total),start+1):
   guard();rec,proof,meta=advance(c,batch,step)
   if writer is None:
    writer=csv.DictWriter(f,fieldnames=list(rec))
    if start==0:writer.writeheader()
   writer.writerow(rec);f.flush()
   if meta:write(ROOT/'diagnostics/meta_last.json',dict(step=step,**meta))
   if step%100==0 or step==1:
    v=snapshot(c,step,plan);save(last,v)
    if step in checkpoints:save(ROOT/'weights_resumed'/f'step_{step:05d}.pt',v)
    write(ROOT/'diagnostics'/f'noise_{step:05d}.json',proof);del v
   if step in checkpoints:online_probes(c,plan,step)
   if step%10==0:status('training',step=step,global_step=6000+step,total=total,seconds=rec['seconds'])
 assert_frozen(c);assert fixed==frozen(c);write(ROOT/'diagnostics/frozen_final.json',dict(passed=True,before=fixed,after=frozen(c)))
 write(ROOT/'diagnostics/training_complete.json',dict(passed=True,steps=total,meta_updates=total//10,final_sha256=sha(ROOT/'weights_resumed'/f'step_{total:05d}.pt')))

PAIRS=[('A','S'),('B','S'),('C','S'),('B','A'),('C','B'),('C','A')]
def summarize(c,plan):
 rows=evaluate_rows(c,plan,plan['steps'],plan['validation_ids'],ROOT/'metrics/endpoint64.json');stats={a+'-'+z:h.compare(rows,a,z) for a,z in PAIRS}
 gates={a:all(stats[a+'-'+z]['psnr']['scene_equal']>=.05 and stats[a+'-'+z]['ssim']['scene_equal']>=0 and stats[a+'-'+z]['psnr']['positive_scenes']>=6 for z in (['A','S'] if a=='B' else ['B','A','S'])) for a in ['B','C']}
 write(ROOT/'reports/summary.json',dict(protocol_id=plan['protocol_id'],additional_steps=plan['steps'],global_steps=6000+plan['steps'],validation=64,test=0,exploratory_only=True,comparisons=stats,gates=gates,next_action='conditional_full_confirmation' if any(gates.values()) else 'stop_instance',deadline_unix=authorized_deadline()))

def finish(plan):
 status('independent_audit');env=dict(os.environ,CUDA_VISIBLE_DEVICES='');subprocess.run([sys.executable,str(ROOT/'scripts/audit_study.py')],check=True,env=env);verify(plan)
 excluded={'artifact_manifest.json','status.json','controller_heartbeat.json','resource_wait.json','worker_exit.json','worker.log','controller.log','archive_audit.json','postcloseout_archive_audit.json','final_review_closeout.json','closeout.json'}
 files=[]
 for p in sorted(ROOT.rglob('*')):
  if not p.is_file() or p.is_symlink() or p.name in excluded or p.suffix in ['.pyc','.tmp'] or 'cache' in p.relative_to(ROOT).parts or '__pycache__' in p.parts:continue
  files.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
 write(ROOT/'artifact_manifest.json',dict(protocol_id=plan['protocol_id'],files=files));subprocess.run([sys.executable,str(ROOT/'scripts/audit_study.py'),'--archive'],check=True,env=env)
 close=dict(success=True,exploratory_only=not (ROOT/'reports/confirmation_summary.json').exists(),protocol_id=plan['protocol_id'],additional_steps=plan['steps'],archive_manifest_sha256=sha(ROOT/'artifact_manifest.json'),next_action=read(ROOT/'reports/summary.json')['next_action'],time_unix=time.time())
 write(ROOT/'reports/final_review_closeout.json',close);write(ROOT/'closeout.json',close);subprocess.run([sys.executable,str(ROOT/'scripts/audit_study.py'),'--postcloseout'],check=True,env=env);status('completed',**close)

def run():
 lock=(ROOT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
 draft=read(ROOT/'config/methods.json');active=read(ROOT/'config/locked.json') if (ROOT/'config/locked.json').exists() else draft;verify(active);c=components(active)
 if not (ROOT/'config/locked.json').exists():preflight(c,draft)
 plan=lock_plan(draft);c['lambda']=plan['lambda_value']
 reg=read(ROOT/'worker_registration.json');reg.update(protocol_id=plan['protocol_id'],scientific_id=plan['scientific_id'],parent_protocol_id=draft['protocol_id']);write(ROOT/'worker_registration.json',reg)
 recovery_preflight(c,plan);train(c,plan);summarize(c,plan)
 if any(read(ROOT/'reports/summary.json')['gates'].values()):
  from confirmation import run_confirmation
  run_confirmation(c,plan)
 del c;gc.collect();torch.cuda.empty_cache();finish(plan)
if __name__=='__main__':
 try:prepare() if '--prepare' in sys.argv else run()
 except BudgetExpired as e:
  status('budget_expired',success=False,reason=str(e));write(ROOT/'closeout.json',dict(success=False,terminal=True,budget_expired=True,time_unix=time.time()))
 except Exception:
  value=dict(time_unix=time.time(),traceback=traceback.format_exc());write(ROOT/'failures'/f'worker_{time.time_ns()}.json',value);status('failed',**value);raise
