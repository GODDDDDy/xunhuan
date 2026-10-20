"""41: frozen40-B inference-only 2x2 pathway intervention, no optimizer.

F-off zeros only post-projection E-feature channels (32:48); the learned
adapter's RGB/current-task slots and all its weights remain unchanged.
"""
from __future__ import annotations
import argparse,fcntl,gc,hashlib,importlib.util,json,math,os,statistics,subprocess,sys,time,traceback
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent
OLD=PROJECT/'g6_matched_enhancement_feedback_20261019'
spec=importlib.util.spec_from_file_location('verified40',OLD/'scripts/run_study.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
read,write,sha,canonical,receipt=old.read,old.write,old.sha,old.canonical,old.receipt
DEVICE=torch.device('cuda:0');MODES={'P1F1':(True,True),'P0F1':(False,True),'P1F0':(True,False),'P0F0':(False,False)}
PAIRS=[('P1F1','P0F1'),('P1F0','P0F0'),('P1F1','P1F0'),('P0F1','P0F0'),('P1F1','P0F0')]
def status(stage,**kw):
 value=dict(stage=stage,time_unix=time.time());value.update(kw);write(ROOT/'status.json',value);print(json.dumps(value),flush=True)
def verify(plan):
 unsigned=dict(plan);unsigned.pop('protocol_id');assert canonical(unsigned)==plan['protocol_id']
 assert plan['experiment']==41 and plan['steps']==0 and plan['modes']==list(MODES)
 assert canonical(plan['scientific'])==plan['scientific_id']
 codes={r['path']:r for r in plan['code_sources']};replacements={};rp=ROOT/'config/implementation_revisions.json'
 for rev in read(rp) if rp.exists() else []:
  assert rev['protocol_id']==plan['protocol_id'] and rev['scientific_id']==plan['scientific_id']
  for r in rev['replacements']:
   assert r['path'] in codes and codes[r['path']]['sha256']==r['original_sha256'] and sha(r['preserved_original_path'])==r['original_sha256']
   replacements[r['path']]=r['repaired_sha256']
 for r in plan['sources']+plan['code_sources']:assert sha(r['path'])==replacements.get(r['path'],r['sha256']),r['path']
def prepare():
 for name in ['config','diagnostics','reports','metrics','logs','failures','cache']:(ROOT/name).mkdir(parents=True,exist_ok=True)
 lp=ROOT/'config/locked.json'
 if lp.exists():verify(read(lp));return
 assert not (ROOT/'worker_registration.json').exists()
 p40=read(OLD/'config/locked.json');old.verify(p40)
 close=read(OLD/'reports/closeout_status_review_20261019/POSTREVIEW_CLOSEOUT.json');assert close['success'] and close['scientific_evidence_valid']
 for r in read(OLD/'artifact_manifest.json')['files']:assert sha(OLD/r['path'])==r['sha256']
 ck=OLD/'weights/final.pt';metrics=OLD/'metrics/step_1000.json'
 scientific=dict(experiment=41,steps=0,source_experiment=40,source_arm='B',source_endpoint=1000,checkpoint_sha256=sha(ck),
  modes=MODES,P_off='use unchanged P0 only in task input bridge; keep E computation/features identical',
  F_off='zero only post-projection E slots32:48 at both decoder interfaces; preserve projection bias removal, current16/statezero16/RGB3/previouszero6/flagzero1',
  all_parameters='frozen',module_modes='task eval, E/feedback train as source40; no_grad',
  validation_ids=p40['validation_ids'],evaluation_seed=2026,phase='validation',index='original872 index',noise='same actual [1,16,256] across four modes',
  reference='P1F1 output SHA and PSNR/SSIM must exactly reproduce40 B every frame',test=0,
  summary='scene-equal conditional effects in both other-path contexts; report every negative scene and output differences',
  conclusion='functional dependence/immediate intervention only; distribution shift and coadaptation preclude replacing matched retraining ablation',
  stop='fixed64 and audit, no new training, no expanded mask/amplitude/seed/Oracle matrix or automatic successor')
 sources={r['path']:receipt(r['path'],r.get('role')) for r in p40['sources']+p40['code_sources']}
 for p,role in [(ROOT/'PROTOCOL_ZH.md','registered41 protocol'),(OLD/'config/locked.json','source40 plan'),(ck,'fixed40 B endpoint'),(metrics,'paired B/G6/A64 reference'),(OLD/'artifact_manifest.json','source40 archive'),(OLD/'reports/closeout_status_review_20261019/POSTREVIEW_CLOSEOUT.json','source40 supplemental closeout')]:sources[str(p)]=receipt(p,role)
 plan=dict(experiment=41,steps=0,modes=list(MODES),source_plan=str(OLD/'config/locked.json'),checkpoint=str(ck),reference_metrics=str(metrics),
  source_config=p40['source_config'],source_root=p40['source_root'],validation_ids=p40['validation_ids'],full_index_map=p40['full_index_map'],
  scientific=scientific,scientific_id=canonical(scientific),sources=list(sources.values()),code_sources=[receipt(p) for p in sorted((ROOT/'scripts').glob('*.py'))],deadline_unix=None)
 plan['protocol_id']=canonical(plan);write(lp,plan);verify(plan);print(json.dumps(dict(prepared=True,protocol_id=plan['protocol_id'],scientific_id=plan['scientific_id'])),flush=True)
def components(plan):
 sys.path.insert(0,plan['source_root'])
 from rgb_final_confirmation.config import ExperimentConfig
 from rgb_final_confirmation.polarfree import OfficialPolarFreeTask
 cfg=yaml.safe_load(Path(plan['source_config']).read_text());raw=yaml.safe_load(Path(cfg['paths']['base_config']).read_text())
 raw['experiment_root']=str(ROOT);raw['paths']['outputs']=str(ROOT);raw['paths']['weight_audit']=str(ROOT/'config/weight_audit.json')
 conf=ExperimentConfig(raw=raw,source=Path(cfg['paths']['base_config']))
 task=OfficialPolarFreeTask(conf,DEVICE).eval().requires_grad_(False)
 block=old.arch.EnhancementBlock(old.arch.PolarEnhancer()).to(DEVICE)
 ck=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False);assert ck['step']==1000
 task.net_g.load_state_dict(ck['weights']['B'],strict=True);block.load_state_dict(ck['blocks']['B'],strict=True);del ck
 block.train().requires_grad_(False)
 return dict(cfg=cfg,task=task,block=block)
def fingerprint(c):return dict(task=old.arch.fingerprint(c['task']),block=old.arch.fingerprint(c['block']))
@torch.no_grad()
def run_frame(c,plan,sid,parity=False):
 from rgb_final_confirmation.evaluation import psnr,ssim
 batch=old.h.full_item(c,sid);old.seed_step(2026,'validation',plan['full_index_map'][sid],[sid]);noise=torch.randn(1,16,256,device=DEVICE)
 block,task=c['block'],c['task'];p,features,_,_=block.enhancer(batch['polar'],batch['polar'],batch['official_rgb'],None,None,None)
 inputs={True:old.arch.anchored_task_inputs(p,batch['polar'],batch['official_rgb']),False:old.arch.anchored_task_inputs(batch['polar'],batch['polar'],batch['official_rgb'])}
 input_hash={str(k):[old.h.tensor_sha(t) for t in v] for k,v in inputs.items()}
 context=dict(image=batch['official_rgb'],available=False,t1=None,features=None)
 metrics={};hashes={};slot_proof={};noise_calls=[];predictions={}
 for mode,(use_p,use_f) in MODES.items():
  hooks=[]
  try:
   if not use_f:
    for layer in block.feedback.polar_projection.values():hooks.append(layer.register_forward_hook(lambda m,args,out:torch.zeros_like(out)))
   for level in ['level3','level2']:
    def hook(module,args,out,level=level):return block.feedback.fuse(out,level,context,features)
    hooks.append(getattr(task.net_g,'decoder_'+level).register_forward_hook(hook))
   with old.h.NoiseCheck({mode:task},noise) as proof:pred,unused=task.forward_inputs(inputs[use_p],noise)
   del unused
   slot_proof[mode]={level:dict(E_slot_max=float(v[:,32:48].abs().max()),previous_slots_zero=bool((v[:,16:32]==0).all() and (v[:,51:58]==0).all())) for level,v in block.feedback.last_fuse_values.items()}
   if not use_f:assert all(x['E_slot_max']==0 for x in slot_proof[mode].values())
   assert all(x['previous_slots_zero'] for x in slot_proof[mode].values())
   metrics[mode]=dict(psnr=psnr(pred,batch['target']),ssim=ssim(pred,batch['target']));hashes[mode]=old.h.tensor_sha(pred);noise_calls.extend(proof.calls);predictions[mode]=pred.detach()
  finally:
   for hook in hooks:hook.remove()
 reference=next(r for r in read(plan['reference_metrics'])['rows'] if r['sample_id']==sid)
 assert hashes['P1F1']==reference['output_sha256']['B'] and metrics['P1F1']==reference['metrics']['B'],(sid,'source40 B parity')
 if parity:
  # Execute source40's original untouched B forward, not only this implementation.
  fake=dict(blocks={'B':block},tasks={'B':task})
  native=old.predict(fake,'B',batch,None,noise)
  assert torch.equal(native,predictions['P1F1']);del native
 difference={}
 for mode,t in predictions.items():
  delta=t-predictions['P1F1'];difference[mode]=dict(max_abs=float(delta.abs().max()),mean_abs=float(delta.abs().mean()),rms=float(delta.square().mean().sqrt()),exact_equal=bool(torch.equal(t,predictions['P1F1'])))
 delta=p-batch['polar']
 result=dict(sample_id=sid,scene=batch['scene'][0],full_index=plan['full_index_map'][sid],shape=list(batch['target'].shape),original_size=True,
  metrics=metrics,output_sha256=hashes,noise_calls=noise_calls,input_sha256=input_hash,slot_proof=slot_proof,output_difference_from_original=difference,
  polar_delta=dict(mean_abs=float(delta.abs().mean()),max_abs=float(delta.abs().max())),source40_B_exact=True,native40_forward_exact=parity,
  reference_metrics={k:reference['metrics'][k] for k in ['G6','A']})
 del predictions,batch,p,features,inputs;gc.collect();torch.cuda.empty_cache();return result
def pair(rows,a,b):
 out={}
 for m in ['psnr','ssim']:
  scenes=sorted({r['scene'] for r in rows});per={s:sum(r['metrics'][a][m]-r['metrics'][b][m] for r in rows if r['scene']==s)/sum(r['scene']==s for r in rows) for s in scenes}
  out[m]=dict(scene_equal=sum(per.values())/len(per),scenes=per,positive_scenes=sum(v>0 for v in per.values()))
 return out
def evaluate(plan):
 if (ROOT/'reports/summary.json').exists():return
 c=components(plan);before=fingerprint(c);torch.cuda.reset_peak_memory_stats()
 from PIL import Image
 selected=[r for r in old.h.rows(c) if r.sample_id in plan['validation_ids']]
 largest=max(selected,key=lambda r:Image.open(r.official_rgb).width*Image.open(r.official_rgb).height).sample_id
 fp=ROOT/'config/preflight.json'
 if not fp.exists():
  status('preflight_largest_full_frame');tick=time.time();row=run_frame(c,plan,largest,parity=True)
  assert before==fingerprint(c)
  write(ROOT/'diagnostics/preflight_frame.json',row)
  write(fp,dict(passed=True,protocol_id=plan['protocol_id'],largest_id=largest,native40_forward_exact=True,source40_B_exact=True,
   all_weights_frozen=True,F_mask_post_projection=True,previous_slots_absent=True,no_optimizer=True,no_training=True,
   frame_seconds=time.time()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
 path=ROOT/'metrics/validation64.json';rows=read(path)['rows'] if path.exists() else []
 assert [r['sample_id'] for r in rows]==plan['validation_ids'][:len(rows)]
 for sid in plan['validation_ids'][len(rows):]:
  row=run_frame(c,plan,sid);rows.append(row);write(path,dict(protocol_id=plan['protocol_id'],frames=len(rows),conditions=4,rows=rows));status('evaluating',completed=len(rows),total=64)
 assert before==fingerprint(c) and all(not p.requires_grad and p.grad is None for model in [c['task'],c['block']] for p in model.parameters())
 write(ROOT/'diagnostics/frozen_final.json',dict(passed=True,before=before,after=fingerprint(c),no_training=True))
 comparisons={a+'-'+b:pair(rows,a,b) for a,b in PAIRS}
 write(ROOT/'reports/summary.json',dict(protocol_id=plan['protocol_id'],training_steps=0,validation=64,test=0,conditions=4,
  comparisons=comparisons,exact_output_invariance={mode:all(r['output_difference_from_original'][mode]['exact_equal'] for r in rows) for mode in MODES},
  means={mode:{m:statistics.mean(r['metrics'][mode][m] for r in rows) for m in ['psnr','ssim']} for mode in MODES},
  limitations=['frozen_weight_intervention','possible_distribution_shift','not_matched_retraining_ablation','not_C_or_R_feedback_quality_test'],next_action='stop_diagnostic_no_automatic_training'))
 del c;gc.collect();torch.cuda.empty_cache()
def independent_audit(plan):
 verify(plan);assert not torch.cuda.is_available();pre=read(ROOT/'config/preflight.json');assert pre['passed'] and pre['no_optimizer'] and pre['no_training'] and pre['native40_forward_exact']
 rows=read(ROOT/'metrics/validation64.json')['rows'];assert [r['sample_id'] for r in rows]==plan['validation_ids'] and len(rows)==64
 reference={r['sample_id']:r for r in read(plan['reference_metrics'])['rows']};calc={}
 for r in rows:
  ref=reference[r['sample_id']];assert r['source40_B_exact'] and r['original_size'] and r['full_index']==ref['full_index'] and r['shape']==ref['shape'] and r['scene']==ref['scene']
  assert r['metrics']['P1F1']==ref['metrics']['B'] and r['output_sha256']['P1F1']==ref['output_sha256']['B']
  assert [x['arm'] for x in r['noise_calls']]==list(MODES) and len({x['sha256'] for x in r['noise_calls']})==1 and all(x['shape']==[1,16,256] for x in r['noise_calls'])
  for mode in MODES:
   assert set(r['slot_proof'][mode])=={'level2','level3'}
   assert all(x['previous_slots_zero'] and (MODES[mode][1] or x['E_slot_max']==0) for x in r['slot_proof'][mode].values())
   assert all(math.isfinite(x) for x in r['metrics'][mode].values())
 summary=read(ROOT/'reports/summary.json')
 for a,b in PAIRS:
  calc[a+'-'+b]={}
  for m in ['psnr','ssim']:
   scenes=sorted({r['scene'] for r in rows});by={s:statistics.mean(r['metrics'][a][m]-r['metrics'][b][m] for r in rows if r['scene']==s) for s in scenes}
   calc[a+'-'+b][m]=dict(scene_equal=statistics.mean(by.values()),scenes=by,positive_scenes=sum(v>0 for v in by.values()))
   actual=summary['comparisons'][a+'-'+b][m];assert abs(actual['scene_equal']-calc[a+'-'+b][m]['scene_equal'])<1e-10 and actual['positive_scenes']==calc[a+'-'+b][m]['positive_scenes']
   assert all(abs(actual['scenes'][s]-by[s])<1e-10 for s in scenes)
 f=read(ROOT/'diagnostics/frozen_final.json');assert f['passed'] and f['before']==f['after'] and f['no_training']
 assert summary['training_steps']==0 and summary['next_action']=='stop_diagnostic_no_automatic_training'
 return dict(passed=True,cpu_only=True,protocol_id=plan['protocol_id'],training_steps=0,validation=64,test=0,conditions=4,comparisons=calc,
  negative_scenes={k:{m:{s:v for s,v in d[m]['scenes'].items() if v<0} for m in ['psnr','ssim']} for k,d in calc.items()},
  scope='independent stored-evidence and arithmetic audit, not independent network inference',time_unix=time.time())
def finish(plan):
 status('independent_audit');env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
 subprocess.run([sys.executable,str(ROOT/'scripts/run_study.py'),'--audit'],env=env,check=True);verify(plan)
 excluded={'artifact_manifest.json','status.json','closeout.json','final_review_closeout.json','archive_audit.json','controller_heartbeat.json','resource_wait.json','worker_exit.json','worker.log','controller.log'}
 records=[]
 for p in sorted(ROOT.rglob('*')):
  if p.is_symlink() or not p.is_file() or p.name in excluded or p.suffix in ['.pyc','.tmp'] or 'cache' in p.relative_to(ROOT).parts or '__pycache__' in p.parts:continue
  records.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
 write(ROOT/'artifact_manifest.json',dict(protocol_id=plan['protocol_id'],files=records))
 for r in records:
  p=ROOT/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
 write(ROOT/'reports/archive_audit.json',dict(passed=True,files=len(records),manifest_sha256=sha(ROOT/'artifact_manifest.json')))
 close=dict(success=True,protocol_id=plan['protocol_id'],training_steps=0,validation=64,test=0,conditions=4,diagnostic_only=True,
  source_frozen=True,independent_audit_verified=True,archive_verified=True,archive_manifest_sha256=sha(ROOT/'artifact_manifest.json'),next_action='stop_diagnostic_no_automatic_training',time_unix=time.time())
 write(ROOT/'reports/final_review_closeout.json',close);write(ROOT/'closeout.json',close);status('completed',**close)
 for r in records:assert sha(ROOT/r['path'])==r['sha256']
def main():
 plan=read(ROOT/'config/locked.json');verify(plan)
 lock=(ROOT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (ROOT/'closeout.json').exists():assert read(ROOT/'closeout.json')['success'];return
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
 evaluate(plan);finish(plan)
if __name__=='__main__':
 try:
  if '--prepare' in sys.argv:prepare()
  elif '--audit' in sys.argv:write(ROOT/'reports/independent_audit.json',independent_audit(read(ROOT/'config/locked.json')));print('CPU audit passed',flush=True)
  else:main()
 except Exception:
  error=dict(time_unix=time.time(),traceback=traceback.format_exc());write(ROOT/'failures'/f'worker_{time.time_ns()}.json',error);status('failed',**error);raise
