from pathlib import Path
p=Path(r'F:/WORK/projects/pidsr_recurrent_joint_k2/analysis/polarapp_postprocess_value_20261019/scripts/run_study.py')
s=p.read_text(encoding='utf-8')
s=s.replace('6000','12000').replace('3000','6000')
s=s.replace("old=read(PROJECT/'g6_adam_state_diagnostic_20261019/config/locked.json')", "old=read(PROJECT/'g6_adam_state_diagnostic_20261019/config/locked.json')")
s=s.replace("schedule=build_training_schedule(train,seed=2027,total_steps=6000,batch_size=2,crop_size=128)", "previous=read(ROOT/'history/user_requested_budget_v1/config/training3000.json')\n extension=build_training_schedule(train,seed=2028,total_steps=3000,batch_size=2,crop_size=128)\n schedule=dict(schema_version=2,seed=2027,extension_seed=2028,total_steps=6000,batch_size=2,crop_size=128,items=previous['items']+extension['items'])\n schedule['sha256']=canonical(schedule);assert len(schedule['items'])==12000 and schedule['items'][:6000]==previous['items']")
s=s.replace("evaluation=dict(probe1000=probe,endpoint6000=ids,test=0)", "evaluation=dict(checkpoints=[0,1000,2000,3000,4500,6000],fixed_val_probe=probe,fixed_train_indices=list(range(16)),endpoint6000=ids,test=0)")
s=s.replace("worker_seconds=86400", "worker_seconds=86400,user_amended_budget=True")
s=s.replace("def prepare_cache(c,plan):", "def prepare_cache(c,plan,warmup_only=False):")
s=s.replace("tick=time.time();completed=0\n for sample_index", "if len(index['entries'])<12000:\n  assert len(index['entries'])==6000;index['entries'] += [None]*6000\n tick=time.time();completed=0\n for sample_index")
s=s.replace("guard();row=train[sample_index];completed+=1", "guard();row=train[sample_index];completed+=1\n  if warmup_only and completed>8:\n   write(ip,index);return")
s=s.replace("saved=rng();rows=read(path)['rows'] if path.exists() else [];assert", "saved=rng();e_mode=c['E'].training;c['E'].eval();rows=read(path)['rows'] if path.exists() else [];assert")
s=s.replace("finally:set_rng(saved)\n return rows", "finally:set_rng(saved);c['E'].train(e_mode)\n return rows")
s=s.replace("6000*np.mean(times[1:])+80*frame", "6000*np.mean(times[1:])+160*frame+96*np.mean(times[1:])")
s=s.replace("status('training',step=start,total=6000)", "online_probes(c,plan,start) if start in CHECKPOINTS else None\n status('training',step=start,total=6000)")
s=s.replace("if step in [1000,6000]:save", "if step in CHECKPOINTS:save")
s=s.replace("if step%100==0 or step==1:", "if step%100==0 or step==1 or step in CHECKPOINTS:")
s=s.replace("write(ROOT/'diagnostics'/f'noise_{step:04d}.json',proof);del v", "write(ROOT/'diagnostics'/f'noise_{step:04d}.json',proof);del v\n   if step in CHECKPOINTS:online_probes(c,plan,step)")
a=s.index('def evaluate(c,plan):');b=s.index('\ndef finish(plan):',a)
s=s[:a]+'''CHECKPOINTS=[0,1000,2000,3000,4500,6000]

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
''' + s[b:]
s=s.replace("validation=64,probe1000=16", "validation=64,probe_checkpoints=CHECKPOINTS,probe_frames=16")
s=s.replace("preflight_before_cache(c,plan);prepare_cache(c,plan);preflight_training(c,plan)", "preflight_before_cache(c,plan);prepare_cache(c,plan,warmup_only=True);preflight_training(c,plan);prepare_cache(c,plan)")
p.write_text(s,encoding='utf-8')