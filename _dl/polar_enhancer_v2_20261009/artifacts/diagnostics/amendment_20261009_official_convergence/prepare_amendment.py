"""Record an explicitly authorized pre-test amendment; never reset GPU timing."""
import hashlib,json,math,time
from pathlib import Path,PurePosixPath
root=Path(__file__).resolve().parents[2]
audit=Path(__file__).resolve().parent
cfg=json.loads((audit/'original_locked_run.json').read_text())
lock=json.loads((audit/'original_protocol_lock.json').read_text())
interruption=json.loads((audit/'interruption.json').read_text())
official={'net_g_latest.pth':'a230e77fb19827a1ada41e4dc86374dee092e476915d2252841f81dd8c257ca2',
          'net_d_latest.pth':'f1d0e9acb244480e5d4ba4b5cee7ba35666fb97d41bccf4659f20911f46f1fb3',
          'net_le_dm_latest.pth':'6b98bd2a715d3e669acd99582d9ea37594ff4a2ee7786df86764d8efdafd0546'}
remote=PurePosixPath(cfg['paths']['source_checkpoint']).parents[3]
official_dir=remote/'weights/polarfree'
old_source=cfg['paths']['source_checkpoint'];old_source_hash=cfg['source_sha256']
cfg['paths']['source_checkpoint']=str(official_dir/'net_g_latest.pth')
cfg['source_sha256']=official['net_g_latest.pth']
cfg['task_initialization']={'kind':'official_polarfree','official_sha256':official,'overwrite_with_joint_weights':False,
                          'archived_source_dir':str(remote/'polar_enhancer_v2_20261009/weights/source_polarfree')}
parent=cfg.pop('protocol_id')
start=interruption['checkpoint_step'];original_total=cfg['pretrain']['steps'];warm=min(500,original_total//10)
start_lr=1e-6+(3e-4-1e-6)*.5*(1+math.cos(math.pi*(start-warm)/(original_total-warm)))
cfg['pretrain'].update(steps=20000,minimum_steps=16000,
    continuation={'start_step':start,'start_lr':start_lr,'original_total_steps':original_total,
                  'parent_protocol_id':parent,'checkpoint_sha256':interruption['checkpoint_sha256'],
                  'immutable_resume_source':str(remote/'polar_enhancer_v2_20261009/diagnostics/amendment_20261009_official_convergence/checkpoint_step_8000.pt')},
    convergence={'minimum_steps':16000,'window':4,'psnr_span_db':.15,'psnr_slope_db_per_1000':.025,
                 'recovery_loss_relative_span':.03,'selection':'stability heuristic on validation only'})
cfg['optimization']={'correction_lr':2e-5,'interfaces_lr':1e-4,'task_lr':1e-6,'task_freeze_steps':500,
                     'task_unfreeze_warmup':200,'gradient_clip':1.}
cfg['retention_monitor']={'validation_proxy_frames':28,'interval':1000,'after_freeze_step':500,
                          'enhancer_bypassed':True,'test_used':False,'used_for_selection':False,
                          'paired_noise':True,'purpose':'monitor net_g performance retention and relative parameter drift'}
cfg['amendment']={'time_unix':time.time(),'authorization':'User requested sufficient correction pretraining and direct official PolarFree initialization with separate learning rates',
                  'parent_protocol_id':parent,'test_seen_before_amendment':False,
                  'original_task_source':old_source,'original_task_source_sha256':old_source_hash,
                  'discarded_unsaved_updates':interruption['discarded_unsaved_updates'],
                  'hard_deadline_unchanged':cfg['budget']['deadline_unix'],
                  'scope':'one latest arm; no additional test-dependent runs or controls'}
code={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in lock['code_sha256']}
cfg['code_sha256']=code
cfg['protocol_id']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
lock['code_sha256']=code;lock['amendment']=cfg['amendment'];lock['active_protocol_id']=cfg['protocol_id']
for name,digest in official.items():lock['sources_sha256'][str(official_dir/name)]=digest
lock['sources_sha256'][cfg['pretrain']['continuation']['immutable_resume_source']]=interruption['checkpoint_sha256']
for path,obj in ((root/'config/locked_run.json',cfg),(root/'protocol_lock.json',lock),(audit/'amended_configuration.json',cfg)):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
assert cfg['budget']['start_unix']==1791518647.4916465 and cfg['budget']['deadline_unix']==1791605047.4916465
assert str(official_dir)=='/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/weights/polarfree'
print(json.dumps({'protocol_id':cfg['protocol_id'],'start_lr':start_lr,'pretrain_minimum':16000,'pretrain_maximum':20000,'task_source':cfg['paths']['source_checkpoint'],'deadline_unix':cfg['budget']['deadline_unix']}))
