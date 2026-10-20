"""Evaluation-only recovery; resource wait has no expiry by explicit user request."""
import csv,fcntl,gc,hashlib,json,math,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.parent.mkdir(parents=True,exist_ok=True)
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False));tmp.replace(path)

def partial(label):
    p=ROOT/'metrics'/f'{label}_partial.json'
    return json.loads(p.read_text())['rows'] if p.exists() else []

def evaluation_guard():
    """Apply the latest runtime authorization without editing frozen source code."""
    policy=json.loads((ROOT/'config/evaluation_runtime_policy.json').read_text())
    assert policy['evaluation_only'] is True
    if policy['time_budget_enforced'] is False:return
    budget=json.loads((ROOT/'budget.json').read_text())
    deadline=budget.get('deadline_unix')
    if deadline is None or time.time()>=deadline:
        raise RuntimeError('Evaluation runtime authorization has expired')

def worker():
    import torch
    import common
    # The unchanged evaluator imports guard_gpu by name. Bind the user-updated
    # scheduling policy before importing it; model/evaluation math stays frozen.
    common.guard_gpu=evaluation_guard
    import c_loop_run as run
    from common import sha,status
    run.guard_gpu=evaluation_guard
    assert run.study.guard_gpu is evaluation_guard
    torch.set_num_threads(2);evaluation_guard()
    cfg,plan,model,task,raw,source,arms=run.load(torch.device('cuda:0'))
    complete=json.loads((ROOT/'diagnostics/training_complete.json').read_text())
    final=ROOT/'weights/final.pt'
    assert sha(final)==complete['endpoint_sha256']
    ck=torch.load(final,map_location='cpu',weights_only=False)
    assert ck['step']==6000 and ck['protocol_id']==plan['protocol_id']
    assert ck['source_initial_sha256']==plan['source_initial_sha256']
    for a,e in arms.items():
        state=e.state_dict();state.update(ck['arms'][a]);e.load_state_dict(state,strict=True)
    run.audit(model,task,source,arms);del ck;gc.collect();torch.cuda.empty_cache()
    write(ROOT/'diagnostics/evaluation_recovery_load.json',dict(
        original_protocol=plan['protocol_id'],endpoint_sha256=sha(final),step=6000,
        training_restarted=False,source_frozen_exact=True,
        recovery_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        deadline_unix=None,time_budget_enforced=False,time_unix=time.time()))
    from rgb_final_confirmation.data import load_manifest,select_rows
    selected=json.loads((ROOT/'config/validation256.json').read_text())['sample_ids']
    saved=partial('validation256')
    assert len({r['sample_id'] for r in saved})==len(saved)
    assert set(r['sample_id'] for r in saved)<=set(selected)
    validation=run.evaluate(cfg,plan,model,task,arms,'validation256',selected,saved)
    reasons=run.need_full(validation['comparisons'])
    write(ROOT/'metrics/validation_precision_decision.json',dict(reasons=reasons,
          complete_validation_required=bool(reasons),test_read_for_decision=False))
    if reasons:
        ids=[r.sample_id for r in select_rows(load_manifest(cfg['paths']['manifest']),'validation')]
        prior=partial('validation872') or validation['rows']
        validation=run.evaluate(cfg,plan,model,task,arms,'validation872',ids,prior)
    ids=[r.sample_id for r in select_rows(load_manifest(cfg['paths']['manifest']),'test')]
    test=run.evaluate(cfg,plan,model,task,arms,'test188',ids,partial('test188'))
    run.audit(model,task,source,arms)
    write(ROOT/'reports/final_summary.json',dict(protocol_id=plan['protocol_id'],steps_per_arm=6000,
          validation_frames=validation['frames'],test_frames=188,validation=validation['comparisons'],
          test=test['comparisons'],source_frozen_exact=True,endpoint_sha256=sha(final),
          validation_expansion_reasons=reasons,test_used_for_selection=False,
          evaluation_only_recovery=True,training_restarted=False,time_budget_enforced=False))
    status('completed',validation=validation['frames'],test=188,steps_per_arm=6000)

def closeout(code):
    files=[]
    for folder in ('config','scripts','weights','metrics','examples','diagnostics','reports','logs'):
        for p in sorted((ROOT/folder).rglob('*')):
            if not p.is_file() or '__pycache__' in str(p):continue
            h=hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
            files.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=h.hexdigest()))
    write(ROOT/'artifact_manifest.json',dict(files=files,server_only=True))
    write(ROOT/'closeout.json',dict(process_returncode=code,completed=code==0,
          files_hashed=len(files),evaluation_only_recovery=True,server_only=True,time_unix=time.time()))

def controller():
    # Keep a single monitor even if a disconnected client repeats the launch.
    ownership=(ROOT/'evaluation_controller.lock').open('a+')
    fcntl.flock(ownership.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    budget=json.loads((ROOT/'budget.json').read_text())
    policy=json.loads((ROOT/'config/evaluation_runtime_policy.json').read_text())
    assert policy['time_budget_enforced'] is False and policy['evaluation_only'] is True
    try:os.kill(budget['controller_pid'],0)
    except ProcessLookupError:pass
    else:
        stat=Path('/proc')/str(budget['controller_pid'])/'stat'
        state=stat.read_text().rsplit(')',1)[1].split()[0] if stat.exists() else 'Z'
        if state!='Z':raise RuntimeError('Previous owned controller still alive')
    budget.update(controller_pid=os.getpid(),controller_pgid=os.getpgrp(),evaluation_only_recovery=True,
                  time_budget_enforced=False,deadline_unix=None)
    write(ROOT/'budget.json',budget)
    # Retire the old deadline killer rather than replacing its date with an
    # arbitrary future timestamp. The monitor exits when evaluation completes.
    write(ROOT/'watchdog_registration.json',dict(pid=None,enabled=False,
          controller_pgid=os.getpgrp(),deadline_unix=None,evaluation_only_recovery=True,
          reason='User explicitly cancelled the time budget'))
    write(ROOT/'evaluation_queue_registration.json',dict(controller_pid=os.getpid(),
          original_start_unix=budget['start_unix'],deadline_unix=None,time_budget_enforced=False,
          conservative_latest_start_unix=None,minimum_free_MiB=11500,poll_interval_seconds=2,
          training_restarted=False,time_unix=time.time()))
    code=1
    attempt=max([int(p.stem.rsplit('_',1)[1]) for p in (ROOT/'logs').glob('eval_recovery_[0-9]*.log')],default=0)
    try:
        while True:
            try:
                query=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10)
                free=int(query.strip().splitlines()[0])
            except (subprocess.SubprocessError,OSError,ValueError):
                write(ROOT/'status.json',dict(stage='waiting_for_gpu_query_retry',training_complete=True,
                      steps_per_arm=6000,poll_interval_seconds=2,monitor_pid=os.getpid(),time_unix=time.time()))
                time.sleep(2);continue
            if free<11500:
                write(ROOT/'status.json',dict(stage='waiting_for_gpu_for_evaluation',training_complete=True,
                      steps_per_arm=6000,free_MiB=free,required_free_MiB=11500,
                      poll_interval_seconds=2,monitor_pid=os.getpid(),
                      deadline_unix=None,latest_conservative_start_unix=None,time_budget_enforced=False,
                      time_unix=time.time()))
                time.sleep(2);continue
            attempt+=1
            write(ROOT/'status.json',dict(stage='evaluation_recovery_starting',training_complete=True,
                  steps_per_arm=6000,attempt=attempt,time_unix=time.time()))
            with (ROOT/'logs'/f'eval_recovery_{attempt}.log').open('a') as log:
                code=subprocess.call([sys.executable,__file__,'worker'],stdout=log,stderr=subprocess.STDOUT)
            if code==0:break
            error=(ROOT/'logs'/f'eval_recovery_{attempt}.log').read_text()
            if 'out of memory' not in error.lower():raise RuntimeError('Evaluation recovery failed; inspect owned log')
            # A competing process may start after the free-memory check. The
            # failed worker releases its context; retain completed rows and wait.
            time.sleep(2)
    except Exception:
        write(ROOT/'diagnostics'/f'eval_recovery_failure_{int(time.time())}.json',dict(traceback=traceback.format_exc(),time_unix=time.time()))
        raise
    finally:
        write(ROOT/'evaluation_monitor_exit.json',dict(controller_pid=os.getpid(),
              time_budget_enforced=False,time_unix=time.time()))
    closeout(code)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker':
        try:worker()
        except Exception:
            write(ROOT/'diagnostics'/f'eval_worker_failure_{int(time.time())}.json',dict(traceback=traceback.format_exc(),time_unix=time.time()))
            raise
    else:controller()
