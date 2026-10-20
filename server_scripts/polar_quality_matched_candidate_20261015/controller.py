"""CPU coordinator and independent watchdog; immutable fresh 12-hour budget."""
import csv,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,ensure_ascii=False));q.replace(p)
def watch():
    b=json.loads((ROOT/'budget.json').read_text());group=b['controller_pgid']
    while True:
        now=time.time();write(ROOT/'watchdog.json',dict(pid=os.getpid(),pgid=os.getpgrp(),controller_pgid=group,deadline_unix=b['deadline_unix'],heartbeat_unix=now))
        if now>=b['deadline_unix']:
            write(ROOT/'status.json',dict(stage='deadline_reached',gpu_prohibited=True,time_unix=now))
            try:os.killpg(group,signal.SIGTERM)
            except ProcessLookupError:return
            time.sleep(5)
            try:os.killpg(group,signal.SIGKILL)
            except ProcessLookupError:pass
            return
        try:os.killpg(group,0)
        except ProcessLookupError:return
        time.sleep(5)
def main():
    assert json.loads((ROOT/'diagnostics/cpu_checks.json').read_text())['passed']
    plan=json.loads((ROOT/'config/plan.json').read_text());now=time.time()
    if (ROOT/'budget.json').exists():
        assert '--resume-execution' in sys.argv,'Never reset budget'
        previous=json.loads((ROOT/'budget.json').read_text());start=previous['start_unix'];end=min(previous['deadline_unix'],start+plan['budget_seconds'])
    else:start=now;end=start+plan['budget_seconds']
    assert now<end-plan['evaluation_reserve_seconds']
    budget=dict(start_unix=start,deadline_unix=end,train_deadline_unix=end-plan['evaluation_reserve_seconds'],immutable=True,protocol_id=plan['protocol_id'],controller_pid=os.getpid(),controller_pgid=os.getpgrp(),hours=plan['budget_seconds']/3600,resumed_execution='--resume-execution' in sys.argv)
    write(ROOT/'budget.json',budget);cpuenv=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    wlog=(ROOT/'logs/watchdog.log').open('a');watcher=subprocess.Popen([sys.executable,__file__,'--watchdog'],stdout=wlog,stderr=subprocess.STDOUT,env=cpuenv,start_new_session=True)
    write(ROOT/'watchdog_registration.json',dict(pid=watcher.pid,pgid=watcher.pid,controller_pgid=os.getpgrp(),deadline_unix=budget['deadline_unix']))
    success=False
    try:
        stages=[('preflight',['preflight']),('real_train',['train','--arm','real_quality']),('fixed_train',['train','--arm','fixed_quality']),('validation',['evaluate','--split','validation']),('test',['evaluate','--split','test'])]
        for label,args in stages:
            assert time.time()<budget['deadline_unix']
            if label=='validation':
                arms=[list(csv.DictReader((ROOT/'arms'/a/'train.csv').open())) for a in plan['arms']]
                assert len(arms[0])==len(arms[1])==2000
                assert [(r['step'],r['sample_ids']) for r in arms[0]]==[(r['step'],r['sample_ids']) for r in arms[1]]
                write(ROOT/'diagnostics/matched_data_receipt.json',dict(steps_per_arm=2000,paired_sample_ids_exact=True,candidate_images_per_arm=3640,test_used=False))
            log=(ROOT/'logs'/f'{label}.log').open('a');p=subprocess.run([sys.executable,str(ROOT/'scripts/matched_run.py')]+args,stdout=log,stderr=subprocess.STDOUT)
            if p.returncode:raise RuntimeError(label+' failed '+str(p.returncode))
        subprocess.run([sys.executable,str(ROOT/'scripts/summarize.py')],env=cpuenv,check=True)
        write(ROOT/'status.json',dict(stage='completed',time_unix=time.time(),protocol_id=plan['protocol_id'],steps_per_arm=2000,validation_frames=872,test_frames=188));success=True
    except Exception:
        write(ROOT/'diagnostics/failure.json',dict(time_unix=time.time(),traceback=traceback.format_exc()));write(ROOT/'status.json',dict(stage='failed',time_unix=time.time(),error=traceback.format_exc()));print(traceback.format_exc(),flush=True)
    finally:
        if watcher.poll() is None:watcher.terminate();watcher.wait(timeout=15)
        write(ROOT/'watchdog_exit.json',dict(pid=watcher.pid,stopped=True,deadline_unix=budget['deadline_unix'],time_unix=time.time()))
    subprocess.run([sys.executable,str(ROOT/'scripts/archive.py')],env=cpuenv,check=True)
if __name__=='__main__':
    if '--watchdog' in sys.argv:watch()
    else:main()
