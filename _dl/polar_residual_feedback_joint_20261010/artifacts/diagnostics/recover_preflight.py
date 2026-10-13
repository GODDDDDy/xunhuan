import json,os,signal,time,shutil
from pathlib import Path
root=Path('/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/polar_residual_feedback_joint_20261010')
assert not (root/'config/locked_run.json').exists()
budget=json.loads((root/'budget.json').read_text());wd=json.loads((root/'watchdog.json').read_text())
group=budget['process_group'];pid=wd['pid']
assert str(root/'scripts/controller.py') in Path(f'/proc/{group}/cmdline').read_bytes().decode()
assert str(root/'scripts/watchdog.py') in Path(f'/proc/{pid}/cmdline').read_bytes().decode()
os.killpg(group,signal.SIGTERM);os.kill(pid,signal.SIGTERM)
time.sleep(1)
target=root/'diagnostics/preflight_attempt1';target.mkdir(parents=True,exist_ok=True)
for name in ('benchmark.log','controller.log','watchdog.log'):
    path=root/'logs'/name
    if path.exists():shutil.move(str(path),str(target/name))
(target/'failure.json').write_text(json.dumps(dict(reason='AF_UNIX temporary IPC path exceeds socket limit',
    algorithm_changed=False,formal_training_started=False,budget_preserved=budget,fix='shorter project-classified TMPDIR'),indent=2))
print(json.dumps(dict(stopped_group=group,stopped_watchdog=pid,deadline_unchanged=budget['deadline_unix'])))
