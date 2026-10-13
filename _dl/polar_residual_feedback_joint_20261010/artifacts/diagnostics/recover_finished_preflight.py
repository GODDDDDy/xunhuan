import json,os,signal,shutil,time
from pathlib import Path
root=Path('/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/polar_residual_feedback_joint_20261010')
assert not (root/'config/locked_run.json').exists()
assert json.loads((root/'status.json').read_text())['stage']=='failed'
budget=json.loads((root/'budget.json').read_text())
assert time.time()<budget['deadline_unix']
wd=json.loads((root/'watchdog.json').read_text())
for pid,expected in ((budget['process_group'],'controller.py'),(wd['pid'],'watchdog.py')):
    path=Path(f'/proc/{pid}/cmdline')
    if not path.exists():continue
    command=path.read_bytes().decode(errors='replace')
    if not command:continue
    assert str(root/'scripts'/expected) in command
    if expected=='controller.py':os.killpg(pid,signal.SIGTERM)
    else:os.kill(pid,signal.SIGTERM)
target=root/'diagnostics/preflight_attempt2';target.mkdir(parents=True,exist_ok=True)
for relative in ('logs/benchmark.log','logs/controller.log','logs/watchdog.log','logs/failure.log',
                 'artifacts.tar.gz','archive_ready.json','archive_manifest.json','RESULTS_ZH.md','status.json'):
    path=root/relative
    if path.exists():shutil.move(str(path),str(target/path.name))
(target/'fix.json').write_text(json.dumps(dict(reason='Enhancer AMP outputs BF16; new feedback projections expect FP32',
    fix='Cast enhancer features to FP32 at feedback boundary',formal_training_started=False,
    architecture_changed=False,budget_preserved=budget),indent=2))
print(json.dumps(dict(recovered=True,deadline_unchanged=budget['deadline_unix'])))
