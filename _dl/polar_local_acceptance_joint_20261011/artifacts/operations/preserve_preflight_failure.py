"""Preserve failed GPU preflight before an equivalent execution repair."""
import json, shutil, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
assert ROOT.name == 'polar_local_acceptance_joint_20261011'
budget = json.loads((ROOT/'budget.json').read_text())
assert budget['start_unix'] == 1791700972.3300269
assert budget['deadline_unix'] == 1791787372.3300269
assert time.time() < budget['deadline_unix']
assert not (ROOT/'config/locked_run.json').exists()
assert not (ROOT/'downstream/checkpoints/initial.pt').exists()
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try: command = (proc/'cmdline').read_bytes().decode(errors='replace')
    except OSError: continue
    if str(ROOT/'scripts/controller.py') in command or str(ROOT/'scripts/watchdog.py') in command:
        raise RuntimeError(f'Run still active: {proc.name}')
destination = ROOT/'diagnostics/gpu_preflight_attempt1'
assert not destination.exists()
destination.mkdir(parents=True)
for name in ('scripts', 'config'):
    shutil.copytree(ROOT/name, destination/name)
for name in ('budget.json','watchdog.json','status.json','diagnostics/cpu_checks.json'):
    path=ROOT/name
    if path.exists(): shutil.copy2(path,destination/path.name)
for name in ('logs/controller.log','logs/benchmark.log','logs/failure.log',
             'artifacts.tar.gz','archive_ready.json','archive_manifest.json','SUMMARY_ZH.md'):
    path=ROOT/name
    if path.exists(): shutil.move(str(path),str(destination/path.name))
(destination/'repair_note.json').write_text(json.dumps(dict(
    failure='Deterministic CUDA spatial NLL implementation unavailable during disposable preflight',
    repair='Equivalent weighted CE via log-softmax and one-hot reduction; source path set before mechanism import',
    formal_training_started=False,discarded_preflight_weights=True,
    immutable_budget=budget),indent=2))
print(json.dumps(dict(preserved=str(destination),budget_unchanged=True)))
