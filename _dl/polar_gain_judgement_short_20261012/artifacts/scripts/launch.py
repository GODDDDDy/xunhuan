"""CPU launcher checks for duplicate controllers and runs in its own process group."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
controller=str(ROOT/'scripts/short_controller.py')
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:command=(p/'cmdline').read_bytes().decode(errors='replace').replace('\x00',' ')
    except (OSError,PermissionError):continue
    if controller in command and 'launch.py' not in command and int(p.name)!=os.getpid():
        raise RuntimeError(f'Controller already running: {p.name}')
resume='--resume-preflight' in sys.argv
if (ROOT/'budget.json').exists() and not resume:
    raise RuntimeError('Budget already exists: inspect/recover using its immutable cutoff; do not relaunch blindly')
with (ROOT/'logs/controller.log').open('a') as log:
    proc=subprocess.Popen([sys.executable,'-u',controller]+(['--resume-preflight'] if resume else []),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                          start_new_session=True)
print(json.dumps(dict(controller_pid=proc.pid,process_group=proc.pid)),flush=True)
