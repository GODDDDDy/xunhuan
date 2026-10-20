"""Resume evaluation under the original immutable deadline, never reset it."""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
budget=json.loads((root/'budget.json').read_text())
remaining=int(budget['deadline_unix']-time.time())
assert remaining>0
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
    try:args=(entry/'cmdline').read_bytes().split(b'\0')
    except (OSError,PermissionError):continue
    if any(x==str(root/'scripts/h_run.py').encode() or x==str(root/'scripts/h_eval_recover.py').encode() for x in args):
        raise RuntimeError('Existing experiment process: '+entry.name)
env=dict(os.environ,PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
with (root/'logs/evaluation_recovery.log').open('a') as log:
    result=subprocess.run(['timeout','--signal=TERM','--kill-after=30s',str(remaining)+'s',sys.executable,'-u',str(root/'scripts/h_eval_recover.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
sys.exit(result.returncode)
