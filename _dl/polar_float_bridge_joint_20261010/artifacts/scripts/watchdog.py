"""Independent hard deadline; kill only the process group registered to this study."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
budget=json.loads((ROOT/'budget.json').read_text())
while time.time()<budget['deadline_unix']:
    if (ROOT/'archive_ready.json').exists():sys.exit(0)
    time.sleep(min(10,max(budget['deadline_unix']-time.time(),0)))
status=json.loads((ROOT/'status.json').read_text())
if status.get('stage') not in ('completed','limited','failed'):
    try:os.killpg(budget['process_group'],signal.SIGKILL)
    except ProcessLookupError:pass
    (ROOT/'status.json').write_text(json.dumps(dict(stage='deadline_stopped',previous=status,time_unix=time.time())))
    subprocess.run([sys.executable,str(ROOT/'scripts/archive.py')],cwd=ROOT)
