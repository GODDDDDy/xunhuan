"""Independent cutoff for diagnostics, preserving the original authorization."""
import json,os,signal,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
b=json.loads((ROOT/'budget.json').read_text())
assert b['deadline_unix']==1791605047.4916465
while time.time()<b['deadline_unix']:
    p=ROOT/'status.json'
    if p.exists() and json.loads(p.read_text()).get('stage') in ('completed','failed'):break
    time.sleep(min(10,max(b['deadline_unix']-time.time(),0)))
else:
    try:os.killpg(b['process_group'],signal.SIGKILL)
    except ProcessLookupError:pass
    (ROOT/'status.json').write_text(json.dumps(dict(stage='deadline_stopped',time_unix=time.time())))
(ROOT/'watchdog.json').write_text(json.dumps(dict(stage='exited',time_unix=time.time(),deadline_unix=b['deadline_unix'])))
