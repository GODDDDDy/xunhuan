"""Reuse the same absolute cutoff for a named read-only diagnostic phase."""
import json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
phase=sys.argv[1];assert phase=='bridge'
b=json.loads((ROOT/f'{phase}_budget.json').read_text())
assert b['deadline_unix']==1791605047.4916465
while time.time()<b['deadline_unix']:
    p=ROOT/f'{phase}_status.json'
    if p.exists() and json.loads(p.read_text()).get('stage') in ('completed','failed'):break
    time.sleep(min(10,max(b['deadline_unix']-time.time(),0)))
else:
    try:os.killpg(b['process_group'],signal.SIGKILL)
    except ProcessLookupError:pass
    (ROOT/f'{phase}_status.json').write_text(json.dumps(dict(stage='deadline_stopped',time_unix=time.time())))
(ROOT/f'{phase}_watchdog.json').write_text(json.dumps(dict(stage='exited',time_unix=time.time(),deadline_unix=b['deadline_unix'])))
