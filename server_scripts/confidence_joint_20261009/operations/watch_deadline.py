"""Enforce the user's eight-hour deadline for this study's own process tree only."""
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path('/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/confidence_joint_20261009_v2')
PID = int((ROOT / 'logs/overnight.pid').read_text())
DEADLINE = json.loads((ROOT / 'protocol_lock.json').read_text())['deadline_unix']

def identity(pid):
    try:
        data = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return data[19], data[0]  # process start tick and state
    except FileNotFoundError:
        return None, None

START, _ = identity(PID)
assert START and Path(f'/proc/{PID}/cwd').resolve() == ROOT
assert b'run_overnight.py' in Path(f'/proc/{PID}/cmdline').read_bytes()
(ROOT / 'operations/watchdog_audit.json').write_text(json.dumps(dict(
    root_pid=PID, root_start_tick=START, deadline_unix=DEADLINE,
    scope='only descendants of the verified study controller; unrelated jobs excluded'), indent=2))
while time.time() < DEADLINE:
    start, state = identity(PID)
    if start != START or state == 'Z':
        raise SystemExit(0)
    if json.loads((ROOT / 'status.json').read_text()).get('stage') == 'completed':
        raise SystemExit(0)
    time.sleep(min(15, max(DEADLINE - time.time(), .1)))
if identity(PID)[0] == START:
    descendants = {PID}
    parents = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit(): continue
        try:
            data = (path / 'stat').read_text().split(') ', 1)[1].split()
            if path.stat().st_uid == os.getuid(): parents[int(path.name)] = int(data[1])
        except (FileNotFoundError, ProcessLookupError, PermissionError): pass
    while True:
        more = {pid for pid, parent in parents.items() if parent in descendants}
        old = len(descendants); descendants.update(more)
        if len(descendants) == old: break
    previous = json.loads((ROOT / 'status.json').read_text())
    for pid in sorted(descendants - {PID}, reverse=True):
        try: os.kill(pid, signal.SIGTERM)
        except ProcessLookupError: pass
    try: os.kill(PID, signal.SIGTERM)
    except ProcessLookupError: pass
    if previous.get('stage') != 'completed':
        value = dict(stage='budget_expired', time_unix=time.time(), test_completed=False,
                     stopped_owned_pids=sorted(descendants), previous=previous,
                     reason='User eight-hour deadline; preserve checkpoints and partial evaluation')
        tmp = ROOT / 'status.json.tmp'; tmp.write_text(json.dumps(value, indent=2)); tmp.replace(ROOT / 'status.json')
