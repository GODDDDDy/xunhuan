"""Read small CPU logs; never touch GPU or weights while training."""
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parent.parent
s = json.loads((ROOT / 'status.json').read_text())
result = {k: s[k] for k in ('arm', 'seed', 'stage', 'state')}
result['completed_new_runs'] = len(s['completed_runs'])
result['total_new_runs'] = s['new_runs_total']
result['stage_minutes'] = round((time.time() - s['time_unix']) / 60, 1)
if s['arm'] in ('G6', 'K1', 'K2'):
    log = ROOT / 'runs' / s['arm'] / f'seed_{s["seed"]}/logs/train.csv'
    if log.is_file():
        line = log.read_text().splitlines()[-1]
        if not line.startswith('step,'):
            result['training_step'] = int(line.split(',')[0])
    if s.get('logfile') and Path(s['logfile']).is_file():
        progress = re.findall(r'Evaluated (\d+) images', Path(s['logfile']).read_text())
        if progress:
            result['evaluated_images'] = int(progress[-1])
if s['state'] == 'failed':
    result['failure_log'] = s.get('logfile')
print(json.dumps(result))
