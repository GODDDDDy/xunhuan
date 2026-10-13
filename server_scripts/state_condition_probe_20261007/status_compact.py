import json,re,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
p=ROOT/'pilot_status.json'
if not p.exists():print(json.dumps(dict(state='starting')))
else:
    s=json.loads(p.read_text());row=dict(arm=s['arm'],stage=s['stage'],state=s['state'],
        elapsed_minutes=round((time.time()-s['time_unix'])/60,1),completed_stages=len(s['completed_stages']))
    if s.get('logfile'):
        log=Path(s['logfile']).read_text(errors='replace');images=re.findall(r'Evaluated (\d+) images',log)
        if images:row['images']=int(images[-1])
        if 'Traceback' in log:row['log_traceback']=True
    train=ROOT/'runs'/s['arm']/'seed_2026/logs/train.csv'
    if train.exists():row['last_training_step']=int(train.read_text().splitlines()[-1].split(',')[0])
    print(json.dumps(row))
