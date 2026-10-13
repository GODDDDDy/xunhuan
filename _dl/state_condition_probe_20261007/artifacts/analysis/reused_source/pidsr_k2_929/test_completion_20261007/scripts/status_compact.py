import json,re,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
s=json.loads((ROOT/'status.json').read_text())
r={k:s[k] for k in ('unit','family','seed','state')}
r['completed_units']=len(s['completed_units']);r['total_units']=s['total_units']
r['minutes']=round((time.time()-s['time_unix'])/60,1)
if s.get('logfile'):
    progress=re.findall(r'Evaluated (\d+) images',Path(s['logfile']).read_text())
    r['images']=int(progress[-1]) if progress else 0
    if s['state']=='failed':r['failure_log']=s['logfile']
print(json.dumps(r))
