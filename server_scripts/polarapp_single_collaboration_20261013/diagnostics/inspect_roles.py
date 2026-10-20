import csv,collections,json
from pathlib import Path
p=Path('/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/polarapp_single_collaboration_20261013')
c=json.loads((p/'config/locked_run.json').read_text())
with open(c['manifest']) as f: rows=list(csv.DictReader(f))
print(json.dumps({'fields':list(rows[0]),'validation_roles':dict(collections.Counter((r['scene']+'/'+r['difficulty']) for r in rows if r['split']=='validation')),'test_count':sum(r['split']=='test' for r in rows)}))
