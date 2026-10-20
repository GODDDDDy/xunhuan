"""Freeze completed or failed experiment records; read-only dereferenced inputs."""
import hashlib,json,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
snap=ROOT/'diagnostics/active_snapshots';snap.mkdir(exist_ok=True)
for name in ('logs/controller.log','watchdog.json'):
    p=ROOT/name
    if p.exists():(snap/(p.name+'.snapshot')).write_bytes(p.read_bytes())
files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name not in ('controller.log','watchdog.json','artifacts.tar.gz','archive_manifest.json','archive_ready.json') and not p.name.endswith('.tmp')]
status=json.loads((ROOT/'status.json').read_text());plan=json.loads((ROOT/'config/plan.json').read_text())
manifest=dict(files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files],protocol_id=plan['protocol_id'],completed=status['stage']=='completed',created_unix=time.time())
mp=ROOT/'archive_manifest.json';mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2));files.append(mp)
with tarfile.open(ROOT/'artifacts.tar.gz','w:gz',dereference=True,compresslevel=3) as tar:
    for p in files:tar.add(p,arcname='artifacts/'+str(p.relative_to(ROOT)),recursive=False)
archive=ROOT/'artifacts.tar.gz';ready=dict(bytes=archive.stat().st_size,sha256=sha(archive),manifest_sha256=sha(mp),files=len(manifest['files']),completed=manifest['completed'],protocol_id=plan['protocol_id'],created_unix=time.time())
(ROOT/'archive_ready.json').write_text(json.dumps(ready,indent=2));print(json.dumps(ready),flush=True)
