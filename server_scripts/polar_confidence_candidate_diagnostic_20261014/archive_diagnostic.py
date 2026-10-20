"""Archive final diagnostic records, including dereferenced source checkpoints."""
import hashlib,json,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
files=[]
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc' or p.name in ('artifacts.tar.gz','archive_ready.json','archive_manifest.json') or p.name.endswith('.tmp'):continue
    if p.name in ('watchdog.json','controller.log'):continue # active files separately copied below
    files.append(p)
snapshot=ROOT/'diagnostics/archive_snapshots';snapshot.mkdir(exist_ok=True)
for name in ('watchdog.json','logs/controller.log'):
    p=ROOT/name
    if p.exists():
        q=snapshot/(p.name+'.snapshot');q.write_bytes(p.read_bytes());files.append(q)
manifest=dict(created_unix=time.time(),protocol_id=json.loads((ROOT/'config/locked_run.json').read_text())['protocol_id'],files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files],original_size=True,training_steps=0,test_frames=0,source_checkpoint_included=True,source_config_and_runtime_code_included=True)
mp=ROOT/'archive_manifest.json';mp.write_text(json.dumps(manifest,indent=2,ensure_ascii=False));files.append(mp)
archive=ROOT/'artifacts.tar.gz'
with tarfile.open(archive,'w:gz',dereference=True,compresslevel=3) as tar:
    for p in files:tar.add(p,arcname='artifacts/'+str(p.relative_to(ROOT)),recursive=False)
ready=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),manifest_sha256=sha(mp),files=len(manifest['files']),protocol_id=manifest['protocol_id'],created_unix=time.time())
(ROOT/'archive_ready.json').write_text(json.dumps(ready,indent=2));print(json.dumps(ready),flush=True)
