import hashlib,json,shutil,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
assert read(ROOT/'pilot_status.json')['state']=='complete'
s=read(ROOT/'analysis/condition_probe_summary.json');assert all(v['passed'] for v in s['audits'].values())
lock=read(ROOT/'training_lock.json');assert all(sha(p)==h for p,h in lock['locked_file_sha256'].items())
mapping={}
for raw,h in lock['locked_file_sha256'].items():
    p=Path(raw)
    if p.suffix in ('.py','.yaml','.json','.csv') and ROOT not in p.parents:
        dest=ROOT/'analysis/reused_source'/p.relative_to(ROOT.parent.parent)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);assert sha(dest)==h
        mapping[raw]=dict(archived_path=dest.relative_to(ROOT).as_posix(),sha256=h)
(ROOT/'analysis/reused_source_map.json').write_text(json.dumps(mapping,indent=2))
index_path=ROOT/'evidence_sha256.json'
files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and p!=index_path
    and p.suffix not in ('.pt','.pth','.pyc','.tmp') and not {'exports','__pycache__'}.intersection(p.parts)]
index={p.relative_to(ROOT).as_posix():sha(p) for p in files};index_path.write_text(json.dumps(index,indent=2))
out=ROOT/'exports';out.mkdir(exist_ok=True);archive=out/'state_condition_probe_evidence_without_checkpoints.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
    for p in files+[index_path]:tf.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
receipt=dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,indexed_files=len(index),
    checkpoints_retained_on_server=True,audits=s['audits'])
(out/'archive_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
