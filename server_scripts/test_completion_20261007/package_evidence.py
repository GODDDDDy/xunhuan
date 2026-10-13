"""Package small outputs only after completed audited analysis and report review."""
import hashlib
import json
from pathlib import Path
import tarfile
import shutil
ROOT=Path(__file__).resolve().parent.parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read(ROOT/'status.json')['state']=='complete'
summary=read(ROOT/'analysis/test_completion_summary.json')
assert all(v['passed'] for v in summary['audits'].values())
assert all(sha(p)==h for p,h in read(ROOT/'protocol_lock.json')['locked_file_sha256'].items())
source_map={}
for raw,h in read(ROOT/'protocol_lock.json')['locked_file_sha256'].items():
    p=Path(raw)
    if p.suffix in {'.py','.yaml','.csv','.json'} and ROOT not in p.parents:
        relative=p.relative_to(ROOT.parent.parent)
        dest=ROOT/'analysis/reused_source'/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest)
        assert sha(dest)==h
        source_map[raw]={'archived_path':dest.relative_to(ROOT).as_posix(),'sha256':h}
(ROOT/'analysis/reused_source_map.json').write_text(json.dumps(source_map,indent=2))
index_path=ROOT/'evidence_sha256.json'
files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and p!=index_path
       and p.suffix not in {'.pt','.pth','.pyc','.tmp'} and not {'exports','__pycache__'}.intersection(p.parts)]
index={p.relative_to(ROOT).as_posix():sha(p) for p in files}
index_path.write_text(json.dumps(index,indent=2))
dest=ROOT/'exports'
dest.mkdir(exist_ok=True)
archive=dest/'test_completion_evidence_without_checkpoints.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
    for p in files+[index_path]:tf.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
receipt={'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'indexed_files':len(index),
         'checkpoints_retained_on_server':True,'checkpoint_fingerprints':summary['source_fingerprints']}
(dest/'archive_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))


