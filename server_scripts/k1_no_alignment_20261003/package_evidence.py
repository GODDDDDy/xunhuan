"""Package small outputs only after completed audited analysis and report review."""
import hashlib
import json
from pathlib import Path
import tarfile
ROOT=Path(__file__).resolve().parent.parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read(ROOT/'status.json')['state']=='complete'
summary=read(ROOT/'analysis/alignment_paired_summary.json')
assert all(v['passed'] for v in summary['audits'].values())
assert all(sha(p)==h for p,h in read(ROOT/'protocol_lock.json')['locked_file_sha256'].items())
index_path=ROOT/'evidence_sha256.json'
files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and p!=index_path
       and p.suffix not in {'.pt','.pth','.pyc','.tmp'} and not {'exports','__pycache__'}.intersection(p.parts)]
index={p.relative_to(ROOT).as_posix():sha(p) for p in files}
index_path.write_text(json.dumps(index,indent=2))
dest=ROOT/'exports'
dest.mkdir(exist_ok=True)
archive=dest/'k1_no_alignment_evidence_without_checkpoints.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
    for p in files+[index_path]:tf.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
receipt={'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'indexed_files':len(index),
         'checkpoints_retained_on_server':True,'checkpoint_fingerprints':summary['source_fingerprints']}
(dest/'archive_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
