"""CPU-only post-closeout review; preserve40 archive/source/failures unchanged.

Repair only terminal status after completed scientific and archive receipts;
never call the worker, optimizer, inference or original finish function again.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import fcntl,hashlib,importlib.util,json,sys,time
from pathlib import Path
ROOT=Path('/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/g6_matched_enhancement_feedback_20261019')
OUT=ROOT/'reports/closeout_status_review_20261019'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(p)
def main():
 assert not (OUT/'POSTREVIEW_CLOSEOUT.json').exists(),'Existing review preserved'
 OUT.mkdir(parents=True,exist_ok=True)
 for kind in ['controller','worker']:
  reg=read(ROOT/(kind+'_registration.json'));proc=Path('/proc')/str(reg['pid'])
  assert not proc.exists(),'Do not review during own active process'
  with (ROOT/(kind+'.lock')).open('r') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 plan=read(ROOT/'config/locked.json');close=read(ROOT/'closeout.json');summary=read(ROOT/'reports/summary.json')
 assert close['success'] and close['steps']==1000 and close['validation']==64 and close['test']==0
 assert summary['next_action']=='stop_instance' and not any(summary['gates'].values())
 manifest=read(ROOT/'artifact_manifest.json')
 for r in manifest['files']:
  p=ROOT/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],r['path']
 assert sha(ROOT/'artifact_manifest.json')==close['archive_manifest_sha256']
 log=(ROOT/'logs/worker.log').read_text()
 assert "dict() got multiple values for keyword argument 'time_unix'" in log
 assert read(ROOT/'worker_exit.json')['exit_code']==1
 sys.path.insert(0,str(ROOT/'scripts'))
 import audit_study
 audit=audit_study.audit(plan)  # In-memory only: do not overwrite the archived audit.
 assert audit['passed']
 write(OUT/'original_status.json',read(ROOT/'status.json'))
 result=dict(passed=True,cpu_only=True,protocol_id=plan['protocol_id'],time_unix=time.time(),
  scientific_audit_repeated_in_memory=audit,original_archive_valid=True,original_archive_files=len(manifest['files']),
  original_archive_sha256=sha(ROOT/'artifact_manifest.json'),worker_exit_code_preserved=1,
  failure_scope='status() duplicate time_unix after successful scientific audit/archive and closeout writes',
  no_new_training=True,no_new_inference=True,no_weights_or_metrics_changed=True,original_source_and_failures_preserved=True)
 write(OUT/'REVIEW.json',result)
 files=[ROOT/'artifact_manifest.json',ROOT/'closeout.json',ROOT/'reports/final_review_closeout.json',ROOT/'worker_exit.json',ROOT/'controller_failure.json',ROOT/'scripts/closeout_status_review.py',OUT/'original_status.json',OUT/'REVIEW.json',*sorted((ROOT/'failures').glob('*.json'))]
 records=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
 write(OUT/'SUPPLEMENTARY_MANIFEST.json',dict(protocol_id=plan['protocol_id'],files=records))
 for r in records:
  p=ROOT/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
 final=dict(success=True,protocol_id=plan['protocol_id'],exploratory_only=True,steps=1000,validation=64,test=0,
  original_archive_valid=True,supplementary_archive_valid=True,original_archive_files=len(manifest['files']),
  supplementary_files=len(records),supplementary_manifest_sha256=sha(OUT/'SUPPLEMENTARY_MANIFEST.json'),
  scientific_evidence_valid=True,worker_exit_code=1,metadata_status_repaired=True,next_action='stop_instance',time_unix=time.time())
 write(OUT/'POSTREVIEW_CLOSEOUT.json',final)
 write(ROOT/'status.json',dict(stage='completed_metadata_status_repaired',**final))
 # Fresh verification after the final status mutation: these files stay immutable.
 for r in manifest['files']+records:
  p=ROOT/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],r['path']
 print(json.dumps(final),flush=True)
if __name__=='__main__':main()
