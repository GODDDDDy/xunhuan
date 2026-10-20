"""One finite CPU continuation after GPU/archive completion; no training or CUDA."""
import json,subprocess,sys,time,traceback,hashlib
from pathlib import Path
ROOT=Path(sys.argv[1]);PY=sys.executable
STATUS=ROOT/'reports/cpu_postprocess_status.json'
def write(stage,**kw):
    p=STATUS.with_suffix('.tmp');p.write_text(json.dumps(dict(stage=stage,time_unix=time.time(),cpu_only=True,**kw),indent=2));p.replace(STATUS)
try:
    write('waiting_for_baseline_and_primary_archive')
    while not (ROOT/'archive_ready.json').exists():
        state=json.loads((ROOT/'status.json').read_text())
        if state.get('stage') in ('failed','deadline_exceeded','deadline_reached'):
            raise RuntimeError('GPU baseline is incomplete; do not extend the budget')
        deadline=json.loads((ROOT/'budget.json').read_text())['deadline_unix']
        if time.time()>deadline+180:raise RuntimeError('No completed archive after fixed GPU deadline')
        time.sleep(10)
    assert json.loads((ROOT/'archive_ready.json').read_text())['completed']
    write('verifying_primary_archive')
    directory=ROOT/'reports/cpu_review'
    with (ROOT/'logs/server_archive_verification.log').open('a') as log:
        subprocess.run([PY,str(directory/'verify_server_archive.py'),str(ROOT)],stdout=log,stderr=subprocess.STDOUT,check=True)
        failed=ROOT/'diagnostics/gpu_initial_attempt1'
        subprocess.run([PY,str(directory/'verify_server_archive.py'),str(failed)],stdout=log,stderr=subprocess.STDOUT,check=True)
    write('analyzing_paired_baselines_and_regions')
    with (ROOT/'logs/cpu_final_analysis.log').open('a') as log:
        subprocess.run([PY,str(directory/'analyze_audit.py'),'--source27',str(ROOT.parent/'polar_quality_matched_candidate_20261015'),
            '--baseline',str(ROOT),'--output',str(ROOT/'reports/analysis28'),'--require-baseline','--no-plots'],
            stdout=log,stderr=subprocess.STDOUT,check=True)
    data=json.loads((ROOT/'reports/analysis28/audit_results.json').read_text())
    assert data['baseline_complete'] and data['paired_training_changes']['frames']==872
    changes=data['paired_training_changes']
    compact=dict(training=False,test_used=False,frames=872,all_T1_hashes_equal=True,
        image_mean_endpoint_minus_initial=changes['image_means'],per_scene=changes['per_scene'],
        regional_frames=data['regional_frames'],regional_rows=len(data['regions']))
    (ROOT/'reports/analysis28/compact_summary.json').write_text(json.dumps(compact,indent=2))
    write('completed_analysis_ready_for_interpretation',next_step_authorized=True,
        no_new_training_started=True,downloaded=False,summary_path='reports/analysis28/compact_summary.json')
except Exception:
    write('failed',error=traceback.format_exc());raise
