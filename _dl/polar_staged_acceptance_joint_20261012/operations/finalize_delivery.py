"""Adapt existing project delivery receipts; keep the raw verified package intact."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,re

ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]
documents=ROOT/'interpretation/documents';documents.mkdir(parents=True,exist_ok=True)
mapping=[(PROJECT/'analysis/POLAR_STAGED_ACCEPTANCE_RESULTS_ZH_20261012.md',documents/'POLAR_STAGED_ACCEPTANCE_RESULTS_ZH_20261012.md'),
         (PROJECT/'analysis/POLAR_STAGED_ACCEPTANCE_PROTOCOL_ZH_20261012.md',documents/'POLAR_STAGED_ACCEPTANCE_PROTOCOL_ZH_20261012.md'),
         (PROJECT/'实验记录.md',documents/'实验记录_20261012第20组完成快照.md')]
for source,destination in mapping:
    text=source.read_text(encoding='utf-8')
    def rebase(match):
        target=match.group(2)
        if target.startswith(('/','#')) or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target):return match.group(0)
        resolved=(source.parent/target).resolve()
        return match.group(1)+os.path.relpath(resolved,destination.parent).replace('\\','/')+')'
    destination.write_text(re.sub(r'(!?\[[^\]]*\]\()([^\)]+)\)',rebase,text),encoding='utf-8')
receipt=json.loads((ROOT/'archive_receipt.json').read_text());summary=json.loads((ROOT/'interpretation/result_summary.json').read_text())
assert receipt['all_hashes_match'] and receipt['files_verified']==243 and receipt['source_archive_unchanged']
assert summary['source_weights_match'] and summary['frozen_code_match']
assert summary['all_frozen_model_tensors_equal_reused_source'] and summary['frozen_model_tensor_keys']==420
assert summary['all_net_g_tensors_equal_reused_source'] and summary['first_round_all_probes_exactly_unchanged']
assert summary['candidate_frozen_during_judgement'] and summary['training']['all5440_in_corresponding_role']
assert all(summary['training']['phases'][p]['steps']==3000 for p in ('candidate','judgement'))
assert summary['calibration']['chosen']['coverage']==0 and not summary['calibration']['test_used']
assert not summary['gpu_used_for_this_analysis'] and summary['evaluation_completed_unix']<summary['budget']['deadline_unix']
for split,count in [('validation',872),('test',188)]:
    x=summary['splits'][split];assert x['frames']==count and x['all_images_t2_equals_t1'] and x['t1_matches_previous_full_evaluation']
assert summary['saved_action_example_scenes']==summary['saved_confidence_example_scenes']==16
assert (ROOT/'artifacts.tar.gz').stat().st_size==receipt['bytes']
for _,destination in mapping[:2]:
    for match in re.finditer(r'!?\[[^\]]*\]\(([^\)]+)\)',destination.read_text(encoding='utf-8')):
        target=match.group(1)
        if target.startswith(('#','https://','http://')):continue
        resolved=Path(target) if re.match(r'^[A-Za-z]:[/\\]',target) else destination.parent/target
        assert resolved.exists(),str(resolved)
supplement=[]
for directory in ('operations','interpretation'):
    for path in sorted((ROOT/directory).rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            supplement.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
completed=dict(recorded_at=datetime.now(timezone.utc).isoformat(),implementation_and_experiment_completed=True,
    method_iteration_improvement_observed=False,reason='All updates rejected: zero coverage and T2 exactly T1 on all1060 frames.',
    protocol_id=summary['protocol_id'],raw_archive_sha256=receipt['archive_sha256'],raw_files_verified=243,
    original_archive_unchanged=True,source_and_final_weights_verified=True,source_checkpoint_sha256='e8a635e052ec86f9abddeaa6731a3cb85533420aa78d3b0ff52fbac5bd38e299',
    final_checkpoint_sha256=summary['checkpoint_sha256'],candidate_steps=3000,judgement_steps=3000,
    training_images_across_roles=5440,validation_frames=872,test_frames=188,original_size=True,tiling=False,
    first_round_exactly_unchanged=True,candidate_frozen_during_judgement=True,saved_action_example_scenes=16,saved_confidence_example_scenes=16,
    evaluation_completed_unix=summary['evaluation_completed_unix'],deadline_unix=summary['budget']['deadline_unix'],
    local_postprocessing_gpu_used=False,post_deadline_gpu_training_or_evaluation=False,supplemental_files=supplement)
(ROOT/'completion_receipt.json').write_text(json.dumps(completed,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in completed.items() if k!='supplemental_files'}))
