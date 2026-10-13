"""Adapt existing project delivery receipts; keep the raw verified package intact."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,re

ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]
documents=ROOT/'interpretation/documents';documents.mkdir(parents=True,exist_ok=True)
mapping=[(PROJECT/'analysis/POLAR_GAIN_JUDGEMENT_SHORT_RESULTS_ZH_20261012.md',documents/'POLAR_GAIN_JUDGEMENT_SHORT_RESULTS_ZH_20261012.md'),
         (PROJECT/'analysis/POLAR_GAIN_JUDGEMENT_SHORT_PROTOCOL_ZH_20261012.md',documents/'POLAR_GAIN_JUDGEMENT_SHORT_PROTOCOL_ZH_20261012.md'),
         (PROJECT/'实验记录.md',documents/'实验记录_20261013第22项完成快照.md')]
mapping.insert(2,(PROJECT/'analysis/POLAR_GAIN_PREDICTABILITY_NEXT_STEP_ZH_20261012.md',documents/'POLAR_GAIN_PREDICTABILITY_NEXT_STEP_ZH_20261012.md'))
import shutil
assets=documents/'sources/external_next_step_20261012';assets.mkdir(parents=True,exist_ok=True)
asset_map={}
for original in (PROJECT/'analysis/sources/external_next_step_20261012').glob('*.png'):
    target=assets/original.name;shutil.copy2(original,target);asset_map[original.resolve()]=target.resolve()
document_map={source.resolve():destination.resolve() for source,destination in mapping}
document_map.update(asset_map)
for source,destination in mapping:
    text=source.read_text(encoding='utf-8')
    def rebase(match):
        target=match.group(2)
        if target.startswith(('/','#')) or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target):return match.group(0)
        resolved=(source.parent/target).resolve()
        resolved=document_map.get(resolved,resolved)
        return match.group(1)+os.path.relpath(resolved,destination.parent).replace('\\','/')+')'
    destination.write_text(re.sub(r'(!?\[[^\]]*\]\()([^\)]+)\)',rebase,text),encoding='utf-8')
receipt=json.loads((ROOT/'archive_receipt.json').read_text());summary=json.loads((ROOT/'interpretation/result_summary.json').read_text())
assert hashlib.sha256((ROOT/'archive_manifest.json').read_bytes()).digest()==hashlib.sha256((ROOT/'artifacts/archive_manifest.json').read_bytes()).digest()
assert receipt['all_hashes_match'] and receipt['files_verified']==1171 and receipt['source_archive_unchanged']
assert summary['source_weights_match'] and summary['frozen_code_match']
assert summary['all_frozen_model_tensors_equal_reused_source'] and summary['frozen_model_tensor_keys']==518
assert summary['all_net_g_tensors_equal_reused_source'] and summary['first_round_all_probes_exactly_unchanged']
assert summary['candidate_and_q_frozen_entire_run'] and summary['training']['all5440_manifest_and_roles_verified']
assert summary['training']['phases']['judgement']['steps']==3000 and summary['candidate_all_probes_exactly_unchanged']
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
    protocol_id=summary['protocol_id'],raw_archive_sha256=receipt['archive_sha256'],raw_files_verified=1171,
    original_archive_unchanged=True,source_and_final_weights_verified=True,source_checkpoint_sha256='cecdcfa47d290b0a2d6b662b96bb4407a7043ef5789e9e87d70768f2c683916d',
    final_checkpoint_sha256=summary['checkpoint_sha256'],candidate_steps=0,judgement_steps=3000,
    training_manifest_images=5440,current_training_unique_images=1140,current_calibration_unique_images=660,candidate_role_images_not_updated=3640,validation_frames=872,test_frames=188,original_size=True,tiling=False,
    first_round_exactly_unchanged=True,candidate_and_q_frozen_entire_run=True,saved_action_example_scenes=16,saved_confidence_example_scenes=16,
    evaluation_completed_unix=summary['evaluation_completed_unix'],deadline_unix=summary['budget']['deadline_unix'],
    local_postprocessing_gpu_used=False,post_deadline_gpu_training_or_evaluation=False,failure_packages=summary['failure_archive_verification'],supplemental_files=supplement)
(ROOT/'completion_receipt.json').write_text(json.dumps(completed,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in completed.items() if k!='supplemental_files'}))
