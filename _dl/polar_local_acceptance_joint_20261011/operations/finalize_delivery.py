"""Archive final CPU interpretation separately from the verified raw archive.

Adapted from this project's existing finalize_delivery.py; no model execution.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import re

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
documents = ROOT / 'interpretation/documents'
documents.mkdir(parents=True, exist_ok=True)
mapping = [
    (PROJECT / 'analysis/POLAR_LOCAL_ACCEPTANCE_RESULTS_ZH_20261011.md', documents / 'POLAR_LOCAL_ACCEPTANCE_RESULTS_ZH_20261011.md'),
    (PROJECT / 'analysis/POLAR_LOCAL_ACCEPTANCE_PROTOCOL_ZH_20261011.md', documents / 'POLAR_LOCAL_ACCEPTANCE_PROTOCOL_ZH_20261011.md'),
    (PROJECT / 'analysis/LOCAL_UPDATE_ACCEPTANCE_DESIGN_ZH_20261011.md', documents / 'LOCAL_UPDATE_ACCEPTANCE_DESIGN_ZH_20261011.md'),
    (PROJECT / '实验记录.md', documents / '实验记录_20261011完成快照.md'),
]
for source, destination in mapping:
    value = source.read_text(encoding='utf-8')
    def rebase(match):
        target = match.group(2)
        if target.startswith(('/', '#')) or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target):
            return match.group(0)
        resolved = (source.parent / target).resolve()
        return match.group(1) + os.path.relpath(resolved, destination.parent).replace('\\', '/') + ')'
    destination.write_text(re.sub(r'(!?\[[^\]]*\]\()([^\)]+)\)', rebase, value), encoding='utf-8')

receipt = json.loads((ROOT / 'archive_receipt.json').read_text(encoding='utf-8'))
summary = json.loads((ROOT / 'interpretation/result_summary.json').read_text(encoding='utf-8'))
assert receipt['all_hashes_match'] and receipt['files_verified'] == 194
assert receipt['source_archive_unchanged']
assert summary['source_weights_match'] and summary['frozen_code_match']
assert summary['checkpoint_step'] == 6000
assert summary['training']['steps'] == 6000 and summary['training']['unique_images'] == 5440
assert summary['training']['rates_and_freeze_verified']
assert summary['all_net_g_tensors_equal_reused_source']
assert summary['all_frozen_model_tensors_equal_reused_source']
assert summary['first_round_all_probes_exactly_unchanged']
assert summary['frozen_model_tensor_keys'] == 412
assert not summary['gpu_used_for_this_analysis']
assert summary['evaluation_completed_unix'] < summary['budget']['deadline_unix']
assert summary['budget']['start_unix'] == 1791700972.3300269
assert summary['budget']['deadline_unix'] == 1791787372.3300269
for split, frames in [('validation',872),('test',188)]:
    assert summary['splits'][split]['frames'] == frames
    assert summary['splits'][split]['t1_matches_previous_full_evaluation']
    assert summary['splits'][split]['local_acceptance']['all_protected_pixels_exactly_preserved']
assert summary['mechanism']['frames'] == 28 and not summary['mechanism']['test_used']
assert not summary['mechanism']['tuned_from_results']
for kind in ['confidence_examples', 'acceptance_examples']:
    assert len(list((ROOT / 'artifacts/downstream/metrics' / kind).glob('*/*.npz'))) == 16
assert len(list((ROOT / 'interpretation/region_examples').glob('*.png'))) == 16
assert (ROOT / 'artifacts.tar.gz').stat().st_size == receipt['bytes']

supplemental = []
for directory in ['operations', 'interpretation', 'supplemental_failure_records']:
    for path in sorted((ROOT / directory).rglob('*')):
        if not path.is_file() or any(p in path.parts for p in ['runtime', '__pycache__', 'mpl_cache']):
            continue
        supplemental.append(dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
completed = dict(recorded_at=datetime.now(timezone.utc).isoformat(),
    implementation_and_experiment_completed=True, method_fully_validated=False,
    reason='Exact preservation at M=0 verified; tiny final gains, weak difficult-region repair and imperfect acceptance; no matched continuation control or repeated seed.',
    protocol_id=summary['protocol_id'], raw_archive_sha256=receipt['archive_sha256'],
    original_archive_unchanged=True, original_files_verified=receipt['files_verified'],
    source_and_final_weights_match=True, final_checkpoint_sha256=summary['checkpoint_sha256'],
    source_checkpoint_sha256='ec7f0e6e79d0dcf21c536dc71958d44514bd7e78a8624bb936f16c02c37e81e8',
    frozen_runtime_sources_preserved=True, first_round_exactly_unchanged=True,
    steps=6000, training_images=5440, validation_frames=872, test_frames=188,
    mechanism_validation_frames=28, saved_confidence_examples=16, saved_acceptance_examples=16,
    local_region_heatmaps=16, evaluation_completed_unix=summary['evaluation_completed_unix'],
    deadline_unix=summary['budget']['deadline_unix'],
    post_deadline_gpu_training_or_evaluation=False, local_postprocessing_gpu_used=False,
    source_auto_report_lineage_annotation_in_final_document=True,
    supplemental_files=supplemental)
(ROOT / 'completion_receipt.json').write_text(json.dumps(completed, ensure_ascii=False, indent=2), encoding='utf-8')

# Validate every report link and every supplemental hash after writing the receipt.
for source, destination in mapping[:3]:
    for match in re.finditer(r'!?\[[^\]]*\]\(([^\)]+)\)', destination.read_text(encoding='utf-8')):
        target = match.group(1)
        if target.startswith(('#', 'https://', 'http://')):
            continue
        resolved = Path(target) if re.match(r'^[A-Za-z]:[/\\]', target) else destination.parent / target
        assert resolved.exists(), str(resolved)
for entry in supplemental:
    assert hashlib.sha256((ROOT / entry['path']).read_bytes()).hexdigest() == entry['sha256']
print(json.dumps({k:v for k,v in completed.items() if k!='supplemental_files'}))
