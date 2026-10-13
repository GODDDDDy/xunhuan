"""Record local interpretation/documents separately from the original archive."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import re

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
documents = ROOT / 'interpretation/documents'
mapping = [
    (PROJECT / 'analysis/POLAR_RESIDUAL_FEEDBACK_RESULTS_ZH_20261010.md', documents / 'POLAR_RESIDUAL_FEEDBACK_RESULTS_ZH_20261010.md'),
    (PROJECT / 'analysis/POLAR_RESIDUAL_FEEDBACK_PROTOCOL_ZH_20261010.md', documents / 'POLAR_RESIDUAL_FEEDBACK_PROTOCOL_ZH_20261010.md'),
    (PROJECT / '实验记录.md', documents / '实验记录_20261011快照.md'),
]
for source, destination in mapping:
    text = source.read_text(encoding='utf-8')
    def rebase(match):
        target = match.group(2)
        if target.startswith(('/', '#')) or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target):
            return match.group(0)
        resolved = (source.parent / target).resolve()
        return match.group(1) + os.path.relpath(resolved, destination.parent).replace('\\', '/') + ')'
    destination.write_text(re.sub(r'(!?\[[^\]]*\]\()([^\)]+)\)', rebase, text), encoding='utf-8')

receipt = json.loads((ROOT / 'archive_receipt.json').read_text(encoding='utf-8'))
summary = json.loads((ROOT / 'interpretation/result_summary.json').read_text(encoding='utf-8'))
assert receipt['all_hashes_match'] and summary['source_weights_match'] and summary['frozen_code_match']
assert summary['training']['unique_images'] == 5440
examples = list((ROOT / 'artifacts/downstream/metrics/confidence_examples').glob('*/*.npz'))
assert len(examples) == 16
supplemental = []
for directory in ['operations', 'interpretation', 'supplemental_failure_records']:
    for path in sorted((ROOT / directory).rglob('*')):
        if not path.is_file() or 'runtime' in path.relative_to(ROOT).parts or '__pycache__' in path.parts:
            continue
        supplemental.append(dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
completed = dict(recorded_at=datetime.now(timezone.utc).isoformat(),
    implementation_and_experiment_completed=True, method_fully_validated=False,
    reason='Small positive PSNR iteration gains, negative image-mean SSIM; no matched training control or multi-seed confirmation.',
    raw_archive_sha256=receipt['archive_sha256'], original_archive_unchanged=True,
    original_files_verified=receipt['files_verified'], source_and_final_weights_match=True,
    frozen_runtime_sources_preserved=True, steps=12000, validation_frames=872, test_frames=188,
    mechanism_validation_frames=28, saved_scene_examples=16,
    post_deadline_gpu_training_or_evaluation=False, local_postprocessing_gpu_used=False,
    supplemental_files=supplemental)
(ROOT / 'completion_receipt.json').write_text(json.dumps(completed, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in completed.items() if k!='supplemental_files'}))
