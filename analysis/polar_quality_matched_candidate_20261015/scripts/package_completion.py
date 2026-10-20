"""Archive completion documents separately, preserving the verified main tar."""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile

project = Path('F:/WORK/projects/pidsr_recurrent_joint_k2')
analysis = project / 'analysis/polar_quality_matched_candidate_20261015'
download = project / '_dl/polar_quality_matched_candidate_20261015'
destination = download / 'completion'
destination.mkdir(exist_ok=True)
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
primary = json.loads((download / 'archive_ready.json').read_text(encoding='utf-8'))
assert sha(download / 'archive_manifest.json') == primary['manifest_sha256']
primary_receipt = json.loads((download / 'archive_receipt.json').read_text(encoding='utf-8'))
assert primary_receipt['all_hashes_match'] and primary_receipt['files_verified'] == 190
copies = [(p, Path('analysis') / p.relative_to(analysis)) for p in analysis.rglob('*')
    if p.is_file() and 'runtime_cache' not in p.parts and p.suffix in {'.md', '.json', '.py', '.png', '.svg'}]
copies += [(project / 'analysis/POLAR_QUALITY_MATCHED_CANDIDATE_PROTOCOL_ZH_20261015.md', Path('protocol.md')),
    (project / '实验记录.md', Path('实验记录_snapshot.md'))]
copies += [(download / name, Path('primary_package') / name)
    for name in ['archive_ready.json', 'archive_manifest.json', 'archive_receipt.json', 'operations/verify_archive.py']]
failure = download / 'failures/controller_collision_attempt1'
copies += [(failure / name, Path('separate_failure_package') / name)
    for name in ['archive_ready.json', 'archive_receipt.json']]
for source, relative in copies:
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
readme = destination / 'README.md'
readme.write_text('# 第27项完成补充归档\n\n主包保持不变，190文件核验通过；本包补充完整结果解释、'
    '图表、CPU完成核验、脚本与实验记录快照。包内primary_package记录绑定的主包SHA。'
    '单独的失败tar位于failures/controller_collision_attempt1，本补充包不重复包含该tar。'
    '失败tar的103项核验中，3项原为外部链接，按白名单引用已核验本机权重物化；原失败tar未修改。'
    '本包内的分析仅事后描述，未改变终点、条件或评价。\n', encoding='utf-8')
files = sorted(p for p in destination.rglob('*') if p.is_file() and p != destination / 'archive_manifest.json')
manifest = {'primary_archive_sha256': primary['sha256'], 'primary_files_verified': 190,
    'files': [{'path': str(p.relative_to(destination)).replace('\\', '/'), 'bytes': p.stat().st_size,
        'sha256': sha(p)} for p in files]}
manifest_path = destination / 'archive_manifest.json'
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
package = download / 'completion_artifacts.tar.gz'
assert not package.exists(), 'Preserve any existing completion package instead of overwriting it'
with tarfile.open(package, 'w:gz', compresslevel=3) as archive:
    for p in files + [manifest_path]:
        archive.add(p, arcname='completion/' + str(p.relative_to(destination)).replace('\\', '/'), recursive=False)
with tarfile.open(package, 'r:gz') as archive:
    members = archive.getmembers()
    assert all(x.isfile() and not x.issym() and not x.islnk() for x in members)
    assert len(members) == len(files) + 1
    assert hashlib.sha256(archive.extractfile('completion/archive_manifest.json').read()).hexdigest() == sha(manifest_path)
    for item in manifest['files']:
        member = archive.getmember('completion/' + item['path'])
        assert member.size == item['bytes']
        assert hashlib.file_digest(archive.extractfile(member), 'sha256').hexdigest() == item['sha256']
ready = {'bytes': package.stat().st_size, 'sha256': sha(package), 'manifest_sha256': sha(manifest_path),
    'files_verified': len(files), 'all_sizes_and_hashes_match': True, 'completed': True,
    'primary_archive_sha256': primary['sha256'], 'cpu_only': True}
(download / 'completion_archive_ready.json').write_text(json.dumps(ready, indent=2), encoding='utf-8')
print(json.dumps(ready, indent=2))
