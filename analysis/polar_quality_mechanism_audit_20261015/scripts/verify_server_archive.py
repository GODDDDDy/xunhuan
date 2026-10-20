"""Server-only streaming archive verification, no extraction or model inference."""
import hashlib,json,sys,tarfile
from pathlib import Path,PurePosixPath
root=Path(sys.argv[1])
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
ready=json.loads((root/'archive_ready.json').read_text())
manifest_path=root/'archive_manifest.json'
assert sha(manifest_path)==ready['manifest_sha256']
manifest=json.loads(manifest_path.read_text());items={x['path']:x for x in manifest['files']}
assert len(items)==len(manifest['files'])==ready['files']
archive=root/'artifacts.tar.gz'
assert archive.stat().st_size==ready['bytes'] and sha(archive)==ready['sha256']
seen=set();manifest_seen=False
with tarfile.open(archive,mode='r|gz') as tar:
    for member in tar:
        name=PurePosixPath(member.name)
        assert member.isfile() and not member.issym() and not member.islnk()
        assert name.parts[0]=='artifacts' and '..' not in name.parts and not name.is_absolute()
        relative=str(PurePosixPath(*name.parts[1:]));stream=tar.extractfile(member)
        value=hashlib.file_digest(stream,'sha256').hexdigest()
        if relative=='archive_manifest.json':
            assert not manifest_seen and value==ready['manifest_sha256'];manifest_seen=True
        else:
            assert relative in items and relative not in seen
            expected=items[relative]
            assert member.size==expected['bytes'] and value==expected['sha256'],relative
            seen.add(relative)
assert seen==set(items) and manifest_seen
receipt=dict(archive_sha256=ready['sha256'],bytes=ready['bytes'],files_verified=len(seen),
    all_sizes_and_hashes_match=True,archive_completed=ready['completed'],
    server_only=True,extracted=False,downloaded=False,verification_gpu_used=False)
out=root/'server_archive_receipt.json';out.write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
