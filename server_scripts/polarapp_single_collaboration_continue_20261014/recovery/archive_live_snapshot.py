"""Reuse frozen archive with byte snapshots of mutable bookkeeping files."""
import hashlib
import sys
from pathlib import Path


def patched_source(source):
    source = source.replace("import argparse\n", "import argparse\nimport io\nSNAPSHOT_BYTES = {}\n", 1)
    old = '''        before = path.stat()
        digest = sha(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"file changed while hashing: {name}")
        included.append({"path": name, "bytes": after.st_size, "sha256": digest})'''
    new = '''        if len(relative.parts) == 1 or relative.parts[0] == "logs":
            data = path.read_bytes()
            SNAPSHOT_BYTES[name] = data
            included.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        else:
            before = path.stat()
            digest = sha(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"file changed while hashing: {name}")
            included.append({"path": name, "bytes": after.st_size, "sha256": digest})'''
    if source.count(old) != 1:
        raise RuntimeError("Unexpected frozen inventory layout")
    source = source.replace(old, new)
    old_add = '            target.add(root / entry["path"], arcname=entry["path"], recursive=False)'
    new_add = '''            if entry["path"] in SNAPSHOT_BYTES:
                info = target.gettarinfo(str(root / entry["path"]), arcname=entry["path"])
                data = SNAPSHOT_BYTES[entry["path"]]
                info.size = len(data)
                target.addfile(info, io.BytesIO(data))
            else:
                target.add(root / entry["path"], arcname=entry["path"], recursive=False)'''
    if source.count(old_add) != 1:
        raise RuntimeError("Unexpected frozen tar-writing layout")
    source = source.replace(old_add, new_add)
    source = source.replace("比较同一来源的普通微调与参数高效元学习协作，部署均为单轮增强加PolarFree。",
                            "只训练参数高效元学习协作，与训练前来源比较，部署为单轮增强加PolarFree。")
    source = source.replace("来源/control/meta配对结果", "来源/meta配对结果")
    return source


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from runtime import read_locked_config
    base = root / "scripts/archive.py"
    config = read_locked_config(root)
    if hashlib.sha256(base.read_bytes()).hexdigest() != config["code_sha256"]["scripts/archive.py"]:
        raise RuntimeError("Frozen base archive code changed")
    exec(compile(patched_source(base.read_text(encoding="utf-8")), str(base), "exec"),
         {"__name__": "__main__", "__file__": __file__})
