"""Register a locked-code repair in config/implementation_revisions.json.

Usage: make_revision.py <script_name.py> "<reason>"

Run AFTER the original script has been copied to
config/preserved/<script_name>.original.py and the repaired script has been
uploaded to scripts/<script_name>. Verifies the preserved copy still hashes to
the locked original before appending the revision record.
"""
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

name = sys.argv[1]
reason = sys.argv[2]
plan = json.loads((ROOT / "config/locked.json").read_text(encoding="utf-8-sig"))
codes = {r["path"]: r for r in plan["code_sources"]}
target = str((ROOT / "scripts" / name).resolve())
preserved = str((ROOT / "config/preserved" / (name + ".original.py")).resolve())
assert target in codes, name + " is not a locked code source"
assert sha(preserved) == codes[target]["sha256"], "preserved copy differs from locked original"
repaired = sha(target)
assert repaired != codes[target]["sha256"], "repaired file identical to original"

out = ROOT / "config/implementation_revisions.json"
existing = json.loads(out.read_text(encoding="utf-8-sig")) if out.exists() else []
# Idempotence: replace any earlier revision touching the same file.
existing = [r for r in existing
            if not any(c["path"] == target for c in r["replacements"])]
existing.append(dict(
    protocol_id=plan["protocol_id"], scientific_id=plan["scientific_id"],
    reason=reason,
    replacements=[dict(path=target, original_sha256=codes[target]["sha256"],
                       preserved_original_path=preserved, repaired_sha256=repaired)]))
out.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(dict(revision_written=True, target=target,
                      repaired_sha256=repaired,
                      original_sha256=codes[target]["sha256"])))
