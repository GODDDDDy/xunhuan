"""Independent CPU-only consistency/aggregation audit; image metrics are not recomputed."""
import argparse
import hashlib
import json
import math
import statistics
import traceback
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def close(a, b, path="summary"):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for key in a:
            close(a[key], b[key], path + "/" + key)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert isinstance(b, (int, float)) and math.isfinite(b), path
        assert abs(a - b) <= 1e-10, (path, a, b)
    else:
        assert a == b, (path, a, b)


def summarize(rows, protocol):
    out = {"protocol_id": protocol, "n_rows": len(rows), "aggregates": {}}
    for split in ("train16", "validation64"):
        selected = [r for r in rows if r["split"] == split]
        out["aggregates"][split] = {}
        for area in ("center96", "all128"):
            dest = out["aggregates"][split][area] = {}
            for model in ("F", "FitP"):
                scenes = {}
                for scene in sorted({r["scene"] for r in selected}):
                    rr = [r for r in selected if r["scene"] == scene]
                    vals = {k: {} for k in ("g_crop", "g_full", "difference", "q_final", "t1_gap")}
                    for metric in ("psnr", "ssim"):
                        pairs = [(r["scores"][area]["crop"], r["scores"][area]["full"]) for r in rr]
                        gc = [c[model][metric] - c["T1"][metric] for c, f in pairs]
                        gf = [f[model][metric] - f["T1"][metric] for c, f in pairs]
                        q = [c[model][metric] - f[model][metric] for c, f in pairs]
                        t = [c["T1"][metric] - f["T1"][metric] for c, f in pairs]
                        for key, values in zip(vals, (gc, gf, [a-b for a,b in zip(gc,gf)], q, t)):
                            vals[key][metric] = statistics.fmean(values)
                    scenes[scene] = vals
                dest[model] = {k: {m: statistics.fmean(s[k][m] for s in scenes.values())
                                  for m in ("psnr", "ssim")} for k in next(iter(scenes.values()))}
                dest[model]["scenes"] = scenes
    return out


def audit(root):
    cfg = read(root / "config/locked.json")
    assert cfg["protocol_id"] and cfg["sources"] and cfg["code_sources"]
    assert len(cfg["code_sources"]) >= 2
    assert str(Path(__file__).resolve()) in {str(Path(s["path"]).resolve()) for s in cfg["code_sources"]}
    assert cfg["scoring"] == {"center96": [16, 16, 96, 96], "all128": [0, 0, 128, 128]}
    source_checks = []
    revision_path = root / "config/implementation_revision_1.json"
    revision = read(revision_path) if revision_path.exists() else None
    changes = {} if revision is None else {r["path"]: r for r in revision["replacements"]}
    if revision:
        assert revision["protocol_id"] == cfg["protocol_id"]
        assert set(changes) <= {str(root / "scripts/run_context.py"), str(root / "scripts/audit_context.py")}
    for item in cfg["sources"] + cfg["code_sources"] + [cfg["reference"]]:
        expected = item["sha256"]
        if item["path"] in changes:
            change = changes[item["path"]]
            assert change["original_sha256"] == expected
            assert sha(change["preserved_original_path"]) == expected
            expected = change["repaired_sha256"]
        actual = sha(item["path"])
        assert actual == expected, ("source hash", item["path"], actual)
        source_checks.append({"path": item["path"], "sha256": actual})
    rows = read(root / "metrics/rows.json")["rows"]
    assert len(rows) == 80 and Counter(r["split"] for r in rows) == {"train16": 16, "validation64": 64}
    items = {r["sample_id"]: r for r in cfg["items"]}
    assert len(items) == 80 and len({r["sample_id"] for r in rows}) == 80
    assert {r["sample_id"] for r in rows} == set(items)
    refs = {r["sample_id"]: r for r in read(cfg["reference"]["path"])["rows"]}
    assert len(refs) == 872
    val_scenes = Counter(r["scene"] for r in rows if r["split"] == "validation64")
    assert len(val_scenes) == 8 and set(val_scenes.values()) == {8}
    for row in rows:
        item = items[row["sample_id"]]
        for key in ("split", "scene", "top", "left"):
            assert row[key] == item[key], (row["sample_id"], key)
        assert row["shape"] == [1, 3, item["height"], item["width"]]
        assert 0 <= row["top"] <= item["height"] - 128 and 0 <= row["left"] <= item["width"] - 128
        assert row["weights_unchanged"] is True
        noise = row["noise"]
        assert noise["shape"] == [1, 16, 256] and len(noise["used_sha256"]) == 64
        assert {c["scope"] for c in noise["calls"]} == {"full", "crop"}
        for call in noise["calls"]:
            assert call["used_sha256"] == noise["used_sha256"] and len(call["native_sha256"]) == 64
            if call["scope"] == "full":
                assert call["native_sha256"] == call["used_sha256"]
        for area in ("center96", "all128"):
            for mode in ("full", "crop"):
                for model in ("T1", "F", "FitP"):
                    for metric in ("psnr", "ssim"):
                        assert math.isfinite(row["scores"][area][mode][model][metric])
        for model in ("T1", "F", "FitP"):
            assert all(math.isfinite(row["full_metrics"][model][m]) for m in ("psnr", "ssim"))
        if row["split"] == "validation64":
            assert row["top"] % 8 == row["left"] % 8 == 0
            assert row["top"] == (item["height"] - 128) // 16 * 8
            assert row["left"] == (item["width"] - 128) // 16 * 8
            ref = refs[row["sample_id"]]
            assert row["reference_t1_match"] is True
            assert row["full_t1_sha256"] == ref["t1_sha256"] and row["shape"] == ref["shape"]
            close(row["full_metrics"]["T1"], ref["metrics"]["T1"], "reference/T1")
            close(row["full_metrics"]["F"], ref["metrics"]["R1F"], "reference/F")
    computed = summarize(rows, cfg["protocol_id"])
    close(computed, read(root / "reports/summary.json"))
    return {"status": "verified", "scope": "CPU consistency and independent scene-equal aggregation; no image-metric recomputation",
            "n_rows": len(rows), "validation_scenes": dict(val_scenes), "checked_sources": source_checks,
            "locked_sha256": sha(root / "config/locked.json"), "rows_sha256": sha(root / "metrics/rows.json"),
            "summary_sha256": sha(root / "reports/summary.json"), "recomputed_summary": computed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.root)
    except Exception:
        result = {"status": "failed", "error": traceback.format_exc()}
    output = args.root / "reports/independent_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "recomputed_summary"}, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "verified" else 1)
