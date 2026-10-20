"""CPU-only provenance archive for the matched single-round collaboration study.

No training or model selection occurs here. A partial archive is explicitly
different from a completed formal experiment. Uses only the Python standard
library, and verifies every member after packaging before publishing ready.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tarfile
import time
from pathlib import Path

SOURCE_SHA = "279d0a329d0efebee9678c2156bf3e7b60c0646e75a5f37f8015c7b100f9e9a2"
FANET_SHA = "20fe6defc1ac37b5d7011bce5b83a43c9e21e6948f36bbb98933a742a7b75a4b"
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
ROOT_OUTPUTS = {"artifacts.tar.gz", "archive_manifest.json", "archive_ready.json"}


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path, default=None):
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def final_checkpoint(root: Path, arm: str) -> Path | None:
    candidates = [
        root / "checkpoints" / f"{arm}_final.pt",
        root / "checkpoints" / arm / "final.pt",
        root / arm / "checkpoints" / "final.pt",
        root / "downstream" / "checkpoints" / f"{arm}_final.pt",
        root / "downstream" / "checkpoints" / arm / "final.pt",
    ]
    return next((path for path in candidates if path.exists()), None)


def completion_audit(root: Path, status: dict, locked: dict) -> dict:
    """Conservative observable completion checks; no claim about effectiveness."""
    reasons = []
    if locked and locked.get("arms") != ["meta"]:
        reasons.append("current user-authorized scope requires meta-only training")
    if status.get("stage") != "completed":
        reasons.append(f"status.stage={status.get('stage', 'missing')}")
    if not (root / "config" / "locked_run.json").exists():
        reasons.append("missing config/locked_run.json")
    lock_receipt_path = next((root / relative for relative in ("lock_receipt.json", "config/lock_receipt.json") if (root / relative).exists()), None)
    if lock_receipt_path is None:
        reasons.append("missing lock_receipt.json")
    elif (root / "config" / "locked_run.json").exists():
        lock_receipt = read_json(lock_receipt_path)
        if lock_receipt.get("locked_sha256") != sha(root / "config" / "locked_run.json"):
            reasons.append("lock receipt does not match archived locked_run.json")
    if status.get("stage") == "completed":
        completed_stages = status.get("completed_stage_names", [])
        for stage in ("preflight", "train", "evaluation"):
            if stage not in completed_stages:
                reasons.append(f"completed controller stage receipt missing: {stage}")
    metrics = {}
    summary_path = root / "downstream" / "metrics" / "summary.json"
    summary = {}
    if summary_path.exists():
        summary = read_json(summary_path)
        counts = summary.get("evaluated_counts", {})
        for split, expected in (("validation", 872), ("test", 188)):
            actual = counts.get(split)
            metrics[split] = {"path": summary_path.relative_to(root).as_posix(), "frames": actual, "expected_frames": expected}
            if actual != expected:
                reasons.append(f"summary/{split} frames={actual}, expected={expected}")
            csv_path = root / "downstream" / "metrics" / f"per_image_{split}.csv"
            if not csv_path.exists():
                reasons.append(f"missing per-image {split} CSV")
            else:
                with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                if len(rows) != expected:
                    reasons.append(f"per-image {split} CSV rows={len(rows)}, expected={expected}")
                if rows and "sample_id" in rows[0] and len({row["sample_id"] for row in rows}) != expected:
                    reasons.append(f"per-image {split} duplicate sample IDs")
            if not (root / "downstream" / "metrics" / f"per_scene_{split}.csv").exists():
                reasons.append(f"missing per-scene {split} CSV")
            split_summary = summary.get(split, {})
            for arm in ("source", "meta"):
                if arm not in split_summary.get("output_metrics", {}):
                    reasons.append(f"summary {split} missing {arm} metrics")
            for pair in ("meta_minus_source",):
                if pair not in split_summary.get("paired_diffs", {}):
                    reasons.append(f"summary {split} missing {pair}")
        if summary.get("tiling") is not False or summary.get("full_native_size_forward") is not True:
            reasons.append("summary reports non-full original-size forward")
        if summary.get("completed") is not True:
            reasons.append("evaluation summary is not completed")
        if summary.get("source_checkpoint_sha256") != SOURCE_SHA:
            reasons.append("evaluation summary source checkpoint SHA mismatch")
        if not (root / "downstream" / "metrics" / "evaluation_context.json").exists():
            reasons.append("missing evaluation_context.json")
    else:
        reasons.append("missing downstream/metrics/summary.json")
    checkpoints = {}
    for arm in ("meta",):
        path = final_checkpoint(root, arm)
        if path is None:
            reasons.append(f"missing {arm} final checkpoint")
            continue
        checkpoints[arm] = {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}
        receipt = summary.get("fixed_final_checkpoints", {}).get(arm, {})
        if receipt.get("sha256") != checkpoints[arm]["sha256"]:
            reasons.append(f"{arm} final checkpoint SHA differs from evaluation receipt")
        if receipt.get("final_step") != locked.get("joint_steps"):
            reasons.append(f"{arm} evaluation final step differs from locked joint_steps")
        if receipt.get("source_sha256") != SOURCE_SHA:
            reasons.append(f"{arm} final checkpoint source receipt mismatch")
        if receipt.get("FANet_used_for_inference") is not False:
            reasons.append(f"{arm} inference must omit FANet")
    required_sources = {"source_checkpoint": SOURCE_SHA, "official_fanet": FANET_SHA}
    sources = {}
    weight_paths = [path for path in (root / "weights").rglob("*") if path.is_file() and not path.is_symlink()] if (root / "weights").exists() else []
    for label, expected in required_sources.items():
        for path in weight_paths:
            actual = sha(path)
            if actual == expected:
                sources[label] = {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": actual}
                break
        if label not in sources:
            reasons.append(f"missing archived {label} with expected SHA256 {expected}")
    return {
        "formal_experiment_completed": not reasons,
        "reasons": reasons,
        "metrics": metrics,
        "final_checkpoints": checkpoints,
        "verified_source_weights": sources,
        "protocol_id": locked.get("protocol_id", status.get("protocol_id")),
        "method_effectiveness": "not_inferred_from_archive_completion",
    }


def inventory(root: Path) -> tuple[list[dict], list[dict]]:
    included, excluded = [], []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        name = relative.as_posix()
        if path.is_symlink():
            excluded.append({"path": name, "reason": "symlink; preserve an ordinary in-root provenance copy"})
            continue
        if not path.is_file():
            continue
        if len(relative.parts) == 1 and path.name in ROOT_OUTPUTS:
            continue
        if any(part in SKIP_PARTS for part in relative.parts):
            continue
        if path.name.endswith((".tmp", ".pyc", ".pyo")):
            continue
        if path.name.endswith((".tar.gz", ".tgz", ".tar", ".zip")):
            excluded.append({"path": name, "reason": "nested archive; separately download and verify", "bytes": path.stat().st_size, "sha256": sha(path)})
            continue
        # Keep runtime *.py and their ordinary copies: frozen evidence must survive
        # archive cache exclusions. Large weights are intentionally retained.
        before = path.stat()
        digest = sha(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"file changed while hashing: {name}")
        included.append({"path": name, "bytes": after.st_size, "sha256": digest})
    return included, excluded


def verify_tar(path: Path, files: list[dict], manifest_bytes: bytes) -> None:
    expected = {entry["path"]: entry for entry in files}
    seen = set()
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if member.name in seen:
                raise RuntimeError(f"duplicate archive member: {member.name}")
            seen.add(member.name)
            if not member.isfile():
                raise RuntimeError(f"non-regular archive member: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"unreadable archive member: {member.name}")
            if member.name == "archive_manifest.json":
                if stream.read() != manifest_bytes:
                    raise RuntimeError("packaged manifest mismatch")
                continue
            if member.name not in expected:
                raise RuntimeError(f"unexpected archive member: {member.name}")
            entry = expected[member.name]
            value = hashlib.sha256()
            size = 0
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                size += len(block)
                value.update(block)
            if size != entry["bytes"] or value.hexdigest() != entry["sha256"]:
                raise RuntimeError(f"packaged file mismatch: {member.name}")
    if seen != set(expected) | {"archive_manifest.json"}:
        raise RuntimeError("archive member coverage mismatch")


def archive(root: Path) -> dict:
    root = root.resolve(strict=True)
    status = read_json(root / "status.json")
    locked = read_json(root / "config" / "locked_run.json")
    audit = completion_audit(root, status, locked)
    budget = read_json(root / "budget.json")
    write_json(root / "archive_completion_audit.json", audit)
    report = root / "RESULTS_ZH.md"
    marker = "<!-- Automatically maintained by archive.py; not an effectiveness analysis. -->"
    if not report.exists() or report.read_text(encoding="utf-8").startswith(marker):
        complete = audit["formal_experiment_completed"]
        report.write_text(
            marker + "\n# 第23项：单轮偏振增强器与PolarFree协作迁移\n\n"
            f"实际运行状态：{status.get('stage', 'unknown')}。正式训练与完整评价完成：{complete}。\n\n"
            "比较同一来源的普通微调与参数高效元学习协作，部署均为单轮增强加PolarFree。"
            "FANet只参与训练；循环、置信、Q、跨轮反馈与动作判断不参与本组前向。\n\n"
            "完整逐图、场景及来源/control/meta配对结果见metrics；归档完成不代表方法有效。"
            "保留单seed、额外元计算、保持代理、多个接口同时适配和历史测试探索限制。\n\n"
            "未完成项：\n" + "\n".join(f"- {reason}" for reason in audit["reasons"]) + "\n",
            encoding="utf-8",
        )
    files, excluded = inventory(root)
    manifest = {
        "schema_version": 1,
        "created_unix": time.time(),
        "status_stage": status.get("stage", "unknown"),
        "formal_experiment_completed": audit["formal_experiment_completed"],
        "protocol_id": audit["protocol_id"],
        "files": files,
        "excluded_files": excluded,
        "nested_archives_require_separate_download": any(item["reason"].startswith("nested archive") for item in excluded),
    }
    write_json(root / "archive_manifest.json", manifest)
    manifest_bytes = (root / "archive_manifest.json").read_bytes()
    temporary = root / "artifacts.tar.gz.tmp"
    with tarfile.open(temporary, "w:gz", compresslevel=1) as target:
        for entry in files:
            target.add(root / entry["path"], arcname=entry["path"], recursive=False)
        target.add(root / "archive_manifest.json", arcname="archive_manifest.json", recursive=False)
    verify_tar(temporary, files, manifest_bytes)
    final = root / "artifacts.tar.gz"
    os.replace(temporary, final)
    ready = {
        "schema_version": 1,
        "created_unix": time.time(),
        "archive_path": final.name,
        "sha256": sha(final),
        "bytes": final.stat().st_size,
        "manifest_sha256": sha(root / "archive_manifest.json"),
        "file_count": len(files),
        "archive_members_verified": True,
        "status_stage": status.get("stage", "unknown"),
        "completed": audit["formal_experiment_completed"],
        "formal_experiment_completed": audit["formal_experiment_completed"],
        "method_effectiveness": "not_inferred_from_archive_completion",
        "incomplete_reasons": audit["reasons"],
        "protocol_id": audit["protocol_id"],
        "original_budget": budget,
        "nested_archives_require_separate_download": manifest["nested_archives_require_separate_download"],
    }
    write_json(root / "archive_ready.json", ready)
    return ready


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = archive(args.root)
    except Exception as error:
        # A failed new package must never leave a stale ready receipt claiming
        # that an older package describes the current run.
        root = args.root.resolve()
        if root.exists():
            write_json(root / "archive_ready.json", {
                "created_unix": time.time(), "archive_members_verified": False,
                "completed": False, "formal_experiment_completed": False,
                "archive_error": f"{type(error).__name__}: {error}",
            })
        raise
    print(json.dumps({"archived": True, "completed": result["completed"], "files": result["file_count"], "bytes": result["bytes"]}), flush=True)


if __name__ == "__main__":
    main()
