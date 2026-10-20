"""CPU-only safety, immutable budget and evidence helpers for the single-round study.

This module intentionally never imports torch. GPU workers must call guard before
initialization and at safe checkpoint boundaries, including after every update.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("POLARAPP_RUN_ROOT", Path(__file__).resolve().parents[1])).resolve()


def atomic_json(path: str | Path, value: Any) -> None:
    """Write and fsync JSON, then atomically publish it in the same directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


json_save = atomic_json


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_budget(root: str | Path | None = None) -> dict[str, Any]:
    root = Path(root or ROOT).resolve()
    budget = read_json(root / "budget.json")
    receipt = read_json(root / "budget_receipt.json")
    if sha(root / "budget.json") != receipt["budget_sha256"]:
        raise RuntimeError("Immutable budget file differs from its original receipt")
    if budget["run_id"] != receipt["run_id"] or str(root) != budget["root"]:
        raise RuntimeError("Budget belongs to another run or directory")
    start = float(budget["start_unix"])
    deadline = float(budget["deadline_unix"])
    training_end = float(budget["training_deadline_unix"])
    if abs(deadline - start - 86400.0) > 1e-6:
        raise RuntimeError("Original 24-hour budget was altered")
    if abs(deadline - training_end - 14400.0) > 1e-6:
        raise RuntimeError("Original four-hour evaluation reserve was altered")
    if not start < training_end < deadline:
        raise RuntimeError("Invalid immutable stage deadlines")
    return budget


def read_locked_config(root: str | Path | None = None) -> dict[str, Any]:
    root = Path(root or ROOT).resolve()
    path = root / "config" / "locked_run.json"
    receipt = read_json(root / "config" / "lock_receipt.json")
    expected = receipt.get("locked_sha256", receipt.get("locked_run_sha256"))
    if not expected or sha(path) != expected:
        raise RuntimeError("Locked configuration hash does not match its receipt")
    config = read_json(path)
    if config.get("protocol_id") != receipt.get("protocol_id"):
        raise RuntimeError("Locked protocol identifier does not match its receipt")
    return config


def guard(stage: str = "train", root: str | Path | None = None, cpu: bool = False) -> dict[str, Any]:
    """Raise before unauthorized GPU work; CPU evaluation may finish after cutoff.

    For training, workers should catch the training-deadline exception at a safe
    boundary and atomically save the same protocol's endpoint before returning.
    The independent watchdog enforces the absolute GPU cutoff regardless.
    """
    budget = read_budget(root)
    now = time.time()
    if stage not in ("train", "evaluation", "eval", "preflight"):
        raise ValueError(f"Unknown guarded stage: {stage}")
    if stage == "train" and now >= budget["training_deadline_unix"]:
        raise RuntimeError("Original training deadline reached; save endpoint without more updates")
    if not cpu and now >= budget["deadline_unix"]:
        raise RuntimeError("Original absolute GPU deadline reached; GPU use is prohibited")
    return budget


def status(stage: str, root: str | Path | None = None, **fields: Any) -> dict[str, Any]:
    value = {"stage": stage, "time_unix": time.time(), **fields}
    atomic_json(Path(root or ROOT) / "status.json", value)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)
    return value


def process_identity(pid: int) -> dict[str, Any] | None:
    """Linux /proc identity prevents killing a recycled PID/process group."""
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        if fields[0] in ("Z", "X"):
            return None
        return {"pid": int(pid), "pgid": int(fields[2]), "session": int(fields[3]),
                "start_ticks": int(fields[19]),
                "cmdline": Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")}
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return None


def identity_matches(expected: dict[str, Any] | None) -> bool:
    if not expected:
        return False
    observed = process_identity(int(expected["pid"]))
    return observed is not None and all(observed[key] == expected[key] for key in ("pid", "pgid", "start_ticks"))


def register_gpu(pid: int, stage: str, root: str | Path | None = None) -> dict[str, Any]:
    root = Path(root or ROOT).resolve()
    budget = guard("train" if stage == "train" else "preflight" if stage == "preflight" else "evaluation", root)
    identity = process_identity(pid)
    if identity is None or identity["pgid"] != pid:
        raise RuntimeError("GPU worker must be a live independently created session/process group")
    owner = process_identity(os.getpid())
    if owner and identity["pgid"] == owner["pgid"]:
        raise RuntimeError("CPU controller must not share its GPU worker process group")
    value = {"active": True, "stage": stage, "registered_unix": time.time(),
             "run_id": budget["run_id"], "deadline_unix": budget["deadline_unix"], "identity": identity}
    atomic_json(root / "active_gpu.json", value)
    return value


def unregister_gpu(pid: int, root: str | Path | None = None, **extra: Any) -> None:
    root = Path(root or ROOT)
    path = root / "active_gpu.json"
    if path.exists():
        value = read_json(path)
        if value.get("identity", {}).get("pid") != pid:
            raise RuntimeError("Refusing to clear a different GPU worker's registration")
        value.update(active=False, finished_unix=time.time(), **extra)
        atomic_json(path, value)
