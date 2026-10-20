"""Single-instance experiment37 controller; resource wait is outside worker budget.

It launches one worker, never terminates GPU jobs, and never retries a failure.
A deliberate same-protocol recovery retains the original worker deadline.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MINIMUM_FREE_MIB = 11500
GPU_INDEX = 0
WORKER = ROOT / "scripts/run_state_source.py"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_id(value, field):
    unsigned = dict(value)
    unsigned.pop(field, None)
    return hashlib.sha256(json.dumps(
        unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def process_start_ticks(pid):
    try:
        tail = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return int(tail[19])
    except (OSError, ValueError, IndexError):
        return None


def existing_worker_is_live():
    path = ROOT / "worker_registration.json"
    if not path.exists():
        return False
    receipt = read(path)
    pid = int(receipt.get("pid") or -1)
    if pid <= 0:
        return False
    try:
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    registered_ticks = receipt.get("start_ticks")
    return (str(WORKER).encode() in command and registered_ticks is not None
            and process_start_ticks(pid) == int(registered_ticks))


def free_memory():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--id=" + str(GPU_INDEX),
             "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=10)
        return int(output.strip().splitlines()[0]), None
    except (subprocess.SubprocessError, OSError, ValueError, IndexError) as error:
        return 0, str(error)


def verify_sources(plan):
    assert plan["scientific_id"] == canonical_id(plan, "scientific_id"), "Scientific ID changed"
    assert plan["experiment"] == 37 and plan["arms"] == ["A", "B"]
    assert plan["steps"] == 3000 and plan["exploratory_only"] is True
    assert plan["max_wall_seconds"] == 43200
    codes = {record["path"]: record for record in plan["code_sources"]}
    assert len(codes) == len(plan["code_sources"]), "Duplicate locked code paths"
    replacements = {}
    revisions_path = ROOT / "config/implementation_revisions.json"
    if revisions_path.exists():
        revisions = read(revisions_path)
        assert isinstance(revisions, list), "Implementation revisions must be a list"
        for revision in revisions:
            assert revision["scientific_id"] == plan["scientific_id"], "Revision science mismatch"
            for replacement in revision["replacements"]:
                path = replacement["path"]
                assert path in codes, "Only locked code_sources may be repaired: " + path
                assert replacement["original_sha256"] == codes[path]["sha256"], path
                assert sha(replacement["preserved_original_path"]) == replacement["original_sha256"], path
                replacements[path] = replacement
    for record in plan["source_receipts"]:
        assert sha(record["path"]) == record["sha256"], record["path"]
    for record in plan["code_sources"]:
        expected = replacements.get(record["path"], {}).get("repaired_sha256", record["sha256"])
        assert sha(record["path"]) == expected, record["path"]
    assert sha(plan["schedule3000_path"]) == plan["schedule_sha256"]
    assert sha(plan["cache_items_path"]) == plan["cache_items_sha256"]
    lock_path = ROOT / "config/locked.json"
    if lock_path.exists():
        lock = read(lock_path)
        assert lock["scientific_id"] == plan["scientific_id"]
        assert lock["proposal_sha256"] == sha(ROOT / "config/proposal.json")
        assert lock["protocol_id"] == canonical_id(lock, "protocol_id")
        assert lock["steps"] == 3000 and lock["arms"] == ["A", "B"]
        assert lock["preflight_sha256"] == sha(ROOT / "config/preflight.json")


def deadline(plan):
    path = ROOT / "worker_registration.json"
    if not path.exists():
        return None
    receipt = read(path)
    assert receipt["scientific_id"] == plan["scientific_id"], "Worker science mismatch"
    start = float(receipt["start_unix"])
    end = float(receipt["deadline_unix"])
    assert abs(end - start - plan["max_wall_seconds"]) < 1e-6, "Worker deadline changed"
    return end


def main():
    for folder in ["logs", "reports", "failures", "cache"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "controller.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Existing controller owns this experiment; no duplicate started.", flush=True)
        return
    with lock:
        plan = read(ROOT / "config/proposal.json")
        verify_sources(plan)
        closeout_path = ROOT / "closeout.json"
        if closeout_path.exists():
            closeout = read(closeout_path)
            if closeout.get("success") is True:
                print("Successful closeout preserved; no worker started.", flush=True)
                return
            if closeout.get("terminal") is True:
                raise RuntimeError("Terminal closeout prohibits automatic restart")
        if (ROOT / "controller_failure.json").exists():
            raise RuntimeError("Prior controller failure requires inspected same-protocol "
                               "recovery; automatic retry is disabled")
        if existing_worker_is_live():
            raise RuntimeError("Registered experiment37 worker is alive; no duplicate launched")
        end = deadline(plan)
        if end is not None and time.time() >= end:
            raise RuntimeError("Original worker deadline elapsed; no worker started or killed")
        start = time.time()
        registration = dict(
            pid=os.getpid(), pgid=os.getpgrp(), start_ticks=process_start_ticks(os.getpid()),
            time_unix=start, start_unix=start, scientific_id=plan["scientific_id"],
            root=str(ROOT), single_instance=True, gpu_index=GPU_INDEX,
            minimum_free_MiB=MINIMUM_FREE_MIB, no_automatic_worker_retry=True,
            worker_deadline_unix=end, waiting_before_first_worker_excluded=True)
        write(ROOT / "controller_registration.json", registration)
        env = dict(os.environ)
        env.update(
            PYTHONDONTWRITEBYTECODE="1", PYTHONPYCACHEPREFIX=str(ROOT / "cache/pycache"),
            MPLCONFIGDIR=str(ROOT / "cache/matplotlib"),
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", CUBLAS_WORKSPACE_CONFIG=":4096:8",
            OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", CUDA_VISIBLE_DEVICES=str(GPU_INDEX))
        last_heartbeat = 0.0
        while True:
            now = time.time()
            if end is not None and now >= end:
                raise RuntimeError("Original worker deadline elapsed during recovery resource wait")
            free, error = free_memory()
            waiting = dict(time_unix=now, gpu_index=GPU_INDEX, free_MiB=free,
                           minimum_MiB=MINIMUM_FREE_MIB, worker_deadline_unix=end,
                           initial_resource_wait_outside_budget=end is None)
            if error:
                waiting["query_error"] = error
            write(ROOT / "resource_wait.json", waiting)
            if now - last_heartbeat >= 5:
                write(ROOT / "controller_heartbeat.json",
                      dict(time_unix=now, pid=os.getpid(), stage="waiting_for_gpu",
                           scientific_id=plan["scientific_id"], worker_pid=None))
                last_heartbeat = now
            if free >= MINIMUM_FREE_MIB:
                break
            time.sleep(2)
        verify_sources(plan)
        assert not existing_worker_is_live(), "Worker appeared while controller waited"
        reg_path = ROOT / "worker_registration.json"
        if reg_path.exists():
            worker_registration = read(reg_path)
            end = deadline(plan)
            assert time.time() < end, "Original worker deadline elapsed"
            worker_registration["recovery_launch_unix"] = time.time()
        else:
            worker_start = time.time()
            worker_registration = dict(
                pid=None, pgid=None, start_ticks=None, start_unix=worker_start,
                time_unix=worker_start, deadline_unix=worker_start + plan["max_wall_seconds"],
                scientific_id=plan["scientific_id"], root=str(ROOT),
                command=[sys.executable, str(WORKER)], launching=True)
        write(reg_path, worker_registration)
        with (ROOT / "logs/worker.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen([sys.executable, str(WORKER)],
                                     stdout=log, stderr=subprocess.STDOUT, env=env,
                                     cwd=str(ROOT), start_new_session=True)
            current = read(reg_path)
            current.update(pid=child.pid, pgid=child.pid,
                           start_ticks=process_start_ticks(child.pid), launching=False,
                           process_launch_unix=time.time())
            write(reg_path, current)
            while child.poll() is None:
                write(ROOT / "controller_heartbeat.json",
                      dict(time_unix=time.time(), pid=os.getpid(), worker_pid=child.pid,
                           stage="worker_running", scientific_id=plan["scientific_id"],
                           worker_deadline_unix=current["deadline_unix"]))
                # The worker checks its own deadline and checkpoints. The controller
                # never kills it or any other GPU process, including after deadline.
                time.sleep(5)
            exit_code = child.returncode
        write(ROOT / "worker_exit.json",
              dict(time_unix=time.time(), pid=child.pid, exit_code=exit_code,
                   scientific_id=plan["scientific_id"], automatic_retry=False))
        if exit_code != 0:
            raise RuntimeError("Worker exited with code " + str(exit_code)
                               + "; inspect preserved failure receipts. No automatic retry.")
        assert closeout_path.is_file(), "Worker exited without closeout"
        closeout = read(closeout_path)
        assert closeout.get("success") is True, "Worker closeout is not successful"
        assert (ROOT / "reports/final_review_closeout.json").is_file(), "Missing final review closeout"
        verify_sources(plan)
        write(ROOT / "controller_heartbeat.json",
              dict(time_unix=time.time(), pid=os.getpid(), worker_pid=child.pid,
                   stage="completed", scientific_id=plan["scientific_id"],
                   worker_exit_code=0, closeout_success=True))
        print(json.dumps(dict(controller_completed=True, scientific_id=plan["scientific_id"],
                              exploratory_only=True, closeout_success=True)), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        previous = None
        status_path = ROOT / "status.json"
        if status_path.exists():
            try:
                previous = read(status_path)
            except (OSError, ValueError):
                previous = dict(unreadable=True)
        failure = dict(time_unix=time.time(), pid=os.getpid(), traceback=traceback.format_exc(),
                       success=False, automatic_retry=False, previous_worker_status=previous)
        receipt = ROOT / "failures" / ("controller_" + str(time.time_ns()) + ".json")
        write(receipt, failure)
        write(ROOT / "controller_failure.json", failure)
        write(status_path, dict(stage="controller_failed", time_unix=time.time(), success=False,
                               automatic_retry=False, failure_receipt=str(receipt),
                               previous_worker_status=previous))
        raise