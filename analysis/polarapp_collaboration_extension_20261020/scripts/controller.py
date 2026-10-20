"""Single-instance supervision for experiment 43, with a fixed nine-hour deadline.

Waits for its own GPU resource requirement, launches one worker, and preserves
failure evidence. At its deadline it stops only its verified own worker group; no failed worker is retried.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MINIMUM_FREE_MIB = 10400
GPU_INDEX = 0


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


def process_start_ticks(pid):
    try:
        # Field 22 in proc stat; process names can contain spaces.
        text = Path("/proc") / str(pid) / "stat"
        tail = text.read_text().rsplit(")", 1)[1].split()
        return int(tail[19])
    except (OSError, ValueError, IndexError):
        return None


def existing_worker_is_live():
    path = ROOT / "worker_registration.json"
    if not path.exists():
        return False
    receipt = read(path)
    pid = int(receipt.get("pid", -1))
    if pid <= 0:
        return False
    try:
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    expected = str(ROOT / "scripts/run_study.py").encode()
    registered_ticks = receipt.get("start_ticks")
    return (expected in command and registered_ticks is not None
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


from run_study import verify as verify_locked_sources, authorized_deadline


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
        plan = read(ROOT / ("config/locked.json" if (ROOT / "config/locked.json").exists() else "config/methods.json"))
        verify_locked_sources(plan)
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
            raise RuntimeError("Registered experiment43 worker is still alive; no duplicate "
                               "launched. Inspect it before any controller recovery.")
        start = time.time()
        registration = dict(
            pid=os.getpid(), pgid=os.getpgrp(),
            start_ticks=process_start_ticks(os.getpid()),
            time_unix=start, start_unix=start,
            protocol_id=plan["protocol_id"], root=str(ROOT),
            single_instance=True, gpu_index=GPU_INDEX,
            minimum_free_MiB=MINIMUM_FREE_MIB,
            no_automatic_worker_retry=True, deadline_unix=authorized_deadline())
        write(ROOT / "controller_registration.json", registration)
        env = dict(os.environ)
        env.update(
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPYCACHEPREFIX=str(ROOT / "cache/pycache"),
            MPLCONFIGDIR=str(ROOT / "cache/matplotlib"),
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
            CUDA_VISIBLE_DEVICES=str(GPU_INDEX))
        last_heartbeat = 0.0
        while True:
            free, error = free_memory()
            now = time.time()
            if now >= authorized_deadline() - 60:
                write(ROOT / "status.json", dict(stage="budget_expired", time_unix=now, success=False))
                write(ROOT / "closeout.json", dict(success=False, terminal=True, budget_expired=True, time_unix=now))
                return
            waiting = dict(time_unix=now, gpu_index=GPU_INDEX, free_MiB=free,
                           minimum_MiB=MINIMUM_FREE_MIB, deadline_unix=authorized_deadline())
            if error:
                waiting["query_error"] = error
            write(ROOT / "resource_wait.json", waiting)
            if now - last_heartbeat >= 5:
                write(ROOT / "controller_heartbeat.json",
                      dict(time_unix=now, pid=os.getpid(), stage="waiting_for_gpu",
                           protocol_id=plan["protocol_id"], worker_pid=None))
                last_heartbeat = now
            if free >= MINIMUM_FREE_MIB:
                break
            time.sleep(2)
        # Re-check identities immediately before starting the sole worker.
        verify_locked_sources(plan)
        first_path = ROOT / "config/first_worker.json"
        if first_path.exists():
            first = read(first_path)
            assert first["protocol_id"] in [plan["protocol_id"], plan.get("parent_protocol_id")]
        else:
            first = dict(protocol_id=plan["protocol_id"],
                         start_unix=time.time(), deadline_unix=authorized_deadline())
            write(first_path, first)
        assert first["deadline_unix"] == 1792462065.0
        assert time.time() < authorized_deadline(), "Fixed worker deadline expired"
        with (ROOT / "logs/worker.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen(
                [sys.executable, str(ROOT / "scripts/run_study.py")],
                stdout=log, stderr=subprocess.STDOUT, env=env,
                cwd=str(ROOT), start_new_session=True)
            write(ROOT / "worker_registration.json",
                  dict(pid=child.pid, pgid=child.pid,
                       start_ticks=process_start_ticks(child.pid),
                       time_unix=time.time(), root=str(ROOT),
                       command=[sys.executable, str(ROOT / "scripts/run_study.py")],
                       protocol_id=plan["protocol_id"], deadline_unix=authorized_deadline()))
            while child.poll() is None:
                locked = ROOT / "config/locked.json"
                if locked.exists():
                    plan = read(locked)
                if time.time() >= authorized_deadline() - 10:
                    registered = read(ROOT / "worker_registration.json")
                    assert registered["pid"] == child.pid and process_start_ticks(child.pid) == registered["start_ticks"]
                    assert os.getpgid(child.pid) == child.pid
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=max(.1, authorized_deadline()-time.time()))
                    except subprocess.TimeoutExpired:
                        if child.poll() is None:
                            assert process_start_ticks(child.pid) == registered["start_ticks"]
                            os.killpg(child.pid, signal.SIGKILL)
                    write(ROOT / "closeout.json", dict(success=False, terminal=True, budget_expired=True, time_unix=time.time()))
                    write(ROOT / "status.json", dict(stage="budget_expired", success=False, time_unix=time.time()))
                    break
                write(ROOT / "controller_heartbeat.json",
                      dict(time_unix=time.time(), pid=os.getpid(),
                           worker_pid=child.pid, stage="worker_running",
                           protocol_id=plan["protocol_id"]))
                time.sleep(5)
            exit_code = child.returncode
        write(ROOT / "worker_exit.json",
              dict(time_unix=time.time(), pid=child.pid, exit_code=exit_code,
                   protocol_id=plan["protocol_id"], automatic_retry=False))
        if closeout_path.exists() and read(closeout_path).get("budget_expired"):
            print("Fixed nine-hour budget reached; state preserved; no retry.", flush=True)
            return
        if exit_code != 0:
            raise RuntimeError("Worker exited with code " + str(exit_code)
                               + "; inspect preserved failure receipts. No automatic retry.")
        assert closeout_path.is_file(), "Worker exited without closeout"
        closeout = read(closeout_path)
        assert closeout.get("success") is True, "Worker closeout is not successful"
        assert (ROOT / "reports/final_review_closeout.json").is_file(), \
            "Missing final review closeout"
        write(ROOT / "controller_heartbeat.json",
              dict(time_unix=time.time(), pid=os.getpid(), worker_pid=child.pid,
                   stage="completed", protocol_id=plan["protocol_id"],
                   worker_exit_code=0, closeout_success=True))
        print(json.dumps(dict(controller_completed=True,
                              protocol_id=plan["protocol_id"],
                              exploratory_only=True, closeout_success=True)), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        prior_status = None
        status_path = ROOT / "status.json"
        if status_path.exists():
            try:
                prior_status = read(status_path)
            except (OSError, ValueError):
                prior_status = dict(unreadable=True)
        failure = dict(time_unix=time.time(), pid=os.getpid(),
                       traceback=traceback.format_exc(), success=False,
                       automatic_retry=False, previous_worker_status=prior_status)
        receipt = ROOT / "failures" / (
            "controller_" + str(time.time_ns()) + ".json")
        write(receipt, failure)
        write(ROOT / "controller_failure.json", failure)
        write(status_path, dict(stage="controller_failed",
                                time_unix=time.time(), success=False,
                                automatic_retry=False, failure_receipt=str(receipt),
                                previous_worker_status=prior_status))
        raise





