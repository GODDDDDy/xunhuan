"""CPU coordination of preflight, two matched arms, full evaluation and archival.

GPU children have independent sessions and identities recorded before they start
work. The immutable 24-hour budget is created once, before the first preflight.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

from runtime import (atomic_json, guard, identity_matches, process_identity, read_budget,
                     read_json, read_locked_config, register_gpu, sha, status, unregister_gpu)


def create_budget(root: Path) -> dict:
    """Exclusive publication prevents accidental replacement on recovery."""
    now = time.time()
    budget = {"run_id": str(uuid.uuid4()), "root": str(root), "start_unix": now,
              "gpu_start_unix": now, "deadline_unix": now + 86400.0,
              "training_deadline_unix": now + 72000.0, "evaluation_reserve_seconds": 14400.0,
              "duration_seconds": 86400.0, "scope": "new user-authorized paired frozen/partial collaboration; independent24h; parent not restarted"}
    temporary = root / (".budget." + budget["run_id"] + ".tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(budget, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and refuses to replace an existing budget.
        os.link(temporary, root / "budget.json")
    finally:
        temporary.unlink(missing_ok=True)
    atomic_json(root / "budget_receipt.json", {"run_id": budget["run_id"], "created_unix": now,
                                               "budget_sha256": sha(root / "budget.json")})
    return budget


def environment(root: Path, gpu: bool) -> dict:
    local = dict(os.environ)
    local.update(POLARAPP_RUN_ROOT=str(root), CUDA_VISIBLE_DEVICES="0" if gpu else "",
                 PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TMPDIR=str(root / "runtime" / "tmp"),
                 XDG_CACHE_HOME=str(root / "runtime" / "cache"), TORCH_HOME=str(root / "runtime" / "cache" / "torch"))
    local["PYTHONPATH"] = os.pathsep.join([str(root), str(root / "scripts"), str(root / "source_snapshot"),
                                         local.get("PYTHONPATH", "")])
    Path(local["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    return local


def require_watchdog(root: Path, budget: dict, gpu_pid: int | None = None) -> dict:
    path = root / "watchdog.json"
    receipt = read_json(path) if path.exists() else {}
    if not identity_matches(receipt.get("identity")):
        raise RuntimeError("Independent watchdog is not alive")
    if receipt.get("run_id") != budget["run_id"] or receipt.get("deadline_unix") != budget["deadline_unix"]:
        raise RuntimeError("Watchdog does not acknowledge the original run/deadline")
    if receipt.get("budget_sha256") != sha(root / "budget.json") or not receipt.get("independent"):
        raise RuntimeError("Watchdog did not acknowledge the immutable budget")
    if time.time() - float(receipt.get("heartbeat_unix", 0)) > 15:
        raise RuntimeError("Independent watchdog heartbeat is stale")
    controller = process_identity(os.getpid())
    if receipt.get("controller_identity") != controller:
        raise RuntimeError("Watchdog has not yet acknowledged the current CPU controller identity")
    if gpu_pid is not None and (receipt.get("active_gpu") or {}).get("pid") != gpu_pid:
        raise RuntimeError("Watchdog has not yet acknowledged the current GPU worker group")
    return receipt


def ensure_watchdog(root: Path, python: str, budget: dict) -> None:
    existing = read_json(root / "watchdog.json") if (root / "watchdog.json").exists() else {}
    if identity_matches(existing.get("identity")):
        end = time.monotonic() + 12
        while time.monotonic() < end and identity_matches(existing.get("identity")):
            try:
                require_watchdog(root, budget)
                return
            except RuntimeError:
                time.sleep(0.2)
        if identity_matches(existing.get("identity")):
            raise RuntimeError("Existing watchdog did not acknowledge the recovering CPU controller")
    with (root / "logs" / "watchdog.log").open("a", encoding="utf-8") as output:
        watcher = subprocess.Popen([python, "-u", str(root / "scripts" / "watchdog.py"), "--root", str(root)],
                                   cwd=root, env=environment(root, False), stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    end = time.monotonic() + 12
    while time.monotonic() < end:
        if watcher.poll() is not None:
            raise RuntimeError(f"Independent watchdog failed during registration: exit {watcher.returncode}")
        try:
            require_watchdog(root, budget)
            return
        except RuntimeError:
            time.sleep(0.2)
    raise RuntimeError("Independent watchdog failed to register before GPU dispatch")


def run_stage(root: Path, python: str, name: str, args: list[str], gpu: bool, budget: dict) -> dict:
    guard("train" if name == "train" else "preflight" if name == "preflight" else "evaluation", root, cpu=not gpu)
    if gpu:
        require_watchdog(root, budget)
    status(name + "_running", root=root, controller_pid=os.getpid(), deadline_unix=budget["deadline_unix"])
    begin = time.time()
    with (root / "logs" / f"{name}.log").open("a", encoding="utf-8") as output:
        # Worker waits on a pipe token until identity registration is complete.
        # This avoids a dispatch race where CUDA starts before the watchdog knows its group.
        command = [python, "-u", "-c",
                   "import os,sys; token=sys.stdin.buffer.read(1); "
                   "assert token==b'G', 'dispatch registration failed'; "
                   "os.execv(sys.executable,[sys.executable]+sys.argv[1:])", *args]
        worker = subprocess.Popen(command, cwd=root, env=environment(root, gpu), stdin=subprocess.PIPE,
                                  stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        registered = False
        try:
            if gpu:
                register_gpu(worker.pid, name, root)
                registered = True
                acknowledgment_end = time.monotonic() + 8
                while True:
                    guard("train" if name == "train" else "preflight" if name == "preflight" else "evaluation", root)
                    try:
                        require_watchdog(root, budget, worker.pid)
                        break
                    except RuntimeError:
                        if time.monotonic() >= acknowledgment_end:
                            raise RuntimeError("Independent watchdog did not acknowledge GPU group before dispatch")
                        time.sleep(0.1)
            worker.stdin.write(b"G")
            worker.stdin.flush()
            worker.stdin.close()
            while worker.poll() is None:
                if gpu:
                    if time.time() >= budget["deadline_unix"]:
                        # Do not depend on a single scheduler tick of the watchdog.
                        try:
                            os.killpg(worker.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    else:
                        try:
                            require_watchdog(root, budget, worker.pid)
                        except Exception:
                            os.killpg(worker.pid, signal.SIGKILL)
                            raise
                time.sleep(1.0)
            if worker.returncode:
                raise RuntimeError(f"{name} exited with code {worker.returncode}; see logs/{name}.log")
        finally:
            if worker.poll() is None:
                try:
                    os.killpg(worker.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                worker.wait()
            if registered:
                unregister_gpu(worker.pid, root, returncode=worker.returncode)
    return {"started_unix": begin, "finished_unix": time.time(), "returncode": worker.returncode,
            "command": args, "gpu": gpu, "budget_sha256": sha(root / "budget.json")}


def run(root: Path, python: str, resume: bool, cpu_evaluation: bool) -> int:
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    (root / "diagnostics").mkdir(exist_ok=True)
    with (root / "controller.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A live controller already owns this run; duplicate start prevented")
        if resume:
            budget = read_budget(root)
        else:
            if (root / "budget.json").exists() or (root / "config" / "locked_run.json").exists():
                raise RuntimeError("Existing run cannot be reset; use explicit --resume with its original budget")
            budget = create_budget(root)
        active = read_json(root / "active_gpu.json") if (root / "active_gpu.json").exists() else {}
        if active.get("active") and identity_matches(active.get("identity")):
            raise RuntimeError("A previous GPU worker is still alive; do not duplicate it")
        controller = {"identity": process_identity(os.getpid()), "run_id": budget["run_id"],
                      "started_unix": time.time(), "deadline_unix": budget["deadline_unix"],
                      "resume": resume, "finished": False, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        if controller["identity"] is None:
            raise RuntimeError("Formal controller requires Linux /proc")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("Launch CPU controller with CUDA_VISIBLE_DEVICES='' explicitly")
        atomic_json(root / "controller.json", controller)
        journal_path = root / "controller_stage_receipts.json"
        journal = read_json(journal_path) if journal_path.exists() else {"run_id": budget["run_id"], "stages": {}}
        if journal["run_id"] != budget["run_id"]:
            raise RuntimeError("Controller journal belongs to another budget")
        failure = None
        try:
            if time.time() < budget["deadline_unix"]:
                ensure_watchdog(root, python, budget)
            if "preflight" not in journal["stages"]:
                journal["stages"]["preflight"] = run_stage(root, python, "preflight",
                    ["-u", str(root / "scripts" / "run_training.py"), "--root", str(root), "--stage", "preflight"], True, budget)
                read_locked_config(root)
                atomic_json(journal_path, journal)
            locked = read_locked_config(root)
            status("plan_locked", root=root, protocol_id=locked["protocol_id"],
                   deadline_unix=budget["deadline_unix"], training_deadline_unix=budget["training_deadline_unix"])
            if "train" not in journal["stages"]:
                journal["stages"]["train"] = run_stage(root, python, "train",
                    ["-u", str(root / "scripts" / "run_training.py"), "--root", str(root), "--stage", "train", "--arm", "both"], True, budget)
                atomic_json(journal_path, journal)
            if "evaluation" not in journal["stages"]:
                journal["stages"]["evaluation"] = run_stage(root, python, "evaluation",
                    ["-u", str(root / "recovery" / "evaluate_existing_final.py"), "--root", str(root),
                     "--device", "cpu" if cpu_evaluation else "cuda:0"], not cpu_evaluation, budget)
                atomic_json(journal_path, journal)
            status("completed", root=root, protocol_id=locked["protocol_id"],
                   completed_stage_names=list(journal["stages"]), deadline_unix=budget["deadline_unix"])
        except BaseException:
            failure = traceback.format_exc()
            atomic_json(root / "diagnostics" / f"controller_failure_{time.time_ns()}.json",
                        {"time_unix": time.time(), "error": failure, "budget": budget, "journal": journal})
            status("failed", root=root, error=failure, deadline_unix=budget["deadline_unix"])
        finally:
            archive_error = None
            archive_returncode = -1
            try:
                with (root / "logs" / "archive.log").open("a", encoding="utf-8") as output:
                    archived = subprocess.run([python, "-u", str(root / "recovery" / "archive_live_snapshot.py"), "--root", str(root)],
                                              cwd=root, env=environment(root, False), stdout=output, stderr=subprocess.STDOUT)
                    archive_returncode = archived.returncode
            except Exception:
                archive_error = traceback.format_exc()
            controller.update(finished=True, finished_unix=time.time(),
                              result="failed" if failure else "completed", archive_returncode=archive_returncode)
            atomic_json(root / "controller.json", controller)
            if archive_returncode:
                status("archive_failed", root=root, previous_result=controller["result"],
                       error=archive_error or f"CPU archival exited {archive_returncode}")
            return 1 if failure or archive_returncode else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu-evaluation", action="store_true")
    args = parser.parse_args()
    sys.exit(run(args.root.resolve(), args.python, args.resume, args.cpu_evaluation))
