"""Independent CPU watchdog: enforce only this run's absolute GPU cutoff."""
from __future__ import annotations

import argparse
import fcntl
import os
import signal
import time
from pathlib import Path

from runtime import atomic_json, identity_matches, process_identity, read_budget, read_json, sha


def run(root: Path, poll_seconds: float = 2.0) -> None:
    budget = read_budget(root)
    self_identity = process_identity(os.getpid())
    if self_identity is None:
        raise RuntimeError("Independent watchdog requires Linux /proc")
    receipt_path = root / "watchdog.json"
    previous = read_json(receipt_path) if receipt_path.exists() else {}
    if identity_matches(previous.get("identity")) and previous["identity"]["pid"] != os.getpid():
        raise RuntimeError("Another live independent watchdog is already registered")
    base = {"identity": self_identity, "pid": os.getpid(), "run_id": budget["run_id"],
            "budget_sha256": sha(root / "budget.json"), "deadline_unix": budget["deadline_unix"],
            "started_unix": time.time(), "independent": True}
    while True:
        # Reading the original receipt on every iteration detects unauthorized edits.
        try:
            checked = read_budget(root)
            budget_valid = checked["run_id"] == budget["run_id"]
        except Exception as error:
            budget_valid = False
            atomic_json(root / "diagnostics" / "watchdog_budget_error.json",
                        {"time_unix": time.time(), "error": str(error), "original_budget": budget})
        now = time.time()
        active_path = root / "active_gpu.json"
        active = read_json(active_path) if active_path.exists() else {"active": False}
        controller_path = root / "controller.json"
        controller = read_json(controller_path) if controller_path.exists() else {}
        current = {**base, "heartbeat_unix": now, "state": "monitoring",
                   "controller_identity": controller.get("identity"),
                   "registered_controller_pid": controller.get("identity", {}).get("pid"),
                   "registered_controller_pgid": controller.get("identity", {}).get("pgid"),
                   "active_gpu": active.get("identity") if active.get("active") else None,
                   "active_stage": active.get("stage") if active.get("active") else None}
        expired = now >= budget["deadline_unix"]
        if expired or not budget_valid:
            killed = False
            if active.get("active"):
                identity = active.get("identity")
                if active.get("run_id") == budget["run_id"] and identity_matches(identity):
                    if identity["pgid"] == self_identity["pgid"]:
                        raise RuntimeError("Watchdog registration accidentally points to itself")
                    if identity["pgid"] == controller.get("identity", {}).get("pgid"):
                        raise RuntimeError("GPU registration accidentally points to CPU controller")
                    try:
                        os.killpg(identity["pgid"], signal.SIGKILL)
                        killed = True
                    except ProcessLookupError:
                        pass
            event = {"time_unix": now, "reason": "absolute_gpu_deadline" if expired else "immutable_budget_changed",
                     "run_id": budget["run_id"], "deadline_unix": budget["deadline_unix"],
                     "registered_gpu": active, "gpu_group_killed": killed, "cpu_controller_killed": False}
            atomic_json(root / "diagnostics" / "watchdog_cutoff.json", event)
            atomic_json(receipt_path, {**current, "state": "deadline_enforced", "event": event})
            return
        if controller.get("finished") and not active.get("active"):
            atomic_json(receipt_path, {**current, "state": "controller_finished"})
            return
        atomic_json(receipt_path, current)
        time.sleep(min(poll_seconds, max(0.01, budget["deadline_unix"] - time.time())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()
    with (args.root.resolve() / "watchdog.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A live independent watchdog already owns this run")
        run(args.root.resolve(), max(0.25, args.poll_seconds))
