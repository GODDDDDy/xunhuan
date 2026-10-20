"""Closeout-only entry point for experiment 38 (implementation repair path).

The worker finished preflight, the full 3000-step three-arm training and the
64-frame original-size screen, but finish() aborted at the independent audit:
the audit's INACTIVE_B expectation omitted the documented dead parameter
blocks.enhancer.de_reduce_chan2.weight (experiment-32 protocol), so the audit
raised before reaching metrics verification. The audit expectation has been
repaired through the registered revision mechanism (config/
implementation_revisions.json, original preserved under config/preserved/).

This script re-runs the exact finish() code path from run_g6_feedback.py:
independent CPU audit -> locked-source verification -> artifact manifest ->
--verify-archive -> success closeout. It performs no network inference, no
training and no evaluation; weights/metrics are only read and hashed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_g6_feedback as worker  # noqa: E402

if __name__ == "__main__":
    plan = worker.read(worker.ROOT / "config/locked.json")
    worker.finish(plan)
    print("CLOSEOUT_OK")
