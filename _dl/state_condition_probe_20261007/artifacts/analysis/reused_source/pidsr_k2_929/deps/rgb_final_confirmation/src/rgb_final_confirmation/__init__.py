"""Frozen-PolarFree Baseline/A0/A1/A2 R2P screening experiment."""

import os
from pathlib import Path

_RUNTIME_ROOT = Path(__file__).resolve().parents[4] / "runtime"
os.environ.setdefault("TORCH_HOME", str(_RUNTIME_ROOT / "cache" / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(_RUNTIME_ROOT / "cache"))
os.environ.setdefault("TMP", str(_RUNTIME_ROOT / "tmp"))
os.environ.setdefault("TEMP", str(_RUNTIME_ROOT / "tmp"))
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(_RUNTIME_ROOT / "pycache"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

__version__ = "0.1.0"
