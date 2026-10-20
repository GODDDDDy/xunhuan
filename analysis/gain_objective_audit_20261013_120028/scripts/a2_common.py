"""Shared helpers for the stage-A2 native judgement-head diagnostic."""
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from acceptance import resize  # noqa: E402

GRID = 32  # 128 / 4 pooled cells per crop
CROPS = {"judgement": 24, "bias": 8, "safety": 16}


def build_head_inputs(record, head):
    """Reconstruct the exact 50-channel FloatActionCostGate input from a cached record.

    Mirrors gain22.FloatActionCostGate.inputs. Only the trainable projection is
    re-applied here, so feature_projection genuinely receives gradients.
    """
    projection = head.feature_projection(record["feat"])
    ratio = record["dp"].abs().sum(1, keepdim=True) / (record["dabs"].sum(1, keepdim=True) + 1e-8)
    return torch.cat([record["img"], record["t1p"], record["Up"], record["dp"], record["polp"],
                      record["err"], record["conf"], projection, record["state"],
                      record["dsq"], record["dabs"], ratio], 1)


def _rank(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    sorted_values = values[order]
    index = 0
    while index < len(sorted_values):
        end = index
        while end + 1 < len(sorted_values) and sorted_values[end + 1] == sorted_values[index]:
            end += 1
        if end > index:
            ranks[order[index:end + 1]] = (index + end) / 2.0
        index = end + 1
    return ranks


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 3:
        return None
    ra, rb = _rank(a), _rank(b)
    sa, sb = ra.std(), rb.std()
    if sa < 1e-12 or sb < 1e-12:
        return None
    value = float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))
    return value if math.isfinite(value) else None


def sha_array(array):
    array = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    return hashlib.sha256(array.tobytes()).hexdigest()
