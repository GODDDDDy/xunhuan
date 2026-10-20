"""Local error ranking adapted from CRL/RankIQA; not a reproduction of either.

Training target: 4x4 pooled RGB MAE of clamped T1. Forward inference needs no GT.
The original error-scale objective is retained; no change to b or C semantics.
"""
import numpy as np
import torch
from torch.nn import functional as F


def scale_loss(b, error):
    return (error.detach() / b + b.log()).mean()


def ranking_loss(b, error, pair_seed, pairs=2048, min_gap=.002):
    """Same-image pairs only; skip near ties, never compare scenes in a pair.

    Hinge margin uses the target log-error gap, capped at one. The constants and
    log-space adaptation are project choices, locked before validation results.
    Use a private CPU RNG so sampling cannot perturb frozen T1's diffusion RNG.
    """
    generator = torch.Generator(device='cpu').manual_seed(int(pair_seed))
    n = b.shape[-2] * b.shape[-1]
    indices = torch.randint(n, (2, pairs), generator=generator).to(b.device)
    pred, target = b.flatten(1).log(), error.detach().flatten(1)
    difference = target[:, indices[0]] - target[:, indices[1]]
    valid = difference.abs() >= min_gap
    gap = ((target[:, indices[0]] + .001).log()
           - (target[:, indices[1]] + .001).log()).abs().clamp(max=1.)
    ordered = difference.sign() * (pred[:, indices[0]] - pred[:, indices[1]])
    loss = F.relu(gap - ordered)
    return (loss * valid).sum() / valid.sum().clamp_min(1), int(valid.sum())


def quality_metrics(prediction, error, spearman):
    """All H-grid cells, per-image metrics. b ascending = C descending.

    Normalized AUSE integrates remaining MAE/original MAE against an oracle on
    removal fractions 0..0.95 (step .01). It is specifically named/defined here,
    rather than assumed numerically interchangeable with optical-flow AUSE.
    """
    b, e = [np.asarray(x, dtype=np.float64).ravel() for x in (prediction, error)]
    assert b.shape == e.shape and np.isfinite(b).all() and np.isfinite(e).all()
    n = len(e)
    predicted_order = np.argsort(b, kind='stable')
    oracle_order = np.argsort(e, kind='stable')
    fractions = np.arange(96) / 100.
    remaining = np.maximum(1, np.ceil(n * (1 - fractions)).astype(int))
    norm = max(float(e.mean()), 1e-12)
    curve = np.cumsum(e[predicted_order])[remaining - 1] / remaining / norm
    oracle = np.cumsum(e[oracle_order])[remaining - 1] / remaining / norm
    # Include the whole tie at the true 80th percentile to avoid arbitrary GT ties.
    threshold = np.quantile(e, .8)
    truth = e >= threshold
    k = max(1, int(np.ceil(.2 * n)))
    selected = predicted_order[-k:]
    caught = int(truth[selected].sum())
    rho = float(spearman(b, e))
    return dict(ause=float(np.trapz(curve - oracle, fractions)),
                spearman_b_error=rho if np.isfinite(rho) else None,
                worst20_precision=caught / k, worst20_recall=caught / int(truth.sum()),
                b_mae=float(np.abs(b - e).mean()),
                scale_loss=float(np.mean(e / b + np.log(b))),
                mean_b=float(b.mean()), mean_error=float(e.mean()),
                sparsification=curve.tolist(), oracle=oracle.tolist())
