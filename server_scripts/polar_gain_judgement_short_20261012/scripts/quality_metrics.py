"""Evaluate confidence versus GT after prediction; GT never enters inference."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F


def spearman_numpy(a, b):
    # NumPy-only average ranks for ties; Pearson correlation of these ranks.
    # This avoids adding SciPy to the established server environment.
    def rank(values):
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        ranks = np.cumsum(counts) - counts + (counts - 1) / 2 + 1
        return ranks[inverse]
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


class QualityCollector:
    def __init__(self, root, total, tau):
        self.root, self.total, self.tau = root, total, tau
        self.split = 'validation' if total == 872 else 'test' if total == 188 else 'proxy'
        self.started = time.time()
        self.count = np.zeros(10, dtype=np.int64)
        self.sums = np.zeros((5, 10), dtype=np.float64)
        self.rows, self.saved_scenes = [], set()

    def observe(self, out, batch):
        with torch.no_grad():
            e1 = F.avg_pool2d((out.t1.clamp(0, 1) - batch['target']).abs().mean(1, keepdim=True), 4)
            e2 = F.avg_pool2d((out.t2.clamp(0, 1) - batch['target']).abs().mean(1, keepdim=True), 4)
            update = F.avg_pool2d((out.refined_polar - out.first.refined_polar).abs().mean(1, keepdim=True), 4)
            b = out.error_scale.detach()
            c = torch.exp(-b / self.tau)
            arrays = [x.float().cpu().numpy().ravel() for x in (c, b, e1, e2, update)]
        bins = np.minimum((arrays[0] * 10).astype(np.int64), 9)
        self.count += np.bincount(bins, minlength=10)
        for i, values in enumerate(arrays):
            self.sums[i] += np.bincount(bins, weights=values, minlength=10)
        stride = max(len(arrays[0]) // 4096, 1)
        corr = spearman_numpy(arrays[0][::stride], arrays[2][::stride])
        self.rows.append(dict(sample_id=batch['sample_id'][0], scene=batch['scene'][0],
                              confidence_mean=float(arrays[0].mean()), error_mean=float(arrays[2].mean()),
                              predicted_error_mean=float(arrays[1].mean()),
                              spearman_confidence_error=corr if np.isfinite(corr) else None))
        scene = batch['scene'][0]
        if self.split != 'proxy' and scene not in self.saved_scenes:
            directory = self.root / 'metrics/confidence_examples' / self.split
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory / f'scene_{scene}.npz',
                                sample_id=batch['sample_id'][0], confidence=arrays[0].reshape(e1.shape[-2:]),
                                predicted_error=arrays[1].reshape(e1.shape[-2:]),
                                error1=arrays[2].reshape(e1.shape[-2:]), error2=arrays[3].reshape(e1.shape[-2:]))
            self.saved_scenes.add(scene)
        if len(self.rows) % 20 == 0 or len(self.rows) == self.total:
            value = dict(split=self.split, completed_frames=len(self.rows), total_frames=self.total,
                         elapsed_seconds=time.time() - self.started)
            path = self.root.parent / 'evaluation_progress.json'
            tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value)); tmp.replace(path)

    def finish(self):
        bins = []
        for i, n in enumerate(self.count):
            values = self.sums[:, i] / n if n else [None] * 5
            bins.append(dict(lower=i / 10, upper=(i + 1) / 10, pixels=int(n),
                             mean_confidence=values[0], mean_predicted_error=values[1],
                             mean_actual_error1=values[2], mean_actual_error2=values[3],
                             mean_polar_update=values[4]))
        corrs = [r['spearman_confidence_error'] for r in self.rows if r['spearman_confidence_error'] is not None]
        scenes = {}
        for scene in sorted({r['scene'] for r in self.rows}):
            values = [r['spearman_confidence_error'] for r in self.rows
                      if r['scene'] == scene and r['spearman_confidence_error'] is not None]
            scenes[scene] = float(np.mean(values)) if values else None
        return dict(semantics='T1 local quality score; not calibrated correctness probability',
                    grid='4x4 pooled RGB MAE', tau=self.tau, bins=bins,
                    spearman_image_mean=float(np.mean(corrs)) if corrs else None,
                    spearman_by_scene=scenes, per_image=self.rows,
                    evaluation_seconds=time.time() - self.started)
