"""Render the first saved scene of each split, without model inference or selection."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
output = ROOT / 'confidence_maps'
for split, scene in [('validation', '01'), ('test', '00')]:
    source = output / 'data' / f'{split}_scene_{scene}.npz'
    with np.load(source, allow_pickle=False) as data:
        sample = str(data['sample_id'])
        c, b, e1, e2 = [data[k] for k in ['confidence', 'predicted_error', 'error1', 'error2']]
    metrics = json.loads((ROOT.parent / 'progress/downstream' / f'final_{split}.json').read_text())
    row = next(x for x in metrics['per_image'] if x['sample_id'] == sample)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    specs = [(c, 'Quality score C (not probability)', 'viridis', 0, 1),
             (b, 'Predicted error scale b', 'magma', 0, .1),
             (e1, 'Observed T1 local RGB MAE', 'magma', 0, .1),
             (e2 - e1, 'T2 - T1 local MAE (blue: lower)', 'RdBu_r', -1e-4, 1e-4)]
    for ax, (array, title, cmap, lower, upper) in zip(axes, specs):
        im = ax.imshow(array, cmap=cmap, vmin=lower, vmax=upper, interpolation='nearest')
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, orientation='horizontal', pad=.02, fraction=.065)
    fig.suptitle(f"{sample} | PSNR T1={row['round1']['psnr']:.6f}, T2={row['round2']['psnr']:.6f} dB\n"
                 "4 x 4 pooled evaluation grid; error panels clip at 0.1; difference clips at +/-0.0001",
                 fontsize=11)
    fig.savefig(output / f'{split}_scene_{scene}.png', dpi=180)
    plt.close(fig)
    print(json.dumps(dict(sample_id=sample,shape=list(c.shape),quality_range=[float(c.min()),float(c.max())],
                         error_change_max_abs=float(np.abs(e2-e1).max()),source=str(source))))
