# Run plan

Seed: 2026; K=2; feedback=False; full gradient; start=200; end=1000.

Loss: 0.5 L1 + L2 + 0 align1.

LR: 0.0001 / 1e-05; batch=2; crop=128.

Preflight passed; peak VRAM=5018.3662109375 MiB.

Initialization: PIDSR e42 + official TaskNet; resume uses last.pt.
Best: round2 scene-macro PSNR, SSIM, earlier step; validation only.
Against matched K2 control, only feedback usage differs.
