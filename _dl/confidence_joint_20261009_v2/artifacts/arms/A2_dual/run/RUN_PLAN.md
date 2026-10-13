# Run plan

Seed: 2026; K=2; feedback=False; full gradient; start=200; end=16000.

Loss: (L1+2L2)/3 + 0.0306518811 align1 + Lconf.

LR: 0.0001 / 1e-05; batch=2; crop=128.

Preflight passed; peak VRAM=5774.4921875 MiB.

Initialization: latest full direct-T1 K2; fresh Adam; quality head only is new.
Evaluation: fixed final common step; clock fallback only, never test selection.
Matched control shares source, schedule, optimizer and quality auxiliary training.
