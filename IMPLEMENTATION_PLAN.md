# K=2 implementation plan

Scope: seed2026, fixed original mixed RGB, shared R2P and shared PolarFree.
No long training is part of this coding task. Original K=1 files remain untouched.

## Data and gradient path

P0 + RGB -> R2P -> P1 -> TaskNet -> T1, F1
P1 + RGB + adapter(F1) -> same R2P -> P2 -> same TaskNet -> T2

Use existing TaF1/TaF2/TaF3 exposed as T1/T2/T3 (96/96/192 channels).
Three bias-free 1x1 projections map these to 48 channels; bilinear resize to
R2P spatial resolution followed by mean. Inject beta * feedback into fused
features before the original residual body. Beta starts at zero, projections
are random (not also zero). PIDSR R2P is strictly loaded before adding modules.
Round 1 delegates to the unchanged reference forward.

The same module objects execute twice. Feedback and P1 stay attached to autograd;
only alignment targets detach. Prior parameters freeze, but prior input gradients
remain enabled. Snapshot/replay torch RNG at TaskNet entry pairs all stochastic
draws across rounds, including initial diffusion noise.

Loss: 0.5 task(T1, GT) + task(T2, GT) + lambda align1.
Recalibrate lambda on the original shared R2P parameters using 4 scheduled batches.
Both K2 modes use identical initialization, data and loss protocols. No-feedback
is an independently trained variant with the feedback branch bypassed.

## Implementation and verification

Adapt a copy of run_experiment.py; preserve loaders, official loss, optimizer and
per-group clipping. Add feedback/no-feedback configs, checkpoint provenance,
round-specific validation, actual clamped polarization updates, drift and gradients.
Best selection uses round2 scene-macro PSNR, SSIM, then earlier step.
Preflight checks repeated deterministic forwards, frozen prior, both rounds,
and feedback activation after one disposable optimizer update; persist no updates.
CPU synthetic tests verify gradient and sharing semantics. Real CUDA preflight
and peak VRAM remain pending until explicitly run on the selected machine.

K2 retains two task graphs during training: memory must be measured on target GPU.
Do not silently change batch/crop to handle OOM. No server paths are assumed.
Compare independent K1, K2 no-feedback and K2 feedback with matched selection
rules on validation; fix design before final test. Improvement is a hypothesis.
