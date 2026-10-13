# Actual condition-representation pilot

Frozen G6 first prediction; R2P condition=mixed_rgb.
One R2P call and two frozen TaskNet calls; same restored diffusion RNG.
Only final corrected task loss supervised, phase0.1; lambda0.
R2P e42 features, zero residual head; only18 R2P tensors trained.
Both arms seed2026, same initialization, data schedule and 1000 additional steps.
Use fixed step1000 validation; no test or checkpoint selection.
