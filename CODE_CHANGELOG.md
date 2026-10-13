# K2 code changes

## 2026-10-09 completion: full confidence comparison and evidence archive

Both A2_dual and A0_direct completed 16,000 additional updates and full 872-frame
validation plus 188-frame test evaluation. Fixed common step16,000; no metric
selection. GPU work ended at Beijing08:04:40 before the08:55:33 cutoff. CPU audit
verified all469 Adam states per arm, source/final checkpoint fingerprints,
unchanged locked inputs, unique full manifests and exact two-round call counts.
Archived142 original evidence files and both final checkpoints locally; download
hash/size checks passed. CPU analysis verifies32 fixed scene-first NPZ maps,
retains sparse/non-monotonic bins, produces maps and paired descriptive results.
The archive helper's first metadata check misread confidence_enabled (head exists
in both arms); mode=direct/dual governs gate use. Corrected audit only, no model
or metric changes. PSNR gain is negative on validation and test; see the complete
research report. Added CPU-only archival/analysis helpers under operations.

## 2026-10-09 overnight: authorized confidence implementation and training

Added the isolated `server_scripts/confidence_joint_20261009/` extension. Reuses
the archived matched trainer, source loading, task loss, data/schedules, checkpoint
resume and original PSNR/SSIM evaluation. Adds a 23,353-parameter local error-scale
head, detached quality supervision/gates, shared RGB-feature conditioning for
both gate and FiLM, and complementary second-round polarization-update gating.
H has its own Adam parameter group and clipping. Existing E/alignment/net_g
weights strictly load from the completed full direct-T1 checkpoint.

Two serial arms target 16,000 additional steps each under the user eight-hour
deadline, with predeclared time-only common-checkpoint fallback and mandatory
872 validation plus 188 test frames per arm. No test-based choices. An independent
watchdog limits only this controller's verified process descendants.

First startup passed real CUDA integrity and updated 200 times, then failed on
missing SciPy in metric collection before saving its checkpoint. Preserved that
run; NumPy tie-aware rank correlation removes the additional dependency, and the
new `_v2` run starts afresh with the same science protocol. GPU preflight and the
first 200-step saved checkpoint passed in v2; its run resumed beyond step400.
Full validation and test results are pending; GPU preflight and initial source
load records are archived locally with the locked configuration.

## 2026-10-09: confidence mechanism design, no model implementation

Added `analysis/CONFIDENCE_MECHANISM_DESIGN_ZH_20261009.md`, with original-paper
evidence, a T1 local error-scale head, detached quality supervision, confidence
condition fusion, complementary polarization-update gating, and matched full-test
ablations. Defaults to the latest completed full direct-T1 checkpoint, including
R2P, alignment projections and net_g; no G6-only initialization or R2P reset.
Archived five primary PDFs with provenance hashes. The older confidence route is
marked superseded. Training code and model weights were not changed or run.

The direct-T1 experiment described below subsequently completed its additional
10000 steps, validation on 872 images and test on all 188 images on 2026-10-08.
See `analysis/DIRECT_T1_JOINT_RESULTS_ZH_20261008.md` for its final status;
earlier progress descriptions below are historical.

Date: 2026-09-29. Original K1 project and experiment outputs are unchanged.

## Files

- run_experiment.py: copied K1 runner; added optional feedback injection, shared
  two-round forward, paired task RNG, recurrent task loss, renewed calibration,
  preflight gate activation audit, round-specific validation, gradient/drift logs,
  schema 2 checkpoints, protocol checks, staged 200 -> 3000 resume and RUN_PLAN.
- config.yaml / config_no_feedback.yaml: separate output roots, same seed2026 and
  schedules, [0.5, 1.0] task weights; only feedback usage/output location differ.
- tests/test_recurrent.py: CPU protocol and runner integration verification.
- IMPLEMENTATION_PLAN.md / README.md: design and execution/transfer limitations.

## Mechanism

Task features are net_g returned TaF1=out_dec_level1_out (96 channels),
TaF2=out_dec_level2 (96), TaF3=out_dec_level3 (192), exposed as T1/T2/T3 by
the existing OfficialPolarFreeTask adapter. Shapes on the real GPU batch remain
to be measured. Each is projected to 48, bilinearly resized and averaged.
18,433 new parameters = 48*(96+96+192) + scalar beta.

The feedback signal remains attached. Only alignment targets detach. The same
R2P and TaskNet instances execute twice. Diffusion prior remains frozen and eval;
input derivatives through it remain enabled. net_g alone is trainable in TaskNet.
All original R2P keys strictly load before adding the external feedback branch.
Zero beta does not make the second R2P application identity.

## Verification status

- Python compilation: passed.
- Seven CPU tests: passed with real PIDSR e42 R2P and synthetic TaskNet.
- Synthetic preflight / checkpoint / training / resume integration: passed.
- Real PolarFree deterministic CUDA preflight: pending on target machine.
- Real K2 peak VRAM: unmeasured.
- Real 200/3000-step runs and validation/test performance: not run.

External K1 data/model/loss/evaluation helpers are reused, not rewritten.
Server transfer still requires path mapping, official dependencies and real GPU audit.

## 2026-10-08: direct T1-image joint training from trained K1

Added an isolated extension under `server_scripts/direct_t1_joint_20261008/`.
It copies the existing matched/phase trainer and reuses data, schedule generation,
losses, optimizer, clipping, checkpointing and evaluation. The second R2P condition
is clamped T1 without detach, while both task calls retain original mixed RGB and
paired diffusion RNG. Full aligned K1 seed2026 step3000 R2P/projection/net_g weights
are strictly inherited; its R2P head is not reset. There is no confidence branch.

The new study starts fresh Adam, continues its own optimizer after the automatic
200-step integrity stage, and targets an additional 10000 steps. First-round
alignment and frozen-prior audits remain active. The real RTX3080 Ti startup
check passed, including nonzero L2-to-T1-image gradient and gradients in R2P,
projection heads and net_g. Training has started; final 872-image validation and
performance results are pending. See `analysis/DIRECT_T1_JOINT_PROTOCOL_ZH_20261008.md`.

The first attempt hit CUDA OOM during proxy validation after 200 updates; its
logs are preserved. Validation now releases consumed gradients and temporarily
offloads Adam moments to CPU, restoring original dtype/device afterward. Resume
and final evaluation read checkpoints on CPU to avoid unused GPU copies. These
memory changes are confined to the copied trainer; the original snapshot is
retained beside it. The current immutable run is
`direct_t1_joint_20261008_memorysafe_v2`, restarted from the same source K1.
This revision passed proxy validation after step200, saved the checkpoint, and
resumed all457 Adam parameter states at step200; training has continued beyond
step350. The copied trainer's obsolete generated RUN_PLAN text was preserved in
diagnostics and replaced by the actual protocol. Final metrics remain pending.

User added the requirement that every experiment must evaluate the full test
split. The running study is extended with a separate test-completion process
without modifying its locked training code or restarting it. It waits for the
training/validation process to exit, reuses its exact fixed-step10000 evaluator,
and checks all188 test IDs, both round metrics,376 forward calls per module and
the same weight hash as validation. It publishes a combined report and marks
completion only after test success. The local default runner now includes this
stage; AGENTS.md records the standing requirement. Evaluation scope is appended
in a separate metadata file, preserving the original training lock.

The study completed at 2026-10-08 20:54 (Asia/Shanghai). All 10000 training rows
are continuous; validation has 872 frames and test 188, with 376 R2P and 376 task
calls on test. Validation and test share the fixed step10000 weights. The downloaded
checkpoint SHA256 matches eb17a5537bcd26583b73670a34a258f948056935f851b42c7a44eb62e1dd15ea.
Test T1/T2 image PSNR is 23.37858336/23.36154630 dB; T2 minus T1 is -0.01703706 dB.
Test means rose versus source K1 but validation means declined. This single-seed
result does not establish iterative benefit or isolate feedback from extra
training. Full final metrics, reports, logs and weights are archived locally;
see analysis/DIRECT_T1_JOINT_RESULTS_ZH_20261008.md.
