# Public CUDA training and event telemetry — 2026-10-08

Validation completed October 10, 2026. Reproducible evidence and the source/binary
audit are in [training-cli-cuda-20261008](training-cli-cuda-20261008/).

The native trainer now accepts `--compute cpu|cuda` (CPU default) and a separate
`--device-memory-mib` arena budget (1024 MiB default). CUDA requests fail when
unavailable or over budget; they never silently replay training on CPU. Pretrain,
CPT, SFT and DPO use the same dataset ordering, supervised-token/pair weighting,
accumulation groups and checkpoint identity. Tokenization and fixed initial DPO
reference probabilities are prepared on CPU before the resident session leases
parameters. Training and held-out graph execution then use the selected device.

Resident checkpoints retain the existing NYARUN/optimizer format. Loading occurs
before device/session creation; saving uses a validated resident snapshot.
Export explicitly detaches the complete state before writing GGUF. Stop/EOF
still finishes a whole update and preserves completed state. Same-device restart
is checked for exact checkpoints/exports. Cross-device trajectories are compared
at the existing 1e-5 scaled bound, not promised universally bit-identical.

A failed held-out pass must not prevent saving the last completed update.
`discard_evaluation` explicitly clears evaluation diagnostics without modifying
parameters, gradients, moments, variance or step. It rejects training failures,
active graphs and pending updates; poisoned devices remain unrecoverable. A
snapshot still validates all persistent state before publishing it. A one-step
large-learning-rate regression provokes post-update evaluation failure and checks
that the checkpoint/model are saved while the overall process returns failure.
This does not claim the resulting high-learning-rate model is useful for inference.

## Timing and memory semantics

Training devices own eight fixed CUDA event slots. Recording a mark adds no
allocation, host transfer or host synchronization; marks are unique, scoped to
the device, and expire when their slot is reused. Querying elapsed time never
waits, requires completed marks and leaves the caller's output untouched on
failure. Stale, foreign, reversed and missing marks are rejected. Event objects
are destroyed with the context, including partial construction cleanup.

The runner records events around queued forward, backward and optimizer work.
Loss observation already synchronizes each microbatch, and optimizer observation
already synchronizes each completed update. Timing queries follow those existing
boundaries; no extra phase fences are introduced. CUDA event intervals include
stream idle gaps caused by host dispatch, not just aggregate kernel busy time.
`preparation_ms` is host elapsed time constructing the graph and uploading its
metadata; some queued initialization can finish after that interval. Whole-step
wall time and tokens/s remain the end-to-end throughput measurement. CPU phase
columns retain their previous host wall-time interpretation. CSV now labels the
compute device and timing source explicitly.

`device_used_bytes` and `device_peak_bytes` describe charged device arena storage,
not total physical VRAM usage. `device_capacity_bytes` is the explicitly reserved
arena. Driver/context memory is additional. Parameter and graph host budgets
retain their existing separate meanings; automatic model/optimizer/graph memory
estimation remains future work. Missing CPU GPU-specific metrics stay blank.

## Desktop integration

The desktop queries its actual trainer's `--capabilities` JSON through a bounded
background subprocess. CUDA is selectable only after that executable can create
a small training device. Detection has a 30-second deadline and a 4096-byte
response limit. It does not use inference backend availability as a substitute.
Saved configurations default to CPU and a 1024 MiB device budget. The Rust argument
boundary validates the device name and positive budget before launching.

Live telemetry distinguishes the selected device, gradient norm and GPU training
arena peak. Missing or invalid measurements are not converted to zero. The
packaging script now searches both native executables' build caches for NVRTC,
covering a CPU inference backend paired with a CUDA trainer.

The initial desktop/browser automation kernels failed to initialize with
`failed to write kernel assets` (OS error 3). After the computer-use plugin update
on October 9, native inspection worked. The release desktop displayed the training
form, detected actual CUDA availability, offered CPU/CUDA selection, and revealed
the 1024 MiB GPU budget when CUDA was selected. At 1282×832 the visible controls
were readable without overlap. Inspection found the overview's stale "CPU autograd"
label; it is now "Native autograd". `visual-check.json` identifies the inspected
binary and precise scope. This was a visual/control check, not a full GUI training
run; command-level and packaged workflow tests cover execution separately.

## Validation evidence

- The public probe runs 100 CPU/CUDA updates and compares all parameter/gradient/
  moment/variance values; maximum observed scaled error is zero on the tiny model.
  A new GPU process resumes update 40 through 100 with an exact final checkpoint.
- GPU evaluation and stop suites pass across pretrain/CPT/SFT/DPO, including
  accumulation, fixed held-out subsets, Unicode paths, interruption and resume.
- Eight timing slots are tested for no hidden allocation/copy/fence and invalid,
  expired, reversed, foreign and poisoned-device handles.
- Twelve frontend tests and three release Rust library tests pass. The first
  Rust build exposed a missing production serde_json dependency (previously
  dev-only); subsequent builds hit Windows commit-memory limits. After stopping
  NVIDIA overlays, a single-job release library test build passed. Failed logs
  are retained. No OS, development or data-processing application was stopped.
- The initial GPU stop test caught a compatibility regression: a separate device
  telemetry line preceded `step=1`. Telemetry now stays on the established progress
  line. The failure and corrected rerun are retained.

The complete 15-stage matrix passed (`matrix/validation.json`): C17 CPU/CUDA,
C23 CUDA, ONNX, Clang ASan/UBSan, physical removal of optcpp/CUDA/CUTLASS,
and frontend tests/build. The CTest totals (including four unavailable ROCm/MLX
hardware skips per configuration) are CPU 61, CUDA 160, C23 CUDA 160, ONNX 62,
ASan/UBSan 61, no-optcpp 61, no-CUDA 61 and no-CUTLASS 155. All twelve frontend
tests pass. These counts do not treat hardware skips as executed tests.

The first sanitizer harness invocation omitted the device suite's `--cuda`
argument, so the test exited without exercising CUDA. Compute Sanitizer correctly
rejected it before any instrumented API call. That failed invocation is retained
in `sanitizers/`; it is not accepted evidence. The corrected harness explicitly
selects CUDA and writes separate logs in `sanitizers-corrected/`.

No accepted performance gain is claimed. All eight GPU sanitizer runs completed
with zero errors; both race checks report zero hazards and warnings. The final
audit verifies current source/binary hashes, all 15 matrix stages, sanitizer logs,
full-state parity, exact resume, frontend/Rust tests, visual evidence and packaging.


The real-model CLI and installer checks are run alongside sanitizer/build work.
Their recorded elapsed times are diagnostic only and excluded from performance
acceptance. A subsequent isolated profile must establish the dominant costs and
measure repeatable throughput before any speed claim is made.


## Real-model public command validation

The public CLI completed two continued-pretraining updates on TinyLlama Q4_K_M
with rank-2 LoRA, an eight-token context and six CPU workers, separately on CPU
and CUDA. Every parameter, gradient, first moment and second moment was compared
under the established scaled `1e-5` bound; maximum observed error was
`2.2737290905124213e-13`. CUDA also completed a public DPO update with a fixed
CPU-prepared initial reference. Each run saved its optimizer checkpoint and a
4,400,929,248-byte merged F32 GGUF. Commands, metrics and model/executable hashes
are in `real-model.json`; raw logs are retained. This does not claim broad-model
coverage or an isolated performance result. Exported-model inference parity was
established separately in the preceding resident-graph milestone; this new probe
specifically covers public command integration and full training-state parity.


## Validation ordering

All functional native changes preceded the complete matrix. A private header's
explanatory timing comment changed after native compilation began; behavior and
module bytes were unchanged. The overview label correction followed the matrix;
all twelve frontend tests passed again (`final-frontend-tests.log`), and the
release desktop was rebuilt. Native binaries were not rebuilt for this text edit.
The final desktop was reopened: persisted CUDA selection and the GPU budget were
correct, availability detection succeeded, and the overview displayed "Native
autograd". The app closed normally. Native workflow evidence may be reused only
after verifying byte equality of every packaged backend executable, library and
resource; final installer extraction and desktop bundle-marker checks remain fresh.


## Final package

The rebuilt NSIS installer contains 406324928 bytes, SHA-256
`85485ab8c3a4fd6050728b4a06507241fb58bb3677a96c174144d1feef9b3963`. Fresh extraction verifies every packaged resource and
the desktop executable after its expected Tauri bundle marker replacement. All
native resources match the first verified package exactly, so its 36 CUDA
evaluation runs, 54 CUDA safe-stop runs, CPU CLI lifecycle and eight authenticated
HTTP checks (including repeated long-prompt CUDA generation) remain applicable.
`final-package/desktop-package.json` identifies the prior execution record by hash
and states the reuse explicitly. Both installer copies are preserved on D:.
The installer was extracted, not installed.


The visual binary hash initially differed from the post-build desktop file.
The audit established the exact cause: Tauri temporarily changes its single
`__TAURI_BUNDLE_TYPE_VAR_UNK` marker to `__TAURI_BUNDLE_TYPE_VAR_NSS` while bundling,
then restores it. Applying that one verified replacement produces the exact
visually inspected hash; no arbitrary byte differences are ignored. Both native
CUDA kernel bundles remain byte-identical to the preceding graph milestone.

This milestone exposes the validated resident trainer publicly. Automatic memory
estimation, mixed precision, recomputation and controlled throughput optimization
remain open work; it does not claim the full long-term training/inference goal.
