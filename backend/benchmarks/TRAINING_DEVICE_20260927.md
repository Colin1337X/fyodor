# Resident training matrix foundation — September 27, 2026

The trainer still executes on the CPU. This change supplies a private C storage
and matrix layer for a future resident training graph; it does **not** enable
GPU training in the CLI or UI. Loss, non-matrix backward operations, optimizer
state, transactional updates, and graph lifetime planning still need integration.

## Motivation and implementation

The [CPU execution profile](TRAINING_EXECUTION_20260925.md) measured roughly
244–300 ms of mapped forward matrices and 250–290 ms of mapped backward matrices
per TinyLlama rank-2, context-8 update. Matrices dominate that workload. Calling
the inference matmul API from each eager training operation would repeatedly
download activations and gradients. The new layer instead owns persistent
buffers and queues matrix work on one CUDA stream.

`core/training_device.h` and its private backend vtable use only C handles,
explicit sizes and status returns. The generic implementation compiles without
CUDA. All CUDA dependencies, host dispatch and device kernels stay in `/CUDA`;
no host C++ compiler, ggml tensors, PyTorch runtime or foreign graph is introduced.

Each device owns one bounded arena with 256-byte-aligned allocations and at most
4096 buffers. Allocation zeroes the requested bytes on the execution stream.
Arena accounting includes inter-buffer alignment. Buffers live until teardown;
there is no individual free or arena reset yet. Process-wide monotonic atomic
IDs reject handles from another device or a destroyed/recreated device. One
caller serializes each device's operations.

The row-major contract is `Y = X W^T`, `dX += dY W`, `dW += dY^T X`.
Forward reuses Fyodor's native 32/64-tile F32-accumulation GEMM. Original tiled
backward kernels give each output one writer and use two F32 reduction chains.
F32/F16/BF16/Q4_0/Q8_0/Q4_K/Q6_K weights remain in their stored representation;
the input-gradient kernel decodes original weight columns into a padded shared
tile without allocating a full transposed/dequantized matrix. Activations,
gradients and full weight gradients use F32. This is not mixed-precision training.

Matrix dispatch performs no transfers, allocation or fences. Explicit host
writes first finish earlier device work, upload, then establish the legacy-copy
stream dependency before the nonblocking execution stream consumes the data.
Reads finish queued work before copying. Counters expose explicit transfer bytes,
launches, synchronization calls, arena use and failure state. These counters do
not include driver/module internal allocations or context initialization.

Invalid shapes, aliases, stale handles, insufficient buffers, overflow and arena
exhaustion fail before a launch and leave the device reusable. Driver/stream
failures poison the device; further reads, writes and kernels are refused until
destruction. No CPU fallback silently continues a partially executed update.
`NYA_TRAIN_DEVICE_FAIL_LAUNCH_AFTER=N` deterministically tests this boundary.
Kernel dispatch success only means work was queued: this layer does not yet
detect nonfinite tensor contents or validate an optimizer transaction.

## Source study and provenance

Studied MIT-licensed llama.cpp/ggml revision
`5266f24da75dc449bd56cbed7addb9c8e4a6a73e`:

- `ggml/src/ggml.c`, `GGML_OP_MUL_MAT` backward: accumulates outer products,
  avoiding a materialized transpose of the larger weight tensor.
- `ggml/src/ggml-cuda/mmf.cu`: matrix layout/stride dispatch and scoped device
  scratch ownership.
- `ggml/src/ggml-cuda/opt-step-adamw.cu`: resident elementwise optimizer state.
  Its in-place update is not adopted: Fyodor's current optimizer validates all
  staged results before mutating parameters, and future device integration must
  preserve that recovery contract.

The new source is original Fyodor code using standard matrix derivative
identities and the existing Fyodor tile layout. No upstream implementation was
copied. [Source and license hashes](training-device-20260927/source-study.json)
identify the exact material inspected.

## Correctness

`nya-test-training-device` checks every matrix element against a double oracle,
with the existing native matrix forward-error bound
`abs(actual-reference) <= 1e-6 * (1 + sum(abs(products)))`. It tests all seven
weight types, both forward tile sizes, non-square shapes, tile tails, one-token
work, 8192-term forward/input-gradient/weight-gradient reductions, repeated
gradient accumulation, output-end sentinels, and explicit zeroing. A separate
cross-entropy case compares both gradients with the CPU autograd graph at an
absolute bound of `1e-6`.

API tests cover invalid/foreign/stale handles, null devices, zero/overflow
allocations, exact-capacity exhaustion, out-of-bounds transfers, invalid types,
quantization alignment, undersized matrices, aliases, and driver failure after
initialization and after a queued forward operation. Counter assertions prove
no transfers, fences or arena growth occur during repeated matrix dispatch.

[Compute Sanitizer memcheck](training-device-20260927/final-memcheck.log),
[synccheck](training-device-20260927/final-synccheck.log),
[initcheck](training-device-20260927/final-initcheck.log), and
[racecheck](training-device-20260927/racecheck.log) report zero errors on the
numerical suite (racecheck also reports zero warnings). These are scoped checks,
not proof of arbitrary graph safety.

The [full validation record](training-device-20260927/matrix/validation.json)
contains all 15 completed build/test, physical-removal and frontend stages:

| Configuration | Passed | Skipped |
|---|---:|---:|
| CPU C17 | 48 | 4 |
| CUDA/Vulkan C17 | 80 | 4 |
| CUDA/Vulkan C23 | 80 | 4 |
| ONNX | 49 | 4 |
| ASan/UBSan | 48 | 4 |
| `backend/optcpp` physically absent | 48 | 4 |
| `CUDA` physically absent | 48 | 4 |
| `CUDA/cutlass` physically absent | 75 | 4 |

The four skips per configuration are unavailable ROCm/MLX hardware tests.
Frontend tests pass 11/11 and the production frontend build passes. Existing
training/checkpoint/resume/export, API, inference and optional-provider checks
remain included. The first physical-removal configure failed because its copy
excluded `benchmarks`, including the new required C harness. All three copy
scripts now include that single source without copying bulky evidence; fresh
physical-removal runs pass. The failed log and
[pre-fix record](training-device-20260927/matrix/validation-before-copy-fix.json)
remain preserved. Already completed configurations were reused because compiled
source did not change during that verifier-only correction.

[Final source hashes](training-device-20260927/final-source-hashes.json) cover the
device layer, kernels, tests, harness and removal scripts in addition to the
standard matrix manifest.

## Primitive measurements

RTX 5060 Ti 16 GB, driver 610.88, Windows WDDM desktop, Ryzen 5 9600X. GCC 16.2,
Release C17, native NVRTC kernels; cuBLAS and CUTLASS disabled. The loaded model
is TinyLlama 1.1B Q4_K_M. Binary, source, model hashes, commands, environment and
resource samples are in [the record](training-device-20260927/accepted/measurements.json).

Three real first-layer matrices, 8 and 64 tokens, deterministic synthetic F32
activations and incoming gradients. Each operation checks 64 spread result
elements against the double oracle before timing. Each of two fresh processes
runs two warmup batches and five measured batches of 20 queued operations with
one terminal fence. The second process reverses job order. Setup, zeroing,
allocation, model load and correctness reads are untimed; dispatch and the
terminal fence are timed. Each cell below combines all ten batch averages.
Repeated dX/dW calls accumulate rather than overwrite. No sample was removed.

Milliseconds per operation, mean ± sample standard deviation:

| Matrix (outputs × inputs; storage) | Tokens | Forward | dX | dW |
|---|---:|---:|---:|---:|
| Query (2048 × 2048; Q4_K) | 8 | 0.2315 ± 0.0074 | 0.2773 ± 0.0089 | 0.1654 ± 0.0157 |
| Query | 64 | 0.3122 ± 0.0099 | 0.4897 ± 0.0148 | 0.3067 ± 0.0076 |
| FFN down (2048 × 5632; Q6_K) | 8 | 0.6570 ± 0.0125 | 0.5222 ± 0.0154 | 0.4802 ± 0.0143 |
| FFN down | 64 | 0.8910 ± 0.0181 | 1.0317 ± 0.0175 | 0.9286 ± 0.0110 |
| FFN up (5632 × 2048; Q4_K) | 8 | 0.5332 ± 0.0120 | 0.7785 ± 0.0146 | 0.4744 ± 0.0170 |
| FFN up | 64 | 0.6239 ± 0.0123 | 1.3667 ± 0.0175 | 0.9330 ± 0.0237 |

This establishes a primitive baseline, **not** a training speedup, convergence
result, or comparison with an equivalent upstream training step. Quantized
weights are frozen storage; dW exercises a full F32 gradient destination and
does not imply updating quantized weights in place. The harness reserves
268,435,456 bytes, uses 265,928,704 bytes across 36 buffers, uploads 42,221,568
bytes at setup and downloads 4,608 bytes for correctness. All timed matrix
batches assert zero additional transfer bytes.

Before and after each process, two consecutive whole-GPU utilization readings
must be 7–12%. Existing desktop GPU identities remain allowed; newly appearing
GPU identities and pure compute competitors reject the whole attempt. No
competitor appeared. The desktop, including Wallpaper Engine, stayed open, so
these results are not headless measurements. Resource samples include startup
and host model mapping; whole-GPU memory includes other applications.
The two short processes each yielded two resource samples: the largest sampled
process peak working set was 162,099,200 bytes, sampled private memory was
518,283,264 bytes, and whole-device memory was 1,972 MiB. Active GPU samples were
98% utilization, 2,812 MHz, 78–81 W and 63–64 °C. These sparse samples cannot
establish an exact per-process VRAM peak.

Reproduce from the repository root:

```powershell
cmake --build build-cuda --parallel 4
ctest --test-dir build-cuda -R training-device --output-on-failure
python backend/benchmarks/training-device-20260927/measure.py new-attempt
python backend/benchmarks/training-20260923/validate.py --output new-validation-directory
```

The benchmark requires `.tools/tinyllama-q4_k_m.gguf`; configure the CUDA build
with the existing driver/NVRTC provider instructions. CPU-only test builds also
compile the harness but cannot execute its CUDA measurement mode.

## Desktop package

Rebuilt `frontend/src-tauri/target/release/bundle/nsis/Fyodor_0.3.0_x64-setup.exe`:
406,283,632 bytes, SHA-256
`28854d806ac6a49dba60922c93712c328f3ffcf232129ca3693f348c34032606`.
[Package verification](training-device-20260927/desktop-package.json) records
extracted resource hashes, trainer CLI exit zero, and eight authenticated/unauthenticated
HTTP checks. Native CUDA inference with cuBLAS/CUTLASS disabled produced identical
eight-token continuations twice from the 725-token prompt, with no CPU recovery.
The desktop executable matches the built binary after the expected Tauri bundle
marker replacement. This does not claim an installed or interactive UI test.

The initial packaging invocation used Windows PowerShell 5, whose `Stop` policy
turned Tauri's informational stderr into a `NativeCommandError` after NSIS had
produced the installer. The wrapper returned 1; its partial stdout log is retained.
The independent verifier subsequently proved that the new installer contains the
current backend and trainer and that both execute correctly. `package.ps1` now
requires PowerShell 7; invoke it with `pwsh -File`, not `powershell -File`.
No successful exit from that initial wrapper is claimed.

## Next integration work

The CLI/UI remain CPU-only. A complete resident graph must add non-matrix
forward/backward operations, losses, finite checks, reusable graph storage and
transactional optimizer state before exposing device selection. The current
shared NVRTC module also includes the training kernels during inference setup;
separating that compilation will avoid unnecessary inference initialization work.
No inference throughput gain or complete GPU training trajectory is claimed here.
