# Reusable CUDA training scratch scopes — September 28, 2026

The private resident training device previously retained every buffer until
device destruction. A repeated graph would eventually exhaust its arena or
4,096 descriptors. It now supports one non-nesting scratch scope: allocations
before the scope survive, while ending it invalidates subsequent handles and
rewinds the arena and descriptor count. Buffer and scope tokens share a global,
never-reused identifier namespace, rejecting expired, foreign and wrong-kind
handles even after physical storage is reused.

Ending a scope adds no CUDA operation or synchronization. All queued consumers
and subsequent allocation zeroing run on the device's single serialized stream;
the arena itself stays allocated. Callers must still finish explicitly to detect
asynchronous failures. A poisoned device cannot begin or end a scope. CPU callers
must serialize operations. Persistent weights, gradients and optimizer state
must be allocated before the scratch scope.

This is original Fyodor C implementation. The source study in
[the reproducibility snapshot](training-scratch-20260928/snapshot.json) records
the MIT-licensed llama.cpp/ggml revision and file hashes: `common.cuh` selects a
pool per device and stream, `ggml_cuda_pool_vmm::free` rewinds its offset in LIFO
order, and `mmf.cu` uses scoped temporaries on that stream. Fyodor uses its own
bounded arena, graph-wide scope and expiring integer handles; no upstream source
was copied and no VMM or C++ dependency was added.

The change extends the internal device API,
not the public training graph. The CLI and UI still execute training on the CPU;
resident non-matrix operations, loss, optimizer and graph integration remain
unfinished. No end-to-end GPU training throughput improvement is claimed.

## Initial evidence

The GCC C17 CUDA build and all four training-device tests passed; see
[the test log](training-scratch-20260928/initial-tests.log). New coverage includes
5,000 successive scopes with queued zero, forward and gradient accumulation,
variable allocation lengths, stale handles, invalid scope tokens, descriptor
exhaustion and recovery, active-scope destruction and injected launch failure.
The repeated-work fixture retains three buffers occupying 516 bytes, reaches
1,036 bytes peak arena usage, adds zero transfers or fences inside the loop, and
produces exactly 60,000 accumulated gradient units.

The real TinyLlama Q4_K_M matrix harness covers first-layer query, feed-forward
down and up matrices at 8 and 64 tokens. It checks 64 spread elements per
operation against a double-precision oracle. Existing synthetic tests check all
elements, all seven supported weight formats and CPU autograd agreement.

| Harness mode | Arena capacity | Peak arena bytes | Final live buffers | Scope resets |
| --- | ---: | ---: | ---: | ---: |
| Retain six jobs | 268,435,456 | 265,928,704 | 36 | 0 |
| Reuse, forward order | 67,108,864 | 59,531,264 | 0 | 6 |
| Reuse, reverse order | 67,108,864 | 59,531,264 | 0 | 6 |

All three runs pass and report identical work/transfer counts: 2,826 launches,
18 uploads totaling 42,221,568 bytes, and 1,152 downloads totaling 4,608 bytes.
The comparison retires each complete matrix job, including its weights and
gradient buffers; it is a storage-lifetime demonstration, not a full persistent
model memory estimate. Capacity is the actual reserved device arena; peak usage
is its allocation high-water mark and excludes CUDA context/module overhead.

Raw outputs and counters are preserved in
[the evidence directory](training-scratch-20260928). Timings in these outputs
are not accepted performance measurements: desktop GPU activity was not isolated.
The harness uses two warmups and five timed batches of twenty calls, with
`NYA_CUDA_BLAS=0`, `NYA_CUDA_CUTLASS=0`, and `NYA_COMPUTE=cpu` (CPU model loading;
the training device explicitly requires CUDA). Reproduce with:

```powershell
build-cuda/nya-bench-training-device.exe .tools/tinyllama-q4_k_m.gguf forward
build-cuda/nya-bench-training-device.exe .tools/tinyllama-q4_k_m.gguf reuse-forward
build-cuda/nya-bench-training-device.exe .tools/tinyllama-q4_k_m.gguf reuse-reverse
```

## Configuration and sanitizer validation

All 15 stages of [the configuration matrix](training-scratch-20260928/matrix/validation.json)
passed. CPU C17 and ASan/UBSan each passed 48 applicable tests, CUDA/Vulkan C17
and C23 each passed 80, and ONNX passed 49. Physical removal passed 48 tests
without `optcpp`, 48 without CUDA, and 75 without CUTLASS. Each configuration
skipped the same four unavailable ROCm/MLX hardware checks. All 11 frontend tests and
the production frontend build also passed.

[CUDA sanitizer records](training-scratch-20260928/sanitizers.json) show complete
memcheck, initcheck and synccheck runs exiting zero with no reported errors.
Each runs the entire program, including all 5,000 queued scratch lifetimes and
the descriptor-exhaustion/recovery checks.

The unrestricted racecheck was deliberately interrupted during lengthy repeated
kernel instrumentation. Its target returned an error on termination; the
zero-hazards summary is **not** a completed pass. Its raw logs and explicit
[exclusion record](training-scratch-20260928/racecheck-interrupted.json) are kept.
The bounded replacement completed with exit zero and zero hazards. It instruments
256 launches after skipping the original
240 launches (238 on the matrix-suite device and two foreign-device allocation
zeroes). This covers the scratch setup and reuse across all 67 allocation
lengths. The full 5,000-iteration program still executes, but race instrumentation
does not cover every iteration. See [the runner](training-scratch-20260928/racecheck.py)
for the exact selection and independent process-exit recording.

## Desktop package and final audit

PowerShell 7 packaging completed successfully. The rebuilt
`frontend/src-tauri/target/release/bundle/nsis/Fyodor_0.3.0_x64-setup.exe` is
406,282,477 bytes with SHA-256
`1e906aaf23dc292a8814cce7bbd2fde93d110437e536daead6d2877377d8d43c`.
[Package verification](training-scratch-20260928/desktop-package.json) checks
all extracted backend resources against the packaged sources, the desktop
binary's expected Tauri bundle marker, the extracted trainer CLI, and eight
authenticated/unauthenticated HTTP requests. Native CUDA inference with optional
BLAS/CUTLASS disabled produced the same eight-token continuation twice from a
725-token prompt, without CPU prefix replay. This is extracted-binary validation,
not an installation or interactive UI test.

[The final evidence audit](training-scratch-20260928/audit.json) verifies source
hashes, all 15 matrix stages, full stress-test outputs under three CUDA sanitizer
tools, the bounded racecheck exit/result, exclusion of the interrupted run, and
the package verification record. The scripts and raw outputs remain alongside
it. No throughput result from this milestone is accepted; complete resident GPU
training and its end-to-end trajectory validation remain future work.
