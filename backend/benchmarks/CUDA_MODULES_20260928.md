# Separate CUDA inference and training compilation — September 27–28, 2026

The [resident training foundation](TRAINING_DEVICE_20260927.md) initially embedded
its kernels in the same NVRTC program as inference. Every inference context then
compiled training derivatives; every training context compiled inference
attention and opened optional inference matrix libraries. This added unnecessary
initialization work and coupled compiler failures across the two domains.

## Change

CMake now embeds two source bundles, sharing the existing matrix sources:

- Inference: `matvec.cu + gemm.cu + resident.cu`.
- Training: `matvec.cu + gemm.cu + training.cu`.

The private C context factory selects the required bundle. The public inference
factory retains its interface and optional cuBLAS/CUTLASS discovery. Training uses
the private factory and does not load those libraries because its current
operations use native CUDA kernels. Training initialization errors now identify
the unavailable training device without claiming CPU fallback.

No kernel arithmetic, dispatch dimensions, precision, matrix ownership or
training semantics change. NVRTC and the CUDA driver remain runtime dependencies;
no host C++ compiler or foreign runtime is introduced. This is original Fyodor
loader/build code, with no newly adapted external source.

[Generated-source evidence](cuda-modules-20260927/source-bundles.json) verifies the
exact byte composition: the former combined bundle was 36,508 bytes including
its terminator; inference is now 32,423 bytes and training 18,533 bytes. Embedding
two bundles duplicates common source text, adding 14,448 embedded bytes while
reducing the source compiled for each context.

## Isolation validation

[The build-only fault test](cuda-modules-20260927/verify_isolation.py) modifies only
the generated header, injecting an NVRTC `#error` into one bundle at a time. It
rebuilds the small executable, checks both domains, and restores the exact header
bytes in a `finally` block before rebuilding the normal executable.

[All checks passed](cuda-modules-20260927/isolation.json):

- A broken training program prevents training initialization but inference still
  initializes and computes the expected result on CUDA.
- A broken inference program prevents CUDA inference initialization but training
  still initializes and computes the expected result on CUDA.
- Both domains work again after restoration.

The inference probe explicitly requires the CUDA provider; CPU fallback cannot
make the positive check pass. The training probe requires device allocation,
uploads, a matrix launch and the expected downloaded result.

[Fresh-process library checks](cuda-modules-20260927/vendor-loading.json) enable
both optional libraries with explicit paths. The original executable loads both
libraries in either domain. The changed executable still loads both for
inference and neither for training. These single-run timings are diagnostic
outputs, not the performance acceptance sample.

[NVRTC/PTX comparison](cuda-modules-20260927/kernel-code.json) covers every entry:
the combined module has 57 kernels, inference 48 and training 45. All inference
entry bodies are byte-identical before and after. The nine training-only entries
have different module-global ordinals in local branch labels; after normalizing
only that ordinal prefix, their PTX also matches exactly. Local block numbers,
register operands and instructions are not normalized. Driver-reported register,
shared-memory and local-memory requirements match for all corresponding entries.
The [raw label diff](cuda-modules-20260927/kernel-code-diff.log) is retained.

## Compatibility and package validation

All 15 stages in [the validation matrix](cuda-modules-20260927/matrix/validation.json)
passed: CPU C17 48 plus four skips; CUDA/Vulkan C17 and C23 each 80 plus four
skips; ONNX 49 plus four skips; ASan/UBSan 48 plus four skips; physical removal
of optcpp and CUDA each 48 plus four skips; physical CUTLASS removal 75 plus four
skips. Skips are unavailable ROCm/MLX hardware. Frontend tests pass 11/11 and the
production frontend build passes. Existing API, training, checkpoint/resume,
export/reload, optional provider and inference tests remain included.

The first CUDA build attempt collided with the independently running real-model
test's Windows executable lock. That was an orchestration mistake, not a compiler
failure. After the real-model process exited, the unchanged build resumed and
passed. The [failed attempt](cuda-modules-20260927/matrix/validation-file-lock.json)
and linker diagnostic remain preserved; subsequent build/test work ran without
that collision.

[Real TinyLlama parity](cuda-modules-20260927/real-model.json) passes native tiles
32 and 64 at prefixes 513/512/511 against optimized CPU logits. The maximum
scaled error is 0.0000114062 against the unchanged 0.001 threshold, identical to
the earlier long-dot results. No vendor matrix library participates.

The rebuilt NSIS installer is 406,284,715 bytes, SHA-256
`f02edbab51652965ac1312e2c22bab89831c71d5ce9ae7b6440577f53550952f`.
The PowerShell 7 build exits zero. [Independent extraction verification](cuda-modules-20260927/desktop-package.json)
checks bundled binary/library hashes, the expected desktop bundle marker,
trainer CLI workflows, and eight loopback HTTP requests. Native CUDA generation
with cuBLAS/CUTLASS disabled produces the same eight-token continuation twice
from a 725-token prompt without CPU replay. This verifies the packaged artifacts,
not an installation or interactive UI session.

## Benchmark method and exclusions

`nya-bench-training-device --startup inference|training` reports context creation
time separately from the first tiny matrix operation. The first operation also
checks the result. These are device-initialization measurements, not model-load,
token-generation or complete training-step throughput.

The preserved baseline executable uses the `d6e2bdd` engine and the same startup
harness as the changed executable. Each process initializes one device. Two
warmup processes per version/order precede five measured processes. Both
before/after and after/before orders run with the driver cache disabled and
enabled. cuBLAS/CUTLASS are disabled for this comparison to isolate native module
initialization. Driver-cache-disabled runs still benefit from warm filesystem
pages and driver libraries; they are not cold OS boot measurements.

The desktop guard requires two consecutive whole-GPU readings at 0–12% before
and after each block. It records process identities and rejects new GPU
processes or pure compute competitors during the block. Resource samples cover
process RAM and whole-GPU clocks, power, temperature and memory. Other desktop
applications remain open; CPU background activity is not independently gated.

Four attempts are excluded in full, with all raw records retained:

- `accepted/`: the inherited 7–12% idle gate rejected sustained **0%** activity
  after the second inference block. No live process remained on resumption.
  The idle lower bound was corrected to zero; these partial measurements are not
  reused.
- `accepted-idle/`: a new RustDesk PID appeared during the second block. The
  entire attempt was rejected. Remote access was not stopped.
- `accepted-remote/`: Windows Shell Experience Host appeared on the GPU during
  the first block. The entire attempt was rejected; the system process was left
  running.
- `accepted-stable/`: Windows Settings and Application Frame Host appeared during
  the second block. The entire attempt was rejected; neither process was stopped.

Despite their provisional directory names, **none of these attempts is accepted
performance evidence**. No initialization speedup or steady-state throughput
gain is claimed. The change is retained for verified compiler-failure isolation,
removal of unused library loading and reduction of unnecessary compiled source.
A clean paired timing run remains outstanding; repeated retries on the actively
changing desktop were stopped rather than selecting individual favorable runs.

Before the fresh September 28 run, the verified Wallpaper Engine process was
stopped under the user's standing benchmark authorization. Its identity is in
[the record](cuda-modules-20260927/stopped-wallpaper.json). System, development and
remote-access processes remained running.

The full GPU training graph is still unfinished. This separation makes further
training-kernel work independent of inference compilation; it does not expose
GPU training in the CLI/UI or claim a training throughput gain.

Reproduction, from the repository root with no concurrent builds or GPU work:

```powershell
cmake --build build-cuda --parallel 4
python backend/benchmarks/cuda-modules-20260927/verify_isolation.py
python backend/benchmarks/cuda-modules-20260927/kernel_code.py
python backend/benchmarks/cuda-modules-20260927/measure.py new-attempt
```

The measurement script expects the preserved baseline at
`.tools/fyodor-device-before-module-split.exe`. To recreate it, build the current
`training_device_bench.c` startup harness against the engine at `d6e2bdd`, then
retain that executable before building the changed engine. The report's
[source hashes](cuda-modules-20260927/source-hashes.json) identify the exercised
implementation; each attempted measurement records its executable hashes.
