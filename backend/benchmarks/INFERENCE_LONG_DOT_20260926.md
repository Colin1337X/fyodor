# Native CUDA long-dot accumulation — September 26–27, 2026

The [matrix profiling pass](INFERENCE_MATRIX_PROFILE_20260926.md) exposed a
native GEMM rounding case while an optional vendor bridge was unavailable.
With K=8192, exact F16/BF16 weights `67/128` and F32 inputs `31/64`, both native
tile sizes returned **2076.98633** instead of the exact representable result
**2077**. Disabling cuBLAS and CUTLASS deliberately reproduces the result.
The expanded regression also proves it occurs with F16, so BF16 decoding is
not the cause. [Initial reproduction](inference-long-dot-20260926/before.log),
[both tile regressions](inference-long-dot-20260926/regression-before.log).

## Implementation

The native 32×32 and 64×64 F32 GEMM kernels already used two independent
accumulation chains for F32 weights. Other storage formats still used one
long chain. The fix extends the existing F32 algorithm to all seven supported
storage formats, summing alternating K positions separately and combining at
the end. This shortens the reduction depth and fixes the exact regression.
It is original Fyodor code extending the existing implementation; no upstream
source or dependency was added.

Weight decoding, F32 precision, shared-memory layout, barriers, output tails,
dispatch geometry, memory admission, transfers, and failure handling are
unchanged. F32 storage retains its prior arithmetic. Quantized decode GEMV
and vendor GEMM are unchanged. No environment control or user precision
setting is added. This improves accumulation accuracy; it does not promise
exact sums for arbitrary values or arbitrary K.

## Numerical contract and regression coverage

Two explicit CTest cases disable both vendor providers and select each native
tile size. Exact positive-value fixtures use K=8192, 68 rows and 33 tokens for
F32, F16, BF16 and Q8_0, including partial output tiles and guard values.
All four formats now equal their independent analytic results exactly.

The new long cancellation fixtures cover all seven storage formats, 65 rows,
K=8192, and batches 3/17/65/32/8. They use the existing independent double
oracle with bound `1e-6 * (1 + sum(abs(products)))`. The original 1024-column
relative checks retain the same inputs, rows, token subset, and 0.00015 bound.
Real-model scaled-error acceptance remains 0.001.

The first expanded test also applied the short fixture's relative check to
new 8192-column GEMV cancellation cases. One unchanged Q4_K decode result near
zero differed by 0.00039047 while satisfying the forward-error oracle. That
diagnostic is retained in `regression-after.log`; it is not a failure of the
changed GEMM. The new long fixtures explicitly use the magnitude-scaled
contract, while the exact positive-sum regression remains exact and the
original short relative checks remain intact. This change does not claim to
improve the unchanged GEMV result.

Both native tiles pass real TinyLlama Q4_K_M prefill at 513/512/511 tokens with
vendor paths disabled, compared with the validated optimized native CPU
kernels. Maximum scaled error is **1.14062e-5**, below **0.001**. Compute
Sanitizer memcheck and synccheck each pass both long tile suites with zero
errors. These four runs do not claim a new racecheck or initcheck result.

Full validation, performance measurements and package evidence are recorded
under `inference-long-dot-20260926/`.

## Build and recovery validation

The 15-stage validation driver passes on the final source:

| Configuration | Passed | Unavailable hardware skips |
|---|---:|---:|
| CPU C17 | 47 | 4 |
| CUDA/Vulkan C17 | 76 | 4 |
| CUDA/Vulkan C23 | 76 | 4 |
| ONNX enabled | 48 | 4 |
| Clang ASan/UBSan | 47 | 4 |
| optcpp physically absent | 47 | 4 |
| CUDA physically absent | 47 | 4 |
| CUTLASS physically absent | 71 | 4 |

The four skips are real ROCm/MLX hardware checks; mock-provider checks pass.
The matrix includes existing inference, API, training, checkpoint/resume,
export/reload and fallback checks. Frontend tests and the production build
also pass. This is Windows validation; no AMD/Apple hardware, POSIX or native
MSVC-backend result is claimed. [Matrix evidence](inference-long-dot-20260926/matrix/validation.json).

## Desktop package

The NSIS installer was rebuilt and extracted without installation. Every
backend resource matches its prepared source by SHA-256; the desktop binary
differs only by Tauri's expected bundle marker. The extracted backend matches
the validated C17 binary and the extracted trainer passes the CLI workflow
checks. Authenticated loopback HTTP checks cover load, CUDA selection, repeated
greedy generation, runtime inspection and unload. Both runs process 725 prompt
tokens and generate eight tokens with identical text. cuBLAS and CUTLASS are
explicitly disabled for this package check, exercising native CUDA prefill.
The logs contain no CPU replay or resident-plan failure. No interactive UI
verification is claimed. [Package evidence](inference-long-dot-20260926/desktop-package.json).

`frontend/src-tauri/target/release/bundle/nsis/Fyodor_0.3.0_x64-setup.exe`:
406,286,667 bytes; SHA-256
`efc0e5766d92065acdfd4b2e116fadd10ee9b636a26cb41643d9e6e75c3cb94c`.
The previous installer and pre-change benchmark remain under ignored `.tools`.
Generated backend/DLL resources retain the existing ignore policy. The tracked
CPU trainer artifact is unchanged.

## Controlled performance comparison

Starting revision `de7208ae0204496a03e93ab5b4f2356cb3b7e8c6`, GCC 16.2
Release C17, Ryzen 5 9600X with six exposed cores, RTX 5060 Ti 16 GiB,
Windows 11 and driver 610.88. TinyLlama Q4_K_M SHA-256 is
`9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`.
Both engines use F32 KV, deterministic matched token IDs, batch/ubatch 512,
six CPU threads and host logits after every decode token. Loading, prefix
preparation, tokenization and sampling are untimed. Each process has two full
warmups and five measured repetitions, with forward and reverse engine order;
the table pools all ten observations per cell. CUTLASS is disabled. The native
mode additionally disables cuBLAS; the default mode uses cuBLAS F32 prefill.
Upstream uses the existing pinned C API harness with Flash Attention disabled.

KoboldCpp was stopped under the user's standing authorization before timing.
The first attempt then caught a newly appearing RustDesk GPU process and was
excluded in full. Remote access was preserved. The final attempt snapshots
that existing client with the other desktop graphics clients; all 20 process
runs pass the same GPU identity guard and two consecutive 7–12% idle readings
before and after each process. No builds, tests or packaging overlap timing.
This remains an interactive-desktop measurement, not a headless one. The
process and thermal/power/memory samples, commands, hashes, and every raw run
are retained. [Exclusion record](inference-long-dot-20260926/EXCLUSIONS.md),
[accepted comparison](inference-long-dot-20260926/accepted-remote/metadata.json).

Mean tokens/s ± pooled sample standard deviation:

| Prefill path | Prefix | Workload | Before | After | Change | llama C API |
|---|---:|---|---:|---:|---:|---:|
| Native | 0 | pp512 | 1466.41 ± 5.18 | 1421.15 ± 5.24 | -3.09% | — |
| Native | 0 | tg128 | 210.19 ± 0.71 | 209.68 ± 1.18 | -0.24% | — |
| Native | 1024 | tg128 | 186.91 ± 0.53 | 187.28 ± 0.57 | +0.20% | — |
| cuBLAS F32 | 0 | pp512 | 3656.82 ± 48.48 | 3662.44 ± 26.05 | +0.15% | 11913.05 ± 176.09 |
| cuBLAS F32 | 0 | tg128 | 207.09 ± 7.61 | 209.68 ± 0.79 | +1.25% | 296.53 ± 3.51 |
| cuBLAS F32 | 1024 | tg128 | 187.03 ± 0.75 | 185.81 ± 4.55 | -0.65% | 213.30 ± 1.25 |

This is a **correctness fix with a measured 3.09% native-prefill cost**, not a
speedup. It is retained under the correctness-first priority. Decode kernels
and vendor GEMM are unchanged; the noisy short baseline and long candidate
remain in the table and are not claimed as improvements or regressions caused
by changed decode arithmetic. The current default path remains about 3.25×
behind upstream prefill, 1.41× behind short decode and 1.15× behind long decode.
The corrected native prefill remains about 2.58× below the optional cuBLAS path.
These measurements cover one model/device and do not establish universal cost.
[Pooled values](inference-long-dot-20260926/summary.json).

Explicit before/after weight, KV and scratch bytes, dispatch counts, transfers,
and graph captures/replays match in each compared mode/shape. The extra
accumulators are per-thread registers. A separate post-benchmark JIT inspection
records Q4_K's 64×64 kernel at 64→72 registers per thread; Q6_K stays at 72.
Both have zero local bytes and unchanged shared bytes. This is a potential
occupancy cost, not proof of the slowdown's cause. F32's 64×64 kernel remains
at 80 registers. [JIT resource counts](inference-long-dot-20260926/registers.json).
[Resource summary](inference-long-dot-20260926/resources.json) keeps startup
process memory and whole-device memory separate from explicit tensor totals.

[The final evidence audit](inference-long-dot-20260926/audit.json) checks current
source hashes, the 15-stage matrix, exact regression evidence, both real-model
checks, four GPU sanitizer runs, all 20 accepted benchmark processes, and the
installer/backend identity. The benchmark and package scripts reproduce these
steps; run correctness/build/package work separately from performance timing.
