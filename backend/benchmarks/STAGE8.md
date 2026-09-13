# Stage 8: fused dequantization and weight transpose

This continuation targets the matrix work identified by stage 7. The runtime
still owns the transformer graph, compressed weights, F32 KV, activations and
scratch. No training, API, tokenizer or sampling behavior changes.

## Execution change

cuBLAS previously consumed expanded row-major weights through a transposed-A
GEMM. Compressed weights now decode directly into column-major scratch. A 32x32
shared tile with pitch 33 combines dequantization and transpose in one kernel:
lanes read neighboring source columns, synchronize once, then write neighboring
output rows. Both tile edges are masked independently. The GEMM uses `N,N` and
the actual chunk row count as its weight leading dimension; output retains the
full projection's leading dimension even for a short final chunk.

The expansion budget, compressed weight storage, F32 precision and dispatch count
are unchanged. There is no additional allocation, pass, host/device transfer or
stream synchronization. F32 weights bypass expansion. CUTLASS keeps its existing
layout and 3xTF32 operator. Native matrices and reference mode remain available.
`NYA_CUDA_BLAS_TRANSPOSE=0` selects the earlier layout for device-specific tuning.
This selection is validated on the RTX 5060 Ti; it is not a claim that every
cuBLAS version and GPU prefers the same layout.

The separately removable CUTLASS build also fixes an unconditional NVCC `-O3`
flag that conflicted with MSVC Debug `/RTC1`. Debug uses `-O0`; other build
configurations retain `-O3`. No C++ escapes `CUDA/cutlass`.

## Measurement and validation

The exploratory paired layout runs are preserved in
[stage8-layout](stage8-layout/metadata.json). Final comparisons use the saved
stage-7 executable, current executable and the same independent llama.cpp C API
harness on the same hardware. Token IDs, F32 KV, context, batch size, thread count,
warmups and timed boundaries follow [the comparison procedure](COMPARISON_CAPI.md).
Historical stage-7 rates must not be substituted for the newly measured baseline:
background activity and clocks differ between sessions.

RTX 5060 Ti 16 GiB, Ryzen 5 9600X (six exposed logical processors), NVIDIA
610.88, TinyLlama Q4_K_M. Each engine runs nine repetitions plus three warmups,
in forward and reverse order: 18 measured samples per cell, mean ± sample SD.
llama.cpp is b10809 / `5266f24da75dc449bd56cbed7addb9c8e4a6a73e`, full GPU
offload, F32 KV, flash attention disabled, batch/ubatch 512, six threads.

| Workload | Stage 7 before, tokens/s | Stage 8 after, tokens/s | llama.cpp C API, tokens/s |
|---|---:|---:|---:|
| pp512 | 3285.499 ± 18.709 | **3584.488 ± 24.947** | 11630.366 ± 143.794 |
| tg128, empty prefix | 206.884 ± 0.652 | 207.015 ± 0.807 | 288.872 ± 4.559 |
| tg128, 1024-token prefix | 140.963 ± 0.257 | 141.166 ± 0.297 | 212.250 ± 2.583 |

Prefill improves **9.1%**, reducing mean prompt latency from 155.841 to
142.844 ms. Decode is effectively unchanged: 4.834 to 4.831 ms/token for the
short case and 7.094 to 7.084 ms/token for the long case. Fyodor still trails
the comparison by about **3.24× prefill**, **1.40× short decode**, and
**1.50× long decode**. This does not establish the primary performance goal.

The final [raw comparison](stage8-final/metadata.json) requires absence of known
competing applications and two consecutive 7–12% GPU readings both before and
after every run. All accepted idle readings were 9%, with the same memory-clock
state. Windows GPU Engine counters identified Desktop Window Manager and Codex
as remaining activity. **These are matched desktop measurements, not fully idle
or headless GPU results.** A Roblox/browser-contaminated attempt and another
attempt mixing 0% and 9% desktop states were rejected and preserved in
`stage8-contended` and `stage8-mixed-idle`. A stricter 2% gate rejected the active
desktop before collecting any measurements. No Fyodor test/build/package job
overlapped the accepted comparison.

Model SHA-256 is
`9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`.
Fyodor accounts for 636.18 MiB weights, 22 MiB short-context KV and
86.564 MiB scratch; the long case uses 49.5 MiB KV and 87.111 MiB scratch.
All are unchanged. These counters exclude vendor/driver overhead and are not
peak process RAM/VRAM telemetry. The long comparison requests capacity 1152;
llama.cpp internally rounds it to 1280, as documented by the harness.
Prefill still downloads/synchronizes once per run; decode once per token.
There are no additional explicit launches, transfers or synchronizations.

Separate CUDA-event profiles over four pp512 workloads (including a warmup)
show matrix time **453.769 → 402.775 ms**, an 11.2% reduction. Attention is
98.511 → 102.365 ms; native and CUTLASS control profiles are effectively
unchanged. Profiling disables graphs and includes submission gaps; these event
aggregates are not benchmark wall time. Matrices still account for about 78%
of the measured post-change GPU interval and attention about 20%.

Validation results, without relaxed tolerances:

* C17 CUDA/Vulkan **47/47**, C23 CUDA/Vulkan **47/47**; C17 CPU **22/22**;
  C23 CPU/ONNX **23/23**; CPU ASan/UBSan **22/22**.
* CUTLASS physically absent **42/42**; all CUDA physically absent:
  CPU **22/22**, independent Vulkan **25/25**.
* Both optional CUTLASS Debug and Release builds succeed and pass the expanded
  quantized-kernel oracle/chunk/edge tests.
* Real TinyLlama prefixes 65/64/63: maximum scaled errors
  0.0000508473 / 0.0000441052 / 0.0000280637.
* Real Gemma prefixes 33/32/31: maximum scaled errors
  0.0003992879 / 0.0002600110 / 0.0003670901. Existing limit: 0.001.
* GPU Compute Sanitizer: full matrix-test **memcheck, initcheck and synccheck:
  zero errors**; transpose-kernel **racecheck: zero errors/warnings/hazards**.
  Initial memcheck failed to attach and is not counted. The successful retry
  uses application-only tracking and port 54321; the initial launch failure's
  root cause is not established.
* HTTP/API **36 passed**, frontend unit/packaging **8 passed**, browser
  **73 passed**. Windows MSI and NSIS bundles rebuilt successfully, including
  the tested backend and both optional matrix libraries; installers were built,
  not installed. The packaged backend hash matches the tested executable.

New exact-result fixtures cover F32, BF16 and Q8_0 chunk boundaries, one- and
four-row tails, odd BF16 column widths 33/65/257, row counts 5/31/33, batch 33,
and output canaries. The seven-format double oracle and existing model,
shared-KV, speculative, long-context and autograd tests remain in the suite.
Raw logs, scripts, source hashes and artifact hashes are in the
[validation manifest](stage8-validation/metadata.json).

## Rejected experiment

An experimental six-product BF16 CUTLASS warp adapter compiled but failed the
unchanged quantized-kernel oracle: the first F32 GEMM case returned -36.0483437
where the double reference was -18.5504028848. It also failed exact chunk/stride
validation. This is a correctness failure, not an acceptable precision tradeoff.
The adapter and build option were removed without performance acceptance or
relaxing tolerances. Source, patch and failure logs remain in
[stage8-validation](stage8-validation/).

NVIDIA's supported BF16x9 cuBLAS emulation is a different implementation. The
installed-version [support table](https://docs.nvidia.com/cuda/archive/13.1.1/cublas/index.html#floating-point-emulation)
lists compute capabilities 10.0 and 10.3, not this GPU's 12.0; it was not enabled
as an optimization for this device.

## Files and remaining work

Production code changes: `CUDA/gemm.cu`, `CUDA/cuda.c`, `CUDA/blas.h`,
`CUDA/blas.inc`, and `CUDA/cutlass/CMakeLists.txt`. Tests change
`backend/tests/test_quant_kernels.c` and `backend/CMakeLists.txt`. Documentation
changes this report, `backend/README.md` and `CUDA/cutlass/README.md`; raw benchmark
and validation evidence is retained alongside them.

The next larger step is precision-correct tensor-core matrix execution over
quantized operands, with a deliberate operand-packing/cache policy. The existing
bounded F32 expansion remains a practical fallback but does not eliminate that
performance gap. Quantized decode projection fusion and cache/data movement,
followed by CPU packed matrices and resident Vulkan execution, remain separate
optimization targets. Hybrid layer offload, continuous batching and GPU training
are still incomplete. This checkpoint does not claim to beat llama.cpp.
