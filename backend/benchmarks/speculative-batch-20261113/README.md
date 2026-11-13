# Standalone speculative batch integration, 2026-11-13

`NYA_SPECULATIVE_BATCH=1` opts standalone draft generation into the optional
all-position CUDA verifier. Default remains serial. MTP assistants remain on
the existing host-state path: this is not resident Gemma MTP support.

The proposal window is bounded by the target and resident draft batch capacities
to retain the earliest possible rejected suffix in ring KV. The target's initial
distribution remains available; accepted proposal j installs verification row j
(the distribution AFTER that input). A rejection truncates the target plan at
the corrected token's position before running the corrected input. Existing
p/q acceptance, residual max(p-q,0), greedy selection, EOS and output bounds
remain in use. Every speculative target position, including discarded work and
the final input, counts in target_steps. A failed verification returns failure
and discards the response. If a plan has fallen back during prefill, generation
uses the serial path. CPU and unsupported providers retain serial generation.

## Correctness evidence

Permanent `generate-speculative-batch-cuda` asserts actual batch execution via
target_steps, all-accepted behavior, all-rejected greedy windows, seeded serial
sampling equality across 32 seeds with both accepted and rejected proposals,
14-token continuation across a 3-token sliding-window ring (batch 8), and failure
injection immediately after successful prefill. The existing 4096-seed CPU
distribution oracle remains. C17/C23 CPU, Vulkan, ordinary CUDA and batch CUDA
generation tests pass. CUDA memcheck passes. CPU ASan/UBSan tests with the opt-in
environment set pass (serial fallback). See logs for exact executed tests.

Real TinyLlama Q4_K_M: greedy and temperature .7/top-k40, seed123, 32 tokens,
window4, same model as standalone draft. Serial/batched text, token count and
stop reason agree for every observed pair. This same-model test is a correctness
and scheduling diagnostic, not a recommendation for a useful draft model.
The model harness compiles with GCC C17 -O3 -Wall -Wextra -Werror, includes
backend/include and backend/core, links build-cuda/libnya-engine.a, bcrypt,
and C:/Windows/System32/vulkan-1.dll. Its positional argument is the model path.

## Rejected performance promotion

Run environment: NYA_COMPUTE=cuda, NYA_CUDA_BATCH=8, NYA_CUDA_RING_KV=1,
NYA_CPU_THREADS=6. `model-order.log` runs eight pairs, reversing order every two
pairs. Ignore runs0/1 for warmup interpretation. Warm serial is approximately
.28s per32 tokens, batched .55s: a material regression, not a speedup. Each call
includes plan creation and prompt work. The initial `model.log` used a two-pair
version of the harness and shows why cold timing alone was misleading.
`model-blas-order.log` adds NYA_CUDA_BLAS=1 and the local llama-b10809-cuda
cublas64_13.dll. It does not improve this workload: current cuBLAS dispatch
requires batch>=32; this verifier uses batch4.

Event profile (`NYA_CUDA_PROFILE=1`, NYA_CUDA_PROFILE_FROM=6) disables graphs.
First serial target: matmul92.484ms. First batched target: matmul351.769ms,
including native batch4 gate/up115.933ms. Norm4.897ms and attention2.617ms are
small by comparison. Separate draft work is still roughly96ms matmul.
These event totals identify small-batch quantized GEMM as the next bottleneck;
they are not comparable to ordinary wall throughput with graphs enabled.

No clean-environment acceptance: desktop/browser/development processes remain.
No new llama.cpp score. Accepted Gemma ratios remain 85.6103% short /106.8707%
at8K. Keep this foundation opt-in while optimizing small-batch kernels and
implementing assistant device hidden/KV binding; do not market it as faster.

Physical-removal validation completed with the opt-in environment set: all82
registered cases pass (four expected ROCm/MLX hardware skips), plus the explicit
training/export/inference workflow. No C++ compiler or optional optimization
directory is required by this generation integration.
