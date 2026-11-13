# Narrow-batch native row-dot scheduling, 2026-11-13

Original Fyodor implementation in optcpp/matvec/small_batch.cu. A second grid
dimension selects input tokens, reusing the existing native per-row reduction
exactly. This avoids padding batches 2..8 to the 32-token GEMM tile. Each block
still computes eight output rows; no full-weight expansion or new quantization.
This is scheduling reuse, not a new tensor-core kernel or explicit cross-token
weight sharing. Single-token decode and batches >8 retain existing dispatch.

Compile NYA_ENABLE_SMALL_BATCH=ON (with NYA_ENABLE_OPTCPP), select
NYA_CUDA_SMALL_BATCH=1. Both default off. Missing source, missing optional symbol,
or reference mode retains native kernels. C owns all dispatch/lifetime. Both
resident plans and the assisted matmul interface select the same optional entry
points. The profile reports small-batch-dot separately from native-gemm.

## Correctness

- C17 and C23: 12 selected tests each, covering quant formats, compute tails,
  CPU/Vulkan/CUDA verification, ring/shared KV, no graphs, failure injection,
  and the speculative acceptance/rejection test. Exact commands: ctest
  --test-dir build-cuda (or build-cuda-c23) -R
  'quant-kernels-cuda-small-batch|validate-cuda-compute|inference-verify|generate-speculative-batch-cuda'
  --output-on-failure, with NYA_CUDA_SMALL_BATCH=1.
- Permanent quant test: F32/F16/BF16/Q4_0/Q8_0/Q4_K/Q6_K; 65 rows,
 1024 columns; batches 2/3/4/8 are byte-equal to native scalar CUDA matvec.
 Existing float64 forward-error and relative oracles remain unchanged.
 Batches9/17/32/65 exercise fallback/tile tails. The dedicated comparison
 disables optional integer/byte dot replacements so both use the same arithmetic.
- Compute F32 tests add batches 2/3/4/8/9 with nonaligned widths. CUDA memcheck
 for both quant and compute tests reports zero errors. NYA_COMPUTE=cuda,
 NYA_CUDA_SMALL_BATCH=1, NYA_TEST_SMALL_BATCH=1 for quant; --tool memcheck
 --error-exitcode 99. Compute argument is --cuda. Logs retained.
- Real Gemma4 12B Q4_K_M: prefixes 7 and 1020, 17 verification inputs,
 replay/replacement/rewind and reset twice, F32 KV, batch 8, ring KV, existing
 accepted matrix options. Fixed scaled bound .0005 unchanged. Maximum errors
 .0000879829416 and .0000180854805; all assertions pass. Exact model commands,
 environment and executable hash in gemma-results.json. These are correctness
 runs; CPU compilation overlapped part of them, not timed comparison.
- Physical non-C removal: all 82 registered CPU tests pass, four expected
 unavailable ROCm/MLX hardware skips, plus train/export/infer. The verifier now
 requests NYA_ENABLE_SMALL_BATCH=ON even when optcpp is absent. A separate CUDA
 configure/build and quant execution against that physically stripped source
 also passes with runtime SMALL_BATCH=1. cuda-absent-verified.log records it.
 cuda-absent.log is a failed harness attempt resolving the broken global CMake;
 the retry used the configured compiler-directory CMake executable.

## Diagnostic timing; no accepted decode claim

The prior standalone-speculation harness is reused unchanged. One executable,
32 generated tokens, same TinyLlama Q4_K_M as target and standalone draft,
window 4, batch 8. Each call includes plan creation and prompt processing.
Four serial processes select small-batch off/on/on/off; each runs eight serial
versus batched generation pairs with reversed inner order. First two pairs per
process are excluded from summary (12 warmed observations per setting).
Greedy and temperature.7/top-k40/seed123 outputs match in every pair. Explicit
commands, environment, model/binary hashes are in timing.json; summary.json
contains aggregation. NYA_CUDA_BLAS=0 for the matched comparison.

| Small-batch kernel | Serial generation call | Batched generation call |
| --- | ---: | ---: |
| Off | .280892s mean | .552586s +/- .009984s |
| On | .217219s mean | .210422s +/- .001511s |

Batched-call speed ratio 2.62608x. Serial calls also improve because their
six-token prompt now uses the narrow kernel: do not label this decode-only.
A same-model draft is an oracle/scheduling probe, not a useful deployment pair.

Profiling (NYA_CUDA_PROFILE=1, NYA_CUDA_PROFILE_FROM=6) disables graphs. First
batched target matrix time 79.500 ms, versus 351.769 ms in the previous checkpoint;
its batch 4 gate/up 25.893 ms versus 115.933 ms. This supports the scheduling diagnosis;
event totals are not normal graph-enabled throughput. profile.log is retained.

The unchanged desktop gate rejected retained GPU identities. No acceptance,
no new llama.cpp score; accepted Gemma ratios remain 85.6103% short / 106.8707% at 8K.
This removes a measured verifier regression but does not provide resident MTP
hidden/KV binding or achieve the 125% goal on both contexts.
