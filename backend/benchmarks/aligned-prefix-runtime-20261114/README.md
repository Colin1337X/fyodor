# Optional aligned prefix integration

This integrates the isolated 2K/8K prefix experiment and batched FDPP kernel.
It is default off at build and runtime and has no accepted throughput score.
Accepted Gemma decode ratios remain 85.61025294% short and 106.8707255% long
of the historical equivalent llama.cpp baseline; the 125% goal remains open.

Build with NYA_ENABLE_ALIGNED_PREFIX=ON, NYA_ENABLE_OPTCPP=ON,
NYA_ENABLE_PORTABLE_ATTENTION=ON, NYA_ENABLE_WIDE_VALUE=ON,
NYA_ENABLE_WIDE_BATCH_SPLIT=ON, NYA_ENABLE_SMALL_BATCH=ON and
NYA_ENABLE_SMALL_BATCH_REUSE=ON. Missing required optional files omit the
feature. Runtime additionally requires NYA_CUDA_ALIGNED_PREFIX=1,
NYA_CUDA_WIDE_VALUE=1, NYA_CUDA_WIDE_BATCH=1, NYA_CUDA_WIDE_BATCH_SPLIT=1,
NYA_CUDA_WIDE_SPLIT=1, NYA_CUDA_SMALL_BATCH=1,
NYA_CUDA_SMALL_BATCH_REUSE=1 and NYA_CUDA_FLASH_DECODING_PP=short.
Do not disable split attention or select portable FDPP.

Admission checks the measured dense Gemma geometry and Q4_K/Q6_K projection
types, capacity >256, active small/reuse kernels and required attention kernels.
Full-attention layers may copy projected keys as values, as the real model does.
Other geometries, small capacities, absent components and unavailable optional
symbols retain existing execution. Missing symbols may leave a conservatively
larger scratch reservation, but never cause a call through a missing pointer.

Large prefix batches use the existing four-input weight-reuse kernel with the
serial dot reduction order, instead of vendor/native tiled GEMM. Narrow batches
retain their existing row-dot/interleaved projection dispatch. Before position
256, batched 256-wide attention uses the serial FDPP arithmetic. Later prefix
batches use per-query wide split attention. Both prefill and verification stop
chunks at position 256. Serial decode and selective long verification attention
are unchanged. Query-major partial storage is checked before multiplication and
reduced with the admitted batch size under memory pressure.

## Evidence

run_verify.py retains test_verify.c lifecycle assertions and the 0.0005 bound,
specializing only the token count. It uses the actual integrated library and
records source/library/binary hashes, settings, exit status and observed admission.
Original files cannot be overwritten; an explicit run label creates new evidence.
Interleaved two/four verification and batched output projections are enabled.

- Initial prefix255/ring1/count4: numerical assertions pass, but alignment was
  not admitted. The runner correctly rejects this as feature validation. The
  guard incorrectly required a separate value matrix on full-attention layers.
  This rejected artifact is retained, not presented as an aligned-path pass.
- Corrected prefix255/ring1/count4-v2: admitted, both resets pass, maximum scaled
  logit error 0.0000178964637. Exercises a verification batch crossing position
  256 and a three-token tail. FDPP batch calls are observed.
- Prefix31/ring0/count4: intentionally not admitted (capacity36), both resets
  pass, maximum scaled error 0.0000524084294. Small-capacity fallback is observed.
- Prefix257/ring1/count4: admitted, both resets pass, maximum scaled error
  0.0000219546455. Existing interleaved verification calls remain observed.
- The isolated prior prefix8192/ring0/count2 lifecycle now passes with zero
  logit difference on both resets, including rewind and replacement replay.
  Its evidence remains in ../fdpp-batch-20261114. It is distinct from this
  integration, which preserves more of the existing verification dispatch.

Strict -O3 C17 and C23 library builds pass with warnings as errors. The
verify-without-non-c target passes all 82 tests with four expected ROCm/MLX
hardware skips, plus CPU inference/train/export lifecycle checks. Its copied
tree physically lacks optcpp, backend/optcpp, CUDA/cutlass and CUDA/cute.
A separate C23 CUDA-host build in that same removed-source tree also passes
with all aligned-prefix prerequisites requested ON; CMake reports the feature
omitted. The removal target now explicitly requests NYA_ENABLE_ALIGNED_PREFIX.
removal.log and removal-cuda-c23.log retain those build/test outputs.

Integrated 8K ring/four-token replay is running. Additional long/linear and
boundary shapes, integrated Compute Sanitizer, constrained-memory admission,
and equivalent uncontaminated generation comparisons remain required before
any performance acceptance or default enablement. Raw batched FDPP and wide
split memory checks from earlier artifacts do not substitute for these gates.
