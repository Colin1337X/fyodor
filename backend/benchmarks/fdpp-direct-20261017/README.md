# Single-launch FDPP attention, October 17, 2026

The original optional FDPP width64/128 implementation can now normalize and
write one-partition output directly. Unsafe scores or values trigger native
stable attention in the same launch. Four-partition decode retains its
separate combine kernel. This removes one launch per short-context attention
layer, without changing the safe exponent interval, F32 KV or tolerances.

NYA_CUDA_FLASH_DECODING_PP=1 remains required; default inference is unchanged.
NYA_CUDA_FDPP_DIRECT=0 restores the two-launch specialized path. Generic,
portable and trace selections also retain the two-launch path. Tracing needs
the partial retry flags and is never used for accepted performance timings.

Thirteen independent double-oracle cases pass the unchanged 2e-6 scaled-error
limit, including GQA, window1, odd sequence lengths, length2051, mixed-range
scores, forced large-score/value retries and output guard regions. All four
CUDA sanitizers pass those cases. C17/C23 each pass the selected five-test
lifecycle suite, including graph reuse and BLAS-unavailable fallback. Real
TinyLlama CPU-logit parity checks cover capacities129 and273. NVRTC compilation
passes compute75/80/86/89/90/100/120; execution is tested only on SM120.

A fresh source copy physically excluding optcpp, CUTLASS and CuTe builds and
runs real-model CUDA decode with FDPP requested, using native fallback without
a host C++ compiler. This is a build/smoke check, not a repeat of the full
physical-removal test suite.

The initial kernel/results.json predates the change that avoids forming the
unused partial-buffer offset in direct mode. The final sanitizer cases and
runtime validation include that correction. source-manifest.json identifies
the final tested source and build configurations.

Vulkan FDPP, AMD execution, FA4 integration and full FDPP flat-GEMM/dataflow
work remain pending. The accepted llama.cpp score remains 93.18% short and
114.31% long; the paired native/FDPP experiment here is a self-comparison.

## Controlled runtime result

All eight serial activities passed the established idle and contamination
gates. No competing process required termination. TinyLlama Q4_K_M, F32 KV,
CPU threads6, batch512, 128 decoded tokens, two warmups/five repetitions per
process, reversed variant order, graphs enabled and host logits every token.
All observations are pooled, including the unusually variable short FDPP run
and long native run; none were discarded.

| Prefix | Native tok/s | FDPP tok/s | Change |
| --- | ---: | ---: | ---: |
| 0 | 356.201 +/- 0.727 | 353.676 +/- 8.213 | -0.71% |
| 1024 | 305.275 +/- 7.586 | 299.176 +/- 0.470 | -2.00% |

This does not establish a throughput win. The direct kernel only affects
short one-partition decode, so the long-path variation is not attributed to
this change. A native/FDPP comparison also does not isolate the direct kernel
from the preceding two-launch FDPP implementation. Default execution remains
native; neither these measurements nor the reduced launch count are used to
claim progress beyond the accepted llama.cpp score.
