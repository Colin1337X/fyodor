# FDPP partition scheduling, October 17, 2026

The preceding direct-path change did not establish an end-to-end speedup.
This experiment checks whether more independent KV partitions improve GPU
occupancy enough to offset the combine launch and extra partial traffic.

The isolated probe compares 1/2/4/8/16 partitions using the same width64/128
specializations and stable retry kernel. Each shape is checked against an
independent double oracle at the unchanged 2e-6 scaled-error limit. It times
100 captured kernel pairs, ten graph replays per sample, two warmups and five
timed samples in both variant orders. Process snapshots are recorded; these
diagnostics do not have continuous contamination checks and are not accepted
runtime scores.

For width64/32 heads, four partitions reduce the isolated 128-token operation
from 7.69 to 3.96 us. At 1025 tokens, four partitions take 14.90 us versus
16.40 us for eight; increasing partitions is not a general improvement.
At 4096 tokens sixteen partitions improve 51.27 to 45.13 us, but that isolated
result is not sufficient to select a new long-context policy. Forced stable
retries get slower as partitions increase. Raw samples and all errors are
retained in probe/results.json and probe-summary.json.

The runtime candidate sets NYA_CUDA_FDPP_SHORT_SPLIT=1 to use four partitions
also in the short-context graph. It retains the existing scratch allocation,
device request positions, F32 arithmetic, safety checks and stable fallback.
FDPP itself still requires NYA_CUDA_FLASH_DECODING_PP=1. A zero or absent short
split setting retains the preceding scheduling. The candidate does not alter
the native path or the long-context four-partition policy.

The real TinyLlama capacity129 CPU oracle passes. Its log reports unavailable
cuBLAS and native fallback, so it does not prove cuBLAS execution. The matched
performance harness explicitly records resident execution and the selected
prefill implementation. Numerical validation is not a timing measurement.

C17 and C23 each pass the five selected lifecycle tests with short splitting
enabled, including graph reuse, windows, shared KV and odd head widths. The
device kernels are unchanged from the preceding sanitized implementation.

## Controlled runtime comparison

All eight serial activities passed the existing idle and contamination gates;
no competing process required termination. TinyLlama Q4_K_M, F32 KV, six CPU
threads, batch512, 128 decoded tokens, two warmups and five repetitions per
process, reversed execution order. All timed observations are pooled.

| Prefix | Previous FDPP tok/s | Short-split FDPP tok/s | Difference |
| --- | ---: | ---: | ---: |
| 0 | 354.864 +/- 1.295 | 363.917 +/- 1.542 | +2.55% |
| 1024 | 297.527 +/- 3.854 | 298.964 +/- 0.846 | +0.48% |

Both execution orders favor four partitions for short decode. The long-path
difference is treated as measurement variation because its scheduling is
unchanged. The option is retained; neither the native default nor the FDPP
default is changed by this experiment. A fresh native/llama.cpp comparison is
required before promoting it into default inference. Accepted llama.cpp scores
remain 93.18% short / 114.31% long, below the 125% target. Vulkan FDPP, AMD
execution, FA4 and full flat-GEMM/dataflow implementation remain outstanding.
