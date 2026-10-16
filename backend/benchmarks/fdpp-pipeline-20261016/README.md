# FDPP width specialization and safe exponent range, October 16, 2026

The preceding coalesced FDPP implementation still regressed 3.17% short and
27.52% long. Its prefix1024 trace retried 24.15% of heads. This experiment
addresses streaming kernel overhead and unnecessary stable recomputation.

## Width and pipeline experiments

Original width64/128 templates avoid inactive register lanes and dimension
checks inside the key loop. Width64 with four-loop unrolling is selected;
width128 uses one-loop unrolling. Unroll2 was not consistently beneficial.
In reverse-order graph diagnostics, width64 prefix1024 partial+combine falls
from about 22.64 us to 14.93 us. Width128 improves about 25.68 to 21.22 us.
These are snapshot-monitored diagnostic timings, not accepted runtime scores.

The separate staged exponent variant reuses native score scratch to evaluate
exponents in parallel. It passed the independent oracle but lost to width
specialization on every probed shape, including a forced retry. It is retained
as unselected diagnostic source under optcpp; CMake does not compile it.

## Range derivation, not a tolerance change

The specialized kernels retain phi=0, F32 storage/accumulation and stable retry,
but accept scores in [-32,32]. The former bound16 was unnecessarily narrow.
For at most 2^32 keys and |V|<=1e12, the exact absolute numerator sum is bounded
by 2^32 * exp(32) * 1e12 < 3.4e35, versus float maximum about 3.4e38. Even
conservative factors for rounding/partial merging leave substantial headroom.
The smallest exponent exp(-32) is normal, not subnormal. Out-of-range/NaN
scores and oversized/NaN values still force stable recomputation. There is no
clipping, downcast or change to the numerical acceptance tolerance.

Mixed-range [-31,31], constant +/-31, rejected +/-33, high/low logits, large
values, odd widths, GQA, empty partitions, windows and late single-head retries
pass the independent double oracle at unchanged 2e-6 scaled error. All four
CUDA sanitizers pass all fifteen cases. Real TinyLlama logits pass unchanged
CPU tolerance. The matched diagnostic trace drops from 340/1408 head retries
to 4/1408 (24.15% to 0.284%); tracing disables graphs and its timing is ignored.

C17 and C23 pass the selected five-test lifecycle suite, including long graph
reuse across resets/prefill. compute75/80/86/89/90/100/120 compile; actual GPU
execution is only verified on SM120. A fresh physical-removal source copy with
optcpp, CUTLASS and CuTe absent builds and runs real-model CUDA decode with the
optional path requested, using native fallback and no host C++ compiler.

NYA_CUDA_FLASH_DECODING_PP=1 remains required. NYA_CUDA_FDPP_GENERIC=1 retains
the previous coalesced width-generic kernel and its bound16. The portable flag
still selects the original generic implementation. Missing the specialization
source retains the coalesced kernel; missing optcpp retains native inference.

Vulkan FDPP, AMD execution, FA4 AOT and flat-GEMM/dataflow work remain pending.
Accepted llama.cpp comparison remains 93.18% short / 114.31% long until refreshed
under matched conditions; no diagnostic number replaces that scorecard.

## Controlled runtime result

All eight serial activities pass the established idle and contamination gates;
no competing process required termination. TinyLlama Q4_K_M, F32 KV, CPU threads6,
batch512, 128 decoded tokens, two warmups/five repetitions per process, reversed
variant order. Model loading/prefix preparation are untimed, host logits return
every token, and every measured observation is pooled.

| Prefix | Native tok/s | Specialized FDPP tok/s | Change |
| --- | ---: | ---: | ---: |
| 0 | 356.430 +/- 1.315 | 353.125 +/- 1.507 | -0.93% |
| 1024 | 307.711 +/- 0.974 | 299.393 +/- 0.566 | -2.70% |

The prior large regression is reduced substantially, but both orderings still
favor native execution. FDPP remains opt-in. These self-comparisons do not
refresh the llama.cpp scorecard. Remaining launch and retry overhead require
further work. Two source header comments were corrected after validation;
source-manifest.json retains both the tested and final hashes.
