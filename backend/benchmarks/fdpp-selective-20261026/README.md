# Selective FDPP decode, October 26, 2026

Clean native comparisons showed four-partition FDPP winning short decode by
2.31% but losing long decode by 2.33%. NYA_CUDA_FLASH_DECODING_PP=short therefore
uses four-partition FDPP at positions0..255 and the existing attention path
afterward. Native attention is the default long path. Full FDPP remains
available with value1; absent/zero retains native inference everywhere.

Graph selection must follow the policy boundary even when native split
attention is disabled or all layers have small windows. Separate cached
graphs preserve this distinction across reset, rewind and interleaved prefill.
Native splitting still honors its own enable flag. No device kernel, F32
precision, numerical tolerance or scratch allocation layout changes.

Permanent lifecycle tests cover the selective mode with native splitting
both enabled and disabled. They check logits and KV against CPU across reset,
prefill and every decoded position, including the boundary. Their fixtures
include window1, odd head widths and shared KV. The graph-count assertion
requires both cached topologies rather than allowing recapture per token.

Real TinyLlama CPU-logit checks pass at capacities129 and273 with unchanged
tolerances. A fresh physical-removal copy excluding optcpp, CUTLASS and CuTe
builds and runs CUDA inference with short FDPP requested, using native
fallback and no host C++ compiler. That check is build/smoke coverage, not a
rerun of the entire physical-removal suite.

The reference experiment compares native, selective FDPP and pinned llama.cpp
in forward/reverse order, serially, with twenty repetitions and ten warmups
per process. TinyLlama Q4_K_M, F32 KV, batch512, six CPU threads, 128 decoded
tokens and prefixes0/1024; host logits are read every token. The hardened
desktop-process gate and CPU/clock diagnostics are retained. All observations
are pooled; model/dependency/binary hashes and raw logs accompany the results.

This is an optional CUDA policy. Vulkan FDPP, AMD execution, FA4 integration,
full FDPP flat-GEMM/dataflow work and the 125% llama.cpp target remain pending.

## Matched result

C17 and C23 each pass all six selected tests. All twelve benchmark activities
pass the established idle/competitor gates; no process needed termination in
this trial. Forty timed observations per cell are pooled, including variance.

| Prefix | Native tok/s | Selective tok/s | llama.cpp tok/s | Selective / llama.cpp |
| --- | ---: | ---: | ---: | ---: |
| 0 | 355.518 +/- 1.227 | 363.055 +/- 6.919 | 383.355 +/- 2.220 | 94.70% |
| 1024 | 306.869 +/- 2.946 | 307.574 +/- 1.017 | 269.254 +/- 1.449 | 114.23% |

Selective FDPP improves short throughput by 2.12% over native. The 0.23% long
difference is treated as timing variation because that path uses native
attention. Native-only ratios in the same run are 92.74% short / 113.97% long.
These are new dated measurements; they do not retroactively alter the earlier
93.18% / 114.31% native result. The selective mode remains opt-in. Enable the
optional attention build and set NYA_CUDA_FLASH_DECODING_PP=short to use it.
