# Coalesced Flash-Decoding++ CUDA path, October 16, 2026

The previous portable FDPP path regressed full-model throughput by 15.87%
short and 64.29% long. It remains available with NYA_CUDA_FDPP_PORTABLE=1.
The new original CUDA specialization streams contiguous K/V with eight
warp-local independent accumulators under the same unified phi=0. The host
selects it only inside the explicit NYA_CUDA_FLASH_DECODING_PP=1 opt-in.
Default native inference is unchanged. The optional warp source can itself be
absent: CMake retains the original portable implementation. Removing optcpp
retains native inference and does not require a host C++ compiler.

## Measured cause and correction

NYA_CUDA_FDPP_TRACE=1 disables graph capture and downloads partition flags for
diagnostics. A prefix1024 TinyLlama trace found 24.15% of heads required stable
recomputation, with individual layers reaching 21 of 32 heads. Trace timing is
not a performance result. The initial coalesced partial kernel was correct,
but still used the expensive generic online fallback.

nya_fdpp_combine_warp now invokes Fyodor's existing coalesced stable attention
helper on unsafe heads. It reuses the budgeted per-head score scratch; no new
allocation or transfer occurs during normal execution. Safe heads retain the
unified-scale reduction. Score/value safety bounds and numerical tolerances
are unchanged. The diagnostic trace is opt-in and intentionally adds a host
allocation, download and synchronization per layer; benchmark runs strip it.

## Validation

Both initial and corrected integrations pass the five selected C17 and C23
tests (including model fixture generation). The permanent attention-shape test
covers graph reuse across reset, prefill, long prefixes, sliding windows, GQA,
shared KV and odd widths. Direct tests pass ten double-oracle cases at 2e-6,
including forced high/low score, large value and late single-head retries.
All four CUDA sanitizers pass after the fallback change. The real TinyLlama
CPU logits oracle passes unchanged tolerance. Compilation succeeds for compute
75/80/86/89/90/100/120; runtime execution is verified only on SM120.

A fresh source copy physically excludes optcpp, CUDA/cutlass and CUDA/cute,
configures with optional attention requested, builds and runs real TinyLlama
CUDA decode with FDPP requested. It uses native fallback, and has no host C++
compiler cache entry. This build/smoke check supplements the prior full removal
157-test run; it does not claim another full removal regression run.

Vulkan FDPP, AMD runtime validation, FA4 AOT, and the paper's flat-GEMM/dataflow
specializations remain pending. Neither compatible syntax nor a CUDA test
proves AMD support. The accepted scorecard remains 93.18% short / 114.31% long
until a new matched reference benchmark passes acceptance.

## Controlled throughput result

All eight serial monitored activities pass the existing idle and contamination
gates. No competitor termination was necessary. F32 KV, TinyLlama Q4_K_M,
CPU threads6, batch512, 128 decoded tokens, two warmups and five repetitions
per process, forward/reverse variant order. Model load and prefill are untimed;
every decoded token returns host logits. Results pool all ten observations.

| Prefix | Native tok/s | Coalesced FDPP tok/s | Change |
| --- | ---: | ---: | ---: |
| 0 | 352.187 +/- 8.955 | 341.018 +/- 0.869 | -3.17% |
| 1024 | 304.125 +/- 11.509 | 220.440 +/- 5.836 | -27.52% |

These are paired self-comparisons, not a fresh llama.cpp score. Native baseline
variance limits precision, especially short decode. Both orderings still favor
native execution. The FDPP path remains opt-in and is not promoted as a default
performance change. Raw observations and monitoring are under clean/.

Kernel resource inspection finds no local-memory spills: partial warp54
registers/8256 shared bytes; specialized combine44 registers/1104 shared bytes;
native split40 registers/1104 shared bytes. Further work should target the
streaming dependency chain and retries rather than assuming register spilling.

The explicit NYA_CUDA_FDPP_PORTABLE=1 dispatcher also passes the real TinyLlama
CPU oracle after extending the specialized combine argument list. The generic
kernel continues to consume only its original arguments.
