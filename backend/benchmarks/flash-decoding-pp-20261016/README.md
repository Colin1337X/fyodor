# Flash-Decoding++ attention integration, October 16, 2026

Original implementation of section 3's unified-scale partitioned attention,
quarantined under root optcpp as requested. It is not the complete upstream
engine or a claim of FA4 support. See optcpp/attention/FLASH_DECODING_PP.md.
The existing accepted baseline remains 93.18% short / 114.31% long versus llama.cpp.

The first nine cases pass an independent double oracle at 2e-6 scaled error
with one and four partitions. memcheck/initcheck/racecheck/synccheck all pass.
A tenth case forces one late retry after earlier tiles accumulated; all heads
match the oracle and exactly one partition is marked. Empty partitions, odd
head width, GQA, sliding windows, request positions, overflow/underflow and
large value bounds are covered. SM75/80/86/89/90/100/120 compile; only SM120 ran.
The real TinyLlama resident CPU-logits oracle passes its unchanged tolerance.

The first regression run passed 37/39. Two graph-count assertions exposed that
scratch reservation accidentally selected two-graph scheduling for windowed or
explicitly unsplit attention. The fix separates scratch needs from the existing
native graph eligibility policy. No test or numerical tolerance was weakened.
The original failure logs are preserved. graph-policy-fix holds the reruns.

The final physical-removal copy excludes root optcpp, backend/optcpp, CUDA/cutlass
and CUDA/cute. Configure requests optional attention anyway; native fallback
builds and all 157 tests have no failures, with five expected unavailable-provider
skips. removal-proof.json verifies missing paths and no host C++/CUDA compiler
cache entries. This does not validate real AMD or Apple execution.

CUDA and Vulkan portable online-softmax support preceded this implementation;
their independent evidence is in ../portable-attention-20261016. Vulkan has not
yet been ported to this unified-scale algorithm. No contaminated or historical
measurement is promoted as a performance win for these new kernels.

Final C17 selection: 39/39 passed. C23 selection: 40/40 passed, including the
new permanent inference-cuda-flash-decoding-pp test. Both language modes rebuild
and pass the dedicated Vulkan test after adding a 32-bit host byte-size guard.
All validation is correctness evidence, not a throughput result.

## Controlled first implementation timing

Eight serial, continuously monitored activities passed the same desktop idle
and contamination gates as prior accepted measurements. No competing process
required termination. Same executable, FDPP off/on, reverse order, TinyLlama
Q4_K_M, F32 KV, CPU threads 6, maximum batch 512, two warmups, five repetitions
per process, 128 decode tokens, graph execution and host logits every token.
Ten samples per variant/context are pooled without discarding observations.

| Prefix | Existing tok/s | Initial FDPP tok/s | Change |
| --- | ---: | ---: | ---: |
| 0 | 354.429 +/- 3.154 | 298.186 +/- 0.704 | -15.87% |
| 1024 | 305.812 +/- 6.758 | 109.199 +/- 2.031 | -64.29% |

The feature remains opt-in and is rejected as a default performance change.
This is a self-comparison, not a refreshed llama.cpp score. Accepted reference
scores remain 93.18% short and 114.31% long. Raw clean/environment.json,
metadata.json and paired-summary.json retain all evidence.

## Coalesced warp diagnostic

optcpp/attention/flash_decoding_pp_warp.cu is a CUDA-only experimental partial
kernel, not selected by production dispatch. Eight warp-local partitions stream
contiguous K/V elements and retain numerator/denominator in registers, then
reduce using the same shared phi. Stable whole-head retry is unchanged.
Ten numerical cases pass unchanged 2e-6 tolerance. Graph diagnostics add the
real 32-query/4-KV-head, width64, prefix1024 shape, also passing the oracle.
That isolated pair improves from 0.06893 ms to 0.02266 ms; the late retry case
still costs about 0.355 ms. These diagnostics use snapshots rather than the
continuous acceptance monitor and are NOT end-to-end performance claims.
The initial host-submission probe also destroyed events after releasing the
CUDA context; its failure is retained, and the corrected probe uses graphs and
correct cleanup ordering. Runtime integration and matched timing remain next.

The coalesced partial kernel also passes memcheck, initcheck, racecheck and
synccheck on all ten direct numerical cases (warp-sanitizer-exits.json).
