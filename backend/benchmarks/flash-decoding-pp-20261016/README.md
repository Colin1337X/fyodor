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
