# Full-runtime launch-bound trial - October 15, 2026

Candidate adds `__launch_bounds__(256,4)` to fused Q4_K gate/up. Launch geometry,
loop unrolling, arithmetic and precision are unchanged. The preceding isolated
probe motivates the trial but is not performance acceptance.

Baseline executable is preserved outside the repository and hashed in provenance.
Validation and monitored paired timing passed. The launch bound is retained;
the accepted matched score is 86.8% short / 107.5% long against pinned llama.cpp.

C17 and C23 builds each passed 25/25 configured CUDA inference, quantization,
and shared training-graph regressions. Real-model CPU-logit checks pass at
existing tolerances. Fused/unfused graph tests matched logits and KV exactly
for 795 decodes across reset and prefix preparation. This does not resolve the
pre-existing strict CPU/GPU KV snapshot discrepancy documented in the previous
accepted trial. No tolerance was changed.

Memcheck, initcheck, racecheck and synccheck passed the fused-kernel harness,
including output guards, odd row counts and loop lengths. All logs and exit
codes are archived. Full-runtime timing runs serially after validation, with
independent GPU process monitoring, unchanged idle gates and reversed order.

Initial paired results (all ten timed samples per cell retained): short
319.618 +/- 13.863 to 331.114 +/- 0.595 tok/s (+3.60%); prefix 512
304.852 +/- 1.070 to 304.521 +/- 14.277 (-0.11%); prefix 1024
284.160 +/- 0.732 to 289.306 +/- 0.994 (+1.81%). The noisy short baseline
and prefix-512 candidate motivate a complete new reversed-order confirmation
for those two contexts, without dropping the initial observations.

## Matched reference refresh

All eight monitored reference activities passed both idle gates with no
competing workload requiring termination. Ten timed samples per cell:

| Decode prefix | Fyodor tok/s | llama.cpp tok/s | Relative throughput |
| --- | ---: | ---: | ---: |
| 0 | 333.229 +/- 0.804 | 383.919 +/- 1.464 | 86.8% |
| 1024 | 290.476 +/- 0.531 | 270.176 +/- 0.808 | 107.5% |

TinyLlama Q4_K_M, F32 KV, full GPU offload, pinned llama.cpp b10809
(5266f24da75dc449bd56cbed7addb9c8e4a6a73e), reference flash attention off,
six CPU threads, maximum batch 512, decode-only 128 tokens, two warmups and
five repetitions per process, reversed engine order. Both engines return host
logits per token; prefix preparation is untimed. RTX 5060 Ti 16 GB,
driver 617.42, Ryzen 5 9600X, Windows interactive desktop. This establishes
no general claim across models, hardware or alternate llama.cpp settings.
Prefill and training were not remeasured or optimized.

## Acceptance and uncertainty

The confirmation pools ten more samples for each short/middle cell and passes
all eight monitoring and idle checks. Its short gain is 0.57%, with noisy
candidate observations; prefix 512 improves 2.11% with low variance.
Pooling both initial and confirmation samples, without excluding any run,
gives short 321.886 +/- 9.833 to 328.562 +/- 9.529 tok/s (+2.07%, 20 samples
per cell), prefix 512 305.213 +/- 1.012 to 308.267 +/- 10.585 (+1.00%, 20),
and prefix 1024 284.160 +/- 0.732 to 289.306 +/- 0.994 (+1.81%, 10).

Retain this small compiler-only optimization: correctness passed, pooled
means improve at all measured contexts and independent kernel diagnostics
support the direction. Short/middle variance limits precision; these means
are not a claim of statistically established gains for every context. The
matched llama.cpp scores come from the complete separately monitored reference
comparison, not from picking the fastest candidate runs. Short remains below
parity. Launch bounds constrain compilation, not guaranteed runtime occupancy.
