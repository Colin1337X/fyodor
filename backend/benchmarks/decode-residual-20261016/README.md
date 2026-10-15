# Decode projection/residual fusion trial - October 16, 2026

Fresh diagnostic profiling after QKV fusion still places matrix work at
1075.916 ms, with 54.739 ms of vector work. Profiling disables graphs and
includes submission gaps; it guides investigation, not performance acceptance.

Candidate writes attention-output and FFN-down Q4_K/Q6_K row sums directly
into the residual stream, preserving the ordered accumulation and rounded F32
addition. Decode-only, non-Gemma, non-reference, widths above 256. The existing
small-width add/norm path, prefill and unsupported formats remain unchanged.
No allocation or packing is introduced. NYA_CUDA_RESIDUAL=0 retains the
separate path for exact comparisons. Matrix profiling labels dot-add operations.

Hot-weight graph diagnostics compare dot-plus-add against the fused operation.
All six cases match exactly with output guards. Target-shape improvements range
from 2% to 5.6%, while a wider 4096-row projection regresses 0.7%; full-runtime
validation and monitored timing below support acceptance. These diagnostic timings use
snapshots only and cannot update the scorecard. Accepted score is now
92.4% short / 113.5% long of pinned llama.cpp.

C17 and C23 each pass 25/25 configured CUDA inference/quantization and shared
training-graph regressions. Production-kernel checks cover six shapes/formats,
including partial row blocks and odd column block counts, with exact output
and canary comparisons. Memcheck, initcheck, racecheck and synccheck pass the
bounded graph-replay checks with zero errors/hazards.

The optimized CPU logits oracle passes at existing tolerances. Dedicated
residual-fusion on/off validation checks 795 graph decodes across reset and
prefix preparation: logits and KV match exactly, saving 34,980 launches.
The pre-existing strict CPU/GPU KV snapshot discrepancy is not resolved by this
work. No tolerance or precision mode is changed.

## Monitored runtime timing

Initial paired means (ten samples per cell, all retained): short
345.083 +/- 14.703 to 350.317 +/- 8.675 tok/s (+1.52%); prefix 512
326.244 +/- 1.752 to 329.946 +/- 1.183 (+1.13%); prefix 1024
302.180 +/- 1.492 to 304.916 +/- 1.423 (+0.91%). Both execution orders
improve at all contexts, but short-run variance motivates one complete new
reversed-order confirmation. Its samples will be pooled with the originals.

All eight separate matched reference activities passed monitoring and idle gates:

| Prefix | Fyodor tok/s | llama.cpp tok/s | Relative throughput |
| --- | ---: | ---: | ---: |
| 0 | 354.815 +/- 0.576 | 383.948 +/- 1.150 | 92.4% |
| 1024 | 306.469 +/- 0.369 | 269.970 +/- 0.615 | 113.5% |

TinyLlama Q4_K_M, F32 KV, full GPU offload, pinned llama.cpp b10809
(5266f24da75dc449bd56cbed7addb9c8e4a6a73e), reference FA off, six CPU threads,
maximum batch 512, decode-only 128 tokens, two warmups and five timed repetitions
per process, reversed engine order. Prefix preparation/loading/tokenization are
untimed; both engines return host logits every token. RTX 5060 Ti 16 GB,
driver 617.42, Ryzen 5 9600X, Windows interactive desktop. No competing workload
required termination during these measurements. Prefill/training are unchanged.

## Acceptance

The short confirmation passes all four monitored activities and repeats positive
gains in both orders: 346.045 +/- 11.926 to 353.705 +/- 1.057 tok/s (+2.21%).
Pool both initial and confirmation samples without dropping any observation:
345.564 +/- 13.039 to 352.011 +/- 6.261 tok/s, **+1.87%** (20 samples each).
Baseline variance limits precision; this is a modest measured mean improvement,
not a claim of a statistically precise universal speedup. Middle/long remain
+1.13%/+0.91% with ten samples per cell. The wider isolated projection's -0.7%
result remains recorded; gains are not established for larger real models.

Retain the implementation: exactness and safety gates pass, all paired process
orders improve on the measured model, and another 44 launches per decoded token
are eliminated without added allocations. The matched-reference percentages
come from the complete separate reference run, not selection of fastest samples.
Short still needs further improvement to beat parity; the goal remains active.
Startup/module compilation and model loading are outside the timing scope.
