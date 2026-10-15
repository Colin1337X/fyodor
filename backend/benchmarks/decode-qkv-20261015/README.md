# Decode grouped Q/K/V trial - October 15, 2026

Candidate combines eligible Q4_K query/key and Q4_K or Q6_K value projections
into one launch. Warp ownership, row accumulation and reductions stay the same;
block ranges select weights and outputs. No packing, allocation or extra copy.
Generic quantization formats, reference mode, shared-KV layers, Gemma and prefill
retain their existing paths. NYA_CUDA_QKV=0 disables the candidate for exact
comparisons. Matrix profiling labels the combined operation as qkv.

Initial hot-weight graph diagnostics passed exact output and canary checks,
including partial row blocks and both value formats. Throughput improved about
43-48% at TinyLlama shape and 14% at the wider 4096 shape. These are snapshots-
only diagnostics, not acceptance timings. Production correctness and monitored end-to-end timing passed. Retain QKV
fusion. Accepted score is now 91.2% short / 112.4% long.

C17 and C23 each passed 25/25 configured CUDA inference/quantization and shared
training-graph regressions. Actual production kernels match separate projections
on seven shapes/format combinations including partial row blocks, odd block
counts and widths through 8192. All four sanitizers pass with zero errors or
hazards, using bounded graph replay rather than instrumenting timing loops.

The real-model CPU logit oracle passes at unchanged tolerances. Dedicated QKV
on/off comparison checks 795 graph decodes across reset and prefix preparation:
logits and KV are byte-exact and 34,980 launches are eliminated. This does not
resolve the pre-existing strict CPU/GPU KV snapshot discrepancy. No tolerance
was loosened, and the unfused path remains available with NYA_CUDA_QKV=0.

## Accepted runtime measurements

All 12 paired activities and eight reference activities passed independent
process monitoring and before/after idle gates. No competing workload needed
termination. Every timed observation is included; each cell pools ten samples.

| Prefix | Baseline tok/s | QKV tok/s | Isolated gain |
| --- | ---: | ---: | ---: |
| 0 | 328.408 +/- 5.503 | 349.926 +/- 0.953 | +6.55% |
| 512 | 311.511 +/- 1.286 | 325.273 +/- 5.186 | +4.42% |
| 1024 | 288.367 +/- 2.720 | 302.891 +/- 1.032 | +5.04% |

The short baseline and middle candidate have higher variance; neither is
excluded. Improvement remains positive in both execution orders at all contexts.

| Decode prefix | Fyodor tok/s | llama.cpp tok/s | Relative throughput |
| --- | ---: | ---: | ---: |
| 0 | 351.359 +/- 0.596 | 385.335 +/- 0.832 | **91.2%** |
| 1024 | 304.001 +/- 0.345 | 270.490 +/- 0.553 | **112.4%** |

Matched reference: TinyLlama Q4_K_M, F32 KV, full GPU offload, pinned llama.cpp
b10809 (5266f24da75dc449bd56cbed7addb9c8e4a6a73e), reference FA off, six CPU
threads, maximum batch 512, decode-only 128 tokens, two warmups and five timed
repetitions per process, reversed engine order. Prefix preparation, model load
and tokenization are untimed; both engines return host logits per token.
RTX 5060 Ti 16 GB, driver 617.42, Ryzen 5 9600X, Windows interactive desktop.
These results do not claim parity across other models, hardware or settings.

The merged runtime reduces decode launches by 44 per TinyLlama token; counters
and graph replays are checked by summarize.py. No extra tensor allocation,
transfer or precision mode is introduced. Short remains below parity, so decode
stays the sole optimization priority. Prefill and training were not remeasured.
The evidence folder was initially named decode-qkv-20261016 and renamed after
the paired run to reflect the October 15 measurement date; original command
paths in that run's metadata are intentionally preserved.
