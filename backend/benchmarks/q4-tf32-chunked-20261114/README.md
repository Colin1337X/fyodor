# Bounded three-product TF32 accumulation: correct sweep, rejected timing

Original Fyodor experiment in `optcpp/matvec/q4_tf32_chunked.cu`, compiled
after `q4_tf32_three.cu` for its conversion/MMA helpers. No production dispatch
or build dependency changes. This follows the numerical failure retained in
`../q4-tf32-three-20261114/oracle.json` without modifying that evidence.

The new variant retains hoisted native Q4_K metadata, four interleaved F32
inputs, and three TF32 products. It accumulates dominant and residual products
separately, resets both accumulators every 256 columns, and combines each
chunk into an F32 output sum. These two changes together repair the observed
sweep failure; the experiment does not isolate their individual effects.
SM120 assembly reports 56 registers, no spills and no barriers.

All 219 independent double-oracle cases and eight unsupported-shape checks
pass raw execution and Compute Sanitizer memcheck, racecheck and synccheck.
All tools report zero errors/hazards. The unchanged numerical contract is
`abs(error) <= 2e-6 * sum(abs(double products)) + 1e-6` on every output.
Maximum fraction of that bound is 0.203877150041 for this candidate and
0.051398116136 for the native comparison. Guards and unchanged inputs pass.
Six CPU tests exercise the sanitizer evidence validator. Independent audit
verified 7148 hashes across compilation, assembly, raw and sanitizer records.

This remains a bounded finite-input test, not universal F32 equivalence.
Near-overflow/nonfinite input fallback, real-model validation and runtime
integration have not been implemented. The failed predecessor is not promoted.

`probe.py` requires verified raw and all three sanitizer results before GPU
context creation. It synchronizes pageable uploads before nonblocking-stream
graph use. Each graph contains ten interleave-plus-matvec pairs, each sample
times 100 pairs, and each order warms up 1000 pairs. Four alternating orders
yield 40 samples per kernel per shape, with means and sample standard
deviations retained. Every timed output is also checked against the independent
double oracle; bitwise equality between precision paths is not required.

| Columns | Rows | Existing metadata kernel, ms | TF32, ms | Latency increase |
| --- | --- | --- | --- | --- |
| 3840 | 512 | 0.00697228 | 0.10879856 | 1460.45% |
| 3840 | 2048 | 0.01915490 | 0.10913431 | 469.75% |
| 3840 | 4096 | 0.03386390 | 0.11938931 | 252.56% |
| 3840 | 8192 | 0.06314378 | 0.13245660 | 109.77% |
| 3840 | 15360 | 0.12499907 | 0.20692463 | 65.54% |
| 4096 | 3840 | 0.03387526 | 0.13197433 | 289.59% |
| 8192 | 3840 | 0.06980763 | 0.26245709 | 275.97% |
| 15360 | 3840 | 0.14152082 | 0.47317173 | 234.35% |

Reject this variant for promotion: every tested shape regresses, including
the dominant 3840-column, 15360-row expansion. These repeated-weight-bank
microbenchmarks have process snapshots but no continuous activity acceptance;
they are diagnostics, never end-to-end llama.cpp scores. Accepted scores remain
85.61025294% short / 106.8707255% at 8K, below the 125% target.

The relatively flat small-row latency motivates investigating insufficient
parallel work and the serial MMA dependency chain. Splitting K across warps
or blocks would require explicit scratch/reduction correctness and measured
costs; it is a next experiment, not a demonstrated speedup.
