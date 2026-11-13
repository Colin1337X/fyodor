# Native narrow-batch launch-bound experiment

Original Fyodor variants in `optcpp/matvec/reuse_launch.cu` derive directly from
small_batch_reuse.cu, changing only symbol names and the 256-thread launch bound
(minimum blocks 3/4/6). No runtime or build integration is added.

All 72 shape/batch cases pass byte-exact comparison and output guards: both
Q4_K/Q6_K, six shapes including rows 1/7/9 and actual Gemma gate/up/down shapes,
batches 1/2/3/4/5/8, six candidate bound/tile combinations each. The small
256-column cases also pass the existing independent float64 error bound.
Synthetic block scales are fixed exact binary fractions. Model-logit validation
and memory sanitization have not been run for these unintegrated candidates.

The same synthetic-weight-bank graph methodology as packed-batch-20261114 is
used, with forward/reverse variant order and ten samples per cell. Results are
diagnostics, not whole-model throughput or a clean accepted benchmark.

| Shape / batch | Bound 3 speed % | Bound 4 speed % | Bound 6 speed % |
| --- | ---: | ---: | ---: |
| Q4 gate/up, 2 | 96.514 | 96.516 | 99.973 |
| Q4 gate/up, 4 | 100.665 | 101.624 | 99.997 |
| Q6 down, 2 | 102.477 | 102.465 | 102.768 |
| Q6 down, 4 | 104.332 | 103.484 | 85.622 |

Native unbounded kernels use 40 registers for Q4 batches 2/4 and Q6 batch 2,
and 48 for Q6 batch 4. Bound 3 instead uses 51/72/52/80 respectively. This
contradicts a simple assumption that fewer registers must run faster. Actual
occupancy and causal hardware-counter attribution were not measured. Local
memory and shared-memory sizes are recorded in kernel-resources.json.

Retain as an experimental candidate only. The small, shape-dependent benefit
and major bound-6 Q6 batch-4 regression do not justify a general dispatch change.
Any integration needs broader shapes, real-model parity and paired decode
measurements. Accepted llama.cpp ratios remain unchanged.
