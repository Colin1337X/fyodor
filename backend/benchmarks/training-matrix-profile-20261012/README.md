# Resident training matrix shape profile

`NYA_TRAIN_PROFILE=1` now attributes resident matrix time by storage format,
mapped/dense operation, input/output dimensions, token count and requested
input/weight gradients. Forward and backward totals reuse the same measured
intervals as the operation aggregates; no additional GPU synchronization is
introduced. Backward includes both dX and dW when both are requested, and finite
checks remain inside the diagnostic intervals. These are synchronized host
diagnostics, not isolated kernel times or throughput acceptance.

The table is optional host storage allocated only for profiled resident graphs.
It holds 63 distinct keys plus one overflow bucket. Repeated keys still reuse
their original entries after overflow. Allocation failure retains aggregate
profiling and reports `available=0`. The table is freed with the graph, including
partial construction. Its byte size is reported separately from the graph's
training-memory budget. Normal execution adds no table allocation or profiling
fences. The graph descriptor gains one private pointer; the public C ABI,
training arithmetic and backend dependencies do not change.

The driver `training-torch-20261011/resident_profile.py` now checks that every
matrix call and measured interval reconciles with its operation aggregate.
Printed-time comparisons allow only the derived decimal-rounding bound. It
retains all individual shape summaries alongside original output hashes.

## Findings

TinyLlama Q4_K_M, rank-2 LoRA, context 32, six CPU threads, RTX 5060 Ti. These
ranges cover updates two and three; all three updates are preserved in
`results.json`. The interactive desktop was not isolated or process-monitored
for this diagnostic. Do not interpret these values as clean competitive timing.

| Matrix (outputs × inputs) | Format | Calls/update | Forward ms/update |
| --- | --- | ---: | ---: |
| 2 × 2048 | F32 dense | 130 | 24.19–24.36 |
| 5632 × 2048 | Q4_K mapped | 44 | 21.27–22.98 |
| 2048 × 2048 | Q4_K mapped | 43 | 9.57–10.31 |
| 2 × 5632 | F32 dense | 22 | 9.43–9.71 |

Although mapped forward is the largest aggregate operation group, the single
largest forward shape is the narrow F32 LoRA projection. The existing narrow
kernel preserves ordered double dots but launches only one 32-token block for
this workload. This motivates investigating work distribution for narrow
projections before another broad quantized-tile change. The Q4_K 5632×2048
shape remains a substantial target, including 24.96–26.12 ms of backward time.

The random-init width-256, FF-704, two-layer, context-128 fixture attributes
2.20–2.37 ms to four 704×256 forward projections. Its profile-on and profile-off
checkpoint/export hashes match exactly. No new throughput gain is claimed.
The established inference and PyTorch scorecard remains unchanged.

## Validation

- All 83 CUDA training tests and 12 CPU training tests passed after rebuilding.
- C17 and C23 each passed the profile lifecycle and shape-overflow tests.
- The new fixture executes 140 matrix calls across 70 shapes, twice each,
  checks every output and verifies 63 reused entries plus 14 overflow calls.
  Its CMake verifier checks both process exit status and reported counts.
- The full diagnostic run reconciles per-shape counts and times for every
  graph, and checks profiling transparency through exact serialized outputs.

Reproduce with:

```text
python backend/benchmarks/training-torch-20261011/resident_profile.py --output NEW_DIRECTORY --context 32 --rank 2
ctest --test-dir build-cuda -R training-graph-cuda-profile --output-on-failure
```

This is original C profiling code with no external source adaptation. No
precision contract or numerical tolerance was changed.
