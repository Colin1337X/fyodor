# Launch bounds across actual Gemma projection shapes

Expands reuse-launch-20261114 to all 11 narrow-batch shapes in the recorded
projection profile, with batches 2 and 4. No runtime dispatch change is made.

The 34 correctness cases include the 11 actual shapes plus six small cases,
each at batches 2/4. All four bound/tile combinations per case match native
outputs byte-for-byte and preserve guards. Block scales vary across positive
and negative binary-fraction and non-binary values represented as F16. Small
256-column cases additionally pass the existing independent float64 bound.

Timings retain ten graph-event samples per cell, forward/reverse variant
order, three warmup graphs, and conversion-free native weight reuse. Synthetic
weights are repeatedly reused; these are diagnostic kernel times, not accepted
decode results. Actual occupancy and hardware counter causality are not measured.

| Type | Columns | Rows | Batch | Bound 3 speed % | Bound 4 speed % |
| --- | ---: | ---: | ---: | ---: | ---: |
| 12 | 3840 | 512 | 2 | 91.774 | 91.640 |
| 12 | 3840 | 512 | 4 | 156.218 | 162.203 |
| 12 | 3840 | 2048 | 2 | 96.122 | 96.117 |
| 12 | 3840 | 2048 | 4 | 114.698 | 116.237 |
| 12 | 3840 | 4096 | 2 | 95.903 | 95.877 |
| 12 | 3840 | 4096 | 4 | 101.297 | 102.329 |
| 12 | 3840 | 8192 | 2 | 95.942 | 95.765 |
| 12 | 3840 | 8192 | 4 | 100.477 | 101.361 |
| 12 | 3840 | 15360 | 2 | 96.544 | 96.486 |
| 12 | 3840 | 15360 | 4 | 100.767 | 101.730 |
| 12 | 4096 | 3840 | 2 | 95.848 | 95.837 |
| 12 | 4096 | 3840 | 4 | 101.491 | 102.607 |
| 12 | 8192 | 3840 | 2 | 96.214 | 96.100 |
| 12 | 8192 | 3840 | 4 | 104.668 | 106.331 |
| 12 | 15360 | 3840 | 2 | 100.766 | 101.002 |
| 12 | 15360 | 3840 | 4 | 106.199 | 108.190 |
| 14 | 3840 | 2048 | 2 | 100.851 | 100.819 |
| 14 | 3840 | 2048 | 4 | 102.003 | 100.726 |
| 14 | 3840 | 262144 | 2 | 99.929 | 100.046 |
| 14 | 3840 | 262144 | 4 | 101.476 | 100.354 |
| 14 | 15360 | 3840 | 2 | 102.492 | 102.478 |
| 14 | 15360 | 3840 | 4 | 104.644 | 103.633 |

Batch-2 Q4 regressions rule out a blanket launch-bound change. Batch 4
improves more consistently, including larger gains for small projections,
but whole-generation benefit remains unmeasured. The subsequent interleaved
input experiment shows larger gains without changing arithmetic, so it is the
current integration candidate. Launch-bound variants remain unintegrated.
