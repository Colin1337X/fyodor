# Separate gate and up warps, October 26, 2026

Gate and up each receive one full warp; four output rows share a 256-thread
block. Warp sums retain the native operation order, then use eight shared
floats and one block barrier before gate activation and multiplication. Invalid
rows skip the dot but still reach the barrier. The partial-row test cases cover
this behavior. Register use falls from 64 to 56, with zero local memory, yet
rotating model-sized kernels lose about 13-14%. More blocks and synchronization
are added; resource counts alone cannot establish which cost dominates.
Reject this assignment. Keep the current one-warp fused pair.

| Workload | Rows x columns | Candidate throughput change |
| --- | --- | ---: |
| hot | 9 x 256 | +11.59% |
| hot | 5632 x 2048 | -12.84% |
| hot | 11008 x 4096 | -12.98% |
| extended | 65 x 768 | +18.71% |
| extended | 257 x 2304 | -7.29% |
| extended | 28672 x 8192 | -11.73% |
| rotating | 5632 x 2048 | -14.36% |
| rotating | 11008 x 4096 | -12.61% |

Original Fyodor experiment under root optcpp/matvec, not runtime-selected or
CMake-compiled. Based on source revision 6fee1202be86f8ec5e3765c294a9707eca05dad3. The profile in
../decode-packed-load-20261026 identifies gate/up as about 40% of matrix time.
No new upstream source is copied. Removing optcpp introduces no build change.

Validation: all six hot shapes and both eight-bank rotating shapes match the
native output bytes and guards. Seeded random payload/metadata and bounded
inputs use fixed finite half scales. This is a diagnostic check, not complete
real-model/exceptional-value validation. Arithmetic and reduction order remain
unchanged. No production code or numerical tolerance changed.

All three activities pass continuous GPU competitor monitoring and idle gates;
no process needed termination. Resource counts, hardware snapshots, source
hashes, compile options, raw timings and environment observations are retained.
NVRTC compute_120/C++11/fmad, RTX 5060 Ti driver 617.42. Fifty kernels per graph,
three warmup replays, five samples of ten replays in each order; ten samples per
cell. summary.json reports means and sample standard deviations. Rotating banks
occupy 103809024 and 405798912 bytes (each layout, where two layouts are used).

These isolated timings do not update whole-model scores: **94.70% short /
114.23% long** relative to llama.cpp, below the **125% on both** requirement.
Run `python backend/benchmarks/decode-warp-pair-20261026/run.py` with fresh output directories.
Final process snapshot text has trailing whitespace normalized only.
