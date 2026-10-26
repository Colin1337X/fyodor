# Transposed Q4 payload, October 26, 2026

Each Q4_K block retains its 144-byte size and unchanged 16-byte metadata.
The 128 payload bytes are transposed from four groups of 32 to 32 groups of
four. Each lane then loads one aligned word and extracts its eight nibbles
without cross-lane shuffles. This directly tests whether the shuffles in the
previous packed-load experiment obscured a useful memory-load reduction.
The host diagnostic packs weights before timing and retains both copies.
A runtime implementation would need an owned packing cache and fallback;
those costs are not represented here. Register use stays 64, zero local bytes.
Rotating cases lose 1.51% at 5632x2048 and 0.79% at 11008x4096. The hot
TinyLlama shape loses about 4.1%. Reject: no integration-worthy gain even
before charging the packing costs. Weight expansion is not the only reason
previous load/repacking experiments lost.

| Workload | Rows x columns | Candidate throughput change |
| --- | --- | ---: |
| hot | 9 x 256 | -0.04% |
| hot | 5632 x 2048 | -4.12% |
| hot | 11008 x 4096 | -1.13% |
| extended | 65 x 768 | +0.07% |
| extended | 257 x 2304 | -1.96% |
| extended | 28672 x 8192 | -0.35% |
| rotating | 5632 x 2048 | -1.51% |
| rotating | 11008 x 4096 | -0.79% |

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
Run `python backend/benchmarks/decode-transposed-payload-20261026/run.py` with fresh output directories.
Final process snapshot text has trailing whitespace normalized only.
