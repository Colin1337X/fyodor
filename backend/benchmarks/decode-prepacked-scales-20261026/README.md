# Precomputed Q4 scale layout, October 26

Original Fyodor diagnostic, not selected by CMake or runtime. The profile in
../decode-packed-load-20261026 identifies gate/up as about 40% of matrix time.
This experiment replaces each 144-byte Q4_K block with a 192-byte block: eight
float scales, eight float minima, then the unchanged 128-byte nibble payload.
Precomputing the half-times-six-bit integer products removes scale extraction
and repeated products in the inner decode loop, at 33.3% higher weight storage.
It preserves lane mapping, accumulation and warp reduction order.

All six hot shapes and both eight-bank rotating shapes matched native output
bytes and guards exactly. Inputs use seeded random payloads/metadata, bounded
input vectors, and fixed finite half scales. This is not comprehensive real-model
numerical validation, NaN handling, or a new supported precision contract.

The layout loses 8.86% at 5632x2048 and 8.47% at 11008x4096 with rotating weights.
Hot model-sized cases lose roughly 9-14%. Both kernels use 64 registers and zero
local-memory bytes. Reject it: removing arithmetic did not compensate for the
larger layout. This does not reject all repacking strategies; future candidates
should avoid scale expansion or change cross-row access, backed by measurements.

All three activities passed continuous competitor monitoring and desktop idle
gates; no competing GPU process needed termination. The monitor retains power,
temperature, utilization and process observations. Timing is diagnostic CUDA
graph replay, 50 launches per graph, three warmup replays, five samples of ten
replays per forward/reverse order. Raw samples, compile options, source hashes
and hardware snapshots are retained. Rotating results.json weight_bank_bytes
names ORIGINAL storage; summary.json also records the larger candidate storage.

Run `python backend/benchmarks/decode-prepacked-scales-20261026/run.py` with fresh
output directories. NVRTC compute_120, C++11, fmad enabled; RTX 5060 Ti driver
617.42. Host packing is untimed and the diagnostic retains both layouts, so it
is not a load-time or peak-runtime-memory benchmark. No runtime changes or
whole-model score update: 94.70% short / 114.23% long against llama.cpp, below
the required 125% on both. Removing optcpp has no new build dependency.
