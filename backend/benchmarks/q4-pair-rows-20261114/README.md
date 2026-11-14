# Native Q4 two-row input reuse: preparation only

Original Fyodor experiment in `optcpp/matvec/q4_pair_rows.cu`. Each warp
computes two adjacent weight rows while sharing one interleaved float4 input
load. Native 144-byte Q4_K blocks, per-row arithmetic and shuffle reduction
order are retained. A tail row uses a valid dummy weight row and zero
coefficients; it never writes an invalid output. There is no shared memory
or block synchronization. The launch is 256 threads and sixteen rows/block.

This targets duplicated input-load instructions in the existing metadata
kernel after the TF32 experiments regressed. Register pressure is a possible
tradeoff: offline SM120 assembly reports 80 registers and no spills/barriers.
That report is not measured occupancy or proof of benefit.

NVRTC compilation and ptxas assembly passed. The reused compile harness
also compiles the earlier TF32 helper source; its legacy console label says
TF32, but the candidate entry is `nya_q4_pair_rows` and its code uses F32
scalar arithmetic, not MMA. No kernel has executed on the GPU yet because
the monitored QKV 8K run is active.

The prepared 224-case sweep requires both byte-identical output versus the
existing metadata kernel and the independent double-oracle bound of
`2e-6 * sum(abs(double products)) + 1e-6`. Guards, unchanged inputs and eight
unsupported-shape rejections are retained. Seven CPU tests validate sanitizer
evidence rejection, including absent/false/non-boolean byte-exact results.
Memcheck, racecheck and synccheck must pass before timing. The prepared
probe retains synchronized uploads, alternating order and double-oracle
checks on all timed outputs across eight shapes.

GPU numerical correctness, sanitizers, performance and real-model admission
remain pending. No production dispatch or mandatory build dependency changes.
Accepted llama.cpp performance remains 85.61025294% short / 106.8707255% at 8K.

Offline disassembly confirms eight `LDG.E.128` input-load instruction sites
for two rows, versus eight for one row in the existing metadata kernel's
retained instruction audit. Quantized byte-load sites double from four to
eight and FFMA sites from forty to eighty, consistent with two independent
row sums sharing the input loads. No HMMA sites occur in the candidate.
All nineteen disassembly evidence hashes independently verified. These are
static offline sites, not dynamic traffic, driver-JIT inspection, achieved
occupancy or measured performance; the 80-register tradeoff still needs testing.
