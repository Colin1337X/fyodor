# Integer scale-product gate/up diagnostic

Original Fyodor implementation, based on source revision 32b5de1. No upstream
code copied. This is a candidate for runtime integration, not an accepted
end-to-end improvement. Whole-model scores remain 96.33% short and 115.36% long
relative to llama.cpp (see decode-fixed-runtime-20261027).

## Mechanism and numerical contract

Replace `d * (float)scale * (float)q` with `d * (float)(scale * q)`.
Q4 scale is at most 63 and nibble at most 15, so the integer product is at most
945. A finite half coefficient has at most 11 significant bits; its product
with these integers fits exactly in F32. No storage format, precision mode,
subtraction, accumulation order, or warp reduction is changed. This is a
change to multiplication order, not a general reassociation permission.

The exhaustive C17 coefficient check compares all 65,011,712 combinations of
finite half bits, six-bit scale, and four-bit nibble, including signed zero.
All match bit-for-bit. Compile coefficient_check.c with GCC -std=c17 -O3
-ffp-contract=off and run; coefficient-check.json preserves its result.
NaN/Inf coefficients are excluded from this proof. GPU tests separately check
actual NVRTC contraction behavior: six widths, guard values, random metadata
and nibbles, and 14 half bit patterns spanning signs, zero, subnormal, normal,
and maximum finite coefficients. Every tested GPU output and guard matches
byte-for-byte. This does not replace real-model CPU-oracle validation.

## Measurement

RTX 5060 Ti, SM120, driver 617.42; NVRTC compute_120, C++11, fmad=true,
NYA_CUDA_FAST_MATH=1. Reference is the same-width fixed_width_gate.cu kernel,
not the generic kernel. Only width 2048 currently has runtime fixed dispatch.
Timing keys 1/2 mean reference/candidate, not warp counts (legacy probe names).
Ten samples per variant, five in each reversed ordering; each sample replays
500 kernels through CUDA graphs after warmups. Rotating cases use eight pairs
of weight banks, totaling 103,809,024 and 405,798,912 bytes respectively.
Raw samples, standard deviations, kernel resources, device limits, and GPU
process snapshots are retained. These are synthetic kernel diagnostics.

All six monitored activities passed idle/competitor gates; no competitor was
killed. The first rotating run is excluded because an attempted CPU compilation
may have overlapped it. rotating-confirm was run after compilation finished.
The monitor checks GPU competitors, not all CPU load, and no hardware profiler
counter evidence is claimed. Do not infer whole-model gains from this table.

| Case | Columns x rows | Reference ms | Candidate ms | Throughput gain |
|---|---:|---:|---:|---:|
| hot | 256 x 9 | 0.001114 | 0.001104 | +0.93% |
| hot | 2048 x 5632 | 0.042224 | 0.038284 | +10.29% |
| hot | 4096 x 11008 | 0.174982 | 0.162854 | +7.45% |
| extended | 768 x 65 | 0.001701 | 0.001688 | +0.74% |
| extended | 2304 x 257 | 0.006178 | 0.006242 | -1.03% |
| extended | 8192 x 28672 | 0.869444 | 0.803177 | +8.25% |
| rotating-confirm | 2048 x 5632 | 0.044243 | 0.041332 | +7.04% |
| rotating-confirm | 4096 x 11008 | 0.175088 | 0.163499 | +7.09% |

## Decision and reproduction

Proceed to an opt-in width-2048 runtime trial, preserving generic/fixed
fallbacks, then real-model correctness and matched short/long decode trials.
Do not enable every width: width 2304 regressed, and other widths have no
end-to-end evidence. Runtime code and CMake remain unchanged in this checkpoint;
removing optcpp still has the existing behavior. No new portability claim.

Run run.py and validate_range.py serially with no other builds/tests. Their
output directories must be absent or renamed before reproduction. Scripts
load the native sources and both optional device sources directly; source
hashes are recorded by each run. source-manifest.json hashes current sources;
the candidate's header comment was clarified after measurement, so combined
source hashes refer to the tested pre-comment form. Device instructions are
unchanged. summary.json contains aggregates; original JSON remains authoritative.
