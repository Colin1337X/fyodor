# Four-warp K decomposition: prepared, not admitted

This original Fyodor experiment follows the correct but slower bounded TF32
variant in `../q4-tf32-chunked-20261114`. Each block now handles sixteen rows.
Its four warps process disjoint 256-column chunks with stride 1024, retaining
separate dominant/residual sums inside each chunk. Each warp writes valid
partials into 1 KiB shared memory; after a block barrier, warp zero sums the
four partials in fixed order. Warps without input chunks contribute zeros.
All block threads take the same out-of-range tile return before the barrier.

The motivation is the preceding candidate's nearly flat latency on small row
counts and long per-warp MMA dependency chain. This changes reduction order
and must establish its own correctness. No speedup is assumed.

Offline NVRTC compute_120 compilation and SM120 ptxas assembly passed. The
candidate uses 59 registers, 1024 shared bytes, one barrier and no spills.
Six CPU sanitizer-evidence validator tests pass. These do not prove GPU
correctness, synchronization safety, throughput or production readiness.

The prepared GPU harness retains the independent C double oracle, unchanged
`2e-6 * sum(abs(double products)) + 1e-6` bound, 219 cases, guards, and eight
unsupported-shape checks. Shapes include partial row tiles and widths not
divisible by 1024. Memcheck, racecheck and synccheck gate the timing probe.
The probe retains synchronized uploads, alternating order, per-output double
checks and the same eight shapes as the preceding experiment.

No GPU execution has occurred: the QKV 8K generation correctness process was
already running. Run this candidate only after that process is terminal.
Near-overflow/nonfinite fallback and real-model validation remain pending.
There is no production dispatch or mandatory build dependency. Accepted
llama.cpp scores remain 85.61025294% short / 106.8707255% at 8K.
