# Four-warp K decomposition: correct sweep, slower than existing kernel

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
`2e-6 * sum(abs(double products)) + 1e-6` bound, 224 cases, guards, and eight
unsupported-shape checks. Shapes include partial row tiles and widths not
divisible by 1024. Five additional 17-row cases at 512, 768, 1024, 1280
and 1792 columns exercise two/three/four active warps and uneven chunk counts. Memcheck, racecheck and synccheck gate the timing probe.
The probe retains synchronized uploads, alternating order, per-output double
checks and the same eight shapes as the preceding experiment.

GPU execution waited until the QKV 8K generation correctness process exited
successfully. All 224 numerical cases and eight rejections then passed raw,
memcheck, racecheck and synccheck, with zero tool errors/hazards. The maximum
error fraction of the unchanged double-oracle bound was 0.152658571397197.
All 7148 evidence hash checks independently verified.

The timing probe completed all eight shapes with output checks, synchronized
uploads, alternating execution order, 40 samples per kernel/shape, and 100
interleave/matvec pairs per sample. Means and sample standard deviations are
retained in `timing-upload-sync.json`.

| Columns | Rows | Existing kernel, ms | Split TF32, ms | Latency increase |
| --- | --- | --- | --- | --- |
| 3840 | 512 | 0.00697413 | 0.02798122 | 301.21% |
| 3840 | 2048 | 0.01934360 | 0.03192070 | 65.02% |
| 3840 | 4096 | 0.03397621 | 0.05377991 | 58.29% |
| 3840 | 8192 | 0.06321168 | 0.09696297 | 53.39% |
| 3840 | 15360 | 0.12515041 | 0.17494347 | 39.79% |
| 4096 | 3840 | 0.03398850 | 0.05500026 | 61.82% |
| 8192 | 3840 | 0.06983489 | 0.10863853 | 55.56% |
| 15360 | 3840 | 0.14156742 | 0.20193416 | 42.64% |

Splitting K improves on the preceding TF32 experiment but still loses on
every shape against the existing metadata kernel. Reject promotion. These
are repeated-weight-bank diagnostics with snapshots, not continuous activity
acceptance or end-to-end decode results. Near-overflow/nonfinite fallback
and real-model validation remain pending. There is no production dispatch or
mandatory build dependency. Accepted llama.cpp scores remain 85.61025294%
short / 106.8707255% at 8K; the 125% target is unmet.
