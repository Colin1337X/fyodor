# Decode dot occupancy investigation - October 16, 2026

Production remains at c3b909c. The previous turn accepted paired Q/K RoPE and
merged it into codex/native-training-execution. Current verified throughput
is 93.18% short / 114.31% long of pinned llama.cpp. The target is at least
125% on both; no score changes from this diagnostic investigation.

Fresh profiling records matrix 1018.789 ms, normalization 99.529 ms,
RoPE 29.676 ms, attention 80.782 ms and gather 1.358 ms. Matrix breakdown
places gate/up first at 414.218 ms, with residual projections and QKV next.
Profiling disables graphs and includes submission gaps; these are diagnostic
costs, not an end-to-end speedup estimate.

This original Fyodor probe tests whether the earlier gate/up launch-bound
benefit transfers to the shared standalone row-dot arithmetic. Wrappers use
launch_bounds(256, 2/4/8), with the same row indexing, accumulation and warp
reduction as production. Kernel labels u2/u4/u8 mean minimum-block bounds in
this script, not unroll factors. No production code is edited. Input uploads
are explicitly synchronized before the nonblocking kernel stream.

Both formats pass all six shapes and all three bounds with byte-exact output
and guards (36 candidate/shape combinations), including odd loop counts and
partial row blocks. Each cell preserves ten timings across forward/reverse
candidate order: graph 50 kernels, three warmup replays, five samples of ten
replays per order. These are hot-weight diagnostics with process snapshots;
there is no continuous idle/competitor gate, so they cannot update the score.

| Format | Rows x columns | Bound 2 | Bound 4 | Bound 8 |
| --- | --- | ---: | ---: | ---: |
| Q4_K | 9 x 256 | +0.41% | +0.46% | +0.01% |
| Q4_K | 65 x 768 | -0.07% | -0.09% | +3.39% |
| Q4_K | 256 x 2048 | -1.03% | -1.04% | +4.92% |
| Q4_K | 2048 x 2048 | +0.75% | +0.34% | +0.15% |
| Q4_K | 2048 x 5632 | +0.79% | +0.80% | +0.85% |
| Q4_K | 32000 x 2048 | +0.08% | -0.13% | -0.59% |
| Q6_K | 9 x 256 | -0.16% | -0.15% | -0.17% |
| Q6_K | 65 x 768 | -0.85% | -0.86% | -0.81% |
| Q6_K | 256 x 2048 | -3.23% | -2.94% | -1.23% |
| Q6_K | 2048 x 2048 | +0.33% | -0.27% | -0.97% |
| Q6_K | 2048 x 5632 | +0.20% | +0.17% | -0.86% |
| Q6_K | 32000 x 2048 | -0.13% | -0.20% | -0.54% |

Reject a general launch-bound change. Q6_K regresses at the 256-row and
32000-row target shapes; Q4_K has no consistent meaningful gain across the
internal and output projections. Driver attributes show Q4_K registers
36 -> 39/39/38 and Q6_K 40 -> 42/42/40, all with zero local bytes. Requested
compiler bounds do not prove actual resident block count. Avoid inferring
occupancy solely from this annotation.

No production build/regression suite is needed for this evidence-only change.
Both diagnostic processes and the profile exited zero. Future work should
reduce matrix instruction/memory traffic rather than transfer more compiler
hints across kernels. Reproduce with `python probe.py 12 NEW_DIRECTORY` or
`python probe.py 14 NEW_DIRECTORY`; bundled CUDA 13.1 NVRTC, compute_120,
RTX 5060 Ti 16 GB, driver 617.42, Windows. Source snapshot, hashes, compiler
options, device resource counts and every timing are retained.
