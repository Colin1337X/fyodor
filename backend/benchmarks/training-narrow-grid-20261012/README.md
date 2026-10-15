# Distributing small LoRA projections across GPU blocks

The [matrix shape profile](../training-matrix-profile-20261012/README.md)
identified 130 dense 2×2048 projections costing about 24 ms per update in
rank-2 TinyLlama training. The existing ordered-double kernel assigned one
output to each lane but placed all 32 tokens in one block. At low rank, few
lanes were active and independent dots were concentrated on one multiprocessor.

The candidate dispatch distributes those independent dots across eight-thread
blocks for F32 matrices with at most eight output rows and at most 128 tokens.
Other shapes retain their original launch layout. Each dot keeps the same
ordered double products/additions and final F32 conversion. Matrix guards,
buffer validation and grid limits remain in force. The checked output byte
count bounds the flattened grid multiplication. This is original scheduling
code entirely within the detachable CUDA backend, with no public C ABI,
precision-mode or dependency change.

## Rejected unconditional layout

The first version applied small blocks at every token count. Initial screening
covered ranks 1/2/8/9, widths 2048/5632 and token counts 8/32/128. It improved
the targeted small matrices, and monitored full training preserved exact
checkpoint/export hashes. Those intermediate results are in `rank2/` and
`rank8/`; they are not the final policy's acceptance evidence.

An expanded screen caught a severe regression at 1,024 tokens: approximately
33–73% lower throughput for the changed ranks. Large batches already expose
parallelism, and too many partial warps undo the benefit. The unconditional
policy is rejected; `unconditional-rejected.patch` and `narrow-grid-wide-*.log`
preserve it. The final dispatch retains the original layout beyond four
32-token tiles. It does not claim eight-thread blocks are universally better.

## Verification and measurement

`nya-bench-training-device --narrow forward` and `--narrow reverse` now cover
48 matrix shapes: ranks 1/2/8/9, widths 2048/5632 and token counts
8/32/128/256/512/1024. Each validates forward, input-gradient and weight-gradient
results independently and checks transfer/synchronization counters. Timings
use two warmups and five samples of 20 dispatches per operation/order. The
`micro-policy/` evidence pools both orders; `summarize_micro.py` verifies shape
and sample counts. These short screens were not process-isolated and are
diagnostic only. In particular, changes to unaffected rank-9 shapes expose
environmental variation rather than a code improvement.

The final device tests add exact ordered-forward and guarded backward checks
at 128, 129 and 257 tokens for ranks 2 and 8. Existing tests cover ranks
1/2/7/8/9, input tails, single-token matrices and an 8192-token narrow matrix.
All 83 CUDA training tests pass. The C23 device suite and all four focused
CUDA sanitizer checks pass, including the new launch-policy boundaries.

Complete training uses TinyLlama Q4_K_M CPT, six CPU threads, 20 updates with
five warmups discarded and two process repetitions in reversed execution
order. Rank 2 covers contexts 8/32/128; rank 8 covers context 32. Every run must
pass the existing monitored desktop gate, and checkpoint/export hashes must
match exactly within each configuration. The reusable driver now accepts
`--rank`, with its previous rank-2 default preserved.

The controller runner temporarily suspends the verified nonessential overlay
user controller and resumes its held handle in `finally`. Essential desktop
rendering remains. A first launcher attempt used a wrong script-directory
date, failed before measurement and successfully resumed the controller;
the setup-failure records are retained and excluded from timing.

A final-policy attempt also failed the original three-consecutive-sample idle
gate after a measured process; it is excluded in `excluded-idle/`. Readings
cycled between 15% and 40% utilization at 6–9 W, preventing three consecutive
samples below 30%. Windows GPU-engine counters attributed the residual load
primarily to the essential desktop compositor. It was not stopped.

The subsequent fresh paired comparison explicitly enables `--desktop-window`:
ten consecutive samples must average at most 30% utilization, every sample
must stay at or below 45%, and every power reading must stay at or below 15 W.
The process-identity rejection rules are unchanged. This longer observation
window covers periodic desktop rendering instead of selecting a brief quiet
phase. It is a different recorded desktop protocol, not a headless benchmark
or proof of zero interference. Both baseline and candidate restart under the
same protocol; no samples from the failed attempt are combined with it.
The original three-sample protocol remains the default, including for the
existing inference/PyTorch score refresh. Six GPU-free tests verify that
default behavior, full-window requirements, mean/burst/power limits and
immediate rejection of competing processes.

To reproduce the complete comparison, use the saved baseline trainer and
`training-torch-20261011/narrow_compare.py --monitor-environment`, then validate
each output directory with `training-q4-dx-20261011/summarize.py`. Exact commands,
source/binary hashes and monitor data accompany the results. The small
all-F32 PyTorch comparison and inference scorecard are separate workloads;
these LoRA improvements must not be added to their percentages.

## Accepted final-policy results

All sixteen final-policy processes passed the windowed desktop protocol and
produced exactly matching checkpoint/export hashes within each configuration.
The controller was resumed successfully. The accepted raw records are
`final-rank2/` and `final-rank8/`; earlier directories retain intermediate or
excluded evidence. The final source and trainer hashes are in
`final-provenance.json`.

| Rank | Context | Baseline update ms | Final policy update ms | Throughput change |
| --- | --- | ---: | ---: | ---: |
| 2 | 8 | 291.396 ± 0.342 | 291.235 ± 0.050 | +0.1% |
| 2 | 32 | 316.808 ± 0.046 | 311.555 ± 1.017 | +1.7% |
| 2 | 128 | 678.148 ± 0.287 | 674.300 ± 2.266 | +0.6% |
| 8 | 32 | 379.465 ± 0.565 | 317.211 ± 0.447 | **+19.6%** |

Values pool both execution orders; ± is sample standard deviation across two
process means, each averaging 15 post-warmup updates. Context-8 rank-2 timing
is effectively unchanged. The rank-2 gains are modest; the substantial benefit
is the measured rank-8 case. These are hardware/workload-specific desktop
measurements on RTX 5060 Ti 16 GB, driver 617.42 and Ryzen 5 9600X. Do not
generalize the rank-8 gain to arbitrary models, ranks or batch sizes.

The small-batch policy is accepted. The unconditional layout remains rejected.
`validate_final.py final-rank2 final-rank8` rechecks every idle window, complete
run/configuration counts, monitoring results and serialized-output hashes,
then regenerates both summaries. Run it with paths relative to the current
working directory. No numerical tolerance was changed.
