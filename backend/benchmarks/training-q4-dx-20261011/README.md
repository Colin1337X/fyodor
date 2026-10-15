# Q4_K backward-input reduction tile

This follows the [parked candidate](../training-q4-dx-pending-20261011/README.md)
and the [rejected universal tile experiment](../training-dx-tile-20261011/README.md).
The mapped backward operation was the largest group in the post-narrow-forward
LoRA profile. A 64-wide reduction tile improved Q4_K primitive timings but hurt
Q6_K, so this candidate specializes only Q4_K and retains 32 for every other
format. Shared memory is 16,640 bytes per block for Q4_K instead of 8,448.
Each output retains the original ordered F32 products and additions. Kernel
names and the native C ABI are unchanged. This is original Fyodor scheduling
work, with no newly adapted third-party source or optional dependency.

## Validation and acceptance

The candidate passes all 82 CUDA training tests. The focused Q4_K suite covers
reduction lengths 63/64/65/127/128/129, token tails and two accumulated backward
passes with a bitwise ordered-F32 oracle. CUDA memcheck, initcheck, synccheck
and racecheck all pass this suite with no errors or hazards.

The complete-training comparison uses TinyLlama Q4_K_M, rank-2 LoRA CPT,
contexts 8/32/128, 20 updates with five warmups discarded, six CPU threads and
two process repetitions in forward/reverse executable order. The baseline is
the trainer built from the committed runtime before this candidate. Model,
source and executable hashes, commands and all per-update metrics are retained.
All checkpoint and exported model hashes must match across versions and runs.

The comparison driver now supports `--monitor-environment`. It uses the same
separate-process monitor and documented desktop gate as the
[score refresh](../score-refresh-20261011/README.md). An overlay user controller
is temporarily suspended with resumption in a `finally` block; only identified
nonessential overlays are stopped. Each benchmark activity must pass idle
checks and contain no unapproved GPU process. Essential desktop rendering
remains active. This is a monitored desktop comparison, not headless timing.

## Reproduction

Keep the committed baseline executable before rebuilding the candidate, then
run the following sequentially with nonessential GPU competitors cleared:

```text
python backend/benchmarks/training-torch-20261011/narrow_compare.py --output NEW_DIRECTORY --before BASELINE_EXE --contexts 8 32 128 --steps 20 --warmup 5 --repeats 2 --monitor-environment
python backend/benchmarks/training-q4-dx-20261011/summarize.py NEW_DIRECTORY
```

The summarizer rejects incomplete runs, failed environmental checks and output
mismatches. It pools all process means and reports their sample standard
deviation; it never selects the fastest result. Models/checkpoints remain
outside Git while their exact hashes are retained. This LoRA experiment is
separate from the small full-rank PyTorch and inference scorecard workloads.

## Accepted result

All twelve runs passed their environment gates and produced exactly matching
checkpoint and exported-model hashes within each context. All 82 CUDA training
tests and all four focused sanitizers passed. The overlay controller resumed
successfully after measurement (`resumed.json`). No overlay children were
present when this run started. The Q4_K-only specialization is accepted for
its repeatable shorter-context benefit. It adds no measured longer-context
improvement.

| Context | Baseline step ms | Specialized step ms | Throughput change |
| --- | ---: | ---: | ---: |
| 8 | 300.532 ± 0.225 | 290.678 ± 1.586 | +3.4% |
| 32 | 333.189 ± 6.319 | 314.375 ± 1.089 | +6.0% |
| 128 | 672.631 ± 8.441 | 674.201 ± 1.204 | −0.2% |

Values pool both execution orders; ± is sample standard deviation of two
process means. Context 128 varies more across the baseline repetitions and
its difference is within that observed variation; do not describe it as a
speedup. The much larger improvement suggested by the earlier interrupted,
uncontrolled experiment did not survive these controlled measurements.
Hardware is RTX 5060 Ti 16 GB, driver 617.42, Ryzen 5 9600X on Windows.

`results.json` retains all 240 update metrics, commands and exact output hashes;
`environment.json` retains the activity samples. `summary.json` reproduces the
pooled comparison. Tests and sanitizer logs are adjacent. This change affects
quantized training backward operations, not inference or the all-F32 small
PyTorch comparison. Their measured scorecard percentages remain unchanged.
`full-build.log` and `relinked-tests.log` confirm that all CUDA targets were rebuilt before the final 82/82 training-suite pass, so every statically linked test used the candidate kernel.
