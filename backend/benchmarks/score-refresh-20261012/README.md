# Benchmark refresh — October 12, 2026

Fresh measurement of committed runtime `d451c0f`, using the same pinned
references, workloads, repetition counts and three-consecutive-sample desktop
idle gate as [October 11](../score-refresh-20261011/README.md). This is an
updated performance measurement, not an isolated estimate of a code change.

## Environment

Before timing, `nvidia-smi` showed no competing compute jobs or NVIDIA overlays.
Windows GPU-engine counters identified DWM and Codex as the active renderers.
No process needed termination or suspension in this refresh. Essential desktop
and development processes remained running; other recorded desktop identities
were retained. These are monitored desktop results, not headless results.
The inference harness's inherited metadata caveat about earlier stopped overlays
refers to the October 11 cleanup, not an action performed in this refresh.

Both comparisons reject new GPU identities or competing compute processes and
require three consecutive readings at or below 30% utilization and 15 W before
and after each activity. Training uses its independent monitoring process.
All samples from accepted runs are pooled; none are selected by speed.

## Reproduction

From the worktree root, run sequentially with fresh output directories:

```text
python backend/benchmarks/score-refresh-20261011/inference.py NEW_INFERENCE_DIRECTORY
python backend/benchmarks/score-refresh-20261011/training.py --output NEW_TRAINING_DIRECTORY
python backend/benchmarks/score-refresh-20261011/summarize.py NEW_INFERENCE_DIRECTORY --training NEW_TRAINING_DIRECTORY --output NEW_SCORECARD.json
```

The optional summary output preserves historical scorecards. Runtime binaries,
model and reference dependency hashes are recorded with the measurements.

## Results and validation

| Workload | Fyodor | Reference | Relative throughput |
| --- | ---: | ---: | ---: |
| Prefill 512 | 4,595.77 ± 33.02 tok/s | 15,021.92 ± 591.61 tok/s | 30.6% |
| Decode 128, prefix 0 | 255.56 ± 6.39 tok/s | 367.37 ± 1.39 tok/s | 69.6% |
| Decode 128, prefix 1024 | 230.67 ± 5.17 tok/s | 256.83 ± 4.50 tok/s | 89.8% |
| Small CUDA full-rank update | 7.699 ± 0.029 ms | 5.466 ± 0.450 ms | 71.0% |

All eight inference processes and nine monitored training activities passed.
All four training pairs passed the unchanged 1e-5 loss and optimizer-state
checks. Inference pools ten timed samples per workload and engine. Training
uses four paired means, each containing 19 timed updates after ten warmups
and excluding the final diagnostic update. ± is sample standard deviation:
across individual inference samples, across paired run means for training.

Training paired throughput ratios were 78.4%, 67.1%, 65.2%, and 73.3%.
The score is the ratio of the overall mean step times, not the best pair or
an average of ratios. This refresh lowers the previous 74.8% training score
to 71.0%; no samples were discarded. It does not isolate a code regression.
The references remain pinned llama.cpp b10809 with F32 KV and flash attention
off, and PyTorch 2.11.0+cu128 eager execution on the small decoder. These
scores do not establish parity with latest/fastest inference configurations,
compiled PyTorch, larger training models, or LoRA training in PyTorch.

Raw training checkpoints and exports remain in
`D:/fyodor-validation/score-refresh-20261012-training`; lightweight logs,
metrics, monitoring records and numerical results are checked in here.
No runtime code changed in this refresh. The summary tool gained `--output`
so a refresh can preserve earlier scorecards; the accepted October 11 data
was also re-summarized and its numerical fields reproduced exactly.
