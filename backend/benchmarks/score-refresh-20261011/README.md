# Monitored benchmark refresh

See [the current scorecard](../SCORECARD.md). This refresh measures the committed
runtime at b8cf762, including the accepted narrow LoRA forward optimization.
The experimental Q4_K backward tile was restored before rebuilding the trainer
and running this comparison. It is not a production change. Inference code was
unchanged; exact binary and source hashes are recorded in the metadata.

## Measurement protocol

`inference.py` derives from the September 26 paired inference harness. It runs
Fyodor with cuBLAS F32 (CUTLASS disabled) and pinned llama.cpp b10809
(5266f24da75dc449bd56cbed7addb9c8e4a6a73e). TinyLlama Q4_K_M, deterministic
token IDs, F32 KV, batch 512, six CPU threads, reference FA off, two warmups and
five timed repetitions are matched. Two reversed-order engine pairs per prefix
produce eight processes. All samples are pooled, not selected by speed.

`training.py` retains the independent decoder and strict 1e-5 comparison from
the training oracle. F32 matmuls disable TF32; RMSNorm, RoPE and clipping use
double precision to match native semantics, and PyTorch uses fused AdamW.
Whole updates include preparation, loss observation, backward, clipping and
optimization. Four pairs alternate execution order. Ten warmups and the final
diagnostic update are excluded, leaving 19 timed updates per process. The
initial checkpoint creation and eight compared activities all passed monitoring.
Numerical checks passed for all pairs. The larger decoder's existing failed
parameter check remains a limitation; no numerical tolerance was relaxed.

`environment.py` monitors from a separate Python process so sampling does not
share the eager reference's Python interpreter lock. Samples include power,
utilization, temperature, memory and GPU process identities. Native child PIDs
are communicated atomically to the monitor. Monitors stop and are reaped after
each activity. Essential desktop C+G identities are recorded; new identities,
unapproved compute processes and overlays invalidate the training activity.

## Cleanup and residual desktop activity

Nonessential overlay processes and the interrupted experimental benchmark were
stopped. The overlay user controller respawned its children, so its verified
SPUser process was temporarily suspended. It was resumed after the final
benchmark; `cleanup/controller-resumed.json` records success. Display-driver
services and essential desktop processes were left running.

An initial <=5% utilization gate failed with no competing compute process.
Windows per-process counters identified DWM as the dominant residual activity:
roughly 23% at low idle clocks and only 6–8 W whole-GPU power. The explicit
desktop gate therefore requires three consecutive samples <=30% utilization
AND <=15 W before and after every measured activity. This is not a claim of
zero background GPU activity. The process/counter snapshots are in `cleanup/`.
Sampling cannot prove that no activity occurred between observations.

Excluded attempts are retained under `excluded/`:

- `new-browser-process`: inference gate detected a newly appearing Chrome GPU
  process; incomplete run discarded.
- `overlay-respawn`: interrupted when overlays reappeared; discarded.
- `strict-idle-failure`: no timed runs; essential compositor exceeded the
  original 5% gate.
- `shared-gil-monitor`: numerical checks passed, but an in-process monitoring
  thread could perturb eager Python dispatch. Its timings are excluded; the
  accepted rerun uses the separate-process monitor.

The accepted records are `inference/` and `training/`. Raw JSON, CSV and logs are
retained; generated models/checkpoints remain outside Git. These measurements
use driver 617.42 and should not be interpreted as isolated code gains against
historical measurements made with a different driver or environment.

## Reproduction

Use the repository's release CUDA build and the pinned local reference tools.
The training environment is Python 3.14 and PyTorch 2.11.0+cu128. From the
repository root, with nonessential GPU competitors cleared, run:

```text
python backend/benchmarks/score-refresh-20261011/inference.py NEW_INFERENCE_DIRECTORY
python backend/benchmarks/score-refresh-20261011/training.py --output NEW_TRAINING_DIRECTORY
python backend/benchmarks/score-refresh-20261011/summarize.py NEW_INFERENCE_DIRECTORY --training NEW_TRAINING_DIRECTORY
```

The scripts expect Windows, nvidia-smi and local `.tools` binaries/model paths.
Run sequentially and use fresh directories. The summarizer requires all eight
inference processes, all nine monitored training activities and all four
numerical comparisons to pass. It regenerates `scorecard.json` and pools every
accepted sample. Checked-in accepted evidence can also be passed to it directly.
