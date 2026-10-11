# Narrow dense CUDA training forward

The resident operation profile identified dense LoRA forward projections as the
largest measured TinyLlama training phase: about 281 ms per update at context 32,
versus 55 ms for frozen quantized forward operations. A rank-2 projection was
using the general 32x32 output tile: most output lanes were unused, yet each
input tile still required two block synchronizations.

The F32 training forward kernel now assigns one actual output to each lane
when there are at most eight output rows. It uses the same input-column order,
explicit double multiplication and addition, and final F32 conversion as the
general path. It needs no block synchronization. Quantized formats and output
row counts above eight retain the general path. This is original Fyodor CUDA
code; it introduces no foreign runtime, dependency, public ABI or precision
mode. CPU and inference kernels are unchanged.

`test_training_device.c` checks ranks 1, 2, 7, 8 and 9 with 2049 input columns and 35
tokens, comparing every forward value bit-for-bit against ordered C dots and
checking output guards and backward results. These cases run in the full
device suite and independently through `--narrow` for sanitizer inspection.
Existing coverage also exercises single-token and 8192-token narrow matrices.

The before/after driver is `training-torch-20261011/narrow_compare.py`. It uses
the saved previous executable and the candidate executable, TinyLlama Q4_K_M,
rank-2 LoRA, contexts 8/32/128, six CPU workers, three updates (first discarded),
and two process repetitions with reversed execution order. Profiling is off.
Whole-device utilization, memory, power, temperature and clocks are sampled
every 500 ms throughout each process. Checkpoint/export hashes are compared
exactly for every before/after repetition at each context. Exports are hashed
then removed only from their exact generated paths to bound disk usage.

Raw evidence is retained under `D:/fyodor-validation/narrow-compare-20261011`.
The benchmark remains a short internal comparison, not a PyTorch comparison.
NVIDIA overlay processes were present; no inference or training competitor was
intentionally run concurrently. Whole-process telemetry includes loading and
export, and does not isolate individual step memory or competing GPU activity.
Therefore these measurements must not be advertised as clean competitive wins.
The previously rejected larger-decoder PyTorch trajectory remains unresolved.

## Results

All twelve runs completed and checkpoint/export hashes matched exactly within
each context. `training-narrow-20261011/results.json` retains all step metrics,
binary/model hashes and commands. `summarize.py` reproduces these process-mean
results (sample standard deviations across two repetitions):

| Context | Previous step ms | Candidate step ms | Tokens/s before → after | Throughput gain |
|---|---:|---:|---:|---:|
| 8 | 551.19 ± 3.14 | 306.10 ± 7.91 | 14.51 → 26.14 | 80.1% |
| 32 | 582.11 ± 12.67 | 331.89 ± 8.62 | 54.97 → 96.42 | 75.4% |
| 128 | 961.20 ± 57.64 | 696.41 ± 17.68 | 133.17 → 183.80 | 38.0% |

The context-128 baseline varied more between orders; use the pooled result,
not the larger first-pair improvement. Hardware is RTX5060Ti 16GB, driver610.88,
Ryzen5 9600X. This remains a short, internally controlled diagnostic comparison
with the environmental limitations above, not a fresh PyTorch or llama.cpp score.

The CUDA training subset passed all 82 tests. After adding rank-boundary cases,
all five selected device tests passed. Focused `--narrow` runs under NVIDIA
memcheck, initcheck, synccheck and racecheck all completed without errors or
hazards; committed logs identify the tested tool output. No tolerance changed.
