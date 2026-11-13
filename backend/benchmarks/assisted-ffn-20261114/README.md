# Native assisted feed-forward residency (2026-11-14)

Experimental, default off: `NYA_CUDA_ASSISTED_FFN=1`. Only the MTP assistant
attempts this path. The target runtime and other backends retain their current
execution. This is original Fyodor C host code using existing native CUDA kernels;
there is no optional C++ dependency.

The preceding assistant stage profile identified feed-forward execution as the
largest short-context assistant stage. This change keeps gate/up intermediate
activations on device across the three projections and activation. Per chain,
it uses one host input upload and one output download rather than three of each.
These are source-level transfer counts, not hardware profiler measurements.

The plain C descriptor specifies exact matrix storage sizes, shapes and types.
CUDA validates those sizes, block alignment, arithmetic, weight-cache admission
and total scratch budget before execution. Partial allocations belong to the
context immediately. A runtime failure drains queued work, disables that context
and requests complete caller recomputation. Unsupported backends decline the
operation. GPU failure injection has not been performed.

## Correctness evidence

- C17 and C23: four tests each (direct FFN plus three MTP bridge variants).
- Direct oracle: F32, F16, BF16, Q4_0, Q8_0, Q4_K, Q6_K; GELU and SiLU;
  two input patterns, cache reuse, output guards, invalid descriptors and F32
  dimensions 33/65. Analytic double-precision oracle, unchanged scaled bound
  0.0005. Quantized fixtures use row-varying constant weights, not arbitrary
  quantization distributions.
- Compute Sanitizer: zero errors across the direct cases (`memcheck.log`).
- Production Gemma + Q4_0 MTP: all 12 short-context generation calls matched
  ordinary greedy output exactly (32 output tokens); 828 FFN chain dispatches.
  This validates output behavior, not equality of every intermediate logit.
- Optional-component removal evidence is in `physical-removal.log`.

## Timing contract

`run_generation.py` builds a diagnostic B512 prompt-ingestion harness and runs
same-binary feature-off/on comparisons. Output projection batching, wide target
batches, borrowed wide attention, weight reuse and greedy early rejection are
held enabled. Each short/8K context has three rounds of four modes, with round
zero excluded and round one reversing mode order. Only two warmed observations
per cell; off/on process order is not reversed. Decode wall time includes the
assistant, sampling and transfers and excludes prompt ingestion and teardown.
32 output tokens include the initial prefill-logit sample. This is not the
llama.cpp fixed-forward comparison contract. CUDA event profiling is disabled.

`idle-gpu.json` records five pre-validation Windows per-engine samples. No
positive activity appeared on the RTX 5060 Ti adapter; desktop/browser activity
was on the other adapter. No process was terminated. Missing activity in this
short sample is not continuous clean-run proof. The historical desktop gate is
retained, and any timing results here remain diagnostics, not accepted scores.

Accepted Gemma/llama.cpp decode remains 85.61025294% short and 106.8707255% at 8K.
The requested 125% on both contexts remains unmet.

## Paired results

All 48 generation calls passed exact output parity. Both off/on processes used
one identical binary and matched recorded settings except ASSISTED_FFN. The off
metadata names revision 2f7f1d9 and on names 280c522 because the validated source
was committed while the control process was live; the binary did not change.
Both historical desktop gates rejected acceptance. Raw one-second resource
snapshots, environments and hashes are retained in runtime-off/runtime-on.

Warmed decode-loop means and sample standard deviations (milliseconds):

| Context / mode | Disabled | Enabled | Enabled speed / disabled |
| --- | ---: | ---: | ---: |
| Short ordinary | 755.945 +/- 0.181 | 758.368 +/- 0.118 | 99.6804% |
| Short serial MTP | 844.267 +/- 0.725 | 828.295 +/- 1.098 | 101.9284% |
| Short batch 2 | 804.826 +/- 0.760 | 788.602 +/- 1.502 | 102.0573% |
| Short batch 4 | 790.948 +/- 1.853 | 772.436 +/- 3.247 | 102.3965% |
| 8K ordinary | 934.589 +/- 1.491 | 934.252 +/- 1.635 | 100.0361% |
| 8K serial MTP | 1059.985 +/- 0.186 | 1042.186 +/- 0.190 | 101.7078% |
| 8K batch 2 | 1128.889 +/- 3.383 | 1110.890 +/- 3.032 | 101.6202% |
| 8K batch 4 | 1069.829 +/- 0.531 | 1041.498 +/- 1.668 | 102.7202% |

The feature saves about 16-19 ms short and 18-28 ms at 8K per generation.
Ordinary decode does not execute this path; its small variation is a control,
not a feature gain. Every MTP mode remains slower than ordinary generation.
With only two warmed observations and fixed off/on process order, these are
limited internal diagnostic results. The default remains off. No llama.cpp or
PyTorch score is promoted.

## Admission recovery

The additional 1 MiB-budget test submits a 1.5 MiB F32 weight set, verifies
refusal leaves every output element unchanged, then executes small GELU and
SiLU chains successfully on the same context with exact-zero outputs and intact
guards. This tests preflight admission recovery, not a driver allocation or
kernel failure. C17/C23 results and the separate CUDA memory-check log are
retained alongside the original correctness evidence.
