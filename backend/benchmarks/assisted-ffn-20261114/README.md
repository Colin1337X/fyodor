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
