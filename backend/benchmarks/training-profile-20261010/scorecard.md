# Performance scorecard

Requested reporting convention: include this scorecard in subsequent progress updates.
A score is Fyodor throughput / reference throughput × 100%. 100% is parity;
above 100% is faster. It is not an engineering completion estimate.

| Workload | Current controlled score | Evidence gap |
|---|---|---|
| Inference prefill vs llama.cpp | 30.7% historical | September 26 matched C API comparison; refresh current build |
| Inference short decode vs llama.cpp | 70.7% historical | Same matched comparison; refresh current build |
| Inference long decode vs llama.cpp | 87.1% historical | Same matched comparison, 1024-token prefix; refresh current build |
| Training vs PyTorch | Not measured | Match architecture, precision, optimizer, LoRA, loss weighting and device |

Report each shape separately, with repetition variance, correctness status and memory.
Do not average incompatible workloads into a single competitive percentage.
The older overview table has KV/warmup/semantic mismatches. The later September 26
comparison does align F32 KV, token IDs, batch limits and warmup, and supplies the
historical scores above. It uses llama.cpp b10809 with Flash Attention disabled;
these scores do not represent its fastest available configuration. Exact source
values and evidence hash are recorded in scorecard.json. The October 10 native training run establishes a
phase/memory baseline; it is not yet a PyTorch comparison. NVIDIA overlays
respawned after termination during the first run, so this run series is diagnostic
and must not be accepted as a clean competitive performance measurement.
