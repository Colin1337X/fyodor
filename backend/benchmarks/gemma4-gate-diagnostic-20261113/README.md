# Gemma gate diagnostic timing — 2026-11-13

**Rejected for performance acceptance: competing desktop GPU applications remained
present. These measurements do not update llama.cpp or PyTorch scores.** No
acceptance gate or allowlist was changed. The separate diagnostic runner records
this rejection status unconditionally in metadata and every environment record.

Same C17 binary for baseline/fused, Gemma4 12B Q4_K_M, resident CUDA, F32 KV,
Gemma QKV enabled, full aligned byte-dot weights, 32 decoded tokens, host logits
per token, one warmup, three repetitions in each of forward/reverse orders.
Both short and 1K workloads actually use full packing: device weights are
13,194,985,664 bytes for every run. Thus these runs do NOT measure partial packing.

| Prepared prefix | Baseline tokens/s ± pooled SD | Fused tokens/s ± pooled SD | Diagnostic difference |
| --- | ---: | ---: | ---: |
| 0 | 39.8384 ± 0.3229 | 40.0216 ± 0.3030 | +0.46% |
| 1024 | 32.8266 ± 0.0109 | 32.4554 ± 0.5214 | -1.13% |

Six observations per cell; pooled sample standard deviation includes within-order
and between-order variation. Every fused run saves exactly 9,216 launches across
96 timed decode steps. The small/mixed throughput changes do not establish a
useful gain. Fusion and partial packing remain off by default. Timing under a
clean environment remains necessary; no causal speed conclusion is drawn here.

`run.py` preserves commands, flags, model/binary/monitor hashes, raw outputs and
one-second GPU/process/power/clock/CPU snapshots. `summarize.py` verifies all eight
successful runs, matching packed weight bytes and the expected launch reduction.
It never calculates a percentage of llama.cpp from these observations.

After all timing finished, `resources.py` queried JIT-compiled CUDA function
attributes on SM120 using the same kernel-source chain as the kernel oracle.
These are compiler/driver resource facts, not measured occupancy or stall counters:

| Kernel / variant | Registers per thread | Local bytes | Shared bytes |
| --- | ---: | ---: | ---: |
| Separate packed Q4 projection | 37 | 0 | 0 |
| Native vector activation | 17 | 0 | 0 |
| Current fused gate (outer unroll 8) | 64 | 0 | 0 |
| Fused gate, outer unroll 1 | 55 | 0 | 0 |
| Fused gate, outer and inner unroll disabled | 48 | 0 | 0 |

The first resource-only sweep also changed an unroll pragma in the unused native
matvec source; its `resources-unroll*.json` files are retained as preliminary
observations. `resources-isolated-unroll1.json` and
`resources-isolated-loops1.json` isolate changes to the Gemma gate source and
confirm the lower register counts. No runtime source was changed. Unroll variants
have not yet passed output parity, sanitizer or throughput tests. They are a
concrete next experiment, not accepted optimizations. Register pressure is a
plausible tradeoff to investigate, not a proven explanation for timing differences.
