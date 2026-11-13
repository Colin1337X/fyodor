# Gate launch/loop scheduling diagnostic

Decision: retain the current runtime. This isolated sweep changes only outer-loop unrolling and the minimum-block argument of `__launch_bounds__`, leaving inner unrolling and arithmetic order intact. `u8b4` reproduces the current packed kernel; u1/b4 is modestly faster here, but the result is not an end-to-end acceptance or a justified default change. The earlier experiment disabling both loops was substantially slower; this sweep preserves inner unrolling. All twelve variants match separate packed projections plus GELU bit-for-bit for 15360 rows.

RTX 5060 Ti SM120, NVRTC 13.1, F32 activations, 15360 rows x 3840 columns, original synthetic weights. CUDA graphs contain twenty operations; three warmups, five samples of ten replays in each forward/reverse order. Results include ten raw samples per variant. Function attributes are JIT resource queries, not measured occupancy or hardware counters.

| Variant | Mean ms | Sample SD ms | Registers/thread | Local bytes |
| --- | ---: | ---: | ---: | ---: |
| separate | 0.198513 | 0.002807 | - | - |
| u1b1 | 0.189075 | 0.002705 | 55 | 0 |
| u1b2 | 0.189247 | 0.003796 | 55 | 0 |
| u1b4 | 0.190598 | 0.002821 | 55 | 0 |
| u4b1 | 0.209544 | 0.003065 | 101 | 0 |
| u4b2 | 0.212416 | 0.004810 | 101 | 0 |
| u4b4 | 0.193308 | 0.002672 | 64 | 0 |
| u8b1 | 0.282605 | 0.004041 | 212 | 0 |
| u8b2 | 0.205352 | 0.002275 | 128 | 0 |
| u8b4 | 0.199853 | 0.005992 | 64 | 0 |
| u15b1 | 0.194685 | 0.001921 | 96 | 0 |
| u15b2 | 0.195486 | 0.002393 | 96 | 0 |
| u15b4 | 0.206687 | 0.004126 | 64 | 0 |

**Not accepted performance evidence:** repeated synthetic weight banks and retained desktop GPU activity. GPU/environment snapshots, source hash, raw samples and resource queries are in `probe-results.json`. No llama.cpp/PyTorch score changes. Accepted Gemma ratios remain 85.61% short / 106.87% at 8K; the separate 143.24% long-decode result remains diagnostic. Neither experiment changes runtime dispatch. Reproduce with `python backend/benchmarks/gemma4-gate-schedule-20261113/probe.py`.
