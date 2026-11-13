# Bounded verification output projection batching

The integrated MTP profile measured 71.65/84.46 ms in serial vocabulary
projection for verification windows 2/4. This candidate reuses existing batched
matrix dispatch for the output head under `NYA_CUDA_BATCH_LOGITS=1`.

The flag is off by default. Planning reserves at most four vocabulary rows,
using checked size arithmetic and the existing complete scratch/memory budget.
The reservation follows any reduction of the maximum batch. For Gemma's 262144
vocabulary, four rows add 3 MiB over the ordinary one-row buffer. Only count-2/4
verification uses batched projection; count-one and other counts retain the
existing path. No C++ dependency, new CUDA kernel or decode-time allocation is
introduced. Per-row synchronization/readback, finite checks, suppressed tokens,
softcap, transfer statistics and failure handling remain in place.

Eight new verification tests cover maximum batches 2/3/4/8 with linear/ring KV.
They compare against an independent serial run with the flag disabled and check
the exact scratch delta, unchanged KV/weight storage, full logits, guards, reset,
invalid inputs, rewind and rejected-suffix replay. Batches 3/8 exercise the serial
fallback. C17 and C23 each passed all 25 selected verification/fixture tests.

`validate.py` derives a count-2/4 real-model harness from the current verification
test. All four Gemma cases (linear/ring, count 2/4, prefix 257) pass, with maximum
scaled difference below 1.7e-5 against the unchanged 5e-4 tolerance. Trace counters
confirm two batched output projections per candidate plan. Source/library/binary
hashes and commands are recorded in validation.json. CPU and Vulkan verification
remain covered by the selected C17/C23 tests.

`memcheck.py` passed both the tiny model and real Gemma with zero CUDA memcheck
errors. Both resets passed; traces confirm eight tiny-model and two real-model
batched output projections. The real-model check includes a costly independent
serial prefix rebuild and must be allowed to finish before timing starts.
`physical-removal.log` records the successful independent C-only copy/build/test/
smoke run: 82 tests, including four expected ROCm/MLX hardware skips.

`run_generation.py` builds a separate executable with diagnostic matched B512
prompt ingestion and runs the same binary off/on at short and 8K contexts.
Weight reuse, wide target/borrowed attention and greedy early rejection remain
enabled in both. `summarize_generation.py` requires all 48 parity results, matched
binary/settings, event profiling disabled, and six candidate projection traces
per optimized context (batch-2/4, three rounds). Ordinary and serial MTP are
controls. Two warmed observations per mode are retained; round zero is excluded.
This is diagnostic full-generation timing, not a fixed-forward llama.cpp test.

## Full-generation results

All 48 generation calls passed exact parity, including cross-process reference
text agreement. Binary and settings checks passed. Both optimized contexts report
six projection traces: nine projections per batch-2 call and seven per batch-4
call. Controls report none; no CUDA event profiler was active.

Warmed decode-loop means and sample standard deviations in milliseconds:

| Context/mode | Off mean +/- SD | On mean +/- SD | On speed / off speed |
| --- | ---: | ---: | ---: |
| Short ordinary | 757.207 +/- 0.648 | 757.315 +/- 0.076 | 99.986% |
| Short serial MTP | 845.700 +/- 0.218 | 846.054 +/- 2.114 | 99.958% |
| Short batch 2 | 820.046 +/- 0.106 | 805.672 +/- 0.013 | 101.784% |
| Short batch 4 | 827.112 +/- 2.545 | 794.857 +/- 0.148 | 104.058% |
| 8K ordinary | 934.876 +/- 0.539 | 933.561 +/- 0.328 | 100.141% |
| 8K serial MTP | 1060.697 +/- 0.187 | 1060.876 +/- 0.489 | 99.983% |
| 8K batch 2 | 1143.469 +/- 0.291 | 1129.166 +/- 0.783 | 101.267% |
| 8K batch 4 | 1104.332 +/- 0.411 | 1073.151 +/- 2.218 | 102.906% |

Batch-2 saves about 14 ms per 32-token generation; batch-4 saves 31-32 ms.
The ordinary and serial-MTP controls vary by at most 0.15%. This supports keeping
the opt-in path, but every MTP case still trails ordinary generation. Do not
enable speculative decoding by default based on this result.

Both conservative desktop gates failed; all raw timing and environment evidence
is retained. There are only two warmed observations per mode, and off/on process
order was not reversed. These are internal diagnostics, not clean accepted
llama.cpp measurements. A correctness commit was created between the off/on
processes, so their revision labels differ; the executable and compared settings
are identical and verified by the summarizer. Production source did not change
during the comparison.

Accepted Gemma scores remain 85.6103% short / 106.8707% at 8K versus llama.cpp;
125% remains unmet. Remaining assistant host operations/transfers and target
matrix work require further optimization; output batching alone does not close
the measured gap.
