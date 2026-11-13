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

No speedup is claimed until the full comparison finishes. Accepted Gemma scores
remain 85.6103% short / 106.8707% at 8K versus llama.cpp; 125% remains unmet.
