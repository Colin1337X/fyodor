# Isolated full-model batched CUDA graphs

`transform.py` generates an isolated C host implementation from the validated
native-metadata/readback baseline. Production sources and the production archive
are unchanged. `build_model.py` verifies the baseline full-model sanitizer and
the request/gather graph sanitizer before generating and linking C17/C23 model
and lifecycle binaries. This is original Fyodor implementation informed by the
llama.cpp graph audit, without importing ggml objects or its graph runtime.

Opt-in `NYA_CUDA_BATCH_GRAPHS=1` admits verification batches two/four only when
ordinary graph execution is enabled and the complete output bank is allocated.
Other batches continue through ordinary lowering. The five-word device request
updates outside capture on the same stream. Linear and ring layouts reserve
batched key/value scratch, including allocation-budget batch reduction.

Six independent cache slots distinguish batch two/four and starting positions
below 256, 256..8191 and 8192 onward. The existing single-token `graph_index`
keeps its original attention semantics. Capture ends after all output kernels
and before host synchronization/download. Both new capture and cached replay
use the existing contiguous bank reader, output validation and state commit.
Reset/rewind retain topology caches while updating the request for each call.
All six executable handles are released with the plan; capture errors use the
existing capture termination, stream synchronization and poisoned-plan fallback.

Replay advances recorded logical kernel and operation counters instead of
silently skipping their host-side increments. Readback transfers and batched
logit counts remain outside capture. Current `nya_blas_bytes` declines batch
sizes below 32, so the admitted two/four-token path uses native matrix kernels;
no vendor GEMM capture compatibility is claimed.

## Evidence and remaining work

Both C17 and C23 model/lifecycle executables compile and link successfully.
`run_lifecycle.py` retains the original 0.0005 scaled-error bound and all existing
invalid-input, reset, rewind, replacement, KV and output guards. It additionally
requires one graph replay and one download for each four-token verification,
and a terminal graph count with more replays than captures, preventing a silent
fallback or capture-only run from passing.

C17 and C23 at prefix 257 each passed both reset paths with maximum scaled error
2.19546455e-5, one capture and two batched graph replays. Each retained the
single-download assertions and recorded 568 integer-kernel calls, including
replayed operations. The result hashes were independently rechecked. The C17 8K
lifecycle check also passed both reset paths, with maximum scaled error
1.63989979e-5, one capture, two replays and 568 integer calls. All 148 hashes
verified after completion.

`run_generation.py` is prepared to require both short lifecycle builds and the
8K lifecycle before testing twelve ordinary/serial2/batch2/joint3 cases. It also
requires six target plans with more batched replays than captures (three batch2
and three joint3 runs), so matching output through fallback alone cannot pass.
Short-context C17 and C23 generation each passed all twelve ordered output-parity cases,
including the required six plans with actual cached batched replay. This is
correctness evidence only; no continuous activity coverage or accepted timing
is provided by that generation run.

`run_model_memcheck.py` is prepared for the new binaries and requires successful
generation and lifecycle proofs. Its validator retains exact case order,
metadata selection, integer execution, output parity and one zero-error summary,
and also requires capture plus reuse in six verification cases. Nine CPU tests
pass, including missing/duplicated graph evidence and capture-without-reuse.
The full-model short graph sanitizer passed all twelve ordered parity cases,
with one zero-error summary and exit code zero. All 165 recorded hashes
independently verified. Six verification plans captured and reused batched
graphs (batch2: one capture/nine replays; joint3: two captures/ten replays).
The run took 2741 seconds. A competing Roblox process was observed and stopped
during this correctness run; `gpu-cleanup-roblox.json` records the cleanup.
Sanitized call times are excluded from performance acceptance. Long-context
generation, graph-specific recovery checks and controlled timing remain pending.

`boundaries.c` and its build/run scripts prepare additional F32 fixture coverage
for head widths 256/512 and linear/ring KV. Within the same plan, tests advance
through positions 250/254/256/258/8190/8192/8196 with batches two/four, changing
tokens after rewinds and repeating after reset. The independent session disables
graphs and produces serial-reference logits. Assertions require exact cache
capture/replay deltas, six captured topologies, 42 verifications, output guards,
one download each and the unchanged 0.0005 scaled-error bound. C17/C23 fixture
binaries compile successfully. All eight GPU fixture combinations (C17/C23,
width 256/512, linear/ring KV) passed 42 cases each: 336 total verifications.
Every fixture reported six captures and 42 replays, including changing-token
rewinds and the second reset cycle. Recorded hashes independently verified.
These small F32 cases complement real-model quantized validation, not replace it;
they have not yet been repeated under Compute Sanitizer.

Real-model lifecycle results, generation parity, full-model memory checks,
two-token and topology-boundary coverage and controlled timing must establish
runtime correctness before production admission. No speedup is claimed from
successful compilation or from the isolated request/gather test alone.

Accepted decode remains 85.61025294% short and 106.8707255% at 8K relative to
llama.cpp. The 125% target on both contexts is still unmet.

## Prepared monitored timing

`run_monitored.py short|long LABEL` retains the process/idle gate, continuous
activity collector, exact generation-case inventory, integer dispatch and
before/after input hashes. It additionally requires capture and reuse in six
verification cases. Before timing it verifies the short full-model sanitizer,
both lifecycle lengths, all eight boundary fixtures, C23 short generation and
C17 generation for the requested context. Thus short timing does not depend on
an unrelated long-generation run, while long timing cannot skip its own parity
proof. Four CPU tests pass and refusal of the pending sanitizer was exercised
without creating timing output or starting another GPU workload. This runner
has not yet produced timing evidence, and does not itself declare benchmark
acceptance from coverage alone.

## Matching generation reference

The historical reference runner's tokenizer proof referenced an older
`build-cpu/libnya-engine.a`. `compare_prompt_tokens.py` preserves that evidence
and rebuilds separate CPU-only C17/C23 tokenizer executables. The C17 runs
match llama.cpp token IDs exactly at 25 and 8205 prompt tokens. The new proof
is under `prompt-token-parity`; 193 available prerequisite hashes verify.
`run_reference.py` uses this proof and this candidate's generation/sanitizer
proofs. Its process gate refused launch while the model sanitizer was active.
No new reference GPU timing has run.

Generation decode intervals exclude prefill and teardown but include sampling,
assistant work and transfers. Their first output uses prefill logits. Compare
them with the matching llama.cpp generation harness, not the historical
fixed-forward scorecard. New generation ratios must be reported separately,
with contamination and repetition/order limitations, until acceptance is
established. The sanitizer's overall call seconds are not decode scores.

Monitored short attempt A was terminated by the observer when the PDH collector
failed on sample 36: `PdhGetFormattedCounterArrayW(data)` returned
`0xc0000bba` (invalid data). The collector recorded its error and failed end
marker, and the observer stopped the model process. The case inventory is
therefore incomplete. Inputs remained unchanged; this is rejected evidence,
not a numerical parity failure or a performance score. A fresh uniquely named
attempt may retry the transient counter collection, retaining this failure.

Attempt B completed all twelve parity cases with six graph-reusing plans,
three positive integer-dispatch counts and 234 verified hashes. Its collector
stopped successfully, but every decode interval had unattributed GPU activity;
two also had invalid target counters. The matching short llama.cpp generation
run completed three rounds with identical output and a healthy collector. All
three reference intervals had unattributed activity; two had invalid counters.
Its nine evidence, five prerequisite and six validator hashes verified.

The following are retained **contaminated diagnostic observations**, not
accepted performance comparisons. First rounds are excluded; no speed ratio
or score promotion is justified. Cross-engine execution order was not reversed.

| Short generation decode loop, 32 outputs | Warm round 1 (ms) | Warm round 2 (ms) |
| --- | ---: | ---: |
| Fyodor ordinary | 791.205 | 802.270 |
| Fyodor serial2 | 914.322 | 888.345 |
| Fyodor batch2 | 794.505 | 808.066 |
| Fyodor joint3 with batched graphs | 611.211 | 621.710 |
| llama.cpp matching generation | 702.589 | 719.456 |

Evidence: `monitored-short-B` and `reference-short-A`. Accepted fixed-forward
scores remain unchanged at 85.61025294% short and 106.8707255% at 8K.
