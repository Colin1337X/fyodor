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
replayed operations. The result hashes were independently rechecked. The 8K
lifecycle check is running; no result for it is claimed yet.

`run_generation.py` is prepared to require both short lifecycle builds and the
8K lifecycle before testing twelve ordinary/serial2/batch2/joint3 cases. It also
requires six target plans with more batched replays than captures (three batch2
and three joint3 runs), so matching output through fallback alone cannot pass.

`run_model_memcheck.py` is prepared for the new binaries and requires successful
generation and lifecycle proofs. Its validator retains exact case order,
metadata selection, integer execution, output parity and one zero-error summary,
and also requires capture plus reuse in six verification cases. Nine CPU tests
pass, including missing/duplicated graph evidence and capture-without-reuse.
No full-model graph sanitizer result is claimed yet.

`boundaries.c` and its build/run scripts prepare additional F32 fixture coverage
for head widths 256/512 and linear/ring KV. Within the same plan, tests advance
through positions 250/254/256/258/8190/8192/8196 with batches two/four, changing
tokens after rewinds and repeating after reset. The independent session disables
graphs and produces serial-reference logits. Assertions require exact cache
capture/replay deltas, six captured topologies, 42 verifications, output guards,
one download each and the unchanged 0.0005 scaled-error bound. C17/C23 fixture
binaries compile successfully; GPU fixture execution is still pending. These
small F32 cases complement real-model quantized validation, not replace it.

Real-model lifecycle results, generation parity, full-model memory checks,
two-token and topology-boundary coverage and controlled timing must establish
runtime correctness before production admission. No speedup is claimed from
successful compilation or from the isolated request/gather test alone.

Accepted decode remains 85.61025294% short and 106.8707255% at 8K relative to
llama.cpp. The 125% target on both contexts is still unmet.
