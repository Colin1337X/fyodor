# Decode-scoped CUDA event profiling

`NYA_CUDA_PROFILE=1 NYA_CUDA_PROFILE_FROM=N` records only batches starting at
absolute position N or later. It does not split a batch across the threshold or
change any inference arithmetic. Earlier batches contribute neither events nor
matrix-shape slots. Resets reapply the position filter. The final scope line
reports completed profiled batches and tokens; the default N=0 preserves the
existing all-work profile. Invalid/overflowing values disable profiling, print a
diagnostic and preserve ordinary execution. The option is read at plan creation.

Both kernel launches and vendor GEMM event brackets obey the filter. Profiling
still disables executable graphs for the entire plan. Event brackets can include
host submission gaps and scheduling effects; these are not hardware-counter
kernel measurements or normal graph-enabled throughput.

## Validation

`scope_oracle.c` compares every logit after prefill and each decode across two
resets, with profiling off and thresholds 0, 10, 16, 17 and 33. With an 8-token
prefill batch, the 8..15 batch crossing threshold 10 is excluded in full. The
16-token prefix followed by three decode tokens yields 6/6 recorded batches/tokens
across two resets for thresholds 10 and 16, 4/4 for 17, and 0/0 for 33. Default
profiling records 10 batches/38 tokens. Malformed and overflowing settings disable
profiling; removing the variable restores the default. All outputs match exactly.
C17 and C23 CUDA builds and both scope oracles pass. The JSON records binary hashes.

## Gemma diagnostics

RTX 5060 Ti, the previously validated Gemma 4 12B Q4_K_M model, F32 KV. Same native
settings as the ring timing diagnostic, except profiling is enabled. Four serial
runs: linear/ring at prefixes 0 and 8192, each with 16 decode tokens, one repetition,
no warmup. Scope assertions require exactly 16 batches and 16 tokens. Every
recorded matrix operation has batch 1; prefix work is excluded. Model provenance
and normal throughput evidence are in `../gemma4-ring-timing-20261113/`.

| Prefix / storage | Matmul event ms/token | Attention event ms/token | Matmul share | Attention share |
| --- | ---: | ---: | ---: | ---: |
| 0 / linear | 23.3594 | 0.7695 | 81.18% | 2.67% |
| 0 / ring | 23.0731 | 0.7216 | 81.80% | 2.56% |
| 8192 / linear | 25.2356 | 16.4083 | 54.52% | 35.45% |
| 8192 / ring | 23.9028 | 15.2716 | 54.44% | 34.78% |

Short linear gate/up projection plus activation accounts for 157.225 of 373.751
matrix-attributed ms across 16 tokens (42.07% of the matrix total). The two down
projection types add 104.997 ms, and the vocabulary projection adds 34.828 ms.
This directs short-decode investigation toward matrix work. At 8K, attention is
a substantial additional target; next separate full/sliding attention costs by
head width and KV grouping before choosing a kernel redesign. This profile alone
does not establish the cause of attention's cost or a cache/bandwidth bottleneck.

Single diagnostic observations, desktop activity retained, graphs disabled. No
accepted speed score changes and no 125% target claim. The profile overhead also
means these per-operation sums should not be reconciled as an exact decomposition
of the separate normal-throughput runs.

Reproduce: build `build-cuda/fyodor-bench.exe`, run `run.py`, then `summarize.py`.
Compile `scope_oracle.c` against the matching native library and run
`check_scope.py` (or `check_scope.py build-cuda-c23`). Scripts are diagnostic tooling;
the runtime remains native C with removable CUDA and optional C++ components.
