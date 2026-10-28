# Larger Q6 integer-product runtime trial

Base1b0babc. Original Fyodor implementation, optional root optcpp source.
NYA_ENABLE_INTEGER_Q6 defaults OFF. NYA_CUDA_INTEGER_Q6=1 selects optional Q6
dot and residual-add symbols only if both load, excluding training/reference
contexts. Device wrappers select integer-product arithmetic only when both
columns and rows are >=2048; smaller shapes use native arithmetic. QKV remains
unchanged. Existing Q4 integer dot and gate switches operate independently.

The Q6 coefficient check in decode-q6-integer-20261028 exhaustively validates
finite-half/signed-scale/quantized-value multiplication including signed zero.
Real-model and build checks pass with no relaxed tolerances.
Optional sources can be physically absent without requiring a host C++ compiler.

Accepted as an opt-in optimization after validation. validate.py builds C17/C23 and six selected tests each;
oracle.py checks real TinyLlama against CPU at capacities129/273;
verify_removal.py builds a source copy without optional compartments and runs
an inference smoke, not the full removal matrix. reference.py compares accepted
Q4 integer-dot/gate plus selective-FDPP baseline to that baseline plus Q6, then
llama.cpp. Candidate uses legacy label fixed in evidence JSON.



## Accepted evidence

C17/C23 each pass six selected tests. Real TinyLlama CPU-oracle checks at
capacities129/273 pass unchanged tolerances. The fresh physical-removal source
copy builds and runs a real-model inference smoke with no host C++ compiler.
This is not the full removal matrix, malformed-model validation, or AMD/Vulkan
execution. The feature is disabled by default and training/reference contexts
retain their existing paths.

Matched workload: TinyLlama Q4_K_M, F32 KV, full GPU,128 decode tokens at prefixes
0/1024, CPU threads6, batch512, deterministic tokens, full host logits every
token; llama FA off. RTX5060Ti SM120, driver617.42, llama b10809 commit
5266f24da75dc449bd56cbed7addb9c8e4a6a73e. Exact model, executable, dependency and
monitor hashes are retained in reference/metadata.json. Baseline includes the
accepted Q4 integer gate/dot and selective FDPP; candidate adds Q6 dot/add.

Forty observations per cell:20 repetitions and10 warmups in each of opposite
engine orders. All12 activities passed pre/post idle and continuous competitor
gates. No competitor was killed or rejected in this run. No other build/test
work overlapped timing. The process completed its protocol without interruption.
Pooled deviations use all per-process counts/means/deviations; the harness does
not emit individual repetition values. In particular, llama short-context
variance is higher here and is retained rather than replaced by an older score.

| Prefix | Baseline tok/s | Q6 candidate tok/s | llama.cpp tok/s | Candidate / llama | Gain vs baseline |
|---|---:|---:|---:|---:|---:|
| 0 | 387.403 +/- 2.016 | 390.452 +/- 1.960 | 380.598 +/- 7.129 | 102.59% | +0.79% |
| 1024 | 324.181 +/- 1.066 | 326.137 +/- 1.014 | 268.616 +/- 1.169 | 121.41% | +0.60% |

Retain the optional larger-Q6 path: its modest gain is positive in both engine
orders for both workloads. Current relative throughput is102.59% short/121.41%
long; >=125% on both remains unmet. This does not change training/prefill scores
or claim a general win across hardware/models. Reprofile the new baseline next;
Q4 gate/up remains the largest identified matrix cost.

Reproduction: run validate.py, oracle.py, verify_removal.py, then reference.py
<fresh output directory> and summarize_reference.py (expects reference/).
Keep GPU tests/builds serial and preserve prior output directories. Optional
reference continuation support was not used for this result.
