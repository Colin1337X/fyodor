# Optional Q4 integer-product decode runtime trial

Source base 4386199. Original Fyodor implementation. Previous kernel diagnostics
are in decode-integer-dot-20261027; the finite-half coefficient proof and exhaustive
check are in decode-integer-scale-20261027. No Q6 arithmetic is changed.

NYA_ENABLE_INTEGER_DOT defaults OFF. When enabled, root optcpp is present, and
NYA_CUDA_INTEGER_DOT=1 is set before context creation, all four optional symbols
must load before dispatch switches. Native kernels remain on any lookup failure.
The group covers Q4 row dot, Q4 residual-add, and both QKV combinations (Q4/Q4/Q4
and Q4/Q4/Q6); Q6 V uses unchanged arithmetic. Explicit reference and training
contexts do not select it. Ordinary C owns dispatch; no host C++ is required.
Removing the source or root optcpp gracefully skips the feature at configure.

Accepted as an opt-in optimization after validation. validate.py runs C17/C23 builds and selected lifecycle
tests. oracle.py compares TinyLlama real-model outputs against CPU at capacities
129 and 273 without tolerance changes. verify_removal.py tests a fresh copy with
optional C++ compartments physically absent and an inference smoke; it does not
prove the entire removal matrix. reference.py compares the accepted integer-gate
baseline (baseline) to integer gate plus Q4 integer dot (legacy label fixed), and
llama.cpp. Both Fyodor cases use selective FDPP. It retains rejected observations
and requires idle/competitor checks around every measured process.



## Validation and matched performance (completed 2026-10-28 KST)

Both C17 and C23 builds pass six selected lifecycle tests. Real TinyLlama CPU
oracles at capacities129/273 pass unchanged tolerances, exercising the fused
Q4/Q6 projection combinations. A fresh source copy with root optcpp, backend
optcpp, CUDA/cutlass and CUDA/cute absent builds and runs a real-model smoke
without a host C++ compiler. This is not the entire physical-removal matrix,
AMD/Vulkan execution, or validation of malformed nonfinite model coefficients.

RTX5060Ti SM120, driver617.42, TinyLlama Q4_K_M, F32 KV, 128 decode tokens,
prefixes0/1024, CPU threads6, batch512, full GPU, llama FA off, identical
deterministic input tokens, full host logits every token. llama b10809 commit
5266f24da75dc449bd56cbed7addb9c8e4a6a73e. Both Fyodor variants use selective FDPP
and integer gate; only candidate enables integer dot. Exact model, dependencies,
binary and monitor hashes are in reference/metadata.json.

Forty observations/cell (20 repetitions and10 warmups per process, two opposite
engine orders). Standard deviations pool all aggregate counts; per-process
mean/stddev retained, individual repetition samples not emitted by harness.
All12 environment activities pass idle and continuous competitor gates. This
run completed uninterrupted, with no competing processes killed or rejected
runs in this directory. No other builds/tests overlapped measurements.

| Prefix | Accepted baseline tok/s | Candidate tok/s | llama.cpp tok/s | Candidate / llama | Gain vs baseline |
|---|---:|---:|---:|---:|---:|
| 0 | 376.147 +/- 3.130 | 388.168 +/- 1.813 | 380.999 +/- 2.382 | 101.88% | +3.20% |
| 1024 | 315.798 +/- 1.276 | 323.800 +/- 1.174 | 268.351 +/- 1.007 | 120.66% | +2.53% |

Retain optional Q4 integer dot/add/QKV dispatch. Both tested workloads now exceed
llama.cpp throughput; the requested >=125% in both remains unmet. No prefill or
training score changes are claimed. Native fallback, explicit reference mode,
Q6 arithmetic, accumulation/reduction order and weight storage are unchanged.

Reproduce with validate.py, oracle.py, verify_removal.py, then reference.py
<fresh-output-directory> and summarize_reference.py (expects reference/).
Run serially. reference.py can finish missing observations in a new directory
using a prior directory as a third argument, checking hashes before skipping
completed observations; that continuation facility was not needed in this run.
