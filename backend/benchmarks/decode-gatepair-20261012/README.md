# Q4_K gate/up fusion for decode

Original Fyodor implementation. Single-token LLaMA-style Q4_K gate and up
projections share activation loads and one CUDA dispatch, then calculate SiLU
in the same kernel. Each projection retains the previous per-lane accumulation
order and warp reduction. The two weight tensors must have matching storage
sizes. Gemma, other formats, reference arithmetic and batched execution keep
the existing path. No new device buffers or transfers are required.

`NYA_CUDA_GATE_UP=0` selects the unfused control when a plan is created.
Profiling labels the fused operation `gate-up-q4`, one call per layer covering
two projections and the activation. The bounded profile table and overflow
behavior are retained. CUDA graph topology contains 44 fewer nodes per token
for TinyLlama's 22 layers (three operations replaced by one per layer).

## Validation protocol

`check_kernel.py` compiles production sources with the runtime's NVRTC options
and compares fused output against the original two dot kernels plus vector
activation. Twenty-one shapes cover 1/7/8/9/63/64/65 rows and 256/2048/8192
columns, independent randomized Q4_K weights, F32 inputs and output canaries.
All outputs match exactly. The final harness uses only Python standard-library
modules, NVRTC and the CUDA driver API, with explicit allocation and cleanup.
An earlier PyTorch-backed version passed the numerical cases but stalled
during sanitizer shutdown; it was stopped and is not counted as a safety pass.

The independent CPU oracle remains unchanged. Forced TinyLlama decode at
273/272/271 tokens passed the established 0.001 scaled-error limit; maximum
observed error was 6.4816883e-6. `NYA_TEST_GATE_UP=1` is a separate test mode
that compares against an unfused CUDA plan with identical batching and requires
exact logits and KV snapshots. It does not replace the CPU oracle or loosen any
existing tolerance. Its reset/interleaved-prefix test crosses the partitioned
attention boundary and verifies graph reuse.

The stricter real-model CPU KV snapshot check failed at layer 17, position 9,
with fusion both enabled and disabled. This pre-existing discrepancy remains
unresolved; both logs are retained. The initial equivalence harness compared
batched GPU preparation with token-at-a-time reference preparation; its failed
run is retained as `exact-unmatched-batching.log`. The final equivalence test
matches batching, which isolates this optimization.

Initial attention fixture tests lacked the benchmark's cuBLAS DLL setting and
fell back to CPU. The configured C17 suite passed 25/25 and C23 passed 24/24.
These include graph reuse, attention shapes, fallback and quantized kernels;
C17 also includes a resident training graph smoke test because CUDA source is
shared at compilation time. The first logs are retained, not counted as passes.

## Timing protocol and exclusions

`clean/` compares the saved committed benchmark executable at d4712b2 against
the initial fused implementation: prefixes 0/512/1024, 128 generated tokens,
two warmups and five timed repetitions, two reversed-order pairs per context.
All twelve activities passed the existing separate-process GPU monitor and
three-consecutive-sample desktop gate (<=30% utilization and <=15 W).
Graphs remain enabled. Loading and prefix preparation are outside the timing.

The first unmonitored diagnostic attempt encountered Roblox GPU activity. That
process was stopped under the user's prior authorization. The interrupted run
is retained in `excluded-roblox/` and contributes no accepted timing samples.

`reference.py` reruns only decode against the pinned llama.cpp reference using
the final binary. Model, KV precision, reference attention mode, CPU threads,
repetition counts and returned-logit behavior retain the scorecard contract.
Prefix preparation remains necessary for long decode, but prefill is neither
timed nor optimized. The user has prioritized decode exclusively until it
exceeds 100% of matched llama.cpp throughput.


The final direct-driver harness passes memcheck, initcheck, racecheck and
synccheck on the fused kernel, with zero errors/warnings/hazards. Initial
filter-syntax and virtual-environment-launcher attempts are archived separately.
C17 and C23 exact graph tests each checked 795 decodes across reset and mixed
prefix execution, retained two graphs, and saved 70,884 launches across the
whole test while preserving every checked logit and KV value exactly.

The final reference refresh was run October 14. It uses `-p 0` for both
engines: the earlier scorecard's separate pp512 measurement is omitted, and
short-workload allocation capacity now follows the decode-only invocation.
This and the changed desktop state mean the scorecard change must not be
attributed solely to fusion. The paired before/after results use identical
invocations and provide the isolated comparison for this change.

## Accepted results

All samples are retained; values below pool ten timed samples per cell and
show sample standard deviation. `summarize.py` validates idle gates, process
identity checks, exits, graph replay counts and transfer counts before pooling.

| Prefix | Before fusion (tok/s) | After fusion (tok/s) | Mean gain |
| --- | ---: | ---: | ---: |
| 0 | 271.10 ± 0.95 | 278.49 ± 0.66 | 2.73% |
| 512 | 257.38 ± 0.86 | 263.97 ± 0.42 | 2.56% |
| 1,024 | 241.71 ± 1.13 | 246.36 ± 5.72 | 1.93% |

The long-context paired result is noisier; this is an observed mean gain,
not a claim of precise improvement at every context.

| Final October 14 decode | Fyodor (tok/s) | llama.cpp (tok/s) | Relative throughput |
| --- | ---: | ---: | ---: |
| Prefix 0, 128 tokens | 280.15 ± 0.64 | 382.69 ± 5.75 | **73.2%** |
| Prefix 1,024, 128 tokens | 249.28 ± 0.72 | 270.04 ± 0.64 | **92.3%** |

All eight reference processes passed monitoring and idle gates. No other GPU
process was observed at preflight. These are monitored Windows desktop results,
not headless measurements. Both engines use the same TinyLlama Q4_K_M model,
F32 KV, full GPU execution, six CPU threads and host logits on every token.
Reference llama.cpp is pinned to b10809 / 5266f24da75dc449bd56cbed7addb9c8e4a6a73e,
with flash attention off. Hardware is RTX 5060 Ti 16 GB, driver 617.42,
Ryzen 5 9600X. These results do not establish parity with other models or the
latest/fastest llama.cpp configuration. Decode remains below the >100% target.

Machine-readable pooled results: [summary.json](summary.json).
Final source, binary, model and dependency hashes plus environment settings:
[reference provenance](reference-20261014/metadata.json).
Safety exit codes: [direct sanitizer results](sanitizer-direct-exits.json).
Exact real-model graph checks: [C17/C23 exits](exact-graph-final-exits.json).

## Follow-up profile

`final-profile/` records diagnostic event profiles, which disable graphs and
are not accepted throughput measurements. Both contexts dispatch `gate-up-q4`
8,448 times over 384 decode tokens, as expected for 22 layers. Short decode
spends 540.15 of 1,345.41 matrix milliseconds in the fused pair (40.1%). This
remains a significant target for further decode work. Long-run logs also
include untimed prefix preparation; those entries are not decode costs.
