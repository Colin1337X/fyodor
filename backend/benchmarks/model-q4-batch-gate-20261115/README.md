# Isolated Q4 batch-four gate/up model integration

The C17 and C23 binaries build from the validated model-q6-pair-20261114
provider, retaining the Q4/Q6 paired-row paths. The embedded optional kernel
comes from optcpp/matvec/q4_batch_gate.cu. Production archives are unchanged.

NYA_CUDA_Q4_BATCH_GATE=1 enables function lookup. Dispatch requires Gemma,
batch four, 15360 rows by 3840 columns, both native Q4_K weight tensors with
exact matching byte sizes, enabled integer/interleaved scratch, enabled gate
fusion, non-reference execution and no active vendor matmul for the shape.
All other cases retain existing dispatch. Missing kernel lookup falls back;
CUDA launch failures propagate through normal plan failure handling.

The fused operation explicitly interleaves the current normalized input, then
writes the activated gate product directly. It does not reuse potentially stale
QKV scratch. A trace counter records successful capture/direct dispatches;
it is not a graph replay count. The existing matrix profile records the fused
operation as one gate/up operation.

Build prerequisites include raw numerical validation, three sanitizer results,
base model memory/parity evidence and the completed paired-projection timing.
Recorded source and artifact identities were rechecked after compilation.

Status: compilation only. Dispatch boundary tests, model lifecycle/parity,
full-model sanitizers and monitored short/long comparison remain pending.
No production promotion or accepted performance score change is justified.


## Dispatch and short lifecycle validation

The actual extracted eligibility predicate passes 9,584 cases in each of C17
and C23. Coverage includes all combinations of optional kernel/reference/Gemma/
fusion/scratch/vendor/null-tensor flags, surrounding dimensions, batch sizes,
Q4 versus other dtypes, and incorrect byte sizes for either weight tensor.

Both C17 and C23 real-Gemma lifecycle checks pass at prefix 257. Each performs
two resets, four-token verification, graph capture/replay, one-bank download,
invalid-argument guards, KV rewind and suffix replacement against serial logits.
Both report maximum scaled error 0.0000219546455 versus the unchanged 0.0005
limit, 48 fused capture/direct dispatches, and one graph capture/two replays.
Evidence hashes were independently rechecked after completion.

The generation marker validator rejects missing/duplicate/malformed/zero
markers, and thirteen sanitizer-evidence tests pass. These tests validate the
admission logic; full-model sanitizer execution is still pending. Long-prefix
lifecycle and generation correctness remain required before model timing.


The fusion on/off monitored runner is prepared, with eight admission tests
passing and an actual missing-sanitizer prerequisite refusal retained in
monitored-preflight-refusal.json. It does not launch a GPU workload before the
required completed lifecycle, generation and model memcheck evidence exists.
Both variants keep Q4/Q6 paired rows and QKV reuse enabled; only fusion toggles.
Existing PDH coverage rules are unchanged. No monitored fusion run is complete.


The extracted host launch helper also passes C17/C23 checks using stubbed CUDA
calls. These verify a fresh norm interleave before fusion, both tensor addresses,
output/scratch pointers, row/column/stride/batch arguments, launch geometry,
profile cleanup, and failure propagation for both conversion and fused launch.
Only a successful fused launch increments its execution counter. This is host
wiring evidence, not additional GPU numerical or sanitizer coverage.


## Completed long-prefix lifecycle

C17 lifecycle at prefix 8192 passes both resets with maximum scaled logit error
0.0000163989979, below the unchanged 0.0005 bound. All 48 Gemma layers dispatch
the fused kernel; one verification graph is captured and replayed twice.
Reset/rewind, suffix replacement, guarded outputs and single-bank readback
assertions pass. All 953 recorded file identities verify after completion.
Short generation validation follows; full-model sanitizer and timing remain
pending. This lifecycle result is correctness evidence, not throughput.
