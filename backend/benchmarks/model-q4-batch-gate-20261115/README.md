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
