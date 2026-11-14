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


## Completed short generation, C17

All twelve ordered short-generation cases pass, with 25-token prompts and
32 generated tokens matching ordinary generation in every mode and round.
Fusion capture/direct markers are [48,48,48], and all six verification plans
capture and reuse their graphs. All 965 recorded input/output identities verify.
This establishes output parity for the short C17 workload, not accepted timing.
C23 generation, full-model memory checking and long generation remain pending.


C23 short generation also passes all twelve cases with identical output parity
requirements and fusion markers [48,48,48]. All 965 recorded identities verify.
The full-model short memcheck can now proceed; it is not yet a passing result.


The effective-settings audit evaluates the actual outer runner environment block
and the inner timing runner's setting construction without launching either.
Both enabled/disabled variants match the completed C17/C23 generation settings,
except the intentional fusion toggle; monitoring controls are also required.
Injected inherited NYA flags are filtered. settings-audit.json retains both full
effective environments and source identities. This audits configuration only;
it does not imply completed monitoring or sanitizer execution.


## Completed full-model short memcheck

Compute Sanitizer memcheck exits zero after all twelve generation cases,
including three joint rounds with fusion execution evidence. The unique error
summary reports zero errors. All output parity, graph reuse, execution inventory
and unchanged-input requirements pass. Independent validation rechecked all 972
recorded file identities. Elapsed instrumented time was 2599.1669611 seconds;
none of these instrumented timings is a performance result.

Long-context generation validation now follows. The kernel remains experimental,
and accepted llama.cpp scores stay 85.6103% short / 106.8707% at 8K.


## Completed long-context generation

All twelve C17 long-generation cases pass with 8205-token prompts and 32 output
tokens matching ordinary generation. Fusion markers are [48,48,48], with graph
capture/reuse and the existing Q4/Q6/QKV execution inventories validated.
All 965 recorded file identities verify independently. This completes the
planned long-context generation correctness gate, without performance acceptance.
Monitored on/off comparisons now follow, retaining the existing coverage rules.


## First monitored short comparison (rejected coverage)

DisabledA and EnabledA both complete all twelve output-parity cases and required
execution inventories. All 1003 file identities per result verify independently.
The three joint decode-loop samples average 567.0631667 ms (sample SD 2.1766702)
without fusion and 549.8707000 ms (SD 0.8012292) with fusion, per 32 output tokens.
This is approximately 3.03% lower diagnostic latency. Every joint interval in
both runs fails coverage for unattributed GPU activity. Thus neither result is
accepted performance, and no llama.cpp score changes. Reverse-order comparison
is the next check. Raw observations and failed coverage audits are preserved.
