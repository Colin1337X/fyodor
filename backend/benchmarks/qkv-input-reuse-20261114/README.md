# Layer-local QKV normalized-input reuse

The current validated code repacks the same normalized input separately for
query, key and value projections. Query normalization between these calls
writes the query buffer; it does not modify norm or interleaved_input.
This C-host experiment reuses that existing layout instead of adding a new
normalization kernel or changing arithmetic.

`NYA_CUDA_QKV_INPUT_REUSE=1` is opt-in. A local readiness variable starts false
for each layer. A successful eligible interleaved projection establishes it.
The helper excludes vendor paths and requires batch two/four plus a matching
interleaved kernel; it always clears the shared reuse flag before returning,
including failure. The readiness variable is never saved across layers,
requests, resets or rewinds. Normalized input and row-major fallback remain
unchanged. The helper's explicit scope extends the earlier struct comment
about adjacent gate/up reuse to this local Q/K/V sequence.

The trace counts host direct/capture reuse hits, not replayed runtime kernel
savings. Existing graph logical kernel-count snapshots remain responsible
for replay accounting. No new device code or public ABI is introduced.

Separate C17/C23 lifecycle and generation binaries built with warnings treated
as errors, from verified batched-graph inputs. Production dispatch/archive
remain unchanged. GPU lifecycle, reset/rewind, failure, generation, sanitizer
and paired timing validation remain pending. The runner refuses GPU work
until the currently running long-context profile has completed successfully.

`run_control.py` extracts the actual generated C helper and compiles it with
controlled dispatch stubs. C17 and C23 each pass 384 combinations of opt-in,
batch, dtype, kernel availability, vendor selection, ready state and injected
projection failure. It checks observed reuse, unchanged argument forwarding,
readiness only after eligible success and clearing the flag on every return.
Two fresh-layer Q/K/V sequences independently require one initial conversion
then two reuse hits. These are control-contract tests, not GPU correctness or
performance evidence.

The generation runner is prepared behind successful C17/C23 short and C17
long lifecycle evidence. It requires the existing twelve-case output parity,
positive QKV reuse markers, metadata/integer selection and graph replay checks.
No candidate GPU validation has run while the long profile owns the GPU.

The lifecycle runner now verifies the recorded C control-test hashes before
execution, records the actual child PID, and independently requires exactly
two reset summaries with the requested prefix, four verified tokens, resident
execution and scaled error no greater than 0.0005. The pending-long-profile
gate was rechecked after this change and still stops before launching a GPU
workload. Candidate GPU behavior remains unverified.

The candidate full-model memcheck runner is prepared behind its own build,
control, C17/C23 lifecycle and generation proofs. It verifies all prerequisite
hashes, requires QKV reuse enabled and positive reuse trace markers alongside
all twelve output comparisons, six graph-reuse plans and a unique zero-error
sanitizer summary. Ten CPU evidence-validator tests cover missing reuse,
failed sanitizer/dispatch evidence and wrong workloads; these tests do not
execute the candidate on the GPU. The missing-lifecycle gate was exercised
and correctly prevented a sanitizer launch.

The generation runner records the actual child PID and command before waiting,
so long runs can be tracked directly without starting a duplicate workload.
An interrupted runner terminates and waits for its child before finalizing
evidence; the ordinary output-parity and prerequisite checks are unchanged.
