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
remain unchanged. Failure, long generation, sanitizer and paired timing
validation remain pending; completed short lifecycle evidence is described below. The runner refuses GPU work
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
Candidate GPU validation began only after the long profile completed successfully.

The lifecycle runner now verifies the recorded C control-test hashes before
execution, records the actual child PID, and independently requires exactly
two reset summaries with the requested prefix, four verified tokens, resident
execution and scaled error no greater than 0.0005. The pending-long-profile
gate was rechecked after this change and still stops before launching a GPU
workload. Subsequent short lifecycle evidence is recorded below.

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

C17 and C23 short-prefix lifecycle checks now pass both reset cycles with
four verified tokens, graph capture/replay, rewind/replacement guards and
maximum scaled logit error 0.0000219546455 versus serial execution, within
the unchanged 0.0005 bound. Each trace records 88 QKV capture/direct reuse
hits and one graph capture with two replays. All 670 hashes per result were
independently verified. These counters prove selection, not elapsed-time
savings; generation, memory safety and paired throughput remain unproven.
The 8192-prefix C17 lifecycle also passed both resets with maximum scaled
logit error 0.0000163989979, 88 capture/direct reuse hits, one capture/two
replays and all 670 hashes independently verified. Reset/rewind/replacement
checks and the one-download assertion remain enabled. This establishes the
candidate lifecycle at short and 8K prefixes, not full generation or speed.

The monitored timing runner is prepared behind candidate control/lifecycle,
context-specific generation, C23 generation and full-model short memcheck
proofs, with the unchanged base device-boundary proofs also verified. It
requires positive QKV reuse and graph replay markers, records the explicit
environment overrides and uses the existing continuous GPU activity collector.
The missing-memcheck gate was exercised without launching another GPU job.
Coverage alone cannot accept a benchmark; activity failures remain disqualifying.

Both C17 and C23 short generation runs now pass all twelve ordered cases
(ordinary, serial2, batch2, joint3), with 25 prompt tokens and 32 generated
tokens. Each run retains exact generated-text parity, six graph-reusing
verification plans, three positive integer-call markers and three QKV reuse
markers of 88. All 682 hashes per result independently verified.

C17 joint decode-loop diagnostics were 610.677, 620.885 and 599.6355 ms in
round order. There was no continuous activity acceptance or paired baseline
in these correctness runs, so they establish no speedup or updated score.
The candidate full-model short Compute Sanitizer run started only after both
standards passed generation. Long generation, sanitizer completion and
controlled timing remain pending; production dispatch is unchanged.

The monitored runner also accepts a final `enabled` (default) or `disabled`
argument. Both variants use the exact same candidate executable and retain
all output/graph checks; only the QKV reuse switch changes. Enabled requires
exactly the three positive reuse markers observed in validated generation;
disabled requires no reuse markers. Five CPU tests cover valid inventories,
missing/extra or malformed markers, zero hits and unexpected reuse in the disabled path.
Use distinct labels in enabled/disabled/disabled/enabled order for a paired
short diagnostic before attributing any timing change to this optimization.
This comparison is still pending and does not substitute for llama.cpp runs.

During the full-model short sanitizer, Roblox PID 31664 appeared as a new
competing GPU process. Its executable identity was verified against the known
Bloxstrap Roblox path, it was stopped under the user's existing authorization,
and disappearance plus before/after GPU inventories were retained in
`gpu-cleanup-20261114-1909.json`. The same sanitizer process remained live.
Its elapsed times remain excluded from performance evidence; no restart was
needed for this correctness-only workload.

`run_reference.py` is prepared for fresh pinned llama.cpp generation runs in
this candidate's evidence folder. It requires candidate build, full-model
short sanitizer and context-specific generation evidence, verifies the existing
identical prompt-token proof, and compares output bytes against the candidate's
validated generation output. The token IDs read by the interval parser come
from that same verified proof directory. It retains continuous GPU monitoring
and checks source, prerequisite and reference-library hashes before and after
the run. The active-GPU refusal was exercised; no new llama.cpp run has started.
