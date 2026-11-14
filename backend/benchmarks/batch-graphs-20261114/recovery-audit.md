# Batched graph recovery coverage

Source audit of the isolated generated `resident.inc` and
`backend/tests/test_verify.c`, 2026-11-14. This is not execution evidence.

The existing `NYA_CUDA_FAIL_AFTER` injection returns before context push,
request upload, capture, or graph replay. The verification test exercises
session invalidation and reset to CPU reference after that early failure.
It cannot establish recovery from graph driver failures.

The batched graph implementation routes request launch, capture begin/end,
instantiation, graph launch and logit readback failures to the common failure
label. That label ends an active capture, frees a returned graph, synchronizes
the stream, restores the context and poisons the plan. Plan destruction frees
all six cached executable handles. Source inspection alone does not establish
that every partial driver failure leaves these handles safe to destroy.

Before production admission, use an isolated driver-function wrapper to inject
one failure at each of these sites:

| Site | Required observable behavior |
| --- | --- |
| Request update | No output or position commit; plan invalidated |
| Capture begin | No attempted end of a capture that never started |
| Kernel lowering during capture | Capture ended; returned graph freed |
| Capture end | Returned graph, if any, freed exactly once |
| Instantiation | Source graph freed; no invalid executable replay |
| Initial launch | Created executable retained only until safe destruction |
| Cached launch after successful replay | No stale logits or KV suffix exposed |
| Readback/synchronization | No partial output accepted as successful |

For each case, require the injection marker, the failure result, unchanged
output guards, correct session invalidation, reset/recovery logits against an
independent serial session at the existing 0.0005 scaled-error bound, and a
successful subsequent model/session lifecycle. Run linear and ring KV cases.
Track graph/graph-executable create/free counts in the wrapper, and run the
GPU cases under Compute Sanitizer. Injection must never enter the production
dispatch or alter the inputs of the currently running model sanitizer.

At the initial audit, none of these graph-specific failure injections had run.
The execution evidence below now covers five injection configurations. Ordinary
boundary fixtures and generation do not establish the remaining failure paths.

`build_faults.py` now builds separate C17/C23 lifecycle and generation binaries
under `build-cuda/batch-graph-faults-20261114`. `fault_hooks.h` wraps capture
begin/end, instantiation and launch. The end hook completes the real capture
before returning an injected error, exercising cleanup of a returned graph.
The launch hook can be armed after warming a cached executable. All four
binaries compiled; input hashes were unchanged. No GPU recovery execution is
claimed. Request/update, readback and resource-count wrappers remain to be
implemented alongside the recovery fixture.

`recovery.c` and `build_recovery.py` now compile a C17/C23 fixture that first
warms a two-token graph, then injects a failure into a cold four-token graph or
cached two-token launch. It checks failed verification, output guards, hidden
logits and rejected rewind/verification, reset to CPU reference, parity after
reset, and a fresh resident session with parity. Widths 256/512 and linear/ring
KV are selectable. `run_recovery.py` runs one case under Compute Sanitizer,
requires exactly one injection marker and one zero-error summary, and checks
input hashes before/after. Its pending-model-sanitizer gate was exercised;
no recovery GPU case has run yet. Resource create/free accounting and the
remaining injection sites above are still outstanding.

## Completed initial recovery matrix

All 40 Compute Sanitizer memcheck runs passed: C17/C23, head widths 256/512,
linear/ring KV, and five injection configurations (capture begin, capture end,
instantiation, initial launch, cached launch). Each run recorded exactly one
expected injection and one zero-error sanitizer summary. The fixture verified
failed-session invalidation, output guards, CPU reset parity at the unchanged
0.0005 scaled-error bound, and fresh resident-session parity.

The complete case inventory, raw markers, terminal results and 306 unique
input/evidence hashes were independently verified after all runs completed.
Evidence is in the 40 `recovery-<standard>-<width>-ring<ring>-<site>-<cache>`
directories. These tiny fixtures establish these injected failure paths only;
they do not prove recovery from every real driver failure or accept any timing.

Request update, kernel lowering during capture, readback/synchronization and
explicit graph resource create/free accounting remain outstanding. Production
admission and the 125% decode performance target remain unproven.

## Extended injection and ownership accounting

Separate v2 C17/C23 builds add request-update failure, failure after the first
captured gather, and synchronization/readback error returns. The latter two
perform the real operation before returning an injected failure, exercising
rejection after work or output may already have completed. They do not emulate
a hung device or every partial-copy behavior. Cached request, launch, sync and
readback paths are selectable. Original evidence/builds remain unchanged.

Test-only wrappers track individual graph and executable handles through
creation and destruction, detecting unknown/duplicate handles, failed frees,
registry exhaustion and resources still live at process exit. The runner
requires positive balanced create/free counts, zero live handles and zero
accounting errors, in addition to the existing numerical/recovery checks.

All twelve configurations passed memcheck for C17, width 256, linear KV. The
remaining C17/C23, width 256/512, linear/ring configurations are not yet claimed
complete. See `recovery-v2-*` terminal results for the exact executed scope.
The initial compilation caught a signedness warning in the test counter; an
explicit unsigned conversion fixed it, and both standards then built with
warnings treated as errors. No production dispatch or archive changed.

The first context-boundary sanitizer (C17, width 256, linear KV) also passed all
42 ordered cases, six captures and 42 replays; 180 hashes independently
verified. The other seven boundary sanitizer configurations remain pending.
None of this correctness evidence updates accepted throughput scores.

## Completed extended matrix

All 96 v2 cases now passed: both C standards, both head widths, both KV layouts,
eight cold failure sites and four cached failure sites. The independent
`audit_recovery_v2.py` checked the complete inventory, terminal results, exact
injection/recovery markers, zero-error memcheck summaries, operation-specific
create/free counts and 571 unique hashes. Every process finished with zero live
tracked handles and zero ownership errors. `recovery-v2-audit.json` records
this completed scope. Four CPU tests exercise valid histories and rejection
of bad terminal states, missing/duplicate evidence and bad resource accounting.

The seven remaining boundary sanitizer configurations are running separately;
only the previously completed C17/256/linear boundary run is claimed here.
These results do not replace full-model validation, prove recovery from all
real driver errors, or establish a clean throughput improvement.

The C17/256/ring boundary sanitizer has now also completed all 42 ordered
cases with six captures and 42 replays. Its 180 hashes independently verified.
C17/256 linear and ring are complete; the remaining six configurations are
still pending or running and are not claimed here.

All four C17 boundary sanitizer configurations have now passed (168 ordered
parity cases). Their 201 unique input/evidence hashes independently verified.
C23 remains in progress. `audit_boundary_matrix.py` rechecks all eight
configuration identities, complete ordered parity/capture markers and recorded
hashes before emitting a complete matrix report. Its incomplete-matrix check
was exercised and rejected the pending C23 result without writing a report.
The four existing boundary-validator tests still pass.

Both C23 width-256 boundary runs now passed, bringing completed boundary
coverage to 252 cases across six configurations. Each new run's 180 hashes
independently verified. The two C23 width-512 configurations remain pending
or running. Five aggregate-audit tests pass, covering a complete 336-case
fixture and rejection of a missing configuration, wrong standard executable,
modified raw evidence and omitted evidence hash.

All eight C17/C23 boundary configurations have now completed successfully:
336 ordered parity cases, each configuration with six graph captures and
42 replays. `boundary-matrix-audit.json` records independent validation of
configuration identities, raw markers, terminal success and 238 unique hashes.
The full boundary sanitizer matrix is complete. Clean performance acceptance
and production integration remain outstanding.
