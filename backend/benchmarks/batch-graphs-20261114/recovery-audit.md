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

None of these graph-specific failure injections has run yet. Successful
ordinary boundary fixtures and full-model generation must not be cited as
evidence for this recovery matrix.

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
