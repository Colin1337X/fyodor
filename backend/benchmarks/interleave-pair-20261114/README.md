# Adjacent gate/up input conversion reuse (experimental)

The retained joint-base profile identifies matrix work as about 85% of recorded
GPU kernel time. Its largest group is batch-four Q4 feed-forward expansion.
`CUDA/resident.inc` currently converts `p->norm` into interleaved input separately
for the gate and up projections, despite both reading the same input and using
the same scratch layout. This candidate skips the second conversion only when
both adjacent projections select an interleaved kernel and neither uses cuBLAS.
Batch sizes are restricted to two/four; aligned prefix batches are excluded.
The reuse flag is cleared immediately after the up call, including its failure
path, so no scratch validity is inferred across layers, requests or graph builds.

`build.py` modifies only an isolated copy of the host translation unit. It reuses
the validated launch-bound candidate header and compiles strict C17/C23 objects.
It checks the original build manifest before and after compilation. Production
source, archive, dispatch and the currently running sanitizer inputs are intact.
No new C++ dependency is introduced; this remains removable CUDA host C code.

Both host builds pass (`build.json`). Initial C17/C23 real-model lifecycle
validation also passes, as recorded below; throughput remains unvalidated. It is not integrated
or enabled by default. It saves one conversion launch per eligible FFN pair;
it does not remove either projection or change their arithmetic. The whole-model
benefit is unknown and must be measured after the current candidate finishes.

Remaining gates: validate long-context lifecycle and short/long generation
logits, output parity, reset/rewind and failure recovery; run Compute Sanitizer;
then compare end-to-end decode with and without reuse under clean monitoring.
Do not replace accepted scores from a compile check or inferred launch savings.

`build_executables.py` now links four isolated C17/C23 executables: joint-base
generation and the existing four-token lifecycle harness. The original build
inputs and production archive hashes match before and after linking. The outputs
are recorded in `executables.json`, with `gpu_validated=false`. Linking is not
admission: the current launch-bound candidate remains first in the GPU queue.
Before throughput acceptance, confirm the reuse branch is exercised and compare
its launch inventory with the original; parity alone does not prove selection.

## Initial real-model lifecycle validation

`run_lifecycle.py` now checks all linked-input hashes before execution and again
at exit, verifies model identity, and retains the original lifecycle assertions.
C17 and C23 at prefix 257 / ring KV / four-token verification both pass: both
reset cycles report max scaled error 2.19546455e-05 against the unchanged 0.0005
bound, with positive integer execution counts and unchanged inputs. These checks
cover reset, rewind, replacement and stale-suffix behavior from `test_verify.c`.
They do not yet prove the reuse branch's launch savings or whole-model speed.
The 8192-prefix C17 lifecycle check is next. Production dispatch remains unchanged.

## Dispatch inventory diagnostic

`build_inventory.py` builds separate baseline and reuse C17 host objects and
lifecycle executables, instrumenting conversion and reuse dispatch counts.
Both builds compile and link (`inventory-build.json`), leaving validated inputs
unchanged. `run_inventory.py` is prepared to require equal projection counts,
zero baseline reuse, positive candidate reuse, and an exact conversion-count
reduction equal to the reused count, alongside the original lifecycle checks.
It has not run on the GPU yet. Counts describe host dispatch/graph construction,
not replayed GPU launches or throughput. No acceptance gate is relaxed.

`run_generation.py` prepares short/long C17/C23 generation correctness checks
for the reuse executables. It requires the completed 257-prefix C17/C23 lifecycle
proofs, retains all twelve forward/reverse/forward cases, verifies dispatch and
output parity, and checks transitive build/evidence hashes before and after the
child. This runner is prepared only; generation has not yet been executed with
reuse. The active 8192-prefix lifecycle run retains exclusive GPU use first.
