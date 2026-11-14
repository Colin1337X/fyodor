# Adjacent gate/up input conversion reuse (unvalidated experiment)

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

Both host builds pass (`build.json`). This is **not GPU correctness or performance
validation**. No executable has been run with this change. It is not integrated
or enabled by default. It saves one conversion launch per eligible FFN pair;
it does not remove either projection or change their arithmetic. The whole-model
benefit is unknown and must be measured after the current candidate finishes.

Next gates: link isolated generation/lifecycle executables; validate short/long
logits, output parity, reset/rewind and failure recovery; run Compute Sanitizer;
then compare end-to-end decode with and without reuse under clean monitoring.
Do not replace accepted scores from a compile check or inferred launch savings.
