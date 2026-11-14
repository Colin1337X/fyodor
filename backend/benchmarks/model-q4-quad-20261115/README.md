# Isolated Q4 four-row model candidate

Adds the validated Q4 four-row kernel on top of the Q6 paired-row/Q4 paired-row
candidate. Dispatch requires NYA_CUDA_Q4_QUAD_ROWS=1, successful function load,
integer interleaved execution, batch four, Q4_K type 12, 3840 columns, 15360 rows,
and native 144-byte-block stride. Existing BLAS precedence remains intact.
Only this expansion shape changes; Q6 and other Q4 shapes keep prior dispatch.
The existing provider/resident host sources remain C. Production is unchanged.

Build verifies prior candidate short memcheck and long generation, raw-kernel
numerical/sanitizer proofs, and both direct paired timing orders before adding
the new source. Isolated C17/C23 O3 builds passed. The actual extracted dispatch
predicate passed 3780 cases per standard with strict warnings as errors.

Lifecycle validation retains output tolerances, two resets, four tokens per
reset, graph reuse, single-bank downloads and input hashes. A positive and
well-formed four-row execution marker is mandatory alongside Q6/Q4/QKV markers.
C17 prefix-257 validation passed both resets, with maximum scaled logit error
2.19546455e-05 and 96 four-row capture/direct calls. All 910 hashes
verified. C23 lifecycle, generation, long lifecycle, full-model sanitization
and model timing are pending. No accepted score changes.

C23 prefix-257 lifecycle also passed both resets, with 96 four-row execution
markers counted and maximum scaled error 2.19546455e-05. All 910 hashes verified.
The C17 8192-prefix lifecycle is running. Generation requires all three lifecycle
proofs and exactly three positive four-row trace entries; eight offline marker
cases passed. Its missing-long-lifecycle refusal was exercised before any GPU
launch. The full-model sanitizer runner retains the same gates and requires
four-row enablement/execution; 13 validator tests pass, and its missing-generation
refusal was exercised. These are harness tests, not completed generation or
full-model sanitizer runs.
