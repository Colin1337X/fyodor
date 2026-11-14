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
