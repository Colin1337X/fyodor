# Current validated decode kernel profile

Profiles the isolated batched-graph candidate with native Q4 metadata and
contiguous verification readback, after full-model short memcheck, all 336
boundary sanitizer cases and all 96 extended recovery cases. Inputs and
recorded evidence hashes are verified before and after the run.

The profiler disables graph capture, so this identifies kernel costs in the
current code rather than graph-enabled wall throughput. Target profiling starts
after the exact short/long prompt token count. Twelve ordered generation cases
must retain output parity; integer selection and metadata markers are required.
There is no continuous activity acceptance, and these results cannot update
accepted performance scores. No production dispatch changes.

The short profile completed all twelve matching generation cases and 648
input/evidence hashes independently verified. Joint rounds recorded matrix
totals of 465.829, 458.028 and 458.080 ms; normalization 46.906, 47.664 and
45.437 ms; attention 16.876, 16.698 and 17.536 ms. These aggregate categories
include borrowed helper work while the target plan is profiling.

The largest matrix group remains Q4 batch-four expansion (15360 rows by
3840 columns), at 136.844, 134.615 and 130.872 ms. Serial fused Q4 gate/up
is next at 49.777, 47.896 and 46.596 ms, followed by batch-four Q6 down
projection at 42.852, 42.329 and 45.083 ms. Dispatch counts include conversion
operations and must not be interpreted as generated-token counts.

The follow-up RMSNorm/interleave prototype targets a separate layout
conversion after normalization, retaining row-major output for fallbacks.
This is a new experiment, not an inferred throughput win. Long profiling
has not run in this folder and accepted llama.cpp scores remain unchanged.
