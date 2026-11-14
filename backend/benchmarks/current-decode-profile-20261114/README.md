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
The later corrected, upload-fenced norm/interleave diagnostic regressed on
Gemma's 3840-column batch-four shape, so that fusion was not promoted.
Layer-local Q/K/V reuse of the existing packed input is the next isolated
candidate; it does not change normalization or matrix arithmetic.

The long profile now also completed all twelve matching cases at 8205 prompt
tokens and 32 outputs. All 651 input/evidence hashes independently verified.
Joint rounds recorded matrix totals of 471.858, 464.495 and 479.628 ms;
normalization 45.881, 47.958 and 49.191 ms; attention 235.588, 238.183 and
239.279 ms. The target-only attention entries sum to 188.279, 189.538 and
192.329 ms; the aggregate category also includes borrowed helper work.

The same Q4 batch-four expansion remains the largest matrix group, at
136.669, 135.365 and 137.816 ms, followed by serial fused Q4 gate/up at
48.661, 47.694 and 49.202 ms and batch-four Q6 down projection at
43.963, 42.723 and 43.922 ms. This supports retaining matrix work as the
short-context priority while attention is a larger long-context cost.
These are profiler costs, not accepted throughput. Accepted llama.cpp
scores remain 85.61025294% short and 106.8707255% at 8K; neither this profile
nor the provisional generation timings establish the requested 125% target.
