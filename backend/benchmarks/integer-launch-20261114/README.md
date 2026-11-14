# Q4 integer launch specialization

Original Fyodor experiment under `optcpp/matvec/integer_launch.cu`. It preserves
the integer kernel arithmetic and reduction order while specializing four/eight
rows per block and minimum block launch bounds. No runtime dispatch, production
library, public ABI or build requirement changes. Removing this optional source
does not affect the production build.

The joint-base profile identifies batch-four Q4 matrix work as the next decode
target. Earlier launch-bound tests covered the non-interleaved weight-reuse
kernel; this experiment applies bounds to the current integer interleaved path.

All four variants passed 207 raw cases against the native narrow-batch oracle,
with byte-exact output and intact guards, including partial row tiles and finite
F16 coefficient extremes. Small cases also use an independent float64 bound.
The same sweep passed actual Compute Sanitizer memcheck with zero errors.

Timings below are mean baseline / candidate kernel speed percentages; 100 means
equal. Each cell retains ten samples, forward/reverse variant order, graph-event
timing, input interleaving, and output equality checks across eight shapes.
This uses synthetic repeated weights and lacks continuous contamination
acceptance. It is diagnostic only, not evidence of end-to-end throughput.

| Columns | Rows | 4 rows / bound 4 | 4 / 8 | 8 / 3 | 8 / 4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3840 | 512 | 141.58% | 139.36% | 142.86% | 142.57% |
| 3840 | 2048 | 120.26% | 118.27% | 115.46% | 114.04% |
| 3840 | 4096 | 104.57% | 103.11% | 102.59% | 101.19% |
| 3840 | 8192 | 102.61% | 100.85% | 101.36% | 99.99% |
| 3840 | 15360 | 102.26% | 100.89% | 102.42% | 100.68% |
| 4096 | 3840 | 105.40% | 104.05% | 103.58% | 102.22% |
| 8192 | 3840 | 95.26% | 93.91% | 102.59% | 100.99% |
| 15360 | 3840 | 99.69% | 100.61% | 109.24% | 108.17% |

The eight-row / bound-three candidate is the next whole-model validation
candidate: it improves all tested diagnostic shapes. Four-row variants regress
on 8192-column projections, so a blanket four-row change is rejected.

The CUDA driver reports 39 registers/thread for the baseline, 64 for bound-three
eight-row and bound-four four-row variants, and 62 for the other two variants.
All report zero static shared and local bytes. These are compiled resource
attributes, not measured achieved occupancy or proof of why timing changed.

Pending: full-model numerical/lifecycle validation, whole-model sanitizer,
short/long decode with continuous contamination and equivalent llama.cpp
comparison. Accepted Gemma scores remain 85.61025294% short and 106.8707255% at
8K; the 125% target has not been achieved.

## Isolated full-model admission

`build_model.py` appends the candidate to a copy of the generated CUDA source
and changes only the integer function lookup to `nya_integer_launch_8_3` in a
copied CUDA host translation unit. Both C17 and C23 compile with strict warnings.
The production archive and runtime dispatch remain unchanged. The generated
header, objects, source inputs and build commands are hashed in model-build.json.

The real Gemma 257-prefix ring-KV lifecycle passed both resets with maximum
scaled logit error 0.0000219546455 against serial execution, within the original
0.0005 bound. The test retains guards, rejected argument checks, rewind, stale
suffix rejection and replacement-token validation. External model bytes are
hashed before execution. The trace confirms integer-path selection and calls;
the isolated build manifest identifies the selected launch-bound symbol.

Joint-base generation links the previously validated core with this isolated
CUDA object. Both C17 and C23 executables passed all twelve short-context cases
(ordinary, serial2, batch2, joint3 over three rounds). Each joint round recorded
2556 integer kernel calls, and generated output matched ordinary decoding.
Input and output hashes were verified. This is output correctness evidence,
without continuous contamination acceptance; wall times are not accepted scores.
Long-context lifecycle, full-model sanitizer with this candidate and monitored
short/long throughput remain pending.

`run_model_memcheck.py` requires the completed short and 8K lifecycle proofs,
matching generation parity and build hashes. It verifies external model identity,
requires a successful child exit and exactly one zero-error sanitizer summary,
and requires all twelve parity cases and three positive integer call counts.
Seven offline validator tests cover failures and both context inventories.
`run_monitored.py` additionally requires short and long generation correctness
and the full-model short sanitizer, then uses the existing idle gate and continuous
PDH collector. It records output parity and dispatch counts; it never declares
performance acceptance on its own. These runners are prepared, not evidence that
their pending GPU runs have passed.

The completed 8192-prefix ring-KV lifecycle now passes both resets, with maximum
scaled logit error 0.0000163989979 in each. Integer execution reports 568 calls;
rewind, replacement-token, stale-suffix, output-guard and original 0.0005-bound
assertions all pass. The child exited zero and all recorded input hashes match.
This is long-context numerical evidence, not long-context generation throughput.
Full-model sanitizer and long joint-base generation remain pending.

The monitored runner now snapshots the executable, transitive proof inputs,
runner/configuration and PDH monitoring Python sources before execution. It
preserves those original digests and rejects changed or missing inputs after the
child exits. Three offline integrity tests cover mutation, deletion and keeping
new output files separate from input identity; all ten admission/integrity tests
pass. This hardens the pending run and does not constitute a monitored GPU pass.
