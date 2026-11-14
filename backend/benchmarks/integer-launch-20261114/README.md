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
