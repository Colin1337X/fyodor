# Q4 four-row activation reuse experiment

Original Fyodor CUDA experiment under optional /optcpp; no production dispatch.
The prior two-row kernel shares a float4 input load across two output rows.
This variant shares across four, retains each row's accumulation/reduction
order and native 144-byte Q4_K layout, and uses 32 rows per 256-thread block.
Launch bounds permit two resident blocks rather than three. Odd tails reuse a
valid weight address and discard invalid-row outputs.

NVRTC 13.1 compute_120 compilation and ptxas SM120 assembly passed: 128 registers,
zero stack, spills and barriers. The two-row candidate uses 80 registers.
The unchanged admission workload passed all 224 independent C double-oracle
cases and eight unsupported-shape rejections, with output guards, immutable
inputs, and byte equivalence to the metadata kernel. Maximum used fraction of
the established numerical error bound was 0.04197500690558822.
Seven sanitizer validator tests passed. Full memcheck, racecheck and synccheck
passed, including all 224 numerical cases and eight rejection cases each. No end-to-end claim or
accepted score change is justified by these kernel checks.

Harnesses are adapted from q4-pair-rows-20261114. The inherited timing label
`pair` refers to this four-row candidate; metadata is the one-row baseline.
The timing harness uses synchronized uploads and forward/reverse ordering.
Synthetic repeated weight banks do not model full-model cache behavior and
have no continuous contamination acceptance. Production remains unchanged.

## Direct two-row comparison

`probe_pair.py` loads both candidates through `common_timing.py`, checks byte
equivalence and independent numerical bounds after each timed order, and retains
40 samples per shape/kernel with sample standard deviations. Its order is
two/four, four/two, four/two, two/four. All sanitizer manifests verified after
completion. These are diagnostic kernel latency changes, not model speedups.

| Columns | Rows | Four-row latency change vs two-row |
| --- | --- | --- |
| 3840 | 512 | +21.869% |
| 3840 | 2048 | -6.623% |
| 3840 | 4096 | -4.650% |
| 3840 | 8192 | +0.595% |
| 3840 | 15360 | -1.474% |
| 4096 | 3840 | +6.300% |
| 8192 | 3840 | +2.086% |
| 15360 | 3840 | +2.567% |

The important 3840x15360 expansion shape improves only about 1.5%; larger
column shapes regress. This rejects a blanket replacement. A separate reversed
initial order and full-model validation would be required before adopting even
the narrow expansion dispatch. No production integration or score update yet.
