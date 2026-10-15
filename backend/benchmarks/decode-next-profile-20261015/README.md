# Decode profile and gate/up loop scheduling - October 15, 2026

Production remains at 6570895. The preceding goal turn made progress by
publishing the validated Q4_K dot metadata improvement. This follow-up profiles
that committed runtime and tests the next dominant kernel. No production code
is changed by this experiment.

## Profile

Short decode matrix time is 1190.257 ms, including 474.770 ms in fused gate/up
(39.9%), 506.401 ms in remaining Q4_K dots (42.5%), and 209.086 ms in Q6_K dots
(17.6%). Norm/rope/vector/attention/gather account for another 345.219 ms.
These event profiles disable graphs and include submission gaps, so they are
bottleneck diagnostics, not scorecard throughput. Long-run logs also contain
untimed prefix preparation and those vendor-GEMM entries are not decode costs.

## Experiment

Original Fyodor experiment. `probe_unroll.py` compiles the archived committed
source and copies of `nya_gate_up_q4` with outer-loop unroll factors 2, 4 and 8.
Variant 1 leaves compiler policy unchanged. Eight warps per block, each lane's
accumulation sequence, reduction, weight format and activation precision remain
unchanged. This does not introduce the previously rejected independent-sum
accumulator chains. All six shapes match the baseline byte-for-byte including
output guards. Shapes with 1, 3 and 9 quantization blocks exercise loop tails.

Each graph contains 50 repeated kernels; three warmup replays precede five
samples of ten replays. Forward/reverse variant order retains ten samples per
shape. Synthetic hot-weight timing is diagnostic: snapshots show only the
probe process, but there is no continuous process monitor or desktop idle
gate. It cannot update accepted whole-model scores.

Mean microseconds per operation:

| Rows x columns | Default | Unroll 2 | Unroll 4 | Unroll 8 | Unroll 8 throughput gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 x 256 | 1.454 | 1.225 | 1.512 | 1.307 | 11.29% |
| 5632 x 2048 | 45.911 | 46.348 | 45.406 | 44.603 | 2.93% |
| 11008 x 4096 | 198.552 | 213.853 | 257.662 | 194.800 | 1.93% |
| 65 x 768 | 2.735 | 2.388 | 2.899 | 2.286 | 19.63% |
| 257 x 2304 | 6.527 | 5.747 | 6.408 | 5.491 | 18.85% |
| 28672 x 8192 | 1009.514 | 1137.371 | 1440.790 | 1011.187 | -0.17% |

Factor eight is a candidate for full-model testing: it improves the TinyLlama
shape modestly and the 4096-column shape by about 2%, while the 8192-column
case is effectively unchanged (slightly slower). Factors two and four regress
substantially on the larger cases and are rejected. Register counts rise from
39 to 40 for all explicit factors; the driver reports zero local-memory bytes.
These attributes do not establish complete occupancy or explain the scheduling
differences on their own.

No end-to-end gain is accepted yet. The next step is a guarded full-runtime
trial of factor eight, including the production edge oracle, GPU sanitizers,
real-model checks and monitored before/after decode. Keep it only if those
measurements show value. Accepted llama.cpp-relative decode remains **81.3%
short / 101.4% long**; the latter is a narrow measured win on the pinned workload.
Prefill and training remain outside the current optimization scope.

## Evidence and reproduction

- `profile/`: committed executable hash, commands, event profile and snapshots.
- `graphs/results.json`: default and factors 2/4/8 for the initial three shapes.
- `extended/results.json`: odd loop lengths and the larger shape.
- `extended/kernel-resources.json`: driver register/local-memory attributes.
- `baseline-matvec.cu.out`: original source used by all variants.

Run `python probe_unroll.py NEW_OUTPUT_DIRECTORY` for the initial shapes, or
add `extended` for the second set. Output directories must not already exist.
The script uses bundled NVRTC plus the CUDA driver, without PyTorch. Both
probes and both profile workloads exited zero. The initial profiler invocation
failed before execution because its parent directory was absent; creating the
parent resolved that setup error and no timing was produced by the failure.
