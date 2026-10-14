# Decode gate/up workgroup experiment - October 14, 2026

Original Fyodor diagnostic, derived from the previous direct-driver kernel
oracle. Production runtime is unchanged. The prior goal turn made progress:
it committed and pushed the validated fusion. This experiment tests the next
measured bottleneck, which still accounts for 40.1% of short-context matrix
profile time.

## Method

Compile the production sources with five generated copies of the fused kernel,
changing only rows per block and matching the launch size to 1/2/4/8/16 warps.
Per-row arithmetic and warp reduction are unchanged. Random seed 712 produces
finite F32 inputs and packed Q4_K data with fixed finite scale fields. Check
all output bytes including canaries against the committed eight-warp kernel
before timing. All fifteen shape/workgroup combinations matched exactly.
This comparison relies on the prior independent fusion validation; it is not
an independent mathematical oracle.

`probe.py` measures ordinary launches, exposing host submission overhead.
`probe_graphs.py` repeats the experiment using graphs of 50 kernels, three
warmup replays, then five event-timed samples of ten replays each. Workgroup
order runs forward and reverse, retaining all ten samples per cell. CUDA event
times are divided by 500 to give per-kernel time. The graphs use the same
weights repeatedly, unlike an entire model's layer sequence. These diagnostic
runs have process snapshots but no continuous contamination/idle gate; they
cannot update the accepted end-to-end scorecard. Snapshots show only the
probe's own Python process. Neither workload modifies model precision.

## Results and decision

Graph-replayed mean microseconds per fused operation:

| Rows x columns | 1 warp | 2 warps | 4 warps | 8 warps (current) | 16 warps | 2-warp throughput gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5632 x 2048 | 54.748 | 54.700 | 54.787 | 54.731 | 54.718 | 0.06% |
| 11008 x 4096 | 273.374 | 228.817 | 229.710 | 233.045 | 253.625 | 1.85% |
| 28672 x 8192 | 1422.319 | 1184.632 | 1187.795 | 1197.253 | 1288.973 | 1.07% |

TinyLlama's 5,632 x 2,048 shape is effectively unchanged (about 0.06%).
Two warps improve the two larger synthetic shapes by about 1-2%, while one
and sixteen warps regress substantially there. Keep production at eight warps:
there is no useful evidence of improvement on the matched llama.cpp workload,
and synthetic gains alone do not justify a new dispatch policy.

Accepted decode remains **73.2% short / 92.3% long** of the pinned llama.cpp
reference. No prefill or training optimization was performed. The next work
should investigate compressed-weight loading/dequantization or another
measured decode bottleneck rather than spending more time on block size alone.

## Reproduction and evidence

Run the scripts with Windows Python from this worktree; they discover the
bundled CUDA 13.1 NVRTC and driver API directly, without PyTorch. The graph
script requires its `graphs/` output directory not to exist; use a fresh copy
of the evidence directory for reruns. No runtime rebuild is necessary.

- `results.json`: all ordinary-launch samples.
- `graphs/results.json`: all graph samples, compilation options and source hash.
- `provenance.json`: committed revision, original source hashes and GPU/driver.
- `processes-before.txt`, `processes-after.txt` and corresponding graph files:
  raw process snapshots. Initial snapshots include the probe after context creation.

Both probes completed with exit code zero. No production code was edited.
