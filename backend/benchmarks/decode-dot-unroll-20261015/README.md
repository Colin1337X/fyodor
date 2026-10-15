# Standalone decode dot unrolling - October 15, 2026

Original Fyodor experiment. Production remains at f4f4ec4, with no changes to
its standalone Q4_K/Q6_K dot kernels. The preceding goal turn made progress
by validating and publishing fused gate/up unrolling. This experiment asks
whether that scheduling change transfers to the remaining measured costs.
The preceding profile records Q4_K dots at 42.5% and Q6_K dots at 17.6% of
short-context matrix time; the intervening runtime change affected only fusion.

## Method

The direct NVRTC/driver harness compiles the archived production source and
copies of its row-dot helper with outer-loop unroll factors 2, 4 and 8. Compile
time type selection chooses Q4_K or Q6_K. Eight warps per block, accumulation
sequence, reduction, quantization and F32 inputs remain unchanged. Random seed
712 produces packed quantized data with finite scale fields and F32 inputs.
Each candidate output including guards must match the default byte-for-byte.
All 45 candidate/shape combinations passed, including one-, three- and nine-
block loop tails and partial row blocks. This checks equivalence with the
production baseline, not an independent mathematical oracle.

Graphs contain 50 repeated kernels. Three warmup replays precede five samples
of ten replays, with forward/reverse candidate order. All ten samples per cell
are retained. Process snapshots are included, but there is no continuous
process monitor or desktop idle gate; these are hot-weight diagnostics, not
whole-model acceptance runs. No production performance score is updated.

## Results

Throughput change relative to the default on the same shape (positive is faster):

| Format | Rows x columns | Unroll 2 | Unroll 4 | Unroll 8 |
| --- | --- | ---: | ---: | ---: |
| Q4_K | 9 x 256 | 3.67% | 1.30% | 1.84% |
| Q4_K | 65 x 768 | 2.98% | 5.25% | 4.37% |
| Q4_K | 256 x 2048 | 4.35% | 6.68% | 2.54% |
| Q4_K | 2048 x 2048 | 3.90% | 3.96% | 4.98% |
| Q4_K | 2048 x 5632 | 4.03% | 3.74% | 4.12% |
| Q4_K | 32000 x 2048 | -1.70% | -4.33% | -3.70% |
| Q4_K | 257 x 2304 | 0.10% | 3.57% | -1.95% |
| Q4_K | 11008 x 4096 | 3.72% | 4.42% | 4.96% |
| Q4_K | 8192 x 8192 | -1.56% | -6.06% | -5.98% |
| Q6_K | 9 x 256 | 3.72% | 3.59% | 3.67% |
| Q6_K | 65 x 768 | -6.74% | 3.56% | 0.76% |
| Q6_K | 256 x 2048 | -15.92% | -17.39% | -15.39% |
| Q6_K | 2048 x 2048 | -1.63% | -1.13% | -1.07% |
| Q6_K | 2048 x 5632 | -2.81% | -2.08% | -1.48% |
| Q6_K | 32000 x 2048 | -1.60% | -1.46% | -1.49% |

Reject uniform standalone unrolling. Q6_K regresses on the key 256x2048,
2048x5632 and 32000x2048 shapes. Q4_K shows modest gains on several internal
projections, but regresses on the large output and wider/larger projections.
Do not introduce a TinyLlama-specific cutoff to hide those regressions. This
result differs from the accepted fused gate/up path and does not invalidate
its measured full-model benefit.

Driver attributes report Q4_K registers rising from 37 to 40; Q6_K remains at
40. All report zero local-memory bytes. These attributes are useful context,
not proof of the reason for each timing difference. The next optimization
should target another measured cost or reduce work rather than apply more
unrolling across formats indiscriminately.

Accepted decode remains **82.8% short / 103.5% long** against pinned llama.cpp.
Short decode is still below parity, and prefill/training remain outside the
current optimization scope. No runtime build or full regression suite is
needed for this evidence-only change; all three diagnostic processes exited
zero and no production source was edited.

## Reproduction

Run `python probe.py 12 NEW_DIRECTORY` or `python probe.py 14 NEW_DIRECTORY`
for the initial six shapes. Append `extended` for the three larger/odd-length
shapes (the archived extended run uses Q4_K). Output directories must not exist.
The script uses bundled CUDA 13.1 NVRTC and the CUDA driver without PyTorch.
`results.json` files contain every sample, compilation options and source hash;
`kernel-resources.json` contains driver resource attributes. The archived
baseline and `provenance.json` identify the committed source. Hardware is RTX
5060 Ti 16 GB, driver 617.42 on Windows, compute_120 target.
