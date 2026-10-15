# Decode compiler follow-up - October 15, 2026

Production remains at ce0930c. Fresh short profiling places gate/up at
404.972 of 1111.907 matrix milliseconds (36.4%). Event profiling disables
graphs and includes host submission gaps; these are not acceptance timings.

The four-block bound changes register allocation, so this experiment revisits
outer-loop unrolling under that bound. All five variants preserve arithmetic
order, eight warps per block and exact outputs including guards. Repeated
hot-weight CUDA graphs use forward/reverse order, ten samples per cell.
Snapshots are recorded, but no continuous competitor monitor or idle gate;
these diagnostic results cannot update whole-model scores.

| Rows x columns | unroll 1 | unroll 2 | unroll 4 | unroll 16 |
| --- | ---: | ---: | ---: | ---: |
| 9 x 256 | -0.92% | +2.51% | -0.82% | -0.03% |
| 5632 x 2048 | -5.82% | -1.62% | -0.72% | -2.09% |
| 11008 x 4096 | -2.84% | -0.03% | -1.42% | +0.39% |
| 65 x 768 | -5.62% | +0.22% | -2.74% | +0.00% |
| 257 x 2304 | -18.04% | -8.92% | -5.36% | -7.84% |
| 28672 x 8192 | -1.47% | +0.41% | -0.85% | +0.30% |

Values are throughput changes relative to current unroll eight. Keep eight:
it wins at the target TinyLlama shape and odd 2304-column case. Small gains
at wider shapes do not justify a runtime shape policy from these diagnostics.

A second probe adds restrict qualifiers only to weights (key 2), input (4),
output (8), or all (16), with key 1 the unmodified production kernel.
All results remain byte-exact. No variant produces a useful consistent gain:

| Rows x columns | weights | input | output | all |
| --- | ---: | ---: | ---: | ---: |
| 9 x 256 | +0.02% | +0.01% | +0.01% | +0.01% |
| 5632 x 2048 | -0.22% | -0.49% | -0.47% | -0.48% |
| 11008 x 4096 | -0.55% | -0.57% | -0.59% | -0.58% |

Reject both changes for production. No runtime source was edited and no
benchmark score changed: **86.8% short / 107.5% long** of pinned llama.cpp.
The next investigation should target instruction or memory traffic rather
than further compiler-hint tuning of this same kernel. Scripts are original
Fyodor diagnostics; source hashes, compiler options, resource counts, device
limits and every sample are archived. Profile and all three probe runs
completed successfully.
