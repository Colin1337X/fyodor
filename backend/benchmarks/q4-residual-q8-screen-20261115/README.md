# Residual Q8 activation error screen

Original Fyodor CPU experiment informed by the pinned llama.cpp source path:
mmvq.cu mul_mat_vec_q -> vecdotq.cuh vec_dot_q4_K_q8_1 -> integer DP4A dot
products. quantize.cu quantize_q8_1 uses 32-value groups, max/127 scaling,
round-to-nearest-away-from-zero integer codes, and half scale storage.
The inspected upstream revision is 5266f24da75dc449bd56cbed7addb9c8e4a6a73e,
MIT licensed; provenance.json records source/license hashes. No source copied.

This experiment deliberately uses F32 scales and repeatedly quantizes the
F32 residual, reconstructing F32 activations. It is not llama's Q8_1 format.
The existing validated independent C double Q4 oracle compares dots of original
and reconstructed activations using the unchanged bound
2e-6 * sum(abs(double products)) + 1e-6. CPU-only: no CUDA import/context/workload.

Twelve fixtures cover K=256,3840,8192,15360, 17 output rows, four inputs,
and random, multiscale and impulse patterns. One/two/three passes yield 36
observations. Zero and positive/negative exactly representable controls pass.

- One pass: 539 output values exceed the bound; maximum fraction 1987.85.
- Two passes: 275 exceed it; maximum fraction 6.35257.
- Three passes: none exceed it; maximum fraction 0.179639.

One/two-pass variants are rejected under this contract. Three passes warrant
further investigation only: this screen excludes GPU accumulation error,
coefficient extremes, broad adversarial patterns, and actual integer-kernel
throughput. It cannot establish a numerical guarantee, model parity, speedup
or acceptable precision mode. The current model sanitizer continues separately;
no live inputs or production code were changed.
