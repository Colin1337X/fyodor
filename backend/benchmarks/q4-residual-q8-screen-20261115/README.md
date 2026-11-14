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

## Adversarial screen: three-pass candidate rejected

screen_adversarial.py isolates one weighted lane while larger zero-weight
lanes control each residual scale. It covers four half coefficients, three
activation magnitudes and four selected lanes, across one through five passes:
240 observations, four token outputs each. All artifact hashes verified.

Outputs outside the unchanged bound: one pass 140, two 92, three 48, four 16,
five 0. The sampled five-pass success is not a general guarantee; these fixtures
have only five distinct nonzero magnitudes, which favors eventual reconstruction.

A concrete three-pass failure uses coefficient 0.029998779296875, activation
magnitude factor 1048576, and selected lane 3. The original weighted activation
is 0.010485759936273098; reconstructed activation is zero. The expected dot is
0.00031455999808827073 against a bound of 0.0000010006291199961765: error is
314.36 times the allowed bound. Four passes also fail with coefficient 1 and
selected lane 4 (10.49 times the bound).

The initial random/multiscale/impulse screen was insufficient. Fixed three-pass
Q8 reconstruction is rejected as a general substitute for F32 activations.
Do not proceed to a three-pass CUDA kernel based on the earlier screen alone.
A future approach would need a proven error-controlled fallback or preservation
of residual information, followed by accumulation and real-model validation.
No numerical tolerance was changed and no production quantization was added.
