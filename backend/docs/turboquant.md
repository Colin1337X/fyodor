# Experimental TurboQuant KV precision

`NYA_COMPUTE=cpu NYA_KV_TYPE=turboquant3` or `turboquant4` selects
native C compressed KV during text inference. `f32` (also unset) remains the
default. Benchmark JSON reports the actual session precision and
`cpu-turboquant-reference`, never an F32 or GPU result for this mode.
GPU-resident, multimodal and speculative/MTP use currently reject the mode
explicitly. This is not yet a throughput optimization or a quality-neutral mode.

The original implementation follows [TurboQuant, Zandieh et al. (2025),
algorithms 1/2 and QJL](https://arxiv.org/abs/2504.19874).
No upstream code was copied. The paper is CC BY 4.0.
A shared Gaussian orthogonal rotation (twice-reorthogonalized QR), a sphere
coordinate Lloyd-Max codebook, packed scalar indices, and Gaussian QJL residual
correction implement keys at (bits-1)+1 bits and values at bits bits.
Codebooks use midpoint quadrature of the exact coordinate density, without
model calibration. This reference uses dense transforms, not a Hadamard variant.

A vector additionally stores its F32 norm; keys store an F32 residual norm.
Byte padding and those norms count toward storage. At width 128, 3-bit keys
are 56 bytes and values are 52 bytes: 108 bytes per KV head per token versus
1024 F32 bytes. Shared dense transforms, scratch and object overhead count in
`nya_turboquant_bytes`; for short contexts that overhead can outweigh savings.
The model integration keeps only one-token F32 staging, not an F32 shadow cache.
CPU session `kv_bytes` includes compressed storage, transform and codec scratch;
this is host memory, not VRAM. Ordinary run-state scratch is separate.

Public C API: `backend/include/turboquant.h`. Widths 32..512; codec bit widths
2..4, runtime exposes 3 and 4. Each request owns the cache. Store is atomic
on invalid input and supports append/overwrite, enabling prefix replay.
Packed-key query products and weighted rotated values are consumed by stable
online softmax without materializing a full F32 cache or attention matrix.
Seed fixes transforms for a cache; serialization of these experimental caches
is not supported. Cross-platform byte-identical transforms are not promised.

Validation separates two contracts: math equivalence to attention over the
decoded cache, and lossy error versus original F32 KV/logits. Synthetic MSE,
QJL bias across seeds, finite outputs and exact request replay do not establish
real-model perplexity or long-context retrieval quality. Existing F32 tolerances
remain unchanged. Do not compare this mode to an F32 llama.cpp run as if the
precision were matched. GPU kernels, speed acceptance, long-context model-quality
evaluation, and reuse of transforms across model layers remain future work.

Measured 12-token real-model logit RMSE versus F32 is 1.65/0.69 (3/4-bit)
for TinyLlama and 3.90/1.73 for Gemma4 12B. These are substantial changes;
quality-neutral deployment is not established. Reproducible diagnostic evidence:
`backend/benchmarks/turboquant-flash-20261029`.
